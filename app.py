#app.py
import os
import re
import uuid
import json
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory
from werkzeug.utils import secure_filename

# Prefer direct import of your combined extractor (located at project root)
try:
    from ocr_pad_extractor import extract_from_report
except Exception:
    extract_from_report = None
    print("Warning: could not import extract_from_report from ocr_pad_extractor.py. Ensure app run from project root.")

# Replace/implement these utilities in your project (they were in your original app.py)
# The server will call them exactly as below.
try:
    from extras.ocr_utils import extract_text_from_file
    from extras.parser_utils import parse_report_text
    from model_utils import load_model, predict_with_model, comparative_predict as rule_based_predict
except Exception:
    # If you don't have these modules yet, define simple fallbacks so app still starts.
    def extract_text_from_file(path):
        # returns list of page texts
        return ["(dummy OCR)"]

    def parse_report_text(text):
        # dummy parser returns dict with minimal keys expected by normalize logic
        return {"Patient Name": "Unknown", "ABI": "Invalid", "Age": "30", "Sex": "M"}

    def load_model(path):
        return None

    def predict_with_model(model, parsed):
        return {"pad_detected": False, "severity": "None"}

    def rule_based_predict(parsed):
        return {"pad_detected": False, "severity": "None"}

# Config
UPLOAD_FOLDER = "uploads"
ALLOWED_EXT = {"png", "jpg", "jpeg", "pdf"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("models", exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("PAD_FLASK_SECRET", "replace-with-secure-key")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Try to load ML model if present
MODEL_PATH = os.path.join("models", "model.pkl")
model = None
try:
    model = load_model(MODEL_PATH)
except Exception:
    model = None

# Serve the raw html files exactly as-is (place your original files in templates/)
@app.route("/", methods=["GET"])
def index():
    # Start with the login page first
    return render_template("login.html")

@app.route("/index.html")
def serve_index_html():
    return render_template("index.html")

@app.route("/login.html")
def serve_login_html():
    return render_template("login.html")

@app.route("/analysis.html")
def serve_analysis_html():
    return render_template("analysis.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not email or not password:
            flash("Please provide email and password.")
            return redirect(url_for("login"))
        session["user_email"] = email
        flash(f"Logged in as {email}")
        # After successful login, redirect to index.html
        return redirect(url_for("serve_index_html"))
    return render_template("login.html")

# Analysis route
@app.route("/analysis", methods=["GET", "POST"])
def analysis():
    result = None
    if request.method == "POST":
        # file upload handling - expects name="report"
        if "report" not in request.files:
            flash("No file part (please ensure analysis.html file input has name='report').")
            return redirect(request.url)
        file = request.files["report"]
        if file.filename == "":
            flash("No file selected")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        if "." not in filename:
            flash("Invalid filename")
            return redirect(request.url)
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext not in ALLOWED_EXT:
            flash("File type not allowed. Upload PNG, JPG, JPEG, or PDF.")
            return redirect(request.url)

        unique_name = f"{uuid.uuid4().hex}_{filename}"
        saved_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        try:
            file.save(saved_path)
        except Exception as e:
            flash(f"Failed to save file: {e}")
            return redirect(request.url)

        # ---------------------------
        # Use the combined extractor directly (OCR + parsing + conclusion)
        # ---------------------------
        if extract_from_report is None:
            flash("Internal error: extractor not available (ocr_pad_extractor.py import failed).")
            return redirect(request.url)

        try:
            parsed, full_text, conclusion = extract_from_report(saved_path)
        except Exception as e:
            flash(f"Failed to extract data from report: {e}")
            return redirect(request.url)

        # Save OCR text for debugging
        try:
            with open(saved_path + ".ocr.txt", "w", encoding="utf-8") as _f:
                _f.write(full_text or "")
        except Exception:
            pass

        if parsed is None or not isinstance(parsed, dict):
            result = {"valid": False, "message": "INVALID REPORT ."}
            return render_template("analysis.html", result=result)

            # Strict ABI validation: if ABI missing or invalid → INVALID REPORT
        abi_raw = parsed.get("ABI")

        # Conditions for invalid ABI
        abi_missing = abi_raw is None or str(abi_raw).strip() == "" or str(abi_raw).lower() in ["invalid", "none", "nan"]

        if abi_missing:
            result = {
                "valid": False,
                "message": "INVALID REPORT!!!!!!."
            }
            return render_template("analysis.html", result=result)

        # --- NEW: parse numeric ABI & prepare ABI->HAS-PAD override ---
        abi_numeric = None
        try:
            abi_str = str(abi_raw)
            tokens = re.findall(r"\d+\.\d+|\d+", abi_str)
            if tokens:
                abi_numeric = float(tokens[0])
        except Exception:
            abi_numeric = None

        # Configurable threshold (env var). Default: 0.9 (common clinical cutoff)
        try:
            ABI_HAS_PAD_THRESHOLD = float(os.environ.get("PAD_ABI_HAS_PAD_THRESHOLD", "0.9"))
        except Exception:
            ABI_HAS_PAD_THRESHOLD = 0.9

        # Optionally enable/disable the ABI override via env var (default enabled)
        ABI_OVERRIDE_ENABLED = os.environ.get("PAD_ABI_OVERRIDE_ENABLED", "1") != "0"

        # Flag when ABI suggests HAS PAD (clinical rule: ABI <= threshold suggests PAD)
        abi_suggests_has_pad = False
        abi_non_compressible = False
        if abi_numeric is not None and ABI_OVERRIDE_ENABLED:
            # sanity check: realistic ABI values (defend against OCR garbage like "108")
            if 0.0 < abi_numeric < 3.0:
                if abi_numeric <= ABI_HAS_PAD_THRESHOLD:
                    abi_suggests_has_pad = True
                elif abi_numeric >= 1.30:
                    abi_non_compressible = True
            else:
                # out-of-range — treat as unreliable; do not override
                abi_suggests_has_pad = False
        # ----------------------------------------------------------------

        # Run predictor: try ML model first, fallback to rule-based comparator
        try:
            if model is not None:
                pred = predict_with_model(model, parsed)
            else:
                pred = rule_based_predict(parsed)
        except Exception as e:
            print("Model prediction failed:", e)
            try:
                pred = rule_based_predict(parsed)
            except Exception as e2:
                print("Rule-based prediction failed too:", e2)
                pred = {"pad_detected": False, "severity": "Unknown"}

        # Normalize pred to expected dict shape
        if pred is None:
            pred = {}
        if not isinstance(pred, dict):
            try:
                pred = {"pad_detected": bool(pred), "severity": str(pred)}
            except Exception:
                pred = {"pad_detected": False, "severity": "Unknown"}

        pad_from_model = pred.get("pad_detected", None)
        severity_from_model = pred.get("severity", None)
        proba_from_model = pred.get("proba", None)

        # compute model_confidence if available (max class probability)
        model_confidence = None
        if isinstance(proba_from_model, (list, tuple)):
            try:
                model_confidence = max([float(x) for x in proba_from_model if x is not None])
            except Exception:
                model_confidence = None

        # Coerce pad_from_model to boolean if possible
        if isinstance(pad_from_model, str):
            pad_from_model = True if re.search(r'\b(1|true|yes|y)\b', pad_from_model, flags=re.IGNORECASE) else False
        elif isinstance(pad_from_model, (int, float)):
            pad_from_model = bool(pad_from_model)
        elif pad_from_model is None:
            pad_from_model = None

        # Extractor conclusion
        pad_from_extractor = bool(conclusion.get("PAD_present", False)) if isinstance(conclusion, dict) else False
        extractor_severity_label = conclusion.get("severity_label", None) if isinstance(conclusion, dict) else None

        # Decision policy
        final_pad = False
        final_severity = None
        decision_note = []

        if model is None:
            final_pad = pad_from_extractor
            final_severity = conclusion.get("severity_text") if isinstance(conclusion, dict) else "Unknown"
            decision_note.append("model_missing -> used_extractor")
        else:
            if model_confidence is not None and model_confidence < 0.60 and pad_from_extractor:
                final_pad = pad_from_extractor
                final_severity = conclusion.get("severity_text")
                decision_note.append(f"model_low_confidence ({model_confidence:.2f}) -> used_extractor")
            elif pad_from_model is None:
                final_pad = pad_from_extractor
                final_severity = conclusion.get("severity_text")
                decision_note.append("model_no_pad_flag -> used_extractor")

            elif pad_from_model and not pad_from_extractor:
                # If the model alone says PAD, be conservative for *mild* predictions:
                # - If model reports "Mild" (class 1) but extractor says no, prefer extractor (avoid false positives).
                # - If model reports Moderate/Severe, accept model.
                # - If model_confidence is high (>=0.75) we may accept even a mild model prediction.
                try:
                    sev_text = (severity_from_model or "").lower()
                except Exception:
                    sev_text = ""

                model_conf_high = (model_confidence is not None and model_confidence >= 0.75)

                if sev_text.startswith("mild") and not model_conf_high:
                    # prefer extractor for mild predictions when extractor says no
                    final_pad = pad_from_extractor
                    final_severity = conclusion.get("severity_text")
                    decision_note.append("model_yes_extractor_no_but_model_mild_lowconf -> used_extractor")
                else:
                    final_pad = pad_from_model
                    final_severity = severity_from_model or conclusion.get("severity_text")
                    decision_note.append("model_yes_extractor_no -> used_model")

            elif not pad_from_model and pad_from_extractor:
                try:
                    sev_label_int = int(extractor_severity_label) if extractor_severity_label is not None else None
                except Exception:
                    sev_label_int = None

                if sev_label_int is not None and sev_label_int >= 1:
                    final_pad = True
                    final_severity = conclusion.get("severity_text")
                    decision_note.append("model_no_extractor_yes_sev>=1 -> used_extractor")
                else:
                    final_pad = pad_from_model
                    final_severity = severity_from_model or conclusion.get("severity_text")
                    decision_note.append("model_no_extractor_yes_but_mild -> used_model")
            else:
                final_pad = bool(pad_from_model)
                final_severity = severity_from_model or conclusion.get("severity_text")
                decision_note.append("model_preferred")

        # --- APPLY ABI-BASED HAS-PAD OVERRIDE (if triggered) ---
        if abi_suggests_has_pad:
            final_pad = True
            final_severity = final_severity or "ABI-based override: PAD"
            decision_note.append(f"abi_override_has_pad (ABI={abi_numeric:.2f} <= {ABI_HAS_PAD_THRESHOLD})")
        elif abi_non_compressible:
            decision_note.append(f"abi_non_compressible (ABI={abi_numeric:.2f} >= 1.30) -> ABI unreliable; recommend TBI/toe pressures")
        # ----------------------------------------------------------------

        # Build UI result
        result = {
            "valid": True,
            "patient_name": " ".join((parsed.get("Patient Name") or parsed.get("Name") or "Unknown").split()[:3]),
            "age_sex": f"{(parsed.get('Age') or 'Unknown')} Years / {(parsed.get('Sex') or 'Unknown')}",
            "pad_detected": "Yes" if final_pad else "No",
            "severity": final_severity or "Unknown",
            "ABI": parsed.get("ABI") or "Unknown",
            "debug": {
                "model_present": model is not None,
                "pad_from_model": pad_from_model,
                "model_confidence": model_confidence,
                "pad_from_extractor": pad_from_extractor,
                "extractor_severity_label": extractor_severity_label,
                "conclusion": conclusion,
                "pred_raw": pred,
                "parsed_snapshot": parsed,
                "decision_notes": decision_note,
                "abi_numeric": abi_numeric,
                "abi_has_pad_threshold": ABI_HAS_PAD_THRESHOLD,
                "abi_override_enabled": ABI_OVERRIDE_ENABLED,
                "abi_suggests_has_pad": abi_suggests_has_pad,
                "abi_non_compressible": abi_non_compressible
            }
        }

        # save debug JSON so you can inspect what happened for each upload
        try:
            debug_path = saved_path + f".{uuid.uuid4().hex}.debug.json"
            with open(debug_path, "w", encoding="utf-8") as _dbg:
                json.dump(result["debug"], _dbg, indent=2, ensure_ascii=False)
            print("Saved upload debug JSON to", debug_path)
        except Exception as _e:
            print("Failed to save debug json:", _e)

        return render_template("analysis.html", result=result)

    # GET
    return render_template("analysis.html", result=result)

# Simple logout
@app.route("/logout")
def logout():
    session.pop("user_email", None)
    flash("Logged out.")
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
