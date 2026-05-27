# tests/test_single_pdf.py  (safe fallback version)
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from extras.ocr_utils import extract_text_from_file
from extras.parser_utils import parse_report_text
from model_utils import load_model, predict_with_model, comparative_predict
import re, json

def quick_field_extract(ocr_text):
    """
    Lightweight regex extractor to salvage numeric fields
    Returns a dict with possible keys: ABI, PSV__<key>, waveform_monophasic_count, waveform_biphasic_count, waveform_triphasic_count
    """
    out = {}
    txt = ocr_text or ""
    # ABI: capture first float-looking token after 'ABI' or 'Ankle'
    m = re.search(r'\bABI[:\s\-]*([0-9]\.\d{1,3})\b', txt, flags=re.IGNORECASE)
    if not m:
        m = re.search(r'Ankle[ \-]Brachial(?: Index)?[:\s\-]*([0-9]\.\d{1,3})\b', txt, flags=re.IGNORECASE)
    if m:
        try:
            out["ABI"] = float(m.group(1))
        except:
            pass

    # PSV numbers: capture lines with PSV or numbers following artery names (simple)
    # Find tokens like "PSV: 120" or "PSV 120" and also any "PSV__" style fallback
    ps = re.findall(r'\bPSV[:\s]*([0-9]{1,4}(?:\.[0-9]+)?)', txt, flags=re.IGNORECASE)
    if ps:
        # if multiple, add PSV__generic or PSV__auto_N
        out["PSV__PSV_generic"] = float(ps[0])
        # add others if present
        for i, val in enumerate(ps[1:], start=1):
            key = f"PSV__PSV_auto_{i}"
            try:
                out[key] = float(val)
            except:
                pass

    # Try artery-specific simple matches: e.g., "Distal SFA 66" or "Distal SFA PSV 66"
    artery_list = ["Common Femoral", "Profundus", "Proximal Superficial", "Mid SFA", "Distal SFA", "Popliteal", "Peroneal", "Posterior Tibial", "Anterior Tibial"]
    for a in artery_list:
        pat = re.compile(re.escape(a) + r'[^\d\n]{0,40}([0-9]{1,4}(?:\.[0-9]+)?)', flags=re.IGNORECASE)
        m = pat.search(txt)
        if m:
            key = "PSV__" + a.replace(" ", "_").replace("/", "_")
            try:
                out[key] = float(m.group(1))
            except:
                pass

    # waveform counts
    out["waveform_monophasic_count"] = len(re.findall(r'\bMonophasic\b', txt, flags=re.IGNORECASE))
    out["waveform_biphasic_count"] = len(re.findall(r'\bBiphasic\b', txt, flags=re.IGNORECASE))
    out["waveform_triphasic_count"] = len(re.findall(r'\bTriphasic\b', txt, flags=re.IGNORECASE))

    return out

def run_on_file(path):
    print("File:", path)
    # 1) get OCR text pages
    texts = extract_text_from_file(path)
    full_text = "\n\n".join(texts).strip()
    preview_fn = os.path.splitext(os.path.basename(path))[0] + ".ocr_preview.txt"
    with open(preview_fn, "w", encoding="utf-8") as f:
        f.write(full_text[:2000])
    print("OCR preview saved to", preview_fn)

    # 2) parse
    parsed = parse_report_text(full_text)
    if parsed is None:
        print("parse_report_text returned None — attempting lightweight fallback extraction.")
        parsed = quick_field_extract(full_text)
        parsed["need_review"] = True
        parsed["_fallback_extractor"] = True
    else:
        parsed["_fallback_extractor"] = False

    print("Parsed dict (preview):", json.dumps(parsed, indent=2, default=str)[:2000])

    # 3) prediction
    model = load_model("models/model.pkl")
    try:
        if model is not None:
            pred = predict_with_model(model, parsed)
        else:
            pred = comparative_predict(parsed)
    except Exception as e:
        print("Model predict failed:", e)
        print("Falling back to rule-based comparative_predict.")
        pred = comparative_predict(parsed)

    print("Prediction:", pred)
    return parsed, pred

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python tests/test_single_pdf.py <path/to/pdf>")
        sys.exit(1)
    run_on_file(sys.argv[1])
