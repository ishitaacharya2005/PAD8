#ocr_pad_extractor.py
import sys
import os
import re
import json
import math
import unicodedata
from pdf2image import convert_from_path
import pytesseract
import cv2
import numpy as np
import pandas as pd

# ----------------------------
# Configuration / thresholds
# ----------------------------
TESSERACT_LANG = None  # None or 'eng'
PDF_DPI = 300
OCR_CONFIG = "--psm 6"  # assume a block of text; change if layout differs

# ----------------------------
# ABI / severity helpers
# ----------------------------
def abi_to_label(abi):
    """
    Map ABI numeric to an integer label:
      0 = None/normal (no ABI-based PAD)
      1 = Mild
      2 = Moderate
      3 = Severe

    Clinical mapping:
      ABI >= 1.30 : non-compressible / unreliable -> treat as normal for ABI-based label (0)
      0.91 - 1.29 : Normal (0)
      0.71 - 0.90 : Mild (1)
      0.41 - 0.70 : Moderate (2)
      <= 0.40     : Severe (3)

    Returns None if ABI is clearly invalid/out-of-range.
    """
    try:
        a = float(abi)
    except Exception:
        return None

    # sanity: ignore obviously invalid OCR results
    if a <= 0.0 or a > 5.0:
        return None

    # non-compressible / calcified range - ABI unreliable clinically; treat as no ABI-based PAD label
    if a >= 1.30:
        return 0

    if 0.91 <= a < 1.30:
        return 0
    if 0.71 <= a <= 0.90:
        return 1
    if 0.41 <= a < 0.71:
        return 2
    return 3

def adjust_severity(base_label, parsed):
    base = int(base_label) if base_label is not None else 0
    score = base * 1.0
    if parsed.get("Absent_Flow") in (True, "True", "true", 1, "YES", "Yes", "yes"):
        score += 1.0
    mono = int(parsed.get("waveform_monophasic_count") or 0)
    tri = int(parsed.get("waveform_triphasic_count") or 0)
    if mono >= 2:
        score += 1.0
    elif mono == 1:
        score += 0.5
    if tri >= 2:
        score -= 0.5
    elif tri == 1:
        score -= 0.25
    try:
        mean_psv = float(parsed.get("mean_PSV")) if parsed.get("mean_PSV") is not None else None
    except Exception:
        mean_psv = None
    if mean_psv is not None:
        if mean_psv < 50:
            score += 1.0
        elif mean_psv < 70:
            score += 0.5
    adjusted = int(round(score))
    adjusted = max(0, min(3, adjusted))
    if base_label is not None:
        adjusted = max(adjusted, int(base_label))
    return adjusted

# ----------------------------
# OCR helpers
# ----------------------------
def load_pages_from_path(path):
    ext = os.path.splitext(path)[1].lower()
    imgs = []
    if ext == ".pdf":
        imgs = convert_from_path(path, dpi=PDF_DPI)
        imgs = [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in imgs]
    else:
        im = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(f"Could not read image {path}")
        imgs = [im]
    return imgs

def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if max(h, w) < 1200:
        scale = 1200.0 / max(h, w)
        gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 12)
    return th

def ocr_image_to_text(img):
    pre = preprocess_for_ocr(img)
    txt = pytesseract.image_to_string(pre, lang=TESSERACT_LANG, config=OCR_CONFIG)
    return txt

# ----------------------------
# Parsing helpers
# ----------------------------
NUM_RE = r"[-+]?\d*\.\d+|\d+"

FIELD_PATTERNS = {
    "Patient ID": [r"Patient\s*ID[:\s]*([A-Za-z0-9\-]+)"],
    "Patient Name": [r"Patient\s*Name[:\s]*([A-Za-z ,.\-]+)"],
    "Age": [r"Age[:\s]*(" + NUM_RE + r")\b"],
    "Sex": [r"Sex[:\s]*(Male|Female|M|F|male|female)"],
    "Limb": [r"Limb[:\s]*([A-Za-z0-9 \-]+)"],
    "ABI": [r"\bABI[:\s]*(" + NUM_RE + r")\b", r"Ankle[- ]Brachial.*?[:\s]*(" + NUM_RE + r")"],
    "Absent_Flow": [r"Absent\s*Flow[:\s]*(Yes|No|True|False|Absent|Present)"],
    "mean_PSV": [r"mean[_\s]*PSV[:\s]*(" + NUM_RE + r")", r"Mean\s*PSV[:\s]*(" + NUM_RE + r")"],
    "Age_alt": [r"Age[:\s]*(" + NUM_RE + r")\s*(?:yrs|years)?\b"],
    "Sex_alt": [r"Sex[:\s]*(Male|Female|M|F|male|female)"],
    "Patient Name alt": [r"Patient\s*Name[:\s]*([A-Za-z ,.\-]+?)(?:\s+Referring|Ref:|Referring Doctor|Doctor)"]
}

PSV_COLUMNS = {
    "Common Femoral Artery PSV": ["Common Femoral", "Common Femoral Artery PSV", "Common Femoral Artery PSV"],
    "Profundus Femoris PSV": ["Profundus Femoris PSV", "Profundus Femoris"],
    "Proximal SFA PSV": ["Proximal SFA PSV", "Proximal Superficial Femoral Artery PSV"],
    "Mid SFA PSV": ["Mid SFA PSV", "Mid Superficial Femoral Artery PSV", "Mid SFA"],
    "Distal SFA PSV": ["Distal SFA PSV", "Distal SFA"],
    "Popliteal Artery PSV": ["Popliteal Artery PSV", "Popliteal Artery"],
    "Peroneal / Posterior Tibial Artery PSV": ["Peroneal / Posterior Tibial Artery PSV", "Peroneal", "Posterior Tibial"],
    "Anterior Tibial Artery PSV": ["Anterior Tibial Artery PSV", "Anterior Tibial"]
}

WAVEFORM_PATTERNS = {
    "monophasic": r"\bmono(?:[-\s]?phasic)?\b|monophasic\b",
    "biphasic": r"\bbi(?:[-\s]?phasic)?\b|biphasic\b",
    "triphasic": r"\btri(?:[-\s]?phasic)?\b|triphasic\b"
}

def find_field_from_text(text, patterns):
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None

def extract_psv_values_from_text(text):
    out = {k: None for k in PSV_COLUMNS.keys()}
    for canon, labels in PSV_COLUMNS.items():
        for lbl in labels:
            pat = re.escape(lbl) + r"[:\s]*(" + NUM_RE + r")"
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                try:
                    out[canon] = float(m.group(1))
                except Exception:
                    out[canon] = None
                break
        if out[canon] is None:
            for lbl in labels:
                m = re.search(re.escape(lbl), text, flags=re.IGNORECASE)
                if m:
                    tail = text[m.end(): m.end()+80]
                    m2 = re.search(NUM_RE, tail)
                    if m2:
                        try:
                            out[canon] = float(m2.group(0))
                        except Exception:
                            out[canon] = None
                        break
    return out

def extract_age_sex(full_text):
    norm = re.sub(r"[\r\n]+", " ", full_text)
    norm = re.sub(r"\s{2,}", " ", norm)
    m = re.search(r"Age\s*[/\\]\s*Sex\s*[:\s]*(" + NUM_RE + r")\s*(?:yrs|years|y)?\s*[/\\]\s*(Male|Female|M|F)\b", norm, flags=re.IGNORECASE)
    if m:
        return m.group(1), m.group(2).capitalize(), 3
    m = re.search(r"Age\s*[/\\]?\s*Sex\s*[:\s]+\s*(" + NUM_RE + r")\s*(?:yrs|years|y)?\s*[/\\,]?\s*(Male|Female|M|F)\b", norm, flags=re.IGNORECASE)
    if m:
        return m.group(1), m.group(2).capitalize(), 3
    m = re.search(r"\b(" + NUM_RE + r")\s*[/\\]\s*(M|F|Male|Female)\b", norm, flags=re.IGNORECASE)
    if m:
        return m.group(1), m.group(2).capitalize(), 3
    m = re.search(r"Age[:\s]*(" + NUM_RE + r")\s*(?:yrs|years|y)?[,\s;/-]+(?:Sex[:\s]*)?(Male|Female|M|F)\b", norm, flags=re.IGNORECASE)
    if m:
        return m.group(1), m.group(2).capitalize(), 2
    age = find_field_from_text(norm, FIELD_PATTERNS.get("Age_alt", []))
    sex = find_field_from_text(norm, FIELD_PATTERNS.get("Sex_alt", []))
    if age or sex:
        return age, (sex.capitalize() if sex else None), 2
    m = re.search(r"\b(" + NUM_RE + r")\s*(?:yrs|years|y)\b[,\s]*(Male|Female|M|F)\b", norm, flags=re.IGNORECASE)
    if m:
        return m.group(1), m.group(2).capitalize(), 2
    m = re.search(r"\b(" + NUM_RE + r")\s*[,/]\s*(Male|Female|M|F)\b", norm, flags=re.IGNORECASE)
    if m:
        return m.group(1), m.group(2).capitalize(), 1
    return None, None, 0

def count_waveforms_in_text(text):
    mono_pat = r"\bmono(?:[-\s]?phasic)?\b|monophasic\b"
    bi_pat = r"\bbi(?:[-\s]?phasic)?\b|biphasic\b"
    tri_pat = r"\btri(?:[-\s]?phasic)?\b|triphasic\b"
    mono = len(re.findall(mono_pat, text, flags=re.IGNORECASE))
    bi = len(re.findall(bi_pat, text, flags=re.IGNORECASE))
    tri = len(re.findall(tri_pat, text, flags=re.IGNORECASE))
    return mono, bi, tri

# ----------------------------
# Main extraction routine
# ----------------------------
def extract_from_report(path):
    pages = load_pages_from_path(path)
    all_text = []
    for p in pages:
        t = ocr_image_to_text(p)
        all_text.append(t)
    full_text = "\n".join(all_text)

    parsed = {k: None for k in [
        "Patient ID", "Patient Name", "Age", "Sex", "Limb", "ABI",
        "Common Femoral Artery PSV", "Common Femoral Artery Waveform", "Common Femoral Artery Remarks",
        "Profundus Femoris PSV", "Profundus Femoris Waveform", "Profundus Femoris Remarks",
        "Proximal SFA PSV", "Proximal SFA Waveform", "Proximal SFA Remarks",
        "Mid SFA PSV", "Mid SFA Waveform", "Mid SFA Remarks",
        "Distal SFA PSV", "Distal SFA Waveform", "Distal SFA Remarks",
        "Popliteal Artery PSV", "Popliteal Artery Waveform", "Popliteal Artery Remarks",
        "Peroneal / Posterior Tibial Artery PSV", "Peroneal / Posterior Tibial Artery Waveform", "Peroneal / Posterior Tibial Artery Remarks",
        "Anterior Tibial Artery PSV", "Anterior Tibial Artery Waveform", "Anterior Tibial Artery Remarks",
        "waveform_monophasic_count", "waveform_biphasic_count", "waveform_triphasic_count", "Absent_Flow", "mean_PSV", "label"
    ]}

    for field, patterns in FIELD_PATTERNS.items():
        v = find_field_from_text(full_text, patterns)
        if v is not None:
            parsed[field] = v

    age_val, sex_val, conf = extract_age_sex(full_text)
    parsed["Age"] = parsed.get("Age") or age_val
    parsed["Sex"] = parsed.get("Sex") or sex_val
    parsed["_age_sex_confidence"] = conf

    pn = find_field_from_text(full_text, FIELD_PATTERNS.get("Patient Name", []))
    if pn:
        pn_clean = re.split(r"\bReferring\b|\bRef:|\bReferring Doctor\b|\bDoctor\b", pn, flags=re.IGNORECASE)[0].strip()
        parsed["Patient Name"] = pn_clean
    else:
        pn2 = find_field_from_text(full_text, FIELD_PATTERNS.get("Patient Name alt", []))
        if pn2:
            parsed["Patient Name"] = pn2

    psv_vals = extract_psv_values_from_text(full_text)
    for k, v in psv_vals.items():
        parsed[k] = v

    lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
    for ln in lines:
        for canon, labels in PSV_COLUMNS.items():
            for lbl in labels:
                if re.search(re.escape(lbl), ln, flags=re.IGNORECASE):
                    num_match = re.search(NUM_RE, ln)
                    if num_match and parsed.get(canon) is None:
                        try:
                            parsed[canon] = float(num_match.group(0))
                        except Exception:
                            pass
                    post = ln[num_match.end():] if num_match else ln
                    wf = None
                    if re.search(WAVEFORM_PATTERNS["monophasic"], post, flags=re.IGNORECASE):
                        wf = "Monophasic"
                    elif re.search(WAVEFORM_PATTERNS["biphasic"], post, flags=re.IGNORECASE):
                        wf = "Biphasic"
                    elif re.search(WAVEFORM_PATTERNS["triphasic"], post, flags=re.IGNORECASE):
                        wf = "Triphasic"
                    if wf:
                        parsed[canon.replace(" PSV", " Waveform")] = wf
                    if "remark" in ln.lower():
                        m = re.search(r"remark[s]?[^\:]*[:\s]*(.*)$", ln, flags=re.IGNORECASE)
                        if m:
                            parsed[canon.replace(" PSV", " Remarks")] = m.group(1).strip()
                    break

    mono, bi, tri = count_waveforms_in_text(full_text)
    parsed["waveform_monophasic_count"] = parsed.get("waveform_monophasic_count") or mono
    parsed["waveform_biphasic_count"] = parsed.get("waveform_biphasic_count") or bi
    parsed["waveform_triphasic_count"] = parsed.get("waveform_triphasic_count") or tri

    af = parsed.get("Absent_Flow")
    if af is None:
        if re.search(r"\b(absent flow|no flow detected|absent arterial flow|no arterial flow)\b", full_text, flags=re.IGNORECASE):
            parsed["Absent_Flow"] = True
        else:
            parsed["Absent_Flow"] = False
    else:
        af_str = str(af).strip()
        parsed["Absent_Flow"] = True if re.search(r"\b(yes|true|absent)\b", af_str, flags=re.IGNORECASE) else False

    if parsed.get("mean_PSV") is None:
        vals = []
        for k in PSV_COLUMNS.keys():
            v = parsed.get(k)
            if v is not None:
                try:
                    vals.append(float(v))
                except Exception:
                    pass
        parsed["mean_PSV"] = float(np.mean(vals)) if vals else None

    label_from_abi = abi_to_label(parsed.get("ABI"))
    parsed["label"] = label_from_abi
    adjusted = adjust_severity(label_from_abi, parsed)
    parsed["label_adjusted"] = adjusted

    # --- CHANGE: only moderate+ are considered PAD here (i.e., label >= 2) ---
    conclusion = {
        "PAD_present": bool(adjusted >= 2),
        "severity_label": adjusted,
        "severity_text": {0: "None", 1: "Mild", 2: "Moderate", 3: "Severe"}[adjusted]
    }

    # Normalization helpers (very aggressive unicode whitespace handling)
    def normalize_str(s):
        if s is None:
            return None
        s = str(s)
        s = unicodedata.normalize("NFKC", s)
        # replace many unicode spaces with ASCII space
        s = re.sub(r"[\u00A0\u1680\u180E\u2000-\u200B\u202F\u205F\u2060\u3000\uFEFF]", " ", s)
        s = re.sub(r"[\r\n\t]+", " ", s)
        s = re.sub(r"\s{2,}", " ", s)
        return s.strip() or None

    for k, v in list(parsed.items()):
        if isinstance(v, str):
            parsed[k] = normalize_str(v)

    # Coerce PSVs (handle "39.8 / 42.2" -> max)
    def extract_max_number(x):
        if x is None:
            return None
        s = str(x)
        nums = re.findall(NUM_RE, s)
        if not nums:
            return None
        try:
            vals = [float(n) for n in nums]
            return round(max(vals), 1)
        except Exception:
            return None

    for psv_key in [
        "Common Femoral Artery PSV", "Profundus Femoris PSV", "Proximal SFA PSV",
        "Mid SFA PSV", "Distal SFA PSV", "Popliteal Artery PSV",
        "Peroneal / Posterior Tibial Artery PSV", "Anterior Tibial Artery PSV",
        "mean_PSV"
    ]:
        parsed[psv_key] = extract_max_number(parsed.get(psv_key))

    for wf_key in ["waveform_monophasic_count", "waveform_biphasic_count", "waveform_triphasic_count"]:
        v = parsed.get(wf_key)
        try:
            parsed[wf_key] = int(float(v)) if v is not None else 0
        except Exception:
            parsed[wf_key] = 0

    af = parsed.get("Absent_Flow")
    if isinstance(af, str):
        af_norm = normalize_str(af)
        parsed["Absent_Flow"] = True if re.search(r"\b(yes|true|absent)\b", af_norm or "", flags=re.IGNORECASE) else False
    else:
        parsed["Absent_Flow"] = bool(af)

    parsed["_flags_missing"] = {}
    for req in ["Age", "ABI", "waveform_triphasic_count", "waveform_monophasic_count"]:
        parsed["_flags_missing"][req] = parsed.get(req) is None
    parsed["_flags_missing"]["age_sex_confidence_low"] = parsed.get("_age_sex_confidence", 0) < 2

    abi_loc = re.search(r"\bABI\b", full_text, flags=re.IGNORECASE)
    parsed["_ocr_snippet_near_ABI"] = (full_text[max(0, abi_loc.start()-60): max(0, abi_loc.start()-60)+180].replace("\n"," ")
                                      if abi_loc else None)

    return parsed, full_text, conclusion

# ----------------------------
# CLI entry (final)
# ----------------------------
def main(argv):
    if len(argv) < 2:
        print("Usage: python ocr_pad_extractor.py path/to/report.pdf")
        return
    path = argv[1]
    parsed, full_text, conclusion = extract_from_report(path)

    out_csv = os.path.join("data", "ocr_extracted_report.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Final cleaning before saving
    def final_norm_val(v):
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return v
        if isinstance(v, (dict, list, tuple)):
            return v
        s = str(v)
        s = unicodedata.normalize("NFKC", s)
        s = re.sub(r"[\u00A0\u1680\u180E\u2000-\u200B\u202F\u205F\u2060\u3000\uFEFF]", " ", s)
        s = re.sub(r"[\r\n\t]+", " ", s)
        s = re.sub(r"\s{2,}", " ", s)
        s = s.strip()
        if s == "":
            return None
        nums = re.findall(NUM_RE, s)
        if len(nums) == 1 and re.fullmatch(r"\s*"+NUM_RE+r"\s*", s):
            try:
                return float(nums[0])
            except Exception:
                return s
        if len(nums) > 1:
            try:
                vals = [float(x) for x in nums]
                return float(max(vals))
            except Exception:
                return s
        return s

    for k in list(parsed.keys()):
        if isinstance(parsed[k], (dict, list, tuple)):
            continue
        parsed[k] = final_norm_val(parsed[k])

    for int_key in ["Patient ID", "Age"]:
        v = parsed.get(int_key)
        if isinstance(v, float) and v.is_integer():
            parsed[int_key] = int(v)

    def collapse_and_strip(s):
        if s is None:
            return None
        if not isinstance(s, str):
            return s
        s = unicodedata.normalize("NFKC", s)
        s = re.sub(r"[\u00A0\u1680\u180E\u2000-\u200B\u202F\u205F\u2060\u3000\uFEFF]", " ", s)
        s = re.sub(r"[\r\n\t]+", " ", s)
        s = re.sub(r"\s{2,}", " ", s)
        return s.strip()

    for k in list(parsed.keys()):
        v = parsed.get(k)
        if isinstance(v, str):
            parsed[k] = collapse_and_strip(v)

    for psv_key in [
        "Common Femoral Artery PSV", "Profundus Femoris PSV", "Proximal SFA PSV",
        "Mid SFA PSV", "Distal SFA PSV", "Popliteal Artery PSV",
        "Peroneal / Posterior Tibial Artery PSV", "Anterior Tibial Artery PSV",
        "mean_PSV"
    ]:
        val = parsed.get(psv_key)
        if isinstance(val, str):
            nums = re.findall(NUM_RE, val)
            if nums:
                try:
                    parsed[psv_key] = round(float(max(map(float, nums))), 1)
                except Exception:
                    parsed[psv_key] = None
            else:
                parsed[psv_key] = parsed[psv_key] or None

    # Final unicode-clean and whitespace collapse for all strings
    def force_clean_str(s):
        if s is None:
            return None
        if not isinstance(s, str):
            return s
        s = unicodedata.normalize("NFKC", s)
        s = re.sub(r"[\u00A0\u1680\u180E\u2000-\u200B\u202F\u205F\u2060\u3000\uFEFF]", " ", s)
        s = re.sub(r"[\r\n\t]+", " ", s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    for k in list(parsed.keys()):
        if isinstance(parsed[k], str):
            parsed[k] = force_clean_str(parsed[k])

    flat = parsed.copy()
    flags = flat.pop("_flags_missing", {})  # should be dict
    for flag_k, flag_v in flags.items():
        flat[f"_missing_{flag_k}"] = flag_v
    flat["_age_sex_confidence"] = flat.get("_age_sex_confidence", 0)

    # ensure csv strings cleaned
    for k, v in list(flat.items()):
        if isinstance(v, str):
            flat[k] = re.sub(r"\s+", " ", v).strip()

    df = pd.DataFrame([flat])
    df["conclusion_PAD_present"] = conclusion["PAD_present"]
    df["conclusion_severity_label"] = conclusion["severity_label"]
    df["conclusion_severity_text"] = conclusion["severity_text"]
    df.to_csv(out_csv, index=False)

    # pretty print helper (ensures no trailing spaces)
    def fmt_for_print(k, v):
        if v is None:
            return "None"
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            if "PSV" in k or k == "mean_PSV":
                return f"{v:.1f}"
            if k == "ABI":
                return f"{v:.2f}"
            return repr(v)
        if isinstance(v, dict):
            return json.dumps(v, ensure_ascii=False)
        return force_clean_str(str(v))

    print("Extracted fields saved to", out_csv)
    print("Conclusion:", conclusion)
    print("\nQuick preview of extracted fields:")
    # final print loop: collapse unicode whitespace and rstrip to eliminate trailing spaces
    for k, v in parsed.items():
        if k == "_ocr_snippet_near_ABI":
            snippet = v[:200] if v else v
            # collapse unicode spaces then rstrip
            if snippet:
                snippet = re.sub(r"[\u00A0\u1680\u180E\u2000-\u200B\u202F\u205F\u2060\u3000\uFEFF]+", " ", snippet)
                snippet = re.sub(r"\s+", " ", snippet).rstrip()
            print(f"{k}: {snippet}")
        else:
            printed = fmt_for_print(k, v)
            # collapse unicode spaces and then rstrip — final safeguard
            printed = re.sub(r"[\u00A0\u1680\u180E\u2000-\u200B\u202F\u205F\u2060\u3000\uFEFF]+", " ", str(printed))
            printed = re.sub(r"\s+", " ", printed).rstrip()
            print(f"{k}: {printed}")

    with open(os.path.join("data", "ocr_full_text.txt"), "w", encoding="utf8") as f:
        f.write(full_text)
    print("\n✅ FINAL CONCLUSION:")
    print(f"PAD Present: {conclusion['PAD_present']}")
    print(f"Severity: {conclusion['severity_text']} ({conclusion['severity_label']})")


if __name__ == "__main__":
    main(sys.argv)
