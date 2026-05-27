# parser_utils.py
"""
Robust parser for Doppler reports.

- Fuzzy label search for Patient Name / Age / Sex
- Improved name cleaning (keeps initials, attempts to prefer multi-word names)
- Age extraction with OCR-failure heuristics (fixes dropped leading digits like 72 -> 2)
- PSV extraction with normalized PSV__ keys
- Waveform counts
Requires: regex (pip install regex)
"""

import regex as re
import math
from typing import Dict, Any
import cv2
import numpy as np
from PIL import Image
import pytesseract

def focused_reocr_age(img_bgr, ocr_text):
    """
    img_bgr : numpy BGR image (full page)
    ocr_text: raw OCR text (string) -- used for fallback matching
    Returns (age_candidate_str, debug_info)
    - tries to locate 'Age' tokens via image_to_data, crops ROI, re-ocr with many ops,
      and aggregates digits from multiple passes/boxes left->right.
    - debug_info contains the list of OCR outputs tried for inspection.
    """
    debug = {"roi_boxes": [], "pass_texts": []}
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    data = pytesseract.image_to_data(pil, config="--oem 3 --psm 6", lang='eng', output_type=pytesseract.Output.DICT)
    # find indices that look like Age/ Age/Sex labels
    indices = [i for i,t in enumerate(data['text']) if t and ('age' in t.lower() or 'age/sex' in t.lower() or 'age:' in t.lower())]
    if not indices:
        # fallback: choose tokens that contain '/' or 'yrs' near digits
        indices = [i for i,t in enumerate(data['text']) if t and ('/' in t or 'yrs' in t.lower() or 'years' in t.lower())]
    if not indices:
        return (None, {"reason": "no_age_token_found"})

    all_digits_seen = []
    ocr_passes = []

    def crop_roi(i):
        x = int(data['left'][i]); y = int(data['top'][i]); w = int(data['width'][i]); h = int(data['height'][i])
        margin_x = max(20, int(w*6)); margin_y = max(10, int(h*3))
        x0 = max(0, x-margin_x); y0 = max(0, y-margin_y)
        x1 = min(img_bgr.shape[1], x+w+margin_x); y1 = min(img_bgr.shape[0], y+h+margin_y)
        roi = img_bgr[y0:y1, x0:x1].copy()
        debug["roi_boxes"].append((x0,y0,x1,y1))
        return roi

    # preproc variants to try
    def preprocess_variants(roi):
        variants = {}
        # resize large
        roi_r = cv2.resize(roi, (int(roi.shape[1]*2.5), int(roi.shape[0]*2.5)), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(roi_r, cv2.COLOR_BGR2GRAY)
        variants["orig"] = gray
        variants["clahe"] = cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8)).apply(gray)
        variants["bilat"] = cv2.bilateralFilter(gray, d=9, sigmaColor=100, sigmaSpace=100)
        blur = cv2.GaussianBlur(gray,(3,3),0)
        _, th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        variants["otsu"] = th
        variants["median"] = cv2.medianBlur(gray, 3)
        variants["inv"] = cv2.bitwise_not(gray)
        return variants

    for idx in indices:
        roi = crop_roi(idx)
        variants = preprocess_variants(roi)
        for vname, mat in variants.items():
            pil_roi = Image.fromarray(mat)
            # try several configs, include digits whitelist
            configs = [
                "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789/",
                "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789/",
                "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789",
                "--oem 3 --psm 6"
            ]
            for cfg in configs:
                try:
                    txt = pytesseract.image_to_string(pil_roi, config=cfg, lang='eng') or ""
                except Exception as e:
                    txt = ""
                cleaned = re.sub(r'[^0-9/]', '', txt)
                ocr_passes.append({"index": idx, "variant": vname, "cfg": cfg, "raw": txt.strip(), "clean": cleaned})
                # collect digits in order of appearance
                for ch in cleaned:
                    if ch.isdigit():
                        all_digits_seen.append((ch, vname, cfg))
    debug["pass_texts"] = ocr_passes

    # Try to reconstruct plausible age from observed digits.
    # Heuristics:
    #  - if a 2-digit number appears anywhere, use it
    #  - else, if we saw multiple single digits in different passes, try to order them by pass preference (rare)
    digits_concat = "".join([d for (d,_,_) in all_digits_seen])
    # search for contiguous two-digit substring
    m = re.search(r'([2-9][0-9])', digits_concat)  # prefer ages >=20
    if m:
        return (m.group(1), debug)
    # fallback: if any digits seen, return the concatenation (may be '72' or '27' depending)
    if digits_concat:
        # attempt to pick left-to-right order via character boxes (best-effort)
        # but here we'll return the first two digits seen
        return (digits_concat[:2], debug)
    return (None, debug)


ARTERIES = [
    "Common Femoral Artery", "Profundus Femoris", "Profunda Femoris",
    "Proximal Superficial Femoral Artery", "Proximal SFA",
    "Mid SFA", "Mid Superficial Femoral Artery",
    "Distal SFA", "Distal Superficial Femoral Artery",
    "Popliteal Artery", "Peroneal / Posterior Tibial Artery",
    "Posterior Tibial Artery", "Peroneal Artery",
    "Anterior Tibial Artery", "Anterior Tibial"
]

def safe_float(s):
    """Coerce many OCR-ish numeric strings to float, or return None."""
    try:
        if s is None:
            return None
        s = str(s).strip()
        s = s.replace(",", ".")
        # remove any non-digit/dot characters
        s = re.sub(r'[^\d\.]', '', s)
        if s == "":
            return None
        return float(s)
    except Exception:
        return None

def _compact_text(text: str) -> str:
    """Normalize newlines and whitespace to make regex simpler."""
    if not text:
        return ""
    text = text.replace('\r', '\n')
    # collapse many newlines to single newline
    text = re.sub(r'\n{2,}', '\n', text)
    # strip unusual Unicode separators that OCR sometimes inserts
    text = text.replace('\u2022', ' ').replace('\u25cf', ' ')
    return text.strip()

def fuzzy_label_search(label: str, text: str, max_errors: int = 1, grab_chars: int = 80):
    """
    Fuzzy search for a label allowing up to max_errors edits.
    Returns the captured following text or None.
    """
    if not text:
        return None
    pat = rf'({re.escape(label)}){{e<={max_errors}}}[\s:—\-]*([^\n]{{1,{grab_chars}}})'
    m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(2).strip().strip(':').strip()
    return None

def _clean_name_fragment(raw: str, max_words: int = 3):
    """
    Clean a name fragment:
      - remove strange OCR symbols
      - remove stray digits and trailing boilerplate
      - keep initials (e.g., 'B S' -> 'B S')
      - prefer alphabetic words, up to max_words
    """
    if raw is None:
        return None
    s = str(raw).strip()
    # common OCR garbage replaced with spaces
    s = re.sub(r'[\=\~\`\@\#\$\%\^\*\_\+\|\[\]\{\}\<\>\;\\\/]+', ' ', s)
    # preserve letters, apostrophes, hyphens, dots and spaces; remove other symbols
    s = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ'\.\-\s]", ' ', s)
    s = re.sub(r'\s{2,}', ' ', s).strip()
    s = s.strip(" .,-")
    # truncate at known trailing labels
    s = re.split(r'\bAge\b|\bSex\b|\bDOB\b|\bMRN\b|\bID\b|\bPatient ID\b', s, flags=re.IGNORECASE)[0].strip()
    if not s:
        return None
    words = [w for w in s.split() if w]
    if not words:
        return None
    # remove trailing tokens that are clearly not names
    bad_end = re.compile(r'^(years|yrs|year|male|female|m|f)$', re.IGNORECASE)
    while words and bad_end.match(words[-1]):
        words = words[:-1]
    if not words:
        return None
    # if tokens are single letters (initials), allow up to max_words (e.g. B S K)
    if all(len(w) <= 2 for w in words):
        words = words[:max_words]
        candidate = " ".join(words)
        # candidate must have at least one letter
        if re.search(r'[A-Za-z]', candidate):
            return candidate
    # otherwise prefer real-word tokens which start with capital letter if possible
    # pick the first max_words tokens that look like a name
    out_words = []
    for w in words:
        # discard tokens that look like garbage (long sequences of same letter)
        if re.match(r'^[A-Za-zÀ-ÖØ-öø-ÿ\'\-\.]{1,30}$', w):
            out_words.append(w)
        if len(out_words) >= max_words:
            break
    if not out_words:
        return None
    candidate = " ".join(out_words)
    # final guard: must contain at least 1 alphabetic char
    if not re.search(r'[A-Za-z]', candidate):
        return None
    # avoid returning generic words
    if re.search(r'\b(report|hospital|department|ref|admission|patient|id|age|sex)\b', candidate, flags=re.IGNORECASE):
        return None
    return candidate

def find_patient_info(text: str) -> Dict[str, str]:
    """
    Extract Patient Name, Age, Sex robustly.
    Returns a dict with keys possibly: Patient Name, Age, Sex
    """
    out: Dict[str, str] = {}
    if not text:
        return out
    t = _compact_text(text)

    # 1) Try explicit labeled forms
    name_capture = fuzzy_label_search("Patient Name", t, max_errors=1, grab_chars=120)
    if not name_capture:
        name_capture = fuzzy_label_search("Name", t, max_errors=1, grab_chars=120)
    if name_capture:
        # stop at Age/Sex if present in the capture
        name_part = re.split(r'\bAge\b|\bAged\b|\bSex\b|\bGender\b', name_capture, flags=re.IGNORECASE)[0].strip()
        cleaned = _clean_name_fragment(name_part, max_words=3)
        if cleaned:
            out["Patient Name"] = cleaned

    # 2) Fallback: scan top lines for likely name (first 12 lines)
    if "Patient Name" not in out:
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        top_lines = lines[:12]
        candidates = []
        for ln in top_lines:
            if len(ln) < 3:
                continue
            if re.search(r'\b(age|sex|dob|report|hospital|ref|mrn|patient id|admission|department)\b', ln, flags=re.IGNORECASE):
                continue
            ln_short = re.split(r'\bAge\b|\bSex\b|\bDOB\b|\bID\b|\bMRN\b', ln, flags=re.IGNORECASE)[0].strip()
            cleaned = _clean_name_fragment(ln_short, max_words=3)
            if not cleaned:
                continue
            # scoring: prefer multi-word, capitalized tokens
            words = cleaned.split()
            score = 0
            score += min(3, len(words))
            score += sum(1 for w in words if re.match(r'^[A-Z][a-z]', w))
            # small penalty for obvious noise words
            if re.match(r'^(patient|name|report|hospital)$', cleaned, flags=re.IGNORECASE):
                score -= 5
            candidates.append((score, cleaned, ln))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best = candidates[0][1]
            if best:
                out["Patient Name"] = best

    # 3) Age extraction with OCR heuristics
    age_str = None
    # try fuzzy label first
    try_age = fuzzy_label_search("Age", t, max_errors=1, grab_chars=40)
    if try_age:
        m = re.search(r'([0-9]{1,3})', try_age)
        if m:
            age_str = m.group(1)
    if not age_str:
        # common patterns in document body
        m = re.search(r'\bAge[:\s]*([0-9]{1,3})\b', t, flags=re.IGNORECASE)
        if m:
            age_str = m.group(1)
        else:
            m2 = re.search(r'\b([0-9]{1,3})\s*(?:yrs|years|y)\b', t, flags=re.IGNORECASE)
            if m2:
                age_str = m2.group(1)

    # Additional robust attempt: find "Age / Sex" patterns like "72 / Female" or "Age/Sex: 72/F"
    if not age_str:
        m3 = re.search(r'Age[^\d\n]{0,6}([0-9]{1,3})\s*[/\\]\s*([A-Za-z])', t, flags=re.IGNORECASE)
        if m3:
            age_str = m3.group(1)

    # Heuristic: if age_str is small (<20) try to find a nearby 2-digit number on same line (OCR dropped tens digit)
    if age_str:
        try:
            age_val = int(float(age_str))
        except Exception:
            age_val = None
        # look on same line for any two-digit number
        if age_val is not None and age_val < 20:
            # attempt to find a 2-digit number near any 'Age' label occurrence
            for m_label in re.finditer(r'Age', t, flags=re.IGNORECASE):
                start = m_label.start()
                # take substring 0..50 chars after label
                tail = t[start:start+60]
                m_two = re.search(r'([0-9]{2,3})', tail)
                if m_two:
                    candidate_age = int(m_two.group(1))
                    if candidate_age >= 20:
                        age_val = candidate_age
                        break
            # general fallback: find any 2-digit number in the whole doc > 20 that might be the age
            if age_val < 20:
                m_global = re.search(r'\b([2-9][0-9])\b', t)
                if m_global:
                    age_val = int(m_global.group(1))

        if age_val is not None:
            out["Age"] = str(age_val)
        else:
            out["Age"] = age_str

    # 4) Sex extraction
    sex = None
    try_sex = fuzzy_label_search("Sex", t, max_errors=1, grab_chars=30)
    if try_sex:
        m = re.search(r'\b(Male|Female|M|F|Other|male|female|m|f)\b', try_sex)
        if m:
            sex = m.group(1)
    if not sex:
        try_gen = fuzzy_label_search("Gender", t, max_errors=1, grab_chars=30)
        if try_gen:
            m = re.search(r'\b(Male|Female|M|F|Other|male|female|m|f)\b', try_gen)
            if m:
                sex = m.group(1)
    if not sex:
        # fallback: look for pattern like "72/M" or "72 / M" near Age text
        m = re.search(r'Age[:\s]*\d{1,3}[^A-Za-z0-9]{0,8}(Male|Female|M\b|F\b)', t, flags=re.IGNORECASE)
        if m:
            sex = m.group(1)
    if sex:
        sex = sex.strip().upper()
        if sex in ("M", "MALE"):
            out["Sex"] = "Male"
        elif sex in ("F", "FEMALE"):
            out["Sex"] = "Female"
        else:
            out["Sex"] = sex.title()

    return out

# ABI finder (unchanged behavior, robust)
def find_abi(text: str):
    if not text:
        return None
    txt = text.replace('\n', ' ')
    patterns = [
        r'\bABI[:\s\-]*([0-9]\.\d{1,3})\b',
        r'Ankle[ \-]Brachial(?: Index)?[:\s\-]*([0-9]\.\d{1,3})\b',
        r'\bABI[:\s\-]*([0]\.\d{1,3})\b',
        r'\bABI[:\s\-]*(\.\d{1,3})\b'
    ]
    for p in patterns:
        m = re.search(p, txt, flags=re.IGNORECASE)
        if m:
            v = safe_float(m.group(1))
            if v is not None and not math.isnan(v):
                return v
    m2 = re.search(r'(ABI|Ankle[ \-]Brachial(?: Index)?)', txt, flags=re.IGNORECASE)
    if m2:
        start = m2.end()
        tail = txt[start:start+40]
        m3 = re.search(r'([0-9]\.\d{1,3}|\.\d{1,3})', tail)
        if m3:
            return safe_float(m3.group(1))
    return None

def extract_psv_rows(text):
    """
    Table-aware PSV extractor:
      - finds a block of artery names above the 'PSV' header
      - finds a block of PSV values below the header
      - maps them in order to produce PSV__<Artery> keys
    Returns dict of PSV__<Normalized_Artery_Name> -> float
    """
    results = {}
    if not text:
        return results

    # Normalize lines (strip, remove empty)
    lines = [ln.strip() for ln in text.splitlines()]
    # keep only non-empty but preserve relative ordering for blocks
    nonempty = [ln for ln in lines if ln and not re.match(r'^\s*$', ln)]

    # find header index for PSV
    header_idx = None
    for i, ln in enumerate(nonempty):
        if re.search(r'psv\s*\(?cm\/s\)?\s*waveform', ln, flags=re.IGNORECASE) or re.search(r'\bpsv\b', ln, flags=re.IGNORECASE) and 'waveform' in ln.lower():
            header_idx = i
            break
    if header_idx is None:
        # fallback: try any line containing 'PSV (cm/s)' or 'PSV' alone
        for i, ln in enumerate(nonempty):
            if re.search(r'psv', ln, flags=re.IGNORECASE):
                header_idx = i
                break

    # helper to sanitize artery name -> key
    def key_from_name(name):
        return "PSV__" + re.sub(r'[^A-Za-z0-9/_]+', '_', name).strip('_')

    # heuristic autocorrect for obvious OCR numeric errors (e.g., 774 -> 77.4)
    def autocorrect_psv_value(v):
        try:
            fv = float(v)
        except Exception:
            return v
        if fv >= 300 and fv < 5000:
            cand1 = fv / 10.0
            cand2 = fv / 100.0
            for c in (cand2, cand1):
                if 5.0 <= c <= 400.0:
                    return round(c, 2)
        return fv

    if header_idx is not None:
        # collect artery block: lines above header until a blank or a known section marker
        # scan upwards from header_idx-1 collecting lines that look like artery names
        arteries = []
        up = header_idx - 1
        while up >= 0:
            ln = nonempty[up].strip()
            # stop if we hit a known section boundary like "Findings" / "Impression" / "Ankle"
            if re.search(r'\b(impression|findings|ankle|referral|technique|left lower limb|right lower limb)\b', ln, re.IGNORECASE):
                break
            # treat lines with letters (not numeric-heavy) as artery names
            if re.search(r'[A-Za-z]', ln) and not re.search(r'\d', ln):
                arteries.insert(0, ln)  # insert at front to preserve top-to-bottom order
                up -= 1
                continue
            else:
                # if line contains both words and digits, might be garbage; stop if too many non-artery lines
                # but allow some lines like 'Proximal Superficial' + 'Femoral Artery' split across two lines:
                # attempt to merge split name with previous if previous exists and was short
                # simple approach: if line has words and previous line also has words, include as continuation
                # check previous
                if arteries and re.search(r'[A-Za-z]', arteries[0]) and len(arteries[0].split()) < 4:
                    # merge this ln before the first artery (since we are scanning upwards)
                    arteries[0] = ln + " " + arteries[0]
                    up -= 1
                    continue
                break

        # collect PSV block: lines below header until a section boundary
        pv = header_idx + 1
        ps_values = []
        while pv < len(nonempty):
            ln = nonempty[pv].strip()
            # stop if next section starts
            if re.search(r'\b(ankle|impression|conclusion|remark|note)\b', ln, re.IGNORECASE):
                break
            # if this line contains a numeric PSV pattern, capture it
            m = re.search(r'([0-9]{1,4}(?:\.[0-9]+)?(?:\s*\/\s*[0-9]{1,4}(?:\.[0-9]+)?)?)', ln)
            if m:
                raw = m.group(1).strip()
                # if slash, take first part (e.g., "39.8 / 42.2")
                if "/" in raw:
                    first = raw.split("/")[0].strip()
                    val = safe_float(first)
                else:
                    val = safe_float(raw)
                if val is not None:
                    val = autocorrect_psv_value(val)
                    ps_values.append(val)
            pv += 1

        # If we found arteries and ps_values, map them in order
        if arteries and ps_values:
            # if counts differ, map up to min length and log remainder
            n = min(len(arteries), len(ps_values))
            for i in range(n):
                art = arteries[i]
                key = key_from_name(art)
                results[key] = float(ps_values[i])
            # if ps_values longer than arteries and generic not present, add generic keys
            if len(ps_values) > len(arteries):
                for j in range(len(arteries), len(ps_values)):
                    results[f"PSV__PSV_auto_{j-len(arteries)+1}"] = float(ps_values[j])
            return results

    # Fallback behaviors (previous/simple logic)

    # 1) same-line artery -> number matches
    for artery in ARTERIES:
        pat = rf'{re.escape(artery)}[^\d\n]{{0,60}}([0-9]{{1,4}}(?:\.[0-9]+)?)'
        for ln in nonempty:
            m = re.search(pat, ln, flags=re.IGNORECASE)
            if m:
                v = safe_float(m.group(1))
                if v is not None:
                    results[key_from_name(artery)] = autocorrect_psv_value(v)
                    break

    # 2) generic PSV tokens
    for ln in nonempty:
        if re.search(r'\bpsv[:\s]', ln, flags=re.IGNORECASE):
            m = re.search(r'PSV[:\s]*([0-9]{1,4}(?:\.[0-9]+)?)', ln, flags=re.IGNORECASE)
            if m:
                results.setdefault("PSV__PSV_generic", safe_float(m.group(1)))

    return results



def autocorrect_psv_value(val):
    """
    If OCR produced a suspiciously large integer (e.g., 774), try inserting a decimal:
    - If val >= 300 and val < 5000: try dividing by 10 or 100 and choose a plausible PSV.
    """
    try:
        v = float(val)
    except Exception:
        return val
    if v >= 300 and v < 5000:
        cand1 = v / 10.0
        cand2 = v / 100.0
        for c in (cand2, cand1):
            if 5.0 <= c <= 400.0:
                return round(c, 2)
    return v


def parse_report_text(text: str) -> Dict[str, Any]:
    """
    Main parser entrypoint.
    Returns dict with keys:
      - Patient Name, Age, Sex (when found)
      - ABI
      - PSV__<Normalized> keys for PSV values
      - waveform_monophasic_count, waveform_biphasic_count, waveform_triphasic_count
    """
    if not text or len(text) < 10:
    # return empty dict so downstream code can still run fallback logic and not crash
        return {}


    out: Dict[str, Any] = {}
    t = _compact_text(text)

    # Patient info
    pinfo = find_patient_info(t)
    out.update(pinfo)

    # ABI
    abi = find_abi(t)
    if abi is not None:
        out["ABI"] = abi

    # PSV extraction and normalise keys
    psvs = extract_psv_rows(text)
    for k, v in psvs.items():
        norm_key = k.replace(" ", "_").replace("/", "_").replace("__", "_").strip("_")
        out[f"PSV__{norm_key}"] = v

    # waveform counts
    out["waveform_monophasic_count"] = len(re.findall(r'\bMonophasic\b', text, flags=re.IGNORECASE))
    out["waveform_biphasic_count"] = len(re.findall(r'\bBiphasic\b', text, flags=re.IGNORECASE))
    out["waveform_triphasic_count"] = len(re.findall(r'\bTriphasic\b', text, flags=re.IGNORECASE))

def heuristic_fix_age(parsed_dict, ocr_text, img_bgr=None):
    """
    parsed_dict: dict returned by parse_report_text (may include 'Age' key)
    ocr_text: raw OCR text that parse_report_text used
    img_bgr: optional image (if available) to run focused_reocr_age
    Returns: (parsed_dict_modified, applied_fix_flag, reason)
    """
    age = parsed_dict.get("Age")
    try:
        if age is not None and str(age).isdigit():
            a = int(age)
        else:
            a = None
    except:
        a = None

    # If age missing or implausible (<18) and image present, try focused reocr
    if (a is None or a < 18) and img_bgr is not None:
        candidate, debug = focused_reocr_age(img_bgr, ocr_text)
        if candidate and candidate.isdigit():
            parsed_dict["Age"] = candidate
            parsed_dict.setdefault("debug", {})["age_fix_attempt"] = debug
            # now validate
            a = int(candidate)

    # If still implausible (<18) but PAD detected True -> conservative +70
    if a is not None and a < 18 and parsed_dict.get("PAD") or parsed_dict.get("pad_detected") == True:
        parsed_dict.setdefault("debug", {})["age_heuristic_applied"] = f"{a} -> {a+70}"
        parsed_dict["Age"] = str(a+70)
        parsed_dict["need_review"] = True
        return (parsed_dict, True, "age_added_70_due_to_PAD")
    # If we made any change and age still suspicious, flag for review
    if a is None or (a is not None and (a < 18 or a > 120)):
        parsed_dict["need_review"] = True
        return (parsed_dict, False, "age_suspicious_flagged")
    return (parsed_dict, False, "ok")

