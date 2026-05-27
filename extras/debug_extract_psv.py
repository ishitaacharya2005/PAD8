# debug_extract_psv.py
import re, os, sys
from pprint import pprint

OCR_PREVIEW = "p1.ocr_preview.txt"

if not os.path.exists(OCR_PREVIEW):
    print("Missing OCR preview:", OCR_PREVIEW)
    sys.exit(1)

with open(OCR_PREVIEW, "r", encoding="utf-8", errors="ignore") as f:
    lines = [ln.rstrip() for ln in f.readlines()]

print("=== File preview (lines with numbers/artery keywords highlighted) ===\n")
for i, ln in enumerate(lines):
    marker = ""
    if re.search(r'\b(artery|psv|triphasic|biphasic|monophasic|ankle|brachial|sfa|peroneal|popliteal)\b', ln, re.I):
        marker = "<--"
    print(f"{i+1:03d}: {ln} {marker}")

print("\n=== Attempting PSV extraction (table-aware) ===\n")

ARTERIES = [
    "Common Femoral", "Profundus Femoris", "Profunda Femoris",
    "Proximal Superficial", "Mid SFA", "Distal SFA",
    "Popliteal", "Peroneal", "Posterior Tibial", "Anterior Tibial", "Anterior Tibial Artery"
]

def safe_float(s):
    try:
        s = s.replace(",", ".")
        return float(s)
    except:
        return None

psv_results = {}

# Strategy A: artery name in line -> number in same line
for idx, ln in enumerate(lines):
    for art in ARTERIES:
        if re.search(re.escape(art), ln, re.I):
            m = re.search(r'([0-9]{1,4}(?:\.[0-9]+)?(?:\s*\/\s*[0-9]{1,4}(?:\.[0-9]+)?)?)', ln)
            if m:
                raw = m.group(1)
                v = safe_float(raw.split("/")[0].strip())
                if v is not None:
                    psv_results[art] = (v, idx+1, "same-line")
            else:
                # look next 1-2 lines for numeric tokens (table-style)
                for j in (idx+1, idx+2):
                    if j < len(lines):
                        m2 = re.search(r'([0-9]{1,4}(?:\.[0-9]+)?(?:\s*\/\s*[0-9]{1,4}(?:\.[0-9]+)?)?)', lines[j])
                        if m2:
                            raw = m2.group(1)
                            v = safe_float(raw.split("/")[0].strip())
                            if v is not None:
                                psv_results[art] = (v, j+1, "next-line")
                                break

# Strategy B: generic PSV: lines containing "PSV" label
for idx, ln in enumerate(lines):
    if re.search(r'\bPSV[:\s]', ln, re.I):
        m = re.search(r'PSV[:\s]*([0-9]{1,4}(?:\.[0-9]+)?)', ln, re.I)
        if m:
            psv_results.setdefault("PSV_generic", (float(m.group(1)), idx+1, "PSV-line"))

# Strategy C: lines that look like table rows: <artery name> newline <number ...>
# Already partially covered by next-line logic above.

print("PSV extraction results (artery -> (value, line#, how_found)):")
pprint(psv_results)

# Print context for each found result
for art, (val, line_no, how) in psv_results.items():
    print("\n---", art, f"found at line {line_no} ({how}) ---")
    start = max(0, line_no-3)
    end = min(len(lines), line_no+2)
    for i in range(start, end):
        print(f"{i+1:03d}: {lines[i]}")
