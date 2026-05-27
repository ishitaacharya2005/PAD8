# tools/generate_eval_labels_from_html.py
import os
import re
import csv
from pathlib import Path

# directory containing your HTML reports
HTML_DIR = Path("output/html")
OUT_CSV = Path("eval_labels.csv")

# patterns to search for in HTML text (case-insensitive)
# we'll try to find severity or PAD present lines
PAD_PRESENT_PAT = re.compile(r"(PAD\s*Present[:\s]*)(Yes|No|True|False|Present|Absent)", re.I)
SEVERITY_PAT = re.compile(r"(Severity[:\s]*)(None|Mild|Moderate|Severe|0|1|2|3)", re.I)
FINAL_BLOCK_PAT = re.compile(r"FINAL\s+CONCLUSION:?|Conclusion:|Impression:", re.I)

# map textual severity -> numeric label (0..3)
SEV_TO_LABEL = {
    "none": 0,
    "0": 0,
    "mild": 1,
    "1": 1,
    "moderate": 2,
    "2": 2,
    "severe": 3,
    "3": 3
}

def text_to_label(text):
    if not text:
        return None
    t = text.strip().lower()
    return SEV_TO_LABEL.get(t)

def robust_extract_severity(html_text):
    # 1) try "Severity: <word>"
    m = SEVERITY_PAT.search(html_text)
    if m:
        label = text_to_label(m.group(2))
        if label is not None:
            return label

    # 2) try "PAD Present: Yes/No" and/or "FINAL CONCLUSION:" block
    m = PAD_PRESENT_PAT.search(html_text)
    if m:
        pad_val = m.group(2).strip().lower()
        if pad_val in ("no", "false", "absent"):
            return 0
        if pad_val in ("yes", "true", "present"):
            # fallback to mild if no severity available
            return 1

    # 3) scan for lines near "FINAL CONCLUSION" or "Conclusion" for severity words
    # take 200 chars after heading and look for keywords
    m = FINAL_BLOCK_PAT.search(html_text)
    if m:
        start = m.start()
        snippet = html_text[start:start+500]
        # look for severity keywords
        for word, lab in SEV_TO_LABEL.items():
            if re.search(r"\b" + re.escape(word) + r"\b", snippet, re.I):
                return lab

    # 4) look for 'Mild', 'Moderate', 'Severe' anywhere as last resort
    for word, lab in SEV_TO_LABEL.items():
        if re.search(r"\b" + re.escape(word) + r"\b", html_text, re.I):
            return lab

    return None

def generate_csv(html_dir=HTML_DIR, out_csv=OUT_CSV):
    if not html_dir.exists():
        raise SystemExit(f"{html_dir} not found. Update HTML_DIR at top of script.")

    rows = []
    files = sorted(html_dir.glob("*.html"))
    if not files:
        print("No HTML files found in", html_dir)
        return

    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = f.read_text(encoding="latin1", errors="ignore")

        label = robust_extract_severity(text)
        # If label is None, set to -1 to mark 'unknown' and review later
        label_out = label if label is not None else -1

        # write relative path so evaluate_model.py can find them (adjust if your evaluator expects different)
        relpath = os.path.join(str(html_dir), f.name)
        rows.append((relpath, label_out))

    # Save CSV (only include rows with known labels by default? we include all so you can review -1)
    with open(out_csv, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["file_path", "label"])
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to {out_csv}. Labels of -1 mean 'not found / needs human review'.")

if __name__ == "__main__":
    generate_csv()
