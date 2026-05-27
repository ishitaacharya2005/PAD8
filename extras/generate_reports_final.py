# generate_exact_reports_final.py
"""
Generates 100 reports preserving the exact HTML/CSS template provided by the user.
- Balanced classes: None / Mild / Moderate / Severe (25 each)
- Limb side balanced 50% Right / 50% Left
- Diverse Indian names, dates 2019-2024
- Varied referring doctor and signing doctor names (signing doctor always uses "Dr.")
- Outputs HTML in output/html/ (exact template) and PDFs in output/pdf/ (if converter installed)
- Writes output/manifest.csv
"""

import os
import csv
import random
import datetime
from string import Template
from typing import Dict, Any

# Import comparator and thresholds from your model_utils.py (must be in same folder)
from model_utils import comparative_predict, ARTERY_PSV_THRESHOLDS, load_model

# Try PDF backends
USE_WEASY = False
USE_PDFKIT = False
PDF_BACKEND = None
try:
    from weasyprint import HTML as WeasyHTML
    USE_WEASY = True
    PDF_BACKEND = "weasyprint"
except Exception:
    try:
        import pdfkit
        USE_PDFKIT = True
        PDF_BACKEND = "pdfkit"
    except Exception:
        PDF_BACKEND = None

# Config
TOTAL = 100
CLASS_LABELS = ["None", "Mild", "Moderate", "Severe"]
PER_CLASS = TOTAL // len(CLASS_LABELS)  # 25 each
OUTPUT_DIR = "output"
HTML_DIR = os.path.join(OUTPUT_DIR, "html")
PDF_DIR = os.path.join(OUTPUT_DIR, "pdf")
MANIFEST = os.path.join(OUTPUT_DIR, "manifest.csv")
MAX_ATTEMPTS = 120
SEED = 42

random.seed(SEED)

# Name pools and doctors
MALE_FIRST = ["Aarav","Arjun","Rohan","Karan","Vikram","Siddharth","Rahul","Amit","Ravi","Suresh",
              "Pranav","Manish","Naveen","Vikas","Anand","Deepak","Ramesh","Harish","Sanjay","Kishore",
              "Yash","Aditya","Ishaan","Kartik","Om"]
FEMALE_FIRST = ["Asha","Priya","Neha","Anjali","Sonal","Sunita","Ritika","Divya","Pooja","Lakshmi",
                "Kavya","Shreya","Nisha","Meera","Sadhana","Swati","Bhavana","Rashmi","Ila","Nandita",
                "Ananya","Ira","Sanya","Mira","Rhea"]
LAST_NAMES = ["Sharma","Patel","Kumar","Singh","Reddy","Gupta","Iyer","Nair","Das","Pillai",
              "Bose","Chatterjee","Menon","Rao","Ghosh","Saxena","Mukherjee","Malhotra","Trivedi","Verma",
              "Shah","Kapoor","Jain","Joshi","Bhatt"]

REFERRING_DOCS = [
    "Dr. S. Patel", "Dr. M. Rao", "Dr. P. Sharma", "Dr. R. Iyer", "Dr. D. Gupta",
    "Dr. N. Banerjee", "Dr. L. Menon", "Dr. V. Desai", "Dr. K. Bhat", "Dr. T. Singh",
    "Dr. H. Kumar", "Dr. R. Joshi", "Dr. A. Chatterjee", "Dr. P. Nair", "Dr. G. Sinha"
]
# Signing doctor last names; prefix will always be "Dr."
SIGN_LASTS = ["Mehta","Chopra","Bhattacharya","Kumar","Rao","Saxena","Jain","Shah","Kapoor","Joshi","Bhat","Singh","Banerjee","Iyer"]

# Artery keys to populate (matching user's table)
ART_KEYS = [
    "PSV__Common_Femoral_Artery",
    "PSV__Profundus_Femoris",
    "PSV__Proximal_Superficial_Femoral_Artery",
    "PSV__Mid_SFA",
    "PSV__Distal_SFA",
    "PSV__Popliteal_Artery",
    "PSV__Peroneal_Artery",
    "PSV__Posterior_Tibial_Artery",
    "PSV__Anterior_Tibial_Artery"
]

# HTML rows mapping (exact order / combined row for Peroneal/Posterior Tibial)
HTML_ROWS = [
    ("Common Femoral Artery", "PSV__Common_Femoral_Artery"),
    ("Profundus Femoris", "PSV__Profundus_Femoris"),
    ("Proximal Superficial Femoral Artery", "PSV__Proximal_Superficial_Femoral_Artery"),
    ("Mid SFA", "PSV__Mid_SFA"),
    ("Distal SFA", "PSV__Distal_SFA"),
    ("Popliteal Artery", "PSV__Popliteal_Artery"),
    ("Peroneal / Posterior Tibial Artery", ("PSV__Peroneal_Artery", "PSV__Posterior_Tibial_Artery")),
    ("Anterior Tibial Artery", "PSV__Anterior_Tibial_Artery")
]

# ---- Helpers ----
def pick_patient_name():
    if random.random() < 0.5:
        first = random.choice(MALE_FIRST)
    else:
        first = random.choice(FEMALE_FIRST)
    return f"{first} {random.choice(LAST_NAMES)}"

def pick_id():
    return str(random.randint(200000, 999999))

def pick_age_sex():
    return f"{random.randint(40,85)} Years / {random.choice(['Male','Female'])}"

def pick_exam_date():
    start = datetime.date(2019,1,1)
    end = datetime.date(2024,12,31)
    days = (end - start).days
    return (start + datetime.timedelta(days=random.randint(0, days))).strftime("%d-%m-%Y")

def pick_ref():
    return random.choice(REFERRING_DOCS)

def pick_sign():
    # ALWAYS prefix with "Dr." as requested
    init = random.choice(["","A.","K.","R.","S."])
    return f"Dr. {init} {random.choice(SIGN_LASTS)}".replace("  ", " ").strip()

# Balanced limb picker (ensures 50% Right / 50% Left across TOTAL)
def pick_limb_side_balanced(limb_counts: Dict[str,int], total_target: int) -> str:
    # target per side
    per_side = total_target // 2
    # if one side has already reached target, force the other
    if limb_counts["Right"] >= per_side and limb_counts["Left"] < per_side:
        return "Left"
    if limb_counts["Left"] >= per_side and limb_counts["Right"] < per_side:
        return "Right"
    # otherwise choose randomly (keeps balance)
    return random.choice(["Right", "Left"])

# PSV generator using artery thresholds if available
def psv_for_target(artery_key: str, target_score: int) -> float:
    best = None
    for k in ARTERY_PSV_THRESHOLDS.keys():
        if k == "generic": continue
        if k.lower() in artery_key.lower().replace(" ", "_"):
            best = k
            break
    if best is None:
        best = next(iter(ARTERY_PSV_THRESHOLDS))
    normal_min = ARTERY_PSV_THRESHOLDS.get(best, (90,))[0]
    if target_score == 0:
        low = normal_min
        high = normal_min + max(8, int(normal_min*0.25))
    elif target_score == 1:
        low = max(1, int(normal_min*0.78))
        high = max(low+3, int(normal_min)-1)
    elif target_score == 2:
        low = max(1, int(normal_min*0.45))
        high = max(low+3, int(normal_min*0.77))
    else:
        low = 5
        high = max(1, int(normal_min*0.44))
    return round(random.uniform(low, max(low+0.5, high)), 1)

def waveform_counts_for_label(label: str) -> Dict[str,int]:
    if label == "None":
        return {"waveform_triphasic_count": random.randint(2,5), "waveform_biphasic_count": random.randint(0,1), "waveform_monophasic_count": 0}
    if label == "Mild":
        return {"waveform_triphasic_count": random.randint(0,2), "waveform_biphasic_count": random.randint(2,5), "waveform_monophasic_count": 0}
    if label == "Moderate":
        return {"waveform_triphasic_count": random.randint(0,1), "waveform_biphasic_count": random.randint(0,2), "waveform_monophasic_count": random.randint(1,2)}
    # Severe
    return {"waveform_triphasic_count": 0, "waveform_biphasic_count": random.randint(0,1), "waveform_monophasic_count": random.randint(3,7)}

def abi_for_label(label: str):
    if label == "None":
        return round(random.uniform(0.90, 1.30), 2)
    if label == "Mild":
        return round(random.uniform(0.70, 0.89), 2)
    if label == "Moderate":
        return round(random.uniform(0.40, 0.69), 2)
    if label == "Severe":
        if random.random() < 0.85:
            return round(random.uniform(0.10, 0.39), 2)
        else:
            return round(random.uniform(0.90, 1.30), 2)
    return None

def generate_parsed(label: str) -> Dict[str,Any]:
    parsed = {}
    parsed["ABI"] = abi_for_label(label)
    parsed.update(waveform_counts_for_label(label))
    for k in ART_KEYS:
        if random.random() < 0.85:
            t = {"None":0,"Mild":1,"Moderate":2,"Severe":3}[label]
            parsed[k] = psv_for_target(k, t)
        else:
            parsed[k] = psv_for_target(k, random.choice([0,1,2,3]))
    if label == "Severe":
        if random.random() < 0.6:
            parsed["Percent_Stenosis"] = random.randint(75, 95)
        if random.random() < 0.4:
            parsed["PSVR"] = round(random.uniform(4.0,8.0),2)
    if label in ("Moderate","Severe") and random.random() < 0.2:
        parsed["Critical_Remark"] = True
    return parsed

# Remark generation per artery depending on PSV & waveform
def remark_for_value(psv, mono_count, bi_count, tri_count):
    texts = []
    p = None
    try:
        if psv is not None:
            p = float(psv)
    except Exception:
        p = None
    if p is not None:
        if p < 40:
            texts.append("Diminished flow / poor distal perfusion")
        elif p < 70:
            texts.append("Reduced amplitude consistent with downstream disease")
        elif p < 100:
            texts.append(random.choice([
                "Mild atherosclerotic wall thickening",
                "Mild disease / waveform damping",
                "Segmental irregularity, correlate clinically"
            ]))
        else:
            texts.append(random.choice([
                "Normal flow",
                "No hemodynamic obstruction visualized",
                "Flow within normal limits"
            ]))
    else:
        texts.append("Poor visualization / flow not well assessed")
    # waveform modifiers
    if mono_count >= 3:
        texts.append("Monophasic waveform — suggests proximal significant obstruction")
    elif tri_count >= 2:
        texts.append("Triphasic waveform preserved")
    elif bi_count >= 2:
        texts.append("Biphasic waveform — mild change")
    # combine
    if len(texts) == 1:
        return texts[0]
    return " ; ".join(texts[:2])

# Impression builder (descriptive, not giving a direct label)
def build_impression(parsed: Dict[str,Any]) -> str:
    lines = []
    abi = parsed.get("ABI")
    mono = parsed.get("waveform_monophasic_count", 0)
    bi = parsed.get("waveform_biphasic_count", 0)
    tri = parsed.get("waveform_triphasic_count", 0)

    try:
        if abi is None:
            lines.append("ABI not available.")
        else:
            a = float(abi)
            if a > 1.30:
                lines.append("ABI > 1.30 (non-compressible) — ABI may be unreliable due to vessel calcification.")
            elif a >= 0.90:
                lines.append("ABI within normal range.")
            elif 0.70 <= a < 0.90:
                lines.append("ABI reduced, suggests peripheral arterial disease and warrants correlation with imaging findings.")
            elif 0.40 <= a < 0.70:
                lines.append("ABI moderately reduced, consistent with impaired perfusion.")
            else:
                lines.append("ABI markedly reduced, consistent with severe perfusion deficit.")
    except Exception:
        pass

    critical = []
    focal_msgs = []
    for k in ART_KEYS:
        v = parsed.get(k)
        try:
            pv = float(v)
        except Exception:
            pv = None
        if pv is not None:
            if pv < 40:
                critical.append((k, pv))
            elif pv < 70:
                focal_msgs.append((k, pv))
    if critical:
        names = ", ".join([c[0].replace("PSV__","").replace("_"," ") for c in critical])
        lines.append(f"Markedly reduced PSV in {names} suggesting severe flow-limiting disease at or proximal to the level(s) described.")
    if focal_msgs and not critical:
        names = ", ".join([f"{c[0].replace('PSV__','').replace('_',' ')} ({c[1]} cm/s)" for c in focal_msgs])
        lines.append(f"Reduced PSV in {names} indicating segmental disease; distal waveforms should be correlated.")
    if mono >= 3:
        lines.append("Monophasic waveforms are prominent distally, consistent with reduced distal perfusion.")
    elif tri >= 2:
        lines.append("Triphasic waveforms are largely preserved indicating good distal arterial compliance.")
    elif bi >= 2:
        lines.append("Biphasic waveforms present in several segments, may indicate early/moderate vascular changes.")
    if parsed.get("Percent_Stenosis") is not None:
        lines.append(f"Reported percent stenosis: {parsed.get('Percent_Stenosis')}% — consider vascular correlation.")
    if parsed.get("PSVR") is not None:
        lines.append(f"Measured PSVR = {parsed.get('PSVR')}; elevated values may indicate hemodynamically significant stenosis.")
    # final suggestion line
    if any([parsed.get("Percent_Stenosis",0) >= 75, parsed.get("PSVR",0) >= 4, len(critical) >= 1]):
        lines.append("Overall findings suggest hemodynamically significant focal disease — recommend vascular surgical/radiological review and correlation with clinical findings.")
    elif "ABI within normal range." in " ".join(lines) and not critical:
        lines.append("No hemodynamically significant arterial obstruction identified on this study; correlate with symptoms.")
    else:
        lines.append("Correlation with clinical picture and further imaging (if indicated) is recommended.")
    return " ".join(lines)

# EXACT HTML template (unchanged structure/CSS) with limb placeholder $limb_side
HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Color Doppler / Duplex Ultrasound Report — $limb_side Lower Limb</title>
  <style>
    :root{
      --bg: #ffffff;
      --muted: #6b7280;
      --accent: #0f172a;
      --card: #f8fafc;
      --border: #e6e9ee;
      --table-header: #f1f5f9;
      --shadow: 0 6px 18px rgba(15,23,42,0.06);
      font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
      color-scheme: light;
    }

    html,body{height:100%}
    body{
      margin:0;
      background:linear-gradient(180deg,#f7fafc 0%, #fff 100%);
      display:flex;
      align-items:center;
      justify-content:center;
      padding:28px;
      color:var(--accent);
    }

    .report {
      width:100%;
      max-width:900px;
      background:var(--bg);
      border-radius:12px;
      box-shadow:var(--shadow);
      border:1px solid var(--border);
      padding:28px;
      box-sizing:border-box;
    }

    header{
      text-align:center;
      margin-bottom:18px;
    }

    .facility-name{
      font-size:22px;
      font-weight:800;
      letter-spacing:0.6px;
    }
    .facility-sub{
      margin-top:6px;
      font-size:13px;
      color:var(--muted);
    }

    h1.report-title{
      font-size:18px;
      margin:18px 0 4px 0;
      text-align:center;
      letter-spacing:0.4px;
    }

    .meta {
      display:flex;
      gap:20px;
      flex-wrap:wrap;
      align-items:flex-start;
      justify-content:space-between;
      margin:14px 0 20px 0;
    }

    .patient, .exam {
      flex:1 1 320px;
      min-width:260px;
      background:var(--card);
      padding:14px;
      border-radius:8px;
      border:1px solid var(--border);
    }

    .field {
      display:flex;
      margin:6px 0;
      gap:10px;
    }

    .field .label {
      width:140px;
      color:var(--muted);
      font-size:13px;
      font-weight:600;
    }

    .field .value {
      flex:1;
      font-size:14px;
      font-weight:600;
    }

    /* Technique / Impression */
    .section {
      margin-top:18px;
    }

    .section h2 {
      font-size:15px;
      margin:8px 0 10px 0;
      border-left:4px solid #e2e8f0;
      padding-left:10px;
      color:var(--accent);
    }

    .technique {
      background:#fff;
      padding:12px;
      border-radius:8px;
      border:1px dashed var(--border);
      color:#111827;
      line-height:1.45;
      font-size:14px;
    }

    /* Findings table */
    .findings {
      margin-top:12px;
      overflow-x:auto;
      border-radius:8px;
      border:1px solid var(--border);
      background:var(--bg);
    }

    table {
      border-collapse:collapse;
      width:100%;
      min-width:720px;
    }

    thead th {
      text-align:left;
      padding:12px;
      font-size:13px;
      background:var(--table-header);
      border-bottom:1px solid var(--border);
      font-weight:700;
      color:var(--accent);
    }

    tbody td {
      padding:12px;
      border-bottom:1px solid #f1f5f9;
      font-size:14px;
      vertical-align:top;
      color:#0b1220;
    }

    tbody tr:last-child td { border-bottom:0; }

    .psv { width:120px; font-weight:700; }
    .wave { width:160px; }

    /* ABI / Impression layout */
    .two-col {
      display:flex;
      gap:18px;
      margin-top:16px;
      flex-wrap:wrap;
    }

    .card {
      flex:1 1 320px;
      padding:14px;
      border-radius:8px;
      border:1px solid var(--border);
      background:var(--card);
    }

    .impression ol {
      margin:8px 0 0 18px;
      color:#111827;
    }

    .signature {
      margin-top:18px;
      text-align:right;
      color:var(--muted);
      font-size:14px;
    }

    /* Print optimizations */
    @media print{
      body{background:transparent;padding:0}
      .report{box-shadow:none;border:0;padding:12px;max-width:100%}
      header .facility-sub { color:#000; }
    }
  </style>
</head>
<body>
  <article class="report" role="document" aria-label="Color Doppler Duplex Ultrasound Report">
    <header>
      <div class="facility-name">MEDIVIEW IMAGING &amp; DIAGNOSTICS</div>
      <div class="facility-sub">Accredited Vascular Ultrasound &amp; Radiology Centre <br> #24, Residency Road, Bengaluru — 560025 • Tel: 31-48854721 • reports@mediviewdiagnostics.in</div>
      <h1 class="report-title">COLOR DOPPLER / DUPLEX ULTRASOUND REPORT<br><small style="font-weight:600;color:var(--muted)">$limb_side Lower Limb Arterial Study</small></h1>
    </header>

    <section class="meta" aria-labelledby="patient-exam">
      <div class="patient" id="patient-info" role="group" aria-label="Patient details">
        <div style="font-size:13px;color:var(--muted);font-weight:700;margin-bottom:6px">Patient Details</div>

        <div class="field">
          <div class="label">Patient Name</div>
          <div class="value">$patient_name</div>
        </div>

        <div class="field">
          <div class="label">Age / Sex</div>
          <div class="value">$age_sex</div>
        </div>

        <div class="field">
          <div class="label">Patient ID</div>
          <div class="value">$patient_id</div>
        </div>
      </div>

      <div class="exam" id="exam-info" role="group" aria-label="Referral and exam">
        <div style="font-size:13px;color:var(--muted);font-weight:700;margin-bottom:6px">Referral / Exam</div>

        <div class="field">
          <div class="label">Referring Doctor</div>
          <div class="value">$referring_doctor</div>
        </div>

        <div class="field">
          <div class="label">Examination Date</div>
          <div class="value">$exam_date</div>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>Technique</h2>
      <div class="technique">
        Duplex evaluation of $limb_side lower limb arteries was performed using B-mode, Color and Spectral Doppler imaging. Peak systolic velocities (PSV) and waveform patterns were analyzed in inguinal arterial segments — Common Femoral, Superficial Femoral (SFA), Peroneal and Tibial arteries.
      </div>
    </section>

    <section class="section" aria-labelledby="findings-heading">
      <h2 id="findings-heading">Findings — $limb_side Lower Limb</h2>

      <div class="findings" role="table" aria-label="Findings table">
        <table>
          <thead>
            <tr>
              <th scope="col">Artery</th>
              <th scope="col" class="psv">PSV (cm/s)</th>
              <th scope="col" class="wave">Waveform</th>
              <th scope="col">Remarks</th>
            </tr>
          </thead>
          <tbody>
$table_body
          </tbody>
        </table>
      </div>
    </section>

        <div style="font-weight:700;margin-bottom:6px">Ankle—Brachial Index (ABI)</div>
        <div style="font-size:16px"> ABI: $ABI</div>

        <div style="font-weight:700;margin-bottom:6px">Impression</div>
        <div class="impression">
          <p>$impression</p>
        </div>

    <div class="signature" aria-label="Author signature">
      <div style="font-weight:700">$signing_doctor</div>
      <div style="font-size:12px;color:var(--muted)"> Consultant Vascular Radiologist</div>
    </div>

  </article>
</body>
</html>
'''

# Build table body exactly as user's structure but with varied remarks
def make_table_body(parsed: Dict[str,Any]) -> str:
    rows = []
    mono = parsed.get("waveform_monophasic_count", 0)
    bi = parsed.get("waveform_biphasic_count", 0)
    tri = parsed.get("waveform_triphasic_count", 0)
    for display_name, key in HTML_ROWS:
        if isinstance(key, tuple):
            v1 = parsed.get(key[0], "")
            v2 = parsed.get(key[1], "")
            if v1 != "" and v2 != "":
                display_val = f"{v1} / {v2}"
                try:
                    p = min(float(v1), float(v2))
                except Exception:
                    p = None
            elif v1 != "":
                display_val = f"{v1}"
                try:
                    p = float(v1)
                except Exception:
                    p = None
            elif v2 != "":
                display_val = f"{v2}"
                try:
                    p = float(v2)
                except Exception:
                    p = None
            else:
                display_val = ""
                p = None
        else:
            display_val = parsed.get(key, "")
            try:
                p = float(display_val)
            except Exception:
                p = None
        # waveform string for the row (use global counts)
        if mono >= 3:
            wave = "Monophasic"
        elif tri >= 2:
            wave = "Triphasic"
        elif bi >= 2:
            wave = "Biphasic"
        elif mono >= 1:
            wave = "Monophasic"
        else:
            wave = random.choice(["Triphasic","Biphasic","Monophasic"])
        remark = remark_for_value(p, mono, bi, tri)
        rows.append("            <tr>\n              <td>{}</td>\n              <td class=\"psv\">{}</td>\n              <td class=\"wave\">{}</td>\n              <td>{}</td>\n            </tr>".format(display_name, display_val, wave, remark))
    return "\n".join(rows)

def html_to_pdf(html_path: str, pdf_path: str) -> bool:
    if USE_WEASY:
        try:
            WeasyHTML(filename=html_path).write_pdf(pdf_path)
            return True
        except Exception as e:
            print("WeasyPrint error:", e)
            return False
    if USE_PDFKIT:
        try:
            import pdfkit
            pdfkit.from_file(html_path, pdf_path)
            return True
        except Exception as e:
            print("pdfkit error:", e)
            return False
    return False

# ---- Main generation ----
def generate():
    os.makedirs(HTML_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)
    # manifest header (include limb_side)
    fields = ["html_path","pdf_path","limb_side","patient_name","patient_id","age_sex","exam_date","referring_doctor","signing_doctor","predicted_severity","score","ABI","Percent_Stenosis","PSVR","wave_mono","wave_bi","wave_tri"]
    with open(MANIFEST, "w", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(mf, fieldnames=fields)
        writer.writeheader()

    counts = {lbl:0 for lbl in CLASS_LABELS}
    limb_counts = {"Right": 0, "Left": 0}
    idx = 1

    for label in CLASS_LABELS:
        attempts = 0
        while counts[label] < PER_CLASS:
            attempts += 1
            if attempts > MAX_ATTEMPTS:
                print(f"[WARN] max attempts for {label} reached; moving on.")
                break
            parsed = generate_parsed(label)
            pred = comparative_predict(parsed)
            pred_label = pred.get("severity","Unknown")
            if pred_label != label:
                if label == "Severe":
                    parsed.setdefault("Percent_Stenosis", random.randint(75,95))
                    if "PSVR" not in parsed and random.random() < 0.6:
                        parsed["PSVR"] = round(random.uniform(4.0,6.0),2)
                    pred = comparative_predict(parsed)
                    pred_label = pred.get("severity","Unknown")
                if pred_label != label:
                    continue
            # accepted
            limb_side = pick_limb_side_balanced(limb_counts, TOTAL)
            limb_counts[limb_side] += 1

            patient_name = pick_patient_name()
            patient_id = pick_id()
            age_sex = pick_age_sex()
            exam_date = pick_exam_date()
            referring_doc = pick_ref()
            signing_doc = pick_sign()
            table_body = make_table_body(parsed)
            impression = build_impression(parsed)
            # Use Template substitution to avoid conflicts with CSS braces
            t = Template(HTML_TEMPLATE)
            html_content = t.substitute(
                limb_side=limb_side,
                patient_name=patient_name,
                age_sex=age_sex,
                patient_id=patient_id,
                referring_doctor=referring_doc,
                exam_date=exam_date,
                table_body=table_body,
                ABI=parsed.get("ABI",""),
                signing_doctor=signing_doc,
                impression=impression
            )
            fname = f"report_{idx:03d}.html"
            html_path = os.path.join(HTML_DIR, fname)
            with open(html_path, "w", encoding="utf-8") as fh:
                fh.write(html_content)
            pdf_name = f"report_{idx:03d}.pdf"
            pdf_path = os.path.join(PDF_DIR, pdf_name)
            pdf_ok = html_to_pdf(html_path, pdf_path)
            if not pdf_ok and os.path.exists(pdf_path):
                try:
                    os.remove(pdf_path)
                except Exception:
                    pass
            # manifest
            with open(MANIFEST, "a", newline="", encoding="utf-8") as mf:
                writer = csv.DictWriter(mf, fieldnames=fields)
                writer.writerow({
                    "html_path": os.path.abspath(html_path),
                    "pdf_path": os.path.abspath(pdf_path) if pdf_ok else "",
                    "limb_side": limb_side,
                    "patient_name": patient_name,
                    "patient_id": patient_id,
                    "age_sex": age_sex,
                    "exam_date": exam_date,
                    "referring_doctor": referring_doc,
                    "signing_doctor": signing_doc,
                    "predicted_severity": pred.get("severity"),
                    "score": pred.get("score"),
                    "ABI": parsed.get("ABI"),
                    "Percent_Stenosis": parsed.get("Percent_Stenosis"),
                    "PSVR": parsed.get("PSVR"),
                    "wave_mono": parsed.get("waveform_monophasic_count"),
                    "wave_bi": parsed.get("waveform_biphasic_count"),
                    "wave_tri": parsed.get("waveform_triphasic_count")
                })
            counts[label] += 1
            idx += 1

    print("Done. Counts:", counts)
    print("Limb distribution:", limb_counts)
    print("HTML saved to:", os.path.abspath(HTML_DIR))
    if PDF_BACKEND:
        print("PDFs (where conversion succeeded) in:", os.path.abspath(PDF_DIR))
        print("PDF backend:", PDF_BACKEND)
    else:
        print("No PDF backend installed. Install WeasyPrint or pdfkit+wkhtmltopdf to generate PDFs.")

if __name__ == "__main__":
    generate()
