# tools/clean_eval_labels.py
import pandas as pd
from pathlib import Path
import sys

IN = Path(r"data/eval_labels.csv")
OUT = Path(r"data/eval_labels_clean.csv")

if not IN.exists():
    print(f"Input file {IN} not found. Make sure you ran generate_eval_labels_from_html.py first.")
    sys.exit(1)

df = pd.read_csv(IN)
print("Loaded", len(df), "rows from", IN)

# Basic validation: ensure 'label' column exists
if "label" not in df.columns:
    print("ERROR: 'label' column not found in CSV.")
    sys.exit(1)

# Drop unknown / placeholder (-1) labels
df_clean = df[df["label"] != -1].copy()
dropped = len(df) - len(df_clean)
print(f"Dropped {dropped} rows with label == -1 (unknown). {len(df_clean)} rows remain.")

# Ensure labels are integers
df_clean["label"] = pd.to_numeric(df_clean["label"], errors="coerce").astype("Int64")

# Remove any rows where conversion failed
bad = df_clean["label"].isna().sum()
if bad:
    print(f"Dropping {bad} rows where label could not be parsed as number.")
    df_clean = df_clean[df_clean["label"].notna()]

# Convert to plain int
df_clean["label"] = df_clean["label"].astype(int)

# Show label distribution
dist = df_clean["label"].value_counts().sort_index().to_dict()
print("Label distribution (label:count):", dist)

if len(dist) == 0:
    print("No labeled rows remain. You must provide clinician labels or ensure extractor wrote labels.")
    sys.exit(2)

if len(dist) == 1:
    lab = next(iter(dist.keys()))
    print(f"WARNING: Only one class present ({lab}) in cleaned CSV. sklearn metrics will be limited or undefined.")
    print("To get meaningful evaluation you need at least two classes in the CSV (e.g. both PAD and No-PAD).")
    # We still write the file so evaluate_model can run if you want to see limited output:
    df_clean.to_csv(OUT, index=False)
    print("Wrote", OUT)
    sys.exit(0)

# All good: write out cleaned CSV
df_clean.to_csv(OUT, index=False)
print("Wrote cleaned CSV to", OUT)
print("Now run:")
print(f"  python evaluate_model.py --csv {OUT} --model models/model.pkl --out eval_report.json")
