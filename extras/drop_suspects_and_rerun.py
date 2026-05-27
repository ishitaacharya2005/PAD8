# drop_suspects_and_rerun.py
import pandas as pd
import os

IN = "data/PAD_Patient_Data.csv"
OUT = "data/PAD_Patient_Data_noid.csv"
DROP = ["Patient ID", "mean_PSV"]  # suspects to drop

if not os.path.exists(IN):
    raise SystemExit(f"Input CSV not found: {IN}")

df = pd.read_csv(IN)
print("Original columns:", df.columns.tolist()[:50])
for c in DROP:
    if c in df.columns:
        print(f"Dropping column: {c}")
        df = df.drop(columns=[c])
    else:
        print(f"Column not present, skipping: {c}")

df.to_csv(OUT, index=False)
print(f"Wrote file without suspects to {OUT}")


