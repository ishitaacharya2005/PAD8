# perm_importance.py
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.inspection import permutation_importance

MODEL_PATH = os.path.join("models", "model.pkl")
CSV_PATH = "data/PAD_Patient_Data.csv"  # change if needed
OUT = "data/perm_importance.csv"

if not os.path.exists(MODEL_PATH):
    raise SystemExit("Model not found at models/model.pkl")
if not os.path.exists(CSV_PATH):
    raise SystemExit(f"CSV not found at {CSV_PATH}")

print("Loading model...")
model = joblib.load(MODEL_PATH)
model_fnames = getattr(model, "feature_names_in_", None)
if model_fnames is None:
    print("Model has no feature_names_in_. Will use CSV columns as-is.")
else:
    model_fnames = [str(x) for x in model_fnames]
    print("Model expects", len(model_fnames), "features.")

print("Loading CSV...")
df = pd.read_csv(CSV_PATH)
df = df[df["label"].notna()].copy()
y = df["label"].astype(int).values

# Build X_df aligned to model.feature_names_in_ if available; else use CSV numeric columns
if model_fnames is not None:
    # Create DataFrame with model's feature names as columns
    X_df = pd.DataFrame(index=df.index)
    for col in model_fnames:
        if col in df.columns:
            X_df[col] = df[col]
        else:
            # not found — put NaN (will impute below)
            X_df[col] = np.nan
else:
    # fallback: use numeric columns from CSV except label
    X_df = df.select_dtypes(include=[np.number]).drop(columns=["label"], errors="ignore")

# Impute medians for missing values to get a usable matrix for permutation importance
med = X_df.median()
X_df = X_df.fillna(med).fillna(0)
X_df = X_df.apply(pd.to_numeric, errors="coerce").astype(float)

print("X_df shape:", X_df.shape)

# Compute permutation importance (this may take time depending on data size and repeats)
print("Computing permutation importance (this may take a few minutes)...")
res = permutation_importance(model, X_df, y, n_repeats=10, random_state=0, n_jobs=-1)
imp_mean = res.importances_mean
imp_std = res.importances_std
feat_names = X_df.columns.tolist()

imp_df = pd.DataFrame({
    "feature": feat_names,
    "importance_mean": imp_mean,
    "importance_std": imp_std
}).sort_values("importance_mean", ascending=False)

print("\nTop 20 features by permutation importance:")
print(imp_df.head(20).to_string(index=False))

imp_df.to_csv(OUT, index=False)
print(f"\nSaved full permutation importances to {OUT}")
