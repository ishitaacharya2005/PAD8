# retrain_no_leak.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer

# Config
CSV = "data/PAD_Patient_Data.csv"   # use your original labelled CSV (or the training CSV you used)
OUT_MODEL = "models/model.pkl"
OUT_FEATURES = "models/feature_names.pkl"
RANDOM_STATE = 42

# Feature list used by evaluation (we keep same canonical set but will drop leak cols)
MODEL_FEATURES = [
    "PSV__Common_Femoral_Artery", "PSV__Profundus_Femoris",
    "PSV__Proximal_Superficial_Femoral_Artery", "PSV__Mid_SFA", "PSV__Distal_SFA",
    "PSV__Popliteal_Artery", "PSV__Peroneal___Posterior_Tibial_Artery",
    "PSV__Anterior_Tibial_Artery",
    "waveform_monophasic_count", "waveform_biphasic_count", "waveform_triphasic_count",
    "ABI"
]

# Columns we want to drop because they may leak or be identifiers
DROP_COLS = ["Patient ID", "mean_PSV"]

if not os.path.exists(CSV):
    raise SystemExit(f"CSV not found: {CSV}")

# Load CSV
df = pd.read_csv(CSV)
# Keep only rows with label
df = df[df["label"].notna()].copy()
df["label"] = df["label"].astype(int)

# We will build X using the exact same logic as evaluate: try to use canonical columns if present,
# else look for CSV named PSV columns. To keep this script simple, prefer these CSV column names:
CSV_TO_MODEL_PSV = {
    "Common Femoral Artery PSV": "PSV__Common_Femoral_Artery",
    "Profundus Femoris PSV": "PSV__Profundus_Femoris",
    "Proximal SFA PSV": "PSV__Proximal_Superficial_Femoral_Artery",
    "Mid SFA PSV": "PSV__Mid_SFA",
    "Distal SFA PSV": "PSV__Distal_SFA",
    "Popliteal Artery PSV": "PSV__Popliteal_Artery",
    "Peroneal / Posterior Tibial Artery PSV": "PSV__Peroneal___Posterior_Tibial_Artery",
    "Anterior Tibial Artery PSV": "PSV__Anterior_Tibial_Artery",
    # older variants included in evaluate_model if needed
}

# Build a DataFrame X with columns corresponding to MODEL_FEATURES in the same order:
def build_X_from_df(df):
    X = pd.DataFrame(index=df.index)
    # numeric PSV canonical keys: prefer explicit columns if present (PSV__...), else CSV names mapping
    for feat in MODEL_FEATURES:
        # If the CSV already contains the canonical feature name, use it
        if feat in df.columns:
            X[feat] = df[feat]
            continue
        # If this is a PSV__ key, try to find matching CSV column via reverse mapping
        found = False
        for csv_col, canonical in CSV_TO_MODEL_PSV.items():
            if canonical == feat and csv_col in df.columns:
                X[feat] = df[csv_col]
                found = True
                break
        if not found:
            # fallback to NaN (will be imputed)
            X[feat] = np.nan

    # waveform counts may already exist as columns in CSV; prefer that
    for w in ["waveform_monophasic_count", "waveform_biphasic_count", "waveform_triphasic_count"]:
        if w in df.columns:
            X[w] = df[w].astype(float)
        else:
            # attempt to derive minimally: set 0 (safe fallback)
            X[w] = 0.0

    # ABI
    if "ABI" in df.columns:
        X["ABI"] = df["ABI"].astype(float)
    else:
        X["ABI"] = np.nan

    return X

print("Building features from CSV...")
X = build_X_from_df(df)
y = df["label"].values

print("Initial X shape:", X.shape)
print("Columns sample:", X.columns.tolist())

# Drop suspicious/leakage columns if present (these are not part of MODEL_FEATURES here, but safe-guard)
for c in DROP_COLS:
    if c in X.columns:
        X = X.drop(columns=[c])

# Final feature list to train on
feature_names = X.columns.tolist()
print("Training using feature names (len={}):".format(len(feature_names)))
print(feature_names)

# Simple pipeline: median imputer -> standard scaler -> classifier
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)

pipeline = Pipeline([
    ("num", num_pipeline),
    ("clf", clf)
])

# Cross-validated scores
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
print("Running cross-validated training (5-fold stratified)...")
scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
print("CV accuracy scores:", scores)
print("Mean CV accuracy: {:.4f} +- {:.4f}".format(scores.mean(), scores.std()))

# Fit on full data
print("Fitting final model on full data...")
pipeline.fit(X, y)

# Save model and feature names (feature_names.pkl used by evaluate script)
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, OUT_MODEL)
joblib.dump(feature_names, OUT_FEATURES)
print("Saved model to", OUT_MODEL)
print("Saved feature list to", OUT_FEATURES)
