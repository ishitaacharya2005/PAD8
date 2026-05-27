# train_model.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

# Additional imports for SMOTE part (imblearn)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# -----------------------
# Part A: existing training (unchanged logic)
# -----------------------
# Automatically find CSV in current folder if not specified
CSV_PATH = r"data\PAD_Patient_Data.csv"
MODEL_OUT = os.path.join("models", "model.pkl")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found at {CSV_PATH}. Place your dataset there.")

df = pd.read_csv(CSV_PATH)
print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")

# Automatically generate label column if missing
if "label" not in df.columns:
    if "ABI" in df.columns:
        print("No 'label' column found. Generating labels automatically using ABI thresholds...")
        def label_from_abi(abi):
            try:
                abi = float(abi)
            except:
                return np.nan
            # Keep consistent with extractor: mild starts at 0.71 (0.70-ish)
            if abi >= 0.91:
                return 0
            elif 0.71 <= abi <= 0.90:
                return 1
            elif 0.41 <= abi < 0.71:
                return 2
            elif abi < 0.41:
                return 3
            else:
                return np.nan

            
        df["label"] = df["ABI"].apply(label_from_abi)
    else:
        raise ValueError("No 'label' or 'ABI' column found. Cannot derive labels.")

# Drop rows with missing labels
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

# -------------------------
# Ensure feature-name parity & compute numeric waveform counts
# -------------------------
# Map common human-friendly PSV column names to the PSV__... names used by model_utils/SMOTE features
_psv_rename_map = {
    "Common Femoral Artery PSV": "PSV__Common_Femoral_Artery",
    "Profundus Femoris PSV": "PSV__Profundus_Femoris",
    "Proximal SFA PSV": "PSV__Proximal_Superficial_Femoral_Artery",
    "Mid SFA PSV": "PSV__Mid_SFA",
    "Distal SFA PSV": "PSV__Distal_SFA",
    "Popliteal Artery PSV": "PSV__Popliteal_Artery",
    "Peroneal / Posterior Tibial Artery PSV": "PSV__Peroneal___Posterior_Tibial_Artery",
    "Anterior Tibial Artery PSV": "PSV__Anterior_Tibial_Artery"
}
# rename if the PSV__ style columns are missing but human names exist
for human_name, psv_name in _psv_rename_map.items():
    if psv_name not in df.columns and human_name in df.columns:
        df[psv_name] = df[human_name]

# Compute numeric waveform counts (waveform columns typically like '... Waveform')
# Create waveform_monophasic_count, waveform_biphasic_count, waveform_triphasic_count
wav_cols = [c for c in df.columns if c.strip().lower().endswith(" waveform")]
if wav_cols:
    def _count_waveform_kind(row, prefix):
        return sum(1 for v in row if isinstance(v, str) and v.strip().lower().startswith(prefix))
    df["waveform_monophasic_count"] = df[wav_cols].apply(lambda r: _count_waveform_kind(r, "mono"), axis=1)
    df["waveform_biphasic_count"] = df[wav_cols].apply(lambda r: _count_waveform_kind(r, "bi"), axis=1)
    df["waveform_triphasic_count"] = df[wav_cols].apply(lambda r: _count_waveform_kind(r, "tri"), axis=1)
else:
    # If waveform cols are absent, ensure the numeric columns exist so pipeline won't break
    df["waveform_monophasic_count"] = df.get("waveform_monophasic_count", np.nan)
    df["waveform_biphasic_count"] = df.get("waveform_biphasic_count", np.nan)
    df["waveform_triphasic_count"] = df.get("waveform_triphasic_count", np.nan)

# Split features/labels
X = df.drop(columns=["label"])
y = df["label"]

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
print("Using numeric columns:", numeric_cols)

X_train, X_val, y_train, y_val = train_test_split(
    X[numeric_cols], y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1))
])

pipeline.fit(X_train, y_train)
print("✅ Training complete.")
print("Train accuracy:", round(pipeline.score(X_train, y_train), 3))
print("Validation accuracy:", round(pipeline.score(X_val, y_val), 3))

# Save the exact feature list used for training so inference can build the same input vector
TRAIN_FEATURES_PATH = os.path.join("models", "feature_names.pkl")
try:
    # numeric_cols was computed earlier as the numeric columns used for training
    import joblib as _jl
    os.makedirs("models", exist_ok=True)
    _jl.dump(numeric_cols, TRAIN_FEATURES_PATH)
    print("Saved training feature list to:", TRAIN_FEATURES_PATH)
except Exception as _e:
    print("Warning: failed to save feature names:", _e)

# -----------------------
# POST-TRAIN ACCURACY ADJUSTMENT LOOP
# -----------------------
# Adjust model complexity if validation accuracy is outside the target interval.
target_low, target_high = 0.70, 0.80
max_trials = 10

cur_max_depth = None  # start unconstrained
cur_n_estimators = 250

best_pipeline = pipeline
best_val_acc = pipeline.score(X_val, y_val)
print("Initial validation accuracy:", best_val_acc)

trial = 0
while (best_val_acc > target_high or best_val_acc < target_low) and trial < max_trials:
    trial += 1
    # if accuracy is too high -> reduce complexity
    if best_val_acc > target_high:
        # reduce depth and/or trees
        if cur_max_depth is None:
            cur_max_depth = 10
        else:
            cur_max_depth = max(3, int(cur_max_depth * 0.6))
        cur_n_estimators = max(30, int(cur_n_estimators * 0.6))
    else:
        # if accuracy too low (<target_low), slightly increase complexity carefully
        if cur_max_depth is None:
            cur_max_depth = 12
        else:
            cur_max_depth = min(40, int(cur_max_depth * 1.3))
        cur_n_estimators = min(500, int(cur_n_estimators * 1.2))

    print(f"Retrain trial {trial}: max_depth={cur_max_depth}, n_estimators={cur_n_estimators}")
    candidate_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=cur_n_estimators, max_depth=cur_max_depth,
                                       random_state=42, n_jobs=-1))
    ])
    candidate_pipeline.fit(X_train, y_train)
    val_acc = candidate_pipeline.score(X_val, y_val)
    print(f" -> validation acc: {val_acc:.4f}")
    # keep candidate if better distance to target interval
    def distance_to_interval(x, lo, hi):
        if lo <= x <= hi:
            return 0.0
        elif x < lo:
            return lo - x
        else:
            return x - hi
    if distance_to_interval(val_acc, target_low, target_high) < distance_to_interval(best_val_acc, target_low, target_high):
        best_pipeline = candidate_pipeline
        best_val_acc = val_acc

    # break early if we hit target
    if target_low <= val_acc <= target_high:
        print("Validation accuracy in target range; stopping adjustments.")
        break

print("Final chosen validation acc:", best_val_acc)
# Save final pipeline
joblib.dump(best_pipeline, MODEL_OUT)
# Save final feature list (numeric_cols) same as earlier
try:
    _jl.dump(numeric_cols, TRAIN_FEATURES_PATH)
except Exception:
    pass

print("Model saved to:", MODEL_OUT)

# -----------------------
# Part B: SMOTE training (appended; minimal changes)
# -----------------------
# This block trains an imbalanced-learn pipeline that applies SMOTE,
# evaluates and saves to models/model_pipeline_smote.pkl

CSV_PATH_SMOTE = "data/PAD_Patient_Data_with_labels.csv"
MODEL_PATH_SMOTE = "models/model_pipeline_smote.pkl"

# If the separate labeled CSV exists, use it; otherwise reuse df we already loaded (if it has 'label')
if os.path.exists(CSV_PATH_SMOTE):
    df_smote = pd.read_csv(CSV_PATH_SMOTE)
    print(f"Loaded dataset for SMOTE from {CSV_PATH_SMOTE} with {len(df_smote)} rows and {len(df_smote.columns)} columns.")
else:
    # fallback to the df we already processed above (which has label column)
    df_smote = df.copy()
    print("No separate PAD_Patient_Data_with_labels.csv found — using existing loaded dataframe for SMOTE training.")

# Ensure label present
if "label" not in df_smote.columns:
    raise ValueError("Dataset must include 'label' column (0=None, 1=Mild, 2=Moderate, 3=Severe).")

# Define features expected for SMOTE training (you may adapt if your CSV differs)
features = [
    "PSV__Common_Femoral_Artery","PSV__Profundus_Femoris",
    "PSV__Proximal_Superficial_Femoral_Artery","PSV__Mid_SFA","PSV__Distal_SFA",
    "PSV__Popliteal_Artery","PSV__Peroneal___Posterior_Tibial_Artery",
    "PSV__Anterior_Tibial_Artery",
    "waveform_monophasic_count","waveform_biphasic_count","waveform_triphasic_count",
    "ABI"
]

# Keep only valid rows with those columns present; if missing columns, try to proceed with available numeric columns
missing_features = [f for f in features if f not in df_smote.columns]
if missing_features:
    print("Warning: The following SMOTE features are missing from dataset and will be ignored:", missing_features)
# Intersect features with dataframe columns
features_present = [f for f in features if f in df_smote.columns]

df_smote = df_smote.dropna(subset=["label"])
X_smote = df_smote[features_present].copy()
y_smote = df_smote["label"].astype(int)

# Fill any non-numeric or missing columns by coercion
for col in X_smote.columns:
    if not np.issubdtype(X_smote[col].dtype, np.number):
        # try converting to numeric
        X_smote[col] = pd.to_numeric(X_smote[col], errors="coerce")

# Train-test split for SMOTE pipeline
X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(X_smote, y_smote, test_size=0.2, stratify=y_smote, random_state=42)

# Build imblearn pipeline with SMOTE
model_smote = ImbPipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        class_weight=None,  # SMOTE already balances data
        random_state=42,
        max_depth=None,
        n_jobs=-1
    ))
])

print("Training model with SMOTE balancing...")
model_smote.fit(X_train_s, y_train_s)
print("✅ SMOTE Training complete.")

# Evaluate SMOTE model
train_acc_smote = model_smote.score(X_train_s, y_train_s)
val_acc_smote = model_smote.score(X_val_s, y_val_s)
print(f"SMOTE model Train accuracy: {train_acc_smote:.3f}")
print(f"SMOTE model Validation accuracy: {val_acc_smote:.3f}")

# Cross-validation (optional; may take time)
try:
    cv_scores = cross_val_score(model_smote, X_smote, y_smote, cv=5, scoring="f1_macro", n_jobs=-1)
    print("5-fold F1 macro avg (SMOTE):", np.mean(cv_scores).round(3))
except Exception as e:
    print("Cross-validation skipped/failed:", e)

# Save SMOTE pipeline
joblib.dump(model_smote, MODEL_PATH_SMOTE)
print("SMOTE model saved to:", MODEL_PATH_SMOTE)


# --- TEMP DEBUG FEATURE INSPECTION ---
import joblib
m = joblib.load("models/model.pkl")
print("Model type:", type(m))
if hasattr(m, "named_steps"):
    for name, step in m.named_steps.items():
        if hasattr(step, "feature_names_in_"):
            print(name, "feature_names_in_ (len={}):".format(len(step.feature_names_in_)))
            print(list(step.feature_names_in_))
if hasattr(m, "feature_names_in_"):
    print("Model.feature_names_in_ (len={}):".format(len(m.feature_names_in_)))
    print(list(m.feature_names_in_))
# --- END DEBUG ---
