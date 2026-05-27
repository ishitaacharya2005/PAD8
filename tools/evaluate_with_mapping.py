#!/usr/bin/env python3
"""
Attempt to evaluate a saved model on a CSV by mapping CSV column names to the model's expected features.
Usage:
  python tools/evaluate_with_mapping.py --csv data/eval_labels_clean.csv --model models/model.pkl --out data/eval_report_mapped.json
"""
import argparse, os, json, sys, re
import pandas as pd, numpy as np, joblib
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

def norm_name(s):
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r'[^a-z0-9]+', '_', s)  # keep alphanum as underscores
    s = re.sub(r'_+', '_', s).strip('_')
    return s

# common synonyms map (human names -> model names)
SYNONYMS = {
    # mapping normalized forms to canonical model names (add as needed)
    norm_name("common femoral artery psv"): "Common Femoral Artery PSV",
    norm_name("common femoral psv"): "Common Femoral Artery PSV",
    norm_name("cf psv"): "Common Femoral Artery PSV",
    norm_name("profundus femoris psv"): "Profundus Femoris PSV",
    norm_name("mid sfa psv"): "Mid SFA PSV",
    norm_name("distal sfa psv"): "Distal SFA PSV",
    norm_name("popliteal artery psv"): "Popliteal Artery PSV",
    norm_name("peroneal psv"): "Peroneal / Posterior Tibial Artery PSV",
    norm_name("posterior tibial psv"): "Peroneal / Posterior Tibial Artery PSV",
    norm_name("anterior tibial psv"): "Anterior Tibial Artery PSV",
    norm_name("psv__common_femoral_artery"): "PSV__Common_Femoral_Artery",
    norm_name("psv__mid_sfa"): "PSV__Mid_SFA",
    norm_name("psv__distal_sfa"): "PSV__Distal_SFA",
    norm_name("abi"): "ABI",
    norm_name("age"): "Age",
    norm_name("waveform_triphasic_count"): "waveform_triphasic_count",
    norm_name("waveform_monophasic_count"): "waveform_monophasic_count",
    norm_name("waveform_biphasic_count"): "waveform_biphasic_count",
    norm_name("mean_psv"): "mean_PSV",
}

def discover_feature_names(model):
    # try multiple places
    try:
        if hasattr(model, "feature_names_in_"):
            return list(getattr(model, "feature_names_in_"))
    except Exception:
        pass
    try:
        if hasattr(model, "named_steps"):
            for name, step in model.named_steps.items():
                if hasattr(step, "feature_names_in_"):
                    return list(getattr(step, "feature_names_in_"))
    except Exception:
        pass
    # fallback scan
    for attr in ("feature_names_in_", "coef_", "n_features_in_"):
        if hasattr(model, attr):
            try:
                val = getattr(model, attr)
                if isinstance(val, (list, tuple, np.ndarray)):
                    return list(val)
            except Exception:
                pass
    return []

def auto_map_columns(model_features, csv_columns):
    # normalized dicts
    mod_norm_map = {norm_name(m): m for m in model_features}
    csv_norm_map = {norm_name(c): c for c in csv_columns}

    mapping = {}
    used_csv = set()

    # first exact normalized matches
    for mn_norm, mn in mod_norm_map.items():
        if mn_norm in csv_norm_map:
            mapping[mn] = csv_norm_map[mn_norm]
            used_csv.add(csv_norm_map[mn_norm])

    # then synonyms
    for syn_norm, target in SYNONYMS.items():
        if target in mapping:
            continue
        if syn_norm in csv_norm_map:
            mapping[target] = csv_norm_map[syn_norm]
            used_csv.add(csv_norm_map[syn_norm])

    # then try substring fuzzy match (model token in csv norm)
    for mn in model_features:
        if mn in mapping:
            continue
        mn_norm = norm_name(mn)
        for csv_norm, csv_orig in csv_norm_map.items():
            if csv_orig in used_csv:
                continue
            # if many tokens from mn_norm in csv_norm, accept
            if mn_norm in csv_norm or csv_norm in mn_norm:
                mapping[mn] = csv_orig
                used_csv.add(csv_orig)
                break
            # token overlap
            mn_tokens = set(mn_norm.split('_'))
            csv_tokens = set(csv_norm.split('_'))
            if len(mn_tokens & csv_tokens) >= max(1, min(2, len(mn_tokens)//2)):
                mapping[mn] = csv_orig
                used_csv.add(csv_orig)
                break

    return mapping

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print("CSV not found:", args.csv); sys.exit(2)
    if not os.path.exists(args.model):
        print("Model not found:", args.model); sys.exit(2)

    df = pd.read_csv(args.csv)
    print("Loaded CSV with columns:", list(df.columns)[:40])

    model = joblib.load(args.model)
    print("Loaded model:", args.model)
    model_features = discover_feature_names(model)
    if not model_features:
        print("Model did not expose feature names (feature_names_in_). Try training pipeline to save feature names.")
        # as fallback, use numeric columns from CSV (but warn)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print("Falling back to numeric columns in CSV:", numeric_cols[:10])
        X = df[numeric_cols].copy()
        X = X.fillna(X.median())
        preds = model.predict(X)
        df["eval_pred_label"] = preds
    else:
        print("Model expects features:", model_features[:20], "...")
        mapping = auto_map_columns(model_features, df.columns)
        missing = [m for m in model_features if m not in mapping]
        print("Auto-mapping produced", len(mapping), "mapped features. Missing features:", missing)
        if missing:
            # attempt to compute mean_PSV from PSV columns if mean_PSV is missing and many PSV present
            if "mean_PSV" in missing:
                psv_like = [c for c in df.columns if 'psv' in c.lower() or 'psv' in norm_name(c)]
                if psv_like:
                    df["mean_PSV"] = df[psv_like].apply(lambda r: pd.to_numeric(r, errors='coerce').median(), axis=1)
                    mapping["mean_PSV"] = "mean_PSV"
                    missing = [m for m in missing if m != "mean_PSV"]
                    print("Computed mean_PSV from detected PSV-like columns:", psv_like[:8])

        if missing:
            print("Cannot assemble ALL features automatically. Missing model features:", missing)
            print("You can either rename CSV columns or add those columns (even empty) so the script can fill with medians.")
            # still try with the mapped columns available
        # Build X with model_features order, using mapping where present, else fill with median (if numeric) or nan
        rows = {}
        for feat in model_features:
            if feat in mapping:
                rows[feat] = pd.to_numeric(df[mapping[feat]], errors='coerce')
            else:
                # try create a numeric column of NaNs then fill median later
                rows[feat] = pd.Series([np.nan]*len(df))

        X = pd.DataFrame(rows, columns=model_features)
        # fill numeric missing with median of available columns where possible
        for c in X.columns:
            if X[c].isna().all():
                # if original CSV had a column with similar name, leave; else fill with 0 or median  (choose median of other PSVs)
                # heuristic: if 'PSV' in c fill with median of any PSV-like columns
                if 'psv' in c.lower():
                    psv_cols = [col for col in X.columns if 'psv' in col.lower() and not X[col].isna().all()]
                    if psv_cols:
                        X[c] = X[c].fillna(X[psv_cols].median(axis=1))
                    else:
                        X[c] = X[c].fillna(0.0)
                elif c.lower() == 'abi':
                    # ABI default invalid -> set NaN so later code can interpret; fill with median of ABI-like cols if any
                    abi_cols = [col for col in X.columns if 'abi' in col.lower() and not X[col].isna().all()]
                    if abi_cols:
                        X[c] = X[c].fillna(X[abi_cols].median(axis=1))
                    else:
                        X[c] = X[c].fillna(np.nan)
                else:
                    X[c] = X[c].fillna(0.0)
        # final fill numeric column-wise median
        X = X.apply(lambda col: pd.to_numeric(col, errors='coerce'))
        X = X.fillna(X.median())

        try:
            preds = model.predict(X)
            df["eval_pred_label"] = preds
        except Exception as e:
            print("Model.predict failed:", e)
            sys.exit(2)

    # evaluation: look for true label column
    possible_true = [c for c in df.columns if c.lower() in ("true_label","label","true_sev","true_severity","gold_label","ground_truth")]
    true_col = possible_true[0] if possible_true else None
    if true_col:
        def normalize(x):
            try:
                if pd.isna(x): return None
                if isinstance(x, (int, np.integer)): return int(x)
                s = str(x).strip().lower()
                if s in ('none','0','normal','no','n'): return 0
                if s in ('mild','1','yes','y','true'): return 1
                if s.startswith('moder'): return 2
                if s.startswith('sever'): return 3
                m = re.search(r'([0-3])', s)
                if m: return int(m.group(1))
            except:
                return None
            return None
        y_true = df[true_col].apply(normalize).tolist()
        y_pred = df["eval_pred_label"].tolist()
        paired = [(t,p) for t,p in zip(y_true, y_pred) if t is not None and p is not None]
        if not paired:
            print("No rows with both true labels and predictions to evaluate.")
            out = {"n_rows": len(df), "mapped_features": mapping if 'mapping' in locals() else {}, "note": "no paired rows for eval"}
            with open(args.out, "w", encoding="utf-8") as f: json.dump(out, f, indent=2)
            print("Wrote JSON report to", args.out)
            return
        y_true_eval, y_pred_eval = zip(*paired)
        acc = accuracy_score(y_true_eval, y_pred_eval)
        bacc = balanced_accuracy_score(y_true_eval, y_pred_eval)
        cr = classification_report(y_true_eval, y_pred_eval, zero_division=0, output_dict=True)
        cm = confusion_matrix(y_true_eval, y_pred_eval).tolist()
        out = {"n_rows": len(df), "accuracy": float(acc), "balanced_accuracy": float(bacc), "classification_report": cr, "confusion_matrix": cm, "mapped_features": mapping}
    else:
        out = {"n_rows": len(df), "mapped_features": mapping if 'mapping' in locals() else {}, "note": "no true label column found for evaluation"}

    # write outputs
    pred_csv = os.path.splitext(args.csv)[0] + ".pred.csv"
    df.to_csv(pred_csv, index=False)
    out["pred_csv"] = pred_csv
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Wrote predictions CSV:", pred_csv)
    print("Wrote evaluation JSON:", args.out)

if __name__ == "__main__":
    main()
