#!/usr/bin/env python3
"""
Evaluate model on data/PAD_Test_Data.csv

Usage:
  python tools/evaluate_pad_test_data.py \
      --csv data/PAD_Test_Data.csv \
      --model models/model.pkl \
      --out data/pad_test_eval_report.json \
      --pred data/pad_test_predictions.csv

This script:
 - loads model (joblib) or uses model_utils wrapper if available
 - loads expected feature list from models/feature_names.pkl (or model.feature_names_in_)
 - reads the test CSV and maps/fills features (median fill)
 - predicts labels and probabilities
 - computes multi-class metrics and binary (PAD/no PAD) metrics
 - writes outputs (JSON report + predictions CSV) and prints a summary
"""
import argparse
import json
import os
import sys
import math
import importlib.util
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

def try_load_module(name, path_hint=None):
    """Try normal import, otherwise attempt to load from file path."""
    try:
        return importlib.import_module(name)
    except Exception:
        if path_hint and os.path.exists(path_hint):
            spec = importlib.util.spec_from_file_location(name, path_hint)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None

def load_joblib_model(path):
    try:
        import joblib
        m = joblib.load(path)
        return m
    except Exception as e:
        raise RuntimeError(f"Failed to load model via joblib: {e}")

def load_feature_list(path):
    if not os.path.exists(path):
        return None
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        try:
            import pickle
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

def numeric_fill_median(df, cols):
    df_local = df.copy()
    for c in cols:
        if c not in df_local.columns:
            df_local[c] = np.nan
        df_local[c] = pd.to_numeric(df_local[c], errors="coerce")
        med = df_local[c].median(skipna=True)
        if math.isnan(med):
            med = 0.0
        df_local[c] = df_local[c].fillna(med)
    return df_local

def ensure_feature_name_mapping(test_df, model_features):
    """
    If test_df contains alternative names (PSV__ style vs human friendly),
    attempt simple mappings: Common Femoral Artery PSV <-> PSV__Common_Femoral_Artery, etc.
    """
    map_pairs = {
        "Common Femoral Artery PSV": "PSV__Common_Femoral_Artery",
        "Profundus Femoris PSV": "PSV__Profundus_Femoris",
        "Proximal SFA PSV": "PSV__Proximal_Superficial_Femoral_Artery",
        "Mid SFA PSV": "PSV__Mid_SFA",
        "Distal SFA PSV": "PSV__Distal_SFA",
        "Popliteal Artery PSV": "PSV__Popliteal_Artery",
        "Peroneal / Posterior Tibial Artery PSV": "PSV__Peroneal___Posterior_Tibial_Artery",
        "Anterior Tibial Artery PSV": "PSV__Anterior_Tibial_Artery",
    }
    df = test_df
    for human, psv in map_pairs.items():
        if human in df.columns and psv not in df.columns:
            df[psv] = df[human]
        if psv in df.columns and human not in df.columns:
            df[human] = df[psv]
    return df

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Test CSV (e.g. data/PAD_Test_Data.csv)")
    p.add_argument("--model", required=True, help="Path to trained model (joblib .pkl)")
    p.add_argument("--out", required=True, help="Output JSON report path")
    p.add_argument("--pred", required=True, help="Output predictions CSV path")
    return p.parse_args()

def main():
    args = parse_args()
    csv_path = args.csv
    model_path = args.model
    out_json = args.out
    pred_csv = args.pred

    if not os.path.exists(csv_path):
        print("Test CSV not found:", csv_path)
        sys.exit(1)
    if not os.path.exists(model_path):
        print("Model not found:", model_path)
        sys.exit(1)

    df_test = pd.read_csv(csv_path)
    print(f"Loaded test CSV with columns: {list(df_test.columns)} (rows={len(df_test)})")

    # Try load model_utils (optional)
    model_utils = try_load_module("model_utils", "model_utils.py")

    # Load model
    model = None
    try:
        model = load_joblib_model(model_path)
        print("Loaded model via joblib:", model_path)
    except Exception as e:
        print("Joblib load failed:", e)
        if model_utils and hasattr(model_utils, "load_model"):
            try:
                model = model_utils.load_model(model_path)
                print("Loaded model via model_utils.load_model")
            except Exception as e2:
                print("model_utils.load_model failed:", e2)
                model = None

    # Load feature_names if present
    feat_path = os.path.join("models", "feature_names.pkl")
    model_features = load_feature_list(feat_path)
    if model_features:
        print("Loaded feature list from", feat_path)
    else:
        # try introspect sklearn pipeline attribute
        if model is not None:
            fn = getattr(model, "feature_names_in_", None)
            if fn is not None:
                model_features = list(fn)
                print("Using model.feature_names_in_ (len=%d)" % len(model_features))
    if model_features is None:
        # fallback common list
        model_features = [
            "Patient ID", "Age", "ABI",
            "Common Femoral Artery PSV", "Profundus Femoris PSV", "Proximal SFA PSV",
            "Mid SFA PSV", "Distal SFA PSV", "Popliteal Artery PSV",
            "Peroneal / Posterior Tibial Artery PSV", "Anterior Tibial Artery PSV",
            "waveform_monophasic_count", "waveform_biphasic_count", "waveform_triphasic_count",
            "mean_PSV",
            "PSV__Common_Femoral_Artery", "PSV__Profundus_Femoris", "PSV__Proximal_Superficial_Femoral_Artery",
            "PSV__Mid_SFA", "PSV__Distal_SFA", "PSV__Popliteal_Artery",
            "PSV__Peroneal___Posterior_Tibial_Artery", "PSV__Anterior_Tibial_Artery"
        ]
        print("No saved feature list; using fallback feature list (len=%d)" % len(model_features))

    # If test CSV is the "PAD_Test_Data.csv" it's likely to already contain feature columns.
    # Try to adapt by making sure both human and PSV__ names exist where possible.
    df_test = ensure_feature_name_mapping(df_test, model_features)

    # Ensure model_features present and numeric-fill medians
    df_filled = numeric_fill_median(df_test, model_features)

    # Prepare X in model_features order
    X = df_filled[model_features].copy()

    # Predict
    preds = None
    pred_probs = None
    pred_labels = []

    # If model_utils has format_features_for_model/predict_with_model and model available, prefer wrapper
    if model_utils and hasattr(model_utils, "predict_with_model") and model is not None:
        pred_labels = []
        pred_probs = []
        for idx, row in df_test.iterrows():
            parsed_row = row.to_dict()
            try:
                res = model_utils.predict_with_model(model, parsed_row)
                lab = None
                proba = None
                if isinstance(res, dict):
                    sev = res.get("severity")
                    # map severity string -> label
                    sev_map = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
                    if isinstance(sev, str) and sev in sev_map:
                        lab = sev_map[sev]
                    elif isinstance(res.get("pad_detected"), (int, float)) and (res.get("pad_detected") in (0,1)):
                        lab = int(res.get("pad_detected"))
                    elif isinstance(res.get("pad_detected"), bool):
                        lab = 1 if res.get("pad_detected") else 0
                    else:
                        lab = None
                    proba = res.get("proba")
                else:
                    lab = int(res)
                pred_labels.append(lab)
                pred_probs.append(proba)
            except Exception as e:
                pred_labels.append(None)
                pred_probs.append(None)
    else:
        # direct sklearn predict
        if model is None:
            print("No model available. Exiting.")
            sys.exit(1)
        try:
            # ensure X has same number of columns as model expects (sklearn pipelines do strict matching)
            # If model has feature_names_in_, scikit will check names; otherwise it will accept array.
            y_pred = model.predict(X)
            pred_labels = [int(x) for x in y_pred]
            try:
                proba = model.predict_proba(X)
                pred_probs = [list(p) for p in proba]
            except Exception:
                pred_probs = [None] * len(pred_labels)
            print("Predicted labels for", len(pred_labels), "rows using model.predict")
        except Exception as e:
            print("Model.predict failed:", e)
            # try using only numeric numpy array fallback
            try:
                X_arr = X.values
                y_pred = model.predict(X_arr)
                pred_labels = [int(x) for x in y_pred]
                pred_probs = [None] * len(pred_labels)
            except Exception as e2:
                print("Fallback predict failed:", e2)
                sys.exit(1)

    # Attach predictions to dataframe
    df_out = df_test.copy()
    df_out["pred_label"] = pred_labels
    df_out["pred_proba"] = pred_probs
    # True label column: try to find common names: label or true_label or severity_label
    true_col = None
    for cand in ["label", "true_label", "severity_label", "label_true"]:
        if cand in df_out.columns:
            true_col = cand
            break
    if true_col is None:
        raise RuntimeError("Test CSV doesn't contain an expected true label column. Expected one of: label, true_label, severity_label")

    df_out["true_label"] = pd.to_numeric(df_out[true_col], errors="coerce").apply(lambda v: int(v) if (not pd.isna(v)) else None)

    # Build binary mapping: 0 -> 0 (no PAD), 1/2/3 -> 1 (PAD present)
    def to_bin(lbl):
        if lbl is None or (isinstance(lbl, float) and math.isnan(lbl)):
            return None
        try:
            li = int(lbl)
        except Exception:
            return None
        return 0 if li == 0 else 1

    df_out["true_bin"] = df_out["true_label"].apply(to_bin)
    df_out["pred_bin"] = df_out["pred_label"].apply(to_bin)

    # Compute multi-class metrics (only on rows where true_label is present and pred_label present)
    df_multi = df_out.dropna(subset=["true_label"])
    metrics = {}
    if df_multi.shape[0] == 0:
        print("No labeled rows found in test CSV.")
        sys.exit(1)

    y_true_multi = df_multi["true_label"].astype(int).values
    y_pred_multi = df_multi["pred_label"].fillna(-999).astype(int).values

    # Multi-class accuracy / precision/recall/f1 (macro + weighted)
    metrics["multi_accuracy"] = float(accuracy_score(y_true_multi, y_pred_multi))
    metrics["multi_precision_macro"] = float(precision_score(y_true_multi, y_pred_multi, average="macro", zero_division=0))
    metrics["multi_recall_macro"] = float(recall_score(y_true_multi, y_pred_multi, average="macro", zero_division=0))
    metrics["multi_f1_macro"] = float(f1_score(y_true_multi, y_pred_multi, average="macro", zero_division=0))
    metrics["classification_report"] = classification_report(y_true_multi, y_pred_multi, zero_division=0, output_dict=True)
    metrics["confusion_matrix"] = confusion_matrix(y_true_multi, y_pred_multi).tolist()

    # Compute binary metrics (PAD vs none) on rows where true_bin is not null AND pred_bin not null
    df_bin = df_out.dropna(subset=["true_bin"])
    if df_bin.shape[0] > 0:
        y_true_bin = df_bin["true_bin"].astype(int).values
        y_pred_bin = df_bin["pred_bin"].fillna(-1).astype(int).values
        # Filter out rows where pred_bin == -1 (missing prediction)
        valid_mask = (y_pred_bin != -1)
        if valid_mask.sum() == 0:
            metrics["binary_error"] = "no_pred_binary"
            metrics["binary_counts"] = {
                "n_total": int(len(df_bin)),
                "n_with_pred": int(valid_mask.sum())
            }
        else:
            y_true_v = y_true_bin[valid_mask]
            y_pred_v = y_pred_bin[valid_mask]
            metrics["binary_accuracy"] = float(accuracy_score(y_true_v, y_pred_v))
            metrics["binary_precision"] = float(precision_score(y_true_v, y_pred_v, zero_division=0))
            metrics["binary_recall"] = float(recall_score(y_true_v, y_pred_v, zero_division=0))
            metrics["binary_f1"] = float(f1_score(y_true_v, y_pred_v, zero_division=0))
            metrics["binary_confusion_matrix"] = confusion_matrix(y_true_v, y_pred_v).tolist()
            metrics["binary_support"] = {
                "n_total": int(len(df_bin)),
                "n_with_pred": int(valid_mask.sum())
            }
    else:
        metrics["binary_error"] = "no_true_bin_rows"

    # Save predictions CSV and JSON report
    os.makedirs(os.path.dirname(pred_csv) or ".", exist_ok=True)
    df_out.to_csv(pred_csv, index=False)
    report = {
        "n_test_rows": int(len(df_out)),
        "metrics": metrics
    }
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print readable summary
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Rows evaluated: {len(df_out)}")
    print(f"Multi-class accuracy: {metrics.get('multi_accuracy'):.4f}")
    print(f"Multi-class macro F1 : {metrics.get('multi_f1_macro'):.4f}")
    print("Confusion matrix (multi-class):")
    for row in metrics.get("confusion_matrix", []):
        print(row)
    if "binary_accuracy" in metrics:
        print("\nBinary (PAD present vs none) accuracy:", metrics["binary_accuracy"])
        print("Binary precision:", metrics["binary_precision"])
        print("Binary recall:", metrics["binary_recall"])
        print("Binary f1:", metrics["binary_f1"])
        print("Binary confusion matrix:")
        for row in metrics.get("binary_confusion_matrix", []):
            print(row)
    else:
        print("\nBinary metrics not available:", metrics.get("binary_error"))

    print(f"\nPredictions written to: {pred_csv}")
    print(f"Report JSON written to: {out_json}")

if __name__ == "__main__":
    main()
