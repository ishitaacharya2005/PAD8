#!/usr/bin/env python3
"""
Simple evaluation script for PAD binary detection.

Usage (from project root):
python tools/evaluate_binary.py --features data/eval_features.csv --model models/model.pkl --out data/eval_metrics.json

Outputs:
 - data/eval_predictions.csv  (per-row true_label, pred_label, true_bin, pred_bin, file_path)
 - data/eval_metrics.json     (accuracy, precision, recall, f1, confusion_matrix, counts)
"""
import argparse
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def load_model_and_features(model_path):
    m = joblib.load(model_path)
    # Try to discover feature names
    feat_list = None
    feat_file = os.path.join("models", "feature_names.pkl")
    if os.path.exists(feat_file):
        try:
            feat_list = joblib.load(feat_file)
            print("Loaded feature list from models/feature_names.pkl")
        except Exception:
            feat_list = None
    if feat_list is None:
        # sklearn pipeline may expose named_steps -> transformer with feature_names_in_
        if hasattr(m, "named_steps"):
            # try to extract from pipeline steps
            for step in m.named_steps.values():
                if hasattr(step, "feature_names_in_"):
                    feat_list = list(getattr(step, "feature_names_in_"))
                    break
        # direct attribute
        if feat_list is None and hasattr(m, "feature_names_in_"):
            feat_list = list(getattr(m, "feature_names_in_"))
    return m, feat_list

def coerce_df_for_model(df, feature_names):
    # Ensure all feature_names exist in df; if missing, create column with NaN
    X = pd.DataFrame(index=df.index)
    for f in feature_names:
        if f in df.columns:
            X[f] = df[f]
        else:
            X[f] = np.nan
    # convert non-numeric to numeric where possible
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    # fill numeric columns with median (simple imputation)
    for c in X.columns:
        if X[c].isna().all():
            X[c] = X[c].fillna(0.0)
        else:
            med = X[c].median()
            X[c] = X[c].fillna(med)
    return X

def map_label_to_binary(label):
    # label is assumed numeric: 0 -> No PAD, 1/2/3 -> PAD
    try:
        if pd.isna(label):
            return None
        v = int(float(label))
        return 0 if v == 0 else 1
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="CSV with features (rows to evaluate) and a 'label' column")
    ap.add_argument("--model", required=True, help="Trained model pickle (joblib)")
    ap.add_argument("--out", required=True, help="JSON output path for metrics (also writes predictions CSV next to it)")
    args = ap.parse_args()

    if not os.path.exists(args.features):
        raise FileNotFoundError(args.features)
    if not os.path.exists(args.model):
        raise FileNotFoundError(args.model)

    df = pd.read_csv(args.features)
    print("Loaded CSV with columns:", list(df.columns))

    model, feature_names = load_model_and_features(args.model)
    if feature_names is None:
        raise RuntimeError("Could not determine model feature names. Save them to models/feature_names.pkl or ensure model exposes feature_names_in_.")

    print("Model expects features:", feature_names)

    X = coerce_df_for_model(df, feature_names)

    # Predict
    preds = model.predict(X)
    # if model outputs multi-class numeric labels
    try:
        pred_labels = [int(np.round(float(p))) for p in preds]
    except Exception:
        pred_labels = preds.tolist()

    # Map to binary
    pred_bin = [0 if (p == 0 or p == "0") else 1 for p in pred_labels]

    # True labels from CSV
    if "label" not in df.columns:
        raise RuntimeError("Input CSV must contain a 'label' column with true labels (0..3).")

    true_labels = df["label"].tolist()
    true_bin = [map_label_to_binary(v) for v in true_labels]

    # Filter rows where true_bin is None (label missing/unknown); we will ignore those rows for metrics
    indices = [i for i, tb in enumerate(true_bin) if tb is not None]
    if len(indices) == 0:
        raise RuntimeError("No valid true labels (0/1/2/3) found in 'label' column to evaluate.")

    y_true = [true_bin[i] for i in indices]
    y_pred = [pred_bin[i] for i in indices]

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    cls_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "n_total_rows": len(df),
        "n_evaluated_rows": len(indices),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm,
        "classification_report": cls_report
    }

    out_json = args.out
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("Wrote evaluation JSON to", out_json)

    # Save predictions CSV
    pred_df = pd.DataFrame({
        "file_path": df.get("file_path"),
        "true_label": true_labels,
        "pred_label": pred_labels,
        "true_bin": true_bin,
        "pred_bin": pred_bin
    })
    pred_csv = os.path.splitext(out_json)[0] + ".pred.csv"
    pred_df.to_csv(pred_csv, index=False)
    print("Wrote predictions CSV to", pred_csv)
    print("Metrics summary -- accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1: {:.3f}".format(acc, prec, rec, f1))

if __name__ == "__main__":
    main()
