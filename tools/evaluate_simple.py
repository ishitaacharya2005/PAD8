#!/usr/bin/env python3
"""
Robust evaluator that can (a) read CSV with ground truth, (b) optionally load a saved model
    and generate predictions (without requiring model_utils), and (c) compute multi-class
    and binary (PAD/no-PAD) metrics safely.

Usage:
  python tools/evaluate_simple.py --csv data/eval_labels_clean.csv --out data/eval_report.json
  python tools/evaluate_simple.py --csv data/eval_labels_clean.csv --model models/model.pkl --out data/eval_report.json
  python tools/evaluate_simple.py --csv data/eval_labels_clean.csv --model models/model.pkl --feature-cols col1,col2,... --out data/eval_report.json
"""
import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
import joblib
import re
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from collections import Counter

SEV_STR_TO_INT = {
    "none": 0, "normal": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3,
    "0": 0, "1": 1, "2": 2, "3": 3
}

def normalize_single_label(x):
    """Return int label 0..3 or None if unknown"""
    if pd.isna(x):
        return None
    # direct ints
    if isinstance(x, (int, np.integer)):
        xv = int(x)
        if xv in (0,1,2,3):
            return xv
        # if 0/1 where 1 maybe PAD (we'll handle in binary), keep as-is if 0/1
        if xv in (0,1):
            return int(xv)
        return None
    s = str(x).strip()
    if s == "":
        return None
    low = s.lower()
    # boolean-like strings
    if low in ("true","yes","y","1"):
        return 1
    if low in ("false","no","n","0"):
        return 0
    # severity keywords
    for k,v in SEV_STR_TO_INT.items():
        if low.startswith(k):
            return int(v)
    # find isolated digit 0-3 in string
    m = re.search(r"\b([0-3])\b", s)
    if m:
        return int(m.group(1))
    # last try: numeric parse
    try:
        fv = float(s)
        iv = int(round(fv))
        if iv in (0,1,2,3):
            return iv
    except Exception:
        pass
    return None

def to_binary_from_severity(label):
    """map severity 0->0 (no PAD), 1/2/3 ->1 (PAD). If label None -> None"""
    if label is None:
        return None
    try:
        li = int(label)
    except Exception:
        return None
    return 0 if li == 0 else 1

def safe_classification_report(y_true, y_pred, labels_all=[0,1,2,3]):
    present = sorted(set([int(x) for x in list(y_true) + list(y_pred) if x is not None]))
    if not present:
        return {"error": "no labels present"}
    labels_param = [l for l in labels_all if l in present]
    if not labels_param:
        labels_param = present
    report = classification_report(y_true, y_pred, labels=labels_param, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels_param).tolist()
    return {"report": report, "labels": labels_param, "confusion_matrix": cm}

def discover_model_feature_names(model):
    """Try multiple heuristics to find feature names expected by model"""
    names = []
    try:
        if hasattr(model, "feature_names_in_"):
            names = list(getattr(model, "feature_names_in_"))
            return names
    except Exception:
        pass
    # pipeline steps
    try:
        if hasattr(model, "named_steps"):
            for name, step in model.named_steps.items():
                if hasattr(step, "feature_names_in_"):
                    names = list(getattr(step, "feature_names_in_"))
                    if names:
                        return names
    except Exception:
        pass
    # some pipelines keep feature names under 'feature_names_in_' deeper - check estimators recursively
    def recurse_for_features(obj):
        try:
            if hasattr(obj, "feature_names_in_"):
                return list(getattr(obj, "feature_names_in_"))
        except Exception:
            pass
        if hasattr(obj, "__dict__"):
            for v in vars(obj).values():
                if isinstance(v, (list, tuple)):
                    continue
                if hasattr(v, "feature_names_in_"):
                    return list(getattr(v, "feature_names_in_"))
        return []
    names = recurse_for_features(model)
    return names

def build_X_from_csv_and_feature_list(df, feature_list):
    # Only keep columns that exist in df
    cols = [c for c in feature_list if c in df.columns]
    if not cols:
        return None, cols
    X = df[cols].copy()
    # Ensure numeric dtype where possible
    for c in cols:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X, cols

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--model", required=False)
    p.add_argument("--out", required=True)
    p.add_argument("--feature-cols", required=False, help="Comma-separated feature column names to use for model input if auto-detection fails")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print("CSV not found:", args.csv)
        sys.exit(2)
    df = pd.read_csv(args.csv)
    df_cols = [c for c in df.columns]

    # find true label column
    possible_true = [c for c in df.columns if c.lower() in ("true_label","label","true_sev","true_severity","gold_label","ground_truth")]
    if not possible_true:
        print("CSV does not contain a recognized true label column. Provide a column named one of: true_label,label,true_sev,true_severity,gold_label,ground_truth")
        sys.exit(2)
    true_col = possible_true[0]
    true_vals_raw = df[true_col].tolist()
    true_norm = [normalize_single_label(x) for x in true_vals_raw]

    # Look for pred column in CSV
    possible_preds = [c for c in df.columns if c.lower() in ("pred_label","pred_severity","pred_sev","rule_sev","rule_severity","predicted_label","predicted_severity")]
    pred_col = possible_preds[0] if possible_preds else None

    pred_norm = None
    model = None
    feature_cols_used = None

    if pred_col:
        pred_vals_raw = df[pred_col].tolist()
        pred_norm = [normalize_single_label(x) for x in pred_vals_raw]

    if args.model:
        # try load model
        try:
            model = joblib.load(args.model)
            print("Loaded model from", args.model)
        except Exception as e:
            print("Failed to load model:", e)
            model = None

        if model is not None:
            # discover feature names
            feat_names = discover_model_feature_names(model)
            if feat_names:
                print("Discovered model.feature_names:", feat_names[:20], "...")
            else:
                print("Model does not expose feature names. You can pass --feature-cols to specify them.")

            X = None
            if feat_names:
                X, used_cols = build_X_from_csv_and_feature_list(df, feat_names)
                feature_cols_used = used_cols
                if X is None or X.shape[1] == 0:
                    print("None of model.feature_names were present in CSV columns.")
                    X = None
            # If feature cols provided by CLI, try them
            if X is None and args.feature_cols:
                cli_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
                X, used_cols = build_X_from_csv_and_feature_list(df, cli_cols)
                feature_cols_used = used_cols
                if X is None:
                    print("Provided --feature-cols not present in CSV.")
            # Last resort: try numeric columns intersection heuristic
            if X is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 1:
                    # if model exposed feature names, intersect with numeric cols
                    if feat_names:
                        intersect = [c for c in feat_names if c in numeric_cols]
                        if intersect:
                            X, used_cols = build_X_from_csv_and_feature_list(df, intersect)
                            feature_cols_used = used_cols
                    # else if model has no feature names, we allow using top numeric columns up to model n_features_in_ if available
                    if X is None:
                        # find expected n_features if available
                        nfeat = None
                        try:
                            if hasattr(model, "n_features_in_"):
                                nfeat = int(getattr(model, "n_features_in_"))
                        except Exception:
                            nfeat = None
                        # use all numeric columns or slice to nfeat if known
                        cols_try = numeric_cols if nfeat is None else numeric_cols[:nfeat]
                        X, used_cols = build_X_from_csv_and_feature_list(df, cols_try)
                        feature_cols_used = used_cols
                else:
                    X = None

            if X is not None:
                # fill missing numeric with median as a safe default
                X = X.fillna(X.median())
                try:
                    preds = model.predict(X)
                    pred_norm = [int(p) for p in preds]
                    print(f"Model predicted {len(pred_norm)} rows using features: {feature_cols_used}")
                except Exception as e:
                    print("Model.predict failed on assembled X:", e)
                    pred_norm = None
            else:
                print("Failed to assemble model input X from CSV. Provide --feature-cols or ensure model.feature_names_in_ saved in model.")
                pred_norm = pred_norm  # keep existing if any

    # If still no predictions available, write JSON explaining and exit
    out = {"n_rows": len(df), "true_label_counts": dict(Counter([v for v in true_norm if v is not None]))}
    if pred_norm is None:
        out["error"] = "No predictions available. Either provide predictions in CSV or pass --model and ensure model.feature_names are discoverable or pass --feature-cols."
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print("No predictions available. See JSON output for details:", args.out)
        return

    # Now evaluate: align rows with valid true labels
    idx_valid = [i for i, v in enumerate(true_norm) if v is not None]
    if not idx_valid:
        print("No valid true labels after normalization.")
        sys.exit(2)

    y_true = [true_norm[i] for i in idx_valid]
    # align predictions
    y_pred_raw = [pred_norm[i] if i < len(pred_norm) else None for i in idx_valid]

    # only rows where both available
    idx_eval = [i for i,(t,p) in enumerate(zip(y_true, y_pred_raw)) if p is not None]
    if not idx_eval:
        print("No rows with both true labels and predicted labels available for evaluation.")
        sys.exit(2)

    y_true_eval = [y_true[i] for i in idx_eval]
    y_pred_eval = [int(y_pred_raw[i]) for i in idx_eval]

    multi = safe_classification_report(y_true_eval, y_pred_eval, labels_all=[0,1,2,3])
    out["multi"] = {}
    if "error" in multi:
        out["multi"]["error"] = multi["error"]
    else:
        out["multi"]["classification_report"] = multi["report"]
        out["multi"]["labels"] = multi["labels"]
        out["multi"]["confusion_matrix"] = multi["confusion_matrix"]
        out["multi"]["accuracy"] = float(accuracy_score(y_true_eval, y_pred_eval))
        out["multi"]["balanced_accuracy"] = float(balanced_accuracy_score(y_true_eval, y_pred_eval))

    # Binary PAD vs No-PAD
    y_true_bin = [to_binary_from_severity(v) for v in y_true_eval]
    y_pred_bin = [to_binary_from_severity(v) for v in y_pred_eval]
    idx_bin = [i for i,(a,b) in enumerate(zip(y_true_bin, y_pred_bin)) if a is not None and b is not None]
    if idx_bin:
        y_true_bin_eval = [y_true_bin[i] for i in idx_bin]
        y_pred_bin_eval = [y_pred_bin[i] for i in idx_bin]
        acc = accuracy_score(y_true_bin_eval, y_pred_bin_eval)
        bal_acc = balanced_accuracy_score(y_true_bin_eval, y_pred_bin_eval)
        prec, rec, f1, sup = precision_recall_fscore_support(y_true_bin_eval, y_pred_bin_eval, average="binary", zero_division=0)
        cm = confusion_matrix(y_true_bin_eval, y_pred_bin_eval, labels=[0,1]).tolist()
        out["binary"] = {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "confusion_matrix": cm,
            "n_eval": len(y_true_bin_eval),
            "true_counts": dict(Counter(y_true_bin_eval)),
            "pred_counts": dict(Counter(y_pred_bin_eval)),
        }
    else:
        out["binary"] = {"error": "Insufficient binary-labeled rows after normalization."}

    # sample mismatches
    mismatches = []
    for i, (t,p) in enumerate(zip(y_true_eval, y_pred_eval)):
        if t != p:
            orig_idx = idx_valid[idx_eval[i]]
            row = df.iloc[orig_idx].to_dict()
            mismatches.append({"index": int(orig_idx), "true": int(t), "pred": int(p), "row_preview": {k: row.get(k) for k in list(df.columns)[:10]}})
            if len(mismatches) >= 50:
                break
    out["mismatches_sample"] = mismatches
    out["n_evaluated_rows"] = len(y_true_eval)
    out["feature_cols_used"] = feature_cols_used

    # save predictions appended CSV
    try:
        pred_col_name = "eval_pred_label"
        df_out = df.copy()
        # attach predictions aligned to original rows (if model predicted full length)
        full_pred = [None] * len(df)
        # if pred_norm length == len(df) assume full alignment
        if len(pred_norm) == len(df):
            full_pred = pred_norm
        else:
            # map back via idx_valid/idx_eval
            for idx_local, orig in enumerate(idx_valid):
                if idx_local < len(pred_norm):
                    full_pred[orig] = pred_norm[idx_local]
        df_out[pred_col_name] = full_pred
        pred_csv = os.path.splitext(args.csv)[0] + ".pred.csv"
        df_out.to_csv(pred_csv, index=False)
        out["pred_csv"] = pred_csv
    except Exception as e:
        out["pred_csv_error"] = str(e)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Evaluation complete. rows evaluated (multi-class):", out["n_evaluated_rows"])
    if "multi" in out and "accuracy" in out["multi"]:
        print("Multi-class accuracy:", out["multi"]["accuracy"])
    if "binary" in out and "error" not in out["binary"]:
        print("Binary accuracy:", out["binary"]["accuracy"], "precision:", out["binary"]["precision"], "recall:", out["binary"]["recall"])
    print("JSON report written to", args.out)
    if "pred_csv" in out:
        print("Predictions CSV written to", out["pred_csv"])

if __name__ == "__main__":
    main()
