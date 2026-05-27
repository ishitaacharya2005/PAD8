#!/usr/bin/env python3
"""
evaluate_model_end_to_end.py

Usage:
  python tools/evaluate_model_end_to_end.py \
      --labels data/eval_labels_clean.csv \
      --model models/model.pkl \
      --out data/eval_report_e2e.json \
      --pred data/eval_predictions_e2e.csv

Input CSV should have columns: file_path,label
  - file_path: relative path to report file (pdf/html/image) that extractor/parser can handle
  - label: integer 0/1/2/3 (or NaN for unknown)
"""

import argparse
import importlib
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict

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

# ----------------------
# Helpers: robust loader
# ----------------------
def load_module_by_path(mod_name, file_path):
    """Load module from file path and return module (or None)."""
    try:
        if not os.path.exists(file_path):
            return None
        spec = importlib.util.spec_from_file_location(mod_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules[mod_name] = mod
        return mod
    except Exception as e:
        print(f"[loader] failed to load {file_path}: {e}")
        return None

def try_import(name, fallback_path=None):
    try:
        return importlib.import_module(name)
    except Exception:
        if fallback_path:
            return load_module_by_path(name, fallback_path)
        return None

# ----------------------
# CLI
# ----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True, help="CSV with columns file_path,label")
    p.add_argument("--model", required=False, help="Path to model.pkl (joblib). If absent, will use rule-based comparator)")
    p.add_argument("--out", required=True, help="JSON output report")
    p.add_argument("--pred", required=True, help="Predictions CSV output")
    p.add_argument("--extractor", default="ocr_pad_extractor.py", help="Path to extractor file (default: ocr_pad_extractor.py at repo root)")
    p.add_argument("--parser_utils", default="extras/parser_utils.py", help="Path to parser_utils (fallback)")
    p.add_argument("--model_utils", default="model_utils.py", help="Path to model_utils (optional)")
    return p.parse_args()

# ----------------------
# Feature assembly helpers
# ----------------------
def median_fill_df(df, cols):
    """Return df with cols present; fill missing with medians"""
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    # fill numeric columns with median
    for c in cols:
        try:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            med = out[c].median()
            out[c] = out[c].fillna(med)
        except Exception:
            out[c] = out[c].fillna(0)
    return out[cols]

def ensure_cols(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

# ----------------------
# Main
# ----------------------
def main():
    args = parse_args()
    labels_csv = args.labels
    out_json = args.out
    pred_csv = args.pred
    model_path = args.model

    # load label file
    if not os.path.exists(labels_csv):
        print("Labels CSV not found:", labels_csv)
        sys.exit(1)
    df_labels = pd.read_csv(labels_csv)
    if "file_path" not in df_labels.columns or "label" not in df_labels.columns:
        raise ValueError("Labels CSV must contain 'file_path' and 'label' columns")

    # load extractor & parser_utils
    script_dir = os.getcwd()
    extractor_mod = try_import("ocr_pad_extractor", os.path.join(script_dir, args.extractor))
    parser_mod = try_import("extras.parser_utils", os.path.join(script_dir, args.parser_utils))
    model_utils_mod = try_import("model_utils", os.path.join(script_dir, args.model_utils))

    if extractor_mod is None and parser_mod is None:
        print("[ERR] No extractor and no parser_utils found. Exiting.")
        sys.exit(1)

    extract_from_report = getattr(extractor_mod, "extract_from_report", None)
    parse_report_text = getattr(parser_mod, "parse_report_text", None)

    # Try loading model (if provided)
    model = None
    load_model = None
    predict_with_model = None
    comparative_predict = None
    if model_utils_mod is not None:
        load_model = getattr(model_utils_mod, "load_model", None)
        predict_with_model = getattr(model_utils_mod, "predict_with_model", None)
        comparative_predict = getattr(model_utils_mod, "comparative_predict", None)

    if model_path:
        # prefer model_utils.load_model if available (it may wrap joblib)
        if load_model:
            try:
                model = load_model(model_path)
                print("Loaded model via model_utils.load_model")
            except Exception as e:
                print("model_utils.load_model failed:", e)
                model = None
        if model is None:
            # try joblib
            try:
                import joblib
                model = joblib.load(model_path)
                print("Loaded model via joblib:", model_path)
            except Exception as e:
                print("Failed to load model via joblib:", e)
                model = None

    # If model missing but model_utils provided, keep comparator
    if model is None and comparative_predict is None and model_utils_mod:
        comparative_predict = getattr(model_utils_mod, "comparative_predict", None)

    # If model exists, try to discover its expected feature list
    model_feature_names = None
    feature_names_path = os.path.join("models", "feature_names.pkl")
    if os.path.exists(feature_names_path):
        try:
            import joblib
            model_feature_names = joblib.load(feature_names_path)
            print("Loaded feature list from", feature_names_path)
        except Exception:
            model_feature_names = None
    # fallback: try model.feature_names_in_ if scikit-learn pipeline
    if model is not None and model_feature_names is None:
        try:
            fn = getattr(model, "feature_names_in_", None)
            if fn is not None:
                model_feature_names = list(fn)
                print("Discovered model.feature_names_in_")
        except Exception:
            model_feature_names = None

    # If still none, we'll attempt to use a common set of PSV+waveform+ABI columns
    if model_feature_names is None:
        model_feature_names = [
            "Patient ID", "Age", "ABI",
            "Common Femoral Artery PSV", "Profundus Femoris PSV",
            "Proximal SFA PSV", "Mid SFA PSV", "Distal SFA PSV",
            "Popliteal Artery PSV", "Peroneal / Posterior Tibial Artery PSV",
            "Anterior Tibial Artery PSV",
            "waveform_monophasic_count", "waveform_biphasic_count", "waveform_triphasic_count",
            "mean_PSV",
            # duplication style names used in some training code:
            "PSV__Common_Femoral_Artery", "PSV__Profundus_Femoris", "PSV__Proximal_Superficial_Femoral_Artery",
            "PSV__Mid_SFA", "PSV__Distal_SFA", "PSV__Popliteal_Artery", "PSV__Peroneal___Posterior_Tibial_Artery", "PSV__Anterior_Tibial_Artery"
        ]
        print("Using fallback feature list (len=%d)" % len(model_feature_names))

    # iterate rows, extract features
    rows = []
    failures = []
    total = len(df_labels)
    for idx, rr in df_labels.iterrows():
        fp = rr["file_path"]
        lbl = rr.get("label", None)
        # try absolute or relative path
        if not os.path.exists(fp):
            # try relative to cwd
            candidate = os.path.join(os.getcwd(), fp)
            if os.path.exists(candidate):
                fp = candidate
        if not os.path.exists(fp):
            failures.append((fp, "file_not_found"))
            rows.append({"file_path": rr["file_path"], "label": lbl, "_extracted": None})
            continue

        parsed = None
        full_text = None
        conclusion = None
        # Primary: use extractor if available
        try:
            if extract_from_report:
                parsed, full_text, conclusion = extract_from_report(fp)
            elif parse_report_text:
                # fallback: read text (HTML/PDF) -> parse_report_text expects text
                # try reading file if html/text, else None
                if fp.lower().endswith(".html") or fp.lower().endswith(".htm"):
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        txt = f.read()
                    parsed = parse_report_text(txt)
                else:
                    # if it's pdf, try parser_utils' html extraction or skip
                    parsed = parse_report_text("")  # best-effort empty
            # normalize ensure parsed is dict
            if parsed is None:
                raise RuntimeError("parsed_none")
            # store parsed snapshot for debugging
            rows.append({"file_path": rr["file_path"], "label": lbl, "_parsed_snapshot": parsed})
        except Exception as e:
            failures.append((fp, str(e)))
            rows.append({"file_path": rr["file_path"], "label": lbl, "_extracted": None})
            continue

    features_df = pd.DataFrame(rows)

    # Merge/expand parsed snapshots into feature columns
    expanded_rows = []
    for i, r in features_df.iterrows():
        parsed = r.get("_parsed_snapshot")
        if isinstance(parsed, dict):
            rowdict = {}
            # copy keys that match model_feature_names heuristically
            for k, v in parsed.items():
                # unify keys: allow PSV__ names and human names
                rowdict[k] = v
                # also map some PSV human names to PSV__ style if present
                # (a few common mappings)
                if k == "Common Femoral Artery PSV":
                    rowdict["PSV__Common_Femoral_Artery"] = v
                if k == "Mid SFA PSV" or k == "Mid SFA":
                    rowdict["PSV__Mid_SFA"] = v
                if k == "ABI":
                    rowdict["ABI"] = v
            # ensure label present
            rowdict["file_path"] = r["file_path"]
            rowdict["label"] = r["label"]
            expanded_rows.append(rowdict)
        else:
            # blank parsed; create minimal row
            expanded_rows.append({"file_path": r["file_path"], "label": r["label"]})

    feat_df = pd.DataFrame(expanded_rows)

    # ensure all model_feature_names present, fill with medians
    feat_df = ensure_cols(feat_df, model_feature_names + ["file_path", "label"])
    # convert numeric columns
    for c in model_feature_names:
        feat_df[c] = pd.to_numeric(feat_df[c], errors="coerce")

    # fill missing numeric features with median
    medians = {}
    for c in model_feature_names:
        med = feat_df[c].median(skipna=True)
        if math.isnan(med):
            med = 0.0
        medians[c] = med
        feat_df[c] = feat_df[c].fillna(med)

    # Build X matrix for model predict (same order as model_feature_names)
    X = feat_df[model_feature_names].copy()

    preds = []
    pred_probs = []
    pred_labels = []
    # If we have a model, use it
    if model is not None and predict_with_model is None:
        # direct sklearn pipeline
        try:
            proba = None
            y_pred = model.predict(X)
            try:
                proba = model.predict_proba(X)
            except Exception:
                proba = None
            for i in range(len(X)):
                lab = int(y_pred[i])
                preds.append({"pred_label": lab})
                pred_probs.append(proba[i].tolist() if proba is not None else None)
                pred_labels.append(lab)
            print("Predicted with model, N=", len(pred_labels))
        except Exception as e:
            print("Model.predict failed:", e)
            model = None

    if model is not None and predict_with_model is not None:
        # model_utils wrapper available
        for i in range(len(X)):
            parsed_row = feat_df.iloc[i].to_dict()
            try:
                res = predict_with_model(model, parsed_row)
                # standardize
                pl = res.get("severity")
                pad_detected = res.get("pad_detected", None)
                proba = res.get("proba", None)
                # convert severity string to label if possible
                sev_map = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
                lab = None
                if isinstance(pl, str) and pl in sev_map:
                    lab = sev_map[pl]
                elif isinstance(pl, (int, float)):
                    lab = int(pl)
                else:
                    # fallback: pad_detected True -> 1 else 0
                    lab = 1 if pad_detected else 0
                preds.append({"pred_label": lab})
                pred_probs.append(proba)
                pred_labels.append(lab)
            except Exception as e:
                preds.append({"pred_label": None})
                pred_probs.append(None)
                pred_labels.append(None)

    # If model not available, but comparator exists:
    if model is None and comparative_predict is not None:
        for i in range(len(X)):
            parsed_row = feat_df.iloc[i].to_dict()
            try:
                res = comparative_predict(parsed_row)
                sev = res.get("severity")
                sev_map = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
                lab = sev_map.get(sev, None) if isinstance(sev, str) else None
                preds.append({"pred_label": lab, "raw":res})
                pred_probs.append(None)
                pred_labels.append(lab)
            except Exception as e:
                preds.append({"pred_label": None})
                pred_probs.append(None)
                pred_labels.append(None)

    # If still empty, mark all None
    if not preds:
        preds = [{"pred_label": None} for _ in range(len(X))]
        pred_probs = [None] * len(X)
        pred_labels = [None] * len(X)

    # attach predictions into dataframe
    feat_df["pred_label"] = [p.get("pred_label") for p in preds]
    feat_df["pred_proba"] = pred_probs
    feat_df["true_label"] = feat_df["label"].apply(lambda x: int(x) if (not pd.isna(x)) else None)

    # compute binary mapping: true_bin = 0 if label==0 else 1 if label in 1..3
    feat_df["true_bin"] = feat_df["true_label"].apply(lambda x: int(0) if x == 0 else (int(1) if x in (1,2,3) else None))
    feat_df["pred_bin"] = feat_df["pred_label"].apply(lambda x: int(0) if x == 0 else (int(1) if x in (1,2,3) else None))

    # compute metrics (only on rows where true_bin not null)
    df_eval = feat_df.dropna(subset=["true_bin"])
    metrics = {}
    if len(df_eval):
        y_true = df_eval["true_bin"].astype(int).values
        y_pred_bin = df_eval["pred_bin"].fillna(-1).astype(int).values  # -1 -> missing
        # consider only rows where pred_bin is 0 or 1 (not -1)
        valid_mask = y_pred_bin != -1
        if valid_mask.sum() == 0:
            metrics["error"] = "no_pred_binary"
        else:
            y_true_v = y_true[valid_mask]
            y_pred_v = y_pred_bin[valid_mask]
            metrics["accuracy"] = float(accuracy_score(y_true_v, y_pred_v))
            metrics["precision"] = float(precision_score(y_true_v, y_pred_v, zero_division=0))
            metrics["recall"] = float(recall_score(y_true_v, y_pred_v, zero_division=0))
            metrics["f1"] = float(f1_score(y_true_v, y_pred_v, zero_division=0))
            metrics["confusion_matrix"] = confusion_matrix(y_true_v, y_pred_v).tolist()
            metrics["classification_report"] = classification_report(df_eval["true_label"].fillna(-999).astype(int), df_eval["pred_label"].fillna(-999).astype(int), zero_division=0, output_dict=True)
    else:
        metrics["error"] = "no_true_labels"

    # save predictions csv and report json
    out_pred_df = feat_df[["file_path", "true_label", "pred_label", "true_bin", "pred_bin", "pred_proba", "_parsed_snapshot"]]
    out_pred_df.to_csv(pred_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "failures": failures, "n_rows": len(feat_df)}, f, indent=2)

    print("Wrote predictions to", pred_csv)
    print("Wrote eval JSON to", out_json)
    print("Summary metrics:", metrics)

if __name__ == "__main__":
    main()
