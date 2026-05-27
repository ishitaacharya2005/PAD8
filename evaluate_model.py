# evaluate_model.py
import os, json, argparse
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

# imports from your codebase
from model_utils import load_model, format_features_for_model, comparative_predict
from ocr_pad_extractor import extract_from_report

def bin_from_label(label):
    try:
        lab = int(label)
        return 0 if lab == 0 else 1
    except:
        return None

def eval_predictions(y_true_bin, y_pred_bin, y_true_multi=None, y_pred_multi=None, proba=None):
    res = {}
    res['accuracy'] = accuracy_score(y_true_bin, y_pred_bin)
    res['precision'] = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    res['recall'] = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    res['f1'] = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    res['confusion_matrix'] = confusion_matrix(y_true_bin, y_pred_bin).tolist()
    if proba is not None:
        try:
            res['roc_auc'] = roc_auc_score(y_true_bin, proba)
        except Exception:
            res['roc_auc'] = None
    # multiclass metrics if provided
    if y_true_multi is not None and y_pred_multi is not None:
        res['classification_report'] = classification_report(y_true_multi, y_pred_multi, zero_division=0, output_dict=True)
    return res

def run_evaluation(csv_path, model_path=None, output_json="evaluation_report.json"):
    df = pd.read_csv(csv_path)
    expected_cols = ['file_path', 'label']  # label is integer 0..3
    for c in expected_cols:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")
    df['label'] = df['label'].astype(int)
    df['pad_true'] = df['label'].apply(lambda x: 0 if x==0 else 1)

    model = None
    if model_path and os.path.exists(model_path):
        try:
            model = load_model(model_path)
            print("Loaded model from", model_path)
        except Exception as e:
            print("Warning: failed to load model:", e)
            model = None

    results = {
        "dataset": csv_path,
        "n": len(df),
        "model_present": bool(model is not None),
        "items": []
    }

    # For model predictions we will collect features per row
    for idx, row in df.iterrows():
        fp = row['file_path']
        true_label = int(row['label'])
        true_bin = 0 if true_label == 0 else 1

        # First: try extractor (gold tested CLI tool)
        try:
            parsed, full_text, conclusion = extract_from_report(fp)
            ext_bin = 1 if conclusion.get("PAD_present", False) else 0
            ext_sev = int(conclusion.get("severity_label")) if conclusion.get("severity_label") is not None else None
        except Exception as e:
            ext_bin = None
            ext_sev = None

        # Second: model predict (if model exists)
        model_bin = None
        model_sev = None
        model_proba_max = None
        if model is not None:
            try:
                # format_features_for_model expects parsed-like dict, but we don't have parsed for ground truth.
                # Use the extractor's parsed (if available); else fallback to empty dict -> NaNs.
                parsed_input = parsed if 'parsed' in locals() else {}
                X = format_features_for_model(parsed_input)  # uses saved feature list
                pred = model.predict(X)[0]
                model_sev = int(pred)
                model_bin = 0 if model_sev == 0 else 1
                try:
                    probs = model.predict_proba(X)[0]
                    model_proba_max = float(max(probs))
                except Exception:
                    model_proba_max = None
            except Exception as e:
                model_bin = None
                model_sev = None
                model_proba_max = None

        # Third: rule-based comparative_predict (uses parsed)
        try:
            rule_res = comparative_predict(parsed if 'parsed' in locals() else {})
            rule_bin = 1 if rule_res.get("pad_detected", False) else 0
            rule_sev_label = None
            try:
                # returned severity string like "Mild" -> convert to numeric mapping if possible
                sev_map = {"None":0,"Mild":1,"Moderate":2,"Severe":3}
                rule_sev_label = sev_map.get(rule_res.get("severity"), None)
            except:
                rule_sev_label = None
        except Exception:
            rule_bin = None
            rule_sev_label = None

        results['items'].append({
            "file": fp,
            "true_label": true_label,
            "true_bin": true_bin,
            "extractor_bin": ext_bin,
            "extractor_sev": ext_sev,
            "model_bin": model_bin,
            "model_sev": model_sev,
            "model_confidence": model_proba_max,
            "rule_bin": rule_bin,
            "rule_sev": rule_sev_label
        })

    # Build DataFrames for metric calc
    items_df = pd.DataFrame(results['items'])

    # Evaluate extractor (only rows where extractor produced prediction)
    ext_df = items_df[items_df['extractor_bin'].notnull()]
    if len(ext_df):
        ext_metrics = eval_predictions(ext_df['true_bin'], ext_df['extractor_bin'], y_true_multi=ext_df['true_label'], y_pred_multi=ext_df['extractor_sev'])
    else:
        ext_metrics = None

    # Evaluate model
    model_df = items_df[items_df['model_bin'].notnull()]
    if len(model_df):
        model_metrics = eval_predictions(model_df['true_bin'], model_df['model_bin'], y_true_multi=model_df['true_label'], y_pred_multi=model_df['model_sev'], proba=model_df['model_confidence'])
    else:
        model_metrics = None

    # Evaluate rule-based
    rule_df = items_df[items_df['rule_bin'].notnull()]
    if len(rule_df):
        rule_metrics = eval_predictions(rule_df['true_bin'], rule_df['rule_bin'], y_true_multi=rule_df['true_label'], y_pred_multi=rule_df['rule_sev'])
    else:
        rule_metrics = None

    results['summary'] = {
        "extractor": ext_metrics,
        "model": model_metrics,
        "rule_based": rule_metrics
    }

    # save results
    with open(output_json, "w", encoding="utf-8") as fout:
        json.dump(results, fout, indent=2, ensure_ascii=False)

    print("Evaluation saved to", output_json)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Labeled CSV with file_path,label")
    parser.add_argument("--model", default="models/model.pkl", help="Path to model.pkl")
    parser.add_argument("--out", default="evaluation_report.json")
    args = parser.parse_args()
    run_evaluation(args.csv, args.model, args.out)
    