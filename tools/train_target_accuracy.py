#tools/train_target_accuracy.py
"""
tools/train_target_accuracy.py

See previous conversation for usage notes. This version is hardened to handle NaNs
and wraps logistic regression in an imputer+scaler pipeline so it won't crash on missing values.
"""
import argparse
import json
import os
import random
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Training CSV (with label column)")
    p.add_argument("--label-col", default="label", help="True label column name in CSV")
    p.add_argument("--out-model", default="models/model_target.pkl", help="Where to save the chosen model")
    p.add_argument("--report", default="data/train_target_report.json", help="JSON report path")
    p.add_argument("--target", type=float, default=0.78, help="Target validation accuracy (0..1)")
    p.add_argument("--tol", type=float, default=0.02, help="Acceptable tolerance (±) from target")
    p.add_argument("--test-size", type=float, default=0.2, help="Validation split fraction")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--allow-noise", action="store_true", help="Allow label noise if necessary")
    p.add_argument("--max-noise", type=float, default=0.20, help="Max fraction of labels to flip if allowed")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

def load_csv(csv_path, label_col):
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in CSV columns: {list(df.columns)}")
    return df

def prepare_X_y(df, label_col):
    # Remove obviously non-numeric text columns that won't be used as numeric features
    X = df.drop(columns=[label_col]).copy()
    drop_like = [c for c in X.columns if any(keyword in c.lower() for keyword in ("name", "remarks", "file", "snapshot", "patient"))]
    X = X.drop(columns=[c for c in drop_like if c in X.columns], errors="ignore")

    # Try to coerce remaining columns to numeric
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Drop columns that are entirely NaN (can't impute a column with no data)
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)
    # Fill NaNs with median; if median is NaN (e.g., constant NaN), fallback to 0
    medians = X.median(numeric_only=True)
    for c in X.columns:
        m = medians.get(c, np.nan)
        if np.isnan(m):
            X[c] = X[c].fillna(0.0)
        else:
            X[c] = X[c].fillna(m)

    # final safety: if any remaining NaN (unlikely), replace with 0
    X = X.fillna(0.0)

    y = df[label_col].astype(int)
    return X, y

def flip_labels(y, noise_fraction, random_state):
    y = y.copy().reset_index(drop=True)
    n = len(y)
    k = int(round(n * noise_fraction))
    if k <= 0:
        return y
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=k, replace=False)
    classes = sorted(list(y.unique()))
    if len(classes) < 2:
        return y
    for i in idx:
        cur = int(y.iloc[i])
        options = [c for c in classes if c != cur]
        y.iloc[i] = int(rng.choice(options))
    return y

def simple_models_grid():
    models = []
    # shallow decision trees
    for depth in [2, 3, 5, 8]:
        models.append(("dt_depth_%d" % depth, DecisionTreeClassifier(max_depth=depth, random_state=0)))
    # Random forests small -> larger
    for n in [10, 50, 100]:
        for md in [3, 5, None]:
            models.append(("rf_n%d_md%s" % (n, str(md)), RandomForestClassifier(n_estimators=n, max_depth=md, random_state=0, n_jobs=-1)))
    # logistic wrapped in pipeline with imputer+scaler
    logpipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=400, solver="lbfgs", C=1.0))
    ])
    logpipe_l1 = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("logreg_l1", LogisticRegression(max_iter=400, solver="liblinear", penalty="l1", C=1.0))
    ])
    models.append(("logreg_l2_pipe", logpipe))
    models.append(("logreg_l1_pipe", logpipe_l1))
    return models

def evaluate_candidate(model, X_train, X_val, y_train, y_val):
    # Fit and evaluate; return fitted model and metrics
    m = deepcopy(model)
    m.fit(X_train, y_train)
    y_pred = m.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_val, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_val, y_pred).tolist()
    return {"model": m, "acc": acc, "prec": prec, "rec": rec, "f1": f1, "cm": cm}

def main():
    args = parse_args()
    df = load_csv(args.csv, args.label_col)
    X, y = prepare_X_y(df, args.label_col)
    if args.verbose:
        print("Prepared features shape:", X.shape)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state, stratify=y)
    candidates = simple_models_grid()
    best = None
    target = args.target
    tol = args.tol

    if args.verbose:
        print(f"Trying {len(candidates)} candidate models on dataset with {X_train.shape[0]} train and {X_val.shape[0]} val rows.")

    results = []
    # First pass: no noise
    for name, est in candidates:
        try:
            r = evaluate_candidate(est, X_train, X_val, y_train, y_val)
        except Exception as e:
            if args.verbose:
                print(f"[error] candidate {name} failed: {e}")
            continue
        r["name"] = name
        results.append(r)
        if args.verbose:
            print(f"[no-noise] {name} acc={r['acc']:.4f} f1={r['f1']:.4f}")
        if abs(r["acc"] - target) <= tol:
            best = r
            best["noise_fraction"] = 0.0
            break

    # If not found and allow_noise, try flipping labels progressively
    if best is None and args.allow_noise:
        noise_levels = list(np.linspace(0.02, args.max_noise, num=10))
        for noise in noise_levels:
            if args.verbose:
                print("Trying with label noise:", noise)
            y_train_noisy = flip_labels(y_train, noise, args.random_state)
            for name, est in candidates:
                try:
                    r = evaluate_candidate(est, X_train, X_val, y_train_noisy, y_val)
                except Exception as e:
                    if args.verbose:
                        print(f"[noise={noise:.2f}] candidate {name} failed: {e}")
                    continue
                r["name"] = name
                r["noise"] = noise
                results.append(r)
                if args.verbose:
                    print(f"[noise={noise:.2f}] {name} acc={r['acc']:.4f}")
                if abs(r["acc"] - target) <= tol:
                    best = r
                    break
            if best is not None:
                break

    # If nothing within tolerance, pick the closest
    if best is None:
        if len(results) == 0:
            raise RuntimeError("No successful candidate evaluations; check data / features.")
        results_sorted = sorted(results, key=lambda r: abs(r["acc"] - target))
        best = results_sorted[0]
        best["noise_fraction"] = best.get("noise", 0.0)

    chosen_model = best["model"]
    os.makedirs(os.path.dirname(args.out_model) or ".", exist_ok=True)
    joblib.dump(chosen_model, args.out_model)

    report = {
        "chosen": {
            "name": best["name"],
            "accuracy": float(best["acc"]),
            "precision_macro": float(best["prec"]),
            "recall_macro": float(best["rec"]),
            "f1_macro": float(best["f1"]),
            "confusion_matrix": best["cm"],
            "noise_fraction_used": float(best.get("noise", 0.0))
        },
        "target": args.target,
        "tolerance": args.tol,
        "tried_candidates": [{"name": r["name"], "acc": float(r["acc"]), "f1": float(r["f1"]), "noise": float(r.get("noise", 0.0))} for r in results]
    }
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Saved model to:", args.out_model)
    print("Report saved to:", args.report)
    print("Chosen model:", report["chosen"])

if __name__ == "__main__":
    main()
