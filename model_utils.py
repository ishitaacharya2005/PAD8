#model_utils.py
"""
Clinical comparative predictor for PAD severity.
Clean, self-contained version — paste over your existing model_utils.py
"""

import joblib
import numpy as np
import math
from typing import Dict, Any
import os
import pandas as pd

MODEL_PATH = "models/model.pkl"

def load_model(path=MODEL_PATH):
    path = path or MODEL_PATH
    if not os.path.exists(path):
        print("Model file not found at", path)
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        print("Failed to load model:", e)
        return None

# Replace your existing format_features_for_model(...) with this


def format_features_for_model(parsed_dict, default_features=None):
    """
    Build a DataFrame (1 row) with columns matching the training feature list.
    """
    if default_features is None:
        feat_path = os.path.join("models", "feature_names.pkl")
        if os.path.exists(feat_path):
            try:
                default_features = joblib.load(feat_path)
            except Exception:
                default_features = []
        else:
            default_features = []

    if not default_features:
        return pd.DataFrame([{}])

    row = {}
    for col in default_features:
        v = parsed_dict.get(col)
        try:
            if v is None or (isinstance(v, str) and v.strip() == ""):
                row[col] = np.nan
            else:
                if isinstance(v, (int, float, np.number)):
                    row[col] = float(v)
                else:
                    try:
                        row[col] = float(str(v))
                    except Exception:
                        row[col] = np.nan
        except Exception:
            row[col] = np.nan

    df = pd.DataFrame([row], columns=default_features)
    return df



def predict_with_model(model, parsed):
    X = format_features_for_model(parsed, default_features=None)
    pred_label = int(model.predict(X)[0])
    severity_map = {0: "None", 1: "Mild", 2: "Moderate", 3: "Severe"}
    pad_detected = pred_label != 0
    severity = severity_map.get(pred_label, "Unknown")
    probs = None
    try:
        probs = model.predict_proba(X)[0].tolist()
    except Exception:
        probs = None
    return {"pad_detected": bool(pad_detected), "severity": severity, "proba": probs, "label": pred_label}


# ------------------------
# PSV thresholds (HIGH PSV -> MORE SEVERE)
# Each tuple: (mild_threshold, moderate_threshold, severe_threshold)
# Interpretation:
#   PSV < mild_threshold -> 0 (None)
#   mild <= PSV < moderate -> 1 (Mild)
#   moderate <= PSV < severe -> 2 (Moderate)
#   PSV >= severe -> 3 (Severe)
ARTERY_PSV_THRESHOLDS = {
    "Common_Femoral_Artery": (90.0, 150.0, 250.0),
    "Profundus_Femoris": (90.0, 150.0, 250.0),
    "Proximal_Superficial_Femoral_Artery": (90.0, 150.0, 230.0),
    "Mid_SFA": (80.0, 120.0, 200.0),
    "Distal_SFA": (80.0, 110.0, 180.0),
    "Popliteal_Artery": (80.0, 120.0, 200.0),
    "Peroneal_Artery": (60.0, 90.0, 150.0),
    "Posterior_Tibial_Artery": (60.0, 90.0, 150.0),
    "Anterior_Tibial_Artery": (60.0, 90.0, 150.0),
    "generic": (90.0, 150.0, 230.0)
}

def _score_from_abi(abi):
    try:
        if abi is None:
            return None
        a = float(abi)
        if a > 1.30:
            return 9
        if a >= 0.90:
            return 0
        if 0.70 <= a < 0.90:
            return 1
        if 0.40 <= a < 0.70:
            return 2
        return 3
    except Exception:
        return None

def _score_from_psv_value_generic(psv):
    try:
        if psv is None:
            return None
        p = float(psv)
        # generic thresholds low->high: <90 normal, 90-150 mild, 150-230 moderate, >=230 severe
        if p < 90:
            return 0
        if p < 150:
            return 1
        if p < 230:
            return 2
        return 3
    except Exception:
        return None

def _score_from_psv_value_by_artery(psv, artery_key):
    if psv is None:
        return None
    try:
        p = float(psv)
    except Exception:
        return None

    key = str(artery_key).replace("PSV__", "").strip().lower().replace(" ", "_")

    for art, thr in ARTERY_PSV_THRESHOLDS.items():
        art_norm = art.lower().replace(" ", "_")
        if art_norm == key or art_norm in key or key in art_norm:
            mild_thr, mod_thr, sev_thr = thr
            if p < mild_thr:
                return 0
            if p < mod_thr:
                return 1
            if p < sev_thr:
                return 2
            return 3

    return _score_from_psv_value_generic(psv)

def _score_from_waveforms(parsed):
    try:
        mono = int(parsed.get("waveform_monophasic_count", 0) or 0)
        bi = int(parsed.get("waveform_biphasic_count", 0) or 0)
        tri = int(parsed.get("waveform_triphasic_count", 0) or 0)
    except Exception:
        mono = bi = tri = 0
    absent_flag = parsed.get("Absent_Flow", False) or parsed.get("No_Flow", False)
    total = mono + bi + tri
    if absent_flag:
        return 3
    if total == 0:
        return None
    if tri >= max(bi, mono) and tri >= 2:
        return 0
    if bi >= max(tri, mono):
        return 1
    if mono >= 1 and mono < 3:
        return 2
    if mono >= 3:
        return 3
    return 1

def _map_score_to_label(score):
    if score is None:
        return "Unknown"
    try:
        if int(score) == 9:
            return "Non-compressible (ABI unreliable)"
    except Exception:
        pass
    rounded = int(round(float(score)))
    mapping = {0: "None", 1: "Mild", 2: "Moderate", 3: "Severe"}
    return mapping.get(max(0, min(3, rounded)), "Unknown")

def comparative_predict(parsed: Dict[str, Any], weights: Dict[str, float] = None):
    if weights is None:
        weights = {"abi": 0.50, "psv": 0.35, "waveform": 0.15}

    abi_raw = parsed.get("ABI")
    abi_score = _score_from_abi(abi_raw)

    note = None
    if abi_score == 9:
        note = "ABI > 1.30 (non-compressible) — ABI considered unreliable; using PSV/waveform primarily."
        abi_score_val = None
    else:
        abi_score_val = None if abi_score is None else float(abi_score)

    psv_scores = []
    psv_scores_map = {}
    for k, v in parsed.items():
        if isinstance(k, str) and k.startswith("PSV__"):
            sc = _score_from_psv_value_by_artery(v, k.replace("PSV__", ""))
            if sc is not None:
                psv_scores.append(float(sc))
                psv_scores_map[k] = int(sc)
    psv_avg = None
    if psv_scores:
        psv_avg = sum(psv_scores) / len(psv_scores)

    wav_score = _score_from_waveforms(parsed)

    comp_weights = weights.copy()
    if abi_score is None or abi_score == 9:
        usable = []
        if psv_avg is not None:
            usable.append("psv")
        if wav_score is not None:
            usable.append("waveform")
        if not usable:
            return {"pad_detected": False, "severity": "Unknown", "score": None, "breakdown": {}, "note": "Insufficient data (no ABI, PSV or waveform)."}
        total_orig = sum(weights[k] for k in usable)
        for k in comp_weights.keys():
            if k not in usable:
                comp_weights[k] = 0.0
        for k in usable:
            comp_weights[k] = weights[k] / total_orig

    weighted_sum = 0.0
    total_weight_used = 0.0
    breakdown = {
        "abi_score": None, "abi_weight": comp_weights.get("abi", 0.0),
        "psv_avg_score": None, "psv_weight": comp_weights.get("psv", 0.0),
        "waveform_score": None, "waveform_weight": comp_weights.get("waveform", 0.0)
    }

    if abi_score is not None and abi_score != 9:
        weighted_sum += float(abi_score) * comp_weights.get("abi", 0.0)
        total_weight_used += comp_weights.get("abi", 0.0)
        breakdown["abi_score"] = float(abi_score)

    if psv_avg is not None:
        weighted_sum += float(psv_avg) * comp_weights.get("psv", 0.0)
        total_weight_used += comp_weights.get("psv", 0.0)
        breakdown["psv_avg_score"] = float(psv_avg)

    if wav_score is not None:
        weighted_sum += float(wav_score) * comp_weights.get("waveform", 0.0)
        total_weight_used += comp_weights.get("waveform", 0.0)
        breakdown["waveform_score"] = float(wav_score)

    if total_weight_used == 0:
        return {"pad_detected": False, "severity": "Unknown", "score": None, "breakdown": breakdown, "note": "No usable components for scoring."}

    score = weighted_sum / total_weight_used

    # focal escalation & overrides
    critical_arteries = [k for k, s in psv_scores_map.items() if s == 3]
    num_critical = len(critical_arteries)

    pct = parsed.get("Percent_Stenosis")
    if pct is not None:
        try:
            if float(pct) >= 75:
                score = 3.0
                breakdown["override"] = "Percent_Stenosis >= 75%"
                severity = _map_score_to_label(score)
                res = {"pad_detected": True, "severity": severity, "score": float(score), "breakdown": breakdown}
                if critical_arteries:
                    res["critical_arteries"] = critical_arteries
                res["note"] = "Forced Severe: percent stenosis >= 75%"
                return res
        except Exception:
            pass

    psvr = parsed.get("PSVR")
    if psvr is not None:
        try:
            psvr_f = float(psvr)
            if psvr_f >= 4.0:
                score = 3.0
                breakdown["override"] = "PSVR >= 4.0"
                severity = _map_score_to_label(score)
                res = {"pad_detected": True, "severity": severity, "score": float(score), "breakdown": breakdown}
                if critical_arteries:
                    res["critical_arteries"] = critical_arteries
                res["note"] = "Forced Severe: PSVR >= 4.0"
                return res
        except Exception:
            pass

    if parsed.get("Critical_Remark", False):
        breakdown["critical_remark"] = True

    if num_critical >= 1:
        if num_critical >= 2:
            score = max(score, 3.0)
            breakdown["focal_escalation"] = f"{num_critical} arteries severe"
        else:
            single_art = critical_arteries[0]
            abi_agrees = (abi_score is not None and abi_score >= 2)
            wav_agrees = (wav_score is not None and wav_score >= 2)
            # current rule: escalate if ABI OR waveform agrees (conservative)
            if abi_agrees and wav_agrees:
                score = max(score, 3.0)
                breakdown["focal_escalation"] = f"Single artery severe ({single_art}), ABI/waveform concur -> escalate"

            else:
                score = max(score, 3.0)
                breakdown["focal_escalation"] = f"Single artery severe ({single_art}), ABI normal -> escalated but review recommended"
                breakdown["need_review"] = True
                breakdown["critical_arteries"] = critical_arteries

    severity = _map_score_to_label(score)
    pad_detected = severity not in ("None", "Unknown", "Non-compressible (ABI unreliable)")
    res = {
        "pad_detected": bool(pad_detected),
        "severity": severity,
        "score": float(score),
        "breakdown": breakdown
    }
    if note:
        res["note"] = note
    if "critical_arteries" in breakdown:
        res["critical_arteries"] = breakdown["critical_arteries"]
    if "need_review" in breakdown:
        res["need_review"] = breakdown["need_review"]
    return res

# --- end of file ---
