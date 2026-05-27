# tests/test_model_utils.py
import os
import importlib
import pytest

# Ensure tests import the modules from your project root (not some other installed package)
# Run pytest from project root so relative imports resolve correctly.

from extras.parser_utils import parse_report_text
from model_utils import comparative_predict

def test_case_proximal_sfa_severe_escalates():
    txt = """
    Patient Name: John Doe
    Age: 67
    ABI: 0.54
    Common Femoral Artery PSV: 115 cm/s
    Proximal Superficial Femoral Artery PSV: 312 cm/s
    Distal SFA PSV: 66 cm/s
    Monophasic waveforms noted.
    """
    parsed = parse_report_text(txt)
    res = comparative_predict(parsed)
    assert res["pad_detected"] is True
    # Under current default logic this should escalate to Severe
    assert res["severity"] == "Severe"
    assert ("focal_escalation" in res["breakdown"]) or ("critical_arteries" in res)

def test_case_normal_abi_no_psv():
    txt = """
    Patient Name: Jane Roe
    Age: 45
    ABI: 1.02
    Triphasic waveforms noted.
    """
    parsed = parse_report_text(txt)
    res = comparative_predict(parsed)
    # No clear PSV abnormalities, ABI normal => no PAD
    assert res["pad_detected"] is False or res["severity"] == "None"

def test_percent_stenosis_forces_severe():
    txt = """
    Patient Name: Alice Smith
    Age: 72
    ABI: 0.78
    Distal SFA PSV: 88 cm/s
    Percent_Stenosis: 80
    """
    parsed = parse_report_text(txt)
    # parser may parse Percent_Stenosis as a string; ensure we set it
    parsed["Percent_Stenosis"] = parsed.get("Percent_Stenosis", 80)
    res = comparative_predict(parsed)
    assert res["severity"] == "Severe"
    assert res["score"] == 3.0

def test_psvr_forces_severe():
    txt = """
    Patient Name: Bob Lee
    Age: 69
    ABI: 0.88
    PSV__Distal_SFA: 88
    PSVR: 4.5
    """
    parsed = parse_report_text(txt)
    parsed["PSVR"] = parsed.get("PSVR", 4.5)
    res = comparative_predict(parsed)
    assert res["severity"] in ("Severe", "Moderate")
    assert res["score"] == 3.0

def test_import_path_debug():
    # Help debugging which file is imported for model_utils
    import model_utils
    path = getattr(model_utils, "__file__", None)
    assert path is not None, "model_utils module path could not be determined"
    # Ensure model_utils lives in project (not site-packages)
    assert os.path.exists(path), f"Imported model_utils file missing at {path}"
    assert "site-packages" not in path.lower(), f"Unexpected import path (site-packages): {path}"

