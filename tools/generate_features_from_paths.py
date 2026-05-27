#!/usr/bin/env python3
"""
Generate a features CSV for evaluation by extracting parsed fields from each report path listed
in a CSV (expected columns: file_path,label). Output will include columns matching common model
feature names and PSV names.

Usage:
  python tools/generate_features_from_paths.py --input data/eval_labels_clean.csv --out data/eval_features.csv

Run this from your project root (where ocr_pad_extractor.py sits) so filesystem-relative imports work.
"""
import argparse
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import importlib
import importlib.util
import importlib.machinery
import traceback

def try_import_extractor_and_parser():
    """
    Try multiple ways to import:
      - import ocr_pad_extractor (module in project root)
      - import extras.parser_utils (if extras is a package)
      - load from files extras/parser_utils.py or ./ocr_pad_extractor.py with SourceFileLoader
    Returns tuple (extractor_callable_or_None, parser_utils_module_or_None)
    """
    extractor = None
    parser_utils = None

    # 1) try normal imports
    try:
        import ocr_pad_extractor as oex
        extractor = getattr(oex, "extract_from_report", None)
        if extractor:
            print("Imported extractor via `import ocr_pad_extractor`.")
    except Exception:
        pass

    try:
        # prefer package-style if extras is a package
        import extras.parser_utils as putils
        parser_utils = putils
        print("Imported parser via `import extras.parser_utils`.")
    except Exception:
        try:
            from extras import parser_utils as putils2
            parser_utils = putils2
            print("Imported parser via `from extras import parser_utils`.")
        except Exception:
            pass

    # 2) If not found, attempt to load from filesystem explicitly
    cwd = Path.cwd()
    root_extractor_path = cwd / "ocr_pad_extractor.py"
    extras_parser_path = cwd / "extras" / "parser_utils.py"

    if extractor is None and root_extractor_path.exists():
        try:
            loader = importlib.machinery.SourceFileLoader("ocr_pad_extractor_fs", str(root_extractor_path))
            spec = importlib.util.spec_from_loader(loader.name, loader)
            mod = importlib.util.module_from_spec(spec)
            loader.exec_module(mod)
            extractor = getattr(mod, "extract_from_report", None)
            if extractor:
                print(f"Loaded extractor from file: {root_extractor_path}")
        except Exception as e:
            print("Failed to load extractor from file:", root_extractor_path, e)
            traceback.print_exc()

    if parser_utils is None and extras_parser_path.exists():
        try:
            loader = importlib.machinery.SourceFileLoader("extras_parser_utils_fs", str(extras_parser_path))
            spec = importlib.util.spec_from_loader(loader.name, loader)
            mod = importlib.util.module_from_spec(spec)
            loader.exec_module(mod)
            parser_utils = mod
            print(f"Loaded parser_utils from file: {extras_parser_path}")
        except Exception as e:
            print("Failed to load parser_utils from file:", extras_parser_path, e)
            traceback.print_exc()

    # 3) final fallback: try to import relative extras if not package
    if parser_utils is None:
        extras_init = cwd / "extras" / "__init__.py"
        if extras_init.exists():
            try:
                import extras as extras_pkg
                if hasattr(extras_pkg, "parser_utils"):
                    parser_utils = extras_pkg.parser_utils
                    print("Imported parser_utils via extras package attribute.")
            except Exception:
                pass

    return extractor, parser_utils

def normalize_key(k):
    return str(k).replace(" ", "_").replace("/", "_").replace("-", "_").strip().lower()

# Feature set to produce (model-used names + human friendly)
BASE_FEATURES = [
    "Patient ID","Age","ABI",
    "Common Femoral Artery PSV","Profundus Femoris PSV","Proximal SFA PSV","Mid SFA PSV","Distal SFA PSV",
    "Popliteal Artery PSV","Peroneal / Posterior Tibial Artery PSV","Anterior Tibial Artery PSV",
    "waveform_monophasic_count","waveform_biphasic_count","waveform_triphasic_count",
    "mean_PSV",
    # PSV__ style fallbacks
    "PSV__Common_Femoral_Artery","PSV__Profundus_Femoris","PSV__Proximal_Superficial_Femoral_Artery",
    "PSV__Mid_SFA","PSV__Distal_SFA","PSV__Popliteal_Artery",
    "PSV__Peroneal___Posterior_Tibial_Artery","PSV__Anterior_Tibial_Artery"
]

def row_from_parsed(parsed):
    r = {}
    parsed_keys = {normalize_key(k): k for k in parsed.keys()} if parsed else {}
    for f in BASE_FEATURES:
        # direct
        if f in parsed:
            r[f] = parsed.get(f)
            continue
        # normalized match
        nf = normalize_key(f)
        if nf in parsed_keys:
            r[f] = parsed.get(parsed_keys[nf])
            continue
        # try to match partials for PSV names
        if "psv" in f.lower():
            # look for any parsed key that includes artery name tokens
            target = normalize_key(f).replace("psv__", "").replace("psv", "").replace("common_femoral_artery", "common femoral").strip()
            matched = None
            for pk in parsed.keys():
                if target and target in normalize_key(pk):
                    matched = parsed.get(pk)
                    break
            if matched is not None:
                r[f] = matched
                continue
        # default None
        r[f] = None
    return r

def extract_parsed_from_path(fp, extractor, parser_utils):
    parsed = {}
    full_text = None
    conclusion = None
    p = Path(fp)
    try:
        if p.exists():
            suffix = p.suffix.lower()
            if suffix in (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                if extractor:
                    parsed, full_text, conclusion = extractor(str(p))
                else:
                    # try to fallback: use parser_utils by reading text from file if possible
                    if parser_utils and hasattr(parser_utils, "parse_report_text"):
                        txt = ""
                        if suffix == ".pdf":
                            # try to extract text with pdf2image + OCR fallback if parser_utils can't parse raw HTML
                            try:
                                from pdf2image import convert_from_path
                                import cv2
                                pages = convert_from_path(str(p), dpi=200)
                                txt_pages = []
                                import pytesseract
                                for pp in pages:
                                    import numpy as np
                                    img = np.array(pp)
                                    txt_pages.append(pytesseract.image_to_string(img))
                                txt = "\n\n".join(txt_pages)
                            except Exception:
                                txt = ""
                        else:
                            try:
                                txt = p.read_text(encoding="utf-8", errors="ignore")
                            except Exception:
                                txt = ""
                        parsed = parser_utils.parse_report_text(txt or "")
                    else:
                        parsed = {}
            else:
                # assume text/html file -> parse text and call parser utils
                try:
                    txt = p.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    txt = ""
                if parser_utils and hasattr(parser_utils, "parse_report_text"):
                    parsed = parser_utils.parse_report_text(txt)
                else:
                    parsed = {}
        else:
            print("File not found:", fp)
            parsed = {}
    except Exception as e:
        print("Error extracting/parsing", fp, ":", e)
        traceback.print_exc()
        parsed = {}
    return parsed, full_text, conclusion

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with file_path and label columns")
    ap.add_argument("--out", required=True, help="Output features CSV path")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print("Input CSV not found:", args.input)
        sys.exit(2)

    extractor, parser_utils = try_import_extractor_and_parser()
    if extractor is None and parser_utils is None:
        print("Warning: Neither ocr_pad_extractor.extract_from_report nor extras.parser_utils available.")
        print("You need at least one to extract features. Aborting.")
        sys.exit(2)

    df = pd.read_csv(args.input)
    if "file_path" not in df.columns:
        print("Input CSV must contain 'file_path' column pointing to report files.")
        sys.exit(2)

    rows = []
    for i, r in df.iterrows():
        fp = r["file_path"]
        label = r.get("label", None)
        print(f"[{i+1}/{len(df)}] Processing: {fp}")
        parsed, full_text, conclusion = extract_parsed_from_path(fp, extractor, parser_utils)
        feats = row_from_parsed(parsed or {})
        feats["file_path"] = fp
        feats["label"] = label
        # store small parsed snapshot for debugging (stringified)
        try:
            feats["_parsed_snapshot"] = json.dumps(parsed, ensure_ascii=False)
        except Exception:
            feats["_parsed_snapshot"] = str(parsed)
        # include conclusion (if extractor produced it)
        if conclusion:
            feats["_conclusion"] = json.dumps(conclusion, ensure_ascii=False)
        rows.append(feats)

    out_df = pd.DataFrame(rows)
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print("Wrote features CSV to", args.out)
    print("Columns written:", list(out_df.columns))

if __name__ == "__main__":
    main()
