#README.md

# PAD Detection Backend (Flask)

## Overview
Backend service for your PAD Detection app. Accepts Doppler scan reports (PNG/JPG/PDF), extracts text via OCR (Tesseract), parses PSV/ABI values, then predicts PAD and severity either via a trained model or a rule-based fallback.

## Requirements
Install system packages:
- tesseract (Tesseract OCR)
- poppler (for PDF->image)

Ubuntu/Debian:
```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass 

& C:/PAD8/.venv/Scripts/Activate.ps1 

python tools/train_target_accuracy.py  --csv data/PAD_Patient_Data.csv  --out-model models/model_target.pkl  --report data/train_target_report.json  --target 0.78 --tol 0.02 --allow-noise --verbose

python tools/train_target_accuracy.py  --csv data/PAD_Test_Data.csv  --out-model models/model_target.pkl  --report data/train_target_report.json  --target 0.78 --tol 0.02 --allow-noise --verbose

python app.py
