"""
verify_ocr_setup.py
-------------------
Runs a diagnostic check for Tesseract + Poppler setup.
- Verifies environment and paths
- Prints detected versions
- Tests OCR on a sample image or PDF (first page)
"""

import os
import sys
import tempfile
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import shutil

print("="*70)
print("🔍 OCR Setup Diagnostic Tool")
print("="*70)
print(f"Python: {sys.executable}")
print(f"Python version: {sys.version.split()[0]}")

# -------------------------------
# 1️⃣ Check Tesseract
# -------------------------------
print("\n[1] Checking Tesseract installation...")
tess_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Try to use environment override if user set one
if os.environ.get("TESSERACT_CMD"):
    tess_path = os.environ["TESSERACT_CMD"]

if os.path.exists(tess_path):
    pytesseract.pytesseract.tesseract_cmd = tess_path
else:
    print(f"⚠️  Tesseract not found at {tess_path}")
    print("    -> Please verify the installation path or set it manually:")
    print("       pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'")
    print("    (in ocr_utils.py)")
    sys.exit(1)

try:
    version = pytesseract.get_tesseract_version()
    print(f"✅ Tesseract version detected: {version}")
    print(f"✅ Executable: {tess_path}")
except Exception as e:
    print(f"❌ Error getting Tesseract version: {e}")
    sys.exit(1)

# -------------------------------
# 2️⃣ Check Poppler (pdf2image)
# -------------------------------
print("\n[2] Checking Poppler path...")
poppler_env = os.environ.get("POPPLER_PATH", None)
if not poppler_env:
    print("⚠️  POPPLER_PATH not set.")
    print("    Set it before running app.py like this in PowerShell:")
    print("    $env:POPPLER_PATH = 'C:\\Program Files\\poppler\\poppler-25.07.0\\Library\\bin'")
else:
    if os.path.exists(poppler_env):
        print(f"✅ POPPLER_PATH detected: {poppler_env}")
    else:
        print(f"❌ POPPLER_PATH set but directory not found: {poppler_env}")

# -------------------------------
# 3️⃣ Test OCR on a sample image or PDF
# -------------------------------
print("\n[3] Testing OCR functionality...")

# Find any sample file
sample_files = [f for f in os.listdir(".") if f.lower().endswith((".png", ".jpg", ".jpeg", ".pdf"))]
sample_path = sample_files[0] if sample_files else None

if not sample_path:
    print("⚠️  No sample image/PDF found in current folder.")
    print("    Place one here (e.g., 'moderate1.pdf') and rerun this script.")
    sys.exit(0)

print(f"🔸 Using sample file: {sample_path}")

try:
    if sample_path.lower().endswith(".pdf"):
        print("   → Converting first page of PDF to image...")
        kwargs = {}
        if poppler_env:
            kwargs["poppler_path"] = poppler_env
        pages = convert_from_path(sample_path, dpi=200, **kwargs)
        img = pages[0]
    else:
        img = Image.open(sample_path).convert("RGB")

    print("   → Running Tesseract OCR...")
    text = pytesseract.image_to_string(img, config="--oem 3 --psm 6", lang="eng")

    # Display a short preview of the OCR output
    print("\n📝 OCR output preview (first 400 characters):")
    print("-"*60)
    print(text[:400])
    print("-"*60)

    if len(text.strip()) > 20:
        print("✅ OCR appears to be functioning correctly.")
    else:
        print("⚠️  OCR output is very short. Check image clarity or resolution.")

except Exception as e:
    print(f"❌ OCR test failed: {e}")

# -------------------------------
# 4️⃣ Cleanup / finish
# -------------------------------
print("\n[4] Summary:")
print(f"   Tesseract path : {tess_path}")
print(f"   POPPLER_PATH   : {poppler_env or 'Not set'}")
print("   Sample tested  :", sample_path or "None")
print("\n✅ If all steps show green checkmarks, your OCR pipeline is ready.")
print("="*70)
