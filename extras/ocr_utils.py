# ocr_utils.py
"""
OCR utilities for PAD project.

- Set environment variable POPPLER_PATH if poppler is not on PATH (Windows).
  Example (PowerShell):
    $env:POPPLER_PATH = "C:\Program Files\poppler\bin"

- Produces debug artifacts next to the uploaded file:
    <uploaded_filename>.debug_preprocessed.png
    <uploaded_filename>.ocr.txt
"""

import os
import tempfile
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import cv2
import numpy as np


# Optional: set POPPLER_PATH env var to point to poppler's bin folder on Windows,
# or leave unset on Linux/macOS where poppler is typically on PATH.
POPPLER_PATH = r"C:\Program Files\poppler\poppler-25.07.0\Library\bin"



def preprocess_image_cv(img, upscale=3.0):
    """
    Preprocess an image for OCR:
      - upscale (Lanczos) to improve small-font OCR
      - convert to gray, denoise (bilateral), sharpen
      - adaptive threshold + small morphology
      - deskew
    Returns a uint8 grayscale image.
    """
    try:
        # Accept either path or numpy BGR array
        if isinstance(img, str):
            arr = cv2.imread(img)
            if arr is None:
                raise FileNotFoundError(f"Unable to open image: {img}")
            img_bgr = arr
        else:
            img_bgr = img

        # High-quality resize via PIL
        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        w, h = pil.size
        pil = pil.resize((int(w * upscale), int(h * upscale)), Image.LANCZOS)
        img_rgb = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        # stronger bilateral denoising to preserve edges while removing noise
        gray = cv2.bilateralFilter(gray, d=11, sigmaColor=90, sigmaSpace=90)

        # slight sharpening kernel to emphasize printed text
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        gray = cv2.filter2D(gray, -1, kernel)

        # adaptive threshold (works better under uneven lighting)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 15, 9)

        # small morphological opening to remove tiny specks
        kernel = np.ones((1, 1), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

        # deskew using text contour minAreaRect
        coords = np.column_stack(np.where(gray < 255))
        if coords.size:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h0, w0) = gray.shape[:2]
            M = cv2.getRotationMatrix2D((w0 // 2, h0 // 2), angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w0, h0), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return gray
    except Exception:
        # fallback: convert to simple grayscale
        try:
            if isinstance(img, str):
                arr = cv2.imread(img)
            else:
                arr = img
            return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        except Exception:
            raise

def image_to_text(img_path):
    """
    OCR a single image and save debug preprocessed image + OCR text next to the file.
    Returns extracted text (string).
    """
    try:
        pil = Image.open(img_path).convert("RGB")
        arr = np.array(pil)
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        arr = cv2.imread(img_path)
        if arr is None:
            raise FileNotFoundError(f"Unable to open image {img_path}")

    processed = preprocess_image_cv(arr, upscale=3.0)

    # Save debug preprocessed image next to the uploaded file
    try:
        img_dir = os.path.dirname(img_path) or "."
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        debug_path = os.path.join(img_dir, f"{img_name}.debug_preprocessed.png")
        cv2.imwrite(debug_path, processed)
    except Exception:
        pass

    # Try multiple Tesseract psm configurations; keep the longest plausible text
    pil2 = Image.fromarray(processed)
    configs = ['--oem 3 --psm 6', '--oem 3 --psm 4', '--oem 3 --psm 3']
    text = ""
    for cfg in configs:
        try:
            t = pytesseract.image_to_string(pil2, config=cfg, lang='eng')
            # choose longest non-empty result (heuristic)
            if len(t.strip()) > len(text.strip()):
                text = t
        except Exception:
            continue

    # Save OCR text next to the uploaded file for debugging
    try:
        with open(img_path + ".ocr.txt", "w", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        pass

    return text

from pdf2image import convert_from_path, convert_from_bytes

def extract_text_from_pdf(pdf_path, dpi=300):
    """
    Robust PDF -> list of page texts.
    Tries:
      1) convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
      2) convert_from_bytes(open(pdf_path,'rb').read(), poppler_path=POPPLER_PATH)
    Returns list of page strings or raises with helpful message.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    kwargs = {}
    if POPPLER_PATH:
        kwargs["poppler_path"] = POPPLER_PATH

    # 1) try convert_from_path first
    try:
        images = convert_from_path(pdf_path, dpi=dpi, **kwargs)
    except Exception as e_path:
        # fallback: try convert_from_bytes (some PDFs are readable this way)
        try:
            with open(pdf_path, "rb") as f:
                data = f.read()
            images = convert_from_bytes(data, dpi=dpi, **kwargs)
        except Exception as e_bytes:
            # Build a useful error message
            raise RuntimeError(
                "pdf2image failed to convert PDF. "
                f"convert_from_path error: {e_path}; convert_from_bytes error: {e_bytes}. "
                "Check POPPLER_PATH, file permissions, and file integrity."
            ) from e_bytes

    texts = []
    for img in images:
        # Save to temp file and OCR
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f.name, "PNG")
            try:
                t = image_to_text(f.name)
                texts.append(t)
            finally:
                try:
                    os.remove(f.name)
                except Exception:
                    pass
    return texts

def extract_text_from_file(path):
    """Auto-detect PDF vs image and return list of page texts."""
    path_lower = path.lower()
    if path_lower.endswith(".pdf"):
        return extract_text_from_pdf(path)
    else:
        # Single image file
        t = image_to_text(path)
        return [t]


def extract_text_from_file(path):
    """
    Auto-detect PDF vs image and return list of page texts.
    """
    if not path:
        return []
    path_lower = path.lower()
    if path_lower.endswith(".pdf"):
        return extract_text_from_pdf(path)
    else:
        return [image_to_text(path)]
