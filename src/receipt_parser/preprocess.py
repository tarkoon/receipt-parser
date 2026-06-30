"""preprocess.py — File loading, PDF conversion, and multi-page handling."""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps
import pdfplumber
from pdf2image import convert_from_path


def pdf_to_images(pdf_path: str, dpi: int = 300) -> list[np.ndarray]:
    """Convert all pages of a PDF to images."""
    pil_images = convert_from_path(pdf_path, dpi=dpi)
    return [np.array(img)[:, :, ::-1] for img in pil_images]  # RGB → BGR


def try_extract_text_layer(pdf_path: str) -> str | None:
    """Extract embedded text from a digital PDF. Returns None for scanned PDFs."""
    with pdfplumber.open(pdf_path) as pdf:
        pages_text = []
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(f"--- PAGE {i + 1} ---\n{text}")
        full_text = "\n".join(pages_text)
    return full_text.strip() if len(full_text.strip()) > 50 else None


def _load_oriented_image(file_path: Path) -> np.ndarray | None:
    """Load image pixels with EXIF orientation applied."""
    try:
        with Image.open(file_path) as pil_img:
            rgb = ImageOps.exif_transpose(pil_img).convert("RGB")
    except Exception:
        return cv2.imread(str(file_path))
    return np.array(rgb)[:, :, ::-1]


def load_image(file_path: Path) -> list[np.ndarray]:
    """Load an image file or PDF. Always returns a list of numpy arrays."""
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return pdf_to_images(str(file_path))

    if suffix not in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"):
        raise ValueError(f"Unsupported file format: {suffix}")

    image = _load_oriented_image(file_path)
    if image is None:
        raise ValueError(f"Failed to load image: {file_path}")

    return [image]
