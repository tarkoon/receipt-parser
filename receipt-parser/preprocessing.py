"""preprocessing.py — Image enhancement, PDF conversion, multi-page handling."""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image
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


def preprocess_receipt(image: np.ndarray) -> np.ndarray:
    """Enhancement pipeline: grayscale → deskew → denoise.
    Returns grayscale (NOT binary/thresholded).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Deskew via Hough line detection on text baselines
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=gray.shape[1] // 4, maxLineGap=10)
    if lines is not None and len(lines) > 0:
        angles = [np.degrees(np.arctan2(l[0][3] - l[0][1], l[0][2] - l[0][0]))
                  for l in lines]
        median_angle = np.median(angles)
        if abs(median_angle) > 0.5:
            h, w = gray.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)

    # Denoise — preserve grayscale, do NOT threshold
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    return denoised


def _fix_exif_orientation(image: np.ndarray, file_path: Path) -> np.ndarray:
    """Rotate image based on EXIF orientation tag. Critical for phone photos."""
    try:
        pil_img = Image.open(file_path)
        exif = pil_img.getexif()
        orientation = exif.get(0x0112)  # Orientation tag
        if orientation == 3:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif orientation == 6:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif orientation == 8:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception:
        pass  # No EXIF data or unreadable — continue with original
    return image


def load_image(file_path: Path) -> list[np.ndarray]:
    """Load an image file or PDF. Always returns a list of numpy arrays."""
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return pdf_to_images(str(file_path))

    if suffix not in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"):
        raise ValueError(f"Unsupported file format: {suffix}")

    image = cv2.imread(str(file_path))
    if image is None:
        raise ValueError(f"Failed to load image: {file_path}")

    image = _fix_exif_orientation(image, file_path)
    return [image]
