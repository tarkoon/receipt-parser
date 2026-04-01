"""ocr.py — Google Cloud Vision OCR backend for receipt text extraction.

Returns blocks in the format:
  [{"text": str, "confidence": float, "x": float, "y": float, "bbox": list}, ...]
"""

import json
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ── OCR result dataclass ──────────────────────────────────────────────

@dataclass
class OCRResult:
    """Structured result from run_cloud_vision()."""
    blocks: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    retried: bool = False
    retry_reason: str | None = None
    source: str = "unknown"      # "cache", "fresh", "digital_pdf"
    chosen_text: str = ""


# ── API call tracking ─────────────────────────────────────────────────

_USAGE_FILE = Path(__file__).parent / ".cloud_vision_usage.json"
_FREE_TIER_LIMIT = 1000
_WARNING_THRESHOLD = 100  # Warn when within this many of the limit
_usage_lock = threading.Lock()


def _load_usage() -> dict:
    """Load monthly API call count from disk."""
    if _USAGE_FILE.exists():
        try:
            data = json.loads(_USAGE_FILE.read_text())
            if data.get("month") == datetime.now().strftime("%Y-%m"):
                return data
        except (json.JSONDecodeError, KeyError):
            pass
    return {"month": datetime.now().strftime("%Y-%m"), "calls": 0}


def _save_usage(data: dict):
    """Save monthly API call count to disk."""
    _USAGE_FILE.write_text(json.dumps(data))


def _track_api_call():
    """Increment the API call counter and warn if approaching the free tier limit."""
    with _usage_lock:
        usage = _load_usage()
        usage["calls"] += 1
        _save_usage(usage)

    remaining = _FREE_TIER_LIMIT - usage["calls"]
    if remaining <= _WARNING_THRESHOLD and remaining > 0:
        import sys
        print(f"WARNING: Cloud Vision API usage: {usage['calls']}/{_FREE_TIER_LIMIT} "
              f"this month ({remaining} remaining before paid tier)",
              file=sys.stderr)
    elif remaining <= 0:
        import sys
        print(f"WARNING: Cloud Vision API free tier exceeded! "
              f"{usage['calls']} calls this month (limit: {_FREE_TIER_LIMIT}). "
              f"Additional calls will be billed.",
              file=sys.stderr)


def get_api_usage() -> dict:
    """Return current month's API usage stats."""
    usage = _load_usage()
    return {
        "month": usage["month"],
        "calls": usage["calls"],
        "free_limit": _FREE_TIER_LIMIT,
        "remaining": max(0, _FREE_TIER_LIMIT - usage["calls"]),
    }


# ── Ollama GPU status ─────────────────────────────────────────────────

def get_ollama_gpu_status() -> dict | None:
    """Query Ollama for current model GPU/VRAM status.

    Returns dict with keys:
        model, size_bytes, size_vram_bytes, gpu_percent, full_gpu
    Returns None if Ollama is not running or no model loaded.
    """
    try:
        import urllib.request
        resp = urllib.request.urlopen("http://localhost:11434/api/ps", timeout=5)
        data = json.loads(resp.read().decode("utf-8"))
        models = data.get("models", [])
        if not models:
            return None
        m = models[0]
        size = m.get("size", 0)
        size_vram = m.get("size_vram", 0)
        gpu_pct = (size_vram / size * 100) if size > 0 else 0
        return {
            "model": m.get("name", "unknown"),
            "size_bytes": size,
            "size_vram_bytes": size_vram,
            "size_gb": round(size / 1024**3, 2),
            "vram_gb": round(size_vram / 1024**3, 2),
            "gpu_percent": round(gpu_pct, 1),
            "full_gpu": size_vram == size and size > 0,
        }
    except Exception:
        return None


# ── Google Cloud Vision backend ───────────────────────────────────────

def init_cloud_vision():
    """Initialize Google Cloud Vision client. Returns the client for reuse.

    Set GOOGLE_CLOUD_PROJECT env var if using Application Default Credentials.
    """
    try:
        from google.cloud import vision
        from google.api_core.client_options import ClientOptions

        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        kwargs = {}
        if project:
            kwargs["client_options"] = ClientOptions(quota_project_id=project)
        return vision.ImageAnnotatorClient(**kwargs)
    except ImportError:
        raise ImportError(
            "google-cloud-vision is not installed. "
            "Run: pip install google-cloud-vision"
        )


def _call_cloud_vision(image: np.ndarray, client):
    """Make a single Cloud Vision API call. Returns the raw response.

    Pins to builtin/stable model for deterministic OCR within a model cycle.
    Falls back to default (no model pin) if builtin/stable is unavailable.
    """
    from google.cloud import vision

    success, buf = cv2.imencode(".png", image)
    if not success:
        return None

    gcp_image = vision.Image(content=buf.tobytes())

    try:
        features = [vision.Feature(
            type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION,
            model="builtin/stable",
        )]
        request = vision.AnnotateImageRequest(
            image=gcp_image,
            features=features,
            image_context=vision.ImageContext(language_hints=["ja", "en"]),
        )
        response = client.annotate_image(request=request)
    except Exception as e:
        import sys
        print(f"WARNING: builtin/stable model unavailable ({e}), "
              f"falling back to default model", file=sys.stderr)
        response = client.document_text_detection(
            image=gcp_image,
            image_context=vision.ImageContext(language_hints=["ja", "en"]),
        )

    _track_api_call()

    if response.error.message:
        raise RuntimeError(f"Cloud Vision API error: {response.error.message}")

    return response


def _extract_blocks_from_response(response) -> list[dict]:
    """Extract paragraph-level blocks from a Cloud Vision response."""
    if not response or not response.full_text_annotation.pages:
        return []

    blocks = []
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                text = "".join(
                    "".join(s.text for s in word.symbols)
                    for word in paragraph.words
                )
                confidence = paragraph.confidence

                if confidence < 0.5 or not text.strip():
                    continue

                vertices = paragraph.bounding_box.vertices
                bbox = [[v.x, v.y] for v in vertices]
                top_y = min(v.y for v in vertices)
                left_x = min(v.x for v in vertices)

                blocks.append({
                    "text": text,
                    "confidence": confidence,
                    "x": left_x,
                    "y": top_y,
                    "bbox": bbox,
                })

    blocks.sort(key=lambda b: (b["y"], b["x"]))
    return blocks


def _extract_fulltext_from_response(response) -> str | None:
    """Extract the full pre-formatted text from a Cloud Vision response."""
    if not response or not response.text_annotations:
        return None
    return response.text_annotations[0].description


_OCR_CACHE_DIR = Path(__file__).parent / ".ocr_cache"


def _ocr_cache_key(image: np.ndarray) -> str:
    """Stable hash for an image to use as cache key."""
    import hashlib
    return hashlib.md5(image.tobytes()).hexdigest()


def compute_ocr_confidence(blocks: list[dict]) -> float:
    """Weighted average of block confidences, weighted by text length.

    Returns a document-level OCR quality score between 0.0 and 1.0.
    """
    if not blocks:
        return 0.0
    total_chars = sum(len(b["text"]) for b in blocks)
    if total_chars == 0:
        return 0.0
    return sum(b["confidence"] * len(b["text"]) for b in blocks) / total_chars


def _fulltext_to_blocks(fulltext: str) -> list[dict]:
    """Convert fulltext string to block dicts (one per line)."""
    lines = [l.strip() for l in fulltext.split('\n') if l.strip()]
    return [{
        "text": line, "confidence": 0.9, "x": 0, "y": i * 50,
        "bbox": [[0, i*50], [500, i*50], [500, i*50+40], [0, i*50+40]],
    } for i, line in enumerate(lines)]


_OCR_CONFIDENCE_RETRY_THRESHOLD = 0.75


def _pick_better_fulltext(ft1: str, ft2: str) -> str:
    """Select the better OCR fulltext from two API call results."""
    has_yen1 = '¥' in ft1
    has_yen2 = '¥' in ft2
    inline1 = len(re.findall(r'[\u3000-\u9fff].*¥|¥.*[\u3000-\u9fff]', ft1))
    inline2 = len(re.findall(r'[\u3000-\u9fff].*¥|¥.*[\u3000-\u9fff]', ft2))
    if has_yen2 and not has_yen1:
        return ft2
    elif inline2 > inline1 + 2:
        return ft2
    elif len(ft2) > len(ft1):
        return ft2
    return ft1


def run_cloud_vision(image: np.ndarray, client=None, *, skip_cache: bool = False) -> OCRResult:
    """Run Google Cloud Vision OCR and return structured result.

    Uses fulltext output which handles rotated images correctly.
    Caches results per image hash to avoid redundant API calls and
    ensure deterministic test results.

    Single-call by default. Makes a second call only if OCR confidence
    is below the retry threshold (0.75), cutting API usage ~50%.

    Args:
        skip_cache: If True, bypass cache read/write and always make fresh
            API calls. Used by benchmarks to test OCR variance.
    """
    # Check cache first (unless skipping)
    if not skip_cache:
        key = _ocr_cache_key(image)
        cache_path = _OCR_CACHE_DIR / f"{key}.txt"
        if cache_path.exists():
            fulltext = cache_path.read_text(encoding="utf-8")
            blocks = _fulltext_to_blocks(fulltext)
            return OCRResult(
                blocks=blocks,
                confidence=compute_ocr_confidence(blocks),
                source="cache",
                chosen_text=fulltext,
            )

    if client is None:
        client = init_cloud_vision()

    response1 = _call_cloud_vision(image, client)
    fulltext1 = _extract_fulltext_from_response(response1)

    if not fulltext1:
        return OCRResult(source="fresh")

    # Compute confidence from first call — retry only if low quality
    blocks1 = _extract_blocks_from_response(response1)
    confidence1 = compute_ocr_confidence(blocks1) if blocks1 else 0.0

    fulltext = fulltext1
    retried = False
    retry_reason = None

    if confidence1 < _OCR_CONFIDENCE_RETRY_THRESHOLD:
        retried = True
        retry_reason = f"confidence {confidence1:.3f} < {_OCR_CONFIDENCE_RETRY_THRESHOLD}"
        response2 = _call_cloud_vision(image, client)
        fulltext2 = _extract_fulltext_from_response(response2)
        if fulltext2:
            fulltext = _pick_better_fulltext(fulltext1, fulltext2)

    # Save to cache (unless skipping)
    if not skip_cache:
        _OCR_CACHE_DIR.mkdir(exist_ok=True)
        cache_path = _OCR_CACHE_DIR / f"{_ocr_cache_key(image)}.txt"
        cache_path.write_text(fulltext, encoding="utf-8")

    blocks = _fulltext_to_blocks(fulltext)
    return OCRResult(
        blocks=blocks,
        confidence=compute_ocr_confidence(blocks),
        retried=retried,
        retry_reason=retry_reason,
        source="fresh",
        chosen_text=fulltext,
    )


# ── Shared text grouping ──────────────────────────────────────────────

def blocks_to_structured_text(blocks: list[dict]) -> str:
    """Convert spatial blocks to line-grouped text for the LLM.
    y_tolerance is calculated dynamically from the median bounding box height.
    """
    if not blocks:
        return ""

    heights = []
    for block in blocks:
        bbox = block["bbox"]
        h = abs(bbox[2][1] - bbox[0][1])
        if h > 0:
            heights.append(h)

    if heights:
        median_height = sorted(heights)[len(heights) // 2]
        y_tolerance = max(10, int(median_height * 0.4))
    else:
        y_tolerance = 15

    lines = []
    current_line = []
    current_y = None

    for block in blocks:
        if current_y is None or abs(block["y"] - current_y) < y_tolerance:
            current_line.append(block["text"])
            current_y = block["y"] if current_y is None else current_y
        else:
            lines.append("  ".join(current_line))
            current_line = [block["text"]]
            current_y = block["y"]

    if current_line:
        lines.append("  ".join(current_line))

    return "\n".join(lines)
