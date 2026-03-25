"""ocr.py — Google Cloud Vision OCR backend for receipt text extraction.

Returns blocks in the format:
  [{"text": str, "confidence": float, "x": float, "y": float, "bbox": list}, ...]
"""

import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ── API call tracking ─────────────────────────────────────────────────

_USAGE_FILE = Path(__file__).parent / ".cloud_vision_usage.json"
_FREE_TIER_LIMIT = 1000
_WARNING_THRESHOLD = 100  # Warn when within this many of the limit


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
    """Make a single Cloud Vision API call. Returns the raw response."""
    from google.cloud import vision

    success, buf = cv2.imencode(".png", image)
    if not success:
        return None

    gcp_image = vision.Image(content=buf.tobytes())
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


def _fulltext_to_blocks(fulltext: str) -> list[dict]:
    """Convert fulltext string to block dicts (one per line)."""
    lines = [l.strip() for l in fulltext.split('\n') if l.strip()]
    return [{
        "text": line, "confidence": 0.9, "x": 0, "y": i * 50,
        "bbox": [[0, i*50], [500, i*50], [500, i*50+40], [0, i*50+40]],
    } for i, line in enumerate(lines)]


def run_cloud_vision(image: np.ndarray, client=None) -> list[dict]:
    """Run Google Cloud Vision OCR and return text blocks.

    Uses fulltext output which handles rotated images correctly.
    Caches results per image hash to avoid redundant API calls and
    ensure deterministic test results.
    """
    # Check cache first
    key = _ocr_cache_key(image)
    cache_path = _OCR_CACHE_DIR / f"{key}.txt"
    if cache_path.exists():
        fulltext = cache_path.read_text(encoding="utf-8")
        return _fulltext_to_blocks(fulltext)

    if client is None:
        client = init_cloud_vision()

    response1 = _call_cloud_vision(image, client)
    fulltext1 = _extract_fulltext_from_response(response1)

    if not fulltext1:
        return []

    # Retry and pick the better result
    response2 = _call_cloud_vision(image, client)
    fulltext2 = _extract_fulltext_from_response(response2)

    fulltext = fulltext1
    if fulltext2:
        has_yen1 = '¥' in fulltext1
        has_yen2 = '¥' in fulltext2
        if (has_yen2 and not has_yen1) or len(fulltext2) > len(fulltext1):
            fulltext = fulltext2

    # Save to cache
    _OCR_CACHE_DIR.mkdir(exist_ok=True)
    cache_path.write_text(fulltext, encoding="utf-8")

    return _fulltext_to_blocks(fulltext)


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
