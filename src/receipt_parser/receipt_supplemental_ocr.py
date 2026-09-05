"""Field-only recovery from a sealed two-tile supplemental OCR result.

The evidence sidecar is deliberately raw: source identity, mapped word layout,
its derived merged OCR text, and exactly two tile texts.  It never stores
fixture IDs or proposed answers.
Location changes require one OCR-backed header extension.  Tax changes require
unambiguous reduced-rate marker rows whose item sums reconcile to printed 8%
and 10% bases.  Every rejected proposal leaves the receipt unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import threading
from contextlib import contextmanager
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np

from .ocr import (
    _OCR_CACHE_DIR,
    _call_cloud_vision,
    _extract_fulltext_from_response,
    _extract_words_from_response,
    _ocr_cache_key,
    blocks_to_structured_text,
    init_cloud_vision,
)
from .receipt_location import (
    _location_has_ocr_evidence,
    _recover_header_branch_store_location,
)


SUPPLEMENTAL_OCR_SCHEMA_VERSION = 1
SUPPLEMENTAL_OCR_TILE_FRACTIONS = ((0.0, 0.58), (0.42, 1.0))
SUPPLEMENTAL_OCR_PIXEL_CAP = 60_000_000
SUPPLEMENTAL_OCR_API_PIXEL_LIMIT = 75_000_000
SUPPLEMENTAL_OCR_PAYLOAD_CAP_BYTES = 18_000_000
SUPPLEMENTAL_OCR_STRATEGY = {
    "id": "uniform_receipt_bbox_two_vertical_tiles",
    "version": "1.2.0",
    "receipt_bbox": "largest bright low-saturation HSV component; S<30, V>70, 1% image padding",
    "tile_y_fractions": SUPPLEMENTAL_OCR_TILE_FRACTIONS,
    "preprocess": "grayscale, CLAHE clipLimit=2.0 tileGridSize=8x8, cubic upscale",
    "scale_rule": "preprocess both 3x candidates; use 3x only if both are <=60000000 pixels and <18000000 PNG bytes, otherwise regenerate both at 2x",
    "encoded_payload_guard": "fail closed if either selected 2x PNG is >=18000000 bytes",
    "api_pixel_guard": "fail closed if either selected tile exceeds 75000000 pixels",
    "coordinate_remap": "round(tile_coordinate / effective_scale) + source_tile_origin",
    "merge": "source-coordinate word IoU>=0.5; prefer richer containing token within 0.25 confidence, otherwise higher confidence",
    "integrity_version": "1",
}
SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT = hashlib.sha256(
    json.dumps(
        SUPPLEMENTAL_OCR_STRATEGY,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
_SEALED_STRATEGY_FINGERPRINT = (
    "4a87f52dca59c609136e787775c631a630e04d3a1a43ca5066b3b793e9b7cca5"
)
if SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT != _SEALED_STRATEGY_FINGERPRINT:
    raise RuntimeError("supplemental OCR strategy drifted from the sealed v1.2 contract")
SUPPLEMENTAL_OCR_CACHE_DIR = _OCR_CACHE_DIR / "supplemental"
_HEX_ID_RE = re.compile(r"[0-9a-f]{32,64}")
_EVIDENCE_KEYS = {
    "schema_version",
    "image_key",
    "image_shape",
    "strategy_fingerprint",
    "merged_text",
    "source_layout",
    "tiles",
}
_LAYOUT_BLOCK_KEYS = {"bbox", "confidence", "page", "text", "x", "y"}


class SupplementalOCRCacheMiss(RuntimeError):
    """Raised when cache-only supplemental OCR has no valid sidecar."""


_NORMAL_CACHE_LOCKS_GUARD = threading.Lock()
_NORMAL_CACHE_LOCKS: dict[str, tuple[threading.Lock, int]] = {}


@contextmanager
def _normal_cache_identity_lock(path: Path):
    key = os.path.normcase(str(path.resolve()))
    with _NORMAL_CACHE_LOCKS_GUARD:
        current = _NORMAL_CACHE_LOCKS.get(key)
        lock, users = current if current is not None else (threading.Lock(), 0)
        _NORMAL_CACHE_LOCKS[key] = (lock, users + 1)
    acquired = False
    try:
        lock.acquire()
        acquired = True
        yield
    finally:
        if acquired:
            lock.release()
        with _NORMAL_CACHE_LOCKS_GUARD:
            current = _NORMAL_CACHE_LOCKS.get(key)
            if current is not None and current[0] is lock:
                if current[1] == 1:
                    _NORMAL_CACHE_LOCKS.pop(key, None)
                else:
                    _NORMAL_CACHE_LOCKS[key] = (lock, current[1] - 1)


def _supplemental_receipt_bbox(image: np.ndarray) -> tuple[int, int, int, int]:
    """Find the largest bright, low-saturation paper-like component."""
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 70)).astype(np.uint8) * 255
    radius = max(5, round(min(height, width) * 0.025))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (radius, radius))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, width, height

    candidates = []
    for contour in contours:
        x, y, candidate_width, candidate_height = cv2.boundingRect(contour)
        if candidate_width * candidate_height >= width * height * 0.05:
            rectangularity = cv2.contourArea(contour) / max(
                1, candidate_width * candidate_height
            )
            candidates.append((
                candidate_width * candidate_height * (0.5 + rectangularity),
                x,
                y,
                candidate_width,
                candidate_height,
            ))
    if not candidates:
        return 0, 0, width, height

    _, x, y, candidate_width, candidate_height = max(candidates)
    pad_x = round(width * 0.01)
    pad_y = round(height * 0.01)
    return (
        max(0, x - pad_x),
        max(0, y - pad_y),
        min(width, x + candidate_width + pad_x),
        min(height, y + candidate_height + pad_y),
    )


def _supplemental_tile_bboxes(
    bbox: tuple[int, int, int, int],
) -> list[tuple[int, int, int, int]]:
    x0, y0, x1, y1 = bbox
    height = y1 - y0
    return [
        (x0, y0 + round(height * start), x1, y0 + round(height * end))
        for start, end in SUPPLEMENTAL_OCR_TILE_FRACTIONS
    ]


def _preprocess_supplemental_bbox(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    scale: int,
) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )


def _supplemental_png_size(image: np.ndarray) -> int:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("supplemental OCR tile PNG encoding failed")
    return len(encoded)


def _choose_supplemental_scale(candidates: list[dict]) -> int:
    return 3 if all(
        candidate["candidate_3x_pixels"] <= SUPPLEMENTAL_OCR_PIXEL_CAP
        and candidate["candidate_3x_png_bytes"] < SUPPLEMENTAL_OCR_PAYLOAD_CAP_BYTES
        for candidate in candidates
    ) else 2


def _prepare_supplemental_tiles(
    image: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
) -> tuple[int, list[np.ndarray]]:
    candidates = []
    for bbox in bboxes:
        processed = _preprocess_supplemental_bbox(image, bbox, scale=3)
        candidates.append({
            "candidate_3x_pixels": int(processed.size),
            "candidate_3x_png_bytes": _supplemental_png_size(processed),
            "processed_3x": processed,
        })

    scale = _choose_supplemental_scale(candidates)
    prepared = []
    for bbox, candidate in zip(bboxes, candidates, strict=True):
        if scale == 3:
            processed = candidate.pop("processed_3x")
        else:
            candidate.pop("processed_3x")
            processed = _preprocess_supplemental_bbox(image, bbox, scale=2)
        if processed.size > SUPPLEMENTAL_OCR_API_PIXEL_LIMIT:
            raise RuntimeError(
                f"supplemental OCR tile exceeds API pixel limit: {processed.size}"
            )
        payload_bytes = _supplemental_png_size(processed)
        if payload_bytes >= SUPPLEMENTAL_OCR_PAYLOAD_CAP_BYTES:
            raise RuntimeError(
                f"supplemental OCR tile exceeds payload limit: {payload_bytes} bytes"
            )
        prepared.append(processed)
    return scale, prepared


def _map_supplemental_word(
    word: dict,
    tile: tuple[int, int, int, int],
    *,
    scale: int,
) -> dict:
    x0, y0, _, _ = tile
    bbox = [
        [round(x / scale) + x0, round(y / scale) + y0]
        for x, y in word["bbox"]
    ]
    return {
        **word,
        "x": min(point[0] for point in bbox),
        "y": min(point[1] for point in bbox),
        "bbox": bbox,
    }


def _supplemental_word_iou(left: dict, right: dict) -> float:
    ax0 = min(point[0] for point in left["bbox"])
    ay0 = min(point[1] for point in left["bbox"])
    ax1 = max(point[0] for point in left["bbox"])
    ay1 = max(point[1] for point in left["bbox"])
    bx0 = min(point[0] for point in right["bbox"])
    by0 = min(point[1] for point in right["bbox"])
    bx1 = max(point[0] for point in right["bbox"])
    by1 = max(point[1] for point in right["bbox"])
    intersection = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(
        0, min(ay1, by1) - max(ay0, by0)
    )
    union = max(
        1,
        (ax1 - ax0) * (ay1 - ay0)
        + (bx1 - bx0) * (by1 - by0)
        - intersection,
    )
    return intersection / union


def _merge_supplemental_words_once(words: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for word in sorted(words, key=lambda item: (item["y"], item["x"])):
        duplicate = next(
            (
                index
                for index, prior in enumerate(merged)
                if _supplemental_word_iou(word, prior) >= 0.5
            ),
            None,
        )
        if duplicate is None:
            merged.append(word)
            continue
        prior = merged[duplicate]
        if prior["text"] in word["text"] and len(word["text"]) > len(prior["text"]):
            if word["confidence"] >= prior["confidence"] - 0.25:
                merged[duplicate] = word
        elif word["text"] not in prior["text"] and word["confidence"] > prior["confidence"]:
            merged[duplicate] = word
    return sorted(merged, key=lambda item: (item["y"], item["x"]))


def _merge_supplemental_words(words: list[dict]) -> list[dict]:
    """Merge until no replacement-created overlap remains."""
    current = list(words)
    while True:
        merged = _merge_supplemental_words_once(current)
        if len(merged) == len(current):
            return merged
        current = merged


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _valid_identity(
    image_key: object,
    image_shape: object,
    strategy_fingerprint: object,
) -> bool:
    return bool(
        isinstance(image_key, str)
        and _HEX_ID_RE.fullmatch(image_key)
        and isinstance(image_shape, (list, tuple))
        and len(image_shape) == 2
        and all(type(value) is int and value > 0 for value in image_shape)
        and isinstance(strategy_fingerprint, str)
        and len(strategy_fingerprint) == 64
        and _HEX_ID_RE.fullmatch(strategy_fingerprint)
    )


def _bounded_coordinate(value: object, maximum: int) -> bool:
    return bool(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and 0 <= value <= maximum
        and math.isfinite(value)
    )


def _validated_evidence(value: object) -> dict | None:
    if not isinstance(value, dict) or set(value) != _EVIDENCE_KEYS:
        return None
    if (
        type(value.get("schema_version")) is not int
        or value["schema_version"] != SUPPLEMENTAL_OCR_SCHEMA_VERSION
    ):
        return None
    if not _valid_identity(
        value.get("image_key"),
        value.get("image_shape"),
        value.get("strategy_fingerprint"),
    ):
        return None
    merged_text = value.get("merged_text")
    source_layout = value.get("source_layout")
    tiles = value.get("tiles")
    if (
        not isinstance(merged_text, str)
        or not merged_text.strip()
        or not isinstance(source_layout, list)
        or not source_layout
        or not isinstance(tiles, list)
        or len(tiles) != 2
        or any(
            not isinstance(tile, dict)
            or set(tile) != {"text"}
            or not isinstance(tile.get("text"), str)
            or not tile["text"].strip()
            for tile in tiles
        )
    ):
        return None
    height, width = value["image_shape"]
    for block in source_layout:
        if (
            not isinstance(block, dict)
            or set(block) != _LAYOUT_BLOCK_KEYS
            or not isinstance(block.get("text"), str)
            or not block["text"].strip()
        ):
            return None
        x = block.get("x")
        y = block.get("y")
        bbox = block.get("bbox")
        confidence = block.get("confidence")
        page = block.get("page")
        if (
            not _bounded_coordinate(x, width)
            or not _bounded_coordinate(y, height)
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= confidence <= 1
            or not math.isfinite(confidence)
            or isinstance(page, bool)
            or not isinstance(page, int)
            or page < 0
            or not isinstance(bbox, list)
            or len(bbox) != 4
        ):
            return None
        for point in bbox:
            if not isinstance(point, list) or len(point) != 2:
                return None
            point_x, point_y = point
            if (
                not _bounded_coordinate(point_x, width)
                or not _bounded_coordinate(point_y, height)
            ):
                return None
    try:
        if blocks_to_structured_text(source_layout) != merged_text:
            return None
    except (IndexError, KeyError, OverflowError, TypeError, ValueError):
        return None
    return {
        "schema_version": SUPPLEMENTAL_OCR_SCHEMA_VERSION,
        "image_key": value["image_key"],
        "image_shape": list(value["image_shape"]),
        "strategy_fingerprint": value["strategy_fingerprint"],
        "merged_text": merged_text,
        "source_layout": deepcopy(source_layout),
        "tiles": [{"text": tile["text"]} for tile in tiles],
    }


def build_supplemental_ocr_evidence(
    *,
    image_key: str,
    image_shape: tuple[int, int] | list[int],
    strategy_fingerprint: str,
    merged_text: str,
    source_layout: list[dict],
    tile_texts: tuple[str, str] | list[str],
) -> dict:
    """Build one strict raw sidecar body; invalid evidence is rejected."""
    evidence = _validated_evidence({
        "schema_version": SUPPLEMENTAL_OCR_SCHEMA_VERSION,
        "image_key": image_key,
        "image_shape": list(image_shape),
        "strategy_fingerprint": strategy_fingerprint,
        "merged_text": merged_text,
        "source_layout": source_layout,
        "tiles": [{"text": text} for text in tile_texts],
    })
    if evidence is None:
        raise ValueError("invalid supplemental OCR evidence")
    return evidence


def supplemental_ocr_cache_path(
    cache_dir: str | Path,
    *,
    image_key: str,
    image_shape: tuple[int, int] | list[int],
    strategy_fingerprint: str,
) -> Path:
    """Return the cache path bound to image bytes, shape, and OCR strategy."""
    if not _valid_identity(image_key, image_shape, strategy_fingerprint):
        raise ValueError("invalid supplemental OCR cache identity")
    height, width = image_shape
    return Path(cache_dir) / (
        f"{image_key}.{height}x{width}.{strategy_fingerprint}.json"
    )


def save_supplemental_ocr_evidence(cache_dir: str | Path, evidence: dict) -> Path:
    """Atomically store validated raw evidence without exposing a partial file."""
    validated = _validated_evidence(evidence)
    if validated is None:
        raise ValueError("invalid supplemental OCR evidence")
    path = supplemental_ocr_cache_path(
        cache_dir,
        image_key=validated["image_key"],
        image_shape=validated["image_shape"],
        strategy_fingerprint=validated["strategy_fingerprint"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        validated,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return path


def load_supplemental_ocr_evidence(
    cache_dir: str | Path,
    *,
    image_key: str,
    image_shape: tuple[int, int] | list[int],
    strategy_fingerprint: str,
) -> dict | None:
    """Load only a valid sidecar matching the complete expected identity."""
    try:
        path = supplemental_ocr_cache_path(
            cache_dir,
            image_key=image_key,
            image_shape=image_shape,
            strategy_fingerprint=strategy_fingerprint,
        )
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return None
    evidence = _validated_evidence(value)
    if evidence is None:
        return None
    expected_shape = list(image_shape)
    if (
        evidence["image_key"] != image_key
        or evidence["image_shape"] != expected_shape
        or evidence["strategy_fingerprint"] != strategy_fingerprint
    ):
        return None
    return evidence


def _acquire_two_tile_evidence(
    image: np.ndarray,
    *,
    client,
    identity: dict,
) -> dict:
    source_bbox = _supplemental_receipt_bbox(image)
    tile_bboxes = _supplemental_tile_bboxes(source_bbox)
    scale, prepared_tiles = _prepare_supplemental_tiles(image, tile_bboxes)
    tile_texts = []
    mapped_words = []
    for tile_bbox, processed in zip(tile_bboxes, prepared_tiles, strict=True):
        response = _call_cloud_vision(processed, client)
        tile_texts.append(_extract_fulltext_from_response(response) or "")
        mapped_words.extend(
            _map_supplemental_word(word, tile_bbox, scale=scale)
            for word in _extract_words_from_response(response)
        )
    source_layout = _merge_supplemental_words(mapped_words)
    evidence = build_supplemental_ocr_evidence(
        **identity,
        merged_text=blocks_to_structured_text(source_layout),
        source_layout=source_layout,
        tile_texts=tile_texts,
    )
    return evidence


def acquire_supplemental_ocr_evidence(
    image: np.ndarray,
    *,
    mode: str = "normal",
    client=None,
    cache_dir: str | Path = SUPPLEMENTAL_OCR_CACHE_DIR,
) -> dict:
    """Load or acquire one sealed two-tile supplemental OCR evidence record.

    ``normal`` reads then fills the cache, ``cache_only`` never contacts Vision,
    and ``fresh`` bypasses both cache reads and writes.
    """
    if mode not in {"normal", "cache_only", "fresh"}:
        raise ValueError("supplemental OCR mode must be normal, cache_only, or fresh")
    height, width = image.shape[:2]
    identity = {
        "image_key": _ocr_cache_key(image),
        "image_shape": [height, width],
        "strategy_fingerprint": SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT,
    }
    if mode != "fresh":
        cached = load_supplemental_ocr_evidence(cache_dir, **identity)
        if cached is not None:
            return cached
        if mode == "cache_only":
            raise SupplementalOCRCacheMiss(
                "no valid supplemental OCR cache entry for this image and strategy"
            )
    if mode == "fresh":
        return _acquire_two_tile_evidence(
            image,
            client=client if client is not None else init_cloud_vision(),
            identity=identity,
        )

    cache_path = supplemental_ocr_cache_path(cache_dir, **identity)
    with _normal_cache_identity_lock(cache_path):
        cached = load_supplemental_ocr_evidence(cache_dir, **identity)
        if cached is not None:
            return cached
        evidence = _acquire_two_tile_evidence(
            image,
            client=client if client is not None else init_cloud_vision(),
            identity=identity,
        )
        save_supplemental_ocr_evidence(cache_dir, evidence)
        return evidence


def _location_candidate(receipt: dict, text: str) -> str | None:
    probe = {"merchant": receipt.get("merchant"), "location": None}
    _recover_header_branch_store_location(probe, text)
    return probe.get("location")


def _propose_location(receipt: dict, evidence: dict) -> tuple[str | None, dict]:
    evidence_texts = [evidence["merged_text"], *(tile["text"] for tile in evidence["tiles"])]
    candidates = {
        _compact(candidate): candidate
        for text in evidence_texts
        if (candidate := _location_candidate(receipt, text))
    }
    current = str(receipt.get("location") or "")
    candidate = str(next(iter(candidates.values()), ""))
    compact_candidate = _compact(candidate)
    compact_current = _compact(current)
    support_lines = {
        _compact(line)
        for text in evidence_texts
        for line in text.splitlines()
        if compact_candidate and compact_candidate in _compact(line)
    }
    evidence_sources = [
        text
        for text in evidence_texts
        if compact_candidate and _location_has_ocr_evidence(candidate, text)
    ]
    current_evidence = any(
        current and _location_has_ocr_evidence(current, text)
        for text in evidence_texts
    )
    current_is_strict_fragment = bool(
        compact_current
        and compact_current != compact_candidate
        and compact_current in compact_candidate
    )
    current_tokens = [token for token in re.split(r"\s+", current.strip()) if token]
    current_is_address_like = bool(
        re.search(r"[\d０-９]", compact_current)
        and re.search(r"[都道府県市区町村郡]", compact_current)
    )
    stable_component_extension = bool(
        not current_evidence
        and len(current_tokens) > 1
        and not current_is_address_like
        and any(
            len(core) >= 3
            and core in compact_candidate
            and len(compact_candidate) - len(core) >= 2
            for token in current_tokens
            if (
                core := re.sub(
                    r"(?:本店|支店|営業所|出張所|料金所|店)$",
                    "",
                    _compact(token),
                )
            )
        )
    )
    criteria = {
        "one_candidate_across_merged_and_tiles": len(candidates) == 1,
        "candidate_present": bool(compact_candidate),
        "candidate_differs": bool(compact_candidate and compact_candidate != compact_current),
        "candidate_has_location_suffix": bool(
            re.search(r"(?:店|市|区|町|村|郡|支店|営業所|料金所)$", compact_candidate)
        ),
        "production_evidence_check": bool(evidence_sources),
        "unique_supporting_header_line": len(support_lines) == 1,
        "candidate_strictly_extends_current_or_unsupported_composite_component": bool(
            current_is_strict_fragment or stable_component_extension
        ),
    }
    accepted = all(criteria.values())
    return (candidate if accepted else None), {
        "accepted": accepted,
        "criteria": criteria,
        "before": current,
        "proposed": candidate or None,
        "candidate_set": sorted(candidates),
        "support_line_count": len(support_lines),
        "evidence_source_count": len(evidence_sources),
    }


def _rate_amounts(receipt: dict) -> dict[str, float]:
    amounts = {}
    for tax in receipt.get("taxes") or []:
        if not isinstance(tax, dict) or tax.get("rate") not in {"8%", "10%"}:
            continue
        try:
            amounts[tax["rate"]] = float(tax.get("amount") or 0)
        except (TypeError, ValueError):
            pass
    return amounts


def _category_sums(items: list[dict]) -> dict[str, float]:
    sums: dict[str, float] = {}
    for item in items:
        rate = str(item.get("tax_category") or "")
        sums[rate] = sums.get(rate, 0.0) + float(item.get("total") or 0)
    return sums


def _matching_rate_base_mode(
    sums: dict[str, float],
    bases: dict[str, float | None],
    taxes: dict[str, float],
) -> str | None:
    rates = {rate for rate, base in bases.items() if rate in {"8%", "10%"} and base}
    if rates != {"8%", "10%"}:
        return None
    for mode in ("net", "gross"):
        if all(
            abs(
                sums.get(rate, 0.0)
                - float(bases[rate] or 0)
                - (taxes.get(rate, 0.0) if mode == "gross" else 0.0)
            )
            <= 2
            for rate in rates
        ):
            return mode
    return None


def _effective_rate_bases(
    receipt: dict,
    printed: dict[str, float | None],
) -> tuple[dict[str, float | None], str]:
    bases = {
        rate: float(base)
        for rate, base in printed.items()
        if rate in {"8%", "10%"} and base is not None and float(base) > 0
    }
    if set(bases) == {"8%", "10%"}:
        return bases, "both_printed"
    if len(bases) != 1 or set(_rate_amounts(receipt)) != {"8%", "10%"}:
        return bases, "incomplete"
    item_sum = sum(
        float(item.get("total") or 0)
        for item in receipt.get("line_items") or []
        if isinstance(item, dict)
    )
    try:
        subtotal = float(receipt.get("subtotal"))
    except (TypeError, ValueError):
        return bases, "incomplete"
    if item_sum <= 0 or abs(item_sum - subtotal) > 2:
        return bases, "incomplete"
    printed_rate = next(iter(bases))
    other_rate = "10%" if printed_rate == "8%" else "8%"
    complement = item_sum - bases[printed_rate]
    if complement <= 0:
        return bases, "incomplete"
    bases[other_rate] = complement
    return bases, "single_printed_plus_subtotal_complement"


def _match_norm(value: str) -> str:
    return re.sub(r"[^0-9a-zぁ-んァ-ン一-龥ー]", "", str(value or "").lower())


def _line_variants(line: str) -> tuple[str, ...]:
    tokens = line.split()
    return tuple({_match_norm(line), _match_norm("".join(reversed(tokens)))})


def _description_line_score(description: str, total: float, line: str) -> tuple[float, float]:
    description = _match_norm(description)
    if len(description) < 2:
        return 0.0, 0.0
    best_similarity = 0.0
    best_coverage = 0.0
    for variant in _line_variants(line):
        if not variant:
            continue
        if description in variant:
            similarity, coverage = 1.0, 1.0
        else:
            matcher = SequenceMatcher(None, description, variant)
            similarity = matcher.ratio()
            coverage = matcher.find_longest_match().size / len(description)
        best_similarity = max(best_similarity, similarity)
        best_coverage = max(best_coverage, coverage)
    amount_tokens = {
        float(token.replace(",", ""))
        for token in re.findall(r"(?<!\d)(\d[\d,]*)(?!\d)", line)
        if token.replace(",", "").isdigit()
    }
    amount_bonus = 0.12 if any(abs(amount - total) <= 1 for amount in amount_tokens) else 0.0
    return best_similarity + amount_bonus, best_coverage


def _line_has_reduced_marker(line: str) -> bool:
    return bool(
        re.search(r"[*＊※]", line)
        or re.search(r"(?:^|\s)[Xx](?:\s|$)", line)
        or re.search(r"\d[\d,]*\s*[Xx%％](?:\s|$)", line)
    )


def _strict_rate_bases(text: str) -> tuple[dict[str, float], bool]:
    """Read only an adjacent ``<rate>% ... 対象額`` and yen value pair."""
    candidates: dict[str, set[float]] = {}
    lines = [line.strip() for line in text.splitlines()]
    for index, line in enumerate(lines):
        compact = _compact(line)
        match = re.search(r"(8|10)[%％].{0,8}(?:対象額|税抜対象)", compact)
        if not match:
            continue
        rate = f"{match.group(1)}%"
        amount_match = re.search(r"[¥￥]\s*([\d,]+)", line)
        if not amount_match:
            for following in lines[index + 1:index + 4]:
                if re.search(r"(?:8|10)[%％]", following):
                    break
                amount_match = re.fullmatch(r"[¥￥]\s*([\d,]+)", following)
                if amount_match:
                    break
        if amount_match:
            candidates.setdefault(rate, set()).add(
                float(amount_match.group(1).replace(",", ""))
            )
    ambiguous = any(len(values) != 1 for values in candidates.values())
    return {
        rate: next(iter(values))
        for rate, values in candidates.items()
        if len(values) == 1
    }, ambiguous


def _layout_marker_projection(receipt: dict, evidence: dict) -> dict:
    tile_text = "\n".join(tile["text"] for tile in evidence["tiles"])
    has_legend = bool(
        re.search(r"(?:[*＊※].{0,16}軽減税率|軽減税率.{0,16}[*＊※])", tile_text)
    )
    lines = []
    for line in evidence["merged_text"].splitlines():
        if re.search(r"小\s*計|合\s*計", line):
            break
        if line.strip():
            lines.append(line.strip())

    items = receipt.get("line_items") or []
    matches = []
    reduced_indices: set[int] = set()
    ambiguous_marker_lines = []
    for line_index, line in enumerate(lines):
        if not _line_has_reduced_marker(line):
            continue
        ranked = []
        for item_index, item in enumerate(items):
            try:
                total = float(item.get("total") or 0)
            except (AttributeError, TypeError, ValueError):
                continue
            score, coverage = _description_line_score(
                item.get("description") or "", total, line
            )
            ranked.append((score, coverage, item_index))
        ranked.sort(reverse=True)
        best_score, best_coverage, best_item_index = (
            ranked[0] if ranked else (0.0, 0.0, -1)
        )
        competing = [
            item_index
            for score, coverage, item_index in ranked[1:]
            if score >= best_score - 0.03 and coverage >= best_coverage - 0.05
        ]
        if best_score >= 0.62 and best_coverage >= 0.45 and not competing:
            if best_item_index in reduced_indices:
                ambiguous_marker_lines.append(line_index)
                continue
            reduced_indices.add(best_item_index)
            matches.append({
                "item_index": best_item_index,
                "line_index": line_index,
                "score": round(best_score, 4),
                "coverage": round(best_coverage, 4),
                "rate": "8%",
            })
        elif best_score >= 0.45 and best_coverage >= 0.35:
            ambiguous_marker_lines.append(line_index)

    proposed = ["8%" if index in reduced_indices else "10%" for index in range(len(items))]
    printed_bases, printed_bases_ambiguous = _strict_rate_bases(tile_text)
    effective_bases, base_evidence = _effective_rate_bases(receipt, printed_bases)
    try:
        sums = _category_sums([
            {**item, "tax_category": proposed[index]}
            for index, item in enumerate(items)
        ])
    except (TypeError, ValueError):
        sums = {}
    reconciliation_mode = _matching_rate_base_mode(
        sums, effective_bases, _rate_amounts(receipt)
    )
    before = [
        str(item.get("tax_category") or "")
        for item in items
        if isinstance(item, dict)
    ]
    changed = [
        index for index, pair in enumerate(zip(before, proposed)) if pair[0] != pair[1]
    ]
    criteria = {
        "reduced_rate_marker_legend": has_legend,
        "reduced_rows_uniquely_aligned": bool(matches) and not ambiguous_marker_lines,
        "candidate_differs": bool(changed),
        "valid_categories_only": all(rate in {"8%", "10%"} for rate in proposed),
        "strict_rate_bases_unambiguous": not printed_bases_ambiguous,
        "two_positive_evidenced_rate_bases": set(effective_bases) == {"8%", "10%"},
        "rate_base_arithmetic_reconciles": reconciliation_mode is not None,
    }
    return {
        "accepted": all(criteria.values()),
        "criteria": criteria,
        "proposed": proposed,
        "changed_item_indices": changed,
        "ambiguous_marker_line_indices": ambiguous_marker_lines,
        "matches": matches,
        "printed_rate_bases": printed_bases,
        "effective_rate_bases": effective_bases,
        "rate_base_evidence": base_evidence,
        "proposed_category_sums": sums,
        "reconciliation_mode": reconciliation_mode,
    }


def apply_supplemental_ocr_evidence(receipt: dict, evidence: dict) -> tuple[dict, dict]:
    """Apply only fully accepted location and tax proposals to a deep copy."""
    output = deepcopy(receipt)
    validated = _validated_evidence(evidence)
    if validated is None:
        return output, {"accepted_fields": [], "evidence_valid": False}

    location, location_meta = _propose_location(receipt, validated)
    tax_meta = _layout_marker_projection(receipt, validated)
    accepted_fields = []
    if location is not None:
        output["location"] = location
        accepted_fields.append("location")
    if tax_meta["accepted"]:
        items = output.get("line_items") or []
        for item, category in zip(items, tax_meta["proposed"]):
            item["tax_category"] = category
        accepted_fields.append("tax_categories")
    return output, {
        "accepted_fields": accepted_fields,
        "evidence_valid": True,
        "location": location_meta,
        "tax_categories": tax_meta,
    }
