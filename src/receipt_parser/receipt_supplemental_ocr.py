"""Field-only recovery from a sealed two-tile supplemental OCR result.

The evidence sidecar is deliberately raw: source identity, mapped word layout,
its derived merged OCR text, and exactly two tile texts.  It never stores
fixture IDs or proposed answers.
Location changes require one OCR-backed header extension. Tax changes require
either marker rows reconciled to printed rate bases or one exact title and
literal price owner with an explicit reduced-rate marker and legend.
Missing rows require a complete literal-price table whose printed count,
subtotal and both marker-backed rate bases agree with the primary OCR.
Existing price changes require mutually unique source and primary titles,
literal source prices, and complete closure against a printed primary subtotal
or one source total corroborated by both printed inclusive tax bases.
Merchant changes require one complete source-geometry header mark also printed
in the primary header; incomplete or competing logo fragments are rejected.
Every rejected proposal leaves the receipt unchanged.
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
from itertools import combinations, product
from pathlib import Path
from statistics import median

import cv2
import numpy as np

from .ocr import (
    _OCR_CACHE_DIR,
    _atomic_write_text,
    layout_blocks_sha256,
    _call_cloud_vision,
    _extract_fulltext_from_response,
    _extract_words_from_response,
    _ocr_cache_key,
    blocks_to_structured_text,
    init_cloud_vision,
)
from .receipt_location import (
    _HEADER_LOCATION_SUFFIX_RE,
    _location_has_ocr_evidence,
    _recover_header_branch_store_location,
)
from .patterns import _OCR_QTY_NOTATION_RE, _OCR_ZONE_END_RE, _OCR_TRAILING_PRICE_RE, _SKIP_PRICE_LINE, _discount_rate_tokens
from .receipt_item_repair import _clean_code_prefixed_item_descriptions, _valid_ocr_item_desc, _SUMMARY_COUNT_DESC_RE
from .receipt_item_cleanup import _clear_discounts_without_nearby_ocr_marker
from .receipt_identity_payment import (
    _CASH_CHANGE_LABEL_RE, _CASH_TENDER_LABEL_RE, _PAYMENT_TOKEN_RE,
    _DATE_LINE_RE, _TIME_HHMM_RE, _select_header_logo, _select_header_merchant_without_branch,
)
from .normalize import normalize_fullwidth
from .receipt_projection import (
    _collect_direct_summary_owners,
    _group_layout_rows,
    _layout_block_center_y,
    _layout_price_value,
    _layout_row_price_candidates,
    _layout_rows_are_local,
    _parse_qty_detail_total,
    _unit_first_qty1_values,
    _LAYOUT_REFERENCE_ROW_RE,
)
from .receipt_financial import (
    _interleaved_rate_tax_summary_entries, _rate_base_tax_pair_is_valid, extract_financial_totals,
    _jpy_summary_amount_options,
)
from .receipt_totals import _canonical_subtotal_from_taxes
from .receipt_tax_categories import _fix_tax_categories_from_ocr_markers, _is_bag_description


SUPPLEMENTAL_OCR_SCHEMA_VERSION = 1
SUPPLEMENTAL_OCR_TILE_FRACTIONS = ((0.0, 0.58), (0.42, 1.0))
SUPPLEMENTAL_OCR_PIXEL_CAP = 60_000_000
SUPPLEMENTAL_OCR_API_PIXEL_LIMIT = 75_000_000
SUPPLEMENTAL_OCR_PAYLOAD_CAP_BYTES = 18_000_000
SUPPLEMENTAL_OCR_STRATEGY = {
    "id": "uniform_receipt_bbox_two_vertical_tiles",
    "version": "1.3.0",
    "receipt_bbox": "largest bright low-saturation HSV component; S<30, V>70, 1% image padding",
    "tile_y_fractions": SUPPLEMENTAL_OCR_TILE_FRACTIONS,
    "preprocess": "grayscale, CLAHE clipLimit=2.0 tileGridSize=8x8, cubic upscale",
    "scale_rule": "preprocess both 3x candidates; use 3x only if both are <=60000000 pixels and <18000000 PNG bytes, otherwise regenerate both at 2x",
    "encoded_payload_guard": "fail closed if either selected 2x PNG is >=18000000 bytes",
    "api_pixel_guard": "fail closed if either selected tile exceeds 75000000 pixels",
    "coordinate_remap": "round(tile_coordinate / effective_scale) + source_tile_origin",
    "merge": "source-coordinate word IoU>=0.5; prefer richer containing token within 0.25 confidence, otherwise higher confidence",
    "integrity_version": "1",
    "text_rendering": "existing median-height vertical groups; words ordered by source x within each group",
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
    "85c90040f1f0438cecd0d5c4dcb74d7795cfa138cf976d080e9f27aee1a72bcb"
)
_LEGACY_STRATEGY_FINGERPRINT = "4a87f52dca59c609136e787775c631a630e04d3a1a43ca5066b3b793e9b7cca5"
if SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT != _SEALED_STRATEGY_FINGERPRINT:
    raise RuntimeError("supplemental OCR strategy drifted from the sealed v1.3 contract")
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


QUANTITY_CROP_OCR_STRATEGY = {
    "version": 1,
    "trigger": "one malformed count/unit row with mutually unique literal title and gross-price owners",
    "roi": "full width; preceding full geometry row, owner, detail, immediate discount control; one median word-height margin",
    "pixels": "EXIF-oriented original BGR uint8 color, cubic4x",
    "request": "one DOCUMENT_TEXT_DETECTION builtin/stable ja/en; no fallback or retry",
    "mapping": "round each coordinate/4 and add original ROI origin",
    "pixel_cap": 60_000_000,
    "png_byte_cap": 18_000_000,
}
QUANTITY_CROP_OCR_FINGERPRINT = hashlib.sha256(json.dumps(
    QUANTITY_CROP_OCR_STRATEGY, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
QUANTITY_CROP_OCR_CACHE_DIR = _OCR_CACHE_DIR / "quantity_rows"

NATIVE_OWNER_CROP_OCR_STRATEGY = {
    **QUANTITY_CROP_OCR_STRATEGY,
    "trigger": "one complete source unit/count/right-total row whose literal product conflicts",
    "roi": "full width; complete preceding title and detail row; one median word-height margin",
    "request": "one TEXT_DETECTION builtin/stable ja/en; no fallback or retry",
}
NATIVE_OWNER_CROP_OCR_FINGERPRINT = hashlib.sha256(json.dumps(
    NATIVE_OWNER_CROP_OCR_STRATEGY, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
NATIVE_OWNER_CROP_OCR_CACHE_DIR = _OCR_CACHE_DIR / "native_owners"

NATIVE_TITLE_CROP_OCR_STRATEGY = {
    **QUANTITY_CROP_OCR_STRATEGY,
    "trigger": "one exact primary POS-title suffix, bare barcode, same-row unit-count-gross, and closed printed summary/settlement packet",
    "roi": "full width around the complete code/title, barcode and unit-count-gross rows; one median word-height margin; expand crossing word boundaries; sealed source retains financial packet",
    "request": "one TEXT_DETECTION builtin/latest ja/en; retry=None; no fallback",
}
NATIVE_TITLE_CROP_OCR_FINGERPRINT = hashlib.sha256(json.dumps(
    NATIVE_TITLE_CROP_OCR_STRATEGY, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

# Raw optical producer identity; semantic admission is a separate table proof.
NATIVE_PRICE_COLUMN_OCR_STRATEGY = {
    "id": "isolated-native-whole-item-amount-column-v1",
    "pixels": "original EXIF-oriented BGR; cubic4; no grayscale/threshold/sharpening",
    "request": "DOCUMENT_TEXT_DETECTION", "model": "builtin/stable",
    "language_hints": ["ja", "en"], "calls_per_selected_owner": 1,
    "retry": None, "allow_fallback": False,
    "selection": "all literal item price cells before the unique printed subtotal; independently printed subtotal disagrees with complete source priced table",
    "margin": "one median height of selected money/attached-marker words in x and y",
    "boundary": "fixed padded union of all price-cell x bounds; y spans complete first through last item-price cells; title tails are non-owning",
    "pixel_cap": 60_000_000, "png_byte_cap": 18_000_000,
    "scope": "optical diagnosis only; no parser admission or existing cache writes",
}
NATIVE_PRICE_COLUMN_OCR_FINGERPRINT = hashlib.sha256(json.dumps(
    NATIVE_PRICE_COLUMN_OCR_STRATEGY, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

_QUANTITY_CONTEXT_KEYS = {
    "schema_version", "strategy_fingerprint", "image_key", "image_shape", "image_pixel_sha256",
    "input_text_sha256", "source_kind", "source_layout_sha256", "owner_title", "printed_amount",
    "quantity_row_text", "roi", "original_crop_pixel_sha256", "processed_shape",
    "processed_pixel_sha256", "png_sha256",
}
_PRICE_COLUMN_CONTEXT_KEYS = _QUANTITY_CONTEXT_KEYS - {
    "input_text_sha256", "owner_title", "printed_amount", "quantity_row_text",
}


def _valid_source_layout(layout, image_shape, *, allow_missing_confidence=False):
    """Shared raw word-shape and native image-bound checks; no semantic repair."""
    if not isinstance(layout, list) or not layout:
        return False
    height, width = image_shape
    for block in layout:
        if (not isinstance(block, dict) or set(block) != _LAYOUT_BLOCK_KEYS
                or not isinstance(block.get("text"), str) or not block["text"].strip()
                or not _bounded_coordinate(block.get("x"), width)
                or not _bounded_coordinate(block.get("y"), height)
                or (not (allow_missing_confidence and block.get("confidence") is None)
                    and not _bounded_coordinate(block.get("confidence"), 1))
                or type(block.get("page")) is not int or block["page"] < 0
                or not isinstance(block.get("bbox"), list) or len(block["bbox"]) != 4):
            return False
        if any(not isinstance(point, list) or len(point) != 2
               or not _bounded_coordinate(point[0], width) or not _bounded_coordinate(point[1], height)
               for point in block["bbox"]):
            return False
    return True


def _quantity_crop_candidate(text, layout, image_shape):
    """Select one malformed pair by exact source geometry and primary price ownership.

    Complete/conflicting primary counts, numeric title differences and duplicate
    owners abstain. The eventual literal crop pair must multiply to this gross.
    """
    def key(value):
        return "".join(c.casefold() for c in normalize_fullwidth(value) if c.isalnum())

    def compact(value):
        return _compact(normalize_fullwidth(value))

    def amount_owner(line):
        line = normalize_fullwidth(line).strip()
        match = re.search(r"([¥￥]?\s*\d[\d,]*)\s*[*＊※%除軽]*\s*$", line)
        return (key(line[:match.start()]), int(re.sub(r"\D", "", match[1]))) if match else None

    def control(line):
        return bool(_OCR_ZONE_END_RE.match(compact(line)) or re.search(
            r"割引|値引|小計|合計|現計|消費税|対象額|外税|内税|お預り|お釣り|税額", line))

    def malformed(line, gross=None):
        line = compact(line)
        match = re.fullmatch(r"[([{（]*(\d{2,4})[)\]）]*[xX×Ⅹ](?:単|@|#|₤|£)?[¥￥]?(\d[\d,]*)[)\]）]*", line)
        return bool(match and re.search(r"単|@|#|₤|£|¥|￥", line) and _parse_qty_detail_total(line) is None
                    and (gross is None or int(match[1]) * int(match[2].replace(",", "")) != gross))

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    start = next((i + 1 for i, line in enumerate(lines) if re.search(r"上記正に領収|担当者", compact(line))), 0)
    end = next((i for i in range(start, len(lines)) if _OCR_ZONE_END_RE.match(lines[i])), len(lines))
    lines = lines[start:end]
    if not any(malformed(line) for line in lines):
        return None
    rows = _group_layout_rows(layout)
    prices = _layout_row_price_candidates(layout)
    outputs = []
    for price in prices:
        title, gross = str(price.get("description") or "").strip(), price.get("value")
        title_key = key(title)
        if (type(gross) is not int or gross <= 0 or control(title) or not any(c.isalpha() for c in title_key)
                or sum(key(str(row.get("description") or "")) == title_key for row in prices) != 1):
            continue
        owners = []
        for i, line in enumerate(lines):
            if amount_owner(line) == (title_key, gross):
                owners.append(i)
            elif key(line) == title_key and i + 1 < len(lines) and amount_owner(lines[i + 1]) == ("", gross):
                owners.append(i + 1)
        if len(owners) != 1 or owners[0] + 1 >= len(lines) or not malformed(lines[owners[0] + 1], gross):
            continue
        owner_rows = [i for i, row in enumerate(rows) for block in row
                      if float(block["x"]) == price["x"] and _layout_block_center_y(block) == price["y"]
                      and _layout_price_value(block["text"], allow_small=True) == gross]
        if len(owner_rows) != 1:
            continue
        owner = owner_rows[0]
        if owner + 1 >= len(rows) or not _layout_rows_are_local(rows[owner], rows[owner + 1]):
            continue
        detail = " ".join(block["text"] for block in rows[owner + 1])
        if control(detail) or not malformed(detail, gross):
            continue
        last = owner + 1
        if last + 1 < len(rows) and re.fullmatch(
            r"(?:割引|値引).*\d+(?:\.\d+)?%.*-\d[\d,]*", compact(" ".join(block["text"] for block in rows[last + 1]))):
            last += 1
        words = [block for row in rows[max(0, owner - 1):last + 1] for block in row]
        heights = [max(p[1] for p in word["bbox"]) - min(p[1] for p in word["bbox"]) for word in words]
        if any(h <= 0 for h in heights):
            continue
        margin = max(1, round(median(heights)))
        ys = [p[1] for word in words for p in word["bbox"]]
        roi = [0, max(0, int(min(ys)) - margin), image_shape[1], min(image_shape[0], int(max(ys)) + margin + 1)]
        outputs.append(dict(owner_title=title, printed_amount=gross, quantity_row_text=detail, roi=roi))
    # ponytail: one eligible owner; use a bounded owner queue if multiple crops become necessary.
    return outputs[0] if len(outputs) == 1 else None


def _prepare_quantity_crop(image, roi, *, full_width=True):
    if (not isinstance(roi, list) or len(roi) != 4 or any(type(value) is not int for value in roi)
            or not 0 <= roi[0] < roi[2] <= image.shape[1] or not 0 <= roi[1] < roi[3] <= image.shape[0]
            or full_width and (roi[0] != 0 or roi[2] != image.shape[1])):
        raise ValueError("quantity crop ROI must be in the original image frame")
    height, width = (roi[3] - roi[1]) * 4, (roi[2] - roi[0]) * 4
    if height * width > QUANTITY_CROP_OCR_STRATEGY["pixel_cap"]:
        raise ValueError("quantity crop exceeds the pixel cap")
    native = image[roi[1]:roi[3], roi[0]:roi[2]]
    processed = cv2.resize(native, (width, height), interpolation=cv2.INTER_CUBIC)
    ok, png = cv2.imencode(".png", processed)
    if not ok or len(png) >= QUANTITY_CROP_OCR_STRATEGY["png_byte_cap"]:
        raise ValueError("quantity crop exceeds the encoded payload cap")
    return processed, dict(original_crop_pixel_sha256=hashlib.sha256(native.tobytes()).hexdigest(),
        processed_shape=list(processed.shape[:2]), processed_pixel_sha256=hashlib.sha256(processed.tobytes()).hexdigest(),
        png_sha256=hashlib.sha256(png.tobytes()).hexdigest())


def build_quantity_crop_context(image, text, *, primary_layout_blocks=None, supplemental_ocr_evidence=None):
    """Independently bind one original-frame owner crop to image, text and layout.

    The caller must authenticate primary geometry as belonging to these original
    pixels. Otherwise only sealed same-image supplemental geometry may be used.
    No crop record or proposed quantity is an input to this function.
    """
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or not isinstance(text, str) or not text.strip()):
        return None
    shape, image_key = list(image.shape[:2]), _ocr_cache_key(image)
    source_kind, layout = "primary", primary_layout_blocks
    if not _valid_source_layout(layout, shape) or any(block["page"] != 0 for block in layout):
        source = _validated_evidence(supplemental_ocr_evidence)
        if (source is None or source["strategy_fingerprint"] != SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT
                or source["image_key"] != image_key or source["image_shape"] != shape):
            return None
        source_kind, layout = "sealed_supplemental", source["source_layout"]
        if any(block["page"] != 0 for block in layout):
            return None
    text = normalize_fullwidth(text)
    candidate = _quantity_crop_candidate(text, layout, shape)
    if candidate is None:
        return None
    try:
        _, prepared = _prepare_quantity_crop(image, candidate["roi"])
    except ValueError:
        return None
    return dict(schema_version=1, strategy_fingerprint=QUANTITY_CROP_OCR_FINGERPRINT,
        image_key=image_key, image_shape=shape, image_pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        input_text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(), source_kind=source_kind,
        source_layout_sha256=layout_blocks_sha256(layout), **candidate, **prepared)


def _valid_quantity_context(context, *, fingerprint=QUANTITY_CROP_OCR_FINGERPRINT):
    price_column = fingerprint == NATIVE_PRICE_COLUMN_OCR_FINGERPRINT
    if (fingerprint not in (QUANTITY_CROP_OCR_FINGERPRINT, NATIVE_OWNER_CROP_OCR_FINGERPRINT,
                            NATIVE_TITLE_CROP_OCR_FINGERPRINT, NATIVE_PRICE_COLUMN_OCR_FINGERPRINT)
            or not isinstance(context, dict)
            or set(context) != (_PRICE_COLUMN_CONTEXT_KEYS if price_column else _QUANTITY_CONTEXT_KEYS)):
        return False
    shape, roi = context["image_shape"], context["roi"]
    return bool(type(context["schema_version"]) is int and context["schema_version"] == 1
        and context["strategy_fingerprint"] == fingerprint
        and (fingerprint == QUANTITY_CROP_OCR_FINGERPRINT or context["source_kind"] == "sealed_supplemental")
        and isinstance(context["image_key"], str) and _HEX_ID_RE.fullmatch(context["image_key"])
        and isinstance(shape, list) and len(shape) == 2 and all(type(n) is int and n > 0 for n in shape)
        and isinstance(roi, list) and len(roi) == 4 and all(type(n) is int for n in roi)
        and 0 <= roi[0] < roi[2] <= shape[1] and 0 <= roi[1] < roi[3] <= shape[0]
        and (price_column or roi[0] == 0 and roi[2] == shape[1])
        and isinstance(context["processed_shape"], list) and all(type(n) is int for n in context["processed_shape"])
        and context["processed_shape"] == [(roi[3] - roi[1]) * 4, (roi[2] - roi[0]) * 4]
        and math.prod(context["processed_shape"]) <= QUANTITY_CROP_OCR_STRATEGY["pixel_cap"]
        and isinstance(context["source_kind"], str) and context["source_kind"] in {"primary", "sealed_supplemental"}
        and (price_column or type(context["printed_amount"]) is int and context["printed_amount"] > 0
             and all(isinstance(context[name], str) and context[name].strip() for name in ("owner_title", "quantity_row_text")))
        and all(isinstance(context[name], str) and re.fullmatch(r"[0-9a-f]{64}", context[name]) for name in
            ("image_pixel_sha256", "source_layout_sha256", "original_crop_pixel_sha256",
             "processed_pixel_sha256", "png_sha256") + (() if price_column else ("input_text_sha256",))))


def _validated_quantity_crop(evidence, expected_context, *, fingerprint=QUANTITY_CROP_OCR_FINGERPRINT):
    if (not _valid_quantity_context(expected_context, fingerprint=fingerprint) or not isinstance(evidence, dict)
            or set(evidence) != {"context", "vision_text", "source_layout", "merged_text"}
            or not _valid_quantity_context(evidence["context"], fingerprint=fingerprint)
            or evidence["context"] != expected_context
            or not isinstance(evidence["vision_text"], str) or not evidence["vision_text"].strip()
            or not isinstance(evidence["merged_text"], str)
            or not _valid_source_layout(evidence["source_layout"], expected_context["image_shape"],
                allow_missing_confidence=fingerprint in (NATIVE_TITLE_CROP_OCR_FINGERPRINT, NATIVE_PRICE_COLUMN_OCR_FINGERPRINT))):
        return None
    x0, y0, x1, y1 = expected_context["roi"]
    for word in evidence["source_layout"]:
        xs, ys = [p[0] for p in word["bbox"]], [p[1] for p in word["bbox"]]
        if (word["page"] != 0 or word["x"] != min(xs) or word["y"] != min(ys)
                or not x0 <= min(xs) < max(xs) <= x1 or not y0 <= min(ys) < max(ys) <= y1):
            return None
    try:
        if blocks_to_structured_text(evidence["source_layout"]) != evidence["merged_text"]:
            return None
    except (IndexError, KeyError, TypeError, ValueError, OverflowError):
        return None
    return deepcopy(evidence)


def quantity_crop_ocr_cache_path(cache_dir, context, *, fingerprint=QUANTITY_CROP_OCR_FINGERPRINT):
    if not _valid_quantity_context(context, fingerprint=fingerprint):
        raise ValueError("invalid quantity crop context")
    digest = hashlib.sha256(json.dumps(context, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return Path(cache_dir) / f'{context["image_key"]}_{digest}.json'


def load_quantity_crop_ocr_evidence(cache_dir, *, expected_context, fingerprint=QUANTITY_CROP_OCR_FINGERPRINT):
    if not _valid_quantity_context(expected_context, fingerprint=fingerprint):
        return None
    path = quantity_crop_ocr_cache_path(cache_dir, expected_context, fingerprint=fingerprint)
    try:
        return _validated_quantity_crop(json.loads(path.read_text(encoding="utf-8")), expected_context, fingerprint=fingerprint)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None


def acquire_quantity_crop_ocr_evidence(image, context, *, mode="normal", client=None,
        cache_dir=QUANTITY_CROP_OCR_CACHE_DIR, fingerprint=QUANTITY_CROP_OCR_FINGERPRINT):
    """At most one strict native4 request, with optional cache-only abstention."""
    if mode not in {"normal", "cache_only", "fresh"}:
        raise ValueError("quantity crop mode must be normal, cache_only, or fresh")
    if (not _valid_quantity_context(context, fingerprint=fingerprint) or not isinstance(image, np.ndarray)
            or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3):
        return None
    if (_ocr_cache_key(image) != context["image_key"] or list(image.shape[:2]) != context["image_shape"]
            or hashlib.sha256(image.tobytes()).hexdigest() != context["image_pixel_sha256"]):
        return None
    if mode != "fresh":
        cached = load_quantity_crop_ocr_evidence(cache_dir, expected_context=context, fingerprint=fingerprint)
        if cached is not None or mode == "cache_only":
            return cached

    def acquire():
        processed, prepared = _prepare_quantity_crop(image, context["roi"],
            full_width=fingerprint != NATIVE_PRICE_COLUMN_OCR_FINGERPRINT)
        if any(context[name] != value for name, value in prepared.items()):
            return None
        native_text = fingerprint in (NATIVE_OWNER_CROP_OCR_FINGERPRINT, NATIVE_TITLE_CROP_OCR_FINGERPRINT)
        response = _call_cloud_vision(processed, client if client is not None else init_cloud_vision(),
            allow_fallback=False, **({"feature": "TEXT_DETECTION"} if native_text else {}),
            **({"model": "builtin/latest"} if fingerprint == NATIVE_TITLE_CROP_OCR_FINGERPRINT else {}))
        words = [_map_supplemental_word(word, tuple(context["roi"]), scale=4)
                 for word in _extract_words_from_response(response, **(
                    {"use_paragraph_confidence": False} if fingerprint == NATIVE_OWNER_CROP_OCR_FINGERPRINT else
                    {"use_paragraph_confidence": False, "preserve_missing_confidence": True}
                    if fingerprint in (NATIVE_TITLE_CROP_OCR_FINGERPRINT, NATIVE_PRICE_COLUMN_OCR_FINGERPRINT) else {}))]
        return _validated_quantity_crop(dict(context=deepcopy(context),
            vision_text=_extract_fulltext_from_response(response) or "", source_layout=words,
            merged_text=blocks_to_structured_text(words)), context, fingerprint=fingerprint)

    if mode == "fresh":
        return acquire()
    path = quantity_crop_ocr_cache_path(cache_dir, context, fingerprint=fingerprint)
    with _normal_cache_identity_lock(path):
        cached = load_quantity_crop_ocr_evidence(cache_dir, expected_context=context, fingerprint=fingerprint)
        if cached is not None:
            return cached
        evidence = acquire()
        if evidence is not None:
            _atomic_write_text(path, json.dumps(evidence, ensure_ascii=False, indent=2))
        return evidence


def _native_owner_crop_candidate(text, layout, image_shape):
    """Acquisition eligibility only; no model prices or captured response input."""
    rows = _group_layout_rows(layout)
    end = next((i for i, row in enumerate(rows)
                if _OCR_ZONE_END_RE.match(_cash_compact("".join(b["text"] for b in row)))), len(rows))
    outputs = []
    primary = normalize_fullwidth(text).splitlines()
    primary = primary[:next((i for i, line in enumerate(primary) if _OCR_ZONE_END_RE.match(line.strip())), len(primary))]
    for i, row in enumerate(rows[:end]):
        pair = _unit_first_qty1_values(row)
        if pair is None or pair[1] == pair[3] or i == 0 or not _layout_rows_are_local(rows[i - 1], row):
            continue
        title = "".join(b["text"] for b in rows[i - 1]).strip()
        if (not _valid_ocr_item_desc(title) or _LITERAL_PRICE_SKIP_RE.search(title)
                or _LAYOUT_REFERENCE_ROW_RE.search(title) or _OCR_ZONE_END_RE.match(title)):
            continue
        units = [int(m[1].replace(",", "")) for line in primary
                 if (m := re.fullmatch(r"@?(\d[\d,]*)\s+1\s*[点個コ]", line.strip()))]
        if units.count(pair[1]) != 1:
            continue
        words = rows[i - 1] + row
        heights = [_cash_box(b)[3] - _cash_box(b)[1] for b in words]
        if any(h <= 0 for h in heights):
            continue
        margin = max(1, round(median(heights)))
        ys = [p[1] for b in words for p in b["bbox"]]
        roi = [0, max(0, int(min(ys)) - margin), image_shape[1], min(image_shape[0], int(max(ys)) + margin + 1)]
        outputs.append(dict(owner_title=title, printed_amount=pair[3],
                            quantity_row_text=" ".join(b["text"] for b in row), roi=roi))
    # ponytail: one ambiguous owner and one request; multiple owners abstain.
    return outputs[0] if len(outputs) == 1 else None


def build_native_owner_crop_context(image, text, *, supplemental_ocr_evidence=None):
    """Bind TEXT owner acquisition to original pixels, exact text and sealed layout."""
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or not isinstance(text, str) or not text.strip()):
        return None
    source = _validated_evidence(supplemental_ocr_evidence)
    shape, image_key = list(image.shape[:2]), _ocr_cache_key(image)
    if (source is None or source["strategy_fingerprint"] != SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT
            or source["image_key"] != image_key or source["image_shape"] != shape
            or any(b["page"] != 0 or b["x"] != _cash_box(b)[0] or b["y"] != _cash_box(b)[1]
                   or _cash_box(b)[0] >= _cash_box(b)[2] or _cash_box(b)[1] >= _cash_box(b)[3]
                   for b in source["source_layout"])):
        return None
    text = normalize_fullwidth(text)
    candidate = _native_owner_crop_candidate(text, source["source_layout"], shape)
    if candidate is None:
        return None
    try:
        _, prepared = _prepare_quantity_crop(image, candidate["roi"])
    except ValueError:
        return None
    return dict(schema_version=1, strategy_fingerprint=NATIVE_OWNER_CROP_OCR_FINGERPRINT,
        image_key=image_key, image_shape=shape, image_pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        input_text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(), source_kind="sealed_supplemental",
        source_layout_sha256=layout_blocks_sha256(source["source_layout"]), **candidate, **prepared)


def native_owner_crop_ocr_cache_path(cache_dir, context):
    return quantity_crop_ocr_cache_path(cache_dir, context, fingerprint=NATIVE_OWNER_CROP_OCR_FINGERPRINT)


def load_native_owner_crop_ocr_evidence(cache_dir, *, expected_context):
    return load_quantity_crop_ocr_evidence(cache_dir, expected_context=expected_context,
                                          fingerprint=NATIVE_OWNER_CROP_OCR_FINGERPRINT)


def acquire_native_owner_crop_ocr_evidence(image, context, *, mode="normal", client=None,
                                         cache_dir=NATIVE_OWNER_CROP_OCR_CACHE_DIR):
    return acquire_quantity_crop_ocr_evidence(image, context, mode=mode, client=client,
        cache_dir=cache_dir, fingerprint=NATIVE_OWNER_CROP_OCR_FINGERPRINT)


def _native_price_column_candidate(layout, image_shape):
    """A source-priced table disagrees with its unique printed subtotal.

    Select every price cell before that subtotal; never a desired amount or
    a model-selected row. Clipped title tails carry no ownership authority.
    """
    subtotal = _collect_direct_summary_owners(layout)["unique"]["subtotal"]
    if subtotal is None:
        return None
    boundary = min(_layout_block_center_y(layout[i]) for i in subtotal["row_indices"])
    rows = [row for row in _layout_row_price_candidates(layout) if row["y"] < boundary]
    if not rows or sum(row["value"] for row in rows) == subtotal["value"]:
        return None
    owners = []
    for row in rows:
        matches = [word for word in layout if float(word["x"]) == row["x"]
                   and _layout_block_center_y(word) == row["y"]
                   and _layout_price_value(word["text"], allow_small=True) == row["value"]]
        if len(matches) != 1:
            return None
        owners.extend(matches)
    boxes = [_cash_box(word) for word in owners]
    margin = max(1, round(median(box[3] - box[1] for box in boxes)))
    height, width = image_shape
    return [max(0, min(box[0] for box in boxes) - margin), max(0, min(box[1] for box in boxes) - margin),
            min(width, max(box[2] for box in boxes) + margin + 1), min(height, max(box[3] for box in boxes) + margin + 1)]


def build_native_price_column_context(image, text, *, supplemental_ocr_evidence=None):
    """Bind original column pixels to sealed geometry; bind current P at admission."""
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or not isinstance(text, str) or not text.strip()):
        return None
    source = _validated_evidence(supplemental_ocr_evidence)
    image_key, shape = _ocr_cache_key(image), list(image.shape[:2])
    if (source is None or source["image_key"] != image_key or source["image_shape"] != shape
            or source["strategy_fingerprint"] != SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT
            or any(word["page"] != 0 for word in source["source_layout"])):
        return None
    subtotal = _collect_direct_summary_owners(source["source_layout"])["unique"]["subtotal"]
    roi = _native_price_column_candidate(source["source_layout"], shape)
    if roi is None or _literal_price_subtotal(normalize_fullwidth(text)) != subtotal["value"]:
        return None
    try:
        _, prepared = _prepare_quantity_crop(image, roi, full_width=False)
    except ValueError:
        return None
    return dict(schema_version=1, strategy_fingerprint=NATIVE_PRICE_COLUMN_OCR_FINGERPRINT,
        image_key=image_key, image_shape=shape, image_pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        source_kind="sealed_supplemental", source_layout_sha256=layout_blocks_sha256(source["source_layout"]),
        roi=roi, **prepared)


def _native_title_detail(row):
    raw = " ".join(normalize_fullwidth(block["text"]).strip() for block in row)
    match = re.fullmatch(r"\s*[¥￥]\s*(\d[\d,]*)\s+(\d{1,3})\s*(?:個|点|コ)\s*[¥￥]\s*(\d[\d,]*)\s*", raw)
    if not match:
        return None
    unit, qty, gross = (int(match.group(i).replace(",", "")) for i in (1, 2, 3))
    return (qty, unit, gross) if qty > 0 and unit > 0 and qty * unit == gross else None


def _native_title_crop_candidate(primary_text, layout, image_shape):
    """Bind one opaque P code to an exact suffix/barcode/detail and closed S packet."""
    if not isinstance(primary_text, str) or not primary_text.strip():
        return None
    rows = _group_layout_rows(layout)
    possible = []
    complete = []
    for row_idx, row in enumerate(rows):
        ordered = sorted(row, key=lambda block: block["x"])
        for code_idx, block in enumerate(ordered):
            code = _cash_compact(block["text"])
            if not re.fullmatch(r"\d{3,6}", code) or code_idx + 1 >= len(ordered):
                continue
            title_blocks = ordered[code_idx + 1:]
            suffix = "".join(_cash_compact(word["text"]) for word in title_blocks)
            if (not _valid_ocr_item_desc(suffix)
                    or any(re.fullmatch(r"\d[\d,]*", _cash_compact(word["text"]))
                           or _LAYOUT_REFERENCE_ROW_RE.search(word["text"])
                           or re.search(r"(?:割引|値引|外税|内税|対象|[%％])", word["text"])
                           for word in title_blocks)):
                continue
            if row_idx + 2 >= len(rows):
                continue
            barcode = _cash_compact("".join(word["text"] for word in rows[row_idx + 1]))
            if (not re.fullmatch(r"\d{10,14}", barcode)
                    or not _layout_rows_are_local(row, rows[row_idx + 1])
                    or not _layout_rows_are_local(rows[row_idx + 1], rows[row_idx + 2])):
                continue
            possible.append((row_idx, code, suffix, block, title_blocks))
            detail = _native_title_detail(rows[row_idx + 2])
            if detail is not None:
                complete.append((row_idx, code, suffix, block, title_blocks, barcode, detail))
    if len(possible) != 1 or len(complete) != 1:
        return None
    code_row, code, suffix, code_block, title_blocks, barcode, detail = complete[0]
    barcode_idx, detail_idx = code_row + 1, code_row + 2
    qty, unit, gross = detail
    summaries = _collect_direct_summary_owners(layout)["owners"]
    subtotals = summaries["subtotal"]
    bases = [owner for owner in summaries["rate_component"] if owner["kind"] == "base"]
    taxes = summaries["tax"]
    totals = [owner for owner in summaries["total"]
              if re.match(r"^(?:合計|総計)", owner["row_text"])]
    if (len(subtotals) != 1 or subtotals[0]["value"] != gross
            or subtotals[0].get("count") != qty or len(bases) != 1 or len(taxes) != 1 or len(totals) != 1):
        return None
    base, tax, total = bases[0], taxes[0], totals[0]
    if (not detail_idx < subtotals[0]["row"] < base["row"] < tax["row"] < total["row"]
            or base["mode"] != "外税" or base["value"] != gross or base["rate"] != tax.get("rate")
            or not _rate_base_tax_pair_is_valid(base["rate"], base["value"], tax["value"], base["mode"])
            or total["value"] != gross + tax["value"]):
        return None
    total_row = total["row"]
    settlement = []
    awaiting_settlement_amount = False
    for idx in range(total_row + 1, len(rows)):
        row_text = "".join(word["text"] for word in rows[idx])
        compact_row = _cash_compact(row_text)
        if ("現計" in compact_row or _PAYMENT_TOKEN_RE.search(row_text)
                or _CASH_TENDER_LABEL_RE.search(row_text) or _CASH_CHANGE_LABEL_RE.search(row_text)):
            settlement.append(idx)
            awaiting_settlement_amount = not bool(re.search(r"\d", compact_row))
        elif awaiting_settlement_amount and re.fullmatch(r"[¥￥]?\d[\d,]*", compact_row):
            settlement.append(idx)
            awaiting_settlement_amount = False
        else:
            break
    if not settlement or not any("現計" in _cash_compact("".join(word["text"] for word in rows[idx]))
                                 for idx in settlement):
        return None
    tax_aggregates = []
    malformed_tax_aggregate = False
    packet_end = max(settlement)
    for idx in range(code_row, packet_end + 1):
        compact = _cash_compact("".join(block["text"] for block in rows[idx]))
        if "外税額計" not in compact:
            continue
        match = re.fullmatch(r"外税額計[¥￥]?(\d[\d,]*)", compact)
        if match:
            tax_aggregates.append((idx, int(match.group(1).replace(",", ""))))
        else:
            malformed_tax_aggregate = True
    if (malformed_tax_aggregate or len(tax_aggregates) > 1
            or tax_aggregates and (base["mode"] != "外税" or tax_aggregates[0][1] != tax["value"])):
        return None
    allowed_rows = {code_row, barcode_idx, detail_idx, subtotals[0]["row"], base["row"],
                    tax["row"], total_row, *settlement,
                    *(idx for idx, _value in tax_aggregates)}
    for idx in range(packet_end + 1):
        if idx in allowed_rows:
            continue
        row = rows[idx]
        compact = _cash_compact("".join(word["text"] for word in row))
        if (re.search(r"[¥￥]\d|割引|値引|クーポン|[%％]|(?<!\d)-\s*[¥￥]?\d", compact)
                or re.search(r"\d+(?:個|点|コ).*(?:[¥￥]\d|[×xX@]\d)", compact)
                or re.fullmatch(r"\d{1,3}(?:個|点|コ)", compact)):
            return None

    primary_raw = [line for line in normalize_fullwidth(primary_text).splitlines() if line.strip()]
    primary = [_cash_compact(line) for line in primary_raw]
    subtotal_rows = [i for i, line in enumerate(primary) if "小計" in line]
    subtotal_idx = subtotal_rows[0] if len(subtotal_rows) == 1 else None
    total_idx = next((i for i, line in enumerate(primary) if re.search(r"(?:合計|総計)", line)), None)
    if subtotal_idx is None or total_idx is None or total_idx <= subtotal_idx:
        return None
    owners = [(i, line) for i, line in enumerate(primary[:subtotal_idx])
              if line.endswith(suffix) and re.fullmatch(r"\d+" + re.escape(suffix), line)]
    if len(owners) != 1:
        return None
    owner_idx, owner = owners[0]
    if owner_idx + 3 >= subtotal_idx or primary[owner_idx + 1] != barcode:
        return None
    detail_match = re.fullmatch(r"\s*[¥￥]\s*(\d[\d,]*)\s+(\d{1,3})\s*(?:個|点|コ)\s*",
                                normalize_fullwidth(primary_raw[owner_idx + 2]))
    gross_match = re.fullmatch(r"[¥￥](\d[\d,]*)", primary[owner_idx + 3])
    if (not detail_match or not gross_match
            or int(detail_match.group(1).replace(",", "")) != unit
            or int(detail_match.group(2)) != qty
            or int(gross_match.group(1).replace(",", "")) != gross):
        return None
    tax_start = next((i for i in range(subtotal_idx + 1, total_idx)
                      if re.search(r"外税|内税|消費税|税額|対象額|\d+(?:\.\d+)?[%％]", primary[i])), total_idx)
    subtotal_text = "".join(primary[subtotal_idx:tax_start])
    counts = [int(match.group(1)) for match in re.finditer(r"(?<!\d)(\d{1,3})(?:点|個|コ)", subtotal_text)]
    subtotal_values = [int(match.group(1).replace(",", "")) for match in
                       re.finditer(r"[¥￥](\d[\d,]*)", subtotal_text)]
    if counts != [qty] or subtotal_values != [gross]:
        return None
    grand_totals = [int(match.group(1).replace(",", "")) for match in re.finditer(
        r"(?<!預り)(?:合計|総計)\s*[¥￥]?\s*(\d[\d,]*)", normalize_fullwidth(primary_text))]
    current_rows = [i for i, line in enumerate(primary) if "現計" in line]
    if grand_totals != [total["value"]] or len(current_rows) != 1 or current_rows[0] <= total_idx:
        return None
    current_end = current_rows[0] + 1
    if current_end < len(primary) and re.fullmatch(r"[¥￥]?\d[\d,]*", primary[current_end]):
        current_end += 1
    primary_packet = "\n".join(primary[owner_idx:current_end])
    source_packet = " ".join(
        _cash_compact("".join(block["text"] for block in rows[idx]))
        for idx in range(code_row, packet_end + 1)
    )
    if re.search(r"[$€£]|(?<![A-Za-z])(?:USD|EUR|GBP)(?![A-Za-z])",
                 primary_packet + "\n" + source_packet, re.IGNORECASE):
        return None
    primary_tax_pairs = {
        (str(entry.get("rate")), int(entry["amount"]))
        for entry in extract_financial_totals(primary_text).get("taxes", [])
        if isinstance(entry, dict) and type(entry.get("amount")) in (int, float)
    }
    if primary_tax_pairs != {(base["rate"], tax["value"])}:
        return None
    if any(re.search(r"(?:割引|値引|クーポン|[%％]|-\s*\d)", line)
           for line in primary[owner_idx:subtotal_idx]):
        return None

    selected_rows = rows[code_row:detail_idx + 1]
    boxes = [block for row in selected_rows for block in row]
    heights = [_cash_box(block)[3] - _cash_box(block)[1] for block in boxes]
    if not boxes or any(height <= 0 for height in heights):
        return None
    margin = max(1, round(median(heights)))
    roi = [0, max(0, min(_cash_box(block)[1] for block in boxes) - margin), image_shape[1],
           min(image_shape[0], max(_cash_box(block)[3] for block in boxes) + margin)]
    while True:
        crossing = [_cash_box(block) for block in layout
                    if any(_cash_box(block)[1] < edge < _cash_box(block)[3] for edge in roi[1::2])]
        if not crossing:
            break
        roi[1] = min(roi[1], *(box[1] for box in crossing))
        roi[3] = max(roi[3], *(box[3] for box in crossing))
    if roi[3] <= roi[1]:
        return None
    return dict(owner_title=suffix, printed_amount=gross,
                quantity_row_text=" ".join(block["text"] for block in rows[detail_idx]), roi=roi)


def build_native_title_crop_context(image, text, *, supplemental_ocr_evidence=None):
    """Bind one fixed latest TEXT crop to a sealed, closed native item packet."""
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or not isinstance(text, str) or not text.strip()):
        return None
    source = _validated_evidence(supplemental_ocr_evidence)
    shape, image_key = list(image.shape[:2]), _ocr_cache_key(image)
    if (source is None or source["strategy_fingerprint"] != SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT
            or source["image_key"] != image_key or source["image_shape"] != shape
            or any(block["page"] != 0 or block["x"] != _cash_box(block)[0] or block["y"] != _cash_box(block)[1]
                   or _cash_box(block)[0] >= _cash_box(block)[2] or _cash_box(block)[1] >= _cash_box(block)[3]
                   for block in source["source_layout"])):
        return None
    text = normalize_fullwidth(text)
    candidate = _native_title_crop_candidate(text, source["source_layout"], shape)
    if candidate is None:
        return None
    try:
        _, prepared = _prepare_quantity_crop(image, candidate["roi"])
    except ValueError:
        return None
    return dict(schema_version=1, strategy_fingerprint=NATIVE_TITLE_CROP_OCR_FINGERPRINT,
        image_key=image_key, image_shape=shape, image_pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        input_text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(), source_kind="sealed_supplemental",
        source_layout_sha256=layout_blocks_sha256(source["source_layout"]), **candidate, **prepared)


def _native_row_box(row):
    boxes = [_cash_box(b) for b in row]
    x, y, right, bottom = min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)
    return dict(bbox=[[x, y], [right, y], [right, bottom], [x, bottom]])


def _recover_native_qty1_owner_prices(receipt, source, primary_text, native, context):
    """Stage a complete literal basket; no subset or residual-derived money."""
    validated = _validated_quantity_crop(native, context, fingerprint=NATIVE_OWNER_CROP_OCR_FINGERPRINT)
    if (validated is None or not isinstance(primary_text, str) or receipt.get("currency") != "JPY"
            or type(receipt.get("total")) not in (int, float) or not math.isfinite(receipt["total"]) or receipt["total"] <= 0
            or source["image_key"] != context["image_key"] or source["image_shape"] != context["image_shape"]
            or source["strategy_fingerprint"] != SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT
            or layout_blocks_sha256(source["source_layout"]) != context["source_layout_sha256"]
            or hashlib.sha256(normalize_fullwidth(primary_text).encode("utf-8")).hexdigest() != context["input_text_sha256"]):
        return None, {}
    selected = _native_owner_crop_candidate(primary_text, source["source_layout"], source["image_shape"])
    if selected is None or any(context[name] != value for name, value in selected.items()):
        return None, {}
    items = receipt.get("line_items") or []
    keys = [_cash_title_key(item.get("description")) for item in items if isinstance(item, dict)]
    if len(keys) != len(items) or any(len(k) < 3 for k in keys) or len(set(keys)) != len(keys):
        return None, {}
    try:
        if any(type(item.get("qty")) is bool or float(item.get("qty")) != 1
               or item.get("gross") is not None or item.get("discount") not in (None, 0) or item.get("discount_rate") not in (None, "")
               or any(type(item.get(f)) is bool or not math.isfinite(float(item.get(f))) or float(item[f]) <= 0
                      for f in ("unit_price", "total")) for item in items):
            return None, {}
    except (ValueError, TypeError, OverflowError):
        return None, {}
    primary = [normalize_fullwidth(line).strip() for line in primary_text.splitlines() if line.strip()]
    stop = next((i for i, line in enumerate(primary) if _OCR_ZONE_END_RE.match(line)), len(primary))
    count_headers = [i for i, line in enumerate(primary[:stop]) if _SUMMARY_COUNT_DESC_RE.fullmatch(_cash_compact(line))]
    if count_headers:
        # A displaced count remains a literal summary token, never a price.
        if count_headers != [stop - 1] or stop + 2 >= len(primary):
            return None, {}
        count = re.fullmatch(r"(\d+)\s*[点個コ]", primary[stop + 1])
        total = re.fullmatch(r"[¥￥](\d[\d,]*)", _cash_compact(primary[stop + 2]))
        if (not count or int(count[1]) != len(items) or not total
                or int(total[1].replace(",", "")) != receipt["total"]):
            return None, {}
        stop -= 1
    positions = [[i for i, line in enumerate(primary[:stop]) if _cash_title_key(line) == k] for k in keys]
    if any(len(p) != 1 for p in positions) or [p[0] for p in positions] != sorted(p[0] for p in positions):
        return None, {}
    primary_units, primary_totals = {}, []
    for index, (k, p) in enumerate(zip(keys, positions, strict=True)):
        end = positions[index + 1][0] if index + 1 < len(positions) else stop
        unit_values = []
        for line in primary[p[0] + 1:end]:
            unit = re.fullmatch(r"@?(\d[\d,]*)\s+1\s*[点個コ]", line)
            amount = re.fullmatch(r"[¥￥]?(\d[\d,]*)", _cash_compact(line))
            if unit:
                unit_values.append(int(unit[1].replace(",", "")))
            elif amount:
                primary_totals.append(int(amount[1].replace(",", "")))
            else:
                return None, {}
        if len(unit_values) != 1:
            return None, {}
        primary_units[k] = unit_values[0]
    gray = _group_layout_rows(source["source_layout"])
    native_rows = _group_layout_rows(validated["source_layout"])
    native_titles = [(i, _cash_title_key("".join(b["text"] for b in row)))
                     for i, row in enumerate(native_rows)
                     if _cash_title_key("".join(b["text"] for b in row)) in keys]
    if len({k for _, k in native_titles}) != len(native_titles):
        return None, {}
    summary = _collect_direct_summary_owners(source["source_layout"])["unique"]["total"]
    if summary is None or summary["value"] <= 0 or receipt.get("total") != summary["value"]:
        return None, {}
    selected_rows = [i for i, row in enumerate(gray)
                     if " ".join(b["text"] for b in row) == context["quantity_row_text"]
                     and i > 0 and "".join(b["text"] for b in gray[i - 1]).strip() == context["owner_title"]]
    if len(selected_rows) != 1:
        return None, {}
    selected_index = selected_rows[0]
    selected_pair = _unit_first_qty1_values(gray[selected_index])
    anchor = gray[selected_index][selected_pair[2]]
    native_prices = _layout_row_price_candidates(validated["source_layout"])
    recovered = [(candidate, b) for candidate in native_prices for b in validated["source_layout"]
                 if candidate["value"] == context["printed_amount"]
                 and float(b["x"]) == candidate["x"] and _layout_block_center_y(b) == candidate["y"]
                 and _layout_price_value(b["text"], allow_small=True) == candidate["value"]
                 and _cash_overlap(anchor, b) >= .5]
    if len(recovered) != 1:
        return None, {}
    candidate, recovered_word = recovered[0]
    native_price_rows = [i for i, row in enumerate(native_rows) if any(b is recovered_word for b in row)]
    if len(native_price_rows) != 1 or native_price_rows[0] == 0:
        return None, {}
    native_index = native_price_rows[0]
    closed_native = _unit_first_qty1_values(native_rows[native_index])
    if (closed_native is None or closed_native[1] != closed_native[3]
            or _cash_overlap(_native_row_box(native_rows[native_index - 1]), _native_row_box(gray[selected_index - 1])) < .5):
        return None, {}
    native_title_rows = {i for i, _ in native_titles}
    for i, row in enumerate(native_rows):
        text = "".join(b["text"] for b in row)
        if _LITERAL_PRICE_SKIP_RE.search(text) or re.search(r"[%％]|[-−－]\s*[¥￥]?\d", text):
            return None, {}
        if i == native_index or i in native_title_rows:
            continue
        for b in row:
            if _layout_price_value(b["text"], allow_small=True) is not None and len([
                owner for owner in source["source_layout"]
                if _cash_compact(owner["text"]) == _cash_compact(b["text"]) and _cash_overlap(owner, b) >= .5
            ]) != 1:
                return None, {}
    packets, title_rows, detail_rows = [], set(), set()
    for i, row in enumerate(gray[:summary["row"]]):
        triple = _unit_first_qty1_values(row)
        numeric = [(j, value) for j, b in enumerate(row)
                   if (value := _layout_price_value(b["text"], allow_small=True)) is not None]
        repeated = (len(numeric) == 2 and numeric[0][1] == numeric[1][1]
                    and _cash_box(row[numeric[0][0]])[2] + 30 <= _cash_box(row[numeric[1][0]])[0]
                    and _cash_compact("".join(b["text"] for b in row))
                    == "@" + _cash_compact(row[numeric[0][0]]["text"]) + _cash_compact(row[numeric[1][0]]["text"]))
        if triple is None and not repeated:
            continue
        if i == 0 or not _layout_rows_are_local(gray[i - 1], row):
            return None, {}
        title = "".join(b["text"] for b in gray[i - 1])
        if _LITERAL_PRICE_SKIP_RE.search(title) or _LAYOUT_REFERENCE_ROW_RE.search(title) or _OCR_ZONE_END_RE.match(title):
            return None, {}
        k = _cash_title_key(title)
        if k not in keys:
            matches = [value for j, value in native_titles
                       if _cash_overlap(_native_row_box(gray[i - 1]), _native_row_box(native_rows[j])) >= .5
                       and re.findall(r"\d+", _cash_compact(title)) == re.findall(r"\d+", value)]
            if len(matches) != 1:
                return None, {}
            k = matches[0]
        unit, total = (triple[1], triple[3]) if triple else (numeric[0][1], numeric[1][1])
        if primary_units[k] != unit:
            return None, {}
        mode = "gray_closed_pair" if triple else "gray_repeat_primary_quantity_native_title"
        if i == selected_index:
            if _cash_title_key(candidate["description"]) != k or unit == total:
                return None, {}
            unit = total = closed_native[3]
            mode = "native_closed_pair"
        elif unit != total:
            return None, {}
        packets.append(dict(key=k, value=total, mode=mode, row=i))
        title_rows.add(i - 1)
        detail_rows.add(i)
    if ([p["key"] for p in packets] != keys or len({p["key"] for p in packets}) != len(keys)
            or selected_index not in detail_rows):
        return None, {}
    start = min(title_rows)
    if any(c["y"] < min(_layout_block_center_y(b) for b in gray[start])
           for c in _layout_row_price_candidates([b for row in gray[:summary["row"]] for b in row])):
        return None, {}
    for i in range(start, summary["row"]):
        text = "".join(b["text"] for b in gray[i])
        if _LITERAL_PRICE_SKIP_RE.search(text) or re.search(r"[%％]|[-−－]\s*[¥￥]?\d", text):
            return None, {}
        if i not in title_rows | detail_rows and (i < max(detail_rows)
                or any(_layout_price_value(b["text"], allow_small=True) is not None for b in gray[i])):
            return None, {}
    tail = [b for row in gray[max(detail_rows) + 1:summary["row"]] for b in row]
    if tail and (not count_headers or not _SUMMARY_COUNT_DESC_RE.fullmatch("".join(
            _cash_compact(b["text"]) for b in tail if not re.fullmatch(r"[点個コ]+", _cash_compact(b["text"]))))):
        return None, {}
    values = [p["value"] for p in packets]
    if len(set(values)) != len(values) or sorted(primary_totals) != sorted(values) or sum(values) != summary["value"]:
        return None, {}
    staged = deepcopy(items)
    for item, value in zip(staged, values, strict=True):
        item["unit_price"] = item["total"] = value
    proof = dict(mode="native_owner_literal_basket", owners=packets,
                 strategy_fingerprint=context["strategy_fingerprint"], roi=deepcopy(context["roi"]))
    return (staged, proof) if staged != items else (None, {})


def reconcile_quantity_crop_rows(text, evidence, *, expected_context=None):
    """Copy the selected literal pair and relocate a corroborated owned deduction.

    Injected text cannot authenticate original pixels: its caller must supply a
    context independently computed from that image and bound source geometry.
    """
    validated = _validated_quantity_crop(evidence, expected_context)
    if validated is None or hashlib.sha256(normalize_fullwidth(text).encode("utf-8")).hexdigest() != expected_context["input_text_sha256"]:
        return text, []
    changed, proposals = _reconcile_quantity_row_text(text, validated["merged_text"], validated["source_layout"])
    owner_key = "".join(c.casefold() for c in normalize_fullwidth(expected_context["owner_title"]) if c.isalnum())
    if len(proposals) != 1 or proposals[0]["title"] != owner_key or proposals[0]["total"] != expected_context["printed_amount"]:
        return text, []
    proposal = proposals[0]
    rows = _group_layout_rows(validated["source_layout"])
    pair_rows = [i for i, row in enumerate(rows)
                 if _compact(normalize_fullwidth(" ".join(word["text"] for word in row)))
                 == _compact(proposal["after"])]
    if len(pair_rows) == 1 and pair_rows[0] + 1 < len(rows):
        pair_index = pair_rows[0]
        control = _compact(normalize_fullwidth(" ".join(word["text"] for word in rows[pair_index + 1])))
        schedule = re.fullmatch(r'(?:割引|値引)(\d+(?:\.\d+)?)%-[¥￥]?(\d[\d,]*)', control)
        if schedule and _layout_rows_are_local(rows[pair_index], rows[pair_index + 1]):
            discount = int(schedule[2].replace(",", ""))
            owned = dict(description=expected_context["owner_title"], qty=proposal["qty"],
                         unit_price=proposal["unit_price"], total=proposal["total"] - discount,
                         discount=discount, discount_rate="")
            supported = _recover_owned_discount_rates(dict(line_items=[owned]), validated)
            lines = changed.split("\n")
            start = proposal["line_index"] + 1
            end = next((i for i in range(start, len(lines)) if _OCR_ZONE_END_RE.match(lines[i].strip())), len(lines))
            label = re.compile(r'(?:割引|値引)')
            rate = re.compile(re.escape(schedule[1]) + r'%')
            prefix = []
            for index in range(start, min(end, start + 3)):
                part = _compact(normalize_fullwidth(lines[index]))
                if label.fullmatch(part) or rate.fullmatch(part) or re.fullmatch(r'(?:割引|値引)' + re.escape(schedule[1]) + r'%', part):
                    prefix.append(index)
                else:
                    break
            deduction = re.compile(r'-\s*[¥￥]?\s*(\d[\d,]*)')
            matches = [i for i in range(end)
                       if (amount := deduction.fullmatch(lines[i].strip()))
                       and int(amount[1].replace(",", "")) == discount]
            # The next optical price owner proves where this control ends.
            following = next((normalize_fullwidth(line).strip()
                              for line in lines[prefix[-1] + 1:end] if line.strip()), "") if prefix else ""
            next_row = rows[pair_index + 2] if pair_index + 2 < len(rows) else []
            next_prices = _layout_row_price_candidates(next_row)
            following_key = "".join(c.casefold() for c in following if c.isalnum())
            next_keys = {"".join(c.casefold() for c in normalize_fullwidth(value) if c.isalnum())
                         for value in ([next_prices[0]["description"], " ".join(word["text"] for word in next_row)]
                                       if len(next_prices) == 1 else [])}
            if (len(supported) == 1 and owned["discount_rate"] == schedule[1] + "%"
                    and len(matches) == 1 and prefix
                    and _compact("".join(lines[i] for i in prefix)) in {"割引" + schedule[1] + "%", "値引" + schedule[1] + "%"}
                    and matches[0] > prefix[-1] + 1
                    and len(next_prices) == 1 and following_key in next_keys
                    and _layout_rows_are_local(rows[pair_index + 1], next_row)):
                moved = lines.pop(matches[0])
                lines.insert(prefix[-1] + 1, moved)
                changed = "\n".join(lines)
                proposal["discount_relocation"] = dict(before_line_index=matches[0],
                    after_line_index=prefix[-1] + 1, text=moved, discount=discount,
                    discount_rate=owned["discount_rate"])
    proposals[0].update(source="native_quantity_owner_crop", strategy_fingerprint=QUANTITY_CROP_OCR_FINGERPRINT,
                        roi=expected_context["roi"], png_sha256=expected_context["png_sha256"])
    return changed, proposals


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
    if not _valid_source_layout(source_layout, value["image_shape"]):
        return None
    try:
        if blocks_to_structured_text(
            source_layout,
            sort_within_rows=value["strategy_fingerprint"] != _LEGACY_STRATEGY_FINGERPRINT,
        ) != merged_text:
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
    for_read: bool = False,
) -> Path:
    """Return the cache path bound to image bytes, shape, and OCR strategy."""
    if not _valid_identity(image_key, image_shape, strategy_fingerprint):
        raise ValueError("invalid supplemental OCR cache identity")
    height, width = image_shape
    path = Path(cache_dir) / (
        f"{image_key}.{height}x{width}.{strategy_fingerprint}.json"
    )
    if for_read and strategy_fingerprint == SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT and not path.is_file():
        legacy = Path(cache_dir) / f"{image_key}.{height}x{width}.{_LEGACY_STRATEGY_FINGERPRINT}.json"
        if legacy.is_file():
            return legacy
    return path


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
            for_read=True,
        )
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return None
    evidence = _validated_evidence(value)
    if evidence is None or evidence["strategy_fingerprint"] != path.stem.rsplit('.', 1)[-1]:
        return None
    expected_shape = list(image_shape)
    legacy_migration = (
        strategy_fingerprint == SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT
        and evidence["strategy_fingerprint"] == _LEGACY_STRATEGY_FINGERPRINT
    )
    if (
        evidence["image_key"] != image_key
        or evidence["image_shape"] != expected_shape
        or (evidence["strategy_fingerprint"] != strategy_fingerprint and not legacy_migration)
    ):
        return None
    if legacy_migration:
        # ponytail: re-render validated words in memory; cache-only stays read-only.
        evidence["merged_text"] = blocks_to_structured_text(evidence["source_layout"])
        evidence["strategy_fingerprint"] = strategy_fingerprint
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
    current = _compact(str(receipt.get("location") or ""))
    merchant = _compact(str(receipt.get("merchant") or ""))
    components = set()
    composite_seen = False
    for line in text.splitlines()[:16]:
        if _DATE_LINE_RE.search(line) or _OCR_ZONE_END_RE.match(line.strip()):
            break
        if "×" not in line:
            continue
        composite_seen = True
        parts = line.split("×")
        if len(parts) != 2:
            return None
        owner, candidate = map(_compact, parts)
        if (
            len(owner) >= 2 and merchant.startswith(owner)
            and current and current != candidate and current in candidate
            and _HEADER_LOCATION_SUFFIX_RE.search(candidate)
            and _location_has_ocr_evidence(candidate, text)
        ):
            components.add(candidate)
        else:
            return None
    if composite_seen:
        return components.pop() if len(components) == 1 else None
    return probe.get("location")


def _propose_location(receipt: dict, evidence: dict) -> tuple[str | None, dict]:
    evidence_texts = [evidence["merged_text"], *(tile["text"] for tile in evidence["tiles"])]
    candidates = {
        _compact(candidate): candidate
        for text in evidence_texts
        if (candidate := _location_candidate(receipt, text))
    }
    if len(candidates) > 1:
        longest = max(candidates, key=len)
        if all(longest.endswith(key) for key in candidates):
            candidates = {longest: candidates[longest]}
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
    merchant = _compact(str(receipt.get("merchant") or ""))
    current_branch = compact_current.removeprefix(merchant) if merchant else compact_current
    current_suffix = _HEADER_LOCATION_SUFFIX_RE.search(current_branch)
    candidate_suffix = _HEADER_LOCATION_SUFFIX_RE.search(compact_candidate)
    same_suffix_stem_extension = bool(
        current_suffix and candidate_suffix
        and current_suffix.group() == candidate_suffix.group()
        and len(current_stem := current_branch[:current_suffix.start()]) >= 2
        and (candidate_stem := compact_candidate[:candidate_suffix.start()]).startswith(current_stem)
        and len(candidate_stem) > len(current_stem)
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
        "production_evidence_check": len(evidence_sources) >= 2,
        "unique_supporting_header_line": len(support_lines) == 1,
        "candidate_strictly_extends_current_or_unsupported_composite_component": bool(
            current_is_strict_fragment or stable_component_extension or same_suffix_stem_extension
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


def _partial_marker_projection(receipt: dict, evidence: dict, *, primary_text: str | None = None) -> dict:
    """Trigger: an explicit code-adjacent marker and literal 8% legend.

    Non-POS rows require a primary-backed literal title, the exact legend
    glyph and a printed 8% tax component. All digits and prices stay intact.
    """
    legends: dict[str, set[str]] = {}
    for text in [evidence["merged_text"], *(tile["text"] for tile in evidence["tiles"])]:
        for line in text.splitlines():
            compact = re.sub(r"\s+", "", normalize_fullwidth(line))
            match = re.search(r"([*＊※])印.{0,32}?(\d+(?:\.\d+)?)[%％]", compact)
            if match:
                rate = match.group(2) if "軽減税率" in compact else "unsupported"
                legends.setdefault(match.group(1), set()).add(rate)
    markers = {marker for marker, rates in legends.items() if rates == {"8"}}
    non_pos_markers = set()
    if primary_text and any(
        owner["rate"] == "8%" and owner["kind"] == "tax" and owner["value"] > 0
        for owner in _collect_direct_summary_owners(evidence["source_layout"])["owners"]["rate_component"]
    ):
        for row in _group_layout_rows(evidence["source_layout"]):
            line = re.sub(r"\s+", "", normalize_fullwidth("".join(block["text"] for block in row)))
            if match := re.fullmatch(r"([*＊※])印.{0,12}軽減税率対象商品(?:です|。)?", line):
                non_pos_markers.add(match.group(1))
        non_pos_markers -= {marker for marker, rates in legends.items() if rates != {"8"}}
    markers |= non_pos_markers
    items = receipt.get("line_items") or []
    if not markers or not items or any(not isinstance(item, dict) for item in items):
        return {"accepted": False}

    def key(text):
        return "".join(char.casefold() for char in normalize_fullwidth(text) if char.isalnum())

    def owns(first, second):
        return (min(len(first), len(second)) >= 3 and (first in second or second in first)
                and re.findall(r"\d+", first) == re.findall(r"\d+", second))

    primary_lines = [key(re.split(r"[¥￥]", line, maxsplit=1)[0]) for line in (primary_text or "").splitlines()]
    primary_items, primary_locks = deepcopy(items), set()
    if primary_text:
        _fix_tax_categories_from_ocr_markers(primary_items, primary_text, locked_indices=primary_locks)

    rows = []
    for candidate in _layout_row_price_candidates(evidence["source_layout"]):
        title = str(candidate.get("description") or "")
        prefix = re.match(r"^\s*\d{3,}\s*([*＊※])\s*(.+?)\s*$", title)
        non_pos = not prefix and re.match(r"^\s*([*＊※])\s*(.+?)\s*$", title)
        parsed = prefix or non_pos
        rows.append((key(parsed.group(2) if parsed else title),
                     parsed.group(1) if parsed else None, candidate, bool(non_pos)))
    proposed = [item.get("tax_category") for item in items]
    matches = []
    for title_key, marker, row, non_pos in rows:
        if marker not in markers or not title_key or non_pos and marker not in non_pos_markers:
            continue
        # A code-prefixed competing title is an ambiguity, never a second owner.
        owners = [other for other, _mark, _row, _non_pos in rows
                  if (owns(other, title_key) if non_pos else
                      other == title_key or re.fullmatch(r"\d{3,}" + re.escape(title_key), other))]
        printed_prices = [block for block in evidence["source_layout"]
                          if float(block["x"]) == row["x"]
                          and _layout_block_center_y(block) == row["y"]
                          and _layout_price_value(block["text"], allow_small=True) == row["value"]]
        item_owners = [index for index, item in enumerate(items)
                       if (owns(key(str(item.get("description") or "")), title_key) if non_pos
                           else key(str(item.get("description") or "")) == title_key)
                       and item.get("total") == row["value"]]
        if len(owners) != 1 or len(printed_prices) != 1 or len(item_owners) != 1:
            continue
        index = item_owners[0]
        item = items[index]
        if non_pos and (
            primary_lines.count(key(str(item.get("description") or ""))) != 1
            or item.get("qty") != 1 or (item.get("discount") or 0) != 0
            or item.get("unit_price") != item.get("total")
            or item.get("_tax_category_locked") in {"0%", "10%"}
            or index in primary_locks and primary_items[index].get("tax_category") != "8%"
        ):
            continue
        if non_pos and sum(owns(other, key(str(item.get("description") or "")))
                           and other_row["value"] == row["value"]
                           for other, _mark, other_row, _non_pos in rows) != 1:
            continue
        proposed[index] = "8%"
        matches.append({"item_index": index, "source_title": row["description"],
                        "printed_value": row["value"], "marker": marker})
    changed = [index for index, item in enumerate(items)
               if item.get("tax_category") != proposed[index]]
    return {"accepted": bool(changed), "proposed": proposed,
            "changed_item_indices": changed, "matches": matches,
            "mode": "partial_explicit_title_price_marker_ownership"}


def _mixed_mode_marker_partition(receipt, evidence, primary_text, partial):
    """Derive one invalid base only from a closed basket and distinct tax modes.

    Exact marker owners and exhaustive unique item subsets prove each standard
    component, including unsupported model zero rates. Printed exemptions veto;
    printed amounts and the rejected OCR base remain unchanged.
    """
    rejected = {"accepted": False}
    items = receipt.get("line_items") or []
    if not primary_text or not partial.get("matches") or not items or any(
        not isinstance(item, dict) or isinstance(item.get("total"), bool)
        or not isinstance(item.get("total"), (int, float)) or not math.isfinite(item["total"])
        or item["total"] <= 0 or int(item["total"]) != item["total"]
        or item.get("tax_category") not in {"0%", "8%", "10%"}
        or item.get("_tax_category_locked") == "0%"
        for item in items
    ):
        return rejected
    if any(re.search(
        r"非課税|不課税|免税|\d[\d,]*非$|"
        r"(?<![\d.])0(?:\.0+)?%(?:対象|タイショウ|内税|外税|税額)|"
        r"(?:対象(?:額)?|タイショウ|内税|外税|税額)\(?0(?:\.0+)?%", _compact(normalize_fullwidth(line)))
        for text in [primary_text, evidence["merged_text"], *(tile["text"] for tile in evidence["tiles"])]
        for line in text.splitlines()):
        return rejected
    summaries = _collect_direct_summary_owners(evidence["source_layout"])
    subtotal, total = (summaries["unique"][role] for role in ("subtotal", "total"))
    components = {}
    for owner in summaries["owners"]["rate_component"]:
        components.setdefault((owner["rate"], owner["mode"], owner["kind"]), []).append(owner)
    required = {("8%", "外税", kind) for kind in ("base", "tax")} | {
        ("10%", mode, kind) for mode in ("外税", "内税") for kind in ("base", "tax")}
    if not subtotal or not total or set(components) != required or any(len(rows) != 1 for rows in components.values()):
        return rejected
    values = {role: rows[0]["value"] for role, rows in components.items()}
    observed = values["8%", "外税", "base"]
    reduced_tax = values["8%", "外税", "tax"]
    external_base, external_tax = (values["10%", "外税", kind] for kind in ("base", "tax"))
    included_base, included_tax = (values["10%", "内税", kind] for kind in ("base", "tax"))
    derived = subtotal["value"] - external_base - included_base
    if (
        sum(item["total"] for item in items) != subtotal["value"]
        or receipt.get("subtotal") != subtotal["value"] or receipt.get("total") != total["value"]
        or subtotal["value"] + reduced_tax + external_tax != total["value"]
        or _rate_amounts(receipt) != {"8%": reduced_tax, "10%": external_tax}
        or not _rate_base_tax_pair_is_valid("10%", external_base, external_tax, "外税")
        or included_base <= 0 or included_tax != 0 or int(included_base * 0.1 / 1.1) != 0
        or derived <= 0 or not _rate_base_tax_pair_is_valid("8%", derived, reduced_tax, "外税")
        or observed == derived or _rate_base_tax_pair_is_valid("8%", observed, reduced_tax, "外税")
    ):
        return rejected
    marked_items, locks = deepcopy(items), set()
    _fix_tax_categories_from_ocr_markers(marked_items, primary_text, locked_indices=locks)
    reduced = {index for index in locks if marked_items[index]["tax_category"] == "8%"}
    reduced.update(match["item_index"] for match in partial["matches"])

    def unique_subset(excluded, target):
        choices = [(index, int(item["total"])) for index, item in enumerate(items)
                   if index not in excluded and item["total"] <= target]
        # ponytail: bound exhaustive proof to 16 candidates; use counted DP if
        # larger unresolved baskets need this distinct-mode recovery.
        if len(choices) > 16:
            return None
        found = None
        for size in range(1, len(choices) + 1):
            for group in combinations(choices, size):
                if sum(amount for _index, amount in group) == target:
                    if found is not None:
                        return None
                    found = {index for index, _amount in group}
        return found

    included = unique_subset(reduced, included_base)
    external = unique_subset(reduced | included, external_base) if included is not None else None
    if external is None:
        return rejected
    standard = included | external
    proposed = ["10%" if index in standard else "8%" for index in range(len(items))]
    if any(proposed[index] != marked_items[index]["tax_category"] for index in locks):
        return rejected
    changed = [index for index, item in enumerate(items) if item["tax_category"] != proposed[index]]
    return {"accepted": bool(changed), "proposed": proposed, "changed_item_indices": changed,
            "mode": "closed_distinct_tax_mode_marker_partition", "matches": partial["matches"],
            "printed_components": summaries["owners"]["rate_component"],
            "rejected_printed_base": observed, "derived_reduced_base": derived,
            "included_item_indices": sorted(included), "external_item_indices": sorted(external)}


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
    full = {
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
    if full["accepted"]:
        return full
    partial = _partial_marker_projection(receipt, evidence)
    return {**partial, "full_projection_rejection": full} if partial["accepted"] else full


def reconcile_supplemental_quantity_rows(text: str, evidence: dict | None) -> tuple[str, list[dict]]:
    """Copy a complete bound pair only for one malformed detail with the same
    unique printed title and extended price in both OCR sources. Preserve all
    digits; a complete or conflicting primary count is never replaced. A split
    pair is joined only when both literal digits already equal the source pair.
    """
    validated = _validated_evidence(evidence)
    if validated is None or validated["strategy_fingerprint"] != SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT:
        return text, []
    return _reconcile_quantity_row_text(text, validated["merged_text"], validated["source_layout"])


def _reconcile_quantity_row_text(text: str, source_text: str, source_layout: list[dict]) -> tuple[str, list[dict]]:
    """Copy independently owned literal quantity/unit fields when both extended
    price owners agree. Existing complete/repeated pairs retain their proof;
    a source unit alone also needs a primary quantity and native row bounds.
    """
    primary = text.split("\n")
    source = normalize_fullwidth(source_text).split("\n")
    price_re = re.compile(r'(?:^|\s)([¥￥]?\s*\d[\d,]*)\s*((?:[*＊※%％A-Z]\s*)*)$')
    complete_re = re.compile(
        r'[\s(<\[]*(?:\d+\s*[個コ点]\s*[xX×Ⅹ]\s*(?:単|@)?\s*\d[\d,]*'
        r'|(?:単|@)\s*\d[\d,]*\s*[xX×Ⅹ]\s*\d+\s*[個コ点])[\s)>\]]*'
    )
    malformed_re = re.compile(r'[\s(<\[]*(\d{2,})[\s\]）)]*[xX×Ⅹ]\s*(?:単|@|#|₤)?\s*(\d[\d,]*)[\s)>\]]*')
    split_head_re = re.compile(r'[\s(<\[]*(\d+)\s*[個コ点]?\s*[xX×Ⅹ]\s*')
    split_unit_re = re.compile(r'[\s(<\[]*(?:単|@|#)?\s*(\d[\d,]*)\s*[)>\]]*')

    def title(line):
        value = price_re.sub("", line.strip())
        return "".join(char.casefold() for char in value if char.isalnum())

    def amount(line, *, standalone=False):
        match = price_re.fullmatch(line.strip()) if standalone else price_re.search(line.strip())
        if match is None or not ("¥" in match.group(1) or "￥" in match.group(1) or match.group(2)):
            return None
        return float(re.sub(r'[^0-9]', '', match.group(1)))

    def item_zone(lines):
        start = next((i + 1 for i, line in enumerate(lines)
                      if re.search(r'上記正に領収|担当者', re.sub(r'\s+', '', line))), 0)
        end = next((i for i in range(start, len(lines))
                    if _OCR_ZONE_END_RE.match(lines[i].strip())), len(lines))
        return start, end

    primary_start, primary_end = item_zone(primary)
    source_start, source_end = item_zone(source)
    proposals = {}
    def repeated_owner_proposal(owner, pair, pair_line, source_owners, primary_owners):
        # ponytail: one exact unit packet disambiguates repeats; all ties abstain.
        qty, unit = pair
        product = qty * unit
        rows = _group_layout_rows(source_layout)
        prices = _layout_row_price_candidates(source_layout)
        geometry_owners = [i for i, row in enumerate(rows)
                           if title(normalize_fullwidth(" ".join(block["text"] for block in row))) == owner]
        if len(geometry_owners) != len(source_owners):
            return None
        packets = []
        for i in geometry_owners:
            if i + 1 >= len(rows) or not _layout_rows_are_local(rows[i], rows[i + 1]):
                continue
            line = normalize_fullwidth(" ".join(block["text"] for block in rows[i + 1])).strip()
            detail = _parse_qty_detail_total(line) if complete_re.fullmatch(line) else None
            if detail and detail[1] == unit:
                packets.append((i, detail))
        if len(packets) != 1 or packets[0][1] != pair:
            return None
        geometry_owner = packets[0][0]
        owner_line = normalize_fullwidth(" ".join(block["text"] for block in rows[geometry_owner]))
        literal = price_re.search(owner_line)
        matches = [row for row in prices if "".join(c.casefold() for c in normalize_fullwidth(row["description"]) if c.isalnum()) == owner
                   and row["value"] == product]
        if (literal is None or float(re.sub(r'[^0-9]', '', literal[1])) != product
                or len(matches) != 1 or sum(row["value"] == product for row in prices) != 1
                or not any(float(block["x"]) == matches[0]["x"]
                           and _layout_block_center_y(block) == matches[0]["y"]
                           and _layout_price_value(block["text"], allow_small=True) == product
                           for block in rows[geometry_owner])):
            return None
        # The next complete native price owner bounds this literal detail in P.
        boundary_index = geometry_owner + 2
        if (boundary_index >= len(rows)
                or not _layout_rows_are_local(rows[geometry_owner + 1], rows[boundary_index])):
            return None
        boundary_line = normalize_fullwidth(" ".join(block["text"] for block in rows[boundary_index]))
        boundary = title(boundary_line)
        if (price_re.search(boundary_line) is None or not any(char.isalpha() for char in boundary)
                or not any("".join(c.casefold() for c in normalize_fullwidth(row["description"]) if c.isalnum()) == boundary
                           and any(float(block["x"]) == row["x"]
                                   and _layout_block_center_y(block) == row["y"]
                                   for block in rows[boundary_index]) for row in prices)):
            return None
        # A displaced total is consumed only as one exact standalone literal.
        amounts = [i for i in range(primary_start, primary_end)
                   if (match := price_re.fullmatch(primary[i].strip()))
                   and float(re.sub(r'[^0-9]', '', match[1])) == product]
        if len(amounts) != 1:
            return None
        details = []
        for i in primary_owners:
            following = [j for j in range(i + 1, primary_end) if primary[j].strip()][:2]
            if not following:
                continue
            line = primary[following[0]].strip()
            complete = _parse_qty_detail_total(line) if complete_re.fullmatch(line) else None
            if complete and complete[1] == unit:
                return None
            malformed = malformed_re.fullmatch(line)
            if malformed and float(malformed[2].replace(',', '')) == unit:
                if float(malformed[1]) * unit == product:
                    return None
                details.append((following[0], following[1] if len(following) == 2 else None))
        if (len(details) != 1 or details[0][1] is None
                or title(primary[details[0][1]]) != boundary):
            return None
        index = details[0][0]
        return {"line_index": index, "line_count": 1, "before": primary[index], "after": pair_line,
                "qty": qty, "unit_price": unit, "total": product, "title": owner}

    for pair_idx in range(source_start + 1, source_end):
        pair_line = source[pair_idx].strip()
        if complete_re.fullmatch(pair_line) is None:
            continue
        pair = _parse_qty_detail_total(pair_line)
        if pair is None:
            continue
        qty, unit = pair
        product = qty * unit
        owner_idx = pair_idx - 1
        owner = title(source[owner_idx])
        if not owner or not any(char.isalpha() for char in owner) or re.search(r'領収|合計|小計|消費税|対象|割引', owner):
            continue
        source_owners = [i for i in range(source_start, source_end) if title(source[i]) == owner]
        primary_owners = [i for i in range(primary_start, primary_end) if title(primary[i]) == owner]
        if source_owners != [owner_idx] or len(primary_owners) != 1:
            if len(source_owners) > 1 and len(source_owners) == len(primary_owners):
                proposal = repeated_owner_proposal(owner, pair, pair_line, source_owners, primary_owners)
                if proposal is not None:
                    proposals.setdefault(proposal["line_index"], []).append(proposal)
            continue
        source_amounts = [amount(source[owner_idx])]
        if owner_idx > source_start:
            source_amounts.append(amount(source[owner_idx - 1], standalone=True))
        source_amounts = [value for value in source_amounts if value is not None]
        if not source_amounts:
            for row in _group_layout_rows(source_layout):
                line = " ".join(str(block["text"]) for block in row)
                match = price_re.search(line)
                if match and title(line) == owner:
                    source_amounts.append(float(re.sub(r'[^0-9]', '', match.group(1))))
        if source_amounts != [product]:
            continue
        primary_owner = primary_owners[0]
        details = []
        owned_amounts = []
        split_tail = None
        for i in range(primary_owner, min(primary_end, primary_owner + 5)):
            if i == split_tail:
                continue
            line = primary[i].strip()
            if complete_re.fullmatch(line):
                details = []
                break
            head = split_head_re.fullmatch(line)
            if head:
                tail = split_unit_re.fullmatch(primary[i + 1].strip()) if i + 1 < primary_end else None
                if tail is None or (int(head.group(1)), float(tail.group(1).replace(',', ''))) != pair:
                    details = []
                    break
                details.append(i)
                split_tail = i + 1
                continue
            match = malformed_re.fullmatch(line)
            if match:
                literal = float(match.group(1)) * float(match.group(2).replace(',', ''))
                if literal == product:
                    details = []
                    break
                details.append(i)
                continue
            value = amount(line, standalone=i != primary_owner)
            if value is not None:
                owned_amounts.append(value)
            elif i != primary_owner and line:
                break
        if len(details) != 1 or owned_amounts != [product]:
            continue
        count = 2 if split_tail == details[0] + 1 else 1
        proposal = {"line_index": details[0], "line_count": count,
                    "before": "\n".join(primary[details[0]:details[0] + count]), "after": pair_line,
                    "qty": qty, "unit_price": unit, "total": product, "title": owner}
        proposals.setdefault(details[0], []).append(proposal)

    # ponytail: fuse only literal same-owner fields; incomplete bounds abstain.
    native_rows = _group_layout_rows(source_layout)
    native_lines = [normalize_fullwidth(" ".join(block["text"] for block in row)).strip()
                    for row in native_rows]
    native_start, native_end = item_zone(native_lines)
    native_packet_re = re.compile(
        r'(?:(?P<qty>\d+)\s*[個コ点]\s*メ\s*[単单]\s*'
        r'|[個コ点]\s*[xX×Ⅹ]\s*(?:単|@)?\s*)'
        r'(?P<unit>\d[\d,]*)\s*[¥￥]\s*(?P<total>\d[\d,]*)'
    )
    primary_partial_re = re.compile(
        r'[\s(<\[]*(?P<qty>\d+)\s*[xX×Ⅹ]\s*(?:単|@|#|₤)?\s*(?P<unit>\d[\d,]*)[\s)>\]]*'
    )
    alternate_re = re.compile(r'(?P<qty>\d+)\s*[個コ点]\s*メ\s*[単单]\s*(?P<unit>\d[\d,]*)')

    def literal_title(line, *, bare=False):
        value = normalize_fullwidth(line).strip()
        if bare and re.search(r'[¥￥]', value):
            return ''
        # Count inline-price purchases too; only explicit currency owns the suffix.
        value = re.sub(r'\s*[¥￥]\s*\d[\d,]*\s*[*＊※除軽外内]?\s*$', '', value).strip()
        if (re.search(r'[¥￥]|領収|合計|小計|消費税|対象|割引|値引', value)
                or _OCR_QTY_NOTATION_RE.search(value)
                or re.fullmatch(r'\d+\s*[xX×Ⅹ].*', value)):
            return ''
        return ''.join(char.casefold() for char in value if char.isalnum())

    for packet_index in range(native_start + 1, native_end):
        packet = native_packet_re.fullmatch(native_lines[packet_index])
        if packet is None:
            continue
        owner_index = packet_index - 1
        owner = literal_title(native_lines[owner_index], bare=True)
        if (len(owner) < 3 or not any(char.isalpha() for char in owner)
                or not _layout_rows_are_local(native_rows[owner_index], native_rows[packet_index])):
            continue
        source_owners = [i for i in range(native_start, native_end) if literal_title(native_lines[i]) == owner]
        primary_owners = [i for i in range(primary_start, primary_end) if literal_title(primary[i]) == owner]
        if source_owners != [owner_index] or len(primary_owners) != 1:
            continue
        following = [i for i in range(primary_owners[0] + 1, primary_end) if primary[i].strip()][:3]
        if len(following) < 2:
            continue
        detail = normalize_fullwidth(primary[following[0]]).strip()
        extended = re.fullmatch(r'[¥￥]\s*(\d[\d,]*)', normalize_fullwidth(primary[following[1]]).strip())
        if extended is None or complete_re.fullmatch(detail):
            continue
        boundary_index = packet_index + 1
        if boundary_index < native_end:
            boundary = _layout_row_price_candidates(native_rows[boundary_index])
            boundary_title = literal_title(boundary[0]['description']) if len(boundary) == 1 else ''
            if (len(following) != 3 or len(boundary) != 1 or len(boundary_title) < 3
                    or not any(char.isalpha() for char in boundary_title) or not boundary[0]['currency_owned']
                    or not _layout_rows_are_local(native_rows[packet_index], native_rows[boundary_index])
                    or literal_title(primary[following[2]]) != boundary_title):
                continue
        elif (len(following) != 2 or native_end == len(native_lines) or primary_end == len(primary)
                or not _layout_rows_are_local(native_rows[packet_index], native_rows[native_end])):
            continue
        elif (_compact(_OCR_ZONE_END_RE.match(primary[primary_end].strip())[0]).lstrip('()（）*＊※')
                != _compact(_OCR_ZONE_END_RE.match(native_lines[native_end])[0]).lstrip('()（）*＊※')):
            continue
        unit, total = int(packet['unit'].replace(',', '')), int(packet['total'].replace(',', ''))
        if int(extended[1].replace(',', '')) != total:
            continue
        if packet['qty'] is None:
            literal = primary_partial_re.fullmatch(detail)
            if literal is None or int(literal['qty']) * int(literal['unit'].replace(',', '')) == total:
                continue
            qty = int(literal['qty'])  # Printed primary quantity; never total/unit.
        else:
            literal = alternate_re.fullmatch(detail)
            if (literal is None or int(literal['qty']) != int(packet['qty'])
                    or int(literal['unit'].replace(',', '')) != unit):
                continue
            qty = int(literal['qty'])
        canonical = f'{qty}コX単{unit}'
        if qty * unit != total or _parse_qty_detail_total(canonical) != (qty, unit):
            continue
        index = following[0]
        proposals.setdefault(index, []).append({
            'line_index': index, 'line_count': 1, 'before': primary[index], 'after': canonical,
            'qty': qty, 'unit_price': unit, 'total': total, 'title': owner,
            'source_fields': {'qty': 'primary_detail', 'unit_price': 'native_source_detail'},
            'source_detail': native_lines[packet_index], 'source_owner_row': owner_index,
        })
    accepted = [values[0] for values in proposals.values() if len(values) == 1]
    for proposal in sorted(accepted, key=lambda row: row["line_index"], reverse=True):
        start = proposal["line_index"]
        primary[start:start + proposal["line_count"]] = [proposal["after"]]
    return "\n".join(primary), accepted


_LITERAL_PRICE_AMOUNT_RE = re.compile(r'[¥￥]\s*(\d[\d,]*)')
_LITERAL_PRICE_SUBTOTAL_RE = re.compile(r'小\s*計(?=$|[\s/：:¥￥\d(（])')
_LITERAL_PRICE_COUNT_RE = re.compile(r'[\s/：:()（）]*\d+\s*(?:点|個|件)?[\s/：:()（）]*')
_LITERAL_PRICE_END_RE = re.compile(r'^(?:小\s*計|合\s*計|現\s*計|税合計|外\s*税|内\s*税|消費税|お預り|お釣|釣銭|WAON|クレジット|お会計)')
_LITERAL_PRICE_SKIP_RE = re.compile(r'割引|値引|対象額|内税|外税|消費税|お預り|お釣|釣銭|会員|ポイント|税合計')
_LITERAL_PRICE_DISCOUNT_RE = re.compile(r'割引|値引')
_LITERAL_PRICE_QTY_RE = re.compile(r'(?:\d+\s*(?:コ|個|点)\s*[xX×Ⅹ]|\d+\s*[xX×Ⅹ]\s*(?:単|@)|(?:単|@)\s*\d[\d,]*\s*[xX×Ⅹ]\s*\d+)')
_LITERAL_PRICE_PERCENT_RE = re.compile(r'^[\s(（\[【]*\d+(?:\.\d+)?\s*[%％]\s*(?:引|off)?[\s)）\]】]*$', re.I)


def _literal_price_subtotal(text):
    lines, labels = (text or '').splitlines(), []
    for i, line in enumerate(lines):
        matches = list(_LITERAL_PRICE_SUBTOTAL_RE.finditer(line))
        if matches:
            labels.extend((i, match.end()) for match in matches)
        elif line.strip() == '小' and i + 1 < len(lines):
            split = re.match(r'\s*計(?=$|[\s/：:¥￥\d(（])', lines[i + 1])
            if split:
                labels.append((i + 1, split.end()))
    if len(labels) != 1:
        return None
    label_line, label_end = labels[0]
    body = lines[label_line][label_end:]
    for offset in range(3):
        if offset:
            body = lines[label_line + offset] if label_line + offset < len(lines) else ''
        matches = list(_LITERAL_PRICE_AMOUNT_RE.finditer(body))
        if matches:
            if len(matches) != 1 or body.count('¥') + body.count('￥') != 1:
                return None
            match = matches[0]
            if (body[:match.start()].strip() and not _LITERAL_PRICE_COUNT_RE.fullmatch(body[:match.start()])
                    or body[match.end():].strip().strip('-−ー')):
                return None
            return int(match.group(1).replace(',', ''))
        if not body.strip() or _LITERAL_PRICE_COUNT_RE.fullmatch(body):
            continue
        break
    return None


def _recover_closed_literal_prices(receipt, evidence, primary_text):
    """Only repair qty=1 prices in a complete, exact-owned inclusive basket.

    Every title occurs once in primary OCR and owns one explicit source-currency
    row. Both printed tax bases and the literal source total close the complete
    basket; quantity/discount details, unknown rows and duplicate owners veto.
    """
    if not primary_text or _literal_price_subtotal(primary_text) is not None:
        return None
    items, layout = receipt.get("line_items") or [], evidence.get("source_layout") or []
    owner = _collect_direct_summary_owners(layout)["unique"]["total"]
    if not items or owner is None or owner["value"] <= 0 or receipt.get("total") != owner["value"]:
        return None
    bases, ambiguous = _strict_rate_bases(evidence.get("merged_text") or "")
    base_values = {rate: {value} for rate, value in bases.items()}
    for line in (evidence.get("merged_text") or "").splitlines():
        match = re.fullmatch(r"\(?((?:8|10))%対象[¥￥](\d[\d,]*)\)?",
                             re.sub(r"\s+", "", normalize_fullwidth(line)))
        if match:
            base_values.setdefault(f"{match.group(1)}%", set()).add(int(match.group(2).replace(",", "")))
    ambiguous |= any(len(values) != 1 for values in base_values.values())
    bases = {rate: next(iter(values)) for rate, values in base_values.items() if len(values) == 1}
    if ambiguous or set(bases) != {"8%", "10%"} or any(v <= 0 for v in bases.values()) or sum(bases.values()) != owner["value"]:
        return None

    def key(text):
        return "".join(c.casefold() for c in normalize_fullwidth(str(text or "")) if c.isalnum())

    try:
        if any(not isinstance(item, dict) or float(item.get("qty")) != 1
               or float(item.get("discount") or 0) != 0 or item.get("discount_rate") not in (None, "")
               or not all(math.isfinite(float(item.get(field))) and float(item[field]) > 0
                          for field in ("unit_price", "total")) for item in items):
            return None
    except (ValueError, TypeError):
        return None
    keys = [key(item.get("description")) for item in items]
    if any(len(k) < 3 for k in keys) or len(set(keys)) != len(keys):
        return None
    primary_lines = [key(re.split(r"[¥￥]", line, maxsplit=1)[0]) for line in primary_text.splitlines()]
    if any(primary_lines.count(k) != 1 for k in keys):
        return None
    summary_y = min(_layout_block_center_y(layout[i]) for i in owner["amount_indices"])
    candidates = [row for row in _layout_row_price_candidates(layout) if row["y"] < summary_y]
    matched = [row for row in candidates if key(row["description"]) in keys]
    if len(matched) != len(keys) or len({key(row["description"]) for row in matched}) != len(keys):
        return None
    start_y = min(row["y"] for row in matched)
    if any(key(row["description"]) not in keys for row in candidates if row["y"] >= start_y):
        return None
    rows = _group_layout_rows(layout)
    for row in matched:
        blocks = [b for b in layout if float(b.get("x") or 0) == row["x"]
                  and _layout_block_center_y(b) == row["y"]
                  and _layout_price_value(str(b.get("text") or ""), allow_small=True) == row["value"]]
        if (len(blocks) != 1 or not row.get("currency_owned")
                or row.get("discount") or row.get("gross", row["value"]) != row["value"]):
            return None
        grouped = [r for r in rows if any(b is blocks[0] for b in r)]
        if len(grouped) != 1:
            return None
        position = next(i for i, b in enumerate(grouped[0]) if b is blocks[0])
        title = "".join(str(b.get("text") or "") for b in grouped[0][:position]).strip()
        if title.endswith(("¥", "￥")):
            title = title[:-1]
        # Printed product digits belong to the exact corroborated title; an
        # extra amount or quantity column cannot be silently discarded.
        if (re.search(r"[¥￥]", title) or key(title) != key(row["description"])
                or any(_layout_price_value(str(b.get("text") or ""), allow_small=True) is not None
                       for b in grouped[0][position + 1:])):
            return None
    body_rows = [row for row in rows if start_y <= min(_layout_block_center_y(b) for b in row) < summary_y]
    body = "\n".join("".join(str(b.get("text") or "") for b in row) for row in body_rows)
    primary_body = "\n".join(primary_text.splitlines()[min(primary_lines.index(k) for k in keys):])
    primary_body = re.split(r"(?:小\s*計|合\s*計)", primary_body, maxsplit=1)[0]
    if any(_LITERAL_PRICE_QTY_RE.search(text) or _LITERAL_PRICE_DISCOUNT_RE.search(text)
           or re.search(r"\d+\s*(?:個|コ|点)|\d+\s*[%％]|(?:^|\n)\s*[-−－]\s*[¥￥]?\d", text)
           for text in (body, primary_body)):
        return None
    if sum(row["value"] for row in matched) != owner["value"]:
        return None
    prices = {key(row["description"]): row["value"] for row in matched}
    staged = deepcopy(items)
    for item, title in zip(staged, keys, strict=True):
        item["unit_price"] = item["total"] = prices[title]
    return staged if staged != items else None


def _literal_price_packet_summary(primary_text, groups):
    """Own a complete counted mixed-mode summary without editing either OCR source."""
    header_re = re.compile(r'小計/?(\d+)(?:点|個)?[¥￥](\d[\d,]*)')
    component_re = re.compile(
        r'(\()?(8|10)%(?:(外税|内税|外|内)(タイショウ|対象(?:額)?)?|(税(?:額)?|消費税))'
        r'([¥￥*＊※])(\d[\d,.]*)(\))?'
    )
    info_re = re.compile(r'\(税合計[¥￥](\d[\d,]*)\)')
    native = [_cash_compact(''.join(block['text'] for block in group)) for group in groups]
    primary = [_cash_compact(line) for line in primary_text.splitlines() if line.strip()]

    def spans(lines, pattern, start=None):
        return [(i, i + count, match) for i in (range(len(lines)) if start is None else (start,))
                for count in range(1, min(3, len(lines) - i) + 1)
                if (match := pattern.fullmatch(''.join(lines[i:i + count])))]

    def read(lines):
        headers = spans(lines, header_re)
        if len(headers) != 1:
            return None
        start, end, match = headers[0]
        count, subtotal = int(match[1]), int(match[2].replace(',', ''))
        entries, position = [], end
        while position < len(lines):
            owners = [owner for owner in spans(lines, component_re, position) if owner[0] == position]
            if not owners:
                break
            if len(owners) != 1:
                return None
            begin, position, value = owners[0]
            if bool(value[1]) != bool(value[8]):
                return None
            kind = 'base' if value[4] else 'tax'
            mode = {'外': '外税', '内': '内税'}.get(value[3], value[3])
            if mode is None:
                if (kind != 'tax' or not entries or entries[-1]['kind'] != 'base'
                        or entries[-1]['rate'] != value[2] + '%'):
                    return None
                mode = entries[-1]['mode']
            if value[6] not in ('¥', '￥') and kind != 'base':
                return None
            entries.append(dict(rate=value[2] + '%', mode=mode, kind=kind,
                                options=_jpy_summary_amount_options(value[7]),
                                raw=''.join(lines[begin:position]), token=value[7],
                                opaque_prefix=value[6], start=begin, end=position))
        info = [owner for owner in spans(lines, info_re, position) if owner[0] == position]
        if len(info) != 1 or len(entries) < 4 or len(entries) % 2:
            return None
        rates = {}
        for index in range(0, len(entries), 2):
            base, tax = entries[index:index + 2]
            owner = (base['rate'], base['mode'])
            if (base['kind'] != 'base' or tax['kind'] != 'tax' or owner in rates
                    or (tax['rate'], tax['mode']) != owner):
                return None
            rates[owner] = (base, tax)
        if {mode for rate, mode in rates} != {'内税', '外税'}:
            return None
        return dict(start=start, end=end, count=count, subtotal=subtotal, groups=rates,
                    info=int(info[0][2][1].replace(',', '')), info_end=info[0][1])

    printed, source = read(primary), read(native)
    if (printed is None or source is None or printed['count'] != source['count']
            or printed['subtotal'] != source['subtotal'] or printed['count'] <= 0
            or printed['info'] != source['info'] or printed['groups'].keys() != source['groups'].keys()):
        return None
    values = {}
    for owner, (base, tax) in source['groups'].items():
        options = [(b, t) for b in base['options'] for t in tax['options']
                   if math.isfinite(b) and math.isfinite(t) and b > 0 and t > 0 and b == int(b) and t == int(t)
                   and _rate_base_tax_pair_is_valid(owner[0], b, t, owner[1])]
        if len(options) != 1:
            return None
        selected_base, selected_tax = options[0]
        pbase, ptax = printed['groups'][owner]
        primary_tax = [value for value in ptax['options'] if math.isfinite(value) and value > 0 and value == int(value)]
        primary_valid = [(b, t) for b in pbase['options'] for t in primary_tax
                         if math.isfinite(b) and b > 0 and b == int(b) and _rate_base_tax_pair_is_valid(owner[0], b, t, owner[1])]
        # A conflicting valid P base vetoes; an invalid printed observation is
        # retained as raw evidence while the independently valid S base closes.
        if primary_tax != [selected_tax] or primary_valid and primary_valid != options:
            return None
        values[owner] = (selected_base, selected_tax)
    if (sum(base for base, tax in values.values()) != source['subtotal']
            or sum(tax for base, tax in values.values()) != source['info']):
        return None
    tender = [(index, value) for index, row in enumerate(groups)
              if (value := _cash_owner(row, _CASH_TENDER_LABEL_RE))]
    change = [(index, value) for index, row in enumerate(groups)
              if (value := _cash_owner(row, _CASH_CHANGE_LABEL_RE))]
    if (len(tender) != 1 or len(change) != 1 or tender[0][0] != source['info_end'] + 1
            or change[0][0] != tender[0][0] + 1):
        return None
    total_row = native[source['info_end']]
    literal = _CASH_LITERAL_AMOUNT_RE.search(total_row)
    prefix = total_row[:literal.start()] if literal else ''
    if (literal is None or total_row.count('¥') + total_row.count('￥') != 1
            or not prefix or re.search(r'[\d¥￥%]', prefix)):
        return None
    total = int(literal[1].replace(',', ''))
    paid, returned = tender[0][1]['value'], change[0][1]['value']
    if (total <= 0 or paid - returned != total
            or source['subtotal'] + sum(tax for (rate, mode), (base, tax) in values.items() if mode == '外税') != total
            or any(not _layout_rows_are_local(groups[i], groups[i + 1])
                   for i in range(source['start'], change[0][0]))):
        return None
    # P may put three cash labels before their columns. Own only this complete
    # prefix; later loyalty amounts cannot supply a missing financial value.
    tail, amounts, labels = [], [], []
    for line in primary[printed['info_end']:]:
        tail.append(line)
        match = _CASH_LITERAL_AMOUNT_RE.fullmatch(line)
        if match:
            amounts.append(int(match[1].replace(',', '')))
        else:
            labels.append(line)
        if len(amounts) == 3:
            break
    if (amounts != [total, paid, returned] or len(labels) != 3
            or sum(bool(_CASH_TENDER_LABEL_RE.fullmatch(line)) for line in labels) != 1
            or sum(bool(_CASH_CHANGE_LABEL_RE.fullmatch(line)) for line in labels) != 1
            or any(re.search(r'[\d¥￥%]', line) for line in labels)):
        return None
    return dict(count=source['count'], subtotal=source['subtotal'], total=total,
                header=source['start'], primary=printed, native=source,
                raw_total=total_row, values=values)


def _recover_literal_price_bundle(receipt, evidence, primary_text):
    """Repair ordinary qty=1 prices or a fully closed literal packet bundle.

    The packet arm retains repeated purchase order and exact changed owner
    titles; complete printed money/count/mixed-summary closure owns every edit.
    The ordinary arm keeps its existing detail-free price proof.
    """
    items = receipt.get('line_items') if isinstance(receipt, dict) else None
    layout = evidence.get('source_layout') if isinstance(evidence, dict) else None
    subtotal = _literal_price_subtotal(primary_text)
    if subtotal is None:
        return _recover_closed_literal_prices(receipt, evidence, primary_text)
    if not isinstance(items, list) or not items or not isinstance(layout, list) or not layout:
        return None
    if not isinstance(subtotal, (int, float)) or isinstance(subtotal, bool) or not math.isfinite(subtotal) or subtotal <= 0:
        return None

    def key(text): return ''.join(c for c in normalize_fullwidth(str(text or '')).casefold() if c.isalnum())

    def digits(text): return tuple(re.findall(r'\d+', normalize_fullwidth(str(text or ''))))

    def number(value): return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None

    def score(a, b): return SequenceMatcher(None, key(a), key(b)).ratio() if digits(a) == digits(b) else 0.0

    def best(title, others):
        ranked = sorted(((score(title, row['title']), i) for i, row in enumerate(others)), reverse=True)
        return ranked[0][1] if ranked and ranked[0][0] >= .90 and (len(ranked) == 1 or ranked[0][0] - ranked[1][0] > .03) else None

    def detail(text): return bool(_LITERAL_PRICE_QTY_RE.search(text) or _LITERAL_PRICE_DISCOUNT_RE.search(text) or _LITERAL_PRICE_PERCENT_RE.fullmatch(text))

    def recover_packets():
        """Stage literal packets only after complete printed money/count closure."""
        groups = _group_layout_rows(layout)
        summary = _literal_price_packet_summary(primary_text, groups)
        if (summary is None or receipt.get('currency') != 'JPY'
                or receipt.get('subtotal') != summary['subtotal'] or receipt.get('total') != summary['total']):
            return None
        texts = [_cash_compact(''.join(block['text'] for block in group)) for group in groups]
        delimiter = [i for i, line in enumerate(texts[:summary['header']]) if re.search(r'上記正に領収|担当者', line)]
        if len(delimiter) != 1:
            return None
        literal_re = re.compile(r'([^¥￥]*)[¥￥](\d[\d,]*)([*＊※★☆軽除]?)')
        quantity_re = re.compile(
            r'(?:(\d+)[個コ点]メ[単单]|[個コ点][xX×Ⅹ](?:単|@)?)'
            r'(\d[\d,]*)[¥￥](\d[\d,]*)'
        )
        off_re = re.compile(r'(?:割引|値引)-[¥￥]?(\d[\d,.]*)')
        point_re = re.compile(r'(?:商品)?(?:単品|ボーナス)?P(?:\d+P)?|\d+P|(?:ボーナス)?ポイント\(?\d+P\)?')
        start = next((i for i in range(delimiter[0] + 1, summary['header'])
                      if literal_re.fullmatch(texts[i]) and not _SKIP_PRICE_LINE.search(texts[i])), None)
        if start is None:
            return None
        stop = next((i for i in range(start, summary['header']) if _OCR_ZONE_END_RE.match(texts[i])), summary['header'])
        packets, pending = [], None
        for index in range(start, stop):
            line = texts[index]
            if point_re.fullmatch(line):
                continue
            if match := off_re.fullmatch(line):
                options = [value for value in _jpy_summary_amount_options(match[1]) if value > 0 and value == int(value)]
                if (pending is not None or not packets or len(options) != 1 or packets[-1]['discount']
                        or index != packets[-1]['price_row'] + 1
                        or not _layout_rows_are_local(groups[index - 1], groups[index])
                        or options[0] >= packets[-1]['gross']):
                    return None
                packets[-1]['discount'] = options[0]
                continue
            if match := quantity_re.fullmatch(line):
                if (pending is None or pending[0] != index - 1
                        or not _layout_rows_are_local(groups[index - 1], groups[index])):
                    return None
                qty = int(match[1]) if match[1] else None
                unit, gross = int(match[2].replace(',', '')), int(match[3].replace(',', ''))
                packets.append(dict(title=pending[1], qty=qty, unit=unit, gross=gross, discount=0,
                                    owner_row=pending[0], price_row=index,
                                    raw_detail=line[:line.index('¥') if '¥' in line else line.index('￥')]))
                pending = None
                continue
            if match := literal_re.fullmatch(line):
                if pending is not None or not any(char.isalpha() for char in match[1]) or '%' in match[1]:
                    return None
                gross = int(match[2].replace(',', ''))
                packets.append(dict(title=match[1], qty=1, unit=gross, gross=gross, discount=0,
                                    owner_row=index, price_row=index, raw_detail='',
                                    prefixes={key(''.join(word['text'] for word in groups[index][:end]))
                                              for end in range(1, len(groups[index]) + 1)
                                              if not re.search(r'[¥￥]', ''.join(word['text'] for word in groups[index][:end]))}))
            elif (pending is None and any(char.isalpha() for char in line)
                  and not re.search(r'[¥￥%]|割引|値引', line)):
                pending = (index, line)
            else:
                return None
        if pending is not None or not packets or not any(packet['discount'] for packet in packets):
            return None
        if any(not 0 < packet['gross'] <= 99999 or not 0 < packet['unit'] <= 99999 for packet in packets):
            return None

        primary = [normalize_fullwidth(line).strip() for line in primary_text.splitlines() if line.strip()]
        primary_delimiter = [i for i, line in enumerate(primary) if re.search(r'上記正に領収|担当者', _cash_compact(line))]
        if len(primary_delimiter) != 1:
            return None
        pstop = next((i for i in range(primary_delimiter[0] + 1, len(primary))
                      if _OCR_ZONE_END_RE.match(primary[i])), len(primary))

        def pair(line):
            compact = _cash_compact(line)
            return _parse_qty_detail_total(compact) if _OCR_QTY_NOTATION_RE.fullmatch(compact) else None

        def title_text(line):
            return re.split(r'[¥￥]', line, maxsplit=1)[0].strip() if literal_re.fullmatch(_cash_compact(line)) else line

        def title_line(line):
            value = _cash_compact(title_text(line))
            return bool(any(char.isalpha() for char in value) and not re.search(r'[¥￥%]|割引|値引', value)
                        and not _SKIP_PRICE_LINE.search(value) and not pair(value) and not point_re.fullmatch(value))

        spans = [(i, i + count, ' '.join(title_text(line) for line in primary[i:i + count]))
                 for i in range(primary_delimiter[0] + 1, pstop) for count in (1, 2)
                 if i + count <= pstop and all(title_line(line) for line in primary[i:i + count])]
        selected, covered, proposed = [], set(), []
        cursor, source_index = primary_delimiter[0] + 1, 0
        for item in items:
            if not isinstance(item, dict) or source_index >= len(packets):
                return None
            packet = packets[source_index]
            title = key(item.get('description'))
            names = [row for row in spans if row[0] >= cursor and key(row[2]) == title]
            fused = [i for i in range(len(packets) - 1)
                     if key(packets[i]['title']) + key(packets[i + 1]['title']) == title]
            detail_suffix = bool(packet['raw_detail'] and title == key(packet['title']) + key(packet['raw_detail']))
            if detail_suffix:
                names = [row for row in spans if row[0] >= cursor and key(row[2]) == key(packet['title'])]
            if not names:
                return None
            anchor = min(names, key=lambda row: (row[0], row[1]))
            if fused:
                if (fused != [source_index] or anchor[1] != anchor[0] + 2
                        or source_index + 1 >= len(packets)
                        or key(title_text(primary[anchor[0]])) != key(packet['title'])
                        or key(title_text(primary[anchor[0] + 1])) != key(packets[source_index + 1]['title'])
                        or packets[source_index + 1]['owner_row'] != packet['price_row'] + 1
                        or not _layout_rows_are_local(groups[packet['price_row']], groups[packets[source_index + 1]['owner_row']])):
                    return None
                width = 2
            else:
                width = 1
            cursor = anchor[1]
            covered.update(range(anchor[0], anchor[1]))
            selected.append(anchor)
            for offset in range(width):
                row = packets[source_index + offset]
                begin = anchor[0] + offset if width == 2 else anchor[0]
                end = anchor[0] + offset + 1 if width == 2 else anchor[1]
                end_owner = min((candidate[0] for candidate in spans
                                 if candidate[0] >= end and key(candidate[2]) in
                                 ({key(value.get('description')) for value in items if isinstance(value, dict)}
                                  | {key(value['title']) for value in packets})), default=pstop)
                details = [(i, parsed) for i in range(end, end_owner) if (parsed := pair(primary[i]))]
                if row['raw_detail']:
                    if (len(details) != 1 or details[0][1][1] != row['unit']
                            or row['qty'] is not None and details[0][1][0] != row['qty']
                            or details[0][1][0] * row['unit'] != row['gross']):
                        return None
                    qty = details[0][1][0]
                    covered.add(details[0][0])
                elif details:
                    return None
                else:
                    qty = row['qty']
                candidate = deepcopy(item)
                candidate.update(qty=qty, unit_price=float(row['unit']), total=float(row['gross'] - row['discount']),
                                 discount=float(row['discount']))
                literal_title = ' '.join(title_text(line) for line in primary[begin:end])
                money_changed = any(candidate.get(field) != item.get(field) for field in ('qty', 'unit_price', 'total', 'discount'))
                if (digits(literal_title) != digits(row['title']) or item.get('discount_rate') not in (None, '')
                        or money_changed and key(literal_title) != key(row['title'])
                        or money_changed and sum(key(other['title']) == key(row['title']) for other in packets) != 1):
                    return None
                if width == 2 or detail_suffix:
                    if key(literal_title) != key(row['title']):
                        return None
                    candidate['description'] = literal_title
                proposed.append(candidate)
                row['qty'] = qty
            source_index += width
        if source_index != len(packets):
            return None
        pstart = primary_delimiter[0] + 1
        first_owner = min(anchor[0] for anchor in selected)
        prefix = [i for i in range(pstart, first_owner)
                  if not re.fullmatch(r'\(?消費税(?:等)?\d[\d,]*円を含みます\)?', _cash_compact(primary[i]))]
        if prefix:
            literal_prefix = ''.join(primary[i] for i in prefix)
            if (re.search(r'[\d¥￥%]', literal_prefix)
                    or key(literal_prefix) not in packets[0].get('prefixes', set())):
                return None
        covered.update(range(pstart, first_owner))
        positives, deductions = [], []
        negative_re = re.compile(r'-[¥￥]?(\d[\d,.]*)')
        for index in range(pstart, pstop):
            line = _cash_compact(primary[index])
            if (index in covered and not literal_re.fullmatch(line) or pair(line)
                    or point_re.fullmatch(line) or line in ('割引', '値引')):
                continue
            if match := literal_re.fullmatch(line):
                positives.append(int(match[2].replace(',', '')))
            elif match := negative_re.fullmatch(line):
                options = [value for value in _jpy_summary_amount_options(match[1]) if value > 0 and value == int(value)]
                if len(options) != 1:
                    return None
                deductions.append(options[0])
            else:
                return None
        if (positives != [row['gross'] for row in packets]
                or deductions != [row['discount'] for row in packets if row['discount']]
                or sum(row['qty'] for row in packets) != summary['count']
                or sum(item['total'] for item in proposed) != summary['subtotal']):
            return None
        return proposed if proposed != items else None

    packet_bundle = recover_packets()
    if packet_bundle is not None:
        return packet_bundle

    # Read explicit primary yen rows; an attached quantity or discount line
    # invalidates the preceding price until a new title/control takes ownership.
    primary, pending, active = [], None, None
    primary_lines = (primary_text or '').splitlines()
    for index, raw in enumerate(primary_lines):
        line = raw.strip()
        if (_LITERAL_PRICE_END_RE.match(line) or line == '小' and index + 1 < len(primary_lines)
                and primary_lines[index + 1].strip() == '計'):
            break
        if not line:
            continue
        if detail(line):
            if active is not None:
                primary.remove(active)
            active, pending = None, None
            continue
        if _LITERAL_PRICE_SKIP_RE.search(line):
            active, pending = None, None
            continue
        amounts = list(_LITERAL_PRICE_AMOUNT_RE.finditer(line))
        if amounts:
            active = None
            if len(amounts) != 1 or line.count('¥') + line.count('￥') != 1:
                pending = None
                continue
            match = amounts[0]
            if line[match.end():].strip().strip('-−ー'):
                pending = None
                continue
            title = line[:match.start()].strip().lstrip('*※＊').strip()
            if not title and pending and index - pending[0] <= 3:
                title = pending[1]
            try:
                value = int(match.group(1).replace(',', ''))
            except ValueError:
                pending = None
                continue
            if title and any(c.isalpha() for c in title) and 0 < value <= 99999:
                active = {'title': title, 'value': value}
                primary.append(active)
            pending = None
        elif any(c.isalpha() for c in line):
            active = None
            pending = (index, line.lstrip('*※＊').strip())
        else:
            active, pending = None, None

    # Keep only literal right-column blocks uniquely bound to a complete row.
    rows = _group_layout_rows(layout)
    source = []
    for candidate in _layout_row_price_candidates(layout):
        value = candidate.get('value')
        if (not isinstance(value, int) or isinstance(value, bool) or not 0 < value <= 99999
                or candidate.get('reduced_marker') or candidate.get('discount', 0)
                or candidate.get('gross', value) != value):
            continue
        matches = [(ri, bi) for ri, row in enumerate(rows) for bi, block in enumerate(row)
                   if abs(float(block.get('x') or 0) - float(candidate.get('x') or 0)) < .01
                   and abs(_layout_block_center_y(block) - float(candidate.get('y') or 0)) < .01
                   and _layout_price_value(str(block.get('text') or ''), allow_small=True) == value]
        if len(matches) != 1:
            continue
        ri, bi = matches[0]
        row_text = ''.join(str(block.get('text') or '') for block in rows[ri])
        amounts = [(j, amount) for j, block in enumerate(rows[ri])
                   if (amount := _layout_price_value(str(block.get('text') or ''), allow_small=True)) is not None]
        title = str(candidate.get('description') or '').strip()
        if (len(amounts) != 1 or amounts[0] != (bi, value) or detail(row_text)
                or any(marker in row_text for marker in ('%', '％'))
                or not any(c.isalpha() for c in title)):
            continue
        attached = False
        for following in rows[ri + 1:]:
            text = ''.join(str(block.get('text') or '') for block in following).strip()
            if not text:
                continue
            if detail(text):
                attached = True
                break
            if (_LITERAL_PRICE_END_RE.match(text) or _LITERAL_PRICE_SKIP_RE.search(text) or _LITERAL_PRICE_AMOUNT_RE.search(text)
                    or any(_layout_price_value(str(block.get('text') or ''), allow_small=True) is not None
                           for block in following)
                    or any(c.isalpha() for c in text) or any(c.isdigit() for c in text)):
                break  # next title, amount or control ends this owner's window
        if not attached:
            source.append({'title': title, 'value': value})

    pairs = {pi: si for pi, row in enumerate(primary)
             if (si := best(row['title'], source)) is not None and best(source[si]['title'], primary) == pi}
    staged, changed, blocked = deepcopy(items), False, False
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        if (number(item.get('qty')) != 1 or number(item.get('discount')) != 0
                or item.get('discount_rate') not in (None, '')):
            continue
        title, title_key = item.get('description'), key(item.get('description'))
        title_digits = digits(title)
        owned = [row for row in primary if key(row['title']) == title_key and digits(row['title']) == title_digits]
        unit, total = number(item.get('unit_price')), number(item.get('total'))
        exact_source_count = sum(key(row['title']) == title_key and digits(row['title']) == title_digits
                                 and row['value'] == unit for row in source)
        if (len(owned) == 1 and unit is not None and total == unit and owned[0]['value'] == unit
                and exact_source_count == 1):
            continue
        if len(owned) != 1:
            blocked |= bool(owned)
            continue
        pi, primary_row = primary.index(owned[0]), owned[0]
        si = pairs.get(pi)
        if si is None:
            blocked |= any(score(primary_row['title'], row['title']) >= .90 for row in source)
            continue
        source_row = source[si]
        if unit == total == source_row['value']:
            continue
        if unit is None or total is None:
            blocked = True
            continue
        staged[index]['unit_price'] = staged[index]['total'] = source_row['value']
        changed = True

    if blocked or not changed:
        return None
    totals = [number(item.get('total')) if isinstance(item, dict) else None for item in staged]
    if any(value is None for value in totals) or abs(sum(totals) - subtotal) > 1e-9:
        return None
    return staged


_CASH_SINGLE_BASE_RE = re.compile(r"\(?(8|10)%対象(?:額)?[¥￥](\d{1,3}(?:,\d{3})+|\d+)\)?")
_CASH_INNER_TAX_RE = re.compile(r"\(?内(?:消費税(?:等)?|税額)[¥￥](\d{1,3}(?:,\d{3})+|\d+)\)?")
_CASH_MALFORMED_TOTAL_RE = re.compile(r"(?:合計|総計|現計)[¥￥]\d+\.\d+")
_CASH_LITERAL_AMOUNT_RE = re.compile(r"[¥￥](\d{1,3}(?:,\d{3})+|\d+)円?$")
_CASH_ITEM_DETAIL_RE = re.compile(r"割引|値引|クーポン|\d+\s*(?:個|コ|点|×|x|X)|[%％]|(?:^|\n)[-−－][¥￥]?\d")


def _build_financial_source_identity(image, primary_text, evidence):
    """Bind a validated loaded capture to independently available original pixels.

    Injected-text callers prepare this from the original image and an
    authenticated capture, not from untrusted identity fields in that capture.
    It is an in-memory expectation, never another cache/acquisition artifact.
    """
    source = _validated_evidence(evidence)
    if source is None or not isinstance(primary_text, str):
        return None
    try:
        image_key, shape = _ocr_cache_key(image), list(image.shape[:2])
    except (AttributeError, TypeError, ValueError):
        return None
    if (source["image_key"] != image_key or source["image_shape"] != shape
            or source["strategy_fingerprint"] != SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT):
        return None
    return {"image_key": image_key, "image_shape": shape,
            "strategy_fingerprint": SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT,
            "primary_text_sha256": hashlib.sha256(primary_text.encode("utf-8")).hexdigest(),
            "source_layout_sha256": layout_blocks_sha256(source["source_layout"])}


def _cash_compact(value):
    return "".join(normalize_fullwidth(str(value or "")).split())


def _cash_title_key(value):
    return "".join(char.casefold() for char in _cash_compact(value) if char.isalnum())


def _cash_box(block):
    points = block["bbox"]
    return (min(p[0] for p in points), min(p[1] for p in points),
            max(p[0] for p in points), max(p[1] for p in points))


def _cash_overlap(first, second):
    ax, ay, ar, ab = _cash_box(first)
    bx, by, br, bb = _cash_box(second)
    intersection = max(0, min(ar, br) - max(ax, bx)) * max(0, min(ab, bb) - max(ay, by))
    area = min((ar - ax) * (ab - ay), (br - bx) * (bb - by))
    return intersection / area if area > 0 else 0


def _cash_owner(row, pattern):
    """Own one complete cash suffix using the existing payment grammar.

    A prefix token may be a repeated, overlapping copy of one owned label
    token. Record it; never edit OCR. Unowned text, digits and competitors veto.
    """
    for start in range(len(row)):
        owned = row[start:]
        text = _cash_compact("".join(block["text"] for block in owned))
        match = _CASH_LITERAL_AMOUNT_RE.search(text)
        if not match or text.count("¥") + text.count("￥") != 1 or not pattern.fullmatch(text):
            continue
        prefix = row[:start]
        label = [block for block in owned if not re.search(r"[\d¥￥]", block["text"])]
        if any(not any(
            (literal := _cash_compact(owner["text"]))
            and _cash_compact(extra["text"]) == literal * (len(_cash_compact(extra["text"])) // len(literal))
            and len(_cash_compact(extra["text"])) > len(literal)
            and _cash_overlap(extra, owner) >= .5
            for owner in label
        ) for extra in prefix):
            continue
        return {"value": int(match[1].replace(",", "")), "blocks": owned,
                "overlapping_repeated_blocks": prefix, "row_text": text}
    return None


def _recover_closed_cash_literal_table(receipt, evidence, primary_text, *, expected_source_identity=None):
    """Prove a complete single-rate inclusive basket independently of a bad total.

    Every item price is literal. The printed rate target is the proposed total;
    complete table sum and independently owned cash/change must equal it. Inner
    tax must fit only the inclusive formula. No subset, digit repair or residual
    price allocation is permitted. All financial fields commit together.
    """
    source = evidence
    if (not isinstance(receipt, dict) or source is None
            or source["strategy_fingerprint"] != SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT
            or receipt.get("document_type") != "receipt" or receipt.get("currency") != "JPY"
            or not isinstance(primary_text, str) or receipt.get("points_used") not in (None, 0)
            or receipt.get("payment_method") not in (None, "cash")):
        return None
    layout = source["source_layout"]
    if expected_source_identity != {
        "image_key": source["image_key"], "image_shape": source["image_shape"],
        "strategy_fingerprint": source["strategy_fingerprint"],
        "primary_text_sha256": hashlib.sha256(primary_text.encode("utf-8")).hexdigest(),
        "source_layout_sha256": layout_blocks_sha256(layout),
    }:
        return None
    groups = _group_layout_rows(layout)
    texts = [_cash_compact("".join(block["text"] for block in row)) for row in groups]
    bad = [(index, row) for index, row in enumerate(groups) if _CASH_MALFORMED_TOTAL_RE.fullmatch(texts[index])]
    summaries = _collect_direct_summary_owners(layout)
    bases = [(index, match) for index, text in enumerate(texts) if (match := _CASH_SINGLE_BASE_RE.fullmatch(text))]
    inner = [(index, match) for index, text in enumerate(texts) if (match := _CASH_INNER_TAX_RE.fullmatch(text))]
    if (len(bad) != 1 or summaries["owners"]["total"] or summaries["owners"]["subtotal"]
            or summaries["owners"]["tax"] or len(summaries["owners"]["tax_info"]) != 1
            or len(bases) != 1 or len(inner) != 1
            or sum(bool(re.search(r"\d+(?:\.\d+)?%.*対象", text)) for text in texts) != 1
            or re.search(r"外税|外枠|税抜対象|ポイント|クーポン", _cash_compact(primary_text) + "".join(texts))
            or _PAYMENT_TOKEN_RE.search("\n".join(texts))):
        return None
    stop = bad[0][0]
    base_index, base_match = bases[0]
    tax_index, tax_match = inner[0]
    rate, total = base_match[1] + "%", int(base_match[2].replace(",", ""))
    tax = int(tax_match[1].replace(",", ""))
    if (not 0 < tax < total or base_index != stop + 1 or tax_index != base_index + 1
            or not _rate_base_tax_pair_is_valid(rate, total, tax, "内税")
            or _rate_base_tax_pair_is_valid(rate, total, tax, "外税")):
        return None
    tender = [(index, owner) for index, row in enumerate(groups)
              if (owner := _cash_owner(row, _CASH_TENDER_LABEL_RE)) is not None]
    change = [(index, owner) for index, row in enumerate(groups)
              if (owner := _cash_owner(row, _CASH_CHANGE_LABEL_RE)) is not None]
    if (len(tender) != 1 or len(change) != 1 or tender[0][0] != tax_index + 1
            or change[0][0] != tender[0][0] + 1 or change[0][1]["value"] < 0
            or tender[0][1]["value"] <= total
            or tender[0][1]["value"] - change[0][1]["value"] != total):
        return None

    # A primary receipt/body delimiter bounds the complete table; no item subset.
    primary = [_cash_compact(line) for line in primary_text.split("\n") if line.strip()]
    starts = [index for index, text in enumerate(primary) if text == "領収証"]
    ends = [index for index, text in enumerate(primary) if text in ("合計", "総計", "現計")]
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        return None
    primary_body = primary[starts[0] + 1:ends[0]]
    primary_titles = [_cash_title_key(re.split(r"[¥￥]", text, maxsplit=1)[0]) for text in primary_body]
    delimiters = [index for index, text in enumerate(texts[:stop]) if "領収証" in text]
    if len(delimiters) != 1 or _CASH_ITEM_DETAIL_RE.search("\n".join(primary_body + texts[delimiters[0] + 1:stop])):
        return None
    body_groups = groups[delimiters[0] + 1:stop]
    prices = [row for row in _layout_row_price_candidates(layout)
              if any(row["x"] == float(block["x"]) and row["y"] == _layout_block_center_y(block)
                     for group in body_groups for block in group)]
    keys = [_cash_title_key(row["description"]) for row in prices]
    if (len(prices) < 2 or len(prices) != len(body_groups) or len(prices) != len(primary_titles)
            or any(len(key) < 3 for key in keys) or len(set(keys)) != len(keys)
            or keys != primary_titles or sum(row["value"] for row in prices) != total):
        return None
    for text, row in zip(primary_body, prices, strict=True):
        if re.search(r"[¥￥]", text):
            pieces = re.split(r"[¥￥]", text)
            if len(pieces) != 2 or _layout_price_value(pieces[1], allow_small=True) != row["value"]:
                return None
    owned_prices = []
    for row, group in zip(prices, body_groups, strict=True):
        tokens = [block for block in group if float(block["x"]) == row["x"]
                  and _layout_block_center_y(block) == row["y"]
                  and _layout_price_value(block["text"], allow_small=True) == row["value"]]
        if (len(tokens) != 1 or not row.get("currency_owned") or row["value"] <= 0
                or sum(block["text"].count("¥") + block["text"].count("￥") for block in group) != 1
                or row.get("discount") or row.get("gross") is not None):
            return None
        token = tokens[0]
        position = group.index(token)
        title = "".join(block["text"] for block in group[:position]).rstrip("¥￥")
        if (_cash_title_key(title) != _cash_title_key(row["description"])
                or any(re.search(r"[\d¥￥]", block["text"]) for block in group[position + 1:])):
            return None
        owned_prices.extend(group)
    current = receipt.get("line_items") or []
    current_keys = [_cash_title_key(item.get("description")) for item in current if isinstance(item, dict)]
    if (not 0 < len(current) < len(prices) or len(current_keys) != len(current)
            or len(set(current_keys)) != len(current_keys) or not set(current_keys) <= set(keys)):
        return None
    by_key = dict(zip(keys, prices, strict=True))
    try:
        if any(isinstance(item.get("qty"), bool) or float(item.get("qty")) != 1
               or float(item.get("discount") or 0) != 0 or item.get("discount_rate")
               or item.get("gross") is not None or item.get("tax_category") != rate
               or float(item.get("unit_price")) != by_key[key]["value"]
               or float(item.get("total")) != by_key[key]["value"]
               for item, key in zip(current, current_keys, strict=True)):
            return None
        if (float(receipt.get("total")) != sum(float(item["total"]) for item in current)
                or receipt.get("amount_paid") not in (None, receipt.get("total"))):
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    taxes = receipt.get("taxes") or []
    if len(taxes) != 1 or not isinstance(taxes[0], dict) or _cash_compact(taxes[0].get("rate")) != rate:
        return None
    used = (owned_prices + bad[0][1] + groups[base_index] + groups[tax_index]
            + tender[0][1]["blocks"] + change[0][1]["blocks"])
    height, width = source["image_shape"]
    if any(not math.isfinite(float(block["confidence"])) or block["confidence"] < .8
           or not (0 < (box := _cash_box(block))[0] < box[2] < width and 0 < box[1] < box[3] < height)
           for block in used):
        return None
    existing = dict(zip(current_keys, current, strict=True))
    proposed_items = []
    for key, row in zip(keys, prices, strict=True):
        item = deepcopy(existing.get(key, {"description": row["description"], "qty": 1,
                                           "discount": 0, "discount_rate": ""}))
        item.update(unit_price=float(row["value"]), total=float(row["value"]), tax_category=rate)
        proposed_items.append(item)
    proposal = {"line_items": proposed_items, "total": float(total),
                "taxes": [dict(taxes[0], rate=rate, label="内税", amount=float(tax))],
                "amount_paid": float(total), "payment_method": "cash"}
    proposal["subtotal"] = _canonical_subtotal_from_taxes(proposal)
    indices = {id(block): index for index, block in enumerate(layout)}
    proof = {"mode": "closed_single_rate_cash_literal_table",
             "source_indices": [indices[id(block)] for block in used],
             "overlapping_repeated_indices": [indices[id(block)] for block in tender[0][1]["overlapping_repeated_blocks"]],
             "malformed_total_row": texts[stop], "printed_target": total,
             "literal_item_sum": sum(row["value"] for row in prices),
             "printed_tender": tender[0][1]["value"], "printed_change": change[0][1]["value"],
             "printed_internal_tax": tax, "printed_rate": rate}
    return proposal, proof


_LITERAL_TABLE_COUNT_RE = re.compile('(?:購入点数|(?:お|御)買上(?:商品数|点数|げ点数)|(?:商品)?点数|商品数)[:：]?([0-9,]+)(?:点)?')


def _recover_counted_literal_packets(receipt, evidence, primary_text):
    """Missing rows require a complete inline table with proved qty packets.

    Each extended price is literal, title prefixes end at native word boundaries,
    and primary/source qty packets agree. Purchased units, both external rate
    groups and the complete subtotal must close before one partition is copied.
    """
    def key(value):
        return ''.join(c.casefold() for c in normalize_fullwidth(str(value or '')) if c.isalnum())

    def pair(text):
        text = normalize_fullwidth(text).strip(' <>()[\uFF08\uFF09]')
        return _parse_qty_detail_total(text) if _OCR_QTY_NOTATION_RE.fullmatch(text) else None

    items, layout = receipt.get('line_items') or [], evidence['source_layout']
    groups = _group_layout_rows(layout)
    texts = [''.join(block['text'] for block in group) for group in groups]
    if not primary_text or not items or not any(pair(text) for text in texts):
        return None
    summaries = _collect_direct_summary_owners(layout)
    subtotal, total = (summaries['unique'][role] for role in ('subtotal', 'total'))
    if not subtotal or not total:
        return None
    rows = [row for row in _layout_row_price_candidates(layout)
            if row['y'] < min(_layout_block_center_y(layout[index]) for index in subtotal['row_indices'])]
    keys = [key(row['description']) for row in rows]
    if not 0 < len(items) < len(rows) or any(len(value) < 3 for value in keys) or len(set(keys)) != len(keys):
        return None
    primary = [normalize_fullwidth(line).strip() for line in primary_text.splitlines() if line.strip()]
    stops = [i for i, line in enumerate(primary) if _LITERAL_PRICE_SUBTOTAL_RE.search(line)]
    if len(stops) != 1:
        return None
    stop = stops[0]
    packets, claimed, anchors = [], set(), []
    for row, title in zip(rows, keys, strict=True):
        literal = [block for block in layout if float(block['x']) == row['x']
                   and _layout_block_center_y(block) == row['y']
                   and _layout_price_value(block['text'], allow_small=True) == row['value']]
        positions = [i for i, group in enumerate(groups) if literal and any(block is literal[0] for block in group)]
        if (len(literal) != 1 or len(positions) != 1 or row.get('gross') is not None
                or row.get('discount') not in (None, 0) or type(row['value']) is not int or row['value'] <= 0):
            return None
        position, group = positions[0], groups[positions[0]]
        amount_index = group.index(literal[0])
        prefix = group[:amount_index]
        if prefix and prefix[-1]['text'].strip() in ('¥', '￥'):
            prefix = prefix[:-1]
        suffix = re.sub(r'^[¥￥]?\s*\d[\d,]*\s*', '', literal[0]['text'])
        suffix = re.sub(r'\s+', '', suffix + ''.join(block['text'] for block in group[amount_index + 1:]))
        if (key(''.join(block['text'] for block in prefix)) != title
                or any(re.search(r'[¥￥]', block['text']) for block in prefix)
                or not re.fullmatch(r'[*＊※%％xX軽除]*[A-Z]?', suffix)):
            return None
        # A primary whole title may omit trailing source words, never title digits
        # or part of a word. The global bijection below rejects shared prefixes.
        prefixes = {key(''.join(block['text'] for block in prefix[:end])) for end in range(1, len(prefix) + 1)}
        owners = []
        for index, line in enumerate(primary[:stop]):
            match = _OCR_TRAILING_PRICE_RE.search(line)
            titles = [line]
            if match and int(re.sub(r'\D', '', match[1])) == row['value']:
                titles.append(line[:match.start()])
            found = {key(candidate) for candidate in titles if key(candidate) in prefixes
                     and len(key(candidate)) >= 3 and re.findall(r'\d+', key(candidate)) == re.findall(r'\d+', title)}
            if found:
                if len(found) != 1:
                    return None
                owners.append((index, next(iter(found))))
        if len(owners) != 1:
            return None
        anchors.append(owners[0])
        claimed.add(position)
        quantity = (1., float(row['value']))
        if position + 1 < subtotal['row'] and (detail := pair(texts[position + 1])):
            if not _layout_rows_are_local(group, groups[position + 1]) or detail[0] * detail[1] != row['value']:
                return None
            quantity = detail
            claimed.add(position + 1)
        packets.append(dict(row=row, group=position, primary=owners[0], pair=quantity, suffix=suffix))
    if (len({anchor[0] for anchor in anchors}) != len(rows)
            or [anchor[0] for anchor in anchors] != sorted(anchor[0] for anchor in anchors)):
        return None
    point = re.compile(r'\(?(?:ボーナス)?ポイント\(?\d+P\)?')
    if any(index not in claimed and not point.fullmatch(re.sub(r'\s+', '', texts[index]))
           for index in range(min(packet['group'] for packet in packets), subtotal['row'])):
        return None
    primary_claimed = {anchor[0] for anchor in anchors}
    for index, packet in enumerate(packets):
        start = packet['primary'][0]
        end = anchors[index + 1][0] if index + 1 < len(anchors) else stop
        details = [(i, parsed) for i in range(start + 1, end) if (parsed := pair(primary[i]))]
        if packet['pair'][0] > 1:
            if (len(details) != 1 or details[0][1] != packet['pair'] or details[0][0] <= start
                    or _layout_price_value(primary[details[0][0] - 1], allow_small=True) != packet['row']['value']):
                return None
            primary_claimed.add(details[0][0])
        elif details:
            return None
    # All other primary body lines must be literal amount cells or complete
    # nonmonetary point controls. Unknown product/detail rows veto the table.
    index = min(primary_claimed)
    while index < stop:
        compact = re.sub(r'\s+', '', primary[index])
        if index in primary_claimed or re.fullmatch(r'[¥￥]?\d[\d,]*[*＊※%％除軽+A]?', compact):
            index += 1
        elif point.fullmatch(compact):
            index += 1
        elif index + 1 < stop and point.fullmatch(compact + re.sub(r'\s+', '', primary[index + 1])):
            index += 2
        else:
            return None
    native = '\n'.join(texts)
    if any(re.search(r'非課税|不課税|免税|\d[\d,]*\s*非\s*$|(?:^|[^\d])0(?:\.0+)?\s*%', line)
           for text in (primary_text, native) for line in text.splitlines()):
        return None
    financials = []
    for text in (primary_text, native):
        bases, ambiguous = _strict_rate_bases(text)
        entries = _interleaved_rate_tax_summary_entries(normalize_fullwidth(text).splitlines())
        expected = {(rate, kind, '外税') for rate in ('8%', '10%') for kind in ('base', 'tax')}
        if (ambiguous or set(bases) != {'8%', '10%'} or len(entries) != 4
                or {(rate, kind, mode) for rate, kind, value, mode in entries} != expected):
            return None
        values = {(rate, kind): value for rate, kind, value, mode in entries}
        if (any(value <= 0 or not math.isfinite(value) or int(value) != value for value in values.values())
                or bases != {rate: values[rate, 'base'] for rate in bases}
                or any(not _rate_base_tax_pair_is_valid(rate, values[rate, 'base'], values[rate, 'tax'], '外税') for rate in bases)
                or _literal_price_subtotal(text) != subtotal['value']):
            return None
        financials.append(values)
    values = financials[0]
    if (financials[1] != values or sum(row['value'] for row in rows) != subtotal['value']
            or sum(values[rate, 'base'] for rate in ('8%', '10%')) != subtotal['value']
            or subtotal['value'] + sum(values[rate, 'tax'] for rate in ('8%', '10%')) != total['value']
            or receipt.get('subtotal') != subtotal['value'] or receipt.get('total') != total['value']
            or _rate_amounts(receipt) != {rate: values[rate, 'tax'] for rate in ('8%', '10%')}
            or len(receipt.get('taxes') or []) != 2
            or any(not isinstance(tax, dict) or tax.get('label') != '外税'
                   for tax in receipt.get('taxes') or [])):
        return None
    unit_count = sum(packet['pair'][0] for packet in packets)
    excluded = sum(packet['pair'][0] for packet in packets
                   if _is_bag_description(packet['row']['description']) and packet['suffix'] == '除')
    for text in (primary_text, native):
        counts = _LITERAL_TABLE_COUNT_RE.findall(re.sub(r'\s+', '', normalize_fullwidth(text)))
        if len(counts) != 1 or int(counts[0].replace(',', '')) != unit_count - excluded:
            return None
    legend = re.compile(r'(?:[*＊※].{0,16}軽減税率8%|軽減税率8%.{0,16}[*＊※])')
    legends = [line for text in (primary_text, native) for line in text.splitlines()
               if legend.search(re.sub(r'\s+', '', line))]
    if not legends or any(re.search(r'軽減税率(?!8%)\d+(?:\.\d+)?%', re.sub(r'\s+', '', line))
                          for text in (primary_text, native) for line in text.splitlines()):
        return None
    current_keys = [key(item.get('description')) for item in items if isinstance(item, dict)]
    aliases = [{title, anchor[1]} for title, anchor in zip(keys, anchors, strict=True)]
    ownership = [[i for i, names in enumerate(aliases) if title in names] for title in current_keys]
    if (len(current_keys) != len(items) or len(set(current_keys)) != len(current_keys)
            or any(len(owners) != 1 for owners in ownership)
            or len({owners[0] for owners in ownership}) != len(items)
            or any(item.get('gross') is not None or item.get('discount') not in (None, 0)
                   or item.get('discount_rate') not in (None, '') for item in items)):
        return None
    staged = []
    existing = {owners[0]: item for owners, item in zip(ownership, items, strict=True)}
    for index, packet in enumerate(packets):
        item = deepcopy(existing.get(index, {'description': packet['row']['description'], 'discount': 0, 'discount_rate': ''}))
        item.update(qty=packet['pair'][0], unit_price=packet['pair'][1], total=float(packet['row']['value']))
        staged.append(item)
    locks = {}
    for index, item in enumerate(staged):
        locked = item.get('_tax_category_locked')
        if locked is not None:
            if not isinstance(locked, str) or locked not in {'8%', '10%'}:
                return None
            locks[index] = {locked}
    marker_scopes = ('\n'.join(primary[min(anchor[0] for anchor in anchors):stop] + legends),
                     '\n'.join(texts[min(packet['group'] for packet in packets):subtotal['row']] + legends))
    for text in marker_scopes:
        marked, indices = deepcopy(staged), set()
        _fix_tax_categories_from_ocr_markers(marked, text, locked_indices=indices)
        for index in indices:
            locks.setdefault(index, set()).add(marked[index]['tax_category'])
    if any(len(rates) != 1 or not rates <= {'8%', '10%'} for rates in locks.values()):
        return None
    # ponytail: at most 16 literal owners; counted DP is the upgrade for larger
    # unresolved baskets. Enumerate the complete standard group, never prices.
    if len(rows) > 16:
        return None
    solutions = []
    for size in range(1, len(rows)):
        for group in combinations(range(len(rows)), size):
            standard = set(group)
            if (sum(rows[index]['value'] for index in standard) == values['10%', 'base']
                    and all(('10%' if index in standard else '8%') in rates for index, rates in locks.items())):
                solutions.append(standard)
                if len(solutions) > 1:
                    return None
    if len(solutions) != 1:
        return None
    for index, item in enumerate(staged):
        item['tax_category'] = '10%' if index in solutions[0] else '8%'
    return staged


def _recover_complete_literal_table(receipt: dict, evidence: dict, primary_text: str | None) -> list[dict] | None:
    """Trigger: missing rows in a uniquely owned, literal qty=1 source table.

    Invariant: titles occur once in primary OCR; its printed count, subtotal
    and both marker-backed rate bases close the entire table. Never infer
    digits, choose a residual-price subset, or alter non-item fields.
    """
    counted = _recover_counted_literal_packets(receipt, evidence, primary_text)
    if counted is not None:
        return counted
    items = receipt.get("line_items") or []
    if not primary_text or not items:
        return None
    rows = _layout_row_price_candidates(evidence["source_layout"])

    def key(text):
        return "".join(char.casefold() for char in normalize_fullwidth(str(text or "")) if char.isalnum())

    keys = [key(row["description"]) for row in rows]
    current_keys = [key(item.get("description")) for item in items if isinstance(item, dict)]
    if (not len(items) < len(rows) or len(current_keys) != len(items)
            or len(set(current_keys)) != len(current_keys) or not set(current_keys) <= set(keys)
            or any(len(value) < 3 for value in keys) or len(set(keys)) != len(keys)
            or len({(row["x"], row["y"]) for row in rows}) != len(rows)):
        return None
    try:
        if any(float(item.get("qty")) != 1 or float(item.get("discount") or 0) != 0
               or item.get("discount_rate") not in (None, "") or item.get("gross") is not None for item in items):
            return None
        if any(not math.isfinite(float(row["value"])) or float(row["value"]) <= 0
               or row.get("gross") is not None or row.get("discount") not in (None, 0) for row in rows):
            return None
    except (TypeError, ValueError):
        return None
    # A qty-product candidate never stands in for a literal source price block.
    if any(len([block for block in evidence["source_layout"]
                if float(block["x"]) == row["x"] and _layout_block_center_y(block) == row["y"]
                and _layout_price_value(block["text"], allow_small=True) == row["value"]]) != 1 for row in rows):
        return None

    zones = []
    for text in (primary_text, evidence["merged_text"]):
        lines = normalize_fullwidth(text).splitlines()
        stop = next((index for index, line in enumerate(lines) if re.search(r"小\s*計", line)), len(lines))
        start = next((index for index, line in enumerate(lines[:stop]) if any(value in key(line) for value in keys)), stop)
        zones.append(lines[start:stop])
    ownership = [[value for value in keys if value in key(line)] for line in zones[0]]
    if any(sum(value in owners for owners in ownership) != 1 for value in keys) or any(len(owners) > 1 for owners in ownership):
        return None
    detail = r"\d+\s*[個コ点]?\s*[×xXⅩ]\s*(?:単|@|#)?\s*[¥￥]?\s*\d[\d,]*|割引|値引|クーポン|discount|(?:^|\n)\s*[-−－]\s*[¥￥]?\s*\d"
    if any(re.search(detail, "\n".join(zone), re.I) for zone in zones):
        return None
    primary_bases, primary_ambiguous = _strict_rate_bases(primary_text)
    source_bases, source_ambiguous = _strict_rate_bases(evidence["merged_text"])
    subtotal = extract_financial_totals(primary_text).get("subtotal")
    source_subtotal = extract_financial_totals(evidence["merged_text"]).get("subtotal")
    if (primary_ambiguous or source_ambiguous or primary_bases != source_bases
            or set(primary_bases) != {"8%", "10%"} or any(value <= 0 for value in primary_bases.values())
            or subtotal is None or subtotal != source_subtotal or sum(row["value"] for row in rows) != subtotal):
        return None
    compact = re.sub(r"\s+", "", normalize_fullwidth(primary_text))
    counts = {int(value.replace(",", "")) for value in _LITERAL_TABLE_COUNT_RE.findall(compact)}
    bags = sum(_is_bag_description(row["description"]) for row in rows)
    if len(counts) != 1 or next(iter(counts)) + bags != len(rows):
        return None
    if not re.search(r"(?:[*＊※].{0,16}軽減税率|軽減税率.{0,16}[*＊※])", compact):
        return None
    marker_items = [{"description": row["description"], "total": row["value"], "tax_category": "10%"} for row in rows]
    _fix_tax_categories_from_ocr_markers(marker_items, primary_text)
    marked = {value for value, row, item in zip(keys, rows, marker_items, strict=True)
              if row.get("reduced_marker") or item["tax_category"] == "8%"}
    sums = {rate: sum(row["value"] for value, row in zip(keys, rows, strict=True)
                     if ("8%" if value in marked else "10%") == rate) for rate in primary_bases}
    if sums != primary_bases:
        return None
    existing = dict(zip(current_keys, items, strict=True))
    proposed = []
    for value, row in zip(keys, rows, strict=True):
        item = deepcopy(existing.get(value, {"description": row["description"], "qty": 1, "discount": 0, "discount_rate": ""}))
        item.update(unit_price=float(row["value"]), total=float(row["value"]), tax_category="8%" if value in marked else "10%")
        proposed.append(item)
    return proposed


def _recover_quantity_owned_title(receipt, evidence, primary_text):
    """Restore a standalone title owning the sole qty/unit/extended-price block.

    Only the description changes. Primary qty * unit, item/total, and
    source extended price and summary must agree; source title must extend the
    adjacent primary fragment. A malformed source pair supplies no numbers.
    """
    items = receipt.get("line_items") or []
    if not isinstance(primary_text, str) or len(items) != 1 or not isinstance(items[0], dict):
        return None
    item = items[0]
    values = [item.get(field) for field in ("qty", "unit_price", "total")]
    if (any(not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value <= 0 for value in values)
            or values[0] <= 1 or values[0] * values[1] != values[2]
            or item.get("discount") or item.get("discount_rate")
            or receipt.get("total") != values[2]):
        return None
    qty, unit, amount = values
    lines = [line.strip() for line in normalize_fullwidth(primary_text).splitlines() if line.strip()]
    stop = next((index for index, line in enumerate(lines)
                 if re.match(r"小\s*計", line)), len(lines))
    owners = []
    for index in range(1, stop - 1):
        pair = _parse_qty_detail_total(lines[index])
        price = re.fullmatch(r"[¥￥]\s*(\d[\d,]*)\s*円?", lines[index + 1])
        fragment = lines[index - 1]
        if pair is not None and price is not None and re.fullmatch(r"[ぁ-んァ-ンー一-龥]+", fragment):
            if any(re.search(r"[¥￥]\s*\d", line) for line in lines[:index - 1]):
                return None
            owners.append((fragment, pair, float(price.group(1).replace(',', ''))))
    if len(owners) != 1 or owners[0][1] != (qty, unit) or owners[0][2] != amount:
        return None
    rows = _group_layout_rows(evidence["source_layout"])
    summaries = _collect_direct_summary_owners(evidence["source_layout"])["unique"]
    subtotal, total = summaries["subtotal"], summaries["total"]
    if (subtotal is None or total is None or subtotal["value"] != amount
            or total["value"] != amount or subtotal["row"] < 2):
        return None
    detail_index = subtotal["row"] - 1
    title_row, detail_row = rows[detail_index - 1], rows[detail_index]
    title = "".join(str(block.get("text") or "") for block in title_row).strip()
    detail = normalize_fullwidth(" ".join(str(block.get("text") or "") for block in detail_row))
    price = re.fullmatch(
        r"\s*(\d+)\s*[個コ点]?\s*[xX×Ⅹ]\s*(?:単|@|#)?\s*\d[\d,]*\s*"
        r"[¥￥]\s*(\d[\d,]*)\s*円?\s*", detail,
    )
    pair = _parse_qty_detail_total(detail)
    if (price is None or float(price.group(2).replace(',', '')) != amount
            or (pair is not None and pair != (qty, unit))
            or not _valid_ocr_item_desc(title)
            or not re.fullmatch(r"[ぁ-んァ-ンー一-龥]+", title)
            or not title.startswith(owners[0][0])
            or any(re.search(r"[¥￥]\s*\d", "".join(str(block.get("text") or "") for block in row))
                   for row in rows[:detail_index - 1])
            or not _layout_rows_are_local(title_row, detail_row)
            or not _layout_rows_are_local(detail_row, rows[subtotal["row"]])):
        return None
    return title


def _recover_counted_modifier_title(receipt, evidence, primary_text):
    """Trigger: one counted item with a priced title and unpriced modifiers.

    Invariant: only the description changes, preserving every printed option,
    all digits and one copy of a shared annotation. Primary OCR and the current
    description must corroborate every modifier; ambiguous owners abstain.
    """
    items = receipt.get("line_items") or []
    if not isinstance(primary_text, str) or len(items) != 1 or not isinstance(items[0], dict):
        return None
    item = items[0]
    if item.get("qty") != 1 or isinstance(item.get("qty"), bool) or item.get("discount") or item.get("discount_rate"):
        return None
    amounts = (item.get("total"), item.get("unit_price"), receipt.get("subtotal"))
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
           or not math.isfinite(value) or value <= 0 for value in amounts) or len(set(amounts)) != 1:
        return None
    amount = amounts[0]
    if len(re.findall(r'(?:本体合計|小計)\(1点\)', re.sub(r'\s+', '', normalize_fullwidth(primary_text)))) != 1:
        return None
    lines = normalize_fullwidth(evidence["merged_text"]).splitlines()
    summary = [(index, float(match.group(1).replace(',', '')))
               for index, line in enumerate(lines)
               if (match := re.fullmatch(r'(?:本体合計|小計)\(1点\)[¥￥]?(\d{1,3}(?:,\d{3})+|\d+)', re.sub(r'\s+', '', line)))]
    if len(summary) != 1 or summary[0][1] != amount:
        return None
    stop = summary[0][0]
    owners = []
    for index, line in enumerate(lines[:stop]):
        price = _OCR_TRAILING_PRICE_RE.search(line)
        if price is None or not re.search(r'[¥￥*＊※軽除外内]', price.group()):
            continue
        title = line[:price.start()].strip()
        if _valid_ocr_item_desc(title) and not _SKIP_PRICE_LINE.search(title):
            value = re.sub(r'[^\d,]', '', price.group(1))
            if not re.fullmatch(r'\d{1,3}(?:,\d{3})+|\d+', value):
                return None
            value = float(value.replace(',', ''))
            owners.append((index, title, value))
    if len(owners) != 1 or owners[0][2] != amount:
        return None
    start, title, _ = owners[0]
    details = [line.strip() for line in lines[start + 1:stop] if line.strip()]
    if not details or any(not _valid_ocr_item_desc(line) or _SKIP_PRICE_LINE.search(line)
                          or re.search(r'\d|[¥￥%@×]|https?://|TEL', line, re.IGNORECASE)
                          for line in details):
        return None
    annotations = [re.search(r'\s*(\([^()]+\))\s*$', detail) for detail in details]
    if len(details) > 1 and all(annotations) and len({re.sub(r'\s+', '', match.group(1)) for match in annotations}) == 1:
        literal = " ".join([title, *(detail[:match.start()].strip() for detail, match in zip(details, annotations)),
                            annotations[0].group(1)])
    else:
        literal = " ".join([title, *details])
    norm = lambda value: ''.join(char.casefold() for char in value if char.isalnum())
    current, source = norm(str(item.get("description") or "")), norm(literal)
    if len(current) < 3 or re.findall(r'\d+', current) != re.findall(r'\d+', source):
        return None
    primary_key = norm(normalize_fullwidth(primary_text))
    for detail in details:
        core = norm(re.sub(r'\([^()]*\)', '', detail))
        if not core or core not in current or core not in primary_key:
            return None
    source_characters = iter(source)
    if not all(any(letter == candidate for candidate in source_characters) for letter in current):
        return None
    return re.sub(r'\s+', ' ', literal).strip()


def _recover_owned_discount_rates(
    receipt: dict, evidence: dict, primary_text: str | None = None
) -> list[dict]:
    """Copy a literal rate only from its unique title/gross/discount row owner."""
    items = receipt.get("line_items") or []
    blocks = evidence["source_layout"]
    groups = _group_layout_rows(blocks)
    amount_end = re.compile(r"(?:[¥￥]\s*)?(\d[\d,]*)\s*(?:[※＊*Xx%％↓軽除]\s*)*$")
    negative = re.compile(r"-\s*[¥￥\\]?\s*(\d[\d,]*)")

    def key(value: object) -> str:
        text = normalize_fullwidth(str(value or ""))
        return "".join(char.casefold() for char in text if char.isalnum())

    row_texts = [" ".join(str(block.get("text") or "") for block in group) for group in groups]
    declaration = re.compile(r"上記(?:金額)?正に領収(?:いたしました|しました)?")
    starts = [index + 1 for index, line in enumerate(row_texts) if declaration.fullmatch(key(line))]
    if starts and any(_OCR_ZONE_END_RE.match(key(line)) for line in row_texts[:starts[0] - 1]):
        primary_lines = normalize_fullwidth(primary_text or "").split('\n')
        primary_starts = [index + 1 for index, line in enumerate(primary_lines)
                          if declaration.fullmatch(key(line))]
        if len(starts) != 1 or len(primary_starts) != 1:
            return []
        start = starts[0]
        end = next((index for index in range(start, len(groups))
                    if _OCR_ZONE_END_RE.match(key(row_texts[index]))), len(groups))
        subtotal = _literal_price_subtotal('\n'.join(row_texts[start:]))
        try:
            nets = [float(item["total"]) for item in items]
            closed = (nets and all(math.isfinite(value) and value > 0 for value in nets)
                      and sum(nets) == float(receipt["subtotal"]) == subtotal
                      == _literal_price_subtotal('\n'.join(primary_lines[primary_starts[0]:])))
        except (KeyError, TypeError, ValueError):
            closed = False
        if not closed or end == len(groups) or not _LITERAL_PRICE_SUBTOTAL_RE.match(row_texts[end]):
            return []
        # ponytail: reopen only this closed detail section; retain its first summary boundary.
        groups = groups[start:end]
        blocks = [block for group in groups for block in group]
    price_candidates = _layout_row_price_candidates(blocks)

    def primary_owns_price_suffix(
        model_title: object, source_title: object, gross: int
    ) -> bool:
        source_key = key(source_title)
        model_text = normalize_fullwidth(str(model_title or "")).strip()
        if not primary_text or not source_key or not model_text:
            return False
        owners = 0
        primary_lines = normalize_fullwidth(primary_text).splitlines()
        for index, line in enumerate(primary_lines[:-1]):
            if key(line.strip()) != source_key:
                continue
            annotation = primary_lines[index + 1].strip()
            amount = amount_end.fullmatch(annotation)
            if amount is None or int(amount.group(1).replace(",", "")) != gross:
                continue
            model_match = re.fullmatch(r"(?P<title>.+?)\s+" + re.escape(annotation), model_text)
            if model_match is not None and key(model_match.group("title")) == source_key:
                owners += 1
        return owners == 1

    source_rows = []
    for group_index, group in enumerate(groups):
        item_line = " ".join(str(block.get("text") or "") for block in group)
        if re.search(r"割\s*引|値\s*引", item_line) or negative.search(item_line):
            continue
        amount_match = amount_end.search(item_line)
        if amount_match is None:
            continue
        title_key = key(item_line[:amount_match.start()])
        gross = int(amount_match.group(1).replace(",", ""))
        if len(title_key) < 3:
            continue
        matching_prices = [
            row for row in price_candidates
            if key(row.get("description")) == title_key
            and float(row.get("value") or 0) == gross
        ]
        if len(matching_prices) != 1 or group_index + 1 >= len(groups):
            continue
        schedule_lines = []
        quantity_details = set()
        valid = True
        for following in groups[group_index + 1:group_index + 9]:
            discount_line = " ".join(str(block.get("text") or "") for block in following)
            if _OCR_ZONE_END_RE.match(discount_line):
                break
            next_owners = [
                row for row in price_candidates
                if len(key(row.get("description"))) >= 3
                and key(row.get("description")) != title_key
                and key(discount_line).startswith(key(row.get("description")))
                and any(row["x"] == float(block.get("x") or 0)
                        and row["y"] == _layout_block_center_y(block) for block in following)
            ]
            if (len(next_owners) == 1 and not negative.search(discount_line)
                    and not re.search(r"割\s*引|値\s*引|会\s*員", discount_line)):
                break
            detail = _parse_qty_detail_total(discount_line)
            if detail and not re.sub(r'[\d.,¥￥@xX×個単<>()（）>\s]', '', discount_line):
                quantity_details.add(detail)
                schedule_lines.append(discount_line)
                continue
            if re.fullmatch(r'\s*\d{10,14}\s*', discount_line) and not schedule_lines:
                continue
            rates = _discount_rate_tokens(discount_line)
            residue = re.sub(r"\d+(?:\.\d+)?\s*[%％]", "", discount_line)
            residue = negative.sub("", residue)
            residue = re.sub(r"会\s*員(?:\s*様)?\s*割(?:\s*引)?|割\s*引|値\s*引|会\s*員(?:\s*様)?", "", residue)
            if re.search(r"\d|[A-Za-zぁ-んァ-ン一-龥]", residue):
                valid = not bool(re.search(r'[%％]|-\s*[¥￥\\]?\s*\d', discount_line))
                break
            if re.search(r'[%％]', discount_line) and not rates:
                valid = False
                break
            schedule_lines.append(discount_line)
        negatives = [int(match.group(1).replace(",", ""))
                     for line in schedule_lines for match in negative.finditer(line)]
        if not valid or not negatives:
            continue
        discount = sum(negatives)
        source_rows.append({
            "title": matching_prices[0].get("description"),
            "title_key": title_key,
            "gross": gross,
            "discount": discount,
            "net": gross - discount,
            # Geometry already owns this complete title and numeric price.
            # Keep the literal schedule; X/% price marks cannot act as rates.
            "schedule_text": str(matching_prices[0]["description"]) + "\n" + str(gross) + "\n" + "\n".join(schedule_lines),
            "quantity_details": quantity_details,
            "group_index": group_index,
        })

    proposals = []
    for item_index, item in enumerate(items):
        if not isinstance(item, dict) or not item.get("discount"):
            continue
        title_key = key(item.get("description"))
        try:
            qty = float(item.get("qty") or 1)
            unit = float(item.get("unit_price"))
            total = float(item.get("total"))
            discount = float(item.get("discount"))
        except (TypeError, ValueError):
            continue
        if qty <= 0 or unit <= 0:
            continue
        matches = [
            (row_index, row) for row_index, row in enumerate(source_rows)
            if (
                row["title_key"] == title_key
                or primary_owns_price_suffix(
                    item.get("description"), row.get("title"), row["gross"]
                )
            )
            and row["gross"] == qty * unit
            and row["discount"] == discount
            and math.isclose(total, row["net"], rel_tol=0, abs_tol=1e-9)
            and (not row["quantity_details"] or row["quantity_details"] == {(qty, unit)})
            and (qty == 1 or row["quantity_details"] == {(qty, unit)})
        ]
        if len(matches) == 1:
            row_index, row = matches[0]
            owned = [dict(item, description=row["title"], discount_rate="")]
            _clear_discounts_without_nearby_ocr_marker(owned, row["schedule_text"], rates_only=True)
            rate = _discount_rate_tokens(owned[0]["discount_rate"], full_match=True)
            if len(rate) == 1:
                proposals.append((item_index, row_index, rate[0]))

    item_counts: dict[int, int] = {}
    row_counts: dict[int, int] = {}
    for item_index, row_index, _rate in proposals:
        item_counts[item_index] = item_counts.get(item_index, 0) + 1
        row_counts[row_index] = row_counts.get(row_index, 0) + 1
    accepted = []
    for item_index, row_index, rate in proposals:
        if item_counts[item_index] != 1 or row_counts[row_index] != 1:
            continue
        item = items[item_index]
        rendered = f"{rate:g}%"
        current = str(item.get("discount_rate") or "").strip()
        if current and current != rendered:
            continue
        changed = current != rendered
        if changed:
            item["discount_rate"] = rendered
        accepted.append({"item_index": item_index, "source_row_index": row_index, "rate": rendered, "changed": changed})
    return accepted


def _recover_native_title(receipt, source, primary_text, native, context):
    """Copy only a full title whose prefix was read in its source-owned cell."""
    fingerprint = NATIVE_TITLE_CROP_OCR_FINGERPRINT
    crop = _validated_quantity_crop(native, context, fingerprint=fingerprint)
    if (crop is None or receipt.get("currency") != "JPY" or not isinstance(primary_text, str)
            or source["image_key"] != context["image_key"] or source["image_shape"] != context["image_shape"]
            or source["strategy_fingerprint"] != SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT
            or layout_blocks_sha256(source["source_layout"]) != context["source_layout_sha256"]
            or hashlib.sha256(normalize_fullwidth(primary_text).encode("utf-8")).hexdigest() != context["input_text_sha256"]):
        return None
    selected = _native_title_crop_candidate(primary_text, source["source_layout"], source["image_shape"])
    if selected is None or any(context[name] != value for name, value in selected.items()):
        return None
    source_detail = _native_title_detail([{"text": selected["quantity_row_text"]}])
    if source_detail is None:
        return None
    qty, unit, gross = source_detail
    items = receipt.get("line_items") or []
    if len(items) != 1 or not isinstance(items[0], dict):
        return None
    item = items[0]
    source_summaries = _collect_direct_summary_owners(source["source_layout"])["owners"]
    source_tax = source_summaries["tax"][0] if len(source_summaries["tax"]) == 1 else None
    if source_tax is None:
        return None
    try:
        if (type(item.get("qty")) is bool or float(item["qty"]) != qty
                or type(item.get("unit_price")) is bool or float(item["unit_price"]) <= 0
                or float(item["unit_price"]) != unit or float(item["unit_price"]) * float(item["qty"]) != gross
                or float(item.get("total")) != gross
                or item.get("tax_category") != source_tax["rate"] or item.get("discount") not in (None, 0, 0.0)
                or item.get("discount_rate") not in (None, "")
                or (item.get("gross") is not None and float(item["gross"]) != gross)
                or float(receipt.get("subtotal")) != gross):
            return None
        taxes = receipt.get("taxes") or []
        totals = [owner for owner in source_summaries["total"]
                  if re.match(r"^(?:合計|総計)", owner["row_text"])]
        if (source_tax is None or len(totals) != 1 or float(receipt.get("total")) != totals[0]["value"]
                or len(taxes) != 1 or taxes[0].get("rate") != source_tax["rate"]
                or taxes[0].get("label") != "外税" or float(taxes[0].get("amount")) != source_tax["value"]):
            return None
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError):
        return None
    source_rows = _group_layout_rows(source["source_layout"])
    suffix = selected["owner_title"]
    source_matches = []
    for row in source_rows:
        ordered = sorted(row, key=lambda block: block["x"])
        codes = [i for i, block in enumerate(ordered)
                 if re.fullmatch(r"\d{3,6}", _cash_compact(block["text"]))]
        if len(codes) == 1 and codes[0] + 1 < len(ordered):
            title_blocks = ordered[codes[0] + 1:]
            if _cash_compact("".join(block["text"] for block in title_blocks)) == suffix:
                source_matches.append((ordered[codes[0]], _native_row_box(title_blocks), _cash_compact(ordered[codes[0]]["text"])))
    if len(source_matches) != 1:
        return None
    source_code, source_suffix, code = source_matches[0]
    primary_codes = {line[:-len(suffix)] for raw in normalize_fullwidth(primary_text).splitlines()
                     if re.fullmatch(r"\d+" + re.escape(suffix), line := _cash_compact(raw))}
    native_rows = _group_layout_rows(crop["source_layout"])
    native_matches = []
    for row in native_rows:
        ordered = sorted(row, key=lambda block: block["x"])
        codes = [i for i, block in enumerate(ordered)
                 if _cash_compact(block["text"]) in (primary_codes | {code})
                 and _cash_overlap(block, source_code) >= 0.5]
        if len(codes) == 1 and codes[0] == 0:
            native_matches.append((ordered, codes[0]))
    if len(native_matches) != 1:
        return None
    ordered, code_idx = native_matches[0]
    title_cells = ordered[code_idx + 1:]
    if len(title_cells) not in (1, 2):
        return None
    if len(title_cells) == 1:
        native_title = normalize_fullwidth(title_cells[0]["text"]).strip()
        compact_title = _cash_compact(native_title)
        prefix = compact_title[:-len(suffix)] if compact_title.endswith(suffix) else ""
        suffix_box = title_cells[0]
        if not _cash_box(source_code)[2] <= _cash_box(title_cells[0])[0] < _cash_box(source_suffix)[0]:
            return None
    else:
        prefix = _cash_compact(title_cells[0]["text"])
        compact_suffix = _cash_compact(title_cells[1]["text"])
        native_title = normalize_fullwidth(title_cells[0]["text"] + title_cells[1]["text"]).strip()
        compact_title = prefix + compact_suffix
        suffix_box = title_cells[1]
        prefix_box, source_gap_end = _cash_box(title_cells[0]), _cash_box(source_suffix)[0]
        if not (_cash_box(source_code)[2] <= prefix_box[0] < prefix_box[2] <= source_gap_end):
            return None
    if (not compact_title.endswith(suffix) or compact_title == suffix
            or not prefix or not prefix.isalpha()
            or _cash_overlap(suffix_box, source_suffix) < 0.5):
        return None
    primary_compact = _cash_compact(item.get("description"))
    primary_owners = {_cash_compact(line) for line in normalize_fullwidth(primary_text).splitlines()
                      if _cash_compact(line).endswith(suffix)
                      and re.fullmatch(r"\d+" + re.escape(suffix), _cash_compact(line))}
    if primary_compact not in ({suffix, compact_title} | primary_owners):
        return None
    return native_title if native_title != item.get("description") else None


def _recover_native_price_column(receipt, source, primary_text, native, context, source_identity):
    """One complete external-tax table may select only observed column digits.

    Exact full-word P/S titles own changed packets. An unchanged current title
    may retain the existing mutually unique 90% title correspondence. Physical
    price-cell order, the independently matched transaction header and body
    inventory, qty packets, count, both rate bases and
    grand total must close uniquely; neither confidence nor residuals invent a
    price. The sealed capture remains untouched.
    """
    fingerprint = NATIVE_PRICE_COLUMN_OCR_FINGERPRINT
    capture = _validated_quantity_crop(native, context, fingerprint=fingerprint)
    if capture is None or not isinstance(primary_text, str) or receipt.get("currency") != "JPY":
        return None, {}
    layout = source["source_layout"]
    expected = dict(image_key=source["image_key"], image_shape=source["image_shape"],
        strategy_fingerprint=SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT,
        primary_text_sha256=hashlib.sha256(primary_text.encode("utf-8")).hexdigest(),
        source_layout_sha256=layout_blocks_sha256(layout))
    if (source_identity != expected or source["strategy_fingerprint"] != expected["strategy_fingerprint"]
            or any(context[name] != expected[name] for name in ("image_key", "image_shape", "source_layout_sha256"))
            or context["roi"] != _native_price_column_candidate(layout, source["image_shape"])):
        return None, {}
    groups = _group_layout_rows(layout)
    texts = [_cash_compact("".join(word["text"] for word in group)) for group in groups]
    summaries = _collect_direct_summary_owners(layout)
    subtotal, total = (summaries["unique"][role] for role in ("subtotal", "total"))
    components = summaries["owners"]["rate_component"]
    roles = {(rate, kind) for rate in ("8%", "10%") for kind in ("base", "tax")}
    if (subtotal is None or total is None or len(components) != 4
            or {(row["rate"], row["kind"]) for row in components} != roles
            or any(row["mode"] != "外税" for row in components)):
        return None, {}
    values = {(row["rate"], row["kind"]): row["value"] for row in components}
    bases = {rate: values[rate, "base"] for rate in ("8%", "10%")}
    primary_bases, ambiguous = _strict_rate_bases(primary_text)
    primary_taxes = [(rate, value, mode) for rate, kind, value, mode in
                     _interleaved_rate_tax_summary_entries(normalize_fullwidth(primary_text).splitlines()) if kind == "tax"]
    positive_taxes = {rate: values[rate, "tax"] for rate in bases if values[rate, "tax"] > 0}
    non_tax = []
    for text in (primary_text, "\n".join(texts)):
        found = re.findall(r"(?m)^非課税対象額\n?[¥￥](\d[\d,]*)$",
                           "\n".join(_cash_compact(line) for line in text.splitlines()))
        if len(found) > 1:
            return None, {}
        non_tax.append(int(found[0].replace(",", "")) if found else 0)
    extra_taxes = [tax for tax in receipt.get("taxes") or [] if isinstance(tax, dict) and tax.get("rate") not in bases]
    if (ambiguous or primary_bases != bases or non_tax[0] != non_tax[1]
            or any(base <= 0 for base in bases.values())
            or any(not _rate_base_tax_pair_is_valid(rate, bases[rate], values[rate, "tax"], "外税") for rate in bases)
            or {rate: amount for rate, amount, mode in primary_taxes} != positive_taxes
            or len(primary_taxes) != len(positive_taxes) or any(mode != "外税" for rate, amount, mode in primary_taxes)
            or any(values[rate, "tax"] == 0 and not re.search(r"(?m)^[¥￥]0$", primary_text) for rate in bases)
            or _literal_price_subtotal(primary_text) != subtotal["value"]
            or sum(bases.values()) + non_tax[0] != subtotal["value"]
            or subtotal["value"] + sum(positive_taxes.values()) != total["value"]
            or receipt.get("subtotal") != subtotal["value"] or receipt.get("total") != total["value"]
            or _rate_amounts(receipt) != positive_taxes
            or any(not isinstance(tax, dict) or tax.get("label") != "外税"
                   for tax in receipt.get("taxes") or [] if tax not in extra_taxes)
            or extra_taxes and (not non_tax[0] or len(extra_taxes) != 1 or extra_taxes[0].get("rate") != "0%"
                               or extra_taxes[0].get("label") != "非課税" or extra_taxes[0].get("amount") != 0)):
        return None, {}
    primary = [normalize_fullwidth(line).strip() for line in primary_text.splitlines() if line.strip()]
    stops = [i for i, line in enumerate(primary) if _LITERAL_PRICE_SUBTOTAL_RE.search(line)]
    if len(stops) != 1:
        return None, {}
    stop = stops[0]
    point = re.compile(r"\(?(?:ボーナス)?ポイント\(?\d+P\)?")
    money = re.compile(r"[¥￥]?(\d[\d,]*)([*＊※%％xX軽除非-]*)")
    glyph = re.compile(r"[¥￥*＊※%％xX軽除非-]+")

    def pair(text):
        text = _cash_compact(text).strip(" <>()[]（）")
        return _parse_qty_detail_total(text) if _OCR_QTY_NOTATION_RE.fullmatch(text) else None

    names = []
    for i, line in enumerate(primary[:stop]):
        if pair(line) or _LITERAL_PRICE_SKIP_RE.search(line) or point.fullmatch(_cash_compact(line)):
            continue
        match = re.search(r"[¥￥]?\s*\d[\d,]*\s*[*＊※%％xX軽除非]*$", line)
        for name in {line, line[:match.start()].strip() if match else line}:
            key = _cash_title_key(name)
            if len(key) >= 3 and any(char.isalpha() for char in key):
                names.append((i, name, key))
    price_rows = [row for row in _layout_row_price_candidates(layout)
                  if row["y"] < min(_layout_block_center_y(layout[i]) for i in subtotal["row_indices"])]
    anchors = {id(word) for row in price_rows for word in layout if float(word["x"]) == row["x"]
               and _layout_block_center_y(word) == row["y"]
               and _layout_price_value(word["text"], allow_small=True) == row["value"]}
    start = next((i for i, group in enumerate(groups) if any(id(word) in anchors for word in group)), None)
    if start is None or start >= subtotal["row"]:
        return None, {}
    items = receipt.get("line_items") or []
    if not items or any(not isinstance(item, dict) or item.get("gross") is not None
                        or item.get("discount") not in (None, 0) or item.get("discount_rate") not in (None, "") for item in items):
        return None, {}
    packets, title_words, control_words, points = [], [], [], []
    for position in range(start, subtotal["row"]):
        group, text = groups[position], texts[position]
        if re.search(r"割引|値引|クーポン|^[-−－][¥￥]?\d", text):
            return None, {}
        if point.fullmatch(text):
            control_words.extend(group)
            points.append(text)
            continue
        detail = pair(text)
        if detail:
            if (not packets or position != packets[-1]["row"] + 1 or packets[-1]["pair"] is not None
                    or not _layout_rows_are_local(groups[position - 1], group)):
                return None, {}
            packets[-1]["pair"] = detail
            control_words.extend(group)
            continue
        prefixes = [(end, _cash_title_key("".join(word["text"] for word in group[:end])))
                    for end in range(1, len(group) + 1) if not any(id(word) in anchors for word in group[:end])]
        matches = [(i, name, key, end) for end, prefix in prefixes for i, name, key in names if key == prefix]
        fuzzy = False
        if not matches and any(not money.fullmatch(_cash_compact(word["text"]))
                               and not glyph.fullmatch(_cash_compact(word["text"])) for word in group):
            ranked = {}
            for end, prefix in prefixes:
                for i, name, key in names:
                    if re.findall(r"\d+", prefix) == re.findall(r"\d+", key):
                        score = SequenceMatcher(None, prefix, key).ratio()
                        if (i, key) not in ranked or (score, end) > ranked[i, key][:2]:
                            ranked[i, key] = (score, end, i, name, key)
            ranked = sorted(ranked.values(), reverse=True)
            if ranked and ranked[0][0] >= .90 and (len(ranked) == 1 or ranked[0][0] - ranked[1][0] > .03):
                score, end, i, name, key = ranked[0]
                matches, fuzzy = [(i, name, key, end)], True
        if not matches:
            if any(not money.fullmatch(_cash_compact(word["text"])) and not glyph.fullmatch(_cash_compact(word["text"])) for word in group):
                return None, {}
            continue
        if len({(i, key) for i, name, key, end in matches}) != 1:
            return None, {}
        i, name, key, end = max(matches, key=lambda match: match[3])
        source_key = _cash_title_key("".join(word["text"] for word in group[:end]))
        boundary = min(position for position, prefix in prefixes if prefix == source_key)
        if re.search(r"[-−－][¥￥]?\d", _cash_compact("".join(word["text"] for word in group[boundary:]))):
            return None, {}
        packets.append(dict(row=position, primary=i, title=name, key=key, words=group[:end], pair=None, fuzzy=fuzzy))
        title_words.extend(group[:end])
    if not packets or len({packet["primary"] for packet in packets}) != len(packets) or len({packet["key"] for packet in packets}) != len(packets):
        return None, {}
    def transaction_header(lines):
        found = []
        for i, line in enumerate(lines):
            dates = list(_DATE_LINE_RE.finditer(line))
            if len(dates) != 1:
                continue
            date = dates[0]
            times = [(j, f"{int(match[1]):02d}:{match[2]}")
                     for j in range(i, min(i + 3, len(lines)))
                     for match in _TIME_HHMM_RE.finditer(lines[j][date.end():] if j == i else lines[j])
                     if int(match[1]) < 24]
            if len(times) == 1:
                start = i - int(i > 0 and re.fullmatch(r"レジ\d+", _cash_compact(lines[i - 1])) is not None)
                found.append((_cash_compact(date[0]), times[0][1], times[0][0], start))
        return found[0] if len(found) == 1 else None

    first = min(packet["primary"] for packet in packets)
    pheader, sheader = transaction_header(primary[:first]), transaction_header(texts[:start])
    if pheader is None or sheader is None or pheader[:2] != sheader[:2]:
        return None, {}
    reference = re.compile(r"(?:取(?:引)?|No\.?)\d+(?:[^\W\d_]?:\d+)+", re.IGNORECASE)
    register = re.compile(r"レジ\d+")
    date_packet = re.compile(_DATE_LINE_RE.pattern + r"(?:\s*[(（][月火水木金土日][)）])?")

    def header_control(line):
        fragment = _cash_compact(_TIME_HHMM_RE.sub(" ", date_packet.sub(" ", line)))
        fragment = reference.sub("", fragment)
        fragment = re.sub(r"(?:レジ|取(?:引)?)\d+", "", fragment)
        return re.fullmatch(r"[\d:#./()（）]*", fragment) is not None

    pre_price_titles = {_cash_title_key("".join(word["text"] for word in group[:end]))
                        for group in groups[sheader[3] + 1:start] for end in range(1, len(group) + 1)}
    if pre_price_titles & {key for i, _name, key in names if i > pheader[3] and not header_control(primary[i])}:
        return None, {}
    # Header tails may contain numeric register references, never unowned
    # letter-bearing packets. Their complete digit groups must agree in P/S.
    tails = [primary[pheader[3]:first], texts[sheader[3]:start]]
    registers = []
    for column, (lines, header, tail) in enumerate(zip((primary, texts), (pheader, sheader), tails, strict=True)):
        owners = register.findall("\n".join(_cash_compact(
            _TIME_HHMM_RE.sub(" ", date_packet.sub(" ", line))) for line in tail))
        if not owners and header[3] > 0:
            previous = ([lines[header[3] - 1]] if column == 0 else
                        ["".join(word["text"] for word in groups[header[3] - 1][:end])
                         for end in range(1, len(groups[header[3] - 1]) + 1)])
            owners = list({match[0] for fragment in previous if (match := register.fullmatch(_cash_compact(fragment)))})
        registers.append(owners)
    if len(registers[0]) > 1 or registers[0] != registers[1]:
        return None, {}
    tail_numbers = [re.findall(r"\d+", "\n".join(register.sub("", _cash_compact(
        _TIME_HHMM_RE.sub(" ", date_packet.sub(" ", line)))) for line in tail)) for tail in tails]
    if (any(not header_control(line) for tail in tails for line in tail)
            or tail_numbers[0] != tail_numbers[1]):
        return None, {}
    owned = {id(word) for word in title_words + control_words}
    remaining = [word for group in groups[start:subtotal["row"]] for word in group if id(word) not in owned]
    if any(not money.fullmatch(_cash_compact(word["text"])) and not glyph.fullmatch(_cash_compact(word["text"])) for word in remaining):
        return None, {}
    source_money = [word for word in remaining if money.fullmatch(_cash_compact(word["text"]))]
    native_money = [word for word in capture["source_layout"] if money.fullmatch(_cash_compact(word["text"]))
                    and any(_cash_overlap(word, owner) >= .5 for owner in source_money)]
    native_currency = [word for word in capture["source_layout"] if _cash_compact(word["text"]) in {"¥", "￥"}]

    def immediately_left(first, second):
        sx, sy, sr, sb = _cash_box(first)
        x, y, right, bottom = _cash_box(second)
        return sx + sr < x + right and x - sr <= max(sb - sy, bottom - y) and min(sb, bottom) > max(sy, y)

    for sign in capture["source_layout"]:
        if not re.fullmatch(r"[¥￥]?[-−－][¥￥]?", _cash_compact(sign["text"])):
            continue
        if any(immediately_left(sign, word) or any(immediately_left(sign, currency) and immediately_left(currency, word)
                for currency in native_currency) for word in native_money):
            return None, {}
    observed = []
    for i, word in enumerate(capture["source_layout"]):
        if (re.fullmatch(r"[¥￥]?[-−－][¥￥]?\d[\d,]*[*＊※%％xX軽除非-]*", _cash_compact(word["text"]))
                and any(_cash_overlap(word, owner) >= .5 for owner in source_money)):
            return None, {}
        if any(_cash_overlap(word, owner) >= .5 for owner in title_words + control_words):
            continue
        literal = money.fullmatch(_cash_compact(word["text"]))
        if literal:
            value = int(literal[1].replace(",", ""))
            if value <= 0 or not any(_cash_overlap(word, owner) >= .5 for owner in source_money):
                return None, {}
            observed.append((word, value, i))
        elif not glyph.fullmatch(_cash_compact(word["text"])):
            return None, {}
    cells = []
    for word, value, index in sorted(observed, key=lambda entry: _layout_block_center_y(entry[0])):
        if cells and any(_cash_overlap(word, previous) >= .5 for previous in cells[-1]["words"]):
            cell = cells[-1]
        else:
            cell = dict(words=[], values=set(), indices=[], source=[])
            cells.append(cell)
        cell["words"].append(word)
        cell["values"].add(value)
        cell["indices"].append(index)
    if len(cells) != len(packets):
        return None, {}
    for word in source_money:
        owners = [cell for cell in cells if any(_cash_overlap(word, observed) >= .5 for observed in cell["words"])]
        if len(owners) != 1:
            return None, {}
        owners[0]["source"].append(word)
    if any(not cell["source"] or abs(median(_layout_block_center_y(word) for word in packet["words"])
             - median(_layout_block_center_y(word) for word in cell["words"]))
             > median(_cash_box(word)[3] - _cash_box(word)[1] for word in packet["words"] + cell["words"])
           for packet, cell in zip(packets, cells, strict=True)):
        return None, {}
    claimed = {packet["primary"] for packet in packets}
    for packet in packets:
        end = min((p["primary"] for p in packets if p["primary"] > packet["primary"]), default=stop)
        details = [(i, pair(primary[i])) for i in range(packet["primary"] + 1, end) if pair(primary[i])]
        if (packet["pair"] is None and details or packet["pair"] is not None
                and (len(details) != 1 or details[0][1] != packet["pair"])):
            return None, {}
        claimed.update(i for i, parsed in details)
    ppoints = []
    for i in range(min(claimed), stop):
        text = _cash_compact(primary[i])
        if i in claimed or money.fullmatch(text):
            continue
        ppoints.append(text)
    if "".join(ppoints) != "".join(points):
        return None, {}
    unit_count = sum(packet["pair"][0] if packet["pair"] else 1 for packet in packets)
    bag_count = sum(packet["pair"][0] if packet["pair"] else 1 for packet in packets if _is_bag_description(packet["title"]))
    counts = [_LITERAL_TABLE_COUNT_RE.findall(_cash_compact(text)) for text in (primary_text, "\n".join(texts))]
    if (any(len(count) != 1 for count in counts) or counts[0] != counts[1]
            or int(counts[0][0].replace(",", "")) not in {unit_count, unit_count - bag_count}
            or any(not re.search(r"(?:[*＊※].{0,16}軽減税率8%|軽減税率8%.{0,16}[*＊※])", _cash_compact(text))
                   for text in (primary_text, "\n".join(texts)))):
        return None, {}
    locks = {}
    for i, cell in enumerate(cells):
        center = median(_layout_block_center_y(word) for word in cell["words"])
        height = median(_cash_box(word)[3] - _cash_box(word)[1] for word in cell["words"])
        right = max(_cash_box(word)[2] for word in cell["words"])
        markers = [word["text"] for word in remaining + capture["source_layout"]
                   if abs(_layout_block_center_y(word) - center) <= height * .55
                   and right - height <= _cash_box(word)[0] <= right + height]
        if any("非" in marker for marker in markers):
            if not non_tax[0] or not any("非" in word["text"] for word in remaining
                                       if abs(_layout_block_center_y(word) - center) <= height * .55) or not any(
                    "非" in word["text"] and abs(_layout_block_center_y(word) - center) <= height * .55
                    for word in capture["source_layout"]):
                return None, {}
            locks[i] = "0%"
        elif any(re.search(r"[*＊※]", marker) for marker in markers):
            locks[i] = "8%"
    # ponytail: bounded literal alternatives and at most 16 unlocked standard
    # owners; counted DP is the upgrade if larger ambiguous tables occur.
    if math.prod(len(cell["values"]) for cell in cells) > 256:
        return None, {}
    solutions = []
    for amounts in product(*(sorted(cell["values"]) for cell in cells)):
        if (sum(amounts) != subtotal["value"]
                or sum(amount for i, amount in enumerate(amounts) if locks.get(i) == "0%") != non_tax[0]
                or any(packet["pair"] and packet["pair"][0] * packet["pair"][1] != amount
                       for packet, amount in zip(packets, amounts, strict=True))):
            continue
        eligible = [i for i, amount in enumerate(amounts) if i not in locks and amount <= bases["10%"]]
        if len(eligible) > 16:
            return None, {}
        for size in range(1, len(eligible) + 1):
            for group in combinations(eligible, size):
                if sum(amounts[i] for i in group) == bases["10%"]:
                    solutions.append((amounts, set(group)))
                    if len(solutions) > 1:
                        return None, {}
    if len(solutions) != 1:
        return None, {}
    amounts, standard = solutions[0]
    staged = []
    for i, (packet, amount) in enumerate(zip(packets, amounts, strict=True)):
        existing = [item for item in items if _cash_title_key(item.get("description")) == packet["key"]]
        category = "0%" if locks.get(i) == "0%" else "10%" if i in standard else "8%"
        qty, unit = packet["pair"] if packet["pair"] else (1, amount)
        if packet["fuzzy"] and (len(existing) != 1 or existing[0].get("qty") != qty
                or existing[0].get("unit_price") != unit or existing[0].get("total") != amount
                or existing[0].get("tax_category") != category or len(cells[i]["values"]) != 1):
            return None, {}
        item = deepcopy(existing[0]) if len(existing) == 1 else {"description": packet["title"]}
        item.update(qty=qty, unit_price=unit, total=amount, discount=0, discount_rate="", tax_category=category)
        staged.append(item)
    if staged == items:
        return None, {}
    indices = {id(word): i for i, word in enumerate(layout)}
    proof = dict(mode="native_price_column_complete_table", strategy_fingerprint=fingerprint, roi=deepcopy(context["roi"]),
        printed_subtotal=subtotal["value"], printed_count=int(counts[0][0].replace(",", "")), literal_unit_count=unit_count,
        transaction_header=dict(date=pheader[0], time=pheader[1], register=registers[0],
                                primary_end=pheader[2], source_end=sheader[2]),
        owners=[dict(title=packet["title"], title_source_indices=[indices[id(word)] for word in packet["words"]],
                     price_source_indices=[indices[id(word)] for word in cell["source"]], native_indices=cell["indices"],
                     observed_candidates=sorted(cell["values"]), selected=amount, unchanged_title_correspondence=packet["fuzzy"])
                for packet, cell, amount in zip(packets, cells, amounts, strict=True)])
    return staged, proof


def apply_supplemental_ocr_evidence(
    receipt: dict, evidence: dict, *, primary_text: str | None = None,
    financial_source_identity: dict | None = None,
    native_owner_ocr_evidence: dict | None = None, native_owner_context: dict | None = None,
    pixel_marker_context: dict | None = None,
) -> tuple[dict, dict]:
    """Apply only fully owned identity, tax, item-price or title proposals."""
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
    table = _recover_complete_literal_table(receipt, validated, primary_text)
    if table is not None:
        output["line_items"] = table
        accepted_fields = [field for field in accepted_fields if field != "tax_categories"]
        accepted_fields.append("line_items")
    prices = _recover_literal_price_bundle(output, validated, primary_text) if table is None else None
    if prices is not None:
        output["line_items"] = prices
        accepted_fields = [field for field in accepted_fields if field != "tax_categories"]
        accepted_fields.append("line_items")
    financial = _recover_closed_cash_literal_table(
        receipt, validated, primary_text, expected_source_identity=financial_source_identity,
    ) if table is None and prices is None else None
    if financial is not None:
        fields, _proof = financial
        output.update(fields)
        accepted_fields = [field for field in accepted_fields if field != "tax_categories"]
        accepted_fields.extend(("line_items", "financial_summary"))
    native_meta = {}
    if table is None and prices is None and financial is None:
        if isinstance(native_owner_context, dict) and native_owner_context.get("strategy_fingerprint") == NATIVE_PRICE_COLUMN_OCR_FINGERPRINT:
            prices, native_meta = _recover_native_price_column(
                output, validated, primary_text, native_owner_ocr_evidence, native_owner_context, financial_source_identity)
        else:
            prices, native_meta = _recover_native_qty1_owner_prices(
                output, validated, primary_text, native_owner_ocr_evidence, native_owner_context)
        if prices is not None:
            output["line_items"] = prices
            accepted_fields = [field for field in accepted_fields if field != "tax_categories"]
            accepted_fields.append("line_items")
    if not tax_meta["accepted"] and financial is None and native_meta.get("mode") != "native_price_column_complete_table":
        partial = _partial_marker_projection(output, validated, primary_text=primary_text)
        partition = _mixed_mode_marker_partition(output, validated, primary_text, partial)
        proposal = partition if partition["accepted"] else partial
        if proposal["accepted"]:
            for item, category in zip(output["line_items"], proposal["proposed"], strict=True):
                item["tax_category"] = category
            tax_meta = proposal
            accepted_fields.append("tax_categories")
    pixel_description_updates = []
    if pixel_marker_context is not None:
        from .receipt_pixel_markers import recover_pixel_marker_tax_partition
        pixel_proposal = recover_pixel_marker_tax_partition(output, validated, primary_text, pixel_marker_context)
        if pixel_proposal["accepted"]:
            for item, category in zip(output["line_items"], pixel_proposal["proposed"], strict=True):
                item["tax_category"] = category
            tax_meta = pixel_proposal
            if "tax_categories" not in accepted_fields:
                accepted_fields.append("tax_categories")
            pixel_description_updates = pixel_proposal.get("description_updates") or []
            for update in pixel_description_updates:
                output["line_items"][update["item_index"]]["description"] = update["description"]
            if pixel_description_updates and "line_items" not in accepted_fields:
                accepted_fields.append("line_items")
    quantity_title = _recover_quantity_owned_title(output, validated, primary_text)
    title = quantity_title if quantity_title is not None else _recover_counted_modifier_title(output, validated, primary_text)
    title_changed = title is not None and title != output["line_items"][0].get("description")
    if title_changed:
        output["line_items"][0]["description"] = title
        if "line_items" not in accepted_fields:
            accepted_fields.append("line_items")
    rate_proposals = _recover_owned_discount_rates(output, validated, primary_text)
    if any(proposal["changed"] for proposal in rate_proposals) and "line_items" not in accepted_fields:
        accepted_fields.append("line_items")
    before_code_cleanup = deepcopy(output.get("line_items") or [])
    _clean_code_prefixed_item_descriptions(output, primary_text or "", validated["source_layout"])
    code_cleanup_changed = output.get("line_items") != before_code_cleanup
    if code_cleanup_changed and "line_items" not in accepted_fields:
        accepted_fields.append("line_items")
    merchant = _select_header_logo(
        validated["merged_text"], validated["source_layout"], primary_text,
        current_merchant=output.get("merchant"),
    )
    if merchant is None:
        merchant = _select_header_merchant_without_branch(
            validated["merged_text"], validated["source_layout"], primary_text,
            current_merchant=output.get("merchant"), current_location=output.get("location"),
        )
    merchant_changed = merchant is not None and merchant != output.get("merchant")
    if merchant_changed:
        output["merchant"] = merchant
        accepted_fields.append("merchant")
    native_title = None
    if (not accepted_fields and isinstance(native_owner_context, dict)
            and native_owner_context.get("strategy_fingerprint") == NATIVE_TITLE_CROP_OCR_FINGERPRINT):
        native_title = _recover_native_title(
            output, validated, primary_text, native_owner_ocr_evidence, native_owner_context)
        if native_title is not None:
            output["line_items"][0]["description"] = native_title
            title_changed = True
            accepted_fields.append("line_items")
    return output, {
        "accepted_fields": accepted_fields,
        "evidence_valid": True,
        "financial_summary": financial[1] if financial is not None else None,
        "location": location_meta,
        "merchant": {"accepted": merchant_changed, "mode": "geometry_owned_header_mark"},
        "tax_categories": tax_meta,
        "line_items": {
            "accepted": table is not None or prices is not None or financial is not None or title_changed
            or any(proposal["changed"] for proposal in rate_proposals) or code_cleanup_changed or bool(pixel_description_updates),
            "mode": native_meta["mode"] if native_meta else "native_title_geometry" if native_title is not None
            else "closed_single_rate_cash_literal_table" if financial is not None
            else "source_owned_discount_rate" if rate_proposals and any(proposal["changed"] for proposal in rate_proposals)
            else "quantity_owned_title" if quantity_title is not None and title_changed
            else "counted_modifier_title" if title_changed
            else "literal_price_bundle" if prices is not None
            else "pixel_owned_marker_description" if pixel_description_updates
            else "code_prefixed_description" if code_cleanup_changed
            else "complete_literal_table",
        },
        "native_owner": native_meta,
        "native_title": {"accepted": native_title is not None, "title": native_title},
        "discount_rates": {"accepted": any(proposal["changed"] for proposal in rate_proposals), "proposals": rate_proposals},
    }
