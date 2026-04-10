"""pipeline.py — Orchestrates all pipeline stages.

Uses Google Cloud Vision OCR → text → LLM (OpenRouter or Ollama) for structured extraction.
Supports batch processing with concurrent API calls via process_batch().
"""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

from .schema import Receipt
from .preprocess import load_image, try_extract_text_layer
from .ocr import init_cloud_vision, run_cloud_vision, blocks_to_structured_text, compute_ocr_confidence, OCRResult
from .llm import check_model_available, extract_with_verification, DEFAULT_MODEL
from .validation import validate_receipt
from .normalize import (normalize_fullwidth, clean_handwritten_ocr, strip_barcode_lines,
                        rejoin_price_lines)
from .tracing import PipelineTrace, draw_ocr_bboxes, draw_field_overlay
from .patterns import (
    UTILITY_BILL_KEYWORDS, PAYMENT_SLIP_KEYWORDS, RECEIPT_KEYWORDS,
    ADMIN_SUFFIX_RE, LOCATION_CLUE_RE, ERA_TABLE,
    era_to_western_year, should_override_field,
)
from .pipeline_receipt import (
    extract_financial_totals, extract_points_used, postprocess_receipt,
)
from .pipeline_bill import postprocess_utility_bill
from .pipeline_slip import postprocess_payment_slip

_PIPELINE_VERSION = "3.0.0"


def detect_document_type(text: str) -> str:
    """Classify document type from OCR text using keyword matching."""
    utility_score = len(UTILITY_BILL_KEYWORDS.findall(text))
    slip_score = len(PAYMENT_SLIP_KEYWORDS.findall(text))
    receipt_score = len(RECEIPT_KEYWORDS.findall(text))

    if utility_score >= 2 and utility_score > receipt_score:
        return "utility_bill"
    if slip_score >= 2 and slip_score > receipt_score:
        return "payment_slip"
    return "receipt"


_USER_RULES_PATH = Path(__file__).parent / "user_rules.json"


def _apply_user_rules(result: dict) -> dict:
    """Apply user_rules.json merchant alias mapping."""
    if not _USER_RULES_PATH.exists():
        return result
    try:
        rules = json.loads(_USER_RULES_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return result

    merchant_map = rules.get("merchant_map", {})
    merchant = result.get("merchant", "") or ""

    for pattern, mapping in merchant_map.items():
        if pattern in merchant:
            if "merchant" in mapping:
                result["merchant"] = mapping["merchant"]
            if "category" in mapping:
                result["_category"] = mapping["category"]
            break

    return result


def _location_needs_resolution(location: str | None, ocr_text: str = "") -> bool:
    """Check if the location needs resolution via a focused LLM call."""
    if location and ADMIN_SUFFIX_RE.search(location):
        return False
    if location:
        return True
    if ocr_text and LOCATION_CLUE_RE.search(ocr_text):
        return True
    return False


def _location_has_ocr_evidence(location: str, ocr_text: str) -> bool:
    """Check if at least part of the location string has evidence in the OCR text.

    Normalizes whitespace before comparison since Japanese OCR frequently
    inserts spaces between characters (e.g., "宗像市 赤間" vs "宗像市赤間").
    """
    if not location or not ocr_text:
        return False
    # Normalize whitespace in both strings for comparison
    loc_norm = re.sub(r'\s+', '', location)
    ocr_norm = re.sub(r'\s+', '', ocr_text)
    if loc_norm in ocr_norm:
        return True
    # Check individual admin-level segments
    parts = re.split(r'[市区町村郡県都道府]', location)
    for part in parts:
        part = part.strip()
        if len(part) >= 2 and part in ocr_norm:
            return True
    return False


def _resolve_location(extracted: dict, ocr_text: str, model: str) -> tuple[str | None, str | None]:
    """Use a focused LLM call to resolve a partial location to city/ward level.

    Returns (resolved_location, warning_or_none). The warning is set when
    resolution was attempted but failed, so the caller can log it.
    """
    from .llm import _llm_chat, sanitize_llm_response

    merchant = extracted.get("merchant") or ""
    raw_location = extracted.get("location") or ""

    phone_match = re.search(r'(?:TEL|電話|☎)\s*[:\s]?\s*(0\d{1,4}[-\s]?\d{1,4}[-\s]?\d{2,4})', ocr_text)
    if not phone_match:
        # Bare phone number on its own line
        phone_match = re.search(r'^(0\d{1,4}-\d{1,4}-\d{2,4})\s*$', ocr_text, re.MULTILINE)
    phone_hint = phone_match.group(1) if phone_match else ""

    # Extract branch/store name (e.g., "赤間店" → "赤間", "八幡店" → "八幡")
    branch_match = re.search(r'([\u3000-\u9fff]{2,})\s*店', ocr_text)
    branch_hint = branch_match.group(1) if branch_match else ""

    # Also extract short standalone Japanese text from the first few lines
    # (often branch/location names like "赤間" above the brand name)
    header_lines = []
    for line in ocr_text.split('\n')[:6]:
        s = line.strip()
        if s and 2 <= len(s) <= 8 and re.match(r'^[\u3000-\u9fff]+$', s):
            header_lines.append(s)
    header_hint = ", ".join(header_lines) if header_lines else ""

    # Phone area code → region hint
    _AREA_CODES = {
        "0940": ("宗像市", "宗像市/福津市 area (Fukuoka pref.)"),
        "093": ("北九州市", "北九州市 area (Fukuoka pref.)"),
        "092": ("福岡市", "福岡市 area (Fukuoka pref.)"),
        "0942": ("久留米市", "久留米市 area (Fukuoka pref.)"),
        "0948": ("飯塚市", "飯塚市 area (Fukuoka pref.)"),
    }
    area_hint = ""
    area_city = ""
    if phone_hint:
        area_code = re.match(r'(0\d{1,4})', phone_hint.replace('-', '').replace(' ', ''))
        if area_code:
            code = area_code.group(1)
            for prefix, (city, desc) in _AREA_CODES.items():
                if code.startswith(prefix):
                    area_hint = f"Phone area code {prefix} = {desc}"
                    area_city = city
                    break

    addr_lines = []
    for line in ocr_text.split('\n'):
        if re.search(r'[都道府県市区町村郡]|〒\d{3}', line):
            addr_lines.append(line.strip())

    clues = [f"- Merchant/brand: {merchant}"]
    if branch_hint:
        clues.append(f"- Branch/store name: {branch_hint}店 (the branch name often indicates the neighborhood)")
    if header_hint and header_hint != branch_hint:
        clues.append(f"- Receipt header text: {header_hint} (may contain location or branch name)")
    clues.append(f"- Current location value: {raw_location or 'unknown'}")
    clues.append(f"- Phone number: {phone_hint or 'not found'}")
    if area_hint:
        clues.append(f"- {area_hint}")
    clues.append(f"- Address fragments from receipt: {'; '.join(addr_lines) if addr_lines else 'none found'}")

    # Deterministic resolution: if area code gives us a city and we have a
    # neighborhood name (from branch or header), combine them directly.
    neighborhood = ""
    if branch_hint:
        # Strip common prefixes from branch name to get neighborhood
        # e.g., "ビバモール赤間" → take last 2-3 chars as neighborhood
        for suffix_len in (3, 2):
            candidate = branch_hint[-suffix_len:]
            if re.match(r'^[\u3000-\u9fff]+$', candidate):
                neighborhood = candidate
                break
        if not neighborhood:
            neighborhood = branch_hint
    elif header_lines:
        # Use the first short header line as neighborhood
        for h in header_lines:
            if 2 <= len(h) <= 4 and re.match(r'^[\u3000-\u9fff]+$', h):
                neighborhood = h
                break

    if area_city and neighborhood:
        candidate = f"{area_city}{neighborhood}"
        if ADMIN_SUFFIX_RE.search(candidate):
            return candidate, None

    prompt = f"""Given these clues from a Japanese receipt, determine the city (市) or ward (区) where this store is located.
Output ONLY a JSON object with a single "location" field. The location should be at the 市区町村 level, e.g. "宗像市赤間", "福岡市博多区", "北九州市八幡区".
The branch name (e.g. 赤間店 → 赤間 is a neighborhood in 宗像市) is the strongest clue for location.

Clues:
{chr(10).join(clues)}

Respond with a JSON object: {{"location": "..."}} or {{"location": null}} if you cannot determine it."""

    try:
        result = _llm_chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            schema={"type": "object", "properties": {"location": {"type": ["string", "null"]}}, "required": ["location"]},
        )
        import json as _json
        data = _json.loads(sanitize_llm_response(result.content))
        resolved = data.get("location")
        if resolved and ADMIN_SUFFIX_RE.search(resolved):
            return resolved, None
    except Exception as e:
        logger.warning("Location resolution failed: %s", e)
        return None, "Location resolution failed: LLM call error"
    return None, "Location resolution: could not determine city/ward from available clues"


def _compute_posthoc_confidence(extracted: dict, warnings: list[str]) -> dict:
    """Compute per-field confidence from validation results (post-hoc).

    Maps warning text to fields using keyword matching that avoids false
    positives from field names appearing in unrelated warning context
    (e.g., "Line 1: total is X" should affect line_items, not total).
    """
    # Map warning prefixes/keywords to the field they pertain to.
    # Checked in order; first match wins per warning to avoid double-assignment.
    _WARNING_FIELD_RULES: list[tuple[str, str, bool]] = [
        # (pattern, field, prefix_only)
        ("Line ", "line_items", True),
        ("Sum of line items", "line_items", True),
        ("Items sum", "line_items", True),
        ("discount_rate", "line_items", False),
        ("Total (", "total", True),
        ("Total does not match", "total", True),
        ("subtotal (", "subtotal", False),
        ("Tax ratio", "taxes", True),
        ("tax rate", "taxes", False),
        ("Unusual tax rate", "taxes", True),
        ("amount_paid", "points_used", False),
        ("usage.", "usage", False),
        ("billing_period", "billing_period", False),
        ("merchant", "merchant", False),
    ]

    affected_fields: set[str] = set()
    for w in warnings:
        for pattern, field, prefix_only in _WARNING_FIELD_RULES:
            matched = w.startswith(pattern) if prefix_only else (pattern in w)
            if matched:
                affected_fields.add(field)
                break

    conf = {}
    for field in ("merchant", "date", "total", "subtotal", "taxes",
                   "payment_method", "line_items", "points_used"):
        val = extracted.get(field)
        if val is None or (isinstance(val, list) and len(val) == 0):
            conf[field] = 0.0
        elif field in affected_fields:
            conf[field] = 0.4
        else:
            conf[field] = 0.9

    return conf


def _build_result(receipt, final_warnings, pass_history, model, debug=False, trace=None,
                   ocr_confidence=None, llm_confidence=None,
                   ocr_source=None, ocr_retried=None, ocr_retry_reason=None,
                   ocr_text=None):
    result = receipt.model_dump()
    result["_warnings"] = final_warnings
    result["_pass_count"] = len(pass_history)
    result["_pass_history"] = pass_history
    result["_model"] = model
    result["_pipeline_version"] = _PIPELINE_VERSION
    line_item_warnings = [w for w in final_warnings if "Line " in w]
    result["_line_items_reliable"] = len(line_item_warnings) == 0
    if ocr_confidence is not None:
        result["_ocr_confidence"] = round(ocr_confidence, 4)
    if llm_confidence is not None:
        result["_llm_confidence"] = llm_confidence
    if ocr_source is not None:
        result["_ocr_source"] = ocr_source
    if ocr_retried is not None:
        result["_ocr_retried"] = ocr_retried
    if ocr_retry_reason is not None:
        result["_ocr_retry_reason"] = ocr_retry_reason
    if ocr_text is not None:
        result["_ocr_text"] = ocr_text
    if debug and trace:
        result["_debug_dir"] = str(trace.debug_dir)
        result["_trace"] = trace.summary()
    return result


def process_document(
    file_path: Path,
    model: str = DEFAULT_MODEL,
    debug: bool = False,
    passes: int = 1,
    ocr_engine=None,
    apply_user_rules: bool = True,
    skip_ocr_cache: bool = False,
    **kwargs,
) -> dict:
    """Main pipeline. Uses Cloud Vision OCR + LLM extraction (OpenRouter or Ollama)."""
    file_path = Path(file_path)
    check_model_available(model)

    # Track document processing
    from .usage import track_document
    track_document(file_path)

    trace = PipelineTrace()
    debug_dir: Path | None = None
    if debug:
        debug_dir = Path("debug") / file_path.stem
        debug_dir.mkdir(parents=True, exist_ok=True)
        trace.debug_dir = debug_dir

    # Step 1: Load
    images = load_image(file_path)
    trace.log_step("original", image=images[0])

    # Digital PDF fast path
    if file_path.suffix.lower() == ".pdf":
        digital_text = try_extract_text_layer(str(file_path))
        if digital_text:
            digital_text = normalize_fullwidth(digital_text)
            trace.log_step("digital_text_extracted", data=digital_text)
            doc_type = detect_document_type(digital_text)

            if debug:
                assert debug_dir is not None
                (debug_dir / "03_ocr_bboxes.txt").write_text(
                    "SKIPPED: Digital PDF fast path — no OCR performed.")

            extracted, pass_history = extract_with_verification(
                digital_text, model=model, passes=passes,
                validate_fn=validate_receipt, doc_type=doc_type,
            )

            if debug:
                for entry in pass_history:
                    n = entry["pass"]
                    trace.log_step(f"pass{n}_llm_response", data=entry["extraction"])
                    if entry["warnings"]:
                        trace.log_step(f"pass{n}_warnings", data="\n".join(entry["warnings"]))

            llm_conf_pdf = extracted.pop("_confidence", None)

            # Location resolution for PDF path
            if "error" not in extracted and _location_needs_resolution(extracted.get("location"), digital_text):
                resolved, _loc_warn = _resolve_location(extracted, digital_text, model)
                if resolved:
                    extracted["location"] = resolved

            try:
                receipt = Receipt(**extracted)
            except Exception:
                receipt = Receipt()
            final_warnings = validate_receipt(receipt)
            for w in receipt._soft_warnings:
                if w not in final_warnings:
                    final_warnings.append(w)

            if debug:
                assert debug_dir is not None
                (debug_dir / "10_field_overlay.txt").write_text(
                    "SKIPPED: Digital PDF fast path — no OCR bounding boxes available.")
                (debug_dir / "pipeline_trace.txt").write_text(trace.summary())

            result = _build_result(receipt, final_warnings, pass_history, model, debug=debug, trace=trace,
                                   ocr_confidence=1.0, llm_confidence=llm_conf_pdf)
            if apply_user_rules:
                result = _apply_user_rules(result)
            return result

    # Step 2: Init OCR engine
    if ocr_engine is None:
        ocr_engine = init_cloud_vision()

    # Step 3: OCR per page, concatenate
    all_ocr_results: list[OCRResult] = []
    text_parts = []

    for i, page_img in enumerate(images):
        ocr_result = run_cloud_vision(page_img, ocr_engine, skip_cache=skip_ocr_cache)
        all_ocr_results.append(ocr_result)
        blocks = ocr_result.blocks

        if len(blocks) < 3:
            # Try all rotations (90°, 180°, 270°), pick best by confidence
            best_result = ocr_result
            best_conf = compute_ocr_confidence(blocks) if blocks else 0.0
            for rotation in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
                rotated = cv2.rotate(page_img, rotation)
                rot_result = run_cloud_vision(rotated, ocr_engine, skip_cache=skip_ocr_cache)
                rot_conf = compute_ocr_confidence(rot_result.blocks) if rot_result.blocks else 0.0
                if len(rot_result.blocks) > len(best_result.blocks) or rot_conf > best_conf:
                    best_result = rot_result
                    best_conf = rot_conf
                if best_conf >= 0.85:
                    break  # Good enough, stop early
            if len(best_result.blocks) > len(blocks):
                ocr_result = best_result
                blocks = ocr_result.blocks
                all_ocr_results[-1] = ocr_result

        if debug:
            assert debug_dir is not None
            draw_ocr_bboxes(page_img, blocks, debug_dir / f"03_page{i+1}_ocr_bboxes.png")

        page_text = blocks_to_structured_text(blocks)
        if i > 0:
            text_parts.append(f"--- PAGE {i+1} ---")
        text_parts.append(page_text)

    unified_text = "\n".join(text_parts)
    unified_text = normalize_fullwidth(unified_text)
    raw_text = unified_text  # Preserve pre-barcode-stripped text
    unified_text = strip_barcode_lines(unified_text)

    # Compute aggregate OCR confidence
    all_blocks_flat = [b for r in all_ocr_results for b in r.blocks]
    ocr_conf = compute_ocr_confidence(all_blocks_flat)

    # Detect document type
    doc_type = detect_document_type(unified_text)

    if not unified_text.strip():
        return {
            "_error": "OCR produced no text.",
            "_warnings": [], "_pass_count": 0, "_model": model,
            "_pipeline_version": _PIPELINE_VERSION, "_line_items_reliable": False,
        }

    trace.log_step("ocr_grouped", data=unified_text)

    # Step 4–5: LLM extraction → post-processing → validation (shared path)
    extracted, pass_history, final_warnings = _run_extraction_pipeline(
        unified_text=unified_text, raw_text=raw_text,
        ocr_conf=ocr_conf, doc_type=doc_type,
        model=model, passes=passes,
    )

    if "_error" in extracted:
        extracted.update({"_warnings": [], "_pass_count": 0, "_model": model,
                          "_pipeline_version": _PIPELINE_VERSION, "_line_items_reliable": False})
        return extracted

    if debug:
        for entry in pass_history:
            n = entry["pass"]
            trace.log_step(f"pass{n}_llm_response", data=entry["extraction"])
            if entry["warnings"]:
                trace.log_step(f"pass{n}_warnings", data="\n".join(entry["warnings"]))

    # Compute post-hoc confidence from validation results
    posthoc_conf = _compute_posthoc_confidence(extracted, final_warnings)

    if debug and images:
        assert debug_dir is not None
        draw_field_overlay(images[0], all_ocr_results[0].blocks, extracted, debug_dir / "10_field_overlay.png")
        (debug_dir / "pipeline_trace.txt").write_text(trace.summary())

    # Aggregate OCR metadata from first page result
    try:
        receipt = Receipt(**extracted)
    except Exception:
        receipt = Receipt()
    primary_ocr = all_ocr_results[0] if all_ocr_results else None
    result = _build_result(
        receipt, final_warnings, pass_history, model, debug=debug, trace=trace,
        ocr_confidence=ocr_conf, llm_confidence=posthoc_conf,
        ocr_source=primary_ocr.source if primary_ocr else None,
        ocr_retried=primary_ocr.retried if primary_ocr else None,
        ocr_retry_reason=primary_ocr.retry_reason if primary_ocr else None,
        ocr_text=primary_ocr.chosen_text if primary_ocr else None,
    )
    if apply_user_rules:
        result = _apply_user_rules(result)
    return result


def _run_extraction_pipeline(
    unified_text: str,
    raw_text: str,
    ocr_conf: float,
    doc_type: str,
    model: str,
    passes: int,
) -> tuple[dict, list[dict], list[str]]:
    """Shared extraction logic: LLM extraction → post-processing → location → validation.

    Used by both process_document() and process_ocr_text() to avoid code
    duplication and ensure fixes are applied consistently.

    Returns (extracted_dict, pass_history, final_warnings).
    """
    # Receipt-specific pre-processing
    ocr_totals = {}
    if doc_type == "receipt":
        ocr_totals = extract_financial_totals(unified_text)
        unified_text = rejoin_price_lines(unified_text)
        unified_text = clean_handwritten_ocr(unified_text, ocr_confidence=ocr_conf)

    if not unified_text.strip():
        return (
            {"_error": "OCR text is empty."},
            [],
            [],
        )

    # LLM extraction with verification
    extracted, pass_history = extract_with_verification(
        unified_text, model=model, passes=passes,
        validate_fn=validate_receipt, doc_type=doc_type,
    )

    if "error" not in extracted:
        extracted["document_type"] = doc_type
        for fkey in ("total", "subtotal"):
            v = extracted.get(fkey)
            if v is not None:
                try:
                    extracted[fkey] = float(v)
                except (TypeError, ValueError):
                    extracted[fkey] = None

    # Document-type-specific post-processing
    llm_conf = extracted.get("_confidence")
    if doc_type == "receipt" and "error" not in extracted:
        extracted = postprocess_receipt(extracted, unified_text, ocr_conf, ocr_totals, llm_conf, model)
    elif doc_type == "utility_bill" and "error" not in extracted:
        extracted = postprocess_utility_bill(extracted, unified_text)
    elif doc_type == "payment_slip" and "error" not in extracted:
        extracted = postprocess_payment_slip(extracted, unified_text, raw_text=raw_text)

    # Universal cash detection (all document types)
    if "error" not in extracted and not extracted.get("payment_method"):
        if re.search(r'領収証|領収書', unified_text) and not re.search(r'小計|合計|対象|税率', unified_text):
            extracted["payment_method"] = "cash"

    # Final cash fallback
    _ELECTRONIC_PAY_RE = re.compile(
        r'クレジット|カード|PayPay|電子マネー|iD|QUICPay|Suica|WAON|nanaco|'
        r'PASMO|楽天Edy|LINE\s*Pay|au\s*PAY|d払い|メルペイ|交通系'
    )
    if "error" not in extracted and not extracted.get("payment_method"):
        has_tender_label = bool(re.search(r'お預り', unified_text))
        has_change_label_final = bool(re.search(r'釣', unified_text))
        has_electronic = bool(_ELECTRONIC_PAY_RE.search(unified_text))
        if has_tender_label and has_change_label_final and not has_electronic:
            extracted["payment_method"] = "cash"

    # Location: clear for utility bills and payment slips
    if "error" not in extracted and doc_type in ("utility_bill", "payment_slip"):
        extracted["location"] = None

    # Common post-processing
    if "error" not in extracted:
        total = extracted.get("total")
        points = extracted.get("points_used")
        if total is not None:
            extracted["amount_paid"] = total - points if points else total

    # Strip _confidence if present
    extracted.pop("_confidence", None)

    # Location resolution (confidence-gated, receipts only)
    location_warnings: list[str] = []
    if "error" not in extracted and doc_type == "receipt" and _location_needs_resolution(extracted.get("location"), unified_text):
        # Check OCR evidence first — skip expensive LLM call if no evidence
        has_evidence = _location_has_ocr_evidence(
            extracted.get("location", ""), unified_text
        )
        has_clues = bool(LOCATION_CLUE_RE.search(unified_text))
        if has_evidence or has_clues:
            resolved, loc_warning = _resolve_location(extracted, unified_text, model)
            if resolved:
                extracted["location"] = resolved
            elif loc_warning:
                location_warnings.append(loc_warning)

    # Location validation: clear if no OCR evidence supports it
    if "error" not in extracted and doc_type == "receipt" and extracted.get("location"):
        if not _location_has_ocr_evidence(extracted["location"], unified_text):
            extracted["location"] = None

    # Final validation
    try:
        receipt = Receipt(**extracted)
    except Exception:
        receipt = Receipt()
    final_warnings = validate_receipt(receipt)
    for w in receipt._soft_warnings:
        if w not in final_warnings:
            final_warnings.append(w)
    final_warnings.extend(location_warnings)

    return extracted, pass_history, final_warnings


def process_ocr_text(
    ocr_text: str,
    model: str = DEFAULT_MODEL,
    passes: int = 1,
    apply_user_rules: bool = True,
) -> dict:
    """Run the pipeline from OCR text onwards (skip image loading + OCR).

    Used for:
    - Testing against saved OCR variants (regression tests)
    - Debugging with specific OCR output
    - Benchmarking LLM extraction independently of OCR variance
    """
    check_model_available(model)

    # Normalize text
    unified_text = normalize_fullwidth(ocr_text)
    unified_text = strip_barcode_lines(unified_text)
    doc_type = detect_document_type(unified_text)
    ocr_conf = 0.9  # default confidence for injected text

    if not unified_text.strip():
        return {
            "_error": "OCR text is empty.",
            "_warnings": [], "_pass_count": 0, "_model": model,
            "_pipeline_version": _PIPELINE_VERSION, "_line_items_reliable": False,
        }

    extracted, pass_history, final_warnings = _run_extraction_pipeline(
        unified_text=unified_text, raw_text=ocr_text,
        ocr_conf=ocr_conf, doc_type=doc_type,
        model=model, passes=passes,
    )

    if "_error" in extracted:
        extracted.update({"_warnings": [], "_pass_count": 0, "_model": model,
                          "_pipeline_version": _PIPELINE_VERSION, "_line_items_reliable": False})
        return extracted

    result = _build_result(
        Receipt(**extracted) if "error" not in extracted else Receipt(),
        final_warnings, pass_history, model,
        ocr_confidence=ocr_conf, ocr_source="injected",
        ocr_text=ocr_text,
    )
    if apply_user_rules:
        result = _apply_user_rules(result)
    return result


def process_batch(
    file_paths: list[Path],
    model: str = DEFAULT_MODEL,
    debug: bool = False,
    passes: int = 1,
    ocr_engine=None,
    apply_user_rules: bool = True,
    max_workers: int = 4,
    on_progress=None,
) -> list[dict]:
    """Process multiple documents concurrently.

    Uses ThreadPoolExecutor for parallel LLM API calls (I/O-bound).
    The Cloud Vision client and OCR cache are thread-safe.
    """
    if not file_paths:
        return []

    check_model_available(model)
    if ocr_engine is None:
        ocr_engine = init_cloud_vision()

    total = len(file_paths)
    results: list[dict | None] = [None] * total
    start_time = time.perf_counter()

    def _process_one(idx: int, file_path: Path) -> tuple[int, dict]:
        try:
            result = process_document(
                file_path, model=model, debug=debug,
                passes=passes, ocr_engine=ocr_engine,
                apply_user_rules=apply_user_rules,
            )
            result["_file"] = str(file_path)
        except Exception as e:
            result = {"_file": str(file_path), "_error": str(e)}
        return idx, result

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_one, i, fp): i
            for i, fp in enumerate(file_paths)
        }
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
            completed += 1
            if on_progress:
                on_progress(file_paths[idx], result, completed, total)

    elapsed = time.perf_counter() - start_time
    for r in results:
        if r:
            r["_batch_total_s"] = round(elapsed, 2)
            r["_batch_workers"] = max_workers

    return results  # type: ignore[return-value]
