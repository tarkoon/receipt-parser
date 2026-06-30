"""pipeline.py — Orchestrates all pipeline stages.

Uses Google Cloud Vision OCR → text → LLM (OpenRouter or Ollama) for structured extraction.
Supports batch processing with concurrent API calls via process_batch().
"""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import cv2



logger = logging.getLogger(__name__)

from .schema import Receipt
from .preprocess import load_image, try_extract_text_layer
from .ocr import init_cloud_vision, run_cloud_vision, blocks_to_structured_text, compute_ocr_confidence, OCRResult
from .llm import check_model_available, extract_with_verification, DEFAULT_MODEL
from .validation import validate_receipt
from .normalize import (normalize_fullwidth, clean_handwritten_ocr, strip_barcode_lines,
                        rejoin_price_lines, _shift_misaligned_inline_prices, strip_bonus_point_lines,
                        join_split_qty_details,
                        rejoin_totals_label_value_columns)
from .tracing import PipelineTrace, draw_ocr_bboxes, draw_field_overlay
from .patterns import (
    UTILITY_BILL_KEYWORDS, PAYMENT_SLIP_KEYWORDS, RECEIPT_KEYWORDS,
    LOCATION_CLUE_RE,
)
from .receipt_financial import extract_financial_totals, extract_points_used
from .receipt_postprocess import postprocess_receipt
from .pipeline_bill import postprocess_utility_bill
from .pipeline_slip import postprocess_payment_slip
from .receipt_location import (
    _location_has_ocr_evidence,
    _location_needs_resolution,
    _recover_header_branch_store_location,  # noqa: F401 - legacy private import surface
    _resolve_location,
    _trim_purchase_store_metadata_location,
    _trim_store_in_store_header_location,
)
from .receipt_output import (
    _apply_final_receipt_output_repairs,  # noqa: F401 - legacy private import surface
    _prepare_receipt_output_payload,
)

import inspect
from typing import Any, Callable, Literal

# Public progress contract — these stage names are stable across pipeline versions.
# Consumers may switch on these strings safely.
StageName = Literal[
    "load",
    "ocr",
    "classify",
    "normalize",
    "extract",
    "postprocess",
    "resolve_location",
    "validate",
    "warn",
    "plan",
    "done",
]

# Callback may be 3-arg (stage, detail, progress) — original contract — or
# 4-arg by naming the extra parameter `payload` (or accepting **kwargs).
# Returning False from any call requests cooperative cancellation.
StageCallback = Callable[..., Any] | None


class PipelineCancelled(Exception):
    """Raised when an on_stage callback returns False to abort the pipeline."""
    def __init__(self, stage: str):
        super().__init__(f"Pipeline cancelled at stage: {stage}")
        self.stage = stage


def _callback_accepts_payload(cb) -> bool:
    """Decide whether to pass the structured payload as a keyword argument.

    A callback opts in to the 4-arg contract by naming a parameter `payload`
    or accepting **kwargs. Counting positional slots is unreliable: callers
    sometimes use a 4th defaulted positional (e.g. ``def cb(s, d, p, _task=t)``)
    as a closure-capture idiom, and we must NOT pass our payload there.
    """
    try:
        sig = inspect.signature(cb)
    except (ValueError, TypeError):
        return False
    params = sig.parameters
    if "payload" in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def _notify(
    on_stage: StageCallback,
    stage: str,
    detail: str,
    progress: float,
    payload: dict | None = None,
) -> None:
    """Fire a progress callback if one is registered.

    Backwards compatible with 3-arg callbacks. The structured payload is only
    passed when the callback's signature opts in (param named `payload` or
    accepts **kwargs). If the callback returns False, raises PipelineCancelled.
    """
    if on_stage is None:
        return
    if _callback_accepts_payload(on_stage):
        result = on_stage(stage, detail, progress, payload=payload)
    else:
        result = on_stage(stage, detail, progress)
    if result is False:
        raise PipelineCancelled(stage)


_PIPELINE_VERSION = "3.1.0"


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


def _build_validate_detail(extracted: dict) -> str:
    """Format a short receipt preview string for the validate-stage callback."""
    if "error" in extracted or "_error" in extracted:
        return "Validating"

    parts: list[str] = []
    doc_type = extracted.get("document_type") or "receipt"

    if doc_type == "receipt":
        n = len(extracted.get("line_items") or [])
        if n:
            parts.append(f"{n} item{'s' if n != 1 else ''}")
    elif doc_type == "utility_bill":
        st = extracted.get("service_type")
        if st:
            parts.append(str(st))
    elif doc_type == "payment_slip":
        parts.append("payment slip")

    total = extracted.get("total")
    if total is not None:
        currency = extracted.get("currency") or ""
        symbol = "¥" if currency in ("JPY", "") else ""
        try:
            parts.append(f"{symbol}{int(round(float(total))):,}")
        except (TypeError, ValueError):
            pass

    return " · ".join(parts) if parts else "Validating"


def _build_plan_payload(
    page_count: int,
    pass_budget: int,
    doc_type: str | None = None,
) -> dict:
    """Build the structured payload for a 'plan' stage event."""
    if doc_type is None:
        path = "tbd"
        will_resolve_location = True  # unknown until classify; assume yes
    else:
        path = doc_type
        will_resolve_location = (doc_type == "receipt")
    return {
        "path": path,
        "page_count": page_count,
        "pass_budget": pass_budget,
        "will_resolve_location": will_resolve_location,
    }


def _expected_stages(doc_type: str, source: str) -> list[str]:
    """The remaining stage keys this document will fire after classify, in order.

    Lets consumers finalize the step list as soon as classify lands, instead of
    inferring the path from doc_type + source themselves.
    """
    if source == "digital_pdf":
        # Fast path: no OCR-grouping, normalize, postprocess, or resolve_location
        # stage events. (Location resolution may still run internally for
        # receipts, but no event is fired for it.)
        return ["extract", "validate", "done"]
    stages = ["extract", "postprocess"]
    if doc_type == "receipt":
        stages.append("resolve_location")
    stages.extend(["validate", "done"])
    return stages


def _build_classify_payload(doc_type: str, source: str) -> dict:
    """Structured payload for the 'classify' event — enough info to finalize the
    UI step list without needing to memorize the per-doc-type path table."""
    if source == "digital_pdf":
        will_resolve = False  # not emitted as a stage even when it runs
    else:
        will_resolve = (doc_type == "receipt")
    return {
        "document_type": doc_type,
        "source": source,
        "will_resolve_location": will_resolve,
        "expected_stages": _expected_stages(doc_type, source),
    }


def _build_result(receipt_payload, final_warnings, pass_history, model, debug=False, trace=None,
                  ocr_confidence=None, llm_confidence=None,
                  ocr_source=None, ocr_retried=None, ocr_retry_reason=None,
                  ocr_text=None, mutation_trace: list[dict] | None = None):
    result = deepcopy(receipt_payload)
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
    if debug and mutation_trace:
        result["_receipt_mutation_trace"] = mutation_trace
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
    on_stage: StageCallback = kwargs.get("on_stage")

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
    _notify(on_stage, "load", "Loading image", 0.0)
    images = load_image(file_path)
    trace.log_step("original", image=images[0])

    # Plan event: declare known shape early so consumers can draw the step list.
    # Doc-type is unknown until classify; path is "tbd" here.
    _notify(
        on_stage, "plan", "Pipeline plan",
        0.02,
        payload=_build_plan_payload(page_count=len(images), pass_budget=passes),
    )

    # Digital PDF fast path
    if file_path.suffix.lower() == ".pdf":
        _notify(on_stage, "ocr", "Checking for digital text", 0.05)
        digital_text = try_extract_text_layer(str(file_path))
        if digital_text:
            digital_text = normalize_fullwidth(digital_text)
            trace.log_step("digital_text_extracted", data=digital_text)
            doc_type = detect_document_type(digital_text)
            _notify(
                on_stage, "classify", f"Detected: {doc_type}",
                0.35,
                payload=_build_classify_payload(doc_type, source="digital_pdf"),
            )

            if debug:
                assert debug_dir is not None
                (debug_dir / "03_ocr_bboxes.txt").write_text(
                    "SKIPPED: Digital PDF fast path — no OCR performed.")

            _notify(on_stage, "extract", "LLM extraction (digital PDF)", 0.40)
            extracted, pass_history = extract_with_verification(
                digital_text, model=model, passes=passes,
                validate_fn=validate_receipt, doc_type=doc_type,
                on_stage=on_stage,
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
            _trim_purchase_store_metadata_location(extracted, digital_text)

            try:
                receipt = Receipt(**extracted)
            except Exception:
                receipt = Receipt()
            _notify(on_stage, "validate", _build_validate_detail(extracted), 0.95)
            final_warnings = validate_receipt(receipt)
            for w in receipt._soft_warnings:
                if w not in final_warnings:
                    final_warnings.append(w)

            if debug:
                assert debug_dir is not None
                (debug_dir / "10_field_overlay.txt").write_text(
                    "SKIPPED: Digital PDF fast path — no OCR bounding boxes available.")
                (debug_dir / "pipeline_trace.txt").write_text(trace.summary())

            receipt_payload = _prepare_receipt_output_payload(receipt)
            result = _build_result(receipt_payload, final_warnings, pass_history, model, debug=debug, trace=trace,
                                   ocr_confidence=1.0, llm_confidence=llm_conf_pdf)
            if apply_user_rules:
                result = _apply_user_rules(result)
            _notify(on_stage, "done", "Complete", 1.0)
            return result

    # Step 2: Init OCR engine
    if ocr_engine is None:
        ocr_engine = init_cloud_vision()

    # Step 3: OCR per page, concatenate
    _notify(on_stage, "ocr", "Running OCR", 0.05)
    all_ocr_results: list[OCRResult] = []
    text_parts = []

    n_pages = max(1, len(images))
    _OCR_BAND_START, _OCR_BAND_END = 0.05, 0.30
    _OCR_BAND = _OCR_BAND_END - _OCR_BAND_START

    rotations = (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE)

    for i, page_img in enumerate(images):
        page_start = _OCR_BAND_START + (i / n_pages) * _OCR_BAND
        page_end = _OCR_BAND_START + ((i + 1) / n_pages) * _OCR_BAND
        if n_pages > 1:
            _notify(on_stage, "ocr", f"OCR page {i+1} of {n_pages}", page_start)
        else:
            _notify(on_stage, "ocr", "Running OCR", page_start)

        ocr_result = run_cloud_vision(page_img, ocr_engine, skip_cache=skip_ocr_cache)
        all_ocr_results.append(ocr_result)
        blocks = ocr_result.blocks

        if len(blocks) < 3:
            # Try all rotations (90°, 180°, 270°), pick best by confidence.
            _notify(
                on_stage, "warn",
                f"OCR returned {len(blocks)} blocks — retrying with rotations",
                page_start,
                payload={
                    "reason": "low_block_count",
                    "page": i + 1,
                    "block_count": len(blocks),
                },
            )
            best_result = ocr_result
            best_conf = compute_ocr_confidence(blocks) if blocks else 0.0
            for r_idx, rotation in enumerate(rotations):
                # Distribute sub-progress within this page's slot, leaving room
                # for the post-rotation work.
                sub_progress = page_start + ((r_idx + 1) / (len(rotations) + 1)) * (page_end - page_start)
                _notify(
                    on_stage, "ocr",
                    f"Page {i+1} retry rotation {r_idx+1}/{len(rotations)}",
                    sub_progress,
                )
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
    _notify(on_stage, "normalize", "Processing OCR text", 0.30)
    all_blocks_flat = [b for r in all_ocr_results for b in r.blocks]
    ocr_conf = compute_ocr_confidence(all_blocks_flat)

    # Detect document type
    doc_type = detect_document_type(unified_text)
    _notify(
        on_stage, "classify", f"Detected: {doc_type}",
        0.35,
        payload=_build_classify_payload(doc_type, source="ocr"),
    )

    if not unified_text.strip():
        return {
            "_error": "OCR produced no text.",
            "_warnings": [], "_pass_count": 0, "_model": model,
            "_pipeline_version": _PIPELINE_VERSION, "_line_items_reliable": False,
        }

    trace.log_step("ocr_grouped", data=unified_text)

    # Step 4–5: LLM extraction → post-processing → validation (shared path)
    _notify(on_stage, "extract", "LLM extraction", 0.40)
    all_layout_blocks = []
    for page_idx, ocr_result in enumerate(all_ocr_results):
        for block in getattr(ocr_result, "layout_blocks", []):
            block_with_page = dict(block)
            block_with_page["page"] = page_idx
            all_layout_blocks.append(block_with_page)
    receipt_mutation_trace: list[dict] | None = [] if debug else None
    extracted, pass_history, final_warnings = _run_extraction_pipeline(
        unified_text=unified_text, raw_text=raw_text,
        ocr_conf=ocr_conf, doc_type=doc_type,
        model=model, passes=passes,
        ocr_layout_blocks=all_layout_blocks,
        on_stage=on_stage,
        mutation_trace=receipt_mutation_trace,
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
    repair_ocr_text = primary_ocr.chosen_text if primary_ocr else None
    receipt_payload = _prepare_receipt_output_payload(
        receipt,
        repair_ocr_text,
        mutation_trace=receipt_mutation_trace,
    )
    result = _build_result(
        receipt_payload, final_warnings, pass_history, model, debug=debug, trace=trace,
        ocr_confidence=ocr_conf, llm_confidence=posthoc_conf,
        ocr_source=primary_ocr.source if primary_ocr else None,
        ocr_retried=primary_ocr.retried if primary_ocr else None,
        ocr_retry_reason=primary_ocr.retry_reason if primary_ocr else None,
        ocr_text=repair_ocr_text,
        mutation_trace=receipt_mutation_trace,
    )
    if apply_user_rules:
        result = _apply_user_rules(result)
    _notify(on_stage, "done", "Complete", 1.0)
    return result


def _receipt_items_target_gap(extracted: dict) -> float | None:
    items = extracted.get("line_items") or []
    if not items:
        return None
    items_sum = sum(
        item.get("total", 0)
        for item in items
        if isinstance(item, dict)
    )
    targets: list[float] = []
    subtotal = extracted.get("subtotal")
    total = extracted.get("total")
    taxes = extracted.get("taxes") or []
    tax_sum = sum(
        tax.get("amount", 0)
        for tax in taxes
        if isinstance(tax, dict) and tax.get("amount") is not None
    )
    canonical_subtotal = None
    if total and tax_sum:
        canonical_subtotal = float(total) - float(tax_sum)
    if subtotal is not None:
        if canonical_subtotal is None or abs(float(subtotal) - canonical_subtotal) <= 2:
            targets.append(float(subtotal))
    if total is not None:
        targets.append(float(total))
    if canonical_subtotal is not None:
        targets.append(canonical_subtotal)
    if not targets:
        return None
    return min(abs(items_sum - target) for target in targets)


def _receipt_printed_tax_gap(extracted: dict, unified_text: str) -> float:
    text = re.sub(r'\s+', ' ', unified_text or "")
    blocks: dict[str, float] = {}
    for m in re.finditer(
        r'\(\s*(\d{2})%対象\s*¥?\s*[\d,]+\s*内税\s*¥?\s*([\d,]+)\s*\)',
        text,
    ):
        blocks[f"{int(m.group(1))}%"] = float(m.group(2).replace(',', ''))
    for m in re.finditer(
        r'\(\s*(\d{2})%対象\s*¥?\s*([\d,]+)\s*\)\s*¥?\s*([\d,]+)\s*内税',
        text,
    ):
        rate = f"{int(m.group(1))}%"
        if rate in blocks:
            continue
        amount = float(m.group(2).replace(',', ''))
        base = float(m.group(3).replace(',', ''))
        try:
            rate_pct = float(rate.rstrip('%')) / 100.0
        except ValueError:
            continue
        expected = round(base * rate_pct / (1 + rate_pct))
        if 0 < amount < base and abs(amount - expected) <= 2:
            blocks[rate] = amount
    if not blocks:
        return 0.0
    taxes_by_rate = {
        tax.get("rate"): float(tax.get("amount") or 0)
        for tax in (extracted.get("taxes") or [])
        if isinstance(tax, dict)
    }
    gap = 0.0
    for rate, amount in blocks.items():
        if rate not in taxes_by_rate:
            gap += 1_000_000.0
        else:
            gap += abs(taxes_by_rate[rate] - amount)
    return gap


def _receipt_candidate_score(extracted: dict, warnings: list[str], unified_text: str = "") -> tuple:
    gap = _receipt_items_target_gap(extracted)
    gap_value = 1_000_000.0 if gap is None else float(gap)
    tax_gap = _receipt_printed_tax_gap(extracted, unified_text)
    item_warning_count = sum(
        1 for warning in warnings
        if "line items" in warning or "Items sum" in warning
    )
    item_count = len([
        item for item in (extracted.get("line_items") or [])
        if isinstance(item, dict)
    ])
    return (
        gap_value > 2,
        gap_value,
        tax_gap > 2,
        tax_gap,
        item_warning_count,
        len(warnings),
        -item_count,
    )


def _select_receipt_postprocessed_candidate(
    extracted: dict,
    pass_history: list[dict],
    unified_text: str,
    ocr_conf: float,
    ocr_totals: dict,
    model: str,
    ocr_layout_blocks: list[dict] | None,
    mutation_trace: list[dict] | None = None,
) -> dict:
    """Post-process all captured receipt candidates and keep the cleanest one.

    Raw LLM retry scoring can miss candidates that deterministic post-processing
    repairs cleanly. This selector scores the post-processed reality while still
    preserving each history entry's raw extraction for diagnostics.
    """
    candidate_refs: list[tuple[int | None, dict]] = [(None, extracted)]
    for idx, entry in enumerate(pass_history):
        candidate = entry.get("extraction")
        if not candidate or not isinstance(candidate, dict) or "error" in candidate:
            continue
        candidate_refs.append((idx, candidate))

    best: tuple[tuple, int, dict, list[str], int | None, list[dict] | None] | None = None
    for order, (history_idx, candidate) in enumerate(candidate_refs):
        postprocessed = deepcopy(candidate)
        llm_conf = postprocessed.get("_confidence")
        candidate_trace: list[dict] | None = [] if mutation_trace is not None else None
        postprocessed = postprocess_receipt(
            postprocessed,
            unified_text,
            ocr_conf,
            deepcopy(ocr_totals),
            llm_conf,
            model,
            ocr_layout_blocks=ocr_layout_blocks,
            mutation_trace=candidate_trace,
        )
        try:
            receipt = Receipt(**postprocessed)
            warnings = validate_receipt(receipt)
            for warning in receipt._soft_warnings:
                if warning not in warnings:
                    warnings.append(warning)
        except Exception as exc:
            warnings = [f"Schema validation failed after postprocess: {exc}"]
        score = _receipt_candidate_score(postprocessed, warnings, unified_text)

        if history_idx is not None:
            entry = pass_history[history_idx]
            entry["postprocess_items_sum_gap"] = _receipt_items_target_gap(postprocessed)
            entry["postprocess_warning_count"] = len(warnings)
            entry["postprocess_warnings"] = warnings
            entry["postprocess_selected"] = False

        ranked = (score, order, postprocessed, warnings, history_idx, candidate_trace)
        if best is None or ranked[:2] < best[:2]:
            best = ranked

    if best is None:
        return extracted

    _score, _order, best_extracted, _warnings, best_history_idx, best_trace = best
    if best_history_idx is not None:
        pass_history[best_history_idx]["postprocess_selected"] = True
    if mutation_trace is not None and best_trace:
        mutation_trace.extend(best_trace)
    return best_extracted


def _run_extraction_pipeline(
    unified_text: str,
    raw_text: str,
    ocr_conf: float,
    doc_type: str,
    model: str,
    passes: int,
    ocr_layout_blocks: list[dict] | None = None,
    on_stage: StageCallback = None,
    mutation_trace: list[dict] | None = None,
) -> tuple[dict, list[dict], list[str]]:
    """Shared extraction logic: LLM extraction → post-processing → location → validation.

    Used by both process_document() and process_ocr_text() to avoid code
    duplication and ensure fixes are applied consistently.

    Returns (extracted_dict, pass_history, final_warnings).
    """
    # Receipt-specific pre-processing
    ocr_totals = {}
    if doc_type == "receipt":
        # Interleave label-value column splits in the totals zone BEFORE
        # extracting financial totals — otherwise the line-by-line walk
        # picks up sibling-label values (お預り) instead of 合計's own value.
        unified_text = rejoin_totals_label_value_columns(unified_text)
        ocr_totals = extract_financial_totals(unified_text)
        # Strip bonus-point lines BEFORE rejoin so item↔price column matching
        # isn't disrupted by stray loyalty-point fragments.
        unified_text = strip_bonus_point_lines(unified_text)
        unified_text = join_split_qty_details(unified_text)
        # strip_banner_lines disabled — even with empty placeholders, line
        # changes affect LLM extraction (position-sensitive). Banner-line
        # phantoms are filtered at item level via _drop_banner_phantom_items.
        unified_text = rejoin_price_lines(unified_text)
        unified_text = _shift_misaligned_inline_prices(unified_text)
        unified_text = clean_handwritten_ocr(unified_text, ocr_confidence=ocr_conf)

    if not unified_text.strip():
        return (
            {"_error": "OCR text is empty."},
            [],
            [],
        )

    # LLM extraction with verification — emits per-pass beats internally.
    extracted, pass_history = extract_with_verification(
        unified_text, model=model, passes=passes,
        validate_fn=validate_receipt, doc_type=doc_type,
        on_stage=on_stage,
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
    _notify(on_stage, "postprocess", "Post-processing", 0.70)
    if doc_type == "receipt" and "error" not in extracted:
        extracted = _select_receipt_postprocessed_candidate(
            extracted,
            pass_history,
            unified_text,
            ocr_conf,
            ocr_totals,
            model,
            ocr_layout_blocks,
            mutation_trace=mutation_trace,
        )
    elif doc_type == "utility_bill" and "error" not in extracted:
        extracted = postprocess_utility_bill(extracted, unified_text)
    elif doc_type == "payment_slip" and "error" not in extracted:
        extracted = postprocess_payment_slip(extracted, unified_text, raw_text=raw_text)

    # Universal cash detection (all document types). For handwritten 領収証
    # forms, accept either explicit tender markers (お預り, 現金, 現計, お釣り)
    # OR the formal cash-receipt acknowledgement (上記正に領収/受領いたしました)
    # as long as nothing electronic or transfer-related contradicts it.
    # The acknowledgement phrase on a small handwritten receipt with no other
    # tender info is the standard Japanese signal that cash was tendered.
    _ELECTRONIC_PAY_RE = re.compile(
        r'クレジット|カード|PayPay|電子マネー|iD|QUICPay|Suica|WAON|nanaco|'
        r'PASMO|楽天Edy|LINE\s*Pay|au\s*PAY|d払い|メルペイ|交通系'
    )
    if "error" not in extracted and not extracted.get("payment_method"):
        is_handwritten = (
            re.search(r'領収証|領収書', unified_text)
            and not re.search(r'小計|合計|対象|税率', unified_text)
        )
        if is_handwritten:
            has_tender = bool(re.search(
                r'(?:お預り金?|お預かり)(?!票)|現金|現計|お釣り|釣銭',
                unified_text,
            ))
            has_acknowledgement = bool(
                re.search(r'上記正に\s*(?:領収|受領)', unified_text)
            )
            has_electronic = bool(_ELECTRONIC_PAY_RE.search(unified_text))
            has_transfer = bool(re.search(r'振込|振替|送金|口座', unified_text))
            if has_tender or (has_acknowledgement and not has_electronic and not has_transfer):
                extracted["payment_method"] = "cash"

    # Final cash fallback
    if "error" not in extracted and not extracted.get("payment_method"):
        has_tender_label = bool(re.search(r'(?:お預り金?|お預かり)(?!票)', unified_text))
        has_change_label_final = bool(re.search(r'釣', unified_text))
        has_electronic = bool(_ELECTRONIC_PAY_RE.search(unified_text))
        if has_tender_label and has_change_label_final and not has_electronic:
            extracted["payment_method"] = "cash"

    # Location: clear for utility bills and payment slips
    if "error" not in extracted and doc_type in ("utility_bill", "payment_slip"):
        extracted["location"] = None

    # Common post-processing
    if "error" not in extracted:
        if doc_type == "receipt":
            ocr_points = extract_points_used(unified_text)
            existing_points = extracted.get("points_used")
            if (
                ocr_points is not None
                and (existing_points is None or (ocr_points > 0 and float(existing_points or 0) == 0))
            ):
                extracted["points_used"] = ocr_points
        total = extracted.get("total")
        points = extracted.get("points_used")
        if total is not None:
            extracted["amount_paid"] = total - points if points else total

    # Strip _confidence if present
    extracted.pop("_confidence", None)

    # Location resolution (confidence-gated, receipts only)
    _notify(on_stage, "resolve_location", "Resolving location", 0.85)
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

    _trim_purchase_store_metadata_location(extracted, unified_text)
    _trim_store_in_store_header_location(extracted, unified_text)

    # Location validation: clear if no OCR evidence supports it
    if "error" not in extracted and doc_type == "receipt" and extracted.get("location"):
        if not _location_has_ocr_evidence(extracted["location"], unified_text):
            extracted["location"] = None
    if "error" not in extracted and doc_type == "receipt" and not extracted.get("location"):
        city_m = re.search(r'(宗像市)', unified_text)
        if city_m:
            extracted["location"] = city_m.group(1)

    # Expand truncated location when OCR has a more detailed address
    if "error" not in extracted and doc_type == "receipt" and extracted.get("location"):
        loc = extracted["location"]
        loc_norm = re.sub(r'\s+', '', loc)
        for line in unified_text.split('\n'):
            line_norm = re.sub(r'\s+', '', line.strip())
            if (len(line_norm) > len(loc_norm) and loc_norm in line_norm
                    and re.search(r'\d+-\d+|丁目|番地', line_norm)):
                extracted["location"] = line_norm
                break

    # Final validation — preview the result so consumers can flash it before "done".
    _notify(on_stage, "validate", _build_validate_detail(extracted), 0.95)
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
    on_stage: StageCallback = None,
    debug: bool = False,
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

    # process_ocr_text skips image loading and OCR — declare the plan upfront
    # with the doc_type already known, so consumers see the same event contract.
    _notify(
        on_stage, "plan", "Pipeline plan",
        0.02,
        payload=_build_plan_payload(page_count=1, pass_budget=passes, doc_type=doc_type),
    )
    _notify(
        on_stage, "classify", f"Detected: {doc_type}",
        0.35,
        payload=_build_classify_payload(doc_type, source="ocr_text_input"),
    )
    _notify(on_stage, "extract", "LLM extraction", 0.40)
    receipt_mutation_trace: list[dict] | None = [] if debug else None
    extracted, pass_history, final_warnings = _run_extraction_pipeline(
        unified_text=unified_text, raw_text=ocr_text,
        ocr_conf=ocr_conf, doc_type=doc_type,
        model=model, passes=passes,
        on_stage=on_stage,
        mutation_trace=receipt_mutation_trace,
    )

    if "_error" in extracted:
        extracted.update({"_warnings": [], "_pass_count": 0, "_model": model,
                          "_pipeline_version": _PIPELINE_VERSION, "_line_items_reliable": False})
        return extracted

    _notify(on_stage, "done", "Complete", 1.0)
    receipt = Receipt(**extracted) if "error" not in extracted else Receipt()
    receipt_payload = _prepare_receipt_output_payload(
        receipt,
        ocr_text,
        mutation_trace=receipt_mutation_trace,
    )
    result = _build_result(
        receipt_payload, final_warnings, pass_history, model,
        debug=debug,
        ocr_confidence=ocr_conf, ocr_source="injected",
        ocr_text=ocr_text,
        mutation_trace=receipt_mutation_trace,
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
    on_stage: StageCallback = None,
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
                on_stage=on_stage,
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
