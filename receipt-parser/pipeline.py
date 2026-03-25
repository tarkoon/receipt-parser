"""pipeline.py — Orchestrates all pipeline stages.

Uses Google Cloud Vision OCR → text → qwen3.5 LLM for structured extraction.
"""

import re
from pathlib import Path

import cv2
import numpy as np

from schema import Receipt
from preprocessing import load_image, try_extract_text_layer
from ocr import init_cloud_vision, run_cloud_vision, blocks_to_structured_text
from extraction import check_ollama_available, extract_with_verification
from validation import validate_receipt
from normalization import (normalize_fullwidth, clean_handwritten_ocr, strip_barcode_lines,
                          rejoin_price_lines)
from debug_visual import PipelineTrace, draw_ocr_bboxes, draw_field_overlay


def _extract_yen_nearby(lines: list[str], idx: int, look_ahead: int = 2):
    """Extract ¥ value from line idx (inline) or the next N pure-¥ lines."""
    # Inline value on the same line
    m = re.search(r'[¥￥]\s*([\d,]+)', lines[idx].strip())
    if m:
        return float(m.group(1).replace(',', ''))
    # Next N lines — must be a standalone ¥ line
    for j in range(idx + 1, min(idx + 1 + look_ahead, len(lines))):
        m = re.match(r'^[¥￥]\s*([\d,]+)[)）]?\s*$', lines[j].strip())
        if m:
            return float(m.group(1).replace(',', ''))
    return None


def _extract_financial_totals(text: str) -> dict:
    """Extract subtotal, total, and per-rate taxes directly from OCR text.

    Returns a dict with optional keys: 'subtotal', 'total', 'taxes'.
    Only high-confidence matches (label near a ¥ value) are returned.
    """
    lines = text.split('\n')
    result: dict = {}
    taxes: list[dict] = []

    for i, raw in enumerate(lines):
        line = raw.strip()

        # Subtotal: 小計 (exclude 税合計)
        if '小計' in line and '税' not in line:
            val = _extract_yen_nearby(lines, i)
            if val is not None:
                result['subtotal'] = val

        # Total: 合計 (exclude 税合計, 対象額合計)
        if '合計' in line and '税合計' not in line and '対象' not in line:
            val = _extract_yen_nearby(lines, i)
            if val is not None:
                result['total'] = val

        # Cash total: 現計 — most reliable total indicator, overrides 合計
        if '現計' in line:
            val = _extract_yen_nearby(lines, i)
            if val is not None:
                result['total'] = val

        # Per-rate tax: 外税N% (NOT 対象額)
        if re.search(r'外税\s*\d+%', line) and '対象' not in line:
            rate_m = re.search(r'(\d+)%', line)
            val = _extract_yen_nearby(lines, i)
            if rate_m and val is not None:
                taxes.append({'rate': rate_m.group(1) + '%', 'label': '外税', 'amount': val})

        # Per-rate tax: 税率N%税額 (NOT 対象額)
        if '税額' in line and '対象' not in line:
            rate_m = re.search(r'(\d+)%', line)
            val = _extract_yen_nearby(lines, i)
            if rate_m and val is not None:
                taxes.append({'rate': rate_m.group(1) + '%', 'label': '税額', 'amount': val})

        # Tax total: 税合計 (fallback — single entry without rate)
        if '税合計' in line and '対象' not in line and not taxes:
            val = _extract_yen_nearby(lines, i)
            if val is not None:
                taxes.append({'rate': 'unknown', 'label': '税合計', 'amount': val})

    if taxes:
        result['taxes'] = taxes

    return result


def _build_result(receipt, final_warnings, pass_history, model, debug=False, trace=None):
    """Assemble the result dict."""
    result = receipt.model_dump()
    result["_warnings"] = final_warnings
    result["_pass_count"] = len(pass_history)
    result["_model"] = model
    result["_pipeline_version"] = "1.2.0"
    line_item_warnings = [w for w in final_warnings if "Line " in w]
    result["_line_items_reliable"] = len(line_item_warnings) == 0
    if debug and trace:
        result["_debug_dir"] = str(trace.debug_dir)
        result["_trace"] = trace.summary()
        result["_pass_history"] = pass_history
    return result


def process_document(
    file_path: Path,
    model: str = "qwen3.5:9b",
    debug: bool = False,
    passes: int = 1,
    ocr_engine=None,
) -> dict:
    """Main pipeline. Uses Cloud Vision OCR + qwen3.5 LLM extraction."""
    file_path = Path(file_path)

    check_ollama_available(model)

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

            if debug:
                assert debug_dir is not None
                (debug_dir / "03_ocr_bboxes.txt").write_text(
                    "SKIPPED: Digital PDF fast path — no OCR performed.")

            extracted, pass_history = extract_with_verification(
                digital_text, model=model, passes=passes,
                validate_fn=validate_receipt,
            )

            if debug:
                for entry in pass_history:
                    n = entry["pass"]
                    trace.log_step(f"pass{n}_llm_response", data=entry["extraction"])
                    if entry["warnings"]:
                        trace.log_step(f"pass{n}_warnings",
                                       data="\n".join(entry["warnings"]))

            try:
                receipt = Receipt(**extracted)
            except Exception:
                receipt = Receipt()
            final_warnings = validate_receipt(receipt)

            if debug:
                assert debug_dir is not None
                (debug_dir / "10_field_overlay.txt").write_text(
                    "SKIPPED: Digital PDF fast path — no OCR bounding boxes available.")
                (debug_dir / "pipeline_trace.txt").write_text(trace.summary())

            return _build_result(receipt, final_warnings, pass_history, model,
                                 debug=debug, trace=trace)

    # Step 2: Init OCR engine
    if ocr_engine is None:
        ocr_engine = init_cloud_vision()

    # Step 3: OCR per page, concatenate
    all_ocr_blocks = []
    text_parts = []

    for i, page_img in enumerate(images):
        blocks = run_cloud_vision(page_img, ocr_engine)
        all_ocr_blocks.append(blocks)

        # 90° rotation fallback
        if len(blocks) < 3:
            rotated = cv2.rotate(page_img, cv2.ROTATE_90_CLOCKWISE)
            rotated_blocks = run_cloud_vision(rotated, ocr_engine)
            if len(rotated_blocks) > len(blocks):
                blocks = rotated_blocks
                all_ocr_blocks[-1] = blocks

        if debug:
            assert debug_dir is not None
            draw_ocr_bboxes(page_img, blocks,
                            debug_dir / f"03_page{i+1}_ocr_bboxes.png")

        page_text = blocks_to_structured_text(blocks)
        if i > 0:
            text_parts.append(f"--- PAGE {i+1} ---")
        text_parts.append(page_text)

    unified_text = "\n".join(text_parts)
    unified_text = normalize_fullwidth(unified_text)
    unified_text = strip_barcode_lines(unified_text)

    # Extract financial totals BEFORE price joining — values are on their own
    # lines at this point, making regex matching reliable.
    ocr_totals = _extract_financial_totals(unified_text)

    unified_text = rejoin_price_lines(unified_text)
    unified_text = clean_handwritten_ocr(unified_text)

    if not unified_text.strip():
        return {
            "_error": "OCR produced no text. Image may be blank, unreadable, or not a document.",
            "_warnings": [], "_pass_count": 0, "_model": model,
            "_pipeline_version": "1.2.0", "_line_items_reliable": False,
        }

    trace.log_step("ocr_grouped", data=unified_text)

    # Step 4: LLM extraction
    extracted, pass_history = extract_with_verification(
        unified_text, model=model, passes=passes,
        validate_fn=validate_receipt,
    )

    if debug:
        for entry in pass_history:
            n = entry["pass"]
            trace.log_step(f"pass{n}_llm_response", data=entry["extraction"])
            if entry["warnings"]:
                trace.log_step(f"pass{n}_warnings",
                               data="\n".join(entry["warnings"]))

    # Step 4.5: Fix financial totals from OCR text (extracted earlier, before
    # price joining, when ¥ values are still on their own lines).
    if "error" not in extracted:
        if "subtotal" in ocr_totals:
            extracted["subtotal"] = ocr_totals["subtotal"]
        if "total" in ocr_totals:
            extracted["total"] = ocr_totals["total"]
        # Compute tax from arithmetic if we have both subtotal and total
        if "subtotal" in ocr_totals and "total" in ocr_totals:
            computed_tax = ocr_totals["total"] - ocr_totals["subtotal"]
            if computed_tax >= 0:
                llm_tax = sum(t.get("amount", 0) for t in extracted.get("taxes", []))
                if abs(llm_tax - computed_tax) > 5:
                    # LLM tax is wrong — override, keeping rate/label from LLM if available
                    if extracted.get("taxes"):
                        # Scale existing entries proportionally
                        if llm_tax > 0:
                            scale = computed_tax / llm_tax
                            for t in extracted["taxes"]:
                                t["amount"] = round(t["amount"] * scale)
                        else:
                            extracted["taxes"] = [{"rate": "unknown", "label": None,
                                                   "amount": computed_tax}]
                    elif computed_tax > 0:
                        extracted["taxes"] = [{"rate": "unknown", "label": None,
                                               "amount": computed_tax}]
        # Also override with direct per-rate tax extraction if available
        if ocr_totals.get("taxes"):
            extracted["taxes"] = ocr_totals["taxes"]

    # Step 4.6: Fix dates by extracting directly from OCR text.
    #           The LLM often misreads years. We always override with OCR ground truth.
    # The LLM often misreads years. We always override with the OCR ground truth.
    if "error" not in extracted:
        # Try western date: YYYY年MM月DD日 or YYYY/M/D
        western = re.search(r'(20\d{2})\s*年\s*0?(\d{1,2})\s*月\s*0?(\d{1,2})\s*日', unified_text)
        if not western:
            western = re.search(r'(20\d{2})/\s*(\d{1,2})/\s*(\d{1,2})', unified_text)
        if western:
            year = int(western.group(1))
            # Fix common OCR misread: 2016↔2026, 2015↔2025 (rotated images)
            # If OCR reads 201X but we're in 202X, the '1' is likely '2'
            if 2010 <= year <= 2019:
                year += 10  # 2016 → 2026, 2015 → 2025
            extracted["date"] = f"{year:04d}-{int(western.group(2)):02d}-{int(western.group(3)):02d}"
        else:
            # Try era date: N 年 M 月 D 日 where N is 1-9 (令和)
            era = re.search(r'(?<!\d)(\d)\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日', unified_text)
            if era and 1 <= int(era.group(1)) <= 9:
                extracted["date"] = f"{2018 + int(era.group(1)):04d}-{int(era.group(2)):02d}-{int(era.group(3)):02d}"

    # Step 4.7: Fix payment_method from OCR text.
    #           現計 = definitive cash total line.
    #           お預り amount > total = cash (overpayment means cash tendered).
    #           Electronic payments (WAON, credit) don't use お預り.
    if "error" not in extracted:
        has_cash = '現計' in unified_text
        if not has_cash:
            oazukari = re.search(r'お預り\s*[¥￥]?\s*([\d,]+)', unified_text)
            if oazukari:
                tendered = float(oazukari.group(1).replace(',', ''))
                total_val = extracted.get("total") or 0
                if total_val and tendered > total_val:
                    has_cash = True
        if has_cash:
            extracted["payment_method"] = "cash"
        elif '領収証' in unified_text and not extracted.get("payment_method"):
            extracted["payment_method"] = "cash"

    # Step 4.8: Fix qty hallucinations from OCR text.
    #           If LLM set qty > 1, check if the resulting total actually appears
    #           in the OCR text. If it doesn't but the unit_price does, reset qty=1.
    if "error" not in extracted and extracted.get("line_items"):
        for item in extracted["line_items"]:
            if not isinstance(item, dict) or item.get("qty", 1) <= 1:
                continue
            total = item.get("total", 0)
            unit_price = item.get("unit_price")
            if unit_price is None:
                continue
            # If qty*unit_price == total AND total appears in OCR, trust it
            total_str = str(int(total)) if total == int(total) else str(total)
            price_str = str(int(unit_price)) if unit_price == int(unit_price) else str(unit_price)
            total_in_ocr = total_str in unified_text
            price_in_ocr = price_str in unified_text
            # Reset if total is NOT in OCR but unit_price IS — the LLM
            # likely hallucinated the qty from a number in the product name
            if not total_in_ocr and price_in_ocr:
                item["qty"] = 1
                item["total"] = unit_price - item.get("discount", 0)

    # Step 4.9: Fix hallucinated line item totals/unit_prices.
    #           If qty=1, discount=0, and total != unit_price, one of them is wrong.
    #           Check which value actually appears as a standalone number on the same
    #           OCR line as the item description, and use that as the source of truth.
    if "error" not in extracted and extracted.get("line_items"):
        ocr_lines = unified_text.split('\n')
        for item in extracted["line_items"]:
            if not isinstance(item, dict):
                continue
            qty = item.get("qty", 1)
            discount = item.get("discount", 0)
            unit_price = item.get("unit_price")
            total = item.get("total")
            if qty != 1 or discount != 0 or unit_price is None or total is None:
                continue
            if abs(total - unit_price) < 1:
                continue
            desc = item.get("description", "")
            desc_prefix = desc[:5] if len(desc) >= 5 else desc
            price_str = str(int(unit_price)) if unit_price == int(unit_price) else str(unit_price)
            total_str = str(int(total)) if total == int(total) else str(total)
            for line in ocr_lines:
                if desc_prefix not in line:
                    continue
                # Use word-boundary check: the number must not be a substring of a larger number
                price_standalone = bool(re.search(r'(?<!\d)' + re.escape(price_str) + r'(?!\d)', line))
                total_standalone = bool(re.search(r'(?<!\d)' + re.escape(total_str) + r'(?!\d)', line))
                if price_standalone and not total_standalone:
                    item["total"] = unit_price
                elif total_standalone and not price_standalone:
                    item["unit_price"] = total
                    item["total"] = total
                break

    # Step 5: Final validation
    try:
        receipt = Receipt(**extracted)
    except Exception:
        receipt = Receipt()
    final_warnings = validate_receipt(receipt)

    if debug and images:
        assert debug_dir is not None
        draw_field_overlay(images[0], all_ocr_blocks[0], extracted,
                           debug_dir / "10_field_overlay.png")
        (debug_dir / "pipeline_trace.txt").write_text(trace.summary())

    return _build_result(receipt, final_warnings, pass_history, model,
                         debug=debug, trace=trace)
