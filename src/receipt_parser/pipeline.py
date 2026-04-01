"""pipeline.py — Orchestrates all pipeline stages.

Uses Google Cloud Vision OCR → text → LLM (OpenRouter or Ollama) for structured extraction.
Supports batch processing with concurrent API calls via process_batch().
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from .schema import Receipt, VALID_TAX_RATES, REDUCED_RATE, STANDARD_RATE
from .preprocess import load_image, try_extract_text_layer
from .ocr import init_cloud_vision, run_cloud_vision, blocks_to_structured_text, compute_ocr_confidence, OCRResult
from .llm import check_model_available, extract_with_verification, DEFAULT_MODEL
from .validation import validate_receipt
from .normalize import (normalize_fullwidth, clean_handwritten_ocr, strip_barcode_lines,
                        rejoin_price_lines)
from .tracing import PipelineTrace, draw_ocr_bboxes, draw_field_overlay


# ── Japanese Era Constants ───────────────────────────────────────────
# Era name → base year (era year 1 = base + 1)
_ERA_TABLE = {
    "令和": 2018,   # 令和1年 = 2019
    "平成": 1988,   # 平成1年 = 1989
}
_DEFAULT_ERA_BASE = 2018  # Assume 令和 when era name is not found


def _era_to_western_year(era_year: int, era_name: str | None = None) -> int | None:
    """Convert Japanese era year to western year.

    Args:
        era_year: The year within the era (e.g. 8 for 令和8年)
        era_name: The era name if detected from OCR text (e.g. "令和", "平成")

    Returns:
        Western year (e.g. 2026) or None if era_year is invalid.
    """
    if era_year < 1 or era_year > 99:
        return None
    base = _ERA_TABLE.get(era_name, _DEFAULT_ERA_BASE) if era_name else _DEFAULT_ERA_BASE
    return base + era_year


# ── Document Type Detection ──────────────────────────────────────────

_UTILITY_BILL_KEYWORDS = re.compile(
    r'検針|使用量|m3|kWh|ガス料金|水道料金|電気料金|'
    r'ご請求額|引落予定|メーター|基本料金|下水道使用料'
)

_PAYMENT_SLIP_KEYWORDS = re.compile(
    r'払込票|振込.*請求書|振込兼|受領証.*払込|'
    r'依頼人|受取人|コンビニ収納|払込金受領書'
)

_RECEIPT_KEYWORDS = re.compile(r'小計|合計|レジ')


def detect_document_type(text: str) -> str:
    """Classify document type from OCR text using keyword matching."""
    utility_score = len(_UTILITY_BILL_KEYWORDS.findall(text))
    slip_score = len(_PAYMENT_SLIP_KEYWORDS.findall(text))
    receipt_score = len(_RECEIPT_KEYWORDS.findall(text))

    if utility_score >= 2 and utility_score > receipt_score:
        return "utility_bill"
    if slip_score >= 1 and slip_score >= receipt_score:
        return "payment_slip"
    return "receipt"


# ── Points Extraction ────────────────────────────────────────────────

def _extract_points_used(text: str) -> float | None:
    """Extract loyalty points applied as payment from OCR text."""
    patterns = [
        r'ポイント利用\s*[¥￥]?\s*([\d,]+)',
        r'ポイント値引\s*-?\s*([\d,]+)',
        r'ポイント\s*-\s*([\d,]+)',
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return float(m.group(1).replace(',', ''))
    return None


# ── Merchant Mapping ─────────────────────────────────────────────────

_MERCHANT_RULES_PATH = Path(__file__).parent / "merchant_rules.json"


def _apply_merchant_mapping(result: dict) -> dict:
    """Apply merchant_rules.json merchant alias mapping."""
    if not _MERCHANT_RULES_PATH.exists():
        return result
    try:
        rules = json.loads(_MERCHANT_RULES_PATH.read_text(encoding="utf-8"))
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


# ── Financial Extraction Helpers ─────────────────────────────────────

# Match ¥ or ￥ prefix, or 円 suffix amounts
_YEN_INLINE = re.compile(r'[¥￥]\s*([\d,]+)|(?<!\d)([\d,]+)\s*円')


def _parse_yen_match(m) -> float | None:
    """Extract the numeric value from a yen regex match."""
    if m is None:
        return None
    val = m.group(1) or m.group(2)
    return float(val.replace(',', '')) if val else None


# Suffix chars allowed after ¥ amounts: closing parens + JP tax rate markers
_YEN_SUFFIX = r'[)）軽※X除]'


def _extract_yen_nearby(lines: list[str], idx: int, look_ahead: int = 2):
    """Extract ¥ value from line idx (inline) or the next N lines with ¥ values."""
    val = _parse_yen_match(_YEN_INLINE.search(lines[idx].strip()))
    if val is not None:
        return val
    for j in range(idx + 1, min(idx + 1 + look_ahead, len(lines))):
        stripped = lines[j].strip()
        # Match pure ¥/円 lines AND lines with leading text before ¥ (e.g. "1 ¥3,990")
        m = re.match(rf'^[¥￥]\s*([\d,]+){_YEN_SUFFIX}?\s*$', stripped)
        if not m:
            m = re.match(rf'^[\d\s]*[¥￥]\s*([\d,]+){_YEN_SUFFIX}?\s*$', stripped)
        if not m:
            m = re.match(rf'^([\d,]+)\s*円{_YEN_SUFFIX}?\s*$', stripped)
        if m:
            return float(m.group(1).replace(',', ''))
    return None


def _extract_yen_max_nearby(lines: list[str], idx: int, look_ahead: int = 5):
    """Extract the LARGEST ¥ value from line idx or the next N lines."""
    values: list[float] = []
    val = _parse_yen_match(_YEN_INLINE.search(lines[idx].strip()))
    if val is not None:
        return val
    for j in range(idx + 1, min(idx + 1 + look_ahead, len(lines))):
        stripped = lines[j].strip()
        m = re.match(rf'^[¥￥]\s*([\d,]+){_YEN_SUFFIX}?\s*$', stripped)
        if not m:
            m = re.match(rf'^([\d,]+)\s*円{_YEN_SUFFIX}?\s*$', stripped)
        if m:
            values.append(float(m.group(1).replace(',', '')))
        elif re.search(r'小\s*計|現\s*計|お釣り|お釣銭|釣\s*銭|お預り|お預り金|^預$|支払い?方法|支払い?\s|現金|釣銭|クレジット', stripped):
            break
    return max(values) if values else None


def _extract_yen_min_nearby(lines: list[str], idx: int, look_ahead: int = 3):
    """Extract the SMALLEST ¥ value from line idx or the next N lines."""
    values: list[float] = []
    val = _parse_yen_match(_YEN_INLINE.search(lines[idx].strip()))
    if val is not None:
        return val
    for j in range(idx + 1, min(idx + 1 + look_ahead, len(lines))):
        stripped = lines[j].strip()
        m = re.match(rf'^[¥￥]\s*([\d,]+){_YEN_SUFFIX}?\s*$', stripped)
        if not m:
            m = re.match(rf'^([\d,]+)\s*円{_YEN_SUFFIX}?\s*$', stripped)
        if m:
            values.append(float(m.group(1).replace(',', '')))
        elif re.search(r'合\s*計|小\s*計|現\s*計|お釣り|お釣銭|釣\s*銭|お預り|お預り金', stripped):
            break
        elif re.search(r'\d+%', stripped) and re.search(r'対象|消費税|内税|外税|軽減', stripped) and stripped != lines[idx].strip():
            break  # New rate section boundary (must look like a section header)
    return min(values) if values else None


def _extract_financial_totals(text: str) -> dict:
    """Extract subtotal, total, and per-rate taxes directly from OCR text."""
    lines = text.split('\n')
    result: dict = {}
    taxes: list[dict] = []
    _rate_context: str | None = None

    for i, raw in enumerate(lines):
        line = raw.strip()

        rate_ctx_m = re.search(r'(\d+)%\s*対象', line)
        if rate_ctx_m:
            _rate_context = rate_ctx_m.group(1) + '%'

        if re.search(r'消費税[等額]', line) and _rate_context and '対象' not in line:
            val = _extract_yen_min_nearby(lines, i, look_ahead=5)
            if val is not None:
                taxes.append({'rate': _rate_context, 'label': '内消費税等', 'amount': val})
            _rate_context = None

        if (re.search(r'小\s*計', line) or 'お買上高' in line) and '税' not in line:
            val = _extract_yen_nearby(lines, i)
            if val is not None:
                result['subtotal'] = val
                # Also check for an alternative value in case the first is an item price
                val_max = _extract_yen_max_nearby(lines, i, look_ahead=5)
                if val_max and val_max != val:
                    result['_subtotal_alt'] = val_max

        is_total_line = re.search(r'合\s*計', line)
        if not is_total_line and re.match(r'^計$', line) and i > 0:
            prev_context = ' '.join(l.strip() for l in lines[max(0, i - 3):i])
            if '合' in prev_context or '税' in prev_context or '対象' in prev_context:
                is_total_line = True
        if is_total_line and not re.search(r'税\s*合\s*計', line) and '対象' not in line:
            val_max = _extract_yen_max_nearby(lines, i, look_ahead=5)
            val_first = _extract_yen_nearby(lines, i, look_ahead=3)
            if val_max is not None:
                result['total'] = val_max
            if val_first is not None and val_first != val_max:
                result['total_first'] = val_first

        if '現計' in line:
            val = _extract_yen_nearby(lines, i)
            if val is not None:
                result['total'] = val

        if '現金支払' in line:
            val = _extract_yen_nearby(lines, i)
            if val is not None:
                result['total'] = val

        if re.search(r'外税\s*\d+%', line) and '対象' not in line:
            rate_m = re.search(r'(\d+)%', line)
            val = _extract_yen_nearby(lines, i)
            if rate_m and val is not None:
                taxes.append({'rate': rate_m.group(1) + '%', 'label': '外税', 'amount': val})

        if '税額' in line and '対象' not in line:
            rate_m = re.search(r'(\d+)%', line)
            val = _extract_yen_min_nearby(lines, i, look_ahead=3)
            if rate_m and val is not None:
                taxes.append({'rate': rate_m.group(1) + '%', 'label': '税額', 'amount': val})

        if '税合計' in line and '対象' not in line and not taxes:
            val = _extract_yen_nearby(lines, i)
            if val is not None:
                taxes.append({'rate': 'unknown', 'label': '税合計', 'amount': val})

        # Handle standalone "内税" line with ¥ value (e.g. "内税 ¥9,061")
        if re.match(r'^内税$', line) and '対象' not in line:
            val = _extract_yen_nearby(lines, i)
            if val is not None and not taxes:
                rate = _rate_context or 'unknown'
                taxes.append({'rate': rate, 'label': '内税', 'amount': val})

        # Handle inline "消費税等 (rate%) amount円" patterns (single or multi-line)
        m_inline_tax = re.search(r'消費税[等額]?\s*\(?\s*(\d+(?:\.\d+)?)\s*%\s*\)?\s*(\d[\d,]*)\s*円', line)
        if m_inline_tax:
            rate_str = str(int(float(m_inline_tax.group(1)))) + '%'
            tax_val = float(m_inline_tax.group(2).replace(',', ''))
            taxes.append({'rate': rate_str, 'label': '内消費税等', 'amount': tax_val})
        elif re.search(r'消費税[等額]?\s*\(?\s*\d+(?:\.\d+)?\s*%\s*\)?', line):
            # Rate on this line, amount on next line (e.g. "消費税等 (10.00%)\n438円)")
            rate_m = re.search(r'(\d+(?:\.\d+)?)\s*%', line)
            if rate_m and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                amt_m = re.match(r'^(\d[\d,]*)\s*円[)）]?\s*$', next_line)
                if amt_m:
                    rate_str = str(int(float(rate_m.group(1)))) + '%'
                    tax_val = float(amt_m.group(1).replace(',', ''))
                    taxes.append({'rate': rate_str, 'label': '内消費税等', 'amount': tax_val})

    # Parse 内訳 (breakdown) sections for per-rate tax amounts and rate bases
    # Pattern: "内訳 10%\n<inclusive_base>\n<tax>\nR 8%\n<inclusive_base>\n<tax>"
    breakdown_rate_bases: dict[str, float] = {}
    if not taxes:
        breakdown_taxes = []
        in_breakdown = False
        current_rate = None
        breakdown_nums: list[float] = []

        def _save_breakdown_entry():
            if current_rate and len(breakdown_nums) >= 2:
                tax_amt = min(breakdown_nums[:2])
                inclusive_base = max(breakdown_nums[:2])
                pre_tax_base = inclusive_base - tax_amt
                breakdown_taxes.append({
                    'rate': current_rate, 'label': '内訳', 'amount': tax_amt
                })
                if pre_tax_base > 0:
                    breakdown_rate_bases[current_rate] = pre_tax_base

        for raw in lines:
            line = raw.strip()
            if '内訳' in line:
                in_breakdown = True
            if in_breakdown:
                rate_m = re.match(r'^(?:R\s*)?(\d+)%\s*$', line) or re.search(r'内訳\s*(\d+)%', line)
                if rate_m:
                    _save_breakdown_entry()
                    current_rate = rate_m.group(1) + '%'
                    breakdown_nums = []
                    continue
                if current_rate:
                    num_m = re.match(r'^([\d,]+)\s*$', line)
                    if num_m:
                        breakdown_nums.append(float(num_m.group(1).replace(',', '')))
                    elif not line:
                        continue
                    else:
                        _save_breakdown_entry()
                        current_rate = None
                        break
        _save_breakdown_entry()
        if breakdown_taxes:
            taxes = breakdown_taxes

    # Use total_first as subtotal fallback when explicit subtotal not found
    if 'subtotal' not in result and result.get('total_first') is not None:
        total_first = result['total_first']
        total_val = result.get('total')
        if total_val and total_first < total_val and total_first >= total_val * 0.5:
            result['subtotal'] = total_first

    # Sanity check: remove tax entries where amount >= total (clearly wrong)
    total = result.get('total')
    if taxes and total:
        taxes = [t for t in taxes if t['amount'] < total]

    if taxes:
        result['taxes'] = taxes

    if breakdown_rate_bases:
        result['_breakdown_rate_bases'] = breakdown_rate_bases

    return result


def _extract_rate_bases(text: str) -> dict[str, float | None]:
    """Extract per-rate taxable base amounts (対象額) from OCR text."""
    bases: dict[str, float | None] = {}
    lines = text.split('\n')

    for i, raw in enumerate(lines):
        line = raw.strip()
        m = re.search(r'(\d+(?:\.\d+)?)\s*%.*対象', line)
        if not m:
            continue
        if '税額' in line and '対象' not in line:
            continue

        rate_num = float(m.group(1))
        rate_str = f"{int(rate_num)}%" if rate_num == int(rate_num) else f"{rate_num}%"

        yen_m = re.search(r'[¥￥]\s*([\d,]+)', line)
        if yen_m:
            bases[rate_str] = float(yen_m.group(1).replace(',', ''))
        else:
            found = False
            for j in range(i + 1, min(i + 3, len(lines))):
                yen_ahead = re.search(r'[¥￥]\s*([\d,]+)', lines[j].strip())
                if yen_ahead:
                    bases[rate_str] = float(yen_ahead.group(1).replace(',', ''))
                    found = True
                    break
            if not found:
                bases[rate_str] = None

    return bases


def _find_subset_sum(items, target, max_k=3, tolerance=5.0):
    from itertools import combinations
    for k in range(1, min(max_k + 1, len(items) + 1)):
        for combo in combinations(items, k):
            total = sum(t for _, t in combo)
            if abs(total - target) <= tolerance:
                return [i for i, _ in combo]
    return None


def _assign_tax_categories(items, unified_text, ocr_totals, rate_bases):
    """Assign tax_category to line items using OCR evidence. Mutates in-place."""
    if not items:
        return

    valid_rates = set(VALID_TAX_RATES) - {"0%"}
    detected_rates: set[str] = set()
    for tax in ocr_totals.get("taxes", []):
        rate = tax.get("rate", "")
        if rate in valid_rates:
            detected_rates.add(rate)
    for rate in rate_bases:
        if rate in valid_rates:
            detected_rates.add(rate)
    if re.search(r'軽減税率.*8%', unified_text):
        detected_rates.add(REDUCED_RATE)
    for m in re.finditer(r'(\d+)%\s*(?:内税|外税)', unified_text):
        r = m.group(1) + "%"
        if r in valid_rates:
            detected_rates.add(r)
    for m in re.finditer(r'(?:内税|外税)\s*(\d+)%', unified_text):
        r = m.group(1) + "%"
        if r in valid_rates:
            detected_rates.add(r)

    if not detected_rates:
        return
    if len(detected_rates) == 1:
        rate = next(iter(detected_rates))
        for item in items:
            item["tax_category"] = rate
        return

    ocr_lines = unified_text.split('\n')
    item_rates: dict[int, str] = {}
    for idx, item in enumerate(items):
        desc = item.get("description", "")
        if not desc:
            continue
        desc_prefix = desc[:4] if len(desc) >= 4 else desc
        for line in ocr_lines:
            if desc_prefix not in line:
                continue
            if '除' in line:
                item_rates[idx] = STANDARD_RATE
            elif re.search(r'[※X\*軽]', line):
                item_rates[idx] = REDUCED_RATE
            break

    # Note: レジ袋/ポリ袋 hardcode removed — LLM assigns tax category from context

    unassigned = [i for i in range(len(items)) if i not in item_rates]
    if not unassigned:
        for idx, rate in item_rates.items():
            items[idx]["tax_category"] = rate
        return

    assigned_counts: dict[str, int] = {}
    for r in item_rates.values():
        assigned_counts[r] = assigned_counts.get(r, 0) + 1

    tax_amounts = {t["rate"]: t.get("amount", 0) for t in ocr_totals.get("taxes", [])}
    majority_rate = max(
        detected_rates,
        key=lambda r: (assigned_counts.get(r, 0), tax_amounts.get(r, 0), rate_bases.get(r, 0) or 0),
    )
    minority_rates = [r for r in detected_rates if r != majority_rate]
    minority_rate = minority_rates[0] if minority_rates else None

    subset_matched = False
    if minority_rate and unassigned:
        unassigned_items = [(i, items[i].get("total", 0)) for i in unassigned]
        # Try both rate bases — the minority base is more selective (fewer items)
        for try_rate in [minority_rate, majority_rate]:
            try_base = rate_bases.get(try_rate)
            if try_base is None:
                continue
            # Use larger tolerance for tax-category subset matching (tax rounding, discounts)
            match = _find_subset_sum(unassigned_items, try_base, tolerance=50.0)
            if match is not None:
                other_rate = minority_rate if try_rate == majority_rate else majority_rate
                subset_matched = True
                for i in match:
                    item_rates[i] = try_rate
                # Assign remaining to the other rate
                for i in unassigned:
                    if i not in item_rates:
                        item_rates[i] = other_rate
                break

    if subset_matched:
        default_rate = majority_rate
    else:
        marker_rates = set(item_rates.values())
        if REDUCED_RATE in marker_rates and STANDARD_RATE not in marker_rates:
            default_rate = STANDARD_RATE
        elif STANDARD_RATE in marker_rates and REDUCED_RATE not in marker_rates:
            default_rate = REDUCED_RATE
        else:
            default_rate = majority_rate

    for idx in range(len(items)):
        if idx not in item_rates:
            item_rates[idx] = default_rate
    for idx, rate in item_rates.items():
        items[idx]["tax_category"] = rate


# ── Confidence Router ────────────────────────────────────────────────

_HIGH_OCR_CONFIDENCE = 0.85
_HIGH_LLM_CONFIDENCE = 0.7
_LOW_LLM_CONFIDENCE = 0.5

# Financial fields always get overridden by OCR evidence when OCR is reliable,
# because LLM self-reported confidence is not calibrated for numeric accuracy.
_FINANCIAL_FIELDS = {"total", "subtotal", "taxes", "points_used"}


def _should_override_field(field: str, ocr_conf: float, llm_conf: dict | None) -> bool:
    """Decide whether regex should override LLM output for a given field.

    For financial fields (total, subtotal, taxes): always override when OCR
    is reliable — LLM confidence is unreliable for numeric accuracy.

    For other fields: override only when LLM confidence is low.
    """
    if ocr_conf < _HIGH_OCR_CONFIDENCE:
        return False  # OCR too unreliable for regex extraction
    if field in _FINANCIAL_FIELDS:
        return True  # Always override financial fields with OCR evidence
    if llm_conf is None:
        return True  # No confidence info — fall back to legacy behavior
    field_conf = llm_conf.get(field, 0.0)
    return field_conf < _LOW_LLM_CONFIDENCE


def _should_use_regex_as_validation(field: str, ocr_conf: float, llm_conf: dict | None) -> bool:
    """Use regex as a validation signal (warn on disagreement) but don't override."""
    if ocr_conf < _HIGH_OCR_CONFIDENCE:
        return False
    if field in _FINANCIAL_FIELDS:
        return False  # Financial fields get overridden, not just validated
    if llm_conf is None:
        return False
    field_conf = llm_conf.get(field, 0.0)
    return _LOW_LLM_CONFIDENCE <= field_conf < _HIGH_LLM_CONFIDENCE


def _compute_posthoc_confidence(extracted: dict, warnings: list[str]) -> dict:
    """Compute per-field confidence from validation results (post-hoc).

    Instead of asking the LLM for confidence (which changes the prompt and output),
    derive confidence from validation warnings and field presence.
    """
    conf = {}
    warning_text = " ".join(warnings)

    for field in ("merchant", "date", "total", "subtotal", "taxes",
                   "payment_method", "line_items", "points_used"):
        val = extracted.get(field)
        if val is None or (isinstance(val, list) and len(val) == 0):
            conf[field] = 0.0
        elif field in warning_text.lower():
            conf[field] = 0.4  # Field mentioned in warnings
        else:
            conf[field] = 0.9  # No warnings for this field

    return conf


# ── Result Builder ───────────────────────────────────────────────────

def _build_result(receipt, final_warnings, pass_history, model, debug=False, trace=None,
                   ocr_confidence=None, llm_confidence=None,
                   ocr_source=None, ocr_retried=None, ocr_retry_reason=None,
                   ocr_text=None):
    result = receipt.model_dump()
    result["_warnings"] = final_warnings
    result["_pass_count"] = len(pass_history)
    result["_pass_history"] = pass_history
    result["_model"] = model
    result["_pipeline_version"] = "3.0.0"
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


# ── Main Pipeline ────────────────────────────────────────────────────

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

            result = _build_result(receipt, final_warnings, pass_history, model, debug=debug, trace=trace,
                                   ocr_confidence=1.0, llm_confidence=llm_conf_pdf)
            if apply_user_rules:
                result = _apply_merchant_mapping(result)
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
            rotated = cv2.rotate(page_img, cv2.ROTATE_90_CLOCKWISE)
            rotated_result = run_cloud_vision(rotated, ocr_engine, skip_cache=skip_ocr_cache)
            if len(rotated_result.blocks) > len(blocks):
                ocr_result = rotated_result
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
    unified_text = strip_barcode_lines(unified_text)

    # Compute aggregate OCR confidence
    all_blocks_flat = [b for r in all_ocr_results for b in r.blocks]
    ocr_conf = compute_ocr_confidence(all_blocks_flat)

    # Step 0: Detect document type
    doc_type = detect_document_type(unified_text)

    # Receipt-specific pre-processing
    ocr_totals = {}
    if doc_type == "receipt":
        ocr_totals = _extract_financial_totals(unified_text)
        unified_text = rejoin_price_lines(unified_text)
        unified_text = clean_handwritten_ocr(unified_text, ocr_confidence=ocr_conf)

    if not unified_text.strip():
        return {
            "_error": "OCR produced no text.",
            "_warnings": [], "_pass_count": 0, "_model": model,
            "_pipeline_version": "2.0.0", "_line_items_reliable": False,
        }

    trace.log_step("ocr_grouped", data=unified_text)

    # Step 4: LLM extraction
    extracted, pass_history = extract_with_verification(
        unified_text, model=model, passes=passes,
        validate_fn=validate_receipt, doc_type=doc_type,
    )

    if debug:
        for entry in pass_history:
            n = entry["pass"]
            trace.log_step(f"pass{n}_llm_response", data=entry["extraction"])
            if entry["warnings"]:
                trace.log_step(f"pass{n}_warnings", data="\n".join(entry["warnings"]))

    if "error" not in extracted:
        extracted["document_type"] = doc_type
        # Type safety: ensure financial values are numeric
        for fkey in ("total", "subtotal"):
            v = extracted.get(fkey)
            if v is not None:
                try:
                    extracted[fkey] = float(v)
                except (TypeError, ValueError):
                    extracted[fkey] = None

    # ── Receipt post-processing ──
    llm_conf = extracted.get("_confidence")
    if doc_type == "receipt" and "error" not in extracted:
        # 4.5: Financial totals override — gated by confidence router
        # Sanity check: OCR subtotal should be plausible (not an item price)
        ocr_total_val = ocr_totals.get("total")
        if "subtotal" in ocr_totals:
            ocr_sub_val = ocr_totals["subtotal"]
            if ocr_total_val and ocr_sub_val < ocr_total_val * 0.5:
                # First value too low — try alternative subtotal (next ¥ value)
                alt_sub = ocr_totals.get("_subtotal_alt")
                if alt_sub and alt_sub >= ocr_total_val * 0.5:
                    ocr_totals["subtotal"] = alt_sub
                    ocr_sub_val = alt_sub
                else:
                    del ocr_totals["subtotal"]
            if "subtotal" in ocr_totals and _should_override_field("subtotal", ocr_conf, llm_conf):
                extracted["subtotal"] = ocr_sub_val
            elif extracted.get("subtotal") is None:
                extracted["subtotal"] = ocr_sub_val  # Fill missing fields regardless
        if "total" in ocr_totals and _should_override_field("total", ocr_conf, llm_conf):
            ocr_total = float(ocr_totals["total"])
            ocr_first = float(ocr_totals["total_first"]) if ocr_totals.get("total_first") is not None else None
            ocr_sub = float(ocr_totals["subtotal"]) if ocr_totals.get("subtotal") is not None else None
            if ocr_sub and ocr_total < ocr_sub:
                pass  # Don't override — OCR total is suspect
            elif ocr_sub and ocr_total > ocr_sub * 2:
                if ocr_first and ocr_first <= ocr_sub * 1.15:
                    extracted["total"] = ocr_first
            else:
                extracted["total"] = ocr_total
        elif "total" in ocr_totals and extracted.get("total") is None:
            extracted["total"] = float(ocr_totals["total"])  # Fill missing
        if "subtotal" in ocr_totals and "total" in ocr_totals:
            computed_tax = ocr_totals["total"] - ocr_totals["subtotal"]
            if computed_tax >= 0 and _should_override_field("taxes", ocr_conf, llm_conf):
                llm_tax = sum(t.get("amount", 0) for t in extracted.get("taxes", []))
                if abs(llm_tax - computed_tax) > 5:
                    if extracted.get("taxes"):
                        if llm_tax > 0:
                            scale = computed_tax / llm_tax
                            for t in extracted["taxes"]:
                                t["amount"] = round(t["amount"] * scale)
                        else:
                            extracted["taxes"] = [{"rate": "unknown", "label": None, "amount": computed_tax}]
                    elif computed_tax > 0:
                        extracted["taxes"] = [{"rate": "unknown", "label": None, "amount": computed_tax}]
        if ocr_totals.get("taxes") and _should_override_field("taxes", ocr_conf, llm_conf):
            extracted["taxes"] = ocr_totals["taxes"]

        # 4.6: Date fix — supports 令和 and 平成 eras
        western = re.search(r'(20\d{2})\s*年\s*0?(\d{1,2})\s*月\s*0?(\d{1,2})\s*日', unified_text)
        if not western:
            western = re.search(r'(20\d{2})/\s*(\d{1,2})/\s*(\d{1,2})', unified_text)
        if not western:
            western = re.search(r'(20\d{2})-(\d{1,2})-(\d{1,2})', unified_text)
        if western:
            year = int(western.group(1))
            if 2010 <= year <= 2019:
                year += 10
            extracted["date"] = f"{year:04d}-{int(western.group(2)):02d}-{int(western.group(3)):02d}"
        else:
            # Try named era first: 令和8年 or 平成31年
            era_named = re.search(r'(令和|平成)\s*(\d{1,2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日', unified_text)
            if era_named:
                era_name = era_named.group(1)
                era_year = int(era_named.group(2))
                w_year = _era_to_western_year(era_year, era_name)
                if w_year:
                    extracted["date"] = f"{w_year:04d}-{int(era_named.group(3)):02d}-{int(era_named.group(4)):02d}"
            else:
                # Unnamed era: single digit year (assume 令和)
                era = re.search(r'(?<!\d)(\d{1,2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日', unified_text)
                if era:
                    era_year = int(era.group(1))
                    # Detect era from nearby text context
                    era_name = None
                    for name in _ERA_TABLE:
                        if name in unified_text:
                            era_name = name
                            break
                    w_year = _era_to_western_year(era_year, era_name)
                    if w_year and 1989 <= w_year <= 2100:
                        extracted["date"] = f"{w_year:04d}-{int(era.group(2)):02d}-{int(era.group(3)):02d}"

        # 4.7: Payment method fix
        has_cash = '現計' in unified_text
        if not has_cash:
            oazukari = re.search(r'お預り金?\s*[¥￥]?\s*([\d,]+)', unified_text)
            if not oazukari:
                oazukari = re.search(r'お預り金?\s*\n[¥￥]\s*([\d,]+)', unified_text)
            if not oazukari:
                oazukari = re.search(r'(?<![お\w])預\s*[¥￥]\s*([\d,]+)', unified_text)
            if oazukari:
                has_cash = True
        if not has_cash and re.search(r'お釣り|お釣銭|釣銭|(?<![お\w])釣\s*[¥￥]', unified_text):
            has_cash = True
        if not has_cash and '現金' in unified_text:
            has_cash = True
        # Strong cash evidence: tender + change labels present
        change_m = re.search(r'(?:お釣り|お釣銭|釣銭|おつり|釣\s*[¥￥])\s*[¥￥]?\s*([\d,]+)', unified_text)
        change_amount = float(change_m.group(1).replace(',', '')) if change_m else -1
        has_tender = bool(re.search(r'お預り|お預り金|預\s*[¥￥]', unified_text))
        has_change_label = bool(re.search(r'釣', unified_text))
        # Strong if: tender + change labels both present AND change isn't explicitly ¥0
        strong_cash = has_cash and has_tender and has_change_label and change_amount != 0

        if has_cash:
            existing = extracted.get("payment_method")
            # Override electronic payments only with strong evidence (non-zero change)
            if strong_cash:
                extracted["payment_method"] = "cash"
            elif not existing or existing == "cash":
                extracted["payment_method"] = "cash"
            elif _should_override_field("payment_method", ocr_conf, llm_conf) and not existing:
                extracted["payment_method"] = "cash"
        elif extracted.get("payment_method") == "cash":
            is_printed = any(kw in unified_text for kw in ['小計', '合計', '対象', '税率'])
            if is_printed:
                extracted["payment_method"] = None

        # 4.7b: Fallback — if total exists but 0 line items, try to create from OCR
        if not extracted.get("line_items") and extracted.get("total"):
            # Look for 部門 NNN pattern (department-coded items in small shops)
            dept_m = re.search(r'部門\s*(\d+)\s*', unified_text)
            if dept_m:
                extracted["line_items"] = [{
                    "description": f"部門{dept_m.group(1).strip()}",
                    "qty": 1,
                    "unit_price": extracted["total"],
                    "total": extracted["total"],
                    "tax_category": "0%",
                    "discount": 0,
                    "discount_rate": "",
                }]

        # 4.7c: Remove zero-total line items (LLM hallucinations)
        if extracted.get("line_items"):
            extracted["line_items"] = [
                item for item in extracted["line_items"]
                if isinstance(item, dict) and (
                    item.get("total", 0) > 0 or
                    (item.get("unit_price") is not None and item.get("unit_price") > 0)
                )
            ]

        # 4.7d: Handwritten receipt guard — remove spurious single item that equals total
        # Handwritten receipts (領収証) have only a total, no itemized list.
        # The LLM sometimes creates a single line item duplicating the total.
        is_handwritten = not any(kw in unified_text for kw in ['小計', '合計', '対象', '税率'])
        if is_handwritten and extracted.get("line_items") and extracted.get("total"):
            items = extracted["line_items"]
            if len(items) == 1 and isinstance(items[0], dict):
                if abs(items[0].get("total", 0) - extracted["total"]) < 1:
                    extracted["line_items"] = []

        # 4.8: Qty hallucination fix
        if extracted.get("line_items"):
            for item in extracted["line_items"]:
                if not isinstance(item, dict) or item.get("qty", 1) <= 1:
                    continue
                total = item.get("total", 0)
                unit_price = item.get("unit_price")
                if unit_price is None:
                    continue
                total_str = str(int(total)) if total == int(total) else str(total)
                price_str = str(int(unit_price)) if unit_price == int(unit_price) else str(unit_price)
                if total_str not in unified_text and price_str in unified_text:
                    item["qty"] = 1
                    item["total"] = unit_price - item.get("discount", 0)

        # 4.8b: Qty from OCR ×N個 patterns
        if extracted.get("line_items"):
            ocr_lines_raw = unified_text.split('\n')
            for item in extracted["line_items"]:
                if not isinstance(item, dict):
                    continue
                unit_price = item.get("unit_price")
                desc = item.get("description", "")
                if unit_price is None or not desc:
                    continue
                desc_prefix = desc[:4] if len(desc) >= 4 else desc
                price_str = str(int(unit_price)) if unit_price == int(unit_price) else str(unit_price)
                pattern = r'(?:単|@)?' + re.escape(price_str) + r'\s*[×xX]\s*(\d+)\s*個?'
                for li, ocr_line in enumerate(ocr_lines_raw):
                    if desc_prefix not in ocr_line:
                        continue
                    for offset in range(0, 4):
                        if li + offset >= len(ocr_lines_raw):
                            break
                        m = re.search(pattern, ocr_lines_raw[li + offset])
                        if m:
                            correct_qty = float(m.group(1))
                            if correct_qty != item.get("qty", 1) and correct_qty > 1:
                                item["qty"] = correct_qty
                                item["total"] = unit_price * correct_qty - item.get("discount", 0)
                            break
                    break

        # 4.8c: Collapsed-item expansion
        # When the LLM collapses N individually-listed identical items into 1
        # item with qty=N, expand back to N separate items.
        # Guards: only triggers when the receipt lists items individually (N
        # separate OCR lines) AND no ×N bulk quantity pattern exists in OCR.
        if extracted.get("line_items") and len(extracted["line_items"]) == 1:
            item = extracted["line_items"][0]
            if isinstance(item, dict):
                qty = item.get("qty", 1)
                unit_price = item.get("unit_price")
                desc = item.get("description", "")
                if qty > 1 and unit_price is not None and desc:
                    # Count separate OCR lines containing the description
                    ocr_lines = unified_text.split('\n')
                    ocr_desc_count = sum(
                        1 for line in ocr_lines
                        if desc in line and '小計' not in line and '合計' not in line
                    )
                    # Check that OCR does NOT have a bulk qty pattern (×N, xN)
                    price_str = str(int(unit_price)) if unit_price == int(unit_price) else str(unit_price)
                    has_bulk_pattern = bool(re.search(
                        re.escape(price_str) + r'\s*[×xX]\s*\d+', unified_text
                    ))
                    if ocr_desc_count >= qty and not has_bulk_pattern:
                        expanded = []
                        for _ in range(int(qty)):
                            expanded.append({
                                "description": desc,
                                "qty": 1,
                                "unit_price": unit_price,
                                "total": unit_price,
                                "tax_category": item.get("tax_category", "0%"),
                                "discount": 0,
                                "discount_rate": "",
                            })
                        extracted["line_items"] = expanded
                        extracted["subtotal"] = unit_price * qty

        # 4.9: Fix hallucinated line item totals/unit_prices
        if extracted.get("line_items"):
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
                    price_standalone = bool(re.search(r'(?<!\d)' + re.escape(price_str) + r'(?!\d)', line))
                    total_standalone = bool(re.search(r'(?<!\d)' + re.escape(total_str) + r'(?!\d)', line))
                    if price_standalone and not total_standalone:
                        item["total"] = unit_price
                    elif total_standalone and not price_standalone:
                        item["unit_price"] = total
                        item["total"] = total
                    break

        # 4.9b: Fix discount totals
        if extracted.get("line_items"):
            for item in extracted["line_items"]:
                if not isinstance(item, dict):
                    continue
                discount = item.get("discount", 0)
                unit_price = item.get("unit_price")
                total = item.get("total")
                qty = item.get("qty", 1)
                if discount > 0 and unit_price is not None and total is not None:
                    expected = qty * unit_price - discount
                    if abs(total - unit_price * qty) < 1 and abs(total - expected) > 1:
                        item["total"] = expected

        # 4.9c: Detect discounts from OCR text
        if extracted.get("line_items"):
            ocr_lines = unified_text.split('\n')
            for item in extracted["line_items"]:
                if not isinstance(item, dict) or item.get("discount", 0) > 0:
                    continue
                desc = item.get("description", "")
                desc_prefix = desc[:4] if len(desc) >= 4 else desc
                if not desc_prefix:
                    continue
                for li, ocr_line in enumerate(ocr_lines):
                    if desc_prefix not in ocr_line:
                        continue
                    for offset in range(1, 4):
                        if li + offset >= len(ocr_lines):
                            break
                        next_line = ocr_lines[li + offset].strip()
                        if '¥' in next_line and re.search(r'[\u3000-\u9fff]', next_line):
                            break
                        if '割引' in next_line:
                            rate_str = ""
                            discount_amount = 0
                            for k in range(li + offset, min(li + offset + 4, len(ocr_lines))):
                                kline = ocr_lines[k].strip()
                                rate_match = re.match(r'^(\d+)%$', kline)
                                if rate_match:
                                    rate_str = rate_match.group(0)
                                amt_match = re.match(r'^-(\d[\d,]*)$', kline)
                                if amt_match:
                                    discount_amount = float(amt_match.group(1).replace(',', ''))
                            if discount_amount > 0:
                                item["discount"] = discount_amount
                                item["discount_rate"] = rate_str
                                up = item.get("unit_price") or item.get("total", 0)
                                item["total"] = item.get("qty", 1) * up - discount_amount
                            break
                    break

        # 4.10: Tax categories
        if extracted.get("line_items"):
            rate_bases = _extract_rate_bases(unified_text)
            # Merge breakdown rate bases (from 内訳 section) with regex-extracted bases
            breakdown_bases = ocr_totals.get('_breakdown_rate_bases', {})
            for rate, base in breakdown_bases.items():
                if rate not in rate_bases or rate_bases[rate] is None:
                    rate_bases[rate] = base
            _assign_tax_categories(extracted["line_items"], unified_text, ocr_totals, rate_bases)

        # 4.11: Points used — gated by confidence + OCR evidence
        points = _extract_points_used(unified_text)
        if points is not None:
            if _should_override_field("points_used", ocr_conf, llm_conf) or extracted.get("points_used") is None:
                extracted["points_used"] = points
        elif extracted.get("points_used") is not None:
            # LLM claims points were used, but OCR regex found no evidence.
            # Verify: require ポイント利用 or ポイント値引 in text to keep the LLM value.
            has_points_evidence = bool(re.search(r'ポイント利用|ポイント値引', unified_text))
            if not has_points_evidence:
                extracted["points_used"] = None

        # 4.12: Fix pre-tax item totals for inclusive-tax receipts
        # If sum of items != subtotal/total but items appear to be pre-tax amounts, fix them
        if extracted.get("line_items") and extracted.get("total"):
            item_sum = sum(i.get("total", 0) for i in extracted["line_items"] if isinstance(i, dict))
            receipt_total = extracted["total"]
            receipt_subtotal = extracted.get("subtotal") or receipt_total
            # If items sum to a pre-tax amount that matches total/(1+rate), fix to match total
            items_fixed = False
            if len(extracted["line_items"]) == 1 and abs(item_sum - receipt_total) > 1:
                item = extracted["line_items"][0]
                if isinstance(item, dict) and abs(item_sum * 1.10 - receipt_total) < 2:
                    item["total"] = receipt_total
                    if item.get("unit_price") and abs(item["unit_price"] - item_sum) < 1:
                        item["unit_price"] = receipt_total
                    items_fixed = True
                elif isinstance(item, dict) and abs(item_sum * 1.08 - receipt_total) < 2:
                    item["total"] = receipt_total
                    if item.get("unit_price") and abs(item["unit_price"] - item_sum) < 1:
                        item["unit_price"] = receipt_total
                    items_fixed = True

            # Recompute subtotal from corrected items
            if items_fixed:
                extracted["subtotal"] = sum(
                    i.get("total", 0) for i in extracted["line_items"] if isinstance(i, dict)
                )

        # 4.13: Inclusive tax subtotal fix
        # When all OCR tax entries are truly inclusive (内税), subtotal = total.
        # "内税" means tax-inclusive pricing. Do NOT confuse with "内消費税等"
        # which comes from "(うち消費税等)" meaning "of which is tax" (informational).
        if extracted.get("subtotal") and extracted.get("total") and extracted.get("taxes"):
            ocr_tax_labels = [t.get("label", "") for t in ocr_totals.get("taxes", [])]
            # Only true inclusive: label is exactly "内税" or starts with "内税"
            all_inclusive = ocr_tax_labels and all(
                (lbl or '').startswith('内税') for lbl in ocr_tax_labels
            )
            if all_inclusive and "subtotal" not in ocr_totals:
                tax_sum = sum(t.get("amount", 0) for t in extracted["taxes"])
                if extracted["subtotal"] and abs(extracted["subtotal"] + tax_sum - extracted["total"]) < 2:
                    extracted["subtotal"] = extracted["total"]

        # Default subtotal = total for receipts when not found
        if extracted.get("subtotal") is None and extracted.get("total") is not None:
            extracted["subtotal"] = extracted["total"]

    # ── Utility bill post-processing ──
    elif doc_type == "utility_bill" and "error" not in extracted:
        # Check for convenience store payment evidence (overrides bank_payment)
        paid_at_store = bool(re.search(
            r'ローソン|セブン|ファミリーマート|コンビニ|収納代行|領収.*いたしました',
            unified_text,
        ))
        if paid_at_store:
            extracted["payment_method"] = "cash"
        elif re.search(r'口座引落|口座振替|振替させて', unified_text):
            extracted["payment_method"] = "bank_payment"
        elif re.search(r'領入済|収納済', unified_text):
            extracted["payment_method"] = "cash"

        # Service type: bills with both 水道 and 下水道 are water bills
        if extracted.get("service_type") == "sewage" and re.search(r'水道', unified_text):
            # Only override if 水道 appears (not just 下水道)
            water_hits = len(re.findall(r'水道', unified_text))
            sewage_hits = len(re.findall(r'下水道', unified_text))
            if water_hits > sewage_hits:
                extracted["service_type"] = "water"

        # Date: prefer 領収日付 stamp date (often formatted as 'YY.M.D)
        ryoshu_date = re.search(r"領収日付[:\s]*'?(\d{2})\.(\d{1,2})\.(\d{1,2})", unified_text)
        if not ryoshu_date:
            # OCR often fragments labels — look for the date pattern directly
            ryoshu_date = re.search(r"'(\d{2})\.(\d{1,2})\.(\d{1,2})", unified_text)
        if ryoshu_date:
            y = int(ryoshu_date.group(1))
            year = 2000 + y if y < 100 else y
            extracted["date"] = f"{year:04d}-{int(ryoshu_date.group(2)):02d}-{int(ryoshu_date.group(3)):02d}"

    # ── Universal cash detection (all document types) ──
    if "error" not in extracted and not extracted.get("payment_method"):
        if re.search(r'領収証|領収書', unified_text) and not re.search(r'小計|合計|対象|税率', unified_text):
            extracted["payment_method"] = "cash"

    # ── Final cash fallback: tender + change labels present but payment still unset ──
    if "error" not in extracted and not extracted.get("payment_method"):
        has_tender_label = bool(re.search(r'お預り', unified_text))
        has_change_label_final = bool(re.search(r'釣', unified_text))
        if has_tender_label and has_change_label_final:
            extracted["payment_method"] = "cash"

    # ── Common post-processing ──
    if "error" not in extracted:
        total = extracted.get("total")
        points = extracted.get("points_used")
        if total is not None:
            extracted["amount_paid"] = total - points if points else total

    # Strip _confidence if present (LLM may include it even though not requested)
    extracted.pop("_confidence", None)

    # Step 5: Final validation
    try:
        receipt = Receipt(**extracted)
    except Exception:
        receipt = Receipt()
    final_warnings = validate_receipt(receipt)

    # Compute post-hoc confidence from validation results
    posthoc_conf = _compute_posthoc_confidence(extracted, final_warnings)

    if debug and images:
        assert debug_dir is not None
        draw_field_overlay(images[0], all_ocr_results[0].blocks, extracted, debug_dir / "10_field_overlay.png")
        (debug_dir / "pipeline_trace.txt").write_text(trace.summary())

    # Aggregate OCR metadata from first page result
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
        result = _apply_merchant_mapping(result)
    return result


# ── OCR Text Entry Point ──────────────────────────────────────────────

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

    # Detect document type
    doc_type = detect_document_type(unified_text)

    # Receipt-specific pre-processing
    ocr_conf = 0.9  # default confidence for injected text
    ocr_totals = {}
    if doc_type == "receipt":
        ocr_totals = _extract_financial_totals(unified_text)
        unified_text = rejoin_price_lines(unified_text)
        unified_text = clean_handwritten_ocr(unified_text, ocr_confidence=ocr_conf)

    if not unified_text.strip():
        return {
            "_error": "OCR text is empty.",
            "_warnings": [], "_pass_count": 0, "_model": model,
            "_pipeline_version": "3.0.0", "_line_items_reliable": False,
        }

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

    # Strip _confidence if present
    extracted.pop("_confidence", None)

    # Final validation
    try:
        receipt = Receipt(**extracted)
    except Exception:
        receipt = Receipt()
    final_warnings = validate_receipt(receipt)

    result = _build_result(
        receipt, final_warnings, pass_history, model,
        ocr_confidence=ocr_conf, ocr_source="injected",
        ocr_text=ocr_text,
    )
    if apply_user_rules:
        result = _apply_merchant_mapping(result)
    return result


# ── Batch Processing ────────────────────────────────────────────────

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

    Args:
        file_paths: List of image/PDF paths to process.
        max_workers: Maximum concurrent API calls (default: 4).
        on_progress: Optional callback(file_path, result, index, total)
            called as each file completes.

    Returns:
        List of result dicts in the same order as file_paths.
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
    # Inject batch metadata into each result
    for r in results:
        if r:
            r["_batch_total_s"] = round(elapsed, 2)
            r["_batch_workers"] = max_workers

    return results  # type: ignore[return-value]
