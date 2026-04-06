"""pipeline_receipt.py — Receipt-specific post-processing and financial extraction.

Extracted from pipeline.py for maintainability. Contains:
- Financial totals extraction from OCR text
- Yen amount helpers
- Tax category assignment
- Receipt post-processing (date, payment, line items, etc.)
"""

import re
from itertools import combinations

from .schema import VALID_TAX_RATES, REDUCED_RATE, STANDARD_RATE
from .patterns import (
    YEN_INLINE, YEN_SUFFIX, ERA_TABLE, should_override_field, era_to_western_year,
)


# ── Tax Normalization ─────────────────────────────────────────────

# Canonical labels: 内税 (inclusive), 外税 (exclusive), 非課税 (exempt)
def normalize_tax_rate(rate: str) -> str:
    """Normalize tax rate string: '10.0%' -> '10%', '8.00%' -> '8%'."""
    if not rate or rate == 'unknown':
        return rate
    m = re.match(r'(\d+(?:\.\d+)?)\s*%', rate)
    if m:
        return str(int(float(m.group(1)))) + '%'
    return rate


def normalize_tax_label(
    label: str | None, text: str = "",
    subtotal: float | None = None, total: float | None = None,
    tax_sum: float | None = None,
) -> str:
    """Normalize a tax label to canonical set: 内税, 外税, 非課税."""
    label = label or ""

    # 1. Unambiguous label keywords
    if '非課税' in label:
        return '非課税'
    if '外税' in label or '税抜' in label:
        return '外税'
    if label == '内税':
        return '内税'

    # 2. OCR text keyword check (more reliable than math for 内/外)
    has_exclusive = bool(re.search(r'外税|外\d+%|税抜対象|\d+%\s*税\s*[¥￥\n]', text))
    has_inclusive = bool(re.search(r'内税|内消費', text))

    if has_exclusive and not has_inclusive:
        return '外税'
    if has_inclusive and not has_exclusive:
        return '内税'
    # If both present, prefer 内税 (内税 is the primary method;
    # 税抜額 is often just a pre-tax summary on 内税 receipts)
    if has_inclusive and has_exclusive:
        return '内税'

    # 3. Math fallback: only when no OCR keywords found
    if subtotal is not None and total is not None and tax_sum is not None and tax_sum > 0:
        # If subtotal ≈ total, tax is inclusive
        if abs(subtotal - total) <= 5:
            return '内税'
        # If subtotal + tax ≈ total and subtotal < total, tax is exclusive
        if abs(subtotal + tax_sum - total) <= 5 and subtotal < total - 1:
            return '外税'

    # 4. Default to 内税 (most common in Japanese receipts)
    return '内税'


# ── Yen Extraction Helpers ─────────────────────────────────────────

def _parse_yen_match(m) -> float | None:
    """Extract the numeric value from a yen regex match."""
    if m is None:
        return None
    val = m.group(1) or m.group(2)
    return float(val.replace(',', '')) if val else None


def _extract_yen_nearby(lines: list[str], idx: int, look_ahead: int = 2):
    """Extract ¥ value from line idx (inline) or the next N lines with ¥ values."""
    val = _parse_yen_match(YEN_INLINE.search(lines[idx].strip()))
    if val is not None:
        return val
    for j in range(idx + 1, min(idx + 1 + look_ahead, len(lines))):
        stripped = lines[j].strip()
        m = re.match(rf'^[¥￥]\s*([\d,]+){YEN_SUFFIX}?\s*$', stripped)
        if not m:
            m = re.match(rf'^[\d\s]*[¥￥]\s*([\d,]+){YEN_SUFFIX}?\s*$', stripped)
        if not m:
            m = re.match(rf'^([\d,]+)\s*円{YEN_SUFFIX}?\s*$', stripped)
        if m:
            return float(m.group(1).replace(',', ''))
    return None


def _extract_yen_max_nearby(lines: list[str], idx: int, look_ahead: int = 5):
    """Extract the LARGEST ¥ value from line idx or the next N lines."""
    values: list[float] = []
    val = _parse_yen_match(YEN_INLINE.search(lines[idx].strip()))
    if val is not None:
        return val
    for j in range(idx + 1, min(idx + 1 + look_ahead, len(lines))):
        stripped = lines[j].strip()
        m = re.match(rf'^[¥￥]\s*([\d,]+){YEN_SUFFIX}?\s*$', stripped)
        if not m:
            m = re.match(rf'^([\d,]+)\s*円{YEN_SUFFIX}?\s*$', stripped)
        if m:
            values.append(float(m.group(1).replace(',', '')))
        elif re.search(r'小\s*計|現\s*計|お釣り|お釣銭|釣\s*銭|お預り|お預り金|^預$|支払い?方法|支払い?\s|現金|釣銭|クレジット', stripped):
            break
    return max(values) if values else None


def _extract_all_yen_nearby(lines: list[str], idx: int, look_ahead: int = 6) -> list[float]:
    """Extract ALL ¥ values from the next N lines (for candidate analysis)."""
    values: list[float] = []
    val = _parse_yen_match(YEN_INLINE.search(lines[idx].strip()))
    if val is not None:
        values.append(val)
    for j in range(idx + 1, min(idx + 1 + look_ahead, len(lines))):
        stripped = lines[j].strip()
        m = re.match(rf'^[¥￥]\s*([\d,]+){YEN_SUFFIX}?\s*$', stripped)
        if not m:
            m = re.match(rf'^([\d,]+)\s*円{YEN_SUFFIX}?\s*$', stripped)
        if m:
            values.append(float(m.group(1).replace(',', '')))
        elif re.search(r'合\s*計|現\s*計|お釣り|お預り', stripped):
            break
    return values


def _extract_yen_min_nearby(lines: list[str], idx: int, look_ahead: int = 3):
    """Extract the SMALLEST ¥ value from line idx or the next N lines."""
    values: list[float] = []
    val = _parse_yen_match(YEN_INLINE.search(lines[idx].strip()))
    if val is not None:
        return val
    for j in range(idx + 1, min(idx + 1 + look_ahead, len(lines))):
        stripped = lines[j].strip()
        m = re.match(rf'^[¥￥]\s*([\d,]+){YEN_SUFFIX}?\s*$', stripped)
        if not m:
            m = re.match(rf'^([\d,]+)\s*円{YEN_SUFFIX}?\s*$', stripped)
        if m:
            values.append(float(m.group(1).replace(',', '')))
        elif re.search(r'合\s*計|小\s*計|現\s*計|お釣り|お釣銭|釣\s*銭|お預り|お預り金', stripped):
            break
        elif re.search(r'\d+%', stripped) and re.search(r'対象|消費税|内税|外税|軽減', stripped) and stripped != lines[idx].strip():
            break
    return min(values) if values else None


# ── Financial Totals ───────────────────────────────────────────────

def extract_financial_totals(text: str) -> dict:
    """Extract subtotal, total, and per-rate taxes directly from OCR text."""
    lines = text.split('\n')
    result: dict = {}
    taxes: list[dict] = []
    _rate_context: str | None = None

    for i, raw in enumerate(lines):
        line = raw.strip()

        rate_ctx_m = re.search(r'(\d+(?:\.\d+)?)%.*対象', line)
        if rate_ctx_m:
            _rate_context = normalize_tax_rate(rate_ctx_m.group(1) + '%')

        _has_specific_taxes = any(t.get('label') in ('税額', '外税', '内税') for t in taxes)
        if re.search(r'消費税[等額]', line) and _rate_context and '対象' not in line and not _has_specific_taxes:
            val = _extract_yen_min_nearby(lines, i, look_ahead=5)
            if val is not None:
                taxes.append({'rate': _rate_context, 'label': '消費税等', 'amount': val})
            _rate_context = None

        if (re.search(r'小\s*計', line) or 'お買上高' in line) and '税' not in line:
            val = _extract_yen_nearby(lines, i)
            if val is not None:
                result['subtotal'] = val
                all_nearby = _extract_all_yen_nearby(lines, i, look_ahead=6)
                alts = [v for v in all_nearby if v != val]
                if alts:
                    result['_subtotal_alt'] = max(alts)
                    result['_subtotal_candidates'] = all_nearby

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
            if val is not None and 'total' not in result:
                result['total'] = val

        if '現金支払' in line:
            val = _extract_yen_nearby(lines, i)
            if val is not None and 'total' not in result:
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
                _rate_context = None

        # Per-rate shorthand tax: N%税 (e.g., 8%税 ¥48)
        if re.match(r'^\s*\d+%\s*税\s*$', line) and '対象' not in line and '合計' not in line:
            rate_m = re.search(r'(\d+)%', line)
            val = _extract_yen_nearby(lines, i, look_ahead=2)
            if rate_m and val is not None and not any(t['rate'] == rate_m.group(1) + '%' for t in taxes):
                taxes.append({'rate': rate_m.group(1) + '%', 'label': '外税', 'amount': val})

        # Per-rate inclusive tax: (N%内) or (※N%内) pattern
        per_rate_incl = re.search(r'(\d+(?:\.\d+)?)\s*%\s*内\s*\)?$', line)
        if per_rate_incl and '対象' not in line:
            val = _extract_yen_nearby(lines, i, look_ahead=2)
            if val is not None:
                rate = normalize_tax_rate(per_rate_incl.group(1) + '%')
                taxes.append({'rate': rate, 'label': '内税', 'amount': val})

        if '税合計' in line and '対象' not in line and not taxes:
            val = _extract_yen_nearby(lines, i)
            if val is not None:
                rate = _rate_context or 'unknown'
                taxes.append({'rate': rate, 'label': '税合計', 'amount': val})

        if re.match(r'^内税$', line) and '対象' not in line:
            val = _extract_yen_nearby(lines, i)
            if val is not None and not taxes:
                rate = _rate_context or 'unknown'
                # Look nearby for a rate if still unknown
                if rate == 'unknown':
                    for j in range(max(0, i - 2), min(i + 4, len(lines))):
                        if j == i:
                            continue
                        nearby_rate_m = re.search(r'(\d+(?:\.\d+)?)%', lines[j].strip())
                        if nearby_rate_m:
                            rate = normalize_tax_rate(nearby_rate_m.group(1) + '%')
                            break
                taxes.append({'rate': rate, 'label': '内税', 'amount': val})

        # Non-taxable (非課税) detection
        if '非課税' in line and not any(t.get('rate') == '0%' for t in taxes):
            taxes.append({'rate': '0%', 'label': '非課税', 'amount': 0})

        m_inline_tax = re.search(r'消費税[等額]?\s*\(?\s*(\d+(?:\.\d+)?)\s*%\s*\)?\s*(\d[\d,]*)\s*円', line)
        if m_inline_tax:
            rate_str = str(int(float(m_inline_tax.group(1)))) + '%'
            tax_val = float(m_inline_tax.group(2).replace(',', ''))
            taxes.append({'rate': rate_str, 'label': '消費税等', 'amount': tax_val})
        elif re.search(r'消費税[等額]?\s*\(?\s*\d+(?:\.\d+)?\s*%\s*\)?', line):
            rate_m = re.search(r'(\d+(?:\.\d+)?)\s*%', line)
            if rate_m and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                amt_m = re.match(r'^(\d[\d,]*)\s*円[)）]?\s*$', next_line)
                if amt_m:
                    rate_str = str(int(float(rate_m.group(1)))) + '%'
                    tax_val = float(amt_m.group(1).replace(',', ''))
                    taxes.append({'rate': rate_str, 'label': '消費税等', 'amount': tax_val})

    # Parse 内訳 (breakdown) sections
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

    # Use total_first as subtotal fallback
    if 'subtotal' not in result and result.get('total_first') is not None:
        total_first = result['total_first']
        total_val = result.get('total')
        if total_val and total_first < total_val and total_first >= total_val * 0.5:
            result['subtotal'] = total_first

    # Sanity check: remove tax entries where amount >= total
    total = result.get('total')
    if taxes and total:
        taxes = [t for t in taxes if t['amount'] < total]

    if taxes:
        result['taxes'] = taxes

    if breakdown_rate_bases:
        result['_breakdown_rate_bases'] = breakdown_rate_bases

    return result


def extract_rate_bases(text: str) -> dict[str, float | None]:
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


# ── Points Extraction ──────────────────────────────────────────────

def extract_points_used(text: str) -> float | None:
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


# ── Tax Category Assignment ────────────────────────────────────────

def _find_subset_sum(items, target, max_k=3, tolerance=5.0):
    for k in range(1, min(max_k + 1, len(items) + 1)):
        for combo in combinations(items, k):
            total = sum(t for _, t in combo)
            if abs(total - target) <= tolerance:
                return [i for i, _ in combo]
    return None


def assign_tax_categories(items, unified_text, ocr_totals, rate_bases):
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
        for try_rate in [minority_rate, majority_rate]:
            try_base = rate_bases.get(try_rate)
            if try_base is None:
                continue
            match = _find_subset_sum(unassigned_items, try_base, tolerance=50.0)
            if match is not None:
                other_rate = minority_rate if try_rate == majority_rate else majority_rate
                subset_matched = True
                for i in match:
                    item_rates[i] = try_rate
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


# ── Receipt Post-Processing ────────────────────────────────────────

def postprocess_receipt(
    extracted: dict,
    unified_text: str,
    ocr_conf: float,
    ocr_totals: dict,
    llm_conf: dict | None,
    model: str,
) -> dict:
    """Apply all receipt-specific post-processing to the LLM extraction.

    Includes: financial override, date/era fix, payment method, line item fixes,
    tax assignment, points, inclusive tax handling, subtotal default.
    """
    # 4.5: Financial totals override — gated by confidence router
    ocr_total_val = ocr_totals.get("total")
    if "subtotal" in ocr_totals:
        ocr_sub_val = ocr_totals["subtotal"]
        if ocr_total_val and ocr_sub_val < ocr_total_val * 0.5:
            candidates = ocr_totals.get("_subtotal_candidates", [])
            best_sub = None
            if candidates and ocr_total_val:
                plausible = [v for v in candidates
                             if ocr_total_val * 0.5 <= v <= ocr_total_val]
                if plausible:
                    best_sub = min(plausible)
            if best_sub:
                ocr_totals["subtotal"] = best_sub
                ocr_sub_val = best_sub
            else:
                alt_sub = ocr_totals.get("_subtotal_alt")
                if alt_sub and alt_sub >= ocr_total_val * 0.5:
                    ocr_totals["subtotal"] = alt_sub
                    ocr_sub_val = alt_sub
                else:
                    del ocr_totals["subtotal"]
        if "subtotal" in ocr_totals and should_override_field("subtotal", ocr_conf, llm_conf):
            extracted["subtotal"] = ocr_sub_val
        elif extracted.get("subtotal") is None:
            extracted["subtotal"] = ocr_sub_val  # Fill missing fields regardless
    if "total" in ocr_totals and should_override_field("total", ocr_conf, llm_conf):
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
        if computed_tax >= 0 and should_override_field("taxes", ocr_conf, llm_conf):
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
    if ocr_totals.get("taxes") and should_override_field("taxes", ocr_conf, llm_conf):
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
        era_named = re.search(r'(令和|平成)\s*(\d{1,2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日', unified_text)
        if era_named:
            era_name = era_named.group(1)
            era_year = int(era_named.group(2))
            w_year = era_to_western_year(era_year, era_name)
            if w_year:
                extracted["date"] = f"{w_year:04d}-{int(era_named.group(3)):02d}-{int(era_named.group(4)):02d}"
        else:
            era = re.search(r'(?<!\d)(\d{1,2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日', unified_text)
            if era:
                era_year = int(era.group(1))
                era_name = None
                for name in ERA_TABLE:
                    if name in unified_text:
                        era_name = name
                        break
                w_year = era_to_western_year(era_year, era_name)
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
    change_m = re.search(r'(?:お釣り|お釣銭|釣銭|おつり|釣\s*[¥￥])\s*[¥￥]?\s*([\d,]+)', unified_text)
    change_amount = float(change_m.group(1).replace(',', '')) if change_m else -1
    has_tender = bool(re.search(r'お預り|お預り金|預\s*[¥￥]', unified_text))
    has_change_label = bool(re.search(r'釣', unified_text))
    strong_cash = has_cash and has_tender and has_change_label and change_amount != 0

    if has_cash:
        existing = extracted.get("payment_method")
        if strong_cash:
            extracted["payment_method"] = "cash"
        elif not existing or existing == "cash":
            extracted["payment_method"] = "cash"
        elif should_override_field("payment_method", ocr_conf, llm_conf) and not existing:
            extracted["payment_method"] = "cash"
    elif extracted.get("payment_method") == "cash":
        is_printed = any(kw in unified_text for kw in ['小計', '合計', '対象', '税率'])
        if is_printed:
            extracted["payment_method"] = None

    # 4.7b: Fallback — department-coded items
    if not extracted.get("line_items") and extracted.get("total"):
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

    # 4.7c: Remove zero-total line items
    if extracted.get("line_items"):
        extracted["line_items"] = [
            item for item in extracted["line_items"]
            if isinstance(item, dict) and (
                item.get("total", 0) > 0 or
                (item.get("unit_price") is not None and item.get("unit_price") > 0)
            )
        ]

    # 4.7d: Handwritten receipt guard
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
                item["total"] = unit_price - (item.get("discount") or 0)

    # 4.8a: Qty from product name confusion (e.g. "集成材 10" → qty=10)
    # Detects when LLM misreads a product dimension/size as quantity
    if extracted.get("line_items"):
        for item in extracted["line_items"]:
            if not isinstance(item, dict) or item.get("qty", 1) <= 1:
                continue
            qty = item["qty"]
            total = item.get("total", 0)
            unit_price = item.get("unit_price")
            if unit_price is None or total <= 0:
                continue
            # Check if ¥total appears in OCR but ¥unit_price does NOT
            # (e.g., OCR has ¥980 but not ¥98 — the "98" was fabricated by dividing)
            total_int = str(int(total)) if total == int(total) else str(total)
            price_int = str(int(unit_price)) if unit_price == int(unit_price) else str(unit_price)
            has_yen_total = bool(re.search(r'[¥￥]\s*' + re.escape(total_int) + r'(?!\d)', unified_text))
            has_yen_price = bool(re.search(r'[¥￥]\s*' + re.escape(price_int) + r'(?!\d)', unified_text))
            if has_yen_total and not has_yen_price:
                item["qty"] = 1
                item["unit_price"] = total
                item["total"] = total - (item.get("discount") or 0)

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
            pattern_mult = r'(?:単|@)?' + re.escape(price_str) + r'\s*[×xX]\s*(\d+)\s*個?'
            pattern_ko = re.escape(price_str) + r'\s+(\d+)\s*個'
            for li, ocr_line in enumerate(ocr_lines_raw):
                if desc_prefix not in ocr_line:
                    continue
                for offset in range(0, 4):
                    if li + offset >= len(ocr_lines_raw):
                        break
                    m = re.search(pattern_mult, ocr_lines_raw[li + offset])
                    if not m:
                        m = re.search(pattern_ko, ocr_lines_raw[li + offset])
                    if m:
                        correct_qty = float(m.group(1))
                        if correct_qty != item.get("qty", 1) and correct_qty > 1:
                            item["qty"] = correct_qty
                            item["total"] = unit_price * correct_qty - (item.get("discount") or 0)
                        break
                break

    # 4.8b2: OCR qty×price scanner (matches by total when desc doesn't match)
    if extracted.get("line_items"):
        ocr_lines_fb = unified_text.split('\n')
        # Scan ALL OCR lines for qty×price patterns
        ocr_qty_prices: list[tuple[float, float, float]] = []  # (qty, unit_price, total)
        for ocr_line in ocr_lines_fb:
            found_qty_str, found_price_str = None, None
            # Pattern 1: Nコ×単P or N個×P (e.g., 2コX単328)
            m = re.search(r'(\d+)\s*[コ個]\s*[×xX]\s*(?:単|@)?\s*(\d[\d,]*)', ocr_line)
            if m:
                found_qty_str, found_price_str = m.group(1), m.group(2)
            if not found_qty_str:
                # Pattern 2: 単P×N個 (e.g., 単235×2個)
                m2 = re.search(r'(?:単|@)\s*(\d[\d,]*)\s*[×xX]\s*(\d+)\s*[コ個]', ocr_line)
                if m2:
                    found_price_str, found_qty_str = m2.group(1), m2.group(2)
            if not found_qty_str:
                # Pattern 3: ¥P N個 (e.g., ¥498 2個)
                m3 = re.search(r'[¥￥]\s*(\d[\d,]*)\s+(\d+)\s*個', ocr_line)
                if m3:
                    found_price_str, found_qty_str = m3.group(1), m3.group(2)
            if found_qty_str and found_price_str:
                ocr_qty_prices.append((
                    float(found_qty_str),
                    float(found_price_str.replace(',', '')),
                    float(found_qty_str) * float(found_price_str.replace(',', '')),
                ))
        # Match OCR patterns to items by total or unit_price
        used_indices: set[int] = set()
        for oq, op, ot in ocr_qty_prices:
            if oq <= 1:
                continue
            for idx, item in enumerate(extracted["line_items"]):
                if not isinstance(item, dict) or idx in used_indices:
                    continue
                item_total = item.get("total", 0)
                item_price = item.get("unit_price")
                matched = False
                if abs(item_total - ot) < 1:
                    matched = True
                elif item_price is not None and abs(item_price - op) < 1 and item.get("qty", 1) != oq:
                    matched = True
                if matched:
                    if item.get("qty", 1) != oq or item.get("unit_price") != op:
                        item["qty"] = oq
                        item["unit_price"] = op
                        item["total"] = op * oq - (item.get("discount") or 0)
                    used_indices.add(idx)
                    break

    # 4.8d: Fuel receipt qty normalization (volume → qty=1)
    _FUEL_KEYWORDS = ('ガソリン', 'レギュラー', 'ハイオク', '軽油', 'ENEOS', '出光', 'コスモ')
    if extracted.get("line_items"):
        for item in extracted["line_items"]:
            if not isinstance(item, dict):
                continue
            qty = item.get("qty", 1)
            if qty == int(qty):
                continue  # Integer qty, not a volume
            total = item.get("total", 0)
            desc = item.get("description", "")
            if any(kw in desc or kw in unified_text for kw in _FUEL_KEYWORDS):
                item["qty"] = 1
                item["unit_price"] = total
                break

    # 4.8c: Collapsed-item expansion
    if extracted.get("line_items") and len(extracted["line_items"]) == 1:
        item = extracted["line_items"][0]
        if isinstance(item, dict):
            qty = item.get("qty", 1)
            unit_price = item.get("unit_price")
            desc = item.get("description", "")
            if qty > 1 and unit_price is not None and desc:
                ocr_lines = unified_text.split('\n')
                ocr_desc_count = sum(
                    1 for line in ocr_lines
                    if desc in line and '小計' not in line and '合計' not in line
                )
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
            discount = (item.get("discount") or 0)
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
            discount = item.get("discount") or 0
            unit_price = item.get("unit_price")
            total = item.get("total")
            qty = item.get("qty", 1)
            if discount > 0 and unit_price is not None and total is not None:
                expected = qty * unit_price - discount
                if abs(total - unit_price * qty) < 1 and abs(total - expected) > 1:
                    item["total"] = expected

    # 4.9b2: Fix misattributed discounts — if no discount/rate is set but
    # total != qty * unit_price, the LLM likely applied a nearby discount
    # to the wrong item. Reset total to qty * unit_price.
    if extracted.get("line_items"):
        for item in extracted["line_items"]:
            if not isinstance(item, dict):
                continue
            discount = item.get("discount") or 0
            discount_rate = item.get("discount_rate") or ""
            unit_price = item.get("unit_price")
            total = item.get("total")
            qty = item.get("qty", 1)
            if discount == 0 and not discount_rate and unit_price is not None and total is not None:
                expected = qty * unit_price
                if abs(expected - total) > 1:
                    item["total"] = expected

    # 4.9c: Detect discounts from OCR text
    if extracted.get("line_items"):
        ocr_lines = unified_text.split('\n')
        for item in extracted["line_items"]:
            if not isinstance(item, dict) or (item.get("discount") or 0) > 0:
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
                            amt_match = re.match(r'^-(\d[\d,.]*)$', kline)
                            if amt_match:
                                amt_str = amt_match.group(1).replace(',', '')
                                # OCR may read comma as dot (e.g., 1.013 → 1013)
                                if '.' in amt_str and float(amt_str) < 10:
                                    amt_str = amt_str.replace('.', '')
                                discount_amount = float(amt_str)
                        if discount_amount > 0:
                            item["discount"] = discount_amount
                            item["discount_rate"] = rate_str
                            up = item.get("unit_price") or item.get("total", 0)
                            item["total"] = item.get("qty", 1) * up - discount_amount
                        break
                break

    # 4.10: Tax categories
    if extracted.get("line_items"):
        rate_bases = extract_rate_bases(unified_text)
        breakdown_bases = ocr_totals.get('_breakdown_rate_bases', {})
        for rate, base in breakdown_bases.items():
            if rate not in rate_bases or rate_bases[rate] is None:
                rate_bases[rate] = base
        assign_tax_categories(extracted["line_items"], unified_text, ocr_totals, rate_bases)

    # 4.11: Points used — gated by confidence + OCR evidence
    points = extract_points_used(unified_text)
    if points is not None:
        if should_override_field("points_used", ocr_conf, llm_conf) or extracted.get("points_used") is None:
            extracted["points_used"] = points
    elif extracted.get("points_used") is not None:
        has_points_evidence = bool(re.search(r'ポイント利用|ポイント値引', unified_text))
        if not has_points_evidence:
            extracted["points_used"] = None

    # 4.12: Fix pre-tax item totals for inclusive-tax receipts
    if extracted.get("line_items") and extracted.get("total"):
        item_sum = sum(i.get("total", 0) for i in extracted["line_items"] if isinstance(i, dict))
        receipt_total = extracted["total"]
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

        if items_fixed:
            extracted["subtotal"] = sum(
                i.get("total", 0) for i in extracted["line_items"] if isinstance(i, dict)
            )

    # 4.13: Inclusive tax subtotal fix
    if extracted.get("subtotal") and extracted.get("total") and extracted.get("taxes"):
        ocr_tax_labels = [t.get("label", "") for t in ocr_totals.get("taxes", [])]
        all_inclusive = ocr_tax_labels and all(
            (lbl or '') in ('内税', '消費税等') or (lbl or '').startswith('内')
            for lbl in ocr_tax_labels
        )
        # But not if the OCR text explicitly says 外税
        if all_inclusive and re.search(r'外税|外\d+%|税抜対象', unified_text):
            all_inclusive = False
        if all_inclusive and "subtotal" not in ocr_totals:
            tax_sum = sum(t.get("amount", 0) for t in extracted["taxes"])
            if extracted["subtotal"] and abs(extracted["subtotal"] + tax_sum - extracted["total"]) < 2:
                extracted["subtotal"] = extracted["total"]

    # 4.14: Normalize tax entries — canonical labels, clean rates, remove zero-amount
    if extracted.get("taxes"):
        subtotal = extracted.get("subtotal")
        total = extracted.get("total")
        tax_sum = sum(t.get("amount", 0) for t in extracted["taxes"])
        for t in extracted["taxes"]:
            t["rate"] = normalize_tax_rate(t.get("rate", "unknown"))
            t["label"] = normalize_tax_label(
                t.get("label"), unified_text,
                subtotal=subtotal, total=total, tax_sum=tax_sum,
            )
        # Filter out zero-amount entries for non-exempt rates
        # (keeps 0% / 非課税 entries which are meaningful tax-exempt markers)
        extracted["taxes"] = [
            t for t in extracted["taxes"]
            if t.get("amount", 0) != 0 or t.get("rate") == "0%"
        ]

    # Default subtotal = total for receipts when not found
    if extracted.get("subtotal") is None and extracted.get("total") is not None:
        extracted["subtotal"] = extracted["total"]

    return extracted
