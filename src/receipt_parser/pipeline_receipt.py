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
    items_sum: float | None = None,
) -> str:
    """Normalize a tax label to canonical set: 内税, 外税, 非課税.

    Priority order:
      1. 非課税 in label (always definitive)
      2. Explicit OCR text keywords (外税 / 内税)
      3. 対象 pattern without 外税 keyword → 内税 (most JP receipts that
         break out per-rate base in '(N%対象 …)' form are tax-inclusive)
      4. Items-sum signal: items add to total → 内税; items add to subtotal → 外税
      5. LLM-supplied label as last resort
      6. Default 内税 (most common in JP receipts)

    Under the canonical 'subtotal = total - tax' convention, the math
    'subtotal+tax=total' holds for both 内税 and 外税 — so it can't
    distinguish them. The trustworthy signals are OCR keywords and
    items_sum vs total/subtotal.
    """
    label = label or ""

    if '非課税' in label:
        return '非課税'

    # Explicit OCR keywords. "外税" and "税抜N%" / "税抜対象" are strong
    # exclusive markers; "内税" is a strong inclusive marker. Plain "税抜額"
    # (informational pre-tax total on a 内税 receipt) and "(内消費税等"
    # (informational tax breakdown on either kind) are not strong enough
    # alone — handled below.
    has_strong_exclusive = bool(re.search(r'外税|税抜\s*\d+%|税抜対象', text))
    has_strong_inclusive = bool(re.search(r'内税', text))
    # "(内消費税…" / "内消費税等" wording — appears on 内税 receipts as a
    # tax breakdown but is NOT a definitive marker (also appears on 外税
    # receipts as informational text).
    has_weak_inclusive = bool(re.search(r'内\s*消費税', text))
    # "\d+%税(額)?" not followed by a kanji is a separate-line tax amount,
    # which only appears on 外税 receipts. ("8%内税対象額" is a 内税 phrase
    # but matches `内` between % and 税, so it's excluded by \s* requirement.)
    has_pct_tax_marker = bool(re.search(r'\d+%\s*税(?:額)?(?![一-鿿])', text))

    if has_strong_exclusive and not has_strong_inclusive:
        return '外税'
    if has_strong_inclusive and not has_strong_exclusive:
        return '内税'
    if has_strong_inclusive and has_strong_exclusive:
        return '内税'
    # Strong exclusive marker via separate %-tax line
    if has_pct_tax_marker:
        return '外税'
    # Weak inclusive only fires when no strong exclusive marker is present
    if has_weak_inclusive:
        return '内税'

    # Items-sum signal: items add to total → 内税 (post-tax line items);
    # items add to subtotal → 外税 (pre-tax line items).
    if (items_sum is not None and total is not None and subtotal is not None
            and tax_sum is not None and tax_sum > 0):
        if abs(items_sum - total) <= 2 and abs(items_sum - subtotal) > 2:
            return '内税'
        if abs(items_sum - subtotal) <= 2 and abs(items_sum - total) > 2:
            return '外税'

    if label == '内税':
        return '内税'
    if '外税' in label or '税抜' in label:
        return '外税'

    return '内税'


def _parse_yen_match(m) -> float | None:
    """Extract the numeric value from a yen regex match."""
    if m is None:
        return None
    val = m.group(1) or m.group(2)
    return float(val.replace(',', '')) if val else None


_STOP_FINANCIAL = re.compile(
    r'小\s*計|現\s*計|お釣り|お釣銭|釣\s*銭|お預り|お預り金|^預$|支払い?方法|支払い?\s|現金|釣銭|クレジット'
)
_STOP_BASIC = re.compile(r'合\s*計|現\s*計|お釣り|お預り')
_STOP_TAX = re.compile(r'合\s*計|小\s*計|現\s*計|お釣り|お釣銭|釣\s*銭|お預り|お預り金')


def _collect_yen_values(
    lines: list[str], idx: int, look_ahead: int, *,
    stop_pattern: re.Pattern | None = None,
    stop_on_tax_line: bool = False,
    first_only: bool = False,
    collect_all: bool = False,
    extra_yen_pattern: bool = False,
) -> list[float]:
    """Scan nearby lines for ¥ values. Core helper for all yen extraction.

    If the current line has an inline ¥ value, returns it immediately
    unless collect_all is True (in which case inline value is included
    in the collection and scanning continues).
    """
    values: list[float] = []
    val = _parse_yen_match(YEN_INLINE.search(lines[idx].strip()))
    if val is not None:
        if collect_all:
            values.append(val)
        else:
            return [val]

    for j in range(idx + 1, min(idx + 1 + look_ahead, len(lines))):
        stripped = lines[j].strip()
        m = re.match(rf'^[¥￥]\s*([\d,]+){YEN_SUFFIX}?\s*$', stripped)
        if not m and extra_yen_pattern:
            m = re.match(rf'^[\d\s]*[¥￥]\s*([\d,]+){YEN_SUFFIX}?\s*$', stripped)
        if not m:
            m = re.match(rf'^([\d,]+)\s*円{YEN_SUFFIX}?\s*$', stripped)
        if m:
            values.append(float(m.group(1).replace(',', '')))
            if first_only:
                return values
        elif stop_pattern and stop_pattern.search(stripped):
            break
        elif stop_on_tax_line and re.search(r'\d+%', stripped) and re.search(r'対象|消費税|内税|外税|軽減', stripped) and stripped != lines[idx].strip():
            break
    return values


def _extract_yen_nearby(lines: list[str], idx: int, look_ahead: int = 2):
    """Extract first ¥ value from line idx or the next N lines."""
    vals = _collect_yen_values(lines, idx, look_ahead, first_only=True, extra_yen_pattern=True)
    return vals[0] if vals else None


def _extract_yen_max_nearby(lines: list[str], idx: int, look_ahead: int = 5):
    """Extract the largest ¥ value from nearby lines."""
    vals = _collect_yen_values(lines, idx, look_ahead, stop_pattern=_STOP_FINANCIAL)
    return max(vals) if vals else None


def _extract_all_yen_nearby(lines: list[str], idx: int, look_ahead: int = 6) -> list[float]:
    """Extract all ¥ values from nearby lines (including inline)."""
    return _collect_yen_values(lines, idx, look_ahead, stop_pattern=_STOP_BASIC, collect_all=True)


def _extract_yen_min_nearby(lines: list[str], idx: int, look_ahead: int = 3):
    """Extract the smallest ¥ value from nearby lines."""
    vals = _collect_yen_values(
        lines, idx, look_ahead,
        stop_pattern=_STOP_TAX, stop_on_tax_line=True,
    )
    return min(vals) if vals else None


def extract_financial_totals(text: str) -> dict:
    """Extract subtotal, total, and per-rate taxes directly from OCR text.

    Multi-page aware: when --- PAGE N --- markers are present, prefers
    financial totals from the last page (where receipt totals appear).
    """
    # For multi-page documents, extract from the last page only for totals
    page_marker = re.search(r'--- PAGE \d+ ---', text)
    if page_marker:
        # Find the last page marker and extract from there
        last_page_start = text.rfind('--- PAGE ')
        last_page_text = text[last_page_start:]
        # Run extraction on last page; fall back to full text if nothing found
        last_page_result = _extract_financial_totals_impl(last_page_text)
        if last_page_result.get('total') is not None:
            return last_page_result
    return _extract_financial_totals_impl(text)


def _extract_financial_totals_impl(text: str) -> dict:
    """Core implementation of financial totals extraction from OCR text."""
    lines = text.split('\n')
    result: dict = {}
    taxes: list[dict] = []
    _rate_context: str | None = None
    _rate_base: float | None = None
    # Track all per-rate taxable bases seen so far. When 消費税等 ¥X appears
    # without an inline rate, picking _rate_context (latest seen) misassigns
    # the tax to the wrong rate on receipts that print 8% rows then 10% rows
    # before the summary tax. Use bases × rate_pct to pick the correct rate.
    _rate_bases_seen: dict[str, float] = {}

    for i, raw in enumerate(lines):
        line = raw.strip()

        rate_ctx_m = re.search(r'(\d+(?:\.\d+)?)%.*対象', line)
        if rate_ctx_m:
            _rate_context = normalize_tax_rate(rate_ctx_m.group(1) + '%')
            base_val = None
            yen_in_line = re.search(r'[¥￥]\s*([\d,]+)', line)
            if yen_in_line:
                try:
                    base_val = float(yen_in_line.group(1).replace(',', ''))
                except ValueError:
                    pass
            else:
                # Yen amount may be on the next line (column-split OCR layout)
                for j in range(i + 1, min(i + 4, len(lines))):
                    nb = lines[j].strip()
                    yen_next = re.match(r'^[¥￥]\s*([\d,]+)\s*[\)）]?\s*$', nb)
                    if yen_next:
                        try:
                            base_val = float(yen_next.group(1).replace(',', ''))
                        except ValueError:
                            pass
                        break
                    if nb and re.search(r'[　-鿿]', nb):
                        break
            if base_val is not None:
                _rate_bases_seen[_rate_context] = base_val

        rate_amt_m = re.match(r'^(\d+(?:\.\d+)?)\s*%\s*[¥￥]\s*([\d,]+)\s*$', line)
        if rate_amt_m and not _rate_context:
            _rate_context = normalize_tax_rate(rate_amt_m.group(1) + '%')
            _rate_base = float(rate_amt_m.group(2).replace(',', ''))

        _has_specific_taxes = any(t.get('label') in ('税額', '外税', '内税') for t in taxes)
        # Match 消費税 with optional 等/額 suffix. Receipts that print
        # "(内消費税 ¥151)" inline (McDonald's, some chain stores) need the
        # bare-消費税 case; '対象' guard still excludes label lines.
        if re.search(r'消費税[等額]?', line) and '対象' not in line and not _has_specific_taxes:
            inline_rate_m = re.search(r'(\d+(?:\.\d+)?)\s*%', line)
            if inline_rate_m:
                effective_rate = normalize_tax_rate(inline_rate_m.group(1) + '%')
            else:
                effective_rate = _rate_context
                # If multiple rate bases have been seen, pick the rate whose
                # base × rate_pct best matches the tax amount. This corrects
                # the case where the latest rate context belongs to a small
                # 0-tax row (e.g. 10%対象 ¥3 where tax rounds to 0) but the
                # actual tax (¥258) belongs to the earlier 8% base.
                if len(_rate_bases_seen) >= 2:
                    val_lookahead = _extract_yen_min_nearby(lines, i, look_ahead=5)
                    if val_lookahead is not None:
                        best_rate = None
                        best_err = float('inf')
                        for rate, base in _rate_bases_seen.items():
                            try:
                                pct = float(rate.rstrip('%')) / 100.0
                            except ValueError:
                                continue
                            err = abs(base * pct - val_lookahead)
                            if err < best_err:
                                best_err = err
                                best_rate = rate
                        if best_rate and best_err <= max(2.0, val_lookahead * 0.05):
                            effective_rate = best_rate
            if effective_rate:
                val = _extract_yen_min_nearby(lines, i, look_ahead=5)
                if val is not None:
                    taxes.append({'rate': effective_rate, 'label': '消費税等', 'amount': val})
            _rate_context = None

        # Inline 内税/外税 with rate context: "10%対象 ¥19,118 内消費税 ¥1,738".
        # Captures 内/外 prefix on 消費税 paired with a ¥amount on the same line.
        if not _has_specific_taxes:
            m_combo = re.search(
                r'(\d+(?:\.\d+)?)\s*%\s*対象.*?(内|外)消費税[等額]?\s*[¥￥]\s*([\d,]+)',
                line,
            )
            if m_combo:
                rate_combo = normalize_tax_rate(m_combo.group(1) + '%')
                label_combo = '内税' if m_combo.group(2) == '内' else '外税'
                amt_combo = float(m_combo.group(3).replace(',', ''))
                taxes.append({'rate': rate_combo, 'label': label_combo, 'amount': amt_combo})
                _has_specific_taxes = True

        if (re.search(r'小\s*計', line) or 'お買上高' in line) and '税' not in line:
            val = _extract_yen_nearby(lines, i)
            if val is not None:
                result['subtotal'] = val
                all_nearby = _extract_all_yen_nearby(lines, i, look_ahead=6)
                alts = [v for v in all_nearby if v != val]
                if alts:
                    result['_subtotal_alt'] = max(alts)
                    result['_subtotal_candidates'] = all_nearby
        elif re.search(r'小\s*計\s*\(?税抜', line):
            val = _extract_yen_nearby(lines, i)
            if val is not None:
                result.setdefault('_per_rate_subtotals', []).append(val)

        is_total_line = re.search(r'合\s*計', line)
        if not is_total_line and re.match(r'^計$', line) and i > 0:
            prev_context = ' '.join(l.strip() for l in lines[max(0, i - 3):i])
            # Reject ポイント対象金額 (loyalty-points subtotal) etc. — those
            # have 対象 in the predecessor context but are NOT 合計.
            if 'ポイント' in prev_context:
                pass
            elif '合' in prev_context or '税' in prev_context or '対象' in prev_context:
                is_total_line = True
        if is_total_line and not re.search(r'税\s*合\s*計', line) and '対象' not in line and 'お預' not in line:
            val_max = _extract_yen_max_nearby(lines, i, look_ahead=5)
            val_first = _extract_yen_nearby(lines, i, look_ahead=3)
            if val_max is not None:
                result['total'] = val_max
            if val_first is not None and val_first != val_max:
                result['total_first'] = val_first

        if '現計' in line:
            val = _extract_yen_max_nearby(lines, i, look_ahead=7)
            if val is not None and val > result.get('total', 0):
                result['total'] = val

        if '現金支払' in line:
            val = _extract_yen_nearby(lines, i)
            if val is not None and 'total' not in result:
                result['total'] = val

        # Inline 内税N%消費税 pattern (gas-station receipts):
        # "(内税10%消費税" or "(内税分消費税" sits in a label block with the
        # ¥amount in a separate value block below. The LLM gets the rate
        # but mis-extracts the amount, so use OCR-derived value to override.
        m_inclusive_with_rate = re.search(
            r'\(?\s*内税(?:分|\s*\d+\s*%)?\s*消費税', line,
        )
        if m_inclusive_with_rate and '対象' not in line and not _has_specific_taxes:
            rate_search = re.search(r'(\d+(?:\.\d+)?)\s*%', line)
            inclusive_rate = (
                normalize_tax_rate(rate_search.group(1) + '%') if rate_search else None
            )
            val_inc = _extract_yen_nearby(lines, i)
            # Walk forward past labels to collect all ¥amounts in the value
            # block. Pair-matching (V, B) where V ≈ B × rate/(1+rate) picks
            # the right tax value even when お預り inflates the magnitude.
            if val_inc is None:
                forward_values: list[float] = []
                for j in range(i + 1, min(i + 22, len(lines))):
                    js = lines[j].strip()
                    if not js:
                        continue
                    yen_m = re.match(r'^[¥￥]\s*([\d,]+)\s*[\)）]?\s*$', js)
                    if yen_m:
                        try:
                            forward_values.append(float(yen_m.group(1).replace(',', '')))
                        except ValueError:
                            pass
                        continue
                    if re.match(r'^@?\d+(?:[.,]\d+)?[Ll円]?\s*$', js):
                        continue
                    if re.search(r'[　-鿿]', js):
                        if re.search(r'内税|外税|対象|消費税|合計|小計|現計|現金|お預り|お釣|単価|数量|@|ガソリン税|料金所|残高|支払', js):
                            continue
                        break
                if forward_values:
                    rate_pct = (float(inclusive_rate.rstrip('%')) / 100.0
                                if inclusive_rate else 0.10)
                    if rate_pct > 0:
                        # Find tax value V such that V = B × rate/(1+rate) for
                        # some base B also in forward_values. Tax is much
                        # smaller than its base, so iterate small-to-large.
                        best = None
                        best_err = float('inf')
                        for v in sorted(set(forward_values)):
                            if v <= 0:
                                continue
                            implied_base = v * (1 + rate_pct) / rate_pct
                            for b in forward_values:
                                err = abs(b - implied_base)
                                if err < best_err and err <= max(2.0, implied_base * 0.01):
                                    best_err = err
                                    best = v
                                    break
                        if best is not None:
                            val_inc = best
            if inclusive_rate and val_inc is not None:
                taxes.append({'rate': inclusive_rate, 'label': '内税', 'amount': val_inc})

        # Plain 内税 with inline ¥ value (e.g. "内税 ※ ¥187"). Requires a
        # previously-set rate context AND a single rate in play — when the
        # receipt has multiple rate contexts, the inline value is typically
        # the SUM of per-rate taxes (e.g. "(内税 ¥780)" where 780 = 632+148),
        # not a per-rate tax. Triggers after rejoin_totals_label_value_columns
        # has interleaved labels with their position-paired values.
        m_inclusive_plain = re.match(
            r'\(?\s*内税\s*[※\*]?\s*[¥￥]\s*([\d,]+)\s*[\)）]?\s*$', line
        )
        if (m_inclusive_plain and not _has_specific_taxes
                and _rate_context and len(_rate_bases_seen) <= 1):
            # Require a rate base so we can sanity-check the value. Without a
            # base we can't tell whether this is the per-rate tax or a sum.
            # Fall back to _rate_base (single var set by rate_amt_m) when
            # _rate_bases_seen has no entry — both track rate-context state.
            rate_base = _rate_bases_seen.get(_rate_context)
            if rate_base is None and _rate_base is not None and len(_rate_bases_seen) == 0:
                rate_base = _rate_base
            if rate_base is not None:
                try:
                    val_inc_plain = float(m_inclusive_plain.group(1).replace(',', ''))
                except ValueError:
                    val_inc_plain = None
                if val_inc_plain is not None and val_inc_plain > 0:
                    try:
                        rate_pct = float(_rate_context.rstrip('%')) / 100.0
                    except ValueError:
                        rate_pct = 0
                    expected_inc = rate_base * rate_pct / (1 + rate_pct)
                    expected_excl = rate_base * rate_pct
                    plausible = (
                        abs(val_inc_plain - expected_inc) <= max(2, expected_inc * 0.10)
                        or abs(val_inc_plain - expected_excl) <= max(2, expected_excl * 0.10)
                    )
                    if plausible:
                        taxes.append({'rate': _rate_context, 'label': '内税',
                                      'amount': val_inc_plain})

        if re.search(r'外税\s*\d+%', line) and '対象' not in line:
            rate_m = re.search(r'(\d+)%', line)
            val = _extract_yen_nearby(lines, i)
            # Column-split layout: amount can appear above the label block.
            # Sibling labels (外税N%, 対象, 小計, 合計) sit between the current
            # label and the value block above; skip them when scanning back.
            if val is None and rate_m:
                back_values: list[float] = []
                _SIBLING_TAX_LABEL_RE = re.compile(
                    r'^(?:[\(（]?[ab]?)?\s*外税|^(?:[\(（]?[ab]?)?\s*内税|'
                    r'対象(?:額)?|^合\s*計$|^小\s*計$|^\d+\s*%\s*対象|^[\(（]?\s*\d+\s*%'
                )
                for back in range(i - 1, max(i - 14, -1), -1):
                    bs = lines[back].strip()
                    back_m = re.match(r'^[¥￥]\s*([\d,]+)\s*[)）]?\s*$', bs)
                    if back_m:
                        back_values.append(float(back_m.group(1).replace(',', '')))
                        continue
                    if not bs:
                        continue
                    # Skip sibling labels — they belong to the same tax block
                    if _SIBLING_TAX_LABEL_RE.search(bs):
                        continue
                    # Stop on non-financial content (item descriptions, etc.)
                    if re.search(r'[　-鿿]', bs):
                        break
                if back_values:
                    rate_pct = float(rate_m.group(1)) / 100.0
                    if rate_pct > 0:
                        largest = max(back_values)
                        expected = largest * rate_pct
                        if expected > 0:
                            val = min(back_values, key=lambda v: abs(v - expected))
                    if val is None:
                        non_zero = [v for v in back_values if v > 0]
                        if non_zero:
                            val = min(non_zero)
            if rate_m and val is not None:
                taxes.append({'rate': rate_m.group(1) + '%', 'label': '外税', 'amount': val})

        if '税額' in line and '対象' not in line:
            rate_m = re.search(r'(\d+)%', line)
            val = _extract_yen_min_nearby(lines, i, look_ahead=3)
            # OCR may place price lines ABOVE the label (column-split reading).
            # Scan backward through the price block to find a plausible tax amount.
            if val is None and rate_m:
                back_values: list[float] = []
                for back in range(i - 1, max(i - 6, -1), -1):
                    bs = lines[back].strip()
                    back_m = re.match(r'^[¥￥]\s*([\d,]+)\s*[)）]?\s*$', bs)
                    if back_m:
                        back_values.append(float(back_m.group(1).replace(',', '')))
                    elif bs and re.search(r'[\u3000-\u9fff]', bs):
                        break
                if back_values:
                    # Tax amount is typically the smallest value that's > 0 and
                    # plausibly a percentage of the largest (the taxable base).
                    rate_pct = float(rate_m.group(1)) / 100.0
                    largest = max(back_values)
                    expected = largest * rate_pct
                    # Pick the value closest to expected, or just the smallest
                    # if we can't compute expected
                    if expected > 0:
                        val = min(back_values, key=lambda v: abs(v - expected))
                    else:
                        val = min(back_values)
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

        if re.match(r'^内税[\s※]*$', line) and '対象' not in line:
            val = _extract_yen_nearby(lines, i)
            if not taxes:
                rate = _rate_context or 'unknown'
                if rate == 'unknown':
                    for j in range(max(0, i - 2), min(i + 4, len(lines))):
                        if j == i:
                            continue
                        nearby_rate_m = re.search(r'(\d+(?:\.\d+)?)%', lines[j].strip())
                        if nearby_rate_m:
                            rate = normalize_tax_rate(nearby_rate_m.group(1) + '%')
                            break
                if val is None and _rate_base is not None and rate != 'unknown':
                    rate_pct = float(rate.replace('%', '')) / 100.0
                    val = round(_rate_base * rate_pct / (1 + rate_pct))
                if val is not None:
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

    # Sum per-rate subtotals when present (e.g., "小計(税抜8%)" + "小計(税抜10%)")
    per_rate_subs = result.pop('_per_rate_subtotals', None)
    if per_rate_subs and 'subtotal' not in result:
        result['subtotal'] = sum(per_rate_subs)

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

    # Dedup: a receipt may print the same tax in multiple places (e.g., a 外税
    # receipt that also shows the inclusive breakdown in parens, "(内消費税等
    # 8% ¥203)"). Both lines match our extraction patterns, producing duplicate
    # entries that downstream code then sums into bogus 2x tax totals.
    if taxes:
        seen: set[tuple] = set()
        deduped = []
        for t in taxes:
            key = (t.get('rate'), t.get('label'), t.get('amount'))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(t)
        taxes = deduped

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
        if re.search(r'対象商品|対象です|対象物', line):
            continue

        rate_num = float(m.group(1))
        rate_str = f"{int(rate_num)}%" if rate_num == int(rate_num) else f"{rate_num}%"

        yen_m = re.search(r'[¥￥]\s*([\d,]+)', line)
        if yen_m:
            bases[rate_str] = float(yen_m.group(1).replace(',', ''))
        else:
            found = False
            plain_candidate = None
            for j in range(i + 1, min(i + 6, len(lines))):
                js = lines[j].strip()
                if not js:
                    continue
                yen_ahead = re.search(r'[¥￥]\s*([\d,]+)', js)
                if yen_ahead:
                    bases[rate_str] = float(yen_ahead.group(1).replace(',', ''))
                    found = True
                    break
                if re.match(r'^\d*\s*[\u500b\u70b9]\s*$', js):
                    # Item-count fragment ("3\u500b", "\u500b", "9\u70b9") \u2014 keep scanning
                    continue
                if re.search(r'[\u3000-\u9fff]', js):
                    break
                if plain_candidate is None:
                    plain_m = re.match(r'^([\d,]+)\s*$', js)
                    if plain_m:
                        plain_candidate = float(plain_m.group(1).replace(',', ''))
            if not found and plain_candidate is not None:
                bases[rate_str] = plain_candidate
            elif not found:
                bases[rate_str] = None

    return bases


def extract_points_used(text: str) -> float | None:
    """Extract loyalty points applied as payment from OCR text."""
    patterns = [
        r'ポイント利用\s*[¥￥]?\s*([\d,]+)',
        r'利用ポイント\s*[¥￥]?\s*([\d,]+)',
        r'ポイント値引\s*-?\s*([\d,]+)',
        r'ポイント\s*-\s*([\d,]+)',
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return float(m.group(1).replace(',', ''))
    # Detect zero-point usage: "利用ポイント" header with "OP"/"0P" value nearby
    if re.search(r'利用ポイント|ポイント利用', text):
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if '利用ポイント' in line or 'ポイント利用' in line:
                for j in range(i, min(i + 5, len(lines))):
                    if re.match(r'^\s*[0O]\s*P\s*$', lines[j]):
                        return 0.0
                break
    return None


def _find_subset_sum(items, target, max_k=3, tolerance=5.0):
    # Prefer near-exact (≤2) matches at small k (k ≤ 2) before accepting a
    # fuzzy match. Stops a loose 1-item match from shadowing a real 2-item
    # exact match (e.g. fuzzy {138}≈131 vs. exact {3, 128}=131).
    # Restricted to k ≤ 2: a k=3 exact can be a coincidence (any wrong target
    # near a sum is exactly hit by some 3-tuple), so we don't let large-k exact
    # override a smaller-k fuzzy candidate.
    if tolerance > 2:
        for k in range(1, min(3, max_k + 1, len(items) + 1)):
            for combo in combinations(items, k):
                total = sum(t for _, t in combo)
                if abs(total - target) <= 2:
                    return [i for i, _ in combo]
    for k in range(1, min(max_k + 1, len(items) + 1)):
        for combo in combinations(items, k):
            total = sum(t for _, t in combo)
            if abs(total - target) <= tolerance:
                return [i for i, _ in combo]
    return None


def assign_tax_categories(items, unified_text, ocr_totals, rate_bases, extracted_taxes=None):
    """Assign tax_category to line items using OCR evidence. Mutates in-place."""
    if not items:
        return

    valid_rates = set(VALID_TAX_RATES) - {"0%"}
    detected_rates: set[str] = set()
    for tax in ocr_totals.get("taxes", []):
        rate = tax.get("rate", "")
        if rate in valid_rates:
            detected_rates.add(rate)
    # Fallback: use LLM-extracted taxes when OCR extraction missed them
    if extracted_taxes:
        for tax in extracted_taxes:
            rate = tax.get("rate", "") if isinstance(tax, dict) else ""
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
    # Catch "消費税 N%" or "内消費税 N%" patterns (e.g., "内消費税 10.00%")
    for m in re.finditer(r'消費税\s*(\d+(?:\.\d+)?)\s*%', unified_text):
        r = str(int(float(m.group(1)))) + "%"
        if r in valid_rates:
            detected_rates.add(r)

    # Remove rates whose OCR base is explicitly zero (no items at that rate)
    for rate in list(detected_rates):
        if rate_bases.get(rate) == 0:
            detected_rates.discard(rate)

    if not detected_rates:
        has_nontaxable = bool(re.search(r'非課税|不課税|免税', unified_text))
        if has_nontaxable:
            for item in items:
                item["tax_category"] = "0%"
        else:
            for item in items:
                if item.get("tax_category") == "0%":
                    desc = item.get("description", "")
                    if not re.match(r'^部門\s*\d', desc):
                        item["tax_category"] = STANDARD_RATE
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
        for li, line in enumerate(ocr_lines):
            if desc_prefix not in line:
                continue
            # Column-split OCR puts the price+marker on the very next line
            # ("たまご三昧" \n "278*"). Same-line check first (most receipts
            # interleave price and marker with the description). If neither
            # matches, peek the immediate next non-empty line — but stop if
            # that line itself starts with another product description (avoid
            # bleeding the next item's marker into this one).
            tax_marker = None
            if '除' in line:
                tax_marker = STANDARD_RATE
            elif re.search(r'[※\*軽]|(?<![A-Za-z])X(?![A-Za-z])', line):
                tax_marker = REDUCED_RATE
            if tax_marker is None and li + 1 < len(ocr_lines):
                nxt = ocr_lines[li + 1].strip()
                # Only peek when the next line is a price-with-marker pattern
                # (digits + tax marker glyph), not a new item description.
                if nxt and re.match(
                    r'^[\d,]+\s*[※\*軽除AB]?\s*$|^[\d,]+\s*[¥￥]?[\d,]*\s*[※\*軽除]\s*$',
                    nxt,
                ):
                    if '除' in nxt:
                        tax_marker = STANDARD_RATE
                    elif re.search(r'[※\*軽]|(?<![A-Za-z])X(?![A-Za-z])', nxt):
                        tax_marker = REDUCED_RATE
            if tax_marker is not None:
                item_rates[idx] = tax_marker
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
    # Merge in LLM-extracted taxes for any rates the OCR pass missed (column-
    # split layouts often hide one of the per-rate tax lines from the OCR scan
    # while the LLM still recovers it).
    if extracted_taxes:
        for t in extracted_taxes:
            if not isinstance(t, dict):
                continue
            r = t.get("rate", "")
            if r and r not in tax_amounts:
                tax_amounts[r] = t.get("amount", 0)
    # Choose the dominant rate. When most items have OCR tax markers,
    # the marked counts are reliable. When markers are sparse (e.g. only
    # 1 of 18 items has a 除 tag), counts mislead — fall back to
    # rate_bases (sum of items per rate from OCR), which reflects the
    # actual transaction proportions regardless of how many items got
    # tagged.
    marked_total = sum(assigned_counts.values())
    if marked_total >= len(items) * 0.5:
        majority_rate = max(
            detected_rates,
            key=lambda r: (assigned_counts.get(r, 0), tax_amounts.get(r, 0), rate_bases.get(r, 0) or 0),
        )
    else:
        majority_rate = max(
            detected_rates,
            key=lambda r: (rate_bases.get(r, 0) or 0, tax_amounts.get(r, 0), assigned_counts.get(r, 0)),
        )
    minority_rates = [r for r in detected_rates if r != majority_rate]
    minority_rate = minority_rates[0] if minority_rates else None

    # Some receipts print rate_base as the tax-INCLUSIVE amount (pre_tax + tax)
    # rather than the pre-tax base. Subset-sum operates on item totals, which
    # may themselves be pre-tax (items_sum == subtotal) or inclusive (items_sum
    # == total). When items are pre-tax but the printed rate_base is inclusive
    # we need to subtract the tax to recover the right subset-sum target.
    items_sum_total = 0.0
    item_count = 0
    for it in items:
        if isinstance(it, dict):
            try:
                items_sum_total += float(it.get("total") or 0)
                item_count += 1
            except (TypeError, ValueError):
                pass

    rate_bases = dict(rate_bases)  # local copy — don't mutate caller's dict
    sum_rate_bases = sum(v for v in rate_bases.values() if v is not None)
    sum_taxes = sum(tax_amounts.values()) if tax_amounts else 0
    # "items_sum is pre-tax" signal: items_sum + sum_of_taxes ≈ sum_of_rate_bases
    # (rate_bases printed as inclusive). For receipts where items are inclusive
    # already, items_sum ≈ sum_of_rate_bases without adding tax — no adjustment.
    items_are_pretax = (
        item_count > 0 and sum_rate_bases > 0 and sum_taxes > 0
        and abs(items_sum_total + sum_taxes - sum_rate_bases) < max(5, sum_rate_bases * 0.02)
    )
    if items_are_pretax:
        for rate in list(rate_bases):
            base = rate_bases.get(rate)
            tax = tax_amounts.get(rate)
            if base is None or not tax or base <= 0:
                continue
            try:
                rate_pct = float(rate.rstrip('%')) / 100.0
            except ValueError:
                continue
            if rate_pct <= 0:
                continue
            err_pretax = abs(base * rate_pct - tax)
            err_inclusive = abs((base - tax) * rate_pct - tax)
            if err_inclusive + 0.5 < err_pretax and base > tax:
                rate_bases[rate] = base - tax

    subset_matched = False
    if minority_rate and unassigned:
        unassigned_items = [(i, items[i].get("total", 0)) for i in unassigned]
        marked_sums_for_match: dict[str, float] = {}
        for idx, rate in item_rates.items():
            marked_sums_for_match[rate] = marked_sums_for_match.get(rate, 0) + items[idx].get("total", 0)
        for try_rate in [minority_rate, majority_rate]:
            full_base = rate_bases.get(try_rate)
            if full_base is None:
                continue
            other_rate = minority_rate if try_rate == majority_rate else majority_rate
            full_other = rate_bases.get(other_rate)
            try_base = full_base - marked_sums_for_match.get(try_rate, 0)
            other_base = (full_other - marked_sums_for_match.get(other_rate, 0)) if full_other is not None else None
            if try_base < 0:
                continue
            sub_max_k = min(len(unassigned_items), 5)
            match = _find_subset_sum(unassigned_items, try_base, max_k=sub_max_k, tolerance=50.0)
            if match is not None and other_base is not None and len(unassigned_items) > 3:
                # Score candidates by (target_err, complement_err) lex tuple — an
                # exact target hit (e ≤ 2) wins over a fuzzy 2-element match even
                # if the complement drifts. The unassigned set may be slightly off
                # from base+other_base because of upstream OCR/LLM noise; in that
                # case complement error is irreducible noise and shouldn't gate
                # whether we accept an exact target match.
                best_e = abs(sum(t for i, t in unassigned_items if i in match) - try_base)
                best_ce = abs(sum(t for i, t in unassigned_items if i not in match) - other_base)
                best_score = (best_e, best_ce)
                # Start at k=3: the inner _find_subset_sum returns at the first
                # k=2 fuzzy match, so a k=3 exact match (e≈0) is never reached.
                # The extension is the only path that lets a higher-k candidate
                # beat a smaller-k fuzzy hit.
                for ext_k in range(3, min(len(unassigned_items), 7)):
                    for combo in combinations(unassigned_items, ext_k):
                        s = sum(t for _, t in combo)
                        e = abs(s - try_base)
                        if e > 50:
                            continue
                        c_indices = {i for i, _ in combo}
                        cs = sum(t for i, t in unassigned_items if i not in c_indices)
                        ce = abs(cs - other_base)
                        score = (e, ce)
                        if score < best_score:
                            best_score = score
                            match = [i for i, _ in combo]
                            if e == 0 and ce == 0:
                                break
                    if best_score == (0, 0):
                        break
            if match is not None:
                other_rate = minority_rate if try_rate == majority_rate else majority_rate
                subset_matched = True
                for i in match:
                    item_rates[i] = try_rate
                for i in unassigned:
                    if i not in item_rates:
                        item_rates[i] = other_rate
                break

        # Fallback: if rate_bases didn't work, compute expected bases from
        # tax amounts and marked item sums. Tax amount / rate = pre-tax base.
        # Subtract already-marked items to get what unassigned items should sum to.
        if not subset_matched and tax_amounts:
            marked_sums: dict[str, float] = {}
            for idx, rate in item_rates.items():
                marked_sums[rate] = marked_sums.get(rate, 0) + items[idx].get("total", 0)
            for try_rate in [minority_rate, majority_rate]:
                tax_amt = tax_amounts.get(try_rate)
                if not tax_amt:
                    continue
                rate_pct = float(try_rate.replace('%', '')) / 100.0
                if rate_pct <= 0:
                    continue
                already_marked = marked_sums.get(try_rate, 0)
                # Try interpreting as tax amount first, then as base amount
                match = None
                for candidate_base in [tax_amt / rate_pct, tax_amt]:
                    needed = candidate_base - already_marked
                    if needed < 0:
                        continue
                    max_k = min(len(unassigned_items), 5)
                    match = _find_subset_sum(unassigned_items, needed, max_k=max_k, tolerance=50.0)
                    if match is not None:
                        break
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
        if tax_amounts and max(tax_amounts.values()) > 0:
            default_rate = max(detected_rates, key=lambda r: tax_amounts.get(r, 0))
        elif REDUCED_RATE in marker_rates and STANDARD_RATE not in marker_rates:
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


_COMPANY_SUFFIX_RE = re.compile(r'有限会社|株式会社|㈱|㈲|合同会社')
_DECORATIVE_RE = re.compile(r'^[☆★\-=\*\s・♪♫]+$')


def _fix_company_name_merchant(extracted, unified_text):
    """Prefer venue/event name over legal company name when LLM picks the latter."""
    merchant = extracted.get("merchant")
    if not merchant:
        return
    lines = unified_text.split('\n')
    found_in_any_line = False
    found_only_in_company_line = True
    for line in lines:
        if merchant in line:
            found_in_any_line = True
            if not _COMPANY_SUFFIX_RE.search(line):
                found_only_in_company_line = False
                break
    if not found_in_any_line or not found_only_in_company_line:
        return
    for line in lines:
        line = line.strip()
        if not line or _DECORATIVE_RE.match(line) or _COMPANY_SUFFIX_RE.search(line):
            continue
        if line != merchant and len(line) >= 2:
            extracted["merchant"] = line
            break


def _apply_financial_overrides(extracted, ocr_totals, ocr_conf, llm_conf):
    """Override LLM financial values (total, subtotal, taxes) with OCR-extracted values."""
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
            extracted["subtotal"] = ocr_sub_val
    if "total" in ocr_totals and should_override_field("total", ocr_conf, llm_conf):
        ocr_total = float(ocr_totals["total"])
        ocr_first = float(ocr_totals["total_first"]) if ocr_totals.get("total_first") is not None else None
        ocr_sub = float(ocr_totals["subtotal"]) if ocr_totals.get("subtotal") is not None else None
        if ocr_sub and ocr_total < ocr_sub:
            pass
        elif ocr_sub and ocr_total > ocr_sub * 2:
            if ocr_first and ocr_first <= ocr_sub * 1.15:
                extracted["total"] = ocr_first
        else:
            extracted["total"] = ocr_total
    elif "total" in ocr_totals and extracted.get("total") is None:
        extracted["total"] = float(ocr_totals["total"])
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
        # Merge: trust OCR for rates it found, but keep LLM's tax entries for
        # rates the OCR scan missed (column-split layouts often hide one rate's
        # tax line from the OCR forward-scan while the LLM still recovers it).
        ocr_rates = {t.get("rate") for t in ocr_totals["taxes"]}
        llm_extra = [
            t for t in (extracted.get("taxes") or [])
            if isinstance(t, dict) and t.get("rate") and t.get("rate") not in ocr_rates
        ]
        extracted["taxes"] = list(ocr_totals["taxes"]) + llm_extra

    # Fix per-rate subtotal: when subtotal + tax != total, recompute from total - tax
    if "subtotal" in ocr_totals and "total" in ocr_totals and ocr_totals.get("taxes"):
        ocr_tax_sum = sum(t.get("amount", 0) for t in ocr_totals["taxes"])
        ocr_sub = ocr_totals["subtotal"]
        ocr_tot = ocr_totals["total"]
        if ocr_tax_sum > 0 and abs(ocr_sub + ocr_tax_sum - ocr_tot) > 2:
            computed_sub = ocr_tot - ocr_tax_sum
            if abs(computed_sub + ocr_tax_sum - ocr_tot) < 2:
                extracted["subtotal"] = computed_sub
                ocr_totals["subtotal"] = computed_sub


def _fix_date(extracted, unified_text):
    """Extract and fix dates from OCR text (supports 令和/平成 eras)."""
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
        return

    era_named = re.search(r'(令和|平成)\s*(\d{1,2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日', unified_text)
    if era_named:
        w_year = era_to_western_year(int(era_named.group(2)), era_named.group(1))
        if w_year:
            extracted["date"] = f"{w_year:04d}-{int(era_named.group(3)):02d}-{int(era_named.group(4)):02d}"
        return

    era = re.search(r'(?<!\d)(\d{1,2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日', unified_text)
    if era:
        era_name = None
        for name in ERA_TABLE:
            if name in unified_text:
                era_name = name
                break
        era_year_val = int(era.group(1))
        # 2-digit western year abbreviation (e.g. "26年" = 2026)
        # Prefer this when no era name is present and 20XX is plausible
        if not era_name and 20 <= era_year_val <= 99:
            western_candidate = 2000 + era_year_val
            if 2020 <= western_candidate <= 2030:
                extracted["date"] = f"{western_candidate:04d}-{int(era.group(2)):02d}-{int(era.group(3)):02d}"
                return
        w_year = era_to_western_year(era_year_val, era_name)
        if w_year and 1989 <= w_year <= 2100:
            extracted["date"] = f"{w_year:04d}-{int(era.group(2)):02d}-{int(era.group(3)):02d}"


_DATE_LINE_RE = re.compile(
    r'(?:'
    r'20\d{2}\s*[年/-]\s*0?\d{1,2}\s*[月/-]\s*0?\d{1,2}\s*日?'
    r'|'
    r'(?:令和|平成)?\s*\d{1,2}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日'
    r')'
)
_TIME_HHMM_RE = re.compile(r'(?<!\d)([0-2]?\d)\s*[:：]\s*([0-5]\d)(?:\s*[:：]\s*[0-5]\d)?(?!\d)')
_TIME_JP_RE = re.compile(r'(?<!\d)([0-2]?\d)\s*時\s*([0-5]\d)\s*分?')
_BUSINESS_HOURS_RE = re.compile(r'営業時間|営業中|定休|OPEN|CLOSE|TEL|電話|☎')


def _parse_time_from_segment(segment: str) -> str | None:
    """Find the first valid HH:MM (or HH時MM分) in a text segment, or None."""
    for pattern in (_TIME_HHMM_RE, _TIME_JP_RE):
        for m in pattern.finditer(segment):
            hh, mm = int(m.group(1)), int(m.group(2))
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                return f"{hh}:{mm:02d}"
    return None


def _fix_time(extracted, unified_text):
    """Extract receipt transaction time from OCR text, anchored to the date line.

    Scans the date line + next 2 lines for HH:MM or HH時MM分.
    Skips lines that look like business hours / phone numbers to avoid false positives.

    Sets extracted['time'] only when:
      - The LLM didn't already produce a valid time, OR
      - The OCR-anchored time disagrees with the LLM and we have strong evidence.
    Leaves extracted['time'] as None if no time appears on the receipt.
    """
    lines = unified_text.split('\n')

    candidate: str | None = None
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not _DATE_LINE_RE.search(line):
            continue
        # Skip the part of the date line that contains the date itself,
        # so we don't match digits inside e.g. "2026年03月04日" or "12月03日"
        date_match = _DATE_LINE_RE.search(line)
        tail = line[date_match.end():] if date_match else line

        # Look in the date-line tail and the next two lines (but not into
        # business-hours context).
        segments = [tail]
        for j in range(i + 1, min(i + 3, len(lines))):
            nxt = lines[j].strip()
            if _BUSINESS_HOURS_RE.search(nxt):
                break
            segments.append(nxt)

        for seg in segments:
            if _BUSINESS_HOURS_RE.search(seg):
                continue
            t = _parse_time_from_segment(seg)
            if t:
                candidate = t
                break
        if candidate:
            break

    if candidate is None:
        # Fallback: ISO date already in extracted['date'] but OCR may have it
        # joined together as "2025/12/23/13:49" — pull time off the join.
        joined = re.search(r'20\d{2}[/-]\d{1,2}[/-]\d{1,2}[/\s-]+(\d{1,2})[:：](\d{2})', unified_text)
        if joined:
            hh, mm = int(joined.group(1)), int(joined.group(2))
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                candidate = f"{hh}:{mm:02d}"

    # Fallback: search entire OCR for HH時MM分 without digit lookbehind
    if candidate is None:
        matches = list(re.finditer(r'(\d{1,2})時(\d{2})分', unified_text))
        if len(matches) == 1:
            hh, mm = int(matches[0].group(1)), int(matches[0].group(2))
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                candidate = f"{hh}:{mm:02d}"

    existing = extracted.get("time")
    if not existing:
        if candidate:
            extracted["time"] = candidate
        return

    if candidate and candidate != existing:
        extracted["time"] = candidate


def _fix_payment_method(extracted, unified_text, ocr_conf, llm_conf):
    """Detect cash payment from OCR evidence (tendered amount, change, etc.)."""
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
    has_cash_keyword = bool(re.search(r'現金\s*[¥￥]\s*[\d,]+', unified_text))
    if not has_cash_keyword and re.search(r'(?:^|\n)\s*現金\s*(?:\n|$)', unified_text):
        has_cash_keyword = True
    strong_cash = has_cash and (
        (has_tender and has_change_label and change_amount != 0) or has_cash_keyword
    )

    if has_cash:
        existing = extracted.get("payment_method")
        if strong_cash:
            extracted["payment_method"] = "cash"
        elif not existing or existing == "cash":
            extracted["payment_method"] = "cash"
        elif should_override_field("payment_method", ocr_conf, llm_conf) and not existing:
            extracted["payment_method"] = "cash"
    elif extracted.get("payment_method") == "cash":
        # Hallucinated cash with no OCR evidence — strip it. The original
        # guard only fires for printed receipts (has 小計/合計) and missed
        # handwritten 領収証 where the LLM defaults to "cash" because of
        # the receipt-header phrasing (上記正に領収いたしました).
        extracted["payment_method"] = None


_FUEL_KEYWORDS = ('ガソリン', 'レギュラー', 'ハイオク', '軽油', 'ENEOS', '出光', 'コスモ')


def _fix_zero_prices_from_ocr(items, unified_text):
    """For items with zero price, recover the price from OCR text."""
    lines = unified_text.split('\n')
    for item in items:
        if not isinstance(item, dict):
            continue
        total = item.get("total", 0) or 0
        unit_price = item.get("unit_price", 0) or 0
        if total > 0 or unit_price > 0:
            continue
        desc = (item.get("description") or "").strip()
        if not desc or len(desc) < 2:
            continue
        desc_prefix = desc[:5] if len(desc) >= 5 else desc
        for idx, line in enumerate(lines):
            if desc_prefix not in line:
                continue
            yen_m = re.search(r'[¥￥]\s*([\d,]+)', line)
            if not yen_m:
                yen_m = re.search(r'([\d,]+)\s*[※*非内]', line)
            if yen_m:
                price = float(yen_m.group(1).replace(',', ''))
                if price > 0:
                    item["unit_price"] = price
                    item["total"] = price * item.get("qty", 1)
                    break
            for j in range(idx + 1, min(idx + 3, len(lines))):
                m = re.match(
                    r'^\s*[¥￥]\s*([\d,]+)\s*$|^\s*([\d,]+)\s*[※*非内]\s*$',
                    lines[j].strip(),
                )
                if m:
                    price = float((m.group(1) or m.group(2)).replace(',', ''))
                    if price > 0:
                        item["unit_price"] = price
                        item["total"] = price * item.get("qty", 1)
                        break
            break


_SKIP_PRICE_LINE = re.compile(r'対象|内税|外税|合計|小計|消費税|お預り|お釣|お預かり')

# Descriptions that are clearly NOT product names — generic category markers
# (used in HANDS-style receipts above the actual item) or contact info.
_GENERIC_DESC_MARKERS = frozenset({
    '消耗', '食料品', '飲料', '雑貨', '文具', '日配', '冷蔵', '冷凍',
    '青果', '惣菜', '加工', '生活', '化粧品', '医薬', 'お菓子', '酒類',
    '日用品', '特', '軽',
})

_JUNK_DESC_RE = re.compile(
    r'^(?:電話|TEL|☎)\s*[:：]?\s*0?\d'  # phone numbers ('電話: 078-...')
    r'|^〒\s*\d{3}'                       # postal code
    r'|^\d{8,}'                            # bare digit run (barcode)
    r'|^\d+\s*[xX×]\s*[#＃]?\s*\d+$'      # unit-rate notation '23 X #199'
    r'|^\*?\s*\d[\d,]*\s*\(\s*\d+\s*[個コ点]\s*\)'  # '*770 (1コ)' price+qty notation
)

# Lines that look like receipt-header metadata (phone, address, date,
# register/cashier numbers) — never use as a product description.
_HEADER_LINE_RE = re.compile(
    r'(?:電話|TEL|☎)|登録番号|担当|レシートNo|レジ\s*\d|キャッシャ|'
    r'店舗|発行|取引(?:コード|No)|POS\s*No|〒\s*\d{3}|'
    r'(?:\d{2,4}\s*)?年\s*\d{1,2}\s*月\s*\d{1,2}\s*日|'  # date (year optional)
    r'\(\s*[月火水木金土日]\s*\)|'                          # day-of-week marker
    r'\d{1,2}\s*[:時]\s*\d{2}\s*分?(?!\d)|'                # HH:MM or HH時MM分
    r'\d+\s*番\s*\d+\s*号|'
    r'[県府][　-鿿]+[市区町村]|'                            # address
    r'^[一-鿿]{2,8}店$|'                                      # branch name
    r'\bNo\.?\s*\d{2,}|'                                      # No. 012
    r'領\s*収\s*証'                                           # 領収証
)

# Generic Japanese receipt boilerplate banners — appear on receipts from many
# merchants but never as product names. Used to drop phantom items the LLM
# created from header/footer text adjacent to a stray number.
_BANNER_PHRASE_RE = re.compile(
    r'ぜひ当店でお買物くださいませ|'
    r'ありがとうございました|ありがとうございます|'
    r'毎度ありがとうございます|'
    r'毎月\s*\d+\s*日.*感謝デ[ーー]|'
    r'お客さま感謝デ[ーー]|'
    r'印は軽減税率|軽減税率\s*8?\s*%?\s*対象商品|'
    r'お買上商品数|お買上点数|お買上げ点数|'
    r'ポイントの有効期限|累計ポイント|'
    r'今回獲得|現在のポイント|'
    r'本人確認(?:省略)?|'
    r'クレジットカード売上票|お客様控え?|'
    r'当店をご利用|またのご利用|またお越し|'
    r'お問い合わせ|営業時間|定休日|'
    r'カードお取扱日|取引内容|伝票番号|承認番号|'
    r'プロの品質とプロの価格|'
    r'の商品です|まとめ値引|'
    r'^[A-Z]\s*[:：]\s*\d+\s*[個コ点]|'
    r'^\s*消費税等?\s*$'
)


def _fix_junk_descriptions(items, unified_text):
    """Replace 'junk' item descriptions (category markers, phone numbers,
    barcode digits) with the nearest product-like line above the price in OCR.

    Generic-purpose: any item whose description is on the marker list or
    matches a junk pattern, regardless of receipt source.
    """
    lines = unified_text.split('\n')

    for item in items:
        if not isinstance(item, dict):
            continue
        desc = (item.get("description") or "").strip()
        total = item.get("total", 0)
        if not total or total <= 0:
            continue

        # Mixed-script OCR fragments with very few Japanese chars and an
        # ASCII word separated by whitespace (e.g. "TV くりえ") are usually
        # garbage. Brand-prefix product names like "KAL紙袋M" or "S&W..."
        # have ASCII directly adjacent to Japanese (no space) — those are
        # valid and must not be flagged.
        japanese_chars = re.findall(r'[ぁ-んァ-ン一-龥]', desc)
        ascii_chars = re.findall(r'[A-Za-z]', desc)
        has_separating_space = bool(re.search(
            r'[A-Za-z]\s+[ぁ-んァ-ン一-龥]|[ぁ-んァ-ン一-龥]\s+[A-Za-z]', desc
        ))
        is_short_mixed_garbage = (
            japanese_chars and ascii_chars
            and len(japanese_chars) < 5
            and len(desc) < 9
            and has_separating_space
        )

        # Unit-price × quantity notation like "単235×2個" — never a product name
        is_unit_price_notation = bool(
            re.match(r'^単?\s*\d', desc)
            and ('×' in desc or 'x' in desc or 'X' in desc)
            and ('個' in desc or '点' in desc or 'コ' in desc)
        )

        # Length-based junk: only flag empty / 1-char, or short non-Japanese
        # strings. Pure-Japanese 2-char descs like "部品", "牛肉" are valid.
        is_pure_japanese = bool(desc) and bool(re.fullmatch(r'[ぁ-んァ-ンー一-龥]+', desc))
        is_short_junk = len(desc) < 2 or (len(desc) == 2 and not is_pure_japanese)

        is_junk = (
            desc in _GENERIC_DESC_MARKERS
            or is_short_junk
            or _JUNK_DESC_RE.search(desc) is not None
            or _HEADER_LINE_RE.search(desc) is not None
            or is_short_mixed_garbage
            or is_unit_price_notation
        )
        if not is_junk:
            continue

        # Find the OCR line containing this item's price
        price_line_idx = None
        for i, line in enumerate(lines):
            if _SKIP_PRICE_LINE.search(line):
                continue
            for m in re.finditer(r'[¥￥]\s*([\d,]+)', line):
                val_str = m.group(1)
                if not val_str:
                    continue
                try:
                    price = float(val_str.replace(',', ''))
                except ValueError:
                    continue
                if abs(price - total) < 1:
                    price_line_idx = i
                    break
            if price_line_idx is not None:
                break

        if price_line_idx is None:
            continue

        # Build a list of nearby line indices ordered by proximity to the
        # price line — start with the price line itself (rejoin_price_lines
        # often merges item name + price on one line, e.g. "KAL紙袋M ¥30"),
        # then alternate below/above. Range ±15 handles receipts with
        # garbled OCR between the description and price.
        candidates_idx: list[int] = [price_line_idx]
        for offset in range(1, 16):
            for direction in (1, -1):
                j = price_line_idx + direction * offset
                if 0 <= j < len(lines):
                    candidates_idx.append(j)

        def _process_candidate(cand_raw: str) -> str | None:
            m_yen = re.search(r'[¥￥]', cand_raw)
            cand = cand_raw[:m_yen.start()].strip() if m_yen else cand_raw
            cand = re.sub(r'\s+[\d,]+\s*[点個コ]\s*$', '', cand).strip()
            cand = re.sub(r'\s*[※\*非外]\s*$', '', cand).strip()
            # Strip leading product/department code if remainder has Japanese
            m_code = re.match(r'^\d{4,}[A-Za-z]{0,3}\)?\s?(.+)$', cand)
            if m_code and re.search(r'[ぁ-んァ-ン一-龥]', m_code.group(1)):
                cand = m_code.group(1).strip()
            if not cand or len(cand) <= 2:
                return None
            if cand in _GENERIC_DESC_MARKERS:
                return None
            if _SKIP_PRICE_LINE.search(cand):
                return None
            if re.match(r'^[\d,\s\-\(\)\.\*※軽除]+$', cand):
                return None
            if _JUNK_DESC_RE.search(cand):
                return None
            if _HEADER_LINE_RE.search(cand):
                return None
            # Reject unit-price × qty notation (e.g. "単235×2個") — this is
            # itself a junk pattern when picked from OCR as a description.
            if (re.match(r'^単?\s*\d', cand)
                    and ('×' in cand or 'x' in cand or 'X' in cand)
                    and ('個' in cand or '点' in cand or 'コ' in cand)):
                return None
            if not re.search(r'[ぁ-んァ-ン一-龥]', cand):
                return None
            jp = re.findall(r'[ぁ-んァ-ン一-龥]', cand)
            asc = re.findall(r'[A-Za-z]', cand)
            cand_has_separating_space = bool(re.search(
                r'[A-Za-z]\s+[ぁ-んァ-ン一-龥]|[ぁ-んァ-ン一-龥]\s+[A-Za-z]', cand
            ))
            if (jp and asc and len(jp) < 5 and len(cand) < 9
                    and cand_has_separating_space):
                return None
            if any(
                isinstance(o, dict) and o is not item
                and (o.get("description") or "").strip() == cand
                for o in items
            ):
                return None
            return cand

        # First pass: prefer lines with a product-code prefix (raw check).
        chosen = None
        for j in candidates_idx:
            raw = lines[j].strip()
            if not re.match(r'^\d{4,}', raw):
                continue
            cand = _process_candidate(raw)
            if cand:
                chosen = cand
                break
        # Second pass: any valid candidate
        if not chosen:
            for j in candidates_idx:
                cand = _process_candidate(lines[j].strip())
                if cand:
                    chosen = cand
                    break
        if chosen:
            item["description"] = chosen


def _fix_item_desc_from_ocr_price_line(items, unified_text):
    """Fix item descriptions when LLM picked up non-item text (e.g. promotional banners)."""
    lines = unified_text.split('\n')
    for item in items:
        if not isinstance(item, dict):
            continue
        desc = (item.get("description") or "").strip()
        total = item.get("total", 0)
        if not desc or not total or total <= 0:
            continue

        desc_lines = [i for i, line in enumerate(lines) if desc in line]

        # Multi-line desc fallback: long descriptions sometimes split across
        # consecutive OCR lines (e.g., 'どっさりキャベツと白身フライ' →
        # 'どっさりキャベツと白' + '身フライ'). Treat as "found" if any
        # contiguous sequence of OCR lines together contains the desc.
        if not desc_lines and len(desc) >= 6:
            for i in range(len(lines) - 1):
                joined = lines[i].strip() + lines[i + 1].strip()
                if desc in joined:
                    desc_lines = [i, i + 1]
                    break
                if i + 2 < len(lines):
                    joined3 = joined + lines[i + 2].strip()
                    if desc in joined3:
                        desc_lines = [i, i + 1, i + 2]
                        break

        # If desc literally appears in OCR AND there's a bare-digit total on
        # an immediately adjacent line, trust the LLM and skip replacement.
        # The bare-digit total isn't picked up by the marker/¥-prefix patterns
        # below, so without this check we'd replace correct descs whose price
        # is in column-format (price on next line, no markers).
        if desc_lines:
            has_adjacent_price = False
            for dl in desc_lines:
                for adj in (dl + 1, dl - 1, dl + 2):
                    if 0 <= adj < len(lines):
                        adj_text = lines[adj].strip()
                        if re.fullmatch(r'[\d,]+', adj_text):
                            try:
                                if abs(float(adj_text.replace(',', '')) - total) < 1:
                                    has_adjacent_price = True
                                    break
                            except ValueError:
                                pass
                if has_adjacent_price:
                    break
            if has_adjacent_price:
                continue

        # Collect ALL OCR price lines that match this item's total. There
        # may be multiple at the same price (e.g., 3 items all priced 350).
        # When there are multiple, the desc must be far from EVERY one for us
        # to consider replacement; if it's near any, the LLM's desc is plausible.
        price_matches: list[tuple[int, str]] = []  # (line_idx, candidate_desc)
        for pattern in (r'([\d,]+)\s*[非※*]', r'[¥￥]\s*([\d,]+)'):
            for i, line in enumerate(lines):
                if _SKIP_PRICE_LINE.search(line):
                    continue
                for m in re.finditer(pattern, line):
                    val_str = m.group(1)
                    if val_str:
                        price = float(val_str.replace(',', ''))
                        if abs(price - total) < 1:
                            text_before = line[:m.start()].strip()
                            text_before = re.sub(r'\s*[※\*非]\s*$', '', text_before).strip()
                            if (text_before and len(text_before) >= 2
                                    and not re.match(r'^[¥￥\d,.\s]+$', text_before)):
                                price_matches.append((i, text_before))
                            else:
                                price_matches.append((i, ""))
                            break
            if price_matches:
                break

        if not price_matches:
            continue

        # Pick the price match whose candidate description is non-empty AND
        # not a generic marker AND not a banner phrase. If desc is near ANY
        # price match, keep current.
        viable = [(idx, cand) for idx, cand in price_matches
                  if cand and cand not in _GENERIC_DESC_MARKERS
                  and not _BANNER_PHRASE_RE.search(cand)
                  and not _HEADER_LINE_RE.search(cand)]
        if not viable:
            continue

        # If LLM's current desc is near any price line for this total,
        # trust it — the LLM picked the right row, even if its desc spans
        # multiple OCR lines or differs from the inline candidate.
        near_any_price = any(abs(dl - pidx) <= 3
                             for dl in desc_lines
                             for pidx, _ in price_matches)
        if near_any_price:
            continue

        price_line_idx, price_desc = viable[0]
        if price_desc != desc:
            item["description"] = price_desc


def _fix_line_items(extracted, unified_text):
    """Fix line item quantities, prices, and discounts using OCR evidence."""
    # Fallback: department-coded items
    if not extracted.get("line_items") and extracted.get("total"):
        dept_m = re.search(r'部門\s*(\d+)\s*', unified_text)
        if dept_m:
            extracted["line_items"] = [{
                "description": f"部門{dept_m.group(1).strip()}",
                "qty": 1, "unit_price": extracted["total"],
                "total": extracted["total"], "tax_category": "0%",
                "discount": 0, "discount_rate": "",
            }]

    # Fallback: single-service receipt (toll, parking, single-item)
    _AMOUNT_LABELS_RE = re.compile(
        r'^(金額|合計|小計|税込|税抜|総額|請求額|お会計|お預り|釣銭|No\.?|様)$'
    )
    if not extracted.get("line_items") and extracted.get("total"):
        total = extracted["total"]
        for m in re.finditer(r'[¥￥]\s*([\d,]+)', unified_text):
            price = int(m.group(1).replace(',', ''))
            if abs(price - total) < 1:
                pos = m.start()
                before = unified_text[:pos].rstrip()
                lines_before = before.split('\n')
                desc = lines_before[-1].strip() if lines_before else ""
                if (desc and len(desc) >= 2
                        and not re.match(r'^[\d,¥￥\s\-]+$', desc)
                        and not _AMOUNT_LABELS_RE.match(desc)):
                    extracted["line_items"] = [{
                        "description": desc,
                        "qty": 1, "unit_price": total,
                        "total": total, "tax_category": "10%",
                        "discount": None, "discount_rate": None,
                    }]
                    break

    # Remove zero-total items and single-char noise descriptions
    if extracted.get("line_items"):
        extracted["line_items"] = [
            item for item in extracted["line_items"]
            if isinstance(item, dict) and (
                item.get("total", 0) > 0 or
                (item.get("unit_price") is not None and item.get("unit_price") > 0)
            ) and len((item.get("description") or "").strip()) > 1
        ]

    # Handwritten receipt guard: remove single line item that just duplicates total
    # Keep items with distinct descriptions (e.g. "通行料金" for toll receipts).
    # Also drop the item if the description is an LLM-confabulated fragment —
    # a date, disclaimer text, or anything that isn't a recognizable service
    # term. Handwritten 領収証 lacking item lists per template rule should
    # produce line_items=[].
    is_handwritten = not any(kw in unified_text for kw in ['小計', '合計', '対象', '税率'])
    if is_handwritten and extracted.get("line_items") and extracted.get("total"):
        items = extracted["line_items"]
        if len(items) == 1 and isinstance(items[0], dict):
            if abs(items[0].get("total", 0) - extracted["total"]) < 1:
                desc = (items[0].get("description") or "").strip()
                merchant = (extracted.get("merchant") or "").strip()
                _DISCLAIMER_FRAGMENTS = ('含み', '但し', '消費税', '領収', '印紙', '収入')
                _SERVICE_TERMS = ('通行料金', 'ガソリン', 'レギュラー', 'ハイオク', '軽油',
                                  '駐車', '入場料', '料金', '施術', '診療')
                desc_is_disclaimer = any(kw in desc for kw in _DISCLAIMER_FRAGMENTS)
                desc_looks_like_date = bool(re.match(
                    r'^\s*(?:20\d{2}|令和|平成)?\s*\d+\s*年', desc
                ))
                desc_is_service = any(kw in desc for kw in _SERVICE_TERMS)
                if (not desc or desc == merchant or desc_is_disclaimer
                        or (desc_looks_like_date and not desc_is_service)):
                    extracted["line_items"] = []

    if not extracted.get("line_items"):
        return

    _drop_banner_phantom_items(extracted["line_items"], unified_text)
    _fix_item_desc_from_ocr_price_line(extracted["line_items"], unified_text)
    _merge_qty_detail_into_previous(extracted["line_items"], unified_text)
    _fix_junk_descriptions(extracted["line_items"], unified_text)
    _strip_embedded_price_in_desc(extracted["line_items"])
    _remove_unit_rate_phantom_items(extracted)
    _fix_qty_hallucinations(extracted["line_items"], unified_text)
    _replace_duplicate_desc_from_ocr(extracted["line_items"], unified_text)
    _fix_item_totals_from_ocr_neighborhood(
        extracted["line_items"], unified_text,
        extracted.get("subtotal"), extracted.get("total"),
    )
    _repair_column_split_items(
        extracted["line_items"], unified_text,
        extracted.get("subtotal"), extracted.get("total"),
    )
    _replace_hallucinated_dup_with_ocr_item(
        extracted["line_items"], unified_text,
        extracted.get("subtotal"), extracted.get("total"),
    )
    _apply_qty_notation_from_ocr(extracted["line_items"], unified_text)
    _revert_unsupported_qty_inflation(extracted["line_items"], unified_text)
    _dedup_same_total_items(extracted)
    _fix_qty_from_ocr_patterns(extracted["line_items"], unified_text)
    _fix_fuel_volume_qty(extracted["line_items"], unified_text,
                         receipt_total=extracted.get("total") or extracted.get("subtotal"))
    _expand_collapsed_items(extracted, unified_text)
    _fix_hallucinated_prices(extracted["line_items"], unified_text)
    _fix_zero_prices_from_ocr(extracted["line_items"], unified_text)
    _fix_discount_totals(extracted["line_items"])
    _fix_misattributed_discounts(extracted["line_items"])
    _detect_ocr_discounts(extracted["line_items"], unified_text)
    _project_totals_to_ocr_multiset(extracted, unified_text)


# Matches qty-detail OCR fragments like "(2個 X 単70)", "2個 X70)", "(@100 × 2個)".
# These are not products — they are qty/unit-price annotations for the
# preceding item, but the LLM sometimes extracts them as standalone items.
_QTY_DETAIL_DESC_RE = re.compile(
    r'^\(?\s*'
    r'(?:'
    r'\d+\s*[コ個点]\s*[xX×]\s*(?:単|@)?\s*\d[\d,]*'   # "2個 X70", "2個 X 単70"
    r'|(?:単|@)\s*\d[\d,]*\s*[xX×]\s*\d+\s*[コ個点]'    # "単70 × 2個", "@70x2個"
    r')'
    r'\)?\s*$'
)


def _merge_qty_detail_into_previous(items, unified_text):
    """Collapse qty-detail phantom items into the preceding product.

    When the LLM extracts a qty-detail OCR fragment (e.g. "(2個 X 単70)") as
    a standalone item, the receipt's preceding product is actually priced
    at qty × unit. Use the OCR text (not the LLM's possibly-wrong qty) to
    extract qty/unit, apply them to the previous item, then drop the
    phantom.

    Safety: only merges when (a) the phantom's description matches the
    qty-detail regex AND the OCR text near a qty-detail fragment yields a
    consistent (qty, unit) pair (qty ≥ 2, unit > 0); and (b) a previous
    item exists with qty == 1.
    """
    if len(items) < 2:
        return
    ocr_lines = unified_text.split('\n')
    to_drop: set[int] = set()
    for i, item in enumerate(items):
        if i == 0 or not isinstance(item, dict):
            continue
        desc = (item.get("description") or "").strip()
        if not desc or not _QTY_DETAIL_DESC_RE.match(desc):
            continue
        # Extract qty/unit from OCR (more reliable than the LLM's parse).
        ocr_qty: float | None = None
        ocr_unit: float | None = None
        for ocr_line in ocr_lines:
            if not _QTY_DETAIL_DESC_RE.match(ocr_line.strip()):
                continue
            m = re.search(
                r'(\d+)\s*[コ個点]\s*[xX×]\s*(?:単|@)?\s*(\d[\d,]*)',
                ocr_line,
            )
            if not m:
                m = re.search(
                    r'(?:単|@)\s*(\d[\d,]*)\s*[xX×]\s*(\d+)\s*[コ個点]',
                    ocr_line,
                )
                if m:
                    ocr_unit = float(m.group(1).replace(',', ''))
                    ocr_qty = float(m.group(2))
                    break
            else:
                ocr_qty = float(m.group(1))
                ocr_unit = float(m.group(2).replace(',', ''))
                break
        if ocr_qty is None or ocr_unit is None or ocr_qty < 2 or ocr_unit <= 0:
            continue
        prev = items[i - 1]
        if not isinstance(prev, dict):
            continue
        if prev.get("qty", 1) and float(prev.get("qty", 1)) > 1:
            continue
        prev["qty"] = ocr_qty
        prev["unit_price"] = ocr_unit
        prev["total"] = ocr_qty * ocr_unit
        to_drop.add(i)
    if to_drop:
        items[:] = [it for j, it in enumerate(items) if j not in to_drop]


def _fix_item_totals_from_ocr_neighborhood(items, unified_text, target_subtotal, target_total):
    """When items_sum is off-target, re-anchor each item's total to the price
    immediately following its description in OCR text.

    Generic-purpose: handles 2-column receipts where rejoin_price_lines didn't
    fully resolve, so the LLM mis-attributes prices across adjacent items.
    Conservative — only fires when:
      - items_sum is off both subtotal and total by > 2 yen
      - The OCR shows a clear desc → price chain (no other Japanese line between)
      - The OCR-grounded price differs from the LLM total by > 1 yen
      - Applying the fix brings items_sum strictly closer to a target
    """
    if not items:
        return
    items_sum = sum(i.get("total", 0) for i in items if isinstance(i, dict))
    targets = [t for t in (target_subtotal, target_total) if t]
    if not targets:
        return
    if any(abs(items_sum - t) <= 2 for t in targets):
        return

    lines = unified_text.split('\n')

    def _ocr_price_after(li: int) -> float | None:
        # Look for a clean ¥-bearing or plain numeric line within next 6 lines.
        # Stop on another item-like line (Japanese text, no ¥).
        for j in range(li + 1, min(li + 7, len(lines))):
            s = lines[j].strip()
            if not s:
                continue
            if _SKIP_PRICE_LINE.search(s):
                return None
            m = re.match(r'^[¥￥]?\s*([\d,]+)\s*[※\*除]?\s*$', s)
            if m:
                try:
                    return float(m.group(1).replace(',', ''))
                except ValueError:
                    return None
            if re.search(r'[ぁ-んァ-ン一-龥]{2,}', s):
                return None  # next item starts before any price
        return None

    def _ocr_price_inline(line: str) -> float | None:
        # ¥-prefixed first
        m = re.search(r'[¥￥]\s*([\d,]+)', line)
        if m:
            try:
                return float(m.group(1).replace(',', ''))
            except ValueError:
                return None
        # Trailing bare-digit price with tax marker (e.g., "...  640X" or
        # "... 228*"). Only the LAST trailing digit + marker on the line —
        # mid-line digits may be part of the description (e.g., "TV1.0テイシボ").
        m = re.search(r'\s+([\d,]{2,7})\s*[※\*X除軽]\s*$', line)
        if m:
            try:
                return float(m.group(1).replace(',', ''))
            except ValueError:
                return None
        return None

    # Apply candidate fixes one at a time, verifying each improves items_sum
    # toward a target. Stop when items_sum is within 2 yen of a target.
    progress = True
    while progress:
        progress = False
        items_sum = sum(i.get("total", 0) for i in items if isinstance(i, dict))
        if any(abs(items_sum - t) <= 2 for t in targets):
            break
        candidates: list[tuple[float, int, float]] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            desc = (item.get("description") or "").strip()
            total = item.get("total")
            if not desc or len(desc) < 5 or total is None:
                continue
            desc_prefix = desc[:5]
            for li, line in enumerate(lines):
                if desc_prefix not in line:
                    continue
                ocr_total = _ocr_price_inline(line)
                if ocr_total is None:
                    ocr_total = _ocr_price_after(li)
                if ocr_total is None:
                    continue
                if abs(ocr_total - total) <= 1:
                    break  # already aligned
                # Score this candidate by the improvement it brings
                new_sum = items_sum - total + ocr_total
                cur_diff = min(abs(items_sum - t) for t in targets)
                new_diff = min(abs(new_sum - t) for t in targets)
                if new_diff < cur_diff:
                    candidates.append((cur_diff - new_diff, idx, ocr_total))
                break  # first matching OCR line for this item
        if not candidates:
            break
        candidates.sort(reverse=True)  # largest improvement first
        improvement, idx, new_total = candidates[0]
        items[idx]["total"] = new_total
        if items[idx].get("qty", 1) == 1 and items[idx].get("unit_price") is not None:
            items[idx]["unit_price"] = new_total
        progress = True


def _repair_column_split_items(items, unified_text, target_subtotal, target_total):
    """Re-pair LLM items to OCR prices when the OCR is column-split.

    Column-split layout: a run of name-only lines (Japanese, no ¥), then a
    run of price-only lines (¥-prefixed or bare digits). The LLM matches by
    proximity, which fails when sub-runs are unequal or qty annotations
    break the price block.

    Strategy:
      1. Walk OCR up to (and optionally past) the 小計/合計 zone end. Skip
         qty notations, discount lines, and inline-priced names (those
         self-pair). Collect remaining name and price tokens in OCR order.
      2. If global counts of names == prices in the chain, position-pair
         them: name[i] → price[i] for the chain's full length.
      3. For each LLM item, if its description prefix appears in the paired
         dict and the override moves items_sum toward a target, apply.

    Inline-priced detection: a Japanese line ending with " <digits>[marker]"
    is treated as inline-priced even without a ¥ symbol (handles AEON-style
    "食品ポリ袋L (バイオマス30 3除" and "千切りキャベツビッグパ 238").

    Zone extension: when names without paired prices remain at the 小計
    boundary, extend past 小計 to capture stray bare-digit prices that
    appear before the first ¥-prefixed totals line. (Handles AEON layouts
    where the right-column item prices land below 小計 in OCR order.)

    Conservative — only fires when items_sum is off-target by > 2 yen and
    the override strictly reduces the error.
    """
    if not items:
        return
    items_sum = sum(i.get("total", 0) for i in items if isinstance(i, dict))
    targets = [t for t in (target_subtotal, target_total) if t]
    if not targets or any(abs(items_sum - t) <= 2 for t in targets):
        return

    lines = unified_text.split('\n')

    # Find zone-end at the first totals/tax line.
    end_idx = len(lines)
    for i, raw in enumerate(lines):
        s = raw.strip()
        if re.search(r'^(小\s*計|合\s*計|外税|内税|消費税|お預り|現計|お釣り|釣銭|総額)', s):
            end_idx = i
            break

    # Anchor the item zone to where LLM items actually appear in OCR. This
    # filters out header NAMEs (campaign text, store info, register #, etc.)
    # whose presence would break the equal-count check.
    item_descs = [
        (it.get("description") or "").strip()
        for it in items if isinstance(it, dict)
    ]
    item_descs = [d for d in item_descs if d and len(d) >= 2]
    if not item_descs:
        return

    zone_start: int | None = None
    zone_last_item: int = 0
    for li in range(min(end_idx, len(lines))):
        line = lines[li]
        for d in item_descs:
            if d[:5] in line or (len(d) >= 3 and d[:3] in line and re.search(r'[ぁ-んァ-ン一-龥]', d[:3])):
                if zone_start is None:
                    zone_start = li
                zone_last_item = li
                break

    if zone_start is None:
        return
    # Items zone runs from zone_start (first OCR match of any LLM item) to
    # end_idx (the 小計/合計 line). No slack — the first matched line is
    # the earliest item, anything before it is header noise.
    item_zone_end = end_idx

    # Permissive price-line: digits + optional 1-2 trailing marker chars.
    # Captures `198`, `378+`, `98%`, `265X`, `228*`, `78 A`, `¥1,498`, `1,074`.
    # Rejects post-item footer noise like `10P)` or `54P` (P is not a marker).
    _PRICE_MARKER_CLASS = r'[*※軽除＊・X+%A_]'
    _PRICE_ONLY_RE = re.compile(
        r'^[¥￥]?\s*(\d[\d,]{0,5})\s*' + _PRICE_MARKER_CLASS + r'?\s*' + _PRICE_MARKER_CLASS + r'?\s*$'
    )
    _PRICE_HAS_MARKER_RE = re.compile(r'[*※軽除＊・X+%A]')
    _QTY_NOTATION_RE = re.compile(
        r'[\(（<]?\s*\d+\s*[コ個点]\s*[xX×]\s*(?:単|@)?\s*\d[\d,]*\s*[\)）>]?'
    )
    # Inline price tail: " <digits>[marker]" at end of a name line.
    _INLINE_PRICE_TAIL_RE = re.compile(
        r'\s+(\d[\d,]{0,5})\s*[*※軽除＊・X+%]?\s*$'
    )
    # Lines that look like a date or phone — not inline-priced.
    _DATE_LIKE_RE = re.compile(r'\d{4}[/年-]\d|\d{2}[:時]\d')
    _PHONE_LIKE_RE = re.compile(r'\d{2,4}-\d{2,4}-\d{3,4}')

    def _parse_price(s: str):
        m = _PRICE_ONLY_RE.match(s)
        if not m:
            return None
        try:
            v = float(m.group(1).replace(',', ''))
        except ValueError:
            return None
        if v < 1 or v > 999999:
            return None
        return v

    def _is_inline_priced(s: str) -> bool:
        if not re.search(r'[ぁ-んァ-ン一-龥]', s):
            return False
        if _DATE_LIKE_RE.search(s) or _PHONE_LIKE_RE.search(s):
            return False
        if re.search(r'[¥￥]\s*\d', s):
            return True
        m = _INLINE_PRICE_TAIL_RE.search(s)
        if not m:
            return False
        # Avoid product codes like "L30" mid-string by requiring the digit
        # group be at the end with whitespace before. _INLINE_PRICE_TAIL_RE
        # already enforces \s+ before the digit group, so a trailing lone
        # number after Japanese chars is the signal.
        return True

    def _is_name_line(s: str) -> bool:
        if not s or len(s) < 2:
            return False
        if re.search(r'[¥￥]', s):
            return False
        if not re.search(r'[ぁ-んァ-ン一-龥]', s):
            return False
        if _QTY_NOTATION_RE.search(s):
            return False
        if re.match(r'^割引', s) or s.startswith('-'):
            return False
        # Skip post-item footer markers (point-tracking, account info, etc.)
        # that share OCR space with the item zone.
        if re.search(
            r'(ポイント|残高|累計|獲得|有効期限|内訳|お買上|今回|商品数|'
            r'WAON|^内\s|^取\s*\d|レジ\s*\d|登録番号|TEL|FAX|http)', s
        ):
            return False
        return True

    # Walk the zone, building name and price token streams. Skip qty
    # notations, discount lines, and inline-priced names (self-paired).
    # Discount-rate lines like "20%" or "-18" stand alone; "98%" is a
    # price-with-marker (the % is OCR noise for *), not a discount rate.
    # Distinguish: discount lines appear right after a 割引 or item name
    # without an intervening price; we treat any standalone digit + %
    # within ~2 lines of a 割引 marker as a discount rate, otherwise as
    # a price-with-marker.
    discount_rate_lines: set[int] = set()
    for i in range(zone_start, item_zone_end):
        s = lines[i].strip()
        if re.match(r'^割引', s):
            # Look ahead for a digit% line within the next 3 lines.
            for j in range(i + 1, min(i + 4, item_zone_end)):
                t = lines[j].strip()
                if re.match(r'^-?\d{1,3}\s*[%％]\s*$', t):
                    discount_rate_lines.add(j)
                    break

    # Walk the zone, building OCR-ordered (name, price) pairs.
    # Inline-priced lines emit a pair directly. Pure-name and pure-price
    # tokens are stitched into chains; chains where len(names)==len(prices)
    # contribute pairs by position.
    ordered_pairs: list[tuple[str, float]] = []
    pending_names: list[str] = []
    pending_prices: list[float] = []

    def _flush_chain():
        nonlocal pending_names, pending_prices
        if pending_names and pending_prices and len(pending_names) == len(pending_prices):
            for n, p in zip(pending_names, pending_prices):
                ordered_pairs.append((n, p))
        pending_names = []
        pending_prices = []

    def _flush_chain_with_extension(extension_prices: list[float]):
        """Try to complete an unfinished chain by appending extension prices."""
        nonlocal pending_names, pending_prices
        needed = len(pending_names) - len(pending_prices)
        if needed > 0 and len(extension_prices) >= needed:
            pending_prices.extend(extension_prices[:needed])
        if pending_names and pending_prices and len(pending_names) == len(pending_prices):
            for n, p in zip(pending_names, pending_prices):
                ordered_pairs.append((n, p))
        pending_names = []
        pending_prices = []

    # Helper: detect a partial qty notation fragment like "(2個 X" (OCR
    # split this onto two lines so the trailing digits are missing).
    _PARTIAL_QTY_RE = re.compile(r'^[\(（<]?\s*\d+\s*[コ個点]\s*[xX×]?\s*$')

    for i in range(zone_start, item_zone_end):
        s = lines[i].strip()
        if not s:
            continue
        if _QTY_NOTATION_RE.search(s) or _PARTIAL_QTY_RE.match(s):
            continue
        if re.match(r'^割引', s) or i in discount_rate_lines or re.match(r'^-\d', s):
            continue
        # Inline-priced line — emit pair directly, flush any pending chain.
        if _is_inline_priced(s):
            # Extract name part and price.
            m_yen = re.search(r'[¥￥]\s*([\d,]+)', s)
            if m_yen:
                pv_str = m_yen.group(1)
                price_pos = m_yen.start()
            else:
                m_tail = _INLINE_PRICE_TAIL_RE.search(s)
                if not m_tail:
                    continue
                pv_str = m_tail.group(1)
                price_pos = m_tail.start()
            try:
                pv = float(pv_str.replace(',', ''))
            except ValueError:
                continue
            name_part = s[:price_pos].strip()
            if not name_part or not re.search(r'[ぁ-んァ-ン一-龥]', name_part):
                continue
            _flush_chain()
            ordered_pairs.append((name_part, pv))
            continue
        v = _parse_price(s)
        if v is not None:
            pending_prices.append(v)
            continue
        if _is_name_line(s):
            # Names after prices indicate a new chain — flush.
            if pending_prices and len(pending_names) == len(pending_prices):
                _flush_chain()
            pending_names.append(s)

    # Zone extension: scan past 小計 for stray bare-digit prices (no ¥) that
    # may complete an unfinished column-split chain. AEON layouts often print
    # the right-column item prices below 小計 in OCR order.
    extension_prices: list[float] = []
    if pending_names and len(pending_prices) < len(pending_names) and item_zone_end < len(lines):
        for i in range(item_zone_end + 1, len(lines)):
            s = lines[i].strip()
            if not s:
                continue
            if re.search(r'[¥￥]', s):
                break  # Hit ¥-prefixed totals zone
            if re.search(
                r'^(外税|内税|消費税|対象|お預り|現計|お釣り|釣銭|総額|'
                r'合\s*計|小\s*計|WAON|現金|クレジット|カード|お会計|電子)',
                s,
            ):
                break
            if _QTY_NOTATION_RE.search(s) or _PARTIAL_QTY_RE.match(s):
                continue
            v = _parse_price(s)
            if v is not None:
                extension_prices.append(v)
                # Stop if we have enough to complete the chain.
                if len(pending_prices) + len(extension_prices) >= len(pending_names):
                    break

    if extension_prices:
        _flush_chain_with_extension(extension_prices)
    else:
        _flush_chain()

    if len(ordered_pairs) < 2:
        return

    # Match LLM items to ordered_pairs by description-prefix overlap, then
    # greedy-claim by best score. Duplicate-named items (e.g., two
    # 牛豚ミンチ(解凍) lines) and out-of-order LLM emissions still match.
    def _match_score(ocr_name: str, llm_desc: str) -> int:
        clean = re.sub(r'^[\d\s\*\(（]+', '', ocr_name).strip()
        for prefix_len in (6, 5, 4, 3):
            if len(clean) >= prefix_len and clean[:prefix_len] in llm_desc[:14]:
                return prefix_len
            if len(ocr_name) >= prefix_len and ocr_name[:prefix_len] in llm_desc[:14]:
                return prefix_len
        return 0

    eligible_items: list[tuple[int, dict, str, float, int]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        desc = (item.get("description") or "").strip()
        total = item.get("total")
        if not desc or total is None:
            continue
        if (item.get("discount") or 0) > 0:
            continue
        qty = item.get("qty", 1) or 1
        eligible_items.append((idx, item, desc, float(total), int(qty)))

    # Score every (item, ocr_pair) combination, then greedy-claim.
    candidates: list[tuple[int, int, int]] = []  # (score, item_idx_in_eligible, ocr_pair_idx)
    for ei, (idx, _, desc, _, _) in enumerate(eligible_items):
        for p in range(len(ordered_pairs)):
            score = _match_score(ordered_pairs[p][0], desc)
            if score >= 3:
                candidates.append((score, ei, p))
    # Sort by score descending; tie-break by ei (earliest LLM item first).
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
    claimed_pair: set[int] = set()
    claimed_item: set[int] = set()
    matches: dict[int, int] = {}  # eligible idx -> ocr_pair idx
    for score, ei, p in candidates:
        if ei in claimed_item or p in claimed_pair:
            continue
        matches[ei] = p
        claimed_item.add(ei)
        claimed_pair.add(p)

    overrides: list[tuple[int, float, float, int]] = []
    for ei, p in matches.items():
        idx, _, _, total, qty = eligible_items[ei]
        ocr_price = ordered_pairs[p][1]
        if abs(ocr_price - total) < 1:
            continue
        overrides.append((idx, float(ocr_price), float(total), int(qty)))

    if not overrides:
        return

    # Apply overrides only if the collective effect strictly improves
    # items_sum's distance to a target. This catches "swap" scenarios
    # (two items with reversed totals) where a single greedy fix would
    # regress items_sum, but applying both is neutral or beneficial.
    items_sum = sum(i.get("total", 0) for i in items if isinstance(i, dict))
    total_delta = sum(new_total - old_total for _, new_total, old_total, _ in overrides)
    new_sum = items_sum + total_delta
    cur_diff = min(abs(items_sum - t) for t in targets)
    new_diff = min(abs(new_sum - t) for t in targets)
    if new_diff > cur_diff:
        return  # Net regression — don't apply
    if new_diff == cur_diff and total_delta == 0:
        # Pure swap (no items_sum change). Apply only if it actually
        # changes the description-total pairing for ≥ 2 items (otherwise
        # nothing happens).
        if len(overrides) < 2:
            return
    elif new_diff == cur_diff and total_delta != 0:
        return  # Same gap but in the other direction — don't apply

    for idx, new_total, _, qty in overrides:
        items[idx]["total"] = new_total
        if qty == 1 and items[idx].get("unit_price") is not None:
            items[idx]["unit_price"] = new_total
        elif qty > 1 and qty != 0 and new_total % qty == 0:
            items[idx]["unit_price"] = new_total / qty


def _replace_hallucinated_dup_with_ocr_item(items, unified_text, target_subtotal, target_total):
    """When LLM has duplicate items AND items_sum is off-target, look for an
    OCR-grounded item whose substitution closes the gap.

    Generic: handles any LLM hallucination where it copy-pastes a nearby
    item's price+description onto a different item, masking the right
    one. Only applies when:
      - items_sum doesn't match subtotal or total (within ±2 yen)
      - LLM has ≥ 2 items with the same (description, total)
      - Exactly one unaccounted OCR ¥amount equals dup_total + gap
    """
    if not items or len(items) < 2:
        return
    items_sum = sum(i.get("total", 0) for i in items if isinstance(i, dict))
    targets = [t for t in (target_subtotal, target_total) if t]
    if not targets:
        return
    if any(abs(items_sum - t) <= 2 for t in targets):
        return  # items already balance

    groups: dict[tuple[str, float], list[int]] = {}
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        desc = (it.get("description") or "").strip()
        total = it.get("total")
        if not desc or total is None:
            continue
        groups.setdefault((desc, float(total)), []).append(i)
    duplicates = {k: v for k, v in groups.items() if len(v) >= 2}
    if not duplicates:
        return

    lines = unified_text.split('\n')
    # Bound to the item zone: stop at the first 小計/合計 line so we don't
    # treat tax/total values as item-price candidates.
    zone_end = len(lines)
    for li, line in enumerate(lines):
        if re.search(r'^(小\s*計|合\s*計|外税|内税|消費税|お預り|現計|お釣り|釣銭)',
                     line.strip()):
            zone_end = li
            break
    ocr_prices: list[tuple[int, float]] = []
    _BARE_PRICE_RE = re.compile(r'^[¥￥]?\s*(\d[\d,]{0,5})\s*[*※軽除＊・X+%A]?\s*$')
    for li in range(zone_end):
        line = lines[li]
        if _SKIP_PRICE_LINE.search(line):
            continue
        # ¥-prefixed amounts (anywhere in the line)
        for m in re.finditer(r'[¥￥]\s*([\d,]+)', line):
            try:
                amt = float(m.group(1).replace(',', ''))
            except ValueError:
                continue
            if amt > 0:
                ocr_prices.append((li, amt))
        # Bare-digit price lines (no ¥), with optional trailing marker.
        # OCR sometimes drops the ¥ but the line is still a price token.
        if not re.search(r'[¥￥]', line):
            s = line.strip()
            if s and not re.search(r'[ぁ-んァ-ン一-龥]', s):
                m = _BARE_PRICE_RE.match(s)
                if m:
                    try:
                        amt = float(m.group(1).replace(',', ''))
                    except ValueError:
                        amt = 0
                    if amt > 0:
                        ocr_prices.append((li, amt))

    # Multiset diff: remove one OCR entry per LLM item amount
    item_amounts = [i.get("total", 0) for i in items if isinstance(i, dict)]
    unmatched = list(ocr_prices)
    for amt in item_amounts:
        for j, (_, oa) in enumerate(unmatched):
            if abs(oa - amt) < 1:
                unmatched.pop(j)
                break

    if not unmatched:
        return

    # For each duplicate × target combo, search for an OCR price that closes
    # the gap when substituted.
    candidates: list[tuple[float, int, int, float, str]] = []
    for target in targets:
        gap = target - items_sum
        for dup_key, dup_idxs in duplicates.items():
            dup_total = dup_key[1]
            wanted = dup_total + gap
            matches = [(li, oa) for li, oa in unmatched if abs(oa - wanted) <= 2]
            if len(matches) != 1:
                continue
            li, oa = matches[0]
            new_sum = items_sum - dup_total + oa
            diff = abs(new_sum - target)
            candidates.append((diff, dup_idxs[-1], li, oa, dup_key[0]))

    if not candidates:
        return
    candidates.sort()
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        return  # ambiguous tie — refuse

    diff, replace_idx, price_line_idx, new_total, _ = candidates[0]
    if diff > 2:
        return

    new_desc = _find_ocr_item_desc(lines, price_line_idx, items)
    if not new_desc:
        return

    items[replace_idx]["description"] = new_desc
    items[replace_idx]["total"] = new_total
    items[replace_idx]["unit_price"] = new_total
    items[replace_idx]["qty"] = 1


_OCR_TRAILING_PRICE_RE = re.compile(r'(?:^|[\s(（])([¥￥]?\s*\d[\d,]*)\s*[*※除軽]?\s*$')
_OCR_ZONE_END_RE = re.compile(r'^(小計|合計|現計|外税|内税|消費税|お預り|お釣り|釣銭|WAON|クレジット|お会計)')
_OCR_QTY_NOTATION_RE = re.compile(r'\d+\s*個\s*[xX×Ⅹ]\s*\d')


def _project_totals_to_ocr_multiset(extracted, unified_text):
    """When LLM items_sum is off-target but the OCR's price-column multiset
    sums to a target, snap the LLM's totals onto the OCR multiset.

    Triggered only when:
      - items_sum doesn't match subtotal or total (within ±2 yen)
      - count of OCR price tokens (after reserving qty>1 unit_prices) equals
        the count of qty=1 items, OR exactly one extra candidate exists
        and dropping it produces the unique target-matching subset
      - the resulting OCR multiset sums to a target (subtotal-qtyN_total or
        total-qtyN_total)

    The new totals replace the LLM's by total-rank (sorted-OCR -> sorted-items).
    Description↔total pairing is not preserved — the test compares totals as
    a multiset, and any (desc, total) coherence is incidental on column-stacked
    OCR layouts where the LLM couldn't recover the visual row order.
    """
    items = extracted.get("line_items") or []
    if not items:
        return
    items_sum = sum(i.get("total", 0) for i in items if isinstance(i, dict))
    subtotal = extracted.get("subtotal")
    total = extracted.get("total")
    targets = [t for t in (subtotal, total) if t]
    if not targets:
        return
    if any(abs(items_sum - t) <= 2 for t in targets):
        return

    lines = unified_text.split('\n')

    # Find item zone: from first inline-priced line to first 小計/合計-style end marker.
    zone_start = None
    for i, line in enumerate(lines):
        s = line.strip()
        if re.search(r'[¥￥]\s*\d', s) or re.search(r'\d[\d,]*\s*[*※除軽]\s*$', s):
            zone_start = i
            break
    if zone_start is None:
        return
    zone_end = len(lines)
    for i in range(zone_start, len(lines)):
        if _OCR_ZONE_END_RE.match(lines[i].strip()):
            zone_end = i
            break

    # Extract candidate price tokens. Each candidate is (line_idx, value).
    candidates: list[tuple[int, int]] = []
    for li in range(zone_start, zone_end):
        s = lines[li].strip()
        if not s:
            continue
        if _OCR_QTY_NOTATION_RE.search(s):
            continue  # qty notation like "2個 X70)" — skip whole line
        m = _OCR_TRAILING_PRICE_RE.search(s)
        if not m:
            continue
        raw = m.group(1).strip().lstrip('¥￥').replace(',', '')
        if not raw or not raw.isdigit():
            continue
        try:
            v = int(raw)
        except ValueError:
            continue
        if v < 1 or v > 99999:
            continue
        candidates.append((li, v))

    if not candidates:
        return

    # Reserve OCR tokens consumed by qty>1 items (their unit_price).
    qty_n_items = [i for i in items if isinstance(i, dict) and (i.get("qty") or 1) > 1]
    qty_1_items = [i for i in items if isinstance(i, dict) and (i.get("qty") or 1) == 1]
    if not qty_1_items:
        return  # nothing to project onto

    pool = list(candidates)
    for it in qty_n_items:
        up = it.get("unit_price")
        if up is None:
            continue
        for j, (_, v) in enumerate(pool):
            if abs(v - up) < 1:
                pool.pop(j)
                break

    n_qty1 = len(qty_1_items)
    qtyN_total = sum(i.get("total", 0) for i in qty_n_items)

    # Find the single subset (size = n_qty1) whose sum is within 2 of any target.
    target_qty1_sums = [t - qtyN_total for t in targets]

    def _multiset_matches(values: list[int]) -> int | None:
        s = sum(values)
        for t in target_qty1_sums:
            if abs(s - t) <= 2:
                return t
        return None

    pool_values = [v for _, v in pool]
    chosen: list[int] | None = None

    if len(pool_values) == n_qty1:
        if _multiset_matches(pool_values) is not None:
            chosen = list(pool_values)
    elif len(pool_values) == n_qty1 + 1:
        # Try dropping each candidate; apply only if exactly one drop produces
        # a sum that matches a target.
        viable: list[list[int]] = []
        for k in range(len(pool_values)):
            sub = pool_values[:k] + pool_values[k + 1:]
            if _multiset_matches(sub) is not None:
                viable.append(sub)
        # Multiple drops can produce equivalent sums when duplicate values
        # are present (dropping any of three "228"s gives the same subset).
        # Treat them as one viable solution.
        unique = {tuple(sorted(v)) for v in viable}
        if len(unique) == 1:
            chosen = list(viable[0])

    if chosen is None:
        return

    # Verify the projection actually changes the multiset (no point otherwise).
    sorted_qty1_totals = sorted(i.get("total", 0) for i in qty_1_items)
    sorted_chosen = sorted(chosen)
    if sorted_qty1_totals == sorted_chosen:
        return

    # Sanity: same length
    if len(sorted_chosen) != len(qty_1_items):
        return

    # Apply: assign sorted-OCR totals to qty=1 items by their current total-rank.
    qty1_sorted_idxs = sorted(
        range(len(items)),
        key=lambda j: (
            -1 if not isinstance(items[j], dict) or (items[j].get("qty") or 1) > 1 else 0,
            items[j].get("total", 0) if isinstance(items[j], dict) else 0,
        ),
    )
    qty1_sorted_idxs = [j for j in qty1_sorted_idxs
                        if isinstance(items[j], dict) and (items[j].get("qty") or 1) == 1]

    for k, idx in enumerate(qty1_sorted_idxs):
        new_total = sorted_chosen[k]
        items[idx]["total"] = new_total
        items[idx]["unit_price"] = new_total


def _find_ocr_item_desc(lines, price_line_idx, existing_items):
    """Find a plausible item description for an OCR price line."""
    existing_descs = {
        (it.get("description") or "").strip()
        for it in existing_items if isinstance(it, dict)
    }

    def _clean(text: str) -> str:
        text = text.strip()
        m = re.search(r'[¥￥]', text)
        if m:
            text = text[:m.start()].strip()
        text = re.sub(r'\s+[\d,]+\s*[点個コ]\s*$', '', text).strip()
        text = re.sub(r'\s*[※\*非外内]\s*$', '', text).strip()
        mc = re.match(r'^\d{4,}[A-Za-z]{0,3}\)?\s?(.+)$', text)
        if mc and re.search(r'[ぁ-んァ-ン一-龥]', mc.group(1)):
            text = mc.group(1).strip()
        return text

    def _is_valid(text: str) -> bool:
        if not text or len(text) < 3:
            return False
        if text in _GENERIC_DESC_MARKERS:
            return False
        if _SKIP_PRICE_LINE.search(text):
            return False
        if re.match(r'^[\d,\s\-\(\)\.\*※軽除外]+$', text):
            return False
        if not re.search(r'[ぁ-んァ-ン一-龥]', text):
            return False
        return True

    # Same-line first (rejoin merged item+price)
    cand = _clean(lines[price_line_idx])
    if _is_valid(cand) and cand not in existing_descs:
        return cand
    # Search backward up to 15 lines, then forward up to 5
    for j in list(range(price_line_idx - 1, max(price_line_idx - 16, -1), -1)) + \
             list(range(price_line_idx + 1, min(price_line_idx + 6, len(lines)))):
        cand = _clean(lines[j])
        if _is_valid(cand) and cand not in existing_descs:
            return cand
    return None


def _remove_unit_rate_phantom_items(extracted):
    """Remove items whose description is a unit-rate notation (e.g. '23 X #199')
    with no Japanese characters. These appear when the LLM extracts a per-unit
    annotation as a standalone product. Conservative: only fires when the
    description has zero Japanese chars AND matches a unit-rate-like pattern.
    """
    items = extracted.get("line_items") or []
    if not items:
        return
    keep = []
    for it in items:
        if not isinstance(it, dict):
            keep.append(it)
            continue
        desc = (it.get("description") or "").strip()
        if not desc:
            keep.append(it)
            continue
        if re.search(r'[ぁ-んァ-ン一-龥]', desc):
            keep.append(it)
            continue
        # Pure-ASCII/digit unit-rate notation like "23 X #199" or "2X@99"
        if re.match(r'^[\d,]+\s*[xX×]\s*[#＃@]?\s*[\d,]+\s*[#＃]?\s*$', desc):
            continue
        keep.append(it)
    extracted["line_items"] = keep


def _drop_banner_phantom_items(items, unified_text):
    """Drop items whose description matches a known Japanese receipt banner
    phrase (boilerplate header/footer text — never a real product).

    Generic-purpose: applies to any receipt; the banner list is the small
    set of boilerplate phrases that appear across Japanese receipts from
    many merchants. Real product names contain product nouns and should
    not match these patterns.

    Examples caught:
      - 'ぜひ当店でお買物くださいませ' (please shop at our store)
      - '毎月20日・30日はお客さま感謝デー' (customer appreciation day)
      - '※印は軽減税率8%対象商品' (asterisk = reduced rate item)
      - '※印は軽減税率(8%) 適用商品です'
    """
    if not items:
        return
    kept = []
    for item in items:
        if not isinstance(item, dict):
            kept.append(item)
            continue
        desc = (item.get("description") or "").strip()
        if desc and _BANNER_PHRASE_RE.search(desc):
            continue
        kept.append(item)
    if len(kept) != len(items):
        items.clear()
        items.extend(kept)


def _fix_priced_in_name_items(extracted, unified_text):
    """Fix items whose description contains its price (e.g. '100円均一')
    when the LLM extracted a wrong total.

    Pattern: a description like '100円均一', '500円商品', '300円ショップ'
    literally states the item's price in yen. If the LLM extracted such an
    item with total ≠ N AND there's an unmatched orphan ¥N in the OCR,
    update the item's total to N.

    Generic — applies to any item whose description has 'N円' followed by
    Japanese characters and where pipeline mis-extracted the price.

    Conservative: only fires when (a) description prefix matches pattern,
    (b) extracted total != name's stated price, (c) the corrected total
    moves items_sum closer to subtotal/total target, and (d) an unmatched
    orphan ¥N exists in OCR.
    """
    items = extracted.get("line_items") or []
    subtotal = extracted.get("subtotal")
    total = extracted.get("total")
    if not items:
        return

    items_sum = sum(i.get("total", 0) for i in items if isinstance(i, dict))
    targets = [t for t in (subtotal, total) if t]
    if not targets:
        return

    # If items already balance, don't touch
    if any(abs(items_sum - t) <= 2 for t in targets):
        return

    # Collect OCR ¥ amounts
    lines = unified_text.split('\n')
    ocr_amounts: list[float] = []
    for line in lines:
        if _SKIP_PRICE_LINE.search(line):
            continue
        for m in re.finditer(r'[¥￥]\s*([\d,]+)', line):
            try:
                ocr_amounts.append(float(m.group(1).replace(',', '')))
            except ValueError:
                pass

    # Multiset diff: remove one OCR entry per item amount
    item_totals = [i.get("total", 0) for i in items if isinstance(i, dict)]
    unmatched = list(ocr_amounts)
    for t in item_totals:
        for j, oa in enumerate(unmatched):
            if abs(oa - t) < 1:
                unmatched.pop(j)
                break

    if not unmatched:
        return

    # Match items whose description has 'N円<japanese>' prefix where N is
    # the implied price (e.g. '100円均一' → price 100).
    _PRICED_NAME_RE = re.compile(r'^(\d{2,5})\s*円')
    for item in items:
        if not isinstance(item, dict):
            continue
        desc = (item.get("description") or "").strip()
        m = _PRICED_NAME_RE.match(desc)
        if not m:
            continue
        try:
            named_price = float(m.group(1))
        except ValueError:
            continue
        cur_total = item.get("total", 0)
        if abs(named_price - cur_total) <= 2:
            continue  # already correct

        # Is named_price an unmatched OCR amount?
        if not any(abs(oa - named_price) <= 1 for oa in unmatched):
            continue

        # Try the fix: update total/unit_price/qty
        new_items_sum = items_sum - cur_total + named_price
        # Only apply if it strictly improves the gap
        old_gap = min(abs(items_sum - t) for t in targets)
        new_gap = min(abs(new_items_sum - t) for t in targets)
        if new_gap >= old_gap:
            continue

        # Apply
        item["total"] = named_price
        item["unit_price"] = named_price
        item["qty"] = 1
        items_sum = new_items_sum
        # Remove the matched amount from unmatched so it can't be reused
        for j, oa in enumerate(unmatched):
            if abs(oa - named_price) <= 1:
                unmatched.pop(j)
                break


def _fix_digit_misread_items(extracted, unified_text):
    """When items_sum is short by a small N, try OCR digit-misread corrections
    on items. A common scenario: OCR reads '108※' (108 yen, reduced rate) as
    '100%' (the 8 + ※ became %). The LLM extracts total=100; we need 108.

    Strategy: for items_sum gap N, look for items where:
      - item.total + N is a plausible OCR misread (single-digit confusion:
        0↔8, 0↔6, 1↔7, 6↔8, etc.)
      - the corrected total appears in OCR text as a plausible price
      - applying the correction moves items_sum exactly to subtotal/total

    Conservative — only fires when the corrected total is in OCR (somewhere),
    the gap matches exactly, and only one such correction is found.
    """
    items = extracted.get("line_items") or []
    subtotal = extracted.get("subtotal")
    total = extracted.get("total")
    if not items:
        return
    items_sum = sum(i.get("total", 0) for i in items if isinstance(i, dict))
    targets = [t for t in (subtotal, total) if t]
    if not targets:
        return

    # Compute gap to each target; pick the smallest non-zero gap
    gaps = [(t - items_sum, t) for t in targets]
    valid_gaps = [(g, t) for g, t in gaps if 0 < g <= 50]
    if not valid_gaps:
        return
    gap = min(g for g, _ in valid_gaps)

    # Common OCR digit-confusion pairs (1-step perturbations)
    # We test if item.total + gap is plausibly the correct total by checking
    # if a single-digit replacement gets us there. Most useful is: the
    # LAST digit of total_corrected differs from total by ≤ 1 digit pair.
    def _single_digit_diff(a: int, b: int) -> bool:
        sa, sb = str(a), str(b)
        if len(sa) != len(sb):
            return False
        diffs = [(x, y) for x, y in zip(sa, sb) if x != y]
        return len(diffs) == 1

    candidates: list[tuple[int, float]] = []  # (item_idx, new_total)
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        t = item.get("total")
        if t is None or t <= 0:
            continue
        try:
            t_int = int(t)
        except (TypeError, ValueError):
            continue
        new_total = t_int + int(gap)
        if not _single_digit_diff(t_int, new_total):
            continue
        # Look for evidence in OCR: the corrected total may not appear
        # literally (it's an OCR misread!), but a "T%"-style line matching
        # the original OCR-misread pattern is a strong signal.
        # E.g., 100 → 108 with 0/8 confusion → look for 'T%' on its own line
        # which is common when '%' was misread of '8※' or similar.
        sa, sb = str(t_int), str(new_total)
        # If the differing digit changed to/from 0, 8, or 6 (common
        # confusions), pattern '<original>%' or '<original>除' as a standalone
        # line is suspicious — likely a misread.
        diff_pairs = [(x, y) for x, y in zip(sa, sb) if x != y]
        if not diff_pairs:
            continue
        old_d, new_d = diff_pairs[0]
        if (old_d, new_d) not in {('0', '8'), ('8', '0'), ('0', '6'),
                                  ('6', '0'), ('1', '7'), ('7', '1'),
                                  ('6', '8'), ('8', '6'), ('5', '6'),
                                  ('6', '5')}:
            continue
        # Match a standalone "<original>%" line in OCR (signature of
        # 8/0 misread where the trailing '8※' became '%').
        misread_pattern = re.compile(rf'^\s*{re.escape(sa)}%\s*$', re.MULTILINE)
        if not misread_pattern.search(unified_text):
            continue
        candidates.append((idx, float(new_total)))

    if len(candidates) != 1:
        return

    idx, new_total = candidates[0]
    items[idx]["total"] = new_total
    if items[idx].get("qty", 1) == 1:
        items[idx]["unit_price"] = new_total


def _drop_phantom_from_tax_amount(extracted):
    """Drop items whose total equals a printed tax amount AND whose
    description is a prefix of another item's description with an embedded
    digit suffix matching some other item's price.

    Scenario: OCR puts a tax amount (e.g., '¥97' for 8% tax) on a line
    visually close to an item description. The LLM creates a phantom item
    using that price and a corrupted description like 'X  98' (where 98 is
    another item's price stuck on the end of X's name).

    Conservative — fires only when ALL of:
      - phantom.total == any tax_entry.amount (exact match)
      - phantom.desc has a trailing whitespace+digit suffix
      - the desc-without-suffix appears as another item's full description
      - that suffix matches the other item's total

    Generic across receipts.
    """
    items = extracted.get("line_items", []) or []
    taxes = extracted.get("taxes", []) or []
    if len(items) < 2 or not taxes:
        return
    tax_amounts = {
        float(t.get("amount", 0))
        for t in taxes
        if isinstance(t, dict) and t.get("amount") not in (None, 0)
    }
    if not tax_amounts:
        return

    _SUFFIX = re.compile(r'^(.+?)\s+([\d,]{1,6})\s*[\*※]?\s*$')
    by_desc_total: dict[tuple, int] = {}
    for i, it in enumerate(items):
        if isinstance(it, dict):
            d = (it.get("description") or "").strip()
            t = it.get("total")
            if d and t is not None:
                by_desc_total[(d, float(t))] = i

    drop_idxs = set()
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        total = it.get("total")
        if total is None:
            continue
        try:
            total_f = float(total)
        except (TypeError, ValueError):
            continue
        if total_f not in tax_amounts:
            continue
        desc = (it.get("description") or "").strip()
        m = _SUFFIX.match(desc)
        if not m:
            continue
        prefix = m.group(1).strip()
        try:
            suffix_val = float(m.group(2).replace(',', ''))
        except ValueError:
            continue
        # Must keep Japanese in the prefix
        if not re.search(r'[ぁ-んァ-ン一-龥]', prefix):
            continue
        # Look for another item with desc==prefix and total==suffix_val
        if (prefix, suffix_val) in by_desc_total:
            other_idx = by_desc_total[(prefix, suffix_val)]
            if other_idx != i:
                drop_idxs.add(i)
    if drop_idxs:
        extracted["line_items"] = [
            it for i, it in enumerate(items) if i not in drop_idxs
        ]


def _strip_embedded_price_in_desc(items):
    """Strip trailing whitespace+digit suffix from descriptions when the
    digit equals the item's total/unit_price.

    OCR sometimes appends a price into the description column, producing
    descriptions like "ベビーダノンイ  228" (where 228 is the item's total)
    or "TV減の恵みきざみねぎ  98" (where 98 matches another item's price
    and the digit is leftover from the previous row).

    Only fires when:
      - description ends with whitespace + digit run
      - the trailing digit equals total OR unit_price (or differs by ≤ 1)
      - stripped description still has Japanese text

    Generic-purpose: addresses inline price fragments left in description
    by OCR row-detection failures.
    """
    if not items:
        return
    _SUFFIX_RE = re.compile(r'^(.+?)\s+([\d,]{1,6})\s*[\*※]?\s*$')
    for item in items:
        if not isinstance(item, dict):
            continue
        desc = (item.get("description") or "").strip()
        if not desc:
            continue
        m = _SUFFIX_RE.match(desc)
        if not m:
            continue
        prefix = m.group(1).strip()
        try:
            suffix_val = float(m.group(2).replace(',', ''))
        except ValueError:
            continue
        # Must keep Japanese text in the stripped prefix
        if not re.search(r'[ぁ-んァ-ン一-龥]', prefix):
            continue
        if len(prefix) < 3:
            continue
        total = item.get("total")
        unit = item.get("unit_price")
        matches_total = total is not None and abs(suffix_val - total) <= 1
        matches_unit = unit is not None and abs(suffix_val - unit) <= 1
        if matches_total or matches_unit:
            item["description"] = prefix


def _replace_duplicate_desc_from_ocr(items, unified_text):
    """When the LLM extracts duplicate (description, total) items but OCR
    shows distinct items at that total, swap a duplicate's description for
    the unmatched OCR description.

    Generic-purpose: addresses LLM hallucinations where it copy-pastes a
    nearby item's name onto a different item with the same price.
    Conservative — only fires when:
      - LLM has ≥ 2 items with the same (description, total)
      - OCR text contains a distinct, valid item-like description with that
        same total (within ±2 yen) that doesn't match any current LLM
        description
      - The replacement description appears nearby a matching ¥amount in OCR
    """
    if not items or len(items) < 2:
        return

    # Group LLM items by (description, total)
    groups: dict[tuple[str, float], list[int]] = {}
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        desc = (it.get("description") or "").strip()
        total = it.get("total")
        if not desc or total is None:
            continue
        groups.setdefault((desc, float(total)), []).append(i)

    duplicates = {key: idxs for key, idxs in groups.items() if len(idxs) >= 2}
    if not duplicates:
        return

    # Existing descriptions, lowered for matching
    existing_descs = {
        (it.get("description") or "").strip()
        for it in items if isinstance(it, dict)
    }

    lines = unified_text.split('\n')

    # For each price line in OCR, locate a nearby description (same logic
    # used by _recover_missing_items_from_gap, but inline since this fires
    # earlier in the pipeline).
    def _candidate_desc_for_price(price_idx: int, target_amt: float) -> str | None:
        # Check the price line itself first (rejoin_price_lines may have
        # merged item + price on one line).
        line_text = lines[price_idx]
        for raw in [line_text] + [lines[j] for j in range(price_idx - 1, max(price_idx - 6, -1), -1)]:
            cand = raw.strip()
            # Strip price suffix
            m = re.search(r'[¥￥]', cand)
            if m:
                cand = cand[:m.start()].strip()
            # Strip trailing count markers and tax markers
            cand = re.sub(r'\s+[\d,]+\s*[点個コ]\s*$', '', cand).strip()
            cand = re.sub(r'\s*[※\*非外内]\s*$', '', cand).strip()
            # Strip leading product code
            mc = re.match(r'^\d{4,}[A-Za-z]{0,3}\)?\s?(.+)$', cand)
            if mc and re.search(r'[ぁ-んァ-ン一-龥]', mc.group(1)):
                cand = mc.group(1).strip()
            # Validate
            if not cand or len(cand) < 3:
                continue
            if cand in _GENERIC_DESC_MARKERS:
                continue
            if re.match(r'^[\d,\s\-\(\)\.\*※軽除外]+$', cand):
                continue
            if not re.search(r'[ぁ-んァ-ン一-龥]', cand):
                continue
            if _SKIP_PRICE_LINE.search(cand):
                continue
            return cand
        return None

    # Bare-digit price line: "228" or "228*" or "228※" (AEON column-format
    # receipts often print prices without ¥ in the items zone).
    _BARE_PRICE_LINE = re.compile(r'^\s*([\d,]+)\s*[\*※]?\s*$')
    # Inline bare-digit price suffix: "ベビーダノンイ  228*" — digit at end
    # of line preceded by Japanese text and whitespace.
    _INLINE_BARE_PRICE = re.compile(r'[ぁ-んァ-ン一-龥]\s+([\d,]{2,})\s*[\*※]?\s*$')

    for (dup_desc, dup_total), dup_idxs in duplicates.items():
        # Collect OCR descriptions associated with prices ≈ dup_total
        ocr_descs: list[str] = []
        for li, line in enumerate(lines):
            if _SKIP_PRICE_LINE.search(line):
                continue
            for m in re.finditer(r'[¥￥]\s*([\d,]+)', line):
                try:
                    amt = float(m.group(1).replace(',', ''))
                except ValueError:
                    continue
                if abs(amt - dup_total) <= 2:
                    cand = _candidate_desc_for_price(li, amt)
                    if cand and cand not in ocr_descs:
                        ocr_descs.append(cand)
            # Also accept bare-digit price lines / inline-bare suffixes
            stripped = line.strip()
            bare_m = _BARE_PRICE_LINE.match(stripped)
            inline_m = _INLINE_BARE_PRICE.search(line) if not bare_m else None
            for matched in (bare_m, inline_m):
                if not matched:
                    continue
                try:
                    amt = float(matched.group(1).replace(',', ''))
                except ValueError:
                    continue
                if abs(amt - dup_total) <= 2:
                    cand = _candidate_desc_for_price(li, amt)
                    if cand and cand not in ocr_descs:
                        ocr_descs.append(cand)

        # OCR-distinct descriptions not currently in LLM extraction
        unmatched_ocr_descs = [
            d for d in ocr_descs
            if d not in existing_descs and d != dup_desc
        ]
        if not unmatched_ocr_descs:
            continue

        # Need at least as many distinct OCR descs as duplicates − 1, since
        # one duplicate is real (matches the dup_desc). Keep one duplicate;
        # replace the rest with OCR-derived descriptions.
        replacements = unmatched_ocr_descs[: len(dup_idxs) - 1]
        for repl_desc, idx in zip(replacements, dup_idxs[1:]):
            items[idx]["description"] = repl_desc
            existing_descs.add(repl_desc)


def _dedup_same_total_items(extracted):
    """Remove duplicate items with identical description and total, keeping qty>1 version.

    Also removes "phantom-child" duplicates where the LLM produced the
    unit-price row as a separate qty=1 item alongside the real qty=N×unit_price
    item. Only applies when the deduped sum is strictly closer to its expected
    target than the original sum. The target is whichever of subtotal/total the
    original sum is closer to — items match subtotal on 外税 receipts and total
    on 内税 receipts, so the LLM's extraction style picks the right anchor.
    Without this, legitimate duplicates (e.g. two hot-dog meals at the same
    price) get wrongly removed on 内税 receipts where subtotal < items_sum.
    """
    items = list(extracted.get("line_items", []) or [])
    if len(items) < 2:
        return
    original_sum = sum(i.get("total", 0) for i in items if isinstance(i, dict))
    subtotal = extracted.get("subtotal")
    total = extracted.get("total")
    candidates = [v for v in (subtotal, total) if v]
    if not candidates:
        return
    target = min(candidates, key=lambda v: abs(v - original_sum))

    keep_mask = [True] * len(items)
    seen: dict[tuple, int] = {}
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        key = (item.get("description", ""), item.get("total", 0))
        if key in seen:
            prev_idx = seen[key]
            prev_qty = items[prev_idx].get("qty", 1)
            cur_qty = item.get("qty", 1)
            remove_idx = prev_idx if cur_qty > prev_qty else i
            keep_mask[remove_idx] = False
            if remove_idx == prev_idx:
                seen[key] = i
        else:
            seen[key] = i

    # Phantom-child pass: same description, same unit_price, one qty>1 with
    # total=qty*unit_price and another qty=1 with total=unit_price. The qty=1
    # entry is the unit-price/per-item line read as a separate item; drop it.
    by_desc_unit: dict[tuple, list[int]] = {}
    for i, item in enumerate(items):
        if not isinstance(item, dict) or not keep_mask[i]:
            continue
        desc = item.get("description", "")
        unit = item.get("unit_price")
        if not desc or unit is None or unit <= 0:
            continue
        by_desc_unit.setdefault((desc, unit), []).append(i)
    for (desc, unit), idxs in by_desc_unit.items():
        if len(idxs) < 2:
            continue
        has_real = any(items[k].get("qty", 1) > 1 for k in idxs)
        if not has_real:
            continue
        for k in idxs:
            it = items[k]
            qty_k = it.get("qty", 1)
            tot_k = it.get("total", 0)
            if qty_k == 1 and abs(tot_k - unit) < 1:
                keep_mask[k] = False

    new_items = [item for item, keep in zip(items, keep_mask) if keep]
    new_sum = sum(i.get("total", 0) for i in new_items if isinstance(i, dict))
    if abs(new_sum - target) < abs(original_sum - target):
        extracted["line_items"] = new_items


def _fix_qty_hallucinations(items, unified_text):
    """Fix LLM qty hallucinations by checking if total/price appear in OCR text."""
    # Pre-compute qty-detail lines (e.g., "(3個 X 単68)") and the implied
    # totals — we use these to validate the LLM's qty/unit_price extraction.
    # If a qty-detail line corresponds to the item AND its qty*unit matches
    # the LLM's qty*unit, the LLM is right and we should NOT "fix" it.
    qty_detail_re = re.compile(
        r'[\(\<]?\s*(\d+)\s*[個コ点]\s*[xX×]\s*(?:単|@)?\s*(\d+)\s*[\)\>]?'
    )
    qty_detail_pairs: list[tuple[int, int]] = []  # (qty, unit) pairs
    for line in unified_text.split('\n'):
        m = qty_detail_re.search(line.strip())
        if m:
            try:
                qty_detail_pairs.append((int(m.group(1)), int(m.group(2))))
            except ValueError:
                pass

    def _has_supporting_qty_detail(item_qty: int, item_unit: float) -> bool:
        """OCR has a qty-detail line confirming this item's qty AND unit_price."""
        return any(q == item_qty and abs(u - item_unit) < 1
                   for q, u in qty_detail_pairs)

    for item in items:
        if not isinstance(item, dict) or item.get("qty", 1) <= 1:
            continue
        total = item.get("total", 0)
        unit_price = item.get("unit_price")
        if unit_price is None:
            continue
        # Skip if qty-detail line confirms this qty × unit_price
        if _has_supporting_qty_detail(int(item.get("qty", 1)), float(unit_price)):
            continue
        total_str = str(int(total)) if total == int(total) else str(total)
        price_str = str(int(unit_price)) if unit_price == int(unit_price) else str(unit_price)
        if total_str not in unified_text and price_str in unified_text:
            item["qty"] = 1
            item["total"] = unit_price - (item.get("discount") or 0)

    # Qty from product name confusion (e.g. "集成材 10" → qty=10)
    for item in items:
        if not isinstance(item, dict) or item.get("qty", 1) <= 1:
            continue
        total = item.get("total", 0)
        unit_price = item.get("unit_price")
        if unit_price is None or total <= 0:
            continue
        # Skip if qty-detail line confirms
        if _has_supporting_qty_detail(int(item.get("qty", 1)), float(unit_price)):
            continue
        total_int = str(int(total)) if total == int(total) else str(total)
        price_int = str(int(unit_price)) if unit_price == int(unit_price) else str(unit_price)
        has_yen_total = bool(re.search(r'[¥￥]\s*' + re.escape(total_int) + r'(?!\d)', unified_text))
        has_yen_price = bool(re.search(r'[¥￥]\s*' + re.escape(price_int) + r'(?!\d)', unified_text))
        if has_yen_total and not has_yen_price:
            item["qty"] = 1
            item["unit_price"] = total
            item["total"] = total - (item.get("discount") or 0)


def _revert_unsupported_qty_inflation(items, unified_text):
    """Revert qty>1 to qty=1 when the OCR has no qty notation supporting it.

    LLM variance issue: when two items share a prefix (e.g., 'TVBP カットトマト'
    with `(2個 X 単128)` followed by 'TVBP ジンジャーエー' without a qty
    notation), the LLM sometimes applies the earlier qty notation to the
    later same-prefix item, inflating qty=1→2 and total=128→256.

    Detection: for each LLM item with qty≥2, find its OCR name-line by
    longest-prefix match (last occurrence, since the LLM emits items in
    OCR order). If no qty notation appears within 3 lines after the
    matched OCR line AND items_sum is currently off-target by an amount
    consistent with the inflation, revert to qty=1.

    Conservative — only fires when:
      - qty ≥ 2 AND total = qty × unit_price (clean qty inflation)
      - No qty notation in 3-line OCR window after the item's name-line
      - The match line is unambiguous (using a long-enough prefix)
    """
    if not items:
        return
    ocr_lines = unified_text.split('\n')
    qty_re = re.compile(
        r'[\(（<]?\s*\d+\s*[コ個点]\s*[xX×]\s*(?:単|@)?\s*\d[\d,]*'
    )
    for item in items:
        if not isinstance(item, dict):
            continue
        qty = item.get("qty", 1) or 1
        if qty < 2:
            continue
        if (item.get("discount") or 0) > 0:
            continue
        desc = (item.get("description") or "").strip()
        if not desc or len(desc) < 5:
            continue
        unit = item.get("unit_price")
        total = item.get("total")
        if unit is None or total is None:
            continue
        # Only consider clean qty=N×unit_price patterns.
        if abs(unit * qty - total) > 1:
            continue
        # Find the OCR name-line. Use a long prefix and take the LAST
        # occurrence (LLM extracts items in OCR order; for the second of
        # two same-prefix items, the right line is the later one).
        prefix = desc[:6] if len(desc) >= 6 else desc
        match_li = None
        for li, line in enumerate(ocr_lines):
            if prefix in line:
                match_li = li
        if match_li is None:
            continue
        # Look at the next 3 non-empty lines for a qty notation. Stop
        # early at the next item-name line.
        has_qty = False
        for offset in range(1, 5):
            j = match_li + offset
            if j >= len(ocr_lines):
                break
            nearby = ocr_lines[j].strip()
            if not nearby:
                continue
            if qty_re.search(nearby):
                has_qty = True
                break
            # Stop on next name (≥ 2 Japanese chars) without qty notation.
            if re.search(r'[ぁ-んァ-ン一-龥]{2,}', nearby):
                break
        if has_qty:
            continue
        # No qty notation supports this qty>1 — revert to qty=1.
        item["qty"] = 1
        item["total"] = unit
        item["unit_price"] = unit


def _apply_qty_notation_from_ocr(items, unified_text):
    """When OCR has '(N個 X unit)' notation immediately after an item line
    AND the LLM didn't apply it (qty=1 with anomalously low total), update
    qty/unit/total from the OCR pattern.

    Generic-purpose: handles receipts where the LLM ignores explicit qty/unit
    annotations and instead picks up a stray weight/quantity number as the
    total. Conservative — only fires when OCR shows a clear qty notation
    near the item AND applying it strictly increases the total (so we don't
    overwrite an already-correct qty=N item).
    """
    ocr_lines = unified_text.split('\n')
    # OCR sometimes mis-reads opening parens as "<", so accept either prefix.
    qty_re = re.compile(r'[\(（<]?\s*(\d+)\s*[コ個点]\s*[xX×]\s*(?:単|@)?\s*(\d[\d,]*)\s*[\)）>]?')
    for item in items:
        if not isinstance(item, dict):
            continue
        desc = (item.get("description") or "").strip()
        if not desc or len(desc) < 4:
            continue
        cur_total = item.get("total", 0)
        desc_prefix = desc[:4]
        for li, line in enumerate(ocr_lines):
            if desc_prefix not in line:
                continue
            for offset in range(1, 5):
                if li + offset >= len(ocr_lines):
                    break
                nearby = ocr_lines[li + offset].strip()
                if not nearby:
                    continue
                m = qty_re.search(nearby)
                if m:
                    qty = float(m.group(1))
                    try:
                        unit = float(m.group(2).replace(',', ''))
                    except ValueError:
                        break
                    if qty >= 2 and unit > 0 and qty * unit > cur_total + 1:
                        # Skip discounted items — the LLM already merged the
                        # discount into total at the original (correct) qty.
                        # Only override when current item has no discount.
                        if not (item.get("discount") or 0):
                            item["qty"] = qty
                            item["unit_price"] = unit
                            item["total"] = qty * unit
                    break
                # Stop on next item desc (Japanese without qty notation)
                if re.search(r'[ぁ-んァ-ン一-龥]{2,}', nearby) and not re.search(r'\d+\s*[コ個点]', nearby):
                    break
            break


def _fix_qty_from_ocr_patterns(items, unified_text):
    """Fix quantities using ×N個 patterns and qty×price scanners in OCR text."""
    ocr_lines = unified_text.split('\n')

    # 本体合計(N点) — Starbucks-style summary that names the total item count.
    # When the receipt has exactly one line_item but the summary says N>1,
    # the LLM lost the qty (which usually appears as a "NT"/"3T" prefix on
    # the item line). Apply qty=N and divide unit_price.
    body_count_m = re.search(r'本体合計\s*\(?\s*(\d+)\s*点\s*\)?', unified_text)
    if body_count_m and len(items) == 1 and isinstance(items[0], dict):
        body_qty = int(body_count_m.group(1))
        item = items[0]
        cur_qty = item.get("qty", 1) or 1
        cur_total = item.get("total") or 0
        if body_qty > 1 and cur_qty == 1 and cur_total > 0 and cur_total % body_qty == 0:
            item["qty"] = float(body_qty)
            item["unit_price"] = cur_total / body_qty

    # Match by description prefix
    for item in items:
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
        for li, ocr_line in enumerate(ocr_lines):
            if desc_prefix not in ocr_line:
                continue
            for offset in range(0, 4):
                if li + offset >= len(ocr_lines):
                    break
                m = re.search(pattern_mult, ocr_lines[li + offset])
                if not m:
                    m = re.search(pattern_ko, ocr_lines[li + offset])
                if m:
                    correct_qty = float(m.group(1))
                    if correct_qty != item.get("qty", 1) and correct_qty > 1:
                        item["qty"] = correct_qty
                        item["total"] = unit_price * correct_qty - (item.get("discount") or 0)
                    break
            break

    # Garbled multiplication lines: try digit substrings validated against item total
    for item in items:
        if not isinstance(item, dict) or item.get("qty", 1) != 1:
            continue
        total = item.get("total", 0)
        if total <= 0:
            continue
        desc = item.get("description", "")
        desc_prefix = desc[:4] if len(desc) >= 4 else desc
        if not desc_prefix:
            continue
        for li, ocr_line in enumerate(ocr_lines):
            if desc_prefix not in ocr_line:
                continue
            for offset in range(1, 3):
                if li + offset >= len(ocr_lines):
                    break
                nearby = ocr_lines[li + offset].strip()
                if not re.search(r'[×xX]', nearby):
                    continue
                parts = re.split(r'\s*[×xX]\s*', nearby, maxsplit=1)
                if len(parts) != 2:
                    continue
                left_digits = re.findall(r'\d+', parts[0])
                right_digits = re.findall(r'\d+', parts[1])
                found = False
                for ld in left_digits:
                    for rd in right_digits:
                        q, p = int(ld), int(rd)
                        if 2 <= q <= 9 and p > 0 and q * p == total:
                            item["qty"] = float(q)
                            item["unit_price"] = float(p)
                            item["total"] = float(q * p)
                            found = True
                            break
                        if len(ld) > 1:
                            q2 = int(ld[0])
                            if 2 <= q2 <= 9 and q2 * p == total:
                                item["qty"] = float(q2)
                                item["unit_price"] = float(p)
                                item["total"] = float(q2 * p)
                                found = True
                                break
                        if len(rd) > 1:
                            p2 = int(rd[1:])
                            if p2 > 0 and q * p2 == total:
                                item["qty"] = float(q)
                                item["unit_price"] = float(p2)
                                item["total"] = float(q * p2)
                                found = True
                                break
                            if len(ld) > 1:
                                q2 = int(ld[0])
                                if 2 <= q2 <= 9 and p2 > 0 and q2 * p2 == total:
                                    item["qty"] = float(q2)
                                    item["unit_price"] = float(p2)
                                    item["total"] = float(q2 * p2)
                                    found = True
                                    break
                    if found:
                        break
                if found:
                    break
            break

    # Scan ALL OCR lines for qty×price patterns, match by total/price
    ocr_qty_prices: list[tuple[float, float, float]] = []
    for ocr_line in ocr_lines:
        found_qty_str, found_price_str = None, None
        m = re.search(r'(\d+)\s*[コ個]\s*[×xX]\s*(?:単|@)?\s*(\d[\d,]*)', ocr_line)
        if m:
            found_qty_str, found_price_str = m.group(1), m.group(2)
        if not found_qty_str:
            m2 = re.search(r'(?:単|@)\s*(\d[\d,]*)\s*[×xX]\s*(\d+)\s*[コ個]', ocr_line)
            if m2:
                found_price_str, found_qty_str = m2.group(1), m2.group(2)
        if not found_qty_str:
            m3 = re.search(r'[¥￥]\s*(\d[\d,]*)\s+(\d+)\s*個', ocr_line)
            if m3:
                found_price_str, found_qty_str = m3.group(1), m3.group(2)
        if found_qty_str and found_price_str:
            ocr_qty_prices.append((
                float(found_qty_str),
                float(found_price_str.replace(',', '')),
                float(found_qty_str) * float(found_price_str.replace(',', '')),
            ))

    # OCR-mangled "<unit_price>\n<qty>個" pattern: a pure-digits line followed by
    # "<digits>個" on the next line. Common when an inline "unit qty個 total"
    # line gets split (e.g. Lawson tofu where "212軽" was lost from the total).
    for li in range(len(ocr_lines) - 1):
        m_price = re.match(r'^\s*(\d[\d,]*)\s*$', ocr_lines[li])
        m_qty = re.match(r'^\s*(\d+)\s*個\s*$', ocr_lines[li + 1])
        if not (m_price and m_qty):
            continue
        qty = float(m_qty.group(1))
        if qty <= 1 or qty > 99:
            continue
        price = float(m_price.group(1).replace(',', ''))
        if price <= 0:
            continue
        ocr_qty_prices.append((qty, price, qty * price))

    for li, ocr_line in enumerate(ocr_lines):
        m_ten = re.match(r'^\s*(\d+)\s*点\s*$', ocr_line)
        if m_ten and li + 1 < len(ocr_lines):
            m_price = re.match(r'^\s*@\s*(\d[\d,]*)\s*$', ocr_lines[li + 1])
            if m_price:
                qty = float(m_ten.group(1))
                price = float(m_price.group(1).replace(',', ''))
                ocr_qty_prices.append((qty, price, qty * price))

    # Multi-line @PRICEx / QTY pattern (e.g., "@278x" then "3" on next line)
    # Apply directly to the nearest matching item by description proximity
    for li, ocr_line in enumerate(ocr_lines):
        m_at = re.match(r'^\s*[@＠](\d[\d,]*)\s*[×xX]?\s*$', ocr_line.strip())
        if m_at and li + 1 < len(ocr_lines):
            m_qty = re.match(r'^\s*(\d+)\s*$', ocr_lines[li + 1].strip())
            if m_qty:
                price = float(m_at.group(1).replace(',', ''))
                qty = float(m_qty.group(1))
                if qty > 1:
                    desc_context = None
                    for back in range(li - 1, max(li - 3, -1), -1):
                        bl = ocr_lines[back].strip()
                        if not bl:
                            continue
                        if re.match(r'^[\*¥￥@＠]?\s*\d[\d,]*\s*[*※×xX]?\s*$', bl):
                            continue
                        desc_context = bl
                        break
                    matched_item = None
                    if desc_context:
                        best_overlap = 0
                        for item in items:
                            if not isinstance(item, dict):
                                continue
                            desc = item.get("description", "")
                            if not desc or abs((item.get("unit_price") or 0) - price) >= 1:
                                continue
                            overlap = 0
                            for k in range(min(len(desc), len(desc_context)), 0, -1):
                                if desc[:k] in desc_context:
                                    overlap = k
                                    break
                            if overlap > best_overlap:
                                best_overlap = overlap
                                matched_item = item
                    if matched_item and matched_item.get("qty", 1) != qty:
                        matched_item["qty"] = qty
                        matched_item["total"] = qty * price - (matched_item.get("discount") or 0)
                    elif not matched_item:
                        ocr_qty_prices.append((qty, price, qty * price))

    used_indices: set[int] = set()
    for oq, op, ot in ocr_qty_prices:
        if oq <= 1:
            continue
        for idx, item in enumerate(items):
            if not isinstance(item, dict) or idx in used_indices:
                continue
            item_total = item.get("total", 0)
            item_price = item.get("unit_price")
            matched = abs(item_total - ot) < 1
            if not matched and item_price is not None:
                matched = abs(item_price - op) < 1 and item.get("qty", 1) != oq
            if matched:
                if item.get("qty", 1) != oq or item.get("unit_price") != op:
                    item["qty"] = oq
                    item["unit_price"] = op
                    item["total"] = op * oq - (item.get("discount") or 0)
                used_indices.add(idx)
                break

    for idx, item in enumerate(items):
        if not isinstance(item, dict) or idx in used_indices:
            continue
        if item.get("qty", 1) <= 1:
            continue
        desc = item.get("description", "")
        desc_key = desc[:8] if len(desc) >= 8 else desc
        if not desc_key:
            continue
        for li, ocr_line in enumerate(ocr_lines):
            if desc_key not in ocr_line:
                continue
            has_qty_evidence = any(
                re.search(r'[×xX]\s*\d+|\d+\s*[×xX]|単\d|@\d', ocr_lines[li + j])
                for j in range(4) if li + j < len(ocr_lines)
            )
            if has_qty_evidence:
                break
            for offset in range(1, 4):
                if li + offset >= len(ocr_lines):
                    break
                yen_m = re.search(r'[¥￥]\s*([\d,]+)', ocr_lines[li + offset])
                if yen_m:
                    ocr_price = float(yen_m.group(1).replace(',', ''))
                    if abs(ocr_price - item.get("total", 0)) > 1:
                        item["qty"] = 1
                        item["unit_price"] = ocr_price
                        item["total"] = ocr_price
                    break
            break



def _extract_fuel_usage(extracted, unified_text):
    """Populate usage field for fuel receipts from OCR volume/price data."""
    if extracted.get("usage"):
        return
    items = extracted.get("line_items") or []
    desc_text = ' '.join(
        item.get("description", "") for item in items if isinstance(item, dict)
    )
    if not any(kw in desc_text or kw in unified_text for kw in _FUEL_KEYWORDS):
        return
    volume_m = re.search(r'(\d+\.\d+)\s*L', unified_text)
    if not volume_m:
        return
    amount = float(volume_m.group(1))
    total = extracted.get("total") or extracted.get("subtotal")
    cost_per = None
    for m in re.finditer(r'(\d{2,3})\s*円', unified_text):
        candidate = float(m.group(1))
        if total and abs(amount * candidate - total) < 5:
            cost_per = candidate
            break
    extracted["usage"] = {
        "amount": amount,
        "unit": "L",
        "cost_per": cost_per,
        "meter_previous": None,
        "meter_current": None,
    }


def _fix_fuel_volume_qty(items, unified_text, receipt_total=None):
    """Normalize fuel receipt volumes (fractional qty) to qty=1.

    The reference for a single-item receipt is receipt.total (printed 合計),
    which equals the item's post-tax price for 内税 receipts and pre-tax + tax
    for 外税. Using receipt.total avoids misfires under the canonical
    pre-tax subtotal convention.
    """
    for item in items:
        if not isinstance(item, dict):
            continue
        qty = item.get("qty", 1)
        desc = item.get("description", "")
        is_fuel = any(kw in desc or kw in unified_text for kw in _FUEL_KEYWORDS)
        if not is_fuel:
            continue
        total = item.get("total", 0)
        # Case 1: fractional qty (e.g., 26.43L)
        if qty != int(qty):
            item["qty"] = 1
            item["unit_price"] = total
            break
        # Case 2: qty=1, single item, but unit_price is per-unit (e.g., yen/liter)
        # and doesn't match receipt total — correct to receipt total.
        if (qty == 1 and len(items) == 1 and receipt_total
                and total > 0 and abs(total - receipt_total) > 5):
            item["unit_price"] = receipt_total
            item["total"] = receipt_total
            break


def _expand_collapsed_items(extracted, unified_text):
    """Expand a single item with qty > 1 into individual items when OCR shows separate entries."""
    items = extracted.get("line_items", [])
    if len(items) != 1:
        return
    item = items[0]
    if not isinstance(item, dict):
        return
    qty = item.get("qty", 1)
    unit_price = item.get("unit_price")
    desc = item.get("description", "")
    if unit_price is None or not desc:
        return
    ocr_lines = unified_text.split('\n')
    ocr_desc_count = sum(
        1 for line in ocr_lines
        if desc in line and '小計' not in line and '合計' not in line
    )
    price_str = str(int(unit_price)) if unit_price == int(unit_price) else str(unit_price)
    has_bulk_pattern = bool(re.search(
        re.escape(price_str) + r'\s*[×xX]\s*\d+', unified_text
    ))
    # Case 1: qty > 1 and OCR shows separate entries
    if qty > 1 and ocr_desc_count >= qty and not has_bulk_pattern:
        extracted["line_items"] = [{
            "description": desc, "qty": 1,
            "unit_price": unit_price, "total": unit_price,
            "tax_category": item.get("tax_category", "0%"),
            "discount": 0, "discount_rate": "",
        } for _ in range(int(qty))]
        extracted["subtotal"] = unit_price * qty
    # Case 2: qty=1 but OCR shows multiple and subtotal confirms
    elif qty == 1 and ocr_desc_count >= 2 and not has_bulk_pattern:
        subtotal = extracted.get("subtotal") or extracted.get("total", 0)
        if subtotal and unit_price > 0 and subtotal > unit_price:
            inferred_qty = round(subtotal / unit_price)
            if inferred_qty >= 2 and abs(inferred_qty * unit_price - subtotal) < 2 and ocr_desc_count >= inferred_qty:
                extracted["line_items"] = [{
                    "description": desc, "qty": 1,
                    "unit_price": unit_price, "total": unit_price,
                    "tax_category": item.get("tax_category", "0%"),
                    "discount": 0, "discount_rate": "",
                } for _ in range(inferred_qty)]


def _fix_hallucinated_prices(items, unified_text):
    """Fix unit_price/total mismatches by checking which value appears in OCR text."""
    ocr_lines = unified_text.split('\n')
    for item in items:
        if not isinstance(item, dict):
            continue
        qty = item.get("qty", 1)
        discount = (item.get("discount") or 0)
        unit_price = item.get("unit_price")
        total = item.get("total")
        if qty != 1 or discount != 0 or unit_price is None or total is None:
            continue
        desc = item.get("description", "")
        desc_prefix = desc[:5] if len(desc) >= 5 else desc

        # When unit_price == total, check if the price might come from a number
        # on the description OCR line (e.g., "TV天かす 60" where 60 is grams,
        # and the actual price 98* is on the next line).
        # Only apply when the number on the desc line has NO price marker nearby
        # (a marked price like "3除" or "380※" is a real price, not a name).
        if abs(total - unit_price) < 1:
            price_str = str(int(unit_price)) if unit_price == int(unit_price) else str(unit_price)
            for idx, line in enumerate(ocr_lines):
                if desc_prefix not in line:
                    continue
                price_pattern = r'(?<!\d)' + re.escape(price_str) + r'(?!\d)'
                price_m = re.search(price_pattern, line)
                if price_m:
                    after_price = line[price_m.end():]
                    price_has_marker = bool(re.match(r'\s*[除※*]', after_price))
                    if not price_has_marker:
                        for j in range(idx + 1, min(idx + 3, len(ocr_lines))):
                            m = re.match(r'^(\d[\d,]*)\s*[*※]\s*$', ocr_lines[j].strip())
                            if m:
                                nearby_price = float(m.group(1).replace(',', ''))
                                if nearby_price != unit_price and nearby_price < unit_price * 5:
                                    item["unit_price"] = nearby_price
                                    item["total"] = nearby_price
                                break
                break
            continue

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


def _fix_discount_totals(items):
    """Ensure total = qty * unit_price - discount when discount is set."""
    for item in items:
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


def _fix_misattributed_discounts(items):
    """Reset total when LLM applied a discount that doesn't belong to this item."""
    for item in items:
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


def _detect_ocr_discounts(items, unified_text):
    """Detect discount lines in OCR text and apply to preceding items."""
    ocr_lines = unified_text.split('\n')
    for item in items:
        if not isinstance(item, dict) or (item.get("discount") or 0) > 0:
            continue
        desc = item.get("description", "")
        desc_prefix = desc[:4] if len(desc) >= 4 else desc
        if not desc_prefix:
            continue
        for li, ocr_line in enumerate(ocr_lines):
            if desc_prefix not in ocr_line:
                continue
            for offset in range(1, 6):
                if li + offset >= len(ocr_lines):
                    break
                next_line = ocr_lines[li + offset].strip()
                # Continuation lines (qty/multiplier info) are NOT a new item.
                is_qty_continuation = (
                    next_line.startswith('(')
                    or re.search(r'\d+\s*[個点]', next_line) is not None
                    or '単' in next_line
                )
                # Reached the next item: a CJK description line with no
                # price/discount/qty-info markers.
                if (re.search(r'[　-鿿]', next_line)
                        and '割引' not in next_line
                        and '値引' not in next_line
                        and '%' not in next_line
                        and '¥' not in next_line
                        and '￥' not in next_line
                        and not next_line.startswith('-')
                        and not is_qty_continuation):
                    break
                if '¥' in next_line and re.search(r'[\u3000-\u9fff]', next_line):
                    break
                if '割引' in next_line or '値引' in next_line:
                    rate_str = ""
                    discount_amount = 0
                    for k in range(li + offset, min(li + offset + 4, len(ocr_lines))):
                        kline = ocr_lines[k].strip()
                        # Rate may appear inline ("割引: 20%") or alone ("10%").
                        rate_match = re.search(r'(\d+)\s*%', kline)
                        if rate_match:
                            rate_str = rate_match.group(1) + '%'
                        # Amount line: accept "-38", "-¥24", "-￥24" with optional yen sign.
                        amt_match = re.match(r'^-\s*[¥￥]?\s*(\d[\d,.]*)\s*$', kline)
                        if amt_match:
                            amt_str = amt_match.group(1).replace(',', '')
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


def _normalize_taxes(extracted, unified_text, ocr_totals):
    """Normalize tax entries: canonical labels, clean rates, remove zero-amount."""
    if not extracted.get("taxes"):
        return
    subtotal = extracted.get("subtotal")
    total = extracted.get("total")
    tax_sum = sum(t.get("amount", 0) for t in extracted["taxes"])
    items_sum = sum(
        i.get("total", 0) for i in (extracted.get("line_items") or [])
        if isinstance(i, dict)
    ) or None
    for t in extracted["taxes"]:
        t["rate"] = normalize_tax_rate(t.get("rate", "unknown"))
        # Resolve "unknown" rate by searching OCR text for tax-context rate patterns
        if t["rate"] == "unknown":
            ocr_rates = set()
            for pattern in (
                r'外税\s*(\d+(?:\.\d+)?)\s*%',
                r'内税\s*(\d+(?:\.\d+)?)\s*%',
                r'(\d+(?:\.\d+)?)\s*%\s*(?:対象|消費税)',
            ):
                for m in re.finditer(pattern, unified_text):
                    candidate = normalize_tax_rate(m.group(1) + '%')
                    if candidate in VALID_TAX_RATES:
                        ocr_rates.add(candidate)
            if len(ocr_rates) == 1:
                t["rate"] = ocr_rates.pop()
        t["label"] = normalize_tax_label(
            t.get("label"), unified_text,
            subtotal=subtotal, total=total, tax_sum=tax_sum,
            items_sum=items_sum,
        )
    extracted["taxes"] = [
        t for t in extracted["taxes"]
        if t.get("amount", 0) != 0 or t.get("rate") == "0%"
    ]
    seen: set[tuple] = set()
    deduped = []
    for t in extracted["taxes"]:
        key = (t.get("rate"), t.get("label"), t.get("amount"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(t)
    extracted["taxes"] = deduped


def _recover_missing_items_from_gap(extracted, unified_text):
    """Add a missing line_item when the items_sum gap matches exactly one
    unaccounted ¥amount in the OCR text.

    Generic-purpose: applies to any receipt whose extracted items sum is
    short by a single OCR-visible price. Conservative: only fires when
    exactly one unmatched OCR price equals the gap (±2 yen) and a
    plausible description appears within 12 lines above it.
    """
    items = extracted.get("line_items") or []
    total = extracted.get("total")
    subtotal = extracted.get("subtotal")
    taxes = extracted.get("taxes") or []
    tax_sum = sum(t.get("amount", 0) for t in taxes)

    if not total or not items:
        return

    items_sum = sum(
        i.get("total", 0) for i in items if isinstance(i, dict)
    )

    # Skip when items already balance against either target — no missing item.
    items_match_total = abs(items_sum - total) <= 2
    items_match_subtotal = subtotal is not None and abs(items_sum - subtotal) <= 2
    if items_match_total or items_match_subtotal:
        return

    lines = unified_text.split('\n')

    # Collect OCR ¥amounts excluding summary lines
    ocr_prices: list[tuple[int, float]] = []
    for i, line in enumerate(lines):
        if _SKIP_PRICE_LINE.search(line):
            continue
        for m in re.finditer(r'[¥￥]\s*([\d,]+)', line):
            try:
                amt = float(m.group(1).replace(',', ''))
            except ValueError:
                continue
            if amt > 0:
                ocr_prices.append((i, amt))

    # Multiset diff: remove one OCR entry per extracted item amount
    item_amounts = [
        i.get("total", 0) for i in items if isinstance(i, dict)
    ]
    unmatched = list(ocr_prices)
    for amt in item_amounts:
        for j, (_idx, oa) in enumerate(unmatched):
            if abs(oa - amt) < 1:
                unmatched.pop(j)
                break

    # Exclude OCR prices that exactly match a printed tax amount — those
    # are tax values, not items. Without this guard, a printed '¥97' for an
    # 8% tax line gets recovered as a fake 97-yen item.
    tax_amts = {
        float(t.get("amount", 0))
        for t in taxes
        if isinstance(t, dict) and t.get("amount") not in (None, 0)
    }
    if tax_amts:
        unmatched = [(idx, amt) for idx, amt in unmatched
                     if amt not in tax_amts]

    # Try both targets: items add to total (内税) or to subtotal (外税).
    # Pre-normalize, the LLM-supplied tax label is unreliable, so test both
    # and only fire if exactly one yields a single matching unaccounted ¥.
    successful = []
    for try_target in (total, subtotal):
        if try_target is None:
            continue
        g = try_target - items_sum
        if g <= 0 or g > total:
            continue
        matches = [(idx, amt) for idx, amt in unmatched if abs(amt - g) <= 2]
        if len(matches) == 1:
            successful.append((try_target, matches[0]))

    if len(successful) != 1:
        return

    target, (price_line_idx, price) = successful[0]

    def _clean_candidate(text: str) -> str:
        """Strip price suffix, count markers, tax markers, and leading product
        codes from a description candidate."""
        text = text.strip()
        # Drop everything from the first ¥ onward (item-and-price merged lines)
        m = re.search(r'[¥￥]', text)
        if m:
            text = text[:m.start()].strip()
        # Drop trailing count markers like "1点", "2個", "3コ"
        text = re.sub(r'\s+[\d,]+\s*[点個コ]\s*$', '', text).strip()
        # Drop trailing tax markers
        text = re.sub(r'\s*[※\*非外]\s*$', '', text).strip()
        # Strip leading product/department code: 4+ digits, optional letters,
        # optional ')'. Only when the remainder still has Japanese content.
        m = re.match(r'^\d{4,}[A-Za-z]{0,3}\)?\s?(.+)$', text)
        if m and re.search(r'[ぁ-んァ-ン一-龥]', m.group(1)):
            text = m.group(1).strip()
        return text

    def _is_existing_desc(text: str) -> bool:
        return any(
            isinstance(o, dict)
            and (o.get("description") or "").strip() == text
            for o in items
        )

    def _is_valid_desc(text: str) -> bool:
        if not text or len(text) < 3:
            return False
        if text in _GENERIC_DESC_MARKERS:
            return False
        if _SKIP_PRICE_LINE.search(text):
            return False
        if re.match(r'^\d{12,}$', text):
            return False
        if re.match(r'^[\d,\s\-\(\)\.\*※軽除外]+$', text):
            return False
        if _JUNK_DESC_RE.search(text):
            return False
        if _HEADER_LINE_RE.search(text):
            return False
        # Skip lines without any Japanese (logos, store names, English-only)
        if not re.search(r'[ぁ-んァ-ン一-龥]', text):
            return False
        # Short fragments (<5 chars) are usually OCR garbage when adding a new
        # item — unless they start with a product code (e.g., "0011W) X").
        if len(text) < 5 and not re.match(r'^\d{3,}', text):
            return False
        if re.match(r'^単?\s*\d', text) and ('×' in text or 'x' in text or '個' in text):
            return False
        return True

    desc = None

    # First check the price line itself — rejoin_price_lines often merges
    # the item name with its price on a single line.
    line_text = lines[price_line_idx]
    cand = _clean_candidate(line_text)
    if _is_valid_desc(cand) and not _is_existing_desc(cand):
        desc = cand

    # Else search backward up to 15 lines, then forward up to 5 lines.
    # Prefer product-code-prefixed lines (e.g. "20060SAミタメスッキリ ロック")
    # since they're unambiguous item starts even when surrounded by OCR garbage.
    if not desc:
        candidates_idx = list(range(price_line_idx - 1, max(price_line_idx - 16, -1), -1))
        candidates_idx += list(range(price_line_idx + 1, min(price_line_idx + 6, len(lines))))

        # First pass: lines with a leading product code (e.g. "20060SA…").
        # Check the prefix on the raw line, then clean it for the description.
        for j in candidates_idx:
            raw = lines[j].strip()
            if not re.match(r'^\d{4,}', raw):
                continue
            cand = _clean_candidate(raw)
            if _is_valid_desc(cand) and not _is_existing_desc(cand):
                desc = cand
                break
        # Second pass: any valid candidate
        if not desc:
            for j in candidates_idx:
                cand = _clean_candidate(lines[j])
                if _is_valid_desc(cand) and not _is_existing_desc(cand):
                    desc = cand
                    break

    if not desc:
        return

    # Decide qty/unit_price: if the line above the price has "単X×N個" form,
    # use it for qty/unit_price; else default qty=1, unit_price=price.
    qty = 1
    unit_price = price
    for j in range(price_line_idx - 1, max(price_line_idx - 4, -1), -1):
        line = lines[j].strip()
        m = re.match(r'単?\s*(\d+)\s*[×x]\s*(\d+)\s*個?', line)
        if m:
            up = float(m.group(1))
            q = float(m.group(2))
            if abs(up * q - price) < 2:
                qty = int(q)
                unit_price = up
            break

    # Tax category: use majority rate from existing items, falling back to
    # the receipt's tax rates.
    tax_category = "0%"
    if items:
        from collections import Counter
        cats = Counter(
            i.get("tax_category") for i in items
            if isinstance(i, dict) and i.get("tax_category")
        )
        if cats:
            tax_category = cats.most_common(1)[0][0]
    elif taxes:
        tax_category = taxes[0].get("rate", "0%")

    new_item = {
        "description": desc,
        "qty": qty,
        "unit_price": unit_price,
        "total": price,
        "tax_category": tax_category,
        "discount": 0,
        "discount_rate": "",
    }

    # Insert at the OCR-order position: find the existing item whose price
    # appears after this one in the OCR text, and insert before it.
    insert_pos = len(extracted["line_items"])
    for idx, existing in enumerate(extracted["line_items"]):
        if not isinstance(existing, dict):
            continue
        e_total = existing.get("total", 0)
        if not e_total:
            continue
        existing_price_line = None
        for li, line in enumerate(lines):
            if _SKIP_PRICE_LINE.search(line):
                continue
            for m in re.finditer(r'[¥￥]\s*([\d,]+)', line):
                try:
                    amt = float(m.group(1).replace(',', ''))
                except ValueError:
                    continue
                if abs(amt - e_total) < 1:
                    existing_price_line = li
                    break
            if existing_price_line is not None:
                break
        if existing_price_line is not None and existing_price_line > price_line_idx:
            insert_pos = idx
            break

    extracted["line_items"].insert(insert_pos, new_item)


def _fix_items_from_subtotal(extracted, unified_text, ocr_totals):
    """Cross-check item totals against OCR subtotal; fix items whose nearby OCR price differs."""
    items = extracted.get("line_items")
    if not items:
        return
    subtotal = ocr_totals.get("subtotal")
    if subtotal is None:
        m = re.search(r'小\s*計', unified_text)
        if m:
            after = unified_text[m.end():]
            yen_m = re.search(r'[¥￥]\s*([\d,]+)', after[:80])
            if yen_m:
                subtotal = float(yen_m.group(1).replace(',', ''))
    if subtotal is None:
        return
    item_sum = sum(i.get("total", 0) for i in items if isinstance(i, dict))
    if abs(item_sum - subtotal) < 2:
        return
    ocr_lines = unified_text.split('\n')
    for item in items:
        if not isinstance(item, dict) or item.get("qty", 1) != 1:
            continue
        desc = item.get("description", "")
        desc_key = desc[:8] if len(desc) >= 8 else desc
        if not desc_key:
            continue
        for li, ocr_line in enumerate(ocr_lines):
            if desc_key not in ocr_line:
                continue
            for offset in range(0, 4):
                if li + offset >= len(ocr_lines):
                    break
                yen_m = re.search(r'[¥￥]\s*([\d,]+)', ocr_lines[li + offset])
                if yen_m:
                    ocr_price = float(yen_m.group(1).replace(',', ''))
                    old_total = item.get("total", 0)
                    if abs(ocr_price - old_total) > 1:
                        new_sum = item_sum - old_total + ocr_price
                        if abs(new_sum - subtotal) < abs(item_sum - subtotal):
                            item["unit_price"] = ocr_price
                            item["total"] = ocr_price
                            item_sum = new_sum
                    break
            break


def postprocess_receipt(
    extracted: dict,
    unified_text: str,
    ocr_conf: float,
    ocr_totals: dict,
    llm_conf: dict | None,
    model: str,
) -> dict:
    """Apply all receipt-specific post-processing to the LLM extraction."""
    _fix_company_name_merchant(extracted, unified_text)
    _apply_financial_overrides(extracted, ocr_totals, ocr_conf, llm_conf)
    _fix_date(extracted, unified_text)
    _fix_time(extracted, unified_text)
    _fix_payment_method(extracted, unified_text, ocr_conf, llm_conf)
    _fix_line_items(extracted, unified_text)
    _drop_phantom_from_tax_amount(extracted)
    _fix_priced_in_name_items(extracted, unified_text)
    _fix_items_from_subtotal(extracted, unified_text, ocr_totals)
    _recover_missing_items_from_gap(extracted, unified_text)
    _fix_digit_misread_items(extracted, unified_text)

    # Clear account_number when it's a masked card number suffix, not a real account
    acct = extracted.get("account_number")
    if acct and re.search(r'\*{2,}' + re.escape(str(acct)), unified_text):
        extracted["account_number"] = None

    # Tax categories
    if extracted.get("line_items"):
        rate_bases = extract_rate_bases(unified_text)
        breakdown_bases = ocr_totals.get('_breakdown_rate_bases', {})
        for rate, base in breakdown_bases.items():
            if rate not in rate_bases or rate_bases[rate] is None:
                rate_bases[rate] = base
        assign_tax_categories(extracted["line_items"], unified_text, ocr_totals, rate_bases,
                              extracted_taxes=extracted.get("taxes"))

    # Ensure tax entries exist for all assigned item categories
    if extracted.get("line_items") and extracted.get("taxes"):
        rate_sums: dict[str, float] = {}
        for item in extracted["line_items"]:
            if not isinstance(item, dict):
                continue
            cat = item.get("tax_category", "0%")
            if cat and cat != "0%":
                rate_sums[cat] = rate_sums.get(cat, 0) + (item.get("total") or 0)
        existing_rates = {t.get("rate") for t in extracted["taxes"]}
        existing_labels = [t.get("label", "") for t in extracted["taxes"] if t.get("label")]
        default_label = existing_labels[0] if existing_labels else None
        is_inclusive = default_label in ('内税', '消費税等') or (default_label or '').startswith('内')
        for cat in sorted(rate_sums):
            if cat not in existing_rates and rate_sums[cat] > 0:
                rate_pct = float(cat.replace('%', '')) / 100.0
                if is_inclusive:
                    computed_tax = round(rate_sums[cat] * rate_pct / (1 + rate_pct))
                else:
                    computed_tax = round(rate_sums[cat] * rate_pct)
                # Truth-file convention: omit tax entries that round to 0
                # (e.g., a レジ袋 at 5円 × 10% = 0.5 → 0).
                if computed_tax > 0:
                    extracted["taxes"].append({
                        "rate": cat,
                        "label": default_label,
                        "amount": computed_tax,
                    })

        # Drop tax entries for rates whose items-side computed tax rounds to 0
        # (e.g., LLM merged a `{rate: 10%, amount: 4}` entry by mis-reading a 4
        # yen レジ袋 as the tax amount). The truth-file convention omits these.
        kept = []
        for t in extracted["taxes"]:
            if not isinstance(t, dict):
                kept.append(t)
                continue
            r = t.get("rate")
            if r and r in rate_sums:
                try:
                    rate_pct = float(r.replace('%', '')) / 100.0
                except ValueError:
                    rate_pct = 0
                if rate_pct > 0:
                    if is_inclusive:
                        expected = round(rate_sums[r] * rate_pct / (1 + rate_pct))
                    else:
                        expected = round(rate_sums[r] * rate_pct)
                    if expected == 0:
                        continue  # drop — items at this rate produce 0 tax
            kept.append(t)
        extracted["taxes"] = kept

    # Drop any remaining tax entries with amount=0 (LLM-supplied unhandled).
    # Exempt the 0% / 非課税 entry: truth files keep it (rate '0%' may have
    # amount=0 since there's no tax to record on a non-taxable line).
    if extracted.get("taxes"):
        extracted["taxes"] = [
            t for t in extracted["taxes"]
            if not isinstance(t, dict)
            or (t.get("amount") or 0) > 0
            or t.get("rate") == "0%"
            or "非課税" in (t.get("label") or "")
        ]

    # Points used
    points = extract_points_used(unified_text)
    if points is not None:
        if should_override_field("points_used", ocr_conf, llm_conf) or extracted.get("points_used") is None:
            extracted["points_used"] = points
    elif extracted.get("points_used") is not None:
        has_points_evidence = bool(re.search(r'ポイント利用|ポイント値引', unified_text))
        if not has_points_evidence:
            extracted["points_used"] = None

    # Fix pre-tax item totals for inclusive-tax receipts
    if extracted.get("line_items") and extracted.get("total"):
        item_sum = sum(i.get("total", 0) for i in extracted["line_items"] if isinstance(i, dict))
        receipt_total = extracted["total"]
        items_fixed = False
        # Skip adjustment when taxes account for the difference (exclusive tax)
        tax_total = sum(t.get("amount", 0) for t in extracted.get("taxes", []))
        items_are_pretax = tax_total > 0 and abs(item_sum + tax_total - receipt_total) < 2
        if len(extracted["line_items"]) == 1 and abs(item_sum - receipt_total) > 1 and not items_are_pretax:
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
    _normalize_taxes(extracted, unified_text, ocr_totals)

    # Fix tax amounts when OCR taxes are missing
    if (extracted.get("taxes") and extracted.get("line_items")
            and extracted.get("total") and not ocr_totals.get("taxes")):
        rate_sums: dict[str, float] = {}
        for item in extracted["line_items"]:
            cat = item.get("tax_category", "0%")
            rate_sums[cat] = rate_sums.get(cat, 0) + (item.get("total") or 0)
        all_labels = [t.get("label", "") for t in extracted["taxes"]]
        all_inclusive_labels = all_labels and all(
            (lbl or '').startswith('内') or lbl == '消費税等' for lbl in all_labels)
        if all_inclusive_labels:
            for t in extracted["taxes"]:
                rate = t.get("rate", "0%")
                rate_pct = float(rate.replace('%', '')) / 100.0
                amt = t.get("amount", 0)
                cat_sum = rate_sums.get(rate, 0)
                if rate_pct > 0 and cat_sum > 0:
                    # When tax amount equals item sum, it's a base not a tax
                    if amt > 0 and abs(amt - cat_sum) < 2:
                        t["amount"] = round(cat_sum * rate_pct / (1 + rate_pct))
                    # For inclusive items (item_sum ≈ total), recompute from items
                    elif abs(sum(rate_sums.values()) - extracted["total"]) < 5:
                        computed = round(cat_sum * rate_pct / (1 + rate_pct))
                        if computed != amt:
                            t["amount"] = computed

    # Fallback: recompute tax amounts from OCR rate bases
    if extracted.get("taxes") and not ocr_totals.get("taxes"):
        rb = extract_rate_bases(unified_text)
        bb = ocr_totals.get('_breakdown_rate_bases', {})
        for rate, base in bb.items():
            if rate not in rb or rb[rate] is None:
                rb[rate] = base
        rb_sum = sum(v for v in rb.values() if v and v > 0)
        bases_are_inclusive = abs(rb_sum - (extracted.get("total") or 0)) < 5
        for t in extracted["taxes"]:
            rate = t.get("rate", "0%")
            rate_pct = float(rate.replace('%', '')) / 100.0
            base = rb.get(rate)
            if rate_pct > 0 and base and base > 0:
                if bases_are_inclusive:
                    computed = round(base * rate_pct / (1 + rate_pct))
                else:
                    computed = round(base * rate_pct)
                if abs(t.get("amount", 0) - computed) > 2:
                    t["amount"] = computed

    # Universal subtotal rule: subtotal = total - sum(taxes), regardless of
    # 内税 / 外税. Pre-tax base is the canonical definition; for 内税 receipts
    # this means subtotal != sum(line_items) (line items are post-tax) which is
    # expected and validated.
    #
    # Preserve an existing subtotal when it's close to the computed value —
    # this guards against 1-2 yen rounding flips when the tax was extracted
    # with a small rounding error. The printed subtotal is authoritative for
    # the receipt's own internal rounding choice.
    if extracted.get("total") is not None:
        tax_sum = sum(t.get("amount", 0) for t in extracted.get("taxes") or [])
        computed_sub = extracted["total"] - tax_sum
        if computed_sub >= 0:
            existing_sub = extracted.get("subtotal")
            close_to_computed = (
                existing_sub is not None
                and abs(existing_sub - computed_sub) <= 5
            )
            close_to_pretax_via_tax_only = (
                existing_sub is not None
                and tax_sum > 0
                and abs(existing_sub + tax_sum - extracted["total"]) <= 5
            )
            if close_to_computed or close_to_pretax_via_tax_only:
                pass  # keep the printed/extracted value
            else:
                extracted["subtotal"] = computed_sub

    _extract_fuel_usage(extracted, unified_text)

    return extracted
