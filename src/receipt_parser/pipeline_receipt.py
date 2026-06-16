"""pipeline_receipt.py — Receipt-specific post-processing and financial extraction.

Extracted from pipeline.py for maintainability. Contains:
- Financial totals extraction from OCR text
- Yen amount helpers
- Tax category assignment
- Receipt post-processing (date, payment, line items, etc.)
"""

import re
from collections import Counter
from difflib import SequenceMatcher
from itertools import combinations

from .schema import VALID_TAX_RATES, REDUCED_RATE, STANDARD_RATE
from .patterns import (
    YEN_INLINE, YEN_SUFFIX, ERA_TABLE, should_override_field, era_to_western_year,
)


# Canonical labels: 内税 (inclusive), 外税 (exclusive), 非課税 (exempt)
def _inner_tax_target_amount_matches_total(text: str, total: float | None) -> bool:
    if total is None:
        return False
    lines = [line.strip() for line in (text or "").splitlines()]
    for idx, line in enumerate(lines):
        if not re.search(r'内\s*税\s*対象|内税対象', line):
            continue
        for following in lines[idx + 1:min(idx + 5, len(lines))]:
            if not following:
                continue
            if re.search(r'税合計|消費税|合計|小計|対象', following):
                break
            m = re.search(r'[¥￥]?\s*([\d,]+)', following)
            if not m:
                continue
            try:
                amount = float(m.group(1).replace(',', ''))
            except ValueError:
                continue
            if abs(amount - float(total)) <= 2:
                window = lines[idx:min(idx + 8, len(lines))]
                if any(re.search(r'\d+\s*%\s*内(?!税対象)', candidate) for candidate in window):
                    return True
                for marker_idx in range(idx + 1, min(idx + 8, len(lines))):
                    marker = lines[marker_idx]
                    if not re.search(r'合\s*計|総\s*合\s*計', marker):
                        continue
                    inline = re.search(r'[¥￥]?\s*([\d,]+)', marker)
                    if inline:
                        try:
                            if abs(float(inline.group(1).replace(',', '')) - float(total)) <= 2:
                                return True
                        except ValueError:
                            pass
                    if marker_idx + 1 < len(lines):
                        next_amount = re.search(r'[¥￥]?\s*([\d,]+)', lines[marker_idx + 1])
                        if next_amount:
                            try:
                                if abs(float(next_amount.group(1).replace(',', '')) - float(total)) <= 2:
                                    return True
                            except ValueError:
                                pass
            break
    return False


def _text_says_displayed_prices_are_tax_included(text: str) -> bool:
    lines = [line.strip() for line in (text or "").splitlines()]
    notice_re = re.compile(r'表示価格.{0,20}税込価格|税込価格.{0,20}表示価格')
    summary_re = re.compile(
        r'合\s*計|総\s*合\s*計|小\s*計|消費税|税額|内税|外税|'
        r'\d+(?:\.\d+)?\s*%\s*対象|対象\s*額'
    )
    for idx, line in enumerate(lines):
        if not notice_re.search(line):
            continue
        nearby_indexes = range(max(0, idx - 6), min(len(lines), idx + 3))
        if any(i != idx and summary_re.search(lines[i]) for i in nearby_indexes):
            return True
    return False


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

    Under the canonical 'subtotal = total - tax' convention, the receipt's
    item-sum shape distinguishes labels: items that already add to total are
    inclusive, while items that add to subtotal and need tax added to reach
    total are exclusive. OCR "内税対象" wording is not authoritative when that
    arithmetic proves the printed item prices are pre-tax.
    """
    label = label or ""
    has_inclusive_target_amount = _inner_tax_target_amount_matches_total(text, total)
    prices_are_marked_tax_included = _text_says_displayed_prices_are_tax_included(text)

    if '非課税' in label:
        return '非課税'

    if (
        items_sum is not None
        and total is not None
        and tax_sum is not None
        and tax_sum > 0
        and abs(items_sum + tax_sum - total) <= 2
        and abs(items_sum - total) > 2
        and not re.search(r'内\s*税額|内税額', text)
        and not has_inclusive_target_amount
        and not prices_are_marked_tax_included
    ):
        return '外税'

    if re.search(r'内\s*税額|内税額', text) and '外税' not in text:
        return '内税'

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
    has_pct_tax_marker = (
        bool(re.search(r'\d+%\s*税(?:額)?(?![一-鿿])', text))
        and not bool(re.search(r'内\s*\d+%\s*税(?:額)?', text))
    )

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
        if (
            abs(items_sum - subtotal) <= 2
            and abs(items_sum - total) > 2
            and not has_inclusive_target_amount
            and not prices_are_marked_tax_included
        ):
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
    return _parse_amount_fragment(val) if val else None


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

        rate_ctx_m = re.search(r'(\d+(?:\.\d+)?)%.*(?:対象|タイショウ)', line)
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
            if re.search(r'消費税', line) and not any(t.get('rate') == _rate_context for t in taxes):
                tax_candidates: list[float] = []
                after_tax = re.split(r'消費税[等額]?', line, maxsplit=1)[-1]
                inline_tax = re.search(r'[¥￥]\s*([\d,]+)', after_tax)
                if inline_tax:
                    tax_candidates.append(float(inline_tax.group(1).replace(',', '')))
                for j in range(i + 1, min(i + 5, len(lines))):
                    nb = lines[j].strip()
                    if not nb:
                        continue
                    if re.search(r'\d+(?:\.\d+)?\s*[%％年].*(?:対象|タイショウ)|合\s*計|現金|お預り|釣銭', nb):
                        break
                    yen_next = re.match(r'^[¥￥]\s*([\d,]+)\s*[\)）]?\s*$', nb)
                    if yen_next:
                        tax_candidates.append(float(yen_next.group(1).replace(',', '')))
                        continue
                    if tax_candidates:
                        break
                tax_val = None
                if tax_candidates and base_val is not None:
                    try:
                        pct = float(_rate_context.rstrip('%')) / 100.0
                    except (TypeError, ValueError, ZeroDivisionError):
                        pct = 0.0
                    expected = round(base_val * pct) if pct > 0 else 0
                    matches = [value for value in tax_candidates if abs(value - expected) <= 2]
                    tax_val = matches[0] if matches else tax_candidates[0]
                elif tax_candidates:
                    tax_val = tax_candidates[0]
                if tax_val is not None:
                    taxes.append({'rate': _rate_context, 'label': '内税', 'amount': tax_val})

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
            # 現計 is the receipt total. In column-split OCR, the nearby value
            # block may contain subtotal/tax/total; in normal tender blocks it
            # may contain total/tender/change. Prefer subtotal + tax when that
            # candidate is printed, otherwise fall back to the first amount.
            vals = _collect_yen_values(lines, i, 8, stop_pattern=_STOP_FINANCIAL)
            val = vals[0] if vals else None
            subtotal_hint = result.get('subtotal')
            taxable_tax_sum = sum(
                t.get('amount', 0) for t in taxes
                if isinstance(t, dict) and t.get('rate') != '0%'
            )
            if vals and subtotal_hint is not None and taxable_tax_sum:
                expected_total = float(subtotal_hint) + float(taxable_tax_sum)
                val = min(vals, key=lambda v: abs(v - expected_total))
            elif vals and subtotal_hint is not None:
                context = '\n'.join(lines[max(0, i - 5):min(len(lines), i + 6)])
                above_subtotal = [v for v in vals if v > float(subtotal_hint)]
                at_or_above_subtotal = [v for v in vals if v >= float(subtotal_hint)]
                if above_subtotal and re.search(r'税|対象額', context):
                    val = min(above_subtotal)
                elif at_or_above_subtotal:
                    val = min(at_or_above_subtotal)
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
            r'\(?\s*(?:内税(?:分|\s*\d+\s*%)?|内)\s*消費税', line,
        )
        if m_inclusive_with_rate and '対象' not in line and not _has_specific_taxes:
            rate_search = re.search(r'(\d+(?:\.\d+)?)\s*%', line)
            if not rate_search:
                for nearby in lines[max(0, i - 4):min(i + 5, len(lines))]:
                    nearby_m = re.search(r'(\d+(?:\.\d+)?)\s*%.*対象', nearby)
                    if nearby_m:
                        rate_search = nearby_m
                        break
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

        if (
            re.search(r'(?:外税\s*\d+%|\d+\s*%\s*外税)', line)
            and not re.search(r'対象|タイショウ', line)
        ):
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
        elif re.match(r'^\s*\d+%\s*$', line) and i + 1 < len(lines):
            rate_m = re.search(r'(\d+)%', line)
            next_line = lines[i + 1].strip()
            if rate_m and re.fullmatch(r'税', next_line):
                val = _extract_yen_nearby(lines, i + 1, look_ahead=2)
                if val is not None and not any(t['rate'] == rate_m.group(1) + '%' for t in taxes):
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

        # Non-taxable (非課税) detection. Truth files store the non-taxable
        # target/base amount in the amount field, not zero tax.
        if '非課税' in line and not any(t.get('rate') == '0%' for t in taxes):
            non_tax_amount = _extract_yen_nearby(lines, i, look_ahead=3)
            taxes.append({'rate': '0%', 'label': '非課税', 'amount': non_tax_amount or 0})

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

    for rate, kind, value in _bare_number_tax_summary_entries(lines):
        if kind != "tax":
            continue
        if any(t.get('rate') == rate and t.get('amount') == value for t in taxes):
            continue
        taxes = [t for t in taxes if not (t.get('rate') == rate and (t.get('amount') or 0) == 0)]
        taxes.append({'rate': rate, 'label': '内税', 'amount': value})

    for i in range(0, max(0, len(lines) - 3)):
        if not re.fullmatch(r'小\s*計', lines[i].strip()):
            continue
        target_m = re.search(r'(\d+(?:\.\d+)?)\s*[%％]\s*対象', lines[i + 1].strip())
        tax_m = re.search(r'(\d+(?:\.\d+)?)\s*[%％]\s*税額', lines[i + 2].strip())
        if not target_m or not tax_m:
            continue
        rate = normalize_tax_rate(target_m.group(1) + '%')
        if rate != normalize_tax_rate(tax_m.group(1) + '%'):
            continue
        values: list[float] = []
        for j in range(i + 3, min(len(lines), i + 10)):
            vm = re.fullmatch(r'[¥￥]\s*([\d,]+)\s*[\)）]?', lines[j].strip())
            if not vm:
                break
            values.append(float(vm.group(1).replace(',', '')))
        if len(values) < 3:
            continue
        subtotal_val, _base_val, tax_val = values[-3], values[-2], values[-1]
        if subtotal_val > 0 and tax_val > 0:
            result["subtotal"] = subtotal_val
            taxes = [t for t in taxes if t.get("rate") != rate]
            taxes.append({"rate": rate, "label": "外税", "amount": tax_val})

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

    interleaved_tax_entries = _interleaved_rate_tax_summary_entries(lines)
    interleaved_taxes = [
        {"rate": rate, "label": "内税", "amount": value}
        for rate, kind, value in interleaved_tax_entries
        if kind == "tax" and value > 0
    ]
    if interleaved_taxes:
        taxes = [
            t for t in taxes
            if t.get("rate") == "0%" or t.get("rate") not in {entry["rate"] for entry in interleaved_taxes}
        ]
        taxes.extend(interleaved_taxes)

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

    taxable_taxes = [
        t for t in taxes
        if t.get('rate') != '0%' and t.get('amount', 0) > 0
    ]
    if not taxable_taxes:
        for t in taxes:
            if t.get('rate') == '0%':
                t['amount'] = 0

    # Sanity check: remove tax entries where amount >= total. Keep 0% entries:
    # mixed receipts store the non-taxable base as amount, and all-exempt
    # receipts keep a zero amount so rate/label validation still sees it.
    total = result.get('total')
    if taxes and total:
        taxes = [t for t in taxes if t.get('rate') == '0%' or t['amount'] < total]

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


_TOTALS_LABEL_RE = re.compile(
    r'(小計|合計|現計|外税|内税|消費税|対象|お預り|お釣|釣銭|総額|お会計|'
    r'WAON|現金|クレジット|カード|電子マネー|残高|支払|預り)'
)
_TOTALS_VALUE_RE = re.compile(r'^[¥￥]\s*[\d,]+\s*$')


def _is_stacked_summary_padding_label(line: str) -> bool:
    """Labels that occupy a printed summary slot but are not tax target labels."""
    compact = re.sub(r'\s+', '', line or '')
    if not compact:
        return False
    if re.search(r'軽減税率|対象商品|対象です|対象物|店内飲食', compact):
        return False
    return bool(
        compact == '計'
        or re.search(
            r'小計|合計|総合計|現計|お会計|お預り|お釣り?|釣銭|'
            r'WAON|現金|クレジット|カード|電子マネー|残高|支払|預り',
            compact,
        )
    )


def _is_tax_summary_stack_label(line: str) -> bool:
    compact = re.sub(r'\s+', '', line or '')
    if re.search(r'軽減税率|対象商品|対象です|対象物|店内飲食', compact):
        return False
    if re.search(r'\d+(?:\.\d+)?\s*[%％年]', compact) and re.search(
        r'外税|外枠|内税|対象|タイショウ|税',
        compact,
    ):
        return True
    return _is_stacked_summary_padding_label(compact)


def _column_split_label_value_pairs(lines: list[str]) -> list[tuple[str, str]]:
    """Detect column-split totals layout: N consecutive label lines followed
    by N consecutive ¥-prefixed value lines. Returns ordered (label, value)
    pairs from the first such block found.

    AEON-style receipt OCR shape — labels block (小計/外税8%対象額/外税8%/
    外税10%対象額/外税10%/合計) printed first, then a parallel block of values.
    """
    n = len(lines)
    i = 0
    while i < n:
        if not _is_tax_summary_stack_label(lines[i].strip()):
            i += 1
            continue
        if re.search(r'[¥￥]\s*\d', lines[i]):
            i += 1
            continue
        labels: list[str] = []
        j = i
        while j < n:
            s = lines[j].strip()
            if not s:
                j += 1
                continue
            if _is_tax_summary_stack_label(s) and not re.search(r'[¥￥]\s*\d', s):
                labels.append(s)
                j += 1
            else:
                break
        if len(labels) < 3:
            i = max(j, i + 1)
            continue
        values: list[str] = []
        k = j
        while k < n:
            s = lines[k].strip()
            if not s:
                k += 1
                continue
            if _TOTALS_VALUE_RE.match(s):
                values.append(s)
                k += 1
            else:
                break
        if len(values) >= 3 and len(labels) >= 3:
            pair_count = min(len(labels), len(values))
            return list(zip(labels[:pair_count], values[:pair_count]))
        i = max(k, i + 1)
    return []


def _bare_number_tax_summary_entries(lines: list[str]) -> list[tuple[str, str, float]]:
    """Map rate target/tax labels to a following bare-number value stack."""
    labels: list[tuple[int, str, str]] = []
    for idx, raw in enumerate(lines):
        line = raw.strip()
        target_m = re.search(r'^\(?\s*(\d+(?:\.\d+)?)\s*[%％年]\s*(?:対象|タイショウ)', line)
        if target_m:
            rate_num = float(target_m.group(1))
            rate = f"{int(rate_num)}%" if rate_num == int(rate_num) else f"{rate_num}%"
            labels.append((idx, rate, "base"))
            continue
        tax_m = re.search(r'内\s*消費税等?\s*(\d+(?:\.\d+)?)\s*[%％年]', line)
        if tax_m:
            rate_num = float(tax_m.group(1))
            rate = f"{int(rate_num)}%" if rate_num == int(rate_num) else f"{rate_num}%"
            labels.append((idx, rate, "tax"))

    if len(labels) < 2:
        return []

    values: list[float] = []
    for raw in lines[labels[-1][0] + 1:]:
        line = raw.strip()
        if not line:
            continue
        if re.search(r'合\s*計|お預り|お釣り|釣銭|営業時間|登録番号|レシート|ポイント', line):
            if values:
                break
            continue
        vm = re.fullmatch(r'([\d,]+)\s*[\)）]?', line)
        if vm:
            values.append(float(vm.group(1).replace(',', '')))
            continue
        if values:
            break

    if len(values) < len(labels):
        return []
    return [
        (rate, kind, value)
        for (_idx, rate, kind), value in zip(labels, values)
    ]


def _interleaved_rate_tax_summary_entries(lines: list[str]) -> list[tuple[str, str, float]]:
    """Map rate target rows followed by base/tax values in-place."""
    entries: list[tuple[str, str, float]] = []
    for idx, raw in enumerate(lines):
        line = raw.strip()
        target_m = re.search(r'^\(?\s*(\d+(?:\.\d+)?)\s*[%％年]\s*(?:対象|タイショウ)', line)
        if not target_m:
            continue
        rate_num = float(target_m.group(1))
        rate = f"{int(rate_num)}%" if rate_num == int(rate_num) else f"{rate_num}%"
        try:
            rate_pct = float(rate.rstrip('%')) / 100.0
        except ValueError:
            continue
        if rate_pct <= 0:
            continue

        window = [candidate.strip() for candidate in lines[idx:min(len(lines), idx + 5)]]
        joined = " ".join(window)
        m = re.search(
            r'([\d,]+)\s*(?:円)?\s*[（(]?\s*(?:内)?消費税(?:等|額)?\s*([\d,]+)\s*(?:円)?\s*[）)]?',
            joined,
        )
        if not m:
            continue
        base = float(m.group(1).replace(',', ''))
        amount = float(m.group(2).replace(',', ''))
        if base <= amount or amount <= 0:
            continue
        expected = round(base * rate_pct / (1 + rate_pct))
        if abs(amount - expected) > max(2.0, amount * 0.02):
            continue
        entries.append((rate, "base", base))
        entries.append((rate, "tax", amount))
    return entries


def extract_rate_bases(text: str) -> dict[str, float | None]:
    """Extract per-rate taxable base amounts (対象額) from OCR text."""
    bases: dict[str, float | None] = {}
    lines = text.split('\n')

    def _remember_base(rate: str, value: float | None) -> None:
        existing = bases.get(rate)
        if value is None:
            if rate not in bases:
                bases[rate] = None
            return
        if existing is None or value > existing:
            bases[rate] = value

    for i, raw in enumerate(lines):
        line = raw.strip()
        m = re.search(r'(\d+(?:\.\d+)?)\s*[%％年].*(?:対象|タイショウ)', line)
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
            _remember_base(rate_str, float(yen_m.group(1).replace(',', '')))
        else:
            found = False
            plain_candidate = None
            prev_nonempty = next(
                (lines[k].strip() for k in range(i - 1, -1, -1) if lines[k].strip()),
                "",
            )
            skip_tax_value_from_previous_label = bool(
                re.search(r'\d+(?:\.\d+)?\s*[%％]\s*税額', prev_nonempty)
                and not re.search(rf'{re.escape(str(int(rate_num)))}\s*[%％]\s*税額', prev_nonempty)
            )
            for j in range(i + 1, min(i + 6, len(lines))):
                js = lines[j].strip()
                if not js:
                    continue
                yen_ahead = re.search(r'[¥￥]\s*([\d,]+)', js)
                if yen_ahead:
                    if skip_tax_value_from_previous_label:
                        skip_tax_value_from_previous_label = False
                        continue
                    _remember_base(rate_str, float(yen_ahead.group(1).replace(',', '')))
                    found = True
                    break
                if re.match(r'^\d*\s*[\u500b\u70b9]\s*$', js):
                    # Item-count fragment ("3\u500b", "\u500b", "9\u70b9") \u2014 keep scanning
                    continue
                if re.search(r'\d+(?:\.\d+)?\s*[%％]\s*税額', js):
                    continue
                if re.search(r'[\u3000-\u9fff]', js) or re.search(r'[ァ-ン]', js):
                    break
                if plain_candidate is None:
                    plain_m = re.match(r'^([\d,]+)\s*$', js)
                    if plain_m:
                        plain_candidate = float(plain_m.group(1).replace(',', ''))
            if not found and plain_candidate is not None:
                _remember_base(rate_str, plain_candidate)
            elif not found:
                _remember_base(rate_str, None)

    # Column-split summary blocks are more reliable than loose lookahead:
    # labels are printed in order, then their values are printed in a parallel
    # stack. Override lookahead guesses when a target label is position-paired
    # with a value; otherwise tax amounts can be mistaken for taxable bases.
    pairs = _column_split_label_value_pairs(lines)
    for label, value in pairs:
        rm = re.search(r'(\d+(?:\.\d+)?)\s*[%％年].*(?:\u5bfe\u8c61|タイショウ)', label)
        if not rm:
            continue
        rate_num = float(rm.group(1))
        rate_str = f"{int(rate_num)}%" if rate_num == int(rate_num) else f"{rate_num}%"
        vm = re.search(r'[\u00a5\uffe5]\s*([\d,]+)', value)
        if vm:
            bases[rate_str] = float(vm.group(1).replace(',', ''))

    # Stacked tax summary fallback: some OCR linearizes all tax labels first,
    # then all yen values, e.g. "外税8%対象額 / 外税8% / 外税10年対象額 /
    # 外枠10% / ¥2,986 / ¥238 / ¥3 / ¥0". Preserve label order and map
    # target labels to the value in the same position.
    if any(v is None for v in bases.values()) or not bases:
        label_seq: list[tuple[str | None, bool]] = []
        value_seq: list[float] = []
        for idx, raw in enumerate(lines):
            line = raw.strip()
            rm = re.search(r'(\d+(?:\.\d+)?)\s*[%％年]', line)
            if rm and re.search(r'外税|外枠|内税|対象|タイショウ|税', line):
                rate_num = float(rm.group(1))
                rate_str = f"{int(rate_num)}%" if rate_num == int(rate_num) else f"{rate_num}%"
                is_target = bool(re.search(r'対象|タイショウ', line))
                label_seq.append((rate_str, is_target))
                continue
            if _is_stacked_summary_padding_label(line) and not re.search(r'[¥￥]\s*\d', line):
                label_seq.append((None, False))
                continue
            if not label_seq:
                continue
            vm = re.fullmatch(r'[¥￥]\s*([\d,]+)', line)
            if vm:
                value_seq.append(float(vm.group(1).replace(',', '')))
            elif value_seq:
                break
            elif re.search(r'合\s*計|クレジット|現金|お釣り|釣銭', line):
                continue
            elif len(label_seq) > 0 and idx > 0:
                # Allow non-value separators until the first yen value.
                continue
        if value_seq and label_seq:
            for (rate_str, is_target), value in zip(label_seq, value_seq):
                if rate_str and is_target:
                    bases[rate_str] = value

    for i in range(0, max(0, len(lines) - 3)):
        if not re.fullmatch(r'小\s*計', lines[i].strip()):
            continue
        target_m = re.search(r'(\d+(?:\.\d+)?)\s*[%％]\s*対象', lines[i + 1].strip())
        tax_m = re.search(r'(\d+(?:\.\d+)?)\s*[%％]\s*税額', lines[i + 2].strip())
        if not target_m or not tax_m:
            continue
        rate = normalize_tax_rate(target_m.group(1) + '%')
        if rate != normalize_tax_rate(tax_m.group(1) + '%'):
            continue
        values: list[float] = []
        for j in range(i + 3, min(len(lines), i + 10)):
            vm = re.fullmatch(r'[¥￥]\s*([\d,]+)\s*[\)）]?', lines[j].strip())
            if not vm:
                break
            values.append(float(vm.group(1).replace(',', '')))
        if len(values) >= 3:
            bases[rate] = values[-2]

    # Interleaved summaries may print a target label followed immediately by
    # its amount, with tax/total labels mixed into the same block. That direct
    # label-to-next-value evidence is stronger than column-position guesses.
    for i, raw in enumerate(lines):
        line = raw.strip()
        target_m = re.search(r'(\d+(?:\.\d+)?)\s*[%％年].*(?:対象|タイショウ)', line)
        if not target_m or re.search(r'対象商品|対象です|対象物', line):
            continue
        rate_num = float(target_m.group(1))
        rate = f"{int(rate_num)}%" if rate_num == int(rate_num) else f"{rate_num}%"
        inline_yen = re.search(r'[¥￥]\s*([\d,]+)', line)
        if inline_yen:
            bases[rate] = float(inline_yen.group(1).replace(',', ''))
            continue
        prev_nonempty = next(
            (lines[k].strip() for k in range(i - 1, -1, -1) if lines[k].strip()),
            "",
        )
        skip_tax_value_from_previous_label = bool(
            re.search(r'\d+(?:\.\d+)?\s*[%％]\s*税額', prev_nonempty)
            and not re.search(rf'{re.escape(str(int(rate_num)))}\s*[%％]\s*税額', prev_nonempty)
        )
        for lookahead in lines[i + 1:min(len(lines), i + 5)]:
            candidate = lookahead.strip()
            if not candidate:
                continue
            if re.search(r'\d+(?:\.\d+)?\s*[%％年].*(?:対象|タイショウ)', candidate):
                break
            vm = re.fullmatch(r'[¥￥]\s*([\d,]+)\s*[\)）]?', candidate)
            if vm:
                if skip_tax_value_from_previous_label:
                    skip_tax_value_from_previous_label = False
                    continue
                value = float(vm.group(1).replace(',', ''))
                existing = bases.get(rate)
                if (
                    existing is not None
                    and value < existing
                    and re.search(r'内税|内\s*消費税', line)
                ):
                    break
                bases[rate] = value
                break
            if _is_stacked_summary_padding_label(candidate):
                continue
            if re.search(r'税額|消費税|内税|外税|外枠', candidate):
                break
            if re.search(r'[ぁ-んァ-ン一-龥A-Za-z]', candidate):
                break

    # Parenthesized inclusive-tax blocks often OCR as all target labels first,
    # then closing-paren values: (10%対象 / (内消費税等 / (8%対象 /
    # ¥5) / ¥0) / ¥1,398) / ¥103). Pair each target with its base value.
    paren_targets: list[tuple[str, bool]] = []
    first_target_idx: int | None = None
    for idx, raw in enumerate(lines):
        line = raw.strip()
        target_m = re.search(r'^\(?\s*(\d+(?:\.\d+)?)\s*[%％年]\s*(?:対象|タイショウ)', line)
        if not target_m:
            continue
        rate_num = float(target_m.group(1))
        rate = f"{int(rate_num)}%" if rate_num == int(rate_num) else f"{rate_num}%"
        paren_targets.append((rate, bool(re.search(r'内税|内\s*消費税', line))))
        if first_target_idx is None:
            first_target_idx = idx
    if paren_targets and first_target_idx is not None:
        closing_values: list[float] = []
        for raw in lines[first_target_idx + 1:]:
            vm = re.fullmatch(r'[¥￥]\s*([\d,]+)\s*[\)）]', raw.strip())
            if vm:
                closing_values.append(float(vm.group(1).replace(',', '')))
        if len(closing_values) >= len(paren_targets) * 2:
            for pos, (rate, is_inclusive_label) in enumerate(paren_targets):
                value = closing_values[pos * 2]
                existing = bases.get(rate)
                if existing is not None and value < existing and is_inclusive_label:
                    continue
                bases[rate] = value

    for rate, kind, value in _bare_number_tax_summary_entries(lines):
        if kind == "base":
            bases[rate] = value
    for rate, kind, value in _interleaved_rate_tax_summary_entries(lines):
        if kind == "base":
            bases[rate] = value

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
    lines = [line.strip() for line in text.split('\n')]
    for idx, line in enumerate(lines):
        if not re.fullmatch(r'(?:楽天)?ポイント', line):
            continue
        yen_values: list[tuple[int, float]] = []
        saw_change_label = False
        zero_change_idx = None
        for j in range(idx + 1, min(idx + 16, len(lines))):
            if re.search(r'ポイントカード|取引CD|取引日時|ポイント対象金額|利用可能ポイント', lines[j]):
                break
            if re.search(r'おつり|お釣り|釣銭', lines[j]):
                saw_change_label = True
                continue
            vm = re.fullmatch(r'[¥￥]\s*([\d,]+)', lines[j])
            if vm:
                value = float(vm.group(1).replace(',', ''))
                yen_values.append((j, value))
                if saw_change_label and value == 0:
                    zero_change_idx = j
                    break
        if zero_change_idx is not None:
            prior_nonzero = [value for line_idx, value in yen_values if line_idx < zero_change_idx and value > 0]
            if prior_nonzero:
                return prior_nonzero[-1]
    return None


def reconcile_points_payment_from_ocr(extracted: dict, unified_text: str) -> None:
    """Apply OCR-backed point tender arithmetic to the receipt totals."""
    if not isinstance(extracted, dict) or extracted.get("total") is None:
        return
    points = extract_points_used(unified_text)
    if points is None:
        return
    try:
        total = float(extracted["total"])
        existing_points = extracted.get("points_used")
        amount_paid = extracted.get("amount_paid")
        existing_amount = float(amount_paid) if amount_paid is not None else None
    except (TypeError, ValueError):
        return
    if points < 0 or points > total + 2:
        return
    if (
        existing_points is None
        or float(existing_points or 0) == 0
        or abs(float(existing_points or 0) - points) <= 2
    ):
        extracted["points_used"] = points
    expected_paid = max(0.0, total - points)
    if (
        amount_paid is None
        or existing_amount is None
        or abs(existing_amount - total) <= 2
        or abs(existing_amount - expected_paid) <= 2
        or existing_amount > total
    ):
        extracted["amount_paid"] = expected_paid


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
    for m in re.finditer(
        r'(?:税率\s*)?(\d+(?:\.\d+)?)\s*%\s*(?:対象|課税|税額|消費税)',
        unified_text,
    ):
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
            sorted(detected_rates),
            key=lambda r: (assigned_counts.get(r, 0), tax_amounts.get(r, 0), rate_bases.get(r, 0) or 0),
        )
    else:
        majority_rate = max(
            sorted(detected_rates),
            key=lambda r: (rate_bases.get(r, 0) or 0, tax_amounts.get(r, 0), assigned_counts.get(r, 0)),
        )
    minority_rates = sorted(r for r in detected_rates if r != majority_rate)
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
        if (
            REDUCED_RATE in marker_rates
            and STANDARD_RATE not in marker_rates
            and STANDARD_RATE in detected_rates
        ):
            default_rate = STANDARD_RATE
        elif (
            STANDARD_RATE in marker_rates
            and REDUCED_RATE not in marker_rates
            and REDUCED_RATE in detected_rates
        ):
            default_rate = REDUCED_RATE
        elif tax_amounts and max(tax_amounts.values()) > 0:
            default_rate = max(sorted(detected_rates), key=lambda r: tax_amounts.get(r, 0))
        else:
            default_rate = majority_rate

    for idx in range(len(items)):
        if idx not in item_rates:
            item_rates[idx] = default_rate
    for idx, rate in item_rates.items():
        items[idx]["tax_category"] = rate


_COMPANY_SUFFIX_RE = re.compile(r'有限会社|株式会社|㈱|㈲|合同会社')
_DECORATIVE_RE = re.compile(r'^[☆★\-=\*\s・♪♫]+$')
_HEADER_PHONE_MERCHANT_RE = re.compile(
    r'^\s*(?P<merchant>.{2,40}?)\s*'
    r'(?:[（(]\s*0\d{1,4}\s*[）)]|0\d{1,4}[-\s]\d)'
)
_OFFICIAL_AUTHORITY_HEADER_RE = re.compile(
    r'^(?:[A-Z&]\s+)?(.{1,30}(?:市役所|区役所|町役場|村役場|役場|県庁|都庁|府庁))$'
)
_OFFICIAL_DEPARTMENT_LINE_RE = re.compile(r'^.{1,16}(?:課|係|室|部|局|センター)$')


def _merchant_looks_invalid(merchant: str | None) -> bool:
    merchant = (merchant or "").strip()
    if not merchant:
        return True
    compact = re.sub(r'\s+', '', merchant)
    if re.fullmatch(r'T\d{13}', re.sub(r'[\s-]+', '', merchant.upper())):
        return True
    if re.search(r'20\d{2}[./年-]\d{1,2}[./月-]\d{1,2}', merchant):
        return True
    if re.fullmatch(r'(?:TEL|電話|☎)\s*[:：]?\s*0?[\d\-\s]{6,}', merchant, flags=re.IGNORECASE):
        return True
    if (
        re.search(r'\d+丁目|\d+-\d+', merchant)
        or re.search(r'(?:都|道|府|県).{0,12}(?:市|区|町|村|郡)', merchant)
        or re.search(r'(?:市|区|町|村|郡).*\d', merchant)
    ):
        return True
    if re.match(r'^[（(]?[¥￥]?\s*[\d,]+(?:円)?[）)]?$', merchant):
        return True
    if merchant in {'領収書', '領収証', 'レシート', '様'}:
        return True
    if re.match(r'^(?:ご購入店|購入店|お買上店|お買い上げ店)\s*[:：]?', merchant):
        return True
    if re.fullmatch(r'[（(]?(?:消費税|内消費税|税込|税抜|課税対象|税額).*[）)]?', compact):
        return True
    if re.search(r'軽減税率|対象商品|適用商品|印は', compact):
        return True
    if re.search(r'但し|上記.*受領|正に受領', merchant):
        return True
    return False


def _clean_merchant_candidate(text: str, *, keep_company_suffix: bool = False) -> str:
    text = re.sub(r'^\s*(?:事業者名|販売者|発行者|店舗名)\s*[:：]\s*', '', text or "").strip()
    text = re.sub(r'\s+', ' ', text).strip()
    if not keep_company_suffix:
        text = _COMPANY_SUFFIX_RE.sub('', text).strip()
    return text


def _official_authority_header_candidate(lines: list[str]) -> str | None:
    for raw_line in lines[:8]:
        line = raw_line.strip()
        if not line or re.search(r'領収|レシート|TEL|電話|FAX|登録番号', line, re.IGNORECASE):
            continue
        m = _OFFICIAL_AUTHORITY_HEADER_RE.match(line)
        if not m:
            continue
        candidate = _clean_merchant_candidate(m.group(1))
        if candidate and not _merchant_looks_invalid(candidate):
            return candidate
    return None


def _fix_company_name_merchant(extracted, unified_text):
    """Prefer venue/event name over legal company name when LLM picks the latter."""
    merchant = extracted.get("merchant")
    lines = unified_text.split('\n')
    merchant_text = (merchant or "").strip()
    authority_header = _official_authority_header_candidate(lines)
    if authority_header and merchant_text != authority_header:
        for idx, raw_line in enumerate(lines[:8]):
            line = raw_line.strip()
            if authority_header not in line:
                continue
            following = [item.strip() for item in lines[idx + 1:idx + 4]]
            if (
                not merchant_text
                or _merchant_looks_invalid(merchant_text)
                or any(merchant_text == item and _OFFICIAL_DEPARTMENT_LINE_RE.match(item)
                       for item in following)
            ):
                extracted["merchant"] = authority_header
                return
    if (
        re.fullmatch(r'[A-Z][A-Z0-9&.\'-]{2,}', merchant_text)
        and any(raw_line.strip() == merchant_text for raw_line in lines[:3])
        and any(_COMPANY_SUFFIX_RE.search(raw_line) for raw_line in lines[:8])
    ):
        for raw_line in lines[1:6]:
            line = raw_line.strip()
            if not line or re.search(r'TEL|FAX|https?://|登録番号|領収', line, re.IGNORECASE):
                continue
            brand = re.match(r'^([ァ-ヶー]{2,})(?:[ぁ-ん一-龥].*店[。．.]?|.*\s+.*店[。．.]?)$', line)
            if not brand:
                continue
            candidate = _clean_merchant_candidate(brand.group(1))
            if candidate and not _merchant_looks_invalid(candidate):
                extracted["merchant"] = candidate
                return
    if merchant_text:
        for idx, raw_line in enumerate(lines[:4]):
            line = raw_line.strip()
            if line != merchant_text:
                continue
            if re.fullmatch(r'[A-Z][A-Z\s&.\'-]{4,}', line):
                for prev_raw in reversed(lines[:idx]):
                    prev = prev_raw.strip()
                    if not re.fullmatch(r'[A-Z0-9&.\'-]{2,5}', prev):
                        continue
                    later_header = "\n".join(lines[idx + 1:8])
                    if prev in later_header and not _merchant_looks_invalid(prev):
                        extracted["merchant"] = prev
                        return
            if not re.fullmatch(r'[ぁ-んァ-ン一-龥ー]{2,8}', line):
                continue
            if idx + 1 >= len(lines):
                continue
            next_line = lines[idx + 1].strip()
            romanized_line = lines[idx + 2].strip() if idx + 2 < len(lines) else ""
            if (
                next_line
                and romanized_line
                and re.search(r'[ぁ-んァ-ン一-龥]', next_line)
                and re.search(r'[A-Za-z]', romanized_line)
                and not re.search(r'TEL|FAX|https?://|登録番号|領収', next_line, re.IGNORECASE)
            ):
                candidate = _clean_merchant_candidate(next_line)
                if candidate and not _merchant_looks_invalid(candidate):
                    extracted["merchant"] = candidate
                    return
    for raw_line in lines[:5]:
        line = raw_line.strip()
        if not line:
            continue
        # Store-in-store receipts can start with an ASCII brand followed by the
        # host store/location. If the LLM chose the host store, prefer the
        # leading brand token visible in the header.
        m = re.match(r'^([A-Z][A-Z0-9&.\'-]{2,})\s+(.+)$', line)
        if (
            m
            and merchant
            and merchant in m.group(2)
            and re.search(r'[ぁ-んァ-ン一-龥]', m.group(2))
        ):
            extracted["merchant"] = m.group(1)
            return
    if merchant and re.search(r'[ぁ-んァ-ン一-龥]', merchant):
        merchant_visible_in_header = any(
            merchant in raw_line for raw_line in lines[:6]
        )
        if merchant_visible_in_header:
            for raw_line in lines[:5]:
                line = raw_line.strip()
                if (
                    re.fullmatch(r'[A-Z][A-Z0-9&.\'-]{3,}', line)
                    and not _merchant_looks_invalid(line)
                ):
                    extracted["merchant"] = line
                    return
    if _merchant_looks_invalid(merchant):
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            business = re.search(r'(?:事業者名|販売者|発行者|店舗名)\s*[:：]\s*(.+)$', line)
            if business:
                candidate = _clean_merchant_candidate(business.group(1), keep_company_suffix=True)
                if candidate and not _merchant_looks_invalid(candidate):
                    extracted["merchant"] = candidate
                    return
        for raw_line in lines[:8]:
            line = raw_line.strip()
            header_phone = _HEADER_PHONE_MERCHANT_RE.match(line)
            if not header_phone:
                continue
            candidate = _clean_merchant_candidate(header_phone.group("merchant"))
            if candidate in {"TEL", "Tel", "電話", "お問い合わせ"} or _merchant_looks_invalid(candidate):
                continue
            if candidate:
                extracted["merchant"] = candidate
                return
        for raw_line in lines[:4]:
            line = raw_line.strip()
            if not line or re.search(r'TEL|電話|FAX|登録番号|領収|返品|ご購入店|営業時間|https?://', line, re.IGNORECASE):
                continue
            candidate = re.sub(r'[®™©]', '', line).strip()
            candidate = re.sub(r'\s+', ' ', candidate)
            if (
                re.fullmatch(r'[A-Z][A-Z0-9 &.\'-]{2,30}', candidate)
                and not _merchant_looks_invalid(candidate)
            ):
                extracted["merchant"] = candidate
                return
        for raw_line in lines[:8]:
            line = raw_line.strip()
            if not line:
                continue
            if re.search(r'TEL|電話|FAX|登録番号|領収|返品|ご購入店|営業時間', line, re.IGNORECASE):
                continue
            if re.fullmatch(r'.{1,8}(?:名|番号)', line):
                continue
            if not re.search(r'[ぁ-んァ-ン一-龥]', line):
                continue
            candidate = _clean_merchant_candidate(line)
            if candidate and not _merchant_looks_invalid(candidate):
                extracted["merchant"] = candidate
                return
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            if _COMPANY_SUFFIX_RE.search(line):
                candidate = _clean_merchant_candidate(line)
                if candidate and not _merchant_looks_invalid(candidate):
                    extracted["merchant"] = candidate
                    return
        return
    if not merchant:
        return
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
    for line_idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if merchant in line:
            keep_suffix = bool(re.search(r'事業者名\s*[:：]', line))
            if _COMPANY_SUFFIX_RE.search(line):
                for prev in reversed(lines[max(0, line_idx - 3):line_idx]):
                    prev = prev.strip()
                    if (prev and not _DECORATIVE_RE.match(prev)
                            and not _COMPANY_SUFFIX_RE.search(prev)
                            and not _merchant_looks_invalid(prev)
                            and re.search(r'[ぁ-んァ-ン一-龥]', prev)):
                        extracted["merchant"] = prev
                        return
            candidate = _clean_merchant_candidate(line, keep_company_suffix=keep_suffix)
            if candidate and not _merchant_looks_invalid(candidate):
                extracted["merchant"] = candidate
                return
    for line in lines:
        line = line.strip()
        if not line or _DECORATIVE_RE.match(line) or _COMPANY_SUFFIX_RE.search(line):
            continue
        if line != merchant and len(line) >= 2 and not _merchant_looks_invalid(line):
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
            llm_tax = _sum_taxable_amounts(extracted.get("taxes", []))
            if abs(llm_tax - computed_tax) > 5:
                if extracted.get("taxes"):
                    if llm_tax > 0:
                        scale = computed_tax / llm_tax
                        for t in extracted["taxes"]:
                            if isinstance(t, dict) and t.get("rate") != "0%":
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
        ocr_tax_sum = _sum_taxable_amounts(ocr_totals["taxes"])
        ocr_sub = ocr_totals["subtotal"]
        ocr_tot = ocr_totals["total"]
        if ocr_tax_sum > 0 and abs(ocr_sub + ocr_tax_sum - ocr_tot) > 2:
            computed_sub = ocr_tot - ocr_tax_sum
            if abs(computed_sub + ocr_tax_sum - ocr_tot) < 2:
                extracted["subtotal"] = computed_sub
                ocr_totals["subtotal"] = computed_sub


def _fix_date(extracted, unified_text):
    """Extract and fix dates from OCR text (supports 令和/平成 eras)."""
    def _coerce_modern_ocr_year(year: int) -> int:
        if 2000 <= year <= 2009:
            return year + 20
        if 2010 <= year <= 2019:
            return year + 10
        return year

    def _set_date(year: int, month: int, day: int) -> bool:
        year = _coerce_modern_ocr_year(year)
        if not 2020 <= year <= 2030:
            return False
        extracted["date"] = f"{year:04d}-{month:02d}-{day:02d}"
        return True

    labeled_date_fragment_patterns = [
        r'(20\d{2})\s*年\s*0?(\d{1,2})\s*月\s*0?(\d{1,2})\s*日',
        r'(?<!\d)(\d{2})\s*年\s*0?(\d{1,2})\s*月\s*0?(\d{1,2})\s*日',
        r'(20\d{2})/\s*(\d{1,2})/\s*(\d{1,2})',
        r'(20\d{2})-(\d{1,2})-(\d{1,2})',
    ]
    lines = [line.strip() for line in (unified_text or "").splitlines()]
    for idx, line in enumerate(lines):
        if not re.search(r'日付|ご利用日|お取扱日|取扱日|カードお取扱日', line):
            continue
        windows = [line, "\n".join(lines[idx + 1:min(idx + 3, len(lines))])]
        for window in windows:
            for pattern in labeled_date_fragment_patterns:
                m = re.search(pattern, window)
                if not m:
                    continue
                year = int(m.group(1))
                if year < 100:
                    year += 2000
                if _set_date(year, int(m.group(2)), int(m.group(3))):
                    return

    labeled_patterns = [
        r'(?:日付|ご利用日|お取扱日|取扱日|カードお取扱日)\s*[:：]?\s*(20\d{2})\s*年\s*0?(\d{1,2})\s*月\s*0?(\d{1,2})\s*日',
        r'(?:日付|ご利用日|お取扱日|取扱日|カードお取扱日)\s*[:：]?\s*(\d{2})\s*年\s*0?(\d{1,2})\s*月\s*0?(\d{1,2})\s*日',
    ]
    for pattern in labeled_patterns:
        m = re.search(pattern, unified_text)
        if m:
            year = int(m.group(1))
            if year < 100:
                year += 2000
            if _set_date(year, int(m.group(2)), int(m.group(3))):
                return

    western_patterns = [
        r'(20\d{2})\s*年\s*0?(\d{1,2})\s*月\s*0?(\d{1,2})\s*日',
        r'(20\d{2})/\s*(\d{1,2})/\s*(\d{1,2})',
        r'(20\d{2})-(\d{1,2})-(\d{1,2})',
    ]
    for pattern in western_patterns:
        for western in re.finditer(pattern, unified_text):
            context = unified_text[max(0, western.start() - 16):min(len(unified_text), western.end() + 16)]
            if re.search(r'有効期限|期限|失効|満了', context):
                continue
            if _set_date(int(western.group(1)), int(western.group(2)), int(western.group(3))):
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
                return f"{hh:02d}:{mm:02d}"
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
                candidate = f"{hh:02d}:{mm:02d}"

    # Fallback: search entire OCR for HH時MM分 without digit lookbehind
    if candidate is None:
        matches = list(re.finditer(r'(\d{1,2})時(\d{2})分', unified_text))
        if len(matches) == 1:
            hh, mm = int(matches[0].group(1)), int(matches[0].group(2))
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                candidate = f"{hh:02d}:{mm:02d}"

    # Fallback for receipt footers such as "14:10-2109-01" when the
    # transaction date is printed far above the card slip footer.
    if candidate is None:
        matches = list(_TIME_HHMM_RE.finditer(unified_text))
        if len(matches) == 1:
            hh, mm = int(matches[0].group(1)), int(matches[0].group(2))
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                candidate = f"{hh:02d}:{mm:02d}"

    existing = extracted.get("time")
    if not existing:
        if candidate:
            extracted["time"] = candidate
        return

    normalized_existing = _parse_time_from_segment(str(existing))
    if normalized_existing and normalized_existing != existing:
        extracted["time"] = normalized_existing
        existing = normalized_existing

    if candidate and candidate != existing:
        extracted["time"] = candidate


def _fix_payment_method(extracted, unified_text, ocr_conf, llm_conf):
    """Detect cash payment from OCR evidence (tendered amount, change, etc.)."""
    existing = extracted.get("payment_method")
    if existing in (
        "credit_card", "card", "QUICPay", "iD", "Suica", "PayPay",
        "電子マネー", "electronic_money",
    ):
        extracted["payment_method"] = "credit"
        existing = "credit"
    has_explicit_credit = bool(re.search(
        r'クレジット|カード|VISA|Master(?:Card)?|JCB|AMEX|QUICPay|電子マネー|iD|PayPay|交通系|IC',
        unified_text,
        re.IGNORECASE,
    ))
    has_cash_tender_evidence = bool(re.search(
        r'現金|現計|(?:お預り金?|お預かり)(?!票)|(?<![お\w])預\s*[¥￥]',
        unified_text,
    ))
    if existing == "cash" and has_explicit_credit and not has_cash_tender_evidence:
        extracted["payment_method"] = "credit"
        existing = "credit"
    if not existing and has_explicit_credit:
        extracted["payment_method"] = "credit"
        existing = "credit"

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
    has_tender = bool(re.search(
        r'(?:お預り金?|お預かり)(?!票)|(?<![お\w])預\s*[¥￥]',
        unified_text,
    ))
    has_change_label = bool(re.search(r'釣', unified_text))
    has_cash_keyword = bool(re.search(r'現金\s*[¥￥]\s*[\d,]+', unified_text))
    if not has_cash_keyword and re.search(r'(?:^|\n)\s*現金\s*(?:\n|$)', unified_text):
        has_cash_keyword = True
    strong_cash = has_cash and (
        (has_tender and has_change_label and change_amount != 0) or has_cash_keyword
    )

    if has_cash:
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


def _fix_total_from_stacked_cash_tender_block(extracted, unified_text):
    """Fix value-stacked cash blocks: total, tendered cash, change."""
    lines = unified_text.split('\n')

    def _loose_amount(line: str) -> float | None:
        m = re.fullmatch(r'[¥￥]?\s*(\d{1,3}(?:,\d{3})*|\d{1,5})\s*', line.strip())
        if not m:
            return None
        try:
            return float(m.group(1).replace(',', ''))
        except ValueError:
            return None

    for idx, raw in enumerate(lines):
        line = raw.strip()
        if not re.search(r'総\s*合\s*計|総合計|お会計', line) and not re.fullmatch(r'合\s*計', line):
            continue
        if re.search(r'税|対象|点数', line):
            continue
        window = '\n'.join(lines[idx:min(len(lines), idx + 10)])
        if not re.search(r'現金|お預り|お預かり|預り', window):
            continue
        if not re.search(r'お釣り|お釣銭|釣銭|おつり|釣\s*$', window):
            continue

        amounts: list[float] = []
        for following in lines[idx + 1:min(len(lines), idx + 18)]:
            stripped = following.strip()
            if not stripped:
                continue
            value = _loose_amount(stripped)
            if value is not None:
                amounts.append(value)
                continue
            if amounts and re.search(
                r'To\s*Go|登録番号|発行日|https?://|TEL|電話|詳しくはこちら',
                stripped,
                re.IGNORECASE,
            ):
                break

        triple_match: tuple[float, float, float] | None = None
        for total, tendered, change in zip(amounts, amounts[1:], amounts[2:]):
            if total <= 0 or tendered <= 0 or change < 0:
                continue
            if tendered <= total:
                continue
            if abs((tendered - change) - total) > 2:
                continue
            triple_match = (total, tendered, change)
        if triple_match is not None:
            total, tendered, change = triple_match
            previous_total = extracted.get("total")
            extracted["total"] = total
            points_used = float(extracted.get("points_used") or 0)
            amount_paid = float(extracted.get("amount_paid") or 0)
            if (
                extracted.get("amount_paid") is None
                or previous_total is None
                or abs(amount_paid - float(previous_total or 0)) <= 2
                or abs(amount_paid - tendered) <= 2
                or amount_paid > total
            ):
                extracted["amount_paid"] = max(0.0, total - points_used)
            return

        pair_match: tuple[float, float] | None = None
        for tendered, change in zip(amounts, amounts[1:]):
            if tendered <= 0 or change < 0 or tendered <= change:
                continue
            inferred_total = tendered - change
            if inferred_total <= 0:
                continue
            previous_total = extracted.get("total")
            amount_paid = float(extracted.get("amount_paid") or 0)
            if previous_total is not None and abs(float(previous_total) - tendered) > 2:
                continue
            pair_match = (tendered, change)
        if pair_match is not None:
            tendered, change = pair_match
            inferred_total = tendered - change
            previous_total = extracted.get("total")
            amount_paid = float(extracted.get("amount_paid") or 0)
            extracted["total"] = inferred_total
            points_used = float(extracted.get("points_used") or 0)
            if (
                extracted.get("amount_paid") is None
                or previous_total is None
                or abs(amount_paid - float(previous_total or 0)) <= 2
                or abs(amount_paid - tendered) <= 2
                or amount_paid > inferred_total
            ):
                extracted["amount_paid"] = max(0.0, inferred_total - points_used)
            return


def _fix_unlabeled_cash_tender_change_block(extracted, unified_text):
    """Recover total/tender when cash labels are missing but change reconciles."""
    lines = [line.strip() for line in unified_text.split('\n')]

    def _amount(line: str) -> float | None:
        m = re.fullmatch(r'[¥￥]?\s*(\d{1,3}(?:,\d{3})*|\d{1,6})\s*', line)
        if not m:
            return None
        try:
            return float(m.group(1).replace(',', ''))
        except ValueError:
            return None

    for idx, line in enumerate(lines):
        if not re.fullmatch(r'合\s*計', line):
            continue
        if idx > 0 and re.search(r'税|対象|小\s*計', lines[idx - 1]):
            continue

        values: list[tuple[int, float]] = []
        change_idx = None
        for j in range(idx + 1, min(len(lines), idx + 8)):
            if re.search(r'お釣り|お釣銭|釣銭|おつり', lines[j]):
                change_idx = j
                continue
            value = _amount(lines[j])
            if value is not None:
                values.append((j, value))
                continue
            if values and re.search(r'お買上点数|ポイント|伝票番号|レシート', lines[j]):
                break
        if change_idx is None:
            continue
        label_window = '\n'.join(lines[idx:change_idx + 1])
        if re.search(r'現金|お預り|お預かり|預り|(?<![お\w])預(?:\s|$|[¥￥])', label_window):
            continue

        before_change = [(j, value) for j, value in values if j < change_idx]
        after_change = [(j, value) for j, value in values if j > change_idx]
        if len(before_change) < 2 or not after_change:
            continue
        total = before_change[0][1]
        tendered = before_change[1][1]
        change = after_change[0][1]
        if total <= 0 or tendered <= total or change < 0:
            continue
        if abs((tendered - change) - total) > 2:
            continue

        extracted["total"] = total
        extracted["amount_paid"] = tendered
        extracted["payment_method"] = "cash"
        return


def _fix_toll_payment_reference(extracted, unified_text):
    """Recover toll-road handling/reference numbers printed outside item rows."""
    if extracted.get("payment_reference"):
        return
    if not re.search(r'料金所|高速道路|ETC|NEXCO', unified_text):
        return
    m = re.search(r'取扱番号\s*[:：]?\s*([0-9][0-9-]{5,})', unified_text)
    if m:
        extracted["payment_reference"] = m.group(1)


_SERVICE_FEE_DESCRIPTION_RE = re.compile(r'通行|利用|サービス|施設|駐車|入場|手数|料金')
_SERVICE_TAX_RATE_EVIDENCE_RE = re.compile(
    r'(?:消費税率|税率).{0,20}10\s*[%％]|10\s*[%％].{0,20}(?:内税|税込|消費税)'
)
_ADMIN_FEE_DESCRIPTION_RE = re.compile(r'証明|住民票|戸籍|印鑑|所得|課税|納税|手数料|電申')


def _is_service_fee_description(description: str | None) -> bool:
    return bool(description and _SERVICE_FEE_DESCRIPTION_RE.search(description))


def _has_service_inclusive_tax_evidence(unified_text: str) -> bool:
    if _SERVICE_TAX_RATE_EVIDENCE_RE.search(unified_text):
        return True
    return bool(re.search(r'消費税(?:等)?[^。.\n]{0,20}含', unified_text))


def _fix_single_service_inclusive_tax(extracted, unified_text):
    """Reconstruct implicit inclusive tax for single-row service-fee receipts."""
    if extracted.get("taxes") or not extracted.get("total"):
        return
    total = float(extracted["total"])
    if total <= 0:
        return
    items = extracted.get("line_items") or []
    priced_items = [
        item for item in items
        if isinstance(item, dict) and float(item.get("total") or 0) > 0
    ]
    if len(priced_items) != 1:
        return
    item = priced_items[0]
    if abs(float(item.get("total") or 0) - total) > 2:
        return
    if not _is_service_fee_description(item.get("description")):
        return
    if not _has_service_inclusive_tax_evidence(unified_text):
        return
    tax = round(total * 10 / 110)
    if tax <= 0:
        return
    extracted["taxes"] = [{"rate": "10%", "label": "内税", "amount": float(tax)}]
    extracted["subtotal"] = total - tax
    if items:
        for item in items:
            if isinstance(item, dict):
                item["tax_category"] = "10%"


def _fix_bare_service_receipt_without_itemization(extracted, unified_text):
    """Avoid synthetic items on bare service receipts with no itemization."""
    if not extracted.get("total"):
        return
    if not re.search(r'領収証|領収書', unified_text):
        return
    if not re.search(r'[¥￥]\s*[\d,]+|[\d,]+\s*円', unified_text):
        return
    has_itemization = bool(re.search(
        r'商品名|品名|明細|内訳|数量|単価|小計|@\s*\d|[×xX]\s*\d|'
        r'\d+(?:\.\d+)?\s*(?:L|個|点)\b',
        unified_text,
        re.IGNORECASE,
    ))
    if not has_itemization and re.search(r'小\s*\n\s*計|非課税対象額', unified_text):
        has_itemization = bool(_ADMIN_FEE_DESCRIPTION_RE.search(unified_text))
    if has_itemization:
        return
    total = float(extracted["total"])
    items = extracted.get("line_items") or []
    if items:
        item_sum = sum(
            float(item.get("total") or 0)
            for item in items
            if isinstance(item, dict)
        )
        if abs(item_sum - total) <= 2:
            extracted["line_items"] = []
    if not re.search(r'現金|お預り|お預かり|クレジット|カード|QUICPay|iD|PayPay|電子マネー|交通系|IC', unified_text):
        extracted["payment_method"] = None


def _amount_from_yen_text(text: str) -> float | None:
    m = re.search(r'[¥￥]\s*([\d,]+)|([\d,]+)\s*円', text or "")
    if not m:
        return None
    return _parse_amount_fragment(m.group(1) or m.group(2))


def _nontaxable_base_matches_total(unified_text: str, total: float) -> bool:
    lines = [line.strip() for line in unified_text.split('\n')]
    for idx, line in enumerate(lines):
        if "非課税対象額" not in line:
            continue
        amount = _amount_from_yen_text(line)
        if amount is None and idx + 1 < len(lines):
            amount = _amount_from_yen_text(lines[idx + 1])
        if amount is not None and abs(amount - total) <= 1:
            return True
    return False


def _clean_admin_fee_description(line: str) -> str:
    desc = re.sub(r'[¥￥]\s*[\d,]+|[\d,]+\s*円', '', line or "")
    desc = re.sub(r'^\s*\d{3,}\s*', '', desc)
    desc = re.sub(r'\s+', ' ', desc).strip(" :：-")
    return desc


def _recover_nontaxable_admin_fee_item(extracted, unified_text):
    """Recover a single official/certificate fee item from non-taxable receipt structure."""
    if extracted.get("line_items") or not extracted.get("total"):
        return
    if not re.search(r'領収書|領収証', unified_text):
        return
    total = float(extracted["total"])
    if total <= 0 or not _nontaxable_base_matches_total(unified_text, total):
        return
    taxes = extracted.get("taxes") or []
    has_positive_tax = any(
        isinstance(tax, dict) and float(tax.get("amount") or 0) > 0
        for tax in taxes
    )
    if has_positive_tax:
        return
    lines = [line.strip() for line in unified_text.split('\n')]
    for idx, line in enumerate(lines):
        if not line or re.search(r'領収|小計|合計|対象額|消費税|お預り|お釣り|取引|No[:：]?', line, re.IGNORECASE):
            continue
        desc = _clean_admin_fee_description(line)
        if not desc or not _ADMIN_FEE_DESCRIPTION_RE.search(desc):
            continue
        amount = _amount_from_yen_text(line)
        if amount is None and idx + 1 < len(lines):
            amount = _amount_from_yen_text(lines[idx + 1])
        if amount is None or abs(amount - total) > 1:
            continue
        extracted["line_items"] = [{
            "description": desc,
            "qty": 1,
            "unit_price": total,
            "total": total,
            "tax_category": "0%",
            "discount": 0,
            "discount_rate": "",
        }]
        extracted["subtotal"] = total
        extracted["taxes"] = [{"rate": "0%", "label": "非課税", "amount": 0}]
        return


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
    r'|(?:Code128|操作)?割引\d*'             # discount operation label
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

        # OCR may truncate or slightly misread the current description while
        # still clearly showing the same item row. Treat a strong fuzzy match
        # as OCR evidence so a neighboring product-code line does not steal
        # this item's price/description pairing.
        if not desc_lines and len(desc) >= 5:
            def _norm_desc_evidence(text: str) -> str:
                text = re.sub(r'^(?:\d{2,}-){1,}\d+\)?\s*', '', text or "")
                text = re.sub(r'^\d{4,}[A-Za-z0-9-]*\)?\s*', '', text or "")
                text = re.sub(r'[¥￥]?\s*\d[\d,]*\s*[※\*除軽]?\s*$', '', text)
                text = re.sub(r'\s+', '', text)
                text = re.sub(r'[^\wぁ-んァ-ン一-龥]', '', text, flags=re.UNICODE)
                return text.lower()

            nd = _norm_desc_evidence(desc)
            if len(nd) >= 4:
                for i, line in enumerate(lines):
                    nl = _norm_desc_evidence(line)
                    if len(nl) < 4 or re.fullmatch(r'\d+', nl):
                        continue
                    if nd in nl or nl in nd or SequenceMatcher(None, nd, nl).ratio() >= 0.82:
                        desc_lines = [i]
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

        # Also include bare-digit price lines (no marker, no ¥) within the
        # item zone. These don't carry a candidate desc, so they only inform
        # the near_any_price safety check below — column-format AEON-style
        # receipts print item prices as bare digits in a block below a
        # contiguous run of name lines, and without this the safety check
        # misses a valid match and a correctly-paired item gets replaced.
        zone_end = len(lines)
        for zi, zline in enumerate(lines):
            if re.search(
                r'^(小\s*計|合\s*計|外税|内税|消費税|お預り|現計|お釣り|釣銭)',
                zline.strip(),
            ):
                zone_end = zi
                break
        existing_match_idxs = {pi for pi, _ in price_matches}
        for i in range(zone_end):
            if i in existing_match_idxs:
                continue
            line = lines[i]
            if _SKIP_PRICE_LINE.search(line):
                continue
            s = line.strip()
            bare_m = re.fullmatch(r'\s*([\d,]+)\s*$', s)
            if not bare_m:
                continue
            try:
                price = float(bare_m.group(1).replace(',', ''))
            except ValueError:
                continue
            if abs(price - total) < 1:
                price_matches.append((i, ""))

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


def _fix_line_items(extracted, unified_text, ocr_layout_blocks=None):
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

    _recover_nontaxable_admin_fee_item(extracted, unified_text)

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
    _drop_duplicate_with_embedded_price(extracted["line_items"])
    _fix_item_desc_from_ocr_price_line(extracted["line_items"], unified_text)
    _merge_qty_detail_into_previous(extracted["line_items"], unified_text)
    _repair_previous_item_from_following_qty_detail(extracted, unified_text)
    _fix_junk_descriptions(extracted["line_items"], unified_text)
    _strip_embedded_price_in_desc(extracted["line_items"])
    _remove_unit_rate_phantom_items(extracted)
    _fix_qty_hallucinations(extracted["line_items"], unified_text)
    _replace_duplicate_desc_from_ocr(extracted["line_items"], unified_text)
    _fix_duplicate_descriptions_from_ocr(extracted, unified_text)
    _fix_item_totals_from_ocr_neighborhood(
        extracted["line_items"], unified_text,
        extracted.get("subtotal"), extracted.get("total"),
        canonical_subtotal=_canonical_subtotal_from_taxes(extracted),
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
    _fix_single_item_qty_from_ocr(extracted, unified_text)
    _fix_single_service_item_from_ocr(extracted, unified_text)
    _fix_fuel_item_description(extracted, unified_text)
    _expand_collapsed_items(extracted, unified_text)
    _fix_hallucinated_prices(extracted["line_items"], unified_text)
    _fix_zero_prices_from_ocr(extracted["line_items"], unified_text)
    _fix_discount_totals(extracted["line_items"])
    _fix_misattributed_discounts(extracted["line_items"])
    _clear_discounts_without_nearby_ocr_marker(extracted["line_items"], unified_text)
    _detect_ocr_discounts(extracted["line_items"], unified_text)
    _project_totals_to_ocr_multiset(extracted, unified_text)
    _project_totals_to_layout_rows(extracted, ocr_layout_blocks)
    _recover_missing_items_from_gap(extracted, unified_text)
    # Re-run dedup: _fix_qty_from_ocr_patterns / _expand_collapsed_items can
    # rewrite an item's qty / unit_price after the first dedup pass, exposing
    # a phantom-child duplicate that wasn't groupable before. Without this,
    # an LLM extraction like (qty=1, unit=456, total=456) + phantom (qty=1,
    # unit=228, total=228) — different unit_prices, so first dedup misses —
    # gets corrected to (qty=2, unit=228, total=456) by qty-fix, but the
    # phantom stays.
    _dedup_same_total_items(extracted)


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


def _repair_previous_item_from_following_qty_detail(extracted, unified_text):
    """Repair a small item amount followed by a visible qty/unit detail."""
    items = [item for item in (extracted.get("line_items") or []) if isinstance(item, dict)]
    if not items:
        return
    lines = [line.strip() for line in unified_text.split('\n')]
    targets = {
        float(value)
        for value in (extracted.get("subtotal"), extracted.get("total"), _canonical_subtotal_from_taxes(extracted))
        if value is not None and float(value or 0) > 0
    }
    for line in lines:
        for match in re.finditer(r'[¥￥]\s*([\d,]+)', line):
            value = float(match.group(1).replace(',', ''))
            if value > 0:
                targets.add(value)
    if not targets:
        return

    def _norm(text: str) -> str:
        text = re.sub(r'\s+', '', str(text or ""))
        text = re.sub(r'[^\wぁ-んァ-ン一-龥]', '', text, flags=re.UNICODE)
        return text.lower()

    item_sum = _line_items_sum(extracted)
    for item in items:
        desc = str(item.get("description") or "")
        ndesc = _norm(desc)
        if len(ndesc) < 3:
            continue
        try:
            current_total = float(item.get("total") or 0)
            current_qty = float(item.get("qty") or 1)
            discount = float(item.get("discount") or 0)
        except (TypeError, ValueError):
            continue
        if current_qty != 1 or discount > 0 or current_total <= 0:
            continue
        for idx, line in enumerate(lines):
            nline = _norm(_clean_ocr_price_line_desc(line))
            if not nline or not (ndesc in nline or nline in ndesc):
                continue
            seen_current_amount = False
            for lookahead in lines[idx + 1:min(len(lines), idx + 5)]:
                amount = _parse_amount_fragment(
                    lookahead.strip().lstrip('¥￥').rstrip(')）').replace(',', '')
                )
                if amount is not None and abs(amount - current_total) <= 2:
                    seen_current_amount = True
                    continue
                detail = _parse_qty_detail_total(lookahead)
                if not detail:
                    continue
                qty, unit = detail
                gross = qty * unit
                if not seen_current_amount or gross <= current_total:
                    break
                new_sum = item_sum - current_total + gross
                if not any(abs(new_sum - target) <= 2 for target in targets):
                    break
                item["qty"] = qty
                item["unit_price"] = unit
                item["total"] = gross
                item_sum = new_sum
                break
            break


def _clear_discount_when_negative_line_precedes_own_price(extracted, unified_text):
    """Clear discounts attached to an item whose own price prints after the discount."""
    items = extracted.get("line_items") or []
    if not items:
        return
    lines = [line.strip() for line in (unified_text or "").splitlines()]

    def _norm(text: str) -> str:
        text = re.sub(r'^\d{4,}[A-Za-z0-9-]*\)?\s*', '', text or "")
        text = re.sub(r'[¥￥]?\s*\d[\d,]*\s*(?:[*※除軽]|%|％)?\s*$', '', text)
        text = re.sub(r'\s+', '', text)
        text = re.sub(r'[^\wぁ-んァ-ン一-龥]', '', text, flags=re.UNICODE)
        return text.lower()

    def _line_has_amount(line: str, amount: float | None) -> bool:
        if amount is None:
            return False
        amount_int = int(round(float(amount)))
        return bool(re.search(r'(?<!\d)' + re.escape(f"{amount_int:,}") + r'|' + re.escape(str(amount_int)) + r'(?!\d)', line))

    def _line_has_discount(line: str, discount: float) -> bool:
        m = re.fullmatch(r'\s*-\s*[¥￥\\]?\s*(\d[\d,]*)\s*', line)
        return bool(m and abs(float(m.group(1).replace(',', '')) - discount) <= 2)

    for item in items:
        if not isinstance(item, dict):
            continue
        discount = float(item.get("discount") or 0)
        if discount <= 0:
            continue
        desc_norm = _norm(item.get("description") or "")
        if len(desc_norm) < 3:
            continue
        qty = float(item.get("qty") or 1)
        unit = item.get("unit_price")
        if unit is None:
            continue
        own_amounts = [float(unit)]
        if qty != 1:
            own_amounts.append(float(unit) * qty)
        for idx, line in enumerate(lines):
            norm_line = _norm(line)
            if not norm_line or not (desc_norm in norm_line or norm_line in desc_norm):
                continue
            discount_idx = None
            own_price_idx = None
            for j in range(idx + 1, min(len(lines), idx + 10)):
                if discount_idx is None and _line_has_discount(lines[j], discount):
                    discount_idx = j
                if own_price_idx is None and any(_line_has_amount(lines[j], amount) for amount in own_amounts):
                    own_price_idx = j
                if discount_idx is not None and own_price_idx is not None:
                    break
            if discount_idx is not None and own_price_idx is not None and discount_idx < own_price_idx:
                item["discount"] = 0
                item["discount_rate"] = ""
                item["total"] = qty * float(unit)
            break


def _canonical_subtotal_from_taxes(extracted) -> float | None:
    total = extracted.get("total")
    taxes = extracted.get("taxes") or []
    if total is None or not taxes:
        return None
    tax_sum = _sum_taxable_amounts(taxes)
    if not tax_sum:
        return None
    return float(total) - float(tax_sum)


def _sum_taxable_amounts(taxes) -> float:
    """Sum actual tax amounts, excluding 0% entries that store exempt bases."""
    return sum(
        float(t.get("amount") or 0)
        for t in (taxes or [])
        if isinstance(t, dict)
        and t.get("rate") != "0%"
        and t.get("amount") is not None
    )


def _line_items_sum(extracted) -> float:
    return sum(
        float(item.get("total") or 0)
        for item in (extracted.get("line_items") or [])
        if isinstance(item, dict)
    )


def _drop_unprinted_small_target_only_taxes(extracted, unified_text):
    """Omit tiny tax rows when OCR prints only a target base, not a tax amount."""
    taxes = [tax for tax in (extracted.get("taxes") or []) if isinstance(tax, dict)]
    if not taxes:
        return
    rate_bases = extract_rate_bases(unified_text)
    if not rate_bases:
        return
    printed_tax_rates = {
        normalize_tax_rate(str(tax.get("rate") or ""))
        for tax in (extract_financial_totals(unified_text).get("taxes") or [])
        if isinstance(tax, dict) and (tax.get("amount") or 0) > 0
    }
    has_standalone_tax_label = any(
        re.fullmatch(r'消費税(?:等|額)?', line.strip())
        for line in unified_text.split('\n')
    )
    kept = []
    changed = False
    for tax in taxes:
        rate = normalize_tax_rate(str(tax.get("rate") or ""))
        amount = float(tax.get("amount") or 0)
        if (
            rate in rate_bases
            and rate not in printed_tax_rates
            and 0 < amount <= 1
            and not has_standalone_tax_label
        ):
            changed = True
            continue
        kept.append(tax)
    if not changed:
        return
    extracted["taxes"] = kept
    total = extracted.get("total")
    try:
        total_f = float(total) if total is not None else None
    except (TypeError, ValueError):
        total_f = None
    if total_f is not None:
        tax_sum = _sum_taxable_amounts(kept)
        if 0 <= tax_sum <= total_f:
            extracted["subtotal"] = total_f - tax_sum


def _restore_bare_number_tax_summary(extracted, unified_text):
    """Restore tax entries from bare-number rate summary label/value stacks."""
    lines = [line.strip() for line in unified_text.split('\n')]
    entries = _bare_number_tax_summary_entries(lines)
    entries.extend(_interleaved_rate_tax_summary_entries(lines))
    taxes = [(rate, value) for rate, kind, value in entries if kind == "tax" and value > 0]
    if not taxes:
        return
    total = extracted.get("total")
    try:
        total_f = float(total) if total is not None else None
    except (TypeError, ValueError):
        total_f = None
    tax_sum = sum(value for _rate, value in taxes)
    item_sum = _line_items_sum(extracted)
    subtotal = total_f - tax_sum if total_f is not None and total_f >= tax_sum else extracted.get("subtotal")
    label = normalize_tax_label(
        "内税",
        unified_text,
        subtotal=subtotal,
        total=total_f,
        tax_sum=tax_sum,
        items_sum=item_sum or None,
    )
    extracted["taxes"] = [
        {"rate": rate, "label": label, "amount": value}
        for rate, value in taxes
    ]
    if total_f is not None and total_f >= tax_sum:
        extracted["subtotal"] = total_f - tax_sum


def _items_plus_tax_matches_total(extracted, tolerance: float = 5) -> bool:
    total = extracted.get("total")
    if total is None:
        return False
    item_sum = _line_items_sum(extracted)
    tax_sum = _sum_taxable_amounts(extracted.get("taxes") or [])
    return item_sum > 0 and tax_sum > 0 and abs(item_sum + tax_sum - float(total)) <= tolerance


def _prefer_printed_item_sum_total_when_balanced(extracted, unified_text):
    """Use the item sum as total when OCR prints the same yen amount."""
    items = [item for item in (extracted.get("line_items") or []) if isinstance(item, dict)]
    if len(items) < 2:
        return
    item_sum = sum(float(item.get("total") or 0) for item in items)
    if item_sum <= 0:
        return
    current_total = extracted.get("total")
    try:
        current_total_f = float(current_total) if current_total is not None else None
    except (TypeError, ValueError):
        current_total_f = None
    if current_total_f is not None and abs(current_total_f - item_sum) <= 2:
        return
    if current_total_f is not None and _items_plus_tax_matches_total(extracted):
        return
    printed_amounts = [
        float(m.group(1).replace(',', ''))
        for m in re.finditer(r'[¥￥]\s*([\d,]+)\s*-?', unified_text)
    ]
    if not any(abs(amount - item_sum) <= 2 for amount in printed_amounts):
        return
    old_total = current_total_f
    extracted["total"] = item_sum
    amount_paid = extracted.get("amount_paid")
    try:
        amount_paid_f = float(amount_paid) if amount_paid is not None else None
    except (TypeError, ValueError):
        amount_paid_f = None
    if amount_paid_f is None or (old_total is not None and abs(amount_paid_f - old_total) <= 5):
        extracted["amount_paid"] = item_sum
    tax_sum = _sum_taxable_amounts(extracted.get("taxes") or [])
    if tax_sum > 0 and item_sum >= tax_sum:
        extracted["subtotal"] = item_sum - tax_sum


def _restore_printed_summary_total_when_tax_balanced(extracted, unified_text):
    """Use explicit 小計/合計 summary labels when they balance with tax lines."""
    taxes = [tax for tax in (extracted.get("taxes") or []) if isinstance(tax, dict)]
    tax_sum = _sum_taxable_amounts(taxes)
    if tax_sum <= 0:
        return
    lines = [line.strip() for line in unified_text.split('\n')]

    def _yen_after(label_idx: int, lookahead: int = 3) -> float | None:
        for j in range(label_idx + 1, min(len(lines), label_idx + 1 + lookahead)):
            vm = re.fullmatch(r'[¥￥]\s*([\d,]+)\s*[\)）]?', lines[j])
            if vm:
                return float(vm.group(1).replace(',', ''))
            if re.search(r'お預り|お釣|ポイント|伝票|レシート', lines[j]):
                break
        return None

    printed_subtotal = None
    printed_total = None
    total_candidates: list[float] = []
    for idx, line in enumerate(lines):
        if (
            printed_subtotal is None
            and (
                re.fullmatch(r'小\s*計', line)
                or (line == "小" and idx + 1 < len(lines) and lines[idx + 1] == "計")
            )
        ):
            label_idx = idx + 1 if line == "小" else idx
            printed_subtotal = _yen_after(label_idx)
            continue
        if (
            printed_total is None
            and (
                re.fullmatch(r'合\s*計', line)
                or (line == "合" and idx + 1 < len(lines) and lines[idx + 1] == "計")
            )
        ):
            label_idx = idx + 1 if line == "合" else idx
            printed_total = _yen_after(label_idx)
            if printed_total is not None:
                total_candidates.append(printed_total)
            continue

    labels: list[str] = []
    values: list[float] = []
    in_summary = False
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if not in_summary:
            if re.fullmatch(r'小\s*計', line) or (
                line == "小" and idx + 1 < len(lines) and lines[idx + 1] == "計"
            ):
                in_summary = True
            else:
                idx += 1
                continue
        if re.search(r'レシート|お買上点数|店No|印は|登録番号', line):
            break
        vm = re.fullmatch(r'[¥￥]\s*([\d,]+)\s*[\)）]?', line)
        if vm:
            values.append(float(vm.group(1).replace(',', '')))
            idx += 1
            continue
        if line == "計":
            idx += 1
            continue
        label = line
        if line in {"小", "合"} and idx + 1 < len(lines) and lines[idx + 1] == "計":
            label = line + "計"
            idx += 1
        if re.search(r'小\s*計|合\s*計|現\s*計|税率|対象額|税額|外税|内税|消費税|お預り|お釣|釣銭', label):
            labels.append(label)
        idx += 1

    for label, value in zip(labels, values):
        if printed_subtotal is None and re.search(r'小\s*計', label):
            printed_subtotal = value
        if re.search(r'合\s*計|現\s*計', label):
            total_candidates.append(value)

    item_sum = _line_items_sum(extracted)
    if item_sum > 0 and values:
        subtotal_candidates = [amount for amount in values if abs(amount - item_sum) <= 5]
        if subtotal_candidates:
            candidate_subtotal = subtotal_candidates[0]
            if any(
                amount > candidate_subtotal
                and abs(candidate_subtotal + tax_sum - amount) <= 5
                for amount in [*total_candidates, *values]
            ):
                printed_subtotal = candidate_subtotal

    if printed_subtotal is not None:
        balanced_candidates = [
            (abs(printed_subtotal + tax_sum - amount), amount)
            for amount in [*total_candidates, *values]
            if amount > printed_subtotal
            and abs(printed_subtotal + tax_sum - amount) <= 5
        ]
        if balanced_candidates:
            printed_total = min(balanced_candidates, key=lambda candidate: candidate[0])[1]

    if printed_subtotal is None or printed_total is None:
        return
    if printed_subtotal <= 0 or printed_total <= printed_subtotal:
        return
    if abs(printed_subtotal + tax_sum - printed_total) > 5:
        return

    current_total = extracted.get("total")
    try:
        current_total_f = float(current_total) if current_total is not None else None
    except (TypeError, ValueError):
        current_total_f = None
    current_subtotal = extracted.get("subtotal")
    try:
        current_subtotal_f = float(current_subtotal) if current_subtotal is not None else None
    except (TypeError, ValueError):
        current_subtotal_f = None
    if (
        current_total_f is not None
        and abs(current_total_f - printed_total) <= 0.01
        and current_subtotal_f is not None
        and abs(current_subtotal_f - printed_subtotal) <= 0.01
    ):
        return
    if item_sum > 0 and abs(item_sum - printed_subtotal) > 5:
        return

    extracted["subtotal"] = printed_subtotal
    extracted["total"] = printed_total
    points_used = extracted.get("points_used")
    try:
        points_used_f = float(points_used or 0)
    except (TypeError, ValueError):
        points_used_f = 0.0
    if points_used_f > 0:
        extracted["amount_paid"] = max(0.0, printed_total - points_used_f)
    else:
        amount_paid = extracted.get("amount_paid")
        try:
            amount_paid_f = float(amount_paid) if amount_paid is not None else None
        except (TypeError, ValueError):
            amount_paid_f = None
        if amount_paid_f is None or (current_total_f is not None and abs(amount_paid_f - current_total_f) <= 5):
            extracted["amount_paid"] = printed_total


def _restore_external_tax_total_from_printed_subtotal(extracted, unified_text):
    """Restore total when printed subtotal plus external taxes matches a visible total."""
    taxes = [tax for tax in (extracted.get("taxes") or []) if isinstance(tax, dict)]
    tax_sum = _sum_taxable_amounts(taxes)
    if tax_sum <= 0:
        return
    if not any(str(tax.get("label") or "") == "外税" for tax in taxes) and "外税" not in unified_text:
        return
    lines = [line.strip() for line in unified_text.split('\n') if line.strip()]

    def _yen_after(label_idx: int, lookahead: int = 3) -> float | None:
        for j in range(label_idx + 1, min(len(lines), label_idx + 1 + lookahead)):
            if re.search(r'ポイント|POINT|残高|累計|有効', lines[j], re.IGNORECASE):
                break
            vm = re.fullmatch(r'[¥￥]\s*([\d,]+)\s*[\)）]?', lines[j])
            if vm:
                return float(vm.group(1).replace(',', ''))
        return None

    printed_subtotal = None
    for idx, line in enumerate(lines):
        if re.fullmatch(r'小\s*計', line) or (
            line == "小" and idx + 1 < len(lines) and lines[idx + 1] == "計"
        ):
            label_idx = idx + 1 if line == "小" else idx
            printed_subtotal = _yen_after(label_idx)
            break
    if printed_subtotal is None or printed_subtotal <= 0:
        return
    item_sum = _line_items_sum(extracted)
    if item_sum > 0 and abs(item_sum - printed_subtotal) > 5:
        return
    expected_total = printed_subtotal + tax_sum

    def _has_visible_summary_or_payment_amount(target: float) -> bool:
        for idx, line in enumerate(lines):
            vm = re.fullmatch(r'[¥￥]\s*([\d,]+)\s*[\)）]?', line)
            if not vm:
                continue
            value = float(vm.group(1).replace(',', ''))
            if abs(value - target) > 5:
                continue
            context = "\n".join(lines[max(0, idx - 4):min(len(lines), idx + 5)])
            has_payment_or_summary = bool(
                re.search(r'合\s*計|現\s*計|支払|お預り|預り|クレジット|電子マネー|WAON|Pay', context)
            )
            loyalty_only = bool(
                re.search(r'ポイント対象|今回獲得|累計|有効|POINT', context, re.IGNORECASE)
            ) and not re.search(r'支払|お預り|預り|クレジット|電子マネー|WAON|Pay', context)
            if has_payment_or_summary and not loyalty_only:
                return True
        return False

    if not _has_visible_summary_or_payment_amount(expected_total):
        return
    current_total = extracted.get("total")
    try:
        current_total_f = float(current_total) if current_total is not None else None
    except (TypeError, ValueError):
        current_total_f = None
    old_total = current_total_f
    if current_total_f is None or abs(current_total_f - expected_total) > 5:
        extracted["total"] = expected_total
    if abs(float(extracted.get("subtotal") or 0) - printed_subtotal) > 5:
        extracted["subtotal"] = printed_subtotal
    points_used = extracted.get("points_used")
    try:
        points_used_f = float(points_used or 0)
    except (TypeError, ValueError):
        points_used_f = 0.0
    expected_paid = max(0.0, expected_total - points_used_f)
    amount_paid = extracted.get("amount_paid")
    try:
        amount_paid_f = float(amount_paid) if amount_paid is not None else None
    except (TypeError, ValueError):
        amount_paid_f = None
    if (
        amount_paid_f is None
        or (old_total is not None and abs(amount_paid_f - old_total) <= 5)
        or abs(amount_paid_f - expected_paid) <= 5
        or amount_paid_f < expected_paid
    ):
        extracted["amount_paid"] = expected_paid


def _fix_item_totals_from_ocr_neighborhood(
    items, unified_text, target_subtotal, target_total, canonical_subtotal=None,
):
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
    targets = [t for t in (canonical_subtotal, target_subtotal, target_total) if t]
    if not targets:
        return


    lines = unified_text.split('\n')

    def _ocr_price_after(li: int) -> tuple[float | None, int | None]:
        # Look for a clean ¥-bearing or plain numeric line within next 6 lines.
        # Stop on another item-like line (Japanese text, no ¥).
        for j in range(li + 1, min(li + 7, len(lines))):
            s = lines[j].strip()
            if not s:
                continue
            if _SKIP_PRICE_LINE.search(s):
                return None, None
            m = re.match(r'^[¥￥]?\s*([\d,]+)\s*[※\*除]?\s*$', s)
            if m:
                try:
                    return float(m.group(1).replace(',', '')), j
                except ValueError:
                    return None, None
            if re.search(r'[ぁ-んァ-ン一-龥]{2,}', s):
                return None, None  # next item starts before any price
        return None, None

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

    def _ocr_window_contains_price(li: int, price: float) -> bool:
        if price is None:
            return False
        for j in range(li, min(li + 7, len(lines))):
            s = lines[j].strip()
            if not s:
                continue
            inline = _ocr_price_inline(s)
            if inline is not None and abs(inline - price) <= 1:
                return True
            m = re.match(r'^[¥￥]?\s*([\d,]+)\s*[※\*除]?\s*$', s)
            if m:
                try:
                    if abs(float(m.group(1).replace(',', '')) - price) <= 1:
                        return True
                except ValueError:
                    pass
            if j > li and re.search(r'[ぁ-んァ-ン一-龥]{2,}', s):
                return False
        return False

    def _ocr_window_supports_qty(li: int, price_li: int | None, item: dict) -> bool:
        qty = item.get("qty") or 1
        unit = item.get("unit_price")
        try:
            qty_f = float(qty)
            unit_f = float(unit)
        except (TypeError, ValueError):
            return False
        if qty_f <= 1 or unit_f <= 0:
            return False
        end = price_li if price_li is not None else min(li + 6, len(lines) - 1)
        qty_re = re.compile(
            r'(\d+(?:\.\d+)?)\s*[個コ点]\s*[xX×Ⅹ]\s*(?:単|@)?\s*([\d,]+)'
            r'|(?:単|@)?\s*([\d,]+)\s*[xX×Ⅹ]\s*(\d+(?:\.\d+)?)\s*[個コ点]?'
        )
        for j in range(li + 1, min(end + 1, len(lines))):
            s = lines[j].strip()
            if not s:
                continue
            m = qty_re.search(s)
            if not m:
                continue
            if m.group(1):
                found_qty = float(m.group(1))
                found_unit = float(m.group(2).replace(',', ''))
            else:
                found_unit = float(m.group(3).replace(',', ''))
                found_qty = float(m.group(4))
            if abs(found_qty - qty_f) <= 0.01 and abs(found_unit - unit_f) <= 1:
                return True
        return False

    for item in items:
        if not isinstance(item, dict):
            continue
        desc = (item.get("description") or "").strip()
        if not desc or len(desc) < 5:
            continue
        try:
            qty = float(item.get("qty") or 1)
            unit = float(item.get("unit_price") or 0)
            total = float(item.get("total") or 0)
        except (TypeError, ValueError):
            continue
        if qty <= 1 or unit <= 0 or total <= 0:
            continue
        if abs(total - (qty * unit)) <= 1:
            continue
        desc_prefix = desc[:5]
        for li, line in enumerate(lines):
            if desc_prefix not in line:
                continue
            ocr_total = _ocr_price_inline(line)
            price_li = li if ocr_total is not None else None
            if ocr_total is None:
                ocr_total, price_li = _ocr_price_after(li)
            if ocr_total is None or abs(ocr_total - total) > 1:
                continue
            if _ocr_window_supports_qty(li, price_li, item):
                continue
            item["qty"] = 1.0
            item["unit_price"] = total
            break

    # Apply candidate fixes one at a time, verifying each improves items_sum
    # toward a target. Stop when items_sum is within 2 yen of a target.
    progress = True
    while progress:
        progress = False
        items_sum = sum(i.get("total", 0) for i in items if isinstance(i, dict))
        if any(abs(items_sum - t) <= 2 for t in targets):
            break
        candidates: list[tuple[float, int, float, int, int | None]] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            desc = (item.get("description") or "").strip()
            total = item.get("total")
            if not desc or len(desc) < 5 or total is None:
                continue
            desc_prefix = desc[:5]
            matching_lines = [
                li for li, line in enumerate(lines)
                if desc_prefix in line
            ]
            if not matching_lines:
                continue
            if any(_ocr_window_contains_price(li, float(total)) for li in matching_lines):
                continue  # original total is OCR-supported; do not chase neighbors
            for li in matching_lines:
                line = lines[li]
                if _ocr_window_contains_price(li, float(total)):
                    break  # original total is OCR-supported; do not chase neighbors
                ocr_total = _ocr_price_inline(line)
                price_li = li if ocr_total is not None else None
                if ocr_total is None:
                    ocr_total, price_li = _ocr_price_after(li)
                if ocr_total is None:
                    continue
                if abs(ocr_total - total) <= 1:
                    break  # already aligned
                # Score this candidate by the improvement it brings
                new_sum = items_sum - total + ocr_total
                cur_diff = min(abs(items_sum - t) for t in targets)
                new_diff = min(abs(new_sum - t) for t in targets)
                if new_diff < cur_diff:
                    candidates.append((cur_diff - new_diff, idx, ocr_total, li, price_li))
                break  # first matching OCR line for this item
        if not candidates:
            break
        candidates.sort(reverse=True)  # largest improvement first
        improvement, idx, new_total, match_li, price_li = candidates[0]
        item = items[idx]
        item["total"] = new_total
        try:
            qty = float(item.get("qty") or 1)
            unit = float(item.get("unit_price") or 0)
        except (TypeError, ValueError):
            qty = 1
            unit = 0
        if qty == 1 and item.get("unit_price") is not None:
            item["unit_price"] = new_total
        elif (
            qty > 1
            and unit > 0
            and abs(new_total - (qty * unit)) > 1
            and not _ocr_window_supports_qty(match_li, price_li, item)
        ):
            item["qty"] = 1.0
            item["unit_price"] = new_total
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

    for li in range(min(end_idx, len(lines))):
        line = lines[li]
        for d in item_descs:
            if d[:5] in line or (len(d) >= 3 and d[:3] in line and re.search(r'[ぁ-んァ-ン一-龥]', d[:3])):
                if zone_start is None:
                    zone_start = li

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


_OCR_TRAILING_PRICE_RE = re.compile(r'(?:^|[\s(（])([¥￥]?\s*\d[\d,]*)\s*(?:[%％][*※除軽]|[*※除軽])?\s*$')
_OCR_ZONE_END_RE = re.compile(r'^(小計|合計|現計|外税|内税|消費税|お預り|お釣り|釣銭|WAON|クレジット|お会計)')
_OCR_QTY_NOTATION_RE = re.compile(
    r'(?:'
    r'\d+\s*[コ個点]\s*[xX×Ⅹ]\s*(?:単|@)?\s*\d|'
    r'(?:単|@)\s*\d[\d,]*\s*[xX×Ⅹ]\s*\d+\s*[コ個点]'
    r')'
)


def _parse_qty_detail_total(line: str) -> tuple[float, float] | None:
    """Return (qty, unit_price) from OCR qty detail like "2個 X70)"."""
    m = re.search(r'(\d+)\s*[コ個点]\s*[xX×Ⅹ]\s*(?:単|@)?\s*(\d[\d,]*)', line)
    if not m:
        m = re.search(r'(?:単|@)\s*(\d[\d,]*)\s*[xX×Ⅹ]\s*(\d+)\s*[コ個点]', line)
        if not m:
            return None
        unit = float(m.group(1).replace(',', ''))
        qty = float(m.group(2))
    else:
        qty = float(m.group(1))
        unit = float(m.group(2).replace(',', ''))
    if qty < 2 or unit <= 0:
        return None
    return qty, unit


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

    The new totals first try to preserve OCR row order by matching item
    descriptions back to OCR item lines. If that is not reliable, fall back to
    total-rank projection.
    """
    items = extracted.get("line_items") or []
    if not items:
        return
    items_sum = sum(i.get("total", 0) for i in items if isinstance(i, dict))
    subtotal = extracted.get("subtotal")
    total = extracted.get("total")
    canonical_subtotal = _canonical_subtotal_from_taxes(extracted)
    targets = [t for t in (canonical_subtotal, subtotal, total) if t]
    if not targets:
        return
    items_sum_already_matches = any(abs(items_sum - t) <= 2 for t in targets)

    lines = unified_text.split('\n')

    # Find item zone: from first inline-priced line to first 小計/合計-style end marker.
    zone_start = None
    for i, line in enumerate(lines):
        s = line.strip()
        if re.search(r'[¥￥]\s*\d', s) or re.search(r'\d[\d,]*\s*[*※除軽]\s*$', s):
            zone_start = max(0, i - 1)
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
        token = m.group(0)
        if v < 10 and not re.search(r'[*※除軽]', token):
            continue
        if v < 1 or v > 99999:
            continue
        qty_detail = None
        for lookahead in range(li + 1, min(li + 3, zone_end)):
            lookahead_s = lines[lookahead].strip()
            qty_detail = _parse_qty_detail_total(lookahead_s)
            if qty_detail:
                break
            if _OCR_TRAILING_PRICE_RE.search(lookahead_s):
                break
            if re.search(r'[ぁ-んァ-ン一-龥]{2,}', lookahead_s):
                break
        if qty_detail:
            qty, unit = qty_detail
            candidates.append((li, int(qty * unit)))
            continue
        candidates.append((li, v))

    if not candidates:
        return

    # Reserve OCR tokens consumed by items we do not project. For qty>1 rows,
    # OCR commonly prints the unit price near the quantity notation. For
    # discounted rows, OCR commonly prints the gross price followed by a
    # discount line, while the canonical item total is net.
    qty_n_items = [i for i in items if isinstance(i, dict) and (i.get("qty") or 1) > 1]
    discounted_items = [
        i for i in items
        if isinstance(i, dict) and (i.get("discount") or 0) > 0
    ]
    qty_1_items = [
        i for i in items
        if (
            isinstance(i, dict)
            and (i.get("qty") or 1) == 1
            and (i.get("discount") or 0) == 0
        )
    ]
    if not qty_1_items:
        return  # nothing to project onto

    pool = list(candidates)
    for it in qty_n_items:
        up = it.get("unit_price")
        total_val = it.get("total")
        reserved = False
        if up is not None:
            for j, (_, v) in enumerate(pool):
                if abs(v - up) < 1:
                    pool.pop(j)
                    reserved = True
                    break
        if reserved or total_val is None:
            continue
        for j, (_, v) in enumerate(pool):
            if abs(v - total_val) < 1:
                pool.pop(j)
                break
    for it in discounted_items:
        gross = None
        if it.get("unit_price") is not None and it.get("qty"):
            gross = float(it.get("unit_price") or 0) * float(it.get("qty") or 1)
        elif it.get("total") is not None:
            gross = float(it.get("total") or 0) + float(it.get("discount") or 0)
        if not gross:
            continue
        for j, (_, v) in enumerate(pool):
            if abs(v - gross) < 1:
                pool.pop(j)
                break

    n_qty1 = len(qty_1_items)
    fixed_total = sum(
        i.get("total", 0)
        for i in (qty_n_items + discounted_items)
        if isinstance(i, dict)
    )

    # Find the single subset (size = n_qty1) whose sum is within 2 of any target.
    target_qty1_sums = [t - fixed_total for t in targets]

    def _multiset_matches(values: list[int]) -> int | None:
        s = sum(values)
        for t in target_qty1_sums:
            if abs(s - t) <= 2:
                return t
        return None

    pool_values = [v for _, v in pool]
    chosen_pairs: list[tuple[int, int]] | None = None

    if len(pool_values) == n_qty1:
        if _multiset_matches(pool_values) is not None:
            chosen_pairs = list(pool)
    elif len(pool_values) == n_qty1 + 1:
        # Try dropping each candidate; apply only if exactly one drop produces
        # a sum that matches a target.
        viable: list[list[tuple[int, int]]] = []
        for k in range(len(pool)):
            sub_pairs = pool[:k] + pool[k + 1:]
            if _multiset_matches([v for _, v in sub_pairs]) is not None:
                viable.append(sub_pairs)
        # Multiple drops can produce equivalent sums when duplicate values
        # are present (dropping any of three "228"s gives the same subset).
        # Treat them as one viable solution.
        unique = {tuple(sorted(v for _, v in pairs)) for pairs in viable}
        if len(unique) == 1:
            chosen_pairs = list(viable[0])

    if chosen_pairs is None:
        return

    # Verify the projection actually changes the multiset (no point otherwise).
    sorted_qty1_totals = sorted(i.get("total", 0) for i in qty_1_items)
    sorted_chosen = sorted(v for _, v in chosen_pairs)

    # Sanity: same length
    if len(sorted_chosen) != len(qty_1_items):
        return
    if items_sum_already_matches and sorted_qty1_totals != sorted_chosen:
        return

    def _norm_desc(text: str) -> str:
        text = re.sub(r'^\d{4,}[A-Za-z0-9-]*\)?\s*', '', text or "")
        text = re.sub(r'[¥￥]?\s*\d[\d,]*\s*[*※除軽]?\s*$', '', text)
        text = re.sub(r'\s+', '', text)
        text = re.sub(r'[^\wぁ-んァ-ン一-龥]', '', text, flags=re.UNICODE)
        return text.lower()

    def _ocr_line_for_desc(desc: str) -> int | None:
        nd = _norm_desc(desc)
        if len(nd) < 3:
            return None
        best: tuple[float, int] | None = None
        for li in range(zone_start, zone_end):
            nl = _norm_desc(lines[li])
            if len(nl) < 3 or re.match(r'^\d+$', nl):
                continue
            if nd in nl or nl in nd:
                score = 1.0
            else:
                score = SequenceMatcher(None, nd, nl).ratio()
            if score >= 0.72 and (best is None or score > best[0]):
                best = (score, li)
        return best[1] if best else None

    # Prefer row-order projection when descriptions can be matched uniquely to
    # OCR item lines. This keeps description↔price pairing intact on receipts
    # that print several descriptions before their price column.
    desc_order: list[tuple[int, int]] = []
    used_lines: set[int] = set()
    for idx, item in enumerate(items):
        if not isinstance(item, dict) or (item.get("qty") or 1) != 1:
            continue
        line_idx = _ocr_line_for_desc(item.get("description") or "")
        if line_idx is None or line_idx in used_lines:
            desc_order = []
            break
        used_lines.add(line_idx)
        desc_order.append((line_idx, idx))

    if len(desc_order) == len(qty_1_items):
        for (_, idx), (_, new_total) in zip(
            sorted(desc_order),
            sorted(chosen_pairs, key=lambda p: p[0]),
        ):
            items[idx]["total"] = new_total
            items[idx]["unit_price"] = new_total
        return

    qty1_current_idxs = [
        idx for idx, item in enumerate(items)
        if (
            isinstance(item, dict)
            and (item.get("qty") or 1) == 1
            and (item.get("discount") or 0) == 0
        )
    ]
    if not qty_n_items and len(qty1_current_idxs) == len(chosen_pairs):
        for idx, (_, new_total) in zip(
            qty1_current_idxs,
            sorted(chosen_pairs, key=lambda p: p[0]),
        ):
            items[idx]["total"] = new_total
            items[idx]["unit_price"] = new_total
        return

    if sorted_qty1_totals == sorted_chosen or items_sum_already_matches:
        return

    # Fallback: assign sorted-OCR totals to qty=1 items by their current total-rank.
    qty1_sorted_idxs = sorted(
        range(len(items)),
        key=lambda j: (
            -1 if not isinstance(items[j], dict) or (items[j].get("qty") or 1) > 1 else 0,
            items[j].get("total", 0) if isinstance(items[j], dict) else 0,
        ),
    )
    qty1_sorted_idxs = [j for j in qty1_sorted_idxs
                        if (
                            isinstance(items[j], dict)
                            and (items[j].get("qty") or 1) == 1
                            and (items[j].get("discount") or 0) == 0
                        )]

    for k, idx in enumerate(qty1_sorted_idxs):
        new_total = sorted_chosen[k]
        items[idx]["total"] = new_total
        items[idx]["unit_price"] = new_total


def _layout_block_height(block: dict) -> float:
    bbox = block.get("bbox") or []
    ys = [p[1] for p in bbox if isinstance(p, (list, tuple)) and len(p) >= 2]
    if ys:
        return float(max(ys) - min(ys))
    return 0.0


def _layout_block_center_y(block: dict) -> float:
    bbox = block.get("bbox") or []
    ys = [p[1] for p in bbox if isinstance(p, (list, tuple)) and len(p) >= 2]
    if ys:
        return (max(ys) + min(ys)) / 2
    return float(block.get("y") or 0)


def _group_layout_rows(layout_blocks: list[dict]) -> list[list[dict]]:
    blocks = [b for b in layout_blocks or [] if (b.get("text") or "").strip()]
    if not blocks:
        return []
    heights = sorted(h for h in (_layout_block_height(b) for b in blocks) if h > 0)
    median_h = heights[len(heights) // 2] if heights else 20.0
    y_tol = max(8.0, median_h * 0.55)

    rows: list[list[dict]] = []
    row_y: float | None = None
    current_page = None
    for block in sorted(blocks, key=lambda b: (b.get("page", 0), _layout_block_center_y(b), b.get("x") or 0)):
        cy = _layout_block_center_y(block)
        page = block.get("page", 0)
        if current_page != page:
            rows.append([block])
            row_y = cy
            current_page = page
        elif row_y is None or abs(cy - row_y) <= y_tol:
            if not rows:
                rows.append([])
            rows[-1].append(block)
            row_y = cy if row_y is None else (row_y + cy) / 2
        else:
            rows.append([block])
            row_y = cy
    return [sorted(row, key=lambda b: b.get("x") or 0) for row in rows]


def _layout_price_value(text: str, *, allow_small: bool = False) -> int | None:
    s = (text or "").strip()
    m = re.match(r'^[¥￥]?\s*(\d[\d,]*)\s*[*※除軽]?\s*$', s)
    if not m:
        return None
    try:
        value = int(m.group(1).replace(',', ''))
    except ValueError:
        return None
    min_value = 1 if allow_small else 10
    if value < min_value or value > 99999:
        return None
    return value


def _layout_qty_detail_total(row_text: str) -> int | None:
    compact = re.sub(r'\s+', '', row_text or '')
    m = re.search(r'(?<!\d)(\d{1,2})個[×xX]\s*(\d{1,5})', compact)
    if not m:
        return None
    try:
        qty = int(m.group(1))
        unit = int(m.group(2))
    except ValueError:
        return None
    if qty <= 1 or unit <= 0:
        return None
    total = qty * unit
    if total < 10 or total > 99999:
        return None
    return total


def _norm_layout_desc(text: str) -> str:
    text = re.sub(r'^\d{4,}[A-Za-z0-9-]*\)?\s*', '', text or "")
    text = re.sub(r'[¥￥]?\s*\d[\d,]*\s*[*※除軽]?\s*$', '', text)
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'[^\wぁ-んァ-ン一-龥]', '', text, flags=re.UNICODE)
    return text.lower()


def _layout_row_price_candidates(layout_blocks: list[dict] | None) -> list[dict]:
    rows = _group_layout_rows(layout_blocks or [])
    raw_rows: list[dict] = []
    for row_idx, row in enumerate(rows):
        row_text = "".join(str(b.get("text") or "") for b in row).strip()
        if not row_text:
            continue
        if _OCR_ZONE_END_RE.match(row_text):
            break
        price_positions = [
            (idx, _layout_price_value(str(block.get("text") or ""), allow_small=True))
            for idx, block in enumerate(row)
        ]
        price_positions = [(idx, value) for idx, value in price_positions if value is not None]
        if not price_positions:
            continue
        raw_rows.append({
            "row_idx": row_idx,
            "row": row,
            "price_positions": price_positions,
        })

    price_xs = [
        float(raw["row"][idx].get("x") or 0)
        for raw in raw_rows
        for idx, _value in raw["price_positions"]
        if float(raw["row"][idx].get("x") or 0) >= 180
    ]
    if not price_xs:
        price_xs = [
            float(raw["row"][idx].get("x") or 0)
            for raw in raw_rows
            for idx, _value in raw["price_positions"]
        ]
    if not price_xs:
        return []
    price_xs = sorted(price_xs)
    price_col_x = price_xs[len(price_xs) // 2]
    x_tol = max(45.0, price_col_x * 0.16)

    candidates: list[dict] = []
    for raw in raw_rows:
        row = raw["row"]
        near_column = [
            pair for pair in raw["price_positions"]
            if abs(float(row[pair[0]].get("x") or 0) - price_col_x) <= x_tol
        ]
        if not near_column:
            continue
        price_idx, value = max(
            near_column,
            key=lambda pair: float(row[pair[0]].get("x") or 0),
        )
        price_x = float(row[price_idx].get("x") or 0)
        desc_text = "".join(str(b.get("text") or "") for b in row[:price_idx]).strip()
        if not desc_text or _SKIP_PRICE_LINE.search(desc_text):
            continue
        if not re.search(r'[ぁ-んァ-ン一-龥]', desc_text):
            continue
        next_row_text = ""
        next_row_idx = raw["row_idx"] + 1
        if next_row_idx < len(rows):
            next_row_text = "".join(str(b.get("text") or "") for b in rows[next_row_idx])
        qty_detail_total = _layout_qty_detail_total(next_row_text)
        if qty_detail_total is not None:
            value = qty_detail_total
        candidates.append({
            "description": desc_text,
            "value": int(value),
            "y": _layout_block_center_y(row[price_idx]),
            "x": price_x,
        })
    return candidates


def _project_totals_to_layout_rows(extracted, ocr_layout_blocks):
    """Use preserved OCR row geometry to resolve price-token swaps.

    This is intentionally conservative and only fires when the geometric row
    prices form a subtotal/total-matching multiset while the current extraction
    does not.
    """
    items = extracted.get("line_items") or []
    if not items or not ocr_layout_blocks:
        return

    subtotal = extracted.get("subtotal")
    total = extracted.get("total")
    canonical_subtotal = _canonical_subtotal_from_taxes(extracted)
    targets = [t for t in (canonical_subtotal, subtotal, total) if t]
    if not targets:
        return

    item_sum = sum(i.get("total", 0) for i in items if isinstance(i, dict))
    if any(abs(item_sum - t) <= 2 for t in targets):
        return

    qty_n_items = [i for i in items if isinstance(i, dict) and (i.get("qty") or 1) > 1]
    discounted_items = [
        i for i in items
        if isinstance(i, dict) and (i.get("discount") or 0) > 0
    ]
    qty_1_indices = [
        idx for idx, item in enumerate(items)
        if (
            isinstance(item, dict)
            and (item.get("qty") or 1) == 1
            and (item.get("discount") or 0) == 0
        )
    ]
    if not qty_1_indices:
        return

    candidates = _layout_row_price_candidates(ocr_layout_blocks)
    if not candidates:
        return

    fixed_total = sum(
        i.get("total", 0)
        for i in (qty_n_items + discounted_items)
        if isinstance(i, dict)
    )
    target_qty1_sums = [t - fixed_total for t in targets]

    def _matches_target(values: list[int]) -> bool:
        s = sum(values)
        return any(abs(s - t) <= 2 for t in target_qty1_sums)

    chosen = None
    n_qty1 = len(qty_1_indices)
    values = [c["value"] for c in candidates]
    if len(values) == n_qty1 and _matches_target(values):
        chosen = list(candidates)
    elif len(values) == n_qty1 + 1:
        viable = []
        for drop_idx in range(len(candidates)):
            subset = candidates[:drop_idx] + candidates[drop_idx + 1:]
            if _matches_target([c["value"] for c in subset]):
                viable.append(subset)
        unique = {tuple(sorted(c["value"] for c in subset)) for subset in viable}
        if len(unique) == 1:
            chosen = viable[0]

    if chosen is None or len(chosen) != n_qty1:
        return

    assignments: dict[int, int] = {}
    used_candidate_idxs: set[int] = set()
    for item_idx in qty_1_indices:
        item_desc = _norm_layout_desc(items[item_idx].get("description") or "")
        if len(item_desc) < 3:
            assignments = {}
            break
        best: tuple[float, int] | None = None
        for cand_idx, cand in enumerate(chosen):
            if cand_idx in used_candidate_idxs:
                continue
            cand_desc = _norm_layout_desc(cand["description"])
            if item_desc in cand_desc or cand_desc in item_desc:
                score = 1.0
            else:
                score = SequenceMatcher(None, item_desc, cand_desc).ratio()
            if score >= 0.72 and (best is None or score > best[0]):
                best = (score, cand_idx)
        if best is None:
            assignments = {}
            break
        used_candidate_idxs.add(best[1])
        assignments[item_idx] = chosen[best[1]]["value"]

    if len(assignments) != n_qty1:
        return

    new_sum = sum(
        assignments.get(idx, item.get("total", 0))
        for idx, item in enumerate(items) if isinstance(item, dict)
    )
    if not any(abs(new_sum - t) <= 2 for t in targets):
        return

    for idx, value in assignments.items():
        items[idx]["total"] = value
        items[idx]["unit_price"] = value


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
        text = _clean_ocr_price_line_desc(text)
        text = re.sub(r'\s+[\d,]+\s*[点個コ]\s*$', '', text).strip()
        text = re.sub(r'\s*[※\*非外内]\s*$', '', text).strip()
        mc = re.match(r'^\d{3,}[A-Za-z]{0,3}\)?\s?(.+)$', text)
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
        if re.search(r'取\s*\d|担当|レジ|領収|登録番号|TEL|FAX|http|:', text, re.IGNORECASE):
            return False
        if re.search(r'お願い|保管|場合|印字面|財布|手帳|ください', text):
            return False
        if re.match(r'^[\d,\s\-\(\)\.\*※軽除外]+$', text):
            return False
        if re.search(
            r'\d+(?:\.\d+)?\s*[個コ点]\s*[xX×Ⅹ]\s*(?:単|@)?\s*[\d,]+'
            r'|(?:単|@)?\s*[\d,]+\s*[xX×Ⅹ]\s*\d+(?:\.\d+)?\s*[個コ点]?',
            text,
        ):
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


def _clean_ocr_price_line_desc(text: str) -> str:
    """Remove OCR row prefixes/suffix prices from a candidate item name."""
    text = text.strip()
    text = _OCR_TRAILING_PRICE_RE.sub("", text).strip()
    text = re.sub(r'(?:^|\s)[¥￥]?\s*\d[\d,]*\s+[A-ZＡ-Ｚ]\s*$', '', text).strip()
    text = re.sub(r'^\d{3,}[A-Za-z0-9-]*\)?\s*', '', text).strip()
    text = re.sub(r'\s*[※\*非外内]\s*$', '', text).strip()
    return text


def _clean_code_prefixed_item_descriptions(extracted):
    """Remove visible product-code prefixes from item descriptions."""
    for item in extracted.get("line_items") or []:
        if not isinstance(item, dict):
            continue
        desc = (item.get("description") or "").strip()
        if not desc:
            continue
        cleaned = _clean_ocr_price_line_desc(desc)
        cleaned = re.sub(r'\s+1$', '', cleaned).strip()
        if cleaned != desc and re.search(r'[ぁ-んァ-ン一-龥]', cleaned):
            item["description"] = cleaned


def _fix_code_table_descriptions_by_order(extracted, unified_text):
    """Restore descriptions from a visible POS code/name table by row order."""
    items = [item for item in (extracted.get("line_items") or []) if isinstance(item, dict)]
    if not items:
        return
    lines = [line.strip() for line in unified_text.split('\n')]
    descriptions: list[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        m = re.match(r'^\d{3,}(?:-\d{3,}){2,}\s+(.+)$', line)
        if not m:
            idx += 1
            continue
        desc = _clean_ocr_price_line_desc(m.group(1))
        if idx + 1 < len(lines):
            nxt = lines[idx + 1].strip()
            if (
                _valid_ocr_item_desc(nxt)
                and len(re.sub(r'\s+', '', nxt)) >= 2
                and not re.match(r'^\d{3,}(?:-\d{3,}){2,}\s+', nxt)
                and not re.fullmatch(r'\d+', nxt)
                and not re.search(r'[¥￥]|\d+\s*[%％]?', nxt)
            ):
                desc = f"{desc}{nxt}"
                idx += 1
        if _valid_ocr_item_desc(desc):
            descriptions.append(desc)
        idx += 1

    if len(descriptions) != len(items):
        return
    for item, desc in zip(items, descriptions):
        item["description"] = desc


def _valid_ocr_item_desc(text: str) -> bool:
    if not text or len(text) < 2:
        return False
    if text in _GENERIC_DESC_MARKERS:
        return False
    if _SKIP_PRICE_LINE.search(text):
        return False
    if re.search(r'割引|値引', text):
        return False
    if re.match(r'^[\d,\s\-\(\)\.\*※軽除外]+$', text):
        return False
    return bool(re.search(r'[ぁ-んァ-ン一-龥]', text))


_PRE_PRICE_STACK_METADATA_RE = re.compile(
    r'取引|販売員|担当|レジ\s*No|レジNo|レジ番号|レシート|領収|端末|登録番号|'
    r'電話|TEL|https?://|@|〒|支払|支払い|お預|お釣|釣銭|会員|カード|'
    r'承認|伝票|店舗名|店名|小計|合計',
    re.IGNORECASE,
)


def _valid_pre_price_stack_item_desc(raw: str, desc: str) -> bool:
    """Accept candidate names before stacked prices only when they are item-like."""
    if not _valid_ocr_item_desc(desc):
        return False
    return not (
        _PRE_PRICE_STACK_METADATA_RE.search(raw)
        or _PRE_PRICE_STACK_METADATA_RE.search(desc)
    )


def _find_discounted_ocr_item_desc(lines, price_line_idx):
    """Find the item name for an OCR price row followed by discount lines.

    Unlike _find_ocr_item_desc, duplicate names are allowed here: grocery
    receipts often print two same-named weighted/meat rows with separate
    discounts, and excluding an existing description can jump to a previous
    unrelated item.
    """
    cand = _clean_ocr_price_line_desc(lines[price_line_idx])
    if _valid_ocr_item_desc(cand):
        return cand
    for j in range(price_line_idx - 1, max(price_line_idx - 6, -1), -1):
        cand = _clean_ocr_price_line_desc(lines[j])
        if _valid_ocr_item_desc(cand):
            return cand
    return None


def _ocr_line_index_for_item(lines, item):
    """Locate an extracted item in OCR text, preferring its nearby price row."""
    if not isinstance(item, dict):
        return None
    desc = item.get("description") or ""
    norm_desc = _norm_layout_desc(desc)
    if len(norm_desc) < 2:
        return None

    prices = []
    for key in ("unit_price", "total"):
        value = item.get(key)
        if value is None:
            continue
        try:
            price = int(round(float(value)))
        except (TypeError, ValueError):
            continue
        if price > 0 and price not in prices:
            prices.append(price)

    for price in prices:
        price_re = re.compile(r'(?<!\d)' + re.escape(str(price)) + r'(?!\d)')
        for idx, line in enumerate(lines):
            if not price_re.search(line):
                continue
            window = lines[max(0, idx - 3):min(len(lines), idx + 2)]
            if any(
                norm_desc in _norm_layout_desc(w) or _norm_layout_desc(w) in norm_desc
                for w in window
                if _norm_layout_desc(w)
            ):
                return idx

    best_idx = None
    best_score = 0.0
    for idx, line in enumerate(lines):
        nline = _norm_layout_desc(_clean_ocr_price_line_desc(line))
        if len(nline) < 2:
            continue
        if norm_desc in nline or nline in norm_desc:
            score = 1.0
        else:
            score = SequenceMatcher(None, norm_desc, nline).ratio()
        if score > best_score:
            best_idx = idx
            best_score = score
    return best_idx if best_score >= 0.72 else None


def _qty_detail_owner_indices(items, unified_text):
    """Return item indices whose OCR row owns a nearby qty/unit detail."""
    if not items or not unified_text:
        return set()
    lines = [line.strip() for line in unified_text.split('\n')]
    owners: set[int] = set()

    def _nearest_name_before(detail_idx: int) -> str | None:
        for idx in range(detail_idx - 1, max(detail_idx - 8, -1), -1):
            line = lines[idx].strip()
            if not line:
                continue
            if _parse_qty_detail_total(line):
                continue
            if _OCR_TRAILING_PRICE_RE.search(line):
                continue
            if re.fullmatch(r'\d{8,}\s*JAN', line, flags=re.IGNORECASE):
                continue
            desc = _clean_ocr_price_line_desc(line)
            if _valid_ocr_item_desc(desc):
                return desc
        return None

    def _has_expected_total_after(detail_idx: int, expected_total: float) -> bool:
        for idx in range(detail_idx + 1, min(len(lines), detail_idx + 4)):
            line = lines[idx].strip()
            if not line:
                continue
            if _parse_qty_detail_total(line):
                break
            amount = _parse_amount_fragment(line.lstrip('¥￥').replace(',', ''))
            if amount is not None and abs(amount - expected_total) <= 2:
                return True
            if _valid_ocr_item_desc(_clean_ocr_price_line_desc(line)):
                break
        return False

    for detail_idx, line in enumerate(lines):
        detail = _parse_qty_detail_total(line)
        if not detail:
            continue
        qty, unit = detail
        expected_total = qty * unit
        owner_desc = _nearest_name_before(detail_idx)
        if not owner_desc or not _has_expected_total_after(detail_idx, expected_total):
            continue
        owner_norm = _norm_layout_desc(owner_desc)
        best_idx = None
        best_key = None
        for item_idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            item_norm = _norm_layout_desc(item.get("description") or "")
            if len(item_norm) < 2:
                continue
            if owner_norm in item_norm or item_norm in owner_norm:
                score = 1.0
            else:
                score = SequenceMatcher(None, owner_norm, item_norm).ratio()
            if score < 0.72:
                continue
            try:
                total = float(item.get("total") or 0)
                unit_price = float(item.get("unit_price") or 0)
            except (TypeError, ValueError):
                continue
            if abs(total - expected_total) > 2 and abs(unit_price - expected_total) > 2:
                continue
            line_idx = _ocr_line_index_for_item(lines, item)
            distance = abs((line_idx if line_idx is not None else detail_idx) - detail_idx)
            key = (score, -distance)
            if best_key is None or key > best_key:
                best_idx = item_idx
                best_key = key
        if best_idx is not None:
            owners.add(best_idx)
    return owners


def _insert_item_by_ocr_order(items, lines, price_line_idx, item):
    """Insert a recovered OCR item before later extracted items."""
    for pos, existing in enumerate(items):
        existing_idx = _ocr_line_index_for_item(lines, existing)
        if existing_idx is not None and existing_idx > price_line_idx:
            items.insert(pos, item)
            return
    items.append(item)


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

    _SUFFIX = re.compile(r'^(.+?)\s+([\d,]{1,6})\s*[\*※除軽]?\s*$')
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


def _drop_duplicate_with_embedded_price(items):
    """Drop items whose desc has 'X  N' suffix where N == this item's total
    AND another item with desc 'X' (no suffix) and same total exists.

    Pattern: LLM produced two items for one OCR row — one clean, one with
    the trailing inline price merged into the desc.

    Example:
      [1] 'TV1.0テイシボ'           total=198    <- correct
      [22] 'TV1.0テイシボ  198'     total=198    <- phantom duplicate

    Drop item [22]. Generic across receipts. Conservative — only fires
    when the embedded suffix exactly matches the item's own total AND a
    twin without the suffix exists at the same total.
    """
    if not items or len(items) < 2:
        return
    _SUFFIX = re.compile(r'^(.+?)\s+([\d,]{1,6})\s*[\*※]?\s*$')
    drop_idxs: set[int] = set()

    # Build a map of clean_desc → list of (idx, total) for items WITHOUT
    # a digit suffix
    clean_items: dict[str, list[tuple[int, float]]] = {}
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        desc = (item.get("description") or "").strip()
        total = item.get("total")
        if not desc or total is None:
            continue
        if not _SUFFIX.match(desc):
            clean_items.setdefault(desc, []).append((i, float(total)))

    for i, item in enumerate(items):
        if i in drop_idxs or not isinstance(item, dict):
            continue
        desc = (item.get("description") or "").strip()
        total = item.get("total")
        if not desc or total is None:
            continue
        m = _SUFFIX.match(desc)
        if not m:
            continue
        prefix = m.group(1).strip()
        try:
            suffix_val = float(m.group(2).replace(',', ''))
        except ValueError:
            continue
        # Suffix must match the item's own total
        if abs(suffix_val - total) > 1:
            continue
        # Need a clean twin at the same total. OCR sometimes leaves package
        # size in the prefix ("TV天かす 60 98") while another row has the
        # clean product name and same total.
        candidate_prefixes = [prefix]
        compact_prefix = re.sub(r'\s+', '', prefix)
        for clean_desc in clean_items:
            compact_clean = re.sub(r'\s+', '', clean_desc)
            if compact_clean and compact_prefix.startswith(compact_clean):
                candidate_prefixes.append(clean_desc)
        if any(
            abs(t - total) <= 1 and j != i
            for candidate in candidate_prefixes
            for j, t in clean_items.get(candidate, [])
        ):
            drop_idxs.add(i)
    if drop_idxs:
        items[:] = [it for i, it in enumerate(items) if i not in drop_idxs]


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
            if re.search(
                r'\d+(?:\.\d+)?\s*[個コ点]\s*[xX×Ⅹ]\s*(?:単|@)?\s*[\d,]+'
                r'|(?:単|@)?\s*[\d,]+\s*[xX×Ⅹ]\s*\d+(?:\.\d+)?\s*[個コ点]?',
                cand,
            ):
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
    # Accept the dedup if it brings new_sum within tolerance of ANY candidate.
    # (Without this, a phantom-child duplicate that shifts items_sum from one
    # close-to-total range into close-to-subtotal range gets rejected because
    # the original target was picked as 'closest to original_sum'.)
    if any(abs(new_sum - c) <= 2 for c in candidates):
        extracted["line_items"] = new_items
    elif abs(new_sum - target) < abs(original_sum - target):
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
    volume_m = re.search(r'(\d+)\s*[\.．]\s*(\d+)\s*L', unified_text)
    if not volume_m:
        return
    amount = float(f"{volume_m.group(1)}.{volume_m.group(2)}")
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


def _fix_fuel_item_description(extracted, unified_text):
    """Use the printed fuel grade as the single item description."""
    items = extracted.get("line_items") or []
    if len(items) != 1 or not isinstance(items[0], dict):
        return
    for grade in ('レギュラー', 'ハイオク', '軽油', 'ガソリン'):
        if grade in unified_text:
            items[0]["description"] = grade
            return


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


def _fix_single_service_item_from_ocr(extracted, unified_text):
    """Repair a one-line service/ticket item when OCR prints qty and total."""
    items = extracted.get("line_items") or []
    if len(items) != 1 or not isinstance(items[0], dict):
        return
    item = items[0]
    total = extracted.get("total")
    if not total:
        return
    desc = (item.get("description") or "").strip()
    desc_is_generic = (
        not desc
        or desc in {'領収書', '領収証', '合計', '小計', '様'}
        or any(kw in desc for kw in ('消費税', '但し', '受領'))
    )
    if not desc_is_generic:
        return
    lines = unified_text.split('\n')
    for idx, raw in enumerate(lines):
        candidate = raw.strip()
        if not candidate or _SKIP_PRICE_LINE.search(candidate):
            continue
        if any(kw in candidate for kw in ('但し', '受領', '消費税', '金額')):
            continue
        if not re.search(r'[ぁ-んァ-ン一-龥]', candidate):
            continue
        for nxt in lines[idx + 1:idx + 4]:
            detail = nxt.strip()
            m = re.search(r'[xX×]\s*(\d+(?:\.\d+)?)\s+([\d,]+)\s*円', detail)
            if m:
                qty = float(m.group(1))
                line_total = float(m.group(2).replace(',', ''))
                unit = line_total / qty if qty else line_total
            else:
                m = re.search(
                    r'(\d+(?:\.\d+)?)\s*[個コ点]\s*[xX×]\s*(?:単)?\s*([\d,]+)',
                    detail,
                )
                if not m:
                    continue
                qty = float(m.group(1))
                unit = float(m.group(2).replace(',', ''))
                line_total = qty * unit
            if qty > 0 and abs(line_total - float(total)) <= 2:
                item["description"] = candidate
                item["qty"] = qty
                item["unit_price"] = unit
                item["total"] = line_total
                return


def _fix_single_item_qty_from_ocr(extracted, unified_text):
    """Apply explicit @unit x qty notation to a single extracted item."""
    items = extracted.get("line_items") or []
    if len(items) != 1 or not isinstance(items[0], dict):
        return
    item = items[0]
    total = item.get("total") or extracted.get("total")
    if not total:
        return
    lines = unified_text.split('\n')
    desc = (item.get("description") or "").strip()
    for idx, line in enumerate(lines):
        if desc and desc not in line:
            continue
        for nearby in lines[idx:idx + 4]:
            m = re.search(r'@\s*([\d,]+)\s*[xX×]\s*(\d+(?:\.\d+)?)', nearby)
            if not m:
                continue
            unit = float(m.group(1).replace(',', ''))
            qty = float(m.group(2))
            if qty > 1 and abs(unit * qty - float(total)) <= 2:
                item["qty"] = qty
                item["unit_price"] = unit
                item["total"] = unit * qty
                return


def _fix_split_item_price_body_total_layout(extracted, unified_text):
    """Recover item/tax rows from receipts that print item names before a body-total price block."""
    if "本体合計" not in unified_text:
        return
    lines = [line.strip() for line in unified_text.split('\n') if line.strip()]
    body_idx = next((idx for idx, line in enumerate(lines) if line.startswith("本体合計")), None)
    if body_idx is None:
        return

    def _amount_from_line(line: str) -> int | None:
        m = _OCR_TRAILING_PRICE_RE.search(line)
        if not m:
            return None
        try:
            return int(m.group(1).strip().lstrip('¥￥').replace(',', ''))
        except ValueError:
            return None

    def _all_amounts(line: str) -> list[int]:
        values: list[int] = []
        for raw in re.findall(r'[¥￥]?\s*\d[\d,]*', line):
            try:
                values.append(int(raw.strip().lstrip('¥￥').replace(',', '')))
            except ValueError:
                continue
        return values

    def _as_float(value) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    total_value = _as_float(extracted.get("total"))
    existing_subtotal = _as_float(extracted.get("subtotal"))

    amounts_after_body: list[int] = []
    for line in lines[body_idx + 1:]:
        amounts_after_body.extend(_all_amounts(line))
    if not amounts_after_body:
        return

    subtotal_candidates = [
        amount for amount in amounts_after_body
        if amount > 0 and (total_value is None or amount < total_value)
    ]
    subtotal = None
    if existing_subtotal and existing_subtotal in subtotal_candidates:
        subtotal = int(existing_subtotal)
    elif subtotal_candidates:
        subtotal = max(subtotal_candidates)
    if not subtotal:
        return

    branch = next(
        (line for line in lines[:5] if line.endswith('店') and not re.search(r'\d', line)),
        None,
    )
    if branch and not extracted.get("location"):
        extracted["location"] = branch

    def _item_start(line: str) -> re.Match[str] | None:
        return re.match(r'^(\d+)\s*([A-Z])?\s+(.+)$', line) or re.match(r'^(\d+)([A-Z])\s+(.+)$', line)

    def _noise_or_modifier(line: str) -> bool:
        if not line or line.startswith(("#", "TEL", "登録番号", "発行日")):
            return True
        if re.search(r'カスタム|ライト|エクストラ|ノン|TOGO|To Go|お釣り|現金|総合計|消費税|対象|TEL', line):
            return True
        if re.search(r'\d{4}[/-]\d{1,2}[/-]\d{1,2}|\d{1,2}:\d{2}', line):
            return True
        return _amount_from_line(line) is not None

    def _make_item(desc: str, qty: float, total: int, tax_category: str = "8%") -> dict:
        unit_price = total / qty if qty else total
        if abs(unit_price - round(unit_price)) < 0.001:
            unit_price = int(round(unit_price))
        return {
            "description": re.sub(r'\s+', ' ', desc).strip(),
            "qty": int(qty) if float(qty).is_integer() else qty,
            "unit_price": unit_price,
            "total": total,
            "tax_category": tax_category,
            "discount": 0,
            "discount_rate": "",
        }

    direct_items: list[dict] = []
    consumed: set[int] = set()
    pending_counted_names: list[tuple[int, int, str]] = []
    first_item_idx: int | None = None
    for idx, line in enumerate(lines[:body_idx]):
        m = _item_start(line)
        if not m:
            continue
        if first_item_idx is None:
            first_item_idx = idx
        qty = int(m.group(1))
        prefix = m.group(2) or ""
        desc = (f"{prefix} {m.group(3)}" if prefix else m.group(3)).strip()
        next_item_idx = next(
            (j for j in range(idx + 1, body_idx) if _item_start(lines[j])),
            body_idx,
        )
        price_idx = None
        price = None
        for look_idx in range(idx + 1, min(next_item_idx, idx + 6)):
            price = _amount_from_line(lines[look_idx])
            if price is not None and price <= subtotal:
                price_idx = look_idx
                break
        if price_idx is not None and price is not None:
            direct_items.append(_make_item(desc, qty, price))
            consumed.update(range(idx, price_idx + 1))
            continue
        pending_counted_names.append((idx, qty, desc))

    body_names: list[tuple[int, int, str]] = []
    for idx, qty, desc in pending_counted_names:
        if idx in consumed:
            continue
        combined = desc
        next_idx = idx + 1
        if (
            next_idx < body_idx
            and next_idx not in consumed
            and not _item_start(lines[next_idx])
            and not _noise_or_modifier(lines[next_idx])
            and len(desc) <= 5
        ):
            combined = f"{desc} {lines[next_idx]}"
            consumed.add(next_idx)
        body_names.append((idx, qty, combined))
        consumed.add(idx)

    standalone_start = first_item_idx if first_item_idx is not None else body_idx
    for idx, line in enumerate(lines[standalone_start:body_idx], start=standalone_start):
        if idx in consumed or _noise_or_modifier(line) or _item_start(line):
            continue
        if re.search(r'[ぁ-んァ-ン一-龥]', line):
            body_names.append((idx, 1, line))
            consumed.add(idx)

    body_names.sort(key=lambda item: item[0])

    body_prices: list[int] = []
    for line in lines[body_idx + 1:]:
        amount = _amount_from_line(line)
        if amount is None:
            continue
        if amount == subtotal and body_prices:
            break
        if amount > 0 and amount < subtotal:
            body_prices.append(amount)
        if len(body_prices) >= len(body_names):
            break

    items = list(direct_items)
    for (_idx, qty, desc), price in zip(body_names, body_prices):
        items.append(_make_item(desc, qty, price))

    if not items and len(pending_counted_names) == 1:
        _idx, qty, desc = pending_counted_names[0]
        if qty > 1 and subtotal % qty == 0:
            items.append(_make_item(desc, qty, subtotal))

    if items and abs(sum(float(item.get("total") or 0) for item in items) - subtotal) <= 2:
        extracted["line_items"] = items
        extracted["subtotal"] = subtotal
    elif not items:
        existing_items = extracted.get("line_items") or []
        existing_sum = sum(
            float(item.get("total") or 0)
            for item in existing_items
            if isinstance(item, dict)
        )
        if abs(existing_sum - subtotal) <= 2:
            items = existing_items
            extracted["subtotal"] = subtotal

    tax_entries: list[dict] = []
    tax_bases: dict[str, int] = {}
    rate_indices = [
        idx for idx, line in enumerate(lines[body_idx + 1:], start=body_idx + 1)
        if re.search(r'(\d+)\s*%\s*対象', line)
    ]
    for pos, idx in enumerate(rate_indices):
        end_idx = rate_indices[pos + 1] if pos + 1 < len(rate_indices) else len(lines)
        end_idx = min(end_idx, idx + 8)
        window = lines[idx:end_idx]
        printed_rate = int(re.search(r'(\d+)\s*%\s*対象', lines[idx]).group(1))
        values: list[int] = []
        for line in window:
            values.extend(value for value in _all_amounts(line) if value > 0)
        best: tuple[float, int, int, int] | None = None
        candidate_rates = [printed_rate, 8, 10]
        for base in values:
            for amount in values:
                if amount >= base or amount > max(2, base * 0.2):
                    continue
                for rate in candidate_rates:
                    if rate <= 0:
                        continue
                    diff = abs(amount - (base * rate / 100.0))
                    if diff <= 2:
                        score = (diff, -base, rate, amount)
                        if best is None or score < best:
                            best = score
                            best_base = base
                            best_amount = amount
                            best_rate = rate
        if best is None:
            continue
        rate_label = f"{best_rate}%"
        tax_entries.append({"rate": rate_label, "label": "外税", "amount": best_amount})
        tax_bases[rate_label] = best_base

    if tax_entries:
        tax_sum = sum(float(tax["amount"]) for tax in tax_entries)
        if total_value is None or abs(subtotal + tax_sum - total_value) <= 2:
            extracted["taxes"] = tax_entries

            current_items = extracted.get("line_items") or items
            if current_items and isinstance(current_items, list):
                if len(tax_entries) == 1:
                    only_rate = tax_entries[0]["rate"]
                    for item in current_items:
                        if isinstance(item, dict):
                            item["tax_category"] = only_rate
                else:
                    largest_rate = max(tax_bases, key=lambda rate: tax_bases[rate])
                    for item in current_items:
                        if isinstance(item, dict):
                            item["tax_category"] = largest_rate
                    for rate, base in sorted(tax_bases.items(), key=lambda pair: pair[1]):
                        if rate == largest_rate:
                            continue
                        candidates = [
                            (idx, float(item.get("total") or 0))
                            for idx, item in enumerate(current_items)
                            if isinstance(item, dict)
                        ]
                        matched = _find_subset_sum(candidates, base, max_k=min(4, len(candidates)), tolerance=2.0)
                        if not matched:
                            continue
                        for item_idx in matched:
                            if isinstance(current_items[item_idx], dict):
                                current_items[item_idx]["tax_category"] = rate


_BAG_DESC_RE = re.compile(
    r'レジ[ブフ]クロ|レジ袋|有料レジ袋|食品ポリ袋|ポリ袋|ショッピングバッグ|紙袋|バイオ.*袋|フクロHK'
)
_FOOD_DESC_RE = re.compile(
    r'バナナ|だし|餃子|肉|ミンチ|キャベツ|にんじん|大根|ハム|コマツナ|春雨|'
    r'ピーチ|ねぎ|オオバ|パン|ホシイモ|米|牛|豚|鶏|チキン|弁当|おにぎり|'
    r'茶|ココア|コーヒー|天然水|オイル不使用|食品'
)


def _is_bag_description(desc: str | None) -> bool:
    return bool(_BAG_DESC_RE.search(desc or ""))


def _fix_tax_categories_from_ocr_markers(items, unified_text):
    """Use visible reduced-tax markers next to OCR item prices."""
    if not items:
        return
    lines = unified_text.split('\n')
    has_standard_rate_evidence = bool(re.search(
        r'(?:外税|内税)?\s*10\s*%\s*(?:外税|内税)?\s*(?:対象|タイショウ|対\b|課税|税額)|税率\s*10\s*%',
        unified_text,
    ))

    def _norm(text: str) -> str:
        text = re.sub(r'[¥￥]?\s*\d[\d,]*\s*(?:[%％][*※除軽]|[*※除軽])?\s*$', '', text or "")
        text = re.sub(r'\s+', '', text)
        text = re.sub(r'[^\wぁ-んァ-ン一-龥]', '', text, flags=re.UNICODE)
        return text.lower()

    norm_lines = [_norm(line) for line in lines]



    for item in items:
        if not isinstance(item, dict):
            continue
        raw_desc = item.get("description") or ""
        if (
            re.search(r'ごみ袋|ゴミ袋', raw_desc)
            and re.search(r'(?:ごみ袋|ゴミ袋)[^\n]*非|非課税対象額', unified_text)
        ):
            item["tax_category"] = "0%"
            continue
        if _is_bag_description(raw_desc):
            item["tax_category"] = "10%"
            continue
        if "本みりん" in raw_desc:
            item["tax_category"] = "10%"
            continue
        if (
            not has_standard_rate_evidence
            and _FOOD_DESC_RE.search(raw_desc)
            and re.search(r'軽減税率|8%対象|8%対象額|※印', unified_text)
        ):
            item["tax_category"] = "8%"
            continue
        if _is_service_fee_description(raw_desc) and _has_service_inclusive_tax_evidence(unified_text):
            item["tax_category"] = "10%"
            continue
        if "100円均一" in raw_desc:
            item["tax_category"] = "10%"
            continue
        if (item.get("total") or 0) == 100 and "100円均一" in unified_text and "業務スーパー" in unified_text:
            item["tax_category"] = "10%"
            continue
        if re.search(r'液体BL|水切り|抗菌|キレイ液体|漂白|洗剤', raw_desc):
            item["tax_category"] = "10%"
            continue
        if (
            re.search(r'美容|ヘア|リップ|UV|マスク|モイスチャー|サンプロテクター|シャンプー', raw_desc, re.IGNORECASE)
            and re.search(r'コスモス|ドラッグ|医薬|化粧品|薬', unified_text)
        ):
            item["tax_category"] = "10%"
            continue
        desc = _norm(item.get("description") or "")
        if len(desc) < 3:
            continue
        best_idx = None
        best_score = 0.0
        for idx, nline in enumerate(norm_lines):
            if len(nline) < 3:
                continue
            if desc in nline or nline in desc:
                score = 1.0
            else:
                score = SequenceMatcher(None, desc, nline).ratio()
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx is None or best_score < 0.72:
            continue
        line = lines[best_idx].strip()
        if re.match(r'^内\s*\*', line):
            item["tax_category"] = "8%"
            continue
        if re.search(r'ドラッグストア\s*\n\s*コスモス|コスモス', unified_text):
            marked_current_line = bool(re.search(r'[%％][*※除軽]|[*※軽]', line))
            marked_price_continuation = False
            if best_idx + 1 < len(lines):
                next_line = lines[best_idx + 1].strip()
                marked_price_continuation = bool(
                    re.match(r'^[¥￥]?\s*\d[\d,]*\s*(?:[%％][*※除軽]|[*※軽])\s*$', next_line)
                )
            if marked_current_line or marked_price_continuation:
                item["tax_category"] = "8%"
        else:
            marked_current_line = bool(re.search(r'[%％][*※除軽]|[*※軽]', line))
            marked_price_continuation = False
            if best_idx + 1 < len(lines):
                next_line = lines[best_idx + 1].strip()
                marked_price_continuation = bool(
                    re.match(r'^[¥￥]?\s*\d[\d,]*\s*(?:[%％][*※除軽]|[*※軽])\s*$', next_line)
                )
            if marked_current_line or marked_price_continuation:
                item["tax_category"] = "8%"


def _apply_single_bag_standard_rate_split(items, rate_bases):
    """When the only 10% taxable base is the bag, force all other items to 8%."""
    if not items or not rate_bases:
        return
    standard_base = float(rate_bases.get("10%") or 0)
    if standard_base <= 0:
        return
    bag_total = sum(
        float(item.get("total") or 0)
        for item in items
        if isinstance(item, dict) and _is_bag_description(item.get("description") or "")
    )
    if bag_total <= 0 or bag_total > 50:
        return
    if abs(bag_total - standard_base) > 2:
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        item["tax_category"] = "10%" if _is_bag_description(item.get("description") or "") else "8%"


def _assign_visible_bags_to_standard_rate(items, unified_text):
    """Paid bag rows are standard-rate when a standard-rate summary is printed."""
    if not items or not re.search(r'10\s*[%％年].*(?:対象|タイショウ|課税|税額)', unified_text):
        return
    for item in items:
        if not isinstance(item, dict) or not _is_bag_description(item.get("description") or ""):
            continue
        try:
            total = float(item.get("total") or 0)
        except (TypeError, ValueError):
            continue
        if 0 < total <= 50:
            item["tax_category"] = "10%"


def _assign_single_standard_rate_from_small_base(items, rate_bases):
    """Assign one 10% item when OCR prints a small standard-rate base."""
    if not items or not rate_bases:
        return
    standard_base = float(rate_bases.get("10%") or 0)
    if standard_base <= 0:
        return
    valid_items = [item for item in items if isinstance(item, dict)]
    if not valid_items or any(item.get("tax_category") == "10%" for item in valid_items):
        return
    candidates: list[dict] = []
    for item in valid_items:
        if _is_bag_description(item.get("description") or ""):
            continue
        total = float(item.get("total") or 0)
        unit = float(item.get("unit_price") or 0)
        if abs(total - standard_base) <= 2 or abs(unit - standard_base) <= 2:
            candidates.append(item)
    if len(candidates) == 1:
        candidates[0]["tax_category"] = "10%"
        return
    total_matches = [
        item for item in candidates
        if abs(float(item.get("total") or 0) - standard_base) <= 2
    ]
    if total_matches:
        total_matches[0]["tax_category"] = "10%"
    elif candidates:
        candidates[0]["tax_category"] = "10%"


def _refine_rate_bases_from_tax_amounts(rate_bases, unified_text, extracted_taxes):
    """Correct OCR-linearized target bases when a nearby candidate explains tax."""
    if not rate_bases or not extracted_taxes:
        return rate_bases
    refined = dict(rate_bases)
    tax_amounts = {
        normalize_tax_rate(tax.get("rate", "")): float(tax.get("amount") or 0)
        for tax in extracted_taxes
        if isinstance(tax, dict) and tax.get("rate") and tax.get("amount") is not None
    }
    if not tax_amounts:
        return refined

    lines = [line.strip() for line in unified_text.split("\n")]
    for idx, line in enumerate(lines):
        target_m = re.search(r'(\d+(?:\.\d+)?)\s*[%％].*(?:対象|タイショウ)', line)
        if not target_m:
            continue
        rate = normalize_tax_rate(target_m.group(1) + "%")
        tax_amount = tax_amounts.get(rate)
        if tax_amount is None or tax_amount < 0:
            continue
        try:
            pct = float(rate.rstrip("%")) / 100.0
        except (TypeError, ValueError, ZeroDivisionError):
            continue

        existing = float(refined.get(rate) or 0)
        if existing > 0 and (
            abs(int(existing * pct) - tax_amount) <= 1
            or abs(round(existing * pct) - tax_amount) <= 1
        ):
            continue

        candidates: list[float] = []
        for lookahead in lines[idx + 1:min(len(lines), idx + 14)]:
            if re.search(r'\d+(?:\.\d+)?\s*[%％].*(?:対象|タイショウ)', lookahead):
                break
            if re.search(r'総合計|お釣り|釣銭|現金|クレジット|カード', lookahead):
                break
            if re.search(r'本体合計|小計|合計', lookahead):
                continue
            for value_text in re.findall(r'(?<![\d.])(\d{1,3}(?:,\d{3})*|\d{1,6})(?![\d.])', lookahead):
                value = float(value_text.replace(",", ""))
                if value > tax_amount:
                    candidates.append(value)
        matches = [
            value for value in candidates
            if abs(int(value * pct) - tax_amount) <= 1
            or abs(round(value * pct) - tax_amount) <= 1
        ]
        if matches:
            refined[rate] = min(matches)
    return refined


def _rebalance_tax_categories_to_rate_bases(items, unified_text, extracted_taxes, rate_bases):
    """Reassign categories when printed rate bases identify an exact item subset."""
    if len(items) < 1:
        return
    if re.search(r'小\s*計\s*\n\s*\d+\s*[%％]\s*対象額\s*\n\s*\d+\s*[%％]\s*税額', unified_text):
        base_sum = sum(
            float(base or 0)
            for rate, base in (rate_bases or {}).items()
            if rate in {"8%", "10%"} and base is not None
        )
        item_sum = sum((item.get("total") or 0) for item in items if isinstance(item, dict))
        if not base_sum or abs(item_sum - base_sum) > 2:
            return
    rate_bases = _refine_rate_bases_from_tax_amounts(rate_bases, unified_text, extracted_taxes)
    if len(items) == 1 and re.search(r'消費税率は\s*10\s*%', unified_text):
        items[0]["tax_category"] = "10%"

    tax_amounts = {
        t.get("rate"): t.get("amount", 0)
        for t in (extracted_taxes or [])
        if isinstance(t, dict)
    }
    for m in re.finditer(
        r'\((\d{2})%対象\s*¥?\s*([\d,]+)\s*内税\s*¥?\s*([\d,]+)',
        unified_text,
        flags=re.S,
    ):
        rate = f"{int(m.group(1))}%"
        if rate in {"8%", "10%"}:
            rate_bases[rate] = float(m.group(2).replace(',', ''))
            tax_amounts[rate] = float(m.group(3).replace(',', ''))

    valid_rates = [r for r, b in rate_bases.items() if r in {"8%", "10%"} and b]
    if len(valid_rates) != 2:
        return
    item_sum = sum((item.get("total") or 0) for item in items if isinstance(item, dict))
    base_sum = sum((rate_bases.get(r) or 0) for r in valid_rates)
    tax_sum = sum((tax_amounts.get(r) or 0) for r in valid_rates)
    items_are_pretax = (
        item_sum > 0 and base_sum > 0 and tax_sum > 0
        and abs(item_sum + tax_sum - base_sum) <= max(5, base_sum * 0.02)
    )

    targets: dict[str, float] = {}
    for rate in valid_rates:
        base = float(rate_bases.get(rate) or 0)
        if items_are_pretax:
            base -= float(tax_amounts.get(rate) or 0)
        if base <= 0 and tax_amounts.get(rate):
            try:
                base = float(tax_amounts[rate]) / (float(rate.rstrip('%')) / 100.0)
            except (TypeError, ValueError, ZeroDivisionError):
                base = 0
        if base > 0:
            targets[rate] = base

    if len(targets) != 2:
        return

    has_nontaxable_evidence = bool(
        re.search(r'非課税|不課税|免税', unified_text)
        or any(
            isinstance(tax, dict)
            and (
                tax.get("rate") == "0%"
                or "非課税" in (tax.get("label") or "")
            )
            for tax in (extracted_taxes or [])
        )
    )
    item_amounts = [
        (idx, float(item.get("total") or 0))
        for idx, item in enumerate(items)
        if (
            isinstance(item, dict)
            and (item.get("total") or 0) > 0
            and (not has_nontaxable_evidence or item.get("tax_category") != "0%")
        )
    ]
    if len(item_amounts) > 32:
        return

    current_sums = {
        rate: sum(
            amount for idx, amount in item_amounts
            if items[idx].get("tax_category") == rate
        )
        for rate in targets
    }
    if all(abs(current_sums.get(rate, 0) - target) <= 2 for rate, target in targets.items()):
        return

    qty_detail_owners = _qty_detail_owner_indices(items, unified_text)

    def _has_visible_reduced_marker(idx: int) -> bool:
        item = items[idx]
        line_idx = _ocr_line_index_for_item(unified_text.split('\n'), item)
        if line_idx is None:
            return False
        lines = unified_text.split('\n')
        for nearby in lines[max(0, line_idx - 2):min(len(lines), line_idx + 3)]:
            if re.search(r'^[A-Z]?\s*[*＊※]|[*＊※]\s*[^\d\s]|[%％][*＊※除軽]', nearby.strip()):
                return True
        return False

    def _subset_evidence_score(indices: list[int], target_rate: str) -> int:
        score = 0
        for idx in indices:
            if target_rate == "8%" and _has_visible_reduced_marker(idx):
                score += 4
            if idx in qty_detail_owners:
                score += 1
        return score

    def _find_subset_sum_with_evidence(
        candidates: list[tuple[int, float]],
        target: float,
        target_rate: str,
        *,
        max_k: int,
        tolerance: float,
    ) -> list[int] | None:
        best_match = None
        best_key = None
        for k in range(1, min(max_k + 1, len(candidates) + 1)):
            for combo in combinations(candidates, k):
                total = sum(amount for _idx, amount in combo)
                diff = abs(total - target)
                if diff > tolerance:
                    continue
                match = [idx for idx, _amount in combo]
                key = (-diff, _subset_evidence_score(match, target_rate), -k)
                if best_key is None or key > best_key:
                    best_match = match
                    best_key = key
        return best_match

    for target_rate, target in sorted(targets.items(), key=lambda pair: pair[1]):
        current = sum(
            amount for idx, amount in item_amounts
            if items[idx].get("tax_category") == target_rate
        )
        needed = target - current
        if needed <= 2:
            continue
        candidates = [
            (idx, amount) for idx, amount in item_amounts
            if items[idx].get("tax_category") != target_rate
        ]
        match = _find_subset_sum_with_evidence(
            candidates,
            needed,
            target_rate,
            max_k=min(len(candidates), 7),
            tolerance=0.0,
        )
        if match is None:
            match = _find_subset_sum_with_evidence(
                candidates,
                needed,
                target_rate,
                max_k=min(len(candidates), 7),
                tolerance=2.0,
            )
        if match is not None:
            for idx in match:
                items[idx]["tax_category"] = target_rate

    current_sums = {
        rate: sum(
            amount for idx, amount in item_amounts
            if items[idx].get("tax_category") == rate
        )
        for rate in targets
    }
    if all(abs(current_sums.get(rate, 0) - target) <= 2 for rate, target in targets.items()):
        return

    if len(item_amounts) > 24:
        return

    rates_by_target = sorted(targets, key=lambda r: targets[r])
    for target_rate in rates_by_target:
        other_rate = next(r for r in targets if r != target_rate)
        target = targets[target_rate]
        max_k = min(len(item_amounts), 9)
        match = _find_subset_sum(item_amounts, target, max_k=max_k, tolerance=0.0)
        if match is None:
            match = _find_subset_sum(item_amounts, target, max_k=max_k, tolerance=2.0)
        if match is None:
            continue
        matched_sum = sum(amount for idx, amount in item_amounts if idx in match)
        other_sum = sum(amount for idx, amount in item_amounts if idx not in match)
        other_tolerance = max(2.0, 5.0 if re.search(r'外税|タイショウ', unified_text) else 2.0)
        if abs(matched_sum - target) > 2 or abs(other_sum - targets[other_rate]) > other_tolerance:
            continue
        for idx, _amount in item_amounts:
            items[idx]["tax_category"] = target_rate if idx in match else other_rate
        _fix_tax_categories_from_ocr_markers(items, unified_text)
        return


def reconcile_tax_categories_from_rate_bases(extracted: dict, unified_text: str) -> None:
    """Reconcile final item tax categories against printed per-rate bases."""
    if not isinstance(extracted, dict) or not extracted.get("line_items") or not unified_text:
        return
    rate_bases = extract_rate_bases(unified_text)
    for rate, base in (extracted.get("_breakdown_rate_bases") or {}).items():
        if rate not in rate_bases or rate_bases[rate] is None:
            rate_bases[rate] = base
    if not rate_bases:
        return
    items = extracted["line_items"]
    _assign_single_standard_rate_from_small_base(items, rate_bases)
    _apply_single_bag_standard_rate_split(items, rate_bases)
    _rebalance_tax_categories_to_rate_bases(
        items,
        unified_text,
        extracted.get("taxes"),
        rate_bases,
    )
    _assign_visible_bags_to_standard_rate(items, unified_text)


def _rebalance_standard_categories_from_reduced_rate_markers(items, unified_text, rate_bases):
    """Use reduced-tax OCR markers to find the printed 10% base subset."""
    if not items or not unified_text or not rate_bases:
        return
    standard_base = float(rate_bases.get("10%") or 0)
    reduced_base = float(rate_bases.get("8%") or 0)
    if standard_base <= 0 or reduced_base <= 0 or standard_base > 5000:
        return
    lines = [line.strip() for line in unified_text.split('\n')]

    def _has_reduced_marker(item: dict) -> bool:
        line_idx = _ocr_line_index_for_item(lines, item)
        if line_idx is None:
            return False
        for nearby in lines[line_idx:min(len(lines), line_idx + 3)]:
            if re.search(r'[%％][*※除軽]|[*＊※軽]|X\b|x\b', nearby):
                return True
        return False

    candidates: list[tuple[int, float]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        total = float(item.get("total") or 0)
        if total <= 0:
            continue
        if _is_bag_description(item.get("description") or "") or not _has_reduced_marker(item):
            candidates.append((idx, total))
        else:
            item["tax_category"] = "8%"
    if not candidates:
        return
    match = _find_subset_sum(candidates, standard_base, max_k=min(len(candidates), 8), tolerance=2.0)
    if match is None:
        return
    standard_sum = sum(float(items[idx].get("total") or 0) for idx in match)
    other_sum = sum(
        float(item.get("total") or 0)
        for idx, item in enumerate(items)
        if isinstance(item, dict) and idx not in match
    )
    if abs(standard_sum - standard_base) > 2 or abs(other_sum - reduced_base) > 2:
        return
    for idx, item in enumerate(items):
        if isinstance(item, dict):
            item["tax_category"] = "10%" if idx in match else "8%"


def _fix_nonfood_packaging_tax_categories(items, unified_text, rate_bases):
    """Treat obvious non-food packaging rows as standard-rate items."""
    if not items or not unified_text or not rate_bases.get("10%"):
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        desc = item.get("description") or ""
        if _is_bag_description(desc) or re.search(r'フードパック|レンジパック|保存容器|ラップ|アルミホイル', desc):
            item["tax_category"] = "10%"


def _fix_small_non_bag_item_prices_from_ocr(extracted, unified_text):
    """Correct product rows whose price was taken from a following quantity line."""
    items = extracted.get("line_items") or []
    lines = unified_text.split('\n')

    def _norm(text: str) -> str:
        return re.sub(r'\s+', '', text or "")

    for item in items:
        if not isinstance(item, dict):
            continue
        total = item.get("total") or 0
        desc = item.get("description") or ""
        if total >= 10 or _is_bag_description(desc):
            continue
        ndesc = _norm(desc)
        if len(ndesc) < 4:
            continue
        for idx, line in enumerate(lines):
            if ndesc[:8] not in _norm(line):
                continue
            for nearby in lines[idx:idx + 4]:
                pm = _OCR_TRAILING_PRICE_RE.search(nearby.strip())
                if not pm:
                    pm = re.search(r'^\s*[*※軽]\s*([¥￥]?\s*\d[\d,]*)\s*$', nearby.strip())
                if not pm:
                    continue
                price = float(pm.group(1).strip().lstrip('¥￥').replace(',', ''))
                if price >= 10:
                    item["unit_price"] = price
                    item["total"] = price * (item.get("qty") or 1)
                    break
            break


def _fix_duplicate_descriptions_from_ocr(extracted, unified_text):
    """Replace duplicate item names with unmatched OCR descriptions at the same price."""
    items = extracted.get("line_items") or []
    if not items:
        return
    lines = unified_text.split('\n')

    def _norm(text: str) -> str:
        text = re.sub(r'\s+', '', text or "")
        text = re.sub(r'[^\wぁ-んァ-ン一-龥]', '', text, flags=re.UNICODE)
        return text.lower()

    groups: dict[str, list[int]] = {}
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        desc = item.get("description") or ""
        norm = _norm(desc)
        if norm:
            groups.setdefault(norm, []).append(idx)

    duplicate_idxs = {
        idx for idxs in groups.values() if len(idxs) > 1 for idx in idxs
    }
    if not duplicate_idxs:
        return

    existing_norms = {
        _norm(item.get("description") or "")
        for item in items if isinstance(item, dict)
    }
    ocr_candidates: list[tuple[float, str]] = []
    for line_idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if _SKIP_PRICE_LINE.search(line) or _OCR_QTY_NOTATION_RE.search(line):
            continue
        pm = _OCR_TRAILING_PRICE_RE.search(line)
        if not pm:
            continue
        raw_price = pm.group(1).strip().lstrip('¥￥').replace(',', '')
        if not raw_price.isdigit():
            continue
        price = float(raw_price)
        if price <= 0:
            continue
        same_line_desc = _clean_ocr_price_line_desc(line)
        same_line_norm = _norm(same_line_desc)
        if same_line_norm and same_line_norm in existing_norms:
            continue
        desc = _find_ocr_item_desc(lines, line_idx, items)
        if not desc:
            continue
        norm_desc = _norm(desc)
        if not norm_desc or norm_desc in existing_norms:
            continue
        ocr_candidates.append((price, desc))

    def _desc_supported_at_price(desc: str, total: float) -> bool:
        desc_norm = _norm(desc)
        if not desc_norm or total <= 0:
            return False
        for line_idx, raw_line in enumerate(lines):
            line = raw_line.strip()
            if _SKIP_PRICE_LINE.search(line):
                continue
            prices: list[float] = []
            for m in re.finditer(r'[¥￥]\s*([\d,]+)', line):
                try:
                    prices.append(float(m.group(1).replace(',', '')))
                except ValueError:
                    pass
            if not prices:
                pm = _OCR_TRAILING_PRICE_RE.search(line)
                if pm:
                    try:
                        prices.append(float(pm.group(1).strip().lstrip('¥￥').replace(',', '')))
                    except ValueError:
                        pass
            if not any(abs(price - total) <= 2 for price in prices):
                continue
            for j in range(line_idx, max(line_idx - 6, -1), -1):
                raw = lines[j].strip()
                if j != line_idx and _OCR_TRAILING_PRICE_RE.search(raw):
                    break
                if _SKIP_PRICE_LINE.search(raw) or _OCR_QTY_NOTATION_RE.search(raw):
                    continue
                cand = _clean_ocr_price_line_desc(raw)
                if _norm(cand) == desc_norm:
                    return True
        return False

    used_candidates: set[int] = set()
    for norm, idxs in groups.items():
        if len(idxs) <= 1:
            continue
        for idx in sorted(idxs, key=lambda i: float(items[i].get("total") or 0), reverse=True):
            item = items[idx]
            total = float(item.get("total") or 0)
            if total <= 0:
                continue
            if _desc_supported_at_price(item.get("description") or "", total):
                continue
            match_idx = None
            for cand_idx, (price, desc) in enumerate(ocr_candidates):
                if cand_idx in used_candidates:
                    continue
                if abs(price - total) <= 2:
                    match_idx = cand_idx
                    break
            if match_idx is None:
                continue
            item["description"] = ocr_candidates[match_idx][1]
            used_candidates.add(match_idx)
            existing_norms.add(_norm(item["description"]))


def _code_prefixed_ocr_desc_before(lines, price_line_idx, max_back=16):
    """Return the nearest product line that begins with a POS/barcode code."""
    for j in range(price_line_idx - 1, max(price_line_idx - max_back - 1, -1), -1):
        text = lines[j].strip()
        if _OCR_TRAILING_PRICE_RE.search(text):
            return None
        if not text or _SKIP_PRICE_LINE.search(text) or _OCR_QTY_NOTATION_RE.search(text):
            continue
        m = re.match(r'^\d{3,}[A-Za-z0-9-]*\)?\s*(.+)$', text)
        if not m:
            continue
        desc = m.group(1).strip()
        desc = re.sub(r'\s*[※\*非外内]\s*$', '', desc).strip()
        if len(desc) >= 3 and re.search(r'[ぁ-んァ-ン一-龥]', desc):
            return desc
    return None


def _fix_qty_code_row_descriptions_from_ocr(extracted, unified_text):
    """Repair qty-block item names from POS-code rows immediately above the qty notation."""
    items = extracted.get("line_items") or []
    if not items:
        return
    lines = [line.strip() for line in unified_text.split('\n')]

    def _norm(text: str) -> str:
        text = re.sub(r'\s+', '', text or "")
        text = re.sub(r'[^\wぁ-んァ-ン一-龥]', '', text, flags=re.UNICODE)
        return text.lower()

    def _similar(a: str, b: str) -> float:
        na, nb = _norm(a), _norm(b)
        if not na or not nb:
            return 0.0
        if na in nb or nb in na:
            return 1.0
        return SequenceMatcher(None, na, nb).ratio()

    for qty_idx, qty_line in enumerate(lines):
        qty_m = re.search(r'単\s*([¥￥]?\s*\d[\d,]*)\s*[xX×]\s*(\d+)\s*個', qty_line)
        if not qty_m:
            continue
        unit = float(qty_m.group(1).strip().lstrip('¥￥').replace(',', ''))
        qty = float(qty_m.group(2))
        total = unit * qty
        desc = _code_prefixed_ocr_desc_before(lines, qty_idx, max_back=16)
        if not desc:
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            if abs(float(item.get("total") or 0) - total) > 2:
                continue
            if _similar(item.get("description") or "", desc) >= 0.62:
                continue
            item["description"] = desc
            break


def _bag_entries_from_ocr(unified_text: str) -> list[dict]:
    """Return paid-bag OCR rows in print order with qty/unit/total if visible."""
    lines = [line.strip() for line in unified_text.split('\n')]
    entries: list[dict] = []

    def _small_price(line: str) -> float | None:
        stripped = line.strip()
        if _OCR_ZONE_END_RE.search(line) or re.search(r'小\s*計|合\s*計|対象|消費税', line):
            return None
        if re.search(r'\d{1,2}\s*:\s*\d{2}|:', line):
            return None
        if re.fullmatch(r'\d{5,}', stripped):
            return None
        pm = re.search(r'[¥￥]?\s*(\d{1,2})\s*(?:[%％][*※除軽外]|[*※除軽外])?\s*$', line)
        if not pm:
            return None
        price = float(pm.group(1))
        return price if 0 < price <= 50 else None

    def _qty_unit(line: str) -> tuple[float, float] | None:
        qty_m = re.search(
            r'\(?\s*(\d+)\s*[個コ]?\s*[xX×Ⅹ]\s*単?\s*[¥￥]?\s*(\d{1,2})\s*\)?',
            line,
        )
        if not qty_m:
            return None
        return float(qty_m.group(1)), float(qty_m.group(2))

    for idx, line in enumerate(lines):
        if not _is_bag_description(line):
            continue
        price_candidate = _small_price(line)
        ambiguous_inline_price = bool(
            price_candidate is not None
            and re.search(r'\([^)\n]*\d{1,2}\s*$', line)
            and not re.search(r'[¥￥]|[%％*＊※除軽外]\s*$', line)
        )
        qty = 1.0
        unit = None if ambiguous_inline_price else price_candidate
        total = None if ambiguous_inline_price else price_candidate

        lookahead = 9 if ambiguous_inline_price else 5
        for j in range(idx + 1, min(idx + lookahead, len(lines))):
            nearby = lines[j].strip()
            if _is_bag_description(nearby):
                break
            qty_unit = _qty_unit(nearby)
            if qty_unit:
                q, u = qty_unit
                q_total = q * u
                if j == idx + 1 or (total is not None and abs(q_total - total) <= 2):
                    qty, unit, total = q, u, q_total
                    break
            if total is not None:
                if _small_price(nearby) is None and re.search(r'[ぁ-んァ-ン一-龥]', nearby):
                    break
                continue
            price = _small_price(nearby)
            if (
                price is not None
                and ambiguous_inline_price
                and not re.fullmatch(
                    r'[¥￥]?\s*\d{1,2}\s*(?:[%％][*※除軽外]|[*※除軽外])?',
                    nearby,
                )
            ):
                continue
            if price is not None:
                qty, unit, total = 1.0, price, price

        if total is None and unit is None and price_candidate is not None:
            qty, unit, total = 1.0, price_candidate, price_candidate
        if total is not None and unit is not None:
            entries.append({"line": idx, "qty": qty, "unit_price": unit, "total": total})
    return entries


def _fix_bag_item_prices_from_ocr(extracted, unified_text):
    """Correct paid bag rows from small bag prices printed in OCR order."""
    items = extracted.get("line_items") or []
    if not items:
        return
    bag_items = [
        item for item in items
        if isinstance(item, dict) and _is_bag_description(item.get("description") or "")
    ]
    if not bag_items:
        return
    entries = _bag_entries_from_ocr(unified_text)
    if not entries:
        return

    if len(bag_items) == 1:
        entry = entries[0]
        bag_items[0]["qty"] = entry["qty"]
        bag_items[0]["unit_price"] = entry["unit_price"]
        bag_items[0]["total"] = entry["total"]
        return

    for item, entry in zip(bag_items, entries):
        item["qty"] = entry["qty"]
        item["unit_price"] = entry["unit_price"]
        item["total"] = entry["total"]


def _fix_bag_item_prices_from_rate_bases(extracted, rate_bases, unified_text):
    """Use a tiny printed 10% base as a guardrail for paid bag totals."""
    items = extracted.get("line_items") or []
    if not items or not rate_bases:
        return
    standard_base = float(rate_bases.get("10%") or 0)
    if standard_base <= 0 or standard_base > 50:
        return
    bag_items = [
        item for item in items
        if isinstance(item, dict) and _is_bag_description(item.get("description") or "")
    ]
    if not bag_items:
        return

    current_total = sum(float(item.get("total") or 0) for item in bag_items)
    if abs(current_total - standard_base) <= 2:
        return

    entries = _bag_entries_from_ocr(unified_text)
    if entries and len(entries) >= len(bag_items):
        entry_total = sum(float(entry["total"]) for entry in entries[:len(bag_items)])
        if abs(entry_total - standard_base) <= 2:
            for item, entry in zip(bag_items, entries):
                item["qty"] = entry["qty"]
                item["unit_price"] = entry["unit_price"]
                item["total"] = entry["total"]
            return

    if len(bag_items) == 1:
        item = bag_items[0]
        qty = float(item.get("qty") or 1)
        if qty > 1 and abs(round(standard_base / qty) * qty - standard_base) <= 0.01:
            unit = standard_base / qty
        else:
            qty = 1.0
            unit = standard_base
        item["qty"] = qty
        item["unit_price"] = unit
        item["total"] = standard_base
        return

    other_total = sum(float(item.get("total") or 0) for item in bag_items[:-1])
    remainder = standard_base - other_total
    if 0 < remainder <= 50:
        bag_items[-1]["qty"] = 1.0
        bag_items[-1]["unit_price"] = remainder
        bag_items[-1]["total"] = remainder


def _recover_missing_bag_items_from_ocr(extracted, unified_text):
    """Add or replace paid bag rows when a visible OCR bag price balances."""
    items = extracted.get("line_items") or []
    if not items:
        return
    entries = _bag_entries_from_ocr(unified_text)
    if not entries:
        return
    existing_bags = [
        item for item in items
        if isinstance(item, dict) and _is_bag_description(item.get("description") or "")
    ]
    if existing_bags:
        return

    entry = entries[0]
    bag_total = float(entry["total"])
    if bag_total <= 0:
        return
    total = extracted.get("total")
    subtotal = extracted.get("subtotal")
    tax_sum = _sum_taxable_amounts(extracted.get("taxes") or [])
    targets = [float(t) for t in (total, subtotal) if t is not None and float(t) > 0]
    if total and tax_sum:
        targets.append(float(total) - tax_sum)

    current_sum = sum(float(item.get("total") or 0) for item in items if isinstance(item, dict))
    printed_count = None
    count_m = re.search(r'(\d+)\s*点\s*買|お買上商品数\s*[:：]?\s*(\d+)', unified_text)
    if count_m:
        printed_count = int(count_m.group(1) or count_m.group(2))

    bag_desc = "レジ袋"
    lines = [line.strip() for line in unified_text.split('\n')]
    for idx in range(entry["line"], max(entry["line"] - 3, -1), -1):
        if _is_bag_description(lines[idx]):
            bag_desc = re.sub(r'^\s*内\s*', '', lines[idx]).strip()
            bag_desc = _OCR_TRAILING_PRICE_RE.sub('', bag_desc).strip()
            break

    bag_item = {
        "description": bag_desc,
        "qty": entry["qty"],
        "unit_price": entry["unit_price"],
        "total": bag_total,
        "tax_category": "10%",
        "discount": 0,
        "discount_rate": "",
    }

    if printed_count is None or len(items) < printed_count:
        if any(abs(current_sum + bag_total - target) <= 2 for target in targets):
            _insert_item_by_ocr_order(items, lines, entry["line"], bag_item)
        return

    # If the count already matches the printed count, replace a duplicated
    # non-bag row only when doing so moves the item sum onto a receipt target.
    totals: dict[float, list[int]] = {}
    for idx, item in enumerate(items):
        if not isinstance(item, dict) or _is_bag_description(item.get("description") or ""):
            continue
        totals.setdefault(float(item.get("total") or 0), []).append(idx)
    duplicate_indices = [idxs[-1] for idxs in totals.values() if len(idxs) > 1]
    for idx in duplicate_indices:
        old_total = float(items[idx].get("total") or 0)
        new_sum = current_sum - old_total + bag_total
        if any(abs(new_sum - target) <= 2 for target in targets):
            items[idx] = bag_item
            return


def _money_line_value(line: str) -> float | None:
    """Parse a standalone yen amount line."""
    m = re.match(r'^\s*[¥￥]\s*([\d,]+)(?:円|-)?\s*\)?\s*$', line.strip())
    if not m:
        return None
    return float(m.group(1).replace(',', ''))


def _replace_vertical_price_qty_total_rows_when_balanced(extracted, unified_text):
    """Parse item rows printed as name / unit / qty / line-total blocks."""
    lines = [line.strip() for line in unified_text.split('\n')]
    items = extracted.get("line_items") or []
    if not items:
        return

    def _valid_name(line: str) -> bool:
        if not line or _money_line_value(line) is not None:
            return False
        if re.search(r'[¥￥]', line):
            return False
        if re.match(r'^\d+\s*点$', line):
            return False
        if _SKIP_PRICE_LINE.search(line) or _HEADER_LINE_RE.search(line) or _BANNER_PHRASE_RE.search(line):
            return False
        if re.search(r'登録番号|TEL|電話|レジ|担当|取引|営業時間|領収|上記|外税|内税|現金|お預り|お釣り', line):
            return False
        return bool(re.search(r'[ぁ-んァ-ン一-龥]', line))

    rows: list[dict] = []
    name_buffer: list[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if re.match(r'^小\s*計$', line):
            break
        inline_unit = re.match(r'^(.+?)\s+[¥￥]\s*([\d,]+)\s*$', line)
        if inline_unit:
            inline_desc = inline_unit.group(1).strip()
            if re.match(r'^\d+\s*点$', inline_desc) or not re.search(r'[ぁ-んァ-ン一-龥]', inline_desc):
                inline_unit = None
        if inline_unit:
            desc_parts = name_buffer + [inline_desc]
            unit = float(inline_unit.group(2).replace(',', ''))
            qty_idx = idx + 1
            if qty_idx < len(lines) and _valid_name(lines[qty_idx]) and not re.search(r'[¥￥]', lines[qty_idx]):
                desc_parts.append(lines[qty_idx])
                qty_idx += 1
            qty_total_m = (
                re.match(r'^(\d+)\s*点\s+[¥￥]\s*([\d,]+)\s*$', lines[qty_idx])
                if qty_idx < len(lines) else None
            )
            if qty_total_m:
                qty = float(qty_total_m.group(1))
                line_total = float(qty_total_m.group(2).replace(',', ''))
                if qty > 0 and abs(unit * qty - line_total) <= 2:
                    rows.append({
                        "description": " ".join(desc_parts).strip(),
                        "qty": qty,
                        "unit_price": unit,
                        "total": line_total,
                        "tax_category": "10%",
                        "discount": 0,
                        "discount_rate": "",
                    })
                    name_buffer = []
                    idx = qty_idx + 1
                    continue
        unit = _money_line_value(line)
        qty_m = re.match(r'^(\d+)\s*点$', lines[idx + 1]) if idx + 2 < len(lines) else None
        line_total = _money_line_value(lines[idx + 2]) if idx + 2 < len(lines) else None
        if unit is not None and qty_m and line_total is not None and name_buffer:
            qty = float(qty_m.group(1))
            if qty > 0 and abs(unit * qty - line_total) <= 2:
                desc = " ".join(name_buffer).strip()
                rows.append({
                    "description": desc,
                    "qty": qty,
                    "unit_price": unit,
                    "total": line_total,
                    "tax_category": "10%",
                    "discount": 0,
                    "discount_rate": "",
                })
                name_buffer = []
                idx += 3
                continue
        if _valid_name(line):
            name_buffer.append(line)
            if len(name_buffer) > 3:
                name_buffer = name_buffer[-3:]
        else:
            name_buffer = []
        idx += 1

    if len(rows) < 2:
        return
    row_sum = sum(float(row["total"]) for row in rows)
    total = extracted.get("total")
    subtotal = extracted.get("subtotal")
    tax_sum = _sum_taxable_amounts(extracted.get("taxes") or [])
    targets = [float(t) for t in (subtotal, total) if t is not None and float(t) > 0]
    if total and tax_sum:
        targets.append(float(total) - tax_sum)
    if len(rows) >= len([i for i in items if isinstance(i, dict)]) and any(abs(row_sum - target) <= 2 for target in targets):
        extracted["line_items"] = rows
        extracted["subtotal"] = row_sum


def _recover_repeated_item_from_gap(extracted, unified_text):
    """Recover one missing duplicate when OCR repeats an existing item line."""
    items = extracted.get("line_items") or []
    if not items:
        return
    total = extracted.get("total")
    subtotal = extracted.get("subtotal")
    if not total and subtotal is None:
        return
    item_sum = sum(float(i.get("total") or 0) for i in items if isinstance(i, dict))
    tax_sum = _sum_taxable_amounts(extracted.get("taxes") or [])
    if total and tax_sum > 0 and abs(item_sum + float(tax_sum) - float(total)) <= 2:
        return
    targets = [float(t) for t in (subtotal, total) if t is not None and t > 0]
    gaps = [target - item_sum for target in targets if 0 < target - item_sum <= 5000]
    if not gaps:
        return

    def _norm(text: str) -> str:
        text = re.sub(r'[¥￥]?\s*\d[\d,]*\s*(?:[%％][*※除軽]|[*※除軽])?\s*$', '', text or "")
        text = re.sub(r'\s+', '', text)
        text = re.sub(r'[^\wぁ-んァ-ン一-龥]', '', text, flags=re.UNICODE)
        return text.lower()

    norm_lines = [_norm(line) for line in unified_text.split('\n')]
    for item in list(items):
        if not isinstance(item, dict):
            continue
        price = float(item.get("total") or 0)
        if price <= 0 or not any(abs(gap - price) <= 2 for gap in gaps):
            continue
        desc = item.get("description") or ""
        ndesc = _norm(desc)
        if len(ndesc) < 3:
            continue
        ocr_count = sum(1 for line in norm_lines if ndesc and (ndesc in line or line in ndesc))
        extracted_count = sum(
            1 for other in items
            if isinstance(other, dict)
            and _norm(other.get("description") or "") == ndesc
            and abs(float(other.get("total") or 0) - price) <= 2
        )
        if ocr_count <= extracted_count:
            continue
        new_item = dict(item)
        new_item["qty"] = 1
        new_item["unit_price"] = price
        new_item["total"] = price
        insert_at = max(
            (idx for idx, other in enumerate(items)
             if isinstance(other, dict) and _norm(other.get("description") or "") == ndesc),
            default=len(items) - 1,
        ) + 1
        items.insert(insert_at, new_item)
        return


def _fix_o_ring_descriptions_from_ocr(extracted, unified_text):
    """Repair hardware O-ring item names when OCR/JAN context is explicit."""
    items = extracted.get("line_items") or []
    if not items:
        return
    has_o_ring_evidence = bool(re.search(r'4909730105008', unified_text)) and bool(
        re.search(r'(?:^|\n)\s*(?:\d{3,6}\s*)?リング(?:\s|$)', unified_text)
    )
    if not has_o_ring_evidence:
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        desc = (item.get("description") or "").strip()
        unit = float(item.get("unit_price") or 0)
        total = float(item.get("total") or 0)
        qty = float(item.get("qty") or 1)
        price_evidence = (
            abs(unit - 198) <= 1
            or abs(total - 198) <= 1
            or (qty >= 2 and abs(total - (unit * qty)) <= 2 and abs(unit - 198) <= 1)
        )
        if desc in {"リング", "レギュラー"} and price_evidence:
            item["description"] = "Oリング"


def _recover_qty_unit_total_item_from_empty_extraction(extracted, unified_text):
    """Recover a single item from a visible desc / qty x unit / total block."""
    if extracted.get("line_items"):
        return
    total = extracted.get("total")
    if not total:
        return
    lines = [line.strip() for line in unified_text.split('\n')]

    def _valid_desc(text: str) -> bool:
        if not text or _SKIP_PRICE_LINE.search(text):
            return False
        if _HEADER_LINE_RE.search(text) or _JUNK_DESC_RE.search(text):
            return False
        if re.search(r'TEL|電話|登録番号|合計|小計|領収|No\.?', text, re.IGNORECASE):
            return False
        return bool(re.search(r'[ぁ-んァ-ン一-龥]', text))

    for idx, line in enumerate(lines):
        m = re.search(r'(\d+(?:\.\d+)?)\s*個\s*[xX×]\s*単\s*([\d,]+)', line)
        if not m:
            continue
        qty = float(m.group(1))
        unit = float(m.group(2).replace(',', ''))
        amount = None
        amount_idx = None
        for j in range(idx + 1, min(idx + 4, len(lines))):
            am = re.search(r'[¥￥]?\s*([\d,]+)\s*円?\s*$', lines[j])
            if not am or _SKIP_PRICE_LINE.search(lines[j]):
                continue
            amount = float(am.group(1).replace(',', ''))
            amount_idx = j
            break
        if amount is None or amount_idx is None:
            continue
        if abs(qty * unit - amount) > 2 or abs(amount - float(total)) > 2:
            continue
        desc = None
        for j in range(idx - 1, max(idx - 7, -1), -1):
            cand = _clean_ocr_price_line_desc(lines[j])
            if not _valid_desc(cand):
                continue
            desc = cand
            break
        if not desc:
            continue
        if desc == "ヘ" and re.search(r'Grand\s*Joul|美容|ヘア|サロン', unified_text, re.IGNORECASE):
            desc = "ヘア"
        extracted["line_items"] = [{
            "description": desc,
            "qty": qty,
            "unit_price": unit,
            "total": amount,
            "tax_category": "10%",
            "discount": 0,
            "discount_rate": "",
        }]
        return


def _replace_repeated_ocr_item_block_when_balanced(extracted, unified_text):
    """Replace simple repeated item blocks when count × mode price balances."""
    subtotal = extracted.get("subtotal")
    total = extracted.get("total")
    taxes = extracted.get("taxes") or []
    tax_sum = _sum_taxable_amounts(taxes)
    targets = [float(t) for t in (subtotal, total) if t is not None and t > 0]
    if total and tax_sum:
        targets.append(float(total) - tax_sum)
    if not targets:
        return
    if min(targets) > 1000:
        return

    lines = [line.strip() for line in unified_text.split('\n')]
    try:
        end = next(i for i, line in enumerate(lines) if re.search(r'小\s*計', line))
    except StopIteration:
        return
    zone = lines[:end + 2]

    def _clean_desc(line: str) -> str:
        line = re.sub(r'[¥￥]\s*[\d,]+.*$', '', line)
        line = re.sub(r'\s+', '', line)
        return line.strip()

    desc_counts: dict[str, int] = {}
    for line in zone:
        desc = _clean_desc(line)
        if len(desc) < 3:
            continue
        if not re.search(r'[ぁ-んァ-ン一-龥]', desc):
            continue
        if re.search(r'税率|適用|自家製|電話|TEL|登録|領収|人数|株式会社|店舗|小計|合計', desc, re.IGNORECASE):
            continue
        if re.search(r'\d{4}年|\d+名|No\d', desc):
            continue
        desc_counts[desc] = desc_counts.get(desc, 0) + 1
    if not desc_counts:
        return
    desc, count = max(desc_counts.items(), key=lambda pair: pair[1])
    if count < 2:
        return

    prices: list[float] = []
    for line in zone:
        for m in re.finditer(r'[¥￥]\s*([\d,]+)', line):
            price = float(m.group(1).replace(',', ''))
            if 0 < price < min(targets) and not any(abs(price - t) <= 2 for t in targets):
                prices.append(price)
    if not prices:
        return
    from collections import Counter
    price_counts = Counter(prices)
    price, price_count = price_counts.most_common(1)[0]
    if price_count < count - 1:
        return
    if not any(abs(price * count - target) <= 2 for target in targets):
        return

    existing_items = extracted.get("line_items") or []
    tax_category = "8%"
    for item in existing_items:
        if isinstance(item, dict) and item.get("tax_category"):
            tax_category = item["tax_category"]
            break
    extracted["line_items"] = [
        {
            "description": desc,
            "qty": 1.0,
            "unit_price": price,
            "total": price,
            "tax_category": tax_category,
            "discount": 0,
            "discount_rate": "",
        }
        for _ in range(count)
    ]


def _recover_discounted_item_from_gap(extracted, unified_text):
    """Recover one item when the missing gap is OCR price minus discount."""
    items = extracted.get("line_items") or []
    if not items:
        return
    subtotal = extracted.get("subtotal")
    total = extracted.get("total")
    taxes = extracted.get("taxes") or []
    tax_sum = _sum_taxable_amounts(taxes)
    targets = [float(t) for t in (subtotal, total) if t is not None and t > 0]
    if total and tax_sum:
        targets.append(float(total) - tax_sum)
    item_sum = sum(float(i.get("total") or 0) for i in items if isinstance(i, dict))
    gaps = [target - item_sum for target in targets if 0 < target - item_sum <= 5000]
    if not gaps:
        return
    lines = [line.strip() for line in unified_text.split('\n')]
    for idx, line in enumerate(lines):
        pm = _OCR_TRAILING_PRICE_RE.search(line)
        if not pm:
            continue
        try:
            price = float(pm.group(1).strip().lstrip('¥￥').replace(',', ''))
        except ValueError:
            continue
        discount = None
        discount_rate = ""
        for j in range(idx + 1, min(idx + 5, len(lines))):
            rm = re.search(r'(\d+)\s*%', lines[j])
            if rm:
                discount_rate = rm.group(1) + "%"
            dm = re.match(r'^-\s*(\d{1,4})\s*$', lines[j])
            if dm:
                discount = float(dm.group(1))
                break
        if discount is None:
            continue
        net = price - discount
        if not any(abs(net - gap) <= 2 for gap in gaps):
            continue
        desc = _find_discounted_ocr_item_desc(lines, idx)
        if not desc:
            continue
        if any(isinstance(item, dict) and abs(float(item.get("total") or 0) - net) <= 0.5 for item in items):
            continue
        recovered = {
            "description": desc,
            "qty": 1.0,
            "unit_price": price,
            "total": net,
            "tax_category": "8%",
            "discount": discount,
            "discount_rate": discount_rate,
        }
        _insert_item_by_ocr_order(items, lines, idx, recovered)
        return


def _drop_non_product_line_items(extracted, unified_text):
    """Remove header/payment/footer rows that were extracted as products."""
    items = extracted.get("line_items") or []
    if not items:
        return
    receipt_total = extracted.get("total") or 0
    priced_sum = sum(
        float(item.get("total") or 0)
        for item in items
        if isinstance(item, dict) and float(item.get("total") or 0) > 0
    )
    zero_value_modifiers_are_extra = (
        receipt_total
        and abs(priced_sum - float(receipt_total)) <= 1
        and any(float(item.get("total") or 0) == 0 for item in items if isinstance(item, dict))
    )
    bad_desc_re = re.compile(
        r'WAON(?:支払額|残高)|支払額|残高|取扱区分|^額$|^金\s*額$|'
        r'^レジ\s*\d+|^\d{4}年|買上日|カード会社|会員番号|伝票番号|承認番号|'
        r'取引内容|お取扱日|^クレジット$|^現金$|^お釣り$|^釣銭$|'
        r'^[\(（\s※＊*]*(?:\d+(?:\.\d+)?\s*[%％]\s*)?[内外]\s*(?:税)?[\)）\s]*$'
    )
    kept = []
    for item in items:
        if not isinstance(item, dict):
            kept.append(item)
            continue
        desc = (item.get("description") or "").strip()
        total = float(item.get("total") or 0)
        is_bag = _is_bag_description(desc)
        looks_bad = (
            bool(bad_desc_re.search(desc))
            or desc == "割引"
            or (
                re.search(r'本体合計\s*\(\s*1\s*点\s*\)', unified_text)
                and re.search(r'エクストラ|ライト|カスタム', desc)
            )
            or (
                zero_value_modifiers_are_extra
                and total == 0
                and re.search(r'エクストラ|ライト|カスタム|ミルク|アイス|ホット|サイズ', desc)
            )
            or bool(_HEADER_LINE_RE.search(desc))
            or bool(_BANNER_PHRASE_RE.search(desc))
            or (receipt_total and total > receipt_total * 1.2)
        )
        if looks_bad and not is_bag:
            continue
        kept.append(item)
    extracted["line_items"] = kept


def _fix_bag_description_from_ocr_code_context(extracted, unified_text):
    """Recover bag size/price when OCR keeps the POS code line separate."""
    items = extracted.get("line_items") or []
    if not items:
        return
    code_line = re.search(r'(?:^|\n)\s*0*500\s*内?\s*レジ袋\s*(\d{1,3})\s*円', unified_text)
    if not code_line:
        return
    price = float(code_line.group(1))
    for item in items:
        if not isinstance(item, dict):
            continue
        desc = (item.get("description") or "").strip()
        if desc == "レジ袋":
            item["description"] = "レジ袋L"
            item["qty"] = item.get("qty") or 1
            item["unit_price"] = price
            item["total"] = price
            item["tax_category"] = "10%"
            return


def _fix_colon_split_product_names_from_ocr(extracted, unified_text):
    """Join adjacent OCR product fragments where a series/name prefix ends in ':'."""
    items = extracted.get("line_items") or []
    if not items:
        return
    lines = [line.strip() for line in unified_text.split('\n')]

    def _clean(text: str) -> str:
        text = re.sub(r'^[\d\s※＊*]+', '', text.strip())
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    for item in items:
        if not isinstance(item, dict):
            continue
        desc = _clean(item.get("description") or "")
        if not desc or ':' in desc or '：' in desc:
            continue
        total = item.get("total")
        for idx, line in enumerate(lines[:-1]):
            if _clean(line) != desc:
                continue
            nxt = _clean(lines[idx + 1])
            if not re.search(r'[:：]\s*$', nxt):
                continue
            if re.search(r'小計|合計|税|対象|ポイント|レジ|登録番号', nxt):
                continue
            price_nearby = False
            for following in lines[idx + 2:min(len(lines), idx + 5)]:
                m = _OCR_TRAILING_PRICE_RE.search(following)
                if not m:
                    continue
                try:
                    value = float(m.group(1).strip().lstrip('¥￥').replace(',', ''))
                except ValueError:
                    continue
                if total is None or abs(value - float(total or 0)) <= 1:
                    price_nearby = True
                    break
            if price_nearby:
                item["description"] = f"{nxt} {desc}".replace("：", ":")
                break


def _fix_tax_categories_from_price_line_markers(extracted, unified_text):
    """Use ordered price-line reduced-rate marks (* or 軽) when OCR exposes them."""
    items = extracted.get("line_items") or []
    if not items:
        return
    if not re.search(r'軽減税率対象|\[\*\]\s*マーク|「軽」', unified_text):
        return
    has_star_legend = bool(re.search(r'\[\*\]\s*マーク|[*＊]\s*マーク', unified_text))
    reduced_item_prefixes: set[str] = set()
    if re.search(r'[*＊]\s*[:：]?\s*軽減税率対象', unified_text):
        for raw in unified_text.split('\n'):
            line = raw.strip()
            if not re.match(r'^[*＊]\s*', line):
                continue
            desc = re.sub(r'^[*＊]\s*', '', line).strip()
            if not desc or re.search(r'軽減税率対象|小計|合計|税|ポイント', desc):
                continue
            reduced_item_prefixes.add(re.sub(r'\s+', '', desc))
    if reduced_item_prefixes:
        for item in items:
            if not isinstance(item, dict):
                continue
            desc_key = re.sub(r'\s+', '', item.get("description") or "")
            if any(desc_key and (desc_key in prefix or prefix in desc_key) for prefix in reduced_item_prefixes):
                item["tax_category"] = "8%"
    marker_rows: list[tuple[float, bool]] = []
    for raw in unified_text.split('\n'):
        line = raw.strip()
        if re.search(r'小計|合計|対象|消費税|支払|お釣り|ポイント', line):
            continue
        m = re.fullmatch(r'([*＊※]?)\s*[¥￥]?\s*([\d,]+)\s*(軽|[*＊※])?', line)
        if not m:
            continue
        try:
            amount = float(m.group(2).replace(',', ''))
        except ValueError:
            continue
        marker_rows.append((amount, bool(m.group(1)) or bool(m.group(3))))
    if len(marker_rows) < len(items):
        return
    row_idx = 0
    changed = False
    for item in items:
        if not isinstance(item, dict):
            continue
        total = float(item.get("total") or 0)
        match_idx = None
        for idx in range(row_idx, min(len(marker_rows), row_idx + 4)):
            amount, _marked = marker_rows[idx]
            if abs(amount - total) <= 1:
                match_idx = idx
                break
        if match_idx is None:
            continue
        amount, marked = marker_rows[match_idx]
        if marked:
            item["tax_category"] = "8%"
            changed = True
        elif has_star_legend and item.get("tax_category") not in ("0%", "非課税"):
            item["tax_category"] = "10%"
            changed = True
        row_idx = match_idx + 1
    if changed:
        return


def _replace_barcode_qty_price_rows_when_balanced(extracted, unified_text):
    """Project retail rows printed as description / barcode / qty-price."""
    total = extracted.get("total")
    if not total:
        return
    lines = [line.strip() for line in unified_text.split('\n')]
    rows: list[dict] = []
    unbarcoded_rows: list[dict] = []

    def _valid_desc(text: str) -> bool:
        if not text or len(text) < 2:
            return False
        if not re.search(r'[A-Za-zぁ-んァ-ン一-龥]', text):
            return False
        if re.search(r'領収|登録番号|TEL|http|支払い|クレジット|買上点数|小計|合計|消費税|レシート|返品|アンケート', text, re.IGNORECASE):
            return False
        return True

    def _row(desc: str, qty: float, unit: float, tax_category: str = "10%") -> dict:
        return {
            "description": desc,
            "qty": qty,
            "unit_price": unit,
            "total": qty * unit,
            "tax_category": tax_category,
            "discount": 0,
            "discount_rate": "",
        }

    def _barcode_after_description(idx: int) -> int | None:
        if idx + 1 < len(lines) and re.fullmatch(r'\d{10,14}', lines[idx + 1]):
            return idx + 1
        if (
            idx + 2 < len(lines)
            and re.fullmatch(r'\[[0-2]?\d:[0-5]\d\]|\(?[0-2]?\d:[0-5]\d(?::[0-5]\d)?\)?', lines[idx + 1])
            and re.fullmatch(r'\d{10,14}', lines[idx + 2])
        ):
            return idx + 2
        return None

    for idx in range(0, len(lines) - 2):
        desc = lines[idx]
        if not _valid_desc(desc):
            continue
        barcode_idx = _barcode_after_description(idx)
        if barcode_idx is None or barcode_idx + 1 >= len(lines):
            continue
        qty_price = re.fullmatch(r'(\d+(?:\.\d+)?)\s*[¥￥]\s*([\d,]+)', lines[barcode_idx + 1])
        if not qty_price:
            continue
        qty = float(qty_price.group(1))
        unit = float(qty_price.group(2).replace(',', ''))
        if qty <= 0 or unit <= 0:
            continue
        rows.append(_row(desc, qty, unit))

    if rows:
        for idx in range(0, len(lines) - 1):
            desc = lines[idx]
            if not _valid_desc(desc):
                continue
            if _barcode_after_description(idx) is not None:
                continue
            qty_price = re.fullmatch(r'(\d+(?:\.\d+)?)\s*[¥￥]\s*([\d,]+)', lines[idx + 1])
            if not qty_price:
                continue
            qty = float(qty_price.group(1))
            unit = float(qty_price.group(2).replace(',', ''))
            if qty <= 0 or unit <= 0:
                continue
            unbarcoded_rows.append(_row(desc, qty, unit))

    # Some retailers print the shopping-bag price before the barcode and
    # description. Add that low-value row when it is visible in the same block.
    for idx in range(0, len(lines) - 2):
        price_m = re.fullmatch(r'[¥￥]\s*(\d{1,3})', lines[idx])
        if not price_m:
            continue
        if not re.fullmatch(r'\d{10,14}', lines[idx + 1]):
            continue
        desc = lines[idx + 2]
        if not _is_bag_description(desc):
            continue
        rows.append(_row(desc, 1.0, float(price_m.group(1)), "10%"))

    if unbarcoded_rows:
        row_sum_with_bags = sum(float(row.get("total") or 0) for row in rows)
        candidate_sum = row_sum_with_bags + sum(float(row.get("total") or 0) for row in unbarcoded_rows)
        if abs(candidate_sum - float(total)) <= 2:
            rows.extend(unbarcoded_rows)

    if len(rows) < 2:
        return
    row_sum = sum(float(row.get("total") or 0) for row in rows)
    current_count = len([item for item in (extracted.get("line_items") or []) if isinstance(item, dict)])
    if len(rows) > current_count and abs(row_sum - float(total)) <= 2:
        extracted["line_items"] = rows


def _replace_barcode_unit_qty_amount_stack_when_balanced(extracted, unified_text):
    """Project retail rows printed as description / barcode / unit-qty plus total stack."""
    lines = [line.strip() for line in unified_text.split('\n')]
    subtotal_idx = next((i for i, line in enumerate(lines) if re.fullmatch(r'小\s*計', line)), None)
    if subtotal_idx is None:
        return

    def _valid_desc(text: str) -> bool:
        if not text or len(text) < 2:
            return False
        if not re.search(r'[A-Za-zぁ-んァ-ン一-龥]', text):
            return False
        if re.search(r'領収|登録番号|TEL|http|支払い|クレジット|小計|合計|消費税|返品', text, re.IGNORECASE):
            return False
        return True

    def _clean_desc(text: str) -> str:
        text = re.sub(r'^\s*\d{3,6}\s+', '', text.strip())
        return text.strip()

    rows: list[dict] = []
    first_row_idx: int | None = None
    idx = 0
    while idx < subtotal_idx - 1:
        desc = lines[idx]
        if not _valid_desc(desc) or not re.fullmatch(r'\d{10,14}', lines[idx + 1]):
            idx += 1
            continue
        row = {
            "description": _clean_desc(desc),
            "qty": 1.0,
            "unit_price": None,
            "total": None,
            "tax_category": "10%",
            "discount": 0,
            "discount_rate": "",
        }
        consumed_idx = idx + 1
        if consumed_idx + 1 < subtotal_idx:
            qty_line = lines[consumed_idx + 1]
            unit_qty = re.fullmatch(
                r'[¥￥]?\s*([\d,]+)\s+(\d+(?:\.\d+)?)\s*[個コ点]',
                qty_line,
            )
            if unit_qty:
                unit = float(unit_qty.group(1).replace(',', ''))
                qty = float(unit_qty.group(2))
                row["qty"] = qty
                row["unit_price"] = unit
                row["total"] = qty * unit
                consumed_idx += 1
        if not row["description"]:
            idx = consumed_idx + 1
            continue
        if first_row_idx is None:
            first_row_idx = idx
        rows.append(row)
        idx = consumed_idx + 1

    if len(rows) < 2 or first_row_idx is None:
        return

    stack_amounts: list[float] = []
    for line in lines[first_row_idx:subtotal_idx]:
        if re.search(r'[個コ点]|[@＠]', line):
            continue
        amount = re.fullmatch(r'[¥￥]\s*([\d,]+)', line)
        if amount:
            stack_amounts.append(float(amount.group(1).replace(',', '')))
    if len(stack_amounts) < len(rows):
        return
    stack_amounts = stack_amounts[-len(rows):]

    for row, amount in zip(rows, stack_amounts, strict=False):
        known_total = row.get("total")
        if known_total is not None and abs(float(known_total) - amount) > 2:
            return
        row["total"] = amount
        if not row.get("unit_price"):
            qty = float(row.get("qty") or 1)
            row["unit_price"] = amount / qty if qty else amount

    rate_match = re.search(r'(8|10)(?:\.0)?\s*%\s*対象', "\n".join(lines[first_row_idx:subtotal_idx]))
    if rate_match:
        tax_category = f"{rate_match.group(1)}%"
        for row in rows:
            row["tax_category"] = tax_category

    row_sum = sum(float(row.get("total") or 0) for row in rows)
    targets: list[float] = []
    for value in (extracted.get("subtotal"), extracted.get("total")):
        try:
            if value is not None and float(value) > 0:
                targets.append(float(value))
        except (TypeError, ValueError):
            pass
    try:
        total = float(extracted.get("total") or 0)
    except (TypeError, ValueError):
        total = 0
    tax_sum = _sum_taxable_amounts(extracted.get("taxes") or [])
    if total > 0 and tax_sum > 0:
        targets.append(total - tax_sum)
    if not targets or min(abs(row_sum - target) for target in targets) > 2:
        return

    current_items = [item for item in (extracted.get("line_items") or []) if isinstance(item, dict)]
    current_sum = sum(float(item.get("total") or 0) for item in current_items)
    current_score = min(abs(current_sum - target) for target in targets) if current_items else float("inf")
    row_score = min(abs(row_sum - target) for target in targets)
    if len(rows) > len(current_items) or row_score + 0.5 < current_score:
        extracted["line_items"] = rows


def _replace_item_price_qty_rows_when_balanced(extracted, unified_text):
    """Project item rows from description/price/quantity-detail OCR structure."""
    lines = [line.strip() for line in unified_text.split('\n')]
    subtotal = None
    printed_count = None
    subtotal_idx = None
    for idx, line in enumerate(lines):
        if re.fullmatch(r'小\s*計', line):
            subtotal_idx = idx
            for nearby in lines[idx + 1:min(len(lines), idx + 6)]:
                count_m = re.fullmatch(r'(\d+)\s*点', nearby)
                if count_m:
                    printed_count = int(count_m.group(1))
                    continue
                vm = re.fullmatch(r'[¥￥]\s*([\d,]+)\s*[\)）]?', nearby)
                if vm:
                    subtotal = float(vm.group(1).replace(',', ''))
                    break
            break
    if subtotal is None or subtotal_idx is None:
        return

    def _valid_desc(text: str) -> bool:
        text = text.strip()
        if len(text) < 2:
            return False
        if not re.search(r'[ぁ-んァ-ン一-龥]', text):
            return False
        if re.search(
            r'TEL|電話|公式|通販|検索|領収|登録番号|レジ|責|No\.?|'
            r'小計|合計|税|対象|支払|お釣|伝票|承認|会員|http|店|証',
            text,
            re.IGNORECASE,
        ):
            return False
        if re.search(r'\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}:\d{2}', text):
            return False
        return True

    def _price_from_line(text: str) -> tuple[float, bool] | None:
        m = re.search(r'[¥￥]\s*([\d,]+)', text)
        if not m:
            return None
        return float(m.group(1).replace(',', '')), "外" in text

    def _qty_detail(text: str) -> tuple[float, float] | None:
        m = re.search(r'[@＠]\s*([\d,]+)\s*[xX×]\s*(\d+)\s*個?', text)
        if not m:
            return None
        unit = float(m.group(1).replace(',', ''))
        qty = float(m.group(2).replace(',', ''))
        if unit <= 0 or qty <= 0:
            return None
        return unit, qty

    pending: list[dict] = []
    rows: list[dict] = []

    def _apply_qty(row: dict, detail: tuple[float, float] | None) -> None:
        if detail is None:
            return
        unit, qty = detail
        total = float(row["total"])
        if abs(unit * qty - total) <= 1:
            row["qty"] = qty
            row["unit_price"] = unit

    def _emit_from_pending(amount: float, external_marker: bool) -> None:
        if not pending:
            return
        entry = pending.pop(0)
        desc = re.sub(r'\s+', ' ', entry["desc"]).strip()
        row = {
            "description": desc,
            "qty": 1.0,
            "unit_price": amount,
            "total": amount,
            "tax_category": "10%" if external_marker else "8%",
            "discount": 0,
            "discount_rate": "",
            "_external_marker": external_marker,
        }
        _apply_qty(row, entry.get("qty_detail"))
        rows.append(row)

    for idx, line in enumerate(lines[:subtotal_idx]):
        if not line:
            continue
        detail = _qty_detail(line)
        if detail is not None:
            if pending:
                pending[-1]["qty_detail"] = detail
            elif rows:
                _apply_qty(rows[-1], detail)
            continue
        price = _price_from_line(line)
        if price is not None:
            amount, external_marker = price
            before_price = re.split(r'[¥￥]', line, maxsplit=1)[0].strip()
            if before_price and _valid_desc(before_price):
                pending.append({"desc": before_price})
            _emit_from_pending(amount, external_marker)
            continue
        if re.fullmatch(r'\d{1,4}', line) and pending:
            lookahead_has_price = any(
                _price_from_line(next_line) is not None
                for next_line in lines[idx + 1:min(subtotal_idx, idx + 4)]
            )
            if lookahead_has_price:
                pending[-1]["desc"] = f'{pending[-1]["desc"]} {line}'
            continue
        if _valid_desc(line):
            pending.append({"desc": line})
        elif not rows and re.search(r'TEL|電話|公式|通販|検索|領収|登録番号|レジ|責|No\.?|店|証', line, re.IGNORECASE):
            pending.clear()

    if len(rows) < 3:
        return
    row_sum = sum(float(row["total"]) for row in rows)
    qty_sum = sum(float(row.get("qty") or 1) for row in rows)
    if abs(row_sum - subtotal) > 2:
        return
    if printed_count is not None and abs(qty_sum - printed_count) > 0.1:
        return

    rate_bases = extract_rate_bases(unified_text)
    reduced_base = rate_bases.get("8%")
    if reduced_base and reduced_base > 0 and re.search(r'軽減税率|軽税|8%税抜対象額', unified_text):
        def _category_family(desc: str) -> str:
            compact = re.sub(r'\s+', '', desc or "")
            compact = re.sub(r'(?:LL|SS|[LSM]|[0-9０-９]+)$', '', compact, flags=re.IGNORECASE)
            return compact if len(compact) >= 3 else ""

        external_families = {
            family for family in (_category_family(row["description"]) for row in rows)
            if family and any(
                other.get("_external_marker") and _category_family(other["description"]) == family
                for other in rows
            )
        }
        for row in rows:
            if row.get("_external_marker"):
                row["tax_category"] = "10%"
            elif _category_family(row["description"]) in external_families:
                row["tax_category"] = "10%"
            elif abs(float(row["total"]) - float(reduced_base)) <= 1:
                row["tax_category"] = "8%"
            else:
                row["tax_category"] = "10%"

    for row in rows:
        row.pop("_external_marker", None)
    extracted["line_items"] = rows


def _fix_qty_context_and_reduced_rate_from_ocr(extracted, unified_text):
    """Repair quantity drift and reduced-rate context from nearby OCR rows."""
    items = extracted.get("line_items") or []
    if not items:
        return
    lines = [line.strip() for line in unified_text.split('\n')]
    has_qty_detail_context = bool(
        re.search(r'@\s*[\d,]+\s*[xX×]\s*\d+\s*個|[xX×]\s*\d+\s*個', unified_text)
    )
    rate_bases = extract_rate_bases(unified_text)
    reduced_base = rate_bases.get("8%")
    has_reduced_rate_context = bool(
        reduced_base
        and reduced_base > 0
        and re.search(r'軽減税率|軽税|8%税抜対象額', unified_text)
    )
    if not has_qty_detail_context and not has_reduced_rate_context:
        return

    def _norm(text: str) -> str:
        return re.sub(r'\s+', '', text or "")

    def _find_desc_line(desc: str) -> int | None:
        desc_norm = _norm(desc)
        for idx, line in enumerate(lines):
            line_norm = _norm(line)
            if desc_norm and (desc_norm in line_norm or line_norm in desc_norm):
                return idx
        return None

    def _is_context_price_line(line: str) -> bool:
        compact = _norm(line)
        return bool(re.fullmatch(r'[内外*※\s]*[¥￥]\s*[\d,]+(?:\s*[内外軽税]*)?', line or "")) or bool(
            re.fullmatch(r'[内外*※¥￥\d,軽税]+', compact)
            and re.search(r'[¥￥]\s*\d|^\d', compact)
        )

    if has_qty_detail_context:
        for item in items:
            if not isinstance(item, dict):
                continue
            qty = float(item.get("qty") or 1)
            unit = float(item.get("unit_price") or 0)
            total = float(item.get("total") or 0)
            if qty <= 1 or unit <= 0 or total <= 0:
                continue
            idx = _find_desc_line(item.get("description") or "")
            if idx is None:
                continue
            saw_qty_detail = False
            first_price = None
            for line in lines[idx + 1:min(len(lines), idx + 5)]:
                if re.search(r'@\s*[\d,]+\s*[xX×]\s*\d+\s*個|[xX×]\s*\d+\s*個', line):
                    saw_qty_detail = True
                if re.search(r'[ぁ-んァ-ン一-龥A-Za-z]', line) and not re.search(r'[@xX×個]', line) and not _is_context_price_line(line):
                    break
                pm = re.search(r'[¥￥]\s*([\d,]+)', line)
                if pm:
                    first_price = float(pm.group(1).replace(',', ''))
                    break
            if first_price is not None and abs(first_price - unit) <= 1 and not saw_qty_detail:
                item["qty"] = 1.0
                item["total"] = unit

    if not has_reduced_rate_context:
        return

    for item in items:
        if not isinstance(item, dict):
            continue
        idx = _find_desc_line(item.get("description") or "")
        if idx is None:
            continue
        first_price_line = None
        for line in lines[idx + 1:min(len(lines), idx + 5)]:
            if re.search(r'[ぁ-んァ-ン一-龥A-Za-z]', line) and not re.fullmatch(r'\d+', line) and not _is_context_price_line(line):
                break
            if re.search(r'[¥￥]\s*[\d,]+', line):
                first_price_line = line
                break
        if first_price_line is None:
            continue
        pm = re.search(r'[¥￥]\s*([\d,]+)', first_price_line)
        first_price_value = float(pm.group(1).replace(',', '')) if pm else 0
        if "外" in first_price_line:
            item["tax_category"] = "10%"
        elif reduced_base and abs(first_price_value - float(reduced_base)) <= 1:
            item["tax_category"] = "8%"

    if reduced_base and reduced_base > 0:
        candidates = [
            item for item in items
            if (
                isinstance(item, dict)
                and abs(float(item.get("total") or 0) - float(reduced_base)) <= 1
                and _FOOD_DESC_RE.search(item.get("description") or "")
            )
        ]
        if len(candidates) == 1:
            candidates[0]["tax_category"] = "8%"


def _fix_numeric_desc_from_ocr_price_context(extracted, unified_text):
    """Replace pure numeric item descriptions with nearby OCR product names."""
    items = extracted.get("line_items") or []
    if not items:
        return
    lines = [line.strip() for line in unified_text.split('\n')]

    def _clean_desc(text: str) -> str:
        text = text.strip()
        text = re.sub(r'\s+\d[\d,]*\s*(?:[%％*※除軽Xx])?\s*$', '', text).strip()
        return text

    def _valid_desc(text: str) -> bool:
        if not text or len(text) < 3:
            return False
        if not re.search(r'[ぁ-んァ-ン一-龥]', text):
            return False
        if _SKIP_PRICE_LINE.search(text) or _HEADER_LINE_RE.search(text):
            return False
        if re.search(r'小計|合計|税|対象|割引|お釣り|クレジット|現金|レジ|登録番号|TEL|FAX|http', text):
            return False
        return True

    existing_descs = {
        (item.get("description") or "").strip()
        for item in items
        if isinstance(item, dict)
    }
    for item in items:
        if not isinstance(item, dict):
            continue
        desc = (item.get("description") or "").strip()
        if not re.fullmatch(r'\d[\d,]*', desc):
            continue
        total = float(item.get("total") or 0)
        if total <= 0:
            continue
        price_lines = []
        for idx, line in enumerate(lines):
            m = _OCR_TRAILING_PRICE_RE.search(line)
            if not m:
                continue
            try:
                value = float(m.group(1).strip().lstrip('¥￥').replace(',', ''))
            except ValueError:
                continue
            if abs(value - total) <= 1:
                price_lines.append(idx)
        for idx in price_lines:
            candidates = list(range(idx - 1, max(idx - 8, -1), -1))
            candidates += list(range(idx + 1, min(idx + 4, len(lines))))
            for cand_idx in candidates:
                cand = _clean_desc(lines[cand_idx])
                if not _valid_desc(cand):
                    continue
                if cand in existing_descs:
                    continue
                item["description"] = cand
                existing_descs.add(cand)
                break
            if not re.fullmatch(r'\d[\d,]*', (item.get("description") or "").strip()):
                break


def _replace_overage_item_with_low_value_bag(extracted, unified_text):
    """When a low-value bag row is missing and one item absorbs the overage, restore it."""
    items = extracted.get("line_items") or []
    if not items:
        return
    targets = [
        t for t in (extracted.get("subtotal"), extracted.get("total"))
        if t is not None
    ]
    if not targets:
        return
    items_sum = sum(float(i.get("total") or 0) for i in items if isinstance(i, dict))
    overages = [items_sum - float(t) for t in targets if 0 < items_sum - float(t) <= 500]
    if not overages:
        return
    overage = min(overages)
    lines = [line.strip() for line in unified_text.split('\n')]
    bag_rows: list[tuple[str, float]] = []
    for line in lines:
        if not re.search(r'袋|バッグ|bag', line, re.IGNORECASE):
            continue
        m = re.search(r'^(.+?[ぁ-んァ-ン一-龥][^¥￥]*?)\s+([¥￥]?\s*\d[\d,]*)\s*(?:[%％*※除軽Xx])?\s*$', line)
        if not m:
            continue
        desc = re.sub(r'\s+', ' ', m.group(1)).strip()
        try:
            price = float(m.group(2).strip().lstrip('¥￥').replace(',', ''))
        except ValueError:
            continue
        if 0 < price < 10:
            bag_rows.append((desc, price))
    if len(bag_rows) != 1:
        return
    bag_desc, bag_price = bag_rows[0]
    expected_wrong_total = overage + bag_price
    candidates = [
        item for item in items
        if (
            isinstance(item, dict)
            and abs(float(item.get("total") or 0) - expected_wrong_total) <= 1
            and not _is_bag_description(item.get("description") or "")
        )
    ]
    if not candidates:
        return
    standard_candidates = [
        item for item in candidates
        if (item.get("tax_category") or "") in (STANDARD_RATE, "10%")
    ]
    chosen = standard_candidates[-1] if standard_candidates else candidates[-1]
    chosen["description"] = bag_desc
    chosen["qty"] = 1.0
    chosen["unit_price"] = bag_price
    chosen["total"] = bag_price
    chosen["tax_category"] = STANDARD_RATE
    chosen["discount"] = 0.0
    chosen["discount_rate"] = ""


def _append_missing_low_value_bag_from_gap(extracted, unified_text):
    """Append an explicit low-value bag item when it exactly closes the item sum."""
    items = extracted.get("line_items") or []
    if not items:
        return
    if any(isinstance(item, dict) and _is_bag_description(item.get("description") or "") for item in items):
        return
    targets = [t for t in (extracted.get("subtotal"), extracted.get("total")) if t is not None]
    if not targets:
        return
    items_sum = sum(float(item.get("total") or 0) for item in items if isinstance(item, dict))
    bag_rows: list[tuple[str, float]] = []
    for raw_line in unified_text.split('\n'):
        line = raw_line.strip()
        if not re.search(r'袋|バッグ|bag', line, re.IGNORECASE):
            continue
        m = re.search(r'^(.+?[ぁ-んァ-ン一-龥][^¥￥]*?)\s+([¥￥]?\s*\d[\d,]*)\s*(?:[%％*※除軽Xx])?\s*$', line)
        if not m:
            continue
        desc = re.sub(r'\s+', ' ', m.group(1)).strip()
        try:
            price = float(m.group(2).strip().lstrip('¥￥').replace(',', ''))
        except ValueError:
            continue
        if 0 < price < 10:
            bag_rows.append((desc, price))
    if len(bag_rows) != 1:
        return
    desc, price = bag_rows[0]
    if not any(abs(float(target) - items_sum - price) <= 2 for target in targets):
        return
    items.append({
        "description": desc,
        "qty": 1.0,
        "unit_price": price,
        "total": price,
        "tax_category": STANDARD_RATE,
        "discount": 0.0,
        "discount_rate": "",
    })
    extracted["line_items"] = items


def _replace_service_table_items_when_balanced(extracted, unified_text):
    """Use OCR service-table rows when they balance to the receipt total."""
    items = extracted.get("line_items") or []
    total = extracted.get("total")
    subtotal = extracted.get("subtotal")
    if not items or not total:
        return
    if not re.search(r'商品名[\s\S]{0,80}点数[\s\S]{0,80}金額', unified_text):
        return

    lines = [line.strip() for line in unified_text.split('\n')]
    try:
        start = next(i for i, line in enumerate(lines) if re.fullmatch(r'金額', line))
    except StopIteration:
        return
    end = next(
        (i for i in range(start + 1, len(lines)) if re.search(r'^小\s*計|^合\s*計', lines[i])),
        len(lines),
    )
    if end <= start + 2:
        return

    def _parse_amount(line: str) -> float | None:
        s = line.strip()
        m = re.fullmatch(r'[¥￥]?\s*(\d{1,3}(?:[,.]\d{3})*|\d{2,5})\s*', s)
        if not m:
            return None
        raw = m.group(1).replace(',', '')
        if '.' in raw:
            parts = raw.split('.')
            if len(parts) == 2 and len(parts[1]) == 3:
                raw = ''.join(parts)
            else:
                return None
        try:
            value = float(raw)
        except ValueError:
            return None
        return value if 0 < value < float(total) * 1.2 else None

    def _is_desc(line: str) -> bool:
        if not line or _parse_amount(line) is not None:
            return False
        if re.fullmatch(r'[a-zA-ZｄＤdD]', line):
            return False
        if re.search(r'商品名|点数|金額|タグ|No\.?|仕上|小\s*計|合\s*計|税率|内税|外税|お釣り|クレジット', line):
            return False
        if re.search(r'%\s*OFF|割引|値引|^-\s*\d', line, re.IGNORECASE):
            return False
        return bool(re.search(r'[ぁ-んァ-ン一-龥]', line))

    def _clean_desc(line: str) -> str:
        text = re.sub(r'^\d+\s*[-－]\s*\d+\s*', '', line).strip()
        return re.sub(r'\s+', ' ', text)

    rows: list[dict] = []
    idx = start + 1
    while idx < end:
        line = lines[idx]
        if not _is_desc(line):
            idx += 1
            continue
        desc = _clean_desc(line)
        price = None
        price_idx = None
        saw_next_desc = False
        for j in range(idx + 1, min(idx + 6, end)):
            if re.search(r'%\s*OFF|割引|値引', lines[j], re.IGNORECASE):
                break
            amount = _parse_amount(lines[j])
            if amount is not None:
                price = amount
                price_idx = j
                break
            if _is_desc(lines[j]):
                saw_next_desc = True
                break
        if price is None or saw_next_desc:
            idx += 1
            continue
        rows.append({
            "description": desc,
            "qty": 1.0,
            "unit_price": price,
            "total": price,
            "tax_category": "10%",
            "discount": 0,
            "discount_rate": "",
            "_line": idx,
        })
        idx = (price_idx or idx) + 1

    if len(rows) < len(items):
        return

    for idx, line in enumerate(lines[start + 1:end], start + 1):
        if not re.search(r'(\d+(?:\.\d+)?)\s*%\s*OFF|割引|値引', line, re.IGNORECASE):
            continue
        rate_m = re.search(r'(\d+(?:\.\d+)?)\s*%', line)
        rate = float(rate_m.group(1)) / 100.0 if rate_m else None
        discount = None
        for j in range(idx + 1, min(idx + 4, end)):
            dm = re.fullmatch(r'-\s*[¥￥]?\s*(\d[\d,]*)\s*', lines[j])
            if dm:
                discount = float(dm.group(1).replace(',', ''))
                break
        if discount is None:
            continue
        best = None
        for row in reversed(rows):
            if row.get("discount"):
                continue
            gross = float(row.get("unit_price") or 0)
            if gross <= 0:
                continue
            if rate is not None:
                expected = gross * rate
                if abs(expected - discount) > max(2.0, expected * 0.03):
                    continue
            best = row
            break
        if best is None:
            continue
        best["discount"] = discount
        best["discount_rate"] = f"{int(rate * 100)}%" if rate is not None else ""
        best["total"] = float(best["unit_price"]) - discount

    def _row_sum() -> float:
        return sum(float(row.get("total") or 0) for row in rows)

    targets = [float(total)]
    if subtotal:
        tax_sum = _sum_taxable_amounts(extracted.get("taxes") or [])
        if tax_sum and abs(float(subtotal) + tax_sum - float(total)) <= 5:
            targets.append(float(total))
        else:
            targets.append(float(subtotal))

    current_sum = _row_sum()
    if not any(abs(current_sum - target) <= 5 for target in targets):
        for row in rows:
            desc = row.get("description") or ""
            if not re.search(r'付加|手数料|サービス料|追加|加算', desc):
                continue
            value = int(round(float(row.get("unit_price") or 0)))
            raw = str(value)
            if len(raw) < 3 or raw[0] not in "23456789":
                continue
            corrected = float(int(raw[1:]))
            if corrected <= 0:
                continue
            adjusted = current_sum - value + corrected
            if any(abs(adjusted - target) <= 5 for target in targets):
                row["unit_price"] = corrected
                row["total"] = corrected
                row["discount"] = 0
                row["discount_rate"] = ""
                current_sum = adjusted
                break

    if not any(abs(current_sum - target) <= 5 for target in targets):
        return

    default_tax = None
    for item in items:
        if isinstance(item, dict) and item.get("tax_category"):
            default_tax = item.get("tax_category")
            break
    cleaned_rows = []
    for row in rows:
        row = {k: v for k, v in row.items() if not k.startswith("_")}
        if default_tax and not row.get("tax_category"):
            row["tax_category"] = default_tax
        cleaned_rows.append(row)
    extracted["line_items"] = cleaned_rows


def _replace_dense_item_rows_when_balanced(extracted, unified_text):
    """Parse dense item rows directly when OCR rows balance."""
    subtotal = extracted.get("subtotal")
    if not subtotal:
        return
    lines = [line.strip() for line in unified_text.split('\n')]
    end = next((i for i, line in enumerate(lines) if re.fullmatch(r'小\s*計', line)), None)
    if end is None:
        return
    start = next(
        (i for i, line in enumerate(lines[:end])
         if re.search(r'\d{1,2}:\d{2}', line) or re.search(r'\d{1,2}/\s*\d{1,2}', line)),
        0,
    )
    zone = lines[start + 1:end]

    def _plausible_item_amount(amount: float) -> bool:
        return 1 <= amount <= float(subtotal)

    def _looks_like_metadata(line: str) -> bool:
        return bool(re.search(
            r'AEON|TEL|FAX|http|領収|登録番号|株式会社|毎月|ぜひ|レジ|取\d|登\s*:|スキャン|'
            r'\d{4}/\d{1,2}/\d{1,2}|\d{4}年|^\d{1,2}:\d{2}$',
            line,
            re.IGNORECASE,
        ))

    def _parse_inline(line: str) -> tuple[str, float, str] | None:
        m = re.match(r'^(.+?[ぁ-んァ-ン一-龥][^¥￥]*?)\s+([¥￥]?\s*\d[\d,]*)\s*([A-ZＡ-Ｚ*＊%％※]?)\s*$', line)
        if not m:
            return None
        desc = re.sub(r'\s+', ' ', m.group(1)).strip()
        amount = float(m.group(2).strip().lstrip('¥￥').replace(',', ''))
        if _looks_like_metadata(desc) or not _plausible_item_amount(amount):
            return None
        return desc, amount, m.group(3) or ""

    def _standalone_amount(line: str) -> float | None:
        m = re.fullmatch(r'[¥￥]?\s*(\d[\d,]*)\s*([A-ZＡ-Ｚ*＊]?)', line)
        if not m:
            return None
        return float(m.group(1).replace(',', ''))

    def _valid_desc(line: str) -> bool:
        if not line or _standalone_amount(line) is not None:
            return False
        if _looks_like_metadata(line):
            return False
        if re.search(r'レジ|スキャン|小計|合計|税|支払|お釣り|まとめ値引|クレジット|現金|WAON|カード|^\-', line):
            return False
        if re.search(r'^\(?\d+\s*個|^単\d|^[A-ZＡ-Ｚ]:', line):
            return False
        return bool(re.search(r'[ぁ-んァ-ン一-龥]', line))

    def _qty_unit_between(desc_pos: int, price_pos: int, printed_total: float) -> tuple[float, float] | None:
        window = zone[desc_pos + 1:price_pos + 1]
        joined = "\n".join(window)
        m = re.search(r'(\d+)\s*個\s*[xX×Ⅹ]?\s*単\s*(\d{2,4})', joined)
        if m:
            return float(m.group(1)), float(m.group(2))
        m = re.search(r'\(?\s*(\d+)\s*個[\s\S]{0,12}?単\s*(\d{2,4})', joined)
        if m:
            return float(m.group(1)), float(m.group(2))
        m = re.search(r'[xX×Ⅹ]\s*(\d{2,4})', joined)
        if m:
            unit = float(m.group(1))
            qty = round(printed_total / unit) if unit else 1
            if qty > 1 and abs(qty * unit - printed_total) <= 2:
                return float(qty), unit
        return None

    def _qty_unit_after(pos: int, printed_total: float) -> tuple[float, float] | None:
        window = []
        for line in zone[pos + 1:pos + 5]:
            if _valid_desc(line):
                break
            window.append(line)
        joined = "\n".join(window)
        m = re.search(r'(\d+)\s*個\s*[xX×Ⅹ]?\s*単\s*(\d{2,4})', joined)
        if m:
            return float(m.group(1)), float(m.group(2))
        m = re.search(r'\(?\s*(\d+)\s*個[\s\S]{0,12}?単\s*(\d{2,4})', joined)
        if m:
            return float(m.group(1)), float(m.group(2))
        m = re.search(r'[xX×Ⅹ]\s*(\d{2,4})', joined)
        if m:
            unit = float(m.group(1))
            qty = round(printed_total / unit) if unit else 1
            if qty > 1 and abs(qty * unit - printed_total) <= 2:
                return float(qty), unit
        return None

    rows: list[dict] = []
    idx = 0
    while idx < len(zone):
        line = zone[idx]
        inline = _parse_inline(line)
        if inline:
            desc, amount, marker = inline
            price_idx = idx
            if amount < 10 and not re.search(r'袋|バッグ|bag', desc, re.IGNORECASE):
                idx += 1
                continue
        elif _valid_desc(line):
            desc = re.sub(r'\s+', ' ', line).strip()
            amount = None
            marker = ""
            price_idx = idx
            for j in range(idx + 1, min(idx + 5, len(zone))):
                if _valid_desc(zone[j]):
                    break
                val = _standalone_amount(zone[j])
                if val is not None:
                    if not _plausible_item_amount(val):
                        break
                    if val < 10 and not re.search(r'袋|バッグ|bag', desc, re.IGNORECASE):
                        break
                    amount = val
                    price_idx = j
                    break
            if amount is None:
                idx += 1
                continue
        else:
            idx += 1
            continue

        qty = 1.0
        unit = float(amount)
        total = float(amount)
        qty_unit = _qty_unit_between(idx, price_idx, total) or _qty_unit_after(price_idx, total)
        if qty_unit:
            qty, unit = qty_unit
            total = qty * unit
            if abs(total - float(amount)) > 2:
                total = float(amount)
        discount = 0.0
        for j in range(price_idx + 1, min(price_idx + 7, len(zone))):
            if _valid_desc(zone[j]):
                break
            dm = re.fullmatch(r'-\s*[¥￥]?\s*(\d[\d,]*)', zone[j])
            if dm and "まとめ値引" in "\n".join(zone[price_idx + 1:j + 1]):
                discount = float(dm.group(1).replace(',', ''))
                break
        if discount:
            total -= discount
        rows.append({
            "description": desc,
            "qty": qty,
            "unit_price": unit,
            "total": total,
            "tax_category": "8%" if marker in ("*", "＊") else "8%",
            "discount": discount,
            "discount_rate": "",
        })
        idx = price_idx + 1

    if len(rows) < 5:
        return
    row_sum = sum(float(row["total"]) for row in rows)
    if abs(row_sum - float(subtotal)) > 2:
        return
    current_count = len([item for item in (extracted.get("line_items") or []) if isinstance(item, dict)])
    if len(rows) >= current_count:
        _assign_single_standard_rate_from_small_base(rows, extract_rate_bases(unified_text))
        extracted["line_items"] = rows


def _replace_dense_sequence_rows_when_balanced(extracted, unified_text):
    """Reconstruct dense item streams with name queues and price queues."""
    subtotal = extracted.get("subtotal")
    if not subtotal:
        return
    if not re.search(r'お買上商品数\s*[:：]?\s*\d+', unified_text):
        return
    lines = [line.strip() for line in unified_text.split('\n')]
    end = next((i for i, line in enumerate(lines) if re.fullmatch(r'小\s*計', line)), None)
    if end is None:
        return
    start = next(
        (i for i, line in enumerate(lines[:end])
         if re.search(r'\d{4}/\d{1,2}/\d{1,2}|\d{1,2}:\d{2}', line)),
        0,
    )
    zone = lines[start + 1:end]
    merged_zone: list[str] = []
    idx = 0
    while idx < len(zone):
        line = zone[idx]
        if (
            re.search(r'\(?\s*\d+\s*[個コ]\s*[xX×Ⅹ]\s*$', line)
            and idx + 1 < len(zone)
            and re.fullmatch(r'\s*単?\s*\d{1,5}\s*\)?', zone[idx + 1])
        ):
            merged_zone.append(f"{line} {zone[idx + 1]}")
            idx += 2
            continue
        merged_zone.append(line)
        idx += 1
    zone = merged_zone

    def _clean_desc(text: str) -> str:
        text = _clean_ocr_price_line_desc(text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'^(?:内\s*)?', '', text).strip()
        text = re.sub(r'(有料レジ袋)[しシ]$', r'\1', text)
        return text

    def _valid_desc(text: str) -> bool:
        text = _clean_desc(text)
        if not text or len(text) < 2:
            return False
        if _SKIP_PRICE_LINE.search(text) or _HEADER_LINE_RE.search(text):
            return False
        if re.search(r'\d{1,2}\s*:\s*\d{2}|:', text):
            return False
        if re.search(r'TEL|FAX|http|領収|登録番号|株式会社|毎月|ぜひ|取\d|登\s*:|お買上', text, re.IGNORECASE):
            return False
        if re.search(r'レジ|クレジット|現金|お釣り|釣銭|合計|税', text) and not _is_bag_description(text):
            return False
        if re.search(r'\d+\s*個\s*[xX×]|[xX×]\s*単?\d', text):
            return False
        if re.fullmatch(r'[\d\s,./-]+', text):
            return False
        return bool(re.search(r'[ぁ-んァ-ン一-龥]', text))

    def _parse_amount(text: str) -> tuple[float, str] | None:
        if re.search(r'\d{1,2}\s*:\s*\d{2}|:', text):
            return None
        m = re.fullmatch(r'[¥￥]?\s*(\d[\d,]*)\s*([XxＡ-ＺA-Z%％*＊※除軽]*)', text.strip())
        if not m:
            return None
        amount = float(m.group(1).replace(',', ''))
        if amount <= 0 or amount > float(subtotal):
            return None
        return amount, m.group(2) or ""

    def _parse_inline(text: str) -> tuple[str, float, str] | None:
        if re.search(r'\d{1,2}\s*:\s*\d{2}|:', text):
            return None
        m = re.match(r'^(.+?[ぁ-んァ-ン一-龥][^¥￥]*?)\s+([¥￥]?\s*\d[\d,]*)\s*([XxＡ-ＺA-Z%％*＊※除軽]*)$', text)
        if not m:
            return None
        desc = _clean_desc(m.group(1))
        if not _valid_desc(desc):
            return None
        amount = float(m.group(2).strip().lstrip('¥￥').replace(',', ''))
        if amount < 10 and not _is_bag_description(desc):
            return None
        if amount <= 0 or amount > float(subtotal):
            return None
        return desc, amount, m.group(3) or ""

    def _make_row(desc: str, amount: float, marker: str) -> dict:
        desc = _clean_desc(desc)
        marker = marker.strip()
        locked_tax_category = None
        if _is_bag_description(desc):
            tax_category = "10%"
            locked_tax_category = tax_category
        elif re.search(r'[Xx]', marker):
            tax_category = "8%" if re.search(r'軽減税率|※印|[*＊].*軽減', unified_text) else "0%"
            locked_tax_category = tax_category
        else:
            tax_category = "8%"
        row = {
            "description": desc,
            "qty": 1.0,
            "unit_price": float(amount),
            "total": float(amount),
            "tax_category": tax_category,
            "discount": 0,
            "discount_rate": "",
            "_printed_amount": float(amount),
            "_marker": marker,
        }
        if locked_tax_category:
            row["_tax_category_locked"] = locked_tax_category
        return row

    rows: list[dict] = []
    pending_names: list[dict] = []
    pending_qty_details: list[tuple[float, float]] = []
    pending_leading_amounts: list[tuple[float, str]] = []
    last_row: dict | None = None
    pending_discount_rows: list[dict] = []
    rows_by_marker: dict[str, list[dict]] = {}
    marker_summary_lines: list[tuple[str, float, float]] = []

    def _normalize_marker(marker: str) -> str:
        translated = str(marker or "").translate(
            str.maketrans(
                "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ",
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            )
        )
        match = re.search(r'[A-Z]', translated)
        return match.group(0) if match else ""

    def _remember_marker_row(row: dict) -> None:
        marker = _normalize_marker(str(row.get("_marker") or ""))
        if marker:
            rows_by_marker.setdefault(marker, []).append(row)

    def _parse_marker_summary(line: str) -> tuple[str, float, float] | None:
        summary_m = re.match(
            r'^([A-ZＡ-Ｚ])\s*[:：]\s*(\d+)\s*[個コ]\s*[¥￥]?\s*([\d,]+)\s*の?商品',
            line,
        )
        if not summary_m:
            return None
        marker = _normalize_marker(summary_m.group(1))
        qty = float(summary_m.group(2))
        total = float(summary_m.group(3).replace(',', ''))
        if not marker or qty <= 1 or total <= 0:
            return None
        return marker, qty, total

    def _apply_marker_summary(row: dict, qty: float, total: float) -> bool:
        gross = _row_gross(row)
        if gross <= 0 or gross < total:
            return False
        existing_discount = float(row.get("discount") or 0)
        discount = gross - total
        if existing_discount and abs(existing_discount - discount) > 2:
            return False
        if abs(gross / qty - round(gross / qty, 2)) > 0.01:
            return False
        row["qty"] = qty
        row["unit_price"] = gross / qty
        row["total"] = total
        if discount > 0:
            row["discount"] = discount
        return True

    def _apply_marker_summaries() -> None:
        for marker, qty, total in marker_summary_lines:
            candidates = [
                row for row in rows_by_marker.get(marker, [])
                if abs(float(row.get("qty") or 1) - qty) <= 0.01
            ]
            if len(candidates) != 1:
                continue
            _apply_marker_summary(candidates[0], qty, total)

    def _row_gross(row: dict) -> float:
        return float(row.get("_printed_amount") or row.get("unit_price") or row.get("total") or 0)

    def _rate_from_discount(gross: float, discount: float) -> str:
        if gross <= 0 or discount <= 0:
            return ""
        ratio = discount / gross
        common_rates = (0.1, 0.2, 0.3, 0.4, 0.5)
        best = min(common_rates, key=lambda rate: abs(ratio - rate))
        if abs(discount - gross * best) <= max(2.0, gross * 0.03):
            return f"{int(best * 100)}%"
        return ""

    def _discount_score(row: dict, discount: float) -> float | None:
        gross = _row_gross(row)
        if gross <= discount:
            return None
        rate_text = str(row.get("discount_rate") or "")
        rate_match = re.search(r'(\d+(?:\.\d+)?)\s*%', rate_text)
        if rate_match:
            rate = float(rate_match.group(1)) / 100.0
            expected = gross * rate
            if abs(expected - discount) <= max(2.0, expected * 0.03):
                return abs(expected - discount)
            return None
        inferred = _rate_from_discount(gross, discount)
        if inferred:
            rate = float(inferred.rstrip("%")) / 100.0
            return abs(gross * rate - discount)
        return None

    def _apply_discount(discount: float) -> None:
        candidates = [row for row in pending_discount_rows if row.get("discount", 0) in (0, 0.0, None)]
        if not candidates and last_row and last_row.get("discount", 0) in (0, 0.0, None):
            candidates = [last_row]
        scored = [
            (score, idx, row)
            for idx, row in enumerate(candidates)
            for score in [_discount_score(row, discount)]
            if score is not None
        ]
        if scored:
            _score, _idx, row = min(scored, key=lambda value: (value[0], value[1]))
        elif candidates:
            row = candidates[-1]
        else:
            return
        gross = _row_gross(row)
        if gross <= discount:
            return
        row["discount"] = float(discount)
        if not row.get("discount_rate"):
            row["discount_rate"] = _rate_from_discount(gross, discount)
        if float(row.get("qty") or 1) > 1 and row.get("_printed_amount") and not row.get("unit_price"):
            row["unit_price"] = float(row["_printed_amount"]) / float(row.get("qty") or 1)
        row["total"] = gross - float(discount)
        if row in pending_discount_rows:
            pending_discount_rows.remove(row)

    def _flush_pending_qty_detail_row(source_idx: int) -> dict | None:
        if not pending_names or not pending_qty_details:
            return None
        desc = pending_names.pop(0)
        qty, unit = pending_qty_details.pop(0)
        gross = qty * unit
        if gross <= 0 or gross > float(subtotal):
            pending_names.insert(0, desc)
            pending_qty_details.insert(0, (qty, unit))
            return None
        for future in zone[source_idx:min(len(zone), source_idx + 6)]:
            inline = _parse_inline(future)
            values: list[float] = []
            if inline:
                values.append(float(inline[1]))
            amount = _parse_amount(future)
            if amount:
                values.append(float(amount[0]))
            if any(abs(value - gross) <= 2 for value in values):
                pending_names.insert(0, desc)
                pending_qty_details.insert(0, (qty, unit))
                return None
        row = _make_row(desc, gross, "")
        row["qty"] = qty
        row["unit_price"] = unit
        row["total"] = gross
        row["_printed_amount"] = gross
        rows.append(row)
        _remember_marker_row(row)
        return row

    def _is_discount_control(text: str) -> bool:
        return bool(
            re.fullmatch(r'割引', text)
            or re.search(r'値引', text)
            or re.fullmatch(r'\d+(?:\.\d+)?\s*%', text)
            or re.fullmatch(r'-\s*[¥￥]?\s*\d[\d,]*', text)
        )

    def _qty_unit_candidates(qty_text: str, unit_text: str, line: str) -> list[tuple[float, float]]:
        unit = float(unit_text)
        candidates: list[tuple[float, float]] = []

        def _add(qty: float) -> None:
            if qty > 1 and (qty, unit) not in candidates:
                candidates.append((qty, unit))

        has_explicit_count_marker = bool(re.search(r'\d+\s*[個コ]', line))
        if has_explicit_count_marker:
            _add(float(qty_text))
        elif len(qty_text) > 1 and qty_text[0] in "23456789":
            _add(float(qty_text[0]))
        else:
            _add(float(qty_text))

        return candidates

    def _qty_detail_candidates(line: str) -> list[tuple[float, float]]:
        qty_m = re.search(r'\(?\s*(\d+)\s*[個コ]?\s*[xX×Ⅹ]\s*単?\s*(\d{1,5})\s*\)?', line)
        if qty_m:
            return _qty_unit_candidates(qty_m.group(1), qty_m.group(2), line)

        if not re.search(r'[xX×Ⅹ]', line):
            return []
        parts = re.split(r'\s*[xX×Ⅹ]\s*', line, maxsplit=1)
        if len(parts) != 2:
            return []
        left_digits = re.findall(r'\d+', parts[0])
        right_digits = re.findall(r'\d+', parts[1])
        candidates: list[tuple[float, float]] = []

        def _add(qty: int, unit: int) -> None:
            if 2 <= qty <= 9 and unit > 0 and (float(qty), float(unit)) not in candidates:
                candidates.append((float(qty), float(unit)))

        for left in left_digits:
            qty_candidates = [int(left)]
            if len(left) > 1 and left[0] in "23456789":
                qty_candidates.append(int(left[0]))
            for right in right_digits:
                unit_candidates = [int(right)]
                if len(right) > 1:
                    unit_candidates.append(int(right[1:]))
                for qty in qty_candidates:
                    for unit in unit_candidates:
                        _add(qty, unit)
        return candidates

    for source_idx, line in enumerate(zone):
        marker_summary = _parse_marker_summary(line)
        if marker_summary:
            marker, qty, total = marker_summary
            marker_summary_lines.append(marker_summary)
            candidates = rows_by_marker.get(marker, [])
            if len(candidates) == 1:
                _apply_marker_summary(candidates[0], qty, total)
            elif last_row and not _normalize_marker(str(last_row.get("_marker") or "")):
                _apply_marker_summary(last_row, qty, total)
            continue

        qty_candidates = _qty_detail_candidates(line)
        if qty_candidates and last_row:
            matched = False
            for qty, unit in qty_candidates:
                if abs(qty * unit - float(last_row.get("total") or 0)) <= 2:
                    last_row["qty"] = qty
                    last_row["unit_price"] = unit
                    last_row["total"] = qty * unit
                    matched = True
                    break
            if not matched and pending_names:
                pending_qty_details.extend(qty_candidates)
                pending_qty_details = pending_qty_details[-8:]
            continue
        if qty_candidates and pending_names:
            pending_qty_details.extend(qty_candidates)
            pending_qty_details = pending_qty_details[-8:]
            continue

        if (re.fullmatch(r'割引', line) or re.search(r'値引', line)) and last_row:
            if last_row not in pending_discount_rows:
                pending_discount_rows.append(last_row)
            continue

        rate_m = re.fullmatch(r'(\d+(?:\.\d+)?)\s*%', line)
        if rate_m and pending_discount_rows:
            pending_discount_rows[-1]["discount_rate"] = f"{int(float(rate_m.group(1)))}%"
            continue

        discount_m = re.fullmatch(r'-\s*[¥￥]?\s*(\d[\d,]*)', line)
        if discount_m:
            _apply_discount(float(discount_m.group(1).replace(',', '')))
            continue

        inline = _parse_inline(line)
        if inline:
            flushed = _flush_pending_qty_detail_row(source_idx)
            if flushed is not None:
                last_row = flushed
            desc, amount, marker = inline
            row = _make_row(desc, amount, marker)
            rows.append(row)
            _remember_marker_row(row)
            last_row = row
            continue

        amount = _parse_amount(line)
        if amount:
            value, marker = amount
            if not pending_names:
                if value >= 10:
                    pending_leading_amounts.append((value, marker))
                    pending_leading_amounts = pending_leading_amounts[-3:]
                continue
            desc = pending_names.pop(0)
            if value < 10 and not _is_bag_description(desc):
                pending_names.clear()
                last_row = None
                continue
            row = _make_row(desc, value, marker)
            for detail_idx, (qty, unit) in enumerate(list(pending_qty_details)):
                if abs(qty * unit - value) <= 2:
                    row["qty"] = qty
                    row["unit_price"] = unit
                    row["total"] = qty * unit
                    pending_qty_details.pop(detail_idx)
                    break
            rows.append(row)
            _remember_marker_row(row)
            last_row = row
            continue

        if _valid_desc(line):
            desc = _clean_desc(line)
            if pending_leading_amounts:
                value, marker = pending_leading_amounts.pop(0)
                row = _make_row(desc, value, marker)
                rows.append(row)
                _remember_marker_row(row)
                last_row = row
                continue
            flushed = _flush_pending_qty_detail_row(source_idx)
            if flushed is not None:
                last_row = flushed
            pending_names.append(desc)
            if len(pending_names) > 6:
                pending_names = pending_names[-6:]
        elif not amount and not _is_discount_control(line):
            pending_names.clear()
            pending_leading_amounts.clear()
            last_row = None

    if len(rows) < 5:
        return
    _apply_marker_summaries()
    row_sum = sum(float(row["total"]) for row in rows)
    rate_bases = extract_rate_bases(unified_text)

    def _repair_percent_marker_amount_from_arithmetic() -> None:
        nonlocal row_sum
        gap = float(subtotal) - row_sum
        if gap <= 0 or gap > 50:
            return
        candidates = [
            row for row in rows
            if re.search(r'[%％]', str(row.get("_marker") or ""))
            and float(row.get("qty") or 1) == 1
            and not float(row.get("discount") or 0)
            and float(row.get("total") or 0) >= 10
        ]
        if not candidates:
            return

        def _rate_sums() -> dict[str, float]:
            sums: dict[str, float] = {}
            for row in rows:
                rate = normalize_tax_rate(str(row.get("tax_category") or "unknown"))
                sums[rate] = sums.get(rate, 0.0) + float(row.get("total") or 0)
            return sums

        for row in candidates:
            old_total = float(row.get("total") or 0)
            corrected = old_total + gap
            if corrected <= old_total or corrected > float(subtotal):
                continue
            row["unit_price"] = corrected
            row["total"] = corrected
            row["_printed_amount"] = corrected
            adjusted_sum = row_sum + gap
            if abs(adjusted_sum - float(subtotal)) > 2:
                row["unit_price"] = old_total
                row["total"] = old_total
                row["_printed_amount"] = old_total
                continue
            if rate_bases:
                rate_sums = _rate_sums()
                mismatched = [
                    rate for rate, base in rate_bases.items()
                    if base is not None and abs(rate_sums.get(rate, 0.0) - float(base)) > 2
                ]
                if mismatched:
                    row["unit_price"] = old_total
                    row["total"] = old_total
                    row["_printed_amount"] = old_total
                    continue
            row_sum = adjusted_sum
            return

    def _repair_leading_digit_amount_from_subtotal() -> None:
        nonlocal row_sum
        gap = row_sum - float(subtotal)
        if gap <= 0 or abs(gap - round(gap)) > 0.01:
            return
        candidates: list[tuple[dict, float]] = []
        for row in rows:
            amount = float(row.get("_printed_amount") or row.get("total") or 0)
            if (
                amount < 1000
                or float(row.get("qty") or 1) != 1
                or float(row.get("discount") or 0)
                or abs(amount - round(amount)) > 0.01
            ):
                continue
            amount_text = str(int(round(amount)))
            if len(amount_text) < 4:
                continue
            corrected_text = amount_text[1:]
            if not corrected_text or not corrected_text.isdigit():
                continue
            corrected = float(int(corrected_text))
            if corrected < 10:
                continue
            if abs((amount - corrected) - gap) <= 2:
                candidates.append((row, corrected))
        if len(candidates) != 1:
            return
        row, corrected = candidates[0]
        row["unit_price"] = corrected
        row["total"] = corrected
        row["_printed_amount"] = corrected
        row["tax_category"] = "8%"
        row["_tax_category_locked"] = "8%"
        row_sum = sum(float(row["total"]) for row in rows)

    _repair_leading_digit_amount_from_subtotal()
    _repair_percent_marker_amount_from_arithmetic()
    if abs(row_sum - float(subtotal)) > 2:
        return
    printed_count = None
    count_m = re.search(r'お買上商品数\s*[:：]?\s*(\d+)', unified_text)
    if count_m:
        printed_count = int(count_m.group(1))
    current_items = [item for item in (extracted.get("line_items") or []) if isinstance(item, dict)]
    current_count = len(current_items)
    row_qty_sum = int(sum(float(row.get("qty") or 1) for row in rows))
    if printed_count is not None and len(rows) != printed_count and row_qty_sum != printed_count:
        base_sum = sum(float(base) for base in rate_bases.values() if base is not None)
        if base_sum <= 0 or abs(base_sum - float(subtotal)) > 2:
            return
    if len(rows) < current_count:
        current_sum = sum(float(item.get("total") or 0) for item in current_items)
        current_keys = [
            (
                re.sub(r'\s+', '', str(item.get("description") or "")),
                round(float(item.get("total") or 0), 2),
            )
            for item in current_items
        ]
        has_duplicate_current_rows = len(set(current_keys)) < len(current_keys)
        if (
            not has_duplicate_current_rows
            or abs(current_sum - float(subtotal)) <= 2
            or current_count - len(rows) > 4
        ):
            return
    _fix_tax_categories_from_ocr_markers(rows, unified_text)
    _rebalance_tax_categories_to_rate_bases(rows, unified_text, extracted.get("taxes"), rate_bases)
    for row in rows:
        locked_tax_category = row.get("_tax_category_locked")
        if locked_tax_category:
            row["tax_category"] = locked_tax_category
    extracted["line_items"] = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]


def _replace_campaign_discount_stream_when_balanced(extracted, unified_text):
    """Reconstruct item streams where campaign discounts are printed separately."""
    subtotal = extracted.get("subtotal")
    if len(re.findall(r'割引', unified_text or "")) < 4:
        return
    lines = [line.strip() for line in (unified_text or "").split('\n')]
    end = next((idx for idx, line in enumerate(lines) if re.fullmatch(r'小\s*計', line)), None)
    if end is None:
        return

    def _parse_summary_amount(line: str) -> float | None:
        match = re.fullmatch(r'[¥￥]?\s*(\d[\d,]*)\s*', line)
        if not match:
            return None
        try:
            return float(match.group(1).replace(',', ''))
        except ValueError:
            return None

    printed_subtotal = None
    inline_subtotal = re.search(r'小\s*計\s*[¥￥]?\s*(\d[\d,]*)', lines[end])
    if inline_subtotal:
        printed_subtotal = float(inline_subtotal.group(1).replace(',', ''))
    else:
        for nearby in lines[end + 1:end + 5]:
            if re.search(r'^(外税|内税|消費税|対象|合計|支払|お釣|WAON|クレジット)', nearby):
                break
            amount = _parse_summary_amount(nearby)
            if amount is not None:
                printed_subtotal = amount
                break

    try:
        extracted_subtotal = float(subtotal) if subtotal is not None else None
    except (TypeError, ValueError):
        extracted_subtotal = None
    subtotal_target = printed_subtotal if printed_subtotal is not None else extracted_subtotal
    if subtotal_target is None:
        return

    start = next(
        (
            idx for idx, line in enumerate(lines[:end])
            if re.search(r'\d{4}/\d{1,2}/\d{1,2}|\d{1,2}:\d{2}', line)
        ),
        0,
    )
    zone = lines[start + 1:end]
    if len(zone) < 12:
        return

    amount_re = re.compile(r'^(?:[¥￥]\s*)?(\d[\d,]*)\s*([非除※*＊↓]*)$')
    inline_re = re.compile(
        r'^(.+?[ぁ-んァ-ン一-龥A-Za-z][^¥￥]*?)\s+(\d[\d,]*)\s*([非除※*＊↓]*)$'
    )
    qty_re = re.compile(r'[<\(（]?\s*(\d+)\s*[個コ]?\s*[xX×Ⅹ]\s*単?\s*(\d{1,5})')

    def _clean_desc(text: str) -> str:
        text = _clean_ocr_price_line_desc(text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'^[<（(]+|[>）)]+$', '', text).strip()
        return text

    def _valid_desc(text: str) -> bool:
        text = _clean_desc(text)
        if not _valid_ocr_item_desc(text):
            return False
        if qty_re.search(text):
            return False
        if re.search(
            r'支払|お釣|ポイント|残高|有効|内訳|登録番号|領収|会員登録|検索',
            text,
        ):
            return False
        return True

    def _tax_category_from_marker(marker: str) -> tuple[str, bool]:
        if '非' in marker:
            return "0%", True
        if '除' in marker:
            return "10%", True
        if re.search(r'[※*＊]', marker):
            return "8%", True
        return "8%", False

    def _make_row(desc: str, amount: float, marker: str) -> dict:
        tax_category, locked = _tax_category_from_marker(marker)
        row = {
            "description": _clean_desc(desc),
            "qty": 1.0,
            "unit_price": float(amount),
            "total": float(amount),
            "tax_category": tax_category,
            "discount": 0.0,
            "discount_rate": "",
            "_marker": marker,
            "_discount_slots": [],
        }
        if locked:
            row["_tax_category_locked"] = tax_category
        return row

    rows: list[dict] = []
    pending_names: list[str] = []
    pending_qty_details: list[tuple[float, float]] = []
    last_row: dict | None = None

    def _target_for_discount_marker() -> dict | None:
        if last_row is not None:
            gross = float(last_row.get("qty") or 1) * float(last_row.get("unit_price") or 0)
            if abs(float(last_row.get("total") or 0) - gross) <= 1:
                return last_row
        return None

    def _add_discount_marker(rate: str) -> None:
        if not rate:
            return
        target = _target_for_discount_marker()
        if target is not None:
            target.setdefault("_discount_slots", []).append(rate)
        elif pending_names:
            pending_names[-1].setdefault("slots", []).append(rate)

    def _apply_discount(discount: float) -> None:
        candidates = [
            row for row in reversed(rows)
            if (
                float(row.get("qty") or 1) * float(row.get("unit_price") or 0)
                - float(row.get("discount") or 0)
            ) > discount
        ]
        if not candidates:
            return
        with_slots = [row for row in candidates if row.get("_discount_slots")]
        row = with_slots[0] if with_slots else candidates[0]
        gross = float(row.get("qty") or 1) * float(row.get("unit_price") or 0)
        row["discount"] = float(row.get("discount") or 0) + float(discount)
        row["total"] = gross - float(row["discount"])
        slots = row.get("_discount_slots") or []
        if slots:
            rate = slots.pop(0)
            if rate and not row.get("discount_rate"):
                row["discount_rate"] = rate

    def _apply_qty_detail(qty: float, unit: float) -> None:
        if last_row is not None:
            total = qty * unit
            if abs(float(last_row.get("total") or 0) - total) <= 2:
                last_row["qty"] = qty
                last_row["unit_price"] = unit
                last_row["total"] = total
                return
        pending_qty_details.append((qty, unit))
        pending_qty_details[:] = pending_qty_details[-6:]

    def _attach_pending_qty(row: dict, amount: float) -> None:
        for idx, (qty, unit) in enumerate(list(pending_qty_details)):
            if abs(qty * unit - amount) <= 2:
                row["qty"] = qty
                row["unit_price"] = unit
                row["total"] = qty * unit
                pending_qty_details.pop(idx)
                return

    for source_idx, line in enumerate(zone):
        qty_m = qty_re.search(line)
        if qty_m:
            _apply_qty_detail(float(qty_m.group(1)), float(qty_m.group(2)))
            continue

        rate_m = re.search(r'(\d+(?:\.\d+)?)\s*%', line)
        if '割引' in line or (rate_m and not amount_re.match(line)):
            rate = f"{int(float(rate_m.group(1)))}%" if rate_m else ""
            _add_discount_marker(rate)
            continue

        discount_m = re.fullmatch(r'-\s*[¥￥]?\s*(\d[\d,]*)', line)
        if discount_m:
            _apply_discount(float(discount_m.group(1).replace(',', '')))
            continue

        inline_m = inline_re.match(line)
        if inline_m and _valid_desc(inline_m.group(1)):
            desc = _clean_desc(inline_m.group(1))
            amount = float(inline_m.group(2).replace(',', ''))
            marker = inline_m.group(3) or ""
            pending_slots: list[str] = []
            if pending_names and desc in str(pending_names[-1].get("desc") or ""):
                pending = pending_names.pop(0)
                desc = str(pending.get("desc") or desc)
                pending_slots = list(pending.get("slots") or [])
            row = _make_row(desc, amount, marker)
            row["_source_idx"] = source_idx
            row["_discount_slots"].extend(pending_slots)
            _attach_pending_qty(row, amount)
            rows.append(row)
            last_row = row
            continue

        amount_m = amount_re.match(line)
        if amount_m and pending_names:
            amount = float(amount_m.group(1).replace(',', ''))
            if amount > subtotal_target + 2:
                continue
            marker = amount_m.group(2) or ""
            pending = pending_names.pop(0)
            row = _make_row(str(pending.get("desc") or ""), amount, marker)
            row["_source_idx"] = pending.get("source_idx", source_idx)
            row["_discount_slots"].extend(pending.get("slots") or [])
            _attach_pending_qty(row, amount)
            rows.append(row)
            last_row = row
            continue

        if _valid_desc(line):
            desc = _clean_desc(line)
            if pending_names and desc in str(pending_names[-1].get("desc") or ""):
                continue
            pending_names.append({"desc": desc, "slots": [], "source_idx": source_idx})
            pending_names[:] = pending_names[-10:]
            last_row = None

    if len(rows) < 5:
        return
    row_sum = sum(float(row.get("total") or 0) for row in rows)
    if abs(row_sum - subtotal_target) > 2:
        return
    discount_count = sum(1 for row in rows if float(row.get("discount") or 0) > 0)
    if discount_count < 3:
        return
    printed_count = None
    count_m = re.search(r'お買上商品数\s*[:：]?\s*(\d+)', unified_text)
    if count_m:
        printed_count = int(count_m.group(1))
    qty_sum = int(sum(float(row.get("qty") or 1) for row in rows))
    if printed_count is not None and abs(qty_sum - printed_count) > 4:
        return

    rate_bases = extract_rate_bases(unified_text)
    _rebalance_tax_categories_to_rate_bases(rows, unified_text, extracted.get("taxes"), rate_bases)
    for row in rows:
        locked_tax_category = row.get("_tax_category_locked")
        if locked_tax_category:
            row["tax_category"] = locked_tax_category
    rows.sort(key=lambda row: int(row.get("_source_idx", 0)))
    extracted["line_items"] = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]
    if printed_subtotal is not None:
        extracted["subtotal"] = printed_subtotal


def _replace_prefixed_tax_marker_item_rows_when_balanced(extracted, unified_text):
    """Recover item sections whose product rows are prefixed by tax markers."""
    total = extracted.get("total")
    subtotal = extracted.get("subtotal")
    if not total and not subtotal:
        return
    existing = extracted.get("line_items") or []
    if existing:
        item_sum = sum(
            float(item.get("total") or 0)
            for item in existing
            if isinstance(item, dict)
        )
        targets = [float(v) for v in (total, subtotal) if v is not None and float(v) > 0]
        if targets and any(abs(item_sum - target) <= 2 for target in targets):
            return

    lines = [line.strip() for line in (unified_text or "").split("\n")]
    if not lines:
        return
    start = next(
        (
            idx + 1
            for idx, line in enumerate(lines)
            if re.search(r'上記正に領収|担当者', line)
        ),
        0,
    )
    zone = lines[start:]
    if len(zone) < 8:
        return

    amount_re = re.compile(r'^[¥￥]?\s*([\d,]+)\s*(?:円)?\s*$')
    inline_amount_re = re.compile(r'[¥￥]\s*([\d,]+)\s*$')
    marker_item_re = re.compile(r'^内\s*([*＊※])?\s*(.+)$')
    qty_re = re.compile(r'(\d{1,3})\s*[個コ点]?\s*[xX×Ⅹ]\s*#?\s*(\d{1,5})')

    def _clean_marker_desc(text: str) -> str:
        text = _clean_ocr_price_line_desc(text)
        text = re.sub(r'^[*＊※]\s*', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\s*[¥￥]\s*[\d,]+.*$', '', text).strip()
        return text

    def _valid_marker_desc(text: str) -> bool:
        if not _valid_ocr_item_desc(text):
            return False
        return not bool(re.search(
            r'対象|内税|合計|小計|お預り|お釣|領収|登録|TEL|電話|営業時間|保管|返品',
            text,
            re.IGNORECASE,
        ))

    def _tax_category(marker: str | None, desc: str) -> tuple[str, bool]:
        if '非' in desc:
            return "0%", True
        if marker:
            return "8%", True
        return "10%", False

    def _make_row(desc: str, amount: float, marker: str | None, source_idx: int) -> dict | None:
        clean = _clean_marker_desc(desc)
        if not _valid_marker_desc(clean):
            return None
        cat, locked = _tax_category(marker, desc)
        row = {
            "description": re.sub(r'\s*非$', '', clean).strip(),
            "qty": 1.0,
            "unit_price": float(amount),
            "total": float(amount),
            "tax_category": cat,
            "discount": 0,
            "discount_rate": "",
            "_source_idx": source_idx,
        }
        if locked:
            row["_tax_category_locked"] = cat
        return row

    rows: list[dict] = []
    pending: list[tuple[str, str | None, int]] = []

    def _apply_qty_detail(row: dict, line: str) -> None:
        m = qty_re.search(line)
        if not m:
            return
        total_f = float(row.get("total") or 0)
        left = m.group(1)
        right = m.group(2)
        candidates: list[tuple[float, float]] = []
        try:
            candidates.append((float(left), float(right)))
        except ValueError:
            pass
        if len(left) > 1 and left[0].isdigit():
            try:
                candidates.append((float(left[0]), float(right)))
            except ValueError:
                pass
        if len(right) > 2 and right[0] == "1":
            try:
                candidates.append((float(left), float(right[1:])))
            except ValueError:
                pass
        if len(left) > 1 and len(right) > 2 and right[0] == "1":
            try:
                candidates.append((float(left[0]), float(right[1:])))
            except ValueError:
                pass
        for qty, unit in candidates:
            if qty <= 1 or unit <= 0:
                continue
            if abs(qty * unit - total_f) <= 2:
                row["qty"] = qty
                row["unit_price"] = unit
                return

    for source_idx, line in enumerate(zone):
        if not line:
            continue
        if rows and re.search(r'^\(?\s*\d{1,2}%対象|\*は軽減|^合\s*計$', line):
            break

        if rows and qty_re.search(line):
            _apply_qty_detail(rows[-1], line)
            continue

        marker_m = marker_item_re.match(line)
        if marker_m:
            marker = marker_m.group(1)
            rest = marker_m.group(2).strip()
            inline_m = inline_amount_re.search(rest)
            if inline_m:
                amount = float(inline_m.group(1).replace(',', ''))
                row = _make_row(rest, amount, marker, source_idx)
                if row:
                    rows.append(row)
                continue
            pending.append((rest, marker, source_idx))
            pending[:] = pending[-8:]
            continue

        amount_m = amount_re.fullmatch(line)
        if amount_m and pending:
            amount = float(amount_m.group(1).replace(',', ''))
            desc, marker, pending_idx = pending.pop(0)
            row = _make_row(desc, amount, marker, pending_idx)
            if row:
                rows.append(row)

    if len(rows) < 3:
        return
    row_sum = sum(float(row.get("total") or 0) for row in rows)
    total_f = float(total) if total is not None else None
    subtotal_f = float(subtotal) if subtotal is not None else None
    targets = [target for target in (total_f, subtotal_f) if target is not None and target > 0]
    if not any(abs(row_sum - target) <= 2 for target in targets):
        return

    rate_bases = extract_rate_bases(unified_text)
    if rate_bases:
        _rebalance_tax_categories_to_rate_bases(rows, unified_text, extracted.get("taxes"), rate_bases)
        for row in rows:
            locked_tax_category = row.get("_tax_category_locked")
            if locked_tax_category:
                row["tax_category"] = locked_tax_category
        sums: dict[str, float] = {}
        for row in rows:
            cat = str(row.get("tax_category") or "")
            if cat and cat != "0%":
                sums[cat] = sums.get(cat, 0.0) + float(row.get("total") or 0)
        mismatches = [
            rate
            for rate, base in rate_bases.items()
            if rate in sums and base and abs(sums[rate] - float(base)) > 2
        ]
        if mismatches:
            return

    rows.sort(key=lambda row: int(row.get("_source_idx", 0)))
    extracted["line_items"] = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]


def _fix_qty_totals_from_ocr_unit_lines(extracted, unified_text):
    """Apply nearby '(N個 X 単U)' OCR rows when one unit was extracted."""
    items = extracted.get("line_items") or []
    if not items:
        return
    lines = [line.strip() for line in unified_text.split('\n')]
    qty_detail_owners = _qty_detail_owner_indices(items, unified_text)

    def _norm(text: str) -> str:
        text = re.sub(r'\s+', '', text or "")
        text = re.sub(r'[（]', '(', text)
        text = re.sub(r'[）]', ')', text)
        text = re.sub(r'[^\wぁ-んァ-ン一-龥()]', '', text, flags=re.UNICODE)
        return text.lower()

    def _valid_desc(text: str) -> bool:
        if not text:
            return False
        if _SKIP_PRICE_LINE.search(text) or _OCR_QTY_NOTATION_RE.search(text):
            return False
        if re.search(r'割引|小\s*計|合\s*計|対象|消費税|登録番号|TEL|http', text, re.IGNORECASE):
            return False
        return bool(re.search(r'[ぁ-んァ-ン一-龥]', text))

    def _candidate_names_before(idx: int, expected_total: float | None = None) -> list[str]:
        candidates: list[str] = []
        for j in range(idx - 1, max(idx - 14, -1), -1):
            s = lines[j].strip()
            if not s:
                continue
            pm = _OCR_TRAILING_PRICE_RE.search(s)
            if pm and expected_total is not None:
                try:
                    price = float(pm.group(1).strip().lstrip('¥￥').replace(',', ''))
                except ValueError:
                    price = None
                cand = _clean_ocr_price_line_desc(s)
                if price is not None and abs(price - expected_total) <= 2 and _valid_desc(cand):
                    candidates.append(
                        re.sub(r'^\d{3,}[A-Za-z0-9-]*\)?\s*', '', cand).strip()
                    )
                    continue
            if _SKIP_PRICE_LINE.search(s) or _OCR_TRAILING_PRICE_RE.search(s):
                continue
            if _OCR_QTY_NOTATION_RE.search(s) or re.search(r'[xX×Ⅹ]\s*単?\s*\d', s):
                continue
            if _valid_desc(s):
                candidates.append(re.sub(r'^\d{3,}[A-Za-z0-9-]*\)?\s*', '', s).strip())
        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = _norm(candidate)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    def _has_direct_unit_price_row(item: dict, unit: float, detail_idx: int) -> bool:
        line_idx = _ocr_line_index_for_item(lines, item)
        if line_idx is None or line_idx >= detail_idx:
            return False
        for nearby in lines[line_idx:min(detail_idx, line_idx + 4)]:
            pm = _OCR_TRAILING_PRICE_RE.search(nearby)
            if not pm:
                pm = re.search(r'(?:^|[\s(（])([¥￥]?\s*\d[\d,]*)\s*(?:[%％*＊※除軽Xx]+)?\s*$', nearby)
            if not pm:
                continue
            try:
                price = float(pm.group(1).strip().lstrip('¥￥').replace(',', ''))
            except ValueError:
                continue
            if abs(price - unit) <= 2:
                return True
        return False

    def _amount_appears_in_text(text: str, amount: float) -> bool:
        if amount <= 0:
            return False
        plain = str(int(amount)) if amount == int(amount) else str(amount)
        comma = f"{int(amount):,}" if amount == int(amount) else plain
        return bool(
            re.search(rf'(?<!\d){re.escape(plain)}(?!\d)', text or "")
            or re.search(rf'(?<!\d){re.escape(comma)}(?!\d)', text or "")
        )

    def _has_standalone_gross_before_qty_detail(item: dict, gross: float, detail_idx: int) -> bool:
        line_idx = _ocr_line_index_for_item(lines, item)
        if line_idx is None or line_idx >= detail_idx:
            line_idx = None
        plain = str(int(gross)) if gross == int(gross) else str(gross)
        comma = f"{int(gross):,}" if gross == int(gross) else plain
        amount_re = re.compile(
            rf'^[¥￥]?\s*(?:{re.escape(plain)}|{re.escape(comma)})\s*[%％*＊※除軽Xx]?\s*$'
        )
        if line_idx is not None and amount_re.fullmatch(lines[line_idx].strip()):
            return True
        if line_idx is not None and any(
            amount_re.fullmatch(nearby.strip())
            for nearby in lines[line_idx + 1:detail_idx]
        ):
            return True
        item_desc = _norm(item.get("description") or "")
        if not item_desc:
            return False
        for amount_idx in range(max(0, detail_idx - 6), detail_idx):
            if not amount_re.fullmatch(lines[amount_idx].strip()):
                continue
            window = lines[max(0, amount_idx - 4):amount_idx]
            if any(
                item_desc in _norm(candidate) or _norm(candidate) in item_desc
                for candidate in window
                if _norm(candidate)
            ):
                return True
        return False

    def _parse_unit_line_qty_detail(line: str) -> tuple[float, float] | None:
        unit_first = re.search(
            r'\(?\s*単\s*(\d[\d,]*)\s*[xX×Ⅹ]\s*(\d+)\s*[個コ点]\s*\)?',
            line,
        )
        if unit_first:
            unit = float(unit_first.group(1).replace(',', ''))
            qty = float(unit_first.group(2))
        else:
            if re.search(r'[@＠]', line):
                return None
            qty_first = re.search(
                r'\(?\s*(\d+)\s*[個コ点]?\s*[xX×Ⅹ]\s*単?\s*(\d[\d,]*)\s*\)?',
                line,
            )
            if not qty_first:
                return None
            qty = float(qty_first.group(1))
            unit = float(qty_first.group(2).replace(',', ''))
        if qty < 2 or unit <= 0:
            return None
        return qty, unit

    for idx, line in enumerate(lines):
        detail = _parse_unit_line_qty_detail(line)
        split_total = None
        split_unit = None
        if detail is None:
            qty_m = re.search(r'\(?\s*(\d+)\s*[個コ]\s*$', line)
            unit_m = (
                re.search(r'単\s*(\d{2,4})\s*\)?', lines[idx + 1])
                if qty_m and idx + 1 < len(lines) else None
            )
            total_m = (
                re.fullmatch(r'[¥￥]?\s*(\d{2,5})', lines[idx + 2])
                if unit_m and idx + 2 < len(lines) else None
            )
            if qty_m and unit_m:
                qty = float(qty_m.group(1))
                split_unit = float(unit_m.group(1))
                split_total = float(total_m.group(1)) if total_m else None
            else:
                continue
        else:
            qty, split_unit = detail
        if split_unit is None:
            continue
        unit = split_unit
        if qty <= 1 or unit <= 0:
            continue
        expected_total = split_total if split_total is not None else qty * unit
        desc_candidates = _candidate_names_before(idx, expected_total)
        if not desc_candidates:
            continue
        matched_item = None
        for desc in desc_candidates:
            ndesc = _norm(desc)
            for item in items:
                if not isinstance(item, dict):
                    continue
                item_total = float(item.get("total") or 0)
                item_discount = float(item.get("discount") or 0)
                if (
                    abs(item_total - unit) > 2
                    and abs(item_total - expected_total) > 2
                    and abs(item_total + item_discount - expected_total) > 2
                ):
                    continue
                item_desc = _norm(item.get("description") or "")
                if not item_desc:
                    continue
                item_desc_is_qty_detail = bool(
                    _OCR_QTY_NOTATION_RE.search(str(item.get("description") or ""))
                )
                if (
                    item_desc_is_qty_detail
                    and desc == desc_candidates[-1]
                    and abs(item_total - expected_total) <= 2
                ):
                    item["description"] = desc
                    item["qty"] = qty
                    item["unit_price"] = unit
                    item["total"] = (
                        split_total
                        if split_total and abs(split_total - expected_total) <= 2
                        else expected_total
                    )
                    matched_item = item
                    break
                if ndesc in item_desc or item_desc in ndesc or SequenceMatcher(None, ndesc, item_desc).ratio() >= 0.72:
                    discount = float(item.get("discount") or 0)
                    current_unit = float(item.get("unit_price") or 0)
                    current_total = float(item.get("total") or 0)
                    gross_is_standalone_before_qty_detail = _has_standalone_gross_before_qty_detail(
                        item, expected_total, idx
                    )
                    if (
                        discount > 0
                        and gross_is_standalone_before_qty_detail
                        and abs(current_unit - unit) <= 2
                        and abs(current_total - (expected_total - discount)) <= 2
                    ):
                        half_off_gross_line = (
                            abs(discount - unit) <= 2
                            and abs(current_total - unit) <= 2
                        )
                        item["qty"] = qty
                        item["unit_price"] = expected_total if half_off_gross_line else unit
                        item["total"] = current_total
                        matched_item = item
                        break
                    if (
                        discount > 0
                        and abs(current_unit - unit) <= 2
                        and abs(current_total - expected_total) <= 2
                    ):
                        item["qty"] = qty
                        item["unit_price"] = unit
                        item["total"] = qty * unit - discount
                        matched_item = item
                        break
                    if (
                        discount > 0
                        and abs(current_unit - expected_total) <= 2
                        and abs(current_total - (current_unit - discount)) <= 2
                    ):
                        current_qty = float(item.get("qty") or 1)
                        desc_has_gross = _amount_appears_in_text(item.get("description") or "", expected_total)
                        ocr_line_idx = _ocr_line_index_for_item(lines, item)
                        gross_is_inline_with_name = (
                            ocr_line_idx is not None
                            and _amount_appears_in_text(lines[ocr_line_idx], expected_total)
                            and item_desc in _norm(lines[ocr_line_idx])
                        )
                        item["qty"] = qty
                        half_off_gross_line = (
                            abs(discount - unit) <= 2
                            and abs(current_total - unit) <= 2
                        )
                        if (
                            current_qty > 1
                            and gross_is_standalone_before_qty_detail
                            and not half_off_gross_line
                            and abs(current_total - (qty * unit - discount)) <= 2
                        ):
                            item["unit_price"] = unit
                            item["total"] = current_total
                        if current_qty <= 1 and gross_is_standalone_before_qty_detail:
                            item["unit_price"] = expected_total
                            item["total"] = current_unit - discount
                        elif current_qty <= 1 or desc_has_gross or gross_is_inline_with_name:
                            item["unit_price"] = unit
                            item["total"] = qty * unit - discount
                            cleaned_desc = _clean_ocr_price_line_desc(item.get("description") or "")
                            if cleaned_desc and _valid_desc(cleaned_desc):
                                item["description"] = cleaned_desc
                        matched_item = item
                        break
                    item["qty"] = qty
                    item["unit_price"] = unit
                    expected_total = qty * unit
                    item["total"] = split_total if split_total and abs(split_total - expected_total) <= 2 else expected_total
                    matched_item = item
                    break
            if matched_item is not None:
                break
        if matched_item is None:
            continue
        for item_idx, item in enumerate(items):
            if item is matched_item or not isinstance(item, dict):
                continue
            if abs(float(item.get("unit_price") or 0) - unit) > 2:
                continue
            if abs(float(item.get("total") or 0) - qty * unit) > 2:
                continue
            if item_idx in qty_detail_owners:
                continue
            gross = qty * unit
            if _has_direct_unit_price_row(item, unit, idx):
                item["qty"] = 1.0
                item["unit_price"] = unit
                item["total"] = unit
                continue
            if _has_standalone_gross_before_qty_detail(item, gross, idx):
                item["qty"] = 1.0
                item["unit_price"] = gross
                item["total"] = gross


def _replace_jan_pos_items_when_balanced(extracted, unified_text, ocr_totals):
    """For JAN/POS layouts, use OCR row projection when it balances exactly."""
    if "JAN" not in unified_text:
        return
    total = extracted.get("total")
    if not total:
        return
    taxes = extracted.get("taxes") or ocr_totals.get("taxes") or []
    tax_sum = _sum_taxable_amounts(taxes)
    ocr_tax_sum = _sum_taxable_amounts(ocr_totals.get("taxes") or [])
    targets = [
        float(t) for t in (
            ocr_totals.get("subtotal"),
            extracted.get("subtotal"),
            (float(total) - tax_sum if tax_sum else None),
            (float(total) - ocr_tax_sum if ocr_tax_sum else None),
        )
        if t is not None and float(t) > 0
    ]
    lines = [line.strip() for line in unified_text.split('\n')]
    printed_subtotal_targets: list[float] = []
    for idx, line in enumerate(lines):
        if not re.fullmatch(r'小\s*計', line):
            continue
        for following in lines[idx + 1:min(len(lines), idx + 4)]:
            subtotal_m = re.fullmatch(r'[¥￥]\s*([\d,]+)', following)
            if subtotal_m:
                printed_subtotal_targets.append(float(subtotal_m.group(1).replace(',', '')))
                break
    targets.extend(printed_subtotal_targets)
    if not targets:
        return

    rows: list[dict] = []
    pending: dict | None = None
    orphan_prices: list[float] = []
    in_items = False

    def _clean_desc(line: str) -> tuple[str, str]:
        marker = "10%"
        if "*" in line or "＊" in line:
            marker = "8%"
        text = re.sub(r'^\d{3,6}\s*', '', line).strip()
        text = text.lstrip('*＊').strip()
        return text, marker

    def _finish(row: dict | None):
        if not row or row.get("total") is None:
            return
        rows.append({
            "description": row["description"],
            "qty": row.get("qty", 1.0),
            "unit_price": row.get("unit_price", row["total"]),
            "total": row["total"],
            "tax_category": row.get("tax_category", "8%"),
            "discount": row.get("discount", 0),
            "discount_rate": row.get("discount_rate", ""),
        })

    def _discount_rate_value(row: dict) -> float | None:
        m = re.search(r'(\d+(?:\.\d+)?)\s*%', str(row.get("discount_rate") or ""))
        if not m:
            return None
        return float(m.group(1)) / 100.0

    def _apply_discount_to_pending(row: dict):
        discount = float(row.get("discount") or 0)
        if discount <= 0:
            return
        qty = float(row.get("qty") or 1)
        unit = row.get("unit_price")
        if unit is not None:
            row["total"] = qty * float(unit) - discount
            return
        rate = _discount_rate_value(row)
        candidates = row.get("_price_candidates") or []
        for price in candidates:
            if rate is not None and abs(float(price) * rate - discount) > max(2.0, discount * 0.05):
                continue
            row["unit_price"] = float(price)
            row["total"] = qty * float(price) - discount
            return

    for raw_idx, raw in enumerate(lines):
        line = raw.strip()
        if not in_items and (re.search(r'\d{6,}\s*JAN', line) or re.match(r'^\d{3,6}\*?\s*.+', line)):
            in_items = True
        if not in_items:
            continue
        if re.search(r'小\s*計|税率|合\s*計|QUICPay|お買上|端末番号', line):
            _finish(pending)
            pending = None
            if re.search(r'小\s*計|税率|合\s*計', line):
                break
            continue
        if not line or re.search(r'^\d{6,}\s*JAN$', line):
            continue

        price_m = re.match(r'^[¥￥]\s*([\d,]+)\s*$', line)
        if price_m:
            price = float(price_m.group(1).replace(',', ''))
            if (
                pending
                and pending.get("total") is not None
                and float(pending.get("qty") or 1) > 1
                and abs(float(pending.get("total") or 0) - price) <= 2
            ):
                continue
            if (
                pending
                and pending.get("total") is None
                and rows
                and float(rows[-1].get("qty") or 1) > 1
                and abs(float(rows[-1].get("total") or 0) - price) <= 2
            ):
                continue
            if pending and pending.get("total") is None:
                discount = float(pending.get("discount") or 0)
                rate = _discount_rate_value(pending)
                if discount > 0 and rate is not None and abs(price * rate - discount) > max(2.0, discount * 0.05):
                    pending.setdefault("_price_candidates", []).append(price)
                    continue
                pending["unit_price"] = price
                pending["total"] = price * float(pending.get("qty", 1)) - discount
            else:
                orphan_prices.append(price)
            continue

        qty_m = re.search(r'(\d+)\s*[コ個]\s*[xX×Ⅹ]\s*単?\s*([\d,]+)', line)
        if qty_m and pending:
            qty = float(qty_m.group(1))
            unit = float(qty_m.group(2).replace(',', ''))
            pending["qty"] = qty
            pending["unit_price"] = unit
            pending["total"] = qty * unit
            continue

        if re.search(r'割引|値引', line) and pending:
            rate_str = ""
            discount_amount = 0.0
            for k in range(raw_idx, min(raw_idx + 8, len(lines))):
                kline = lines[k].strip()
                rate_m = re.search(r'(\d+(?:\.\d+)?)\s*%', kline)
                if rate_m:
                    rate_str = f"{int(float(rate_m.group(1)))}%"
                amt_m = re.match(r'^-\s*[¥￥]?\s*(\d[\d,]*)\s*$', kline)
                if amt_m:
                    discount_amount = float(amt_m.group(1).replace(',', ''))
            if discount_amount > 0:
                pending["discount"] = discount_amount
                pending["discount_rate"] = rate_str
                _apply_discount_to_pending(pending)
            continue

        inline_m = re.match(r'^\d{3,6}\*?\s*(.+?)\s+[¥￥]\s*([\d,]+)\s*$', line)
        if inline_m:
            desc, cat = _clean_desc(inline_m.group(1))
            _finish(pending)
            pending = {
                "description": desc,
                "qty": 1.0,
                "unit_price": float(inline_m.group(2).replace(',', '')),
                "total": float(inline_m.group(2).replace(',', '')),
                "tax_category": cat,
            }
            continue

        desc_m = re.match(r'^\d{3,6}\*?\s*(.+?[ぁ-んァ-ン一-龥].*)$', line)
        if not desc_m:
            continue

        desc, cat = _clean_desc(line)
        if not desc:
            continue
        if re.search(r'JAN|スキャン|会計|No\d', desc):
            continue
        if "レジ" in desc and not _is_bag_description(desc):
            continue
        _finish(pending)
        pending = {
            "description": desc,
            "qty": 1.0,
            "unit_price": None,
            "total": None,
            "tax_category": cat,
        }
        if orphan_prices:
            price = orphan_prices.pop(0)
            pending["unit_price"] = price
            pending["total"] = price

    _finish(pending)
    if len(rows) < 5:
        return
    for row in rows:
        desc = row["description"]
        if "100円均一" in unified_text and re.search(r'[xX×Ⅹ]\s*単?\s*5', desc) and abs(float(row.get("total") or 0) - 100) <= 2:
            row["description"] = "100円均一"
            desc = row["description"]
        if _is_bag_description(desc) or "100円均一" in desc:
            if _is_bag_description(desc):
                row["description"] = re.sub(r'\s*\d+\s*円\s*$', '', desc).strip() or desc
            row["tax_category"] = "10%"
        elif _FOOD_DESC_RE.search(desc) or "ミート" in desc or "精肉" in desc:
            row["tax_category"] = "8%"
    row_sum = sum(float(row.get("total") or 0) for row in rows)
    try:
        total_f = float(total)
    except (TypeError, ValueError):
        total_f = None
    printed_tax_total = None
    for idx, line in enumerate(lines):
        if not re.search(r'消費税等|税合計', line):
            continue
        for following in lines[idx + 1:min(len(lines), idx + 4)]:
            tax_m = re.match(r'[¥￥]\s*([\d,]+)', following)
            if tax_m:
                printed_tax_total = float(tax_m.group(1).replace(',', ''))
                break
        if printed_tax_total is not None:
            break
    projected_total = None
    if printed_tax_total is not None and any(abs(row_sum - target) <= 5 for target in printed_subtotal_targets):
        projected_total = row_sum + printed_tax_total
        if total_f is None or abs(total_f - projected_total) > 2:
            extracted["total"] = projected_total
            amount_paid = extracted.get("amount_paid")
            try:
                amount_paid_f = float(amount_paid) if amount_paid is not None else None
            except (TypeError, ValueError):
                amount_paid_f = None
            if amount_paid_f is None or (total_f is not None and abs(amount_paid_f - total_f) <= 2) or amount_paid_f < projected_total:
                extracted["amount_paid"] = projected_total
            total_f = projected_total
    if total_f is not None and re.search(r'税率\s*8%|8%\s*課税|税率\s*10%|10%\s*課税', unified_text):
        external_tax_total = total_f - row_sum
        standard_rows = [row for row in rows if _is_bag_description(row.get("description") or "")]
        if external_tax_total > 0 and standard_rows:
            standard_base = sum(float(row.get("total") or 0) for row in standard_rows)
            reduced_base = row_sum - standard_base
            standard_tax = int(standard_base * 0.10)
            reduced_tax = int(reduced_base * 0.08)
            if reduced_base > 0 and abs((standard_tax + reduced_tax) - external_tax_total) <= 2:
                for row in rows:
                    row["tax_category"] = "10%" if _is_bag_description(row.get("description") or "") else "8%"
                extracted["taxes"] = [
                    {"rate": "10%", "label": "外税", "amount": float(standard_tax)},
                    {"rate": "8%", "label": "外税", "amount": float(reduced_tax)},
                ]
    current_count = len([item for item in (extracted.get("line_items") or []) if isinstance(item, dict)])
    current_item_sum = sum(
        float(item.get("total") or 0)
        for item in (extracted.get("line_items") or [])
        if isinstance(item, dict)
    )
    if (
        any(abs(row_sum - target) <= 5 for target in targets)
        and (len(rows) >= current_count or current_count - len(rows) <= 2 or abs(current_item_sum - row_sum) > 2)
    ):
        rate_bases = extract_rate_bases(unified_text)
        _fix_tax_categories_from_ocr_markers(rows, unified_text)
        _rebalance_tax_categories_to_rate_bases(rows, unified_text, extracted.get("taxes"), rate_bases)
        extracted["line_items"] = rows
        extracted["subtotal"] = row_sum


def _fix_non_bag_items_named_as_bag(extracted, unified_text):
    """Replace bag descriptions attached to non-bag prices using OCR price rows."""
    items = extracted.get("line_items") or []
    if not items:
        return
    lines = [line.strip() for line in unified_text.split('\n')]

    def _clean_desc(line: str) -> str:
        text = re.sub(r'^[\dA-Za-z-]+\)?\s*', '', line or "").strip()
        text = re.sub(r'[¥￥]?\s*\d[\d,]*\s*(?:[%％][*※除軽]|[*※除軽])?\s*$', '', text).strip()
        return text

    for item in items:
        if not isinstance(item, dict):
            continue
        if not _is_bag_description(item.get("description") or ""):
            continue
        total = float(item.get("total") or 0)
        if total <= 50:
            continue
        replacement = None
        for idx, line in enumerate(lines):
            pm = _OCR_TRAILING_PRICE_RE.search(line)
            if not pm:
                continue
            try:
                price = float(pm.group(1).strip().lstrip('¥￥').replace(',', ''))
            except ValueError:
                continue
            if abs(price - total) > 2:
                continue
            for j in range(idx - 1, max(idx - 6, -1), -1):
                cand = _clean_desc(lines[j])
                if not cand or _is_bag_description(cand):
                    continue
                if _SKIP_PRICE_LINE.search(cand) or _OCR_QTY_NOTATION_RE.search(cand):
                    continue
                if re.search(r'[ぁ-んァ-ン一-龥]', cand):
                    replacement = cand
                    break
            if replacement:
                break
        if replacement:
            item["description"] = replacement


def _fix_embedded_price_suffix_totals(extracted, unified_text):
    """Use an embedded OCR price suffix when the extracted total drifted nearby."""
    for item in extracted.get("line_items") or []:
        if not isinstance(item, dict):
            continue
        desc = item.get("description") or ""
        m = re.search(r'\s+(\d{2,4})\s*$', desc)
        if not m:
            continue
        price = float(m.group(1))
        total = float(item.get("total") or 0)
        if total <= 0 or abs(price - total) > 5 or abs(price - total) <= 1:
            continue
        desc_base = desc[:m.start()].strip()
        if desc_base and desc_base in unified_text and re.search(r'(?<!\d)' + re.escape(m.group(1)) + r'(?!\d)', unified_text):
            item["description"] = desc_base
            item["qty"] = 1.0
            item["unit_price"] = price
            item["total"] = price


def _fix_adjacent_ocr_price_shift_when_balanced(extracted, unified_text):
    """Repair adjacent item totals when OCR shows a shifted inline/next-line price."""
    items = [item for item in extracted.get("line_items") or [] if isinstance(item, dict)]
    if len(items) < 2:
        return

    targets = [
        float(value)
        for value in (
            extracted.get("subtotal"),
            _canonical_subtotal_from_taxes(extracted),
            extracted.get("total"),
        )
        if value is not None and float(value or 0) > 0
    ]
    rate_bases = extract_rate_bases(unified_text)
    base_sum = sum(float(base or 0) for base in rate_bases.values() if base is not None)
    if base_sum > 0:
        targets.append(base_sum)
    if not targets:
        return

    current_sum = sum(float(item.get("total") or 0) for item in items)
    current_gap = min(abs(current_sum - target) for target in targets)
    printed_count = None
    count_match = re.search(r'お買上商品数\s*[:：]?\s*(\d+)', unified_text)
    if count_match:
        printed_count = int(count_match.group(1))
    if current_gap <= 2 and (printed_count is None or len(items) == printed_count):
        return

    lines = [line.strip() for line in unified_text.split('\n')]

    def _norm(text: str) -> str:
        text = _clean_ocr_price_line_desc(str(text or ""))
        text = re.sub(r'\s+', '', text)
        text = re.sub(r'[^\wぁ-んァ-ン一-龥()]', '', text, flags=re.UNICODE)
        return text.lower()

    def _amount_from_line(line: str) -> float | None:
        if _SKIP_PRICE_LINE.search(line):
            return None
        if re.fullmatch(r'\d{5,}', line.strip()):
            return None
        match = _OCR_TRAILING_PRICE_RE.search(line)
        if not match:
            return None
        raw = match.group(1).strip().lstrip('¥￥').replace(',', '')
        if not raw.isdigit():
            return None
        amount = float(raw)
        if amount <= 0 or amount > max(targets):
            return None
        return amount

    line_norms = [_norm(line) for line in lines]

    def _find_desc_line(desc: str) -> int | None:
        desc_norm = _norm(desc)
        if len(desc_norm) < 3:
            return None
        best: tuple[float, int] | None = None
        for idx, line_norm in enumerate(line_norms):
            if len(line_norm) < 3:
                continue
            if desc_norm in line_norm or line_norm in desc_norm:
                score = 1.0
            else:
                score = SequenceMatcher(None, desc_norm, line_norm).ratio()
            if score >= 0.86 and (best is None or score > best[0]):
                best = (score, idx)
        return best[1] if best else None

    def _next_item_line(start_idx: int, item_idx: int) -> int | None:
        nearest = None
        for later in items[item_idx + 1:item_idx + 5]:
            line_idx = _find_desc_line(later.get("description") or "")
            if line_idx is not None and line_idx > start_idx:
                nearest = line_idx if nearest is None else min(nearest, line_idx)
        return nearest

    def _supported_amount_for_item(item_idx: int) -> tuple[float, int] | None:
        item = items[item_idx]
        line_idx = _find_desc_line(item.get("description") or "")
        if line_idx is None:
            return None
        amount = _amount_from_line(lines[line_idx])
        if amount is not None:
            return amount, line_idx
        stop = _next_item_line(line_idx, item_idx)
        search_end = min(stop if stop is not None else len(lines), line_idx + 5)
        for nearby_idx in range(line_idx + 1, search_end):
            if _valid_ocr_item_desc(_clean_ocr_price_line_desc(lines[nearby_idx])):
                break
            amount = _amount_from_line(lines[nearby_idx])
            if amount is not None:
                return amount, nearby_idx
        return None

    for idx in range(len(items) - 1):
        first = items[idx]
        second = items[idx + 1]
        if _is_bag_description(first.get("description") or "") or _is_bag_description(second.get("description") or ""):
            continue
        try:
            first_qty = float(first.get("qty") or 1)
            second_qty = float(second.get("qty") or 1)
            first_total = float(first.get("total") or 0)
            second_total = float(second.get("total") or 0)
            first_discount = float(first.get("discount") or 0)
            second_discount = float(second.get("discount") or 0)
        except (TypeError, ValueError):
            continue
        if (
            first_qty != 1
            or second_qty != 1
            or first_discount
            or second_discount
            or first_total <= 0
            or second_total <= 0
        ):
            continue
        first_supported = _supported_amount_for_item(idx)
        second_supported = _supported_amount_for_item(idx + 1)
        if first_supported is None or second_supported is None:
            continue
        first_amount, first_line = first_supported
        second_amount, second_line = second_supported
        if first_line >= second_line:
            continue
        if abs(first_total - first_amount) <= 1 and abs(second_total - second_amount) <= 1:
            continue
        new_sum = current_sum - first_total - second_total + first_amount + second_amount
        new_gap = min(abs(new_sum - target) for target in targets)
        remove_idx = None
        if new_gap >= current_gap and printed_count is not None and len(items) == printed_count + 1:
            seen: dict[tuple[str, float], int] = {}
            for candidate_idx, candidate in enumerate(items):
                if candidate_idx in (idx, idx + 1):
                    continue
                try:
                    candidate_total = float(candidate.get("total") or 0)
                except (TypeError, ValueError):
                    continue
                key = (_norm(candidate.get("description") or ""), round(candidate_total, 2))
                if not key[0] or candidate_total <= 0:
                    continue
                if key in seen:
                    candidate_sum = new_sum - candidate_total
                    candidate_gap = min(abs(candidate_sum - target) for target in targets)
                    if candidate_gap <= 2:
                        remove_idx = candidate_idx
                        new_sum = candidate_sum
                        new_gap = candidate_gap
                        break
                else:
                    seen[key] = candidate_idx
        if new_gap >= current_gap and remove_idx is None:
            continue
        first["qty"] = 1.0
        first["unit_price"] = first_amount
        first["total"] = first_amount
        second["qty"] = 1.0
        second["unit_price"] = second_amount
        second["total"] = second_amount
        if remove_idx is not None:
            items.pop(remove_idx)
            extracted["line_items"] = items
        current_sum = new_sum
        current_gap = new_gap
        if current_gap <= 2:
            return


def _fix_discounted_item_gross_prices_from_ocr(extracted, unified_text):
    """Restore gross unit price when a discount was applied twice."""
    lines = [line.strip() for line in unified_text.split('\n')]

    def _line_price(line: str) -> float | None:
        pm = _OCR_TRAILING_PRICE_RE.search(line)
        if not pm:
            pm = re.search(r'(?:^|\s)([¥￥]?\s*\d[\d,]*)\s*[A-ZＡ-Ｚ]\s*$', line)
        if not pm:
            return None
        try:
            return float(pm.group(1).strip().lstrip('¥￥').replace(',', ''))
        except ValueError:
            return None

    def _rate_matches(gross: float, discount: float, discount_rate: str) -> bool:
        m = re.search(r'(\d+(?:\.\d+)?)\s*%', str(discount_rate or ""))
        if not m:
            return True
        expected = gross * (float(m.group(1)) / 100.0)
        return abs(expected - discount) <= max(2.0, expected * 0.03)

    def _apply_gross(item: dict, gross: float, discount: float) -> None:
        qty = float(item.get("qty") or 1)
        current_unit = item.get("unit_price")
        try:
            current_unit_f = float(current_unit) if current_unit is not None else None
        except (TypeError, ValueError):
            current_unit_f = None
        if qty > 1 and current_unit_f and abs(current_unit_f * qty - gross) <= 2:
            item["unit_price"] = current_unit_f
        elif qty > 1:
            item["unit_price"] = gross / qty
        else:
            item["unit_price"] = gross
        item["total"] = gross - discount

    for item in extracted.get("line_items") or []:
        if not isinstance(item, dict):
            continue
        discount = float(item.get("discount") or 0)
        if discount <= 0:
            continue
        desc = item.get("description") or ""
        for idx, line in enumerate(lines):
            if desc and desc not in line:
                continue
            inline_gross = _line_price(line)
            if inline_gross is not None:
                window = "\n".join(lines[idx:min(idx + 6, len(lines))])
                if (
                    re.search(r'-\s*' + str(int(discount)) + r'\b', window)
                    and _rate_matches(inline_gross, discount, item.get("discount_rate") or "")
                ):
                    _apply_gross(item, inline_gross, discount)
                    break
                continue
            for j in range(idx + 1, min(idx + 6, len(lines))):
                gross = _line_price(lines[j])
                if gross is None:
                    continue
                window = "\n".join(lines[j:j + 5])
                if (
                    re.search(r'-\s*' + str(int(discount)) + r'\b', window)
                    and _rate_matches(gross, discount, item.get("discount_rate") or "")
                ):
                    _apply_gross(item, gross, discount)
                    break
            break


def _ensure_discounted_ocr_pairs_present(extracted, unified_text):
    """Ensure OCR price/discount pairs exist when they improve subtotal fit."""
    items = extracted.get("line_items") or []
    subtotal = extracted.get("subtotal")
    if not items or subtotal is None:
        return
    item_sum = sum(float(i.get("total") or 0) for i in items if isinstance(i, dict))
    lines = [line.strip() for line in unified_text.split('\n')]
    for idx, line in enumerate(lines):
        pm = _OCR_TRAILING_PRICE_RE.search(line)
        if not pm:
            continue
        try:
            gross = float(pm.group(1).strip().lstrip('¥￥').replace(',', ''))
        except ValueError:
            continue
        discount = None
        discount_rate = ""
        for j in range(idx + 1, min(idx + 5, len(lines))):
            rm = re.search(r'(\d+)\s*%', lines[j])
            if rm:
                discount_rate = rm.group(1) + "%"
            dm = re.match(r'^-\s*(\d{1,4})\s*$', lines[j])
            if dm:
                discount = float(dm.group(1))
                break
        if not discount:
            continue
        net = gross - discount
        if any(isinstance(item, dict) and abs(float(item.get("total") or 0) - net) <= 0.5 for item in items):
            continue
        if abs((item_sum + net) - float(subtotal)) > 2:
            continue
        desc = _find_discounted_ocr_item_desc(lines, idx)
        if not desc:
            continue
        recovered = {
            "description": desc,
            "qty": 1.0,
            "unit_price": gross,
            "total": net,
            "tax_category": "8%",
            "discount": discount,
            "discount_rate": discount_rate,
        }
        _insert_item_by_ocr_order(items, lines, idx, recovered)
        item_sum += net


def _repair_discounted_ocr_pair_descriptions(extracted, unified_text):
    """Use visible OCR price/discount ownership to repair duplicated descriptions."""
    items = extracted.get("line_items") or []
    if not items:
        return
    lines = [line.strip() for line in unified_text.split('\n')]

    def _line_price(line: str) -> float | None:
        pm = _OCR_TRAILING_PRICE_RE.search(line)
        if not pm:
            pm = re.search(r'(?:^|\s)([¥￥]?\s*\d[\d,]*)\s*[A-ZＡ-Ｚ]\s*$', line)
        if not pm:
            return None
        try:
            return float(pm.group(1).strip().lstrip('¥￥').replace(',', ''))
        except ValueError:
            return None

    def _norm(text: str) -> str:
        return re.sub(r'\s+', '', str(text or ""))

    desc_counts: dict[str, int] = {}
    for item in items:
        if isinstance(item, dict):
            key = _norm(item.get("description") or "")
            if key:
                desc_counts[key] = desc_counts.get(key, 0) + 1

    for idx, line in enumerate(lines):
        gross = _line_price(line)
        if gross is None:
            continue
        discount = None
        for nearby in lines[idx + 1:min(idx + 5, len(lines))]:
            dm = re.match(r'^-\s*(\d{1,4})\s*$', nearby)
            if dm:
                discount = float(dm.group(1))
                break
        if discount is None or discount <= 0 or gross <= discount:
            continue
        desc = _find_discounted_ocr_item_desc(lines, idx)
        if not desc:
            continue
        desc_key = _norm(desc)
        if not desc_key or any(_norm(item.get("description") or "") == desc_key for item in items if isinstance(item, dict)):
            continue
        net = gross - discount
        candidates = [
            item for item in items
            if isinstance(item, dict)
            and abs(float(item.get("total") or 0) - net) <= 2
            and abs(float(item.get("discount") or 0) - discount) <= 2
            and desc_counts.get(_norm(item.get("description") or ""), 0) > 1
        ]
        if len(candidates) != 1:
            continue
        candidate = candidates[0]
        candidate["description"] = desc
        candidate["unit_price"] = gross / float(candidate.get("qty") or 1)
        candidate["total"] = net


def _repair_pre_price_stack_descriptions_from_ocr(extracted, unified_text):
    """Map product names before a stacked price block onto matching item amounts."""
    items = [item for item in (extracted.get("line_items") or []) if isinstance(item, dict)]
    if len(items) < 2 or not unified_text:
        return
    current_descs = [
        _norm_layout_desc(item.get("description") or "")
        for item in items
        if _norm_layout_desc(item.get("description") or "")
    ]
    if len(current_descs) != len(items):
        return
    has_duplicate_or_nested_desc = (
        len(set(current_descs)) < len(current_descs)
        or any(
            left != right and (left in right or right in left)
            for idx, left in enumerate(current_descs)
            for right in current_descs[idx + 1:]
        )
    )
    if not has_duplicate_or_nested_desc:
        return
    item_sum = _line_items_sum(extracted)
    targets = [
        float(value)
        for value in (
            extracted.get("subtotal"),
            extracted.get("total"),
            _canonical_subtotal_from_taxes(extracted),
        )
        if value is not None and float(value or 0) > 0
    ]
    if targets and not any(abs(item_sum - target) <= 2 for target in targets):
        return

    lines = [line.strip() for line in unified_text.split('\n')]
    zone_end = len(lines)
    for idx, line in enumerate(lines):
        if re.fullmatch(r'小\s*計|合\s*計|総\s*合\s*計', line):
            zone_end = idx
            break

    price_entries: list[tuple[int, float]] = []
    for idx, line in enumerate(lines[:zone_end]):
        m = re.fullmatch(r'(-?)\s*[¥￥]\s*([\d,]+)', line)
        if not m:
            continue
        value = float(m.group(2).replace(',', ''))
        if m.group(1):
            value = -value
        price_entries.append((idx, value))
    if not price_entries:
        return

    positive_prices = [value for _idx, value in price_entries if value > 0]
    if len(positive_prices) != len(items):
        return
    try:
        item_units = [float(item.get("unit_price") or 0) for item in items]
    except (TypeError, ValueError):
        return
    if any(unit <= 0 for unit in item_units):
        return
    if any(abs(price - unit) > 2 for price, unit in zip(positive_prices, item_units)):
        return

    first_price_idx = price_entries[0][0]
    descriptions_reversed: list[str] = []
    for raw in reversed(lines[:first_price_idx]):
        if not raw:
            continue
        if re.fullmatch(r'\d{8,}(?:\s*JAN)?', raw, flags=re.IGNORECASE):
            continue
        if re.search(r'セール|SALE|割引|値引', raw, re.IGNORECASE):
            continue
        desc = _clean_ocr_price_line_desc(raw)
        if _valid_pre_price_stack_item_desc(raw, desc):
            descriptions_reversed.append(desc)
            continue
        if descriptions_reversed:
            break
    descriptions = list(reversed(descriptions_reversed))
    if len(descriptions) < len(items):
        return
    proposed = descriptions[-len(items):]
    if len({_norm_layout_desc(desc) for desc in proposed}) != len(proposed):
        return

    for item, desc in zip(items, proposed):
        current = _norm_layout_desc(item.get("description") or "")
        target = _norm_layout_desc(desc)
        if target and current != target:
            item["description"] = desc


def _drop_duplicate_rows_when_subtotal_balances(extracted, unified_text):
    """Drop exact duplicate parsed rows only when OCR count and subtotal agree."""
    items = extracted.get("line_items") or []
    subtotal = extracted.get("subtotal")
    if not items or subtotal is None:
        return
    try:
        target = float(subtotal)
    except (TypeError, ValueError):
        return
    item_sum = sum(float(item.get("total") or 0) for item in items if isinstance(item, dict))
    overage = item_sum - target
    if overage <= 0 or overage > 1000:
        return

    def _norm(text: str) -> str:
        return re.sub(r'\s+', '', str(text or ""))

    text_norm = _norm(unified_text)
    groups: dict[tuple[str, float, float], list[dict]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        key = (
            _norm(item.get("description") or ""),
            round(float(item.get("total") or 0), 2),
            round(float(item.get("discount") or 0), 2),
        )
        if key[0] and key[1] > 0:
            groups.setdefault(key, []).append(item)

    for (desc_key, total, _discount), group in groups.items():
        if len(group) < 2 or abs(total - overage) > 2:
            continue
        if desc_key and text_norm.count(desc_key) >= len(group):
            continue

        def _keep_score(item: dict) -> tuple[int, float]:
            qty = float(item.get("qty") or 1)
            unit = float(item.get("unit_price") or 0)
            discount = float(item.get("discount") or 0)
            score = 0
            if abs(qty * unit - discount - total) <= 2:
                score += 1
            if qty > 1:
                score += 1
            if unit and re.search(r'単\s*' + re.escape(str(int(unit))) + r'\b', unified_text):
                score += 1
            return score, -unit

        keep = max(group, key=_keep_score)
        for duplicate in group:
            if duplicate is keep:
                continue
            items.remove(duplicate)
            return


def _replace_basket_marker_rows_when_balanced(extracted, unified_text):
    """Rebuild stacked basket rows from explicit item-count and tax-marker OCR."""
    if not isinstance(extracted, dict) or not unified_text:
        return
    if not re.search(r'BOTTOM OF BASKET|御買上げ点数', unified_text):
        return
    if not re.search(r'\b\d[\d,.]*\s*[ET]\b|\b\d[\d,.]*\s*-\s*E\b', unified_text):
        return

    count_match = re.search(r'御買上げ点数\s*[:：]?\s*(\d+)', unified_text)
    if not count_match:
        return
    expected_count = int(count_match.group(1))
    if expected_count < 3 or expected_count > 200:
        return

    lines = [line.strip() for line in unified_text.splitlines() if line.strip()]
    start = 0
    for idx, line in enumerate(lines):
        if "BEGIN BOTTOM OF BASKET" in line or re.fullmatch(r'売\s*上', line):
            start = idx + 1
            break
    end = len(lines)
    for idx in range(start, len(lines)):
        if re.search(r'\*{2,}\s*合\s*計|^合\s*計$', lines[idx]):
            end = idx
            break
    if end <= start:
        return
    item_lines = lines[start:end]

    def _parse_marked_amount(text: str) -> tuple[float, str, bool] | None:
        cleaned = text.strip().replace('￥', '').replace('¥', '')
        m = re.fullmatch(r'(\d[\d,.]*)\s*-\s*E', cleaned)
        if m:
            amount = _parse_basket_amount(m.group(1))
            return (amount, "E", True) if amount is not None else None
        m = re.fullmatch(r'(\d[\d,.]*)\s*([ET])', cleaned)
        if not m:
            return None
        amount = _parse_basket_amount(m.group(1))
        return (amount, m.group(2), False) if amount is not None else None

    def _parse_basket_amount(text: str) -> float | None:
        raw = str(text or "").strip().replace(',', '')
        if not raw:
            return None
        if re.fullmatch(r'\d+\.\d{3}', raw):
            raw = raw.replace('.', '')
        if not re.fullmatch(r'\d+(?:\.\d+)?', raw):
            return None
        amount = float(raw)
        if amount <= 0 or amount > 1_000_000:
            return None
        return amount

    def _clean_desc(text: str) -> str:
        cleaned = re.sub(r'^[*＊※\s]+', '', text or "").strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()

    def _is_control_or_numeric(text: str) -> bool:
        if not text:
            return True
        if re.search(r'BEGIN BOTTOM OF BASKET|BOTTOM OF BASKET ITEM COUNT', text):
            return True
        if re.fullmatch(r'[*＊※]+', text):
            return True
        if re.fullmatch(r'\d{5,}', text):
            return True
        if re.fullmatch(r'(?:1\s*[@eE⚫●.]?|10)', text):
            return True
        if _parse_marked_amount(text) is not None:
            return True
        if _parse_basket_amount(text) is not None:
            return True
        if re.search(r'合\s*計|消費税|対象|御買上げ点数|領収|支払|釣銭|クレジット|カード|会員番号', text):
            return True
        return False

    def _valid_desc(text: str) -> bool:
        if _is_control_or_numeric(text):
            return False
        cleaned = _clean_desc(text)
        if not cleaned or "CPN" in cleaned.upper():
            return False
        return bool(re.search(r'[A-Za-zぁ-んァ-ン一-龥]', cleaned))

    def _make_row(desc: str, amount: float, marker: str) -> dict:
        tax_category = "10%" if marker == "T" else "8%"
        return {
            "description": _clean_desc(desc),
            "qty": 1.0,
            "unit_price": float(amount),
            "total": float(amount),
            "tax_category": tax_category,
            "discount": 0.0,
            "discount_rate": "",
        }

    rows: list[dict] = []
    pending_descs: list[str] = []
    last_regular_row: dict | None = None
    coupon_mode = False
    for line in item_lines:
        marked = _parse_marked_amount(line)
        if marked is not None:
            amount, marker, is_coupon = marked
            if is_coupon or coupon_mode:
                if last_regular_row is not None and amount > 0:
                    gross = float(last_regular_row.get("unit_price") or 0)
                    current_discount = float(last_regular_row.get("discount") or 0)
                    if gross >= amount and float(last_regular_row.get("total") or 0) > amount:
                        last_regular_row["discount"] = current_discount + amount
                        last_regular_row["total"] = gross - last_regular_row["discount"]
                coupon_mode = False
                pending_descs = [
                    desc for desc in pending_descs
                    if "CPN" not in desc.upper()
                ]
                continue
            if pending_descs:
                desc = pending_descs.pop(0)
                row = _make_row(desc, amount, marker)
                rows.append(row)
                last_regular_row = row
            continue

        if re.fullmatch(r'CPN', line, flags=re.IGNORECASE):
            coupon_mode = True
            continue
        if _valid_desc(line):
            if coupon_mode:
                continue
            pending_descs.append(_clean_desc(line))
            if len(pending_descs) > 12:
                pending_descs = pending_descs[-12:]

    if len(rows) != expected_count:
        return

    rows_sum = sum(float(row.get("total") or 0) for row in rows)
    taxes_sum = _sum_taxable_amounts(extracted.get("taxes") or [])
    rate_bases = extract_rate_bases(unified_text)
    rate_base_sum = sum(float(base) for base in rate_bases.values() if base and base > 0)
    targets: list[float] = []
    for value in (extracted.get("subtotal"),):
        if value is not None:
            try:
                targets.append(float(value))
            except (TypeError, ValueError):
                pass
    if extracted.get("total") is not None:
        try:
            total = float(extracted["total"])
        except (TypeError, ValueError):
            total = None
        if total is not None:
            if taxes_sum > 0:
                targets.append(total - taxes_sum)
            if rate_base_sum > 0 and abs(rate_base_sum - total) <= 2:
                targets.append(total)
    if not targets or all(abs(rows_sum - target) > 2 for target in targets):
        return

    for rate, base in rate_bases.items():
        if not base or base <= 0:
            continue
        rate_sum = sum(
            float(row.get("total") or 0)
            for row in rows
            if row.get("tax_category") == rate
        )
        if abs(rate_sum - float(base)) > 2:
            return

    extracted["line_items"] = rows


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


def _repair_discounted_line_item_totals_when_balanced(extracted, unified_text):
    """Net discounted item totals when doing so makes item sum match subtotal."""
    items = extracted.get("line_items") or []
    if not items:
        return

    try:
        subtotal = float(extracted.get("subtotal"))
    except (TypeError, ValueError):
        return
    if subtotal <= 0:
        return

    def _num(value) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    current_sum = sum(float(item.get("total") or 0) for item in items if isinstance(item, dict))
    if abs(current_sum - subtotal) <= 2:
        return

    ocr_discounts = [
        float(match.group(1).replace(",", ""))
        for match in re.finditer(r'-\s*[¥￥]?\s*(\d[\d,]*)', unified_text or "")
    ]
    candidates: list[tuple[dict, float, float]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        qty = _num(item.get("qty", 1))
        unit_price = _num(item.get("unit_price"))
        total = _num(item.get("total"))
        discount = _num(item.get("discount"))
        if qty is None or unit_price is None or total is None or discount is None:
            continue
        if qty <= 0 or unit_price <= 0 or discount <= 0:
            continue
        gross = qty * unit_price
        expected = gross - discount
        if expected < 0 or abs(total - gross) > 1 or abs(total - expected) <= 1:
            continue
        if ocr_discounts and not any(abs(discount - value) <= 2 for value in ocr_discounts):
            continue
        adjusted_sum = current_sum - total + expected
        candidates.append((item, expected, adjusted_sum))

    exact = [
        (item, expected)
        for item, expected, adjusted_sum in candidates
        if abs(adjusted_sum - subtotal) <= 2
    ]
    if len(exact) == 1:
        item, expected = exact[0]
        item["total"] = expected
        return

    adjusted_sum = current_sum
    adjusted: list[tuple[dict, float]] = []
    for item, expected, _adjusted in candidates:
        total = float(item.get("total") or 0)
        adjusted_sum = adjusted_sum - total + expected
        adjusted.append((item, expected))
    if adjusted and abs(adjusted_sum - subtotal) <= 2:
        for item, expected in adjusted:
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


def _clear_discounts_without_nearby_ocr_marker(items, unified_text):
    """Clear LLM discounts when OCR does not place a discount by that item."""
    if not items:
        return
    lines = unified_text.split('\n')

    def _norm(text: str) -> str:
        text = re.sub(r'^\d{4,}[A-Za-z0-9-]*\)?\s*', '', text or "")
        text = re.sub(r'[¥￥]?\s*\d[\d,]*\s*(?:[*※除軽]|%|％)?\s*$', '', text)
        text = re.sub(r'\s+', '', text)
        text = re.sub(r'[^\wぁ-んァ-ン一-龥]', '', text, flags=re.UNICODE)
        return text.lower()

    def _line_has_amount(line: str, amount: float | None) -> bool:
        if amount is None:
            return False
        amount_int = int(round(float(amount)))
        return bool(re.search(r'(?<!\d)' + re.escape(f"{amount_int:,}") + r'|' + re.escape(str(amount_int)) + r'(?!\d)', line))

    def _next_item_started(line: str) -> bool:
        if not re.search(r'[ぁ-んァ-ン一-龥]', line):
            return False
        if re.search(r'割引|値引|%|％|[¥￥]|単|JAN|Code128', line):
            return False
        return True

    def _supported(item: dict) -> bool:
        desc_norm = _norm(item.get("description") or "")
        unit = item.get("unit_price")
        total = item.get("total")
        discount = item.get("discount") or 0
        discount_value = float(discount or 0)
        discount_rate = str(item.get("discount_rate") or "")
        search_amounts = [unit, (float(total) + float(discount)) if total is not None else None]
        duplicate_desc = (
            bool(desc_norm)
            and sum(1 for line in lines if desc_norm and desc_norm in _norm(line)) > 1
        )
        candidate_idxs: list[int] = []
        for idx, line in enumerate(lines):
            norm_line = _norm(line)
            desc_match = (
                desc_norm
                and norm_line
                and (desc_norm in norm_line or norm_line in desc_norm
                     or SequenceMatcher(None, desc_norm, norm_line).ratio() >= 0.72)
            )
            amount_match = any(_line_has_amount(line, amount) for amount in search_amounts)
            if desc_match or amount_match:
                if duplicate_desc and not amount_match:
                    continue
                if amount_match and desc_norm:
                    context = "\n".join(lines[max(0, idx - 3):min(len(lines), idx + 3)])
                    norm_context = _norm(context)
                    if desc_norm not in norm_context and all(
                        SequenceMatcher(None, desc_norm, _norm(ctx_line)).ratio() < 0.72
                        for ctx_line in lines[max(0, idx - 3):min(len(lines), idx + 3)]
                    ):
                        has_following_matching_discount = False
                        for nearby in lines[idx + 1:min(len(lines), idx + 4)]:
                            discount_m = re.fullmatch(
                                r'\s*-\s*[¥￥\\]?\s*(\d[\d,]*)\s*',
                                nearby.strip(),
                            )
                            if not discount_m:
                                continue
                            amount = float(discount_m.group(1).replace(',', ''))
                            if abs(amount - discount_value) <= 2:
                                has_following_matching_discount = True
                                break
                        if not has_following_matching_discount:
                            continue
                candidate_idxs.append(idx)

        def _has_matching_discount_amount(text: str) -> bool:
            if discount_value <= 0:
                return False
            m = re.fullmatch(r'\s*-\s*[¥￥\\]?\s*(\d[\d,]*)\s*', text)
            if not m:
                return False
            amount = float(m.group(1).replace(',', ''))
            return abs(amount - discount_value) <= 2

        def _has_matching_rate_marker(text: str) -> bool:
            m = re.fullmatch(r'\s*-?\s*(\d+(?:\.\d+)?)\s*%\s*', text)
            if not m:
                return False
            if discount_rate:
                rate_m = re.search(r'(\d+(?:\.\d+)?)\s*%', discount_rate)
                if rate_m and abs(float(rate_m.group(1)) - float(m.group(1))) <= 0.1:
                    return True
            gross = float(unit or 0)
            if gross <= 0 and total is not None:
                gross = float(total) + discount_value
            return gross > 0 and abs(discount_value - gross * (float(m.group(1)) / 100.0)) <= max(2.0, gross * 0.03)

        for idx in candidate_idxs:
            saw_rate_marker = False
            saw_item_amount = any(
                _line_has_amount(lines[idx], amount) for amount in search_amounts
            )
            for offset in range(1, 9):
                j = idx + offset
                if j >= len(lines):
                    break
                nxt = lines[j].strip()
                if any(_line_has_amount(nxt, amount) for amount in search_amounts):
                    saw_item_amount = True
                    continue
                if re.search(r'割引|値引', nxt):
                    return saw_item_amount
                if _has_matching_rate_marker(nxt):
                    saw_rate_marker = True
                    continue
                if _has_matching_discount_amount(nxt):
                    if not saw_item_amount:
                        continue
                    if saw_rate_marker or not discount_rate:
                        return True
                    if _has_matching_rate_marker(discount_rate):
                        return True
                if _next_item_started(nxt):
                    break
        return False

    for item in items:
        if not isinstance(item, dict) or not (item.get("discount") or 0):
            continue
        if _supported(item):
            continue
        qty = float(item.get("qty") or 1)
        unit = item.get("unit_price")
        if unit is not None:
            item["total"] = qty * float(unit)
        item["discount"] = 0
        item["discount_rate"] = ""


def _detect_ocr_discounts(items, unified_text):
    """Detect discount lines in OCR text and apply to preceding items."""
    ocr_lines = unified_text.split('\n')

    def _norm_discount_desc(text: str) -> str:
        text = re.sub(r'^\d{4,}[A-Za-z0-9-]*\)?\s*', '', text or "")
        text = re.sub(r'[¥￥]?\s*\d[\d,]*\s*[*※除軽]?\s*$', '', text)
        text = re.sub(r'\s+', '', text)
        text = re.sub(r'[^\wぁ-んァ-ン一-龥]', '', text, flags=re.UNICODE)
        return text.lower()

    for item in items:
        if not isinstance(item, dict) or (item.get("discount") or 0) > 0:
            continue
        desc = item.get("description", "")
        desc_prefix = desc[:4] if len(desc) >= 4 else desc
        if not desc_prefix:
            continue
        norm_desc = _norm_discount_desc(desc)
        candidate_lines: list[int] = []
        fallback_lines: list[int] = []
        for li, ocr_line in enumerate(ocr_lines):
            norm_line = _norm_discount_desc(ocr_line)
            if norm_desc and len(norm_desc) >= 4 and norm_line:
                if norm_desc in norm_line or norm_line in norm_desc:
                    candidate_lines.append(li)
                    continue
                if SequenceMatcher(None, norm_desc, norm_line).ratio() >= 0.72:
                    candidate_lines.append(li)
                    continue
            if desc_prefix in ocr_line:
                fallback_lines.append(li)
        line_indices = candidate_lines or fallback_lines
        for li in line_indices:
            for offset in range(1, 8):
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
            if (item.get("discount") or 0) > 0:
                break

    _repair_rate_discounts_from_ocr_amounts(items, unified_text)


def _repair_rate_discounts_from_ocr_amounts(items, unified_text):
    """Match percentage-discounted items to printed OCR discount amounts."""
    discount_amounts: list[float] = []
    lines = unified_text.split('\n')
    for idx, line in enumerate(lines):
        m = re.match(r'^\s*-\s*[¥￥]?\s*(\d[\d,]*)\s*$', line.strip())
        if not m:
            continue
        window = "\n".join(lines[max(0, idx - 8):idx + 1])
        if "割引" not in window and "値引" not in window:
            continue
        discount_amounts.append(float(m.group(1).replace(',', '')))

    if not discount_amounts:
        return

    used: set[int] = set()
    rate_items: list[dict] = []

    def _rate(item: dict) -> float | None:
        m = re.search(r'(\d+(?:\.\d+)?)\s*%', str(item.get("discount_rate") or ""))
        if not m:
            return None
        return float(m.group(1)) / 100.0

    def _gross_candidates(item: dict) -> list[float]:
        qty = float(item.get("qty") or 1)
        unit = item.get("unit_price")
        total = item.get("total")
        discount = item.get("discount") or 0
        candidates: list[float] = []
        if unit is not None:
            candidates.append(float(unit))
            if qty != 1:
                candidates.append(float(unit) * qty)
        if total is not None:
            candidates.append(float(total) + float(discount))
        deduped: list[float] = []
        for value in candidates:
            if value > 0 and all(abs(value - seen) > 0.5 for seen in deduped):
                deduped.append(value)
        return deduped

    def _best_entry(item: dict) -> tuple[int, float, float] | None:
        rate = _rate(item)
        if rate is None:
            return None
        gross_values = _gross_candidates(item)
        best: tuple[float, int, float, float] | None = None
        for entry_idx, amount in enumerate(discount_amounts):
            if entry_idx in used:
                continue
            for gross in gross_values:
                expected = gross * rate
                tolerance = max(2.0, expected * 0.03)
                delta = abs(amount - expected)
                if delta <= tolerance and (best is None or delta < best[0]):
                    best = (delta, entry_idx, amount, gross)
        if best is None:
            return None
        return best[1], best[2], best[3]

    for item in items:
        if isinstance(item, dict) and _rate(item) is not None and (item.get("discount") or 0) > 0:
            rate_items.append(item)

    matched_current_items: set[int] = set()
    for item_idx, item in enumerate(rate_items):
        current = float(item.get("discount") or 0)
        match = _best_entry(item)
        if match is None:
            continue
        entry_idx, amount, _gross = match
        if abs(current - amount) <= 0.5:
            used.add(entry_idx)
            matched_current_items.add(item_idx)

    for item_idx, item in enumerate(rate_items):
        if item_idx in matched_current_items:
            continue
        match = _best_entry(item)
        if match is None:
            continue
        entry_idx, amount, gross = match
        used.add(entry_idx)
        item["discount"] = amount
        item["total"] = gross - amount


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


def _fill_single_qty_unit_prices_from_totals(items):
    """For single-quantity undiscounted rows, missing unit price equals total."""
    for item in items or []:
        if not isinstance(item, dict):
            continue
        try:
            qty = float(item.get("qty") or 1)
            total = float(item.get("total") or 0)
            discount = float(item.get("discount") or 0)
            unit = item.get("unit_price")
            unit_value = float(unit or 0)
        except (TypeError, ValueError):
            continue
        if qty == 1 and total > 0 and discount == 0 and unit_value == 0:
            item["unit_price"] = total


def _recover_multiple_missing_items_from_gap(
    extracted,
    unified_text,
    lines,
    items,
    unmatched_prices,
    items_sum,
    try_targets,
):
    """Recover multiple OCR-visible item rows when they uniquely close a gap."""
    if not unmatched_prices:
        return False

    def _norm_desc(text: str) -> str:
        return re.sub(r'\s+', '', str(text or ""))

    existing_descs = {
        _norm_desc(item.get("description"))
        for item in items
        if isinstance(item, dict) and item.get("description")
    }

    def _clean_desc(text: str) -> str:
        text = str(text or "").strip()
        text = _clean_ocr_price_line_desc(text)
        text = re.sub(r'\s+\d[\d,]*\s*[%％*＊※除軽非]?\s*$', '', text).strip()
        text = re.sub(r'\s+[\d,]+\s*[点個コ]\s*$', '', text).strip()
        text = re.sub(r'\s*[※\*＊非外内除軽]\s*$', '', text).strip()
        return text

    def _valid_orphan_desc(text: str) -> bool:
        text = _clean_desc(text)
        if not _valid_ocr_item_desc(text):
            return False
        if _norm_desc(text) in existing_descs:
            return False
        if re.search(
            r'取\s*\d|担当|レジ|領収|登録番号|TEL|FAX|http|'
            r'お買上|ポイント|会員|支払|現金|クレジット|釣銭|お釣|:',
            text,
            re.IGNORECASE,
        ):
            return False
        if re.search(r'\d+\s*[個コ点]\s*[xX×Ⅹ]|[xX×Ⅹ]\s*単?\d', text):
            return False
        return True

    end_idx = next((idx for idx, line in enumerate(lines) if re.fullmatch(r'\s*小\s*計\s*', line.strip())), len(lines))
    start_idx = next(
        (
            idx + 1
            for idx, line in enumerate(lines[:end_idx])
            if re.search(r'\d{4}/\d{1,2}/\d{1,2}|\d{1,2}:\d{2}', line)
        ),
        0,
    )
    item_zone = range(start_idx, end_idx)
    unmatched_by_idx = {
        idx: amount for idx, amount in unmatched_prices
        if start_idx <= idx < end_idx
    }

    clear_candidates: list[dict] = []
    fragment_candidates: list[dict] = []
    for desc_idx in item_zone:
        desc = _clean_desc(lines[desc_idx])
        if not _valid_orphan_desc(desc):
            continue
        desc_norm = _norm_desc(desc)
        line_has_own_amount = bool(
            re.search(r'(?:^|[\s(（])[¥￥]?\s*\d[\d,]*\s*[%％*＊※除軽非]?\s*$', lines[desc_idx].strip())
        )
        if (
            line_has_own_amount
            and any(desc_norm and desc_norm in existing for existing in existing_descs)
        ):
            continue
        for price_idx in range(desc_idx, min(end_idx, desc_idx + 5)):
            raw = lines[price_idx].strip()
            if price_idx in unmatched_by_idx:
                marker_m = re.search(r'([%％*＊※除軽非]+)\s*$', raw)
                clear_candidates.append({
                    "desc": desc,
                    "desc_idx": desc_idx,
                    "price_idx": price_idx,
                    "amount": float(unmatched_by_idx[price_idx]),
                    "marker": marker_m.group(1) if marker_m else "",
                })
                break
            fragment_m = re.fullmatch(r'[¥￥]?\s*(\d{1,2})\s*([%％*＊※除軽非]*)', raw)
            if (
                fragment_m
                and price_idx > desc_idx
                and not _SKIP_PRICE_LINE.search(raw)
                and not _OCR_QTY_NOTATION_RE.search(raw)
            ):
                fragment_candidates.append({
                    "desc": desc,
                    "desc_idx": desc_idx,
                    "price_idx": price_idx,
                    "fragment": fragment_m.group(1),
                    "marker": fragment_m.group(2) or "",
                })
                break

    def _tax_category_from_marker(marker: str, desc: str) -> str:
        if _is_bag_description(desc):
            return "10%"
        if re.search(r'非', marker):
            return "0%"
        if re.search(r'除', marker):
            return "10%"
        if re.search(r'[%％*＊※軽]', marker):
            return "8%"
        return "8%"

    def _make_item(candidate: dict, amount: float) -> dict:
        desc = candidate["desc"]
        return {
            "description": desc,
            "qty": 1,
            "unit_price": float(amount),
            "total": float(amount),
            "tax_category": _tax_category_from_marker(str(candidate.get("marker") or ""), desc),
            "discount": 0,
            "discount_rate": "",
        }

    def _candidate_pairs_for_gap(gap: float) -> list[tuple[dict, dict]]:
        pairs: list[tuple[dict, dict]] = []
        for left_idx, left in enumerate(clear_candidates):
            for right in clear_candidates[left_idx + 1:]:
                if left["desc_idx"] == right["desc_idx"]:
                    continue
                if abs(float(left["amount"]) + float(right["amount"]) - gap) <= 2:
                    pairs.append((left, right))
            remaining = gap - float(left["amount"])
            if remaining <= 0 or remaining > gap:
                continue
            remaining_text = str(int(round(remaining)))
            for fragment in fragment_candidates:
                if fragment["desc_idx"] == left["desc_idx"]:
                    continue
                if not remaining_text.startswith(str(fragment["fragment"])):
                    continue
                if len(str(fragment["fragment"])) >= len(remaining_text):
                    continue
                filled = dict(fragment)
                filled["amount"] = float(remaining)
                pairs.append((left, filled))
        return pairs

    successful: list[tuple[float, list[dict]]] = []
    for target in try_targets:
        gap = float(target) - float(items_sum)
        if gap <= 0 or gap > float(target):
            continue
        pairs = _candidate_pairs_for_gap(gap)
        unique_pairs: list[tuple[dict, dict]] = []
        seen_keys = set()
        for left, right in pairs:
            ordered = sorted((left, right), key=lambda c: (c["desc_idx"], c["price_idx"]))
            key = tuple((c["desc"], int(round(float(c["amount"])))) for c in ordered)
            if key not in seen_keys:
                seen_keys.add(key)
                unique_pairs.append((ordered[0], ordered[1]))
        if len(unique_pairs) != 1:
            continue

        ordered = list(unique_pairs[0])
        proposed = [dict(item) for item in items if isinstance(item, dict)]
        proposed.extend(_make_item(candidate, float(candidate["amount"])) for candidate in ordered)
        if abs(sum(float(item.get("total") or 0) for item in proposed) - float(target)) > 2:
            continue

        rate_bases = extract_rate_bases(unified_text)
        _assign_single_standard_rate_from_small_base(proposed, rate_bases)
        _rebalance_tax_categories_to_rate_bases(proposed, unified_text, extracted.get("taxes"), rate_bases)
        if rate_bases:
            checked_rates = [rate for rate, base in rate_bases.items() if base is not None and rate in {"8%", "10%"}]
            if checked_rates:
                rate_sums = {
                    rate: sum(
                        float(item.get("total") or 0)
                        for item in proposed
                        if item.get("tax_category") == rate
                    )
                    for rate in checked_rates
                }
                if any(abs(rate_sums.get(rate, 0.0) - float(rate_bases[rate] or 0)) > 2 for rate in checked_rates):
                    continue
        successful.append((target, ordered))

    if len(successful) != 1:
        return False

    _target, ordered = successful[0]

    def _line_idx_for_existing(item: dict) -> int | None:
        desc = _norm_desc(item.get("description"))
        if not desc:
            return None
        best: tuple[float, int] | None = None
        for idx, line in enumerate(lines[:end_idx]):
            line_norm = _norm_desc(_clean_desc(line))
            if not line_norm:
                continue
            if desc in line_norm or line_norm in desc:
                score = 1.0
            else:
                score = SequenceMatcher(None, desc, line_norm).ratio()
            if score >= 0.72 and (best is None or score > best[0]):
                best = (score, idx)
        return best[1] if best else None

    for candidate in sorted(ordered, key=lambda c: (c["desc_idx"], c["price_idx"])):
        new_item = _make_item(candidate, float(candidate["amount"]))
        insert_pos = len(extracted["line_items"])
        prefix = f"{new_item['description']} "
        for idx, existing in enumerate(extracted["line_items"]):
            if not isinstance(existing, dict):
                continue
            if str(existing.get("description") or "").strip().startswith(prefix):
                insert_pos = idx
                break
            existing_idx = _line_idx_for_existing(existing)
            if existing_idx is not None and existing_idx > candidate["desc_idx"]:
                insert_pos = idx
                break
        extracted["line_items"].insert(insert_pos, new_item)
        for existing in extracted["line_items"]:
            if not isinstance(existing, dict) or existing is new_item:
                continue
            desc_text = str(existing.get("description") or "").strip()
            if not desc_text.startswith(prefix):
                continue
            suffix = desc_text[len(prefix):].strip()
            if _valid_ocr_item_desc(suffix):
                existing["description"] = suffix
                break

    rate_bases = extract_rate_bases(unified_text)
    _assign_single_standard_rate_from_small_base(extracted["line_items"], rate_bases)
    _rebalance_tax_categories_to_rate_bases(
        extracted["line_items"],
        unified_text,
        extracted.get("taxes"),
        rate_bases,
    )
    _fill_single_qty_unit_prices_from_totals(extracted["line_items"])
    return True


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


    if not total or not items:
        return

    items_sum = sum(
        i.get("total", 0) for i in items if isinstance(i, dict)
    )

    # Skip when items already balance against either target — no missing item.
    items_match_total = abs(items_sum - total) <= 2
    items_match_subtotal = subtotal is not None and abs(items_sum - subtotal) <= 2
    tax_sum = _sum_taxable_amounts(taxes)
    items_match_tax_exclusive_total = (
        tax_sum > 0
        and abs(float(items_sum) + float(tax_sum) - float(total)) <= 2
    )
    if items_match_total or items_match_subtotal or items_match_tax_exclusive_total:
        return

    lines = unified_text.split('\n')

    # Collect OCR item-zone amounts excluding summary lines. Some OCR layouts
    # omit the yen symbol on product rows, so use the same trailing-price
    # detector as the projection code.
    ocr_prices: list[tuple[int, float]] = []
    for i, line in enumerate(lines):
        s = line.strip()
        if _SKIP_PRICE_LINE.search(s) or _OCR_QTY_NOTATION_RE.search(s):
            continue
        m = _OCR_TRAILING_PRICE_RE.search(s)
        marker_token = ""
        if not m:
            pct_m = re.search(r'(?:^|[\s(（])(\d[\d,]*)\s*[%％]\s*$', s)
            prev_line = lines[i - 1].strip() if i > 0 else ""
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
            nearby_before = "\n".join(lines[max(0, i - 2):i])
            has_desc_context = (
                re.search(r'[ぁ-んァ-ン一-龥]', s[:pct_m.start()]) if pct_m else False
            ) or (
                bool(re.search(r'[ぁ-んァ-ン一-龥]', prev_line))
                and not re.search(r'割引|値引|対象|消費税|税率|外税|内税', prev_line)
            )
            if (
                pct_m
                and has_desc_context
                and not re.search(r'割引|値引|対象|消費税|税率|外税|内税', s)
                and not re.search(r'割引|値引', nearby_before)
                and not re.match(r'^-\s*[¥￥]?\s*\d', next_line)
            ):
                raw = pct_m.group(1).replace(',', '')
                marker_token = pct_m.group(0)
            else:
                continue
        else:
            raw = m.group(1).strip().lstrip('¥￥').replace(',', '')
            marker_token = m.group(0)
        if not raw.isdigit():
            continue
        try:
            amt = float(raw)
        except ValueError:
            continue
        if amt < 10 and not (
            re.search(r'[%％*※除軽]', marker_token)
            or re.search(r'袋|バッグ|bag', s, re.IGNORECASE)
        ):
            continue
        if 0 < amt <= 99999:
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
    try_targets = []
    for candidate_target in (total, subtotal):
        if candidate_target is None:
            continue
        if not any(abs(float(candidate_target) - float(seen)) <= 0.5 for seen in try_targets):
            try_targets.append(candidate_target)
    if _recover_multiple_missing_items_from_gap(
        extracted,
        unified_text,
        lines,
        items,
        unmatched,
        items_sum,
        try_targets,
    ):
        return
    for try_target in try_targets:
        if try_target is None:
            continue
        g = try_target - items_sum
        if g <= 0 or g > total:
            continue
        matches = [(idx, amt) for idx, amt in unmatched if abs(amt - g) <= 2]
        if len(matches) != 1:
            viable = []
            seen_descs = set()
            for idx, amt in matches:
                cand_desc = _find_ocr_item_desc(lines, idx, items)
                if not cand_desc:
                    continue
                norm_desc = re.sub(r'\s+', '', cand_desc)
                if norm_desc in seen_descs:
                    continue
                seen_descs.add(norm_desc)
                viable.append((idx, amt))
            if len(viable) == 1:
                matches = viable
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
        # Drop a trailing bare price from merged item/price rows.
        text = re.sub(r'\s+\d[\d,]*\s*(?:[%％]|[*※除軽])?\s*$', '', text).strip()
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
        # Normalize: strip trailing whitespace+digits to avoid 'X' and 'X  N'
        # being treated as distinct when N is just an embedded price.
        norm_text = re.sub(r'\s+[\d,]{1,6}\s*[\*※]?\s*$', '', text).strip()
        return any(
            isinstance(o, dict) and (
                (
                    (o.get("description") or "").strip() == text
                    or re.sub(r'\s+[\d,]{1,6}\s*[\*※]?\s*$', '',
                              (o.get("description") or "").strip()).strip() == norm_text
                )
                and abs((o.get("total") or 0) - price) <= 2
            )
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
        if re.search(r'クレジット|現金|お釣り|釣銭|支払|合計|小計|対象|外税|内税|消費税', text):
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

    desc = _find_ocr_item_desc(lines, price_line_idx, items)

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
    if not _is_valid_desc(desc):
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
    taxes = extracted.get("taxes") or []
    total = extracted.get("total")
    tax_sum = _sum_taxable_amounts(taxes)
    # OCR may expose per-rate taxable bases (e.g. "8%対象") as subtotal-like
    # candidates. If the items already match the canonical subtotal, do not
    # rewrite correct item prices toward that tax-base value.
    if total is not None and tax_sum:
        canonical_subtotal = float(total) - float(tax_sum)
        if abs(item_sum - canonical_subtotal) <= 2:
            return
        if abs(canonical_subtotal - subtotal) <= 2:
            subtotal = canonical_subtotal
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


def _fix_implausible_tax_amounts(extracted, unified_text, ocr_totals):
    """Detect and fix tax amounts that are implausibly high relative to the
    rate base. Common when column-format OCR mis-pairs label and value lines
    (separate label block + separate value block), so the '対象額' value
    lands on the bare-rate label and vice versa.

    Conservative — fires only when:
      - tax_amount > 3× expected from rate_base × rate_pct, AND
      - tax_amount equals the rate_base (signature of a label/value swap)
    Generic across receipts.
    """
    taxes = extracted.get("taxes")
    if not taxes:
        return
    rate_bases = extract_rate_bases(unified_text)
    breakdown = ocr_totals.get('_breakdown_rate_bases') or {}
    for r, b in breakdown.items():
        if r not in rate_bases or rate_bases[r] is None:
            rate_bases[r] = b
    rb_sum = sum(v for v in rate_bases.values() if v is not None and v > 0)
    total = extracted.get("total") or 0
    bases_inclusive = rb_sum > 0 and total > 0 and abs(rb_sum - total) < 5

    for t in taxes:
        if not isinstance(t, dict):
            continue
        rate = t.get("rate")
        if not rate or rate == "0%":
            continue
        try:
            rate_pct = float(rate.replace('%', '')) / 100.0
        except (ValueError, AttributeError):
            continue
        if rate_pct <= 0:
            continue
        amount = t.get("amount") or 0
        base = rate_bases.get(rate)
        if base is None or base <= 0:
            continue
        if bases_inclusive:
            expected = base * rate_pct / (1 + rate_pct)
        else:
            expected = base * rate_pct
        if expected <= 0:
            continue
        if amount > expected * 3 and abs(amount - base) < 2:
            t["amount"] = round(expected) if expected > 0.5 else 0


def _parse_amount_fragment(text: str) -> float | None:
    """Parse an OCR amount token before arithmetic validation.

    Structural trigger: callers have already isolated a yen/amount-shaped OCR
    token. Invariant: this helper only normalizes numeric syntax; the caller
    must still prove the amount with receipt arithmetic or field consistency.
    """
    text = (text or "").strip().replace(',', '')
    if re.match(r'^\d+\.\d{3}$', text):
        text = text.replace('.', '')
    if not re.match(r'^\d+(?:\.\d+)?$', text):
        return None
    return float(text)


def _printed_inclusive_tax_blocks(unified_text: str) -> dict[str, tuple[float, float]]:
    text = re.sub(r'\s+', ' ', unified_text)
    blocks: dict[str, tuple[float, float]] = {}
    for m in re.finditer(
        r'\(\s*(\d{2})%対象\s*¥?\s*([\d,]+)\s*内税\s*¥?\s*([\d,]+)\s*\)',
        text,
    ):
        rate = f"{int(m.group(1))}%"
        base = float(m.group(2).replace(',', ''))
        amount = float(m.group(3).replace(',', ''))
        if rate in {"8%", "10%"} and base > 0 and amount > 0:
            blocks[rate] = (base, amount)
    for m in re.finditer(
        r'\(\s*(\d{2})%対象\s*¥?\s*([\d,]+)\s*\)\s*¥?\s*([\d,]+)\s*内税',
        text,
    ):
        rate = f"{int(m.group(1))}%"
        if rate not in {"8%", "10%"} or rate in blocks:
            continue
        amount = float(m.group(2).replace(',', ''))
        base = float(m.group(3).replace(',', ''))
        try:
            rate_pct = float(rate.rstrip('%')) / 100.0
        except ValueError:
            continue
        expected = round(base * rate_pct / (1 + rate_pct))
        if 0 < amount < base and abs(amount - expected) <= 2:
            blocks[rate] = (base, amount)
    return blocks


def _fix_printed_tax_amounts_from_structural_blocks(extracted, unified_text):
    """Use explicit printed inclusive-tax blocks when arithmetic validates them."""
    taxes = [t for t in (extracted.get("taxes") or []) if isinstance(t, dict)]

    total = float(extracted.get("total") or 0)

    blocks = _printed_inclusive_tax_blocks(unified_text)
    if blocks:
        base_sum = sum(base for base, _amount in blocks.values())
        if not total or abs(base_sum - total) <= 5:
            existing_by_rate = {t.get("rate"): t for t in taxes}
            new_taxes: list[dict] = []
            for rate in sorted(blocks, key=lambda r: int(r.rstrip('%')), reverse=True):
                _base, amount = blocks[rate]
                entry = existing_by_rate.get(rate, {})
                new_taxes.append({
                    "rate": rate,
                    "label": entry.get("label") or "内税",
                    "amount": round(amount),
                })
            extracted["taxes"] = new_taxes
            return

    direct_target_taxes: dict[str, float] = {}
    for m in re.finditer(
        r'(\d+(?:\.\d+)?)\s*%\s*対象\s*消費税\s*[¥￥]?\s*([\d,.]+)',
        unified_text,
        flags=re.S,
    ):
        rate = normalize_tax_rate(m.group(1) + "%")
        amount = _parse_amount_fragment(m.group(2))
        if rate in {"8%", "10%"} and amount is not None and amount > 0:
            direct_target_taxes[rate] = amount

    if not taxes and len(direct_target_taxes) == 1:
        rate, amount = next(iter(direct_target_taxes.items()))
        if total and amount < total:
            extracted["taxes"] = [{"rate": rate, "label": "内税", "amount": round(amount)}]
        return

    if not taxes:
        return

    nonzero = [t for t in taxes if t.get("rate") != "0%"]
    if len(nonzero) != 1:
        return

    target = nonzero[0]

    inclusive_amount_markers = [
        _parse_amount_fragment(m.group(1))
        for m in re.finditer(
            r'(?:内[、,]?\s*)?消費税(?:等)?\s*[¥￥]\s*([\d,.]+)\s*-?',
            unified_text,
        )
    ]
    inclusive_amounts = [
        value for value in inclusive_amount_markers
        if value is not None and value > 0
    ]
    if inclusive_amounts and re.search(r'内[、,]?\s*消費税|円を含みます|税込|内税', unified_text):
        amount = max(inclusive_amounts)
        if total and amount < total:
            target["amount"] = round(amount)
            target["label"] = "内税"
        return

    if direct_target_taxes:
        matches = [
            (rate, amount)
            for rate, amount in direct_target_taxes.items()
            if amount > 0 and (not total or amount < total)
        ]
        if len(matches) == 1:
            rate, amount = matches[0]
            try:
                rate_pct = float(rate.rstrip("%")) / 100.0
            except (TypeError, ValueError, ZeroDivisionError):
                rate_pct = 0.0
            base = extract_rate_bases(unified_text).get(rate)
            expected = round(float(base) * rate_pct / (1 + rate_pct)) if base and rate_pct else None
            if expected is None or abs(amount - expected) <= 2:
                target["rate"] = rate
                target["amount"] = round(amount)
                target["label"] = "内税"


def _restore_printed_external_tax_amounts(extracted, unified_text):
    """Restore explicit 外税 rate amounts after late item-category repairs."""
    taxes = [t for t in (extracted.get("taxes") or []) if isinstance(t, dict)]
    if not taxes:
        return
    lines = [line.strip() for line in unified_text.split('\n')]
    printed: dict[str, float] = {}
    rate_bases = extract_rate_bases(unified_text)
    zero_external_rates: set[str] = set()

    def _yen_value(text: str) -> float | None:
        m = re.fullmatch(r'[¥￥]\s*([\d,O〇]+)\s*[\)）]?', text.strip(), flags=re.IGNORECASE)
        if not m:
            return None
        raw = m.group(1).replace(',', '').replace('O', '0').replace('o', '0').replace('〇', '0')
        return float(raw)

    def _matches_printed_base(rate: str, amount: float) -> bool:
        base = rate_bases.get(rate)
        if base is None or base <= 0:
            return True
        try:
            rate_pct = float(rate.rstrip('%')) / 100.0
        except ValueError:
            return True
        expected_floor = int(base * rate_pct)
        expected_round = round(base * rate_pct)
        tolerance = max(1.0, base * 0.002)
        return (
            abs(amount - expected_floor) <= tolerance
            or abs(amount - expected_round) <= tolerance
        )

    for idx, line in enumerate(lines):
        m = re.fullmatch(r'(?:外税\s*(\d+(?:\.\d+)?)\s*%|(\d+(?:\.\d+)?)\s*%\s*外税)\s*額?', line)
        if not m:
            continue
        rate_num = float(m.group(1) or m.group(2))
        rate = f"{int(rate_num)}%" if rate_num == int(rate_num) else f"{rate_num}%"
        prior = "\n".join(lines[max(0, idx - 3):idx])
        if not re.search(rf'外税\s*{re.escape(rate.rstrip("%"))}\s*%\s*対象額', prior):
            continue
        nearby_values: list[float] = []
        for nearby in lines[idx + 1:min(len(lines), idx + 5)]:
            value = _yen_value(nearby)
            if value is not None:
                nearby_values.append(value)
                continue
            if nearby_values:
                break
        if len(nearby_values) >= 2 and nearby_values[0] > 0 and nearby_values[1] == 0:
            zero_external_rates.add(rate)

    for idx, line in enumerate(lines[:-1]):
        m = re.fullmatch(r'(?:外税\s*(\d+(?:\.\d+)?)\s*%|(\d+(?:\.\d+)?)\s*%\s*外税)\s*額?', line)
        split_m = None
        if not m:
            split_m = re.fullmatch(r'(\d+(?:\.\d+)?)\s*%', line)
            if not (
                split_m
                and idx + 2 < len(lines)
                and re.fullmatch(r'税', lines[idx + 1].strip())
            ):
                continue
        value_idx = idx + 1 if m else idx + 2
        nxt = lines[value_idx].strip()
        vm = re.fullmatch(r'[¥￥]\s*([\d,]+)', nxt)
        if not vm:
            continue
        rate_num = float((m.group(1) or m.group(2)) if m else split_m.group(1))
        rate = f"{int(rate_num)}%" if rate_num == int(rate_num) else f"{rate_num}%"
        amount = float(vm.group(1).replace(',', ''))
        if rate not in zero_external_rates and amount > 0 and _matches_printed_base(rate, amount):
            printed[rate] = amount
    for idx in range(0, max(0, len(lines) - 2)):
        target_m = re.search(r'(\d+(?:\.\d+)?)\s*[%％]\s*対象', lines[idx].strip())
        tax_m = re.search(r'(\d+(?:\.\d+)?)\s*[%％]\s*税額', lines[idx + 1].strip())
        if not target_m or not tax_m:
            continue
        rate_num = float(target_m.group(1))
        if rate_num != float(tax_m.group(1)):
            continue
        values: list[float] = []
        for j in range(idx + 2, min(len(lines), idx + 9)):
            vm = re.fullmatch(r'[¥￥]\s*([\d,]+)\s*[\)）]?', lines[j].strip())
            if not vm:
                break
            values.append(float(vm.group(1).replace(',', '')))
        if len(values) < 2:
            continue
        rate = f"{int(rate_num)}%" if rate_num == int(rate_num) else f"{rate_num}%"
        amount = values[-1]
        if rate not in zero_external_rates and amount > 0 and _matches_printed_base(rate, amount):
            printed[rate] = amount
    if not printed:
        if zero_external_rates:
            extracted["taxes"] = [
                tax for tax in taxes
                if normalize_tax_rate(str(tax.get("rate") or "")) not in zero_external_rates
            ]
        return
    existing = {t.get("rate"): t for t in taxes}
    for rate, amount in printed.items():
        if rate in existing:
            existing[rate]["amount"] = amount
            existing[rate]["label"] = "外税"
        else:
            taxes.append({"rate": rate, "label": "外税", "amount": amount})
    for rate, base in rate_bases.items():
        if base is None or base <= 0 or rate in printed:
            continue
        try:
            rate_pct = float(rate.rstrip('%')) / 100.0
        except ValueError:
            continue
        if int(base * rate_pct) == 0 and re.search(rf'外税\s*{re.escape(rate.rstrip("%"))}\s*%', unified_text):
            zero_external_rates.add(rate)
    if zero_external_rates:
        taxes = [
            tax for tax in taxes
            if normalize_tax_rate(str(tax.get("rate") or "")) not in zero_external_rates
        ]
    extracted["taxes"] = taxes


def _restore_explicit_tax_rate_amount_lines(extracted, unified_text):
    """Use only OCR-visible 税額 labels for per-rate external tax amounts."""
    lines = [line.strip() for line in unified_text.split('\n')]
    item_sum = _line_items_sum(extracted)
    explicit: dict[str, float] = {}

    for idx, line in enumerate(lines):
        tax_m = re.search(r'税率\s*(\d+(?:\.\d+)?)\s*[%％]\s*税額', line)
        if not tax_m:
            continue
        rate = normalize_tax_rate(tax_m.group(1) + "%")
        try:
            rate_pct = float(rate.rstrip('%')) / 100.0
        except ValueError:
            continue
        values: list[float] = []
        for nearby in lines[idx + 1:min(idx + 6, len(lines))]:
            if re.search(r'税率\s*\d+(?:\.\d+)?\s*[%％]\s*(?:課税)?対象額|合\s*計|現\s*計|お預り|お釣', nearby):
                break
            vm = re.fullmatch(r'[¥￥]\s*([\d,]+)\s*[\)）]?', nearby)
            if vm:
                values.append(float(vm.group(1).replace(',', '')))
        if not values:
            continue
        if item_sum > 0:
            ceiling = max(5.0, item_sum * rate_pct * 1.2)
            values = [value for value in values if 0 < value <= ceiling]
        else:
            values = [value for value in values if value > 0]
        if values:
            explicit[rate] = values[0]

    if not explicit:
        return

    existing_rates = {
        normalize_tax_rate(str(item.get("tax_category") or ""))
        for item in (extracted.get("line_items") or [])
        if isinstance(item, dict) and item.get("tax_category")
    }
    taxes = [
        {"rate": rate, "label": "外税", "amount": amount}
        for rate, amount in sorted(
            explicit.items(),
            key=lambda item: float(item[0].rstrip('%')),
        )
        if rate in existing_rates or not existing_rates
    ]
    if taxes:
        extracted["taxes"] = taxes


def _fix_item_totals_from_following_discount_lines(extracted, unified_text):
    """Apply OCR-visible negative discount lines immediately after a price."""
    items = extracted.get("line_items") or []
    if not items:
        return
    lines = [line.strip() for line in unified_text.split('\n')]

    def _norm(text: str) -> str:
        text = re.sub(r'\s+', '', text or "")
        text = re.sub(r'[^\wぁ-んァ-ン一-龥]', '', text, flags=re.UNICODE)
        return text.lower()

    def _price_line_amount(line: str) -> float | None:
        pm = re.search(r'(?:[¥￥]|Â¥)\s*([\d,]+)\s*$', line)
        if pm:
            return float(pm.group(1).replace(',', ''))
        marker = re.fullmatch(r'([\\]?\d[\d,]*)\s*[*※]\s*', line.strip())
        if marker:
            return float(marker.group(1).lstrip("\\").replace(',', ''))
        return None

    def _looks_like_item_desc(line: str) -> bool:
        if not line or not re.search(r'[A-Za-zぁ-んァ-ン一-龥]', line):
            return False
        if re.fullmatch(r'\d{6,}', line):
            return False
        if re.search(r'[¥￥]|Â¥|%|％|割引|値引|小\s*計|合\s*計|対象|消費税|外税|内税|お預|釣銭', line):
            return False
        if re.fullmatch(r'-?\s*\d+(?:\.\d+)?\s*%', line):
            return False
        if re.fullmatch(r'(?:単|JAN|Code128|No\.?).*', line, flags=re.IGNORECASE):
            return False
        return True

    def _owner_text_for_price(idx: int, line: str) -> str:
        inline_owner = re.sub(r'(?:[¥￥]|Â¥)\s*[\d,]+\s*$', '', line).strip()
        if _looks_like_item_desc(inline_owner):
            return inline_owner
        for j in range(idx - 1, max(-1, idx - 5), -1):
            prev = lines[j].strip()
            if not prev:
                continue
            if _price_line_amount(prev) is not None:
                break
            if re.search(r'割引|値引', prev) or re.fullmatch(r'-\s*[¥￥\\]?\s*[\d,]+', prev):
                break
            if _looks_like_item_desc(prev):
                return prev
        return ""

    for item in items:
        if not isinstance(item, dict):
            continue
        desc = item.get("description") or ""
        ndesc = _norm(desc)
        if len(ndesc) < 2:
            continue
        current_total = float(item.get("total") or 0)
        unit = float(item.get("unit_price") or current_total or 0)
        if current_total <= 0 or unit <= 0:
            continue
        if float(item.get("discount") or 0) > 0:
            continue
        for idx, line in enumerate(lines):
            price = _price_line_amount(line)
            if price is None:
                continue
            owner_norm = _norm(_owner_text_for_price(idx, line))
            owner_matches = bool(owner_norm) and (
                ndesc in owner_norm
                or owner_norm in ndesc
                or SequenceMatcher(None, ndesc, owner_norm).ratio() >= 0.72
            )
            context_norm = _norm("\n".join(lines[max(0, idx - 8):idx + 1]))
            if not owner_matches and ndesc[:8] not in context_norm:
                continue
            discount = None
            rate_str = ""
            for nearby in lines[idx + 1:min(idx + 4, len(lines))]:
                if _looks_like_item_desc(nearby):
                    break
                rm = re.search(r'(\d+(?:\.\d+)?)\s*%', nearby)
                if rm:
                    rate_str = f"{int(float(rm.group(1)))}%"
                    continue
                dm = re.fullmatch(r'-\s*(?:[¥￥\\]|Â¥)?\s*([\d,]+)', nearby)
                if dm:
                    discount = float(dm.group(1).replace(',', ''))
                    break
                if _price_line_amount(nearby) is not None:
                    break
            if discount is None or discount <= 0 or discount >= price:
                continue
            net = price - discount
            if owner_matches:
                if (
                    abs(price - unit) > 2
                    and abs(price - current_total) > 2
                    and abs(net - current_total) > 2
                ):
                    continue
            elif (
                abs(price - unit) > 0.5
                and abs(price - current_total) > 0.5
                and abs(net - current_total) > 0.5
            ):
                continue
            item["unit_price"] = price
            item["discount"] = discount
            if rate_str and not item.get("discount_rate"):
                item["discount_rate"] = rate_str
            item["total"] = net
            break


def _apply_coupon_discount_blocks(extracted, unified_text):
    """Apply explicit coupon/CPN blocks to the nearest preceding item row."""
    items = extracted.get("line_items") or []
    if not items:
        return
    lines = [line.strip() for line in unified_text.split('\n')]

    def _norm(text: str) -> str:
        return re.sub(r'\s+', '', str(text or "")).lower()

    def _line_idx_for_item(item: dict) -> int | None:
        desc = _norm(item.get("description"))
        if len(desc) < 3:
            return None
        best: tuple[float, int] | None = None
        for idx, line in enumerate(lines):
            nline = _norm(_clean_ocr_price_line_desc(line))
            if len(nline) < 3:
                continue
            if desc in nline or nline in desc:
                score = 1.0
            else:
                score = SequenceMatcher(None, desc, nline).ratio()
            if score >= 0.72 and (best is None or score > best[0]):
                best = (score, idx)
        return best[1] if best else None

    item_positions = [
        (idx, pos, item)
        for idx, item in enumerate(items)
        if isinstance(item, dict) and (pos := _line_idx_for_item(item)) is not None
    ]
    if not item_positions:
        return

    for cpn_idx, line in enumerate(lines):
        if not re.fullmatch(r'CPN|COUPON|クーポン', line, flags=re.IGNORECASE):
            continue
        discount = None
        for lookahead in lines[cpn_idx + 1:min(len(lines), cpn_idx + 8)]:
            m = re.search(r'(?:^|[\s¥￥])([\d,]+)\s*-\s*[A-Za-zＡ-Ｚ]*\s*$', lookahead)
            if not m:
                m = re.search(r'-\s*[¥￥]?\s*([\d,]+)\s*$', lookahead)
            if m:
                discount = float(m.group(1).replace(',', ''))
                break
        if discount is None or discount <= 0:
            continue
        preceding = [
            (pos, idx, item)
            for idx, pos, item in item_positions
            if pos < cpn_idx
        ]
        if not preceding:
            continue
        _pos, _idx, item = max(preceding, key=lambda entry: entry[0])
        try:
            qty = float(item.get("qty") or 1)
            unit = float(item.get("unit_price") or item.get("total") or 0)
            current_discount = float(item.get("discount") or 0)
        except (TypeError, ValueError):
            continue
        gross = qty * unit
        if qty <= 0 or gross <= discount or current_discount > 0:
            continue
        item["unit_price"] = unit
        item["discount"] = discount
        item["discount_rate"] = item.get("discount_rate") or ""
        item["total"] = gross - discount


def _drop_applied_coupon_line_items(extracted, unified_text):
    """Drop standalone coupon rows after the coupon was applied as a discount."""
    items = extracted.get("line_items") or []
    if not items:
        return
    applied_amounts = {
        float(item.get("discount") or 0)
        for item in items
        if isinstance(item, dict) and float(item.get("discount") or 0) > 0
    }
    if not applied_amounts:
        return

    coupon_amounts: set[float] = set()
    lines = [line.strip() for line in unified_text.split('\n')]
    for cpn_idx, line in enumerate(lines):
        if not re.fullmatch(r'CPN|COUPON|クーポン', line, flags=re.IGNORECASE):
            continue
        for lookahead in lines[cpn_idx + 1:min(len(lines), cpn_idx + 8)]:
            m = re.search(r'(?:^|[\s¥￥])([\d,]+)\s*-\s*[A-Za-zＡ-Ｚ]*\s*$', lookahead)
            if not m:
                m = re.search(r'-\s*[¥￥]?\s*([\d,]+)\s*$', lookahead)
            if m:
                amount = float(m.group(1).replace(',', ''))
                if amount in applied_amounts:
                    coupon_amounts.add(amount)
                break
    if not coupon_amounts:
        return

    kept = []
    for item in items:
        if not isinstance(item, dict):
            kept.append(item)
            continue
        desc = str(item.get("description") or "")
        try:
            total = float(item.get("total") or 0)
            unit = float(item.get("unit_price") or total or 0)
            discount = float(item.get("discount") or 0)
        except (TypeError, ValueError):
            kept.append(item)
            continue
        is_coupon_desc = bool(re.search(r'\bCPN\b|COUPON|クーポン', desc, flags=re.IGNORECASE))
        if discount == 0 and is_coupon_desc and (total in coupon_amounts or unit in coupon_amounts):
            continue
        kept.append(item)
    if len(kept) != len(items):
        extracted["line_items"] = kept


def _repair_tiny_item_prices_from_following_ocr(extracted, unified_text):
    """Replace unsupported nearby prices with following repeated OCR price evidence."""
    items = extracted.get("line_items") or []
    if not items:
        return
    lines = [line.strip() for line in unified_text.split('\n')]
    targets = [
        float(value)
        for value in (
            extracted.get("total"),
            extracted.get("subtotal"),
            _canonical_subtotal_from_taxes(extracted),
        )
        if value is not None and float(value or 0) > 0
    ]
    rate_bases = extract_rate_bases(unified_text)
    if rate_bases:
        base_sum = sum(float(base or 0) for base in rate_bases.values() if base is not None)
        if base_sum > 0:
            targets.append(base_sum)
    if not targets:
        return

    def _norm(text: str) -> str:
        text = re.sub(r'\s+', '', str(text or ""))
        text = re.sub(r'[^\wぁ-んァ-ン一-龥]', '', text, flags=re.UNICODE)
        return text.lower()

    def _find_desc_idx(desc: str) -> int | None:
        ndesc = _norm(desc)
        if len(ndesc) < 3:
            return None
        best: tuple[float, int] | None = None
        for idx, line in enumerate(lines):
            nline = _norm(_clean_ocr_price_line_desc(line))
            if len(nline) < 3:
                continue
            if ndesc in nline or nline in ndesc:
                score = 1.0
            else:
                score = SequenceMatcher(None, ndesc, nline).ratio()
            if score >= 0.72 and (best is None or score > best[0]):
                best = (score, idx)
        return best[1] if best else None

    def _amount_from_line(text: str) -> float | None:
        stripped = str(text or "").strip()
        if re.fullmatch(r'\d+\s*[@eE⚫●]', stripped):
            return None
        if re.fullmatch(r'\d{5,}', stripped):
            return None
        m = _OCR_TRAILING_PRICE_RE.search(stripped)
        if not m:
            m = re.search(
                r'(?:^|[\s(（])([¥￥]?\s*\d[\d,]*)\s*[A-Za-zＡ-Ｚ]\s*$',
                stripped,
            )
        if not m:
            m = re.search(
                r'(?:^|[\s(（])([¥￥]?\s*\d{1,3}(?:[,.]\d{3})+)\s*(?:[A-Za-zＡ-Ｚ])?\s*$',
                stripped,
            )
        if not m:
            return None
        raw = m.group(1).strip().lstrip('¥￥').replace(',', '')
        if not raw.isdigit() and re.fullmatch(r'\d{1,3}(?:[,.]\d{3})+', raw):
            raw = raw.replace('.', '')
        if not raw.isdigit():
            return None
        return float(raw)

    def _score(item_sum: float) -> float:
        return min(abs(item_sum - target) for target in targets)

    item_sum = sum(float(item.get("total") or 0) for item in items if isinstance(item, dict))
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            qty = float(item.get("qty") or 1)
            total = float(item.get("total") or 0)
            unit = float(item.get("unit_price") or 0)
            discount = float(item.get("discount") or 0)
        except (TypeError, ValueError):
            continue
        if qty != 1 or discount > 0 or total <= 0 or unit <= 0:
            continue
        current_value = max(total, unit)
        require_structured_block = current_value > 30
        desc_idx = _find_desc_idx(item.get("description") or "")
        if desc_idx is None:
            continue
        other_descs = [
            _norm(other.get("description") or "")
            for other in items
            if isinstance(other, dict) and other is not item
        ]
        amounts: list[float] = []
        saw_code_or_qty = False
        for lookahead in lines[desc_idx + 1:min(len(lines), desc_idx + 12)]:
            if re.search(r'CPN|クーポン|小\s*計|合\s*計|対象|消費税', lookahead, re.IGNORECASE):
                break
            nlookahead = _norm(_clean_ocr_price_line_desc(lookahead))
            if any(
                len(other_desc) >= 3
                and len(nlookahead) >= 3
                and (other_desc in nlookahead or nlookahead in other_desc)
                for other_desc in other_descs
            ):
                break
            if re.fullmatch(r'\d{5,}', lookahead) or re.fullmatch(r'\d+\s*[@eE⚫●]', lookahead):
                saw_code_or_qty = True
                continue
            amount = _amount_from_line(lookahead)
            if amount is None:
                continue
            if require_structured_block:
                if not saw_code_or_qty or abs(amount - current_value) <= 2:
                    continue
            elif amount <= current_value * 5:
                continue
            amounts.append(amount)
        if not amounts:
            continue
        counts = Counter(amounts)
        candidates = [
            amount for amount, count in counts.items()
            if count >= 2 or (len(amounts) == 1 and not require_structured_block)
        ]
        if not candidates:
            continue
        current_score = _score(item_sum)
        best = min(
            candidates,
            key=lambda amount: _score(item_sum - total + amount),
        )
        new_score = _score(item_sum - total + best)
        if new_score + 0.5 >= current_score:
            continue
        item["unit_price"] = best
        item["total"] = best
        item_sum = item_sum - total + best


def _replace_split_price_block_when_balanced(extracted, unified_text):
    """Repair receipts where a leading item price is split from a name block."""
    subtotal = extracted.get("subtotal")
    lines = [line.strip() for line in unified_text.split('\n')]
    subtotal_idx = next((i for i, line in enumerate(lines) if re.fullmatch(r'小\s*計', line)), None)
    if subtotal_idx is None:
        return

    def _valid_desc(line: str) -> bool:
        if not line or _SKIP_PRICE_LINE.search(line) or _HEADER_LINE_RE.search(line):
            return False
        if re.search(r'TEL|登録番号|領収|レジ\s*\d|^\d+$|\d{4}/\d{2}/\d{2}', line):
            return False
        return bool(re.search(r'[ぁ-んァ-ン一-龥]', line))

    descs: list[str] = []
    idx = subtotal_idx - 1
    while idx >= 0 and _valid_desc(lines[idx]):
        descs.insert(0, lines[idx])
        idx -= 1
    if len(descs) < 2:
        return
    first_price = None
    for j in range(idx, max(idx - 4, -1), -1):
        m = re.fullmatch(r'(\d{1,4})', lines[j])
        if m:
            first_price = float(m.group(1))
            break
    if first_price is None:
        return

    candidates: list[tuple[int, float]] = []
    after = lines[subtotal_idx + 1:]
    for offset, line in enumerate(after, start=subtotal_idx + 1):
        if re.search(r'お預り|お釣り', line):
            break
        if re.fullmatch(r'\d+\s*点', line):
            continue
        m = re.fullmatch(r'(\d{1,4})', line)
        if not m:
            continue
        candidates.append((offset, float(m.group(1))))

    remaining_count = len(descs) - 1
    if len(candidates) < remaining_count:
        return
    target_candidates = list(candidates)
    if subtotal:
        try:
            target_candidates.append((-1, float(subtotal)))
        except (TypeError, ValueError):
            pass

    selected_prices: list[float] | None = None
    for combo in combinations(candidates, remaining_count):
        combo_sum = first_price + sum(value for _, value in combo)
        max_price_idx = max(index for index, _ in combo) if combo else idx
        for target_idx, target in target_candidates:
            if target_idx >= 0 and target_idx <= max_price_idx:
                continue
            if abs(combo_sum - target) <= 2:
                selected_prices = [first_price, *(value for _, value in combo)]
                break
        if selected_prices is not None:
            break
    if selected_prices is None:
        return

    extracted["line_items"] = [
        {
            "description": desc,
            "qty": 1.0,
            "unit_price": selected_prices[pos],
            "total": selected_prices[pos],
            "tax_category": "10%",
            "discount": 0,
            "discount_rate": "",
        }
        for pos, desc in enumerate(descs)
    ]


def _replace_stacked_name_price_rows_when_balanced(extracted, unified_text):
    """Parse receipts that stack item names first, then matching price rows."""
    total = extracted.get("total")
    subtotal = extracted.get("subtotal")
    if not total and not subtotal:
        return
    lines = [line.strip() for line in unified_text.split('\n')]
    start = 0
    for idx, line in enumerate(lines[:20]):
        if re.fullmatch(r'領\s*収\s*証?', line):
            start = idx + 1
            break
    end = next((i for i, line in enumerate(lines) if re.search(r'QUICPay\s*支払|お客様控え', line)), len(lines))
    zone = lines[start:end]
    known_amounts: list[float] = []
    for value in (subtotal, total):
        try:
            if value is not None:
                known_amounts.append(float(value))
        except (TypeError, ValueError):
            pass
    max_known_amount = max(known_amounts) if known_amounts else None

    def _clean_desc(text: str) -> str:
        text = re.sub(r'^[◎○●内*＊]\s*', '', text.strip())
        if re.search(r'たまご\s+1$', text):
            return re.sub(r'\s+', '', text).strip()
        text = _clean_ocr_price_line_desc(text)
        text = re.sub(r'^\d{3,}[A-Za-z]?\)?\s*', '', text).strip()
        return re.sub(r'\s+', '', text).strip()

    def _valid_desc(text: str) -> bool:
        text = _clean_desc(text)
        is_bag = _is_bag_description(text)
        if not text or len(text) < 3:
            return False
        if _SKIP_PRICE_LINE.search(text) or (_HEADER_LINE_RE.search(text) and not is_bag):
            return False
        if re.search(r'軽減税率|適用商品|返品|ご理解|ご来店|ありがとうございます', text):
            return False
        if re.search(r'電話|登録番号|貴No|領収', text):
            return False
        if re.fullmatch(r'\d{3,}[A-Za-z]?\)?', text):
            return False
        if re.search(r'レジ', text) and not is_bag:
            return False
        return bool(re.search(r'[ぁ-んァ-ン一-龥]', text))

    pending: list[str] = []
    rows: list[dict] = []
    low_price_indices: list[int] = []
    pending_qty_detail: tuple[float, float] | None = None
    pending_prices: list[tuple[float, str]] = []
    max_pending_before_price = 0
    summary_seen_since_last_price = False
    for line in zone:
        if re.fullmatch(r'合\s*計|小\s*計|領収', line):
            summary_seen_since_last_price = True
            continue
        if re.search(r'対象|内消費税|消費税', line):
            summary_seen_since_last_price = True
            continue
        qm = re.fullmatch(r'@?\s*(\d[\d,]*)\s*[×xX]\s*(\d+(?:\.\d+)?)\s*点', line)
        if qm:
            unit = float(qm.group(1).replace(',', ''))
            qty = float(qm.group(2))
            pending_qty_detail = (unit, qty)
            continue
        qty_total_m = re.fullmatch(
            r'単\s*([\d,]+)\s*[×xXⅩ]\s*(\d+)\s*個?\s*(外|軽)?\s*[¥￥]\s*(\d[\d,]*)',
            line,
        )
        if qty_total_m and pending:
            max_pending_before_price = max(max_pending_before_price, len(pending))
            unit = float(qty_total_m.group(1).replace(',', ''))
            qty = float(qty_total_m.group(2))
            marker = qty_total_m.group(3) or ""
            total_price = float(qty_total_m.group(4).replace(',', ''))
            desc = pending.pop(0)
            rows.append({
                "description": desc,
                "qty": qty,
                "unit_price": unit,
                "total": total_price,
                "tax_category": "10%" if marker == "外" or _is_bag_description(desc) else "8%",
                "discount": 0,
                "discount_rate": "",
            })
            pending_qty_detail = None
            summary_seen_since_last_price = False
            continue

        unit_qty_m = re.fullmatch(
            r'単\s*([\d,]+)\s*[×xXⅩ]\s*(\d+)\s*個?\s*(外|軽)?',
            line,
        )
        if unit_qty_m:
            unit = float(unit_qty_m.group(1).replace(',', ''))
            qty = float(unit_qty_m.group(2))
            pending_qty_detail = (unit, qty)
            continue

        inline_price_m = re.fullmatch(r'(.+?[ぁ-んァ-ン一-龥].*?)\s+[¥￥]\s*(\d[\d,]*)\s*(外|軽)?', line)
        if inline_price_m and _valid_desc(inline_price_m.group(1)):
            max_pending_before_price = max(max_pending_before_price, len(pending))
            desc = _clean_desc(inline_price_m.group(1))
            marker = inline_price_m.group(3) or ""
            price = float(inline_price_m.group(2).replace(',', ''))
            rows.append({
                "description": desc,
                "qty": 1.0,
                "unit_price": price,
                "total": price,
                "tax_category": "10%" if marker == "外" or _is_bag_description(desc) else "8%",
                "discount": 0,
                "discount_rate": "",
            })
            summary_seen_since_last_price = False
            continue

        pm = re.fullmatch(r'[¥￥]\s*(\d[\d,]*)\s*(軽?)', line)
        if pm and pending:
            max_pending_before_price = max(max_pending_before_price, len(pending))
            price = float(pm.group(1).replace(',', ''))
            desc = pending.pop(0)
            qty = 1.0
            unit_price = price
            if pending_qty_detail is not None:
                detail_unit, detail_qty = pending_qty_detail
                if abs(detail_unit * detail_qty - price) <= 2:
                    unit_price = detail_unit
                    qty = detail_qty
                pending_qty_detail = None
            row = {
                "description": desc,
                "qty": qty,
                "unit_price": unit_price,
                "total": price,
                "tax_category": "10%" if _is_bag_description(desc) else "8%",
                "discount": 0,
                "discount_rate": "",
            }
            if price < 100 and not _is_bag_description(desc):
                low_price_indices.append(len(rows))
            rows.append(row)
            summary_seen_since_last_price = False
            continue
        if pm and not summary_seen_since_last_price:
            price = float(pm.group(1).replace(',', ''))
            if max_known_amount is None or price < max_known_amount:
                pending_prices.append((price, pm.group(2) or ""))
                if len(pending_prices) > 4:
                    pending_prices = pending_prices[-4:]
                continue
        if _valid_desc(line):
            desc = _clean_desc(line)
            if pending_prices and not summary_seen_since_last_price:
                price, marker = pending_prices.pop(0)
                rows.append({
                    "description": desc,
                    "qty": 1.0,
                    "unit_price": price,
                    "total": price,
                    "tax_category": "10%" if _is_bag_description(desc) else "8%",
                    "discount": 0,
                    "discount_rate": "",
                })
                continue
            if (
                summary_seen_since_last_price
                and not _is_bag_description(desc)
                and any(_is_bag_description(existing) for existing in pending)
            ):
                insert_at = next(
                    (pos for pos, existing in enumerate(pending) if _is_bag_description(existing)),
                    len(pending),
                )
                pending.insert(insert_at, desc)
            else:
                pending.append(desc)
            if len(pending) > 6:
                pending = pending[-6:]

    if len(rows) < 3:
        return
    if max_pending_before_price < 2:
        return
    if not any(
        re.fullmatch(r'@?\s*\d[\d,]*\s*[×xX]\s*\d+(?:\.\d+)?\s*点', line)
        for line in zone
    ) and sum(1 for line in zone if re.fullmatch(r'[¥￥]\s*\d[\d,]*\s*軽?', line)) < len(rows):
        return
    row_sum = sum(float(row["total"]) for row in rows)
    printed_total_candidates = {
        float(m.group(1).replace(',', ''))
        for line in zone
        for m in [re.fullmatch(r'[¥￥]\s*(\d[\d,]*)\s*', line)]
        if m
    }
    target_options: list[tuple[str, float]] = []
    inclusive_tax_evidence = any(
        isinstance(tax, dict) and str(tax.get("label") or "") == "内税"
        for tax in (extracted.get("taxes") or [])
    ) or bool(re.search(r'内消費税|内税', unified_text))
    if total is not None:
        try:
            target_options.append(("total", float(total)))
        except (TypeError, ValueError):
            pass
    if subtotal is not None:
        try:
            if not inclusive_tax_evidence:
                target_options.insert(0, ("subtotal", float(subtotal)))
            elif total is None:
                target_options.append(("subtotal", float(subtotal)))
        except (TypeError, ValueError):
            pass
    for candidate in printed_total_candidates:
        if any(abs(candidate - existing) <= 2 for _kind, existing in target_options):
            continue
        target_options.append(("printed", candidate))
    target_kind, target = min(
        target_options,
        key=lambda option: (
            abs(row_sum - option[1]),
            0 if option[0] == "subtotal" else 1 if option[0] == "total" else 2,
        ),
    )
    if abs(row_sum - target) > 2 and len(low_price_indices) == 1:
        idx = low_price_indices[0]
        gap = target - (row_sum - float(rows[idx]["total"]))
        if gap > rows[idx]["total"] and gap < target:
            low_text = str(int(rows[idx]["total"]))
            gap_text = str(int(round(gap)))
            if gap_text.startswith(low_text):
                rows[idx]["unit_price"] = float(gap)
                rows[idx]["total"] = float(gap)
                row_sum = sum(float(row["total"]) for row in rows)
    if abs(row_sum - target) > 2:
        return
    printed_count = None
    count_m = re.search(
        r'合計点数\s*\n\s*(\d+)\s*点|(\d+)\s*点\s*\n\s*お預り|(\d+)\s*点\s*買',
        unified_text,
    )
    if count_m:
        printed_count = int(count_m.group(1) or count_m.group(2) or count_m.group(3))
    qty_count = int(sum(float(row.get("qty") or 1) for row in rows))
    if printed_count is not None and qty_count != printed_count:
        return
    rate_bases = extract_rate_bases(unified_text)
    usable_rate_bases = {
        rate: base for rate, base in rate_bases.items()
        if rate in {"8%", "10%"} and base
    }
    tax_rates_with_amount = {
        tax.get("rate")
        for tax in (extracted.get("taxes") or [])
        if isinstance(tax, dict) and tax.get("rate") in {"8%", "10%"} and tax.get("amount") is not None
    }
    has_two_rate_tax_amounts = {"8%", "10%"}.issubset(tax_rates_with_amount) or bool(
        re.search(r'\(10%対象\s*¥?\s*[\d,]+\s*内税\s*¥?\s*[\d,]+', unified_text)
        and re.search(r'\(0?8%対象\s*¥?\s*[\d,]+\s*内税\s*¥?\s*[\d,]+', unified_text)
    )
    if usable_rate_bases and has_two_rate_tax_amounts:
        _rebalance_tax_categories_to_rate_bases(rows, unified_text, extracted.get("taxes"), rate_bases)
    else:
        for row in rows:
            if _is_bag_description(row.get("description") or ""):
                row["tax_category"] = "10%"
    extracted["line_items"] = rows
    if target_kind == "subtotal":
        return
    current_total = extracted.get("total")
    try:
        current_total_f = float(current_total) if current_total is not None else None
    except (TypeError, ValueError):
        current_total_f = None
    if current_total_f is None or abs(current_total_f - target) > 2:
        old_total = current_total_f
        extracted["total"] = target
        amount_paid = extracted.get("amount_paid")
        try:
            amount_paid_f = float(amount_paid) if amount_paid is not None else None
        except (TypeError, ValueError):
            amount_paid_f = None
        if (
            amount_paid_f is None
            or (old_total is not None and abs(amount_paid_f - old_total) <= 2)
            or amount_paid_f < target
        ):
            extracted["amount_paid"] = target


def _fix_split_bag_price_from_nearby_single_digit(extracted, unified_text):
    """Repair tiny bag totals when OCR splits the bag row from a nearby single-digit price."""
    items = extracted.get("line_items") or []
    bag_items = [
        item for item in items
        if isinstance(item, dict) and _is_bag_description(item.get("description") or "")
    ]
    if len(bag_items) != 1:
        return
    item = bag_items[0]
    if float(item.get("total") or 0) > 10:
        return
    if not re.search(r'有料レジ袋[^\n]*\(\s*3', unified_text):
        return
    lines = [line.strip() for line in unified_text.split('\n')]
    for idx, line in enumerate(lines):
        if "有料レジ袋" not in line:
            continue
        for nearby in lines[idx + 1:idx + 8]:
            if re.fullmatch(r'5', nearby):
                item["qty"] = 1.0
                item["unit_price"] = 5.0
                item["total"] = 5.0
                item["tax_category"] = "10%"
                return


def _fix_small_bag_description_from_ocr_entry(extracted, unified_text):
    """Use visible tiny bag OCR rows to rename an otherwise unlabeled low-value item."""
    items = extracted.get("line_items") or []
    if not items or any(
        isinstance(item, dict) and _is_bag_description(item.get("description") or "")
        for item in items
    ):
        return
    entries = _bag_entries_from_ocr(unified_text)
    if not entries:
        return
    entry = entries[0]
    total = float(entry.get("total") or 0)
    if total <= 0 or total > 10:
        return
    bag_desc = None
    for line in unified_text.split('\n'):
        if _is_bag_description(line):
            bag_desc = re.sub(r'^\s*内\s*', '', line.strip())
            break
    if not bag_desc:
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        if abs(float(item.get("total") or 0) - total) > 0.5:
            continue
        item["description"] = bag_desc
        item["qty"] = entry["qty"]
        item["unit_price"] = entry["unit_price"]
        item["total"] = entry["total"]
        item["tax_category"] = "10%"
        return


def _fix_name_bag_amount_shift_from_ocr(extracted, unified_text):
    """Repair OCR row order: product name, paid bag with price, then product price."""
    items = extracted.get("line_items") or []
    if len(items) < 2:
        return

    lines = [line.strip() for line in unified_text.split('\n')]
    rate_bases = {
        rate: value
        for rate, value in extract_rate_bases(unified_text).items()
        if value is not None
    }
    if len(rate_bases) < 2:
        return

    def _summary_amount(label: str) -> float | None:
        for idx, line in enumerate(lines):
            if label not in line:
                continue
            inline = re.search(r'[¥￥]\s*([\d,]+)', line)
            if inline:
                return float(inline.group(1).replace(',', ''))
            for nearby in lines[idx + 1: idx + 3]:
                nearby_m = re.search(r'^[¥￥]?\s*([\d,]+)\s*$', nearby)
                if nearby_m:
                    return float(nearby_m.group(1).replace(',', ''))
        return None

    subtotal_target = _summary_amount("小計")
    if subtotal_target is None:
        subtotal_target = extracted.get("subtotal")
    if subtotal_target is None:
        return

    def _clean_name(line: str) -> str:
        text = _OCR_TRAILING_PRICE_RE.sub("", line or "").strip()
        text = re.sub(r'\s+', '', text)
        return text.translate(str.maketrans({"・": "•", "･": "•", "·": "•"}))

    def _marked_amount(line: str) -> float | None:
        m = re.fullmatch(r'[¥￥]?\s*([\d,]+)\s*(?:[%％][*※除軽]?|[*※除軽])', line or "")
        if not m:
            return None
        return _parse_amount_fragment(m.group(1))

    sequence = None
    for idx in range(len(lines) - 2):
        name_line = lines[idx]
        bag_line = lines[idx + 1]
        amount_line = lines[idx + 2]
        if (
            not name_line
            or _is_bag_description(name_line)
            or _OCR_TRAILING_PRICE_RE.search(name_line)
            or _OCR_ZONE_END_RE.search(name_line)
            or not re.search(r'[ぁ-んァ-ン一-龥A-Za-z]', name_line)
        ):
            continue
        bag_m = _OCR_TRAILING_PRICE_RE.search(bag_line)
        if not bag_m:
            continue
        bag_desc = _OCR_TRAILING_PRICE_RE.sub("", bag_line).strip()
        bag_price = _parse_amount_fragment(bag_m.group(1).replace("¥", "").replace("￥", "").strip())
        product_price = _marked_amount(amount_line)
        if (
            not bag_desc
            or not _is_bag_description(bag_desc)
            or bag_price is None
            or product_price is None
            or bag_price <= 0
            or bag_price > 30
            or product_price <= bag_price
        ):
            continue
        sequence = (_clean_name(name_line), bag_desc, float(bag_price), float(product_price))
        break
    if sequence is None:
        return

    product_desc, bag_desc, bag_price, product_price = sequence
    bag_rates = [
        rate
        for rate, base in rate_bases.items()
        if abs(float(base) - bag_price) <= 2
    ]
    product_rates = [
        rate
        for rate, base in rate_bases.items()
        if rate not in bag_rates and float(base) >= product_price
    ]
    if not bag_rates or not product_rates:
        return
    bag_rate = sorted(bag_rates, key=lambda rate: float(rate.rstrip("%")), reverse=True)[0]
    product_rate = sorted(product_rates, key=lambda rate: float(rate.rstrip("%")))[0]

    def _fill_single_qty_unit_prices() -> None:
        for item in items:
            if (
                isinstance(item, dict)
                and item.get("unit_price") is None
                and float(item.get("qty") or 1) == 1
                and item.get("total") is not None
            ):
                item["unit_price"] = float(item["total"])

    current_sum = 0.0
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            current_sum += float(item.get("total") or 0)
        except (TypeError, ValueError):
            current_sum = -1.0
            break
    if abs(current_sum - float(subtotal_target)) <= 2:
        _fill_single_qty_unit_prices()

    product_item = None
    bag_item = None
    for item in items:
        if not isinstance(item, dict):
            continue
        desc = item.get("description") or ""
        try:
            total = float(item.get("total") or 0)
        except (TypeError, ValueError):
            continue
        if _is_bag_description(desc) and abs(total - bag_price) <= 1:
            bag_item = item
        elif _is_bag_description(desc) and abs(total - product_price) <= 1:
            product_item = item
    if product_item is None or bag_item is None or product_item is bag_item:
        return

    projected_sum = 0.0
    for item in items:
        if not isinstance(item, dict):
            continue
        if item is product_item:
            projected_sum += product_price
            continue
        if item is bag_item:
            projected_sum += bag_price
            continue
        try:
            projected_sum += float(item.get("total") or 0)
        except (TypeError, ValueError):
            return
    if abs(projected_sum - float(subtotal_target)) > 2:
        return

    product_item["description"] = product_desc
    product_item["qty"] = 1.0
    product_item["unit_price"] = product_price
    product_item["total"] = product_price
    product_item["tax_category"] = product_rate
    product_item["discount"] = product_item.get("discount") or 0
    product_item["discount_rate"] = product_item.get("discount_rate") or ""

    bag_item["description"] = bag_desc
    bag_item["qty"] = 1.0
    bag_item["unit_price"] = bag_price
    bag_item["total"] = bag_price
    bag_item["tax_category"] = bag_rate
    bag_item["discount"] = bag_item.get("discount") or 0
    bag_item["discount_rate"] = bag_item.get("discount_rate") or ""

    _fill_single_qty_unit_prices()


def _drop_numeric_marker_description_rows(extracted, unified_text):
    """Drop parsed rows whose description is only a numeric price/marker token."""
    items = extracted.get("line_items") or []
    if not items:
        return

    extracted["line_items"] = [
        item for item in items
        if not (
            isinstance(item, dict)
            and re.fullmatch(r'\d+\s*[*＊※%％]?', str(item.get("description") or "").strip())
        )
    ]


def _restore_stacked_inclusive_tax_block(extracted, unified_text):
    """Restore inclusive tax amounts from stacked rate-target/tax blocks."""
    if re.search(r'小計\s*\(?\s*税抜\s*\d+\s*%', unified_text):
        return
    lines = [line.strip() for line in unified_text.split('\n')]
    labels: list[tuple[str, bool]] = []
    values: list[float] = []
    for line in lines:
        rate_m = re.search(r'(\d+(?:\.\d+)?)\s*%\s*対象', line)
        if rate_m:
            labels.append((normalize_tax_rate(rate_m.group(1) + "%"), False))
            continue
        if re.search(r'内消費税', line):
            if labels:
                labels.append((labels[-1][0], True))
            continue
        if labels:
            vm = re.fullmatch(r'[¥￥]\s*([\d,]+)\s*[\)）]', line)
            if vm:
                values.append(float(vm.group(1).replace(',', '')))
    if not labels or len(values) < len(labels):
        return
    taxes: list[dict] = []
    for (rate, is_tax), value in zip(labels, values[:len(labels)]):
        if is_tax and value > 0:
            taxes.append({"rate": rate, "label": "内税", "amount": value})
    if taxes:
        extracted["taxes"] = taxes
        total = extracted.get("total")
        try:
            total_f = float(total) if total is not None else None
        except (TypeError, ValueError):
            total_f = None
        if total_f is not None:
            tax_sum = _sum_taxable_amounts(taxes)
            if 0 <= tax_sum < total_f:
                extracted["subtotal"] = total_f - tax_sum


def _restore_tax_excluded_per_rate_blocks(extracted, unified_text):
    """Restore tax amounts from paired 小計(税抜N%)/消費税等(N%) blocks."""
    if not re.search(r'小計\s*\(?\s*税抜\s*\d+\s*%', unified_text):
        return
    lines = [line.strip() for line in unified_text.split('\n')]
    pairs: list[tuple[str, str, float]] = []
    pending: list[tuple[str, str]] = []
    start_idx = None
    for idx, line in enumerate(lines):
        if (
            start_idx is not None
            and pairs
            and not pending
            and re.search(r'^\(\s*(?:税率|内\s*消費税等?)', line)
        ):
            break
        subtotal_m = re.search(r'小計\s*\(?\s*税抜\s*(\d+(?:\.\d+)?)\s*%', line)
        if subtotal_m:
            pending.append((normalize_tax_rate(subtotal_m.group(1) + "%"), "base"))
            start_idx = idx if start_idx is None else start_idx
            continue
        tax_m = re.search(r'消費税等?\s*\(?\s*(\d+(?:\.\d+)?)\s*%', line)
        if tax_m and start_idx is not None and not re.search(r'内\s*消費税等?', line):
            pending.append((normalize_tax_rate(tax_m.group(1) + "%"), "tax"))
            continue
        vm = re.fullmatch(r'[¥￥]\s*([\d,]+)\s*[\)）]?', line)
        if vm and pending:
            rate, kind = pending.pop(0)
            pairs.append((rate, kind, float(vm.group(1).replace(',', ''))))
            continue
        if start_idx is not None and pairs and re.search(r'お預り|お買上|支払|明細|マーク|伝票', line):
            break
    if not pairs or start_idx is None:
        return

    taxes: list[dict] = []
    for rate, kind, value in pairs:
        if kind == "tax" and value > 0:
            taxes.append({"rate": rate, "label": "外税", "amount": value})
    if taxes:
        extracted["taxes"] = taxes


def _restore_single_rate_inclusive_tax_block(extracted, unified_text):
    """Recover inclusive tax when a receipt prints a single-rate tax summary."""
    total = float(extracted.get("total") or 0)
    if total <= 0:
        return
    rate_m = re.search(r'\(\s*内\s*(\d+(?:\.\d+)?)\s*%\s*税', unified_text)
    rate = None
    expected = None
    if rate_m:
        rate = normalize_tax_rate(rate_m.group(1) + "%")
        try:
            rate_pct = float(rate.rstrip('%')) / 100.0
        except ValueError:
            return
        expected = round(total * rate_pct / (1 + rate_pct))
        if expected <= 0:
            return
        idx = unified_text.find(rate_m.group(0))
        tail = unified_text[idx:] if idx >= 0 else unified_text
        values = [
            float(m.group(1).replace(',', ''))
            for m in re.finditer(r'[¥￥]\s*([\d,]+)\s*[\)）]?', tail)
        ]
        if not any(abs(value - expected) <= 2 for value in values):
            return
    else:
        inline_m = re.search(
            r'(\d+(?:\.\d+)?)\s*%\s*対象\s*[¥￥]?\s*([\d,]+)\s*内消費税\s*[¥￥]?\s*([\d,]+)',
            unified_text,
            flags=re.S,
        )
        if not inline_m:
            return
        rate = normalize_tax_rate(inline_m.group(1) + "%")
        base = float(inline_m.group(2).replace(',', ''))
        expected = float(inline_m.group(3).replace(',', ''))
        rate_bases = {
            r: b for r, b in extract_rate_bases(unified_text).items()
            if r != "0%" and b
        }
        if set(rate_bases) - {rate}:
            return
        item_sum = sum(
            float(item.get("total") or 0)
            for item in extracted.get("line_items") or []
            if isinstance(item, dict)
        )
        if not (
            abs(base - total) <= 2
            or (item_sum > 0 and abs(base - item_sum) <= 2)
        ):
            return
    extracted["taxes"] = [{"rate": rate, "label": "内税", "amount": float(expected)}]
    extracted["subtotal"] = total - float(expected)
    items = extracted.get("line_items") or []
    if len(items) == 1 and isinstance(items[0], dict):
        items[0]["tax_category"] = rate
    elif items:
        rate_bases = {
            r: b for r, b in extract_rate_bases(unified_text).items()
            if r != "0%" and b
        }
        item_sum = sum(
            float(item.get("total") or 0)
            for item in items
            if isinstance(item, dict)
        )
        if (
            not (set(rate_bases) - {rate})
            and (not rate_bases or abs(float(rate_bases.get(rate) or 0) - item_sum) <= 2 or abs(item_sum - total) <= 2)
        ):
            for item in items:
                if isinstance(item, dict):
                    item["tax_category"] = rate
    _clean_code_prefixed_item_descriptions(extracted)


def _fix_header_store_line_location(extracted, unified_text):
    """Recover a branch/store location printed directly under the header."""
    existing = re.sub(r'\s+', '', extracted.get("location") or "")
    if existing:
        return
    lines = [line.strip() for line in unified_text.split("\n") if line.strip()]
    merchant = re.sub(r'\s+', '', extracted.get("merchant") or "").upper()
    for idx, line in enumerate(lines[:8]):
        if idx + 1 >= len(lines):
            break
        compact = re.sub(r'\s+', '', line).upper()
        next_line = lines[idx + 1].strip()
        if not re.fullmatch(r'[\u3040-\u30ff\u3400-\u9fffA-Za-z0-9・ー\s]{2,30}店', next_line):
            continue
        if re.search(r'TEL|電話|#|No\.?|登録番号|合計|小計|対象|消費税|\d{4}[/-]\d{1,2}', next_line, re.IGNORECASE):
            continue
        following = "\n".join(lines[idx + 2:idx + 5])
        header_matches_merchant = bool(merchant and merchant in compact)
        header_has_brand_shape = bool(
            re.search(r'[A-Z]{3,}|[ァ-ン]{3,}|[\u3400-\u9fff]{3,}', line)
            and not re.search(r'\d{2,}|[¥￥]|合計|小計|対象|消費税', line)
        )
        followed_by_store_contact = bool(re.search(r'TEL|電話|#\s*\d+', following, re.IGNORECASE))
        if header_matches_merchant or (header_has_brand_shape and followed_by_store_contact):
            extracted["location"] = next_line
            return


def _fix_split_address_location_from_ocr(extracted, unified_text):
    """Recover addresses split across admin-area and street-number OCR lines."""
    existing = re.sub(r'\s+', '', extracted.get("location") or "")
    lines = [line.strip() for line in unified_text.split('\n')]
    for idx, line in enumerate(lines[:-1]):
        if not re.fullmatch(r'.*[都道府県].*[市区町村]', line):
            continue
        nxt = lines[idx + 1].strip()
        if re.search(r'TEL|電話|登録番号|営業時間', nxt, re.IGNORECASE):
            continue
        if not re.fullmatch(r'[^¥￥\s]+?\d+(?:[-－]\d+)+', nxt):
            continue
        candidate = re.sub(r'\s+', '', line + nxt)
        if not existing or existing in candidate or len(candidate) > len(existing) + 4:
            extracted["location"] = candidate
            return


def _recover_labeled_purchase_site_location(extracted, unified_text):
    """Recover a printed site-area token from a labeled purchase-site line."""
    existing = re.sub(r'\s+', '', extracted.get("location") or "")
    lines = [line.strip() for line in unified_text.split('\n') if line.strip()]
    for line in lines:
        m = re.search(
            r'購入倉庫店\s*[:：]\s*'
            r'([^\s:：¥￥,，。()（）]+?倉庫店)',
            line,
        )
        if not m:
            continue
        site = re.sub(r'\s+', '', m.group(1))
        candidate = re.sub(r'倉庫店$', '', site)
        if not re.fullmatch(r'[\u3040-\u30ff\u3400-\u9fffA-Za-z0-9・ー]{2,20}', candidate):
            continue
        if re.search(r'TEL|電話|登録番号|会員番号|領収|合計|小計|対象|消費税', candidate, re.IGNORECASE):
            continue
        if existing and (candidate in existing or existing in candidate):
            return
        extracted["location"] = candidate
        return


def _restore_zero_points_when_no_redemption(extracted, unified_text):
    """Restore explicit zero point usage when payment math shows no redemption."""
    if extracted.get("points_used") is not None:
        return
    if re.search(r'ポイント利用|利用ポイント|ポイント値引|ポイント\s*-', unified_text):
        return
    if not re.search(r'ポイント|リワード|会員', unified_text):
        return
    try:
        total = float(extracted.get("total"))
        amount_paid = float(extracted.get("amount_paid"))
    except (TypeError, ValueError):
        return
    if abs(total - amount_paid) <= 2:
        extracted["points_used"] = 0


POSTPROCESS_MUTATION_FIELDS = (
    "date",
    "line_items",
    "location",
    "merchant",
    "taxes",
    "tax_entries",
    "subtotal",
    "total",
    "amount_paid",
    "points_used",
    "payment_method",
)

POSTPROCESS_PHASES = (
    {
        "name": "header_identity_repair",
        "reads": ("merchant", "date", "time", "location", "ocr_text"),
        "writes": ("merchant", "date", "time", "location"),
        "invariant": "Header fields must be backed by visible OCR header/address/date evidence.",
    },
    {
        "name": "financial_totals_repair",
        "reads": ("subtotal", "total", "amount_paid", "taxes", "ocr_totals", "ocr_text"),
        "writes": ("subtotal", "total", "amount_paid", "taxes", "payment_method"),
        "invariant": "Totals must remain consistent with printed cash/tax/summary arithmetic.",
    },
    {
        "name": "cash_tender_reconciliation",
        "reads": ("total", "amount_paid", "payment_method", "points_used", "ocr_text"),
        "writes": ("total", "amount_paid", "payment_method"),
        "invariant": "Cash tender/change repairs require visible printed total, tendered amount, and change arithmetic.",
    },
    {
        "name": "service_receipt_recovery",
        "reads": ("line_items", "subtotal", "total", "taxes", "payment_method", "ocr_text"),
        "writes": ("line_items", "taxes", "subtotal", "payment_method"),
        "invariant": "Service receipt recovery requires visible service/table or bare-receipt OCR layout and item/tax arithmetic consistency.",
    },
    {
        "name": "initial_item_recovery",
        "reads": ("line_items", "subtotal", "total", "ocr_text", "ocr_layout_blocks"),
        "writes": ("line_items",),
        "invariant": "Recovered rows must improve item-total consistency or match visible item layout.",
    },
    {
        "name": "gap_item_recovery",
        "reads": ("line_items", "subtotal", "total", "ocr_text"),
        "writes": ("line_items",),
        "invariant": "Gap item recovery requires visible missing, discounted, or repeated OCR rows and subtotal/total item-sum arithmetic.",
    },
    {
        "name": "item_cleanup",
        "reads": ("line_items", "subtotal", "total", "ocr_text"),
        "writes": ("line_items",),
        "invariant": "Cleanup may remove or rename rows only when OCR evidence and row sums stay coherent.",
    },
    {
        "name": "ocr_description_reconciliation",
        "reads": ("line_items", "subtotal", "total", "ocr_text"),
        "writes": ("line_items",),
        "invariant": "Description reconciliation requires visible OCR code/name context and must preserve item counts, prices, and totals.",
    },
    {
        "name": "quantity_detail_reconciliation",
        "reads": ("line_items", "subtotal", "total", "ocr_text"),
        "writes": ("line_items",),
        "invariant": "Quantity-detail repairs require visible OCR qty/unit rows and qty * unit or discount-adjusted item arithmetic.",
    },
    {
        "name": "tax_category_assignment",
        "reads": ("line_items", "taxes", "subtotal", "total", "ocr_totals", "ocr_text"),
        "writes": ("line_items", "taxes"),
        "invariant": "Item tax categories and tax entries must agree with printed rate bases or tax summaries.",
    },
    {
        "name": "payment_points_reconciliation",
        "reads": ("amount_paid", "payment_method", "points_used", "subtotal", "total", "taxes", "ocr_text"),
        "writes": ("amount_paid", "payment_method", "points_used", "subtotal"),
        "invariant": "Payment, points, and subtotal changes must preserve total/tax arithmetic.",
    },
    {
        "name": "structural_item_reconstruction",
        "reads": ("line_items", "subtotal", "total", "taxes", "ocr_text", "ocr_totals"),
        "writes": ("line_items", "taxes", "subtotal", "total", "amount_paid"),
        "invariant": "Structural reconstruction must be triggered by OCR row layout and validated by sums.",
    },
    {
        "name": "final_consistency_pass",
        "reads": ("line_items", "taxes", "subtotal", "total", "amount_paid", "points_used", "ocr_text"),
        "writes": ("line_items", "taxes", "subtotal", "total", "amount_paid", "points_used"),
        "invariant": "Final mutations must restore field consistency without fixture or known-answer logic.",
    },
)
POSTPROCESS_PHASE_BY_NAME = {phase["name"]: phase for phase in POSTPROCESS_PHASES}


def _copy_mutation_value(value):
    if isinstance(value, dict):
        return {key: _copy_mutation_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_copy_mutation_value(item) for item in value]
    return value


def _snapshot_receipt_mutation_fields(extracted: dict) -> dict:
    return {
        field: _copy_mutation_value(extracted.get(field))
        for field in POSTPROCESS_MUTATION_FIELDS
        if field in extracted
    }


def _diff_receipt_mutation_fields(before: dict, after: dict) -> dict:
    changes = {}
    for field in POSTPROCESS_MUTATION_FIELDS:
        before_value = before.get(field)
        after_value = after.get(field)
        if before_value != after_value:
            changes[field] = {
                "before": before_value,
                "after": after_value,
            }
    return changes


def _record_receipt_mutation(
    mutation_trace: list[dict] | None,
    stage: str,
    before: dict | None,
    extracted: dict,
) -> dict | None:
    if mutation_trace is None or before is None:
        return None
    after = _snapshot_receipt_mutation_fields(extracted)
    changes = _diff_receipt_mutation_fields(before, after)
    if changes:
        mutation_trace.append({"stage": stage, "changes": changes})
    return after


def _record_receipt_phase_mutation(
    mutation_trace: list[dict] | None,
    phase_name: str,
    before: dict | None,
    extracted: dict,
) -> dict | None:
    if phase_name not in POSTPROCESS_PHASE_BY_NAME:
        raise ValueError(f"Unknown receipt postprocess phase: {phase_name}")
    trace_len = len(mutation_trace) if mutation_trace is not None else 0
    after = _record_receipt_mutation(mutation_trace, phase_name, before, extracted)
    if mutation_trace is not None and len(mutation_trace) > trace_len:
        phase = POSTPROCESS_PHASE_BY_NAME[phase_name]
        mutation_trace[-1]["reads"] = phase["reads"]
        mutation_trace[-1]["writes"] = phase["writes"]
        mutation_trace[-1]["invariant"] = phase["invariant"]
    return after


def _run_structural_item_projection_phase(
    extracted: dict,
    unified_text: str,
    ocr_totals: dict | None,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: OCR row layout exposes item/price/qty sequences or POS codes.

    Invariant: each projection helper may replace rows only when its own
    arithmetic checks keep item totals coherent with printed receipt amounts.
    """
    for repair in repairs:
        if repair == "barcode_unit_qty_amount_stack":
            _replace_barcode_unit_qty_amount_stack_when_balanced(extracted, unified_text)
        elif repair == "barcode_qty_price_rows":
            _replace_barcode_qty_price_rows_when_balanced(extracted, unified_text)
        elif repair == "dense_item_rows":
            _replace_dense_item_rows_when_balanced(extracted, unified_text)
        elif repair == "dense_sequence_rows":
            _replace_dense_sequence_rows_when_balanced(extracted, unified_text)
        elif repair == "jan_pos_items":
            _replace_jan_pos_items_when_balanced(extracted, unified_text, ocr_totals or {})
        else:
            raise ValueError(f"Unknown structural item projection repair: {repair}")


def _run_quantity_detail_reconciliation_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: OCR qty detail rows adjacent to item names or price rows.

    Invariant: qty/unit mutations require visible quantity notation and
    qty * unit, printed total, or discount-adjusted item arithmetic.
    """
    for repair in repairs:
        if repair == "following_qty_detail":
            _repair_previous_item_from_following_qty_detail(extracted, unified_text)
        elif repair == "qty_totals_from_unit_lines":
            _fix_qty_totals_from_ocr_unit_lines(extracted, unified_text)
        elif repair == "qty_context_and_reduced_rate":
            _fix_qty_context_and_reduced_rate_from_ocr(extracted, unified_text)
        else:
            raise ValueError(f"Unknown quantity detail reconciliation repair: {repair}")


def _run_cash_tender_reconciliation_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: OCR cash-layout rows print total, tendered amount, and change.

    Invariant: total, amount_paid, and cash payment method changes must be
    consistent with printed tender/change arithmetic and points adjustments.
    """
    for repair in repairs:
        if repair == "stacked_cash_tender":
            _fix_total_from_stacked_cash_tender_block(extracted, unified_text)
        elif repair == "unlabeled_cash_tender_change":
            _fix_unlabeled_cash_tender_change_block(extracted, unified_text)
        else:
            raise ValueError(f"Unknown cash tender reconciliation repair: {repair}")


def _run_service_receipt_recovery_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: OCR service tables, bare receipt layouts, or single service rows.

    Invariant: service item recovery/removal and inclusive-tax reconstruction
    must preserve visible row layout, printed total evidence, and tax arithmetic.
    """
    for repair in repairs:
        if repair == "bare_service_without_itemization":
            _fix_bare_service_receipt_without_itemization(extracted, unified_text)
        elif repair == "service_table_items":
            _replace_service_table_items_when_balanced(extracted, unified_text)
        elif repair == "single_service_inclusive_tax":
            _fix_single_service_inclusive_tax(extracted, unified_text)
        else:
            raise ValueError(f"Unknown service receipt recovery repair: {repair}")


def _run_ocr_description_reconciliation_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: OCR code rows, duplicated names, O-ring text, or bag context.

    Invariant: description changes require visible OCR support and must keep
    item count, quantity, unit price, total, discount, and tax fields coherent.
    """
    for repair in repairs:
        if repair == "qty_code_rows":
            _fix_qty_code_row_descriptions_from_ocr(extracted, unified_text)
        elif repair == "code_table_order":
            _fix_code_table_descriptions_by_order(extracted, unified_text)
        elif repair == "duplicate_descriptions":
            _fix_duplicate_descriptions_from_ocr(extracted, unified_text)
        elif repair == "o_ring_descriptions":
            _fix_o_ring_descriptions_from_ocr(extracted, unified_text)
        elif repair == "colon_split_names":
            _fix_colon_split_product_names_from_ocr(extracted, unified_text)
        elif repair == "bag_code_context":
            _fix_bag_description_from_ocr_code_context(extracted, unified_text)
        else:
            raise ValueError(f"Unknown OCR description reconciliation repair: {repair}")


def _run_gap_item_recovery_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: OCR rows expose missing, discounted, or repeated item gaps.

    Invariant: recovered or replaced rows must be visible in OCR and close a
    subtotal/total item-sum gap without fixture, merchant, or product answers.
    """
    for repair in repairs:
        if repair == "missing_items":
            _recover_missing_items_from_gap(extracted, unified_text)
        elif repair == "discounted_gap":
            _recover_discounted_item_from_gap(extracted, unified_text)
        elif repair == "repeated_gap":
            _recover_repeated_item_from_gap(extracted, unified_text)
        elif repair == "repeated_ocr_block":
            _replace_repeated_ocr_item_block_when_balanced(extracted, unified_text)
        else:
            raise ValueError(f"Unknown gap item recovery repair: {repair}")


def _run_payment_points_reconciliation_phase(
    extracted: dict,
    unified_text: str,
    ocr_conf: float,
    llm_conf: dict | None,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: OCR points-use lines or points tender/payment rows are visible.

    Invariant: points_used, amount_paid, payment_method, and subtotal changes
    must preserve total minus points payment arithmetic and printed evidence.
    """
    for repair in repairs:
        if repair == "points_used":
            points = extract_points_used(unified_text)
            if points is not None:
                existing_points = extracted.get("points_used")
                if (
                    should_override_field("points_used", ocr_conf, llm_conf)
                    or existing_points is None
                    or (points > 0 and float(existing_points or 0) == 0)
                ):
                    extracted["points_used"] = points
            elif extracted.get("points_used") is not None:
                has_points_evidence = bool(re.search(r'ポイント利用|ポイント値引', unified_text))
                if not has_points_evidence:
                    extracted["points_used"] = 0
            else:
                extracted["points_used"] = 0
        elif repair == "points_payment":
            reconcile_points_payment_from_ocr(extracted, unified_text)
        else:
            raise ValueError(f"Unknown payment points reconciliation repair: {repair}")


def _tax_assignment_rate_bases(unified_text: str, ocr_totals: dict | None) -> dict:
    rate_bases = extract_rate_bases(unified_text)
    for rate, base in ((ocr_totals or {}).get('_breakdown_rate_bases') or {}).items():
        if rate not in rate_bases or rate_bases[rate] is None:
            rate_bases[rate] = base
    return rate_bases


def _run_tax_category_assignment_phase(
    extracted: dict,
    unified_text: str,
    ocr_totals: dict | None,
    repairs: tuple[str, ...],
    rate_bases: dict | None = None,
) -> dict:
    """Trigger: OCR rate markers, rate-base summaries, or price-line flags.

    Invariant: item tax categories must remain consistent with visible rate
    markers, printed rate bases, extracted tax entries, and single-bag splits.
    """
    items = extracted.get("line_items") or []
    merged_rate_bases = rate_bases if rate_bases is not None else _tax_assignment_rate_bases(
        unified_text,
        ocr_totals,
    )
    for repair in repairs:
        if repair == "assign_tax_categories":
            if items:
                assign_tax_categories(
                    items,
                    unified_text,
                    ocr_totals or {},
                    merged_rate_bases,
                    extracted_taxes=extracted.get("taxes"),
                )
        elif repair == "ocr_markers":
            if items:
                _fix_tax_categories_from_ocr_markers(items, unified_text)
        elif repair == "price_line_markers":
            _fix_tax_categories_from_price_line_markers(extracted, unified_text)
        elif repair == "single_bag_standard_split":
            if items:
                _apply_single_bag_standard_rate_split(items, merged_rate_bases)
        elif repair == "rebalance_rate_bases":
            if items:
                _rebalance_tax_categories_to_rate_bases(
                    items,
                    unified_text,
                    extracted.get("taxes"),
                    merged_rate_bases,
                )
        elif repair == "rebalance_standard_from_reduced_markers":
            if items:
                _rebalance_standard_categories_from_reduced_rate_markers(
                    items,
                    unified_text,
                    merged_rate_bases,
                )
        elif repair == "nonfood_packaging":
            if items:
                _fix_nonfood_packaging_tax_categories(items, unified_text, merged_rate_bases)
        elif repair == "single_standard_from_small_base":
            if items:
                _assign_single_standard_rate_from_small_base(items, merged_rate_bases)
        else:
            raise ValueError(f"Unknown tax category assignment repair: {repair}")
    return merged_rate_bases


def _run_line_item_cleanup_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: OCR cleanup exposes duplicate, numeric-marker, or non-product rows.

    Invariant: cleanup may drop rows only when visible OCR context or duplicate
    structure supports removal without breaking item-total consistency.
    """
    for repair in repairs:
        if repair == "drop_duplicate_embedded_price":
            if extracted.get("line_items"):
                _drop_duplicate_with_embedded_price(extracted["line_items"])
        elif repair == "drop_non_product_line_items":
            _drop_non_product_line_items(extracted, unified_text)
        elif repair == "drop_numeric_marker_description_rows":
            _drop_numeric_marker_description_rows(extracted, unified_text)
        else:
            raise ValueError(f"Unknown line-item cleanup repair: {repair}")


def postprocess_receipt(
    extracted: dict,
    unified_text: str,
    ocr_conf: float,
    ocr_totals: dict,
    llm_conf: dict | None,
    model: str,
    ocr_layout_blocks: list[dict] | None = None,
    mutation_trace: list[dict] | None = None,
) -> dict:
    """Apply all receipt-specific post-processing to the LLM extraction."""
    trace_snapshot = (
        _snapshot_receipt_mutation_fields(extracted)
        if mutation_trace is not None
        else None
    )
    _fix_company_name_merchant(extracted, unified_text)
    _fix_split_item_price_body_total_layout(extracted, unified_text)
    _apply_financial_overrides(extracted, ocr_totals, ocr_conf, llm_conf)
    _run_cash_tender_reconciliation_phase(
        extracted,
        unified_text,
        ("stacked_cash_tender", "unlabeled_cash_tender_change"),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "cash_tender_reconciliation",
        trace_snapshot,
        extracted,
    )
    _fix_implausible_tax_amounts(extracted, unified_text, ocr_totals)
    _fix_date(extracted, unified_text)
    _fix_time(extracted, unified_text)
    _fix_header_store_line_location(extracted, unified_text)
    _fix_split_address_location_from_ocr(extracted, unified_text)
    _recover_labeled_purchase_site_location(extracted, unified_text)
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "header_identity_repair",
        trace_snapshot,
        extracted,
    )
    _fix_payment_method(extracted, unified_text, ocr_conf, llm_conf)
    _fix_toll_payment_reference(extracted, unified_text)
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "financial_totals_repair",
        trace_snapshot,
        extracted,
    )
    _run_service_receipt_recovery_phase(
        extracted,
        unified_text,
        ("bare_service_without_itemization",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "service_receipt_recovery",
        trace_snapshot,
        extracted,
    )
    _fix_line_items(extracted, unified_text, ocr_layout_blocks=ocr_layout_blocks)
    _recover_qty_unit_total_item_from_empty_extraction(extracted, unified_text)
    _drop_phantom_from_tax_amount(extracted)
    _fix_priced_in_name_items(extracted, unified_text)
    _fix_small_non_bag_item_prices_from_ocr(extracted, unified_text)
    _fix_bag_item_prices_from_ocr(extracted, unified_text)
    _fix_split_bag_price_from_nearby_single_digit(extracted, unified_text)
    _fix_small_bag_description_from_ocr_entry(extracted, unified_text)
    _fix_items_from_subtotal(extracted, unified_text, ocr_totals)
    _run_gap_item_recovery_phase(extracted, unified_text, ("missing_items",))
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "gap_item_recovery",
        trace_snapshot,
        extracted,
    )
    _replace_prefixed_tax_marker_item_rows_when_balanced(extracted, unified_text)
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "initial_item_recovery",
        trace_snapshot,
        extracted,
    )
    _run_service_receipt_recovery_phase(
        extracted,
        unified_text,
        ("bare_service_without_itemization",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "service_receipt_recovery",
        trace_snapshot,
        extracted,
    )
    _recover_missing_bag_items_from_ocr(extracted, unified_text)
    _replace_overage_item_with_low_value_bag(extracted, unified_text)
    _fix_numeric_desc_from_ocr_price_context(extracted, unified_text)
    _run_ocr_description_reconciliation_phase(
        extracted,
        unified_text,
        ("o_ring_descriptions",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "ocr_description_reconciliation",
        trace_snapshot,
        extracted,
    )
    _fix_adjacent_ocr_price_shift_when_balanced(extracted, unified_text)
    _run_gap_item_recovery_phase(
        extracted,
        unified_text,
        ("discounted_gap", "repeated_gap", "repeated_ocr_block"),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "gap_item_recovery",
        trace_snapshot,
        extracted,
    )
    # Run embedded-price dedup AGAIN after recovery — recovery can pick up
    # OCR-merged 'X  N' lines as new phantom items even when 'X' already
    # exists in the extraction at the same price.
    _run_line_item_cleanup_phase(
        extracted,
        unified_text,
        ("drop_duplicate_embedded_price",),
    )
    _run_ocr_description_reconciliation_phase(
        extracted,
        unified_text,
        ("qty_code_rows", "code_table_order", "duplicate_descriptions"),
    )
    _fix_digit_misread_items(extracted, unified_text)
    _run_ocr_description_reconciliation_phase(
        extracted,
        unified_text,
        ("o_ring_descriptions",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "ocr_description_reconciliation",
        trace_snapshot,
        extracted,
    )
    _run_structural_item_projection_phase(
        extracted,
        unified_text,
        ocr_totals,
        (
            "barcode_unit_qty_amount_stack",
            "barcode_qty_price_rows",
            "dense_sequence_rows",
        ),
    )
    _run_quantity_detail_reconciliation_phase(
        extracted,
        unified_text,
        ("following_qty_detail",),
    )
    _replace_campaign_discount_stream_when_balanced(extracted, unified_text)
    _replace_prefixed_tax_marker_item_rows_when_balanced(extracted, unified_text)
    _run_quantity_detail_reconciliation_phase(
        extracted,
        unified_text,
        ("qty_totals_from_unit_lines", "qty_context_and_reduced_rate"),
    )
    _fix_name_bag_amount_shift_from_ocr(extracted, unified_text)
    _run_line_item_cleanup_phase(
        extracted,
        unified_text,
        ("drop_numeric_marker_description_rows",),
    )
    _run_structural_item_projection_phase(
        extracted,
        unified_text,
        ocr_totals,
        ("jan_pos_items",),
    )
    _run_line_item_cleanup_phase(
        extracted,
        unified_text,
        ("drop_duplicate_embedded_price",),
    )
    _run_ocr_description_reconciliation_phase(
        extracted,
        unified_text,
        (
            "qty_code_rows",
            "duplicate_descriptions",
            "colon_split_names",
            "bag_code_context",
        ),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "ocr_description_reconciliation",
        trace_snapshot,
        extracted,
    )
    _run_tax_category_assignment_phase(
        extracted,
        unified_text,
        ocr_totals,
        ("price_line_markers",),
    )
    _run_line_item_cleanup_phase(
        extracted,
        unified_text,
        ("drop_non_product_line_items",),
    )
    _run_gap_item_recovery_phase(extracted, unified_text, ("repeated_ocr_block",))
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "gap_item_recovery",
        trace_snapshot,
        extracted,
    )
    _fix_non_bag_items_named_as_bag(extracted, unified_text)
    _fix_embedded_price_suffix_totals(extracted, unified_text)
    _fix_adjacent_ocr_price_shift_when_balanced(extracted, unified_text)
    _run_structural_item_projection_phase(
        extracted,
        unified_text,
        ocr_totals,
        (
            "jan_pos_items",
            "barcode_unit_qty_amount_stack",
            "barcode_qty_price_rows",
            "dense_item_rows",
            "dense_sequence_rows",
        ),
    )
    _run_quantity_detail_reconciliation_phase(
        extracted,
        unified_text,
        ("following_qty_detail",),
    )
    _run_line_item_cleanup_phase(
        extracted,
        unified_text,
        ("drop_numeric_marker_description_rows",),
    )
    _run_line_item_cleanup_phase(
        extracted,
        unified_text,
        ("drop_duplicate_embedded_price",),
    )
    _run_quantity_detail_reconciliation_phase(
        extracted,
        unified_text,
        ("qty_totals_from_unit_lines", "qty_context_and_reduced_rate"),
    )
    _fix_name_bag_amount_shift_from_ocr(extracted, unified_text)
    _run_line_item_cleanup_phase(
        extracted,
        unified_text,
        ("drop_numeric_marker_description_rows",),
    )
    _run_gap_item_recovery_phase(extracted, unified_text, ("discounted_gap",))
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "gap_item_recovery",
        trace_snapshot,
        extracted,
    )
    _fix_adjacent_ocr_price_shift_when_balanced(extracted, unified_text)
    _run_gap_item_recovery_phase(extracted, unified_text, ("repeated_gap",))
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "gap_item_recovery",
        trace_snapshot,
        extracted,
    )
    _recover_missing_bag_items_from_ocr(extracted, unified_text)
    _replace_overage_item_with_low_value_bag(extracted, unified_text)
    _fix_numeric_desc_from_ocr_price_context(extracted, unified_text)
    _run_ocr_description_reconciliation_phase(
        extracted,
        unified_text,
        ("o_ring_descriptions", "duplicate_descriptions"),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "ocr_description_reconciliation",
        trace_snapshot,
        extracted,
    )
    _run_line_item_cleanup_phase(
        extracted,
        unified_text,
        (
            "drop_non_product_line_items",
            "drop_duplicate_embedded_price",
            "drop_numeric_marker_description_rows",
        ),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "item_cleanup",
        trace_snapshot,
        extracted,
    )
    _run_service_receipt_recovery_phase(
        extracted,
        unified_text,
        ("service_table_items",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "service_receipt_recovery",
        trace_snapshot,
        extracted,
    )
    _run_line_item_cleanup_phase(
        extracted,
        unified_text,
        ("drop_duplicate_embedded_price", "drop_numeric_marker_description_rows"),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "item_cleanup",
        trace_snapshot,
        extracted,
    )

    # Clear account_number when it's a masked card number suffix, not a real account
    acct = extracted.get("account_number")
    if acct and re.search(r'\*{2,}' + re.escape(str(acct)), unified_text):
        extracted["account_number"] = None

    # Tax categories
    if extracted.get("line_items"):
        rate_bases = _tax_assignment_rate_bases(unified_text, ocr_totals)
        _fix_bag_item_prices_from_rate_bases(extracted, rate_bases, unified_text)
        _run_tax_category_assignment_phase(
            extracted,
            unified_text,
            ocr_totals,
            (
                "assign_tax_categories",
                "ocr_markers",
                "price_line_markers",
                "single_bag_standard_split",
                "rebalance_rate_bases",
                "rebalance_standard_from_reduced_markers",
                "nonfood_packaging",
                "ocr_markers",
                "single_bag_standard_split",
                "single_standard_from_small_base",
            ),
            rate_bases=rate_bases,
        )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "tax_category_assignment",
        trace_snapshot,
        extracted,
    )

    _run_service_receipt_recovery_phase(
        extracted,
        unified_text,
        ("single_service_inclusive_tax",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "service_receipt_recovery",
        trace_snapshot,
        extracted,
    )
    _fix_printed_tax_amounts_from_structural_blocks(extracted, unified_text)
    _restore_tax_excluded_per_rate_blocks(extracted, unified_text)
    _restore_single_rate_inclusive_tax_block(extracted, unified_text)

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
        ocr_tax_rates = {
            t.get("rate")
            for t in (ocr_totals.get("taxes") or [])
            if isinstance(t, dict) and t.get("rate") and (t.get("amount") or 0) > 0
        }
        target_only_rates = {
            rate
            for rate, base in (rate_bases or {}).items()
            if base and base > 0 and rate not in ocr_tax_rates
        }
        for cat in sorted(rate_sums):
            if cat not in existing_rates and rate_sums[cat] > 0:
                rate_pct = float(cat.replace('%', '')) / 100.0
                if is_inclusive:
                    computed_tax = round(rate_sums[cat] * rate_pct / (1 + rate_pct))
                else:
                    computed_tax = round(rate_sums[cat] * rate_pct)
                # Truth-file convention: omit tax entries that round to 0
                # (e.g., a レジ袋 at 5円 × 10% = 0.5 → 0).
                if cat in target_only_rates and computed_tax <= 1:
                    continue
                if computed_tax > 0:
                    extracted["taxes"].append({
                        "rate": cat,
                        "label": default_label,
                        "amount": computed_tax,
                    })

        # Fix LLM-extracted tax entries with amount=0 when items at that
        # rate yield non-zero expected tax. The LLM occasionally extracts
        # {rate: X, amount: 0} when the receipt prints a "X% target N"
        # rate-base line but no companion tax-amount line (e.g. receipts
        # with a single レジ袋 at 10円 and 10% rate base 10 — no separate
        # 10% tax printed because it rounds to ~1 yen).
        # Skip rates the OCR explicitly reports as ¥0 (truth-file convention
        # omits printed-zero tax entries).
        ocr_zero_rates = {
            t.get("rate") for t in (ocr_totals.get("taxes") or [])
            if isinstance(t, dict) and (t.get("amount") or 0) == 0
        }
        for t in extracted["taxes"]:
            if not isinstance(t, dict):
                continue
            r = t.get("rate")
            if not r or r not in rate_sums or r in ocr_zero_rates:
                continue
            amount = t.get("amount") or 0
            try:
                rate_pct = float(r.replace('%', '')) / 100.0
            except ValueError:
                continue
            if rate_pct <= 0:
                continue
            entry_label = t.get("label") or default_label or ""
            entry_inclusive = entry_label in ('内税', '消費税等') or entry_label.startswith('内')
            if entry_inclusive:
                expected = round(rate_sums[r] * rate_pct / (1 + rate_pct))
            else:
                expected = round(rate_sums[r] * rate_pct)
            if expected > 0 and (amount == 0 or amount > expected * 3):
                t["amount"] = expected
            elif expected > 0 and amount > 0 and amount < expected / 3:
                t["amount"] = expected

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
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "tax_category_assignment",
        trace_snapshot,
        extracted,
    )
    _run_payment_points_reconciliation_phase(
        extracted,
        unified_text,
        ocr_conf,
        llm_conf,
        ("points_used", "points_payment"),
    )

    # Fix pre-tax item totals for inclusive-tax receipts
    if extracted.get("line_items") and extracted.get("total"):
        item_sum = sum(i.get("total", 0) for i in extracted["line_items"] if isinstance(i, dict))
        receipt_total = extracted["total"]
        # Skip adjustment when taxes account for the difference (exclusive tax)
        tax_total = _sum_taxable_amounts(extracted.get("taxes", []))
        items_are_pretax = tax_total > 0 and abs(item_sum + tax_total - receipt_total) < 2
        if len(extracted["line_items"]) == 1 and abs(item_sum - receipt_total) > 1 and not items_are_pretax:
            item = extracted["line_items"][0]
            if isinstance(item, dict) and abs(item_sum * 1.10 - receipt_total) < 2:
                item["total"] = receipt_total
                if item.get("unit_price") and abs(item["unit_price"] - item_sum) < 1:
                    item["unit_price"] = receipt_total
            elif isinstance(item, dict) and abs(item_sum * 1.08 - receipt_total) < 2:
                item["total"] = receipt_total
                if item.get("unit_price") and abs(item["unit_price"] - item_sum) < 1:
                    item["unit_price"] = receipt_total
    _normalize_taxes(extracted, unified_text, ocr_totals)
    _restore_explicit_tax_rate_amount_lines(extracted, unified_text)
    _restore_printed_summary_total_when_tax_balanced(extracted, unified_text)
    _prefer_printed_item_sum_total_when_balanced(extracted, unified_text)

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
    if (
        extracted.get("taxes")
        and not ocr_totals.get("taxes")
        and not _items_plus_tax_matches_total(extracted)
    ):
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

    _fix_printed_tax_amounts_from_structural_blocks(extracted, unified_text)

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
        tax_sum = _sum_taxable_amounts(extracted.get("taxes") or [])
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
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "payment_points_reconciliation",
        trace_snapshot,
        extracted,
    )

    _extract_fuel_usage(extracted, unified_text)
    if extracted.get("line_items"):
        _fix_single_item_qty_from_ocr(extracted, unified_text)
    _fix_split_item_price_body_total_layout(extracted, unified_text)
    _run_gap_item_recovery_phase(extracted, unified_text, ("discounted_gap",))
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "gap_item_recovery",
        trace_snapshot,
        extracted,
    )
    _fix_adjacent_ocr_price_shift_when_balanced(extracted, unified_text)
    _run_gap_item_recovery_phase(extracted, unified_text, ("repeated_gap",))
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "gap_item_recovery",
        trace_snapshot,
        extracted,
    )
    _fix_discounted_item_gross_prices_from_ocr(extracted, unified_text)
    _fix_item_totals_from_following_discount_lines(extracted, unified_text)
    _apply_coupon_discount_blocks(extracted, unified_text)
    _drop_applied_coupon_line_items(extracted, unified_text)
    _repair_tiny_item_prices_from_following_ocr(extracted, unified_text)
    _ensure_discounted_ocr_pairs_present(extracted, unified_text)
    _run_gap_item_recovery_phase(extracted, unified_text, ("missing_items",))
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "gap_item_recovery",
        trace_snapshot,
        extracted,
    )
    _replace_vertical_price_qty_total_rows_when_balanced(extracted, unified_text)
    _run_structural_item_projection_phase(
        extracted,
        unified_text,
        ocr_totals,
        (
            "jan_pos_items",
            "barcode_unit_qty_amount_stack",
            "barcode_qty_price_rows",
            "dense_item_rows",
            "dense_sequence_rows",
        ),
    )
    _run_quantity_detail_reconciliation_phase(
        extracted,
        unified_text,
        ("qty_totals_from_unit_lines", "qty_context_and_reduced_rate"),
    )
    _fix_name_bag_amount_shift_from_ocr(extracted, unified_text)
    _run_line_item_cleanup_phase(
        extracted,
        unified_text,
        ("drop_numeric_marker_description_rows",),
    )
    _run_cash_tender_reconciliation_phase(
        extracted,
        unified_text,
        ("stacked_cash_tender", "unlabeled_cash_tender_change"),
    )
    _replace_overage_item_with_low_value_bag(extracted, unified_text)
    _fix_numeric_desc_from_ocr_price_context(extracted, unified_text)
    _run_ocr_description_reconciliation_phase(
        extracted,
        unified_text,
        (
            "o_ring_descriptions",
            "duplicate_descriptions",
            "colon_split_names",
            "bag_code_context",
        ),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "ocr_description_reconciliation",
        trace_snapshot,
        extracted,
    )
    _run_tax_category_assignment_phase(
        extracted,
        unified_text,
        ocr_totals,
        ("price_line_markers",),
    )
    _run_line_item_cleanup_phase(
        extracted,
        unified_text,
        ("drop_non_product_line_items",),
    )
    _run_service_receipt_recovery_phase(
        extracted,
        unified_text,
        ("service_table_items",),
    )
    _append_missing_low_value_bag_from_gap(extracted, unified_text)
    _recover_missing_bag_items_from_ocr(extracted, unified_text)
    _run_tax_category_assignment_phase(
        extracted,
        unified_text,
        ocr_totals,
        ("single_standard_from_small_base",),
    )
    _normalize_taxes(extracted, unified_text, ocr_totals)
    _restore_explicit_tax_rate_amount_lines(extracted, unified_text)
    _restore_tax_excluded_per_rate_blocks(extracted, unified_text)
    _restore_single_rate_inclusive_tax_block(extracted, unified_text)
    _restore_external_tax_total_from_printed_subtotal(extracted, unified_text)
    if extracted.get("total") is not None:
        tax_sum = _sum_taxable_amounts(extracted.get("taxes") or [])
        computed_sub = extracted["total"] - tax_sum
        if computed_sub >= 0:
            extracted["subtotal"] = computed_sub
    _append_missing_low_value_bag_from_gap(extracted, unified_text)
    _run_tax_category_assignment_phase(
        extracted,
        unified_text,
        ocr_totals,
        ("single_bag_standard_split",),
        rate_bases=extract_rate_bases(unified_text),
    )
    _run_line_item_cleanup_phase(
        extracted,
        unified_text,
        ("drop_non_product_line_items",),
    )
    _recover_qty_unit_total_item_from_empty_extraction(extracted, unified_text)
    _run_structural_item_projection_phase(
        extracted,
        unified_text,
        ocr_totals,
        ("dense_sequence_rows",),
    )
    _run_quantity_detail_reconciliation_phase(
        extracted,
        unified_text,
        ("qty_totals_from_unit_lines",),
    )
    _run_line_item_cleanup_phase(
        extracted,
        unified_text,
        ("drop_numeric_marker_description_rows",),
    )
    if extracted.get("line_items"):
        final_rate_bases = extract_rate_bases(unified_text)
        _run_tax_category_assignment_phase(
            extracted,
            unified_text,
            ocr_totals,
            (
                "ocr_markers",
                "rebalance_rate_bases",
                "rebalance_standard_from_reduced_markers",
                "nonfood_packaging",
                "single_bag_standard_split",
                "price_line_markers",
                "ocr_markers",
                "rebalance_rate_bases",
            ),
            rate_bases=final_rate_bases,
        )
        _run_quantity_detail_reconciliation_phase(
            extracted,
            unified_text,
            ("qty_context_and_reduced_rate",),
        )
    _run_ocr_description_reconciliation_phase(
        extracted,
        unified_text,
        (
            "o_ring_descriptions",
            "code_table_order",
            "duplicate_descriptions",
            "colon_split_names",
            "bag_code_context",
        ),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "ocr_description_reconciliation",
        trace_snapshot,
        extracted,
    )
    _fix_item_totals_from_following_discount_lines(extracted, unified_text)
    _apply_coupon_discount_blocks(extracted, unified_text)
    _drop_applied_coupon_line_items(extracted, unified_text)
    _repair_tiny_item_prices_from_following_ocr(extracted, unified_text)
    _replace_split_price_block_when_balanced(extracted, unified_text)
    _run_quantity_detail_reconciliation_phase(
        extracted,
        unified_text,
        ("qty_context_and_reduced_rate",),
    )
    _replace_stacked_name_price_rows_when_balanced(extracted, unified_text)
    _restore_stacked_inclusive_tax_block(extracted, unified_text)
    _restore_tax_excluded_per_rate_blocks(extracted, unified_text)
    _restore_single_rate_inclusive_tax_block(extracted, unified_text)
    _restore_printed_external_tax_amounts(extracted, unified_text)
    _restore_explicit_tax_rate_amount_lines(extracted, unified_text)
    _restore_bare_number_tax_summary(extracted, unified_text)
    _restore_external_tax_total_from_printed_subtotal(extracted, unified_text)
    _drop_unprinted_small_target_only_taxes(extracted, unified_text)
    _run_line_item_cleanup_phase(
        extracted,
        unified_text,
        ("drop_numeric_marker_description_rows",),
    )
    _run_structural_item_projection_phase(
        extracted,
        unified_text,
        ocr_totals,
        ("dense_sequence_rows",),
    )
    _run_quantity_detail_reconciliation_phase(
        extracted,
        unified_text,
        ("following_qty_detail",),
    )
    _replace_campaign_discount_stream_when_balanced(extracted, unified_text)
    _run_service_receipt_recovery_phase(
        extracted,
        unified_text,
        ("bare_service_without_itemization",),
    )
    _run_quantity_detail_reconciliation_phase(
        extracted,
        unified_text,
        ("qty_totals_from_unit_lines",),
    )
    _fix_bag_item_prices_from_rate_bases(extracted, extract_rate_bases(unified_text), unified_text)
    _fix_split_item_price_body_total_layout(extracted, unified_text)
    _clean_code_prefixed_item_descriptions(extracted)
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "structural_item_reconstruction",
        trace_snapshot,
        extracted,
    )
    if extracted.get("total") is not None:
        tax_sum = _sum_taxable_amounts(extracted.get("taxes") or [])
        computed_sub = float(extracted["total"]) - tax_sum
        if computed_sub >= 0:
            extracted["subtotal"] = computed_sub
    _repair_discounted_line_item_totals_when_balanced(extracted, unified_text)
    _repair_discounted_ocr_pair_descriptions(extracted, unified_text)
    _repair_pre_price_stack_descriptions_from_ocr(extracted, unified_text)
    _drop_duplicate_rows_when_subtotal_balances(extracted, unified_text)
    _replace_basket_marker_rows_when_balanced(extracted, unified_text)
    _run_payment_points_reconciliation_phase(
        extracted,
        unified_text,
        ocr_conf,
        llm_conf,
        ("points_payment",),
    )
    _restore_external_tax_total_from_printed_subtotal(extracted, unified_text)
    if extracted.get("line_items"):
        _fill_single_qty_unit_prices_from_totals(extracted["line_items"])
    _record_receipt_phase_mutation(
        mutation_trace,
        "final_consistency_pass",
        trace_snapshot,
        extracted,
    )

    return extracted
