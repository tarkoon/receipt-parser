"""Financial and tax parsing helpers for receipt OCR text."""

import re
from itertools import combinations

from .patterns import YEN_INLINE, YEN_SUFFIX


_STOP_FINANCIAL = re.compile(
    r'小\s*計|現\s*計|お釣り|お釣銭|釣\s*銭|お預り|お預り金|^預$|支払い?方法|支払い?\s|現金|釣銭|クレジット'
)
_STOP_BASIC = re.compile(r'合\s*計|現\s*計|お釣り|お預り')
_STOP_TAX = re.compile(r'合\s*計|小\s*計|現\s*計|お釣り|お釣銭|釣\s*銭|お預り|お預り金')
_TOTALS_VALUE_RE = re.compile(r'^[¥￥]\s*[\d,]+\s*$')


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
