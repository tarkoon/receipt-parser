"""Receipt tax category assignment helpers."""

import re
from difflib import SequenceMatcher
from itertools import combinations

from .patterns import (
    _PAID_CONTAINER_DESC_RE,
    _has_service_inclusive_tax_evidence,
    _is_service_fee_description,
)
from .receipt_financial import (
    _find_subset_sum,
    _jpy_summary_amount_options,
    extract_financial_totals,
    extract_rate_bases,
    normalize_tax_rate,
)
from .receipt_item_repair import _ocr_line_index_for_item
from .receipt_projection import _layout_row_price_candidates, _norm_layout_desc
from .schema import REDUCED_RATE, STANDARD_RATE, VALID_TAX_RATES


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
    item_sum = sum(
        float(item.get("total") or 0)
        for item in items
        if isinstance(item, dict)
    )
    nonzero_rate_bases = {
        rate: float(base)
        for rate, base in rate_bases.items()
        if rate in valid_rates and base is not None and float(base or 0) > 0
    }
    single_standard_base_covers_items = (
        item_sum > 0
        and set(nonzero_rate_bases) == {STANDARD_RATE}
        and abs(item_sum - nonzero_rate_bases[STANDARD_RATE]) <= 2
    )
    if re.search(r'軽減税率.*8%', unified_text) and not single_standard_base_covers_items:
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


def _is_bag_description(desc: str | None) -> bool:
    return bool(_PAID_CONTAINER_DESC_RE.search(desc or ""))


def _fix_tax_categories_from_ocr_markers(
    items,
    unified_text,
    *,
    stacked_only: bool = False,
    locked_indices: set[int] | None = None,
):
    """Use visible reduced-tax markers next to OCR item prices."""
    if not items:
        return
    lines = unified_text.split('\n')

    def _norm(text: str) -> str:
        text = re.sub(r'[¥￥]?\s*\d[\d,]*\s*(?:[%％][*※除軽]|[*※除軽非内外xX])?\s*$', '', text or "")
        text = re.sub(r'\s+', '', text)
        text = re.sub(r'[^\wぁ-んァ-ン一-龥]', '', text, flags=re.UNICODE)
        return text.lower()

    norm_lines = [_norm(line) for line in lines]

    marker_footnote = re.search(
        r'([*＊※☆★A-Za-z])\s*(?:印|マーク)?.{0,12}軽減税率|'
        r'軽減税率.{0,12}([*＊※☆★A-Za-z])\s*(?:印|マーク)?',
        unified_text,
    )
    reduced_markers = {marker for marker in (marker_footnote.groups() if marker_footnote else ()) if marker}
    reduced_marker_chars = ''.join(re.escape(marker) for marker in reduced_markers)
    marker_strip_chars = r'\*＊※☆★' if any(marker in '*＊※☆★' for marker in reduced_markers) else reduced_marker_chars
    has_reduced_marker_footnote = bool(reduced_marker_chars)

    def _is_code_continuation(line: str) -> bool:
        return bool(re.fullmatch(r'\d{6,14}\s*(?:JAN)?', line or "", re.IGNORECASE))

    def _is_price_row(line: str) -> bool:
        return bool(re.fullmatch(r'[¥￥]?\s*\d[\d,]*\s*(?:[%％][*※除軽外内]|[*※除軽外内非xX])?\s*$', line or ""))

    def _row_marker_category(line: str) -> str | None:
        line = (line or "").strip()
        if (
            re.search(r'(?:^|\s)\d+\s+[*＊]\s*$', line)
            or re.search(
                r'(?:^|\s)\d+\s*[*＊]\s*[¥￥]?\s*\d[\d,]*\s*$',
                line,
            )
        ):
            return None
        if re.search(r'\d[\d,]*\s*非\s*$', line):
            return "0%"
        if re.search(r'\d[\d,]*\s*[除内外]\s*$', line):
            return "10%"
        if re.search(r'(?:^|\s|\d)[*＊※]\s*|\d[\d,]*\s*軽\s*$', line):
            return "8%"
        if has_reduced_marker_footnote and re.search(r'\d[\d,]*\s*[xX]\s*$', line):
            return "8%"
        if has_reduced_marker_footnote and re.match(rf'^[{reduced_marker_chars}]\s*', line):
            return "8%"
        return None

    def _has_marked_modifier_price(item: dict) -> bool:
        if not has_reduced_marker_footnote:
            return False
        description = _norm(item.get("description") or "")
        try:
            item_total = float(item.get("total") or 0)
        except (TypeError, ValueError):
            return False
        if not description or item_total <= 0:
            return False
        matching_rows = 0
        # ponytail: receipt-sized item-by-line scan avoids another row aligner.
        for line_idx, line in enumerate(lines):
            marker_match = re.match(
                rf'^[{reduced_marker_chars}]\s*([ぁ-んァ-ン一-龥A-Za-z].*)$',
                line.strip(),
            )
            if not marker_match:
                continue
            modifier = _norm(marker_match.group(1))
            for nearby in lines[line_idx + 1:line_idx + 3]:
                nearby = nearby.strip()
                if not nearby:
                    continue
                if not _is_price_row(nearby):
                    break
                amount_match = re.search(r'\d[\d,]*', nearby)
                if amount_match and abs(float(amount_match.group(0).replace(',', '')) - item_total) <= 2:
                    if len(modifier) >= 2 and modifier in description:
                        return True
                    matching_rows += 1
                break
        same_total_items = 0
        for candidate in items:
            if not isinstance(candidate, dict):
                continue
            try:
                candidate_total = float(candidate.get("total") or 0)
            except (TypeError, ValueError):
                continue
            if abs(candidate_total - item_total) <= 2:
                same_total_items += 1
        return matching_rows == 1 and same_total_items == 1

    def _precedes_marked_stacked_reduced_item(line_idx: int) -> bool:
        if not has_reduced_marker_footnote:
            return False
        saw_code = False
        for nearby in lines[line_idx + 1:line_idx + 5]:
            nearby = nearby.strip()
            if not nearby:
                continue
            if _is_price_row(nearby):
                return False
            if _is_code_continuation(nearby):
                saw_code = True
                continue
            if saw_code and re.match(rf'^[{reduced_marker_chars}]\s*[ぁ-んァ-ン一-龥A-Za-z]', nearby):
                return True
            if saw_code and re.search(r'[ぁ-んァ-ン一-龥A-Za-z]', nearby):
                return False
        return False

    def _inherits_adjacent_description_marker(
        item: dict,
        line_idx: int,
        *,
        require_reconstruction: bool,
    ) -> bool:
        """Accept only an immediately adjacent marked text row as item evidence."""
        prior_idx = next(
            (idx for idx in range(line_idx - 1, -1, -1) if lines[idx].strip()),
            None,
        )
        if prior_idx is None:
            return False
        marker_match = re.match(
            rf'^[{reduced_marker_chars}]\s*([ぁ-んァ-ン一-龥A-Za-z].*)$',
            lines[prior_idx].strip(),
        )
        if not marker_match:
            return False
        if not require_reconstruction:
            return True
        marker_desc = _norm(marker_match.group(1))
        current_desc = norm_lines[line_idx]
        item_desc = _norm(item.get("description") or "")
        combined = marker_desc + current_desc
        return bool(
            marker_desc
            and current_desc
            and item_desc
            and (
                combined in item_desc
                or (
                    item_desc in combined
                    and item_desc not in marker_desc
                    and item_desc not in current_desc
                )
                or (marker_desc in item_desc and current_desc in item_desc)
            )
        )

    best_matches: dict[int, tuple[int, float]] = {}
    row_owners: dict[int, list[int]] = {}
    for item_idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        desc = _norm(item.get("description") or "")
        if len(desc) < 3:
            continue
        best_idx = None
        best_key = (0.0, 0, 0)
        for line_idx, nline in enumerate(norm_lines):
            if len(nline) < 3:
                continue
            matcher = SequenceMatcher(None, desc, nline)
            if desc == nline:
                score, overlap, ownership = 1.0, len(desc), 3
            elif desc in nline:
                score, overlap, ownership = 1.0, len(desc), 2
            elif nline in desc:
                score, overlap, ownership = 1.0, len(nline), 1
            else:
                score = matcher.ratio()
                overlap = matcher.find_longest_match().size
                ownership = 0
            match_key = (score, overlap, ownership)
            if match_key > best_key:
                best_idx = line_idx
                best_key = match_key
        if best_idx is not None and best_key[0] >= 0.72:
            best_matches[item_idx] = (best_idx, best_key[0])
            row_owners.setdefault(best_idx, []).append(item_idx)

    def _assign(item_idx: int, rate: str) -> None:
        items[item_idx]["tax_category"] = rate
        if locked_indices is not None:
            locked_indices.add(item_idx)

    line_owners = {
        line_idx: item_indices[0]
        for line_idx, item_indices in row_owners.items()
        if len(item_indices) == 1
    }
    line_idx = 0
    while line_idx < len(lines):
        if line_idx not in line_owners:
            line_idx += 1
            continue
        description_indices = []
        while line_idx < len(lines) and line_idx in line_owners:
            description_indices.append(line_owners[line_idx])
            line_idx += 1
        if len(description_indices) < 2:
            continue
        price_lines = lines[line_idx:line_idx + len(description_indices)]
        if len(price_lines) != len(description_indices) or not all(
            _is_price_row(price_line.strip()) for price_line in price_lines
        ):
            continue
        for item_idx, price_line in zip(description_indices, price_lines):
            category = _row_marker_category(price_line)
            if category is not None:
                _assign(item_idx, category)
        line_idx += len(description_indices)

    for item_idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        raw_desc = item.get("description") or ""
        if _has_marked_modifier_price(item):
            _assign(item_idx, "8%")
            continue
        if not stacked_only:
            if _is_service_fee_description(raw_desc) and _has_service_inclusive_tax_evidence(unified_text):
                _assign(item_idx, "10%")
                continue
            if has_reduced_marker_footnote:
                marker_cleaned = re.sub(rf'^[{marker_strip_chars}]\s*', '', raw_desc).strip()
                if marker_cleaned != raw_desc and re.search(r'[ぁ-んァ-ン一-龥]', marker_cleaned):
                    item["description"] = marker_cleaned
                    raw_desc = marker_cleaned
        match = best_matches.get(item_idx)
        if match is None:
            continue
        best_idx, _best_score = match
        if len(row_owners.get(best_idx, ())) != 1:
            continue
        line = lines[best_idx].strip()
        if re.match(r'^内\s*\*', line):
            _assign(item_idx, "8%")
            continue
        if has_reduced_marker_footnote and _inherits_adjacent_description_marker(
            item,
            best_idx,
            require_reconstruction=stacked_only,
        ):
            _assign(item_idx, "8%")
            continue
        if _precedes_marked_stacked_reduced_item(best_idx):
            _assign(item_idx, "8%")
            continue
        category = _row_marker_category(line)
        if category is None and best_idx + 1 < len(lines):
            next_line = lines[best_idx + 1].strip()
            if _is_price_row(next_line):
                category = _row_marker_category(next_line)
        if category is not None:
            _assign(item_idx, category)


def _assign_single_standard_rate_from_small_base(items, rate_bases):
    """Assign 10% only when one item total uniquely matches the printed base."""
    if not items or not rate_bases:
        return
    standard_base = float(rate_bases.get("10%") or 0)
    if standard_base <= 0:
        return
    valid_items = [item for item in items if isinstance(item, dict)]
    if not valid_items or any(item.get("tax_category") == "10%" for item in valid_items):
        return
    total_matches = [
        item for item in valid_items
        if abs(float(item.get("total") or 0) - standard_base) <= 2
    ]
    if len(total_matches) == 1:
        total_matches[0]["tax_category"] = "10%"


def _assign_unique_inner_rate_targets(items, unified_text):
    """Apply an explicit inner-tax target only to one unique item-total match."""
    if not items or not unified_text:
        return
    valid_items = [item for item in items if isinstance(item, dict)]
    pending: dict[int, set[str]] = {}
    lines = unified_text.splitlines()
    for idx, raw_line in enumerate(lines):
        line = re.sub(r'\s+', '', raw_line)
        target = re.search(
            r'(?:(\d+(?:\.\d+)?)[%％年](?:内税|内消費税)|'
            r'(?:内税|内消費税)(\d+(?:\.\d+)?)[%％年])'
            r'(?:対象|タイショウ)(?:額)?',
            line,
        )
        if not target:
            continue
        rate = normalize_tax_rate((target.group(1) or target.group(2)) + "%")
        if rate not in VALID_TAX_RATES or rate == "0%":
            continue
        amount_match = re.search(r'[¥￥]\s*([\d,]+)', raw_line)
        if not amount_match:
            following = next(
                (candidate.strip() for candidate in lines[idx + 1:idx + 3] if candidate.strip()),
                "",
            )
            amount_match = re.fullmatch(r'[¥￥]\s*([\d,]+)\s*[)）]?', following)
        if not amount_match:
            continue
        amount = float(amount_match.group(1).replace(',', ''))
        matches = []
        for item_idx, item in enumerate(valid_items):
            try:
                total = float(item.get("total") or 0)
            except (TypeError, ValueError):
                continue
            if total > 0 and abs(total - amount) <= 2:
                matches.append(item_idx)
        if len(matches) == 1:
            pending.setdefault(matches[0], set()).add(rate)
    for item_idx, rates in pending.items():
        if len(rates) == 1:
            valid_items[item_idx]["tax_category"] = next(iter(rates))


def _lock_binary_tax_column_when_balanced(
    items,
    unified_text,
    rate_bases,
    tax_amounts,
    locked_indices,
):
    """Map repeated 0/1 item flags only when one rate-sum mapping balances."""
    lines = [line.strip() for line in unified_text.splitlines()]
    starts = [idx for idx, line in enumerate(lines) if re.match(r'^商品名(?:\s|$)', line)]
    if len(starts) != len(items) or len(starts) < 3:
        return
    summary_start = next(
        (
            idx for idx in range(starts[-1] + 1, len(lines))
            if re.fullmatch(r'合\s*計|購入点数|税率', lines[idx])
        ),
        len(lines),
    )
    flags = []
    for pos, start in enumerate(starts):
        stop = starts[pos + 1] if pos + 1 < len(starts) else summary_start
        matches = []
        for line in lines[start + 1:stop]:
            match = re.fullmatch(r'[¥￥]?\s*[\d,]+\s+([01])|([01])', line)
            if match:
                matches.append(match.group(1) or match.group(2))
        if len(set(matches)) != 1:
            return
        flags.append(matches[0])

    flag_sums = {flag: 0.0 for flag in set(flags)}
    if set(flag_sums) != {"0", "1"}:
        return
    for item, flag in zip(items, flags):
        try:
            flag_sums[flag] += float(item.get("total") or 0)
        except (AttributeError, TypeError, ValueError):
            return

    targets = []
    for include_tax in (False, True):
        candidate = {
            rate: float(rate_bases.get(rate) or 0)
            + (float(tax_amounts.get(rate) or 0) if include_tax else 0)
            for rate in ("8%", "10%")
        }
        if all(value > 0 for value in candidate.values()):
            targets.append(candidate)
    mappings = []
    for target in targets:
        for zero_rate, one_rate in (("8%", "10%"), ("10%", "8%")):
            if (
                abs(flag_sums["0"] - target[zero_rate]) <= 2
                and abs(flag_sums["1"] - target[one_rate]) <= 2
            ):
                mappings.append({"0": zero_rate, "1": one_rate})
    if len(mappings) != 1:
        return
    for idx, (item, flag) in enumerate(zip(items, flags)):
        item["tax_category"] = mappings[0][flag]
        item["_tax_category_locked"] = mappings[0][flag]
        locked_indices.add(idx)


def _refine_rate_bases_from_tax_amounts(
    rate_bases,
    unified_text,
    extracted_taxes,
    *,
    item_sum: float | None = None,
):
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

    positive_rates = [
        rate for rate in ("8%", "10%")
        if float(refined.get(rate) or 0) > 0 and float(tax_amounts.get(rate) or 0) >= 0
    ]
    if item_sum and len(positive_rates) == 2:
        low_rate, high_rate = sorted(positive_rates, key=lambda rate: float(refined[rate]))
        low_base = float(refined[low_rate])
        high_base = float(refined[high_rate])
        tax_sum = sum(float(tax_amounts.get(rate) or 0) for rate in positive_rates)
        if (
            high_base > low_base
            and (
                abs(high_base - item_sum) <= 2
                or abs(high_base - tax_sum - item_sum) <= 2
            )
        ):
            cumulative_candidates = []
            for inclusive in (False, True):
                first = low_base - (float(tax_amounts.get(low_rate) or 0) if inclusive else 0)
                second = high_base - low_base - (float(tax_amounts.get(high_rate) or 0) if inclusive else 0)
                if first > 0 and second > 0:
                    cumulative_candidates.append((abs(first + second - item_sum), first, second))
            if cumulative_candidates:
                error, first, second = min(cumulative_candidates)
                if error <= 2:
                    refined[low_rate] = first
                    refined[high_rate] = second

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


def _reconcile_layout_markers_to_rate_bases(
    items,
    unified_text,
    rate_bases,
    ocr_layout_blocks,
    locked_indices,
) -> bool:
    """Apply one full layout assignment backed by marker locks and both bases.

    Every item must match one aligned layout price row. Visible marker-column
    glyphs lock reduced-rate rows; OCR-missed markers may be filled only by one
    exact minimum-cardinality residual subset. Equal minima fail closed.
    """
    if (
        not ocr_layout_blocks
        or len(items) < 2
        or not re.search(r'(?:[*＊※].{0,16}軽減税率|軽減税率.{0,16}[*＊※])', unified_text)
    ):
        return False
    try:
        targets = {rate: float(rate_bases[rate]) for rate in ("8%", "10%")}
    except (KeyError, TypeError, ValueError):
        return False
    if any(target <= 0 for target in targets.values()):
        return False

    candidates = _layout_row_price_candidates(ocr_layout_blocks)
    if len(candidates) != len(items) or len(items) > 24:
        return False
    if any(not isinstance(item, dict) for item in items):
        return False
    item_descs = [_norm_layout_desc(item.get("description") or "") for item in items]
    candidate_descs = [_norm_layout_desc(candidate["description"]) for candidate in candidates]
    if (
        any(len(desc) < 3 for desc in item_descs + candidate_descs)
        or len(set(item_descs)) != len(item_descs)
        or len(set(candidate_descs)) != len(candidate_descs)
    ):
        return False

    amounts: list[float] = []
    marker_indices: set[int] = set()
    used_candidates: set[int] = set()
    for item_idx, (item, item_desc) in enumerate(zip(items, item_descs)):
        try:
            amount = float(item.get("total") or 0)
        except (TypeError, ValueError):
            return False
        if amount <= 0:
            return False
        matches = []
        for candidate_idx, (candidate, candidate_desc) in enumerate(zip(candidates, candidate_descs)):
            if abs(float(candidate["value"]) - amount) > 2:
                continue
            score = (
                1.0
                if item_desc in candidate_desc or candidate_desc in item_desc
                else SequenceMatcher(None, item_desc, candidate_desc).ratio()
            )
            if score >= 0.86:
                matches.append(candidate_idx)
        if len(matches) != 1 or matches[0] in used_candidates or matches[0] != item_idx:
            return False
        candidate_idx = matches[0]
        used_candidates.add(candidate_idx)
        amounts.append(amount)
        if candidates[candidate_idx].get("reduced_marker"):
            marker_indices.add(item_idx)
    if not marker_indices or len(marker_indices) == len(items):
        return False
    if abs(sum(amounts) - sum(targets.values())) > 2:
        return False

    marker_sum = sum(amounts[idx] for idx in marker_indices)
    residual = targets["8%"] - marker_sum
    if residual < -0.01:
        return False
    unmarked = [(idx, amounts[idx]) for idx in range(len(items)) if idx not in marker_indices]
    if abs(residual) <= 0.01:
        residual_indices: set[int] = set()
    else:
        residual_matches: list[set[int]] = []
        for size in range(1, min(len(unmarked), 9) + 1):
            residual_matches = [
                {idx for idx, _amount in combo}
                for combo in combinations(unmarked, size)
                if abs(sum(amount for _idx, amount in combo) - residual) <= 0.01
            ]
            if residual_matches:
                break
        if len(residual_matches) != 1:
            return False
        residual_indices = residual_matches[0]

    reduced_indices = marker_indices | residual_indices
    reduced_sum = sum(amounts[idx] for idx in reduced_indices)
    standard_sum = sum(amounts[idx] for idx in range(len(items)) if idx not in reduced_indices)
    if abs(reduced_sum - targets["8%"]) > 0.01 or abs(standard_sum - targets["10%"]) > 0.01:
        return False
    proposed = ["8%" if idx in reduced_indices else "10%" for idx in range(len(items))]
    if any(
        idx in locked_indices and items[idx].get("tax_category") != proposed[idx]
        for idx in range(len(items))
    ):
        return False
    for idx, rate in enumerate(proposed):
        items[idx]["tax_category"] = rate
    locked_indices.update(marker_indices)
    return True


def _rebalance_tax_categories_to_rate_bases(
    items,
    unified_text,
    extracted_taxes,
    rate_bases,
    *,
    ocr_layout_blocks=None,
):
    """Assign printed rate-base residuals without overwriting row evidence."""
    if not items or not isinstance(rate_bases, dict):
        return

    locked_indices = {
        idx for idx, item in enumerate(items)
        if isinstance(item, dict) and item.get("_tax_category_locked") in {"0%", "8%", "10%"}
    }
    for idx in locked_indices:
        items[idx]["tax_category"] = items[idx]["_tax_category_locked"]
    _fix_tax_categories_from_ocr_markers(
        items,
        unified_text,
        locked_indices=locked_indices,
    )

    item_sum = sum(float(item.get("total") or 0) for item in items if isinstance(item, dict))
    if re.search(r'小\s*計\s*\n\s*\d+\s*[%％]\s*対象額\s*\n\s*\d+\s*[%％]\s*税額', unified_text):
        printed_base_sum = sum(
            float(base or 0)
            for rate, base in rate_bases.items()
            if rate in {"8%", "10%"} and base is not None
        )
        if not printed_base_sum or abs(item_sum - printed_base_sum) > 2:
            return
    if len(items) == 1 and re.search(r'消費税率は\s*10\s*%', unified_text):
        items[0]["tax_category"] = "10%"
        return

    tax_amounts = {}
    for tax in extracted_taxes or []:
        if not isinstance(tax, dict) or not tax.get("rate"):
            continue
        try:
            amount = float(tax.get("amount") or 0)
        except (TypeError, ValueError):
            continue
        tax_amounts[normalize_tax_rate(str(tax["rate"]))] = amount
    rate_bases = dict(rate_bases)
    for match in re.finditer(
        r'\((\d{2})%対象\s*¥?\s*([\d,]+)\s*内税\s*¥?\s*([\d,]+)',
        unified_text,
        flags=re.S,
    ):
        rate = f"{int(match.group(1))}%"
        if rate in {"8%", "10%"}:
            rate_bases[rate] = float(match.group(2).replace(',', ''))
            tax_amounts[rate] = float(match.group(3).replace(',', ''))
    rate_bases = _refine_rate_bases_from_tax_amounts(
        rate_bases,
        unified_text,
        extracted_taxes,
        item_sum=item_sum,
    )
    valid_rates = [
        rate for rate in ("8%", "10%")
        if float(rate_bases.get(rate) or 0) > 0
    ]
    targets = {rate: float(rate_bases[rate]) for rate in valid_rates}
    if len(valid_rates) == 2:
        base_sum = sum(targets.values())
        tax_sum = sum(float(tax_amounts.get(rate) or 0) for rate in valid_rates)
        items_are_pretax = (
            item_sum > 0 and tax_sum > 0
            and abs(item_sum + tax_sum - base_sum) <= max(5, base_sum * 0.02)
        )
        if items_are_pretax:
            targets = {
                rate: targets[rate] - float(tax_amounts.get(rate) or 0)
                for rate in valid_rates
            }
    if _reconcile_layout_markers_to_rate_bases(
        items,
        unified_text,
        targets,
        ocr_layout_blocks,
        locked_indices,
    ):
        return True
    _lock_binary_tax_column_when_balanced(
        items,
        unified_text,
        rate_bases,
        tax_amounts,
        locked_indices,
    )

    locked_nontaxable_indices = {
        idx for idx in locked_indices
        if items[idx].get("tax_category") == "0%"
    }

    item_amounts = [
        (idx, float(item.get("total") or 0))
        for idx, item in enumerate(items)
        if (
            isinstance(item, dict)
            and float(item.get("total") or 0) > 0
            and idx not in locked_nontaxable_indices
        )
    ]
    if not item_amounts or len(item_amounts) > 32:
        return

    if len(valid_rates) == 1:
        rate = valid_rates[0]
        target = targets[rate]
        locked_sum = sum(
            amount for idx, amount in item_amounts
            if idx in locked_indices and items[idx].get("tax_category") == rate
        )
        unresolved = [(idx, amount) for idx, amount in item_amounts if idx not in locked_indices]
        if unresolved and abs(locked_sum - target) <= 2:
            other_rate = "10%" if rate == "8%" else "8%"
            for idx, _amount in unresolved:
                items[idx]["tax_category"] = other_rate
        return
    if len(valid_rates) != 2:
        return

    if any(target <= 0 for target in targets.values()):
        return

    current_sums = {
        rate: sum(
            amount for idx, amount in item_amounts
            if items[idx].get("tax_category") == rate
        )
        for rate in valid_rates
    }
    if all(abs(current_sums[rate] - targets[rate]) <= 2 for rate in valid_rates):
        return

    locked_sums = {
        rate: sum(
            amount for idx, amount in item_amounts
            if idx in locked_indices and items[idx].get("tax_category") == rate
        )
        for rate in valid_rates
    }
    residuals = {rate: targets[rate] - locked_sums[rate] for rate in valid_rates}
    if any(residual < -2 for residual in residuals.values()):
        return
    unresolved = [(idx, amount) for idx, amount in item_amounts if idx not in locked_indices]
    balance_tolerance = 5.0 if re.search(r'外税|タイショウ', unified_text) else 2.0
    if abs(sum(amount for _idx, amount in unresolved) - sum(residuals.values())) > balance_tolerance:
        return

    lines = unified_text.split('\n')

    def _has_visible_reduced_marker(idx: int) -> bool:
        line_idx = _ocr_line_index_for_item(lines, items[idx])
        if line_idx is None:
            return False
        return any(
            re.search(r'^[A-Z]?\s*[*＊※]|[*＊※]\s*[^\d\s]|[%％][*＊※除軽]', nearby.strip())
            for nearby in lines[max(0, line_idx - 2):min(len(lines), line_idx + 3)]
        )

    def _evidence_score(indices: list[int], rate: str) -> int:
        return sum(
            4 if rate == "8%" and _has_visible_reduced_marker(idx) else 0
            for idx in indices
        )

    def _unique_subset(
        candidates: list[tuple[int, float]],
        target: float,
        rate: str,
        tolerance: float,
    ) -> tuple[list[int] | None, bool]:
        best_match = None
        best_key = None
        ambiguous = False
        max_k = min(len(candidates), 9)
        for size in range(1, max_k + 1):
            found_at_size = False
            for combo in combinations(candidates, size):
                diff = abs(sum(amount for _idx, amount in combo) - target)
                if diff > tolerance:
                    continue
                found_at_size = True
                match = [idx for idx, _amount in combo]
                key = (-diff, _evidence_score(match, rate), -size)
                if best_key is None or key > best_key:
                    best_match = match
                    best_key = key
                    ambiguous = False
                elif key == best_key and match != best_match:
                    ambiguous = True
            if tolerance == 0 and found_at_size:
                break
        return (None, True) if ambiguous else (best_match, False)

    unresolved_indices = {idx for idx, _amount in unresolved}

    # Keep non-conflicting current assignments when one unique residual fill
    # makes both printed bases balance. A full reassignment is only the fallback.
    for rate in sorted(valid_rates, key=lambda candidate: targets[candidate] - current_sums[candidate]):
        other_rate = next(candidate for candidate in valid_rates if candidate != rate)
        needed = targets[rate] - current_sums[rate]
        if needed <= 2:
            continue
        candidates = [
            (idx, amount) for idx, amount in unresolved
            if items[idx].get("tax_category") != rate
        ]
        match, ambiguous = _unique_subset(candidates, needed, rate, 0.0)
        if match is None and not ambiguous:
            match, ambiguous = _unique_subset(candidates, needed, rate, 2.0)
        if match is None or ambiguous:
            continue
        matched = set(match)
        proposed_rate_indices = {
            idx for idx, _amount in item_amounts
            if items[idx].get("tax_category") == rate
        } | matched
        proposed_rate_sum = sum(
            amount for idx, amount in item_amounts
            if idx in proposed_rate_indices
        )
        proposed_other_sum = sum(
            amount for idx, amount in item_amounts
            if idx not in proposed_rate_indices
        )
        if (
            abs(proposed_rate_sum - targets[rate]) > 2
            or abs(proposed_other_sum - targets[other_rate]) > balance_tolerance
        ):
            continue
        for idx, _amount in unresolved:
            items[idx]["tax_category"] = rate if idx in proposed_rate_indices else other_rate
        return

    for rate in sorted(valid_rates, key=lambda candidate: residuals[candidate]):
        other_rate = next(candidate for candidate in valid_rates if candidate != rate)
        target = residuals[rate]
        if target <= 2:
            match, ambiguous = [], False
        else:
            match, ambiguous = _unique_subset(unresolved, target, rate, 0.0)
            if match is None and not ambiguous:
                match, ambiguous = _unique_subset(unresolved, target, rate, 2.0)
        if match is None or ambiguous:
            continue
        matched = set(match)
        matched_sum = sum(amount for idx, amount in unresolved if idx in matched)
        other_sum = sum(amount for idx, amount in unresolved if idx not in matched)
        if abs(matched_sum - target) > 2 or abs(other_sum - residuals[other_rate]) > balance_tolerance:
            continue
        reduced_indices = matched if rate == "8%" else unresolved_indices - matched
        for idx, _amount in unresolved:
            items[idx]["tax_category"] = "8%" if idx in reduced_indices else "10%"
        return


def _assign_tax_categories_from_opaque_suffix_groups(items, unified_text, rate_bases):
    """Map repeated opaque price suffixes only when rate-base sums identify them."""
    if not items or not unified_text or not rate_bases:
        return
    valid_items = [item for item in items if isinstance(item, dict)]
    if len(valid_items) != len(items) or len(valid_items) < 4:
        return

    targets = {
        rate: float(base)
        for rate, base in rate_bases.items()
        if rate in {REDUCED_RATE, STANDARD_RATE} and base and float(base) > 0
    }
    if len(targets) < 2:
        return

    marker_rows: list[tuple[float, str]] = []
    for raw_line in unified_text.split('\n'):
        match = re.fullmatch(
            r'[¥￥$]?\s*(\d{1,3}(?:[,.]\d{3})+|\d+)\s+([A-Za-z])\s*',
            raw_line.strip(),
        )
        if match:
            marker_rows.append((float(re.sub(r'[,.]', '', match.group(1))), match.group(2).upper()))
    if len(marker_rows) != len(valid_items):
        return

    item_amounts: list[float] = []
    for item in valid_items:
        try:
            amount = float(item.get("total") or 0)
        except (TypeError, ValueError):
            return
        if amount <= 0:
            return
        item_amounts.append(amount)
    if any(abs(item_amount - row_amount) > 2 for item_amount, (row_amount, _marker) in zip(item_amounts, marker_rows)):
        return

    marker_sums: dict[str, float] = {}
    marker_counts: dict[str, int] = {}
    for amount, marker in marker_rows:
        marker_sums[marker] = marker_sums.get(marker, 0) + amount
        marker_counts[marker] = marker_counts.get(marker, 0) + 1
    if len(marker_sums) != len(targets) or any(count < 2 for count in marker_counts.values()):
        return

    marker_rates = {
        marker: [rate for rate, target in targets.items() if abs(group_sum - target) <= 2]
        for marker, group_sum in marker_sums.items()
    }
    if any(len(rates) != 1 for rates in marker_rates.values()):
        return
    resolved_rates = [rates[0] for rates in marker_rates.values()]
    if len(set(resolved_rates)) != len(resolved_rates):
        return
    for item, (_amount, marker) in zip(valid_items, marker_rows):
        item["tax_category"] = marker_rates[marker][0]


def reconcile_tax_categories_from_rate_bases(
    extracted: dict,
    unified_text: str,
    *,
    ocr_layout_blocks=None,
) -> None:
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
    if _rebalance_tax_categories_to_rate_bases(
        items,
        unified_text,
        extracted.get("taxes"),
        rate_bases,
        ocr_layout_blocks=ocr_layout_blocks,
    ):
        return
    _assign_tax_categories_from_opaque_suffix_groups(items, unified_text, rate_bases)
    _assign_unique_inner_rate_targets(items, unified_text)
    _fix_tax_categories_from_price_line_markers(extracted, unified_text)
    _fix_tax_categories_from_ocr_markers(items, unified_text, stacked_only=True)
    for item in items:
        if isinstance(item, dict) and item.get("_tax_category_locked") in {"0%", "8%", "10%"}:
            item["tax_category"] = item["_tax_category_locked"]


def reconcile_single_rate_tax_entries_from_item_arithmetic(
    extracted: dict,
    unified_text: str,
) -> None:
    """Collapse stale tax splits when OCR and final items prove one rate.

    Trigger: every final item and every OCR percent token identify the same
    non-zero rate, while multiple extracted tax entries remain. Invariant: the
    replacement amount comes from one printed tax value and must exactly bridge
    the final item sum to total.
    """
    items = extracted.get("line_items") or []
    taxes = extracted.get("taxes") or []
    if not unified_text or not items or len(taxes) < 2:
        return

    item_rates: set[str] = set()
    item_sum = 0.0
    for item in items:
        if not isinstance(item, dict):
            return
        rate = normalize_tax_rate(str(item.get("tax_category") or ""))
        if rate not in VALID_TAX_RATES or rate == "0%":
            return
        item_rates.add(rate)
        try:
            item_sum += float(item.get("total") or 0)
        except (TypeError, ValueError):
            return
    if len(item_rates) != 1 or item_sum <= 0:
        return
    sole_rate = next(iter(item_rates))

    visible_rates = {
        rate
        for match in re.finditer(r'(\d+(?:\.\d+)?)\s*[%％]', unified_text)
        if (rate := normalize_tax_rate(match.group(1) + "%")) in VALID_TAX_RATES
        and rate != "0%"
    }
    if visible_rates != {sole_rate}:
        return

    current_rates: set[str] = set()
    matching_label = None
    for tax in taxes:
        if not isinstance(tax, dict):
            return
        rate = normalize_tax_rate(str(tax.get("rate") or ""))
        if rate not in VALID_TAX_RATES or rate == "0%":
            return
        try:
            amount = float(tax.get("amount") or 0)
        except (TypeError, ValueError):
            return
        if amount <= 0:
            return
        current_rates.add(rate)
        if rate == sole_rate and matching_label is None:
            matching_label = tax.get("label")
    if sole_rate not in current_rates or current_rates == {sole_rate}:
        return

    try:
        total = float(extracted["total"])
        subtotal = float(extracted["subtotal"])
    except (KeyError, TypeError, ValueError):
        return
    if abs(item_sum - subtotal) > 2:
        return

    labeled_amounts = {
        amount
        for match in re.finditer(
            r'(?:消費税(?:等|額)?|税額)[ \t　]*[:：]?[ \t　]*[¥￥]?[ \t　]*'
            r'(?<![\d.,])(\d{1,3}(?:[,.]\d{3})+|\d+)(?![\d.,])'
            r'(?![ \t　]*[%％])',
            unified_text,
        )
        for amount in _jpy_summary_amount_options(match.group(1))
        if amount > 0
    }
    if labeled_amounts:
        printed_amounts = labeled_amounts
    else:
        printed_amounts = set()
        for tax in extract_financial_totals(unified_text).get("taxes") or []:
            if (
                not isinstance(tax, dict)
                or normalize_tax_rate(str(tax.get("rate") or "")) != sole_rate
            ):
                continue
            try:
                amount = float(tax.get("amount"))
            except (TypeError, ValueError):
                continue
            if amount > 0:
                printed_amounts.add(amount)
    if len(printed_amounts) != 1:
        return
    printed_amount = next(iter(printed_amounts))
    if abs(item_sum + printed_amount - total) > 0.01:
        return
    extracted["taxes"] = [{
        "rate": sole_rate,
        "label": matching_label,
        "amount": printed_amount,
    }]


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
        if not _has_reduced_marker(item):
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
    marker_rows: list[tuple[float, str]] = []
    for raw in unified_text.split('\n'):
        line = raw.strip()
        if re.search(r'小計|合計|対象|消費税|支払|お釣り|ポイント', line):
            continue
        m = re.fullmatch(r'([*＊※]?)\s*[¥￥]?\s*([\d,]+)\s*(軽|[*＊※]|非|除|内)?', line)
        if not m:
            continue
        try:
            amount = float(m.group(2).replace(',', ''))
        except ValueError:
            continue
        marker_rows.append((amount, (m.group(1) or m.group(3) or "")))
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
            amount, _marker = marker_rows[idx]
            if abs(amount - total) <= 1:
                match_idx = idx
                break
        if match_idx is None:
            continue
        amount, marker = marker_rows[match_idx]
        if marker == "非":
            item["tax_category"] = "0%"
            changed = True
        elif marker in {"除", "内"}:
            item["tax_category"] = "10%"
            changed = True
        elif marker:
            item["tax_category"] = "8%"
            changed = True
        elif has_star_legend and item.get("tax_category") not in ("0%", "非課税"):
            item["tax_category"] = "10%"
            changed = True
        row_idx = match_idx + 1
    if changed:
        return
