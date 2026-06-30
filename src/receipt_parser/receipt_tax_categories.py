"""Receipt tax category assignment helpers."""

import re
from difflib import SequenceMatcher
from itertools import combinations

from .patterns import (
    _BAG_DESC_RE,
    _FOOD_DESC_RE,
    _has_service_inclusive_tax_evidence,
    _is_service_fee_description,
)
from .receipt_financial import (
    _find_subset_sum,
    extract_rate_bases,
    normalize_tax_rate,
)
from .receipt_item_repair import (
    _ocr_line_index_for_item,
    _qty_detail_owner_indices,
)
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
