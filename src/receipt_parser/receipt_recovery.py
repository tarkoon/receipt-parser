"""Receipt recovery and printed total repair helpers."""

import re
from collections import Counter
from difflib import SequenceMatcher
from itertools import combinations

from .patterns import (
    _GENERIC_DESC_MARKERS,
    _HEADER_LINE_RE,
    _JUNK_DESC_RE,
    _OCR_QTY_NOTATION_RE,
    _OCR_TRAILING_PRICE_RE,
    _SKIP_PRICE_LINE,
)
from .receipt_financial import _parse_amount_fragment, extract_rate_bases, normalize_tax_rate
from .receipt_item_cleanup import _fill_single_qty_unit_prices_from_totals
from .receipt_item_repair import _valid_ocr_item_desc
from .receipt_projection import (
    _clean_ocr_price_line_desc,
    _find_ocr_item_desc,
)
from .receipt_tax_categories import (
    _assign_single_standard_rate_from_small_base,
    _is_bag_description,
    _rebalance_tax_categories_to_rate_bases,
)
from .receipt_totals import (
    _canonical_subtotal_from_taxes,
    _line_items_sum,
    _sum_taxable_amounts,
)


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

    def _has_visible_tax_amount(rate: str, amount: float) -> bool:
        rate_num = re.escape(rate.rstrip("%"))
        lines = [line.strip() for line in unified_text.split("\n")]
        for idx, line in enumerate(lines):
            if not re.search(rf'{rate_num}\s*[%％]\s*.*税額', line) or "対象" in line:
                continue
            values: list[float] = []
            for nearby in lines[idx + 1:min(len(lines), idx + 5)]:
                vm = re.fullmatch(r'[¥￥]\s*([\d,]+)\s*[\)）]?', nearby)
                if vm:
                    values.append(float(vm.group(1).replace(",", "")))
                    continue
                if values or re.search(r'合\s*計|お預り|お釣|登録番号', nearby):
                    break
            if values and abs(min(values) - amount) <= 2:
                return True
        return False

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
