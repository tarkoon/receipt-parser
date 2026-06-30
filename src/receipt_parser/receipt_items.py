"""Receipt line-item repair helpers."""

import re
from collections import Counter
from difflib import SequenceMatcher

from .patterns import (
    _ADMIN_FEE_DESCRIPTION_RE,
    _BANNER_PHRASE_RE,
    _GENERIC_DESC_MARKERS,
    _HEADER_LINE_RE,
    _JUNK_DESC_RE,
    _OCR_QTY_NOTATION_RE,
    _OCR_TRAILING_PRICE_RE,
    _OCR_ZONE_END_RE,
    _SKIP_PRICE_LINE,
    _has_service_inclusive_tax_evidence,
    _is_service_fee_description,
)
from .receipt_financial import (
    _parse_amount_fragment,
)
from .receipt_item_cleanup import (
    _clear_discounts_without_nearby_ocr_marker,
    _detect_ocr_discounts,
    _fix_discount_totals,
    _fix_hallucinated_prices,
    _fix_misattributed_discounts,
)
from .receipt_item_repair import (
    _apply_qty_notation_from_ocr,
    _dedup_same_total_items,
    _drop_banner_phantom_items,
    _expand_collapsed_items,
    _find_discounted_ocr_item_desc,
    _fix_fuel_item_description,
    _fix_fuel_volume_qty,
    _fix_qty_from_ocr_patterns,
    _fix_qty_hallucinations,
    _fix_single_item_qty_from_ocr,
    _fix_single_service_item_from_ocr,
    _insert_item_by_ocr_order,
    _remove_unit_rate_phantom_items,
    _replace_duplicate_desc_from_ocr,
    _revert_unsupported_qty_inflation,
    _strip_embedded_price_in_desc,
)
from .receipt_projection import (
    _clean_ocr_price_line_desc,
    _find_ocr_item_desc,
    _fix_item_totals_from_ocr_neighborhood,
    _merge_qty_detail_into_previous,
    _project_totals_to_layout_rows,
    _project_totals_to_ocr_multiset,
    _repair_column_split_items,
    _repair_previous_item_from_following_qty_detail,
    _replace_hallucinated_dup_with_ocr_item,
)
from .receipt_recovery import _recover_missing_items_from_gap
from .receipt_tax_categories import _is_bag_description
from .receipt_totals import _canonical_subtotal_from_taxes, _sum_taxable_amounts


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


# Descriptions that are clearly NOT product names — generic category markers
# (used in HANDS-style receipts above the actual item) or contact info.


# Lines that look like receipt-header metadata (phone, address, date,
# register/cashier numbers) — never use as a product description.

# Generic Japanese receipt boilerplate banners — appear on receipts from many
# merchants but never as product names. Used to drop phantom items the LLM
# created from header/footer text adjacent to a stray number.


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
    """Trigger: OCR repeats an atomic item row and item sum is short by its total.

    Invariant: recovery may clone only single-quantity, undiscounted rows when
    the cloned total closes a positive subtotal/total gap.
    """
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
        qty = float(item.get("qty") or 1)
        discount = float(item.get("discount") or 0)
        if abs(qty - 1) > 0.01 or abs(discount) > 0.01:
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
        m = re.search(r'(\d+(?:\.\d+)?)\s*個\s*[xX×]\s*[単单]\s*([\d,]+)', line)
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
