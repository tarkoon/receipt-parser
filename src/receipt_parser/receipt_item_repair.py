"""Receipt OCR item repair helpers."""

import math
import re
from difflib import SequenceMatcher

from .patterns import (
    _BANNER_PHRASE_RE,
    _GENERIC_DESC_MARKERS,
    _HEADER_LINE_RE,
    _OCR_QTY_NOTATION_RE,
    _OCR_TRAILING_PRICE_RE,
    _OCR_ZONE_END_RE,
    _SKIP_PRICE_LINE,
)
from .receipt_financial import (
    _find_subset_sum,
    _parse_amount_fragment,
    extract_financial_totals,
)
from .receipt_projection import (
    _clean_ocr_price_line_desc,
    _group_layout_rows,
    _layout_block_center_y,
    _layout_block_height,
    _layout_price_value,
    _layout_row_price_candidates,
    _norm_layout_desc,
    _parse_qty_detail_total,
)

_PURPOSE_STEM_CHARS = r'ぁ-んァ-ン一-龥A-Za-z0-9ー・'
_FORMAL_PURPOSE_SUFFIX_RE = re.compile(
    rf'([{_PURPOSE_STEM_CHARS}]{{2,}})代(?:として|$|[\s。、，,）)])'
)
_SUMMARY_COUNT_DESC_RE = re.compile(
    r'^(?:(?:お|御)?買上(?:げ)?(?:商品)?(?:点数|商品数)|(?:商品)?点数|商品数)$'
)
_COUNT_EXTENDED_PRICE_RE = re.compile(
    r'^(\d+(?:\.\d+)?)\s*[個コ点]\s+'
    r'[¥￥]?\s*(\d[\d,]*(?:\.\d+)?)\s*円?$'
)


def _clean_code_prefixed_item_descriptions(extracted, unified_text="", source_layout=None):
    """Remove product codes with unique literal title and quantity ownership."""
    raw_items = extracted.get("line_items") or []
    items = [item for item in raw_items if isinstance(item, dict)]
    suffix_candidates = []
    code_owner_counts = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        desc = (item.get("description") or "").strip()
        if not desc:
            continue
        suffix = re.fullmatch(
            r'(?P<title>.+?)\s*(?:\(\s*(?P<paren>\d{8,})\s*\)'
            r'|（\s*(?P<fullwidth>\d{8,})\s*）'
            r'|\s+(?P<bare>\d{8,}))',
            desc,
        )
        if suffix:
            code = suffix.group("paren") or suffix.group("fullwidth") or suffix.group("bare")
            title = suffix.group("title").strip()
            suffix_candidates.append((item, title, code))
            code_owner_counts[code] = code_owner_counts.get(code, 0) + 1

    lines = [line.strip() for line in (unified_text or "").splitlines()]
    code_rows = {}
    for line_idx, line in enumerate(lines):
        row = re.fullmatch(r'(\d{8,})\s+(\d+(?:\.\d+)?)(?:\s*[個コ点])?', line)
        if row:
            code_rows.setdefault(row.group(1), []).append((line_idx, float(row.group(2))))

    suffix_updates = []
    for item, title, code in suffix_candidates:
        rows = code_rows.get(code, [])
        if code_owner_counts[code] != 1 or len(rows) != 1:
            break
        try:
            item_qty = item.get("qty")
            qty = float(1 if item_qty is None else item_qty)
        except (TypeError, ValueError, OverflowError):
            break
        if not (qty > 0 and abs(qty - rows[0][1]) <= 0.01):
            break

        title_idx = rows[0][0] - 1
        while title_idx >= 0:
            line = lines[title_idx]
            if _OCR_ZONE_END_RE.search(line):
                title_idx = -1
                break
            if (
                re.fullmatch(r'(?:[¥￥]\s*)?\d[\d,]*(?:\.\d+)?\s*(?:円)?\s*(?:外|内|軽|[*＊※])?', line)
                or re.fullmatch(r'(?:外|内|軽|[*＊※]|対象|非課税)', line)
            ):
                title_idx -= 1
                continue
            break
        if title_idx < 0:
            break
        printed_title = lines[title_idx]
        normalized_title = re.sub(r'\s+', '', title)
        if re.sub(r'\s+', '', printed_title) != normalized_title:
            category_title = re.fullmatch(r'[一-龥](?:\s*\d+)?\s+(.+)', printed_title)
            if not category_title or re.sub(r'\s+', '', category_title.group(1)) != normalized_title:
                break
        suffix_updates.append((item, title))

    if len(suffix_updates) != len(suffix_candidates):
        return
    prefix_updates = []
    # ponytail: same-row POS tables only; split-row SKU titles need their own owner proof.
    code_re = r"\d{3,}(?:-\d{3,})+"
    prefix_candidates = []
    marked_candidates = []
    for item in items:
        description = str(item.get("description") or "").strip()
        match = re.fullmatch(rf"(?P<code>{code_re})\s+(?P<title>.+)", description)
        if match:
            prefix_candidates.append((item, match.group("code"), match.group("title")))
        marked = re.fullmatch(
            r"(?P<code>\d{6,})\s*(?P<marker>[*＊※★#])\s*(?P<title>.+)",
            description,
        )
        if marked:
            marked_candidates.append((item, marked.group("code"), marked.group("marker"), marked.group("title").strip()))
    discount_re = re.compile(r"割\s*引|値\s*引|クーポン|discount|(?:^|\s)[-−－]\s*[¥￥]?\s*\d", re.I)
    count_re = re.compile(r"(?m)^(?:購入点数|(?:お|御)買上(?:商品数|点数|げ点数)|(?:商品)?点数|商品数)[ \t]*[:：]?[ \t]*(?:\r?\n[ \t]*)?([0-9,]+)(?:点)?[ \t]*$")
    subtotal_re = re.compile(r"(?m)^小[ \t]*計[ \t]*[:：]?[ \t]*(?:\r?\n[ \t]*)?[¥￥][ \t]*(\d{1,3}(?:,\d{3})+|\d+)[ \t]*$")
    groups = _group_layout_rows(source_layout) if source_layout else []
    if marked_candidates and (
        not source_layout or len(marked_candidates) != 1 or len(items) != 1
        or len(raw_items) != 1 or suffix_candidates or prefix_candidates
    ):
        return
    if source_layout and prefix_candidates:
        if len(prefix_candidates) != len(items) or len(items) < 2 or len(items) != len(raw_items):
            return
        items_by_code = {code: (item, title) for item, code, title in prefix_candidates}
        if len(items_by_code) != len(prefix_candidates):
            return

        source_rows, qty_xs, qty_heights = {}, [], []
        for group in groups:
            row_text = "".join(str(block.get("text") or "") for block in group).strip()
            if _OCR_ZONE_END_RE.search(row_text) or _SUMMARY_COUNT_DESC_RE.fullmatch(row_text):
                break
            if discount_re.search(row_text):
                return
        for row in _layout_row_price_candidates(source_layout):
            try:
                x, y = float(row["x"]), float(row["y"])
            except (KeyError, TypeError, ValueError, OverflowError):
                return
            owners = [(group, block) for group in groups for block in group
                      if float(block.get("x") or 0) == x and _layout_block_center_y(block) == y]
            if len(owners) != 1:
                return
            group, price = owners[0]
            left = [block for block in group if float(block.get("x") or 0) < x]
            currency = [block for block in left if str(block.get("text") or "").strip() in ("¥", "￥")]
            before = [block for block in left if block not in currency]
            code_blocks = [(i, block) for i, block in enumerate(before)
                           if re.fullmatch(code_re, str(block.get("text") or "").strip())]
            if not code_blocks:
                continue
            amount, price_text = row["value"], str(price.get("text") or "").strip()
            if (len(code_blocks) != 1 or code_blocks[0][0] != 0 or len(before) < 3
                    or len(currency) + price_text.startswith(("¥", "￥")) != 1
                    or _layout_price_value(price_text, allow_small=True) != amount
                    or row.get("gross") is not None or row.get("discount") not in (None, 0)):
                return
            qty_block = before[-1]
            qty_match = re.fullmatch(r"(\d+(?:\.\d+)?)(?:個|コ|点)?", str(qty_block.get("text") or "").strip())
            code = str(code_blocks[0][1].get("text") or "").strip()
            title = "".join(str(block.get("text") or "") for block in before[1:-1]).strip()
            if qty_match is None or not re.search(r"[A-Za-zぁ-んァ-ン一-龥]", title) or code in source_rows:
                return
            try:
                qty = float(qty_match.group(1))
            except (TypeError, ValueError, OverflowError):
                return
            if not (math.isfinite(qty) and qty > 0):
                return
            source_rows[code] = (title, qty, amount)
            qty_xs.append(float(qty_block.get("x") or 0))
            qty_heights.append(_layout_block_height(qty_block))

        primary_owners = {}
        for index, line in enumerate(lines):
            if _OCR_ZONE_END_RE.search(line) or _SUMMARY_COUNT_DESC_RE.fullmatch(line):
                break
            if discount_re.search(line):
                return
            match = re.fullmatch(rf"(?P<code>{code_re})\s+(?P<title>.+)", line)
            if match:
                primary_owners.setdefault(match.group("code"), []).append((index, match.group("title")))
        layout_codes = [str(group[0].get("text") or "").strip() for group in groups
                        if group and re.fullmatch(code_re, str(group[0].get("text") or "").strip())]
        if (set(source_rows) != set(items_by_code) or len(layout_codes) != len(set(layout_codes))
                or set(layout_codes) != set(items_by_code) or set(primary_owners) != set(items_by_code)
                or any(len(owners) != 1 for owners in primary_owners.values())
                or max(qty_xs) - min(qty_xs) > max(3, sorted(qty_heights)[len(qty_heights) // 2] * .25)):
            return

        counts, subtotals = count_re.findall(unified_text or ""), subtotal_re.findall(unified_text or "")
        if (len(counts) != 1 or len(subtotals) != 1
                or int(counts[0].replace(",", "")) != sum(row[1] for row in source_rows.values())
                or int(subtotals[0].replace(",", "")) != sum(row[2] for row in source_rows.values())):
            return

        for item, code, item_title in prefix_candidates:
            source_title, source_qty, source_amount = source_rows[code]
            if any(isinstance(item.get(field), bool) for field in ("qty", "unit_price", "total", "discount")):
                return
            try:
                qty, unit, total = float(item["qty"]), float(item["unit_price"]), float(item["total"])
                discount = float(item.get("discount") or 0)
            except (KeyError, TypeError, ValueError, OverflowError):
                return
            if (not all(math.isfinite(value) for value in (qty, unit, total, discount)) or qty <= 0
                    or abs(qty - source_qty) > .01 or abs(unit * qty - total) > .01
                    or abs(total - source_amount) > .01 or discount != 0
                    or item.get("discount_rate") not in (None, "") or item.get("gross") is not None
                    or re.findall(r"\d+", source_title) != re.findall(r"\d+", item_title)):
                return
            printed_title = primary_owners[code][0][1].strip()
            if re.sub(r"\s+", "", printed_title) != re.sub(r"\s+", "", item_title):
                match = re.fullmatch(r"(?P<title>.+?)\s+(?P<qty>\d+(?:\.\d+)?)(?:\s*[個コ点])?", printed_title)
                if match is None or abs(float(match.group("qty")) - source_qty) > .01:
                    return
                printed_title = match.group("title").strip()
            if re.sub(r"\s+", "", printed_title) != re.sub(r"\s+", "", item_title):
                return
            prefix_updates.append((item, printed_title))

    if marked_candidates:
        item, code, marker, title = marked_candidates[0]
        sale_end = next((index for index, line in enumerate(lines) if _OCR_ZONE_END_RE.match(line)), len(lines))
        sale_lines = lines[:sale_end]
        primary_re = re.compile(
            rf"^{re.escape(code)}\s*{re.escape(marker)}\s*{re.escape(title)}\s+[¥￥]\s*([\d,]+)$"
        )
        primary_rows = [(line, match) for line in sale_lines if (match := primary_re.fullmatch(line))]
        counts, subtotals = count_re.findall(unified_text or ""), subtotal_re.findall(unified_text or "")
        if (
            any(discount_re.search(line) for line in sale_lines) or len(primary_rows) != 1
            or sum(bool(re.match(r"\d{3,}.*[¥￥]\s*\d", line)) for line in sale_lines) != 1
            or sum(bool(re.search(rf"(?<!\d){re.escape(code)}(?!\d)", line)) for line in sale_lines) != 1
            or sum(title in line and re.search(r"[¥￥]\s*\d", line) is not None for line in sale_lines) != 1
            or len(counts) != 1 or len(subtotals) != 1 or int(counts[0].replace(",", "")) != 1
        ):
            return
        primary_match = primary_rows[0][1]
        sale_groups = []
        for group in groups:
            row_text = "".join(str(block.get("text") or "") for block in group).strip()
            if _OCR_ZONE_END_RE.match(row_text):
                break
            if discount_re.search(row_text):
                return
            sale_groups.append(group)
        if sum(bool(re.fullmatch(r"\d{3,}", str(group[0].get("text") or "").strip())
                    and re.search(r"[¥￥]\s*\d", "".join(str(block.get("text") or "") for block in group)))
               for group in sale_groups) != 1:
            return
        owners = [group for group in sale_groups if str(group[0].get("text") or "").strip() == code]
        if len(owners) != 1:
            return
        group = owners[0]
        prices = [block for index, block in enumerate(group)
                  if _layout_price_value(str(block.get("text") or "").strip(), allow_small=True) is not None
                  and (str(block.get("text") or "").strip().startswith(("¥", "￥"))
                       or (index and str(group[index-1].get("text") or "").strip() in ("¥", "￥")))]
        if len(prices) != 1:
            return
        price = prices[0]
        x = float(price.get("x") or 0)
        amount = int(primary_match.group(1).replace(",", ""))
        price_text = str(price.get("text") or "").strip()
        left = [block for block in group if float(block.get("x") or 0) < x]
        currency = [block for block in left if str(block.get("text") or "").strip() in ("¥", "￥")]
        before = [block for block in left if block not in currency]
        after = [block for block in group if float(block.get("x") or 0) > x]
        # Keep a distinct leading marker cell separate from the description's owned mark.
        marker_index = 2 if (len(before) > 2 and str(before[1].get("text") or "").strip() in ("※", "*", "＊")
                             and str(before[1].get("text") or "").strip() != marker) else 1
        source_title = "".join(str(block.get("text") or "") for block in before[marker_index+1:]).strip()
        if (
            len(before) < marker_index+2 or str(before[0].get("text") or "").strip() != code
            or str(before[marker_index].get("text") or "").strip() != marker
            or re.sub(r"\s+", "", source_title) != re.sub(r"\s+", "", title)
            or not _valid_ocr_item_desc(source_title) or after
            or sum(str(block.get("text") or "").strip() == code for g in sale_groups for block in g) != 1
            or sum(
                re.sub(r"\s+", "", title) in re.sub(r"\s+", "", "".join(str(block.get("text") or "") for block in g))
                for g in sale_groups
            ) != 1
            or len(currency) + price_text.startswith(("¥", "￥")) != 1
            or _layout_price_value(price_text, allow_small=True) != amount
            or int(subtotals[0].replace(",", "")) != amount
        ):
            return
        if any(isinstance(item.get(field), bool) for field in ("qty", "unit_price", "total", "discount")):
            return
        try:
            qty = float(item["qty"])
            unit = float(item["unit_price"])
            total = float(item["total"])
            discount = float(0 if item.get("discount") is None else item["discount"])
            subtotal = float(extracted["subtotal"])
            receipt_total = extracted.get("total")
            if isinstance(receipt_total, bool) or isinstance(extracted.get("subtotal"), bool):
                return
            if receipt_total is not None:
                receipt_total = float(receipt_total)
        except (KeyError, TypeError, ValueError, OverflowError):
            return
        amounts = (qty, unit, total, discount, subtotal) + (() if receipt_total is None else (receipt_total,))
        if (
            not all(math.isfinite(value) for value in amounts) or qty != 1 or abs(unit - total) > .01
            or abs(total - amount) > .01 or abs(subtotal - amount) > .01 or discount != 0
            or item.get("discount_rate") not in (None, "") or item.get("gross") is not None
        ):
            return
        prefix_updates.append((item, title))

    for item, title in suffix_updates + prefix_updates:
        item["description"] = title


def _formal_receipt_purpose_stems(unified_text):
    """Read exact item stems from explicit 但…代 purpose rows on a receipt."""
    if "領収" not in unified_text or "但" not in unified_text or "代" not in unified_text:
        return set()
    stems = set()
    # ponytail: purpose phrases are treated as one OCR line; split-line OCR can
    # graduate to layout-aware recovery if we find that failure mode.
    for line in unified_text.splitlines():
        if "但" not in line or "代" not in line:
            continue
        tail = line.split("但", 1)[1]
        tail = re.sub(r'^\s*し[、,。\s]+', '', tail)
        tail = re.sub(rf'^[^{_PURPOSE_STEM_CHARS}]+', '', tail)
        stems.update(
            re.sub(r'\s+', '', match.group(1))
            for match in _FORMAL_PURPOSE_SUFFIX_RE.finditer(tail)
        )
    return stems


def _single_formal_receipt_purpose_description(unified_text):
    """Read one purpose title only when no item table competes for the amount."""
    stems = _formal_receipt_purpose_stems(unified_text)
    if len(stems) != 1 or re.search(r'商品名|品名|明細|数量|単価|小\s*計', unified_text):
        return None
    purpose = next(iter(stems))
    for line in unified_text.splitlines():
        if not re.search(r'[¥￥]\s*\d', line):
            continue
        title = re.split(r'[¥￥]', line, maxsplit=1)[0].strip()
        purpose_match = _FORMAL_PURPOSE_SUFFIX_RE.fullmatch(title)
        if (
            _valid_ocr_item_desc(title) and not title.startswith('但')
            and not _HEADER_LINE_RE.search(title)
            and not _SKIP_PRICE_LINE.search(title)
            and title != purpose
            and not (purpose_match and purpose_match.group(1) == purpose)
        ):
            return None
    return purpose


def _clean_formal_receipt_purpose_suffix_descriptions(extracted, unified_text):
    """Drop formal receipt purpose suffixes when the OCR purpose line proves it."""
    stems = _formal_receipt_purpose_stems(unified_text)
    if not stems:
        return
    for item in extracted.get("line_items") or []:
        if not isinstance(item, dict):
            continue
        desc = (item.get("description") or "").strip()
        if not desc.endswith("代"):
            continue
        stem = desc[:-1].strip()
        if re.sub(r'\s+', '', stem) in stems:
            item["description"] = stem


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
    if _SUMMARY_COUNT_DESC_RE.fullmatch(text):
        return False
    if re.fullmatch(
        r'\s*(?:(?:[@＠]|単)\s*)?\d[\d,]*(?:\.\d+)?\s*'
        r'(?:円|点|個|コ|本|枚|袋|足|杯|組)\s*',
        text,
    ):
        return False
    if re.fullmatch(r'\s*\d+\s*(?:\u540d|\u4eba)(?:\s*(?:\u69d8|\u3055\u307e|\u30b5\u30de))?\s*', text):
        return False
    if _SKIP_PRICE_LINE.search(text):
        return False
    if re.search(r'割引|値引', text):
        return False
    if re.match(r'^[\d,\s\-\(\)\.\*※軽除外]+$', text):
        return False
    return bool(re.search(r'[ぁ-んァ-ン一-龥]', text))


def _has_local_qty_unit_evidence(item, unified_text):
    """Prove qty/unit from one description-owned OCR block and its arithmetic."""
    if not isinstance(item, dict):
        return False
    try:
        qty = float(item.get("qty") or 1)
        unit = float(item.get("unit_price") or 0)
        total = float(item.get("total") or 0)
        discount = float(item.get("discount") or 0)
    except (TypeError, ValueError):
        return False
    desc = _norm_layout_desc(item.get("description") or "")
    gross = qty * unit
    if (
        not desc
        or qty <= 0
        or unit <= 0
        or min(abs(gross - total), abs(gross - discount - total)) > 2
    ):
        return False

    lines = [line.strip() for line in unified_text.splitlines()]
    for owner_idx, line in enumerate(lines):
        visible_desc = _norm_layout_desc(_clean_ocr_price_line_desc(line))
        if not visible_desc or (visible_desc != desc and not (
            len(desc) >= 2 and (desc in visible_desc or visible_desc in desc)
        )):
            continue

        owned = [line]
        for nearby in lines[owner_idx + 1:min(len(lines), owner_idx + 6)]:
            if _OCR_ZONE_END_RE.search(nearby) or _HEADER_LINE_RE.search(nearby):
                break
            candidate = _clean_ocr_price_line_desc(nearby)
            is_detail = bool(
                _OCR_QTY_NOTATION_RE.search(nearby)
                or _COUNT_EXTENDED_PRICE_RE.fullmatch(nearby)
                or re.fullmatch(r'[¥￥]?\s*\d[\d,]*(?:\.\d+)?\s*円?', nearby)
                or re.fullmatch(r'\d+(?:\.\d+)?\s*[個コ点]', nearby)
                or re.fullmatch(
                    r'[xX×Ⅹ]\s*\d+(?:\.\d+)?\s+'
                    r'[¥￥]?\s*\d[\d,]*(?:\.\d+)?\s*円',
                    nearby,
                )
            )
            if _valid_ocr_item_desc(candidate) and not is_detail:
                break
            owned.append(nearby)
        window = "\n".join(owned)

        pairs: list[tuple[float, float]] = []
        pairs.extend(
            (float(match.group(1)), float(match.group(2).replace(',', '')))
            for match in re.finditer(
                r'(\d+(?:\.\d+)?)\s*[個コ点]\s*[xX×Ⅹ]\s*'
                r'(?:単|@|＠)?\s*[¥￥]?\s*(\d[\d,]*(?:\.\d+)?)',
                window,
            )
        )
        pairs.extend(
            (float(match.group(2)), float(match.group(1).replace(',', '')))
            for match in re.finditer(
                r'(?:単|@|＠)\s*[¥￥]?\s*(\d[\d,]*(?:\.\d+)?)\s*'
                r'[xX×Ⅹ]?\s*(\d+(?:\.\d+)?)\s*[個コ点]',
                window,
            )
        )
        if any(
            abs(seen_qty - qty) <= 0.01 and abs(seen_unit - unit) <= 1
            for seen_qty, seen_unit in pairs
        ):
            return True

        owner_price = _OCR_TRAILING_PRICE_RE.search(line)
        first_detail = next(
            (part.strip() for part in owned[1:] if part.strip()),
            "",
        )
        count_extended = _COUNT_EXTENDED_PRICE_RE.fullmatch(first_detail)
        if owner_price and count_extended:
            seen_unit = float(
                owner_price.group(1).strip().lstrip('¥￥').replace(',', '')
            )
            seen_qty = float(count_extended.group(1))
            seen_total = float(count_extended.group(2).replace(',', ''))
            if (
                abs(seen_qty - qty) <= 0.01
                and abs(seen_unit - unit) <= 1
                and min(abs(seen_total - total), abs(seen_total - gross)) <= 2
                and min(
                    abs(seen_qty * seen_unit - seen_total),
                    abs(seen_qty * seen_unit - discount - seen_total),
                ) <= 2
            ):
                return True

        nonempty = [part.strip() for part in owned[1:] if part.strip()]
        for first, second, third in zip(nonempty, nonempty[1:], nonempty[2:]):
            unit_match = re.fullmatch(r'[¥￥]\s*(\d[\d,]*(?:\.\d+)?)', first)
            qty_match = re.fullmatch(r'(\d+(?:\.\d+)?)\s*[個コ点]', second)
            total_match = re.fullmatch(
                r'[¥￥]\s*(\d[\d,]*(?:\.\d+)?)', third
            )
            if not (unit_match and qty_match and total_match):
                continue
            seen_unit = float(unit_match.group(1).replace(',', ''))
            seen_qty = float(qty_match.group(1))
            seen_total = float(total_match.group(1).replace(',', ''))
            if (
                abs(seen_qty - qty) <= 0.01
                and abs(seen_unit - unit) <= 1
                and min(abs(seen_total - total), abs(seen_total - gross)) <= 2
                and min(
                    abs(seen_qty * seen_unit - seen_total),
                    abs(seen_qty * seen_unit - discount - seen_total),
                ) <= 2
            ):
                return True

        for match in re.finditer(
            r'[xX×Ⅹ]\s*(\d+(?:\.\d+)?)\s+'
            r'[¥￥]?\s*(\d[\d,]*(?:\.\d+)?)\s*円',
            window,
        ):
            seen_qty = float(match.group(1))
            seen_total = float(match.group(2).replace(',', ''))
            if (
                seen_qty > 0
                and abs(seen_qty - qty) <= 0.01
                and min(abs(seen_total - total), abs(seen_total - gross)) <= 2
                and abs((seen_total / seen_qty) - unit) <= 1
            ):
                return True
    return False


def _fix_compact_count_amount_layout(items, ocr_layout_blocks):
    """Split one OCR token spanning count/amount columns when arithmetic proves it.

    Trigger: a table has separate quantity and amount headers, but OCR merged a
    row's count prefix with its extended total. Invariant: the row admits one
    description-owned split whose suffix divides evenly by its count and
    already equals the upstream row total, so basket arithmetic cannot change.
    """
    if not items or not ocr_layout_blocks:
        return

    def _x_bounds(block):
        xs = [
            float(point[0])
            for point in (block.get("bbox") or [])
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
        x = float(block.get("x") or 0)
        return (min(xs), max(xs)) if xs else (x, x)

    rows = _group_layout_rows(ocr_layout_blocks)
    proposals: dict[int, set[tuple[int, int, int]]] = {}
    for header_idx, header in enumerate(rows):
        qty_header = next(
            (
                block for block in header
                if re.fullmatch(r'(?:点数|数量|個数)', str(block.get("text") or "").strip())
            ),
            None,
        )
        amount_header = next(
            (
                block for block in header
                if str(block.get("text") or "").strip() == "金額"
            ),
            None,
        )
        if qty_header is None or amount_header is None:
            continue
        _, qty_right = _x_bounds(qty_header)
        amount_left, _ = _x_bounds(amount_header)
        if qty_right >= amount_left:
            continue
        boundary = (qty_right + amount_left) / 2
        page = qty_header.get("page", 0)

        for row in rows[header_idx + 1:]:
            if row[0].get("page", 0) != page:
                break
            row_text = "".join(str(block.get("text") or "") for block in row)
            if _OCR_ZONE_END_RE.search(row_text):
                break
            for token_idx, block in enumerate(row):
                raw = str(block.get("text") or "").strip()
                if not re.fullmatch(r'\d[\d,\s]*', raw):
                    continue
                left, right = _x_bounds(block)
                gap = amount_left - qty_right
                if not (
                    left <= qty_right + gap * 0.25
                    and left < boundary
                    and right >= amount_left
                ):
                    continue
                compact = re.sub(r'[\s,]', '', raw)
                row_desc = _norm_layout_desc(
                    "".join(str(part.get("text") or "") for part in row[:token_idx])
                )
                matches: list[tuple[float, int, int, int, int]] = []
                for item_idx, item in enumerate(items):
                    if not isinstance(item, dict) or item.get("discount") not in (None, 0, 0.0, "", "0", "0.0"):
                        continue
                    try:
                        current_qty = float(item.get("qty") or 1)
                        unit = float(item.get("unit_price") or 0)
                        total = float(item.get("total") or 0)
                    except (TypeError, ValueError):
                        continue
                    if (
                        current_qty != 1
                        or total <= 0
                        or not total.is_integer()
                        or abs(unit - total) > 1
                    ):
                        continue
                    item_desc = _norm_layout_desc(item.get("description") or "")
                    if len(item_desc) < 2 or not row_desc:
                        continue
                    score = SequenceMatcher(None, item_desc, row_desc).ratio()
                    if score < 0.72:
                        continue
                    splits: set[tuple[int, int, int]] = set()
                    for split_idx in range(1, min(3, len(compact))):
                        count_text = compact[:split_idx]
                        total_text = compact[split_idx:]
                        if total_text.startswith("0"):
                            continue
                        count = int(count_text)
                        extended_total = int(total_text)
                        if (
                            count <= 1
                            or extended_total <= 0
                            or extended_total % count
                            or int(total) != extended_total
                        ):
                            continue
                        splits.add((count, extended_total // count, extended_total))
                    if len(splits) == 1:
                        count, split_unit, extended_total = splits.pop()
                        matches.append(
                            (score, item_idx, count, split_unit, extended_total)
                        )
                if not matches:
                    continue
                best_score = max(match[0] for match in matches)
                best = [match for match in matches if abs(match[0] - best_score) <= 0.01]
                if len(best) == 1:
                    _, item_idx, count, unit, extended_total = best[0]
                    proposals.setdefault(item_idx, set()).add(
                        (count, unit, extended_total)
                    )

    for item_idx, item_proposals in proposals.items():
        if len(item_proposals) != 1:
            continue
        count, unit, extended_total = item_proposals.pop()
        items[item_idx]["qty"] = count
        items[item_idx]["unit_price"] = unit
        items[item_idx]["total"] = extended_total


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

    def literal_title(line):
        # Explicit currency owns the price suffix; title digits remain literal.
        value = re.sub(r'\s*[¥￥]\s*\d[\d,]*\s*[*＊※除軽外内]?\s*$', '', str(line or '')).strip()
        return ''.join(char.casefold() for char in value if char.isalnum())

    lines = (unified_text or '').splitlines()
    headings = [i for i, line in enumerate(lines)
                if 'まとめ' in line and _OCR_ZONE_END_RE.fullmatch(line.strip())]
    summary_only = set()
    if len(headings) == 1:
        start = headings[0]
        end = next((i for i in range(start + 1, len(lines))
                    if _OCR_ZONE_END_RE.match(lines[i].strip())), None)
        if end is not None:
            purchased = {literal_title(line) for line in lines[:start]}
            summary_only = {literal_title(line) for line in lines[start + 1:end]} - purchased
    kept = []
    for item in items:
        if not isinstance(item, dict):
            kept.append(item)
            continue
        desc = (item.get("description") or "").strip()
        if desc and (_BANNER_PHRASE_RE.search(desc) or literal_title(desc) in summary_only):
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
            if len(cand) < 3 or not _valid_ocr_item_desc(cand):
                continue
            if re.search(
                r'\d+(?:\.\d+)?\s*[個コ点]\s*[xX×Ⅹ]\s*(?:単|@)?\s*[\d,]+'
                r'|(?:単|@)?\s*[\d,]+\s*[xX×Ⅹ]\s*\d+(?:\.\d+)?\s*[個コ点]?',
                cand,
            ):
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


def _dedup_same_total_items(extracted, unified_text=""):
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

    # Separately printed qty=1 purchases stay separate even if another price
    # makes their deletion improve the arithmetic. Keep all title digits.
    def title_key(text):
        return ''.join(char.casefold() for char in (text or '') if char.isalnum())

    literal_owners = {}
    lines = [line.strip() for line in unified_text.split('\n')]
    price_pattern = r'[¥￥]\s*(\d[\d,]*)\s*[※*＊除軽非]?\s*$'
    for index, line in enumerate(lines):
        if _OCR_ZONE_END_RE.match(line):
            break
        price = re.search(price_pattern, line)
        title = line[:price.start()].strip() if price else line
        detail_index = index + 1
        if price is None and index + 1 < len(lines):
            price = re.fullmatch(price_pattern, lines[index + 1])
            detail_index += 1
        if (price is not None and _valid_ocr_item_desc(title)
                and not _OCR_QTY_NOTATION_RE.search(title)
                and not (detail_index < len(lines)
                         and _OCR_QTY_NOTATION_RE.search(lines[detail_index]))):
            owner = (title_key(title), float(price[1].replace(',', '')))
            literal_owners[owner] = literal_owners.get(owner, 0) + 1
    groups = {}
    for item in items:
        if isinstance(item, dict):
            groups.setdefault((item.get("description", ""), item.get("total", 0)), []).append(item)
    protected = set()
    for signature, group in groups.items():
        if (len(group) > 1 and all(row.get("qty", 1) == 1 and not row.get("discount")
                                  and row.get("unit_price") == row.get("total") for row in group)
                and literal_owners.get((title_key(signature[0]), signature[1]), 0) >= len(group)):
            protected.add(signature)

    keep_mask = [True] * len(items)
    seen: dict[tuple, int] = {}
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        key = (item.get("description", ""), item.get("total", 0))
        if key in protected:
            continue
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
    for item in items:
        if not isinstance(item, dict) or item.get("qty", 1) <= 1:
            continue
        total = item.get("total", 0)
        unit_price = item.get("unit_price")
        if unit_price is None:
            continue
        if _has_local_qty_unit_evidence(item, unified_text):
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
        if _has_local_qty_unit_evidence(item, unified_text):
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
        r'|[\(（<]?\s*(?:単|@)?\s*\d[\d,]*\s*[xX×]\s*\d+\s*[コ個点]'
        r'|(?:^|\s)\d+\s*(?:[*＊xX×])\s*$'
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
        if _has_local_qty_unit_evidence(item, unified_text):
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
        for offset in (0, -1, -2, 1, 2, 3, 4):
            j = match_li + offset
            if j < 0:
                continue
            if j >= len(ocr_lines):
                break
            nearby = ocr_lines[j].strip()
            if not nearby:
                continue
            if qty_re.search(nearby):
                has_qty = True
                break
            # Stop on next name (≥ 2 Japanese chars) without qty notation.
            if offset > 0 and re.search(r'[ぁ-んァ-ン一-龥]{2,}', nearby):
                break
        if has_qty:
            continue
        # No qty notation supports this qty>1 — revert to qty=1.
        visible_total = None
        visible_unit = None
        for nearby in ocr_lines[match_li + 1:min(len(ocr_lines), match_li + 4)]:
            amount_m = re.search(r'[¥￥]?\s*([\d,]+)\s*(?:外|内|軽|[*＊※])?\s*$', nearby.strip())
            if amount_m:
                try:
                    amount = float(amount_m.group(1).replace(',', ''))
                except ValueError:
                    amount = None
                if amount is not None and abs(amount - total) <= 1:
                    visible_total = amount
                    break
                if amount is not None and abs(amount - unit) <= 1:
                    visible_unit = amount
                    break
            if re.search(r'[ぁ-んァ-ン一-龥]{2,}', nearby):
                break
        if visible_total is None and visible_unit is None:
            total_text = str(int(total)) if float(total).is_integer() else str(total)
            total_comma = f"{int(total):,}" if float(total).is_integer() else total_text
            token = rf'(?:{re.escape(total_comma)}|{re.escape(total_text)})'
            visible_matches = re.findall(
                rf'(?:[¥￥]\s*{token}(?!\d)|(?<!\d){token}\s*[%％*＊※除軽外内])',
                unified_text,
            )
            if len(visible_matches) == 1:
                visible_total = total
        if visible_total is None and visible_unit is None:
            try:
                unit_is_fractional = abs(float(unit) - round(float(unit))) > 0.001
            except (TypeError, ValueError):
                unit_is_fractional = False
            total_text = str(int(total)) if float(total).is_integer() else str(total)
            if unit_is_fractional and re.search(rf'(?<!\d){re.escape(total_text)}(?!\d)', unified_text):
                visible_total = total
        item["qty"] = 1
        if visible_total is not None:
            item["total"] = visible_total
            item["unit_price"] = visible_total
            continue
        if visible_unit is not None:
            item["total"] = visible_unit
            item["unit_price"] = visible_unit
            continue
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

    # Match extracted rows to OCR descriptions once, in order. Repeated
    # descriptions then consume repeated OCR occurrences instead of all
    # borrowing the first row's quantity detail.
    owner_lines: dict[int, int] = {}
    cursor = 0
    for item_idx, item in enumerate(items):
        if not isinstance(item, dict):
            return
        desc_norm = _norm_layout_desc(item.get("description") or "")
        if len(desc_norm) < 4:
            return
        candidates: list[tuple[float, int]] = []
        for line_idx in range(cursor, len(ocr_lines)):
            line = _clean_ocr_price_line_desc(ocr_lines[line_idx])
            if not _valid_ocr_item_desc(line):
                continue
            line_norm = _norm_layout_desc(line)
            if desc_norm in line_norm or line_norm in desc_norm:
                score = 1.0
            else:
                score = SequenceMatcher(None, desc_norm, line_norm).ratio()
            if score >= 0.8:
                candidates.append((score, line_idx))
        if not candidates:
            return
        best_score = max(score for score, _line_idx in candidates)
        line_idx = min(
            line_idx
            for score, line_idx in candidates
            if abs(score - best_score) <= 0.01
        )
        owner_lines[item_idx] = line_idx
        cursor = line_idx + 1

    for item_idx, item in enumerate(items):
        current_qty = item.get("qty", 1)
        if isinstance(current_qty, bool):
            continue
        try:
            current_qty = float(current_qty)
        except (TypeError, ValueError):
            continue
        if current_qty != 1:
            continue
        cur_total = item.get("total", 0)
        line_idx = owner_lines[item_idx]
        next_owner = owner_lines.get(item_idx + 1, len(ocr_lines))
        for nearby_idx in range(line_idx + 1, min(next_owner, line_idx + 5)):
            nearby = ocr_lines[nearby_idx].strip()
            if not nearby:
                continue
            if (
                _OCR_ZONE_END_RE.match(nearby)
                or _HEADER_LINE_RE.search(nearby)
                or _BANNER_PHRASE_RE.search(nearby)
            ):
                break
            m = qty_re.search(nearby)
            if not m:
                continue
            qty = float(m.group(1))
            unit = float(m.group(2).replace(',', ''))
            if (
                qty >= 2
                and unit > 0
                and qty * unit > cur_total + 1
                and not (item.get("discount") or 0)
            ):
                item["qty"] = qty
                item["unit_price"] = unit
                item["total"] = qty * unit
            break


def _fix_qty_from_ocr_patterns(items, unified_text):
    """Fix quantities using ×N個 patterns and qty×price scanners in OCR text."""
    ocr_lines = unified_text.split('\n')
    initial_finances = [
        (item.get("qty"), item.get("unit_price"), item.get("total"))
        if isinstance(item, dict) else None for item in items
    ]

    # A dropped leading quantity digit can leave "コX単U". Recover it only
    # when the next local amount is an exact multiple of U and one nearby
    # extracted description owns the row.
    missing_qty_updates: dict[int, set[tuple[float, float]]] = {}
    for line_idx, ocr_line in enumerate(ocr_lines):
        marker = re.fullmatch(
            r'\s*[コ個]\s*[xX×Ⅹ]\s*(?:単|@)\s*[¥￥]?\s*(\d[\d,]*)\s*',
            ocr_line,
        )
        if not marker:
            continue
        unit = float(marker.group(1).replace(',', ''))
        if unit <= 0:
            continue
        following_idx = next(
            (
                idx
                for idx in range(line_idx + 1, min(line_idx + 3, len(ocr_lines)))
                if ocr_lines[idx].strip()
            ),
            None,
        )
        if following_idx is None:
            continue
        amount_match = _OCR_TRAILING_PRICE_RE.fullmatch(
            ocr_lines[following_idx].strip()
        )
        if not amount_match:
            continue
        try:
            line_total = float(
                amount_match.group(1).strip().lstrip('¥￥').replace(',', '')
            )
        except ValueError:
            continue
        qty = line_total / unit
        if qty < 2 or qty > 99 or not qty.is_integer():
            continue

        owner_indices: list[int] = []
        for back_idx in range(line_idx - 1, max(line_idx - 6, -1), -1):
            nearby = ocr_lines[back_idx].strip()
            if not nearby:
                continue
            if _OCR_ZONE_END_RE.search(nearby) or _HEADER_LINE_RE.search(nearby):
                break
            if _OCR_QTY_NOTATION_RE.search(nearby):
                break
            if _OCR_TRAILING_PRICE_RE.fullmatch(nearby):
                continue
            if not _valid_ocr_item_desc(nearby):
                continue
            nearby_norm = _norm_layout_desc(nearby)
            owner_indices = [
                idx
                for idx, item in enumerate(items)
                if isinstance(item, dict)
                and item.get("discount") in (None, 0, 0.0, "", "0", "0.0")
                and nearby_norm
                and (
                    nearby_norm == _norm_layout_desc(item.get("description") or "")
                    or nearby_norm
                    in _norm_layout_desc(item.get("description") or "")
                    or _norm_layout_desc(item.get("description") or "")
                    in nearby_norm
                )
            ]
            break
        if len(owner_indices) != 1:
            continue
        missing_qty_updates.setdefault(owner_indices[0], set()).add((qty, unit))

    local_qty_owner_indices: set[int] = set()
    for item_idx, updates in missing_qty_updates.items():
        if len(updates) != 1:
            continue
        qty, unit = updates.pop()
        items[item_idx]["qty"] = qty
        items[item_idx]["unit_price"] = unit
        items[item_idx]["total"] = qty * unit
        local_qty_owner_indices.add(item_idx)

    # Garbled multiplication lines: try digit substrings validated against one
    # row-owned amount (or the extracted total when no amount separates the rows).
    for item_idx, item in enumerate(items):
        if (
            not isinstance(item, dict)
            or item.get("qty", 1) != 1
            or item.get("discount") not in (None, 0, 0.0, "", "0", "0.0")
        ):
            continue
        total = item.get("total", 0)
        if total <= 0:
            continue
        desc = item.get("description", "")
        desc_prefix = desc[:4] if len(desc) >= 4 else desc
        if not desc_prefix:
            continue
        matching_lines = [
            line_idx
            for line_idx, ocr_line in enumerate(ocr_lines)
            if desc_prefix in ocr_line
        ]
        if len(matching_lines) != 1:
            continue
        li = matching_lines[0]
        owner_desc = _norm_layout_desc(
            _clean_ocr_price_line_desc(ocr_lines[li])
        )
        owner_indices = [
            idx
            for idx, candidate in enumerate(items)
            if isinstance(candidate, dict)
            and (candidate_desc := _norm_layout_desc(
                candidate.get("description") or ""
            ))
            and owner_desc
            and (
                candidate_desc == owner_desc
                or candidate_desc in owner_desc
                or owner_desc in candidate_desc
            )
        ]
        if owner_indices != [item_idx]:
            continue
        for offset in range(1, 3):
            if li + offset >= len(ocr_lines):
                break
            nearby = ocr_lines[li + offset].strip()
            if _OCR_ZONE_END_RE.match(nearby) or _HEADER_LINE_RE.search(nearby):
                break
            if (
                _valid_ocr_item_desc(_clean_ocr_price_line_desc(nearby))
                and not re.search(r'[×xX]', nearby)
            ):
                break
            if not re.search(r'[×xX]', nearby):
                continue
            parts = re.split(r'\s*[×xX]\s*', nearby, maxsplit=1)
            if len(parts) != 2:
                continue
            local_amounts = []
            for local_line in ocr_lines[li + 1:li + offset]:
                amount_match = re.fullmatch(
                    r'\s*[¥￥]?\s*(\d[\d,]*)\s*'
                    r'(?:[A-ZＡ-Ｚ%％*＊※除軽]\s*)*',
                    local_line,
                )
                if amount_match:
                    local_amounts.append(
                        float(amount_match.group(1).replace(',', ''))
                    )
            if len(local_amounts) > 1:
                continue
            target_total = local_amounts[0] if local_amounts else float(total)
            left_digits = re.findall(r'\d+', parts[0])
            right_digits = re.findall(r'\d+', parts[1])
            candidates: set[tuple[float, float]] = set()
            for ld in left_digits:
                for rd in right_digits:
                    qty_candidates = {int(ld)}
                    unit_candidates = {int(rd)}
                    candidates.update(
                        (float(qty), float(unit))
                        for qty in qty_candidates
                        for unit in unit_candidates
                        if 2 <= qty <= 9
                        and unit > 0
                        and qty * unit == target_total
                    )
            if len(candidates) == 1:
                qty, unit = candidates.pop()
                item["qty"] = qty
                item["unit_price"] = unit
                item["total"] = qty * unit
                break

    # Collect explicit row-local qty/unit arithmetic with its OCR position.
    ocr_qty_prices: list[tuple[float, float, float, int]] = []
    forward_proposals: dict[int, set[tuple[float, float]]] = {}
    for line_idx, ocr_line in enumerate(ocr_lines):
        found_qty_str, found_price_str = None, None
        m = re.search(r'(\d+)\s*[コ個]\s*[×xX]\s*(?:単|@)?\s*(\d[\d,]*)', ocr_line)
        if m:
            found_qty_str, found_price_str = m.group(1), m.group(2)
        if not found_qty_str:
            m2 = re.search(
                r'(?:'
                r'(?:単|@)\s*(\d[\d,]*)\s*[×xX]\s*(\d+)\s*[コ個]?'
                r'|(\d[\d,]*)\s*[×xX]\s*(\d+)\s*[コ個]'
                r')',
                ocr_line,
            )
            if m2:
                found_price_str = m2.group(1) or m2.group(3)
                found_qty_str = m2.group(2) or m2.group(4)
        if not found_qty_str:
            m3 = re.search(r'(?:[¥￥]\s*)?(\d[\d,]*)\s+(\d+)\s*個', ocr_line)
            if m3:
                found_price_str, found_qty_str = m3.group(1), m3.group(2)
        if found_qty_str and found_price_str:
            ocr_qty_prices.append((
                float(found_qty_str),
                float(found_price_str.replace(',', '')),
                float(found_qty_str) * float(found_price_str.replace(',', '')),
                line_idx,
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
        ocr_qty_prices.append((qty, price, qty * price, li))

    for li, ocr_line in enumerate(ocr_lines):
        m_ten = re.match(r'^\s*(\d+)\s*点\s*$', ocr_line)
        if m_ten and li + 1 < len(ocr_lines):
            m_price = re.match(r'^\s*@\s*(\d[\d,]*)\s*$', ocr_lines[li + 1])
            if m_price:
                qty = float(m_ten.group(1))
                price = float(m_price.group(1).replace(',', ''))
                ocr_qty_prices.append((qty, price, qty * price, li))
            elif li + 2 < len(ocr_lines):
                # A preposed count/unit block owns only a uniquely titled next
                # sale whose complete printed total equals count × unit.
                if any(_OCR_ZONE_END_RE.match(line.strip()) for line in ocr_lines[:li]):
                    continue
                unit_m = re.fullmatch(r'\s*(\d[\d,]*)\s*', ocr_lines[li + 1])
                sale_m = re.fullmatch(
                    r'\s*(?P<title>.+?)\s+[¥￥]\s*(?P<amount>\d[\d,]*)\s*[*＊※除軽]?\s*',
                    ocr_lines[li + 2],
                )
                if not unit_m or not sale_m or not _valid_ocr_item_desc(sale_m.group("title")):
                    continue
                qty = float(m_ten.group(1))
                unit = float(unit_m.group(1).replace(',', ''))
                total = float(sale_m.group("amount").replace(',', ''))
                if qty <= 1 or unit <= 0 or qty * unit != total:
                    continue
                title = re.sub(r'\s+', '', sale_m.group("title"))
                source_owners = [
                    match for line in ocr_lines
                    if (match := sale_m.re.fullmatch(line))
                    and re.sub(r'\s+', '', match.group("title")) == title
                    and float(match.group("amount").replace(',', '')) == total
                ]
                if len(source_owners) != 1:
                    continue
                owners = [
                    idx for idx, item in enumerate(items)
                    if isinstance(item, dict)
                    and re.sub(r'\s+', '', item.get("description") or "") == title
                    and item.get("total") == total and not item.get("discount")
                    and not item.get("discount_rate")
                    and (item.get("qty"), item.get("unit_price"), item.get("total"))
                    in (initial_finances[idx], (qty, unit, total))
                ]
                if len(owners) == 1:
                    forward_proposals.setdefault(owners[0], set()).add((qty, unit))

    # Multi-line @PRICEx / QTY pattern (e.g., "@278x" then "3" on next line).
    for li, ocr_line in enumerate(ocr_lines):
        m_at = re.match(r'^\s*[@＠](\d[\d,]*)\s*[×xX]?\s*$', ocr_line.strip())
        if m_at and li + 1 < len(ocr_lines):
            m_qty = re.match(r'^\s*(\d+)\s*$', ocr_lines[li + 1].strip())
            if m_qty:
                price = float(m_at.group(1).replace(',', ''))
                qty = float(m_qty.group(1))
                if qty > 1:
                    ocr_qty_prices.append((qty, price, qty * price, li))

    def _local_owner(detail_idx: int) -> int | None:
        for back_idx in range(detail_idx - 1, max(detail_idx - 8, -1), -1):
            nearby = ocr_lines[back_idx].strip()
            if not nearby:
                continue
            if (
                _OCR_ZONE_END_RE.match(nearby)
                or _HEADER_LINE_RE.search(nearby)
                or _BANNER_PHRASE_RE.search(nearby)
                or _SKIP_PRICE_LINE.search(nearby)
            ):
                return None
            if _OCR_QTY_NOTATION_RE.search(nearby):
                return None
            if _OCR_TRAILING_PRICE_RE.fullmatch(nearby) or re.fullmatch(r'\d{8,}', nearby):
                continue
            owner_desc = _clean_ocr_price_line_desc(nearby)
            if not _valid_ocr_item_desc(owner_desc):
                continue
            owner_norm = _norm_layout_desc(owner_desc)
            matches: list[tuple[float, int]] = []
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
                if score >= 0.8:
                    matches.append((score, item_idx))
            if not matches:
                return None
            best_score = max(score for score, _item_idx in matches)
            best = {
                item_idx
                for score, item_idx in matches
                if abs(score - best_score) <= 0.01
            }
            return best.pop() if len(best) == 1 else None
        return None

    proposals: dict[int, set[tuple[float, float]]] = forward_proposals
    for oq, op, _ot, detail_idx in ocr_qty_prices:
        if oq <= 1 or op <= 0:
            continue
        owner_idx = _local_owner(detail_idx)
        if owner_idx is not None:
            proposals.setdefault(owner_idx, set()).add((oq, op))

    used_indices: set[int] = set(local_qty_owner_indices)
    for item_idx, owner_proposals in proposals.items():
        if len(owner_proposals) != 1:
            continue
        qty, unit = owner_proposals.pop()
        item = items[item_idx]
        item["qty"] = qty
        item["unit_price"] = unit
        item["total"] = qty * unit - (item.get("discount") or 0)
        used_indices.add(item_idx)

    for idx, item in enumerate(items):
        if not isinstance(item, dict) or idx in used_indices:
            continue
        if item.get("qty", 1) <= 1:
            continue
        desc = item.get("description", "")
        desc_norm = _norm_layout_desc(desc)
        if len(desc_norm) < 2:
            continue
        matching_lines = [
            line_idx
            for line_idx, ocr_line in enumerate(ocr_lines)
            if (
                (ocr_norm := _norm_layout_desc(
                    _clean_ocr_price_line_desc(ocr_line)
                ))
                and (desc_norm in ocr_norm or ocr_norm in desc_norm)
            )
        ]
        if len(matching_lines) != 1:
            continue
        li = matching_lines[0]
        local_lines = [ocr_lines[li]]
        for nearby in ocr_lines[li + 1:li + 4]:
            stripped = nearby.strip()
            if (
                _OCR_ZONE_END_RE.match(stripped)
                or _HEADER_LINE_RE.search(stripped)
                or _BANNER_PHRASE_RE.search(stripped)
                or (
                    _valid_ocr_item_desc(_clean_ocr_price_line_desc(stripped))
                    and not _OCR_QTY_NOTATION_RE.search(stripped)
                )
            ):
                break
            local_lines.append(nearby)
        has_qty_evidence = any(
            re.search(r'[×xX]\s*\d+|\d+\s*[×xX]|単\d|@\d', nearby)
            for nearby in local_lines
        )
        if has_qty_evidence:
            continue
        for nearby in local_lines[1:]:
            price_m = _OCR_TRAILING_PRICE_RE.fullmatch(
                nearby.strip().lstrip('*＊').strip()
            )
            if not price_m:
                continue
            ocr_price = float(
                price_m.group(1).strip().lstrip('¥￥').replace(',', '')
            )
            if abs(ocr_price - item.get("total", 0)) > 1:
                item["qty"] = 1
                item["unit_price"] = ocr_price
                item["total"] = ocr_price
            break

    # A unit/count annotation can also be extracted as an extra item. Drop
    # only its unique otherwise unowned copy when every kept sale has
    # one printed title owner and the printed subtotal closes exactly.
    zone_end = next(
        (idx for idx, line in enumerate(ocr_lines) if _OCR_ZONE_END_RE.match(line.strip())),
        len(ocr_lines),
    )
    sale_titles = {
        idx: _norm_layout_desc(_clean_ocr_price_line_desc(line))
        for idx, line in enumerate(ocr_lines[:zone_end])
        if _valid_ocr_item_desc(_clean_ocr_price_line_desc(line))
        and not _SKIP_PRICE_LINE.search(line)
        and not re.fullmatch(r'[¥￥]?\s*\d[\d,]*\s+\d+\s*個', line.strip())
    }
    # ASCII sales require an exact model title and its own printed currency
    # amount; this validates ownership without rewriting any title or digits.
    for idx, line in enumerate(ocr_lines[:zone_end]):
        sale = re.fullmatch(r'\s*(.+?)\s+[¥￥]\s*(\d[\d,]*)\s*[*＊※除軽]?\s*', line)
        if not sale or not re.search(r'[A-Za-z]', sale.group(1)):
            continue
        if any(pattern.search(line) for pattern in (_HEADER_LINE_RE, _BANNER_PHRASE_RE, _SKIP_PRICE_LINE)):
            continue
        title = re.sub(r'\s+', '', sale.group(1)).casefold()
        amount = float(sale.group(2).replace(',', ''))
        if any(
            isinstance(item, dict) and item.get("total") == amount
            and re.sub(r'\s+', '', item.get("description") or '').casefold() == title
            for item in items
        ):
            sale_titles[idx] = _norm_layout_desc(sale.group(1))
    phantom_drops = []
    for qty, unit, line_total, detail_idx in ocr_qty_prices:
        if not re.fullmatch(r'[¥￥]?\s*\d[\d,]*\s+\d+\s*個', ocr_lines[detail_idx].strip()):
            continue
        owner_idx = _local_owner(detail_idx)
        if owner_idx is None or owner_idx not in used_indices:
            continue
        owner = items[owner_idx]
        if (owner.get("qty"), owner.get("unit_price"), owner.get("total")) != (qty, unit, line_total):
            continue
        following = detail_idx + 1
        owner_titles = [
            idx for idx, title in sale_titles.items()
            if idx < detail_idx and title == _norm_layout_desc(owner.get("description") or "")
        ]
        if len(owner_titles) != 1:
            continue
        inline_amount = _OCR_TRAILING_PRICE_RE.search(ocr_lines[owner_titles[0]].strip())
        following_amount = (
            _OCR_TRAILING_PRICE_RE.fullmatch(ocr_lines[following].strip())
            if following < zone_end else None
        )
        if following < zone_end and ocr_lines[following].strip() and not following_amount:
            continue
        printed_amounts = {
            _parse_amount_fragment(match.group(1))
            for match in (inline_amount, following_amount) if match
        }
        if printed_amounts != {line_total}:
            continue
        extra_indices = [
            idx for idx, item in enumerate(items)
            if idx != owner_idx and isinstance(item, dict)
            and not item.get("discount") and not item.get("discount_rate")
            and _norm_layout_desc(_clean_ocr_price_line_desc(item.get("description") or ""))
            not in sale_titles.values()
        ]
        if len(extra_indices) != 1:
            continue
        extra_idx = extra_indices[0]
        extra = items[extra_idx]
        if not isinstance(extra, dict) or extra.get("discount") or extra.get("discount_rate"):
            continue
        extra_signature = (extra.get("qty"), extra.get("unit_price"), extra.get("total"))
        if extra_signature not in ((qty, unit, line_total), (1, unit, unit), (1, line_total, line_total)):
            continue
        extra_title = _norm_layout_desc(_clean_ocr_price_line_desc(extra.get("description") or ""))
        if extra_title and extra_title in sale_titles.values():
            continue
        kept = [item for idx, item in enumerate(items) if idx != extra_idx]
        owners = []
        for kept_idx, item in enumerate(items):
            if kept_idx == extra_idx:
                continue
            if not isinstance(item, dict) or item.get("discount") or item.get("discount_rate"):
                break
            title = _norm_layout_desc(item.get("description") or "")
            matches = [idx for idx, source_title in sale_titles.items() if title and title == source_title]
            if len(matches) != 1:
                break
            title_idx = matches[0]
            local_end = min((idx for idx in sale_titles if idx > title_idx), default=zone_end)
            owned_amounts = [
                _parse_amount_fragment(match.group(1))
                for line in ocr_lines[title_idx:local_end]
                if (match := _OCR_TRAILING_PRICE_RE.search(line.strip()))
            ]
            if item.get("total") not in owned_amounts:
                break
            owners.append(title_idx)
        else:
            subtotal = extract_financial_totals(unified_text).get("subtotal")
            if (
                len(owners) == len(set(owners))
                and subtotal is not None
                and sum(item.get("total") or 0 for item in kept) == subtotal
            ):
                phantom_drops.append(extra_idx)
    if len(phantom_drops) == 1:
        items[:] = [item for idx, item in enumerate(items) if idx not in phantom_drops]



def _liter_usage_measurement(unified_text, receipt_total):
    """Return the unique printed liters × unit-rate identity for the total."""
    if not receipt_total:
        return None
    try:
        total = float(receipt_total)
    except (TypeError, ValueError):
        return None

    amounts = {
        float(f"{match.group(1)}.{match.group(2)}")
        for match in re.finditer(
            r'(\d+)\s*[\.．]\s*(\d+)\s*[LＬ](?![A-Za-z])',
            unified_text,
        )
    }
    rates = {
        float(match.group(1).replace('．', '.'))
        for pattern in (
            r'[@＠]\s*[¥￥]?\s*(\d{2,4}(?:[\.．]\d+)?)',
            r'(\d{2,4}(?:[\.．]\d+)?)\s*円',
        )
        for match in re.finditer(pattern, unified_text)
    }
    lines = [line.strip() for line in unified_text.splitlines()]
    for label_idx, line in enumerate(lines):
        if '単価' not in line:
            continue
        for nearby in lines[label_idx + 1:label_idx + 9]:
            if re.search(r'支払|決済|現金計|お預|お釣|釣銭', nearby):
                break
            match = re.fullmatch(r'0*(\d{2,4}(?:[\.．]\d+)?)', nearby)
            if match:
                rates.add(float(match.group(1).replace('．', '.')))

    matches = {
        (amount, rate)
        for amount in amounts
        for rate in rates
        if amount > 0 and rate > 0 and abs(amount * rate - total) <= 5
    }
    if len(matches) != 1:
        return None
    return matches.pop()


def _extract_fuel_usage(extracted, unified_text):
    """Populate liter usage from a printed volume/rate/total identity."""
    total = extracted.get("total") or extracted.get("subtotal")
    measurement = _liter_usage_measurement(unified_text, total)
    if not measurement:
        return
    amount, cost_per = measurement
    usage = extracted.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    recovered = {
        "amount": amount,
        "unit": "L",
        "cost_per": cost_per,
        "meter_previous": None,
        "meter_current": None,
    }
    for key, value in recovered.items():
        if usage.get(key) is None:
            usage[key] = value
    extracted["usage"] = usage


def _fix_fuel_volume_qty(items, unified_text, receipt_total=None):
    """Normalize a liter volume to one receipt item when arithmetic proves it.

    The reference for a single-item receipt is receipt.total (printed 合計),
    which equals the item's post-tax price for 内税 receipts and pre-tax + tax
    for 外税. Using receipt.total avoids misfires under the canonical
    pre-tax subtotal convention.
    """
    if (
        len(items) != 1
        or not isinstance(items[0], dict)
        or not _liter_usage_measurement(unified_text, receipt_total)
    ):
        return
    item = items[0]
    item["qty"] = 1
    item["unit_price"] = receipt_total
    item["total"] = receipt_total


def _expand_collapsed_items(extracted, unified_text):
    """Split collapsed rows only when repeated OCR rows prove every unit."""
    items = extracted.get("line_items") or []
    valued_items = []
    for item_idx, candidate in enumerate(items):
        if not isinstance(candidate, dict):
            continue
        try:
            has_value = max(
                float(candidate.get("total") or 0),
                float(candidate.get("unit_price") or 0),
            ) > 0
        except (TypeError, ValueError):
            has_value = False
        if has_value:
            valued_items.append((item_idx, candidate))
    if not valued_items:
        return

    printed_counts = {
        int(next(value for value in match.groups() if value))
        for match in re.finditer(
            r'(?:(?:(?:お|御)?買上(?:げ)?(?:商品)?(?:点数|商品数)|'
            r'(?:商品)?点数|商品数)\s*[:：]?\s*(\d{1,3})(?:\s*点)?'
            r'|(?:本体合計\s*[（(]?\s*|小\s*計\s*[/：:]?\s*)'
            r'(\d{1,3})\s*点)',
            unified_text,
        )
    }
    lines = [line.strip() for line in unified_text.splitlines()]
    valued_descs = {
        _norm_layout_desc(item.get("description") or "")
        for _item_idx, item in valued_items
    }
    valued_descs.discard("")
    if not valued_descs:
        return
    item_start = next(
        (
            idx for idx, line in enumerate(lines)
            if _norm_layout_desc(_clean_ocr_price_line_desc(line)) in valued_descs
        ),
        None,
    )
    if item_start is None:
        return
    item_end = next(
        (
            idx for idx, line in enumerate(lines[item_start + 1:], item_start + 1)
            if _OCR_ZONE_END_RE.search(line)
        ),
        len(lines),
    )
    visible_zero_rows = 0
    for price_idx, line in enumerate(
        lines[item_start:item_end], start=item_start
    ):
        if not re.fullmatch(r'\s*[¥￥]\s*0\s*(?:円|[*※除軽外内])?\s*', line):
            continue
        owner_idx = next(
            (
                idx
                for idx in range(price_idx - 1, max(price_idx - 4, -1), -1)
                if lines[idx]
            ),
            None,
        )
        if owner_idx is None:
            continue
        owner = lines[owner_idx]
        if (
            not _HEADER_LINE_RE.search(owner)
            and not _SKIP_PRICE_LINE.search(owner)
            and not _OCR_TRAILING_PRICE_RE.fullmatch(owner)
            and bool(re.search(r'[ぁ-んァ-ン一-龥A-Za-z0-9]', owner))
        ):
            visible_zero_rows += 1

    try:
        visible_qty = sum(float(item.get("qty") or 1) for _idx, item in valued_items)
    except (TypeError, ValueError):
        return
    if printed_counts and not any(
        abs(float(count) - visible_qty - visible_zero_rows) <= 0.01
        for count in printed_counts
    ):
        return

    replacements: dict[int, list[dict]] = {}
    for item_idx, item in valued_items:
        if item.get("discount") not in (None, 0, 0.0, "", "0", "0.0"):
            continue
        if item.get("discount_rate") not in (None, "", 0, 0.0, "0", "0.0"):
            continue
        try:
            qty = float(item.get("qty") or 1)
            unit = float(item.get("unit_price") or 0)
            total = float(item.get("total") or 0)
        except (TypeError, ValueError):
            continue
        if qty < 2 or not qty.is_integer() or unit <= 0 or abs(qty * unit - total) > 2:
            continue
        count = int(qty)
        desc_norm = _norm_layout_desc(item.get("description") or "")
        owner_indices = [
            idx
            for idx, line in enumerate(
                lines[item_start:item_end], start=item_start
            )
            if _norm_layout_desc(_clean_ocr_price_line_desc(line)) == desc_norm
        ]
        if not desc_norm or len(owner_indices) != count:
            continue

        owned_amounts = []
        for pos, owner_idx in enumerate(owner_indices):
            next_owner = owner_indices[pos + 1] if pos + 1 < count else item_end
            matches = []
            for line in lines[owner_idx:min(next_owner, owner_idx + 4)]:
                amount_match = _OCR_TRAILING_PRICE_RE.search(line)
                if not amount_match:
                    continue
                token = amount_match.group(1).strip().lstrip('¥￥')
                amount = _parse_amount_fragment(token)
                if amount is not None and abs(amount - unit) <= 1:
                    matches.append(amount)
            if len(matches) != 1:
                break
            owned_amounts.append(matches[0])
        if len(owned_amounts) != count or abs(sum(owned_amounts) - total) > 2:
            continue

        replacements[item_idx] = []
        for _ in range(count):
            row = dict(item)
            row.update(qty=1, unit_price=unit, total=unit, discount=0, discount_rate="")
            replacements[item_idx].append(row)

    if not replacements:
        return
    valued_indices = {item_idx for item_idx, _item in valued_items}
    rebuilt = []
    for item_idx, item in enumerate(items):
        if item_idx in replacements:
            rebuilt.extend(replacements[item_idx])
        elif item_idx in valued_indices:
            rebuilt.append(item)
    extracted["line_items"] = rebuilt


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

    # A leading count plus one-letter size/type code is quantity evidence only
    # when an independent printed item count and the extended price agree.
    if float(item.get("qty") or 1) != 1 or item.get("discount"):
        return
    printed_counts = {
        int(match.group(1))
        for pattern in (
            r'本体合計\s*[（(]\s*(\d+)\s*点',
            r'小計\s*[/（(]?\s*(\d+)\s*点',
            r'(?:お|御)?買上(?:げ)?(?:商品)?(?:点数|商品数)\s*[:：]?\s*(\d+)',
        )
        for match in re.finditer(pattern, unified_text)
    }
    if len(printed_counts) != 1:
        return
    printed_count = printed_counts.pop()
    if printed_count <= 1 or float(total) % printed_count:
        return
    desc_norm = _norm_layout_desc(desc)
    candidates = []
    for line in lines:
        prefix = re.match(r'^\s*(\d{1,2})\s*([A-Za-zＡ-Ｚ])\s+(.+)$', line)
        if not prefix or int(prefix.group(1)) != printed_count:
            continue
        candidate_desc = _clean_ocr_price_line_desc(
            f"{prefix.group(2)} {prefix.group(3)}"
        )
        candidate_norm = _norm_layout_desc(candidate_desc)
        if not candidate_norm or not desc_norm:
            continue
        if not (desc_norm in candidate_norm or candidate_norm in desc_norm):
            continue
        price_match = _OCR_TRAILING_PRICE_RE.search(line)
        if not price_match:
            continue
        visible = float(
            price_match.group(1).strip().lstrip('¥￥').replace(',', '')
        )
        if abs(visible - float(total)) <= 2:
            candidates.append(line)
    if len(candidates) == 1:
        item["qty"] = float(printed_count)
        item["unit_price"] = float(total) / printed_count
        item["total"] = float(total)


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
