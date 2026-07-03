"""Receipt structural row projection helpers."""

import re

from .patterns import (
    _FOOD_DESC_RE,
    _HEADER_LINE_RE,
    _OCR_TRAILING_PRICE_RE,
    _SKIP_PRICE_LINE,
)
from .receipt_financial import extract_rate_bases, normalize_tax_rate
from .receipt_projection import _clean_ocr_price_line_desc
from .receipt_tax_categories import (
    _assign_single_standard_rate_from_small_base,
    _fix_tax_categories_from_ocr_markers,
    _is_bag_description,
    _rebalance_tax_categories_to_rate_bases,
)
from .receipt_totals import _sum_taxable_amounts
from .schema import STANDARD_RATE


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
    current_items = [
        item for item in (extracted.get("line_items") or [])
        if isinstance(item, dict)
    ]
    if current_items:
        current_sum = sum(float(item.get("total") or 0) for item in current_items)
        current_qty = sum(float(item.get("qty") or 1) for item in current_items)
        if (
            abs(current_sum - subtotal) <= 2
            and (printed_count is None or abs(current_qty - printed_count) <= 0.1)
        ):
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


def _low_value_bag_rows_from_ocr(lines: list[str]) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []

    def _standalone_small_price(line: str) -> float | None:
        m = re.fullmatch(r'[¥￥]?\s*(\d{1,2})\s*[%％*※除軽Xx]?', line.strip())
        if not m:
            return None
        price = float(m.group(1))
        return price if 0 < price < 10 else None

    for idx, line in enumerate(lines):
        if not re.search(r'袋|バッグ|bag', line, re.IGNORECASE):
            continue
        desc = re.sub(r'\s+', ' ', re.sub(r'\s+[¥￥]?\s*\d[\d,]*\s*(?:[%％*※除軽Xx])?\s*$', '', line)).strip()
        inline_price = None
        m = re.search(r'^(.+?[ぁ-んァ-ン一-龥][^¥￥]*?)\s+([¥￥]?\s*\d[\d,]*)\s*(?:[%％*※除軽Xx])?\s*$', line)
        if m:
            desc = re.sub(r'\s+', ' ', m.group(1)).strip()
            try:
                inline_price = float(m.group(2).strip().lstrip('¥￥').replace(',', ''))
            except ValueError:
                inline_price = None
        price = None
        for nearby in lines[idx + 1:min(len(lines), idx + 6)]:
            if _is_bag_description(nearby):
                break
            nearby_price = _standalone_small_price(nearby)
            if nearby_price is not None:
                price = nearby_price
                break
        if price is None:
            price = inline_price
        if price is not None and 0 < price < 10:
            rows.append((desc, price))
    return rows


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
    bag_rows = _low_value_bag_rows_from_ocr(lines)
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
    bag_rows = _low_value_bag_rows_from_ocr([line.strip() for line in unified_text.split('\n')])
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
