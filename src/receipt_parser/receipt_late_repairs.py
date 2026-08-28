"""Late receipt postprocess repair helpers."""

import re

from .patterns import (
    _HEADER_LINE_RE,
    _OCR_TRAILING_PRICE_RE,
    _OCR_ZONE_END_RE,
    _SKIP_PRICE_LINE,
)
from .receipt_financial import _parse_amount_fragment, extract_rate_bases, normalize_tax_rate
from .receipt_item_repair import _clean_code_prefixed_item_descriptions
from .receipt_items import (
    _bag_entries_from_ocr,
)
from .receipt_location import _PURCHASE_STORE_METADATA_RE
from .receipt_projection import _clean_ocr_price_line_desc
from .receipt_tax_categories import (
    _is_bag_description,
    _rebalance_tax_categories_to_rate_bases,
)
from .receipt_totals import _sum_taxable_amounts


def _replace_stacked_name_price_rows_when_balanced(extracted, unified_text):
    """Parse receipts that stack item names first, then matching price rows."""
    def _rate_from_marker(marker: str) -> str:
        return "10%" if marker == "外" else "8%" if marker == "軽" else "0%"

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
    current_items = [item for item in (extracted.get("line_items") or []) if isinstance(item, dict)]
    if len(current_items) >= 2 and known_amounts:
        current_sum = sum(float(item.get("total") or 0) for item in current_items)
        if any(abs(current_sum - amount) <= 2 for amount in known_amounts):
            return

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
                "tax_category": _rate_from_marker(marker),
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
                "tax_category": _rate_from_marker(marker),
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
                "tax_category": _rate_from_marker(pm.group(2) or ""),
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
                    "tax_category": _rate_from_marker(marker),
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
        if re.search(r'\d+\s*/\s*\d+', bag_line):
            continue
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
    return True


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
    tax_rates = {rate for rate, is_tax in labels if is_tax}
    if len(tax_rates) == 1 and values:
        rate = next(iter(tax_rates))
        try:
            rate_pct = float(rate.rstrip("%")) / 100.0
        except ValueError:
            rate_pct = 0.0
        base = max(values)
        candidates = [value for value in values if 0 < value < base]
        if rate_pct > 0 and candidates:
            expected = base * rate_pct / (1 + rate_pct)
            amount = min(candidates, key=lambda value: abs(value - expected))
            if abs(amount - expected) <= max(2.0, expected * 0.10):
                taxes.append({"rate": rate, "label": "内税", "amount": amount})
    else:
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
    base = total
    if rate_m:
        rate = normalize_tax_rate(rate_m.group(1) + "%")
        idx = unified_text.find(rate_m.group(0))
        tail = unified_text[idx:] if idx >= 0 else unified_text
        values = [
            float(m.group(1).replace(',', ''))
            for m in re.finditer(r'[¥￥]\s*([\d,]+)\s*[\)）]?', tail)
        ]
    else:
        inline_m = re.search(
            r'(\d+(?:\.\d+)?)\s*%\s*対象\s*[¥￥]?\s*([\d,]+)\s*'
            r'(?P<tax_label>(?:内\s*)?消費税等?)\s*[¥￥]?\s*(?P<tax_amount>[\d,]+)',
            unified_text,
            flags=re.S,
        )
        if not inline_m:
            return
        rate = normalize_tax_rate(inline_m.group(1) + "%")
        base = float(inline_m.group(2).replace(',', ''))
        expected = float(inline_m.group("tax_amount").replace(',', ''))
        rate_bases = {
            r: b for r, b in extract_rate_bases(unified_text).items()
            if r != "0%" and b
        }
        if set(rate_bases) - {rate}:
            return
        if abs(base - total) > 2:
            return
    try:
        rate_pct = float(rate.rstrip('%')) / 100.0
    except ValueError:
        return
    arithmetic_tax = round(base * rate_pct / (1 + rate_pct))
    if arithmetic_tax <= 0:
        return
    if rate_m:
        expected = arithmetic_tax
        if not any(abs(value - expected) <= 2 for value in values):
            return
    elif abs(float(expected) - arithmetic_tax) > 2:
        return

    current_taxes = [
        tax for tax in (extracted.get("taxes") or [])
        if isinstance(tax, dict)
        and tax.get("rate") != "0%"
        and tax.get("amount") is not None
    ]
    preserve_existing = False
    if len(current_taxes) == 1 and extracted.get("subtotal") is not None:
        current_tax = current_taxes[0]
        try:
            current_amount = float(current_tax["amount"])
            current_subtotal = float(extracted["subtotal"])
        except (KeyError, TypeError, ValueError):
            pass
        else:
            preserve_existing = (
                current_tax.get("label") == "内税"
                and normalize_tax_rate(str(current_tax.get("rate") or "")) == rate
                and abs(current_amount - arithmetic_tax) <= 2
                and abs(current_subtotal + current_amount - total) <= 2
            )
    if not preserve_existing:
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
    """Recover one printed address that uniquely extends an admin fragment."""
    existing = re.sub(r'\s+', '', extracted.get("location") or "")
    lines = [line.strip() for line in unified_text.split('\n')]
    address_chars = r'[\u3040-\u30ff\u3400-\u9fff々〆ー]'
    noise = re.compile(
        r'TEL|電話|FAX|☎|登録番号|営業時間|領収|合計|小計|消費税|'
        r'本社|本店所在地|所在地|住所|配送先|請求先|お問い合わせ|〒|'
        r'株式会社|有限会社|合同会社|㈱|㈲',
        re.IGNORECASE,
    )
    scoped_label = re.compile(
        r'(?:本社(?:所在地|住所)?|本店所在地|会社所在地|配送先|請求先|'
        r'送付先|お届け先|返送先|お問い合わせ(?:先|窓口)?)[:：]?'
    )
    candidates = []
    if existing and re.search(r'[都道府県市区町村郡]$', existing):
        for idx, line in enumerate(lines):
            compact = re.sub(r'\s+', '', line)
            previous = next(
                (
                    re.sub(r'\s+', '', lines[pos])
                    for pos in range(idx - 1, -1, -1)
                    if lines[pos]
                ),
                "",
            )
            if (
                not noise.search(compact)
                and not scoped_label.fullmatch(previous)
                and compact.startswith(existing)
                and re.fullmatch(
                    rf'{address_chars}+[都道府県市区町村郡]'
                    rf'{address_chars}+[\d０-９]{{1,6}}',
                    compact,
                )
                and compact not in candidates
            ):
                candidates.append(compact)
    for idx, line in enumerate(lines[:-1]):
        first = re.sub(r'\s+', '', line)
        if noise.search(first):
            continue
        previous = next(
            (re.sub(r'\s+', '', lines[pos]) for pos in range(idx - 1, -1, -1) if lines[pos]),
            "",
        )
        if scoped_label.fullmatch(previous):
            continue
        admin_only = re.fullmatch(
            rf'{address_chars}+[都道府県]{address_chars}+[市区町村]',
            first,
        )
        ward_with_locality = re.fullmatch(
            rf'{address_chars}+[都道府県]{address_chars}+区{address_chars}+',
            first,
        )
        if not (admin_only or ward_with_locality):
            continue
        nxt = re.sub(r'\s+', '', lines[idx + 1])
        if noise.search(nxt):
            continue
        numeric_street = bool(
            re.fullmatch(r'[\d０-９][\d０-９丁目番号地\-－‐―ー]*', nxt)
            and re.search(r'丁目|番(?:地)?|号|[-－‐―ー]', nxt)
        )
        locality_street = bool(
            admin_only
            and re.fullmatch(
                rf'{address_chars}+[\d０-９][\d０-９丁目番号地\-－‐―ー]*',
                nxt,
            )
            and re.search(r'丁目|番(?:地)?|号|[-－‐―ー]', nxt)
        )
        if not (numeric_street or locality_street):
            continue
        if re.fullmatch(
            r'[0０][\d０-９]{1,4}[-－‐―ー][\d０-９]{1,4}'
            r'[-－‐―ー][\d０-９]{2,4}',
            nxt,
        ) or re.fullmatch(
            r'[\d０-９]{3}[-－‐―ー][\d０-９]{4}|'
            r'[\d０-９]{4}[-－‐―ー][\d０-９]{1,2}[-－‐―ー][\d０-９]{1,2}',
            nxt,
        ):
            continue
        candidate = first + nxt
        if candidate not in candidates:
            candidates.append(candidate)
    if len(candidates) != 1:
        return
    candidate = candidates[0]
    if not existing or existing in candidate or len(candidate) > len(existing) + 4:
        extracted["location"] = candidate


def _recover_labeled_purchase_site_location(extracted, unified_text):
    """Recover the exact site named by a bounded purchase-site label."""
    existing = re.sub(r'\s+', '', extracted.get("location") or "")
    lines = [line.strip() for line in unified_text.split('\n') if line.strip()]
    for line in lines:
        metadata = _PURCHASE_STORE_METADATA_RE.match(line)
        if metadata:
            candidate = re.sub(r'\s+', '', metadata.group("store"))
        else:
            warehouse = re.search(
                r'購入倉庫店\s*[:：]\s*'
                r'([^\s:：¥￥,，。()（）]+?倉庫店)',
                line,
            )
            if not warehouse:
                continue
            site = re.sub(r'\s+', '', warehouse.group(1))
            candidate = re.sub(r'倉庫店$', '', site)
        if not re.fullmatch(r'[\u3040-\u30ff\u3400-\u9fffA-Za-z0-9・ー]{2,30}', candidate):
            continue
        if re.search(r'TEL|電話|登録番号|会員番号|領収|合計|小計|対象|消費税', candidate, re.IGNORECASE):
            continue
        if metadata is None and existing and (candidate in existing or existing in candidate):
            return
        extracted["location"] = candidate
        return
