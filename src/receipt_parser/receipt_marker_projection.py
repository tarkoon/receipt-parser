"""Receipt marker and quantity projection helpers."""

import re
from difflib import SequenceMatcher

from .patterns import (
    _FOOD_DESC_RE,
    _OCR_QTY_NOTATION_RE,
    _OCR_TRAILING_PRICE_RE,
    _SKIP_PRICE_LINE,
)
from .receipt_financial import extract_rate_bases
from .receipt_item_repair import (
    _ocr_line_index_for_item,
    _valid_ocr_item_desc,
)
from .receipt_projection import (
    _clean_ocr_price_line_desc,
)
from .receipt_tax_categories import (
    _fix_tax_categories_from_ocr_markers,
    _is_bag_description,
    _rebalance_tax_categories_to_rate_bases,
)
from .receipt_totals import _sum_taxable_amounts

_DISCOUNT_WORD = chr(0x5272) + chr(0x5F15)


def _replace_campaign_discount_stream_when_balanced(extracted, unified_text):
    """Reconstruct item streams where campaign discounts are printed separately."""
    subtotal = extracted.get("subtotal")
    if len(re.findall(_DISCOUNT_WORD, unified_text or "")) < 1:
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
        r'^(.+?[ぁ-んァ-ン一-龥A-Za-z][^¥￥]*?)\s+[¥￥]?\s*(\d[\d,]*)\s*([非除※*＊↓]*)$'
    )
    qty_re = re.compile(r'[<\(（]?\s*(\d+)\s*[個コ]?\s*[xX×Ⅹ]\s*単?\s*(\d{1,5})')

    def _clean_desc(text: str) -> str:
        text = _clean_ocr_price_line_desc(text)
        text = re.sub(r'^\d{3,}\s*[※*＊]?\s*', '', text).strip()
        text = re.sub(r'^[※*＊]\s*', '', text).strip()
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

    def _amount_from_line(line: str) -> tuple[float, str] | None:
        match = amount_re.match(line)
        if not match:
            return None
        return float(match.group(1).replace(',', '')), match.group(2) or ""

    def _append_inline_rows(target_rows: list[dict], start_idx: int, stop_idx: int) -> None:
        source_idx = start_idx
        while source_idx < stop_idx:
            inline_m = inline_re.match(zone[source_idx])
            if not inline_m or not _valid_desc(inline_m.group(1)):
                desc = _clean_desc(zone[source_idx])
                next_amount = (
                    _amount_from_line(zone[source_idx + 1])
                    if source_idx + 1 < stop_idx
                    else None
                )
                if _valid_desc(desc) and next_amount:
                    row = _make_row(desc, next_amount[0], next_amount[1])
                    row["_source_idx"] = source_idx
                    target_rows.append(row)
                    source_idx += 2
                    continue
                source_idx += 1
                continue
            row = _make_row(
                _clean_desc(inline_m.group(1)),
                float(inline_m.group(2).replace(',', '')),
                inline_m.group(3) or "",
            )
            row["_source_idx"] = source_idx
            target_rows.append(row)
            source_idx += 1

    def _replace_single_interleaved_discount_stack() -> bool:
        if len(re.findall(_DISCOUNT_WORD, "\n".join(zone))) != 1:
            return False
        discount_idx = next((idx for idx, line in enumerate(zone) if _DISCOUNT_WORD in line), None)
        if discount_idx is None:
            return False

        discounted_desc = ""
        discounted_idx = None
        for idx in range(discount_idx - 1, max(discount_idx - 4, -1), -1):
            cand = _clean_desc(zone[idx])
            if _valid_desc(cand):
                discounted_desc = cand
                discounted_idx = idx
                break
        if discounted_idx is None:
            return False

        first_after_discount = _amount_from_line(zone[discount_idx + 1]) if discount_idx + 1 < len(zone) else None
        second_after_discount = _amount_from_line(zone[discount_idx + 2]) if discount_idx + 2 < len(zone) else None
        immediate_rate = ""
        immediate_discount_idx = None
        if first_after_discount and second_after_discount:
            for idx in range(discount_idx + 3, min(discount_idx + 6, len(zone))):
                rate_m = re.search(r'^(\d+(?:\.\d+)?)\s*%$', zone[idx])
                if rate_m:
                    immediate_rate = f"{int(float(rate_m.group(1)))}%"
                    continue
                if re.match(r'^-\s*[¥￥]?\s*\d', zone[idx]):
                    immediate_discount_idx = idx
                    break
        if immediate_discount_idx is not None:
            previous_desc = ""
            previous_idx = None
            for idx in range(discounted_idx - 1, max(discounted_idx - 4, -1), -1):
                cand = _clean_desc(zone[idx])
                if _valid_desc(cand):
                    previous_desc = cand
                    previous_idx = idx
                    break
            discount_m = re.fullmatch(r'-\s*[¥￥]?\s*(\d[\d,]*)', zone[immediate_discount_idx])
            if previous_idx is not None and discount_m:
                discount = float(discount_m.group(1).replace(',', ''))
                rebuilt: list[dict] = []
                _append_inline_rows(rebuilt, 0, previous_idx)

                row = _make_row(previous_desc, first_after_discount[0], first_after_discount[1])
                row["_source_idx"] = previous_idx
                rebuilt.append(row)

                row = _make_row(discounted_desc, second_after_discount[0], second_after_discount[1])
                row["_source_idx"] = discounted_idx
                row["discount"] = discount
                row["discount_rate"] = immediate_rate
                row["total"] = second_after_discount[0] - discount
                rebuilt.append(row)

                _append_inline_rows(rebuilt, immediate_discount_idx + 1, len(zone))
                if len(rebuilt) >= len(extracted.get("line_items") or []):
                    row_sum = sum(float(row.get("total") or 0) for row in rebuilt)
                    if abs(row_sum - subtotal_target) <= 2:
                        rate_bases = extract_rate_bases(unified_text)
                        _rebalance_tax_categories_to_rate_bases(
                            rebuilt,
                            unified_text,
                            extracted.get("taxes"),
                            rate_bases,
                        )
                        extracted["line_items"] = [
                            {key: value for key, value in row.items() if not key.startswith("_")}
                            for row in rebuilt
                        ]
                        if printed_subtotal is not None:
                            extracted["subtotal"] = printed_subtotal
                        return True

        following: list[tuple[str, int]] = []
        scan = discount_idx + 1
        while scan < len(zone):
            line = zone[scan]
            if _amount_from_line(line) or re.search(r'^\d+(?:\.\d+)?\s*%$', line) or re.match(r'^-', line):
                break
            cand = _clean_desc(line)
            if _valid_desc(cand):
                following.append((cand, scan))
            scan += 1
        if len(following) < 2:
            return False

        first_amount = _amount_from_line(zone[scan]) if scan < len(zone) else None
        second_amount = _amount_from_line(zone[scan + 1]) if scan + 1 < len(zone) else None
        if not first_amount or not second_amount:
            return False
        rate = ""
        discount_line_idx = None
        for idx in range(scan + 2, min(scan + 5, len(zone))):
            rate_m = re.search(r'^(\d+(?:\.\d+)?)\s*%$', zone[idx])
            if rate_m:
                rate = f"{int(float(rate_m.group(1)))}%"
                continue
            if re.match(r'^-\s*[¥￥]?\s*\d', zone[idx]):
                discount_line_idx = idx
                break
        if discount_line_idx is None:
            return False
        discount_m = re.fullmatch(r'-\s*[¥￥]?\s*(\d[\d,]*)', zone[discount_line_idx])
        if not discount_m:
            return False
        discount = float(discount_m.group(1).replace(',', ''))
        third_amount = (
            _amount_from_line(zone[discount_line_idx + 1])
            if discount_line_idx + 1 < len(zone)
            else None
        )
        if not third_amount:
            return False

        rebuilt: list[dict] = []
        _append_inline_rows(rebuilt, 0, discounted_idx)

        last_desc, last_idx = following[-1]
        row = _make_row(last_desc, first_amount[0], first_amount[1])
        row["_source_idx"] = last_idx
        rebuilt.append(row)

        row = _make_row(discounted_desc, second_amount[0], second_amount[1])
        row["_source_idx"] = discounted_idx
        row["discount"] = discount
        row["discount_rate"] = rate
        row["total"] = second_amount[0] - discount
        rebuilt.append(row)

        middle_desc, middle_idx = following[0]
        row = _make_row(middle_desc, third_amount[0], third_amount[1])
        row["_source_idx"] = middle_idx
        rebuilt.append(row)

        _append_inline_rows(rebuilt, discount_line_idx + 2, len(zone))
        if len(rebuilt) < len(extracted.get("line_items") or []):
            return False
        row_sum = sum(float(row.get("total") or 0) for row in rebuilt)
        if abs(row_sum - subtotal_target) > 2:
            return False
        rate_bases = extract_rate_bases(unified_text)
        _rebalance_tax_categories_to_rate_bases(rebuilt, unified_text, extracted.get("taxes"), rate_bases)
        extracted["line_items"] = [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in rebuilt
        ]
        if printed_subtotal is not None:
            extracted["subtotal"] = printed_subtotal
        return True

    if _replace_single_interleaved_discount_stack():
        return

    for source_idx, line in enumerate(zone):
        qty_m = qty_re.search(line)
        if qty_m:
            _apply_qty_detail(float(qty_m.group(1)), float(qty_m.group(2)))
            continue

        rate_m = re.search(r'(\d+(?:\.\d+)?)\s*%', line)
        if _DISCOUNT_WORD in line or (rate_m and not amount_re.match(line)):
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

    def _desc_match_rank(ndesc: str, item: dict) -> int:
        item_desc = _norm(item.get("description") or "")
        if not item_desc:
            return 2
        if ndesc in item_desc or item_desc in ndesc:
            return 0
        return 1 if SequenceMatcher(None, ndesc, item_desc).ratio() >= 0.72 else 2

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
        for nearby_idx in range(line_idx, min(detail_idx, line_idx + 4)):
            nearby = lines[nearby_idx]
            if nearby_idx > line_idx and _valid_desc(_clean_ocr_price_line_desc(nearby)):
                break
            pm = _OCR_TRAILING_PRICE_RE.search(nearby)
            if not pm:
                pm = re.search(r'(?:^|[\s(（])([¥￥]?\s*\d[\d,]*)\s*(?:[%％*＊※除軽Xx外内]+)?\s*$', nearby)
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

    def _own_visible_amount_before_next_desc(item: dict, detail_idx: int) -> float | None:
        item_desc = _norm(item.get("description") or "")
        line_idx = next(
            (
                idx for idx, line in enumerate(lines[:detail_idx])
                if item_desc and item_desc == _norm(_clean_ocr_price_line_desc(line))
            ),
            None,
        )
        if line_idx is None:
            line_idx = _ocr_line_index_for_item(lines, item)
        if line_idx is None or line_idx >= detail_idx:
            return None
        for nearby in lines[line_idx + 1:detail_idx]:
            pm = re.fullmatch(
                r'[¥￥]?\s*(\d[\d,]*)\s*(?:[%％*＊※除軽Xx外内])?',
                nearby.strip(),
            )
            if pm:
                return float(pm.group(1).replace(',', ''))
            if _valid_desc(_clean_ocr_price_line_desc(nearby)):
                break
        return None

    def _has_standalone_gross_before_qty_detail(item: dict, gross: float, detail_idx: int) -> bool:
        line_idx = _ocr_line_index_for_item(lines, item)
        if line_idx is None or line_idx >= detail_idx:
            line_idx = None
        plain = str(int(gross)) if gross == int(gross) else str(gross)
        comma = f"{int(gross):,}" if gross == int(gross) else plain
        amount_re = re.compile(
            rf'^[¥￥]?\s*(?:{re.escape(plain)}|{re.escape(comma)})\s*[%％*＊※除軽Xx外内]?\s*$'
        )
        if line_idx is not None and amount_re.fullmatch(lines[line_idx].strip()):
            return True
        if line_idx is not None:
            for nearby in lines[line_idx + 1:detail_idx]:
                if amount_re.fullmatch(nearby.strip()):
                    return True
                if _valid_desc(_clean_ocr_price_line_desc(nearby)):
                    break
        item_desc = _norm(item.get("description") or "")
        if not item_desc:
            return False
        if line_idx is not None:
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

    def _nearby_standalone_amount(detail_idx: int) -> float | None:
        for nearby_idx in (detail_idx - 1, detail_idx + 1, detail_idx - 2, detail_idx + 2):
            if nearby_idx < 0 or nearby_idx >= len(lines):
                continue
            nearby = lines[nearby_idx].strip()
            if not nearby or any(ch.isalpha() for ch in nearby):
                continue
            digits = "".join(ch for ch in nearby if ch.isdigit() or ch == ",")
            if not digits:
                continue
            try:
                return float(digits.replace(',', ''))
            except ValueError:
                continue
        return None

    def _parse_unit_line_qty_detail(line: str, detail_idx: int) -> tuple[float, float] | None:
        unit_first = re.search(
            r'\(?\s*(?:[単单]|[@＠])\s*(\d[\d,]*)\s*[xX×Ⅹ]\s*(\d+)\s*[個コ点]\s*\)?',
            line,
        )
        if unit_first:
            unit = float(unit_first.group(1).replace(',', ''))
            qty = float(unit_first.group(2))
        else:
            if re.search(r'[@＠]', line):
                bare_unit_first = re.search(
                    r'\(?\s*[@＠]\s*(\d[\d,]*)\s*[xX×Ⅹ]\s*(\d+)\s*\)?',
                    line,
                )
                if not bare_unit_first:
                    return None
                unit = float(bare_unit_first.group(1).replace(',', ''))
                qty = float(bare_unit_first.group(2))
                nearby_total = _nearby_standalone_amount(detail_idx)
                if (
                    qty < 2
                    or unit <= 0
                    or nearby_total is None
                    or abs(qty * unit - nearby_total) > 2
                ):
                    return None
                return qty, unit
            qty_first = re.search(
                r'\(?\s*(\d+)\s*(?P<marker>[個コ点]\s*[xX×Ⅹ]?\s*[単单]?|[xX×Ⅹ]\s*[単单]?|[単单])\s*(\d[\d,]*)\s*\)?',
                line,
            )
            if not qty_first:
                compact_qty = re.search(
                    r'\(?\s*(\d{2,4})\)?\s*[xX]\s*(\d[\d,]*)\s*\)?',
                    line,
                )
                if not compact_qty:
                    return None
                compact_digits = compact_qty.group(1)
                unit = float(compact_qty.group(2).replace(',', ''))
                nearby_total = _nearby_standalone_amount(detail_idx)
                qty = float(compact_digits[0])
                if (
                    qty < 2
                    or unit <= 0
                    or nearby_total is None
                    or abs(qty * unit - nearby_total) > 2
                ):
                    return None
                return qty, unit
            qty = float(qty_first.group(1))
            unit = float(qty_first.group(3).replace(',', ''))
            marker = qty_first.group("marker")
            if len(qty_first.group(1)) > 1 and re.search(r'[xX×Ⅹ]', marker) and not re.search(r'[個コ点]', marker):
                nearby_total = _nearby_standalone_amount(detail_idx)
                if nearby_total is None or abs(qty * unit - nearby_total) > 2:
                    return None
        if qty < 2 or unit <= 0:
            return None
        return qty, unit

    def _strip_embedded_qty_detail(desc: str) -> str:
        cleaned = _OCR_QTY_NOTATION_RE.sub("", desc or "")
        cleaned = re.sub(r'\s*[（(]\s*[）)]\s*$', '', cleaned).strip()
        return cleaned

    qty_detail_matched_items: set[int] = set()
    for idx, line in enumerate(lines):
        detail = _parse_unit_line_qty_detail(line, idx)
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
            ranked_items = sorted(
                [item for item in items if isinstance(item, dict)],
                key=lambda item: _desc_match_rank(ndesc, item),
            )
            for item in ranked_items:
                desc_rank = _desc_match_rank(ndesc, item)
                item_desc_is_qty_detail = bool(
                    _OCR_QTY_NOTATION_RE.search(str(item.get("description") or ""))
                )
                if desc_rank >= 2 and not item_desc_is_qty_detail:
                    continue
                item_total = float(item.get("total") or 0)
                item_discount = float(item.get("discount") or 0)
                nearby_total = _nearby_standalone_amount(idx)
                supported_by_nearby_total = (
                    desc_rank < 2
                    and nearby_total is not None
                    and abs(nearby_total - expected_total) <= 2
                )
                if (
                    abs(item_total - unit) > 2
                    and abs(item_total - expected_total) > 2
                    and abs(item_total + item_discount - expected_total) > 2
                    and not supported_by_nearby_total
                ):
                    continue
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
                if desc_rank < 2:
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
                        item_desc = _norm(item.get("description") or "")
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
                    cleaned_desc = _strip_embedded_qty_detail(item.get("description") or "")
                    if cleaned_desc and _valid_desc(cleaned_desc):
                        item["description"] = cleaned_desc
                    matched_item = item
                    qty_detail_matched_items.add(id(item))
                    break
            if matched_item is not None:
                break
        if matched_item is None:
            continue
        qty_detail_matched_items.add(id(matched_item))
        for item_idx, item in enumerate(items):
            if item is matched_item or not isinstance(item, dict):
                continue
            if id(item) in qty_detail_matched_items:
                continue
            if abs(float(item.get("unit_price") or 0) - unit) > 2:
                continue
            if abs(float(item.get("total") or 0) - qty * unit) > 2:
                continue
            gross = qty * unit
            own_amount = _own_visible_amount_before_next_desc(item, idx)
            if own_amount is not None and abs(own_amount - gross) <= 2:
                item["qty"] = 1.0
                item["unit_price"] = gross
                item["total"] = gross
                continue
            if own_amount is not None and abs(own_amount - unit) <= 2:
                item["qty"] = 1.0
                item["unit_price"] = unit
                item["total"] = unit
                continue
            if _has_standalone_gross_before_qty_detail(item, gross, idx):
                item["qty"] = 1.0
                item["unit_price"] = gross
                item["total"] = gross
                continue
            if _has_direct_unit_price_row(item, unit, idx):
                item["qty"] = 1.0
                item["unit_price"] = unit
                item["total"] = unit


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
