"""Receipt item and discount cleanup helpers."""

import re
from difflib import SequenceMatcher

from .patterns import (
    _OCR_QTY_NOTATION_RE,
    _OCR_TRAILING_PRICE_RE,
    _SKIP_PRICE_LINE,
)
from .receipt_financial import extract_rate_bases, normalize_tax_label, normalize_tax_rate
from .receipt_item_repair import (
    _find_discounted_ocr_item_desc,
    _insert_item_by_ocr_order,
    _valid_ocr_item_desc,
    _valid_pre_price_stack_item_desc,
)
from .receipt_projection import (
    _clean_ocr_price_line_desc,
    _norm_layout_desc,
)
from .receipt_tax_categories import _is_bag_description
from .receipt_totals import (
    _canonical_subtotal_from_taxes,
    _line_items_sum,
    _sum_taxable_amounts,
)
from .schema import VALID_TAX_RATES

_CATALOG_METADATA_RE = re.compile(
    "|".join(
        (
            "".join(chr(cp) for cp in (0x30E9, 0x30D9, 0x30EB)),
            "".join(chr(cp) for cp in (0x901A, 0x5E38, 0x4FA1, 0x683C)),
            "".join(chr(cp) for cp in (0x5546, 0x54C1, 0x540D)),
            "Customer",
            "Return",
            "model",
        )
    ),
    re.IGNORECASE,
)


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


def _catalog_model_amounts(line: str) -> list[float]:
    text = line.strip()
    if re.fullmatch(r'\d{1,6}(?:\s+0)?', text):
        return [float(text.split()[0])]
    m = re.fullmatch(r'\d+\s*\*\s*(\d{1,6})', text)
    return [float(m.group(1))] if m else []


def _catalog_model_detail_desc(line: str, following: list[str]) -> tuple[str, int]:
    if not re.search(r'[A-Za-z?]{2,}', line) or not re.search(r'[ぁ-んァ-ン一-龥]', line):
        return "", 0
    head = re.sub(r'^[A-Za-z0-9?][A-Za-z0-9?/\- ]{2,}\s+', '', line).strip()
    if not _valid_pre_price_stack_item_desc(line, head):
        return "", 0
    parts = [head]
    for raw in following[:2]:
        if _CATALOG_METADATA_RE.search(raw):
            break
        desc = _clean_ocr_price_line_desc(raw)
        if not _valid_pre_price_stack_item_desc(raw, desc):
            break
        parts.append(desc)
    desc = re.sub(r'\s*,\s*', ', ', " ".join(parts)).strip()
    desc = re.sub(r'[/／]\s*$', '', desc).strip()
    return desc, len(parts)


def _repair_catalog_model_detail_stack_descriptions(extracted, lines: list[str]) -> bool:
    items = [item for item in (extracted.get("line_items") or []) if isinstance(item, dict)]
    if len(items) < 2:
        return False
    if not any(re.search(r'[A-Za-z?]{2,}', item.get("description") or "") for item in items):
        return False

    def _previous_line_has_amount(idx: int) -> bool:
        return idx > 0 and bool(_catalog_model_amounts(lines[idx - 1]))

    def _amount_after(idx: int, total: float) -> bool:
        for raw in lines[idx + 1:min(idx + 10, len(lines))]:
            if any(abs(amount - total) <= 2 for amount in _catalog_model_amounts(raw)):
                return True
        return False

    candidates_by_total: dict[float, list[tuple[int, str, int]]] = {}
    for idx, line in enumerate(lines):
        amounts = _catalog_model_amounts(line)
        if amounts:
            for lookahead in range(idx + 1, min(idx + 4, len(lines))):
                if _catalog_model_amounts(lines[lookahead]):
                    break
                desc, part_count = _catalog_model_detail_desc(lines[lookahead], [])
                if desc:
                    for amount in amounts:
                        candidates_by_total.setdefault(amount, []).append((lookahead, desc, part_count + 2))
                    break
            continue
        if _previous_line_has_amount(idx):
            continue
        desc, part_count = _catalog_model_detail_desc(line, lines[idx + 1:idx + 3])
        if not desc:
            continue
        for item in items:
            try:
                total = float(item.get("total") or 0)
            except (TypeError, ValueError):
                continue
            if total > 0 and _amount_after(idx, total):
                candidates_by_total.setdefault(total, []).append((idx, desc, part_count))

    chosen: dict[int, tuple[int, str]] = {}
    for item_idx, item in enumerate(items):
        try:
            total = float(item.get("total") or 0)
        except (TypeError, ValueError):
            return False
        options = candidates_by_total.get(total, [])
        if not options:
            return False
        best = max(options, key=lambda row: (row[2], len(_norm_layout_desc(row[1]))))
        if sum(1 for row in options if (row[2], len(_norm_layout_desc(row[1]))) == (best[2], len(_norm_layout_desc(best[1])))) > 1:
            return False
        chosen[item_idx] = (best[0], best[1])

    if len({idx for idx, _desc in chosen.values()}) != len(items):
        return False
    for item_idx, (_line_idx, desc) in chosen.items():
        items[item_idx]["description"] = desc
    order_by_id = {id(items[item_idx]): line_idx for item_idx, (line_idx, _desc) in chosen.items()}
    items.sort(key=lambda item: order_by_id[id(item)])
    extracted["line_items"] = items
    return True


def _repair_pre_price_stack_descriptions_from_ocr(extracted, unified_text):
    """Map product names before a stacked price block onto matching item amounts."""
    items = [item for item in (extracted.get("line_items") or []) if isinstance(item, dict)]
    if len(items) < 2 or not unified_text:
        return
    lines = [line.strip() for line in unified_text.split('\n')]
    if _repair_catalog_model_detail_stack_descriptions(extracted, lines):
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


def _drop_duplicate_rows_when_subtotal_balances(
    extracted,
    unified_text,
):
    """Drop or merge duplicate parsed rows only when OCR and subtotal agree."""
    items = extracted.get("line_items") or []
    subtotal = extracted.get("subtotal")
    if not items or subtotal is None:
        return
    try:
        target = float(subtotal)
    except (TypeError, ValueError):
        return
    def _norm(text: str) -> str:
        return re.sub(r'\s+', '', str(text or ""))

    def _line_amount(line: str) -> float | None:
        match = re.search(
            r'(?:^|[\s(（])([¥￥]?\s*\d[\d,]*)\s*(?:[%％][*※除軽外内]|[*※除軽外内])?\s*$',
            line,
        )
        if not match:
            return None
        try:
            return float(match.group(1).strip().lstrip('¥￥').replace(',', ''))
        except ValueError:
            return None

    def _single_ocr_desc_has_amount(desc_key: str, amount: float) -> bool:
        lines = [line.strip() for line in unified_text.splitlines() if line.strip()]
        for idx, line in enumerate(lines):
            line_desc = _norm(_clean_ocr_price_line_desc(line))
            if not line_desc or not (desc_key in line_desc or line_desc in desc_key):
                continue
            for nearby in lines[idx + 1:min(idx + 4, len(lines))]:
                nearby_desc = _clean_ocr_price_line_desc(nearby)
                if _valid_ocr_item_desc(nearby_desc):
                    break
                value = _line_amount(nearby)
                if value is not None and abs(value - amount) <= 2:
                    return True
        return False

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

    item_sum = sum(float(item.get("total") or 0) for item in items if isinstance(item, dict))
    if abs(item_sum - target) <= 2:
        for (desc_key, total, discount), group in groups.items():
            if len(group) < 2 or discount:
                continue
            if desc_key and text_norm.count(desc_key) >= len(group):
                continue
            if len({item.get("tax_category") for item in group}) > 1:
                continue
            combined = sum(float(item.get("total") or 0) for item in group)
            if not _single_ocr_desc_has_amount(desc_key, combined):
                continue
            keep = group[0]
            keep["qty"] = 1
            keep["unit_price"] = combined
            keep["total"] = combined
            for duplicate in group[1:]:
                items.remove(duplicate)
            return

    overage = item_sum - target
    if overage <= 0 or overage > 1000:
        return

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
