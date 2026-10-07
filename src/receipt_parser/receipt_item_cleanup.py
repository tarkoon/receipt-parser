"""Receipt item and discount cleanup helpers."""

import re
from math import ceil, isfinite
from collections import Counter
from difflib import SequenceMatcher

from .patterns import (
    _BANNER_PHRASE_RE,
    _DECORATIVE_RE,
    _OCR_QTY_NOTATION_RE,
    _OCR_TRAILING_PRICE_RE,
    _OCR_ZONE_END_RE,
    _SKIP_PRICE_LINE,
    _COMPANY_SUFFIX_RE,
    _HEADER_LINE_RE,
    _discount_rate_tokens,
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
    _parse_qty_detail_total,
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
_DISCOUNT_LABEL_RE = re.compile(
    r'\u5272\s*\u5f15|\u5024\s*\u5f15|'
    r'\u4f1a\s*\u54e1(?:\s*\u69d8)?\s*\u5272(?:\s*\u5f15)?(?=\s|[:\uff1a]|\d|$)'
)


def _printed_fixed_discount_row(lines: list[str]) -> tuple[float, float, float] | None:
    """Read a complete gross, amount-off, quantity, net row; never infer a rate."""
    match = re.fullmatch(
        r'[¥￥]?\s*(\d[\d,]*)\s+[¥￥]?\s*(\d[\d,]*)\s*[引号]\s*'
        r'(\d+(?:\.\d+)?)\s*[個点]\s+[¥￥]?\s*(\d[\d,]*)\s*(?:軽|※)?',
        ' '.join(line.strip() for line in lines if line.strip()),
    )
    if not match:
        return None
    gross, discount, qty, net = (float(value.replace(',', '')) for value in match.groups())
    # ponytail: multi-unit rows need an explicit unit-versus-row price role.
    if qty != 1 or not 0 < discount < gross or gross - discount != net:
        return None
    return gross, discount, net


def _discount_line_price(line: str) -> float | None:
    match = _OCR_TRAILING_PRICE_RE.search(line) or re.search(
        r'(?:^|\s)([¥￥]?\s*\d[\d,]*)\s*[A-ZＡ-Ｚ]\s*$', line
    )
    if not match:
        return None
    try:
        return float(match.group(1).strip().lstrip('¥￥').replace(',', ''))
    except ValueError:
        return None


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
    try:
        extracted_unit_count = sum(float(item.get("qty") or 1) for item in items)
    except (TypeError, ValueError):
        extracted_unit_count = None
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

    def _find_desc_line(desc: str, *, require_unique: bool = False) -> int | None:
        desc_norm = _norm(desc)
        if len(desc_norm) < 3:
            return None
        best: tuple[float, int] | None = None
        match_count = 0
        for idx, line_norm in enumerate(line_norms):
            if len(line_norm) < 3:
                continue
            if line_norm and (desc_norm in line_norm or line_norm in desc_norm):
                score = 1.0
            else:
                score = SequenceMatcher(None, desc_norm, line_norm).ratio()
            if score >= 0.86 and (best is None or score > best[0]):
                best = (score, idx)
            if score >= 0.86:
                match_count += 1
        if require_unique and match_count != 1:
            return None
        return best[1] if best else None

    basket_balanced = current_gap <= 2 and (
        printed_count is None
        or (
            extracted_unit_count is not None
            and abs(extracted_unit_count - printed_count) < 0.01
        )
    )

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
        # A complete literal title owns its digits; its attached price owns money.
        if (not re.search(r'[¥￥]', lines[line_idx])
                and re.sub(r'\s+', '', lines[line_idx]).casefold()
                == re.sub(r'\s+', '', str(item.get("description") or "")).casefold()):
            amount = None
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
        if basket_balanced:
            first_desc = _norm(first.get("description") or "")
            second_desc = _norm(second.get("description") or "")
            first_line = _find_desc_line(first.get("description") or "", require_unique=True)
            second_line = _find_desc_line(second.get("description") or "", require_unique=True)
            if (
                first_desc == second_desc
                or first_line is None
                or second_line != first_line + 1
                or second_line + 2 >= len(lines)
            ):
                continue
            previous_line = lines[first_line - 1] if first_line else ""
            following_idx = second_line + 3
            following_line = lines[following_idx] if following_idx < len(lines) else ""
            if (
                _amount_from_line(lines[first_line]) is not None
                or _amount_from_line(lines[second_line]) is not None
                or _norm(lines[second_line + 1])
                or _norm(lines[second_line + 2])
                or (
                    previous_line
                    and _amount_from_line(previous_line) is None
                    and _valid_ocr_item_desc(_clean_ocr_price_line_desc(previous_line))
                )
                or (
                    _amount_from_line(following_line) is not None
                    and not _norm(following_line)
                )
            ):
                continue
            first_amount = _amount_from_line(lines[second_line + 1])
            second_amount = _amount_from_line(lines[second_line + 2])
            if first_amount is None or second_amount is None:
                continue
            item_amounts = tuple(round(value, 2) for value in (first_total, second_total))
            ocr_amounts = tuple(round(value, 2) for value in (first_amount, second_amount))
            if (
                ocr_amounts[0] == ocr_amounts[1]
                or Counter(item_amounts) != Counter(ocr_amounts)
                or item_amounts == ocr_amounts
            ):
                continue
            first["qty"] = 1.0
            first["unit_price"] = first_amount
            first["total"] = first_amount
            second["qty"] = 1.0
            second["unit_price"] = second_amount
            second["total"] = second_amount
            return
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
        if (
            new_gap >= current_gap
            and printed_count is not None
            and extracted_unit_count is not None
            and abs(extracted_unit_count - printed_count - 1) < 0.01
        ):
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

    def _rate_matches(gross: float, discount: float, discount_rate: str) -> bool:
        rates = _discount_rate_tokens(discount_rate, full_match=True, allow_unmarked=True)
        if not rates:
            return True
        expected = gross * (rates[0] / 100.0)
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
        try:
            qty = float(item.get("qty") or 1)
            unit = float(item.get("unit_price"))
            total = float(item.get("total"))
        except (TypeError, ValueError):
            pass
        else:
            if abs(qty * unit - discount - total) <= 1:
                continue
        desc = item.get("description") or ""
        for idx, line in enumerate(lines):
            if desc and desc not in line:
                continue
            inline_gross = _discount_line_price(line)
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
                gross = _discount_line_price(lines[j])
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
            rates = _discount_rate_tokens(lines[j])
            if rates:
                discount_rate = f"{rates[-1]:g}%"
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

    def _norm(text: str) -> str:
        return re.sub(r'\s+', '', str(text or ""))

    desc_counts: dict[str, int] = {}
    for item in items:
        if isinstance(item, dict):
            key = _norm(item.get("description") or "")
            if key:
                desc_counts[key] = desc_counts.get(key, 0) + 1

    for idx, line in enumerate(lines):
        gross = _discount_line_price(line)
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
        if re.fullmatch(r'1\s*[@eEoOº°⚫●.]?\s+\d[\d,.]*(?:\s*[ET])?', text):
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

    pack_size_re = re.compile(r'\d{1,4}\s*(?:PC(?:S)?|個\s*入)', re.IGNORECASE)

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
    for line_idx, line in enumerate(item_lines):
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
        if (
            pack_size_re.fullmatch(line)
            and pending_descs
            and line_idx > 0
            and _clean_desc(item_lines[line_idx - 1]) == pending_descs[-1]
            and line_idx + 1 < len(item_lines)
            and _is_control_or_numeric(item_lines[line_idx + 1])
        ):
            pending_descs[-1] = f"{pending_descs[-1]} {_clean_desc(line)}"
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
                            if (
                                _OCR_ZONE_END_RE.search(ocr_lines[j])
                                or _valid_ocr_item_desc(ocr_lines[j].strip())
                            ):
                                break
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
            # ponytail: counted rows need printed quantity proof before changing their total.
            if qty > 1:
                continue
            expected = qty * unit_price
            if abs(expected - total) > 1:
                item["total"] = expected


def _clear_discounts_without_nearby_ocr_marker(items, unified_text, *, rates_only=False):
    """Clear LLM discounts when OCR does not place a discount by that item."""
    if not items:
        return
    lines = unified_text.split('\n')

    def _norm(text: str) -> str:
        text = re.sub(r'^\d{4,}[A-Za-z0-9-]*\)?\s*', '', text or "")
        text = re.sub(r'[¥￥]?\s*\d[\d,]*\s*(?:[*※除軽↓]|%|％)?\s*$', '', text)
        text = re.sub(r'\s+', '', text)
        text = re.sub(r'[^\wぁ-んァ-ン一-龥]', '', text, flags=re.UNICODE)
        return text.lower() if any(char.isalpha() for char in text.rstrip("円")) else ""

    def _literal_title_key(text: str) -> str:
        return ''.join(char.casefold() for char in text if char.isalnum())

    def _line_has_amount(line: str, amount: float | None) -> bool:
        if amount is None:
            return False
        amount_int = int(round(float(amount)))
        return bool(re.search(r'(?<!\d)' + re.escape(f"{amount_int:,}") + r'|' + re.escape(str(amount_int)) + r'(?!\d)', line))

    def _title_like_line(line: str) -> bool:
        stripped = line.strip()
        return bool(
            stripped
            and not _DISCOUNT_LABEL_RE.search(stripped)
            and not re.search(r'割引|値引|[%％]|単|JAN|Code128', stripped, re.IGNORECASE)
            and not any(pattern.search(stripped) for pattern in (
                _COMPANY_SUFFIX_RE, _HEADER_LINE_RE, _BANNER_PHRASE_RE,
                _SKIP_PRICE_LINE, _OCR_ZONE_END_RE,
            ))
            and not _OCR_QTY_NOTATION_RE.search(stripped)
            and not _DECORATIVE_RE.fullmatch(stripped)
            and bool(re.search(r'[A-Za-z]|[ぁ-んァ-ン一-龥]', stripped))
        )

    def _positive_amount(line: str) -> float | None:
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith('-')
            or _DISCOUNT_LABEL_RE.search(stripped)
            or _discount_rate_tokens(stripped, full_match=True)
            or _SKIP_PRICE_LINE.search(stripped)
            or _OCR_QTY_NOTATION_RE.search(stripped)
        ):
            return None
        match = re.search(
            r'(?<![-\d])(?:[¥￥]\s*)?(\d[\d,]*)\s*'
            r'(?:[%％*＊※除軽非Xx↓A-Za-zＡ-Ｚ]\s*)*$',
            stripped,
        )
        if not match:
            return None
        return float(match.group(1).replace(',', ''))

    # Own printed percentage schedules and amount-only discounts separately.
    # An amount-only bundle proves the money, but does not print a rate.
    owner_lines: dict[int, int] = {}
    cursor = 0
    for item_idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        desc_norm = _norm(item.get("description") or "")
        if len(desc_norm) < 2:
            continue
        candidates: list[tuple[float, int]] = []
        for line_idx in range(cursor, len(lines)):
            if (
                _COMPANY_SUFFIX_RE.search(lines[line_idx])
                or _HEADER_LINE_RE.search(lines[line_idx])
                or _BANNER_PHRASE_RE.search(lines[line_idx])
                or _SKIP_PRICE_LINE.search(lines[line_idx])
            ):
                continue
            line_norm = _norm(lines[line_idx])
            if len(line_norm) < 2:
                line_norm = ""
            if line_norm and (desc_norm in line_norm or line_norm in desc_norm):
                score = 1.0
            else:
                score = SequenceMatcher(None, desc_norm, line_norm).ratio() if line_norm else 0.0
            if score >= 0.8:
                candidates.append((score, line_idx))
            target_key = _literal_title_key(item.get("description") or "")
            for end_idx in range(line_idx + 1, min(len(lines), line_idx + 4)):
                parts = lines[line_idx:end_idx + 1]
                if not all(_title_like_line(part) for part in parts):
                    break
                if _literal_title_key(''.join(parts)) == target_key:
                    candidates.append((1.0, line_idx))
                    break
                if _positive_amount(parts[-1]) is not None:
                    break
        if not candidates:
            continue
        best_score = max(score for score, _line_idx in candidates)
        owner = min(
            line_idx
            for score, line_idx in candidates
            if abs(score - best_score) <= 0.01
        )
        owner_lines[item_idx] = owner
        cursor = owner + 1

    def _negative_amounts(block: list[str]) -> list[float]:
        return [
            float(match.group(1).replace(',', ''))
            for line in block
            if not _discount_rate_tokens(line, full_match=True)
            for match in re.finditer(r'-\s*[¥￥\\]?\s*(\d[\d,]*)', line)
        ]

    zone_start = min(owner_lines.values(), default=0)
    boundary_start = max(owner_lines.values(), default=-1) + 1
    zone_end = next(
        (
            idx
            for idx in range(boundary_start, len(lines))
            if _OCR_ZONE_END_RE.match(lines[idx].strip())
        ),
        len(lines),
    )
    def _source_title_at(
        line_idx: int, end: int, owner_idx: int, owner_desc: str
    ) -> bool:
        line = lines[line_idx].strip()
        if not _title_like_line(line):
            return False
        desc_key = _literal_title_key(owner_desc)
        # Adjacent OCR title fragments that exactly form this model title are
        # one owner, not a boundary between rows.
        if desc_key and line_idx > owner_idx:
            for start in range(owner_idx, line_idx):
                if _literal_title_key(''.join(lines[start:line_idx + 1])) == desc_key:
                    return False
        # A count fragment belongs to its POS/barcode row only when the
        # complete gross, printed discount and repeated unit price close it.
        if (
            line_idx == owner_idx + 2
            and re.fullmatch(r'[Xx×]\s*\d+', line)
            and re.fullmatch(r'\d{1,4}\s+\d{10,14}', lines[line_idx - 1].strip())
            and not any(_literal_title_key(str(item.get("description") or ""))
                        == _literal_title_key(line) for item in items if isinstance(item, dict))
        ):
            owners = [item for item in items if isinstance(item, dict)
                      and _literal_title_key(str(item.get("description") or "")) == desc_key]
            price = _positive_amount(lines[line_idx + 1]) if line_idx + 1 < end else None
            repeated = [idx for idx in range(line_idx + 2, min(end, line_idx + 8))
                        if (match := re.fullmatch(r'@\s*(\d[\d,]*)', lines[idx].strip()))
                        and float(match.group(1).replace(',', '')) == price]
            if len(owners) == len(repeated) == 1 and price is not None:
                owner_item = owners[0]
                block = lines[line_idx + 1:repeated[0]]
                if (
                    _exact_source_title(owner_item, owner_idx, price)
                    and float(owner_item.get("qty") or 1) == 1
                    and owner_item.get("unit_price") == price
                    and sum(_negative_amounts(block)) == owner_item.get("discount")
                    and price - float(owner_item.get("discount") or 0) == owner_item.get("total")
                    and _supported_rate_schedule(block, price, 1.0, price, price_offset=0)
                ):
                    return False
        if _positive_amount(line) is not None:
            return True
        for next_idx in range(line_idx + 1, min(end, line_idx + 4)):
            following = lines[next_idx].strip()
            if _OCR_QTY_NOTATION_RE.search(following):
                continue
            if _positive_amount(following) is not None:
                return True
            if (
                _DISCOUNT_LABEL_RE.search(following)
                or _OCR_ZONE_END_RE.match(following)
                or _title_like_line(following)
            ):
                break
        return False

    def _next_item_started(
        line: str, line_idx: int | None = None, owner_idx: int | None = None,
        owner_desc: str = "",
    ) -> bool:
        if line_idx is None:
            return _title_like_line(line)
        return _source_title_at(
            line_idx, len(lines), line_idx if owner_idx is None else owner_idx,
            owner_desc,
        )

    def _exact_source_title(
        item: dict, owner_idx: int, gross: float, price_idx: int | None = None
    ) -> bool:
        desc = str(item.get("description") or "")
        desc_key = _literal_title_key(desc)
        if not desc_key:
            return False
        for end_idx in range(owner_idx, min(len(lines), owner_idx + 4)):
            source = lines[end_idx].strip()
            if _literal_title_key(source) == desc_key:
                return True
            if _literal_title_key(''.join(lines[owner_idx:end_idx + 1])) == desc_key:
                return True
            if end_idx > owner_idx and _positive_amount(source) is not None:
                break
            # Strip a prefix only when it has the existing explicit long-code
            # shape and the remainder equals the complete model title.
            coded = re.sub(r'^\d{4,}[A-Za-z0-9-]*\)?\s*', '', source)
            if coded != source and _literal_title_key(coded) == desc_key:
                return True
            # A price suffix is removable only when its literal currency or
            # row marker owns the expected gross; otherwise terminal title
            # digits remain meaningful.
            price_suffix = re.fullmatch(
                r'(.*?)\s+(?:[¥￥]\s*)?(\d[\d,]*)\s*((?:[%％*＊※除軽↓]\s*)+)?', source
            )
            if price_suffix:
                amount = float(price_suffix.group(2).replace(',', ''))
                explicit_price = bool(re.search(r'[¥￥]|[*＊※除軽↓]', source))
                if (
                    (explicit_price or price_idx == end_idx)
                    and amount == gross
                    and _literal_title_key(price_suffix.group(1)) == desc_key
                ):
                    return True
            if end_idx > owner_idx and (
                _positive_amount(source) is not None
                or _title_like_line(source)
                or _OCR_ZONE_END_RE.match(source)
            ):
                break
        return False

    exact_owner_items: set[int] = set()

    def _discount_bundle(
        owner: int, owner_end: int, gross: float, discount: float,
        owner_desc: str,
    ) -> tuple[int, int] | None:
        matches: list[tuple[int, int]] = []
        owner_end = min(owner_end, zone_end)
        for price_idx in range(owner, owner_end):
            if price_idx > owner and _source_title_at(
                price_idx, owner_end, owner, owner_desc
            ):
                break
            price = _positive_amount(lines[price_idx])
            if price is None or price != gross:
                continue
            inline_currency_owner = (
                price_idx == owner
                and bool(re.fullmatch(r'.+\s+[¥￥]\s*\d[\d,]*\s*[*＊※除軽↓]?', lines[owner].strip()))
                and _exact_source_title({"description": owner_desc}, owner, gross, price_idx)
            )
            amounts: list[float] = []
            last_discount_idx = price_idx
            for nearby_idx in range(price_idx + 1, min(owner_end, price_idx + 10)):
                nearby_line = lines[nearby_idx].strip()
                if _source_title_at(nearby_idx, owner_end, owner, owner_desc):
                    break
                nearby_amounts = _negative_amounts([nearby_line])
                if nearby_amounts:
                    amounts.extend(nearby_amounts)
                    last_discount_idx = nearby_idx
                    if sum(amounts) == discount:
                        # A bare trailing deduction can belong to a queued
                        # earlier item; a later schedule cannot be omitted.
                        remaining = lines[nearby_idx + 1:owner_end]
                        if not any(_DISCOUNT_LABEL_RE.search(line)
                                   or re.search(r'\d+(?:\.\d+)?\s*[%％]', line)
                                   for line in remaining):
                            matches.append((price_idx, last_discount_idx))
                    continue
                if _positive_amount(nearby_line) is not None:
                    # One bare annotation before a labeled schedule cannot
                    # replace the explicit currency price on the title row.
                    if (
                        inline_currency_owner and nearby_idx == price_idx + 1
                        and re.fullmatch(r'\d+', nearby_line)
                        and nearby_idx + 1 < owner_end
                        and _DISCOUNT_LABEL_RE.search(lines[nearby_idx + 1])
                        and _discount_rate_tokens(lines[nearby_idx + 1])
                    ):
                        continue
                    break
            if amounts and sum(amounts) == discount and (price_idx, last_discount_idx) not in matches:
                matches.append((price_idx, last_discount_idx))
        return matches[0] if len(matches) == 1 else None

    def _supported_rate_schedule(
        block: list[str], gross: float, qty: float, unit: float | None,
        *, price_offset: int,
    ) -> tuple[float, ...]:
        stages: list[tuple[tuple[float, ...], list[float]]] = []
        current_rates: list[float] | None = None
        current_amounts: list[float] = []

        def finish_stage() -> bool:
            nonlocal current_rates, current_amounts
            if current_rates is None:
                return True
            if not current_rates:
                return False
            stages.append((tuple(current_rates), current_amounts))
            current_rates = None
            current_amounts = []
            return True

        for line_idx, line in enumerate(block):
            stripped = line.strip()
            if not stripped or line_idx == price_offset:
                continue
            if _OCR_ZONE_END_RE.match(stripped):
                break
            labeled = bool(_DISCOUNT_LABEL_RE.search(stripped))
            if labeled:
                labeled_rates = _discount_rate_tokens(stripped)
                if not labeled_rates and re.search(r'[%％]', stripped):
                    return ()
                if not labeled_rates and current_rates and not current_amounts:
                    current_amounts.extend(_negative_amounts([stripped]))
                    continue
                if not labeled_rates and _negative_amounts([stripped]):
                    return ()
                if not finish_stage():
                    return ()
                current_rates = list(labeled_rates) or None
                current_amounts = _negative_amounts([stripped])
                continue
            amounts = _negative_amounts([stripped])
            rate_text = re.sub(r'-\s*[¥￥\\]?\s*\d[\d,]*', '', stripped) if amounts else stripped
            rates = _discount_rate_tokens(rate_text, full_match=True)
            if current_rates is None:
                if rates:
                    current_rates = list(rates)
                    current_amounts = amounts
                continue
            if rates and current_amounts:
                if not finish_stage():
                    return ()
                current_rates = list(rates)
                current_amounts = amounts
                continue
            if rates:
                current_rates.extend(rates)
            if amounts:
                current_amounts.extend(amounts)
            elif not rates and _positive_amount(stripped) is not None:
                return ()
            elif (
                not rates
                and not _DECORATIVE_RE.fullmatch(stripped)
                and not re.fullmatch(r'[\s%％\-−]+', stripped)
            ):
                # A schedule stage cannot skip unowned text before its amount.
                return ()
        if not finish_stage() or not stages:
            return ()
        # Column-order OCR can put all labeled rates before their deductions.
        # Pair only the complete printed orders inside this exact item owner.
        amounts = [amount for _rates, deductions in stages for amount in deductions]
        if len(amounts) != len(stages) or amounts != _negative_amounts(block[price_offset + 1:]):
            return ()

        explicit_details = {
            detail
            for line in block
            if (detail := _parse_qty_detail_total(line)) is not None
        }
        per_unit_owned = (
            qty > 1
            and unit is not None
            and qty.is_integer()
            and explicit_details == {(qty, float(unit))}
        )
        remaining_gross = gross
        for (rates, _deductions), amount in zip(stages, amounts, strict=True):
            stage_remaining = 1.0
            for rate in rates:
                if not 0 < rate <= 100:
                    return ()
                stage_remaining *= 1.0 - rate / 100.0
            expected = remaining_gross * (1.0 - stage_remaining)
            if abs(amount - expected) > 1.0:
                per_unit_rounded = qty * ceil(expected / qty) if per_unit_owned else None
                if amount != per_unit_rounded:
                    return ()
            remaining_gross -= amount
        return tuple(rate for rates, _amount in stages for rate in rates)

    supported_rate_ids: set[int] = set()
    supported_bundle_ids: set[int] = set()
    unowned_discount_ids: set[int] = set()
    owner_scope_ends: dict[int, int] = {}
    ordered_owners = sorted(owner_lines.items(), key=lambda pair: pair[1])
    for owner_pos, (item_idx, owner) in enumerate(ordered_owners):
        item = items[item_idx]
        if not isinstance(item, dict) or not (item.get("discount") or 0):
            continue
        next_owner = (
            ordered_owners[owner_pos + 1][1]
            if owner_pos + 1 < len(ordered_owners)
            else len(lines)
        )
        discount = float(item.get("discount") or 0)
        qty = float(item.get("qty") or 1)
        unit = item.get("unit_price")
        total = item.get("total")
        gross = qty * float(unit) if unit is not None else float(total or 0) + discount
        exact_title_owner = _exact_source_title(item, owner, gross)
        owner_end = min(next_owner, zone_end)
        fixed = _printed_fixed_discount_row(lines[owner + 1:owner_end])
        fixed_owner = (
            exact_title_owner and qty == 1
            and fixed is not None and (gross, discount, total) == fixed
        )
        for source_idx in range(owner + 1, owner_end):
            # Complete closed money/quantity rows cannot start another item.
            if fixed_owner:
                break
            if _source_title_at(
                source_idx, owner_end, owner,
                str(item.get("description") or ""),
            ):
                owner_end = source_idx
                break
        owner_scope_ends[item_idx] = owner_end
        bundle = _discount_bundle(
            owner, owner_end, gross, discount, str(item.get("description") or "")
        )
        exact_title_owner = _exact_source_title(
            item, owner, gross, bundle[0] if bundle else None
        )
        if exact_title_owner:
            exact_owner_items.add(item_idx)
        else:
            unowned_discount_ids.add(id(item))
        fixed_end = owner_end
        fixed = _printed_fixed_discount_row(lines[owner + 1:fixed_end])
        if exact_title_owner and (bundle or (
            fixed is not None
            and qty == 1
            and (gross, discount, total) == fixed
        )):
            supported_bundle_ids.add(id(item))
        price_idx, discount_idx = bundle or (owner, owner)
        schedule = (
            _supported_rate_schedule(
                lines[owner:discount_idx + 1], gross, qty,
                float(unit) if unit is not None else None,
                price_offset=price_idx - owner,
            )
            if bundle and item_idx in exact_owner_items
            else ()
        )
        if not schedule:
            item["discount_rate"] = ""
        else:
            remaining = 1.0
            for rate in schedule:
                remaining *= 1.0 - (rate / 100.0)
            effective = round((1.0 - remaining) * 100, 8)
            item["discount_rate"] = f"{effective:g}%"
            supported_rate_ids.add(id(item))

    for item in items:
        if (
            isinstance(item, dict)
            and (item.get("discount") or 0)
            and id(item) not in supported_rate_ids
        ):
            item["discount_rate"] = ""

    if rates_only:
        return

    def _supported(item: dict) -> bool:
        if id(item) in supported_bundle_ids:
            return True
        if id(item) in unowned_discount_ids:
            return False
        item_idx = next(
            (i for i, candidate in enumerate(items) if candidate is item), -1
        )
        owner_idx = owner_lines.get(item_idx)
        owner_desc = str(item.get("description") or "")
        scope_end = owner_scope_ends.get(item_idx, len(lines))
        # A complete ordered title group can precede its flattened price
        # column. Later source titles still bound money; percentages retain
        # the stricter individual owner above.
        gross = float(item.get("total") or 0) + float(item.get("discount") or 0)
        first_price = next((idx for idx in range(zone_start, zone_end)
                            if _positive_amount(lines[idx]) is not None), zone_end)
        leading_titles = [_literal_title_key(line) for line in lines[zone_start:first_price]
                          if _title_like_line(line)]
        modeled_titles = [_literal_title_key(str(items[index].get("description") or ""))
                          for index, line_idx in ordered_owners if line_idx < first_price]
        if item_idx in exact_owner_items and leading_titles and leading_titles == modeled_titles:
            money_bundles = [bundle for idx, line in enumerate(lines[:zone_end])
                             if _positive_amount(line) == gross
                             and not any(_source_title_at(source_idx, zone_end, owner_idx, owner_desc)
                                         for source_idx in range(first_price, idx))
                             and (bundle := _discount_bundle(idx, zone_end, gross,
                                 float(item.get("discount") or 0), owner_desc)) is not None]
            if len(money_bundles) == 1:
                return True
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
        if owner_idx is not None:
            candidate_idxs = [idx for idx in candidate_idxs if owner_idx <= idx < scope_end]

        def _has_matching_discount_amount(text: str) -> bool:
            if discount_value <= 0:
                return False
            m = re.fullmatch(r'\s*-\s*[¥￥\\]?\s*(\d[\d,]*)\s*', text)
            if not m:
                return False
            amount = float(m.group(1).replace(',', ''))
            return abs(amount - discount_value) <= 2

        def _has_matching_rate_marker(text: str) -> bool:
            rates = _discount_rate_tokens(text, full_match=True)
            if not rates:
                return False
            if discount_rate:
                row_rates = _discount_rate_tokens(
                    discount_rate, full_match=True, allow_unmarked=True
                )
                if row_rates and abs(row_rates[0] - rates[0]) <= 0.1:
                    return True
            gross = float(unit or 0)
            if gross <= 0 and total is not None:
                gross = float(total) + discount_value
            return gross > 0 and abs(discount_value - gross * (rates[0] / 100.0)) <= max(2.0, gross * 0.03)

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
                if _next_item_started(nxt, j, owner_idx, owner_desc):
                    break
                if any(_line_has_amount(nxt, amount) for amount in search_amounts):
                    saw_item_amount = True
                    continue
                if _DISCOUNT_LABEL_RE.search(nxt):
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


def _detect_ocr_discounts(items, unified_text, *, fixed_only=False):
    """Detect discounts within their item owners.

    Trigger: adjacent top title candidates exactly form one description.
    Invariant: join only that complete span; repeated or nonadjacent owners
    remain ambiguous, and later independent discount owners retain boundaries.
    fixed_only restores literal qty=1 gross/amount-off fields while retaining
    the current net amount and complete title digits of a balanced basket.
    """
    ocr_lines = unified_text.split('\n')

    def _norm_discount_desc(text: str) -> str:
        text = re.sub(r'^\d{4,}[A-Za-z0-9-]*\)?\s*', '', text or "")
        text = re.sub(r'[¥￥]?\s*\d[\d,]*\s*[*※除軽]?\s*$', '', text)
        text = re.sub(r'\s+', '', text)
        text = re.sub(r'[^\wぁ-んァ-ン一-龥]', '', text, flags=re.UNICODE)
        return text.lower() if any(char.isalpha() for char in text.rstrip("円")) else ""

    owner_lines: dict[int, int] = {}
    unique_owner_items: set[int] = set()
    exact_owner_items: set[int] = set()
    cursor = 0
    for item_idx, item in enumerate(items):
        if not isinstance(item, dict):
            return
        norm_desc = _norm_discount_desc(item.get("description") or "")
        if len(norm_desc) < 2:
            continue
        literal_desc = ''.join(char.casefold() for char in str(item.get("description") or "") if char.isalnum())
        candidates: list[tuple[float, int]] = []
        for line_idx in range(cursor, len(ocr_lines)):
            line = ocr_lines[line_idx].strip()
            if (
                not line
                or _OCR_ZONE_END_RE.match(line)
                or '割引' in line
                or '値引' in line
                or _DECORATIVE_RE.fullmatch(line)
                or _BANNER_PHRASE_RE.search(line)
            ):
                continue
            norm_line = _norm_discount_desc(line)
            if len(norm_line) < 2:
                continue
            if ''.join(char.casefold() for char in line if char.isalnum()) == literal_desc:
                score = 2.0
            elif norm_desc in norm_line or norm_line in norm_desc:
                score = 1.0
            else:
                score = SequenceMatcher(None, norm_desc, norm_line).ratio()
            if score >= 0.72:
                candidates.append((score, line_idx))
        if not candidates:
            continue
        best_score = max(score for score, _line_idx in candidates)
        best_lines = [
            line_idx
            for score, line_idx in candidates
            if abs(score - best_score) <= 0.01
        ]
        remaining_same = sum(
            (
                ''.join(char.casefold() for char in str(later.get("description") or "") if char.isalnum())
                == literal_desc
                if best_score == 2.0
                else _norm_discount_desc(later.get("description") or "") == norm_desc
            )
            for later in items[item_idx:]
            if isinstance(later, dict)
        )
        split_title_span = (
            len(best_lines) > 1
            and len(best_lines) != remaining_same
            and best_lines == list(range(best_lines[0], best_lines[-1] + 1))
            and _norm_discount_desc(" ".join(ocr_lines[idx] for idx in best_lines)) == norm_desc
            and not any(
                _norm_discount_desc(other.get("description") or "")
                in {_norm_discount_desc(ocr_lines[idx]) for idx in best_lines}
                for other_idx, other in enumerate(items)
                if other_idx != item_idx and isinstance(other, dict)
            )
        )
        if len(best_lines) > 1 and len(best_lines) != remaining_same and not split_title_span:
            return
        # ponytail: exact adjacent fragments only; other ambiguous owners still abstain.
        line_idx = best_lines[-1] if split_title_span else min(best_lines)
        owner_lines[item_idx] = line_idx
        if len(best_lines) == 1 or split_title_span:
            unique_owner_items.add(item_idx)
            if (split_title_span or norm_desc == _norm_discount_desc(ocr_lines[line_idx])) and (
                not fixed_only
                or literal_desc == ''.join(
                    char.casefold() for char in ' '.join(ocr_lines[idx] for idx in best_lines)
                    if char.isalnum()
                )
            ):
                exact_owner_items.add(item_idx)
        cursor = line_idx + 1

    def _apply_counted_bundle_offer():
        # ponytail: one exact amount-only counted offer; add formats after full owner proof.
        offer_re = re.compile(
            r'[<＜（(]?\s*(?P<title>.+?)\s*(?P<count>\d+)\s*'
            r'(?P<unit>足|個|点|本|組|枚|袋|杯)\s*'
            r'(?:[¥￥]\s*)?(?P<net>\d[\d,]*)\s*円\s*[>＞）)]?'
        )
        negative_re = re.compile(r'[-−－]\s*[¥￥]?\s*(\d[\d,]*)')
        money_re = re.compile(r'[¥￥]?\s*(\d[\d,]*)\s*(?:円)?')
        unit_price_re = re.compile(r'[@＠]\s*[¥￥]?\s*(\d[\d,]*)')
        count_re = re.compile(
            r'(?:お買上(?:商品)?点数|お買上商品数|お買上げ点数)'
            r'\s*[:：]?\s*(\d+)\s*(?:点|個)?'
        )
        qty_token_re = re.compile(r'(?<!\d)(\d+)\s*(?:個|点|本|組|枚|袋|杯|足|コ|ケ|ヶ)')
        percentage_re = re.compile(r'\d+(?:[.,]\d+)?\s*[%％]')
        control_re = re.compile(
            r'^\s*(?:小計|合計|値引|割引|クーポン|返金|お支払|ポイント|預り|お釣り)'
        )

        def title_key(value):
            value = re.sub(r'^[*＊※]\s*', '', str(value or '').strip())
            return re.sub(r'\s+', '', value).casefold()

        def number(value):
            if value is None or isinstance(value, bool):
                return None
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError):
                return None
            return value if isfinite(value) else None

        offers = [(i, match) for i, line in enumerate(ocr_lines)
                  if (match := offer_re.fullmatch(line.strip()))]
        if not offers:
            return None
        if len(offers) != 1:
            return False

        offer_index, offer = offers[0]
        group_count = int(offer['count'])
        offer_net = float(offer['net'].replace(',', ''))
        offer_title = title_key(offer['title'])
        deduction_match = (
            negative_re.fullmatch(ocr_lines[offer_index + 1].strip())
            if offer_index + 1 < len(ocr_lines) else None
        )
        total_owners = []
        for index, line in enumerate(ocr_lines):
            match = re.fullmatch(r'合計\s*[¥￥]\s*(\d[\d,]*)', line.strip())
            if match:
                total_owners.append((index, float(match.group(1).replace(',', ''))))
            elif line.strip() == '合計':
                if index + 1 < len(ocr_lines):
                    match = money_re.fullmatch(ocr_lines[index + 1].strip())
                    if match:
                        total_owners.append((index, float(match.group(1).replace(',', ''))))
                # Row grouping may put the total beside the preceding item count.
                match = re.fullmatch(r'(\d+\s*[点個])\s+[¥￥]\s*(\d[\d,]*)',
                                     ocr_lines[index - 1].strip()) if index > 1 else None
                if match and count_re.fullmatch(ocr_lines[index - 2] + '\n' + match.group(1)):
                    total_owners.append((index, float(match.group(2).replace(',', ''))))
        count_matches = list(count_re.finditer(unified_text))
        if (
            group_count < 2 or not offer_title or deduction_match is None
            or len(total_owners) != 1 or len(count_matches) != 1
        ):
            return False
        deduction = float(deduction_match.group(1).replace(',', ''))
        total_index, printed_total = total_owners[0]
        count_index = unified_text[:count_matches[0].start()].count('\n')
        if (
            deduction <= 0 or offer_net <= 0 or offer_index >= total_index
            or count_index >= total_index
            or any(control_re.match(line) for line in ocr_lines[:total_index]
                   if line.strip() != ocr_lines[offer_index])
        ):
            return False
        negatives = [
            (index, float(match.group(1).replace(',', '')))
            for index, line in enumerate(ocr_lines[:total_index])
            if (match := negative_re.fullmatch(line.strip()))
        ]
        if negatives != [(offer_index + 1, deduction)]:
            return False

        row_ids = set(range(len(items)))
        if set(owner_lines) != row_ids:
            return False
        titles = [title_key(item.get('description')) if isinstance(item, dict) else ''
                  for item in items]
        owners = [owner_lines[index] for index in range(len(items))]
        if (
            any(not title or owner >= offer_index or title_key(ocr_lines[owner]) != title
                for title, owner in zip(titles, owners, strict=True))
            or len(set(owners)) != len(items) or owners != sorted(owners)
            or [index for index, line in enumerate(ocr_lines[:offer_index])
                if title_key(line) in set(titles)] != owners
        ):
            return False
        if any(percentage_re.search(line) for line in ocr_lines[owners[0]:total_index + 1]):
            return False

        group = [index for index, title in enumerate(titles) if title.endswith(offer_title)]
        if (
            len(group) != group_count
            or group != list(range(group[0], group[0] + group_count))
            or len({titles[index] for index in group}) != 1
        ):
            return False

        gross_rows, group_units, quantities, group_state = [], [], [], []
        for index, (item, owner) in enumerate(zip(items, owners, strict=True)):
            end = owners[index + 1] if index + 1 < len(owners) else offer_index
            segment = [line.strip() for line in ocr_lines[owner + 1:end]]
            markers = [float(match.group(1).replace(',', '')) for line in segment
                       if (match := unit_price_re.search(line))]
            qty_tokens = [float(match.group(1)) for line in segment
                          for match in qty_token_re.finditer(unit_price_re.sub('', line))]
            qty, unit, total = (number(item.get(key)) for key in ('qty', 'unit_price', 'total'))
            if 'discount' not in item:
                return False
            raw_discount = item['discount']
            discount = number(0 if raw_discount is None else raw_discount)
            if (
                qty is None or qty <= 0 or unit is None or unit <= 0
                or total is None or total <= 0 or discount is None or discount < 0
                or str(item.get('discount_rate') or '').strip()
                or len(markers) != 1 or abs(markers[0] - unit) > 0.01
                or (qty_tokens and qty_tokens != [qty])
            ):
                return False
            gross = unit * qty
            gross_rows.append(gross)
            quantities.append(qty)
            if index in group:
                if qty != 1 or len(markers) != 1:
                    return False
                group_units.append(unit)
                group_state.append((item, unit, total, discount))
            else:
                if abs(discount) > 0.01 or abs(total - gross) > 0.01:
                    return False
                if qty > 1:
                    if qty != round(qty) or len(markers) != 1 or qty_tokens != [qty]:
                        return False

        # Owned unit/count packets determine every gross independently. The
        # complete bare-price multiset may interleave across adjacent titles.
        printed_gross = [float(match.group(1).replace(',', ''))
                         for line in ocr_lines[owners[0]:offer_index]
                         if (match := money_re.fullmatch(line.strip()))]
        if Counter(printed_gross) != Counter(gross_rows):
            return False
        share = deduction / group_count
        pristine = all(abs(discount) <= 0.01 and abs(total - unit) <= 0.01
                       for _, unit, total, discount in group_state)
        applied = all(abs(discount - share) <= 0.01 and abs(total - (unit - share)) <= 0.01
                      for _, unit, total, discount in group_state)
        missing_discount = all(abs(discount) <= 0.01 and abs(total - (unit - share)) <= 0.01
                               for _, unit, total, discount in group_state)
        if (
            len(group_units) != group_count or max(group_units) - min(group_units) > 0.01
            or abs(group_count * group_units[0] - offer_net - deduction) > 0.01
            or abs(share - round(share)) > 0.000001
            or abs(sum(quantities) - int(count_matches[0].group(1))) > 0.000001
            or not (pristine or applied or missing_discount)
            or (fixed_only and pristine)
            or abs(sum(gross_rows) - deduction - printed_total) > 0.01
            or (applied and abs(sum(number(item.get('total')) for item in items)
                                - printed_total) > 0.01)
        ):
            return False

        if pristine or missing_discount:
            updates = [(item, share, unit - share)
                       for item, unit, _, _ in group_state]
            for item, discount, total in updates:
                item.update(discount=discount, total=total, discount_rate='')
        return True

    result = _apply_counted_bundle_offer()
    if result is not None:
        return

    # Distinct local reductions may be summed only as one fully owned basket
    # whose printed aggregate discount and subtotal independently reconcile.
    summary_indices = [idx for idx, line in enumerate(ocr_lines) if '値引合計' in line]
    if not fixed_only and summary_indices and len(unique_owner_items) == len(items):
        summary_idx = min(summary_indices)
        aggregate = re.fullmatch(
            r'[^0-9-]*値引合計\s*-\s*[¥￥]?(\d[\d,]*)[)）]?\s*',
            ' '.join(ocr_lines[summary_idx:summary_idx + 2]).strip(),
        ) if len(summary_indices) == 1 else None
        zone_end = next(
            (idx for idx in range(max(owner_lines.values()) + 1, summary_idx + 1)
             if re.match(r'^[（(]?\s*(?:商品代金|値引合計|小計|合計)', ocr_lines[idx])),
            summary_idx,
        )
        proposals = {}
        has_multiple = False
        for item_idx, owner in owner_lines.items():
            item = items[item_idx]
            end = min((idx for idx in owner_lines.values() if idx > owner), default=zone_end)
            end = min(end, zone_end)
            gross = []
            discounts = []
            labelled = False
            valid = (
                item.get('qty') == 1
                and item_idx in exact_owner_items
            )
            for line in ocr_lines[owner + 1:end]:
                line = line.strip()
                if not line:
                    continue
                if _DISCOUNT_LABEL_RE.search(line) and not re.search(r'\d', line):
                    labelled = True
                    continue
                negative = re.fullmatch(r'-\s*[¥￥]?(\d[\d,]*)', line)
                positive = re.fullmatch(r'[*＊¥￥]?\s*(\d[\d,]*)\s*[-*＊※軽]?', line)
                if negative:
                    discounts.append(float(negative.group(1).replace(',', '')))
                    valid &= labelled
                    labelled = False
                elif positive and not labelled:
                    gross.append(float(positive.group(1).replace(',', '')))
                else:
                    valid = False
            has_multiple |= len(discounts) > 1
            if (valid and not labelled and len(gross) == 1
                    and gross[0] == item.get('unit_price') and sum(discounts) < gross[0]):
                proposals[item_idx] = (gross[0], sum(discounts))
        if has_multiple:
            subtotals = {
                float(amount.replace(',', ''))
                for amount in re.findall(r'小計[^\n]*\n\s*[¥￥]?(\d[\d,]*)', unified_text)
            }
            if (aggregate and len(subtotals) == 1 and len(proposals) == len(items)
                    and sum(discount for gross, discount in proposals.values())
                    == float(aggregate.group(1).replace(',', ''))
                    and sum(gross - discount for gross, discount in proposals.values()) in subtotals):
                for item_idx, (gross, discount) in proposals.items():
                    items[item_idx].update(
                        unit_price=gross, discount=discount, total=gross - discount, discount_rate='',
                    )
            return

    for item_idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        li = owner_lines.get(item_idx)
        if li is None:
            continue
        next_owner_line = min(
            (line for line in owner_lines.values() if line > li),
            default=len(ocr_lines),
        )
        fixed_end = next(
            (idx for idx in range(li + 1, next_owner_line) if _OCR_ZONE_END_RE.match(ocr_lines[idx].strip())),
            next_owner_line,
        )
        fixed = _printed_fixed_discount_row(ocr_lines[li + 1:fixed_end])
        if fixed is not None and item_idx in exact_owner_items:
            gross, discount, net = fixed
            if not fixed_only or item.get("qty") == 1 and item.get("total") == net:
                item.update(qty=1, unit_price=gross, discount=discount, total=net, discount_rate="")
                continue
        if fixed_only:
            continue
        if (item.get("discount") or 0) > 0:
            continue
        for offset in range(1, 8):
            if li + offset >= len(ocr_lines):
                break
            next_line = ocr_lines[li + offset].strip()
            is_discount_line = '割引' in next_line or '値引' in next_line
            if (
                _DECORATIVE_RE.fullmatch(next_line)
                or (_BANNER_PHRASE_RE.search(next_line) and not is_discount_line)
                or _OCR_ZONE_END_RE.match(next_line)
                or re.search(
                    r'支払|決済|現金|カード|電子マネー|Pay|VISA|Master',
                    next_line,
                    re.IGNORECASE,
                )
            ):
                break
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
            if is_discount_line:
                rate_str = ""
                discount_amount = 0
                discount_amounts = []
                missing_gross = item.get("unit_price") is None
                scan_end = next_owner_line if missing_gross else len(ocr_lines)
                for k in range(li + offset, min(li + offset + 4, scan_end)):
                    kline = ocr_lines[k].strip()
                    # Rate may appear inline ("割引: 20%") or alone ("10%").
                    has_percent_syntax = bool(re.search(r'\d+(?:\.\d+)?\s*[%％]', kline))
                    rates = _discount_rate_tokens(kline)
                    if rates:
                        rate_str = f"{rates[-1]:g}%"
                    # Amount line: accept "-38", "-¥24", "-￥24" with optional yen sign.
                    amt_match = re.match(r'^-\s*[¥￥]?\s*(\d[\d,.]*)\s*$', kline)
                    if missing_gross and k > li + offset and kline and not (has_percent_syntax or amt_match):
                        break
                    if amt_match:
                        amt_str = amt_match.group(1).replace(',', '')
                        if '.' in amt_str and float(amt_str) < 10:
                            amt_str = amt_str.replace('.', '')
                        discount_amount = float(amt_str)
                        discount_amounts.append(discount_amount)
                if discount_amount > 0 and (not missing_gross or len(discount_amounts) == 1):
                    up = item.get("unit_price")
                    if up is None:
                        price_lines = [
                            line for line in ocr_lines[li:li + offset]
                            if re.search(r'[¥￥]\s*\d[\d,]*', line)
                        ]
                        if (
                            item.get("qty") != 1
                            or item_idx not in unique_owner_items
                            or li + offset >= next_owner_line
                            or len(price_lines) != 1
                            or len(re.findall(r'[¥￥]\s*\d[\d,]*', price_lines[0])) != 1
                        ):
                            break
                        up = _discount_line_price(price_lines[0])
                        if up is None or up <= discount_amount:
                            break
                        item["unit_price"] = up
                    item["discount"] = discount_amount
                    item["discount_rate"] = rate_str
                    item["total"] = item.get("qty", 1) * up - discount_amount
                break

    if not fixed_only:
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
        rates = _discount_rate_tokens(
            item.get("discount_rate") or "", full_match=True, allow_unmarked=True
        )
        if not rates:
            return None
        return rates[0] / 100.0

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
            rate=t.get("rate"), amount=t.get("amount"),
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
