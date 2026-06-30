"""Receipt item total and layout projection helpers."""

import re
from difflib import SequenceMatcher

from .patterns import (
    _GENERIC_DESC_MARKERS,
    _OCR_QTY_NOTATION_RE,
    _OCR_TRAILING_PRICE_RE,
    _OCR_ZONE_END_RE,
    _QTY_DETAIL_DESC_RE,
    _SKIP_PRICE_LINE,
)
from .receipt_financial import (
    _parse_amount_fragment,
)
from .receipt_totals import _canonical_subtotal_from_taxes, _line_items_sum


def _merge_qty_detail_into_previous(items, unified_text):
    """Collapse qty-detail phantom items into the preceding product.

    When the LLM extracts a qty-detail OCR fragment (e.g. "(2個 X 単70)") as
    a standalone item, the receipt's preceding product is actually priced
    at qty × unit. Use the OCR text (not the LLM's possibly-wrong qty) to
    extract qty/unit, apply them to the previous item, then drop the
    phantom.

    Safety: only merges when (a) the phantom's description matches the
    qty-detail regex AND the OCR text near a qty-detail fragment yields a
    consistent (qty, unit) pair (qty ≥ 2, unit > 0); and (b) a previous
    item exists with qty == 1.
    """
    if len(items) < 2:
        return
    ocr_lines = unified_text.split('\n')
    to_drop: set[int] = set()
    for i, item in enumerate(items):
        if i == 0 or not isinstance(item, dict):
            continue
        desc = (item.get("description") or "").strip()
        if not desc or not _QTY_DETAIL_DESC_RE.match(desc):
            continue
        # Extract qty/unit from OCR (more reliable than the LLM's parse).
        ocr_qty: float | None = None
        ocr_unit: float | None = None
        for ocr_line in ocr_lines:
            if not _QTY_DETAIL_DESC_RE.match(ocr_line.strip()):
                continue
            m = re.search(
                r'(\d+)\s*[コ個点]\s*[xX×]\s*(?:単|@)?\s*(\d[\d,]*)',
                ocr_line,
            )
            if not m:
                m = re.search(
                    r'(?:単|@)\s*(\d[\d,]*)\s*[xX×]\s*(\d+)\s*[コ個点]',
                    ocr_line,
                )
                if m:
                    ocr_unit = float(m.group(1).replace(',', ''))
                    ocr_qty = float(m.group(2))
                    break
            else:
                ocr_qty = float(m.group(1))
                ocr_unit = float(m.group(2).replace(',', ''))
                break
        if ocr_qty is None or ocr_unit is None or ocr_qty < 2 or ocr_unit <= 0:
            continue
        prev = items[i - 1]
        if not isinstance(prev, dict):
            continue
        if prev.get("qty", 1) and float(prev.get("qty", 1)) > 1:
            continue
        prev["qty"] = ocr_qty
        prev["unit_price"] = ocr_unit
        prev["total"] = ocr_qty * ocr_unit
        to_drop.add(i)
    if to_drop:
        items[:] = [it for j, it in enumerate(items) if j not in to_drop]


def _repair_previous_item_from_following_qty_detail(extracted, unified_text):
    """Repair a small item amount followed by a visible qty/unit detail."""
    items = [item for item in (extracted.get("line_items") or []) if isinstance(item, dict)]
    if not items:
        return
    lines = [line.strip() for line in unified_text.split('\n')]
    targets = {
        float(value)
        for value in (extracted.get("subtotal"), extracted.get("total"), _canonical_subtotal_from_taxes(extracted))
        if value is not None and float(value or 0) > 0
    }
    for line in lines:
        for match in re.finditer(r'[¥￥]\s*([\d,]+)', line):
            value = float(match.group(1).replace(',', ''))
            if value > 0:
                targets.add(value)
    if not targets:
        return

    def _norm(text: str) -> str:
        text = re.sub(r'\s+', '', str(text or ""))
        text = re.sub(r'[^\wぁ-んァ-ン一-龥]', '', text, flags=re.UNICODE)
        return text.lower()

    item_sum = _line_items_sum(extracted)
    for item in items:
        desc = str(item.get("description") or "")
        ndesc = _norm(desc)
        if len(ndesc) < 3:
            continue
        try:
            current_total = float(item.get("total") or 0)
            current_qty = float(item.get("qty") or 1)
            discount = float(item.get("discount") or 0)
        except (TypeError, ValueError):
            continue
        if current_qty != 1 or discount > 0 or current_total <= 0:
            continue
        for idx, line in enumerate(lines):
            nline = _norm(_clean_ocr_price_line_desc(line))
            if not nline or not (ndesc in nline or nline in ndesc):
                continue
            seen_current_amount = False
            for lookahead in lines[idx + 1:min(len(lines), idx + 5)]:
                amount = _parse_amount_fragment(
                    lookahead.strip().lstrip('¥￥').rstrip(')）').replace(',', '')
                )
                if amount is not None and abs(amount - current_total) <= 2:
                    seen_current_amount = True
                    continue
                detail = _parse_qty_detail_total(lookahead)
                if not detail:
                    continue
                qty, unit = detail
                gross = qty * unit
                if not seen_current_amount or gross <= current_total:
                    break
                new_sum = item_sum - current_total + gross
                if not any(abs(new_sum - target) <= 2 for target in targets):
                    break
                item["qty"] = qty
                item["unit_price"] = unit
                item["total"] = gross
                item_sum = new_sum
                break
            break


def _clear_discount_when_negative_line_precedes_own_price(extracted, unified_text):
    """Clear discounts attached to an item whose own price prints after the discount."""
    items = extracted.get("line_items") or []
    if not items:
        return
    lines = [line.strip() for line in (unified_text or "").splitlines()]

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

    def _line_has_discount(line: str, discount: float) -> bool:
        m = re.fullmatch(r'\s*-\s*[¥￥\\]?\s*(\d[\d,]*)\s*', line)
        return bool(m and abs(float(m.group(1).replace(',', '')) - discount) <= 2)

    for item in items:
        if not isinstance(item, dict):
            continue
        discount = float(item.get("discount") or 0)
        if discount <= 0:
            continue
        desc_norm = _norm(item.get("description") or "")
        if len(desc_norm) < 3:
            continue
        qty = float(item.get("qty") or 1)
        unit = item.get("unit_price")
        if unit is None:
            continue
        own_amounts = [float(unit)]
        if qty != 1:
            own_amounts.append(float(unit) * qty)
        for idx, line in enumerate(lines):
            norm_line = _norm(line)
            if not norm_line or not (desc_norm in norm_line or norm_line in desc_norm):
                continue
            discount_idx = None
            own_price_idx = None
            for j in range(idx + 1, min(len(lines), idx + 10)):
                if discount_idx is None and _line_has_discount(lines[j], discount):
                    discount_idx = j
                if own_price_idx is None and any(_line_has_amount(lines[j], amount) for amount in own_amounts):
                    own_price_idx = j
                if discount_idx is not None and own_price_idx is not None:
                    break
            if discount_idx is not None and own_price_idx is not None and discount_idx < own_price_idx:
                item["discount"] = 0
                item["discount_rate"] = ""
                item["total"] = qty * float(unit)
            break




def _fix_item_totals_from_ocr_neighborhood(
    items, unified_text, target_subtotal, target_total, canonical_subtotal=None,
):
    """When items_sum is off-target, re-anchor each item's total to the price
    immediately following its description in OCR text.

    Generic-purpose: handles 2-column receipts where rejoin_price_lines didn't
    fully resolve, so the LLM mis-attributes prices across adjacent items.
    Conservative — only fires when:
      - items_sum is off both subtotal and total by > 2 yen
      - The OCR shows a clear desc → price chain (no other Japanese line between)
      - The OCR-grounded price differs from the LLM total by > 1 yen
      - Applying the fix brings items_sum strictly closer to a target
    """
    if not items:
        return
    items_sum = sum(i.get("total", 0) for i in items if isinstance(i, dict))
    targets = [t for t in (canonical_subtotal, target_subtotal, target_total) if t]
    if not targets:
        return


    lines = unified_text.split('\n')

    def _ocr_price_after(li: int) -> tuple[float | None, int | None]:
        # Look for a clean ¥-bearing or plain numeric line within next 6 lines.
        # Stop on another item-like line (Japanese text, no ¥).
        for j in range(li + 1, min(li + 7, len(lines))):
            s = lines[j].strip()
            if not s:
                continue
            if _SKIP_PRICE_LINE.search(s):
                return None, None
            m = re.match(r'^[¥￥]?\s*([\d,]+)\s*[※\*除]?\s*$', s)
            if m:
                try:
                    return float(m.group(1).replace(',', '')), j
                except ValueError:
                    return None, None
            if re.search(r'[ぁ-んァ-ン一-龥]{2,}', s):
                return None, None  # next item starts before any price
        return None, None

    def _ocr_price_inline(line: str) -> float | None:
        # ¥-prefixed first
        m = re.search(r'[¥￥]\s*([\d,]+)', line)
        if m:
            try:
                return float(m.group(1).replace(',', ''))
            except ValueError:
                return None
        # Trailing bare-digit price with tax marker (e.g., "...  640X" or
        # "... 228*"). Only the LAST trailing digit + marker on the line —
        # mid-line digits may be part of the description (e.g., "TV1.0テイシボ").
        m = re.search(r'\s+([\d,]{2,7})\s*[※\*X除軽]\s*$', line)
        if m:
            try:
                return float(m.group(1).replace(',', ''))
            except ValueError:
                return None
        return None

    def _ocr_window_contains_price(li: int, price: float) -> bool:
        if price is None:
            return False
        for j in range(li, min(li + 7, len(lines))):
            s = lines[j].strip()
            if not s:
                continue
            inline = _ocr_price_inline(s)
            if inline is not None and abs(inline - price) <= 1:
                return True
            m = re.match(r'^[¥￥]?\s*([\d,]+)\s*[※\*除]?\s*$', s)
            if m:
                try:
                    if abs(float(m.group(1).replace(',', '')) - price) <= 1:
                        return True
                except ValueError:
                    pass
            if j > li and re.search(r'[ぁ-んァ-ン一-龥]{2,}', s):
                return False
        return False

    def _ocr_window_supports_qty(li: int, price_li: int | None, item: dict) -> bool:
        qty = item.get("qty") or 1
        unit = item.get("unit_price")
        try:
            qty_f = float(qty)
            unit_f = float(unit)
        except (TypeError, ValueError):
            return False
        if qty_f <= 1 or unit_f <= 0:
            return False
        end = price_li if price_li is not None else min(li + 6, len(lines) - 1)
        qty_re = re.compile(
            r'(\d+(?:\.\d+)?)\s*[個コ点]\s*[xX×Ⅹ]\s*(?:単|@)?\s*([\d,]+)'
            r'|(?:単|@)?\s*([\d,]+)\s*[xX×Ⅹ]\s*(\d+(?:\.\d+)?)\s*[個コ点]?'
        )
        for j in range(li + 1, min(end + 1, len(lines))):
            s = lines[j].strip()
            if not s:
                continue
            m = qty_re.search(s)
            if not m:
                continue
            if m.group(1):
                found_qty = float(m.group(1))
                found_unit = float(m.group(2).replace(',', ''))
            else:
                found_unit = float(m.group(3).replace(',', ''))
                found_qty = float(m.group(4))
            if abs(found_qty - qty_f) <= 0.01 and abs(found_unit - unit_f) <= 1:
                return True
        return False

    for item in items:
        if not isinstance(item, dict):
            continue
        desc = (item.get("description") or "").strip()
        if not desc or len(desc) < 5:
            continue
        try:
            qty = float(item.get("qty") or 1)
            unit = float(item.get("unit_price") or 0)
            total = float(item.get("total") or 0)
        except (TypeError, ValueError):
            continue
        if qty <= 1 or unit <= 0 or total <= 0:
            continue
        if abs(total - (qty * unit)) <= 1:
            continue
        desc_prefix = desc[:5]
        for li, line in enumerate(lines):
            if desc_prefix not in line:
                continue
            ocr_total = _ocr_price_inline(line)
            price_li = li if ocr_total is not None else None
            if ocr_total is None:
                ocr_total, price_li = _ocr_price_after(li)
            if ocr_total is None or abs(ocr_total - total) > 1:
                continue
            if _ocr_window_supports_qty(li, price_li, item):
                continue
            item["qty"] = 1.0
            item["unit_price"] = total
            break

    # Apply candidate fixes one at a time, verifying each improves items_sum
    # toward a target. Stop when items_sum is within 2 yen of a target.
    progress = True
    while progress:
        progress = False
        items_sum = sum(i.get("total", 0) for i in items if isinstance(i, dict))
        if any(abs(items_sum - t) <= 2 for t in targets):
            break
        candidates: list[tuple[float, int, float, int, int | None]] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            desc = (item.get("description") or "").strip()
            total = item.get("total")
            if not desc or len(desc) < 5 or total is None:
                continue
            desc_prefix = desc[:5]
            matching_lines = [
                li for li, line in enumerate(lines)
                if desc_prefix in line
            ]
            if not matching_lines:
                continue
            if any(_ocr_window_contains_price(li, float(total)) for li in matching_lines):
                continue  # original total is OCR-supported; do not chase neighbors
            for li in matching_lines:
                line = lines[li]
                if _ocr_window_contains_price(li, float(total)):
                    break  # original total is OCR-supported; do not chase neighbors
                ocr_total = _ocr_price_inline(line)
                price_li = li if ocr_total is not None else None
                if ocr_total is None:
                    ocr_total, price_li = _ocr_price_after(li)
                if ocr_total is None:
                    continue
                if abs(ocr_total - total) <= 1:
                    break  # already aligned
                # Score this candidate by the improvement it brings
                new_sum = items_sum - total + ocr_total
                cur_diff = min(abs(items_sum - t) for t in targets)
                new_diff = min(abs(new_sum - t) for t in targets)
                if new_diff < cur_diff:
                    candidates.append((cur_diff - new_diff, idx, ocr_total, li, price_li))
                break  # first matching OCR line for this item
        if not candidates:
            break
        candidates.sort(reverse=True)  # largest improvement first
        improvement, idx, new_total, match_li, price_li = candidates[0]
        item = items[idx]
        item["total"] = new_total
        try:
            qty = float(item.get("qty") or 1)
            unit = float(item.get("unit_price") or 0)
        except (TypeError, ValueError):
            qty = 1
            unit = 0
        if qty == 1 and item.get("unit_price") is not None:
            item["unit_price"] = new_total
        elif (
            qty > 1
            and unit > 0
            and abs(new_total - (qty * unit)) > 1
            and not _ocr_window_supports_qty(match_li, price_li, item)
        ):
            item["qty"] = 1.0
            item["unit_price"] = new_total
        progress = True


def _repair_column_split_items(items, unified_text, target_subtotal, target_total):
    """Re-pair LLM items to OCR prices when the OCR is column-split.

    Column-split layout: a run of name-only lines (Japanese, no ¥), then a
    run of price-only lines (¥-prefixed or bare digits). The LLM matches by
    proximity, which fails when sub-runs are unequal or qty annotations
    break the price block.

    Strategy:
      1. Walk OCR up to (and optionally past) the 小計/合計 zone end. Skip
         qty notations, discount lines, and inline-priced names (those
         self-pair). Collect remaining name and price tokens in OCR order.
      2. If global counts of names == prices in the chain, position-pair
         them: name[i] → price[i] for the chain's full length.
      3. For each LLM item, if its description prefix appears in the paired
         dict and the override moves items_sum toward a target, apply.

    Inline-priced detection: a Japanese line ending with " <digits>[marker]"
    is treated as inline-priced even without a ¥ symbol (handles AEON-style
    "食品ポリ袋L (バイオマス30 3除" and "千切りキャベツビッグパ 238").

    Zone extension: when names without paired prices remain at the 小計
    boundary, extend past 小計 to capture stray bare-digit prices that
    appear before the first ¥-prefixed totals line. (Handles AEON layouts
    where the right-column item prices land below 小計 in OCR order.)

    Conservative — only fires when items_sum is off-target by > 2 yen and
    the override strictly reduces the error.
    """
    if not items:
        return
    items_sum = sum(i.get("total", 0) for i in items if isinstance(i, dict))
    targets = [t for t in (target_subtotal, target_total) if t]
    if not targets or any(abs(items_sum - t) <= 2 for t in targets):
        return

    lines = unified_text.split('\n')

    # Find zone-end at the first totals/tax line.
    end_idx = len(lines)
    for i, raw in enumerate(lines):
        s = raw.strip()
        if re.search(r'^(小\s*計|合\s*計|外税|内税|消費税|お預り|現計|お釣り|釣銭|総額)', s):
            end_idx = i
            break

    # Anchor the item zone to where LLM items actually appear in OCR. This
    # filters out header NAMEs (campaign text, store info, register #, etc.)
    # whose presence would break the equal-count check.
    item_descs = [
        (it.get("description") or "").strip()
        for it in items if isinstance(it, dict)
    ]
    item_descs = [d for d in item_descs if d and len(d) >= 2]
    if not item_descs:
        return

    zone_start: int | None = None

    for li in range(min(end_idx, len(lines))):
        line = lines[li]
        for d in item_descs:
            if d[:5] in line or (len(d) >= 3 and d[:3] in line and re.search(r'[ぁ-んァ-ン一-龥]', d[:3])):
                if zone_start is None:
                    zone_start = li

                break

    if zone_start is None:
        return
    # Items zone runs from zone_start (first OCR match of any LLM item) to
    # end_idx (the 小計/合計 line). No slack — the first matched line is
    # the earliest item, anything before it is header noise.
    item_zone_end = end_idx

    # Permissive price-line: digits + optional 1-2 trailing marker chars.
    # Captures `198`, `378+`, `98%`, `265X`, `228*`, `78 A`, `¥1,498`, `1,074`.
    # Rejects post-item footer noise like `10P)` or `54P` (P is not a marker).
    _PRICE_MARKER_CLASS = r'[*※軽除＊・X+%A_]'
    _PRICE_ONLY_RE = re.compile(
        r'^[¥￥]?\s*(\d[\d,]{0,5})\s*' + _PRICE_MARKER_CLASS + r'?\s*' + _PRICE_MARKER_CLASS + r'?\s*$'
    )
    _PRICE_HAS_MARKER_RE = re.compile(r'[*※軽除＊・X+%A]')
    _QTY_NOTATION_RE = re.compile(
        r'[\(（<]?\s*\d+\s*[コ個点]\s*[xX×]\s*(?:単|@)?\s*\d[\d,]*\s*[\)）>]?'
    )
    # Inline price tail: " <digits>[marker]" at end of a name line.
    _INLINE_PRICE_TAIL_RE = re.compile(
        r'\s+(\d[\d,]{0,5})\s*[*※軽除＊・X+%]?\s*$'
    )
    # Lines that look like a date or phone — not inline-priced.
    _DATE_LIKE_RE = re.compile(r'\d{4}[/年-]\d|\d{2}[:時]\d')
    _PHONE_LIKE_RE = re.compile(r'\d{2,4}-\d{2,4}-\d{3,4}')

    def _parse_price(s: str):
        m = _PRICE_ONLY_RE.match(s)
        if not m:
            return None
        try:
            v = float(m.group(1).replace(',', ''))
        except ValueError:
            return None
        if v < 1 or v > 999999:
            return None
        return v

    def _is_inline_priced(s: str) -> bool:
        if not re.search(r'[ぁ-んァ-ン一-龥]', s):
            return False
        if _DATE_LIKE_RE.search(s) or _PHONE_LIKE_RE.search(s):
            return False
        if re.search(r'[¥￥]\s*\d', s):
            return True
        m = _INLINE_PRICE_TAIL_RE.search(s)
        if not m:
            return False
        # Avoid product codes like "L30" mid-string by requiring the digit
        # group be at the end with whitespace before. _INLINE_PRICE_TAIL_RE
        # already enforces \s+ before the digit group, so a trailing lone
        # number after Japanese chars is the signal.
        return True

    def _is_name_line(s: str) -> bool:
        if not s or len(s) < 2:
            return False
        if re.search(r'[¥￥]', s):
            return False
        if not re.search(r'[ぁ-んァ-ン一-龥]', s):
            return False
        if _QTY_NOTATION_RE.search(s):
            return False
        if re.match(r'^割引', s) or s.startswith('-'):
            return False
        # Skip post-item footer markers (point-tracking, account info, etc.)
        # that share OCR space with the item zone.
        if re.search(
            r'(ポイント|残高|累計|獲得|有効期限|内訳|お買上|今回|商品数|'
            r'WAON|^内\s|^取\s*\d|レジ\s*\d|登録番号|TEL|FAX|http)', s
        ):
            return False
        return True

    # Walk the zone, building name and price token streams. Skip qty
    # notations, discount lines, and inline-priced names (self-paired).
    # Discount-rate lines like "20%" or "-18" stand alone; "98%" is a
    # price-with-marker (the % is OCR noise for *), not a discount rate.
    # Distinguish: discount lines appear right after a 割引 or item name
    # without an intervening price; we treat any standalone digit + %
    # within ~2 lines of a 割引 marker as a discount rate, otherwise as
    # a price-with-marker.
    discount_rate_lines: set[int] = set()
    for i in range(zone_start, item_zone_end):
        s = lines[i].strip()
        if re.match(r'^割引', s):
            # Look ahead for a digit% line within the next 3 lines.
            for j in range(i + 1, min(i + 4, item_zone_end)):
                t = lines[j].strip()
                if re.match(r'^-?\d{1,3}\s*[%％]\s*$', t):
                    discount_rate_lines.add(j)
                    break

    # Walk the zone, building OCR-ordered (name, price) pairs.
    # Inline-priced lines emit a pair directly. Pure-name and pure-price
    # tokens are stitched into chains; chains where len(names)==len(prices)
    # contribute pairs by position.
    ordered_pairs: list[tuple[str, float]] = []
    pending_names: list[str] = []
    pending_prices: list[float] = []

    def _flush_chain():
        nonlocal pending_names, pending_prices
        if pending_names and pending_prices and len(pending_names) == len(pending_prices):
            for n, p in zip(pending_names, pending_prices):
                ordered_pairs.append((n, p))
        pending_names = []
        pending_prices = []

    def _flush_chain_with_extension(extension_prices: list[float]):
        """Try to complete an unfinished chain by appending extension prices."""
        nonlocal pending_names, pending_prices
        needed = len(pending_names) - len(pending_prices)
        if needed > 0 and len(extension_prices) >= needed:
            pending_prices.extend(extension_prices[:needed])
        if pending_names and pending_prices and len(pending_names) == len(pending_prices):
            for n, p in zip(pending_names, pending_prices):
                ordered_pairs.append((n, p))
        pending_names = []
        pending_prices = []

    # Helper: detect a partial qty notation fragment like "(2個 X" (OCR
    # split this onto two lines so the trailing digits are missing).
    _PARTIAL_QTY_RE = re.compile(r'^[\(（<]?\s*\d+\s*[コ個点]\s*[xX×]?\s*$')

    for i in range(zone_start, item_zone_end):
        s = lines[i].strip()
        if not s:
            continue
        if _QTY_NOTATION_RE.search(s) or _PARTIAL_QTY_RE.match(s):
            continue
        if re.match(r'^割引', s) or i in discount_rate_lines or re.match(r'^-\d', s):
            continue
        # Inline-priced line — emit pair directly, flush any pending chain.
        if _is_inline_priced(s):
            # Extract name part and price.
            m_yen = re.search(r'[¥￥]\s*([\d,]+)', s)
            if m_yen:
                pv_str = m_yen.group(1)
                price_pos = m_yen.start()
            else:
                m_tail = _INLINE_PRICE_TAIL_RE.search(s)
                if not m_tail:
                    continue
                pv_str = m_tail.group(1)
                price_pos = m_tail.start()
            try:
                pv = float(pv_str.replace(',', ''))
            except ValueError:
                continue
            name_part = s[:price_pos].strip()
            if not name_part or not re.search(r'[ぁ-んァ-ン一-龥]', name_part):
                continue
            _flush_chain()
            ordered_pairs.append((name_part, pv))
            continue
        v = _parse_price(s)
        if v is not None:
            pending_prices.append(v)
            continue
        if _is_name_line(s):
            # Names after prices indicate a new chain — flush.
            if pending_prices and len(pending_names) == len(pending_prices):
                _flush_chain()
            pending_names.append(s)

    # Zone extension: scan past 小計 for stray bare-digit prices (no ¥) that
    # may complete an unfinished column-split chain. AEON layouts often print
    # the right-column item prices below 小計 in OCR order.
    extension_prices: list[float] = []
    if pending_names and len(pending_prices) < len(pending_names) and item_zone_end < len(lines):
        for i in range(item_zone_end + 1, len(lines)):
            s = lines[i].strip()
            if not s:
                continue
            if re.search(r'[¥￥]', s):
                break  # Hit ¥-prefixed totals zone
            if re.search(
                r'^(外税|内税|消費税|対象|お預り|現計|お釣り|釣銭|総額|'
                r'合\s*計|小\s*計|WAON|現金|クレジット|カード|お会計|電子)',
                s,
            ):
                break
            if _QTY_NOTATION_RE.search(s) or _PARTIAL_QTY_RE.match(s):
                continue
            v = _parse_price(s)
            if v is not None:
                extension_prices.append(v)
                # Stop if we have enough to complete the chain.
                if len(pending_prices) + len(extension_prices) >= len(pending_names):
                    break

    if extension_prices:
        _flush_chain_with_extension(extension_prices)
    else:
        _flush_chain()

    if len(ordered_pairs) < 2:
        return

    # Match LLM items to ordered_pairs by description-prefix overlap, then
    # greedy-claim by best score. Duplicate-named items (e.g., two
    # 牛豚ミンチ(解凍) lines) and out-of-order LLM emissions still match.
    def _match_score(ocr_name: str, llm_desc: str) -> int:
        clean = re.sub(r'^[\d\s\*\(（]+', '', ocr_name).strip()
        for prefix_len in (6, 5, 4, 3):
            if len(clean) >= prefix_len and clean[:prefix_len] in llm_desc[:14]:
                return prefix_len
            if len(ocr_name) >= prefix_len and ocr_name[:prefix_len] in llm_desc[:14]:
                return prefix_len
        return 0

    eligible_items: list[tuple[int, dict, str, float, int]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        desc = (item.get("description") or "").strip()
        total = item.get("total")
        if not desc or total is None:
            continue
        if (item.get("discount") or 0) > 0:
            continue
        qty = item.get("qty", 1) or 1
        eligible_items.append((idx, item, desc, float(total), int(qty)))

    # Score every (item, ocr_pair) combination, then greedy-claim.
    candidates: list[tuple[int, int, int]] = []  # (score, item_idx_in_eligible, ocr_pair_idx)
    for ei, (idx, _, desc, _, _) in enumerate(eligible_items):
        for p in range(len(ordered_pairs)):
            score = _match_score(ordered_pairs[p][0], desc)
            if score >= 3:
                candidates.append((score, ei, p))
    # Sort by score descending; tie-break by ei (earliest LLM item first).
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
    claimed_pair: set[int] = set()
    claimed_item: set[int] = set()
    matches: dict[int, int] = {}  # eligible idx -> ocr_pair idx
    for score, ei, p in candidates:
        if ei in claimed_item or p in claimed_pair:
            continue
        matches[ei] = p
        claimed_item.add(ei)
        claimed_pair.add(p)

    overrides: list[tuple[int, float, float, int]] = []
    for ei, p in matches.items():
        idx, _, _, total, qty = eligible_items[ei]
        ocr_price = ordered_pairs[p][1]
        if abs(ocr_price - total) < 1:
            continue
        overrides.append((idx, float(ocr_price), float(total), int(qty)))

    if not overrides:
        return

    # Apply overrides only if the collective effect strictly improves
    # items_sum's distance to a target. This catches "swap" scenarios
    # (two items with reversed totals) where a single greedy fix would
    # regress items_sum, but applying both is neutral or beneficial.
    items_sum = sum(i.get("total", 0) for i in items if isinstance(i, dict))
    total_delta = sum(new_total - old_total for _, new_total, old_total, _ in overrides)
    new_sum = items_sum + total_delta
    cur_diff = min(abs(items_sum - t) for t in targets)
    new_diff = min(abs(new_sum - t) for t in targets)
    if new_diff > cur_diff:
        return  # Net regression — don't apply
    if new_diff == cur_diff and total_delta == 0:
        # Pure swap (no items_sum change). Apply only if it actually
        # changes the description-total pairing for ≥ 2 items (otherwise
        # nothing happens).
        if len(overrides) < 2:
            return
    elif new_diff == cur_diff and total_delta != 0:
        return  # Same gap but in the other direction — don't apply

    for idx, new_total, _, qty in overrides:
        items[idx]["total"] = new_total
        if qty == 1 and items[idx].get("unit_price") is not None:
            items[idx]["unit_price"] = new_total
        elif qty > 1 and qty != 0 and new_total % qty == 0:
            items[idx]["unit_price"] = new_total / qty


def _replace_hallucinated_dup_with_ocr_item(items, unified_text, target_subtotal, target_total):
    """When LLM has duplicate items AND items_sum is off-target, look for an
    OCR-grounded item whose substitution closes the gap.

    Generic: handles any LLM hallucination where it copy-pastes a nearby
    item's price+description onto a different item, masking the right
    one. Only applies when:
      - items_sum doesn't match subtotal or total (within ±2 yen)
      - LLM has ≥ 2 items with the same (description, total)
      - Exactly one unaccounted OCR ¥amount equals dup_total + gap
    """
    if not items or len(items) < 2:
        return
    items_sum = sum(i.get("total", 0) for i in items if isinstance(i, dict))
    targets = [t for t in (target_subtotal, target_total) if t]
    if not targets:
        return
    if any(abs(items_sum - t) <= 2 for t in targets):
        return  # items already balance

    groups: dict[tuple[str, float], list[int]] = {}
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        desc = (it.get("description") or "").strip()
        total = it.get("total")
        if not desc or total is None:
            continue
        groups.setdefault((desc, float(total)), []).append(i)
    duplicates = {k: v for k, v in groups.items() if len(v) >= 2}
    if not duplicates:
        return

    lines = unified_text.split('\n')
    # Bound to the item zone: stop at the first 小計/合計 line so we don't
    # treat tax/total values as item-price candidates.
    zone_end = len(lines)
    for li, line in enumerate(lines):
        if re.search(r'^(小\s*計|合\s*計|外税|内税|消費税|お預り|現計|お釣り|釣銭)',
                     line.strip()):
            zone_end = li
            break
    ocr_prices: list[tuple[int, float]] = []
    _BARE_PRICE_RE = re.compile(r'^[¥￥]?\s*(\d[\d,]{0,5})\s*[*※軽除＊・X+%A]?\s*$')
    for li in range(zone_end):
        line = lines[li]
        if _SKIP_PRICE_LINE.search(line):
            continue
        # ¥-prefixed amounts (anywhere in the line)
        for m in re.finditer(r'[¥￥]\s*([\d,]+)', line):
            try:
                amt = float(m.group(1).replace(',', ''))
            except ValueError:
                continue
            if amt > 0:
                ocr_prices.append((li, amt))
        # Bare-digit price lines (no ¥), with optional trailing marker.
        # OCR sometimes drops the ¥ but the line is still a price token.
        if not re.search(r'[¥￥]', line):
            s = line.strip()
            if s and not re.search(r'[ぁ-んァ-ン一-龥]', s):
                m = _BARE_PRICE_RE.match(s)
                if m:
                    try:
                        amt = float(m.group(1).replace(',', ''))
                    except ValueError:
                        amt = 0
                    if amt > 0:
                        ocr_prices.append((li, amt))

    # Multiset diff: remove one OCR entry per LLM item amount
    item_amounts = [i.get("total", 0) for i in items if isinstance(i, dict)]
    unmatched = list(ocr_prices)
    for amt in item_amounts:
        for j, (_, oa) in enumerate(unmatched):
            if abs(oa - amt) < 1:
                unmatched.pop(j)
                break

    if not unmatched:
        return

    # For each duplicate × target combo, search for an OCR price that closes
    # the gap when substituted.
    candidates: list[tuple[float, int, int, float, str]] = []
    for target in targets:
        gap = target - items_sum
        for dup_key, dup_idxs in duplicates.items():
            dup_total = dup_key[1]
            wanted = dup_total + gap
            matches = [(li, oa) for li, oa in unmatched if abs(oa - wanted) <= 2]
            if len(matches) != 1:
                continue
            li, oa = matches[0]
            new_sum = items_sum - dup_total + oa
            diff = abs(new_sum - target)
            candidates.append((diff, dup_idxs[-1], li, oa, dup_key[0]))

    if not candidates:
        return
    candidates.sort()
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        return  # ambiguous tie — refuse

    diff, replace_idx, price_line_idx, new_total, _ = candidates[0]
    if diff > 2:
        return

    new_desc = _find_ocr_item_desc(lines, price_line_idx, items)
    if not new_desc:
        return

    items[replace_idx]["description"] = new_desc
    items[replace_idx]["total"] = new_total
    items[replace_idx]["unit_price"] = new_total
    items[replace_idx]["qty"] = 1


def _parse_qty_detail_total(line: str) -> tuple[float, float] | None:
    """Return (qty, unit_price) from OCR qty detail like "2個 X70)"."""
    m = re.search(r'(\d+)\s*[コ個点]\s*[xX×Ⅹ]\s*(?:単|@)?\s*(\d[\d,]*)', line)
    if not m:
        m = re.search(r'(?:単|@)\s*(\d[\d,]*)\s*[xX×Ⅹ]\s*(\d+)\s*[コ個点]', line)
        if not m:
            return None
        unit = float(m.group(1).replace(',', ''))
        qty = float(m.group(2))
    else:
        qty = float(m.group(1))
        unit = float(m.group(2).replace(',', ''))
    if qty < 2 or unit <= 0:
        return None
    return qty, unit


def _project_totals_to_ocr_multiset(extracted, unified_text):
    """When LLM items_sum is off-target but the OCR's price-column multiset
    sums to a target, snap the LLM's totals onto the OCR multiset.

    Triggered only when:
      - items_sum doesn't match subtotal or total (within ±2 yen)
      - count of OCR price tokens (after reserving qty>1 unit_prices) equals
        the count of qty=1 items, OR exactly one extra candidate exists
        and dropping it produces the unique target-matching subset
      - the resulting OCR multiset sums to a target (subtotal-qtyN_total or
        total-qtyN_total)

    The new totals first try to preserve OCR row order by matching item
    descriptions back to OCR item lines. If that is not reliable, fall back to
    total-rank projection.
    """
    items = extracted.get("line_items") or []
    if not items:
        return
    items_sum = sum(i.get("total", 0) for i in items if isinstance(i, dict))
    subtotal = extracted.get("subtotal")
    total = extracted.get("total")
    canonical_subtotal = _canonical_subtotal_from_taxes(extracted)
    targets = [t for t in (canonical_subtotal, subtotal, total) if t]
    if not targets:
        return
    items_sum_already_matches = any(abs(items_sum - t) <= 2 for t in targets)

    lines = unified_text.split('\n')

    # Find item zone: from first inline-priced line to first 小計/合計-style end marker.
    zone_start = None
    for i, line in enumerate(lines):
        s = line.strip()
        if re.search(r'[¥￥]\s*\d', s) or re.search(r'\d[\d,]*\s*[*※除軽]\s*$', s):
            zone_start = max(0, i - 1)
            break
    if zone_start is None:
        return
    zone_end = len(lines)
    for i in range(zone_start, len(lines)):
        if _OCR_ZONE_END_RE.match(lines[i].strip()):
            zone_end = i
            break

    # Extract candidate price tokens. Each candidate is (line_idx, value).
    candidates: list[tuple[int, int]] = []
    for li in range(zone_start, zone_end):
        s = lines[li].strip()
        if not s:
            continue
        if _OCR_QTY_NOTATION_RE.search(s):
            continue  # qty notation like "2個 X70)" — skip whole line
        m = _OCR_TRAILING_PRICE_RE.search(s)
        if not m:
            continue
        raw = m.group(1).strip().lstrip('¥￥').replace(',', '')
        if not raw or not raw.isdigit():
            continue
        try:
            v = int(raw)
        except ValueError:
            continue
        token = m.group(0)
        if v < 10 and not re.search(r'[*※除軽]', token):
            continue
        if v < 1 or v > 99999:
            continue
        qty_detail = None
        for lookahead in range(li + 1, min(li + 3, zone_end)):
            lookahead_s = lines[lookahead].strip()
            qty_detail = _parse_qty_detail_total(lookahead_s)
            if qty_detail:
                break
            if _OCR_TRAILING_PRICE_RE.search(lookahead_s):
                break
            if re.search(r'[ぁ-んァ-ン一-龥]{2,}', lookahead_s):
                break
        if qty_detail:
            qty, unit = qty_detail
            candidates.append((li, int(qty * unit)))
            continue
        candidates.append((li, v))

    if not candidates:
        return

    # Reserve OCR tokens consumed by items we do not project. For qty>1 rows,
    # OCR commonly prints the unit price near the quantity notation. For
    # discounted rows, OCR commonly prints the gross price followed by a
    # discount line, while the canonical item total is net.
    qty_n_items = [i for i in items if isinstance(i, dict) and (i.get("qty") or 1) > 1]
    discounted_items = [
        i for i in items
        if isinstance(i, dict) and (i.get("discount") or 0) > 0
    ]
    qty_1_items = [
        i for i in items
        if (
            isinstance(i, dict)
            and (i.get("qty") or 1) == 1
            and (i.get("discount") or 0) == 0
        )
    ]
    if not qty_1_items:
        return  # nothing to project onto

    pool = list(candidates)
    for it in qty_n_items:
        up = it.get("unit_price")
        total_val = it.get("total")
        reserved = False
        if up is not None:
            for j, (_, v) in enumerate(pool):
                if abs(v - up) < 1:
                    pool.pop(j)
                    reserved = True
                    break
        if reserved or total_val is None:
            continue
        for j, (_, v) in enumerate(pool):
            if abs(v - total_val) < 1:
                pool.pop(j)
                break
    for it in discounted_items:
        gross = None
        if it.get("unit_price") is not None and it.get("qty"):
            gross = float(it.get("unit_price") or 0) * float(it.get("qty") or 1)
        elif it.get("total") is not None:
            gross = float(it.get("total") or 0) + float(it.get("discount") or 0)
        if not gross:
            continue
        for j, (_, v) in enumerate(pool):
            if abs(v - gross) < 1:
                pool.pop(j)
                break

    n_qty1 = len(qty_1_items)
    fixed_total = sum(
        i.get("total", 0)
        for i in (qty_n_items + discounted_items)
        if isinstance(i, dict)
    )

    # Find the single subset (size = n_qty1) whose sum is within 2 of any target.
    target_qty1_sums = [t - fixed_total for t in targets]

    def _multiset_matches(values: list[int]) -> int | None:
        s = sum(values)
        for t in target_qty1_sums:
            if abs(s - t) <= 2:
                return t
        return None

    pool_values = [v for _, v in pool]
    chosen_pairs: list[tuple[int, int]] | None = None

    if len(pool_values) == n_qty1:
        if _multiset_matches(pool_values) is not None:
            chosen_pairs = list(pool)
    elif len(pool_values) == n_qty1 + 1:
        # Try dropping each candidate; apply only if exactly one drop produces
        # a sum that matches a target.
        viable: list[list[tuple[int, int]]] = []
        for k in range(len(pool)):
            sub_pairs = pool[:k] + pool[k + 1:]
            if _multiset_matches([v for _, v in sub_pairs]) is not None:
                viable.append(sub_pairs)
        # Multiple drops can produce equivalent sums when duplicate values
        # are present (dropping any of three "228"s gives the same subset).
        # Treat them as one viable solution.
        unique = {tuple(sorted(v for _, v in pairs)) for pairs in viable}
        if len(unique) == 1:
            chosen_pairs = list(viable[0])

    if chosen_pairs is None:
        return

    # Verify the projection actually changes the multiset (no point otherwise).
    sorted_qty1_totals = sorted(i.get("total", 0) for i in qty_1_items)
    sorted_chosen = sorted(v for _, v in chosen_pairs)

    # Sanity: same length
    if len(sorted_chosen) != len(qty_1_items):
        return
    if items_sum_already_matches and sorted_qty1_totals != sorted_chosen:
        return

    def _norm_desc(text: str) -> str:
        text = re.sub(r'^\d{4,}[A-Za-z0-9-]*\)?\s*', '', text or "")
        text = re.sub(r'[¥￥]?\s*\d[\d,]*\s*[*※除軽]?\s*$', '', text)
        text = re.sub(r'\s+', '', text)
        text = re.sub(r'[^\wぁ-んァ-ン一-龥]', '', text, flags=re.UNICODE)
        return text.lower()

    def _ocr_line_for_desc(desc: str) -> int | None:
        nd = _norm_desc(desc)
        if len(nd) < 3:
            return None
        best: tuple[float, int] | None = None
        for li in range(zone_start, zone_end):
            nl = _norm_desc(lines[li])
            if len(nl) < 3 or re.match(r'^\d+$', nl):
                continue
            if nd in nl or nl in nd:
                score = 1.0
            else:
                score = SequenceMatcher(None, nd, nl).ratio()
            if score >= 0.72 and (best is None or score > best[0]):
                best = (score, li)
        return best[1] if best else None

    # Prefer row-order projection when descriptions can be matched uniquely to
    # OCR item lines. This keeps description↔price pairing intact on receipts
    # that print several descriptions before their price column.
    desc_order: list[tuple[int, int]] = []
    used_lines: set[int] = set()
    for idx, item in enumerate(items):
        if not isinstance(item, dict) or (item.get("qty") or 1) != 1:
            continue
        line_idx = _ocr_line_for_desc(item.get("description") or "")
        if line_idx is None or line_idx in used_lines:
            desc_order = []
            break
        used_lines.add(line_idx)
        desc_order.append((line_idx, idx))

    if len(desc_order) == len(qty_1_items):
        for (_, idx), (_, new_total) in zip(
            sorted(desc_order),
            sorted(chosen_pairs, key=lambda p: p[0]),
        ):
            items[idx]["total"] = new_total
            items[idx]["unit_price"] = new_total
        return

    qty1_current_idxs = [
        idx for idx, item in enumerate(items)
        if (
            isinstance(item, dict)
            and (item.get("qty") or 1) == 1
            and (item.get("discount") or 0) == 0
        )
    ]
    if not qty_n_items and len(qty1_current_idxs) == len(chosen_pairs):
        for idx, (_, new_total) in zip(
            qty1_current_idxs,
            sorted(chosen_pairs, key=lambda p: p[0]),
        ):
            items[idx]["total"] = new_total
            items[idx]["unit_price"] = new_total
        return

    if sorted_qty1_totals == sorted_chosen or items_sum_already_matches:
        return

    # Fallback: assign sorted-OCR totals to qty=1 items by their current total-rank.
    qty1_sorted_idxs = sorted(
        range(len(items)),
        key=lambda j: (
            -1 if not isinstance(items[j], dict) or (items[j].get("qty") or 1) > 1 else 0,
            items[j].get("total", 0) if isinstance(items[j], dict) else 0,
        ),
    )
    qty1_sorted_idxs = [j for j in qty1_sorted_idxs
                        if (
                            isinstance(items[j], dict)
                            and (items[j].get("qty") or 1) == 1
                            and (items[j].get("discount") or 0) == 0
                        )]

    for k, idx in enumerate(qty1_sorted_idxs):
        new_total = sorted_chosen[k]
        items[idx]["total"] = new_total
        items[idx]["unit_price"] = new_total


def _layout_block_height(block: dict) -> float:
    bbox = block.get("bbox") or []
    ys = [p[1] for p in bbox if isinstance(p, (list, tuple)) and len(p) >= 2]
    if ys:
        return float(max(ys) - min(ys))
    return 0.0


def _layout_block_center_y(block: dict) -> float:
    bbox = block.get("bbox") or []
    ys = [p[1] for p in bbox if isinstance(p, (list, tuple)) and len(p) >= 2]
    if ys:
        return (max(ys) + min(ys)) / 2
    return float(block.get("y") or 0)


def _group_layout_rows(layout_blocks: list[dict]) -> list[list[dict]]:
    blocks = [b for b in layout_blocks or [] if (b.get("text") or "").strip()]
    if not blocks:
        return []
    heights = sorted(h for h in (_layout_block_height(b) for b in blocks) if h > 0)
    median_h = heights[len(heights) // 2] if heights else 20.0
    y_tol = max(8.0, median_h * 0.55)

    rows: list[list[dict]] = []
    row_y: float | None = None
    current_page = None
    for block in sorted(blocks, key=lambda b: (b.get("page", 0), _layout_block_center_y(b), b.get("x") or 0)):
        cy = _layout_block_center_y(block)
        page = block.get("page", 0)
        if current_page != page:
            rows.append([block])
            row_y = cy
            current_page = page
        elif row_y is None or abs(cy - row_y) <= y_tol:
            if not rows:
                rows.append([])
            rows[-1].append(block)
            row_y = cy if row_y is None else (row_y + cy) / 2
        else:
            rows.append([block])
            row_y = cy
    return [sorted(row, key=lambda b: b.get("x") or 0) for row in rows]


def _layout_price_value(text: str, *, allow_small: bool = False) -> int | None:
    s = (text or "").strip()
    m = re.match(r'^[¥￥]?\s*(\d[\d,]*)\s*[*※除軽]?\s*$', s)
    if not m:
        return None
    try:
        value = int(m.group(1).replace(',', ''))
    except ValueError:
        return None
    min_value = 1 if allow_small else 10
    if value < min_value or value > 99999:
        return None
    return value


def _layout_qty_detail_total(row_text: str) -> int | None:
    compact = re.sub(r'\s+', '', row_text or '')
    m = re.search(r'(?<!\d)(\d{1,2})個[×xX]\s*(\d{1,5})', compact)
    if not m:
        return None
    try:
        qty = int(m.group(1))
        unit = int(m.group(2))
    except ValueError:
        return None
    if qty <= 1 or unit <= 0:
        return None
    total = qty * unit
    if total < 10 or total > 99999:
        return None
    return total


def _norm_layout_desc(text: str) -> str:
    text = re.sub(r'^\d{4,}[A-Za-z0-9-]*\)?\s*', '', text or "")
    text = re.sub(r'[¥￥]?\s*\d[\d,]*\s*[*※除軽]?\s*$', '', text)
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'[^\wぁ-んァ-ン一-龥]', '', text, flags=re.UNICODE)
    return text.lower()


def _layout_row_price_candidates(layout_blocks: list[dict] | None) -> list[dict]:
    rows = _group_layout_rows(layout_blocks or [])
    raw_rows: list[dict] = []
    for row_idx, row in enumerate(rows):
        row_text = "".join(str(b.get("text") or "") for b in row).strip()
        if not row_text:
            continue
        if _OCR_ZONE_END_RE.match(row_text):
            break
        price_positions = [
            (idx, _layout_price_value(str(block.get("text") or ""), allow_small=True))
            for idx, block in enumerate(row)
        ]
        price_positions = [(idx, value) for idx, value in price_positions if value is not None]
        if not price_positions:
            continue
        raw_rows.append({
            "row_idx": row_idx,
            "row": row,
            "price_positions": price_positions,
        })

    price_xs = [
        float(raw["row"][idx].get("x") or 0)
        for raw in raw_rows
        for idx, _value in raw["price_positions"]
        if float(raw["row"][idx].get("x") or 0) >= 180
    ]
    if not price_xs:
        price_xs = [
            float(raw["row"][idx].get("x") or 0)
            for raw in raw_rows
            for idx, _value in raw["price_positions"]
        ]
    if not price_xs:
        return []
    price_xs = sorted(price_xs)
    price_col_x = price_xs[len(price_xs) // 2]
    x_tol = max(45.0, price_col_x * 0.16)

    candidates: list[dict] = []
    for raw in raw_rows:
        row = raw["row"]
        near_column = [
            pair for pair in raw["price_positions"]
            if abs(float(row[pair[0]].get("x") or 0) - price_col_x) <= x_tol
        ]
        if not near_column:
            continue
        price_idx, value = max(
            near_column,
            key=lambda pair: float(row[pair[0]].get("x") or 0),
        )
        price_x = float(row[price_idx].get("x") or 0)
        desc_text = "".join(str(b.get("text") or "") for b in row[:price_idx]).strip()
        if not desc_text or _SKIP_PRICE_LINE.search(desc_text):
            continue
        if not re.search(r'[ぁ-んァ-ン一-龥]', desc_text):
            continue
        next_row_text = ""
        next_row_idx = raw["row_idx"] + 1
        if next_row_idx < len(rows):
            next_row_text = "".join(str(b.get("text") or "") for b in rows[next_row_idx])
        qty_detail_total = _layout_qty_detail_total(next_row_text)
        if qty_detail_total is not None:
            value = qty_detail_total
        candidates.append({
            "description": desc_text,
            "value": int(value),
            "y": _layout_block_center_y(row[price_idx]),
            "x": price_x,
        })
    return candidates


def _project_totals_to_layout_rows(extracted, ocr_layout_blocks):
    """Use preserved OCR row geometry to resolve price-token swaps.

    This is intentionally conservative and only fires when the geometric row
    prices form a subtotal/total-matching multiset while the current extraction
    does not.
    """
    items = extracted.get("line_items") or []
    if not items or not ocr_layout_blocks:
        return

    subtotal = extracted.get("subtotal")
    total = extracted.get("total")
    canonical_subtotal = _canonical_subtotal_from_taxes(extracted)
    targets = [t for t in (canonical_subtotal, subtotal, total) if t]
    if not targets:
        return

    item_sum = sum(i.get("total", 0) for i in items if isinstance(i, dict))
    if any(abs(item_sum - t) <= 2 for t in targets):
        return

    qty_n_items = [i for i in items if isinstance(i, dict) and (i.get("qty") or 1) > 1]
    discounted_items = [
        i for i in items
        if isinstance(i, dict) and (i.get("discount") or 0) > 0
    ]
    qty_1_indices = [
        idx for idx, item in enumerate(items)
        if (
            isinstance(item, dict)
            and (item.get("qty") or 1) == 1
            and (item.get("discount") or 0) == 0
        )
    ]
    if not qty_1_indices:
        return

    candidates = _layout_row_price_candidates(ocr_layout_blocks)
    if not candidates:
        return

    fixed_total = sum(
        i.get("total", 0)
        for i in (qty_n_items + discounted_items)
        if isinstance(i, dict)
    )
    target_qty1_sums = [t - fixed_total for t in targets]

    def _matches_target(values: list[int]) -> bool:
        s = sum(values)
        return any(abs(s - t) <= 2 for t in target_qty1_sums)

    chosen = None
    n_qty1 = len(qty_1_indices)
    values = [c["value"] for c in candidates]
    if len(values) == n_qty1 and _matches_target(values):
        chosen = list(candidates)
    elif len(values) == n_qty1 + 1:
        viable = []
        for drop_idx in range(len(candidates)):
            subset = candidates[:drop_idx] + candidates[drop_idx + 1:]
            if _matches_target([c["value"] for c in subset]):
                viable.append(subset)
        unique = {tuple(sorted(c["value"] for c in subset)) for subset in viable}
        if len(unique) == 1:
            chosen = viable[0]

    if chosen is None or len(chosen) != n_qty1:
        return

    assignments: dict[int, int] = {}
    used_candidate_idxs: set[int] = set()
    for item_idx in qty_1_indices:
        item_desc = _norm_layout_desc(items[item_idx].get("description") or "")
        if len(item_desc) < 3:
            assignments = {}
            break
        best: tuple[float, int] | None = None
        for cand_idx, cand in enumerate(chosen):
            if cand_idx in used_candidate_idxs:
                continue
            cand_desc = _norm_layout_desc(cand["description"])
            if item_desc in cand_desc or cand_desc in item_desc:
                score = 1.0
            else:
                score = SequenceMatcher(None, item_desc, cand_desc).ratio()
            if score >= 0.72 and (best is None or score > best[0]):
                best = (score, cand_idx)
        if best is None:
            assignments = {}
            break
        used_candidate_idxs.add(best[1])
        assignments[item_idx] = chosen[best[1]]["value"]

    if len(assignments) != n_qty1:
        return

    new_sum = sum(
        assignments.get(idx, item.get("total", 0))
        for idx, item in enumerate(items) if isinstance(item, dict)
    )
    if not any(abs(new_sum - t) <= 2 for t in targets):
        return

    for idx, value in assignments.items():
        items[idx]["total"] = value
        items[idx]["unit_price"] = value


def _find_ocr_item_desc(lines, price_line_idx, existing_items):
    """Find a plausible item description for an OCR price line."""
    existing_descs = {
        (it.get("description") or "").strip()
        for it in existing_items if isinstance(it, dict)
    }

    def _clean(text: str) -> str:
        text = text.strip()
        m = re.search(r'[¥￥]', text)
        if m:
            text = text[:m.start()].strip()
        text = _clean_ocr_price_line_desc(text)
        text = re.sub(r'\s+[\d,]+\s*[点個コ]\s*$', '', text).strip()
        text = re.sub(r'\s*[※\*非外内]\s*$', '', text).strip()
        mc = re.match(r'^\d{3,}[A-Za-z]{0,3}\)?\s?(.+)$', text)
        if mc and re.search(r'[ぁ-んァ-ン一-龥]', mc.group(1)):
            text = mc.group(1).strip()
        return text

    def _is_valid(text: str) -> bool:
        if not text or len(text) < 3:
            return False
        if text in _GENERIC_DESC_MARKERS:
            return False
        if _SKIP_PRICE_LINE.search(text):
            return False
        if re.search(r'取\s*\d|担当|レジ|領収|登録番号|TEL|FAX|http|:', text, re.IGNORECASE):
            return False
        if re.search(r'お願い|保管|場合|印字面|財布|手帳|ください', text):
            return False
        if re.match(r'^[\d,\s\-\(\)\.\*※軽除外]+$', text):
            return False
        if re.search(
            r'\d+(?:\.\d+)?\s*[個コ点]\s*[xX×Ⅹ]\s*(?:単|@)?\s*[\d,]+'
            r'|(?:単|@)?\s*[\d,]+\s*[xX×Ⅹ]\s*\d+(?:\.\d+)?\s*[個コ点]?',
            text,
        ):
            return False
        if not re.search(r'[ぁ-んァ-ン一-龥]', text):
            return False
        return True

    # Same-line first (rejoin merged item+price)
    cand = _clean(lines[price_line_idx])
    if _is_valid(cand) and cand not in existing_descs:
        return cand
    # Search backward up to 15 lines, then forward up to 5
    for j in list(range(price_line_idx - 1, max(price_line_idx - 16, -1), -1)) + \
             list(range(price_line_idx + 1, min(price_line_idx + 6, len(lines)))):
        cand = _clean(lines[j])
        if _is_valid(cand) and cand not in existing_descs:
            return cand
    return None


def _clean_ocr_price_line_desc(text: str) -> str:
    """Remove OCR row prefixes/suffix prices from a candidate item name."""
    text = text.strip()
    text = _OCR_TRAILING_PRICE_RE.sub("", text).strip()
    text = re.sub(r'(?:^|\s)[¥￥]?\s*\d[\d,]*\s+[A-ZＡ-Ｚ]\s*$', '', text).strip()
    text = re.sub(r'^\d{3,}[A-Za-z0-9-]*\)?\s*', '', text).strip()
    text = re.sub(r'\s*[※\*非外内]\s*$', '', text).strip()
    return text
