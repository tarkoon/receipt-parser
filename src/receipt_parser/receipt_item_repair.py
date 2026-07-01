"""Receipt OCR item repair helpers."""

import re
from difflib import SequenceMatcher

from .patterns import (
    _BANNER_PHRASE_RE,
    _FUEL_KEYWORDS,
    _GENERIC_DESC_MARKERS,
    _OCR_TRAILING_PRICE_RE,
    _SKIP_PRICE_LINE,
)
from .receipt_financial import (
    _find_subset_sum,
    _parse_amount_fragment,
)
from .receipt_projection import (
    _clean_ocr_price_line_desc,
    _norm_layout_desc,
    _parse_qty_detail_total,
)


def _clean_code_prefixed_item_descriptions(extracted):
    """Remove visible product-code prefixes from item descriptions."""
    for item in extracted.get("line_items") or []:
        if not isinstance(item, dict):
            continue
        desc = (item.get("description") or "").strip()
        if not desc:
            continue
        cleaned = _clean_ocr_price_line_desc(desc)
        cleaned = re.sub(r'\s+1$', '', cleaned).strip()
        if cleaned != desc and re.search(r'[ぁ-んァ-ン一-龥]', cleaned):
            item["description"] = cleaned


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
    if _SKIP_PRICE_LINE.search(text):
        return False
    if re.search(r'割引|値引', text):
        return False
    if re.match(r'^[\d,\s\-\(\)\.\*※軽除外]+$', text):
        return False
    return bool(re.search(r'[ぁ-んァ-ン一-龥]', text))


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


def _qty_detail_owner_indices(items, unified_text):
    """Return item indices whose OCR row owns a nearby qty/unit detail."""
    if not items or not unified_text:
        return set()
    lines = [line.strip() for line in unified_text.split('\n')]
    owners: set[int] = set()

    def _nearest_name_before(detail_idx: int) -> str | None:
        for idx in range(detail_idx - 1, max(detail_idx - 8, -1), -1):
            line = lines[idx].strip()
            if not line:
                continue
            if _parse_qty_detail_total(line):
                continue
            if _OCR_TRAILING_PRICE_RE.search(line):
                continue
            if re.fullmatch(r'\d{8,}\s*JAN', line, flags=re.IGNORECASE):
                continue
            desc = _clean_ocr_price_line_desc(line)
            if _valid_ocr_item_desc(desc):
                return desc
        return None

    def _has_expected_total_after(detail_idx: int, expected_total: float) -> bool:
        for idx in range(detail_idx + 1, min(len(lines), detail_idx + 4)):
            line = lines[idx].strip()
            if not line:
                continue
            if _parse_qty_detail_total(line):
                break
            amount = _parse_amount_fragment(line.lstrip('¥￥').replace(',', ''))
            if amount is not None and abs(amount - expected_total) <= 2:
                return True
            if _valid_ocr_item_desc(_clean_ocr_price_line_desc(line)):
                break
        return False

    for detail_idx, line in enumerate(lines):
        detail = _parse_qty_detail_total(line)
        if not detail:
            continue
        qty, unit = detail
        expected_total = qty * unit
        owner_desc = _nearest_name_before(detail_idx)
        if not owner_desc or not _has_expected_total_after(detail_idx, expected_total):
            continue
        owner_norm = _norm_layout_desc(owner_desc)
        best_idx = None
        best_key = None
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
            if score < 0.72:
                continue
            try:
                total = float(item.get("total") or 0)
                unit_price = float(item.get("unit_price") or 0)
            except (TypeError, ValueError):
                continue
            if abs(total - expected_total) > 2 and abs(unit_price - expected_total) > 2:
                continue
            line_idx = _ocr_line_index_for_item(lines, item)
            distance = abs((line_idx if line_idx is not None else detail_idx) - detail_idx)
            key = (score, -distance)
            if best_key is None or key > best_key:
                best_idx = item_idx
                best_key = key
        if best_idx is not None:
            owners.add(best_idx)
    return owners


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
    kept = []
    for item in items:
        if not isinstance(item, dict):
            kept.append(item)
            continue
        desc = (item.get("description") or "").strip()
        if desc and _BANNER_PHRASE_RE.search(desc):
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


def _fix_digit_misread_items(extracted, unified_text):
    """When items_sum is short by a small N, try OCR digit-misread corrections
    on items. A common scenario: OCR reads '108※' (108 yen, reduced rate) as
    '100%' (the 8 + ※ became %). The LLM extracts total=100; we need 108.

    Strategy: for items_sum gap N, look for items where:
      - item.total + N is a plausible OCR misread (single-digit confusion:
        0↔8, 0↔6, 1↔7, 6↔8, etc.)
      - the corrected total appears in OCR text as a plausible price
      - applying the correction moves items_sum exactly to subtotal/total

    Conservative — only fires when the corrected total is in OCR (somewhere),
    the gap matches exactly, and only one such correction is found.
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

    # Compute gap to each target; pick the smallest non-zero gap
    gaps = [(t - items_sum, t) for t in targets]
    valid_gaps = [(g, t) for g, t in gaps if 0 < g <= 50]
    if not valid_gaps:
        return
    gap = min(g for g, _ in valid_gaps)

    # Common OCR digit-confusion pairs (1-step perturbations)
    # We test if item.total + gap is plausibly the correct total by checking
    # if a single-digit replacement gets us there. Most useful is: the
    # LAST digit of total_corrected differs from total by ≤ 1 digit pair.
    def _single_digit_diff(a: int, b: int) -> bool:
        sa, sb = str(a), str(b)
        if len(sa) != len(sb):
            return False
        diffs = [(x, y) for x, y in zip(sa, sb) if x != y]
        return len(diffs) == 1

    candidates: list[tuple[int, float]] = []  # (item_idx, new_total)
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        t = item.get("total")
        if t is None or t <= 0:
            continue
        try:
            t_int = int(t)
        except (TypeError, ValueError):
            continue
        new_total = t_int + int(gap)
        if not _single_digit_diff(t_int, new_total):
            continue
        # Look for evidence in OCR: the corrected total may not appear
        # literally (it's an OCR misread!), but a "T%"-style line matching
        # the original OCR-misread pattern is a strong signal.
        # E.g., 100 → 108 with 0/8 confusion → look for 'T%' on its own line
        # which is common when '%' was misread of '8※' or similar.
        sa, sb = str(t_int), str(new_total)
        # If the differing digit changed to/from 0, 8, or 6 (common
        # confusions), pattern '<original>%' or '<original>除' as a standalone
        # line is suspicious — likely a misread.
        diff_pairs = [(x, y) for x, y in zip(sa, sb) if x != y]
        if not diff_pairs:
            continue
        old_d, new_d = diff_pairs[0]
        if (old_d, new_d) not in {('0', '8'), ('8', '0'), ('0', '6'),
                                  ('6', '0'), ('1', '7'), ('7', '1'),
                                  ('6', '8'), ('8', '6'), ('5', '6'),
                                  ('6', '5')}:
            continue
        # Match a standalone "<original>%" line in OCR (signature of
        # 8/0 misread where the trailing '8※' became '%').
        misread_pattern = re.compile(rf'^\s*{re.escape(sa)}%\s*$', re.MULTILINE)
        if not misread_pattern.search(unified_text):
            continue
        candidates.append((idx, float(new_total)))

    if len(candidates) != 1:
        return

    idx, new_total = candidates[0]
    items[idx]["total"] = new_total
    if items[idx].get("qty", 1) == 1:
        items[idx]["unit_price"] = new_total


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
            if not cand or len(cand) < 3:
                continue
            if cand in _GENERIC_DESC_MARKERS:
                continue
            if re.match(r'^[\d,\s\-\(\)\.\*※軽除外]+$', cand):
                continue
            if re.search(
                r'\d+(?:\.\d+)?\s*[個コ点]\s*[xX×Ⅹ]\s*(?:単|@)?\s*[\d,]+'
                r'|(?:単|@)?\s*[\d,]+\s*[xX×Ⅹ]\s*\d+(?:\.\d+)?\s*[個コ点]?',
                cand,
            ):
                continue
            if not re.search(r'[ぁ-んァ-ン一-龥]', cand):
                continue
            if _SKIP_PRICE_LINE.search(cand):
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


def _dedup_same_total_items(extracted):
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

    keep_mask = [True] * len(items)
    seen: dict[tuple, int] = {}
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        key = (item.get("description", ""), item.get("total", 0))
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
    # Pre-compute qty-detail lines (e.g., "(3個 X 単68)") and the implied
    # totals — we use these to validate the LLM's qty/unit_price extraction.
    # If a qty-detail line corresponds to the item AND its qty*unit matches
    # the LLM's qty*unit, the LLM is right and we should NOT "fix" it.
    qty_detail_re = re.compile(
        r'[\(\<]?\s*(\d+)\s*[個コ点]\s*[xX×]\s*(?:単|@)?\s*(\d+)\s*[\)\>]?'
    )
    qty_detail_pairs: list[tuple[int, int]] = []  # (qty, unit) pairs
    for line in unified_text.split('\n'):
        m = qty_detail_re.search(line.strip())
        if m:
            try:
                qty_detail_pairs.append((int(m.group(1)), int(m.group(2))))
            except ValueError:
                pass

    def _has_supporting_qty_detail(item_qty: int, item_unit: float) -> bool:
        """OCR has a qty-detail line confirming this item's qty AND unit_price."""
        return any(q == item_qty and abs(u - item_unit) < 1
                   for q, u in qty_detail_pairs)

    for item in items:
        if not isinstance(item, dict) or item.get("qty", 1) <= 1:
            continue
        total = item.get("total", 0)
        unit_price = item.get("unit_price")
        if unit_price is None:
            continue
        # Skip if qty-detail line confirms this qty × unit_price
        if _has_supporting_qty_detail(int(item.get("qty", 1)), float(unit_price)):
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
        # Skip if qty-detail line confirms
        if _has_supporting_qty_detail(int(item.get("qty", 1)), float(unit_price)):
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
    for item in items:
        if not isinstance(item, dict):
            continue
        desc = (item.get("description") or "").strip()
        if not desc or len(desc) < 4:
            continue
        cur_total = item.get("total", 0)
        desc_prefix = desc[:4]
        for li, line in enumerate(ocr_lines):
            if desc_prefix not in line:
                continue
            for offset in range(1, 5):
                if li + offset >= len(ocr_lines):
                    break
                nearby = ocr_lines[li + offset].strip()
                if not nearby:
                    continue
                m = qty_re.search(nearby)
                if m:
                    qty = float(m.group(1))
                    try:
                        unit = float(m.group(2).replace(',', ''))
                    except ValueError:
                        break
                    if qty >= 2 and unit > 0 and qty * unit > cur_total + 1:
                        # Skip discounted items — the LLM already merged the
                        # discount into total at the original (correct) qty.
                        # Only override when current item has no discount.
                        if not (item.get("discount") or 0):
                            item["qty"] = qty
                            item["unit_price"] = unit
                            item["total"] = qty * unit
                    break
                # Stop on next item desc (Japanese without qty notation)
                if re.search(r'[ぁ-んァ-ン一-龥]{2,}', nearby) and not re.search(r'\d+\s*[コ個点]', nearby):
                    break
            break


def _fix_qty_from_ocr_patterns(items, unified_text):
    """Fix quantities using ×N個 patterns and qty×price scanners in OCR text."""
    ocr_lines = unified_text.split('\n')

    # 本体合計(N点) — Starbucks-style summary that names the total item count.
    # When the receipt has exactly one line_item but the summary says N>1,
    # the LLM lost the qty (which usually appears as a "NT"/"3T" prefix on
    # the item line). Apply qty=N and divide unit_price.
    body_count_m = re.search(r'本体合計\s*\(?\s*(\d+)\s*点\s*\)?', unified_text)
    if body_count_m and len(items) == 1 and isinstance(items[0], dict):
        body_qty = int(body_count_m.group(1))
        item = items[0]
        cur_qty = item.get("qty", 1) or 1
        cur_total = item.get("total") or 0
        if body_qty > 1 and cur_qty == 1 and cur_total > 0 and cur_total % body_qty == 0:
            item["qty"] = float(body_qty)
            item["unit_price"] = cur_total / body_qty

    # Match by description prefix
    for item in items:
        if not isinstance(item, dict):
            continue
        unit_price = item.get("unit_price")
        desc = item.get("description", "")
        if unit_price is None or not desc:
            continue
        desc_prefix = desc[:4] if len(desc) >= 4 else desc
        price_str = str(int(unit_price)) if unit_price == int(unit_price) else str(unit_price)
        pattern_mult = r'(?:単|@)?' + re.escape(price_str) + r'\s*[×xX]\s*(\d+)\s*個?'
        pattern_ko = re.escape(price_str) + r'\s+(\d+)\s*個'
        for li, ocr_line in enumerate(ocr_lines):
            if desc_prefix not in ocr_line:
                continue
            for offset in range(0, 4):
                if li + offset >= len(ocr_lines):
                    break
                m = re.search(pattern_mult, ocr_lines[li + offset])
                if not m:
                    m = re.search(pattern_ko, ocr_lines[li + offset])
                if m:
                    correct_qty = float(m.group(1))
                    if correct_qty != item.get("qty", 1) and correct_qty > 1:
                        item["qty"] = correct_qty
                        item["total"] = unit_price * correct_qty - (item.get("discount") or 0)
                    break
            break

    # Garbled multiplication lines: try digit substrings validated against item total
    for item in items:
        if not isinstance(item, dict) or item.get("qty", 1) != 1:
            continue
        total = item.get("total", 0)
        if total <= 0:
            continue
        desc = item.get("description", "")
        desc_prefix = desc[:4] if len(desc) >= 4 else desc
        if not desc_prefix:
            continue
        for li, ocr_line in enumerate(ocr_lines):
            if desc_prefix not in ocr_line:
                continue
            for offset in range(1, 3):
                if li + offset >= len(ocr_lines):
                    break
                nearby = ocr_lines[li + offset].strip()
                if not re.search(r'[×xX]', nearby):
                    continue
                parts = re.split(r'\s*[×xX]\s*', nearby, maxsplit=1)
                if len(parts) != 2:
                    continue
                left_digits = re.findall(r'\d+', parts[0])
                right_digits = re.findall(r'\d+', parts[1])
                found = False
                for ld in left_digits:
                    for rd in right_digits:
                        q, p = int(ld), int(rd)
                        if 2 <= q <= 9 and p > 0 and q * p == total:
                            item["qty"] = float(q)
                            item["unit_price"] = float(p)
                            item["total"] = float(q * p)
                            found = True
                            break
                        if len(ld) > 1:
                            q2 = int(ld[0])
                            if 2 <= q2 <= 9 and q2 * p == total:
                                item["qty"] = float(q2)
                                item["unit_price"] = float(p)
                                item["total"] = float(q2 * p)
                                found = True
                                break
                        if len(rd) > 1:
                            p2 = int(rd[1:])
                            if p2 > 0 and q * p2 == total:
                                item["qty"] = float(q)
                                item["unit_price"] = float(p2)
                                item["total"] = float(q * p2)
                                found = True
                                break
                            if len(ld) > 1:
                                q2 = int(ld[0])
                                if 2 <= q2 <= 9 and p2 > 0 and q2 * p2 == total:
                                    item["qty"] = float(q2)
                                    item["unit_price"] = float(p2)
                                    item["total"] = float(q2 * p2)
                                    found = True
                                    break
                    if found:
                        break
                if found:
                    break
            break

    # Scan ALL OCR lines for qty×price patterns, match by total/price
    ocr_qty_prices: list[tuple[float, float, float]] = []
    for ocr_line in ocr_lines:
        found_qty_str, found_price_str = None, None
        m = re.search(r'(\d+)\s*[コ個]\s*[×xX]\s*(?:単|@)?\s*(\d[\d,]*)', ocr_line)
        if m:
            found_qty_str, found_price_str = m.group(1), m.group(2)
        if not found_qty_str:
            m2 = re.search(r'(?:単|@)\s*(\d[\d,]*)\s*[×xX]\s*(\d+)\s*[コ個]', ocr_line)
            if m2:
                found_price_str, found_qty_str = m2.group(1), m2.group(2)
        if not found_qty_str:
            m3 = re.search(r'[¥￥]\s*(\d[\d,]*)\s+(\d+)\s*個', ocr_line)
            if m3:
                found_price_str, found_qty_str = m3.group(1), m3.group(2)
        if found_qty_str and found_price_str:
            ocr_qty_prices.append((
                float(found_qty_str),
                float(found_price_str.replace(',', '')),
                float(found_qty_str) * float(found_price_str.replace(',', '')),
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
        ocr_qty_prices.append((qty, price, qty * price))

    for li, ocr_line in enumerate(ocr_lines):
        m_ten = re.match(r'^\s*(\d+)\s*点\s*$', ocr_line)
        if m_ten and li + 1 < len(ocr_lines):
            m_price = re.match(r'^\s*@\s*(\d[\d,]*)\s*$', ocr_lines[li + 1])
            if m_price:
                qty = float(m_ten.group(1))
                price = float(m_price.group(1).replace(',', ''))
                ocr_qty_prices.append((qty, price, qty * price))

    # Multi-line @PRICEx / QTY pattern (e.g., "@278x" then "3" on next line)
    # Apply directly to the nearest matching item by description proximity
    for li, ocr_line in enumerate(ocr_lines):
        m_at = re.match(r'^\s*[@＠](\d[\d,]*)\s*[×xX]?\s*$', ocr_line.strip())
        if m_at and li + 1 < len(ocr_lines):
            m_qty = re.match(r'^\s*(\d+)\s*$', ocr_lines[li + 1].strip())
            if m_qty:
                price = float(m_at.group(1).replace(',', ''))
                qty = float(m_qty.group(1))
                if qty > 1:
                    desc_context = None
                    for back in range(li - 1, max(li - 3, -1), -1):
                        bl = ocr_lines[back].strip()
                        if not bl:
                            continue
                        if re.match(r'^[\*¥￥@＠]?\s*\d[\d,]*\s*[*※×xX]?\s*$', bl):
                            continue
                        desc_context = bl
                        break
                    matched_item = None
                    if desc_context:
                        best_overlap = 0
                        for item in items:
                            if not isinstance(item, dict):
                                continue
                            desc = item.get("description", "")
                            item_total = float(item.get("total") or 0)
                            item_unit = float(item.get("unit_price") or 0)
                            if (
                                not desc
                                or (
                                    abs(item_unit - price) >= 1
                                    and abs(item_total - price * qty) >= 1
                                )
                            ):
                                continue
                            overlap = 0
                            for k in range(min(len(desc), len(desc_context)), 0, -1):
                                if desc[:k] in desc_context:
                                    overlap = k
                                    break
                            if overlap > best_overlap:
                                best_overlap = overlap
                                matched_item = item
                    if matched_item and (
                        matched_item.get("qty", 1) != qty
                        or abs(float(matched_item.get("unit_price") or 0) - price) > 1
                    ):
                        matched_item["qty"] = qty
                        matched_item["unit_price"] = price
                        matched_item["total"] = qty * price - (matched_item.get("discount") or 0)
                        for item in items:
                            if item is matched_item or not isinstance(item, dict):
                                continue
                            if abs(float(item.get("total") or 0) - qty * price) > 1:
                                continue
                            if abs(float(item.get("unit_price") or 0) - price) > 1:
                                continue
                            item["qty"] = 1
                            item["total"] = price
                    elif not matched_item:
                        ocr_qty_prices.append((qty, price, qty * price))

    used_indices: set[int] = set()
    for oq, op, ot in ocr_qty_prices:
        if oq <= 1:
            continue
        candidates = [
            (idx, item) for idx, item in enumerate(items)
            if isinstance(item, dict) and idx not in used_indices
        ]
        total_match = next(
            ((idx, item) for idx, item in candidates if abs(float(item.get("total") or 0) - ot) < 1),
            None,
        )
        unit_match = next(
            (
                (idx, item) for idx, item in candidates
                if item.get("unit_price") is not None
                and abs(float(item.get("unit_price") or 0) - op) < 1
                and item.get("qty", 1) != oq
            ),
            None,
        )
        match = total_match or unit_match
        if match:
            idx, item = match
            if item.get("qty", 1) != oq or item.get("unit_price") != op:
                item["qty"] = oq
                item["unit_price"] = op
                item["total"] = op * oq - (item.get("discount") or 0)
            used_indices.add(idx)

    for idx, item in enumerate(items):
        if not isinstance(item, dict) or idx in used_indices:
            continue
        if item.get("qty", 1) <= 1:
            continue
        desc = item.get("description", "")
        desc_key = desc[:8] if len(desc) >= 8 else desc
        if not desc_key:
            continue
        for li, ocr_line in enumerate(ocr_lines):
            if desc_key not in ocr_line:
                continue
            has_qty_evidence = any(
                re.search(r'[×xX]\s*\d+|\d+\s*[×xX]|単\d|@\d', ocr_lines[li + j])
                for j in range(4) if li + j < len(ocr_lines)
            )
            if has_qty_evidence:
                break
            for offset in range(1, 4):
                if li + offset >= len(ocr_lines):
                    break
                yen_m = re.search(r'[¥￥]\s*([\d,]+)', ocr_lines[li + offset])
                if yen_m:
                    ocr_price = float(yen_m.group(1).replace(',', ''))
                    if abs(ocr_price - item.get("total", 0)) > 1:
                        item["qty"] = 1
                        item["unit_price"] = ocr_price
                        item["total"] = ocr_price
                    break
            break



def _extract_fuel_usage(extracted, unified_text):
    """Populate usage field for fuel receipts from OCR volume/price data."""
    if extracted.get("usage"):
        return
    items = extracted.get("line_items") or []
    desc_text = ' '.join(
        item.get("description", "") for item in items if isinstance(item, dict)
    )
    if not any(kw in desc_text or kw in unified_text for kw in _FUEL_KEYWORDS):
        return
    volume_m = re.search(r'(\d+)\s*[\.．]\s*(\d+)\s*L', unified_text)
    if not volume_m:
        return
    amount = float(f"{volume_m.group(1)}.{volume_m.group(2)}")
    total = extracted.get("total") or extracted.get("subtotal")
    cost_per = None
    for m in re.finditer(r'(\d{2,3})\s*円', unified_text):
        candidate = float(m.group(1))
        if total and abs(amount * candidate - total) < 5:
            cost_per = candidate
            break
    extracted["usage"] = {
        "amount": amount,
        "unit": "L",
        "cost_per": cost_per,
        "meter_previous": None,
        "meter_current": None,
    }


def _fix_fuel_item_description(extracted, unified_text):
    """Use the printed fuel grade as the single item description."""
    items = extracted.get("line_items") or []
    if len(items) != 1 or not isinstance(items[0], dict):
        return
    for grade in ('レギュラー', 'ハイオク', '軽油', 'ガソリン'):
        if grade in unified_text:
            items[0]["description"] = grade
            return


def _fix_fuel_volume_qty(items, unified_text, receipt_total=None):
    """Normalize fuel receipt volumes (fractional qty) to qty=1.

    The reference for a single-item receipt is receipt.total (printed 合計),
    which equals the item's post-tax price for 内税 receipts and pre-tax + tax
    for 外税. Using receipt.total avoids misfires under the canonical
    pre-tax subtotal convention.
    """
    for item in items:
        if not isinstance(item, dict):
            continue
        qty = item.get("qty", 1)
        desc = item.get("description", "")
        is_fuel = any(kw in desc or kw in unified_text for kw in _FUEL_KEYWORDS)
        if not is_fuel:
            continue
        total = item.get("total", 0)
        # Case 1: fractional qty (e.g., 26.43L)
        if qty != int(qty):
            item["qty"] = 1
            item["unit_price"] = total
            break
        # Case 2: qty=1, single item, but unit_price is per-unit (e.g., yen/liter)
        # and doesn't match receipt total — correct to receipt total.
        if (qty == 1 and len(items) == 1 and receipt_total
                and total > 0 and abs(total - receipt_total) > 5):
            item["unit_price"] = receipt_total
            item["total"] = receipt_total
            break


def _expand_collapsed_items(extracted, unified_text):
    """Expand a single item with qty > 1 into individual items when OCR shows separate entries."""
    items = extracted.get("line_items", [])
    if len(items) != 1:
        return
    item = items[0]
    if not isinstance(item, dict):
        return
    qty = item.get("qty", 1)
    unit_price = item.get("unit_price")
    desc = item.get("description", "")
    if unit_price is None or not desc:
        return
    ocr_lines = unified_text.split('\n')
    ocr_desc_count = sum(
        1 for line in ocr_lines
        if desc in line and '小計' not in line and '合計' not in line
    )
    price_str = str(int(unit_price)) if unit_price == int(unit_price) else str(unit_price)
    has_bulk_pattern = bool(re.search(
        re.escape(price_str) + r'\s*[×xX]\s*\d+', unified_text
    ))
    # Case 1: qty > 1 and OCR shows separate entries
    if qty > 1 and ocr_desc_count >= qty and not has_bulk_pattern:
        extracted["line_items"] = [{
            "description": desc, "qty": 1,
            "unit_price": unit_price, "total": unit_price,
            "tax_category": item.get("tax_category", "0%"),
            "discount": 0, "discount_rate": "",
        } for _ in range(int(qty))]
        extracted["subtotal"] = unit_price * qty
    # Case 2: qty=1 but OCR shows multiple and subtotal confirms
    elif qty == 1 and ocr_desc_count >= 2 and not has_bulk_pattern:
        subtotal = extracted.get("subtotal") or extracted.get("total", 0)
        if subtotal and unit_price > 0 and subtotal > unit_price:
            inferred_qty = round(subtotal / unit_price)
            if inferred_qty >= 2 and abs(inferred_qty * unit_price - subtotal) < 2 and ocr_desc_count >= inferred_qty:
                extracted["line_items"] = [{
                    "description": desc, "qty": 1,
                    "unit_price": unit_price, "total": unit_price,
                    "tax_category": item.get("tax_category", "0%"),
                    "discount": 0, "discount_rate": "",
                } for _ in range(inferred_qty)]


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
