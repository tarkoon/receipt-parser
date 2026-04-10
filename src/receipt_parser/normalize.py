"""normalize.py — Full-width number conversion (NFKC), OCR text cleanup."""

import re
import unicodedata

# Price-only line: ¥-prefixed (¥656, ¥2,279, ¥168, ¥200外) or number + JP tax marker (278※, 3除)
_PRICE_LINE_RE = re.compile(
    r'^[¥￥]\s*[\d,]+\s*[)）軽外内]?\s*$'
    r'|'
    r'^\d[\d,]*\s*[※X除軽]\s*$'
)


def normalize_fullwidth(text: str) -> str:
    """Normalize full-width characters to ASCII equivalents.
    Uses NFKC normalization (standard for JP text processing).
    Keeps ¥ symbols so the LLM can distinguish prices from codes.
    """
    text = unicodedata.normalize('NFKC', text)
    return text


def strip_barcode_lines(text: str) -> str:
    """Remove barcode lines from OCR text.

    Handles JAN/EAN, UPC-A, GTIN-14, Code 128, and other long digit-only lines.
    These confuse the LLM into misinterpreting them as prices or quantities.
    """
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that are just long digit sequences (8+ digits) — covers
        # JAN/EAN, UPC-A, GTIN-14, Code 128, and other barcode formats
        if re.match(r'^\d{8,}\s*(JAN|EAN|UPC|GTIN)?\s*$', stripped, re.IGNORECASE):
            continue
        # Strip inline item codes at start of lines, preserving tax markers
        # e.g., "000406*トーラク" → "※トーラク" (preserve * as ※ tax marker)
        m_code = re.match(r'^0{2,}\d{1,4}(\*?)\s*', stripped)
        if m_code:
            had_marker = m_code.group(1) == '*'
            stripped = stripped[m_code.end():]
            if had_marker and stripped:
                stripped = '※' + stripped  # Preserve tax marker
        if stripped:
            cleaned.append(stripped)
    return '\n'.join(cleaned)


def rejoin_price_lines(text: str) -> str:
    """Join orphan price lines with their corresponding item lines.

    Only operates within the item section of the receipt — between the first
    item-like line and the subtotal (小計) or equivalent summary marker.
    Lines outside this zone are left untouched.

    Within the item zone, handles:
    1. Single orphan: price line joined to the nearest priceless item above.
    2. Block pattern: N priceless items followed by N price lines → matched in order.
    """
    lines = text.split('\n')

    def _is_price(s: str) -> bool:
        return bool(_PRICE_LINE_RE.match(s.strip()))

    def _price_value(s: str) -> float | None:
        """Extract numeric value from a price line, or None if not a price."""
        s = s.strip()
        m = re.search(r'[\d,]+', s)
        if m:
            try:
                return float(m.group(0).replace(',', ''))
            except ValueError:
                pass
        return None

    # Markers that signal the END of the item section
    _SECTION_END = re.compile(r'小計|合計|現計|税率|外税|内税|消費税|WAON|クレジット|お預り|お釣り')

    # Pattern for inline price suffix: digit(s) + tax marker at end of line
    # e.g. "食品ポリ袋L (バイオマス30 3除" has "3除" = inline price
    _INLINE_PRICE_SUFFIX = re.compile(r'\d[\d,]*\s*[※X除軽]\s*$')

    def _has_inline_price(s: str) -> bool:
        """Check if a line already has a price suffix (digit + tax marker)."""
        return bool(_INLINE_PRICE_SUFFIX.search(s.strip()))

    def _is_item_candidate(s: str) -> bool:
        """Line that looks like an item: has Japanese text, no ¥, not a summary."""
        s = s.strip()
        if not s or _is_price(s):
            return False
        if not re.search(r'[\u3000-\u9fff]', s):
            return False
        if '¥' in s or '￥' in s:
            return False
        if _SECTION_END.search(s):
            return False
        # Qty/price detail lines like "(@100 × 2個)" are not items
        if re.match(r'^\(.*[×xX].*[個コ点]\s*\)', s):
            return False
        return True

    def _needs_price(s: str) -> bool:
        """Item candidate that does NOT already have an inline price."""
        return _is_item_candidate(s) and not _has_inline_price(s)

    # --- Step 1: Find the item section boundaries ---
    # Item section starts at the first line with ¥ or a price-line pattern,
    # and ends at the first summary marker (小計, 合計, etc.)
    def _count_trailing_priceless(end_idx: int, start_idx: int = 0) -> int:
        """Count consecutive priceless item candidates ending at end_idx."""
        count = 0
        for back in range(end_idx, start_idx - 1, -1):
            if _needs_price(lines[back].strip()):
                count += 1
            else:
                break
        return count

    def _collect_prices_after(marker_idx: int, needed: int) -> list[int]:
        """Scan past a section marker for price line indices.

        Stops early if a price equals the running sum of collected prices
        (that's the subtotal, not an item price).
        """
        found: list[int] = []
        running_sum = 0.0
        for j in range(marker_idx + 1, min(marker_idx + needed * 3 + 5, len(lines))):
            if _is_price(lines[j].strip()):
                val = _price_value(lines[j].strip())
                if val is not None and len(found) >= 2 and abs(val - running_sum) < 1:
                    break  # this price IS the subtotal of items so far
                found.append(j)
                running_sum += val or 0
                if len(found) >= needed:
                    return found
        return found

    item_start = None
    item_end = len(lines)
    for i, line in enumerate(lines):
        s = line.strip()
        if item_start is None:
            # Look for the first line that has a price or is a priced item
            if '¥' in s or '￥' in s or _is_price(s) or (
                    _is_item_candidate(s) and i + 1 < len(lines) and _is_price(lines[i + 1].strip())):
                item_start = i
            elif _SECTION_END.search(s):
                # Section marker hit before any priced line — OCR may have
                # read items in one column and prices in another.
                # Directly join trailing priceless items with post-marker prices.
                trailing = _count_trailing_priceless(i - 1)
                if trailing > 0:
                    price_indices = _collect_prices_after(i, trailing)
                    if price_indices:
                        n_pairs = min(trailing, len(price_indices))
                        for k in range(n_pairs):
                            item_idx = i - n_pairs + k
                            price_idx = price_indices[k]
                            lines[item_idx] += '  ' + lines[price_idx].strip()
                            lines[price_idx] = ''
                        # Lines modified in-place; rebuild and return since
                        # item_start was never set (no normal section to process)
                        return '\n'.join(l for l in lines if l)
        elif _SECTION_END.search(s):
            # Edge case: OCR may place a section marker (小計/合計) between
            # items and their prices when it reads a two-column layout.
            # Directly join trailing priceless items with post-marker prices.
            trailing = _count_trailing_priceless(i - 1, item_start)
            if trailing > 0:
                price_indices = _collect_prices_after(i, trailing)
                if price_indices:
                    n_pairs = min(trailing, len(price_indices))
                    for k in range(n_pairs):
                        item_idx = i - n_pairs + k
                        price_idx = price_indices[k]
                        lines[item_idx] += '  ' + lines[price_idx].strip()
                        lines[price_idx] = ''

            item_end = i
            break

    if item_start is None:
        return text  # No item section found, return as-is

    # --- Step 2: Within the item section, do block matching ---
    before = lines[:item_start]
    section = lines[item_start:item_end]
    after = lines[item_end:]

    # Step 2a: Pull item candidates from "before" zone into the section start
    # if there are more leading prices than items.  E.g., the section might start
    # with one item name followed by two prices — the extra price belongs to
    # an item that ended up just above the section boundary.
    lead_items = 0
    for l in section:
        if _is_item_candidate(l.strip()):
            lead_items += 1
        else:
            break
    lead_prices = 0
    for l in section[lead_items:]:
        if _is_price(l.strip()):
            lead_prices += 1
        else:
            break
    deficit = lead_prices - lead_items
    while deficit > 0 and before:
        if _is_item_candidate(before[-1].strip()):
            section.insert(0, before.pop())
            deficit -= 1
        else:
            break

    # Block matching: find runs of priceless items followed by price lines.
    # Items that already have an inline price (e.g. "食品ポリ袋L 3除") are
    # skipped so they don't consume prices meant for subsequent items.
    resolved = list(section)
    i = 0
    while i < len(resolved):
        if not _needs_price(resolved[i].strip()):
            i += 1
            continue

        # Count consecutive priceless item-candidate lines
        istart = i
        while i < len(resolved) and _needs_price(resolved[i].strip()):
            i += 1
        iend = i

        # Skip over items that already have inline prices (they don't need pairing)
        while i < len(resolved) and _is_item_candidate(resolved[i].strip()) and _has_inline_price(resolved[i].strip()):
            i += 1

        # Count consecutive price lines immediately after
        pstart = i
        while i < len(resolved) and _is_price(resolved[i].strip()):
            i += 1
        pend = i

        pairs = min(iend - istart, pend - pstart)
        if pairs == 0:
            continue

        for j in range(pairs):
            resolved[istart + j] += '  ' + resolved[pstart + j].strip()
        for j in range(pairs):
            resolved[pstart + j] = None  # mark for removal

    section = [l for l in resolved if l is not None]

    # --- Step 3: Single orphan pass within the item section ---
    result: list[str] = []
    for line in section:
        stripped = line.strip()
        if not _is_price(stripped):
            result.append(line)
            continue

        # Look back for nearest priceless item candidate
        joined = False
        for back in range(1, min(4, len(result) + 1)):
            prev = result[-back].strip()
            if _is_item_candidate(prev):
                result[-back] += '  ' + stripped
                joined = True
                break
            # Stop at pure price lines or non-Japanese content
            if _is_price(prev):
                break  # Pure price line = boundary
            if not re.search(r'[\u3000-\u9fff]', prev):
                break  # Non-Japanese line = boundary
            # Lines with ¥ AND Japanese text = priced items → skip over them

        if not joined:
            result.append(line)

    # --- Step 4: Handle orphan prices at section start ---
    # When item names appear in the "before" zone (just above the detected section
    # start), their prices land inside the section as unmatched orphans.
    # E.g.: "ミルクカスタードシュー" (before) → "¥138軽" (section orphan).
    # Attach leading orphan prices to trailing item candidates in "before".
    while result and before and _is_price(result[0].strip()):
        # Find the last item candidate in "before"
        attached = False
        for bi in range(len(before) - 1, -1, -1):
            if _is_item_candidate(before[bi].strip()):
                before[bi] += '  ' + result[0].strip()
                result.pop(0)
                attached = True
                break
            # Stop at non-candidate lines that aren't blank
            if before[bi].strip():
                break
        if not attached:
            break

    return '\n'.join(before + result + after)


def clean_handwritten_ocr(text: str, ocr_confidence: float | None = None) -> str:
    """Clean up OCR text from handwritten receipts (領収証).

    Handwritten receipt forms have pre-printed labels (税抜金額, 消費税額, etc.)
    that Cloud Vision OCR fragments into confusing noise. This strips those
    fragments so the LLM sees only the actual handwritten content.

    Detection uses OCR confidence when available (handwritten = low avg confidence),
    with line count as fallback.
    """
    lines = text.strip().split('\n')

    # Handwritten receipt detection:
    # Primary: low OCR confidence (< 0.7) suggests handwritten content
    # Fallback: short text (< 35 lines), no printed receipt markers
    _printed_markers = ('小計', '合計', 'レジ', 'TEL', '税', '円', '%対象', 'お預り', '釣銭')
    is_printed = any(m in l for l in lines for m in _printed_markers)
    if ocr_confidence is not None:
        is_handwritten = ocr_confidence < 0.7 and not is_printed
    else:
        is_short = len(lines) < 35
        is_handwritten = is_short and not is_printed
    if not is_handwritten:
        return text  # Printed receipt, don't clean

    # Remove common pre-printed form label fragments
    noise_patterns = [
        r'^内税消.*$', r'^消$', r'^抜費$', r'^金税金税$', r'^訳額額額額$',
        r'^収$', r'^証$', r'^領$', r'^D$',
        r'^金額.*年$',  # form label fragment line like "金額  訳額額額額  8年  年"
        r'^\d年$',  # standalone year fragments like "8年"
    ]
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        is_noise = any(re.match(p, stripped) for p in noise_patterns)
        if not is_noise:
            cleaned.append(line)

    result = '\n'.join(cleaned)

    # Strip inline noise from pre-printed form labels
    # These fragments come from blank form fields (税抜金額, 消費税額, etc.)
    inline_noise = ['金税金税', '訳額額額額', '  収']
    for noise in inline_noise:
        result = result.replace(noise, '')

    # Clean up extra whitespace from removals
    result = re.sub(r'  +', ' ', result)
    result = re.sub(r'^\s+$', '', result, flags=re.MULTILINE)
    result = re.sub(r'\n{2,}', '\n', result)

    # Replace ¥ with explicit text marker in handwritten receipts.
    # The handwritten yen sign ¥ looks like the digit 1 to LLMs,
    # causing ¥3000 to be read as 13000. Making it explicit fixes this.
    result = re.sub(r'¥(\d)', r'金額:\1', result)

    # Fix absorbed ¥: if a line near "金額" is just a number starting with 1,
    # the leading 1 is likely the ¥ sign misread as a digit.
    # e.g. "金額\n13000" → "金額\n金額:3000" (the 1 was ¥)
    # Guards: only apply when the number is 3-5 digits after the leading 1
    # (i.e., 1,000–99,999 range), the context marker is on the immediately
    # preceding line (not 2 lines away), and the amount hasn't already been
    # tagged with 金額: by the earlier ¥ replacement pass.
    result_lines = result.split('\n')
    for i in range(len(result_lines)):
        curr = result_lines[i].strip()
        if re.match(r'^1\d{3,4}$', curr):  # 1,000–19,999 range only
            # Only check immediately preceding line (not 2 lines back)
            prev = result_lines[i-1].strip() if i > 0 else ''
            # Context marker must be a label-only line (no digits = not an amount)
            if ('金額' in prev or '但' in prev) and not re.search(r'\d{2,}', prev):
                result_lines[i] = f'金額:{curr[1:]}'
    result = '\n'.join(result_lines)

    return result
