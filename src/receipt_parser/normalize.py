"""normalize.py — Full-width number conversion (NFKC), OCR text cleanup."""

import re
import unicodedata

# Generic Japanese receipt boilerplate banners — never product names.
# Used to keep rejoin_price_lines from pairing orphan totals-zone prices
# with header/footer banner text in the "before" zone (which would
# fabricate phantom items the LLM later picks up).
_BANNER_PHRASE_RE = re.compile(
    r'ぜひ当店でお買物くださいませ|'
    r'ありがとうございました|ありがとうございます|'
    r'毎度ありがとうございます|'
    r'毎月\s*\d+\s*日.*感謝デ[ーー]|'
    r'お客さま感謝デ[ーー]|'
    r'印は軽減税率|軽減税率\s*8?\s*%?\s*対象商品|'
    r'お買上商品数|お買上点数|お買上げ点数|'
    r'ポイントの有効期限|累計ポイント|'
    r'今回獲得|現在のポイント|'
    r'本人確認(?:省略)?|'
    r'クレジットカード売上票|お客様控え?|'
    r'当店をご利用|またのご利用|またお越し|'
    r'お問い合わせ|営業時間|定休日|'
    r'カードお取扱日|取引内容|伝票番号|承認番号|'
    r'プロの品質とプロの価格|'
    r'上記金額正に領収|上記正に領収|'
    r'本書保管|印字面|'
    r'の商品です|まとめ値引|'
    r'^[A-Z]\s*[:：]\s*\d+\s*[個コ点]|'
    r'^\s*消費税等?\s*$'
)

# Price-only line: ¥-prefixed (¥656, ¥2,279, ¥168, ¥200外) or number + JP tax marker (278※, 3除)
_PRICE_LINE_RE = re.compile(
    r'^[¥￥]\s*[\d,]+\s*[)）軽外内]?\s*$'
    r'|'
    r'^\d[\d,]*\s*[※\*X除軽]\s*$'
)


def normalize_fullwidth(text: str) -> str:
    """Normalize full-width characters to ASCII equivalents.
    Uses NFKC normalization (standard for JP text processing).
    Keeps ¥ symbols so the LLM can distinguish prices from codes.
    """
    text = unicodedata.normalize('NFKC', text)
    # Known OCR-segmentation errors on financial labels. Cloud Vision
    # occasionally splits 計 (言+十) into 富 + 士, so 小計 reads as 小富士.
    # The replacement is safe because 小富士 is virtually never a product
    # name in a Japanese receipt context.
    text = re.sub(r'(?<!\S)小富士(?!\S)', '小計', text)
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


# Lines that are pure loyalty-point / bonus-point indicators in the item zone.
# Japanese supermarket receipts (Aeon, Maxvalu, etc.) print bonus points
# inline between items, often as standalone fragments like "(ボーナスポイント"
# / "10P)" or one-line "(ボーナスポイント 10P)". They are NOT items, but they
# appear in the price column and break per-row item↔price matching during
# rejoin_price_lines, scrambling subsequent items' prices.
_BONUS_POINT_LINE_RE = re.compile(
    r'^\(?\s*(?:'
    r'ボーナスポイント[\s\d]*\)?'        # "(ボーナスポイント" or "(ボーナスポイント 10P)"
    r'|\d+\s*P\)?'                        # "(10P)", "10P)", "40P"
    r')\s*$'
)


def strip_bonus_point_lines(text: str) -> str:
    """Remove standalone bonus/loyalty-point indicator lines.

    Generic for Japanese receipts: any line that is exclusively a loyalty
    point fragment is dropped. Real product lines that mention "P" or
    "ポイント" alongside a description are preserved (the regex requires
    the entire line to match).
    """
    return '\n'.join(
        line for line in text.split('\n')
        if not _BONUS_POINT_LINE_RE.match(line.strip())
    )


_STRIP_SAFE_BANNER_RE = re.compile(
    r'ぜひ当店でお買物くださいませ|'
    r'毎度ありがとうございます|'
    r'毎月\s*\d+\s*日.*感謝デ[ーー]|'
    r'お客さま感謝デ[ーー]|'
    r'印は軽減税率|軽減税率\s*8?\s*%?\s*対象商品|'
    r'お買上商品数|お買上点数|お買上げ点数|'
    r'ポイントの有効期限|累計ポイント|'
    r'今回獲得|現在のポイント|'
    r'クレジットカード売上票|お客様控え?|'
    r'当店をご利用|またのご利用|またお越し|'
    r'プロの品質とプロの価格|'
    r'本書保管|印字面|'
    r'の商品です|まとめ値引|'
    r'^[A-Z]\s*[:：]\s*\d+\s*[個コ点]'
)


def strip_banner_lines(text: str) -> str:
    """Replace standalone receipt-banner lines with empty placeholders.

    Conservative — only the specific boilerplate phrases listed in
    _STRIP_SAFE_BANNER_RE are stripped. Acknowledgement phrases like
    '上記正に領収' which sometimes contain the receipt date inline are
    NOT stripped (handled separately by item-level filters).

    IMPORTANT: empties the line content but preserves line count. The LLM
    is sensitive to line positioning — wholesale removing lines causes
    extraction regressions on otherwise-passing receipts.

    Generic across receipts.
    """
    out = []
    for line in text.split('\n'):
        s = line.strip()
        if not s:
            out.append(line)
            continue
        if _STRIP_SAFE_BANNER_RE.search(s):
            out.append('')  # preserve line count
            continue
        out.append(line)
    return '\n'.join(out)


_QTY_DETAIL_RE = re.compile(
    r'^[\(\<]?\s*(\d+)\s*[個コ点]\s*[xX×]\s*(?:単)?\s*(\d+)\s*[\)\>]?\s*$'
    r'|'
    r'^[\(\<]?\s*(\d+)\s*[個コ点]\s*[xX×]?\s*(\d+)\s*[\)\>]?\s*$'
)


def join_split_qty_details(text: str) -> str:
    """Join qty-detail lines that OCR split across two lines.

    Example:
        '(2個 X        <- L1: open paren + qty + multiplier
        '128)'         <- L2: unit price + close paren

    Becomes:
        '(2個 X 128)'  <- joined onto L1; L2 becomes empty

    Generic across receipts where Cloud Vision splits parenthetical qty
    notation across rows. Preserves line count via empty placeholder.
    """
    lines = text.split('\n')
    out = list(lines)
    _OPEN = re.compile(r'^[\(\<]?\s*(\d+)\s*[個コ点]\s*[xX×]\s*(?:単)?\s*$')
    _CLOSE = re.compile(r'^\s*(\d+)\s*[\)\>]\s*$')
    for i in range(len(out) - 1):
        if _OPEN.match(out[i].strip()) and _CLOSE.match(out[i + 1].strip()):
            out[i] = out[i].rstrip() + ' ' + out[i + 1].strip()
            out[i + 1] = ''
    return '\n'.join(out)


def _qty_detail_total(s: str) -> float | None:
    """If s is a qty-detail line, return qty * unit. Else None.

    Examples:
        '(3個 X 単68)' -> 204
        '2コX単5' -> 10
        '<2個 X 単248)' -> 496
        '3コ X358' -> 1074
    """
    m = _QTY_DETAIL_RE.match(s.strip())
    if not m:
        return None
    if m.group(1) and m.group(2):
        return float(m.group(1)) * float(m.group(2))
    if m.group(3) and m.group(4):
        return float(m.group(3)) * float(m.group(4))
    return None


def _shift_misaligned_inline_prices(text: str) -> str:
    """Detect inline prices that belong to the previous priceless line.

    Pattern: line N has 'desc + inline_price_X', line N+1 is a qty-detail
    saying qty*unit=Y, where Y != X. Then inline_price_X actually belongs
    to line N-1 (which is a priceless description), and line N's true
    total is Y (often appearing on a later line as bare digits).

    Concrete example (AEON column-format):
        TV いりごま 白         <- L33 priceless
        たまねぎ バラ  98*     <- L34 desc + inline 98 (BELONGS TO L33)
        (3個 X 単68)          <- L35 qty=3 unit=68 → 204 (true total for L34)
        204 A                 <- L36 bare price 204 (matches qty*unit)

    Transform:
        TV いりごま 白  98*    <- L33 + L34's misaligned price
        たまねぎ バラ          <- L34 priceless (price moved away)
        (3個 X 単68)
        204 A

    Generic: only fires when the qty*unit math contradicts the inline price.
    """
    lines = text.split('\n')
    out = list(lines)

    # Match a desc with a trailing inline bare-digit price (with optional marker).
    # Capture: prefix description (must contain Japanese), the digit value,
    # optional marker.
    _DESC_WITH_INLINE = re.compile(
        r'^(.*?[ぁ-んァ-ン一-龥].*?)(\s+)([\d,]+)(\s*[\*※]?)\s*$'
    )

    for i in range(len(out) - 1):
        m_desc = _DESC_WITH_INLINE.match(out[i].strip())
        if not m_desc:
            continue
        prefix = m_desc.group(1).strip()
        try:
            inline_price = float(m_desc.group(3).replace(',', ''))
        except ValueError:
            continue
        # Tiny prices (single digit) and very large prices are unlikely to be
        # the misalignment pattern — skip to avoid false positives.
        if inline_price < 5 or inline_price > 9999:
            continue

        qty_total = _qty_detail_total(out[i + 1])
        if qty_total is None:
            continue
        # Only fire when math contradicts the inline price clearly
        if abs(qty_total - inline_price) <= 2:
            continue

        # Look upward for a priceless desc line to attach to
        for back in range(1, 4):
            j = i - back
            if j < 0:
                break
            up = out[j].strip()
            if not up or '¥' in up or '￥' in up:
                continue
            # Skip qty-detail lines themselves
            if _qty_detail_total(up) is not None:
                continue
            # Must look like an item line (Japanese, no inline trailing digits)
            if not re.search(r'[ぁ-んァ-ン一-龥]', up):
                continue
            if _DESC_WITH_INLINE.match(up):
                continue  # already has its own price
            if re.search(r'小計|合計|外税|内税|消費税|お預|釣銭', up):
                break  # totals zone — stop
            # Attach misaligned price to this priceless line
            marker = m_desc.group(4) or ''
            digit_str = m_desc.group(3) + marker
            out[j] = out[j].rstrip() + '  ' + digit_str.strip()
            # Strip the inline price from line i, leaving just the description
            out[i] = prefix
            break

    return '\n'.join(out)


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
        # Qty/price detail lines like "(@100 × 2個)" or "<2個 X 単248)" — the
        # OCR sometimes reads "(" as "<" so accept either as the open bracket,
        # and the multiplier symbol may appear before or after the count marker.
        if re.match(r'^[\(\<](?=.*[×xX])(?=.*[個コ点])', s):
            return False
        # Receipt boilerplate banners — never product names. Without this,
        # rejoin_price_lines pairs orphan totals-zone prices with banner
        # lines in the "before" zone, fabricating phantom items.
        if _BANNER_PHRASE_RE.search(s):
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

    # Bare-digit price line — accepted only inside _collect_prices_after where
    # we already know we're in the totals zone. Outside the totals zone bare
    # digits could be quantities, codes, etc.
    _BARE_DIGIT_PRICE_RE = re.compile(r'^\d{1,3}(?:,\d{3})*$|^\d{1,6}$')

    def _collect_prices_after(marker_idx: int, needed: int) -> list[int]:
        """Scan past a section marker for price line indices.

        Stops early if a price equals the running sum of collected prices
        (that's the subtotal, not an item price).
        """
        found: list[int] = []
        running_sum = 0.0
        for j in range(marker_idx + 1, min(marker_idx + needed * 3 + 5, len(lines))):
            stripped = lines[j].strip()
            if _is_price(stripped) or _BARE_DIGIT_PRICE_RE.match(stripped):
                val = _price_value(stripped)
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
    def _is_price_or_bare_digit(s: str) -> bool:
        """Inside the items zone, bare digits like '1,498' or '198' are
        legitimate price lines (column-format receipts)."""
        s = s.strip()
        return _is_price(s) or bool(_BARE_DIGIT_PRICE_RE.match(s))

    # Misread tax-marker pattern: '100%' alone is OCR's mistake for '100※'
    # or '108※' (the trailing '8※' becomes '%'). Treat as a price line in
    # the items zone.
    _MISREAD_MARKER_RE = re.compile(r'^\d{2,6}\s*%\s*$')

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

        # Count consecutive price lines immediately after. Accept bare digits
        # as price lines only when ≥2 priceless items precede AND ≥2 bare-digit
        # lines follow (column-format signal). Single bare digits are
        # ambiguous (could be product code, qty, etc.).
        pstart = i
        n_priceless = iend - istart
        accept_bare = n_priceless >= 2
        # Bare digits and misread '%' markers accepted only when ≥3 priceless
        # items precede AND the count of price-like lines that follow exactly
        # matches. Conservative — column-format with multiple unmarked prices
        # in a row is rare; matching multiple of these is a strong signal.
        n_priceless = iend - istart
        accept_extended = n_priceless >= 3
        while i < len(resolved):
            s = resolved[i].strip()
            if _is_price(s):
                i += 1
            elif accept_extended and _BARE_DIGIT_PRICE_RE.match(s):
                i += 1
            elif accept_extended and _MISREAD_MARKER_RE.match(s):
                i += 1
            else:
                break
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


_TOTALS_LABEL_RE = re.compile(
    r'^\(?\s*('
    r'小\s*計|合\s*計|現計|総額|総合計|お会計|'
    r'外税(?:\s*\d+\s*%)?(?:\s*対象?額?)?|'
    r'内税(?:\s*\d+\s*%)?(?:\s*対象?額?)?|'
    r'\d+\s*%\s*対象(?:額)?|消費税[等額]?(?:\s*\d+\s*%)?(?:\s*対象?額?)?|'
    r'税率\s*\d+\s*%\s*(?:(?:課税)?対象?額?|税額)|'
    r'内\s*消費税(?:\s*\d+\s*%)?|内\s*ガソリン税|内\s*石油|内\s*税分|'
    r'非課税対象|非課税|軽減?税率?|'
    r'お預り|お預\s*り|お釣り|お釣\s*り|おつり|釣銭|'
    r'現金|電子マネー|WAON支払|クレジット|カード'
    r')\s*[\)）]?\s*[※\*]?\s*$'
)
_VALUE_LINE_RE = re.compile(r'^[\d\s\)\]コX]*\s*([¥￥])?\s*([\d,]+)\s*[\)）]?\s*$')


def rejoin_totals_label_value_columns(text: str) -> str:
    """Interleave label-block + value-block patterns in the totals zone.

    Some receipts (notably McDonald's) print every totals label on its own
    line in a left column, then every yen value in a right column:

        小計            ¥2,050
        (内消費税        ¥151)
        合計            ¥2,050
        お預り           ¥5,050
        おつり           ¥3,000

    OCR reconstructs this column-by-column:

        小計
        (内消費税
        合計
        お預り
        おつり
        ¥2,050
        ¥151)
        ¥2,050
        ¥5,050
        ¥3,000

    The forward look-ahead in `_extract_financial_totals_impl` then picks
    お預り's value as 合計 because it falls within the look-ahead window
    after the 合計 label. This function detects the pattern (≥3 consecutive
    label-only lines followed by ≥N value-only lines) and rewrites the
    text so each label sits next to its position-paired value.

    Conservative — only fires when:
      - ≥3 consecutive lines match a known totals label
      - At least N value-only lines follow (with optional non-Japanese noise)
      - No item-name lines (Japanese text without label) intervene
    """
    lines = text.split('\n')

    def _is_label_line(s: str) -> bool:
        s = s.strip()
        if not s or '¥' in s or '￥' in s:
            return False
        return bool(_TOTALS_LABEL_RE.match(s))

    def _value_in(s: str) -> float | None:
        s = s.strip()
        if not s or re.search(r'[ぁ-んァ-ン一-龥]', s):
            return None
        m = _VALUE_LINE_RE.match(s)
        if not m:
            return None
        try:
            return float(m.group(2).replace(',', ''))
        except ValueError:
            return None

    new_lines = list(lines)
    i = 0
    while i < len(lines):
        if not _is_label_line(lines[i]):
            i += 1
            continue
        # Collect a run of consecutive label lines.
        run_start = i
        labels: list[tuple[int, str]] = []
        while i < len(lines) and _is_label_line(lines[i]):
            labels.append((i, lines[i].strip()))
            i += 1
        # Require ≥2 labels. Higher confidence when ≥3.
        if len(labels) < 2:
            continue
        # Collect ALL consecutive values starting after the last label.
        # Allow noise lines (digit fragments like "1コ" or "11]") between
        # values. Stop at the first name-like Japanese line, OR at any
        # 小計-style label that begins a new sub-block.
        values: list[tuple[int, str, float]] = []
        scan_end = min(len(lines), labels[-1][0] + len(labels) * 4 + 12)
        j = labels[-1][0] + 1
        while j < scan_end:
            s = lines[j].strip()
            if not s:
                j += 1
                continue
            v = _value_in(s)
            if v is not None:
                values.append((j, s, v))
                j += 1
                continue
            # Tolerate noise fragments (item-count remnants, OCR bracket
            # remnants) but break on name-like Japanese.
            if re.match(r'^[\d\s\)\]コ個X\(]+$', s):
                j += 1
                continue
            if re.search(r'[ぁ-んァ-ン一-龥]', s):
                break
            j += 1
        # Require EXACT count match — fewer values than labels means we
        # missed some, more values means the value run extends past the
        # current label set (e.g., item prices interleaved with totals).
        if len(values) != len(labels):
            continue
        # Apply: append paired value to label line, blank the value line.
        for (li, lab), (vi, vstr, vv) in zip(labels, values[:len(labels)]):
            # Strip any leading-noise prefix from vstr (e.g. "11] ¥2,050" → "¥2,050").
            cleaned_v = re.sub(r'^[\d\s\)\]コX]+(?=[¥￥])', '', vstr)
            if not re.match(r'^[¥￥]', cleaned_v.strip()):
                # Add a ¥ prefix if the value is bare digits.
                cleaned_v = '¥' + cleaned_v.strip().lstrip('¥￥').strip()
            new_lines[li] = lab + ' ' + cleaned_v.strip()
            new_lines[vi] = ''
    return '\n'.join(new_lines)


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
