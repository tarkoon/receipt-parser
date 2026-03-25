"""normalization.py — Full-width number conversion (NFKC), OCR text cleanup."""

import re
import unicodedata

# Price-only line: ¥-prefixed (¥656, ¥2,279, ¥168)) or number + JP tax marker (278※, 3除)
_PRICE_LINE_RE = re.compile(
    r'^[¥￥]\s*[\d,]+[)）]?\s*$'
    r'|'
    r'^\d[\d,]*\s*[※X除]\s*$'
)


def normalize_fullwidth(text: str) -> str:
    """Normalize full-width characters to ASCII equivalents.
    Uses NFKC normalization (standard for JP text processing).
    Keeps ¥ symbols so the LLM can distinguish prices from codes.
    """
    text = unicodedata.normalize('NFKC', text)
    return text


def strip_barcode_lines(text: str) -> str:
    """Remove JAN/EAN barcode lines from OCR text.

    Receipts often have barcode numbers like '4580374970018JAN' on separate lines.
    These confuse the LLM into misinterpreting them as prices or quantities.
    """
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that are just barcode numbers (13+ digits optionally followed by JAN/EAN)
        if re.match(r'^\d{8,}(JAN|EAN)?$', stripped):
            continue
        # Strip inline item codes at start of lines (e.g., "000406アマンディ" → "アマンディ")
        stripped = re.sub(r'^0{2,}\d{1,4}[*]?', '', stripped)
        if stripped:
            cleaned.append(stripped)
    return '\n'.join(cleaned)


def rejoin_price_lines(text: str) -> str:
    """Join orphan price lines with the preceding item or label line.

    Cloud Vision's fulltext mode often puts items/labels and their ¥ prices
    on separate lines. This joins each orphan price upward:
    - With a bare summary keyword (小計, 合計, …) if that's the previous line
    - With any Japanese-text line that doesn't already contain ¥
    """
    lines = text.split('\n')
    result: list[str] = []

    for line in lines:
        stripped = line.strip()

        if not _PRICE_LINE_RE.match(stripped):
            result.append(line)
            continue

        # Join upward with previous line if it's a suitable target
        if result:
            prev = result[-1].strip()
            has_jp = bool(re.search(r'[\u3000-\u9fff]', prev))
            if has_jp and '¥' not in prev:
                result[-1] += '  ' + stripped
                continue

        result.append(line)

    return '\n'.join(result)


def clean_handwritten_ocr(text: str) -> str:
    """Clean up OCR text from handwritten receipts (領収証).

    Handwritten receipt forms have pre-printed labels (税抜金額, 消費税額, etc.)
    that Cloud Vision OCR fragments into confusing noise. This strips those
    fragments so the LLM sees only the actual handwritten content.
    """
    lines = text.strip().split('\n')

    # Handwritten receipt detection: short text (< 20 lines), has 金額 or standalone ¥NNNN,
    # and does NOT have typical printed receipt markers like 小計 or 合計 with ¥ amounts
    is_printed = any('小計' in l or '合計' in l for l in lines)
    is_short = len(lines) < 35
    if is_printed or not is_short:
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
    result_lines = result.split('\n')
    for i in range(len(result_lines)):
        curr = result_lines[i].strip()
        if re.match(r'^1\d{3,5}$', curr):
            # Check if nearby lines contain 金額 or 但 (amount/purpose markers)
            context = ' '.join(result_lines[max(0, i-2):i])
            if '金額' in context or '但' in context:
                result_lines[i] = f'金額:{curr[1:]}'
    result = '\n'.join(result_lines)

    return result
