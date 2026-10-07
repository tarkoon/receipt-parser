"""pipeline_slip.py — Payment slip-specific post-processing.

Extracted from pipeline.py for maintainability.
"""

import re


_REFERENCE_LABEL_RE = re.compile(
    r'手番|通番|収納番号|収納No\.?|参照番号|参照No\.?|'
    r'受付番号|受付No\.?|取扱番号|取扱No\.?|照会番号|照会No\.?'
)
_REFERENCE_BLOCK_START_RE = re.compile(
    r'支払票|払込票(?:受領証)?|払込金受領(?:書|証)|払込取扱票|'
    r'(?:CVS|コンビニ)収納用'
)
_REFERENCE_BLOCK_END_RE = re.compile(r'受領印|収納印|収納代行会社|登録番号')
_NON_REFERENCE_LABEL_RE = re.compile(
    r'お客様番号|顧客番号|口座番号|会員番号|カード番号|'
    r'バーコード|請求コード|電話番号|登録番号'
)
_STANDALONE_REFERENCE_RE = re.compile(r'^\s*(\d{13})\s*$')
_COMPOSITE_REFERENCE_RE = re.compile(
    r'^\s*(?:\(\d{2,4}\)\s*)?\d{6}\s*[-‐‑‒–—]\s*'
    r'(\d{13})\d+\s*$'
)
_PAYER_LABEL_RE = re.compile(
    r'ご?依頼人(?!用)|支払人|払込人|納付者|'
    r'\u304a\u5ba2\u69d8\u540d|氏名|宛名|お名前'
)
_STAFF_CONTEXT_RE = re.compile(
    r'(?:\u62c5\u5f53(?:\u8005)?|\u8cac\u4efb(?:\u8005)?|'
    r'\u5f93\u696d\u54e1|\u793e\u54e1|\u5e97\u54e1|\u4fc2\u54e1|\u8ca9\u58f2\u54e1)'
)
_PAYER_LABEL_ONLY_RE = re.compile(
    rf'^\s*(?:{_PAYER_LABEL_RE.pattern})\s*[:：]?\s*$'
)
_ADDRESSEE_LINE_RE = re.compile(
    r'^(?!.*(?:支払|受取人|お客様|責(?:任者)?|担当者?|従業員|社員|店員|係員|レジ|キャッシャ))'
    r'\s*(?P<name>.+?)\s*(?:様|御中)\s*$'
)
_PAYER_NOISE_RE = re.compile(
    r'^(?:(?:受取人|住所|電話|金額|銀行|支店|手数料|収納|請求|支払|登録番号)\s*)+$'
)


def _fix_account_number(extracted: dict, text: str) -> None:
    """Trigger: a printed account owner or an excluded identifier owner.

    Invariant: one distinct customer/billing/account value wins; masked,
    member, pickup and corroborated composite receipt references cannot win.
    """
    owner = re.compile(
        r'(?:お客(?:様|さま)(?:番号|[ \t]*No\.?)|'
        r'(?:顧客|ご?請求|契約|口座)(?:番号|コード|[ \t]*No\.?)|'
        r'\b(?:customer|billing|account)[ \t]+(?:number|no\.?|id))'
        r'[ \t]*[:：#]?[ \t]*', re.IGNORECASE,
    )
    identifier = re.compile(r'(?=[A-Za-z0-9-]*\d)[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*')
    excluded = re.compile(
        r'(?:会員(?:番号|[ \t]*No\.?)?|メンバー(?:番号)?|'
        r'\bmember(?:ship)?\b(?:[ \t]*(?:number|no\.?|id))?|'
        r'\bcard[ \t]+(?:number|no\.?|id)|'
        r'カード(?:番号|[ \t]*No\.?)|(?:WAON|IC)[ \t]*番号|'
        r'(?:レシート|取引|受付|受取|注文|伝票|領収書|呼出|呼び出し|整理|レジ)(?:番号|[ \t]*No\.?))'
        r'[ \t]*[:：#]?[ \t]*(?:\n[ \t]*)?', re.IGNORECASE,
    )
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidates = set()
    rejected = set()
    for index, line in enumerate(lines):
        label = owner.search(line)
        if not label:
            continue
        tail = line[label.end():].strip(' \t[]【】')
        if not tail:
            following = lines[index + 1:index + 3]
            # Currency-marked amounts are not identifiers and cannot compete.
            if following and re.fullmatch(r'[¥￥][\d,]+(?:\.\d+)?', following[0]):
                following = following[1:]
            tail = following[0] if following else ''
        if not identifier.fullmatch(tail):
            continue
        context = ''.join(lines[index + 1:index + 6])
        nearby = '\n'.join(lines[max(0, index - 8):index + 10])
        parts = {part.lstrip('0') or '0' for part in tail.split('-')}
        receipt = re.findall(r'レシート[ \t]*(?:番号|No\.?)\s*(\d+)', nearby, re.IGNORECASE)
        store = re.findall(r'店[ \t]*(?:番号|No\.?)\s*(\d+)', nearby, re.IGNORECASE)
        is_composite_receipt = '-' in tail and any(
            (value.lstrip('0') or '0') in parts for value in receipt
        ) and any((value.lstrip('0') or '0') in parts for value in store)
        is_pickup = '上記' in context and re.search(r'呼び|カウンター', context)
        (rejected if is_composite_receipt or is_pickup else candidates).add(tail)

    current = str(extracted.get('account_number') or '').strip()
    for value in candidates | {current}:
        if value and (
            re.search(excluded.pattern + re.escape(value) + r'(?![A-Za-z0-9])', text, re.IGNORECASE)
            or (value not in candidates and re.search(r'\*{2,}\s*' + re.escape(value.lstrip('*')), text))
        ):
            rejected.add(value)
    if candidates:
        extracted['account_number'] = next(iter(candidates)) if len(candidates) == 1 and not candidates & rejected else None
    elif current in rejected:
        extracted['account_number'] = None


def _payment_reference_candidates(text: str) -> set[str]:
    """Return 13-digit references backed by a label or payment-slip structure."""
    lines = text.splitlines()
    candidates: set[str] = set()

    for index, line in enumerate(lines):
        label = _REFERENCE_LABEL_RE.search(line)
        if label:
            match = re.search(r'(?<!\d)(\d{13})(?!\d)', line[label.end():])
            if match:
                candidates.add(match.group(1))
            elif index + 1 < len(lines):
                match = _STANDALONE_REFERENCE_RE.fullmatch(lines[index + 1])
                if match:
                    candidates.add(match.group(1))

        composite = _COMPOSITE_REFERENCE_RE.fullmatch(line)
        if composite:
            candidates.add(composite.group(1))

    in_reference_block = False
    previous_nonempty = ""
    for line in lines:
        stripped = line.strip()
        if _REFERENCE_BLOCK_START_RE.search(stripped):
            in_reference_block = True
        match = _STANDALONE_REFERENCE_RE.fullmatch(stripped)
        if (
            in_reference_block
            and match
            and not _NON_REFERENCE_LABEL_RE.search(previous_nonempty)
        ):
            candidates.add(match.group(1))
        if _REFERENCE_BLOCK_END_RE.search(stripped):
            in_reference_block = False
        if stripped:
            previous_nonempty = stripped

    return candidates


def _clean_payer_candidate(value: str) -> str | None:
    candidate = value.strip(" \t:：")
    label = _PAYER_LABEL_RE.match(candidate)
    if label:
        candidate = candidate[label.end():].strip(" \t:：")
    addressee = _ADDRESSEE_LINE_RE.fullmatch(candidate)
    if addressee:
        candidate = addressee.group("name").strip()
    candidate = re.sub(r'\s+', ' ', candidate)
    if (
        not 2 <= len(candidate) <= 60
        or re.search(r'\d|[¥￥]', candidate)
        or _PAYER_NOISE_RE.fullmatch(candidate)
        or not re.search(r'[A-Za-zぁ-んァ-ヶ一-鿿]', candidate)
    ):
        return None
    return candidate


def _payer_candidates(text: str) -> dict[str, str]:
    """Return unique printed payers keyed without OCR whitespace variation."""
    lines = text.splitlines()
    candidates: dict[str, str] = {}

    def add(value: str) -> None:
        candidate = _clean_payer_candidate(value)
        if candidate:
            key = re.sub(r'\s+', '', candidate).casefold()
            candidates.setdefault(key, candidate)

    for index, line in enumerate(lines):
        label = _PAYER_LABEL_RE.search(line)
        if label and not _STAFF_CONTEXT_RE.search(line):
            tail = line[label.end():].strip(" \t:：")
            if tail:
                add(tail)
            elif _PAYER_LABEL_ONLY_RE.fullmatch(line) and index + 1 < len(lines):
                add(lines[index + 1])
        addressee = _ADDRESSEE_LINE_RE.fullmatch(line)
        if addressee:
            add(addressee.group("name"))

    return candidates


def postprocess_payment_slip(extracted: dict, unified_text: str, raw_text: str = "") -> dict:
    """Apply payment slip-specific post-processing to the LLM extraction.

    Args:
        extracted: LLM extraction dict
        unified_text: Normalized text (after barcode stripping)
        raw_text: Original text before barcode stripping (for reference extraction)
    """
    # Trigger: a reference label, a payment-reference block, or the composite
    # payment-barcode layout. Invariant: exactly one distinct reference wins.
    text = raw_text or unified_text
    _fix_account_number(extracted, text)
    references = _payment_reference_candidates(text)
    extracted["payment_reference"] = next(iter(references)) if len(references) == 1 else None

    # Trigger: an explicit payer label or addressee honorific. Invariant: one
    # whitespace-normalized identity is required; absent/competing names clear.
    payers = _payer_candidates(text)
    extracted["payer"] = next(iter(payers.values())) if len(payers) == 1 else None

    # Date: null out dates from 発行日 (billing issuance date, not payment date)
    if extracted.get("date"):
        date_text = unified_text or raw_text or ""
        m = re.search(r'発行日\s*(20\d{2})[年/]0?(\d{1,2})[月/]0?(\d{1,2})', date_text)
        if m:
            billing_date = f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
            if extracted["date"] == billing_date:
                extracted["date"] = None

    return extracted
