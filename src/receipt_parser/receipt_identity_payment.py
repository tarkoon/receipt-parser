"""Receipt identity, date, and payment repair helpers."""

import re

from .patterns import (
    ERA_TABLE,
    _COMPANY_SUFFIX_RE,
    _DECORATIVE_RE,
    _HEADER_PHONE_MERCHANT_RE,
    _OFFICIAL_AUTHORITY_HEADER_RE,
    _OFFICIAL_DEPARTMENT_LINE_RE,
    era_to_western_year,
    should_override_field,
)
from .normalize import _TOTALS_LABEL_RE
from .receipt_location import (
    _ASCII_BRAND_HEADER_SCAN_LIMIT,
    _is_ascii_brand_location_suffix,
)
from .pipeline_slip import _payer_candidates
from .receipt_totals import _sum_taxable_amounts


_CREDIT_LABEL = ''.join(chr(c) for c in (0x30AF, 0x30EC, 0x30B8, 0x30C3, 0x30C8))
_ELECTRONIC_MONEY_LABEL = ''.join(chr(c) for c in (0x96FB, 0x5B50, 0x30DE, 0x30CD, 0x30FC))
_TRANSPORT_LABEL = ''.join(chr(c) for c in (0x4EA4, 0x901A, 0x7CFB))
_CARD_LABEL = ''.join(chr(c) for c in (0x30AB, 0x30FC, 0x30C9))
_CARD_NOISE_RE = re.compile(
    '|'.join(
        ''.join(chr(c) for c in chars)
        for chars in (
            (0x30AE, 0x30D5, 0x30C8),
            (0x9664, 0x304F),
            (0x6709, 0x52B9, 0x671F, 0x9650),
            (0x7D2F, 0x8A08),
            (0x30DD, 0x30A4, 0x30F3, 0x30C8),
            (0x4F1A, 0x54E1, 0x52DF, 0x96C6, 0x4E2D),
            (0x304A, 0x3059, 0x3059, 0x3081),
        )
    )
)
_PAYMENT_TOKEN_RE = re.compile(
    rf'{_CREDIT_LABEL}|VISA|Master(?:Card)?|JCB|AMEX|QUICPay|'
    rf'デビット(?:カード)?|銀行振込|口座振替|口座引落|'
    rf'{_ELECTRONIC_MONEY_LABEL}|(?<![A-Za-z])iD(?![A-Za-z])|'
    rf'PayPay|{_TRANSPORT_LABEL}|(?<![A-Za-z])IC(?![A-Za-z])',
    re.IGNORECASE,
)
_PAYMENT_AMOUNT_PATTERN = r'(?:[¥￥][ \t]*)?[0-9][0-9,]*(?:\.[0-9]+)?[ \t]*円?'
_CASH_TENDER_LABEL_RE = re.compile(
    r'^[ \t]*(?:現金[ \t]*)?(?:お?[ \t]*預(?:[ \t]*(?:か[ \t]*)?り(?:[ \t]*金(?:[ \t]*額)?)?)?)'
    r'[ \t]*(?:合計)?[ \t]*(?:[:：]?[ \t]*(?:[¥￥]?[ \t]*[\d,]+[ \t]*円?)?)?[ \t]*$',
    re.MULTILINE,
)
_CASH_CHANGE_LABEL_RE = re.compile(
    r'^[ \t]*(?:お?[ \t]*釣(?:[ \t]*(?:り(?:[ \t]*銭)?|銭))?|おつり)'
    r'[ \t]*(?:[:：]?[ \t]*(?:[¥￥]?[ \t]*[\d,]+[ \t]*円?)?)?[ \t]*$',
    re.MULTILINE,
)
_WAON_TENDER_RE = re.compile(
    rf'^[ \t]*[（(]?WAON[ \t]*(?:支払(?:い|額)?|決済)[）)]?[ \t]*[:：]?[ \t]*'
    rf'(?P<amount>{_PAYMENT_AMOUNT_PATTERN})?[ \t]*$',
    re.IGNORECASE | re.MULTILINE,
)
_NAMED_NONCASH_TENDER_RE = re.compile(
    rf'^[ \t]*[（(]?(?:'
    rf'{_CREDIT_LABEL}(?:[ \t]*(?:{_CARD_LABEL}|支払(?:い|額)?|決済|計|お買上|ご利用額|A))?'
    rf'|信用(?:[ \t]*\d+)?|{_CARD_LABEL}|'
    rf'{_ELECTRONIC_MONEY_LABEL}|PayPay|QUICPay|(?<![A-Za-z])(?-i:iD)(?![A-Za-z])|'
    rf'Suica|nanaco|{_TRANSPORT_LABEL}(?:[ \t]*IC)?|'
    rf'(?<![A-Za-z])IC(?![A-Za-z])|VISA|Master(?:Card)?|JCB|AMEX|バーコード決済)'
    rf'[ \t]*(?:[（(][^）)\n]{{1,12}}[）)]?)?'
    rf'[ \t]*(?:支払(?:い|額)?|決済|計|お買上|ご利用額)?[）)]?[ \t]*[:：]?[ \t]*'
    rf'(?P<amount>{_PAYMENT_AMOUNT_PATTERN})?[ \t]*$',
    re.IGNORECASE,
)
_DEBIT_TENDER_RE = re.compile(
    rf'^[ \t]*[（(]?(?:デビット(?:[ \t]*カード)?|Debit(?:[ \t]+Card)?)'
    rf'[ \t]*(?:支払(?:い|額)?|決済|計)?[）)]?[ \t]*[:：]?[ \t]*'
    rf'(?P<amount>{_PAYMENT_AMOUNT_PATTERN})?[ \t]*$',
    re.IGNORECASE,
)
_BANK_PAYMENT_TENDER_RE = re.compile(
    rf'^[ \t]*[（(]?(?:銀行振込|口座振替|口座引落)'
    rf'[ \t]*(?:支払(?:い|額)?|決済|計)?[）)]?[ \t]*[:：]?[ \t]*'
    rf'(?P<amount>{_PAYMENT_AMOUNT_PATTERN})?[ \t]*$',
    re.IGNORECASE,
)
_SETTLEMENT_AMOUNT_RE = re.compile(
    r'^[ \t]*[¥￥]?[ \t]*(?P<amount>\d{1,3}(?:,\d{3})*|\d+)'
    r'(?:\.\d+)?[ \t]*(?:円|[¥￥\\])?-?[）)]?[ \t]*$'
)
_CASH_SETTLEMENT_LABEL_RE = re.compile(
    rf'^[ \t]*[（(]?(?:現金|現計)(?:[ \t]*(?:支払(?:い|額)?|決済|計))?'
    rf'(?:フリー(?:[ \t]+[A-Z0-9-]{{6,}})?)?[）)]?'
    rf'[ \t]*(?:[:：]?[ \t]*(?P<amount>{_PAYMENT_AMOUNT_PATTERN}))?[ \t]*$',
    re.IGNORECASE,
)
_PAYMENT_ADVERTISING_RE = re.compile(
    r'カード払(?:い)?で|(?:カード|クレジット).*(?:募集中|おすすめ|特典|なら)|'
    r'(?:電子マネー|カード).*(?:チャージ|入金額|残高)',
    re.IGNORECASE,
)
_NONCASH_TENDER_PATTERNS = (
    _WAON_TENDER_RE,
    _DEBIT_TENDER_RE,
    _BANK_PAYMENT_TENDER_RE,
    _NAMED_NONCASH_TENDER_RE,
)
_EXPLICIT_SETTLEMENT_AMOUNT_RE = re.compile(
    rf'^[ \t]*(?:取引|ご?利用|支払|決済|売上)(?:金)?額[ \t]*[:：]?[ \t]*'
    rf'(?P<amount>{_PAYMENT_AMOUNT_PATTERN})?[ \t]*$',
    re.IGNORECASE,
)
_TAX_SUMMARY_RE = re.compile(r'消費税|税率|税額|課税|対象額?')
_REFERENCE_LABEL_RE = re.compile(
    r'(?P<label>'
    r'取扱番号|伝票\s*(?:No\.?|番号)|レシート\s*(?:No\.?|番号)|'
    r'取引\s*(?:No\.?|番号|CD|コード)|領\s*No\.?|'
    r'領収(?:書|証)?\s*(?:No\.?|番号)|データ\s*No\.?|クレ通番|'
    r'照会番号|受付番号|参照番号'
    r')\s*[:：]?\s*',
    re.IGNORECASE,
)
_REFERENCE_VALUE_RE = re.compile(
    r'^(?P<value>(?=[A-Z0-9#/-]*\d)[A-Z0-9][A-Z0-9#/-]{1,})'
    r'(?=\s|$|[ぁ-んァ-ヶ一-龥])',
    re.IGNORECASE,
)
_REFERENCE_TRAILING_METADATA_RE = re.compile(
    r'(?:\u62c5\u5f53(?:\u8005)?\s*'
    r'(?:\d+(?:\s*/\s*[^\W\d_]{1,30})?|[^\W\d_]{1,30})|'
    r'\u8ca9\u58f2\u54e1(?:\s*'
    r'(?:\d+(?:\s*/\s*[^\W\d_]{1,30})?|[^\W\d_]{1,30}))?|'
    r'(?<=\s)\d+\s*\u70b9\u8cb7|\u70b9\u8cb7)\s*$'
)
_GENERIC_NO_REFERENCE_RE = re.compile(
    r'^\s*No\.?\s*[:：]?\s*(?P<value>\d{8,})\s*$',
    re.IGNORECASE,
)
_NONREFERENCE_VALUE_OWNER_RE = re.compile(
    r'^(?:金額|税額|対象額|残高|点数|(?=.*ポイント).*(?:金額|残高|点数|ポイント))$'
)
_FORMAL_RECEIPT_HEADER_RE = re.compile(r'^\s*領\s*収\s*(?:書|証)\s*$')


def _matches_target(amount: float | None, targets: tuple[float, ...]) -> bool:
    return amount is not None and any(abs(amount - target) <= 2 for target in targets)


def _is_financial_value_owner(line: str) -> bool:
    """Recognize a printed label that owns an adjacent numeric amount."""
    return bool(
        _TOTALS_LABEL_RE.fullmatch(line)
        or _NONREFERENCE_VALUE_OWNER_RE.fullmatch(re.sub(r'\s+', '', line))
    )


def _line_amount(line: str) -> float | None:
    amount = _settlement_amount(line)
    return amount if amount is not None else _amount_at_end(line)


def _is_tax_summary_amount(lines: list[str], idx: int) -> bool:
    if _TAX_SUMMARY_RE.search(lines[idx]):
        return True
    for preceding in reversed(lines[max(0, idx - 2):idx]):
        if preceding.strip():
            return bool(_TAX_SUMMARY_RE.search(preceding))
    return False


def _is_exact_tender_row(line: str) -> bool:
    if any(pattern.fullmatch(line) for pattern in _NONCASH_TENDER_PATTERNS):
        return True
    if _CASH_SETTLEMENT_LABEL_RE.fullmatch(line):
        return True
    return bool(
        re.search(r'現金|現計|預', line)
        and _CASH_TENDER_LABEL_RE.fullmatch(line)
    )


def _allows_backward_tender_amount(lines: list[str], idx: int) -> bool:
    line = lines[idx]
    named = _NAMED_NONCASH_TENDER_RE.fullmatch(line)
    if named:
        nearby = [line]
        for following in lines[idx + 1:]:
            if following.strip():
                nearby.append(following)
                if len(nearby) == 3:
                    break
        return bool(
            re.search(r'支払|決済|計|お買上|ご利用額', line)
            or not _PAYMENT_ADVERTISING_RE.search('\n'.join(nearby))
        )
    if any(
        pattern.fullmatch(line)
        for pattern in (_WAON_TENDER_RE, _DEBIT_TENDER_RE, _BANK_PAYMENT_TENDER_RE)
    ):
        return True
    return bool(
        _CASH_SETTLEMENT_LABEL_RE.fullmatch(line)
        and re.fullmatch(r'[（(][^）)\n]+[）)]', line.strip())
    )


def _owned_tender_amount(
    lines: list[str],
    idx: int,
    targets: tuple[float, ...],
    *,
    allow_backward: bool,
) -> float | None:
    """Own one bounded amount by exact tender structure and target arithmetic."""
    for following in lines[idx + 1:idx + 3]:
        if not following.strip():
            continue
        amount = _settlement_amount(following)
        if amount is not None:
            return amount
        break

    for amount_idx in range(idx + 1, min(len(lines), idx + 13)):
        explicit = _EXPLICIT_SETTLEMENT_AMOUNT_RE.fullmatch(lines[amount_idx])
        if explicit:
            attached = explicit.group("amount")
            if attached:
                return _amount_at_end(attached)
            for following in lines[amount_idx + 1:amount_idx + 3]:
                if not following.strip():
                    continue
                return _settlement_amount(following)
        amount = _settlement_amount(lines[amount_idx])
        if (
            targets
            and amount is not None
            and not _is_tax_summary_amount(lines, amount_idx)
            and _matches_target(amount, targets)
        ):
            return amount

    if allow_backward and targets:
        for amount_idx in range(idx - 1, max(-1, idx - 9), -1):
            if _is_exact_tender_row(lines[amount_idx]):
                break
            amount = _line_amount(lines[amount_idx])
            if amount is None:
                continue
            if _is_tax_summary_amount(lines, amount_idx):
                continue
            return amount
    return None


def _has_tender_amount(
    text: str,
    patterns: tuple[re.Pattern, ...],
    targets: tuple[float, ...] = (),
) -> bool:
    """Require an exact tender row with positive, arithmetically owned value."""
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        match = None
        for pattern in patterns:
            match = pattern.fullmatch(line)
            if match:
                break
        if not match:
            continue
        attached = match.groupdict().get("amount")
        if attached:
            amount = _amount_at_end(attached)
            if (
                amount is not None
                and amount > 0
                and (not targets or _matches_target(amount, targets))
            ):
                return True
            continue
        elif "amount" not in match.groupdict():
            amount = _amount_at_end(line)
            if amount is not None:
                if amount > 0 and (not targets or _matches_target(amount, targets)):
                    return True
                continue
        allow_backward = _allows_backward_tender_amount(lines, idx)
        amount = _owned_tender_amount(
            lines, idx, targets, allow_backward=allow_backward,
        )
        if (
            amount is not None
            and amount > 0
            and (not targets or _matches_target(amount, targets))
        ):
            return True
    return False


def _has_waon_tender_amount(text: str, targets: tuple[float, ...] = ()) -> bool:
    return _has_tender_amount(text, (_WAON_TENDER_RE,), targets)


def _has_noncash_tender_amount(text: str, targets: tuple[float, ...] = ()) -> bool:
    return _has_tender_amount(
        text,
        _NONCASH_TENDER_PATTERNS,
        targets,
    )


def _settlement_amount(line: str) -> float | None:
    match = _SETTLEMENT_AMOUNT_RE.fullmatch(line)
    if not match:
        return None
    try:
        return float(match.group("amount").replace(',', ''))
    except ValueError:
        return None


def _amount_at_end(line: str) -> float | None:
    match = re.search(
        r'[¥￥]?[ \t]*(\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?'
        r'[ \t]*(?:円|[¥￥\\])?[ \t]*$',
        line,
    )
    if not match:
        return None
    try:
        return float(match.group(1).replace(',', ''))
    except ValueError:
        return None


def _noncash_tender_evidence(
    lines: list[str], idx: int, targets: tuple[float, ...] = (),
) -> tuple[bool, float | None]:
    line = lines[idx]
    for pattern in _NONCASH_TENDER_PATTERNS:
        match = pattern.fullmatch(line)
        if not match:
            continue
        if match.group("amount"):
            amount = _amount_at_end(match.group("amount"))
            return amount is not None and amount > 0, amount
        amount = _owned_tender_amount(
            lines,
            idx,
            targets,
            allow_backward=bool(targets) and _allows_backward_tender_amount(lines, idx),
        )
        return amount is not None and amount > 0, amount

    stripped = line.strip()
    if not re.fullmatch(r'[^\d¥￥]{2,32}カード', stripped):
        return False, None
    if _CARD_NOISE_RE.search(stripped) or _PAYMENT_ADVERTISING_RE.search(stripped):
        return False, None
    nearby = '\n'.join(lines[max(0, idx - 4):min(len(lines), idx + 3)])
    return bool(re.search(r'カード残高|残高|[#*Xx]{4,}', nearby)), None


def _cash_tender_evidence(lines: list[str], idx: int) -> tuple[bool, float | None]:
    line = lines[idx]
    if not (
        _CASH_SETTLEMENT_LABEL_RE.fullmatch(line)
        or _CASH_TENDER_LABEL_RE.fullmatch(line)
    ):
        return False, None
    amount = None if "フリー" in line else _amount_at_end(line)
    if amount is not None:
        return True, amount if amount > 0 else 0.0
    for following in lines[idx + 1:idx + 3]:
        if not following.strip():
            continue
        return True, _settlement_amount(following)
    return True, None


def _column_selected_tender(
    lines: list[str], targets: tuple[float, ...],
) -> str | None:
    """Resolve adjacent tender labels when their ordered values are target then zero."""
    def _kind(line: str) -> str | None:
        if _WAON_TENDER_RE.fullmatch(line):
            return "WAON"
        if _DEBIT_TENDER_RE.fullmatch(line):
            return "debit"
        if _BANK_PAYMENT_TENDER_RE.fullmatch(line):
            return "bank_payment"
        if _NAMED_NONCASH_TENDER_RE.fullmatch(line):
            return "credit"
        if _CASH_SETTLEMENT_LABEL_RE.fullmatch(line):
            return "cash"
        return None

    for idx in range(len(lines) - 1):
        first = _kind(lines[idx])
        second = _kind(lines[idx + 1])
        if not first or not second or first == second:
            continue
        amounts = [
            amount
            for line in lines[idx + 2:min(len(lines), idx + 8)]
            if (amount := _settlement_amount(line)) is not None
        ][:2]
        if len(amounts) != 2:
            continue
        if _matches_target(amounts[0], targets) and amounts[1] == 0:
            return first
        if amounts[0] == 0 and _matches_target(amounts[1], targets):
            return second
    return None


def _settlement_kind(total, text: str, amount_paid=None) -> str | None:
    """Return one tender or mixed only when bounded settlement math closes."""
    try:
        total_value = float(total)
    except (TypeError, ValueError):
        return None
    if total_value <= 0:
        return None

    target_values = {total_value}
    try:
        paid_value = float(amount_paid)
    except (TypeError, ValueError):
        paid_value = 0
    if paid_value > 0:
        target_values.add(paid_value)
    targets = tuple(target_values)

    lines = text.splitlines()
    selected = _column_selected_tender(lines, targets)
    if selected:
        return selected
    cash_rows: dict[int, float | None] = {}
    generic_cash_rows: set[int] = set()
    for idx in range(len(lines)):
        present, amount = _cash_tender_evidence(lines, idx)
        if present:
            cash_rows[idx] = amount
            if (
                _CASH_TENDER_LABEL_RE.fullmatch(lines[idx])
                and not _CASH_SETTLEMENT_LABEL_RE.fullmatch(lines[idx])
            ):
                generic_cash_rows.add(idx)
    change_rows = [
        idx for idx, line in enumerate(lines)
        if _CASH_CHANGE_LABEL_RE.fullmatch(line)
    ]
    noncash_rows: dict[int, float | None] = {}
    for idx in range(len(lines)):
        present, amount = _noncash_tender_evidence(lines, idx, targets)
        if present:
            noncash_rows[idx] = amount

    labeled_pairs = [
        (noncash, cash)
        for noncash_idx, noncash in noncash_rows.items()
        for cash_idx, cash in cash_rows.items()
        if noncash is not None
        and cash is not None
        and noncash > 0
        and cash > 0
        and cash_idx not in generic_cash_rows
        and abs(noncash_idx - cash_idx) <= 8
    ]
    if labeled_pairs:
        if any(
            _matches_target(noncash + cash, targets)
            for noncash, cash in labeled_pairs
        ):
            return "mixed"
        return "ambiguous"
    balanced_cash = False

    for cash_idx in cash_rows:
        for change_idx in change_rows:
            if abs(cash_idx - change_idx) > 14:
                continue
            nearby_noncash = [
                idx for idx in noncash_rows
                if min(cash_idx, change_idx) - 8 <= idx <= max(cash_idx, change_idx) + 8
            ]
            label_rows = [cash_idx, change_idx, *nearby_noncash]
            start = max(0, min(label_rows) - 6)
            stop = min(len(lines), max(label_rows) + 24)
            values: list[float] = []
            for idx in range(start, stop):
                value = _settlement_amount(lines[idx])
                if value is None and (
                    idx in cash_rows or idx in change_rows or idx in noncash_rows
                ):
                    value = _amount_at_end(lines[idx])
                if value is not None:
                    values.append(value)

            if nearby_noncash:
                labeled_amounts = [
                    noncash_rows[idx]
                    for idx in nearby_noncash
                    if noncash_rows[idx] is not None
                ]
                for noncash_pos, noncash in enumerate(values):
                    if noncash <= 0:
                        continue
                    if labeled_amounts and not any(
                        abs(noncash - labeled) <= 2 for labeled in labeled_amounts
                    ):
                        continue
                    for cash_pos, cash in enumerate(values):
                        if cash_pos == noncash_pos or cash <= 0:
                            continue
                        for change_pos, change in enumerate(values):
                            if change_pos in (noncash_pos, cash_pos) or cash <= change:
                                continue
                            if _matches_target(noncash + cash - change, targets):
                                return "mixed"

            for tender_pos, tender in enumerate(values):
                if tender <= 0:
                    continue
                for change_pos, change in enumerate(values):
                    if tender_pos == change_pos or tender < change:
                        continue
                    if _matches_target(tender - change, targets):
                        balanced_cash = True

    return "cash" if balanced_cash else None


def _merchant_looks_invalid(merchant: str | None) -> bool:
    merchant = (merchant or "").strip()
    if not merchant:
        return True
    compact = re.sub(r'\s+', '', merchant)
    if re.fullmatch(r'T\d{13}', re.sub(r'[\s-]+', '', merchant.upper())):
        return True
    if re.fullmatch(
        r'(?:https?://)?(?:www\.)?(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+'
        r'[a-z]{2,}(?:/[a-z0-9_./?%&=+~-]*)?',
        merchant,
        re.IGNORECASE,
    ):
        return True
    if re.search(r'20\d{2}[./年-]\d{1,2}[./月-]\d{1,2}', merchant):
        return True
    if re.fullmatch(r'(?:TEL|電話|☎)\s*[:：]?\s*0?[\d\-\s]{6,}', merchant, flags=re.IGNORECASE):
        return True
    if (
        re.search(r'\d+丁目|\d+-\d+', merchant)
        or re.search(r'(?:都|道|府|県).{0,12}(?:市|区|町|村|郡)', merchant)
        or re.search(r'(?:市|区|町|村|郡).*\d', merchant)
    ):
        return True
    if re.match(r'^[（(]?[¥￥]?\s*[\d,]+(?:円)?[）)]?$', merchant):
        return True
    if merchant in {'領収書', '領収証', 'レシート', '様', '売上'}:
        return True
    if re.fullmatch(r'[\u5370\u6e08\u4ed8\u7a0e]', merchant):
        return True
    if re.search(r'お買い上げありがとうございます|ご利用ありがとうございます', merchant):
        return True
    if re.match(r'^(?:ご購入店|購入店|お買上店|お買い上げ店)\s*[:：]?', merchant):
        return True
    if re.fullmatch(r'[（(]?(?:消費税|内消費税|税込|税抜|課税対象|税額).*[）)]?', compact):
        return True
    if re.search(r'軽減税率|対象商品|適用商品|印は', compact):
        return True
    if re.search(r'返品交換|レシート|ご来店|未使用品|ポイント|割引券', compact):
        return True
    if re.search(r'[!！]', merchant) and not re.search(r'[A-Za-z0-9]', merchant):
        return True
    if re.search(r'税務|承認済|付につき|印紙税申告納', compact):
        return True
    if re.search(r'但し|上記.*受領|正に受領', merchant):
        return True
    return False


def _clean_merchant_candidate(text: str, *, keep_company_suffix: bool = False) -> str:
    text = re.sub(r'^\s*(?:事業者名|販売者|発行者|店舗名)\s*[:：]\s*', '', text or "").strip()
    text = re.sub(r'\s+', ' ', text).strip()
    if not keep_company_suffix:
        text = _COMPANY_SUFFIX_RE.sub('', text).strip()
    return text


def _official_authority_header_candidate(lines: list[str]) -> str | None:
    for raw_line in lines[:8]:
        line = raw_line.strip()
        if not line or re.search(r'領収|レシート|TEL|電話|FAX|登録番号', line, re.IGNORECASE):
            continue
        m = _OFFICIAL_AUTHORITY_HEADER_RE.match(line)
        if not m:
            continue
        candidate = _clean_merchant_candidate(m.group(1))
        if candidate and not _merchant_looks_invalid(candidate):
            return candidate
    return None


def _receipt_owner_header_candidate(lines: list[str], title_idx: int) -> str | None:
    """Return one logo-like owner from the header ending at a receipt title."""
    noise = re.compile(
        r'CREDIT|RECEIPT|CUSTOMER|SALE|PAYMENT|TOTAL|SUBTOTAL|VOID(?:ED)?|'
        r'ID|NO|POS|TEL|FAX|SS|WELCOME|APP|'
        r'(?:(?:CUSTOMER|MERCHANT|STORE|OFFICE|CARDHOLDER|RECEIPT)[&.\'-]?)?'
        r'(?:COPY|DUPLICATE|ORIGINAL|REPRINT)'
        r'(?:[&.\'-]?(?:COPY|\d+))*',
        re.IGNORECASE,
    )

    def clean(value: str) -> str | None:
        candidate = re.sub(r'[®™©]', '', value).strip()
        if (
            not re.fullmatch(r'[A-Za-z][A-Za-z0-9&.\'-]{2,30}', candidate)
            or noise.fullmatch(candidate)
            or _PAYMENT_TOKEN_RE.fullmatch(candidate)
            or _merchant_looks_invalid(candidate)
        ):
            return None
        return candidate

    title = lines[title_idx].strip()
    inline = re.match(
        r'^\s*(?P<owner>[A-Za-z][A-Za-z0-9&.\'-]{2,30})\s*'
        r'[（(][^）)]*(?:領\s*収\s*(?:書|証)|レシート)[^）)]*[）)]',
        title,
        re.IGNORECASE,
    )
    if inline:
        owner = clean(inline.group("owner"))
        if not owner:
            return None
        continuation = (
            clean(lines[title_idx + 1].strip())
            if title_idx + 1 < len(lines)
            else None
        )
        nearby_facility = next((
            idx
            for idx in range(title_idx + 1, min(len(lines), title_idx + 7))
            if re.search(r'(?:給油所|サービスステーション|\bSS\b)', lines[idx])
        ), None)
        if (
            continuation
            and nearby_facility is not None
            and any(
                _COMPANY_SUFFIX_RE.search(line)
                for line in lines[nearby_facility + 1:nearby_facility + 3]
            )
        ):
            return owner + continuation
        return owner

    candidates: dict[str, str] = {}
    for raw_line in lines[:title_idx]:
        candidate = clean(raw_line.strip())
        if candidate:
            candidates.setdefault(candidate.casefold(), candidate)
    return next(iter(candidates.values())) if len(candidates) == 1 else None


def _fix_company_name_merchant(extracted, unified_text):
    """Prefer venue/event name over legal company name when LLM picks the latter."""
    merchant = extracted.get("merchant")
    lines = unified_text.split('\n')
    merchant_text = (merchant or "").strip()
    authority_header = _official_authority_header_candidate(lines)
    if authority_header and merchant_text != authority_header:
        for idx, raw_line in enumerate(lines[:8]):
            line = raw_line.strip()
            if authority_header not in line:
                continue
            following = [item.strip() for item in lines[idx + 1:idx + 4]]
            if (
                not merchant_text
                or _merchant_looks_invalid(merchant_text)
                or any(merchant_text == item and _OFFICIAL_DEPARTMENT_LINE_RE.match(item)
                       for item in following)
            ):
                extracted["merchant"] = authority_header
                return
    title_idx = next((
        idx
        for idx, raw_line in enumerate(lines)
        if (
            re.search(r'領\s*収\s*(?:書|証)|レシート', raw_line)
            and not re.search(
                r'\u518d\u767a\u884c|\u756a\u53f7|No\.?|\u5834\u5408',
                raw_line,
                re.IGNORECASE,
            )
        )
    ), None)
    if title_idx is not None and len(re.sub(r'\s+', '', merchant_text)) >= 3:
        merchant_key = re.sub(r'\s+', '', merchant_text).casefold()
        header_has_merchant = any(
            merchant_key in re.sub(r'\s+', '', line).casefold()
            for line in lines[:title_idx + 1]
        )
        item_owned = any(
            re.sub(r'\s+', '', str(item.get("description") or "")).casefold().startswith(
                merchant_key
            )
            for item in extracted.get("line_items") or []
            if isinstance(item, dict)
        )
        usage_owned = bool(extracted.get("usage")) and any(
            re.sub(r'\s+', '', line).casefold().startswith(merchant_key)
            for line in lines[title_idx + 1:]
        )
        owner = _receipt_owner_header_candidate(lines, title_idx)
        if (
            owner
            and owner.casefold() != merchant_key
            and not header_has_merchant
            and (item_owned or usage_owned)
        ):
            extracted["merchant"] = owner
            return
    for facility_idx, raw_line in enumerate(lines):
        facility = raw_line.strip()
        if not re.search(r'(?:給油所|サービスステーション|\bSS\b)', facility):
            continue
        if not merchant_text or merchant_text not in facility:
            continue
        if not any(
            _COMPANY_SUFFIX_RE.search(line)
            for line in lines[facility_idx + 1:facility_idx + 3]
        ):
            continue
        brand_candidates: list[str] = []
        for preceding in lines[max(0, facility_idx - 4):facility_idx]:
            candidate = re.sub(r'[（(].*?[）)]', '', preceding).strip()
            if not re.fullmatch(r'[A-Za-z][A-Za-z .&\'-]{1,24}', candidate):
                continue
            words = re.findall(r'[A-Za-z]+', candidate)
            if any(
                re.fullmatch(r'(?:CREDIT|RECEIPT|CUSTOMER|COPY|SALE|ID|NO|POS|TEL|FAX|SS|WELCOME)', word, re.IGNORECASE)
                for word in words
            ):
                continue
            candidate = ''.join(words)
            if 3 <= len(candidate) <= 30 and not _merchant_looks_invalid(candidate):
                brand_candidates.append(candidate)
        if brand_candidates:
            candidate = brand_candidates[-1]
            if candidate.islower() and len(brand_candidates) >= 2:
                candidate = brand_candidates[-2] + candidate
            extracted["merchant"] = candidate
            return
    if merchant_text and re.search(r'店$', merchant_text):
        compact_merchant = re.sub(r'\s+', '', merchant_text)
        for raw_line in lines[:4]:
            line = raw_line.strip()
            if line == merchant_text or re.search(r'TEL|FAX|https?://|登録番号|領収', line, re.IGNORECASE):
                continue
            for candidate in re.findall(r'[ァ-ヶー]{2,}', line):
                if compact_merchant.startswith(candidate) and not _merchant_looks_invalid(candidate):
                    extracted["merchant"] = candidate
                    return
    if (
        re.fullmatch(r'[A-Z][A-Z0-9&.\'-]{2,}', merchant_text)
        and any(raw_line.strip() == merchant_text for raw_line in lines[:3])
        and any(_COMPANY_SUFFIX_RE.search(raw_line) for raw_line in lines[:8])
    ):
        for raw_line in lines[1:6]:
            line = raw_line.strip()
            if not line or re.search(r'TEL|FAX|https?://|登録番号|領収', line, re.IGNORECASE):
                continue
            brand = re.match(r'^([ァ-ヶー]{2,})(?:[ぁ-ん一-龥].*店[。．.]?|.*\s+.*店[。．.]?)$', line)
            if not brand:
                continue
            candidate = _clean_merchant_candidate(brand.group(1))
            if candidate and not _merchant_looks_invalid(candidate):
                extracted["merchant"] = candidate
                return
    if merchant_text:
        for idx, raw_line in enumerate(lines[:4]):
            line = raw_line.strip()
            if line != merchant_text:
                continue
            if re.fullmatch(r'[A-Z][A-Z0-9&.\'-]{2,8}', merchant_text):
                for offset, nearby_raw in enumerate(lines[idx + 1:idx + 4], start=1):
                    nearby = nearby_raw.strip()
                    if not nearby or re.search(r'TEL|FAX|https?://|登録番号|領収', nearby, re.IGNORECASE):
                        continue
                    if re.search(r'[ぁ-んァ-ン一-龥]', nearby):
                        if idx == 0 and any(
                            re.fullmatch(r'[A-Z][A-Z\s&.\'-]{3,}', between.strip())
                            for between in lines[idx + 1:idx + offset]
                        ):
                            return
                        if re.search(r'(?:店$|ホームセンター|スーパー|ショッピング|モール)', nearby):
                            return
                        candidate = _clean_merchant_candidate(nearby)
                        if candidate and not _merchant_looks_invalid(candidate):
                            extracted["merchant"] = candidate
                            return
            if re.fullmatch(r'[A-Z][A-Z\s&.\'-]{4,}', line):
                for prev_raw in reversed(lines[:idx]):
                    prev = prev_raw.strip()
                    if not re.fullmatch(r'[A-Z0-9&.\'-]{2,5}', prev):
                        continue
                    later_header = "\n".join(lines[idx + 1:8])
                    if prev in later_header and not _merchant_looks_invalid(prev):
                        extracted["merchant"] = prev
                        return
            if not re.fullmatch(r'[ぁ-んァ-ン一-龥ー]{2,8}', line):
                continue
            if idx + 1 >= len(lines):
                continue
            next_line = lines[idx + 1].strip()
            romanized_line = lines[idx + 2].strip() if idx + 2 < len(lines) else ""
            if (
                next_line
                and romanized_line
                and re.search(r'[ぁ-んァ-ン一-龥]', next_line)
                and re.search(r'[A-Za-z]', romanized_line)
                and not re.search(r'TEL|FAX|https?://|登録番号|領収', next_line, re.IGNORECASE)
            ):
                candidate = _clean_merchant_candidate(next_line)
                if candidate and not _merchant_looks_invalid(candidate):
                    extracted["merchant"] = candidate
                    return
    for raw_line in lines[:_ASCII_BRAND_HEADER_SCAN_LIMIT]:
        line = raw_line.strip()
        if not line:
            continue
        # Store-in-store receipts can start with an ASCII brand followed by the
        # host store/location. If the LLM chose the host store, prefer the
        # leading brand token visible in the header.
        m = re.match(r'^([A-Z][A-Z0-9&.\'-]{2,})\s+(.+)$', line)
        if not m or not merchant:
            continue
        suffix = m.group(2)
        merchant_compact = re.sub(r'\s+', '', str(merchant))
        location_compact = re.sub(r'\s+', '', str(extracted.get("location") or ""))
        suffix_compact = re.sub(r'\s+', '', suffix)
        is_exact_location = (
            merchant_compact == suffix_compact
            and location_compact == suffix_compact
            and _is_ascii_brand_location_suffix(suffix)
        )
        is_host_store = (
            merchant in suffix
            and re.search(r'[ぁ-んァ-ン一-龥]', suffix)
            and re.search(r'(?:店|モール|センター)$', suffix)
        )
        if is_exact_location or is_host_store:
            extracted["merchant"] = m.group(1)
            return
    if (
        merchant
        and re.search(r'[ぁ-んァ-ン一-龥]', merchant)
        and not _merchant_looks_invalid(merchant)
        and re.search(r'(?:店$|ホームセンター|スーパー|ショッピング|モール)', merchant)
    ):
        merchant_visible_in_header = any(
            merchant in raw_line for raw_line in lines[:6]
        )
        if merchant_visible_in_header:
            for raw_line in lines[:5]:
                line = raw_line.strip()
                if (
                    re.fullmatch(r'[A-Z][A-Z0-9&.\'-]{3,}', line)
                    and not _merchant_looks_invalid(line)
                ):
                    extracted["merchant"] = line
                    return
    if _merchant_looks_invalid(merchant):
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            app_brand = re.search(r'([ァ-ヶー]{3,})\s*アプリ', line)
            if app_brand:
                candidate = _clean_merchant_candidate(app_brand.group(1))
                if candidate and not _merchant_looks_invalid(candidate):
                    extracted["merchant"] = candidate
                    return
            business = re.search(r'(?:事業者名|販売者|発行者|店舗名)\s*[:：]\s*(.+)$', line)
            if business:
                candidate = _clean_merchant_candidate(business.group(1), keep_company_suffix=True)
                if candidate and not _merchant_looks_invalid(candidate):
                    extracted["merchant"] = candidate
                    return
        for raw_line in lines[:8]:
            line = raw_line.strip()
            header_phone = _HEADER_PHONE_MERCHANT_RE.match(line)
            if not header_phone:
                continue
            candidate = _clean_merchant_candidate(header_phone.group("merchant"))
            if candidate in {"TEL", "Tel", "電話", "お問い合わせ"} or _merchant_looks_invalid(candidate):
                continue
            if candidate:
                extracted["merchant"] = candidate
                return
        for raw_line in lines[:4]:
            line = raw_line.strip()
            if not line or re.search(r'TEL|電話|FAX|登録番号|領収|返品|ご購入店|営業時間|https?://', line, re.IGNORECASE):
                continue
            candidate = re.sub(r'[®™©]', '', line).strip()
            candidate = re.sub(r'\s+', ' ', candidate)
            if (
                re.fullmatch(r'[A-Z][A-Z0-9 &.\'-]{2,30}', candidate)
                and not _merchant_looks_invalid(candidate)
            ):
                extracted["merchant"] = candidate
                return
        for raw_line in lines[:8]:
            line = raw_line.strip()
            if not line:
                continue
            if re.search(r'TEL|電話|FAX|登録番号|領収|返品|ご購入店|営業時間', line, re.IGNORECASE):
                continue
            if re.fullmatch(r'.{1,8}(?:名|番号)', line):
                continue
            if not re.search(r'[ぁ-んァ-ン一-龥]', line):
                continue
            candidate = _clean_merchant_candidate(line)
            if candidate and not _merchant_looks_invalid(candidate):
                extracted["merchant"] = candidate
                return
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            if _COMPANY_SUFFIX_RE.search(line):
                candidate = _clean_merchant_candidate(line)
                if candidate and not _merchant_looks_invalid(candidate):
                    extracted["merchant"] = candidate
                    return
        return
    if not merchant:
        return
    found_in_any_line = False
    found_only_in_company_line = True
    for line in lines:
        if merchant in line:
            found_in_any_line = True
            if not _COMPANY_SUFFIX_RE.search(line):
                found_only_in_company_line = False
                break
    if not found_in_any_line or not found_only_in_company_line:
        return
    for line_idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if merchant in line:
            keep_suffix = bool(re.search(r'事業者名\s*[:：]', line))
            if _COMPANY_SUFFIX_RE.search(line):
                for prev in reversed(lines[max(0, line_idx - 3):line_idx]):
                    prev = prev.strip()
                    ascii_logo = bool(
                        re.fullmatch(r'[A-Za-z][A-Za-z0-9&.\'-]{2,20}', prev)
                        and not re.fullmatch(
                            r'(?:ID|NO|POS|TEL|FAX|SS|ATC|AID|IC|ETC)',
                            prev,
                            re.IGNORECASE,
                        )
                    )
                    if (prev and not _DECORATIVE_RE.match(prev)
                            and not _COMPANY_SUFFIX_RE.search(prev)
                            and not _merchant_looks_invalid(prev)
                            and (re.search(r'[ぁ-んァ-ン一-龥]', prev) or ascii_logo)):
                        extracted["merchant"] = prev
                        return
            candidate = _clean_merchant_candidate(line, keep_company_suffix=keep_suffix)
            if candidate and not _merchant_looks_invalid(candidate):
                extracted["merchant"] = candidate
                return
    for line in lines:
        line = line.strip()
        if not line or _DECORATIVE_RE.match(line) or _COMPANY_SUFFIX_RE.search(line):
            continue
        if line != merchant and len(line) >= 2 and not _merchant_looks_invalid(line):
            extracted["merchant"] = line
            break


def _apply_financial_overrides(extracted, ocr_totals, ocr_conf, llm_conf):
    """Override LLM financial values (total, subtotal, taxes) with OCR-extracted values."""
    ocr_total_val = ocr_totals.get("total")
    if "subtotal" in ocr_totals:
        ocr_sub_val = ocr_totals["subtotal"]
        if ocr_total_val and ocr_sub_val < ocr_total_val * 0.5:
            candidates = ocr_totals.get("_subtotal_candidates", [])
            best_sub = None
            if candidates and ocr_total_val:
                plausible = [v for v in candidates
                             if ocr_total_val * 0.5 <= v <= ocr_total_val]
                if plausible:
                    best_sub = min(plausible)
            if best_sub:
                ocr_totals["subtotal"] = best_sub
                ocr_sub_val = best_sub
            else:
                alt_sub = ocr_totals.get("_subtotal_alt")
                if alt_sub and alt_sub >= ocr_total_val * 0.5:
                    ocr_totals["subtotal"] = alt_sub
                    ocr_sub_val = alt_sub
                else:
                    del ocr_totals["subtotal"]
        if "subtotal" in ocr_totals and should_override_field("subtotal", ocr_conf, llm_conf):
            extracted["subtotal"] = ocr_sub_val
        elif extracted.get("subtotal") is None:
            extracted["subtotal"] = ocr_sub_val
    if "total" in ocr_totals and should_override_field("total", ocr_conf, llm_conf):
        ocr_total = float(ocr_totals["total"])
        ocr_first = float(ocr_totals["total_first"]) if ocr_totals.get("total_first") is not None else None
        ocr_sub = float(ocr_totals["subtotal"]) if ocr_totals.get("subtotal") is not None else None
        if ocr_sub and ocr_total < ocr_sub:
            pass
        elif ocr_sub and ocr_total > ocr_sub * 2:
            if ocr_first and ocr_first <= ocr_sub * 1.15:
                extracted["total"] = ocr_first
        else:
            extracted["total"] = ocr_total
    elif "total" in ocr_totals and extracted.get("total") is None:
        extracted["total"] = float(ocr_totals["total"])
    if "subtotal" in ocr_totals and "total" in ocr_totals:
        computed_tax = ocr_totals["total"] - ocr_totals["subtotal"]
        if computed_tax >= 0 and should_override_field("taxes", ocr_conf, llm_conf):
            llm_tax = _sum_taxable_amounts(extracted.get("taxes", []))
            ocr_tax = _sum_taxable_amounts(ocr_totals.get("taxes") or [])
            if not (ocr_tax > 0 and abs(ocr_tax - computed_tax) > 5) and abs(llm_tax - computed_tax) > 5:
                if extracted.get("taxes"):
                    if llm_tax > 0:
                        scale = computed_tax / llm_tax
                        for t in extracted["taxes"]:
                            if isinstance(t, dict) and t.get("rate") != "0%":
                                t["amount"] = round(t["amount"] * scale)
                    else:
                        extracted["taxes"] = [{"rate": "unknown", "label": None, "amount": computed_tax}]
                elif computed_tax > 0:
                    extracted["taxes"] = [{"rate": "unknown", "label": None, "amount": computed_tax}]
    if ocr_totals.get("taxes") and should_override_field("taxes", ocr_conf, llm_conf):
        # Merge: trust OCR for rates it found, but keep LLM's tax entries for
        # rates the OCR scan missed (column-split layouts often hide one rate's
        # tax line from the OCR forward-scan while the LLM still recovers it).
        existing_total = extracted.get("total")
        existing_subtotal = extracted.get("subtotal")
        existing_tax_sum = _sum_taxable_amounts(extracted.get("taxes") or [])
        ocr_tax_sum = _sum_taxable_amounts(ocr_totals.get("taxes") or [])
        try:
            existing_total_f = float(existing_total) if existing_total is not None else None
            existing_subtotal_f = float(existing_subtotal) if existing_subtotal is not None else None
        except (TypeError, ValueError):
            existing_total_f = existing_subtotal_f = None
        if (
            existing_total_f is not None
            and existing_subtotal_f is not None
            and existing_tax_sum > 0
            and abs(existing_subtotal_f + existing_tax_sum - existing_total_f) <= 2
            and ocr_tax_sum > 0
            and abs(existing_subtotal_f + ocr_tax_sum - existing_total_f) > 5
        ):
            return
        ocr_rates = {t.get("rate") for t in ocr_totals["taxes"]}
        llm_extra = [
            t for t in (extracted.get("taxes") or [])
            if isinstance(t, dict) and t.get("rate") and t.get("rate") not in ocr_rates
        ]
        extracted["taxes"] = list(ocr_totals["taxes"]) + llm_extra

    # Fix per-rate subtotal: when subtotal + tax != total, recompute from total - tax
    if "subtotal" in ocr_totals and "total" in ocr_totals and ocr_totals.get("taxes"):
        ocr_tax_sum = _sum_taxable_amounts(ocr_totals["taxes"])
        ocr_sub = ocr_totals["subtotal"]
        ocr_tot = ocr_totals["total"]
        if ocr_tax_sum > 0 and abs(ocr_sub + ocr_tax_sum - ocr_tot) > 2:
            computed_sub = ocr_tot - ocr_tax_sum
            if abs(computed_sub + ocr_tax_sum - ocr_tot) < 2:
                extracted["subtotal"] = computed_sub
                ocr_totals["subtotal"] = computed_sub


def _fix_date(extracted, unified_text):
    """Extract and fix dates from OCR text (supports 令和/平成 eras)."""
    def _coerce_modern_ocr_year(year: int) -> int:
        if 2000 <= year <= 2009:
            return year + 20
        if 2010 <= year <= 2019:
            return year + 10
        return year

    def _set_date(year: int, month: int, day: int) -> bool:
        year = _coerce_modern_ocr_year(year)
        if not 2020 <= year <= 2030:
            return False
        extracted["date"] = f"{year:04d}-{month:02d}-{day:02d}"
        return True

    labeled_date_fragment_patterns = [
        r'(20\d{2})\s*年\s*0?(\d{1,2})\s*月\s*0?(\d{1,2})\s*日',
        r'(?<!\d)(\d{2})\s*年\s*0?(\d{1,2})\s*月\s*0?(\d{1,2})\s*日',
        r'(20\d{2})/\s*(\d{1,2})/\s*(\d{1,2})',
        r'(20\d{2})-(\d{1,2})-(\d{1,2})',
    ]
    lines = [line.strip() for line in (unified_text or "").splitlines()]
    for idx, line in enumerate(lines):
        if not re.search(r'日付|ご利用日|お取扱日|取扱日|カードお取扱日', line):
            continue
        windows = [line, "\n".join(lines[idx + 1:min(idx + 3, len(lines))])]
        for window in windows:
            for pattern in labeled_date_fragment_patterns:
                m = re.search(pattern, window)
                if not m:
                    continue
                year = int(m.group(1))
                if year < 100:
                    year += 2000
                if _set_date(year, int(m.group(2)), int(m.group(3))):
                    return

    labeled_patterns = [
        r'(?:日付|ご利用日|お取扱日|取扱日|カードお取扱日)\s*[:：]?\s*(20\d{2})\s*年\s*0?(\d{1,2})\s*月\s*0?(\d{1,2})\s*日',
        r'(?:日付|ご利用日|お取扱日|取扱日|カードお取扱日)\s*[:：]?\s*(\d{2})\s*年\s*0?(\d{1,2})\s*月\s*0?(\d{1,2})\s*日',
    ]
    for pattern in labeled_patterns:
        m = re.search(pattern, unified_text)
        if m:
            year = int(m.group(1))
            if year < 100:
                year += 2000
            if _set_date(year, int(m.group(2)), int(m.group(3))):
                return

    western_patterns = [
        r'(20\d{2})\s*年\s*0?(\d{1,2})\s*月\s*0?(\d{1,2})\s*日',
        r'(20\d{2})/\s*(\d{1,2})/\s*(\d{1,2})',
        r'(20\d{2})-(\d{1,2})-(\d{1,2})',
    ]
    for pattern in western_patterns:
        for western in re.finditer(pattern, unified_text):
            context = unified_text[max(0, western.start() - 16):min(len(unified_text), western.end() + 16)]
            if re.search(r'有効期限|期限|失効|満了', context):
                continue
            if _set_date(int(western.group(1)), int(western.group(2)), int(western.group(3))):
                return

    era_named = re.search(r'(令和|平成)\s*(\d{1,2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日', unified_text)
    if era_named:
        w_year = era_to_western_year(int(era_named.group(2)), era_named.group(1))
        if w_year:
            extracted["date"] = f"{w_year:04d}-{int(era_named.group(3)):02d}-{int(era_named.group(4)):02d}"
        return

    era = re.search(r'(?<!\d)(\d{1,2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日', unified_text)
    if era:
        era_name = None
        for name in ERA_TABLE:
            if name in unified_text:
                era_name = name
                break
        era_year_val = int(era.group(1))
        # 2-digit western year abbreviation (e.g. "26年" = 2026)
        # Prefer this when no era name is present and 20XX is plausible
        if not era_name and 20 <= era_year_val <= 99:
            western_candidate = 2000 + era_year_val
            if 2020 <= western_candidate <= 2030:
                extracted["date"] = f"{western_candidate:04d}-{int(era.group(2)):02d}-{int(era.group(3)):02d}"
                return
        w_year = era_to_western_year(era_year_val, era_name)
        if w_year and 1989 <= w_year <= 2100:
            extracted["date"] = f"{w_year:04d}-{int(era.group(2)):02d}-{int(era.group(3)):02d}"


_DATE_LINE_RE = re.compile(
    r'(?:'
    r'20\d{2}\s*[年/-]\s*0?\d{1,2}\s*[月/-]\s*0?\d{1,2}\s*日?'
    r'|'
    r'(?:令和|平成)?\s*\d{1,2}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日'
    r')'
)
_TIME_HHMM_RE = re.compile(r'(?<!\d)([0-2]?\d)\s*[:：]\s*([0-5]\d)(?:\s*[:：]\s*[0-5]\d)?(?!\d)')
_TIME_JP_RE = re.compile(r'(?<!\d)([0-2]?\d)\s*時\s*([0-5]\d)\s*分?')
_BUSINESS_HOURS_RE = re.compile(r'営業時間|営業中|定休|OPEN|CLOSE|TEL|電話|☎')


def _parse_time_from_segment(segment: str) -> str | None:
    """Find the first valid HH:MM (or HH時MM分) in a text segment, or None."""
    for pattern in (_TIME_HHMM_RE, _TIME_JP_RE):
        for m in pattern.finditer(segment):
            hh, mm = int(m.group(1)), int(m.group(2))
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                return f"{hh:02d}:{mm:02d}"
    return None


def _fix_time(extracted, unified_text):
    """Extract receipt transaction time from OCR text, anchored to the date line.

    Scans the date line + next 2 lines for HH:MM or HH時MM分.
    Skips lines that look like business hours / phone numbers to avoid false positives.

    Sets extracted['time'] only when:
      - The LLM didn't already produce a valid time, OR
      - The OCR-anchored time disagrees with the LLM and we have strong evidence.
    Leaves extracted['time'] as None if no time appears on the receipt.
    """
    lines = unified_text.split('\n')

    candidate: str | None = None
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not _DATE_LINE_RE.search(line):
            continue
        # Skip the part of the date line that contains the date itself,
        # so we don't match digits inside e.g. "2026年03月04日" or "12月03日"
        date_match = _DATE_LINE_RE.search(line)
        tail = line[date_match.end():] if date_match else line

        # Look in the date-line tail and the next two lines (but not into
        # business-hours context).
        segments = [tail]
        for j in range(i + 1, min(i + 3, len(lines))):
            nxt = lines[j].strip()
            if _BUSINESS_HOURS_RE.search(nxt):
                break
            segments.append(nxt)

        for seg in segments:
            if _BUSINESS_HOURS_RE.search(seg):
                continue
            t = _parse_time_from_segment(seg)
            if t:
                candidate = t
                break
        if candidate:
            break

    if candidate is None:
        # Fallback: ISO date already in extracted['date'] but OCR may have it
        # joined together as "2025/12/23/13:49" — pull time off the join.
        joined = re.search(r'20\d{2}[/-]\d{1,2}[/-]\d{1,2}[/\s-]+(\d{1,2})[:：](\d{2})', unified_text)
        if joined:
            hh, mm = int(joined.group(1)), int(joined.group(2))
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                candidate = f"{hh:02d}:{mm:02d}"

    # Fallback: search entire OCR for HH時MM分 without digit lookbehind
    if candidate is None:
        matches = list(re.finditer(r'(\d{1,2})時(\d{2})分', unified_text))
        if len(matches) == 1:
            hh, mm = int(matches[0].group(1)), int(matches[0].group(2))
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                candidate = f"{hh:02d}:{mm:02d}"

    # Fallback for receipt footers such as "14:10-2109-01" when the
    # transaction date is printed far above the card slip footer.
    if candidate is None:
        matches = list(_TIME_HHMM_RE.finditer(unified_text))
        if len(matches) == 1:
            hh, mm = int(matches[0].group(1)), int(matches[0].group(2))
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                candidate = f"{hh:02d}:{mm:02d}"

    existing = extracted.get("time")
    if not existing:
        if candidate:
            extracted["time"] = candidate
        return

    normalized_existing = _parse_time_from_segment(str(existing))
    if normalized_existing and normalized_existing != existing:
        extracted["time"] = normalized_existing
        existing = normalized_existing

    if candidate and candidate != existing:
        extracted["time"] = candidate


def _fix_payment_method(extracted, unified_text, ocr_conf, llm_conf):
    """Set payment method only from an exact tender or balanced settlement."""
    targets: list[float] = []
    for raw in (extracted.get("total"), extracted.get("amount_paid")):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value > 0 and value not in targets:
            targets.append(value)
    target_values = tuple(targets)

    settlement_kind = _settlement_kind(
        extracted.get("total"), unified_text, extracted.get("amount_paid"),
    )
    if settlement_kind in {"mixed", "ambiguous"}:
        extracted["payment_method"] = None
        return
    if settlement_kind in {"cash", "credit", "debit", "bank_payment", "WAON"}:
        extracted["payment_method"] = settlement_kind
        return

    tender_methods = [
        method
        for method, patterns in (
            ("WAON", (_WAON_TENDER_RE,)),
            ("debit", (_DEBIT_TENDER_RE,)),
            ("bank_payment", (_BANK_PAYMENT_TENDER_RE,)),
            ("credit", (_NAMED_NONCASH_TENDER_RE,)),
        )
        if _has_tender_amount(unified_text, patterns, target_values)
    ]
    has_strong_cash = _has_tender_amount(
        unified_text, (_CASH_SETTLEMENT_LABEL_RE,), target_values,
    )
    has_generic_cash = _has_tender_amount(
        unified_text, (_CASH_TENDER_LABEL_RE,), target_values,
    )
    if len(tender_methods) > 1 or (tender_methods and has_strong_cash):
        extracted["payment_method"] = None
        return
    if tender_methods:
        extracted["payment_method"] = tender_methods[0]
    elif has_strong_cash or has_generic_cash:
        extracted["payment_method"] = "cash"
    else:
        extracted["payment_method"] = None


def _fix_total_from_stacked_cash_tender_block(extracted, unified_text):
    """Fix value-stacked cash blocks: total, tendered cash, change."""
    lines = unified_text.split('\n')

    def _loose_amount(line: str) -> float | None:
        m = re.fullmatch(
            r'[¥￥]?\s*(\d{1,3}(?:,\d{3})*|\d{1,5})\s*(?:円|[¥￥\\])?\s*',
            line.strip(),
        )
        if not m:
            return None
        try:
            return float(m.group(1).replace(',', ''))
        except ValueError:
            return None

    for idx, raw in enumerate(lines):
        line = raw.strip()
        if not re.search(r'総\s*合\s*計|総合計|お会計', line) and not re.fullmatch(r'合\s*計', line):
            continue
        if re.search(r'税|対象|点数', line):
            continue
        short_end = min(len(lines), idx + 10)
        window = '\n'.join(lines[idx:short_end])
        has_tender = bool(
            re.search(r'現金|現計', window) or _CASH_TENDER_LABEL_RE.search(window)
        )
        has_change = bool(_CASH_CHANGE_LABEL_RE.search(window))
        if not (has_tender and has_change):
            end = min(len(lines), idx + 18)
            wide_window = '\n'.join(lines[idx:end])
            if (
                _has_noncash_tender_amount(wide_window)
                or not _CASH_CHANGE_LABEL_RE.search(wide_window)
            ):
                continue
            try:
                previous_total = float(extracted.get("total"))
            except (TypeError, ValueError):
                continue
            for cash_idx in range(idx + 1, end):
                cash_match = _CASH_SETTLEMENT_LABEL_RE.fullmatch(
                    lines[cash_idx].strip()
                )
                if not cash_match:
                    continue
                tendered = _loose_amount(cash_match.group("amount") or "")
                if tendered is None:
                    for following in lines[cash_idx + 1:cash_idx + 3]:
                        if following.strip():
                            tendered = _loose_amount(following)
                            break
                if tendered is None or tendered <= 0 or abs(previous_total - tendered) > 2:
                    continue
                for change_idx in range(cash_idx + 1, end):
                    if not _CASH_CHANGE_LABEL_RE.fullmatch(lines[change_idx].strip()):
                        continue
                    change = _amount_at_end(lines[change_idx])
                    if change is None:
                        for following in lines[change_idx + 1:min(end, change_idx + 3)]:
                            if not following.strip():
                                continue
                            change = _loose_amount(following)
                            if change is not None:
                                break
                            if not _CASH_SETTLEMENT_LABEL_RE.fullmatch(following.strip()):
                                break
                    if change is None or change < 0 or tendered <= change:
                        continue
                    inferred_total = tendered - change
                    printed_totals = [
                        printed
                        for j in range(idx + 1, cash_idx)
                        if (printed := _loose_amount(lines[j])) is not None
                    ]
                    if (
                        not printed_totals
                        or abs(printed_totals[-1] - inferred_total) > 2
                    ):
                        continue
                    extracted["total"] = inferred_total
                    points_used = float(extracted.get("points_used") or 0)
                    extracted["amount_paid"] = max(0.0, inferred_total - points_used)
                    return
            continue

        has_noncash_tender_amount = _has_noncash_tender_amount(window)
        amounts: list[float] = []
        for following in lines[idx + 1:min(len(lines), idx + 18)]:
            stripped = following.strip()
            if not stripped:
                continue
            value = _loose_amount(stripped)
            if value is not None:
                amounts.append(value)
                continue
            if amounts and re.search(
                r'To\s*Go|登録番号|発行日|https?://|TEL|電話|詳しくはこちら',
                stripped,
                re.IGNORECASE,
            ):
                break

        triple_match: tuple[float, float, float] | None = None
        for total, tendered, change in zip(amounts, amounts[1:], amounts[2:]):
            if total <= 0 or tendered <= 0 or change < 0:
                continue
            if tendered <= total:
                continue
            if abs((tendered - change) - total) > 2:
                continue
            triple_match = (total, tendered, change)
        if triple_match is not None and not has_noncash_tender_amount:
            total, tendered, change = triple_match
            previous_total = extracted.get("total")
            extracted["total"] = total
            points_used = float(extracted.get("points_used") or 0)
            amount_paid = float(extracted.get("amount_paid") or 0)
            if (
                extracted.get("amount_paid") is None
                or previous_total is None
                or abs(amount_paid - float(previous_total or 0)) <= 2
                or abs(amount_paid - tendered) <= 2
                or amount_paid > total
            ):
                extracted["amount_paid"] = max(0.0, total - points_used)
            return

        pair_match: tuple[float, float] | None = None
        for tendered, change in zip(amounts, amounts[1:]):
            if tendered <= 0 or change < 0 or tendered <= change:
                continue
            inferred_total = tendered - change
            if inferred_total <= 0:
                continue
            previous_total = extracted.get("total")
            amount_paid = float(extracted.get("amount_paid") or 0)
            if previous_total is not None and abs(float(previous_total) - tendered) > 2:
                continue
            pair_match = (tendered, change)
        if pair_match is not None and not has_noncash_tender_amount:
            tendered, change = pair_match
            inferred_total = tendered - change
            previous_total = extracted.get("total")
            amount_paid = float(extracted.get("amount_paid") or 0)
            extracted["total"] = inferred_total
            points_used = float(extracted.get("points_used") or 0)
            if (
                extracted.get("amount_paid") is None
                or previous_total is None
                or abs(amount_paid - float(previous_total or 0)) <= 2
                or abs(amount_paid - tendered) <= 2
                or amount_paid > inferred_total
            ):
                extracted["amount_paid"] = max(0.0, inferred_total - points_used)
            return


def _fix_unlabeled_cash_tender_change_block(extracted, unified_text):
    """Recover total/tender when cash labels are missing but change reconciles."""
    if (
        _has_noncash_tender_amount(unified_text)
        or _settlement_kind(extracted.get("total"), unified_text) in {"mixed", "ambiguous"}
    ):
        return
    lines = [line.strip() for line in unified_text.split('\n')]

    def _amount(line: str) -> float | None:
        m = re.fullmatch(r'[¥￥]?\s*(\d{1,3}(?:,\d{3})*|\d{1,6})\s*', line)
        if not m:
            return None
        try:
            return float(m.group(1).replace(',', ''))
        except ValueError:
            return None

    for idx, line in enumerate(lines):
        if not re.fullmatch(r'合\s*計', line):
            continue
        if idx > 0 and re.search(r'税|対象|小\s*計', lines[idx - 1]):
            continue

        values: list[tuple[int, float]] = []
        change_idx = None
        for j in range(idx + 1, min(len(lines), idx + 8)):
            if _CASH_CHANGE_LABEL_RE.search(lines[j]):
                change_idx = j
                continue
            value = _amount(lines[j])
            if value is not None:
                values.append((j, value))
                continue
            if values and re.search(r'お買上点数|ポイント|伝票番号|レシート', lines[j]):
                break
        if change_idx is None:
            continue
        label_window = '\n'.join(lines[idx:change_idx + 1])
        if re.search(r'現金|現計', label_window) or _CASH_TENDER_LABEL_RE.search(label_window):
            continue

        before_change = [(j, value) for j, value in values if j < change_idx]
        after_change = [(j, value) for j, value in values if j > change_idx]
        if len(before_change) < 2 or not after_change:
            continue
        total = before_change[0][1]
        tendered = before_change[1][1]
        change = after_change[0][1]
        if total <= 0 or tendered <= total or change < 0:
            continue
        if abs((tendered - change) - total) > 2:
            continue

        extracted["total"] = total
        points_used = float(extracted.get("points_used") or 0)
        extracted["amount_paid"] = max(0.0, total - points_used)
        extracted["payment_method"] = "cash"
        return


def _fix_payment_reference(extracted, unified_text):
    """Prefer one printed document reference, then one secondary reference."""
    lines = unified_text.splitlines()
    primary: list[tuple[str, set[str], int]] = []
    secondary: list[tuple[str, set[str], int]] = []

    generic_no_matches = [
        (idx, match.group("value"))
        for idx, line in enumerate(lines)
        if (match := _GENERIC_NO_REFERENCE_RE.fullmatch(line))
    ]
    generic_no_values = {value for _idx, value in generic_no_matches}
    if len(generic_no_values) == 1:
        value = next(iter(generic_no_values))
        occurrences = len(generic_no_matches) + sum(
            line.strip() == value for line in lines
        )
        if occurrences >= 2:
            secondary.append(("No", {value}, generic_no_matches[0][0]))

    for idx, line in enumerate(lines):
        label_matches = list(_REFERENCE_LABEL_RE.finditer(line))
        for label_idx, match in enumerate(label_matches):
            compact_label = re.sub(r'\s+', '', match.group("label"))
            if (
                compact_label.startswith(("レシート", "伝票"))
                and re.search(r'\S*ポイント(?:カード)?\s*$', line[:match.start()], re.IGNORECASE)
            ):
                continue

            body_end = (
                label_matches[label_idx + 1].start()
                if label_idx + 1 < len(label_matches)
                else len(line)
            )
            body = line[match.end():body_end].strip()
            trailing_metadata = _REFERENCE_TRAILING_METADATA_RE.search(body)
            if trailing_metadata:
                body = body[:trailing_metadata.start()].rstrip()
            if (
                not body
                and 0 < idx < len(lines) - 1
                and all(
                    "ポイント" in lines[adjacent_idx]
                    for adjacent_idx in (idx - 1, idx + 1)
                )
            ):
                continue
            value_match = _REFERENCE_VALUE_RE.fullmatch(
                re.sub(r'[ \t]+', '', body)
            )
            values = {value_match.group("value")} if value_match else set()
            if not body:
                for adjacent_idx in (idx - 1, idx + 1):
                    if 0 <= adjacent_idx < len(lines):
                        if (
                            adjacent_idx < idx
                            and idx > 1
                            and (
                                re.search(
                                    r'(?:番号|No\.?|TID)\s*[:：]?\s*$',
                                    lines[idx - 2],
                                    re.IGNORECASE,
                                )
                                or (
                                    lines[idx - 2].strip()
                                    and _SETTLEMENT_AMOUNT_RE.fullmatch(
                                        lines[adjacent_idx].strip()
                                    )
                                    and (
                                        _is_financial_value_owner(lines[idx - 2])
                                        or not re.fullmatch(
                                            r'\d{8,}', lines[adjacent_idx].strip()
                                        )
                                    )
                                    and not _REFERENCE_LABEL_RE.search(lines[idx - 2])
                                )
                            )
                        ):
                            continue
                        adjacent = _REFERENCE_VALUE_RE.fullmatch(lines[adjacent_idx].strip())
                        if adjacent:
                            values.add(adjacent.group("value"))

            target = (
                primary
                if compact_label.startswith(("レシート", "伝票", "領"))
                else secondary
            )
            target.append((compact_label, values, idx))

    has_receipt_number = any(
        label.startswith(("レシート", "領")) and len(values) == 1
        for label, values, _idx in primary
    )
    if has_receipt_number:
        primary = [
            entry
            for entry in primary
            if not (
                entry[0].startswith("伝票")
                and re.search(
                    r'カード|会員番号|端末番号|取扱|取引内容|承認番号|'
                    r'決済.*番号|TID|AID|ATC|合計金額',
                    '\n'.join(lines[max(0, entry[2] - 4):entry[2] + 5]),
                    re.IGNORECASE,
                )
                and (
                    len(entry[1]) != 1
                    or len(next(iter(entry[1]))) <= 2
                )
            )
        ]

    chosen = primary or secondary
    candidates = [entry[1] for entry in chosen]
    values = set().union(*candidates) if candidates else set()
    extracted["payment_reference"] = (
        next(iter(values))
        if len(values) == 1 and all(len(candidate) == 1 for candidate in candidates)
        else None
    )


def _fix_receipt_payer(extracted, unified_text):
    """Keep payer only when one explicit receipt addressee owns the field."""
    lines = [line.strip() for line in unified_text.splitlines() if line.strip()]
    split_addressees = [
        f"{lines[idx - 1]} {line}"
        for idx, line in enumerate(lines)
        if (
            idx >= 2
            and line in {"様", "御中"}
            and _FORMAL_RECEIPT_HEADER_RE.fullmatch(lines[idx - 2])
            and re.match(r'^[A-Za-zぁ-んァ-ヶ一-鿿]', lines[idx - 1])
        )
    ]
    payers = _payer_candidates(
        "\n".join((unified_text, *split_addressees))
    )
    extracted["payer"] = next(iter(payers.values())) if len(payers) == 1 else None
