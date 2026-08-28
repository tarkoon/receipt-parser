"""pipeline_bill.py — Utility bill-specific post-processing.

Extracted from pipeline.py for maintainability.
"""

import re
from datetime import date, timedelta

from .patterns import ERA_TABLE, UTILITY_BILL_KEYWORDS
from .pipeline_slip import _PAYER_LABEL_RE, _clean_payer_candidate
from .receipt_identity_payment import _fix_payment_reference


def _date_fragment(prefix: str) -> str:
    return (
        rf'(?:(?P<{prefix}_era>令和|平成|昭和)?\s*'
        rf'(?P<{prefix}_year>\d{{1,4}})\s*年\s*)?'
        rf'(?P<{prefix}_month>\d{{1,2}})\s*月\s*'
        rf'(?P<{prefix}_day>\d{{1,2}})'
        rf'(?:\s+(?P={prefix}_day))?\s*日'
    )


_EXPLICIT_PERIOD_RE = re.compile(
    r'(?:ご?使用期間|料金算定期間|対象期間|請求期間).{0,60}?'
    + _date_fragment("start")
    + r'\s*[-‐‑‒–—~〜～]\s*'
    + _date_fragment("end"),
    re.DOTALL,
)
_BILL_MONTH_RE = re.compile(
    r'(?:(令和|平成|昭和)\s*)?(\d{1,4})\s*年\s*\d{1,2}\s*月分'
)
_PROVIDER_OR_ISSUER_RE = re.compile(
    r'会社|組合|局|水道|ガス|電力|登録番号|'
    r'発行(?:元|者)|請求(?:元|者)|供給(?:元|者)|事業者'
)
_RECIPIENT_LINE_RE = re.compile(r'(?:様|御中)\s*$|受取人|支払人')
_UTILITY_NON_PAYER_RE = re.compile(
    r'責(?:任者)?|担当者?|従業員|社員|店員|係員|レジ|キャッシャ'
)
_STANDALONE_HONORIFIC_RE = re.compile(r'^(?:様|御中)$')
_METER_DATE_RE = re.compile(
    r'(?<!\d)(?P<year>\d{4})\s*年\s*'
    r'(?P<month>\d{1,2})\s*月\s*(?P<day>\d{1,2})\s*日'
)
_METER_BLOCK_END_RE = re.compile(
    r'交換(?:時|後)指針|ご使用量|ご利用明細|請求|納期限'
)


def _period_year(era: str | None, value: str | None) -> int | None:
    if not value:
        return None
    year = int(value)
    if era:
        return ERA_TABLE[era] + year
    return year if year >= 1000 else None


def _explicit_billing_period(text: str, extracted: dict) -> dict | None:
    """Return one printed period; printed endpoints are never advanced."""
    anchor = None
    bill_month = _BILL_MONTH_RE.search(text)
    if bill_month:
        anchor = _period_year(bill_month.group(1), bill_month.group(2))
    if anchor is None:
        for value in (
            extracted.get("date"),
            (extracted.get("billing_period") or {}).get("end")
            if isinstance(extracted.get("billing_period"), dict) else None,
        ):
            if isinstance(value, str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
                anchor = int(value[:4])
                break

    candidates = set()
    for match in _EXPLICIT_PERIOD_RE.finditer(text):
        start_month = int(match.group("start_month"))
        end_month = int(match.group("end_month"))
        start_year = _period_year(match.group("start_era"), match.group("start_year"))
        end_year = _period_year(match.group("end_era"), match.group("end_year"))
        if start_year is None and end_year is None:
            end_year = anchor
            start_year = anchor - (start_month > end_month) if anchor else None
        elif start_year is None:
            start_year = end_year - (start_month > end_month)
        elif end_year is None:
            end_year = start_year + (end_month < start_month)
        if start_year is None or end_year is None:
            continue
        try:
            start = date(start_year, start_month, int(match.group("start_day"))).isoformat()
            end = date(end_year, end_month, int(match.group("end_day"))).isoformat()
        except ValueError:
            continue
        candidates.add((start, end))

    if len(candidates) != 1:
        return None
    start, end = next(iter(candidates))
    return {"start": start, "end": end}


def _merchant_has_provider_evidence(merchant: object, text: str) -> bool:
    """Require the extracted name and provider/issuer evidence on one line."""
    if not isinstance(merchant, str):
        return False
    compact_merchant = re.sub(r'[\s　:：・.,，。()（）「」『』]', '', merchant).casefold()
    if not compact_merchant:
        return False
    for line in text.splitlines():
        compact_line = re.sub(r'[\s　:：・.,，。()（）「」『』]', '', line).casefold()
        if (
            compact_merchant in compact_line
            and not _RECIPIENT_LINE_RE.search(line)
            and _PROVIDER_OR_ISSUER_RE.search(line)
        ):
            return True
    return False


def _utility_payer_candidates(text: str) -> tuple[dict[str, str], bool]:
    """Return honorific-backed utility addressees and whether one was printed."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidates: dict[str, str] = {}
    saw_addressee = False

    def add(value: str, context: str = "") -> None:
        label = _PAYER_LABEL_RE.search(value)
        if label:
            value = value[label.end():].strip(" \t:：")
        if (
            not value
            or _UTILITY_NON_PAYER_RE.search(context or value)
            or UTILITY_BILL_KEYWORDS.search(value)
        ):
            return
        candidate = _clean_payer_candidate(value)
        if candidate:
            key = re.sub(r'\s+', '', candidate).casefold()
            candidates.setdefault(key, candidate)

    for index, line in enumerate(lines):
        if not _STANDALONE_HONORIFIC_RE.fullmatch(line):
            if re.search(r'(?:様|御中)\s*$', line):
                saw_addressee = True
                add(line, line)
            continue

        saw_addressee = True
        candidate_index = index - 1
        if (
            candidate_index >= 0
            and UTILITY_BILL_KEYWORDS.search(lines[candidate_index])
        ):
            candidate_index -= 1
        if candidate_index < 0:
            continue
        context = " ".join(lines[max(0, candidate_index - 1):candidate_index + 1])
        add(lines[candidate_index], context)

    return candidates, saw_addressee


def _meter_reading_pairs(text: str) -> set[tuple[date, date]]:
    """Find unique bounded (previous, current) meter-reading date pairs."""
    lines = text.splitlines()
    pairs: set[tuple[date, date]] = set()
    current_dates: set[date] = set()
    previous_dates: set[date] = set()

    def parsed_dates(value: str) -> list[date]:
        dates = []
        for match in _METER_DATE_RE.finditer(value):
            try:
                dates.append(date(
                    int(match.group("year")),
                    int(match.group("month")),
                    int(match.group("day")),
                ))
            except ValueError:
                continue
        return dates

    for line in lines:
        label = re.search(
            r'(?:(?P<current>今回)|(?P<previous>前回))\s*検\s*針\s*日',
            line,
        )
        if label:
            dates = parsed_dates(line[label.end():])
            if len(dates) == 1:
                target = current_dates if label.group("current") else previous_dates
                target.add(dates[0])
    if len(current_dates) == len(previous_dates) == 1:
        current = next(iter(current_dates))
        previous = next(iter(previous_dates))
        if current > previous:
            pairs.add((previous, current))

    for index, line in enumerate(lines):
        marker = re.search(r'検\s*針\s*日', line)
        if not marker or re.search(r'(?:今回|前回)\s*検\s*針\s*日', line):
            continue
        block_dates = parsed_dates(line[marker.end():])
        for following in lines[index + 1:index + 9]:
            if _METER_BLOCK_END_RE.search(following):
                break
            block_dates.extend(parsed_dates(following))
        if len(block_dates) == 2 and block_dates[0] > block_dates[1]:
            pairs.add((block_dates[1], block_dates[0]))

    return pairs


def _advance_matching_meter_period(extracted: dict, text: str) -> None:
    """Advance only a unique printed prior reading that exactly owns start."""
    period = extracted.get("billing_period")
    if not isinstance(period, dict):
        return
    try:
        start = date.fromisoformat(period.get("start", ""))
        end = date.fromisoformat(period.get("end", ""))
    except (TypeError, ValueError):
        return
    pairs = _meter_reading_pairs(text)
    if len(pairs) != 1:
        return
    previous, current = next(iter(pairs))
    advanced = previous + timedelta(days=1)
    if start == previous and end == current and advanced < current:
        period["start"] = advanced.isoformat()


def postprocess_utility_bill(
    extracted: dict,
    unified_text: str,
    payment_reference_text: str | None = None,
) -> dict:
    """Apply utility bill-specific post-processing to the LLM extraction."""
    _fix_payment_reference(extracted, payment_reference_text or unified_text)

    if (
        extracted.get("merchant") is not None
        and not _merchant_has_provider_evidence(extracted["merchant"], unified_text)
    ):
        extracted["merchant"] = None

    # Trigger: a utility-only addressee label/honorific. Invariant: exactly one
    # distinct, non-employee printed identity owns payer; ambiguity clears it.
    payers, saw_addressee = _utility_payer_candidates(unified_text)
    if saw_addressee:
        extracted["payer"] = next(iter(payers.values())) if len(payers) == 1 else None

    # Check for convenience store payment evidence (overrides bank_payment)
    paid_at_store = bool(re.search(
        r'ローソン|セブン|ファミリーマート|コンビニ|収納代行|領収.*いたしました',
        unified_text,
    ))
    if paid_at_store:
        extracted["payment_method"] = "cash"
    elif re.search(r'口座引落|口座振替|振替させて', unified_text):
        extracted["payment_method"] = "bank_payment"
    elif re.search(r'領入済|収納済', unified_text):
        extracted["payment_method"] = "cash"

    # Service type: bills with both 水道 and 下水道 are water bills
    if extracted.get("service_type") == "sewage" and re.search(r'水道', unified_text):
        water_hits = len(re.findall(r'水道', unified_text))
        sewage_hits = len(re.findall(r'下水道', unified_text))
        if water_hits > sewage_hits:
            extracted["service_type"] = "water"

    # Date: prefer 領収日付 stamp date (often formatted as 'YY.M.D)
    ryoshu_date = re.search(r"領収日付[:\s]*'?(\d{2})\.(\d{1,2})\.(\d{1,2})", unified_text)
    if not ryoshu_date:
        ryoshu_date = re.search(r"'(\d{2})\.(\d{1,2})\.(\d{1,2})", unified_text)
    if ryoshu_date:
        y = int(ryoshu_date.group(1))
        year = 2000 + y if y < 100 else y
        extracted["date"] = f"{year:04d}-{int(ryoshu_date.group(2)):02d}-{int(ryoshu_date.group(3)):02d}"

    # Trigger: a labeled, explicit usage range. Invariant: its printed start
    # wins verbatim; previous-reading + 1 remains only an extraction fallback.
    explicit_period = _explicit_billing_period(unified_text, extracted)
    if explicit_period:
        extracted["billing_period"] = explicit_period
    elif not _EXPLICIT_PERIOD_RE.search(unified_text):
        _advance_matching_meter_period(extracted, unified_text)

    return extracted
