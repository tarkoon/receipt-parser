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
from .receipt_totals import _sum_taxable_amounts


def _merchant_looks_invalid(merchant: str | None) -> bool:
    merchant = (merchant or "").strip()
    if not merchant:
        return True
    compact = re.sub(r'\s+', '', merchant)
    if re.fullmatch(r'T\d{13}', re.sub(r'[\s-]+', '', merchant.upper())):
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
    if merchant in {'領収書', '領収証', 'レシート', '様'}:
        return True
    if re.match(r'^(?:ご購入店|購入店|お買上店|お買い上げ店)\s*[:：]?', merchant):
        return True
    if re.fullmatch(r'[（(]?(?:消費税|内消費税|税込|税抜|課税対象|税額).*[）)]?', compact):
        return True
    if re.search(r'軽減税率|対象商品|適用商品|印は', compact):
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
    for raw_line in lines[:5]:
        line = raw_line.strip()
        if not line:
            continue
        # Store-in-store receipts can start with an ASCII brand followed by the
        # host store/location. If the LLM chose the host store, prefer the
        # leading brand token visible in the header.
        m = re.match(r'^([A-Z][A-Z0-9&.\'-]{2,})\s+(.+)$', line)
        if (
            m
            and merchant
            and merchant in m.group(2)
            and re.search(r'[ぁ-んァ-ン一-龥]', m.group(2))
        ):
            extracted["merchant"] = m.group(1)
            return
    if merchant and re.search(r'[ぁ-んァ-ン一-龥]', merchant):
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
                    if (prev and not _DECORATIVE_RE.match(prev)
                            and not _COMPANY_SUFFIX_RE.search(prev)
                            and not _merchant_looks_invalid(prev)
                            and re.search(r'[ぁ-んァ-ン一-龥]', prev)):
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
            if abs(llm_tax - computed_tax) > 5:
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
    """Detect cash payment from OCR evidence (tendered amount, change, etc.)."""
    existing = extracted.get("payment_method")
    if existing in (
        "credit_card", "card", "QUICPay", "iD", "Suica", "PayPay",
        "電子マネー", "electronic_money",
    ):
        extracted["payment_method"] = "credit"
        existing = "credit"
    has_explicit_credit = bool(re.search(
        r'クレジット|カード|VISA|Master(?:Card)?|JCB|AMEX|QUICPay|電子マネー|iD|PayPay|交通系|IC',
        unified_text,
        re.IGNORECASE,
    ))
    has_cash_tender_evidence = bool(re.search(
        r'現金|現計|(?:お預り金?|お預かり)(?!票)|(?<![お\w])預\s*[¥￥]',
        unified_text,
    ))
    if existing == "cash" and has_explicit_credit and not has_cash_tender_evidence:
        extracted["payment_method"] = "credit"
        existing = "credit"
    if not existing and has_explicit_credit:
        extracted["payment_method"] = "credit"
        existing = "credit"

    has_cash = '現計' in unified_text
    if not has_cash:
        oazukari = re.search(r'お預り金?\s*[¥￥]?\s*([\d,]+)', unified_text)
        if not oazukari:
            oazukari = re.search(r'お預り金?\s*\n[¥￥]\s*([\d,]+)', unified_text)
        if not oazukari:
            oazukari = re.search(r'(?<![お\w])預\s*[¥￥]\s*([\d,]+)', unified_text)
        if oazukari:
            has_cash = True
    if not has_cash and re.search(r'お釣り|お釣銭|釣銭|(?<![お\w])釣\s*[¥￥]', unified_text):
        has_cash = True
    if not has_cash and '現金' in unified_text:
        has_cash = True
    change_m = re.search(r'(?:お釣り|お釣銭|釣銭|おつり|釣\s*[¥￥])\s*[¥￥]?\s*([\d,]+)', unified_text)
    change_amount = float(change_m.group(1).replace(',', '')) if change_m else -1
    has_tender = bool(re.search(
        r'(?:お預り金?|お預かり)(?!票)|(?<![お\w])預\s*[¥￥]',
        unified_text,
    ))
    has_change_label = bool(re.search(r'釣', unified_text))
    has_cash_keyword = bool(re.search(r'現金\s*[¥￥]\s*[\d,]+', unified_text))
    if not has_cash_keyword and re.search(r'(?:^|\n)\s*現金\s*(?:\n|$)', unified_text):
        has_cash_keyword = True
    strong_cash = has_cash and (
        (has_tender and has_change_label and change_amount != 0) or has_cash_keyword
    )

    if has_cash:
        if strong_cash:
            extracted["payment_method"] = "cash"
        elif not existing or existing == "cash":
            extracted["payment_method"] = "cash"
        elif should_override_field("payment_method", ocr_conf, llm_conf) and not existing:
            extracted["payment_method"] = "cash"
    elif extracted.get("payment_method") == "cash":
        # Hallucinated cash with no OCR evidence — strip it. The original
        # guard only fires for printed receipts (has 小計/合計) and missed
        # handwritten 領収証 where the LLM defaults to "cash" because of
        # the receipt-header phrasing (上記正に領収いたしました).
        extracted["payment_method"] = None


def _fix_total_from_stacked_cash_tender_block(extracted, unified_text):
    """Fix value-stacked cash blocks: total, tendered cash, change."""
    lines = unified_text.split('\n')

    def _loose_amount(line: str) -> float | None:
        m = re.fullmatch(r'[¥￥]?\s*(\d{1,3}(?:,\d{3})*|\d{1,5})\s*', line.strip())
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
        window = '\n'.join(lines[idx:min(len(lines), idx + 10)])
        if not re.search(r'現金|お預り|お預かり|預り', window):
            continue
        if not re.search(r'お釣り|お釣銭|釣銭|おつり|釣\s*$', window):
            continue

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
        if triple_match is not None:
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
        if pair_match is not None:
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
            if re.search(r'お釣り|お釣銭|釣銭|おつり', lines[j]):
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
        if re.search(r'現金|お預り|お預かり|預り|(?<![お\w])預(?:\s|$|[¥￥])', label_window):
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
        extracted["amount_paid"] = tendered
        extracted["payment_method"] = "cash"
        return


def _fix_toll_payment_reference(extracted, unified_text):
    """Recover toll-road handling/reference numbers printed outside item rows."""
    if extracted.get("payment_reference"):
        return
    if not re.search(r'料金所|高速道路|ETC|NEXCO', unified_text):
        return
    m = re.search(r'取扱番号\s*[:：]?\s*([0-9][0-9-]{5,})', unified_text)
    if m:
        extracted["payment_reference"] = m.group(1)
