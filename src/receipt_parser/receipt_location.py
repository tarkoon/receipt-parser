"""Receipt location repair and resolution helpers."""

import logging
import re

from .patterns import ADMIN_SUFFIX_RE, LOCATION_CLUE_RE, _COMPANY_SUFFIX_RE


logger = logging.getLogger(__name__)

_ASCII_BRAND_HEADER_SCAN_LIMIT = 16
_LOCATION_TOKEN_TRAILING_PUNCTUATION = " \t\u3000。、，,．.!！:：;；"
_ASCII_BRAND_LOCATION_NOISE_RE = re.compile(
    r'領収(?:書|証)?|レシート|請求書|納品書|'
    r'税|課税|'
    r'支払|決済|現金|会計|クレジット|電子マネー|売上(?:票)?|お客様控|'
    r'商品(?:名)?|加盟店(?:名)?|店舗(?:名)?|登録番号|営業時間|合計|小計|購入点数|'
    r'(?:ご購入店|購入店)|お買上店|お買い上げ店|ご来[店]|店[名]|'
    r'[カ][ス][タ][マ][ー](?:[サ][ポ][ー][ト])?(?:[セ][ン][タ][ー])?|'
    r'[サ][ポ][ー][ト](?:[セ][ン][タ][ー])?|[コ][ー][ル][セ][ン][タ][ー]|'
    r'[お][問][い]?[合][わ]?[せ](?:[窓][口])?'
)
_HEADER_LOCATION_SUFFIX_RE = re.compile(
    r'(?:(?:[シ][ョ][ッ][プ]|支店|[本]店|営業所|[出][張][所]|料金所|店|IC)'
    r'|(?P<generic>モール|センター|[館]))$',
    re.IGNORECASE,
)
_HEADER_LOCATION_NOISE_RE = re.compile(
    r'^\s*HQ(?=\s|[本]|$)|[本][社]|[本]店[所][在][地]|[印][紙]税|税[務](?:[署])?|'
    r'[承][認][済]?|[付][に][つ][き]|[申][告][納]|'
    r'[カ][ス][タ][マ][ー](?:[サ][ポ][ー][ト])?(?:[セ][ン][タ][ー])?|'
    r'[サ][ポ][ー][ト](?:[セ][ン][タ][ー])?|[コ][ー][ル][セ][ン][タ][ー]|'
    r'[お][問][い]?[合][わ]?[せ](?:[窓][口])?|'
    r'^\s*[本][店](?=\s|TEL|[電][話]|☎|$)|'
    r'^\s*(?:(?:[担][当][者]?|[責][任][者]?|[係][員]?|[ス][タ][ッ][フ])'
    r'\s*(?:No\.?|Ｎｏ\.?)?|[ス](?:No\.?|Ｎｏ\.?))'
    r'\s*[:：]?[A-Za-z0-9-]*[ぁ-んァ-ン一-龥ー]{2,}\s*$'
)


def _strip_location_token_punctuation(value: str) -> str:
    """Strip suffix-only punctuation from an otherwise exact location token."""
    return str(value or "").rstrip(_LOCATION_TOKEN_TRAILING_PUNCTUATION)


def _location_is_replaceable_header_noise(value: str, ocr_text: str) -> bool:
    """Return true only when OCR identifies a metadata/HQ/stamp source."""
    location = re.sub(r'\s+', '', value or "")
    if not location:
        return False
    if _ASCII_BRAND_LOCATION_NOISE_RE.fullmatch(location):
        return True
    matching_lines = [
        line for line in (ocr_text or "").splitlines()
        if location in re.sub(r'\s+', '', line)
    ]
    return bool(matching_lines) and all(
        _HEADER_LOCATION_NOISE_RE.search(line) for line in matching_lines
    )


def _is_ascii_brand_location_suffix(value: str) -> bool:
    """Accept only compact Japanese location text, not receipt header noise."""
    suffix = _strip_location_token_punctuation(re.sub(r'\s+', '', value or ""))
    return bool(
        2 <= len(suffix) <= 10
        and re.fullmatch(r'[ぁ-んァ-ン一-龥ー]+', suffix)
        and not ADMIN_SUFFIX_RE.fullmatch(suffix)
        and not _COMPANY_SUFFIX_RE.search(suffix)
        and not _ASCII_BRAND_LOCATION_NOISE_RE.search(suffix)
    )


def _location_needs_resolution(location: str | None, ocr_text: str = "") -> bool:
    """Check if the location needs resolution via a focused LLM call."""
    if location and ADMIN_SUFFIX_RE.search(location):
        # Force resolution if the location comes from a corporate HQ address line
        # AND there's a facility indicator (toll gate, branch office) suggesting
        # the transaction occurred elsewhere
        loc_norm = re.sub(r'\s+', '', location)
        is_hq_address = False
        for line in ocr_text.split('\n'):
            line_norm = re.sub(r'\s+', '', line)
            if (loc_norm in line_norm
                    and re.search(r'\d+-\d+', line_norm)
                    and re.match(r'.+[都道府県]', line_norm)):
                is_hq_address = True
                break
        if is_hq_address:
            has_facility = bool(re.search(
                r'(?:料金所|営業所|支店|出張所)\s*\n?\s*[\u3000-\u9fff]{2,}', ocr_text))
            return has_facility
        return False
    if location:
        loc_norm = re.sub(r'\s+', '', location)
        if (
            _is_ascii_brand_location_suffix(loc_norm)
            and any(
                re.fullmatch(
                    rf"[A-Z][A-Z0-9&.'-]{{2,}}{re.escape(loc_norm)}",
                    re.sub(r'\s+', '', line).upper(),
                )
                for line in ocr_text.splitlines()[:_ASCII_BRAND_HEADER_SCAN_LIMIT]
            )
        ):
            return False
        return True
    if ocr_text and LOCATION_CLUE_RE.search(ocr_text):
        return True
    return False


def _location_has_ocr_evidence(location: str, ocr_text: str) -> bool:
    """Require every claimed administrative component to have OCR evidence.

    Normalizes whitespace before comparison since Japanese OCR frequently
    inserts spaces between characters.
    """
    if not location or not ocr_text:
        return False
    # Normalize whitespace in both strings for comparison
    loc_norm = _strip_location_token_punctuation(re.sub(r'\s+', '', location))
    ocr_norm = re.sub(r'\s+', '', ocr_text)
    if _location_is_replaceable_header_noise(loc_norm, ocr_text):
        return False
    if loc_norm in ocr_norm:
        return True
    components = []
    tail = loc_norm
    while match := re.match(
        r'([ぁ-んァ-ン一-龥ー]{1,10}?)[都道府県市区町村郡]',
        tail,
    ):
        components.append(match.group(0))
        tail = tail[match.end():]
    return bool(components) and all(
        component in ocr_norm for component in components
    ) and (not tail or tail in ocr_norm)


_PURCHASE_STORE_METADATA_RE = re.compile(
    r'^\s*(?:ご購入店|購入店|お買上店|お買い上げ店)'
    r'(?=[\s:：])\s*[:：]?\s*'
    r'(?P<store>.+店)[\s。、，,．.!！:：;；]*$'
)


def _trim_purchase_store_metadata_location(extracted: dict, ocr_text: str) -> None:
    """Normalize labeled purchase-store evidence without adopting its host brand."""
    location = re.sub(r'\s+', '', extracted.get("location") or "")
    if not location:
        return
    if location in re.sub(r'\s+', '', ocr_text):
        return

    metadata_stores = [
        _strip_location_token_punctuation(
            re.sub(r'\s+', '', metadata.group("store"))
        )
        for raw_line in ocr_text.splitlines()
        if (metadata := _PURCHASE_STORE_METADATA_RE.match(raw_line.strip()))
    ]
    base_match = re.match(r'(?P<base>.*?[市区町村])(?P<tail>.+)$', location)
    if base_match and len(base_match.group("tail")) >= 2:
        for store_line in metadata_stores:
            if base_match.group("tail") in store_line:
                location = base_match.group("base")
                extracted["location"] = location
                break

    if bare_city := re.fullmatch(
        r'(?P<core>[ぁ-んァ-ン一-龥ー]{1,10})市', location
    ):
        printed_store = f'{bare_city.group("core")}店'
        if any(store.endswith(printed_store) for store in metadata_stores):
            extracted["location"] = printed_store


def _recover_header_branch_store_location(extracted: dict, ocr_text: str) -> None:
    """Recover a visible branch/store token from the receipt header."""
    if not ocr_text:
        return
    current_location = re.sub(r'\s+', '', str(extracted.get("location") or ""))
    was_replaceable_noise = _location_is_replaceable_header_noise(current_location, ocr_text)
    if was_replaceable_noise:
        current_location = ""
    can_override_admin_fragment = _is_broad_japanese_admin_location(current_location)
    header_lines = ocr_text.splitlines()[:16]
    deferred_generic_candidate = None

    merchant = re.sub(r'\s+', '', str(extracted.get("merchant") or ""))

    for raw_line in header_lines:
        line = raw_line.strip()
        if not line:
            continue
        line_before_contact = re.split(
            r'(?:TEL|電話|☎)', line, maxsplit=1, flags=re.IGNORECASE
        )[0].strip()
        contact_backed = line_before_contact != line
        line_for_branch = _strip_location_token_punctuation(line_before_contact)
        if re.search(r'www\.|https?://|登録番号|領収|レシート|合計|小計|支払', line, re.IGNORECASE):
            continue
        if _HEADER_LOCATION_NOISE_RE.search(line):
            continue
        if re.search(r'(?:19|20)\d{2}[年/-]\d{1,2}[月/-]\d{1,2}', line):
            break
        if _PURCHASE_STORE_METADATA_RE.match(line_for_branch):
            continue
        if not re.search(r'[ぁ-んァ-ン一-龥]', line_for_branch):
            continue
        parts = [
            _strip_location_token_punctuation(part)
            for part in re.split(r'\s+', line_for_branch)
            if part.strip()
        ]
        candidates = [
            part for part in parts
            if re.search(r'[ぁ-んァ-ン一-龥]', part)
            and _HEADER_LOCATION_SUFFIX_RE.search(part)
        ]
        if not candidates and _HEADER_LOCATION_SUFFIX_RE.search(line_for_branch):
            candidates = [line_for_branch]
        if not candidates:
            continue
        candidate = candidates[-1]
        candidate_compact = _strip_location_token_punctuation(
            re.sub(r'\s+', '', candidate)
        )
        if (
            candidate_compact == merchant
            or _ASCII_BRAND_LOCATION_NOISE_RE.search(candidate_compact)
        ):
            continue
        if merchant and candidate_compact.startswith(merchant):
            candidate_compact = candidate_compact[len(merchant):]
        stem = _HEADER_LOCATION_SUFFIX_RE.sub('', candidate_compact)
        if current_location and not can_override_admin_fragment:
            continue
        if can_override_admin_fragment and not _branch_extends_admin_fragment(current_location, candidate_compact):
            continue
        if (
            len(stem) >= (1 if contact_backed else 2)
            and len(candidate_compact) <= 30
        ):
            if (
                not contact_backed
                and (
                    suffix_match := _HEADER_LOCATION_SUFFIX_RE.search(
                        candidate_compact
                    )
                )
                and suffix_match.lastgroup == "generic"
            ):
                deferred_generic_candidate = (
                    deferred_generic_candidate or candidate_compact
                )
                continue
            extracted["location"] = candidate_compact
            return

    if deferred_generic_candidate:
        extracted["location"] = deferred_generic_candidate
        return

    if current_location:
        return

    merchant_tokens = re.findall(
        r'[A-Za-z0-9&.\'-]{2,}|[ぁ-んー]{2,}|[ァ-ンー]{2,}|[一-龥]{2,}',
        str(extracted.get("merchant") or ""),
    )
    for phone_idx, raw_line in enumerate(header_lines):
        if not re.search(r'TEL|電話|☎|[（(]\s*0\d{1,4}\s*[）)]|^0\d{1,4}[-\s]', raw_line, re.IGNORECASE):
            continue
        context = re.sub(
            r'\s+', '',
            "".join(header_lines[max(0, phone_idx - 3):phone_idx + 1]),
        )
        if merchant_tokens and not any(
            re.sub(r'\s+', '', token) in context for token in merchant_tokens
        ):
            continue
        for neighbor in range(phone_idx - 1, max(-1, phone_idx - 4), -1):
            candidate = _strip_location_token_punctuation(
                re.sub(r'\s+', '', header_lines[neighbor].strip())
            )
            if not candidate:
                continue
            if candidate == merchant:
                continue
            if candidate in merchant:
                remaining_merchant = merchant.replace(candidate, "", 1)
                if not (
                    re.fullmatch(r'[一-龥]{2,}', candidate)
                    and re.search(r'[A-Za-zぁ-んァ-ンー]', remaining_merchant)
                ):
                    continue
            if merchant and candidate.startswith(merchant):
                candidate = candidate[len(merchant):]
            if not _is_ascii_brand_location_suffix(candidate):
                continue
            trailing_place = re.search(r'([一-龥]{2,})$', candidate)
            if trailing_place:
                extracted["location"] = trailing_place.group(1)
                return

    if was_replaceable_noise:
        extracted.pop("location", None)


def _recover_ascii_brand_header_location(extracted: dict, ocr_text: str) -> None:
    """Recover a compact suffix when an early ASCII header prefix repeats."""
    if not ocr_text:
        return
    header_lines = ocr_text.splitlines()[:_ASCII_BRAND_HEADER_SCAN_LIMIT]
    merchant = re.sub(r'\s+', '', str(extracted.get("merchant") or "")).upper()
    brands = {
        compact.upper()
        for raw_line in header_lines
        if re.fullmatch(
            r"[A-Z][A-Z0-9&.'-]{2,}",
            compact := re.sub(r'\s+', '', raw_line.strip()),
        )
    }
    if merchant_prefix := re.match(r"[A-Z][A-Z0-9&.'-]{2,}", merchant):
        brands.add(merchant_prefix.group(0))
    if not brands:
        return
    current = re.sub(r'\s+', '', str(extracted.get("location") or ""))
    if (
        current
        and not _is_broad_japanese_admin_location(current)
        and not _location_is_replaceable_header_noise(current, ocr_text)
    ):
        return
    for raw_line in header_lines:
        compact = _strip_location_token_punctuation(
            re.sub(r'\s+', '', raw_line.strip())
        )
        for brand in sorted(brands, key=len, reverse=True):
            if not compact.upper().startswith(brand):
                continue
            suffix = compact[len(brand):]
            if _is_ascii_brand_location_suffix(suffix):
                extracted["location"] = suffix
                return


def _is_broad_japanese_admin_location(value: str) -> bool:
    """Return true for city/ward/prefecture fragments that are not store names."""
    location = re.sub(r'\s+', '', str(value or ""))
    if not location or "店" in location or re.search(r'\d', location):
        return False
    admin_unit = r'[一-龥]{1,10}(?:都|道|府|県|市|区|町|村)'
    return bool(re.fullmatch(rf'(?:{admin_unit}){{1,4}}', location))


def _branch_extends_admin_fragment(location: str, branch: str) -> bool:
    location = re.sub(r'\s+', '', str(location or ""))
    stem = _strip_location_token_punctuation(
        re.sub(r'\s+', '', str(branch or ""))
    )
    stem = _HEADER_LOCATION_SUFFIX_RE.sub('', stem)
    admin_roots = re.findall(r'([一-龥]{1,10}?)(?:都|道|府|県|市|区|町|村)', location)
    if not admin_roots:
        return False
    root = admin_roots[-1]
    return (
        root in stem
        and len(stem) <= 20
        and len(stem) - len(root) >= 1
    )


def _extract_japanese_phone_hint(ocr_text: str) -> str:
    phone_match = re.search(r'(?:TEL|電話|☎)\s*[:\s]?\s*(0\d{1,4}[-\s]?\d{1,4}[-\s]?\d{2,4})', ocr_text)
    if phone_match:
        return phone_match.group(1)
    phone_match = re.search(r'^(0\d{1,4}-\d{1,4}-\d{2,4})\s*$', ocr_text, re.MULTILINE)
    if phone_match:
        return phone_match.group(1)
    paren_phone = re.search(
        r'[（(]\s*(0\d{1,4})\s*[）)]\s*(\d{1,4})[-\s]?(\d{2,4})',
        ocr_text,
    )
    if paren_phone:
        return "-".join(paren_phone.groups())
    return ""


def _normalize_noisy_city_location(extracted: dict, ocr_text: str) -> None:
    location = re.sub(r'\s+', '', str(extracted.get("location") or ""))
    if not location:
        return
    match = re.match(r'(?P<base>.*?[市区町村])(?P<tail>.+)$', location)
    if not match:
        return
    base = match.group("base")
    tail = match.group("tail")
    compact_ocr = re.sub(r'\s+', '', ocr_text or "")
    if location in compact_ocr:
        return
    if tail and f"{tail}店" in compact_ocr:
        visible_branch = next(
            (
                f"{tail}店"
                for line in (ocr_text or "").splitlines()
                if re.fullmatch(
                    rf"\s*{re.escape(tail)}店(?:\s+(?:TEL|電話|☎).*)?\s*",
                    line,
                    re.IGNORECASE,
                )
            ),
            None,
        )
        if visible_branch:
            extracted["location"] = visible_branch
            return
        candidate = dict(extracted)
        candidate["location"] = base
        _recover_header_branch_store_location(candidate, ocr_text)
        recovered = candidate.get("location")
        if recovered and recovered != base:
            extracted["location"] = recovered
            return
        extracted["location"] = base
        return
    if re.match(r'^(?:ご来|ご利用|ありが|担当|No|レジ)', tail):
        extracted["location"] = base


def _resolve_location(extracted: dict, ocr_text: str, model: str) -> tuple[str | None, str | None]:
    """Use a focused LLM call to resolve a partial location to city/ward level.

    Returns (resolved_location, warning_or_none). The warning is set when
    resolution was attempted but failed, so the caller can log it.
    """
    from .llm import _llm_chat, sanitize_llm_response

    merchant = extracted.get("merchant") or ""
    merchant_norm = re.sub(r'\s+', '', str(merchant))
    raw_location = extracted.get("location") or ""

    phone_hint = _extract_japanese_phone_hint(ocr_text)

    # Extract branch/store name (e.g., "赤間店" → "赤間", "八幡店" → "八幡")
    branch_match = None
    for raw_line in ocr_text.splitlines():
        line = raw_line.strip()
        if _PURCHASE_STORE_METADATA_RE.match(line):
            continue
        branch_match = re.search(r'([\u3000-\u9fff]{2,})\s*店', line)
        if branch_match:
            printed_branch = re.sub(r'\s+', '', f"{branch_match.group(1)}店")
            if printed_branch == merchant_norm:
                branch_match = None
                continue
            break
    branch_hint = branch_match.group(1) if branch_match else ""
    facility_match = re.search(
        r'(?:料金所|営業所|支店|出張所)'
        r'(?:[ \t　]+|\r?\n[ \t　]*)([ぁ-んァ-ン一-龥ー]{2,20})',
        ocr_text,
    )
    facility_hint = facility_match.group(1) if facility_match else ""

    # Also extract short standalone Japanese text from the first few lines
    # (often branch/location names like "赤間" above the brand name)
    _FINANCIAL_KEYWORDS = {
        '合計', '小計', '税込', '税抜', '総額', '釣銭', '預金', '現計', '点数',
        '領収証', '領収書', 'ドラッグストア', 'ドラックストア',
    }
    header_lines = []
    for line in ocr_text.split('\n')[:8]:
        s = line.strip()
        if (
            s
            and 2 <= len(s) <= 8
            and re.match(r'^[\u3000-\u9fff]+$', s)
            and s not in _FINANCIAL_KEYWORDS
            and re.sub(r'\s+', '', s) != merchant_norm
        ):
            header_lines.append(s)
    header_hint = ", ".join(header_lines) if header_lines else ""

    addr_lines = []
    for line in ocr_text.split('\n'):
        if re.search(r'[都道府県市区町村郡]|〒\d{3}', line):
            addr_lines.append(line.strip())

    if not any((raw_location, branch_hint, facility_hint, addr_lines, header_lines)):
        return None, "Location resolution: could not determine city/ward from available clues"

    clues = [f"- Merchant/brand: {merchant}"]
    if branch_hint:
        clues.append(f"- Branch/store name: {branch_hint}店 (the branch name often indicates the neighborhood)")
    if header_hint and header_hint != branch_hint:
        clues.append(f"- Receipt header text: {header_hint} (may contain location or branch name)")
    clues.append(f"- Current location value: {raw_location or 'unknown'}")
    clues.append(f"- Phone number: {phone_hint or 'not found'}")
    clues.append(f"- Address fragments from receipt: {'; '.join(addr_lines) if addr_lines else 'none found'}")

    if (
        not raw_location
        and branch_hint
        and not branch_hint.endswith("店")
        and re.fullmatch(r'[\wぁ-んァ-ン一-龥ー・]{2,20}', branch_hint)
    ):
        return f"{branch_hint}店", None
    if facility_hint:
        return facility_hint, None

    prompt = f"""Given these clues from a Japanese receipt, determine the city (市) or ward (区) where this store is located.
Output ONLY a JSON object with a single "location" field. Preserve a full address when it is explicitly printed. Otherwise return only a city, ward, or locality supported by explicit geographic text in the receipt.
An ambiguous branch name or phone number alone is not sufficient evidence.

Clues:
{chr(10).join(clues)}

Respond with a JSON object: {{"location": "..."}} or {{"location": null}} if you cannot determine it."""

    try:
        result = _llm_chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            schema={"type": "object", "properties": {"location": {"type": ["string", "null"]}}, "required": ["location"]},
        )
        import json as _json
        data = _json.loads(sanitize_llm_response(result.content))
        resolved = data.get("location")
        if (
            resolved
            and ADMIN_SUFFIX_RE.search(resolved)
            and _location_has_ocr_evidence(resolved, ocr_text)
        ):
            return resolved, None
    except Exception as e:
        logger.warning("Location resolution failed: %s", e)
        return None, "Location resolution failed: LLM call error"
    return None, "Location resolution: could not determine city/ward from available clues"
