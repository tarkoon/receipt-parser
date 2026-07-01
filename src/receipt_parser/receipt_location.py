"""Receipt location repair and resolution helpers."""

import logging
import re

from .patterns import ADMIN_SUFFIX_RE, LOCATION_CLUE_RE


logger = logging.getLogger(__name__)


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
        return True
    if ocr_text and LOCATION_CLUE_RE.search(ocr_text):
        return True
    return False


def _location_has_ocr_evidence(location: str, ocr_text: str) -> bool:
    """Check if at least part of the location string has evidence in the OCR text.

    Normalizes whitespace before comparison since Japanese OCR frequently
    inserts spaces between characters (e.g., "宗像市 赤間" vs "宗像市赤間").
    """
    if not location or not ocr_text:
        return False
    # Normalize whitespace in both strings for comparison
    loc_norm = re.sub(r'\s+', '', location)
    ocr_norm = re.sub(r'\s+', '', ocr_text)
    if loc_norm in ocr_norm:
        return True
    # Check individual admin-level segments. Also split on facility suffixes
    # (IC/インターチェンジ) so a toll-gate name like "若宮IC" is matched against
    # the bare "若宮" that appears in OCR after 料金所.
    parts = re.split(r'[市区町村郡県都道府]|IC$|インターチェンジ$', location)
    for part in parts:
        part = part.strip()
        if len(part) >= 2 and part in ocr_norm:
            return True
    return False


_PURCHASE_STORE_METADATA_RE = re.compile(
    r'^\s*(?:ご購入店|購入店|お買上店|お買い上げ店)\s*[:：]?\s*(?P<store>.+店)\s*$'
)


def _trim_purchase_store_metadata_location(extracted: dict, ocr_text: str) -> None:
    """Avoid expanding location from a labeled host-store metadata line."""
    location = re.sub(r'\s+', '', extracted.get("location") or "")
    if not location:
        return
    if location in re.sub(r'\s+', '', ocr_text):
        return

    base_match = re.match(r'(?P<base>.*?[市区町村])(?P<tail>.+)$', location)
    if not base_match:
        return
    base = base_match.group("base")
    tail = base_match.group("tail")
    if len(tail) < 2:
        return

    for raw_line in ocr_text.splitlines():
        metadata = _PURCHASE_STORE_METADATA_RE.match(raw_line.strip())
        if not metadata:
            continue
        store_line = re.sub(r'\s+', '', metadata.group("store"))
        if tail in store_line:
            extracted["location"] = base
            return


def _trim_store_in_store_header_location(extracted: dict, ocr_text: str) -> None:
    """Avoid treating a host store in a mixed brand/store header as the location."""
    location = re.sub(r'\s+', '', str(extracted.get("location") or ""))
    if not location or not ocr_text:
        return
    merchant = re.sub(r'\s+', '', str(extracted.get("merchant") or "")).upper()
    for raw_line in ocr_text.splitlines()[:8]:
        line = re.split(r'(?:TEL|電話|☎)', raw_line.strip(), maxsplit=1, flags=re.IGNORECASE)[0].strip()
        if not line:
            continue
        match = re.match(r"^(?P<brand>[A-Z][A-Z0-9&.'-]{2,})\s+(?P<host>.+店)$", line)
        if not match:
            continue
        brand = match.group("brand").upper()
        host = re.sub(r'\s+', '', match.group("host"))
        if merchant and merchant != brand:
            continue
        if not re.search(r'[ぁ-んァ-ン一-龥]', host):
            continue
        city = _city_from_phone_area_hint(ocr_text) or _city_from_geographic_marker(host)
        if not city:
            continue
        whole_header = re.sub(r'\s+', '', line)
        if location in {host, whole_header} or (brand in location and host in location):
            extracted["location"] = city
            return


def _recover_header_branch_store_location(extracted: dict, ocr_text: str) -> None:
    """Recover a visible branch/store token from the receipt header."""
    if not ocr_text:
        return
    current_location = re.sub(r'\s+', '', str(extracted.get("location") or ""))
    phone_area_city = _city_from_phone_area_hint(ocr_text)
    can_override_phone_area_city = bool(
        current_location and phone_area_city and current_location == phone_area_city
    )
    can_override_admin_fragment = _is_broad_japanese_admin_location(current_location)
    if current_location and not can_override_phone_area_city and not can_override_admin_fragment:
        return
    for raw_line in ocr_text.splitlines()[:16]:
        line = raw_line.strip()
        if not line:
            continue
        line_for_branch = re.split(r'(?:TEL|電話|☎)', line, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        if re.search(r'www\.|https?://|登録番号|領収|レシート|合計|小計|支払', line, re.IGNORECASE):
            continue
        if re.search(r'\d{4}[年/-]\d{1,2}[月/-]\d{1,2}', line):
            break
        if _PURCHASE_STORE_METADATA_RE.match(line_for_branch):
            continue
        if not re.search(r'[ぁ-んァ-ン一-龥]', line_for_branch):
            continue
        parts = [part.strip() for part in re.split(r'\s+', line_for_branch) if part.strip()]
        candidates = [part for part in parts if re.search(r'[ぁ-んァ-ン一-龥]', part) and part.endswith("店")]
        if not candidates and line_for_branch.endswith("店"):
            candidates = [line_for_branch]
        if not candidates:
            continue
        candidate = candidates[-1]
        stem = candidate[:-1] if candidate.endswith("店") else candidate
        if can_override_phone_area_city and len(stem) > 6:
            continue
        if can_override_admin_fragment and not _branch_extends_admin_fragment(current_location, candidate):
            continue
        if 2 <= len(stem) and len(candidate) <= 20:
            extracted["location"] = candidate
            return


def _recover_ascii_brand_header_location(extracted: dict, ocr_text: str) -> None:
    """Recover compact location text from an early "ASCII brand + suffix" line."""
    if not ocr_text:
        return
    merchant = re.sub(r'\s+', '', str(extracted.get("merchant") or "")).upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9&.'-]{2,}", merchant):
        return
    current = re.sub(r'\s+', '', str(extracted.get("location") or ""))
    if current and not _is_broad_japanese_admin_location(current):
        return
    for raw_line in ocr_text.splitlines()[:8]:
        compact = re.sub(r'\s+', '', raw_line.strip())
        if not compact.upper().startswith(merchant):
            continue
        suffix = compact[len(merchant):]
        if (
            2 <= len(suffix) <= 10
            and re.fullmatch(r'[ぁ-んァ-ン一-龥ー]+', suffix)
            and not ADMIN_SUFFIX_RE.fullmatch(suffix)
        ):
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
    stem = re.sub(r'\s+', '', str(branch or ""))
    if stem.endswith("店"):
        stem = stem[:-1]
    admin_roots = re.findall(r'([一-龥]{1,10}?)(?:都|道|府|県|市|区|町|村)', location)
    if not admin_roots:
        return False
    root = admin_roots[-1]
    return (
        (stem.startswith(root) or stem.endswith(root))
        and 3 <= len(stem) <= 6
        and len(stem) - len(root) >= 2
    )


_PHONE_AREA_CITY_HINTS = {
    "0940": ("宗像市", (("福津", "福津市"), ("宗像", "宗像市"))),
    "093": ("北九州市", ()),
    "092": ("福岡市", ()),
    "0942": ("久留米市", ()),
    "0948": ("飯塚市", ()),
}


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


def _city_from_phone_area_hint(ocr_text: str) -> str:
    phone_hint = _extract_japanese_phone_hint(ocr_text)
    if not phone_hint:
        return ""
    area_code = re.match(r'(0\d{1,4})', phone_hint.replace('-', '').replace(' ', ''))
    if not area_code:
        return ""
    code = area_code.group(1)
    for prefix, (default_city, markers) in _PHONE_AREA_CITY_HINTS.items():
        if not code.startswith(prefix):
            continue
        for marker, city in markers:
            if marker in ocr_text:
                return city
        return default_city
    return ""


def _city_from_geographic_marker(text: str) -> str:
    for _prefix, (_default_city, markers) in _PHONE_AREA_CITY_HINTS.items():
        for marker, city in markers:
            if marker and marker in text:
                return city
    return ""


def _recover_phone_area_city_location(extracted: dict, ocr_text: str) -> None:
    if extracted.get("location") or not ocr_text:
        return
    has_ambiguous_store_phone_line = any(
        re.search(r'[ぁ-んァ-ン一-龥]{1,}店.*(?:TEL|電話|☎)', line, re.IGNORECASE)
        for line in ocr_text.splitlines()[:12]
    )
    if not has_ambiguous_store_phone_line:
        return
    city = _city_from_phone_area_hint(ocr_text)
    if city:
        extracted["location"] = city


def _recover_short_branch_over_phone_area_city(extracted: dict, ocr_text: str) -> None:
    current_location = re.sub(r'\s+', '', str(extracted.get("location") or ""))
    phone_area_city = _city_from_phone_area_hint(ocr_text or "")
    if not current_location or not phone_area_city or not current_location.startswith(phone_area_city):
        return
    if current_location == phone_area_city:
        return
    current_tail = current_location[len(phone_area_city):]
    if ADMIN_SUFFIX_RE.search(current_tail):
        return
    for raw_line in (ocr_text or "").splitlines()[:16]:
        line = raw_line.strip()
        if re.search(r'購入店|お買上店|登録番号|領収|レシート|合計|小計|支払', line):
            continue
        line_for_branch = re.split(r'(?:TEL|電話|☎)', line, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        match = re.fullmatch(r'([ぁ-んァ-ン一-龥]{2,6}店)', line_for_branch)
        if match:
            branch = match.group(1)
            stem = branch[:-1]
            if (
                stem in current_location
                or (current_tail and current_tail in stem)
            ):
                extracted["location"] = branch
                return


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
            break
    if not branch_match:
        # Location-type indicators: toll gate, station, branch office
        branch_match = re.search(r'(?:料金所|営業所|支店|出張所)\s*\n?\s*([\u3000-\u9fff]{2,})', ocr_text)
    branch_hint = branch_match.group(1) if branch_match else ""

    # Also extract short standalone Japanese text from the first few lines
    # (often branch/location names like "赤間" above the brand name)
    _FINANCIAL_KEYWORDS = {
        '合計', '小計', '税込', '税抜', '総額', '釣銭', '預金', '現計', '点数',
        '領収証', '領収書', 'ドラッグストア', 'ドラックストア',
    }
    header_lines = []
    for line in ocr_text.split('\n')[:8]:
        s = line.strip()
        if s and 2 <= len(s) <= 8 and re.match(r'^[\u3000-\u9fff]+$', s) and s not in _FINANCIAL_KEYWORDS:
            header_lines.append(s)
    header_hint = ", ".join(header_lines) if header_lines else ""

    # Phone area code → region hint
    area_hint = ""
    area_city = ""
    area_prefix = ""
    if phone_hint:
        area_code = re.match(r'(0\d{1,4})', phone_hint.replace('-', '').replace(' ', ''))
        if area_code:
            code = area_code.group(1)
            for prefix, (city, _markers) in _PHONE_AREA_CITY_HINTS.items():
                if code.startswith(prefix):
                    desc = f"{city} area"
                    area_hint = f"Phone area code {prefix} = {desc}"
                    area_city = city
                    area_prefix = prefix
                    break

    addr_lines = []
    for line in ocr_text.split('\n'):
        if re.search(r'[都道府県市区町村郡]|〒\d{3}', line):
            addr_lines.append(line.strip())

    clues = [f"- Merchant/brand: {merchant}"]
    if branch_hint:
        clues.append(f"- Branch/store name: {branch_hint}店 (the branch name often indicates the neighborhood)")
    if header_hint and header_hint != branch_hint:
        clues.append(f"- Receipt header text: {header_hint} (may contain location or branch name)")
    clues.append(f"- Current location value: {raw_location or 'unknown'}")
    clues.append(f"- Phone number: {phone_hint or 'not found'}")
    if area_hint:
        clues.append(f"- {area_hint}")
    clues.append(f"- Address fragments from receipt: {'; '.join(addr_lines) if addr_lines else 'none found'}")

    # Deterministic resolution: if area code gives us a city and we have a
    # neighborhood name (from branch or header), combine them directly.
    neighborhood = ""
    if branch_hint:
        # Strip common prefixes from branch name to get neighborhood
        # e.g., "ビバモール赤間" → take last 2-3 chars as neighborhood
        for suffix_len in (3, 2):
            candidate = branch_hint[-suffix_len:]
            if re.match(r'^[\u3000-\u9fff]+$', candidate):
                neighborhood = candidate
                break
        if not neighborhood:
            neighborhood = branch_hint
    elif header_lines:
        # Use the first short header line as neighborhood
        for h in header_lines:
            if (
                2 <= len(h) <= 4
                and re.match(r'^[\u3000-\u9fff]+$', h)
                and h != merchant
            ):
                neighborhood = h
                break

    if area_city and neighborhood:
        area_root = re.sub(r'[都道府県市区町村郡]+$', '', area_city)
        if neighborhood == area_root:
            return area_city, None
        candidate = f"{area_city}{neighborhood}"
        if ADMIN_SUFFIX_RE.search(candidate):
            return candidate, None

    if area_prefix in _PHONE_AREA_CITY_HINTS:
        for marker, city in _PHONE_AREA_CITY_HINTS[area_prefix][1]:
            if marker in ocr_text:
                return city, None

    if area_city and raw_location and not ADMIN_SUFFIX_RE.search(raw_location):
        return area_city, None

    business_line = re.search(r'事業者名\s*[:：]\s*(.+)$', ocr_text, re.MULTILINE)
    if not raw_location and business_line:
        business = business_line.group(1).strip()
        rail = re.match(r'(.{2,12}?)(?:鉄道)?株式会社$', business)
        if rail and '登山' in rail.group(1):
            return rail.group(1), None

    if (
        not raw_location
        and branch_hint
        and not branch_hint.endswith("店")
        and re.fullmatch(r'[\wぁ-んァ-ン一-龥ー・]{2,20}', branch_hint)
    ):
        return f"{branch_hint}店", None

    if area_city and not raw_location:
        return area_city, None

    # Toll-receipt deterministic path: NEXCO/expressway receipts print 料金所
    # alone on its own line, followed by the toll-gate name on the next non-
    # empty line (e.g. "若宮", "小倉南"). Without a phone area code (toll-free
    # 0120) the LLM otherwise falls back to the corporate HQ address — the
    # geographically wrong place. Use the toll-gate name with an "IC" suffix
    # as the location instead. Skip the noise line "料金所では一旦停車…".
    is_toll = bool(re.search(r'料金所|高速道路|NEXCO', ocr_text))
    if is_toll:
        toll_lines = ocr_text.split('\n')
        toll_name = ""
        for idx, raw in enumerate(toll_lines):
            line = raw.strip()
            # Exact "料金所" on a line by itself (not "料金所では一旦…")
            if line == "料金所":
                for nxt in toll_lines[idx + 1:idx + 4]:
                    cand = nxt.strip()
                    # 2-4 char Japanese name (kanji + kana), no punctuation
                    if cand and re.match(r'^[　-鿿]{2,8}$', cand) and 'です' not in cand:
                        toll_name = cand
                        break
                if toll_name:
                    break
        if toll_name:
            if not re.search(r'(?:IC|インターチェンジ|料金所)$', toll_name):
                return f"{toll_name}IC", None
            return toll_name, None

    prompt = f"""Given these clues from a Japanese receipt, determine the city (市) or ward (区) where this store is located.
Output ONLY a JSON object with a single "location" field. The location should be at the 市区町村 level, e.g. "宗像市赤間", "福岡市博多区", "北九州市八幡区".
The branch name (e.g. 赤間店 → 赤間 is a neighborhood in 宗像市) is the strongest clue for location.

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
        if resolved and ADMIN_SUFFIX_RE.search(resolved):
            return resolved, None
    except Exception as e:
        logger.warning("Location resolution failed: %s", e)
        return None, "Location resolution failed: LLM call error"
    return None, "Location resolution: could not determine city/ward from available clues"
