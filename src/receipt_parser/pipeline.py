"""pipeline.py — Orchestrates all pipeline stages.

Uses Google Cloud Vision OCR → text → LLM (OpenRouter or Ollama) for structured extraction.
Supports batch processing with concurrent API calls via process_batch().
"""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import cv2



logger = logging.getLogger(__name__)

from .schema import Receipt
from .preprocess import load_image, try_extract_text_layer
from .ocr import init_cloud_vision, run_cloud_vision, blocks_to_structured_text, compute_ocr_confidence, OCRResult
from .llm import check_model_available, extract_with_verification, DEFAULT_MODEL
from .validation import validate_receipt
from .normalize import (normalize_fullwidth, clean_handwritten_ocr, strip_barcode_lines,
                        rejoin_price_lines, _shift_misaligned_inline_prices, strip_bonus_point_lines,
                        join_split_qty_details,
                        rejoin_totals_label_value_columns)
from .tracing import PipelineTrace, draw_ocr_bboxes, draw_field_overlay
from .patterns import (
    UTILITY_BILL_KEYWORDS, PAYMENT_SLIP_KEYWORDS, RECEIPT_KEYWORDS,
    ADMIN_SUFFIX_RE, LOCATION_CLUE_RE,
)
from .pipeline_receipt import (
    extract_financial_totals, extract_points_used, postprocess_receipt,
    extract_rate_bases,
    reconcile_tax_categories_from_rate_bases,
    reconcile_points_payment_from_ocr,
    _drop_unprinted_small_target_only_taxes,
    _restore_bare_number_tax_summary,
    _drop_duplicate_with_embedded_price,
    _fix_company_name_merchant,
    _fix_item_totals_from_following_discount_lines,
    _apply_coupon_discount_blocks,
    _drop_applied_coupon_line_items,
    _repair_tiny_item_prices_from_following_ocr,
    _repair_discounted_line_item_totals_when_balanced,
    _repair_discounted_ocr_pair_descriptions,
    _repair_pre_price_stack_descriptions_from_ocr,
    _drop_duplicate_rows_when_subtotal_balances,
    _replace_basket_marker_rows_when_balanced,
    _fix_adjacent_ocr_price_shift_when_balanced,
    _fix_o_ring_descriptions_from_ocr,
    _fix_qty_totals_from_ocr_unit_lines,
    _fix_bag_item_prices_from_rate_bases,
    _fix_code_table_descriptions_by_order,
    _fix_unlabeled_cash_tender_change_block,
    _clear_discount_when_negative_line_precedes_own_price,
    _prefer_printed_item_sum_total_when_balanced,
    _replace_campaign_discount_stream_when_balanced,
    _replace_dense_sequence_rows_when_balanced,
    _replace_prefixed_tax_marker_item_rows_when_balanced,
    _replace_jan_pos_items_when_balanced,
    _replace_barcode_unit_qty_amount_stack_when_balanced,
    _recover_labeled_purchase_site_location,
    _restore_printed_external_tax_amounts,
    _restore_printed_summary_total_when_tax_balanced,
    _restore_external_tax_total_from_printed_subtotal,
    _replace_barcode_qty_price_rows_when_balanced,
    _replace_item_price_qty_rows_when_balanced,
    _recover_repeated_item_from_gap,
    _recover_missing_items_from_gap,
    _replace_split_price_block_when_balanced,
    _fix_split_item_price_body_total_layout,
    _replace_stacked_name_price_rows_when_balanced,
    _restore_stacked_inclusive_tax_block,
    _restore_single_rate_inclusive_tax_block,
    POSTPROCESS_PHASE_BY_NAME,
    _record_receipt_mutation,
    _snapshot_receipt_mutation_fields,
)
from .pipeline_bill import postprocess_utility_bill
from .pipeline_slip import postprocess_payment_slip

import inspect
from typing import Any, Callable, Literal

# Public progress contract — these stage names are stable across pipeline versions.
# Consumers may switch on these strings safely.
StageName = Literal[
    "load",
    "ocr",
    "classify",
    "normalize",
    "extract",
    "postprocess",
    "resolve_location",
    "validate",
    "warn",
    "plan",
    "done",
]

# Callback may be 3-arg (stage, detail, progress) — original contract — or
# 4-arg by naming the extra parameter `payload` (or accepting **kwargs).
# Returning False from any call requests cooperative cancellation.
StageCallback = Callable[..., Any] | None


class PipelineCancelled(Exception):
    """Raised when an on_stage callback returns False to abort the pipeline."""
    def __init__(self, stage: str):
        super().__init__(f"Pipeline cancelled at stage: {stage}")
        self.stage = stage


def _callback_accepts_payload(cb) -> bool:
    """Decide whether to pass the structured payload as a keyword argument.

    A callback opts in to the 4-arg contract by naming a parameter `payload`
    or accepting **kwargs. Counting positional slots is unreliable: callers
    sometimes use a 4th defaulted positional (e.g. ``def cb(s, d, p, _task=t)``)
    as a closure-capture idiom, and we must NOT pass our payload there.
    """
    try:
        sig = inspect.signature(cb)
    except (ValueError, TypeError):
        return False
    params = sig.parameters
    if "payload" in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def _notify(
    on_stage: StageCallback,
    stage: str,
    detail: str,
    progress: float,
    payload: dict | None = None,
) -> None:
    """Fire a progress callback if one is registered.

    Backwards compatible with 3-arg callbacks. The structured payload is only
    passed when the callback's signature opts in (param named `payload` or
    accepts **kwargs). If the callback returns False, raises PipelineCancelled.
    """
    if on_stage is None:
        return
    if _callback_accepts_payload(on_stage):
        result = on_stage(stage, detail, progress, payload=payload)
    else:
        result = on_stage(stage, detail, progress)
    if result is False:
        raise PipelineCancelled(stage)


_PIPELINE_VERSION = "3.1.0"


def detect_document_type(text: str) -> str:
    """Classify document type from OCR text using keyword matching."""
    utility_score = len(UTILITY_BILL_KEYWORDS.findall(text))
    slip_score = len(PAYMENT_SLIP_KEYWORDS.findall(text))
    receipt_score = len(RECEIPT_KEYWORDS.findall(text))

    if utility_score >= 2 and utility_score > receipt_score:
        return "utility_bill"
    if slip_score >= 2 and slip_score > receipt_score:
        return "payment_slip"
    return "receipt"


_USER_RULES_PATH = Path(__file__).parent / "user_rules.json"


def _apply_user_rules(result: dict) -> dict:
    """Apply user_rules.json merchant alias mapping."""
    if not _USER_RULES_PATH.exists():
        return result
    try:
        rules = json.loads(_USER_RULES_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return result

    merchant_map = rules.get("merchant_map", {})
    merchant = result.get("merchant", "") or ""

    for pattern, mapping in merchant_map.items():
        if pattern in merchant:
            if "merchant" in mapping:
                result["merchant"] = mapping["merchant"]
            if "category" in mapping:
                result["_category"] = mapping["category"]
            break

    return result


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


def _compute_posthoc_confidence(extracted: dict, warnings: list[str]) -> dict:
    """Compute per-field confidence from validation results (post-hoc).

    Maps warning text to fields using keyword matching that avoids false
    positives from field names appearing in unrelated warning context
    (e.g., "Line 1: total is X" should affect line_items, not total).
    """
    # Map warning prefixes/keywords to the field they pertain to.
    # Checked in order; first match wins per warning to avoid double-assignment.
    _WARNING_FIELD_RULES: list[tuple[str, str, bool]] = [
        # (pattern, field, prefix_only)
        ("Line ", "line_items", True),
        ("Sum of line items", "line_items", True),
        ("Items sum", "line_items", True),
        ("discount_rate", "line_items", False),
        ("Total (", "total", True),
        ("Total does not match", "total", True),
        ("subtotal (", "subtotal", False),
        ("Tax ratio", "taxes", True),
        ("tax rate", "taxes", False),
        ("Unusual tax rate", "taxes", True),
        ("amount_paid", "points_used", False),
        ("usage.", "usage", False),
        ("billing_period", "billing_period", False),
        ("merchant", "merchant", False),
    ]

    affected_fields: set[str] = set()
    for w in warnings:
        for pattern, field, prefix_only in _WARNING_FIELD_RULES:
            matched = w.startswith(pattern) if prefix_only else (pattern in w)
            if matched:
                affected_fields.add(field)
                break

    conf = {}
    for field in ("merchant", "date", "total", "subtotal", "taxes",
                   "payment_method", "line_items", "points_used"):
        val = extracted.get(field)
        if val is None or (isinstance(val, list) and len(val) == 0):
            conf[field] = 0.0
        elif field in affected_fields:
            conf[field] = 0.4
        else:
            conf[field] = 0.9

    return conf


def _build_validate_detail(extracted: dict) -> str:
    """Format a short receipt preview string for the validate-stage callback."""
    if "error" in extracted or "_error" in extracted:
        return "Validating"

    parts: list[str] = []
    doc_type = extracted.get("document_type") or "receipt"

    if doc_type == "receipt":
        n = len(extracted.get("line_items") or [])
        if n:
            parts.append(f"{n} item{'s' if n != 1 else ''}")
    elif doc_type == "utility_bill":
        st = extracted.get("service_type")
        if st:
            parts.append(str(st))
    elif doc_type == "payment_slip":
        parts.append("payment slip")

    total = extracted.get("total")
    if total is not None:
        currency = extracted.get("currency") or ""
        symbol = "¥" if currency in ("JPY", "") else ""
        try:
            parts.append(f"{symbol}{int(round(float(total))):,}")
        except (TypeError, ValueError):
            pass

    return " · ".join(parts) if parts else "Validating"


def _build_plan_payload(
    page_count: int,
    pass_budget: int,
    doc_type: str | None = None,
) -> dict:
    """Build the structured payload for a 'plan' stage event."""
    if doc_type is None:
        path = "tbd"
        will_resolve_location = True  # unknown until classify; assume yes
    else:
        path = doc_type
        will_resolve_location = (doc_type == "receipt")
    return {
        "path": path,
        "page_count": page_count,
        "pass_budget": pass_budget,
        "will_resolve_location": will_resolve_location,
    }


def _expected_stages(doc_type: str, source: str) -> list[str]:
    """The remaining stage keys this document will fire after classify, in order.

    Lets consumers finalize the step list as soon as classify lands, instead of
    inferring the path from doc_type + source themselves.
    """
    if source == "digital_pdf":
        # Fast path: no OCR-grouping, normalize, postprocess, or resolve_location
        # stage events. (Location resolution may still run internally for
        # receipts, but no event is fired for it.)
        return ["extract", "validate", "done"]
    stages = ["extract", "postprocess"]
    if doc_type == "receipt":
        stages.append("resolve_location")
    stages.extend(["validate", "done"])
    return stages


def _build_classify_payload(doc_type: str, source: str) -> dict:
    """Structured payload for the 'classify' event — enough info to finalize the
    UI step list without needing to memorize the per-doc-type path table."""
    if source == "digital_pdf":
        will_resolve = False  # not emitted as a stage even when it runs
    else:
        will_resolve = (doc_type == "receipt")
    return {
        "document_type": doc_type,
        "source": source,
        "will_resolve_location": will_resolve,
        "expected_stages": _expected_stages(doc_type, source),
    }


def _record_final_receipt_output_repair(
    stage: str,
    result: dict,
    mutation_trace: list[dict] | None,
    repair: Callable[[], None],
) -> None:
    before = (
        _snapshot_receipt_mutation_fields(result)
        if mutation_trace is not None
        else None
    )
    repair()
    trace_len = len(mutation_trace) if mutation_trace is not None else 0
    _record_receipt_mutation(mutation_trace, stage, before, result)
    if mutation_trace is not None and len(mutation_trace) > trace_len:
        owner_phase, justification = FINAL_RECEIPT_OUTPUT_REPAIR_JUSTIFICATIONS[stage]
        mutation_trace[-1]["owner_phase"] = owner_phase
        mutation_trace[-1]["owner_invariant"] = POSTPROCESS_PHASE_BY_NAME[owner_phase][
            "invariant"
        ]
        mutation_trace[-1]["justification"] = justification


def _run_final_structural_item_projection_phase(
    result: dict,
    ocr_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: visible JAN/barcode item rows followed by unit x qty amounts.

    Invariant: projected rows may replace collapsed line items only when
    OCR-derived item totals balance with the receipt subtotal and tax summary.
    """
    for repair in repairs:
        if repair == "barcode_unit_qty_amount_stack":
            _replace_barcode_unit_qty_amount_stack_when_balanced(result, ocr_text)
        else:
            raise ValueError(
                f"Unknown final structural item projection repair: {repair}"
            )


def _run_final_barcode_qty_price_projection_phase(
    result: dict,
    ocr_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: visible barcode/JAN rows followed by quantity-price rows.

    Invariant: projected items may replace collapsed duplicates only when the
    OCR-derived item sum remains consistent with the printed receipt total.
    """
    for repair in repairs:
        if repair == "barcode_qty_price_rows":
            _replace_barcode_qty_price_rows_when_balanced(result, ocr_text)
        else:
            raise ValueError(
                f"Unknown final barcode quantity-price projection repair: {repair}"
            )


def _run_final_item_price_qty_projection_phase(
    result: dict,
    ocr_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: description rows paired with price and quantity-detail rows.

    Invariant: projected items may replace current rows only when OCR-derived
    totals match the printed subtotal and, when present, printed item count.
    """
    for repair in repairs:
        if repair == "item_price_qty_rows":
            _replace_item_price_qty_rows_when_balanced(result, ocr_text)
        else:
            raise ValueError(
                f"Unknown final item price quantity projection repair: {repair}"
            )


def _run_final_split_price_block_projection_phase(
    result: dict,
    ocr_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: split description block paired with separated price rows.

    Invariant: projected items may replace current rows only when OCR-derived
    prices balance with the printed subtotal or later total target.
    """
    for repair in repairs:
        if repair == "split_price_block":
            _replace_split_price_block_when_balanced(result, ocr_text)
        else:
            raise ValueError(
                f"Unknown final split price block projection repair: {repair}"
            )


def _run_final_body_total_layout_reconstruction_phase(
    result: dict,
    ocr_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: item rows appear before a printed body-total block.

    Invariant: reconstructed items, subtotal, and tax entries must remain
    backed by visible body-total layout rows and subtotal plus tax arithmetic.
    """
    for repair in repairs:
        if repair == "split_item_price_body_total":
            _fix_split_item_price_body_total_layout(result, ocr_text)
        else:
            raise ValueError(
                f"Unknown final body-total layout reconstruction repair: {repair}"
            )


def _run_final_stacked_name_price_projection_phase(
    result: dict,
    ocr_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: stacked description rows paired with nearby price rows.

    Invariant: projected items may replace current rows only when OCR-derived
    totals balance against the printed subtotal and, when present, rate bases.
    """
    for repair in repairs:
        if repair == "stacked_name_price_rows":
            _replace_stacked_name_price_rows_when_balanced(result, ocr_text)
        else:
            raise ValueError(
                f"Unknown final stacked name/price projection repair: {repair}"
            )


FINAL_RECEIPT_OUTPUT_REPAIR_JUSTIFICATIONS = {
    "barcode_unit_qty_amount_stack": (
        "structural_item_reconstruction",
        "Owned by the final structural item-projection helper until this barcode stack projection no longer needs post-serialization repair.",
    ),
    "barcode_qty_price_rows": (
        "structural_item_reconstruction",
        "Owned by the final barcode quantity-price projection helper until barcode row projection moves out of post-serialization repair.",
    ),
    "item_price_qty_rows": (
        "structural_item_reconstruction",
        "Owned by the final item price/quantity projection helper until row projection moves out of post-serialization repair.",
    ),
    "labeled_purchase_site_location": (
        "header_identity_repair",
        "Retained late until location resolution no longer needs post-serialization correction.",
    ),
    "store_in_store_header_location": (
        "header_identity_repair",
        "Late-only location cleanup for mixed brand and host-store headers.",
    ),
    "header_branch_store_location": (
        "header_identity_repair",
        "Late-only branch recovery after generic location resolution.",
    ),
    "phone_area_city_location": (
        "header_identity_repair",
        "Late-only location recovery from phone-area evidence.",
    ),
    "short_branch_over_phone_area_city": (
        "header_identity_repair",
        "Late-only correction when a visible branch is more specific than phone-area city.",
    ),
    "noisy_city_location": (
        "header_identity_repair",
        "Late-only cleanup for OCR-expanded city locations.",
    ),
    "single_rate_inclusive_tax_block": (
        "tax_category_assignment",
        "Retained late for serialized tax block consistency.",
    ),
    "following_discount_lines": (
        "structural_item_reconstruction",
        "Retained late for discount-line totals exposed by final item layout repair.",
    ),
    "coupon_discount_blocks": (
        "structural_item_reconstruction",
        "Retained late for coupon rows exposed by final item layout repair.",
    ),
    "drop_applied_coupon_line_items": (
        "item_cleanup",
        "Retained late to remove coupon rows introduced or exposed by reconstruction.",
    ),
    "tiny_item_prices_from_following_ocr": (
        "structural_item_reconstruction",
        "Retained late for tiny item price projection after serialization.",
    ),
    "split_price_block": (
        "structural_item_reconstruction",
        "Owned by the final split price block projection helper until split price blocks move out of post-serialization repair.",
    ),
    "split_item_price_body_total": (
        "structural_item_reconstruction",
        "Owned by the final body-total layout reconstruction helper until body-total split layouts move out of post-serialization repair.",
    ),
    "stacked_name_price_rows": (
        "structural_item_reconstruction",
        "Owned by the final stacked name/price projection helper until stacked row projection moves out of post-serialization repair.",
    ),
    "stacked_inclusive_tax_block": (
        "tax_category_assignment",
        "Retained late for stacked inclusive tax summaries after item reconstruction.",
    ),
    "printed_summary_total_tax_balanced": (
        "financial_totals_repair",
        "Retained late to restore printed totals when final tax balance proves them.",
    ),
    "printed_item_sum_total": (
        "financial_totals_repair",
        "Retained late to prefer visible printed item-sum totals when balanced.",
    ),
    "o_ring_descriptions": (
        "item_cleanup",
        "Retained late because JAN/barcode reconstruction can expose O-ring descriptions.",
    ),
    "company_name_merchant": (
        "header_identity_repair",
        "Temporary debt: direct final-output callers still need invoice/registration merchant cleanup.",
    ),
    "adjacent_ocr_price_shift": (
        "structural_item_reconstruction",
        "Retained late for adjacent OCR price shifts before final item cleanup.",
    ),
    "repeated_item_gap": (
        "initial_item_recovery",
        "Retained late for repeated item gaps exposed by final row projections.",
    ),
    "drop_duplicate_embedded_price": (
        "item_cleanup",
        "Retained late to remove embedded-price duplicates after final reconstruction.",
    ),
    "dense_sequence_rows": (
        "structural_item_reconstruction",
        "Retained late for dense sequence reconstruction pending single phase ownership.",
    ),
    "campaign_discount_stream": (
        "structural_item_reconstruction",
        "Retained late for campaign discount streams before final discount cleanup.",
    ),
    "jan_pos_items": (
        "structural_item_reconstruction",
        "Retained late for JAN/POS row projection pending postprocess-only ownership.",
    ),
    "qty_totals_from_unit_lines": (
        "quantity_detail_reconciliation",
        "Retained late to restore quantity totals exposed by final row projection.",
    ),
    "bag_item_prices_from_rate_bases": (
        "tax_category_assignment",
        "Retained late for bag price/category consistency with final rate bases.",
    ),
    "code_table_descriptions": (
        "item_cleanup",
        "Retained late for code-table descriptions exposed by final item projection.",
    ),
    "printed_external_tax_amounts": (
        "tax_category_assignment",
        "Retained late for external tax summaries after final total repair.",
    ),
    "bare_number_tax_summary": (
        "tax_category_assignment",
        "Retained late for bare-number tax summaries after item reconstruction.",
    ),
    "external_tax_total_from_printed_subtotal": (
        "financial_totals_repair",
        "Retained late to restore externally taxed totals from printed subtotal arithmetic.",
    ),
    "drop_small_target_only_taxes": (
        "tax_category_assignment",
        "Retained late to remove unprinted small target-only taxes after rate repair.",
    ),
    "printed_summary_total_tax_balanced_2": (
        "financial_totals_repair",
        "Temporary debt: repeated after later tax repairs can change total balance.",
    ),
    "unlabeled_cash_tender_change": (
        "payment_points_reconciliation",
        "Retained late for cash tender/change blocks after final total repair.",
    ),
    "points_payment": (
        "payment_points_reconciliation",
        "Retained late to reconcile points and payment after final total changes.",
    ),
    "clear_discount_before_own_price": (
        "item_cleanup",
        "Late-only cleanup for negative discount lines before their owner price.",
    ),
    "campaign_discount_stream_2": (
        "structural_item_reconstruction",
        "Temporary debt: repeated after discount cleanup can expose balanced campaign streams.",
    ),
    "following_discount_lines_after_layout": (
        "structural_item_reconstruction",
        "Temporary debt: repeated after layout repair can expose discount lines.",
    ),
    "discounted_line_item_totals": (
        "final_consistency_pass",
        "Retained late for discounted line totals after all row reconstruction.",
    ),
    "adjacent_ocr_price_shift_final": (
        "structural_item_reconstruction",
        "Temporary debt: repeated after discount cleanup can expose adjacent price shifts.",
    ),
    "prefixed_tax_marker_item_rows": (
        "structural_item_reconstruction",
        "Retained late for prefixed tax marker rows after final cleanup.",
    ),
    "missing_items_from_gap": (
        "initial_item_recovery",
        "Retained late for item gaps exposed by final row cleanup.",
    ),
    "discounted_ocr_pair_descriptions": (
        "item_cleanup",
        "Retained late to repair descriptions after discounted rows settle.",
    ),
    "pre_price_stack_descriptions": (
        "item_cleanup",
        "Retained late to repair stacked descriptions after price rows settle.",
    ),
    "drop_duplicate_rows_when_subtotal_balances": (
        "item_cleanup",
        "Retained late to remove duplicates only after final subtotal balance is known.",
    ),
    "basket_marker_rows": (
        "structural_item_reconstruction",
        "Retained late for basket marker projection after discount repairs.",
    ),
    "tax_categories_from_rate_bases": (
        "tax_category_assignment",
        "Late-only tax category reconciliation from final rate bases.",
    ),
    "external_tax_total_from_printed_subtotal_final": (
        "financial_totals_repair",
        "Temporary debt: final reassertion after item/tax category repairs.",
    ),
}


def _apply_final_receipt_output_repairs(
    result: dict,
    ocr_text: str | None,
    mutation_trace: list[dict] | None = None,
) -> None:
    """Apply legacy receipt repairs that still run after model validation."""
    if result.get("document_type") != "receipt" or not ocr_text:
        return

    def run(stage: str, repair: Callable[[], None]) -> None:
        _record_final_receipt_output_repair(stage, result, mutation_trace, repair)

    run(
        "barcode_unit_qty_amount_stack",
        lambda: _run_final_structural_item_projection_phase(
            result,
            ocr_text,
            ("barcode_unit_qty_amount_stack",),
        ),
    )
    run(
        "barcode_qty_price_rows",
        lambda: _run_final_barcode_qty_price_projection_phase(
            result,
            ocr_text,
            ("barcode_qty_price_rows",),
        ),
    )
    run(
        "item_price_qty_rows",
        lambda: _run_final_item_price_qty_projection_phase(
            result,
            ocr_text,
            ("item_price_qty_rows",),
        ),
    )
    run("labeled_purchase_site_location", lambda: _recover_labeled_purchase_site_location(result, ocr_text))
    run("store_in_store_header_location", lambda: _trim_store_in_store_header_location(result, ocr_text))
    run("header_branch_store_location", lambda: _recover_header_branch_store_location(result, ocr_text))
    run("phone_area_city_location", lambda: _recover_phone_area_city_location(result, ocr_text))
    run("short_branch_over_phone_area_city", lambda: _recover_short_branch_over_phone_area_city(result, ocr_text))
    run("noisy_city_location", lambda: _normalize_noisy_city_location(result, ocr_text))
    run("single_rate_inclusive_tax_block", lambda: _restore_single_rate_inclusive_tax_block(result, ocr_text))
    run("following_discount_lines", lambda: _fix_item_totals_from_following_discount_lines(result, ocr_text))
    run("coupon_discount_blocks", lambda: _apply_coupon_discount_blocks(result, ocr_text))
    run("drop_applied_coupon_line_items", lambda: _drop_applied_coupon_line_items(result, ocr_text))
    run("tiny_item_prices_from_following_ocr", lambda: _repair_tiny_item_prices_from_following_ocr(result, ocr_text))
    run(
        "split_price_block",
        lambda: _run_final_split_price_block_projection_phase(
            result,
            ocr_text,
            ("split_price_block",),
        ),
    )
    run(
        "split_item_price_body_total",
        lambda: _run_final_body_total_layout_reconstruction_phase(
            result,
            ocr_text,
            ("split_item_price_body_total",),
        ),
    )
    run(
        "stacked_name_price_rows",
        lambda: _run_final_stacked_name_price_projection_phase(
            result,
            ocr_text,
            ("stacked_name_price_rows",),
        ),
    )
    run("stacked_inclusive_tax_block", lambda: _restore_stacked_inclusive_tax_block(result, ocr_text))
    run("printed_summary_total_tax_balanced", lambda: _restore_printed_summary_total_when_tax_balanced(result, ocr_text))
    run("printed_item_sum_total", lambda: _prefer_printed_item_sum_total_when_balanced(result, ocr_text))
    run("o_ring_descriptions", lambda: _fix_o_ring_descriptions_from_ocr(result, ocr_text))
    run("company_name_merchant", lambda: _fix_company_name_merchant(result, ocr_text))
    run("adjacent_ocr_price_shift", lambda: _fix_adjacent_ocr_price_shift_when_balanced(result, ocr_text))
    run("repeated_item_gap", lambda: _recover_repeated_item_from_gap(result, ocr_text))
    if result.get("line_items"):
        run(
            "drop_duplicate_embedded_price",
            lambda: _drop_duplicate_with_embedded_price(result["line_items"]),
        )
    run("dense_sequence_rows", lambda: _replace_dense_sequence_rows_when_balanced(result, ocr_text))
    run("campaign_discount_stream", lambda: _replace_campaign_discount_stream_when_balanced(result, ocr_text))
    run(
        "jan_pos_items",
        lambda: _replace_jan_pos_items_when_balanced(
            result,
            ocr_text,
            extract_financial_totals(ocr_text),
        ),
    )
    run("qty_totals_from_unit_lines", lambda: _fix_qty_totals_from_ocr_unit_lines(result, ocr_text))
    run(
        "bag_item_prices_from_rate_bases",
        lambda: _fix_bag_item_prices_from_rate_bases(
            result,
            extract_rate_bases(ocr_text),
            ocr_text,
        ),
    )
    run("code_table_descriptions", lambda: _fix_code_table_descriptions_by_order(result, ocr_text))
    run("printed_external_tax_amounts", lambda: _restore_printed_external_tax_amounts(result, ocr_text))
    run("bare_number_tax_summary", lambda: _restore_bare_number_tax_summary(result, ocr_text))
    run("external_tax_total_from_printed_subtotal", lambda: _restore_external_tax_total_from_printed_subtotal(result, ocr_text))
    run("drop_small_target_only_taxes", lambda: _drop_unprinted_small_target_only_taxes(result, ocr_text))
    run("printed_summary_total_tax_balanced_2", lambda: _restore_printed_summary_total_when_tax_balanced(result, ocr_text))
    run("unlabeled_cash_tender_change", lambda: _fix_unlabeled_cash_tender_change_block(result, ocr_text))
    run("points_payment", lambda: reconcile_points_payment_from_ocr(result, ocr_text))
    run(
        "clear_discount_before_own_price",
        lambda: _clear_discount_when_negative_line_precedes_own_price(result, ocr_text),
    )
    run("campaign_discount_stream_2", lambda: _replace_campaign_discount_stream_when_balanced(result, ocr_text))
    run("following_discount_lines_after_layout", lambda: _fix_item_totals_from_following_discount_lines(result, ocr_text))
    run("discounted_line_item_totals", lambda: _repair_discounted_line_item_totals_when_balanced(result, ocr_text))
    run("adjacent_ocr_price_shift_final", lambda: _fix_adjacent_ocr_price_shift_when_balanced(result, ocr_text))
    run("prefixed_tax_marker_item_rows", lambda: _replace_prefixed_tax_marker_item_rows_when_balanced(result, ocr_text))
    run("missing_items_from_gap", lambda: _recover_missing_items_from_gap(result, ocr_text))
    run("discounted_ocr_pair_descriptions", lambda: _repair_discounted_ocr_pair_descriptions(result, ocr_text))
    run("pre_price_stack_descriptions", lambda: _repair_pre_price_stack_descriptions_from_ocr(result, ocr_text))
    run("drop_duplicate_rows_when_subtotal_balances", lambda: _drop_duplicate_rows_when_subtotal_balances(result, ocr_text))
    run("basket_marker_rows", lambda: _replace_basket_marker_rows_when_balanced(result, ocr_text))
    run("tax_categories_from_rate_bases", lambda: reconcile_tax_categories_from_rate_bases(result, ocr_text))
    run("external_tax_total_from_printed_subtotal_final", lambda: _restore_external_tax_total_from_printed_subtotal(result, ocr_text))


def _prepare_receipt_output_payload(
    receipt,
    ocr_text: str | None = None,
    mutation_trace: list[dict] | None = None,
) -> dict:
    result = receipt.model_dump()
    _apply_final_receipt_output_repairs(result, ocr_text, mutation_trace=mutation_trace)
    return result


def _build_result(receipt_payload, final_warnings, pass_history, model, debug=False, trace=None,
                  ocr_confidence=None, llm_confidence=None,
                  ocr_source=None, ocr_retried=None, ocr_retry_reason=None,
                  ocr_text=None, mutation_trace: list[dict] | None = None):
    result = deepcopy(receipt_payload)
    result["_warnings"] = final_warnings
    result["_pass_count"] = len(pass_history)
    result["_pass_history"] = pass_history
    result["_model"] = model
    result["_pipeline_version"] = _PIPELINE_VERSION
    line_item_warnings = [w for w in final_warnings if "Line " in w]
    result["_line_items_reliable"] = len(line_item_warnings) == 0
    if ocr_confidence is not None:
        result["_ocr_confidence"] = round(ocr_confidence, 4)
    if llm_confidence is not None:
        result["_llm_confidence"] = llm_confidence
    if ocr_source is not None:
        result["_ocr_source"] = ocr_source
    if ocr_retried is not None:
        result["_ocr_retried"] = ocr_retried
    if ocr_retry_reason is not None:
        result["_ocr_retry_reason"] = ocr_retry_reason
    if ocr_text is not None:
        result["_ocr_text"] = ocr_text
    if debug and trace:
        result["_debug_dir"] = str(trace.debug_dir)
        result["_trace"] = trace.summary()
    if debug and mutation_trace:
        result["_receipt_mutation_trace"] = mutation_trace
    return result


def process_document(
    file_path: Path,
    model: str = DEFAULT_MODEL,
    debug: bool = False,
    passes: int = 1,
    ocr_engine=None,
    apply_user_rules: bool = True,
    skip_ocr_cache: bool = False,
    **kwargs,
) -> dict:
    """Main pipeline. Uses Cloud Vision OCR + LLM extraction (OpenRouter or Ollama)."""
    file_path = Path(file_path)
    check_model_available(model)
    on_stage: StageCallback = kwargs.get("on_stage")

    # Track document processing
    from .usage import track_document
    track_document(file_path)

    trace = PipelineTrace()
    debug_dir: Path | None = None
    if debug:
        debug_dir = Path("debug") / file_path.stem
        debug_dir.mkdir(parents=True, exist_ok=True)
        trace.debug_dir = debug_dir

    # Step 1: Load
    _notify(on_stage, "load", "Loading image", 0.0)
    images = load_image(file_path)
    trace.log_step("original", image=images[0])

    # Plan event: declare known shape early so consumers can draw the step list.
    # Doc-type is unknown until classify; path is "tbd" here.
    _notify(
        on_stage, "plan", "Pipeline plan",
        0.02,
        payload=_build_plan_payload(page_count=len(images), pass_budget=passes),
    )

    # Digital PDF fast path
    if file_path.suffix.lower() == ".pdf":
        _notify(on_stage, "ocr", "Checking for digital text", 0.05)
        digital_text = try_extract_text_layer(str(file_path))
        if digital_text:
            digital_text = normalize_fullwidth(digital_text)
            trace.log_step("digital_text_extracted", data=digital_text)
            doc_type = detect_document_type(digital_text)
            _notify(
                on_stage, "classify", f"Detected: {doc_type}",
                0.35,
                payload=_build_classify_payload(doc_type, source="digital_pdf"),
            )

            if debug:
                assert debug_dir is not None
                (debug_dir / "03_ocr_bboxes.txt").write_text(
                    "SKIPPED: Digital PDF fast path — no OCR performed.")

            _notify(on_stage, "extract", "LLM extraction (digital PDF)", 0.40)
            extracted, pass_history = extract_with_verification(
                digital_text, model=model, passes=passes,
                validate_fn=validate_receipt, doc_type=doc_type,
                on_stage=on_stage,
            )

            if debug:
                for entry in pass_history:
                    n = entry["pass"]
                    trace.log_step(f"pass{n}_llm_response", data=entry["extraction"])
                    if entry["warnings"]:
                        trace.log_step(f"pass{n}_warnings", data="\n".join(entry["warnings"]))

            llm_conf_pdf = extracted.pop("_confidence", None)

            # Location resolution for PDF path
            if "error" not in extracted and _location_needs_resolution(extracted.get("location"), digital_text):
                resolved, _loc_warn = _resolve_location(extracted, digital_text, model)
                if resolved:
                    extracted["location"] = resolved
            _trim_purchase_store_metadata_location(extracted, digital_text)

            try:
                receipt = Receipt(**extracted)
            except Exception:
                receipt = Receipt()
            _notify(on_stage, "validate", _build_validate_detail(extracted), 0.95)
            final_warnings = validate_receipt(receipt)
            for w in receipt._soft_warnings:
                if w not in final_warnings:
                    final_warnings.append(w)

            if debug:
                assert debug_dir is not None
                (debug_dir / "10_field_overlay.txt").write_text(
                    "SKIPPED: Digital PDF fast path — no OCR bounding boxes available.")
                (debug_dir / "pipeline_trace.txt").write_text(trace.summary())

            receipt_payload = _prepare_receipt_output_payload(receipt)
            result = _build_result(receipt_payload, final_warnings, pass_history, model, debug=debug, trace=trace,
                                   ocr_confidence=1.0, llm_confidence=llm_conf_pdf)
            if apply_user_rules:
                result = _apply_user_rules(result)
            _notify(on_stage, "done", "Complete", 1.0)
            return result

    # Step 2: Init OCR engine
    if ocr_engine is None:
        ocr_engine = init_cloud_vision()

    # Step 3: OCR per page, concatenate
    _notify(on_stage, "ocr", "Running OCR", 0.05)
    all_ocr_results: list[OCRResult] = []
    text_parts = []

    n_pages = max(1, len(images))
    _OCR_BAND_START, _OCR_BAND_END = 0.05, 0.30
    _OCR_BAND = _OCR_BAND_END - _OCR_BAND_START

    rotations = (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE)

    for i, page_img in enumerate(images):
        page_start = _OCR_BAND_START + (i / n_pages) * _OCR_BAND
        page_end = _OCR_BAND_START + ((i + 1) / n_pages) * _OCR_BAND
        if n_pages > 1:
            _notify(on_stage, "ocr", f"OCR page {i+1} of {n_pages}", page_start)
        else:
            _notify(on_stage, "ocr", "Running OCR", page_start)

        ocr_result = run_cloud_vision(page_img, ocr_engine, skip_cache=skip_ocr_cache)
        all_ocr_results.append(ocr_result)
        blocks = ocr_result.blocks

        if len(blocks) < 3:
            # Try all rotations (90°, 180°, 270°), pick best by confidence.
            _notify(
                on_stage, "warn",
                f"OCR returned {len(blocks)} blocks — retrying with rotations",
                page_start,
                payload={
                    "reason": "low_block_count",
                    "page": i + 1,
                    "block_count": len(blocks),
                },
            )
            best_result = ocr_result
            best_conf = compute_ocr_confidence(blocks) if blocks else 0.0
            for r_idx, rotation in enumerate(rotations):
                # Distribute sub-progress within this page's slot, leaving room
                # for the post-rotation work.
                sub_progress = page_start + ((r_idx + 1) / (len(rotations) + 1)) * (page_end - page_start)
                _notify(
                    on_stage, "ocr",
                    f"Page {i+1} retry rotation {r_idx+1}/{len(rotations)}",
                    sub_progress,
                )
                rotated = cv2.rotate(page_img, rotation)
                rot_result = run_cloud_vision(rotated, ocr_engine, skip_cache=skip_ocr_cache)
                rot_conf = compute_ocr_confidence(rot_result.blocks) if rot_result.blocks else 0.0
                if len(rot_result.blocks) > len(best_result.blocks) or rot_conf > best_conf:
                    best_result = rot_result
                    best_conf = rot_conf
                if best_conf >= 0.85:
                    break  # Good enough, stop early
            if len(best_result.blocks) > len(blocks):
                ocr_result = best_result
                blocks = ocr_result.blocks
                all_ocr_results[-1] = ocr_result

        if debug:
            assert debug_dir is not None
            draw_ocr_bboxes(page_img, blocks, debug_dir / f"03_page{i+1}_ocr_bboxes.png")

        page_text = blocks_to_structured_text(blocks)
        if i > 0:
            text_parts.append(f"--- PAGE {i+1} ---")
        text_parts.append(page_text)

    unified_text = "\n".join(text_parts)
    unified_text = normalize_fullwidth(unified_text)
    raw_text = unified_text  # Preserve pre-barcode-stripped text
    unified_text = strip_barcode_lines(unified_text)

    # Compute aggregate OCR confidence
    _notify(on_stage, "normalize", "Processing OCR text", 0.30)
    all_blocks_flat = [b for r in all_ocr_results for b in r.blocks]
    ocr_conf = compute_ocr_confidence(all_blocks_flat)

    # Detect document type
    doc_type = detect_document_type(unified_text)
    _notify(
        on_stage, "classify", f"Detected: {doc_type}",
        0.35,
        payload=_build_classify_payload(doc_type, source="ocr"),
    )

    if not unified_text.strip():
        return {
            "_error": "OCR produced no text.",
            "_warnings": [], "_pass_count": 0, "_model": model,
            "_pipeline_version": _PIPELINE_VERSION, "_line_items_reliable": False,
        }

    trace.log_step("ocr_grouped", data=unified_text)

    # Step 4–5: LLM extraction → post-processing → validation (shared path)
    _notify(on_stage, "extract", "LLM extraction", 0.40)
    all_layout_blocks = []
    for page_idx, ocr_result in enumerate(all_ocr_results):
        for block in getattr(ocr_result, "layout_blocks", []):
            block_with_page = dict(block)
            block_with_page["page"] = page_idx
            all_layout_blocks.append(block_with_page)
    receipt_mutation_trace: list[dict] | None = [] if debug else None
    extracted, pass_history, final_warnings = _run_extraction_pipeline(
        unified_text=unified_text, raw_text=raw_text,
        ocr_conf=ocr_conf, doc_type=doc_type,
        model=model, passes=passes,
        ocr_layout_blocks=all_layout_blocks,
        on_stage=on_stage,
        mutation_trace=receipt_mutation_trace,
    )

    if "_error" in extracted:
        extracted.update({"_warnings": [], "_pass_count": 0, "_model": model,
                          "_pipeline_version": _PIPELINE_VERSION, "_line_items_reliable": False})
        return extracted

    if debug:
        for entry in pass_history:
            n = entry["pass"]
            trace.log_step(f"pass{n}_llm_response", data=entry["extraction"])
            if entry["warnings"]:
                trace.log_step(f"pass{n}_warnings", data="\n".join(entry["warnings"]))

    # Compute post-hoc confidence from validation results
    posthoc_conf = _compute_posthoc_confidence(extracted, final_warnings)

    if debug and images:
        assert debug_dir is not None
        draw_field_overlay(images[0], all_ocr_results[0].blocks, extracted, debug_dir / "10_field_overlay.png")
        (debug_dir / "pipeline_trace.txt").write_text(trace.summary())

    # Aggregate OCR metadata from first page result
    try:
        receipt = Receipt(**extracted)
    except Exception:
        receipt = Receipt()
    primary_ocr = all_ocr_results[0] if all_ocr_results else None
    repair_ocr_text = primary_ocr.chosen_text if primary_ocr else None
    receipt_payload = _prepare_receipt_output_payload(
        receipt,
        repair_ocr_text,
        mutation_trace=receipt_mutation_trace,
    )
    result = _build_result(
        receipt_payload, final_warnings, pass_history, model, debug=debug, trace=trace,
        ocr_confidence=ocr_conf, llm_confidence=posthoc_conf,
        ocr_source=primary_ocr.source if primary_ocr else None,
        ocr_retried=primary_ocr.retried if primary_ocr else None,
        ocr_retry_reason=primary_ocr.retry_reason if primary_ocr else None,
        ocr_text=repair_ocr_text,
        mutation_trace=receipt_mutation_trace,
    )
    if apply_user_rules:
        result = _apply_user_rules(result)
    _notify(on_stage, "done", "Complete", 1.0)
    return result


def _receipt_items_target_gap(extracted: dict) -> float | None:
    items = extracted.get("line_items") or []
    if not items:
        return None
    items_sum = sum(
        item.get("total", 0)
        for item in items
        if isinstance(item, dict)
    )
    targets: list[float] = []
    subtotal = extracted.get("subtotal")
    total = extracted.get("total")
    taxes = extracted.get("taxes") or []
    tax_sum = sum(
        tax.get("amount", 0)
        for tax in taxes
        if isinstance(tax, dict) and tax.get("amount") is not None
    )
    canonical_subtotal = None
    if total and tax_sum:
        canonical_subtotal = float(total) - float(tax_sum)
    if subtotal is not None:
        if canonical_subtotal is None or abs(float(subtotal) - canonical_subtotal) <= 2:
            targets.append(float(subtotal))
    if total is not None:
        targets.append(float(total))
    if canonical_subtotal is not None:
        targets.append(canonical_subtotal)
    if not targets:
        return None
    return min(abs(items_sum - target) for target in targets)


def _receipt_printed_tax_gap(extracted: dict, unified_text: str) -> float:
    text = re.sub(r'\s+', ' ', unified_text or "")
    blocks: dict[str, float] = {}
    for m in re.finditer(
        r'\(\s*(\d{2})%対象\s*¥?\s*[\d,]+\s*内税\s*¥?\s*([\d,]+)\s*\)',
        text,
    ):
        blocks[f"{int(m.group(1))}%"] = float(m.group(2).replace(',', ''))
    for m in re.finditer(
        r'\(\s*(\d{2})%対象\s*¥?\s*([\d,]+)\s*\)\s*¥?\s*([\d,]+)\s*内税',
        text,
    ):
        rate = f"{int(m.group(1))}%"
        if rate in blocks:
            continue
        amount = float(m.group(2).replace(',', ''))
        base = float(m.group(3).replace(',', ''))
        try:
            rate_pct = float(rate.rstrip('%')) / 100.0
        except ValueError:
            continue
        expected = round(base * rate_pct / (1 + rate_pct))
        if 0 < amount < base and abs(amount - expected) <= 2:
            blocks[rate] = amount
    if not blocks:
        return 0.0
    taxes_by_rate = {
        tax.get("rate"): float(tax.get("amount") or 0)
        for tax in (extracted.get("taxes") or [])
        if isinstance(tax, dict)
    }
    gap = 0.0
    for rate, amount in blocks.items():
        if rate not in taxes_by_rate:
            gap += 1_000_000.0
        else:
            gap += abs(taxes_by_rate[rate] - amount)
    return gap


def _receipt_candidate_score(extracted: dict, warnings: list[str], unified_text: str = "") -> tuple:
    gap = _receipt_items_target_gap(extracted)
    gap_value = 1_000_000.0 if gap is None else float(gap)
    tax_gap = _receipt_printed_tax_gap(extracted, unified_text)
    item_warning_count = sum(
        1 for warning in warnings
        if "line items" in warning or "Items sum" in warning
    )
    item_count = len([
        item for item in (extracted.get("line_items") or [])
        if isinstance(item, dict)
    ])
    return (
        gap_value > 2,
        gap_value,
        tax_gap > 2,
        tax_gap,
        item_warning_count,
        len(warnings),
        -item_count,
    )


def _select_receipt_postprocessed_candidate(
    extracted: dict,
    pass_history: list[dict],
    unified_text: str,
    ocr_conf: float,
    ocr_totals: dict,
    model: str,
    ocr_layout_blocks: list[dict] | None,
    mutation_trace: list[dict] | None = None,
) -> dict:
    """Post-process all captured receipt candidates and keep the cleanest one.

    Raw LLM retry scoring can miss candidates that deterministic post-processing
    repairs cleanly. This selector scores the post-processed reality while still
    preserving each history entry's raw extraction for diagnostics.
    """
    candidate_refs: list[tuple[int | None, dict]] = [(None, extracted)]
    for idx, entry in enumerate(pass_history):
        candidate = entry.get("extraction")
        if not candidate or not isinstance(candidate, dict) or "error" in candidate:
            continue
        candidate_refs.append((idx, candidate))

    best: tuple[tuple, int, dict, list[str], int | None, list[dict] | None] | None = None
    for order, (history_idx, candidate) in enumerate(candidate_refs):
        postprocessed = deepcopy(candidate)
        llm_conf = postprocessed.get("_confidence")
        candidate_trace: list[dict] | None = [] if mutation_trace is not None else None
        postprocessed = postprocess_receipt(
            postprocessed,
            unified_text,
            ocr_conf,
            deepcopy(ocr_totals),
            llm_conf,
            model,
            ocr_layout_blocks=ocr_layout_blocks,
            mutation_trace=candidate_trace,
        )
        try:
            receipt = Receipt(**postprocessed)
            warnings = validate_receipt(receipt)
            for warning in receipt._soft_warnings:
                if warning not in warnings:
                    warnings.append(warning)
        except Exception as exc:
            warnings = [f"Schema validation failed after postprocess: {exc}"]
        score = _receipt_candidate_score(postprocessed, warnings, unified_text)

        if history_idx is not None:
            entry = pass_history[history_idx]
            entry["postprocess_items_sum_gap"] = _receipt_items_target_gap(postprocessed)
            entry["postprocess_warning_count"] = len(warnings)
            entry["postprocess_warnings"] = warnings
            entry["postprocess_selected"] = False

        ranked = (score, order, postprocessed, warnings, history_idx, candidate_trace)
        if best is None or ranked[:2] < best[:2]:
            best = ranked

    if best is None:
        return extracted

    _score, _order, best_extracted, _warnings, best_history_idx, best_trace = best
    if best_history_idx is not None:
        pass_history[best_history_idx]["postprocess_selected"] = True
    if mutation_trace is not None and best_trace:
        mutation_trace.extend(best_trace)
    return best_extracted


def _run_extraction_pipeline(
    unified_text: str,
    raw_text: str,
    ocr_conf: float,
    doc_type: str,
    model: str,
    passes: int,
    ocr_layout_blocks: list[dict] | None = None,
    on_stage: StageCallback = None,
    mutation_trace: list[dict] | None = None,
) -> tuple[dict, list[dict], list[str]]:
    """Shared extraction logic: LLM extraction → post-processing → location → validation.

    Used by both process_document() and process_ocr_text() to avoid code
    duplication and ensure fixes are applied consistently.

    Returns (extracted_dict, pass_history, final_warnings).
    """
    # Receipt-specific pre-processing
    ocr_totals = {}
    if doc_type == "receipt":
        # Interleave label-value column splits in the totals zone BEFORE
        # extracting financial totals — otherwise the line-by-line walk
        # picks up sibling-label values (お預り) instead of 合計's own value.
        unified_text = rejoin_totals_label_value_columns(unified_text)
        ocr_totals = extract_financial_totals(unified_text)
        # Strip bonus-point lines BEFORE rejoin so item↔price column matching
        # isn't disrupted by stray loyalty-point fragments.
        unified_text = strip_bonus_point_lines(unified_text)
        unified_text = join_split_qty_details(unified_text)
        # strip_banner_lines disabled — even with empty placeholders, line
        # changes affect LLM extraction (position-sensitive). Banner-line
        # phantoms are filtered at item level via _drop_banner_phantom_items.
        unified_text = rejoin_price_lines(unified_text)
        unified_text = _shift_misaligned_inline_prices(unified_text)
        unified_text = clean_handwritten_ocr(unified_text, ocr_confidence=ocr_conf)

    if not unified_text.strip():
        return (
            {"_error": "OCR text is empty."},
            [],
            [],
        )

    # LLM extraction with verification — emits per-pass beats internally.
    extracted, pass_history = extract_with_verification(
        unified_text, model=model, passes=passes,
        validate_fn=validate_receipt, doc_type=doc_type,
        on_stage=on_stage,
    )

    if "error" not in extracted:
        extracted["document_type"] = doc_type
        for fkey in ("total", "subtotal"):
            v = extracted.get(fkey)
            if v is not None:
                try:
                    extracted[fkey] = float(v)
                except (TypeError, ValueError):
                    extracted[fkey] = None

    # Document-type-specific post-processing
    _notify(on_stage, "postprocess", "Post-processing", 0.70)
    if doc_type == "receipt" and "error" not in extracted:
        extracted = _select_receipt_postprocessed_candidate(
            extracted,
            pass_history,
            unified_text,
            ocr_conf,
            ocr_totals,
            model,
            ocr_layout_blocks,
            mutation_trace=mutation_trace,
        )
    elif doc_type == "utility_bill" and "error" not in extracted:
        extracted = postprocess_utility_bill(extracted, unified_text)
    elif doc_type == "payment_slip" and "error" not in extracted:
        extracted = postprocess_payment_slip(extracted, unified_text, raw_text=raw_text)

    # Universal cash detection (all document types). For handwritten 領収証
    # forms, accept either explicit tender markers (お預り, 現金, 現計, お釣り)
    # OR the formal cash-receipt acknowledgement (上記正に領収/受領いたしました)
    # as long as nothing electronic or transfer-related contradicts it.
    # The acknowledgement phrase on a small handwritten receipt with no other
    # tender info is the standard Japanese signal that cash was tendered.
    _ELECTRONIC_PAY_RE = re.compile(
        r'クレジット|カード|PayPay|電子マネー|iD|QUICPay|Suica|WAON|nanaco|'
        r'PASMO|楽天Edy|LINE\s*Pay|au\s*PAY|d払い|メルペイ|交通系'
    )
    if "error" not in extracted and not extracted.get("payment_method"):
        is_handwritten = (
            re.search(r'領収証|領収書', unified_text)
            and not re.search(r'小計|合計|対象|税率', unified_text)
        )
        if is_handwritten:
            has_tender = bool(re.search(
                r'(?:お預り金?|お預かり)(?!票)|現金|現計|お釣り|釣銭',
                unified_text,
            ))
            has_acknowledgement = bool(
                re.search(r'上記正に\s*(?:領収|受領)', unified_text)
            )
            has_electronic = bool(_ELECTRONIC_PAY_RE.search(unified_text))
            has_transfer = bool(re.search(r'振込|振替|送金|口座', unified_text))
            if has_tender or (has_acknowledgement and not has_electronic and not has_transfer):
                extracted["payment_method"] = "cash"

    # Final cash fallback
    if "error" not in extracted and not extracted.get("payment_method"):
        has_tender_label = bool(re.search(r'(?:お預り金?|お預かり)(?!票)', unified_text))
        has_change_label_final = bool(re.search(r'釣', unified_text))
        has_electronic = bool(_ELECTRONIC_PAY_RE.search(unified_text))
        if has_tender_label and has_change_label_final and not has_electronic:
            extracted["payment_method"] = "cash"

    # Location: clear for utility bills and payment slips
    if "error" not in extracted and doc_type in ("utility_bill", "payment_slip"):
        extracted["location"] = None

    # Common post-processing
    if "error" not in extracted:
        if doc_type == "receipt":
            ocr_points = extract_points_used(unified_text)
            existing_points = extracted.get("points_used")
            if (
                ocr_points is not None
                and (existing_points is None or (ocr_points > 0 and float(existing_points or 0) == 0))
            ):
                extracted["points_used"] = ocr_points
        total = extracted.get("total")
        points = extracted.get("points_used")
        if total is not None:
            extracted["amount_paid"] = total - points if points else total

    # Strip _confidence if present
    extracted.pop("_confidence", None)

    # Location resolution (confidence-gated, receipts only)
    _notify(on_stage, "resolve_location", "Resolving location", 0.85)
    location_warnings: list[str] = []
    if "error" not in extracted and doc_type == "receipt" and _location_needs_resolution(extracted.get("location"), unified_text):
        # Check OCR evidence first — skip expensive LLM call if no evidence
        has_evidence = _location_has_ocr_evidence(
            extracted.get("location", ""), unified_text
        )
        has_clues = bool(LOCATION_CLUE_RE.search(unified_text))
        if has_evidence or has_clues:
            resolved, loc_warning = _resolve_location(extracted, unified_text, model)
            if resolved:
                extracted["location"] = resolved
            elif loc_warning:
                location_warnings.append(loc_warning)

    _trim_purchase_store_metadata_location(extracted, unified_text)
    _trim_store_in_store_header_location(extracted, unified_text)

    # Location validation: clear if no OCR evidence supports it
    if "error" not in extracted and doc_type == "receipt" and extracted.get("location"):
        if not _location_has_ocr_evidence(extracted["location"], unified_text):
            extracted["location"] = None
    if "error" not in extracted and doc_type == "receipt" and not extracted.get("location"):
        city_m = re.search(r'(宗像市)', unified_text)
        if city_m:
            extracted["location"] = city_m.group(1)

    # Expand truncated location when OCR has a more detailed address
    if "error" not in extracted and doc_type == "receipt" and extracted.get("location"):
        loc = extracted["location"]
        loc_norm = re.sub(r'\s+', '', loc)
        for line in unified_text.split('\n'):
            line_norm = re.sub(r'\s+', '', line.strip())
            if (len(line_norm) > len(loc_norm) and loc_norm in line_norm
                    and re.search(r'\d+-\d+|丁目|番地', line_norm)):
                extracted["location"] = line_norm
                break

    # Final validation — preview the result so consumers can flash it before "done".
    _notify(on_stage, "validate", _build_validate_detail(extracted), 0.95)
    try:
        receipt = Receipt(**extracted)
    except Exception:
        receipt = Receipt()
    final_warnings = validate_receipt(receipt)
    for w in receipt._soft_warnings:
        if w not in final_warnings:
            final_warnings.append(w)
    final_warnings.extend(location_warnings)

    return extracted, pass_history, final_warnings


def process_ocr_text(
    ocr_text: str,
    model: str = DEFAULT_MODEL,
    passes: int = 1,
    apply_user_rules: bool = True,
    on_stage: StageCallback = None,
    debug: bool = False,
) -> dict:
    """Run the pipeline from OCR text onwards (skip image loading + OCR).

    Used for:
    - Testing against saved OCR variants (regression tests)
    - Debugging with specific OCR output
    - Benchmarking LLM extraction independently of OCR variance
    """
    check_model_available(model)

    # Normalize text
    unified_text = normalize_fullwidth(ocr_text)
    unified_text = strip_barcode_lines(unified_text)
    doc_type = detect_document_type(unified_text)
    ocr_conf = 0.9  # default confidence for injected text

    if not unified_text.strip():
        return {
            "_error": "OCR text is empty.",
            "_warnings": [], "_pass_count": 0, "_model": model,
            "_pipeline_version": _PIPELINE_VERSION, "_line_items_reliable": False,
        }

    # process_ocr_text skips image loading and OCR — declare the plan upfront
    # with the doc_type already known, so consumers see the same event contract.
    _notify(
        on_stage, "plan", "Pipeline plan",
        0.02,
        payload=_build_plan_payload(page_count=1, pass_budget=passes, doc_type=doc_type),
    )
    _notify(
        on_stage, "classify", f"Detected: {doc_type}",
        0.35,
        payload=_build_classify_payload(doc_type, source="ocr_text_input"),
    )
    _notify(on_stage, "extract", "LLM extraction", 0.40)
    receipt_mutation_trace: list[dict] | None = [] if debug else None
    extracted, pass_history, final_warnings = _run_extraction_pipeline(
        unified_text=unified_text, raw_text=ocr_text,
        ocr_conf=ocr_conf, doc_type=doc_type,
        model=model, passes=passes,
        on_stage=on_stage,
        mutation_trace=receipt_mutation_trace,
    )

    if "_error" in extracted:
        extracted.update({"_warnings": [], "_pass_count": 0, "_model": model,
                          "_pipeline_version": _PIPELINE_VERSION, "_line_items_reliable": False})
        return extracted

    _notify(on_stage, "done", "Complete", 1.0)
    receipt = Receipt(**extracted) if "error" not in extracted else Receipt()
    receipt_payload = _prepare_receipt_output_payload(
        receipt,
        ocr_text,
        mutation_trace=receipt_mutation_trace,
    )
    result = _build_result(
        receipt_payload, final_warnings, pass_history, model,
        debug=debug,
        ocr_confidence=ocr_conf, ocr_source="injected",
        ocr_text=ocr_text,
        mutation_trace=receipt_mutation_trace,
    )
    if apply_user_rules:
        result = _apply_user_rules(result)
    return result


def process_batch(
    file_paths: list[Path],
    model: str = DEFAULT_MODEL,
    debug: bool = False,
    passes: int = 1,
    ocr_engine=None,
    apply_user_rules: bool = True,
    max_workers: int = 4,
    on_progress=None,
    on_stage: StageCallback = None,
) -> list[dict]:
    """Process multiple documents concurrently.

    Uses ThreadPoolExecutor for parallel LLM API calls (I/O-bound).
    The Cloud Vision client and OCR cache are thread-safe.
    """
    if not file_paths:
        return []

    check_model_available(model)
    if ocr_engine is None:
        ocr_engine = init_cloud_vision()

    total = len(file_paths)
    results: list[dict | None] = [None] * total
    start_time = time.perf_counter()

    def _process_one(idx: int, file_path: Path) -> tuple[int, dict]:
        try:
            result = process_document(
                file_path, model=model, debug=debug,
                passes=passes, ocr_engine=ocr_engine,
                apply_user_rules=apply_user_rules,
                on_stage=on_stage,
            )
            result["_file"] = str(file_path)
        except Exception as e:
            result = {"_file": str(file_path), "_error": str(e)}
        return idx, result

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_one, i, fp): i
            for i, fp in enumerate(file_paths)
        }
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
            completed += 1
            if on_progress:
                on_progress(file_paths[idx], result, completed, total)

    elapsed = time.perf_counter() - start_time
    for r in results:
        if r:
            r["_batch_total_s"] = round(elapsed, 2)
            r["_batch_workers"] = max_workers

    return results  # type: ignore[return-value]
