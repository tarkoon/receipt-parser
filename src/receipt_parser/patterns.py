"""patterns.py — Centralized regex patterns and confidence routing constants.

Single source of truth for all regex patterns used across pipeline modules,
plus confidence threshold constants and routing functions.
"""

import re


# ── Document Type Detection ────────────────────────────────────────

UTILITY_BILL_KEYWORDS = re.compile(
    r'検針|使用量|m3|kWh|ガス料金|水道料金|電気料金|'
    r'ご請求額|引落予定|メーター|基本料金|下水道使用料'
)

PAYMENT_SLIP_KEYWORDS = re.compile(
    r'払込票|振込.*請求書|振込兼|受領証.*払込|'
    r'依頼人|受取人|コンビニ収納|払込金受領書'
)

RECEIPT_KEYWORDS = re.compile(r'小計|合計|レジ')


# ── Yen Extraction ─────────────────────────────────────────────────

# Match ¥ or ￥ prefix, or 円 suffix amounts
YEN_INLINE = re.compile(r'[¥￥]\s*([\d,]+)|(?<!\d)([\d,]+)\s*円')

# Suffix chars allowed after ¥ amounts: closing parens + JP tax rate markers
YEN_SUFFIX = r'[)）軽※X除]'


# ── Location ───────────────────────────────────────────────────────

ADMIN_SUFFIX_RE = re.compile(r'[市区町村]')

LOCATION_CLUE_RE = re.compile(
    r'[\w\u3000-\u9fff]+店'           # X店, X支店, X赤間店, etc.
    r'|[\w\u3000-\u9fff]+モール'       # Xモール (mall)
    r'|[都道府県市区町村郡]'             # Address text with admin units
    r'|〒\d{3}'                        # Postal code
)


# ── Japanese Era Constants ─────────────────────────────────────────

# Era name → base year (era year 1 = base + 1)
ERA_TABLE = {
    "令和": 2018,   # 令和1年 = 2019
    "平成": 1988,   # 平成1年 = 1989
    "昭和": 1925,   # 昭和1年 = 1926
}
DEFAULT_ERA_BASE = 2018  # Assume 令和 when era name is not found


def era_to_western_year(era_year: int, era_name: str | None = None) -> int | None:
    """Convert Japanese era year to western year.

    Args:
        era_year: The year within the era (e.g. 8 for 令和8年)
        era_name: The era name if detected from OCR text (e.g. "令和", "平成")

    Returns:
        Western year (e.g. 2026) or None if era_year is invalid.

    When no era name is provided, uses a plausibility heuristic:
    - era_year <= 8: assume 令和 (produces 2019-2026, current era)
    - era_year > 8: assume 平成 if result is within last 30 years
    """
    if era_year < 1 or era_year > 99:
        return None
    if era_name:
        base = ERA_TABLE.get(era_name, DEFAULT_ERA_BASE)
        return base + era_year
    # No era name — disambiguate using plausibility
    reiwa_year = 2018 + era_year
    if reiwa_year <= 2026:
        return reiwa_year  # Plausible 令和 date (current era, not in the future)
    # era_year > 8: 令和 would be future; try 平成
    heisei_year = 1988 + era_year
    if 1996 <= heisei_year <= 2019:
        return heisei_year  # Plausible 平成 date (within last ~30 years)
    return reiwa_year  # Fall back to 令和


# ── Confidence Router ──────────────────────────────────────────────

HIGH_OCR_CONFIDENCE = 0.85
HIGH_LLM_CONFIDENCE = 0.7
LOW_LLM_CONFIDENCE = 0.5

# Financial fields always get overridden by OCR evidence when OCR is reliable,
# because LLM self-reported confidence is not calibrated for numeric accuracy.
FINANCIAL_FIELDS = {"total", "subtotal", "taxes", "points_used"}


def should_override_field(field: str, ocr_conf: float, llm_conf: dict | None) -> bool:
    """Decide whether regex should override LLM output for a given field.

    For financial fields (total, subtotal, taxes): always override when OCR
    is reliable — LLM confidence is unreliable for numeric accuracy.

    For other fields: override only when LLM confidence is low.
    """
    if ocr_conf < HIGH_OCR_CONFIDENCE:
        return False  # OCR too unreliable for regex extraction
    if field in FINANCIAL_FIELDS:
        return True  # Always override financial fields with OCR evidence
    if llm_conf is None:
        return True  # No confidence info — fall back to legacy behavior
    field_conf = llm_conf.get(field, 0.0)
    return field_conf < LOW_LLM_CONFIDENCE


def should_use_regex_as_validation(field: str, ocr_conf: float, llm_conf: dict | None) -> bool:
    """Use regex as a validation signal (warn on disagreement) but don't override."""
    if ocr_conf < HIGH_OCR_CONFIDENCE:
        return False
    if field in FINANCIAL_FIELDS:
        return False  # Financial fields get overridden, not just validated
    if llm_conf is None:
        return False
    field_conf = llm_conf.get(field, 0.0)
    return LOW_LLM_CONFIDENCE <= field_conf < HIGH_LLM_CONFIDENCE
