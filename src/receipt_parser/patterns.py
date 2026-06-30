"""patterns.py — Centralized regex patterns and confidence routing constants.

Single source of truth for all regex patterns used across pipeline modules,
plus confidence threshold constants and routing functions.
"""

import re


UTILITY_BILL_KEYWORDS = re.compile(
    r'検針|使用量|m3|kWh|ガス料金|水道料金|電気料金|'
    r'ご請求額|引落予定|メーター|基本料金|下水道使用料'
)

PAYMENT_SLIP_KEYWORDS = re.compile(
    r'払込票|振込.*請求書|振込兼|受領証.*払込|'
    r'依頼人|受取人|コンビニ収納|払込金受領書'
)

RECEIPT_KEYWORDS = re.compile(r'小計|合計|レジ')


# Match ¥ or ￥ prefix, or 円 suffix amounts
YEN_INLINE = re.compile(r'[¥￥]\s*([\d,]+)|(?<!\d)([\d,]+)\s*円')

# Suffix chars allowed after ¥ amounts: closing parens + JP tax rate markers
YEN_SUFFIX = r'[)）軽※X除]'


ADMIN_SUFFIX_RE = re.compile(r'[市区町村]')

LOCATION_CLUE_RE = re.compile(
    r'[\w\u3000-\u9fff]+店'           # X店, X支店, X赤間店, etc.
    r'|[\w\u3000-\u9fff]+モール'       # Xモール (mall)
    r'|[都道府県市区町村郡]'             # Address text with admin units
    r'|〒\d{3}'                        # Postal code
    r'|(?:TEL|電話|☎)\s*[:\s]?\s*0\d'  # Labeled phone number
    r'|[（(]\s*0\d{1,4}\s*[）)]\s*\d{1,4}'  # Parenthesized area code, e.g. (0940) 38-0130
    r'|^0\d{1,4}-\d{1,4}-\d{2,4}$'    # Bare phone number on its own line
, re.MULTILINE)


_COMPANY_SUFFIX_RE = re.compile(r'有限会社|株式会社|㈱|㈲|合同会社')
_DECORATIVE_RE = re.compile(r'^[☆★\-=\*\s・♪♫]+$')
_HEADER_PHONE_MERCHANT_RE = re.compile(
    r'^\s*(?P<merchant>.{2,40}?)\s*'
    r'(?:[（(]\s*0\d{1,4}\s*[）)]|0\d{1,4}[-\s]\d)'
)
_OFFICIAL_AUTHORITY_HEADER_RE = re.compile(
    r'^(?:[A-Z&]\s+)?(.{1,30}(?:市役所|区役所|町役場|村役場|役場|県庁|都庁|府庁))$'
)
_OFFICIAL_DEPARTMENT_LINE_RE = re.compile(r'^.{1,16}(?:課|係|室|部|局|センター)$')

_SERVICE_FEE_DESCRIPTION_RE = re.compile(r'通行|利用|サービス|施設|駐車|入場|手数|料金')
_SERVICE_TAX_RATE_EVIDENCE_RE = re.compile(
    r'(?:消費税率|税率).{0,20}10\s*[%％]|10\s*[%％].{0,20}(?:内税|税込|消費税)'
)
_ADMIN_FEE_DESCRIPTION_RE = re.compile(r'証明|住民票|戸籍|印鑑|所得|課税|納税|手数料|電申')

_FUEL_KEYWORDS = ('ガソリン', 'レギュラー', 'ハイオク', '軽油', 'ENEOS', '出光', 'コスモ')

_SKIP_PRICE_LINE = re.compile(r'対象|内税|外税|合計|小計|消費税|お預り|お釣|お預かり')
_GENERIC_DESC_MARKERS = frozenset({
    '消耗', '食料品', '飲料', '雑貨', '文具', '日配', '冷蔵', '冷凍',
    '青果', '惣菜', '加工', '生活', '化粧品', '医薬', 'お菓子', '酒類',
    '日用品', '特', '軽',
})
_JUNK_DESC_RE = re.compile(
    r'^(?:電話|TEL|☎)\s*[:：]?\s*0?\d'  # phone numbers ('電話: 078-...')
    r'|^〒\s*\d{3}'                       # postal code
    r'|^\d{8,}'                            # bare digit run (barcode)
    r'|(?:Code128|操作)?割引\d*'             # discount operation label
    r'|^\d+\s*[xX×]\s*[#＃]?\s*\d+$'      # unit-rate notation '23 X #199'
    r'|^\*?\s*\d[\d,]*\s*\(\s*\d+\s*[個コ点]\s*\)'  # '*770 (1コ)' price+qty notation
)
_HEADER_LINE_RE = re.compile(
    r'(?:電話|TEL|☎)|登録番号|担当|レシートNo|レジ\s*\d|キャッシャ|'
    r'店舗|発行|取引(?:コード|No)|POS\s*No|〒\s*\d{3}|'
    r'(?:\d{2,4}\s*)?年\s*\d{1,2}\s*月\s*\d{1,2}\s*日|'  # date (year optional)
    r'\(\s*[月火水木金土日]\s*\)|'                          # day-of-week marker
    r'\d{1,2}\s*[:時]\s*\d{2}\s*分?(?!\d)|'                # HH:MM or HH時MM分
    r'\d+\s*番\s*\d+\s*号|'
    r'[県府][　-鿿]+[市区町村]|'                            # address
    r'^[一-鿿]{2,8}店$|'                                      # branch name
    r'\bNo\.?\s*\d{2,}|'                                      # No. 012
    r'領\s*収\s*証'                                           # 領収証
)
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
    r'の商品です|まとめ値引|'
    r'^[A-Z]\s*[:：]\s*\d+\s*[個コ点]|'
    r'^\s*消費税等?\s*$'
)

_QTY_DETAIL_DESC_RE = re.compile(
    r'^\(?\s*'
    r'(?:'
    r'\d+\s*[コ個点]\s*[xX×]\s*(?:単|@)?\s*\d[\d,]*'   # "2個 X70", "2個 X 単70"
    r'|(?:単|@)\s*\d[\d,]*\s*[xX×]\s*\d+\s*[コ個点]'    # "単70 × 2個", "@70x2個"
    r')'
    r'\)?\s*$'
)
_OCR_TRAILING_PRICE_RE = re.compile(
    r'(?:^|[\s(（])([¥￥]?\s*\d[\d,]*)\s*(?:[%％][*※除軽]|[*※除軽])?\s*$'
)
_OCR_ZONE_END_RE = re.compile(
    r'^(小計|合計|現計|外税|内税|消費税|お預り|お釣り|釣銭|WAON|クレジット|お会計)'
)
_OCR_QTY_NOTATION_RE = re.compile(
    r'(?:'
    r'\d+\s*[コ個点]\s*[xX×Ⅹ]\s*(?:単|@)?\s*\d|'
    r'(?:単|@)\s*\d[\d,]*\s*[xX×Ⅹ]\s*\d+\s*[コ個点]'
    r')'
)
_BAG_DESC_RE = re.compile(
    r'レジ[ブフ]クロ|レジ袋|有料レジ袋|食品ポリ袋|ポリ袋|ショッピングバッグ|紙袋|バイオ.*袋|フクロHK'
)
_FOOD_DESC_RE = re.compile(
    r'バナナ|だし|餃子|肉|ミンチ|キャベツ|にんじん|大根|ハム|コマツナ|春雨|'
    r'ピーチ|ねぎ|オオバ|パン|ホシイモ|米|牛|豚|鶏|チキン|弁当|おにぎり|'
    r'茶|ココア|コーヒー|天然水|オイル不使用|食品'
)


def _is_service_fee_description(description: str | None) -> bool:
    return bool(description and _SERVICE_FEE_DESCRIPTION_RE.search(description))


def _has_service_inclusive_tax_evidence(unified_text: str) -> bool:
    if _SERVICE_TAX_RATE_EVIDENCE_RE.search(unified_text):
        return True
    return bool(re.search(r'消費税(?:等)?[^。.\n]{0,20}含', unified_text))


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
