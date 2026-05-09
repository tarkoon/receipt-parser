"""schema.py — Single source of truth for receipt data extraction.

All extraction fields, prompt generation, validation rules, and debug
overlay colors are defined here. To add a new field, see the
EXTENDING THE SCHEMA section in the build plan.
"""

from __future__ import annotations
from pydantic import BaseModel, PrivateAttr, field_validator, model_validator
from typing import Literal, Optional
import json
import re


class FieldMeta:
    """Metadata for a single extractable field."""
    def __init__(
        self,
        name: str,
        debug_color_bgr: tuple[int, int, int],
        prompt_hint: str | None = None,
        extraction_aliases: list[str] | None = None,
        doc_types: list[str] | None = None,
    ):
        self.name = name
        self.debug_color_bgr = debug_color_bgr
        self.prompt_hint = prompt_hint
        self.extraction_aliases = extraction_aliases or []
        self.doc_types = doc_types or ["receipt", "utility_bill", "payment_slip"]


FIELD_REGISTRY: list[FieldMeta] = [
    # Common fields (all types)
    FieldMeta(
        name="document_type",
        debug_color_bgr=(255, 255, 255),
        prompt_hint="Classify as 'receipt' for store purchases, 'utility_bill' for gas/water/electric bills, or 'payment_slip' for bank transfer or convenience store payment slips.",
    ),
    FieldMeta(
        name="merchant",
        debug_color_bgr=(255, 165, 0),
        prompt_hint="Consumer-facing BRAND name only (see merchant rules above). If a subtitle describes the business type (e.g. '自家製生パスタの店') but a proper name also appears (e.g. 'チャオ'), use the proper name. For handwritten receipts (領収証), the merchant name is near the stamp/seal at the bottom. For payment slips, this is the 受取人. For dual English+Japanese names, use the most prominent form.",
        extraction_aliases=["店名", "store", "shop", "受取人"],
    ),
    FieldMeta(
        name="date",
        debug_color_bgr=(0, 255, 0),
        prompt_hint="Parse Japanese dates: 令和8年=2026, 令和7年=2025. Convert 2026年3月15日 to 2026-03-15. Always output as YYYY-MM-DD. For bills: use payment date if visible, else due date (支払期限, 引落予定日), else issue date.",
        extraction_aliases=["日付", "日時", "date", "支払期限"],
    ),
    FieldMeta(
        name="time",
        debug_color_bgr=(0, 220, 60),
        prompt_hint="Transaction time of day, 24-hour H:MM (no leading zero on hour, two-digit minute). Look immediately after the date — receipts usually print it as '2026年3月15日 (水) 17:05', '2026/3/15 17:05', or '13時30分'. Drop seconds (17:05:42 → 17:05). Convert 13時30分 → 13:30. Convert 09:32 → 9:32 (strip leading zero on the hour). Output null if no time appears, or if the only time-looking text is a business-hours range (営業時間 9:00~21:00) or a phone number. Do NOT use the time portion of a printer/issued timestamp on utility bills or payment slips unless it clearly marks the transaction.",
        extraction_aliases=["時刻", "time"],
        doc_types=["receipt"],
    ),
    FieldMeta(
        name="location",
        debug_color_bgr=(200, 200, 0),
        prompt_hint="City/ward level location. Use branch name to infer location (e.g. '赤間店' → 赤間 is in 宗像市, so output '宗像市赤間'; '八幡店' → 八幡 is in 北九州市, so output '北九州市八幡区'). Also use explicit address lines or shopping complex names. Output format: '宗像市赤間', '福岡市博多区', '北九州市八幡区'. Always include at least 市 or 区. Extract city/ward only, not full street address. Phone numbers alone are NOT sufficient — only use as supplementary hint. Output null if no location clues exist.",
        extraction_aliases=["住所", "address"],
        doc_types=["receipt"],
    ),
    FieldMeta(
        name="currency",
        debug_color_bgr=(180, 180, 180),
        prompt_hint="If the receipt uses ¥ or ￥ symbols, or is entirely in Japanese, currency is JPY. If it uses $ or is in English with no other currency indicator, currency is USD. Always output the three-letter ISO code.",
        extraction_aliases=["¥", "￥", "$", "円"],
    ),
    FieldMeta(
        name="total",
        debug_color_bgr=(0, 0, 255),
        prompt_hint="The final amount due. For utility bills this is ご請求額 or 引落予定額.",
        extraction_aliases=["合計", "total", "お会計", "ご請求額", "引落予定額"],
    ),
    FieldMeta(
        name="payment_method",
        debug_color_bgr=(128, 0, 128),
        prompt_hint="Must be one of: cash, credit, debit, bank_payment, WAON, or null. Use 'cash' only if お預り or 現計 is shown. Use 'bank_payment' for 口座引落, 口座振替, or 振込. Use null if no evidence on the document.",
        extraction_aliases=["支払", "payment"],
    ),
    FieldMeta(
        name="account_number",
        debug_color_bgr=(100, 100, 0),
        prompt_hint="Customer or account number for recurring bill tracking. Look for お客様番号 or similar.",
        extraction_aliases=["お客様番号", "口座番号"],
    ),
    FieldMeta(
        name="points_used",
        debug_color_bgr=(200, 100, 200),
        prompt_hint="Loyalty points applied as payment (d-point, WAON point, etc.). Output as a number. Look for ポイント利用, ポイント値引. Do NOT confuse with ポイント対象額 (eligible amount) or 獲得ポイント (points earned).",
        extraction_aliases=["ポイント利用", "ポイント値引"],
        doc_types=["receipt"],
    ),
    FieldMeta(
        name="amount_paid",
        debug_color_bgr=(200, 50, 50),
        prompt_hint="Actual out-of-pocket cost. Equals total minus points_used. If no points used, equals total.",
    ),

    # Receipt-specific fields
    FieldMeta(
        name="line_items",
        debug_color_bgr=(255, 255, 0),
        prompt_hint="Match description, qty, unit_price, and total per row. Default qty=1. Numbers followed by tax markers at line end ARE prices: 278※ means ¥278 at 8%, 3除 means ¥3 at 10%. Tax markers: ※/*/X → reduced 8%, 除 → standard 10%. A single OCR line may contain multiple items with prices — split them (e.g. '食品ポリ袋L3除日清チャック付328※' = two items: ¥3 and ¥328). '部門NNN' lines are department-coded items. See discount rules above.",
        doc_types=["receipt"],
    ),
    FieldMeta(
        name="subtotal",
        debug_color_bgr=(0, 255, 255),
        prompt_hint="ALWAYS the pre-tax base: subtotal = total - sum(taxes). For 内税 (tax-inclusive) receipts where 合計 already includes tax, subtotal is LESS than 合計 — compute subtotal = 合計 - 消費税. For 外税 receipts subtotal equals 小計 (printed pre-tax). Never set subtotal equal to total when there is a non-zero tax amount.",
        extraction_aliases=["小計", "subtotal"],
        doc_types=["receipt"],
    ),
    FieldMeta(
        name="taxes",
        debug_color_bgr=(255, 0, 255),
        prompt_hint="Output as a list of objects with 'rate' (string like '10%' or '8%'), 'label' (e.g. '内税'), and 'amount' (number). JP tax: 軽減税率/※ = reduced 8%, standard = 10%. Look for patterns like '10%内税対象' or '(10%内)' to determine the rate. 内税 means tax-inclusive, 外税 means tax-exclusive.",
        extraction_aliases=["税", "消費税", "tax", "内税", "外税"],
        doc_types=["receipt"],
    ),

    # Utility bill-specific fields
    FieldMeta(
        name="service_type",
        debug_color_bgr=(0, 200, 100),
        prompt_hint="Must be one of: gas, water, electric, sewage, internet, phone, or null.",
        extraction_aliases=["ガス", "水道", "電気", "下水道"],
        doc_types=["utility_bill"],
    ),
    FieldMeta(
        name="billing_period",
        debug_color_bgr=(100, 200, 200),
        prompt_hint="Output as an object with 'start' and 'end' in YYYY-MM-DD format. Look for 使用期間, ご利用期間, or derive from 前回検針日 to 今回検針日.",
        extraction_aliases=["使用期間", "検針日"],
        doc_types=["utility_bill"],
    ),
    FieldMeta(
        name="usage",
        debug_color_bgr=(200, 200, 100),
        prompt_hint="Output as an object with 'amount' (number), 'unit' (m3, kWh, or L), 'cost_per' (price per unit, e.g. ¥181/L — number or null), 'meter_previous' (number or null), 'meter_current' (number or null). Look for ご使用量, 指針, 単価.",
        extraction_aliases=["ご使用量", "使用量", "指針"],
        doc_types=["utility_bill"],
    ),

    # Payment slip-specific fields
    FieldMeta(
        name="payer",
        debug_color_bgr=(100, 0, 200),
        prompt_hint="The person or entity making the payment. Look for 依頼人, ご依頼人.",
        extraction_aliases=["依頼人", "ご依頼人"],
        doc_types=["payment_slip"],
    ),
    FieldMeta(
        name="payment_reference",
        debug_color_bgr=(0, 100, 200),
        prompt_hint="Tracking or reference number on the payment slip.",
        extraction_aliases=["手番", "収納番号"],
        doc_types=["payment_slip"],
    ),
]


def get_field_meta(name: str) -> FieldMeta | None:
    """Look up field metadata by name."""
    for f in FIELD_REGISTRY:
        if f.name == name:
            return f
    return None


def get_debug_color_map() -> dict[str, tuple[int, int, int]]:
    """Return {field_name: bgr_color} for all registered fields."""
    return {f.name: f.debug_color_bgr for f in FIELD_REGISTRY}


VALID_TAX_RATES = ("8%", "10%", "0%")
REDUCED_RATE = "8%"
STANDARD_RATE = "10%"
EXEMPT_RATE = "0%"

TaxCategoryType = Literal["8%", "10%", "0%"]


_TIME_RE_COLON = re.compile(r'^\s*(\d{1,2})\s*[:：]\s*(\d{1,2})(?:\s*[:：]\s*\d{1,2})?\s*$')
_TIME_RE_JP = re.compile(r'^\s*(\d{1,2})\s*時\s*(\d{1,2})\s*分?\s*$')
_TIME_RE_AMPM = re.compile(r'^\s*(午前|午後|AM|PM|am|pm)\s*(\d{1,2})\s*[:：時]\s*(\d{1,2})')


def _normalize_time(v) -> str | None:
    """Normalize a time string to 24-hour HH:MM. Return None if unparseable."""
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("null", "none"):
        return None

    ampm = _TIME_RE_AMPM.match(s)
    if ampm:
        marker, hh, mm = ampm.group(1), int(ampm.group(2)), int(ampm.group(3))
        if marker in ("午後", "PM", "pm") and hh < 12:
            hh += 12
        elif marker in ("午前", "AM", "am") and hh == 12:
            hh = 0
    else:
        m = _TIME_RE_COLON.match(s) or _TIME_RE_JP.match(s)
        if not m:
            return None
        hh, mm = int(m.group(1)), int(m.group(2))

    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return None
    # Hour: no leading zero ('9:32', not '09:32'). Minute: always two digits.
    return f"{hh}:{mm:02d}"


def _coerce_numeric(v) -> float | None:
    """Coerce a value to float, handling strings with commas."""
    if v is None:
        return None
    try:
        return float(str(v).replace(',', ''))
    except (TypeError, ValueError):
        return None


def _coerce_tax_category(v) -> str:
    """Coerce tax_category to a valid literal value."""
    if v in VALID_TAX_RATES:
        return v
    if v and "8" in str(v):
        return REDUCED_RATE
    if v and "10" in str(v):
        return STANDARD_RATE
    return EXEMPT_RATE


class LineItem(BaseModel):
    model_config = {"populate_by_name": True}

    description: str
    qty: float = 1
    unit_price: Optional[float] = None
    total: float
    tax_category: TaxCategoryType = EXEMPT_RATE
    discount: float = 0
    discount_rate: str = ""

    @field_validator("total", "unit_price", "qty", "discount", mode="before")
    @classmethod
    def coerce_numeric_fields(cls, v):
        if v is None:
            return v
        try:
            return float(str(v).replace(',', ''))
        except (TypeError, ValueError):
            return v

    @field_validator("tax_category", mode="before")
    @classmethod
    def coerce_tax_category(cls, v):
        return _coerce_tax_category(v)

    @field_validator("discount_rate", mode="before")
    @classmethod
    def coerce_discount_rate(cls, v):
        return "" if v is None else str(v)

    @field_validator("discount", mode="before")
    @classmethod
    def coerce_discount_default(cls, v):
        if v is None:
            return 0
        try:
            return float(str(v).replace(',', ''))
        except (TypeError, ValueError):
            return 0

    @field_validator("total")
    @classmethod
    def total_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("Line item total cannot be negative")
        return v


class TaxEntry(BaseModel):
    rate: str
    label: Optional[str] = None
    amount: float

    @field_validator("amount", mode="before")
    @classmethod
    def coerce_amount(cls, v):
        if v is None:
            return 0
        try:
            return float(str(v).replace(',', ''))
        except (TypeError, ValueError):
            return 0


class BillingPeriod(BaseModel):
    start: Optional[str] = None
    end: Optional[str] = None


class UsageData(BaseModel):
    amount: Optional[float] = None
    unit: Optional[str] = None
    cost_per: Optional[float] = None
    meter_previous: Optional[float] = None
    meter_current: Optional[float] = None


class Document(BaseModel):
    """Unified extraction model for all document types."""
    model_config = {"populate_by_name": True}

    # Common
    document_type: Literal["receipt", "utility_bill", "payment_slip"] = "receipt"
    merchant: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    location: Optional[str] = None
    currency: Optional[str] = None
    total: Optional[float] = None
    payment_method: Optional[str] = None
    account_number: Optional[str] = None
    points_used: Optional[float] = None
    amount_paid: Optional[float] = None

    # Receipt-specific
    line_items: list[LineItem] = []
    subtotal: Optional[float] = None
    taxes: list[TaxEntry] = []

    # Utility bill-specific
    service_type: Optional[str] = None
    billing_period: Optional[BillingPeriod] = None
    usage: Optional[UsageData] = None

    # Payment slip-specific
    payer: Optional[str] = None
    payment_reference: Optional[str] = None

    raw_text_summary: Optional[str] = None

    # Soft validation warnings collected during model construction (non-raising)
    _soft_warnings: list[str] = PrivateAttr(default_factory=list)

    @field_validator("total", "subtotal", "points_used", "amount_paid", mode="before")
    @classmethod
    def coerce_financial_fields(cls, v):
        if v is None or v == "null":
            return None
        try:
            return float(str(v).replace(',', ''))
        except (TypeError, ValueError):
            return None

    @field_validator("merchant", "date", "time", "location", "currency", "payment_method",
                     "account_number", "payer", "payment_reference",
                     "service_type", "raw_text_summary", mode="before")
    @classmethod
    def coerce_null_strings(cls, v):
        """Convert the string 'null' to actual None for optional string fields."""
        if v == "null" or v == "None":
            return None
        return v

    @model_validator(mode="before")
    @classmethod
    def handle_llm_aliases(cls, data):
        """Handle common LLM output mismatches before field validation."""
        if not isinstance(data, dict):
            return data

        # Strip _confidence (metadata, not a schema field)
        data.pop("_confidence", None)

        # Coerce line_items: handle 'quantity' → 'qty', 'name' → 'description'
        if "line_items" in data and isinstance(data["line_items"], list):
            fixed = []
            for item in data["line_items"]:
                if not isinstance(item, dict):
                    continue
                if "quantity" in item and "qty" not in item:
                    item["qty"] = item.pop("quantity")
                if "name" in item and "description" not in item:
                    item["description"] = item.pop("name")
                if not item.get("description"):
                    continue
                if "total" not in item or item["total"] is None:
                    continue
                # Drop negative-total items: usually a discount line the LLM
                # mis-extracted as a standalone item. Keeping it would trigger
                # LineItem.total_must_be_positive and reject the whole receipt.
                try:
                    if float(str(item["total"]).replace(',', '')) < 0:
                        continue
                except (TypeError, ValueError):
                    pass
                fixed.append(item)
            data["line_items"] = fixed

        # Coerce taxes from various formats
        taxes = data.get("taxes")
        if isinstance(taxes, (int, float)):
            data["taxes"] = [{"rate": "unknown", "label": None, "amount": taxes}]
        elif isinstance(taxes, dict):
            data["taxes"] = [taxes]
        elif taxes is None:
            data["taxes"] = []

        # Normalize time to 24-hour HH:MM (drop seconds, parse 13時30分)
        time_val = data.get("time")
        if time_val is not None and time_val != "":
            time_str = str(time_val).strip()
            normalized = _normalize_time(time_str)
            data["time"] = normalized  # may be None if unparseable

        # Fix Japanese era dates in LLM output
        date_val = data.get("date")
        if date_val:
            date_str = str(date_val)
            m = re.match(r'^(\d{4})-(\d{2})-(\d{2})$', date_str)
            if m:
                year = int(m.group(1))
                if year < 100:
                    data["date"] = f"{2018 + year:04d}-{m.group(2)}-{m.group(3)}"
                elif 2000 <= year <= 2018:
                    era_year = year - 2000
                    if 1 <= era_year <= 20:
                        data["date"] = f"{2018 + era_year:04d}-{m.group(2)}-{m.group(3)}"

        return data

    @model_validator(mode="after")
    def check_arithmetic(self) -> "Document":
        """Soft arithmetic validation — collects warnings without raising.

        Mirrors the checks in validation.py but stores them as _soft_warnings
        on the model instance for integration with instructor and pipeline.
        """
        warnings = self._soft_warnings

        # Line item math: qty * unit_price - discount ≈ total (±1)
        for i, item in enumerate(self.line_items):
            if item.unit_price is not None and item.qty:
                expected = item.qty * item.unit_price - item.discount
                if abs(expected - item.total) > 1:
                    warnings.append(
                        f"Line {i+1}: qty*price-discount={expected}, total={item.total}. "
                        f"Suggested: set total to {expected} or adjust qty/unit_price."
                    )

            # Discount rate consistency
            if item.discount_rate and item.discount > 0 and item.unit_price is not None and item.qty:
                rate_match = re.match(r'(\d+(?:\.\d+)?)', item.discount_rate)
                if rate_match:
                    rate_pct = float(rate_match.group(1)) / 100.0
                    expected_discount = round(item.unit_price * item.qty * rate_pct)
                    if abs(expected_discount - item.discount) > 2:
                        warnings.append(
                            f"Line {i+1}: discount_rate {item.discount_rate} "
                            f"implies ~{expected_discount}, but discount={item.discount}"
                        )

        # Sum of items matches subtotal (pre-tax items) or total (post-tax items)
        if self.subtotal is not None and self.line_items:
            items_sum = sum(item.total for item in self.line_items)
            items_match_subtotal = abs(items_sum - self.subtotal) <= 2
            items_match_total = (self.total is not None
                                 and abs(items_sum - self.total) <= 2)
            if not (items_match_subtotal or items_match_total):
                warnings.append(
                    f"Items sum {items_sum} != subtotal {self.subtotal} "
                    f"or total {self.total}"
                )

        # Universal: subtotal + tax_sum = total
        if self.total is not None and self.subtotal is not None and self.taxes:
            tax_sum = sum(t.amount for t in self.taxes)
            if abs((self.subtotal + tax_sum) - self.total) > 2:
                warnings.append(
                    f"Total {self.total} != subtotal {self.subtotal} + taxes {tax_sum}"
                )

        # Tax ratio cross-check: subtotal × known rate ≈ total
        if self.total is not None and self.subtotal is not None and self.taxes:
            tax_sum = sum(t.amount for t in self.taxes)
            if tax_sum > 0:
                known_rates = [0.08, 0.10]
                ratio_ok = any(
                    abs(self.subtotal * (1 + r) - self.total) <= 2 for r in known_rates
                ) or abs(self.subtotal + tax_sum - self.total) <= 2
                if not ratio_ok:
                    warnings.append(
                        f"Tax ratio: subtotal {self.subtotal} × known rate "
                        f"!= total {self.total}"
                    )

        # Tax rate membership
        for tax in self.taxes:
            rate_str = tax.rate.replace('%', '').strip()
            try:
                rate_val = float(rate_str)
                valid_rate_values = {float(r.replace('%', '')) for r in VALID_TAX_RATES}
                if rate_val not in valid_rate_values and rate_str != "unknown":
                    warnings.append(f"Unusual tax rate: {tax.rate}")
            except ValueError:
                pass

        return self


# Backward compatibility
Receipt = Document


def _build_field_hints(doc_type: str = "receipt") -> str:
    """Build the FIELD-SPECIFIC RULES block from the registry."""
    hints = []
    for f in FIELD_REGISTRY:
        if doc_type not in f.doc_types:
            continue
        parts = []
        if f.prompt_hint:
            parts.append(f.prompt_hint)
        if f.extraction_aliases:
            parts.append(f"Look for labels: {', '.join(f.extraction_aliases)}")
        if parts:
            hints.append(f"- {f.name}: {' '.join(parts)}")
    return "\n".join(hints)


BASE_EXTRACTION_RULES = """You are a receipt/invoice data extraction engine. Extract structured data from the OCR text below.

RULES:
1. Use null for any field you cannot confidently determine. Never guess or hallucinate values.
2. Amounts: Remove currency symbols (¥, $, ￥). Output as numbers, not strings.
   Handle full-width numbers: ￥１，５００ → 1500
3. CRITICAL — ¥ is a currency symbol, NOT the digit 1. OCR often misreads the handwritten yen sign ¥ as the number 1.
   If you see a number like 13000 but the OCR text shows ¥3000, the actual amount is 3000 (the 1 is the ¥ symbol).
   Always check: does the number start with 1 and does the OCR text have ¥ before the remaining digits?
4. For contracts/bills: "total" is the amount due. Line items are the billed services.
5. Line items may span across page boundaries marked by --- PAGE N ---. Treat all pages as one continuous document.
6. OCR may merge multiple lines into one. If a single line contains multiple product names with prices, split them into separate line items.
7. For handwritten receipts (領収証): the 金額 (amount) field IS the total. Use EXACTLY the number shown after ¥.
   Do NOT add tax unless actual tax numbers are handwritten. Empty pre-printed form labels (税抜金額, 消費税額, etc.) with no numbers filled in mean no tax — output taxes as an empty list.
   Do NOT create line_items for handwritten receipts unless individual items are listed.
8. 令和7年=2025, 令和8年=2026. If OCR shows just '7年' or '8年' with no era name, assume 令和.
9. The merchant name is the consumer-facing BRAND name ONLY — do NOT include the branch name, store location, or shopping complex. Strip everything after the core brand: 'ダイソー' not 'ダイソー ビバモール赤間店', 'カルディ' not 'カルディコーヒーファーム', 'マックスバリュ' not 'マックスバリュくりえいと宗像店'. Do NOT use the parent company (AEON/イオン), corporate entity (株式会社..., 有限会社...), franchisee/operator name, or abbreviation. Short text next to a number code (e.g. 'コウソ 003080') is a cashier/operator identifier, NOT the merchant. If a website domain appears (e.g. 'clickcycle.com'), the brand name is often derived from it (e.g. 'click'). For gas stations, use the fuel brand not the station operator.
10. If the receipt shows a specific payment method like WAON, クレジット, Suica, PayPay, etc., use that. Only default to "cash" if no electronic payment is named and you see お預り (cash tendered) or 現計 (cash total).
11. OCR may put item names and their prices on SEPARATE lines. Associate each item with the ¥ amount on the NEXT line. Example: "サニーレタス" followed by "¥129" means サニーレタス costs 129.
12. When OCR shows distinct adjacent item prices, emit each item's total verbatim from OCR. Do NOT duplicate a price across multiple items unless OCR shows that same price multiple times. If item names are in one block and prices are in a following block, preserve the price sequence one-for-one.
13. The subtotal is ALWAYS the pre-tax base: subtotal = total - sum(taxes). This holds for both 外税 (tax-exclusive) and 内税 (tax-inclusive) receipts. For 内税 receipts where printed item prices already include tax, subtotal is LESS than the printed 合計; do NOT set subtotal equal to total. The tax lines (外税8%税額, 消費税 etc.) show the TAX amount, NOT the subtotal.
14. 課税対象額 means "taxable amount" (the BASE that tax is calculated on) — this is NOT a tax. Only 税額 (tax amount) entries should be in the taxes list. Example: "税率8%課税対象額 ¥2274" is the taxable base; "税率8%税額 ¥168" is the actual tax of 168.
15. Labels (合計, 小計, 税額) may appear on a DIFFERENT line from their ¥ values, especially in rotated receipts where all labels are in one block and all values in another. Use arithmetic to match: tax = total − subtotal. If you see many ¥ amounts together (e.g. "¥2,279  ¥2,111  ¥168)"), match them with labels elsewhere in the text.
16. DISCOUNTS: If a discount line follows an item (e.g., "割引 20%" with "-¥94" after "銀さけ切身 ¥467"), do NOT create a separate line item for the discount. Merge it into the parent item: set total = price AFTER discount (467 - 94 = 373), set discount = 94, set discount_rate = "20%". Every line item total must be positive.
"""

UTILITY_BILL_RULES = """You are a utility bill data extraction engine. Extract structured data from the OCR text below.

RULES:
1. Use null for any field you cannot confidently determine. Never guess or hallucinate values.
2. Amounts: Remove currency symbols (¥, $, ￥). Output as numbers, not strings.
3. 令和7年=2025, 令和8年=2026. If OCR shows just '7年' or '8年' with no era name, assume 令和.
4. Set document_type to "utility_bill".
5. The merchant is the utility company (gas, water, electric provider). Do NOT include 株式会社 or similar suffixes.
6. The total is the ご請求額 or 引落予定額 (amount to be charged).
7. For date: use the payment/debit date (引落予定日) if shown, else the meter reading date (検針日).
8. service_type must be one of: gas, water, electric, sewage, internet, phone. If the bill covers both 水道 (water supply) and 下水道 (sewage), use 'water' — combined water/sewage bills are water bills.
9. Extract billing_period as start/end dates in YYYY-MM-DD format from 前回検針日 → 今回検針日 or 使用期間.
10. Extract usage: amount (ご使用量), unit (m3/kWh/L), cost_per (単価, price per unit — null if not shown or if tiered pricing), meter_previous (前回指針), meter_current (今回指針).
11. payment_method should be "bank_payment" if 口座引落/口座振替/振替 is mentioned, else null.
12. Do NOT create line_items — leave as empty list.
13. account_number is the お客様番号 if present.
"""

PAYMENT_SLIP_RULES = """You are a payment slip data extraction engine. Extract structured data from the OCR text below.

RULES:
1. Use null for any field you cannot confidently determine. Never guess or hallucinate values.
2. Amounts: Remove currency symbols (¥, $, ￥). Output as numbers, not strings.
3. 令和7年=2025, 令和8年=2026. If OCR shows just '7年' or '8年' with no era name, assume 令和.
4. Set document_type to "payment_slip".
5. The merchant is the company receiving the money (受取人). Look for the 受取人 field specifically. Do NOT use the credit card company, payment processor, or intermediary (e.g., do NOT use アプラス if the 受取人 is a different company). Do NOT include 株式会社 or similar suffixes.
6. For date: use the payment date (stamp date, 収納日) if visible, else the due date (支払期限, 納付期限).
7. The total is the 金額 (amount).
8. payer is the person/entity making the payment (依頼人).
9. payment_reference is any tracking/reference number on the slip.
10. payment_method should be null unless explicitly clear (e.g., if stamp shows コンビニ, it was paid there but we use null).
11. Do NOT create line_items — leave as empty list.
"""


def generate_extraction_prompt(ocr_text: str, doc_type: str = "receipt") -> tuple[str, str]:
    """Build the full LLM extraction prompt from the field registry.

    Returns (system_prompt, user_prompt) for system/user message separation.
    """
    if doc_type == "utility_bill":
        rules = UTILITY_BILL_RULES
    elif doc_type == "payment_slip":
        rules = PAYMENT_SLIP_RULES
    else:
        rules = BASE_EXTRACTION_RULES

    system_prompt = f"""{rules}
FIELD-SPECIFIC RULES:
{_build_field_hints(doc_type)}

Respond with a single JSON object containing all extracted fields. Return only JSON with no additional text or explanation."""

    user_prompt = f"""OCR TEXT:
{ocr_text}"""

    return system_prompt, user_prompt


_VERIFICATION_SYSTEM_PROMPT = """You are a receipt/invoice data extraction engine performing a VERIFICATION PASS.

Below is the original OCR text from a receipt, your previous extraction attempt, and
a list of validation warnings (arithmetic errors, consistency issues).

CRITICAL RULES FOR VERIFICATION:
1. ONLY fix the specific fields mentioned in the warnings. Do NOT change other fields.
2. Receipt totals, subtotals, and tax amounts are AUTHORITATIVE — use values printed on the receipt.
3. If qty × unit_price does not match total, prefer unit_price from the OCR text and set qty=1 unless an explicit multiplier (×N, N点) exists.
4. If sum of line items does not match subtotal, look for items with qty > 1 that should be qty=1.
5. Keep all other fields exactly as they were."""


def generate_verification_prompt(
    ocr_text: str,
    previous_extraction: dict,
    validation_warnings: list[str],
) -> tuple[str, str]:
    """Build the verification pass prompt.

    Returns (system_prompt, user_prompt) for system/user message separation.
    """
    warnings_block = "\n".join(f"- {w}" for w in validation_warnings) if validation_warnings else "None"

    doc_type = previous_extraction.get("document_type", "receipt")
    doc_type_label = {"receipt": "receipt/store purchase",
                      "utility_bill": "utility bill (gas/water/electric)",
                      "payment_slip": "bank transfer or convenience store payment slip"
                      }.get(doc_type, doc_type)
    field_hints = "\n".join(
        f"- {f.name}: {f.prompt_hint}" for f in FIELD_REGISTRY
        if f.prompt_hint and doc_type in f.doc_types
    )

    system_prompt = f"""{_VERIFICATION_SYSTEM_PROMPT}

Document type: {doc_type_label}. Apply rules specific to this document type.

FIELD-SPECIFIC RULES:
{field_hints}

Respond with a single JSON object containing all extracted fields. Return only JSON with no additional text or explanation."""

    user_prompt = f"""PREVIOUS EXTRACTION:
{json.dumps(previous_extraction, ensure_ascii=False, indent=2)}

VALIDATION WARNINGS:
{warnings_block}

ORIGINAL OCR TEXT:
{ocr_text}"""

    return system_prompt, user_prompt
