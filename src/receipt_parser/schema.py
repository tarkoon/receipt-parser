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
        prompt_hint="The consumer-facing seller or provider name. For payment slips, use the named recipient. Keep branch, location, parent-company, and operator metadata out of this field. For dual English and Japanese names, use the most prominent form.",
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
        prompt_hint="A clearly marked transaction or payment time in 24-hour H:MM format, with no leading zero on the hour. Drop seconds. Output null for business hours, phone numbers, and printer or issue timestamps that do not clearly mark the transaction.",
        extraction_aliases=["時刻", "time"],
    ),
    FieldMeta(
        name="location",
        debug_color_bgr=(200, 200, 0),
        prompt_hint="Use the most-specific reliable location exactly as printed. Prefer a full address; otherwise retain the full printed branch, facility, or locality instead of reducing it to a broader place. Never infer missing geography from an ambiguous name or phone number. Output null if no reliable location is printed.",
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
        prompt_hint="Must be one of: cash, credit, debit, bank_payment, WAON, or null. Populate it only when one actual tender is established. Use null for mixed tender or when no single method is clear.",
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
        prompt_hint="Loyalty points actually applied as payment. Use null when no redemption is printed, preserve an explicitly printed zero, and use a positive number only with direct redemption evidence such as ポイント利用 or ポイント値引. Do not confuse eligible or earned points with redemption.",
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
        prompt_hint="Match description, qty, unit_price, and total per row. Default qty=1. A tax marker attached to a trailing number does not stop that number from being the price. Split multiple item-and-price pairs merged onto one OCR line. discount_rate is the positive effective percentage removed from the pre-discount extended price, formatted as <number>%; leave it empty when no effective rate is established.",
        doc_types=["receipt"],
    ),
    FieldMeta(
        name="subtotal",
        debug_color_bgr=(0, 255, 255),
        prompt_hint="The pre-tax base: subtotal = total - sum(taxes). For 内税 receipts where 合計 includes a printed tax amount, compute subtotal = 合計 - 消費税. For 外税 receipts subtotal equals the printed pre-tax 小計. When only a tax rate or gross rate target is printed and neither a tax amount nor pre-tax base is printed, leave subtotal null instead of inferring it.",
        extraction_aliases=["小計", "subtotal"],
        doc_types=["receipt"],
    ),
    FieldMeta(
        name="taxes",
        debug_color_bgr=(255, 0, 255),
        prompt_hint="Output as a list of objects with 'rate' (string like '10%' or '8%'), 'label' (e.g. '内税'), and 'amount' (number). Prefer a printed tax amount; arithmetic reconstruction from an explicitly printed taxable base and rate is allowed when it reconciles. JP tax: 軽減税率/※ = reduced 8%, standard = 10%. 内税 means tax-inclusive, 外税 means tax-exclusive. When only a rate or gross rate target is printed, keep the rate on each item's tax_category but output taxes=[]; never invent a numeric tax amount from the rate alone.",
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
        prompt_hint="Output 'start' and 'end' in YYYY-MM-DD format. Prefer an explicitly printed usage period. When deriving from meter-reading dates, start is the calendar day after the previous reading and end is the current reading date.",
        extraction_aliases=["使用期間", "検針日"],
        doc_types=["utility_bill"],
    ),
    FieldMeta(
        name="usage",
        debug_color_bgr=(200, 200, 100),
        prompt_hint="Measured consumption or purchased volume. Output 'amount', 'unit', 'cost_per', 'meter_previous', and 'meter_current', using null for unprinted members. Retain explicitly printed fuel quantity, unit, and unit price on receipts; service_type remains null because it is utility-only.",
        extraction_aliases=["ご使用量", "使用量", "指針"],
        doc_types=["receipt", "utility_bill"],
    ),

    # Named parties and references can appear on any document type
    FieldMeta(
        name="payer",
        debug_color_bgr=(100, 0, 200),
        prompt_hint="The person or entity explicitly named as payer or document addressee. Do not infer it from unrelated names, account numbers, or membership identifiers.",
        extraction_aliases=["依頼人", "ご依頼人", "宛名"],
    ),
    FieldMeta(
        name="payment_reference",
        debug_color_bgr=(0, 100, 200),
        prompt_hint="A value-only printed document reference, preserving leading zeroes. Prefer a uniquely labeled receipt or slip number; if none is printed, use one uniquely labeled transaction, data, handling, inquiry, acceptance, or reference number. Do not include the label and do not use an unlabeled serial, account, membership, or card number.",
        extraction_aliases=["レシートNo", "伝票No", "取扱番号", "手番", "収納番号", "参照番号"],
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
        match = re.fullmatch(r'\s*[+-]?(\d+(?:\.\d+)?)\s*[%％]?\s*', str(v or ""))
        if not match:
            return ""
        rate = float(match.group(1))
        return f"{rate:g}%" if rate > 0 else ""

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

    @model_validator(mode="after")
    def clear_ineffective_discount_rate(self) -> "LineItem":
        if self.discount <= 0:
            self.discount_rate = ""
        return self


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

    # Utility-only fields, plus usage shared with fuel receipts
    service_type: Optional[str] = None
    billing_period: Optional[BillingPeriod] = None
    usage: Optional[UsageData] = None

    # Optional named party/reference fields shared across document types
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

    @field_validator("account_number", mode="before")
    @classmethod
    def coerce_optional_account_number(cls, v):
        if v is None or v == "null" or v == "None" or isinstance(v, bool):
            return None
        if isinstance(v, (str, int, float)):
            return str(v).strip() or None
        return None

    @field_validator("merchant", "date", "time", "location", "currency", "payment_method",
                     "payer", "payment_reference",
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
    def clear_out_of_scope_fields(self) -> "Document":
        if self.document_type != "utility_bill":
            self.service_type = None
            self.billing_period = None
        if self.document_type == "payment_slip":
            self.usage = None
        return self

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
9. The merchant is the consumer-facing seller or provider. Keep branch, location, shopping-complex, parent-company, corporate, franchisee, and operator metadata out of this field. For fuel purchases, use the displayed fuel brand rather than an operator name.
10. Use only the canonical payment values. Preserve WAON as "WAON"; map named card or electronic payments to "credit" unless the document explicitly says debit. Use "cash" only when cash tender evidence is printed. If multiple tenders contributed and no single method represents the transaction, use null.
11. OCR may put item descriptions and prices on separate lines. Associate each item with its structurally adjacent amount.
12. When OCR shows distinct adjacent item prices, emit each item's total verbatim from OCR. Do NOT duplicate a price across multiple items unless OCR shows that same price multiple times. If item names are in one block and prices are in a following block, preserve the price sequence one-for-one.
13. The subtotal is the pre-tax base: subtotal = total - sum(taxes). This holds for both 外税 (tax-exclusive) and 内税 (tax-inclusive) receipts. For 内税 receipts where printed item prices already include tax, subtotal is LESS than the printed 合計. Exception: if only a tax rate or gross rate target is printed and no numeric tax amount or explicit pre-tax base exists, do not derive either amount; output subtotal=null and taxes=[]. Keep the printed rate only as each item's tax_category. The tax lines (外税8%税額, 消費税 etc.) show the TAX amount, NOT the subtotal.
14. 課税対象額 means "taxable amount" (the BASE that tax is calculated on) — this is NOT a tax. Only 税額 (tax amount) entries should be in the taxes list. Example: "税率8%課税対象額 ¥2274" is the taxable base; "税率8%税額 ¥168" is the actual tax of 168.
15. Labels (合計, 小計, 税額) may appear on a DIFFERENT line from their ¥ values, especially in rotated receipts where all labels are in one block and all values in another. Use arithmetic to match: tax = total − subtotal. If you see many ¥ amounts together (e.g. "¥2,279  ¥2,111  ¥168)"), match them with labels elsewhere in the text.
16. DISCOUNTS: Do not create a separate line item for a discount. Merge it into the affected item: total is the post-discount amount and discount is the positive amount removed. discount_rate is the positive effective percentage removed from the pre-discount extended price, canonically formatted as <number>%; leave it empty when no effective rate is established. Every line item total must be positive.
17. payment_reference is value-only and preserves leading zeroes. Prefer one explicitly labeled receipt or slip number; only when none is printed, use one uniquely labeled transaction, data, handling, inquiry, acceptance, or reference number. Otherwise use null.
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
9. Extract billing_period as start/end dates in YYYY-MM-DD format. Prefer an explicit usage period. When deriving from meter readings, start is the calendar day after the previous reading and end is the current reading date.
10. Extract usage: amount (ご使用量), unit (m3/kWh/L), cost_per (単価, price per unit — null if not shown or if tiered pricing), meter_previous (前回指針), meter_current (今回指針).
11. payment_method is "bank_payment" only when bank payment is the single established tender; use null for mixed tender or when no single method is clear.
12. Do NOT create line_items — leave as empty list.
13. account_number is the お客様番号 if present.
"""

PAYMENT_SLIP_RULES = """You are a payment slip data extraction engine. Extract structured data from the OCR text below.

RULES:
1. Use null for any field you cannot confidently determine. Never guess or hallucinate values.
2. Amounts: Remove currency symbols (¥, $, ￥). Output as numbers, not strings.
3. 令和7年=2025, 令和8年=2026. If OCR shows just '7年' or '8年' with no era name, assume 令和.
4. Set document_type to "payment_slip".
5. The merchant is the company receiving the money (受取人). Look for the 受取人 field specifically. Do not substitute a card company, payment processor, or intermediary, and omit corporate suffixes.
6. For date: use the payment date (stamp date, 収納日) if visible, else the due date (支払期限, 納付期限).
7. The total is the 金額 (amount).
8. payer is the person or entity explicitly named as payer or addressee.
9. payment_reference is value-only and preserves leading zeroes. Prefer one explicitly labeled receipt or slip number; only when none is printed, use one uniquely labeled transaction, data, handling, inquiry, acceptance, or reference number. Otherwise use null.
10. payment_method is populated only when one actual tender is clearly established; use null for mixed tender or indirect payment-location evidence.
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


_VERIFICATION_SYSTEM_PROMPT = """You are a document data extraction engine performing a VERIFICATION PASS.

Below is the original OCR text from a document, your previous extraction attempt, and
a list of validation warnings (arithmetic errors, consistency issues).

CRITICAL RULES FOR VERIFICATION:
1. ONLY fix the specific fields mentioned in the warnings. Do NOT change other fields.
2. Printed totals, subtotals, and tax amounts are AUTHORITATIVE — use values printed on the document.
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
