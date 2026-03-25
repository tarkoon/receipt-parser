"""schema.py — Single source of truth for receipt data extraction.

All extraction fields, prompt generation, validation rules, and debug
overlay colors are defined here. To add a new field, see the
EXTENDING THE SCHEMA section in the build plan.
"""

from __future__ import annotations
from pydantic import BaseModel, field_validator
from typing import Literal, Optional
import json


# ── Field Registry ────────────────────────────────────────────────────
class FieldMeta:
    """Metadata for a single extractable field."""
    def __init__(
        self,
        name: str,
        debug_color_bgr: tuple[int, int, int],
        prompt_hint: str | None = None,
        extraction_aliases: list[str] | None = None,
    ):
        self.name = name
        self.debug_color_bgr = debug_color_bgr
        self.prompt_hint = prompt_hint
        self.extraction_aliases = extraction_aliases or []


FIELD_REGISTRY: list[FieldMeta] = [
    FieldMeta(
        name="merchant",
        debug_color_bgr=(255, 165, 0),
        prompt_hint="The store/merchant name is usually the largest text at the very top of the receipt, often spanning multiple lines. Combine all parts of the name (e.g. chain name + sub-brand + location). Do NOT use the parent company name at the bottom.",
        extraction_aliases=["店名", "store", "shop"],
    ),
    FieldMeta(
        name="date",
        debug_color_bgr=(0, 255, 0),
        prompt_hint="Parse Japanese dates: 令和8年=2026, 令和7年=2025. Convert 2026年3月15日 to 2026-03-15. Always output as YYYY-MM-DD.",
        extraction_aliases=["日付", "日時", "date"],
    ),
    FieldMeta(
        name="location",
        debug_color_bgr=(200, 200, 0),
        extraction_aliases=["住所", "address"],
    ),
    FieldMeta(
        name="currency",
        debug_color_bgr=(180, 180, 180),
        prompt_hint="If the receipt uses ¥ or ￥ symbols, or is entirely in Japanese, currency is JPY. If it uses $ or is in English with no other currency indicator, currency is USD. Always output the three-letter ISO code.",
        extraction_aliases=["¥", "￥", "$", "円"],
    ),
    FieldMeta(
        name="line_items",
        debug_color_bgr=(255, 255, 0),
        prompt_hint="Match description, quantity, unit price, and line total on the same row. If quantity is not shown, assume 1. For tax_category: if the receipt shows '10%内税対象' or '8%対象', set tax_category to the matching rate. Items marked ※ or X are reduced 8% tax. Items marked '除' are exempt. OCR may merge multiple items into one line — look for multiple prices (e.g. '食品ポリ袋L3除日清チャック付328※' = two items: ¥3 and ¥328). Common JP receipt items: レジ袋 (plastic bag). DISCOUNTS: If a discount line follows an item (e.g., '割引 20%' with -¥94), do NOT create a separate line item. Instead merge it into the parent item: set total to the price AFTER discount (unit_price - discount), set discount to the discount amount, and set discount_rate to the rate string (e.g., '20%'). The total must always be positive.",
    ),
    FieldMeta(
        name="subtotal",
        debug_color_bgr=(0, 255, 255),
        extraction_aliases=["小計", "subtotal"],
    ),
    FieldMeta(
        name="taxes",
        debug_color_bgr=(255, 0, 255),
        prompt_hint="Output as a list of objects with 'rate' (string like '10%' or '8%'), 'label' (e.g. '内税'), and 'amount' (number). JP tax: 軽減税率/※ = reduced 8%, standard = 10%. Look for patterns like '10%内税対象' or '(10%内)' to determine the rate. 内税 means tax-inclusive, 外税 means tax-exclusive.",
        extraction_aliases=["税", "消費税", "tax", "内税", "外税"],
    ),
    FieldMeta(
        name="total",
        debug_color_bgr=(0, 0, 255),
        extraction_aliases=["合計", "total", "お会計"],
    ),
    FieldMeta(
        name="payment_method",
        debug_color_bgr=(128, 0, 128),
        extraction_aliases=["支払", "payment"],
    ),
    FieldMeta(
        name="invoice_number",
        debug_color_bgr=(0, 128, 128),
        extraction_aliases=["番号", "No.", "invoice"],
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


# ── Pydantic Models ───────────────────────────────────────────────────

class LineItem(BaseModel):
    description: str
    qty: float = 1
    unit_price: Optional[float] = None
    total: float
    tax_category: Literal["8%", "10%", "0%"] = "0%"
    discount: float = 0
    discount_rate: str = ""

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


class Receipt(BaseModel):
    """Main extraction model."""
    merchant: Optional[str] = None
    date: Optional[str] = None
    location: Optional[str] = None
    currency: Optional[str] = None
    line_items: list[LineItem] = []
    subtotal: Optional[float] = None
    taxes: list[TaxEntry] = []
    total: Optional[float] = None
    payment_method: Optional[str] = None
    invoice_number: Optional[str] = None
    raw_text_summary: Optional[str] = None


# ── Prompt Helpers ────────────────────────────────────────────────────

def _build_field_hints() -> str:
    """Build the FIELD-SPECIFIC RULES block from the registry."""
    hints = []
    for f in FIELD_REGISTRY:
        parts = []
        if f.prompt_hint:
            parts.append(f.prompt_hint)
        if f.extraction_aliases:
            parts.append(f"Look for labels: {', '.join(f.extraction_aliases)}")
        if parts:
            hints.append(f"- {f.name}: {' '.join(parts)}")
    return "\n".join(hints)


# ── OCR-based Prompt Generation ───────────────────────────────────────

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
9. The merchant name should NOT include parent company names like AEON/イオン or corporate suffixes like 株式会社. Use the specific store name only.
10. If the receipt shows a specific payment method like WAON, クレジット, Suica, PayPay, etc., use that. Only default to "cash" if no electronic payment is named and you see お預り (cash tendered) or 現計 (cash total).
11. OCR may put item names and their prices on SEPARATE lines. Associate each item with the ¥ amount on the NEXT line. Example: "サニーレタス" followed by "¥129" means サニーレタス costs 129.
12. The subtotal (小計) is the sum of item prices BEFORE tax. The tax lines (外税8%税額, 消費税 etc.) show the TAX amount, NOT the subtotal. Do NOT confuse 小計 with tax.
13. 課税対象額 means "taxable amount" (the BASE that tax is calculated on) — this is NOT a tax. Only 税額 (tax amount) entries should be in the taxes list. Example: "税率8%課税対象額 ¥2274" is the taxable base; "税率8%税額 ¥168" is the actual tax of 168.
14. Labels (合計, 小計, 税額) may appear on a DIFFERENT line from their ¥ values, especially in rotated receipts where all labels are in one block and all values in another. Use arithmetic to match: tax = total − subtotal. If you see many ¥ amounts together (e.g. "¥2,279  ¥2,111  ¥168)"), match them with labels elsewhere in the text.
15. DISCOUNTS: If a discount line follows an item (e.g., "割引 20%" with "-¥94" after "銀さけ切身 ¥467"), do NOT create a separate line item for the discount. Merge it into the parent item: set total = price AFTER discount (467 - 94 = 373), set discount = 94, set discount_rate = "20%". Every line item total must be positive.
"""


def generate_extraction_prompt(ocr_text: str) -> str:
    """Build the full LLM extraction prompt from the field registry."""
    prompt = f"""{BASE_EXTRACTION_RULES}
FIELD-SPECIFIC RULES:
{_build_field_hints()}

OCR TEXT:
{ocr_text}
"""
    return prompt


def generate_verification_prompt(
    ocr_text: str,
    previous_extraction: dict,
    validation_warnings: list[str],
) -> str:
    """Build the verification pass prompt."""
    warnings_block = "\n".join(f"- {w}" for w in validation_warnings) if validation_warnings else "None"

    prompt = f"""You are a receipt/invoice data extraction engine performing a VERIFICATION PASS.

Below is the original OCR text from a receipt, your previous extraction attempt, and
a list of validation warnings (arithmetic errors, consistency issues).

CRITICAL RULES FOR VERIFICATION:
1. ONLY fix the specific fields mentioned in the warnings. Do NOT change other fields.
2. Receipt totals, subtotals, and tax amounts are AUTHORITATIVE — use values printed on the receipt.
3. If qty × unit_price does not match total, prefer unit_price from the OCR text and set qty=1 unless an explicit multiplier (×N, N点) exists.
4. If sum of line items does not match subtotal, look for items with qty > 1 that should be qty=1.
5. Keep all other fields exactly as they were.

FIELD-SPECIFIC RULES:
{chr(10).join(f"- {f.name}: {f.prompt_hint}" for f in FIELD_REGISTRY if f.prompt_hint)}

PREVIOUS EXTRACTION:
{json.dumps(previous_extraction, ensure_ascii=False, indent=2)}

VALIDATION WARNINGS:
{warnings_block}

ORIGINAL OCR TEXT:
{ocr_text}
"""
    return prompt
