# New Schema Plan — Multi-Document Type Support

## Problem

The current schema assumes all documents are retail receipts. Receipts 14-21 revealed
three distinct document types that need different fields, extraction logic, and validation.

## Document Types

| Type | Examples | Key characteristics |
|---|---|---|
| `receipt` | #17 UNIQLO, #18 HANDS, #19 コジマ | Line items, subtotal, taxes, store purchase |
| `utility_bill` | #15 gas, #16 water, #20 gas | Usage data, billing period, meter readings, tiered rates |
| `payment_slip` | #14 払込票, #21 振込請求書 | Proof of payment, payer, reference numbers |

---

## Enum Reference (all constrained fields)

Single source of truth for every field that uses a fixed set of values.
Update this section when adding new allowed values — the Pydantic Literals,
LLM prompts, validation rules, and tests should all derive from here.

### document_type
```
Literal["receipt", "utility_bill", "payment_slip"]
```
| Value | When to use |
|---|---|
| `receipt` | Store purchases with line items (supermarket, retail, convenience store, handwritten 領収証) |
| `utility_bill` | Recurring service bills with usage data (gas, water, electric, internet, phone) |
| `payment_slip` | Proof-of-payment documents (払込票, 振込請求書, コンビニ収納) |

### payment_method
```
Literal["cash", "credit", "debit", "bank_payment", "WAON"] | null
```
| Value | Trigger (OCR evidence required) |
|---|---|
| `cash` | 現計, お預り > total, お釣り, 現金 |
| `credit` | クレジット, VISA, Mastercard, JCB |
| `debit` | デビット |
| `bank_payment` | 口座引落, 口座振替, 振替させていただきます, 振込 |
| `WAON` | WAON (electronic money) |
| `null` | No evidence on document — do NOT guess |

### currency
```
Literal["JPY", "USD"]
```
| Value | Trigger |
|---|---|
| `JPY` | ¥, ￥, 円, or entirely Japanese text |
| `USD` | $, or English text with no other currency indicator |

### tax_category (per line item)
```
Literal["8%", "10%", "0%"]
```
| Value | Meaning |
|---|---|
| `8%` | Reduced tax rate (軽減税率) — food, newspapers. Markers: ※, 軽, X |
| `10%` | Standard tax rate. Markers: 除 (excluded from reduced rate) |
| `0%` | Unknown / unassigned (default — pipeline overrides via OCR evidence) |

### tax rate (per tax entry)
```
str — typically "8%", "10%", or "unknown"
```
| Value | When to use |
|---|---|
| `8%` | Reduced rate tax amount identified |
| `10%` | Standard rate tax amount identified |
| `unknown` | Tax amount found but rate not determinable |

### tax label (per tax entry)
```
str | null — describes how tax is applied
```
| Value | Meaning |
|---|---|
| `内税` | Tax-inclusive (tax included in displayed prices) |
| `外税` | Tax-exclusive (tax added on top of displayed prices) |
| `内消費税` | Consumption tax (inclusive) |
| `税額` | Tax amount line |
| `税合計` | Total tax (when only one aggregate amount) |
| `null` | Label not determinable |

### service_type (utility bills only)
```
Literal["gas", "water", "electric", "sewage", "internet", "phone"] | null
```
| Value | Trigger (OCR keywords) |
|---|---|
| `gas` | ガス, ガス料金, ガス検針 |
| `water` | 水道, 上水道, 水道料金 |
| `electric` | 電気, 電力, 電気料金 |
| `sewage` | 下水道, 下水道使用料 |
| `internet` | インターネット, 通信料 (or via merchant_map) |
| `phone` | 電話, 携帯, 通話料 |
| `null` | Service type not determinable |

### usage unit (utility bills only)
```
Literal["m3", "kWh", "L"] | null
```
| Value | Service |
|---|---|
| `m3` | Gas, water |
| `kWh` | Electric |
| `L` | Water (rare, some areas) |

---

## Schema Design

### Common Fields (all document types)

```
document_type       Literal["receipt", "utility_bill", "payment_slip"] (strict — adding new types requires a schema change)
merchant            The actual company (after alias mapping)
date                Payment date if available, else due date for bills,
                    transaction date for receipts
location            Address if present
currency            "JPY" | "USD"
total               Amount due / purchase total
payment_method      "cash" | "credit" | "bank_payment" | null
                    null unless explicitly stated on document.
                    auto_debit only when bill says 口座引落/振替
account_number      Customer/account number (NEW — for recurring bill tracking)
points_used         Loyalty points applied (NEW — d-point, WAON point, etc.)
amount_paid         Actual out-of-pocket cost: total - points_used (NEW)
```

### Receipt-Specific Fields

```
line_items[]        List of purchased items
  description       Item name
  qty               Quantity (default 1)
  unit_price        Price per unit
  total             Line total after discount
  tax_category      "8%" | "10%" | "0%"
  discount          Discount amount
  discount_rate     Discount percentage string
subtotal            Pre-tax sum of line items
taxes[]             Per-rate tax entries
  rate              "8%" | "10%" | "unknown"
  label             "内税" | "外税" | etc.
  amount            Tax amount
```

### Utility Bill-Specific Fields

```
service_type        "gas" | "water" | "electric" | "sewage" | "internet" | "phone"
billing_period
  start             "YYYY-MM-DD"
  end               "YYYY-MM-DD"
usage
  amount            Numeric usage (e.g., 15.2)
  unit              "m3" | "kWh" | "L"
  cost_per          Price per unit (e.g., 181.0 for ¥181/L). null if tiered pricing.
  meter_previous    Previous meter reading
  meter_current     Current meter reading
```

### Payment Slip-Specific Fields

```
payer               Who pays (customer name if present)
payment_reference   Tracking/reference number on the slip
```

---

## Pydantic Model Changes (schema.py)

### New models to add

```python
class BillingPeriod(BaseModel):
    start: Optional[str] = None    # YYYY-MM-DD
    end: Optional[str] = None      # YYYY-MM-DD

class UsageData(BaseModel):
    amount: Optional[float] = None
    unit: Optional[str] = None     # "m3", "kWh", "L"
    meter_previous: Optional[float] = None
    meter_current: Optional[float] = None
```

### Modified Receipt model

The current `Receipt` model becomes a unified `Document` model with a discriminator:

```python
class Document(BaseModel):
    # Common
    document_type: Literal["receipt", "utility_bill", "payment_slip"] = "receipt"
    merchant: Optional[str] = None
    date: Optional[str] = None
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
```

### Backward compatibility

Keep `Receipt` as an alias: `Receipt = Document`. This avoids breaking every import
across pipeline.py, extraction.py, validation.py, cli.py, benchmark_models.py, and tests.

---

## Pipeline Changes

### Step 0 (NEW): Document Type Detection

Insert before LLM extraction. Uses OCR text keywords to classify:

```
utility_bill indicators:
  検針, 使用量, m3, kWh, ガス料金, 水道料金, 電気料金,
  ご請求額, 引落予定, メーター, 基本料金

payment_slip indicators:
  払込票, 振込, 受領証 (with 払込/振込 context), 請求書,
  依頼人, 受取人, コンビニ収納

receipt (default):
  小計, 合計, レジ, 領収証 (without 払込/振込 context)
```

This classification happens in pipeline.py BEFORE the LLM call, so we can select
the appropriate prompt and schema.

### Step 0.5 (NEW): Type-Specific Prompt Selection

Each document type gets tailored extraction rules:

- **receipt**: Current BASE_EXTRACTION_RULES (15 rules) — unchanged
- **utility_bill**: New rules focused on meter readings, usage, billing periods,
  service type detection, tiered rate extraction
- **payment_slip**: New rules focused on merchant/payer identification, reference
  numbers, payment date vs due date

The field registry (FIELD_REGISTRY) needs type-aware hints. Options:
1. Separate registries per type
2. Single registry with `applicable_types` filter on each field

Recommend option 2 — less duplication, easier to maintain.

### Step 4.5 Enhancement: points_used / amount_paid

After LLM extraction, scan OCR text for point usage patterns:
```
ポイント利用    ¥1,500   → points_used = 1500
ポイント値引    -500     → points_used = 500
dポイント       1,500P   → points_used = 1500
WAONポイント    200      → points_used = 200
```

Compute: `amount_paid = total - points_used` (if points_used found, else amount_paid = total)

### Step 4.7 Enhancement: Payment Method

Expand detection for bill types:
```
口座引落 / 口座振替 / 振替させていただきます → "bank_payment"
コンビニ + 払込/収納                          → "conbini_payment"
振込                                          → "bank_transfer"
```

Default: null for bills (unless explicitly stated), existing logic for receipts.

### Step 4.11 (NEW): Merchant Alias Mapping

After all extraction, apply user_rules.json mapping:

```python
def apply_merchant_mapping(result: dict, rules_path: Path) -> dict:
    rules = json.loads(rules_path.read_text())
    merchant_map = rules.get("merchant_map", {})

    # Check merchant field for matches
    for field in ["merchant"]:
        value = result.get(field, "") or ""
        for pattern, mapping in merchant_map.items():
            if pattern in value:
                if "merchant" in mapping:
                    result["merchant"] = mapping["merchant"]
                if "category" in mapping:
                    result["_category"] = mapping["category"]
                break
    return result
```

### Step 4.12 (NEW): Date Logic for Bills

For `utility_bill` and `payment_slip`:
1. Look for payment date (支払日, 納付日, stamp date) → use if found
2. Else look for due date (支払期限, 引落予定日, 納付期限) → use as fallback
3. Else use issue date (発行日, 検針日)

For `receipt`: existing logic (transaction date) — unchanged.

---

## Validation Changes (validation.py)

### Type-aware validation

```python
def validate_document(doc: Document) -> list[str]:
    warnings = []

    # Common validation
    if doc.points_used and doc.total:
        expected_paid = doc.total - doc.points_used
        if doc.amount_paid and abs(doc.amount_paid - expected_paid) > 1:
            warnings.append(f"amount_paid ({doc.amount_paid}) != total ({doc.total}) - points_used ({doc.points_used})")

    if doc.document_type == "receipt":
        warnings.extend(_validate_receipt(doc))
    elif doc.document_type == "utility_bill":
        warnings.extend(_validate_utility_bill(doc))
    elif doc.document_type == "payment_slip":
        warnings.extend(_validate_payment_slip(doc))

    return warnings
```

**Receipt validation**: existing checks (line item math, subtotal sum, total consistency).

**Utility bill validation**:
- usage.amount should be positive
- meter_current > meter_previous
- total should be positive
- billing_period.end > billing_period.start

**Payment slip validation**:
- total must be present and positive
- merchant should be present

---

## user_rules.json

New file at `receipt-parser/user_rules.json`:

```json
{
  "merchant_map": {
    "スマートビリングサービス": {
      "merchant": "Bizmo",
      "category": "internet"
    },
    "西部ガス": {
      "category": "gas"
    },
    "北九州市上下水道局": {
      "category": "water"
    }
  }
}
```

- Substring matching against merchant field
- `merchant` key overrides the extracted merchant name
- `category` key adds a `_category` metadata field to results
- File is optional — pipeline works without it

---

## Test Changes

### Integration test updates

The test discovery and field checks need to be type-aware:

```python
# Common checks (all types)
test_total, test_date, test_currency, test_payment_method, test_merchant_similarity

# Receipt-only checks
test_subtotal, test_line_items_count, test_line_items_totals,
test_tax_amount, test_tax_categories

# Utility bill checks (NEW)
test_service_type, test_usage_amount, test_billing_period

# Payment slip checks (NEW)
test_payer

# All types with points (NEW)
test_points_used, test_amount_paid
```

The test framework reads `document_type` from the truth file and only runs
applicable checks.

### Benchmark updates

`benchmark_models.py` field checks need the same type-awareness. Add
`document_type` to the check dispatch logic.

---

## New Files

| File | Purpose |
|---|---|
| `user_rules.json` | Merchant alias mapping + categories |
| `_truth_template.json` | Updated with new fields + LLM prompt |

## Modified Files

| File | Changes |
|---|---|
| `schema.py` | Add BillingPeriod, UsageData, expand Receipt→Document, new field registry entries |
| `pipeline.py` | Add document type detection, points extraction, merchant mapping, bill date logic |
| `extraction.py` | Type-specific prompt selection |
| `validation.py` | Type-aware validation |
| `cli.py` | No changes needed (outputs whatever Document contains) |
| `normalization.py` | May need bill-specific text cleanup |
| `tests/test_integration.py` | Type-aware field checks |
| `benchmark_models.py` | Type-aware field checks |

---

## Implementation Order

```
1. Schema (schema.py)
   - Add new models (BillingPeriod, UsageData)
   - Expand Receipt → Document with all new fields
   - Keep Receipt alias for backward compat
   - Add new FIELD_REGISTRY entries
   - Add type-specific prompt rules

2. Truth template (_truth_template.json)
   - Rewrite with new fields + LLM extraction prompt
   - Create truth files for receipts 14-21

3. Document type detection (pipeline.py)
   - Add keyword-based classifier
   - Wire into process_document before LLM call

4. Type-specific extraction (extraction.py, schema.py)
   - Add utility_bill and payment_slip prompt rules
   - Type-specific schema generation for Ollama structured output

5. Post-processing (pipeline.py)
   - Add points_used / amount_paid extraction
   - Add bill date logic
   - Add merchant mapping from user_rules.json

6. Validation (validation.py)
   - Add type-aware validation

7. Tests (test_integration.py)
   - Type-aware field checks
   - Add receipts 14-21 as fixtures with truth files

8. Benchmark (benchmark_models.py)
   - Type-aware field checks
```

---

## Truth File Validator (`validate_truth.py`)

Standalone script that validates all `*_truth.json` files against the schema
constraints defined in the Enum Reference above. Run after creating or editing
truth files to catch errors before they reach the test suite.

### What it checks

**Structural checks (all document types):**
- Required fields present: `document_type`, `currency`, `total`
- `document_type` is one of: `receipt`, `utility_bill`, `payment_slip`
- `currency` is one of: `JPY`, `USD`
- `payment_method` is one of: `cash`, `credit`, `debit`, `bank_payment`, `WAON`, or `null`
- `total` is a positive number
- `date` matches `YYYY-MM-DD` format (if not null)
- `points_used` is null or >= 0
- `amount_paid` is null or equals `total - points_used` (±1 tolerance)
- No unknown top-level keys (catches typos like `"merchnt"`)

**Receipt-specific checks (when `document_type == "receipt"`):**
- Each line item has `description` (non-empty string) and `total` (positive number)
- `tax_category` per item is one of: `8%`, `10%`, `0%`
- `discount` is null or >= 0
- `qty` is > 0
- If `unit_price` and `qty` set: `qty * unit_price - discount ≈ total` (±1)
- `subtotal ≈ sum(line_item.total)` (±2) when both present
- Tax entries have valid `rate` (string containing `%` or `unknown`)
- Utility/payment fields should be null/empty (no cross-contamination)

**Utility bill-specific checks (when `document_type == "utility_bill"`):**
- `service_type` is one of: `gas`, `water`, `electric`, `sewage`, `internet`, `phone`, or `null`
- `usage.unit` is one of: `m3`, `kWh`, `L`, or `null`
- `usage.amount` is > 0 (if set)
- `usage.meter_current > usage.meter_previous` (if both set)
- `billing_period.end > billing_period.start` (if both set)
- `line_items` should be empty

**Payment slip-specific checks (when `document_type == "payment_slip"`):**
- `merchant` should be present (non-null)
- `line_items` should be empty
- `subtotal` should be null

### CLI usage

```bash
# Validate all truth files in fixtures directory
python validate_truth.py

# Validate specific files
python validate_truth.py tests/fixtures/receipt_14_truth.json receipt_22_truth.json

# Validate all truth files across the entire project
python validate_truth.py --all
```

### Output format

```
Validating 37 truth files...

  receipt_14_truth.json .............. OK
  receipt_15_truth.json .............. OK
  receipt_17_truth.json .............. FAIL
    - payment_method "electronic" is not a valid value (expected: cash, credit, debit, bank_payment, WAON, or null)
    - line_items[0].tax_category "5%" is not a valid value (expected: 8%, 10%, 0%)
  receipt_22_truth.json .............. WARN
    - amount_paid is null but points_used is null and total is set (should amount_paid = total?)

Results: 35 OK, 1 FAIL, 1 WARN
```

Three severity levels:
- **FAIL** — violates a schema constraint (must fix before tests will work)
- **WARN** — suspicious but technically valid (e.g., missing amount_paid)
- **OK** — all checks pass

### Implementation notes

- Reads allowed values directly from hardcoded constants that mirror the Enum Reference
  section above. When adding new enum values, update both places.
- Discovers truth files via glob: `tests/fixtures/*_truth.json` + root `receipt_*_truth.json`
- `--all` flag also searches subdirectories recursively
- Exit code 0 if all pass, exit code 1 if any FAIL
- Can be wired into a pre-commit hook or CI step later

---

## API Cost Impact

No additional API cost. Document type detection uses OCR text that's already extracted.
The LLM still makes the same number of calls — just with different prompts per type.
