# Receipt Parser — Testing & Evaluation Report

*Last updated: 2026-03-24*

## Project Overview

Receipt parser for Japanese receipts using Google Cloud Vision OCR + local LLM (qwen3.5:9b via Ollama) for structured data extraction. Converts receipt images into structured JSON with merchant name, date, line items, taxes, and totals.

## Architecture

```
Receipt Image → Cloud Vision OCR (fulltext, cached) → Normalization → Barcode Stripping
    → Price-Line Rejoining → Handwritten Cleanup → qwen3.5:9b LLM
    → Regex Financial Totals Override → Date Correction → Validation → JSON
```

**Stack:**
- **OCR:** Google Cloud Vision API (`document_text_detection`, fulltext mode, cached per image hash)
- **LLM:** qwen3.5:9b via Ollama (local, structured output with `format=` + `think=False`)
- **Post-processing:** Regex extraction of subtotal/total/tax from OCR text (overrides LLM)
- **Validation:** Pydantic schema + arithmetic cross-checks
- **Multi-pass:** Pass 1 extracts, Pass 2 corrects based on validation warnings
- **Normalization:** NFKC, barcode stripping, price-line rejoining, handwritten cleanup
- **Date correction:** Japanese era dates, OCR year misreads (201X → 202X)
- **API tracking:** Monthly Cloud Vision call counter with free tier warnings

## Test Suite

### Unit Tests (17 tests, ~1.5s, no API calls)
```powershell
python -m pytest tests/test_unit.py -v
```
Tests normalization, validation, schema, prompt generation, handwritten cleanup, API tracking, and mocked pipeline.

### Integration Tests (auto-discovered, ~2-4min, uses cached OCR + Ollama)
```powershell
python -m pytest tests/test_integration.py -v
```
Auto-discovers all `*_truth.json` + matching image files in `tests/fixtures/`. Each fixture gets 9 field tests: total, date, currency, subtotal, payment_method, line_items_count, line_items_totals, tax_amount, merchant_similarity.

To add a new fixture: drop `my_receipt.jpg` + `my_receipt_truth.json` in `tests/fixtures/`. No code changes needed.

**OCR caching:** Cloud Vision results are cached in `.ocr_cache/` by image hash. First run calls the API (2 calls per receipt); subsequent runs use cached text. Delete `.ocr_cache/` to force fresh API calls.

## Test Receipts

| Receipt | Type | Merchant | Key Challenges |
|---|---|---|---|
| **Receipt 1** | Printed supermarket (thermal) | サンリブ くりえいと宗像 | Decorative/logo font on merchant name |
| **Receipt 2** | Printed supermarket (thermal, AEON) | マックスバリュくりえいと宗像店 | Fulltext OCR separates items from prices on different lines |
| **Receipt 3** | Handwritten 領収証 | とも笑助産院 | Handwritten text, ¥ misread as digit 1, era date |
| **Receipt 4** | Printed supermarket (rotated 90°) | 業務スーパー 新宗像店 | Image rotated, 10 line items, JAN barcodes, date misread |
| **Receipt 5** | Printed supermarket (AEON, discount) | マックスバリュくりえいと宗像店 | 20% discount line on an item, credit card payment |
| **Receipt 6** | Printed hardware store | スーパービバホーム 赤間店 | Rotated, 2 items only, cash payment |

## Current Test Results (69/71 passing = 97%)

| Test | Receipt 1 | Receipt 2 | Receipt 3 | Receipt 4 | Receipt 5 | Receipt 6 |
|---|---|---|---|---|---|---|
| total | PASS | PASS | PASS | PASS | PASS | PASS |
| date | PASS | PASS | PASS | PASS | PASS | PASS |
| currency | PASS | PASS | PASS | PASS | PASS | PASS |
| subtotal | PASS | PASS | PASS | PASS | PASS | PASS |
| payment_method | PASS | PASS | PASS | PASS | PASS | **FAIL** |
| line_items_count | PASS | PASS | PASS | PASS | PASS | PASS |
| line_items_totals | PASS | PASS | PASS | PASS | **FAIL** | PASS |
| tax_amount | PASS | PASS | PASS | PASS | PASS | PASS |
| merchant_similarity | PASS | PASS | PASS | PASS | PASS | PASS |

**Receipt 5 items:** LLM merges multiple items incorrectly when processing the discount — needs model tuning or a better model.
**Receipt 6 payment:** LLM says "credit" instead of "cash" despite お預り/お釣り indicators — LLM non-determinism.

## Implemented Fixes (this session)

### 1. Price-line rejoining (normalization.py)
**Problem:** Cloud Vision fulltext puts items and prices on separate lines, confusing the LLM.
**Fix:** `rejoin_price_lines()` joins orphan `¥NNN` lines (and `NNN※`/`NNN除` tax-marked lines) upward with the preceding Japanese text line, as long as it doesn't already contain `¥`. ~20 lines.
**Impact:** Fixed Receipt 2 line item extraction (was 0 items, now 4/4).

### 2. OCR caching (ocr.py)
**Problem:** Cloud Vision returns different text on each call for the same image, causing non-deterministic test results.
**Fix:** Cache fulltext per image MD5 hash in `.ocr_cache/`. First call uses API + caches; subsequent calls hit cache.
**Impact:** Tests are now deterministic. Zero API calls on cached runs. ~2 min test time (vs 3-6 min without cache).

### 3. Regex financial totals extraction (pipeline.py)
**Problem:** LLM confuses 課税対象額 (taxable base) with 税額 (tax amount), or picks wrong ¥ value for totals.
**Fix:** `_extract_financial_totals()` scans OCR text for 小計/合計/現計/外税N%/税額 labels and extracts adjacent ¥ values. Runs BEFORE price-line rejoining (when values are on their own lines). Overrides LLM's subtotal/total/taxes. Falls back to `tax = total - subtotal` arithmetic.
**Impact:** Fixed Receipt 4 tax (was 2279, now 168). Financial totals are now rock-solid.

### 4. tax_category enum (schema.py)
**Problem:** LLM outputs free-form strings like "reduced", "exempt", "standard" for tax_category.
**Fix:** Changed to `Literal["8%", "10%", "0%"]` with default `"0%"`. Used non-Optional type to avoid Ollama's `anyOf` schema bug. Added coercion in `_coerce_llm_output()` to map invalid values.
**Impact:** Consistent tax category values across all receipts.

### 5. Discount fields (schema.py)
**Problem:** Discount lines (e.g., "割引 20% -¥94") were modeled as negative line items.
**Fix:** Added `discount: float = 0` and `discount_rate: str = ""` to LineItem. Prompt rule 15 tells LLM to merge discounts into parent items. Used non-Optional types to avoid Ollama schema issues.
**Impact:** Receipt 5 salmon shows `total: 373, discount: 94, discount_rate: "20%"` instead of two separate items.

### 6. Prompt improvements (schema.py)
- Rule 10 updated: specific payment methods (WAON, credit) override cash indicators
- Rule 14 added: arithmetic matching for disconnected labels and values
- Rule 15 added: discount merging into parent items
- Total: 15 extraction rules

### 7. Ollama timeout (extraction.py)
Increased from 120s → 180s to handle longer prompts.

## Ollama Schema Constraints — Known Issues

`Optional[Literal[...]]` generates `anyOf` in JSON Schema, which Ollama handles poorly with qwen3.5 — causes empty/null responses. Workaround: use non-Optional types with defaults instead.

```python
# Broken (generates anyOf → empty LLM output):
tax_category: Optional[Literal["8%", "10%", "0%"]] = None

# Works (simple enum, no anyOf):
tax_category: Literal["8%", "10%", "0%"] = "0%"
```

Same issue applies to `Optional[float]` and `Optional[str]` for new fields — use `float = 0` and `str = ""` instead when adding fields to the Ollama-facing schema.

## Backends Evaluated

| Backend | Printed Receipts | Handwritten | Rotated | Verdict |
|---|---|---|---|---|
| **Cloud Vision + LLM** | Excellent | Good (with fixes) | Good (fulltext) | **Selected** |
| PaddleOCR + LLM | Good | Not tested | Poor | Removed (complexity) |
| Document AI Expense Parser | Poor (JP) | Poor | N/A | Not suitable for Japanese |
| Azure Doc Intelligence | Poor items | Best ¥ reading | N/A | Handwriting only |
| qwen3-vl (vision LLM) | Unreliable | Timeout | N/A | Ollama format= bug |
| gemma3 / minicpm-v | Hallucinated | N/A | N/A | Not suitable |

## Key Technical Findings

### 1. Regex beats LLM for financial totals
The highest-impact improvement was extracting subtotal, total, and tax directly from OCR text via regex — bypassing the LLM entirely for these fields. The LLM now focuses on what it's good at: parsing item names, quantities, and matching disconnected prices.

### 2. Prompt engineering drives line item accuracy
Major improvements came from adding rules to `BASE_EXTRACTION_RULES` in `schema.py` (now 15 rules). The most impactful rules:
- Associating items with prices on separate lines (rule 11)
- Distinguishing 課税対象額 from 税額 (rule 13)
- Arithmetic matching for disconnected labels/values (rule 14)
- Discount merging (rule 15)

### 3. Cloud Vision fulltext vs paragraph mode tradeoff
Fulltext mode handles rotation correctly but separates items from prices. The price-line rejoining normalization step mitigates this by joining orphan `¥NNN` lines with their preceding item/label.

### 4. OCR caching eliminates the biggest source of test flakiness
Cloud Vision is non-deterministic — same image, different text each call. Caching by image hash makes tests reproducible and saves API credits.

### 5. Ollama structured output has sharp edges with Optional types
`anyOf` schema (from `Optional[Literal[...]]` or new `Optional` fields) breaks qwen3.5. Non-Optional types with defaults are the safe path.

### 6. Handwritten ¥ → 1 is a persistent OCR issue
Cloud Vision sometimes reads the handwritten yen sign ¥ as the digit 1, turning ¥3000 into 13000. Mitigated by `clean_handwritten_ocr()` with ¥→金額 replacement and absorbed-¥ detection.

### 7. Japanese era dates need post-processing
The pipeline extracts dates directly from OCR text using regex, overriding the LLM:
- Western dates: `2026年03月22日` or `2026/3/11` → extract directly
- Era dates: `7年12月22日` → `2025-12-22` (令和)
- OCR misreads: `2016年` on rotated images → `2026年` (201X→202X)

## File Structure

```
receipt-parser/
├── cli.py              — Typer CLI (parse + usage commands)
├── pipeline.py         — Pipeline orchestrator + regex financial extraction
├── ocr.py              — Cloud Vision OCR + caching + API tracking
├── extraction.py       — Ollama LLM extraction + multi-pass verification
├── schema.py           — Pydantic models, prompt generation (15 extraction rules)
├── validation.py       — Arithmetic cross-checks (discount-aware)
├── normalization.py    — NFKC, barcode stripping, price-line rejoining, handwritten cleanup
├── preprocessing.py    — Image loading, PDF conversion, EXIF rotation
├── debug_visual.py     — Bounding box overlays, pipeline trace
├── .env                — GOOGLE_CLOUD_PROJECT (loaded automatically)
├── .ocr_cache/         — Cached Cloud Vision fulltext (per image hash)
├── conftest.py         — Loads .env, adds project root to sys.path
├── pyproject.toml      — Pytest config, warning filters
├── requirements.txt
├── TESTING_REPORT.md   — This file
├── tests/
│   ├── fixtures/       — Receipt images + *_truth.json (auto-discovered)
│   │   └── _truth_template.json  — Template for new fixtures
│   ├── test_unit.py    — 17 fast unit tests
│   └── test_integration.py — Auto-discovered fixture tests (9 per receipt)
└── debug/              — Debug artifacts (when --debug flag used)
```

## What to Try Next

### 1. Better LLM model
qwen3.5:9b handles most cases but intermittently merges items incorrectly or picks wrong payment methods. Worth benchmarking:
- **qwen3:14b** — same family, larger context for complex receipts
- **gemma-3:12b** — strong multilingual, potentially better structured output
- **phi-4:14b** — strong instruction following

Constraint: must fit in 8GB VRAM (RTX 4070 Laptop).

### 2. Image auto-rotation
Detect rotated receipts before OCR and rotate to upright. Would make paragraph mode viable (better item/price grouping) and fix date misreads.

### 3. Receipt 5 discount handling
The LLM sometimes merges discount items with the wrong parent or combines multiple items. May need post-processing to detect and fix discount associations.

### What NOT to pursue

- **PaddleOCR**: removed for good reason — worse accuracy, heavy dependencies
- **Vision models via Ollama**: broken due to Ollama `format=` + images bug
- **Document AI / Azure as primary**: both lack Japanese receipt format understanding
- **`Optional[Literal[...]]` in schema**: Ollama can't handle `anyOf` — use non-Optional with defaults
