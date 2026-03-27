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

## Current Test Results (162/162 = 100% on benchmark, with known gap)

**Benchmark:** qwen3.5:9b × 6 fixtures × 3 runs × 9 fields = 162 checks, all passing.

| Test | Receipt 1 | Receipt 2 | Receipt 3 | Receipt 4 | Receipt 5 | Receipt 6 |
|---|---|---|---|---|---|---|
| total | PASS | PASS | PASS | PASS | PASS | PASS |
| date | PASS | PASS | PASS | PASS | PASS | PASS |
| currency | PASS | PASS | PASS | PASS | PASS | PASS |
| subtotal | PASS | PASS | PASS | PASS | PASS | PASS |
| payment_method | PASS | PASS | PASS | PASS | PASS | PASS |
| line_items_count | PASS | PASS | PASS | PASS | PASS | PASS |
| line_items_totals | PASS | PASS | PASS | PASS | PASS | PASS |
| tax_amount | PASS | PASS | PASS | PASS | PASS | PASS |
| merchant_similarity | PASS | PASS | PASS | PASS | PASS | PASS |

**Known gap:** `tax_category` per line item is NOT tested by the benchmark. The LLM frequently defaults to `"0%"` instead of correctly assigning `"8%"` or `"10%"` based on ※/X markers. This needs a new test assertion and either a prompt fix or pipeline post-processing step.

## Previous Test Results (before 2026-03-25 session)

69/71 passing (97%). Two failures:
- **Receipt 5 line_items_totals:** LLM set `本仕込食パン` qty=8 (misread "(8)" in product name as quantity), hallucinated `糸島のたまご` total=295 instead of 328
- **Receipt 6 payment_method:** LLM output "credit" instead of "cash" — triggered by `クレジット` appearing in loyalty card disclaimer text

## Implemented Fixes (initial session, 2026-03-24)

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

## LLM Benchmark Session (2026-03-25)

### Model Comparison

Benchmarked 6 models × 6 fixtures × 3 runs using `benchmark_models.py` (located in `local/`):

| Model | Accuracy | Consistency | tok/s | Avg Eval(s) | Errors |
|---|---|---|---|---|---|
| qwen3.5:9b | 156/162 (96%) | 100% | 22.4 | 28.1s | 0 |
| qwen3:8b | 150/162 (93%) | 100% | 41.0 | 10.1s | 0 |
| gemma3:4b | 132/162 (81%) | 100% | 69.5 | 12.6s | 0 |
| qwen2.5:7b | 123/162 (76%) | 100% | 49.6 | 10.9s | 0 |
| qwen3.5:4b | 105/162 (65%) | 100% | 55.1 | 8.6s | 0 |
| gemma3:12b | 92/162 (57%) | 83% | 14.2 | 27.6s | 5 |

Extra passes (3 instead of 2) made zero difference — validation only catches arithmetic errors, not comprehension failures.

### Root Cause Analysis (with Gemini second opinion)

**Receipt 5 / line_items_totals:**
- LLM misread `本仕込食パン (8)` as qty=8 (the "(8)" is pack size, not quantity)
- LLM hallucinated `糸島のたまご` total=295 (should be 328, price is on OCR line)
- Verification Rule 3 ("don't change the total") actively entrenched the error
- The `295` value appears nowhere in OCR text — pure hallucination

**Receipt 6 / payment_method:**
- LLM matched `クレジット` from `ビバ倶楽部カード(クレジット...除く)` ("excluding credit") as payment method
- `お預り ¥1,600` + `お釣り ¥30` are definitive cash indicators but were ignored
- `お釣り` amount was on the line ABOVE the label in OCR (unusual layout)

### Fixes Applied — Pipeline Post-Processing Approach

**Key learning: deterministic pipeline fixes beat prompt engineering.** Adding rules to the LLM prompt caused intermittent format violations where the model broke out of Ollama's `format=` structured output and generated explanation text. Moving logic to Python post-processing eliminated this instability.

#### extraction.py
- `sanitize_llm_response`: Added regex JSON block extraction as recovery for explanation-wrapped output (`re.search(r'\{.*\}', raw, re.DOTALL)`)

#### pipeline.py — New post-processing steps
- **Step 4.7 (payment_method):** `現計` in OCR = cash. `お預り` amount > total = cash (handles amounts on same or next line). `領収証` with no payment indicator = cash. Electronic methods only if cash indicators absent.
- **Step 4.8 (qty hallucination):** If LLM set qty > 1 but the resulting total doesn't appear in OCR text while unit_price does → reset qty=1, total=unit_price.
- **Step 4.9 (hallucinated totals):** If qty=1, discount=0, total ≠ unit_price, checks which value appears as a standalone number on the same OCR line as the item description. Uses word-boundary regex to avoid substring matches (e.g., `98` inside `980`). Corrects whichever value is wrong.

#### schema.py — Verification rules tightened
- Rule 3: "prefer unit_price from OCR text, set qty=1 unless explicit multiplier"
- Rule 4: "if sum doesn't match subtotal, look for items with qty > 1 that should be qty=1"
- Wording kept concise to avoid triggering explanation mode

#### validation.py
- Subtotal mismatch warning now includes actionable context about qty/product name confusion

#### normalization.py — Improved text processing
- **Barcode stripping:** Fixed regex to handle `JAN` with spaces (`^\d{8,}\s*(JAN|EAN)?\s*$`), item codes with spaces after them (`^0{2,}\d{1,4}[*]?\s*`)
- **`rejoin_price_lines` rewritten:** Surgical item-section-aware approach:
  1. Detects item section boundaries (first priced line → first summary marker like 小計)
  2. Block matching: N priceless items followed by N price lines → matched in order
  3. Single orphan lookback: searches up to 3 lines back for a priceless item (skips lines with existing prices)
  4. Lines outside the item section are never modified

### Approaches Tried and Rejected

**Prompt-based fixes (Rules 10, 16):** Adding payment method and quantity rules directly to the LLM prompt caused intermittent format violations. The model generated explanation text ("Based on Rule 16...") instead of pure JSON. Reverted in favor of pipeline post-processing.

**Option A — Paragraph-level bounding boxes:** Changed `run_cloud_vision` to use `_extract_blocks_from_response` (real bboxes) instead of fulltext. Failed: paragraph blocks for rotated receipts had nearly identical y-coordinates (~1024-1047), grouping everything into one massive line.

**Option B — Word-level extraction:** Extracted word-level blocks from Cloud Vision with real bounding boxes. X-coordinate matching was excellent for associating items with prices on rotated receipts. But word-level text has spaces between Japanese characters (`小 計` instead of `小計`), breaking regex matching and LLM parsing. Scored 129/162 (80%) — major regression.

**Option C — Text-based rejoining (adopted):** Enhanced `rejoin_price_lines` to detect block patterns (N items then N prices) and match them in order, within a detected item section. No bounding box changes needed. Handles both normal and rotated receipt layouts through text pattern analysis.

### OCR Non-Determinism Discovery

Cloud Vision is non-deterministic — same image can produce different text layouts across API calls. During testing, deleting the OCR cache and regenerating produced different formatting (labels and amounts on same line vs separate lines) that broke previously working receipts. The pipeline was hardened to handle both variants:
- `お預り` detection handles amount on same line or next line
- `_extract_yen_nearby` looks ahead 2 lines for ¥ values
- `rejoin_price_lines` handles both inline and separated item/price layouts

The `.ocr_cache/` directory provides determinism for known images and should be committed to git.

## What to Try Next

### 1. Fix tax_category per line item (PRIORITY)
The LLM defaults all `tax_category` to `"0%"` instead of correctly assigning `"8%"` (reduced, marked ※/X) or `"10%"` (standard). The benchmark doesn't test this field. Needs:
- Add `tax_category` assertion to `test_integration.py` and `benchmark_models.py`
- Fix via pipeline post-processing (scan OCR text for ※/X markers near items) or prompt improvement
- Be cautious with prompt changes — they can trigger explanation mode (see "Approaches Tried and Rejected")

### 2. Commit OCR cache to git
`.ocr_cache/` is not tracked. If deleted and regenerated, Cloud Vision non-determinism may produce different text requiring pipeline adjustments. Commit to preserve known-good OCR results.

### 3. Image auto-rotation
Detect rotated receipts before OCR and rotate to upright. Would make paragraph mode viable (better item/price grouping) and fix date misreads.

### What NOT to pursue

- **PaddleOCR**: removed for good reason — worse accuracy, heavy dependencies
- **Vision models via Ollama**: broken due to Ollama `format=` + images bug
- **Document AI / Azure as primary**: both lack Japanese receipt format understanding
- **`Optional[Literal[...]]` in schema**: Ollama can't handle `anyOf` — use non-Optional with defaults
