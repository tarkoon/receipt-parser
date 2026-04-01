# Pipeline Robustness & Confidence Scoring Plan

> **Base branch:** `master`
> **IMPORTANT:** Create a new branch from `master` before starting implementation. 

---

## Background & Motivation

A deep audit of the receipt-parser pipeline revealed that while the overall architecture (OCR -> normalize -> LLM extract -> validate) is sound, the pipeline is brittle in ways that would cause failures on unseen receipts. The core problem: **too much work is done in hardcoded regex that should be delegated to the LLM**, and there is **no confidence-based decision routing** to know when regex overrides are safe vs dangerous.

---

## Audit Findings

### Critical Issues

#### 1. OCR Regex Unconditionally Overrides LLM Output
**Location:** `pipeline.py:583-620`
`_extract_financial_totals()` runs regex against OCR text, then **replaces** the LLM's extracted values regardless of whether the LLM was correct. If the regex misparses (unusual layout, unexpected spacing), correct LLM output gets overwritten with garbage and there is no fallback.

#### 2. ~250 Lines of Japanese-Only Regex
**Locations:** `pipeline.py:25-65, 140-250, 321-400, 623-650, 857-868`
Every post-processing step is hardwired to Japanese vocabulary:
- `小計`/`合計`/`現計` for subtotal/total detection
- `消費税`/`内税`/`外税` for tax type detection
- `お預り`/`お釣り`/`釣銭` for cash payment detection
- `ポイント利用`/`ポイント値引` for points extraction
- `レジ袋` hardcoded to 10% tax category (`pipeline.py:360-361`)
- `※`/`X`/`軽`/`除` as tax rate markers

Any non-Japanese receipt or a Japanese receipt with non-standard terminology will fail.

#### 3. Era Date Magic Numbers
**Locations:** `pipeline.py:630-636`, `extraction.py:153-164`
The Reiwa era offset `2018` is a magic number with no era name detection. Only 令和 (Reiwa) is handled. 平成 (Heisei) receipts from before May 2019 will produce incorrect dates. (Note: we only need 令和 and 平成 — no receipts older than 1989.)

#### 4. Yen-as-Digit-1 Fix Corrupts Valid Data
**Location:** `normalization.py:203-217`
Leading `1` near `金額` is replaced with `金額:` to fix ¥-misread-as-1. But legitimate amounts starting with 1 (e.g., ¥13,500) get corrupted to `金額:3500`. The LLM can reason about context; regex can't.

#### 5. Handwritten Receipt Detection is Fragile
**Location:** `normalization.py:166-170`
Uses `len(lines) < 35` + absence of `小計`/`合計` to classify as handwritten. A short printed receipt (coffee shop, 3 items) gets misclassified and has its ¥ symbols stripped.

#### 6. Frozen Tax Rate Enum
**Locations:** `schema.py`, `extraction.py:134-140`, `validation.py:75`
`Literal["8%", "10%", "0%"]` is hardcoded everywhere. Japan has changed tax rates before (5% -> 8% -> 10%). The subset-sum assignment in `_assign_tax_categories` also assumes exactly these rates.

#### 7. Shadow Schema in `_coerce_llm_output`
**Location:** `extraction.py:100-170`
70 lines of field renaming (`quantity`->`qty`, `name`->`description`), type coercion, and format fixes that exist **outside** the Pydantic schema. This drifts from the real schema and duplicates logic.

#### 8. Dual-Call OCR Burns 2x API Budget
**Location:** `ocr.py` — `run_cloud_vision()`
Every image costs 2 API calls. With 1000 free calls/month, that's ~500 receipts max. The "more ¥ symbols = better" selection heuristic is fragile.

#### 9. Price Line Rejoin is Positional
**Location:** `normalization.py:45-140`
95 lines assuming orphan prices appear in same order as items within a contiguous block. Promotional text between items or variable spacing causes misalignment.

#### 10. Barcode Detection is JAN/EAN Only
**Location:** `normalization.py:32-37`
Only strips JAN/EAN. UPC-A, GTIN-14, Code 128 barcodes pass through and may be misinterpreted as prices.

### What's Working Well
- Schema-driven extraction via `FIELD_REGISTRY` + Pydantic models
- Multi-pass LLM verification with self-correction
- Arithmetic validation in `validation.py` (deterministic, format-independent)
- OCR caching by image hash
- Debug visualization with confidence color-coding

---

## Confidence Scoring System (New)

### Current State

| Layer | What Exists | What's Missing |
|---|---|---|
| **OCR** | Per-block confidence from Cloud Vision (filtered at <0.5, color-coded in debug) | No aggregate score; no confidence-based routing |
| **LLM** | `_line_items_reliable` boolean flag (pass/fail) | No per-field confidence; no granular scoring |
| **Pipeline** | Fixed control flow regardless of input quality | No confidence-based decision routing |

### Target State

#### A. OCR Confidence Score
Aggregate block-level confidences into a single document-level OCR quality score:

```python
def compute_ocr_confidence(blocks: list[dict]) -> float:
    """Weighted average of block confidences, weighted by text length."""
    if not blocks:
        return 0.0
    total_chars = sum(len(b["text"]) for b in blocks)
    if total_chars == 0:
        return 0.0
    return sum(b["confidence"] * len(b["text"]) for b in blocks) / total_chars
```

**Use this score to:**
- **Replace dual-call OCR:** Only make a second API call if `ocr_confidence < 0.75` (instead of always calling twice)
- **Gate regex overrides:** Only let regex override LLM output when `ocr_confidence > 0.85` AND the regex extraction passes arithmetic validation
- **Include in output metadata** as `_ocr_confidence` for downstream consumers

#### B. LLM Field-Level Confidence
Ask the LLM to return a confidence object alongside extracted fields:

```json
{
  "merchant": "セブンイレブン",
  "total": 1500,
  "_confidence": {
    "merchant": 0.95,
    "total": 0.90,
    "date": 0.70,
    "line_items": 0.60
  }
}
```

**Use this to:**
- **Selective post-processing:** Only apply regex fixes to fields where LLM confidence < threshold
- **Selective verification passes:** Only re-extract fields the LLM was unsure about (reduces token usage)
- **Trust routing:** High-confidence LLM output bypasses regex override entirely

#### C. Pipeline Confidence Router
Replace the current fixed control flow with confidence-based routing:

```
IF ocr_confidence >= HIGH and llm_confidence[field] >= HIGH:
    → Trust LLM output directly, skip regex override
IF ocr_confidence >= HIGH and llm_confidence[field] < HIGH:
    → Use regex extraction as validation signal, warn on disagreement
IF ocr_confidence < HIGH:
    → Trigger second OCR call, then re-extract with LLM
    → Apply regex as fallback only if LLM still fails validation
```

---

## Implementation Phases

> **Reminder:** Create a new branch from `master` before starting. Base implementation on the work in `worktree-deepseek-testing`.

### Phase 1: Confidence Infrastructure
**Goal:** Add confidence scoring without changing pipeline behavior yet.

- [ ] **1.1** Add `compute_ocr_confidence()` to `ocr.py` — weighted average of block confidences
- [ ] **1.2** Add `_ocr_confidence` to pipeline output metadata (alongside `_line_items_reliable`, `_warnings`, etc.)
- [ ] **1.3** Add `_confidence` field to LLM extraction prompt — update `schema.py` prompt to request per-field confidence scores
- [ ] **1.4** Parse and validate `_confidence` from LLM output in `extraction.py`
- [ ] **1.5** Add `_llm_confidence` to pipeline output metadata
- [ ] **1.6** Unit tests for confidence computation and parsing

### Phase 2: Conditional Dual-Call OCR
**Goal:** Cut API usage ~50% by only making second call when needed.

- [ ] **2.1** Refactor `run_cloud_vision()` to single-call-by-default
- [ ] **2.2** Add retry logic: second call only if `ocr_confidence < 0.75`
- [ ] **2.3** Benchmark: compare accuracy of single-call+retry vs always-dual-call across all fixtures
- [ ] **2.4** Update API usage tracking and budget estimates

### Phase 3: Trust Inversion — LLM Over Regex
**Goal:** Flip the override model so regex validates LLM output instead of replacing it.

- [ ] **3.1** Refactor `_extract_financial_totals()` to return values WITHOUT overriding LLM output
- [ ] **3.2** Add confidence router: only override LLM output when `ocr_confidence > 0.85` AND LLM confidence for that field < 0.5 AND regex result passes arithmetic validation
- [ ] **3.3** Apply same pattern to `_extract_points_used()`, `_assign_tax_categories()`, payment method detection
- [ ] **3.4** Integration tests: verify all existing fixtures still pass with new routing
- [ ] **3.5** Add "unseen receipt" test set — receipts from stores not in current fixtures — to validate generalization

### Phase 4: Fix Brittle Patterns
**Goal:** Address the hardcoded/bespoke issues found in the audit.

- [ ] **4.1** Era date fix — add proper era-to-year converter with both 令和 (base 2018) and 平成 (base 1988) support, with era name matching from OCR text
- [ ] **4.2** Move `_coerce_llm_output` logic into Pydantic model — use `Field(alias=...)`, `field_validator`, `model_validator` so the schema is the single source of truth
- [ ] **4.3** Remove yen-as-digit-1 string manipulation — replace with LLM prompt rule for handwritten amounts
- [ ] **4.4** Fix handwritten receipt detection — use OCR confidence distribution (low avg confidence = handwritten) instead of line count heuristic
- [ ] **4.5** Make tax rates configurable — extract `VALID_TAX_RATES`, `REDUCED_RATE`, `STANDARD_RATE` into config rather than `Literal` hardcoding
- [ ] **4.6** Generalize barcode stripping — match `^\d{8,}\s*$` (any long digit-only line) instead of JAN/EAN-specific pattern
- [ ] **4.7** Remove `レジ袋` hardcode (`pipeline.py:360-361`) — let the LLM assign tax categories based on context, not product name matching

### Phase 5: Reduce Regex Post-Processing
**Goal:** Gradually shift semantic extraction from regex to LLM.

- [ ] **5.1** Move payment method detection into LLM prompt (currently ~20 lines of regex in `pipeline.py:623-648`)
- [ ] **5.2** Move price-line rejoining into LLM prompt rule instead of `rejoin_price_lines()` (95 lines) — add prompt rule: "Prices may appear on separate lines below their items — match them by proximity"
- [ ] **5.3** Simplify tax category assignment — encode `※`=8%, `除`=exempt rules in prompt, use `_assign_tax_categories` only as validation fallback
- [ ] **5.4** Benchmark each change: confirm no regression on fixtures before and after

### Phase 6: Instructor Library Integration (Optional)
**Goal:** Replace manual coercion/parsing chain with Pydantic-native structured output.

- [ ] **6.1** Evaluate `instructor` library (`pip install instructor`) for Ollama and OpenRouter backends
- [ ] **6.2** If viable: replace `sanitize_llm_response` + `_coerce_llm_output` + `_parse_llm_json` chain with `instructor` structured output + automatic retry
- [ ] **6.3** Benchmark: compare extraction accuracy and latency

---

## Success Criteria

1. **All existing fixtures pass** — no regressions on current test set
2. **OCR API usage reduced ~50%** — single-call default with confidence-based retry
3. **No unconditional regex overrides** — every override gated by confidence scores
4. **Era dates handle 令和 + 平成** — no magic numbers without named constants
5. **No product-name hardcodes** — `レジ袋` and similar removed
6. **`_coerce_llm_output` eliminated** — all coercion lives in Pydantic models
7. **Unseen receipt test set** added — pipeline works on receipts from new stores without code changes
