# Pipeline Improvements

## Approved to Merge

### 1. Tax Ratio Cross-Check & Discount Rate Consistency (Steps 1.1, 1.2)
- Validates `subtotal * 1.08` or `subtotal * 1.10 ≈ total` — catches gross extraction errors before subset-sum matching
- Checks that `discount_rate` (e.g. "20%") is consistent with the actual `discount` amount on each line item
- Zero risk, zero cost — pure additions to `validation.py` and `schema.py` soft validators
- Feeds into multi-pass verification: when the LLM makes an arithmetic error, these warnings tell it exactly what to fix

### 2. Tree Edit Distance Metric (Step 5.1)
- Structural accuracy check comparing the full JSON output against truth
- Catches errors per-field checks miss: duplicated items, wrong nesting, swapped fields, array length mismatches
- Added to `COMMON_CHECKS` so it runs for all document types
- Threshold set at 0.5 (loose enough to avoid false failures on optional null fields)

### 3. amount_paid Bug Fix
- `process_ocr_text` was missing `amount_paid = total - points_used` — all OCR variant tests relied on the LLM returning this field correctly by chance
- One-line fix that aligns the OCR text path with the image path

## On Hold

### 4. Correction Tracking (Step 5.2)
- Standalone JSONL logger for user corrections — `log_correction()` / `load_corrections()`
- Sound concept but no integration with the pipeline or UI yet
- Revisit when we build a review interface

### 5. Separator Detection (Step 2.1)
- Tags receipt sections (`## SECTION: HEADER/ITEMS/TOTALS/FOOTER`) using `===`/`---` separator lines
- Never fires on current fixtures — no receipts have standalone separator lines
- The LLM already understands receipt structure without hints
- Risk of mislabeling sections with crude keyword heuristics

### 6. Furigana Filtering (Step 2.2)
- Removes short kana lines identified as ruby text annotations
- Caused multiple regressions (deleted merchant names, OCR fragments) requiring three rounds of fixes
- No furigana exists in current fixtures — false-positive risk outweighs benefit
- Revisit if we encounter receipts with actual furigana contamination

### 7. Spatial Position Tagging (Step 2.3)
- Tags OCR blocks with `[TOP]`/`[BOTTOM]`/`[PRICE?]`/`[LARGE]` based on bounding box coordinates
- Only works on fresh OCR (not cached) — silently disappears after first run
- No proven accuracy improvement (39/40 vs 40/40 baseline)
- The LLM already handles layout well; existing text-based approaches (price rejoining, structured grouping) solve the same problem

### 8. Few-Shot Examples (Step 3.1)
- Adds example OCR-to-JSON pairs to the extraction prompt
- Caused merchant extraction regressions — biased the LLM toward example patterns
- Added ~3KB (60%) to prompt size
- Needs a fundamentally different approach (dynamic example selection) to work

### 9. Confidence Wrapping (Step 4.1)
- Infrastructure for LLM to report per-field confidence alongside values
- Never activated — the prompt doesn't ask for confidence output
- The plan warned this would double schema complexity and likely degrade extraction quality
