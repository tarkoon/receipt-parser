# Pipeline Brittleness Audit

Date: 2026-06-12

Scope: `src/receipt_parser/pipeline.py` and `src/receipt_parser/pipeline_receipt.py` as they exist in the current working tree. This is an audit only: no truth files, fixtures, tests, or pipeline behavior were changed.

Guiding rule: per the debug-receipt skill, production fixes should be general-purpose. A rule is brittle when it only works because one known receipt has known OCR tokens, item names, dates, totals, or fixture-range context.

## Executive Summary

`pipeline_receipt.py` has crossed from "receipt-specific post-processing" into a large, order-sensitive patch stack. The highest-risk area is the late override section around `src/receipt_parser/pipeline_receipt.py:10487`, where distinctive OCR token checks replace full financial fields and full `line_items` lists with hardcoded values. Those branches are not general receipt logic; they are embedded answer keys.

The second major risk is repeated mutation in `postprocess_receipt` at `src/receipt_parser/pipeline_receipt.py:11496`. The same mutators are run multiple times in one call, often after other mutators have rewritten the same fields. That makes correctness depend on accidental ordering and on whether each helper is idempotent.

`pipeline.py` is much smaller, but it now reaches into private receipt helpers and re-applies a subset of receipt postprocessing in `_build_result` at `src/receipt_parser/pipeline.py:541`. That creates a second receipt postprocess path outside the main candidate selector.

## High-Risk Brittle Code

### 1. Hardcoded full-receipt reconstruction

Evidence:

- `src/receipt_parser/pipeline_receipt.py:10487` defines `_known_item`, a helper for constructing fixed item dictionaries.
- `src/receipt_parser/pipeline_receipt.py:10507` through `src/receipt_parser/pipeline_receipt.py:10697` contain functions named for specific fixture/merchant layouts, including `_fix_maxvalu_receipt_86_layout`, `_fix_nishimatsuya_receipt_89_layout`, `_fix_maxvalu_receipt_97_layout`, `_fix_maxvalu_receipt_98_layout`, and `_fix_costco_receipt_99_layout`.
- `src/receipt_parser/pipeline_receipt.py:10708` defines `_apply_target_101_182_layout_overrides`, a roughly 600-line chain of token-gated full-field overrides.
- `src/receipt_parser/pipeline_receipt.py:11308` defines `fix_final_known_financial_overrides`, another last-mile override chain that rewrites totals, taxes, and full item lists.

Why this is brittle:

These branches are not extracting from receipt structure. They match distinctive token combinations and then assign known totals, tax entries, payment values, locations, and item lists. This is equivalent to embedding fixture answers in the production pipeline. It may improve benchmark scores for the known receipts while making future receipts harder to reason about and easier to accidentally corrupt when a different receipt shares enough tokens.

Recommended cleanup:

- Move full-receipt override branches out of the production pipeline, or quarantine them behind an explicit benchmark/debug-only compatibility mode.
- Replace each branch with a generalized layout detector where possible. For example, "dense item name block plus price block" should become one tested parser, not one branch per store/date/item set.
- Add a guardrail test that fails when new production code introduces function names or comments containing fixture ranges like `receipt_###`, `target_###`, or hardcoded full `line_items` answer lists.

### 2. Large merchant/layout-specific blocks in general postprocessing

Evidence:

- `_fix_starbucks_receipt_layout` starts at `src/receipt_parser/pipeline_receipt.py:5187`.
- MaxValu-specific logic appears in several separate helpers, including `src/receipt_parser/pipeline_receipt.py:5623`, `src/receipt_parser/pipeline_receipt.py:5671`, `src/receipt_parser/pipeline_receipt.py:7267`, `src/receipt_parser/pipeline_receipt.py:7452`, `src/receipt_parser/pipeline_receipt.py:7621`, `src/receipt_parser/pipeline_receipt.py:9758`, and `src/receipt_parser/pipeline_receipt.py:9813`.
- `_fix_maxvalu_suffix_marker_rows` alone is about 397 lines and contains multiple token-gated item-list replacements starting at `src/receipt_parser/pipeline_receipt.py:9813`.
- FamilyMart, Daiso, Gyomu, Seria, Donki, Cosmos, Nafco, and other brand/layout handlers are all run from the same postprocess chain.

Why this is brittle:

Some store-specific logic can be legitimate when a chain prints a stable layout. The risk here is that several helpers go beyond parsing a stable layout and instead rewrite item identities and totals from known product combinations. Brand-specific code also spreads business logic across many independent helpers, so a new MaxValu failure may be "fixed" by adding another branch instead of improving a shared MaxValu layout parser.

Recommended cleanup:

- Split store/layout handling into named parser strategies with explicit inputs, outputs, and invariants.
- Keep strategies generic to a printed layout pattern, not to known item names or known totals.
- Add tests that each strategy is idempotent and only fires when its structural preconditions are met.

### 3. Repeated mutation and ordering dependence in `postprocess_receipt`

Evidence:

`postprocess_receipt` at `src/receipt_parser/pipeline_receipt.py:11496` runs many mutators repeatedly. Static counting inside that function found these repeated calls:

| Helper | Calls in `postprocess_receipt` |
|---|---:|
| `_fix_maxvalu_suffix_marker_rows` | 8 |
| `_drop_duplicate_with_embedded_price` | 5 |
| `_fix_daiso_qty_and_reduced_rate_context` | 5 |
| `_fix_duplicate_descriptions_from_ocr` | 5 |
| `_fix_o_ring_descriptions_from_ocr` | 5 |
| `_replace_maxvalu_sequence_rows_when_balanced` | 4 |
| `_fix_tax_categories_from_price_line_markers` | 4 |
| `_drop_non_product_line_items` | 4 |
| `_replace_barcode_qty_price_rows_when_balanced` | 3 |
| `_replace_jan_pos_items_when_balanced` | 3 |

Why this is brittle:

Repeated mutation can be intentional when a later repair exposes a new cleanup opportunity, but the current chain has no phase model or invariant checks explaining why each repeat is necessary. This makes the result sensitive to helper order. It also means a helper that is not idempotent can change output on the second, third, or eighth call.

Recommended cleanup:

- Replace the long imperative chain with named phases, for example: `financial_fields`, `item_recovery`, `item_cleanup`, `tax_assignment`, `final_consistency`.
- For each repeated helper, document the phase dependency or collapse it to one call.
- Add an idempotence test: running `postprocess_receipt` twice on the same extracted payload and OCR text should not change the second output.
- Add debug tracing of before/after field diffs per phase so future fixes can prove which invariant they repair.

### 4. Receipt postprocessing leaks into `pipeline.py`

Evidence:

- `pipeline.py` imports private receipt helpers at `src/receipt_parser/pipeline.py:35`.
- `_build_result` applies `_replace_barcode_qty_price_rows_when_balanced`, `_fix_maxvalu_suffix_marker_rows`, `fix_final_known_financial_overrides`, and `_drop_duplicate_with_embedded_price` at `src/receipt_parser/pipeline.py:541`.
- The main candidate selector already postprocesses receipt candidates through `postprocess_receipt` at `src/receipt_parser/pipeline.py:957`.

Why this is brittle:

There are now two places where receipt-specific output can be rewritten: the main candidate selector and the final result builder. The final builder is supposed to serialize and attach metadata, but it can mutate receipt fields again after validation. That can make warnings, pass history, and final output disagree.

Recommended cleanup:

- Keep receipt mutation inside `postprocess_receipt` or a single explicit receipt-finalization function.
- Make `_build_result` pure serialization plus metadata.
- If final cleanup is needed after schema construction, name it as a separate stage and validate after it.

### 5. Self-canceling field mutations

Evidence:

- `_fix_bowling_service_receipt` creates a synthetic service line item for bowling receipts at `src/receipt_parser/pipeline_receipt.py:1915`. It sets `line_items` to a single `Bowling` item and may set `payment_method` to `cash`.
- `postprocess_receipt` calls `_fix_bowling_service_receipt` early at `src/receipt_parser/pipeline_receipt.py:11517`.
- After validation, `_build_result` calls `fix_final_known_financial_overrides` at `src/receipt_parser/pipeline.py:541`.
- `fix_final_known_financial_overrides` then matches the City Bowl case and clears both fields at `src/receipt_parser/pipeline_receipt.py:11388`: `payment_method = None` and `line_items = []`.
- The unit test at `tests/test_unit.py:2116` asserts the helper-level behavior, but that test does not cover the later `_build_result` override that cancels the helper in the full pipeline path.

Why this is brittle:

The code contains two contradictory decisions for the same receipt: one helper says a bowling receipt should become a single service item, while a later last-mile override says this specific City Bowl receipt should have no line items and no payment method. Both can be locally "correct" in isolation, but the final behavior depends on hidden cross-stage ordering rather than an explicit convention.

Other instances of the same pattern:

| Pattern | Earlier mutation | Later mutation | Risk |
|---|---|---|---|
| Toll tax created, then cleared or replaced | `_fix_toll_inclusive_tax` sets inclusive tax and subtotal at `src/receipt_parser/pipeline_receipt.py:1889` | `fix_final_known_financial_overrides` clears NEXCO taxes at `src/receipt_parser/pipeline_receipt.py:11382`, then may later re-replace them through `_apply_target_101_182_layout_overrides` at `src/receipt_parser/pipeline_receipt.py:11445` | Tax convention for toll receipts depends on which token-gated branch runs last. |
| Service/item fallback created, then cleared | `_fix_line_items` can create fallback department/service items at `src/receipt_parser/pipeline_receipt.py:2362` and `src/receipt_parser/pipeline_receipt.py:2376` | The handwritten receipt guard in the same function can clear the single item at `src/receipt_parser/pipeline_receipt.py:2406` | The function both invents and removes single total-matching items; this should be one explicit decision point. |
| Non-product cleanup, then empty-item recovery | `_drop_non_product_line_items` removes extracted rows at `src/receipt_parser/pipeline_receipt.py:6596` and is called repeatedly from `postprocess_receipt` | `_recover_qty_unit_total_item_from_empty_extraction` can recreate a single item at `src/receipt_parser/pipeline_receipt.py:6398`, including immediately after a late drop at `src/receipt_parser/pipeline_receipt.py:11897` and `src/receipt_parser/pipeline_receipt.py:11898` | A row can be dropped as non-product and then reintroduced because the list is empty, without a phase-level invariant saying which result is desired. |
| Printed subtotal preservation, then unconditional recompute | The first universal subtotal block preserves a printed/extracted subtotal when it is close at `src/receipt_parser/pipeline_receipt.py:11816` | Later blocks unconditionally set `subtotal = total - tax_sum` at `src/receipt_parser/pipeline_receipt.py:11889` and `src/receipt_parser/pipeline_receipt.py:11929` | The earlier "preserve printed value" guard is effectively overridden later in the same function. |

Recommended cleanup:

- Introduce a finalization policy for each mutable field (`line_items`, `taxes`, `subtotal`, `payment_method`) so a later stage cannot silently cancel an earlier stage.
- Replace helper-level tests for these cases with full-path tests or add explicit tests for both helper behavior and final pipeline behavior.
- Add mutation tracing around `postprocess_receipt` and `_build_result`: for each stage, record field diffs for `line_items`, `taxes`, `subtotal`, `total`, `amount_paid`, `points_used`, and `payment_method`.
- Move `fix_final_known_financial_overrides` out of `_build_result`; serialization should not mutate receipt semantics after validation.

### 6. Oversized financial and item parsers

Evidence:

Largest functions in scope:

| Function | Start | Approx. lines |
|---|---:|---:|
| `_apply_target_101_182_layout_overrides` | `pipeline_receipt.py:10708` | 600 |
| `_extract_financial_totals_impl` | `pipeline_receipt.py:221` | 530 |
| `postprocess_receipt` | `pipeline_receipt.py:11496` | 440 |
| `_fix_maxvalu_suffix_marker_rows` | `pipeline_receipt.py:9813` | 397 |
| `_repair_column_split_items` | `pipeline_receipt.py:2718` | 351 |
| `_recover_missing_items_from_gap` | `pipeline_receipt.py:8671` | 314 |
| `assign_tax_categories` | `pipeline_receipt.py:1004` | 312 |

Why this is brittle:

Long stateful functions make it hard to tell whether a new condition is a general rule, a local exception, or an accidental interaction with earlier state. `_extract_financial_totals_impl` has many overlapping regex branches, lookahead windows, and context variables. That is understandable for OCR repair, but it needs a smaller internal model to avoid more patch layering.

Recommended cleanup:

- Split `_extract_financial_totals_impl` into independent extractors that return candidates with evidence and confidence, then choose candidates in one scoring function.
- Split item repair functions into reusable primitives: row grouping, amount parsing, description matching, subset-sum validation, and output projection.
- Add small tests for the primitives so future receipt fixes do not require editing one giant function.

## Dead or Unused-Looking Code

Static token search across `src` and `tests` found these private functions with only their definition as a reference:

| Function | Location | Note |
|---|---|---|
| `_fix_item_descriptions_from_ocr_price_rows` | `src/receipt_parser/pipeline_receipt.py:5814` | Similar to nearby called description repair helpers, but currently not called. |
| `_fix_code_row_descriptions_from_ocr` | `src/receipt_parser/pipeline_receipt.py:5880` | Similar to `_fix_qty_code_row_descriptions_from_ocr`, which is called. |
| `_fix_explicit_tax_amounts_from_ocr` | `src/receipt_parser/pipeline_receipt.py:9101` | Appears superseded by printed-tax helpers around `src/receipt_parser/pipeline_receipt.py:9186` and later calls. |
| `_fix_location_from_ocr_context` | `src/receipt_parser/pipeline_receipt.py:11448` | Generic location repair exists but is not called; a narrower yakitori-specific location helper is called instead. |

`ruff check --select F401,F841,F821` also reported:

| Finding | Location |
|---|---|
| Unused import `numpy as np` | `src/receipt_parser/pipeline.py:16` |
| Unused import `strip_banner_lines` | `src/receipt_parser/pipeline.py:26` |
| Unused imports `ERA_TABLE`, `era_to_western_year`, `should_override_field` | `src/receipt_parser/pipeline.py:32` |
| Unused local `items_sum_already_matches` | `src/receipt_parser/pipeline_receipt.py:2602` |
| Unused local `zone_last_item` | `src/receipt_parser/pipeline_receipt.py:2783` |
| Unused local `reduced_base` | `src/receipt_parser/pipeline_receipt.py:5379` |
| Unused local `tax_sum` | `src/receipt_parser/pipeline_receipt.py:8684` |
| Unused local `items_fixed` | `src/receipt_parser/pipeline_receipt.py:11764` |

Recommended cleanup:

- Delete unused imports and unused locals after a unit/validation test run.
- For the four unused private helpers, either wire them into an intentional phase with tests or remove them. Do not leave dormant repair functions in this file, because future agents may assume they are active behavior.

## Repetitive Code Patterns

### Repeated local normalization helpers

Evidence:

`pipeline_receipt.py` repeatedly defines local helpers named `_norm`, `_clean_desc`, `_valid_desc`, `_similar`, `_parse_amount`, and `_row`. Examples appear at `src/receipt_parser/pipeline_receipt.py:5371`, `src/receipt_parser/pipeline_receipt.py:5688`, `src/receipt_parser/pipeline_receipt.py:5725`, `src/receipt_parser/pipeline_receipt.py:5821`, `src/receipt_parser/pipeline_receipt.py:5887`, `src/receipt_parser/pipeline_receipt.py:6333`, `src/receipt_parser/pipeline_receipt.py:6479`, `src/receipt_parser/pipeline_receipt.py:6766`, `src/receipt_parser/pipeline_receipt.py:9821`, and `src/receipt_parser/pipeline_receipt.py:10218`.

Why this matters:

Many of these helpers are almost the same, but small differences in regex cleanup or similarity thresholds can create inconsistent behavior. When a future fix needs to "normalize descriptions", it is unclear which local definition encodes the desired convention.

Recommended cleanup:

- Add shared helpers for description normalization, OCR amount parsing, row construction, and similarity scoring.
- Keep local wrappers only when a parser genuinely needs a different normalization policy, and name that policy.

### Repeated row reconstruction shape

Evidence:

Several functions build item dicts with the same fields: `description`, `qty`, `unit_price`, `total`, `tax_category`, `discount`, and `discount_rate`. There is `_known_item` at `src/receipt_parser/pipeline_receipt.py:10487`, local `_row` at `src/receipt_parser/pipeline_receipt.py:6766`, local `_row` at `src/receipt_parser/pipeline_receipt.py:9821`, and local `_row` at `src/receipt_parser/pipeline_receipt.py:10218`.

Why this matters:

Repeated constructors invite subtle field differences and encourage full-answer reconstruction. A single constructor is fine, but it should be used by generic parsers, not as an answer-key convenience.

Recommended cleanup:

- Create one `make_line_item` helper if this shape needs to be standardized.
- Avoid using the helper in production code with literal full receipt item lists.

## Suggested Cleanup Plan

1. Add static guardrails first.
   - Enable `ruff` for at least `F401` and `F841`.
   - Add a custom test that rejects new production functions with fixture-range naming or known-answer override patterns.

2. Make `postprocess_receipt` phase-driven.
   - Convert the repeated helper list into named phases.
   - Run each helper once unless a comment and test prove it must repeat.
   - Add an idempotence regression test.

3. Quarantine known-answer overrides.
   - Move full item-list and total override blocks behind a benchmark-only compatibility flag, or remove them as generic parsers replace them.
   - Start with `_apply_target_101_182_layout_overrides`, because it is the largest and clearest non-general block.

4. Collapse duplicated primitives.
   - Extract shared description normalization, amount parsing, row building, and similarity helpers.
   - Replace local copies incrementally, with unit tests around each primitive.

5. Split extraction from selection.
   - Financial extraction should gather candidate totals/taxes with evidence.
   - A scorer should choose among candidates, instead of mutating a single `result` dict through a long sequence of regex branches.

## Verification Performed

- Inspected current source with `rg`, targeted line reads, and `ruff check --select F401,F841,F821`.
- Counted repeated helper invocations inside `postprocess_receipt`.
- Searched `src` and `tests` for private helper call sites.

No tests or benchmarks were run because this task requested documentation only and no pipeline behavior was changed.
