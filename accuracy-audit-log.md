# Receipt Accuracy Audit Log

Last updated: 2026-06-15

## Rules

- Production parser fixes must be general-purpose.
- Do not add receipt-specific, merchant-specific, known-date, known-total, known-product-list, fixture-ID, or fixture-range logic.
- Do not preserve benchmark pass rate by embedding answers.
- Do not modify `*_truth.json` files without explicit user approval.

## Current Snapshot

- Full cached accuracy: 100.0% field checks; 312/312 receipt cases passed, plus the production guardrail passed.
- Fast unit/validation/guardrail gate: 329 passed.
- Guardrail-only check: 1 passed.
- Receipts requiring user review: none.
- Receipts blocked pending user help: none.
- Guardrail allowlist status: empty; `tests/test_pipeline_guardrails.py` passes with no known production violations allowlisted.
- Targeted `receipt_172` benchmark after removing the Donki/product-name category override: 100.0% (220/220), robust, deterministic across 10 runs.
- Targeted `receipt_167` benchmark after stacked pre-price description repair: 100.0% (220/220), robust, deterministic across 10 runs.
- Targeted `receipt_90`/`receipt_130` metadata regression benchmark after tightening the stacked-name window: 100.0% (440/440), 2/2 robust, deterministic across 10 runs.
- Final affected-receipt benchmark (`receipt_172`, `receipt_167`, `receipt_90`, `receipt_130`): 100.0% (880/880), 4/4 robust, deterministic across 10 runs.
- Targeted failed-full-suite benchmark (`receipt_151`, `receipt_167`, `receipt_171`, `receipt_23`, `receipt_82`, `receipt_98`): 96.8% (1278/1320), 2/6 robust, 4/6 fragile across 10 runs.
- Targeted `receipt_82` header-branch location benchmark: 100.0% (220/220), robust, deterministic across 10 runs.
- Targeted `receipt_98` dotted-thousands price-alignment benchmark: 100.0% (220/220), robust, deterministic across 10 runs.
- Targeted `receipt_151` barcode unit-qty amount-stack benchmark: 100.0% (220/220), robust, deterministic across 10 runs.
- Targeted location regression slice (`receipt_17`, `receipt_18`, `receipt_19`, `receipt_48`, `receipt_81`, `receipt_90`, `receipt_123`, `receipt_130`, `receipt_147`, `receipt_82`): 100.0% (2200/2200), 10/10 robust, deterministic across 10 runs.
- Targeted `receipt_165` location control slice (`receipt_165`, `receipt_82`, `receipt_17`, `receipt_147`): 100.0% (880/880), 4/4 robust, deterministic across 10 runs.
- Targeted final mixed grocery/tax benchmark (`receipt_8`, `receipt_50`, `receipt_68`, `receipt_84`, `receipt_86`, `receipt_92`, `receipt_95`, `receipt_97`, `receipt_126`, `receipt_133`, `receipt_138`, `receipt_155`, `receipt_163`, `receipt_168`): 100.0% (3080/3080), 14/14 robust, deterministic across 10 runs.
- Targeted `receipt_171` benchmark after labeled date and discounted-row reconciliation fixes: 100.0% (220/220), robust, deterministic across 10 runs.
- Targeted `receipt_98` basket-marker benchmark after final Costco-style regression fix: 100.0% (220/220), robust, deterministic across 10 runs.
- Final affected-receipt benchmark (`receipt_98`, `receipt_171`) had one classified transient OCR/API failure for `receipt_171` (`Cloud Vision API error: Internal error encountered`, OCR confidence 0.00); `receipt_98` was robust and `receipt_171` reran at 100.0% (220/220), robust, deterministic across 10 runs.
- Targeted `receipt_160` benchmark after removing the hardcoded full-list dense grocery branch: 100.0% (220/220), robust, deterministic across 10 runs.
- Targeted `receipt_147` benchmark after store-in-store header location cleanup: 100.0% (220/220), robust, deterministic across 10 runs.
- Targeted `receipt_42` benchmark after a classified transient full-suite qty/unit-price failure: 100.0% (220/220), robust, deterministic across 10 runs. Targeted accuracy for `receipt_42` plus `receipt_42_v1` also passed.
- Targeted `receipt_97` benchmark after external-tax total repair: 100.0% (220/220), robust, deterministic across 10 runs.
- Targeted name/bag amount-shift benchmark (`receipt_155`, `receipt_168`): 100.0% (440/440), 2/2 robust, deterministic across 10 runs.
- Targeted `receipt_173` benchmark: 100.0% (220/220), robust, deterministic across 10 runs.
- Targeted split item/body-total benchmark (`receipt_69`, `receipt_82`, `receipt_164`, `receipt_170`): 100.0% (880/880), 4/4 robust, deterministic across 10 runs.
- Targeted split address location benchmark (`receipt_94`): 100.0% (220/220), robust, deterministic across 10 runs.
- Targeted structural bag/dense grocery slice benchmark (`receipt_8`, `receipt_50`, `receipt_68`, `receipt_84`, `receipt_86`, `receipt_92`, `receipt_95`, `receipt_97`, `receipt_126`, `receipt_133`, `receipt_138`, `receipt_155`, `receipt_163`, `receipt_168`): 100.0% (3080/3080), 14/14 robust, deterministic across 10 runs.
- Targeted structural qty-context benchmark (`receipt_10`, `receipt_150`, `receipt_159`): 100.0% (660/660), 3/3 robust, deterministic across 10 runs.
- Targeted `receipt_157` unit-price invariant benchmark: 100.0% (220/220), robust, deterministic across 10 runs.
- Targeted printed inclusive-tax structural benchmark (`receipt_8`, `receipt_19`, `receipt_50`, `receipt_86`, `receipt_90`, `receipt_92`, `receipt_95`, `receipt_126`, `receipt_130`, `receipt_133`, `receipt_159`): 100.0% (2420/2420), 11/11 robust, deterministic across 10 runs.
- Targeted JAN/POS shifted-discount benchmark (`receipt_4`, `receipt_73`, `receipt_162`): 100.0% (660/660), 3/3 robust, deterministic across 10 runs.
- Historical full cached accuracy after printed-tax structural cleanup: blocked at that time by `receipt_171` and `receipt_171_v1` date mismatches; this blocker was resolved later by the user truth update plus structural parser fixes.
- Former `receipt_172` tax-category blocker is resolved: the user corrected the truth convention, `_fix_donki_discount_shop_categories` was removed from production code, and the structural qty/rate-base parser now covers the receipt without product-name logic.
- Truth contents modified by Codex: none.
- User-approved duplicate deletion completed for `receipt_134`; active fixture/cache files are absent, and the stale local removed-cache copy was deleted. Historical benchmark/audit references were left as run history.

## Latest Work Completed

- Removed `_fix_donki_discount_shop_categories`, the final product-name category override identified in the guardrail allowlist.
  - The replacement behavior is structural: qty-detail ownership and printed rate-base subset arithmetic resolve duplicate totals and tax-category assignment without merchant, fixture, or product-name gates.
  - `tests/test_pipeline_guardrails.py` now passes with an empty `KNOWN_VIOLATIONS` set.
  - Targeted `receipt_172` benchmark passed at 100.0% (220/220), robust and deterministic across 10 runs.
- Fixed the nondeterministic `receipt_167` description-stack failure structurally:
  - Added `_repair_pre_price_stack_descriptions_from_ocr`, which only repairs duplicate/nested parsed descriptions when OCR shows a contiguous product-name block immediately before a stacked price block and item sums match the receipt arithmetic.
  - Tightened the helper after the first regression set so transaction/register/promo/footer text cannot be used as product names.
  - Added unit coverage for restoring names before stacked prices and for ignoring metadata before stacked prices.
  - Targeted `receipt_167`, `receipt_90`, and `receipt_130` benchmarks passed at 100.0% across 10 runs after the fix.
- Replaced `_fix_printed_tax_amounts_for_known_layouts` with `_fix_printed_tax_amounts_from_structural_blocks`.
  - Removed merchant/store token gates for the printed-tax correction paths.
  - The helper now fires on explicit OCR tax-summary structures: parenthesized per-rate inclusive tax blocks, `N%対象消費税` blocks, and single inclusive `内、消費税` amount lines.
  - Added unit coverage using merchant-free OCR snippets for all three structures.
  - Removed the stale known-layout helper entry from the guardrail allowlist.
- Removed `_fix_gyomu_super_jan_discount_layout`, the merchant/product-specific JAN discount patch.
  - The shifted discount case is now covered by the existing structural `_replace_jan_pos_items_when_balanced` parser.
  - Added merchant-free unit coverage for repeated JAN item rows, quantity rows, visible discount markers, and printed subtotal balance.
  - Removed the stale Gyomu helper entry from the guardrail allowlist.
- Confirmed the old unused-code audit slice is stale for the currently inspected imports/symbols: `pipeline.py` imports checked here are used, and the receipt-side era/override symbols are live. `ruff` is not installed in the `financial-aid` environment, so this was verified by source search rather than a `ruff` run.
- Resolved the former Donki/product-list category blocker after the user corrected the `receipt_172` truth convention:
  - Removed the exact product-name category helper from production code.
  - Replaced it with structural qty-detail ownership and printed rate-base subset arithmetic.
  - Verified `receipt_172` at 100.0% across 10 benchmark runs and in the full cached accuracy suite.
- Replaced `_fix_daiso_qty_and_reduced_rate_context` with `_fix_qty_context_and_reduced_rate_from_ocr`, a structural qty/reduced-rate helper:
  - Removed the Daiso/merchant token gate and now requires visible quantity-detail OCR (`@... xN` / `xN個`) and/or printed reduced-rate tax-summary evidence.
  - Limited reduced-rate context consumption to standalone price/marker rows so a following product's inline price row is not stolen.
  - Removed the stale Daiso helper allowlist entry from the guardrail.
- Classified the first full fast-suite regression for the qty-context cluster as deterministic pipeline fallout: a following inline price row was being consumed as context. Fixed it by requiring context price-line structure, then re-ran the fast gate successfully.
- Classified the first full cached-accuracy regression for `receipt_157_v1` as deterministic late-reconstruction fallout: item totals were correct but one single-quantity unit price was left as zero after final layout repair.
- Fixed the `receipt_157_v1` fallout with the existing general invariant at the final consistency boundary: for single-quantity, undiscounted rows, a missing/zero unit price is filled from the positive line total.
- Replaced `_fix_maxvalu_water_bag_split` with `_fix_name_bag_amount_shift_from_ocr`, a structural parser for OCR sequences shaped as product-name row, paid-bag row with small inline price, then marked standalone product amount.
  - Removed the merchant/product gate and known product/price rewrite.
  - Requires visible per-rate taxable bases and printed subtotal arithmetic before mutating rows.
  - Added unit coverage for both shifted-row repair and the already-balanced row case where single-quantity unit prices were missing.
  - Removed the stale MaxValu water-bag entry from the guardrail allowlist.
- Classified the first targeted benchmark regression for this cluster as deterministic pipeline replacement fallout: `receipt_168` kept correct item descriptions/totals/taxes but missed two single-quantity unit prices. Fixed it generally by filling missing unit prices from totals only when the structural OCR sequence and printed subtotal balance are present.
- Deleted the stale local removed-cache copy for approved duplicate `receipt_134`; `receipt_93` is retained as the canonical fixture.
- Confirmed the user-fixed VivaHome receipts (`receipt_6`, `receipt_7`, `receipt_9`, `receipt_24`, `receipt_25`, `receipt_28`) and `receipt_165` location in the full cached accuracy sweep.
- Replaced two merchant-named tiny-bag helpers with structural OCR parsers:
  - `_fix_split_bag_price_from_nearby_single_digit` repairs tiny bag totals from nearby single-digit OCR prices and visible bag-row evidence.
  - `_fix_small_bag_description_from_ocr_entry` recovers an unlabeled low-value item from a visible bag OCR entry.
  - Added unit coverage for both helpers without merchant gates and removed their stale guardrail allowlist entries.
- Classified a `receipt_97` target benchmark failure as nondeterministic LLM-variance fallout plus a pipeline robustness gap: the dense campaign-discount parser rejected valid OCR rows because it trusted a bad mutable subtotal.
- Fixed the dense campaign-discount parser structurally by using the printed `小計` amount as the balance target when visible, then requiring row-sum and tax-summary invariants.
- Added a unit test proving campaign-discount reconstruction uses a visible printed subtotal when the mutable extracted subtotal has drifted.
- Fixed `receipt_173` structurally: marker-bearing standalone price rows such as `510*` can now feed the following discount-line repair, and the final receipt repair phase reapplies balanced discount repairs after layout reconstruction.
- Added a unit test covering marker price rows followed by discount rows.
- Replaced the brittle Starbucks/body-total allowlist cluster with a structural split item/body-total parser:
  - Triggered by `本体合計`, count-prefixed item rows, nearby price rows, printed tax target/tax amount pairs, and subtotal/tax/total arithmetic.
  - Recovers delayed body-price items, external tax labels, and rate categories through rate-base arithmetic rather than merchant/product/known-total checks.
  - Added a unit test proving the layout recovers without a merchant gate.
  - Wired the structural repair into the final receipt output repair path after generic split-price repair so late mutations cannot leave the intermediate bad shape.
- Removed the stale Starbucks/body-total known-violation entries from the guardrail allowlist.
- Replaced the yakitori-named location repair with a generic split-address parser:
  - Triggered by an address prefix line ending in an administrative unit followed by a street-number line.
  - Removes the merchant/type token gate and keeps contact, registration, and summary lines excluded.
  - Added a unit test proving split-address recovery without a merchant gate.
  - Removed the stale yakitori helper entry from the guardrail allowlist.
- No brittle production branch was added.
- Replaced two MaxValu-named tax-category helpers with structural helpers:
  - `_rebalance_standard_categories_from_reduced_rate_markers` uses reduced-tax OCR markers plus printed 8%/10% rate-base subset arithmetic.
  - `_fix_nonfood_packaging_tax_categories` applies standard tax to visible non-food packaging rows when a printed 10% rate base exists.
  - Removed their stale merchant-name guardrail allowlist entries.
- Added `_restore_external_tax_total_from_printed_subtotal`, a structural finalization guard that restores `total` and `amount_paid` from printed subtotal + external taxes when the matching total/payment amount is visible, while ignoring loyalty footer totals.
- Classified the first full-slice `receipt_97` regression as nondeterministic postprocess ordering fallout: final repairs could still replace the true total with a footer amount after an earlier repair. Fixed it by rerunning the external-tax invariant as the final receipt repair step.
- Fixed the deterministic `receipt_82` location regression structurally:
  - The existing header-branch recovery now allows a visible short `...店` header token to replace a broad administrative fragment such as a city/ward string.
  - Specific addresses and already-selected store names remain protected.
  - Added unit coverage for broad-fragment replacement and address preservation.
- Classified the first full-accuracy location regression set as deterministic pipeline fallout from the `receipt_82` location rule being too broad.
- Narrowed header-branch admin-fragment replacement so it only fires when the short branch stem structurally extends the final admin root (`八幡区` -> `八幡平野店`) or uses a short store prefix plus that root (`宗像市` -> `サンリブ宗像店`).
  - Added regression coverage for district-only branches preserving city/ward conventions and short store-prefixed branches recovering the visible branch.
  - Verified the affected location slice and `receipt_165` controls with 10-run benchmarks.
- Fixed the nondeterministic `receipt_98` item-price regression structurally:
  - The following-OCR price repair now accepts dot thousands separators such as `1.118 E` as repeated amount evidence.
  - The repair still requires code/quantity-like context, repeated OCR amount support, and item-sum arithmetic improvement.
  - Added unit coverage for dotted-thousands barcode price evidence.
- Fixed the nondeterministic `receipt_151` one-item collapse structurally:
  - Added a barcode/unit-quantity/amount-stack projection for receipts printed as `description / barcode / ¥unit qty個`, followed by stacked item totals before `小計`.
  - Requires barcode evidence, row-count improvement or item-sum improvement, and subtotal or total-minus-tax arithmetic balance.
  - Added unit and final-output-repair coverage, including the O-ring description repair that follows from JAN/barcode evidence.
- Fixed `receipt_171` / `receipt_171_v1` structurally after the user corrected the truth date:
  - `_fix_date` now supports labeled transaction dates printed on the following OCR line and coerces OCR-mangled early-2000s years into the existing modern receipt year window.
  - Discounted gross price repair now preserves quantity/unit arithmetic when OCR prints a gross line total such as `316 A` with `2個 X 単158`.
  - Late output reconciliation now repairs discounted item descriptions from visible OCR owner lines and drops exact duplicate rows only when OCR occurrence count and printed subtotal arithmetic prove the duplicate.
  - Added unit coverage for the labeled next-line date, discounted gross line arithmetic, discounted OCR owner repair, duplicate-drop subtotal invariant, and campaign-marker suffix cleanup.
  - No receipt-specific, merchant-specific, known-date, known-total, or known-product-list branch was added.
- Fixed the deterministic full-suite `receipt_98` / `receipt_98_v1` regression structurally:
  - Added `_replace_basket_marker_rows_when_balanced`, which reconstructs explicit basket-marker streams from bottom-of-basket evidence, printed item count, marked `E`/`T` amount rows, coupon markers, and per-rate base arithmetic.
  - The parser is not merchant-gated; it requires the reconstructed row count, row sum, and rate sums to match printed OCR invariants before replacing line items.
  - Wired the repair into the final receipt output repair path so later generic cleanup cannot leave stacked basket rows in a wrong-but-balanced shape.
  - Added merchant-free unit coverage for stacked names, bottom-of-basket markers, `E`/`T` tax categories, and coupon subtraction.
  - No receipt-specific, merchant-specific, known-total, or known-product-list branch was added.
- Removed `_fix_maxvalu_suffix_marker_rows`, including the hardcoded full product-list branch for dense grocery output.
  - The remaining behavior is now `_drop_numeric_marker_description_rows`, which only drops parsed rows whose description is a numeric price/marker token.
  - Existing dense sequence parsing covers `receipt_160` structurally through queued names/prices and arithmetic balance.
  - Added merchant-free unit coverage for numeric marker description cleanup.
  - Removed the stale MaxValu suffix and hardcoded `line_items[12]` entries from the guardrail allowlist.
- Fixed the `receipt_147` full-suite location regression structurally:
  - Added store-in-store header cleanup for OCR shaped as `ASCII brand + Japanese host-store branch`, trimming a model-returned whole header/host store back to a city derived from structural geographic evidence.
  - Added unit coverage for the mixed brand/host-store header shape.
  - Classified the follow-up `receipt_42` / `receipt_42_v1` full-suite qty/unit-price failure as transient/nondeterministic extraction variance after `receipt_42` benchmarked at 100.0% across 10 runs and targeted accuracy passed for both image and OCR variant.

## Remaining Cleanup Notes

- Full cached accuracy is currently 100%, and there are no receipts blocked pending user review.
- The production guardrail allowlist is empty. New parser work should keep it that way.
- The Starbucks/body-total, yakitori location, NAFCO split-bag, Cosmos small-bag, MaxValu water-bag, MaxValu suffix marker, Daiso qty-context, printed-tax known-layout, hardcoded item-index, and Gyomu JAN discount entries were removed from the allowlist.
- Remaining cleanup is architectural rather than receipt-blocked: continue simplifying phase order and duplicated primitives while keeping replacements structural and invariant-checked.

## Receipt Status Matrix - 2026-06-14

| Receipt | Status |
|---|---|
| receipt_1 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_1, receipt_1_v1. |
| receipt_2 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_2, receipt_2_v1. |
| receipt_3 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_3, receipt_3_v1. |
| receipt_4 | PASS - Full accuracy passed for image + 2 OCR variant(s): receipt_4, receipt_4_v1, receipt_4_v2. |
| receipt_5 | PASS - Full accuracy passed for image fixture. |
| receipt_6 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_6, receipt_6_v1. |
| receipt_7 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_7, receipt_7_v1. |
| receipt_8 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_8, receipt_8_v1. |
| receipt_9 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_9, receipt_9_v1. |
| receipt_10 | PASS - Full accuracy passed for image + 2 OCR variant(s): receipt_10, receipt_10_v1, receipt_10_v2. |
| receipt_11 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_11, receipt_11_v1. |
| receipt_12 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_12, receipt_12_v1. |
| receipt_13 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_13, receipt_13_v1. |
| receipt_14 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_14, receipt_14_v1. |
| receipt_15 | PASS - Full accuracy passed for image fixture. |
| receipt_16 | PASS - Full accuracy passed for image fixture. |
| receipt_17 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_17, receipt_17_v1. |
| receipt_18 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_18, receipt_18_v1. |
| receipt_19 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_19, receipt_19_v1. |
| receipt_20 | PASS - Full accuracy passed for image fixture. |
| receipt_21 | PASS - Full accuracy passed for image fixture. |
| receipt_22 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_22, receipt_22_v1. |
| receipt_23 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_23, receipt_23_v1. |
| receipt_24 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_24, receipt_24_v1. |
| receipt_25 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_25, receipt_25_v1. |
| receipt_26 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_26, receipt_26_v1. |
| receipt_27 | PASS - Full accuracy passed for image fixture. |
| receipt_28 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_28, receipt_28_v1. |
| receipt_29 | PASS - Full accuracy passed for image + 2 OCR variant(s): receipt_29, receipt_29_v1, receipt_29_v2. |
| receipt_30 | PASS - Full accuracy passed for image fixture. |
| receipt_31 | PASS - Full accuracy passed for image fixture. |
| receipt_32 | PASS - Full accuracy passed for image fixture. |
| receipt_33 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_33, receipt_33_v1. |
| receipt_34 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_34, receipt_34_v1. |
| receipt_35 | PASS - Full accuracy passed for image fixture. |
| receipt_36 | PASS - Full accuracy passed for image fixture. |
| receipt_37 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_37, receipt_37_v1. |
| receipt_38 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_38, receipt_38_v1. |
| receipt_39 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_39, receipt_39_v1. |
| receipt_40 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_40, receipt_40_v1. |
| receipt_41 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_41, receipt_41_v1. |
| receipt_42 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_42, receipt_42_v1. |
| receipt_43 | PASS - Full accuracy passed for image fixture. |
| receipt_44 | PASS - Full accuracy passed for image + 2 OCR variant(s): receipt_44, receipt_44_v1, receipt_44_v2. |
| receipt_45 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_45, receipt_45_v1. |
| receipt_46 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_46, receipt_46_v1. |
| receipt_47 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_47, receipt_47_v1. |
| receipt_48 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_48, receipt_48_v1. |
| receipt_49 | REMOVED - Duplicate excluded; receipt_131 retained per user decision. |
| receipt_50 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_50, receipt_50_v1. |
| receipt_51 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_51, receipt_51_v1. |
| receipt_52 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_52, receipt_52_v1. |
| receipt_53 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_53, receipt_53_v1. |
| receipt_54 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_54, receipt_54_v1. |
| receipt_55 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_55, receipt_55_v1. |
| receipt_56 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_56, receipt_56_v1. |
| receipt_57 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_57, receipt_57_v1. |
| receipt_58 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_58, receipt_58_v1. |
| receipt_59 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_59, receipt_59_v1. |
| receipt_60 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_60, receipt_60_v1. |
| receipt_61 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_61, receipt_61_v1. |
| receipt_62 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_62, receipt_62_v1. |
| receipt_63 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_63, receipt_63_v1. |
| receipt_64 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_64, receipt_64_v1. |
| receipt_65 | PASS - Full accuracy passed for image + 2 OCR variant(s): receipt_65, receipt_65_v1, receipt_65_v2. |
| receipt_66 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_66, receipt_66_v1. |
| receipt_67 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_67, receipt_67_v1. |
| receipt_68 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_68, receipt_68_v1. |
| receipt_69 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_69, receipt_69_v1. |
| receipt_70 | PASS - Full accuracy passed for image fixture. |
| receipt_71 | PASS - Full accuracy passed for image fixture. |
| receipt_72 | PASS - Full accuracy passed for image + 2 OCR variant(s): receipt_72, receipt_72_v1, receipt_72_v2. |
| receipt_73 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_73, receipt_73_v1. |
| receipt_74 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_74, receipt_74_v1. |
| receipt_75 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_75, receipt_75_v1. |
| receipt_76 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_76, receipt_76_v1. |
| receipt_77 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_77, receipt_77_v1. |
| receipt_78 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_78, receipt_78_v1. |
| receipt_79 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_79, receipt_79_v1. |
| receipt_80 | PASS - Full accuracy passed for image + 2 OCR variant(s): receipt_80, receipt_80_v1, receipt_80_v2. |
| receipt_81 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_81, receipt_81_v1. |
| receipt_82 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_82, receipt_82_v1. |
| receipt_83 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_83, receipt_83_v1. |
| receipt_84 | PASS - Full accuracy passed for image + 2 OCR variant(s): receipt_84, receipt_84_v1, receipt_84_v2. |
| receipt_85 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_85, receipt_85_v1. |
| receipt_86 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_86, receipt_86_v1. |
| receipt_87 | PASS - Full accuracy passed for image fixture. |
| receipt_88 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_88, receipt_88_v1. |
| receipt_89 | PASS - Full accuracy passed for image + 2 OCR variant(s): receipt_89, receipt_89_v1, receipt_89_v2. |
| receipt_90 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_90, receipt_90_v1. |
| receipt_91 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_91, receipt_91_v1. |
| receipt_92 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_92, receipt_92_v1. |
| receipt_93 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_93, receipt_93_v1. |
| receipt_94 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_94, receipt_94_v1. |
| receipt_95 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_95, receipt_95_v1. |
| receipt_96 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_96, receipt_96_v1. |
| receipt_97 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_97, receipt_97_v1. |
| receipt_98 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_98, receipt_98_v1. |
| receipt_99 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_99, receipt_99_v1. |
| receipt_100 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_100, receipt_100_v1. |
| receipt_101 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_101, receipt_101_v1. |
| receipt_102 | PASS - Full accuracy passed for image fixture. |
| receipt_103 | PASS - Full accuracy passed for image fixture. |
| receipt_104 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_104, receipt_104_v1. |
| receipt_105 | PASS - Full accuracy passed for image fixture. |
| receipt_106 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_106, receipt_106_v1. |
| receipt_107 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_107, receipt_107_v1. |
| receipt_108 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_108, receipt_108_v1. |
| receipt_109 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_109, receipt_109_v1. |
| receipt_110 | PASS - Full accuracy passed for image fixture. |
| receipt_111 | PASS - Full accuracy passed for image fixture. |
| receipt_112 | PASS - Full accuracy passed for image fixture. |
| receipt_113 | PASS - Full accuracy passed for image fixture. |
| receipt_114 | PASS - Full accuracy passed for image fixture. |
| receipt_115 | REMOVED - Deleted after user confirmed receipt_39 is correct. |
| receipt_116 | REMOVED - Duplicate removed; receipt_72 retained per user decision. |
| receipt_117 | PASS - Full accuracy passed for image fixture. |
| receipt_118 | PASS - Full accuracy passed for image fixture. |
| receipt_119 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_119, receipt_119_v1. |
| receipt_120 | PASS - Full accuracy passed for image fixture. |
| receipt_121 | PASS - Full accuracy passed for image fixture. |
| receipt_122 | PASS - Full accuracy passed for image fixture. |
| receipt_123 | PASS - Full accuracy passed for image fixture. |
| receipt_124 | REMOVED - Duplicate removed; receipt_82 retained per user decision. |
| receipt_125 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_125, receipt_125_v1. |
| receipt_126 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_126, receipt_126_v1. |
| receipt_127 | PASS - Full accuracy passed for image fixture. |
| receipt_128 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_128, receipt_128_v1. |
| receipt_129 | PASS - Full accuracy passed for image + 2 OCR variant(s): receipt_129, receipt_129_v1, receipt_129_v2. |
| receipt_130 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_130, receipt_130_v1. |
| receipt_131 | PASS - Full accuracy passed for image + 2 OCR variant(s): receipt_131, receipt_131_v1, receipt_131_v2. |
| receipt_132 | PASS - Full accuracy passed for image fixture. |
| receipt_133 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_133, receipt_133_v1. |
| receipt_134 | REMOVED - Duplicate removed; receipt_93 retained per user decision. |
| receipt_135 | REMOVED - Duplicate removed; receipt_94 retained per user decision. |
| receipt_136 | REMOVED - Duplicate removed; receipt_95 retained per user decision. |
| receipt_137 | REMOVED - Duplicate removed; receipt_96 retained per user decision. |
| receipt_138 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_138, receipt_138_v1. |
| receipt_139 | REMOVED - Duplicate removed; receipt_98 retained per user decision. |
| receipt_140 | PASS - Full accuracy passed for image fixture. |
| receipt_141 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_141, receipt_141_v1. |
| receipt_142 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_142, receipt_142_v1. |
| receipt_143 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_143, receipt_143_v1. |
| receipt_144 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_144, receipt_144_v1. |
| receipt_145 | PASS - Full accuracy passed for image + 2 OCR variant(s): receipt_145, receipt_145_v1, receipt_145_v2. |
| receipt_146 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_146, receipt_146_v1. |
| receipt_147 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_147, receipt_147_v1. |
| receipt_148 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_148, receipt_148_v1. |
| receipt_149 | PASS - Full accuracy passed for image fixture. |
| receipt_150 | PASS - Full accuracy passed for image + 2 OCR variant(s): receipt_150, receipt_150_v1, receipt_150_v2. |
| receipt_151 | PASS - Full accuracy passed for image + 2 OCR variant(s): receipt_151, receipt_151_v1, receipt_151_v2. |
| receipt_152 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_152, receipt_152_v1. |
| receipt_153 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_153, receipt_153_v1. |
| receipt_154 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_154, receipt_154_v1. |
| receipt_155 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_155, receipt_155_v1. |
| receipt_156 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_156, receipt_156_v1. |
| receipt_157 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_157, receipt_157_v1. |
| receipt_158 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_158, receipt_158_v1. |
| receipt_159 | PASS - Full accuracy passed for image + 3 OCR variant(s): receipt_159, receipt_159_v1, receipt_159_v2, receipt_159_v3. |
| receipt_160 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_160, receipt_160_v1. |
| receipt_161 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_161, receipt_161_v1. |
| receipt_162 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_162, receipt_162_v1. |
| receipt_163 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_163, receipt_163_v1. |
| receipt_164 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_164, receipt_164_v1. |
| receipt_165 | PASS - Full accuracy passed for image fixture. |
| receipt_166 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_166, receipt_166_v1. |
| receipt_167 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_167, receipt_167_v1. |
| receipt_168 | PASS - Full accuracy passed for image + 2 OCR variant(s): receipt_168, receipt_168_v1, receipt_168_v2. |
| receipt_169 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_169, receipt_169_v1. |
| receipt_170 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_170, receipt_170_v1. |
| receipt_171 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_171, receipt_171_v1. Targeted rerun after transient OCR API failure passed at 100.0% (220/220), robust across 10 runs. |
| receipt_172 | PASS - Full accuracy passed for image + 1 OCR variant(s): receipt_172, receipt_172_v1. Former tax-category cleanup blocker resolved; product-name override removed and structural parser verified. |
| receipt_173 | PASS - Full accuracy passed for image + 2 OCR variant(s): receipt_173, receipt_173_v1, receipt_173_v2. |
