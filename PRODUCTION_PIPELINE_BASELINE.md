# Production Pipeline Baseline

Baseline commit: `c175c17`

Date: 2026-06-15

## Current Parser Version

The production parser baseline is the receipt parser code at commit `c175c17`.
Future parser changes must preserve the guardrail suite and full cached
accuracy unless the user explicitly approves a new baseline.

## Baseline Evidence

- Fast gate: `tests/test_unit.py`, `tests/test_validation.py`, and
  `tests/test_pipeline_guardrails.py` passed from `c175c17`.
- Full cached accuracy: `tests/test_accuracy.py` passed 312/312 collected
  cases with JSON report saved at `local/baseline/c175c17/accuracy.json`.
- Benchmark CI evidence: `tests/benchmark.py --ci --workers 4` passed with
  3541/3541 checks, 164/164 robust fixtures, determinism rate 1.0, and artifact
  saved at `local/baseline/c175c17/benchmark_ci_latest.json`.
- Truth-file status before parser hardening: no dirty `*_truth.json` files.

## Known Architectural Risks

- `postprocess_receipt` is still a large ordered repair stack with repeated
  mutator calls. Guardrails now freeze the current call count and require any
  repeated mutator debt to be explicit.
- `_apply_final_receipt_output_repairs` is still a late semantic repair path
  after model serialization. Guardrails now require every late repair to be an
  explicit traced stage.
- Several shared parsing helpers remain oversized. Future cleanup should split
  them into structural OCR classifiers, amount parsers, row projection helpers,
  tax-summary extractors, subset-sum validators, and candidate scorers.
- Production parser code must not add fixture, merchant, product-list, date,
  known-total, location-answer, or receipt-ID logic.

## Baseline Rule

Any future parser behavior change must preserve:

- no dirty truth files unless the user explicitly approves truth edits;
- `tests/test_pipeline_guardrails.py`;
- the fast unit and validation gate;
- full cached accuracy for shared parser changes;
- targeted `--runs 10` benchmark evidence for affected behavior clusters.
