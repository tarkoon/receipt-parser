---
name: debug-receipt
description: Debug and fix receipt parsing failures in the financial-aid receipt parser. Use this skill whenever one or more scanned receipts produce incorrect extraction results, a benchmark or accuracy test fails, or the user says things like "receipt X is broken", "debug receipt", "fix parsing for receipt N", "debug receipts N, M, and P", "receipt errors", or "run benchmark". Also triggers for truth file questions, adding new receipt fixtures, or investigating extraction accuracy. This is the go-to skill for any receipt pipeline debugging workflow, single or batch.
---

# Debug Receipt

A systematic playbook for diagnosing and fixing receipt parsing failures. This skill drives an iterative loop: benchmark → diagnose → fix → verify → repeat until 100%.

Works for **one receipt or many** — the loop is the same, with some multi-receipt-specific guidance called out inline.

## Critical Rules

1. **Never modify truth files.** If the truth file is wrong, tell the user exactly what to change and wait for confirmation before continuing.
2. **No brittle fixes or embedded answer keys.** Production parsing code must not special-case specific merchants, receipt IDs, known fixture ranges, known dates, known product lists, or known final totals. It may implement general layout/format strategies when they are triggered by structural OCR evidence and validated by arithmetic consistency. No hardcoded receipt-specific mappings, no `if receipt_37`-style logic, no target-range helpers, no full hardcoded `line_items` answer lists, and no late "known final output" overrides. If a fix would not work on a receipt you've never seen, it is too brittle.
3. **Benchmark runs = 10.** Always use `--runs 10` to confirm determinism.
4. **In multi-receipt mode, group failures by type before fixing.** A single root cause often spans multiple receipts — fixing it once is better than fixing five symptoms.
5. **A passing score is not valid if the implementation is invalid.** 100% accuracy is the acceptance test, not the implementation strategy. A score achieved through merchant/product/date/fixture-specific overrides is a failed run and must be cleaned up before completion.

## Reviewability Rules

- Do not make a parser PR over roughly 1,500 changed lines without explicit user approval.
- Keep one behavior cluster per commit.
- Every behavior cluster must include: root cause, generality check, targeted `--runs 10`, guardrail run, and full accuracy when shared parser behavior changed.
- Any new production parser helper must state its structural trigger and arithmetic or field-consistency invariant in a test name, docstring, or phase metadata.
- For broad multi-receipt cleanup, use separate investigations for code archaeology, fixture convention survey, and test/benchmark evidence review before editing parser behavior.
- Do not grow `postprocess_receipt` or `_apply_final_receipt_output_repairs` unless the guardrail test is updated with an explicit temporary-debt reason and a plan to shrink it later.
- Late final-output repairs must remain explicit traced stages. Silent semantic mutation after validation is not acceptable.

## Environment

All Python commands run through conda:

```
conda run -n financial-aid <command>
```

## Project Layout

```
tests/
├── benchmark.py                    # Determinism benchmark (--fixtures, --runs)
├── test_accuracy.py                # Pytest accuracy suite across all fixtures
├── test_unit.py                    # Unit tests
├── test_validation.py              # Validation tests
├── fixtures/
│   ├── receipt_N_truth.json        # Ground truth for receipt N
│   ├── receipt_N.{jpg,png}         # Receipt image
│   └── _truth_template.json        # Conventions & allowed values
└── results/
    ├── benchmark/latest.json       # Last benchmark output
    └── accuracy/latest.json        # Last accuracy output

src/receipt_parser/                  # Pipeline code (extraction, normalization, post-processing)
```

## Mode Selection

Before starting the loop, determine which mode you're in:

- **Single-receipt mode** — one receipt to debug (e.g., `/debug-receipt receipt_42`)
- **Multi-receipt mode** — two or more receipts (e.g., `/debug-receipt receipt_40 receipt_41 receipt_42`, or "debug all the new ones")

In multi-receipt mode, the loop is the same but with a different emphasis: you diagnose across the full set first, then fix root causes in batches rather than walking each receipt end-to-end in isolation.

## Compaction / Resume Checkpoint

After any context compaction, resume, or long-running continuation, stop before editing files or running more benchmarks and restate:

- Target fixture(s) and current mode
- Non-negotiable rules from this skill, especially the no-brittle-fixes rule
- Last trusted benchmark or accuracy artifact
- Last explicit user decision
- Current blockers or truth-file questions
- Next planned action and why it is still general-purpose

If any item is unknown, inspect the current repo state and artifacts first. Do not continue from memory alone. If a future step would contradict the last user decision, stop and ask. If identical image/OCR fixtures have conflicting truth expectations, stop immediately and ask the user which truth convention should win; do not add parser logic that distinguishes identical inputs by fixture identity.

## Adding Flagged PROD Fixtures

When the user asks to add flagged receipts from production, use the tracked exporter instead of generating truth with an LLM:

```bash
PYTHONIOENCODING=utf-8 python scripts/add_flagged_receipts.py --source prod --dry-run --limit 3
PYTHONIOENCODING=utf-8 python scripts/add_flagged_receipts.py --source prod --apply --limit 3
```

The exporter reads flagged saved receipts from Stardust Postgres over SSH, copies images from `/home/tarkoon/data/paper-ledger/storage`, and writes `tests/fixtures/receipt_N_truth.json` from the existing `_truth_template.json` shape.

Important conventions:

- Production saved rows are the truth source: `receipts`, `line_items`, `tax_entries`, `billing_periods`, and `usage_data`. `raw_json` is audit context only.
- Every truth JSON must follow the stripped fixture template order. Empty list sections are `[]`; object sections stay present with null members, e.g. `billing_period: {"start": null, "end": null}` and the full null-shaped `usage` object.
- The exporter tracks production `receipt_id`, `image_path`, `updated_at`, fixture name, and checksum in `local/prod_flagged_receipts_manifest.json`.
- Do not overwrite existing fixture images or truth files unless the user explicitly approves `--overwrite`.
- After exporting, continue with the normal benchmark/debug loop below.

## The Loop

### Step 1 — Benchmark the target receipt(s)

**Single-receipt:**

```bash
conda run -n financial-aid python tests/benchmark.py --fixtures receipt_N --runs 10
```

**Multi-receipt:** pass all targets in a single benchmark call (faster, and produces a unified result set to analyze):

```bash
conda run -n financial-aid python tests/benchmark.py --fixtures receipt_N receipt_M receipt_P --runs 10
```

For 3+ receipts, add `--workers 4` (or up to 8) to parallelize fixture processing and cut wall-clock time substantially:

```bash
conda run -n financial-aid python tests/benchmark.py --fixtures receipt_N receipt_M receipt_P --runs 10 --workers 4
```

If unsure whether `--fixtures` accepts multiple values, check once with `python tests/benchmark.py --help`. It does as of the current implementation.

Parse the output. For each field on each receipt, note:

- **Pass/Fail** and robustness % across runs
- Which fields are non-deterministic (< 100% robustness) vs deterministic failures

If all fields pass at 100% on all targets, stop — nothing to fix.

### Step 2 — Investigate failures

For each failing field, gather three things:

| What            | Where                                       | How              |
| --------------- | ------------------------------------------- | ---------------- |
| Expected value  | `tests/fixtures/receipt_N_truth.json`       | Read the file    |
| Pipeline output | `tests/results/benchmark/latest.json`       | Read the results |
| Raw OCR text    | OCR cache in project (search for receipt_N) | Find and read    |

Use **sub-agents** (Explore type) for parallel investigation. Agent strategy depends on mode:

**Single-receipt mode:**

- One agent to inspect the truth file + benchmark results + receipt image
- One agent to read pipeline code (extraction, normalization, post-processing)
- One agent to survey conventions across all fixtures (when a pattern question arises)

**Multi-receipt mode:**

- One agent per **failure cluster** (see Step 3), not per receipt — e.g., "investigate all tax-label failures across receipts 40, 41, 42" is one agent, not three
- One agent to read pipeline code relevant to the clustered failures
- One agent to survey conventions across all fixtures if the cluster exposes a convention question

Spawning one agent per receipt is almost always the wrong call in multi-receipt mode — you'll duplicate work and miss shared root causes.

### Step 3 — Classify each failure

Every failure is one of:

| Classification                                                  | Action                                                                                           |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Truth file error** — the expected value is wrong              | Tell the user exactly what to change. Do NOT edit the file yourself. Wait for confirmation.      |
| **Pipeline bug** — the code produces wrong output               | Fix the code (see Step 4).                                                                       |
| **Convention mismatch** — truth file uses a non-standard format | Survey all fixtures to establish the convention, then recommend a truth file update to the user. |

**Multi-receipt mode: cluster before classifying.** Before going field-by-field, build a small table of failures across all target receipts:

| Receipt    | Field     | Expected  | Got         | Likely cause |
| ---------- | --------- | --------- | ----------- | ------------ |
| receipt_40 | tax_label | 内税      | tax         | convention   |
| receipt_41 | tax_label | 内税      | (missing)   | pipeline     |
| receipt_41 | total     | 5248      | 5200        | pipeline     |
| receipt_42 | merchant  | スマート… | スマートビ… | truth?       |

Group rows by likely cause. Each cluster becomes one investigation + one fix, regardless of how many receipts it affects. This is where multi-receipt debugging earns its efficiency.

When in doubt about conventions, **survey all fixtures first**. Spawn a sub-agent to read every `*_truth.json` and tabulate the patterns (e.g., tax label formats, zero-amount entry handling, location granularity).

### Step 4 — Fix pipeline bugs

Before writing any fix, state:

1. **Root cause** — what specifically goes wrong and why
2. **Fix approach** — what you'll change
3. **Generality check** — why this fix works for any receipt, not just the current targets. In multi-receipt mode, explicitly name which receipts in the target set the fix addresses, and why it won't regress others.

The generality check must explicitly reject known-answer behavior. A valid fix may key off OCR structure such as row geometry, repeated column format, printed tax-summary labels, item/price alignment, or arithmetic consistency. It must not key off merchant identity, date, receipt ID, fixture range, a known product set, or a known final total except as diagnostic evidence reported to the user.

After applying fixes:

```bash
# Run unit + validation + production-code guardrails first
conda run -n financial-aid python -m pytest tests/test_unit.py tests/test_validation.py tests/test_pipeline_guardrails.py -v

# Then re-benchmark the target receipt(s) — all of them in one call
conda run -n financial-aid python tests/benchmark.py --fixtures receipt_N [receipt_M receipt_P ...] --runs 10
```

`tests/test_pipeline_guardrails.py` enforces the no-brittle-production-code rule. If it fails, the benchmark score does not count as a valid pass. Either remove/redesign the brittle production code or, for a pre-existing documented violation only, update the exact allowlist with a clear explanation and a plan to shrink it later. Do not add a new allowlist entry for code introduced in the current fix unless the user explicitly approves keeping that debt.

**Before re-benchmarking, snapshot the pre-fix results** so you can diff them against the post-fix run. The benchmark overwrites `latest.json` on each run, so copy it aside first:

```bash
# Snapshot the pre-fix results, then re-benchmark with --compare
cp tests/results/benchmark/latest.json tests/results/benchmark/before-fix.json
conda run -n financial-aid python tests/benchmark.py --fixtures receipt_N [receipt_M receipt_P ...] --runs 10 --compare tests/results/benchmark/before-fix.json
```

`--compare` will highlight exactly which fields changed — progress where the fix worked, and any unexpected regressions. This is far more reliable than eyeballing two JSON files, especially in multi-receipt mode where changes are spread across many fields.

If the fix breaks unit tests, revert and rethink.

**Multi-receipt mode:** fix one cluster at a time, then re-benchmark all targets (with `--compare`) before moving to the next cluster. This catches any cross-cluster interactions early and makes it obvious when a fix for cluster A accidentally solved or broke something in cluster B.

### Step 5 — Full accuracy sweep

Once all target receipts pass at 100%:

```bash
conda run -n financial-aid python -m pytest tests/test_accuracy.py tests/test_pipeline_guardrails.py -v
```

If ANY other receipt regresses, go back to Step 2 for each regression. The fix was too broad or exposed a latent issue. In multi-receipt mode this is especially important because a batched fix has more surface area to cause unintended regressions.

### Step 6 — Iterate

Repeat Steps 2–5 until full accuracy is 100% across all fixtures.

### Step 7 — Final confirmation

Re-run the original benchmark one last time with **all** target receipts:

```bash
conda run -n financial-aid python tests/benchmark.py --fixtures receipt_N [receipt_M receipt_P ...] --runs 10
```

Confirm 100% pass rate, 100% determinism, on every target.

Before marking the task complete, audit the production diff for brittle-code smells:

```bash
conda run -n financial-aid python -m pytest tests/test_pipeline_guardrails.py -v
rg -n "receipt_[0-9]+|target_[0-9]+|known|final_known|override|_known_item|line_items\\] = \\[|merchant|date" src/receipt_parser tests
```

For each new production-code match, explain why it is a general layout/format strategy rather than a fixture answer key. If it cannot be justified, remove or redesign it before completing. Final-result assembly must not silently mutate receipt fields after validation unless that stage is explicitly named, tested, and included in the benchmark evidence. A failing `tests/test_pipeline_guardrails.py` is a completion blocker even if all receipt benchmarks pass.

### Step 8 — Document

If any conventions were clarified during debugging, update `_truth_template.json` with:

- Allowed values for the field (e.g., tax labels: `内税`, `外税`, `非課税`)
- Rules for edge cases (e.g., "omit zero-amount tax entries")
- Ask the user to apply the update if the file should be treated as read-only

Summarize what was fixed. In multi-receipt mode, organize the summary by cluster (root cause) rather than by receipt:

- **Pipeline changes** — file, function, what changed, why, which receipts it resolved
- **Truth file corrections** — which files changed, by whom (always the user), under which cluster
- **New conventions documented** — one line per convention with the template section updated

## Common Pitfalls

- **`合計` regex matching too broadly** — Japanese receipts have multiple "total"-like lines (お預り合計 = cash tendered, 小計 = subtotal). Exclusion logic must be general pattern-based, not receipt-specific.
- **Tax label normalization** — Always normalize to canonical labels. Check `_truth_template.json` for the allowed set.
- **Zero-amount tax entries** — Current convention is to omit them. Survey fixtures if unsure.
- **Fuzzy match thresholds** — `location` uses 50% similarity, `merchant` uses 40%. Low scores may pass but indicate extraction gaps worth improving.
- **Multi-receipt — receipt-by-receipt tunnel vision** — walking each receipt end-to-end in isolation misses the shared root cause. Cluster failures first, fix once.
- **Multi-receipt — one sub-agent per receipt** — explodes the investigation and duplicates work. One agent per failure cluster is the right unit.
- **Multi-receipt — partial re-benchmarking** — after a fix, always re-run the benchmark on the full target set, not just the receipts the fix was "aimed at". Side effects love to hide in the receipts you weren't watching.
