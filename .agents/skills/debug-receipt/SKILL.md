---
name: debug-receipt
description: Debug and fix receipt parsing failures in the financial-aid receipt parser. Use this skill whenever one or more scanned receipts produce incorrect extraction results, a benchmark or accuracy test fails, or the user says things like "receipt X is broken", "debug receipt", "fix parsing for receipt N", "debug receipts N, M, and P", "receipt errors", or "run benchmark". Also triggers for truth file questions, adding new receipt fixtures, or investigating extraction accuracy. This is the go-to skill for any receipt pipeline debugging workflow, single or batch.
---

# Debug Receipt

A systematic playbook for diagnosing and fixing receipt parsing failures. This skill drives an iterative loop: benchmark → diagnose → fix → verify → repeat until 100%.

Works for **one receipt or many** — the loop is the same, with some multi-receipt-specific guidance called out inline.

## Critical Rules

1. **Never modify truth files.** If the truth file is wrong, tell the user exactly what to change and wait for confirmation before continuing.
2. **No brittle fixes.** Every code change must be general-purpose. No hardcoded receipt-specific mappings, no `if receipt_37`-style logic, no lookup tables that only solve one case. If a fix wouldn't work on a receipt you've never seen, it's too brittle.
3. **Benchmark runs = 10.** Always use `--runs 10` to confirm determinism.
4. **In multi-receipt mode, group failures by type before fixing.** A single root cause often spans multiple receipts — fixing it once is better than fixing five symptoms.

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

After applying fixes:

```bash
# Run unit + validation tests first
conda run -n financial-aid python -m pytest tests/test_unit.py tests/test_validation.py -v

# Then re-benchmark the target receipt(s) — all of them in one call
conda run -n financial-aid python tests/benchmark.py --fixtures receipt_N [receipt_M receipt_P ...] --runs 10
```

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
conda run -n financial-aid python -m pytest tests/test_accuracy.py -v
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
