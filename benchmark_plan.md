# Benchmark & Testing Restructure Plan

## Context

The receipt-parser pipeline has three overlapping test/evaluation tools:
- `test_integration.py` — pytest integration tests (cached OCR, pass/fail only)
- `benchmark_robustness.py` — OCR variance stress-test (fresh OCR, rich output)
- `benchmark_models.py` — multi-model comparison (cached OCR, speed + accuracy)

All three duplicate the same field-checking logic, the benchmark uses fragile monkey-patching
to instrument OCR and LLM calls, and there's no regression testing against known OCR variants.

This plan consolidates the testing story into a clean architecture.

---

## Goals

1. **Single source of truth** for field checks — no more duplicated check functions
2. **No monkey-patching** — core functions return rich metadata natively
3. **Regression testing** against known OCR variants that caused failures
4. **Parallel benchmark execution** across fixtures
5. **Improved results output** with schema versioning and companion files

---

## Architecture Overview

```
src/receipt_parser/
  checks.py          # NEW — all field checks, doc-type-aware
  ocr.py             # MODIFIED — OCRResult dataclass, skip_cache param
  llm.py             # MODIFIED — LLMResult dataclass, timing in return
  pipeline.py        # MODIFIED — process_ocr_text() entry point, richer metadata

tests/
  test_accuracy.py   # NEW — replaces test_integration.py
  test_unit.py       # unchanged
  test_validation.py # unchanged
  benchmark.py       # MOVED from scripts/benchmark_robustness.py, refactored
  conftest.py        # updated — accuracy summary plugin
  fixtures/          # images + truth JSONs
  ocr_variants/      # NEW — auto-saved OCR text variants for regression
  results/           # NEW — all test/benchmark output
    accuracy/        #   pytest-json-report output
      latest.json
    benchmark/       #   benchmark JSON output + companion files
      latest.json
      ocr/           #   companion OCR text per run
      llm/           #   companion LLM raw extraction per run

scripts/
  benchmark_models.py  # MODIFIED — imports checks from checks.py
  ...
```

---

## Phase 1: Foundation

### 1.1 Create `src/receipt_parser/checks.py`

Extract all field-checking logic into a single module. Every check function takes
`(result: dict, truth: dict)` and returns `{"pass": bool, "expected": ..., "got": ..., "detail": str}`.

```python
# Common checks (all document types)
COMMON_CHECKS = {
    "total":              check_total,
    "date":               check_date,
    "currency":           check_currency,
    "merchant_similarity": check_merchant_similarity,
    "document_type":      check_document_type,
    "amount_paid":        check_amount_paid,
    "payment_method":     check_payment_method,
}

# Receipt-specific
RECEIPT_CHECKS = {
    "subtotal":           check_subtotal,
    "line_items_count":   check_line_items_count,
    "line_items_totals":  check_line_items_totals,
    "tax_amount":         check_tax_amount,
    "tax_categories":     check_tax_categories,
    "item_descriptions":  check_item_descriptions,
}

# Utility bill-specific
UTILITY_CHECKS = {
    "service_type":       check_service_type,
    "usage_amount":       check_usage_amount,
}

# Payment slip-specific
SLIP_CHECKS = {
    "payer":              check_payer,
}

def get_checks_for(truth: dict) -> dict:
    """Return the right checks based on document_type in truth."""
    doc_type = truth.get("document_type", "receipt")
    checks = dict(COMMON_CHECKS)
    if doc_type == "receipt":
        checks.update(RECEIPT_CHECKS)
    elif doc_type == "utility_bill":
        checks.update(UTILITY_CHECKS)
    elif doc_type == "payment_slip":
        checks.update(SLIP_CHECKS)
    return checks
```

**Source of check logic:**
- `check_total`, `check_date`, `check_currency`, `check_subtotal`, `check_payment_method`,
  `check_merchant_similarity`, `check_line_items_count`, `check_line_items_totals`,
  `check_tax_amount`, `check_tax_categories`, `check_item_descriptions`
  — ported from `benchmark_models.py` `FIELD_CHECKS`
- `check_document_type`, `check_amount_paid`, `check_service_type`, `check_usage_amount`,
  `check_payer`
  — ported from `test_integration.py` (these exist there but NOT in the benchmarks today)

The merchant similarity function (with katakana-to-romaji fallback) also lives here since
all three files currently duplicate it.

### 1.2 `OCRResult` dataclass in `ocr.py`

Replace the raw `list[dict]` return from `run_cloud_vision()` with a structured result.
This eliminates OCR monkey-patching — the benchmark reads metadata from the return value.

```python
@dataclass
class OCRResult:
    blocks: list[dict]
    confidence: float          # weighted average from compute_ocr_confidence()
    retried: bool              # True if confidence < threshold triggered retry
    retry_reason: str | None   # e.g. "confidence 0.68 < 0.75"
    source: str                # "cache", "fresh", "digital_pdf"
    chosen_text: str           # the fulltext that was used (for variant saving)
```

**`run_cloud_vision()` changes:**

```python
def run_cloud_vision(image, client=None, *, skip_cache=False) -> OCRResult:
```

- `skip_cache=False` (default): existing behavior, checks `.ocr_cache/` first
- `skip_cache=True`: always makes fresh API call, does NOT write to cache
- Both paths populate all `OCRResult` fields
- The benchmark calls `run_cloud_vision(image, skip_cache=True)` — no patching needed

**Callers updated:**
- `pipeline.py` — uses `result.blocks` where it previously used the raw list,
  reads `result.confidence`, `result.source` for metadata
- `benchmark.py` — reads `result.confidence`, `result.retried`, `result.chosen_text`

### 1.3 `LLMResult` dataclass in `llm.py`

Replace the raw `str` return from `_llm_chat()` with a structured result.
This eliminates LLM timing monkey-patching.

```python
@dataclass
class LLMResult:
    content: str                   # the JSON response text
    input_tokens: int | None       # from API usage / Ollama metadata
    output_tokens: int | None
    eval_duration_ns: int | None   # wall time for generation
    total_duration_ns: int | None  # total including load
    load_duration_ns: int | None   # model load time (Ollama only)
    backend: str                   # "api" or "ollama"
```

**`_llm_chat()` changes:**

```python
def _llm_chat(model, messages, schema, ...) -> LLMResult:
```

- Ollama path: reads timing from `response` dict (already available, currently discarded)
- API path: reads `response.usage` + wall time (currently discarded)
- Returns `LLMResult` instead of raw string

**Callers updated:**
- `extract_with_llm()` — uses `result.content` where it previously used the raw string
- `extract_with_verification()` — collects `LLMResult` from each pass, makes timing
  available in the pass history
- `pipeline.py` `process_document()` — passes timing through to result metadata
- `benchmark.py` — reads timing fields from pass history, no patching needed

### 1.4 `process_ocr_text()` in `pipeline.py`

New entry point that accepts raw OCR text and runs the pipeline from the LLM extraction
stage onwards. This is what `test_accuracy.py` uses for OCR variant regression tests.

```python
def process_ocr_text(
    ocr_text: str,
    model: str = DEFAULT_MODEL,
    passes: int = 1,
    apply_user_rules: bool = True,
) -> dict:
    """Run the pipeline from OCR text onwards (skip image loading + OCR).

    Used for:
    - Testing against saved OCR variants (regression tests)
    - Debugging with specific OCR output
    - Benchmarking LLM extraction independently of OCR variance
    """
```

This function:
1. Runs `normalize_fullwidth()` + `strip_barcode_lines()` + `clean_handwritten_ocr()`
2. Detects document type
3. Calls `extract_with_verification()`
4. Runs validation + post-processing
5. Returns the same result dict as `process_document()`

It does NOT:
- Load images
- Call Cloud Vision
- Write to OCR cache
- Generate debug visualizations (no bounding boxes to draw)

---

## Phase 2: Test Layer

### 2.1 Create `tests/test_accuracy.py`

Replaces `test_integration.py`. Discovers and tests against:
1. **Image fixtures** — `tests/fixtures/receipt_N.jpg` + `receipt_N_truth.json` (uses cached OCR)
2. **OCR variants** — `tests/ocr_variants/receipt_N_vM.txt` (uses `process_ocr_text()`)

```python
"""Accuracy tests — field checks against cached pipeline results and OCR variants."""

import json
from pathlib import Path
import pytest
from receipt_parser.pipeline import process_document, process_ocr_text
from receipt_parser.checks import get_checks_for

FIXTURES = Path(__file__).parent / "fixtures"
VARIANTS = Path(__file__).parent / "ocr_variants"


def _discover_test_cases():
    """Auto-discover image fixtures and OCR variants."""
    cases = []

    # Image fixtures (standard path through process_document with cached OCR)
    for truth_file in sorted(FIXTURES.glob("*_truth.json")):
        base = truth_file.stem.replace("_truth", "")
        image = _find_image(base)
        if not image:
            continue
        truth = json.loads(truth_file.read_text(encoding="utf-8"))
        cases.append((base, {"type": "image", "path": image}, truth))

    # OCR variant fixtures (process_ocr_text, skip OCR stage)
    if VARIANTS.exists():
        for variant_file in sorted(VARIANTS.glob("*.txt")):
            # Extract base fixture name: receipt_14_v1.txt -> receipt_14
            stem = variant_file.stem
            # Find the matching truth file
            base = _extract_base_name(stem)  # receipt_14_v1 -> receipt_14
            truth_file = FIXTURES / f"{base}_truth.json"
            if not truth_file.exists():
                continue
            truth = json.loads(truth_file.read_text(encoding="utf-8"))
            cases.append((stem, {"type": "ocr_text", "path": variant_file}, truth))

    return cases


_CASES = _discover_test_cases()
_CASE_IDS = [c[0] for c in _CASES]
_RESULTS_CACHE: dict[str, dict] = {}


def _get_result(case_id: str, source: dict) -> dict:
    """Run pipeline and cache the result."""
    if case_id not in _RESULTS_CACHE:
        if source["type"] == "image":
            _RESULTS_CACHE[case_id] = process_document(
                source["path"], passes=3, apply_user_rules=False
            )
        else:
            ocr_text = source["path"].read_text(encoding="utf-8")
            _RESULTS_CACHE[case_id] = process_ocr_text(
                ocr_text, passes=3, apply_user_rules=False
            )
    return _RESULTS_CACHE[case_id]


@pytest.mark.parametrize("name,source,truth", _CASES, ids=_CASE_IDS)
def test_fields(name, source, truth):
    """Run all applicable field checks for this fixture."""
    result = _get_result(name, source)
    checks = get_checks_for(truth)
    failures = []
    for field_name, check_fn in checks.items():
        check_result = check_fn(result, truth)
        if not check_result["pass"]:
            failures.append(f"{field_name}: {check_result.get('detail', 'failed')}")
    assert not failures, f"Failed checks:\n" + "\n".join(f"  - {f}" for f in failures)
```

**Key properties:**
- Auto-discovers fixtures + variants with no code changes needed
- Uses `get_checks_for()` so doc-type-specific checks apply automatically
- Result caching so each fixture/variant is only processed once per test run
- Cloud Vision skip: tests that use OCR variants don't need Cloud Vision at all

### 2.2 Accuracy test output

`test_accuracy.py` produces two forms of output:

**A) pytest-json-report (machine-readable)**

Add `pytest-json-report` to dev dependencies. Run with:
```bash
python -m pytest tests/test_accuracy.py -v \
    --json-report --json-report-file=tests/results/accuracy/latest.json
```

This gives a structured JSON with every test case, duration, and failure detail —
useful for CI dashboards, trend tracking, or programmatic comparison.

**B) Summary table (human-readable, printed at end of run)**

A pytest plugin/fixture in `conftest.py` collects check results across all test cases
and prints a summary table after the session:

```
=== Accuracy Summary ===
Image fixtures:  36 tested, 34 passed, 2 failed
OCR variants:     4 tested,  3 passed, 1 failed
Total:           40 tested, 37 passed, 3 failed (92.5%)

Failed:
  receipt_14        line_items_count: expected 5, got 4
  receipt_29        merchant_similarity: got 'コスモス', expected 'コスモス薬品' (similarity 0.38)
  receipt_14_v2     tax_amount: expected 240, got 216

Per-field pass rate:
  total               40/40  100%
  date                39/40   98%
  currency            40/40  100%
  merchant_similarity 38/40   95%
  line_items_count    38/40   95%
  ...
```

This summary is always printed to console. The JSON report is saved to
`tests/results/accuracy/latest.json` when the `--json-report` flag is used.

### 2.3 Delete `tests/test_integration.py`

All functionality is covered by `test_accuracy.py`:
- Image fixture tests = same as current integration tests
- OCR variant tests = new regression coverage
- Doc-type-specific checks = ported to `checks.py` (which integration tests were missing from benchmarks)

### 2.4 Create `tests/ocr_variants/` directory

Empty initially. Populated automatically by benchmark runs. Each file is:
- Named `{fixture_base}_v{N}.txt` (e.g., `receipt_14_v1.txt`)
- Contains the raw OCR fulltext that caused a pipeline failure
- Tracked in git as part of the regression corpus

---

## Phase 3: Benchmark Refactor

### 3.1 Move and rename

`scripts/benchmark_robustness.py` -> `tests/benchmark.py`

The benchmark tests the system with richer output than pytest. It belongs with other
test tooling. `benchmark_models.py` stays in `scripts/` (different purpose: model comparison).

### 3.2 Remove all monkey-patching

**Before (current):**
```python
# Patch OCR to bypass cache and collect metadata
ocr_mod.run_cloud_vision = _instrumented_run_cloud_vision
pipeline_mod.run_cloud_vision = _instrumented_run_cloud_vision

# Patch LLM to collect timing
extraction._ollama_chat_with_timeout = _instrumented_chat_with_timeout
extraction._openrouter_chat = _instrumented_openrouter_chat
```

**After (new):**
```python
# No patching needed — use parameters and return values
result = process_document(image, passes=passes, skip_ocr_cache=True)
# result now contains:
#   _ocr_confidence, _ocr_retried, _ocr_source, _ocr_text
#   _llm_timing (per-pass with token counts, durations)
#   _pass_history (with LLM raw extraction per pass)
```

The benchmark becomes a straightforward caller of the public API.

**Pipeline changes to support this:**

`process_document()` gains a `skip_ocr_cache` parameter that passes through to
`run_cloud_vision(skip_cache=True)`. The result dict is extended with:

```python
result["_ocr_text"] = ocr_result.chosen_text          # for variant saving
result["_ocr_confidence"] = ocr_result.confidence      # already exists
result["_ocr_retried"] = ocr_result.retried            # new
result["_ocr_retry_reason"] = ocr_result.retry_reason  # new
result["_llm_timing"] = aggregate_timing_from_history   # new (from LLMResult)
```

### 3.3 Thread-safe parallel execution

With monkey-patching eliminated, parallelizing across fixtures is safe.
Each `process_document()` call is independent — no global collectors to clobber.

```python
def run_benchmark(fixtures, runs, workers=4, ...):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_fixture, name, image, truth, runs, ...): name
            for name, image, truth in fixtures
        }
        for future in as_completed(futures):
            fixture_name, fixture_result = future.result()
            per_fixture[fixture_name] = fixture_result
            _save_progress(...)
```

Each fixture's N runs execute sequentially within `_run_fixture()` (needed for
budget checks and resume). But multiple fixtures run concurrently.

New CLI flag: `--workers N` (default: 1, max: 8)

### 3.4 Auto-save OCR variants

When a run fails fields that passed in other runs for the same fixture, the benchmark
auto-saves the OCR text to `tests/ocr_variants/`:

```python
VARIANTS_DIR = Path(__file__).parent / "ocr_variants"

def _save_variant(fixture_name: str, ocr_text: str) -> Path | None:
    """Save a unique failing OCR variant. Returns path if saved, None if deduplicated."""
    VARIANTS_DIR.mkdir(exist_ok=True)

    # Find next version number
    existing = sorted(VARIANTS_DIR.glob(f"{fixture_name}_v*.txt"))
    
    # Deduplicate: skip if too similar to any existing variant
    for existing_file in existing:
        existing_text = existing_file.read_text(encoding="utf-8")
        if SequenceMatcher(None, ocr_text, existing_text).ratio() > 0.98:
            return None

    version = len(existing) + 1
    path = VARIANTS_DIR / f"{fixture_name}_v{version}.txt"
    path.write_text(ocr_text, encoding="utf-8")
    return path
```

The variant is immediately available to `test_accuracy.py` on the next pytest run.
No manual promotion step. If a variant is noise, delete the file from git.

### 3.5 Output locations and results JSON

All benchmark output goes to `tests/results/benchmark/`:

```
tests/results/benchmark/
  latest.json                   # most recent results (overwritten each run)
  2026-04-01_abc123.json        # timestamped archive (git_sha in filename)
  ocr/                          # companion OCR text per run
    receipt_14_run2.txt
  llm/                          # companion LLM raw extraction per run
    receipt_14_run2.json
```

This is separate from `tests/ocr_variants/` (which stores promoted regression fixtures)
and `tests/results/accuracy/` (which stores pytest-json-report output).

**Results JSON structure (schema v2):**

```json
{
  "schema_version": 2,
  "metadata": {
    "timestamp": "2026-04-01T14:30:00",
    "git_sha": "8e6c89b",
    "model": "deepseek-chat",
    "runs_per_fixture": 3,
    "passes": 2,
    "workers": 4,
    "cloud_vision_model": "builtin/stable",
    "fixtures": ["receipt_1", "receipt_2", "..."],
    "api_calls_used": 90,
    "api_calls_remaining": 869
  },

  "summary": {
    "score": 0.9714,
    "total_checks": 385,
    "total_passed": 374,
    "fixtures_robust": 33,
    "fixtures_fragile": 3,
    "fragile": ["receipt_14", "receipt_29", "receipt_33"],
    "cost_usd": 0.0312,
    "mean_wall_s": 2.1,
    "determinism_rate": 0.92,
    "variants_saved": 2
  },

  "per_fixture": {
    "receipt_1": {
      "status": "ROBUST",
      "score": 1.0,
      "deterministic": true,
      "runs": [
        {
          "run": 1,
          "passed": true,
          "pass_count": 11,
          "total_fields": 11,
          "wall_time_s": 1.8,
          "fields": {
            "total": {"pass": true},
            "date": {"pass": true, "expected": "2026-01-15", "got": "2026-01-15"},
            "...": "..."
          },
          "ocr": {
            "confidence": 0.94,
            "retried": false,
            "source": "fresh",
            "text_file": "ocr/receipt_1_run1.txt"
          },
          "llm_timing": {
            "passes": 2,
            "eval_s": 1.2,
            "input_tokens": 1850,
            "output_tokens": 420,
            "cost_usd": 0.00096
          },
          "llm_raw_file": "llm/receipt_1_run1.json",
          "warnings": [],
          "overrides": []
        }
      ],
      "field_robustness": {
        "total": {"pass_rate": 1.0, "consistent": true},
        "...": "..."
      },
      "variance_attribution": {},
      "ocr_analysis": {
        "mean_confidence": 0.94,
        "retried_pct": 0.0,
        "cross_run_similarity": {"mean": 0.99, "min": 0.97}
      }
    }
  },

  "overall": {
    "robustness_score": 0.9714,
    "field_robustness": {"total": 1.0, "date": 0.98, "...": "..."},
    "variance_attribution": {"OCR_VARIANCE": 8, "LLM_VARIANCE": 2, "POST_PROCESSING": 1},
    "cost_summary": {
      "total_cost_usd": 0.0312,
      "cost_per_receipt_usd": 0.00029
    },
    "llm_timing_summary": {
      "mean_eval_s": 1.1,
      "mean_wall_s": 2.1
    },
    "determinism_summary": {
      "deterministic_fixtures": 33,
      "total_fixtures": 36,
      "rate": 0.92
    }
  }
}
```

**Key improvements over current format:**
- `schema_version` — forward compatibility for tooling
- `summary` at top level — one glance for pass/fail, cost, speed
- `git_sha` in metadata — ties results to exact code version
- OCR text + LLM raw saved as companion files (referenced by path, not stripped)
- `status` + `score` per fixture at top level — no digging into nested runs
- `passed: bool` per run — flat boolean for filtering
- `variants_saved` count in summary — shows regression corpus growth

### 3.6 CLI changes

```bash
# Default: 3 runs, all fixtures, sequential
python tests/benchmark.py

# Parallel across fixtures
python tests/benchmark.py --workers 4

# Quick test on specific fixtures
python tests/benchmark.py --fixtures receipt_14 receipt_29

# CI mode: 1 run, cached OCR, exit non-zero on failure
python tests/benchmark.py --ci

# Compare against previous run
python tests/benchmark.py --compare tests/results/benchmark/latest.json
```

New/changed flags:
| Flag | Default | Description |
|---|---|---|
| `--workers N` | 1 | Concurrent fixture processing (max 8) |
| `--ci` | false | Cached OCR, 1 run, exit non-zero on failure |
| `--output PATH` | `tests/results/benchmark/latest.json` | Results output |

---

## Phase 4: Cleanup

### 4.1 Update `scripts/benchmark_models.py`

Replace inline field check functions with imports from `checks.py`:

```python
# Before:
FIELD_CHECKS = {"total": check_total, ...}  # locally defined

# After:
from receipt_parser.checks import get_checks_for, COMMON_CHECKS, RECEIPT_CHECKS
```

`benchmark_models.py` stays in `scripts/` — its purpose (multi-model Ollama comparison)
is distinct from the robustness benchmark.

### 4.2 Add `pytest-json-report` dependency

Add to `[project.optional-dependencies]` in `pyproject.toml`:
```toml
[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-json-report>=1.5"]
```

### 4.3 Update `CLAUDE.md`

Update run instructions to reflect new file locations and commands:
```
# Run accuracy tests (fast, cached OCR)
python -m pytest tests/test_accuracy.py tests/test_unit.py tests/test_validation.py -v

# Run accuracy tests with JSON report
python -m pytest tests/test_accuracy.py -v \
    --json-report --json-report-file=tests/results/accuracy/latest.json

# Run robustness benchmark
python tests/benchmark.py --workers 4

# Compare models
python scripts/benchmark_models.py --models deepseek-chat ollama/qwen3.5:9b
```

### 4.4 Remove stale files

- Delete `tests/test_integration.py`
- Delete `scripts/benchmark_robustness.py` (moved to `tests/benchmark.py`)
- Remove `robustness_debug/` references (variants now go to `tests/ocr_variants/`)

---

## Implementation Order

```
Phase 1: Foundation                          (no existing behavior changes)
  1.1 checks.py                              — extract + add doc-type checks
  1.2 OCRResult dataclass + skip_cache       — ocr.py refactor
  1.3 LLMResult dataclass                    — llm.py refactor
  1.4 process_ocr_text()                     — pipeline.py new entry point
  1.5 Update process_document()              — pipe through skip_ocr_cache, richer metadata
  >>> Validate: existing unit tests still pass

Phase 2: Test layer                          (swap integration tests)
  2.1 test_accuracy.py                       — new pytest using checks.py
  2.2 Accuracy summary plugin in conftest.py — per-field table printed after run
  2.3 Create tests/ocr_variants/             — empty, ready for benchmark
  2.4 Create tests/results/ dirs             — accuracy/ and benchmark/ output
  2.5 Delete test_integration.py
  >>> Validate: pytest tests/ passes with same coverage

Phase 3: Benchmark refactor                  (rewrite benchmark)
  3.1 Move + rename to tests/benchmark.py
  3.2 Remove monkey-patching, use skip_cache + dataclass returns
  3.3 Import checks from checks.py
  3.4 Add --workers for parallel fixtures
  3.5 Auto-save variants to tests/ocr_variants/
  3.6 New results JSON format (schema v2)
  3.7 Output to tests/results/benchmark/     — latest.json + companion files
  3.8 Add --ci mode
  >>> Validate: benchmark produces equivalent output to old script

Phase 4: Cleanup
  4.1 Update benchmark_models.py imports
  4.2 Add pytest-json-report to dev deps
  4.3 Update CLAUDE.md
  4.4 Remove stale files
```

---

## Risk Mitigations

### Breaking `process_document()` return type
The `OCRResult` and `LLMResult` changes modify internal function signatures. Risk: callers
break. Mitigation: phase 1 ends with "existing unit tests still pass" gate. The public
return type (result dict) gains fields but doesn't lose any.

### OCR variant spam
If a fixture is inherently fragile (OCR differs every time), the benchmark could save
many variants. Mitigation: deduplication at 0.98 similarity threshold. If a fixture
generates >5 variants, the benchmark prints a warning suggesting the fixture may need
image preprocessing.

### Thread safety of `process_document()`
With `--workers > 1`, multiple threads call `process_document()` concurrently. The
function must not use module-level mutable state. Current concern: `ocr._last_ocr_source`
is a module global. Fix: return source in `OCRResult` instead of setting a global.
Also: `_track_api_call()` uses file I/O for the usage counter — add a threading lock.

### Budget management with parallel execution
With N workers making concurrent API calls, the budget counter can race. Fix: use
`threading.Lock` around `_track_api_call()` and check budget in the main thread
between fixture completions, not inside workers.
