# Robustness Benchmark Plan

## Overview

Four coordinated changes to make the receipt-parser pipeline testably robust:

1. **Pin Cloud Vision to `builtin/stable`** — deterministic OCR within a model cycle
2. **Add `_ocr_source` metadata** — results report whether OCR was fresh or cached
3. **Build `benchmark_robustness.py`** — stress-test pipeline against OCR variation
4. **Dual-call analysis** — determine if the 2-call "pick best" strategy is needed
5. **Variance attribution** — isolate whether failures come from OCR or LLM

---

## Phase 1: Pin Cloud Vision to `builtin/stable`

**File:** `ocr.py`
**What:** Replace the implicit model in `_call_cloud_vision` with an explicit `builtin/stable` pin.

**Current code (line 93-112):**
```python
def _call_cloud_vision(image, client):
    from google.cloud import vision
    success, buf = cv2.imencode(".png", image)
    gcp_image = vision.Image(content=buf.tobytes())
    response = client.document_text_detection(
        image=gcp_image,
        image_context=vision.ImageContext(language_hints=["ja", "en"]),
    )
    _track_api_call()
    return response
```

**New approach:**
```python
def _call_cloud_vision(image, client):
    from google.cloud import vision
    success, buf = cv2.imencode(".png", image)
    gcp_image = vision.Image(content=buf.tobytes())
    features = [vision.Feature(
        type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION,
        model="builtin/stable",
    )]
    request = vision.AnnotateImageRequest(
        image=gcp_image,
        features=features,
        image_context=vision.ImageContext(language_hints=["ja", "en"]),
    )
    response = client.annotate_image(request=request)
    _track_api_call()
    return response
```

**Risk:** Minimal. `builtin/stable` is the current default — we're just making it explicit.
Response shape is identical (`AnnotateImageResponse`), so `_extract_fulltext_from_response`
and `_extract_blocks_from_response` need no changes.

**Fallback:** If `builtin/stable` is unavailable in the user's region, catch the error and
fall back to no model pin with a printed warning.

**Validation:** Run `python -m pytest tests/test_integration.py -v` (cached OCR, should pass unchanged).
Then delete one cache file and re-run to verify the pinned API call works.

---

## Phase 2: Add `_ocr_source` Metadata

**Files:** `ocr.py`, `pipeline.py`

### ocr.py changes

Add module-level tracking:

```python
_last_ocr_source: str = "unknown"   # "cache", "fresh", "digital_pdf", "unknown"

def get_last_ocr_source() -> str:
    return _last_ocr_source
```

In `run_cloud_vision`:
- Cache hit path (line 185-187): set `_last_ocr_source = "cache"`
- Fresh API path (line 192+): set `_last_ocr_source = "fresh"`

### pipeline.py changes

In `process_document`:
- After `run_cloud_vision` call (line 459): capture `ocr_source = get_last_ocr_source()`
- Digital PDF fast path (line 411): set `ocr_source = "digital_pdf"`
- Pass `ocr_source` through to `_build_result`

In `_build_result` (line 371):
- Add parameter `ocr_source: str = "unknown"`
- Add `result["_ocr_source"] = ocr_source` to output

### Result example
```json
{
  "merchant": "...",
  "total": 2447,
  "_ocr_source": "cache",
  "_warnings": [],
  ...
}
```

---

## Phase 3: Robustness Benchmark (`benchmark_robustness.py`)

**New file:** `receipt-parser/benchmark_robustness.py`

### 3.1 Purpose

Stress-test the full pipeline against OCR variation by:
- Bypassing OCR cache to make fresh API calls every run
- Running N iterations per fixture
- Capturing both OCR responses (call A and call B) per run
- Tracking per-field accuracy across iterations
- Attributing failures to OCR vs LLM variance
- Analyzing whether dual-call is needed

### 3.2 Instrumented OCR (monkey-patch approach)

The benchmark replaces `run_cloud_vision` with an instrumented version that:
1. **Skips** cache lookup
2. **Captures** both `fulltext1` and `fulltext2` in a collector before pick-best logic
3. **Does NOT** write to cache
4. Returns blocks normally so the rest of the pipeline works

```python
# Module-level collector, reset before each pipeline run
_ocr_collector = {
    "call_a_text": None,      # fulltext from first API call
    "call_b_text": None,      # fulltext from second API call
    "chosen_text": None,      # which was selected by pick-best
    "chose_b": False,         # True if call B won
    "chose_b_reason": None,   # "yen_symbol" | "longer" | None
}
```

**Patch target:** Since `pipeline.py` does `from ocr import ... run_cloud_vision` at
module load, we must patch the name in pipeline's namespace too:
```python
import pipeline as pipeline_mod
import ocr as ocr_mod
pipeline_mod.run_cloud_vision = instrumented_run_cloud_vision
ocr_mod.run_cloud_vision = instrumented_run_cloud_vision
```

Restore originals after each run.

### 3.3 Variance Attribution (OCR vs LLM)

This is critical. A field failure could come from:
- **OCR variance** — the fresh OCR text differs from cached, and the pipeline can't handle it
- **LLM variance** — the OCR text is the same (or equivalent), but the LLM extracts differently

**How to isolate:**

For each run, the benchmark captures:
1. The OCR text fed to the LLM (`chosen_text` from collector)
2. The LLM raw extraction (from pass history)
3. The final pipeline result (after post-processing)

**Attribution logic per failed field:**

```
For a field F that fails in run N but passed in run M:

1. Compare OCR text between run N and run M:
   - If similarity < 0.95: → "OCR_VARIANCE" (different OCR input)
   - If similarity >= 0.95: → check step 2

2. Compare LLM raw extraction (pre-post-processing) between runs:
   - If field F differs in raw LLM output: → "LLM_VARIANCE" (same input, different LLM output)
   - If field F is same in raw LLM output: → "POST_PROCESSING" (pipeline bug — same LLM output, different final result)
```

**Implementation:** Store per-run data:
```python
run_data = {
    "ocr_text": chosen_text,           # what went to LLM
    "llm_raw": pass_history[0]["extraction"],  # what LLM returned (pass 1)
    "final_result": result,            # after post-processing
}
```

Compare against a reference run (first passing run for that fixture).

**Report format per failure:**
```
receipt_8, run 3: line_items_totals FAILED
  Attribution: OCR_VARIANCE
  OCR similarity to passing run: 0.87
  Key diff: items on lines 5-8 reordered, price for コスモス split to separate line
  Saved: robustness_debug/receipt_8_run3_ocr.txt
```

### 3.4 Dual-Call Analysis

For each fixture across N runs, compute:

| Metric | How |
|---|---|
| A-vs-B similarity (per run) | `SequenceMatcher(call_a, call_b).ratio()` |
| Mean/min/max A-vs-B | Across all runs for that fixture |
| Times B chosen | Count of `chose_b == True` |
| B-chosen reason breakdown | "yen_symbol" vs "longer" counts |
| Cross-run similarity | Pairwise `SequenceMatcher` on `chosen_text` across runs |

**Recommendation logic:**
```
IF mean A-vs-B similarity > 0.99 across ALL fixtures
   AND B is never chosen (or chosen < 5% of runs):
   → "SUGGEST: Consider single-call mode (saves 50% API budget)"
   → "Run with --single-call to verify"

ELIF B is chosen > 20% of the time with meaningful improvement:
   → "KEEP: Dual-call is valuable. B chosen {N}% with mean {X}% more text"

ELSE:
   → "INCONCLUSIVE: Need more runs to determine"
```

### 3.5 Per-Field Accuracy Tracking

Reuse check functions from `benchmark_models.py` (import, don't duplicate):

```python
from benchmark_models import (
    check_total, check_date, check_currency, check_subtotal,
    check_payment_method, check_line_items_count, check_line_items_totals,
    check_tax_amount, check_merchant_similarity, check_tax_categories,
)
```

Track per fixture × per run × per field: pass/fail + attribution if failed.

**Robustness score:**
```
overall = (total field passes across all runs) / (total field checks)
e.g., 13 fixtures × 3 runs × 10 fields = 390 checks → "385/390 = 98.7%"
```

**Per-fixture robustness:**
```
"receipt_2: 10/10 fields × 3/3 runs = ROBUST"
"receipt_8: 9/10 fields in run 3 (line_items_totals: OCR_VARIANCE) = FRAGILE"
```

### 3.6 Failure Variant Capture

When a run fails fields that passed in other runs:
- Save the OCR text to `robustness_debug/{fixture}_run{N}.txt`
- Deduplicate: if similarity to an already-saved variant > 0.98, skip
- In the report, list unique failure-causing OCR variants with:
  - Which fields they broke
  - A short diff summary vs the passing variant
  - Attribution (OCR vs LLM)

### 3.7 API Budget Management

**Cost model:**
```
calls_per_run = fixtures × 2 (dual-call) + rotation_retries (~2 fixtures × 2 calls)
total_calls = calls_per_run × runs
```

| Scenario | Fixtures | Runs | Est. Calls | Budget % |
|---|---|---|---|---|
| Quick (2 fixtures) | 2 | 3 | ~18 | 1.8% |
| Default (all) | 13 | 3 | ~90 | 9% |
| Thorough | 13 | 5 | ~150 | 15% |
| Deep | 13 | 10 | ~300 | 30% |

**Safety measures:**
1. **Pre-flight check:** Calculate estimated calls, compare to `get_api_usage()["remaining"]`.
   If budget would exceed 50% of remaining, warn and require `--force`.
2. **Real-time tracking:** Print remaining budget after each fixture-run.
3. **Hard limit:** `--budget-limit N` (default: 200) stops the benchmark when reached.
4. **Rotation opt-out:** `--no-rotation` skips rotation fallback, saves ~4 calls per rotation fixture per run.

### 3.8 CLI Interface

```bash
# Default: 3 runs, all fixtures
python benchmark_robustness.py

# Quick test on 2 fixtures
python benchmark_robustness.py --runs 3 --fixtures receipt_2 receipt_8

# Thorough with budget cap
python benchmark_robustness.py --runs 5 --budget-limit 150

# Save results
python benchmark_robustness.py --runs 3 --output robustness_results.json

# Skip rotation fallback
python benchmark_robustness.py --runs 3 --no-rotation

# Resume from partial run
python benchmark_robustness.py --runs 5 --resume robustness_results.json
```

**All flags:**
| Flag | Default | Description |
|---|---|---|
| `--runs N` | 3 | Fresh OCR iterations per fixture |
| `--fixtures NAME...` | all | Specific fixtures to test |
| `--output PATH` | `robustness_results.json` | JSON output file |
| `--budget-limit N` | 200 | Max API calls before stopping |
| `--no-rotation` | false | Skip rotation fallback |
| `--model MODEL` | `qwen3.5:9b` | Ollama model |
| `--passes N` | 2 | LLM verification passes |
| `--resume PATH` | none | Resume from partial results |
| `--force` | false | Skip budget warnings |

### 3.9 Output Format

```json
{
  "metadata": {
    "timestamp": "2026-03-26T14:30:00",
    "runs_per_fixture": 3,
    "model": "qwen3.5:9b",
    "passes": 2,
    "cloud_vision_model": "builtin/stable",
    "fixtures": ["01_supermarket_receipt", "receipt_2", "..."],
    "api_calls_used": 90,
    "api_calls_remaining": 869
  },
  "per_fixture": {
    "receipt_2": {
      "runs": [
        {
          "run": 1,
          "call_a_hash": "abc123",
          "call_b_hash": "def456",
          "ab_similarity": 0.97,
          "chose_b": true,
          "chose_b_reason": "yen_symbol",
          "fields": {
            "total": { "pass": true },
            "date": { "pass": true },
            "line_items_totals": {
              "pass": false,
              "expected": [129, 328],
              "got": [129, 656],
              "attribution": "OCR_VARIANCE",
              "ocr_similarity_to_ref": 0.87
            }
          },
          "pass_count": 9,
          "total_fields": 10,
          "wall_time_s": 45.2,
          "error": null
        }
      ],
      "ocr_analysis": {
        "mean_ab_similarity": 0.98,
        "min_ab_similarity": 0.95,
        "max_ab_similarity": 1.0,
        "cross_run_similarity": { "mean": 0.96, "min": 0.91, "max": 1.0 },
        "times_b_chosen": 2,
        "times_b_chosen_pct": 66.7,
        "b_chosen_reasons": { "yen_symbol": 1, "longer": 1 }
      },
      "field_robustness": {
        "total": { "pass_rate": 1.0, "consistent": true },
        "line_items_totals": { "pass_rate": 0.67, "consistent": false,
                                "failure_attribution": { "OCR_VARIANCE": 1 } }
      },
      "robustness": "FRAGILE",
      "failure_variants_saved": 1
    }
  },
  "overall": {
    "robustness_score": 0.95,
    "robustness_summary": "95.0% (370/390) across 3 iterations for 13 fixtures",
    "perfect_fixtures": 11,
    "fragile_fixtures": ["receipt_2", "receipt_8"],
    "field_robustness": {
      "total": 1.0, "date": 1.0, "currency": 1.0, "subtotal": 1.0,
      "payment_method": 1.0, "line_items_count": 0.97,
      "line_items_totals": 0.92, "tax_amount": 0.95,
      "merchant_similarity": 1.0, "tax_categories": 0.95
    },
    "variance_attribution": {
      "OCR_VARIANCE": 15,
      "LLM_VARIANCE": 3,
      "POST_PROCESSING": 2
    },
    "dual_call_recommendation": "KEEP: B chosen 15% of the time with mean 3% longer text"
  }
}
```

### 3.10 Console Output

```
=== Robustness Benchmark ===
Model: qwen3.5:9b | Runs: 3 | Fixtures: 13 | Est. API calls: ~90

Budget check: 41/1000 used this month, 959 remaining. Proceeding.

[1/13] 01_supermarket_receipt
  Run 1: 10/10  A-B sim: 0.99  chose: A  (45.2s)
  Run 2: 10/10  A-B sim: 1.00  chose: A  (43.8s)
  Run 3: 10/10  A-B sim: 0.99  chose: A  (44.1s)
  → ROBUST (30/30)

[2/13] receipt_2
  Run 1: 10/10  A-B sim: 0.97  chose: B (yen)  (48.3s)
  Run 2:  9/10  A-B sim: 0.95  chose: B (len)  (47.1s)  ← line_items_totals [OCR_VARIANCE]
  Run 3: 10/10  A-B sim: 0.98  chose: A        (46.9s)
  → FRAGILE (29/30) — 1 OCR_VARIANCE failure saved

...

=== Summary ===
Overall: 370/390 (95.0%)
Perfect: 11/13 fixtures
Fragile: receipt_2, receipt_8

Variance Attribution:
  OCR_VARIANCE:    15 failures (75%)
  LLM_VARIANCE:     3 failures (15%)
  POST_PROCESSING:  2 failures (10%)

Dual-Call Analysis:
  Mean A-B similarity: 0.98
  B chosen: 15% of runs (reasons: yen_symbol=8, longer=4)
  → KEEP dual-call: B provides meaningful improvement

API calls used: 87 (budget: 200)
Results saved: robustness_results.json
Failure variants: robustness_debug/ (3 files)
```

---

## Phase 4: Implementation Order

```
Phase 1: builtin/stable pin (ocr.py)
  ├── No dependencies, do first
  ├── Validate: existing tests pass
  └── ~15 min

Phase 2: _ocr_source metadata (ocr.py, pipeline.py)
  ├── Depends on: Phase 1 (same file)
  ├── Validate: cli.py output shows _ocr_source
  └── ~30 min

Phase 3: benchmark_robustness.py (new file)
  ├── Depends on: Phase 1 (needs stable pin)
  ├── Can overlap with Phase 2 (different files)
  ├── Sub-steps:
  │   3a. Fixture discovery + CLI args
  │   3b. Instrumented OCR (monkey-patch)
  │   3c. Main benchmark loop
  │   3d. Field checks (import from benchmark_models.py)
  │   3e. Variance attribution (OCR vs LLM vs post-processing)
  │   3f. Text similarity + dual-call analysis
  │   3g. Failure variant capture + dedup
  │   3h. Budget management
  │   3i. Summary computation + console output
  │   3j. JSON output
  └── ~3 hours
```

---

## Potential Challenges

### 1. Monkey-patch import binding
`pipeline.py` does `from ocr import run_cloud_vision` at module load. Patching `ocr.run_cloud_vision`
after import won't affect pipeline's already-bound name. **Fix:** patch both:
```python
import pipeline as pipeline_mod
pipeline_mod.run_cloud_vision = instrumented_fn
```

### 2. Rotation fallback generates extra OCR calls
The rotation fallback at `pipeline.py:462-468` calls `run_cloud_vision` again with a rotated image.
The instrumented version will capture these too. **Fix:** track calls by image hash — the rotation
call will have a different hash. Collector stores a list of call pairs, keyed by hash.

### 3. LLM non-determinism at temperature 0
Ollama at `temperature: 0.0` should be deterministic for the same input. If the OCR text is
identical between runs but LLM output differs, this indicates an Ollama bug or quantization
artifact. The attribution system will flag this as `LLM_VARIANCE`.

### 4. `builtin/stable` region availability
If `builtin/stable` is unavailable in the user's GCP project, the API call will fail.
**Fix:** catch the error, fall back to no model pin, print a warning.

### 5. benchmark_models.py import compatibility
The robustness benchmark imports check functions from `benchmark_models.py`. If those
function signatures change, the import breaks. **Fix:** if imports fail, define local
copies of the check functions as a fallback.
