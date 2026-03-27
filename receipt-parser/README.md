# Receipt Parser

A Japanese receipt parser that extracts structured financial data from photos and PDFs using Google Cloud Vision OCR and LLM inference. Supports OpenRouter (default, using DeepSeek v3.2) and local Ollama models.

## Installation

**Prerequisites:**

- Python 3.10+
- [Poppler](https://poppler.freedesktop.org/) (required for PDF support via `pdf2image`)
- An [OpenRouter API key](https://openrouter.ai/keys) (default LLM provider), **or** [Ollama](https://ollama.com/) installed and running for local inference

**Setup:**

```bash
# 1. Create or activate the conda environment
conda activate financial-aid

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenRouter API key in .env
#    (or set OPENROUTER_API_KEY in your environment)

# 4. (Optional) For local Ollama inference instead:
ollama pull qwen3.5:9b

# 5. (Optional) Install dev dependencies for tests
pip install -r requirements-dev.txt
```

## Usage

**Basic usage -- JSON to stdout:**

```bash
python cli.py receipt.jpg
```

**Multi-pass verification (re-runs extraction to correct errors):**

```bash
python cli.py receipt.jpg -p 2
```

**Debug mode -- save intermediate artifacts to `debug/`:**

```bash
python cli.py receipt.jpg --debug
```

**Batch processing -- run on an entire directory:**

```bash
python cli.py ./receipts/ -o results.json
```

**PDF input with verbose output:**

```bash
python cli.py receipt.pdf -v
```

**CSV output:**

```bash
python cli.py receipt.jpg -f csv -o output.csv
```

### CLI Options

| Flag | Long | Default | Description |
|------|------|---------|-------------|
| (positional) | | | Image, PDF, or directory to process |
| `-o` | `--output` | stdout | Output file path |
| `-m` | `--model` | `deepseek/deepseek-v3.2` | LLM model (prefix `ollama/` for local) |
| `-p` | `--passes` | `1` | Extraction passes (2+ enables verification) |
| `-f` | `--format` | `json` | Output format: `json` or `csv` |
| `-d` | `--debug` | off | Save debug artifacts to `debug/<filename>/` |
| `-v` | `--verbose` | off | Print per-pass summaries and warnings to stderr |
| | `--version` | | Show version and exit |

## Testing

There are two distinct test suites: **fixture tests** (correctness) and the **robustness benchmark** (stability under OCR variation).

### 1. Unit Tests (no API calls, ~1.5s)

Fast, offline tests for normalization, validation, schema, and mocked pipeline logic.

```bash
python -m pytest tests/test_unit.py -v
```

### 2. Integration / Fixture Tests (cached OCR, ~2-4 min)

Runs the full pipeline against 13 real receipts and checks 10 fields per receipt against ground truth. Uses cached Cloud Vision results so no API calls are made after the first run.

```bash
python -m pytest tests/test_integration.py -v
```

Each fixture is a pair of files in `tests/fixtures/`:
- `receipt_N.jpg` -- the receipt image
- `receipt_N_truth.json` -- expected extraction results

**Adding a new fixture:** drop a new image + truth JSON in `tests/fixtures/`. Tests auto-discover it.

**10 fields checked per fixture:**

| Field | Tolerance | Notes |
|-------|-----------|-------|
| `total` | exact | |
| `date` | exact | YYYY-MM-DD |
| `currency` | exact | |
| `subtotal` | exact | accepts null or subtotal==total |
| `payment_method` | exact | |
| `line_items_count` | exact | |
| `line_items_totals` | exact (sorted) | order-independent |
| `tax_amount` | +/-5 | |
| `merchant_similarity` | >=40% | fuzzy SequenceMatcher |
| `tax_categories` | exact (sorted) | per-item 8%/10%/0% |

### 3. Robustness Benchmark (fresh OCR, uses API budget)

Stress-tests the pipeline against **real Cloud Vision non-determinism** by bypassing the OCR cache and making fresh API calls every run. This answers three questions:

1. **How stable is our pipeline?** -- Do results change when OCR text varies?
2. **Where do failures come from?** -- OCR variance, LLM variance, or post-processing bugs?
3. **Is dual-call worth it?** -- Does the 2-call "pick best" OCR strategy justify its API cost?

```bash
# Default: 3 runs x all 13 fixtures (~90 API calls, ~15 min)
python benchmark_robustness.py

# Quick test on 2 fixtures (~18 API calls)
python benchmark_robustness.py --runs 3 --fixtures receipt_2 receipt_8

# Thorough with budget cap
python benchmark_robustness.py --runs 5 --budget-limit 150

# Resume an interrupted run
python benchmark_robustness.py --runs 5 --resume robustness_results.json

# Skip rotation fallback (saves ~4 API calls per rotation fixture per run)
python benchmark_robustness.py --runs 3 --no-rotation
```

**All flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--runs N` | 3 | Fresh OCR iterations per fixture |
| `--fixtures NAME...` | all | Specific fixtures to test |
| `--output PATH` | `robustness_results.json` | JSON output file |
| `--budget-limit N` | 200 | Max API calls before stopping |
| `--no-rotation` | false | Skip rotation fallback |
| `--model MODEL` | `deepseek/deepseek-v3.2` | LLM model (prefix `ollama/` for local) |
| `--passes N` | 2 | LLM verification passes |
| `--resume PATH` | none | Resume from partial results |
| `--force` | false | Skip budget warnings |

### Fixture Tests vs Robustness Benchmark

| | Fixture Tests | Robustness Benchmark |
|---|---|---|
| **Purpose** | "Does the pipeline produce correct results?" | "Does the pipeline produce *consistent* results?" |
| **OCR source** | Cached (deterministic) | Fresh API calls every run (non-deterministic) |
| **API cost** | 0 calls (after first run) | ~2-6 calls per fixture per run |
| **Runtime** | ~2-4 min | ~15-60 min depending on runs |
| **When to run** | After every code change | Before releases, after OCR/normalization changes |
| **Output** | pytest pass/fail | JSON report with attribution, LLM timing, and dual-call analysis |
| **Failure tells you** | "The pipeline is broken" | "The pipeline is fragile against OCR variation" |

### Robustness Benchmark -- Example Console Output

```
============================================================
=== Robustness Benchmark ===
Model: qwen3.5:9b | Runs: 3 | Fixtures: 6 | Passes: 2
Fixtures: ['01_supermarket_receipt', 'receipt_2', 'receipt_3', 'receipt_4', 'receipt_5', 'receipt_6']

Budget check: 41/1000 used this month, 959 remaining.
Estimated calls: ~42 (budget limit: 200)
Proceeding.

[1/6] 01_supermarket_receipt
  Run 1: 10/10  A-B sim: 0.99  chose: A  LLM: 18.2s (24 tok/s, load 0.3s)  wall: 38.2s
  Run 2: 10/10  A-B sim: 1.00  chose: A  LLM: 17.8s (25 tok/s, load 0.1s)  wall: 36.8s
  Run 3: 10/10  A-B sim: 0.99  chose: A  LLM: 18.4s (24 tok/s, load 0.1s)  wall: 37.4s
  -> ROBUST (30/30)

[2/6] receipt_2
  Run 1: 10/10  A-B sim: 0.97  chose: B (yen)  LLM: 22.3s (22 tok/s, load 0.1s)  wall: 42.3s
  Run 2:  9/10  A-B sim: 0.93  chose: B (len)  LLM: 21.1s (23 tok/s, load 0.1s)  wall: 41.1s  <- line_items_totals [OCR_VARIANCE]
  Run 3: 10/10  A-B sim: 0.98  chose: A  LLM: 20.9s (23 tok/s, load 0.1s)  wall: 40.9s
  -> FRAGILE (29/30) - 1 OCR variant(s) saved

[3/6] receipt_3
  Run 1: 10/10  A-B sim: 1.00  chose: A  LLM: 8.5s (28 tok/s, load 0.1s)  wall: 28.5s
  Run 2: 10/10  A-B sim: 1.00  chose: A  LLM: 7.9s (29 tok/s, load 0.1s)  wall: 27.9s
  Run 3: 10/10  A-B sim: 0.99  chose: A  LLM: 9.1s (27 tok/s, load 0.1s)  wall: 29.1s
  -> ROBUST (30/30)

[4/6] receipt_4
  Run 1: 10/10  A-B sim: 0.96  chose: B (yen)  LLM: 24.7s (21 tok/s, load 0.1s)  wall: 44.7s
  Run 2: 10/10  A-B sim: 0.97  chose: A  LLM: 23.2s (22 tok/s, load 0.1s)  wall: 43.2s
  Run 3:  9/10  A-B sim: 0.94  chose: B (len)  LLM: 25.8s (20 tok/s, load 0.1s)  wall: 45.8s  <- tax_amount [OCR_VARIANCE]
  -> FRAGILE (29/30) - 1 OCR variant(s) saved

[5/6] receipt_5
  Run 1: 10/10  A-B sim: 0.98  chose: A  LLM: 26.1s (21 tok/s, load 0.1s)  wall: 46.1s
  Run 2: 10/10  A-B sim: 0.99  chose: A  LLM: 24.5s (22 tok/s, load 0.1s)  wall: 44.5s
  Run 3: 10/10  A-B sim: 0.97  chose: A  LLM: 25.3s (21 tok/s, load 0.1s)  wall: 45.3s
  -> ROBUST (30/30)

[6/6] receipt_6
  Run 1: 10/10  A-B sim: 0.95  chose: B (yen)  LLM: 19.4s (23 tok/s, load 0.1s)  wall: 39.4s
  Run 2: 10/10  A-B sim: 0.98  chose: A  LLM: 18.1s (24 tok/s, load 0.1s)  wall: 38.1s
  Run 3: 10/10  A-B sim: 0.96  chose: A  LLM: 19.8s (23 tok/s, load 0.1s)  wall: 39.8s
  -> ROBUST (30/30)

============================================================
=== Summary ===
============================================================
Overall: 98.9% (178/180) across 3 iterations for 6 fixtures
Perfect: 4/6 fixtures
Fragile: receipt_2, receipt_4

Variance Attribution:
  OCR_VARIANCE             2 failures (100%)

Dual-Call Analysis:
  INCONCLUSIVE: B chosen 22.2% of runs, mean A-B similarity 0.975. Need more runs to determine.

Per-Field Robustness:
  line_items_totals          94.4%  ##################
  tax_amount                 94.4%  ##################
  currency                  100.0%  ####################
  date                      100.0%  ####################
  line_items_count          100.0%  ####################
  merchant_similarity       100.0%  ####################
  payment_method            100.0%  ####################
  subtotal                  100.0%  ####################
  tax_categories            100.0%  ####################
  total                     100.0%  ####################

LLM Performance (qwen3.5:9b):
  Mean eval time:     20.3s
  Mean load time:      0.1s
  Mean tok/s:         23.2
  Mean wall time:     40.3s
  Total tokens:      8364

API calls used this session: 39
API calls remaining: 920
Results saved: robustness_results.json
Failure variants: robustness_debug/ (2 files)
```

### Robustness Benchmark -- Example JSON Output

<details>
<summary>Click to expand full JSON example</summary>

```json
{
  "metadata": {
    "timestamp": "2026-03-26T14:30:00",
    "runs_per_fixture": 3,
    "model": "qwen3.5:9b",
    "passes": 2,
    "cloud_vision_model": "builtin/stable",
    "fixtures": [
      "01_supermarket_receipt",
      "receipt_2",
      "receipt_3",
      "receipt_4",
      "receipt_5",
      "receipt_6"
    ],
    "no_rotation": false,
    "api_calls_used": 39,
    "api_calls_remaining": 920,
    "output_path": "robustness_results.json"
  },
  "per_fixture": {
    "01_supermarket_receipt": {
      "runs": [
        {
          "run": 1,
          "pass_count": 10,
          "total_fields": 10,
          "wall_time_s": 38.2,
          "llm_timing": {
            "passes": 2,
            "total_duration_s": 19.1,
            "load_s": 0.3,
            "prompt_eval_s": 0.6,
            "eval_s": 18.2,
            "tokens_generated": 437,
            "prompt_tokens": 1842,
            "tokens_per_second": 24.0
          },
          "error": null,
          "fields": {
            "total": { "pass": true, "detail": "got 725, expected 725" },
            "date": { "pass": true, "detail": "got 2026-03-19, expected 2026-03-19" },
            "currency": { "pass": true, "detail": "got JPY, expected JPY" },
            "subtotal": { "pass": true, "detail": "got 671, expected 671" },
            "payment_method": { "pass": true, "detail": "got cash, expected cash" },
            "line_items_count": { "pass": true, "detail": "got 2, expected 2" },
            "line_items_totals": { "pass": true, "detail": "got [323, 348], expected [323, 348]" },
            "tax_amount": { "pass": true, "detail": "got 54, expected 54 (tol +-5)" },
            "merchant_similarity": { "pass": true, "detail": "'サンリブ' vs 'サンリブ' (100%)" },
            "tax_categories": { "pass": true, "detail": "got ['10%', '8%'], expected ['10%', '8%']" }
          },
          "ocr_data": {
            "call_a_hash": "8a3f2b1c9d4e",
            "call_b_hash": "7b2e1a0d8c3f",
            "chose_b": false,
            "chose_b_reason": null,
            "ab_similarity": 0.9934
          },
          "final_result_summary": {
            "total": 725,
            "date": "2026-03-19",
            "merchant": "サンリブ",
            "subtotal": 671,
            "currency": "JPY",
            "payment_method": "cash",
            "line_items_count": 2,
            "tax_sum": 54
          }
        }
      ],
      "ocr_analysis": {
        "mean_ab_similarity": 0.9949,
        "min_ab_similarity": 0.9912,
        "max_ab_similarity": 1.0,
        "times_b_chosen": 0,
        "times_b_chosen_pct": 0.0,
        "b_chosen_reasons": {},
        "cross_run_similarity": { "mean": 0.98, "min": 0.97, "max": 0.99 }
      },
      "field_robustness": {
        "total": { "pass_rate": 1.0, "consistent": true },
        "date": { "pass_rate": 1.0, "consistent": true },
        "line_items_totals": { "pass_rate": 1.0, "consistent": true }
      },
      "robustness": "ROBUST",
      "failure_variants_saved": 0
    },
    "receipt_2": {
      "runs": [
        {
          "run": 1,
          "pass_count": 10,
          "total_fields": 10,
          "wall_time_s": 42.3,
          "llm_timing": {
            "passes": 2,
            "total_duration_s": 23.1,
            "load_s": 0.1,
            "prompt_eval_s": 0.8,
            "eval_s": 22.3,
            "tokens_generated": 491,
            "prompt_tokens": 2105,
            "tokens_per_second": 22.0
          },
          "error": null,
          "fields": {
            "total": { "pass": true, "detail": "got 2447, expected 2447" },
            "line_items_totals": { "pass": true, "detail": "got [128, 193, 298, 348, 1298], expected [128, 193, 298, 348, 1298]" }
          },
          "ocr_data": {
            "call_a_hash": "a1b2c3d4e5f6",
            "call_b_hash": "f6e5d4c3b2a1",
            "chose_b": true,
            "chose_b_reason": "yen_symbol",
            "ab_similarity": 0.9703
          }
        },
        {
          "run": 2,
          "pass_count": 9,
          "total_fields": 10,
          "wall_time_s": 41.1,
          "llm_timing": {
            "passes": 2,
            "total_duration_s": 21.9,
            "load_s": 0.1,
            "prompt_eval_s": 0.7,
            "eval_s": 21.1,
            "tokens_generated": 485,
            "prompt_tokens": 2098,
            "tokens_per_second": 23.0
          },
          "error": null,
          "fields": {
            "total": { "pass": true, "detail": "got 2447, expected 2447" },
            "line_items_totals": {
              "pass": false,
              "detail": "got [128, 193, 298, 348, 1196], expected [128, 193, 298, 348, 1298]",
              "attribution": "OCR_VARIANCE",
              "ocr_similarity_to_ref": 0.8734
            }
          },
          "ocr_data": {
            "call_a_hash": "b2c3d4e5f6a1",
            "call_b_hash": "e5d4c3b2a1f6",
            "chose_b": true,
            "chose_b_reason": "longer",
            "ab_similarity": 0.9289
          }
        }
      ],
      "ocr_analysis": {
        "mean_ab_similarity": 0.9601,
        "min_ab_similarity": 0.9289,
        "max_ab_similarity": 0.9812,
        "times_b_chosen": 2,
        "times_b_chosen_pct": 66.7,
        "b_chosen_reasons": { "yen_symbol": 1, "longer": 1 },
        "cross_run_similarity": { "mean": 0.94, "min": 0.89, "max": 0.98 }
      },
      "field_robustness": {
        "line_items_totals": {
          "pass_rate": 0.6667,
          "consistent": false,
          "failure_attribution": { "OCR_VARIANCE": 1 }
        }
      },
      "robustness": "FRAGILE",
      "failure_variants_saved": 1
    }
  },
  "overall": {
    "robustness_score": 0.9889,
    "robustness_summary": "98.9% (178/180) across 3 iterations for 6 fixtures",
    "perfect_fixtures": 4,
    "perfect_fixture_names": ["01_supermarket_receipt", "receipt_3", "receipt_5", "receipt_6"],
    "fragile_fixtures": ["receipt_2", "receipt_4"],
    "field_robustness": {
      "total": 1.0,
      "date": 1.0,
      "currency": 1.0,
      "subtotal": 1.0,
      "payment_method": 1.0,
      "line_items_count": 1.0,
      "line_items_totals": 0.9444,
      "tax_amount": 0.9444,
      "merchant_similarity": 1.0,
      "tax_categories": 1.0
    },
    "variance_attribution": {
      "OCR_VARIANCE": 2
    },
    "dual_call_recommendation": "INCONCLUSIVE: B chosen 22.2% of runs, mean A-B similarity 0.975. Need more runs to determine.",
    "llm_timing_summary": {
      "mean_eval_s": 20.3,
      "mean_load_s": 0.1,
      "mean_tps": 23.2,
      "mean_wall_s": 40.3,
      "total_tokens": 8364
    }
  }
}
```

</details>

### Reading the Results

**Fixture-level robustness:**
- `ROBUST` -- all fields passed on all runs. The pipeline handles OCR variation for this receipt.
- `FRAGILE` -- at least one field failed on at least one run. Check `failure_attribution` to see why.

**Variance attribution (per failed field):**
- `OCR_VARIANCE` -- Cloud Vision returned different text, and the pipeline couldn't compensate. Fix: harden normalization/post-processing for the affected pattern.
- `LLM_VARIANCE` -- OCR text was nearly identical but the LLM extracted differently. Fix: add a post-processing rule instead of relying on the LLM.
- `POST_PROCESSING` -- Both OCR and LLM output matched, but pipeline post-processing produced different results. Fix: debug the post-processing step.

**Dual-call recommendation:**
- `SUGGEST` -- A and B are always nearly identical, B is rarely chosen. Consider switching to single-call to save 50% API budget.
- `KEEP` -- B is chosen frequently with meaningful improvement (e.g., B has the yen symbol when A doesn't).
- `INCONCLUSIVE` -- Not enough data to decide. Run more iterations.

**LLM timing:**
- `eval_s` -- time spent generating tokens (the actual inference). This is the main LLM cost.
- `load_s` -- time loading the model into memory. High on first run, near-zero after (model stays resident).
- `prompt_eval_s` -- time processing the input prompt. Scales with OCR text length.
- `tokens_per_second` -- generation throughput. Compare across models to pick the speed/accuracy tradeoff.
- `wall_time_s` -- total end-to-end time including OCR API calls, normalization, and post-processing.

**Failure variant files** in `robustness_debug/` contain the raw OCR text from failing runs. Compare them against passing runs to see exactly what changed in the OCR output.

## Debug Mode

When `--debug` is passed, the pipeline writes intermediate artifacts to `debug/<input_stem>/`. These files let you trace exactly where a parsing error originates.

### Artifacts

| File | Contents |
|------|----------|
| `01_original.png` | The raw input image as loaded |
| `02_preprocessed.png` | After grayscale conversion, deskew, and contrast normalization |
| `03_ocr_bboxes.png` | Bounding boxes drawn on the preprocessed image, color-coded by OCR confidence: green (>= 90%), yellow (>= 70%), red (< 70%) |
| `04_ocr_grouped.txt` | OCR text after spatial grouping into lines |
| `05_pass1_llm_response.json` | Raw structured output from the first LLM extraction pass |
| `06_pass1_warnings.txt` | Validation warnings from pass 1 (arithmetic mismatches, etc.) |
| `10_field_overlay.png` | Extracted fields mapped back to their OCR bounding boxes, color-coded per field with a legend |
| `pipeline_trace.txt` | Step-by-step timing for the entire pipeline |

### Diagnostic Workflow

When a result looks wrong, work backwards through the artifacts:

1. **Check `10_field_overlay.png`** -- Are fields mapped to the correct regions of the receipt? If a field points to the wrong text, the problem is in LLM extraction.
2. **Check `05_pass1_llm_response.json`** -- Does the raw LLM output contain the correct values? If not, the LLM misinterpreted the OCR text.
3. **Check `03_ocr_bboxes.png` and `04_ocr_grouped.txt`** -- Is the OCR text accurate? Red bounding boxes indicate low-confidence detections. If text is garbled, the problem is upstream of the LLM.
4. **Check `02_preprocessed.png`** -- Is the image clean enough for OCR? If it is heavily skewed, low-contrast, or blurry, preprocessing may need adjustment.

## Extending the Schema

All extraction fields are defined in a single registry. Adding a new field requires no prompt editing and no changes to the debug overlay logic.

### Step 1: Add a `FieldMeta` entry to `FIELD_REGISTRY` in `schema.py`

```python
FieldMeta(
    name="tip",
    debug_color_bgr=(0, 200, 100),
    prompt_hint="Look for tip, gratuity, or service charge amounts.",
    extraction_aliases=["tip", "gratuity", "チップ"],
),
```

### Step 2: Add the field to the `Receipt` Pydantic model in `schema.py`

```python
class Receipt(BaseModel):
    # ... existing fields ...
    tip: Optional[float] = None
```

### Step 3 (optional): Add validation logic in `validation.py`

```python
# Example: check that tip + subtotal + tax ~ total
if receipt.tip is not None and receipt.subtotal is not None:
    expected = receipt.subtotal + receipt.tip + tax_sum
    if abs(expected - receipt.total) > 2:
        warnings.append(f"Total does not match subtotal + tip + taxes")
```

### What auto-updates

Once the field is in `FIELD_REGISTRY` and the `Receipt` model, the following adapt automatically:

- **LLM prompt hints** -- `generate_extraction_prompt()` includes the new field's `prompt_hint` and aliases
- **LLM schema enforcement** -- the Pydantic model is used as the structured output format
- **Debug overlay** -- `draw_field_overlay()` picks up the new field's color from the registry
- **JSON/CSV output** -- Pydantic serialization includes the new field in all output

## Model Selection

The `--model` flag accepts any OpenRouter model name by default. Prefix with `ollama/` for local Ollama models.

**OpenRouter models (default):**

| Model | JP Quality | Speed | Use Case |
|-------|------------|-------|----------|
| `deepseek/deepseek-v3.2` | Excellent | Fast | Default -- best cost/quality tradeoff |

**Local Ollama models** (prefix with `ollama/`):

| Model | VRAM | JP Quality | Speed | Use Case |
|-------|------|------------|-------|----------|
| `ollama/qwen3.5:9b` | ~5 GB | Excellent | ~22 tok/s | Best local accuracy (100% on benchmark) |
| `ollama/qwen3:8b` | ~5 GB | Very Good | ~41 tok/s | Best local speed/accuracy tradeoff |
| `ollama/gemma3:4b` | ~3 GB | Good | ~35 tok/s | Low-VRAM machines |

```bash
# OpenRouter (default)
python cli.py receipt.jpg

# Local Ollama
python cli.py receipt.jpg -m ollama/qwen3.5:9b
```
