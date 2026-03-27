# LLM Speed Optimization Plan

## Current Baseline

**Hardware:** NVIDIA GeForce RTX 4070 Laptop GPU — 8188 MiB VRAM (~6968 MiB free)
**Model:** qwen3.5:9b (Q4_K_M, 6.6 GB weights)
**Current performance (from benchmarks):**

| Metric | Value |
|---|---|
| Mean eval time | 28-31s |
| Mean wall time | 43-47s |
| Tokens/sec | 21.8-22.4 |
| Pass 1 load duration | ~8.3-9.7s (model reload) |
| Pass 2 load duration | ~0.7s (model warm) |
| GPU layers | 32/33 (output layer on CPU) |
| Processor split | ~28% CPU / 72% GPU |

**Key problem:** The model doesn't fully fit in VRAM. The output layer runs on CPU,
which means a CPU<->GPU data transfer on every generated token. The ~8s "reload"
on pass 1 suggests the model is being partially re-initialized between calls.

---

## Phase 1: Zero-Risk Changes

These changes cannot affect accuracy. Implement and verify with a quick smoke test.

### 1.1 Set `keep_alive="60m"`

**File:** `extraction.py`
**What:** Add `keep_alive="60m"` to both `ollama.chat()` calls (lines 182-188 and 224-230).
**Why:** Ensures the model stays loaded between receipts during batch processing.
Default is 5 minutes, which is usually fine, but being explicit prevents edge cases.
**Risk:** None. Only affects how long the model stays in memory after the last call.
**Expected gain:** Eliminates any cold-start risk during batch processing.

### 1.2 Reduce `num_predict` from 4096 to 1024

**File:** `extraction.py`
**What:** Change `"num_predict": 4096` to `"num_predict": 1024` in both `ollama.chat()` calls.
**Why:** Receipt JSON output is typically 200-500 tokens. 4096 is wasteful headroom.
Structured output (`format=schema`) stops at the closing `}` anyway, so this is
just a safety cap.
**Risk:** None, unless a receipt generates >1024 output tokens (extremely unlikely for
our schema). Verify by checking `eval_count` in benchmark data — if max is <1024, safe.
**Pre-check:** Grep benchmark results for max `eval_count`:
```bash
python -c "import json; d=json.load(open('benchmark_results_final.json')); print(max(r.get('eval_count',0) for fix in d['results'].values() for run in fix['runs'] for r in run.get('ollama_responses',[]) if isinstance(r,dict)))"
```
**Expected gain:** Marginal — acts as a tighter safety net, no direct speed change.

### 1.3 Reduce `num_ctx` from 4096 to 2048

**File:** `extraction.py`
**What:** Add `"num_ctx": 2048` to the `options` dict in both `ollama.chat()` calls.
**Why:** Our prompts are ~1500-2000 tokens (BASE_EXTRACTION_RULES ~172 lines + OCR text).
Reducing context window from 4096 to 2048 frees ~700 MiB of KV cache VRAM.
**This is the most impactful zero-risk change** because it may free enough VRAM to
fit the 33rd layer on GPU, eliminating the CPU<->GPU bottleneck.
**Risk:** If any receipt's prompt + output exceeds 2048 tokens, Ollama truncates silently.
**Pre-check:** Measure actual token counts:
```bash
# Check prompt_eval_count (input tokens) + eval_count (output tokens) from benchmarks
python -c "import json; d=json.load(open('benchmark_results_final.json')); print(max(r.get('prompt_eval_count',0)+r.get('eval_count',0) for fix in d['results'].values() for run in fix['runs'] for r in run.get('ollama_responses',[]) if isinstance(r,dict)))"
```
If max total < 2048, it's safe. If close, use 3072 instead.
**Expected gain:** ~700 MiB freed → may push from 32/33 to 33/33 GPU layers.

### 1.1-1.3 Verification

After implementing all three:
1. Run `ollama ps` to check if processor split improved
2. Run `curl -s http://localhost:11434/api/ps` to get exact `size` vs `size_vram`
3. Run benchmark on 2-3 fixtures and compare timing to baseline

---

## Phase 2: Low-Risk Changes (Benchmark Before/After)

These changes are well-documented as safe but touch inference internals.
Implement one at a time, benchmark after each.

### 2.1 Enable Flash Attention

**How:** Set environment variable before starting Ollama:
```powershell
$env:OLLAMA_FLASH_ATTENTION="1"
```
Or add to system environment variables for persistence.
**Why:** Reduces attention memory footprint. Mathematically equivalent computation,
just uses a more memory-efficient algorithm.
**Risk:** Very low. Ollama auto-disables for incompatible models. Text-only structured
output on qwen3.5 should be fine. Rare edge cases reported with some vision models.
**Benchmark:** Run full 6-fixture benchmark, compare accuracy + timing vs baseline.
**Expected gain:** Memory savings → more room for GPU layers. Speed gain is indirect.

### 2.2 KV Cache Quantization

**How:** Set environment variable before starting Ollama:
```powershell
$env:OLLAMA_KV_CACHE_TYPE="q8_0"
```
**Why:** Cuts KV cache memory by ~50% (f16 → q8). For num_ctx=2048 on qwen3.5:9b,
this saves ~350 MiB. Combined with Phase 1.3 (reduced num_ctx), total KV savings
could be ~1 GiB compared to current setup.
**Risk:** Low but real. KV cache quantization introduces minor precision loss
(+0.002-0.05 perplexity in published benchmarks). Structured JSON output with
schema enforcement could be more sensitive than free-form text.
**Benchmark:** Run full 6-fixture benchmark. If accuracy drops on ANY field, revert.
**Expected gain:** ~350 MiB freed. Combined with 1.3 and 2.1, very likely achieves 33/33 GPU.

### 2.1-2.2 Verification

After each change:
1. `ollama ps` — check processor split
2. Run full benchmark (6 fixtures × 3 runs × 10 fields = 180 checks)
3. Compare: accuracy must stay at 100% (surgical baseline), timing should improve
4. If accuracy drops, revert that specific change

---

## Phase 3: Medium-Risk Changes (Requires Careful Testing)

### 3.1 Prompt Restructuring for Prefix Caching

**File:** `extraction.py`, `schema.py`
**What:** Split the current single user message into a system message (static rules)
and a user message (dynamic OCR text).

**Current structure:**
```python
messages=[{"role": "user", "content": full_prompt}]  # rules + OCR text combined
```

**New structure:**
```python
messages=[
    {"role": "system", "content": BASE_EXTRACTION_RULES + field_hints},
    {"role": "user", "content": f"OCR TEXT:\n{ocr_text}"},
]
```

**Why:** Ollama reuses KV cache for byte-identical prompt prefixes. If the system
message is identical across receipts (it is — only the OCR text changes), the
~1.6s prompt eval for the static prefix becomes ~0.05s after the first receipt.
**Risk:** Medium. Some models weight system vs user messages differently. The
extraction behavior could change subtly because the LLM "sees" the rules as
system instructions rather than user input.
**Benchmark:** Full benchmark + manual spot-check of edge cases (handwritten receipt,
rotated receipt, discount receipt).
**Expected gain:** ~1.5s saved per receipt after first (prompt eval: 1.6s → 0.1s).

---

## Phase 4: VRAM Reporting in Benchmarks and Tests

### 4.1 Add GPU status helper to `ocr.py`

Add a utility function that queries Ollama's `/api/ps` endpoint:

```python
def get_ollama_gpu_status() -> dict | None:
    """Query Ollama for current model GPU/VRAM status.

    Returns dict with keys:
        model, size_bytes, size_vram_bytes, gpu_percent, full_gpu, layers_gpu, layers_total
    Returns None if Ollama is not running or no model loaded.
    """
```

This function:
- Calls `GET http://localhost:11434/api/ps`
- Parses the response for `size` and `size_vram`
- Computes `gpu_percent` and `full_gpu` boolean
- Returns None gracefully if Ollama isn't running

### 4.2 Add VRAM metadata to pipeline results

**File:** `pipeline.py`
**What:** After the first LLM call completes, query GPU status and include in results.

In `_build_result`:
```python
result["_gpu_status"] = {
    "gpu_percent": 72.0,
    "full_gpu": False,
    "size_gb": 8.16,
    "vram_gb": 5.85,
}
```

This gives every pipeline result a record of whether it ran fully on GPU.

### 4.3 Add VRAM check to `benchmark_models.py`

**What:** At benchmark start, capture GPU status and include in the output JSON.

In the benchmark metadata section:
```json
{
  "metadata": {
    "gpu_status": {
      "gpu_name": "NVIDIA GeForce RTX 4070 Laptop GPU",
      "vram_total_mb": 8188,
      "vram_free_mb": 6968,
      "model_gpu_percent": 72.0,
      "model_full_gpu": false,
      "model_layers_gpu": 32,
      "model_layers_total": 33
    }
  }
}
```

Also print a warning at benchmark start if not 100% GPU:
```
WARNING: Model is 72% GPU (32/33 layers). Results may be slower than full-GPU baseline.
```

### 4.4 Add VRAM check to `benchmark_robustness.py`

Same as 4.3 — capture GPU status in metadata and warn if not 100%.

### 4.5 Add VRAM assertion to integration tests

**File:** `tests/test_integration.py`
**What:** Add an optional test that checks GPU status and warns (not fails) if the
model isn't fully on GPU.

```python
@pytest.fixture(scope="session", autouse=True)
def check_gpu_status():
    """Warn if Ollama model is not fully loaded on GPU."""
    status = get_ollama_gpu_status()
    if status and not status["full_gpu"]:
        warnings.warn(
            f"Model running at {status['gpu_percent']:.0f}% GPU. "
            f"Test timing may not reflect full-GPU performance.",
            stacklevel=2,
        )
```

This is a warning, not a failure — tests should still pass regardless of GPU status.

### 4.6 Add VRAM data to per-run timing in benchmarks

Currently benchmarks track `load_duration_ns`, `prompt_eval_duration_ns`, `eval_duration_ns`.
Add `gpu_percent` per-run so we can correlate GPU offload status with timing:

```json
{
  "run": 1,
  "fields": { ... },
  "timing": { ... },
  "gpu_percent": 100.0,
  "full_gpu": true
}
```

This lets us answer: "did the speed improvement come from the optimization or from
the model happening to load fully on GPU this time?"

---

## Implementation Order

```
Phase 1 (zero-risk, do immediately):
  1.1 keep_alive="60m"          → extraction.py (2 lines)
  1.2 num_predict: 1024         → extraction.py (2 lines, after pre-check)
  1.3 num_ctx: 2048             → extraction.py (2 lines, after pre-check)
  → Verify with ollama ps + smoke test

Phase 4.1-4.2 (VRAM reporting, needed for measurement):
  4.1 get_ollama_gpu_status()   → ocr.py (new function)
  4.2 _gpu_status in results    → pipeline.py
  → Now we can measure Phase 2 properly

Phase 2 (low-risk, one at a time):
  2.1 Flash attention           → Ollama env var + benchmark
  2.2 KV cache quantization     → Ollama env var + benchmark
  → After each: ollama ps + full benchmark

Phase 4.3-4.6 (VRAM in benchmarks/tests):
  4.3 benchmark_models.py       → add gpu_status to metadata
  4.4 benchmark_robustness.py   → add gpu_status to metadata
  4.5 test_integration.py       → add gpu warning fixture
  4.6 per-run gpu_percent       → benchmark output format
  → Verify benchmarks run and report GPU status

Phase 3 (medium-risk, last):
  3.1 Prompt restructuring      → extraction.py, schema.py + full benchmark
  → Only if Phases 1+2 don't achieve target speed
```

---

## Success Criteria

| Metric | Current | Target | How to verify |
|---|---|---|---|
| GPU layers | 32/33 | 33/33 | `ollama ps` shows `100% GPU` |
| Pass 1 load | ~8.5s | <2s | Benchmark `load_duration_ns` |
| Mean wall time | ~43-47s | <35s | Benchmark timing |
| Accuracy | 100% (surgical) | 100% | Benchmark field checks |
| VRAM in results | not tracked | always present | `_gpu_status` in pipeline output |
| VRAM in benchmarks | not tracked | always present | `gpu_status` in benchmark JSON |

---

## Risk Mitigation

- **Never implement two changes at once.** Each change gets its own benchmark run.
- **Revert immediately** if accuracy drops on any field.
- **Pre-check token counts** before reducing num_ctx or num_predict.
- **Keep the surgical benchmark (162/162 = 100%)** as the accuracy baseline.
  Any change that drops below 100% gets reverted.
- **GPU status tracking (Phase 4)** ensures we can attribute speed changes to
  VRAM improvements vs other factors.

---

## Appendix: Quick Reference Commands

```bash
# Check current GPU status
ollama ps

# Programmatic GPU check
curl -s http://localhost:11434/api/ps | python -c "
import sys, json
data = json.load(sys.stdin)
for m in data.get('models', []):
    pct = m['size_vram'] / m['size'] * 100 if m['size'] > 0 else 0
    full = 'FULL GPU' if m['size_vram'] == m['size'] else f'{pct:.0f}% GPU'
    print(f\"{m['name']}: {full} ({m['size_vram']/1024**3:.1f}/{m['size']/1024**3:.1f} GiB)\")
"

# Check VRAM with nvidia-smi
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader

# Check Ollama server logs (Windows)
Get-Content "$env:LOCALAPPDATA\Ollama\server.log" -Tail 50

# Set Ollama env vars (PowerShell, current session)
$env:OLLAMA_FLASH_ATTENTION="1"
$env:OLLAMA_KV_CACHE_TYPE="q8_0"

# Set Ollama env vars (persistent, requires Ollama restart)
[Environment]::SetEnvironmentVariable("OLLAMA_FLASH_ATTENTION", "1", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_KV_CACHE_TYPE", "q8_0", "User")
```
