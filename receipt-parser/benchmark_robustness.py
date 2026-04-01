"""benchmark_robustness.py -- Stress-test pipeline against OCR variation.

Bypasses OCR cache to make fresh Cloud Vision API calls every run,
uses confidence-based retry (single-call default, retry if confidence < 0.75),
tracks per-field accuracy, and attributes failures to OCR vs LLM variance.

Usage:
    python benchmark_robustness.py
    python benchmark_robustness.py --runs 4 --fixtures receipt_2 receipt_8
    python benchmark_robustness.py --runs 4 --budget-limit 200
    python benchmark_robustness.py --resume robustness_results.json
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import cv2
import numpy as np

import ocr as ocr_mod
import pipeline as pipeline_mod
from ocr import (
    init_cloud_vision, _call_cloud_vision, _extract_fulltext_from_response,
    _extract_blocks_from_response, _fulltext_to_blocks, _pick_better_fulltext,
    compute_ocr_confidence, _OCR_CONFIDENCE_RETRY_THRESHOLD,
    get_api_usage, get_ollama_gpu_status,
)
from benchmark_models import (
    discover_fixtures, FIELD_CHECKS,
    check_total, check_date, check_currency, check_subtotal,
    check_payment_method, check_line_items_count, check_line_items_totals,
    check_tax_amount, check_merchant_similarity, check_tax_categories,
    check_item_descriptions,
)
import extraction
from extraction import check_model_available

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEBUG_DIR = Path(__file__).parent / "robustness_debug"
DEFAULT_OUTPUT = Path(__file__).parent / "robustness_results.json"
DEFAULT_MODEL = extraction.DEFAULT_MODEL  # deepseek-chat
DEFAULT_BUDGET_LIMIT = 200

# ---------------------------------------------------------------------------
# LLM timing instrumentation — capture durations per call (Ollama + API)
# ---------------------------------------------------------------------------

_timing_collector: list[dict] = []
_original_chat_with_timeout = extraction._ollama_chat_with_timeout
_original_openrouter_chat = extraction._openrouter_chat


def _instrumented_chat_with_timeout(timeout: int = extraction.OLLAMA_TIMEOUT_SECONDS, **kwargs):
    """Wrapper that captures Ollama timing metadata from each LLM call."""
    response = _original_chat_with_timeout(timeout=timeout, **kwargs)
    _timing_collector.append({
        "total_duration_ns": response.get("total_duration"),
        "load_duration_ns": response.get("load_duration"),
        "prompt_eval_count": response.get("prompt_eval_count"),
        "prompt_eval_duration_ns": response.get("prompt_eval_duration"),
        "eval_count": response.get("eval_count"),
        "eval_duration_ns": response.get("eval_duration"),
        "backend": "ollama",
    })
    return response


def _instrumented_openrouter_chat(model, messages, temperature=0.0, max_tokens=4096):
    """Wrapper that captures API call wall time and token usage."""
    client = extraction._get_api_client()
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=temperature,
        max_tokens=max_tokens,
        seed=extraction._LLM_SEED,
    )
    elapsed_ns = int((time.perf_counter() - t0) * 1e9)
    usage = response.usage
    _timing_collector.append({
        "total_duration_ns": elapsed_ns,
        "load_duration_ns": 0,
        "prompt_eval_count": usage.prompt_tokens if usage else None,
        "prompt_eval_duration_ns": None,
        "eval_count": usage.completion_tokens if usage else None,
        "eval_duration_ns": elapsed_ns,
        "input_tokens": usage.prompt_tokens if usage else None,
        "output_tokens": usage.completion_tokens if usage else None,
        "backend": "api",
    })
    return response.choices[0].message.content


def _install_llm_instrumentation():
    extraction._ollama_chat_with_timeout = _instrumented_chat_with_timeout
    extraction._openrouter_chat = _instrumented_openrouter_chat


def _restore_llm_instrumentation():
    extraction._ollama_chat_with_timeout = _original_chat_with_timeout
    extraction._openrouter_chat = _original_openrouter_chat


# DeepSeek pricing (per million tokens)
_DEEPSEEK_INPUT_COST_PER_M = 0.27
_DEEPSEEK_OUTPUT_COST_PER_M = 1.10


def _aggregate_timing(entries: list[dict]) -> dict:
    """Aggregate LLM timing entries into a summary dict with token usage and cost."""
    total_duration_ns = sum(t.get("total_duration_ns") or 0 for t in entries)
    load_ns = sum(t.get("load_duration_ns") or 0 for t in entries)
    eval_ns = sum(t.get("eval_duration_ns") or 0 for t in entries)
    prompt_ns = sum(t.get("prompt_eval_duration_ns") or 0 for t in entries)
    total_tokens = sum(t.get("eval_count") or 0 for t in entries)
    total_prompt_tokens = sum(t.get("prompt_eval_count") or 0 for t in entries)
    input_tokens = sum(t.get("input_tokens") or 0 for t in entries)
    output_tokens = sum(t.get("output_tokens") or 0 for t in entries)
    cost_usd = (input_tokens * _DEEPSEEK_INPUT_COST_PER_M +
                output_tokens * _DEEPSEEK_OUTPUT_COST_PER_M) / 1_000_000
    return {
        "passes": len(entries),
        "total_duration_s": total_duration_ns / 1e9,
        "load_s": load_ns / 1e9,
        "prompt_eval_s": prompt_ns / 1e9,
        "eval_s": eval_ns / 1e9,
        "tokens_generated": total_tokens,
        "prompt_tokens": total_prompt_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost_usd, 6),
        "tokens_per_second": total_tokens / (eval_ns / 1e9) if eval_ns else 0,
        "per_pass": entries,
    }


# ---------------------------------------------------------------------------
# Instrumented OCR — monkey-patch approach
# ---------------------------------------------------------------------------

# Module-level collector, reset before each pipeline run
_ocr_collector: dict = {}


def _reset_collector():
    """Reset the OCR collector before each pipeline run."""
    global _ocr_collector
    _ocr_collector = {
        "call_1_text": None,
        "call_1_confidence": None,
        "call_2_text": None,
        "retried": False,
        "retry_reason": None,
        "chosen_text": None,
    }


# Track total API calls made by the benchmark
_benchmark_api_calls = 0


def _instrumented_run_cloud_vision(image: np.ndarray, client=None) -> list[dict]:
    """Replacement for run_cloud_vision that skips cache.

    Uses confidence-based retry: single call by default, retry only if
    OCR confidence < threshold (matching pipeline v3.0.0 behavior).
    """
    global _ocr_collector, _benchmark_api_calls

    if client is None:
        client = init_cloud_vision()

    # Call 1
    response1 = _call_cloud_vision(image, client)
    fulltext1 = _extract_fulltext_from_response(response1)
    _benchmark_api_calls += 1

    if not fulltext1:
        _ocr_collector["call_1_text"] = ""
        _ocr_collector["chosen_text"] = ""
        return []

    # Compute confidence from first call
    blocks1 = _extract_blocks_from_response(response1)
    confidence1 = compute_ocr_confidence(blocks1) if blocks1 else 0.0

    _ocr_collector["call_1_text"] = fulltext1
    _ocr_collector["call_1_confidence"] = round(confidence1, 4)

    fulltext = fulltext1

    # Confidence-based retry: only call again if quality is low
    if confidence1 < _OCR_CONFIDENCE_RETRY_THRESHOLD:
        response2 = _call_cloud_vision(image, client)
        fulltext2 = _extract_fulltext_from_response(response2)
        _benchmark_api_calls += 1

        _ocr_collector["retried"] = True
        _ocr_collector["retry_reason"] = f"confidence {confidence1:.3f} < {_OCR_CONFIDENCE_RETRY_THRESHOLD}"
        _ocr_collector["call_2_text"] = fulltext2 or ""

        if fulltext2:
            fulltext = _pick_better_fulltext(fulltext1, fulltext2)

    _ocr_collector["chosen_text"] = fulltext

    # Set OCR source metadata (the real module's global)
    ocr_mod._last_ocr_source = "fresh"

    # Return blocks WITHOUT writing to cache
    return _fulltext_to_blocks(fulltext)


# Store originals for restore
_original_ocr_run = ocr_mod.run_cloud_vision
_original_pipeline_run = pipeline_mod.run_cloud_vision


def _install_instrumented_ocr():
    """Monkey-patch run_cloud_vision in both modules."""
    ocr_mod.run_cloud_vision = _instrumented_run_cloud_vision
    pipeline_mod.run_cloud_vision = _instrumented_run_cloud_vision


def _restore_original_ocr():
    """Restore original run_cloud_vision."""
    ocr_mod.run_cloud_vision = _original_ocr_run
    pipeline_mod.run_cloud_vision = _original_pipeline_run


# ---------------------------------------------------------------------------
# Variance attribution
# ---------------------------------------------------------------------------

def _text_similarity(a: str, b: str) -> float:
    """Compute SequenceMatcher similarity ratio between two texts."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _attribute_failure(
    failed_field: str,
    failed_run: dict,
    ref_run: dict,
) -> str:
    """Attribute a field failure to OCR_VARIANCE, LLM_VARIANCE, or POST_PROCESSING.

    Compare against a reference (passing) run:
    1. If OCR text similarity < 0.95 → OCR_VARIANCE
    2. If LLM raw extraction differs for that field → LLM_VARIANCE
    3. Otherwise → POST_PROCESSING
    """
    ocr_sim = _text_similarity(
        failed_run.get("ocr_text", ""),
        ref_run.get("ocr_text", ""),
    )

    if ocr_sim < 0.95:
        return "OCR_VARIANCE"

    # Compare LLM raw extraction for the specific field
    failed_llm = failed_run.get("llm_raw", {})
    ref_llm = ref_run.get("llm_raw", {})

    # Map field check names to result dict keys
    field_key_map = {
        "total": "total",
        "date": "date",
        "currency": "currency",
        "subtotal": "subtotal",
        "payment_method": "payment_method",
        "line_items_count": "line_items",
        "line_items_totals": "line_items",
        "tax_amount": "taxes",
        "merchant_similarity": "merchant",
        "tax_categories": "line_items",
    }

    key = field_key_map.get(failed_field, failed_field)
    failed_val = failed_llm.get(key)
    ref_val = ref_llm.get(key)

    if json.dumps(failed_val, sort_keys=True, default=str) != \
       json.dumps(ref_val, sort_keys=True, default=str):
        return "LLM_VARIANCE"

    return "POST_PROCESSING"


# ---------------------------------------------------------------------------
# Budget management
# ---------------------------------------------------------------------------

def _estimate_api_calls(n_fixtures: int, n_runs: int, no_rotation: bool) -> int:
    """Estimate total API calls for the benchmark run.

    Single-call default with confidence-based retry (~10% of images need retry).
    """
    calls_per_fixture_per_run = 1  # single-call default
    retry_estimate = max(1, int(n_fixtures * 0.10))  # ~10% need retry
    if not no_rotation:
        rotation_extras = max(1, int(n_fixtures * 0.15))
    else:
        rotation_extras = 0
    return (n_fixtures * calls_per_fixture_per_run + retry_estimate + rotation_extras) * n_runs


def _check_budget(estimated_calls: int, budget_limit: int, force: bool) -> bool:
    """Pre-flight budget check. Returns True if OK to proceed."""
    usage = get_api_usage()
    remaining = usage["remaining"]

    print(f"\nBudget check: {usage['calls']}/{usage['free_limit']} used this month, "
          f"{remaining} remaining.")
    print(f"Estimated calls: ~{estimated_calls} (budget limit: {budget_limit})")

    if estimated_calls > budget_limit:
        print(f"WARNING: Estimated calls ({estimated_calls}) exceed budget limit ({budget_limit}).")
        if not force:
            print("Use --force to override, or reduce --runs / --fixtures.")
            return False

    if estimated_calls > remaining * 0.5:
        print(f"WARNING: Estimated calls ({estimated_calls}) would use >{50}% of remaining "
              f"free tier ({remaining}).")
        if not force:
            print("Use --force to override.")
            return False

    print("Proceeding.\n")
    return True


# ---------------------------------------------------------------------------
# Failure variant capture
# ---------------------------------------------------------------------------

def _save_failure_variant(
    fixture_name: str,
    run_idx: int,
    ocr_text: str,
    saved_variants: dict[str, list[str]],
) -> bool:
    """Save OCR text for a failing run if it's sufficiently different from existing variants.

    Returns True if saved, False if deduplicated away.
    """
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    existing = saved_variants.get(fixture_name, [])
    for existing_text in existing:
        if _text_similarity(ocr_text, existing_text) > 0.98:
            return False  # Too similar, skip

    filename = f"{fixture_name}_run{run_idx}.txt"
    (DEBUG_DIR / filename).write_text(ocr_text, encoding="utf-8")
    saved_variants.setdefault(fixture_name, []).append(ocr_text)
    return True


# ---------------------------------------------------------------------------
# OCR confidence & retry analysis
# ---------------------------------------------------------------------------

def _compute_ocr_analysis(fixture_runs: list[dict]) -> dict:
    """Compute OCR confidence and retry statistics for a fixture's runs."""
    confidences = []
    retried_count = 0

    for run in fixture_runs:
        ocr_data = run.get("ocr_data", {})
        conf = ocr_data.get("call_1_confidence")
        if conf is not None:
            confidences.append(conf)
        if ocr_data.get("retried", False):
            retried_count += 1

    n = len(fixture_runs) or 1
    return {
        "mean_confidence": sum(confidences) / len(confidences) if confidences else 0,
        "min_confidence": min(confidences) if confidences else 0,
        "max_confidence": max(confidences) if confidences else 0,
        "retried_count": retried_count,
        "retried_pct": round(retried_count / n * 100, 1),
    }


def _compute_cross_run_similarity(fixture_runs: list[dict]) -> dict:
    """Compute pairwise similarity of chosen OCR text across runs."""
    texts = [r.get("ocr_text", "") for r in fixture_runs]
    sims = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sims.append(_text_similarity(texts[i], texts[j]))
    return {
        "mean": sum(sims) / len(sims) if sims else 1.0,
        "min": min(sims) if sims else 1.0,
        "max": max(sims) if sims else 1.0,
    }


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def _compute_overall_summary(per_fixture: dict, metadata: dict) -> dict:
    """Compute overall robustness summary across all fixtures."""
    total_field_checks = 0
    total_field_passes = 0
    field_pass_counts: dict[str, int] = defaultdict(int)
    field_total_counts: dict[str, int] = defaultdict(int)
    variance_attribution: dict[str, int] = defaultdict(int)
    perfect_fixtures = []
    fragile_fixtures = []

    # OCR confidence & retry aggregation
    all_confidences = []
    total_retried = 0
    total_runs = 0

    for fixture_name, fdata in per_fixture.items():
        runs = fdata.get("runs", [])
        total_runs += len(runs)

        fixture_all_pass = True
        for run in runs:
            fields = run.get("fields", {})
            for field_name, field_result in fields.items():
                total_field_checks += 1
                field_total_counts[field_name] += 1
                if field_result.get("pass"):
                    total_field_passes += 1
                    field_pass_counts[field_name] += 1
                else:
                    fixture_all_pass = False
                    attr = field_result.get("attribution")
                    if attr:
                        variance_attribution[attr] += 1

        if fixture_all_pass:
            perfect_fixtures.append(fixture_name)
        else:
            fragile_fixtures.append(fixture_name)

        # OCR confidence & retry
        ocr_analysis = fdata.get("ocr_analysis", {})
        if ocr_analysis.get("mean_confidence") is not None:
            all_confidences.append(ocr_analysis["mean_confidence"])
        total_retried += ocr_analysis.get("retried_count", 0)

    # Per-field robustness
    field_robustness = {}
    for field_name in FIELD_CHECKS:
        total = field_total_counts.get(field_name, 0)
        passes = field_pass_counts.get(field_name, 0)
        field_robustness[field_name] = round(passes / total, 4) if total else 1.0

    # OCR retry summary
    mean_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 1.0
    retry_pct = (total_retried / total_runs * 100) if total_runs else 0
    ocr_retry_summary = (
        f"Mean OCR confidence: {mean_conf:.3f}. "
        f"Retried {total_retried}/{total_runs} runs ({retry_pct:.1f}%). "
        f"Single-call mode saves ~{100 - retry_pct:.0f}% API budget vs dual-call."
    )

    score = total_field_passes / total_field_checks if total_field_checks else 1.0

    # LLM timing & token aggregation
    all_eval_s = []
    all_load_s = []
    all_tps = []
    all_wall_s = []
    total_tokens = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    for fdata in per_fixture.values():
        for run in fdata.get("runs", []):
            timing = run.get("llm_timing", {})
            if timing.get("eval_s"):
                all_eval_s.append(timing["eval_s"])
            if timing.get("load_s") is not None:
                all_load_s.append(timing["load_s"])
            if timing.get("tokens_per_second"):
                all_tps.append(timing["tokens_per_second"])
            if run.get("wall_time_s"):
                all_wall_s.append(run["wall_time_s"])
            total_tokens += timing.get("tokens_generated", 0)
            total_input_tokens += timing.get("input_tokens", 0)
            total_output_tokens += timing.get("output_tokens", 0)
            total_cost += timing.get("cost_usd", 0)

    n_runs = total_runs or 1
    llm_timing_summary = {
        "mean_eval_s": sum(all_eval_s) / len(all_eval_s) if all_eval_s else 0,
        "mean_load_s": sum(all_load_s) / len(all_load_s) if all_load_s else 0,
        "mean_tps": sum(all_tps) / len(all_tps) if all_tps else 0,
        "mean_wall_s": sum(all_wall_s) / len(all_wall_s) if all_wall_s else 0,
        "total_tokens": total_tokens,
    }

    # Token & cost summary
    cost_summary = {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cost_usd": round(total_cost, 4),
        "cost_per_receipt_usd": round(total_cost / n_runs, 6) if n_runs else 0,
        "avg_input_tokens_per_receipt": round(total_input_tokens / n_runs),
        "avg_output_tokens_per_receipt": round(total_output_tokens / n_runs),
    }

    # Override summary
    total_overrides = 0
    override_by_field: dict[str, int] = defaultdict(int)
    for fdata in per_fixture.values():
        oa = fdata.get("override_analysis", {})
        total_overrides += oa.get("total_overrides", 0)
        for field, count in oa.get("by_field", {}).items():
            override_by_field[field] += count
    override_summary = {
        "total_overrides": total_overrides,
        "overrides_per_receipt": round(total_overrides / n_runs, 2),
        "by_field": dict(override_by_field),
    }

    # Determinism summary
    deterministic_count = sum(
        1 for fdata in per_fixture.values()
        if fdata.get("determinism", {}).get("is_deterministic", False)
    )
    determinism_summary = {
        "deterministic_fixtures": deterministic_count,
        "total_fixtures": len(per_fixture),
        "rate": round(deterministic_count / len(per_fixture), 4) if per_fixture else 1.0,
    }

    # Warning summary
    total_warnings = sum(
        fdata.get("warning_analysis", {}).get("total_warnings", 0)
        for fdata in per_fixture.values()
    )
    clean_runs_total = sum(
        fdata.get("warning_analysis", {}).get("clean_runs", 0)
        for fdata in per_fixture.values()
    )
    warning_summary = {
        "total_warnings": total_warnings,
        "warnings_per_receipt": round(total_warnings / n_runs, 2),
        "clean_runs": clean_runs_total,
        "clean_rate": round(clean_runs_total / n_runs, 4) if n_runs else 1.0,
    }

    return {
        "robustness_score": round(score, 4),
        "robustness_summary": (
            f"{score:.1%} ({total_field_passes}/{total_field_checks}) across "
            f"{metadata.get('runs_per_fixture', '?')} iterations for "
            f"{len(per_fixture)} fixtures"
        ),
        "perfect_fixtures": len(perfect_fixtures),
        "perfect_fixture_names": perfect_fixtures,
        "fragile_fixtures": fragile_fixtures,
        "field_robustness": field_robustness,
        "variance_attribution": dict(variance_attribution),
        "ocr_retry_summary": ocr_retry_summary,
        "llm_timing_summary": llm_timing_summary,
        "cost_summary": cost_summary,
        "override_summary": override_summary,
        "determinism_summary": determinism_summary,
        "warning_summary": warning_summary,
    }


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def _print_run_line(fixture_idx: int, n_fixtures: int, fixture_name: str,
                    run_idx: int, run_data: dict):
    """Print a single run result line."""
    pc = run_data["pass_count"]
    tf = run_data["total_fields"]
    ocr_data = run_data.get("ocr_data", {})
    conf = ocr_data.get("call_1_confidence", 0)
    retried = "retry" if ocr_data.get("retried") else "1-call"
    wall = run_data.get("wall_time_s", 0)

    # LLM timing
    timing = run_data.get("llm_timing", {})
    eval_s = timing.get("eval_s", 0)

    # Find failed fields
    failed = [f for f, r in run_data.get("fields", {}).items() if not r.get("pass")]
    fail_str = ""
    if failed:
        attrs = [run_data["fields"][f].get("attribution", "?") for f in failed]
        fail_str = f"  <- {', '.join(failed)} [{', '.join(attrs)}]"

    status = f"{pc}/{tf}" if not run_data.get("error") else "ERROR"
    print(f"  Run {run_idx}: {status:5s}  OCR conf: {conf:.2f} ({retried})  "
          f"LLM: {eval_s:.1f}s  wall: {wall:.1f}s{fail_str}")


def _print_summary(overall: dict, metadata: dict):
    """Print final summary to console."""
    print(f"\n{'=' * 70}")
    print("=== Summary ===")
    print(f"{'=' * 70}")
    print(f"Overall: {overall['robustness_summary']}")
    print(f"Perfect: {overall['perfect_fixtures']}/{overall['perfect_fixtures'] + len(overall['fragile_fixtures'])} fixtures")
    if overall["fragile_fixtures"]:
        print(f"Fragile: {', '.join(overall['fragile_fixtures'])}")

    print(f"\nVariance Attribution:")
    total_failures = sum(overall["variance_attribution"].values())
    if total_failures:
        for attr, count in sorted(overall["variance_attribution"].items()):
            pct = count / total_failures * 100
            print(f"  {attr:20s} {count:3d} failures ({pct:.0f}%)")
    else:
        print("  No failures to attribute.")

    print(f"\nOCR Confidence & Retry:")
    print(f"  {overall.get('ocr_retry_summary', 'N/A')}")

    print(f"\nPer-Field Robustness:")
    for field, score in sorted(overall["field_robustness"].items(),
                               key=lambda x: x[1]):
        bar = "#" * int(score * 20)
        print(f"  {field:25s} {score:6.1%}  {bar}")

    # LLM timing summary
    llm_summary = overall.get("llm_timing_summary", {})
    if llm_summary:
        print(f"\nLLM Performance ({metadata.get('model', '?')}):")
        print(f"  Mean eval time:   {llm_summary.get('mean_eval_s', 0):6.1f}s")
        print(f"  Mean wall time:   {llm_summary.get('mean_wall_s', 0):6.1f}s")

    # Token usage & cost
    cost = overall.get("cost_summary", {})
    if cost.get("total_input_tokens"):
        print(f"\nToken Usage & Cost:")
        print(f"  Avg input tokens:   {cost.get('avg_input_tokens_per_receipt', 0):,}")
        print(f"  Avg output tokens:  {cost.get('avg_output_tokens_per_receipt', 0):,}")
        print(f"  Cost per receipt:   ${cost.get('cost_per_receipt_usd', 0):.4f}")
        print(f"  Total cost:         ${cost.get('total_cost_usd', 0):.4f}")

    # Override analysis
    overrides = overall.get("override_summary", {})
    print(f"\nPipeline Override Analysis:")
    print(f"  Overrides per receipt: {overrides.get('overrides_per_receipt', 0):.1f}")
    if overrides.get("by_field"):
        for field, count in sorted(overrides["by_field"].items(),
                                   key=lambda x: -x[1]):
            print(f"    {field:20s} {count:3d} overrides")
    else:
        print(f"  No overrides — LLM output used as-is.")

    # Determinism
    det = overall.get("determinism_summary", {})
    print(f"\nDeterminism:")
    print(f"  {det.get('deterministic_fixtures', 0)}/{det.get('total_fixtures', 0)} "
          f"fixtures produce identical output across runs ({det.get('rate', 0):.0%})")

    # Warnings
    warns = overall.get("warning_summary", {})
    print(f"\nValidation Warnings:")
    print(f"  Clean runs (0 warnings): {warns.get('clean_runs', 0)}/{metadata.get('runs_per_fixture', 0) * len(overall.get('perfect_fixture_names', []) + overall.get('fragile_fixtures', []))}"
          f" ({warns.get('clean_rate', 0):.0%})")
    print(f"  Avg warnings/receipt:    {warns.get('warnings_per_receipt', 0):.1f}")

    usage = get_api_usage()
    print(f"\nAPI Budget:")
    print(f"  Calls this session: {metadata.get('api_calls_used', '?')}")
    print(f"  Remaining:          {usage['remaining']}")
    print(f"\nResults saved: {metadata.get('output_path', '?')}")

    variant_count = sum(1 for f in DEBUG_DIR.glob("*.txt")) if DEBUG_DIR.exists() else 0
    if variant_count:
        print(f"Failure variants: {DEBUG_DIR}/ ({variant_count} files)")


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def _load_resume(resume_path: Path | None) -> tuple[dict, set[tuple[str, int]]]:
    """Load partial results for resume. Returns (results_dict, completed_keys)."""
    if not resume_path or not resume_path.exists():
        return {}, set()
    data = json.loads(resume_path.read_text(encoding="utf-8"))
    completed = set()
    for fixture_name, fdata in data.get("per_fixture", {}).items():
        for run in fdata.get("runs", []):
            completed.add((fixture_name, run["run"]))
    return data, completed


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_robustness_benchmark(
    runs: int = 3,
    fixture_names: list[str] | None = None,
    output_path: Path = DEFAULT_OUTPUT,
    budget_limit: int = DEFAULT_BUDGET_LIMIT,
    no_rotation: bool = False,
    model: str = DEFAULT_MODEL,
    passes: int = 2,
    resume_path: Path | None = None,
    force: bool = False,
) -> dict:
    """Main robustness benchmark entry point."""
    global _benchmark_api_calls

    # Discover fixtures
    fixtures = discover_fixtures(fixture_names)
    if not fixtures:
        print("No fixtures found. Exiting.")
        sys.exit(1)

    n_fixtures = len(fixtures)
    fixture_name_list = [f[0] for f in fixtures]

    print(f"{'=' * 60}")
    print(f"=== Robustness Benchmark ===")
    print(f"Model: {model} | Runs: {runs} | Fixtures: {n_fixtures} | "
          f"Passes: {passes}")
    print(f"Fixtures: {fixture_name_list}")

    # Preflight: Ollama
    check_model_available(model)

    # Preflight: Cloud Vision
    try:
        cv_client = init_cloud_vision()
    except Exception as e:
        print(f"ERROR: Cloud Vision init failed: {e}")
        sys.exit(1)

    # Budget check
    estimated = _estimate_api_calls(n_fixtures, runs, no_rotation)
    if not _check_budget(estimated, budget_limit, force):
        sys.exit(1)

    # Resume
    prev_results, completed_keys = _load_resume(resume_path)

    # Initialize results structure
    # Capture GPU status
    gpu_status = get_ollama_gpu_status()
    if gpu_status and not gpu_status["full_gpu"]:
        print(f"WARNING: Model is {gpu_status['gpu_percent']:.0f}% GPU "
              f"({gpu_status['vram_gb']}/{gpu_status['size_gb']} GiB VRAM). "
              f"Results may be slower than full-GPU baseline.")

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "runs_per_fixture": runs,
        "model": model,
        "passes": passes,
        "cloud_vision_model": "builtin/stable",
        "fixtures": fixture_name_list,
        "no_rotation": no_rotation,
        "api_calls_used": 0,
        "api_calls_remaining": get_api_usage()["remaining"],
        "output_path": str(output_path),
        "gpu_status": gpu_status,
    }

    per_fixture: dict = prev_results.get("per_fixture", {})

    # Saved failure variants for deduplication
    saved_variants: dict[str, list[str]] = {}

    # Install instrumented OCR + LLM timing
    _install_instrumented_ocr()
    _install_llm_instrumentation()
    _benchmark_api_calls = 0

    try:
        for fix_idx, (fixture_name, fixture_image, fixture_truth) in enumerate(fixtures):
            print(f"\n[{fix_idx + 1}/{n_fixtures}] {fixture_name}")

            if fixture_name not in per_fixture:
                per_fixture[fixture_name] = {"runs": []}

            fixture_runs = per_fixture[fixture_name]["runs"]

            for run_idx in range(1, runs + 1):
                if (fixture_name, run_idx) in completed_keys:
                    print(f"  Run {run_idx}: (resumed, skipping)")
                    continue

                # Check budget limit
                if _benchmark_api_calls >= budget_limit:
                    print(f"\nBudget limit reached ({budget_limit} calls). Stopping.")
                    raise _BudgetExceeded()

                # Reset collectors
                _reset_collector()
                _timing_collector.clear()

                # Run pipeline
                wall_start = time.perf_counter()
                error = None
                result = {}
                try:
                    result = pipeline_mod.process_document(
                        fixture_image, model=model, passes=passes,
                        apply_user_rules=False,
                    )
                except Exception as e:
                    error = str(e)
                wall_time = time.perf_counter() - wall_start

                # Capture LLM timing + GPU status
                llm_timing = _aggregate_timing(list(_timing_collector))
                run_gpu = get_ollama_gpu_status()

                # Capture OCR data from collector
                ocr_data = dict(_ocr_collector)

                # Evaluate fields
                field_results = {}
                for field_name, check_fn in FIELD_CHECKS.items():
                    field_results[field_name] = check_fn(result, fixture_truth)

                pass_count = sum(1 for f in field_results.values() if f["pass"])
                total_fields = len(field_results)

                # Compute LLM vs pipeline override tracking
                llm_raw_extraction = (
                    result.get("_pass_history", [{}])[0].get("extraction", {})
                    if result.get("_pass_history") else {}
                )
                final_summary = {
                    "total": result.get("total"),
                    "date": result.get("date"),
                    "merchant": result.get("merchant"),
                    "subtotal": result.get("subtotal"),
                    "currency": result.get("currency"),
                    "payment_method": result.get("payment_method"),
                    "line_items_count": len(result.get("line_items", [])),
                    "tax_sum": sum(
                        t.get("amount", 0) for t in result.get("taxes", [])
                    ),
                }
                override_fields = []
                for key in ("total", "subtotal", "payment_method", "date"):
                    llm_val = llm_raw_extraction.get(key)
                    final_val = final_summary.get(key)
                    if llm_val != final_val and final_val is not None:
                        override_fields.append(key)
                llm_tax_sum = sum(
                    t.get("amount", 0) for t in llm_raw_extraction.get("taxes", [])
                    if isinstance(t, dict)
                )
                if abs(llm_tax_sum - final_summary["tax_sum"]) > 1:
                    override_fields.append("taxes")
                llm_items_count = len(llm_raw_extraction.get("line_items", []))
                if llm_items_count != final_summary["line_items_count"]:
                    override_fields.append("line_items")

                # Build run record
                run_record = {
                    "run": run_idx,
                    "pass_count": pass_count,
                    "total_fields": total_fields,
                    "wall_time_s": round(wall_time, 2),
                    "llm_timing": llm_timing,
                    "gpu_percent": run_gpu["gpu_percent"] if run_gpu else None,
                    "full_gpu": run_gpu["full_gpu"] if run_gpu else None,
                    "error": error,
                    "fields": field_results,
                    "ocr_data": {
                        "call_1_text": ocr_data.get("call_1_text", ""),
                        "call_1_confidence": ocr_data.get("call_1_confidence"),
                        "call_2_text": ocr_data.get("call_2_text", ""),
                        "retried": ocr_data.get("retried", False),
                        "retry_reason": ocr_data.get("retry_reason"),
                        "chosen_text": ocr_data.get("chosen_text", ""),
                    },
                    "ocr_text": ocr_data.get("chosen_text", ""),
                    "llm_raw": llm_raw_extraction,
                    "final_result_summary": final_summary,
                    "confidence": {
                        "ocr": result.get("_ocr_confidence"),
                        "llm": result.get("_llm_confidence"),
                    },
                    "warnings": result.get("_warnings", []),
                    "warning_count": len(result.get("_warnings", [])),
                    "overrides": override_fields,
                    "override_count": len(override_fields),
                    "llm_passes_used": result.get("_pass_count", 1),
                }

                fixture_runs.append(run_record)
                _print_run_line(fix_idx, n_fixtures, fixture_name, run_idx, run_record)

                # Print budget status
                usage = get_api_usage()
                remaining = usage["remaining"]
                if remaining <= 100:
                    print(f"    [Budget: {_benchmark_api_calls} benchmark calls, "
                          f"{remaining} free tier remaining]")

            # After all runs for this fixture: compute attribution and analysis
            _finalize_fixture(fixture_name, per_fixture[fixture_name], saved_variants)

            # Save progress after each fixture
            results = _assemble_results(metadata, per_fixture)
            _save_results(results, output_path)

    except (_BudgetExceeded, KeyboardInterrupt):
        print("\nStopping early. Saving partial results...")
    finally:
        _restore_original_ocr()
        _restore_llm_instrumentation()

    # Final save
    metadata["api_calls_used"] = _benchmark_api_calls
    metadata["api_calls_remaining"] = get_api_usage()["remaining"]
    results = _assemble_results(metadata, per_fixture)
    _save_results(results, output_path)

    # Print summary
    _print_summary(results["overall"], metadata)

    return results


class _BudgetExceeded(Exception):
    pass


def _finalize_fixture(fixture_name: str, fdata: dict, saved_variants: dict):
    """Compute attribution, dual-call analysis, and field robustness for a fixture."""
    runs = fdata["runs"]
    if not runs:
        return

    # Find a reference (passing) run — first run where all fields pass
    ref_run = None
    for r in runs:
        if r["pass_count"] == r["total_fields"]:
            ref_run = r
            break
    # If no perfect run, use the best one
    if ref_run is None:
        ref_run = max(runs, key=lambda r: r["pass_count"])

    # Attribution for failed fields
    for run in runs:
        for field_name, field_result in run["fields"].items():
            if not field_result["pass"] and ref_run["fields"][field_name]["pass"]:
                attr = _attribute_failure(field_name, run, ref_run)
                field_result["attribution"] = attr
                field_result["ocr_similarity_to_ref"] = round(
                    _text_similarity(
                        run.get("ocr_text", ""),
                        ref_run.get("ocr_text", ""),
                    ), 4
                )

    # Save failure variants
    variants_saved = 0
    for run in runs:
        has_failure = any(not f["pass"] for f in run["fields"].values())
        if has_failure and run.get("ocr_text"):
            if _save_failure_variant(
                fixture_name, run["run"], run["ocr_text"], saved_variants
            ):
                variants_saved += 1

    # OCR confidence & retry analysis
    fdata["ocr_analysis"] = _compute_ocr_analysis(runs)
    fdata["ocr_analysis"]["cross_run_similarity"] = _compute_cross_run_similarity(runs)

    # Per-field robustness
    field_robustness = {}
    failure_attribution: dict[str, dict[str, int]] = {}
    for field_name in FIELD_CHECKS:
        passes = sum(1 for r in runs if r["fields"][field_name]["pass"])
        total = len(runs)
        consistent = all(r["fields"][field_name]["pass"] for r in runs) or \
                     not any(r["fields"][field_name]["pass"] for r in runs)
        field_robustness[field_name] = {
            "pass_rate": round(passes / total, 4) if total else 1.0,
            "consistent": consistent,
        }
        # Failure attributions for this field
        attrs: dict[str, int] = defaultdict(int)
        for r in runs:
            attr = r["fields"][field_name].get("attribution")
            if attr:
                attrs[attr] += 1
        if attrs:
            field_robustness[field_name]["failure_attribution"] = dict(attrs)

    fdata["field_robustness"] = field_robustness
    fdata["failure_variants_saved"] = variants_saved

    # Determinism score: are all runs producing identical output?
    summaries = [json.dumps(r.get("final_result_summary", {}), sort_keys=True)
                 for r in runs]
    unique_outputs = len(set(summaries))
    fdata["determinism"] = {
        "unique_outputs": unique_outputs,
        "is_deterministic": unique_outputs == 1,
        "score": round(1.0 / unique_outputs, 4) if unique_outputs else 1.0,
    }

    # Override analysis
    all_overrides = [o for r in runs for o in r.get("overrides", [])]
    override_counts: dict[str, int] = defaultdict(int)
    for o in all_overrides:
        override_counts[o] += 1
    fdata["override_analysis"] = {
        "total_overrides": len(all_overrides),
        "overrides_per_run": round(len(all_overrides) / len(runs), 2) if runs else 0,
        "by_field": dict(override_counts),
    }

    # Warning analysis
    all_warnings = [w for r in runs for w in r.get("warnings", [])]
    fdata["warning_analysis"] = {
        "total_warnings": len(all_warnings),
        "warnings_per_run": round(len(all_warnings) / len(runs), 2) if runs else 0,
        "clean_runs": sum(1 for r in runs if r.get("warning_count", 0) == 0),
        "clean_rate": round(
            sum(1 for r in runs if r.get("warning_count", 0) == 0) / len(runs), 4
        ) if runs else 1.0,
    }

    # Confidence summary
    ocr_confs = [r.get("confidence", {}).get("ocr") for r in runs
                 if r.get("confidence", {}).get("ocr") is not None]
    fdata["confidence_summary"] = {
        "mean_ocr": round(sum(ocr_confs) / len(ocr_confs), 4) if ocr_confs else None,
    }

    # Overall fixture status
    all_pass = all(
        all(f["pass"] for f in r["fields"].values())
        for r in runs
    )
    fdata["robustness"] = "ROBUST" if all_pass else "FRAGILE"

    total_pass = sum(r["pass_count"] for r in runs)
    total_checks = sum(r["total_fields"] for r in runs)
    det_tag = "" if fdata["determinism"]["is_deterministic"] else " NON-DET"
    status = f"ROBUST ({total_pass}/{total_checks}){det_tag}" if all_pass else \
             f"FRAGILE ({total_pass}/{total_checks}){det_tag}"
    if variants_saved:
        status += f" - {variants_saved} OCR variant(s) saved"
    print(f"  -> {status}")


def _assemble_results(metadata: dict, per_fixture: dict) -> dict:
    """Assemble the full results dict with overall summary."""
    # Strip large OCR text from output to keep JSON manageable
    per_fixture_clean = {}
    for fname, fdata in per_fixture.items():
        clean_fdata = dict(fdata)
        clean_runs = []
        for run in fdata.get("runs", []):
            clean_run = dict(run)
            # Keep ocr_data but truncate raw text fields for JSON output
            if "ocr_data" in clean_run:
                od = dict(clean_run["ocr_data"])
                od["call_1_hash"] = str(hash(od.get("call_1_text", "")))[:12]
                od.pop("call_1_text", None)
                od.pop("call_2_text", None)
                od.pop("chosen_text", None)
                clean_run["ocr_data"] = od
            # Remove full OCR text and LLM raw from JSON (saved in debug dir)
            clean_run.pop("ocr_text", None)
            clean_run.pop("llm_raw", None)
            # Strip verbose per-pass timing (keep aggregated summary)
            if "llm_timing" in clean_run:
                clean_timing = dict(clean_run["llm_timing"])
                clean_timing.pop("per_pass", None)
                clean_run["llm_timing"] = clean_timing
            clean_runs.append(clean_run)
        clean_fdata["runs"] = clean_runs
        per_fixture_clean[fname] = clean_fdata

    overall = _compute_overall_summary(per_fixture, metadata)

    return {
        "metadata": metadata,
        "per_fixture": per_fixture_clean,
        "overall": overall,
    }


def _save_results(results: dict, output_path: Path):
    """Save results to JSON."""
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Historical comparison
# ---------------------------------------------------------------------------

def _compare_results(current: dict, previous: dict):
    """Compare current benchmark results against a previous run and print deltas."""
    curr_overall = current.get("overall", {})
    prev_overall = previous.get("overall", {})

    print(f"\n{'=' * 70}")
    print("=== Comparison vs Previous Benchmark ===")
    print(f"{'=' * 70}")
    print(f"Previous: {previous.get('metadata', {}).get('timestamp', '?')}")
    print(f"Current:  {current.get('metadata', {}).get('timestamp', '?')}")

    # Accuracy delta
    curr_score = curr_overall.get("robustness_score", 0)
    prev_score = prev_overall.get("robustness_score", 0)
    delta = curr_score - prev_score
    arrow = "+" if delta > 0 else "" if delta == 0 else ""
    print(f"\nAccuracy: {prev_score:.1%} -> {curr_score:.1%} ({arrow}{delta:+.1%})")

    # Per-field comparison
    curr_fields = curr_overall.get("field_robustness", {})
    prev_fields = prev_overall.get("field_robustness", {})
    all_fields = sorted(set(list(curr_fields.keys()) + list(prev_fields.keys())))
    changed_fields = []
    for field in all_fields:
        curr_rate = curr_fields.get(field, 0)
        prev_rate = prev_fields.get(field, 0)
        if curr_rate != prev_rate:
            changed_fields.append((field, prev_rate, curr_rate))
    if changed_fields:
        print(f"\nField changes:")
        for field, prev_rate, curr_rate in changed_fields:
            delta = curr_rate - prev_rate
            print(f"  {field:25s} {prev_rate:6.1%} -> {curr_rate:6.1%} ({delta:+.1%})")
    else:
        print(f"\nNo per-field accuracy changes.")

    # Speed delta
    curr_timing = curr_overall.get("llm_timing_summary", {})
    prev_timing = prev_overall.get("llm_timing_summary", {})
    curr_wall = curr_timing.get("mean_wall_s", 0)
    prev_wall = prev_timing.get("mean_wall_s", 0)
    if prev_wall > 0:
        speed_change = (curr_wall - prev_wall) / prev_wall * 100
        print(f"\nSpeed: {prev_wall:.1f}s -> {curr_wall:.1f}s ({speed_change:+.1f}%)")

    # Cost delta
    curr_cost = curr_overall.get("cost_summary", {})
    prev_cost = prev_overall.get("cost_summary", {})
    curr_cpr = curr_cost.get("cost_per_receipt_usd", 0)
    prev_cpr = prev_cost.get("cost_per_receipt_usd", 0)
    if prev_cpr > 0:
        cost_change = (curr_cpr - prev_cpr) / prev_cpr * 100
        print(f"Cost:  ${prev_cpr:.4f} -> ${curr_cpr:.4f} per receipt ({cost_change:+.1f}%)")

    # Determinism delta
    curr_det = curr_overall.get("determinism_summary", {}).get("rate", 0)
    prev_det = prev_overall.get("determinism_summary", {}).get("rate", 0)
    if curr_det != prev_det:
        print(f"Determinism: {prev_det:.0%} -> {curr_det:.0%}")

    # Fixtures that changed status
    curr_fragile = set(curr_overall.get("fragile_fixtures", []))
    prev_fragile = set(prev_overall.get("fragile_fixtures", []))
    fixed = prev_fragile - curr_fragile
    regressed = curr_fragile - prev_fragile
    if fixed:
        print(f"\nFixed (were fragile, now robust): {', '.join(sorted(fixed))}")
    if regressed:
        print(f"Regressed (were robust, now fragile): {', '.join(sorted(regressed))}")
    if not fixed and not regressed:
        print(f"\nNo fixture status changes.")

    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Robustness benchmark: stress-test pipeline against OCR variation"
    )
    parser.add_argument("--runs", type=int, default=3,
                        help="Fresh OCR iterations per fixture (default: 3)")
    parser.add_argument("--fixtures", nargs="+", default=None,
                        help="Specific fixture names to test (default: all)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help=f"JSON output file (default: {DEFAULT_OUTPUT.name})")
    parser.add_argument("--budget-limit", type=int, default=DEFAULT_BUDGET_LIMIT,
                        help=f"Max API calls before stopping (default: {DEFAULT_BUDGET_LIMIT})")
    parser.add_argument("--no-rotation", action="store_true",
                        help="Skip rotation fallback (saves ~4 calls per rotation fixture per run)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"LLM model (default: {DEFAULT_MODEL})")
    parser.add_argument("--passes", type=int, default=2,
                        help="LLM verification passes (default: 2)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from partial results JSON")
    parser.add_argument("--compare", type=str, default=None,
                        help="Compare results against a previous benchmark JSON")
    parser.add_argument("--force", action="store_true",
                        help="Skip budget warnings")
    args = parser.parse_args()

    results = run_robustness_benchmark(
        runs=args.runs,
        fixture_names=args.fixtures,
        output_path=Path(args.output),
        budget_limit=args.budget_limit,
        no_rotation=args.no_rotation,
        model=args.model,
        passes=args.passes,
        resume_path=Path(args.resume) if args.resume else None,
        force=args.force,
    )

    if args.compare:
        compare_path = Path(args.compare)
        if compare_path.exists():
            previous = json.loads(compare_path.read_text(encoding="utf-8"))
            _compare_results(results, previous)
        else:
            print(f"Comparison file not found: {compare_path}")


if __name__ == "__main__":
    main()
