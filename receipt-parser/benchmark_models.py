"""benchmark_models.py -- Compare LLM models on receipt extraction accuracy and speed.

Usage:
    conda run -n financial-aid python benchmark_models.py --pull --runs 3
    conda run -n financial-aid python benchmark_models.py --models qwen3.5:4b --fixtures 01_supermarket_receipt --runs 1
"""

import argparse
import json
import platform
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import ollama as ollama_client
import extraction
from ocr import get_ollama_gpu_status
from pipeline import process_document

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCHMARK_MODELS = [
    "qwen3:8b",
    "qwen3.5:4b",
    "gemma3:4b",
    "gemma3:12b",
    "qwen2.5:7b",
]

FIXTURES_DIR = Path(__file__).parent / "tests" / "fixtures"

# ---------------------------------------------------------------------------
# Fixture discovery (mirrors test_integration.py)
# ---------------------------------------------------------------------------

def discover_fixtures(names: list[str] | None = None) -> list[tuple[str, Path, dict]]:
    fixtures = []
    for truth_file in sorted(FIXTURES_DIR.glob("*_truth.json")):
        if truth_file.name == "_truth_template.json":
            continue
        base = truth_file.stem.replace("_truth", "")
        if names and base not in names:
            continue
        image = None
        for ext in (".jpg", ".jpeg", ".png", ".pdf", ".tiff", ".bmp"):
            candidate = FIXTURES_DIR / f"{base}{ext}"
            if candidate.exists():
                image = candidate
                break
        if image is None:
            continue
        truth = json.loads(truth_file.read_text(encoding="utf-8"))
        fixtures.append((base, image, truth))
    return fixtures

# ---------------------------------------------------------------------------
# 9 field checks (same logic as test_integration.py)
# ---------------------------------------------------------------------------

def check_total(result: dict, truth: dict) -> dict:
    got, exp = result.get("total"), truth.get("total")
    ok = got == exp
    return {"pass": ok, "detail": f"got {got}, expected {exp}"}


def check_date(result: dict, truth: dict) -> dict:
    got, exp = result.get("date"), truth.get("date")
    ok = got == exp
    return {"pass": ok, "detail": f"got {got}, expected {exp}"}


def check_currency(result: dict, truth: dict) -> dict:
    got, exp = result.get("currency"), truth.get("currency")
    ok = got == exp
    return {"pass": ok, "detail": f"got {got}, expected {exp}"}


def check_subtotal(result: dict, truth: dict) -> dict:
    got = result.get("subtotal")
    exp = truth.get("subtotal")
    if exp is None:
        ok = got is None or got == result.get("total")
        return {"pass": ok, "detail": f"got {got}, expected None or {result.get('total')}"}
    ok = got == exp
    return {"pass": ok, "detail": f"got {got}, expected {exp}"}


def check_payment_method(result: dict, truth: dict) -> dict:
    got, exp = result.get("payment_method"), truth.get("payment_method")
    ok = got == exp
    return {"pass": ok, "detail": f"got {got}, expected {exp}"}


def check_line_items_count(result: dict, truth: dict) -> dict:
    got = len(result.get("line_items", []))
    exp = len(truth.get("line_items", []))
    ok = got == exp
    return {"pass": ok, "detail": f"got {got}, expected {exp}"}


def check_line_items_totals(result: dict, truth: dict) -> dict:
    got = sorted(i.get("total", 0) for i in result.get("line_items", []))
    exp = sorted(i.get("total", 0) for i in truth.get("line_items", []))
    ok = got == exp
    return {"pass": ok, "detail": f"got {got}, expected {exp}"}


def check_tax_amount(result: dict, truth: dict) -> dict:
    got = sum(t.get("amount", 0) for t in result.get("taxes", []))
    exp = sum(t.get("amount", 0) for t in truth.get("taxes", []))
    ok = abs(got - exp) < 5
    return {"pass": ok, "detail": f"got {got}, expected {exp} (tol +-5)"}


_KATAKANA_MAP = {
    'ア': 'a', 'イ': 'i', 'ウ': 'u', 'エ': 'e', 'オ': 'o',
    'カ': 'ka', 'キ': 'ki', 'ク': 'ku', 'ケ': 'ke', 'コ': 'ko',
    'サ': 'sa', 'シ': 'shi', 'ス': 'su', 'セ': 'se', 'ソ': 'so',
    'タ': 'ta', 'チ': 'chi', 'ツ': 'tsu', 'テ': 'te', 'ト': 'to',
    'ナ': 'na', 'ニ': 'ni', 'ヌ': 'nu', 'ネ': 'ne', 'ノ': 'no',
    'ハ': 'ha', 'ヒ': 'hi', 'フ': 'fu', 'ヘ': 'he', 'ホ': 'ho',
    'マ': 'ma', 'ミ': 'mi', 'ム': 'mu', 'メ': 'me', 'モ': 'mo',
    'ヤ': 'ya', 'ユ': 'yu', 'ヨ': 'yo',
    'ラ': 'ra', 'リ': 'ri', 'ル': 'ru', 'レ': 're', 'ロ': 'ro',
    'ワ': 'wa', 'ヲ': 'wo', 'ン': 'n',
    'ガ': 'ga', 'ギ': 'gi', 'グ': 'gu', 'ゲ': 'ge', 'ゴ': 'go',
    'ザ': 'za', 'ジ': 'ji', 'ズ': 'zu', 'ゼ': 'ze', 'ゾ': 'zo',
    'ダ': 'da', 'ヂ': 'di', 'ヅ': 'du', 'デ': 'de', 'ド': 'do',
    'バ': 'ba', 'ビ': 'bi', 'ブ': 'bu', 'ベ': 'be', 'ボ': 'bo',
    'パ': 'pa', 'ピ': 'pi', 'プ': 'pu', 'ペ': 'pe', 'ポ': 'po',
    'ッ': '', 'ー': '', 'ャ': 'ya', 'ュ': 'yu', 'ョ': 'yo',
    'ァ': 'a', 'ィ': 'i', 'ゥ': 'u', 'ェ': 'e', 'ォ': 'o',
}


def _merchant_similarity(pred: str, truth: str) -> float:
    """Compare merchant names with cross-script fallback."""
    ratio = SequenceMatcher(None, pred, truth).ratio()
    if ratio >= 0.4:
        return ratio
    pred_r = ''.join(_KATAKANA_MAP.get(c, c) for c in pred).lower()
    truth_r = ''.join(_KATAKANA_MAP.get(c, c) for c in truth).lower()
    return max(ratio, SequenceMatcher(None, pred_r, truth_r).ratio())


def check_merchant_similarity(result: dict, truth: dict) -> dict:
    got = result.get("merchant") or ""
    exp = truth.get("merchant") or ""
    if not exp:
        return {"pass": True, "detail": "no merchant in truth, skipped"}
    ratio = _merchant_similarity(got, exp)
    ok = ratio >= 0.4
    return {"pass": ok, "detail": f"'{got}' vs '{exp}' ({ratio:.0%})"}


def check_tax_categories(result: dict, truth: dict) -> dict:
    true_cats = sorted(
        i.get("tax_category", "0%") for i in truth.get("line_items", [])
    )
    if not true_cats:
        return {"pass": True, "detail": "no line items in truth, skipped"}
    pred_cats = sorted(
        i.get("tax_category", "0%") for i in result.get("line_items", [])
    )
    ok = pred_cats == true_cats
    return {"pass": ok, "detail": f"got {pred_cats}, expected {true_cats}"}


FIELD_CHECKS = {
    "total": check_total,
    "date": check_date,
    "currency": check_currency,
    "subtotal": check_subtotal,
    "payment_method": check_payment_method,
    "line_items_count": check_line_items_count,
    "line_items_totals": check_line_items_totals,
    "tax_amount": check_tax_amount,
    "merchant_similarity": check_merchant_similarity,
    "tax_categories": check_tax_categories,
}

# ---------------------------------------------------------------------------
# Instrumentation -- capture Ollama timing without modifying extraction.py
# ---------------------------------------------------------------------------

_timing_collector: list[dict] = []
_original_chat_with_timeout = extraction._ollama_chat_with_timeout


def _instrumented_chat_with_timeout(timeout: int = extraction.OLLAMA_TIMEOUT_SECONDS, **kwargs):
    response = _original_chat_with_timeout(timeout=timeout, **kwargs)
    _timing_collector.append({
        "total_duration_ns": response.get("total_duration"),
        "load_duration_ns": response.get("load_duration"),
        "prompt_eval_count": response.get("prompt_eval_count"),
        "prompt_eval_duration_ns": response.get("prompt_eval_duration"),
        "eval_count": response.get("eval_count"),
        "eval_duration_ns": response.get("eval_duration"),
    })
    return response


def install_instrumentation():
    extraction._ollama_chat_with_timeout = _instrumented_chat_with_timeout


def restore_instrumentation():
    extraction._ollama_chat_with_timeout = _original_chat_with_timeout

# ---------------------------------------------------------------------------
# Ollama cache management
# ---------------------------------------------------------------------------

def unload_model(model: str):
    """Unload model from Ollama to clear KV cache."""
    try:
        ollama_client.generate(model=model, prompt="", keep_alive=0)
    except Exception:
        pass  # Model may not be loaded yet


def aggregate_timing(entries: list[dict]) -> dict:
    total_eval_ns = sum(t.get("eval_duration_ns") or 0 for t in entries)
    total_prompt_ns = sum(t.get("prompt_eval_duration_ns") or 0 for t in entries)
    total_tokens = sum(t.get("eval_count") or 0 for t in entries)
    total_prompt_tokens = sum(t.get("prompt_eval_count") or 0 for t in entries)
    return {
        "passes": len(entries),
        "total_eval_s": total_eval_ns / 1e9,
        "total_prompt_eval_s": total_prompt_ns / 1e9,
        "total_tokens_generated": total_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "tokens_per_second": total_tokens / (total_eval_ns / 1e9) if total_eval_ns else 0,
        "per_pass": entries,
    }

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def preflight_checks(models: list[str], pull_missing: bool) -> list[str]:
    try:
        available = ollama_client.list()
    except Exception:
        print("ERROR: Ollama is not running. Start with: ollama serve")
        sys.exit(1)

    available_names = [m.model or "" for m in available.models] if hasattr(available, "models") else []
    ready = []
    for model in models:
        if any(model in m for m in available_names):
            ready.append(model)
        elif pull_missing:
            print(f"  Pulling {model} ...")
            try:
                ollama_client.pull(model)
                ready.append(model)
                print(f"  Pulled {model}")
            except Exception as e:
                print(f"  FAILED to pull {model}: {e}")
        else:
            print(f"  WARNING: {model} not installed (use --pull). Skipping.")
    return ready

# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def compute_consistency(model_runs: list[dict]) -> dict:
    by_fixture: dict[str, list[dict]] = defaultdict(list)
    for r in model_runs:
        by_fixture[r["fixture"]].append(r)
    consistent = 0
    total = 0
    for fixture_runs in by_fixture.values():
        for field_name in FIELD_CHECKS:
            results = [r["fields"][field_name]["pass"] for r in fixture_runs]
            total += 1
            if all(results) or not any(results):
                consistent += 1
    return {
        "consistent_fields": consistent,
        "total_fields": total,
        "consistency_rate": consistent / total if total else 0,
    }


def compute_summary(runs: list[dict]) -> dict:
    summary = {}
    for model in sorted(set(r["model"] for r in runs)):
        model_runs = [r for r in runs if r["model"] == model]
        total_checks = sum(r["total_fields"] for r in model_runs)
        total_passes = sum(r["pass_count"] for r in model_runs)

        field_stats = {}
        for field_name in FIELD_CHECKS:
            passes = sum(1 for r in model_runs if r["fields"][field_name]["pass"])
            field_stats[field_name] = {
                "pass_rate": passes / len(model_runs) if model_runs else 0,
                "passes": passes,
                "total": len(model_runs),
            }

        eval_times = [r["llm_timing"]["total_eval_s"] for r in model_runs]
        tps_values = [r["llm_timing"]["tokens_per_second"]
                      for r in model_runs if r["llm_timing"]["tokens_per_second"] > 0]
        wall_times = [r["wall_time_s"] for r in model_runs]

        summary[model] = {
            "accuracy": total_passes / total_checks if total_checks else 0,
            "accuracy_fraction": f"{total_passes}/{total_checks}",
            "field_stats": field_stats,
            "timing": {
                "mean_eval_s": statistics.mean(eval_times) if eval_times else 0,
                "median_eval_s": statistics.median(eval_times) if eval_times else 0,
                "std_eval_s": statistics.stdev(eval_times) if len(eval_times) > 1 else 0,
                "mean_wall_s": statistics.mean(wall_times) if wall_times else 0,
                "mean_tps": statistics.mean(tps_values) if tps_values else 0,
            },
            "consistency": compute_consistency(model_runs),
            "errors": [r["error"] for r in model_runs if r["error"]],
        }
    return summary

# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_progress(record: dict, current: int, total: int):
    passed = record["pass_count"]
    total_f = record["total_fields"]
    eval_s = record["llm_timing"]["total_eval_s"]
    tps = record["llm_timing"]["tokens_per_second"]
    status = "ERROR" if record["error"] else f"{passed}/{total_f}"
    print(f"  [{current:3d}/{total}] {record['model']:20s} | "
          f"{record['fixture']:30s} | run {record['run']} | "
          f"{status:5s} | {eval_s:6.1f}s | {tps:6.1f} tok/s")


def print_summary_table(summary: dict):
    ranked = sorted(summary.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    w = 88
    print("\n" + "=" * w)
    print(f"{'Model':<20} {'Accuracy':<12} {'Consist.':<10} {'tok/s':<10} "
          f"{'Eval(s)':<10} {'Wall(s)':<10} {'Errors':<8}")
    print("=" * w)
    for model, s in ranked:
        print(f"{model:<20} "
              f"{s['accuracy_fraction']:<12} "
              f"{s['consistency']['consistency_rate']:>5.0%}{'':>4} "
              f"{s['timing']['mean_tps']:<10.1f} "
              f"{s['timing']['mean_eval_s']:<10.1f} "
              f"{s['timing']['mean_wall_s']:<10.1f} "
              f"{len(s['errors']):<8}")
    print("=" * w)


def print_field_breakdown(summary: dict):
    ranked = sorted(summary.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    fields = list(FIELD_CHECKS.keys())
    short = {
        "total": "total", "date": "date", "currency": "curr",
        "subtotal": "sub", "payment_method": "pay",
        "line_items_count": "#items", "line_items_totals": "item$",
        "tax_amount": "tax", "merchant_similarity": "merch",
        "tax_categories": "taxCat",
    }
    header = f"{'Model':<20} " + " ".join(f"{short.get(f, f):>6}" for f in fields)
    w = len(header) + 2
    print("\n" + "-" * w)
    print("Per-field pass counts (across all fixture*run combos):")
    print("-" * w)
    print(header)
    print("-" * w)
    for model, s in ranked:
        cells = []
        for f in fields:
            fs = s["field_stats"][f]
            cells.append(f"{fs['passes']:>2}/{fs['total']:<2}")
        print(f"{model:<20} " + "  ".join(cells))
    print("-" * w)


def save_results(results: dict, output_path: Path):
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def load_completed_runs(resume_path: Path | None) -> tuple[set[tuple[str, str, int]], list[dict]]:
    if not resume_path or not resume_path.exists():
        return set(), []
    data = json.loads(resume_path.read_text(encoding="utf-8"))
    runs = data.get("runs", [])
    keys = {(r["model"], r["fixture"], r["run"]) for r in runs}
    return keys, runs

# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    models: list[str],
    fixture_names: list[str] | None,
    runs: int,
    passes: int,
    output_path: Path,
    pull_missing: bool,
    resume_path: Path | None,
) -> dict:
    print(f"Benchmark: {len(models)} models x {runs} runs x {passes} passes")

    # Preflight
    ready_models = preflight_checks(models, pull_missing)
    if not ready_models:
        print("No models available. Exiting.")
        sys.exit(1)

    fixtures = discover_fixtures(fixture_names)
    if not fixtures:
        print("No fixtures found. Exiting.")
        sys.exit(1)

    print(f"  Models: {ready_models}")
    print(f"  Fixtures: {[f[0] for f in fixtures]}")

    total_combos = len(ready_models) * len(fixtures) * runs

    # Resume
    completed_keys, prev_runs = load_completed_runs(resume_path)
    if completed_keys:
        print(f"  Resuming: {len(completed_keys)} runs already completed")

    # Init OCR engine once
    try:
        from ocr import init_cloud_vision
        init_cloud_vision()
    except Exception as e:
        print(f"  WARNING: Cloud Vision init: {e} (will rely on OCR cache)")

    install_instrumentation()

    # Capture GPU status
    gpu_status = get_ollama_gpu_status()
    if gpu_status and not gpu_status["full_gpu"]:
        print(f"  WARNING: Model is {gpu_status['gpu_percent']:.0f}% GPU "
              f"({gpu_status['vram_gb']}/{gpu_status['size_gb']} GiB VRAM). "
              f"Results may be slower than full-GPU baseline.")

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "platform": f"{platform.system()} {platform.release()}",
            "pipeline_version": "1.2.0",
            "passes_per_run": passes,
            "runs_per_combo": runs,
            "total_combinations": total_combos,
            "models": ready_models,
            "fixtures": [f[0] for f in fixtures],
            "gpu_status": gpu_status,
        },
        "runs": list(prev_runs),
        "summary": {},
    }

    current = len(prev_runs)

    try:
        for model in ready_models:
            print(f"\n--- {model} ---")
            for fixture_name, fixture_image, fixture_truth in fixtures:
                for run_idx in range(1, runs + 1):
                    if (model, fixture_name, run_idx) in completed_keys:
                        current += 1
                        continue

                    # Clear KV cache
                    unload_model(model)

                    # Reset collectors
                    _timing_collector.clear()

                    # Run pipeline
                    wall_start = time.perf_counter()
                    error = None
                    try:
                        result = process_document(
                            fixture_image, model=model, passes=passes,
                        )
                        wall_time = time.perf_counter() - wall_start
                    except Exception as e:
                        wall_time = time.perf_counter() - wall_start
                        result = {}
                        error = str(e)

                    # Timing + GPU status
                    llm_timing = aggregate_timing(list(_timing_collector))
                    run_gpu = get_ollama_gpu_status()

                    # Evaluate fields
                    field_results = {}
                    for field_name, check_fn in FIELD_CHECKS.items():
                        field_results[field_name] = check_fn(result, fixture_truth)

                    pass_count = sum(1 for f in field_results.values() if f["pass"])

                    record = {
                        "model": model,
                        "fixture": fixture_name,
                        "run": run_idx,
                        "wall_time_s": round(wall_time, 2),
                        "llm_timing": llm_timing,
                        "gpu_percent": run_gpu["gpu_percent"] if run_gpu else None,
                        "full_gpu": run_gpu["full_gpu"] if run_gpu else None,
                        "fields": field_results,
                        "pass_count": pass_count,
                        "total_fields": len(field_results),
                        "error": error,
                        "extracted": {
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
                        },
                    }

                    results["runs"].append(record)
                    current += 1
                    print_progress(record, current, total_combos)

            # Save after each model
            results["summary"] = compute_summary(results["runs"])
            save_results(results, output_path)
            print(f"  Saved progress ({len(results['runs'])} runs) -> {output_path}")

    except KeyboardInterrupt:
        print("\nInterrupted. Saving partial results...")
    finally:
        restore_instrumentation()
        results["summary"] = compute_summary(results["runs"])
        save_results(results, output_path)

    # Print summary
    if results["summary"]:
        print_summary_table(results["summary"])
        print_field_breakdown(results["summary"])

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark receipt parser models")
    parser.add_argument("--models", nargs="+", default=None,
                        help=f"Models to test (default: {BENCHMARK_MODELS})")
    parser.add_argument("--fixtures", nargs="+", default=None,
                        help="Fixture names to test (default: all)")
    parser.add_argument("--runs", type=int, default=3,
                        help="Runs per model-fixture combo (default: 3)")
    parser.add_argument("--passes", type=int, default=2,
                        help="LLM passes per run (default: 2)")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                        help="Output JSON path (default: benchmark_results.json)")
    parser.add_argument("--pull", action="store_true",
                        help="Auto-pull missing Ollama models")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from partial results JSON")
    args = parser.parse_args()

    models = args.models or BENCHMARK_MODELS
    output_path = Path(args.output)
    resume_path = Path(args.resume) if args.resume else None

    run_benchmark(
        models=models,
        fixture_names=args.fixtures,
        runs=args.runs,
        passes=args.passes,
        output_path=output_path,
        pull_missing=args.pull,
        resume_path=resume_path,
    )


if __name__ == "__main__":
    main()
