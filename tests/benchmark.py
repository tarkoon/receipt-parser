"""benchmark.py — Robustness benchmark for the receipt parser pipeline.

Stress-tests the pipeline against OCR variation by making fresh Cloud Vision
API calls (skipping cache) and tracking per-field accuracy, variance attribution,
timing, and cost across multiple iterations per fixture.

Auto-saves unique failing OCR variants to tests/ocr_variants/ for regression testing.

Usage:
    python tests/benchmark.py
    python tests/benchmark.py --workers 4 --runs 3
    python tests/benchmark.py --fixtures receipt_2 receipt_8 --runs 5
    python tests/benchmark.py --ci
    python tests/benchmark.py --compare tests/results/benchmark/latest.json
"""

import argparse
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from receipt_parser.checks import get_checks_for, ALL_CHECKS
from receipt_parser.llm import check_model_available, DEFAULT_MODEL
from receipt_parser.ocr import init_cloud_vision, get_api_usage, get_ollama_gpu_status
from receipt_parser.pipeline import process_document

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
VARIANTS_DIR = Path(__file__).resolve().parent / "ocr_variants"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "benchmark"
DEFAULT_OUTPUT = RESULTS_DIR / "latest.json"
DEFAULT_BUDGET_LIMIT = 200

# DeepSeek pricing (per million tokens)
_DEEPSEEK_INPUT_COST_PER_M = 0.27
_DEEPSEEK_OUTPUT_COST_PER_M = 1.10


# ---------------------------------------------------------------------------
# Fixture discovery
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
# Variance attribution
# ---------------------------------------------------------------------------

def _text_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _attribute_failure(failed_field: str, failed_run: dict, ref_run: dict) -> str:
    """Attribute a field failure to OCR_VARIANCE, LLM_VARIANCE, or POST_PROCESSING."""
    ocr_sim = _text_similarity(
        failed_run.get("ocr_text", ""),
        ref_run.get("ocr_text", ""),
    )
    if ocr_sim < 0.95:
        return "OCR_VARIANCE"

    failed_llm = failed_run.get("llm_raw", {})
    ref_llm = ref_run.get("llm_raw", {})

    field_key_map = {
        "total": "total", "date": "date", "currency": "currency",
        "subtotal": "subtotal", "payment_method": "payment_method",
        "line_items_count": "line_items", "line_items_totals": "line_items",
        "tax_amount": "taxes", "merchant_similarity": "merchant",
        "tax_categories": "line_items", "document_type": "document_type",
        "amount_paid": "amount_paid", "item_descriptions": "line_items",
        "service_type": "service_type", "usage_amount": "usage",
        "payer": "payer",
    }

    key = field_key_map.get(failed_field, failed_field)
    failed_val = failed_llm.get(key)
    ref_val = ref_llm.get(key)

    if json.dumps(failed_val, sort_keys=True, default=str) != \
       json.dumps(ref_val, sort_keys=True, default=str):
        return "LLM_VARIANCE"

    return "POST_PROCESSING"


# ---------------------------------------------------------------------------
# OCR variant auto-save
# ---------------------------------------------------------------------------

def _save_variant(fixture_name: str, ocr_text: str) -> Path | None:
    """Save a unique failing OCR variant. Returns path if saved, None if deduplicated."""
    VARIANTS_DIR.mkdir(parents=True, exist_ok=True)

    existing = sorted(VARIANTS_DIR.glob(f"{fixture_name}_v*.txt"))

    for existing_file in existing:
        existing_text = existing_file.read_text(encoding="utf-8")
        if _text_similarity(ocr_text, existing_text) > 0.98:
            return None

    version = len(existing) + 1
    path = VARIANTS_DIR / f"{fixture_name}_v{version}.txt"
    path.write_text(ocr_text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Budget management
# ---------------------------------------------------------------------------

def _estimate_api_calls(n_fixtures: int, n_runs: int) -> int:
    calls_per_fixture_per_run = 1
    retry_estimate = max(1, int(n_fixtures * 0.10))
    rotation_extras = max(1, int(n_fixtures * 0.15))
    return (n_fixtures * calls_per_fixture_per_run + retry_estimate + rotation_extras) * n_runs


def _check_budget(estimated_calls: int, budget_limit: int, force: bool) -> bool:
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
        print(f"WARNING: Estimated calls ({estimated_calls}) would use >50% of remaining "
              f"free tier ({remaining}).")
        if not force:
            print("Use --force to override.")
            return False

    print("Proceeding.\n")
    return True


# ---------------------------------------------------------------------------
# Single fixture runner
# ---------------------------------------------------------------------------

def _run_fixture(
    fixture_name: str,
    fixture_image: Path,
    fixture_truth: dict,
    runs: int,
    model: str,
    passes: int,
    cv_client,
) -> tuple[str, dict]:
    """Run all iterations for a single fixture. Returns (name, fixture_data)."""
    checks = get_checks_for(fixture_truth)
    fixture_runs = []

    for run_idx in range(1, runs + 1):
        wall_start = time.perf_counter()
        error = None
        result = {}
        try:
            result = process_document(
                fixture_image, model=model, passes=passes,
                apply_user_rules=False, skip_ocr_cache=True,
                ocr_engine=cv_client,
            )
        except Exception as e:
            error = str(e)
        wall_time = time.perf_counter() - wall_start

        # Evaluate fields
        field_results = {}
        for field_name, check_fn in checks.items():
            field_results[field_name] = check_fn(result, fixture_truth)

        pass_count = sum(1 for f in field_results.values() if f["pass"])
        total_fields = len(field_results)

        # Capture LLM raw extraction from pass history
        llm_raw = {}
        pass_history = result.get("_pass_history", [])
        if pass_history:
            llm_raw = pass_history[0].get("extraction", {})

        # Build run record
        run_record = {
            "run": run_idx,
            "passed": pass_count == total_fields,
            "pass_count": pass_count,
            "total_fields": total_fields,
            "wall_time_s": round(wall_time, 2),
            "error": error,
            "fields": field_results,
            "ocr": {
                "confidence": result.get("_ocr_confidence"),
                "retried": result.get("_ocr_retried", False),
                "retry_reason": result.get("_ocr_retry_reason"),
                "source": result.get("_ocr_source", "unknown"),
            },
            "ocr_text": result.get("_ocr_text", ""),
            "llm_raw": llm_raw,
            "warnings": result.get("_warnings", []),
            "warning_count": len(result.get("_warnings", [])),
            "llm_passes_used": result.get("_pass_count", 1),
        }

        fixture_runs.append(run_record)

        # Print progress
        failed = [f for f, r in field_results.items() if not r["pass"]]
        fail_str = f"  <- {', '.join(failed)}" if failed else ""
        conf = run_record["ocr"]["confidence"] or 0
        retry_tag = "retry" if run_record["ocr"]["retried"] else "1-call"
        status = f"{pass_count}/{total_fields}" if not error else "ERROR"
        print(f"  Run {run_idx}: {status:5s}  OCR conf: {conf:.2f} ({retry_tag})  "
              f"wall: {wall_time:.1f}s{fail_str}")

    # Finalize fixture
    fixture_data = {"runs": fixture_runs}
    _finalize_fixture(fixture_name, fixture_data)
    return fixture_name, fixture_data


def _finalize_fixture(fixture_name: str, fdata: dict):
    """Compute attribution, field robustness, determinism, and save variants."""
    runs = fdata["runs"]
    if not runs:
        return

    checks_used = list(runs[0]["fields"].keys())

    # Find reference (passing) run
    ref_run = None
    for r in runs:
        if r["pass_count"] == r["total_fields"]:
            ref_run = r
            break
    if ref_run is None:
        ref_run = max(runs, key=lambda r: r["pass_count"])

    # Attribution for failed fields
    for run in runs:
        for field_name, field_result in run["fields"].items():
            if not field_result["pass"] and ref_run["fields"][field_name]["pass"]:
                attr = _attribute_failure(field_name, run, ref_run)
                field_result["attribution"] = attr

    # Auto-save OCR variants for failing runs
    variants_saved = 0
    for run in runs:
        has_failure = any(not f["pass"] for f in run["fields"].values())
        if has_failure and run.get("ocr_text"):
            path = _save_variant(fixture_name, run["ocr_text"])
            if path:
                variants_saved += 1

    # Per-field robustness
    field_robustness = {}
    for field_name in checks_used:
        passes = sum(1 for r in runs if r["fields"][field_name]["pass"])
        total = len(runs)
        consistent = (all(r["fields"][field_name]["pass"] for r in runs) or
                      not any(r["fields"][field_name]["pass"] for r in runs))
        fr = {"pass_rate": round(passes / total, 4) if total else 1.0, "consistent": consistent}
        attrs: dict[str, int] = defaultdict(int)
        for r in runs:
            attr = r["fields"][field_name].get("attribution")
            if attr:
                attrs[attr] += 1
        if attrs:
            fr["failure_attribution"] = dict(attrs)
        field_robustness[field_name] = fr
    fdata["field_robustness"] = field_robustness

    # OCR analysis
    confidences = [r["ocr"]["confidence"] for r in runs if r["ocr"]["confidence"] is not None]
    retried_count = sum(1 for r in runs if r["ocr"].get("retried"))
    texts = [r.get("ocr_text", "") for r in runs]
    sims = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sims.append(_text_similarity(texts[i], texts[j]))
    fdata["ocr_analysis"] = {
        "mean_confidence": sum(confidences) / len(confidences) if confidences else 0,
        "min_confidence": min(confidences) if confidences else 0,
        "max_confidence": max(confidences) if confidences else 0,
        "retried_pct": round(retried_count / len(runs) * 100, 1) if runs else 0,
        "cross_run_similarity": {
            "mean": sum(sims) / len(sims) if sims else 1.0,
            "min": min(sims) if sims else 1.0,
        },
    }

    # Determinism
    summaries = [json.dumps({k: r["fields"][k]["pass"] for k in checks_used}, sort_keys=True)
                 for r in runs]
    unique = len(set(summaries))
    fdata["deterministic"] = unique == 1

    # Status
    all_pass = all(r["passed"] for r in runs)
    fdata["status"] = "ROBUST" if all_pass else "FRAGILE"
    fdata["score"] = round(sum(r["pass_count"] for r in runs) /
                           sum(r["total_fields"] for r in runs), 4) if runs else 1.0
    fdata["variants_saved"] = variants_saved

    det_tag = "" if fdata["deterministic"] else " NON-DET"
    total_p = sum(r["pass_count"] for r in runs)
    total_c = sum(r["total_fields"] for r in runs)
    status = f"{fdata['status']} ({total_p}/{total_c}){det_tag}"
    if variants_saved:
        status += f" - {variants_saved} variant(s) saved"
    print(f"  -> {status}")


# ---------------------------------------------------------------------------
# Overall summary
# ---------------------------------------------------------------------------

def _compute_summary(per_fixture: dict, metadata: dict) -> dict:
    total_checks = 0
    total_passed = 0
    field_pass: dict[str, int] = defaultdict(int)
    field_total: dict[str, int] = defaultdict(int)
    variance_attr: dict[str, int] = defaultdict(int)
    fragile = []

    for fname, fdata in per_fixture.items():
        for run in fdata.get("runs", []):
            for field_name, fr in run.get("fields", {}).items():
                total_checks += 1
                field_total[field_name] += 1
                if fr.get("pass"):
                    total_passed += 1
                    field_pass[field_name] += 1
                else:
                    attr = fr.get("attribution")
                    if attr:
                        variance_attr[attr] += 1
        if fdata.get("status") == "FRAGILE":
            fragile.append(fname)

    score = total_passed / total_checks if total_checks else 1.0
    n_fixtures = len(per_fixture)

    # Timing
    all_wall = [r["wall_time_s"] for fd in per_fixture.values() for r in fd.get("runs", [])
                if r.get("wall_time_s")]
    # Determinism
    det_count = sum(1 for fd in per_fixture.values() if fd.get("deterministic"))
    # Variants
    total_variants = sum(fd.get("variants_saved", 0) for fd in per_fixture.values())

    return {
        "score": round(score, 4),
        "total_checks": total_checks,
        "total_passed": total_passed,
        "fixtures_robust": n_fixtures - len(fragile),
        "fixtures_fragile": len(fragile),
        "fragile": fragile,
        "mean_wall_s": round(sum(all_wall) / len(all_wall), 1) if all_wall else 0,
        "determinism_rate": round(det_count / n_fixtures, 2) if n_fixtures else 1.0,
        "variants_saved": total_variants,
        "field_robustness": {f: round(field_pass[f] / field_total[f], 4) if field_total[f] else 1.0
                             for f in sorted(field_total)},
        "variance_attribution": dict(variance_attr),
    }


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def _print_summary(summary: dict, metadata: dict):
    print(f"\n{'=' * 70}")
    print("=== Summary ===")
    print(f"{'=' * 70}")
    print(f"Overall: {summary['score']:.1%} ({summary['total_passed']}/{summary['total_checks']})")
    print(f"Fixtures: {summary['fixtures_robust']} robust, {summary['fixtures_fragile']} fragile")
    if summary["fragile"]:
        print(f"Fragile: {', '.join(summary['fragile'])}")

    if summary["variance_attribution"]:
        print(f"\nVariance Attribution:")
        total_failures = sum(summary["variance_attribution"].values())
        for attr, count in sorted(summary["variance_attribution"].items()):
            pct = count / total_failures * 100
            print(f"  {attr:20s} {count:3d} failures ({pct:.0f}%)")

    print(f"\nPer-Field Robustness:")
    for field, score in sorted(summary["field_robustness"].items(), key=lambda x: x[1]):
        bar = "#" * int(score * 20)
        print(f"  {field:25s} {score:6.1%}  {bar}")

    print(f"\nPerformance:")
    print(f"  Mean wall time: {summary['mean_wall_s']:.1f}s")
    print(f"  Determinism:    {summary['determinism_rate']:.0%}")

    if summary["variants_saved"]:
        print(f"\nOCR variants saved: {summary['variants_saved']} (in {VARIANTS_DIR})")

    usage = get_api_usage()
    print(f"\nAPI Budget: {usage['remaining']} calls remaining")
    print(f"Results saved: {metadata.get('output_path', '?')}")


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def _get_git_sha() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _assemble_results(metadata: dict, per_fixture: dict) -> dict:
    # Clean run data for JSON output (strip large text fields)
    clean_fixtures = {}
    for fname, fdata in per_fixture.items():
        clean_fdata = dict(fdata)
        clean_runs = []
        for run in fdata.get("runs", []):
            clean_run = dict(run)
            # Save OCR text as companion file, remove from JSON
            if clean_run.get("ocr_text"):
                ocr_dir = RESULTS_DIR / "ocr"
                ocr_dir.mkdir(parents=True, exist_ok=True)
                ocr_path = ocr_dir / f"{fname}_run{run['run']}.txt"
                ocr_path.write_text(clean_run["ocr_text"], encoding="utf-8")
                clean_run["ocr"]["text_file"] = f"ocr/{fname}_run{run['run']}.txt"
            clean_run.pop("ocr_text", None)
            # Save LLM raw as companion file
            if clean_run.get("llm_raw"):
                llm_dir = RESULTS_DIR / "llm"
                llm_dir.mkdir(parents=True, exist_ok=True)
                llm_path = llm_dir / f"{fname}_run{run['run']}.json"
                llm_path.write_text(json.dumps(clean_run["llm_raw"], ensure_ascii=False, indent=2),
                                    encoding="utf-8")
                clean_run["llm_raw_file"] = f"llm/{fname}_run{run['run']}.json"
            clean_run.pop("llm_raw", None)
            clean_runs.append(clean_run)
        clean_fdata["runs"] = clean_runs
        clean_fixtures[fname] = clean_fdata

    summary = _compute_summary(per_fixture, metadata)

    return {
        "schema_version": 2,
        "metadata": metadata,
        "summary": summary,
        "per_fixture": clean_fixtures,
    }


def _save_results(results: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    # Also save timestamped archive
    git_sha = results["metadata"].get("git_sha", "unknown")
    ts = datetime.now().strftime("%Y-%m-%d")
    archive_path = output_path.parent / f"{ts}_{git_sha}.json"
    if not archive_path.exists():
        archive_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _compare_results(current: dict, previous: dict):
    curr = current.get("summary", {})
    prev = previous.get("summary", previous.get("overall", {}))

    print(f"\n{'=' * 70}")
    print("=== Comparison vs Previous ===")
    print(f"{'=' * 70}")

    curr_score = curr.get("score", curr.get("robustness_score", 0))
    prev_score = prev.get("score", prev.get("robustness_score", 0))
    delta = curr_score - prev_score
    print(f"Accuracy: {prev_score:.1%} -> {curr_score:.1%} ({delta:+.1%})")

    curr_fragile = set(curr.get("fragile", curr.get("fragile_fixtures", [])))
    prev_fragile = set(prev.get("fragile", prev.get("fragile_fixtures", [])))
    fixed = prev_fragile - curr_fragile
    regressed = curr_fragile - prev_fragile
    if fixed:
        print(f"Fixed: {', '.join(sorted(fixed))}")
    if regressed:
        print(f"Regressed: {', '.join(sorted(regressed))}")
    if not fixed and not regressed:
        print("No fixture status changes.")


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

class _BudgetExceeded(Exception):
    pass


def run_benchmark(
    runs: int = 3,
    fixture_names: list[str] | None = None,
    output_path: Path = DEFAULT_OUTPUT,
    budget_limit: int = DEFAULT_BUDGET_LIMIT,
    model: str = DEFAULT_MODEL,
    passes: int = 2,
    force: bool = False,
    workers: int = 1,
    ci: bool = False,
) -> dict:
    """Main benchmark entry point."""
    # CI mode overrides
    if ci:
        runs = 1

    fixtures = discover_fixtures(fixture_names)
    if not fixtures:
        print("No fixtures found. Exiting.")
        sys.exit(1)

    n_fixtures = len(fixtures)
    fixture_name_list = [f[0] for f in fixtures]

    print(f"{'=' * 60}")
    print(f"=== Robustness Benchmark ===")
    print(f"Model: {model} | Runs: {runs} | Fixtures: {n_fixtures} | "
          f"Workers: {workers} | Passes: {passes}")
    if ci:
        print(f"CI mode: cached OCR, 1 run, exit non-zero on failure")

    # Preflight
    check_model_available(model)
    try:
        cv_client = init_cloud_vision()
    except Exception as e:
        print(f"ERROR: Cloud Vision init failed: {e}")
        sys.exit(1)

    # Budget check (skip in CI mode — uses cached OCR)
    if not ci:
        estimated = _estimate_api_calls(n_fixtures, runs)
        if not _check_budget(estimated, budget_limit, force):
            sys.exit(1)

    git_sha = _get_git_sha()
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "git_sha": git_sha,
        "model": model,
        "runs_per_fixture": runs,
        "passes": passes,
        "workers": workers,
        "ci_mode": ci,
        "fixtures": fixture_name_list,
        "output_path": str(output_path),
    }

    per_fixture: dict = {}

    if workers > 1 and not ci:
        # Parallel execution across fixtures
        print(f"\nRunning {n_fixtures} fixtures with {workers} workers...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_fixture, name, image, truth, runs, model, passes, cv_client): name
                for name, image, truth in fixtures
            }
            for future in as_completed(futures):
                fname, fdata = future.result()
                per_fixture[fname] = fdata
    else:
        # Sequential
        for fix_idx, (name, image, truth) in enumerate(fixtures):
            print(f"\n[{fix_idx + 1}/{n_fixtures}] {name}")
            skip_cache = not ci  # CI mode uses cached OCR
            _, fdata = _run_fixture_sequential(
                name, image, truth, runs, model, passes, cv_client, skip_cache,
            )
            per_fixture[name] = fdata

    # Assemble and save
    results = _assemble_results(metadata, per_fixture)
    _save_results(results, output_path)
    _print_summary(results["summary"], metadata)

    # CI mode: exit non-zero if any fixture failed
    if ci:
        fragile = results["summary"].get("fragile", [])
        if fragile:
            print(f"\nCI FAILURE: {len(fragile)} fragile fixtures")
            sys.exit(1)

    return results


def _run_fixture_sequential(
    fixture_name, fixture_image, fixture_truth, runs, model, passes, cv_client, skip_cache,
):
    """Sequential fixture runner with progress printing."""
    checks = get_checks_for(fixture_truth)
    fixture_runs = []

    for run_idx in range(1, runs + 1):
        wall_start = time.perf_counter()
        error = None
        result = {}
        try:
            result = process_document(
                fixture_image, model=model, passes=passes,
                apply_user_rules=False, skip_ocr_cache=skip_cache,
                ocr_engine=cv_client,
            )
        except Exception as e:
            error = str(e)
        wall_time = time.perf_counter() - wall_start

        field_results = {}
        for field_name, check_fn in checks.items():
            field_results[field_name] = check_fn(result, fixture_truth)

        pass_count = sum(1 for f in field_results.values() if f["pass"])
        total_fields = len(field_results)

        llm_raw = {}
        pass_history = result.get("_pass_history", [])
        if pass_history:
            llm_raw = pass_history[0].get("extraction", {})

        run_record = {
            "run": run_idx,
            "passed": pass_count == total_fields,
            "pass_count": pass_count,
            "total_fields": total_fields,
            "wall_time_s": round(wall_time, 2),
            "error": error,
            "fields": field_results,
            "ocr": {
                "confidence": result.get("_ocr_confidence"),
                "retried": result.get("_ocr_retried", False),
                "retry_reason": result.get("_ocr_retry_reason"),
                "source": result.get("_ocr_source", "unknown"),
            },
            "ocr_text": result.get("_ocr_text", ""),
            "llm_raw": llm_raw,
            "warnings": result.get("_warnings", []),
            "warning_count": len(result.get("_warnings", [])),
            "llm_passes_used": result.get("_pass_count", 1),
        }
        fixture_runs.append(run_record)

        failed = [f for f, r in field_results.items() if not r["pass"]]
        fail_str = f"  <- {', '.join(failed)}" if failed else ""
        conf = run_record["ocr"]["confidence"] or 0
        retry_tag = "retry" if run_record["ocr"]["retried"] else "1-call"
        status = f"{pass_count}/{total_fields}" if not error else "ERROR"
        print(f"  Run {run_idx}: {status:5s}  OCR conf: {conf:.2f} ({retry_tag})  "
              f"wall: {wall_time:.1f}s{fail_str}")

    fixture_data = {"runs": fixture_runs}
    _finalize_fixture(fixture_name, fixture_data)
    return fixture_name, fixture_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Robustness benchmark: stress-test pipeline against OCR variation"
    )
    parser.add_argument("--runs", type=int, default=3,
                        help="OCR iterations per fixture (default: 3)")
    parser.add_argument("--fixtures", nargs="+", default=None,
                        help="Specific fixture names (default: all)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help=f"JSON output file (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--budget-limit", type=int, default=DEFAULT_BUDGET_LIMIT,
                        help=f"Max API calls before stopping (default: {DEFAULT_BUDGET_LIMIT})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"LLM model (default: {DEFAULT_MODEL})")
    parser.add_argument("--passes", type=int, default=2,
                        help="LLM verification passes (default: 2)")
    parser.add_argument("--compare", type=str, default=None,
                        help="Compare results against a previous benchmark JSON")
    parser.add_argument("--force", action="store_true",
                        help="Skip budget warnings")
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent fixture processing (default: 1, max: 8)")
    parser.add_argument("--ci", action="store_true",
                        help="CI mode: cached OCR, 1 run, exit non-zero on failure")
    args = parser.parse_args()

    results = run_benchmark(
        runs=args.runs,
        fixture_names=args.fixtures,
        output_path=Path(args.output),
        budget_limit=args.budget_limit,
        model=args.model,
        passes=args.passes,
        force=args.force,
        workers=min(args.workers, 8),
        ci=args.ci,
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
