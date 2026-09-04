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
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import cv2
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from receipt_parser.checks import get_checks_for
from receipt_parser.llm import check_model_available, DEFAULT_MODEL
from receipt_parser.ocr import (
    _OCR_CACHE_DIR,
    _ocr_cache_key,
    get_api_usage,
    init_cloud_vision,
)
from receipt_parser.pipeline import process_document, process_ocr_text
from receipt_parser.preprocess import load_image, try_extract_text_layer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
OCR_FIXTURES_DIR = Path(__file__).resolve().parent / "ocr_fixtures"
VARIANTS_DIR = Path(__file__).resolve().parent.parent / ".data" / "ocr_cache" / "variants"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "benchmark"
DEFAULT_OUTPUT = RESULTS_DIR / "latest.json"
DEFAULT_BUDGET_LIMIT = 200
REPO_ROOT = Path(__file__).resolve().parent.parent
OCR_CACHE_DIR = _OCR_CACHE_DIR


class _CacheOnlyOCREngine:
    def annotate_image(self, **_kwargs):
        raise RuntimeError("Cached OCR required; refusing a Vision API call")

    def document_text_detection(self, **_kwargs):
        raise RuntimeError("Cached OCR required; refusing a Vision API call")

# DeepSeek pricing — imported from the canonical source
from receipt_parser.usage import (
    DEEPSEEK_CACHE_MISS_COST_PER_M,
    DEEPSEEK_OUTPUT_COST_PER_M,
)
# Backwards-compat alias (benchmark uses aggregate input cost estimate)
_DEEPSEEK_INPUT_COST_PER_M = DEEPSEEK_CACHE_MISS_COST_PER_M
_DEEPSEEK_OUTPUT_COST_PER_M = DEEPSEEK_OUTPUT_COST_PER_M


# ---------------------------------------------------------------------------
# Fixture discovery
# ---------------------------------------------------------------------------

def discover_fixtures(names: list[str] | None = None) -> list[tuple[str, Path, dict]]:
    """Discover test fixtures. Prefers images + original truth; falls back to named OCR + public truth."""
    fixtures = []
    discovered = set()

    # Image fixtures (original truth, full pipeline)
    for truth_file in sorted(FIXTURES_DIR.glob("*_truth.json")):
        if truth_file.name == "_truth_template.json":
            continue
        if "_public_truth" in truth_file.name:
            continue
        base = truth_file.stem.replace("_truth", "")
        if names is not None and base not in names:
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
        discovered.add(base)

    # Public fixtures (anonymized truth + named OCR text, no images needed)
    for truth_file in sorted(FIXTURES_DIR.glob("*_public_truth.json")):
        base = truth_file.stem.replace("_public_truth", "")
        if base in discovered:
            continue
        if names is not None and base not in names:
            continue
        ocr_file = OCR_FIXTURES_DIR / f"{base}.txt"
        if not ocr_file.exists():
            continue
        truth = json.loads(truth_file.read_text(encoding="utf-8"))
        fixtures.append((base, ocr_file, truth))
        discovered.add(base)

    # Explicit OCR variants (variant text + its base receipt truth, no image needed)
    if names is not None and VARIANTS_DIR.exists():
        for variant_file in sorted(VARIANTS_DIR.glob("*.txt")):
            stem = variant_file.stem
            base = re.sub(r"_v\d+$", "", stem)
            if base == stem or stem not in names:
                continue
            truth_file = FIXTURES_DIR / f"{base}_truth.json"
            if not truth_file.exists():
                truth_file = FIXTURES_DIR / f"{base}_public_truth.json"
            if not truth_file.exists():
                continue
            truth = json.loads(truth_file.read_text(encoding="utf-8"))
            fixtures.append((stem, variant_file, truth))

    return fixtures


def _select_fixtures(names: list[str] | None) -> list[tuple[str, Path, dict]]:
    if names is None:
        return discover_fixtures()
    if not names:
        raise ValueError("--fixtures requires at least one fixture name")
    requested = list(dict.fromkeys(names))
    fixtures = discover_fixtures(requested)
    resolved = {name for name, _source, _truth in fixtures}
    unresolved = [name for name in requested if name not in resolved]
    if unresolved:
        raise ValueError(f"Unknown or unavailable fixture(s): {', '.join(unresolved)}")
    return fixtures


def _cached_ocr_artifacts(source: Path) -> list[tuple[str, Path]]:
    """Return cache text/layout files that the cached OCR path will read."""
    if source.suffix.lower() == ".pdf" and try_extract_text_layer(str(source)):
        return []
    artifacts = []
    for page, image in enumerate(load_image(source)):
        key = _ocr_cache_key(image)
        text_path = OCR_CACHE_DIR / f"{key}.txt"
        artifacts.append((f"page:{page}:ocr_text", text_path))
        if not text_path.exists():
            continue
        artifacts.append((f"page:{page}:ocr_layout", OCR_CACHE_DIR / f"{key}.layout.json"))
        block_count = len([line for line in text_path.read_text(encoding="utf-8").splitlines() if line.strip()])
        if block_count >= 3:
            continue
        best_count = block_count
        best_confidence = 0.9 if block_count else 0.0
        rotations = (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE)
        for rotation_index, rotation in enumerate(rotations, 1):
            rotated_key = _ocr_cache_key(cv2.rotate(image, rotation))
            rotated_text = OCR_CACHE_DIR / f"{rotated_key}.txt"
            prefix = f"page:{page}:rotation:{rotation_index}"
            artifacts.append((f"{prefix}:ocr_text", rotated_text))
            if not rotated_text.exists():
                break
            artifacts.append((f"{prefix}:ocr_layout", OCR_CACHE_DIR / f"{rotated_key}.layout.json"))
            rotated_count = len([
                line for line in rotated_text.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ])
            rotated_confidence = 0.9 if rotated_count else 0.0
            if rotated_count > best_count or rotated_confidence > best_confidence:
                best_count = rotated_count
                best_confidence = rotated_confidence
            if best_confidence >= 0.85:
                break
    return artifacts


def _update_path_fingerprint(digest, label: str, path: Path) -> None:
    digest.update(label.encode())
    digest.update(b"\0")
    if path.is_file():
        digest.update(b"present\0")
        digest.update(path.read_bytes())
    else:
        digest.update(b"missing\0")


def _fixture_corpus_sha256(
    fixtures: list[tuple[str, Path, dict]], *, cached_ocr: bool = False,
) -> str:
    digest = hashlib.sha256()
    for name, source, truth in fixtures:
        digest.update(name.encode())
        digest.update(source.read_bytes())
        digest.update(json.dumps(truth, ensure_ascii=False, sort_keys=True).encode())
        if cached_ocr and source.suffix != ".txt":
            for label, path in _cached_ocr_artifacts(source):
                _update_path_fingerprint(digest, label, path)
    return digest.hexdigest()


def _missing_cached_ocr(fixtures: list[tuple[str, Path, dict]]) -> list[Path]:
    return [
        path
        for _name, source, _truth in fixtures
        if source.suffix != ".txt"
        for label, path in _cached_ocr_artifacts(source)
        if label.endswith("ocr_text") and not path.is_file()
    ]


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
        "line_items_qty": "line_items", "line_items_unit_price": "line_items",
        "line_items_discounts": "line_items",
        "tax_amount": "taxes", "tax_rates": "taxes", "tax_labels": "taxes",
        "merchant_similarity": "merchant",
        "tax_categories": "line_items", "document_type": "document_type",
        "amount_paid": "amount_paid", "item_descriptions": "line_items",
        "points_used": "points_used",
        "service_type": "service_type", "usage_amount": "usage",
        "payer": "payer", "account_number": "account_number",
        "payment_reference": "payment_reference",
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
    fixture_source: Path,
    fixture_truth: dict,
    runs: int,
    model: str,
    passes: int,
    cv_client,
    skip_cache: bool = True,
    save_variants: bool = True,
) -> tuple[str, dict]:
    """Run all iterations for a single fixture. Returns (name, fixture_data)."""
    is_ocr_text = fixture_source.suffix == ".txt"
    checks = get_checks_for(fixture_truth)
    fixture_runs = []

    for run_idx in range(1, runs + 1):
        wall_start = time.perf_counter()
        error = None
        result = {}
        try:
            if is_ocr_text:
                ocr_text = fixture_source.read_text(encoding="utf-8")
                result = process_ocr_text(
                    ocr_text, model=model, passes=passes,
                    apply_user_rules=False,
                )
            else:
                digital_pdf = (
                    not skip_cache
                    and fixture_source.suffix.lower() == ".pdf"
                    and bool(try_extract_text_layer(str(fixture_source)))
                )
                result = process_document(
                    fixture_source, model=model, passes=passes,
                    apply_user_rules=False, skip_ocr_cache=skip_cache,
                    ocr_engine=cv_client,
                )
                if not skip_cache and not digital_pdf and result.get("_ocr_source") != "cache":
                    raise RuntimeError(
                        "Cached benchmark received non-cache OCR source: "
                        f"{result.get('_ocr_source', 'missing')}"
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
            "passed": error is None and pass_count == total_fields,
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
            "llm_raw": deepcopy(llm_raw),
            "llm_pass_history": deepcopy(pass_history),
            "final_extraction": _public_extraction(result),
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
    _finalize_fixture(fixture_name, fixture_data, save_variants=save_variants)
    return fixture_name, fixture_data


def _finalize_fixture(fixture_name: str, fdata: dict, *, save_variants: bool = True):
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
    if save_variants:
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

    try:
        usage = get_api_usage()
        print(f"\nAPI Budget: {usage['remaining']} calls remaining")
    except Exception:
        pass
    print(f"Results saved: {metadata.get('output_path', '?')}")


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def _git_output(*args: str) -> bytes | None:
    try:
        result = subprocess.run(
            ["git", *args], capture_output=True, cwd=REPO_ROOT, timeout=10,
        )
        return result.stdout if result.returncode == 0 else None
    except Exception:
        return None


def _untracked_sha256() -> str | None:
    paths = _git_output("ls-files", "--others", "--exclude-standard", "-z")
    if paths is None:
        return None
    root = REPO_ROOT.resolve()
    digest = hashlib.sha256()
    for raw_path in sorted(path for path in paths.split(b"\0") if path):
        digest.update(raw_path)
        digest.update(b"\0")
        candidate = REPO_ROOT / os.fsdecode(raw_path)
        try:
            if candidate.is_symlink():
                digest.update(b"symlink\0")
                digest.update(os.fsencode(os.readlink(candidate)))
                continue
            candidate.resolve().relative_to(root)
            digest.update(b"file\0")
            digest.update(candidate.read_bytes())
        except (OSError, ValueError):
            digest.update(b"unreadable-or-outside\0")
    return digest.hexdigest()


def _get_git_state() -> dict:
    """Fingerprint tracked changes plus untracked contents inside this repo."""
    sha = _git_output("rev-parse", "--short", "HEAD")
    status = _git_output("status", "--porcelain=v1", "--untracked-files=normal")
    diff = _git_output("diff", "--binary", "HEAD", "--")
    untracked_sha = _untracked_sha256()
    if diff is None and untracked_sha is not None:
        diff = b""  # Unborn repositories have no HEAD yet.
    dirty_sha = None
    if diff is not None and untracked_sha is not None:
        dirty_sha = hashlib.sha256(diff + b"\0" + untracked_sha.encode()).hexdigest()
    return {
        "git_sha": sha.decode(errors="replace").strip() if sha else "unknown",
        "git_dirty": bool(status) if status is not None else None,
        "git_status_sha256": hashlib.sha256(status).hexdigest() if status is not None else None,
        "git_tracked_diff_sha256": hashlib.sha256(diff).hexdigest() if diff is not None else None,
        "git_untracked_sha256": untracked_sha,
        "git_diff_sha256": dirty_sha,
        "git_dirty_tree_sha256": dirty_sha,
        "git_diff_scope": "tracked_changes_plus_untracked_contents",
    }


def _get_git_sha() -> str:
    """Backwards-compatible SHA helper for callers outside this script."""
    sha = _git_output("rev-parse", "--short", "HEAD")
    return sha.decode(errors="replace").strip() if sha else "unknown"


def _artifact_report_path(path: Path) -> str:
    try:
        return path.relative_to(RESULTS_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def _public_extraction(result: dict) -> dict:
    return {
        key: deepcopy(value)
        for key, value in (result or {}).items()
        if not key.startswith("_")
    }


def _assemble_results(metadata: dict, per_fixture: dict) -> dict:
    # Clean run data for JSON output (strip large text fields)
    artifact_dir = Path(metadata.get("artifact_dir") or (RESULTS_DIR / "artifacts" / "latest"))
    clean_fixtures = {}
    for fname, fdata in per_fixture.items():
        clean_fdata = dict(fdata)
        clean_runs = []
        for run in fdata.get("runs", []):
            clean_run = dict(run)
            # Save OCR text as companion file, remove from JSON
            if clean_run.get("ocr_text"):
                ocr_dir = artifact_dir / "ocr"
                ocr_dir.mkdir(parents=True, exist_ok=True)
                ocr_path = ocr_dir / f"{fname}_run{run['run']}.txt"
                ocr_path.write_text(clean_run["ocr_text"], encoding="utf-8")
                clean_run["ocr"]["text_file"] = _artifact_report_path(ocr_path)
            clean_run.pop("ocr_text", None)
            # Save LLM raw as companion file
            if clean_run.get("llm_raw"):
                llm_dir = artifact_dir / "llm"
                llm_dir.mkdir(parents=True, exist_ok=True)
                llm_path = llm_dir / f"{fname}_run{run['run']}.json"
                llm_path.write_text(json.dumps(clean_run["llm_raw"], ensure_ascii=False, indent=2),
                                    encoding="utf-8")
                clean_run["llm_raw_file"] = _artifact_report_path(llm_path)
            clean_run.pop("llm_raw", None)
            # Save full pass history as companion file. This preserves rejected
            # sanity candidates without bloating the main benchmark report.
            if clean_run.get("llm_pass_history"):
                llm_dir = artifact_dir / "llm"
                llm_dir.mkdir(parents=True, exist_ok=True)
                history_path = llm_dir / f"{fname}_run{run['run']}_history.json"
                history_path.write_text(
                    json.dumps(clean_run["llm_pass_history"], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                clean_run["llm_pass_history_file"] = _artifact_report_path(history_path)
            clean_run.pop("llm_pass_history", None)
            # Save final extraction evaluated by checks. This is distinct from
            # pass history because deterministic post-processing can select or
            # mutate candidates after the LLM responses are recorded.
            if clean_run.get("final_extraction"):
                final_dir = artifact_dir / "final"
                final_dir.mkdir(parents=True, exist_ok=True)
                final_path = final_dir / f"{fname}_run{run['run']}_final.json"
                final_path.write_text(
                    json.dumps(clean_run["final_extraction"], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                clean_run["final_file"] = _artifact_report_path(final_path)
            clean_run.pop("final_extraction", None)
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
    run_id = results["metadata"].get("run_id")
    archive_stem = f"{ts}_{git_sha}"
    if run_id:
        archive_stem += f"_{run_id}"
    archive_path = output_path.parent / f"{archive_stem}.json"
    if not archive_path.exists():
        archive_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _comparison_scope(report: dict) -> dict:
    metadata = report.get("metadata", {})
    fixtures = metadata.get("fixtures")
    if fixtures is None and report.get("per_fixture"):
        fixtures = list(report["per_fixture"])
    return {
        "fixtures": frozenset(fixtures) if fixtures is not None else None,
        "fixture_corpus_sha256": metadata.get("fixture_corpus_sha256"),
        "model": metadata.get("model"),
        "passes": metadata.get("passes"),
        "runs_per_fixture": metadata.get("runs_per_fixture"),
        "ci_mode": metadata.get("ci_mode"),
    }


def _fixture_scope_mismatch(current: frozenset | None, previous: frozenset | None) -> str | None:
    if current is None or previous is None:
        return f"fixtures: missing scope metadata (previous={previous}, current={current})"
    if current == previous:
        return None

    def _sample(names: set[str]) -> str:
        ordered = sorted(names)
        suffix = f", +{len(ordered) - 8} more" if len(ordered) > 8 else ""
        return ", ".join(ordered[:8]) + suffix

    if current < previous:
        omitted = set(previous - current)
        return (f"fixtures: current is a subset ({len(current)}/{len(previous)}); "
                f"omitted {_sample(omitted)}")
    if previous < current:
        added = set(current - previous)
        return (f"fixtures: previous is a subset ({len(previous)}/{len(current)}); "
                f"new in current {_sample(added)}")
    return (f"fixtures: sets differ (removed {_sample(set(previous - current))}; "
            f"added {_sample(set(current - previous))})")


def _comparison_scope_mismatches(current: dict, previous: dict) -> list[str]:
    curr_scope = _comparison_scope(current)
    prev_scope = _comparison_scope(previous)
    mismatches = []
    fixture_mismatch = _fixture_scope_mismatch(
        curr_scope["fixtures"], prev_scope["fixtures"],
    )
    if fixture_mismatch:
        mismatches.append(fixture_mismatch)
    for key in ("fixture_corpus_sha256", "model", "passes", "runs_per_fixture", "ci_mode"):
        current_value = curr_scope[key]
        previous_value = prev_scope[key]
        if current_value is None or previous_value is None:
            mismatches.append(
                f"{key}: missing scope metadata "
                f"(previous={previous_value!r}, current={current_value!r})"
            )
        elif current_value != previous_value:
            mismatches.append(
                f"{key}: previous={previous_value!r}, current={current_value!r}"
            )
    return mismatches


def _compare_results(current: dict, previous: dict):
    curr = current.get("summary", {})
    prev = previous.get("summary", previous.get("overall", {}))

    print(f"\n{'=' * 70}")
    print("=== Comparison vs Previous ===")
    print(f"{'=' * 70}")

    mismatches = _comparison_scope_mismatches(current, previous)
    if mismatches:
        print("Comparison refused: benchmark scopes are not comparable.")
        for mismatch in mismatches:
            print(f"  - {mismatch}")
        return False

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
    return True


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
    cached_ocr: bool = False,
    save_variants: bool = True,
) -> dict:
    """Main benchmark entry point."""
    # CI mode overrides
    if ci:
        runs = 1
    use_cached_ocr = ci or cached_ocr

    try:
        fixtures = _select_fixtures(fixture_names)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2) from exc
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
    elif cached_ocr:
        print(f"Cached OCR mode: {runs} run(s), no Vision API calls")

    if use_cached_ocr:
        missing_cache = _missing_cached_ocr(fixtures)
        if missing_cache:
            print(f"ERROR: cached OCR is missing for {len(missing_cache)} image page(s).")
            raise SystemExit(1)

    # Preflight
    check_model_available(model)

    # Check if any fixtures need Cloud Vision (image sources)
    has_image_fixtures = any(f[1].suffix != ".txt" for f in fixtures)
    cv_client = None
    if has_image_fixtures:
        if use_cached_ocr:
            cv_client = _CacheOnlyOCREngine()
        else:
            try:
                cv_client = init_cloud_vision()
            except Exception as e:
                print(f"ERROR: Cloud Vision init failed: {e}")
                sys.exit(1)

        # Budget check (cached modes make no Vision calls)
        if not use_cached_ocr:
            image_count = sum(1 for f in fixtures if f[1].suffix != ".txt")
            estimated = _estimate_api_calls(image_count, runs)
            if not _check_budget(estimated, budget_limit, force):
                sys.exit(1)
    else:
        print("All fixtures use OCR text — no Cloud Vision needed.")

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        **_get_git_state(),
        "model": model,
        "runs_per_fixture": runs,
        "passes": passes,
        "workers": workers,
        "ci_mode": ci,
        "cached_ocr": use_cached_ocr,
        "save_variants": save_variants,
        "fixture_scope": "selected" if fixture_names is not None else "all_discovered",
        "fixture_filter": list(fixture_names) if fixture_names is not None else None,
        "fixture_count": n_fixtures,
        "fixture_corpus_sha256": _fixture_corpus_sha256(
            fixtures, cached_ocr=use_cached_ocr,
        ),
        "fixture_source_counts": {
            "image": sum(1 for _, source, _ in fixtures if source.suffix != ".txt"),
            "ocr_text": sum(1 for _, source, _ in fixtures if source.suffix == ".txt"),
        },
        "fixtures": fixture_name_list,
        "output_path": str(output_path),
        "artifact_dir": str(RESULTS_DIR / "artifacts" / run_id),
    }

    per_fixture: dict = {}
    skip_cache = not use_cached_ocr

    if workers > 1 and not ci:
        # Parallel execution across fixtures
        print(f"\nRunning {n_fixtures} fixtures with {workers} workers...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _run_fixture, name, source, truth, runs, model, passes, cv_client,
                    skip_cache, save_variants,
                ): name
                for name, source, truth in fixtures
            }
            for future in as_completed(futures):
                fname, fdata = future.result()
                per_fixture[fname] = fdata
    else:
        # Sequential
        for fix_idx, (name, source, truth) in enumerate(fixtures):
            print(f"\n[{fix_idx + 1}/{n_fixtures}] {name}")
            _, fdata = _run_fixture_sequential(
                name, source, truth, runs, model, passes, cv_client, skip_cache,
                save_variants,
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
    fixture_name, fixture_source, fixture_truth, runs, model, passes, cv_client, skip_cache,
    save_variants=True,
):
    """Sequential fixture runner with progress printing."""
    is_ocr_text = fixture_source.suffix == ".txt"
    checks = get_checks_for(fixture_truth)
    fixture_runs = []

    for run_idx in range(1, runs + 1):
        wall_start = time.perf_counter()
        error = None
        result = {}
        try:
            if is_ocr_text:
                ocr_text = fixture_source.read_text(encoding="utf-8")
                result = process_ocr_text(
                    ocr_text, model=model, passes=passes,
                    apply_user_rules=False,
                )
            else:
                digital_pdf = (
                    fixture_source.suffix.lower() == ".pdf"
                    and bool(try_extract_text_layer(str(fixture_source)))
                )
                result = process_document(
                    fixture_source, model=model, passes=passes,
                    apply_user_rules=False, skip_ocr_cache=skip_cache,
                    ocr_engine=cv_client,
                )
                if not skip_cache and not digital_pdf and result.get("_ocr_source") != "cache":
                    raise RuntimeError(
                        "Cached benchmark received non-cache OCR source: "
                        f"{result.get('_ocr_source', 'missing')}"
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
            "passed": error is None and pass_count == total_fields,
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
            "llm_raw": deepcopy(llm_raw),
            "llm_pass_history": deepcopy(pass_history),
            "final_extraction": _public_extraction(result),
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
    _finalize_fixture(fixture_name, fixture_data, save_variants=save_variants)
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
    parser.add_argument("--cached-ocr", action="store_true",
                        help="Use cached OCR without enabling CI pass/fail behavior")
    parser.add_argument("--no-save-variants", action="store_false", dest="save_variants",
                        help="Do not save failing OCR text as regression variants")
    args = parser.parse_args()

    compare_path = Path(args.compare) if args.compare else None
    if compare_path is not None and not compare_path.exists():
        print(f"Comparison file not found: {compare_path}")
        raise SystemExit(1)

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
        cached_ocr=args.cached_ocr,
        save_variants=args.save_variants,
    )

    if compare_path is not None:
        previous = json.loads(compare_path.read_text(encoding="utf-8"))
        if not _compare_results(results, previous):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
