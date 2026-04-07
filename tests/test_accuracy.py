"""Accuracy tests — field checks against cached pipeline results and OCR variants.

Auto-discovers:
1. Image fixtures: tests/fixtures/receipt_N.jpg + receipt_N_truth.json (cached OCR)
2. OCR variants: tests/ocr_variants/receipt_N_vM.txt (injected text, skips OCR)

Run with:
    python -m pytest tests/test_accuracy.py -v
    python -m pytest tests/test_accuracy.py -v --json-report --json-report-file=tests/results/accuracy/latest.json
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
VARIANTS = Path(__file__).resolve().parent.parent / ".data" / "ocr_cache" / "variants"

# Skip if Cloud Vision is not configured (needed for image fixtures)
_cv_available = True
try:
    if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        _cv_available = False
    else:
        from receipt_parser.ocr import init_cloud_vision
        init_cloud_vision()
except Exception:
    _cv_available = False


# ---------------------------------------------------------------------------
# Fixture discovery
# ---------------------------------------------------------------------------

def _find_image(base: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".pdf", ".tiff", ".bmp"):
        candidate = FIXTURES / f"{base}{ext}"
        if candidate.exists():
            return candidate
    return None


def _extract_base_name(variant_stem: str) -> str:
    """receipt_14_v1 -> receipt_14, receipt_1_v3 -> receipt_1."""
    return re.sub(r'_v\d+$', '', variant_stem)


def _discover_test_cases():
    cases = []

    # Image fixtures (process_document with cached OCR)
    if _cv_available:
        for truth_file in sorted(FIXTURES.glob("*_truth.json")):
            if truth_file.name == "_truth_template.json":
                continue
            base = truth_file.stem.replace("_truth", "")
            image = _find_image(base)
            if not image:
                continue
            truth = json.loads(truth_file.read_text(encoding="utf-8"))
            cases.append((base, {"type": "image", "path": image}, truth))

    # OCR variant fixtures (process_ocr_text, skip OCR)
    if VARIANTS.exists():
        for variant_file in sorted(VARIANTS.glob("*.txt")):
            stem = variant_file.stem
            base = _extract_base_name(stem)
            truth_file = FIXTURES / f"{base}_truth.json"
            if not truth_file.exists():
                continue
            truth = json.loads(truth_file.read_text(encoding="utf-8"))
            cases.append((stem, {"type": "ocr_text", "path": variant_file}, truth))

    return cases


_CASES = _discover_test_cases()
_CASE_IDS = [c[0] for c in _CASES]
_RESULTS_CACHE: dict[str, dict] = {}

# Collect check results for summary plugin
_check_results: list[dict] = []


def _process_one(case_id: str, source: dict) -> tuple[str, dict, float]:
    """Process a single fixture/variant. Thread-safe — each call is independent."""
    t0 = time.perf_counter()
    if source["type"] == "image":
        from receipt_parser.pipeline import process_document
        result = process_document(source["path"], passes=3, apply_user_rules=False)
    else:
        from receipt_parser.pipeline import process_ocr_text
        ocr_text = source["path"].read_text(encoding="utf-8")
        result = process_ocr_text(ocr_text, passes=3, apply_user_rules=False)
    elapsed = time.perf_counter() - t0
    return case_id, result, elapsed


def _get_result(case_id: str, source: dict) -> dict:
    if case_id not in _RESULTS_CACHE:
        _, result, _ = _process_one(case_id, source)
        _RESULTS_CACHE[case_id] = result
    return _RESULTS_CACHE[case_id]


@pytest.fixture(scope="session", autouse=True)
def preprocess_fixtures(request):
    """Pre-process all fixtures concurrently before tests run."""
    workers = request.config.getoption("--workers", default=4)
    if not _CASES:
        return

    if _cv_available:
        from receipt_parser.ocr import init_cloud_vision
        try:
            init_cloud_vision()
        except Exception:
            return

    n = len(_CASES)
    if workers <= 1:
        print(f"\nProcessing {n} fixtures sequentially...")
        for name, source, _truth in _CASES:
            _, result, elapsed = _process_one(name, source)
            _RESULTS_CACHE[name] = result
            print(f"  {name:25s} {elapsed:5.1f}s")
        return

    print(f"\nProcessing {n} fixtures with {workers} workers...")
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_one, name, source): name
            for name, source, _truth in _CASES
        }
        for future in as_completed(futures):
            case_id, result, elapsed = future.result()
            _RESULTS_CACHE[case_id] = result
            done += 1
            print(f"  [{done:2d}/{n}] {case_id:25s} {elapsed:5.1f}s")


# ---------------------------------------------------------------------------
# Parametrized test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,source,truth", _CASES, ids=_CASE_IDS)
def test_fields(name, source, truth):
    """Run all applicable field checks for this fixture/variant."""
    from receipt_parser.checks import get_checks_for

    result = _get_result(name, source)
    checks = get_checks_for(truth)
    failures = []
    for field_name, check_fn in checks.items():
        check_result = check_fn(result, truth)
        # Record for summary
        _check_results.append({
            "fixture": name,
            "field": field_name,
            "pass": check_result["pass"],
            "detail": check_result.get("detail", ""),
            "source_type": source["type"],
        })
        if not check_result["pass"]:
            failures.append(f"{field_name}: {check_result.get('detail', 'failed')}")

    assert not failures, "Failed checks:\n" + "\n".join(f"  - {f}" for f in failures)
