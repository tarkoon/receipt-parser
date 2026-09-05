"""Accuracy tests — field checks against cached pipeline results and OCR variants.

Auto-discovers:
1. Image fixtures: tests/fixtures/receipt_N.jpg + receipt_N_truth.json (full pipeline)
2. Public fixtures: tests/fixtures/receipt_N_public_truth.json + .data/ocr_cache/named/receipt_N.txt
   (anonymized, committed to git, skips Cloud Vision OCR)
3. OCR variants: .data/ocr_cache/variants/receipt_N_vM.txt (injected text, skips OCR)

When images are available, uses process_document() with the original truth file.
When only public fixtures exist, falls back to process_ocr_text() with anonymized data.

Run with:
    python -m pytest tests/test_accuracy.py -v
    python -m pytest tests/test_accuracy.py -v --json-report --json-report-file=tests/results/accuracy/latest.json
"""

import hashlib
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import pytest
from receipt_parser.llm import DEFAULT_MODEL
from receipt_parser.normalize import normalize_fullwidth, strip_barcode_lines
from receipt_parser.ocr import (
    _OCR_CACHE_DIR,
    _ocr_cache_key,
    load_cached_ocr_layout,
    load_ocr_replay_evidence,
    ocr_layout_sidecar_path,
)
from receipt_parser.preprocess import load_image, try_extract_text_layer
from receipt_parser.pipeline import detect_document_type
from receipt_parser.receipt_supplemental_ocr import (
    SUPPLEMENTAL_OCR_CACHE_DIR,
    SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT,
    load_supplemental_ocr_evidence,
    supplemental_ocr_cache_path,
)

FIXTURES = Path(__file__).parent / "fixtures"
OCR_FIXTURES = Path(__file__).parent / "ocr_fixtures"
VARIANTS = Path(__file__).resolve().parent.parent / ".data" / "ocr_cache" / "variants"
REPO_ROOT = Path(__file__).resolve().parent.parent
OCR_CACHE_DIR = _OCR_CACHE_DIR
ACCURACY_PASSES = 3


class _CacheOnlyOCREngine:
    def annotate_image(self, **_kwargs):
        raise RuntimeError("Cached OCR required; refusing a Vision API call")

    def document_text_detection(self, **_kwargs):
        raise RuntimeError("Cached OCR required; refusing a Vision API call")


_CACHE_ONLY_OCR = _CacheOnlyOCREngine()

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


def _discover_test_cases_with_exclusions():
    cases = []
    discovered_bases = set()
    unavailable_images = []

    # Image fixtures (process_document with original truth, full pipeline)
    for truth_file in sorted(FIXTURES.glob("*_truth.json")):
        if truth_file.name == "_truth_template.json":
            continue
        if "_public_truth" in truth_file.name:
            continue
        base = truth_file.stem.replace("_truth", "")
        image = _find_image(base)
        if not image:
            continue
        truth = json.loads(truth_file.read_text(encoding="utf-8"))
        case = (base, {"type": "image", "path": image}, truth)
        if _missing_cached_ocr([case]):
            unavailable_images.append(base)
            continue
        cases.append(case)
        discovered_bases.add(base)

    # Public fixtures (anonymized truth + named OCR text, no images needed)
    for truth_file in sorted(FIXTURES.glob("*_public_truth.json")):
        base = truth_file.stem.replace("_public_truth", "")
        if base in discovered_bases:
            continue  # already discovered as image fixture
        ocr_file = OCR_FIXTURES / f"{base}.txt"
        if not ocr_file.exists():
            continue
        truth = json.loads(truth_file.read_text(encoding="utf-8"))
        cases.append((base, {"type": "ocr_text", "path": ocr_file}, truth))
        discovered_bases.add(base)

    # OCR variant fixtures (process_ocr_text, skip OCR)
    if VARIANTS.exists():
        for variant_file in sorted(VARIANTS.glob("*.txt")):
            stem = variant_file.stem
            base = _extract_base_name(stem)
            truth_file = FIXTURES / f"{base}_truth.json"
            if not truth_file.exists():
                truth_file = FIXTURES / f"{base}_public_truth.json"
            if not truth_file.exists():
                continue
            truth = json.loads(truth_file.read_text(encoding="utf-8"))
            cases.append((stem, {"type": "ocr_text", "path": variant_file}, truth))

    return cases, unavailable_images


def _discover_test_cases():
    return _discover_test_cases_with_exclusions()[0]


def _receipt_number(case_id: str) -> int | None:
    m = re.match(r'^receipt_(\d+)(?:_|$)', case_id)
    return int(m.group(1)) if m else None


def _receipt_ceiling() -> int | None:
    raw = os.environ.get("RECEIPT_MAX_FIXTURE")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid RECEIPT_MAX_FIXTURE: {raw!r}") from exc


def _requested_fixture_names() -> set[str] | None:
    raw = os.environ.get("RECEIPT_FIXTURES")
    if raw is None:
        return None
    requested = {
        name.strip()
        for name in re.split(r'[,;\s]+', raw)
        if name.strip()
    }
    if not requested:
        raise ValueError("RECEIPT_FIXTURES was set but selected no fixture names")
    return requested


def _filter_by_receipt_ceiling(cases):
    limit = _receipt_ceiling()
    if limit is None:
        return cases
    return [
        case for case in cases
        if (num := _receipt_number(case[0])) is None or num <= limit
    ]


def _resolve_requested_fixtures(cases, requested: set[str] | None):
    if requested is None:
        return cases

    def _matches(case_id: str) -> bool:
        base = _extract_base_name(case_id)
        return case_id in requested or base in requested

    selected = [case for case in cases if _matches(case[0])]
    resolved = {
        requested_name
        for requested_name in requested
        if any(
            case_id == requested_name or _extract_base_name(case_id) == requested_name
            for case_id, _source, _truth in cases
        )
    }
    unresolved = sorted(requested - resolved)
    if unresolved:
        raise ValueError(f"Unknown or unavailable fixture(s): {', '.join(unresolved)}")
    if not selected:
        raise ValueError("Fixture filters selected no cases")
    return selected


def _filter_by_requested_fixtures(cases):
    return _resolve_requested_fixtures(cases, _requested_fixture_names())


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


def _supplemental_ocr_artifact(
    case_id: str,
    source: dict,
) -> tuple[Path, dict] | None:
    """Resolve evidence only for one-page scanned receipt inputs/replays."""
    image_source = source["path"]
    if source["type"] == "ocr_text":
        image_source = _find_image(_extract_base_name(case_id))
        if image_source is None:
            return None
    if (
        image_source.suffix.lower() == ".pdf"
        and try_extract_text_layer(str(image_source))
    ):
        return None
    images = load_image(image_source)
    if len(images) != 1:
        return None
    image = images[0]
    if source["type"] == "ocr_text":
        classification_text = source["path"].read_text(encoding="utf-8")
    else:
        classification_text = _selected_cached_ocr_text(image)
        if classification_text is None:
            return None
    classification_text = strip_barcode_lines(
        normalize_fullwidth(classification_text),
    )
    if detect_document_type(classification_text) != "receipt":
        return None
    identity = {
        "image_key": _ocr_cache_key(image),
        "image_shape": list(image.shape[:2]),
        "strategy_fingerprint": SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT,
    }
    return (
        supplemental_ocr_cache_path(SUPPLEMENTAL_OCR_CACHE_DIR, **identity),
        identity,
    )


def _selected_cached_ocr_text(image) -> str | None:
    """Mirror the cached one-page rotation choice used by process_document."""
    text_path = OCR_CACHE_DIR / f"{_ocr_cache_key(image)}.txt"
    if not text_path.is_file():
        return None
    best_text = text_path.read_text(encoding="utf-8")
    best_count = len([line for line in best_text.splitlines() if line.strip()])
    if best_count >= 3:
        return "\n".join(line.strip() for line in best_text.splitlines() if line.strip())
    for rotation in (
        cv2.ROTATE_90_CLOCKWISE,
        cv2.ROTATE_180,
        cv2.ROTATE_90_COUNTERCLOCKWISE,
    ):
        rotated_path = OCR_CACHE_DIR / f"{_ocr_cache_key(cv2.rotate(image, rotation))}.txt"
        if not rotated_path.is_file():
            return None
        rotated_text = rotated_path.read_text(encoding="utf-8")
        rotated_count = len([
            line for line in rotated_text.splitlines() if line.strip()
        ])
        if rotated_count > best_count:
            best_text = rotated_text
            best_count = rotated_count
        if best_count:
            break
    return "\n".join(line.strip() for line in best_text.splitlines() if line.strip())


def _update_path_fingerprint(digest, label: str, path: Path) -> None:
    digest.update(label.encode())
    digest.update(b"\0")
    if path.is_file():
        digest.update(b"present\0")
        digest.update(path.read_bytes())
    else:
        digest.update(b"missing\0")


def _corpus_sha256(cases) -> str:
    digest = hashlib.sha256()
    for case_id, source, truth in cases:
        digest.update(case_id.encode())
        digest.update(source["type"].encode())
        digest.update(source["path"].read_bytes())
        digest.update(json.dumps(truth, ensure_ascii=False, sort_keys=True).encode())
        if source["type"] == "image":
            for label, path in _cached_ocr_artifacts(source["path"]):
                _update_path_fingerprint(digest, label, path)
        else:
            _update_path_fingerprint(
                digest,
                "ocr_layout_sidecar",
                ocr_layout_sidecar_path(source["path"]),
            )
        supplemental = _supplemental_ocr_artifact(case_id, source)
        if supplemental is not None:
            path, _identity = supplemental
            _update_path_fingerprint(
                digest, f"supplemental_ocr:{path.name}", path,
            )
    return digest.hexdigest()


def _missing_cached_ocr(cases) -> list[Path]:
    return [
        path
        for _case_id, source, _truth in cases
        if source["type"] == "image"
        for label, path in _cached_ocr_artifacts(source["path"])
        if label.endswith("ocr_text") and not path.is_file()
    ]


def _preflight_text_layouts(cases) -> dict[Path, dict | None]:
    """Validate every selected text sidecar before any result is scored."""
    return {
        source["path"]: load_ocr_replay_evidence(source["path"])
        for _case_id, source, _truth in cases
        if source["type"] == "ocr_text"
    }


def _preflight_cached_image_layouts(cases) -> None:
    """Reject corrupt bound cache geometry before accuracy scoring starts."""
    for _case_id, source, _truth in cases:
        if source["type"] != "image":
            continue
        for label, layout_path in _cached_ocr_artifacts(source["path"]):
            if not label.endswith("ocr_layout") or not layout_path.is_file():
                continue
            text_path = layout_path.with_name(
                layout_path.name.removesuffix(".layout.json") + ".txt"
            )
            load_cached_ocr_layout(
                text_path,
                expected_ocr_text=text_path.read_text(encoding="utf-8"),
                strict_envelope=True,
            )


def _preflight_supplemental_ocr(cases) -> dict[Path, dict]:
    """Validate required image-bound evidence before accuracy scoring."""
    replay_evidence = {}
    for case_id, source, truth in cases:
        artifact = _supplemental_ocr_artifact(case_id, source)
        if artifact is None:
            continue
        path, identity = artifact
        evidence = load_supplemental_ocr_evidence(
            SUPPLEMENTAL_OCR_CACHE_DIR, **identity,
        )
        if evidence is None:
            state = "Invalid" if path.is_file() else "Missing"
            raise ValueError(f"{state} supplemental OCR sidecar: {path}")
        if source["type"] == "ocr_text":
            replay_evidence[source["path"]] = evidence
    return replay_evidence


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


def _git_state() -> dict:
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


try:
    _REQUESTED_FIXTURES = _requested_fixture_names()
    _RECEIPT_CEILING = _receipt_ceiling()
    _ALL_CASES, _UNAVAILABLE_IMAGE_CASES = _discover_test_cases_with_exclusions()
    _CASES = _resolve_requested_fixtures(
        _filter_by_receipt_ceiling(_ALL_CASES), _REQUESTED_FIXTURES,
    )
    if (_REQUESTED_FIXTURES is not None or _RECEIPT_CEILING is not None) and not _CASES:
        raise ValueError("Fixture filters selected no cases")
except ValueError as exc:
    raise pytest.UsageError(str(exc)) from exc

_CASE_IDS = [c[0] for c in _CASES]
_FILTERED_SCOPE = _REQUESTED_FIXTURES is not None or _RECEIPT_CEILING is not None
_ACCURACY_SCOPE = {
    **_git_state(),
    "fixture_scope": (
        "selected" if _FILTERED_SCOPE
        else "available_discovered" if _UNAVAILABLE_IMAGE_CASES
        else "all_discovered"
    ),
    "fixture_filter": sorted(_REQUESTED_FIXTURES) if _REQUESTED_FIXTURES else None,
    "receipt_max_fixture": _RECEIPT_CEILING,
    "available_case_count": len(_ALL_CASES),
    "unavailable_image_case_count": len(_UNAVAILABLE_IMAGE_CASES),
    "discovery_exclusions": {
        "cached_ocr_missing": len(_UNAVAILABLE_IMAGE_CASES),
    },
    "selected_case_count": len(_CASES),
    "selected_case_ids_sha256": hashlib.sha256("\n".join(_CASE_IDS).encode()).hexdigest(),
    "fixture_corpus_sha256": _corpus_sha256(_CASES),
    "source_counts": {
        source_type: sum(1 for _, source, _ in _CASES if source["type"] == source_type)
        for source_type in ("image", "ocr_text")
    },
    "cloud_vision_available": _cv_available,
    "model": DEFAULT_MODEL,
    "passes": ACCURACY_PASSES,
    "apply_user_rules": False,
}
_RESULTS_CACHE: dict[str, dict] = {}
_OCR_TEXT_LAYOUTS: dict[Path, dict | None] = {}
_SUPPLEMENTAL_OCR_EVIDENCE: dict[Path, dict] = {}

# Collect check results for summary plugin
_check_results: list[dict] = []


def _process_one(case_id: str, source: dict) -> tuple[str, dict, float]:
    """Process a single fixture/variant. Thread-safe — each call is independent."""
    t0 = time.perf_counter()
    if source["type"] == "image":
        from receipt_parser.pipeline import process_document
        digital_pdf = (
            source["path"].suffix.lower() == ".pdf"
            and bool(try_extract_text_layer(str(source["path"])))
        )
        result = process_document(
            source["path"], passes=ACCURACY_PASSES, apply_user_rules=False,
            ocr_engine=_CACHE_ONLY_OCR, ocr_cache_only=True,
        )
        if not digital_pdf and result.get("_ocr_source") != "cache":
            raise RuntimeError(
                "Cached accuracy run received non-cache OCR source: "
                f"{result.get('_ocr_source', 'missing')}"
            )
    else:
        from receipt_parser.pipeline import process_ocr_text
        ocr_text = source["path"].read_text(encoding="utf-8")
        replay_evidence = (
            _OCR_TEXT_LAYOUTS[source["path"]]
            if source["path"] in _OCR_TEXT_LAYOUTS
            else load_ocr_replay_evidence(source["path"])
        )
        result = process_ocr_text(
            ocr_text, passes=ACCURACY_PASSES, apply_user_rules=False,
            ocr_layout_blocks=(
                replay_evidence["layout_blocks"] if replay_evidence else None
            ),
            ocr_confidence=(
                replay_evidence["ocr_confidence"] if replay_evidence else None
            ),
            supplemental_ocr_evidence=_SUPPLEMENTAL_OCR_EVIDENCE.get(
                source["path"],
            ),
        )
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
    global _OCR_TEXT_LAYOUTS, _SUPPLEMENTAL_OCR_EVIDENCE
    workers = request.config.getoption("--workers", default=4)
    scope = {**_ACCURACY_SCOPE, "workers": workers}
    request.config._metadata = {
        **(getattr(request.config, "_metadata", {}) or {}),
        "accuracy_scope": scope,
    }
    if not _CASES:
        return

    try:
        _OCR_TEXT_LAYOUTS = _preflight_text_layouts(_CASES)
        _SUPPLEMENTAL_OCR_EVIDENCE = _preflight_supplemental_ocr(_CASES)
        _preflight_cached_image_layouts(_CASES)
    except ValueError as exc:
        pytest.fail(str(exc), pytrace=False)

    missing_cache = _missing_cached_ocr(_CASES)
    if missing_cache:
        pytest.fail(
            f"Cached OCR is missing for {len(missing_cache)} image page(s)",
            pytrace=False,
        )

    n = len(_CASES)
    print(
        f"\nAccuracy scope: {_ACCURACY_SCOPE['fixture_scope']}, "
        f"{n}/{_ACCURACY_SCOPE['available_case_count']} discovered cases, "
        f"passes={ACCURACY_PASSES}, git_dirty={_ACCURACY_SCOPE['git_dirty']}, "
        f"diff={str(_ACCURACY_SCOPE['git_diff_sha256'] or 'unknown')[:12]}"
    )
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
