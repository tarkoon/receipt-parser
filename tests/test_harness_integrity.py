"""Offline checks for benchmark/accuracy scope and reproducibility metadata."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from receipt_parser.ocr import (
    load_ocr_layout_sidecar,
    load_ocr_replay_evidence,
    ocr_layout_sidecar_path,
    write_ocr_layout_sidecar,
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def benchmark_module():
    return _load_module(Path(__file__).with_name("benchmark.py"), "benchmark_harness_under_test")


@pytest.fixture(scope="module")
def accuracy_module():
    project = os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
    try:
        return _load_module(Path(__file__).with_name("test_accuracy.py"), "accuracy_harness_under_test")
    finally:
        if project is not None:
            os.environ["GOOGLE_CLOUD_PROJECT"] = project


def _image_and_cache(module, monkeypatch, tmp_path: Path):
    image_path = tmp_path / "receipt.png"
    assert cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(module, "OCR_CACHE_DIR", cache_dir)
    key = module._ocr_cache_key(module.load_image(image_path)[0])
    text_path = cache_dir / f"{key}.txt"
    layout_path = cache_dir / f"{key}.layout.json"
    text_path.write_text("one\ntwo\nthree\n", encoding="utf-8")
    layout_path.write_text('[{"text":"one","x":1}]', encoding="utf-8")
    return image_path, text_path, layout_path


@pytest.mark.parametrize("module_fixture", ["benchmark_module", "accuracy_module"])
def test_corpus_fingerprint_includes_cached_text_and_layout(
    request, module_fixture, monkeypatch, tmp_path,
):
    module = request.getfixturevalue(module_fixture)
    image_path, text_path, layout_path = _image_and_cache(module, monkeypatch, tmp_path)
    truth = {"document_type": "receipt", "total": 1}
    if module_fixture == "benchmark_module":
        corpus = [("receipt_x", image_path, truth)]
        fingerprint = lambda: module._fixture_corpus_sha256(corpus, cached_ocr=True)
    else:
        corpus = [("receipt_x", {"type": "image", "path": image_path}, truth)]
        fingerprint = lambda: module._corpus_sha256(corpus)

    original = fingerprint()
    layout_path.write_text('[{"text":"changed","x":2}]', encoding="utf-8")
    after_layout = fingerprint()
    text_path.write_text("changed\ntwo\nthree\n", encoding="utf-8")
    after_text = fingerprint()

    assert original != after_layout != after_text


@pytest.mark.parametrize("module_fixture", ["benchmark_module", "accuracy_module"])
def test_cached_image_preflight_allows_legacy_layout_but_rejects_bad_envelope(
    request, module_fixture, monkeypatch, tmp_path,
):
    module = request.getfixturevalue(module_fixture)
    image_path, text_path, layout_path = _image_and_cache(
        module, monkeypatch, tmp_path,
    )
    if module_fixture == "benchmark_module":
        cases = [("receipt_x", image_path, {})]
    else:
        cases = [("receipt_x", {"type": "image", "path": image_path}, {})]

    module._preflight_cached_image_layouts(cases)
    layout_path.write_text('[{"text":"one"},7]', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid cached OCR layout"):
        module._preflight_cached_image_layouts(cases)

    exact_text = text_path.read_text(encoding="utf-8")
    write_ocr_layout_sidecar(
        text_path,
        [{"text": "one", "x": 1}],
        provenance_kind="same_call_capture",
        provenance_source="cache-key",
        expected_ocr_text=exact_text,
    )
    text_path.write_text("changed\ntwo\nthree\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid cached OCR layout"):
        module._preflight_cached_image_layouts(cases)


@pytest.mark.parametrize("module_fixture", ["benchmark_module", "accuracy_module"])
def test_text_corpus_fingerprint_includes_layout_sidecar_presence_and_bytes(
    request, module_fixture, tmp_path,
):
    module = request.getfixturevalue(module_fixture)
    text_path = tmp_path / "receipt_7_v1.txt"
    text_path.write_text("exact OCR text", encoding="utf-8")
    truth = {"document_type": "receipt", "total": 1}
    if module_fixture == "benchmark_module":
        corpus = [("receipt_7_v1", text_path, truth)]
        fingerprint = lambda: module._fixture_corpus_sha256(corpus)
    else:
        corpus = [("receipt_7_v1", {"type": "ocr_text", "path": text_path}, truth)]
        fingerprint = lambda: module._corpus_sha256(corpus)

    missing = fingerprint()
    write_ocr_layout_sidecar(
        text_path,
        [{"text": "exact", "x": 1}],
        provenance_kind="same_call_capture",
        provenance_source="receipt_7.png",
    )
    present = fingerprint()
    sidecar = ocr_layout_sidecar_path(text_path)
    envelope = json.loads(sidecar.read_text(encoding="utf-8"))
    envelope["provenance"]["source"] = "other-origin.png"
    sidecar.write_text(json.dumps(envelope), encoding="utf-8")
    changed = fingerprint()

    assert missing != present != changed


def test_layout_sidecar_requires_exact_text_layout_hashes_and_provenance(tmp_path):
    text_path = tmp_path / "receipt_7_v1.txt"
    text_path.write_text("exact OCR text", encoding="utf-8")
    layout = [{"text": "exact", "x": 1, "y": 2}]
    assert load_ocr_layout_sidecar(text_path) is None
    sidecar = write_ocr_layout_sidecar(
        text_path,
        layout,
        provenance_kind="exact_text_cache_reuse",
        provenance_source="cache-key",
        ocr_confidence=0.73,
    )
    assert load_ocr_layout_sidecar(text_path) == layout
    assert load_ocr_replay_evidence(text_path)["ocr_confidence"] == 0.73

    text_path.write_text("different OCR text", encoding="utf-8")
    with pytest.raises(ValueError, match="text hash mismatch"):
        load_ocr_layout_sidecar(text_path)
    text_path.write_text("exact OCR text", encoding="utf-8")

    envelope = json.loads(sidecar.read_text(encoding="utf-8"))
    envelope["layout_blocks"][0]["x"] = 9
    sidecar.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="layout hash mismatch"):
        load_ocr_layout_sidecar(text_path)

    write_ocr_layout_sidecar(
        text_path,
        layout,
        provenance_kind="same_call_capture",
        provenance_source="receipt_7.png",
    )
    envelope = json.loads(sidecar.read_text(encoding="utf-8"))
    envelope["schema_version"] = True
    sidecar.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar schema"):
        load_ocr_layout_sidecar(text_path)

    write_ocr_layout_sidecar(
        text_path,
        layout,
        provenance_kind="same_call_capture",
        provenance_source="receipt_7.png",
    )
    envelope = json.loads(sidecar.read_text(encoding="utf-8"))
    envelope["provenance"]["kind"] = "similar_text_guess"
    sidecar.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance kind"):
        load_ocr_layout_sidecar(text_path)

    write_ocr_layout_sidecar(
        text_path,
        layout,
        provenance_kind="same_call_capture",
        provenance_source="receipt_7.png",
    )
    envelope = json.loads(sidecar.read_text(encoding="utf-8"))
    envelope["ocr_confidence"] = True
    sidecar.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid OCR layout sidecar"):
        load_ocr_replay_evidence(text_path)


def test_accuracy_preflight_rejects_corrupt_selected_sidecar(
    accuracy_module, tmp_path,
):
    text_path = tmp_path / "receipt_7_v1.txt"
    text_path.write_text("exact OCR", encoding="utf-8")
    ocr_layout_sidecar_path(text_path).write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid OCR layout sidecar"):
        accuracy_module._preflight_text_layouts([
            ("receipt_7_v1", {"type": "ocr_text", "path": text_path}, {}),
        ])


def test_benchmark_saves_and_replays_failing_variant_layout(
    benchmark_module, monkeypatch, tmp_path,
):
    variants = tmp_path / "variants"
    monkeypatch.setattr(benchmark_module, "VARIANTS_DIR", variants)
    layout = [{"text": "item", "x": 1, "y": 2}]

    path = benchmark_module._save_variant(
        "receipt_7",
        "failing OCR",
        layout,
        provenance_kind="same_call_capture",
        provenance_source="receipt_7.png",
    )

    assert path == variants / "receipt_7_v1.txt"
    assert load_ocr_layout_sidecar(path) == layout
    assert benchmark_module._save_variant(
        "receipt_7", "failing OCR", layout,
    ) is None

    changed_layout = [{"text": "item", "x": 9, "y": 2}]
    second = benchmark_module._save_variant(
        "receipt_7", "failing OCR", changed_layout,
    )
    assert second == variants / "receipt_7_v2.txt"
    assert load_ocr_layout_sidecar(second) == changed_layout

    reserved = variants / "receipt_7_v5.txt"
    reserved.write_text("do not overwrite", encoding="utf-8")
    sixth = benchmark_module._save_variant(
        "receipt_7", "another OCR", layout,
    )
    assert sixth == variants / "receipt_7_v6.txt"
    assert reserved.read_text(encoding="utf-8") == "do not overwrite"


def test_primary_cache_only_trusts_bound_layout_and_removes_stale_geometry(
    monkeypatch, tmp_path,
):
    from receipt_parser import ocr

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(ocr, "_OCR_CACHE_DIR", cache_dir)
    key = ocr._ocr_cache_key(image)
    text_path = cache_dir / f"{key}.txt"
    layout_path = ocr.ocr_layout_sidecar_path(text_path)
    legacy_layout = [{"text": "legacy", "x": 1}]
    text_path.write_text("legacy text", encoding="utf-8")
    layout_path.write_text(json.dumps(legacy_layout), encoding="utf-8")

    legacy = ocr.run_cloud_vision(image, client=object())
    assert legacy.layout_blocks == legacy_layout
    assert legacy.layout_trusted is False

    text_path.unlink()
    layout_path.unlink()
    fresh_layout = [{"text": "fresh", "x": 2}]
    monkeypatch.setattr(ocr, "_call_cloud_vision", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(ocr, "_extract_fulltext_from_response", lambda _response: "fresh text")
    monkeypatch.setattr(
        ocr,
        "_extract_blocks_from_response",
        lambda _response: [{"text": "fresh", "confidence": 0.9}],
    )
    monkeypatch.setattr(ocr, "_extract_words_from_response", lambda _response: fresh_layout)

    fresh = ocr.run_cloud_vision(image, client=object())
    assert fresh.layout_trusted is True
    assert load_ocr_layout_sidecar(text_path) == fresh_layout
    cached = ocr.run_cloud_vision(image, client=object())
    assert cached.layout_blocks == fresh_layout
    assert cached.layout_trusted is True

    text_path.unlink()
    monkeypatch.setattr(ocr, "_extract_words_from_response", lambda _response: [])
    without_layout = ocr.run_cloud_vision(image, client=object())
    assert without_layout.layout_blocks == []
    assert without_layout.layout_trusted is False
    assert not layout_path.exists()
    assert not list(cache_dir.glob("*.tmp"))


def test_cache_binding_uses_explicit_text_snapshot_during_interleaving(tmp_path):
    from receipt_parser import ocr

    text_path = tmp_path / "cache.txt"
    text_path.write_text("writer B", encoding="utf-8")
    layout = [{"text": "writer A", "x": 1}]
    ocr.write_ocr_layout_sidecar(
        text_path,
        layout,
        provenance_kind="same_call_capture",
        provenance_source="cache-key",
        expected_ocr_text="writer A",
    )

    with pytest.raises(ValueError, match="Invalid cached OCR layout"):
        ocr.load_cached_ocr_layout(
            text_path,
            expected_ocr_text="writer B",
            strict_envelope=True,
        )
    assert ocr.load_cached_ocr_layout(
        text_path,
        expected_ocr_text="writer A",
        strict_envelope=True,
    ) == (layout, True)


def test_variant_sidecar_failure_rolls_back_only_new_pair(
    benchmark_module, monkeypatch, tmp_path,
):
    variants = tmp_path / "variants"
    variants.mkdir()
    unrelated = variants / "receipt_7_v3.txt"
    unrelated.write_text("keep me", encoding="utf-8")
    monkeypatch.setattr(benchmark_module, "VARIANTS_DIR", variants)

    def fail_after_partial(text_path, *_args, **_kwargs):
        ocr_layout_sidecar_path(text_path).write_text("partial", encoding="utf-8")
        raise OSError("disk full")

    monkeypatch.setattr(benchmark_module, "write_ocr_layout_sidecar", fail_after_partial)
    with pytest.raises(OSError, match="disk full"):
        benchmark_module._save_variant(
            "receipt_7",
            "new OCR",
            [{"text": "new", "x": 1}],
        )

    assert unrelated.read_text(encoding="utf-8") == "keep me"
    assert not (variants / "receipt_7_v4.txt").exists()
    assert not (variants / "receipt_7_v4.layout.json").exists()


def test_benchmark_saves_scored_replay_text_layout_and_confidence(
    benchmark_module, monkeypatch, tmp_path,
):
    variants = tmp_path / "variants"
    image_path = tmp_path / "receipt_7.png"
    layout = [{"text": "ITEM A", "x": 1, "y": 2, "page": 0}]
    scored_text = "ITEM A    ITEM B\nTOTAL    100"
    provider_text = "ITEM A\nITEM B\nTOTAL 100"
    monkeypatch.setattr(benchmark_module, "VARIANTS_DIR", variants)
    monkeypatch.setattr(
        benchmark_module,
        "get_checks_for",
        lambda _truth: {
            "total": lambda result, _expected: {"pass": result["total"] == 100},
        },
    )
    monkeypatch.setattr(
        benchmark_module,
        "process_document",
        lambda *_args, **_kwargs: {
            "total": 0,
            "_ocr_text": provider_text,
            "_ocr_replay_text": scored_text,
            "_ocr_replay_confidence": 0.73,
            "_ocr_layout_blocks": layout,
            "_ocr_layout_trusted": True,
            "_ocr_page_count": 1,
            "_ocr_source": "fresh",
            "_warnings": [],
            "_pass_history": [],
        },
    )

    _, fixture = benchmark_module._run_fixture_sequential(
        "receipt_7", image_path, {"total": 100}, 1, "model", 1, None, True,
    )

    saved = variants / "receipt_7_v1.txt"
    evidence = load_ocr_replay_evidence(saved)
    assert fixture["variants_saved"] == 1
    assert saved.read_text(encoding="utf-8") == scored_text
    assert saved.read_text(encoding="utf-8") != provider_text
    assert evidence["layout_blocks"] == layout
    assert evidence["ocr_confidence"] == 0.73


def test_benchmark_forwards_valid_variant_layout(
    benchmark_module, monkeypatch, tmp_path,
):
    text_path = tmp_path / "receipt_7_v1.txt"
    text_path.write_text("exact OCR text", encoding="utf-8")
    layout = [{"text": "exact", "x": 1, "y": 2}]
    write_ocr_layout_sidecar(
        text_path,
        layout,
        provenance_kind="exact_text_cache_reuse",
        provenance_source="cache-key",
        ocr_confidence=0.73,
    )
    captured = {}

    def process(_text, **kwargs):
        captured["layout"] = kwargs.get("ocr_layout_blocks")
        captured["confidence"] = kwargs.get("ocr_confidence")
        return {"total": 1, "_ocr_source": "injected", "_ocr_text": _text}

    monkeypatch.setattr(benchmark_module, "process_ocr_text", process)
    monkeypatch.setattr(
        benchmark_module,
        "get_checks_for",
        lambda _truth: {"total": lambda result, _expected: {"pass": result["total"] == 1}},
    )
    benchmark_module._run_fixture_sequential(
        "receipt_7_v1", text_path, {"total": 1}, 1, "model", 1, None, False,
        save_variants=False,
    )

    assert captured == {"layout": layout, "confidence": 0.73}


def test_accuracy_forwards_valid_variant_layout(
    accuracy_module, monkeypatch, tmp_path,
):
    import receipt_parser.pipeline as pipeline

    text_path = tmp_path / "receipt_7_v1.txt"
    text_path.write_text("exact OCR text", encoding="utf-8")
    layout = [{"text": "exact", "x": 1, "y": 2}]
    write_ocr_layout_sidecar(
        text_path,
        layout,
        provenance_kind="exact_text_cache_reuse",
        provenance_source="cache-key",
        ocr_confidence=0.73,
    )
    captured = {}

    def process(_text, **kwargs):
        captured["layout"] = kwargs.get("ocr_layout_blocks")
        captured["confidence"] = kwargs.get("ocr_confidence")
        return {"total": 1}

    monkeypatch.setattr(pipeline, "process_ocr_text", process)
    accuracy_module._process_one(
        "receipt_7_v1", {"type": "ocr_text", "path": text_path},
    )

    assert captured == {"layout": layout, "confidence": 0.73}


def test_benchmark_archives_each_run_layout(benchmark_module, tmp_path):
    layout = [{"text": "item", "x": 1, "y": 2}]
    run = {
        "run": 1,
        "passed": True,
        "pass_count": 1,
        "total_fields": 1,
        "fields": {"total": {"pass": True}},
        "ocr": {
            "confidence": 0.73,
            "retried": False,
            "source": "fresh",
            "layout_trusted": True,
            "page_count": 1,
        },
        "ocr_text": "item",
        "ocr_provider_text": "provider\nitem",
        "ocr_layout_blocks": layout,
        "llm_raw": {},
        "llm_pass_history": [],
        "final_extraction": {"total": 1},
        "warnings": [],
        "wall_time_s": 0.1,
    }
    results = benchmark_module._assemble_results(
        {"artifact_dir": str(tmp_path / "artifacts")},
        {"receipt_7": {"runs": [run], "status": "ROBUST", "deterministic": True}},
    )
    archived = results["per_fixture"]["receipt_7"]["runs"][0]
    text_path = benchmark_module.RESULTS_DIR / archived["ocr"]["text_file"]
    provider_path = benchmark_module.RESULTS_DIR / archived["ocr"]["provider_text_file"]
    layout_path = benchmark_module.RESULTS_DIR / archived["ocr"]["layout_file"]

    assert text_path.read_text(encoding="utf-8") == "item"
    assert provider_path.read_text(encoding="utf-8") == "provider\nitem"
    assert layout_path == ocr_layout_sidecar_path(text_path)
    assert load_ocr_layout_sidecar(text_path) == layout
    assert load_ocr_replay_evidence(text_path)["ocr_confidence"] == 0.73
    assert "ocr_layout_blocks" not in archived


@pytest.mark.parametrize(
    "module_fixture,state_function",
    [("benchmark_module", "_get_git_state"), ("accuracy_module", "_git_state")],
)
def test_dirty_tree_fingerprint_includes_untracked_contents(
    request, module_fixture, state_function, monkeypatch, tmp_path,
):
    module = request.getfixturevalue(module_fixture)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    untracked = tmp_path / "new.txt"
    untracked.write_text("first", encoding="utf-8")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    first = getattr(module, state_function)()
    untracked.write_text("second", encoding="utf-8")
    second = getattr(module, state_function)()

    assert first["git_status_sha256"] == second["git_status_sha256"]
    assert first["git_untracked_sha256"] != second["git_untracked_sha256"]
    assert first["git_diff_sha256"] != second["git_diff_sha256"]


def test_explicit_filters_reject_empty_and_mixed_unknown(
    benchmark_module, accuracy_module, monkeypatch, tmp_path,
):
    fixtures = tmp_path / "fixtures"
    ocr = tmp_path / "ocr"
    fixtures.mkdir()
    ocr.mkdir()
    (fixtures / "receipt_good_public_truth.json").write_text(
        json.dumps({"document_type": "receipt"}), encoding="utf-8",
    )
    (ocr / "receipt_good.txt").write_text("cached text", encoding="utf-8")
    monkeypatch.setattr(benchmark_module, "FIXTURES_DIR", fixtures)
    monkeypatch.setattr(benchmark_module, "OCR_FIXTURES_DIR", ocr)

    with pytest.raises(ValueError, match="at least one"):
        benchmark_module._select_fixtures([])
    with pytest.raises(ValueError, match="receipt_typo"):
        benchmark_module._select_fixtures(["receipt_good", "receipt_typo"])

    cases = [("receipt_good", {"type": "ocr_text", "path": ocr / "receipt_good.txt"}, {})]
    with pytest.raises(ValueError, match="receipt_typo"):
        accuracy_module._resolve_requested_fixtures(cases, {"receipt_good", "receipt_typo"})
    monkeypatch.setenv("RECEIPT_FIXTURES", "   ")
    with pytest.raises(ValueError, match="selected no fixture names"):
        accuracy_module._requested_fixture_names()


def test_benchmark_selects_exact_variant_with_base_truth(
    benchmark_module, monkeypatch, tmp_path,
):
    fixtures = tmp_path / "fixtures"
    variants = tmp_path / "variants"
    fixtures.mkdir()
    variants.mkdir()
    truth = {"document_type": "receipt", "total": 123}
    (fixtures / "receipt_7_truth.json").write_text(
        json.dumps(truth), encoding="utf-8",
    )
    selected_variant = variants / "receipt_7_v2.txt"
    selected_variant.write_text("selected OCR", encoding="utf-8")
    (variants / "receipt_7_v1.txt").write_text("other OCR", encoding="utf-8")
    monkeypatch.setattr(benchmark_module, "FIXTURES_DIR", fixtures)
    monkeypatch.setattr(benchmark_module, "OCR_FIXTURES_DIR", tmp_path / "named")
    monkeypatch.setattr(benchmark_module, "VARIANTS_DIR", variants)

    assert benchmark_module.discover_fixtures() == []
    selected = benchmark_module._select_fixtures(["receipt_7_v2"])

    assert selected == [("receipt_7_v2", selected_variant, truth)]


def test_benchmark_can_disable_variant_autosave(benchmark_module, monkeypatch):
    run = {
        "passed": False,
        "pass_count": 0,
        "total_fields": 1,
        "fields": {"total": {"pass": False}},
        "ocr": {"confidence": None, "retried": False},
        "ocr_text": "failing OCR",
    }
    monkeypatch.setattr(
        benchmark_module, "_save_variant",
        lambda *_args, **_kwargs: pytest.fail("variant autosave was not disabled"),
    )

    fixture = {"runs": [run]}
    benchmark_module._finalize_fixture("receipt_7", fixture, save_variants=False)

    assert fixture["variants_saved"] == 0


def test_benchmark_suppresses_variant_autosave_for_multi_page_source(
    benchmark_module, monkeypatch,
):
    run = {
        "passed": False,
        "pass_count": 0,
        "total_fields": 1,
        "fields": {"total": {"pass": False}},
        "ocr": {
            "confidence": 0.9,
            "retried": False,
            "page_count": 2,
            "layout_trusted": True,
        },
        "ocr_text": "first page only",
        "ocr_layout_blocks": [{"text": "first", "x": 1}],
    }
    monkeypatch.setattr(
        benchmark_module,
        "_save_variant",
        lambda *_args, **_kwargs: pytest.fail("multi-page OCR was autosaved"),
    )

    fixture = {"runs": [run]}
    benchmark_module._finalize_fixture("receipt_7", fixture)

    assert fixture["variants_saved"] == 0


def test_benchmark_preflights_corrupt_sidecar_before_model_or_scoring(
    benchmark_module, monkeypatch, tmp_path,
):
    text_path = tmp_path / "receipt_7_v1.txt"
    text_path.write_text("exact OCR", encoding="utf-8")
    ocr_layout_sidecar_path(text_path).write_text("{}", encoding="utf-8")
    fixture = ("receipt_7_v1", text_path, {"total": 1})
    monkeypatch.setattr(benchmark_module, "_select_fixtures", lambda _names: [fixture])
    monkeypatch.setattr(
        benchmark_module,
        "check_model_available",
        lambda _model: pytest.fail("model preflight ran after corrupt sidecar"),
    )
    monkeypatch.setattr(
        benchmark_module,
        "_run_fixture_sequential",
        lambda *_args: pytest.fail("corrupt sidecar reached scoring"),
    )

    with pytest.raises(SystemExit) as exc:
        benchmark_module.run_benchmark(
            fixture_names=["receipt_7_v1"],
            output_path=tmp_path / "result.json",
        )

    assert exc.value.code == 1


@pytest.mark.parametrize("workers", [1, 2])
def test_cached_ocr_mode_keeps_run_count_and_never_initializes_vision(
    benchmark_module, monkeypatch, tmp_path, workers,
):
    image = tmp_path / "receipt.png"
    image.write_bytes(b"fixture bytes")
    fixture = ("receipt_7", image, {"total": 123})
    calls = []

    monkeypatch.setattr(benchmark_module, "_select_fixtures", lambda _names: [fixture])
    monkeypatch.setattr(benchmark_module, "_missing_cached_ocr", lambda _fixtures: [])
    monkeypatch.setattr(
        benchmark_module, "_preflight_cached_image_layouts", lambda _fixtures: None,
    )
    monkeypatch.setattr(benchmark_module, "check_model_available", lambda _model: None)
    monkeypatch.setattr(
        benchmark_module, "init_cloud_vision",
        lambda: pytest.fail("cached OCR initialized Cloud Vision"),
    )
    monkeypatch.setattr(benchmark_module, "_get_git_state", lambda: {})
    monkeypatch.setattr(
        benchmark_module, "_fixture_corpus_sha256",
        lambda _fixtures, *, cached_ocr: "cached" if cached_ocr else "fresh",
    )

    def capture_runner(*args):
        calls.append(args)
        return args[0], {"runs": []}

    monkeypatch.setattr(benchmark_module, "_run_fixture", capture_runner)
    monkeypatch.setattr(benchmark_module, "_run_fixture_sequential", capture_runner)
    monkeypatch.setattr(
        benchmark_module, "_assemble_results",
        lambda metadata, _fixtures: {"metadata": metadata, "summary": {"fragile": []}},
    )
    monkeypatch.setattr(benchmark_module, "_save_results", lambda *_args: None)
    monkeypatch.setattr(benchmark_module, "_print_summary", lambda *_args: None)

    result = benchmark_module.run_benchmark(
        runs=7, workers=workers, cached_ocr=True, output_path=tmp_path / "result.json",
    )

    assert len(calls) == 1
    assert calls[0][3] == 7
    assert isinstance(calls[0][6], benchmark_module._CacheOnlyOCREngine)
    assert calls[0][7] is False
    assert result["metadata"]["runs_per_fixture"] == 7
    assert result["metadata"]["cached_ocr"] is True
    assert result["metadata"]["ci_mode"] is False


def test_cached_ocr_mode_fails_closed_when_cache_is_missing(
    benchmark_module, monkeypatch, tmp_path,
):
    image = tmp_path / "receipt.png"
    image.write_bytes(b"fixture bytes")
    missing = tmp_path / "missing-cache.txt"
    monkeypatch.setattr(
        benchmark_module, "_select_fixtures",
        lambda _names: [("receipt_7", image, {"total": 123})],
    )
    monkeypatch.setattr(benchmark_module, "_missing_cached_ocr", lambda _fixtures: [missing])
    monkeypatch.setattr(
        benchmark_module, "init_cloud_vision",
        lambda: pytest.fail("missing cached OCR fell back to Cloud Vision"),
    )

    with pytest.raises(SystemExit) as exc:
        benchmark_module.run_benchmark(cached_ocr=True, output_path=tmp_path / "result.json")

    assert exc.value.code == 1


def test_accuracy_discovers_cached_images_without_vision_configuration(
    accuracy_module, monkeypatch, tmp_path,
):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    image_path = fixtures / "receipt_1.png"
    assert cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))
    (fixtures / "receipt_1_truth.json").write_text("{}", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(accuracy_module, "FIXTURES", fixtures)
    monkeypatch.setattr(accuracy_module, "OCR_FIXTURES", tmp_path / "missing-ocr")
    monkeypatch.setattr(accuracy_module, "VARIANTS", tmp_path / "missing-variants")
    monkeypatch.setattr(accuracy_module, "OCR_CACHE_DIR", cache_dir)
    monkeypatch.setattr(accuracy_module, "_cv_available", False)
    key = accuracy_module._ocr_cache_key(accuracy_module.load_image(image_path)[0])
    cached_text = cache_dir / f"{key}.txt"
    cached_text.write_text("one\ntwo\nthree\n", encoding="utf-8")

    cases, unavailable = accuracy_module._discover_test_cases_with_exclusions()

    assert [case[0] for case in cases] == ["receipt_1"]
    assert cases[0][1]["type"] == "image"
    assert unavailable == []

    cached_text.unlink()
    assert accuracy_module._discover_test_cases_with_exclusions() == ([], ["receipt_1"])


def test_cached_runs_reject_fresh_ocr(benchmark_module, accuracy_module, monkeypatch, tmp_path):
    import receipt_parser.pipeline as pipeline

    image_path = tmp_path / "receipt.png"
    image_path.write_bytes(b"not read by patched processor")
    monkeypatch.setattr(
        benchmark_module, "process_document",
        lambda *args, **kwargs: {"total": 1, "_ocr_source": "fresh"},
    )
    monkeypatch.setattr(
        benchmark_module, "get_checks_for",
        lambda truth: {"total": lambda result, expected: {"pass": result.get("total") == 1}},
    )
    _name, fixture = benchmark_module._run_fixture_sequential(
        "receipt_x", image_path, {"total": 1}, 1, "model", 1, None, False,
    )
    assert fixture["status"] == "FRAGILE"
    assert "non-cache OCR source" in fixture["runs"][0]["error"]

    monkeypatch.setattr(
        pipeline, "process_document", lambda *args, **kwargs: {"_ocr_source": "fresh"},
    )
    with pytest.raises(RuntimeError, match="non-cache OCR source"):
        accuracy_module._process_one("receipt_x", {"type": "image", "path": image_path})


def _scope(corpus: str):
    return {
        "metadata": {
            "fixtures": ["receipt_x"],
            "fixture_corpus_sha256": corpus,
            "model": "model",
            "passes": 1,
            "runs_per_fixture": 1,
            "ci_mode": True,
        },
        "summary": {},
    }


def test_compare_cli_exits_nonzero_for_missing_file_and_scope_mismatch(
    benchmark_module, monkeypatch, tmp_path,
):
    current = _scope("current")
    monkeypatch.setattr(benchmark_module, "run_benchmark", lambda **kwargs: current)

    missing = tmp_path / "missing.json"
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--compare", str(missing)])
    with pytest.raises(SystemExit) as missing_exit:
        benchmark_module.main()
    assert missing_exit.value.code == 1

    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps(_scope("previous")), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--compare", str(previous)])
    with pytest.raises(SystemExit) as mismatch_exit:
        benchmark_module.main()
    assert mismatch_exit.value.code == 1
