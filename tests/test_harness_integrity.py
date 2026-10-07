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
from receipt_parser.receipt_supplemental_ocr import (
    _LEGACY_STRATEGY_FINGERPRINT,
    build_supplemental_ocr_evidence,
    save_supplemental_ocr_evidence,
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


def _write_supplemental_sidecar(module, monkeypatch, image_path: Path, tmp_path: Path, *, legacy=False):
    cache_dir = tmp_path / "supplemental"
    monkeypatch.setattr(module, "SUPPLEMENTAL_OCR_CACHE_DIR", cache_dir)
    image = module.load_image(image_path)[0]
    layout = [{
        "text": "SUPPLEMENT",
        "x": 1,
        "y": 1,
        "bbox": [[1, 1], [2, 1], [2, 2], [1, 2]],
        "confidence": 0.9,
        "page": 0,
    }]
    evidence = build_supplemental_ocr_evidence(
        image_key=module._ocr_cache_key(image),
        image_shape=image.shape[:2],
        strategy_fingerprint=_LEGACY_STRATEGY_FINGERPRINT if legacy else module.SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT,
        merged_text="SUPPLEMENT",
        source_layout=layout,
        tile_texts=["top", "bottom"],
    )
    return evidence, save_supplemental_ocr_evidence(cache_dir, evidence)


@pytest.mark.parametrize("module_fixture", ["benchmark_module", "accuracy_module"])
def test_corpus_fingerprint_includes_cached_text_and_layout(
    request, module_fixture, monkeypatch, tmp_path,
):
    module = request.getfixturevalue(module_fixture)
    image_path, text_path, layout_path = _image_and_cache(module, monkeypatch, tmp_path)
    truth = {"document_type": "utility_bill", "total": 1}
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
@pytest.mark.parametrize("legacy", [False, True])
def test_corpus_fingerprint_includes_supplemental_sidecar_bytes(
    request, module_fixture, monkeypatch, tmp_path, legacy,
):
    module = request.getfixturevalue(module_fixture)
    image_path, _text_path, _layout_path = _image_and_cache(
        module, monkeypatch, tmp_path,
    )
    supplemental_dir = tmp_path / "supplemental"
    monkeypatch.setattr(module, "SUPPLEMENTAL_OCR_CACHE_DIR", supplemental_dir)
    truth = {"document_type": "receipt", "total": 1}
    if module_fixture == "benchmark_module":
        corpus = [("receipt_x", image_path, truth)]
        fingerprint = lambda: module._fixture_corpus_sha256(
            corpus, cached_ocr=True,
        )
    else:
        corpus = [("receipt_x", {"type": "image", "path": image_path}, truth)]
        fingerprint = lambda: module._corpus_sha256(corpus)

    missing = fingerprint()
    _evidence, sidecar = _write_supplemental_sidecar(
        module, monkeypatch, image_path, tmp_path, legacy=legacy,
    )
    present = fingerprint()
    sidecar.write_text("{}", encoding="utf-8")
    changed = fingerprint()

    assert missing != present != changed


@pytest.mark.parametrize("module_fixture", ["benchmark_module", "accuracy_module"])
@pytest.mark.parametrize("legacy", [False, True])
def test_supplemental_preflight_binds_variant_to_base_image_and_fails_closed(
    request, module_fixture, monkeypatch, tmp_path, legacy,
):
    module = request.getfixturevalue(module_fixture)
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    image_path = fixtures_dir / "receipt_7.png"
    assert cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))
    text_path = tmp_path / "receipt_7_v1.txt"
    text_path.write_text("variant OCR", encoding="utf-8")
    truth = {"document_type": "utility_bill"}
    if module_fixture == "benchmark_module":
        monkeypatch.setattr(module, "FIXTURES_DIR", fixtures_dir)
        cases = [("receipt_7_v1", text_path, truth)]
        preflight = lambda: module._preflight_supplemental_ocr(
            cases, cached_images=False,
        )
    else:
        monkeypatch.setattr(module, "FIXTURES", fixtures_dir)
        cases = [("receipt_7_v1", {
            "type": "ocr_text", "path": text_path,
        }, truth)]
        preflight = lambda: module._preflight_supplemental_ocr(cases)

    monkeypatch.setattr(
        module, "SUPPLEMENTAL_OCR_CACHE_DIR", tmp_path / "supplemental",
    )
    with pytest.raises(ValueError, match="Missing supplemental OCR sidecar"):
        preflight()

    evidence, sidecar = _write_supplemental_sidecar(
        module, monkeypatch, image_path, tmp_path, legacy=legacy,
    )
    if legacy:
        evidence["strategy_fingerprint"] = module.SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT
    assert preflight() == {text_path: evidence}

    sidecar.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid supplemental OCR sidecar"):
        preflight()

    truth["document_type"] = "receipt"
    text_path.write_text("検針\n使用量\nご請求額", encoding="utf-8")
    assert preflight() == {}


@pytest.mark.parametrize("module_fixture", ["benchmark_module", "accuracy_module"])
def test_cached_supplemental_preflight_uses_ocr_classifier_not_truth(
    request, module_fixture, monkeypatch, tmp_path,
):
    module = request.getfixturevalue(module_fixture)
    image_path, text_path, _layout_path = _image_and_cache(
        module, monkeypatch, tmp_path,
    )
    monkeypatch.setattr(
        module, "SUPPLEMENTAL_OCR_CACHE_DIR", tmp_path / "supplemental",
    )
    truth = {"document_type": "utility_bill"}
    if module_fixture == "benchmark_module":
        cases = [("receipt_x", image_path, truth)]
        preflight = lambda: module._preflight_supplemental_ocr(
            cases, cached_images=True,
        )
    else:
        cases = [("receipt_x", {"type": "image", "path": image_path}, truth)]
        preflight = lambda: module._preflight_supplemental_ocr(cases)

    with pytest.raises(ValueError, match="Missing supplemental OCR sidecar"):
        preflight()

    evidence, _sidecar = _write_supplemental_sidecar(
        module, monkeypatch, image_path, tmp_path,
    )
    assert preflight() == {}

    truth["document_type"] = "receipt"
    text_path.write_text("検針\n使用量\nご請求額", encoding="utf-8")
    assert preflight() == {}


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


@pytest.mark.parametrize("runner", ["_run_fixture", "_run_fixture_sequential"])
def test_benchmark_forwards_valid_variant_layout(
    benchmark_module, monkeypatch, tmp_path, runner,
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
    supplemental = {"image_key": "exact-image-bound-evidence"}
    crop = {"evidence": {"vision_text": "RAW"}, "context": {"input_text_sha256": "BOUND"}}
    pixels = {"source_identity": "independently-bound-original-pixels"}
    vision = {"attempts": 1, "source": "fresh", "cost_usd": 0.001}

    def process(_text, **kwargs):
        captured["layout"] = kwargs.get("ocr_layout_blocks")
        captured["confidence"] = kwargs.get("ocr_confidence")
        captured["supplemental"] = kwargs.get("supplemental_ocr_evidence")
        captured["crop"] = kwargs.get("quantity_crop_ocr_evidence")
        captured["context"] = kwargs.get("quantity_crop_context")
        captured["pixels"] = kwargs.get("pixel_marker_context")
        captured["vision_mode"] = kwargs.get("vision_mode")
        return {"total": 1, "_ocr_source": "injected", "_ocr_text": _text, "_vision_fallback": vision}

    monkeypatch.setattr(benchmark_module, "process_ocr_text", process)
    monkeypatch.setattr(
        benchmark_module,
        "get_checks_for",
        lambda _truth: {"total": lambda result, _expected: {"pass": result["total"] == 1}},
    )
    _, fixture = getattr(benchmark_module, runner)(
        "receipt_7_v1", text_path, {"total": 1}, 1, "model", 1, None, False,
        save_variants=False, preflight_supplemental_evidence=supplemental,
        preflight_quantity_crop=crop,
        preflight_pixel_marker_context=pixels, vision_mode="fresh",
    )

    assert captured == {
        "layout": layout,
        "confidence": 0.73,
        "supplemental": supplemental,
        "crop": crop["evidence"],
        "context": crop["context"],
        "pixels": pixels,
        "vision_mode": "fresh",
    }
    assert fixture["runs"][0]["vision_fallback"] == vision


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
    supplemental = {"image_key": "exact-image-bound-evidence"}
    crop = {"evidence": {"vision_text": "RAW"}, "context": {"input_text_sha256": "BOUND"}}
    pixels = {"source_identity": "independently-bound-original-pixels"}
    vision = {"attempts": 1, "source": "fresh", "cost_usd": 0.001}

    def process(_text, **kwargs):
        captured["layout"] = kwargs.get("ocr_layout_blocks")
        captured["confidence"] = kwargs.get("ocr_confidence")
        captured["supplemental"] = kwargs.get("supplemental_ocr_evidence")
        captured["crop"] = kwargs.get("quantity_crop_ocr_evidence")
        captured["context"] = kwargs.get("quantity_crop_context")
        captured["pixels"] = kwargs.get("pixel_marker_context")
        captured["vision_mode"] = kwargs.get("vision_mode")
        return {"total": 1, "_vision_fallback": vision}

    monkeypatch.setattr(pipeline, "process_ocr_text", process)
    monkeypatch.setattr(
        accuracy_module, "_SUPPLEMENTAL_OCR_EVIDENCE", {text_path: supplemental},
    )
    monkeypatch.setattr(accuracy_module, "_QUANTITY_CROP_EVIDENCE", {text_path: crop})
    monkeypatch.setattr(accuracy_module, "_PIXEL_MARKER_CONTEXTS", {text_path: pixels})
    monkeypatch.setattr(accuracy_module, "_VISION_MODE", "fresh")
    _, result, _ = accuracy_module._process_one(
        "receipt_7_v1", {"type": "ocr_text", "path": text_path},
    )

    assert captured == {
        "layout": layout,
        "confidence": 0.73,
        "supplemental": supplemental,
        "crop": crop["evidence"],
        "context": crop["context"],
        "pixels": pixels,
        "vision_mode": "fresh",
    }
    assert result["_vision_fallback"] == vision


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

    assert benchmark_module.discover_fixtures() == [
        ("receipt_7_v1", variants / "receipt_7_v1.txt", truth),
        ("receipt_7_v2", selected_variant, truth),
    ]
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


def test_benchmark_preflights_supplemental_cache_before_model_or_vision(
    benchmark_module, monkeypatch, tmp_path,
):
    image_path = tmp_path / "receipt.png"
    assert cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))
    fixture = (
        "receipt_x", image_path, {"document_type": "receipt", "total": 1},
    )
    monkeypatch.setattr(benchmark_module, "_select_fixtures", lambda _names: [fixture])
    monkeypatch.setattr(
        benchmark_module, "SUPPLEMENTAL_OCR_CACHE_DIR", tmp_path / "supplemental",
    )
    monkeypatch.setattr(
        benchmark_module,
        "check_model_available",
        lambda _model: pytest.fail("model setup ran before supplemental preflight"),
    )
    monkeypatch.setattr(
        benchmark_module,
        "init_cloud_vision",
        lambda: pytest.fail("Vision setup ran before supplemental preflight"),
    )

    with pytest.raises(SystemExit) as exc:
        benchmark_module.run_benchmark(
            cached_ocr=True, output_path=tmp_path / "result.json",
        )

    assert exc.value.code == 1


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("ci", [False, True])
def test_cached_ocr_mode_keeps_run_count_and_never_initializes_vision(
    benchmark_module, monkeypatch, tmp_path, workers, ci,
):
    image = tmp_path / "receipt.png"
    assert cv2.imwrite(str(image), np.zeros((8, 8, 3), dtype=np.uint8))
    fixture = ("receipt_7", image, {"total": 123})
    calls = []
    runners = []

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

    def capture_runner(mode, *args):
        calls.append(args)
        runners.append(mode)
        return args[0], {"runs": []}

    monkeypatch.setattr(benchmark_module, "_run_fixture", lambda *args: capture_runner("parallel", *args))
    monkeypatch.setattr(benchmark_module, "_run_fixture_sequential", lambda *args: capture_runner("sequential", *args))
    monkeypatch.setattr(
        benchmark_module, "_assemble_results",
        lambda metadata, _fixtures: {"metadata": metadata, "summary": {"fragile": []}},
    )
    monkeypatch.setattr(benchmark_module, "_save_results", lambda *_args: None)
    monkeypatch.setattr(benchmark_module, "_print_summary", lambda *_args: None)

    result = benchmark_module.run_benchmark(
        runs=7, workers=workers, cached_ocr=not ci, ci=ci, output_path=tmp_path / "result.json",
    )

    assert len(calls) == 1
    expected_runs = 1 if ci else 7
    assert calls[0][3] == expected_runs
    assert runners == ["parallel" if workers > 1 else "sequential"]
    assert isinstance(calls[0][6], benchmark_module._CacheOnlyOCREngine)
    assert calls[0][7] is False
    assert result["metadata"]["runs_per_fixture"] == expected_runs
    assert result["metadata"]["cached_ocr"] is True
    assert result["metadata"]["ci_mode"] is ci


def test_cached_ocr_mode_fails_closed_when_cache_is_missing(
    benchmark_module, monkeypatch, tmp_path,
):
    image = tmp_path / "receipt.png"
    assert cv2.imwrite(str(image), np.zeros((8, 8, 3), dtype=np.uint8))
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


def test_fresh_api_estimate_counts_two_supplemental_calls_and_one_optional_crop(
    benchmark_module,
):
    assert benchmark_module._estimate_api_calls(3, 4) == 20
    assert benchmark_module._estimate_api_calls(3, 4, 2) == 36
    assert benchmark_module._estimate_api_calls(3, 4, 2, 2) == 44


def test_fresh_budget_eligibility_is_structural_not_cached_classification(
    benchmark_module, monkeypatch, tmp_path,
):
    image_path, text_path, _layout_path = _image_and_cache(
        benchmark_module, monkeypatch, tmp_path,
    )
    text_path.write_text("検針\n使用量\nご請求額", encoding="utf-8")

    assert benchmark_module._supplemental_ocr_artifact(
        "receipt_x", image_path,
    ) is None
    assert benchmark_module._supplemental_ocr_artifact(
        "receipt_x", image_path, require_receipt=False,
    ) is not None


@pytest.mark.parametrize("vision_mode", ["normal", "cache_only", "fresh"])
def test_cached_harnesses_pass_explicit_cache_only_mode(
    benchmark_module, accuracy_module, monkeypatch, tmp_path, vision_mode,
):
    import receipt_parser.pipeline as pipeline

    image_path = tmp_path / "receipt.png"
    image_path.write_bytes(b"not read by patched processors")
    benchmark_kwargs = {}
    accuracy_kwargs = {}
    monkeypatch.setenv("RECEIPT_VISION_MODE", vision_mode)
    monkeypatch.setattr(accuracy_module, "_VISION_MODE", vision_mode)

    def benchmark_process(*_args, **kwargs):
        benchmark_kwargs.update(kwargs)
        return {"total": 1, "_ocr_source": "cache", "_pass_history": []}

    def accuracy_process(*_args, **kwargs):
        accuracy_kwargs.update(kwargs)
        return {"_ocr_source": "cache"}

    monkeypatch.setattr(benchmark_module, "process_document", benchmark_process)
    monkeypatch.setattr(
        benchmark_module,
        "get_checks_for",
        lambda _truth: {
            "total": lambda result, _expected: {"pass": result["total"] == 1},
        },
    )
    benchmark_module._run_fixture_sequential(
        "receipt_x", image_path, {"total": 1}, 1, "model", 1, None, False,
        save_variants=False,
    )

    monkeypatch.setattr(pipeline, "process_document", accuracy_process)
    accuracy_module._process_one(
        "receipt_x", {"type": "image", "path": image_path},
    )

    assert benchmark_kwargs["ocr_cache_only"] is True
    assert accuracy_kwargs["ocr_cache_only"] is True
    assert benchmark_kwargs["vision_mode"] == accuracy_kwargs["vision_mode"] == vision_mode


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
            "triage_models": [],
            "triage_max_tokens": 2048,
            "package_source": "src/receipt_parser/__init__.py",
            "vision_mode": "normal",
            "vision_model": "test/vision-model",
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


def test_determinism_compares_public_extractions_not_pass_booleans(benchmark_module):
    def run(extraction, *, error=None):
        return {
            "fields": {"total": {"pass": True}},
            "pass_count": 1,
            "total_fields": 1,
            "passed": error is None,
            "error": error,
            "final_extraction": extraction,
            "ocr": {"confidence": None},
            "ocr_text": "",
        }

    different_but_passing = {"runs": [run({"total": 1}), run({"total": 2})]}
    benchmark_module._finalize_fixture(
        "receipt_x", different_but_passing, save_variants=False,
    )
    assert different_but_passing["determinism_comparable"]
    assert not different_but_passing["deterministic"]

    equal_semantic_values = {"runs": [run({"total": 1}), run({"total": 1.0})]}
    benchmark_module._finalize_fixture(
        "receipt_x", equal_semantic_values, save_variants=False,
    )
    assert equal_semantic_values["deterministic"]

    errored = {"runs": [run({"total": 1}), run({"total": 1}, error="failed")]}
    benchmark_module._finalize_fixture("receipt_x", errored, save_variants=False)
    assert not errored["determinism_comparable"]
    assert not errored["deterministic"]

    missing = {"runs": [run({"total": 1}), run(None)]}
    benchmark_module._finalize_fixture("receipt_x", missing, save_variants=False)
    assert not missing["determinism_comparable"]
    assert not missing["deterministic"]


def test_summary_lists_exact_nondeterministic_fixtures(benchmark_module):
    summary = benchmark_module._compute_summary({
        "receipt_b": {"runs": [], "deterministic": False},
        "receipt_a": {"runs": [], "deterministic": True},
    }, {})
    assert summary["nondeterministic_fixture_count"] == 1
    assert summary["nondeterministic_fixtures"] == ["receipt_b"]
    assert summary["deterministic_fixture_count"] == 1
    assert summary["fixture_count"] == 2


def test_determinism_rate_preserves_single_fixture_variance(benchmark_module):
    per_fixture = {
        f"receipt_{i}": {"runs": [], "deterministic": i != 0}
        for i in range(352)
    }
    summary = benchmark_module._compute_summary(per_fixture, {})
    assert summary["deterministic_fixture_count"] == 351
    assert summary["fixture_count"] == 352
    assert summary["determinism_rate"] == 351 / 352
    assert summary["determinism_rate"] < 1.0


def test_comparison_scope_includes_effective_triage_settings(benchmark_module):
    current = _scope("same")
    previous = _scope("same")
    previous["metadata"]["package_source"] = "C:/clean-checkout/receipt_parser/__init__.py"
    assert benchmark_module._comparison_scope_mismatches(current, previous) == []

    previous = _scope("same")
    previous["metadata"]["triage_models"] = ["other/model"]
    mismatches = benchmark_module._comparison_scope_mismatches(current, previous)
    assert any("triage_models" in mismatch for mismatch in mismatches)

    previous = _scope("same")
    previous["metadata"]["triage_max_tokens"] = 4096
    mismatches = benchmark_module._comparison_scope_mismatches(current, previous)
    assert any("triage_max_tokens" in mismatch for mismatch in mismatches)

    for field, value in (("vision_mode", "fresh"), ("vision_model", "")):
        previous = _scope("same")
        previous["metadata"][field] = value
        assert any(field in mismatch for mismatch in benchmark_module._comparison_scope_mismatches(current, previous))


def test_accuracy_scope_records_triage_settings_and_imported_package(accuracy_module):
    from receipt_parser.llm import _configured_triage_models, _triage_max_tokens
    import receipt_parser

    assert accuracy_module._ACCURACY_SCOPE["triage_models"] == _configured_triage_models()
    assert accuracy_module._ACCURACY_SCOPE["triage_max_tokens"] == _triage_max_tokens()
    assert accuracy_module._ACCURACY_SCOPE["package_source"] == str(
        Path(receipt_parser.__file__).resolve(),
    )
    assert accuracy_module._ACCURACY_SCOPE["vision_mode"] == accuracy_module._VISION_MODE
    assert accuracy_module._ACCURACY_SCOPE["vision_model"] == accuracy_module.configured_vision_model()


@pytest.mark.parametrize("module_fixture", ["benchmark_module", "accuracy_module"])
def test_crop_corpus_fingerprint_counts_explicit_absence_and_invalid_present_bytes(
    request, module_fixture, monkeypatch, tmp_path,
):
    module = request.getfixturevalue(module_fixture)
    image_path, _, _ = _image_and_cache(module, monkeypatch, tmp_path)
    crop_path = tmp_path / "quantity.json"
    monkeypatch.setattr(module, "_quantity_crop_artifact", lambda *_: (crop_path, None))
    if module_fixture == "benchmark_module":
        corpus = [("sample", image_path, {"total": 1})]
        fingerprint = lambda: module._fixture_corpus_sha256(corpus, cached_ocr=True)
        preflight = lambda: module._preflight_quantity_crop(corpus, cached_images=True)
    else:
        corpus = [("sample", {"type": "image", "path": image_path}, {"total": 1})]
        fingerprint = lambda: module._corpus_sha256(corpus)
        preflight = lambda: module._preflight_quantity_crop(corpus)
    absent = fingerprint()
    assert preflight() == {}
    crop_path.write_text("{invalid", encoding="utf-8")
    present = fingerprint()
    crop_path.write_text("{}", encoding="utf-8")
    changed = fingerprint()
    assert absent != present != changed
    with pytest.raises(ValueError, match="Invalid quantity crop OCR sidecar"):
        preflight()


@pytest.mark.parametrize("module_fixture", ["benchmark_module", "accuracy_module"])
def test_replay_pixel_context_requires_original_frame_geometry_without_native_crop(
    request, module_fixture, monkeypatch, tmp_path,
):
    module = request.getfixturevalue(module_fixture)
    image_path, original_text, _ = _image_and_cache(module, monkeypatch, tmp_path)
    text_path = tmp_path / "sample_v1.txt"
    text_path.write_text(original_text.read_text(encoding="utf-8"), encoding="utf-8")
    image = module.load_image(image_path)[0]
    layout = [{"text": "one", "x": 1, "y": 1, "confidence": 0.9, "page": 0,
               "bbox": [[1, 1], [2, 1], [2, 2], [1, 2]]}]
    write_ocr_layout_sidecar(original_text, layout, provenance_kind="same_call_capture",
                           provenance_source=module._ocr_cache_key(image))
    evidence, _ = _write_supplemental_sidecar(module, monkeypatch, image_path, tmp_path)
    monkeypatch.setattr(module, "detect_document_type", lambda _text: "receipt")
    if module_fixture == "benchmark_module":
        monkeypatch.setattr(module, "_fixture_image", lambda _base: image_path)
        source = text_path
    else:
        monkeypatch.setattr(module, "_find_image", lambda _base: image_path)
        source = {"type": "ocr_text", "path": text_path}
    context = module._pixel_marker_artifact("sample_v1", source, evidence)
    assert context is not None and context["primary_layout"] == layout
    assert np.array_equal(context["image"], image)

    # A different replay's own sidecar must prove its frame; its filename cannot.
    text_path.write_text("different replay", encoding="utf-8")
    write_ocr_layout_sidecar(text_path, layout, provenance_kind="same_call_capture",
                           provenance_source="unrelated-image")
    assert module._pixel_marker_artifact("sample_v1", source, evidence) is None


@pytest.mark.parametrize("module_fixture", ["benchmark_module", "accuracy_module"])
def test_vision_corpus_hashes_only_bound_cache_inputs_and_records_absence(
    request, module_fixture, monkeypatch, tmp_path,
):
    module = request.getfixturevalue(module_fixture)
    image_path, _, _ = _image_and_cache(module, monkeypatch, tmp_path)
    bound = tmp_path / "bound-vision.json"
    monkeypatch.setattr(module, "vision_cache_files", lambda _image: [bound] if bound.is_file() else [])
    if module_fixture == "benchmark_module":
        corpus = [("sample", image_path, {"total": 1})]
        fingerprint = lambda: module._fixture_corpus_sha256(corpus, cached_ocr=True)
    else:
        corpus = [("sample", {"type": "image", "path": image_path}, {"total": 1})]
        fingerprint = lambda: module._corpus_sha256(corpus)
    absent = fingerprint()
    (tmp_path / "unrelated-vision.json").write_text("unrelated", encoding="utf-8")
    assert fingerprint() == absent
    bound.write_text("invalid but present", encoding="utf-8")
    present = fingerprint()
    bound.write_text("{}", encoding="utf-8")
    assert absent != present != fingerprint()


@pytest.mark.parametrize("module_fixture", ["benchmark_module", "accuracy_module"])
def test_harness_vision_settings_validate_mode_and_preserve_explicit_disable(
    request, module_fixture, monkeypatch,
):
    module = request.getfixturevalue(module_fixture)
    monkeypatch.delenv("RECEIPT_VISION_MODE", raising=False)
    assert module._vision_mode() == "normal"
    monkeypatch.setenv("RECEIPT_VISION_MODE", "invalid")
    with pytest.raises(ValueError, match="RECEIPT_VISION_MODE"):
        module._vision_mode()
    monkeypatch.setenv("RECEIPT_VISION_MODEL", "")
    assert module.configured_vision_model() == ""


def test_accuracy_records_vision_fallback_metadata_once_per_case(accuracy_module, monkeypatch):
    fallback = {"attempts": 1, "source": "fresh", "cost_usd": 0.001}
    monkeypatch.setattr(accuracy_module, "_RESULTS_CACHE", {"sample": {"_vision_fallback": fallback}})
    scope = {}
    accuracy_module._record_vision_fallbacks(scope)
    fallback["attempts"] = 2
    assert scope["vision_fallback"] == {"sample": {"attempts": 1, "source": "fresh", "cost_usd": 0.001}}
