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
