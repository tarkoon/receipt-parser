import json
from copy import deepcopy

import pytest

import receipt_parser.receipt_supplemental_ocr as supplemental
from receipt_parser.receipt_supplemental_ocr import (
    apply_supplemental_ocr_evidence,
    build_supplemental_ocr_evidence,
    load_supplemental_ocr_evidence,
    save_supplemental_ocr_evidence,
    supplemental_ocr_cache_path,
)


IMAGE_KEY = "a" * 32
STRATEGY_FINGERPRINT = "b" * 64
IMAGE_SHAPE = [1200, 800]


def _layout_for(text):
    return [
        {
            "text": line,
            "confidence": 0.99,
            "x": 0,
            "y": index * 50,
            "bbox": [
                [0, index * 50],
                [500, index * 50],
                [500, index * 50 + 40],
                [0, index * 50 + 40],
            ],
            "page": 0,
        }
        for index, line in enumerate(text.splitlines())
    ]


def _evidence(merged_text, tile_one, tile_two="footer"):
    return build_supplemental_ocr_evidence(
        image_key=IMAGE_KEY,
        image_shape=IMAGE_SHAPE,
        strategy_fingerprint=STRATEGY_FINGERPRINT,
        merged_text=merged_text,
        source_layout=_layout_for(merged_text),
        tile_texts=[tile_one, tile_two],
    )


def _without_tax_categories(value):
    masked = deepcopy(value)
    for item in masked.get("line_items") or []:
        item["tax_category"] = "<TARGET>"
    return json.dumps(
        masked,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def test_unique_ocr_backed_location_extension_changes_only_location():
    receipt = {
        "merchant": "ブランド",
        "location": "元店",
        "subtotal": 100,
        "line_items": [{"description": "商品", "total": 100, "tax_category": "10%"}],
        "taxes": [{"rate": "10%", "amount": 9}],
    }
    original = deepcopy(receipt)
    header = "ブランド\n稲元店 TEL0940-00-0000"

    output, metadata = apply_supplemental_ocr_evidence(
        receipt,
        _evidence(header, header),
    )

    assert receipt == original
    assert output == {**original, "location": "稲元店"}
    assert metadata["accepted_fields"] == ["location"]
    assert metadata["location"]["criteria"] == {
        "one_candidate_across_merged_and_tiles": True,
        "candidate_present": True,
        "candidate_differs": True,
        "candidate_has_location_suffix": True,
        "production_evidence_check": True,
        "unique_supporting_header_line": True,
        "candidate_strictly_extends_current_or_unsupported_composite_component": True,
    }


def test_layout_marker_tax_vector_applies_only_when_printed_bases_reconcile():
    receipt = {
        "merchant": "ストア",
        "location": None,
        "subtotal": 300,
        "line_items": [
            {"description": "食品A", "total": 100, "tax_category": "10%"},
            {"description": "用品B", "total": 200, "tax_category": "8%"},
        ],
        "taxes": [
            {"rate": "8%", "amount": 8},
            {"rate": "10%", "amount": 20},
        ],
    }
    original = deepcopy(receipt)
    evidence = _evidence(
        "明細\n* 食品A ¥100\n用品B ¥200\n小計\n¥300",
        "*は軽減税率8%適用商品\n8% 対象額 ¥100\n10% 対象額 ¥200",
    )

    output, metadata = apply_supplemental_ocr_evidence(receipt, evidence)

    assert receipt == original
    assert [item["tax_category"] for item in output["line_items"]] == ["8%", "10%"]
    assert _without_tax_categories(output) == _without_tax_categories(original)
    assert metadata["accepted_fields"] == ["tax_categories"]
    assert metadata["tax_categories"]["reconciliation_mode"] == "net"
    assert metadata["tax_categories"]["proposed_category_sums"] == {
        "8%": 100.0,
        "10%": 200.0,
    }


def test_ambiguous_marker_rows_reject_the_whole_tax_vector_without_partial_mutation():
    receipt = {
        "merchant": "ストア",
        "location": None,
        "subtotal": 200,
        "line_items": [
            {"description": "食品A", "total": 100, "tax_category": "10%"},
            {"description": "食品A", "total": 100, "tax_category": "8%"},
        ],
        "taxes": [
            {"rate": "8%", "amount": 8},
            {"rate": "10%", "amount": 9},
        ],
    }
    original = deepcopy(receipt)
    evidence = _evidence(
        "明細\n* 食品A ¥100\n食品A ¥100\n小計\n¥200",
        "*は軽減税率8%適用商品\n8% 対象額 ¥100\n10% 対象額 ¥100",
    )

    output, metadata = apply_supplemental_ocr_evidence(receipt, evidence)

    assert receipt == original
    assert output == original
    assert metadata["accepted_fields"] == []
    assert not metadata["tax_categories"]["criteria"]["reduced_rows_uniquely_aligned"]


def test_non_reconciling_printed_rate_bases_reject_the_whole_tax_vector():
    receipt = {
        "merchant": "ストア",
        "location": None,
        "subtotal": 300,
        "line_items": [
            {"description": "食品A", "total": 100, "tax_category": "10%"},
            {"description": "用品B", "total": 200, "tax_category": "8%"},
        ],
        "taxes": [
            {"rate": "8%", "amount": 8},
            {"rate": "10%", "amount": 20},
        ],
    }
    original = deepcopy(receipt)
    evidence = _evidence(
        "明細\n* 食品A ¥100\n用品B ¥200\n小計\n¥300",
        "*は軽減税率8%適用商品\n8% 対象額 ¥110\n10% 対象額 ¥190",
    )

    output, metadata = apply_supplemental_ocr_evidence(receipt, evidence)

    assert receipt == original
    assert output == original
    assert metadata["accepted_fields"] == []
    assert not metadata["tax_categories"]["criteria"]["rate_base_arithmetic_reconciles"]


def test_sidecar_round_trip_is_raw_only_and_identity_includes_image_shape(tmp_path):
    evidence = _evidence("merged text", "top tile", "bottom tile")

    path = save_supplemental_ocr_evidence(tmp_path, evidence)
    loaded = load_supplemental_ocr_evidence(
        tmp_path,
        image_key=IMAGE_KEY,
        image_shape=IMAGE_SHAPE,
        strategy_fingerprint=STRATEGY_FINGERPRINT,
    )

    assert loaded == evidence
    assert set(json.loads(path.read_text(encoding="utf-8"))) == {
        "schema_version",
        "image_key",
        "image_shape",
        "strategy_fingerprint",
        "merged_text",
        "source_layout",
        "tiles",
    }
    assert path != supplemental_ocr_cache_path(
        tmp_path,
        image_key=IMAGE_KEY,
        image_shape=[1201, 800],
        strategy_fingerprint=STRATEGY_FINGERPRINT,
    )


def test_sidecar_load_fails_closed_on_corruption_or_answer_fields(tmp_path):
    evidence = _evidence("merged text", "top tile", "bottom tile")
    path = save_supplemental_ocr_evidence(tmp_path, evidence)
    expected = {
        "image_key": IMAGE_KEY,
        "image_shape": IMAGE_SHAPE,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
    }

    path.write_text("{not json", encoding="utf-8")
    assert load_supplemental_ocr_evidence(tmp_path, **expected) is None

    contaminated = {**evidence, "fixture": "receipt_49", "proposal": "known answer"}
    path.write_text(json.dumps(contaminated, ensure_ascii=False), encoding="utf-8")
    assert load_supplemental_ocr_evidence(tmp_path, **expected) is None

    wrong_identity = {**evidence, "image_shape": [1, 1]}
    path.write_text(json.dumps(wrong_identity, ensure_ascii=False), encoding="utf-8")
    assert load_supplemental_ocr_evidence(tmp_path, **expected) is None

    boolean_schema = {**evidence, "schema_version": True}
    path.write_text(json.dumps(boolean_schema, ensure_ascii=False), encoding="utf-8")
    assert load_supplemental_ocr_evidence(tmp_path, **expected) is None


def test_sidecar_build_and_load_reject_incoherent_layout_text(tmp_path):
    with pytest.raises(ValueError, match="invalid supplemental OCR evidence"):
        build_supplemental_ocr_evidence(
            image_key=IMAGE_KEY,
            image_shape=IMAGE_SHAPE,
            strategy_fingerprint=STRATEGY_FINGERPRINT,
            merged_text="caller supplied text",
            source_layout=_layout_for("raw mapped text"),
            tile_texts=["top tile", "bottom tile"],
        )

    evidence = _evidence("raw mapped text", "top tile", "bottom tile")
    path = save_supplemental_ocr_evidence(tmp_path, evidence)
    tampered = {**evidence, "merged_text": "tampered text"}
    path.write_text(json.dumps(tampered, ensure_ascii=False), encoding="utf-8")

    assert load_supplemental_ocr_evidence(
        tmp_path,
        image_key=IMAGE_KEY,
        image_shape=IMAGE_SHAPE,
        strategy_fingerprint=STRATEGY_FINGERPRINT,
    ) is None


def test_sidecar_load_rejects_malformed_nonfinite_and_out_of_bounds_layout(tmp_path):
    evidence = _evidence("raw mapped text", "top tile", "bottom tile")
    path = save_supplemental_ocr_evidence(tmp_path, evidence)
    expected = {
        "image_key": IMAGE_KEY,
        "image_shape": IMAGE_SHAPE,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
    }
    corrupt_layouts = []
    for field, value in (("y", "not-a-number"), ("y", float("inf"))):
        layout = deepcopy(evidence["source_layout"])
        layout[0][field] = value
        corrupt_layouts.append(layout)
    layout = deepcopy(evidence["source_layout"])
    layout[0]["bbox"][0] = [IMAGE_SHAPE[1] + 1, 0]
    corrupt_layouts.append(layout)
    layout = deepcopy(evidence["source_layout"])
    layout[0]["bbox"][0] = [0, float("nan")]
    corrupt_layouts.append(layout)

    for layout in corrupt_layouts:
        path.write_text(
            json.dumps({**evidence, "source_layout": layout}, ensure_ascii=False),
            encoding="utf-8",
        )
        assert load_supplemental_ocr_evidence(tmp_path, **expected) is None


def test_sidecar_load_rejects_nested_answers_and_invalid_block_metadata(tmp_path):
    evidence = _evidence("raw mapped text", "top tile", "bottom tile")
    path = save_supplemental_ocr_evidence(tmp_path, evidence)
    expected = {
        "image_key": IMAGE_KEY,
        "image_shape": IMAGE_SHAPE,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
    }
    corrupt_layouts = []
    for field, value in (
        ("confidence", float("inf")),
        ("confidence", -0.01),
        ("confidence", 1.01),
        ("page", True),
        ("page", -1),
        ("page", 0.0),
        ("text", ""),
    ):
        layout = deepcopy(evidence["source_layout"])
        layout[0][field] = value
        corrupt_layouts.append(layout)
    layout = deepcopy(evidence["source_layout"])
    layout[0]["expected_answer"] = "known value"
    corrupt_layouts.append(layout)

    for layout in corrupt_layouts:
        path.write_text(
            json.dumps({**evidence, "source_layout": layout}, ensure_ascii=False),
            encoding="utf-8",
        )
        assert load_supplemental_ocr_evidence(tmp_path, **expected) is None


def test_atomic_sidecar_replace_failure_preserves_old_file_and_cleans_temp(
    tmp_path,
    monkeypatch,
):
    original = _evidence("first merged text", "first top", "first bottom")
    path = save_supplemental_ocr_evidence(tmp_path, original)
    old_bytes = path.read_bytes()
    replacement = _evidence("second merged text", "second top", "second bottom")

    def fail_replace(_source, _destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(supplemental.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        save_supplemental_ocr_evidence(tmp_path, replacement)

    assert path.read_bytes() == old_bytes
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []
