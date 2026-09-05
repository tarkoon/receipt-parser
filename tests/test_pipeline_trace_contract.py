"""Contracts for pipeline-owned receipt mutations and digital PDFs."""

from copy import deepcopy
from pathlib import Path

import pytest


def _extraction(**overrides):
    payload = {
        "document_type": "receipt",
        "merchant": "ALIAS SOURCE",
        "currency": "JPY",
        "total": 100,
        "amount_paid": 100,
        "line_items": [],
        "taxes": [],
    }
    payload.update(overrides)
    return payload


def _history(payload):
    return [{"pass": 1, "extraction": payload, "warnings": []}]


def test_pipeline_owned_mutations_are_traced_by_declared_field_owner(monkeypatch):
    from receipt_parser import pipeline

    source = _extraction(location=None, points_used=None)
    monkeypatch.setattr(pipeline, "check_model_available", lambda _model: None)
    monkeypatch.setattr(
        pipeline,
        "extract_with_verification",
        lambda *_args, **_kwargs: (source, _history(source)),
    )
    monkeypatch.setattr(
        pipeline,
        "_select_receipt_postprocessed_candidate",
        lambda extracted, *_args, **_kwargs: dict(extracted),
    )

    def reconcile(extracted, _text):
        extracted["points_used"] = 10
        extracted["amount_paid"] = 90

    monkeypatch.setattr(pipeline, "reconcile_points_payment_from_ocr", reconcile)
    monkeypatch.setattr(pipeline, "_location_needs_resolution", lambda *_args: True)
    monkeypatch.setattr(pipeline, "_location_has_ocr_evidence", lambda *_args: True)
    monkeypatch.setattr(
        pipeline,
        "_resolve_location",
        lambda *_args: ("東京都千代田区", None),
    )
    monkeypatch.setattr(
        pipeline,
        "_trim_purchase_store_metadata_location",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        pipeline,
        "_prepare_receipt_output_payload",
        lambda receipt, *_args, **_kwargs: receipt.model_dump(),
    )

    def apply_alias(result):
        result["merchant"] = "ALIAS TARGET"
        return result

    monkeypatch.setattr(pipeline, "_apply_user_rules", apply_alias)

    result = pipeline.process_ocr_text(
        "ALIAS SOURCE\n東京都千代田区1-2\n合計 ¥100",
        debug=True,
    )

    events = {
        event["stage"]: event
        for event in result["_receipt_mutation_trace"]
        if event["stage"] in pipeline._PIPELINE_RECEIPT_MUTATION_PHASES
    }
    assert set(events["common_points_payment_reconciliation"]["changes"]) == {
        "amount_paid",
        "points_used",
    }
    assert set(events["location_resolution_validation"]["changes"]) == {
        "location",
    }
    assert set(events["user_rule_alias"]["changes"]) == {"merchant"}
    for stage, event in events.items():
        assert event["owner_phase"] == stage
        assert set(event["changes"]) <= set(event["writes"])
        assert all(
            set(change) == {"before", "after"}
            for change in event["changes"].values()
        )


def test_finalizer_traces_user_alias_and_schema_canonicalization(monkeypatch):
    from receipt_parser import pipeline

    result = _extraction(points_used="10", amount_paid="90")
    trace = []

    def apply_alias(payload):
        payload["merchant"] = "ALIAS TARGET"
        return payload

    monkeypatch.setattr(pipeline, "_apply_user_rules", apply_alias)
    finalized = pipeline._finalize_receipt_result(
        result,
        apply_user_rules=True,
        mutation_trace=trace,
    )

    events = {event["stage"]: event for event in trace}
    assert set(events["user_rule_alias"]["changes"]) == {"merchant"}
    assert set(events["final_schema_canonicalization"]["changes"]) == {
        "amount_paid",
        "points_used",
    }
    assert finalized["merchant"] == "ALIAS TARGET"
    assert finalized["points_used"] == 10
    assert finalized["_receipt_mutation_trace"] is trace
    assert all(
        set(event["changes"]) <= set(event["writes"])
        for event in trace
    )


def test_digital_pdf_uses_shared_postprocess_and_one_post_final_validate(monkeypatch):
    from receipt_parser import pipeline, usage

    digital_text = "TEST\n商品 ¥100\n合計 ¥100\n現金 ¥100"
    source = _extraction(merchant="TEST", payment_method=None)
    postprocess_texts = []
    output_texts = []
    events = []
    finalized = False

    monkeypatch.setattr(pipeline, "check_model_available", lambda _model: None)
    monkeypatch.setattr(usage, "track_document", lambda _path: None)
    monkeypatch.setattr(pipeline, "load_image", lambda _path: [object()])
    monkeypatch.setattr(pipeline, "try_extract_text_layer", lambda _path: digital_text)
    monkeypatch.setattr(
        pipeline,
        "extract_with_verification",
        lambda *_args, **_kwargs: (source, _history(source)),
    )
    monkeypatch.setattr(pipeline, "_location_needs_resolution", lambda *_args: False)

    def postprocess(extracted, unified_text, *_args, **_kwargs):
        postprocess_texts.append(unified_text)
        extracted["payment_method"] = "cash"
        return extracted

    monkeypatch.setattr(pipeline, "postprocess_receipt", postprocess)
    real_prepare = pipeline._prepare_receipt_output_payload

    def prepare(receipt, ocr_text=None, **kwargs):
        output_texts.append(ocr_text)
        return real_prepare(receipt, ocr_text, **kwargs)

    monkeypatch.setattr(pipeline, "_prepare_receipt_output_payload", prepare)
    real_finalize = pipeline._finalize_receipt_result

    def finalize(*args, **kwargs):
        nonlocal finalized
        result = real_finalize(*args, **kwargs)
        finalized = True
        return result

    monkeypatch.setattr(pipeline, "_finalize_receipt_result", finalize)

    def on_stage(stage, _detail, _progress, payload=None):
        del payload
        if stage == "validate":
            assert finalized is True
        events.append(stage)

    result = pipeline.process_document(Path("digital.pdf"), on_stage=on_stage)

    assert result["payment_method"] == "cash"
    assert postprocess_texts and set(postprocess_texts) == {digital_text}
    assert output_texts == [digital_text]
    assert events.count("validate") == 1
    assert events[-2:] == ["validate", "done"]


def test_extraction_error_keeps_public_error_contract_for_injected_text(monkeypatch):
    from receipt_parser import pipeline

    monkeypatch.setattr(pipeline, "check_model_available", lambda _model: None)
    monkeypatch.setattr(
        pipeline,
        "extract_with_verification",
        lambda *_args, **_kwargs: ({"error": "model failed"}, []),
    )

    result = pipeline.process_ocr_text("TEST\n合計 ¥100")

    assert result["_error"] == "model failed"
    assert "total" not in result


def test_extraction_error_keeps_public_error_contract_for_digital_pdf(monkeypatch):
    from receipt_parser import pipeline, usage

    monkeypatch.setattr(pipeline, "check_model_available", lambda _model: None)
    monkeypatch.setattr(usage, "track_document", lambda _path: None)
    monkeypatch.setattr(pipeline, "load_image", lambda _path: [object()])
    monkeypatch.setattr(
        pipeline,
        "try_extract_text_layer",
        lambda _path: "TEST\n合計 ¥100",
    )
    monkeypatch.setattr(
        pipeline,
        "extract_with_verification",
        lambda *_args, **_kwargs: ({"error": "model failed"}, []),
    )

    result = pipeline.process_document(Path("digital.pdf"))

    assert result["_error"] == "model failed"
    assert "total" not in result


def test_common_arithmetic_uses_schema_coerced_points(monkeypatch):
    from receipt_parser import pipeline

    source = _extraction(total="1000", points_used="100", amount_paid=None)
    monkeypatch.setattr(pipeline, "check_model_available", lambda _model: None)
    monkeypatch.setattr(
        pipeline,
        "extract_with_verification",
        lambda *_args, **_kwargs: (source, _history(source)),
    )
    monkeypatch.setattr(
        pipeline,
        "_select_receipt_postprocessed_candidate",
        lambda extracted, *_args, **_kwargs: dict(extracted),
    )
    monkeypatch.setattr(pipeline, "_location_needs_resolution", lambda *_args: False)

    result = pipeline.process_ocr_text("TEST\n合計 ¥1,000")

    assert result["points_used"] == 100
    assert result["amount_paid"] == 900


def test_candidate_selection_skips_schema_invalid_retry_before_gap_ranking(
    monkeypatch,
):
    from receipt_parser import pipeline

    base = _extraction(merchant="VALID BASE", line_items=[])
    invalid_retry = _extraction(
        merchant="INVALID RETRY",
        payer={"not": "a string"},
        line_items=[{
            "description": "item",
            "qty": 1,
            "unit_price": 100,
            "total": 100,
        }],
    )
    history = _history(invalid_retry)
    monkeypatch.setattr(
        pipeline,
        "postprocess_receipt",
        lambda extracted, *_args, **_kwargs: extracted,
    )

    selected = pipeline._select_receipt_postprocessed_candidate(
        base,
        history,
        "VALID BASE\n合計 ¥100",
        0.9,
        {},
        "test-model",
        None,
    )

    assert selected["merchant"] == "VALID BASE"
    assert history[0]["postprocess_selected"] is False
    assert history[0]["postprocess_warning_count"] == 1
    assert history[0]["postprocess_warnings"][0].startswith(
        "Receipt schema validation failed"
    )


def test_candidate_selection_prefers_explicit_printed_item_count_when_balanced(
    monkeypatch,
):
    from receipt_parser import pipeline

    def rows(count, unit_price):
        return [
            {
                "description": f"row {index}",
                "qty": 1,
                "unit_price": unit_price,
                "total": unit_price,
            }
            for index in range(count)
        ]

    ten_rows = _extraction(total=900, amount_paid=900, line_items=rows(10, 90))
    nine_rows = _extraction(total=900, amount_paid=900, line_items=rows(9, 100))
    history = _history(nine_rows)
    monkeypatch.setattr(
        pipeline,
        "postprocess_receipt",
        lambda extracted, *_args, **_kwargs: extracted,
    )

    selected = pipeline._select_receipt_postprocessed_candidate(
        ten_rows,
        history,
        "STORE\nお買上商品数:9\n合計 ¥900",
        0.9,
        {},
        "test-model",
        None,
    )

    assert len(selected["line_items"]) == 9
    assert history[0]["postprocess_selected"] is True


def test_candidate_selection_uses_exact_balanced_layout_count_as_tiebreaker(
    monkeypatch,
):
    from receipt_parser import pipeline

    def rows(totals):
        return [
            {
                "description": f"row {index}",
                "qty": 1,
                "unit_price": total,
                "total": total,
            }
            for index, total in enumerate(totals)
        ]

    four_rows = _extraction(total=100, amount_paid=100, line_items=rows([25] * 4))
    five_rows = _extraction(total=100, amount_paid=100, line_items=rows([20] * 5))
    history = _history(five_rows)
    monkeypatch.setattr(
        pipeline,
        "postprocess_receipt",
        lambda extracted, *_args, **_kwargs: extracted,
    )
    monkeypatch.setattr(
        pipeline,
        "_apply_final_receipt_output_repairs",
        lambda *_args, **_kwargs: None,
    )
    layout_count_calls = []

    def balanced_layout_item_count(extracted, layout):
        layout_count_calls.append((extracted, layout))
        return 4 if layout else None

    monkeypatch.setattr(
        pipeline,
        "_balanced_layout_item_count",
        balanced_layout_item_count,
    )

    selected = pipeline._select_receipt_postprocessed_candidate(
        four_rows,
        history,
        "STORE\n合計 ¥100",
        0.9,
        {},
        "test-model",
        [{"text": "layout"}],
    )

    assert len(selected["line_items"]) == 4
    assert history[0]["postprocess_selected"] is False
    assert len(layout_count_calls) == 1
    assert layout_count_calls[0][0]["total"] == 100
    assert layout_count_calls[0][1] == [{"text": "layout"}]


def test_candidate_layout_count_uses_postprocessed_primary_financials(monkeypatch):
    from receipt_parser import pipeline

    def rows(count, unit_price):
        return [
            {
                "description": f"row {index}",
                "qty": 1,
                "unit_price": unit_price,
                "total": unit_price,
            }
            for index in range(count)
        ]

    four_rows = _extraction(total=120, amount_paid=120, line_items=rows(4, 25))
    five_rows = _extraction(total=220, amount_paid=220, line_items=rows(5, 40))
    history = _history(five_rows)

    def postprocess(extracted, *_args, **_kwargs):
        target = 100 if len(extracted["line_items"]) == 4 else 200
        extracted["total"] = target
        extracted["amount_paid"] = target
        return extracted

    monkeypatch.setattr(pipeline, "postprocess_receipt", postprocess)
    monkeypatch.setattr(
        pipeline,
        "_apply_final_receipt_output_repairs",
        lambda *_args, **_kwargs: None,
    )
    layout_count_bases = []

    def balanced_layout_item_count(extracted, _layout):
        layout_count_bases.append(extracted["total"])
        return 4 if extracted["total"] == 100 else 5

    monkeypatch.setattr(
        pipeline,
        "_balanced_layout_item_count",
        balanced_layout_item_count,
    )

    selected = pipeline._select_receipt_postprocessed_candidate(
        four_rows,
        history,
        "STORE\n合計 ¥100",
        0.9,
        {},
        "test-model",
        [{"text": "layout"}],
    )

    assert len(selected["line_items"]) == 4
    assert history[0]["postprocess_selected"] is False
    assert layout_count_bases == [100.0]


def test_candidate_selection_prefers_printed_count_over_layout_count(monkeypatch):
    from receipt_parser import pipeline

    def rows(count, unit_price):
        return [
            {
                "description": f"row {index}",
                "qty": 1,
                "unit_price": unit_price,
                "total": unit_price,
            }
            for index in range(count)
        ]

    four_rows = _extraction(total=100, amount_paid=100, line_items=rows(4, 25))
    five_rows = _extraction(total=100, amount_paid=100, line_items=rows(5, 20))
    history = _history(five_rows)
    monkeypatch.setattr(
        pipeline,
        "postprocess_receipt",
        lambda extracted, *_args, **_kwargs: extracted,
    )
    monkeypatch.setattr(
        pipeline,
        "_apply_final_receipt_output_repairs",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        pipeline,
        "_balanced_layout_item_count",
        lambda _extracted, _layout: 4,
    )

    selected = pipeline._select_receipt_postprocessed_candidate(
        four_rows,
        history,
        "STORE\nお買上商品数:5\n合計 ¥100",
        0.9,
        {},
        "test-model",
        [{"text": "layout"}],
    )

    assert len(selected["line_items"]) == 5
    assert history[0]["postprocess_selected"] is True


def test_candidate_score_ranks_layout_exact_then_absent_then_mismatch():
    from receipt_parser import pipeline

    extracted = _extraction(
        total=100,
        amount_paid=100,
        line_items=[
            {
                "description": f"row {index}",
                "qty": 1,
                "unit_price": 25,
                "total": 25,
            }
            for index in range(4)
        ],
    )

    exact = pipeline._receipt_candidate_score(extracted, [], layout_item_count=4)
    absent = pipeline._receipt_candidate_score(extracted, [], layout_item_count=None)
    mismatch = pipeline._receipt_candidate_score(extracted, [], layout_item_count=5)

    assert exact < absent < mismatch


def test_candidate_selection_considers_cross_alt_and_deduplicates(monkeypatch):
    from receipt_parser import pipeline

    base = _extraction(line_items=[{
        "description": "item",
        "qty": 1,
        "unit_price": 90,
        "total": 90,
    }])
    alternate = _extraction(line_items=[{
        "description": "item",
        "qty": 1,
        "unit_price": 100,
        "total": 100,
    }])
    history = [{
        "pass": "1-cross",
        "extraction": dict(base),
        "alt_extraction": alternate,
        "warnings": [],
    }]
    postprocess_calls = []

    def postprocess(extracted, *_args, **_kwargs):
        postprocess_calls.append(extracted["line_items"][0]["total"])
        return extracted

    monkeypatch.setattr(pipeline, "postprocess_receipt", postprocess)
    monkeypatch.setattr(
        pipeline,
        "_apply_final_receipt_output_repairs",
        lambda *_args, **_kwargs: None,
    )

    selected = pipeline._select_receipt_postprocessed_candidate(
        base,
        history,
        "STORE\n合計 ¥100",
        0.9,
        {},
        "test-model",
        None,
    )

    assert selected["line_items"][0]["total"] == 100
    assert postprocess_calls == [90, 100]
    assert history[0]["postprocess_selected"] is True
    assert history[0]["postprocess_selected_source"] == "alt_extraction"
    assert history[0]["postprocess_items_sum_gap"] == 0
    assert history[0]["postprocess_candidates"]["extraction"]["items_sum_gap"] == 10
    assert history[0]["postprocess_candidates"]["alt_extraction"]["items_sum_gap"] == 0


def test_candidate_selection_scores_late_repaired_copy_but_returns_pre_final(
    monkeypatch,
):
    from receipt_parser import pipeline

    base = _extraction(
        merchant="BASE",
        line_items=[{
            "description": "item",
            "qty": 1,
            "unit_price": 99,
            "total": 99,
        }],
    )
    late_repair_candidate = _extraction(
        merchant="LATE REPAIR",
        line_items=[{
            "description": "item",
            "qty": 1,
            "unit_price": 90,
            "total": 90,
        }],
    )
    history = _history(late_repair_candidate)

    monkeypatch.setattr(
        pipeline,
        "postprocess_receipt",
        lambda extracted, *_args, **_kwargs: extracted,
    )

    def apply_late_repairs(result, *_args, **_kwargs):
        if result["merchant"] == "LATE REPAIR":
            result["line_items"][0]["unit_price"] = 100
            result["line_items"][0]["total"] = 100

    monkeypatch.setattr(
        pipeline,
        "_apply_final_receipt_output_repairs",
        apply_late_repairs,
    )

    selected = pipeline._select_receipt_postprocessed_candidate(
        base,
        history,
        "STORE\n合計 ¥100",
        0.9,
        {},
        "test-model",
        None,
    )

    assert selected["merchant"] == "LATE REPAIR"
    assert selected["line_items"][0]["total"] == 90
    assert late_repair_candidate["line_items"][0]["total"] == 90
    assert history[0]["postprocess_items_sum_gap"] == 0
    assert history[0]["postprocess_selected"] is True


def test_candidate_selection_validates_malformed_merchant_before_postprocess(
    monkeypatch,
):
    from receipt_parser import pipeline

    base = _extraction(merchant="VALID BASE")
    malformed = _extraction(merchant=123)
    history = _history(malformed)
    seen_merchants = []

    def postprocess(extracted, *_args, **_kwargs):
        seen_merchants.append(extracted["merchant"])
        assert isinstance(extracted["merchant"], str)
        return extracted

    monkeypatch.setattr(pipeline, "postprocess_receipt", postprocess)

    selected = pipeline._select_receipt_postprocessed_candidate(
        base,
        history,
        "VALID BASE\n合計 ¥100",
        0.9,
        {},
        "test-model",
        None,
    )

    assert selected["merchant"] == "VALID BASE"
    assert seen_merchants == ["VALID BASE"]
    assert history[0]["postprocess_selected"] is False
    assert history[0]["postprocess_warnings"][0].startswith(
        "Receipt schema validation failed"
    )


def test_location_expansion_rejects_hq_metadata_candidate(monkeypatch):
    from receipt_parser import pipeline

    source = _extraction(location="北丘")
    text = "TEST\n北丘\n本社 東京都北丘市中央1-2\n合計 ¥100"
    monkeypatch.setattr(
        pipeline,
        "extract_with_verification",
        lambda *_args, **_kwargs: (source, _history(source)),
    )
    monkeypatch.setattr(
        pipeline,
        "_select_receipt_postprocessed_candidate",
        lambda extracted, *_args, **_kwargs: dict(extracted),
    )
    monkeypatch.setattr(pipeline, "_location_needs_resolution", lambda *_args: False)

    extracted, _history_rows, _warnings, receipt = pipeline._run_extraction_pipeline(
        unified_text=text,
        raw_text=text,
        ocr_conf=0.9,
        doc_type="receipt",
        model="test-model",
        passes=1,
    )

    assert receipt is not None
    assert extracted["location"] == "北丘"


def test_utility_bill_postprocess_mutations_have_one_declared_owner(monkeypatch):
    from receipt_parser import pipeline

    source = _extraction(
        document_type="utility_bill",
        merchant="recipient",
        payment_method=None,
        service_type="sewage",
        billing_period=None,
        payer=None,
    )
    trace = []
    monkeypatch.setattr(
        pipeline,
        "extract_with_verification",
        lambda *_args, **_kwargs: (source, _history(source)),
    )

    reference_text = "水道料金\n受付番号\n123456789012345678901234"

    def postprocess(extracted, _text, payment_reference_text=None):
        assert payment_reference_text == reference_text
        extracted["merchant"] = None
        extracted["payment_method"] = "bank_payment"
        extracted["service_type"] = "water"
        extracted["billing_period"] = {
            "start": "2026-01-01",
            "end": "2026-01-31",
        }
        extracted["payer"] = "テスト タロウ"
        extracted["payment_reference"] = "1234567890"
        return extracted

    monkeypatch.setattr(pipeline, "postprocess_utility_bill", postprocess)

    pipeline._run_extraction_pipeline(
        unified_text="水道料金\n口座振替",
        raw_text="水道料金\n口座振替",
        ocr_conf=0.9,
        doc_type="utility_bill",
        model="test-model",
        passes=1,
        mutation_trace=trace,
        payment_reference_text=reference_text,
    )

    event = next(row for row in trace if row["stage"] == "utility_bill_postprocess")
    assert set(event["changes"]) == {
        "billing_period",
        "merchant",
        "payment_method",
        "payment_reference",
        "payer",
        "service_type",
    }
    assert set(event["changes"]) <= set(event["writes"])
    assert "raw_text" in event["reads"]


def test_payment_slip_payer_and_reference_have_one_declared_owner(monkeypatch):
    from receipt_parser import pipeline

    source = _extraction(
        document_type="payment_slip",
        payer=None,
        payment_reference=None,
    )
    trace = []
    monkeypatch.setattr(
        pipeline,
        "extract_with_verification",
        lambda *_args, **_kwargs: (source, _history(source)),
    )

    def postprocess(extracted, _text, raw_text=""):
        assert raw_text
        extracted["payer"] = "テスト タロウ"
        extracted["payment_reference"] = "1234567890123"
        return extracted

    monkeypatch.setattr(pipeline, "postprocess_payment_slip", postprocess)

    pipeline._run_extraction_pipeline(
        unified_text="払込票",
        raw_text="払込票\nご依頼人 テスト タロウ\n1234567890123",
        ocr_conf=0.9,
        doc_type="payment_slip",
        model="test-model",
        passes=1,
        mutation_trace=trace,
    )

    event = next(row for row in trace if row["stage"] == "payment_slip_postprocess")
    assert set(event["changes"]) == {"payer", "payment_reference"}
    assert set(event["changes"]) <= set(event["writes"])


def test_schema_failure_fails_closed_for_injected_text(monkeypatch):
    from receipt_parser import pipeline

    source = _extraction(line_items="invalid")
    monkeypatch.setattr(pipeline, "check_model_available", lambda _model: None)
    monkeypatch.setattr(
        pipeline,
        "extract_with_verification",
        lambda *_args, **_kwargs: (source, _history(source)),
    )
    monkeypatch.setattr(
        pipeline,
        "_select_receipt_postprocessed_candidate",
        lambda extracted, *_args, **_kwargs: dict(extracted),
    )

    result = pipeline.process_ocr_text("TEST\n合計 ¥100")

    assert result["_error"].startswith("Receipt schema validation failed")
    assert result["_warnings"] == [result["_error"]]


def test_schema_failure_fails_closed_for_digital_pdf(monkeypatch):
    from receipt_parser import pipeline, usage

    source = _extraction(line_items="invalid")
    monkeypatch.setattr(pipeline, "check_model_available", lambda _model: None)
    monkeypatch.setattr(usage, "track_document", lambda _path: None)
    monkeypatch.setattr(pipeline, "load_image", lambda _path: [object()])
    monkeypatch.setattr(
        pipeline,
        "try_extract_text_layer",
        lambda _path: "TEST\n合計 ¥100",
    )
    monkeypatch.setattr(
        pipeline,
        "extract_with_verification",
        lambda *_args, **_kwargs: (source, _history(source)),
    )
    monkeypatch.setattr(
        pipeline,
        "_select_receipt_postprocessed_candidate",
        lambda extracted, *_args, **_kwargs: dict(extracted),
    )

    result = pipeline.process_document(Path("digital.pdf"))

    assert result["_error"].startswith("Receipt schema validation failed")
    assert result["_warnings"] == [result["_error"]]


def test_schema_failure_fails_closed_for_scanned_document(monkeypatch):
    from receipt_parser import pipeline, usage
    from receipt_parser.ocr import OCRResult

    source = _extraction(line_items="invalid")
    ocr_result = OCRResult(
        blocks=[{}, {}, {}],
        chosen_text="TEST\n合計 ¥100",
        source="mock",
    )
    monkeypatch.setattr(pipeline, "check_model_available", lambda _model: None)
    monkeypatch.setattr(usage, "track_document", lambda _path: None)
    monkeypatch.setattr(pipeline, "load_image", lambda _path: [object()])
    monkeypatch.setattr(pipeline, "init_cloud_vision", lambda: object())
    monkeypatch.setattr(pipeline, "run_cloud_vision", lambda *_args, **_kwargs: ocr_result)
    monkeypatch.setattr(
        pipeline,
        "blocks_to_structured_text",
        lambda _blocks: "TEST\n合計 ¥100",
    )
    monkeypatch.setattr(pipeline, "compute_ocr_confidence", lambda _blocks: 0.9)
    monkeypatch.setattr(
        pipeline,
        "extract_with_verification",
        lambda *_args, **_kwargs: (source, _history(source)),
    )
    monkeypatch.setattr(
        pipeline,
        "_select_receipt_postprocessed_candidate",
        lambda extracted, *_args, **_kwargs: dict(extracted),
    )

    result = pipeline.process_document(Path("scan.png"))

    assert result["_error"].startswith("Receipt schema validation failed")
    assert result["_warnings"] == [result["_error"]]


def test_scanned_layout_blocks_reach_final_output_reconciliation(monkeypatch):
    from receipt_parser import pipeline, usage
    from receipt_parser.ocr import OCRResult

    source = _extraction()
    layout_block = {
        "text": "商品甲",
        "x": 100,
        "y": 100,
        "bbox": [[100, 100], [200, 100], [200, 120], [100, 120]],
    }
    ocr_result = OCRResult(
        blocks=[{}, {}, {}],
        layout_blocks=[layout_block],
        layout_trusted=True,
        chosen_text="TEST 合計 ¥100",
        confidence=0.9,
        source="mock",
    )
    monkeypatch.setattr(pipeline, "check_model_available", lambda _model: None)
    monkeypatch.setattr(usage, "track_document", lambda _path: None)
    monkeypatch.setattr(pipeline, "load_image", lambda _path: [object()])
    monkeypatch.setattr(pipeline, "init_cloud_vision", lambda: object())
    monkeypatch.setattr(pipeline, "run_cloud_vision", lambda *_args, **_kwargs: ocr_result)
    monkeypatch.setattr(
        pipeline,
        "blocks_to_structured_text",
        lambda _blocks: "TEST\n合計 ¥100",
    )
    monkeypatch.setattr(pipeline, "compute_ocr_confidence", lambda _blocks: 0.9)
    monkeypatch.setattr(
        pipeline,
        "extract_with_verification",
        lambda *_args, **_kwargs: (source, _history(source)),
    )
    monkeypatch.setattr(
        pipeline,
        "_select_receipt_postprocessed_candidate",
        lambda extracted, *_args, **_kwargs: dict(extracted),
    )
    monkeypatch.setattr(pipeline, "_location_needs_resolution", lambda *_args: False)
    captured = {}

    def prepare(receipt, _ocr_text=None, **kwargs):
        captured["ocr_layout_blocks"] = kwargs.get("ocr_layout_blocks")
        return receipt.model_dump()

    monkeypatch.setattr(pipeline, "_prepare_receipt_output_payload", prepare)

    result = pipeline.process_document(Path("scan.png"))

    assert "_error" not in result
    assert "_ocr_layout_blocks" not in result
    assert captured["ocr_layout_blocks"] == [{**layout_block, "page": 0}]

    captured_result = pipeline.process_document(
        Path("scan.png"), capture_ocr_layout=True,
    )
    assert captured_result["_ocr_layout_blocks"] == [{**layout_block, "page": 0}]
    assert captured_result["_ocr_layout_trusted"] is True
    assert captured_result["_ocr_page_count"] == 1
    assert captured_result["_ocr_text"] == "TEST 合計 ¥100"
    assert captured_result["_ocr_replay_text"] == "TEST\n合計 ¥100"
    assert captured_result["_ocr_replay_confidence"] == 0.9

    ocr_result.layout_trusted = False
    legacy_result = pipeline.process_document(
        Path("scan.png"), capture_ocr_layout=True,
    )
    assert legacy_result["_ocr_layout_blocks"] == []
    assert legacy_result["_ocr_layout_trusted"] is False

    ocr_result.layout_trusted = True
    monkeypatch.setattr(pipeline, "load_image", lambda _path: [object(), object()])
    multi_page_result = pipeline.process_document(
        Path("scan.png"), capture_ocr_layout=True,
    )
    assert multi_page_result["_ocr_page_count"] == 2
    assert multi_page_result["_ocr_layout_blocks"] == [{**layout_block, "page": 0}]


def test_injected_layout_blocks_reach_pipeline_and_final_output_reconciliation(
    monkeypatch,
):
    from receipt_parser import pipeline

    source = _extraction()
    layout = [{
        "text": "商品甲",
        "x": 100,
        "y": 100,
        "bbox": [[100, 100], [200, 100], [200, 120], [100, 120]],
        "page": 0,
    }]
    captured = {}
    monkeypatch.setattr(pipeline, "check_model_available", lambda _model: None)
    monkeypatch.setattr(
        pipeline,
        "extract_with_verification",
        lambda *_args, **_kwargs: (source, _history(source)),
    )

    def select(extracted, *_args, **kwargs):
        captured["pipeline"] = kwargs.get("ocr_layout_blocks")
        return dict(extracted)

    def prepare(receipt, _ocr_text=None, **kwargs):
        captured["final_output"] = kwargs.get("ocr_layout_blocks")
        return receipt.model_dump()

    monkeypatch.setattr(pipeline, "_select_receipt_postprocessed_candidate", select)
    monkeypatch.setattr(pipeline, "_prepare_receipt_output_payload", prepare)
    monkeypatch.setattr(pipeline, "_location_needs_resolution", lambda *_args: False)

    result = pipeline.process_ocr_text(
        "TEST\n合計 ¥100",
        ocr_layout_blocks=layout,
    )

    assert "_error" not in result
    assert captured == {"pipeline": layout, "final_output": layout}


def test_injected_replay_preserves_scored_text_semantics_and_confidence(monkeypatch):
    from receipt_parser import pipeline

    source = _extraction()
    receipt = pipeline.Receipt.model_validate(source)
    injected_text = "ＴＥＳＴ    商品\n4901234567894\n合計 ￥１００"
    normalized_text = pipeline.normalize_fullwidth(injected_text)
    scored_text = pipeline.strip_barcode_lines(normalized_text)
    captured = {}
    monkeypatch.setattr(pipeline, "check_model_available", lambda _model: None)

    def run_pipeline(**kwargs):
        captured["run"] = kwargs
        return source, _history(source), [], receipt

    def prepare(prepared_receipt, ocr_text=None, **_kwargs):
        captured["final_repair_text"] = ocr_text
        return prepared_receipt.model_dump()

    monkeypatch.setattr(pipeline, "_run_extraction_pipeline", run_pipeline)
    monkeypatch.setattr(pipeline, "_prepare_receipt_output_payload", prepare)

    result = pipeline.process_ocr_text(
        injected_text,
        apply_user_rules=False,
        ocr_confidence=0.73,
    )

    assert captured["run"]["unified_text"] == scored_text
    assert captured["run"]["raw_text"] == normalized_text
    assert captured["run"]["payment_reference_text"] == normalized_text
    assert captured["run"]["ocr_conf"] == 0.73
    assert captured["final_repair_text"] == normalized_text
    assert result["_ocr_confidence"] == 0.73

    legacy = pipeline.process_ocr_text(injected_text, apply_user_rules=False)
    assert legacy["_ocr_confidence"] == 0.9

    for invalid in (True, "0.9", float("nan"), 1.01):
        with pytest.raises(ValueError, match="OCR confidence"):
            pipeline.process_ocr_text(injected_text, ocr_confidence=invalid)


def test_injected_supplemental_ocr_is_field_only_traced_and_never_acquired(
    monkeypatch,
):
    from receipt_parser import pipeline

    source = _extraction(
        location="旧店",
        subtotal=100,
        line_items=[{
            "description": "商品",
            "qty": 1,
            "unit_price": 100,
            "total": 100,
            "tax_category": "10%",
        }],
    )
    receipt = pipeline.Receipt.model_validate(source)
    events = []
    captured = {}
    monkeypatch.setattr(pipeline, "check_model_available", lambda _model: None)
    monkeypatch.setattr(
        pipeline,
        "acquire_supplemental_ocr_evidence",
        lambda *_args, **_kwargs: pytest.fail(
            "process_ocr_text attempted supplemental Vision acquisition"
        ),
    )

    def run_pipeline(**kwargs):
        captured["trace"] = kwargs["mutation_trace"]
        return source, _history(source), [], receipt

    monkeypatch.setattr(pipeline, "_run_extraction_pipeline", run_pipeline)
    monkeypatch.setattr(
        pipeline,
        "_prepare_receipt_output_payload",
        lambda prepared, *_args, **_kwargs: prepared.model_dump(),
    )
    real_finalize = pipeline._finalize_receipt_result

    def finalize(*args, **kwargs):
        result = real_finalize(*args, **kwargs)
        events.append("finalize")
        return result

    monkeypatch.setattr(pipeline, "_finalize_receipt_result", finalize)

    def propose(payload, _evidence):
        events.append("supplemental")
        candidate = deepcopy(payload)
        candidate["location"] = "新店"
        candidate["total"] = 999
        candidate["line_items"][0]["description"] = "not allowed"
        candidate["line_items"][0]["tax_category"] = "8%"
        return candidate, {
            "accepted_fields": ["location", "tax_categories"],
        }

    monkeypatch.setattr(pipeline, "apply_supplemental_ocr_evidence", propose)

    def confidence(payload, warnings):
        events.append("confidence")
        assert payload["total"] == 100
        assert warnings == []
        return {"line_items": 0.9}

    monkeypatch.setattr(pipeline, "_compute_posthoc_confidence", confidence)

    def on_stage(stage, _detail, _progress, payload=None):
        del payload
        if stage in {"validate", "done"}:
            events.append(stage)

    result = pipeline.process_ocr_text(
        "ALIAS SOURCE\n商品 ¥100\n合計 ¥100",
        apply_user_rules=False,
        debug=True,
        supplemental_ocr_evidence={"sealed": True},
        on_stage=on_stage,
    )

    assert events == ["finalize", "supplemental", "validate", "confidence", "done"]
    assert result["location"] == "新店"
    assert result["total"] == 100
    assert result["line_items"][0]["description"] == "商品"
    assert result["line_items"][0]["tax_category"] == "8%"
    assert result["_warnings"] == []
    assert result["_line_items_reliable"] is True
    assert result["_llm_confidence"] == {"line_items": 0.9}
    assert result["_receipt_mutation_trace"] is captured["trace"]
    event = next(
        event
        for event in result["_receipt_mutation_trace"]
        if event["stage"] == "supplemental_ocr_field_recovery"
    )
    assert event["owner_phase"] == "supplemental_ocr_field_recovery"
    assert set(event["changes"]) == {"location", "line_items"}
    assert set(event["writes"]) == {"location", "line_items"}


def test_invalid_supplemental_ocr_evidence_is_an_exact_noop():
    from receipt_parser import pipeline

    trace = []
    result = {
        **_extraction(location="旧店"),
        "_warnings": ["existing warning"],
        "_line_items_reliable": False,
        "_receipt_mutation_trace": trace,
    }
    before = deepcopy(result)

    pipeline._apply_final_supplemental_ocr_evidence(
        result,
        {"invalid": True},
        mutation_trace=trace,
    )

    assert result == before
    assert result["_receipt_mutation_trace"] is trace


def test_supplemental_ocr_evidence_is_receipt_only(monkeypatch):
    from receipt_parser import pipeline

    result = {
        **_extraction(document_type="utility_bill", location="旧店"),
        "_warnings": [],
        "_line_items_reliable": True,
    }
    before = deepcopy(result)
    monkeypatch.setattr(
        pipeline,
        "apply_supplemental_ocr_evidence",
        lambda *_args: pytest.fail("non-receipt evidence was applied"),
    )

    pipeline._apply_final_supplemental_ocr_evidence(result, {"sealed": True})

    assert result == before


def test_scanned_single_page_receipt_maps_supplemental_ocr_cache_modes(monkeypatch):
    from receipt_parser import pipeline, usage
    from receipt_parser.ocr import OCRResult

    source = _extraction()
    receipt = pipeline.Receipt.model_validate(source)
    image = object()
    ocr_result = OCRResult(
        blocks=[{}, {}, {}],
        chosen_text="TEST\n商品 ¥100\n合計 ¥100",
        source="mock",
    )
    modes = []
    client = object()
    monkeypatch.setattr(pipeline, "check_model_available", lambda _model: None)
    monkeypatch.setattr(usage, "track_document", lambda _path: None)
    monkeypatch.setattr(pipeline, "load_image", lambda _path: [image])
    monkeypatch.setattr(pipeline, "init_cloud_vision", lambda: client)
    monkeypatch.setattr(
        pipeline,
        "run_cloud_vision",
        lambda *_args, **_kwargs: ocr_result,
    )
    monkeypatch.setattr(
        pipeline,
        "blocks_to_structured_text",
        lambda _blocks: "TEST\n商品 ¥100\n合計 ¥100",
    )
    monkeypatch.setattr(pipeline, "compute_ocr_confidence", lambda _blocks: 0.9)
    monkeypatch.setattr(
        pipeline,
        "_run_extraction_pipeline",
        lambda **_kwargs: (source, _history(source), [], receipt),
    )
    monkeypatch.setattr(
        pipeline,
        "_prepare_receipt_output_payload",
        lambda prepared, *_args, **_kwargs: prepared.model_dump(),
    )

    def acquire(acquired_image, *, mode, client: object):
        assert acquired_image is image
        modes.append((mode, client))
        return None

    monkeypatch.setattr(pipeline, "acquire_supplemental_ocr_evidence", acquire)

    pipeline.process_document(Path("scan.png"))
    pipeline.process_document(Path("scan.png"), skip_ocr_cache=True)
    pipeline.process_document(Path("scan.png"), ocr_cache_only=True)

    assert modes == [("normal", client), ("fresh", client), ("cache_only", client)]

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("supplemental unavailable")

    monkeypatch.setattr(pipeline, "acquire_supplemental_ocr_evidence", unavailable)
    assert pipeline.process_document(Path("scan.png"))["total"] == 100
    with pytest.raises(RuntimeError, match="supplemental unavailable"):
        pipeline.process_document(Path("scan.png"), skip_ocr_cache=True)
    with pytest.raises(RuntimeError, match="supplemental unavailable"):
        pipeline.process_document(Path("scan.png"), ocr_cache_only=True)

    with pytest.raises(ValueError, match="mutually exclusive"):
        pipeline.process_document(
            Path("scan.png"),
            skip_ocr_cache=True,
            ocr_cache_only=True,
        )


@pytest.mark.parametrize(
    ("path", "text", "page_count"),
    [
        (Path("digital.pdf"), "TEST\n商品 ¥100\n合計 ¥100", 1),
        (Path("scan.png"), "電気料金\nご使用量\n請求額 ¥100", 1),
        (Path("scan.png"), "TEST\n商品 ¥100\n合計 ¥100", 2),
    ],
)
def test_supplemental_ocr_skips_digital_nonreceipt_and_multipage_documents(
    monkeypatch,
    path,
    text,
    page_count,
):
    from receipt_parser import pipeline, usage
    from receipt_parser.ocr import OCRResult

    doc_type = pipeline.detect_document_type(text)
    source = _extraction(document_type=doc_type)
    receipt = pipeline.Receipt.model_validate(source)
    ocr_result = OCRResult(
        blocks=[{}, {}, {}],
        chosen_text=text,
        source="mock",
    )
    monkeypatch.setattr(pipeline, "check_model_available", lambda _model: None)
    monkeypatch.setattr(usage, "track_document", lambda _path: None)
    monkeypatch.setattr(pipeline, "load_image", lambda _path: [object()] * page_count)
    monkeypatch.setattr(
        pipeline,
        "try_extract_text_layer",
        lambda _path: text if path.suffix == ".pdf" else None,
    )
    monkeypatch.setattr(pipeline, "init_cloud_vision", lambda: object())
    monkeypatch.setattr(
        pipeline,
        "run_cloud_vision",
        lambda *_args, **_kwargs: ocr_result,
    )
    monkeypatch.setattr(pipeline, "blocks_to_structured_text", lambda _blocks: text)
    monkeypatch.setattr(pipeline, "compute_ocr_confidence", lambda _blocks: 0.9)
    monkeypatch.setattr(
        pipeline,
        "_run_extraction_pipeline",
        lambda **_kwargs: (source, _history(source), [], receipt),
    )
    monkeypatch.setattr(
        pipeline,
        "_prepare_receipt_output_payload",
        lambda prepared, *_args, **_kwargs: prepared.model_dump(),
    )
    monkeypatch.setattr(
        pipeline,
        "acquire_supplemental_ocr_evidence",
        lambda *_args, **_kwargs: pytest.fail("ineligible document acquired OCR"),
    )

    result = pipeline.process_document(path)

    assert result["document_type"] == doc_type
