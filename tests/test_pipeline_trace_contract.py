"""Contracts for pipeline-owned receipt mutations and digital PDFs."""

from pathlib import Path


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
        chosen_text="TEST\n合計 ¥100",
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
    assert captured["ocr_layout_blocks"] == [{**layout_block, "page": 0}]
