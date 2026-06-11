"""Fast unit tests — no cloud APIs or Ollama needed."""

import json
import os
import shutil
import tempfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch, MagicMock

import cv2
import numpy as np
import pytest

from receipt_parser.schema import Receipt, generate_extraction_prompt, get_debug_color_map
import receipt_parser.llm as llm_module
from receipt_parser.llm import get_ollama_schema, _extract_confidence
from receipt_parser.validation import validate_receipt
from receipt_parser.normalize import normalize_fullwidth, clean_handwritten_ocr
from receipt_parser.ocr import compute_ocr_confidence, OCRResult
from receipt_parser.checks import check_time, check_tree_edit_distance
from scripts import add_flagged_receipts as flagged_exporter


# --- Tree Edit Distance tests ---

def test_tree_ed_identical():
    d = {"total": 100, "merchant": "Test", "line_items": [{"description": "a", "total": 100}]}
    result = check_tree_edit_distance(d, d)
    assert result["score"] == 1.0
    assert result["pass"]


def test_tree_ed_wrong_value():
    truth = {"total": 100, "merchant": "Test"}
    result_d = {"total": 100, "merchant": "Wrong", "extra": "value"}
    result = check_tree_edit_distance(result_d, truth)
    assert result["score"] < 1.0


def test_tree_ed_missing_line_item():
    truth = {"line_items": [{"description": "a", "total": 50}, {"description": "b", "total": 50}]}
    result_d = {"line_items": [{"description": "a", "total": 50}]}
    result = check_tree_edit_distance(result_d, truth)
    assert result["score"] < 1.0


def test_tree_ed_completely_wrong():
    truth = {"total": 100, "merchant": "Store", "date": "2026-01-01"}
    result_d = {"total": 999, "merchant": "Wrong", "date": "1999-01-01"}
    result = check_tree_edit_distance(result_d, truth)
    assert result["score"] < 0.5


def test_time_check_tolerates_leading_zero_difference():
    result = check_time({"time": "9:53"}, {"time": "09:53"})
    assert result["pass"]


# --- PROD flagged receipt fixture exporter tests ---

def _export_template():
    return flagged_exporter.load_template_shape(
        Path(__file__).parent / "fixtures" / "_truth_template.json"
    )


def test_prod_export_truth_shape_matches_template_order():
    template = _export_template()
    row = {
        "id": "receipt-id",
        "image_path": "images/receipt.jpg",
        "updated_at": "2026-06-11T00:00:00",
        "document_type": "receipt",
        "merchant": "Store",
        "date": "2026-06-10",
        "time": "09:53",
        "location": "Tokyo",
        "currency": "JPY",
        "total": 1100,
        "payment_method": "cash",
        "account_number": None,
        "points_used": 100,
        "amount_paid": 1000,
        "subtotal": 1000,
        "service_type": None,
        "payer": None,
        "payment_reference": None,
        "alias_id": "not-exported",
        "notes": "not-exported",
        "line_items": [
            {
                "id": "line-item-id",
                "description": "りんご",
                "description_clean": "not-exported",
                "category_id": "not-exported",
                "qty": 2,
                "unit_price": 500,
                "total": 1000,
                "tax_category": "10%",
                "discount": 0,
                "discount_rate": "",
            }
        ],
        "tax_entries": [
            {"id": "tax-id", "rate": "10%", "label": "内税", "amount": 100}
        ],
        "billing_period": None,
        "usage_data": None,
    }

    truth = flagged_exporter.prod_receipt_to_truth(row, template)

    assert list(truth) == list(template)
    assert truth["currency"] == "JPY"
    assert truth["total"] == 1100
    assert truth["points_used"] == 100
    assert truth["amount_paid"] == 1000
    assert truth["line_items"] == [
        {
            "description": "りんご",
            "qty": 2,
            "unit_price": 500,
            "total": 1000,
            "tax_category": "10%",
            "discount": 0,
            "discount_rate": "",
        }
    ]
    assert truth["taxes"] == [{"rate": "10%", "label": "内税", "amount": 100}]
    assert truth["billing_period"] == {"start": None, "end": None}
    assert truth["usage"] == {
        "amount": None,
        "unit": None,
        "cost_per": None,
        "meter_previous": None,
        "meter_current": None,
    }
    assert "_llm_prompt" not in truth
    assert "_comment" not in json.dumps(truth, ensure_ascii=False)
    assert "alias_id" not in json.dumps(truth, ensure_ascii=False)
    assert "category_id" not in json.dumps(truth, ensure_ascii=False)
    assert "description_clean" not in json.dumps(truth, ensure_ascii=False)


def test_prod_export_empty_receipt_sections_keep_template_pattern():
    template = _export_template()
    row = {
        "id": "empty-id",
        "image_path": "images/empty.jpg",
        "updated_at": "2026-06-11T00:00:00",
        "currency": None,
        "line_items": [],
        "tax_entries": [],
        "billing_period": None,
        "usage_data": None,
    }

    truth = flagged_exporter.prod_receipt_to_truth(row, template)

    assert list(truth) == list(template)
    assert truth["currency"] == "JPY"
    assert truth["line_items"] == []
    assert truth["taxes"] == []
    assert truth["billing_period"] == {"start": None, "end": None}
    assert truth["usage"] == {
        "amount": None,
        "unit": None,
        "cost_per": None,
        "meter_previous": None,
        "meter_current": None,
    }


def test_prod_export_utility_sections_are_adapted():
    template = _export_template()
    row = {
        "id": "utility-id",
        "image_path": "images/utility.jpg",
        "updated_at": "2026-06-11T00:00:00",
        "document_type": "utility_bill",
        "currency": "JPY",
        "total": 3210,
        "line_items": [],
        "tax_entries": [],
        "billing_period": {"start_date": "2026-05-01", "end_date": "2026-05-31"},
        "usage_data": {
            "amount": 12.5,
            "unit": "m3",
            "cost_per": 120.5,
            "meter_previous": 100.0,
            "meter_current": 112.5,
        },
    }

    truth = flagged_exporter.prod_receipt_to_truth(row, template)

    assert truth["billing_period"] == {"start": "2026-05-01", "end": "2026-05-31"}
    assert truth["usage"] == {
        "amount": 12.5,
        "unit": "m3",
        "cost_per": 120.5,
        "meter_previous": 100.0,
        "meter_current": 112.5,
    }


def test_manifest_freshness_requires_updated_at_checksum_and_files():
    row = {"id": "receipt-id", "updated_at": "2026-06-11T00:00:00"}
    entry = {"updated_at": "2026-06-11T00:00:00", "checksum": "abc"}

    assert flagged_exporter.manifest_entry_current(
        entry, row, "abc", fixture_files_exist=True
    )
    assert not flagged_exporter.manifest_entry_current(
        entry, {**row, "updated_at": "later"}, "abc", fixture_files_exist=True
    )
    assert not flagged_exporter.manifest_entry_current(
        entry, row, "different", fixture_files_exist=True
    )
    assert not flagged_exporter.manifest_entry_current(
        entry, row, "abc", fixture_files_exist=False
    )


def test_fixture_overwrite_guard():
    scratch = Path(tempfile.mkdtemp(dir=Path(__file__).parents[1] / "local"))
    try:
        image_path = scratch / "receipt_1.jpg"
        truth_path = scratch / "receipt_1_truth.json"
        image_path.write_bytes(b"already here")

        with pytest.raises(FileExistsError):
            flagged_exporter.ensure_can_write_fixture(
                image_path, truth_path, overwrite=False
            )

        flagged_exporter.ensure_can_write_fixture(
            image_path, truth_path, overwrite=True
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# --- Normalization tests ---

def test_fullwidth_digits():
    assert normalize_fullwidth("￥１，５００") == "¥1,500"


def test_fullwidth_mixed():
    assert normalize_fullwidth("２０２６年") == "2026年"


def test_yen_preserved():
    result = normalize_fullwidth("¥100")
    assert "¥" in result
    assert "100" in result


def test_handwritten_cleanup_fixes_yen_as_1():
    """The ¥ sign misread as 1: ¥3000 → 13000 should be fixed to 金額:3000."""
    text = "金額\n13000\n宗像市産後ケア"
    result = clean_handwritten_ocr(text)
    assert "金額:3000" in result
    assert "13000" not in result


def test_handwritten_cleanup_skips_printed():
    """Printed receipts (with 小計/合計) should not be cleaned."""
    text = "小計 ¥837\n合計 ¥903\n¥3000"
    result = clean_handwritten_ocr(text)
    assert result == text


# --- Validation tests ---

def test_valid_receipt_no_warnings():
    receipt = Receipt(
        total=324, subtotal=300,
        line_items=[{"description": "おにぎり", "qty": 2, "unit_price": 150, "total": 300}],
        taxes=[{"rate": "8%", "amount": 24}],
    )
    assert validate_receipt(receipt) == []


def test_bad_line_item_math():
    receipt = Receipt(
        total=300,
        line_items=[{"description": "item", "qty": 2, "unit_price": 100, "total": 300}],
    )
    warnings = validate_receipt(receipt)
    assert any("200" in w for w in warnings)


def test_bad_subtotal_plus_tax():
    receipt = Receipt(
        total=3240, subtotal=300,
        taxes=[{"rate": "8%", "amount": 24}],
    )
    warnings = validate_receipt(receipt)
    assert len(warnings) > 0


def test_tax_inclusive_no_false_warning():
    """内税 (tax-inclusive): subtotal is pre-tax base (total - tax)."""
    receipt = Receipt(
        total=324, subtotal=300,
        taxes=[{"rate": "8%", "label": "内税", "amount": 24}],
    )
    assert validate_receipt(receipt) == []


# --- OCR confidence tests ---

def test_ocr_confidence_empty():
    assert compute_ocr_confidence([]) == 0.0


def test_ocr_confidence_uniform():
    blocks = [
        {"text": "hello", "confidence": 0.95},
        {"text": "world", "confidence": 0.95},
    ]
    assert abs(compute_ocr_confidence(blocks) - 0.95) < 0.001


def test_ocr_confidence_weighted():
    blocks = [
        {"text": "a", "confidence": 0.5},      # 1 char, weight 1
        {"text": "bbbbb", "confidence": 1.0},   # 5 chars, weight 5
    ]
    expected = (0.5 * 1 + 1.0 * 5) / 6  # 0.9167
    assert abs(compute_ocr_confidence(blocks) - expected) < 0.001


# --- LLM confidence extraction tests ---

def test_extract_confidence_valid():
    data = {"merchant": "test", "_confidence": {"merchant": 0.9, "total": 0.8}}
    conf = _extract_confidence(data)
    assert conf == {"merchant": 0.9, "total": 0.8}
    assert "_confidence" not in data  # should be popped


def test_extract_confidence_invalid_values():
    data = {"_confidence": {"merchant": 1.5, "total": "bad", "date": -0.1, "subtotal": 0.7}}
    conf = _extract_confidence(data)
    assert conf == {"subtotal": 0.7}


def test_extract_confidence_missing():
    data = {"merchant": "test"}
    conf = _extract_confidence(data)
    assert conf is None


# --- Era date conversion tests ---

def test_era_reiwa():
    from receipt_parser.patterns import era_to_western_year as _era_to_western_year
    assert _era_to_western_year(8, "令和") == 2026
    assert _era_to_western_year(1, "令和") == 2019


def test_era_heisei():
    from receipt_parser.patterns import era_to_western_year as _era_to_western_year
    assert _era_to_western_year(31, "平成") == 2019
    assert _era_to_western_year(1, "平成") == 1989


def test_era_default_assumes_reiwa():
    from receipt_parser.patterns import era_to_western_year as _era_to_western_year
    assert _era_to_western_year(8) == 2026


def test_era_invalid():
    from receipt_parser.patterns import era_to_western_year as _era_to_western_year
    assert _era_to_western_year(0) is None
    assert _era_to_western_year(100) is None


# --- Pydantic coercion tests ---

def test_pydantic_coerces_quantity_alias():
    r = Receipt(**{"total": 100, "line_items": [
        {"name": "item1", "quantity": 2, "unit_price": 50, "total": 100}
    ]})
    assert r.line_items[0].description == "item1"
    assert r.line_items[0].qty == 2


def test_pydantic_coerces_tax_category():
    r = Receipt(**{"total": 100, "line_items": [
        {"description": "item", "total": 100, "tax_category": "8percent"}
    ]})
    assert r.line_items[0].tax_category == "8%"


def test_pydantic_coerces_taxes_from_number():
    r = Receipt(**{"total": 100, "taxes": 24})
    assert len(r.taxes) == 1
    assert r.taxes[0].amount == 24


def test_pydantic_coerces_string_amounts():
    r = Receipt(**{"total": "1,500", "subtotal": "1,200"})
    assert r.total == 1500.0
    assert r.subtotal == 1200.0


# --- Configurable tax rates tests ---

def test_valid_tax_rates_constant():
    from receipt_parser.schema import VALID_TAX_RATES, REDUCED_RATE, STANDARD_RATE, EXEMPT_RATE
    assert REDUCED_RATE in VALID_TAX_RATES
    assert STANDARD_RATE in VALID_TAX_RATES
    assert EXEMPT_RATE in VALID_TAX_RATES


# --- Schema tests ---

def test_receipt_model_json_schema():
    schema = Receipt.model_json_schema()
    assert "total" in schema.get("properties", {})
    assert "line_items" in schema.get("properties", {})


def test_ollama_schema_no_refs():
    schema_str = json.dumps(get_ollama_schema())
    assert "$ref" not in schema_str
    assert "$defs" not in schema_str


def test_prompt_includes_hints_and_aliases():
    system_prompt, user_prompt = generate_extraction_prompt("test OCR text")
    prompt = system_prompt + user_prompt
    assert "merchant" in prompt
    assert "PAGE" in prompt or "OCR TEXT" in prompt
    assert "店名" in prompt
    assert "合計" in prompt
    assert "Look for labels:" in prompt


def test_sanity_retry_records_rejected_candidates(monkeypatch):
    base = {
        "document_type": "receipt",
        "line_items": [
            {"description": "A", "qty": 1, "unit_price": 100, "total": 100},
            {"description": "B", "qty": 1, "unit_price": 100, "total": 100},
        ],
        "subtotal": 300,
        "total": 330,
        "taxes": [],
        "currency": "JPY",
    }
    same_gap = {
        **base,
        "line_items": [
            {"description": "A", "qty": 1, "unit_price": 100, "total": 100},
            {"description": "B", "qty": 1, "unit_price": 100, "total": 100},
        ],
    }
    worse_gap = {
        **base,
        "line_items": [
            {"description": "A", "qty": 1, "unit_price": 150, "total": 150},
        ],
    }
    attempts = iter([same_gap, worse_gap])

    monkeypatch.setattr(
        llm_module,
        "extract_with_llm",
        lambda *args, **kwargs: (base, llm_module.LLMResult(content="{}")),
    )

    def fake_alt(*args, **kwargs):
        return next(attempts), llm_module.LLMResult(content="{}"), None

    monkeypatch.setattr(llm_module, "_alternate_seed_extract_with_result", fake_alt)

    extracted, history = llm_module.extract_with_verification(
        "OCR", passes=1, validate_fn=lambda receipt: ["Sum of line items mismatch"]
    )

    sanity_entries = [h for h in history if h.get("retry_kind") == "sanity"]
    assert extracted == base
    assert len(sanity_entries) == 2
    assert all(entry["accepted"] is False for entry in sanity_entries)
    assert all(entry["rejection_reason"] == "items_sum_gap_not_improved"
               for entry in sanity_entries)
    assert [entry["items_sum_gap_after"] for entry in sanity_entries] == [100, 150]


def test_items_sum_gap_uses_total_minus_taxes_canonical_subtotal():
    extracted = {
        "line_items": [
            {"description": "A", "total": 100},
            {"description": "B", "total": 200},
        ],
        "subtotal": 500,
        "total": 330,
        "taxes": [{"rate": "unknown", "amount": 30}],
    }

    assert llm_module._items_sum_gap(extracted) == 0


def test_items_sum_gap_ignores_subtotal_that_contradicts_total_minus_taxes():
    extracted = {
        "line_items": [
            {"description": "A", "total": 100},
            {"description": "B", "total": 200},
        ],
        "subtotal": 300,
        "total": 330,
        "taxes": [{"rate": "unknown", "amount": 10}],
    }

    assert llm_module._items_sum_gap(extracted) == 20


def test_cross_prompt_rescue_accepts_validator_improving_candidate(monkeypatch):
    bad = {
        "document_type": "receipt",
        "line_items": [
            {"description": "A", "qty": 1, "unit_price": 100, "total": 100},
            {"description": "B", "qty": 1, "unit_price": 100, "total": 100},
        ],
        "subtotal": 300,
        "total": 330,
        "taxes": [],
        "currency": "JPY",
    }
    good = {
        **bad,
        "line_items": [
            {"description": "A", "qty": 1, "unit_price": 100, "total": 100},
            {"description": "B", "qty": 1, "unit_price": 200, "total": 200},
        ],
    }

    monkeypatch.setattr(
        llm_module,
        "extract_with_llm",
        lambda *args, **kwargs: (bad, llm_module.LLMResult(content="{}")),
    )
    monkeypatch.setattr(
        llm_module,
        "_alternate_seed_extract_with_result",
        lambda *args, **kwargs: (bad, llm_module.LLMResult(content="{}"), None),
    )
    monkeypatch.setattr(
        llm_module,
        "_cross_prompt_extract_with_result",
        lambda *args, **kwargs: (good, llm_module.LLMResult(content="{}"), None),
    )

    extracted, history = llm_module.extract_with_verification(
        "OCR", passes=1, validate_fn=lambda receipt: ["Sum of line items mismatch"]
    )

    cross_entries = [h for h in history if h.get("retry_kind") == "cross_prompt"]
    assert extracted == good
    assert len(cross_entries) == 1
    assert cross_entries[0]["accepted"] is True
    assert cross_entries[0]["items_sum_gap_before"] == 100
    assert cross_entries[0]["items_sum_gap_after"] == 0


def test_cross_prompt_rescue_rejects_financial_anchor_manipulation(monkeypatch):
    base = {
        "document_type": "receipt",
        "line_items": [
            {"description": "A", "qty": 1, "unit_price": 100, "total": 100},
            {"description": "B", "qty": 1, "unit_price": 100, "total": 100},
        ],
        "subtotal": 300,
        "total": 330,
        "taxes": [{"rate": "unknown", "amount": 30}],
        "currency": "JPY",
    }
    manipulative = {
        **base,
        "subtotal": 200,
        "taxes": [{"rate": "unknown", "amount": 130}],
    }

    monkeypatch.setattr(
        llm_module,
        "extract_with_llm",
        lambda *args, **kwargs: (base, llm_module.LLMResult(content="{}")),
    )
    monkeypatch.setattr(
        llm_module,
        "_alternate_seed_extract_with_result",
        lambda *args, **kwargs: (base, llm_module.LLMResult(content="{}"), None),
    )
    monkeypatch.setattr(
        llm_module,
        "_cross_prompt_extract_with_result",
        lambda *args, **kwargs: (manipulative, llm_module.LLMResult(content="{}"), None),
    )

    extracted, history = llm_module.extract_with_verification(
        "OCR", passes=1, validate_fn=lambda receipt: ["Sum of line items mismatch"]
    )

    cross_entries = [h for h in history if h.get("retry_kind") == "cross_prompt"]
    assert extracted == base
    assert len(cross_entries) == 1
    assert cross_entries[0]["accepted"] is False
    assert cross_entries[0]["rejection_reason"] == "items_sum_gap_not_improved"


def test_cross_prompt_history_records_anchored_accepted_candidate(monkeypatch):
    bad = {
        "document_type": "receipt",
        "line_items": [
            {"description": "A", "qty": 1, "unit_price": 100, "total": 100},
        ],
        "subtotal": 300,
        "total": 330,
        "taxes": [{"rate": "unknown", "amount": 30}],
        "currency": "JPY",
    }
    raw_alt = {
        **bad,
        "line_items": [
            {"description": "A", "qty": 1, "unit_price": 300, "total": 300},
        ],
        "subtotal": 200,
        "taxes": [{"rate": "unknown", "amount": 130}],
    }
    anchored_alt = {
        **raw_alt,
        "subtotal": 300,
        "taxes": [{"rate": "unknown", "amount": 30}],
    }

    monkeypatch.setattr(
        llm_module,
        "extract_with_llm",
        lambda *args, **kwargs: (bad, llm_module.LLMResult(content="{}")),
    )
    monkeypatch.setattr(
        llm_module,
        "_alternate_seed_extract_with_result",
        lambda *args, **kwargs: (bad, llm_module.LLMResult(content="{}"), None),
    )
    monkeypatch.setattr(
        llm_module,
        "_cross_prompt_extract_with_result",
        lambda *args, **kwargs: (raw_alt, llm_module.LLMResult(content="{}"), None),
    )

    extracted, history = llm_module.extract_with_verification(
        "OCR", passes=1, validate_fn=lambda receipt: ["Sum of line items mismatch"]
    )

    cross_entry = [h for h in history if h.get("retry_kind") == "cross_prompt"][0]
    assert extracted == anchored_alt
    assert cross_entry["accepted"] is True
    assert cross_entry["extraction"] == anchored_alt
    assert cross_entry["raw_extraction"] == raw_alt
    assert cross_entry["items_sum_gap_after"] == 0


def test_model_triage_runs_only_when_configured_and_validator_gap_remains(monkeypatch):
    bad = {
        "document_type": "receipt",
        "line_items": [
            {"description": "A", "qty": 1, "unit_price": 100, "total": 100},
        ],
        "subtotal": 300,
        "total": 330,
        "taxes": [{"rate": "unknown", "amount": 30}],
        "currency": "JPY",
    }
    good = {
        **bad,
        "line_items": [
            {"description": "A", "qty": 1, "unit_price": 300, "total": 300},
        ],
    }

    monkeypatch.setenv("RECEIPT_TRIAGE_MODELS", "openrouter/test-model")
    monkeypatch.setattr(
        llm_module,
        "extract_with_llm",
        lambda *args, **kwargs: (bad, llm_module.LLMResult(content="{}")),
    )
    monkeypatch.setattr(
        llm_module,
        "_alternate_seed_extract_with_result",
        lambda *args, **kwargs: (bad, llm_module.LLMResult(content="{}"), None),
    )
    monkeypatch.setattr(
        llm_module,
        "_cross_prompt_extract_with_result",
        lambda *args, **kwargs: (bad, llm_module.LLMResult(content="{}"), None),
    )
    monkeypatch.setattr(
        llm_module,
        "_model_triage_extract_with_result",
        lambda *args, **kwargs: (good, llm_module.LLMResult(content="{}"), None),
    )

    extracted, history = llm_module.extract_with_verification(
        "OCR", passes=1, validate_fn=lambda receipt: ["Sum of line items mismatch"]
    )

    triage_entries = [h for h in history if h.get("retry_kind") == "model_triage"]
    assert extracted == good
    assert len(triage_entries) == 1
    assert triage_entries[0]["triage_model"] == "openrouter/test-model"
    assert triage_entries[0]["accepted"] is True


def test_cross_prompt_and_model_triage_require_validator_item_sum_warning(monkeypatch):
    bad = {
        "document_type": "receipt",
        "line_items": [
            {"description": "A", "qty": 1, "unit_price": 100, "total": 100},
        ],
        "subtotal": 300,
        "total": 330,
        "taxes": [{"rate": "unknown", "amount": 30}],
        "currency": "JPY",
    }
    calls = {"cross_prompt": 0, "model_triage": 0}

    monkeypatch.setenv("RECEIPT_TRIAGE_MODELS", "openrouter/test-model")
    monkeypatch.setattr(
        llm_module,
        "extract_with_llm",
        lambda *args, **kwargs: (bad, llm_module.LLMResult(content="{}")),
    )
    monkeypatch.setattr(
        llm_module,
        "_alternate_seed_extract_with_result",
        lambda *args, **kwargs: (bad, llm_module.LLMResult(content="{}"), None),
    )

    def fake_cross_prompt(*args, **kwargs):
        calls["cross_prompt"] += 1
        return bad, llm_module.LLMResult(content="{}"), None

    def fake_model_triage(*args, **kwargs):
        calls["model_triage"] += 1
        return bad, llm_module.LLMResult(content="{}"), None

    monkeypatch.setattr(llm_module, "_cross_prompt_extract_with_result", fake_cross_prompt)
    monkeypatch.setattr(llm_module, "_model_triage_extract_with_result", fake_model_triage)

    extracted, history = llm_module.extract_with_verification(
        "OCR", passes=1, validate_fn=lambda receipt: ["Tax ratio check only"]
    )

    assert extracted == bad
    assert calls == {"cross_prompt": 0, "model_triage": 0}
    assert not [h for h in history if h.get("retry_kind") == "cross_prompt"]
    assert not [h for h in history if h.get("retry_kind") == "model_triage"]


def test_model_triage_rejects_partial_item_sum_improvement(monkeypatch):
    bad = {
        "document_type": "receipt",
        "line_items": [
            {"description": "A", "qty": 1, "unit_price": 100, "total": 100},
        ],
        "subtotal": 300,
        "total": 330,
        "taxes": [{"rate": "unknown", "amount": 30}],
        "currency": "JPY",
    }
    partial = {
        **bad,
        "line_items": [
            {"description": "A", "qty": 1, "unit_price": 290, "total": 290},
        ],
    }

    monkeypatch.setenv("RECEIPT_TRIAGE_MODELS", "openrouter/test-model")
    monkeypatch.setattr(
        llm_module,
        "extract_with_llm",
        lambda *args, **kwargs: (bad, llm_module.LLMResult(content="{}")),
    )
    monkeypatch.setattr(
        llm_module,
        "_alternate_seed_extract_with_result",
        lambda *args, **kwargs: (bad, llm_module.LLMResult(content="{}"), None),
    )
    monkeypatch.setattr(
        llm_module,
        "_cross_prompt_extract_with_result",
        lambda *args, **kwargs: (bad, llm_module.LLMResult(content="{}"), None),
    )
    monkeypatch.setattr(
        llm_module,
        "_model_triage_extract_with_result",
        lambda *args, **kwargs: (partial, llm_module.LLMResult(content="{}"), None),
    )

    extracted, history = llm_module.extract_with_verification(
        "OCR", passes=1, validate_fn=lambda receipt: ["Sum of line items mismatch"]
    )

    triage_entries = [h for h in history if h.get("retry_kind") == "model_triage"]
    assert extracted == bad
    assert len(triage_entries) == 1
    assert triage_entries[0]["accepted"] is False
    assert triage_entries[0]["items_sum_gap_before"] == 200
    assert triage_entries[0]["items_sum_gap_after"] == 10
    assert triage_entries[0]["rejection_reason"] == "items_sum_gap_not_closed"


def test_default_model_requires_deepseek_key_even_with_openrouter(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    assert llm_module._api_client_key_for_model(llm_module.DEFAULT_MODEL) == (
        "deepseek",
        llm_module.DEFAULT_MODEL,
    )
    try:
        llm_module.check_model_available(llm_module.DEFAULT_MODEL)
    except RuntimeError as exc:
        assert "DeepSeek API key is required" in str(exc)
    else:
        raise AssertionError("default model should not silently route through OpenRouter")


def test_explicit_openrouter_model_uses_openrouter_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    assert llm_module._api_client_key_for_model("openrouter/test-model") == (
        "openrouter",
        "test-model",
    )
    llm_module.check_model_available("openrouter/test-model")


def test_openrouter_credit_errors_get_friendly_message():
    raw_error = RuntimeError(
        "Error code: 402 - This request requires more credits. "
        "Visit https://openrouter.ai/settings/credits"
    )

    message = llm_module._format_llm_error(raw_error, "openrouter/test-model")

    assert message.startswith("openrouter_insufficient_credits:")
    assert "Top off your OpenRouter account" in message
    assert "RECEIPT_TRIAGE_MODELS" in message


def test_location_resolution_uses_parenthesized_area_code_and_header():
    from receipt_parser.pipeline import _location_needs_resolution, _resolve_location

    ocr_text = "\n".join([
        "くり早いと",
        "宗像",
        "サンリブ (0940) 38-0130",
        "領収証",
        "小計",
        "¥3,705",
    ])

    assert _location_needs_resolution(None, ocr_text)
    resolved, warning = _resolve_location(
        {"merchant": "サンリブ", "location": None},
        ocr_text,
        llm_module.DEFAULT_MODEL,
    )

    assert resolved == "宗像市"
    assert warning is None


def test_triage_max_tokens_defaults_lower_than_base_extraction(monkeypatch):
    monkeypatch.delenv("RECEIPT_TRIAGE_MAX_TOKENS", raising=False)
    assert llm_module._triage_max_tokens() == 2048

    monkeypatch.setenv("RECEIPT_TRIAGE_MAX_TOKENS", "99999")
    assert llm_module._triage_max_tokens() == 8192

    monkeypatch.setenv("RECEIPT_TRIAGE_MAX_TOKENS", "bad")
    assert llm_module._triage_max_tokens() == 2048


def test_item_subtotal_fix_does_not_chase_tax_base_when_items_match_canonical_subtotal():
    from receipt_parser.pipeline_receipt import _fix_items_from_subtotal

    extracted = {
        "total": 3282,
        "taxes": [
            {"rate": "8%", "label": "内税", "amount": 242},
            {"rate": "10%", "label": "内税", "amount": 1},
        ],
        "line_items": [
            {"description": "サルシッチャドッグ", "qty": 1, "unit_price": 380, "total": 380},
            {"description": "塩バター", "qty": 1, "unit_price": 180, "total": 180},
            {"description": "宗像牛カレーパン", "qty": 1, "unit_price": 280, "total": 280},
            {"description": "バゲット", "qty": 1, "unit_price": 300, "total": 300},
            {"description": "どっさりキャベツと白身フライ", "qty": 1, "unit_price": 350, "total": 350},
            {"description": "あまおう苺のデニッシ", "qty": 1, "unit_price": 450, "total": 450},
            {"description": "ヴルストクロワッサン", "qty": 1, "unit_price": 420, "total": 420},
            {"description": "てりたま", "qty": 1, "unit_price": 320, "total": 320},
            {"description": "シアン食パン", "qty": 1, "unit_price": 350, "total": 350},
            {"description": "レジ袋", "qty": 1, "unit_price": 10, "total": 10},
        ],
    }
    ocr_text = "\n".join([
        "8%対象",
        "3,272",
        "105-000008-000 ヴルストクロワッサン",
        "450*",
        "1",
        "420*",
    ])

    _fix_items_from_subtotal(extracted, ocr_text, {"subtotal": 3272})

    totals = [item["total"] for item in extracted["line_items"]]
    assert 420 in totals
    assert totals.count(450) == 1


def test_neighborhood_item_fix_does_not_chase_tax_base_when_items_match_canonical_subtotal():
    from receipt_parser.pipeline_receipt import _fix_item_totals_from_ocr_neighborhood

    items = [
        {"description": "あまおう苺のデニッシ", "qty": 1, "unit_price": 450, "total": 450},
        {"description": "ヴルストクロワッサン", "qty": 1, "unit_price": 420, "total": 420},
        {"description": "シアン食パン", "qty": 1, "unit_price": 350, "total": 350},
    ]
    prefix_items_sum = 3040 - sum(item["total"] for item in items)
    filler = {"description": "filler", "qty": 1, "unit_price": prefix_items_sum, "total": prefix_items_sum}
    items.insert(0, filler)
    ocr_text = "\n".join([
        "105-000002-000 あまおう苺のデニッシ",
        "105-000008-000 ヴルストクロワッサン",
        "350*",
        "450*",
        "1",
        "420*",
        "101-000001-000 シアン食パン",
        "1",
        "350*",
    ])

    _fix_item_totals_from_ocr_neighborhood(
        items, ocr_text, target_subtotal=3272, target_total=3282,
        canonical_subtotal=3039,
    )

    croissant = next(i for i in items if i["description"] == "ヴルストクロワッサン")
    assert croissant["total"] == 420
    assert croissant["unit_price"] == 420


def test_ocr_multiset_projection_uses_qty_detail_total():
    from receipt_parser.pipeline_receipt import _project_totals_to_ocr_multiset

    totals = [3, 598, 398, 398, 228, 228, 228, 268, 980, 98,
              128, 248, 98, 398, 158, 138, 98]
    extracted = {
        "subtotal": 4963,
        "total": 5362,
        "line_items": [
            {"description": "qty item", "qty": 2, "unit_price": 70, "total": 140},
            *[
                {"description": f"item {i}", "qty": 1, "unit_price": t, "total": t}
                for i, t in enumerate(totals)
            ],
        ],
    }
    ocr_text = "\n".join([
        "item 3除",
        "598",
        "398",
        "398",
        "228",
        "228*",
        "228*",
        "10",
        "2個 X70)",
        "268",
        "228",
        "980*",
        "98",
        "128",
        "248",
        "98*",
        "398",
        "158",
        "138",
        "小計",
    ])

    _project_totals_to_ocr_multiset(extracted, ocr_text)

    projected = sorted(int(item["total"]) for item in extracted["line_items"])
    assert projected == sorted([3, 98, 98, 128, 138, 140, 158, 228, 228,
                                228, 228, 248, 268, 398, 398, 398, 598, 980])
    qty_item = next(item for item in extracted["line_items"] if item["description"] == "qty item")
    assert qty_item["qty"] == 2
    assert qty_item["unit_price"] == 70
    assert qty_item["total"] == 140


def test_ocr_multiset_projection_uses_tax_derived_subtotal_target():
    from receipt_parser.pipeline_receipt import _project_totals_to_ocr_multiset

    extracted = {
        "subtotal": 3272,
        "total": 3282,
        "taxes": [
            {"rate": "8%", "label": "内税", "amount": 242},
            {"rate": "10%", "label": "内税", "amount": 1},
        ],
        "line_items": [
            {"description": "item 0", "qty": 1, "unit_price": 380, "total": 380},
            {"description": "item 1", "qty": 1, "unit_price": 180, "total": 180},
            {"description": "item 2", "qty": 1, "unit_price": 280, "total": 280},
            {"description": "item 3", "qty": 1, "unit_price": 300, "total": 300},
            {"description": "item 4", "qty": 1, "unit_price": 350, "total": 350},
            {"description": "item 5", "qty": 1, "unit_price": 450, "total": 450},
            {"description": "item 6", "qty": 1, "unit_price": 350, "total": 350},
            {"description": "item 7", "qty": 1, "unit_price": 320, "total": 320},
            {"description": "item 8", "qty": 1, "unit_price": 350, "total": 350},
            {"description": "item 9", "qty": 1, "unit_price": 10, "total": 10},
        ],
    }
    ocr_text = "\n".join([
        "サルシッチャドッグ",
        "380*",
        "塩バター",
        "180*",
        "宗像牛カレーパン",
        "280*",
        "バゲット",
        "300*",
        "どっさりキャベツと白身フライ",
        "350*",
        "あまおう苺のデニッシ",
        "450*",
        "ヴルストクロワッサン",
        "420*",
        "てりたま",
        "320*",
        "シアン食パン",
        "350*",
        "レジ袋",
        "10*",
        "8%対象",
        "3,272",
        "小計",
    ])

    _project_totals_to_ocr_multiset(extracted, ocr_text)

    projected = sorted(int(item["total"]) for item in extracted["line_items"])
    assert projected == sorted([10, 180, 280, 300, 320, 350, 350, 380, 420, 450])


def test_ocr_multiset_projection_preserves_description_price_order():
    from receipt_parser.pipeline_receipt import _project_totals_to_ocr_multiset

    extracted = {
        "subtotal": 3272,
        "total": 3282,
        "taxes": [
            {"rate": "8%", "label": "内税", "amount": 242},
            {"rate": "10%", "label": "内税", "amount": 1},
        ],
        "line_items": [
            {"description": "サルシッチャドッグ", "qty": 1, "unit_price": 380, "total": 380},
            {"description": "塩バター", "qty": 1, "unit_price": 180, "total": 180},
            {"description": "宗像牛カレーパン", "qty": 1, "unit_price": 280, "total": 280},
            {"description": "バゲット", "qty": 1, "unit_price": 300, "total": 300},
            {"description": "どっさりキャベツと白身フライ", "qty": 1, "unit_price": 350, "total": 350},
            {"description": "あまおう苺のデニッシュ", "qty": 1, "unit_price": 450, "total": 450},
            {"description": "ヴルストクロワッサン", "qty": 1, "unit_price": 350, "total": 350},
            {"description": "てりたま", "qty": 1, "unit_price": 320, "total": 320},
            {"description": "シアン食パン", "qty": 1, "unit_price": 350, "total": 350},
            {"description": "レジ袋", "qty": 1, "unit_price": 10, "total": 10},
        ],
    }
    ocr_text = "\n".join([
        "サルシッチャドッグ",
        "380*",
        "塩バター",
        "180*",
        "102-000008-000 宗像牛カレーパン",
        "1",
        "106-000001-000 バゲット",
        "1",
        "300※",
        "280*",
        "どっさりキャベツと白身フライ",
        "350*",
        "あまおう苺のデニッシ",
        "450*",
        "ヴルストクロワッサン",
        "420*",
        "てりたま",
        "320*",
        "シアン食パン",
        "350*",
        "レジ袋",
        "10*",
        "8%対象",
        "3,272",
        "小計",
    ])

    _project_totals_to_ocr_multiset(extracted, ocr_text)

    by_desc = {item["description"]: int(item["total"]) for item in extracted["line_items"]}
    assert by_desc["宗像牛カレーパン"] == 300
    assert by_desc["バゲット"] == 280
    assert by_desc["ヴルストクロワッサン"] == 420


def test_ocr_multiset_projection_preserves_discounted_net_total():
    from receipt_parser.pipeline_receipt import _project_totals_to_ocr_multiset

    extracted = {
        "subtotal": 600,
        "total": 600,
        "taxes": [],
        "line_items": [
            {"description": "商品A", "qty": 1, "unit_price": 100, "total": 120},
            {"description": "商品B", "qty": 1, "unit_price": 200, "total": 200},
            {"description": "値引商品", "qty": 1, "unit_price": 330, "total": 280,
             "discount": 50, "discount_rate": "15%"},
        ],
    }
    ocr_text = "\n".join([
        "商品A",
        "120*",
        "商品B",
        "200*",
        "値引商品",
        "330*",
        "割引",
        "-50",
        "小計",
    ])

    _project_totals_to_ocr_multiset(extracted, ocr_text)

    assert extracted["line_items"][2]["unit_price"] == 330
    assert extracted["line_items"][2]["total"] == 280
    assert extracted["line_items"][2]["discount"] == 50


def test_ocr_discount_detection_uses_full_description_not_shared_prefix():
    from receipt_parser.pipeline_receipt import _detect_ocr_discounts

    items = [
        {"description": "FFB いちごジャムパン", "qty": 1, "unit_price": 158,
         "total": 158, "discount": 0, "discount_rate": ""},
        {"description": "FFB メロンパン", "qty": 1, "unit_price": 148,
         "total": 148, "discount": 0, "discount_rate": ""},
    ]
    ocr_text = "\n".join([
        "FFB いちごジャムパン",
        "158*",
        "FFB メロンパン",
        "148*",
        "割引",
        "20%",
        "-30",
        "小計",
    ])

    _detect_ocr_discounts(items, ocr_text)

    assert items[0]["discount"] == 0
    assert items[0]["total"] == 158
    assert items[1]["discount"] == 30
    assert items[1]["discount_rate"] == "20%"
    assert items[1]["total"] == 118


def test_item_total_repair_keeps_duplicate_description_supported_price():
    from receipt_parser.pipeline_receipt import _fix_item_totals_from_ocr_neighborhood

    items = [
        {"description": "じゃがいも", "qty": 1, "unit_price": 259, "total": 259},
        {"description": "じゃがいも", "qty": 1, "unit_price": 231, "total": 231},
        {"description": "ごまスティック", "qty": 1, "unit_price": 228, "total": 228},
    ]
    ocr_text = "\n".join([
        "じゃがいも",
        "259*",
        "じゃがいも",
        "231 ※",
        "ごまスティック",
        "228*",
        "小計",
    ])

    _fix_item_totals_from_ocr_neighborhood(items, ocr_text, 700, 700)

    assert [item["total"] for item in items] == [259, 231, 228]


def test_time_fix_zero_pads_single_digit_hour():
    from receipt_parser.pipeline_receipt import _fix_time

    extracted = {"date": "2026-05-08", "time": "9:53"}
    _fix_time(extracted, "2026/5/8(金)\n9:53 レジ 0143")

    assert extracted["time"] == "09:53"


def test_payment_method_normalizes_quicpay_when_credit_is_printed():
    from receipt_parser.pipeline_receipt import _fix_payment_method

    extracted = {"payment_method": "QUICPay"}
    _fix_payment_method(extracted, "クレジット(内\n¥3,630)", 0.9, {})

    assert extracted["payment_method"] == "credit"


def test_location_resolution_uses_area_code_when_no_clean_branch():
    from receipt_parser.pipeline import _resolve_location

    extracted = {"merchant": "コスモス", "location": None}
    ocr_text = "\n".join([
        "ドラッグストア",
        "コスモス",
        "元店 TEL0940-72-5355",
        "領収証",
    ])

    location, warning = _resolve_location(extracted, ocr_text, "unused")

    assert location == "宗像市"
    assert warning is None


def test_location_resolution_uses_explicit_city_marker_over_area_default():
    from receipt_parser.pipeline import _resolve_location

    extracted = {"merchant": "Mister Donut", "location": None}
    ocr_text = "\n".join([
        "mister",
        "Donut",
        "イオンモール福津ショップ",
        "電話0940-43-8016",
    ])

    location, warning = _resolve_location(extracted, ocr_text, "unused")

    assert location == "福津市"
    assert warning is None


def test_item_desc_repair_keeps_fuzzy_ocr_supported_description():
    from receipt_parser.pipeline_receipt import _fix_item_desc_from_ocr_price_line

    items = [
        {"description": "あまおう苺のデニッシュ", "qty": 1, "unit_price": 450, "total": 450},
        {"description": "ヴルストクロワッサン", "qty": 1, "unit_price": 420, "total": 420},
    ]
    ocr_text = "\n".join([
        "105-000002-000 あまおう苺のデニッシ",
        "105-000008-000 ヴルストクロワッサン 450*",
        "420*",
    ])

    _fix_item_desc_from_ocr_price_line(items, ocr_text)

    assert items[0]["description"] == "あまおう苺のデニッシュ"
    assert items[1]["description"] == "ヴルストクロワッサン"


def test_layout_row_projection_repairs_price_swap():
    from receipt_parser.pipeline_receipt import _project_totals_to_layout_rows

    extracted = {
        "subtotal": 554,
        "total": 554,
        "taxes": [],
        "line_items": [
            {"description": "ベビーダノンもも", "qty": 1, "unit_price": 228, "total": 228},
            {"description": "ヨーグルト", "qty": 1, "unit_price": 98, "total": 98},
            {"description": "プチダノンリンゴ", "qty": 1, "unit_price": 98, "total": 98},
        ],
    }
    layout = [
        {"text": "ベビーダノンもも", "x": 10, "y": 10, "bbox": [[10, 10], [100, 10], [100, 25], [10, 25]]},
        {"text": "228*", "x": 180, "y": 10, "bbox": [[180, 10], [220, 10], [220, 25], [180, 25]]},
        {"text": "ヨーグルト", "x": 10, "y": 35, "bbox": [[10, 35], [100, 35], [100, 50], [10, 50]]},
        {"text": "98*", "x": 180, "y": 35, "bbox": [[180, 35], [220, 35], [220, 50], [180, 50]]},
        {"text": "プチダノンリンゴ", "x": 10, "y": 60, "bbox": [[10, 60], [100, 60], [100, 75], [10, 75]]},
        {"text": "228*", "x": 180, "y": 60, "bbox": [[180, 60], [220, 60], [220, 75], [180, 75]]},
        {"text": "小計", "x": 10, "y": 85, "bbox": [[10, 85], [100, 85], [100, 100], [10, 100]]},
        {"text": "554", "x": 180, "y": 85, "bbox": [[180, 85], [220, 85], [220, 100], [180, 100]]},
    ]

    _project_totals_to_layout_rows(extracted, layout)

    assert [item["total"] for item in extracted["line_items"]] == [228, 98, 228]
    assert [item["unit_price"] for item in extracted["line_items"]] == [228, 98, 228]


def test_layout_row_projection_skips_when_description_alignment_fails():
    from receipt_parser.pipeline_receipt import _project_totals_to_layout_rows

    extracted = {
        "subtotal": 300,
        "total": 300,
        "taxes": [],
        "line_items": [
            {"description": "商品A", "qty": 1, "unit_price": 100, "total": 100},
            {"description": "商品B", "qty": 1, "unit_price": 100, "total": 100},
        ],
    }
    layout = [
        {"text": "別商品X", "x": 10, "y": 10, "bbox": [[10, 10], [100, 10], [100, 25], [10, 25]]},
        {"text": "100", "x": 180, "y": 10, "bbox": [[180, 10], [220, 10], [220, 25], [180, 25]]},
        {"text": "別商品Y", "x": 10, "y": 35, "bbox": [[10, 35], [100, 35], [100, 50], [10, 50]]},
        {"text": "200", "x": 180, "y": 35, "bbox": [[180, 35], [220, 35], [220, 50], [180, 50]]},
        {"text": "小計", "x": 10, "y": 60, "bbox": [[10, 60], [60, 60], [60, 75], [10, 75]]},
        {"text": "300", "x": 180, "y": 60, "bbox": [[180, 60], [220, 60], [220, 75], [180, 75]]},
    ]

    _project_totals_to_layout_rows(extracted, layout)

    assert [item["total"] for item in extracted["line_items"]] == [100, 100]
    assert [item["unit_price"] for item in extracted["line_items"]] == [100, 100]


def test_layout_row_projection_uses_price_column_and_qty_detail():
    from receipt_parser.pipeline_receipt import _project_totals_to_layout_rows

    extracted = {
        "subtotal": 121,
        "total": 121,
        "taxes": [],
        "line_items": [
            {"description": "食品ポリ袋Lバイオマス30", "qty": 1, "unit_price": 30, "total": 30},
            {"description": "キャベツ1/2カット", "qty": 2, "unit_price": 10, "total": 20},
            {"description": "プチダノンリンコ", "qty": 1, "unit_price": 98, "total": 98},
        ],
    }
    layout = [
        {"text": "レジ", "x": 10, "y": 5, "bbox": [[10, 5], [40, 5], [40, 20], [10, 20]]},
        {"text": "0142", "x": 80, "y": 5, "bbox": [[80, 5], [120, 5], [120, 20], [80, 20]]},
        {"text": "食品ポリ袋L", "x": 10, "y": 35, "bbox": [[10, 35], [100, 35], [100, 50], [10, 50]]},
        {"text": "バイオマス", "x": 110, "y": 35, "bbox": [[110, 35], [170, 35], [170, 50], [110, 50]]},
        {"text": "30", "x": 190, "y": 35, "bbox": [[190, 35], [220, 35], [220, 50], [190, 50]]},
        {"text": "3", "x": 260, "y": 35, "bbox": [[260, 35], [275, 35], [275, 50], [260, 50]]},
        {"text": "キャベツ", "x": 10, "y": 60, "bbox": [[10, 60], [80, 60], [80, 75], [10, 75]]},
        {"text": "10", "x": 260, "y": 60, "bbox": [[260, 60], [280, 60], [280, 75], [260, 75]]},
        {"text": "2", "x": 40, "y": 85, "bbox": [[40, 85], [50, 85], [50, 100], [40, 100]]},
        {"text": "個", "x": 55, "y": 85, "bbox": [[55, 85], [70, 85], [70, 100], [55, 100]]},
        {"text": "X10", "x": 80, "y": 85, "bbox": [[80, 85], [115, 85], [115, 100], [80, 100]]},
        {"text": "プチダノンリンコ", "x": 10, "y": 110, "bbox": [[10, 110], [120, 110], [120, 125], [10, 125]]},
        {"text": "98", "x": 260, "y": 110, "bbox": [[260, 110], [280, 110], [280, 125], [260, 125]]},
        {"text": "小計", "x": 10, "y": 135, "bbox": [[10, 135], [60, 135], [60, 150], [10, 150]]},
        {"text": "121", "x": 260, "y": 135, "bbox": [[260, 135], [300, 135], [300, 150], [260, 150]]},
    ]

    _project_totals_to_layout_rows(extracted, layout)

    assert [item["total"] for item in extracted["line_items"]] == [3, 20, 98]
    assert [item["unit_price"] for item in extracted["line_items"]] == [3, 10, 98]


def test_layout_row_projection_preserves_discounted_net_total():
    from receipt_parser.pipeline_receipt import _project_totals_to_layout_rows

    extracted = {
        "subtotal": 600,
        "total": 600,
        "taxes": [],
        "line_items": [
            {"description": "商品A", "qty": 1, "unit_price": 120, "total": 120},
            {"description": "商品B", "qty": 1, "unit_price": 200, "total": 220},
            {"description": "値引商品", "qty": 1, "unit_price": 330, "total": 280,
             "discount": 50, "discount_rate": "15%"},
        ],
    }
    layout = [
        {"text": "商品A", "x": 10, "y": 10, "bbox": [[10, 10], [80, 10], [80, 25], [10, 25]]},
        {"text": "120", "x": 220, "y": 10, "bbox": [[220, 10], [260, 10], [260, 25], [220, 25]]},
        {"text": "商品B", "x": 10, "y": 35, "bbox": [[10, 35], [80, 35], [80, 50], [10, 50]]},
        {"text": "200", "x": 220, "y": 35, "bbox": [[220, 35], [260, 35], [260, 50], [220, 50]]},
        {"text": "値引商品", "x": 10, "y": 60, "bbox": [[10, 60], [90, 60], [90, 75], [10, 75]]},
        {"text": "330", "x": 220, "y": 60, "bbox": [[220, 60], [260, 60], [260, 75], [220, 75]]},
        {"text": "小計", "x": 10, "y": 85, "bbox": [[10, 85], [60, 85], [60, 100], [10, 100]]},
    ]

    _project_totals_to_layout_rows(extracted, layout)

    assert extracted["line_items"][1]["total"] == 200
    assert extracted["line_items"][2]["unit_price"] == 330
    assert extracted["line_items"][2]["total"] == 280


def test_receipt_postprocess_selector_prefers_clean_history_candidate():
    from receipt_parser.pipeline import _select_receipt_postprocessed_candidate

    bad_selected = {
        "document_type": "receipt",
        "merchant": "テスト店",
        "date": "2026-05-02",
        "currency": "JPY",
        "total": 100,
        "subtotal": 100,
        "taxes": [],
        "line_items": [
            {"description": "商品A", "qty": 1, "unit_price": 40, "total": 40},
            {"description": "商品B", "qty": 1, "unit_price": 50, "total": 50},
        ],
    }
    clean_history = {
        "document_type": "receipt",
        "merchant": "テスト店",
        "date": "2026-05-02",
        "currency": "JPY",
        "total": 100,
        "subtotal": 100,
        "taxes": [],
        "line_items": [
            {"description": "商品A", "qty": 1, "unit_price": 40, "total": 40},
            {"description": "商品B", "qty": 1, "unit_price": 60, "total": 60},
        ],
    }
    history = [
        {"pass": 1, "extraction": clean_history, "warnings": []},
        {"pass": "sanity-retry-seed45", "extraction": bad_selected, "warnings": []},
    ]

    selected = _select_receipt_postprocessed_candidate(
        bad_selected,
        history,
        "テスト店\n小計\n¥100\n合計\n¥100",
        0.9,
        {},
        "test-model",
        None,
    )

    assert [item["total"] for item in selected["line_items"]] == [40, 60]
    assert history[0]["postprocess_selected"] is True
    assert history[1]["postprocess_selected"] is False
    assert history[0]["postprocess_items_sum_gap"] == 0


def test_benchmark_artifacts_are_run_specific(monkeypatch):
    import importlib.util

    benchmark_path = Path(__file__).with_name("benchmark.py")
    spec = importlib.util.spec_from_file_location("benchmark_under_test", benchmark_path)
    benchmark = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(benchmark)

    scratch = Path("local/test_benchmark_artifacts").resolve()
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    monkeypatch.setattr(benchmark, "RESULTS_DIR", scratch)
    run_id = "unit-run"
    metadata = {
        "timestamp": "test",
        "run_id": run_id,
        "git_sha": "test",
        "model": "test-model",
        "runs_per_fixture": 1,
        "passes": 1,
        "workers": 1,
        "ci_mode": True,
        "fixtures": ["receipt_x"],
        "output_path": str(scratch / "latest.json"),
        "artifact_dir": str(scratch / "artifacts" / run_id),
    }
    per_fixture = {
        "receipt_x": {
            "runs": [
                {
                    "run": 1,
                    "passed": True,
                    "pass_count": 1,
                    "total_fields": 1,
                    "wall_time_s": 0,
                    "error": None,
                    "fields": {"total": {"pass": True}},
                    "ocr": {},
                    "ocr_text": "OCR TEXT",
                    "llm_raw": {"total": 1},
                    "llm_pass_history": [
                        {"pass": 1, "extraction": {"total": 1}, "warnings": []}
                    ],
                    "final_extraction": {"total": 1},
                    "warnings": [],
                    "warning_count": 0,
                    "llm_passes_used": 1,
                }
            ]
        }
    }

    try:
        results = benchmark._assemble_results(metadata, per_fixture)
        run = results["per_fixture"]["receipt_x"]["runs"][0]
        paths = [
            run["ocr"]["text_file"],
            run["llm_raw_file"],
            run["llm_pass_history_file"],
            run["final_file"],
        ]

        assert all(path.startswith(f"artifacts/{run_id}/") for path in paths)
        assert all((scratch / path).exists() for path in paths)
        assert "ocr_text" not in run
        assert "llm_raw" not in run
        assert "llm_pass_history" not in run
        assert "final_extraction" not in run
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


def test_debug_color_map_has_all_fields():
    color_map = get_debug_color_map()
    expected_fields = ["merchant", "date", "location", "currency", "line_items",
                       "subtotal", "taxes", "total", "payment_method"]
    for field in expected_fields:
        assert field in color_map, f"Missing field: {field}"



# --- Pipeline tests with mocked Cloud Vision + LLM ---

@patch("receipt_parser.pipeline.extract_with_verification")
@patch("receipt_parser.pipeline.run_cloud_vision")
@patch("receipt_parser.pipeline.init_cloud_vision")
@patch("receipt_parser.pipeline.check_model_available")
def test_pipeline_with_mocked_inference(mock_check, mock_init_cv, mock_ocr, mock_extract):
    mock_check.return_value = None
    mock_init_cv.return_value = MagicMock()
    mock_blocks = [
        {"text": "セブンイレブン", "confidence": 0.95, "x": 100, "y": 10,
         "bbox": [[0, 0], [200, 0], [200, 20], [0, 20]]},
        {"text": "合計 ¥150", "confidence": 0.92, "x": 100, "y": 50,
         "bbox": [[0, 40], [200, 40], [200, 60], [0, 60]]},
    ]
    mock_ocr.return_value = OCRResult(
        blocks=mock_blocks, confidence=0.93, source="fresh",
        chosen_text="セブンイレブン\n合計 ¥150",
    )
    mock_extract.return_value = (
        {"merchant": "セブンイレブン", "total": 150, "line_items": [],
         "taxes": [], "subtotal": None, "date": None, "currency": "JPY",
         "payment_method": None, "location": None,
         "raw_text_summary": "convenience store receipt"},
        [{"pass": 1, "extraction": {}, "warnings": []}]
    )

    img = np.zeros((100, 200, 3), dtype=np.uint8)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        cv2.imwrite(tmp.name, img)
        tmp.close()
        from receipt_parser.pipeline import process_document
        result = process_document(Path(tmp.name), passes=1)
    finally:
        os.unlink(tmp.name)

    assert result["merchant"] == "セブンイレブン"
    assert result.get("_pass_count", 0) >= 1
    assert "_model" in result
    assert "_line_items_reliable" in result
    assert "_pipeline_version" in result


@patch("receipt_parser.pipeline.run_cloud_vision")
@patch("receipt_parser.pipeline.init_cloud_vision")
@patch("receipt_parser.pipeline.check_model_available")
def test_pipeline_blank_image_returns_error(mock_check, mock_init_cv, mock_ocr):
    mock_check.return_value = None
    mock_init_cv.return_value = MagicMock()
    mock_ocr.return_value = OCRResult(source="fresh")

    img = np.zeros((100, 200, 3), dtype=np.uint8)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        cv2.imwrite(tmp.name, img)
        tmp.close()
        from receipt_parser.pipeline import process_document
        result = process_document(Path(tmp.name), passes=1)
    finally:
        os.unlink(tmp.name)

    assert "_error" in result


# --- API usage tracking tests ---

def _isolate_usage_files(monkeypatch, name: str) -> Path:
    import receipt_parser.usage as usage_module

    scratch = Path("local") / name
    if scratch.exists():
        shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(usage_module, "_USAGE_FILE", scratch / "api_usage.json")
    monkeypatch.setattr(usage_module, "_HISTORY_FILE", scratch / "api_usage_history.json")
    monkeypatch.setattr(usage_module, "_SETTINGS_FILE", scratch / "api_usage_settings.json")
    return scratch


def test_api_usage_tracking(monkeypatch):
    scratch = _isolate_usage_files(monkeypatch, "test_usage_tracking")
    from receipt_parser.usage import (
        _save, _empty_month, get_usage, sync_usage, track_openrouter_call,
        _billing_period_month_key,
    )
    try:
        from receipt_parser.ocr import get_api_usage
        month_key = _billing_period_month_key()
        _save(_empty_month(month_key))
        # Unified tracker
        stats = get_usage()
        assert stats["cloud_vision"]["calls"] == 0
        assert stats["cloud_vision"]["remaining_free"] == 1000
        assert stats["deepseek"]["calls"] == 0
        assert stats["deepseek"]["cache_hit_tokens"] == 0
        assert stats["deepseek"]["cache_miss_tokens"] == 0
        assert stats["openrouter"]["calls"] == 0
        assert stats["openrouter"]["cache_hit_tokens"] == 0
        assert stats["openrouter"]["cache_miss_tokens"] == 0
        assert stats["openrouter"]["output_tokens"] == 0
        assert stats["documents"]["total_processed"] == 0
        assert stats["documents"]["unique_processed"] == 0
        # Legacy interface
        legacy = get_api_usage()
        assert legacy["calls"] == 0
        assert legacy["remaining"] == 1000
        # Sync with cache hit/miss breakdown
        sync_usage(cv_calls=42, ds_cache_hit=18_507_008, ds_cache_miss=1_019_604,
                   ds_output=2_662_239, ds_calls=6959)
        stats = get_usage()
        assert stats["cloud_vision"]["calls"] == 42
        assert stats["deepseek"]["calls"] == 6959
        assert stats["deepseek"]["cache_hit_tokens"] == 18_507_008
        assert stats["deepseek"]["cache_miss_tokens"] == 1_019_604
        assert stats["deepseek"]["output_tokens"] == 2_662_239
        assert stats["deepseek"]["est_cost_usd"] > 0
        # OpenRouter triage model accounting: aggregate plus per-model stats.
        track_openrouter_call(
            "anthropic/claude-sonnet-4.6",
            cache_hit_tokens=100,
            cache_miss_tokens=900,
            output_tokens=250,
            cost_usd=0.0123,
            reasoning_tokens=7,
            cache_write_tokens=50,
        )
        track_openrouter_call(
            "anthropic/claude-opus-4.7",
            cache_hit_tokens=25,
            cache_miss_tokens=475,
            output_tokens=100,
            cost_usd=0.0456,
        )
        stats = get_usage()
        or_stats = stats["openrouter"]
        assert or_stats["calls"] == 2
        assert or_stats["cache_hit_tokens"] == 125
        assert or_stats["cache_miss_tokens"] == 1375
        assert or_stats["output_tokens"] == 350
        assert or_stats["reasoning_tokens"] == 7
        assert or_stats["cache_write_tokens"] == 50
        assert or_stats["est_cost_usd"] == 0.0579
        assert or_stats["models"]["anthropic/claude-sonnet-4.6"]["calls"] == 1
        assert or_stats["models"]["anthropic/claude-sonnet-4.6"]["est_cost_usd"] == 0.0123
        assert or_stats["models"]["anthropic/claude-opus-4.7"]["calls"] == 1
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_openrouter_llm_call_tracks_openrouter_usage(monkeypatch):
    scratch = _isolate_usage_files(monkeypatch, "test_openrouter_usage_tracking")
    from receipt_parser.usage import (
        _save, _empty_month, get_usage, _billing_period_month_key,
    )
    try:
        month_key = _billing_period_month_key()
        _save(_empty_month(month_key))

        usage = SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=250,
            prompt_tokens_details=SimpleNamespace(cached_tokens=100, cache_write_tokens=40),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=12),
            model_extra={"cost": 0.1234},
        )
        response = SimpleNamespace(
            usage=usage,
            model="anthropic/claude-sonnet-4.6",
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"total": 1}'))],
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: response
                )
            )
        )

        monkeypatch.setattr(llm_module, "_get_api_client", lambda model=None: fake_client)
        result = llm_module._openrouter_chat(
            "openrouter/anthropic/claude-sonnet-4.6",
            messages=[{"role": "user", "content": "x"}],
        )

        assert result.input_tokens == 1000
        assert result.output_tokens == 250
        stats = get_usage()
        or_stats = stats["openrouter"]
        assert or_stats["calls"] == 1
        assert or_stats["cache_hit_tokens"] == 100
        assert or_stats["cache_miss_tokens"] == 900
        assert or_stats["output_tokens"] == 250
        assert or_stats["reasoning_tokens"] == 12
        assert or_stats["cache_write_tokens"] == 40
        assert or_stats["est_cost_usd"] == 0.1234
        assert or_stats["models"]["anthropic/claude-sonnet-4.6"]["calls"] == 1
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# --- Stage callback tests ---

def test_stage_callback_fires_in_order():
    """on_stage callback should fire with monotonically increasing progress."""
    from receipt_parser.pipeline import process_ocr_text

    calls = []

    def on_stage(stage, detail, progress):
        calls.append((stage, detail, progress))

    # Use a minimal OCR text that will produce a result
    ocr_text = "マクドナルド\n2025-01-15\n合計 ¥500\nビッグマック ¥500"

    with patch("receipt_parser.pipeline.extract_with_verification") as mock_extract:
        mock_extract.return_value = (
            {"merchant": "マクドナルド", "date": "2025-01-15", "total": 500,
             "line_items": [{"description": "ビッグマック", "total": 500}],
             "currency": "JPY", "_confidence": {"overall": "high"}},
            [{"pass": 1, "extraction": {}, "warnings": []}],
        )
        process_ocr_text(ocr_text, on_stage=on_stage)

    # Should have received calls
    assert len(calls) >= 3, f"Expected at least 3 stage calls, got {len(calls)}"

    # All values should be correct types
    for stage, detail, progress in calls:
        assert isinstance(stage, str)
        assert isinstance(detail, str)
        assert isinstance(progress, float) or isinstance(progress, int)

    # Progress should be monotonically non-decreasing
    progress_values = [c[2] for c in calls]
    for i in range(1, len(progress_values)):
        assert progress_values[i] >= progress_values[i - 1], \
            f"Progress not monotonic: {progress_values}"

    # Should end with "done" at 1.0
    assert calls[-1][0] == "done"
    assert calls[-1][2] == 1.0


def test_stage_callback_none_default():
    """on_stage=None (default) should not break anything."""
    from receipt_parser.pipeline import process_ocr_text

    ocr_text = "マクドナルド\n2025-01-15\n合計 ¥500"

    with patch("receipt_parser.pipeline.extract_with_verification") as mock_extract:
        mock_extract.return_value = (
            {"merchant": "マクドナルド", "date": "2025-01-15", "total": 500,
             "currency": "JPY", "_confidence": {"overall": "high"}},
            [{"pass": 1, "extraction": {}, "warnings": []}],
        )
        # Should not raise — on_stage defaults to None
        result = process_ocr_text(ocr_text)
        assert "merchant" in result


def test_stage_callback_emits_classify_and_plan():
    """Phase 2 contract: a 4-arg callback (param named `payload`) receives the
    structured plan and classify payloads."""
    from receipt_parser.pipeline import process_ocr_text

    events: list[tuple[str, str, float, dict | None]] = []

    def on_stage(stage, detail, progress, payload=None):
        events.append((stage, detail, progress, payload))

    with patch("receipt_parser.pipeline.extract_with_verification") as mock_extract:
        mock_extract.return_value = (
            {"merchant": "マクドナルド", "date": "2025-01-15", "total": 500,
             "currency": "JPY", "line_items": [
                 {"description": "ビッグマック", "total": 500}]},
            [{"pass": 1, "extraction": {}, "warnings": []}],
        )
        process_ocr_text(
            "マクドナルド\n2025-01-15\n合計 ¥500\nビッグマック ¥500",
            on_stage=on_stage,
        )

    stages = [e[0] for e in events]
    assert "plan" in stages
    assert "classify" in stages
    assert "validate" in stages
    assert stages[-1] == "done"

    plan = next(e for e in events if e[0] == "plan")
    assert plan[3] is not None and plan[3]["page_count"] == 1
    assert plan[3]["pass_budget"] >= 1
    assert plan[3]["path"] in ("receipt", "tbd", "utility_bill", "payment_slip")

    classify = next(e for e in events if e[0] == "classify")
    assert classify[3] is not None
    assert classify[3].get("document_type") in ("receipt", "utility_bill", "payment_slip")
    # Classify payload must let the consumer finalize the step list without
    # memorizing the path table.
    assert "expected_stages" in classify[3]
    assert classify[3]["expected_stages"][-1] == "done"
    assert "will_resolve_location" in classify[3]
    assert isinstance(classify[3]["will_resolve_location"], bool)

    # Validate-stage detail should include the result preview ("N items · ¥M").
    validate_event = next(e for e in events if e[0] == "validate")
    assert "item" in validate_event[1]
    assert "500" in validate_event[1]


def test_stage_callback_legacy_three_arg_still_works():
    """3-arg callbacks must keep working; payload must NOT be passed positionally."""
    from receipt_parser.pipeline import process_ocr_text

    events: list[tuple] = []

    def on_stage(stage, detail, progress):
        events.append((stage, detail, progress))

    with patch("receipt_parser.pipeline.extract_with_verification") as mock_extract:
        mock_extract.return_value = (
            {"merchant": "マクドナルド", "total": 500, "currency": "JPY",
             "line_items": []},
            [{"pass": 1, "extraction": {}, "warnings": []}],
        )
        process_ocr_text("マクドナルド\n合計 ¥500", on_stage=on_stage)

    assert events[-1][0] == "done"
    # Legacy callback must never receive a 4th element.
    assert all(len(e) == 3 for e in events)


def test_stage_callback_default_arg_closure_pattern_preserved():
    """The CLI-style closure-capture-via-default trick (`def cb(s,d,p,_task=t)`)
    must not be misclassified as a 4-arg payload-receiving callback. The
    default-bound 4th arg is closure state, not an output channel."""
    from receipt_parser.pipeline import _callback_accepts_payload

    sentinel = object()

    def cb(stage, detail, pct, _task=sentinel):
        return _task  # would expose accidental override

    # Conservative: only the explicit `payload` name (or **kwargs) opts in.
    assert _callback_accepts_payload(cb) is False

    def cb_payload(stage, detail, progress, payload=None):
        return payload

    assert _callback_accepts_payload(cb_payload) is True

    def cb_kwargs(*args, **kwargs):
        return kwargs

    assert _callback_accepts_payload(cb_kwargs) is True


def test_stage_callback_cancellation_raises():
    """A callback returning False must abort the pipeline with PipelineCancelled."""
    from receipt_parser.pipeline import process_ocr_text, PipelineCancelled

    def on_stage(stage, detail, progress):
        if stage == "extract":
            return False  # request cancellation
        return None

    with patch("receipt_parser.pipeline.extract_with_verification") as mock_extract:
        mock_extract.return_value = (
            {"merchant": "X", "total": 1, "currency": "JPY", "line_items": []},
            [{"pass": 1, "extraction": {}, "warnings": []}],
        )
        try:
            process_ocr_text("X\n合計 ¥1", on_stage=on_stage)
        except PipelineCancelled as e:
            assert e.stage == "extract"
        else:
            raise AssertionError("Expected PipelineCancelled to be raised")


def test_stage_callback_warn_fires_on_low_block_retry():
    """The warn stage should be emitted when OCR returns <3 blocks and rotation
    retry kicks in. Verifies the engineer's example: 'OCR returned <3 blocks,
    retrying'."""
    from receipt_parser.pipeline import process_document
    from receipt_parser.ocr import OCRResult

    events: list[tuple] = []

    def on_stage(stage, detail, progress, payload=None):
        events.append((stage, detail, progress, payload))

    # Two blocks → triggers retry. Subsequent rotations also return weak results
    # (so the loop runs but doesn't early-stop).
    weak_blocks = [
        {"text": "a", "confidence": 0.5, "x": 0, "y": 0,
         "bbox": [[0, 0], [10, 0], [10, 10], [0, 10]]},
        {"text": "b", "confidence": 0.5, "x": 0, "y": 12,
         "bbox": [[0, 12], [10, 12], [10, 22], [0, 22]]},
    ]

    def fake_ocr(*_args, **_kwargs):
        return OCRResult(blocks=list(weak_blocks),
                         source="cloud_vision", retried=False)

    scratch = Path("local/test_temp_stage_callback").resolve()
    if scratch.exists():
        shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        img_path = scratch / "tiny.png"
        img = np.full((50, 50, 3), 255, dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", img)
        assert ok
        img_path.write_bytes(encoded.tobytes())

        with patch("receipt_parser.pipeline.run_cloud_vision",
                   side_effect=fake_ocr), \
             patch("receipt_parser.pipeline.init_cloud_vision",
                   return_value=MagicMock()), \
             patch("receipt_parser.pipeline.extract_with_verification") as mock_extract:
            mock_extract.return_value = (
                {"merchant": "X", "total": 1, "currency": "JPY",
                 "line_items": []},
                [{"pass": 1, "extraction": {}, "warnings": []}],
            )
            process_document(img_path, on_stage=on_stage)
    finally:
        if scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)

    warn_events = [e for e in events if e[0] == "warn"]
    assert warn_events, f"Expected at least one 'warn' event; got {[e[0] for e in events]}"
    # Payload should describe the reason and page index.
    payload = warn_events[0][3]
    assert payload is not None
    assert payload.get("reason") == "low_block_count"
    assert payload.get("page") == 1
