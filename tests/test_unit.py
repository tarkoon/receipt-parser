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


def test_date_prefers_labeled_two_digit_transaction_date_over_expiry():
    from receipt_parser.pipeline_receipt import _fix_date

    extracted = {"date": "2026-08-22"}
    text = "\n".join([
        "日付26年05月19日 14:54",
        "会員有効期限 2026年08月22日",
    ])

    _fix_date(extracted, text)

    assert extracted["date"] == "2026-05-19"


def test_date_skips_implausible_receipt_year_for_card_usage_date():
    from receipt_parser.pipeline_receipt import _fix_date

    extracted = {"date": "2024-05-18"}
    text = "\n".join([
        "2006年05月18日(月) 06:45",
        "QUICPay ご利用日 2026年05月18日",
    ])

    _fix_date(extracted, text)

    assert extracted["date"] == "2026-05-18"


def test_date_prefers_labeled_next_line_ocr_mangled_modern_year():
    from receipt_parser.pipeline_receipt import _fix_date

    extracted = {"date": "2025-05-23"}
    text = "\n".join([
        "2005/ 5/23(土)",
        "QUICPay",
        "お取扱日",
        "2006年05月23日",
    ])

    _fix_date(extracted, text)

    assert extracted["date"] == "2026-05-23"


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


def test_ocr_multiset_projection_ignores_discounted_items_in_fallback_targets():
    from receipt_parser.pipeline_receipt import _project_totals_to_ocr_multiset

    extracted = {
        "subtotal": 353,
        "line_items": [
            {
                "description": "グレープ100%ジュー",
                "qty": 1,
                "unit_price": 208,
                "total": 145,
                "discount": 63,
            },
            {
                "description": "めんつゆ",
                "qty": 1,
                "unit_price": 999,
                "total": 999,
                "discount": 0,
            },
        ],
    }
    text = "\n".join([
        "グレープ100%ジュー 208*",
        "割引",
        "-63",
        "OCR別名 208*",
        "小計",
        "¥353",
    ])

    _project_totals_to_ocr_multiset(extracted, text)

    assert extracted["line_items"][0]["total"] == 145
    assert extracted["line_items"][1]["total"] == 208


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


def test_rate_discount_repair_uses_unused_ocr_amount_for_duplicate_current_discount():
    from receipt_parser.pipeline_receipt import _repair_rate_discounts_from_ocr_amounts

    items = [
        {
            "description": "牛肉かたロースステーキ",
            "qty": 1,
            "unit_price": 684,
            "total": 410,
            "discount": 274,
            "discount_rate": "40%",
        },
        {
            "description": "牛肉かたロースステーキ",
            "qty": 1,
            "unit_price": 706,
            "total": 432,
            "discount": 274,
            "discount_rate": "40%",
        },
    ]
    text = "\n".join([
        "牛肉かたロースステーキ 684X",
        "割引",
        "40%",
        "-274",
        "牛肉かたロースステーキ 706*",
        "割引",
        "40%",
        "-283",
    ])

    _repair_rate_discounts_from_ocr_amounts(items, text)

    assert items[0]["discount"] == 274
    assert items[0]["total"] == 410
    assert items[1]["discount"] == 283
    assert items[1]["total"] == 423


def test_clears_discount_when_next_item_starts_before_discount_marker():
    from receipt_parser.pipeline_receipt import _clear_discounts_without_nearby_ocr_marker

    items = [
        {
            "description": "豚肉",
            "qty": 1,
            "unit_price": 980,
            "total": 844,
            "discount": 136,
            "discount_rate": "20%",
        },
        {
            "description": "豚肉",
            "qty": 1,
            "unit_price": 680,
            "total": 544,
            "discount": 136,
            "discount_rate": "20%",
        },
    ]
    text = "\n".join([
        "000062 豚肉",
        "¥980",
        "000062 豚肉",
        "¥680",
        "操作割引2",
        "20%",
        "-136",
    ])

    _clear_discounts_without_nearby_ocr_marker(items, text)

    assert items[0]["discount"] == 0
    assert items[0]["total"] == 980
    assert items[1]["discount"] == 136
    assert items[1]["total"] == 544


def test_missing_item_recovery_accepts_percent_marker_price_when_gap_matches():
    from receipt_parser.pipeline_receipt import _recover_missing_items_from_gap

    extracted = {
        "total": 120,
        "subtotal": 120,
        "taxes": [],
        "line_items": [
            {"description": "こまつな", "qty": 1, "unit_price": 100, "total": 100}
        ],
    }
    text = "\n".join([
        "こまつな",
        "100*",
        "はくさい 1/4カット",
        "20%",
        "小計",
        "¥120",
    ])

    _recover_missing_items_from_gap(extracted, text)

    assert extracted["line_items"][-1]["description"] == "はくさい 1/4カット"
    assert extracted["line_items"][-1]["total"] == 20


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


def test_payment_method_normalizes_electronic_money_alias():
    from receipt_parser.pipeline_receipt import _fix_payment_method

    extracted = {"payment_method": "electronic_money"}
    _fix_payment_method(extracted, "電子マネー\n¥1,375", 0.9, {})

    assert extracted["payment_method"] == "credit"


def test_payment_method_detects_electronic_money_from_ocr():
    from receipt_parser.pipeline_receipt import _fix_payment_method

    extracted = {"payment_method": None}
    _fix_payment_method(extracted, "合計\n¥1,375\n電子マネー\n¥1,375", 0.9, {})

    assert extracted["payment_method"] == "credit"


def test_payment_method_credit_overrides_hallucinated_cash_without_cash_tender():
    from receipt_parser.pipeline_receipt import _fix_payment_method

    extracted = {"payment_method": "cash"}
    _fix_payment_method(extracted, "お預り票\nクレジットカード\nお釣り\n0", 0.9, {})

    assert extracted["payment_method"] == "credit"


def test_stacked_cash_tender_block_sets_total_not_tendered_cash():
    from receipt_parser.pipeline_receipt import _fix_total_from_stacked_cash_tender_block

    extracted = {"total": 1070, "amount_paid": 1070, "points_used": 0}
    text = "\n".join([
        "総合計",
        "現金",
        "お釣り",
        "「軽」印は軽減税率(8%)適用商品",
        "570",
        "1,070",
        "500",
        "To Go",
    ])

    _fix_total_from_stacked_cash_tender_block(extracted, text)

    assert extracted["total"] == 570
    assert extracted["amount_paid"] == 570


def test_cash_tender_block_infers_total_when_shifted_out_of_value_stack():
    from receipt_parser.pipeline_receipt import _fix_total_from_stacked_cash_tender_block

    extracted = {"total": 1070, "amount_paid": 1070, "points_used": 0}
    text = "\n".join([
        "ライト アイス 570",
        "総合計",
        "現金",
        "お釣り",
        "「軽」印は軽減税率(8%)適用商品",
        "1,070",
        "500",
        "To Go",
    ])

    _fix_total_from_stacked_cash_tender_block(extracted, text)

    assert extracted["total"] == 570
    assert extracted["amount_paid"] == 570


def test_plain_total_cash_tender_block_infers_total_from_change():
    from receipt_parser.pipeline_receipt import _fix_total_from_stacked_cash_tender_block

    extracted = {"total": 10542, "amount_paid": 10542, "points_used": 0}
    text = "\n".join([
        "合計",
        "お預り",
        "¥10,542",
        "お釣り",
        "¥5,000",
    ])

    _fix_total_from_stacked_cash_tender_block(extracted, text)

    assert extracted["total"] == 5542
    assert extracted["amount_paid"] == 5542


def test_service_table_projection_balances_duplicate_rows_discounts_and_fee():
    from receipt_parser.pipeline_receipt import _replace_service_table_items_when_balanced

    extracted = {
        "total": 6336,
        "subtotal": 5760,
        "taxes": [{"rate": "10%", "label": "内税", "amount": 576}],
        "line_items": [
            {"description": "礼服上", "qty": 1, "unit_price": 1210, "total": 1210, "tax_category": "10%"},
            {"description": "礼服コース", "qty": 1, "unit_price": 1100, "total": 1100, "tax_category": "10%"},
            {"description": "シルキーウェット 500", "qty": 1, "unit_price": 550, "total": 550, "tax_category": "10%"},
            {"description": "礼服ワンピース", "qty": 1, "unit_price": 1870, "total": 1870, "tax_category": "10%"},
            {"description": "シルキーウェット 700", "qty": 1, "unit_price": 770, "total": 770, "tax_category": "10%"},
            {"description": "物販", "qty": 1, "unit_price": 244, "total": 244, "tax_category": "10%"},
            {"description": "特別付加金", "qty": 1, "unit_price": 244, "total": 244, "tax_category": "10%"},
        ],
    }
    text = "\n".join([
        "商品名",
        "点数",
        "金額",
        "2-329 礼服上",
        "1,210",
        "礼服コース",
        "d",
        "1,100",
        "シルキーウェット 500",
        "d",
        "550",
        "10%OFF",
        "-121",
        "2-330 礼服ワンピース",
        "1.870",
        "礼服コース",
        "d",
        "1,100",
        "シルキーウェット 700",
        "d",
        "770",
        "10%OFF",
        "-187",
        "物販",
        "特別付加金",
        "244",
        "小計 クリーニング 2点 物販 2点",
        "6,336",
    ])

    _replace_service_table_items_when_balanced(extracted, text)

    assert [item["description"] for item in extracted["line_items"]] == [
        "礼服上",
        "礼服コース",
        "シルキーウェット 500",
        "礼服ワンピース",
        "礼服コース",
        "シルキーウェット 700",
        "特別付加金",
    ]
    assert [item["total"] for item in extracted["line_items"]] == [
        1089,
        1100,
        550,
        1683,
        1100,
        770,
        44,
    ]


def test_cash_tender_block_prefers_last_valid_value_triple():
    from receipt_parser.pipeline_receipt import _fix_total_from_stacked_cash_tender_block

    extracted = {"total": 980, "amount_paid": 980, "points_used": 0}
    text = "\n".join([
        "合計",
        "お預り",
        "お釣り",
        "お買上点数",
        "¥980",
        "¥1,078",
        "¥98",
        "¥98)",
        "¥1,078",
        "¥1,080",
        "¥2",
    ])

    _fix_total_from_stacked_cash_tender_block(extracted, text)

    assert extracted["total"] == 1078
    assert extracted["amount_paid"] == 1078


def test_unlabeled_cash_tender_change_block_uses_printed_total_and_tender():
    from receipt_parser.pipeline_receipt import _fix_unlabeled_cash_tender_change_block

    extracted = {"total": 2200, "amount_paid": 2200, "payment_method": "credit"}
    text = "\n".join([
        "小計",
        "¥2,018",
        "※8%内税対象",
        "¥2,175",
        "( ※ 8% 内)",
        "¥161",
        "合計",
        "¥2,179",
        "¥2,200",
        "お釣り",
        "¥21",
        "お買上点数",
        "7点",
    ])

    _fix_unlabeled_cash_tender_change_block(extracted, text)

    assert extracted["total"] == 2179
    assert extracted["amount_paid"] == 2200
    assert extracted["payment_method"] == "cash"


def test_final_receipt_output_repairs_apply_unlabeled_cash_tender_change_block():
    from receipt_parser.pipeline import _apply_final_receipt_output_repairs

    result = {
        "document_type": "receipt",
        "total": 2179,
        "amount_paid": 2179,
        "payment_method": "cash",
        "line_items": [],
        "taxes": [],
    }
    text = "\n".join([
        "小計",
        "¥2,018",
        "8%内税対象",
        "¥2,175",
        "税合計",
        "¥161",
        "合計",
        "¥2,179",
        "¥2,200",
        "お釣り",
        "¥21",
        "お買上点数",
        "7点",
    ])

    _apply_final_receipt_output_repairs(result, text)

    assert result["total"] == 2179
    assert result["amount_paid"] == 2200
    assert result["payment_method"] == "cash"


def test_final_receipt_output_repairs_restore_external_tax_summary_after_item_projection():
    from receipt_parser.pipeline import _apply_final_receipt_output_repairs

    result = {
        "document_type": "receipt",
        "subtotal": 7377,
        "total": 8913,
        "amount_paid": 8913,
        "payment_method": "credit",
        "points_used": 0,
        "taxes": [
            {"rate": "10%", "label": "外税", "amount": 711},
            {"rate": "8%", "label": "外税", "amount": 427},
        ],
        "line_items": [
            {"description": "日用品A", "qty": 2, "unit_price": 199, "total": 398, "tax_category": "10%"},
            {"description": "食品A", "qty": 1, "unit_price": 1404, "total": 1404, "tax_category": "8%"},
            {"description": "日用品B", "qty": 1, "unit_price": 7111, "total": 7111, "tax_category": "10%"},
        ],
    }
    ocr_text = "\n".join([
        "日用品A",
        "2コX単199",
        "¥398",
        "食品A",
        "¥1,404",
        "日用品B",
        "¥7,111",
        "小計",
        "8% 対象額",
        "8%税額",
        "¥398",
        "¥8,913",
        "¥1,404",
        "¥112",
        "10% 対象額",
        "10% 税額",
        "¥7,111",
        "¥711",
        "(税額合計",
        "¥823)",
        "合計",
        "¥9,736",
        "QUICPay",
        "¥9,736",
    ])

    _apply_final_receipt_output_repairs(result, ocr_text)

    assert result["subtotal"] == 8913.0
    assert result["total"] == 9736.0
    assert result["amount_paid"] == 9736.0
    assert result["taxes"] == [
        {"rate": "10%", "label": "外税", "amount": 711.0},
        {"rate": "8%", "label": "外税", "amount": 112.0},
    ]


def test_unlabeled_cash_tender_change_block_skips_explicit_cash_label():
    from receipt_parser.pipeline_receipt import _fix_unlabeled_cash_tender_change_block

    extracted = {"total": 449, "amount_paid": 449, "payment_method": "cash"}
    text = "\n".join([
        "合計",
        "¥449",
        "現金",
        "¥10,000",
        "お釣り",
        "¥9,551",
    ])

    _fix_unlabeled_cash_tender_change_block(extracted, text)

    assert extracted["total"] == 449
    assert extracted["amount_paid"] == 449


def test_unlabeled_cash_tender_change_block_skips_single_azukari_label():
    from receipt_parser.pipeline_receipt import _fix_unlabeled_cash_tender_change_block

    extracted = {"total": 380, "amount_paid": 380, "payment_method": "cash"}
    text = "\n".join([
        "合計",
        "¥380",
        "預",
        "¥400",
        "釣銭",
        "¥20",
    ])

    _fix_unlabeled_cash_tender_change_block(extracted, text)

    assert extracted["total"] == 380
    assert extracted["amount_paid"] == 380


def test_tax_label_uses_exclusive_when_items_plus_tax_matches_total():
    from receipt_parser.pipeline_receipt import normalize_tax_label

    assert normalize_tax_label(
        "内税",
        text="(8%対象 528 消費税 42)",
        subtotal=1028,
        total=570,
        tax_sum=42,
        items_sum=528,
    ) == "外税"


def test_tax_label_math_overrides_inner_tax_keyword_when_items_are_pretax():
    from receipt_parser.pipeline_receipt import normalize_tax_label

    assert normalize_tax_label(
        "内税",
        text="※8%内税対象\n¥2,033\n¥150",
        subtotal=1883,
        total=2033,
        tax_sum=150,
        items_sum=1883,
    ) == "外税"


def test_tax_label_keeps_explicit_inner_tax_amount_wording():
    from receipt_parser.pipeline_receipt import normalize_tax_label

    assert normalize_tax_label(
        "消費税",
        text="10%対象\n333(内税額30)",
        subtotal=303,
        total=333,
        tax_sum=30,
        items_sum=303,
    ) == "内税"


def test_tax_label_strong_tax_excluded_math_overrides_inner_consumption_tax_block():
    from receipt_parser.pipeline_receipt import normalize_tax_label

    assert normalize_tax_label(
        "内税",
        text="小計(税抜 8%)\n消費税等 (8%)\n(内消費税等 8%)",
        subtotal=961,
        total=1040,
        tax_sum=79,
        items_sum=961,
    ) == "外税"


def test_tax_label_treats_inner_percent_tax_line_as_inclusive():
    from receipt_parser.pipeline_receipt import normalize_tax_label

    assert normalize_tax_label(
        "内税",
        text="(8%対象\n(内 8%税\n¥570)\n¥42)",
        subtotal=528,
        total=570,
        tax_sum=42,
        items_sum=570,
    ) == "内税"


def test_tax_label_inner_target_total_marker_preserves_inclusive_label():
    from receipt_parser.pipeline_receipt import normalize_tax_label

    assert normalize_tax_label(
        "外税",
        text="10%内税対象\n¥725\n(10%内)\n¥65\n(税合計\n¥65)",
        subtotal=660,
        total=725,
        tax_sum=65,
        items_sum=660,
    ) == "内税"


def test_tax_label_inner_tax_target_wording_yields_to_pretax_item_math():
    from receipt_parser.pipeline_receipt import normalize_tax_label

    assert normalize_tax_label(
        "内税",
        text="10%内税対象\n(税合計\n¥98)\n合計\n¥1,078",
        subtotal=980,
        total=1078,
        tax_sum=98,
        items_sum=980,
    ) == "外税"


def test_tax_label_inner_target_amount_matching_total_preserves_inclusive_label():
    from receipt_parser.pipeline_receipt import normalize_tax_label

    assert normalize_tax_label(
        "内税",
        text="10%内税対象\n¥725\n¥65\n(税合計\n¥65)\n合計\n¥725",
        subtotal=660,
        total=725,
        tax_sum=65,
        items_sum=660,
    ) == "内税"


def test_tax_label_tax_included_price_notice_preserves_plain_target_label():
    from receipt_parser.pipeline_receipt import normalize_tax_label

    assert normalize_tax_label(
        "消費税",
        text="(8%対象\n1,770\n消費税\n141)\n総合計\n1,922\n※表示価格は店内税込価格",
        subtotal=1780,
        total=1922,
        tax_sum=142,
        items_sum=1780,
    ) == "内税"


def test_tax_label_footer_tax_included_notice_yields_to_pretax_item_math():
    from receipt_parser.pipeline_receipt import normalize_tax_label

    assert normalize_tax_label(
        "消費税",
        text=(
            "(8%対象\n1,770\n消費税\n141)\n総合計\n1,922\n"
            "現金\n5,002\n3,080\nお釣り\n発行日:05/07\n毎月\n10日\nは\n"
            "タンブラー\nDAY\n※表示価格は店内税込価格、お持ち帰\n"
            "税込価格は¥54OFF。"
        ),
        subtotal=1780,
        total=1922,
        tax_sum=142,
        items_sum=1780,
    ) == "外税"


def test_points_used_from_stacked_rakuten_point_tender():
    from receipt_parser.pipeline_receipt import extract_points_used

    text = "\n".join([
        "\u70b9\u6570",
        "\u697d\u5929\u30dd\u30a4\u30f3\u30c8",
        "\u304a\u3064\u308a",
        "\uffe5570",
        "\uffe5570",
        "\uffe5570)",
        "\uffe542)",
        "\uffe5570",
        "\uffe542)",
        "2\u70b9",
        "\uffe5570",
        "\uffe50",
        "\u697d\u5929\u30dd\u30a4\u30f3\u30c8\u30ab\u30fc\u30c9 ************ 8469",
        "\u30dd\u30a4\u30f3\u30c8\u5bfe\u8c61\u91d1\u984d",
        "\uffe5570",
    ])

    assert extract_points_used(text) == 570


def test_points_tender_reconciles_amount_paid_from_ocr():
    from receipt_parser.pipeline_receipt import reconcile_points_payment_from_ocr

    text = "\n".join([
        "\u70b9\u6570",
        "\u697d\u5929\u30dd\u30a4\u30f3\u30c8",
        "\u304a\u3064\u308a",
        "\uffe5570",
        "\uffe5570",
        "\uffe5570)",
        "\uffe542)",
        "\uffe5570",
        "\uffe542)",
        "2\u70b9",
        "\uffe5570",
        "\uffe50",
        "\u697d\u5929\u30dd\u30a4\u30f3\u30c8\u30ab\u30fc\u30c9 ************ 8469",
    ])
    extracted = {"total": 570.0, "points_used": 0.0, "amount_paid": 570.0}

    reconcile_points_payment_from_ocr(extracted, text)

    assert extracted["points_used"] == 570
    assert extracted["amount_paid"] == 0


def test_zero_points_restored_from_reward_context_without_redemption():
    from receipt_parser.pipeline_receipt import _restore_zero_points_when_no_redemption

    extracted = {"total": 36460, "amount_paid": 36460, "points_used": None}
    text = "\n".join([
        "クレジットカード",
        "36,460円",
        "グローバルカードで最大 1.5% 547円",
        "本日それぞれリワードが獲得できます!",
    ])

    _restore_zero_points_when_no_redemption(extracted, text)

    assert extracted["points_used"] == 0


def test_zero_points_restore_preserves_absent_value_when_redemption_printed():
    from receipt_parser.pipeline_receipt import _restore_zero_points_when_no_redemption

    extracted = {"total": 1000, "amount_paid": 1000, "points_used": None}
    text = "\n".join([
        "ポイント利用",
        "100 P",
        "会員番号 1234",
    ])

    _restore_zero_points_when_no_redemption(extracted, text)

    assert extracted["points_used"] is None


def test_amount_fragment_parser_normalizes_common_ocr_amount_tokens():
    from receipt_parser.pipeline_receipt import (
        _amount_from_yen_text,
        _parse_amount_fragment,
    )

    assert _parse_amount_fragment("1,234") == 1234
    assert _parse_amount_fragment("1.118") == 1118
    assert _parse_amount_fragment("12.5") == 12.5
    assert _parse_amount_fragment("abc") is None
    assert _amount_from_yen_text("合計 ¥1,234") == 1234


def test_postprocess_receipt_payment_phase_traces_zero_points_restore():
    from receipt_parser.pipeline_receipt import postprocess_receipt

    extracted = {
        "document_type": "receipt",
        "total": 36460,
        "amount_paid": 36460,
        "points_used": None,
        "line_items": [],
        "taxes": [],
    }
    text = "\n".join([
        "グローバルカードで最大 1.5% 547円",
        "本日それぞれリワードが獲得できます!",
    ])
    trace = []

    postprocess_receipt(
        extracted,
        text,
        0.9,
        {},
        {},
        "test-model",
        mutation_trace=trace,
    )

    assert extracted["points_used"] == 0
    payment_events = [
        event for event in trace
        if event["stage"] == "payment_points_reconciliation"
    ]
    assert payment_events
    assert payment_events[0]["changes"]["points_used"] == {"before": None, "after": 0}
    assert payment_events[0]["writes"]
    assert payment_events[0]["invariant"]


def test_single_count_customization_line_is_not_product():
    from receipt_parser.pipeline_receipt import _drop_non_product_line_items

    extracted = {
        "total": 570,
        "line_items": [
            {"description": "アイス トリプル エスプレッソ ラテ", "total": 528},
            {"description": "エクストラ ミルク", "total": 528},
        ],
    }
    text = "\n".join([
        "STARBUCKS",
        "アイス トリプル エスプレッソ ラテ",
        "528軽",
        "エクストラ ミルク",
        "(カスタム)",
        "本体合計(1点)",
    ])

    _drop_non_product_line_items(extracted, text)

    assert [item["description"] for item in extracted["line_items"]] == [
        "アイス トリプル エスプレッソ ラテ",
    ]


def test_zero_total_modifier_line_drops_when_priced_items_balance_total():
    from receipt_parser.pipeline_receipt import _drop_non_product_line_items

    extracted = {
        "total": 570,
        "line_items": [
            {"description": "アイス G ラテ", "total": 570},
            {"description": "エクストラ ミルク ライト アイス", "total": 0},
        ],
    }
    text = "\n".join([
        "* アイス G ラテ",
        "* エクストラ ミルク ライト アイス",
        "小計",
        "合計",
        "¥570",
        "¥570",
        "点数",
        "2点",
    ])

    _drop_non_product_line_items(extracted, text)

    assert [item["description"] for item in extracted["line_items"]] == ["アイス G ラテ"]


def test_item_line_star_legend_marks_matching_priced_item_reduced_rate():
    from receipt_parser.pipeline_receipt import _fix_tax_categories_from_price_line_markers

    extracted = {
        "line_items": [
            {"description": "アイス G ラテ", "total": 570, "tax_category": "0%"},
        ],
    }
    text = "\n".join([
        "* アイス G ラテ",
        "小計",
        "(8%対象",
        "合計",
        "¥570",
        "¥570",
        "* : 軽減税率対象商品です",
    ])

    _fix_tax_categories_from_price_line_markers(extracted, text)

    assert extracted["line_items"][0]["tax_category"] == "8%"


def test_tax_category_defaults_unmarked_items_to_standard_when_both_rates_printed():
    from receipt_parser.pipeline_receipt import assign_tax_categories

    items = [
        {"description": "ヤキモチ", "total": 118},
        {"description": "惣菜弁当", "total": 462},
    ]
    text = "\n".join([
        "ヤキモチ ※",
        "惣菜弁当",
        "8%対象",
        "消費税",
        "10%対象",
        "消費税",
    ])

    assign_tax_categories(
        items,
        text,
        {"taxes": [{"rate": "8%", "label": "外税", "amount": 9}]},
        {},
        extracted_taxes=[],
    )

    assert [item["tax_category"] for item in items] == ["8%", "10%"]


def test_ocr_food_shortcut_does_not_override_standard_category_on_mixed_rate_receipt():
    from receipt_parser.pipeline_receipt import _fix_tax_categories_from_ocr_markers

    items = [
        {"description": "九州産小麦九州ちゃんぽ", "total": 496, "tax_category": "10%"},
        {"description": "TV天かす60", "total": 98, "tax_category": "8%"},
    ]
    text = "\n".join([
        "外税8%対象額 ¥98",
        "外税10%対象額 ¥496",
        "※印は軽減税率8%対象商品です",
    ])

    _fix_tax_categories_from_ocr_markers(items, text)

    assert [item["tax_category"] for item in items] == ["10%", "8%"]


def test_reduced_tax_footnote_does_not_override_truncated_standard_tax_block():
    from receipt_parser.pipeline_receipt import _fix_tax_categories_from_ocr_markers

    items = [
        {"description": "牛サーロイン メンチ膳", "total": 1375, "tax_category": "10%"},
    ]
    text = "\n".join([
        "L 牛サーロイン メンチ膳",
        "¥1,375 内",
        "(10%内税対",
        "¥1,375)",
        "(10%内税額",
        "¥125)",
        "T印は軽減税率(8%) 適用商品",
    ])

    _fix_tax_categories_from_ocr_markers(items, text)

    assert items[0]["tax_category"] == "10%"


def test_extract_rate_bases_accepts_ocr_year_for_percent_marker():
    from receipt_parser.pipeline_receipt import extract_rate_bases

    text = "\n".join([
        "外税8%対象額",
        "¥2,986",
        "外税10年対象額",
        "¥3",
    ])

    assert extract_rate_bases(text) == {"8%": 2986.0, "10%": 3.0}


def test_extract_rate_bases_maps_stacked_tax_labels_to_values():
    from receipt_parser.pipeline_receipt import extract_rate_bases

    text = "\n".join([
        "外税8%対象額",
        "外税8%",
        "外税10年対象額",
        "外枠10%",
        "合計",
        "クレジット",
        "お釣り",
        "¥2,986",
        "¥238",
        "¥3",
        "¥0",
    ])

    assert extract_rate_bases(text) == {"8%": 2986.0, "10%": 3.0}


def test_extract_rate_bases_ignores_reduced_rate_footnotes_before_stacked_values():
    from receipt_parser.pipeline_receipt import extract_rate_bases

    text = "\n".join([
        "軽:軽減税率対象商品",
        "E:軽減税率対象商品 (店内飲食)",
        "外税10%対象額",
        "外税10%",
        "外税8%対象額",
        "外税8%",
        "合おお",
        "計",
        "お預り",
        "釣",
        "¥3",
        "¥0",
        "¥814",
        "¥65",
        "¥882",
        "¥1,000",
        "¥118",
    ])

    assert extract_rate_bases(text) == {"10%": 3.0, "8%": 814.0}


def test_extract_rate_bases_maps_stacked_tax_labels_past_item_values():
    from receipt_parser.pipeline_receipt import extract_rate_bases

    text = "\n".join([
        "小計",
        "外税8%対象額",
        "外税8%",
        "外税10年対象額",
        "外枠10%",
        "合計",
        "クレジット",
        "お釣り",
        "278*",
        "268",
        "118*",
        "¥2,989",
        "¥2,986",
        "¥238",
        "¥3",
        "¥0",
        "¥3,227",
        "¥3,227",
    ])

    assert extract_rate_bases(text) == {"8%": 2986.0, "10%": 3.0}


def test_extract_rate_bases_skips_previous_tax_value_before_next_target_label():
    from receipt_parser.pipeline_receipt import extract_rate_bases

    text = "\n".join([
        "税率 8%課税対象額",
        "¥7,800",
        "税率 8%税額",
        "税率10%課税対象額",
        "¥577",
        "¥231",
        "(消費税等",
        "税率10%税額",
        "合計",
        "QUICPay",
        "¥21",
        "¥8,031",
    ])

    assert extract_rate_bases(text) == {"8%": 7800.0, "10%": 231.0}


def test_extract_rate_bases_uses_immediate_values_in_interleaved_summary():
    from receipt_parser.pipeline_receipt import extract_rate_bases

    text = "\n".join([
        "小計",
        "税率 8% 課税対象額",
        "¥2,111",
        "¥2,274",
        "税率 8%税額",
        "¥168",
        "計",
        "税率10%課税対象額",
        "合計",
        "¥5",
        "¥2,279",
        "現計",
        "¥2,279",
    ])

    assert extract_rate_bases(text) == {"8%": 2111.0, "10%": 5.0}


def test_extract_rate_bases_maps_parenthesized_inclusive_block_values():
    from receipt_parser.pipeline_receipt import extract_rate_bases

    text = "\n".join([
        "(10% 対象",
        "(内消費税等",
        "(8% 対象",
        "合計",
        "¥5",
        "¥1,403",
        "¥5)",
        "¥0)",
        "¥1,398)",
        "(内消費税等",
        "¥103)",
    ])

    assert extract_rate_bases(text) == {"10%": 5.0, "8%": 1398.0}


def test_parenthesized_target_consumption_tax_uses_base_rate_arithmetic():
    from receipt_parser.pipeline_receipt import extract_financial_totals, extract_rate_bases

    text = "\n".join([
        "合計",
        "¥1,919",
        "(10%対象",
        "¥0 消費税",
        "(8%対象 ¥1,777 消費税",
        "¥0)",
        "¥142)",
    ])

    assert extract_rate_bases(text) == {"10%": 0.0, "8%": 1777.0}
    assert extract_financial_totals(text)["taxes"] == [
        {"rate": "10%", "label": "消費税等", "amount": 0.0},
        {"rate": "8%", "label": "内税", "amount": 142.0},
    ]


def test_bare_number_rate_summary_stack_maps_targets_and_tax_amounts():
    from receipt_parser.pipeline_receipt import extract_financial_totals, extract_rate_bases

    text = "\n".join([
        "8%対象",
        "350*",
        "450*",
        "420",
        "320*",
        "350*",
        "10内",
        "注)※印は軽減税率適用商品",
        "(内消費税等 8%",
        "10%対象",
        "3,272",
        "242)",
        "10",
        "合計",
        "10点",
        "¥3,282",
    ])

    assert extract_rate_bases(text) == {"8%": 3272.0, "10%": 10.0}
    assert extract_financial_totals(text)["taxes"] == [
        {"rate": "8%", "label": "内税", "amount": 242.0},
    ]


def test_interleaved_rate_summary_maps_bases_and_tax_amounts_without_yen_marks():
    from receipt_parser.pipeline_receipt import extract_financial_totals, extract_rate_bases

    text = "\n".join([
        "合計",
        "36,460",
        "8%対象",
        "30,713 (消費税",
        "2,275 )",
        "10% 対象",
        "5,747 (消費税",
        "522)",
        "領収金額",
        "36,460円",
        "(内消費税",
        "2,797円)",
    ])

    assert extract_rate_bases(text) == {"8%": 30713.0, "10%": 5747.0}
    assert extract_financial_totals(text)["taxes"] == [
        {"rate": "8%", "label": "内税", "amount": 2275.0},
        {"rate": "10%", "label": "内税", "amount": 522.0},
    ]


def test_bare_number_summary_restores_tax_and_drops_tiny_target_only_tax():
    from receipt_parser.pipeline_receipt import (
        _drop_unprinted_small_target_only_taxes,
        _restore_bare_number_tax_summary,
    )

    text = "\n".join([
        "8%対象",
        "(内消費税等 8%",
        "10%対象",
        "3,272",
        "242)",
        "10",
        "合計",
        "¥3,282",
    ])
    extracted = {
        "total": 3282,
        "subtotal": 3039,
        "taxes": [
            {"rate": "8%", "label": "外税", "amount": 242},
            {"rate": "10%", "label": "外税", "amount": 1},
        ],
        "line_items": [
            {"description": "A", "total": 3030, "tax_category": "8%"},
            {"description": "レジ袋", "total": 10, "tax_category": "10%"},
        ],
    }

    _restore_bare_number_tax_summary(extracted, text)
    _drop_unprinted_small_target_only_taxes(extracted, text)

    assert extracted["taxes"] == [{"rate": "8%", "label": "外税", "amount": 242.0}]
    assert extracted["subtotal"] == 3040.0


def test_code_prefixed_table_restores_item_descriptions_by_order():
    from receipt_parser.pipeline_receipt import _fix_code_table_descriptions_by_order

    text = "\n".join([
        "コード",
        "品名",
        "金額",
        "104-000004-000 サルシッチャドッグ",
        "1",
        "104-000010-000 どっさりキャベツと白",
        "身フライ",
        "105-000008-000 ヴルストクロワッサン",
        "1",
        "104-000003-000 てりたま",
        "1",
    ])
    extracted = {
        "line_items": [
            {"description": "サルシッチャドッグ", "total": 380},
            {"description": "シアン食パン", "total": 350},
            {"description": "シアン食パン", "total": 420},
            {"description": "シアン食パン", "total": 320},
        ]
    }

    _fix_code_table_descriptions_by_order(extracted, text)

    assert [item["description"] for item in extracted["line_items"]] == [
        "サルシッチャドッグ",
        "どっさりキャベツと白身フライ",
        "ヴルストクロワッサン",
        "てりたま",
    ]


def test_rate_base_refinement_uses_nearby_tax_amount_arithmetic():
    from receipt_parser.pipeline_receipt import (
        _refine_rate_bases_from_tax_amounts,
        extract_rate_bases,
    )

    text = "\n".join([
        "本体合計(5点)",
        "(10%対象",
        "200",
        "50軽",
        "355 軽",
        "(カスタム)",
        "10",
        "1,780",
        "1)",
        "10",
        "消費税",
        "(8%対象",
        "1,770",
        "消費税",
        "141)",
        "総合計",
        "1,922",
    ])
    taxes = [
        {"rate": "10%", "label": "外税", "amount": 1},
        {"rate": "8%", "label": "外税", "amount": 141},
    ]

    assert extract_rate_bases(text) == {"10%": 200.0, "8%": 1770.0}
    assert _refine_rate_bases_from_tax_amounts(
        extract_rate_bases(text), text, taxes
    ) == {"10%": 10.0, "8%": 1770.0}


def test_trial_kana_target_and_split_tax_lines_are_extracted():
    from receipt_parser.pipeline_receipt import extract_financial_totals, extract_rate_bases

    text = "\n".join([
        "8%外税 タイショウ",
        "¥378",
        "8%",
        "税",
        "¥30",
        "10%外税 タイショウ",
        "¥5,859",
        "10%外税",
        "¥585",
        "税合計",
        "¥615",
    ])

    assert extract_rate_bases(text) == {"8%": 378.0, "10%": 5859.0}
    assert extract_financial_totals(text)["taxes"] == [
        {"rate": "8%", "label": "外税", "amount": 30.0},
        {"rate": "10%", "label": "外税", "amount": 585.0},
    ]


def test_external_rate_base_survives_later_inclusive_zero_block():
    from receipt_parser.pipeline_receipt import extract_rate_bases

    text = "\n".join([
        "8%外税 タイショウ",
        "¥378",
        "8%",
        "税",
        "¥30",
        "10%外税 タイショウ",
        "¥5,859",
        "10%外税",
        "¥585",
        "(10%内税 タイショウ",
        "¥5)",
        "10%",
        "¥0)",
    ])

    assert extract_rate_bases(text) == {"8%": 378.0, "10%": 5859.0}


def test_trial_external_bases_rebalance_with_zero_tax_bag_and_restore_split_tax():
    from receipt_parser.pipeline_receipt import (
        _rebalance_tax_categories_to_rate_bases,
        _restore_printed_external_tax_amounts,
    )

    text = "\n".join([
        "8%外税 タイショウ",
        "¥378",
        "8%",
        "税",
        "¥30",
        "10%外税 タイショウ",
        "¥5,859",
        "10%外税",
        "¥585",
    ])
    items = [
        {"description": "HDX78 HW トラックBスタントボック", "total": 1999, "tax_category": "8%"},
        {"description": "パンパースさらさらケア", "total": 2498, "tax_category": "8%"},
        {"description": "ジョンソンベビーベビー", "total": 548, "tax_category": "8%"},
        {"description": "トイレマジックリン 消", "total": 158, "tax_category": "8%"},
        {"description": "赤ちゃんの手口ふき 3P", "total": 228, "tax_category": "8%"},
        {"description": "*フレッシュ北海道産生ク", "total": 378, "tax_category": "8%"},
        {"description": "クリネックスティシュー", "total": 428, "tax_category": "8%"},
        {"description": "内レジ袋", "total": 5, "tax_category": "10%"},
    ]
    extracted = {
        "taxes": [
            {"rate": "8%", "label": "内税", "amount": 248},
            {"rate": "10%", "label": "外税", "amount": 585},
        ]
    }

    _rebalance_tax_categories_to_rate_bases(items, text, extracted["taxes"], {"8%": 378, "10%": 5859})
    _restore_printed_external_tax_amounts(extracted, text)

    assert [item["tax_category"] for item in items] == [
        "10%", "10%", "10%", "10%", "10%", "8%", "10%", "10%"
    ]
    assert extracted["taxes"] == [
        {"rate": "8%", "label": "外税", "amount": 30.0},
        {"rate": "10%", "label": "外税", "amount": 585.0},
    ]


def test_external_tax_summary_with_later_inclusive_zero_block_survives_final_repairs():
    from receipt_parser.pipeline import _apply_final_receipt_output_repairs
    from receipt_parser.pipeline_receipt import extract_financial_totals, postprocess_receipt

    text = "\n".join([
        "領収証",
        "¥6,857-",
        "上記正に領収しました",
        "(消費税等 615円を含みます)",
        "HDX78 HW トラックBスタントボック",
        "値引",
        "¥3,998",
        "-1,999",
        "パンパースさらさらケア",
        "ジョンソンベビーベビー",
        "トイレマジックリン 消",
        "¥2,498",
        "¥548",
        "¥158",
        "赤ちゃんの手口ふき 3P",
        "¥228",
        "*フレッシュ北海道産生ク",
        "¥378",
        "クリネックスティシュー",
        "¥428",
        "内レジ袋",
        "¥5",
        "小計/",
        "8点",
        "¥6,242",
        "8%外税 タイショウ",
        "¥378",
        "8%",
        "税",
        "¥30",
        "10%外税 タイショウ",
        "¥5,859",
        "10%外税",
        "¥585",
        "(10%内税 タイショウ",
        "¥5)",
        "10%",
        "¥0)",
        "(税合計",
        "合計",
        "お預り",
        "¥615)",
        "¥6,857",
        "¥10,000",
    ])
    extracted = {
        "document_type": "receipt",
        "merchant": "RECEIPT",
        "date": "2026-06-03",
        "time": "20:18",
        "location": "",
        "currency": "JPY",
        "total": 6857.0,
        "payment_method": "cash",
        "amount_paid": 6857.0,
        "points_used": None,
        "line_items": [
            {"description": "HDX78 HW トラックBスタントボック", "qty": 1.0, "unit_price": 3998.0, "total": 1999.0, "tax_category": "0%", "discount": 1999.0, "discount_rate": ""},
            {"description": "パンパースさらさらケア", "qty": 1.0, "unit_price": 2498.0, "total": 2498.0, "tax_category": "0%", "discount": 0, "discount_rate": ""},
            {"description": "ジョンソンベビーベビー", "qty": 1.0, "unit_price": 548.0, "total": 548.0, "tax_category": "0%", "discount": 0, "discount_rate": ""},
            {"description": "トイレマジックリン 消", "qty": 1.0, "unit_price": 158.0, "total": 158.0, "tax_category": "0%", "discount": 0, "discount_rate": ""},
            {"description": "赤ちゃんの手口ふき 3P", "qty": 1.0, "unit_price": 228.0, "total": 228.0, "tax_category": "0%", "discount": 0, "discount_rate": ""},
            {"description": "フレッシュ北海道産生ク", "qty": 1.0, "unit_price": 378.0, "total": 378.0, "tax_category": "0%", "discount": 0, "discount_rate": ""},
            {"description": "クリネックスティシュー", "qty": 1.0, "unit_price": 428.0, "total": 428.0, "tax_category": "0%", "discount": 0, "discount_rate": ""},
            {"description": "内レジ袋", "qty": 1.0, "unit_price": 5.0, "total": 5.0, "tax_category": "0%", "discount": 0, "discount_rate": ""},
        ],
        "subtotal": 6242.0,
        "taxes": [
            {"rate": "8%", "label": "外税", "amount": 30.0},
            {"rate": "10%", "label": "外税", "amount": 585.0},
        ],
    }

    result = postprocess_receipt(
        extracted,
        text,
        0.9,
        extract_financial_totals(text),
        None,
        "unit-test",
    )
    _apply_final_receipt_output_repairs(result, text)

    assert result["total"] == 6857.0
    assert result["amount_paid"] == 6857.0
    assert result["subtotal"] == 6242.0
    assert result["taxes"] == [
        {"rate": "8%", "label": "外税", "amount": 30.0},
        {"rate": "10%", "label": "外税", "amount": 585.0},
    ]
    assert [item["tax_category"] for item in result["line_items"]] == [
        "10%", "10%", "10%", "10%", "10%", "8%", "10%", "10%"
    ]


def test_postprocess_recovers_header_location_and_external_tax_split():
    from receipt_parser.pipeline_receipt import postprocess_receipt

    text = "\n".join([
        "STARBUCKS ®",
        "八幡平野店",
        "#4301 TEL 093-883-7570",
        "1 T アイス トリプル エスプレッソ ラテ",
        "528軽",
        "1 V バニラクリーム フラペチーノ",
        "637",
        "1 キッズ",
        "アイス ココア",
        "ホイップ",
        "1 あんバタースコーンサンド",
        "1 有料ショッピングバッグ",
        "本体合計(5点)",
        "(10%対象",
        "200",
        "50軽",
        "355 軽",
        "10",
        "1,780",
        "1)",
        "10",
        "消費税",
        "(8%対象",
        "1,770",
        "消費税",
        "141)",
        "総合計",
        "1,922",
    ])
    extracted = {
        "document_type": "receipt",
        "merchant": "STARBUCKS",
        "location": None,
        "total": 1922,
        "subtotal": 1780,
        "amount_paid": 1922,
        "payment_method": "cash",
        "taxes": [
            {"rate": "10%", "label": "内税", "amount": 1},
            {"rate": "8%", "label": "内税", "amount": 141},
        ],
        "line_items": [
            {"description": "アイス トリプル エスプレッソ ラテ", "qty": 1, "unit_price": 528, "total": 528, "tax_category": "8%"},
            {"description": "バニラクリーム フラペチーノ", "qty": 1, "unit_price": 637, "total": 637, "tax_category": "8%"},
            {"description": "キッズ アイス ココア", "qty": 1, "unit_price": 200, "total": 200, "tax_category": "8%"},
            {"description": "ホイップ", "qty": 1, "unit_price": 50, "total": 50, "tax_category": "10%"},
            {"description": "あんバタースコーンサンド", "qty": 1, "unit_price": 355, "total": 355, "tax_category": "8%"},
            {"description": "有料ショッピングバッグ", "qty": 1, "unit_price": 10, "total": 10, "tax_category": "10%"},
        ],
    }

    postprocess_receipt(extracted, text, 0.9, {}, {}, "unit-test")

    assert extracted["location"] == "八幡平野店"
    assert {tax["rate"]: (tax["label"], tax["amount"]) for tax in extracted["taxes"]} == {
        "10%": ("外税", 1),
        "8%": ("外税", 141),
    }
    assert [item["tax_category"] for item in extracted["line_items"]] == [
        "8%", "8%", "8%", "8%", "8%", "10%"
    ]


def test_postprocess_recovers_split_body_total_layout_without_merchant_gate():
    from receipt_parser.pipeline_receipt import postprocess_receipt

    text = "\n".join([
        "COFFEE RECEIPT",
        "八幡平野店",
        "#4301 TEL 093-883-7570",
        "1 T アイス トリプル エスプレッソ ラテ",
        "528軽",
        "1 V バニラクリーム フラペチーノ",
        "637",
        "1 キッズ",
        "アイス ココア",
        "ホイップ",
        "1 あんバタースコーンサンド",
        "TOGOバッグ",
        "1 有料ショッピングバッグ",
        "本体合計(5点)",
        "(10%対象",
        "200",
        "50軽",
        "355 軽",
        "10",
        "1,780",
        "1)",
        "10",
        "消費税",
        "(8%対象",
        "1,770",
        "消費税",
        "141)",
        "総合計",
        "1,922",
    ])
    extracted = {
        "document_type": "receipt",
        "merchant": "COFFEE RECEIPT",
        "location": None,
        "total": 1922,
        "subtotal": None,
        "amount_paid": 1922,
        "payment_method": "cash",
        "taxes": [],
        "line_items": [],
    }

    postprocess_receipt(extracted, text, 0.9, {}, {}, "unit-test")

    assert extracted["location"] == "八幡平野店"
    assert extracted["subtotal"] == 1780
    assert [item["total"] for item in extracted["line_items"]] == [
        528, 637, 200, 50, 355, 10,
    ]
    assert {tax["rate"]: (tax["label"], tax["amount"]) for tax in extracted["taxes"]} == {
        "10%": ("外税", 1),
        "8%": ("外税", 141),
    }
    assert [item["tax_category"] for item in extracted["line_items"]] == [
        "8%", "8%", "8%", "8%", "8%", "10%"
    ]


def test_rate_base_rebalance_excludes_non_taxable_rows_in_larger_item_stream():
    from receipt_parser.pipeline_receipt import _rebalance_tax_categories_to_rate_bases

    totals = [
        5, 652, 186, 948, 150, 159, 311, 140, 254, 473, 412, 266, 261, 216,
        222, 3, 150, 90, 74, 131, 378, 104, 140, 93, 304, 330, 340, 197,
    ]
    items = [
        {"description": f"商品{idx:02d}", "total": total, "tax_category": "8%"}
        for idx, total in enumerate(totals, start=1)
    ]
    items[0]["tax_category"] = "10%"
    items[1]["tax_category"] = "0%"
    items[15]["tax_category"] = "10%"
    text = "\n".join([
        "外税8%対象額",
        "¥5,914",
        "外税8%",
        "¥473",
        "外税10%対象額",
        "¥423",
        "外税10%",
        "¥42",
        "非課税対象額",
        "¥652",
    ])

    _rebalance_tax_categories_to_rate_bases(
        items,
        text,
        [{"rate": "8%", "amount": 473}, {"rate": "10%", "amount": 42}],
        {"8%": 5914, "10%": 423},
    )

    ten_percent_totals = [
        item["total"] for item in items if item.get("tax_category") == "10%"
    ]
    assert ten_percent_totals == [5, 311, 3, 104]
    assert items[1]["tax_category"] == "0%"
    assert sum(item["total"] for item in items if item.get("tax_category") == "8%") == 5914


def test_postprocess_uses_printed_rate_bases_for_split_inclusive_tax_items():
    from receipt_parser.pipeline_receipt import postprocess_receipt

    text = "\n".join([
        "領収証",
        "(10%対象",
        "¥1,972",
        "内税 ¥179)",
        "(08%対象",
        "¥148 内税",
        "¥10)",
        "合計",
        "¥2,120",
        "内レジ袋 LL",
        "¥6",
        "内ソフィ 超熟睡ガード 3 ¥398",
        "内*カロリミットアップルスパークリングリフ ¥148",
        "内くっつかないクッキング",
        "内ケイト スーパーシャープライナーEX4",
        "¥138",
        "¥1,430",
        "*は軽減税率8%適用商品",
    ])
    extracted = {
        "document_type": "receipt",
        "merchant": "ドラッグストア",
        "currency": "JPY",
        "total": 2120,
        "subtotal": 1931,
        "amount_paid": 2120,
        "line_items": [
            {"description": "内レジ袋 LL", "qty": 1, "unit_price": 6, "total": 6, "tax_category": "0%"},
            {"description": "内ソフィ 超熟睡ガード 3", "qty": 1, "unit_price": 398, "total": 398, "tax_category": "0%"},
            {"description": "内*カロリミットアップルスパークリングリフ", "qty": 1, "unit_price": 148, "total": 148, "tax_category": "0%"},
            {"description": "内くっつかないクッキング", "qty": 1, "unit_price": 138, "total": 138, "tax_category": "0%"},
            {"description": "内ケイト スーパーシャープライナーEX4", "qty": 1, "unit_price": 1430, "total": 1430, "tax_category": "0%"},
        ],
        "taxes": [
            {"rate": "10%", "label": "内税", "amount": 179},
            {"rate": "8%", "label": "内税", "amount": 10},
        ],
    }

    postprocess_receipt(extracted, text, 0.9, {}, {}, "test-model")

    assert [item["tax_category"] for item in extracted["line_items"]] == [
        "10%", "10%", "8%", "10%", "10%"
    ]


def test_postprocess_rebalances_zero_categories_when_no_nontaxable_evidence():
    from receipt_parser.pipeline_receipt import postprocess_receipt

    text = "\n".join([
        "領収証",
        "食品ポリ袋L (バイオマス30 3",
        "液体BL替ミント",
        "198",
        "TV 水切りST浅型抗菌 180",
        "キレイ液体大増量",
        "248",
        "餃子の皮うす皮",
        "236X",
        "(2個 X 単118)",
        "牛豚ミンチ(解凍)",
        "640X",
        "小計",
        "¥4,810",
        "外税8%対象額",
        "¥4,181",
        "外税8%",
        "¥334",
        "外税10%対象額",
        "¥629",
        "外税10%",
        "¥62",
        "合計",
        "¥5,206",
        "※印は軽減税率8%対象商品",
    ])
    totals = [
        178, 3, 28, 98, 256, 198, 415, 138, 198, 488, 236, 158,
        70, 168, 198, 640, 606, 180, 248, 228, 78,
    ]
    descriptions = [
        "フランスサンコウホノル",
        "食品ポリ袋L (バイオマス30",
        "メイスィビジンモ",
        "TV減の恵みきざみねぎ",
        "TVオオバ",
        "にんじん 袋",
        "豚肉小間切れ",
        "バナナ",
        "キャベツ",
        "マルコメ だし入り料亭",
        "餃子の皮うす皮",
        "脂肪 ピーチ",
        "大根1/2カット",
        "TV緑豆春雨ショートタ",
        "液体BL替ミント",
        "牛豚ミンチ(解凍)",
        "牛豚ミンチ(解凍)",
        "TV 水切りST浅型抗菌",
        "キレイ液体大増量",
        "ロースハム",
        "コマツナ",
    ]
    extracted = {
        "document_type": "receipt",
        "merchant": "スーパー",
        "currency": "JPY",
        "total": 5206,
        "subtotal": 4810,
        "amount_paid": 5206,
        "line_items": [
            {
                "description": desc,
                "qty": 1,
                "unit_price": total,
                "total": total,
                "tax_category": "0%",
            }
            for desc, total in zip(descriptions, totals)
        ],
        "taxes": [
            {"rate": "8%", "label": "外税", "amount": 334},
            {"rate": "10%", "label": "外税", "amount": 62},
        ],
    }

    postprocess_receipt(extracted, text, 0.9, {}, {}, "test-model")

    categories = [item["tax_category"] for item in extracted["line_items"]]
    assert categories.count("10%") == 4
    assert categories.count("8%") == 17
    assert categories.count("0%") == 0
    assert sum(
        item["total"] for item in extracted["line_items"] if item["tax_category"] == "10%"
    ) == 629


def test_dense_sequence_rows_rebalance_tax_categories_after_reconstruction():
    from receipt_parser.pipeline_receipt import _replace_dense_sequence_rows_when_balanced

    text = "\n".join([
        "領収証",
        "2026/4/20 (月)",
        "レジ 0173",
        "11:04",
        "フランスサンコウホノル",
        "178",
        "食品ポリ袋L (バイオマス30 3",
        "メイスィビジンモ",
        "28",
        "TV減の恵みきざみねぎ 98",
        "TVオオバ",
        "(2個 X",
        "128)",
        "にんじん 袋",
        "豚肉小間切れ",
        "バナナ",
        "キャベツ",
        "256",
        "198",
        "415",
        "138※",
        "198",
        "マルコメ だし入り料亭 488",
        "餃子の皮うす皮",
        "236X",
        "(2個 X 単118)",
        "脂肪 ピーチ",
        "158",
        "大根1/2カット",
        "88*",
        "割引",
        "20%",
        "-18",
        "TV緑豆春雨ショートタ 168",
        "液体BL替ミント",
        "198",
        "牛豚ミンチ(解凍)",
        "640X",
        "牛豚ミンチ(解凍)",
        "606",
        "TV 水切りST浅型抗菌 180",
        "キレイ液体大増量",
        "248",
        "ロースハム",
        "228",
        "コマツナ",
        "78 A",
        "小計",
        "¥4,810",
        "外税8%対象額",
        "¥4,181",
        "外税8%",
        "¥334",
        "外税10%対象額",
        "¥629",
        "外税10%",
        "¥62",
        "合計",
        "¥5,206",
        "お買上商品数:22",
        "※印は軽減税率8%対象商品",
    ])
    extracted = {
        "document_type": "receipt",
        "subtotal": 4810,
        "total": 5206,
        "taxes": [
            {"rate": "8%", "label": "外税", "amount": 334},
            {"rate": "10%", "label": "外税", "amount": 62},
        ],
        "line_items": [],
    }

    _replace_dense_sequence_rows_when_balanced(extracted, text)

    categories = [item["tax_category"] for item in extracted["line_items"]]
    assert categories.count("10%") == 4
    assert categories.count("8%") == 17
    assert categories.count("0%") == 0
    assert sum(
        item["total"] for item in extracted["line_items"] if item["tax_category"] == "10%"
    ) == 629


def test_printed_external_tax_ignores_target_base_when_tax_is_zero():
    from receipt_parser.pipeline_receipt import _restore_printed_external_tax_amounts

    text = "\n".join([
        "外税8%対象額",
        "¥5,118",
        "外税8%",
        "¥409",
        "外税10%対象額",
        "外税10%",
        "¥3",
        "¥0",
        "合計",
        "¥5,530",
    ])
    extracted = {
        "taxes": [
            {"rate": "8%", "label": "外税", "amount": 412},
            {"rate": "10%", "label": "外税", "amount": 3},
        ]
    }

    _restore_printed_external_tax_amounts(extracted, text)

    assert extracted["taxes"] == [{"rate": "8%", "label": "外税", "amount": 409.0}]


def test_printed_inclusive_tax_blocks_restore_amounts_without_merchant_gate():
    from receipt_parser.pipeline_receipt import _fix_printed_tax_amounts_from_structural_blocks

    text = "\n".join([
        "領収証",
        "¥2,018-",
        "(10%対象",
        "¥1,800 内税",
        "¥163)",
        "(08%対象",
        "¥218 内税",
        "¥16)",
        "合計",
        "¥2,018",
    ])
    extracted = {
        "total": 2018,
        "taxes": [
            {"rate": "10%", "label": "内税", "amount": 1800},
            {"rate": "8%", "label": "内税", "amount": 218},
        ],
    }

    _fix_printed_tax_amounts_from_structural_blocks(extracted, text)

    assert extracted["taxes"] == [
        {"rate": "10%", "label": "内税", "amount": 163},
        {"rate": "8%", "label": "内税", "amount": 16},
    ]


def test_target_consumption_tax_block_creates_missing_tax_without_merchant_gate():
    from receipt_parser.pipeline_receipt import _fix_printed_tax_amounts_from_structural_blocks

    text = "\n".join([
        "領収書",
        "¥8,459-",
        "(10%対象",
        "¥8,459)",
        "(10%対象消費税",
        "¥769)",
    ])
    extracted = {"total": 8459, "taxes": []}

    _fix_printed_tax_amounts_from_structural_blocks(extracted, text)

    assert extracted["taxes"] == [{"rate": "10%", "label": "内税", "amount": 769}]


def test_single_inclusive_consumption_tax_line_repairs_amount_without_merchant_gate():
    from receipt_parser.pipeline_receipt import _fix_printed_tax_amounts_from_structural_blocks

    text = "\n".join([
        "領収証",
        "¥32,580-",
        "(内、 消費税 ¥2,961-)",
        "10%対象 ¥32,580",
        "合計",
        "¥32,580",
    ])
    extracted = {
        "total": 32580,
        "taxes": [{"rate": "10%", "label": "内税", "amount": 32580}],
    }

    _fix_printed_tax_amounts_from_structural_blocks(extracted, text)

    assert extracted["taxes"] == [{"rate": "10%", "label": "内税", "amount": 2961}]


def test_stacked_subtotal_target_tax_block_ignores_leading_item_price():
    from receipt_parser.pipeline_receipt import (
        _restore_printed_external_tax_amounts,
        extract_financial_totals,
        extract_rate_bases,
    )

    text = "\n".join([
        "A* スーパービックチョコ",
        "小計",
        "8% 対象額",
        "8%税額",
        "¥69",
        "¥8,515",
        "¥1,404",
        "¥112",
        "10% 対象額",
        "10% 税額",
        "¥7,111",
        "¥711",
        "合計",
        "¥9,338",
    ])

    taxes = sorted(extract_financial_totals(text)["taxes"], key=lambda t: t["rate"])
    assert taxes == [
        {"rate": "10%", "label": "税額", "amount": 711.0},
        {"rate": "8%", "label": "外税", "amount": 112.0},
    ]
    assert extract_financial_totals(text)["subtotal"] == 8515.0
    assert extract_rate_bases(text) == {"8%": 1404.0, "10%": 7111.0}
    extracted = {
        "taxes": [
            {"rate": "8%", "label": "外税", "amount": 427},
            {"rate": "10%", "label": "外税", "amount": 711},
        ]
    }
    _restore_printed_external_tax_amounts(extracted, text)
    assert sorted(extracted["taxes"], key=lambda t: t["rate"]) == [
        {"rate": "10%", "label": "外税", "amount": 711.0},
        {"rate": "8%", "label": "外税", "amount": 112.0},
    ]


def test_qty_detail_owner_repairs_duplicate_total_without_changing_unrelated_item():
    from receipt_parser.pipeline_receipt import _fix_qty_totals_from_ocr_unit_lines

    extracted = {
        "line_items": [
            {"description": "マイクロファイバークロス30P", "qty": 2, "unit_price": 199, "total": 398},
            {"description": "無添加ココナッツミルク", "qty": 1, "unit_price": 398, "total": 398},
        ]
    }
    text = "\n".join([
        "マイクロファイバークロス30P",
        "¥398",
        "無添加ココナッツミルク",
        "2コX単199",
        "¥398",
    ])

    _fix_qty_totals_from_ocr_unit_lines(extracted, text)

    assert extracted["line_items"][0] == {
        "description": "マイクロファイバークロス30P",
        "qty": 1.0,
        "unit_price": 398.0,
        "total": 398.0,
    }
    assert extracted["line_items"][1] == {
        "description": "無添加ココナッツミルク",
        "qty": 2.0,
        "unit_price": 199.0,
        "total": 398.0,
    }


def test_rate_base_rebalance_uses_qty_detail_owner_to_break_duplicate_amount_tie():
    from receipt_parser.pipeline_receipt import _rebalance_tax_categories_to_rate_bases

    items = [
        {"description": "マイクロファイバークロス30P", "qty": 1, "unit_price": 398, "total": 398, "tax_category": "10%"},
        {"description": "レノハピホワイトムスク替", "qty": 1, "unit_price": 498, "total": 498, "tax_category": "10%"},
        {"description": "レジ袋", "qty": 1, "unit_price": 5, "total": 5, "tax_category": "10%"},
        {"description": "OSBGバー45", "qty": 1, "unit_price": 980, "total": 980, "tax_category": "10%"},
        {"description": "ジュルルシャインマスカット", "qty": 1, "unit_price": 219, "total": 219, "tax_category": "8%"},
        {"description": "クレヨンしんちゃん リフレッ", "qty": 1, "unit_price": 580, "total": 580, "tax_category": "10%"},
        {"description": "無添加ココナッツミルク", "qty": 2, "unit_price": 199, "total": 398, "tax_category": "10%"},
        {"description": "MKMEX + クール26", "qty": 1, "unit_price": 1300, "total": 1300, "tax_category": "10%"},
        {"description": "スカイハイ07", "qty": 1, "unit_price": 1540, "total": 1540, "tax_category": "10%"},
        {"description": "FMリキッドFR12", "qty": 1, "unit_price": 1810, "total": 1810, "tax_category": "10%"},
        {"description": "情熱価格 白桃缶詰白桃ジ", "qty": 1, "unit_price": 359, "total": 359, "tax_category": "8%"},
        {"description": "あり情 大粒みかん(ジュース", "qty": 1, "unit_price": 359, "total": 359, "tax_category": "8%"},
        {"description": "スーパービックチョコ", "qty": 1, "unit_price": 69, "total": 69, "tax_category": "8%"},
    ]
    text = "\n".join([
        "マイクロファイバークロス30P",
        "¥398",
        "* ジュルルシャインマスカット",
        "¥219",
        "無添加ココナッツミルク",
        "2コX単199",
        "¥398",
        "* 情熱価格 白桃缶詰白桃ジ ¥359",
        "*あり情 大粒みかん(ジュース ¥359",
        "A* スーパービックチョコ",
        "小計",
        "8% 対象額",
        "8%税額",
        "¥69",
        "¥8,515",
        "¥1,404",
        "¥112",
        "10% 対象額",
        "10% 税額",
        "¥7,111",
        "¥711",
    ])

    _rebalance_tax_categories_to_rate_bases(
        items,
        text,
        [{"rate": "8%", "label": "外税", "amount": 112}, {"rate": "10%", "label": "外税", "amount": 711}],
        {"8%": 1404, "10%": 7111},
    )

    by_desc = {item["description"]: item["tax_category"] for item in items}
    assert by_desc["マイクロファイバークロス30P"] == "10%"
    assert by_desc["無添加ココナッツミルク"] == "8%"
    assert by_desc["OSBGバー45"] == "10%"
    assert by_desc["FMリキッドFR12"] == "10%"


def test_name_bag_amount_shift_repairs_shifted_price_and_unit_prices():
    from receipt_parser.pipeline_receipt import _fix_name_bag_amount_shift_from_ocr

    extracted = {
        "line_items": [
            {"description": "バイオマス有料レジ袋Mサイ 2", "qty": 1, "unit_price": 78, "total": 78, "tax_category": "10%"},
            {"description": "バイオマス有料レジ袋Mサイ", "qty": 1, "unit_price": 2, "total": 2, "tax_category": "10%"},
            {"description": "通常商品B", "qty": 1, "unit_price": None, "total": 248, "tax_category": "8%"},
            {"description": "通常商品C", "qty": 1, "unit_price": 88, "total": 88, "tax_category": "8%"},
        ]
    }
    ocr_text = "\n".join([
        "商品A・小",
        "バイオマス有料レジ袋Mサイ 2",
        "78%",
        "通常商品B 248",
        "通常商品C",
        "88*",
        "小計",
        "¥416",
        "外税8%対象額",
        "¥414",
        "外税10%対象額",
        "¥2",
    ])

    _fix_name_bag_amount_shift_from_ocr(extracted, ocr_text)

    assert extracted["line_items"][0]["description"] == "商品A•小"
    assert extracted["line_items"][0]["unit_price"] == 78.0
    assert extracted["line_items"][0]["tax_category"] == "8%"
    assert extracted["line_items"][1]["total"] == 2.0
    assert extracted["line_items"][2]["unit_price"] == 248.0


def test_name_bag_amount_shift_fills_unit_prices_when_rows_already_balance():
    from receipt_parser.pipeline_receipt import _fix_name_bag_amount_shift_from_ocr

    extracted = {
        "line_items": [
            {"description": "商品A・小", "qty": 1, "unit_price": 78, "total": 78, "tax_category": "8%"},
            {"description": "バイオマス有料レジ袋Mサイ", "qty": 1, "unit_price": 2, "total": 2, "tax_category": "10%"},
            {"description": "通常商品B", "qty": 1, "unit_price": None, "total": 248, "tax_category": "8%"},
            {"description": "通常商品C", "qty": 1, "unit_price": None, "total": 88, "tax_category": "8%"},
        ]
    }
    ocr_text = "\n".join([
        "商品A・小",
        "バイオマス有料レジ袋Mサイ 2",
        "78%",
        "通常商品B 248",
        "通常商品C",
        "88*",
        "小計",
        "¥416",
        "外税8%対象額",
        "¥414",
        "外税10%対象額",
        "¥2",
    ])

    _fix_name_bag_amount_shift_from_ocr(extracted, ocr_text)

    assert [item["unit_price"] for item in extracted["line_items"]] == [78, 2, 248.0, 88.0]


def test_jan_pos_projection_repairs_shifted_discount_rows_without_merchant_gate():
    from receipt_parser.pipeline_receipt import _replace_jan_pos_items_when_balanced

    extracted = {
        "total": 1911,
        "subtotal": 1911,
        "line_items": [
            {"description": "豚肉", "qty": 1, "unit_price": 680, "total": 544, "tax_category": "8%", "discount": 136, "discount_rate": "20%"},
            {"description": "レジ袋5円", "qty": 2, "unit_price": 5, "total": 10, "tax_category": "10%"},
        ],
        "taxes": [],
    }
    ocr_text = "\n".join([
        "462000009801 JAN",
        "000200 特級うまくち",
        "¥308",
        "462000009802 JAN",
        "000062 豚肉",
        "¥980",
        "462000006800 JAN",
        "000062 豚肉",
        "¥680",
        "操作割引2",
        "20%",
        "-136",
        "4571228941636JAN",
        "000226 レジ袋5円",
        "2コX単5",
        "4968454111443 JAN",
        "¥10",
        "2100119400994JAN",
        "000051 * 白菜",
        "操作割引3",
        "4902170045507 JAN",
        "¥436",
        "¥99",
        "30%",
        "-30",
        "小計",
        "¥1,911",
    ])

    _replace_jan_pos_items_when_balanced(extracted, ocr_text, {"subtotal": 1911})

    assert any(item["description"] == "特級うまくち" for item in extracted["line_items"])
    pork = [item for item in extracted["line_items"] if item["description"] == "豚肉"]
    assert [item["total"] for item in pork] == [980.0, 544.0]
    bag = next(item for item in extracted["line_items"] if item["description"] == "レジ袋")
    assert bag["qty"] == 2
    assert bag["unit_price"] == 5
    assert bag["total"] == 10
    cabbage = next(item for item in extracted["line_items"] if item["description"] == "白菜")
    assert cabbage["qty"] == 1
    assert cabbage["unit_price"] == 99
    assert cabbage["total"] == 69


def test_jan_pos_projection_keeps_confirming_price_with_previous_quantity_row():
    from receipt_parser.pipeline_receipt import _replace_jan_pos_items_when_balanced

    extracted = {
        "total": 1204,
        "subtotal": 1115,
        "taxes": [{"rate": "8%", "label": "外税", "amount": 89}],
        "line_items": [
            {"description": "商品A", "qty": 1, "unit_price": 436, "total": 436},
            {"description": "商品B", "qty": 1, "unit_price": 436, "total": 436},
            {"description": "レジ袋5円", "qty": 1, "unit_price": 10, "total": 10},
            {"description": "商品C", "qty": 1, "unit_price": 100, "total": 100},
            {"description": "商品D", "qty": 1, "unit_price": 200, "total": 200},
            {"description": "商品E", "qty": 1, "unit_price": 300, "total": 300},
        ],
    }
    ocr_text = "\n".join([
        "1111111111111JAN",
        "000001*商品A",
        "2コX単218",
        "2222222222222JAN",
        "000002*商品B",
        "操作割引3",
        "3333333333333JAN",
        "¥436",
        "¥99",
        "30%",
        "-30",
        "4444444444444JAN",
        "000003 レジ袋5円",
        "2コX単5",
        "¥10",
        "5555555555555JAN",
        "000004*商品C",
        "¥100",
        "6666666666666JAN",
        "000005*商品D",
        "¥200",
        "7777777777777JAN",
        "000006*商品E",
        "¥300",
        "小計",
        "¥1,115",
        "税率 8%課税対象額",
        "¥1,105",
        "税率 8%税額",
        "¥88",
        "税率10%課税対象額",
        "¥10",
        "税率10%税額",
        "¥1",
        "合計",
        "¥1,204",
    ])

    _replace_jan_pos_items_when_balanced(
        extracted,
        ocr_text,
        {"subtotal": 1115, "taxes": [{"rate": "8%", "label": "外税", "amount": 89}]},
    )

    assert extracted["line_items"] == [
        {
            "description": "商品A",
            "qty": 2.0,
            "unit_price": 218.0,
            "total": 436.0,
            "tax_category": "8%",
            "discount": 0,
            "discount_rate": "",
        },
        {
            "description": "商品B",
            "qty": 1.0,
            "unit_price": 99.0,
            "total": 69.0,
            "tax_category": "8%",
            "discount": 30.0,
            "discount_rate": "30%",
        },
        {
            "description": "レジ袋",
            "qty": 2.0,
            "unit_price": 5.0,
            "total": 10.0,
            "tax_category": "10%",
            "discount": 0,
            "discount_rate": "",
        },
        {
            "description": "商品C",
            "qty": 1.0,
            "unit_price": 100.0,
            "total": 100.0,
            "tax_category": "8%",
            "discount": 0,
            "discount_rate": "",
        },
        {
            "description": "商品D",
            "qty": 1.0,
            "unit_price": 200.0,
            "total": 200.0,
            "tax_category": "8%",
            "discount": 0,
            "discount_rate": "",
        },
        {
            "description": "商品E",
            "qty": 1.0,
            "unit_price": 300.0,
            "total": 300.0,
            "tax_category": "8%",
            "discount": 0,
            "discount_rate": "",
        },
    ]
    assert extracted["taxes"] == [
        {"rate": "10%", "label": "外税", "amount": 1.0},
        {"rate": "8%", "label": "外税", "amount": 88.0},
    ]


def test_jan_pos_projection_rebalances_interleaved_external_tax_bases():
    from receipt_parser.pipeline_receipt import _replace_jan_pos_items_when_balanced

    extracted = {
        "total": 8031,
        "subtotal": 7433,
        "taxes": [
            {"rate": "8%", "label": "外税", "amount": 577},
            {"rate": "10%", "label": "外税", "amount": 21},
        ],
        "line_items": [],
    }
    ocr_text = "\n".join([
        "4902053118854 JAN",
        "000406 トーラクホイップ",
        "¥328",
        "466000004804JAN",
        "000066*ミートデリカ",
        "¥480",
        "¥100",
        "000501 100円均一",
        "4571228941636JAN",
        "000226 レジ袋NO50",
        "2コX単5",
        "4942355503682JAN",
        "¥10",
        "000404 冷凍パプリカ",
        "¥198",
        "2100001400996 JAN",
        "000051 レタス",
        "¥99",
        "4942355102816JAN",
        "1000308 * れんこん薄切り",
        "¥98",
        "4942355064824JAN",
        "000302 クリームチーズ [要冷蔵]",
        "3コ X358",
        "000501 100円均一",
        "¥1,074",
        "¥100",
        "465000009802JAN",
        "000065 精肉惣菜",
        "¥980",
        "4961681007510JAN",
        "000302 ふりかけるチーズ2 ¥338",
        "8801073142800JAN",
        "000214 カルボナーラブルダ ¥648",
        "4512776221030JAN",
        "000217 福岡県産夢つくし ¥2,980",
        "小計",
        "¥7,433",
        "税率 8%課税対象額",
        "¥7,800",
        "税率 8%税額",
        "税率10%課税対象額",
        "¥577",
        "¥231",
        "(消費税等",
        "税率10%税額",
        "合計",
        "QUICPay",
        "¥21",
        "¥8,031",
        "*印は軽減税率(8%) 適用商品です",
    ])

    _replace_jan_pos_items_when_balanced(
        extracted,
        ocr_text,
        {"subtotal": 7433, "taxes": extracted["taxes"]},
    )

    categories = [item["tax_category"] for item in extracted["line_items"]]
    assert categories.count("10%") == 3
    assert categories.count("8%") == 10
    assert sum(
        item["total"] for item in extracted["line_items"] if item["tax_category"] == "10%"
    ) == 210


def test_single_service_receipt_reconstructs_inclusive_tax_without_tax_line():
    from receipt_parser.pipeline_receipt import _fix_single_service_inclusive_tax

    extracted = {
        "total": 510,
        "subtotal": 510,
        "taxes": [],
        "line_items": [
            {"description": "通行料金", "qty": 1, "unit_price": 510, "total": 510}
        ],
    }

    _fix_single_service_inclusive_tax(extracted, "通行料金\n通行料金の消費税率は 10%\n(クレジット)")

    assert extracted["subtotal"] == 464
    assert extracted["taxes"] == [{"rate": "10%", "label": "内税", "amount": 46.0}]
    assert extracted["line_items"][0]["tax_category"] == "10%"


def test_bare_service_receipt_without_itemization_does_not_create_item():
    from receipt_parser.pipeline_receipt import _fix_bare_service_receipt_without_itemization

    extracted = {
        "total": 3050,
        "payment_method": "cash",
        "line_items": [{
            "description": "Bowling",
            "qty": 1,
            "unit_price": 3050.0,
            "total": 3050.0,
            "tax_category": "10%",
            "discount": 0,
            "discount_rate": "",
        }],
    }
    text = "領収証\n¥3,050-\n但し、消費税等\n円を含みます。)\nBOWLING/ AMUSEMENT"

    _fix_bare_service_receipt_without_itemization(extracted, text)

    assert extracted["payment_method"] is None
    assert extracted["line_items"] == []


def test_official_department_header_prefers_authority_name():
    from receipt_parser.pipeline_receipt import _fix_company_name_merchant

    extracted = {"merchant": "市民課"}
    text = "\n".join([
        "領収書",
        "テスト市役所",
        "市民課",
        "2026年 3月 9日 (月) 11:48",
        "0001 証明書",
        "¥300",
    ])

    _fix_company_name_merchant(extracted, text)

    assert extracted["merchant"] == "テスト市役所"


def test_nontaxable_admin_fee_receipt_recovers_single_certificate_item():
    from receipt_parser.pipeline_receipt import _fix_line_items

    extracted = {
        "total": 300,
        "subtotal": 300,
        "taxes": [],
        "line_items": [],
    }
    text = "\n".join([
        "領収書",
        "テスト市役所",
        "市民課",
        "2026年 3月 9日 (月) 11:48",
        "0001 納税証明",
        "¥300",
        "小",
        "計",
        "¥300",
        "非課税対象額",
        "¥300",
        "合計",
        "¥300",
        "(消費税等",
        "¥0)",
    ])

    _fix_line_items(extracted, text)

    assert extracted["line_items"] == [{
        "description": "納税証明",
        "qty": 1,
        "unit_price": 300.0,
        "total": 300.0,
        "tax_category": "0%",
        "discount": 0,
        "discount_rate": "",
    }]
    assert extracted["taxes"] == [{"rate": "0%", "label": "非課税", "amount": 0}]


def test_postprocess_preserves_nontaxable_admin_fee_item_with_split_subtotal_label():
    from receipt_parser.pipeline_receipt import postprocess_receipt

    extracted = {
        "merchant": "市民課",
        "total": 300,
        "subtotal": 300,
        "amount_paid": 300,
        "payment_method": "cash",
        "taxes": [],
        "line_items": [],
    }
    text = "\n".join([
        "領収書",
        "テスト市役所",
        "市民課",
        "2026年 3月 9日 (月) 11:48",
        "0001 納税証明",
        "¥300",
        "小",
        "計",
        "¥300",
        "非課税対象額",
        "¥300",
        "合計",
        "¥300",
        "お預り",
        "¥300",
        "(消費税等",
        "¥0)",
    ])

    result = postprocess_receipt(extracted, text, 0.9, {}, {}, "test-model")

    assert result["merchant"] == "テスト市役所"
    assert result["line_items"] == [{
        "description": "納税証明",
        "qty": 1,
        "unit_price": 300.0,
        "total": 300.0,
        "tax_category": "0%",
        "discount": 0,
        "discount_rate": "",
    }]
    assert result["taxes"] == [{"rate": "0%", "label": "非課税", "amount": 0}]


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


def test_location_resolution_uses_area_code_for_non_admin_fragment():
    from receipt_parser.pipeline import _resolve_location

    extracted = {"merchant": "コスモス", "location": "元店"}
    ocr_text = "\n".join([
        "ドラックストア",
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


def test_purchase_store_metadata_location_expansion_trims_to_printed_city():
    from receipt_parser.pipeline import _trim_purchase_store_metadata_location

    extracted = {"merchant": "テストストア", "location": "宗像市くりえいと"}
    ocr_text = "\n".join([
        "Baby & Kids",
        "テストストア",
        "ご購入店 サンリブくりえいと宗像店",
        "TEL070-1234-5678",
        "X 福岡県",
    ])

    _trim_purchase_store_metadata_location(extracted, ocr_text)

    assert extracted["location"] == "宗像市"


def test_purchase_store_metadata_location_preserves_exact_printed_address():
    from receipt_parser.pipeline import _trim_purchase_store_metadata_location

    extracted = {"merchant": "テストストア", "location": "宗像市くりえいと1丁目5-1"}
    ocr_text = "\n".join([
        "テストストア",
        "ご購入店 テストモール中央店",
        "住所 宗像市くりえいと1丁目5-1",
        "TEL070-1234-5678",
    ])

    _trim_purchase_store_metadata_location(extracted, ocr_text)

    assert extracted["location"] == "宗像市くりえいと1丁目5-1"


def test_labeled_purchase_site_location_recovers_printed_area_token():
    from receipt_parser.pipeline_receipt import _recover_labeled_purchase_site_location

    extracted = {"merchant": "テストストア", "location": ""}
    ocr_text = "\n".join([
        "御支払方法:",
        "クレジットカード",
        "購入倉庫店:北九州倉庫店",
        "電話番号 0570-200-800",
    ])

    _recover_labeled_purchase_site_location(extracted, ocr_text)

    assert extracted["location"] == "北九州"


def test_labeled_purchase_site_location_overrides_registered_address():
    from receipt_parser.pipeline_receipt import _recover_labeled_purchase_site_location

    extracted = {"merchant": "テストストア", "location": "千葉県木更津市瓜倉361番地"}
    ocr_text = "\n".join([
        "登録番号",
        "千葉県木更津市瓜倉361番地",
        "御支払方法:",
        "クレジットカード",
        "購入倉庫店:北九州倉庫店",
    ])

    _recover_labeled_purchase_site_location(extracted, ocr_text)

    assert extracted["location"] == "北九州"


def test_split_address_location_recovers_street_line_without_merchant_gate():
    from receipt_parser.pipeline_receipt import _fix_split_address_location_from_ocr

    extracted = {"merchant": "テストストア", "location": "宗像市"}
    ocr_text = "\n".join([
        "[ 領収書 ]",
        "テストストア",
        "福岡県宗像市",
        "赤間駅前2-6-10",
        "TEL:0940-32-5666",
        "登録番号:T4810624886892",
    ])

    _fix_split_address_location_from_ocr(extracted, ocr_text)

    assert extracted["location"] == "福岡県宗像市赤間駅前2-6-10"


def test_header_branch_store_location_recovers_visible_store_token():
    from receipt_parser.pipeline import _recover_header_branch_store_location

    extracted = {"merchant": "BRAND", "location": ""}
    ocr_text = "\n".join([
        "BRAND",
        "ブランド サンリブ店",
        "TEL",
        "050-0000-0000",
        "** 領収証 **",
        "2026年05月16日",
    ])

    _recover_header_branch_store_location(extracted, ocr_text)

    assert extracted["location"] == "サンリブ店"


def test_header_branch_store_location_overrides_broad_admin_fragment():
    from receipt_parser.pipeline import _recover_header_branch_store_location

    extracted = {"merchant": "BRAND", "location": "北九州市八幡区"}
    ocr_text = "\n".join([
        "BRAND",
        "八幡平野店",
        "#4301 TEL 093-883-7570",
        "総合計",
        "1,922",
    ])

    _recover_header_branch_store_location(extracted, ocr_text)

    assert extracted["location"] == "八幡平野店"


def test_header_branch_store_location_preserves_admin_fragment_for_district_only_branch():
    from receipt_parser.pipeline import _recover_header_branch_store_location

    extracted = {"merchant": "BRAND", "location": "福岡市博多区"}
    ocr_text = "\n".join([
        "BRAND",
        "博多店",
        "TEL 092-000-0000",
        "合計",
        "1,650",
    ])

    _recover_header_branch_store_location(extracted, ocr_text)

    assert extracted["location"] == "福岡市博多区"


def test_header_branch_store_location_overrides_admin_with_short_store_prefixed_root():
    from receipt_parser.pipeline import _recover_header_branch_store_location

    extracted = {"merchant": "BRAND", "location": "宗像市"}
    ocr_text = "\n".join([
        "BRAND",
        "サンリブ宗像店",
        "TEL 0940-00-0000",
        "合計",
        "3,036",
    ])

    _recover_header_branch_store_location(extracted, ocr_text)

    assert extracted["location"] == "サンリブ宗像店"


def test_header_branch_store_location_preserves_specific_address():
    from receipt_parser.pipeline import _recover_header_branch_store_location

    extracted = {"merchant": "BRAND", "location": "福岡県宗像市赤間駅前2-6-10"}
    ocr_text = "\n".join([
        "BRAND",
        "サンリブ店",
        "TEL 0940-32-5666",
    ])

    _recover_header_branch_store_location(extracted, ocr_text)

    assert extracted["location"] == "福岡県宗像市赤間駅前2-6-10"


def test_final_receipt_output_repairs_recover_blank_location_from_phone_area():
    from receipt_parser.pipeline import _apply_final_receipt_output_repairs

    result = {
        "document_type": "receipt",
        "merchant": "テストストア",
        "location": None,
        "line_items": [],
    }
    ocr_text = "\n".join([
        "365日毎日安い!",
        "ドラックストア",
        "テストストア",
        "元店 TEL0940-72-5355",
        "領収証",
    ])

    _apply_final_receipt_output_repairs(result, ocr_text)

    assert result["location"] == "宗像市"


def test_final_receipt_output_repairs_trim_footer_noise_from_city_location():
    from receipt_parser.pipeline import _apply_final_receipt_output_repairs

    result = {
        "document_type": "receipt",
        "merchant": "テストストア",
        "location": "宗像市ご来",
        "line_items": [],
    }
    ocr_text = "\n".join([
        "テストストア",
        "元店 TEL0940-72-5355",
        "ご来店ありがとうございます。",
    ])

    _apply_final_receipt_output_repairs(result, ocr_text)

    assert result["location"] == "宗像市"


def test_final_receipt_output_repairs_keeps_exact_phone_city_over_short_branch():
    from receipt_parser.pipeline import _apply_final_receipt_output_repairs

    result = {
        "document_type": "receipt",
        "merchant": "テストストア",
        "location": "宗像市",
        "line_items": [],
    }
    ocr_text = "\n".join([
        "テストストア",
        "赤間店 TEL0940-72-5355",
        "領収証",
    ])

    _apply_final_receipt_output_repairs(result, ocr_text)

    assert result["location"] == "宗像市"


def test_final_receipt_output_repairs_prefers_branch_over_phone_city_fragment():
    from receipt_parser.pipeline import _apply_final_receipt_output_repairs

    result = {
        "document_type": "receipt",
        "merchant": "テストストア",
        "location": "宗像市赤間",
        "line_items": [],
    }
    ocr_text = "\n".join([
        "テストストア",
        "赤間店",
        "TEL: 0940-35-7611",
    ])

    _apply_final_receipt_output_repairs(result, ocr_text)

    assert result["location"] == "赤間店"


def test_final_receipt_output_repairs_keeps_phone_city_over_long_store_header():
    from receipt_parser.pipeline import _apply_final_receipt_output_repairs

    result = {
        "document_type": "receipt",
        "merchant": "テストストア",
        "location": "宗像市",
        "line_items": [],
    }
    ocr_text = "\n".join([
        "テストストア",
        "マックスバリュくりえいと宗像店",
        "TEL0940-39-2100",
    ])

    _apply_final_receipt_output_repairs(result, ocr_text)

    assert result["location"] == "宗像市"


def test_final_receipt_output_repairs_trims_store_in_store_header_location():
    from receipt_parser.pipeline import _apply_final_receipt_output_repairs

    result = {
        "document_type": "receipt",
        "merchant": "BRAND",
        "location": "BRAND サンリブくりえいと宗像店",
        "line_items": [],
    }
    ocr_text = "\n".join([
        "BRAND サンリブくりえいと宗像店",
        "TEL 070-1556-6722",
        "領収書",
    ])

    _apply_final_receipt_output_repairs(result, ocr_text)

    assert result["location"] == "宗像市"


def test_location_resolution_ignores_purchase_store_metadata_as_branch(monkeypatch):
    from receipt_parser.pipeline import _resolve_location
    import receipt_parser.llm as llm

    captured = {}

    class Response:
        content = '{"location": null}'

    def fake_chat(*_args, **kwargs):
        captured["prompt"] = kwargs["messages"][0]["content"]
        return Response()

    monkeypatch.setattr(llm, "_llm_chat", fake_chat)
    extracted = {"merchant": "テストストア", "location": None}
    ocr_text = "\n".join([
        "Baby & Kids",
        "テストストア",
        "ご購入店 サンリブくりえいと宗像店",
        "TEL070-1234-5678",
    ])

    location, warning = _resolve_location(extracted, ocr_text, "unit-test")

    assert location is None
    assert warning == "Location resolution: could not determine city/ward from available clues"
    assert "Branch/store name: ご購入店" not in captured["prompt"]
    assert "Branch/store name:" not in captured["prompt"]


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


def test_postprocess_receipt_mutation_trace_records_semantic_phase_changes():
    from receipt_parser.pipeline_receipt import postprocess_receipt

    extracted = {
        "document_type": "receipt",
        "merchant": "テスト店",
        "currency": "JPY",
        "total": 100,
        "subtotal": 90,
        "taxes": [],
        "line_items": [
            {"description": "商品A", "qty": 1, "unit_price": 90, "total": 90},
        ],
        "points_used": None,
    }
    trace: list[dict] = []

    postprocess_receipt(
        extracted,
        "テスト店\n商品A\n¥90\n合計\n¥100",
        0.9,
        {},
        {},
        "test-model",
        mutation_trace=trace,
    )

    assert trace
    payment_events = [
        event for event in trace
        if event["stage"] == "payment_points_reconciliation"
    ]
    assert payment_events
    assert payment_events[0]["reads"]
    assert payment_events[0]["writes"]
    assert payment_events[0]["invariant"]
    changed_fields = {
        field
        for event in trace
        for field in event["changes"]
    }
    assert {"subtotal", "points_used"} <= changed_fields


def test_postprocess_receipt_phase_metadata_declares_field_ownership():
    from receipt_parser.pipeline_receipt import POSTPROCESS_PHASES

    phases = {phase["name"]: phase for phase in POSTPROCESS_PHASES}

    assert tuple(phases) == (
        "header_identity_repair",
        "financial_totals_repair",
        "cash_tender_reconciliation",
        "service_receipt_recovery",
        "body_total_layout_reconstruction",
        "initial_item_recovery",
        "gap_item_recovery",
        "low_value_bag_recovery",
        "adjacent_price_shift_reconciliation",
        "bag_amount_shift_reconciliation",
        "item_cleanup",
        "ocr_description_reconciliation",
        "quantity_detail_reconciliation",
        "single_rate_inclusive_tax_restoration",
        "tax_excluded_rate_block_restoration",
        "explicit_tax_amount_restoration",
        "external_tax_total_restoration",
        "tax_category_assignment",
        "payment_points_reconciliation",
        "structural_item_reconstruction",
        "final_consistency_pass",
    )
    assert "location" in phases["header_identity_repair"]["writes"]
    assert "date" in phases["header_identity_repair"]["writes"]
    assert "location" in phases["body_total_layout_reconstruction"]["writes"]
    assert "subtotal" in phases["body_total_layout_reconstruction"]["writes"]
    assert "line_items" in phases["initial_item_recovery"]["writes"]
    assert "line_items" in phases["body_total_layout_reconstruction"]["writes"]
    assert "line_items" in phases["gap_item_recovery"]["writes"]
    assert "line_items" in phases["low_value_bag_recovery"]["writes"]
    assert "line_items" in phases["adjacent_price_shift_reconciliation"]["writes"]
    assert "line_items" in phases["bag_amount_shift_reconciliation"]["writes"]
    assert "line_items" in phases["item_cleanup"]["writes"]
    assert "line_items" in phases["ocr_description_reconciliation"]["writes"]
    assert "line_items" in phases["service_receipt_recovery"]["writes"]
    assert "line_items" in phases["quantity_detail_reconciliation"]["writes"]
    assert "line_items" in phases["single_rate_inclusive_tax_restoration"]["writes"]
    assert "subtotal" in phases["single_rate_inclusive_tax_restoration"]["writes"]
    assert "taxes" in phases["single_rate_inclusive_tax_restoration"]["writes"]
    assert "taxes" in phases["tax_excluded_rate_block_restoration"]["writes"]
    assert "taxes" in phases["explicit_tax_amount_restoration"]["writes"]
    assert "total" in phases["external_tax_total_restoration"]["writes"]
    assert "amount_paid" in phases["external_tax_total_restoration"]["writes"]
    assert "line_items" in phases["structural_item_reconstruction"]["writes"]
    assert "taxes" in phases["tax_category_assignment"]["writes"]
    assert "amount_paid" in phases["cash_tender_reconciliation"]["writes"]
    assert "total" in phases["financial_totals_repair"]["writes"]
    assert "amount_paid" in phases["payment_points_reconciliation"]["reads"]
    assert "payment_method" in phases["payment_points_reconciliation"]["writes"]
    assert "total" in phases["final_consistency_pass"]["writes"]
    expected_owners = {
        "line_items": {
            "adjacent_price_shift_reconciliation",
            "bag_amount_shift_reconciliation",
            "body_total_layout_reconstruction",
            "gap_item_recovery",
            "initial_item_recovery",
            "low_value_bag_recovery",
            "item_cleanup",
            "ocr_description_reconciliation",
            "quantity_detail_reconciliation",
            "service_receipt_recovery",
            "single_rate_inclusive_tax_restoration",
            "tax_category_assignment",
            "structural_item_reconstruction",
            "final_consistency_pass",
        },
        "taxes": {
            "financial_totals_repair",
            "service_receipt_recovery",
            "single_rate_inclusive_tax_restoration",
            "tax_excluded_rate_block_restoration",
            "explicit_tax_amount_restoration",
            "tax_category_assignment",
            "structural_item_reconstruction",
            "final_consistency_pass",
            "body_total_layout_reconstruction",
        },
        "subtotal": {
            "body_total_layout_reconstruction",
            "external_tax_total_restoration",
            "financial_totals_repair",
            "payment_points_reconciliation",
            "service_receipt_recovery",
            "single_rate_inclusive_tax_restoration",
            "structural_item_reconstruction",
            "final_consistency_pass",
        },
        "total": {
            "external_tax_total_restoration",
            "financial_totals_repair",
            "cash_tender_reconciliation",
            "structural_item_reconstruction",
            "final_consistency_pass",
        },
        "amount_paid": {
            "external_tax_total_restoration",
            "financial_totals_repair",
            "cash_tender_reconciliation",
            "payment_points_reconciliation",
            "structural_item_reconstruction",
            "final_consistency_pass",
        },
        "payment_method": {
            "financial_totals_repair",
            "cash_tender_reconciliation",
            "payment_points_reconciliation",
            "service_receipt_recovery",
        },
        "merchant": {"header_identity_repair"},
        "location": {
            "body_total_layout_reconstruction",
            "header_identity_repair",
        },
        "date": {"header_identity_repair"},
    }
    actual_owners = {
        field: {
            phase["name"]
            for phase in POSTPROCESS_PHASES
            if field in phase["writes"]
        }
        for field in expected_owners
    }
    assert actual_owners == expected_owners
    assert all(phase["invariant"] for phase in POSTPROCESS_PHASES)


def test_postprocess_receipt_is_idempotent_for_balanced_simple_receipt():
    from receipt_parser.pipeline_receipt import (
        _snapshot_receipt_mutation_fields,
        postprocess_receipt,
    )

    extracted = {
        "document_type": "receipt",
        "merchant": "テスト店",
        "currency": "JPY",
        "total": 100,
        "subtotal": 100,
        "taxes": [],
        "line_items": [
            {"description": "商品A", "qty": 1, "unit_price": 100, "total": 100},
        ],
        "points_used": 0,
    }
    text = "テスト店\n商品A\n¥100\n小計\n¥100\n合計\n¥100"

    postprocess_receipt(extracted, text, 0.9, {}, {}, "test-model")
    once = _snapshot_receipt_mutation_fields(extracted)
    postprocess_receipt(extracted, text, 0.9, {}, {}, "test-model")
    twice = _snapshot_receipt_mutation_fields(extracted)

    assert twice == once


def test_gap_item_recovery_skips_tax_exclusive_balanced_receipt():
    from receipt_parser.pipeline_receipt import (
        _recover_missing_items_from_gap,
        _recover_repeated_item_from_gap,
    )

    extracted = {
        "total": 5362,
        "taxes": [
            {"rate": "8%", "label": "外税", "amount": 386},
            {"rate": "10%", "label": "外税", "amount": 13},
        ],
        "line_items": [
            {"description": "既存商品", "qty": 1, "unit_price": 4565, "total": 4565},
            {"description": "魚屋の焼さけほぐし弁当", "qty": 1, "unit_price": 398, "total": 398},
        ],
    }
    unified_text = "\n".join(
        [
            "既存商品",
            "4565",
            "魚屋の焼さけほぐし弁当",
            "398",
            "魚屋の焼さけほぐし弁当",
            "398",
            "小計",
            "¥4,963",
            "外税8%",
            "¥386",
            "外税10%",
            "¥13",
            "合計",
            "¥5,362",
        ]
    )

    _recover_missing_items_from_gap(extracted, unified_text)
    _recover_repeated_item_from_gap(extracted, unified_text)

    assert len(extracted["line_items"]) == 2
    assert sum(item["total"] for item in extracted["line_items"]) == 4963


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


def test_process_ocr_text_debug_exposes_receipt_mutation_trace():
    from receipt_parser.pipeline import process_ocr_text

    ocr_text = "\n".join([
        "テスト店",
        "2025-01-15",
        "テスト商品",
        "¥1,000",
        "外税",
        "¥100",
        "合計",
        "¥1,100",
    ])
    extraction = {
        "document_type": "receipt",
        "merchant": "テスト店",
        "date": "2025-01-15",
        "total": 1100,
        "subtotal": None,
        "taxes": [{"rate": "10%", "label": "外税", "amount": 100}],
        "line_items": [
            {"description": "テスト商品", "qty": 1, "unit_price": 1000, "total": 1000},
        ],
        "currency": "JPY",
        "_confidence": {"overall": "high"},
    }

    with patch("receipt_parser.pipeline.extract_with_verification") as mock_extract:
        mock_extract.return_value = (
            extraction,
            [{"pass": 1, "extraction": extraction, "warnings": []}],
        )
        normal = process_ocr_text(ocr_text, debug=False)
        debug = process_ocr_text(ocr_text, debug=True)

    assert "_receipt_mutation_trace" not in normal
    trace = debug["_receipt_mutation_trace"]
    assert any(event["stage"] == "payment_points_reconciliation" for event in trace)
    assert all("changes" in event for event in trace)
    assert any("invariant" in event for event in trace)


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


def test_dense_item_rows_preserves_inline_small_bag_price():
    from receipt_parser.pipeline_receipt import _replace_dense_item_rows_when_balanced

    extracted = {
        "merchant": "マックスバリュ",
        "subtotal": 2471,
        "line_items": [{"description": "dummy", "qty": 1, "unit_price": 2471, "total": 2471}],
    }
    ocr_text = "\n".join([
        "AEON",
        "2026/6/1 (月)",
        "ミドリ牛乳",
        "268*",
        "食品ポリ袋L (バイオマス30 3",
        "アクエリアス",
        "98*",
        "バナナ",
        "138",
        "おにぎり",
        "128*",
        "小計",
        "¥635",
        "外税8%対象額",
        "¥632",
        "外税10%対象額",
        "¥3",
    ])
    extracted["subtotal"] = 635

    _replace_dense_item_rows_when_balanced(extracted, ocr_text)

    assert [item["total"] for item in extracted["line_items"]] == [268.0, 3.0, 98.0, 138.0, 128.0]
    assert extracted["line_items"][1]["description"] == "食品ポリ袋L (バイオマス30"


def test_dense_item_rows_stops_before_payment_and_tax_summary():
    from receipt_parser.pipeline_receipt import _replace_dense_item_rows_when_balanced

    extracted = {
        "merchant": "マックスバリュ",
        "subtotal": 1678,
        "line_items": [{"description": "dummy", "qty": 1, "unit_price": 1678, "total": 1678}],
    }
    ocr_text = "\n".join([
        "AEON",
        "2026/6/2 (火)",
        "ブロッコリー",
        "198",
        "食品ポリ袋L (バイオマス30 3",
        "アスパラガス",
        "450",
        "(3個 X 単150)",
        "銀さけ切身",
        "517",
        "割引",
        "10%",
        "-52",
        "銀さけ切身",
        "510*",
        "10%",
        "-51",
        "小計",
        "¥1,678",
        "外税8%",
        "¥238",
        "合計",
        "クレジット",
        "¥3,227",
    ])

    _replace_dense_item_rows_when_balanced(extracted, ocr_text)

    descriptions = [item["description"] for item in extracted["line_items"]]
    assert "クレジット" not in descriptions
    assert [item["total"] for item in extracted["line_items"]] == [198.0, 3.0, 450.0, 517.0, 510.0]


def test_numeric_marker_description_cleanup_is_merchant_free():
    from receipt_parser.pipeline_receipt import _drop_numeric_marker_description_rows

    extracted = {
        "line_items": [
            {"description": "商品A", "qty": 1, "unit_price": 198, "total": 198},
            {"description": "148※", "qty": 1, "unit_price": 148, "total": 148},
            {"description": "商品B", "qty": 1, "unit_price": 98, "total": 98},
            {"description": "200%", "qty": 1, "unit_price": 200, "total": 200},
        ]
    }

    _drop_numeric_marker_description_rows(extracted, "GENERIC MARKET\n商品A\n198\n商品B\n98")

    assert [item["description"] for item in extracted["line_items"]] == ["商品A", "商品B"]


def test_adjacent_ocr_price_shift_repairs_when_subtotal_balances_without_merchant_gate():
    from receipt_parser.pipeline_receipt import _fix_adjacent_ocr_price_shift_when_balanced

    extracted = {
        "subtotal": 1000,
        "line_items": [
            {"description": "商品甲", "qty": 1, "unit_price": 180, "total": 180, "discount": 0},
            {"description": "商品乙", "qty": 1, "unit_price": 250, "total": 250, "discount": 0},
            {"description": "商品丙", "qty": 1, "unit_price": 400, "total": 400, "discount": 0},
        ],
    }
    ocr_text = "\n".join([
        "領収証",
        "商品甲 420",
        "商品乙",
        "180*",
        "商品丙 400",
        "小計",
        "¥1,000",
    ])

    _fix_adjacent_ocr_price_shift_when_balanced(extracted, ocr_text)

    assert [item["total"] for item in extracted["line_items"]] == [420.0, 180.0, 400]
    assert sum(item["total"] for item in extracted["line_items"]) == 1000


def test_adjacent_ocr_price_shift_does_not_reprice_bag_rows():
    from receipt_parser.pipeline_receipt import _fix_adjacent_ocr_price_shift_when_balanced

    extracted = {
        "subtotal": 301,
        "total": 301,
        "line_items": [
            {"description": "有料レジ袋HI 3L HI (3", "qty": 1, "unit_price": 5, "total": 5, "tax_category": "10%"},
            {"description": "次の商品", "qty": 1, "unit_price": 298, "total": 298, "tax_category": "10%"},
        ],
    }
    ocr_text = "\n".join([
        "有料レジ袋HI 3L HI (3",
        "次の商品",
        "298",
        "小計",
        "¥301",
    ])

    _fix_adjacent_ocr_price_shift_when_balanced(extracted, ocr_text)

    assert extracted["line_items"][0]["unit_price"] == 5
    assert extracted["line_items"][0]["total"] == 5


def test_final_receipt_output_repairs_apply_adjacent_ocr_price_shift_before_duplicate_recovery():
    from receipt_parser.pipeline import _apply_final_receipt_output_repairs

    result = {
        "document_type": "receipt",
        "subtotal": 1000,
        "total": 1000,
        "line_items": [
            {"description": "商品甲", "qty": 1, "unit_price": 180, "total": 180, "discount": 0},
            {"description": "商品乙", "qty": 1, "unit_price": 250, "total": 250, "discount": 0},
            {"description": "商品丙", "qty": 1, "unit_price": 400, "total": 400, "discount": 0},
            {"description": "商品丙", "qty": 1, "unit_price": 400, "total": 400, "discount": 0},
        ],
    }
    ocr_text = "\n".join([
        "領収証",
        "商品甲 420",
        "商品乙",
        "180*",
        "商品丙 400",
        "小計",
        "¥1,000",
        "お買上商品数:3",
    ])

    _apply_final_receipt_output_repairs(result, ocr_text)

    assert [item["total"] for item in result["line_items"]] == [420.0, 180.0, 400]
    assert len(result["line_items"]) == 3


def test_missing_item_recovery_allows_low_value_bag_rows():
    from receipt_parser.pipeline_receipt import _recover_missing_items_from_gap

    extracted = {
        "total": 2471,
        "subtotal": 2289,
        "taxes": [{"rate": "8%", "amount": 182, "label": "外税"}],
        "line_items": [
            {"description": "ミドリ牛乳", "qty": 1, "unit_price": 268, "total": 268, "tax_category": "8%"},
            {"description": "たまご三昧", "qty": 1, "unit_price": 278, "total": 278, "tax_category": "8%"},
            {"description": "アクエリアス", "qty": 1, "unit_price": 98, "total": 98, "tax_category": "8%"},
            {"description": "ガーナブラック", "qty": 2, "unit_price": 198, "total": 396, "tax_category": "8%"},
            {"description": "プロテインドリンクカフ", "qty": 1, "unit_price": 248, "total": 248, "tax_category": "8%"},
            {"description": "ぷるんぷるんQooぶど", "qty": 1, "unit_price": 148, "total": 148, "tax_category": "8%"},
            {"description": "その他", "qty": 1, "unit_price": 850, "total": 850, "tax_category": "8%"},
        ],
    }
    ocr_text = "\n".join([
        "ミドリ牛乳",
        "268*",
        "たまご三昧",
        "278*",
        "食品ポリ袋L (バイオマス30 3",
        "アクエリアス",
        "98*",
        "小計",
        "¥2,289",
    ])

    _recover_missing_items_from_gap(extracted, ocr_text)

    recovered = extracted["line_items"]
    assert any(item["description"] == "食品ポリ袋L (バイオマス30" and item["total"] == 3.0 for item in recovered)


def test_missing_item_recovery_uses_orphan_rows_and_subtotal_gap_for_dense_streams():
    from receipt_parser.pipeline_receipt import _recover_missing_items_from_gap

    extracted = {
        "total": 1780,
        "subtotal": 1697,
        "taxes": [
            {"rate": "8%", "amount": 83, "label": "外税"},
            {"rate": "10%", "amount": 0, "label": "外税"},
        ],
        "line_items": [
            {"description": "非課税用品", "qty": 1, "unit_price": 652, "total": 652, "tax_category": "0%"},
            {"description": "レジ袋", "qty": 1, "unit_price": 3, "total": 3, "tax_category": "10%"},
            {"description": "商品丙", "qty": 1, "unit_price": 478, "total": 478, "tax_category": "8%"},
            {"description": "商品丁", "qty": 1, "unit_price": 128, "total": 128, "tax_category": "8%"},
        ],
    }
    ocr_text = "\n".join([
        "2026/4/14",
        "12:00",
        "商品甲",
        "非課税用品 652非",
        "198*",
        "商品乙",
        "2",
        "レジ袋 3除",
        "商品丙",
        "478*",
        "商品丁",
        "128*",
        "小計",
        "¥1,697",
        "外税8%対象額",
        "¥1,042",
        "外税8%",
        "¥83",
        "外税10%対象額",
        "¥3",
        "外税10%",
        "¥0",
        "非課税対象額",
        "¥652",
        "合計",
        "¥1,780",
    ])

    _recover_missing_items_from_gap(extracted, ocr_text)

    assert [item["description"] for item in extracted["line_items"]] == [
        "商品甲",
        "非課税用品",
        "商品乙",
        "レジ袋",
        "商品丙",
        "商品丁",
    ]
    assert [item["total"] for item in extracted["line_items"]] == [
        198.0,
        652,
        238.0,
        3,
        478,
        128,
    ]
    assert sum(item["total"] for item in extracted["line_items"]) == 1697.0
    assert sum(
        item["total"]
        for item in extracted["line_items"]
        if item["tax_category"] == "8%"
    ) == 1042.0


def test_final_receipt_output_repairs_run_missing_item_gap_recovery_after_row_repairs():
    from receipt_parser.pipeline import _apply_final_receipt_output_repairs

    result = {
        "document_type": "receipt",
        "total": 1780,
        "subtotal": 1697,
        "taxes": [
            {"rate": "8%", "amount": 83, "label": "外税"},
            {"rate": "10%", "amount": 0, "label": "外税"},
        ],
        "line_items": [
            {"description": "商品甲 非課税用品", "qty": 1, "unit_price": 652, "total": 652, "tax_category": "0%"},
            {"description": "レジ袋", "qty": 1, "unit_price": 0, "total": 3, "tax_category": "10%"},
            {"description": "商品丙", "qty": 1, "unit_price": 0, "total": 478, "tax_category": "8%"},
            {"description": "商品丁", "qty": 1, "unit_price": 0, "total": 128, "tax_category": "8%"},
        ],
    }
    ocr_text = "\n".join([
        "2026/4/14",
        "12:00",
        "商品甲",
        "非課税用品 652非",
        "198*",
        "商品乙",
        "2",
        "レジ袋 3除",
        "商品丙",
        "478*",
        "商品丁",
        "128*",
        "小計",
        "¥1,697",
        "外税8%対象額",
        "¥1,042",
        "外税8%",
        "¥83",
        "外税10%対象額",
        "¥3",
        "外税10%",
        "¥0",
        "非課税対象額",
        "¥652",
        "合計",
        "¥1,780",
    ])

    _apply_final_receipt_output_repairs(result, ocr_text)

    assert [item["description"] for item in result["line_items"][:4]] == [
        "商品甲",
        "非課税用品",
        "商品乙",
        "レジ袋",
    ]
    assert sum(item["total"] for item in result["line_items"]) == 1697.0
    assert sorted(item["unit_price"] for item in result["line_items"]) == [
        3.0,
        128.0,
        198.0,
        238.0,
        478.0,
        652.0,
    ]


def test_non_product_cleanup_drops_credit_payment_item():
    from receipt_parser.pipeline_receipt import _drop_non_product_line_items

    extracted = {
        "total": 3227,
        "line_items": [
            {"description": "ブロッコリー", "qty": 1, "unit_price": 198, "total": 198},
            {"description": "クレジット", "qty": 1, "unit_price": 238, "total": 238},
        ],
    }

    _drop_non_product_line_items(extracted, "合計\nクレジット\n¥3,227")

    assert [item["description"] for item in extracted["line_items"]] == ["ブロッコリー"]


def test_missing_item_recovery_rejects_payment_label_description():
    from receipt_parser.pipeline_receipt import _recover_missing_items_from_gap

    extracted = {
        "total": 3227,
        "subtotal": 2989,
        "taxes": [],
        "line_items": [
            {"description": "ブロッコリー", "qty": 1, "unit_price": 198, "total": 198, "tax_category": "8%"},
            {"description": "その他", "qty": 1, "unit_price": 2553, "total": 2553, "tax_category": "8%"},
        ],
    }
    ocr_text = "\n".join([
        "ブロッコリー",
        "198",
        "小計",
        "¥2,989",
        "外税8%",
        "¥238",
        "合計",
        "クレジット",
        "¥3,227",
    ])

    _recover_missing_items_from_gap(extracted, ocr_text)

    assert all(item["description"] != "クレジット" for item in extracted["line_items"])


def test_low_value_bag_overage_repair_and_numeric_description_context():
    from receipt_parser.pipeline_receipt import (
        _fix_numeric_desc_from_ocr_price_context,
        _replace_overage_item_with_low_value_bag,
    )

    extracted = {
        "subtotal": 2289,
        "total": 2471,
        "line_items": [
            {"description": "ミドリ牛乳", "qty": 1, "unit_price": 268, "total": 268, "tax_category": "8%"},
            {"description": "手巻きおにぎり (紅鮭)", "qty": 1, "unit_price": 128, "total": 128, "tax_category": "8%"},
            {"description": "ぷるんぷるんQooぶど", "qty": 1, "unit_price": 128, "total": 128, "tax_category": "10%"},
            {"description": "148", "qty": 1, "unit_price": 148, "total": 148, "tax_category": "8%"},
            {"description": "その他", "qty": 1, "unit_price": 1742, "total": 1742, "tax_category": "8%"},
        ],
    }
    ocr_text = "\n".join([
        "ミドリ牛乳",
        "268X",
        "食品ポリ袋L (バイオマス30 3",
        "手巻きおにぎり (紅鮭)",
        "ぷるんぷるんQooぶど",
        "128",
        "148",
        "小計",
        "¥2,289",
    ])

    _replace_overage_item_with_low_value_bag(extracted, ocr_text)
    _fix_numeric_desc_from_ocr_price_context(extracted, ocr_text)

    items = extracted["line_items"]
    assert any(item["description"] == "食品ポリ袋L (バイオマス30" and item["total"] == 3.0 for item in items)
    assert any(item["description"] == "ぷるんぷるんQooぶど" and item["total"] == 148 for item in items)
    assert sum(item["total"] for item in items) == 2289


def test_appends_missing_low_value_bag_when_it_closes_gap():
    from receipt_parser.pipeline_receipt import _append_missing_low_value_bag_from_gap

    extracted = {
        "subtotal": 2289,
        "total": 2471,
        "line_items": [
            {"description": "ミドリ牛乳", "qty": 1, "unit_price": 268, "total": 268, "tax_category": "8%"},
            {"description": "その他", "qty": 1, "unit_price": 2018, "total": 2018, "tax_category": "8%"},
        ],
    }
    ocr_text = "ミドリ牛乳\n268X\n食品ポリ袋L (バイオマス30 3\n小計\n¥2,289"

    _append_missing_low_value_bag_from_gap(extracted, ocr_text)

    assert extracted["line_items"][-1]["description"] == "食品ポリ袋L (バイオマス30"
    assert extracted["line_items"][-1]["total"] == 3.0


def test_bag_price_repair_skips_time_before_small_price():
    from receipt_parser.pipeline_receipt import _fix_bag_item_prices_from_ocr

    extracted = {
        "line_items": [
            {"description": "有料レジ袋", "qty": 1, "unit_price": 47, "total": 47},
            {"description": "きゅうり", "qty": 1, "unit_price": 58, "total": 58},
        ]
    }
    ocr_text = "\n".join([
        "有料レジ袋し",
        "きゅうり",
        "9:47",
        ": 003493750",
        "5",
        "58%",
        "小計",
    ])

    _fix_bag_item_prices_from_ocr(extracted, ocr_text)

    bag = extracted["line_items"][0]
    assert bag["unit_price"] == 5.0
    assert bag["total"] == 5.0


def test_bag_price_repair_prefers_standalone_price_after_code_row():
    from receipt_parser.pipeline_receipt import _fix_bag_item_prices_from_ocr

    extracted = {
        "line_items": [
            {"description": "有料レジ袋HI 3L HI", "qty": 1, "unit_price": 5, "total": 5},
        ]
    }
    ocr_text = "\n".join([
        "07 有料レジ袋HI 3L HI (3",
        "21843617",
        "20 次の商品",
        "23664616",
        "01 型番つき商品 LM-30",
        "20257736",
        "5",
        "298",
        "小計",
    ])

    _fix_bag_item_prices_from_ocr(extracted, ocr_text)

    bag = extracted["line_items"][0]
    assert bag["unit_price"] == 5.0
    assert bag["total"] == 5.0


def test_split_bag_price_uses_nearby_single_digit_without_merchant_gate():
    from receipt_parser.pipeline_receipt import _fix_split_bag_price_from_nearby_single_digit

    extracted = {
        "line_items": [
            {"description": "有料レジ袋HI", "qty": 1, "unit_price": 3, "total": 3, "tax_category": "8%"},
        ],
    }
    ocr_text = "\n".join([
        "SHOP",
        "07 有料レジ袋HI 3L HI (3",
        "商品A",
        "5",
        "小計",
    ])

    _fix_split_bag_price_from_nearby_single_digit(extracted, ocr_text)

    assert extracted["line_items"][0] == {
        "description": "有料レジ袋HI",
        "qty": 1.0,
        "unit_price": 5.0,
        "total": 5.0,
        "tax_category": "10%",
    }


def test_small_bag_description_uses_visible_ocr_entry_without_merchant_gate():
    from receipt_parser.pipeline_receipt import _fix_small_bag_description_from_ocr_entry

    extracted = {
        "line_items": [
            {"description": "不明商品", "qty": 1, "unit_price": 4, "total": 4, "tax_category": "8%"},
        ],
    }
    ocr_text = "\n".join([
        "SHOP",
        "内レジ袋 LL",
        "4",
        "小計",
    ])

    _fix_small_bag_description_from_ocr_entry(extracted, ocr_text)

    assert extracted["line_items"][0] == {
        "description": "レジ袋 LL",
        "qty": 1.0,
        "unit_price": 4.0,
        "total": 4.0,
        "tax_category": "10%",
    }


def test_missing_bag_recovery_uses_visible_bag_row_when_subtotal_balances():
    from receipt_parser.pipeline_receipt import _recover_missing_bag_items_from_ocr

    extracted = {
        "subtotal": 2289,
        "total": 2471,
        "taxes": [{"rate": "8%", "label": "外税", "amount": 182}],
        "line_items": [
            {"description": "ミドリ牛乳", "qty": 1, "unit_price": 268, "total": 268, "tax_category": "8%"},
            {"description": "その他", "qty": 1, "unit_price": 2018, "total": 2018, "tax_category": "8%"},
        ],
    }
    ocr_text = "\n".join([
        "ミドリ牛乳",
        "268X",
        "食品ポリ袋L (バイオマス30 3",
        "小計",
        "¥2,289",
        "お買上商品数:3",
    ])

    _recover_missing_bag_items_from_ocr(extracted, ocr_text)

    assert any(item["description"] == "食品ポリ袋L (バイオマス30" and item["total"] == 3.0 for item in extracted["line_items"])


def test_qty_detail_applies_to_inline_owner_not_previous_same_unit_item():
    from receipt_parser.pipeline_receipt import _fix_qty_totals_from_ocr_unit_lines

    extracted = {
        "line_items": [
            {"description": "シャウエッセン", "qty": 2, "unit_price": 398, "total": 796},
            {"description": "よつ葉バター(食塩不使用)", "qty": 2, "unit_price": 398, "total": 796},
        ]
    }
    ocr_text = "\n".join([
        "シャウエッセン",
        "398**",
        "よつ葉バター(食塩不使用) 796",
        "(2個X 単398)",
        "小計",
    ])

    _fix_qty_totals_from_ocr_unit_lines(extracted, ocr_text)

    assert extracted["line_items"][0]["qty"] == 1.0
    assert extracted["line_items"][0]["total"] == 398.0
    assert extracted["line_items"][1]["qty"] == 2.0
    assert extracted["line_items"][1]["total"] == 796.0


def test_dense_sequence_rows_assigns_queued_names_and_prices():
    from receipt_parser.pipeline_receipt import _replace_dense_sequence_rows_when_balanced

    extracted = {
        "merchant": "マックスバリュ",
        "subtotal": 1289,
        "taxes": [],
        "line_items": [{"description": "dummy", "qty": 1, "unit_price": 1289, "total": 1289}],
    }
    ocr_text = "\n".join([
        "AEON",
        "レジ 0173 2026/6/1 (月)",
        "11:25",
        "ミドリ牛乳",
        "268X",
        "食品ポリ袋L (バイオマス30 3",
        "アクエリアス",
        "98*",
        "ガーナブラック",
        "396*",
        "(2個 X 単198)",
        "プロテインドリンクカフ",
        "手巻きおにぎり (紅鮭)",
        "ぷるんぷるんQooぶど",
        "248※",
        "128",
        "148",
        "小計",
        "¥1,289",
        "お買上商品数:7",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, ocr_text)

    assert [item["description"] for item in extracted["line_items"]] == [
        "ミドリ牛乳",
        "食品ポリ袋L (バイオマス30",
        "アクエリアス",
        "ガーナブラック",
        "プロテインドリンクカフ",
        "手巻きおにぎり (紅鮭)",
        "ぷるんぷるんQooぶど",
    ]
    assert [item["total"] for item in extracted["line_items"]] == [268.0, 3.0, 98.0, 396.0, 248.0, 128.0, 148.0]
    assert extracted["line_items"][3]["qty"] == 2.0
    assert extracted["line_items"][3]["unit_price"] == 198.0


def test_dense_sequence_rows_applies_pending_quantity_detail_before_price():
    from receipt_parser.pipeline_receipt import _replace_dense_sequence_rows_when_balanced

    extracted = {
        "subtotal": 640,
        "taxes": [],
        "line_items": [{"description": "dummy", "qty": 1, "unit_price": 640, "total": 640}],
    }
    ocr_text = "\n".join([
        "TEST MARKET",
        "2026/6/1(月)",
        "商品ア",
        "(2個 X 単128)",
        "256*",
        "商品イ",
        "128*",
        "商品ウ",
        "64*",
        "商品エ",
        "96*",
        "商品オ",
        "96*",
        "小計",
        "¥640",
        "お買上商品数:6",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, ocr_text)

    assert [item["description"] for item in extracted["line_items"]] == [
        "商品ア",
        "商品イ",
        "商品ウ",
        "商品エ",
        "商品オ",
    ]
    assert extracted["line_items"][0]["qty"] == 2.0
    assert extracted["line_items"][0]["unit_price"] == 128.0
    assert extracted["line_items"][0]["total"] == 256.0


def test_dense_sequence_rows_balances_split_discount_controls():
    from receipt_parser.pipeline_receipt import _replace_dense_sequence_rows_when_balanced

    extracted = {
        "merchant": "テストマート",
        "subtotal": 2498,
        "taxes": [],
        "line_items": [{"description": "dummy", "qty": 1, "unit_price": 2498, "total": 2498}],
    }
    ocr_text = "\n".join([
        "PARENTCO",
        "テストマート中央店",
        "2026/6/6(土)",
        "商品ア 200*",
        "割引",
        "50%",
        "-100",
        "商品イ 600*",
        "割引",
        "商品ウ 1000*",
        "割引",
        "50%",
        "-500",
        "商品エ 1200*",
        "割引",
        "50%",
        "商品オ",
        "商品カ",
        "50%",
        "-300",
        "-600",
        "400*",
        "500*",
        "商品キ",
        "196",
        "(2個 X 単98)",
        "割引",
        "50%",
        "-98",
        "小計",
        "¥2,498",
        "お買上商品数:7",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, ocr_text)

    assert [item["description"] for item in extracted["line_items"]] == [
        "商品ア",
        "商品イ",
        "商品ウ",
        "商品エ",
        "商品オ",
        "商品カ",
        "商品キ",
    ]
    assert [item["total"] for item in extracted["line_items"]] == [
        100.0,
        300.0,
        500.0,
        600.0,
        400.0,
        500.0,
        98.0,
    ]
    assert extracted["line_items"][6]["qty"] == 2.0
    assert extracted["line_items"][6]["unit_price"] == 98.0
    assert extracted["line_items"][6]["discount"] == 98.0
    assert extracted["line_items"][6]["discount_rate"] == "50%"


def test_dense_sequence_rows_accepts_quantity_count_and_marker_summary_discount():
    from receipt_parser.pipeline_receipt import _replace_dense_sequence_rows_when_balanced

    extracted = {
        "subtotal": 3402,
        "taxes": [{"rate": "8%", "label": "外税", "amount": 272}],
        "line_items": [{"description": "dummy", "qty": 1, "unit_price": 3402, "total": 3402}],
    }
    ocr_text = "\n".join([
        "テストマート",
        "2026/4/12(日)",
        "商品ア",
        "268",
        "商品イ 338",
        "商品ウ",
        "375",
        "割引",
        "10%",
        "-38",
        "商品ウ",
        "526",
        "割引",
        "10%",
        "-53",
        "商品エ 138",
        "商品オ",
        "158",
        "商品カ",
        "998*",
        "商品キ 396※",
        "(2個 X 単198)",
        "商品ク",
        "商品ケ",
        "(3個 X 単68)",
        "98*",
        "204 A",
        "まとめ値引",
        "-6",
        "A: 3個 ¥198 の商品です",
        "小計",
        "¥3,402",
        "外税8%",
        "¥272",
        "合計",
        "¥3,674",
        "お買上商品数:13",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, ocr_text)

    assert [item["total"] for item in extracted["line_items"]] == [
        268.0,
        338.0,
        337.0,
        473.0,
        138.0,
        158.0,
        998.0,
        396.0,
        98.0,
        198.0,
    ]
    assert extracted["line_items"][2]["unit_price"] == 375.0
    assert extracted["line_items"][2]["discount"] == 38.0
    assert extracted["line_items"][7]["qty"] == 2.0
    assert extracted["line_items"][7]["unit_price"] == 198.0
    assert extracted["line_items"][9]["qty"] == 3.0
    assert extracted["line_items"][9]["unit_price"] == 68.0
    assert extracted["line_items"][9]["discount"] == 6.0


def test_dense_sequence_rows_applies_later_matching_marker_summary():
    from receipt_parser.pipeline_receipt import _replace_dense_sequence_rows_when_balanced

    extracted = {
        "subtotal": 1598,
        "taxes": [{"rate": "8%", "label": "外税", "amount": 127}],
        "line_items": [{"description": "dummy", "qty": 1, "unit_price": 1598, "total": 1598}],
    }
    ocr_text = "\n".join([
        "テスト店",
        "2026/4/30(木)",
        "商品ア 100*",
        "商品イ 200*",
        "商品ウ 300*",
        "商品エ 400*",
        "商品オ 500*",
        "商品カ",
        "(2個 X 単58)",
        "116 C",
        "まとめ値引",
        "A: 2個 ¥780 の商品です",
        "C: 2個 ¥98 の商品です",
        "小計",
        "¥1,598",
        "外税8%",
        "¥127",
        "合計",
        "¥1,725",
        "お買上商品数:7",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, ocr_text)

    assert [item["total"] for item in extracted["line_items"]] == [
        100.0,
        200.0,
        300.0,
        400.0,
        500.0,
        98.0,
    ]
    assert extracted["line_items"][5]["qty"] == 2.0
    assert extracted["line_items"][5]["unit_price"] == 58.0
    assert extracted["line_items"][5]["discount"] == 18.0


def test_dense_sequence_rows_flushes_pending_quantity_row_before_next_inline_item():
    from receipt_parser.pipeline_receipt import _replace_dense_sequence_rows_when_balanced

    extracted = {
        "subtotal": 2419,
        "taxes": [{"rate": "8%", "label": "外税", "amount": 193}],
        "line_items": [{"description": "dummy", "qty": 1, "unit_price": 2419, "total": 2419}],
    }
    ocr_text = (Path(__file__).parent.parent / ".data/ocr_cache/variants/receipt_96_v1.txt").read_text(
        encoding="utf-8"
    )

    _replace_dense_sequence_rows_when_balanced(extracted, ocr_text)

    items = extracted["line_items"]
    assert [item["total"] for item in items] == [
        6.0,
        78.0,
        398.0,
        96.0,
        128.0,
        98.0,
        248.0,
        205.0,
        98.0,
        98.0,
        393.0,
        387.0,
        88.0,
        98.0,
    ]
    assert sum(item["total"] for item in items) == 2419.0
    assert items[0]["qty"] == 2.0
    assert items[0]["unit_price"] == 3.0
    assert items[-1]["description"] == "たまねぎ バラ"
    assert items[-1]["qty"] == 2.0
    assert items[-1]["unit_price"] == 58.0
    assert items[-1]["discount"] == 18.0


def test_discounted_line_item_total_repairs_when_subtotal_balances():
    from receipt_parser.pipeline_receipt import _repair_discounted_line_item_totals_when_balanced

    extracted = {
        "subtotal": 1598,
        "line_items": [
            {"description": "商品ア", "qty": 1, "unit_price": 100, "total": 100, "discount": 0},
            {"description": "商品イ", "qty": 1, "unit_price": 200, "total": 200, "discount": 0},
            {"description": "商品ウ", "qty": 1, "unit_price": 300, "total": 300, "discount": 0},
            {"description": "商品エ", "qty": 1, "unit_price": 400, "total": 400, "discount": 0},
            {"description": "商品オ", "qty": 1, "unit_price": 500, "total": 500, "discount": 0},
            {"description": "商品カ", "qty": 2, "unit_price": 58, "total": 116, "discount": 18},
        ],
    }

    _repair_discounted_line_item_totals_when_balanced(extracted, "まとめ値引\n-18\n小計\n¥1,598")

    assert extracted["line_items"][5]["total"] == 98.0
    assert sum(item["total"] for item in extracted["line_items"]) == 1598.0


def test_campaign_discount_stream_reconstructs_shifted_price_block_when_balanced():
    from receipt_parser.pipeline_receipt import _replace_campaign_discount_stream_when_balanced

    extracted = {
        "subtotal": 766,
        "taxes": [
            {"rate": "8%", "label": "外税", "amount": 36},
            {"rate": "10%", "label": "外税", "amount": 5},
        ],
        "line_items": [{"description": "dummy", "qty": 1, "unit_price": 766, "total": 766}],
    }
    ocr_text = "\n".join([
        "テスト店",
        "2026/3/30(月) 22:19",
        "有料袋",
        "5除",
        "非課税サービス 300非",
        "商品ア",
        "196",
        "(2個 X 単98)",
        "会員様割引5%",
        "-10",
        "商品イ",
        "会員様割引5%",
        "商品ウ",
        "割引",
        "30%",
        "会員様割引5%",
        "商品エ",
        "100↓",
        "-5",
        "200*",
        "-60",
        "-10",
        "50*",
        "小計",
        "¥766",
        "外税8%対象額",
        "¥461",
        "外税8%",
        "¥36",
        "外税10%対象額",
        "¥5",
        "外税10%",
        "¥0",
        "お買上商品数:7",
    ])

    _replace_campaign_discount_stream_when_balanced(extracted, ocr_text)

    assert [item["description"] for item in extracted["line_items"]] == [
        "有料袋",
        "非課税サービス",
        "商品ア",
        "商品イ",
        "商品ウ",
        "商品エ",
    ]
    assert [item["total"] for item in extracted["line_items"]] == [
        5.0,
        300.0,
        186.0,
        95.0,
        130.0,
        50.0,
    ]
    assert extracted["line_items"][2]["qty"] == 2.0
    assert extracted["line_items"][2]["unit_price"] == 98.0
    assert extracted["line_items"][4]["discount"] == 70.0


def test_campaign_discount_stream_ignores_standalone_code_before_late_amount():
    from receipt_parser.pipeline_receipt import _replace_campaign_discount_stream_when_balanced

    extracted = {
        "subtotal": 2112,
        "taxes": [
            {"rate": "8%", "label": "外税", "amount": 42},
            {"rate": "10%", "label": "外税", "amount": 0},
        ],
        "line_items": [{"description": "dummy", "qty": 1, "unit_price": 2112, "total": 2112}],
    }
    ocr_text = "\n".join([
        "テスト店",
        "2026/3/30(月) 22:19",
        "有料袋",
        "0909143",
        "非課税サービス 652非",
        "5除",
        "商品ア",
        "998",
        "会員様割引5%",
        "-50",
        "商品イ",
        "100*",
        "会員様割引5%",
        "-5",
        "商品ウ",
        "割引",
        "30%",
        "会員様割引5%",
        "621*",
        "-187",
        "-22",
        "小計",
        "¥2,112",
        "外税8%対象額",
        "¥1,040",
        "外税8%",
        "¥83",
        "外税10%対象額",
        "¥5",
        "外税10%",
        "¥0",
        "お買上商品数:5",
    ])

    _replace_campaign_discount_stream_when_balanced(extracted, ocr_text)

    assert [item["description"] for item in extracted["line_items"]] == [
        "有料袋",
        "非課税サービス",
        "商品ア",
        "商品イ",
        "商品ウ",
    ]
    assert [item["total"] for item in extracted["line_items"]] == [
        5.0,
        652.0,
        948.0,
        95.0,
        412.0,
    ]
    assert extracted["line_items"][0]["tax_category"] == "10%"
    assert extracted["line_items"][1]["tax_category"] == "0%"


def test_campaign_discount_stream_uses_printed_subtotal_when_mutable_subtotal_drifted():
    from receipt_parser.pipeline_receipt import _replace_campaign_discount_stream_when_balanced

    extracted = {
        "subtotal": 1452,
        "taxes": [
            {"rate": "8%", "label": "外税", "amount": 83},
            {"rate": "10%", "label": "外税", "amount": 0},
        ],
        "line_items": [{"description": "dummy", "qty": 1, "unit_price": 1452, "total": 1452}],
    }
    ocr_text = "\n".join([
        "テスト店",
        "2026/3/30(月) 22:19",
        "有料袋",
        "0909143",
        "非課税サービス 652非",
        "5除",
        "商品ア",
        "998",
        "会員様割引5%",
        "-50",
        "商品イ",
        "100*",
        "会員様割引5%",
        "-5",
        "商品ウ",
        "割引",
        "30%",
        "会員様割引5%",
        "621*",
        "-187",
        "-22",
        "小計",
        "¥2,112",
        "外税8%対象額",
        "¥1,040",
        "外税8%",
        "¥83",
        "外税10%対象額",
        "¥5",
        "外税10%",
        "¥0",
        "お買上商品数:5",
    ])

    _replace_campaign_discount_stream_when_balanced(extracted, ocr_text)

    assert extracted["subtotal"] == 2112.0
    assert [item["total"] for item in extracted["line_items"]] == [
        5.0,
        652.0,
        948.0,
        95.0,
        412.0,
    ]


def test_prefixed_tax_marker_item_rows_recover_inline_and_queued_amounts():
    from receipt_parser.pipeline_receipt import _replace_prefixed_tax_marker_item_rows_when_balanced

    extracted = {
        "total": 8963,
        "subtotal": 8183,
        "taxes": [
            {"rate": "10%", "label": "内税", "amount": 632},
            {"rate": "8%", "label": "内税", "amount": 148},
        ],
        "line_items": [],
    }
    ocr_text = "\n".join([
        "領収証",
        "2026年04月24日 (金) 10時33分",
        "¥8,963-",
        "(10%対象",
        "¥6,962 内税",
        "¥632)",
        "(08%対象",
        "¥2,001 内税 ¥148)",
        "上記正に領収しました",
        "担当者 No828",
        "内クーリストアセダレーヌ",
        "¥880",
        "内フィーノ プレミアムタッチ 濃厚美容 ¥648",
        "内マシェリ ヘアオイル EX",
        "¥948",
        "内ルルルンオーラブライトマスクW",
        "¥770",
        "内YOLU カームナイト",
        "¥308",
        "内ベアディープモイスチャーリップ ハ ¥448",
        "内ビオレUVアクアリッチアクアプロテクトミス ¥980",
        "内じゃがりこサラダ 4連 ¥158",
        "内*ピュレグミプレミアム 白桃",
        "内*やかんの麦茶",
        "¥158",
        "¥79",
        "内キットカット オトナの ¥278",
        "内エクセラふわラテ まったり深 ¥498",
        "内スキットルズオリジナル ¥118",
        "内モッチュ シャインマスカット味 ¥128",
        "内*ぷっちょ袋 4種アソー ¥148",
        "内*ブラックサンダーミニ アーモンド&1 ¥238",
        "内*タネなしほしウメ ¥198",
        "23 X #199",
        "内アンラリージェEX サンプロテクター ¥1,980",
        "(10%対象 ¥6,962 内税 ¥632)",
        "*は軽減税率8%適用商品",
    ])

    _replace_prefixed_tax_marker_item_rows_when_balanced(extracted, ocr_text)

    assert len(extracted["line_items"]) == 18
    assert sum(item["total"] for item in extracted["line_items"]) == 8963.0
    assert sum(item["total"] for item in extracted["line_items"] if item["tax_category"] == "10%") == 6962.0
    assert sum(item["total"] for item in extracted["line_items"] if item["tax_category"] == "8%") == 2001.0
    ume = extracted["line_items"][-2]
    assert ume["qty"] == 2.0
    assert ume["unit_price"] == 99.0
    assert ume["total"] == 198.0


def test_dense_sequence_rows_splits_ambiguous_quantity_ocr_by_printed_amount():
    from receipt_parser.pipeline_receipt import _replace_dense_sequence_rows_when_balanced

    extracted = {
        "subtotal": 1606,
        "taxes": [{"rate": "8%", "label": "外税", "amount": 128}],
        "line_items": [{"description": "dummy", "qty": 1, "unit_price": 1606, "total": 1606}],
    }
    ocr_text = "\n".join([
        "テストマート",
        "2026/6/1(月)",
        "商品ア",
        "商品イ",
        "100*",
        "200*",
        "50*",
        "商品ウ",
        "商品エ",
        "(21 X 270)",
        "商品オ",
        "商品カ",
        "540*",
        "278*",
        "438*",
        "小計",
        "¥1,606",
        "外税8%対象額",
        "¥1,606",
        "外税8%",
        "¥128",
        "合計",
        "¥1,734",
        "お買上商品数:7",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, ocr_text)

    assert [item["description"] for item in extracted["line_items"]] == [
        "商品ア",
        "商品イ",
        "商品ウ",
        "商品エ",
        "商品オ",
        "商品カ",
    ]
    assert [item["total"] for item in extracted["line_items"]] == [
        100.0,
        200.0,
        50.0,
        540.0,
        278.0,
        438.0,
    ]
    assert extracted["line_items"][3]["qty"] == 2.0
    assert extracted["line_items"][3]["unit_price"] == 270.0


def test_dense_sequence_rows_merges_split_quantity_detail_lines():
    from receipt_parser.pipeline_receipt import _replace_dense_sequence_rows_when_balanced

    extracted = {
        "subtotal": 1392,
        "taxes": [{"rate": "8%", "label": "外税", "amount": 111}],
        "line_items": [{"description": "dummy", "qty": 1, "unit_price": 1392, "total": 1392}],
    }
    ocr_text = "\n".join([
        "テストマート",
        "2026/6/1(月)",
        "商品ア 100*",
        "商品イ",
        "796*",
        "(2個 X",
        "398)",
        "商品ウ",
        "商品エ",
        "商品オ",
        "200*",
        "148*",
        "148*",
        "小計",
        "¥1,392",
        "外税8%対象額",
        "¥1,392",
        "外税8%",
        "¥111",
        "合計",
        "¥1,503",
        "お買上商品数:6",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, ocr_text)

    assert [item["total"] for item in extracted["line_items"]] == [
        100.0,
        796.0,
        200.0,
        148.0,
        148.0,
    ]
    assert extracted["line_items"][1]["qty"] == 2.0
    assert extracted["line_items"][1]["unit_price"] == 398.0


def test_dense_sequence_rows_recovers_ocr_mangled_quantity_detail():
    from receipt_parser.pipeline_receipt import _replace_dense_sequence_rows_when_balanced

    extracted = {
        "subtotal": 1700,
        "taxes": [{"rate": "8%", "label": "外税", "amount": 136}],
        "line_items": [{"description": "dummy", "qty": 1, "unit_price": 1700, "total": 1700}],
    }
    ocr_text = "\n".join([
        "テストマート",
        "2026/6/1(月)",
        "商品ア 398*",
        "商品イ",
        "796%",
        "(218] X #1398)",
        "商品ウ",
        "商品エ",
        "商品オ",
        "200*",
        "148*",
        "158*",
        "小計",
        "¥1,700",
        "外税8%対象額",
        "¥1,700",
        "外税8%",
        "¥136",
        "合計",
        "¥1,836",
        "お買上商品数:6",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, ocr_text)

    assert [item["total"] for item in extracted["line_items"]] == [
        398.0,
        796.0,
        200.0,
        148.0,
        158.0,
    ]
    assert extracted["line_items"][1]["qty"] == 2.0
    assert extracted["line_items"][1]["unit_price"] == 398.0


def test_dense_sequence_rows_repairs_single_leading_digit_amount_by_subtotal():
    from receipt_parser.pipeline_receipt import _replace_dense_sequence_rows_when_balanced

    extracted = {
        "subtotal": 2050,
        "taxes": [
            {"rate": "8%", "label": "外税", "amount": 76},
            {"rate": "10%", "label": "外税", "amount": 110},
        ],
        "line_items": [{"description": "dummy", "qty": 1, "unit_price": 2050, "total": 2050}],
    }
    ocr_text = "\n".join([
        "テストマート",
        "2026/6/1(月)",
        "商品ア",
        "400X",
        "商品イ",
        "500*",
        "商品ウ",
        "商品エ",
        "1800",
        "300*",
        "商品オ",
        "50*",
        "小計",
        "¥2,050",
        "外税8%対象額",
        "¥950",
        "外税10%対象額",
        "¥1,100",
        "お買上商品数:5",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, ocr_text)

    assert [item["description"] for item in extracted["line_items"]] == [
        "商品ア",
        "商品イ",
        "商品ウ",
        "商品エ",
        "商品オ",
    ]
    assert [item["total"] for item in extracted["line_items"]] == [
        400.0,
        500.0,
        800.0,
        300.0,
        50.0,
    ]
    assert extracted["line_items"][0]["tax_category"] == "0%"
    assert extracted["line_items"][2]["tax_category"] == "8%"


def test_qty_detail_repair_preserves_discounted_gross_line_convention():
    from receipt_parser.pipeline_receipt import _fix_qty_totals_from_ocr_unit_lines

    extracted = {
        "line_items": [
            {
                "description": "商品キ",
                "qty": 2,
                "unit_price": 196,
                "total": 98,
                "tax_category": "8%",
                "discount": 98,
                "discount_rate": "50%",
            }
        ]
    }
    ocr_text = "\n".join([
        "商品キ",
        "196",
        "(2個 X 単98)",
        "割引",
        "50%",
        "-98",
    ])

    _fix_qty_totals_from_ocr_unit_lines(extracted, ocr_text)

    assert extracted["line_items"][0]["qty"] == 2
    assert extracted["line_items"][0]["unit_price"] == 196
    assert extracted["line_items"][0]["total"] == 98


def test_following_qty_detail_repairs_previous_small_amount_when_summary_balances():
    from receipt_parser.pipeline_receipt import _repair_previous_item_from_following_qty_detail

    extracted = {
        "subtotal": 4447,
        "total": 4833,
        "line_items": [
            {"description": "食品ポリ袋L (バイオマス30", "qty": 1, "unit_price": 3, "total": 3},
            {"description": "やわらかステーキ重", "qty": 1, "unit_price": 598, "total": 598},
            {"description": "魚屋の焼さけほぐし弁当", "qty": 1, "unit_price": 398, "total": 398},
            {"description": "鶏ささみ大葉チーズロー", "qty": 1, "unit_price": 398, "total": 398},
            {"description": "ベビーダノンもも&緑黄", "qty": 1, "unit_price": 228, "total": 228},
            {"description": "ベビーダノンイ", "qty": 1, "unit_price": 228, "total": 228},
            {"description": "プチダノンリンコ", "qty": 1, "unit_price": 228, "total": 228},
            {"description": "キャベツ (1/2カット)", "qty": 1, "unit_price": 10, "total": 10},
            {"description": "TV純米料理酒", "qty": 1, "unit_price": 268, "total": 268},
            {"description": "カゴメ醸熱ソースとんか", "qty": 1, "unit_price": 228, "total": 228},
            {"description": "三ツ矢特濃グレープ", "qty": 1, "unit_price": 980, "total": 980},
            {"description": "ボスコEXVオリーブ", "qty": 1, "unit_price": 98, "total": 98},
            {"description": "マイネームツイン YKT", "qty": 1, "unit_price": 128, "total": 128},
            {"description": "オイコス プロテインド", "qty": 1, "unit_price": 248, "total": 248},
            {"description": "温州みかん100%ジュ", "qty": 1, "unit_price": 98, "total": 98},
            {"description": "たまねぎ 大袋", "qty": 1, "unit_price": 398, "total": 398},
            {"description": "本仕込食パン (8)", "qty": 1, "unit_price": 158, "total": 158},
            {"description": "りんごクリームデニッ", "qty": 1, "unit_price": 138, "total": 138},
        ],
    }
    ocr_path = Path(__file__).parent.parent / ".data/ocr_cache/variants/receipt_58_v1.txt"

    _repair_previous_item_from_following_qty_detail(extracted, ocr_path.read_text(encoding="utf-8"))

    cabbage = next(item for item in extracted["line_items"] if item["description"].startswith("キャベツ"))
    assert cabbage["qty"] == 2.0
    assert cabbage["unit_price"] == 70.0
    assert cabbage["total"] == 140.0
    assert sum(item["total"] for item in extracted["line_items"]) == 4963.0


def test_qty_detail_repair_preserves_standalone_discounted_gross_when_qty_was_missing():
    from receipt_parser.pipeline_receipt import _fix_qty_totals_from_ocr_unit_lines

    extracted = {
        "line_items": [
            {
                "description": "TV減の恵みきざみねぎ",
                "qty": 1,
                "unit_price": 196,
                "total": 98,
                "tax_category": "8%",
                "discount": 98,
                "discount_rate": "50%",
            }
        ]
    }
    ocr_text = "\n".join([
        "TV減の恵みきざみねぎ",
        "196",
        "(2個 X 単98)",
        "割引",
        "50%",
        "-98",
    ])

    _fix_qty_totals_from_ocr_unit_lines(extracted, ocr_text)

    assert extracted["line_items"][0]["qty"] == 2
    assert extracted["line_items"][0]["unit_price"] == 196
    assert extracted["line_items"][0]["total"] == 98


def test_qty_detail_repair_restores_standalone_gross_from_per_unit_discounted_row():
    from receipt_parser.pipeline_receipt import _fix_qty_totals_from_ocr_unit_lines

    extracted = {
        "line_items": [
            {
                "description": "TV減の恵みきざみねぎ",
                "qty": 2,
                "unit_price": 98,
                "total": 98,
                "tax_category": "8%",
                "discount": 98,
                "discount_rate": "50%",
            }
        ]
    }
    ocr_text = "\n".join([
        "TV減の恵みきざみねぎ",
        "196",
        "(2個 X 単98)",
        "割引",
        "50%",
        "-98",
    ])

    _fix_qty_totals_from_ocr_unit_lines(extracted, ocr_text)

    assert extracted["line_items"][0]["qty"] == 2
    assert extracted["line_items"][0]["unit_price"] == 196
    assert extracted["line_items"][0]["total"] == 98


def test_qty_detail_repair_divides_collapsed_discounted_gross_unit_price():
    from receipt_parser.pipeline_receipt import _fix_qty_totals_from_ocr_unit_lines

    extracted = {
        "line_items": [
            {
                "description": "商品ア",
                "qty": 1,
                "unit_price": 1850,
                "total": 1294,
                "tax_category": "8%",
                "discount": 556,
                "discount_rate": "30%",
            }
        ]
    }
    ocr_text = "\n".join([
        "をお願いします。",
        "取8309 : 109099143",
        "商品ア 1,850",
        "<2個 X 単925)",
        "割引",
        "30%",
        "-556",
    ])

    _fix_qty_totals_from_ocr_unit_lines(extracted, ocr_text)

    assert extracted["line_items"][0]["qty"] == 2
    assert extracted["line_items"][0]["unit_price"] == 925
    assert extracted["line_items"][0]["total"] == 1294


def test_qty_detail_repair_divides_collapsed_discounted_gross_when_qty_already_set():
    from receipt_parser.pipeline_receipt import _fix_qty_totals_from_ocr_unit_lines

    extracted = {
        "line_items": [
            {
                "description": "はくさい 1/4カット",
                "qty": 2,
                "unit_price": 196,
                "total": 186,
                "tax_category": "8%",
                "discount": 10,
                "discount_rate": "5%",
            }
        ]
    }
    ocr_text = "\n".join([
        "はくさい 1/4カット",
        "196",
        "(2個 X 単98)",
        "会員様割引5%",
        "-10",
    ])

    _fix_qty_totals_from_ocr_unit_lines(extracted, ocr_text)

    assert extracted["line_items"][0]["qty"] == 2
    assert extracted["line_items"][0]["unit_price"] == 98
    assert extracted["line_items"][0]["total"] == 186


def test_qty_detail_repair_divides_embedded_gross_description_unit_price():
    from receipt_parser.pipeline_receipt import _fix_qty_totals_from_ocr_unit_lines

    extracted = {
        "line_items": [
            {
                "description": "商品ア 1,850",
                "qty": 2,
                "unit_price": 1850,
                "total": 1294,
                "tax_category": "8%",
                "discount": 556,
                "discount_rate": "30%",
            }
        ]
    }
    ocr_text = "\n".join([
        "商品ア 1,850",
        "<2個 X 単925)",
        "割引",
        "30%",
        "-556",
    ])

    _fix_qty_totals_from_ocr_unit_lines(extracted, ocr_text)

    assert extracted["line_items"][0]["description"] == "商品ア"
    assert extracted["line_items"][0]["qty"] == 2
    assert extracted["line_items"][0]["unit_price"] == 925
    assert extracted["line_items"][0]["total"] == 1294


def test_qty_detail_repair_divides_inline_gross_price_after_description_cleanup():
    from receipt_parser.pipeline_receipt import _fix_qty_totals_from_ocr_unit_lines

    extracted = {
        "line_items": [
            {
                "description": "商品ア",
                "qty": 2,
                "unit_price": 1850,
                "total": 1294,
                "tax_category": "8%",
                "discount": 556,
                "discount_rate": "30%",
            }
        ]
    }
    ocr_text = "\n".join([
        "商品ア 1,850",
        "<2個 X 単925)",
        "割引",
        "30%",
        "-556",
    ])

    _fix_qty_totals_from_ocr_unit_lines(extracted, ocr_text)

    assert extracted["line_items"][0]["qty"] == 2
    assert extracted["line_items"][0]["unit_price"] == 925
    assert extracted["line_items"][0]["total"] == 1294


def test_qty_detail_repair_applies_discount_when_total_was_reset_to_gross():
    from receipt_parser.pipeline_receipt import _fix_qty_totals_from_ocr_unit_lines

    extracted = {
        "line_items": [
            {
                "description": "商品ア",
                "qty": 2,
                "unit_price": 925,
                "total": 1850,
                "tax_category": "8%",
                "discount": 556,
                "discount_rate": "30%",
            }
        ]
    }
    ocr_text = "\n".join([
        "商品ア 1,850",
        "<2個 X 単925)",
        "割引",
        "30%",
        "-556",
    ])

    _fix_qty_totals_from_ocr_unit_lines(extracted, ocr_text)

    assert extracted["line_items"][0]["qty"] == 2
    assert extracted["line_items"][0]["unit_price"] == 925
    assert extracted["line_items"][0]["total"] == 1294


def test_qty_detail_repair_uses_unit_first_ocr_row_when_total_invariant_matches():
    """Trigger: unit-first OCR qty detail; invariant: qty * unit equals item total."""
    from receipt_parser.pipeline_receipt import _fix_qty_totals_from_ocr_unit_lines

    extracted = {
        "line_items": [
            {
                "description": "商品A",
                "qty": 1,
                "unit_price": 1258,
                "total": 1258,
                "tax_category": "10%",
            }
        ]
    }
    ocr_text = "\n".join([
        "20060SA商品A",
        "単629×2個",
        "外 ¥1,258",
    ])

    _fix_qty_totals_from_ocr_unit_lines(extracted, ocr_text)

    assert extracted["line_items"][0]["qty"] == 2
    assert extracted["line_items"][0]["unit_price"] == 629
    assert extracted["line_items"][0]["total"] == 1258


def test_qty_detail_repair_ignores_unit_first_ocr_row_when_total_invariant_fails():
    """Trigger: unit-first OCR qty detail; invariant failure leaves row unchanged."""
    from receipt_parser.pipeline_receipt import _fix_qty_totals_from_ocr_unit_lines

    extracted = {
        "line_items": [
            {
                "description": "商品A",
                "qty": 1,
                "unit_price": 1200,
                "total": 1200,
                "tax_category": "10%",
            }
        ]
    }
    ocr_text = "\n".join([
        "20060SA商品A",
        "単629×2個",
        "外 ¥1,258",
    ])

    _fix_qty_totals_from_ocr_unit_lines(extracted, ocr_text)

    assert extracted["line_items"][0]["qty"] == 1
    assert extracted["line_items"][0]["unit_price"] == 1200
    assert extracted["line_items"][0]["total"] == 1200


def test_qty_detail_repair_names_unit_first_detail_row_from_nearest_owner_when_total_matches():
    """Trigger: item row is OCR qty detail; invariant: nearest owner and qty * unit total."""
    from receipt_parser.pipeline_receipt import _fix_qty_totals_from_ocr_unit_lines

    extracted = {
        "line_items": [
            {
                "description": "単629×2個",
                "qty": 2,
                "unit_price": 629,
                "total": 1258,
                "tax_category": "10%",
            }
        ]
    }
    ocr_text = "\n".join([
        "20060SA商品A",
        "単629×2個",
        "外 ¥1,258",
    ])

    _fix_qty_totals_from_ocr_unit_lines(extracted, ocr_text)

    assert extracted["line_items"][0]["description"] == "商品A"
    assert extracted["line_items"][0]["qty"] == 2
    assert extracted["line_items"][0]["unit_price"] == 629
    assert extracted["line_items"][0]["total"] == 1258


def test_qty_detail_repair_uses_matching_owner_before_intervening_noise_when_total_matches():
    """Trigger: unit-first qty row after OCR noise; invariant: matching owner and total."""
    from receipt_parser.pipeline_receipt import _fix_qty_totals_from_ocr_unit_lines

    extracted = {
        "line_items": [
            {
                "description": "商品A",
                "qty": 1,
                "unit_price": 1258,
                "total": 1258,
                "tax_category": "10%",
            }
        ]
    }
    ocr_text = "\n".join([
        "20060SA商品A",
        "取引情報",
        "明細情報",
        "登録情報",
        "処理情報",
        "確認情報",
        "印字情報",
        "受付情報",
        "端末情報",
        "管理情報",
        "単629×2個",
        "外 ¥1,258",
    ])

    _fix_qty_totals_from_ocr_unit_lines(extracted, ocr_text)

    assert extracted["line_items"][0]["description"] == "商品A"
    assert extracted["line_items"][0]["qty"] == 2
    assert extracted["line_items"][0]["unit_price"] == 629
    assert extracted["line_items"][0]["total"] == 1258


def test_ocr_neighborhood_total_repair_resets_unsupported_carried_quantity():
    from receipt_parser.pipeline_receipt import _fix_item_totals_from_ocr_neighborhood

    items = [
        {
            "description": "コドモトタベル ホシイモ",
            "qty": 2,
            "unit_price": 329,
            "total": 658,
            "tax_category": "8%",
        },
        {
            "description": "サッシロック2コグミ",
            "qty": 2,
            "unit_price": 329,
            "total": 658,
            "tax_category": "8%",
        },
        {
            "description": "ベビーハブラシジドウシタ",
            "qty": 1,
            "unit_price": 499,
            "total": 499,
            "tax_category": "10%",
        },
    ]
    ocr_text = "\n".join([
        "0030コドモトタベル ホシイモ",
        "単329×2個",
        "¥658",
        "20060Wサッシロック2コグミ 外",
        "¥599",
        "ベビーハブラシジドウシタ 外",
        "¥499",
        "小計",
        "¥1,756",
    ])

    _fix_item_totals_from_ocr_neighborhood(
        items,
        ocr_text,
        target_subtotal=1756,
        target_total=1933,
    )

    assert items[1]["qty"] == 1.0
    assert items[1]["unit_price"] == 599
    assert items[1]["total"] == 599


def test_ocr_neighborhood_repair_normalizes_inconsistent_qty_when_sum_balanced():
    from receipt_parser.pipeline_receipt import _fix_item_totals_from_ocr_neighborhood

    items = [
        {
            "description": "コドモトタベル ホシイモ",
            "qty": 1,
            "unit_price": 658,
            "total": 658,
            "tax_category": "8%",
        },
        {
            "description": "サッシロック2コグミ",
            "qty": 2,
            "unit_price": 329,
            "total": 599,
            "tax_category": "10%",
        },
        {
            "description": "ベビーハブラシジドウシタ",
            "qty": 1,
            "unit_price": 499,
            "total": 499,
            "tax_category": "10%",
        },
    ]
    ocr_text = "\n".join([
        "0030コドモトタベル ホシイモ",
        "単329×2個",
        "¥658",
        "20060Wサッシロック2コグミ 外",
        "¥599",
        "ベビーハブラシジドウシタ 外",
        "¥499",
        "小計",
        "¥1,756",
    ])

    _fix_item_totals_from_ocr_neighborhood(
        items,
        ocr_text,
        target_subtotal=1756,
        target_total=1933,
    )

    assert items[1]["qty"] == 1.0
    assert items[1]["unit_price"] == 599
    assert items[1]["total"] == 599


def test_balanced_external_tax_not_recomputed_from_misaligned_rate_base_stack():
    from receipt_parser.pipeline_receipt import postprocess_receipt

    extracted = {
        "merchant": "Grocery",
        "total": 3227,
        "subtotal": 2989,
        "taxes": [
            {"rate": "8%", "label": "外税", "amount": 238},
            {"rate": "10%", "label": "外税", "amount": 0},
        ],
        "line_items": [
            {"description": "Item A", "qty": 1, "unit_price": 1500, "total": 1500, "tax_category": "8%"},
            {"description": "Item B", "qty": 1, "unit_price": 1486, "total": 1486, "tax_category": "8%"},
            {"description": "Bag", "qty": 1, "unit_price": 3, "total": 3, "tax_category": "10%"},
        ],
    }
    ocr_text = "\n".join([
        "Item A",
        "Item B",
        "Bag",
        "小計",
        "外税8%対象額",
        "外税8%",
        "外税10年対象額",
        "外枠10%",
        "合計",
        "クレジット",
        "お釣り",
        "278*",
        "268",
        "118*",
        "¥2,989",
        "¥2,986",
        "¥238",
        "¥3",
        "¥0",
        "¥3,227",
        "¥3,227",
        "お買上商品数:15",
    ])

    result = postprocess_receipt(extracted, ocr_text, 0.9, {}, {}, "test-model")

    assert result["subtotal"] == 2989
    assert result["taxes"] == [{"rate": "8%", "label": "外税", "amount": 238}]


def test_nonfood_packaging_rows_use_standard_tax_category():
    from receipt_parser.pipeline_receipt import _fix_nonfood_packaging_tax_categories

    items = [
        {"description": "フードパック L", "total": 98, "tax_category": "8%"},
        {"description": "NEWレンジパック角3", "total": 128, "tax_category": "8%"},
        {"description": "はたらくのりものピック", "total": 698, "tax_category": "8%"},
    ]

    _fix_nonfood_packaging_tax_categories(items, "税率別対象額 10%", {"10%": 929})

    assert [item["tax_category"] for item in items] == ["10%", "10%", "8%"]


def test_external_tax_total_restored_from_printed_subtotal_ignores_loyalty_footer():
    from receipt_parser.pipeline_receipt import _restore_external_tax_total_from_printed_subtotal

    extracted = {
        "line_items": [
            {"description": "食品", "qty": 1, "unit_price": 5914, "total": 5914, "tax_category": "8%"},
            {"description": "日用品", "qty": 1, "unit_price": 423, "total": 423, "tax_category": "10%"},
            {"description": "非課税品", "qty": 1, "unit_price": 652, "total": 652, "tax_category": "0%"},
        ],
        "subtotal": 6474,
        "total": 6845,
        "amount_paid": 6845,
        "points_used": 0,
        "taxes": [
            {"rate": "8%", "label": "外税", "amount": 473},
            {"rate": "10%", "label": "外税", "amount": 42},
            {"rate": "0%", "label": "非課税", "amount": 652},
        ],
    }
    ocr_text = "\n".join([
        "小計",
        "¥6,989",
        "外税8%対象額",
        "¥5,914",
        "外税8%",
        "¥473",
        "外税10%対象額",
        "¥423",
        "外税10%",
        "¥42",
        "非課税対象額",
        "¥652",
        "¥7,504",
        "合計",
        "電子マネー支払",
        "¥7,504",
        "ポイント対象金額(税込)",
        "¥6,844",
        "合",
        "計",
        "68P",
    ])

    _restore_external_tax_total_from_printed_subtotal(extracted, ocr_text)

    assert extracted["subtotal"] == 6989
    assert extracted["total"] == 7504
    assert extracted["amount_paid"] == 7504


def test_following_discount_line_reduces_item_total():
    from receipt_parser.pipeline_receipt import _fix_item_totals_from_following_discount_lines

    extracted = {
        "line_items": [
            {
                "description": "使い捨てないカイロ モバイルバッテリー付ホワイト",
                "qty": 1,
                "unit_price": 3828,
                "total": 3828,
                "discount": 0,
            },
            {"description": "マルチワイヤースプーン", "qty": 1, "unit_price": 1540, "total": 1540},
        ]
    }
    ocr_text = "\n".join([
        "使い捨てないカイロ モバイルバッテリー付ホワイト",
        "4944370053289",
        "セール",
        "マルチワイヤースプーン",
        "4988760010957",
        "¥3,828",
        "- ¥383",
        "¥1,540",
    ])

    _fix_item_totals_from_following_discount_lines(extracted, ocr_text)

    item = extracted["line_items"][0]
    assert item["unit_price"] == 3828.0
    assert item["discount"] == 383.0
    assert item["total"] == 3445.0

    extracted["line_items"][0].update({"unit_price": 3445, "total": 3445, "discount": 0})
    _fix_item_totals_from_following_discount_lines(extracted, ocr_text)

    item = extracted["line_items"][0]
    assert item["unit_price"] == 3828.0
    assert item["discount"] == 383.0
    assert item["total"] == 3445.0


def test_following_discount_line_handles_inline_price_and_backslash_yen_marker():
    from receipt_parser.pipeline_receipt import _fix_item_totals_from_following_discount_lines

    extracted = {
        "line_items": [
            {
                "description": "埋め込みスイッチ",
                "qty": 1,
                "unit_price": 228,
                "total": 228,
                "discount": 0,
            },
            {
                "description": "EXTRA ゼリー状 4g",
                "qty": 1,
                "unit_price": 598,
                "total": 598,
                "discount": 0,
            },
            {
                "description": "PAアルカリボタン電池2P",
                "qty": 1,
                "unit_price": 428,
                "total": 428,
                "discount": 0,
            },
        ]
    }
    ocr_text = "\n".join([
        "0014 PA 埋め込みスイッチ ¥228",
        "4902710561986",
        "部門会員割 10%",
        "-\\23",
        "0018 EXTRA ゼリー状 4g ¥598",
        "4901490052745",
        "部門会員割 10%",
        "-\\60",
        "0008 PAアルカリボタン電池2P ¥428",
        "4984824719940",
        "部門会員割 10%",
        "-\\43",
    ])

    _fix_item_totals_from_following_discount_lines(extracted, ocr_text)

    assert [item["unit_price"] for item in extracted["line_items"]] == [228.0, 598.0, 428.0]
    assert [item["discount"] for item in extracted["line_items"]] == [23.0, 60.0, 43.0]
    assert [item["total"] for item in extracted["line_items"]] == [205.0, 538.0, 385.0]


def test_following_discount_line_handles_marker_price_row():
    from receipt_parser.pipeline_receipt import _fix_item_totals_from_following_discount_lines

    extracted = {
        "line_items": [
            {
                "description": "銀さけ切身",
                "qty": 1,
                "unit_price": 510,
                "total": 510,
                "discount": 0,
                "discount_rate": "",
            }
        ]
    }
    ocr_text = "\n".join([
        "銀さけ切身",
        "510*",
        "10%",
        "-51",
    ])

    _fix_item_totals_from_following_discount_lines(extracted, ocr_text)

    item = extracted["line_items"][0]
    assert item["unit_price"] == 510.0
    assert item["discount"] == 51.0
    assert item["discount_rate"] == "10%"
    assert item["total"] == 459.0


def test_discounted_gross_price_repair_preserves_quantity_unit_price():
    from receipt_parser.pipeline_receipt import _fix_discounted_item_gross_prices_from_ocr

    extracted = {
        "line_items": [
            {
                "description": "ちょっと贅沢 ぶどう",
                "qty": 2,
                "unit_price": 158,
                "total": 316,
                "tax_category": "8%",
                "discount": 8,
                "discount_rate": "",
            },
        ],
    }
    ocr_text = "\n".join([
        "ちょっと贅沢 ぶどう",
        "316 A",
        "(2個 X 単158)",
        "まとめ値引",
        "-8",
    ])

    _fix_discounted_item_gross_prices_from_ocr(extracted, ocr_text)

    assert extracted["line_items"][0]["qty"] == 2
    assert extracted["line_items"][0]["unit_price"] == 158
    assert extracted["line_items"][0]["total"] == 308


def test_discounted_ocr_pair_repair_uses_visible_owner_for_duplicate_description():
    from receipt_parser.pipeline_receipt import _repair_discounted_ocr_pair_descriptions

    extracted = {
        "line_items": [
            {"description": "商品ア", "qty": 1, "unit_price": 158, "total": 158, "discount": 0},
            {"description": "商品ア", "qty": 1, "unit_price": 158, "total": 150, "discount": 8},
        ],
    }
    ocr_text = "\n".join([
        "商品ア 158*",
        "商品イ 158 A",
        "まとめ値引",
        "-8",
    ])

    _repair_discounted_ocr_pair_descriptions(extracted, ocr_text)

    assert extracted["line_items"][1]["description"] == "商品イ"
    assert extracted["line_items"][1]["unit_price"] == 158
    assert extracted["line_items"][1]["total"] == 150


def test_duplicate_row_drop_requires_subtotal_balance_and_single_ocr_occurrence():
    from receipt_parser.pipeline_receipt import _drop_duplicate_rows_when_subtotal_balances

    extracted = {
        "subtotal": 4318,
        "line_items": [
            {"description": "商品ア", "qty": 1, "unit_price": 100, "total": 100, "discount": 0},
            {"description": "ちょっと贅沢 ぶどう", "qty": 2, "unit_price": 158, "total": 308, "discount": 8},
            {"description": "ちょっと贅沢 ぶどう", "qty": 1, "unit_price": 316, "total": 308, "discount": 8},
            {"description": "商品イ", "qty": 1, "unit_price": 3910, "total": 3910, "discount": 0},
        ],
    }
    ocr_text = "\n".join([
        "商品ア 100",
        "ちょっと贅沢 ぶどう",
        "316 A",
        "(2個 X 単158)",
        "まとめ値引",
        "-8",
        "商品イ 3910",
        "小計",
        "¥4,318",
    ])

    _drop_duplicate_rows_when_subtotal_balances(extracted, ocr_text)

    assert len(extracted["line_items"]) == 3
    assert sum(item["total"] for item in extracted["line_items"]) == 4318
    kept = [item for item in extracted["line_items"] if item["description"] == "ちょっと贅沢 ぶどう"]
    assert kept == [
        {"description": "ちょっと贅沢 ぶどう", "qty": 2, "unit_price": 158, "total": 308, "discount": 8}
    ]


def test_basket_marker_rows_reconstruct_stacked_names_and_coupon_without_merchant_gate():
    from receipt_parser.pipeline_receipt import _replace_basket_marker_rows_when_balanced

    ocr_text = "\n".join([
        "WAREHOUSE",
        "売上",
        ">>> BEGIN BOTTOM OF BASKET <<<",
        "* SPARKLING VARIETY",
        "1791435",
        "10",
        "2,448",
        "515685",
        "TISSUE 10PC",
        "10",
        "780",
        "2,448 E",
        "780 T",
        ">>> BOTTOM OF BASKET ITEM COUNT 2 <<<",
        "60769",
        "* SMOKED CHICKEN",
        "* CHEESE 800G",
        "1@",
        "939",
        "939 E",
        "80763",
        "10",
        "1,118",
        "1.118 E",
        "TOY DOLLS",
        "74511",
        "10",
        "4,967",
        "4,967 T",
        "LINDT EASTER MIX",
        "77616",
        "1@",
        "3,980",
        "3,980 E",
        "CPN",
        "* LINDT COUPON CPN",
        "86243",
        "1⚫",
        "1,000",
        "1,000-E",
        "**** 合計",
        "14,308",
        "8%対象",
        "7,485 (消費税",
        "554 )",
        "10% 対象",
        "5,747 (消費税",
        "522)",
        "御買上げ点数 :6",
    ])
    extracted = {
        "total": 14308,
        "subtotal": 13232,
        "taxes": [
            {"rate": "8%", "label": "内税", "amount": 554},
            {"rate": "10%", "label": "内税", "amount": 522},
        ],
        "line_items": [
            {"description": "wrong", "qty": 1, "unit_price": 10, "total": 10, "tax_category": "10%"},
        ],
    }

    _replace_basket_marker_rows_when_balanced(extracted, ocr_text)

    rows = extracted["line_items"]
    assert [row["description"] for row in rows] == [
        "SPARKLING VARIETY",
        "TISSUE 10PC",
        "SMOKED CHICKEN",
        "CHEESE 800G",
        "TOY DOLLS",
        "LINDT EASTER MIX",
    ]
    assert [row["total"] for row in rows] == [2448, 780, 939, 1118, 4967, 2980]
    assert rows[-1]["unit_price"] == 3980
    assert rows[-1]["discount"] == 1000
    assert sum(row["total"] for row in rows if row["tax_category"] == "8%") == 7485
    assert sum(row["total"] for row in rows if row["tax_category"] == "10%") == 5747


def test_following_discount_line_handles_short_description_percent_marker():
    from receipt_parser.pipeline_receipt import (
        _clear_discounts_without_nearby_ocr_marker,
        _fix_item_totals_from_following_discount_lines,
    )

    ocr_text = "\n".join([
        "自転車",
        "¥69,900",
        "-5%",
        "% -",
        "-3,495",
        "アクセサリー",
        "¥4,700",
    ])
    extracted = {
        "line_items": [
            {
                "description": "自転車",
                "qty": 1,
                "unit_price": 69900,
                "total": 69900,
                "discount": 0,
                "discount_rate": "",
            },
            {"description": "アクセサリー", "qty": 1, "unit_price": 4700, "total": 4700},
        ]
    }

    _fix_item_totals_from_following_discount_lines(extracted, ocr_text)

    item = extracted["line_items"][0]
    assert item["unit_price"] == 69900.0
    assert item["discount"] == 3495.0
    assert item["discount_rate"] == "5%"
    assert item["total"] == 66405.0

    _clear_discounts_without_nearby_ocr_marker(extracted["line_items"], ocr_text)

    assert item["discount"] == 3495.0
    assert item["discount_rate"] == "5%"
    assert item["total"] == 66405.0


def test_following_discount_line_does_not_reuse_later_discount_block():
    from receipt_parser.pipeline_receipt import _fix_item_totals_from_following_discount_lines

    ocr_text = "\n".join([
        "サントリーグリーンダカラヤサシイムキ",
        "¥119",
        "オオバチーズinチクワサラダ",
        "¥194",
        "Rヤキカレーパン  ¥137",
        "Yヤキモチ(ツブアン)  ¥118",
        "割引: 20%",
        "-¥24",
    ])
    extracted = {
        "line_items": [
            {
                "description": "サントリーグリーンダカラヤサシイムキ",
                "qty": 1,
                "unit_price": 119,
                "total": 119,
                "discount": 0,
                "discount_rate": "",
            },
            {
                "description": "Rヤキカレーパン",
                "qty": 1,
                "unit_price": 137,
                "total": 137,
                "discount": 0,
                "discount_rate": "",
            },
            {
                "description": "Yヤキモチ(ツブアン)",
                "qty": 1,
                "unit_price": 118,
                "total": 118,
                "discount": 0,
                "discount_rate": "",
            },
        ]
    }

    _fix_item_totals_from_following_discount_lines(extracted, ocr_text)

    assert extracted["line_items"][0]["total"] == 119
    assert extracted["line_items"][0]["discount"] == 0
    assert extracted["line_items"][1]["total"] == 137
    assert extracted["line_items"][1]["discount"] == 0
    assert extracted["line_items"][2]["total"] == 94
    assert extracted["line_items"][2]["discount"] == 24
    assert extracted["line_items"][2]["discount_rate"] == "20%"


def test_coupon_discount_block_applies_to_nearest_preceding_item():
    from receipt_parser.pipeline_receipt import _apply_coupon_discount_blocks

    extracted = {
        "line_items": [
            {
                "description": "ALPHA MIX",
                "qty": 1,
                "unit_price": 3980,
                "total": 3980,
                "discount": 0,
                "discount_rate": "",
            },
            {
                "description": "BETA PACK",
                "qty": 1,
                "unit_price": 500,
                "total": 500,
                "discount": 0,
                "discount_rate": "",
            },
        ]
    }
    ocr_text = "\n".join([
        "ALPHA MIX",
        "77616",
        "1@",
        "3,980",
        "3,980 E",
        "CPN",
        "* ALPHA MIX COUPON CPN",
        "86243",
        "1@",
        "1,000-E",
        "BETA PACK",
        "500",
    ])

    _apply_coupon_discount_blocks(extracted, ocr_text)

    assert extracted["line_items"][0]["unit_price"] == 3980.0
    assert extracted["line_items"][0]["discount"] == 1000.0
    assert extracted["line_items"][0]["total"] == 2980.0
    assert extracted["line_items"][1]["total"] == 500


def test_applied_coupon_line_item_is_dropped_after_discount_application():
    from receipt_parser.pipeline_receipt import (
        _apply_coupon_discount_blocks,
        _drop_applied_coupon_line_items,
    )

    extracted = {
        "line_items": [
            {
                "description": "ALPHA MIX",
                "qty": 1,
                "unit_price": 3980,
                "total": 3980,
                "discount": 0,
                "discount_rate": "",
            },
            {
                "description": "ALPHA MIX CPN",
                "qty": 1,
                "unit_price": 1000,
                "total": 1000,
                "discount": 0,
                "discount_rate": "",
            },
            {
                "description": "BETA PACK",
                "qty": 1,
                "unit_price": 500,
                "total": 500,
                "discount": 0,
                "discount_rate": "",
            },
        ]
    }
    ocr_text = "\n".join([
        "ALPHA MIX",
        "3,980",
        "CPN",
        "* ALPHA MIX COUPON CPN",
        "1,000-E",
        "BETA PACK",
        "500",
    ])

    _apply_coupon_discount_blocks(extracted, ocr_text)
    _drop_applied_coupon_line_items(extracted, ocr_text)

    assert [item["description"] for item in extracted["line_items"]] == ["ALPHA MIX", "BETA PACK"]
    assert extracted["line_items"][0]["discount"] == 1000.0
    assert extracted["line_items"][0]["total"] == 2980.0


def test_tiny_item_price_repair_uses_repeated_following_ocr_amount_when_sum_improves():
    from receipt_parser.pipeline_receipt import _repair_tiny_item_prices_from_following_ocr

    extracted = {
        "subtotal": 2288,
        "line_items": [
            {
                "description": "ALPHA SNACK",
                "qty": 1,
                "unit_price": 10,
                "total": 10,
                "discount": 0,
            },
            {
                "description": "BETA PACK",
                "qty": 1,
                "unit_price": 1000,
                "total": 1000,
                "discount": 0,
            },
        ],
    }
    ocr_text = "\n".join([
        "ALPHA SNACK",
        "566689",
        "1@",
        "1,288",
        "1,288 E",
        "BETA PACK",
        "1,000",
        "小計",
        "2,288",
    ])

    _repair_tiny_item_prices_from_following_ocr(extracted, ocr_text)

    assert extracted["line_items"][0]["unit_price"] == 1288.0
    assert extracted["line_items"][0]["total"] == 1288.0
    assert sum(item["total"] for item in extracted["line_items"]) == 2288.0


def test_following_barcode_price_block_repairs_wrong_non_tiny_price_when_sum_improves():
    from receipt_parser.pipeline_receipt import _repair_tiny_item_prices_from_following_ocr

    extracted = {
        "subtotal": 2288,
        "line_items": [
            {
                "description": "ALPHA SNACK",
                "qty": 1,
                "unit_price": 698,
                "total": 698,
                "discount": 0,
            },
            {
                "description": "BETA PACK",
                "qty": 1,
                "unit_price": 1000,
                "total": 1000,
                "discount": 0,
            },
        ],
    }
    ocr_text = "\n".join([
        "ALPHA SNACK",
        "10",
        "2,298",
        "2,298 E",
        "698",
        "566689",
        "1e",
        "1,288",
        "1,288 E",
        "BETA PACK",
        "1,000",
        "小計",
        "2,288",
    ])

    _repair_tiny_item_prices_from_following_ocr(extracted, ocr_text)

    assert extracted["line_items"][0]["unit_price"] == 1288.0
    assert extracted["line_items"][0]["total"] == 1288.0
    assert sum(item["total"] for item in extracted["line_items"]) == 2288.0


def test_following_barcode_price_block_uses_second_repeated_price_after_noise():
    from receipt_parser.pipeline_receipt import _repair_tiny_item_prices_from_following_ocr

    extracted = {
        "total": 2288,
        "line_items": [
            {
                "description": "ALPHA SNACK",
                "qty": 1,
                "unit_price": 10,
                "total": 10,
                "discount": 0,
            },
            {
                "description": "BETA PACK",
                "qty": 1,
                "unit_price": 1000,
                "total": 1000,
                "discount": 0,
            },
        ],
    }
    ocr_text = "\n".join([
        "ALPHA SNACK",
        "10",
        "2,298",
        "2.298 E",
        "1.",
        "698",
        "698 E",
        "566689",
        "1e",
        "1,288",
        "1,288 E",
        "BETA PACK",
        "1,000",
        "合計",
        "2,288",
    ])

    _repair_tiny_item_prices_from_following_ocr(extracted, ocr_text)

    assert extracted["line_items"][0]["unit_price"] == 1288.0
    assert extracted["line_items"][0]["total"] == 1288.0
    assert sum(item["total"] for item in extracted["line_items"]) == 2288.0


def test_following_barcode_price_block_accepts_dotted_thousands_price():
    from receipt_parser.pipeline_receipt import _repair_tiny_item_prices_from_following_ocr

    extracted = {
        "subtotal": 2057,
        "line_items": [
            {
                "description": "ALPHA SLICE",
                "qty": 1,
                "unit_price": 939,
                "total": 939,
                "discount": 0,
            },
            {
                "description": "BETA CHEESE 800G",
                "qty": 1,
                "unit_price": 939,
                "total": 939,
                "discount": 0,
            },
        ],
    }
    ocr_text = "\n".join([
        "ALPHA SLICE",
        "1@",
        "939",
        "939 E",
        "BETA CHEESE 800G",
        "80763",
        "10",
        "1.118",
        "1.118 E",
        "小計",
        "2,057",
    ])

    _repair_tiny_item_prices_from_following_ocr(extracted, ocr_text)

    assert extracted["line_items"][1]["unit_price"] == 1118.0
    assert extracted["line_items"][1]["total"] == 1118.0
    assert sum(item["total"] for item in extracted["line_items"]) == 2057.0


def test_seria_split_price_block_maps_prices_to_names():
    from receipt_parser.pipeline_receipt import _replace_split_price_block_when_balanced

    extracted = {
        "merchant": "Seria",
        "subtotal": 303,
        "line_items": [
            {"description": "開き戸安全ロックシンプル", "qty": 1, "unit_price": 100, "total": 100},
            {"description": "いたずら防止ストッパーLシンプル", "qty": 1, "unit_price": 303, "total": 303},
            {"description": "レジ袋小", "qty": 1, "unit_price": 3, "total": 3},
        ],
    }
    ocr_text = "\n".join([
        "Seria",
        "2026/06/03(水) 10:32",
        "87239",
        "200",
        "開き戸安全ロックシンプル",
        "いたずら防止ストッパーLシンプル",
        "レジ袋小",
        "小計",
        "消費税",
        "合計",
        "10%対象",
        "4点",
        "100",
        "3",
        "303",
        "30",
        "¥333",
    ])

    _replace_split_price_block_when_balanced(extracted, ocr_text)

    assert [item["total"] for item in extracted["line_items"]] == [200.0, 100.0, 3.0]
    assert [item["unit_price"] for item in extracted["line_items"]] == [200.0, 100.0, 3.0]


def test_split_price_block_uses_printed_subtotal_when_mutable_subtotal_drifted():
    from receipt_parser.pipeline_receipt import _replace_split_price_block_when_balanced

    extracted = {
        "merchant": "Any store",
        "subtotal": 323,
        "line_items": [
            {"description": "開き戸安全ロックシンプル", "qty": 1, "unit_price": 100, "total": 100},
            {"description": "いたずら防止ストッパーLシンプル", "qty": 1, "unit_price": 100, "total": 100},
            {"description": "レジ袋小", "qty": 1, "unit_price": 3, "total": 3},
        ],
    }
    ocr_text = "\n".join([
        "SHOP",
        "87239",
        "200",
        "開き戸安全ロックシンプル",
        "いたずら防止ストッパーLシンプル",
        "レジ袋小",
        "小計",
        "消費税",
        "合計",
        "10%対象",
        "4点",
        "100",
        "3",
        "303",
        "30",
        "¥333",
    ])

    _replace_split_price_block_when_balanced(extracted, ocr_text)

    assert [item["total"] for item in extracted["line_items"]] == [200.0, 100.0, 3.0]


def test_final_receipt_output_repairs_apply_split_price_block():
    from receipt_parser.pipeline import _apply_final_receipt_output_repairs

    result = {
        "document_type": "receipt",
        "merchant": "Any store",
        "subtotal": 303,
        "total": 333,
        "line_items": [
            {"description": "開き戸安全ロックシンプル", "qty": 1, "unit_price": 100, "total": 100},
            {"description": "いたずら防止ストッパーLシンプル", "qty": 1, "unit_price": 303, "total": 303},
            {"description": "レジ袋小", "qty": 1, "unit_price": 3, "total": 3},
        ],
    }
    ocr_text = "\n".join([
        "SHOP",
        "87239",
        "200",
        "開き戸安全ロックシンプル",
        "いたずら防止ストッパーLシンプル",
        "レジ袋小",
        "小計",
        "消費税",
        "合計",
        "10%対象",
        "4点",
        "100",
        "3",
        "303",
        "30",
        "¥333",
    ])

    _apply_final_receipt_output_repairs(result, ocr_text)

    assert [item["total"] for item in result["line_items"]] == [200.0, 100.0, 3.0]


def test_familymart_stacked_rows_repairs_one_truncated_price_by_total():
    from receipt_parser.pipeline_receipt import _replace_stacked_name_price_rows_when_balanced

    extracted = {
        "merchant": "FamilyMart",
        "total": 2049,
        "line_items": [{"description": "dummy", "qty": 1, "unit_price": 2049, "total": 2049}],
    }
    ocr_text = "\n".join([
        "FamilyMart",
        "◎背脂ニンニク醤油",
        "¥258軽",
        "三ツ矢サイダークラシッ",
        "¥183軽",
        "濃厚カスタードシュー",
        "¥180",
        "もちむにシューミルク",
        "¥213",
        "ザクほろシュー(チョコ",
        "RCソルティハニーバタ",
        "グリーンティー",
        "¥238",
        "¥37",
        "¥372",
        "ぎっしりアーモンドバ",
        "レジ袋20号バイオマス",
        "合計",
        "¥228軽",
        "¥5",
        "¥2,049",
    ])

    _replace_stacked_name_price_rows_when_balanced(extracted, ocr_text)

    assert [item["description"] for item in extracted["line_items"]][-4:] == [
        "RCソルティハニーバタ",
        "グリーンティー",
        "ぎっしりアーモンドバ",
        "レジ袋20号バイオマス",
    ]
    assert [item["total"] for item in extracted["line_items"]] == [258.0, 183.0, 180.0, 213.0, 238.0, 372.0, 372.0, 228.0, 5.0]


def test_stacked_rows_prefer_total_when_subtotal_is_inclusive_tax_net_value():
    from receipt_parser.pipeline_receipt import _replace_stacked_name_price_rows_when_balanced

    extracted = {
        "total": 2049,
        "subtotal": 1898,
        "taxes": [{"rate": "8%", "label": "内税", "amount": 151}],
        "line_items": [{"description": "dummy", "qty": 1, "unit_price": 2049, "total": 2049}],
    }
    ocr_text = "\n".join([
        "領収",
        "◎背脂ニンニク醤油",
        "¥258軽",
        "三ツ矢サイダークラシッ",
        "¥183軽",
        "濃厚カスタードシュー",
        "¥180",
        "もちむにシューミルク",
        "¥213",
        "ザクほろシュー(チョコ",
        "RCソルティハニーバタ",
        "グリーンティー",
        "¥238",
        "¥37",
        "¥372",
        "ぎっしりアーモンドバ",
        "レジ袋20号バイオマス",
        "合計",
        "¥228軽",
        "¥5",
        "¥2,049",
        "(8% 対象",
        "(内消費税等",
        "¥2,044)",
        "¥151)",
    ])

    _replace_stacked_name_price_rows_when_balanced(extracted, ocr_text)

    assert [item["total"] for item in extracted["line_items"]] == [258.0, 183.0, 180.0, 213.0, 238.0, 372.0, 372.0, 228.0, 5.0]


def test_familymart_stacked_rows_handles_spaced_receipt_marker_and_qty_detail():
    from receipt_parser.pipeline_receipt import _replace_stacked_name_price_rows_when_balanced

    extracted = {
        "merchant": "FamilyMart",
        "total": 992,
        "line_items": [{"description": "dummy", "qty": 1, "unit_price": 992, "total": 992}],
    }
    ocr_text = "\n".join([
        "FamilyMart",
        "宗像三郎丸店",
        "領 収 証",
        "◎コクと旨みのたまご 1",
        "レジ袋弁当大バイオマス",
        "アルフォートミニチョコ",
        "上白糖 500G",
        "¥358軽",
        "¥5",
        "¥203軽",
        "@213× 2点",
        "¥426軽",
        "合計",
        "¥992",
    ])

    _replace_stacked_name_price_rows_when_balanced(extracted, ocr_text)

    assert [item["description"] for item in extracted["line_items"]] == [
        "コクと旨みのたまご1",
        "レジ袋弁当大バイオマス",
        "アルフォートミニチョコ",
        "上白糖500G",
    ]
    assert [item["total"] for item in extracted["line_items"]] == [358.0, 5.0, 203.0, 426.0]
    assert extracted["line_items"][3]["qty"] == 2.0
    assert extracted["line_items"][3]["unit_price"] == 213.0


def test_stacked_inclusive_tax_block_maps_label_values():
    from receipt_parser.pipeline_receipt import _restore_stacked_inclusive_tax_block

    extracted = {"taxes": [{"rate": "10%", "label": "内税", "amount": 58}]}
    ocr_text = "\n".join([
        "Convenience Store",
        "10% 対象",
        "(内消費税等",
        "(8% 対象",
        "(内消費税等",
        "¥430",
        "¥1,403",
        "¥5)",
        "¥0)",
        "¥987)",
        "¥73)",
    ])

    _restore_stacked_inclusive_tax_block(extracted, ocr_text)

    assert extracted["taxes"] == [{"rate": "8%", "label": "内税", "amount": 73.0}]


def test_stacked_inclusive_tax_block_preserves_tax_excluded_receipts():
    from receipt_parser.pipeline_receipt import _restore_stacked_inclusive_tax_block

    extracted = {"taxes": [{"rate": "8%", "label": "外税", "amount": 203.0}]}
    ocr_text = "\n".join([
        "小計 (税抜 8%)",
        "消費税等 (8%)",
        "小計(税抜10%)",
        "合計",
        "¥2,541",
        "¥203",
        "¥4",
        "¥2,748",
        "(税率 8% 対象",
        "¥2,744)",
        "(税率10% 対象",
        "¥4)",
        "(内消費税等 8%",
        "¥203)",
    ])

    _restore_stacked_inclusive_tax_block(extracted, ocr_text)

    assert extracted["taxes"] == [{"rate": "8%", "label": "外税", "amount": 203.0}]


def test_tax_excluded_per_rate_block_restores_external_amounts():
    from receipt_parser.pipeline_receipt import _restore_tax_excluded_per_rate_blocks

    extracted = {"taxes": [{"rate": "8%", "label": "外税", "amount": 16}]}
    ocr_text = "\n".join([
        "小計(税抜 8%)",
        "消費税等 (8%)",
        "小計 (税抜10%)",
        "消費税等 (10%)",
        "合計",
        "(税率 8% 対象",
        "(税率 10% 対象",
        "¥796",
        "¥63",
        "¥165",
        "¥16",
        "¥1,040",
    ])

    _restore_tax_excluded_per_rate_blocks(extracted, ocr_text)

    assert extracted["taxes"] == [
        {"rate": "8%", "label": "外税", "amount": 63.0},
        {"rate": "10%", "label": "外税", "amount": 16.0},
    ]


def test_tax_excluded_per_rate_block_handles_split_label_value_stacks():
    from receipt_parser.pipeline_receipt import _restore_tax_excluded_per_rate_blocks

    extracted = {"taxes": [{"rate": "8%", "label": "外税", "amount": 4}]}
    ocr_text = "\n".join([
        "小計 (税抜 8%)",
        "消費税等 (8%)",
        "¥2,541",
        "¥203",
        "小計 (税抜10%)",
        "合計",
        "¥4",
        "¥2,748",
        "(税率 8% 対象",
        "¥2,744)",
        "(税率 10% 対象",
        "¥4)",
        "(内消費税等 8%",
        "¥203)",
    ])

    _restore_tax_excluded_per_rate_blocks(extracted, ocr_text)

    assert extracted["taxes"] == [{"rate": "8%", "label": "外税", "amount": 203.0}]


def test_single_rate_inclusive_tax_block_restores_tax_from_expected_value():
    from receipt_parser.pipeline_receipt import _restore_single_rate_inclusive_tax_block

    extracted = {"total": 570, "taxes": []}
    ocr_text = "\n".join([
        "(8%対象",
        "(内 8%税",
        "合計",
        "¥570",
        "¥42)",
        "¥570",
    ])

    _restore_single_rate_inclusive_tax_block(extracted, ocr_text)

    assert extracted["taxes"] == [{"rate": "8%", "label": "内税", "amount": 42.0}]
    assert extracted["subtotal"] == 528.0


def test_single_rate_inclusive_tax_block_restores_inline_target_summary():
    from receipt_parser.pipeline_receipt import _restore_single_rate_inclusive_tax_block

    extracted = {
        "total": 19118,
        "subtotal": 18650,
        "taxes": [{"rate": "10%", "label": "内税", "amount": 468}],
        "line_items": [
            {"description": "パンスト", "total": 759, "tax_category": "8%"},
            {"description": "フォーマル", "total": 1969, "tax_category": "8%"},
            {"description": "クリ", "total": 2420, "tax_category": "8%"},
            {"description": "フォーマル", "total": 13970, "tax_category": "8%"},
        ],
    }
    ocr_text = "\n".join([
        "小計",
        "¥19,118",
        "合計",
        "¥19,118",
        "内消費税",
        "¥1,738",
        "10%対象 ¥19,118 内消費税 ¥1,738",
    ])

    _restore_single_rate_inclusive_tax_block(extracted, ocr_text)

    assert extracted["taxes"] == [{"rate": "10%", "label": "内税", "amount": 1738.0}]
    assert extracted["subtotal"] == 17380.0
    assert [item["tax_category"] for item in extracted["line_items"]] == ["10%", "10%", "10%", "10%"]


def test_code_prefixed_item_descriptions_are_cleaned_generically():
    from receipt_parser.pipeline_receipt import _clean_code_prefixed_item_descriptions

    extracted = {
        "line_items": [
            {"description": "470-0244 パンスト 1", "qty": 1, "total": 759},
            {"description": "340-0059 フォーマル", "qty": 1, "total": 1969},
        ]
    }

    _clean_code_prefixed_item_descriptions(extracted)

    assert [item["description"] for item in extracted["line_items"]] == ["パンスト", "フォーマル"]


def test_price_line_reduced_markers_assign_categories_by_item_order():
    from receipt_parser.pipeline_receipt import _fix_tax_categories_from_price_line_markers

    extracted = {
        "line_items": [
            {"description": "ジューシーハムサンド", "total": 330, "tax_category": "10%"},
            {"description": "カシミヤEXポケットティシュー15組4コ", "total": 162, "tax_category": "8%"},
            {"description": "バイオ30レジ袋中1枚", "total": 3, "tax_category": "8%"},
        ]
    }
    ocr_text = "\n".join([
        "ジューシーハムサンド",
        "*330",
        "カシミヤEXポケットティシュー15組4コ",
        "162",
        "バイオ30レジ袋中1枚",
        "3",
        "[*] マークは軽減税率対象です。",
    ])

    _fix_tax_categories_from_price_line_markers(extracted, ocr_text)

    assert [item["tax_category"] for item in extracted["line_items"]] == ["8%", "10%", "10%"]


def test_header_ascii_brand_preferred_when_host_store_was_extracted():
    from receipt_parser.pipeline_receipt import _fix_company_name_merchant

    extracted = {"merchant": "サンリブ"}
    ocr_text = "HAPNS サンリブくりえいと宗像店\n2026/06/01\n領収書"

    _fix_company_name_merchant(extracted, ocr_text)

    assert extracted["merchant"] == "HAPNS"


def test_stacked_header_ascii_brand_preferred_when_host_store_was_extracted():
    from receipt_parser.pipeline_receipt import _fix_company_name_merchant

    extracted = {"merchant": "スーパービバホーム"}
    ocr_text = "\n".join([
        "Super",
        "VIVAHOME",
        "ホームセンター スーパービバホーム",
        "赤間店",
        "領収証",
    ])

    _fix_company_name_merchant(extracted, ocr_text)

    assert extracted["merchant"] == "VIVAHOME"


def test_invoice_registration_number_merchant_recovers_phone_header_name():
    from receipt_parser.pipeline_receipt import _fix_company_name_merchant

    extracted = {"merchant": "T1234567890123"}
    ocr_text = "\n".join([
        "支店名",
        "テストストア (0940)38-0130",
        "登録番号",
        "T1234567890123",
        "領収証",
    ])

    _fix_company_name_merchant(extracted, ocr_text)

    assert extracted["merchant"] == "テストストア"


def test_visible_japanese_merchant_not_replaced_by_registration_number():
    from receipt_parser.pipeline_receipt import _fix_company_name_merchant

    extracted = {"merchant": "テストストア"}
    ocr_text = "\n".join([
        "くりいと",
        "テストストア (0940)38-0130",
        "登録番号",
        "T1234567890123",
        "領収証",
    ])

    _fix_company_name_merchant(extracted, ocr_text)

    assert extracted["merchant"] == "テストストア"


def test_phone_number_merchant_recovers_visible_header_brand():
    from receipt_parser.pipeline_receipt import _fix_company_name_merchant

    extracted = {"merchant": "TEL070-1234-5678"}
    ocr_text = "\n".join([
        "Baby & Kids",
        "テストストア",
        "ご購入店 テストモール店",
        "TEL070-1234-5678",
        "領収証",
    ])

    _fix_company_name_merchant(extracted, ocr_text)

    assert extracted["merchant"] == "テストストア"


def test_final_receipt_output_repairs_reject_invoice_registration_merchant():
    from receipt_parser.pipeline import _apply_final_receipt_output_repairs

    result = {
        "document_type": "receipt",
        "merchant": "T1234567890123",
        "line_items": [],
        "taxes": [],
    }
    ocr_text = "\n".join([
        "支店名",
        "テストストア (0940)38-0130",
        "登録番号",
        "T1234567890123",
        "領収証",
    ])
    trace = []

    _apply_final_receipt_output_repairs(result, ocr_text, mutation_trace=trace)

    assert result["merchant"] == "テストストア"
    company_events = [
        event for event in trace if event["stage"] == "company_name_merchant"
    ]
    assert company_events
    assert company_events[0]["owner_phase"] == "header_identity_repair"
    assert company_events[0]["owner_invariant"]
    assert company_events[0]["justification"]


def test_final_receipt_output_repairs_recover_visible_repeated_item_gap():
    from receipt_parser.pipeline import _apply_final_receipt_output_repairs

    result = {
        "document_type": "receipt",
        "merchant": "TEST",
        "total": 1650,
        "subtotal": 1500,
        "taxes": [{"rate": "10%", "label": "内税", "amount": 150}],
        "line_items": [
            {"description": "商品ア", "qty": 1, "unit_price": 550, "total": 550, "tax_category": "10%"},
            {"description": "商品イ", "qty": 1, "unit_price": 550, "total": 550, "tax_category": "10%"},
        ],
    }
    ocr_text = "\n".join([
        "TEST STORE",
        "商品ア",
        "¥550",
        "商品イ",
        "¥550",
        "商品ア",
        "¥550",
        "3点/合計",
        "¥1,650",
    ])

    _apply_final_receipt_output_repairs(result, ocr_text)

    assert [item["description"] for item in result["line_items"]] == ["商品ア", "商品ア", "商品イ"]
    assert [item["total"] for item in result["line_items"]] == [550, 550, 550]


def test_parent_company_header_prefers_consumer_katakana_store_brand():
    from receipt_parser.pipeline_receipt import _fix_company_name_merchant

    extracted = {"merchant": "PARENTCO"}
    ocr_text = "\n".join([
        "PARENTCO",
        "テストマート中央店",
        "TEL 0999-99-9999",
        "領収証",
        "テストマート運営株式会社",
    ])

    _fix_company_name_merchant(extracted, ocr_text)

    assert extracted["merchant"] == "テストマート"


def test_invalid_tax_annotation_merchant_recovers_explicit_business_name():
    from receipt_parser.pipeline_receipt import _fix_company_name_merchant

    for invalid_merchant in ["(消費税含む)", "223円)"]:
        extracted = {"merchant": invalid_merchant}
        ocr_text = "\n".join([
            "2026.05.07",
            "領収書",
            "¥2,460",
            "様",
            "(消費税含む)",
            "但し、上記の金額正に受領いたしました",
            "サービス大人",
            "合計 2,460円",
            "事業者名: テスト登山鉄道株式会社",
            "登録番号: T1234567890123",
        ])

        _fix_company_name_merchant(extracted, ocr_text)

        assert extracted["merchant"] == "テスト登山鉄道株式会社"


def test_invalid_tax_annotation_merchant_does_not_use_service_line_before_business_name():
    from receipt_parser.pipeline_receipt import _fix_company_name_merchant

    extracted = {"merchant": "223円)"}
    ocr_text = "\n".join([
        "2026.05.07",
        "領収書",
        "¥2,460",
        "様",
        "(消費税含む)",
        "但し、上記の金額正に受領いたしました",
        "サービス大人",
        "合計 2,460円",
        "事業者名: テスト登山鉄道株式会社",
        "登録番号: T1234567890123",
    ])

    _fix_company_name_merchant(extracted, ocr_text)

    assert extracted["merchant"] == "テスト登山鉄道株式会社"


def test_purchase_store_metadata_merchant_recovers_visible_header_brand():
    from receipt_parser.pipeline_receipt import _fix_company_name_merchant

    extracted = {"merchant": "ご購入店 テストモール中央店"}
    ocr_text = "\n".join([
        "Baby & Kids",
        "テストストア",
        "ご購入店 テストモール中央店",
        "TEL070-1234-5678",
        "領収証",
    ])

    _fix_company_name_merchant(extracted, ocr_text)

    assert extracted["merchant"] == "テストストア"


def test_short_logo_header_preferred_over_uppercase_category_subtitle():
    from receipt_parser.pipeline_receipt import _fix_company_name_merchant

    extracted = {"merchant": "RESTAURANTS"}
    ocr_text = "\n".join([
        "TK",
        "RESTAURANTS",
        "<領収書>",
        "TKレストラン 中央店",
        "TEL:000-000-0000",
    ])

    _fix_company_name_merchant(extracted, ocr_text)

    assert extracted["merchant"] == "TK"


def test_tagline_above_brand_with_romanized_line_prefers_brand_line():
    from receipt_parser.pipeline_receipt import _fix_company_name_merchant

    extracted = {"merchant": "牛カツ"}
    ocr_text = "\n".join([
        "牛カツ",
        "京都勝牛",
        "Gyukaten Kyoto Katsugyu",
        "中央店",
        "TEL 000-000-0000",
    ])

    _fix_company_name_merchant(extracted, ocr_text)

    assert extracted["merchant"] == "京都勝牛"


def test_parent_company_header_accepts_punctuated_store_line():
    from receipt_parser.pipeline_receipt import _fix_company_name_merchant

    extracted = {"merchant": "PARENTCO"}
    ocr_text = "\n".join([
        "PARENTCO",
        "テストマート中央店。",
        "TEL 0999-99-9999",
        "領収証",
        "テストマート運営株式会社",
    ])

    _fix_company_name_merchant(extracted, ocr_text)

    assert extracted["merchant"] == "テストマート"


def test_drop_non_product_removes_percent_inner_tax_marker_item():
    from receipt_parser.pipeline_receipt import _drop_non_product_line_items

    extracted = {
        "total": 2179,
        "line_items": [
            {"description": "ヤサイ", "qty": 1, "unit_price": 480, "total": 480},
            {"description": "( ※ 8% 内)", "qty": 1, "unit_price": 21, "total": 21},
        ],
    }

    _drop_non_product_line_items(extracted, "合計\n¥2,179")

    assert [item["description"] for item in extracted["line_items"]] == ["ヤサイ"]


def test_colon_split_product_name_rejoins_adjacent_ocr_prefix():
    from receipt_parser.pipeline_receipt import _fix_colon_split_product_names_from_ocr

    extracted = {
        "line_items": [
            {"description": "りんご", "qty": 1, "unit_price": 228, "total": 228},
        ]
    }
    ocr_text = "\n".join([
        "220210",
        "りんご",
        "蒟蒻畑:",
        "¥228",
    ])

    _fix_colon_split_product_names_from_ocr(extracted, ocr_text)

    assert extracted["line_items"][0]["description"] == "蒟蒻畑: りんご"


def test_bag_description_from_ocr_code_context_recovers_size_and_price():
    from receipt_parser.pipeline_receipt import _fix_bag_description_from_ocr_code_context

    extracted = {
        "line_items": [
            {"description": "レジ袋", "qty": 1, "unit_price": 21, "total": 21, "tax_category": "8%"},
        ]
    }
    ocr_text = "000500内レジ袋 4円\n¥4"

    _fix_bag_description_from_ocr_code_context(extracted, ocr_text)

    assert extracted["line_items"][0] == {
        "description": "レジ袋L",
        "qty": 1,
        "unit_price": 4.0,
        "total": 4.0,
        "tax_category": "10%",
    }


def test_barcode_qty_price_rows_replace_collapsed_retail_duplicates_when_balanced():
    from receipt_parser.pipeline_receipt import _replace_barcode_qty_price_rows_when_balanced

    extracted = {
        "total": 10970,
        "line_items": [
            {"description": "Wリブブラトップ", "qty": 1, "unit_price": 1990, "total": 1990},
            {"description": "K UT", "qty": 1, "unit_price": 990, "total": 990},
        ],
    }
    ocr_text = "\n".join([
        "WクシーニットT",
        "[11:37]",
        "2000219782528",
        "1 ¥1,990",
        "Wリブブラトップ",
        "2000193266465",
        "1 ¥1,990",
        "WクシーニットT",
        "2000214809329",
        "1 ¥1,990",
        "WクシーニットT",
        "2000220661157",
        "1 ¥1,990",
        "K UT",
        "2000213718943",
        "1 ¥990",
        "WクシーニットT",
        "1 ¥1,990",
        "¥30",
        "2000216056660",
        "ショッピングバッグ",
        "合計",
        "¥10,970",
    ])

    _replace_barcode_qty_price_rows_when_balanced(extracted, ocr_text)

    assert len(extracted["line_items"]) == 7
    assert sum(item["total"] for item in extracted["line_items"]) == 10970
    assert [item["description"] for item in extracted["line_items"]].count("WクシーニットT") == 4


def test_barcode_unit_qty_amount_stack_replaces_collapsed_items_when_balanced():
    from receipt_parser.pipeline_receipt import _replace_barcode_unit_qty_amount_stack_when_balanced

    extracted = {
        "total": 1504,
        "subtotal": 1368,
        "taxes": [{"label": "外税", "rate": "10%", "amount": 136}],
        "line_items": [
            {"description": "レギュラー", "qty": 2, "unit_price": 585, "total": 1170},
        ],
    }
    ocr_text = "\n".join([
        "0018 マジックバンド超薄型",
        "4905429012718",
        "¥585 2個",
        "0036 リング",
        "4909730105008",
        "(外税10.0%対象額",
        "¥1,170",
        "¥198",
        "小計",
        "3点",
        "¥1,368",
        "10.0% 消費税等",
        "¥136",
        "合計",
        "¥1,504",
    ])

    _replace_barcode_unit_qty_amount_stack_when_balanced(extracted, ocr_text)

    assert [item["description"] for item in extracted["line_items"]] == [
        "マジックバンド超薄型",
        "リング",
    ]
    assert [item["qty"] for item in extracted["line_items"]] == [2.0, 1.0]
    assert [item["unit_price"] for item in extracted["line_items"]] == [585.0, 198.0]
    assert [item["total"] for item in extracted["line_items"]] == [1170.0, 198.0]


def test_final_receipt_output_repairs_recover_barcode_unit_qty_amount_stack():
    from receipt_parser.pipeline import _apply_final_receipt_output_repairs

    result = {
        "document_type": "receipt",
        "total": 1504,
        "subtotal": 1368,
        "taxes": [{"label": "外税", "rate": "10%", "amount": 136}],
        "line_items": [
            {"description": "レギュラー", "qty": 2, "unit_price": 585, "total": 1170},
        ],
    }
    ocr_text = "\n".join([
        "0018 マジックバンド超薄型",
        "4905429012718",
        "¥585 2個",
        "0036 リング",
        "4909730105008",
        "(外税10.0%対象額",
        "¥1,170",
        "¥198",
        "小計",
        "3点",
        "¥1,368",
        "10.0% 消費税等",
        "¥136",
        "合計",
        "¥1,504",
    ])

    _apply_final_receipt_output_repairs(result, ocr_text)

    assert [item["description"] for item in result["line_items"]] == [
        "マジックバンド超薄型",
        "Oリング",
    ]
    assert [item["total"] for item in result["line_items"]] == [1170.0, 198.0]


def test_item_price_qty_rows_project_adjacent_descriptions_when_subtotal_balances():
    from receipt_parser.pipeline_receipt import _replace_item_price_qty_rows_when_balanced

    extracted = {
        "subtotal": 905,
        "total": 984,
        "taxes": [
            {"rate": "10%", "label": "外税", "amount": 70},
            {"rate": "8%", "label": "外税", "amount": 8},
        ],
        "line_items": [
            {"description": "袋", "qty": 1, "unit_price": 5, "total": 5, "tax_category": "10%"},
            {"description": "商品A", "qty": 2, "unit_price": 100, "total": 200, "tax_category": "10%"},
            {"description": "商品B", "qty": 2, "unit_price": 100, "total": 200, "tax_category": "10%"},
            {"description": "食品", "qty": 1, "unit_price": 100, "total": 100, "tax_category": "10%"},
        ],
    }
    ocr_text = "\n".join([
        "買物袋",
        "¥5外",
        "ケース",
        "¥300",
        "商品A",
        "(@100 ×2個)",
        "¥200外",
        "商品B",
        "¥200外",
        "(@100 ×2個)",
        "雑貨",
        "食品",
        "¥100外",
        "55",
        "¥100",
        "小計",
        "8点",
        "¥905",
        "10%税抜対象額",
        "¥705",
        "10%税額",
        "¥70",
        "8%税抜対象額",
        "¥100",
        "8",
        "合計",
        "¥983",
    ])

    _replace_item_price_qty_rows_when_balanced(extracted, ocr_text)

    assert extracted["line_items"] == [
        {"description": "買物袋", "qty": 1.0, "unit_price": 5.0, "total": 5.0, "tax_category": "10%", "discount": 0, "discount_rate": ""},
        {"description": "ケース", "qty": 1.0, "unit_price": 300.0, "total": 300.0, "tax_category": "10%", "discount": 0, "discount_rate": ""},
        {"description": "商品A", "qty": 2.0, "unit_price": 100.0, "total": 200.0, "tax_category": "10%", "discount": 0, "discount_rate": ""},
        {"description": "商品B", "qty": 2.0, "unit_price": 100.0, "total": 200.0, "tax_category": "10%", "discount": 0, "discount_rate": ""},
        {"description": "雑貨", "qty": 1.0, "unit_price": 100.0, "total": 100.0, "tax_category": "10%", "discount": 0, "discount_rate": ""},
        {"description": "食品 55", "qty": 1.0, "unit_price": 100.0, "total": 100.0, "tax_category": "8%", "discount": 0, "discount_rate": ""},
    ]


def test_qty_context_does_not_attach_next_item_quantity_detail():
    from receipt_parser.pipeline_receipt import _fix_qty_context_and_reduced_rate_from_ocr

    extracted = {
        "line_items": [
            {"description": "デスク整理 L", "qty": 2, "unit_price": 100, "total": 200, "tax_category": "10%"},
            {"description": "デスク整理 S", "qty": 2, "unit_price": 100, "total": 200, "tax_category": "10%"},
            {"description": "サントリー天然水 55", "qty": 1, "unit_price": 100, "total": 100, "tax_category": "10%"},
            {"description": "トラック荷台 アリさん", "qty": 1, "unit_price": 100, "total": 100, "tax_category": "8%"},
        ]
    }
    ocr_text = "\n".join([
        "DAISO",
        "デスク整理 L",
        "¥100外",
        "デスク整理 S",
        "(@100 ×2個)",
        "¥200外",
        "トラック荷台 アリさん",
        "¥100",
        "サントリー天然水",
        "55",
        "¥100",
        "10%税抜対象額",
        "¥2,005",
        "8%税抜対象額",
        "¥100",
    ])

    _fix_qty_context_and_reduced_rate_from_ocr(extracted, ocr_text)

    assert extracted["line_items"][0]["qty"] == 1
    assert extracted["line_items"][0]["total"] == 100
    assert extracted["line_items"][1]["qty"] == 2
    assert extracted["line_items"][2]["tax_category"] == "8%"
    assert extracted["line_items"][3]["tax_category"] == "8%"


def test_qty_context_repair_ignores_receipts_without_structural_evidence():
    from receipt_parser.pipeline_receipt import _fix_qty_context_and_reduced_rate_from_ocr

    extracted = {
        "line_items": [
            {"description": "ケース", "qty": 2, "unit_price": 100, "total": 200, "tax_category": "8%"},
        ]
    }
    before = [dict(item) for item in extracted["line_items"]]
    ocr_text = "\n".join([
        "ケース",
        "¥100",
        "小計",
        "¥200",
    ])

    _fix_qty_context_and_reduced_rate_from_ocr(extracted, ocr_text)

    assert extracted["line_items"] == before


def test_o_ring_description_repaired_from_jan_context():
    from receipt_parser.pipeline_receipt import _fix_o_ring_descriptions_from_ocr

    extracted = {
        "line_items": [
            {"description": "レギュラー", "qty": 5, "unit_price": 198, "total": 990},
            {"description": "リング", "qty": 1, "unit_price": 198, "total": 198},
        ]
    }
    ocr_text = "\n".join([
        "20036 リング",
        "4909730105008",
        "¥198 5個",
    ])

    _fix_o_ring_descriptions_from_ocr(extracted, ocr_text)

    assert [item["description"] for item in extracted["line_items"]] == ["Oリング", "Oリング"]


def test_final_receipt_output_repairs_preserve_o_ring_jan_context():
    from receipt_parser.pipeline import _apply_final_receipt_output_repairs

    result = {
        "document_type": "receipt",
        "total": 1089,
        "line_items": [
            {"description": "レギュラー", "qty": 5, "unit_price": 198, "total": 990},
        ],
    }
    ocr_text = "\n".join([
        "20036 リング",
        "4909730105008",
        "¥198 5個",
        "¥990",
        "会員ランク",
        "レギュラー",
    ])

    _apply_final_receipt_output_repairs(result, ocr_text)

    assert result["line_items"][0]["description"] == "Oリング"


def test_repeated_item_from_gap_recovers_visible_duplicate():
    from receipt_parser.pipeline_receipt import _recover_repeated_item_from_gap

    extracted = {
        "total": 1650,
        "subtotal": 1650,
        "line_items": [
            {"description": "ディズニーブロックシール", "qty": 1, "unit_price": 550, "total": 550},
            {"description": "ぷくぷくシール", "qty": 1, "unit_price": 550, "total": 550},
        ],
    }
    ocr_text = "\n".join([
        "HAPNS サンリブくりえいと宗像店",
        "ディズニーブロックシール",
        "¥550",
        "ぷくぷくシール",
        "¥550",
        "ディズニーブロックシール",
        "¥550",
        "合計",
        "¥1,650",
    ])

    _recover_repeated_item_from_gap(extracted, ocr_text)

    assert [item["description"] for item in extracted["line_items"]].count("ディズニーブロックシール") == 2
    assert sum(item["total"] for item in extracted["line_items"]) == 1650


def test_duplicate_description_repair_ignores_inline_price_suffix_candidate():
    from receipt_parser.pipeline_receipt import _fix_duplicate_descriptions_from_ocr

    extracted = {
        "line_items": [
            {
                "description": "商品ア",
                "qty": 2,
                "unit_price": 925,
                "total": 1294,
                "discount": 556,
                "discount_rate": "30%",
            },
            {
                "description": "商品ア",
                "qty": 1,
                "unit_price": 840,
                "total": 588,
                "discount": 252,
                "discount_rate": "30%",
            },
        ]
    }
    ocr_text = "\n".join([
        "商品ア 1,850",
        "<2個 X 単925)",
        "割引",
        "30%",
        "-556",
        "商品ア",
        "840",
        "割引",
        "30%",
        "-252",
    ])

    _fix_duplicate_descriptions_from_ocr(extracted, ocr_text)

    assert extracted["line_items"][0]["description"] == "商品ア"
    assert extracted["line_items"][0]["total"] == 1294


def test_duplicate_description_replacement_skips_quantity_notation_rows():
    from receipt_parser.pipeline_receipt import _fix_duplicate_descriptions_from_ocr

    extracted = {
        "line_items": [
            {"description": "商品ア", "qty": 1, "unit_price": 658, "total": 658},
            {"description": "商品イ", "qty": 1, "unit_price": 599, "total": 599},
            {"description": "商品ウ", "qty": 1, "unit_price": 499, "total": 499},
            {"description": "商品ウ", "qty": 1, "unit_price": 329, "total": 329},
        ]
    }
    ocr_text = "\n".join([
        "商品ア",
        "単329×2個",
        "¥658",
        "商品イ",
        "¥599",
        "商品ウ",
        "¥499",
        "商品エ",
        "¥329",
    ])

    _fix_duplicate_descriptions_from_ocr(extracted, ocr_text)

    assert extracted["line_items"][2]["description"] == "商品ウ"
    assert extracted["line_items"][3]["description"] == "商品エ"


def test_line_item_cleanup_anchors_duplicates_before_price_repair():
    from receipt_parser.pipeline_receipt import _fix_line_items

    extracted = {
        "total": 5248,
        "subtotal": 4816,
        "taxes": [
            {"rate": "10%", "label": "外税", "amount": 236},
            {"rate": "8%", "label": "外税", "amount": 196},
        ],
        "line_items": [
            {"description": "コドモトタベル ホシイモ", "qty": 1, "unit_price": 658, "total": 658},
            {"description": "サッシロック2コグミ", "qty": 2, "unit_price": 329, "total": 599},
            {"description": "ベビーハブラシジドウシタ", "qty": 1, "unit_price": 499, "total": 499},
            {"description": "ベビーハブラシジドウシタ", "qty": 1, "unit_price": 329, "total": 329},
        ],
    }
    ocr_text = "\n".join([
        "0030コドモトタベル ホシイモ",
        "単329×2個",
        "¥658",
        "20060Wサッシロック2コグミ 外",
        "¥599",
        "0012ベビーハブラシジドウシタ",
        "¥499",
        "0011W) RCBオデカケカレーバー",
        "¥329",
        "小計",
        "¥4,816",
    ])

    _fix_line_items(extracted, ocr_text)

    descriptions = [item["description"] for item in extracted["line_items"]]
    sashi = next(item for item in extracted["line_items"] if "サッシロック" in item["description"])
    assert "単329×2個" not in descriptions
    assert "RCBオデカケカレーバー" in descriptions
    assert sashi["qty"] == 1.0
    assert sashi["unit_price"] == 599
    assert sashi["total"] == 599


def test_qty_unit_total_block_recovers_empty_salon_item():
    from receipt_parser.pipeline_receipt import _recover_qty_unit_total_item_from_empty_extraction

    extracted = {
        "merchant": "Grand Joul",
        "total": 1100,
        "line_items": [],
    }
    ocr_text = "\n".join([
        "Grand Joul",
        "ヘ",
        "2個 x 単550",
        "¥1,100円",
        "小計",
        "¥1,100",
    ])

    _recover_qty_unit_total_item_from_empty_extraction(extracted, ocr_text)

    assert extracted["line_items"] == [{
        "description": "ヘア",
        "qty": 2.0,
        "unit_price": 550.0,
        "total": 1100.0,
        "tax_category": "10%",
        "discount": 0,
        "discount_rate": "",
    }]


def test_stacked_name_price_rows_use_quantity_detail_when_balanced():
    from receipt_parser.pipeline_receipt import _replace_stacked_name_price_rows_when_balanced

    extracted = {
        "document_type": "receipt",
        "merchant": "Convenience Store",
        "total": 992,
        "line_items": [
            {"description": "レジ袋", "qty": 1, "unit_price": 5, "total": 5, "tax_category": "10%"},
            {"description": "@213× 2点", "qty": 1, "unit_price": 426, "total": 426, "tax_category": "8%"},
        ],
    }
    ocr_text = "\n".join([
        "Convenience Store",
        "領 収 証",
        "◎コクと旨みのたまご 1",
        "レジ袋弁当大バイオマス",
        "アルフォートミニチョコ",
        "上白糖 500G",
        "¥358軽",
        "¥5",
        "¥203軽",
        "@213× 2点",
        "¥426軽",
        "合計",
        "¥992",
    ])

    _replace_stacked_name_price_rows_when_balanced(extracted, ocr_text)

    assert extracted["line_items"] == [
        {
            "description": "コクと旨みのたまご1",
            "qty": 1.0,
            "unit_price": 358.0,
            "total": 358.0,
            "tax_category": "8%",
            "discount": 0,
            "discount_rate": "",
        },
        {
            "description": "レジ袋弁当大バイオマス",
            "qty": 1.0,
            "unit_price": 5.0,
            "total": 5.0,
            "tax_category": "10%",
            "discount": 0,
            "discount_rate": "",
        },
        {
            "description": "アルフォートミニチョコ",
            "qty": 1.0,
            "unit_price": 203.0,
            "total": 203.0,
            "tax_category": "8%",
            "discount": 0,
            "discount_rate": "",
        },
        {
            "description": "上白糖500G",
            "qty": 2.0,
            "unit_price": 213.0,
            "total": 426.0,
            "tax_category": "8%",
            "discount": 0,
            "discount_rate": "",
        },
    ]


def test_stacked_name_price_rows_handle_summary_interrupted_item_columns():
    from receipt_parser.pipeline_receipt import _replace_stacked_name_price_rows_when_balanced

    extracted = {
        "document_type": "receipt",
        "merchant": "Convenience Store",
        "total": 430,
        "amount_paid": 430,
        "line_items": [{"description": "dummy", "qty": 1, "unit_price": 430, "total": 430}],
    }
    ocr_text = "\n".join([
        "Convenience Store",
        "領 収",
        "◎ホットサンド",
        "スープヌードル",
        "フライドチキン",
        "¥389軽",
        "¥198軽",
        "¥198",
        "(10% 対象",
        "(内消費税等",
        "(8% 対象",
        "レジ袋弁当大バイオマス",
        "合計",
        "タコスサンド",
        "ビタミンゼリー",
        "¥430軽",
        "¥183軽",
        "¥5",
        "¥1,403",
    ])

    _replace_stacked_name_price_rows_when_balanced(extracted, ocr_text)

    assert [item["description"] for item in extracted["line_items"]] == [
        "ホットサンド",
        "スープヌードル",
        "フライドチキン",
        "タコスサンド",
        "ビタミンゼリー",
        "レジ袋弁当大バイオマス",
    ]
    assert [item["total"] for item in extracted["line_items"]] == [
        389.0,
        198.0,
        198.0,
        430.0,
        183.0,
        5.0,
    ]
    assert extracted["line_items"][-1]["tax_category"] == "10%"
    assert extracted["total"] == 1403.0
    assert extracted["amount_paid"] == 1403.0


def test_stacked_name_price_rows_balance_subtotal_and_rate_bases():
    from receipt_parser.pipeline_receipt import _replace_stacked_name_price_rows_when_balanced

    extracted = {
        "document_type": "receipt",
        "merchant": "Generic Store",
        "subtotal": 1839,
        "total": 2018,
        "amount_paid": 2018,
        "taxes": [
            {"rate": "10%", "label": "内税", "amount": 163},
            {"rate": "8%", "label": "内税", "amount": 16},
        ],
        "line_items": [],
    }
    ocr_text = "\n".join([
        "領収証",
        "内レジ袋 LL",
        "¥6",
        "内クリーナー",
        "¥498",
        "内洗剤A",
        "内洗剤B",
        "内 お惣菜A",
        "*お惣菜B",
        "¥498",
        "¥798",
        "¥109",
        "¥109",
        "(10%対象 ¥1,800 内税 ¥163)",
        "(08%対象 ¥218 内税 ¥16)",
        "*は軽減税率8%適用商品",
        "合計",
        "¥2,018",
        "6点買",
    ])

    _replace_stacked_name_price_rows_when_balanced(extracted, ocr_text)

    assert [item["description"] for item in extracted["line_items"]] == [
        "レジ袋LL",
        "クリーナー",
        "洗剤A",
        "洗剤B",
        "お惣菜A",
        "お惣菜B",
    ]
    assert [item["total"] for item in extracted["line_items"]] == [6.0, 498.0, 498.0, 798.0, 109.0, 109.0]
    assert [item["tax_category"] for item in extracted["line_items"]] == ["10%", "10%", "10%", "10%", "8%", "8%"]
    assert extracted["total"] == 2018


def test_stacked_name_price_rows_handles_code_prefixed_rows_and_quantity_count():
    from receipt_parser.pipeline_receipt import _replace_stacked_name_price_rows_when_balanced

    extracted = {
        "document_type": "receipt",
        "merchant": "Generic Store",
        "subtotal": 4816,
        "total": 5248,
        "amount_paid": 5250,
        "taxes": [
            {"rate": "10%", "label": "外税", "amount": 236},
            {"rate": "8%", "label": "外税", "amount": 196},
        ],
        "line_items": [],
    }
    ocr_text = "\n".join([
        "領収証",
        "4987244186614",
        "特",
        "0011W) 商品フード ¥1,139",
        "4562130920598",
        "20060商品ロック",
        "単629×2個 外 ¥1,258",
        "4941983022718",
        "0030商品スナック",
        "単329×2個",
        "4969133325953",
        "20060商品ラッチ 外",
        "4589898190100",
        "0012商品ブラシ",
        "4987244194183",
        "特",
        "0011W) 商品ミールA",
        "4987244194190",
        "特",
        "0011W) 商品ミールB",
        "4571138755224",
        "0025レジフクロ 外",
        "小計",
        "¥658",
        "¥599",
        "¥499",
        "¥329",
        "¥329",
        "S",
        "¥5",
        "¥4,816",
        "外税10%対象額",
        "¥2,361",
        "10%外税額",
        "¥236",
        "外税8%対象額",
        "¥2,455",
        "8%外税額",
        "¥196",
        "合計",
        "¥5,248",
        "10点買",
    ])

    _replace_stacked_name_price_rows_when_balanced(extracted, ocr_text)

    assert [item["description"] for item in extracted["line_items"]] == [
        "商品フード",
        "商品ロック",
        "商品スナック",
        "商品ラッチ",
        "商品ブラシ",
        "商品ミールA",
        "商品ミールB",
        "レジフクロ",
    ]
    assert [item["qty"] for item in extracted["line_items"]] == [1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert [item["unit_price"] for item in extracted["line_items"]] == [
        1139.0,
        629.0,
        329.0,
        599.0,
        499.0,
        329.0,
        329.0,
        5.0,
    ]
    assert [item["total"] for item in extracted["line_items"]] == [
        1139.0,
        1258.0,
        658.0,
        599.0,
        499.0,
        329.0,
        329.0,
        5.0,
    ]
    assert [item["tax_category"] for item in extracted["line_items"]] == [
        "8%",
        "10%",
        "8%",
        "10%",
        "10%",
        "8%",
        "8%",
        "10%",
    ]


def test_stacked_name_price_rows_handles_prices_before_following_descriptions():
    from receipt_parser.pipeline_receipt import _replace_stacked_name_price_rows_when_balanced

    extracted = {
        "document_type": "receipt",
        "merchant": "Generic Store",
        "subtotal": 4816,
        "total": 5248,
        "amount_paid": 5250,
        "taxes": [
            {"rate": "10%", "label": "外税", "amount": 236},
            {"rate": "8%", "label": "外税", "amount": 196},
        ],
        "line_items": [],
    }
    ocr_text = "\n".join([
        "領収証",
        "4987244186614",
        "特",
        "0011W) 商品フード ¥1,139",
        "4562130920598",
        "20060商品ロック",
        "単629×2個 外 ¥1,258",
        "4941983022718",
        "0030商品スナック",
        "単329×2個",
        "4969133325953",
        "20060商品ラッチ 外",
        "4589898190100",
        "※",
        "¥658",
        "¥599",
        "¥499",
        "特",
        "¥329",
        "特",
        "0012商品ブラシ",
        "4987244194183",
        "0011W) 商品ミールA",
        "4987244194190",
        "0011W) 商品ミールB",
        "4571138755224",
        "0025レジフクロ 外",
        "小計",
        "¥329",
        "S",
        "¥5",
        "¥4,816",
        "外税10%対象額",
        "¥2,361",
        "10%外税額",
        "¥236",
        "外税8%対象額",
        "¥2,455",
        "8%外税額",
        "¥196",
        "合計",
        "¥5,248",
        "取引No1579 10点買",
    ])

    _replace_stacked_name_price_rows_when_balanced(extracted, ocr_text)

    assert [item["description"] for item in extracted["line_items"]] == [
        "商品フード",
        "商品ロック",
        "商品スナック",
        "商品ラッチ",
        "商品ブラシ",
        "商品ミールA",
        "商品ミールB",
        "レジフクロ",
    ]
    assert [item["qty"] for item in extracted["line_items"]] == [1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert [item["total"] for item in extracted["line_items"]] == [
        1139.0,
        1258.0,
        658.0,
        599.0,
        499.0,
        329.0,
        329.0,
        5.0,
    ]
    assert [item["tax_category"] for item in extracted["line_items"]] == [
        "8%",
        "10%",
        "8%",
        "10%",
        "10%",
        "8%",
        "8%",
        "10%",
    ]


def test_printed_item_sum_total_repairs_close_financial_drift():
    from receipt_parser.pipeline_receipt import _prefer_printed_item_sum_total_when_balanced

    extracted = {
        "total": 8453,
        "amount_paid": 8453,
        "subtotal": 7684,
        "taxes": [{"rate": "10%", "label": "内税", "amount": 769}],
        "line_items": [
            {"description": "工具", "qty": 1, "unit_price": 698, "total": 698},
            {"description": "調理器具", "qty": 1, "unit_price": 2780, "total": 2780},
            {"description": "レジ袋", "qty": 1, "unit_price": 5, "total": 5},
            {"description": "タオル", "qty": 1, "unit_price": 298, "total": 298},
            {"description": "台車", "qty": 1, "unit_price": 3880, "total": 3880},
            {"description": "冷却用品", "qty": 1, "unit_price": 798, "total": 798},
        ],
    }
    ocr_text = "\n".join([
        "お買上明細",
        "小計金額",
        "¥8,459",
        "合計",
        "¥8,459",
        "(10%対象消費税",
        "¥769)",
    ])

    _prefer_printed_item_sum_total_when_balanced(extracted, ocr_text)

    assert extracted["total"] == 8459.0
    assert extracted["amount_paid"] == 8459.0
    assert extracted["subtotal"] == 7690.0


def test_printed_summary_total_uses_tax_balanced_labeled_total_and_points():
    from receipt_parser.pipeline_receipt import _restore_printed_summary_total_when_tax_balanced

    extracted = {
        "subtotal": 3381,
        "total": 3705,
        "amount_paid": 3213,
        "points_used": 492,
        "taxes": [
            {"rate": "8%", "label": "内税", "amount": 180},
            {"rate": "10%", "label": "内税", "amount": 144},
        ],
        "line_items": [
            {"description": "A", "total": 2025},
            {"description": "B", "total": 348},
            {"description": "C", "total": 1332},
        ],
    }
    ocr_text = "\n".join([
        "小計",
        "¥3,705",
        "※8%内税対象",
        "¥2,437",
        "( ※ 8% 内)",
        "¥180",
        "10%内税対象",
        "¥1,592",
        "(10%内)",
        "¥144",
        "(税合計",
        "¥324)",
        "合計",
        "¥4,029",
        "dポイント利用",
        "¥492",
    ])

    _restore_printed_summary_total_when_tax_balanced(extracted, ocr_text)

    assert extracted["subtotal"] == 3705.0
    assert extracted["total"] == 4029.0
    assert extracted["amount_paid"] == 3537.0


def test_printed_summary_total_handles_split_subtotal_label():
    from receipt_parser.pipeline_receipt import _restore_printed_summary_total_when_tax_balanced

    extracted = {
        "subtotal": 4814,
        "total": 5246,
        "amount_paid": 5246,
        "taxes": [
            {"rate": "10%", "label": "外税", "amount": 236},
            {"rate": "8%", "label": "外税", "amount": 196},
        ],
        "line_items": [{"description": "A", "total": 4816}],
    }
    ocr_text = "\n".join([
        "小",
        "計",
        "¥4,816",
        "外税10%対象額",
        "¥2,361",
        "10%外税額",
        "¥236",
        "外税8%対象額",
        "¥2,455",
        "8%外税額",
        "¥196",
        "合計",
        "¥5,248",
    ])

    _restore_printed_summary_total_when_tax_balanced(extracted, ocr_text)

    assert extracted["subtotal"] == 4816.0
    assert extracted["total"] == 5248.0
    assert extracted["amount_paid"] == 5248.0


def test_split_external_tax_amount_labels_restore_financial_summary():
    from receipt_parser.pipeline_receipt import (
        _restore_printed_external_tax_amounts,
        _restore_printed_summary_total_when_tax_balanced,
    )

    extracted = {
        "subtotal": 4204,
        "total": 4816,
        "amount_paid": 4816,
        "taxes": [
            {"rate": "10%", "label": "外税", "amount": 848},
            {"rate": "8%", "label": "外税", "amount": 196},
        ],
        "line_items": [
            {"description": "A", "total": 1139},
            {"description": "B", "total": 1258},
            {"description": "C", "total": 658},
            {"description": "D", "total": 599},
            {"description": "E", "total": 499},
            {"description": "F", "total": 329},
            {"description": "G", "total": 329},
            {"description": "H", "total": 5},
        ],
    }
    ocr_text = "\n".join([
        "小計",
        "¥4,816",
        "外税10%対象額",
        "¥2,361",
        "10%外税額",
        "¥236",
        "外税8%対象額",
        "¥2,455",
        "8%外税額",
        "¥196",
        "合計",
        "¥5,248",
    ])

    _restore_printed_external_tax_amounts(extracted, ocr_text)
    _restore_printed_summary_total_when_tax_balanced(extracted, ocr_text)

    assert extracted["taxes"] == [
        {"rate": "10%", "label": "外税", "amount": 236.0},
        {"rate": "8%", "label": "外税", "amount": 196.0},
    ]
    assert extracted["subtotal"] == 4816.0
    assert extracted["total"] == 5248.0
    assert extracted["amount_paid"] == 5248.0


def test_printed_summary_total_repairs_subtotal_when_total_already_matches():
    from receipt_parser.pipeline_receipt import _restore_printed_summary_total_when_tax_balanced

    extracted = {
        "subtotal": 4204,
        "total": 5248,
        "amount_paid": 5248,
        "taxes": [
            {"rate": "10%", "label": "外税", "amount": 236},
            {"rate": "8%", "label": "外税", "amount": 196},
        ],
        "line_items": [{"description": "A", "total": 4816}],
    }
    ocr_text = "\n".join([
        "小計",
        "¥4,816",
        "10%外税額",
        "¥236",
        "8%外税額",
        "¥196",
        "合計",
        "¥5,248",
    ])

    _restore_printed_summary_total_when_tax_balanced(extracted, ocr_text)

    assert extracted["subtotal"] == 4816.0
    assert extracted["total"] == 5248.0


def test_printed_summary_total_handles_interleaved_tax_base_before_total_value():
    from receipt_parser.pipeline_receipt import _restore_printed_summary_total_when_tax_balanced

    extracted = {
        "subtotal": 2111,
        "total": 2111,
        "amount_paid": 2111,
        "taxes": [{"rate": "8%", "label": "外税", "amount": 168}],
        "line_items": [{"description": "A", "total": 2111}],
    }
    ocr_text = "\n".join([
        "小計",
        "税率 8% 課税対象額",
        "¥199",
        "¥159",
        "¥2,111",
        "¥2,274",
        "税率 8%税額",
        "¥168",
        "計",
        "税率10%課税対象額",
        "合計",
        "¥5",
        "¥2,279",
        "現計",
        "¥2,279",
        "お釣り",
        "¥0",
    ])

    _restore_printed_summary_total_when_tax_balanced(extracted, ocr_text)

    assert extracted["subtotal"] == 2111.0
    assert extracted["total"] == 2279.0
    assert extracted["amount_paid"] == 2279.0


def test_explicit_tax_amount_lines_drop_target_only_rates_before_summary_repair():
    from receipt_parser.pipeline_receipt import (
        _restore_printed_external_tax_amounts,
        _restore_explicit_tax_rate_amount_lines,
        _restore_printed_summary_total_when_tax_balanced,
    )

    extracted = {
        "subtotal": 1701,
        "total": 2111,
        "amount_paid": 2111,
        "taxes": [
            {"rate": "10%", "label": "外税", "amount": 1},
            {"rate": "8%", "label": "外税", "amount": 409},
        ],
        "line_items": [
            {"description": "A", "total": 2106, "tax_category": "8%"},
            {"description": "レジ袋", "total": 5, "tax_category": "10%"},
        ],
    }
    ocr_text = "\n".join([
        "業務スーパー",
        "小計",
        "税率 8% 課税対象額",
        "¥199",
        "¥159",
        "¥2,111",
        "¥2,274",
        "税率 8%税額",
        "¥168",
        "計",
        "税率10%課税対象額",
        "合計",
        "¥5",
        "¥2,279",
        "現計",
        "¥2,279",
        "お釣り",
        "¥0",
    ])

    _restore_explicit_tax_rate_amount_lines(extracted, ocr_text)
    _restore_printed_summary_total_when_tax_balanced(extracted, ocr_text)
    _restore_printed_external_tax_amounts(extracted, ocr_text)
    _restore_explicit_tax_rate_amount_lines(extracted, ocr_text)

    assert extracted["taxes"] == [{"rate": "8%", "label": "外税", "amount": 168.0}]
    assert extracted["subtotal"] == 2111.0
    assert extracted["total"] == 2279.0
    assert extracted["amount_paid"] == 2279.0


def test_final_receipt_output_repairs_apply_stacked_name_price_rows():
    from receipt_parser.pipeline import _apply_final_receipt_output_repairs

    result = {
        "document_type": "receipt",
        "merchant": "Convenience Store",
        "total": 992,
        "line_items": [
            {"description": "レジ袋弁当大バイオマス", "qty": 1, "unit_price": 5, "total": 5, "tax_category": "10%"},
            {"description": "アルフォートミニチョコ", "qty": 1, "unit_price": 203, "total": 203, "tax_category": "10%"},
            {"description": "上白糖 500G", "qty": 1, "unit_price": None, "total": 203, "tax_category": "8%"},
            {"description": "@213× 2点", "qty": 1, "unit_price": 426, "total": 426, "tax_category": "8%"},
        ],
    }
    ocr_text = "\n".join([
        "Convenience Store",
        "領 収 証",
        "◎コクと旨みのたまご 1",
        "レジ袋弁当大バイオマス",
        "アルフォートミニチョコ",
        "上白糖 500G",
        "¥358軽",
        "¥5",
        "¥203軽",
        "@213× 2点",
        "¥426軽",
        "合計",
        "¥992",
    ])

    _apply_final_receipt_output_repairs(result, ocr_text)

    assert [item["total"] for item in result["line_items"]] == [358.0, 5.0, 203.0, 426.0]
    assert result["line_items"][3]["qty"] == 2.0
    assert result["line_items"][3]["unit_price"] == 213.0


def test_final_receipt_output_repairs_restore_standalone_discounted_qty_gross():
    from receipt_parser.pipeline import _apply_final_receipt_output_repairs

    result = {
        "document_type": "receipt",
        "line_items": [
            {
                "description": "TV減の恵みきざみねぎ",
                "qty": 2,
                "unit_price": 98,
                "total": 98,
                "tax_category": "8%",
                "discount": 98,
                "discount_rate": "50%",
            }
        ],
    }
    ocr_text = "\n".join([
        "TV減の恵みきざみねぎ",
        "196",
        "(2個 X 単98)",
        "割引",
        "50%",
        "-98",
    ])

    _apply_final_receipt_output_repairs(result, ocr_text)

    assert result["line_items"][0]["qty"] == 2
    assert result["line_items"][0]["unit_price"] == 196
    assert result["line_items"][0]["total"] == 98


def test_dense_sequence_repair_uses_tax_bases_for_percent_marker_price_ocr():
    from receipt_parser.pipeline_receipt import _replace_dense_sequence_rows_when_balanced

    extracted = {
        "subtotal": 426,
        "total": 460,
        "taxes": [{"rate": "8%", "label": "外税", "amount": 33}],
        "line_items": [
            {"description": "商品甲", "qty": 1, "unit_price": 100, "total": 100, "tax_category": "8%"},
            {"description": "商品乙", "qty": 1, "unit_price": 200, "total": 200, "tax_category": "8%"},
            {"description": "商品乙", "qty": 1, "unit_price": 200, "total": 200, "tax_category": "8%"},
            {"description": "商品丙", "qty": 1, "unit_price": 50, "total": 50, "tax_category": "8%"},
            {"description": "商品丁", "qty": 1, "unit_price": 60, "total": 60, "tax_category": "8%"},
            {"description": "有料レジ袋", "qty": 1, "unit_price": 8, "total": 8, "tax_category": "10%"},
        ],
    }
    ocr_text = "\n".join([
        "2026/4/14",
        "商品甲",
        "100%",
        "商品乙",
        "200*",
        "商品丙",
        "50*",
        "商品丁",
        "60*",
        "有料レジ袋",
        "8",
        "小計",
        "¥426",
        "外税 8% 対象額",
        "¥418",
        "外税8%",
        "¥33",
        "外税10%対象額",
        "¥8",
        "外税10%",
        "¥0",
        "合計",
        "¥459",
        "お買上商品数:4",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, ocr_text)

    assert [item["total"] for item in extracted["line_items"]] == [108.0, 200.0, 50.0, 60.0, 8.0]
    assert sum(item["total"] for item in extracted["line_items"]) == 426.0
    assert [item["tax_category"] for item in extracted["line_items"]] == ["8%", "8%", "8%", "8%", "10%"]


def test_final_receipt_output_repairs_finish_with_dense_sequence_reconciliation():
    from receipt_parser.pipeline import _apply_final_receipt_output_repairs

    result = {
        "document_type": "receipt",
        "subtotal": 426,
        "total": 460,
        "taxes": [{"rate": "8%", "label": "外税", "amount": 33}],
        "line_items": [
            {"description": "商品甲", "qty": 1, "unit_price": 100, "total": 100, "tax_category": "8%"},
            {"description": "商品乙", "qty": 1, "unit_price": 200, "total": 200, "tax_category": "8%"},
            {"description": "商品乙", "qty": 1, "unit_price": 200, "total": 200, "tax_category": "8%"},
            {"description": "商品丙", "qty": 1, "unit_price": 50, "total": 50, "tax_category": "8%"},
            {"description": "商品丁", "qty": 1, "unit_price": 60, "total": 60, "tax_category": "8%"},
            {"description": "有料レジ袋", "qty": 1, "unit_price": 8, "total": 8, "tax_category": "10%"},
        ],
    }
    ocr_text = "\n".join([
        "2026/4/14",
        "商品甲",
        "100%",
        "商品乙",
        "200*",
        "商品丙",
        "50*",
        "商品丁",
        "60*",
        "有料レジ袋",
        "8",
        "小計",
        "¥426",
        "外税 8% 対象額",
        "¥418",
        "外税8%",
        "¥33",
        "外税10%対象額",
        "¥8",
        "外税10%",
        "¥0",
        "合計",
        "¥459",
        "お買上商品数:4",
    ])

    _apply_final_receipt_output_repairs(result, ocr_text)

    assert [item["total"] for item in result["line_items"]] == [108.0, 200.0, 50.0, 60.0, 8.0]


def test_final_receipt_output_repairs_finish_with_campaign_discount_stream():
    from receipt_parser.pipeline import _apply_final_receipt_output_repairs

    result = {
        "document_type": "receipt",
        "subtotal": 2112,
        "taxes": [
            {"rate": "8%", "label": "外税", "amount": 83},
            {"rate": "10%", "label": "外税", "amount": 0},
        ],
        "line_items": [
            {"description": "会員様割引5%", "qty": 1, "unit_price": 43, "total": 43, "tax_category": "8%"},
            {"description": "有料袋", "qty": 1, "unit_price": 43, "total": 43, "tax_category": "10%"},
            {"description": "非課税サービス 652", "qty": 1, "unit_price": 652, "total": 652, "tax_category": "0%"},
            {"description": "商品ア", "qty": 1, "unit_price": 998, "total": 948, "tax_category": "10%", "discount": 50, "discount_rate": "5%"},
            {"description": "商品イ", "qty": 1, "unit_price": 100, "total": 95, "tax_category": "10%", "discount": 5, "discount_rate": "5%"},
            {"description": "商品ウ", "qty": 1, "unit_price": 621, "total": 412, "tax_category": "10%", "discount": 209, "discount_rate": "30%"},
        ],
    }
    ocr_text = "\n".join([
        "テスト店",
        "2026/3/30(月) 22:19",
        "有料袋",
        "0909143",
        "非課税サービス 652非",
        "5除",
        "商品ア",
        "998",
        "会員様割引5%",
        "-50",
        "商品イ",
        "100*",
        "会員様割引5%",
        "-5",
        "商品ウ",
        "割引",
        "30%",
        "会員様割引5%",
        "621*",
        "-187",
        "-22",
        "小計",
        "¥2,112",
        "外税8%対象額",
        "¥1,040",
        "外税8%",
        "¥83",
        "外税10%対象額",
        "¥5",
        "外税10%",
        "¥0",
        "お買上商品数:5",
    ])

    _apply_final_receipt_output_repairs(result, ocr_text)

    assert [item["description"] for item in result["line_items"]] == [
        "有料袋",
        "非課税サービス",
        "商品ア",
        "商品イ",
        "商品ウ",
    ]
    assert [item["total"] for item in result["line_items"]] == [5.0, 652.0, 948.0, 95.0, 412.0]
    assert all(item["description"] != "会員様割引5%" for item in result["line_items"])


def test_final_receipt_output_repairs_clear_discount_before_items_own_price():
    from receipt_parser.pipeline import _apply_final_receipt_output_repairs

    result = {
        "document_type": "receipt",
        "subtotal": 4532,
        "total": 4985,
        "taxes": [{"rate": "10%", "label": "内税", "amount": 453}],
        "line_items": [
            {
                "description": "充電式カイロ",
                "qty": 1,
                "unit_price": 3828,
                "total": 3445,
                "tax_category": "10%",
                "discount": 383,
                "discount_rate": "",
            },
            {
                "description": "ワイヤースプーン",
                "qty": 1,
                "unit_price": 1540,
                "total": 1540,
                "tax_category": "10%",
                "discount": 383,
                "discount_rate": "10%",
            },
        ],
    }
    ocr_text = "\n".join([
        "充電式カイロ",
        "4911111111111",
        "セール",
        "ワイヤースプーン",
        "4922222222222",
        "¥3,828",
        "- ¥383",
        "¥1,540",
        "小計",
        "¥4,985",
    ])

    _apply_final_receipt_output_repairs(result, ocr_text)

    assert [item["discount"] for item in result["line_items"]] == [383, 0]
    assert [item["total"] for item in result["line_items"]] == [3445, 1540]


def test_final_receipt_output_repairs_restore_names_before_stacked_price_block():
    from receipt_parser.pipeline import _apply_final_receipt_output_repairs

    result = {
        "document_type": "receipt",
        "subtotal": 4532,
        "total": 4985,
        "taxes": [{"rate": "10%", "label": "内税", "amount": 453}],
        "line_items": [
            {
                "description": "商品B",
                "qty": 1,
                "unit_price": 3828,
                "total": 3445,
                "tax_category": "10%",
                "discount": 383,
                "discount_rate": "",
            },
            {
                "description": "商品B セール",
                "qty": 1,
                "unit_price": 1540,
                "total": 1540,
                "tax_category": "10%",
                "discount": 0,
                "discount_rate": "",
            },
        ],
    }
    ocr_text = "\n".join([
        "店舗名",
        "商品A 長い商品名",
        "4911111111111",
        "セール",
        "商品B",
        "4922222222222",
        "¥3,828",
        "- ¥383",
        "¥1,540",
        "小計",
        "¥4,985",
    ])

    _apply_final_receipt_output_repairs(result, ocr_text)

    assert [item["description"] for item in result["line_items"]] == [
        "商品A 長い商品名",
        "商品B",
    ]
    assert [item["total"] for item in result["line_items"]] == [3445, 1540]


def test_final_receipt_output_repairs_ignore_metadata_before_stacked_price_block():
    from receipt_parser.pipeline import _apply_final_receipt_output_repairs

    result = {
        "document_type": "receipt",
        "subtotal": 5368,
        "total": 5905,
        "taxes": [{"rate": "10%", "label": "内税", "amount": 537}],
        "line_items": [
            {
                "description": "ワイヤレス充電器",
                "qty": 1,
                "unit_price": 3828,
                "total": 3828,
                "tax_category": "10%",
                "discount": 0,
                "discount_rate": "",
            },
            {
                "description": "ワイヤレス充電器 セール",
                "qty": 1,
                "unit_price": 1540,
                "total": 1540,
                "tax_category": "10%",
                "discount": 0,
                "discount_rate": "",
            },
        ],
    }
    ocr_text = "\n".join([
        "取引 No9454 販売員 1947",
        "レジNo6800101",
        "¥3,828",
        "¥1,540",
        "小計",
        "¥5,368",
    ])

    _apply_final_receipt_output_repairs(result, ocr_text)

    assert [item["description"] for item in result["line_items"]] == [
        "ワイヤレス充電器",
        "ワイヤレス充電器 セール",
    ]


def test_final_receipt_output_repairs_reconcile_tax_categories_from_rate_bases():
    from receipt_parser.pipeline import _apply_final_receipt_output_repairs

    result = {
        "document_type": "receipt",
        "subtotal": 5182,
        "total": 5615,
        "taxes": [
            {"rate": "8%", "label": "外税", "amount": 336},
            {"rate": "10%", "label": "外税", "amount": 97},
        ],
        "line_items": [
            {"description": "食品A", "qty": 1, "unit_price": 4212, "total": 4212, "tax_category": "8%"},
            {"description": "袋", "qty": 1, "unit_price": 2, "total": 2, "tax_category": "10%"},
            {"description": "日用品A", "qty": 1, "unit_price": 168, "total": 168, "tax_category": "10%"},
            {"description": "日用品B", "qty": 1, "unit_price": 800, "total": 800, "tax_category": "8%"},
        ],
    }
    ocr_text = "\n".join([
        "外税8%対象額",
        "外税8%",
        "外税10%対象額",
        "外税10%",
        "合計",
        "¥4,212",
        "¥336",
        "¥970",
        "¥97",
        "¥5,615",
    ])

    _apply_final_receipt_output_repairs(result, ocr_text)

    assert [item["tax_category"] for item in result["line_items"]] == [
        "8%", "10%", "10%", "10%"
    ]


def test_final_receipt_output_repairs_preserve_visible_bag_standard_rate():
    from receipt_parser.pipeline import _apply_final_receipt_output_repairs

    result = {
        "document_type": "receipt",
        "subtotal": 2111,
        "total": 2279,
        "taxes": [{"rate": "8%", "label": "外税", "amount": 168}],
        "line_items": [
            {"description": "食品A", "qty": 1, "unit_price": 2106, "total": 2106, "tax_category": "8%"},
            {"description": "レジ袋", "qty": 1, "unit_price": 5, "total": 5, "tax_category": "8%"},
        ],
    }
    ocr_text = "\n".join([
        "食品A",
        "¥2,106",
        "000226 レジ袋5円",
        "小計",
        "税率 8% 課税対象額",
        "¥2,111",
        "税率 8%税額",
        "¥168",
        "計",
        "税率10%課税対象額",
        "合計",
        "¥5",
        "¥2,279",
        "*印は軽減税率(8%) 適用商品です",
    ])

    _apply_final_receipt_output_repairs(result, ocr_text)

    assert [item["tax_category"] for item in result["line_items"]] == ["8%", "10%"]
