"""Offline request-contract checks for the deterministic LLM settings."""

import json
from types import SimpleNamespace

import pytest

import receipt_parser.llm as llm_module


def _api_response(data):
    return SimpleNamespace(
        usage=None,
        model=None,
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))],
    )


def test_extraction_verification_and_row_audit_use_fixed_api_settings(monkeypatch):
    from receipt_parser import usage

    clean = {
        "document_type": "receipt",
        "merchant": "Shop",
        "currency": "JPY",
        "total": 100,
        "subtotal": 100,
        "taxes": [],
        "line_items": [
            {"description": "A", "qty": 1, "unit_price": 100, "total": 100},
        ],
    }
    bad = {
        **clean,
        "subtotal": 300,
        "total": 330,
        "taxes": [{"rate": "unknown", "amount": 30}],
    }
    verified = {
        **bad,
        "line_items": [
            {"description": "A", "qty": 1, "unit_price": 100, "total": 100},
            {"description": "B", "qty": 1, "unit_price": 200, "total": 200},
        ],
    }
    responses = iter([clean, bad, verified, clean, clean])
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return _api_response(next(responses))

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(llm_module, "_get_api_client", lambda model=None: fake_client)
    monkeypatch.setattr(usage, "track_openrouter_call", lambda **kwargs: None)

    model = "openrouter/test-model"
    extracted, _ = llm_module.extract_with_llm("OCR", model=model)
    assert extracted["merchant"] == "Shop"

    def validate_items(receipt):
        items_sum = sum(item.total or 0 for item in receipt.line_items)
        return ["Sum of line items mismatch"] if items_sum != receipt.subtotal else []

    _verified, verification_history = llm_module.extract_with_verification(
        "OCR", model=model, passes=2, validate_fn=validate_items
    )
    assert [entry["pass"] for entry in verification_history] == [1, 2]

    _clean, row_audit_history = llm_module.extract_with_verification(
        "OCR", model=model, passes=2, validate_fn=lambda _receipt: []
    )
    assert any(
        entry.get("retry_kind") == "candidate_diversity"
        and entry.get("prompt_variant") == "row_audit"
        for entry in row_audit_history
    )

    assert len(requests) == 5
    assert all(request["temperature"] == 0.0 for request in requests)
    assert all(request["seed"] == 42 for request in requests)
    assert requests[2]["messages"][1]["content"].startswith("PREVIOUS EXTRACTION:")
    assert "ADDITIONAL ROW-AUDIT RULES" in requests[4]["messages"][0]["content"]


def test_instructor_and_ollama_requests_use_fixed_settings(monkeypatch):
    instructor_requests = []
    receipt = llm_module.Receipt(document_type="receipt")
    instructor_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: instructor_requests.append(kwargs) or receipt
            )
        )
    )
    monkeypatch.setattr(llm_module, "_get_instructor_client", lambda: instructor_client)

    extracted, _ = llm_module._instructor_extract("test-model", [])
    assert extracted is receipt
    assert instructor_requests[0]["temperature"] == 0.0
    assert instructor_requests[0]["seed"] == 42

    ollama_requests = []

    def fake_ollama(**kwargs):
        ollama_requests.append(kwargs)
        return {"message": {"content": "{}"}}

    monkeypatch.setattr(llm_module, "_ollama_chat_with_timeout", fake_ollama)
    llm_module._llm_chat("ollama/test-model", [], {"type": "object"}, max_tokens=123)

    request = ollama_requests[0]
    assert request["options"] == {"temperature": 0.0, "num_predict": 123, "seed": 42}
    assert request["think"] is False


def test_verification_stops_after_request_makes_no_warning_progress(monkeypatch):
    extracted = {
        "document_type": "receipt",
        "line_items": [
            {"description": "A", "qty": 1, "unit_price": 100, "total": 100},
        ],
        "subtotal": 100,
        "total": 100,
        "taxes": [],
    }
    verification_requests = []
    monkeypatch.setattr(
        llm_module,
        "extract_with_llm",
        lambda *args, **kwargs: (extracted, llm_module.LLMResult(content="{}")),
    )

    def fake_chat(**kwargs):
        verification_requests.append(kwargs)
        return llm_module.LLMResult(content=json.dumps(extracted))

    monkeypatch.setattr(llm_module, "_llm_chat", fake_chat)
    result, history = llm_module.extract_with_verification(
        "OCR", passes=5, validate_fn=lambda _receipt: ["Merchant missing"]
    )

    assert result == extracted
    assert len(verification_requests) == 1
    assert [entry["pass"] for entry in history] == [1, 2]


def test_default_model_never_falls_back_to_openrouter_for_missing_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(llm_module, "_api_clients", {})

    with pytest.raises(RuntimeError, match="No API key set"):
        llm_module._get_api_client(llm_module.DEFAULT_MODEL)

    assert llm_module._api_clients == {}
