"""llm.py — LLM extraction via DeepSeek API (default), OpenRouter, or Ollama, multi-pass."""

import json
import logging
import os
import re
import signal
import platform
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

import ollama as ollama_client
from dotenv import load_dotenv
from .schema import Receipt, generate_extraction_prompt, generate_verification_prompt

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", encoding="utf-8")

OLLAMA_TIMEOUT_SECONDS = 180
DEFAULT_MODEL = "deepseek-v4-flash"
OLLAMA_PREFIX = "ollama/"
_LLM_SEED = 42  # Fixed seed for deterministic output


@dataclass
class LLMResult:
    """Structured result from _llm_chat()."""
    content: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    eval_duration_ns: int | None = None
    total_duration_ns: int | None = None
    load_duration_ns: int | None = None
    backend: str = "unknown"  # "api" or "ollama"


def _is_ollama_model(model: str) -> bool:
    """Check if model should use Ollama backend (prefixed with 'ollama/')."""
    return model.startswith(OLLAMA_PREFIX)


def _ollama_model_name(model: str) -> str:
    """Strip the 'ollama/' prefix to get the actual Ollama model name."""
    return model[len(OLLAMA_PREFIX):]


def check_model_available(model: str = DEFAULT_MODEL) -> None:
    """Verify the LLM backend is reachable."""
    if _is_ollama_model(model):
        _check_ollama_available(_ollama_model_name(model))
        return

    provider, routed_model = _api_client_key_for_model(model)
    if provider == "deepseek":
        if not os.environ.get("DEEPSEEK_API_KEY"):
            raise RuntimeError(
                "DeepSeek API key is required for the default extraction model.\n\n"
                "Quick fix:\n"
                "  1. Copy .env.example to .env:  cp .env.example .env\n"
                "  2. Add your DeepSeek API key:   DEEPSEEK_API_KEY=sk-...\n"
                "     Get one at: https://platform.deepseek.com/api_keys\n\n"
                "OpenRouter models must be requested explicitly with an openrouter/ "
                "prefix or provider/model name."
            )
        return

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError(
            "OpenRouter API key is required for routed model "
            f"'{routed_model or model}'. Add OPENROUTER_API_KEY to .env."
        )


def _check_ollama_available(model: str) -> None:
    """Verify Ollama is running and the model is pulled."""
    try:
        models = ollama_client.list()
        available = [m.model or "" for m in models.models] if hasattr(models, 'models') else []
        if not any(model in m for m in available):
            raise RuntimeError(f"Model '{model}' not found. Run: ollama pull {model}")
    except Exception as e:
        if "ConnectionError" in type(e).__name__ or "refused" in str(e).lower():
            raise RuntimeError(
                "Ollama is not running. Start it with: ollama serve\n"
                "Or on Windows, launch the Ollama desktop app."
            ) from e
        raise


def get_ollama_schema() -> dict:
    """Return a flattened JSON schema with $ref pointers resolved."""
    schema = Receipt.model_json_schema()

    if "$defs" not in schema:
        return schema

    defs = schema.pop("$defs")

    def resolve_refs(obj: object) -> object:
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_name = obj["$ref"].split("/")[-1]
                if ref_name in defs:
                    return resolve_refs(defs[ref_name])
            return {k: resolve_refs(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_refs(item) for item in obj]
        return obj

    result = resolve_refs(schema)
    return result  # type: ignore[return-value]


def sanitize_llm_response(raw: str) -> str:
    """Strip markdown code fences and extract JSON block from LLM output.

    Handles cases where the model wraps valid JSON in explanation text.
    """
    # First try: strip code fences
    cleaned = re.sub(r'^```json\n|\n```$', '', raw, flags=re.MULTILINE).strip()
    if cleaned.startswith('{'):
        return cleaned
    # Fallback: extract the outermost JSON object from the response
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        return match.group(0)
    return cleaned


def _extract_confidence(data: dict) -> dict | None:
    """Extract and validate the _confidence field from LLM output."""
    conf = data.pop("_confidence", None)
    if not isinstance(conf, dict):
        return None
    validated = {}
    for key, val in conf.items():
        try:
            v = float(val)
            if 0.0 <= v <= 1.0:
                validated[key] = round(v, 4)
        except (TypeError, ValueError):
            pass
    return validated if validated else None


def _parse_llm_json(raw: str) -> dict:
    """Parse LLM JSON output with Pydantic validation.

    Extracts _confidence before Pydantic validation (not part of schema).
    Coercion is handled by Pydantic's model_validator in schema.py.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return {"error": f"LLM output is not valid JSON: {e}", "raw": raw}

    confidence = _extract_confidence(data)
    try:
        receipt = Receipt(**data)
        result = receipt.model_dump()
        if confidence:
            result["_confidence"] = confidence
        return result
    except Exception as e:
        logger.warning("Pydantic validation failed: %s", e)
        return {"error": f"LLM output failed schema validation: {e}", "raw": raw}


_api_clients = {}
_instructor_client = None
_client_lock = threading.Lock()
_OPENROUTER_CREDIT_MESSAGE = (
    "OpenRouter credits are exhausted. Top off your OpenRouter account and "
    "rerun, or clear RECEIPT_TRIAGE_MODELS to skip optional model triage."
)


def _api_client_key_for_model(model: str | None = None) -> tuple[str, str | None]:
    """Select DeepSeek direct for DeepSeek models, OpenRouter for routed models."""
    if model and model.startswith("openrouter/"):
        return "openrouter", model.removeprefix("openrouter/")
    if model and "/" in model:
        return "openrouter", model
    return "deepseek", model


def _is_openrouter_credit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "openrouter credits are exhausted" in text
        or ("402" in text and "credit" in text)
        or "requires more credits" in text
        or "openrouter.ai/settings/credits" in text
    )


def _format_llm_error(exc: Exception, model: str | None = None) -> str:
    provider, _ = _api_client_key_for_model(model)
    if provider == "openrouter" and _is_openrouter_credit_error(exc):
        return f"openrouter_insufficient_credits: {_OPENROUTER_CREDIT_MESSAGE}"
    return f"llm_error: {exc}"


def _get_api_client(model: str | None = None):
    """Get or create the API client (DeepSeek direct or OpenRouter fallback)."""
    provider, _ = _api_client_key_for_model(model)
    if provider in _api_clients:
        return _api_clients[provider]
    with _client_lock:
        if provider not in _api_clients:
            from openai import OpenAI
            deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
            openrouter_key = os.environ.get("OPENROUTER_API_KEY")
            if provider == "deepseek" and deepseek_key:
                _api_clients[provider] = OpenAI(
                    base_url="https://api.deepseek.com",
                    api_key=deepseek_key,
                    timeout=OLLAMA_TIMEOUT_SECONDS,
                )
            elif openrouter_key:
                _api_clients[provider] = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=openrouter_key,
                    timeout=OLLAMA_TIMEOUT_SECONDS,
                )
            else:
                raise RuntimeError(
                    "No API key set. Add DEEPSEEK_API_KEY for the default DeepSeek "
                    "model, or OPENROUTER_API_KEY for explicitly routed models."
                )
    return _api_clients[provider]


def _get_instructor_client():
    """Get or create an instructor-patched client for Pydantic-native structured output."""
    global _instructor_client
    if _instructor_client is not None:
        return _instructor_client
    with _client_lock:
        if _instructor_client is None:
            import instructor
            base_client = _get_api_client()
            _instructor_client = instructor.from_openai(
                base_client,
                mode=instructor.Mode.JSON,
            )
    return _instructor_client


def _usage_attr(obj: object, key: str, default: object = None) -> object:
    """Read usage fields from OpenAI SDK objects, dicts, or model_extra."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    value = getattr(obj, key, None)
    if value is not None:
        return value
    extra = getattr(obj, "model_extra", None)
    if isinstance(extra, dict):
        return extra.get(key, default)
    return default


def _usage_nested(obj: object, parent_key: str, child_key: str) -> object:
    parent = _usage_attr(obj, parent_key)
    return _usage_attr(parent, child_key)


def _as_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _openrouter_chat(
    model: str,
    messages: list,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    seed: int = _LLM_SEED,
) -> LLMResult:
    """Call DeepSeek/OpenRouter API and return structured result."""
    provider, routed_model = _api_client_key_for_model(model)
    client = _get_api_client(model)
    t0 = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=routed_model or model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            extra_body={"thinking": {"type": "disabled"}},
        )
    except Exception as e:
        if provider == "openrouter" and _is_openrouter_credit_error(e):
            raise RuntimeError(_OPENROUTER_CREDIT_MESSAGE) from e
        raise
    elapsed_ns = int((time.perf_counter() - t0) * 1e9)
    usage = response.usage
    input_toks = _as_int(_usage_attr(usage, "prompt_tokens"))
    output_toks = _as_int(_usage_attr(usage, "completion_tokens"))

    if provider == "deepseek":
        # DeepSeek returns cache hit/miss breakdown in usage.
        cache_hit = _as_int(_usage_attr(usage, "prompt_cache_hit_tokens"))
        cache_miss = _as_int(_usage_attr(usage, "prompt_cache_miss_tokens"))
        if cache_hit is None and cache_miss is None and input_toks:
            cache_hit = 0
            cache_miss = input_toks

        from .usage import track_deepseek_call
        track_deepseek_call(cache_hit, cache_miss, output_toks)
    else:
        cache_hit = _as_int(_usage_nested(usage, "prompt_tokens_details", "cached_tokens"))
        cache_write = _as_int(_usage_nested(usage, "prompt_tokens_details", "cache_write_tokens"))
        if cache_hit is None:
            cache_hit = _as_int(_usage_attr(usage, "prompt_cache_hit_tokens")) or 0
        cache_miss = None
        if input_toks is not None:
            cache_miss = max(0, input_toks - (cache_hit or 0))
        reasoning = _as_int(_usage_nested(usage, "completion_tokens_details", "reasoning_tokens"))
        cost_usd = _as_float(_usage_attr(usage, "cost"))
        response_model = getattr(response, "model", None) or routed_model or model

        from .usage import track_openrouter_call
        track_openrouter_call(
            model=response_model,
            cache_hit_tokens=cache_hit,
            cache_miss_tokens=cache_miss,
            output_tokens=output_toks,
            cost_usd=cost_usd,
            reasoning_tokens=reasoning,
            cache_write_tokens=cache_write,
        )

    return LLMResult(
        content=response.choices[0].message.content,
        input_tokens=input_toks,
        output_tokens=output_toks,
        eval_duration_ns=elapsed_ns,
        total_duration_ns=elapsed_ns,
        backend="api",
    )


def _instructor_extract(
    model: str,
    messages: list,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    max_retries: int = 2,
) -> tuple[Receipt | None, LLMResult | None]:
    """Use instructor for Pydantic-native structured output with automatic retry.

    Returns (Receipt, LLMResult) tuple. Receipt is None if instructor is unavailable
    or extraction fails. LLMResult captures timing metadata.
    """
    client = _get_instructor_client()
    if client is None:
        return None, None
    try:
        t0 = time.perf_counter()
        receipt = client.chat.completions.create(
            model=model,
            messages=messages,
            response_model=Receipt,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=_LLM_SEED,
            max_retries=max_retries,
            extra_body={"thinking": {"type": "disabled"}},
        )
        elapsed_ns = int((time.perf_counter() - t0) * 1e9)
        llm_result = LLMResult(
            content="(instructor)",
            eval_duration_ns=elapsed_ns,
            total_duration_ns=elapsed_ns,
            backend="api",
        )
        return receipt, llm_result
    except Exception as e:
        logger.warning("Instructor extraction failed: %s", e)
        return None, None


_RETRYABLE_ERRORS = ("GGML_ASSERT", "model failed to load", "resource limitations")
_MAX_RETRIES = 2
_RETRY_DELAY = 5


def _ollama_chat_with_timeout(timeout: int = OLLAMA_TIMEOUT_SECONDS, **kwargs: object) -> dict:
    """Wrapper around ollama.chat() with a wall-clock timeout and retry for transient errors."""
    import time

    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return _ollama_chat_once(timeout, **kwargs)
        except Exception as e:
            err_str = str(e)
            if any(msg in err_str for msg in _RETRYABLE_ERRORS) and attempt < _MAX_RETRIES:
                last_error = e
                time.sleep(_RETRY_DELAY)
                continue
            raise
    raise last_error  # type: ignore[misc]  # unreachable but satisfies type checker


def _ollama_chat_once(timeout: int, **kwargs: object) -> dict:
    """Single attempt at ollama.chat() with a wall-clock timeout."""
    if platform.system() != "Windows":
        def _handler(signum: int, frame: object) -> None:
            raise TimeoutError(f"Ollama did not respond within {timeout}s")
        old_handler = signal.signal(signal.SIGALRM, _handler)  # type: ignore[attr-defined]
        signal.alarm(timeout)  # type: ignore[attr-defined]
        try:
            response = ollama_client.chat(**kwargs)  # type: ignore[arg-type]
        finally:
            signal.alarm(0)  # type: ignore[attr-defined]
            signal.signal(signal.SIGALRM, old_handler)  # type: ignore[attr-defined]
        return response  # type: ignore[return-value]
    else:
        result: list[dict | None] = [None]
        error: list[Exception | None] = [None]
        def _call() -> None:
            try:
                result[0] = ollama_client.chat(**kwargs)  # type: ignore[arg-type]
            except Exception as e:
                error[0] = e
        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            raise TimeoutError(f"Ollama did not respond within {timeout}s")
        if error[0]:
            raise error[0]
        return result[0]  # type: ignore[return-value]


def _llm_chat(
    model: str,
    messages: list,
    schema: dict,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    seed: int = _LLM_SEED,
) -> LLMResult:
    """Unified LLM chat — dispatches to Ollama or OpenRouter."""
    if _is_ollama_model(model):
        response = _ollama_chat_with_timeout(
            model=_ollama_model_name(model),
            messages=messages,
            format=schema,
            options={"temperature": temperature, "num_predict": max_tokens, "seed": seed},
            think=False,
            keep_alive="60m",
        )
        return LLMResult(
            content=response["message"]["content"],
            input_tokens=response.get("prompt_eval_count"),
            output_tokens=response.get("eval_count"),
            eval_duration_ns=response.get("eval_duration"),
            total_duration_ns=response.get("total_duration"),
            load_duration_ns=response.get("load_duration"),
            backend="ollama",
        )
    else:
        return _openrouter_chat(model, messages, temperature, max_tokens, seed=seed)


def extract_with_llm(
    ocr_text: str,
    model: str = DEFAULT_MODEL,
    doc_type: str = "receipt",
    use_instructor: bool = True,
) -> tuple[dict, LLMResult | None]:
    """Single-pass extraction with structured output enforcement.

    Returns (parsed_dict, llm_result) where llm_result contains timing metadata.

    Strategy: normal JSON extraction first (proven deterministic), then
    instructor as a fallback if JSON parsing fails. This preserves the
    proven extraction behavior while gaining instructor's auto-retry for
    malformed responses.
    """
    system_prompt, user_prompt = generate_extraction_prompt(ocr_text, doc_type=doc_type)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Primary path: normal JSON extraction (deterministic with seed=42)
    llm_result = _llm_chat(
        model=model,
        messages=messages,
        schema=get_ollama_schema(),
    )

    parsed = _parse_llm_json(sanitize_llm_response(llm_result.content))

    # Fallback: if JSON parsing failed, try instructor for auto-retry
    if "error" in parsed and use_instructor and not _is_ollama_model(model):
        receipt, instructor_result = _instructor_extract(model=model, messages=messages)
        if receipt is not None:
            return receipt.model_dump(), instructor_result or llm_result

    # Content quality retry: re-extract when items grossly exceed total
    if "error" not in parsed and _extraction_is_low_quality(parsed):
        logger.info("Low-quality extraction detected, retrying with seed=%d", _LLM_SEED + 1)
        retry_result = _llm_chat(
            model=model,
            messages=messages,
            schema=get_ollama_schema(),
            temperature=0.3,
        )
        retry_parsed = _parse_llm_json(sanitize_llm_response(retry_result.content))
        if "error" not in retry_parsed and not _extraction_is_low_quality(retry_parsed):
            return retry_parsed, retry_result

    return parsed, llm_result


def _has_duplicate_descs(extracted: dict) -> bool:
    """Detect duplicate (description, total) pairs in line_items.

    Returns True if 2+ items share the same normalized description AND total
    — a strong signal the LLM copy-pasted a nearby item's name onto a
    distinct adjacent row.
    """
    items = extracted.get("line_items", []) or []
    if len(items) < 2:
        return False
    seen: dict[tuple, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        desc = (item.get("description") or "").strip()
        # Normalize: strip trailing whitespace+digits (the embedded-price
        # case where 'X' and 'X  228' should compare equal).
        desc = re.sub(r'\s+[\d,]{1,6}\s*[\*※]?\s*$', '', desc).strip()
        total = item.get("total")
        if not desc or total is None:
            continue
        key = (desc, float(total))
        seen[key] = seen.get(key, 0) + 1
        if seen[key] >= 2:
            return True
    return False


def _alternate_seed_extract(
    ocr_text: str, model: str, doc_type: str, seed_offset: int = 1,
) -> dict | None:
    """Re-run extraction with a non-default seed using the SAME extraction
    prompt — not verification.

    Verification prompts bias toward the previous pass's mistake; for cross-
    check we want an independent extraction. Returns parsed dict or None on
    failure.

    seed_offset: 1 → seed=43, 2 → seed=44, etc.
    """
    parsed, _, error_reason = _alternate_seed_extract_with_result(
        ocr_text, model, doc_type, seed_offset=seed_offset
    )
    if error_reason:
        return None
    return parsed


def _alternate_seed_extract_with_result(
    ocr_text: str, model: str, doc_type: str, seed_offset: int = 1,
) -> tuple[dict | None, LLMResult | None, str | None]:
    """Re-run extraction and return enough detail for diagnostic history."""
    system_prompt, user_prompt = generate_extraction_prompt(ocr_text, doc_type=doc_type)
    try:
        result = _llm_chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            schema=get_ollama_schema(),
            seed=_LLM_SEED + seed_offset,
        )
    except Exception as e:
        return None, None, _format_llm_error(e, model)
    parsed = _parse_llm_json(sanitize_llm_response(result.content))
    if "error" in parsed:
        return parsed, result, str(parsed.get("error") or "parse_error")
    return parsed, result, None


_CROSS_PROMPT_VARIANTS = {
    "row_audit": """
ADDITIONAL ROW-AUDIT RULES:
- Treat line item extraction as a row-by-row transcription task, not a summary.
- Before returning JSON, verify every line_item total against a visible OCR price.
- If item names and prices are split into separate blocks, preserve the OCR price
  sequence one-for-one. Do not substitute an adjacent price just because it makes
  a subtotal look closer.
- Do not change a distinct visible OCR price into a duplicated neighboring price
  unless the OCR itself shows that duplicated price.
""".strip(),
}


def _cross_prompt_extract_with_result(
    ocr_text: str, model: str, doc_type: str, variant: str,
    max_tokens: int = 8192,
) -> tuple[dict | None, LLMResult | None, str | None]:
    """Run an extraction with an alternate prompt surface for validator rescue."""
    system_prompt, user_prompt = generate_extraction_prompt(ocr_text, doc_type=doc_type)
    extra_rules = _CROSS_PROMPT_VARIANTS.get(variant)
    if not extra_rules:
        return None, None, f"unknown_prompt_variant: {variant}"
    try:
        result = _llm_chat(
            model=model,
            messages=[
                {"role": "system", "content": f"{system_prompt}\n\n{extra_rules}"},
                {"role": "user", "content": user_prompt},
            ],
            schema=get_ollama_schema(),
            seed=_LLM_SEED,
            max_tokens=max_tokens,
        )
    except Exception as e:
        return None, None, _format_llm_error(e, model)
    parsed = _parse_llm_json(sanitize_llm_response(result.content))
    if "error" in parsed:
        return parsed, result, str(parsed.get("error") or "parse_error")
    return parsed, result, None


def _configured_triage_models() -> list[str]:
    raw = os.environ.get("RECEIPT_TRIAGE_MODELS", "")
    return [m.strip() for m in raw.split(",") if m.strip()]


def _triage_max_tokens() -> int:
    raw = os.environ.get("RECEIPT_TRIAGE_MAX_TOKENS", "2048")
    try:
        value = int(raw)
    except ValueError:
        return 2048
    return min(max(value, 1024), 8192)


def _model_triage_extract_with_result(
    ocr_text: str, model: str, doc_type: str,
) -> tuple[dict | None, LLMResult | None, str | None]:
    """Run validator-flagged residual through a configured alternate model."""
    return _cross_prompt_extract_with_result(
        ocr_text, model=model, doc_type=doc_type, variant="row_audit",
        max_tokens=_triage_max_tokens(),
    )


def _items_sum_gap(extracted: dict) -> float | None:
    """Return absolute gap between items_sum and the closest of subtotal/total.

    Returns None if no items or no targets to compare against. A small gap
    (≤ 2) means items_sum agrees with one of the targets — the extraction
    is internally consistent. Larger gaps indicate the LLM is missing or
    duplicating items, or has wrong totals.
    """
    items = extracted.get("line_items") or []
    if not items:
        return None
    items_sum = sum(
        i.get("total", 0) for i in items if isinstance(i, dict)
    )
    subtotal = extracted.get("subtotal")
    total = extracted.get("total")
    targets = []
    taxes = extracted.get("taxes") or []
    canonical_subtotal = None
    if total and taxes:
        tax_sum = sum(
            t.get("amount", 0) for t in taxes
            if isinstance(t, dict) and t.get("amount") is not None
        )
        if tax_sum:
            canonical_subtotal = total - tax_sum
    if subtotal and (
        canonical_subtotal is None or abs(float(subtotal) - float(canonical_subtotal)) <= 2
    ):
        targets.append(subtotal)
    if total:
        targets.append(total)
    if canonical_subtotal is not None:
        targets.append(canonical_subtotal)
    if not targets:
        return None
    return min(abs(items_sum - t) for t in targets)


def _has_item_sum_warning(warnings: list[str]) -> bool:
    return any(
        "Sum of line items" in warning or "Items sum" in warning
        for warning in warnings
    )


def _validator_visible_item_sum_gap(extracted: dict, warnings: list[str]) -> float | None:
    """Return the item-sum gap only when validation has surfaced that issue."""
    if not _has_item_sum_warning(warnings):
        return None
    gap = _items_sum_gap(extracted)
    if gap is None or gap <= 5:
        return None
    return gap


def _item_sum_gap_if_validator_flagged(
    extracted: dict,
    *warning_sets: list[str],
) -> float | None:
    if not any(_has_item_sum_warning(warnings) for warnings in warning_sets):
        return None
    gap = _items_sum_gap(extracted)
    if gap is None or gap <= 5:
        return None
    return gap


def _with_financial_anchors(candidate: dict, anchor: dict) -> dict:
    """Preserve trusted financial fields when scoring item-retry candidates."""
    anchored = deepcopy(candidate)
    for key in ("total", "subtotal", "taxes"):
        if anchor.get(key) is not None:
            anchored[key] = deepcopy(anchor[key])
    return anchored


def _retry_history_extractions(
    raw_candidate: dict | None,
    scored_candidate: dict | None,
) -> dict:
    """Record the exact scored candidate, preserving raw LLM output if anchored."""
    entry = {
        "extraction": (
            deepcopy(scored_candidate)
            if scored_candidate is not None
            else deepcopy(raw_candidate) if raw_candidate else {}
        )
    }
    if raw_candidate is not None and scored_candidate is not None and raw_candidate != scored_candidate:
        entry["raw_extraction"] = deepcopy(raw_candidate)
    return entry


def _substitute_dup_descs_from_alt(extracted: dict, alt: dict) -> int:
    """For each duplicate-desc item in `extracted`, look for an alt-pass item
    with the same total (±1) but a description not already present. If found,
    substitute. Returns count of substitutions made.

    This addresses the LLM copy-paste failure mode: when the OCR text has
    two distinct adjacent items at the same price, the model sometimes
    duplicates the first item's description over the second. A different
    seed often picks up the second item's actual description.
    """
    items = extracted.get("line_items", []) or []
    alt_items = alt.get("line_items", []) or []
    if not items or not alt_items:
        return 0

    # Group extracted items by (normalized_desc, total)
    def _norm(d: str) -> str:
        return re.sub(r'\s+[\d,]{1,6}\s*[\*※]?\s*$', '', (d or "").strip()).strip()

    groups: dict[tuple, list[int]] = {}
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        desc = _norm(item.get("description") or "")
        total = item.get("total")
        if not desc or total is None:
            continue
        groups.setdefault((desc, float(total)), []).append(i)

    duplicates = {k: v for k, v in groups.items() if len(v) >= 2}
    if not duplicates:
        return 0

    # Existing descriptions (normalized) currently in the extraction
    existing = {_norm(it.get("description") or "")
                for it in items if isinstance(it, dict)}

    # For each duplicate group, find candidates in the alt extraction
    substituted = 0
    for (dup_desc, dup_total), idxs in duplicates.items():
        # Alt items with matching total but different normalized desc
        candidates = []
        for ai in alt_items:
            if not isinstance(ai, dict):
                continue
            a_desc = (ai.get("description") or "").strip()
            a_total = ai.get("total")
            if not a_desc or a_total is None:
                continue
            if abs(float(a_total) - dup_total) > 1:
                continue
            a_norm = _norm(a_desc)
            if a_norm == dup_desc or a_norm in existing:
                continue
            if a_desc not in candidates:
                candidates.append(a_desc)

        if not candidates:
            continue

        # Decide how many of the duplicates to replace:
        # - If dup_desc ALSO appears in this extraction as a distinct item with
        #   a DIFFERENT total, then ALL occurrences at dup_total are spurious
        #   copy-pastes (the real item is the one at the different total).
        #   Replace all of them.
        # - Otherwise, assume one is the real original — keep it, replace the
        #   rest.
        real_at_diff_total = any(
            _norm(it.get("description") or "") == dup_desc
            and it.get("total") is not None
            and abs(float(it.get("total")) - dup_total) > 1
            for it in items
            if isinstance(it, dict)
        )
        targets = idxs if real_at_diff_total else idxs[1:]

        for cand_desc, idx in zip(candidates, targets):
            items[idx]["description"] = cand_desc
            existing.add(_norm(cand_desc))
            substituted += 1

    # Same-desc, different-total fix: when pass 1 has duplicate (desc, total)
    # but the alt pass has the same desc with DIFFERENT totals (e.g. one
    # at 640 and one at 606, when pass 1 has both at 606), substitute one
    # of the dup's total to match the alt's distinct total.
    #
    # Conservative: only when the alt pass has exactly one item with the
    # same desc at a different total, and applying that substitution moves
    # the items_sum gap toward subtotal/total.
    items_sum = sum(i.get("total", 0) for i in items if isinstance(i, dict))
    target_v = (extracted.get("subtotal") or 0) or (extracted.get("total") or 0)
    for (dup_desc, dup_total), idxs in duplicates.items():
        # Find alt items with same desc but DIFFERENT total
        alt_diff_totals: list[tuple[float, float, dict]] = []  # (total, unit, item)
        for ai in alt_items:
            if not isinstance(ai, dict):
                continue
            a_desc = _norm(ai.get("description") or "")
            a_total = ai.get("total")
            if not a_desc or a_total is None:
                continue
            if a_desc != dup_desc:
                continue
            if abs(float(a_total) - dup_total) <= 1:
                continue
            alt_diff_totals.append(
                (float(a_total), float(ai.get("unit_price") or a_total), ai)
            )

        if not alt_diff_totals:
            continue

        # Try each alt total: would substituting one of the dup items
        # (changing its total) move us closer to target?
        for a_total, a_unit, _ai in alt_diff_totals:
            new_sum = items_sum - dup_total + a_total
            if target_v and abs(new_sum - target_v) < abs(items_sum - target_v):
                # Apply: replace the LAST dup occurrence with alt's values
                target_idx = idxs[-1]
                items[target_idx]["total"] = a_total
                items[target_idx]["unit_price"] = a_unit
                items[target_idx]["qty"] = 1
                items_sum = new_sum
                substituted += 1
                break

    return substituted


def _extraction_is_low_quality(parsed: dict) -> bool:
    """Detect structurally valid but content-broken extractions."""
    items = parsed.get("line_items", [])
    total = parsed.get("total")
    if not items or not total or total <= 0:
        return False
    item_sum = sum(i.get("total", 0) for i in items if isinstance(i, dict))
    if item_sum > total * 1.3:
        return True
    if len(items) >= 3 and all(i.get("unit_price", 0) == 0 for i in items if isinstance(i, dict)):
        return True
    return False


def _llm_result_to_timing(result: LLMResult | None) -> dict | None:
    """Convert LLMResult to a serializable timing dict for pass history."""
    if result is None:
        return None
    return {
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "eval_duration_ns": result.eval_duration_ns,
        "total_duration_ns": result.total_duration_ns,
        "load_duration_ns": result.load_duration_ns,
        "backend": result.backend,
    }


def extract_with_verification(
    ocr_text: str,
    model: str = DEFAULT_MODEL,
    passes: int = 1,
    validate_fn=None,
    doc_type: str = "receipt",
    on_stage=None,
) -> tuple[dict, list[dict]]:
    """Multi-pass text extraction. Pass 1 extracts. Pass 2+ self-corrects.

    on_stage, if provided, fires once per pass with stage="extract" and
    monotonically increasing progress within the [0.45, 0.70) band. Imported
    lazily to avoid a circular import with pipeline.py.
    """
    passes = max(1, passes)
    history = []

    # Lazy import to avoid pipeline.py <-> llm.py circular import at module load.
    if on_stage is not None:
        from .pipeline import _notify
    else:
        _notify = None

    _PASS_BAND_START, _PASS_BAND_END = 0.45, 0.70
    _PASS_BAND = _PASS_BAND_END - _PASS_BAND_START

    def _pass_progress(pass_num: int) -> float:
        # Spread N passes evenly across the pass band so each pass beat is monotonic.
        return _PASS_BAND_START + ((pass_num - 1) / passes) * _PASS_BAND

    if _notify is not None:
        _notify(
            on_stage, "extract", f"LLM pass 1 of {passes}",
            _pass_progress(1),
            payload={"pass": 1, "pass_budget": passes},
        )

    extracted, llm_result = extract_with_llm(ocr_text, model=model, doc_type=doc_type)
    warnings = []
    if validate_fn and "error" not in extracted:
        try:
            receipt = Receipt(**extracted)
            warnings = validate_fn(receipt)
        except Exception:
            warnings = ["Schema validation failed on pass 1"]

    history.append({
        "pass": 1, "extraction": deepcopy(extracted), "warnings": warnings,
        "llm_timing": _llm_result_to_timing(llm_result),
    })

    # Cross-check pass: when pass 1 has duplicate-description items, run a
    # fresh extraction with a DIFFERENT seed (not the verification prompt —
    # that biases toward pass 1's mistake). Then for each duplicate in pass
    # 1, look for an alternate-pass item with the same total but a distinct
    # description. Substitute. This fixes the common LLM failure mode where
    # the model copies a nearby item's name onto a distinct adjacent row.
    if "error" not in extracted and _has_duplicate_descs(extracted):
        alt_extracted = _alternate_seed_extract(ocr_text, model, doc_type)
        if alt_extracted is not None and "error" not in alt_extracted:
            substituted = _substitute_dup_descs_from_alt(extracted, alt_extracted)
            if substituted > 0:
                # Re-validate after substitution
                alt_warnings = []
                if validate_fn:
                    try:
                        receipt = Receipt(**extracted)
                        alt_warnings = validate_fn(receipt)
                    except Exception:
                        alt_warnings = ["Schema validation failed after cross-check"]
                history.append({
                    "pass": "1-cross", "extraction": deepcopy(extracted),
                    "warnings": alt_warnings, "alt_extraction": deepcopy(alt_extracted),
                    "substitutions": substituted,
                    "llm_timing": None,
                })
                warnings = alt_warnings

    # Track the best extraction across all passes. Two filters apply:
    # (1) only LLM-correctable warnings trigger a retry — subtotal-arithmetic
    #     and tax-ratio warnings are auto-fixed in pipeline post-processing,
    #     so retrying on them risks the LLM "fixing" the wrong thing
    #     (e.g., dropping a valid duplicate item to make items_sum match a
    #     wrongly-computed subtotal);
    # (2) the verification prompt always references pass 1 as the baseline
    #     so each retry sees the same problem fresh, not a possibly worse
    #     pass-N-1 attempt.
    best_extracted = extracted
    best_warnings = warnings
    best_llm_warnings = _llm_correctable(warnings)

    for pass_num in range(2, passes + 1):
        if not best_llm_warnings:
            break

        if _notify is not None:
            _notify(
                on_stage, "extract", f"LLM pass {pass_num} of {passes}",
                _pass_progress(pass_num),
                payload={"pass": pass_num, "pass_budget": passes,
                         "warnings_to_fix": len(best_llm_warnings)},
            )

        v_system, v_user = generate_verification_prompt(
            ocr_text=ocr_text,
            previous_extraction=best_extracted,
            validation_warnings=best_llm_warnings,
        )

        # Vary seed across passes so a deterministic LLM (temperature=0,
        # seed=42) actually produces a different response on retry. Without
        # this, pass 2+ just repeats pass 1 verbatim.
        llm_result = _llm_chat(
            model=model,
            messages=[
                {"role": "system", "content": v_system},
                {"role": "user", "content": v_user},
            ],
            schema=get_ollama_schema(),
            seed=_LLM_SEED + pass_num - 1,
        )

        raw = sanitize_llm_response(llm_result.content)
        pass_extracted = _parse_llm_json(raw) if raw.strip() else {"error": "empty response"}

        pass_warnings: list[str] = []
        if validate_fn and "error" not in pass_extracted:
            try:
                receipt = Receipt(**pass_extracted)
                pass_warnings = validate_fn(receipt)
            except Exception:
                pass_warnings = [f"Schema validation failed on pass {pass_num}"]

        history.append({
            "pass": pass_num, "extraction": deepcopy(pass_extracted), "warnings": pass_warnings,
            "llm_timing": _llm_result_to_timing(llm_result),
        })

        pass_llm_warnings = _llm_correctable(pass_warnings)
        if "error" not in pass_extracted and len(pass_llm_warnings) < len(best_llm_warnings):
            best_extracted = pass_extracted
            best_warnings = pass_warnings
            best_llm_warnings = pass_llm_warnings

    # Final substitution: if the chosen pass still has duplicate-desc items,
    # walk the other passes for distinct alternates at the same total. This
    # rescues the case where pass 1 has a duplicate and pass 2 has the correct
    # distinct desc but more total-side warnings — keep pass 1's structure
    # (which scored better) but borrow pass 2's better desc for the dup slot.
    if "error" not in best_extracted and _has_duplicate_descs(best_extracted):
        for entry in history:
            alt = entry.get("extraction")
            if not alt or alt is best_extracted or "error" in alt:
                continue
            substituted = _substitute_dup_descs_from_alt(best_extracted, alt)
            if substituted > 0 and not _has_duplicate_descs(best_extracted):
                break

    # Sanity-retry: if the chosen extraction has items_sum that doesn't match
    # subtotal/total, try up to 2 fresh-seed extractions (extraction prompt,
    # NOT verification — verification biases toward the previous pass's
    # mistake). Pick the one with the smallest items_sum gap.
    #
    # Catches LLM API non-determinism: the same seed=42 occasionally
    # produces different totals across runs. Trying additional seeds gives
    # multiple chances at the correct extraction. We accept an alternate
    # only if its gap is strictly better (or matches subtotal exactly).
    if "error" not in best_extracted:
        sanity = _items_sum_gap(best_extracted)
        if sanity is not None and sanity > 5:
            for offset in (2, 3):  # seeds 44, 45
                sanity_alt, sanity_llm_result, error_reason = _alternate_seed_extract_with_result(
                    ocr_text, model, doc_type, seed_offset=offset
                )
                sanity_scored = (
                    _with_financial_anchors(sanity_alt, best_extracted)
                    if sanity_alt and "error" not in sanity_alt else None
                )
                alt_gap = _items_sum_gap(sanity_scored) if sanity_scored else None
                alt_warnings: list[str] = []
                if sanity_scored and validate_fn:
                    try:
                        receipt_alt = Receipt(**sanity_scored)
                        alt_warnings = validate_fn(receipt_alt)
                    except Exception:
                        alt_warnings = []

                accepted = (
                    error_reason is None
                    and sanity_alt is not None
                    and "error" not in sanity_alt
                    and alt_gap is not None
                    and alt_gap < sanity
                )
                if error_reason:
                    rejection_reason = error_reason
                elif sanity_alt is None or "error" in sanity_alt:
                    rejection_reason = "invalid_extraction"
                elif alt_gap is None:
                    rejection_reason = "items_sum_gap_unavailable"
                elif not accepted:
                    rejection_reason = "items_sum_gap_not_improved"
                else:
                    rejection_reason = None

                history.append({
                    "pass": f"sanity-retry-seed{_LLM_SEED + offset}",
                    "retry_kind": "sanity",
                    "seed": _LLM_SEED + offset,
                    "accepted": accepted,
                    "rejection_reason": rejection_reason,
                    "financial_anchors_preserved": True,
                    **_retry_history_extractions(sanity_alt, sanity_scored),
                    "warnings": alt_warnings,
                    "items_sum_gap_before": sanity,
                    "items_sum_gap_after": alt_gap,
                    "llm_timing": _llm_result_to_timing(sanity_llm_result),
                })

                if accepted:
                    best_extracted = sanity_scored
                    best_warnings = alt_warnings
                    sanity = alt_gap  # update baseline for next iteration
                    if sanity <= 2:
                        break  # close enough; stop spending LLM calls

    # Cross-prompt rescue: if validator-visible item-sum inconsistency remains,
    # try a different prompt surface before escalating to a different model.
    if "error" not in best_extracted:
        current_gap = _item_sum_gap_if_validator_flagged(
            best_extracted, best_warnings, warnings
        )
        if current_gap is not None:
            for variant in _CROSS_PROMPT_VARIANTS:
                alt, alt_llm_result, error_reason = _cross_prompt_extract_with_result(
                    ocr_text, model, doc_type, variant=variant
                )
                alt_scored = (
                    _with_financial_anchors(alt, best_extracted)
                    if alt and "error" not in alt else None
                )
                alt_gap = _items_sum_gap(alt_scored) if alt_scored else None
                alt_warnings: list[str] = []
                if alt_scored and validate_fn:
                    try:
                        receipt_alt = Receipt(**alt_scored)
                        alt_warnings = validate_fn(receipt_alt)
                    except Exception:
                        alt_warnings = []

                accepted = (
                    error_reason is None
                    and alt is not None
                    and "error" not in alt
                    and alt_gap is not None
                    and alt_gap <= 2
                )
                if error_reason:
                    rejection_reason = error_reason
                elif alt is None or "error" in alt:
                    rejection_reason = "invalid_extraction"
                elif alt_gap is None:
                    rejection_reason = "items_sum_gap_unavailable"
                elif alt_gap > 2 and alt_gap < current_gap:
                    rejection_reason = "items_sum_gap_not_closed"
                elif not accepted:
                    rejection_reason = "items_sum_gap_not_improved"
                else:
                    rejection_reason = None

                history.append({
                    "pass": f"cross-prompt-{variant}",
                    "retry_kind": "cross_prompt",
                    "prompt_variant": variant,
                    "accepted": accepted,
                    "rejection_reason": rejection_reason,
                    "financial_anchors_preserved": True,
                    **_retry_history_extractions(alt, alt_scored),
                    "warnings": alt_warnings,
                    "items_sum_gap_before": current_gap,
                    "items_sum_gap_after": alt_gap,
                    "llm_timing": _llm_result_to_timing(alt_llm_result),
                })

                if accepted:
                    best_extracted = alt_scored
                    best_warnings = alt_warnings
                    current_gap = alt_gap
                    if current_gap <= 2:
                        break

    # Model triage: disabled by default. Set RECEIPT_TRIAGE_MODELS to a
    # comma-separated list and benchmark candidates before promoting any one
    # model into normal production routing.
    if "error" not in best_extracted:
        current_gap = _item_sum_gap_if_validator_flagged(
            best_extracted, best_warnings, warnings
        )
        if current_gap is not None:
            for triage_model in _configured_triage_models():
                alt, alt_llm_result, error_reason = _model_triage_extract_with_result(
                    ocr_text, model=triage_model, doc_type=doc_type
                )
                alt_scored = (
                    _with_financial_anchors(alt, best_extracted)
                    if alt and "error" not in alt else None
                )
                alt_gap = _items_sum_gap(alt_scored) if alt_scored else None
                alt_warnings: list[str] = []
                if alt_scored and validate_fn:
                    try:
                        receipt_alt = Receipt(**alt_scored)
                        alt_warnings = validate_fn(receipt_alt)
                    except Exception:
                        alt_warnings = []

                accepted = (
                    error_reason is None
                    and alt is not None
                    and "error" not in alt
                    and alt_gap is not None
                    and alt_gap <= 2
                )
                if error_reason:
                    rejection_reason = error_reason
                elif alt is None or "error" in alt:
                    rejection_reason = "invalid_extraction"
                elif alt_gap is None:
                    rejection_reason = "items_sum_gap_unavailable"
                elif alt_gap > 2 and alt_gap < current_gap:
                    rejection_reason = "items_sum_gap_not_closed"
                elif not accepted:
                    rejection_reason = "items_sum_gap_not_improved"
                else:
                    rejection_reason = None

                history.append({
                    "pass": f"model-triage-{triage_model}",
                    "retry_kind": "model_triage",
                    "triage_model": triage_model,
                    "accepted": accepted,
                    "rejection_reason": rejection_reason,
                    "financial_anchors_preserved": True,
                    **_retry_history_extractions(alt, alt_scored),
                    "warnings": alt_warnings,
                    "items_sum_gap_before": current_gap,
                    "items_sum_gap_after": alt_gap,
                    "llm_timing": _llm_result_to_timing(alt_llm_result),
                })

                if accepted:
                    best_extracted = alt_scored
                    best_warnings = alt_warnings
                    current_gap = alt_gap
                    if current_gap <= 2:
                        break

    return best_extracted, history


# Warnings the pipeline auto-corrects in post-processing. Including them in
# the retry trigger or pass-selection metric causes the LLM to "fix" them by
# corrupting valid extractions (e.g. dropping a duplicate item to make
# items_sum match a wrongly-computed subtotal).
_PIPELINE_FIXED_WARNING_PREFIXES = (
    "Total ",          # "Total (X) does not match subtotal (Y) + taxes (Z)..."
    "Tax ratio check", # "Tax ratio check: subtotal × rate does not produce total..."
)


def _llm_correctable(warnings: list[str]) -> list[str]:
    """Filter to warnings the LLM should retry on (drops pipeline-fixable ones)."""
    return [
        w for w in warnings
        if not any(w.startswith(p) for p in _PIPELINE_FIXED_WARNING_PREFIXES)
    ]
