"""extraction.py — LLM extraction via DeepSeek API (default), OpenRouter, or Ollama, multi-pass."""

import json
import os
import re
import signal
import platform
import threading

import ollama as ollama_client
from schema import Receipt, generate_extraction_prompt, generate_verification_prompt

OLLAMA_TIMEOUT_SECONDS = 180
DEFAULT_MODEL = "deepseek-chat"
OLLAMA_PREFIX = "ollama/"
_LLM_SEED = 42  # Fixed seed for deterministic output


def _is_ollama_model(model: str) -> bool:
    """Check if model should use Ollama backend (prefixed with 'ollama/')."""
    return model.startswith(OLLAMA_PREFIX)


def _ollama_model_name(model: str) -> str:
    """Strip the 'ollama/' prefix to get the actual Ollama model name."""
    return model[len(OLLAMA_PREFIX):]


# ── Model availability check ────────────────────────────────────────

def check_model_available(model: str = DEFAULT_MODEL) -> None:
    """Verify the LLM backend is reachable."""
    if _is_ollama_model(model):
        _check_ollama_available(_ollama_model_name(model))
    else:
        if not os.environ.get("DEEPSEEK_API_KEY") and not os.environ.get("OPENROUTER_API_KEY"):
            raise RuntimeError(
                "No API key set. Add DEEPSEEK_API_KEY or OPENROUTER_API_KEY to .env.\n"
                "DeepSeek: https://platform.deepseek.com/api_keys\n"
                "OpenRouter: https://openrouter.ai/keys"
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


# ── JSON schema for structured output ────────────────────────────────

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


# ── Response parsing ─────────────────────────────────────────────────

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


def _coerce_llm_output(data: dict) -> dict:
    """Fix common LLM output mismatches before Pydantic validation."""
    if "line_items" in data and isinstance(data["line_items"], list):
        fixed_items = []
        for item in data["line_items"]:
            if not isinstance(item, dict):
                continue
            if "quantity" in item and "qty" not in item:
                item["qty"] = item.pop("quantity")
            if "name" in item and "description" not in item:
                item["description"] = item.pop("name")
            if not item.get("description"):
                continue
            if "total" not in item or item["total"] is None:
                continue
            # Coerce numeric fields to float (LLM may return strings)
            for nfield in ("total", "unit_price", "qty", "discount"):
                if nfield in item and item[nfield] is not None:
                    try:
                        item[nfield] = float(str(item[nfield]).replace(',', ''))
                    except (TypeError, ValueError):
                        pass
            # Coerce discount_rate null to empty string (schema requires str)
            if item.get("discount_rate") is None:
                item["discount_rate"] = ""
            # Coerce discount null to 0
            if item.get("discount") is None:
                item["discount"] = 0
            # Coerce tax_category to valid Literal values
            tc = item.get("tax_category")
            if tc is None or tc not in ("8%", "10%", "0%"):
                if tc and "8" in str(tc):
                    item["tax_category"] = "8%"
                elif tc and "10" in str(tc):
                    item["tax_category"] = "10%"
                else:
                    item["tax_category"] = "0%"
            fixed_items.append(item)
        data["line_items"] = fixed_items

    if "taxes" in data:
        taxes = data["taxes"]
        if isinstance(taxes, (int, float)):
            data["taxes"] = [{"rate": "unknown", "label": None, "amount": taxes}]
        elif isinstance(taxes, dict):
            data["taxes"] = [taxes]
        elif taxes is None:
            data["taxes"] = []

    # Fix Japanese era dates: years < 100 are era years (令和 N → 2018+N)
    if "date" in data and data["date"]:
        date_str = str(data["date"])
        m = re.match(r'^(\d{4})-(\d{2})-(\d{2})$', date_str)
        if m:
            year = int(m.group(1))
            if year < 100:
                data["date"] = f"{2018 + year:04d}-{m.group(2)}-{m.group(3)}"
            elif 2000 <= year <= 2018:
                era_year = year - 2000
                if 1 <= era_year <= 20:
                    data["date"] = f"{2018 + era_year:04d}-{m.group(2)}-{m.group(3)}"

    return data


def _parse_llm_json(raw: str) -> dict:
    """Parse LLM JSON output with Pydantic validation, coercion fallback."""
    try:
        receipt = Receipt.model_validate_json(raw)
        return receipt.model_dump()
    except Exception:
        try:
            data = json.loads(raw)
            data = _coerce_llm_output(data)
            receipt = Receipt(**data)
            return receipt.model_dump()
        except Exception as e2:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"error": f"LLM output failed validation: {e2}", "raw": raw}


# ── OpenRouter backend ───────────────────────────────────────────────

_api_client = None


def _get_api_client():
    """Get or create the API client (DeepSeek direct or OpenRouter fallback)."""
    global _api_client
    if _api_client is None:
        from openai import OpenAI
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if deepseek_key:
            _api_client = OpenAI(
                base_url="https://api.deepseek.com",
                api_key=deepseek_key,
                timeout=OLLAMA_TIMEOUT_SECONDS,
            )
        elif openrouter_key:
            _api_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
                timeout=OLLAMA_TIMEOUT_SECONDS,
            )
        else:
            raise RuntimeError(
                "No API key set. Add DEEPSEEK_API_KEY or OPENROUTER_API_KEY to .env."
            )
    return _api_client


def _openrouter_chat(
    model: str,
    messages: list,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> str:
    """Call DeepSeek/OpenRouter API and return the response content."""
    client = _get_api_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=temperature,
        max_tokens=max_tokens,
        seed=_LLM_SEED,
    )
    return response.choices[0].message.content


# ── Ollama backend ───────────────────────────────────────────────────

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
        t = threading.Thread(target=_call)
        t.start()
        t.join(timeout)
        if t.is_alive():
            raise TimeoutError(f"Ollama did not respond within {timeout}s")
        if error[0]:
            raise error[0]
        return result[0]  # type: ignore[return-value]


# ── Unified LLM chat ────────────────────────────────────────────────

def _llm_chat(
    model: str,
    messages: list,
    schema: dict,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> str:
    """Unified LLM chat — dispatches to Ollama or OpenRouter."""
    if _is_ollama_model(model):
        response = _ollama_chat_with_timeout(
            model=_ollama_model_name(model),
            messages=messages,
            format=schema,
            options={"temperature": temperature, "num_predict": max_tokens},
            think=False,
            keep_alive="60m",
        )
        return response["message"]["content"]
    else:
        return _openrouter_chat(model, messages, temperature, max_tokens)


# ── Extraction functions ─────────────────────────────────────────────

def extract_with_llm(ocr_text: str, model: str = DEFAULT_MODEL, doc_type: str = "receipt") -> dict:
    """Single-pass extraction with structured output enforcement."""
    prompt = generate_extraction_prompt(ocr_text, doc_type=doc_type)

    raw = _llm_chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        schema=get_ollama_schema(),
    )

    return _parse_llm_json(sanitize_llm_response(raw))


def extract_with_verification(
    ocr_text: str,
    model: str = DEFAULT_MODEL,
    passes: int = 1,
    validate_fn=None,
    doc_type: str = "receipt",
) -> tuple[dict, list[dict]]:
    """Multi-pass text extraction. Pass 1 extracts. Pass 2+ self-corrects."""
    passes = max(1, passes)
    history = []

    extracted = extract_with_llm(ocr_text, model=model, doc_type=doc_type)
    warnings = []
    if validate_fn and "error" not in extracted:
        try:
            receipt = Receipt(**extracted)
            warnings = validate_fn(receipt)
        except Exception:
            warnings = ["Schema validation failed on pass 1"]

    history.append({"pass": 1, "extraction": extracted, "warnings": warnings})

    for pass_num in range(2, passes + 1):
        if not warnings:
            break

        verification_prompt = generate_verification_prompt(
            ocr_text=ocr_text,
            previous_extraction=extracted,
            validation_warnings=warnings,
        )

        raw = _llm_chat(
            model=model,
            messages=[{"role": "user", "content": verification_prompt}],
            schema=get_ollama_schema(),
        )

        raw = sanitize_llm_response(raw)
        if raw.strip():
            extracted = _parse_llm_json(raw)

        warnings = []
        if validate_fn and "error" not in extracted:
            try:
                receipt = Receipt(**extracted)
                warnings = validate_fn(receipt)
            except Exception:
                warnings = [f"Schema validation failed on pass {pass_num}"]

        history.append({"pass": pass_num, "extraction": extracted, "warnings": warnings})

    return extracted, history
