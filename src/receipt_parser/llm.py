"""llm.py — LLM extraction via DeepSeek API (default), OpenRouter, or Ollama, multi-pass."""

import json
import os
import re
import signal
import platform
import threading
import time
from dataclasses import dataclass

import ollama as ollama_client
from .schema import Receipt, generate_extraction_prompt, generate_verification_prompt

OLLAMA_TIMEOUT_SECONDS = 180
DEFAULT_MODEL = "deepseek-chat"
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


# ── Model availability check ────────────────────────────────────────

def check_model_available(model: str = DEFAULT_MODEL) -> None:
    """Verify the LLM backend is reachable."""
    if _is_ollama_model(model):
        _check_ollama_available(_ollama_model_name(model))
    else:
        if not os.environ.get("DEEPSEEK_API_KEY") and not os.environ.get("OPENROUTER_API_KEY"):
            raise RuntimeError(
                "No API key configured.\n\n"
                "Quick fix:\n"
                "  1. Copy .env.example to .env:  cp .env.example .env\n"
                "  2. Add your DeepSeek API key:   DEEPSEEK_API_KEY=sk-...\n"
                "     Get one at: https://platform.deepseek.com/api_keys\n\n"
                "Or run the setup wizard:  receipt-parser setup"
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
    """Parse LLM JSON output with Pydantic validation, raw dict fallback.

    Extracts _confidence before Pydantic validation (not part of schema).
    Coercion is handled by Pydantic's model_validator in schema.py.
    """
    confidence = None
    try:
        data = json.loads(raw)
        confidence = _extract_confidence(data)
        receipt = Receipt(**data)
        result = receipt.model_dump()
        if confidence:
            result["_confidence"] = confidence
        return result
    except Exception as e:
        # Pydantic validation failed — fall back to raw dict
        try:
            data = json.loads(raw)
            confidence = _extract_confidence(data)
            if confidence:
                data["_confidence"] = confidence
            return data
        except json.JSONDecodeError:
            return {"error": f"LLM output failed validation: {e}", "raw": raw}


# ── OpenRouter backend ───────────────────────────────────────────────

_api_client = None
_instructor_client = None
_client_lock = threading.Lock()


def _get_api_client():
    """Get or create the API client (DeepSeek direct or OpenRouter fallback)."""
    global _api_client
    if _api_client is not None:
        return _api_client
    with _client_lock:
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


def _openrouter_chat(
    model: str,
    messages: list,
    temperature: float = 0.0,
    max_tokens: int = 8192,
) -> LLMResult:
    """Call DeepSeek/OpenRouter API and return structured result."""
    client = _get_api_client()
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=temperature,
        max_tokens=max_tokens,
        seed=_LLM_SEED,
    )
    elapsed_ns = int((time.perf_counter() - t0) * 1e9)
    usage = response.usage
    input_toks = usage.prompt_tokens if usage else None
    output_toks = usage.completion_tokens if usage else None

    # DeepSeek returns cache hit/miss breakdown in usage
    cache_hit = getattr(usage, "prompt_cache_hit_tokens", None) if usage else None
    cache_miss = getattr(usage, "prompt_cache_miss_tokens", None) if usage else None
    # Fallback: if cache fields not available, treat all input as cache miss
    if cache_hit is None and cache_miss is None and input_toks:
        cache_hit = 0
        cache_miss = input_toks

    # Track DeepSeek token usage
    from .usage import track_deepseek_call
    track_deepseek_call(cache_hit, cache_miss, output_toks)

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
        )
        elapsed_ns = int((time.perf_counter() - t0) * 1e9)
        llm_result = LLMResult(
            content="(instructor)",
            eval_duration_ns=elapsed_ns,
            total_duration_ns=elapsed_ns,
            backend="api",
        )
        return receipt, llm_result
    except Exception:
        return None, None


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
    max_tokens: int = 8192,
) -> LLMResult:
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
        return _openrouter_chat(model, messages, temperature, max_tokens)


# ── Extraction functions ─────────────────────────────────────────────

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

    return parsed, llm_result


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
) -> tuple[dict, list[dict]]:
    """Multi-pass text extraction. Pass 1 extracts. Pass 2+ self-corrects."""
    passes = max(1, passes)
    history = []

    extracted, llm_result = extract_with_llm(ocr_text, model=model, doc_type=doc_type)
    warnings = []
    if validate_fn and "error" not in extracted:
        try:
            receipt = Receipt(**extracted)
            warnings = validate_fn(receipt)
        except Exception:
            warnings = ["Schema validation failed on pass 1"]

    history.append({
        "pass": 1, "extraction": extracted, "warnings": warnings,
        "llm_timing": _llm_result_to_timing(llm_result),
    })

    for pass_num in range(2, passes + 1):
        if not warnings:
            break

        v_system, v_user = generate_verification_prompt(
            ocr_text=ocr_text,
            previous_extraction=extracted,
            validation_warnings=warnings,
        )

        llm_result = _llm_chat(
            model=model,
            messages=[
                {"role": "system", "content": v_system},
                {"role": "user", "content": v_user},
            ],
            schema=get_ollama_schema(),
        )

        raw = sanitize_llm_response(llm_result.content)
        if raw.strip():
            extracted = _parse_llm_json(raw)

        warnings = []
        if validate_fn and "error" not in extracted:
            try:
                receipt = Receipt(**extracted)
                warnings = validate_fn(receipt)
            except Exception:
                warnings = [f"Schema validation failed on pass {pass_num}"]

        history.append({
            "pass": pass_num, "extraction": extracted, "warnings": warnings,
            "llm_timing": _llm_result_to_timing(llm_result),
        })

    return extracted, history
