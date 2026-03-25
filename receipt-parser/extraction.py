"""extraction.py — LLM extraction via Ollama structured output, multi-pass."""

import json
import re
import signal
import platform
import threading

import ollama as ollama_client
from schema import Receipt, generate_extraction_prompt, generate_verification_prompt

OLLAMA_TIMEOUT_SECONDS = 180


def check_ollama_available(model: str = "qwen3.5:9b") -> None:
    """Verify Ollama is running and the model is pulled. Call at pipeline init."""
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


def _ollama_chat_with_timeout(timeout: int = OLLAMA_TIMEOUT_SECONDS, **kwargs: object) -> dict:
    """Wrapper around ollama.chat() with a wall-clock timeout."""
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


def extract_with_llm(ocr_text: str, model: str = "qwen3.5:9b") -> dict:
    """Single-pass extraction with Ollama structured output enforcement."""
    prompt = generate_extraction_prompt(ocr_text)

    response = _ollama_chat_with_timeout(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format=get_ollama_schema(),
        options={"temperature": 0.0, "num_predict": 4096},
        think=False,
    )

    return _parse_llm_json(sanitize_llm_response(response["message"]["content"]))


def extract_with_verification(
    ocr_text: str,
    model: str = "qwen3.5:9b",
    passes: int = 1,
    validate_fn=None,
) -> tuple[dict, list[dict]]:
    """Multi-pass text extraction. Pass 1 extracts. Pass 2+ self-corrects."""
    passes = max(1, passes)
    history = []

    extracted = extract_with_llm(ocr_text, model=model)
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

        response = _ollama_chat_with_timeout(
            model=model,
            messages=[{"role": "user", "content": verification_prompt}],
            format=get_ollama_schema(),
            options={"temperature": 0.0, "num_predict": 4096},
            think=False,
        )

        raw = sanitize_llm_response(response["message"]["content"])
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
