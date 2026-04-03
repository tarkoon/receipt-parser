# 08. receipt-ocr (bhimrazy/receipt-ocr)

**GitHub:** https://github.com/bhimrazy/receipt-ocr
**PyPI:** `pip install receipt-ocr`

---

## 1. Overview

receipt-ocr is a lightweight Python package for extracting structured data from receipt images using LLM vision APIs. It wraps OpenAI-compatible vision endpoints (GPT-4o, Gemini, Groq) behind a simple CLI and FastAPI service. It is the most architecturally similar project to ours among the five being reviewed, but far simpler.

- **Stars:** ~214
- **Language:** Python
- **License:** MIT
- **Last updated:** Active (69 commits on main)
- **Approach:** Send receipt image directly to a vision LLM (GPT-4o etc.) with a JSON schema, get structured JSON back. No separate OCR step.

This is essentially a "send image to GPT-4o and ask for JSON" wrapper with nice packaging.

---

## 2. Architecture & How It Works

### Pipeline (extremely simple)

```
Receipt Image -> base64 encode -> Vision LLM API call -> JSON parse -> Structured output
```

There is no OCR step, no text normalization, no post-processing, no validation. The entire extraction is delegated to the vision LLM.

### Core Components

**`ReceiptProcessor`** (`src/receipt_ocr/processors.py`):
```python
class ReceiptProcessor:
    def __init__(self, provider=None, parser=None):
        self.provider = provider or OpenAIProvider()
        self.parser = parser or ReceiptParser()

    def process_receipt(self, image_path, json_schema, model=None, response_format_type=None):
        response = self.provider.get_response(image_path, json_schema, model, response_format_type)
        content = response.choices[0].message.content
        return self.parser.parse(content)
```

**`OpenAIProvider`** (`src/receipt_ocr/providers.py`):
- Abstract base `LLMProvider` with concrete `OpenAIProvider`
- Accepts `api_key`, `base_url` from env vars
- Encodes images to base64 for API transmission
- Temperature: 0.2 (not 0)
- Supports 3 response format types: `json_object`, `json_schema`, `text`
- Uses OpenAI SDK (`openai==2.29`)

**CLI** (`src/receipt_ocr/cli.py`):
- `receipt-ocr <image_path>` with optional `--schema_path`, `--model`, `--api_key`, `--base_url`
- Loads default schema if none provided
- Outputs JSON to stdout

**FastAPI Service** (`app/`):
- `POST /ocr/` -- multipart file upload
- `GET /health` -- health check
- Docker Compose support

### Default Output Schema

```json
{
  "merchant_name": "string",
  "merchant_address": "string",
  "transaction_date": "YYYY-MM-DD",
  "transaction_time": "HH:MM:SS",
  "total_amount": "number",
  "line_items": [
    {
      "item_name": "string",
      "item_quantity": "number",
      "item_price": "number",
      "item_total": "number"
    }
  ]
}
```

### Second Module: Tesseract OCR

A separate `src/tesseract_ocr/` module provides raw Tesseract-based text extraction with its own FastAPI endpoint and Docker container. This is independent from the LLM-based receipt extraction.

### Dependencies

- `openai==2.29` (sole LLM dependency)
- `pillow==12.1.1`
- `python-dotenv==1.2.2`
- Dev: pytest, coverage, codecov

### Repository Structure

```
src/
  receipt_ocr/
    processors.py    # ReceiptProcessor class (~30 lines)
    providers.py     # LLMProvider ABC + OpenAIProvider (~84 lines)
    cli.py           # CLI entry point
    parsers.py       # JSON response parsing
  tesseract_ocr/
    main.py          # Tesseract-based extraction
app/
  docker-compose.yml
tests/
  receipt_ocr/
  tesseract_ocr/
images/              # Sample receipt images
pyproject.toml       # hatchling build, receipt-ocr CLI entry point
```

---

## 3. Key Features

- **Pip-installable**: `pip install receipt-ocr` -- ready to use in seconds
- **Provider abstraction**: Abstract `LLMProvider` class makes it easy to swap LLM backends
- **Custom JSON schema**: Pass any schema to extract any fields from any receipt
- **Three output modes**: `json_object`, `json_schema` (strict), `text`
- **FastAPI service**: Ready-made REST API for receipt processing
- **Docker support**: Containerized deployment
- **CLI tool**: Single-command receipt processing
- **Clean code**: Pre-commit hooks, ruff linting, pytest, coverage tracking

---

## 4. Japanese Support

**None.** There is no mention of Japanese language support anywhere in the project. The schema uses English field names with Western date/time formats (YYYY-MM-DD, HH:MM:SS). The project relies entirely on the vision LLM's ability to read non-English text, with no language-specific handling, normalization, or post-processing.

If used with GPT-4o on a Japanese receipt:
- Basic text extraction would likely work (GPT-4o handles Japanese)
- Date formats would be wrong (no era date conversion)
- Tax calculations would not be verified
- Merchant name normalization would not occur
- Zenkaku/hankaku number issues would not be handled

---

## 5. Strengths vs Our Project

| Their Strength | Detail |
|----------------|--------|
| **Simplicity** | ~30 lines of core logic vs our multi-stage pipeline. Extremely easy to understand and maintain |
| **Pip-installable** | `pip install receipt-ocr` -- we have no package distribution |
| **Provider abstraction** | Clean `LLMProvider` ABC makes backend swapping trivial; our LLM integration is more coupled |
| **Custom schema** | Users can pass any JSON schema for any extraction task; our schema is hardcoded to our receipt format |
| **FastAPI service** | Ready-made REST API; we have no web service layer |
| **Docker deployment** | Production-ready containers; we have no containerization |
| **CLI tool** | `receipt-ocr image.jpg` is simpler than our invocation |
| **Build tooling** | Modern Python packaging (hatchling, uv, ruff, pre-commit) |

---

## 6. Weaknesses vs Our Project

| Our Strength | Detail |
|--------------|--------|
| **Accuracy** | Single LLM call with no verification vs our multi-pass extraction + validation |
| **No OCR stage** | Relies entirely on vision LLM's OCR capability; we use dedicated Google Cloud Vision |
| **No post-processing** | No tax verification, no subset-sum matching, no field normalization |
| **No confidence scoring** | No way to know if the extraction is reliable; we have OCR + LLM confidence routing |
| **No retry logic** | Single-shot extraction; we have confidence-gated retry |
| **Temperature 0.2** | Not deterministic; we use seed=42 + temperature=0 |
| **No Japanese handling** | No era dates, no zenkaku conversion, no merchant rules |
| **No test fixtures** | No ground truth comparison; we have 36 fixture-based accuracy tests |
| **No normalization** | Raw LLM output returned as-is; we normalize dates, amounts, categories |
| **Shallow schema** | Flat structure (merchant, date, total, items); we have nested categories, tax breakdowns, payment methods, location data |
| **API cost** | GPT-4o vision calls are expensive ($2.50-10/1K images); we use DeepSeek V3.2 (much cheaper) |
| **No batch processing** | One image at a time; no benchmark or bulk processing support |

---

## 7. What We Can Learn

1. **Provider abstraction pattern**: Their `LLMProvider` ABC with `OpenAIProvider` concrete implementation is clean and extensible. We should consider a similar pattern if we ever want to support multiple LLM backends (DeepSeek, GPT-4o, Gemini, local models). The key insight is using the OpenAI SDK as the universal interface since most providers offer OpenAI-compatible endpoints.

2. **Pip-installable package**: Their use of hatchling + `pyproject.toml` with a CLI entry point (`receipt-ocr = receipt_ocr.cli:main`) is a pattern we should adopt. Our project could benefit from being installable via `pip install -e .` with a proper entry point.

3. **Custom JSON schema parameter**: Allowing users to pass their own extraction schema at runtime is a powerful feature. We could expose a similar option for ad-hoc extraction tasks beyond our standard receipt schema.

4. **FastAPI service layer**: If we ever need to serve receipt parsing as an API, their `POST /ocr/` endpoint pattern with multipart file upload is a clean reference implementation.

5. **Response format types**: Their support for `json_object`, `json_schema` (strict), and `text` modes is worth noting. The `json_schema` mode with strict validation is particularly relevant -- it forces the LLM to match the schema exactly, which could reduce our post-processing burden.

6. **Modern Python packaging**: hatchling build backend, uv package manager, ruff linter, pre-commit hooks. Our project should modernize similarly.

---

## 8. Recommendation

**Not useful as a tool, but useful as a design reference for packaging and API design.**

receipt-ocr is architecturally too simple for our needs -- it's essentially a thin wrapper around GPT-4o vision. The extraction quality will be limited by whatever the vision LLM can do in a single pass with no verification.

**What we should borrow:**
- **Provider abstraction**: Implement `LLMProvider` ABC in our codebase for backend flexibility
- **Package structure**: Adopt hatchling + pyproject.toml for pip-installable distribution
- **FastAPI service**: Use as reference when we build our own API layer
- **CLI entry point**: Add a `receipt-parser` CLI command to our package
- **`json_schema` response format**: Use OpenAI's strict JSON schema mode where supported to reduce parsing errors

**What we should NOT adopt:**
- Single-shot extraction with no verification
- Vision-only approach (no dedicated OCR stage)
- Temperature 0.2 (too much randomness for production)
- Their shallow receipt schema (insufficient for Japanese receipts)
