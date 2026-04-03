# katanaml/sparrow

> https://github.com/katanaml/sparrow

| Field | Value |
|-------|-------|
| Stars | ~5,100 |
| Language | Python |
| License | GPL-3.0 (free for orgs under $5M revenue) |
| Last Updated | 2026-04-02 |
| Approach | Pluggable Vision LLM document extraction framework |

---

## 1. Overview

Sparrow is a modular framework for extracting structured data from documents (invoices, receipts, statements, forms) using large language models. Unlike traditional OCR pipelines, Sparrow's core thesis is that Vision LLMs can directly parse document images into structured JSON without explicit OCR stages. It acts as an orchestration layer across multiple LLM backends.

The project is at version 0.4.4 and comprises five components:
- **Sparrow ML LLM** -- Primary API engine
- **Sparrow Parse** -- Vision LLM library for JSON extraction from images
- **Sparrow Agents** -- Workflow orchestration for multi-step processing
- **Sparrow OCR** -- Text recognition preprocessing (optional)
- **Sparrow UI** -- Web interface for interactive document handling

---

## 2. Architecture & How It Works

### Code Structure

```
sparrow-ml/llm/
  engine.py           # Core orchestration - factory pattern for pipeline selection
  api.py              # REST API endpoints
  assistant.py        # Assistant/chat functionality
  config_utils.py     # Configuration management
  db_pool.py          # Database connection pooling
  pipelines/
    interface.py      # Pipeline interface definition
    sparrow_parse/    # Vision LLM extraction
      sparrow_parse.py
      sparrow_table.py
      sparrow_markdown.py
      sparrow_validator.py
      sparrow_experimental.py
      sparrow_utils.py
      table_templates/
    instructor/       # Text LLM pipeline
    sparrow_instructor/ # Sparrow-specific instructor variant
```

### Pipeline Flow

1. **Engine (`engine.py`)** receives a request (CLI or API) with document path, query/schema, and pipeline selection
2. **Factory Pattern**: `get_pipeline(user_selected_pipeline)` instantiates the appropriate backend
3. **Three extraction modes**:
   - **Sparrow Parse**: Vision LLM processes document image directly -> JSON output
   - **Instructor**: Text LLM processes OCR text with instruction following
   - **Agent Pipeline**: Multi-step orchestrated workflow for complex documents
4. **Schema Validation**: JSON hint files define expected output structure; wildcard `"*"` extracts all fields
5. **Post-processing**: Validation, normalization, and optional table template matching

### Supported Backends

| Backend | Type | Platform |
|---------|------|----------|
| Mistral | Vision LLM | Cloud API |
| Qwen 2.5-VL-72B | Vision LLM | Local/Cloud |
| DeepSeek OCR | Vision LLM | Cloud API |
| dots.ocr / dots-mocr | Vision LLM | Cloud API |
| GPT-OSS | Text LLM | Cloud API |
| Qwen 3.5 | Text LLM | Local |
| MLX | Inference | Apple Silicon |
| Ollama | Inference | Local |
| vLLM | Inference | GPU |
| Hugging Face | Inference | Cloud |

### Key Technical Details

- Disables HuggingFace tokenizer parallelism to prevent deadlocks
- Image cropping with configurable size for large documents
- Multi-page PDF support
- Async file handling for API mode
- Model caching across requests
- Temporary directory management for secure file processing

---

## 3. Key Features

- **Backend agnostic**: Swap between Mistral, Qwen, DeepSeek, local Ollama, etc. with config change
- **Schema-driven extraction**: JSON schema defines what to extract; supports wildcards
- **Table extraction**: Dedicated `sparrow_table.py` with templates for common table formats
- **Markdown pipeline**: Can extract structured data from markdown-converted documents
- **Validation layer**: `sparrow_validator.py` provides post-extraction validation
- **API-first design**: REST endpoints for integration
- **Multi-step agents**: Complex extraction workflows with chained LLM calls

---

## 4. Japanese Support

**Not explicitly supported.** The README and examples are entirely English-focused. No mention of:
- CJK character handling
- Japanese-specific formatting (yen, era dates)
- Multilingual OCR configuration
- Right-to-left or vertical text

However, since Sparrow delegates to Vision LLMs (Qwen, Mistral, etc.), Japanese support would depend on the underlying model's multilingual capabilities. Qwen 2.5-VL-72B and DeepSeek OCR both handle Japanese well, so it would likely work -- but without explicit testing or optimization.

---

## 5. Strengths vs Our Project

| Area | Sparrow Advantage |
|------|-------------------|
| **Backend Flexibility** | Plug in any Vision LLM (Mistral, Qwen, DeepSeek, etc.) with a config change. We're locked to DeepSeek V3.2. |
| **Vision LLM Approach** | Skips explicit OCR entirely -- sends images directly to VLMs. Eliminates OCR error propagation. |
| **Agent Orchestration** | Multi-step agent workflows for complex documents. Our pipeline is single-pass OCR -> LLM. |
| **API Layer** | Production-ready REST API with async file handling and DB pooling. We have no API layer. |
| **Table Extraction** | Dedicated table parsing with templates. We don't handle tabular receipt data. |
| **Schema Wildcards** | `"*"` wildcard extracts all fields without predefined schema. Our extraction requires explicit field definitions. |

---

## 6. Weaknesses vs Our Project

| Area | Our Advantage |
|------|---------------|
| **Receipt Domain Expertise** | We have merchant rules, tax categories, subset-sum matching, field registry -- deep receipt-specific logic. Sparrow is generic. |
| **Confidence Routing** | Our OCR vs LLM confidence routing per field. Sparrow trusts the LLM output directly. |
| **Japanese Specialization** | Era dates, yen formatting, kanji normalization, Japanese utility bill parsing. Sparrow has zero Japanese-specific logic. |
| **Multi-Pass Verification** | Our LLM verification pass catches extraction errors. Sparrow does single-pass extraction. |
| **Determinism** | seed=42, variance attribution benchmarks. Sparrow has no determinism guarantees. |
| **Ground Truth Testing** | 36 fixtures with truth files, accuracy benchmarks with robustness testing. Sparrow has no equivalent test infrastructure. |
| **Post-Processing** | Text normalization, field-specific cleanup, validation. Sparrow's validation is lighter. |
| **License** | Our pipeline is proprietary. Sparrow is GPL-3.0 which requires derivative works to be GPL. |

---

## 7. What We Can Learn

### 7.1 Vision LLM Direct Extraction
Sparrow's core insight -- skip OCR, send images directly to VLMs -- is worth testing. Modern Vision LLMs (Qwen-VL, DeepSeek-VL) can read receipts directly. This could:
- Eliminate OCR error propagation
- Remove GCV API dependency
- Simplify the pipeline

**Test**: Send our 36 fixture images directly to DeepSeek V3.2 (if it supports vision) or Qwen-VL and compare accuracy against our OCR->LLM pipeline.

### 7.2 Backend Abstraction Layer
Sparrow's factory pattern for pipeline selection is clean. We could add a similar abstraction to support multiple LLM backends (DeepSeek, Qwen, local Ollama) with a config switch, enabling:
- Fallback to local models when API is down
- A/B testing of different LLMs
- Cost optimization (cheap model for easy receipts, expensive model for hard ones)

### 7.3 Schema-Driven Extraction
Sparrow's JSON hint files and wildcard extraction are elegant. We could externalize our field definitions into JSON schemas that drive extraction, rather than hard-coding them in the prompt.

### 7.4 Table Template Matching
`sparrow_table.py` with `table_templates/` suggests pre-defined patterns for common table formats. We could create similar templates for common Japanese receipt formats (konbini, supermarket, restaurant).

---

## 8. Recommendation

**Study for architectural ideas, don't adopt directly.** Sparrow is architecturally interesting but not a good fit for direct adoption because:

1. **GPL-3.0 license** is restrictive for commercial use above $5M revenue
2. **No Japanese support** would require significant work to add
3. **Generic extraction** lacks the receipt-domain expertise we've built
4. **Less rigorous** testing/benchmarking infrastructure

However, two ideas are worth stealing:
- **Vision LLM bypass**: Test whether sending receipt images directly to a VLM matches our OCR->LLM pipeline accuracy
- **Backend abstraction**: Add a factory pattern to support multiple LLM backends with easy swapping

Do not integrate Sparrow as a dependency. Build equivalent patterns ourselves.
