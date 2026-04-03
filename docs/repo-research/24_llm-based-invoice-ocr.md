# 24. ShafqaatMalik/llm-based-invoice-ocr

**URL:** https://github.com/ShafqaatMalik/llm-based-invoice-ocr
**Stars:** 3 | **Forks:** 0 | **Language:** Python 100%
**Created:** 2025-09-30 | **Last Updated:** 2026-02-02
**License:** None | **Commits:** 2
**Topics:** python, computer-vision, artificial-intelligence, data-extraction, tesseract-ocr, gradio, fastapi, invoice-parser, llm

---

## 1. Overview

llm-based-invoice-ocr is a hybrid invoice data extraction system offering dual processing modes: a "paid" mode using Together AI's Qwen2.5-VL-72B-Instruct vision-language model, and an "open-source" mode using Tesseract OCR with regex heuristics. The system provides both a FastAPI REST backend and a Gradio web frontend, making it accessible via API or browser interface.

Despite being very new (created September 2025) with only 2 commits, it is a well-structured project with clear architecture and good documentation. The dual-mode approach (paid VLM vs free OCR) is a pragmatic design choice.

---

## 2. Architecture & How It Works

### Pipeline Flow

```
Invoice (PDF or Image)
  -> FastAPI Backend (/extract_invoice/ endpoint)
  -> File validation + temp storage
  -> PDF: pdf2image conversion at 300 DPI (Poppler)
  -> Image: direct copy
  -> Mode Selection:
     |
     +-> "paid" mode:
     |     -> Base64 encode image
     |     -> Together AI API (Qwen2.5-VL-72B-Instruct)
     |     -> JSON extraction from response (with regex fallback)
     |
     +-> "open_source" mode:
           -> pytesseract OCR
           -> Regex pattern matching (invoice#, dates, totals, vendors)
           -> Line items parsing (text + trailing numbers)
  -> aggregate_results() (multi-page merge)
  -> validate_invoice_json() (schema compliance)
  -> JSON response + page count
```

### Core Components

| File | Purpose |
|------|---------|
| `src/backend/main.py` | FastAPI app -- CORS, file upload, PDF conversion, mode routing, cleanup |
| `src/backend/together_api.py` | Together AI integration -- base64 encoding, API calls, JSON extraction |
| `src/backend/open_source_mode.py` | Tesseract OCR + regex heuristics baseline extractor |
| `src/backend/parser.py` | Post-processing -- multi-page aggregation, schema validation |
| `src/frontend/gradio_app.py` | Gradio UI -- file upload, mode selection, JSON display |
| `scripts/generate_complex_invoices.py` | Utility to generate test invoices |
| `sample_invoices/` | 11 test invoices (PDF, PNG, JPG) including multi-page |

### Paid Mode: Together AI + Qwen2.5-VL-72B

The vision-language model approach:

1. **Image encoding:** Invoice images converted to base64 `data:image/png;base64,...` format
2. **API request:** Chat Completions format with system prompt defining the JSON schema, image content block, temperature=0.1 for consistency
3. **Response parsing:** Attempts direct JSON parse, falls back to regex extraction of JSON objects from response text
4. **Error handling:** Returns minimal schema with "error" field on failure, allowing pipeline continuation
5. **Token limit:** 1000 max tokens per response

### Open-Source Mode: Tesseract + Regex

The fallback extractor:

1. **OCR:** `pytesseract.image_to_string()` for full-page text extraction
2. **Field detection:** Regex patterns for invoice number (`Invoice\s*#?\s*(\w+)`), dates, totals, vendor names
3. **Line items:** Lines ending with numeric values split into description + amount
4. **Raw text preserved:** The raw OCR text is included in output for debugging

### Multi-Page Aggregation

The `aggregate_results()` function intelligently merges across pages:
- **Scalar fields** (vendor, invoice#): first non-empty value wins
- **Line items:** concatenated across all pages
- **Financial totals:** last non-empty value preferred (typically on final page)

### Schema Validation

The `validate_invoice_json()` function enforces compliance:
- Missing keys filled with defaults
- Type coercion (non-strings to strings, non-lists to empty lists)
- Ensures consistent output shape regardless of extraction quality

### Gradio Frontend

Polished UI with:
- File upload (PDF, PNG, JPG, JPEG)
- Mode selection: "AI (High Accuracy)" vs "OCR (Fast & Free)"
- Custom olive-green/orange CSS theme
- Event handlers that clear results on file/mode changes
- JSON output panel

---

## 3. Key Features

- **Dual processing modes** -- paid VLM for accuracy, free OCR for cost/speed. Users choose per-request.
- **Vision-language model (Qwen2.5-VL-72B-Instruct)** -- sends the image directly to the model, no separate OCR step needed for paid mode
- **Multi-page PDF support** -- converts each page to 300 DPI PNG, processes individually, aggregates results
- **FastAPI + Gradio dual interface** -- both API and browser access
- **Cross-platform Poppler configuration** -- handles Windows, macOS, Linux path differences
- **Schema validation layer** -- ensures consistent JSON output regardless of extraction quality
- **UUID-based temp files** -- prevents filename collisions in concurrent processing
- **Sample invoices included** -- 11 test documents for immediate evaluation
- **Graceful degradation** -- extraction errors return a minimal valid schema instead of crashing

---

## 4. Japanese Support

**None explicitly, but partially possible.** Analysis:

- **Qwen2.5-VL-72B paid mode:** Qwen models have strong CJK support. The VL model could potentially process Japanese invoices since it "sees" the image directly. However, the JSON schema prompt is English-only and the expected fields (vendor_name, invoice_number, line_items) are Western invoice-focused.
- **Tesseract open-source mode:** pytesseract supports Japanese via `lang='jpn'` parameter, but the code does not pass any language parameter. The regex patterns are English-specific (e.g., `Invoice\s*#?`).
- **No yen/era date handling:** No currency-aware processing, no Japanese era date parsing.
- **Conclusion:** The paid mode could theoretically handle Japanese with prompt modifications, but the open-source mode would fail without language configuration.

---

## 5. Strengths vs Our Project

- **Vision-language model approach (paid mode):** Sending the image directly to Qwen2.5-VL-72B bypasses OCR entirely for the paid mode. This is similar to the Donut approach but uses a general-purpose VLM instead of a fine-tuned model. No OCR confidence issues, no text ordering problems.
- **Dual-mode flexibility:** Users can choose accuracy vs cost per request. We only have one pipeline path. A similar toggle in our system could let users choose between high-accuracy (multi-pass) and fast (single-pass) modes.
- **FastAPI + Gradio architecture:** Clean separation of backend (API) and frontend (web UI). Our project lacks a web interface entirely. Their architecture pattern is good for when we eventually add a UI.
- **Multi-page document handling:** Their page-by-page processing with intelligent aggregation handles multi-page documents. Our pipeline processes single images. The "first non-empty wins for headers, concatenate for line items, last wins for totals" aggregation logic is well-thought-out.
- **Schema validation as a safety net:** Their `validate_invoice_json()` ensures every response has a consistent shape. This is a good defensive pattern we could adopt.
- **Test invoice generation:** The `generate_complex_invoices.py` script creates test data, reducing dependency on real documents.

---

## 6. Weaknesses vs Our Project

- **Only 2 commits:** This project is essentially a proof-of-concept with no iteration history. Our 51+ commits show extensive refinement.
- **No confidence scoring:** No OCR confidence, no extraction confidence, no field-level quality assessment. Our confidence routing is a major differentiator.
- **No multi-pass verification:** Single extraction attempt per page. Our multi-pass LLM verification catches errors.
- **No post-processing intelligence:** No tax calculation, no subset-sum matching, no merchant rules. Just raw extraction + schema validation.
- **Tesseract as fallback:** Tesseract is significantly worse than Google Cloud Vision for complex documents, especially non-English text. Our Google Cloud Vision integration is superior.
- **No test suite:** No unit tests, no accuracy benchmarks, no ground truth fixtures. Our 36-fixture test suite with robustness benchmarks is far more rigorous.
- **No determinism controls:** Temperature=0.1 is not zero, and no seed parameter. Our seed=42 approach is more deterministic.
- **Together AI dependency:** Requires Together AI subscription for the good mode. Our DeepSeek API is cheaper.
- **Hardcoded 1000 token limit:** May truncate complex invoices. No dynamic token allocation.
- **Invoice-only:** Designed for invoices specifically. Our pipeline handles receipts, utility bills, and payment slips.
- **No Japanese support:** English-only prompts and regex patterns.

---

## 7. What We Can Learn

1. **VLM as an alternative extraction path:** Their paid mode sends images directly to Qwen2.5-VL, bypassing OCR. We could add a VLM mode to our pipeline where, for low-confidence OCR results, we send the original image to a vision-language model (DeepSeek-VL or Qwen-VL) for direct extraction. This would give us a third extraction strategy alongside OCR+LLM and OCR-retry.

2. **Multi-page document aggregation:** Their aggregation logic ("first non-empty for headers, concatenate for items, last for totals") is a clean pattern we should implement when/if we add multi-page document support.

3. **Dual-mode processing toggle:** Offering users a choice between high-accuracy (expensive, multi-pass) and fast (cheap, single-pass) processing could be valuable. This maps well to our existing confidence routing -- we could expose it as a user-facing quality/speed tradeoff.

4. **Schema validation safety net:** Their `validate_invoice_json()` that fills missing keys with defaults and coerces types is a good defensive layer. We could add a similar post-validation step that ensures every output field exists and has the correct type, even when the LLM produces partial results.

5. **FastAPI + Gradio pattern:** When we build a web UI, the FastAPI backend + Gradio frontend pattern is clean and well-documented. Gradio is particularly good for ML-focused interfaces with file upload + JSON display.

---

## 8. Recommendation

**Do not adopt this tool.** It is a 2-commit proof-of-concept with no tests, no Japanese support, and no production hardening. However, several architectural ideas are worth borrowing:

- **VLM extraction mode:** Add an optional vision-language model path for cases where OCR confidence is very low, sending the raw image to a VLM for direct extraction
- **Multi-page aggregation logic:** Implement their "first-for-headers, concat-for-items, last-for-totals" pattern for future multi-page support
- **Output schema validation:** Add a defensive validation layer that ensures consistent JSON shape regardless of extraction quality
- **Dual-mode user toggle:** Consider exposing a quality/speed tradeoff to users
