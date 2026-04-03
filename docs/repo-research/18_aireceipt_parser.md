# 18. JustCabaret/AIReceiptParser

**Repository:** https://github.com/JustCabaret/AIReceiptParser
**Stars:** ~43 | **Language:** Python (36.4%), HTML (23.3%), JS (20.3%), CSS (20%) | **Last Updated:** Dec 2024
**License:** MIT | **Python:** 3.10+

## Overview

A full-stack web application that automates receipt parsing using Tesseract OCR + GPT-4o-mini + SQLite + Flask. It provides a responsive web UI for uploading receipt images, viewing parsed results, and browsing historical receipts. The project includes a clever "test mode" that bypasses OCR and GPT APIs for rapid frontend development. It is a well-structured but simple implementation -- a good example of a complete receipt parsing web app, though it lacks depth in OCR quality and extraction accuracy.

## Architecture & How It Works

```
Frontend (Vanilla JS + HTML + CSS)
    |
    v  POST /process_receipt (multipart form with image + API key)
    |
Flask Backend (app.py -> routes.py -> controllers.py)
    |
    v
controllers.py: process_receipt()
    |
    +--[if api_key == "test"]--> Return mock data (no API calls)
    |
    +--[else]-->
        |
        v
    image_processing.py: preprocess_image()
      - cv2.cvtColor (BGR -> grayscale)
      - cv2.threshold (Otsu's binarization)
        |
        v
    text_extraction.py: extract_text()
      - pytesseract.image_to_string(image)
      - No language specification (defaults to English)
        |
        v
    gpt_processing.py: process_text_with_chatgpt()
      - Model: gpt-4o-mini
      - JSON mode enabled (response_format={"type": "json_object"})
      - Prompt: categorize items, clarify product names, convert to cents
        |
        v
    database.py: insert_receipt()
      - SQLite: receipts table + receipt_items table
      - Returns receipt_id
        |
        v
    JSON response to frontend
```

**API Endpoints:**
- `POST /process_receipt` -- Upload image, get parsed JSON back
- `GET /receipts` -- List all stored receipts
- `GET /receipts/<id>` -- Get line items for a specific receipt

**Database Schema:**
- `receipts` (id, total, source/store, parsed_at timestamp)
- `receipt_items` (id, receipt_id FK, product, quantity, price, category)

## Key Features

1. **Full-stack web app** -- Complete Flask backend + vanilla JS frontend with responsive design. Upload receipts, view results, browse history.
2. **GPT-4o-mini with JSON mode** -- Uses OpenAI's `response_format={"type": "json_object"}` for reliable structured output. The prompt instructs GPT to categorize items (Groceries, Household, Personal Care, Electronics, Others).
3. **Test/mock mode** -- Entering "test" as the API key returns hardcoded sample data. Brilliant for frontend development without API costs. This pattern is worth stealing.
4. **SQLite persistence** -- Receipts and line items stored with proper foreign key relationships. Supports historical browsing.
5. **Product name improvement** -- The GPT prompt instructs "Improve product titles: clarify incomplete names while keeping the original language" -- an interesting use of GPT for OCR error correction.
6. **Price normalization to cents** -- All prices stored as integers (cents) to avoid floating-point issues.
7. **CORS enabled** -- Flask-CORS configured for cross-origin requests, making API integration flexible.
8. **Clean separation of concerns** -- Routes -> Controllers -> Modules pattern with proper Flask app factory.

## Japanese Support

**None.** The project has zero Japanese language support:
- Tesseract is called without specifying `lang=` parameter (defaults to English)
- GPT prompt says "keep the original language" but the categories are English-only
- No Japanese-specific preprocessing, normalization, or handling
- All documentation in English/Portuguese
- The SQLite schema has no encoding considerations

## Strengths vs Our Project

1. **Complete web application** -- This is the main advantage. It has a working frontend, REST API, database persistence, and historical receipt browsing. Our project is CLI/library only.
2. **Test/mock mode pattern** -- The ability to bypass expensive API calls during development with `api_key == "test"` is elegant and saves developer time and money. We don't have this.
3. **GPT product name improvement** -- Using GPT to not just extract but also *improve* product names from OCR output is an interesting post-processing step we haven't considered.
4. **Price-as-integer storage** -- Storing amounts in cents avoids floating-point precision issues. A good database design pattern.
5. **App factory pattern** -- Clean Flask setup with blueprints, CORS, and dotenv. Good reference architecture for a future web interface.

## Weaknesses vs Our Project

1. **Terrible OCR** -- Tesseract without language specification, with only grayscale + Otsu binarization preprocessing. This will produce poor results on any non-trivial receipt.
2. **No confidence scoring** -- No OCR confidence, no LLM confidence, no way to know if the result is reliable.
3. **No validation** -- The GPT output is parsed with `json.loads()` but never validated against a schema. No Pydantic, no field validation, no range checks.
4. **Single-pass extraction** -- One Tesseract call + one GPT call. No retry logic, no multi-pass verification, no confidence routing.
5. **No test suite** -- Zero tests. 14 commits total.
6. **Naive image preprocessing** -- Only grayscale + Otsu threshold. No deskewing, denoising, or adaptive processing.
7. **No date extraction or normalization** -- The GPT prompt doesn't specifically ask for dates in any normalized format.
8. **No tax handling** -- No tax calculation, no tax category assignment, no rate detection.
9. **Hardcoded categories** -- Only 5 fixed categories with no way to customize.
10. **Security issues** -- API key passed in form data, wildcard CORS, debug mode enabled.

## What We Can Learn

1. **Test/mock mode for development** -- We should add a mock mode to our pipeline that returns cached/fixture results without calling Cloud Vision or DeepSeek. This would speed up development and eliminate API costs during UI work. The `api_key == "test"` pattern is simple and effective.
2. **GPT for product name improvement** -- Beyond just extracting what the OCR reads, using the LLM to *improve* garbled product names is interesting. We could add a "clean_item_name" step that asks DeepSeek to clarify truncated or OCR-damaged item names while preserving the original language.
3. **Web app architecture reference** -- When we build a web UI, this project's Flask + vanilla JS structure (routes -> controllers -> modules) is a clean starting point. The REST API design (POST to process, GET to list, GET by ID) is sensible.
4. **Integer price storage** -- If we add database persistence, storing yen amounts as integers (which they already are for JPY) is the right approach.

## Recommendation

**Do not use this project.** It is too simplistic for production use and has no Japanese support. However, two patterns are worth adopting:

1. **Mock/test mode** -- Add a development mode to our pipeline that bypasses API calls using cached fixtures
2. **LLM name improvement** -- Consider adding a "clarify product name" instruction to our DeepSeek prompt for OCR-damaged item names
