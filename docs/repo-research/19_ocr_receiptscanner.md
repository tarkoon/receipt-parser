# 19. GiulioLecci11/OCR_ReceiptScanner

**Repository:** https://github.com/GiulioLecci11/OCR_ReceiptScanner
**Stars:** ~3 | **Language:** PowerShell (85%), Python (8.6%), Batch (6.4%) | **Last Updated:** Dec 2024
**License:** MIT

## Overview

A minimal receipt scanning script that combines OpenCV image preprocessing, Tesseract OCR, and OpenAI GPT-4o to extract structured JSON from receipt images. The entire pipeline lives in a single Python file (`main.py`). This is essentially a proof-of-concept / tutorial-level project -- the simplest possible implementation of the "preprocess -> OCR -> LLM" pattern. The high PowerShell/Batch percentage in the language breakdown is because the repo accidentally includes a Python virtual environment (`pyvenv.cfg`, `Scripts/` directory).

## Architecture & How It Works

```
receipt.jpg
    |
    v
OpenCV Preprocessing (in main.py)
  - cv2.imread()
  - cv2.cvtColor (BGR -> grayscale)
  - cv2.threshold (Otsu's binarization)
    |
    v
Tesseract OCR
  - pytesseract.image_to_string(image)
  - No language specification
    |
    v
OpenAI GPT-4o
  - System prompt: "You are a receipt parser AI"
  - User prompt: OCR text appended
  - Asks for JSON with: total, business, items (title/quantity/price), transaction_timestamp
  - Response parsed by finding first '{' and last '}'
    |
    v
receipt.json (written to disk)
```

**That's it.** Three functions in one file:
1. `preprocess_image(image_path)` -- grayscale + Otsu threshold
2. `extract_text(image)` -- pytesseract.image_to_string
3. `parse_receipt(text)` -- sends to GPT-4o, extracts JSON from response

## Key Features

1. **Minimal implementation** -- The entire pipeline is roughly 50 lines of Python. Easy to understand, easy to modify.
2. **GPT-4o (not mini)** -- Uses the full GPT-4o model, which should produce higher quality parsing than gpt-4o-mini.
3. **Timestamp extraction** -- Asks GPT for `transaction_timestamp` in ISO 8601 format, which is a nice touch that some other projects miss.
4. **JSON extraction via string search** -- Uses `response.find('{')` and `response.rfind('}')` to extract JSON from GPT's response. This is fragile but handles cases where GPT adds text before/after the JSON.

## Japanese Support

**None.** Zero Japanese support:
- Tesseract called without language specification
- GPT prompt is English-only with no Japanese context
- No Japanese text processing, normalization, or handling
- Sample output uses English receipt data ("Supermarket X", "Milk", "Bread")
- No documentation mentioning Japanese or multilingual support

## Strengths vs Our Project

1. **Simplicity** -- This project demonstrates that the core "OCR -> LLM -> JSON" pattern can be implemented in under 50 lines. When explaining our pipeline to others, this is a good reference for the baseline approach we've built upon.
2. **GPT-4o quality** -- Using the full GPT-4o model (rather than mini) likely produces better parsing results for complex receipts, though at higher cost.
3. **ISO 8601 timestamps** -- Requesting timestamps in standardized format is good practice that we should ensure we're doing consistently.

That's about it. This project's simplicity is both its only strength and its primary limitation.

## Weaknesses vs Our Project

1. **No error handling** -- No try/except anywhere. If Tesseract fails, GPT returns bad JSON, or the file doesn't exist, the script crashes.
2. **No validation** -- GPT output parsed with string search, no schema validation, no type checking.
3. **No confidence scoring** -- No way to know if OCR or GPT output is reliable.
4. **No retry logic** -- Single attempt at each step. If GPT hallucinates, there's no fallback.
5. **Tesseract without language config** -- Same limitation as AIReceiptParser.
6. **Fragile JSON extraction** -- `response.find('{')` will break if GPT includes JSON in explanatory text or uses nested objects in non-JSON context.
7. **No tests** -- Zero tests, minimal commits.
8. **Hardcoded paths** -- API key stored as a constant in the source file (`MY_API_KEY`). Security antipattern.
9. **Committed virtual environment** -- The `pyvenv.cfg` and `Scripts/` directory are committed to the repo, indicating poor development practices.
10. **No image deskewing, denoising, or adaptive processing** -- Identical to AIReceiptParser's preprocessing.
11. **No post-processing** -- No text normalization, no field validation, no merchant rules, no tax handling.
12. **Prices as integers in cents** -- While this avoids floating-point issues, it means GPT must interpret and convert prices, adding another failure mode.

## What We Can Learn

1. **The baseline** -- This project is valuable as a "minimum viable receipt parser" reference. It shows exactly what the simplest OCR+LLM pipeline looks like, which helps us appreciate the value of every layer we've added (confidence scoring, validation, retry logic, merchant rules, etc.).
2. **String-based JSON extraction as a fallback** -- While we use proper JSON parsing, the `find('{') / rfind('}')` pattern could serve as a last-resort fallback if the LLM response contains markdown code fences or explanatory text around the JSON. We may already handle this, but it's worth verifying.

## Recommendation

**Do not use this project.** It is a minimal proof-of-concept with no practical advantages over our pipeline. It does serve as a useful illustration of the baseline "OCR -> LLM -> JSON" pattern, demonstrating by contrast the value of our multi-layer validation, confidence routing, and post-processing pipeline.

The only marginally useful technique is the string-search JSON extraction as a fallback parser, which could supplement our existing response parsing if we don't already handle markdown-wrapped JSON responses.
