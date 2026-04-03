# 16. YoshiRi/Receipt_OCR

**Repository:** https://github.com/YoshiRi/Receipt_OCR
**Stars:** ~5 | **Language:** Python (100%) | **Last Updated:** 2022 (2 commits total)
**License:** Not specified

## Overview

A minimal Japanese receipt OCR pipeline that uses Google Cloud Vision API to extract text from receipt images, then parses the raw OCR output into structured CSV data using rule-based regex extractors. The project was built as a personal tool for expense tracking with the Japanese household finance app "Zaim" (hence the `ZaimCSVGenerator` class). It is a small, focused project with only 2 commits and no tests.

## Architecture & How It Works

```
Receipt Images (folder)
    |
    v
Google Cloud Vision API (document_text_detection)
    |
    v
JSON responses saved to disk
    |
    v
Rule-based extractors (regex):
  - extract_shopname.py  (bounding box height heuristic)
  - extract_date.py      (regex date patterns)
  - extract_price.py     (keyword + price association)
    |
    v
Pandas DataFrame -> CSV output
```

**Two-step CLI pipeline:**
1. `python main_scan.py <image_folder>` -- sends images to Vision API, saves raw JSON responses
2. `python main.py <json_folder>` -- parses JSON into CSV with date, shopname, price, category columns

**Key modules in `lib/`:**
- `ocr_by_vision_api.py` -- Wraps `google.cloud.vision.ImageAnnotatorClient`, calls `document_text_detection()`, serializes responses as JSON with UTF-8 encoding
- `extract_shopname.py` -- Heuristic: takes top 3 lines of OCR output, picks the one with the largest average bounding box height (assumes shop name is in larger font)
- `extract_date.py` -- Regex `r'[12]\d{3}[/\-年 ](0?[1-9]|1[0-2])[/\-月 ]([12][0-9]|3[01]|0?[0-9])(日?)'` with fallback year-then-lookahead search
- `extract_price.py` -- Searches for keywords like "合計", "小計", "決済", "金額" then associates nearby price patterns `r'[¥\*][ \d,.]+'`; validates with 1.08x tax cross-check

## Key Features

- **Google Cloud Vision API** for high-quality Japanese OCR (same engine we use)
- **Bounding box heuristic for shop names** -- uses physical text size rather than position alone
- **Tax cross-validation** -- multiplies subtotal by 1.08 to verify against total, providing a sanity check
- **Zaim CSV format** -- outputs directly compatible with the popular Japanese household finance app
- **Keyword-based total detection** -- searches for 合計/小計/決済/金額 and associates with nearby yen amounts

## Japanese Support

**Native Japanese focus.** All comments are in Japanese. The project is specifically designed for Japanese receipts:
- Searches for Japanese keywords (合計, 小計, 現計, 決済, 金額, 釣り, 預かり, 外税)
- Handles yen symbol (¥) and asterisk (*) price prefixes common in Japanese receipts
- Date regex supports 年/月/日 separators
- **Does NOT handle Japanese era dates** (令和, 平成) -- only Western calendar years starting with `[12]\d{3}`

## Strengths vs Our Project

1. **Bounding box height heuristic for shop name** -- This is a clever physical-layout approach. Rather than relying on text content alone, it exploits the fact that shop names are typically printed in larger font. Our project doesn't use spatial/font-size heuristics from the Vision API response.
2. **Tax cross-validation** -- The 1.08x multiplication check between subtotal and total is a simple but effective validation we could consider. It's similar in spirit to our subset-sum matching but much simpler.
3. **Minimal dependencies** -- Just google-cloud-vision and pandas. Clean, focused code.

## Weaknesses vs Our Project

1. **No LLM** -- Pure regex extraction is brittle. No ability to handle unusual formats or infer missing fields.
2. **No era date support** -- Cannot parse 令和/平成 dates, which our pipeline handles.
3. **No line item extraction** -- Only extracts total, date, shopname. No individual items, tax categories, or payment methods.
4. **No validation or confidence scoring** -- No Pydantic schema, no confidence routing, no retry logic.
5. **No tests** -- Zero test coverage. Only 2 commits ever made.
6. **Hardcoded year list** -- Date extraction defaults to `['2021','2022']`, making it brittle for other years.
7. **No text normalization** -- Relies on raw Vision API output without cleaning OCR artifacts.
8. **Abandoned** -- Last updated 2022, only 2 commits. Not maintained.

## What We Can Learn

1. **Bounding box height as a feature for field extraction** -- We could use the Vision API's bounding polygon data to compute text height and use it as a signal for identifying store names or headers. This spatial feature is currently unused in our pipeline.
2. **Tax ratio cross-validation** -- A quick `subtotal * 1.08 ~= total` or `subtotal * 1.10 ~= total` check could be added as a lightweight validation step before or alongside our subset-sum matching.
3. **Keyword-based total detection as a pre-filter** -- Their approach of first finding "合計" lines and then extracting associated prices could serve as a fast pre-filter before sending to the LLM, potentially reducing LLM load for straightforward receipts.

## Recommendation

**Do not use this project directly.** It is abandoned, minimal, and our pipeline already surpasses it in every dimension. However, two specific techniques are worth considering:
- The bounding box height heuristic for store name detection could be added as a confidence signal in our field registry
- The tax ratio cross-validation could complement our existing subset-sum matching as a quick sanity check
