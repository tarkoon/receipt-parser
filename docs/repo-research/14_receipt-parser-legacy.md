# 14. ReceiptManager/receipt-parser-legacy

## Overview

| Field | Value |
|---|---|
| **Repository** | [ReceiptManager/receipt-parser-legacy](https://github.com/ReceiptManager/receipt-parser-legacy) |
| **Stars** | 852 |
| **Forks** | 199 |
| **Language** | Python (97%), Makefile (2.1%), Dockerfile (0.9%) |
| **License** | Apache-2.0 |
| **Created** | 2015-10-02 |
| **Last Push** | 2024-08-28 (maintained, low activity) |
| **Approach** | Tesseract OCR + regex/fuzzy matching + config-driven parsing |

The most-starred receipt parser in this batch (852 stars). A "fuzzy receipt parser" that uses Tesseract OCR and `difflib.get_close_matches()` for approximate string matching against known market names. Originally a hackathon project, it was featured on the trivago tech blog and HackerNews. Published on PyPI as `receipt-parser-core`. Part of the ReceiptManager ecosystem (iOS + Android apps).

## Architecture & How It Works

### Pipeline

```
Receipt Image
    |
    v
[Image Pre-processing]
    |  - Rotation detection (landscape->portrait)
    |  - Deskew via histogram scoring (scipy.ndimage)
    |  - Rescale 1.2x with cubic interpolation
    |  - Sharpening + contrast + auto-level (ImageMagick/Wand)
    v
[Tesseract OCR]
    |  - Language: German (configurable)
    |  - PSM 6 (assume uniform block of text)
    |  - 60-second timeout
    v
Raw text (saved to data/txt/)
    |
    v
[Receipt class -- regex + fuzzy parsing]
    |  - normalize(): strip empty lines, lowercase
    |  - parse_market(): fuzzy match against config.markets dict
    |  - parse_date(): regex match with dateutil validation
    |  - parse_sum(): fuzzy find sum keywords, regex extract amount
    |  - parse_items(): regex match item patterns, skip ignored words
    v
Structured output: {market, date, sum, items[]}
```

### Core Source Files

**`receipt_parser_core/enhancer.py`** -- Image pre-processing pipeline:
- `rotate_image()` -- Detects landscape orientation, rotates 90 degrees
- `deskew_image()` -- Scores rotation angles by horizontal histogram variance, applies best correction
- `rescale_image()` -- 1.2x upscale with cubic interpolation
- `run_tesseract()` -- Wand image conversion + pytesseract with language parameter
- Uses OpenCV, scipy, Wand (ImageMagick), PIL

**`receipt_parser_core/receipt.py`** -- Core parsing logic (Receipt class):
- `normalize()` -- Lowercases all lines, strips empties
- `fuzzy_find(keyword, accuracy=0.6)` -- Iterates lines, splits into words, uses `difflib.get_close_matches()` to find approximate matches
- `parse_market()` -- Cascading accuracy search (1.0 -> 0.7) against configured market spellings. Each market has multiple known OCR-error variants
- `parse_date()` -- Regex match against configurable date format pattern, validated with `dateutil.parser.parse()`
- `parse_sum()` -- Fuzzy-finds sum keyword lines (e.g., "summe", "gesamtbetrag", "total"), then regex-extracts the amount
- `parse_items()` -- Regex item pattern matching with ignore/stop word filtering. Handles negative amounts (refunds). Market-specific item formats (Metro has a different pattern)
- `to_json()` -- Serializes to JSON

**`receipt_parser_core/parse.py`** -- CLI orchestration:
- `get_files_in_folder()` -- Lists receipt text files
- `ocr_receipts()` -- Batch processes files, builds terminal table, tracks statistics
- `results_to_json()` -- Batch JSON export
- Statistics tracking: counts of successful market/date/sum extractions

**`config.yml`** -- Rule configuration:
- `language: deu` -- Tesseract language
- `markets:` -- Dict of market names -> list of known spellings/OCR variants
- `sum_keys:` -- Ordered list of sum indicator words (summe, gesamtbetrag, total, bar, etc.)
- `ignore_keys:` -- Words to skip in item parsing (mwst, kg x, stk, etc.)
- `sum_format:` -- Regex for amount pattern
- `item_format:` -- Regex for item line pattern (name + amount)
- `item_format_metro:` -- Market-specific item format
- `date_format:` -- Regex for date patterns

### Config-driven market recognition example:
```yaml
markets:
  Penny:
    - penny
    - p e n n y        # OCR often spaces out letters
    - m a r k t gmbh   # OCR variant of "Markt GmbH"
  Kaiser's:
    - kaiser
    - kaiserswerther straße 270  # Address-based fallback
```

## Key Features

1. **Fuzzy matching with cascading accuracy**: Starts at 100% match accuracy and degrades to 70%, returning the first hit. Handles OCR errors gracefully
2. **Market-specific parsing rules**: Different regex patterns per market (Metro has unique item format)
3. **OCR spelling variants in config**: Markets are identified by multiple known OCR-error spellings
4. **Image deskewing**: Histogram-based rotation correction (scores angles by variance, picks best)
5. **Docker support**: Full containerization with volume mounts for input images
6. **PyPI package**: Installable via `pip install receipt-parser-core`
7. **Batch processing**: Process entire directories of receipt images
8. **Statistics tracking**: Counts successful extraction rates across markets

## Japanese Support

**None.** Configured for German (`deu`) Tesseract language. Market names are all German/European retailers. Date format regex targets DD.MM.YY and DD/MM/YYYY patterns. Sum keywords are German (summe, gesamtbetrag). Amount format assumes European comma-as-decimal notation. No CJK character handling.

## Strengths vs Our Project

1. **OCR spelling variant database**: The market config with multiple known OCR-error spellings per store is excellent. For example, knowing that OCR often renders "PENNY" as "P E N N Y" (spaced letters) or that an address can identify a store when the name is garbled. We could build a similar OCR-error variant database for Japanese merchants.

2. **Fuzzy matching with cascading accuracy**: The `fuzzy_find()` approach of starting strict (1.0) and degrading to loose (0.7) is practical. We could apply this to our merchant matching -- try exact match first, then progressively looser fuzzy matching.

3. **Image pre-processing pipeline**: The deskew algorithm (histogram scoring of rotation angles) is more sophisticated than what we do. Our Google Cloud Vision handles this server-side, but if we ever use PaddleOCR locally, this pipeline would be valuable.

4. **Market-specific parsing rules**: The idea that different merchants require different item format regexes is something we partially do with our merchant rules, but their config-driven approach with per-market `item_format` is cleaner.

5. **Config-driven architecture**: Everything is in YAML -- markets, date formats, sum keywords, item patterns. Adding a new market or locale means editing config, not code. Our merchant rules are in code.

6. **Mature project (10 years)**: Battle-tested across many receipt formats with 199 forks contributing edge cases.

## Weaknesses vs Our Project

1. **No LLM -- pure regex**: All extraction is regex + fuzzy matching. Cannot handle novel layouts, ambiguous text, or semantic reasoning. Our LLM pipeline handles cases that no regex could.

2. **4-field extraction only**: Market, date, sum, items. No tax details, payment method, receipt number, merchant address, etc.

3. **German-only in practice**: Config would need complete rewrite for Japanese (different date formats, amount formats, market names, sum keywords)

4. **No confidence scoring**: Returns extracted values or None. No indication of extraction quality.

5. **No validation beyond dateutil**: No cross-field validation, no total-vs-items check, no Pydantic schema.

6. **Tesseract OCR is weak on receipts**: Thermal receipt print quality + small fonts + Japanese characters = Tesseract struggles. Google Cloud Vision is significantly more robust.

7. **No benchmark suite**: No ground truth fixtures, no accuracy tracking, no regression tests beyond basic unit tests.

8. **Legacy/archived feel**: While technically maintained (last push 2024-08), the architecture is 2015-era Python with no modern patterns.

## What We Can Learn

1. **OCR-error variant database for merchants**: Build a Japanese merchant variant database in our config:
   ```yaml
   merchants:
     セブンイレブン:
       - セブンイレブン
       - セブン-イレブン
       - セブン一イレブン    # OCR confuses ー with 一
       - 7-ELEVEN
       - ７-ＥＬＥＶＥＮ    # Full-width variant
   ```
   This would make our merchant matching more robust against OCR errors without relying on the LLM.

2. **Cascading accuracy for fuzzy matching**: Implement our merchant/keyword matching with decreasing strictness:
   ```python
   for threshold in [1.0, 0.9, 0.8, 0.7]:
       match = fuzzy_match(text, known_merchants, threshold)
       if match: return match
   ```

3. **Config-driven market-specific rules**: Move our merchant rules from code to YAML config. Each merchant could have its own item format, date format, and known field positions.

4. **Image deskewing for local OCR**: If we ever move to PaddleOCR for cost reduction, adopt their histogram-based deskew algorithm as a pre-processing step.

5. **Sum keyword lists**: Their ordered list of sum keywords (prioritized by likelihood) is a simple but effective pattern. We could maintain similar keyword lists for Japanese receipt fields:
   ```yaml
   sum_keys:
     - 合計
     - お買上合計
     - 総合計
     - 小計
     - ご請求額
   ```

## Recommendation

**Do not adopt as a dependency** (German-only, regex-only, no LLM). However, **adopt the config-driven architecture pattern**:

1. **OCR variant database**: Create a YAML config of known OCR error variants for Japanese merchants and keywords
2. **Cascading fuzzy match**: Implement decreasing-strictness fuzzy matching for merchant identification
3. **Move merchant rules to config**: Externalize our merchant rules from Python code to YAML, making it easier to add new merchants without code changes
