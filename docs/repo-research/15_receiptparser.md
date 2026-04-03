# 15. knipknap/receiptparser

## Overview

| Field | Value |
|---|---|
| **Repository** | [knipknap/receiptparser](https://github.com/knipknap/receiptparser) |
| **Stars** | 23 |
| **Forks** | 7 |
| **Language** | Python (85.3%), Makefile (8.2%), Shell (6.5%) |
| **License** | MIT |
| **Created** | 2020-08-17 |
| **Last Push** | 2021-06-20 (inactive) |
| **Approach** | Tesseract OCR + YAML config + regex + fuzzy matching + dual-pass scanning |

A lightweight German receipt parser forked from (and completely rewriting) the receipt-parser-legacy project (#14 above). Focuses on clean API design, YAML-driven locale configs, and a dual-pass OCR strategy. Published on PyPI as `receiptparser`. Smaller community (23 stars) but cleaner architecture than its predecessor.

## Architecture & How It Works

### Pipeline

```
Receipt Image
    |
    v
[Wand (ImageMagick) image loading]
    |
    v
[Pass 1: Tesseract OCR - unsharpened]
    |
    v
[Receipt parsing - regex + fuzzy match]
    |
    |-- If all fields found: DONE
    |
    v (if incomplete)
[Pass 2: Tesseract OCR - sharpened]
    |  - auto_level()
    |  - sharpen(radius=0, sigma=4.0)
    |  - contrast()
    v
[Receipt parsing - second attempt]
    |
    v
[Merge: fill missing fields from Pass 2]
    |
    v
Receipt(company, date, postal, sum)
```

### Core Source Files

**`receiptparser/parser.py`** -- Main processing:
- `ocr_image(input_file, language, sharpen=False)` -- Wand image load -> optional sharpening -> PIL conversion -> pytesseract
- `_process_receipt(config, filename, out_dir, sharpen)` -- OCR + Receipt construction
- `process_receipt(config, filename, out_dir, verbosity)` -- **Dual-pass logic**: First pass unsharpened. If `receipt.is_complete()` returns False, runs second pass with sharpening. Calls `receipt.merge(receipt2)` to fill gaps.
- Also handles pre-existing `.txt` files (skip OCR, parse directly)

**`receiptparser/receipt.py`** -- Receipt class:
- `__init__(config, filename, raw)` -- Splits raw text into lowercased lines, parses
- `is_complete()` -- Returns True only if ALL fields (company, date, postal, sum) are non-None
- `merge(receipt)` -- Fills None fields from another Receipt instance
- `to_dict()` -- Exports filename, company, date, postal, sum
- `for_format_string()` -- Provides defaults for missing fields (e.g., date defaults to epoch)
- `fuzzy_find(keyword, accuracy=0.6)` -- First tries regex word boundary match, then falls back to `difflib.get_close_matches()`
- `parse_company()` -- Cascading accuracy (1.0 -> 0.7) against configured company spellings
- `parse_postal()` -- Regex match for 5-digit postal code followed by text
- `parse_date()` -- Regex match for German date formats, validated with `dateutil.parser.parse()`
- `parse_sum()` -- Fuzzy find sum keyword, regex extract amount

**`receiptparser/config.py`** -- YAML config loader:
- Uses `munch` library to convert YAML dict to attribute-accessible object (`config.formats.date` instead of `config['formats']['date']`)
- Loads from `receiptparser/data/configs/` directory

**`receiptparser/data/configs/germany.yml`** -- German locale config:
```yaml
language: deu

companys:
  Aldi: [aldi]
  Lidl: [lidl]
  REWE: [rewe]
  Penny: [penny, p e n n y, m a r k t gmbh]
  # ... 20+ retailers

sum_keys: [summe, gesamtbetrag, gesamt, total, sum, zwischensumme, bar, te betalen]

formats:
  sum: '\d+(\.\s?|,\s?|[^a-zA-Z\d])\d{2}'
  date: '\b([0123]?\d\s?\.\s?[01]?\d\s?\.\s?(?:20)?\d\d)\b'
  postal_code: '\b(\d{5})\s+[a-z]'
```

### CLI Usage

```bash
# Custom format string output
receiptparser -v0 --format "{date:%Y-%m-%d} - {company}" /path/to/images/

# Python API
from receiptparser.config import read_config
from receiptparser.parser import process_receipt
config = read_config('germany.yml')
receipt = process_receipt(config, "image.jpg", verbosity=1)
```

## Key Features

1. **Dual-pass OCR with merge**: First pass without sharpening; if incomplete, second pass with sharpening (auto-level + sharpen + contrast). ~6% accuracy improvement at cost of 2x processing time. Missing fields from pass 1 are filled by pass 2.

2. **`is_complete()` / `merge()` pattern**: Clean abstraction for determining when extraction is "good enough" and combining results from multiple attempts.

3. **YAML locale configs**: Adding a new country/language means creating a new YAML file with company names, sum keywords, date formats, postal code patterns -- no code changes needed.

4. **Attribute-accessible config via munch**: `config.formats.date` instead of `config['formats']['date']` -- cleaner API.

5. **Pre-existing OCR text support**: If a `.txt` file exists, skip OCR and parse directly -- useful for cached OCR results.

6. **Format string output**: CLI supports Python format strings for custom output: `{date:%Y-%m-%d} - {company}`.

7. **Published accuracy numbers**: 94% company, 87% postal, 87% date, 63% sum across 182 receipts. Honest about limitations (sum extraction is hardest).

### Performance on 182 German receipts:
| Field | Accuracy |
|---|---|
| Company | 94% (171/182) |
| Postal Code | 87% (158/182) |
| Date | 87% (159/182) |
| Sum/Total | 63% (114/182) |

## Japanese Support

**None.** German-only. Tesseract language set to `deu`. Company names, sum keywords, date formats, postal code regex -- all German. However, the YAML config architecture makes locale extension straightforward in theory. A `japan.yml` could define:

```yaml
language: jpn
companys:
  セブンイレブン: [セブンイレブン, セブン-イレブン]
  ローソン: [ローソン, LAWSON]
sum_keys: [合計, お買上合計, 小計, 総合計]
formats:
  sum: '[\d,]+円'
  date: '(\d{4}年\d{1,2}月\d{1,2}日)'
  postal_code: '〒?(\d{3}-\d{4})'
```

But Tesseract's Japanese OCR quality is poor compared to Cloud Vision or PaddleOCR, so the OCR layer would be the bottleneck.

## Strengths vs Our Project

1. **Dual-pass OCR retry with merge**: The "try once, check completeness, retry with different preprocessing, merge results" pattern is the clearest implementation of OCR retry I've seen. Our confidence-gated OCR retry is more sophisticated (uses confidence scores), but their `is_complete()` + `merge()` pattern is simpler and worth studying.

2. **Clean locale abstraction**: The YAML config approach makes adding a new country truly zero-code. Our project hardcodes Japanese-specific logic. If we ever wanted to support multiple Asian receipt formats, this pattern would scale better.

3. **Published accuracy metrics**: Honest reporting of per-field accuracy on a real dataset (182 receipts). The 63% sum accuracy is a useful calibration point -- it shows how hard sum extraction is with pure regex/fuzzy matching (vs our LLM-based approach).

4. **Format string CLI output**: The `--format "{date:%Y-%m-%d} - {company}"` pattern is user-friendly for scripting. Our CLI could benefit from similar flexibility.

5. **Simple, auditable codebase**: Under 300 lines of actual parsing logic. Easy to understand, debug, and extend. Our pipeline is more powerful but significantly more complex.

## Weaknesses vs Our Project

1. **63% sum accuracy**: The hardest field (total amount) succeeds only 63% of the time with pure regex. Our LLM-based extraction achieves much higher accuracy.

2. **4-field extraction only**: Company, postal code, date, sum. No line items, no tax details, no payment methods.

3. **No LLM intelligence**: Cannot handle novel formats, contextual reasoning, or ambiguous text.

4. **Tesseract is the weakest OCR option**: Especially for receipt-quality images. Google Cloud Vision and PaddleOCR both outperform it significantly.

5. **No validation or post-processing**: If regex matches something wrong, there's no second opinion or sanity check.

6. **Inactive since 2021**: No updates in 5 years. Dependencies may be outdated.

7. **No test fixtures with ground truth**: Published accuracy numbers but no reproducible benchmark.

## What We Can Learn

1. **`is_complete()` + `merge()` pattern for multi-pass extraction**: This is a clean abstraction we could adopt:
   ```python
   class ExtractionResult:
       def is_complete(self) -> bool:
           """True if all required fields are non-None."""
           return all(getattr(self, f) is not None for f in REQUIRED_FIELDS)
       
       def merge(self, other: 'ExtractionResult'):
           """Fill None fields from another result."""
           for field in REQUIRED_FIELDS:
               if getattr(self, field) is None:
                   setattr(self, field, getattr(other, field))
   ```
   This could simplify our confidence routing logic -- run LLM extraction, check completeness, if incomplete run a second pass with different prompt/parameters, merge.

2. **Dual-pass with different preprocessing**: The idea of trying different image preprocessing (unsharpened vs sharpened) and merging results is applicable to our OCR retry strategy. We could try different Cloud Vision parameters or different image preprocessing before the second OCR attempt.

3. **YAML locale config template**: If we externalize our field extraction rules to YAML, the `germany.yml` structure provides a good template. A `japan.yml` with merchant variants, sum keywords, date formats, and field patterns would make our pipeline more configurable.

4. **Attribute-accessible config with munch**: The `munch.munchify()` pattern (YAML dict -> attribute-accessible object) is cleaner than nested dict access. We could adopt this for our own config handling.

5. **Honest accuracy reporting per field**: Their breakdown (94% company, 63% sum) shows that some fields are inherently harder. We should report per-field accuracy in our benchmarks, not just overall accuracy.

## Recommendation

**Do not adopt as a dependency** (German-only, Tesseract-based, 63% sum accuracy, inactive). However, **adopt the dual-pass merge pattern**:

1. **`is_complete()` + `merge()` abstraction**: Apply this pattern to our LLM extraction results. When the first LLM pass returns incomplete data, run a second pass with different parameters and merge the results.
2. **YAML locale config**: Consider externalizing Japanese-specific field patterns, merchant variants, and keywords to a YAML config file following this project's structure.
3. **Per-field accuracy reporting**: Add per-field accuracy breakdowns to our benchmark output, not just overall pass/fail rates.
