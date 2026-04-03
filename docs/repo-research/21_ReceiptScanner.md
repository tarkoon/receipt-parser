# 21. Martin36/ReceiptScanner

**URL:** https://github.com/Martin36/ReceiptScanner
**Stars:** 0 | **Forks:** 0 | **Language:** Python 100%
**Created:** 2020-11-05 | **Last Updated:** 2022-11-07
**License:** None | **Commits:** 51

---

## 1. Overview

ReceiptScanner is a Python algorithm for extracting structured data from Swedish supermarket receipts. It uses Google Cloud Vision API for OCR and applies rule-based parsing to convert raw OCR annotations into structured article data with prices, quantities, dates, and retailer identification. The author explicitly warns that "it may not work very well for your receipts, if they are very different from these" -- it is purpose-built for Coop, Hemkop, and ICA grocery chains in Sweden.

Despite 51 commits and a reasonably complete codebase, the project has zero stars and zero forks, suggesting it was a personal/academic project that never gained community traction.

---

## 2. Architecture & How It Works

### Pipeline Flow

```
Receipt PDF
  -> pdf2image (convert PDF pages to images)
  -> Google Cloud Vision API (text detection with bounding boxes)
  -> GcloudParser (rule-based spatial parsing)
  -> Prettyfier (diacritics cleaning)
  -> Categorizer (regex-based product categorization)
  -> validate_receipt_data (math verification)
  -> JSON / CSV output
```

### Core Components

| File | Purpose |
|------|---------|
| `src/main.py` | Pipeline orchestrator -- runs parse, categorize, validate, export stages |
| `src/receipt_parser.py` | `GcloudParser` class -- the core extraction engine using spatial/rule logic |
| `src/categorizer.py` | Regex-based mapping of ~100 Swedish product names to 20 categories |
| `src/categories.py` | Swedish grocery category constants (Mejeri, Kott, Frukt, Gronsaker, etc.) |
| `src/prettyfier.py` | Text cleaning -- strips diacritics while preserving Swedish chars (a, a, o) |
| `src/utils.py` | Price parsing, number extraction, bounding box utilities, deduplication |
| `src/validate_receipt_data.py` | Math checks: sum(items) vs receipt total, date consistency, article counts |
| `src/write_to_csv.py` | CSV export for downstream analysis |

### Key Parsing Logic (GcloudParser)

The parser is notably spatial-aware, using Google Cloud Vision's bounding polygon coordinates:

1. **Text Classification:** Each OCR annotation is classified as "number", "date", "int", "market", "total", "text", or "hanging" (comma-trailing words)
2. **Spatial Price Detection:** Numbers positioned beyond 70% of page width are classified as prices (right-aligned on receipts)
3. **Article Recognition:** Items starting with capital letters or asterisks, with retailer-specific patterns (all-caps for Coop/Hemkop, mixed-case for ICA)
4. **Quantity Parsing:** Regex patterns for Swedish weight/quantity formats: "0,538 kg x 21,95 SEK/kg", "3 st x 15,00"
5. **Discount Handling:** Negative values and specific patterns like "2F20" (group prices) trigger special processing
6. **Bottle Deposit (Pant):** Special offset handling for Swedish bottle deposit lines

### Validation

The `ReceiptDataValidator` performs:
- Article math: `amount * price == scanned_sum` per item
- Total count: parsed item count vs receipt-stated count
- Sum verification: sum of item totals vs receipt grand total
- Date consistency: flags multiple conflicting dates

---

## 3. Key Features

- **Spatial-aware parsing** using Google Cloud Vision bounding polygons, not just text order
- **Retailer-specific rules** for Coop, Hemkop, ICA Swedish grocery chains
- **20 Swedish grocery categories** with ~100 product regex mappings
- **Math validation** that cross-checks individual items against receipt totals
- **Swedish character preservation** during diacritics removal (keeps a, a, o)
- **CI integration** via Travis CI (.travis.yml)
- **Comprehensive test suite** covering parser, categorizer, utils, validator, and CSV writer

---

## 4. Japanese Support

**None.** This project is entirely Swedish-focused. The category system, regex patterns, retailer rules, and text normalization are all Swedish-specific. The Unidecode dependency removes diacritics but would destroy kanji entirely. There is no CJK awareness whatsoever.

---

## 5. Strengths vs Our Project

- **Spatial bounding-box parsing:** Their use of Google Cloud Vision's polygon coordinates to determine price positions (>70% page width = price) is clever. We could use similar spatial heuristics as a pre-processing step or confidence signal before sending to the LLM.
- **Math validation pipeline:** Their `validate_receipt_data.py` does item-level math checks (qty * unit_price == line_total, sum(lines) == grand_total). We do subset-sum matching for tax categories, but not this kind of per-item arithmetic verification.
- **Retailer-specific parsing rules:** They differentiate between Coop (all-caps items), Hemkop, and ICA (mixed-case). This is similar to our merchant rules system but more deeply integrated into the parsing logic itself.
- **Bottle deposit handling:** Swedish "pant" handling shows awareness of culture-specific receipt line types, analogous to our handling of Japanese tax categories.
- **Simpler, more auditable pipeline:** No LLM black box -- every parsing decision is traceable to a specific rule. Good for debugging.

---

## 6. Weaknesses vs Our Project

- **Zero flexibility:** Pure rule-based means it breaks completely on unfamiliar receipt formats. Our LLM-based approach generalizes to unseen layouts.
- **No LLM at all:** Relies entirely on regex and spatial heuristics. Cannot handle ambiguous text, OCR errors, or novel formatting.
- **No confidence scoring:** No concept of OCR confidence or extraction confidence. Either the rule matches or it does not.
- **Single language/locale:** Only Swedish. Our pipeline handles Japanese kanji, era dates, yen formatting, and multiple document types.
- **Outdated dependencies:** Pinned to 2020-era versions (google-cloud-vision 2.0.0, numpy 1.19.3). Last updated 2022.
- **No OCR retry or multi-pass:** Single shot OCR, no confidence-gated retry like our system.
- **PDF-only input:** Assumes PDF receipts, not direct image input.
- **Inactive project:** Zero community engagement, no recent development.

---

## 7. What We Can Learn

1. **Spatial heuristics as a pre-LLM filter:** Their approach of using bounding box x-coordinates to classify right-aligned text as prices is a good heuristic. We could add a spatial pre-processing step that tags OCR text with position hints (e.g., "right-aligned", "header-region", "footer-region") before passing to the LLM, potentially improving extraction accuracy.

2. **Per-item math verification:** We do subset-sum matching for tax categories, but we could add individual line-item arithmetic checks: does `quantity * unit_price == line_total`? This would catch LLM hallucinations on numeric fields.

3. **Retailer-adaptive parsing rules:** Their per-retailer customization (Coop vs ICA patterns) is similar to but more granular than our merchant rules. Consider whether our merchant rules could include format hints (e.g., "this merchant uses all-caps item names").

4. **Culture-specific line type awareness:** Their "pant" (bottle deposit) handling is a good pattern. We should ensure our pipeline has similar awareness of Japanese-specific line types like point cards, tax breakdowns, and loyalty discounts.

---

## 8. Recommendation

**Do not adopt this tool.** It is a dead project with zero community, Swedish-only focus, and purely rule-based architecture that would not help with Japanese receipts. However, two ideas are worth borrowing:

- **Spatial position tagging:** Add bounding box position hints to OCR text before LLM extraction
- **Per-item arithmetic validation:** Add a post-processing check that verifies `qty * unit_price == line_total` for each extracted line item
