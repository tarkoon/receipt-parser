# 17. yrarchi/household_accounts

**Repository:** https://github.com/yrarchi/household_accounts
**Stars:** ~22 | **Language:** Jupyter Notebook (99.4%), Python (0.6%) | **Last Updated:** Active (197 commits)
**License:** Not specified | **Python:** 3.11+

## Overview

A desktop GUI application for digitizing Japanese receipts into structured household expense data. Unlike most receipt OCR projects that focus only on text extraction, this project provides a complete human-in-the-loop workflow: it detects receipt boundaries in photos, runs OCR, presents results in an editable GUI for manual correction, and exports verified data to CSV. The project uses Tesseract (via PyOCR) for OCR and OpenCV for image processing -- no LLM is involved.

## Architecture & How It Works

```
Input: Photo containing one or more receipts
    |
    v
cut_out_receipts.py
  - OpenCV adaptive thresholding + contour detection
  - Perspective correction (warpPerspective)
  - Extracts individual receipt images
    |
    v
edit_image.py
  - Resizes images to consistent dimensions (aspect-ratio preserving)
    |
    v
ocr.py (Tesseract via PyOCR, lang='jpn')
  - Date extraction (regex, no era date support)
  - Total/subtotal detection (合計, 小計, 消費税, etc.)
  - Item + price separation
  - OCR error correction (O->0, b->6 character mapping)
  - Levenshtein distance fuzzy matching against historical items
  - Category assignment from CSV lookup
    |
    v
GUI (tkinter)
  - gui_each_receipt.py: Per-receipt verification with editable fields
  - gui_show_receipt_contours.py: Visual contour confirmation
  - gui_last_page.py: Completion summary
    |
    v
edit_csv.py
  - Deduplication via set difference
  - Correction tracking (item fixes, category fixes)
  - Daily CSV export (YYYYMMDD.csv)
```

**Entry point:** `python -m household_accounts` runs the full pipeline: cut_out -> OCR -> GUI -> export.

**Key dependencies:** numpy, pyocr (Tesseract wrapper), opencv-python. Poetry-managed.

## Key Features

1. **Multi-receipt detection from single photo** -- OpenCV contour detection with adaptive thresholding, polygon approximation (4-vertex filtering with 80-100 degree angle validation), and perspective correction. Handles multiple receipts in one scan.
2. **Human-in-the-loop GUI** -- tkinter interface shows the receipt image alongside extracted data. Users can edit dates, shop names, item names, prices, categories, and tax types. Real-time validation (red highlighting for invalid dates, integer validation for prices).
3. **Levenshtein distance fuzzy matching** -- Matches OCR'd item names against historical purchase data to correct OCR errors and auto-assign categories. This is a learning system that improves with use.
4. **OCR error correction mappings** -- Character substitution rules (O->0, b->6) for common Tesseract misreads on Japanese receipts.
5. **Japanese tax handling** -- Supports both standard (10%) and reduced (8%) consumption tax rates. Distinguishes 内税 (tax-included) vs 外税 (tax-excluded) items. Calculates tax-inclusive prices with proper rounding.
6. **Category hierarchy** -- Three-level categorization: shop type (supermarket, convenience store, etc.) -> major category (food, household, transport) -> medium category (rice, vegetables, meat, etc.).
7. **Correction tracking** -- Logs OCR corrections and category assignments to separate CSVs, creating a feedback loop for future accuracy improvements.
8. **Deduplication** -- Set-based duplicate detection prevents re-importing already processed receipts.

## Japanese Support

**Excellent native Japanese support.** This project is built entirely for Japanese household expense tracking:
- Tesseract with `lang='jpn'` for Japanese text recognition
- Japanese keyword detection (合計, 小計, 消費税, 対象計, 釣り, 預かり, 外税, 内税)
- Reduced tax rate markers (*, ※, etc.) for identifying 軽減税率 items
- Japanese category names (食費, 日用雑貨, 交通費, 教育費, 被服費, 交際費)
- Shop types in Japanese (スーパー, コンビニ, ドラッグストア, etc.)
- Date extraction with 年/月/日 separators
- All UI labels and messages in Japanese
- **Does NOT handle Japanese era dates** (令和, 平成)

## Strengths vs Our Project

1. **Human-in-the-loop verification GUI** -- This is the biggest differentiator. While our project is fully automated, this project acknowledges that OCR/extraction is imperfect and provides a polished correction interface. For household use, this ensures 100% accuracy on every receipt.
2. **Multi-receipt detection from photos** -- OpenCV-based contour detection can extract multiple receipts from a single photograph. Our project assumes one receipt per image.
3. **Levenshtein fuzzy matching** -- Using edit distance to match OCR'd item names against historical purchases is clever. As the user processes more receipts, the system gets better at correcting OCR errors and auto-categorizing items. Our merchant rules are static.
4. **Correction feedback loop** -- Logging user corrections creates training data that could be used to improve future extraction. This self-improving pattern is absent from our pipeline.
5. **Perspective correction** -- The warpPerspective transform normalizes skewed/tilted receipt photos before OCR, which we don't do (we rely on Cloud Vision handling this internally).
6. **Mature project** -- 197 commits, proper packaging with pyproject.toml/Poetry, CI via GitHub Actions. Much more mature than most receipt OCR projects.

## Weaknesses vs Our Project

1. **Tesseract OCR quality** -- Tesseract with `lang='jpn'` produces significantly lower quality output than Google Cloud Vision for Japanese text. This is why the project needs the human-in-the-loop correction step.
2. **No LLM extraction** -- Pure regex/heuristic extraction. Cannot handle unusual receipt formats or infer missing fields.
3. **Desktop-only** -- tkinter GUI means it only runs as a local desktop app. No API, no web interface, no batch processing mode.
4. **No confidence scoring** -- No way to programmatically assess extraction quality. Relies entirely on human review.
5. **No line item detail** -- While it extracts individual items, it doesn't capture the rich metadata our pipeline produces (payment method, invoice numbers, location details, etc.).
6. **No era date support** -- Same limitation as Receipt_OCR.
7. **No automated testing** -- Despite having CI configured, the test coverage appears minimal.

## What We Can Learn

1. **Human-in-the-loop as a feature, not a weakness** -- For a future UI/web app built on our pipeline, we should consider a "review and correct" mode where users can verify and fix LLM extractions. This is especially valuable for edge cases where confidence is low.
2. **Levenshtein fuzzy matching for merchant/item correction** -- We could add edit-distance matching to our merchant rules system. When the LLM returns a merchant name that's close but not exact to a known merchant, fuzzy matching could auto-correct it.
3. **Correction tracking for continuous improvement** -- Logging user corrections (original vs. corrected values) would create training data for improving our prompt engineering or fine-tuning.
4. **Multi-receipt detection** -- If users photograph multiple receipts at once, OpenCV contour detection could split them before sending to Cloud Vision. This could be a useful pre-processing step.
5. **Category hierarchy design** -- Their three-level category system (shop type -> major -> medium) is well-thought-out for Japanese household expenses and could inform our category taxonomy.

## Recommendation

**Do not use this project as a dependency**, but it is the most architecturally interesting project in this research batch. Three specific ideas worth adopting:

1. **Fuzzy matching for merchant names** -- Add Levenshtein distance matching to our merchant rules system as a fallback when exact matches fail
2. **Correction tracking** -- When we build a UI, log corrections as a feedback signal for pipeline improvements
3. **Review/edit mode in future UI** -- Design our UI with a verification step, especially for low-confidence extractions, following this project's human-in-the-loop pattern
