# 10. CORD (clovaai/cord)

**GitHub:** https://github.com/clovaai/cord
**Paper:** https://openreview.net/pdf?id=SJl3z659UH
**Hugging Face:** https://huggingface.co/datasets/katanaml/cord
**License:** CC-BY-4.0

---

## 1. Overview

CORD (COnsolidated Receipt Dataset) is a benchmark dataset for post-OCR receipt parsing from Naver CLOVA AI. It provides 1,000 annotated Indonesian receipt images with hierarchical semantic labels, bounding boxes, and row grouping -- designed to evaluate models that extract structured information from OCR output.

- **Stars:** ~471
- **Language:** Dataset (Python tools/examples)
- **License:** CC-BY-4.0 (Creative Commons Attribution)
- **Last updated:** July 2022 (v2 release)
- **Approach:** Not a tool or model -- it's a **benchmark dataset** with labeled receipts and a standardized evaluation framework for receipt parsing research.

CORD is the most widely-cited receipt parsing benchmark in the research community, referenced by papers on DoNUT, LayoutLM, and many other document understanding models.

---

## 2. Architecture & How It Works

### This Is a Dataset, Not Software

CORD provides:
1. **Receipt images**: 1,000 Indonesian receipts (800 train / 100 dev / 100 test)
2. **Annotations**: JSON files with bounding boxes, text, semantic labels, and grouping
3. **Evaluation metrics**: Standardized parsing accuracy measurement

The "architecture" is the annotation schema, not a processing pipeline.

### Dataset Versions

| Version | Date | Source | Changes |
|---------|------|--------|---------|
| v0 | Dec 2019 | Google Drive | Original release |
| v1 | Jul 2022 | Hugging Face | HF integration |
| v2 | Jul 2022 | Hugging Face | Corrected labels + `sub_group_id` for improved hierarchy |

### Annotation Schema

Each receipt is annotated with a JSON structure:

```json
{
  "valid_line": [
    {
      "words": [
        {
          "quad": {"x1": 0, "y1": 0, "x2": 100, "y2": 0, "x3": 100, "y3": 20, "x4": 0, "y4": 20},
          "is_key": true,
          "row_id": 1,
          "text": "TOTAL"
        }
      ],
      "category": "total.total_price",
      "group_id": 5
    }
  ],
  "meta": {
    "version": "v2",
    "image_id": "receipt_00001",
    "split": "train",
    "image_size": {"width": 640, "height": 960}
  },
  "roi": {"x1": 10, "y1": 10, "x2": 630, "y2": 950},
  "repeating_symbol": {"text": "=", "quad": {...}}
}
```

### Label Taxonomy (42 subclasses across 5 superclasses)

**menu** (14 subclasses):
- `menu.nm` (item name), `menu.num` (quantity), `menu.unitprice`, `menu.price`, `menu.itemsubtotal`
- `menu.discountprice`, `menu.sub_nm`, `menu.sub_unitprice`, `menu.sub_price`, `menu.sub_cnt`
- `menu.vatyn`, `menu.etc`, `menu.cnt` (count), `menu.sub_etc`

**void_menu** (2 subclasses):
- `void_menu.nm`, `void_menu.price`

**sub_total** (6 subclasses):
- `sub_total.subtotal_price`, `sub_total.discount_price`, `sub_total.service_price`
- `sub_total.othersvc_price`, `sub_total.tax_price`, `sub_total.etc`

**total** (8 subclasses):
- `total.total_price`, `total.total_etc`, `total.cashprice`, `total.changeprice`
- `total.creditcardprice`, `total.emoneyprice`, `total.menutype_cnt`, `total.menuqty_cnt`

**void_total**: Removed from publication due to legal issues.

### Additional Annotations

- **Quadrilateral bounding boxes**: 4-point coordinates for each word (handles rotated/skewed text)
- **Row grouping** (`row_id`): Groups words that belong to the same logical row
- **Region of interest** (`roi`): Bounding box of the receipt area within the image
- **Repeating symbols**: Detection of separator lines (===, ---, ...)
- **`is_key` flag**: Distinguishes label text from value text
- **`sub_group_id`** (v2): Improved hierarchical grouping for nested line items

### Related Research

1. **CORD paper** (Park et al., 2019): Original dataset and post-OCR parsing challenge
2. **BIO tagging** (Hwang et al., 2019): Sequential labeling approach to receipt parsing
3. **DoNUT** (Kim et al., 2021): OCR-free document understanding transformer trained/evaluated on CORD

---

## 3. Key Features

- **Hierarchical label taxonomy**: 42 fine-grained subclasses organized into 5 superclasses -- the most detailed receipt annotation scheme in any public dataset
- **Row grouping**: `row_id` links words that belong together (e.g., item name + price on same line)
- **Quadrilateral bounding boxes**: Handles rotated, skewed, and perspective-distorted text
- **Key/value distinction**: `is_key` flag separates label text ("Total:") from value text ("$25.00")
- **Separator detection**: `repeating_symbol` identifies line separators (===, ---)
- **Standardized splits**: 800/100/100 train/dev/test for reproducible evaluation
- **Hugging Face integration**: Easy loading via `datasets` library
- **CC-BY-4.0 license**: Free for commercial use with attribution

---

## 4. Japanese Support

**None. Indonesian receipts only.**

The entire dataset consists of Indonesian receipts from shops and restaurants. Key limitations for Japanese receipt research:

- **Language**: Indonesian only (Latin script); no CJK characters
- **Currency**: Indonesian Rupiah; no Yen formatting
- **Tax system**: Indonesian tax rules; no Japanese consumption tax (8%/10% split)
- **Date formats**: Western date formats; no Japanese era dates
- **Receipt layout**: Indonesian retail conventions; different from Japanese receipt layouts
- **Store information**: Removed from published version due to Indonesian legal issues

However, the **annotation schema and evaluation methodology** are language-agnostic and could be adapted for Japanese receipts. The hierarchical label taxonomy is the most directly useful component.

---

## 5. Strengths vs Our Project

| Their Strength | Detail |
|----------------|--------|
| **Standardized benchmark** | 1,000 labeled receipts with standardized splits for reproducible evaluation; we have 36 fixtures |
| **Hierarchical label taxonomy** | 42 subclass labels with 5 superclasses is more granular than our schema in some areas |
| **Row grouping** | `row_id` and `group_id` explicitly link related fields; we infer relationships from LLM extraction |
| **Key/value distinction** | `is_key` flag separates labels from values; we rely on LLM to understand this implicitly |
| **Bounding box annotations** | Quadrilateral coordinates for every word; we don't preserve spatial information |
| **Separator detection** | `repeating_symbol` annotation identifies receipt section boundaries; we don't explicitly detect these |
| **Research community adoption** | Used as benchmark by DoNUT, LayoutLM, and many others; our fixtures are private |
| **`sub_group_id`** (v2) | Handles nested line items (sub-items under a main item); our schema handles this less explicitly |

---

## 6. Weaknesses vs Our Project

| Our Strength | Detail |
|--------------|--------|
| **Working pipeline** | CORD is a dataset; we have a complete extraction pipeline |
| **Japanese receipts** | We handle kanji, era dates, zenkaku numbers, Japanese tax; CORD is Indonesian only |
| **Tax verification** | Subset-sum matching validates totals; CORD just labels them |
| **Confidence scoring** | OCR + LLM confidence routing; CORD has no confidence model |
| **Real-world receipt diversity** | Our 36 fixtures include utility bills, payment slips, convenience stores, restaurants, etc. |
| **End-to-end processing** | Image -> structured JSON in one pipeline; CORD requires external OCR + parsing model |
| **Active maintenance** | We iterate on failures; CORD last updated July 2022 |
| **Merchant rules** | We have brand-specific parsing rules; CORD treats all receipts identically |
| **Payment method tracking** | Our schema tracks cash/card/e-money; CORD has `cashprice`/`creditcardprice`/`emoneyprice` but they're static labels |

---

## 7. What We Can Learn

1. **Hierarchical label taxonomy**: CORD's 5-superclass / 42-subclass structure is worth studying for our own schema design. Specific ideas:
   - **`is_key` flag**: Annotating whether a text span is a field label or a field value would help our LLM understand receipt structure better
   - **`row_id` grouping**: Explicitly linking items on the same receipt line could improve line item extraction accuracy
   - **Void items**: CORD has `void_menu` labels for cancelled items -- something our schema doesn't handle
   - **Sub-items**: `menu.sub_nm`, `menu.sub_price`, `menu.sub_cnt` for nested menu items (e.g., toppings, modifiers)

2. **Bounding box preservation**: CORD preserves quadrilateral bounding boxes for every word. If we stored Cloud Vision's bounding box data through our pipeline, we could:
   - Better detect line item boundaries
   - Identify receipt sections (header, items, totals) spatially
   - Handle multi-column receipt layouts more reliably
   - Visualize extraction results overlaid on the receipt image

3. **Separator detection**: The `repeating_symbol` annotation identifies lines of `===`, `---`, or `...` that separate receipt sections. We could add a simple regex-based separator detection step after OCR to identify section boundaries before LLM extraction.

4. **Standardized evaluation methodology**: CORD defines clear train/dev/test splits and evaluation metrics. We should formalize our 36 fixtures into similar splits and establish standardized accuracy metrics that we track over time.

5. **Key/value flag in training data**: The `is_key` annotation distinguishes "TOTAL:" (key/label) from "$25.00" (value). If we ever fine-tune a model on receipt data, this distinction in training data would significantly improve extraction.

6. **Receipt ROI detection**: CORD's `roi` annotation defines the receipt region within the image. Adding receipt boundary detection as a preprocessing step could help with photos that include background elements, multiple receipts, or partial receipt captures.

7. **Hugging Face dataset integration**: Publishing our fixtures (or a subset) as a Hugging Face dataset would enable:
   - Community contributions and benchmarking
   - Integration with training pipelines for fine-tuning
   - Reproducible evaluation by other researchers

---

## 8. Recommendation

**Highly valuable as a design reference and evaluation methodology guide, not as a direct tool.**

CORD is the gold standard for receipt parsing benchmarks. Even though the receipts are Indonesian (not Japanese), the annotation schema, evaluation methodology, and hierarchical label design are directly applicable to our project.

**Immediate actions:**
1. **Adopt CORD's label taxonomy ideas**: Add `is_key` distinction, `row_id` grouping, void item handling, and sub-item support to our schema
2. **Preserve bounding boxes**: Start carrying Cloud Vision's word-level bounding boxes through our pipeline for spatial reasoning
3. **Add separator detection**: Simple regex to identify `===`/`---`/`...` lines as section boundaries
4. **Formalize evaluation splits**: Create train/dev/test splits from our 36 fixtures and define standardized metrics

**Longer-term:**
5. **Build a Japanese CORD**: Create a properly annotated Japanese receipt dataset following CORD's schema conventions. This would be a significant contribution to the research community (no public Japanese receipt dataset exists with this level of annotation).
6. **Benchmark against CORD models**: Test DoNUT, LayoutLM, and other CORD-evaluated models on our Japanese receipts to see how they perform out-of-domain.

**Priority:** Medium-high for schema and evaluation methodology improvements; low for direct usage as a tool.
