# PaddlePaddle/PaddleOCR

> https://github.com/PaddlePaddle/PaddleOCR

| Field | Value |
|-------|-------|
| Stars | ~74,700 |
| Language | Python (PaddlePaddle framework) |
| License | Apache-2.0 |
| Last Updated | 2026-04-02 |
| Approach | Modular OCR toolkit + VLM + document structure engine |

---

## 1. Overview

PaddleOCR is Baidu's flagship open-source OCR toolkit and document AI engine. It has evolved from a straightforward OCR library into a comprehensive document understanding platform with three major subsystems:

- **PP-OCRv5** -- Scene text recognition supporting 100+ languages in a single model, with a 13% accuracy boost over v4.
- **PaddleOCR-VL-1.5** -- A 0.9B-parameter vision-language model achieving 94.5% on OmniDocBench, handling skewed/warped/scanned documents, seal recognition, cross-page table merging, and 111 languages.
- **PP-StructureV3** -- A complex document parsing pipeline that converts PDFs/images into Markdown or JSON with fine-grained coordinate data for tables, formulas, charts, and seals.

The project is deeply integrated into production RAG ecosystems (Dify, RAGFlow, Pathway, Cherry Studio) and has 6,000+ dependent repositories.

---

## 2. Architecture & How It Works

### Code Structure

```
paddleocr/
  _models/          # Model definitions (detection, recognition, layout, table, etc.)
  _pipelines/       # Pipeline orchestration
    ocr.py           # Standard OCR pipeline
    pp_structurev3.py # Complex document parsing
    table_recognition_v2.py
    formula_recognition.py
    seal_recognition.py
    doc_understanding.py
    paddleocr_vl.py  # Vision-language pipeline
    ...
  _utils/           # Shared utilities
  _abstract.py      # Abstract base classes for pipeline stages
  _cli.py           # CLI entry point
  _constants.py     # Configuration constants
```

### PP-StructureV3 Pipeline Flow

1. **Document Preprocessing** -- Orientation classification, image unwarping/deskewing
2. **Layout Detection** -- NMS-based region detection (text, table, formula, chart, seal, figure)
3. **Branched Recognition**:
   - Text regions -> PP-OCRv5 (detection + recognition + direction classification)
   - Tables -> Separate wired/wireless table pipelines (structure recognition + cell detection)
   - Formulas -> Dedicated formula recognition model
   - Charts -> Chart interpretation model
   - Seals -> Seal recognition model
4. **Output Formatting** -- Structured Markdown or JSON with coordinate metadata

### Model Selection

The pipeline automatically selects OCR models based on language parameter and PP-OCR version (v3/v4/v5). Language-specific models exist for CJK, Latin, Cyrillic, Devanagari, Arabic, etc.

### VL Pipeline

PaddleOCR-VL-1.5 operates differently -- it's an end-to-end 0.9B VLM that processes images directly without the modular pipeline. It uses the PP-DocLayoutV3 algorithm for irregular document handling and supports hierarchical heading identification and cross-page table merging.

---

## 3. Key Features

- **Single-model multilingual OCR**: One PP-OCRv5 model handles 100+ languages without switching
- **Production deployment**: ONNX export, TensorRT, OpenVINO, C++ inference, Docker, multi-GPU parallel
- **PP-StructureV3**: Best-in-class document structure parsing with coordinate-level precision
- **Seal recognition**: Specialized pipeline for Japanese hanko/inkan stamps (highly relevant to our use case)
- **Ultra-small footprint**: Designed for edge/mobile deployment
- **Layout-aware output**: Markdown with spatial metadata, not just flat text
- **Cross-page table merging**: Handles tables that span multiple pages

---

## 4. Japanese Support

**Strong**. Japanese is explicitly supported across all three subsystems:

- PP-OCRv5 includes Japanese as one of the core supported languages
- The recognition model handles kanji, hiragana, katakana natively
- PaddleOCR-VL-1.5 expanded coverage to 111 languages including Japanese
- PP-StructureV3 can process Japanese document layouts

However, PaddleOCR is trained on a very broad multilingual corpus. Japanese-specific optimizations (vertical text, furigana, era dates, yen formatting) are not explicitly called out. It is a generalist system, not a Japanese specialist.

---

## 5. Strengths vs Our Project

| Area | PaddleOCR Advantage |
|------|---------------------|
| **OCR Engine** | Native high-accuracy OCR models vs our dependency on Google Cloud Vision API. PaddleOCR can run fully offline with no API costs. |
| **Document Structure** | PP-StructureV3 provides layout detection, table extraction, formula recognition -- capabilities we don't have at all. |
| **Seal Recognition** | Dedicated seal/stamp recognition pipeline. We have no equivalent for hanko detection on Japanese receipts. |
| **Deployment Flexibility** | ONNX, TensorRT, C++, Docker, mobile. We're locked to Python + GCV API. |
| **Scale** | Handles complex multi-page documents with cross-page table merging. Our pipeline processes single receipt images. |
| **VLM Option** | PaddleOCR-VL-1.5 offers an end-to-end alternative that bypasses the OCR->LLM pipeline entirely. |

---

## 6. Weaknesses vs Our Project

| Area | Our Advantage |
|------|---------------|
| **Receipt-Specific Logic** | We have merchant rules, tax category assignment via subset-sum matching, and field-specific post-processing. PaddleOCR outputs raw text/structure. |
| **Confidence Routing** | Our v3.0 confidence routing (OCR confidence vs LLM confidence per field) is a unique architectural innovation. PaddleOCR has confidence scores but no cross-system routing. |
| **LLM Integration** | Our DeepSeek V3.2 extraction with multi-pass verification is purpose-built for receipt fields. PaddleOCR's VL model is general-purpose. |
| **Validation Layer** | Pydantic schema validation, field registry, ground truth testing with 36 fixtures. PaddleOCR provides OCR output, not validated structured data. |
| **Japanese Receipt Domain** | Era dates (Reiwa/Heisei), yen formatting, Japanese tax categories, utility bill parsing -- none of this exists in PaddleOCR. |
| **Determinism** | Our seed=42 deterministic pipeline with variance attribution benchmarks. PaddleOCR focuses on accuracy, not reproducibility. |

---

## 7. What We Can Learn

### 7.1 Replace Google Cloud Vision with PP-OCRv5
PaddleOCR could replace our Google Cloud Vision dependency entirely. Benefits:
- Zero API cost (runs locally)
- No network latency
- Full control over OCR model behavior
- We already have PaddleOCR 3.x in our project history (see MEMORY.md references)

**Risk**: GCV may still be more accurate for our specific Japanese receipt corpus. Would need head-to-head benchmark.

### 7.2 Seal/Stamp Recognition
PP-StructureV3's seal recognition pipeline could help us detect and extract information from hanko stamps on receipts and payment slips. This is a gap in our current pipeline.

### 7.3 Layout-Aware Preprocessing
PP-StructureV3's layout detection could pre-classify receipt regions (header, line items, totals, tax section) before sending to our LLM. This would reduce LLM hallucination by providing structured spatial context.

### 7.4 Document Orientation/Deskewing
PaddleOCR's preprocessing (orientation classification, unwarping) could improve OCR accuracy on poorly-photographed receipts before they hit our pipeline.

### 7.5 Abstract Pipeline Pattern
PaddleOCR's `_abstract.py` base classes and `_pipelines/` module structure is a clean architectural pattern. Our pipeline could benefit from similar formalization of stage interfaces.

---

## 8. Recommendation

**Consider as OCR replacement or preprocessing layer.** PaddleOCR is the strongest candidate for replacing our Google Cloud Vision dependency. The path would be:

1. **Short-term**: Use PP-OCRv5 as an alternative OCR backend alongside GCV. Benchmark both on our 36 fixtures.
2. **Medium-term**: Add PP-StructureV3 layout detection as a preprocessing stage to provide spatial context to our LLM.
3. **Long-term**: Evaluate PaddleOCR-VL-1.5 as a potential replacement for our entire OCR->LLM pipeline, with our validation/post-processing layer on top.

PaddleOCR does NOT replace our receipt-specific extraction logic, confidence routing, or validation layer. It's an OCR engine, not a receipt parser. We would still need everything from normalization onward.

**License**: Apache-2.0 is fully compatible with commercial use.
