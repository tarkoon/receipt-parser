# 09. Unstructured (Unstructured-IO/unstructured)

**GitHub:** https://github.com/Unstructured-IO/unstructured
**Docs:** https://docs.unstructured.io
**Website:** https://unstructured.io

---

## 1. Overview

Unstructured is a large-scale open-source document ETL (Extract, Transform, Load) library that converts complex documents into clean, structured data for LLM pipelines. It handles 25+ file formats with automatic type detection, multiple processing strategies, and pluggable OCR/layout detection backends.

- **Stars:** ~14.4k
- **Language:** Python
- **License:** Apache 2.0
- **Last updated:** Active (1,876+ commits on main)
- **Approach:** Modular document processing pipeline with strategy-based routing (fast, hi_res, ocr_only), pluggable OCR engines, layout detection via deep learning models, and extensive format support.

This is not an OCR tool or a receipt parser -- it's a **document preprocessing framework** that sits between raw documents and LLM applications. Think of it as the plumbing that turns any document into structured elements.

---

## 2. Architecture & How It Works

### Core Pipeline

```
Document Input -> File Type Detection -> Strategy Selection -> Partitioning -> Element Extraction -> Post-processing -> Output
```

### Key Abstraction: `partition()`

The central API is a single function that auto-detects file types and routes to appropriate handlers:

```python
from unstructured.partition.auto import partition
elements = partition(filename="receipt.pdf")
```

### Processing Strategies

| Strategy | Method | Speed | Quality | Use Case |
|----------|--------|-------|---------|----------|
| **FAST** | PDFMiner text extraction | Fast | Good for text PDFs | Extractable text documents |
| **HI_RES** | Layout detection model + OCR | Slow | Best | Scanned documents, complex layouts |
| **OCR_ONLY** | Tesseract/PaddleOCR | Medium | Good for images | Pure image inputs |
| **AUTO** | Heuristic selection | Varies | Adaptive | Default -- picks best strategy per document |

### Auto-Strategy Logic

For PDFs:
1. If table structure inference needed -> HI_RES
2. If image extraction needed -> HI_RES
3. If extractable text found -> FAST
4. Otherwise -> OCR_ONLY

For images: Always HI_RES (since "images are only about one page").

### Complexity Detection

`is_pdf_too_complex()` prevents performance issues:
- Skips files under 1MB
- Counts graphics/text operators in PDF content streams
- Flags vector-heavy CAD/engineering documents that would bog down processing

### Module Structure

```
unstructured/
  partition/           # Core document processing
    auto.py            # Auto file-type detection and routing
    pdf.py             # PDF partitioning (3 strategies)
    image.py           # Image partitioning (delegates to pdf_image)
    pdf_image/         # Shared PDF/image processing
    html/              # HTML partitioning
    docx.py            # Word document partitioning
    xlsx.py            # Excel partitioning
    email.py           # Email partitioning
    strategies.py      # Strategy definitions and selection logic
    ... (25+ format handlers)
  chunking/            # Text segmentation
  cleaners/            # Data cleaning utilities
  documents/           # Document representations
  embed/               # Embedding generation
  metrics/             # Performance measurement
  models/              # Data models and schemas
  nlp/                 # NLP components
  common/              # Shared utilities
```

### Supported File Formats (25+)

PDF, Images (PNG/JPG/TIFF/BMP), DOCX, DOC, PPTX, PPT, XLSX, HTML, XML, Markdown, RST, ODT, RTF, EPUB, CSV, TSV, JSON, NDJSON, EML, MSG, ORG, Audio, and more.

### OCR Integration

- **Tesseract** (primary): Language packs installable, `ocr_languages` parameter
- **PaddleOCR**: Supported as alternative
- Language detection per-element supported (`detect_language_per_element=True`)

### Output: Elements

Documents are parsed into typed `Element` objects:
- `Title`, `NarrativeText`, `ListItem`, `Table`, `Image`, `FigureCaption`, `Address`, `EmailAddress`, `Header`, `Footer`, `PageBreak`
- Each element has `.text`, `.metadata` (coordinates, page number, file info), and optional `.embeddings`
- Tables can include `text_as_html` for structure preservation

### System Dependencies

- `libmagic-dev` (filetype detection)
- `poppler-utils` (PDF rendering)
- `tesseract-ocr` + language packs
- `libreoffice` (MS Office document conversion)
- `pandoc` (various format conversions)

---

## 3. Key Features

- **Universal format support**: 25+ document types with automatic detection
- **Strategy-based processing**: Fast/HI_RES/OCR_ONLY with intelligent auto-selection
- **Layout detection**: Deep learning models (LayoutParser-based) for document structure analysis
- **Table extraction**: Structured HTML output for tables with `infer_table_structure=True`
- **Chunking**: Built-in text segmentation for LLM context window management
- **Embedding integration**: Generate embeddings as part of the pipeline
- **Element typing**: Rich semantic element types (Title, Table, Address, etc.)
- **Image extraction**: Extract embedded images with base64 encoding
- **Form extraction**: Handle PDF forms
- **Metadata preservation**: File metadata, coordinates, page numbers carried through pipeline
- **Telemetry opt-in**: Disabled by default, transparent data collection
- **Enterprise Platform**: Production workflows, UI, API (commercial offering)

---

## 4. Japanese Support

**Supported via Tesseract language packs, but not first-class.**

- Tesseract Japanese language pack (`tesseract-ocr-jpn`) can be installed and passed via `ocr_languages="jpn"`
- No Japanese-specific text normalization, date handling, or format awareness
- Layout detection models are trained primarily on English/Latin documents
- Table extraction would work for structured Japanese receipt tables, but field extraction requires additional processing
- The `detect_language_per_element` feature could identify Japanese text elements

**Key limitation:** Unstructured extracts document **structure** (elements, tables, text blocks) but does not extract **meaning** (merchant name, total amount, date). For receipt parsing, you'd still need an LLM extraction layer on top of Unstructured's output.

---

## 5. Strengths vs Our Project

| Their Strength | Detail |
|----------------|--------|
| **Format universality** | Handles 25+ formats; we only handle images |
| **Strategy routing** | Intelligent fast/hi_res/ocr_only selection based on document characteristics; we always use Cloud Vision |
| **Layout detection** | Deep learning layout analysis identifies document structure; we rely on OCR text ordering |
| **Table extraction** | HTML-structured table output; we reconstruct table data from raw text |
| **Complexity detection** | `is_pdf_too_complex()` prevents runaway processing; we have no such safeguard |
| **Element typing** | Semantic types (Title, Table, Address) provide structure before LLM extraction |
| **Chunking** | Built-in text segmentation for LLM input; we send full OCR text |
| **Community & ecosystem** | 14.4k stars, Apache 2.0, active development, enterprise support |
| **Pluggable OCR** | Tesseract + PaddleOCR with language packs; we're locked to Cloud Vision |
| **Metadata pipeline** | Coordinates, page numbers, file metadata flow through the entire pipeline |

---

## 6. Weaknesses vs Our Project

| Our Strength | Detail |
|--------------|--------|
| **Receipt-specific extraction** | We extract typed receipt fields; Unstructured extracts generic document elements |
| **No semantic extraction** | Unstructured tells you "this is a Table" or "this is NarrativeText" but not "this is the total amount" |
| **No LLM integration** | Unstructured is preprocessing only -- no LLM extraction step |
| **No validation** | No tax calculation verification, no subset-sum matching |
| **No confidence scoring** | Elements don't have extraction confidence; we have OCR + LLM confidence per field |
| **Heavy dependencies** | Requires libmagic, poppler, tesseract, libreoffice, pandoc; we need only Python + API keys |
| **Complexity for our use case** | Massively over-engineered for single-receipt parsing |
| **No receipt domain knowledge** | No merchant rules, no Japanese receipt conventions, no era dates |
| **Tesseract OCR quality** | Their primary OCR (Tesseract) is significantly worse than Google Cloud Vision for Japanese text |
| **No determinism controls** | No seed/temperature for reproducible results |

---

## 7. What We Can Learn

1. **Strategy-based processing**: The AUTO/FAST/HI_RES/OCR_ONLY pattern is excellent. We could implement similar quality tiers:
   - **FAST**: Use cached OCR + single-pass LLM extraction (for known-good receipt formats)
   - **HI_RES**: Full pipeline with confidence routing, multi-pass verification, retry (for difficult receipts)
   - **OCR_ONLY**: Return raw OCR text without LLM extraction (for debugging/preview)

2. **Complexity detection heuristic**: Their `is_pdf_too_complex()` that checks graphics/text operator ratios could inspire a similar receipt complexity score. Receipts with many columns, mixed text/images, or unusual layouts could be routed to more thorough processing.

3. **Element typing as intermediate representation**: Instead of passing raw OCR text to the LLM, we could first identify element types (header, line items table, totals section, address block) and then extract fields from each element type separately. This would make extraction more modular and debuggable.

4. **Table structure inference**: Their `infer_table_structure=True` that outputs HTML tables is a technique we could use for line item extraction. If we could identify the line items portion of a receipt as a table and get HTML structure, field extraction would be much simpler.

5. **Pluggable OCR backend**: Their support for multiple OCR engines (Tesseract, PaddleOCR) via a common interface is a pattern we should adopt. This would let us swap Cloud Vision for cheaper alternatives on easy receipts.

6. **`partition()` as universal entry point**: Their single `partition()` function that handles any input is a clean API pattern. We could expose a similar `parse_receipt()` that accepts file paths, byte streams, or URLs.

7. **Metadata pipeline**: Carrying coordinates, page numbers, and source file information through every processing step is valuable for debugging and auditing. We should preserve more OCR metadata (bounding boxes, confidence per word) through our pipeline.

---

## 8. Recommendation

**Not a replacement for our pipeline, but potentially useful as a preprocessing layer.**

Unstructured solves the "turn any document into text" problem very well, but it stops short of semantic extraction. For our use case:

**Could be useful for:**
- Processing receipt PDFs (we currently only handle images)
- Extracting table structure from complex receipts before LLM extraction
- Adding multi-format support (if we ever need to handle DOCX invoices, HTML receipts, email receipts)
- As a preprocessing step when receipts are embedded in larger documents

**Not useful for:**
- Replacing our LLM extraction pipeline
- Japanese-specific receipt parsing
- Validation and confidence scoring

**Recommended integration point:** Use `unstructured` to partition complex documents into elements, then feed those elements to our existing extraction pipeline. Specifically:
```python
from unstructured.partition.image import partition_image
elements = partition_image(filename="receipt.jpg", strategy="hi_res", infer_table_structure=True)
# Extract tables for line items, text blocks for header/footer info
# Feed to our LLM extraction pipeline
```

**Priority:** Low. Our current Cloud Vision -> text -> DeepSeek pipeline works well for receipt images. Unstructured would add value only if we expand to multi-format document support.
