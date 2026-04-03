# kotaro-kinoshita/yomitoku

> https://github.com/kotaro-kinoshita/yomitoku

| Field | Value |
|-------|-------|
| Stars | ~1,400 |
| Language | Python (PyTorch) |
| License | CC BY-NC-SA 4.0 (non-commercial; commercial license available) |
| Last Updated | 2026-04-02 |
| Approach | Japanese-specialized AI document analysis engine |

---

## 1. Overview

Yomitoku ("reading tool" in Japanese) is a Python package purpose-built for Japanese document analysis. Unlike general-purpose OCR systems, every model in Yomitoku was trained specifically on Japanese document images. It performs full-text OCR with layout analysis, table structure recognition, reading order estimation, and structured data extraction.

This is the most directly relevant project to our receipt parser -- it's the only repo in this comparison that was designed from the ground up for Japanese documents.

Key capabilities:
- OCR for 7,000+ Japanese characters (kanji, hiragana, katakana)
- Vertical text (tategaki) and horizontal text (yokogaki) handling
- Ruby text (furigana) detection and filtering
- Table structure recognition with cell-level content mapping
- Reading order estimation adapted for Japanese layout conventions
- Both rule-based and LLM-based structured data extraction
- Export to HTML, Markdown, JSON, CSV, searchable PDF

---

## 2. Architecture & How It Works

### Code Structure

```
src/yomitoku/
  cli/                # Command-line interface
  configs/            # Model and pipeline configuration
  data/               # Data handling utilities
  export/             # Output formatters (HTML, Markdown, JSON, CSV, PDF)
  extractor/          # Structured data extraction
    pipeline.py         # LLM-based extraction orchestration
    rule_pipeline.py    # Rule-based extraction
    llm_client.py       # LLM API client
    prompt.py           # LLM prompt construction
    normalizer.py       # Value normalization
    resolver.py         # Field resolution against OCR data
    schema.py           # Extraction schema definitions
    visualizer.py       # Results visualization
  models/             # Neural network model definitions
  onnx/               # ONNX runtime integration
  postprocessor/      # Post-processing logic
  resource/           # Resource management
  schemas/            # Pydantic data schemas
  utils/              # Shared utilities

  # Core pipeline modules:
  ocr.py                    # OCR orchestration (detection + recognition)
  text_detector.py          # Text region localization
  text_recognizer.py        # Character recognition
  layout_analyzer.py        # Document layout classification
  layout_parser.py          # Layout interpretation
  reading_order.py          # Reading order estimation
  table_cell_detector.py    # Table cell identification
  table_structure_recognizer.py  # Table grid structure
  table_semantic_parser.py  # Semantic table understanding
  grid_parser.py            # Grid-based content parsing
  kv_parser.py              # Key-value pair extraction
  document_analyzer.py      # Full document analysis orchestration
  base.py                   # Base classes
  constants.py              # Constants
```

### Full Pipeline (DocumentAnalyzer)

1. **Parallel Detection & Layout Analysis** (ThreadPoolExecutor)
   - Text detection: Locates all text regions with bounding boxes
   - Layout analysis: Classifies page regions (paragraphs, tables, figures, headers, footers)

2. **Text Recognition**
   - Character-level OCR on detected text regions
   - Produces `det_score` (detection confidence) and `rec_score` (recognition confidence) per word
   - Detects text direction (horizontal vs vertical)

3. **Table Processing** (optional)
   - Cell boundary detection
   - Structure recognition (rows, columns, spanning cells)
   - Content allocation: overlap ratio computation between words and cells

4. **Aggregation**
   - Words assigned to structural elements (paragraphs, table cells, figure captions)
   - Orphaned words become standalone paragraphs
   - Figures extract contained paragraphs as sub-elements

5. **Ruby Text Filtering**
   - Histogram-based size distribution analysis
   - Small annotations (furigana) detected and optionally removed

6. **Reading Order Estimation**
   - Graph-based DFS algorithm adapted for text direction
   - Three modes: `top2bottom` (vertical), `right2left`, `left2right`
   - Obstruction detection prevents illogical reading order jumps
   - Priority sorting by spatial distance metric per direction

7. **Export**
   - Structured output in chosen format (JSON, HTML, Markdown, CSV, searchable PDF)

### OCR Module (ocr.py)

```python
# Two-stage pipeline
det_outputs, vis = self.detector(img)          # Stage 1: Text detection
rec_outputs, vis = self.recognizer(img, det_outputs.points, vis=vis)  # Stage 2: Recognition

# Unified word objects with dual confidence scores
word = {
    'det_score': detection_confidence,
    'rec_score': recognition_confidence,
    'direction': text_direction,
    'points': spatial_coordinates,
    'content': recognized_text
}
```

Output validated through `OCRSchema` (Pydantic) before returning.

### Extractor Module

**Two extraction modes:**

1. **Rule-Based** (`rule_pipeline.py`): Pattern matching, regex, key-value search. Fast, deterministic, high-precision for standard forms.

2. **LLM-Based** (`pipeline.py`):
   - `build_messages()`: Constructs prompts from OCR semantic info + extraction schema
   - `call_llm()`: Invokes LLM (via vLLM) with temperature/token controls
   - `resolve_fields()`: Maps LLM output back to OCR-detected elements via lookup tables
   - `_normalize_resolved_fields()`: Field-specific value transformations
   - Two output formats: detailed (with cell IDs, bounding boxes) or simple (values only)

### Reading Order Algorithm

Graph-based DFS with direction-aware priority:
- **Vertical text**: Priority = `box[0] + box[1]` (top-right to bottom-left)
- **Right-to-left**: Priority = `(max_x - box[2]) + box[1]`
- **Left-to-right**: Priority = `box[0] * x_weight + box[1] * y_weight`
- Intersection checking creates directional edges between text blocks
- Obstruction detection ensures no illogical jumps across intervening content

---

## 3. Key Features

- **Japanese-first design**: All models trained specifically on Japanese documents, not adapted from general-purpose models
- **7,000+ character support**: Full kanji coverage including rare characters
- **Vertical text handling**: Native support for tategaki with correct reading order
- **Ruby/furigana filtering**: Histogram-based detection removes furigana annotations that confuse text extraction
- **Dual confidence scores**: Separate detection and recognition confidence per word
- **Reading order intelligence**: Graph-based algorithm respects Japanese reading conventions
- **Dual extraction modes**: Rule-based (fast, deterministic) + LLM-based (flexible, context-aware)
- **Lightweight variant**: CPU-only inference with `--lite` flag for deployment without GPU
- **Searchable PDF output**: Re-embeds OCR text into PDF for searchability
- **ONNX export**: Cross-platform model deployment

---

## 4. Japanese Support

**Best-in-class for this comparison.** Yomitoku is the only project built exclusively for Japanese:

- Every model trained on Japanese document images specifically
- 7,000+ Japanese character recognition
- Vertical text (tategaki) as a first-class layout mode
- Ruby text (furigana) detection with histogram-based size analysis
- Reading order adapted for Japanese conventions (top-to-bottom, right-to-left column ordering)
- Header/footer detection for Japanese document layouts

This is not "Japanese support" bolted onto a multilingual system -- it's a Japanese document engine that happens to also read some Latin characters.

---

## 5. Strengths vs Our Project

| Area | Yomitoku Advantage |
|------|-------------------|
| **Japanese OCR Quality** | Purpose-trained Japanese models vs our reliance on GCV's general multilingual OCR. Likely better on rare kanji, handwritten text, and vertical layouts. |
| **Vertical Text** | Native tategaki support with correct reading order. Our pipeline treats vertical text as a special case in post-processing. |
| **Ruby/Furigana Handling** | Histogram-based detection and filtering. We have no furigana-aware processing. |
| **Layout Analysis** | Full document structure (paragraphs, tables, figures, headers/footers). We treat receipts as flat text. |
| **Table Extraction** | Cell-level table parsing with structure recognition. We don't handle tables. |
| **Reading Order** | Graph-based DFS algorithm adapted for Japanese. Our pipeline relies on OCR output order. |
| **Dual Extraction** | Rule-based + LLM-based extraction modes. We only have LLM-based with post-processing rules. |
| **Dual Confidence** | Separate det_score and rec_score per word. We get a single confidence from GCV. |
| **Key-Value Parser** | Dedicated `kv_parser.py` for form-like documents. We use LLM for all extraction. |
| **Offline Operation** | Runs fully locally with PyTorch/ONNX. No API dependencies. |

---

## 6. Weaknesses vs Our Project

| Area | Our Advantage |
|------|---------------|
| **Receipt-Specific Logic** | Merchant rules, tax category assignment, subset-sum matching, field registry. Yomitoku is a document analyzer, not a receipt parser. |
| **Confidence Routing** | Our v3.0 cross-system confidence routing (OCR vs LLM per field). Yomitoku has dual confidence but no cross-system routing. |
| **LLM Integration Depth** | DeepSeek V3.2 multi-pass verification. Yomitoku's LLM integration is optional and simpler. |
| **Determinism** | seed=42, variance attribution benchmarks, robustness testing. Yomitoku doesn't emphasize reproducibility. |
| **Ground Truth Testing** | 36 fixtures with truth files, automated accuracy benchmarks. Yomitoku has no published test infrastructure. |
| **Post-Processing Sophistication** | Text normalization, field-specific cleanup, tax validation. Yomitoku's normalizer is lighter. |
| **Multilingual** | Our pipeline handles any GCV-supported language. Yomitoku is Japanese-only. |
| **License** | CC BY-NC-SA 4.0 prohibits commercial use without a separate license. Our pipeline has no such restriction. |
| **LLM Model Choice** | We use DeepSeek V3.2 via API (state of the art). Yomitoku uses vLLM with unspecified models. |

---

## 7. What We Can Learn

### 7.1 Replace GCV with Yomitoku OCR for Japanese Receipts
Yomitoku's Japanese-specific OCR models could significantly outperform GCV on our receipt corpus:
- Better rare kanji recognition
- Proper vertical text handling
- Furigana filtering (prevents ruby text from contaminating extracted values)
- Separate det_score and rec_score for more granular confidence routing

**Action**: Benchmark Yomitoku OCR vs GCV on our 36 fixtures. If Yomitoku wins, use it as our OCR backend for Japanese documents.

### 7.2 Ruby/Furigana Filtering
Yomitoku's histogram-based furigana detection is a technique we should adopt immediately. Furigana annotations on receipts (e.g., store name readings) can contaminate our text extraction. A histogram-based size filter is:
- Computationally cheap
- Language-agnostic in implementation
- Directly applicable to our pipeline as a post-OCR filter

### 7.3 Reading Order Algorithm
Yomitoku's graph-based DFS reading order is more sophisticated than relying on OCR output order. For receipts with complex layouts (multiple columns, vertical headers), this could improve extraction accuracy. Key insight: the obstruction detection that prevents illogical jumps.

### 7.4 Dual Confidence Scoring
Separate detection and recognition confidence per word is more informative than a single OCR confidence score. We could:
- Use det_score to decide if a text region was correctly localized
- Use rec_score to decide if the characters were correctly read
- Route differently based on which confidence is low

### 7.5 Rule-Based + LLM Extraction Pattern
Yomitoku's dual extraction mode (rule-based for standard forms, LLM for variable layouts) matches our intuition. We could formalize this:
- Rule-based extraction for known receipt formats (7-Eleven, Lawson, etc.)
- LLM extraction for unknown/variable formats
- This would reduce LLM API calls and improve speed/cost for common receipts

### 7.6 Key-Value Parser
`kv_parser.py` for form-like documents is directly applicable to receipts. Many receipt fields are key-value pairs ("Tax: 800", "Total: 10,800"). A dedicated KV parser before LLM extraction could handle easy fields cheaply.

### 7.7 ThreadPoolExecutor for Parallel Stages
Yomitoku runs text detection and layout analysis in parallel with ThreadPoolExecutor. We could parallelize our OCR and any preprocessing stages similarly.

---

## 8. Recommendation

**Highest relevance to our project. Evaluate as OCR replacement and adopt specific techniques.**

Yomitoku is the most directly applicable project in this comparison. It's solving adjacent problems with Japanese-specific solutions. The recommended approach:

1. **Immediate**: Benchmark Yomitoku OCR vs GCV on our 36 Japanese receipt fixtures
2. **Adopt techniques**: Ruby/furigana filtering, dual confidence scoring, graph-based reading order
3. **Consider hybrid**: Use Yomitoku for Japanese OCR + layout analysis, keep our LLM extraction + post-processing + validation on top
4. **Key-value parser**: Implement a rule-based KV parser for common receipt fields before LLM extraction

**Blockers**:
- **License**: CC BY-NC-SA 4.0 is non-commercial. Commercial use requires a separate license (available via AWS Marketplace or on-premise licensing). This must be resolved before any production use.
- **GPU requirement**: Standard models need CUDA. Lite variant has 50-char-per-line limitation.
- **VRAM**: Under 8GB but still requires GPU allocation.

Yomitoku is not a replacement for our pipeline -- it's a document analyzer, not a receipt parser. But it could be a far superior OCR backend for Japanese documents compared to GCV.
