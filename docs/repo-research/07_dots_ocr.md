# 07. dots.ocr / dots.mocr (rednote-hilab/dots.ocr)

**GitHub:** https://github.com/rednote-hilab/dots.ocr
**Rebranded repo:** https://github.com/rednote-hilab/dots.mocr (March 2026)
**Paper:** https://arxiv.org/abs/2512.02498
**Demo:** https://dotsocr.xiaohongshu.com

---

## 1. Overview

dots.ocr is a multilingual document layout parsing VLM from RedNote (Xiaohongshu) HiLab. It performs OCR, layout detection, table recognition, formula parsing, and reading order determination in a single 1.7B-parameter vision-language model. In March 2026, version 1.5 was rebranded as **dots.mocr** ("Multimodal OCR").

- **Stars:** ~1k+ (dots.ocr), growing (dots.mocr)
- **Language:** Python
- **License:** Custom (dots.ocr LICENSE AGREEMENT)
- **Last updated:** March 2026 (dots.mocr rebrand)
- **Approach:** Single compact VLM that handles all document parsing tasks via prompt switching. Built on Qwen2.5-1.5B LLM with a custom 1.2B vision encoder trained from scratch.

The key differentiator is the **extreme compactness** -- a 3B total parameter model (1.7B LLM + 1.2B vision encoder) that outperforms models 10-40x larger on document parsing benchmarks.

---

## 2. Architecture & How It Works

### Model Architecture

| Component | Details |
|-----------|---------|
| **LLM backbone** | Qwen2.5-1.5B (1.7B parameters) |
| **Vision encoder** | Custom 1.2B params, trained from scratch, NaviT architecture |
| **Max resolution** | 11 million pixels |
| **Total params** | ~3B (far smaller than competitors like Qwen2.5-VL-72B) |

### Training Pipeline (3 stages)

1. **Vision Encoder Pretraining**: Trained on image-text pairs from scratch
2. **Continued Pretraining**: OCR, video, grounding data; LLM frozen; produces `dots.vit`
3. **OCR Specialization**: Pure OCR dataset with phased parameter unfreezing; produces `dots.ocr.base`
4. **Supervised Fine-tuning**: ~300K samples, 3 cycles of iterative error-case refinement, multi-expert data QA

### Data Flow

1. Input image/PDF (up to 11M pixels)
2. Prompt selects task: `prompt_layout_all_en`, `prompt_web_parsing`, `prompt_scene_spotting`, `prompt_image_to_svg`
3. Model generates structured JSON with layout categories, bounding boxes, and text content
4. `post_process_output()` cleans and validates output
5. Output: JSON layout data + annotated images + markdown text

### Repository Structure

```
dots_ocr/
  model/        # Model loading and inference
  utils/        # Post-processing, visualization
  __init__.py
  parser.py     # Core DotsOCRParser class
demo/           # Example scripts and images
docker/         # Container configuration
tools/          # Model download, evaluation scripts
```

### Key Class: `DotsOCRParser`

- Supports both vLLM server and local HuggingFace inference
- ThreadPool-based parallel page processing for PDFs
- Configurable: temperature, top_p, max_completion_tokens, DPI, min/max pixels
- Output formats: JSON layout, annotated images, markdown, JSONL index

### Deployment Options

- **vLLM** (recommended, officially integrated since v0.11.0): Single GPU, ~0.9 memory utilization
- **HuggingFace Transformers**: Flash Attention 2, bfloat16
- **Docker**: Pre-built container

---

## 3. Key Features

- **Extreme efficiency**: 3B params competing with 72B+ models
- **Unified multi-task**: Layout detection, OCR, table recognition, formula parsing, reading order -- all via prompt switching in a single model
- **SVG generation** (dots.mocr-svg): Directly converts charts and diagrams to SVG code
- **Web screen parsing**: Analyzes webpage layouts
- **Scene text detection**: Handles text in natural images
- **11 layout categories**: Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text, Title
- **Grounding OCR**: Bounding boxes for every detected text element
- **100-language benchmark**: Evaluated on 1,493 PDF images across 100 languages

### Benchmark Highlights

| Benchmark | dots.ocr Score | Notable Comparison |
|-----------|---------------|-------------------|
| OmniDocBench EN Edit Distance | 0.125 | GPT-4o: 0.233, Qwen2.5-VL-72B: 0.214 |
| OmniDocBench EN Table TEDS | 88.6 | Competitive with much larger models |
| olmOCR-bench Overall | 79.1 +/- 1.0 | Across arXiv, scans, tables, multi-column |
| dots.ocr-bench Multilingual | 0.177 edit distance | 100 languages, 1,493 images |
| Layout F1@IoU .50 | 0.930 | DocLayout-YOLO: 0.806 |
| Elo Score (olmOCR-Bench + OmniDocBench + XDocParse) | 1124.7 | HuanyuanOCR: 984.2, PaddleOCR-VL: 920.5 |

---

## 4. Japanese Support

**Supported but not specifically benchmarked.** The multilingual benchmark covers 100 languages, and the model handles CJK scripts well given its Chinese-origin training data. However:

- No Japanese-specific accuracy numbers are published
- The blog post mentions "low-resource language" support but lists examples like Tibetan, Kannada, and Dutch -- not Japanese
- Japanese is NOT a low-resource language for OCR, so it likely benefits from the extensive CJK training data
- Receipt-specific Japanese formats (era dates, zenkaku numbers, tax-inclusive pricing) are not addressed
- The model outputs layout categories (Table, Text, Title, etc.) which don't map to receipt-specific fields

**Likely capability**: Good at reading Japanese text from document images, but no receipt schema extraction.

---

## 5. Strengths vs Our Project

| Their Strength | Detail |
|----------------|--------|
| **Compact single-model** | 3B params does everything -- no Cloud Vision API, no separate LLM call |
| **Layout understanding** | Native bounding box detection with 93% F1@IoU0.5 |
| **Multi-format output** | JSON layout, markdown, annotated images in one pass |
| **Multilingual by default** | 100 languages without separate OCR engine configuration |
| **SVG generation** | Can convert charts/diagrams to vector graphics |
| **Self-contained deployment** | Single model, single GPU, no external API dependencies |
| **Iterative data refinement** | 3-cycle error-case sampling + annotation is a rigorous approach to training data quality |

---

## 6. Weaknesses vs Our Project

| Our Strength | Detail |
|--------------|--------|
| **Receipt-specific extraction** | We extract typed fields (merchant, date, total, items, tax); dots.ocr outputs generic layout categories |
| **Validation pipeline** | Tax subset-sum matching, confidence routing, multi-pass verification are absent |
| **No structured schema** | dots.ocr produces layout JSON (bounding boxes + categories), not receipt data models |
| **GPU requirement** | Needs CUDA GPU; we use API calls with no local GPU |
| **Receipt domain knowledge** | Merchant rules, Japanese address parsing, era date conversion, payment slip handling |
| **Ground truth testing** | 36 fixtures with known-correct outputs; dots.ocr benchmarks are generic document metrics |
| **Post-processing** | Our field registry, normalization rules, and confidence-gated retry have no equivalent |
| **Known limitations** | dots.ocr struggles with "complex tables and formulas", "pictures within documents", and "continuous special characters" -- some of which appear on receipts |

---

## 7. What We Can Learn

1. **Iterative error-case data refinement**: Their 3-cycle approach (train -> find errors -> annotate error cases -> retrain) is a disciplined methodology we could apply to our test fixtures. When we find receipt types that fail, we should systematically add those as fixtures and tune the pipeline.

2. **Prompt-based task switching**: A single model handling multiple tasks via prompt variation is elegant. We could consider designing our LLM prompts to support multiple extraction modes (e.g., "receipt", "utility bill", "payment slip") via prompt templates rather than separate code paths.

3. **NaviT for high-resolution input**: Their use of NaviT architecture to handle up to 11M pixels efficiently is worth studying if we ever need to handle very high-resolution receipt scans.

4. **Reading order normalization**: dots.ocr explicitly trains on reading order. Our OCR text sometimes has ordering issues (e.g., columns merged incorrectly). If we pre-process OCR text with a reading order model, extraction accuracy might improve.

5. **Multi-expert data quality assurance**: Using multiple annotators/experts to validate training data is a practice we should adopt for our ground truth fixtures.

6. **ThreadPool-based PDF processing**: Their `DotsOCRParser` uses ThreadPool for parallel page processing -- a pattern we already use in benchmarking but could apply more broadly.

---

## 8. Recommendation

**Not a direct replacement, but a promising future OCR backend.**

dots.ocr/dots.mocr is impressive for its size-to-performance ratio, but it outputs layout structure, not receipt data. Like DeepSeek-OCR, we'd need to layer our extraction pipeline on top.

**Most interesting as:** A potential replacement for Google Cloud Vision as an OCR backend. The advantages would be:
- No API costs
- Layout-aware text ordering (addresses a real pain point in our OCR text)
- Single-GPU deployment (~4GB VRAM for 3B model in bf16)
- Multilingual out of the box

**Blockers:**
- Still requires GPU (even a small one)
- Output format (layout JSON) would need adapter code to feed into our LLM extraction stage
- No receipt-specific training or validation
- Custom license (not MIT/Apache) may restrict commercial use

**Action items:**
- Monitor dots.mocr releases for receipt-specific capabilities
- Consider benchmarking dots.ocr as an alternative OCR engine on our 36 test fixtures
- The compact size (3B) makes it feasible to run locally even on consumer GPUs
