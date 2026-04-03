# 06. DeepSeek-OCR (deepseek-ai/DeepSeek-OCR)

**GitHub:** https://github.com/deepseek-ai/DeepSeek-OCR
**Paper:** https://arxiv.org/html/2510.18234v1

---

## 1. Overview

DeepSeek-OCR is an end-to-end vision-language model for OCR that frames the problem as "Contexts Optical Compression" -- compressing visual information into minimal tokens while preserving textual fidelity. Released October 2025 by DeepSeek-AI, with DeepSeek OCR 2 following in January 2026.

- **Stars:** ~22.8k
- **Language:** Python (100%)
- **License:** MIT
- **Last updated:** October 2025 (v1), January 2026 (v2)
- **Approach:** Single end-to-end VLM that does OCR, layout detection, document-to-markdown conversion, spatial grounding, and chart/figure parsing -- all in one model.

This is fundamentally different from our pipeline. DeepSeek-OCR replaces the entire OCR + LLM extraction chain with a single model that sees the image and outputs structured text directly.

---

## 2. Architecture & How It Works

### Core Components

| Component | Details |
|-----------|---------|
| **DeepEncoder** | ~380M params: SAM-base (80M) + CLIP-large (300M) in series, connected by 16x convolutional downsampling |
| **Decoder** | DeepSeek-3B-MoE: 570M activated params (6/64 routed experts + 2 shared experts) |
| **Resolution modes** | Tiny (512x512, 64 tokens), Small (640x640, 100 tokens), Base (1024x1024, 256 tokens), Large (1280x1280, 400 tokens), Gundam (dynamic n*640+1024, <800 tokens) |

### Data Flow

1. Image input at configurable resolution
2. DeepEncoder compresses visual information to a small number of vision tokens (64-400)
3. Prompt injected (`<image>\nFree OCR` or `<image>\n<|grounding|>Convert the document to markdown`)
4. MoE decoder generates text output (markdown, grounding coordinates, or free-form OCR)
5. Post-processing extracts bounding boxes from `<|ref|>...<|/ref|><|det|>...<|/det|>` tags

### Inference Backends

- **vLLM** (recommended): ~2,500 tokens/sec on A100-40G, streaming output, batch evaluation
- **HuggingFace Transformers**: Direct loading with Flash Attention 2, bfloat16

### Key Files

```
DeepSeek-OCR-master/
  DeepSeek-OCR-vllm/
    run_dpsk_ocr_image.py   # Streaming image OCR (303 lines)
    run_dpsk_ocr_pdf.py     # Concurrent PDF processing
    run_dpsk_ocr_eval_batch.py  # Benchmark evaluation
  DeepSeek-OCR-hf/
    run_dpsk_ocr.py          # HuggingFace inference (simple)
requirements.txt             # transformers, PyMuPDF, einops, etc.
DeepSeek_OCR_paper.pdf
```

### Training Data Scale

- OCR 1.0: 30M PDFs (~100 languages), 20M fine annotations (Chinese/English), 3M Word docs, 10M scene images
- OCR 2.0: 10M charts, 5M chemical formulas, 1M geometry figures
- General vision: 20% of training mix
- Production: 200k+ pages/day on single A100, 33M pages/day on 20-node cluster

---

## 3. Key Features

- **Extreme token efficiency**: 100 tokens (Small mode) outperforms GOT-OCR2.0 at 256 tokens on OmniDocBench
- **Single-model versatility**: OCR, layout detection, markdown conversion, chart parsing, spatial grounding -- all via prompt variation
- **Document-to-markdown**: Directly converts document images to structured markdown preserving layout
- **Grounding**: Can locate specific text regions with bounding box coordinates
- **Production-ready vLLM integration**: Streaming, batching, high throughput
- **Multi-resolution inference**: Same model handles different quality/speed tradeoffs
- **NoRepeatNGram logits processor**: Custom n-gram blocking (size 30, window 90) with whitelist tokens for table markup -- prevents repetitive output

---

## 4. Japanese Support

**Implicit but not explicitly validated.** The model was trained on "nearly 100 languages" for PDF documents, and the OCR 1.0 dataset includes 30M multilingual PDFs. However:

- The paper provides **no Japanese-specific benchmarks** -- only English and Chinese metrics on OmniDocBench
- CJK support is mentioned broadly but Japanese is not called out specifically
- Third-party reviews confirm Japanese text can be processed, but accuracy metrics for Japanese receipts specifically are unavailable
- The model's strong Chinese performance (shared kanji) suggests reasonable Japanese capability, but receipt-specific formats (era dates, zenkaku numbers, Japanese address formats) are unlikely to be well-handled without fine-tuning

---

## 5. Strengths vs Our Project

| Their Strength | Detail |
|----------------|--------|
| **No OCR dependency** | Eliminates Google Cloud Vision entirely -- the model sees the image directly |
| **Token efficiency** | 100-400 vision tokens vs our full OCR text output (often 500+ tokens to the LLM) |
| **Layout preservation** | Spatial understanding is baked into the model, not reconstructed from OCR text |
| **Throughput** | 2,500 tokens/sec on A100 with vLLM; designed for production scale |
| **Grounding** | Can identify WHERE text appears, not just WHAT it says |
| **Chart/figure parsing** | Handles charts, formulas, geometry that our pipeline ignores |

---

## 6. Weaknesses vs Our Project

| Our Strength | Detail |
|--------------|--------|
| **Receipt-specific schema** | We have Pydantic models, field registry, merchant rules tailored to Japanese receipts; DeepSeek-OCR outputs generic markdown |
| **No post-processing** | DeepSeek-OCR has no tax calculation verification, subset-sum matching, or validation logic |
| **No confidence routing** | We gate OCR confidence vs LLM confidence per field; DeepSeek-OCR is all-or-nothing |
| **GPU requirement** | DeepSeek-OCR requires CUDA 11.8 + A100-class GPU; we run with API calls |
| **No structured extraction** | DeepSeek-OCR outputs markdown/text, not structured JSON with typed fields |
| **Receipt domain expertise** | Japanese receipt conventions (108% tax, era dates, payment slip formats) require domain knowledge DeepSeek-OCR lacks |
| **Determinism** | Our pipeline uses seed=42 with DeepSeek V3.2 API; DeepSeek-OCR uses temperature=0.0 but model behavior may still vary |
| **Test infrastructure** | We have 36 fixtures with ground truth; DeepSeek-OCR has generic benchmarks, not receipt-specific ones |

---

## 7. What We Can Learn

1. **NoRepeatNGram logits processing**: Their custom `NoRepeatNGramLogitsProcessor` (n-gram=30, window=90) with whitelist tokens for table markers is a clever technique. We could apply similar n-gram blocking if we ever switch to local model inference to prevent repetitive LLM output.

2. **Multi-resolution inference strategy**: The idea of offering different resolution/speed tradeoffs (Tiny/Small/Base/Large) could inspire a similar approach in our OCR stage -- perhaps lower-resolution first pass for confidence estimation, full resolution only when needed.

3. **Vision token compression**: Their approach of compressing images to 64-400 tokens while maintaining OCR accuracy is impressive. If we ever build a local OCR model, the DeepEncoder architecture (SAM + CLIP in series with convolutional downsampling) is worth studying.

4. **Document-to-markdown as intermediate representation**: Their markdown output format could be a useful intermediate step. Instead of raw OCR text, converting to markdown-structured text before LLM extraction might improve field boundary detection.

5. **vLLM production deployment patterns**: Their streaming + batching patterns in `run_dpsk_ocr_image.py` are well-engineered for production throughput.

---

## 8. Recommendation

**Not directly usable for our pipeline, but worth monitoring.**

DeepSeek-OCR solves a different problem (general document understanding) than ours (structured receipt parsing). The key gap is the lack of structured output -- it produces markdown, not typed JSON with receipt fields. We would still need an LLM extraction layer on top.

**Potential hybrid approach:** Use DeepSeek-OCR as an alternative OCR backend (replacing Google Cloud Vision) to get markdown-structured text, then feed that to our existing DeepSeek V3.2 extraction pipeline. This could:
- Eliminate Cloud Vision API costs
- Provide better layout-aware text (especially for multi-column receipts)
- But would require GPU infrastructure (A100-class) and add significant deployment complexity

**Wait for:** DeepSeek-OCR fine-tuning support or a receipt-specific prompt template that outputs structured JSON directly. That would make it much more compelling as a drop-in replacement for our OCR+LLM two-stage pipeline.
