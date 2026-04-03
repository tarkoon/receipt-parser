# Tencent-Hunyuan/HunyuanOCR

> https://github.com/Tencent-Hunyuan/HunyuanOCR

| Field | Value |
|-------|-------|
| Stars | ~1,600 |
| Language | Python |
| License | Custom (see License.txt) |
| Last Updated | 2026-04-02 |
| Approach | Lightweight 1B end-to-end VLM for OCR tasks |

---

## 1. Overview

HunyuanOCR is Tencent's specialized OCR Vision Language Model -- a 1-billion-parameter model built on Hunyuan's native multimodal architecture. It represents the "OCR expert VLM" approach: a compact model specifically trained for OCR tasks that outperforms both traditional OCR pipelines and general-purpose VLMs many times its size.

Key claim: 1B parameters achieves SOTA on text spotting, document parsing, and information extraction benchmarks, beating models 2-235x larger (including Qwen3-VL-235B and Gemini-2.5-Pro on OCR-specific tasks).

Released November 2025 with weights and inference code, with an online demo launched January 2026.

---

## 2. Architecture & How It Works

### Code Structure

```
HunyuanOCR/
  Hunyuan-OCR-master/
    Hunyuan-OCR-vllm/     # vLLM inference (recommended)
    Hunyuan-OCR-hf/       # HuggingFace Transformers inference
    utils.py              # Shared utilities
  assets/                 # Documentation images
  HunyuanOCR_Technical_Report.pdf
  requirements.txt
  README.md / README_zh.md
  License.txt
```

### Architecture

- **Type**: End-to-end Vision Language Model
- **Parameters**: 1 billion
- **Foundation**: Hunyuan native multimodal architecture
- **Approach**: Single unified model processes images directly -- no cascading pipeline, no separate detection/recognition stages
- **Max output**: 16,384 tokens per inference
- **Decoding**: Temperature=0 greedy decoding for deterministic output

### Inference Flow

1. **Image input** processed by `AutoProcessor` for image-text alignment
2. **Task-specific prompt** (in Chinese) tells the model what to do:
   - Text spotting: "Detect and recognize text, format coordinates"
   - Document parsing: "Parse this document" (tables -> HTML, formulas -> LaTeX, charts -> Mermaid)
   - Information extraction: "Extract these JSON fields: {keys}"
   - Subtitle extraction: "Extract subtitles from image"
   - Translation: "OCR + translate to {language}"
3. **Single-pass generation** via vLLM or Transformers
4. **Post-processing**: Repeated substring cleaning (handles decoder hallucination patterns)

### Deployment

**vLLM (recommended)**:
```bash
vllm serve tencent/HunyuanOCR \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --gpu-memory-utilization 0.2
```

**Requirements**: Python 3.12+, CUDA 12.9, PyTorch 2.7.1, 20GB GPU VRAM, 6GB disk for weights.

Note: vLLM produces better accuracy than Transformers framework (accuracy parity work ongoing).

---

## 3. Key Features

- **1B parameter efficiency**: SOTA OCR quality at a fraction of the compute cost of larger VLMs
- **Multi-task via prompts**: Single model handles text spotting, parsing, extraction, translation, QA -- just change the prompt
- **Coordinate output**: Text spotting returns bounding box coordinates, not just text
- **Format-aware parsing**: Tables -> HTML, formulas -> LaTeX, charts -> Mermaid/Markdown
- **Deterministic decoding**: Temperature=0 greedy generation
- **Hallucination mitigation**: Built-in repeated substring cleaning
- **Video subtitle extraction**: Handles multi-frame sequences
- **Photo translation**: OCR + translate in one pass

### Benchmark Results

**Text Spotting** (in-house benchmark):

| Scenario | HunyuanOCR | BaiduOCR | Qwen3VL-235B |
|----------|-----------|----------|--------------|
| Overall | 70.92% | 61.90% | 53.62% |
| Handwriting | 77.10% | - | - |
| Screen text | 76.58% | - | - |

**Document Parsing** (OmniDocBench):

| Metric | HunyuanOCR | Best Competitor |
|--------|-----------|-----------------|
| Edit Distance | 0.042 | 0.048 |
| Overall Score | 94.10% | ~92% |

**Information Extraction**:

| Document Type | HunyuanOCR | Gemini-2.5 |
|--------------|-----------|------------|
| Cards | 92.29% | 80.59% |
| Receipts | 92.53% | 80.66% |
| Video subtitles | 92.87% | 60.45% |

**Receipt extraction at 92.53% is particularly relevant to our project.**

---

## 4. Japanese Support

**Supported but not specifically highlighted.** HunyuanOCR claims "robust support for over 100 languages" and demonstrates strong performance on mixed-language documents. The model's multimodal training likely includes CJK data given Tencent's Chinese-language focus.

Key considerations for Japanese:
- The prompts in documentation are in Chinese, suggesting the model is optimized for Chinese-first, with other CJK languages benefiting from shared character representations
- No explicit Japanese benchmarks are published
- Chinese kanji and Japanese kanji share many characters, so transfer is strong
- Japanese-specific challenges (furigana, vertical text, era dates) are not addressed

Likely performance: Good on general Japanese text, but probably not as strong as Yomitoku's Japanese-specific models.

---

## 5. Strengths vs Our Project

| Area | HunyuanOCR Advantage |
|------|---------------------|
| **Receipt Extraction** | 92.53% accuracy on receipt extraction benchmark -- a directly comparable metric. |
| **Single-Pass Simplicity** | One model call: image -> structured JSON. No OCR stage, no LLM stage, no post-processing pipeline. |
| **Cost Efficiency** | 1B parameters runs on 20GB VRAM. Far cheaper than our GCV API + DeepSeek API dual costs. |
| **Deterministic by Default** | Temperature=0 greedy decoding. We had to explicitly configure seed=42. |
| **Coordinate Data** | Returns bounding boxes alongside text. GCV gives us this, but our LLM stage loses it. |
| **Hallucination Handling** | Built-in repeated substring cleaning. We handle hallucination in post-processing. |
| **Format-Aware Output** | Tables as HTML, formulas as LaTeX. Our pipeline outputs flat field values. |
| **No API Dependency** | Runs locally. No external service reliance. |

---

## 6. Weaknesses vs Our Project

| Area | Our Advantage |
|------|---------------|
| **Domain Specialization** | Receipt-specific merchant rules, tax categories, subset-sum matching, field registry. HunyuanOCR is a general OCR model. |
| **Japanese Specificity** | Era dates, yen formatting, furigana awareness, Japanese utility bill parsing. HunyuanOCR's Japanese support is inherited from multilingual training. |
| **Confidence Routing** | Per-field OCR vs LLM confidence routing. HunyuanOCR outputs a single result with no confidence gating. |
| **Multi-Pass Verification** | Our LLM verification pass. HunyuanOCR is single-pass. |
| **Post-Processing Logic** | Text normalization, field-specific cleanup, validation against Pydantic schema. HunyuanOCR relies on the model getting it right the first time. |
| **Testing Infrastructure** | 36 fixtures, ground truth, variance attribution benchmarks. HunyuanOCR has benchmark numbers but no receipt-specific test suite. |
| **Incremental Fixability** | We can add a post-processing rule to fix a specific error pattern. HunyuanOCR requires retraining or prompt engineering. |
| **Hardware Requirements** | We need only API access. HunyuanOCR needs 20GB VRAM GPU with CUDA 12.9. |
| **Maturity** | Released Nov 2025. Transformers framework accuracy still being improved. vLLM-specific. |
| **License** | Custom license (not a standard open-source license). Must review terms carefully. |

---

## 7. What We Can Learn

### 7.1 End-to-End VLM as Alternative Pipeline
HunyuanOCR's 92.53% receipt extraction accuracy from a single model call is compelling. We should benchmark it against our full pipeline:
- Send our 36 fixture images to HunyuanOCR with receipt extraction prompt
- Compare field-level accuracy against our OCR -> LLM -> post-processing pipeline
- If HunyuanOCR matches our accuracy, it dramatically simplifies deployment

### 7.2 Temperature=0 Greedy Decoding
HunyuanOCR uses temperature=0 for deterministic output by default. This aligns with our seed=42 approach but is cleaner -- greedy decoding is truly deterministic regardless of seed. We should consider whether temperature=0 (no sampling at all) is better than our temperature + seed approach for DeepSeek.

### 7.3 Repeated Substring Cleaning
HunyuanOCR has built-in handling for decoder hallucination patterns (repeated substrings). This is a known failure mode of autoregressive models. We should add similar detection to our LLM output parsing:
- Detect repeated substrings in LLM output
- Flag or clean them before validation
- Log as a specific error type in our benchmarks

### 7.4 Prompt-Based Task Switching
HunyuanOCR uses the same model for text spotting, parsing, extraction, translation, and QA -- just by changing the prompt. Our pipeline could adopt this pattern:
- Receipt extraction prompt for standard receipts
- Table extraction prompt for utility bills with tabular data
- QA prompt for edge cases where we need to ask specific questions about the document

### 7.5 Coordinate Preservation Through Pipeline
HunyuanOCR returns bounding boxes alongside extracted text. Our pipeline gets coordinates from GCV but loses them through the LLM stage. Preserving coordinates through the full pipeline would enable:
- Visual debugging (highlight which receipt region each field came from)
- Spatial validation (e.g., "total" should be near the bottom of the receipt)
- Multi-region extraction for complex layouts

### 7.6 Hybrid Architecture
The most promising approach: use HunyuanOCR as the OCR+extraction backbone, then apply our domain-specific post-processing:
```
Receipt Image
  -> HunyuanOCR (single-pass extraction to JSON)
    -> Our post-processing (merchant rules, tax categories, subset-sum, validation)
      -> Final structured output
```
This gives us HunyuanOCR's extraction quality + our domain expertise.

---

## 8. Recommendation

**Strong candidate for pipeline simplification. Benchmark immediately.**

HunyuanOCR is the most promising "replace our OCR+LLM stages with one model" candidate. At 92.53% receipt extraction accuracy with a 1B model, it could match or exceed our multi-stage pipeline while being simpler, faster, and cheaper (no API costs).

**Recommended action plan**:

1. **Benchmark** (this week): Run our 36 fixture images through HunyuanOCR's receipt extraction mode. Compare per-field accuracy against our pipeline.
2. **Evaluate Japanese**: Test specifically on Japanese receipts with era dates, vertical text, and yen formatting to assess CJK transfer quality.
3. **Prototype hybrid**: If extraction accuracy is close, prototype: HunyuanOCR extraction -> our post-processing/validation -> output. Measure end-to-end accuracy.
4. **Hardware decision**: 20GB VRAM / CUDA 12.9 requirement means we need a GPU. Evaluate cost vs GCV+DeepSeek API costs.

**Blockers**:
- **Custom license**: Must review `License.txt` carefully. Not a standard OSS license.
- **GPU requirement**: 20GB VRAM is a meaningful hardware investment
- **vLLM dependency**: Currently better accuracy with vLLM than Transformers. Locks us to specific inference framework.
- **Maturity**: Released 5 months ago. May have undiscovered issues.

HunyuanOCR doesn't replace our domain logic, but it could dramatically simplify the OCR + LLM extraction stages into a single model call, with our post-processing layer providing the receipt-specific intelligence.
