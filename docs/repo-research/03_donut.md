# clovaai/donut

> https://github.com/clovaai/donut

| Field | Value |
|-------|-------|
| Stars | ~6,800 |
| Language | Python (PyTorch) |
| License | MIT |
| Last Updated | 2026-04-01 |
| Approach | OCR-free end-to-end document understanding transformer |

---

## 1. Overview

Donut (Document Understanding Transformer) is a research project from Clova AI (Naver/LINE) that pioneered the "OCR-free" approach to document understanding. Instead of OCR -> text -> NLP, Donut takes a document image and directly outputs structured JSON using an encoder-decoder transformer. Published at ECCV 2022.

The key innovation: the model learns to read, understand layout, and extract structured data in a single end-to-end pass, without any OCR component. This eliminates OCR error propagation and simplifies the pipeline dramatically.

Donut also includes **SynthDoG**, a synthetic document generator for creating multilingual pretraining data (English, Chinese, Japanese, Korean -- 0.5M images per language).

---

## 2. Architecture & How It Works

### Code Structure

```
donut/
  __init__.py
  _version.py
  model.py        # Core model: SwinEncoder + BARTDecoder + DonutModel
  util.py          # Data loading, training utilities
config/            # Training configurations
synthdog/          # Synthetic document generator
train.py           # PyTorch Lightning training
lightning_module.py # LightningModule wrapper
test.py            # Evaluation
app.py             # Demo application
```

### Model Architecture

**Encoder: `SwinEncoder`** (Swin Transformer)
- Input: Document image (2560x1920 for base model)
- Preprocessing: RGB conversion, conditional 90-degree rotation (if image orientation mismatches canvas), resize, symmetric padding, ImageNet normalization
- Backbone: `swin_base_patch4_window12_384` from timm
- Adaptive position bias interpolation for variable window sizes
- Output: Visual feature map

**Decoder: `BARTDecoder`** (Modified Multilingual BART)
- Base: `MBartForCausalLM` from HuggingFace Transformers
- Tokenizer: `XLMRobertaTokenizer` (multilingual, supports Japanese)
- Special tokens: `<sep/>` for JSON lists, custom `<s_key>` tokens for structured output fields
- Resizable position embeddings for variable sequence lengths
- Autoregressive generation of JSON token sequences

**Combined: `DonutModel`**
- `json2token()`: Converts ground truth JSON objects into special token sequences for training
- `token2json()`: Parses generated token sequences back into nested JSON during inference
- `inference()`: Full generation pipeline with optional attention visualization

### Data Flow

```
Document Image
  -> SwinEncoder (visual features)
    -> BARTDecoder (autoregressive JSON generation)
      -> token2json() (parse tokens into structured JSON)
        -> Structured Output
```

### Training

- Pretrained on IIT-CDIP (11M document images) + SynthDoG multilingual synthetic data
- Fine-tuned per task (receipt parsing, classification, VQA)
- Uses PyTorch Lightning for training orchestration
- 64 A100 GPUs for base model pretraining

---

## 3. Key Features

- **Truly OCR-free**: No text detection, no text recognition, no OCR at all. Image -> JSON directly.
- **Unified framework**: Same model architecture handles classification, parsing, and VQA
- **SynthDoG**: Synthetic document generator creates pretraining data in 4 languages (EN, ZH, JA, KO)
- **JSON as task formulation**: All tasks are framed as "generate the right JSON from this image"
- **Multilingual tokenizer**: XLMRobertaTokenizer handles CJK characters natively
- **Attention visualization**: Can show what parts of the image the model focuses on per token

### Benchmark Results

| Task | Dataset | Accuracy |
|------|---------|----------|
| Receipt Parsing | CORD | 91.3% |
| Document Classification | RVL-CDIP | 95.3% |
| Document VQA | DocVQA | 67.5% |
| Chinese Tickets | Train Ticket | 98.7% |

---

## 4. Japanese Support

**Explicitly supported and well-integrated.** Japanese is one of the four pretraining languages:

- SynthDoG generates 0.5M synthetic Japanese document images for pretraining
- `XLMRobertaTokenizer` handles Japanese characters (kanji, hiragana, katakana)
- The model was pretrained on Japanese alongside English, Chinese, and Korean
- No separate Japanese model needed -- the base model is multilingual

However, Donut was trained on general documents, not specifically on Japanese receipts. Receipt-specific patterns (era dates, tax breakdowns, hanko stamps) are not part of the training data. Fine-tuning on Japanese receipts would be required.

---

## 5. Strengths vs Our Project

| Area | Donut Advantage |
|------|-----------------|
| **Simplicity** | One model, one pass, image to JSON. No OCR stage, no text normalization, no LLM prompt engineering. Our pipeline has 5+ stages. |
| **No OCR Error Propagation** | OCR errors (misread kanji, missed characters) cannot occur because there is no OCR. |
| **No API Dependencies** | Runs fully locally with PyTorch. No GCV API, no DeepSeek API. |
| **Unified Architecture** | Same model does classification + extraction + VQA. We have separate components for each concern. |
| **SynthDoG** | Can generate unlimited synthetic training data in Japanese. We have no data augmentation capability. |
| **Attention Interpretability** | Can visualize what image regions drive each output token. We have no interpretability. |

---

## 6. Weaknesses vs Our Project

| Area | Our Advantage |
|------|---------------|
| **Domain Accuracy** | 91.3% on CORD (general receipts) vs our accuracy on Japanese-specific receipt fixtures. Domain-specific pipeline beats generic model. |
| **Post-Processing** | We have merchant rules, tax category assignment, subset-sum matching, field registry. Donut outputs raw JSON with no domain logic. |
| **Confidence Routing** | Our per-field OCR vs LLM confidence routing. Donut has no confidence-gated output. |
| **Incremental Improvement** | We can fix specific extraction errors with post-processing rules. Donut requires retraining. |
| **No Fine-Tuning Required** | Our pipeline works with zero training data. Donut requires fine-tuning per task. |
| **Determinism & Testing** | seed=42, 36 fixtures, variance attribution. Donut's research focus doesn't emphasize production determinism. |
| **Compute Requirements** | Donut base was trained on 64 A100 GPUs. Our pipeline runs on any machine with API access. |
| **Maintenance** | Research project, last meaningful model update circa 2022. Our pipeline is actively maintained. |
| **Multi-Pass Verification** | We verify LLM outputs with a second pass. Donut's single-pass has no self-correction. |

---

## 7. What We Can Learn

### 7.1 End-to-End Evaluation Baseline
Donut provides a clean baseline for "how well does a single-model approach work on our receipts?" We could:
1. Fine-tune `donut-base` on our 36 receipt fixtures (or a larger training set)
2. Compare its accuracy against our full OCR->LLM->post-processing pipeline
3. If Donut matches our pipeline accuracy, that suggests our pipeline complexity may not be justified

### 7.2 SynthDoG for Data Augmentation
SynthDoG's synthetic Japanese document generation could be invaluable:
- Generate thousands of synthetic Japanese receipts with known ground truth
- Use for training/fine-tuning any model (not just Donut)
- Expand our test fixture set without manual labeling

### 7.3 JSON-as-Task Formulation
Donut's insight that all document understanding tasks can be framed as "generate the right JSON" aligns with our LLM extraction approach. The difference is Donut does it end-to-end while we do it in stages. This validates our direction.

### 7.4 Attention Visualization for Debugging
Donut's attention maps show which image regions drive each output field. We could add similar visualization (e.g., highlighting which OCR text spans the LLM used for each field) to debug extraction errors.

### 7.5 `token2json()` / `json2token()` Pattern
Donut's structured output parsing (special tokens -> nested JSON) is a clean implementation pattern. Our LLM output parsing could adopt a similar approach with defined delimiters for nested structures.

---

## 8. Recommendation

**Use as a research baseline and steal SynthDoG, but don't adopt for production.**

Donut is conceptually important -- it proved OCR-free document understanding is viable -- but it's a 2022 research project that has been surpassed by newer VLMs (Qwen-VL, PaddleOCR-VL, HunyuanOCR). The model itself is not competitive with modern alternatives.

Specific actions:
1. **SynthDoG** is the most valuable component. Use it to generate synthetic Japanese receipt training data.
2. **Benchmark**: Fine-tune donut-base on Japanese receipts as a baseline to quantify the value of our multi-stage pipeline.
3. **Don't adopt**: The model is outdated, compute-heavy to fine-tune, and lacks the domain-specific logic we've built.

The real successor to Donut's ideas is the Vision LLM approach (covered in other reviews). Modern VLMs do what Donut does but better, with larger training data and stronger multilingual support.

**License**: MIT -- fully permissive, no restrictions.
