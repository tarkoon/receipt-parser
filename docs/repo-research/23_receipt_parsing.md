# 23. fahmiaziz98/receipt_parsing

**URL:** https://github.com/fahmiaziz98/receipt_parsing
**Stars:** 19 | **Forks:** 8 | **Language:** Jupyter Notebook 99.7%
**Created:** 2023-09-24 | **Last Updated:** 2026-03-12
**License:** None | **Commits:** 23
**Topics:** donut, flask, image-to-text, parsing, transformer

---

## 1. Overview

receipt_parsing is a project that uses the Donut (Document Understanding Transformer) model fine-tuned on the CORD-v2 dataset to extract structured data from receipt images. The key innovation is that Donut performs **end-to-end document understanding without OCR** -- it goes directly from image pixels to structured JSON output using a vision-encoder + text-decoder architecture, bypassing traditional OCR entirely.

The project includes a Flask web interface for inference, Jupyter notebooks for training/evaluation, and plans (stated in the description) to "add using LLM + OCR or VLM" approaches -- suggesting the author sees Donut as a baseline to compare against.

This is the most recently active of the five repos (last updated March 2026) and has meaningful community engagement (19 stars, 8 forks).

---

## 2. Architecture & How It Works

### Pipeline Flow

```
Receipt Image
  -> Donut Processor (image -> pixel_values tensor)
  -> Donut Encoder (VIT-based vision encoder)
  -> Donut Decoder (BART-based text decoder, with task prompt "<s_cord-v2>")
  -> Token sequence -> token2json() post-processing
  -> Structured JSON output
```

### Core Components

| File | Purpose |
|------|---------|
| `vision.py` | Core inference -- loads fine-tuned Donut model, processes images, generates JSON |
| `server.py` | Flask web app -- image upload endpoint, renders results via templates |
| `notebook/HyperParam_Donut.ipynb` | Training notebook -- hyperparameter tuning with WandB integration |
| `notebook/Quick_inference_with_DONUT_for_Document_Parsing.ipynb` | Inference demo notebook |
| `gemini-vision/` | Experimental Gemini Vision integration (PDF/image converters + pytesseract) |
| `templates/index.html` | Web UI for upload and results display |

### Donut Model Architecture

Donut (Document Understanding Transformer) is fundamentally different from our OCR + LLM approach:

1. **No OCR Step:** The model directly processes the image through a Vision Transformer (ViT) encoder. No separate OCR engine is needed.
2. **Task Prompt:** The decoder is prompted with `"<s_cord-v2>"` which tells it to output CORD-format structured data.
3. **Beam Search Decoding:** Uses beam search with bad_words_ids filtering to avoid generating unknown tokens.
4. **Token-to-JSON:** Post-processes the token sequence by removing special tokens and converting to nested JSON structure.

### Fine-Tuned Model

- **Base model:** `naver-clova-ix/donut-base`
- **Fine-tuned as:** `fahmiaziz/finetune-donut-cord-v2.5`
- **Dataset:** CORD-v2 (Consolidated Receipt Dataset) -- a large-scale receipt understanding benchmark
- **Performance:** >90% accuracy on test set
- **Tracking:** Weights & Biases (WandB) for training monitoring
- **Evaluation metric:** Tree Edit Distance (Tree ED) for structural accuracy

### Gemini Vision Directory

The `gemini-vision/` directory contains experimental code for an alternative approach:
- `extract_file.py` -- PDF/JPEG to image converters using pdf2image and pytesseract
- `gdrive/` -- Google Drive integration for loading documents
- This appears to be an in-progress exploration of VLM approaches (not Gemini itself despite the directory name)

---

## 3. Key Features

- **OCR-free document understanding** -- Donut eliminates the OCR step entirely, which removes an entire class of errors (OCR misreads, confidence thresholds, text ordering)
- **End-to-end trainable** -- the entire pipeline from image to JSON is differentiable and can be fine-tuned on domain-specific data
- **CORD-v2 fine-tuning** -- trained on a large, well-annotated receipt dataset with structured ground truth
- **WandB integration** -- proper ML experiment tracking for hyperparameter optimization
- **Tree Edit Distance evaluation** -- measures structural accuracy of JSON output, not just field-level F1
- **Flask web interface** -- simple but functional upload-and-parse interface
- **Active project** -- still being updated as of March 2026
- **YouTube demo** -- video demonstration of the parsing in action
- **Plans for VLM comparison** -- the author intends to add LLM + OCR and VLM approaches for comparison

---

## 4. Japanese Support

**Partial / Possible.** Donut's architecture is language-agnostic at the vision encoder level -- it processes pixel values, not text characters. However:

- **CORD-v2 dataset limitation:** CORD receipts are primarily Korean and English. The fine-tuned model has not been exposed to Japanese receipt layouts.
- **Decoder vocabulary:** The BART decoder's token vocabulary is trained on CORD data and may not include Japanese characters (kanji, katakana, hiragana).
- **Fine-tuning potential:** One could fine-tune Donut on a Japanese receipt dataset (if one existed) and it would theoretically learn Japanese receipt structures. The architecture itself does not preclude Japanese support.
- **No era date handling:** No concept of Japanese era dates, yen formatting, or tax category structures.

In short: the architecture could support Japanese with appropriate fine-tuning data, but this specific model does not.

---

## 5. Strengths vs Our Project

- **No OCR dependency:** Eliminating the OCR step removes Google Cloud Vision as a dependency and cost center. No OCR confidence issues, no text ordering problems, no bounding box logic needed.
- **End-to-end differentiable:** The entire pipeline can be optimized jointly. Our pipeline has multiple non-differentiable stages (OCR -> text normalization -> LLM -> post-processing) where errors compound.
- **Faster inference potential:** A single model forward pass is faster than OCR API call + LLM API call. No network latency for two separate services.
- **Structural evaluation:** Tree Edit Distance measures whether the output JSON structure is correct, not just individual field values. This is a more holistic metric than our field-level accuracy.
- **No prompt engineering needed:** The model learns the extraction task from training data, not from carefully crafted prompts. Less brittle than prompt-dependent systems.
- **Offline capability:** Once the model is downloaded, no API calls needed. Can run entirely on local GPU.

---

## 6. Weaknesses vs Our Project

- **Requires training data:** Fine-tuning needs annotated receipt images with ground truth JSON. Our LLM-based approach works zero-shot or with few-shot prompting, requiring no training data.
- **Fixed schema:** The CORD-v2 schema is rigid. Adding new fields requires retraining. Our pipeline can extract new fields by modifying the prompt and Pydantic schema.
- **No confidence routing:** No concept of per-field confidence. The model either generates the right token or does not. Our confidence routing system can selectively trust OCR vs LLM per field.
- **Domain-specific:** The fine-tuned model only handles CORD-like receipts. Our pipeline handles receipts, utility bills, and payment slips with the same infrastructure.
- **GPU required:** Donut inference needs a GPU. Our pipeline runs OCR in the cloud and LLM via API, requiring minimal local compute.
- **No post-processing validation:** No subset-sum tax matching, no arithmetic verification, no merchant rules. Raw model output only.
- **No Japanese training data:** CORD-v2 is Korean/English. Would need a Japanese receipt dataset to be useful for our use case.
- **Small-scale project:** 23 commits, no tests, no CI, no structured error handling.
- **Limited document variety:** Only handles standard retail receipts. Cannot parse utility bills, payment slips, or other document types we support.
- **Accuracy ceiling:** >90% on CORD test set sounds good but means ~10% error rate. Our pipeline achieves higher accuracy on our test set through multi-pass verification and post-processing.

---

## 7. What We Can Learn

1. **OCR-free as a future direction:** Donut and similar VDU (Visual Document Understanding) models represent the future of document parsing. As these models improve and multilingual training data becomes available, an OCR-free approach could replace our OCR + LLM pipeline with a single model. Worth monitoring the space (Donut, Pix2Struct, LayoutLMv3, Florence-2).

2. **Tree Edit Distance as an evaluation metric:** Their use of Tree ED to measure structural accuracy of JSON output is a better metric than field-level accuracy alone. We could implement Tree ED as an additional benchmark metric to catch structural errors (wrong nesting, missing arrays, etc.) that field-level checks miss.

3. **WandB for experiment tracking:** If we ever do model comparisons (e.g., DeepSeek V3 vs V3.2 vs Qwen), WandB integration would provide proper experiment tracking. Their setup shows this is straightforward with HuggingFace Transformers.

4. **Hybrid approach potential:** The author's stated plan to "add using LLM + OCR or VLM" suggests they see value in comparing approaches. We could experiment with using a VDU model as an additional extraction pass, comparing its output against our OCR + LLM output for disagreement detection.

---

## 8. Recommendation

**Do not adopt this tool directly.** The fine-tuned model is trained on Korean/English CORD data and would not work for Japanese receipts without retraining, which requires a Japanese receipt dataset we do not have. However, the approach is strategically important:

- **Monitor VDU model developments:** As models like Donut, Florence-2, and similar architectures improve multilingual support, they could eventually replace our OCR + LLM pipeline. Keep an eye on Japanese document understanding models.
- **Add Tree Edit Distance metric:** Implement Tree ED as a supplementary evaluation metric in our benchmark suite.
- **Consider as a comparison baseline:** If we ever build a Japanese receipt training dataset, fine-tuning a Donut variant would give us a useful comparison point against our LLM-based approach.
