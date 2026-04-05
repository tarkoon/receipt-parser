# Potential Next Steps: Benchmarking Candidates

Projects to benchmark against our current Google Cloud Vision + DeepSeek V3.2 pipeline on our 36 test fixtures.

## OCR Replacement Candidates (vs Google Cloud Vision)

- **PaddleOCR PP-OCRv5** -- 100+ language single-model OCR, runs offline, zero API cost.
  - License: **Apache-2.0** -- Full commercial use, no restrictions. **CLEAR**

- **Yomitoku** -- Japanese-only OCR with 4 custom models (7,000+ kanji), furigana filtering, dual confidence scoring.
  - License: **CC BY-NC-SA 4.0** -- Commercial use **PROHIBITED** under open-source license. Must purchase a separate commercial license via [mlism.com](https://www.mlism.com/) (on-premise) or AWS Marketplace ("YomiToku-Pro Document Analyzer"). **PAID LICENSE REQUIRED**

- **dots.ocr / dots.mocr** -- 3B param VLM (1.7B LLM + 1.2B vision encoder), layout-aware text output with bounding boxes. ~4GB VRAM.
  - License: **Custom (dots.ocr LICENSE AGREEMENT)** -- Commercial use allowed. Must retain attribution, follow acceptable use policy (no fraud, privacy violations, unauthorized copyright digitization). No revenue caps or user thresholds. **CLEAR**

## OCR + LLM Replacement Candidates (vs GCV + DeepSeek combined)

- **HunyuanOCR** -- 1B VLM, single-pass image-to-JSON. 92.53% on receipt extraction benchmarks. 20GB VRAM, CUDA 12.9.
  - License: **Tencent Hunyuan Community License** -- Free commercial use **only if <100M MAU**. Above that, must negotiate with Tencent. **Cannot use outputs to train competing AI models.** Excluded territories: **EU, UK, South Korea**. Must disclose Tencent as provider. **CONDITIONAL -- territory and training restrictions**

- **DeepSeek-OCR** -- End-to-end VLM (380M encoder + 570M decoder), outputs markdown not JSON. Would still need our LLM extraction layer on top.
  - License: **MIT** -- Full commercial use, no restrictions. **CLEAR**

- **PaddleOCR-VL-1.5** -- 0.9B VLM achieving 94.5% on OmniDocBench, handles skewed/warped documents. Part of PaddleOCR ecosystem.
  - License: **Apache-2.0** -- Full commercial use, no restrictions. **CLEAR**

## Pipeline Improvements (no model swaps needed)

### Schema
- **Per-field confidence wrapping** -- Modify Pydantic schema so the LLM outputs `{value, confidence}` for every field natively, instead of deriving confidence post-hoc. The LLM rates its own certainty per field at extraction time. *(Inspired by Well #12)*

### Pre-Processing
- **Furigana filtering** -- Histogram-based detection of small ruby text (furigana annotations) in OCR output. These contaminate extracted values when store names or items have reading annotations above them. Filter before sending to LLM. *(Inspired by Yomitoku #4)*
- **Separator detection** -- Regex to identify `===`, `---`, `...` lines in OCR text as receipt section boundaries (header / line items / totals / footer). Tag sections before LLM extraction to reduce hallucination. *(Inspired by CORD #10)*
- **Spatial position tagging** -- Use Cloud Vision bounding box coordinates to tag OCR text with position hints (e.g., right-aligned text = likely a price, top-of-page large text = likely store name, bottom region = likely totals). Feed tags alongside text to the LLM. *(Inspired by ReceiptScanner #21, Receipt_OCR #16)*

### Validation
- **Per-item arithmetic checks** -- After extraction, verify `qty * unit_price - discount == line_total` for every line item. Flag mismatches as warnings for multi-pass correction. *(Inspired by ReceiptScanner #21)*
- **Tax ratio cross-check** -- Quick sanity check: does `subtotal * 1.08 ≈ total` or `subtotal * 1.10 ≈ total`? Catches gross total/subtotal extraction errors before detailed subset-sum matching runs. *(Inspired by Receipt_OCR #16)*

### Prompt Engineering
- **Few-shot examples in prompt** -- Add 2-3 complete Japanese receipt examples (OCR text input -> expected JSON output) from our test fixtures into the DeepSeek prompt. Few-shot learning is proven to improve extraction consistency for structured output tasks. *(Inspired by Receipt_Scanner #13, OCR_TO_JSON #22)*

### Testing & Quality
- **Tree Edit Distance metric** -- Add Tree ED as a benchmark metric alongside field-level accuracy. Measures structural correctness of the entire JSON output (nesting, arrays, missing keys), catching errors that per-field checks miss. *(Inspired by receipt_parsing #23)*
- **Correction tracking for training signals** -- When we build a UI, log every user correction (original value -> corrected value) to a structured file. Over time this creates a dataset of pipeline failure modes that can inform prompt tuning, post-processing rules, or future fine-tuning. *(Inspired by household_accounts #17)*
