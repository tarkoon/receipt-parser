# Receipt Parser Landscape: 25-Repo Research Summary

> Generated 2026-04-03 | Compared against our receipt-parser v3.0 pipeline
> (Google Cloud Vision OCR + DeepSeek V3.2 LLM + confidence routing + multi-pass verification)

---

## Executive Summary

After deep-diving into all 25 repositories, one thing is clear: **our pipeline is among the most sophisticated receipt parsing systems in the open-source ecosystem.** No single project matches our combination of Japanese-specific intelligence, confidence routing, multi-pass verification, and rigorous testing infrastructure.

However, the landscape is shifting fast. Vision-Language Models (VLMs) like HunyuanOCR, DeepSeek-OCR, and dots.ocr are collapsing the OCR+LLM two-stage pipeline into single-model solutions. Japanese-specialized tools like Yomitoku offer techniques we should adopt. And even small projects contribute patterns worth stealing.

### Top 5 Most Relevant Repos (for us)

| Rank | Repo | Why It Matters |
|------|------|---------------|
| 1 | **Yomitoku** (#4) | Japanese-first OCR with furigana filtering, dual confidence, reading order -- the closest peer to our Japanese focus |
| 2 | **HunyuanOCR** (#5) | 92.53% receipt extraction from a single 1B model call -- could replace our OCR+LLM stages |
| 3 | **PaddleOCR** (#1) | Free offline OCR replacement for Cloud Vision, plus layout/seal detection we lack |
| 4 | **CORD** (#10) | Gold-standard receipt annotation schema -- hierarchical labels, row grouping, key/value flags |
| 5 | **Well** (#12) | Per-field confidence wrapping in schema -- the LLM outputs confidence natively per field |

### Strategic Takeaway

The industry is moving toward **single-model VLM pipelines** (image-in, JSON-out). Our multi-stage approach (OCR -> normalize -> LLM -> post-process -> validate) provides better accuracy today, but the gap is closing. The smartest move is to **keep our domain-specific post-processing layer** while being ready to swap the OCR+LLM stages for a VLM when accuracy reaches parity.

---

## Ratings Table: All 25 Repositories

Ratings are on a 1-5 scale relative to our project's needs:
- **Relevance**: How applicable is this to our Japanese receipt parsing pipeline?
- **Quality**: Code quality, architecture, testing, documentation
- **Japanese**: How well does it handle Japanese text/receipts?
- **Learn**: How much can we learn/steal from this project?
- **Adopt**: Should we consider using or integrating this?

| # | Repository | Type | Stars | Relevance | Quality | Japanese | Learn | Adopt | Overall |
|---|-----------|------|-------|-----------|---------|----------|-------|-------|---------|
| 1 | **PaddlePaddle/PaddleOCR** | OCR Engine | 74.7k | 5 | 5 | 4 | 5 | 4 | **A** |
| 2 | **katanaml/sparrow** | Framework | 5.1k | 3 | 4 | 1 | 4 | 2 | **B** |
| 3 | **clovaai/donut** | ML Model | 6.8k | 3 | 4 | 3 | 3 | 2 | **B-** |
| 4 | **kotaro-kinoshita/yomitoku** | OCR Engine | 1.4k | 5 | 5 | 5 | 5 | 4 | **A+** |
| 5 | **Tencent-Hunyuan/HunyuanOCR** | VLM | 1.6k | 5 | 4 | 3 | 5 | 4 | **A** |
| 6 | **deepseek-ai/DeepSeek-OCR** | VLM | 22.8k | 3 | 4 | 2 | 3 | 2 | **B-** |
| 7 | **rednote-hilab/dots.ocr** | VLM | 1k+ | 3 | 4 | 2 | 3 | 2 | **B** |
| 8 | **bhimrazy/receipt-ocr** | Wrapper | 214 | 2 | 3 | 1 | 3 | 1 | **C+** |
| 9 | **Unstructured-IO/unstructured** | ETL | 14.4k | 2 | 5 | 2 | 4 | 2 | **B** |
| 10 | **clovaai/cord** | Dataset | 471 | 4 | 4 | 1 | 5 | 3 | **A-** |
| 11 | **zzzDavid/ICDAR-2019-SROIE** | Dataset | 413 | 2 | 3 | 1 | 2 | 1 | **C** |
| 12 | **WellApp-ai/Well** | Extractor | 317 | 3 | 4 | 1 | 4 | 2 | **B+** |
| 13 | **lisstasy/Receipt_Scanner** | Demo | 28 | 1 | 2 | 1 | 2 | 1 | **D+** |
| 14 | **ReceiptManager/receipt-parser-legacy** | Parser | 852 | 2 | 3 | 1 | 3 | 1 | **C+** |
| 15 | **knipknap/receiptparser** | Parser | 23 | 2 | 3 | 1 | 3 | 1 | **C+** |
| 16 | **YoshiRi/Receipt_OCR** | Parser | ~5 | 2 | 1 | 3 | 2 | 1 | **C-** |
| 17 | **yrarchi/household_accounts** | App | ~22 | 3 | 3 | 4 | 3 | 1 | **B-** |
| 18 | **JustCabaret/AIReceiptParser** | App | ~43 | 1 | 2 | 1 | 2 | 1 | **D+** |
| 19 | **GiulioLecci11/OCR_ReceiptScanner** | Demo | ~3 | 1 | 1 | 1 | 1 | 1 | **D** |
| 20 | **Asprise/receipt-ocr** | API Client | ~99 | 2 | 2 | 2 | 2 | 1 | **C** |
| 21 | **Martin36/ReceiptScanner** | Parser | 0 | 2 | 3 | 1 | 3 | 1 | **C** |
| 22 | **RecieptsParse/OCR_TO_JSON** | Academic | 1 | 1 | 2 | 1 | 2 | 1 | **D+** |
| 23 | **fahmiaziz98/receipt_parsing** | ML Model | 19 | 2 | 2 | 1 | 2 | 1 | **C-** |
| 24 | **ShafqaatMalik/llm-based-invoice-ocr** | App | 3 | 2 | 3 | 1 | 3 | 1 | **C** |
| 25 | **billstark/receipt-scanner** | ML OCR | 98 | 1 | 3 | 1 | 2 | 1 | **C-** |

### Grade Distribution
- **A+**: Yomitoku (1)
- **A**: PaddleOCR, HunyuanOCR (2)
- **A-**: CORD (1)
- **B+**: Well (1)
- **B**: Sparrow, dots.ocr, Unstructured (3)
- **B-**: Donut, DeepSeek-OCR, household_accounts (3)
- **C+**: receipt-ocr (bhimrazy), receipt-parser-legacy, receiptparser (3)
- **C**: SROIE, Asprise, ReceiptScanner (Martin), llm-based-invoice-ocr (4)
- **C-**: Receipt_OCR (YoshiRi), receipt_parsing (fahmi), receipt-scanner (billstark) (3)
- **D+**: Receipt_Scanner (lisstasy), AIReceiptParser, OCR_TO_JSON (3)
- **D**: OCR_ReceiptScanner (1)

---

## Category Breakdown

### Tier 1: Strategic -- Could Change Our Architecture

| Repo | Action | Priority | Effort |
|------|--------|----------|--------|
| **PaddleOCR** | Benchmark PP-OCRv5 vs Cloud Vision on our 36 fixtures. If competitive, use as offline OCR backend. | High | Medium |
| **Yomitoku** | Benchmark Japanese OCR accuracy. Adopt furigana filtering, dual confidence scoring, KV parser pattern. | High | Medium |
| **HunyuanOCR** | Benchmark receipt extraction on our fixtures. Could replace OCR+LLM with single model. | High | High |

### Tier 2: Tactical -- Specific Techniques to Adopt

| Repo | Technique | Priority |
|------|-----------|----------|
| **CORD** (#10) | Hierarchical label taxonomy, `is_key` flag, `row_id` grouping, separator detection | High |
| **Well** (#12) | Schema-level per-field confidence wrapping (`{value, confidence}` on every field) | High |
| **Sparrow** (#2) | Backend abstraction / factory pattern for LLM providers | Medium |
| **receipt-parser-legacy** (#14) | OCR spelling variant database for merchants (YAML config) | Medium |
| **receiptparser** (#15) | `is_complete()` + `merge()` dual-pass pattern | Medium |
| **household_accounts** (#17) | Levenshtein fuzzy matching, correction tracking, human-in-the-loop | Medium |
| **Unstructured** (#9) | Strategy routing (fast/hi_res/ocr_only), complexity detection | Low |
| **ReceiptScanner** (#21) | Spatial bounding-box heuristics (right-aligned = price), per-item math validation | Low |

### Tier 3: Reference Only -- Ideas Worth Knowing

| Repo | Idea |
|------|------|
| **DeepSeek-OCR** (#6) | NoRepeatNGram logits processor, multi-resolution inference |
| **dots.ocr** (#7) | Iterative error-case data refinement, NaviT high-res architecture |
| **Donut** (#3) | SynthDoG for synthetic Japanese document generation |
| **receipt-ocr** (#8) | Provider abstraction, pip-installable packaging, `json_schema` strict mode |
| **Receipt_Scanner** (#13) | Few-shot prompting with complete receipt examples |
| **OCR_TO_JSON** (#22) | FAISS + embeddings for semantic category matching |
| **receipt_parsing** (#23) | Tree Edit Distance as an evaluation metric |
| **llm-based-invoice-ocr** (#24) | Dual-mode (VLM vs OCR+regex) toggle, multi-page aggregation |
| **receipt-scanner** (#25) | Synthetic receipt generator with paper textures + augmentation |

### Tier 4: Not Useful -- Already Better Than Them

| Repo | Why We're Better |
|------|-----------------|
| **Receipt_OCR** (#16) | 2 commits, no LLM, no era dates, no validation |
| **AIReceiptParser** (#18) | Tesseract English-only, no validation, no tests |
| **OCR_ReceiptScanner** (#19) | 50-line proof-of-concept with hardcoded API key |
| **SROIE** (#11) | Archived 2019 dataset, English-only, pre-transformer |
| **Asprise** (#20) | Black-box commercial API, unknown Japanese quality |

---

## Where We Stand: Competitive Analysis

### What We Do Better Than Everyone

1. **Japanese receipt domain expertise** -- Era dates (Reiwa/Heisei), fullwidth normalization, tax category assignment (8%/10% subset-sum matching), Japanese utility bill parsing. No other project matches this depth.

2. **Confidence routing** -- Per-field OCR-vs-LLM confidence gating is a unique architectural innovation. Yomitoku has dual confidence (det/rec) but no cross-system routing. Well has per-field LLM confidence but no OCR confidence integration.

3. **Multi-pass LLM verification** -- Feed warnings back to the LLM for self-correction. Only Sparrow has comparable agent orchestration, but with no Japanese support.

4. **Rigorous testing infrastructure** -- 36 fixtures with ground truth, robustness benchmarks with variance attribution (OCR_VARIANCE vs LLM_VARIANCE vs POST_PROCESSING). No other project has this level of quality assurance.

5. **Post-processing sophistication** -- Merchant rules, tax category assignment via subset-sum, points detection, payment method inference, line item deduplication/expansion. These are hard-won domain rules.

### What Others Do Better

1. **Offline OCR** -- PaddleOCR, Yomitoku, HunyuanOCR all run locally. We depend on Cloud Vision API ($$$).

2. **Layout understanding** -- PaddleOCR PP-StructureV3, Unstructured, dots.ocr all provide spatial document structure. We treat receipts as flat text.

3. **Single-model simplicity** -- HunyuanOCR gets 92.53% receipt accuracy in one model call. We need OCR + normalization + LLM + post-processing + validation.

4. **Furigana filtering** -- Yomitoku's histogram-based ruby text detection. We have no awareness of furigana.

5. **Config-driven extensibility** -- receipt-parser-legacy's YAML locale configs. Our Japanese rules are hardcoded in Python.

6. **UI/API layer** -- Several projects (Sparrow, receipt-ocr, AIReceiptParser) have web UIs or REST APIs. We're CLI-only.

---

## Recommended Action Plan

### Immediate (This Week)

1. **Benchmark HunyuanOCR** on our 36 fixtures -- if it matches our accuracy, it radically simplifies the pipeline
2. **Adopt per-field confidence in schema** from Well (#12) -- wrap Pydantic fields with `{value, confidence}`

### Short-term (This Month)

3. **Benchmark Yomitoku OCR vs Cloud Vision** on Japanese fixtures
4. **Add furigana filtering** inspired by Yomitoku's histogram-based approach
5. **Build OCR spelling variant database** for Japanese merchants (YAML config, from #14)
6. **Add separator detection** from CORD's `repeating_symbol` pattern
7. **Add per-item arithmetic validation** (`qty * unit_price == line_total`) from #21

### Medium-term (Next Quarter)

8. **Evaluate PaddleOCR PP-OCRv5** as offline OCR backend
9. **Implement strategy routing** (fast/hi_res modes) from Unstructured
10. **Add LLM provider abstraction** (factory pattern from Sparrow/receipt-ocr)
11. **Build dual-pass `is_complete()` + `merge()` pattern** from receiptparser
12. **Add Tree Edit Distance** as benchmark metric from #23

### Long-term (Future)

13. **Prepare for VLM transition** -- keep post-processing layer, be ready to swap OCR+LLM for a single VLM
14. **Build a Japanese CORD** -- properly annotated Japanese receipt dataset following CORD conventions
15. **Synthetic receipt generator** for expanding test fixtures (inspired by #25, SynthDoG from #3)
16. **Web UI with human-in-the-loop correction** (inspired by #17)

---

## Project Type Distribution

```
OCR Engines/VLMs:     6 (#1, #4, #5, #6, #7, #25)
Frameworks/ETL:       3 (#2, #9, #12)
Receipt Parsers:      7 (#8, #14, #15, #16, #17, #21, #24)
ML Models:            3 (#3, #23, #25)
Datasets:             2 (#10, #11)
LLM Wrappers/Demos:   4 (#13, #18, #19, #22)
API Clients:          1 (#20)
```

## Approach Distribution

```
OCR + Rules (classic):       5 (#14, #15, #16, #17, #21)
OCR + LLM (our approach):    6 (#8, #13, #18, #19, #22, #24)
Vision LLM (no OCR):         5 (#2, #5, #6, #7, #24-paid)
OCR-free Transformer:        2 (#3, #23)
Full ML OCR:                 1 (#25)
Commercial API:              1 (#20)
Dataset only:                2 (#10, #11)
Multi-approach framework:    1 (#9)
```

The OCR+LLM approach we use is the current mainstream, but Vision LLMs are the clear future direction.

---

## Appendix: Individual Research Files

All detailed analysis files are in `docs/repo-research/`:

| File | Repo |
|------|------|
| [01_paddleocr.md](01_paddleocr.md) | PaddlePaddle/PaddleOCR |
| [02_sparrow.md](02_sparrow.md) | katanaml/sparrow |
| [03_donut.md](03_donut.md) | clovaai/donut |
| [04_yomitoku.md](04_yomitoku.md) | kotaro-kinoshita/yomitoku |
| [05_hunyuan_ocr.md](05_hunyuan_ocr.md) | Tencent-Hunyuan/HunyuanOCR |
| [06_deepseek_ocr.md](06_deepseek_ocr.md) | deepseek-ai/DeepSeek-OCR |
| [07_dots_ocr.md](07_dots_ocr.md) | rednote-hilab/dots.ocr |
| [08_receipt_ocr.md](08_receipt_ocr.md) | bhimrazy/receipt-ocr |
| [09_unstructured.md](09_unstructured.md) | Unstructured-IO/unstructured |
| [10_cord.md](10_cord.md) | clovaai/cord |
| [11_ICDAR-2019-SROIE.md](11_ICDAR-2019-SROIE.md) | zzzDavid/ICDAR-2019-SROIE |
| [12_WellApp-Well.md](12_WellApp-Well.md) | WellApp-ai/Well |
| [13_Receipt_Scanner.md](13_Receipt_Scanner.md) | lisstasy/Receipt_Scanner |
| [14_receipt-parser-legacy.md](14_receipt-parser-legacy.md) | ReceiptManager/receipt-parser-legacy |
| [15_receiptparser.md](15_receiptparser.md) | knipknap/receiptparser |
| [16_receipt_ocr_yoshiri.md](16_receipt_ocr_yoshiri.md) | YoshiRi/Receipt_OCR |
| [17_household_accounts.md](17_household_accounts.md) | yrarchi/household_accounts |
| [18_aireceipt_parser.md](18_aireceipt_parser.md) | JustCabaret/AIReceiptParser |
| [19_ocr_receiptscanner.md](19_ocr_receiptscanner.md) | GiulioLecci11/OCR_ReceiptScanner |
| [20_asprise_receipt_ocr.md](20_asprise_receipt_ocr.md) | Asprise/receipt-ocr |
| [21_ReceiptScanner.md](21_ReceiptScanner.md) | Martin36/ReceiptScanner |
| [22_OCR_TO_JSON.md](22_OCR_TO_JSON.md) | RecieptsParse/OCR_TO_JSON |
| [23_receipt_parsing.md](23_receipt_parsing.md) | fahmiaziz98/receipt_parsing |
| [24_llm-based-invoice-ocr.md](24_llm-based-invoice-ocr.md) | ShafqaatMalik/llm-based-invoice-ocr |
| [25_receipt-scanner.md](25_receipt-scanner.md) | billstark/receipt-scanner |
