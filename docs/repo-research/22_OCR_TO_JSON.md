# 22. RecieptsParse/OCR_TO_JSON

**URL:** https://github.com/RecieptsParse/OCR_TO_JSON
**Stars:** 1 | **Forks:** 0 | **Language:** Python 99.6%, Shell 0.4%
**Created:** 2023-11-18 | **Last Updated:** 2024-10-04
**License:** None | **Commits:** 171
**Team:** 4 students (Jeremiah Dy, Kylie Higashionna, Grayson Levy, Amanda Nitta)
**Context:** Fall 2023 Big Data Analytics course project under Dr. Mahdi Belcaid

---

## 1. Overview

OCR_TO_JSON is a university course project that transforms OCR text from receipt images into structured JSON using GPT-4, then classifies vendors and products using KNN with FAISS vector search and Jina embeddings. The project aims to "uncover the nuances of the types of items that people buy" through receipt analysis. It is a two-stage pipeline: LLM-based JSON conversion followed by embedding-based classification.

Despite 171 commits (active student development), it has only 1 star and is essentially a completed course project.

---

## 2. Architecture & How It Works

### Pipeline Flow

```
OCR .txt files (pre-extracted text, not raw images)
  -> Stage 1: LangChain + GPT-4 (OCR text -> structured JSON)
     -> Pydantic schema validation (Item, ReceiptInfo models)
     -> Few-shot prompt with 7 example receipts
  -> Stage 2: KNN Classification
     -> Jina Embeddings v2 (text -> 768d vectors)
     -> FAISS IndexFlatL2 (nearest neighbor search)
     -> Vendor classification (k=5) + Product classification (k=10)
  -> Output: Classified JSON
```

### Core Components

| File | Purpose |
|------|---------|
| `master.py` | Pipeline orchestrator -- database setup, chain creation, batch processing |
| `convert.py` | LangChain chain builder -- Pydantic models, few-shot prompt, GPT-4 integration |
| `classification.py` | KNN wrapper -- queries FAISS for vendor and product categories |
| `search.py` | FAISS + Jina embeddings classifier -- encodes queries, majority voting |
| `prompt_examples.py` | 7 few-shot examples mapping OCR text to structured JSON |
| `config.py` | Model config (Jina embeddings v2), 6 vendor categories, 15 product categories |
| `vendor_database.py` | Builds FAISS index from vendor category descriptions |
| `product_database.py` | Builds FAISS index from product category descriptions |

### LLM Integration (LangChain + GPT-4)

The `convert.py` module uses LangChain's structured output pipeline:

1. **Pydantic Models:** `Item` (price, quantity, description) and `ReceiptInfo` (merchant, transaction, items list) define the output schema
2. **Field Validators:** Auto-convert strings to floats for prices, ceiling-round decimal quantities, strip "UNKNOWN" markers, classify payment types (credit/debit/cash)
3. **Few-Shot Prompt:** 7 hand-crafted examples teach GPT-4 the OCR-to-JSON conversion pattern
4. **Chain Assembly:** `make_chain()` connects prompt template -> GPT model -> Pydantic parser

### KNN Classification (FAISS + Jina)

The classification stage is the most architecturally interesting part:

1. **Database Creation:** Vendor/product category descriptions are encoded via `jinaai/jina-embeddings-v2-base-en` (768d) and stored in FAISS `IndexFlatL2`
2. **Query:** Receipt text (merchant name + item descriptions) is encoded with the same model
3. **Search:** FAISS finds k nearest neighbors in the category embedding space
4. **Voting:** `Counter(results).most_common(1)` selects the majority category

---

## 3. Key Features

- **LangChain-based structured extraction** with Pydantic schema validation -- the chain ensures type-safe output
- **Few-shot learning with 7 curated examples** -- demonstrates effective prompt engineering for receipt parsing
- **Embedding-based classification** using FAISS + Jina -- semantically categorizes vendors and products without keyword matching
- **Two-stage pipeline** separating extraction (LLM) from classification (embeddings) -- good separation of concerns
- **Large OCR dataset:** 100+ pre-extracted receipt text files organized by alphanumeric prefix
- **NER evaluation data:** Golden annotations in `data/receipts/ner_evaluate/annotations/` for evaluation
- **Visualization companion:** Separate `RecieptsParse/visualization` repo for data analysis

---

## 4. Japanese Support

**None.** The system processes English OCR text exclusively. The Jina embedding model (`jina-embeddings-v2-base-en`) is English-focused. Category taxonomies are English-only (Groceries, Restaurants, Clothing, etc.). The few-shot examples are all English receipts. No CJK text handling whatsoever.

---

## 5. Strengths vs Our Project

- **Embedding-based classification:** Their FAISS + Jina approach for vendor/product categorization is semantically richer than keyword matching. It can handle novel vendor names and product descriptions by finding semantically similar categories, rather than requiring exact regex matches.
- **LangChain structured output:** Their use of LangChain's Pydantic output parser with field validators ensures type-safe JSON extraction. The auto-conversion validators (string-to-float, decimal-to-int rounding) handle edge cases elegantly.
- **Few-shot prompt engineering:** 7 curated OCR-to-JSON examples provide the model strong in-context learning. This is a well-proven approach for extraction tasks.
- **Separation of extraction and classification:** Their two-stage architecture cleanly separates "what's on the receipt" (LLM extraction) from "what category does it belong to" (embedding classification). This modularity is good engineering.
- **NER evaluation dataset:** They have golden annotations for evaluation, suggesting systematic accuracy measurement.

---

## 6. Weaknesses vs Our Project

- **No OCR pipeline:** Assumes pre-extracted text files. No image-to-text capability. Our pipeline handles the full image -> OCR -> extraction flow.
- **No confidence scoring:** No OCR confidence, no LLM confidence, no confidence routing. Just single-shot extraction.
- **No multi-pass verification:** One LLM call, no retry logic. Our pipeline has confidence-gated OCR retry and multi-pass LLM verification.
- **GPT-4 cost:** Uses OpenAI GPT-4 which is expensive. Our DeepSeek V3.2 is far cheaper for comparable extraction quality.
- **API key hardcoded:** "Insert OpenAI API key into line 17 of master.py" -- no environment variable handling.
- **No post-processing:** No tax validation, no subset-sum matching, no arithmetic checks. Raw LLM output goes directly to classification.
- **No determinism controls:** No temperature/seed settings documented. Our `seed=42` approach ensures reproducible results.
- **Course project quality:** No error handling, no logging, no configuration management. The codebase reflects a semester project, not production software.
- **English-only:** No internationalization. Our Japanese support is a major differentiator.
- **Stale LangChain:** Uses 2023-era LangChain which has undergone major breaking changes since then.

---

## 7. What We Can Learn

1. **Embedding-based category assignment:** Their FAISS + Jina approach for vendor/product categorization is worth considering. Instead of our regex-based merchant rules, we could embed merchant names and find nearest-neighbor matches in a pre-built category database. This would handle spelling variations and OCR errors more gracefully than exact string matching.

2. **Pydantic field validators for edge cases:** Their auto-conversion validators in the Pydantic model (string prices to floats, rounding decimal quantities) are a clean pattern. We could add similar `@field_validator` decorators to our Pydantic schemas to handle common LLM output quirks at the schema level rather than in post-processing.

3. **Few-shot example curation:** Their 7 carefully crafted OCR-to-JSON examples demonstrate that a small number of high-quality examples can be very effective. We could curate a set of Japanese receipt examples for our LLM prompt to improve extraction on challenging formats.

4. **Two-stage extraction + classification:** The clean separation between "extract structured data" and "classify into categories" is a good architectural pattern. We could more explicitly separate our tax category assignment from the initial field extraction.

---

## 8. Recommendation

**Do not adopt this tool.** It is a student course project with no OCR capability, no Japanese support, and no production-quality engineering. However, two techniques are worth investigating:

- **Embedding-based merchant/category matching:** Consider using sentence embeddings + FAISS for fuzzy merchant name matching and product categorization, as a complement to our rule-based approach. This would be more robust to OCR errors in merchant names.
- **Pydantic field validators:** Add more `@field_validator` decorators to our output schemas to catch and auto-correct common LLM output issues at the schema level.
