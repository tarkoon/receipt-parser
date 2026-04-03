# 13. lisstasy/Receipt_Scanner

## Overview

| Field | Value |
|---|---|
| **Repository** | [lisstasy/Receipt_Scanner](https://github.com/lisstasy/Receipt_Scanner) |
| **Stars** | 28 |
| **Forks** | 2 |
| **Language** | Jupyter Notebook (97.5%), Python (2.5%) |
| **License** | None |
| **Created** | 2024-04-18 |
| **Last Push** | 2024-05-20 (inactive) |
| **Approach** | PaddleOCR + GPT-3.5-turbo + LangChain few-shot + Pydantic + Gradio UI |

A small demo project that processes Spanish-language supermarket receipts using PaddleOCR for text extraction and GPT-3.5-turbo (via LangChain) for structured data extraction. Deployed on Hugging Face Spaces. Most of the code lives in `code.py` (pipeline) and `app.py` (Gradio interface).

## Architecture & How It Works

### Pipeline (code.py)

```
Receipt Image(s)
    |
    v
[PaddleOCR - PP-OCRv4, Spanish]
    |
    v
Raw OCR text (concatenated)
    |
    v
[LangChain FewShotChatMessagePromptTemplate]
    |  - System: "POS receipt data expert"
    |  - 3 complete example receipts (few-shot)
    |  - User: raw OCR text
    v
[GPT-3.5-turbo-0125, json_mode]
    |
    v
[Pydantic validation: ReceiptInfo schema]
    |
    v
[Pandas DataFrame transformation]
    |  - Date parsing (9 formats)
    |  - Numeric coercion
    |  - Category validation
    v
[Session CSV storage + Plotly visualizations]
```

### Pydantic Schema

```python
class ProductCategory(Enum):
    # 19 categories: fruits, vegetables, protein_foods, seafood,
    # dairy, grains, nuts_and_seeds, sweets, spices, beverages,
    # snacks, condiments, frozen_foods, bakery, canned_goods,
    # household, personal_care, pet_supplies, other

class ItemInfo(BaseModel):
    name: str
    unit_quantity: float
    unit_price: float
    total_amount: float
    category: ProductCategory

class ReceiptInfo(BaseModel):
    store_name: str
    store_address: str
    store_city: str
    store_phone: str
    receipt_no: str
    date: str
    time: str
    items: list[ItemInfo]
    total_amount: float
    item_count: int
    payment_method: Literal["tarjeta", "efectivo"]
```

### LLM Integration

- Uses `ChatOpenAI(model="gpt-3.5-turbo-0125")` from `langchain_openai`
- Structured output via `model.with_structured_output(ReceiptInfo, method="json_mode")`
- **Few-shot learning**: 3 complete receipt examples hardcoded as (input OCR text, expected JSON output) pairs
- Prompt chain: `final_prompt_cat | structured_llm` (LangChain LCEL pipe)

### UI (app.py)

- Gradio Blocks interface with emerald theme
- Multi-file image upload with gallery preview
- Editable 17-column dataframe for manual corrections
- 6 Plotly visualizations (donut chart, stacked bars, pie charts, box plots, line charts)
- Per-session CSV persistence (`user_data/{uuid}.csv`)

## Key Features

1. **Few-shot prompting with complete examples**: 3 full receipt input/output pairs teach the model the expected format
2. **LangChain structured output**: Uses `with_structured_output()` to enforce Pydantic schema
3. **Automatic category classification**: LLM assigns one of 19 product categories per item
4. **9-format date parser**: `parse_dates()` tries 9 different date formats before falling back to pandas inference
5. **Interactive data editing**: Users can correct extraction errors in the Gradio UI before confirming
6. **Expense visualization suite**: 6 different chart types for spending analysis
7. **Session management**: UUID-based isolation with cumulative CSV history

## Japanese Support

**None.** PaddleOCR is configured for Spanish (`lang='es'`), the few-shot examples are in Spanish, and the product categories use English names. The payment method enum is Spanish (`tarjeta`/`efectivo`). Would need complete reconfiguration for Japanese.

## Strengths vs Our Project

1. **Few-shot learning approach**: The 3 complete receipt examples in the prompt are effective for teaching the LLM expected output format. We use instruction-based prompting; adding few-shot examples could improve consistency.

2. **Expense categorization**: Automatically assigns 19 product categories to each item. We don't categorize individual items (we handle tax categories at the receipt level).

3. **Interactive correction UI**: The Gradio interface lets users edit extracted data before confirming. This human-in-the-loop approach catches errors we'd miss. We have no user-facing correction interface.

4. **Visualization pipeline**: The 6-chart visualization suite turns extracted data into actionable spending insights. We output raw JSON with no analytics layer.

5. **PaddleOCR integration**: Direct PaddleOCR usage (PP-OCRv4) without cloud API dependency. We use Google Cloud Vision which has per-request costs.

## Weaknesses vs Our Project

1. **Single LLM call, no verification**: One shot with GPT-3.5-turbo and done. No retry, no confidence scoring, no multi-pass verification. If the LLM hallucinates, it passes through.

2. **No OCR confidence handling**: PaddleOCR returns confidence scores per text region, but this project ignores them completely -- just concatenates raw text.

3. **Minimal validation**: Only `ensure_numeric_columns()` (coerce to float) and `ensure_category()` (map invalid to 'other'). No structural validation like our subset-sum tax matching.

4. **No ground truth or testing**: Zero test fixtures, no accuracy benchmarks, no regression testing. Just a demo.

5. **GPT-3.5-turbo is weak**: Using one of the least capable models. Our DeepSeek V3.2 is significantly more capable for structured extraction.

6. **Hardcoded few-shot examples**: The 3 example receipts are hardcoded in the source. If receipt formats change, the examples become stale. No dynamic example selection.

7. **Spanish-only**: Would need significant rework for Japanese receipts.

8. **No post-processing**: No merchant normalization, no field registry, no deterministic corrections. Raw LLM output with minimal cleanup.

9. **Notebook-heavy codebase**: 97.5% Jupyter Notebook -- not production-ready architecture.

## What We Can Learn

1. **Few-shot receipt examples in prompts**: Adding 2-3 complete Japanese receipt examples (OCR text -> expected JSON) to our LLM prompt could improve extraction consistency. This is a well-known technique, but seeing it applied specifically to receipts is instructive. We could create few-shot examples from our test fixtures.

2. **Multi-format date parsing with fallback chain**: The `parse_dates()` function that tries 9 formats sequentially is a practical pattern. We handle Japanese era dates, but a similar fallback chain for edge cases could reduce date parsing failures:
   ```python
   formats = ["%Y-%m-%d", "%d/%m/%Y", "%Y年%m月%d日", ...]
   for fmt in formats:
       try: return datetime.strptime(date_str, fmt)
       except: continue
   ```

3. **Interactive correction workflow**: The Gradio edit-before-confirm pattern is worth considering if we ever build a UI. Showing extracted data in an editable table with "Confirm" button captures corrections that can feed back into training data.

4. **Product-level categorization**: If we need per-item categorization (beyond receipt-level tax categories), the 19-category enum with LLM assignment is a simple starting approach.

## Recommendation

**Do not adopt.** This is a small demo project with 28 stars, no tests, notebook-heavy code, and Spanish-only support. The architecture is a simplified version of what we already have. However, two ideas are worth testing:

1. **Add few-shot examples to our LLM prompt**: Select 2-3 representative Japanese receipts from our test fixtures and include them as input/output pairs in the DeepSeek prompt
2. **Date format fallback chain**: Implement a similar multi-format date parser as a post-processing step for edge cases our current parser misses
