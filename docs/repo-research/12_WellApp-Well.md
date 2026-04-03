# 12. WellApp-ai/Well

## Overview

| Field | Value |
|---|---|
| **Repository** | [WellApp-ai/Well](https://github.com/WellApp-ai/Well) |
| **Stars** | 317 |
| **Forks** | 45 |
| **Language** | TypeScript (64.6%), Python (32.3%) |
| **License** | MIT |
| **Created** | 2025-04-04 |
| **Last Push** | 2026-01-18 (active) |
| **Approach** | Multi-LLM invoice extraction with Zod schema validation, multi-vendor support, fraud detection |

Well is a Chrome extension + backend system for automated invoice retrieval and processing. It is a **commercial product** (WellApp.ai) with an open-source core. Unlike our project, it focuses on invoice/receipt retrieval from 100,000+ supplier portals, not just parsing. The AI extraction component is one module in a larger system.

## Architecture & How It Works

### Monorepo Structure
```
ai-invoice-extractor/     -- Core LLM-based extraction (TypeScript)
  src/
    extractors/           -- Base extractor + provider abstraction
    prompts/              -- Prompt templates per output format
    models/               -- Data schemas
    exporters/            -- XML/JSON output formatters
    libs/                 -- Logger, utilities
ai-invoice-receipt-fraud-detector/  -- Fraud detection module
  packages/
    core/                 -- Heuristic + LLM analysis pipeline
    types/                -- Shared TypeScript interfaces
    models/               -- LLM adapters (OpenAI)
    cli/                  -- CLI interface
ai-receipt-generator/     -- Test data generation
ai-connector/             -- Portal scraping/retrieval
```

### Extraction Pipeline

1. **File Input**: Images (PNG, JPG, WebP) or PDFs up to 20MB
2. **Provider Selection**: User selects LLM vendor via CLI flag (`-v openai`, `-v anthropic`, etc.)
3. **LLM Call**: File is base64-encoded and sent to chosen model with structured output schema
4. **Schema Validation**: Zod schema enforces field structure + confidence scores
5. **Export**: Converts to JSON, XML (FatturaPA Italian e-invoicing), or Factur-X format

### Multi-LLM Provider Architecture

Uses Vercel AI SDK (`ai` package) with provider-specific adapters:
- `@ai-sdk/openai` -- OpenAI (o4-mini, gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo)
- `@ai-sdk/anthropic` -- Claude (claude-4-opus, claude-4-sonnet)
- `@ai-sdk/mistral` -- Mistral (mistral-small, pixtral-large)
- `@ai-sdk/google` -- Gemini (2.0-flash, 1.5-pro, 1.5-flash)
- `ollama-ai-provider` -- Local Ollama models (llama3.2)

The `BaseExtractor` abstract class uses `generateObject()` from the AI SDK which enforces the Zod schema on the LLM output -- the model is forced to return valid JSON matching the schema.

### Confidence Scoring (Key Detail)

The extraction prompt schema (`extract-invoice.prompt.ts`) wraps **every field** in a confidence envelope:

```typescript
const confidenceValue = z.object({
  value: z.union([z.string(), z.number(), z.null()]),
  confidence: z.number().min(0).max(1)
})
```

Every extracted field (invoice_number, date, supplier name, each line item, etc.) comes with a 0.0-1.0 confidence score. The prompt instructs: "For each field, include a confidence score from 0.0 to 1.0 that reflects your certainty based on the OCR and context. If the field is not present, return value: null and confidence: 0.0."

### Fraud Detection Module

Separate from extraction, uses a hybrid approach:
- **Heuristic checks**: Layout inconsistencies, metadata inspection (suspicious PDF producers), mathematical validation (total vs item sum)
- **LLM analysis**: GPT-4 for semantic hallucination detection
- **Output**: Boolean verdict, 0-100 confidence score, categorized indicators (visual, textual, metadata, behavioral, ai_generated)

## Key Features

1. **Per-field confidence scoring** via Zod schema -- not just a single document-level confidence
2. **Multi-vendor LLM support** with clean provider abstraction (swap between OpenAI/Anthropic/Gemini/Mistral/Ollama with a flag)
3. **Fraud detection** as a separate verification step
4. **Italian e-invoicing compliance** (FatturaPA XML, Factur-X)
5. **MCP integration** for AI assistant queries (Claude, Cursor, etc.)
6. **Invoice retrieval** from 100K+ supplier portals (beyond just parsing)
7. **Self-healing automations** that adapt when portal UIs change

## Japanese Support

**None apparent.** The prompts and schemas are generic (no language-specific handling), but the multi-LLM approach means it inherits whatever Japanese capability the chosen model has. There is no explicit Japanese receipt handling, era date parsing, or yen formatting. The export formats target European standards (FatturaPA, Factur-X).

## Strengths vs Our Project

1. **Per-field confidence scoring via schema**: Their Zod `confidenceValue` wrapper around every field is elegant. The LLM is structurally required to output a confidence score for each field, not just overall. This is similar to our v3.0 confidence routing but enforced at the schema level rather than post-hoc calculation.

2. **Multi-LLM provider abstraction**: Clean swappable providers via Vercel AI SDK. We are locked to DeepSeek. Their approach would let us A/B test models trivially.

3. **Fraud detection module**: We have no receipt fraud detection. Their heuristic + LLM hybrid approach (check if total matches item sum, detect AI-generated PDFs) could be valuable for our validation layer.

4. **Structured output via SDK**: Using `generateObject()` from the AI SDK forces the LLM to output valid schema-conforming JSON. We achieve similar with Pydantic but via prompt engineering + post-processing rather than SDK-level enforcement.

5. **Export format variety**: JSON, CSV, XML, UBL -- we only output JSON.

## Weaknesses vs Our Project

1. **No OCR pipeline**: Well relies on the LLM's built-in vision capability (sending images directly to GPT-4o/Claude). We use Google Cloud Vision OCR which is more reliable for Japanese text than multimodal LLM OCR.

2. **No multi-pass verification**: Single LLM call per document. We have confidence-gated retry, multi-pass LLM verification, and OCR-vs-LLM confidence routing.

3. **No post-processing pipeline**: No tax category assignment, no subset-sum matching, no merchant rules, no field registry. Their extraction is purely "ask the LLM and trust the answer."

4. **No test fixtures or ground truth**: No benchmark suite, no accuracy tracking, no robustness testing. We have 36 fixtures with ground truth and variance attribution.

5. **TypeScript, not Python**: Different ecosystem. Integration would require bridging or rewriting.

6. **LLM confidence is self-reported**: The confidence scores come from the LLM itself ("how confident are you?"), which is known to be poorly calibrated. Our confidence routing uses structural signals (OCR confidence from Cloud Vision, deterministic checks) rather than just asking the model.

7. **Commercial product**: Open-source core, but the full system is a commercial SaaS. Feature direction serves business needs, not community.

## What We Can Learn

1. **Schema-level confidence wrapping**: The pattern of wrapping every field in `{value, confidence}` at the schema level is worth adopting. Instead of adding confidence as a separate post-processing step, we could define our Pydantic schema with confidence fields and have the LLM output them natively:
   ```python
   class ConfidenceField(BaseModel):
       value: Optional[str]
       confidence: float = Field(ge=0.0, le=1.0)
   ```

2. **Provider abstraction pattern**: The Vercel AI SDK's `generateObject()` pattern -- where you define a schema and the SDK handles JSON mode, retries, and validation -- could inspire a similar abstraction in Python. Libraries like `instructor` or `marvin` provide similar functionality.

3. **Fraud detection heuristics**: The idea of checking mathematical consistency (total vs item sum) and metadata anomalies as a separate verification pass could complement our existing validation. We could add a "sanity check" step that catches obviously fabricated receipts.

4. **Multi-format export**: If we ever need to export to accounting software, their exporter pattern (BaseExporter with `@ExporterFactory.register()` decorator) is a clean plugin architecture.

## Recommendation

**Do not adopt as a dependency** (TypeScript, commercial focus, no Japanese support), but **adopt two specific patterns**:

1. **Schema-level confidence fields**: Modify our Pydantic schema to include per-field confidence that the LLM outputs directly, reducing the need for post-hoc confidence calculation
2. **Mathematical validation**: Add a fraud/sanity-check step that verifies totals match item sums, tax amounts are plausible, etc. -- similar to their fraud detector's heuristic layer
