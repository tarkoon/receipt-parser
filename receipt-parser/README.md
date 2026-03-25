# Receipt Parser

A local, privacy-first receipt and invoice parser that extracts structured data from photos and PDFs. It combines PaddleOCR for text detection (supporting both Japanese and English) with Ollama-hosted LLMs for field extraction, producing structured JSON or CSV output. All processing runs on your machine -- no cloud APIs, no data leaves your device.

## Installation

**Prerequisites:**

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running
- [Poppler](https://poppler.freedesktop.org/) (required for PDF support via `pdf2image`)

**Setup:**

```bash
# 1. Create or activate the conda environment
conda activate financial-aid

# 2. Pull the default model
ollama pull qwen3.5

# 3. Install dependencies (GPU — requires CUDA)
pip install -r requirements.txt

# 3 (alt). CPU-only fallback — replace paddlepaddle-gpu with paddlepaddle
pip install paddlepaddle paddleocr opencv-python-headless pdf2image pdfplumber \
    "pydantic>=2.0,<3.0" "typer[all]" "ollama>=0.4.0" "Pillow>=9.0" numpy

# 4. (Optional) Install dev dependencies for tests
pip install -r requirements-dev.txt
```

## Usage

**Basic usage — JSON to stdout:**

```bash
python cli.py receipt.jpg
```

**Multi-pass verification (re-runs extraction to correct errors):**

```bash
python cli.py receipt.jpg -p 2
```

**Debug mode — save intermediate artifacts to `debug/`:**

```bash
python cli.py receipt.jpg --debug
```

**Batch processing — run on an entire directory:**

```bash
python cli.py ./receipts/ -o results.json
```

**PDF input with verbose output:**

```bash
python cli.py receipt.pdf -v
```

**CSV output:**

```bash
python cli.py receipt.jpg -f csv -o output.csv
```

**Show version:**

```bash
python cli.py --version
```

### CLI Options

| Flag | Long | Default | Description |
|------|------|---------|-------------|
| (positional) | | | Image, PDF, or directory to process |
| `-o` | `--output` | stdout | Output file path |
| `-m` | `--model` | `qwen3.5` | Ollama model name |
| `-p` | `--passes` | `1` | Extraction passes (2+ enables verification) |
| `-f` | `--format` | `json` | Output format: `json` or `csv` |
| `-d` | `--debug` | off | Save debug artifacts to `debug/<filename>/` |
| `-v` | `--verbose` | off | Print per-pass summaries and warnings to stderr |
| | `--version` | | Show version and exit |

## Debug Mode

When `--debug` is passed, the pipeline writes intermediate artifacts to `debug/<input_stem>/`. These files let you trace exactly where a parsing error originates.

### Artifacts

| File | Contents |
|------|----------|
| `01_original.png` | The raw input image as loaded |
| `02_preprocessed.png` | After grayscale conversion, deskew, and contrast normalization |
| `03_ocr_bboxes.png` | Bounding boxes drawn on the preprocessed image, color-coded by OCR confidence: green (>= 90%), yellow (>= 70%), red (< 70%) |
| `04_ocr_grouped.txt` | OCR text after spatial grouping into lines |
| `05_pass1_llm_response.json` | Raw structured output from the first LLM extraction pass |
| `06_pass1_warnings.txt` | Validation warnings from pass 1 (arithmetic mismatches, etc.) |
| `10_field_overlay.png` | Extracted fields mapped back to their OCR bounding boxes, color-coded per field with a legend |
| `pipeline_trace.txt` | Step-by-step timing for the entire pipeline |

### Diagnostic Workflow

When a result looks wrong, work backwards through the artifacts:

1. **Check `10_field_overlay.png`** -- Are fields mapped to the correct regions of the receipt? If a field points to the wrong text, the problem is in LLM extraction.
2. **Check `05_pass1_llm_response.json`** -- Does the raw LLM output contain the correct values? If not, the LLM misinterpreted the OCR text.
3. **Check `03_ocr_bboxes.png` and `04_ocr_grouped.txt`** -- Is the OCR text accurate? Red bounding boxes indicate low-confidence detections. If text is garbled, the problem is upstream of the LLM.
4. **Check `02_preprocessed.png`** -- Is the image clean enough for OCR? If it is heavily skewed, low-contrast, or blurry, preprocessing may need adjustment.

## Extending the Schema

All extraction fields are defined in a single registry. Adding a new field requires no prompt editing and no changes to the debug overlay logic.

### Step 1: Add a `FieldMeta` entry to `FIELD_REGISTRY` in `schema.py`

```python
FieldMeta(
    name="tip",
    debug_color_bgr=(0, 200, 100),
    prompt_hint="Look for tip, gratuity, or service charge amounts.",
    extraction_aliases=["tip", "gratuity", "チップ"],
),
```

### Step 2: Add the field to the `Receipt` Pydantic model in `schema.py`

```python
class Receipt(BaseModel):
    # ... existing fields ...
    tip: Optional[float] = None
```

### Step 3 (optional): Add validation logic in `validation.py`

```python
# Example: check that tip + subtotal + tax ~ total
if receipt.tip is not None and receipt.subtotal is not None:
    expected = receipt.subtotal + receipt.tip + tax_sum
    if abs(expected - receipt.total) > 2:
        warnings.append(f"Total does not match subtotal + tip + taxes")
```

### What auto-updates

Once the field is in `FIELD_REGISTRY` and the `Receipt` model, the following adapt automatically:

- **LLM prompt hints** -- `generate_extraction_prompt()` includes the new field's `prompt_hint` and aliases
- **Ollama schema enforcement** -- the Pydantic model is used as the structured output format
- **Debug overlay** -- `draw_field_overlay()` picks up the new field's color from the registry
- **JSON/CSV output** -- Pydantic serialization includes the new field in all output

## Model Selection

The `--model` flag accepts any Ollama model name. The table below covers tested options:

| Model | VRAM | JP Quality | EN Quality | Speed | Use Case |
|-------|------|------------|------------|-------|----------|
| `qwen3.5` | ~5 GB | Excellent | Excellent | ~3 s | Default -- best balance of quality and speed |
| `qwen2.5:14b` | ~8 GB (Q4) | Top tier | Top tier | ~6 s | Accuracy upgrade when VRAM allows |
| `qwen2.5:3b` | ~2.5 GB | Good | Good | ~1.5 s | Speed priority or low-VRAM machines |

Speed estimates are per receipt on a mid-range GPU. CPU inference will be significantly slower.
