# Receipt Parser

Extract structured data from Japanese receipts, utility bills, and payment slips using Google Cloud Vision OCR and DeepSeek LLM.

## Quick Start

```bash
# Clone and install
git clone <repo-url> && cd financial-aid
conda create -n financial-aid python=3.12
conda activate financial-aid
pip install -e .

# Run the setup wizard
receipt-parser setup
```

The setup wizard walks you through configuring API keys, GCP credentials, and runs a test to verify everything works.

## Requirements

- Python 3.10+
- [DeepSeek API key](https://platform.deepseek.com/api_keys) (primary LLM)
- [Google Cloud](https://console.cloud.google.com) project with Cloud Vision API enabled

## Manual Setup

If you prefer to configure manually instead of using the wizard:

1. **Create `.env`** from the template:
   ```bash
   cp .env.example .env
   ```

2. **Add your DeepSeek API key** to `.env`:
   ```
   DEEPSEEK_API_KEY=sk-your-key-here
   ```

3. **Set up Google Cloud Vision**:
   ```bash
   # Set your project ID in .env
   GOOGLE_CLOUD_PROJECT=your-project-id

   # Enable the Cloud Vision API
   # Visit: https://console.cloud.google.com/apis/library/vision.googleapis.com

   # Authenticate
   gcloud auth application-default login
   ```

## Usage

### Parse receipts

```bash
# Single file
receipt-parser parse receipt.jpg

# Directory (batch)
receipt-parser parse ./receipts/ --workers 4

# With verification pass (recommended)
receipt-parser parse receipt.jpg --passes 2

# CSV output
receipt-parser parse ./receipts/ --format csv --output results.csv

# Debug mode (saves OCR bboxes, pipeline trace)
receipt-parser parse receipt.jpg --debug

# Verbose (show warnings and pass details)
receipt-parser parse receipt.jpg -v
```

### Check API usage and costs

```bash
# Show current billing period (auto-fetches Cloud Vision from GCP)
receipt-parser usage

# JSON output
receipt-parser usage --json

# Skip Cloud Vision API fetch (faster, offline-friendly)
receipt-parser usage --no-fetch

# View historical usage
receipt-parser usage --history
receipt-parser usage --history --page 2

# Set billing period start day (default: 1st, persisted)
receipt-parser usage --billing-day 15

# Sync DeepSeek tokens from dashboard
receipt-parser usage --sync
receipt-parser usage --set-ds-hit 18507008 --set-ds-miss 1019604 --set-ds-out 2662239
```

### Clean up

```bash
# Remove debug artifacts
receipt-parser clean

# Also clear OCR cache
receipt-parser clean --cache

# Clear everything (cache + usage history)
receipt-parser clean --all
```

## Project Structure

```
financial-aid/
  src/receipt_parser/     # Installable Python package
    pipeline.py           # Main orchestrator
    ocr.py                # Google Cloud Vision OCR
    llm.py                # DeepSeek / Ollama LLM backend
    schema.py             # Pydantic models + prompt generation
    validation.py         # Arithmetic & consistency checks
    normalize.py          # OCR text normalization
    pipeline_receipt.py   # Receipt-specific post-processing
    pipeline_bill.py      # Utility bill post-processing
    pipeline_slip.py      # Payment slip post-processing
    patterns.py           # Regex patterns + confidence routing
    usage.py              # API usage tracking + cost estimation
    cli.py                # Typer CLI
    user_rules.json   # User-maintained merchant mappings
  tests/
    fixtures/             # Test images + truth files (local, gitignored)
      _truth_template.json  # Schema template for creating truth files
    test_accuracy.py      # Fixture-based accuracy tests
    test_unit.py          # Unit tests (no API calls)
    test_validation.py    # Validation logic tests
    benchmark.py          # Robustness benchmark
  .data/                  # Local data (gitignored)
    ocr_cache/            # Cached OCR results
      variants/           # Saved OCR variants for regression tests
    api_usage.json        # Current period usage
    api_usage_history.json
```

## User Rules

The file `src/receipt_parser/user_rules.json` lets you define custom merchant name mappings and categories. The pipeline applies these after LLM extraction. The file is created automatically by `receipt-parser setup` with an empty mapping — add your own entries as needed.

### Format

```json
{
  "merchant_map": {
    "OCR text pattern": {
      "merchant": "Canonical name",
      "category": "optional category"
    }
  }
}
```

### How it works

The pipeline checks if any key in `merchant_map` appears as a substring of the extracted merchant name. If matched, it overrides the merchant (and optionally sets a category).

### Example

```json
{
  "merchant_map": {
    "スマートビリングサービス": {
      "merchant": "Bizmo",
      "category": "internet"
    },
    "マクドナルド": {
      "merchant": "McDonald's",
      "category": "food"
    },
    "ENEOS": {
      "merchant": "ENEOS",
      "category": "gas"
    }
  }
}
```

### Adding entries

1. Open `src/receipt_parser/user_rules.json`
2. Add your pattern under `merchant_map`
3. The pattern should be the text that appears in the OCR output (usually Japanese)
4. The merchant value is what you want in the output
5. Category is optional metadata for your own tracking

No restart needed -- the file is read on each pipeline run.

## Supported Document Types

| Type | Detection | Key fields |
|------|-----------|------------|
| **Receipt** | `"receipt"` — 小計, 合計, レジ keywords | merchant, date, location, total, subtotal, line_items, taxes, payment_method, points_used |
| **Utility Bill** | `"utility_bill"` — 検針, 使用量, kWh, m3 keywords | merchant, date, total, service_type, usage, billing_period |
| **Payment Slip** | `"payment_slip"` — 払込票, 振込 keywords | merchant, total, payer, payment_reference, account_number |

## Testing

```bash
# Unit + validation tests (fast, no API calls)
python -m pytest tests/test_unit.py tests/test_validation.py -v

# Accuracy tests (uses cached OCR, needs Cloud Vision configured)
python -m pytest tests/test_accuracy.py -v

# Robustness benchmark (fresh OCR, multiple runs)
python tests/benchmark.py --runs 5 --workers 4
```

### Test Fixtures

Test fixtures (receipt images + truth JSON files) are stored locally in `tests/fixtures/` and are **not tracked in git** to avoid shipping personal data. The repo includes a `_truth_template.json` schema template for creating your own.

To add a fixture:
1. Place the receipt image in `tests/fixtures/` (e.g., `receipt_1.jpg`)
2. Copy `tests/fixtures/_truth_template.json` to `receipt_1_truth.json`
3. Fill in the expected extraction values
4. Run `python -m pytest tests/test_accuracy.py -v -k receipt_1` to validate

## API Cost Tracking

Usage is tracked automatically:
- **Cloud Vision**: call count auto-fetched from GCP Monitoring API
- **DeepSeek**: tokens tracked per-call with cache hit/miss breakdown

The `receipt-parser usage` command shows real-time cost estimates. Cloud Vision offers 1,000 free calls/month. DeepSeek pricing: $0.028/1M (cache hit), $0.28/1M (cache miss), $0.42/1M (output).

## Troubleshooting

### "No API key configured"
Run `receipt-parser setup` or manually add `DEEPSEEK_API_KEY` to your `.env` file.

### "GOOGLE_CLOUD_PROJECT environment variable is not set"
Add `GOOGLE_CLOUD_PROJECT=your-project-id` to `.env` and run `gcloud auth application-default login`.

### "Cloud Vision API has not been used in project"
Enable it at: https://console.cloud.google.com/apis/library/vision.googleapis.com

### "quota exceeded" warning on usage command
Add `--no-fetch` to skip the GCP Monitoring API call, or set a quota project:
```bash
gcloud auth application-default set-quota-project your-project-id
```

### Japanese text garbled in terminal
Set `PYTHONIOENCODING=utf-8` before running commands, or use `--output file.json` to write to a file.
