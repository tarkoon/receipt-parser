# Project Setup
- Python: Use `python` instead of `python3`. Conda is installed - Use the environment `financial-aid`
- Default LLM: DeepSeek V3.2 via direct API (api.deepseek.com). Always use `seed=42` for deterministic output.
- Prefer latest library versions over pinning old ones - adapt code to new APIs instead of downgrading

## Testing Rules
- **NEVER modify truth/fixture files** (`*_truth.json`) without explicit permission from the user. Fix the pipeline to produce correct output — do not change the expected answers.

## Running Tests & Benchmarks
```bash
# Unit + validation tests (fast, no API calls)
python -m pytest tests/test_unit.py tests/test_validation.py -v

# Accuracy tests (cached OCR, needs Cloud Vision configured)
python -m pytest tests/test_accuracy.py -v

# Accuracy tests with JSON report
python -m pytest tests/test_accuracy.py -v --json-report --json-report-file=tests/results/accuracy/latest.json

# Robustness benchmark (fresh OCR, multiple runs)
python tests/benchmark.py --workers 4

# Benchmark specific fixtures
python tests/benchmark.py --fixtures receipt_14 receipt_29 --runs 5

# CI mode (cached OCR, 1 run, exit non-zero on failure)
python tests/benchmark.py --ci
```

## Project Structure
- `src/receipt_parser/` — installable Python package (core pipeline)
- `tests/` — pytest tests + benchmark + fixtures + OCR variants
- `scripts/` — one-off utilities (benchmark_models.py, etc.)
- `docs/` — planning documents

## Windows / Conda Environment
- Always set `PYTHONIOENCODING=utf-8` when running Python commands (Japanese text breaks cp932 default)
- Never rely on stdout for Japanese/non-ASCII output through `conda run` — write results to a UTF-8 file instead, then read the file
- For diagnostic/debug scripts: always write output to a JSON or text file with `encoding='utf-8'`, never print to console

## LLM Pipeline Debugging
- When debugging LLM-based pipelines, confirm determinism FIRST (same input → same output). If the model API supports seed/temperature controls, lock those down before investigating individual failures.
- After the first failing test run, STOP and classify errors (deterministic vs non-deterministic, code bug vs model behavior vs OCR variance). Present the classification and proposed fix strategy BEFORE running tests again.

## UI Design & Review Workflow

### Agents
- **ui-designer** — generates mockups via Gemini CLI, self-validates with Playwright before returning
- **ui-reviewer** — audits the live running app with Playwright, returns a structured report
- You orchestrate the handoff — agents do not talk to each other directly

### Phase 1: Design
Invoke `ui-designer` with a brief covering: design system, tech stack, output path (`local/design/`), task description, and any reference files. The designer will generate, self-review, and only return an approved mockup.

### Phase 2: Review
Invoke `ui-reviewer` with: the live URL, the design criteria or mockup path as reference, and scope. Screenshots save to `local/screenshots/review/`.

### Phase 3: Iterate (if needed)
Pass blocking issues from the reviewer back to the designer. The designer revises and re-validates before returning. Repeat Phase 2. Minor issues go directly to the implementer.

### File locations
| Artefact | Path |
|---|---|
| Mockups | `local/design/` |
| Self-review screenshots | `local/screenshots/mockup/` |
| Live app screenshots | `local/screenshots/review/` |
