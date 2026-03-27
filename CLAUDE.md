# Project Setup

- Python: Use `python` instead of `python3`. Conda is installed - Use the environment `financial-aid`
- Default Ollama model: `qwen3.5:9b` (reasoning model - requires `think=False` for structured output calls)
- Prefer latest library versions over pinning old ones - adapt code to new APIs instead of downgrading

## Testing Rules

- **NEVER modify truth/fixture files** (`*_truth.json`) without explicit permission from the user. Fix the pipeline to produce correct output — do not change the expected answers.

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
