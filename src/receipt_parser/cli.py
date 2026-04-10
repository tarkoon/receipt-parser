"""cli.py — Typer CLI entry point."""

import csv
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", encoding="utf-8")

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from .pipeline import process_document, process_batch
from .llm import DEFAULT_MODEL
from . import __version__

app = typer.Typer(help="Receipt Parser — extract structured data from receipts and invoices.")


def version_callback(value: bool):
    if value:
        typer.echo(f"receipt-parser {__version__}")
        raise typer.Exit()


def _flatten_for_csv(result: dict, file_path: str = "") -> list[dict]:
    """Flatten a Receipt result into CSV rows (one per line item)."""
    base = {
        "file": file_path,
        "merchant": result.get("merchant", ""),
        "date": result.get("date", ""),
        "currency": result.get("currency", ""),
        "total": result.get("total", ""),
        "subtotal": result.get("subtotal", ""),
        "payment_method": result.get("payment_method", ""),
        "location": result.get("location", ""),
    }

    taxes = result.get("taxes", [])
    base["tax_rates"] = ";".join(t.get("rate", "") for t in taxes) if taxes else ""
    base["tax_amounts"] = ";".join(str(t.get("amount", "")) for t in taxes) if taxes else ""

    line_items = result.get("line_items", [])
    if not line_items:
        base.update({
            "item_description": "", "item_qty": "", "item_unit_price": "",
            "item_total": "", "item_tax_category": "",
        })
        return [base]

    rows = []
    for item in line_items:
        row = dict(base)
        row["item_description"] = item.get("description", "")
        row["item_qty"] = item.get("qty", "")
        row["item_unit_price"] = item.get("unit_price", "")
        row["item_total"] = item.get("total", "")
        row["item_tax_category"] = item.get("tax_category", "")
        rows.append(row)

    return rows


CSV_FIELDNAMES = [
    "file", "merchant", "date", "currency", "total", "subtotal",
    "payment_method", "location",
    "item_description", "item_qty", "item_unit_price", "item_total",
    "item_tax_category", "tax_rates", "tax_amounts",
]


@app.command()
def parse(
    input_path: Path = typer.Argument(..., help="Image, PDF, or directory to process"),
    output: Path = typer.Option(None, "--output", "-o", help="Output file (default: stdout)"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m",
                              help="LLM model (default: DeepSeek; prefix 'ollama/' for Ollama)"),
    passes: int = typer.Option(2, "--passes", "-p", min=1, max=3,
                               help="Extraction passes (2+ enables verification)"),
    format: str = typer.Option("json", "--format", "-f", help="Output format: json or csv"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Save debug artifacts"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print OCR text and warnings"),
    workers: int = typer.Option(1, "--workers", "-w", min=1, max=16,
                                help="Concurrent workers for batch processing (default: 1)"),
    version: bool = typer.Option(False, "--version", callback=version_callback,
                                 is_eager=True, help="Show version"),
):
    """Parse receipts and invoices into structured data."""
    if not input_path.exists():
        typer.echo(f"Error: {input_path} does not exist", err=True)
        raise typer.Exit(1)

    if input_path.is_dir():
        extensions = {".jpg", ".jpeg", ".png", ".pdf", ".tiff", ".tif", ".bmp"}
        files = sorted([f for f in input_path.iterdir()
                        if f.suffix.lower() in extensions])
        if not files:
            typer.echo(f"No supported files found in {input_path}", err=True)
            raise typer.Exit(1)
    else:
        files = [input_path]

    is_batch = len(files) > 1

    from .ocr import init_cloud_vision
    engine = init_cloud_vision()

    # Use batch processing for multiple files with workers > 1
    if is_batch and workers > 1:
        effective_workers = min(workers, len(files))

        if verbose:
            typer.echo(f"Batch processing {len(files)} files with {effective_workers} workers", err=True)
            batch_start = time.perf_counter()

            def _on_progress(file_path, result, completed, total):
                status = "OK" if "_error" not in result else f"ERROR: {result['_error']}"
                typer.echo(f"  [{completed}/{total}] {file_path.name}: {status}", err=True)

            results = process_batch(
                files, model=model, debug=debug, passes=passes,
                ocr_engine=engine, max_workers=effective_workers,
                on_progress=_on_progress,
            )

            elapsed = time.perf_counter() - batch_start
            per_file = elapsed / len(files)
            typer.echo(f"  Batch complete: {elapsed:.1f}s total, "
                       f"{per_file:.1f}s/file avg ({len(files)/elapsed:.1f} files/min)",
                       err=True)
        else:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total} files"),
                TimeElapsedColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task("Processing", total=len(files))

                def _on_progress_bar(file_path, result, completed, total):
                    progress.update(task, completed=completed)

                results = process_batch(
                    files, model=model, debug=debug, passes=passes,
                    ocr_engine=engine, max_workers=effective_workers,
                    on_progress=_on_progress_bar,
                )
    else:
        # Sequential processing
        results = []

        if verbose:
            for file in files:
                typer.echo(f"Processing: {file.name}", err=True)
                try:
                    result = process_document(
                        file, model=model, debug=debug,
                        passes=passes, ocr_engine=engine,
                    )
                    result["_file"] = str(file)

                    warnings = result.get("_warnings", [])
                    history = result.get("_pass_history", [])
                    for entry in history:
                        n = entry["pass"]
                        w = entry["warnings"]
                        status = f"{len(w)} warnings" if w else "clean"
                        typer.echo(f"  Pass {n}: {status}", err=True)
                    if warnings:
                        typer.echo("  Final warnings:", err=True)
                        for w in warnings:
                            typer.echo(f"    - {w}", err=True)

                    if debug:
                        trace = result.get("_trace", "")
                        if trace:
                            typer.echo(f"\n{trace}", err=True)

                except Exception as e:
                    result = {"_file": str(file), "_error": str(e)}
                    typer.echo(f"  Error: {e}", err=True)

                results.append(result)
        else:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                transient=True,
            ) as progress:
                if is_batch:
                    overall = progress.add_task("Files", total=len(files))
                file_task = progress.add_task("Starting", total=100)

                for i, file in enumerate(files):
                    def _on_stage(stage, detail, pct, _task=file_task):
                        progress.update(_task, completed=int(pct * 100), description=detail)

                    try:
                        result = process_document(
                            file, model=model, debug=debug,
                            passes=passes, ocr_engine=engine,
                            on_stage=_on_stage,
                        )
                        result["_file"] = str(file)
                    except Exception as e:
                        result = {"_file": str(file), "_error": str(e)}

                    results.append(result)

                    if is_batch:
                        progress.update(overall, completed=i + 1)
                        progress.update(file_task, completed=0, description="Starting")

    final_output = results if is_batch else results[0]

    if format == "csv":
        all_rows = []
        for r in results:
            all_rows.extend(_flatten_for_csv(r, r.get("_file", "")))

        if output:
            with open(output, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES,
                                        extrasaction="ignore")
                writer.writeheader()
                writer.writerows(all_rows)
        else:
            writer = csv.DictWriter(sys.stdout, fieldnames=CSV_FIELDNAMES,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
    else:
        json_str = json.dumps(final_output, ensure_ascii=False, indent=2)
        if output:
            output.write_text(json_str, encoding="utf-8")
        else:
            typer.echo(json_str)


@app.command()
def usage(
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    reset: bool = typer.Option(False, "--reset", help="Reset all counters for current period"),
    history: bool = typer.Option(False, "--history", "-H", help="Show historical usage log"),
    page: int = typer.Option(1, "--page", help="Page number for history view"),
    sync: bool = typer.Option(False, "--sync", help="Set counters from online dashboard values"),
    set_ds_hit: int = typer.Option(None, "--set-ds-hit", help="Set DeepSeek cache hit token count"),
    set_ds_miss: int = typer.Option(None, "--set-ds-miss", help="Set DeepSeek cache miss token count"),
    set_ds_out: int = typer.Option(None, "--set-ds-out", help="Set DeepSeek output token count"),
    set_ds_calls: int = typer.Option(None, "--set-ds-calls", help="Set DeepSeek API call count"),
    billing_day: int = typer.Option(None, "--billing-day", min=1, max=28,
                                     help="Set billing period start day (1-28, persisted)"),
    no_fetch: bool = typer.Option(False, "--no-fetch", help="Skip Cloud Vision API auto-fetch"),
):
    """Show API usage and estimated costs for the current billing period."""
    from .usage import (
        get_usage, reset_usage, sync_usage, get_history,
        set_billing_start_day, get_billing_period_label,
    )

    # Billing day config
    if billing_day is not None:
        set_billing_start_day(billing_day)
        typer.echo(f"Billing period start day set to {billing_day}.")
        if not any([reset, history, sync, json_output,
                     set_ds_hit, set_ds_miss, set_ds_out, set_ds_calls]):
            return

    if reset:
        reset_usage()
        typer.echo("Usage counters reset.")
        return

    # Sync: interactive or flag-based
    if sync:
        _run_sync_interactive()
        return

    if any(v is not None for v in (set_ds_hit, set_ds_miss, set_ds_out, set_ds_calls)):
        sync_usage(ds_cache_hit=set_ds_hit, ds_cache_miss=set_ds_miss,
                   ds_output=set_ds_out, ds_calls=set_ds_calls)
        typer.echo("Usage counters updated.")
        # Fall through to display updated stats

    # History view
    if history:
        _show_history(page, json_output)
        return

    # Current period view (auto-fetches Cloud Vision from GCP)
    should_fetch = not no_fetch
    if should_fetch and not json_output:
        typer.echo("Fetching Cloud Vision usage from GCP...", err=True)

    stats = get_usage(auto_fetch_cv=should_fetch)

    if json_output:
        typer.echo(json.dumps(stats, indent=2))
        return

    cv = stats["cloud_vision"]
    ds = stats["deepseek"]
    docs = stats["documents"]

    typer.echo(f"API Usage — {stats['billing_period_label']}  ({stats['days_until_reset']} days left)")
    typer.echo("─" * 56)

    # Documents
    typer.echo(f"\n  Documents")
    typer.echo(f"    Total processed:  {docs['total_processed']:,}")
    typer.echo(f"    Unique files:     {docs['unique_processed']:,}")

    # Cloud Vision
    cv_source = "live" if should_fetch and "_cv_fetch_error" not in stats else "local"
    typer.echo(f"\n  Google Cloud Vision  ({cv_source})")
    typer.echo(f"    Calls:          {cv['calls']:,}")
    typer.echo(f"    Free tier:      {cv['remaining_free']:,} / {cv['free_limit']:,} remaining")
    if cv["billable_calls"] > 0:
        typer.echo(f"    Billable calls: {cv['billable_calls']:,}")
        typer.echo(f"    Est. cost:      ${cv['est_cost_usd']:.4f}")
    else:
        typer.echo(f"    Est. cost:      $0.00 (within free tier)")
    if "_cv_fetch_error" in stats:
        typer.echo(f"    (fetch failed: {stats['_cv_fetch_error']})", err=True)

    # DeepSeek
    typer.echo(f"\n  DeepSeek API")
    typer.echo(f"    Calls:            {ds['calls']:,}")
    typer.echo(f"    Cache hit tokens: {ds['cache_hit_tokens']:,}  (@ $0.028/1M)")
    typer.echo(f"    Cache miss tokens:{ds['cache_miss_tokens']:>11,}  (@ $0.28/1M)")
    typer.echo(f"    Output tokens:    {ds['output_tokens']:,}  (@ $0.42/1M)")
    typer.echo(f"    Est. cost:        ${ds['est_cost_usd']:.4f}")

    # Total
    typer.echo(f"\n{'─' * 56}")
    typer.echo(f"  Total est. cost:  ${stats['total_est_cost_usd']:.4f}")


def _run_sync_interactive():
    """Interactive sync: prompt user for DeepSeek dashboard values.

    Cloud Vision is auto-fetched from GCP, so only DeepSeek needs manual sync.
    """
    from .usage import sync_usage, get_usage

    current = get_usage()

    typer.echo("Sync DeepSeek usage from your dashboard.")
    typer.echo("(Cloud Vision is auto-fetched from GCP — no manual sync needed.)")
    typer.echo("Press Enter to keep current value, or type a new number.\n")

    ds_calls_cur = current["deepseek"]["calls"]
    ds_hit_cur = current["deepseek"]["cache_hit_tokens"]
    ds_miss_cur = current["deepseek"]["cache_miss_tokens"]
    ds_out_cur = current["deepseek"]["output_tokens"]

    ds_calls_input = typer.prompt(
        f"  DeepSeek API calls [{ds_calls_cur:,}]",
        default="", show_default=False,
    ).strip()

    ds_hit_input = typer.prompt(
        f"  DeepSeek cache hit tokens [{ds_hit_cur:,}]",
        default="", show_default=False,
    ).strip()

    ds_miss_input = typer.prompt(
        f"  DeepSeek cache miss tokens [{ds_miss_cur:,}]",
        default="", show_default=False,
    ).strip()

    ds_out_input = typer.prompt(
        f"  DeepSeek output tokens [{ds_out_cur:,}]",
        default="", show_default=False,
    ).strip()

    ds_calls_val = int(ds_calls_input) if ds_calls_input else None
    ds_hit_val = int(ds_hit_input) if ds_hit_input else None
    ds_miss_val = int(ds_miss_input) if ds_miss_input else None
    ds_out_val = int(ds_out_input) if ds_out_input else None

    if all(v is None for v in (ds_calls_val, ds_hit_val, ds_miss_val, ds_out_val)):
        typer.echo("\nNo changes made.")
        return

    sync_usage(ds_calls=ds_calls_val,
               ds_cache_hit=ds_hit_val, ds_cache_miss=ds_miss_val, ds_output=ds_out_val)
    typer.echo("\nUsage counters synced.")


def _show_history(page: int, json_output: bool):
    """Display paginated historical usage log."""
    from .usage import get_history, get_usage

    result = get_history(page=page)

    if json_output:
        typer.echo(json.dumps(result, indent=2))
        return

    if not result["entries"]:
        typer.echo("No historical data yet. History is recorded when the month rolls over.")
        return

    typer.echo(f"Usage History  (page {result['page']}/{result['total_pages']}, "
               f"{result['total_months']} months)")
    typer.echo("─" * 62)
    typer.echo(f"  {'Month':<10} {'CV Calls':>10} {'DS Tokens':>14} {'Docs':>7} {'Cost':>10}")
    typer.echo(f"  {'─'*10} {'─'*10} {'─'*14} {'─'*7} {'─'*10}")

    for entry in result["entries"]:
        cv = entry.get("cloud_vision", {})
        ds = entry.get("deepseek", {})
        docs = entry.get("documents", {})
        total_tokens = (ds.get("cache_hit_tokens", 0) + ds.get("cache_miss_tokens", 0)
                        + ds.get("output_tokens", 0))
        cost = entry.get("est_cost_usd", 0)
        doc_count = docs.get("total_processed", 0)

        typer.echo(f"  {entry['month']:<10} {cv.get('calls', 0):>10,} "
                   f"{total_tokens:>14,} {doc_count:>7,} ${cost:>9.4f}")

    typer.echo(f"{'─' * 62}")
    typer.echo(f"  Lifetime cost: ${result['lifetime_cost_usd']:.4f}")

    if result["total_pages"] > 1:
        typer.echo(f"\n  Use --page N to navigate (1-{result['total_pages']})")


@app.command()
def setup():
    """Interactive first-run setup wizard."""
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    env_path = _PROJECT_ROOT / ".env"
    env_example = _PROJECT_ROOT / ".env.example"

    typer.echo("Receipt Parser — Setup Wizard")
    typer.echo("=" * 40)

    # Step 0: Ensure user_rules.json exists
    _user_rules = Path(__file__).resolve().parent / "user_rules.json"
    if not _user_rules.exists():
        _user_rules.write_text(
            '{\n  "merchant_map": {}\n}\n', encoding="utf-8"
        )
        typer.echo("Created user_rules.json (empty — see README to customize).")

    # Step 1: .env file
    typer.echo("\n[1/4] Environment file")
    if env_path.exists():
        typer.echo("  .env file found.")
    elif env_example.exists():
        typer.echo("  No .env file found. Creating from .env.example...")
        import shutil
        shutil.copy(env_example, env_path)
        typer.echo("  Created .env — you'll fill in the values next.")
    else:
        typer.echo("  No .env or .env.example found. Creating .env...")
        env_path.write_text("# Receipt Parser config\n", encoding="utf-8")

    # Step 2: DeepSeek API key
    typer.echo("\n[2/4] DeepSeek API key")
    import os
    # Reload .env to pick up existing values
    from dotenv import dotenv_values
    env_vals = dotenv_values(env_path)
    existing_ds_key = env_vals.get("DEEPSEEK_API_KEY", "") or os.environ.get("DEEPSEEK_API_KEY", "")

    if existing_ds_key and existing_ds_key != "your-deepseek-api-key-here":
        masked = existing_ds_key[:8] + "..." + existing_ds_key[-4:]
        typer.echo(f"  Key found: {masked}")
        change = typer.prompt("  Replace it? [y/N]", default="n", show_default=False).strip().lower()
        if change != "y":
            ds_key = existing_ds_key
        else:
            ds_key = typer.prompt("  Enter DeepSeek API key").strip()
    else:
        typer.echo("  Get your key at: https://platform.deepseek.com/api_keys")
        ds_key = typer.prompt("  Enter DeepSeek API key").strip()

    if ds_key:
        _update_env_var(env_path, "DEEPSEEK_API_KEY", ds_key)
        # Validate with a test call
        typer.echo("  Testing connection...", nl=False)
        os.environ["DEEPSEEK_API_KEY"] = ds_key
        try:
            from openai import OpenAI
            client = OpenAI(base_url="https://api.deepseek.com", api_key=ds_key, timeout=10)
            client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            typer.echo(" OK")
        except Exception as e:
            typer.echo(f" FAILED: {e}")
            typer.echo("  You can fix this later in .env and re-run setup.")

    # Step 3: GCP / Cloud Vision
    typer.echo("\n[3/4] Google Cloud Vision")
    existing_project = env_vals.get("GOOGLE_CLOUD_PROJECT", "") or os.environ.get("GOOGLE_CLOUD_PROJECT", "")

    if existing_project and existing_project != "your-gcp-project-id":
        typer.echo(f"  Project: {existing_project}")
        change = typer.prompt("  Replace it? [y/N]", default="n", show_default=False).strip().lower()
        if change != "y":
            gcp_project = existing_project
        else:
            gcp_project = typer.prompt("  Enter GCP project ID").strip()
    else:
        typer.echo("  You need a GCP project with Cloud Vision API enabled.")
        typer.echo("  Enable at: https://console.cloud.google.com/apis/library/vision.googleapis.com")
        typer.echo("  Then authenticate: gcloud auth application-default login")
        gcp_project = typer.prompt("  Enter GCP project ID (or press Enter to skip)", default="",
                                    show_default=False).strip()

    if gcp_project:
        _update_env_var(env_path, "GOOGLE_CLOUD_PROJECT", gcp_project)
        os.environ["GOOGLE_CLOUD_PROJECT"] = gcp_project
        typer.echo("  Testing Cloud Vision...", nl=False)
        try:
            from .ocr import init_cloud_vision
            init_cloud_vision()
            typer.echo(" OK")
        except Exception as e:
            typer.echo(f" FAILED: {e}")
            typer.echo("  Make sure you've run: gcloud auth application-default login")

    # Step 4: Verify end-to-end
    typer.echo("\n[4/4] End-to-end test")
    run_test = typer.prompt("  Run a quick test with a sample receipt? [Y/n]",
                             default="y", show_default=False).strip().lower()
    if run_test != "n":
        fixtures_dir = _PROJECT_ROOT / "tests" / "fixtures"
        sample = next(fixtures_dir.glob("receipt_1.*"), None) if fixtures_dir.exists() else None
        if sample:
            typer.echo(f"  Processing {sample.name}...")
            try:
                # Reload env
                load_dotenv(env_path, override=True)
                result = process_document(sample, passes=1)
                merchant = result.get("merchant", "?")
                total = result.get("total", "?")
                typer.echo(f"  Result: {merchant} — total {total}")
                typer.echo("  Pipeline is working!")
            except Exception as e:
                typer.echo(f"  Test failed: {e}")
        else:
            typer.echo("  No test fixtures found, skipping.")

    typer.echo("\n" + "=" * 40)
    typer.echo("Setup complete! Run 'receipt-parser parse <image>' to get started.")


def _update_env_var(env_path: Path, key: str, value: str):
    """Update or add an env var in a .env file."""
    if not env_path.exists():
        env_path.write_text(f"{key}={value}\n", encoding="utf-8")
        return

    lines = env_path.read_text(encoding="utf-8").splitlines()
    found = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(f"{key}=") or stripped.startswith(f"# {key}="):
            lines[i] = f"{key}={value}"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@app.command()
def clean(
    cache: bool = typer.Option(False, "--cache", help="Also clear OCR cache"),
    all_data: bool = typer.Option(False, "--all", help="Clear all data (cache + usage history)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Clean up debug artifacts and optionally cached data."""
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    targets = []

    # Always: debug directories
    debug_dir = _PROJECT_ROOT / "debug"
    if debug_dir.exists():
        size = sum(f.stat().st_size for f in debug_dir.rglob("*") if f.is_file())
        targets.append(("Debug artifacts", debug_dir, size))

    # --cache or --all: OCR cache
    if cache or all_data:
        from .usage import _DATA_DIR
        cache_dir = _DATA_DIR / "ocr_cache"
        # Also check old location
        old_cache = Path(__file__).resolve().parent / ".ocr_cache"
        for cd in [cache_dir, old_cache]:
            if cd.exists():
                size = sum(f.stat().st_size for f in cd.rglob("*") if f.is_file())
                targets.append(("OCR cache", cd, size))

    # --all: usage data
    if all_data:
        from .usage import _DATA_DIR
        if _DATA_DIR.exists():
            for f in _DATA_DIR.glob("*.json"):
                targets.append(("Usage data", f, f.stat().st_size))

    if not targets:
        typer.echo("Nothing to clean.")
        return

    typer.echo("Will remove:")
    total_size = 0
    for label, path, size in targets:
        total_size += size
        size_str = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / 1024 / 1024:.1f} MB"
        typer.echo(f"  {label}: {path} ({size_str})")

    if not yes:
        confirm = typer.prompt("Proceed? [y/N]", default="n", show_default=False).strip().lower()
        if confirm != "y":
            typer.echo("Cancelled.")
            return

    import shutil
    for label, path, _ in targets:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        typer.echo(f"  Removed: {path}")

    typer.echo("Done.")


if __name__ == "__main__":
    app()
