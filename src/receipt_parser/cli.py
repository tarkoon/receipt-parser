"""cli.py — Typer CLI entry point."""

import csv
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

import typer

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
            if verbose:
                status = "OK" if "_error" not in result else f"ERROR: {result['_error']}"
                typer.echo(f"  [{completed}/{total}] {file_path.name}: {status}", err=True)

        results = process_batch(
            files, model=model, debug=debug, passes=passes,
            ocr_engine=engine, max_workers=effective_workers,
            on_progress=_on_progress if verbose else None,
        )

        if verbose:
            elapsed = time.perf_counter() - batch_start
            per_file = elapsed / len(files)
            typer.echo(f"  Batch complete: {elapsed:.1f}s total, "
                       f"{per_file:.1f}s/file avg ({len(files)/elapsed:.1f} files/min)",
                       err=True)
    else:
        # Sequential processing
        results = []
        for file in files:
            if verbose:
                typer.echo(f"Processing: {file.name}", err=True)

            try:
                result = process_document(
                    file, model=model, debug=debug,
                    passes=passes, ocr_engine=engine,
                )
                result["_file"] = str(file)

                if verbose:
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
                if verbose:
                    typer.echo(f"  Error: {e}", err=True)

            results.append(result)

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
    reset: bool = typer.Option(False, "--reset", help="Reset all counters for current month"),
    history: bool = typer.Option(False, "--history", "-H", help="Show historical usage log"),
    page: int = typer.Option(1, "--page", help="Page number for history view"),
    sync: bool = typer.Option(False, "--sync", help="Set counters from online dashboard values"),
    set_cv: int = typer.Option(None, "--set-cv", help="Set Cloud Vision call count"),
    set_ds_hit: int = typer.Option(None, "--set-ds-hit", help="Set DeepSeek cache hit token count"),
    set_ds_miss: int = typer.Option(None, "--set-ds-miss", help="Set DeepSeek cache miss token count"),
    set_ds_out: int = typer.Option(None, "--set-ds-out", help="Set DeepSeek output token count"),
    set_ds_calls: int = typer.Option(None, "--set-ds-calls", help="Set DeepSeek API call count"),
):
    """Show API usage and estimated costs for the current billing period."""
    from .usage import get_usage, reset_usage, sync_usage, get_history

    if reset:
        reset_usage()
        typer.echo("Usage counters reset.")
        return

    # ── Sync: interactive or flag-based ──
    if sync:
        _run_sync_interactive()
        return

    if any(v is not None for v in (set_cv, set_ds_hit, set_ds_miss, set_ds_out, set_ds_calls)):
        sync_usage(cv_calls=set_cv, ds_cache_hit=set_ds_hit, ds_cache_miss=set_ds_miss,
                   ds_output=set_ds_out, ds_calls=set_ds_calls)
        typer.echo("Usage counters updated.")
        # Fall through to display updated stats

    # ── History view ──
    if history:
        _show_history(page, json_output)
        return

    # ── Current month view ──
    stats = get_usage()

    if json_output:
        typer.echo(json.dumps(stats, indent=2))
        return

    cv = stats["cloud_vision"]
    ds = stats["deepseek"]
    docs = stats["documents"]

    typer.echo(f"API Usage — {stats['month']}  ({stats['days_until_reset']} days until reset)")
    typer.echo("─" * 52)

    # Documents
    typer.echo(f"\n  Documents")
    typer.echo(f"    Total processed:  {docs['total_processed']:,}")
    typer.echo(f"    Unique files:     {docs['unique_processed']:,}")

    # Cloud Vision
    typer.echo(f"\n  Google Cloud Vision")
    typer.echo(f"    Calls:          {cv['calls']:,}")
    typer.echo(f"    Free tier:      {cv['remaining_free']:,} / {cv['free_limit']:,} remaining")
    if cv["billable_calls"] > 0:
        typer.echo(f"    Billable calls: {cv['billable_calls']:,}")
        typer.echo(f"    Est. cost:      ${cv['est_cost_usd']:.4f}")
    else:
        typer.echo(f"    Est. cost:      $0.00 (within free tier)")

    # DeepSeek
    typer.echo(f"\n  DeepSeek API")
    typer.echo(f"    Calls:            {ds['calls']:,}")
    typer.echo(f"    Cache hit tokens: {ds['cache_hit_tokens']:,}  (@ $0.028/1M)")
    typer.echo(f"    Cache miss tokens:{ds['cache_miss_tokens']:>11,}  (@ $0.28/1M)")
    typer.echo(f"    Output tokens:    {ds['output_tokens']:,}  (@ $0.42/1M)")
    typer.echo(f"    Est. cost:        ${ds['est_cost_usd']:.4f}")

    # Total
    typer.echo(f"\n{'─' * 52}")
    typer.echo(f"  Total est. cost:  ${stats['total_est_cost_usd']:.4f}")


def _run_sync_interactive():
    """Interactive sync: prompt user for dashboard values."""
    from .usage import sync_usage, get_usage

    current = get_usage()

    typer.echo("Sync usage counters with your online dashboard values.")
    typer.echo("Press Enter to keep current value, or type a new number.\n")

    cv_cur = current["cloud_vision"]["calls"]
    ds_calls_cur = current["deepseek"]["calls"]
    ds_hit_cur = current["deepseek"]["cache_hit_tokens"]
    ds_miss_cur = current["deepseek"]["cache_miss_tokens"]
    ds_out_cur = current["deepseek"]["output_tokens"]

    cv_input = typer.prompt(
        f"  Cloud Vision calls [{cv_cur:,}]",
        default="", show_default=False,
    ).strip()

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

    cv_val = int(cv_input) if cv_input else None
    ds_calls_val = int(ds_calls_input) if ds_calls_input else None
    ds_hit_val = int(ds_hit_input) if ds_hit_input else None
    ds_miss_val = int(ds_miss_input) if ds_miss_input else None
    ds_out_val = int(ds_out_input) if ds_out_input else None

    if all(v is None for v in (cv_val, ds_calls_val, ds_hit_val, ds_miss_val, ds_out_val)):
        typer.echo("\nNo changes made.")
        return

    sync_usage(cv_calls=cv_val, ds_calls=ds_calls_val,
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
        total_tokens = ds.get("input_tokens", 0) + ds.get("output_tokens", 0)
        cost = entry.get("est_cost_usd", 0)
        doc_count = docs.get("total_processed", 0)

        typer.echo(f"  {entry['month']:<10} {cv.get('calls', 0):>10,} "
                   f"{total_tokens:>14,} {doc_count:>7,} ${cost:>9.4f}")

    typer.echo(f"{'─' * 62}")
    typer.echo(f"  Lifetime cost: ${result['lifetime_cost_usd']:.4f}")

    if result["total_pages"] > 1:
        typer.echo(f"\n  Use --page N to navigate (1-{result['total_pages']})")


if __name__ == "__main__":
    app()
