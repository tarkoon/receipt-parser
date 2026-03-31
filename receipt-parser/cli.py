"""cli.py — Typer CLI entry point."""

import csv
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import typer

from pipeline import process_document
from extraction import DEFAULT_MODEL

__version__ = "1.2.0"

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
        "invoice_number": result.get("invoice_number", ""),
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
    "payment_method", "invoice_number", "location",
    "item_description", "item_qty", "item_unit_price", "item_total",
    "item_tax_category", "tax_rates", "tax_amounts",
]


@app.command()
def parse(
    input_path: Path = typer.Argument(..., help="Image, PDF, or directory to process"),
    output: Path = typer.Option(None, "--output", "-o", help="Output file (default: stdout)"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m",
                              help="LLM model (default: DeepSeek; prefix 'ollama/' for Ollama)"),
    passes: int = typer.Option(1, "--passes", "-p", min=1, max=3,
                               help="Extraction passes (2+ enables verification)"),
    format: str = typer.Option("json", "--format", "-f", help="Output format: json or csv"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Save debug artifacts"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print OCR text and warnings"),
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

    from ocr import init_cloud_vision
    engine = init_cloud_vision()

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
def usage():
    """Show Cloud Vision API usage for the current month."""
    from ocr import get_api_usage
    stats = get_api_usage()
    typer.echo(f"Cloud Vision API Usage ({stats['month']}):")
    typer.echo(f"  Calls this month: {stats['calls']}")
    typer.echo(f"  Free tier limit:  {stats['free_limit']}")
    typer.echo(f"  Remaining:        {stats['remaining']}")


if __name__ == "__main__":
    app()
