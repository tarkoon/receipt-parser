"""Export flagged Paper Ledger PROD receipts into receipt-parser fixtures.

The exporter is intentionally deterministic: production saved rows are adapted
into the stripped fixture template shape, then images are copied from Stardust
only when --apply is explicitly provided.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
TRUTH_TEMPLATE = FIXTURES_DIR / "_truth_template.json"
DEFAULT_MANIFEST = REPO_ROOT / "local" / "prod_flagged_receipts_manifest.json"

DEFAULT_STARDUST_HOST = "stardust"
DEFAULT_REMOTE_COMPOSE_DIR = "/home/tarkoon/apps/paper-ledger-deploy/pi"
DEFAULT_REMOTE_STORAGE_ROOT = "/home/tarkoon/data/paper-ledger/storage"
DEFAULT_POSTGRES_USER = "paper_ledger"
DEFAULT_POSTGRES_DB = "paper_ledger"

SCALAR_FIELDS = (
    "document_type",
    "merchant",
    "date",
    "time",
    "location",
    "currency",
    "total",
    "payment_method",
    "account_number",
    "points_used",
    "amount_paid",
    "subtotal",
    "service_type",
    "payer",
    "payment_reference",
)

MONEY_FIELDS = {"total", "points_used", "amount_paid", "subtotal"}
LINE_ITEM_MONEY_FIELDS = {"unit_price", "total", "discount"}
TAX_MONEY_FIELDS = {"amount"}
CURRENCY_MINOR_UNITS = {"JPY": 1, "USD": 100, "EUR": 100}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export flagged Paper Ledger PROD receipts into fixture truth files."
    )
    parser.add_argument("--source", choices=("prod",), default="prod")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Preview exports without writing files.")
    mode.add_argument("--apply", action="store_true", help="Write fixture images, truth files, and manifest.")

    parser.add_argument("--limit", type=int, help="Maximum number of changed receipts to export.")
    parser.add_argument(
        "--receipt-id",
        action="append",
        default=[],
        help="Export only this production receipt id. Can be provided multiple times.",
    )
    parser.add_argument("--start-number", type=int, help="First fixture number to consider.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing fixture files.")
    parser.add_argument("--stardust-host", default=DEFAULT_STARDUST_HOST)
    parser.add_argument("--remote-compose-dir", default=DEFAULT_REMOTE_COMPOSE_DIR)
    parser.add_argument("--remote-storage-root", default=DEFAULT_REMOTE_STORAGE_ROOT)
    parser.add_argument("--postgres-user", default=DEFAULT_POSTGRES_USER)
    parser.add_argument("--postgres-db", default=DEFAULT_POSTGRES_DB)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def debug(message: str, *, enabled: bool) -> None:
    if enabled:
        print(f"[debug] {message}", file=sys.stderr, flush=True)


def strip_template_comments(value: Any) -> Any:
    """Remove prompt/comment keys while preserving JSON object order."""
    if isinstance(value, dict):
        return {
            key: strip_template_comments(item)
            for key, item in value.items()
            if key != "_llm_prompt" and not key.startswith("_comment")
        }
    if isinstance(value, list):
        return [strip_template_comments(item) for item in value]
    return value


def load_template_shape(path: Path = TRUTH_TEMPLATE) -> dict[str, Any]:
    return strip_template_comments(json.loads(path.read_text(encoding="utf-8")))


def money_from_minor(value: Any, currency: str) -> float | int | None:
    if value is None:
        return None
    multiplier = CURRENCY_MINOR_UNITS.get(currency or "JPY", 1)
    converted = value / multiplier
    return int(converted) if converted == int(converted) else converted


def normalize_date(value: Any) -> str | None:
    return str(value) if value is not None else None


def prod_receipt_to_truth(row: dict[str, Any], template: dict[str, Any] | None = None) -> dict[str, Any]:
    """Adapt one Paper Ledger saved receipt row into fixture-template shape."""
    template_shape = template if template is not None else load_template_shape()
    currency = row.get("currency") or "JPY"
    truth: dict[str, Any] = {}

    for key, template_value in template_shape.items():
        if key in SCALAR_FIELDS:
            value = row.get(key)
            if key == "currency":
                value = value or "JPY"
            elif key in MONEY_FIELDS:
                value = money_from_minor(value, currency)
            elif key == "date":
                value = normalize_date(value)
            truth[key] = value
        elif key == "line_items":
            truth[key] = adapt_line_items(row.get("line_items") or [], currency, template_value)
        elif key == "taxes":
            truth[key] = adapt_taxes(row.get("tax_entries") or [], currency)
        elif key == "billing_period":
            truth[key] = adapt_billing_period(row.get("billing_period"), template_value)
        elif key == "usage":
            truth[key] = adapt_usage(row.get("usage_data"), template_value)
        else:
            truth[key] = None

    return truth


def adapt_line_items(
    line_items: list[dict[str, Any]],
    currency: str,
    template_value: Any,
) -> list[dict[str, Any]]:
    if not line_items:
        return []
    item_template = template_value[0] if isinstance(template_value, list) and template_value else {}
    output = []
    for item in line_items:
        adapted: dict[str, Any] = {}
        for key in item_template:
            value = item.get(key)
            if key in LINE_ITEM_MONEY_FIELDS:
                value = money_from_minor(value, currency)
            adapted[key] = value
        output.append(adapted)
    return output


def adapt_taxes(tax_entries: list[dict[str, Any]], currency: str) -> list[dict[str, Any]]:
    taxes = []
    for tax in tax_entries:
        taxes.append(
            {
                "rate": tax.get("rate"),
                "label": tax.get("label"),
                "amount": money_from_minor(tax.get("amount"), currency),
            }
        )
    return taxes


def adapt_billing_period(value: dict[str, Any] | None, template_value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: normalize_date(
            (value or {}).get("start_date" if key == "start" else "end_date")
        )
        for key in template_value
    }


def adapt_usage(value: dict[str, Any] | None, template_value: dict[str, Any]) -> dict[str, Any]:
    source = value or {}
    return {key: source.get(key) for key in template_value}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def checksum_truth(truth: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(truth).encode("utf-8")).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "source": "paper-ledger-prod", "receipts": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    data.setdefault("version", 1)
    data.setdefault("source", "paper-ledger-prod")
    data.setdefault("receipts", {})
    if not isinstance(data["receipts"], dict):
        raise ValueError(f"Manifest receipts must be a JSON object: {path}")
    return data


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def manifest_entry_current(
    entry: dict[str, Any] | None,
    row: dict[str, Any],
    truth_checksum: str,
    *,
    fixture_files_exist: bool,
) -> bool:
    if not entry or not fixture_files_exist:
        return False
    return (
        entry.get("updated_at") == row.get("updated_at")
        and entry.get("checksum") == truth_checksum
    )


def make_manifest_entry(
    row: dict[str, Any],
    fixture_name: str,
    truth_checksum: str,
    exported_at: str,
) -> dict[str, Any]:
    return {
        "receipt_id": row["id"],
        "image_path": row["image_path"],
        "fixture": fixture_name,
        "updated_at": row.get("updated_at"),
        "exported_at": exported_at,
        "checksum": truth_checksum,
    }


def fixture_paths(fixture_name: str) -> tuple[Path, Path]:
    return (
        FIXTURES_DIR / f"{fixture_name}.jpg",
        FIXTURES_DIR / f"{fixture_name}_truth.json",
    )


def remote_storage_path(storage_root: str, image_path: str) -> str:
    if image_path.startswith("/"):
        return image_path
    return f"{storage_root.rstrip('/')}/{image_path}"


def entry_fixture_files_exist(entry: dict[str, Any] | None) -> bool:
    if not entry or not entry.get("fixture"):
        return False
    image_path, truth_path = fixture_paths(entry["fixture"])
    return image_path.exists() and truth_path.exists()


def existing_fixture_numbers() -> set[int]:
    numbers: set[int] = set()
    for path in list(FIXTURES_DIR.glob("receipt_*.jpg")) + list(FIXTURES_DIR.glob("receipt_*_truth.json")):
        stem = path.stem.replace("_truth", "")
        parts = stem.split("_")
        if len(parts) == 2 and parts[0] == "receipt":
            try:
                numbers.add(int(parts[1]))
            except ValueError:
                pass
    return numbers


def allocate_fixture_names(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    start_number: int | None,
    overwrite: bool,
) -> dict[str, str]:
    used_numbers = existing_fixture_numbers()
    assigned: dict[str, str] = {}
    next_number = start_number if start_number is not None else (max(used_numbers) + 1 if used_numbers else 1)

    for row in rows:
        entry = manifest["receipts"].get(row["id"])
        if entry and entry.get("fixture"):
            assigned[row["id"]] = entry["fixture"]
            continue

        while not overwrite and next_number in used_numbers:
            next_number += 1
        assigned[row["id"]] = f"receipt_{next_number}"
        used_numbers.add(next_number)
        next_number += 1

    return assigned


def ensure_can_write_fixture(image_path: Path, truth_path: Path, *, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [str(path) for path in (image_path, truth_path) if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing fixture file(s): " + ", ".join(existing))


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(text)
    temp_path.replace(path)


def write_bytes_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(content)
    temp_path.replace(path)


def build_flagged_receipts_sql(receipt_ids: list[str], limit: int | None) -> str:
    where = ["r.flagged = true", "r.image_path IS NOT NULL"]
    if receipt_ids:
        quoted = ", ".join(sql_quote(receipt_id) for receipt_id in receipt_ids)
        where.append(f"r.id IN ({quoted})")
    limit_sql = f"\nLIMIT {int(limit)}" if limit is not None else ""
    where_sql = " AND ".join(where)
    return f"""
WITH selected AS (
  SELECT r.*
  FROM receipts r
  WHERE {where_sql}
  ORDER BY r.created_at ASC, r.id ASC
  {limit_sql}
),
payloads AS (
  SELECT
    s.created_at,
    s.id,
    jsonb_build_object(
      'id', s.id,
      'document_type', s.document_type,
      'merchant', s.merchant,
      'date', s.date,
      'time', s.time,
      'location', s.location,
      'currency', COALESCE(s.currency, 'JPY'),
      'total', s.total,
      'payment_method', s.payment_method,
      'account_number', s.account_number,
      'points_used', s.points_used,
      'amount_paid', s.amount_paid,
      'subtotal', s.subtotal,
      'service_type', s.service_type,
      'payer', s.payer,
      'payment_reference', s.payment_reference,
      'image_path', s.image_path,
      'updated_at', s.updated_at,
      'line_items', COALESCE((
        SELECT jsonb_agg(
          jsonb_build_object(
            'description', li.description,
            'qty', li.qty,
            'unit_price', li.unit_price,
            'total', li.total,
            'tax_category', li.tax_category,
            'discount', li.discount,
            'discount_rate', li.discount_rate
          )
          ORDER BY COALESCE(li.sort_order, 1000000), li.id
        )
        FROM line_items li
        WHERE li.receipt_id = s.id
      ), '[]'::jsonb),
      'tax_entries', COALESCE((
        SELECT jsonb_agg(
          jsonb_build_object(
            'rate', te.rate,
            'label', te.label,
            'amount', te.amount
          )
          ORDER BY te.rate, te.label, te.id
        )
        FROM tax_entries te
        WHERE te.receipt_id = s.id
      ), '[]'::jsonb),
      'billing_period', (
        SELECT jsonb_build_object(
          'start_date', bp.start_date,
          'end_date', bp.end_date
        )
        FROM billing_periods bp
        WHERE bp.receipt_id = s.id
        LIMIT 1
      ),
      'usage_data', (
        SELECT jsonb_build_object(
          'amount', ud.amount,
          'unit', ud.unit,
          'cost_per', ud.cost_per,
          'meter_previous', ud.meter_previous,
          'meter_current', ud.meter_current
        )
        FROM usage_data ud
        WHERE ud.receipt_id = s.id
        LIMIT 1
      )
    ) AS payload
  FROM selected s
)
SELECT COALESCE(jsonb_agg(payload ORDER BY created_at ASC, id ASC), '[]'::jsonb)::text
FROM payloads;
"""


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def fetch_prod_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    sql = build_flagged_receipts_sql(args.receipt_id, args.limit)
    remote_command = (
        f"cd {shlex.quote(args.remote_compose_dir)} && "
        "docker compose --env-file .env exec -T postgres "
        f"psql -U {shlex.quote(args.postgres_user)} -d {shlex.quote(args.postgres_db)} "
        "-X -q -t -A -v ON_ERROR_STOP=1"
    )
    debug(remote_command, enabled=args.debug)
    result = subprocess.run(
        ["ssh", args.stardust_host, remote_command],
        input=sql,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to query Stardust Postgres:\n{result.stderr.strip()}")
    output = result.stdout.strip()
    debug(output[:500], enabled=args.debug)
    return json.loads(output or "[]")


def read_remote_file(host: str, remote_path: str, *, debug_enabled: bool) -> bytes:
    command = f"cat {shlex.quote(remote_path)}"
    debug(f"ssh {host} {command}", enabled=debug_enabled)
    result = subprocess.run(
        ["ssh", host, command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to read remote image {remote_path}:\n"
            f"{result.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return result.stdout


def export_rows(args: argparse.Namespace, rows: list[dict[str, Any]]) -> int:
    template = load_template_shape()
    manifest = load_manifest(args.manifest)

    candidates: list[dict[str, Any]] = []
    for row in rows:
        truth = prod_receipt_to_truth(row, template)
        truth_checksum = checksum_truth(truth)
        entry = manifest["receipts"].get(row["id"])
        current = manifest_entry_current(
            entry,
            row,
            truth_checksum,
            fixture_files_exist=entry_fixture_files_exist(entry),
        )
        if current:
            continue
        candidates.append({"row": row, "truth": truth, "checksum": truth_checksum})

    if args.limit is not None:
        candidates = candidates[: args.limit]

    assignments = allocate_fixture_names(
        [candidate["row"] for candidate in candidates],
        manifest,
        start_number=args.start_number,
        overwrite=args.overwrite,
    )

    print(f"Flagged rows fetched : {len(rows)}")
    print(f"Changed/new exports : {len(candidates)}")
    print(f"Mode                : {'apply' if args.apply else 'dry-run'}")
    print()

    exported = 0
    exported_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    for index, candidate in enumerate(candidates, 1):
        row = candidate["row"]
        fixture_name = assignments[row["id"]]
        image_path, truth_path = fixture_paths(fixture_name)
        remote_image = remote_storage_path(args.remote_storage_root, row["image_path"])

        print(f"[{index}/{len(candidates)}] {row['id']} -> {fixture_name}")
        print(f"  image: {row['image_path']}")
        print(f"  truth checksum: {candidate['checksum'][:12]}")

        if args.dry_run:
            continue

        ensure_can_write_fixture(image_path, truth_path, overwrite=args.overwrite)
        image_bytes = read_remote_file(args.stardust_host, remote_image, debug_enabled=args.debug)
        write_bytes_atomic(image_path, image_bytes)
        write_text_atomic(
            truth_path,
            json.dumps(candidate["truth"], ensure_ascii=False, indent=2) + "\n",
        )
        manifest["receipts"][row["id"]] = make_manifest_entry(
            row,
            fixture_name,
            candidate["checksum"],
            exported_at,
        )
        save_manifest(args.manifest, manifest)
        exported += 1
        print(f"  wrote: {image_path.relative_to(REPO_ROOT)}")
        print(f"  wrote: {truth_path.relative_to(REPO_ROOT)}")

    return exported


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be greater than 0")
    if args.start_number is not None and args.start_number < 1:
        raise SystemExit("--start-number must be greater than 0")

    rows = fetch_prod_rows(args)
    exported = export_rows(args, rows)
    if args.apply:
        print(f"\nDone. Exported {exported} fixture(s).")


if __name__ == "__main__":
    main()
