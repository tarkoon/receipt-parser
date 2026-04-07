"""usage.py — Unified API usage tracking for Cloud Vision and DeepSeek.

Tracks calls, tokens, estimated costs, and document counts in a single
JSON file with automatic monthly rollover and historical archiving.

Thread-safe for concurrent pipeline use (process_batch).
"""

import hashlib
import json
import threading
from datetime import datetime, date
from pathlib import Path

_USAGE_FILE = Path(__file__).parent / ".api_usage.json"
_HISTORY_FILE = Path(__file__).parent / ".api_usage_history.json"
_lock = threading.Lock()

# ── Pricing constants ────────────────────────────────────────────────

# Google Cloud Vision: 1000 free calls/month, then $1.50/1000
CLOUD_VISION_FREE_TIER = 1000
CLOUD_VISION_COST_PER_1K = 1.50  # USD per 1000 calls beyond free tier

# DeepSeek V3.2 (deepseek-chat): per-million-token pricing (as of 2026-04)
# Input tokens are split into cache hit (discounted) and cache miss (full price)
DEEPSEEK_CACHE_HIT_COST_PER_M = 0.028  # USD per 1M cached input tokens
DEEPSEEK_CACHE_MISS_COST_PER_M = 0.28  # USD per 1M non-cached input tokens
DEEPSEEK_OUTPUT_COST_PER_M = 0.42      # USD per 1M output tokens


# ── Data model ───────────────────────────────────────────────────────

def _empty_month(month: str) -> dict:
    """Return a fresh usage record for a given month."""
    return {
        "month": month,
        "cloud_vision": {
            "calls": 0,
        },
        "deepseek": {
            "calls": 0,
            "cache_hit_tokens": 0,
            "cache_miss_tokens": 0,
            "output_tokens": 0,
        },
        "documents": {
            "total_processed": 0,
            "unique_hashes": [],
        },
    }


def _current_month() -> str:
    return datetime.now().strftime("%Y-%m")


def _ensure_ds_keys(ds: dict):
    """Ensure DeepSeek dict has cache_hit/cache_miss keys (migrate from old format)."""
    if "cache_hit_tokens" not in ds:
        # Migrate: old format had only "input_tokens" — treat as cache_miss
        old_input = ds.pop("input_tokens", 0)
        ds["cache_hit_tokens"] = 0
        ds["cache_miss_tokens"] = old_input
    if "output_tokens" not in ds:
        ds["output_tokens"] = 0
    if "calls" not in ds:
        ds["calls"] = 0


def _compute_costs(data: dict) -> dict:
    """Compute cost estimates from raw usage data."""
    cv = data.get("cloud_vision", {})
    ds = data.get("deepseek", {})
    _ensure_ds_keys(ds)

    cv_calls = cv.get("calls", 0)
    cv_billable = max(0, cv_calls - CLOUD_VISION_FREE_TIER)
    cv_cost = cv_billable * CLOUD_VISION_COST_PER_1K / 1000

    ds_hit_cost = ds.get("cache_hit_tokens", 0) * DEEPSEEK_CACHE_HIT_COST_PER_M / 1_000_000
    ds_miss_cost = ds.get("cache_miss_tokens", 0) * DEEPSEEK_CACHE_MISS_COST_PER_M / 1_000_000
    ds_out_cost = ds.get("output_tokens", 0) * DEEPSEEK_OUTPUT_COST_PER_M / 1_000_000
    ds_cost = ds_hit_cost + ds_miss_cost + ds_out_cost

    return {
        "cv_cost": round(cv_cost, 4),
        "cv_billable": cv_billable,
        "ds_hit_cost": round(ds_hit_cost, 4),
        "ds_miss_cost": round(ds_miss_cost, 4),
        "ds_out_cost": round(ds_out_cost, 4),
        "ds_cost": round(ds_cost, 4),
        "total_cost": round(cv_cost + ds_cost, 4),
    }


# ── Persistence ──────────────────────────────────────────────────────

def _load() -> dict:
    """Load usage data, archiving and resetting if the month has rolled over."""
    month = _current_month()
    if _USAGE_FILE.exists():
        try:
            data = json.loads(_USAGE_FILE.read_text(encoding="utf-8"))
            if data.get("month") == month:
                # Forward-compat: ensure all keys exist
                if "deepseek" not in data:
                    data["deepseek"] = _empty_month(month)["deepseek"]
                else:
                    _ensure_ds_keys(data["deepseek"])
                if "cloud_vision" not in data:
                    data["cloud_vision"] = _empty_month(month)["cloud_vision"]
                if "documents" not in data:
                    data["documents"] = _empty_month(month)["documents"]
                return data
            # Month rolled over — archive old data before resetting
            _archive_month(data)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return _empty_month(month)


def _save(data: dict):
    """Persist usage data to disk."""
    _USAGE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _archive_month(data: dict):
    """Append a completed month's data to the history file."""
    history = _load_history()
    # Avoid duplicate entries
    existing_months = {entry["month"] for entry in history}
    month = data.get("month", "")
    if month and month not in existing_months:
        costs = _compute_costs(data)
        docs = data.get("documents", {})
        ds = data.get("deepseek", {})
        _ensure_ds_keys(ds)
        entry = {
            "month": month,
            "cloud_vision": data.get("cloud_vision", {}),
            "deepseek": {
                "calls": ds.get("calls", 0),
                "cache_hit_tokens": ds.get("cache_hit_tokens", 0),
                "cache_miss_tokens": ds.get("cache_miss_tokens", 0),
                "output_tokens": ds.get("output_tokens", 0),
            },
            "documents": {
                "total_processed": docs.get("total_processed", 0),
                "unique_processed": len(docs.get("unique_hashes", [])),
            },
            "est_cost_usd": costs["total_cost"],
        }
        history.append(entry)
        _save_history(history)


def _load_history() -> list[dict]:
    """Load the historical usage log."""
    if _HISTORY_FILE.exists():
        try:
            data = json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, TypeError):
            pass
    return []


def _save_history(history: list[dict]):
    """Persist the historical usage log."""
    _HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")


# ── Tracking functions (called by ocr.py, llm.py, pipeline.py) ──────

def track_cloud_vision_call():
    """Record one Cloud Vision API call."""
    with _lock:
        data = _load()
        data["cloud_vision"]["calls"] += 1
        _save(data)

    # Warn on stderr if approaching limits
    calls = data["cloud_vision"]["calls"]
    remaining = CLOUD_VISION_FREE_TIER - calls
    if 0 < remaining <= 100:
        import sys
        print(f"WARNING: Cloud Vision: {calls}/{CLOUD_VISION_FREE_TIER} "
              f"free calls used ({remaining} remaining)",
              file=sys.stderr)
    elif remaining <= 0:
        import sys
        print(f"WARNING: Cloud Vision free tier exceeded! "
              f"{calls} calls (limit: {CLOUD_VISION_FREE_TIER}). "
              f"Additional calls will be billed.",
              file=sys.stderr)


def track_deepseek_call(
    cache_hit_tokens: int | None,
    cache_miss_tokens: int | None,
    output_tokens: int | None,
):
    """Record one DeepSeek API call with per-category token counts."""
    with _lock:
        data = _load()
        ds = data["deepseek"]
        ds["calls"] += 1
        ds["cache_hit_tokens"] += cache_hit_tokens or 0
        ds["cache_miss_tokens"] += cache_miss_tokens or 0
        ds["output_tokens"] += output_tokens or 0
        _save(data)


def track_document(file_path: str | Path):
    """Record a document processed. Tracks total and unique (by file hash)."""
    path = Path(file_path)
    file_hash = ""
    if path.exists():
        file_hash = hashlib.md5(path.read_bytes()).hexdigest()[:12]

    with _lock:
        data = _load()
        docs = data["documents"]
        docs["total_processed"] += 1
        if file_hash and file_hash not in docs["unique_hashes"]:
            docs["unique_hashes"].append(file_hash)
        _save(data)


# ── Sync function (set counters from dashboard values) ───────────────

def sync_usage(
    cv_calls: int | None = None,
    ds_cache_hit: int | None = None,
    ds_cache_miss: int | None = None,
    ds_output: int | None = None,
    ds_calls: int | None = None,
):
    """Set usage counters to match values from the online dashboards.

    Only updates fields that are explicitly provided (non-None).
    """
    with _lock:
        data = _load()
        if cv_calls is not None:
            data["cloud_vision"]["calls"] = cv_calls
        if ds_calls is not None:
            data["deepseek"]["calls"] = ds_calls
        if ds_cache_hit is not None:
            data["deepseek"]["cache_hit_tokens"] = ds_cache_hit
        if ds_cache_miss is not None:
            data["deepseek"]["cache_miss_tokens"] = ds_cache_miss
        if ds_output is not None:
            data["deepseek"]["output_tokens"] = ds_output
        _save(data)


# ── Query functions (called by CLI and benchmark) ────────────────────

def get_usage() -> dict:
    """Return full usage stats with cost estimates."""
    data = _load()
    costs = _compute_costs(data)
    docs = data.get("documents", {})

    # Days until billing reset (1st of next month)
    today = date.today()
    if today.month == 12:
        next_reset = date(today.year + 1, 1, 1)
    else:
        next_reset = date(today.year, today.month + 1, 1)
    days_until_reset = (next_reset - today).days

    cv = data["cloud_vision"]
    ds = data["deepseek"]

    return {
        "month": data["month"],
        "days_until_reset": days_until_reset,
        "cloud_vision": {
            "calls": cv["calls"],
            "free_limit": CLOUD_VISION_FREE_TIER,
            "remaining_free": max(0, CLOUD_VISION_FREE_TIER - cv["calls"]),
            "billable_calls": costs["cv_billable"],
            "est_cost_usd": costs["cv_cost"],
        },
        "deepseek": {
            "calls": ds["calls"],
            "cache_hit_tokens": ds["cache_hit_tokens"],
            "cache_miss_tokens": ds["cache_miss_tokens"],
            "output_tokens": ds["output_tokens"],
            "est_cost_usd": costs["ds_cost"],
        },
        "documents": {
            "total_processed": docs.get("total_processed", 0),
            "unique_processed": len(docs.get("unique_hashes", [])),
        },
        "total_est_cost_usd": costs["total_cost"],
    }


def get_history(page: int = 1, page_size: int = 6) -> dict:
    """Return paginated historical usage log (newest first)."""
    history = _load_history()
    history.sort(key=lambda e: e.get("month", ""), reverse=True)

    total = len(history)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = max(1, min(page, total_pages))

    start = (page - 1) * page_size
    end = start + page_size
    entries = history[start:end]

    lifetime_cost = sum(e.get("est_cost_usd", 0) for e in history)

    return {
        "page": page,
        "total_pages": total_pages,
        "total_months": total,
        "entries": entries,
        "lifetime_cost_usd": round(lifetime_cost, 4),
    }


def reset_usage():
    """Manually reset all usage counters for the current month."""
    with _lock:
        _save(_empty_month(_current_month()))
