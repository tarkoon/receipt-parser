"""usage.py — Unified API usage tracking for Cloud Vision and DeepSeek.

Tracks calls, tokens, estimated costs, and document counts in a single
JSON file with automatic monthly rollover and historical archiving.

Cloud Vision usage is auto-fetched from the GCP Monitoring API on each
CLI query, so the local counter always matches the real billing data.

Thread-safe for concurrent pipeline use (process_batch).
"""

import hashlib
import json
import os
import threading
import warnings
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

# Data directory: .data/ at project root (sibling of src/)
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / ".data"
_DATA_DIR.mkdir(exist_ok=True)

_USAGE_FILE = _DATA_DIR / "api_usage.json"
_HISTORY_FILE = _DATA_DIR / "api_usage_history.json"
_SETTINGS_FILE = _DATA_DIR / "api_usage_settings.json"
_lock = threading.Lock()

# ── Pricing constants ────────────────────────────────────────────────

# Google Cloud Vision: 1000 free calls/month, then $1.50/1000
CLOUD_VISION_FREE_TIER = 1000
CLOUD_VISION_COST_PER_1K = 1.50  # USD per 1000 calls beyond free tier

# DeepSeek V3.2 (deepseek-chat): per-million-token pricing (as of 2026-04)
DEEPSEEK_CACHE_HIT_COST_PER_M = 0.028  # USD per 1M cached input tokens
DEEPSEEK_CACHE_MISS_COST_PER_M = 0.28  # USD per 1M non-cached input tokens
DEEPSEEK_OUTPUT_COST_PER_M = 0.42      # USD per 1M output tokens


# ── Settings (billing period config) ─────────────────────────────────

def _load_settings() -> dict:
    """Load user settings (billing start day, etc.)."""
    if _SETTINGS_FILE.exists():
        try:
            return json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, TypeError):
            pass
    return {"billing_start_day": 1}


def _save_settings(settings: dict):
    _SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def get_billing_start_day() -> int:
    return _load_settings().get("billing_start_day", 1)


def set_billing_start_day(day: int):
    """Set the day of month when the billing period starts (1-28)."""
    day = max(1, min(28, day))
    settings = _load_settings()
    settings["billing_start_day"] = day
    _save_settings(settings)


# ── Billing period helpers ───────────────────────────────────────────

def get_billing_period(ref_date: date | None = None) -> tuple[date, date]:
    """Return (start, end) of the current billing period.

    With billing_start_day=1: Apr 1 → Apr 30
    With billing_start_day=15: Mar 15 → Apr 14 (if today < Apr 15)
                                Apr 15 → May 14 (if today >= Apr 15)
    """
    today = ref_date or date.today()
    start_day = get_billing_start_day()

    if today.day >= start_day:
        # Current period started this month
        start = today.replace(day=start_day)
    else:
        # Current period started last month
        first_of_month = today.replace(day=1)
        prev_month = first_of_month - timedelta(days=1)
        start = prev_month.replace(day=start_day)

    # End is the day before next period starts
    if start.month == 12:
        next_start = date(start.year + 1, 1, start_day)
    else:
        next_start = date(start.year, start.month + 1, start_day)
    end = next_start - timedelta(days=1)

    return start, end


def get_billing_period_label() -> str:
    """Return a human-readable label for the current billing period."""
    start, end = get_billing_period()
    return f"{start.strftime('%b %d')} - {end.strftime('%b %d, %Y')}"


def _billing_period_month_key() -> str:
    """Return the month key for the current billing period (YYYY-MM of start)."""
    start, _ = get_billing_period()
    return start.strftime("%Y-%m")


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


def _ensure_ds_keys(ds: dict):
    """Ensure DeepSeek dict has cache_hit/cache_miss keys (migrate from old format)."""
    if "cache_hit_tokens" not in ds:
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
        "ds_cost": round(ds_cost, 4),
        "total_cost": round(cv_cost + ds_cost, 4),
    }


# ── Persistence ──────────────────────────────────────────────────────

def _load() -> dict:
    """Load usage data, archiving and resetting if the billing period has rolled over."""
    month = _billing_period_month_key()
    if _USAGE_FILE.exists():
        try:
            data = json.loads(_USAGE_FILE.read_text(encoding="utf-8"))
            if data.get("month") == month:
                if "deepseek" not in data:
                    data["deepseek"] = _empty_month(month)["deepseek"]
                else:
                    _ensure_ds_keys(data["deepseek"])
                if "cloud_vision" not in data:
                    data["cloud_vision"] = _empty_month(month)["cloud_vision"]
                if "documents" not in data:
                    data["documents"] = _empty_month(month)["documents"]
                return data
            _archive_month(data)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return _empty_month(month)


def _save(data: dict):
    _USAGE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _archive_month(data: dict):
    """Append a completed period's data to the history file."""
    history = _load_history()
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
    if _HISTORY_FILE.exists():
        try:
            data = json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, TypeError):
            pass
    return []


def _save_history(history: list[dict]):
    _HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")


# ── Cloud Vision auto-fetch from GCP Monitoring API ──────────────────

def fetch_cloud_vision_usage() -> tuple[int | None, str | None]:
    """Fetch Cloud Vision API call count from GCP Monitoring API.

    Returns (call_count, error_message). call_count is None on failure.
    Uses the current billing period for the query window.
    """
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        return None, "GOOGLE_CLOUD_PROJECT env var not set"

    try:
        import google.auth
        import google.auth.transport.requests
        import requests

        # Suppress the ADC quota project warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*quota project.*")
            creds, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/monitoring.read"]
            )
            auth_req = google.auth.transport.requests.Request()
            creds.refresh(auth_req)
            token = creds.token

        start, end = get_billing_period()
        # Query from billing period start to now (end of today)
        start_time = f"{start.isoformat()}T00:00:00Z"
        end_time = f"{(date.today() + timedelta(days=1)).isoformat()}T00:00:00Z"

        url = f"https://monitoring.googleapis.com/v3/projects/{project_id}/timeSeries"
        headers = {"Authorization": f"Bearer {token}"}
        params = {
            "filter": (
                'metric.type="serviceruntime.googleapis.com/api/request_count"'
                ' AND resource.labels.service="vision.googleapis.com"'
            ),
            "interval.startTime": start_time,
            "interval.endTime": end_time,
        }

        total = 0
        while True:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            for series in data.get("timeSeries", []):
                for point in series.get("points", []):
                    total += int(point["value"]["int64Value"])

            next_token = data.get("nextPageToken")
            if not next_token:
                break
            params["pageToken"] = next_token

        return total, None

    except Exception as e:
        return None, str(e)


def refresh_cloud_vision():
    """Auto-fetch Cloud Vision usage from GCP and update the local counter.

    Returns (calls, error_or_none) for display purposes.
    """
    count, error = fetch_cloud_vision_usage()
    if count is not None:
        with _lock:
            data = _load()
            data["cloud_vision"]["calls"] = count
            _save(data)
    return count, error


# ── Tracking functions (called by ocr.py, llm.py, pipeline.py) ──────

def track_cloud_vision_call():
    """Record one Cloud Vision API call (local counter, supplemented by auto-fetch)."""
    with _lock:
        data = _load()
        data["cloud_vision"]["calls"] += 1
        _save(data)

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
    """Set usage counters to match values from the online dashboards."""
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

def get_usage(auto_fetch_cv: bool = False) -> dict:
    """Return full usage stats with cost estimates.

    Args:
        auto_fetch_cv: If True, refresh Cloud Vision count from GCP API first.
    """
    cv_fetch_error = None
    if auto_fetch_cv:
        _, cv_fetch_error = refresh_cloud_vision()

    data = _load()
    costs = _compute_costs(data)
    docs = data.get("documents", {})

    start, end = get_billing_period()
    days_until_reset = (end - date.today()).days + 1

    cv = data["cloud_vision"]
    ds = data["deepseek"]

    result = {
        "billing_period": f"{start.isoformat()} to {end.isoformat()}",
        "billing_period_label": get_billing_period_label(),
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
    if cv_fetch_error:
        result["_cv_fetch_error"] = cv_fetch_error
    return result


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
    """Manually reset all usage counters for the current billing period."""
    with _lock:
        _save(_empty_month(_billing_period_month_key()))
