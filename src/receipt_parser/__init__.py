"""receipt_parser — Extract structured data from receipts and invoices."""

__version__ = "3.1.0"

# Public progress-callback contract — re-exported here so consumers don't have
# to import from the internal pipeline module.
from .pipeline import (
    StageName,
    StageCallback,
    PipelineCancelled,
    process_document,
    process_ocr_text,
    process_batch,
)

__all__ = [
    "StageName",
    "StageCallback",
    "PipelineCancelled",
    "process_document",
    "process_ocr_text",
    "process_batch",
]
