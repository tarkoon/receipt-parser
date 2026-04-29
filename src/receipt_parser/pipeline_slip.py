"""pipeline_slip.py — Payment slip-specific post-processing.

Extracted from pipeline.py for maintainability.
"""

import re


def postprocess_payment_slip(extracted: dict, unified_text: str, raw_text: str = "") -> dict:
    """Apply payment slip-specific post-processing to the LLM extraction.

    Args:
        extracted: LLM extraction dict
        unified_text: Normalized text (after barcode stripping)
        raw_text: Original text before barcode stripping (for reference extraction)
    """
    # Payment reference: prefer 通番-like long digit sequences over LLM value
    if raw_text:
        m = re.search(r'手番\s*(\d+)', raw_text)
        if m:
            extracted["payment_reference"] = m.group(1)
        else:
            lines = raw_text.split('\n')
            for line in lines:
                stripped = line.strip()
                if re.match(r'^\d{10,}$', stripped):
                    extracted["payment_reference"] = stripped
                    break

    # Date: null out dates from 発行日 (billing issuance date, not payment date)
    if extracted.get("date"):
        text = unified_text or raw_text or ""
        m = re.search(r'発行日\s*(20\d{2})[年/]0?(\d{1,2})[月/]0?(\d{1,2})', text)
        if m:
            billing_date = f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
            if extracted["date"] == billing_date:
                extracted["date"] = None

    return extracted
