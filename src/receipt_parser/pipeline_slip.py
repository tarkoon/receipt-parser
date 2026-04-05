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
    # Extract payment_reference from raw text if LLM missed it
    if not extracted.get("payment_reference") and raw_text:
        # Look for 手番 label followed by digits (common on Japanese payment slips)
        m = re.search(r'手番\s*(\d+)', raw_text)
        if m:
            extracted["payment_reference"] = m.group(1)
        else:
            # Look for standalone long digit sequences that could be reference numbers
            # (these get stripped by strip_barcode_lines)
            lines = raw_text.split('\n')
            for line in lines:
                stripped = line.strip()
                if re.match(r'^\d{10,}$', stripped):
                    extracted["payment_reference"] = stripped
                    break

    return extracted
