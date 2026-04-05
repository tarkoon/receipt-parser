"""pipeline_bill.py — Utility bill-specific post-processing.

Extracted from pipeline.py for maintainability.
"""

import re

from .patterns import era_to_western_year


def postprocess_utility_bill(extracted: dict, unified_text: str) -> dict:
    """Apply utility bill-specific post-processing to the LLM extraction."""
    # Check for convenience store payment evidence (overrides bank_payment)
    paid_at_store = bool(re.search(
        r'ローソン|セブン|ファミリーマート|コンビニ|収納代行|領収.*いたしました',
        unified_text,
    ))
    if paid_at_store:
        extracted["payment_method"] = "cash"
    elif re.search(r'口座引落|口座振替|振替させて', unified_text):
        extracted["payment_method"] = "bank_payment"
    elif re.search(r'領入済|収納済', unified_text):
        extracted["payment_method"] = "cash"

    # Service type: bills with both 水道 and 下水道 are water bills
    if extracted.get("service_type") == "sewage" and re.search(r'水道', unified_text):
        water_hits = len(re.findall(r'水道', unified_text))
        sewage_hits = len(re.findall(r'下水道', unified_text))
        if water_hits > sewage_hits:
            extracted["service_type"] = "water"

    # Date: prefer 領収日付 stamp date (often formatted as 'YY.M.D)
    ryoshu_date = re.search(r"領収日付[:\s]*'?(\d{2})\.(\d{1,2})\.(\d{1,2})", unified_text)
    if not ryoshu_date:
        ryoshu_date = re.search(r"'(\d{2})\.(\d{1,2})\.(\d{1,2})", unified_text)
    if ryoshu_date:
        y = int(ryoshu_date.group(1))
        year = 2000 + y if y < 100 else y
        extracted["date"] = f"{year:04d}-{int(ryoshu_date.group(2)):02d}-{int(ryoshu_date.group(3)):02d}"

    return extracted
