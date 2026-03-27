"""Quick test of key receipts."""
from pipeline import process_document
from pathlib import Path

for rid in ["receipt_17", "receipt_33", "receipt_34"]:
    img = Path(f"tests/fixtures/{rid}.jpg")
    if not img.exists():
        continue
    r = process_document(img, passes=2)
    print(f"{rid}: total={r.get('total')} sub={r.get('subtotal')} pay={r.get('payment_method')} merchant={r.get('merchant')}")
