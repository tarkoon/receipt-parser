"""Quick smoke test for schema, pipeline, and validation changes."""

from receipt_parser.schema import Document, Receipt, LineItem, TaxEntry, BillingPeriod, UsageData

# 1. Receipt alias
assert Receipt is Document, 'Receipt should be alias for Document'

# 2. Basic document creation
r = Document(document_type='receipt', total=1000, currency='JPY')
assert r.document_type == 'receipt'
assert r.total == 1000
assert r.amount_paid is None

# 3. Utility bill with usage + cost_per
u = Document(
    document_type='utility_bill', total=15946, currency='JPY',
    service_type='gas', payment_method='bank_payment',
    usage=UsageData(amount=15.2, unit='m3', cost_per=None, meter_previous=554.2, meter_current=569.4),
    billing_period=BillingPeriod(start='2025-12-05', end='2026-01-05'),
)
assert u.service_type == 'gas'
assert u.usage.amount == 15.2
assert u.usage.cost_per is None
print('  [OK] Schema models')

# 4. Prompt generation per type
from receipt_parser.schema import generate_extraction_prompt, generate_verification_prompt
prompt_r = generate_extraction_prompt('test text', doc_type='receipt')
prompt_u = generate_extraction_prompt('test text', doc_type='utility_bill')
prompt_p = generate_extraction_prompt('test text', doc_type='payment_slip')
assert 'DISCOUNTS' in prompt_r
assert 'meter' in prompt_u.lower()
assert 'payment_slip' in prompt_p
print('  [OK] Type-specific prompts')

# 5. Document type detection
from receipt_parser.pipeline import detect_document_type
assert detect_document_type('小計 ¥1000 合計 ¥1100') == 'receipt'
assert detect_document_type('ガス検針のお知らせ 使用量 15.2m3 ご請求額 基本料金') == 'utility_bill'
assert detect_document_type('払込票受領証 受取人 依頼人') == 'payment_slip'
print('  [OK] Document type detection')

# 6. Type-aware validation
from receipt_parser.validation import validate_receipt
r2 = Document(document_type='receipt', total=1000, subtotal=900, taxes=[TaxEntry(rate='10%', amount=100)])
assert len(validate_receipt(r2)) == 0

u2 = Document(document_type='utility_bill', total=15946)
assert len(validate_receipt(u2)) == 0

p2 = Document(document_type='payment_slip', total=6270, merchant='Test')
assert len(validate_receipt(p2)) == 0

p3 = Document(document_type='payment_slip', total=6270)
warnings = validate_receipt(p3)
assert any('merchant' in w.lower() for w in warnings)
print('  [OK] Type-aware validation')

# 7. Merchant mapping
from receipt_parser.pipeline import _apply_merchant_mapping
result = {'merchant': 'スマートビリングサービス株式会社'}
mapped = _apply_merchant_mapping(result)
assert mapped['merchant'] == 'Bizmo'
assert mapped.get('_category') == 'internet'
print('  [OK] Merchant mapping')

# 8. Ollama schema generation
from receipt_parser.llm import get_ollama_schema
schema = get_ollama_schema()
assert 'properties' in schema
assert 'document_type' in schema['properties']
assert 'usage' in schema['properties']
print('  [OK] Ollama schema generation')

print('\nAll smoke tests passed!')
