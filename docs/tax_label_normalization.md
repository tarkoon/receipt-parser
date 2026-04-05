# Tax Label Normalization: Proposed Canonical Labels

## Purpose

The accuracy tests compare pipeline output tax labels against truth file labels.
Currently there are 12+ distinct label strings across 29 fixtures with taxes.
This document proposes normalizing them to a small canonical set and maps every
existing truth file label to its canonical replacement.

**No truth files have been changed yet.** This document is for review and approval.

---

## Proposed Canonical Label Set

| Canonical Label | Meaning | When to use |
|---|---|---|
| `内税` | Tax-inclusive (tax included in displayed price) | Receipt shows 内税, 内消費税, 内消費税等, 内消費税額, or (N%内) |
| `外税` | Tax-exclusive (tax added on top of displayed price) | Receipt shows 外税, 外税N%, 税抜対象額 + 税額, or 消費税等 with separate addition |
| `非課税` | Tax-exempt / non-taxable | Receipt shows 非課税対象額 (government fees, certificates, etc.) |

### Labels NOT in the canonical set (and why)

| Rejected Label | Reason |
|---|---|
| `内消費税` / `内消費税等` / `内消費税額` | These all mean the same thing as `内税` -- the consumption tax is included in the price. The suffix (税/税等/税額) is formatting variation, not semantic. |
| `外税10.0% 消費税等` | This is a verbose OCR rendering of `外税`. The rate is already captured in the `rate` field. |
| `8%対象 消費税` | The "8%対象" part is the rate (already in `rate`), and the tax is calculated on a displayed subtotal, making it `外税`. |
| `8%税` | Just a short form of the tax amount at 8% rate. The receipt (receipt_29) shows "8%対象 / 8%税" which is an exclusive-style breakdown, so it maps to `外税`. |
| `8%内税` / `10%内税` | Rate-prefixed variants of `内税`. The rate is redundant with the `rate` field. |
| `軽減税率 R` | "Reduced tax rate" -- this describes the rate category, not inclusive/exclusive. The receipt (receipt_30) shows external tax (外税額 line), so it maps to `外税`. |
| `標準税率` | "Standard tax rate" -- same issue, describes rate not method. Maps to `外税` on receipt_30. |
| `非課税対象額` | Verbose form of `非課税`. |
| `内消費税 (軽減税率)` | `内税` with a parenthetical noting it's reduced rate. The rate is already `8%`. |

---

## Fixture-by-Fixture Mapping

### Legend
- **OCR Evidence**: Key tax-related lines from the OCR cache text or receipt image
- **Current**: The label currently in the truth file
- **Proposed**: The canonical label it should become
- **Rationale**: Why this mapping is correct

### Fixtures with taxes (29 of 36)

| # | Fixture | Rate | Current Label | Proposed Label | OCR Evidence | Rationale |
|---|---|---|---|---|---|---|
| 1 | receipt_1 | 10% | `内税` | `内税` | "10%内税対象 / (10%内) ¥65" | Already canonical. Price includes tax. |
| 2 | receipt_2 | 8% | `外税` | `外税` | "外税8%対象額 ¥834 / 外税8% ¥66" | Already canonical. Tax added separately. |
| 3 | receipt_4 | 8% | `外税` | `外税` | Receipt shows 外税8% pattern (業務スーパー). Cache miss but same store format as receipt_2. | Already canonical. |
| 4 | receipt_4 | 10% | `外税` | `外税` | Same receipt, 10% rate line. | Already canonical. |
| 5 | receipt_5 | 8% | `外税` | `外税` | Receipt image shows AEON/マックスバリュ tax-exclusive breakdown. | Already canonical. |
| 6 | receipt_6 | 10% | `外税` | `外税` | Receipt image shows スーパービバホーム with 外税 pattern. | Already canonical. |
| 7 | receipt_7 | 10% | `外税` | `外税` | "(外税10.0%対象額 ¥96) / 10.0% 消費税等 ¥9 / 外税額計 ¥9" | Already canonical. |
| 8 | receipt_8 | 8% | `内税` | `内税` | "(08%対象 ¥665 内税 ¥49)" | Already canonical. Tax included in price. |
| 9 | receipt_8 | 10% | `内税` | `内税` | "(10%対象 ¥1,576 内税 ¥143)" | Already canonical. |
| 10 | receipt_9 | 10% | `外税` | `外税` | "(外税10.0%対象額 ... 10.0% 消費税等 ¥112)" | Already canonical. |
| 11 | receipt_10 | 10% | `外税` | `外税` | "10%税抜対象額 / 10%税額 ¥50". Image shows ダイソー with 税抜 (pre-tax) amounts. | Already canonical. Tax-exclusive (税抜 = before tax). |
| 12 | receipt_11 | 8% | `内税` | `内税` | "(8%対象 ¥1,625) / (内消費税等 ¥120)" | Already canonical. FamilyMart uses 内消費税等 = tax-inclusive. |
| 13 | receipt_11 | 10% | `内税` | `内税` | "(10% 対象 ¥5) / (内消費税等 ¥0)" | Already canonical. |
| 14 | receipt_12 | 8% | `内税` | `内税` | "8%内税対象額 ¥2,014 / 8%税額 ¥149" | Already canonical. KALDI uses 内税 explicitly. |
| 15 | receipt_12 | 10% | `内税` | `内税` | "10%内税対象額 10.00% / 10%内税額 ¥2" | Already canonical. |
| 16 | receipt_13 | 8% | `外税` | `外税` | "(a外8% 対象額 ¥2,156) / a外8% ¥172" | Already canonical. ゆめマート uses 外8% explicitly. |
| 17 | receipt_17 | 10% | `内消費税` | **`内税`** | "内消費税 10.00% ¥362". UNIQLO receipt image confirms: "内消費税額 10.00% ¥362". | Tax included in price (内). Normalize verbose form. |
| 18 | receipt_18 | 10% | `内消費税` | **`内税`** | "(10%対象 ¥1,650) / (内消費税額 ¥150)". HANDS receipt. | Tax included in price (内消費税額 = inclusive consumption tax amount). |
| 19 | receipt_19 | 10% | `内消費税` | **`内税`** | "(内、消費税等 ¥2,961-)" / "10%対象 ¥32,580 消費税 ¥2,961". コジマ receipt. | Tax included in total (内 prefix). |
| 20 | receipt_22 | 8% | `8%対象 消費税` | **`外税`** | "(8%対象 ¥1,167 消費税 ¥93)". Mister Donut receipt shows tax breakdown without 内 prefix. Image confirms separate tax display. | Tax displayed as separate line from subtotal. No 内 marker. The total (¥1,260) = 8% subtotal (¥1,167) + tax (¥93), confirming tax-inclusive presentation but the label pattern matches the way the receipt prints it. Actually looking more carefully: the total is ¥1,260 and the 8% target is ¥1,167 with tax ¥93. 1167 + 93 = 1260 = total. This is tax-inclusive pricing (items priced with tax, then broken out). Should be `内税`. |
| 21 | receipt_23 | 10% | `内消費税等` | **`内税`** | "(10% 対象 ¥948) / (内消費税等 ¥86)". FamilyMart receipt image confirms. | 内消費税等 = tax-inclusive. Normalize to `内税`. |
| 22 | receipt_24 | 10% | `外税10.0% 消費税等` | **`外税`** | "(外税10.0%対象額 ¥4,377) / 10.0% 消費税等 ¥437 / 外税額計 ¥437". スーパービバホーム. | 外税 explicitly stated. Rate info is redundant with `rate` field. |
| 23 | receipt_25 | 10% | `外税10.0% 消費税等` | **`外税`** | "(外税10.0%対象額 ¥1,992) / 10.0% 消費税等 ¥199 / 外税額計 ¥199". Same store format. | Same as receipt_24. |
| 24 | receipt_26 | 8% | `8%内税` | **`内税`** | "※8%内税対象 ¥2,437 / (※8%内) ¥180". サンリブ宗像. | 内税 with rate prefix. Rate already in `rate` field. |
| 25 | receipt_26 | 10% | `10%内税` | **`内税`** | "10%内税対象 ¥1,592 / (10%内) ¥144". Same receipt. | Same reasoning. |
| 26 | receipt_27 | 0% | `非課税対象額` | **`非課税`** | "非課税対象額 ¥250 / (消費税等 ¥0)". 宗像市役所 receipt for 住民票. | Government certificate fee, tax-exempt. Normalize verbose form. |
| 27 | receipt_28 | 10% | `外税10.0% 消費税等` | **`外税`** | "(外税10.0%対象額 ¥1,485) / 10.0% 消費税等 ¥148 / 外税計 ¥148". | Same pattern as receipts 24/25. |
| 28 | receipt_29 | 8% | `8%税` | **`外税`** | "8%対象 ¥600 / 8%税 ¥48 / 合計 ¥648 / (うち消費税等 ¥48)". Small restaurant (チャオ). | Subtotal ¥600 + tax ¥48 = total ¥648. Tax added on top = exclusive. |
| 29 | receipt_30 | 8% | `軽減税率 R` | **`外税`** | "外税額 / 合計 / (うち消費税等 / 内訳 10% ... R 8% / R印が軽減税率対象商品となります". 小倉井筒屋. | Receipt explicitly says 外税額. "R" marks reduced-rate items. |
| 30 | receipt_30 | 10% | `標準税率` | **`外税`** | Same receipt, 10% portion. "内訳 10% / 22 / 2" (taxable ¥22, tax ¥2). | Same receipt, same tax method. |
| 31 | receipt_31 | 0% | `非課税対象額` | **`非課税`** | "非課税対象額 ¥300 / (消費税等 ¥0)". 宗像市役所 for 納税証明. | Government fee, tax-exempt. |
| 32 | receipt_32 | 0% | `非課税対象額` | **`非課税`** | "非課税対象額 ¥250 / (消費税等 ¥0)". 宗像市役所 for 所得課税証明. | Government fee, tax-exempt. |
| 33 | receipt_33 | 10% | `内税` | `内税` | "内税 ¥9,061 / 10.0% ¥99,670". clickcycle.com bicycle shop. | Already canonical. |
| 34 | receipt_35 | 8% | `内消費税 (軽減税率)` | **`内税`** | "(内消費税 ¥84) / 8%対象 ¥1,140 / (内消費税 ¥84)". McDonald's. Image confirms: "(内消費税 ¥84)". | 内消費税 = tax-inclusive. "(軽減税率)" is redundant with rate=8%. |
| 35 | receipt_36 | 10% | `内消費税等` | **`内税`** | "(内、消費税等 (10.00%) 438円)". ENEOS gas station. | 内 prefix = tax-inclusive. |

### Correction: receipt_22

Looking more carefully at receipt_22 (Mister Donut): The receipt shows `(10%対象 ¥0 消費税 ¥0)` and `(8%対象 ¥1,167 消費税 ¥93)`. The total is ¥1,260. Since 1,167 + 93 = 1,260 = total, the prices shown on the receipt INCLUDE tax, and the receipt is breaking out the tax-inclusive amounts. This is `内税` behavior (tax included in displayed total, then decomposed). The receipt does not use the 外税 label anywhere.

**Updated mapping for receipt_22: `8%対象 消費税` -> `内税`**

---

## Summary of Changes

### Labels that stay the same (no change needed): 16 entries
- `内税` (receipts 1, 8x2, 11x2, 12x2, 33)
- `外税` (receipts 2, 4x2, 5, 6, 7, 9, 10, 13)

### Labels to normalize: 19 entries

| Current Label | Proposed | Count | Fixtures |
|---|---|---|---|
| `内消費税` | `内税` | 3 | receipt_17, receipt_18, receipt_19 |
| `内消費税等` | `内税` | 2 | receipt_23, receipt_36 |
| `内消費税 (軽減税率)` | `内税` | 1 | receipt_35 |
| `8%対象 消費税` | `内税` | 1 | receipt_22 |
| `8%内税` | `内税` | 1 | receipt_26 (8% entry) |
| `10%内税` | `内税` | 1 | receipt_26 (10% entry) |
| `外税10.0% 消費税等` | `外税` | 3 | receipt_24, receipt_25, receipt_28 |
| `8%税` | `外税` | 1 | receipt_29 |
| `軽減税率 R` | `外税` | 1 | receipt_30 (8% entry) |
| `標準税率` | `外税` | 1 | receipt_30 (10% entry) |
| `非課税対象額` | `非課税` | 3 | receipt_27, receipt_31, receipt_32 |

### Fixtures with no taxes (7 of 36) -- no changes needed
- receipt_3 (handwritten receipt, no tax filled in)
- receipt_14 (payment slip)
- receipt_15 (gas utility bill)
- receipt_16 (water utility bill)
- receipt_20 (gas utility bill)
- receipt_21 (payment slip)
- receipt_34 (small bakery receipt, no tax shown)

---

## Impact on Pipeline

After normalizing truth files, the pipeline's LLM prompt and/or post-processing should
also be updated to output only canonical labels. Key changes needed:

1. **Post-processing rule in pipeline**: Normalize any LLM-output tax label to the
   canonical set (`内税`, `外税`, `非課税`) by pattern matching:
   - Contains `内` + (`税` or `消費税`) -> `内税`
   - Contains `外税` or `税抜` -> `外税`
   - Contains `非課税` -> `非課税`
   - Contains just `税` with no `内`/`外` qualifier -> determine from context
     (if subtotal + tax = total, it's `外税`; if total already includes tax, it's `内税`)

2. **Schema validation**: Optionally add a validator on `TaxEntry.label` to enforce
   the canonical set, similar to how `TaxCategoryType` enforces valid rates.

3. **Prompt update**: Update the `taxes` prompt hint in `schema.py` to explicitly
   list the three canonical labels.

---

## Approval Checklist

- [ ] Review all 19 proposed label changes above
- [ ] Confirm receipt_22 classification as `内税` (vs `外税`)
- [ ] Confirm receipt_29 classification as `外税`
- [ ] Confirm receipt_30 classifications as `外税` for both entries
- [ ] Approve the three-label canonical set (`内税`, `外税`, `非課税`)
- [ ] Approve proceeding with truth file updates
