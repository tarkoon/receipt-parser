"""Receipt postprocess phase metadata and mutation tracing."""

from copy import deepcopy


POSTPROCESS_MUTATION_FIELDS = (
    "date",
    "line_items",
    "location",
    "merchant",
    "taxes",
    "tax_entries",
    "subtotal",
    "total",
    "amount_paid",
    "points_used",
    "payment_method",
)

POSTPROCESS_PHASES = (
    {
        "name": "header_identity_repair",
        "reads": ("merchant", "date", "time", "location", "ocr_text"),
        "writes": ("merchant", "date", "time", "location"),
        "invariant": "Header fields must be backed by visible OCR header/address/date evidence.",
    },
    {
        "name": "transaction_datetime_repair",
        "reads": ("date", "time", "ocr_text"),
        "writes": ("date", "time"),
        "invariant": "Transaction date/time repair requires visible OCR transaction date labels or date-line anchored time evidence and preserves plausible date/time fields.",
    },
    {
        "name": "financial_totals_repair",
        "reads": (
            "subtotal",
            "total",
            "taxes",
            "ocr_totals",
            "ocr_confidence",
            "llm_confidence",
        ),
        "writes": ("subtotal", "total", "taxes"),
        "invariant": "Reliable OCR subtotal, total, and tax overrides must preserve subtotal plus tax equals total arithmetic.",
    },
    {
        "name": "implausible_tax_amount_repair",
        "reads": ("taxes", "total", "ocr_totals", "ocr_text"),
        "writes": ("taxes",),
        "invariant": "Implausible tax amount repair requires OCR rate-base evidence and repairs only rate-base/tax swaps that violate rate arithmetic.",
    },
    {
        "name": "payment_method_repair",
        "reads": ("payment_method", "ocr_text", "ocr_confidence", "llm_confidence"),
        "writes": ("payment_method",),
        "invariant": "Payment method repair requires visible OCR cash or card/e-money markers and preserves payment_method field consistency.",
    },
    {
        "name": "toll_payment_reference_repair",
        "reads": ("payment_reference", "ocr_text"),
        "writes": ("payment_reference",),
        "invariant": "Toll payment-reference repair requires visible toll-road OCR context and a printed handling-number label, and preserves existing references.",
    },
    {
        "name": "cash_tender_reconciliation",
        "reads": ("total", "amount_paid", "payment_method", "points_used", "ocr_text"),
        "writes": ("total", "amount_paid", "payment_method"),
        "invariant": "Cash tender/change repairs require visible printed total, tendered amount, and change arithmetic.",
    },
    {
        "name": "service_receipt_recovery",
        "reads": ("line_items", "subtotal", "total", "taxes", "payment_method", "ocr_text"),
        "writes": ("line_items", "taxes", "subtotal", "payment_method"),
        "invariant": "Service receipt recovery requires visible service/table or bare-receipt OCR layout and item/tax arithmetic consistency.",
    },
    {
        "name": "body_total_layout_reconstruction",
        "reads": ("line_items", "subtotal", "total", "taxes", "location", "ocr_text"),
        "writes": ("line_items", "taxes", "subtotal", "location"),
        "invariant": "Body-total layout reconstruction requires visible item rows before a printed body-total block and subtotal/tax arithmetic consistency.",
    },
    {
        "name": "initial_item_recovery",
        "reads": ("line_items", "subtotal", "total", "ocr_text", "ocr_layout_blocks"),
        "writes": ("line_items",),
        "invariant": "Recovered rows must improve item-total consistency or match visible item layout.",
    },
    {
        "name": "gap_item_recovery",
        "reads": ("line_items", "subtotal", "total", "ocr_text"),
        "writes": ("line_items",),
        "invariant": "Gap item recovery requires visible missing, discounted, or repeated OCR rows and subtotal/total item-sum arithmetic.",
    },
    {
        "name": "prefixed_tax_marker_item_rows",
        "reads": ("line_items", "subtotal", "total", "taxes", "ocr_text"),
        "writes": ("line_items", "subtotal"),
        "invariant": "Prefixed tax-marker row projection requires visible marker-prefixed OCR item rows and subtotal/rate-base arithmetic consistency.",
    },
    {
        "name": "low_value_bag_recovery",
        "reads": ("line_items", "subtotal", "total", "ocr_text"),
        "writes": ("line_items",),
        "invariant": "Low-value bag recovery requires visible small-bag or numeric OCR context and subtotal/total item-sum arithmetic.",
    },
    {
        "name": "adjacent_price_shift_reconciliation",
        "reads": ("line_items", "subtotal", "total", "ocr_text"),
        "writes": ("line_items",),
        "invariant": "Adjacent price-shift reconciliation requires neighboring OCR item/price rows and subtotal/total item-sum arithmetic.",
    },
    {
        "name": "bag_amount_shift_reconciliation",
        "reads": ("line_items", "subtotal", "total", "ocr_text"),
        "writes": ("line_items",),
        "invariant": "Paid-bag/product amount-shift reconciliation requires adjacent OCR name/bag/amount rows, printed rate bases, and subtotal item-sum arithmetic.",
    },
    {
        "name": "item_cleanup",
        "reads": ("line_items", "subtotal", "total", "taxes", "ocr_text", "ocr_layout_blocks"),
        "writes": ("line_items", "taxes"),
        "invariant": "Cleanup may recover, remove, or rename rows only when OCR evidence, layout context, and row sums stay coherent.",
    },
    {
        "name": "phantom_tax_amount_cleanup",
        "reads": ("line_items", "taxes", "ocr_text"),
        "writes": ("line_items",),
        "invariant": "Phantom tax-amount cleanup requires a line total matching a printed tax amount plus suffix/clean-sibling item description consistency.",
    },
    {
        "name": "subtotal_item_price_repair",
        "reads": ("line_items", "subtotal", "total", "taxes", "ocr_text"),
        "writes": ("line_items",),
        "invariant": "Subtotal item price repair requires OCR subtotal evidence, nearby OCR item price evidence, and improved item-sum arithmetic.",
    },
    {
        "name": "discount_consistency_reconciliation",
        "reads": ("line_items", "subtotal", "total", "ocr_text"),
        "writes": ("line_items",),
        "invariant": "Discount consistency reconciliation requires visible negative discount placement and item total/discount field arithmetic.",
    },
    {
        "name": "coupon_discount_projection",
        "reads": ("line_items", "subtotal", "total", "ocr_text"),
        "writes": ("line_items",),
        "invariant": "Coupon discount projection requires visible coupon/CPN markers and item gross-minus-discount or subtotal arithmetic.",
    },
    {
        "name": "following_ocr_price_projection",
        "reads": ("line_items", "subtotal", "total", "taxes", "ocr_text"),
        "writes": ("line_items",),
        "invariant": "Following-OCR price projection requires repeated nearby OCR amount evidence and item-sum or rate-base arithmetic improvement.",
    },
    {
        "name": "vertical_price_qty_total_projection",
        "reads": ("line_items", "subtotal", "total", "taxes", "ocr_text"),
        "writes": ("line_items", "subtotal"),
        "invariant": "Vertical price/qty/total projection requires name/unit/qty/total OCR row blocks, unit*qty arithmetic, and subtotal/total sum consistency.",
    },
    {
        "name": "item_name_price_cleanup",
        "reads": ("line_items", "subtotal", "total", "ocr_text"),
        "writes": ("line_items",),
        "invariant": "Item name/price cleanup requires visible OCR item names or embedded price suffixes and preserves item field consistency.",
    },
    {
        "name": "priced_name_item_repair",
        "reads": ("line_items", "subtotal", "total", "ocr_text"),
        "writes": ("line_items",),
        "invariant": "Priced-name item repair requires an OCR-visible N円 item name, an unmatched OCR amount for that price, and improved subtotal/total item-sum arithmetic.",
    },
    {
        "name": "discounted_ocr_item_repair",
        "reads": ("line_items", "subtotal", "total", "taxes", "ocr_text"),
        "writes": ("line_items",),
        "invariant": "Discounted OCR item repair requires visible discount rows or stacked price/name blocks and item-sum or field-consistency arithmetic.",
    },
    {
        "name": "ocr_description_reconciliation",
        "reads": ("line_items", "subtotal", "total", "ocr_text"),
        "writes": ("line_items",),
        "invariant": "Description reconciliation requires visible OCR code/name context and must preserve item counts, prices, and totals.",
    },
    {
        "name": "digit_misread_item_repair",
        "reads": ("line_items", "subtotal", "total", "ocr_text"),
        "writes": ("line_items",),
        "invariant": "Digit-misread item repair requires a small subtotal/total item-sum gap, exactly one digit-confusion candidate, and OCR percent-marker evidence.",
    },
    {
        "name": "split_price_block_projection",
        "reads": ("line_items", "subtotal", "total", "ocr_text"),
        "writes": ("line_items",),
        "invariant": "Split-price block projection requires visible separated OCR name and price blocks and subtotal/total item-sum arithmetic.",
    },
    {
        "name": "quantity_detail_reconciliation",
        "reads": ("line_items", "subtotal", "total", "ocr_text"),
        "writes": ("line_items",),
        "invariant": "Quantity-detail repairs require visible OCR qty/unit rows and qty * unit or discount-adjusted item arithmetic.",
    },
    {
        "name": "stacked_name_price_projection",
        "reads": ("line_items", "subtotal", "total", "amount_paid", "taxes", "ocr_text"),
        "writes": ("line_items", "total", "amount_paid"),
        "invariant": "Stacked name/price projection requires visible OCR name rows followed by price rows and item-sum consistency with printed subtotal, total, or rate-base arithmetic.",
    },
    {
        "name": "stacked_inclusive_tax_restoration",
        "reads": ("taxes", "subtotal", "total", "ocr_text"),
        "writes": ("taxes", "subtotal"),
        "invariant": "Stacked inclusive tax restoration requires visible rate-target/tax label stacks and total-minus-tax subtotal arithmetic.",
    },
    {
        "name": "single_rate_inclusive_tax_restoration",
        "reads": ("line_items", "subtotal", "total", "taxes", "ocr_text"),
        "writes": ("line_items", "taxes", "subtotal"),
        "invariant": "Single-rate inclusive tax restoration requires a visible printed target/tax block and total/tax arithmetic consistency.",
    },
    {
        "name": "tax_excluded_rate_block_restoration",
        "reads": ("taxes", "ocr_text"),
        "writes": ("taxes",),
        "invariant": "Tax-excluded rate-block restoration requires visible paired 小計(税抜N%) and 消費税等(N%) rows with rate-consistent tax entries.",
    },
    {
        "name": "explicit_tax_amount_restoration",
        "reads": ("line_items", "taxes", "ocr_text"),
        "writes": ("taxes",),
        "invariant": "Explicit tax amount restoration requires visible 税率N%税額 rows and tax amounts bounded by item/tax-rate arithmetic.",
    },
    {
        "name": "printed_summary_total_tax_repair",
        "reads": ("line_items", "subtotal", "total", "amount_paid", "points_used", "taxes", "ocr_text"),
        "writes": ("subtotal", "total", "amount_paid"),
        "invariant": "Printed summary total repair requires visible 小計/合計 rows whose subtotal plus tax equals the printed total and preserves payment minus points arithmetic.",
    },
    {
        "name": "printed_item_sum_total_repair",
        "reads": ("line_items", "subtotal", "total", "amount_paid", "taxes", "ocr_text"),
        "writes": ("subtotal", "total", "amount_paid"),
        "invariant": "Printed item-sum total repair requires item rows whose sum matches a visible printed amount and preserves item, tax, and payment arithmetic.",
    },
    {
        "name": "printed_external_tax_amount_restoration",
        "reads": ("line_items", "taxes", "subtotal", "total", "ocr_text"),
        "writes": ("taxes",),
        "invariant": "Printed external-tax amount restoration requires visible per-rate external tax amount rows and tax/base/total consistency.",
    },
    {
        "name": "bare_number_tax_summary_restoration",
        "reads": ("line_items", "taxes", "subtotal", "total", "ocr_text"),
        "writes": ("taxes", "subtotal"),
        "invariant": "Bare-number tax summary restoration requires visible rate labels, numeric tax amounts, and printed total arithmetic.",
    },
    {
        "name": "external_tax_total_restoration",
        "reads": ("line_items", "subtotal", "total", "amount_paid", "points_used", "taxes", "ocr_text"),
        "writes": ("subtotal", "total", "amount_paid"),
        "invariant": "External tax total restoration requires printed subtotal plus external-tax arithmetic and a visible summary/payment total.",
    },
    {
        "name": "small_target_only_tax_pruning",
        "reads": ("taxes", "subtotal", "total", "ocr_text"),
        "writes": ("taxes", "subtotal"),
        "invariant": "Small target-only tax pruning requires visible rate bases, no matching printed tax amount, and total-minus-tax subtotal arithmetic.",
    },
    {
        "name": "bag_item_rate_base_reconciliation",
        "reads": ("line_items", "subtotal", "total", "taxes", "ocr_text"),
        "writes": ("line_items",),
        "invariant": "Bag item price/rate-base reconciliation requires a tiny printed 10% rate base and paid-bag totals that can be reconciled to it.",
    },
    {
        "name": "tax_category_assignment",
        "reads": ("line_items", "taxes", "subtotal", "total", "ocr_totals", "ocr_text"),
        "writes": ("line_items", "taxes"),
        "invariant": "Item tax categories and tax entries must agree with printed rate bases or tax summaries.",
    },
    {
        "name": "payment_points_reconciliation",
        "reads": ("amount_paid", "payment_method", "points_used", "subtotal", "total", "taxes", "ocr_text"),
        "writes": ("amount_paid", "payment_method", "points_used", "subtotal"),
        "invariant": "Payment, points, and subtotal changes must preserve total/tax arithmetic.",
    },
    {
        "name": "single_item_quantity_repair",
        "reads": ("line_items", "total", "ocr_text"),
        "writes": ("line_items",),
        "invariant": "Single-item quantity repair requires OCR @unit x qty notation and unit*qty arithmetic matching the item or receipt total.",
    },
    {
        "name": "jan_pos_row_projection",
        "reads": ("line_items", "subtotal", "total", "ocr_text", "ocr_totals"),
        "writes": ("line_items",),
        "invariant": "JAN/POS row projection requires visible item-code, quantity, and price OCR rows with item-sum arithmetic consistent with printed totals.",
    },
    {
        "name": "barcode_row_projection",
        "reads": ("line_items", "subtotal", "total", "ocr_text"),
        "writes": ("line_items",),
        "invariant": "Barcode row projection requires visible barcode/quantity/price OCR rows and item-sum arithmetic consistent with printed totals.",
    },
    {
        "name": "dense_item_row_projection",
        "reads": ("line_items", "subtotal", "total", "taxes", "ocr_text"),
        "writes": ("line_items",),
        "invariant": "Dense item row projection requires visible dense OCR item/price rows and item-sum arithmetic consistent with printed totals.",
    },
    {
        "name": "dense_sequence_row_projection",
        "reads": ("line_items", "subtotal", "total", "taxes", "ocr_text"),
        "writes": ("line_items", "taxes"),
        "invariant": "Dense sequence row projection requires visible dense OCR row sequences and item/tax arithmetic consistent with printed totals.",
    },
    {
        "name": "structural_item_reconstruction",
        "reads": ("line_items", "subtotal", "total", "taxes", "ocr_text", "ocr_totals"),
        "writes": ("line_items", "taxes", "subtotal", "total", "amount_paid"),
        "invariant": "Structural reconstruction must be triggered by OCR row layout and validated by sums.",
    },
    {
        "name": "code_prefixed_description_cleanup",
        "reads": ("line_items", "ocr_text"),
        "writes": ("line_items",),
        "invariant": "Code-prefixed description cleanup requires visible OCR/POS item-code prefixes and preserves item description field consistency.",
    },
    {
        "name": "duplicate_row_cleanup",
        "reads": ("line_items", "subtotal", "ocr_text"),
        "writes": ("line_items",),
        "invariant": "Duplicate row cleanup requires an OCR singleton item occurrence and subtotal-overage arithmetic matching exactly one duplicate row total.",
    },
    {
        "name": "basket_marker_rows",
        "reads": ("line_items", "subtotal", "total", "taxes", "ocr_text"),
        "writes": ("line_items",),
        "invariant": "Basket marker row projection requires visible basket-marker/stacked price OCR layout and subtotal/rate-base arithmetic consistency.",
    },
    {
        "name": "final_consistency_pass",
        "reads": ("line_items", "taxes", "subtotal", "total", "amount_paid", "points_used", "ocr_text"),
        "writes": ("line_items", "taxes", "subtotal", "total", "amount_paid", "points_used"),
        "invariant": "Final mutations must restore field consistency without fixture or known-answer logic.",
    },
)
POSTPROCESS_PHASE_BY_NAME = {phase["name"]: phase for phase in POSTPROCESS_PHASES}


def _snapshot_receipt_mutation_fields(extracted: dict) -> dict:
    return {
        field: deepcopy(extracted.get(field))
        for field in POSTPROCESS_MUTATION_FIELDS
        if field in extracted
    }


def _diff_receipt_mutation_fields(before: dict, after: dict) -> dict:
    changes = {}
    for field in POSTPROCESS_MUTATION_FIELDS:
        before_value = before.get(field)
        after_value = after.get(field)
        if before_value != after_value:
            changes[field] = {
                "before": before_value,
                "after": after_value,
            }
    return changes


def _record_receipt_mutation(
    mutation_trace: list[dict] | None,
    stage: str,
    before: dict | None,
    extracted: dict,
) -> dict | None:
    if mutation_trace is None or before is None:
        return None
    after = _snapshot_receipt_mutation_fields(extracted)
    changes = _diff_receipt_mutation_fields(before, after)
    if changes:
        mutation_trace.append({"stage": stage, "changes": changes})
    return after


def _record_receipt_phase_mutation(
    mutation_trace: list[dict] | None,
    phase_name: str,
    before: dict | None,
    extracted: dict,
) -> dict | None:
    if phase_name not in POSTPROCESS_PHASE_BY_NAME:
        raise ValueError(f"Unknown receipt postprocess phase: {phase_name}")
    trace_len = len(mutation_trace) if mutation_trace is not None else 0
    after = _record_receipt_mutation(mutation_trace, phase_name, before, extracted)
    if mutation_trace is not None and len(mutation_trace) > trace_len:
        phase = POSTPROCESS_PHASE_BY_NAME[phase_name]
        mutation_trace[-1]["reads"] = phase["reads"]
        mutation_trace[-1]["writes"] = phase["writes"]
        mutation_trace[-1]["invariant"] = phase["invariant"]
    return after
