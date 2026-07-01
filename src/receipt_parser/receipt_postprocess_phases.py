"""Named receipt postprocess phase runners."""

import re

from .patterns import should_override_field
from .receipt_financial import (
    extract_rate_bases,
    extract_points_used,
    reconcile_points_payment_from_ocr,
)
from .receipt_identity_payment import (
    _apply_financial_overrides,
    _fix_company_name_merchant,
    _fix_date,
    _fix_payment_method,
    _fix_time,
    _fix_toll_payment_reference,
    _fix_total_from_stacked_cash_tender_block,
    _fix_unlabeled_cash_tender_change_block,
)
from .receipt_items import (
    _drop_non_product_line_items,
    _fix_bag_description_from_ocr_code_context,
    _fix_bag_item_prices_from_ocr,
    _fix_bag_item_prices_from_rate_bases,
    _fix_bare_service_receipt_without_itemization,
    _fix_colon_split_product_names_from_ocr,
    _fix_duplicate_descriptions_from_ocr,
    _fix_line_items,
    _fix_o_ring_descriptions_from_ocr,
    _fix_qty_code_row_descriptions_from_ocr,
    _fix_single_service_inclusive_tax,
    _fix_small_non_bag_item_prices_from_ocr,
    _recover_discounted_item_from_gap,
    _recover_missing_bag_items_from_ocr,
    _recover_qty_unit_total_item_from_empty_extraction,
    _recover_repeated_item_from_gap,
    _replace_repeated_ocr_item_block_when_balanced,
    _replace_vertical_price_qty_total_rows_when_balanced,
)
from .receipt_item_cleanup import (
    _drop_duplicate_rows_when_subtotal_balances,
    _fix_adjacent_ocr_price_shift_when_balanced,
    _fix_discounted_item_gross_prices_from_ocr,
    _fix_embedded_price_suffix_totals,
    _fix_non_bag_items_named_as_bag,
    _normalize_taxes,
    _repair_discounted_line_item_totals_when_balanced,
    _repair_discounted_ocr_pair_descriptions,
    _repair_pre_price_stack_descriptions_from_ocr,
    _replace_basket_marker_rows_when_balanced,
)
from .receipt_item_repair import (
    _clean_code_prefixed_item_descriptions,
    _drop_duplicate_with_embedded_price,
    _drop_phantom_from_tax_amount,
    _fix_code_table_descriptions_by_order,
    _fix_digit_misread_items,
    _fix_priced_in_name_items,
    _revert_unsupported_qty_inflation,
    _fix_single_item_qty_from_ocr,
    _fix_split_item_price_body_total_layout,
)
from .receipt_late_repairs import (
    _drop_numeric_marker_description_rows,
    _fix_header_store_line_location,
    _fix_name_bag_amount_shift_from_ocr,
    _fix_small_bag_description_from_ocr_entry,
    _fix_split_address_location_from_ocr,
    _fix_split_bag_price_from_nearby_single_digit,
    _recover_labeled_purchase_site_location,
    _replace_stacked_name_price_rows_when_balanced,
    _restore_single_rate_inclusive_tax_block,
    _restore_stacked_inclusive_tax_block,
    _restore_tax_excluded_per_rate_blocks,
)
from .receipt_location import _recover_ascii_brand_header_location
from .receipt_marker_projection import (
    _fix_qty_totals_from_ocr_unit_lines,
    _replace_campaign_discount_stream_when_balanced,
    _replace_jan_pos_items_when_balanced,
    _replace_prefixed_tax_marker_item_rows_when_balanced,
)
from .receipt_recovery import (
    _apply_coupon_discount_blocks,
    _drop_applied_coupon_line_items,
    _fix_implausible_tax_amounts,
    _fix_item_totals_from_following_discount_lines,
    _fix_items_from_subtotal,
    _fix_printed_tax_amounts_from_structural_blocks,
    _recover_missing_items_from_gap,
    _repair_tiny_item_prices_from_following_ocr,
    _replace_split_price_block_when_balanced,
    _restore_explicit_tax_rate_amount_lines,
    _restore_printed_external_tax_amounts,
)
from .receipt_row_projection import (
    _append_missing_low_value_bag_from_gap,
    _fix_numeric_desc_from_ocr_price_context,
    _fix_qty_context_and_reduced_rate_from_ocr,
    _replace_barcode_qty_price_rows_when_balanced,
    _replace_barcode_unit_qty_amount_stack_when_balanced,
    _replace_dense_item_rows_when_balanced,
    _replace_dense_sequence_rows_when_balanced,
    _replace_overage_item_with_low_value_bag,
    _replace_service_table_items_when_balanced,
)
from .receipt_tax_categories import (
    _apply_single_bag_standard_rate_split,
    _assign_single_standard_rate_from_small_base,
    _fix_nonfood_packaging_tax_categories,
    _fix_tax_categories_from_ocr_markers,
    _fix_tax_categories_from_price_line_markers,
    _rebalance_standard_categories_from_reduced_rate_markers,
    _rebalance_tax_categories_to_rate_bases,
    assign_tax_categories,
)
from .receipt_totals import (
    _drop_unprinted_small_target_only_taxes,
    _prefer_printed_item_sum_total_when_balanced,
    _restore_bare_number_tax_summary,
    _restore_external_tax_total_from_printed_subtotal,
    _restore_printed_summary_total_when_tax_balanced,
)
from .receipt_projection import (
    _repair_previous_item_from_following_qty_detail,
)


def _run_barcode_row_projection_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: OCR exposes barcode-adjacent quantity, unit, or price rows.

    Invariant: barcode projections may replace rows only when barcode row
    totals remain coherent with printed receipt amounts.
    """
    for repair in repairs:
        if repair == "barcode_unit_qty_amount_stack":
            _replace_barcode_unit_qty_amount_stack_when_balanced(extracted, unified_text)
        elif repair == "barcode_qty_price_rows":
            _replace_barcode_qty_price_rows_when_balanced(extracted, unified_text)
        else:
            raise ValueError(f"Unknown barcode row projection repair: {repair}")


def _run_dense_item_row_projection_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: dense OCR item rows expose adjacent item names and prices.

    Invariant: projected dense rows may replace current items only when visible
    OCR row totals remain coherent with printed receipt amounts.
    """
    for repair in repairs:
        if repair == "dense_item_rows":
            _replace_dense_item_rows_when_balanced(extracted, unified_text)
        else:
            raise ValueError(f"Unknown dense item row projection repair: {repair}")


def _run_dense_sequence_row_projection_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: OCR exposes dense item/quantity/price row sequences.

    Invariant: projected dense sequences may replace rows only when visible OCR
    row totals and tax categories remain coherent with printed receipt amounts.
    """
    for repair in repairs:
        if repair == "dense_sequence_rows":
            _replace_dense_sequence_rows_when_balanced(extracted, unified_text)
        else:
            raise ValueError(f"Unknown dense sequence row projection repair: {repair}")


def _run_jan_pos_row_projection_phase(
    extracted: dict,
    unified_text: str,
    ocr_totals: dict | None,
) -> None:
    """Trigger: OCR exposes JAN/POS item-code rows with adjacent qty/price text.

    Invariant: projected JAN/POS rows may replace line items only when item
    sums remain coherent with printed receipt amounts.
    """
    _replace_jan_pos_items_when_balanced(extracted, unified_text, ocr_totals or {})


def _run_campaign_discount_projection_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: OCR item streams interleave campaign discount marker rows.

    Invariant: projected discounted rows may replace current items only when
    visible campaign discount amounts reconcile to the printed subtotal.
    """
    for repair in repairs:
        if repair == "campaign_discount_stream":
            _replace_campaign_discount_stream_when_balanced(extracted, unified_text)
        else:
            raise ValueError(f"Unknown campaign discount projection repair: {repair}")


def _run_quantity_detail_reconciliation_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: OCR qty detail rows adjacent to item names or price rows.

    Invariant: qty/unit mutations and empty-item recovery require visible
    quantity notation and qty * unit, printed total, or discount-adjusted item
    arithmetic.
    """
    for repair in repairs:
        if repair == "following_qty_detail":
            _repair_previous_item_from_following_qty_detail(extracted, unified_text)
        elif repair == "qty_totals_from_unit_lines":
            _fix_qty_totals_from_ocr_unit_lines(extracted, unified_text)
        elif repair == "qty_context_and_reduced_rate":
            _fix_qty_context_and_reduced_rate_from_ocr(extracted, unified_text)
        elif repair == "empty_qty_unit_total_block":
            _recover_qty_unit_total_item_from_empty_extraction(extracted, unified_text)
        else:
            raise ValueError(f"Unknown quantity detail reconciliation repair: {repair}")


def _run_cash_tender_reconciliation_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: OCR cash-layout rows print total, tendered amount, and change.

    Invariant: total, amount_paid, and cash payment method changes must be
    consistent with printed tender/change arithmetic and points adjustments.
    """
    for repair in repairs:
        if repair == "stacked_cash_tender":
            _fix_total_from_stacked_cash_tender_block(extracted, unified_text)
        elif repair == "unlabeled_cash_tender_change":
            _fix_unlabeled_cash_tender_change_block(extracted, unified_text)
        else:
            raise ValueError(f"Unknown cash tender reconciliation repair: {repair}")


def _run_payment_method_repair_phase(
    extracted: dict,
    unified_text: str,
    ocr_conf: float | None,
    llm_conf: dict | None,
) -> None:
    """Trigger: OCR-visible cash, card, e-money, tender, or change markers.

    Invariant: payment_method changes must be backed by visible payment
    markers and preserve the field's cash-vs-credit consistency.
    """
    _fix_payment_method(extracted, unified_text, ocr_conf, llm_conf)


def _run_toll_payment_reference_repair_phase(
    extracted: dict,
    unified_text: str,
) -> None:
    """Trigger: toll-road OCR markers with a printed handling/reference number.

    Invariant: only fill a missing payment_reference from a visible handling
    number label, preserving any reference supplied by upstream extraction.
    """
    _fix_toll_payment_reference(extracted, unified_text)


def _run_service_receipt_recovery_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: OCR service tables, bare receipt layouts, or single service rows.

    Invariant: service item recovery/removal and inclusive-tax reconstruction
    must preserve visible row layout, printed total evidence, and tax arithmetic.
    """
    for repair in repairs:
        if repair == "bare_service_without_itemization":
            _fix_bare_service_receipt_without_itemization(extracted, unified_text)
        elif repair == "service_table_items":
            _replace_service_table_items_when_balanced(extracted, unified_text)
        elif repair == "single_service_inclusive_tax":
            _fix_single_service_inclusive_tax(extracted, unified_text)
        else:
            raise ValueError(f"Unknown service receipt recovery repair: {repair}")


def _run_body_total_layout_reconstruction_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: item rows appear before a printed 本体合計 body-total block.

    Invariant: reconstructed items, subtotal, tax entries, and optional branch
    location must be backed by visible body-total layout rows and subtotal plus
    tax arithmetic.
    """
    for repair in repairs:
        if repair == "split_item_price_body_total":
            _fix_split_item_price_body_total_layout(extracted, unified_text)
        else:
            raise ValueError(
                f"Unknown body-total layout reconstruction repair: {repair}"
            )


def _run_coupon_discount_projection_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: OCR-visible coupon, COUPON, or CPN rows.

    Invariant: item totals must represent gross minus the printed coupon
    amount, and standalone coupon rows may be dropped only after that discount
    is present on an item row.
    """
    for repair in repairs:
        if repair == "coupon_discount_blocks":
            _apply_coupon_discount_blocks(extracted, unified_text)
        elif repair == "drop_applied_coupon_line_items":
            _drop_applied_coupon_line_items(extracted, unified_text)
        else:
            raise ValueError(
                f"Unknown coupon discount projection repair: {repair}"
            )


def _run_discount_consistency_reconciliation_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: OCR-visible negative discount rows adjacent to item prices.

    Invariant: item totals must reconcile to the printed gross item price minus
    the visible discount amount, preserving subtotal consistency.
    """
    for repair in repairs:
        if repair == "discounted_item_gross_prices":
            _fix_discounted_item_gross_prices_from_ocr(extracted, unified_text)
        elif repair == "following_discount_lines":
            _fix_item_totals_from_following_discount_lines(extracted, unified_text)
        else:
            raise ValueError(
                f"Unknown discount consistency reconciliation repair: {repair}"
            )


def _run_following_ocr_price_projection_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: repeated following OCR amount rows near an item description.

    Invariant: projected prices must improve item-sum or printed rate-base
    arithmetic without changing unrelated items.
    """
    for repair in repairs:
        if repair == "tiny_item_prices_from_following_ocr":
            _repair_tiny_item_prices_from_following_ocr(extracted, unified_text)
        else:
            raise ValueError(
                f"Unknown following OCR price projection repair: {repair}"
            )


def _run_discounted_ocr_item_repair_phase(extracted: dict, unified_text: str) -> None:
    """Trigger: visible discount rows or stacked price/name blocks in OCR.

    Invariant: discounted totals must close the printed item sum, and
    description repairs must remain backed by visible OCR field ownership.
    """
    _repair_discounted_line_item_totals_when_balanced(extracted, unified_text)
    _repair_discounted_ocr_pair_descriptions(extracted, unified_text)
    _repair_pre_price_stack_descriptions_from_ocr(extracted, unified_text)


def _run_ocr_description_reconciliation_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: OCR code rows, duplicated names, O-ring text, or bag context.

    Invariant: description changes require visible OCR support and must keep
    item count, quantity, unit price, total, discount, and tax fields coherent.
    """
    for repair in repairs:
        if repair == "qty_code_rows":
            _fix_qty_code_row_descriptions_from_ocr(extracted, unified_text)
        elif repair == "code_table_order":
            _fix_code_table_descriptions_by_order(extracted, unified_text)
        elif repair == "duplicate_descriptions":
            _fix_duplicate_descriptions_from_ocr(extracted, unified_text)
        elif repair == "o_ring_descriptions":
            _fix_o_ring_descriptions_from_ocr(extracted, unified_text)
        elif repair == "colon_split_names":
            _fix_colon_split_product_names_from_ocr(extracted, unified_text)
        elif repair == "bag_code_context":
            _fix_bag_description_from_ocr_code_context(extracted, unified_text)
        else:
            raise ValueError(f"Unknown OCR description reconciliation repair: {repair}")


def _run_split_price_block_projection_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: OCR separates item names from a nearby price block.

    Invariant: projected prices may replace item rows only when the visible
    split name/price blocks balance against printed subtotal or total.
    """
    for repair in repairs:
        if repair == "split_price_block":
            _replace_split_price_block_when_balanced(extracted, unified_text)
        else:
            raise ValueError(
                f"Unknown split-price block projection repair: {repair}"
            )


def _run_gap_item_recovery_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: OCR rows expose missing, discounted, or repeated item gaps.

    Invariant: recovered or replaced rows must be visible in OCR and close a
    subtotal/total item-sum gap without fixture, merchant, or product answers.
    """
    for repair in repairs:
        if repair == "missing_items":
            _recover_missing_items_from_gap(extracted, unified_text)
        elif repair == "discounted_gap":
            _recover_discounted_item_from_gap(extracted, unified_text)
        elif repair == "repeated_gap":
            _recover_repeated_item_from_gap(extracted, unified_text)
        elif repair == "repeated_ocr_block":
            _replace_repeated_ocr_item_block_when_balanced(extracted, unified_text)
        else:
            raise ValueError(f"Unknown gap item recovery repair: {repair}")


def _run_prefixed_tax_marker_item_rows_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: OCR item rows prefixed by tax markers in the item block.

    Invariant: projected marker-prefixed rows must balance to the printed
    subtotal or total and preserve rate-base totals implied by the markers.
    """
    for repair in repairs:
        if repair == "prefixed_tax_marker_item_rows":
            _replace_prefixed_tax_marker_item_rows_when_balanced(
                extracted,
                unified_text,
            )
        else:
            raise ValueError(
                "Unknown prefixed tax-marker item row repair: "
                f"{repair}"
            )


def _run_low_value_bag_recovery_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: OCR exposes low-value bag rows or numeric bag price context.

    Invariant: missing bag rows, gap-closing bag appends, overage replacement,
    and numeric-description repair must be visible in OCR and keep item sums
    consistent with subtotal or total arithmetic.
    """
    for repair in repairs:
        if repair == "missing_bag_items":
            _recover_missing_bag_items_from_ocr(extracted, unified_text)
        elif repair == "missing_low_value_bag_gap":
            _append_missing_low_value_bag_from_gap(extracted, unified_text)
        elif repair == "overage_low_value_bag":
            _replace_overage_item_with_low_value_bag(extracted, unified_text)
        elif repair == "numeric_description_context":
            _fix_numeric_desc_from_ocr_price_context(extracted, unified_text)
        else:
            raise ValueError(f"Unknown low-value bag recovery repair: {repair}")


def _run_adjacent_price_shift_reconciliation_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: adjacent OCR item/price rows expose a shifted amount.

    Invariant: price shifts may mutate rows only when visible OCR adjacency and
    subtotal/total item-sum arithmetic stay balanced.
    """
    for repair in repairs:
        if repair == "adjacent_ocr_price_shift":
            _fix_adjacent_ocr_price_shift_when_balanced(extracted, unified_text)
        else:
            raise ValueError(
                f"Unknown adjacent price-shift reconciliation repair: {repair}"
            )


def _run_bag_amount_shift_reconciliation_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: adjacent OCR product name, paid-bag price, and product amount rows.

    Invariant: bag/product amount shifts and unsupported quantity reversions may
    mutate rows only when printed rate bases identify tax categories and
    subtotal item-sum arithmetic remains balanced.
    """
    items = extracted.get("line_items") or []
    try:
        if items and extracted.get("total") is not None:
            row_sum = sum(float(item.get("total") or 0) for item in items if isinstance(item, dict))
            if len(items) >= 2 and abs(row_sum - float(extracted["total"])) <= 2:
                return
    except (TypeError, ValueError):
        pass
    for repair in repairs:
        if repair == "name_bag_amount_shift":
            changed = _fix_name_bag_amount_shift_from_ocr(extracted, unified_text)
            if changed and extracted.get("line_items"):
                _revert_unsupported_qty_inflation(extracted["line_items"], unified_text)
        else:
            raise ValueError(
                f"Unknown bag amount-shift reconciliation repair: {repair}"
            )


def _run_payment_points_reconciliation_phase(
    extracted: dict,
    unified_text: str,
    ocr_conf: float,
    llm_conf: dict | None,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: OCR points-use lines or points tender/payment rows are visible.

    Invariant: points_used, amount_paid, payment_method, and subtotal changes
    must preserve total minus points payment arithmetic and printed evidence.
    """
    for repair in repairs:
        if repair == "points_used":
            points = extract_points_used(unified_text)
            if points is not None:
                existing_points = extracted.get("points_used")
                if (
                    should_override_field("points_used", ocr_conf, llm_conf)
                    or existing_points is None
                    or (points > 0 and float(existing_points or 0) == 0)
                ):
                    extracted["points_used"] = points
            elif extracted.get("points_used") is not None:
                has_points_evidence = bool(re.search(r'ポイント利用|ポイント値引', unified_text))
                if not has_points_evidence:
                    extracted["points_used"] = 0
            else:
                extracted["points_used"] = 0
        elif repair == "points_payment":
            reconcile_points_payment_from_ocr(extracted, unified_text)
        else:
            raise ValueError(f"Unknown payment points reconciliation repair: {repair}")


def _run_stacked_inclusive_tax_restoration_phase(
    extracted: dict,
    unified_text: str,
) -> None:
    """Trigger: stacked OCR rate-target labels followed by inclusive tax values.

    Invariant: restored inclusive tax entries must come from visible stacked
    target/tax rows and update subtotal only through total-minus-tax arithmetic.
    """
    _restore_stacked_inclusive_tax_block(extracted, unified_text)


def _run_stacked_name_price_projection_phase(
    extracted: dict,
    unified_text: str,
) -> None:
    """Trigger: OCR stacks item names before matching price rows.

    Invariant: projected rows may replace items only when item totals reconcile
    to printed subtotal, total, printed amount, or rate-base arithmetic.
    """
    _replace_stacked_name_price_rows_when_balanced(extracted, unified_text)


def _run_single_rate_inclusive_tax_restoration_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: printed inclusive target/tax summary rows and amount blocks.

    Invariant: restored tax entries, subtotal, and item tax categories must
    agree with the receipt total and visible inclusive tax arithmetic.
    """
    for repair in repairs:
        if repair == "single_rate_inclusive_tax_block":
            _restore_single_rate_inclusive_tax_block(extracted, unified_text)
        elif repair == "printed_inclusive_tax_structural_blocks":
            _fix_printed_tax_amounts_from_structural_blocks(extracted, unified_text)
        else:
            raise ValueError(
                f"Unknown single-rate inclusive tax restoration repair: {repair}"
            )


def _run_tax_excluded_rate_block_restoration_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: paired 小計(税抜N%) and 消費税等(N%) printed rows.

    Invariant: restored external-tax entries must come from visible tax rows
    whose rates match the paired printed tax-excluded subtotal labels.
    """
    for repair in repairs:
        if repair == "tax_excluded_per_rate_blocks":
            _restore_tax_excluded_per_rate_blocks(extracted, unified_text)
        else:
            raise ValueError(
                f"Unknown tax-excluded rate block restoration repair: {repair}"
            )


def _run_explicit_tax_amount_restoration_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: visible 税率N%税額 rows followed by yen amount candidates.

    Invariant: restored external-tax entries must be bounded by item totals
    and the printed rate arithmetic, and must match visible item categories
    when categories are already assigned.
    """
    for repair in repairs:
        if repair == "explicit_tax_rate_amount_lines":
            _restore_explicit_tax_rate_amount_lines(extracted, unified_text)
        else:
            raise ValueError(
                f"Unknown explicit tax amount restoration repair: {repair}"
            )


def _run_printed_summary_total_tax_repair_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: visible printed 小計/合計 rows and tax-balance summary lines.

    Invariant: subtotal plus tax must match a visible printed total, and
    amount_paid may change only to preserve total minus points_used arithmetic.
    """
    for repair in repairs:
        if repair == "printed_summary_total_tax_balanced":
            _restore_printed_summary_total_when_tax_balanced(extracted, unified_text)
        else:
            raise ValueError(
                "Unknown printed summary total/tax repair: "
                f"{repair}"
            )


def _run_printed_item_sum_total_repair_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: visible printed item-sum or summary total amount.

    Invariant: total/subtotal changes must be backed by printed item sums and
    preserve item, tax, payment, and amount_paid arithmetic consistency.
    """
    for repair in repairs:
        if repair == "printed_item_sum_total":
            _prefer_printed_item_sum_total_when_balanced(extracted, unified_text)
        else:
            raise ValueError(
                "Unknown printed item-sum total repair: "
                f"{repair}"
            )


def _run_printed_external_tax_amount_restoration_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: printed per-rate external tax amount rows.

    Invariant: restored tax amounts must remain consistent with printed
    taxable bases and subtotal plus external tax total arithmetic.
    """
    for repair in repairs:
        if repair == "printed_external_tax_amounts":
            _restore_printed_external_tax_amounts(extracted, unified_text)
        else:
            raise ValueError(
                "Unknown printed external-tax amount restoration repair: "
                f"{repair}"
            )


def _run_bare_number_tax_summary_restoration_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: bare numeric rate labels and tax amount rows.

    Invariant: restored tax entries and subtotal must remain consistent with
    visible rate labels, printed tax amounts, and receipt total arithmetic.
    """
    for repair in repairs:
        if repair == "bare_number_tax_summary":
            _restore_bare_number_tax_summary(extracted, unified_text)
        else:
            raise ValueError(
                "Unknown bare-number tax summary restoration repair: "
                f"{repair}"
            )


def _run_external_tax_total_restoration_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: printed subtotal plus external-tax entries imply a total.

    Invariant: subtotal + external taxes must match a visible summary/payment
    total, and amount_paid may change only to preserve total minus points_used
    arithmetic.
    """
    for repair in repairs:
        if repair == "external_tax_total_from_printed_subtotal":
            _restore_external_tax_total_from_printed_subtotal(extracted, unified_text)
        else:
            raise ValueError(
                f"Unknown external tax total restoration repair: {repair}"
            )


def _run_small_target_only_tax_pruning_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: rate-base-only tax rows with tiny unprinted tax amounts.

    Invariant: any removed tax entry must lack a printed tax amount, have a
    visible target base for the same rate, and leave subtotal consistent with
    total minus the remaining tax entries.
    """
    for repair in repairs:
        if repair == "drop_small_target_only_taxes":
            _drop_unprinted_small_target_only_taxes(extracted, unified_text)
        else:
            raise ValueError(
                f"Unknown small target-only tax pruning repair: {repair}"
            )


def _tax_assignment_rate_bases(unified_text: str, ocr_totals: dict | None) -> dict:
    rate_bases = extract_rate_bases(unified_text)
    for rate, base in ((ocr_totals or {}).get('_breakdown_rate_bases') or {}).items():
        if rate not in rate_bases or rate_bases[rate] is None:
            rate_bases[rate] = base
    return rate_bases


def _has_printed_vertical_tax_block(unified_text: str) -> bool:
    return bool(
        re.search(r'税率\s*\n\s*(?:\d+(?:\.\d+)?)\s*\n\s*\*+\s*\n\s*税抜き\s*\n\s*税額', unified_text)
        or re.search(r'税率\s*\n\s*(?:\d+(?:\.\d+)?)\s*%?\s*\n.*?\n\s*税抜き\s*\n\s*税額', unified_text, re.S)
    )


def _restore_tax_entries_from_item_rate_sums(
    extracted: dict,
    unified_text: str,
    ocr_totals: dict | None,
    rate_bases: dict | None,
) -> None:
    """Trigger: item tax categories exist but tax entries are missing or stale.

    Invariant: restored tax entries must be derived from item rate sums unless a
    printed vertical tax block owns the tax amounts.
    """
    if (
        not extracted.get("line_items")
        or not extracted.get("taxes")
        or _has_printed_vertical_tax_block(unified_text)
    ):
        return

    rate_sums: dict[str, float] = {}
    for item in extracted["line_items"]:
        if not isinstance(item, dict):
            continue
        cat = item.get("tax_category", "0%")
        if cat and cat != "0%":
            rate_sums[cat] = rate_sums.get(cat, 0) + (item.get("total") or 0)
    existing_rates = {tax.get("rate") for tax in extracted["taxes"]}
    existing_labels = [
        tax.get("label", "") for tax in extracted["taxes"] if tax.get("label")
    ]
    default_label = existing_labels[0] if existing_labels else None
    is_inclusive = default_label in ("内税", "消費税等") or (
        default_label or ""
    ).startswith("内")
    ocr_tax_rates = {
        tax.get("rate")
        for tax in ((ocr_totals or {}).get("taxes") or [])
        if isinstance(tax, dict) and tax.get("rate") and (tax.get("amount") or 0) > 0
    }
    target_only_rates = {
        rate
        for rate, base in (rate_bases or {}).items()
        if base and base > 0 and rate not in ocr_tax_rates
    }
    for cat in sorted(rate_sums):
        if cat not in existing_rates and rate_sums[cat] > 0:
            rate_pct = float(cat.replace("%", "")) / 100.0
            if is_inclusive:
                computed_tax = round(rate_sums[cat] * rate_pct / (1 + rate_pct))
            else:
                computed_tax = round(rate_sums[cat] * rate_pct)
            if cat in target_only_rates and computed_tax <= 1:
                continue
            if computed_tax > 0:
                extracted["taxes"].append({
                    "rate": cat,
                    "label": default_label,
                    "amount": computed_tax,
                })

    ocr_zero_rates = {
        tax.get("rate")
        for tax in ((ocr_totals or {}).get("taxes") or [])
        if isinstance(tax, dict) and (tax.get("amount") or 0) == 0
    }
    for tax in extracted["taxes"]:
        if not isinstance(tax, dict):
            continue
        rate = tax.get("rate")
        if not rate or rate not in rate_sums or rate in ocr_zero_rates:
            continue
        amount = tax.get("amount") or 0
        try:
            rate_pct = float(rate.replace("%", "")) / 100.0
        except ValueError:
            continue
        if rate_pct <= 0:
            continue
        entry_label = tax.get("label") or default_label or ""
        entry_inclusive = entry_label in ("内税", "消費税等") or entry_label.startswith("内")
        if entry_inclusive:
            expected = round(rate_sums[rate] * rate_pct / (1 + rate_pct))
        else:
            expected = round(rate_sums[rate] * rate_pct)
        if expected > 0 and (amount == 0 or amount > expected * 3):
            tax["amount"] = expected
        elif expected > 0 and amount > 0 and amount < expected / 3:
            tax["amount"] = expected

    kept = []
    for tax in extracted["taxes"]:
        if not isinstance(tax, dict):
            kept.append(tax)
            continue
        rate = tax.get("rate")
        if rate and rate in rate_sums:
            try:
                rate_pct = float(rate.replace("%", "")) / 100.0
            except ValueError:
                rate_pct = 0
            if rate_pct > 0:
                if is_inclusive:
                    expected = round(rate_sums[rate] * rate_pct / (1 + rate_pct))
                else:
                    expected = round(rate_sums[rate] * rate_pct)
                if expected == 0:
                    continue
        kept.append(tax)
    extracted["taxes"] = kept


def _run_tax_category_assignment_phase(
    extracted: dict,
    unified_text: str,
    ocr_totals: dict | None,
    repairs: tuple[str, ...],
    rate_bases: dict | None = None,
) -> dict:
    """Trigger: OCR rate markers, rate-base summaries, or price-line flags.

    Invariant: item tax categories and normalized/restored tax entries must
    remain consistent with visible rate markers, printed rate bases,
    subtotal/total arithmetic, and single-bag splits.
    """
    items = extracted.get("line_items") or []
    merged_rate_bases = rate_bases if rate_bases is not None else _tax_assignment_rate_bases(
        unified_text,
        ocr_totals,
    )
    for repair in repairs:
        if repair == "assign_tax_categories":
            if items:
                assign_tax_categories(
                    items,
                    unified_text,
                    ocr_totals or {},
                    merged_rate_bases,
                    extracted_taxes=extracted.get("taxes"),
                )
        elif repair == "ocr_markers":
            if items:
                _fix_tax_categories_from_ocr_markers(items, unified_text)
        elif repair == "price_line_markers":
            _fix_tax_categories_from_price_line_markers(extracted, unified_text)
        elif repair == "single_bag_standard_split":
            if items:
                _apply_single_bag_standard_rate_split(items, merged_rate_bases)
        elif repair == "rebalance_rate_bases":
            if items:
                _rebalance_tax_categories_to_rate_bases(
                    items,
                    unified_text,
                    extracted.get("taxes"),
                    merged_rate_bases,
                )
        elif repair == "rebalance_standard_from_reduced_markers":
            if items:
                _rebalance_standard_categories_from_reduced_rate_markers(
                    items,
                    unified_text,
                    merged_rate_bases,
                )
        elif repair == "nonfood_packaging":
            if items:
                _fix_nonfood_packaging_tax_categories(items, unified_text, merged_rate_bases)
        elif repair == "single_standard_from_small_base":
            if items:
                _assign_single_standard_rate_from_small_base(items, merged_rate_bases)
        elif repair == "normalize_tax_entries":
            _normalize_taxes(extracted, unified_text, ocr_totals)
        elif repair == "restore_item_rate_sum_tax_entries":
            _restore_tax_entries_from_item_rate_sums(
                extracted,
                unified_text,
                ocr_totals,
                merged_rate_bases,
            )
        else:
            raise ValueError(f"Unknown tax category assignment repair: {repair}")
    return merged_rate_bases


def _run_bag_item_rate_base_reconciliation_phase(
    extracted: dict,
    unified_text: str,
    rate_bases: dict | None,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: tiny printed 10% rate base with paid bag item rows.

    Invariant: paid-bag qty, unit_price, and total may change only when their
    combined total reconciles to the visible 10% rate base.
    """
    for repair in repairs:
        if repair == "bag_item_prices_from_rate_bases":
            _fix_bag_item_prices_from_rate_bases(
                extracted,
                rate_bases or {},
                unified_text,
            )
        else:
            raise ValueError(
                f"Unknown bag item rate-base reconciliation repair: {repair}"
            )


def _run_line_item_cleanup_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
    ocr_layout_blocks: list[dict] | None = None,
) -> None:
    """Trigger: OCR cleanup exposes recoverable, duplicate, numeric, or non-product rows.

    Invariant: cleanup may recover or drop rows only when visible OCR/layout
    context supports the change without breaking item-total consistency.
    """
    for repair in repairs:
        if repair == "broad_ocr_line_item_repair":
            _fix_line_items(
                extracted,
                unified_text,
                ocr_layout_blocks=ocr_layout_blocks,
            )
        elif repair == "drop_duplicate_embedded_price":
            if extracted.get("line_items"):
                _drop_duplicate_with_embedded_price(extracted["line_items"])
        elif repair == "drop_non_product_line_items":
            _drop_non_product_line_items(extracted, unified_text)
        elif repair == "drop_numeric_marker_description_rows":
            _drop_numeric_marker_description_rows(extracted, unified_text)
        else:
            raise ValueError(f"Unknown line-item cleanup repair: {repair}")


def _run_phantom_tax_amount_cleanup_phase(extracted: dict) -> None:
    """Trigger: OCR tax amounts are parsed as corrupted duplicate item rows.

    Invariant: a dropped row must match a printed tax amount and have a
    suffix-corrupted description whose clean sibling carries that suffix total.
    """
    _drop_phantom_from_tax_amount(extracted)


def _run_subtotal_item_price_repair_phase(
    extracted: dict,
    unified_text: str,
    ocr_totals: dict,
) -> None:
    """Trigger: OCR subtotal conflicts with parsed item sum and nearby prices.

    Invariant: item price changes must use nearby OCR price evidence and
    strictly improve the item-sum gap to OCR/canonical subtotal.
    """
    _fix_items_from_subtotal(extracted, unified_text, ocr_totals)


def _run_implausible_tax_amount_repair_phase(
    extracted: dict,
    unified_text: str,
    ocr_totals: dict,
) -> None:
    """Trigger: OCR rate-base evidence shows tax amount/base column swap.

    Invariant: tax amount changes must correct an implausible amount to the
    rate-derived value for the printed base and receipt tax inclusion mode.
    """
    _fix_implausible_tax_amounts(extracted, unified_text, ocr_totals)


def _run_vertical_price_qty_total_projection_phase(
    extracted: dict,
    unified_text: str,
) -> None:
    """Trigger: OCR prints repeated name/unit/qty/line-total item blocks.

    Invariant: projected rows must satisfy unit*qty totals and their row sum
    must reconcile to subtotal, total, or total-minus-tax arithmetic.
    """
    _replace_vertical_price_qty_total_rows_when_balanced(extracted, unified_text)


def _run_single_item_quantity_repair_phase(
    extracted: dict,
    unified_text: str,
) -> None:
    """Trigger: a single parsed item has nearby OCR @unit x qty notation.

    Invariant: qty and unit_price may change only when unit * qty reconciles
    to the item total or receipt total already present on the extraction.
    """
    _fix_single_item_qty_from_ocr(extracted, unified_text)


def _run_item_name_price_cleanup_phase(extracted: dict, unified_text: str) -> None:
    """Trigger: visible OCR names or embedded price suffixes repair item rows.

    Invariant: description, qty, unit_price, and total changes must be backed
    by visible OCR row ownership or embedded price field consistency.
    """
    _fix_non_bag_items_named_as_bag(extracted, unified_text)
    _fix_embedded_price_suffix_totals(extracted, unified_text)


def _run_priced_name_item_repair_phase(extracted: dict, unified_text: str) -> None:
    """Trigger: item text prints its own N円 price plus an unmatched OCR amount.

    Invariant: unit/qty/total changes must consume visible OCR amount evidence
    and strictly improve subtotal or total item-sum arithmetic.
    """
    _fix_priced_in_name_items(extracted, unified_text)


def _run_digit_misread_item_repair_phase(extracted: dict, unified_text: str) -> None:
    """Trigger: a small item-sum gap matches one OCR digit-confusion marker.

    Invariant: item total changes require exactly one candidate whose corrected
    amount closes the subtotal/total gap and whose OCR row exposes the marker.
    """
    _fix_digit_misread_items(extracted, unified_text)


def _run_code_prefixed_description_cleanup_phase(extracted: dict) -> None:
    """Trigger: visible OCR/POS code prefixes remain in item descriptions.

    Invariant: cleanup may change only item description text when stripping a
    generic code prefix preserves a Japanese product-name field.
    """
    _clean_code_prefixed_item_descriptions(extracted)


def _run_duplicate_row_cleanup_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: duplicated parsed rows whose item text appears once in OCR.

    Invariant: removing one row is allowed only when subtotal overage matches
    that row total, proving the remaining item sum reconciles to subtotal.
    """
    for repair in repairs:
        if repair == "drop_duplicate_rows_when_subtotal_balances":
            _drop_duplicate_rows_when_subtotal_balances(extracted, unified_text)
        else:
            raise ValueError(f"Unknown duplicate row cleanup repair: {repair}")


def _run_basket_marker_rows_phase(extracted: dict, unified_text: str) -> None:
    """Trigger: OCR basket markers and stacked name/price rows are visible.

    Invariant: projected basket rows must balance against printed subtotal,
    total, coupon, and rate-base arithmetic before replacing line items.
    """
    _replace_basket_marker_rows_when_balanced(extracted, unified_text)


def _run_merchant_identity_repair_phase(extracted: dict, unified_text: str) -> None:
    """Trigger: OCR header exposes legal-name, authority, or brand evidence.

    Invariant: merchant changes must be backed by visible header text and keep
    the merchant field consistent with valid receipt identity candidates.
    """
    _fix_company_name_merchant(extracted, unified_text)


def _run_header_location_repair_phase(extracted: dict, unified_text: str) -> None:
    """Trigger: OCR header, split address, or labeled purchase-site rows.

    Invariant: location changes must be backed by visible header/address/site
    evidence and preserve a more specific existing location when present.
    """
    _fix_header_store_line_location(extracted, unified_text)
    _fix_split_address_location_from_ocr(extracted, unified_text)
    _recover_ascii_brand_header_location(extracted, unified_text)
    _recover_labeled_purchase_site_location(extracted, unified_text)


def _run_bag_item_ocr_repair_phase(extracted: dict, unified_text: str) -> None:
    """Trigger: visible small item/bag price rows in OCR layout.

    Invariant: repaired unit prices, quantities, descriptions, and totals must
    remain backed by nearby item/bag OCR evidence and item total consistency.
    """
    _fix_small_non_bag_item_prices_from_ocr(extracted, unified_text)
    _fix_bag_item_prices_from_ocr(extracted, unified_text)
    _fix_split_bag_price_from_nearby_single_digit(extracted, unified_text)
    _fix_small_bag_description_from_ocr_entry(extracted, unified_text)


def _run_transaction_datetime_repair_phase(
    extracted: dict,
    unified_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: OCR-visible transaction date labels or date-line time anchors.

    Invariant: date/time changes must come from plausible transaction date
    evidence or date-line anchored time evidence, avoiding expiry and business
    hours contexts.
    """
    for repair in repairs:
        if repair == "transaction_date":
            _fix_date(extracted, unified_text)
        elif repair == "transaction_time":
            _fix_time(extracted, unified_text)
        else:
            raise ValueError(f"Unknown transaction datetime repair: {repair}")


def _run_financial_totals_repair_phase(
    extracted: dict,
    ocr_totals: dict,
    ocr_conf: float,
    llm_conf: dict | None,
) -> None:
    """Trigger: reliable OCR subtotal, total, or tax summary values are present.

    Invariant: subtotal, total, and tax overrides must preserve OCR confidence
    policy and subtotal/total/tax arithmetic consistency.
    """
    _apply_financial_overrides(extracted, ocr_totals, ocr_conf, llm_conf)
