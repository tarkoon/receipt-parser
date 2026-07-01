"""Receipt postprocess orchestration."""

import re

from .receipt_financial import extract_rate_bases
from .receipt_phase_trace import (
    _record_receipt_phase_mutation,
    _snapshot_receipt_mutation_fields,
)
from .receipt_postprocess_phases import (
    _run_barcode_row_projection_phase,
    _run_dense_item_row_projection_phase,
    _run_dense_sequence_row_projection_phase,
    _run_jan_pos_row_projection_phase,
    _run_campaign_discount_projection_phase,
    _run_quantity_detail_reconciliation_phase,
    _run_cash_tender_reconciliation_phase,
    _run_payment_method_repair_phase,
    _run_toll_payment_reference_repair_phase,
    _run_service_receipt_recovery_phase,
    _run_body_total_layout_reconstruction_phase,
    _run_coupon_discount_projection_phase,
    _run_discount_consistency_reconciliation_phase,
    _run_following_ocr_price_projection_phase,
    _run_discounted_ocr_item_repair_phase,
    _run_ocr_description_reconciliation_phase,
    _run_split_price_block_projection_phase,
    _run_gap_item_recovery_phase,
    _run_prefixed_tax_marker_item_rows_phase,
    _run_low_value_bag_recovery_phase,
    _run_adjacent_price_shift_reconciliation_phase,
    _run_bag_amount_shift_reconciliation_phase,
    _run_payment_points_reconciliation_phase,
    _run_stacked_inclusive_tax_restoration_phase,
    _run_stacked_name_price_projection_phase,
    _run_single_rate_inclusive_tax_restoration_phase,
    _run_tax_excluded_rate_block_restoration_phase,
    _run_explicit_tax_amount_restoration_phase,
    _run_printed_summary_total_tax_repair_phase,
    _run_printed_item_sum_total_repair_phase,
    _run_printed_external_tax_amount_restoration_phase,
    _run_bare_number_tax_summary_restoration_phase,
    _run_external_tax_total_restoration_phase,
    _run_small_target_only_tax_pruning_phase,
    _run_tax_category_assignment_phase,
    _run_bag_item_rate_base_reconciliation_phase,
    _run_line_item_cleanup_phase,
    _run_phantom_tax_amount_cleanup_phase,
    _run_subtotal_item_price_repair_phase,
    _run_implausible_tax_amount_repair_phase,
    _run_vertical_price_qty_total_projection_phase,
    _run_single_item_quantity_repair_phase,
    _run_item_name_price_cleanup_phase,
    _run_priced_name_item_repair_phase,
    _run_digit_misread_item_repair_phase,
    _run_code_prefixed_description_cleanup_phase,
    _run_duplicate_row_cleanup_phase,
    _run_basket_marker_rows_phase,
    _run_merchant_identity_repair_phase,
    _run_header_location_repair_phase,
    _run_bag_item_ocr_repair_phase,
    _run_transaction_datetime_repair_phase,
    _run_financial_totals_repair_phase,
    _has_printed_vertical_tax_block,
    _tax_assignment_rate_bases,
)

from .receipt_item_cleanup import (
    _ensure_discounted_ocr_pairs_present,
    _fill_single_qty_unit_prices_from_totals,
)
from .receipt_item_repair import _extract_fuel_usage
from .receipt_totals import (
    _items_plus_tax_matches_total,
    _sum_taxable_amounts,
)


def postprocess_receipt(
    extracted: dict,
    unified_text: str,
    ocr_conf: float,
    ocr_totals: dict,
    llm_conf: dict | None,
    model: str,
    ocr_layout_blocks: list[dict] | None = None,
    mutation_trace: list[dict] | None = None,
) -> dict:
    """Apply all receipt-specific post-processing to the LLM extraction."""
    trace_snapshot = (
        _snapshot_receipt_mutation_fields(extracted)
        if mutation_trace is not None
        else None
    )
    _run_merchant_identity_repair_phase(extracted, unified_text)
    _run_body_total_layout_reconstruction_phase(
        extracted,
        unified_text,
        ("split_item_price_body_total",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "body_total_layout_reconstruction",
        trace_snapshot,
        extracted,
    )
    _run_financial_totals_repair_phase(extracted, ocr_totals, ocr_conf, llm_conf)
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "financial_totals_repair",
        trace_snapshot,
        extracted,
    )
    _run_cash_tender_reconciliation_phase(
        extracted,
        unified_text,
        ("stacked_cash_tender", "unlabeled_cash_tender_change"),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "cash_tender_reconciliation",
        trace_snapshot,
        extracted,
    )
    _run_implausible_tax_amount_repair_phase(extracted, unified_text, ocr_totals)
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "implausible_tax_amount_repair",
        trace_snapshot,
        extracted,
    )
    _run_transaction_datetime_repair_phase(
        extracted,
        unified_text,
        ("transaction_date", "transaction_time"),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "transaction_datetime_repair",
        trace_snapshot,
        extracted,
    )
    _run_header_location_repair_phase(extracted, unified_text)
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "header_identity_repair",
        trace_snapshot,
        extracted,
    )
    _run_payment_method_repair_phase(extracted, unified_text, ocr_conf, llm_conf)
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "payment_method_repair",
        trace_snapshot,
        extracted,
    )
    _run_toll_payment_reference_repair_phase(extracted, unified_text)
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "toll_payment_reference_repair",
        trace_snapshot,
        extracted,
    )
    _run_service_receipt_recovery_phase(
        extracted,
        unified_text,
        ("bare_service_without_itemization",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "service_receipt_recovery",
        trace_snapshot,
        extracted,
    )
    _run_line_item_cleanup_phase(
        extracted,
        unified_text,
        ("broad_ocr_line_item_repair",),
        ocr_layout_blocks=ocr_layout_blocks,
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "item_cleanup",
        trace_snapshot,
        extracted,
    )
    _run_quantity_detail_reconciliation_phase(
        extracted,
        unified_text,
        ("empty_qty_unit_total_block",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "quantity_detail_reconciliation",
        trace_snapshot,
        extracted,
    )
    _run_phantom_tax_amount_cleanup_phase(extracted)
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "phantom_tax_amount_cleanup",
        trace_snapshot,
        extracted,
    )
    _run_priced_name_item_repair_phase(extracted, unified_text)
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "priced_name_item_repair",
        trace_snapshot,
        extracted,
    )
    _run_bag_item_ocr_repair_phase(extracted, unified_text)
    _run_subtotal_item_price_repair_phase(extracted, unified_text, ocr_totals)
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "subtotal_item_price_repair",
        trace_snapshot,
        extracted,
    )
    _run_gap_item_recovery_phase(extracted, unified_text, ("missing_items",))
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "gap_item_recovery",
        trace_snapshot,
        extracted,
    )
    _run_prefixed_tax_marker_item_rows_phase(
        extracted,
        unified_text,
        ("prefixed_tax_marker_item_rows",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "prefixed_tax_marker_item_rows",
        trace_snapshot,
        extracted,
    )
    _run_service_receipt_recovery_phase(
        extracted,
        unified_text,
        ("bare_service_without_itemization",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "service_receipt_recovery",
        trace_snapshot,
        extracted,
    )
    _run_low_value_bag_recovery_phase(
        extracted,
        unified_text,
        ("missing_bag_items", "overage_low_value_bag", "numeric_description_context"),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "low_value_bag_recovery",
        trace_snapshot,
        extracted,
    )
    _run_ocr_description_reconciliation_phase(
        extracted,
        unified_text,
        ("o_ring_descriptions",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "ocr_description_reconciliation",
        trace_snapshot,
        extracted,
    )
    _run_adjacent_price_shift_reconciliation_phase(
        extracted,
        unified_text,
        ("adjacent_ocr_price_shift",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "adjacent_price_shift_reconciliation",
        trace_snapshot,
        extracted,
    )
    _run_gap_item_recovery_phase(
        extracted,
        unified_text,
        ("discounted_gap", "repeated_gap", "repeated_ocr_block"),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "gap_item_recovery",
        trace_snapshot,
        extracted,
    )
    # Run embedded-price dedup AGAIN after recovery — recovery can pick up
    # OCR-merged 'X  N' lines as new phantom items even when 'X' already
    # exists in the extraction at the same price.
    _run_line_item_cleanup_phase(
        extracted,
        unified_text,
        ("drop_duplicate_embedded_price",),
    )
    _run_ocr_description_reconciliation_phase(
        extracted,
        unified_text,
        ("qty_code_rows", "code_table_order", "duplicate_descriptions"),
    )
    _run_digit_misread_item_repair_phase(extracted, unified_text)
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "digit_misread_item_repair",
        trace_snapshot,
        extracted,
    )
    _run_ocr_description_reconciliation_phase(
        extracted,
        unified_text,
        ("o_ring_descriptions",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "ocr_description_reconciliation",
        trace_snapshot,
        extracted,
    )
    _run_barcode_row_projection_phase(
        extracted,
        unified_text,
        (
            "barcode_unit_qty_amount_stack",
            "barcode_qty_price_rows",
        ),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "barcode_row_projection",
        trace_snapshot,
        extracted,
    )
    _run_dense_sequence_row_projection_phase(
        extracted,
        unified_text,
        ("dense_sequence_rows",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "dense_sequence_row_projection",
        trace_snapshot,
        extracted,
    )
    _run_quantity_detail_reconciliation_phase(
        extracted,
        unified_text,
        ("following_qty_detail",),
    )
    _run_prefixed_tax_marker_item_rows_phase(
        extracted,
        unified_text,
        ("prefixed_tax_marker_item_rows",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "prefixed_tax_marker_item_rows",
        trace_snapshot,
        extracted,
    )
    _run_quantity_detail_reconciliation_phase(
        extracted,
        unified_text,
        ("qty_totals_from_unit_lines", "qty_context_and_reduced_rate"),
    )
    _run_bag_amount_shift_reconciliation_phase(
        extracted,
        unified_text,
        ("name_bag_amount_shift",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "bag_amount_shift_reconciliation",
        trace_snapshot,
        extracted,
    )
    _run_line_item_cleanup_phase(
        extracted,
        unified_text,
        ("drop_numeric_marker_description_rows",),
    )
    _run_jan_pos_row_projection_phase(extracted, unified_text, ocr_totals)
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "jan_pos_row_projection",
        trace_snapshot,
        extracted,
    )
    _run_line_item_cleanup_phase(
        extracted,
        unified_text,
        ("drop_duplicate_embedded_price",),
    )
    _run_ocr_description_reconciliation_phase(
        extracted,
        unified_text,
        (
            "qty_code_rows",
            "duplicate_descriptions",
            "colon_split_names",
            "bag_code_context",
        ),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "ocr_description_reconciliation",
        trace_snapshot,
        extracted,
    )
    _run_tax_category_assignment_phase(
        extracted,
        unified_text,
        ocr_totals,
        ("price_line_markers",),
    )
    _run_line_item_cleanup_phase(
        extracted,
        unified_text,
        ("drop_non_product_line_items",),
    )
    _run_gap_item_recovery_phase(extracted, unified_text, ("repeated_ocr_block",))
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "gap_item_recovery",
        trace_snapshot,
        extracted,
    )
    _run_item_name_price_cleanup_phase(extracted, unified_text)
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "item_name_price_cleanup",
        trace_snapshot,
        extracted,
    )
    _run_adjacent_price_shift_reconciliation_phase(
        extracted,
        unified_text,
        ("adjacent_ocr_price_shift",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "adjacent_price_shift_reconciliation",
        trace_snapshot,
        extracted,
    )
    _run_jan_pos_row_projection_phase(extracted, unified_text, ocr_totals)
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "jan_pos_row_projection",
        trace_snapshot,
        extracted,
    )
    _run_barcode_row_projection_phase(
        extracted,
        unified_text,
        (
            "barcode_unit_qty_amount_stack",
            "barcode_qty_price_rows",
        ),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "barcode_row_projection",
        trace_snapshot,
        extracted,
    )
    _run_dense_item_row_projection_phase(
        extracted,
        unified_text,
        ("dense_item_rows",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "dense_item_row_projection",
        trace_snapshot,
        extracted,
    )
    _run_dense_sequence_row_projection_phase(
        extracted,
        unified_text,
        ("dense_sequence_rows",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "dense_sequence_row_projection",
        trace_snapshot,
        extracted,
    )
    _run_quantity_detail_reconciliation_phase(
        extracted,
        unified_text,
        ("following_qty_detail",),
    )
    _run_line_item_cleanup_phase(
        extracted,
        unified_text,
        ("drop_numeric_marker_description_rows",),
    )
    _run_line_item_cleanup_phase(
        extracted,
        unified_text,
        ("drop_duplicate_embedded_price",),
    )
    _run_quantity_detail_reconciliation_phase(
        extracted,
        unified_text,
        ("qty_totals_from_unit_lines", "qty_context_and_reduced_rate"),
    )
    _run_bag_amount_shift_reconciliation_phase(
        extracted,
        unified_text,
        ("name_bag_amount_shift",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "bag_amount_shift_reconciliation",
        trace_snapshot,
        extracted,
    )
    _run_line_item_cleanup_phase(
        extracted,
        unified_text,
        ("drop_numeric_marker_description_rows",),
    )
    _run_gap_item_recovery_phase(extracted, unified_text, ("discounted_gap",))
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "gap_item_recovery",
        trace_snapshot,
        extracted,
    )
    _run_adjacent_price_shift_reconciliation_phase(
        extracted,
        unified_text,
        ("adjacent_ocr_price_shift",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "adjacent_price_shift_reconciliation",
        trace_snapshot,
        extracted,
    )
    _run_gap_item_recovery_phase(extracted, unified_text, ("repeated_gap",))
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "gap_item_recovery",
        trace_snapshot,
        extracted,
    )
    _run_low_value_bag_recovery_phase(
        extracted,
        unified_text,
        ("missing_bag_items", "overage_low_value_bag", "numeric_description_context"),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "low_value_bag_recovery",
        trace_snapshot,
        extracted,
    )
    _run_ocr_description_reconciliation_phase(
        extracted,
        unified_text,
        ("o_ring_descriptions", "duplicate_descriptions"),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "ocr_description_reconciliation",
        trace_snapshot,
        extracted,
    )
    _run_line_item_cleanup_phase(
        extracted,
        unified_text,
        (
            "drop_non_product_line_items",
            "drop_duplicate_embedded_price",
            "drop_numeric_marker_description_rows",
        ),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "item_cleanup",
        trace_snapshot,
        extracted,
    )
    _run_service_receipt_recovery_phase(
        extracted,
        unified_text,
        ("service_table_items",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "service_receipt_recovery",
        trace_snapshot,
        extracted,
    )
    _run_line_item_cleanup_phase(
        extracted,
        unified_text,
        ("drop_duplicate_embedded_price", "drop_numeric_marker_description_rows"),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "item_cleanup",
        trace_snapshot,
        extracted,
    )

    # Clear account_number when it's a masked card number suffix, not a real account
    acct = extracted.get("account_number")
    if acct and re.search(r'\*{2,}' + re.escape(str(acct)), unified_text):
        extracted["account_number"] = None

    # Tax categories
    rate_bases = None
    if extracted.get("line_items"):
        rate_bases = _tax_assignment_rate_bases(unified_text, ocr_totals)
        _run_bag_item_rate_base_reconciliation_phase(
            extracted,
            unified_text,
            rate_bases,
            ("bag_item_prices_from_rate_bases",),
        )
        trace_snapshot = _record_receipt_phase_mutation(
            mutation_trace,
            "bag_item_rate_base_reconciliation",
            trace_snapshot,
            extracted,
        )
        _run_tax_category_assignment_phase(
            extracted,
            unified_text,
            ocr_totals,
            (
                "assign_tax_categories",
                "ocr_markers",
                "price_line_markers",
                "single_bag_standard_split",
                "rebalance_rate_bases",
                "rebalance_standard_from_reduced_markers",
                "nonfood_packaging",
                "ocr_markers",
                "single_bag_standard_split",
                "single_standard_from_small_base",
            ),
            rate_bases=rate_bases,
        )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "tax_category_assignment",
        trace_snapshot,
        extracted,
    )

    _run_service_receipt_recovery_phase(
        extracted,
        unified_text,
        ("single_service_inclusive_tax",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "service_receipt_recovery",
        trace_snapshot,
        extracted,
    )
    _run_single_rate_inclusive_tax_restoration_phase(
        extracted,
        unified_text,
        ("printed_inclusive_tax_structural_blocks",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "single_rate_inclusive_tax_restoration",
        trace_snapshot,
        extracted,
    )
    _run_tax_excluded_rate_block_restoration_phase(
        extracted,
        unified_text,
        ("tax_excluded_per_rate_blocks",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "tax_excluded_rate_block_restoration",
        trace_snapshot,
        extracted,
    )
    _run_single_rate_inclusive_tax_restoration_phase(
        extracted,
        unified_text,
        ("single_rate_inclusive_tax_block",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "single_rate_inclusive_tax_restoration",
        trace_snapshot,
        extracted,
    )

    # Drop any remaining tax entries with amount=0 (LLM-supplied unhandled).
    # Exempt the 0% / 非課税 entry: truth files keep it (rate '0%' may have
    # amount=0 since there's no tax to record on a non-taxable line).
    if extracted.get("taxes"):
        extracted["taxes"] = [
            t for t in extracted["taxes"]
            if not isinstance(t, dict)
            or (t.get("amount") or 0) > 0
            or t.get("rate") == "0%"
            or "非課税" in (t.get("label") or "")
        ]
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "tax_category_assignment",
        trace_snapshot,
        extracted,
    )
    has_printed_vertical_tax_block = _has_printed_vertical_tax_block(unified_text)
    _run_payment_points_reconciliation_phase(
        extracted,
        unified_text,
        ocr_conf,
        llm_conf,
        ("points_used", "points_payment"),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "payment_points_reconciliation",
        trace_snapshot,
        extracted,
    )

    # Fix pre-tax item totals for inclusive-tax receipts
    if extracted.get("line_items") and extracted.get("total"):
        item_sum = sum(i.get("total", 0) for i in extracted["line_items"] if isinstance(i, dict))
        receipt_total = extracted["total"]
        # Skip adjustment when taxes account for the difference (exclusive tax)
        tax_total = _sum_taxable_amounts(extracted.get("taxes", []))
        items_are_pretax = tax_total > 0 and abs(item_sum + tax_total - receipt_total) < 2
        if len(extracted["line_items"]) == 1 and abs(item_sum - receipt_total) > 1 and not items_are_pretax:
            item = extracted["line_items"][0]
            if isinstance(item, dict) and abs(item_sum * 1.10 - receipt_total) < 2:
                item["total"] = receipt_total
                if item.get("unit_price") and abs(item["unit_price"] - item_sum) < 1:
                    item["unit_price"] = receipt_total
            elif isinstance(item, dict) and abs(item_sum * 1.08 - receipt_total) < 2:
                item["total"] = receipt_total
                if item.get("unit_price") and abs(item["unit_price"] - item_sum) < 1:
                    item["unit_price"] = receipt_total
    _run_tax_category_assignment_phase(
        extracted,
        unified_text,
        ocr_totals,
        ("restore_item_rate_sum_tax_entries", "normalize_tax_entries"),
        rate_bases=rate_bases,
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "tax_category_assignment",
        trace_snapshot,
        extracted,
    )
    _run_explicit_tax_amount_restoration_phase(
        extracted,
        unified_text,
        ("explicit_tax_rate_amount_lines",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "explicit_tax_amount_restoration",
        trace_snapshot,
        extracted,
    )
    _run_printed_summary_total_tax_repair_phase(
        extracted,
        unified_text,
        ("printed_summary_total_tax_balanced",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "printed_summary_total_tax_repair",
        trace_snapshot,
        extracted,
    )
    _run_printed_item_sum_total_repair_phase(
        extracted,
        unified_text,
        ("printed_item_sum_total",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "printed_item_sum_total_repair",
        trace_snapshot,
        extracted,
    )

    # Fix tax amounts when OCR taxes are missing
    if (extracted.get("taxes") and extracted.get("line_items")
            and extracted.get("total") and not ocr_totals.get("taxes")
            and not has_printed_vertical_tax_block):
        rate_sums: dict[str, float] = {}
        for item in extracted["line_items"]:
            cat = item.get("tax_category", "0%")
            rate_sums[cat] = rate_sums.get(cat, 0) + (item.get("total") or 0)
        all_labels = [t.get("label", "") for t in extracted["taxes"]]
        all_inclusive_labels = all_labels and all(
            (lbl or '').startswith('内') or lbl == '消費税等' for lbl in all_labels)
        if all_inclusive_labels:
            for t in extracted["taxes"]:
                rate = t.get("rate", "0%")
                rate_pct = float(rate.replace('%', '')) / 100.0
                amt = t.get("amount", 0)
                cat_sum = rate_sums.get(rate, 0)
                if rate_pct > 0 and cat_sum > 0:
                    # When tax amount equals item sum, it's a base not a tax
                    if amt > 0 and abs(amt - cat_sum) < 2:
                        t["amount"] = round(cat_sum * rate_pct / (1 + rate_pct))
                    # For inclusive items (item_sum ≈ total), recompute from items
                    elif abs(sum(rate_sums.values()) - extracted["total"]) < 5:
                        computed = round(cat_sum * rate_pct / (1 + rate_pct))
                        if computed != amt:
                            t["amount"] = computed

    # Fallback: recompute tax amounts from OCR rate bases
    if (
        extracted.get("taxes")
        and not ocr_totals.get("taxes")
        and not _items_plus_tax_matches_total(extracted)
        and not has_printed_vertical_tax_block
    ):
        rb = extract_rate_bases(unified_text)
        bb = ocr_totals.get('_breakdown_rate_bases', {})
        for rate, base in bb.items():
            if rate not in rb or rb[rate] is None:
                rb[rate] = base
        rb_sum = sum(v for v in rb.values() if v and v > 0)
        bases_are_inclusive = abs(rb_sum - (extracted.get("total") or 0)) < 5
        for t in extracted["taxes"]:
            rate = t.get("rate", "0%")
            rate_pct = float(rate.replace('%', '')) / 100.0
            base = rb.get(rate)
            if rate_pct > 0 and base and base > 0:
                if bases_are_inclusive:
                    computed = round(base * rate_pct / (1 + rate_pct))
                else:
                    computed = round(base * rate_pct)
                if abs(t.get("amount", 0) - computed) > 2:
                    t["amount"] = computed

    _run_single_rate_inclusive_tax_restoration_phase(
        extracted,
        unified_text,
        ("printed_inclusive_tax_structural_blocks",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "single_rate_inclusive_tax_restoration",
        trace_snapshot,
        extracted,
    )

    # Universal subtotal rule: subtotal = total - sum(taxes), regardless of
    # 内税 / 外税. Pre-tax base is the canonical definition; for 内税 receipts
    # this means subtotal != sum(line_items) (line items are post-tax) which is
    # expected and validated.
    #
    # Preserve an existing subtotal when it's close to the computed value —
    # this guards against 1-2 yen rounding flips when the tax was extracted
    # with a small rounding error. The printed subtotal is authoritative for
    # the receipt's own internal rounding choice.
    if extracted.get("total") is not None:
        tax_sum = _sum_taxable_amounts(extracted.get("taxes") or [])
        computed_sub = extracted["total"] - tax_sum
        if computed_sub >= 0:
            existing_sub = extracted.get("subtotal")
            close_to_computed = (
                existing_sub is not None
                and abs(existing_sub - computed_sub) <= 5
            )
            close_to_pretax_via_tax_only = (
                existing_sub is not None
                and tax_sum > 0
                and abs(existing_sub + tax_sum - extracted["total"]) <= 5
            )
            if close_to_computed or close_to_pretax_via_tax_only:
                pass  # keep the printed/extracted value
            else:
                extracted["subtotal"] = computed_sub
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "payment_points_reconciliation",
        trace_snapshot,
        extracted,
    )

    _extract_fuel_usage(extracted, unified_text)
    if extracted.get("line_items"):
        _run_single_item_quantity_repair_phase(extracted, unified_text)
        trace_snapshot = _record_receipt_phase_mutation(
            mutation_trace,
            "single_item_quantity_repair",
            trace_snapshot,
            extracted,
        )
    _run_body_total_layout_reconstruction_phase(
        extracted,
        unified_text,
        ("split_item_price_body_total",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "body_total_layout_reconstruction",
        trace_snapshot,
        extracted,
    )
    _run_gap_item_recovery_phase(extracted, unified_text, ("discounted_gap",))
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "gap_item_recovery",
        trace_snapshot,
        extracted,
    )
    _run_adjacent_price_shift_reconciliation_phase(
        extracted,
        unified_text,
        ("adjacent_ocr_price_shift",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "adjacent_price_shift_reconciliation",
        trace_snapshot,
        extracted,
    )
    _run_gap_item_recovery_phase(extracted, unified_text, ("repeated_gap",))
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "gap_item_recovery",
        trace_snapshot,
        extracted,
    )
    _run_discount_consistency_reconciliation_phase(
        extracted,
        unified_text,
        (
            "discounted_item_gross_prices",
            "following_discount_lines",
        ),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "discount_consistency_reconciliation",
        trace_snapshot,
        extracted,
    )
    _run_coupon_discount_projection_phase(
        extracted,
        unified_text,
        (
            "coupon_discount_blocks",
            "drop_applied_coupon_line_items",
        ),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "coupon_discount_projection",
        trace_snapshot,
        extracted,
    )
    _run_following_ocr_price_projection_phase(
        extracted,
        unified_text,
        ("tiny_item_prices_from_following_ocr",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "following_ocr_price_projection",
        trace_snapshot,
        extracted,
    )
    _ensure_discounted_ocr_pairs_present(extracted, unified_text)
    _run_gap_item_recovery_phase(extracted, unified_text, ("missing_items",))
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "gap_item_recovery",
        trace_snapshot,
        extracted,
    )
    _run_vertical_price_qty_total_projection_phase(extracted, unified_text)
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "vertical_price_qty_total_projection",
        trace_snapshot,
        extracted,
    )
    _run_jan_pos_row_projection_phase(extracted, unified_text, ocr_totals)
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "jan_pos_row_projection",
        trace_snapshot,
        extracted,
    )
    _run_barcode_row_projection_phase(
        extracted,
        unified_text,
        (
            "barcode_unit_qty_amount_stack",
            "barcode_qty_price_rows",
        ),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "barcode_row_projection",
        trace_snapshot,
        extracted,
    )
    _run_dense_item_row_projection_phase(
        extracted,
        unified_text,
        ("dense_item_rows",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "dense_item_row_projection",
        trace_snapshot,
        extracted,
    )
    _run_dense_sequence_row_projection_phase(
        extracted,
        unified_text,
        ("dense_sequence_rows",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "dense_sequence_row_projection",
        trace_snapshot,
        extracted,
    )
    _run_quantity_detail_reconciliation_phase(
        extracted,
        unified_text,
        ("qty_totals_from_unit_lines", "qty_context_and_reduced_rate"),
    )
    _run_bag_amount_shift_reconciliation_phase(
        extracted,
        unified_text,
        ("name_bag_amount_shift",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "bag_amount_shift_reconciliation",
        trace_snapshot,
        extracted,
    )
    _run_line_item_cleanup_phase(
        extracted,
        unified_text,
        ("drop_numeric_marker_description_rows",),
    )
    _run_cash_tender_reconciliation_phase(
        extracted,
        unified_text,
        ("stacked_cash_tender", "unlabeled_cash_tender_change"),
    )
    _run_low_value_bag_recovery_phase(
        extracted,
        unified_text,
        ("overage_low_value_bag", "numeric_description_context"),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "low_value_bag_recovery",
        trace_snapshot,
        extracted,
    )
    _run_ocr_description_reconciliation_phase(
        extracted,
        unified_text,
        (
            "o_ring_descriptions",
            "duplicate_descriptions",
            "colon_split_names",
            "bag_code_context",
        ),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "ocr_description_reconciliation",
        trace_snapshot,
        extracted,
    )
    _run_tax_category_assignment_phase(
        extracted,
        unified_text,
        ocr_totals,
        ("price_line_markers",),
    )
    _run_line_item_cleanup_phase(
        extracted,
        unified_text,
        ("drop_non_product_line_items",),
    )
    _run_service_receipt_recovery_phase(
        extracted,
        unified_text,
        ("service_table_items",),
    )
    _run_low_value_bag_recovery_phase(
        extracted,
        unified_text,
        ("missing_low_value_bag_gap", "missing_bag_items"),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "low_value_bag_recovery",
        trace_snapshot,
        extracted,
    )
    _run_tax_category_assignment_phase(
        extracted,
        unified_text,
        ocr_totals,
        ("single_standard_from_small_base",),
    )
    _run_tax_category_assignment_phase(
        extracted,
        unified_text,
        ocr_totals,
        ("normalize_tax_entries",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "tax_category_assignment",
        trace_snapshot,
        extracted,
    )
    _run_explicit_tax_amount_restoration_phase(
        extracted,
        unified_text,
        ("explicit_tax_rate_amount_lines",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "explicit_tax_amount_restoration",
        trace_snapshot,
        extracted,
    )
    _run_tax_excluded_rate_block_restoration_phase(
        extracted,
        unified_text,
        ("tax_excluded_per_rate_blocks",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "tax_excluded_rate_block_restoration",
        trace_snapshot,
        extracted,
    )
    _run_single_rate_inclusive_tax_restoration_phase(
        extracted,
        unified_text,
        ("single_rate_inclusive_tax_block",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "single_rate_inclusive_tax_restoration",
        trace_snapshot,
        extracted,
    )
    _run_external_tax_total_restoration_phase(
        extracted,
        unified_text,
        ("external_tax_total_from_printed_subtotal",),
    )
    if extracted.get("total") is not None:
        tax_sum = _sum_taxable_amounts(extracted.get("taxes") or [])
        computed_sub = extracted["total"] - tax_sum
        if computed_sub >= 0:
            extracted["subtotal"] = computed_sub
    _run_low_value_bag_recovery_phase(
        extracted,
        unified_text,
        ("missing_low_value_bag_gap",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "low_value_bag_recovery",
        trace_snapshot,
        extracted,
    )
    _run_tax_category_assignment_phase(
        extracted,
        unified_text,
        ocr_totals,
        ("single_bag_standard_split",),
        rate_bases=extract_rate_bases(unified_text),
    )
    _run_line_item_cleanup_phase(
        extracted,
        unified_text,
        ("drop_non_product_line_items",),
    )
    _run_quantity_detail_reconciliation_phase(
        extracted,
        unified_text,
        ("empty_qty_unit_total_block",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "quantity_detail_reconciliation",
        trace_snapshot,
        extracted,
    )
    _run_dense_sequence_row_projection_phase(
        extracted,
        unified_text,
        ("dense_sequence_rows",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "dense_sequence_row_projection",
        trace_snapshot,
        extracted,
    )
    _run_quantity_detail_reconciliation_phase(
        extracted,
        unified_text,
        ("qty_totals_from_unit_lines",),
    )
    _run_line_item_cleanup_phase(
        extracted,
        unified_text,
        ("drop_numeric_marker_description_rows",),
    )
    if extracted.get("line_items"):
        final_rate_bases = extract_rate_bases(unified_text)
        _run_tax_category_assignment_phase(
            extracted,
            unified_text,
            ocr_totals,
            (
                "ocr_markers",
                "rebalance_rate_bases",
                "rebalance_standard_from_reduced_markers",
                "nonfood_packaging",
                "single_bag_standard_split",
                "price_line_markers",
                "ocr_markers",
                "rebalance_rate_bases",
            ),
            rate_bases=final_rate_bases,
        )
        _run_quantity_detail_reconciliation_phase(
            extracted,
            unified_text,
            ("qty_context_and_reduced_rate",),
        )
    _run_ocr_description_reconciliation_phase(
        extracted,
        unified_text,
        (
            "o_ring_descriptions",
            "code_table_order",
            "duplicate_descriptions",
            "colon_split_names",
            "bag_code_context",
        ),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "ocr_description_reconciliation",
        trace_snapshot,
        extracted,
    )
    _run_discount_consistency_reconciliation_phase(
        extracted,
        unified_text,
        ("following_discount_lines",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "discount_consistency_reconciliation",
        trace_snapshot,
        extracted,
    )
    _run_coupon_discount_projection_phase(
        extracted,
        unified_text,
        (
            "coupon_discount_blocks",
            "drop_applied_coupon_line_items",
        ),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "coupon_discount_projection",
        trace_snapshot,
        extracted,
    )
    _run_following_ocr_price_projection_phase(
        extracted,
        unified_text,
        ("tiny_item_prices_from_following_ocr",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "following_ocr_price_projection",
        trace_snapshot,
        extracted,
    )
    _run_split_price_block_projection_phase(
        extracted,
        unified_text,
        ("split_price_block",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "split_price_block_projection",
        trace_snapshot,
        extracted,
    )
    _run_quantity_detail_reconciliation_phase(
        extracted,
        unified_text,
        ("qty_context_and_reduced_rate",),
    )
    _run_stacked_name_price_projection_phase(extracted, unified_text)
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "stacked_name_price_projection",
        trace_snapshot,
        extracted,
    )
    _run_stacked_inclusive_tax_restoration_phase(extracted, unified_text)
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "stacked_inclusive_tax_restoration",
        trace_snapshot,
        extracted,
    )
    _run_tax_excluded_rate_block_restoration_phase(
        extracted,
        unified_text,
        ("tax_excluded_per_rate_blocks",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "tax_excluded_rate_block_restoration",
        trace_snapshot,
        extracted,
    )
    _run_single_rate_inclusive_tax_restoration_phase(
        extracted,
        unified_text,
        ("single_rate_inclusive_tax_block",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "single_rate_inclusive_tax_restoration",
        trace_snapshot,
        extracted,
    )
    _run_printed_external_tax_amount_restoration_phase(
        extracted,
        unified_text,
        ("printed_external_tax_amounts",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "printed_external_tax_amount_restoration",
        trace_snapshot,
        extracted,
    )
    _run_explicit_tax_amount_restoration_phase(
        extracted,
        unified_text,
        ("explicit_tax_rate_amount_lines",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "explicit_tax_amount_restoration",
        trace_snapshot,
        extracted,
    )
    _run_bare_number_tax_summary_restoration_phase(
        extracted,
        unified_text,
        ("bare_number_tax_summary",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "bare_number_tax_summary_restoration",
        trace_snapshot,
        extracted,
    )
    _run_external_tax_total_restoration_phase(
        extracted,
        unified_text,
        ("external_tax_total_from_printed_subtotal",),
    )
    _run_small_target_only_tax_pruning_phase(
        extracted,
        unified_text,
        ("drop_small_target_only_taxes",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "small_target_only_tax_pruning",
        trace_snapshot,
        extracted,
    )
    _run_line_item_cleanup_phase(
        extracted,
        unified_text,
        ("drop_numeric_marker_description_rows",),
    )
    _run_dense_sequence_row_projection_phase(
        extracted,
        unified_text,
        ("dense_sequence_rows",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "dense_sequence_row_projection",
        trace_snapshot,
        extracted,
    )
    _run_quantity_detail_reconciliation_phase(
        extracted,
        unified_text,
        ("following_qty_detail",),
    )
    _run_campaign_discount_projection_phase(
        extracted,
        unified_text,
        ("campaign_discount_stream",),
    )
    _run_service_receipt_recovery_phase(
        extracted,
        unified_text,
        ("bare_service_without_itemization",),
    )
    _run_quantity_detail_reconciliation_phase(
        extracted,
        unified_text,
        ("qty_totals_from_unit_lines",),
    )
    _run_bag_item_rate_base_reconciliation_phase(
        extracted,
        unified_text,
        extract_rate_bases(unified_text),
        ("bag_item_prices_from_rate_bases",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "bag_item_rate_base_reconciliation",
        trace_snapshot,
        extracted,
    )
    _run_body_total_layout_reconstruction_phase(
        extracted,
        unified_text,
        ("split_item_price_body_total",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "body_total_layout_reconstruction",
        trace_snapshot,
        extracted,
    )
    _run_code_prefixed_description_cleanup_phase(extracted)
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "code_prefixed_description_cleanup",
        trace_snapshot,
        extracted,
    )
    if extracted.get("total") is not None:
        tax_sum = _sum_taxable_amounts(extracted.get("taxes") or [])
        computed_sub = float(extracted["total"]) - tax_sum
        if computed_sub >= 0:
            extracted["subtotal"] = computed_sub
    _run_discounted_ocr_item_repair_phase(extracted, unified_text)
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "discounted_ocr_item_repair",
        trace_snapshot,
        extracted,
    )
    _run_duplicate_row_cleanup_phase(
        extracted,
        unified_text,
        ("drop_duplicate_rows_when_subtotal_balances",),
    )
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "duplicate_row_cleanup",
        trace_snapshot,
        extracted,
    )
    _run_basket_marker_rows_phase(extracted, unified_text)
    trace_snapshot = _record_receipt_phase_mutation(
        mutation_trace,
        "basket_marker_rows",
        trace_snapshot,
        extracted,
    )
    _run_payment_points_reconciliation_phase(
        extracted,
        unified_text,
        ocr_conf,
        llm_conf,
        ("points_payment",),
    )
    _run_external_tax_total_restoration_phase(
        extracted,
        unified_text,
        ("external_tax_total_from_printed_subtotal",),
    )
    if extracted.get("line_items"):
        _fill_single_qty_unit_prices_from_totals(extracted["line_items"])
    _record_receipt_phase_mutation(
        mutation_trace,
        "final_consistency_pass",
        trace_snapshot,
        extracted,
    )

    return extracted
