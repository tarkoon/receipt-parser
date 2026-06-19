"""Guardrails against brittle production receipt parsing code.

Production parsing code must not special-case specific merchants, stores,
receipt IDs, fixture ranges, known dates, known product lists, or known final
totals. Production code may implement general layout/format strategies only
when they are triggered by structural OCR evidence and validated by
arithmetic/format invariants.

This test intentionally allows known violations documented in
pipeline_brittleness_audit.md so the current tree can still run the guardrail.
The allowlist is exact and should shrink as those production branches are
removed or replaced with general parsers.
"""

from __future__ import annotations

import ast
import copy
import io
import re
import subprocess
import tokenize
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARSER_DIR = ROOT / "src" / "receipt_parser"
SCANNED_FILES = tuple(
    sorted({PARSER_DIR / "pipeline.py", *PARSER_DIR.glob("pipeline_*.py")})
)

MERCHANT_OR_STORE_RE = re.compile(
    r"("
    r"maxvalu|max_value|familymart|family_mart|daiso|costco|gyomu|"
    r"starbucks|donki|seria|cosmos|nafco|nishimatsuya|yakitori"
    r")",
    re.IGNORECASE,
)
FIXTURE_REFERENCE_RE = re.compile(
    r"("
    r"receipt[_-]?\d+|"
    r"target[_-]?\d+(?:[_-]to[_-]\d+|[_-]\d+)?|"
    r"known[_-]?\d+(?:[_-]to[_-]\d+)?"
    r")",
    re.IGNORECASE,
)
KNOWN_ANSWER_NAME_RE = re.compile(
    r"(^|_)known(_|$)|final_known|known_answer|known_financial",
    re.IGNORECASE,
)
KNOWN_DATE_RE = re.compile(r"\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b")

SEMANTIC_FIELDS = {
    "amount_paid",
    "billing_period",
    "date",
    "line_items",
    "location",
    "merchant",
    "payment_method",
    "points_earned",
    "points_used",
    "subtotal",
    "tax_entries",
    "time",
    "total",
    "usage",
}
FINAL_RESULT_MUTATORS = {
    "_drop_duplicate_with_embedded_price",
    "_replace_barcode_qty_price_rows_when_balanced",
    "fix_final_known_financial_overrides",
    "postprocess_receipt",
}
FINAL_OUTPUT_FUNCTIONS = {
    "_apply_final_receipt_output_repairs",
    "_prepare_receipt_output_payload",
    "_build_result",
}
FINAL_OUTPUT_KNOWN_ANSWER_MUTATORS = {
    "fix_final_known_financial_overrides",
    "postprocess_receipt",
}
BASELINE_COMMIT = "c175c17"
POSTPROCESS_REPAIR_CALL_LIMIT = 5

REPAIR_CALL_PREFIXES = (
    "_append_",
    "_apply_",
    "_clear_",
    "_clean_",
    "_drop_",
    "_fix_",
    "_normalize_",
    "_rebalance_",
    "_recover_",
    "_repair_",
    "_replace_",
    "_restore_",
    "assign_",
    "reconcile_",
)
POSTPROCESS_MUTATOR_REPEAT_ALLOWLIST = {
}
JAN_POS_ROW_PROJECTION_REPAIRS = {
    "_replace_jan_pos_items_when_balanced",
}
JAN_POS_ROW_PROJECTION_PHASE_HELPER = "_run_jan_pos_row_projection_phase"
JAN_POS_ROW_PROJECTION_PHASE_CALL_LIMIT = 3
RETIRED_STRUCTURAL_ITEM_PROJECTION_PHASE_HELPER = (
    "_run_structural_item_projection_phase"
)
BARCODE_ROW_PROJECTION_REPAIRS = {
    "_replace_barcode_qty_price_rows_when_balanced",
    "_replace_barcode_unit_qty_amount_stack_when_balanced",
}
BARCODE_ROW_PROJECTION_PHASE_HELPER = "_run_barcode_row_projection_phase"
BARCODE_ROW_PROJECTION_PHASE_CALL_LIMIT = 3
DENSE_ITEM_ROW_PROJECTION_REPAIRS = {
    "_replace_dense_item_rows_when_balanced",
}
DENSE_ITEM_ROW_PROJECTION_PHASE_HELPER = "_run_dense_item_row_projection_phase"
DENSE_ITEM_ROW_PROJECTION_PHASE_CALL_LIMIT = 2
DENSE_SEQUENCE_ROW_PROJECTION_REPAIRS = {
    "_replace_dense_sequence_rows_when_balanced",
}
DENSE_SEQUENCE_ROW_PROJECTION_PHASE_HELPER = "_run_dense_sequence_row_projection_phase"
DENSE_SEQUENCE_ROW_PROJECTION_PHASE_CALL_LIMIT = 5
CAMPAIGN_DISCOUNT_PROJECTION_REPAIRS = {
    "_replace_campaign_discount_stream_when_balanced",
}
CAMPAIGN_DISCOUNT_PROJECTION_PHASE_HELPER = (
    "_run_campaign_discount_projection_phase"
)
CAMPAIGN_DISCOUNT_PROJECTION_PHASE_CALL_LIMIT = 1
FINAL_CAMPAIGN_DISCOUNT_PROJECTION_STAGE_LIMIT = 2
FINAL_STRUCTURAL_ITEM_PROJECTION_REPAIRS = {
    "_replace_barcode_unit_qty_amount_stack_when_balanced",
}
FINAL_STRUCTURAL_ITEM_PROJECTION_HELPER = (
    "_run_final_structural_item_projection_phase"
)
FINAL_STRUCTURAL_ITEM_PROJECTION_STAGE_LIMIT = 1
FINAL_JAN_POS_ITEM_PROJECTION_REPAIRS = {
    "_replace_jan_pos_items_when_balanced",
}
FINAL_JAN_POS_ITEM_PROJECTION_HELPER = "_run_final_jan_pos_item_projection_phase"
FINAL_JAN_POS_ITEM_PROJECTION_STAGE_LIMIT = 1
FINAL_BARCODE_QTY_PRICE_PROJECTION_REPAIRS = {
    "_replace_barcode_qty_price_rows_when_balanced",
}
FINAL_BARCODE_QTY_PRICE_PROJECTION_HELPER = (
    "_run_final_barcode_qty_price_projection_phase"
)
FINAL_BARCODE_QTY_PRICE_PROJECTION_STAGE_LIMIT = 1
FINAL_ITEM_PRICE_QTY_PROJECTION_REPAIRS = {
    "_replace_item_price_qty_rows_when_balanced",
}
FINAL_ITEM_PRICE_QTY_PROJECTION_HELPER = (
    "_run_final_item_price_qty_projection_phase"
)
FINAL_ITEM_PRICE_QTY_PROJECTION_STAGE_LIMIT = 1
FINAL_SPLIT_PRICE_BLOCK_PROJECTION_REPAIRS = {
    "_replace_split_price_block_when_balanced",
}
FINAL_SPLIT_PRICE_BLOCK_PROJECTION_HELPER = (
    "_run_final_split_price_block_projection_phase"
)
FINAL_SPLIT_PRICE_BLOCK_PROJECTION_STAGE_LIMIT = 1
SPLIT_PRICE_BLOCK_PROJECTION_REPAIRS = {
    "_replace_split_price_block_when_balanced",
}
SPLIT_PRICE_BLOCK_PROJECTION_PHASE_HELPER = (
    "_run_split_price_block_projection_phase"
)
SPLIT_PRICE_BLOCK_PROJECTION_PHASE_CALL_LIMIT = 1
FINAL_BODY_TOTAL_LAYOUT_RECONSTRUCTION_REPAIRS = {
    "_fix_split_item_price_body_total_layout",
}
FINAL_BODY_TOTAL_LAYOUT_RECONSTRUCTION_HELPER = (
    "_run_final_body_total_layout_reconstruction_phase"
)
FINAL_BODY_TOTAL_LAYOUT_RECONSTRUCTION_STAGE_LIMIT = 1
FINAL_STACKED_NAME_PRICE_PROJECTION_REPAIRS = {
    "_replace_stacked_name_price_rows_when_balanced",
}
FINAL_STACKED_NAME_PRICE_PROJECTION_HELPER = (
    "_run_final_stacked_name_price_projection_phase"
)
FINAL_STACKED_NAME_PRICE_PROJECTION_STAGE_LIMIT = 1
FINAL_DENSE_SEQUENCE_PROJECTION_REPAIRS = {
    "_replace_dense_sequence_rows_when_balanced",
}
FINAL_DENSE_SEQUENCE_PROJECTION_HELPER = (
    "_run_final_dense_sequence_projection_phase"
)
FINAL_DENSE_SEQUENCE_PROJECTION_STAGE_LIMIT = 1
FINAL_HEADER_LOCATION_REPAIR_HELPERS = {
    "_recover_labeled_purchase_site_location",
    "_trim_store_in_store_header_location",
    "_recover_header_branch_store_location",
    "_recover_phone_area_city_location",
    "_recover_short_branch_over_phone_area_city",
    "_normalize_noisy_city_location",
}
FINAL_HEADER_LOCATION_REPAIR_HELPER = "_run_final_header_location_repair_phase"
FINAL_HEADER_LOCATION_REPAIR_STAGE_LIMIT = 6
HEADER_LOCATION_REPAIR_REPAIRS = {
    "_fix_header_store_line_location",
    "_fix_split_address_location_from_ocr",
    "_recover_labeled_purchase_site_location",
}
HEADER_LOCATION_REPAIR_PHASE_HELPER = "_run_header_location_repair_phase"
HEADER_LOCATION_REPAIR_PHASE_CALL_LIMIT = 1
BAG_ITEM_OCR_REPAIR_REPAIRS = {
    "_fix_small_non_bag_item_prices_from_ocr",
    "_fix_bag_item_prices_from_ocr",
    "_fix_split_bag_price_from_nearby_single_digit",
    "_fix_small_bag_description_from_ocr_entry",
}
BAG_ITEM_OCR_REPAIR_PHASE_HELPER = "_run_bag_item_ocr_repair_phase"
BAG_ITEM_OCR_REPAIR_PHASE_CALL_LIMIT = 1
FINAL_SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_REPAIRS = {
    "_restore_single_rate_inclusive_tax_block",
}
FINAL_SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_HELPER = (
    "_run_final_single_rate_inclusive_tax_restoration_phase"
)
FINAL_SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_STAGE_LIMIT = 1
FINAL_STACKED_INCLUSIVE_TAX_RESTORATION_REPAIRS = {
    "_restore_stacked_inclusive_tax_block",
}
FINAL_STACKED_INCLUSIVE_TAX_RESTORATION_HELPER = (
    "_run_final_stacked_inclusive_tax_restoration_phase"
)
FINAL_STACKED_INCLUSIVE_TAX_RESTORATION_STAGE_LIMIT = 1
FINAL_PRINTED_SUMMARY_TOTAL_REPAIR_HELPERS = {
    "_restore_printed_summary_total_when_tax_balanced",
}
FINAL_PRINTED_SUMMARY_TOTAL_REPAIR_HELPER = (
    "_run_final_printed_summary_total_tax_repair_phase"
)
FINAL_PRINTED_SUMMARY_TOTAL_REPAIR_STAGE_LIMIT = 2
PRINTED_SUMMARY_TOTAL_REPAIR_REPAIRS = {
    "_restore_printed_summary_total_when_tax_balanced",
}
PRINTED_SUMMARY_TOTAL_REPAIR_PHASE_HELPER = (
    "_run_printed_summary_total_tax_repair_phase"
)
PRINTED_SUMMARY_TOTAL_REPAIR_PHASE_CALL_LIMIT = 1
FINAL_PRINTED_ITEM_SUM_TOTAL_REPAIR_HELPERS = {
    "_prefer_printed_item_sum_total_when_balanced",
}
FINAL_PRINTED_ITEM_SUM_TOTAL_REPAIR_HELPER = (
    "_run_final_printed_item_sum_total_repair_phase"
)
FINAL_PRINTED_ITEM_SUM_TOTAL_REPAIR_STAGE_LIMIT = 1
PRINTED_ITEM_SUM_TOTAL_REPAIR_REPAIRS = {
    "_prefer_printed_item_sum_total_when_balanced",
}
PRINTED_ITEM_SUM_TOTAL_REPAIR_PHASE_HELPER = (
    "_run_printed_item_sum_total_repair_phase"
)
PRINTED_ITEM_SUM_TOTAL_REPAIR_PHASE_CALL_LIMIT = 1
FINAL_CASH_TENDER_RECONCILIATION_HELPERS = {
    "_fix_unlabeled_cash_tender_change_block",
}
FINAL_CASH_TENDER_RECONCILIATION_HELPER = (
    "_run_final_cash_tender_reconciliation_phase"
)
FINAL_CASH_TENDER_RECONCILIATION_STAGE_LIMIT = 1
FINAL_PAYMENT_POINTS_RECONCILIATION_HELPERS = {
    "reconcile_points_payment_from_ocr",
}
FINAL_PAYMENT_POINTS_RECONCILIATION_HELPER = (
    "_run_final_payment_points_reconciliation_phase"
)
FINAL_PAYMENT_POINTS_RECONCILIATION_STAGE_LIMIT = 1
FINAL_TAX_CATEGORY_RECONCILIATION_HELPERS = {
    "reconcile_tax_categories_from_rate_bases",
}
FINAL_TAX_CATEGORY_RECONCILIATION_HELPER = (
    "_run_final_tax_category_reconciliation_phase"
)
FINAL_TAX_CATEGORY_RECONCILIATION_STAGE_LIMIT = 1
FINAL_EXTERNAL_TAX_TOTAL_RESTORATION_HELPERS = {
    "_restore_external_tax_total_from_printed_subtotal",
}
FINAL_EXTERNAL_TAX_TOTAL_RESTORATION_HELPER = (
    "_run_final_external_tax_total_restoration_phase"
)
FINAL_EXTERNAL_TAX_TOTAL_RESTORATION_STAGE_LIMIT = 2
FINAL_PRINTED_EXTERNAL_TAX_AMOUNT_RESTORATION_REPAIRS = {
    "_restore_printed_external_tax_amounts",
}
FINAL_PRINTED_EXTERNAL_TAX_AMOUNT_RESTORATION_HELPER = (
    "_run_final_printed_external_tax_amount_restoration_phase"
)
FINAL_PRINTED_EXTERNAL_TAX_AMOUNT_RESTORATION_STAGE_LIMIT = 1
PRINTED_EXTERNAL_TAX_AMOUNT_RESTORATION_REPAIRS = {
    "_restore_printed_external_tax_amounts",
}
PRINTED_EXTERNAL_TAX_AMOUNT_RESTORATION_PHASE_HELPER = (
    "_run_printed_external_tax_amount_restoration_phase"
)
PRINTED_EXTERNAL_TAX_AMOUNT_RESTORATION_PHASE_CALL_LIMIT = 1
FINAL_BARE_NUMBER_TAX_SUMMARY_RESTORATION_REPAIRS = {
    "_restore_bare_number_tax_summary",
}
FINAL_BARE_NUMBER_TAX_SUMMARY_RESTORATION_HELPER = (
    "_run_final_bare_number_tax_summary_restoration_phase"
)
FINAL_BARE_NUMBER_TAX_SUMMARY_RESTORATION_STAGE_LIMIT = 1
BARE_NUMBER_TAX_SUMMARY_RESTORATION_REPAIRS = {
    "_restore_bare_number_tax_summary",
}
BARE_NUMBER_TAX_SUMMARY_RESTORATION_PHASE_HELPER = (
    "_run_bare_number_tax_summary_restoration_phase"
)
BARE_NUMBER_TAX_SUMMARY_RESTORATION_PHASE_CALL_LIMIT = 1
FINAL_SMALL_TARGET_ONLY_TAX_PRUNING_REPAIRS = {
    "_drop_unprinted_small_target_only_taxes",
}
FINAL_SMALL_TARGET_ONLY_TAX_PRUNING_HELPER = (
    "_run_final_small_target_only_tax_pruning_phase"
)
FINAL_SMALL_TARGET_ONLY_TAX_PRUNING_STAGE_LIMIT = 1
FINAL_COUPON_DISCOUNT_PROJECTION_REPAIRS = {
    "_fix_item_totals_from_following_discount_lines",
    "_apply_coupon_discount_blocks",
    "_drop_applied_coupon_line_items",
    "_repair_discounted_line_item_totals_when_balanced",
}
FINAL_COUPON_DISCOUNT_PROJECTION_HELPER = (
    "_run_final_coupon_discount_projection_phase"
)
FINAL_COUPON_DISCOUNT_PROJECTION_STAGE_LIMIT = 2
COUPON_DISCOUNT_PROJECTION_REPAIRS = {
    "_apply_coupon_discount_blocks",
    "_drop_applied_coupon_line_items",
}
COUPON_DISCOUNT_PROJECTION_PHASE_HELPER = (
    "_run_coupon_discount_projection_phase"
)
COUPON_DISCOUNT_PROJECTION_PHASE_CALL_LIMIT = 2
FINAL_FOLLOWING_OCR_PRICE_PROJECTION_REPAIRS = {
    "_repair_tiny_item_prices_from_following_ocr",
}
FINAL_FOLLOWING_OCR_PRICE_PROJECTION_HELPER = (
    "_run_final_following_ocr_price_projection_phase"
)
FINAL_FOLLOWING_OCR_PRICE_PROJECTION_STAGE_LIMIT = 1
FOLLOWING_OCR_PRICE_PROJECTION_REPAIRS = {
    "_repair_tiny_item_prices_from_following_ocr",
}
FOLLOWING_OCR_PRICE_PROJECTION_PHASE_HELPER = (
    "_run_following_ocr_price_projection_phase"
)
FOLLOWING_OCR_PRICE_PROJECTION_PHASE_CALL_LIMIT = 2
MERCHANT_IDENTITY_REPAIR_REPAIRS = {
    "_fix_company_name_merchant",
}
MERCHANT_IDENTITY_REPAIR_PHASE_HELPER = "_run_merchant_identity_repair_phase"
MERCHANT_IDENTITY_REPAIR_PHASE_CALL_LIMIT = 1
TRANSACTION_DATETIME_REPAIR_REPAIRS = {
    "_fix_date",
    "_fix_time",
}
TRANSACTION_DATETIME_REPAIR_PHASE_HELPER = "_run_transaction_datetime_repair_phase"
TRANSACTION_DATETIME_REPAIR_PHASE_CALL_LIMIT = 1
OUTPUT_MERCHANT_IDENTITY_REPAIR_PHASE_HELPER = (
    "_run_receipt_output_merchant_identity_phase"
)
OUTPUT_MERCHANT_IDENTITY_REPAIR_PHASE_CALL_LIMIT = 1
RETIRED_FINAL_MERCHANT_IDENTITY_REPAIR_HELPER = (
    "_run_final_merchant_identity_repair_phase"
)
RETIRED_FINAL_MERCHANT_IDENTITY_REPAIR_STAGE = "company_name_merchant"
FINAL_EMBEDDED_PRICE_DUPLICATE_CLEANUP_REPAIRS = {
    "_drop_duplicate_with_embedded_price",
}
RETIRED_FINAL_EMBEDDED_PRICE_DUPLICATE_CLEANUP_HELPER = (
    "_run_final_embedded_price_duplicate_cleanup_phase"
)
RETIRED_FINAL_EMBEDDED_PRICE_DUPLICATE_CLEANUP_STAGE = (
    "drop_duplicate_embedded_price"
)
FINAL_DUPLICATE_ROW_CLEANUP_REPAIRS = {
    "_drop_duplicate_rows_when_subtotal_balances",
}
RETIRED_FINAL_DUPLICATE_ROW_CLEANUP_HELPER = (
    "_run_final_duplicate_row_cleanup_phase"
)
RETIRED_FINAL_DUPLICATE_ROW_CLEANUP_STAGE = (
    "drop_duplicate_rows_when_subtotal_balances"
)
FINAL_DISCOUNT_CONSISTENCY_RECONCILIATION_REPAIRS = {
    "_clear_discount_when_negative_line_precedes_own_price",
}
FINAL_DISCOUNT_CONSISTENCY_RECONCILIATION_HELPER = (
    "_run_final_discount_consistency_reconciliation_phase"
)
FINAL_DISCOUNT_CONSISTENCY_RECONCILIATION_STAGE_LIMIT = 1
DISCOUNT_CONSISTENCY_RECONCILIATION_REPAIRS = {
    "_fix_discounted_item_gross_prices_from_ocr",
    "_fix_item_totals_from_following_discount_lines",
}
DISCOUNT_CONSISTENCY_RECONCILIATION_PHASE_HELPER = (
    "_run_discount_consistency_reconciliation_phase"
)
DISCOUNT_CONSISTENCY_RECONCILIATION_PHASE_CALL_LIMIT = 2
FINAL_QUANTITY_DETAIL_RECONCILIATION_REPAIRS = {
    "_fix_qty_totals_from_ocr_unit_lines",
}
FINAL_QUANTITY_DETAIL_RECONCILIATION_HELPER = (
    "_run_final_quantity_detail_reconciliation_phase"
)
FINAL_QUANTITY_DETAIL_RECONCILIATION_STAGE_LIMIT = 1
FINAL_OCR_DESCRIPTION_RECONCILIATION_REPAIRS = {
    "_fix_code_table_descriptions_by_order",
    "_fix_o_ring_descriptions_from_ocr",
    "_repair_discounted_ocr_pair_descriptions",
    "_repair_pre_price_stack_descriptions_from_ocr",
}
FINAL_OCR_DESCRIPTION_RECONCILIATION_HELPER = (
    "_run_final_ocr_description_reconciliation_phase"
)
FINAL_OCR_DESCRIPTION_RECONCILIATION_STAGE_LIMIT = 3
DISCOUNTED_OCR_ITEM_REPAIR_REPAIRS = {
    "_repair_discounted_line_item_totals_when_balanced",
    "_repair_discounted_ocr_pair_descriptions",
    "_repair_pre_price_stack_descriptions_from_ocr",
}
DISCOUNTED_OCR_ITEM_REPAIR_PHASE_HELPER = (
    "_run_discounted_ocr_item_repair_phase"
)
DISCOUNTED_OCR_ITEM_REPAIR_PHASE_CALL_LIMIT = 1
FINAL_ADJACENT_PRICE_SHIFT_RECONCILIATION_REPAIRS = {
    "_fix_adjacent_ocr_price_shift_when_balanced",
}
FINAL_ADJACENT_PRICE_SHIFT_RECONCILIATION_HELPER = (
    "_run_final_adjacent_price_shift_reconciliation_phase"
)
FINAL_ADJACENT_PRICE_SHIFT_RECONCILIATION_STAGE_LIMIT = 2
FINAL_PREFIXED_TAX_MARKER_ITEM_ROWS_REPAIRS = {
    "_replace_prefixed_tax_marker_item_rows_when_balanced",
}
FINAL_PREFIXED_TAX_MARKER_ITEM_ROWS_HELPER = (
    "_run_final_prefixed_tax_marker_item_rows_phase"
)
FINAL_PREFIXED_TAX_MARKER_ITEM_ROWS_STAGE_LIMIT = 1
PREFIXED_TAX_MARKER_ITEM_ROWS_REPAIRS = {
    "_replace_prefixed_tax_marker_item_rows_when_balanced",
}
PREFIXED_TAX_MARKER_ITEM_ROWS_PHASE_HELPER = (
    "_run_prefixed_tax_marker_item_rows_phase"
)
PREFIXED_TAX_MARKER_ITEM_ROWS_PHASE_CALL_LIMIT = 2
FINAL_GAP_ITEM_RECOVERY_REPAIRS = {
    "_recover_missing_items_from_gap",
}
FINAL_GAP_ITEM_RECOVERY_HELPER = "_run_final_gap_item_recovery_phase"
FINAL_GAP_ITEM_RECOVERY_STAGE_LIMIT = 1
RETIRED_FINAL_REPEATED_GAP_ITEM_RECOVERY_REPAIR = "_recover_repeated_item_from_gap"
RETIRED_FINAL_REPEATED_GAP_ITEM_RECOVERY_STAGE = "repeated_item_gap"
FINAL_BASKET_MARKER_ROWS_REPAIRS = {
    "_replace_basket_marker_rows_when_balanced",
}
FINAL_BASKET_MARKER_ROWS_HELPER = "_run_final_basket_marker_rows_phase"
FINAL_BASKET_MARKER_ROWS_STAGE_LIMIT = 1
BASKET_MARKER_ROWS_REPAIRS = {
    "_replace_basket_marker_rows_when_balanced",
}
BASKET_MARKER_ROWS_PHASE_HELPER = "_run_basket_marker_rows_phase"
BASKET_MARKER_ROWS_PHASE_CALL_LIMIT = 1
QUANTITY_DETAIL_RECONCILIATION_REPAIRS = {
    "_fix_qty_context_and_reduced_rate_from_ocr",
    "_fix_qty_totals_from_ocr_unit_lines",
    "_recover_qty_unit_total_item_from_empty_extraction",
    "_repair_previous_item_from_following_qty_detail",
}
QUANTITY_DETAIL_RECONCILIATION_PHASE_HELPER = "_run_quantity_detail_reconciliation_phase"
QUANTITY_DETAIL_RECONCILIATION_PHASE_CALL_LIMIT = 12
TAX_CATEGORY_ASSIGNMENT_REPAIRS = {
    "_apply_single_bag_standard_rate_split",
    "_assign_single_standard_rate_from_small_base",
    "_fix_nonfood_packaging_tax_categories",
    "_fix_tax_categories_from_ocr_markers",
    "_fix_tax_categories_from_price_line_markers",
    "_normalize_taxes",
    "_rebalance_standard_categories_from_reduced_rate_markers",
    "_rebalance_tax_categories_to_rate_bases",
    "assign_tax_categories",
}
TAX_CATEGORY_ASSIGNMENT_PHASE_HELPER = "_run_tax_category_assignment_phase"
TAX_CATEGORY_ASSIGNMENT_PHASE_CALL_LIMIT = 8
BAG_ITEM_RATE_BASE_RECONCILIATION_REPAIRS = {
    "_fix_bag_item_prices_from_rate_bases",
}
BAG_ITEM_RATE_BASE_RECONCILIATION_PHASE_HELPER = (
    "_run_bag_item_rate_base_reconciliation_phase"
)
BAG_ITEM_RATE_BASE_RECONCILIATION_PHASE_CALL_LIMIT = 2
FINAL_BAG_ITEM_RATE_BASE_RECONCILIATION_REPAIRS = {
    "_fix_bag_item_prices_from_rate_bases",
}
FINAL_BAG_ITEM_RATE_BASE_RECONCILIATION_HELPER = (
    "_run_final_bag_item_rate_base_reconciliation_phase"
)
FINAL_BAG_ITEM_RATE_BASE_RECONCILIATION_STAGE_LIMIT = 1
SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_REPAIRS = {
    "_fix_printed_tax_amounts_from_structural_blocks",
    "_restore_single_rate_inclusive_tax_block",
}
SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_PHASE_HELPER = (
    "_run_single_rate_inclusive_tax_restoration_phase"
)
SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_PHASE_CALL_LIMIT = 5
TAX_EXCLUDED_RATE_BLOCK_RESTORATION_REPAIRS = {
    "_restore_tax_excluded_per_rate_blocks",
}
TAX_EXCLUDED_RATE_BLOCK_RESTORATION_PHASE_HELPER = (
    "_run_tax_excluded_rate_block_restoration_phase"
)
TAX_EXCLUDED_RATE_BLOCK_RESTORATION_PHASE_CALL_LIMIT = 3
EXPLICIT_TAX_AMOUNT_RESTORATION_REPAIRS = {
    "_restore_explicit_tax_rate_amount_lines",
}
EXPLICIT_TAX_AMOUNT_RESTORATION_PHASE_HELPER = (
    "_run_explicit_tax_amount_restoration_phase"
)
EXPLICIT_TAX_AMOUNT_RESTORATION_PHASE_CALL_LIMIT = 3
EXTERNAL_TAX_TOTAL_RESTORATION_REPAIRS = {
    "_restore_external_tax_total_from_printed_subtotal",
}
EXTERNAL_TAX_TOTAL_RESTORATION_PHASE_HELPER = (
    "_run_external_tax_total_restoration_phase"
)
EXTERNAL_TAX_TOTAL_RESTORATION_PHASE_CALL_LIMIT = 3
CASH_TENDER_RECONCILIATION_REPAIRS = {
    "_fix_total_from_stacked_cash_tender_block",
    "_fix_unlabeled_cash_tender_change_block",
}
CASH_TENDER_RECONCILIATION_PHASE_HELPER = "_run_cash_tender_reconciliation_phase"
CASH_TENDER_RECONCILIATION_PHASE_CALL_LIMIT = 2
PAYMENT_METHOD_REPAIR_REPAIRS = {
    "_fix_payment_method",
}
PAYMENT_METHOD_REPAIR_PHASE_HELPER = "_run_payment_method_repair_phase"
PAYMENT_METHOD_REPAIR_PHASE_CALL_LIMIT = 1
TOLL_PAYMENT_REFERENCE_REPAIR_REPAIRS = {
    "_fix_toll_payment_reference",
}
TOLL_PAYMENT_REFERENCE_REPAIR_PHASE_HELPER = (
    "_run_toll_payment_reference_repair_phase"
)
TOLL_PAYMENT_REFERENCE_REPAIR_PHASE_CALL_LIMIT = 1
PAYMENT_POINTS_RECONCILIATION_REPAIRS = {
    "extract_points_used",
    "reconcile_points_payment_from_ocr",
}
PAYMENT_POINTS_RECONCILIATION_PHASE_HELPER = "_run_payment_points_reconciliation_phase"
PAYMENT_POINTS_RECONCILIATION_PHASE_CALL_LIMIT = 2
SERVICE_RECEIPT_RECOVERY_REPAIRS = {
    "_fix_bare_service_receipt_without_itemization",
    "_fix_single_service_inclusive_tax",
    "_replace_service_table_items_when_balanced",
}
SERVICE_RECEIPT_RECOVERY_PHASE_HELPER = "_run_service_receipt_recovery_phase"
SERVICE_RECEIPT_RECOVERY_PHASE_CALL_LIMIT = 6
BODY_TOTAL_LAYOUT_RECONSTRUCTION_REPAIRS = {
    "_fix_split_item_price_body_total_layout",
}
BODY_TOTAL_LAYOUT_RECONSTRUCTION_PHASE_HELPER = (
    "_run_body_total_layout_reconstruction_phase"
)
BODY_TOTAL_LAYOUT_RECONSTRUCTION_PHASE_CALL_LIMIT = 3
OCR_DESCRIPTION_RECONCILIATION_REPAIRS = {
    "_fix_bag_description_from_ocr_code_context",
    "_fix_code_table_descriptions_by_order",
    "_fix_colon_split_product_names_from_ocr",
    "_fix_duplicate_descriptions_from_ocr",
    "_fix_o_ring_descriptions_from_ocr",
    "_fix_qty_code_row_descriptions_from_ocr",
}
OCR_DESCRIPTION_RECONCILIATION_PHASE_HELPER = "_run_ocr_description_reconciliation_phase"
OCR_DESCRIPTION_RECONCILIATION_PHASE_CALL_LIMIT = 7
GAP_ITEM_RECOVERY_REPAIRS = {
    "_recover_discounted_item_from_gap",
    "_recover_missing_items_from_gap",
    "_recover_repeated_item_from_gap",
    "_replace_repeated_ocr_item_block_when_balanced",
}
GAP_ITEM_RECOVERY_PHASE_HELPER = "_run_gap_item_recovery_phase"
GAP_ITEM_RECOVERY_PHASE_CALL_LIMIT = 8
LOW_VALUE_BAG_RECOVERY_REPAIRS = {
    "_append_missing_low_value_bag_from_gap",
    "_fix_numeric_desc_from_ocr_price_context",
    "_recover_missing_bag_items_from_ocr",
    "_replace_overage_item_with_low_value_bag",
}
LOW_VALUE_BAG_RECOVERY_PHASE_HELPER = "_run_low_value_bag_recovery_phase"
LOW_VALUE_BAG_RECOVERY_PHASE_CALL_LIMIT = 5
ADJACENT_PRICE_SHIFT_REPAIRS = {
    "_fix_adjacent_ocr_price_shift_when_balanced",
}
ADJACENT_PRICE_SHIFT_PHASE_HELPER = "_run_adjacent_price_shift_reconciliation_phase"
ADJACENT_PRICE_SHIFT_PHASE_CALL_LIMIT = 4
BAG_AMOUNT_SHIFT_REPAIRS = {
    "_fix_name_bag_amount_shift_from_ocr",
}
BAG_AMOUNT_SHIFT_PHASE_HELPER = "_run_bag_amount_shift_reconciliation_phase"
BAG_AMOUNT_SHIFT_PHASE_CALL_LIMIT = 3
LINE_ITEM_CLEANUP_REPAIRS = {
    "_drop_duplicate_with_embedded_price",
    "_drop_non_product_line_items",
    "_drop_numeric_marker_description_rows",
}
LINE_ITEM_CLEANUP_PHASE_HELPER = "_run_line_item_cleanup_phase"
LINE_ITEM_CLEANUP_PHASE_CALL_LIMIT = 14
PHANTOM_TAX_AMOUNT_CLEANUP_REPAIRS = {
    "_drop_phantom_from_tax_amount",
}
PHANTOM_TAX_AMOUNT_CLEANUP_PHASE_HELPER = (
    "_run_phantom_tax_amount_cleanup_phase"
)
PHANTOM_TAX_AMOUNT_CLEANUP_PHASE_CALL_LIMIT = 1
ITEM_NAME_PRICE_CLEANUP_REPAIRS = {
    "_fix_non_bag_items_named_as_bag",
    "_fix_embedded_price_suffix_totals",
}
ITEM_NAME_PRICE_CLEANUP_PHASE_HELPER = "_run_item_name_price_cleanup_phase"
ITEM_NAME_PRICE_CLEANUP_PHASE_CALL_LIMIT = 1
PRICED_NAME_ITEM_REPAIR_REPAIRS = {
    "_fix_priced_in_name_items",
}
PRICED_NAME_ITEM_REPAIR_PHASE_HELPER = "_run_priced_name_item_repair_phase"
PRICED_NAME_ITEM_REPAIR_PHASE_CALL_LIMIT = 1
DIGIT_MISREAD_ITEM_REPAIR_REPAIRS = {
    "_fix_digit_misread_items",
}
DIGIT_MISREAD_ITEM_REPAIR_PHASE_HELPER = "_run_digit_misread_item_repair_phase"
DIGIT_MISREAD_ITEM_REPAIR_PHASE_CALL_LIMIT = 1
SUBTOTAL_ITEM_PRICE_REPAIR_REPAIRS = {
    "_fix_items_from_subtotal",
}
SUBTOTAL_ITEM_PRICE_REPAIR_PHASE_HELPER = "_run_subtotal_item_price_repair_phase"
SUBTOTAL_ITEM_PRICE_REPAIR_PHASE_CALL_LIMIT = 1
IMPLAUSIBLE_TAX_AMOUNT_REPAIR_REPAIRS = {
    "_fix_implausible_tax_amounts",
}
IMPLAUSIBLE_TAX_AMOUNT_REPAIR_PHASE_HELPER = (
    "_run_implausible_tax_amount_repair_phase"
)
IMPLAUSIBLE_TAX_AMOUNT_REPAIR_PHASE_CALL_LIMIT = 1
VERTICAL_PRICE_QTY_TOTAL_PROJECTION_REPAIRS = {
    "_replace_vertical_price_qty_total_rows_when_balanced",
}
VERTICAL_PRICE_QTY_TOTAL_PROJECTION_PHASE_HELPER = (
    "_run_vertical_price_qty_total_projection_phase"
)
VERTICAL_PRICE_QTY_TOTAL_PROJECTION_PHASE_CALL_LIMIT = 1
CODE_PREFIXED_DESCRIPTION_CLEANUP_REPAIRS = {
    "_clean_code_prefixed_item_descriptions",
}
CODE_PREFIXED_DESCRIPTION_CLEANUP_PHASE_HELPER = (
    "_run_code_prefixed_description_cleanup_phase"
)
CODE_PREFIXED_DESCRIPTION_CLEANUP_PHASE_CALL_LIMIT = 1
DUPLICATE_ROW_CLEANUP_PHASE_HELPER = "_run_duplicate_row_cleanup_phase"
FINAL_OUTPUT_REPAIR_STAGES = (
    "barcode_unit_qty_amount_stack",
    "barcode_qty_price_rows",
    "item_price_qty_rows",
    "labeled_purchase_site_location",
    "store_in_store_header_location",
    "header_branch_store_location",
    "phone_area_city_location",
    "short_branch_over_phone_area_city",
    "noisy_city_location",
    "single_rate_inclusive_tax_block",
    "coupon_discount_projection",
    "tiny_item_prices_from_following_ocr",
    "split_price_block",
    "split_item_price_body_total",
    "stacked_name_price_rows",
    "stacked_inclusive_tax_block",
    "printed_summary_total_tax_balanced",
    "printed_item_sum_total",
    "ocr_description_reconciliation",
    "adjacent_price_shift_reconciliation",
    "dense_sequence_rows",
    "campaign_discount_stream",
    "jan_pos_items",
    "qty_totals_from_unit_lines",
    "bag_item_prices_from_rate_bases",
    "code_table_description_reconciliation",
    "printed_external_tax_amounts",
    "bare_number_tax_summary",
    "external_tax_total_from_printed_subtotal",
    "drop_small_target_only_taxes",
    "printed_summary_total_tax_balanced_2",
    "unlabeled_cash_tender_change",
    "points_payment",
    "clear_discount_before_own_price",
    "campaign_discount_stream_2",
    "coupon_discount_projection_after_layout",
    "adjacent_price_shift_reconciliation_after_layout",
    "prefixed_tax_marker_item_rows",
    "missing_items_from_gap",
    "ocr_description_reconciliation_after_layout",
    "basket_marker_rows",
    "tax_categories_from_rate_bases",
    "external_tax_total_from_printed_subtotal_final",
)
STRUCTURAL_JAPANESE_LITERAL_RE = re.compile(
    r"(小計|合計|内税|外税|非課税|消費税|税|対象|軽減|税込|税抜|"
    r"現金|預|釣|支払|ポイント|領収|レシート|登録番号|電話|TEL|"
    r"店|支店|営業所|料金所|住所|市|区|町|村|県|都|道|府|"
    r"年|月|日|時|分|個|点|円|品番|JAN|バーコード)"
)
JAPANESE_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    function: str
    rule: str
    detail: str

    @property
    def key(self) -> tuple[str, int, str, str]:
        return self.path, self.line, self.rule, self.detail

    @property
    def signature(self) -> tuple[str, str, str]:
        return self.path, self.rule, self.detail

    def format(self) -> str:
        return (
            f"{self.path}:{self.line} in {self.function}: "
            f"{self.rule}: {self.detail}"
        )


# Known violations from pipeline_brittleness_audit.md. Keep the entries tied to
# source locations for review, but compare by signature counts so harmless line
# movement does not make the guardrail stale.
KNOWN_VIOLATIONS = set()


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _enclosing_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current.name
    return "<module>"


def _literal_key(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_repair_call_name(name: str | None) -> bool:
    return bool(name) and name.startswith(REPAIR_CALL_PREFIXES)


def _function_def(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Function not found: {name}")


def _has_function_def(tree: ast.AST, name: str) -> bool:
    return any(
        isinstance(node, ast.FunctionDef) and node.name == name
        for node in ast.walk(tree)
    )


def _call_names_in_function(function: ast.FunctionDef) -> list[str]:
    names = []
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name:
                names.append(name)
    return names


def _parse_file(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _postprocess_repair_calls() -> list[tuple[str, int]]:
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    function = _function_def(tree, "postprocess_receipt")
    calls = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if _is_repair_call_name(name):
            calls.append((name or "", node.lineno))
    return calls


def _final_output_repair_stage_calls() -> list[tuple[str, int]]:
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    function = _function_def(tree, "_apply_final_receipt_output_repairs")
    stages = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or _call_name(node.func) != "run":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            stages.append(("<nonliteral>", node.lineno))
            continue
        stages.append((str(node.args[0].value), node.lineno))
    return sorted(stages, key=lambda item: item[1])


def _final_output_repair_justifications() -> dict[str, tuple[str, str]]:
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "FINAL_RECEIPT_OUTPUT_REPAIR_JUSTIFICATIONS"
                for target in node.targets
            )
        ):
            try:
                value = ast.literal_eval(node.value)
            except (SyntaxError, ValueError) as exc:
                raise AssertionError(
                    "FINAL_RECEIPT_OUTPUT_REPAIR_JUSTIFICATIONS must be literal"
                ) from exc
            return value
    raise AssertionError("FINAL_RECEIPT_OUTPUT_REPAIR_JUSTIFICATIONS not found")


def _postprocess_phase_names() -> set[str]:
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "POSTPROCESS_PHASES"
                for target in node.targets
            )
        ):
            try:
                value = ast.literal_eval(node.value)
            except (SyntaxError, ValueError) as exc:
                raise AssertionError("POSTPROCESS_PHASES must be literal") from exc
            return {
                phase["name"]
                for phase in value
                if isinstance(phase, dict) and isinstance(phase.get("name"), str)
            }
    raise AssertionError("POSTPROCESS_PHASES not found")


def _current_japanese_string_counts() -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for path in SCANNED_FILES:
        tree = _parse_file(path)
        rel = _relative(path)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and JAPANESE_CHAR_RE.search(node.value)
            ):
                counts[(rel, node.value)] += 1
    return counts


def _baseline_japanese_string_counts() -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for path in SCANNED_FILES:
        rel = _relative(path)
        try:
            source = subprocess.check_output(
                ["git", "show", f"{BASELINE_COMMIT}:{rel}"],
                cwd=ROOT,
                text=True,
                encoding="utf-8",
            )
        except subprocess.CalledProcessError as exc:
            raise AssertionError(
                f"Could not read {rel} from baseline {BASELINE_COMMIT}"
            ) from exc
        tree = ast.parse(source, filename=rel)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and JAPANESE_CHAR_RE.search(node.value)
            ):
                counts[(rel, node.value)] += 1
    return counts


def _looks_structural_japanese_literal(value: str) -> bool:
    return bool(STRUCTURAL_JAPANESE_LITERAL_RE.search(value))


def _assigned_semantic_fields(node: ast.AST) -> set[str]:
    fields: set[str] = set()
    targets = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    elif isinstance(node, ast.AugAssign):
        targets = [node.target]

    for target in targets:
        if isinstance(target, ast.Subscript):
            field = _literal_key(target.slice)
            if field in SEMANTIC_FIELDS:
                fields.add(field)
    return fields


def _condition_known_value_gates(node: ast.AST, source: str) -> list[str]:
    if not isinstance(node, ast.Compare):
        return []
    if not any(isinstance(op, (ast.Eq, ast.In)) for op in node.ops):
        return []

    constants = [
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant)
        and isinstance(child.value, (str, int, float))
    ]
    if not constants:
        return []

    suspicious = []
    for value in constants:
        if isinstance(value, str):
            if (
                KNOWN_DATE_RE.search(value)
                or MERCHANT_OR_STORE_RE.search(value)
                or FIXTURE_REFERENCE_RE.search(value)
            ):
                suspicious.append(value)
        elif isinstance(value, (int, float)) and abs(value) >= 1000:
            suspicious.append(value)

    if not suspicious:
        return []
    return [(ast.get_source_segment(source, node) or repr(suspicious)).replace("\n", " ")]


def _scan_ast(path: Path) -> list[Violation]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    parents = _parents(tree)
    rel = _relative(path)
    violations: list[Violation] = []

    for node in ast.walk(tree):
        function = _enclosing_function(node, parents)

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if MERCHANT_OR_STORE_RE.search(node.name):
                violations.append(
                    Violation(rel, node.lineno, function, "merchant_or_store_name", node.name)
                )
            if FIXTURE_REFERENCE_RE.search(node.name):
                violations.append(
                    Violation(rel, node.lineno, function, "fixture_reference_name", node.name)
                )
            if KNOWN_ANSWER_NAME_RE.search(node.name):
                violations.append(
                    Violation(rel, node.lineno, function, "known_answer_helper_name", node.name)
                )

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and _literal_key(target.slice) == "line_items"
                    and isinstance(node.value, ast.List)
                    and len(node.value.elts) >= 2
                ):
                    violations.append(
                        Violation(
                            rel,
                            node.lineno,
                            function,
                            "hardcoded_line_items_assignment",
                            f"line_items[{len(node.value.elts)}]",
                        )
                    )

        for detail in _condition_known_value_gates(node, source):
            violations.append(
                Violation(rel, node.lineno, function, "known_value_gate", detail)
            )

        if function == "_build_result" and isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in FINAL_RESULT_MUTATORS:
                violations.append(
                    Violation(
                        rel,
                        node.lineno,
                        function,
                        "final_result_semantic_mutation",
                        name,
                    )
                )
        if function in FINAL_OUTPUT_FUNCTIONS and isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in FINAL_OUTPUT_KNOWN_ANSWER_MUTATORS:
                violations.append(
                    Violation(
                        rel,
                        node.lineno,
                        function,
                        "final_output_known_answer_mutation",
                        name,
                    )
                )

        if isinstance(node, ast.If):
            condition_text = ast.get_source_segment(source, node.test) or ""
            condition_has_known_gate = (
                MERCHANT_OR_STORE_RE.search(condition_text)
                or FIXTURE_REFERENCE_RE.search(condition_text)
                or KNOWN_DATE_RE.search(condition_text)
            )
            if condition_has_known_gate:
                fields = set()
                for body_node in node.body:
                    for child in ast.walk(body_node):
                        fields.update(_assigned_semantic_fields(child))
                if fields:
                    violations.append(
                        Violation(
                            rel,
                            node.lineno,
                            function,
                            "known_gate_semantic_assignment",
                            ",".join(sorted(fields)),
                        )
                    )

    return violations


def _scan_comments(path: Path) -> list[Violation]:
    violations: list[Violation] = []
    rel = _relative(path)
    with tokenize.open(path) as handle:
        tokens = tokenize.generate_tokens(handle.readline)
        for token in tokens:
            if token.type != tokenize.COMMENT:
                continue
            comment = token.string
            if FIXTURE_REFERENCE_RE.search(comment):
                violations.append(
                    Violation(
                        rel,
                        token.start[0],
                        "<comment>",
                        "fixture_reference_comment",
                        comment.strip(),
                    )
                )
    return violations


def _collect_violations() -> list[Violation]:
    violations: list[Violation] = []
    for path in SCANNED_FILES:
        violations.extend(_scan_ast(path))
        violations.extend(_scan_comments(path))
    return sorted(violations, key=lambda item: item.key)


def test_production_pipeline_has_no_new_brittle_known_answer_overrides():
    violations = _collect_violations()
    known_counts = Counter(
        (path, rule, detail) for path, _line, rule, detail in KNOWN_VIOLATIONS
    )
    violation_counts = Counter(violation.signature for violation in violations)
    unexpected = [
        violation
        for violation in violations
        if violation_counts[violation.signature] > known_counts[violation.signature]
    ]
    seen_unexpected: set[tuple[str, str, str]] = set()
    unexpected = [
        violation
        for violation in unexpected
        if violation.signature not in seen_unexpected
        and not seen_unexpected.add(violation.signature)
    ]
    stale_allowlist = sorted(
        signature
        for signature, count in known_counts.items()
        if violation_counts[signature] < count
    )

    message = io.StringIO()
    if unexpected:
        message.write(
            "Production parser code contains brittle known-answer patterns.\n"
            "Use structural OCR evidence plus arithmetic/format invariants instead.\n"
            "Unexpected violations:\n"
        )
        for violation in unexpected:
            message.write(f"  - {violation.format()}\n")
    if stale_allowlist:
        message.write(
            "\nThe guardrail allowlist contains entries that no longer match the "
            "current source. Remove these known-violation entries:\n"
        )
        for key in stale_allowlist:
            message.write(
                f"  - {key} "
                f"(expected {known_counts[key]}, found {violation_counts[key]})\n"
            )

    assert not unexpected and not stale_allowlist, message.getvalue()


def test_postprocess_receipt_repair_stack_does_not_grow_without_review():
    calls = _postprocess_repair_calls()
    assert len(calls) <= POSTPROCESS_REPAIR_CALL_LIMIT, (
        "postprocess_receipt gained repair/mutator calls. Split the work into "
        "a named phase with structural trigger and invariant, or explicitly "
        "lower existing debt before adding more.\n"
        f"Current count: {len(calls)}; limit: {POSTPROCESS_REPAIR_CALL_LIMIT}"
    )


def test_postprocess_receipt_repeated_mutators_are_explicitly_allowlisted():
    counts = Counter(name for name, _line in _postprocess_repair_calls())
    repeated = {name: count for name, count in counts.items() if count >= 3}
    unexpected = sorted(set(repeated) - set(POSTPROCESS_MUTATOR_REPEAT_ALLOWLIST))
    grown = sorted(
        (name, repeated[name], allowed)
        for name, allowed in POSTPROCESS_MUTATOR_REPEAT_ALLOWLIST.items()
        if repeated.get(name, 0) > allowed
    )

    assert not unexpected and not grown, (
        "Repeated mutator calls in postprocess_receipt must be explained as "
        "temporary debt and must not grow.\n"
        f"Unexpected repeated mutators: {unexpected}\n"
        f"Allowlisted mutators whose call counts grew: {grown}"
    )


def test_jan_pos_row_projection_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, JAN_POS_ROW_PROJECTION_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        JAN_POS_ROW_PROJECTION_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "JAN POS row projection repairs must be owned by the named "
        f"{JAN_POS_ROW_PROJECTION_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{JAN_POS_ROW_PROJECTION_PHASE_HELPER} must document the OCR/layout "
        "trigger and arithmetic or field-consistency invariant it enforces."
    )


def test_postprocess_jan_pos_row_projection_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_projection_calls = [
        name for name in postprocess_calls if name in JAN_POS_ROW_PROJECTION_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == JAN_POS_ROW_PROJECTION_PHASE_HELPER
    ]

    assert not direct_projection_calls, (
        "JAN POS row projection repairs should run through the named phase "
        "helper so JAN/item-code OCR triggers and arithmetic invariants have one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_projection_calls}"
    )
    assert RETIRED_STRUCTURAL_ITEM_PROJECTION_PHASE_HELPER not in postprocess_calls, (
        "The broad structural item projection phase should not remain after "
        "JAN POS row projection has a named postprocess owner."
    )
    assert not _has_function_def(tree, RETIRED_STRUCTURAL_ITEM_PROJECTION_PHASE_HELPER), (
        "The broad structural item projection helper should be retired once "
        "JAN POS row projection has a named postprocess owner."
    )
    assert 0 < len(phase_calls) <= JAN_POS_ROW_PROJECTION_PHASE_CALL_LIMIT, (
        "JAN POS row projection phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {JAN_POS_ROW_PROJECTION_PHASE_CALL_LIMIT}"
    )


def test_barcode_row_projection_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, BARCODE_ROW_PROJECTION_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        BARCODE_ROW_PROJECTION_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Barcode row projection repairs must be owned by the named "
        f"{BARCODE_ROW_PROJECTION_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{BARCODE_ROW_PROJECTION_PHASE_HELPER} must document the OCR/layout "
        "trigger and arithmetic or field-consistency invariant it enforces."
    )


def test_postprocess_barcode_row_projection_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_barcode_calls = [
        name for name in postprocess_calls if name in BARCODE_ROW_PROJECTION_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == BARCODE_ROW_PROJECTION_PHASE_HELPER
    ]

    assert not direct_barcode_calls, (
        "Barcode row projection repairs should run through the named phase "
        "helper so barcode OCR triggers and arithmetic invariants have one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_barcode_calls}"
    )
    assert 0 < len(phase_calls) <= BARCODE_ROW_PROJECTION_PHASE_CALL_LIMIT, (
        "Barcode row projection phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {BARCODE_ROW_PROJECTION_PHASE_CALL_LIMIT}"
    )


def test_dense_item_row_projection_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, DENSE_ITEM_ROW_PROJECTION_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        DENSE_ITEM_ROW_PROJECTION_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Dense item row projection must be owned by the named "
        f"{DENSE_ITEM_ROW_PROJECTION_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{DENSE_ITEM_ROW_PROJECTION_PHASE_HELPER} must document the OCR/layout "
        "trigger and arithmetic or field-consistency invariant it enforces."
    )


def test_postprocess_dense_item_row_projection_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_dense_calls = [
        name for name in postprocess_calls if name in DENSE_ITEM_ROW_PROJECTION_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == DENSE_ITEM_ROW_PROJECTION_PHASE_HELPER
    ]

    assert not direct_dense_calls, (
        "Dense item row projection repairs should run through the named phase "
        "helper so dense OCR row triggers and arithmetic invariants have one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_dense_calls}"
    )
    assert 0 < len(phase_calls) <= DENSE_ITEM_ROW_PROJECTION_PHASE_CALL_LIMIT, (
        "Dense item row projection phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {DENSE_ITEM_ROW_PROJECTION_PHASE_CALL_LIMIT}"
    )


def test_dense_sequence_row_projection_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, DENSE_SEQUENCE_ROW_PROJECTION_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        DENSE_SEQUENCE_ROW_PROJECTION_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Dense sequence row projection must be owned by the named "
        f"{DENSE_SEQUENCE_ROW_PROJECTION_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{DENSE_SEQUENCE_ROW_PROJECTION_PHASE_HELPER} must document the OCR/layout "
        "trigger and arithmetic or field-consistency invariant it enforces."
    )


def test_postprocess_dense_sequence_row_projection_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_sequence_calls = [
        name
        for name in postprocess_calls
        if name in DENSE_SEQUENCE_ROW_PROJECTION_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == DENSE_SEQUENCE_ROW_PROJECTION_PHASE_HELPER
    ]

    assert not direct_sequence_calls, (
        "Dense sequence row projection repairs should run through the named "
        "phase helper so dense OCR sequence triggers and arithmetic invariants "
        "have one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_sequence_calls}"
    )
    assert 0 < len(phase_calls) <= DENSE_SEQUENCE_ROW_PROJECTION_PHASE_CALL_LIMIT, (
        "Dense sequence row projection phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {DENSE_SEQUENCE_ROW_PROJECTION_PHASE_CALL_LIMIT}"
    )


def test_campaign_discount_projection_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, CAMPAIGN_DISCOUNT_PROJECTION_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        CAMPAIGN_DISCOUNT_PROJECTION_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Campaign discount stream projection must be owned by the named "
        f"{CAMPAIGN_DISCOUNT_PROJECTION_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{CAMPAIGN_DISCOUNT_PROJECTION_PHASE_HELPER} must document the "
        "campaign discount OCR trigger and subtotal/discount invariant."
    )


def test_postprocess_campaign_discount_projection_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_projection_calls = [
        name
        for name in postprocess_calls
        if name in CAMPAIGN_DISCOUNT_PROJECTION_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == CAMPAIGN_DISCOUNT_PROJECTION_PHASE_HELPER
    ]

    assert not direct_projection_calls, (
        "Campaign discount projection should run through the named phase "
        "helper so campaign discount OCR triggers and subtotal/discount "
        "invariants have one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_projection_calls}"
    )
    assert 0 < len(phase_calls) <= CAMPAIGN_DISCOUNT_PROJECTION_PHASE_CALL_LIMIT, (
        "Campaign discount projection phase calls must be explicit and "
        "bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {CAMPAIGN_DISCOUNT_PROJECTION_PHASE_CALL_LIMIT}"
    )


def test_final_campaign_discount_projection_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_projection_calls = [
        name
        for name in final_calls
        if name in CAMPAIGN_DISCOUNT_PROJECTION_REPAIRS
    ]
    phase_calls = [
        name
        for name in final_calls
        if name == CAMPAIGN_DISCOUNT_PROJECTION_PHASE_HELPER
    ]

    assert not direct_projection_calls, (
        "Late campaign discount projection should run through the named phase "
        "helper so repeated final stages share the same OCR trigger and "
        "subtotal/discount invariant owner.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_projection_calls}"
    )
    assert 0 < len(phase_calls) <= FINAL_CAMPAIGN_DISCOUNT_PROJECTION_STAGE_LIMIT, (
        "Late campaign discount projection phase calls must be explicit and "
        "bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {FINAL_CAMPAIGN_DISCOUNT_PROJECTION_STAGE_LIMIT}"
    )


def test_final_structural_item_projection_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    helper = _function_def(tree, FINAL_STRUCTURAL_ITEM_PROJECTION_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        FINAL_STRUCTURAL_ITEM_PROJECTION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Late barcode/unit/qty/amount stack projection must be owned by the "
        f"named {FINAL_STRUCTURAL_ITEM_PROJECTION_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{FINAL_STRUCTURAL_ITEM_PROJECTION_HELPER} must document the visible "
        "barcode/JAN stack trigger and item-sum arithmetic invariant."
    )


def test_final_structural_item_projection_debt_is_helper_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_projection_calls = [
        name
        for name in final_calls
        if name in FINAL_STRUCTURAL_ITEM_PROJECTION_REPAIRS
    ]
    helper_calls = [
        name
        for name in final_calls
        if name == FINAL_STRUCTURAL_ITEM_PROJECTION_HELPER
    ]

    assert not direct_projection_calls, (
        "Late barcode/unit/qty/amount stack projection should run through the "
        "named helper so OCR row-stack triggers and item-sum invariants have "
        "one owner.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_projection_calls}"
    )
    assert 0 < len(helper_calls) <= FINAL_STRUCTURAL_ITEM_PROJECTION_STAGE_LIMIT, (
        "Late barcode/unit/qty/amount stack projection helper calls must be "
        "explicit and bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {FINAL_STRUCTURAL_ITEM_PROJECTION_STAGE_LIMIT}"
    )


def test_final_jan_pos_item_projection_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    helper = _function_def(tree, FINAL_JAN_POS_ITEM_PROJECTION_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        FINAL_JAN_POS_ITEM_PROJECTION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Late JAN/POS item projection must be owned by the named "
        f"{FINAL_JAN_POS_ITEM_PROJECTION_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{FINAL_JAN_POS_ITEM_PROJECTION_HELPER} must document the "
        "JAN/POS row trigger and subtotal/rate-base arithmetic invariant."
    )


def test_final_jan_pos_item_projection_debt_is_helper_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_jan_calls = [
        name
        for name in final_calls
        if name in FINAL_JAN_POS_ITEM_PROJECTION_REPAIRS
    ]
    helper_calls = [
        name
        for name in final_calls
        if name == FINAL_JAN_POS_ITEM_PROJECTION_HELPER
    ]

    assert not direct_jan_calls, (
        "Late JAN/POS item projection should run through the named helper so "
        "barcode/JAN row evidence and subtotal arithmetic have one owner.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_jan_calls}"
    )
    assert (
        0
        < len(helper_calls)
        <= FINAL_JAN_POS_ITEM_PROJECTION_STAGE_LIMIT
    ), (
        "Late JAN/POS item projection helper calls must be explicit and "
        "bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {FINAL_JAN_POS_ITEM_PROJECTION_STAGE_LIMIT}"
    )


def test_final_barcode_qty_price_projection_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    helper = _function_def(tree, FINAL_BARCODE_QTY_PRICE_PROJECTION_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        FINAL_BARCODE_QTY_PRICE_PROJECTION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Late barcode quantity/price row projection must be owned by the "
        f"named {FINAL_BARCODE_QTY_PRICE_PROJECTION_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{FINAL_BARCODE_QTY_PRICE_PROJECTION_HELPER} must document the "
        "barcode/JAN quantity-price row trigger and arithmetic invariant."
    )


def test_final_barcode_qty_price_projection_debt_is_helper_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_projection_calls = [
        name
        for name in final_calls
        if name in FINAL_BARCODE_QTY_PRICE_PROJECTION_REPAIRS
    ]
    helper_calls = [
        name
        for name in final_calls
        if name == FINAL_BARCODE_QTY_PRICE_PROJECTION_HELPER
    ]

    assert not direct_projection_calls, (
        "Late barcode quantity/price row projection should run through the "
        "named helper so OCR row-stack triggers and item-sum invariants have "
        "one owner.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_projection_calls}"
    )
    assert (
        0 < len(helper_calls) <= FINAL_BARCODE_QTY_PRICE_PROJECTION_STAGE_LIMIT
    ), (
        "Late barcode quantity/price projection helper calls must be explicit "
        "and bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {FINAL_BARCODE_QTY_PRICE_PROJECTION_STAGE_LIMIT}"
    )


def test_final_item_price_qty_projection_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    helper = _function_def(tree, FINAL_ITEM_PRICE_QTY_PROJECTION_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        FINAL_ITEM_PRICE_QTY_PROJECTION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Late item price/quantity row projection must be owned by the named "
        f"{FINAL_ITEM_PRICE_QTY_PROJECTION_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{FINAL_ITEM_PRICE_QTY_PROJECTION_HELPER} must document the "
        "description-price-quantity OCR trigger and subtotal/count invariant."
    )


def test_final_item_price_qty_projection_debt_is_helper_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_projection_calls = [
        name
        for name in final_calls
        if name in FINAL_ITEM_PRICE_QTY_PROJECTION_REPAIRS
    ]
    helper_calls = [
        name
        for name in final_calls
        if name == FINAL_ITEM_PRICE_QTY_PROJECTION_HELPER
    ]

    assert not direct_projection_calls, (
        "Late item price/quantity row projection should run through the "
        "named helper so OCR layout triggers and subtotal/count invariants "
        "have one owner.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_projection_calls}"
    )
    assert 0 < len(helper_calls) <= FINAL_ITEM_PRICE_QTY_PROJECTION_STAGE_LIMIT, (
        "Late item price/quantity projection helper calls must be explicit "
        "and bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {FINAL_ITEM_PRICE_QTY_PROJECTION_STAGE_LIMIT}"
    )


def test_final_split_price_block_projection_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    helper = _function_def(tree, FINAL_SPLIT_PRICE_BLOCK_PROJECTION_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        FINAL_SPLIT_PRICE_BLOCK_PROJECTION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Late split price block projection must be owned by the named "
        f"{FINAL_SPLIT_PRICE_BLOCK_PROJECTION_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{FINAL_SPLIT_PRICE_BLOCK_PROJECTION_HELPER} must document the "
        "split description/price OCR trigger and subtotal invariant."
    )


def test_final_split_price_block_projection_debt_is_helper_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_projection_calls = [
        name
        for name in final_calls
        if name in FINAL_SPLIT_PRICE_BLOCK_PROJECTION_REPAIRS
    ]
    helper_calls = [
        name
        for name in final_calls
        if name == FINAL_SPLIT_PRICE_BLOCK_PROJECTION_HELPER
    ]

    assert not direct_projection_calls, (
        "Late split price block projection should run through the named "
        "helper so OCR layout triggers and subtotal invariants have one "
        "owner.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_projection_calls}"
    )
    assert 0 < len(helper_calls) <= FINAL_SPLIT_PRICE_BLOCK_PROJECTION_STAGE_LIMIT, (
        "Late split price block projection helper calls must be explicit and "
        "bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {FINAL_SPLIT_PRICE_BLOCK_PROJECTION_STAGE_LIMIT}"
    )


def test_split_price_block_projection_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, SPLIT_PRICE_BLOCK_PROJECTION_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        SPLIT_PRICE_BLOCK_PROJECTION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Postprocess split-price block projection must be owned by the named "
        f"{SPLIT_PRICE_BLOCK_PROJECTION_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{SPLIT_PRICE_BLOCK_PROJECTION_PHASE_HELPER} must document split OCR "
        "name/price block evidence and subtotal/total consistency invariant."
    )


def test_postprocess_split_price_block_projection_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_projection_calls = [
        name
        for name in postprocess_calls
        if name in SPLIT_PRICE_BLOCK_PROJECTION_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == SPLIT_PRICE_BLOCK_PROJECTION_PHASE_HELPER
    ]

    assert not direct_projection_calls, (
        "Postprocess split-price block projection should run through a named "
        "phase helper so split OCR name/price evidence and arithmetic "
        "consistency have one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_projection_calls}"
    )
    assert 0 < len(phase_calls) <= SPLIT_PRICE_BLOCK_PROJECTION_PHASE_CALL_LIMIT, (
        "Postprocess split-price block phase calls must be explicit and "
        "bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {SPLIT_PRICE_BLOCK_PROJECTION_PHASE_CALL_LIMIT}"
    )


def test_final_body_total_layout_reconstruction_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    helper = _function_def(tree, FINAL_BODY_TOTAL_LAYOUT_RECONSTRUCTION_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        FINAL_BODY_TOTAL_LAYOUT_RECONSTRUCTION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Late body-total layout reconstruction must be owned by the named "
        f"{FINAL_BODY_TOTAL_LAYOUT_RECONSTRUCTION_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{FINAL_BODY_TOTAL_LAYOUT_RECONSTRUCTION_HELPER} must document "
        "the printed body-total layout trigger and subtotal/tax invariant."
    )


def test_final_body_total_layout_reconstruction_debt_is_helper_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_reconstruction_calls = [
        name
        for name in final_calls
        if name in FINAL_BODY_TOTAL_LAYOUT_RECONSTRUCTION_REPAIRS
    ]
    helper_calls = [
        name
        for name in final_calls
        if name == FINAL_BODY_TOTAL_LAYOUT_RECONSTRUCTION_HELPER
    ]

    assert not direct_reconstruction_calls, (
        "Late body-total layout reconstruction should run through the named "
        "helper so printed body-total layout triggers and subtotal/tax "
        "invariants have one owner.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_reconstruction_calls}"
    )
    assert 0 < len(helper_calls) <= FINAL_BODY_TOTAL_LAYOUT_RECONSTRUCTION_STAGE_LIMIT, (
        "Late body-total layout reconstruction helper calls must be explicit "
        "and bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {FINAL_BODY_TOTAL_LAYOUT_RECONSTRUCTION_STAGE_LIMIT}"
    )


def test_final_stacked_name_price_projection_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    helper = _function_def(tree, FINAL_STACKED_NAME_PRICE_PROJECTION_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        FINAL_STACKED_NAME_PRICE_PROJECTION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Late stacked name/price projection must be owned by the named "
        f"{FINAL_STACKED_NAME_PRICE_PROJECTION_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{FINAL_STACKED_NAME_PRICE_PROJECTION_HELPER} must document the "
        "stacked description/price OCR trigger and subtotal/rate-base "
        "invariant."
    )


def test_final_stacked_name_price_projection_debt_is_helper_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_projection_calls = [
        name
        for name in final_calls
        if name in FINAL_STACKED_NAME_PRICE_PROJECTION_REPAIRS
    ]
    helper_calls = [
        name
        for name in final_calls
        if name == FINAL_STACKED_NAME_PRICE_PROJECTION_HELPER
    ]

    assert not direct_projection_calls, (
        "Late stacked name/price projection should run through the named "
        "helper so stacked OCR row triggers and subtotal/rate-base invariants "
        "have one owner.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_projection_calls}"
    )
    assert 0 < len(helper_calls) <= FINAL_STACKED_NAME_PRICE_PROJECTION_STAGE_LIMIT, (
        "Late stacked name/price projection helper calls must be explicit and "
        "bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {FINAL_STACKED_NAME_PRICE_PROJECTION_STAGE_LIMIT}"
    )


def test_final_dense_sequence_projection_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    helper = _function_def(tree, FINAL_DENSE_SEQUENCE_PROJECTION_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        FINAL_DENSE_SEQUENCE_PROJECTION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Late dense sequence projection must be owned by the named "
        f"{FINAL_DENSE_SEQUENCE_PROJECTION_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{FINAL_DENSE_SEQUENCE_PROJECTION_HELPER} must document the dense "
        "OCR item/price sequence trigger and subtotal/count invariant."
    )


def test_final_dense_sequence_projection_debt_is_helper_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_projection_calls = [
        name
        for name in final_calls
        if name in FINAL_DENSE_SEQUENCE_PROJECTION_REPAIRS
    ]
    helper_calls = [
        name
        for name in final_calls
        if name == FINAL_DENSE_SEQUENCE_PROJECTION_HELPER
    ]

    assert not direct_projection_calls, (
        "Late dense sequence projection should run through the named helper "
        "so dense OCR row triggers and subtotal/count invariants have one "
        "owner.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_projection_calls}"
    )
    assert 0 < len(helper_calls) <= FINAL_DENSE_SEQUENCE_PROJECTION_STAGE_LIMIT, (
        "Late dense sequence projection helper calls must be explicit and "
        "bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {FINAL_DENSE_SEQUENCE_PROJECTION_STAGE_LIMIT}"
    )


def test_final_header_location_repair_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    helper = _function_def(tree, FINAL_HEADER_LOCATION_REPAIR_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        FINAL_HEADER_LOCATION_REPAIR_HELPERS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Late header/location repairs must be owned by the named "
        f"{FINAL_HEADER_LOCATION_REPAIR_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{FINAL_HEADER_LOCATION_REPAIR_HELPER} must document OCR header, "
        "address, or phone-area triggers and a location field-consistency "
        "invariant."
    )


def test_final_header_location_repair_debt_is_helper_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_location_calls = [
        name
        for name in final_calls
        if name in FINAL_HEADER_LOCATION_REPAIR_HELPERS
    ]
    helper_calls = [
        name
        for name in final_calls
        if name == FINAL_HEADER_LOCATION_REPAIR_HELPER
    ]

    assert not direct_location_calls, (
        "Late header/location repairs should run through the named helper so "
        "OCR header/address/phone triggers and location consistency have one "
        "owner.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_location_calls}"
    )
    assert 0 < len(helper_calls) <= FINAL_HEADER_LOCATION_REPAIR_STAGE_LIMIT, (
        "Late header/location repair helper calls must be explicit and bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {FINAL_HEADER_LOCATION_REPAIR_STAGE_LIMIT}"
    )


def test_final_single_rate_inclusive_tax_restoration_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    helper = _function_def(tree, FINAL_SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        FINAL_SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Late single-rate inclusive tax restoration must be owned by the named "
        f"{FINAL_SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{FINAL_SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_HELPER} must document "
        "the printed single-rate inclusive tax trigger and total/tax "
        "arithmetic invariant."
    )


def test_final_single_rate_inclusive_tax_restoration_debt_is_helper_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_tax_calls = [
        name
        for name in final_calls
        if name in FINAL_SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_REPAIRS
    ]
    helper_calls = [
        name
        for name in final_calls
        if name == FINAL_SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_HELPER
    ]

    assert not direct_tax_calls, (
        "Late single-rate inclusive tax restoration should run through the "
        "named helper so printed target/tax triggers and total/tax invariants "
        "have one owner.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_tax_calls}"
    )
    assert (
        0
        < len(helper_calls)
        <= FINAL_SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_STAGE_LIMIT
    ), (
        "Late single-rate inclusive tax restoration helper calls must be "
        "explicit and bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {FINAL_SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_STAGE_LIMIT}"
    )


def test_final_stacked_inclusive_tax_restoration_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    helper = _function_def(tree, FINAL_STACKED_INCLUSIVE_TAX_RESTORATION_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        FINAL_STACKED_INCLUSIVE_TAX_RESTORATION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Late stacked inclusive tax restoration must be owned by the named "
        f"{FINAL_STACKED_INCLUSIVE_TAX_RESTORATION_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{FINAL_STACKED_INCLUSIVE_TAX_RESTORATION_HELPER} must document the "
        "stacked printed inclusive tax trigger and tax summary arithmetic "
        "invariant."
    )


def test_final_stacked_inclusive_tax_restoration_debt_is_helper_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_tax_calls = [
        name
        for name in final_calls
        if name in FINAL_STACKED_INCLUSIVE_TAX_RESTORATION_REPAIRS
    ]
    helper_calls = [
        name
        for name in final_calls
        if name == FINAL_STACKED_INCLUSIVE_TAX_RESTORATION_HELPER
    ]

    assert not direct_tax_calls, (
        "Late stacked inclusive tax restoration should run through the named "
        "helper so stacked summary triggers and tax arithmetic invariants have "
        "one owner.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_tax_calls}"
    )
    assert (
        0 < len(helper_calls) <= FINAL_STACKED_INCLUSIVE_TAX_RESTORATION_STAGE_LIMIT
    ), (
        "Late stacked inclusive tax restoration helper calls must be explicit "
        "and bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {FINAL_STACKED_INCLUSIVE_TAX_RESTORATION_STAGE_LIMIT}"
    )


def test_final_printed_summary_total_repair_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    helper = _function_def(tree, FINAL_PRINTED_SUMMARY_TOTAL_REPAIR_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        FINAL_PRINTED_SUMMARY_TOTAL_REPAIR_HELPERS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Late printed summary total repair must be owned by the named "
        f"{FINAL_PRINTED_SUMMARY_TOTAL_REPAIR_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{FINAL_PRINTED_SUMMARY_TOTAL_REPAIR_HELPER} must document the "
        "printed summary total/tax-balance trigger and total/tax/payment "
        "arithmetic invariant."
    )


def test_final_printed_summary_total_repair_debt_is_helper_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_total_calls = [
        name
        for name in final_calls
        if name in FINAL_PRINTED_SUMMARY_TOTAL_REPAIR_HELPERS
    ]
    helper_calls = [
        name
        for name in final_calls
        if name == FINAL_PRINTED_SUMMARY_TOTAL_REPAIR_HELPER
    ]

    assert not direct_total_calls, (
        "Late printed summary total repair should run through the named "
        "helper so printed total/tax triggers and total/tax/payment "
        "invariants have one owner.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_total_calls}"
    )
    assert (
        0 < len(helper_calls) <= FINAL_PRINTED_SUMMARY_TOTAL_REPAIR_STAGE_LIMIT
    ), (
        "Late printed summary total repair helper calls must be explicit "
        "and bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {FINAL_PRINTED_SUMMARY_TOTAL_REPAIR_STAGE_LIMIT}"
    )


def test_postprocess_printed_summary_total_repair_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, PRINTED_SUMMARY_TOTAL_REPAIR_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        PRINTED_SUMMARY_TOTAL_REPAIR_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Postprocess printed summary total repair must be owned by the named "
        f"{PRINTED_SUMMARY_TOTAL_REPAIR_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{PRINTED_SUMMARY_TOTAL_REPAIR_PHASE_HELPER} must document the "
        "printed summary total/tax-balance trigger and total/tax/payment "
        "arithmetic invariant."
    )


def test_postprocess_printed_summary_total_repair_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_total_calls = [
        name
        for name in postprocess_calls
        if name in PRINTED_SUMMARY_TOTAL_REPAIR_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == PRINTED_SUMMARY_TOTAL_REPAIR_PHASE_HELPER
    ]

    assert not direct_total_calls, (
        "Postprocess printed summary total repair should run through the named "
        "phase helper so printed total/tax triggers and total/tax/payment "
        "invariants have one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_total_calls}"
    )
    assert 0 < len(phase_calls) <= PRINTED_SUMMARY_TOTAL_REPAIR_PHASE_CALL_LIMIT, (
        "Printed summary total repair phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {PRINTED_SUMMARY_TOTAL_REPAIR_PHASE_CALL_LIMIT}"
    )


def test_final_printed_item_sum_total_repair_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    helper = _function_def(tree, FINAL_PRINTED_ITEM_SUM_TOTAL_REPAIR_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        FINAL_PRINTED_ITEM_SUM_TOTAL_REPAIR_HELPERS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Late printed item-sum total repair must be owned by the named "
        f"{FINAL_PRINTED_ITEM_SUM_TOTAL_REPAIR_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{FINAL_PRINTED_ITEM_SUM_TOTAL_REPAIR_HELPER} must document the "
        "printed item-sum total trigger and item/tax/payment arithmetic "
        "invariant."
    )


def test_final_printed_item_sum_total_repair_debt_is_helper_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_total_calls = [
        name
        for name in final_calls
        if name in FINAL_PRINTED_ITEM_SUM_TOTAL_REPAIR_HELPERS
    ]
    helper_calls = [
        name
        for name in final_calls
        if name == FINAL_PRINTED_ITEM_SUM_TOTAL_REPAIR_HELPER
    ]

    assert not direct_total_calls, (
        "Late printed item-sum total repair should run through the named "
        "helper so printed total triggers and item/tax/payment arithmetic "
        "invariants have one owner.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_total_calls}"
    )
    assert (
        0 < len(helper_calls) <= FINAL_PRINTED_ITEM_SUM_TOTAL_REPAIR_STAGE_LIMIT
    ), (
        "Late printed item-sum total repair helper calls must be explicit "
        "and bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {FINAL_PRINTED_ITEM_SUM_TOTAL_REPAIR_STAGE_LIMIT}"
    )


def test_postprocess_printed_item_sum_total_repair_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, PRINTED_ITEM_SUM_TOTAL_REPAIR_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        PRINTED_ITEM_SUM_TOTAL_REPAIR_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Postprocess printed item-sum total repair must be owned by the named "
        f"{PRINTED_ITEM_SUM_TOTAL_REPAIR_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{PRINTED_ITEM_SUM_TOTAL_REPAIR_PHASE_HELPER} must document the "
        "printed item-sum/summary total trigger and item, tax, and payment "
        "arithmetic invariant."
    )


def test_postprocess_printed_item_sum_total_repair_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_total_calls = [
        name
        for name in postprocess_calls
        if name in PRINTED_ITEM_SUM_TOTAL_REPAIR_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == PRINTED_ITEM_SUM_TOTAL_REPAIR_PHASE_HELPER
    ]

    assert not direct_total_calls, (
        "Postprocess printed item-sum total repair should run through the named "
        "phase helper so printed item/summary total triggers and item/tax/payment "
        "invariants have one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_total_calls}"
    )
    assert 0 < len(phase_calls) <= PRINTED_ITEM_SUM_TOTAL_REPAIR_PHASE_CALL_LIMIT, (
        "Printed item-sum total repair phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {PRINTED_ITEM_SUM_TOTAL_REPAIR_PHASE_CALL_LIMIT}"
    )


def test_final_cash_tender_reconciliation_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    helper = _function_def(tree, FINAL_CASH_TENDER_RECONCILIATION_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        FINAL_CASH_TENDER_RECONCILIATION_HELPERS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Late cash tender/change repair must be owned by the named "
        f"{FINAL_CASH_TENDER_RECONCILIATION_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{FINAL_CASH_TENDER_RECONCILIATION_HELPER} must document the "
        "visible cash tender/change trigger and printed total/tender/change "
        "arithmetic invariant."
    )


def test_final_cash_tender_reconciliation_debt_is_helper_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_cash_calls = [
        name
        for name in final_calls
        if name in FINAL_CASH_TENDER_RECONCILIATION_HELPERS
    ]
    helper_calls = [
        name
        for name in final_calls
        if name == FINAL_CASH_TENDER_RECONCILIATION_HELPER
    ]

    assert not direct_cash_calls, (
        "Late cash tender/change repair should run through the named helper "
        "so printed total, tendered amount, and change arithmetic have one "
        "owner.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_cash_calls}"
    )
    assert 0 < len(helper_calls) <= FINAL_CASH_TENDER_RECONCILIATION_STAGE_LIMIT, (
        "Late cash tender/change helper calls must be explicit and bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {FINAL_CASH_TENDER_RECONCILIATION_STAGE_LIMIT}"
    )


def test_final_payment_points_reconciliation_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    helper = _function_def(tree, FINAL_PAYMENT_POINTS_RECONCILIATION_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        FINAL_PAYMENT_POINTS_RECONCILIATION_HELPERS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Late points/payment repair must be owned by the named "
        f"{FINAL_PAYMENT_POINTS_RECONCILIATION_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{FINAL_PAYMENT_POINTS_RECONCILIATION_HELPER} must document the OCR "
        "points/payment trigger and total minus points payment invariant."
    )


def test_final_payment_points_reconciliation_debt_is_helper_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_payment_points_calls = [
        name
        for name in final_calls
        if name in FINAL_PAYMENT_POINTS_RECONCILIATION_HELPERS
    ]
    helper_calls = [
        name
        for name in final_calls
        if name == FINAL_PAYMENT_POINTS_RECONCILIATION_HELPER
    ]

    assert not direct_payment_points_calls, (
        "Late points/payment repair should run through the named helper so "
        "OCR point-use evidence and total minus points payment arithmetic "
        "have one owner.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_payment_points_calls}"
    )
    assert (
        0 < len(helper_calls) <= FINAL_PAYMENT_POINTS_RECONCILIATION_STAGE_LIMIT
    ), (
        "Late points/payment helper calls must be explicit and bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {FINAL_PAYMENT_POINTS_RECONCILIATION_STAGE_LIMIT}"
    )


def test_final_tax_category_reconciliation_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    helper = _function_def(tree, FINAL_TAX_CATEGORY_RECONCILIATION_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        FINAL_TAX_CATEGORY_RECONCILIATION_HELPERS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Late tax category repair must be owned by the named "
        f"{FINAL_TAX_CATEGORY_RECONCILIATION_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{FINAL_TAX_CATEGORY_RECONCILIATION_HELPER} must document the "
        "printed rate-base trigger and per-item tax category/rate-base "
        "consistency invariant."
    )


def test_final_tax_category_reconciliation_debt_is_helper_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_tax_calls = [
        name
        for name in final_calls
        if name in FINAL_TAX_CATEGORY_RECONCILIATION_HELPERS
    ]
    helper_calls = [
        name
        for name in final_calls
        if name == FINAL_TAX_CATEGORY_RECONCILIATION_HELPER
    ]

    assert not direct_tax_calls, (
        "Late tax category repair should run through the named helper so "
        "printed rate-base evidence and per-item category assignment have "
        "one owner.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_tax_calls}"
    )
    assert (
        0 < len(helper_calls) <= FINAL_TAX_CATEGORY_RECONCILIATION_STAGE_LIMIT
    ), (
        "Late tax category helper calls must be explicit and bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {FINAL_TAX_CATEGORY_RECONCILIATION_STAGE_LIMIT}"
    )


def test_final_external_tax_total_restoration_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    helper = _function_def(tree, FINAL_EXTERNAL_TAX_TOTAL_RESTORATION_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        FINAL_EXTERNAL_TAX_TOTAL_RESTORATION_HELPERS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Late external tax total repair must be owned by the named "
        f"{FINAL_EXTERNAL_TAX_TOTAL_RESTORATION_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{FINAL_EXTERNAL_TAX_TOTAL_RESTORATION_HELPER} must document the "
        "printed subtotal/tax trigger and subtotal plus external tax total "
        "arithmetic invariant."
    )


def test_final_external_tax_total_restoration_debt_is_helper_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_tax_total_calls = [
        name
        for name in final_calls
        if name in FINAL_EXTERNAL_TAX_TOTAL_RESTORATION_HELPERS
    ]
    helper_calls = [
        name
        for name in final_calls
        if name == FINAL_EXTERNAL_TAX_TOTAL_RESTORATION_HELPER
    ]

    assert not direct_tax_total_calls, (
        "Late external tax total repair should run through the named helper "
        "so printed subtotal, external tax, and total arithmetic have one "
        "owner.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_tax_total_calls}"
    )
    assert (
        0 < len(helper_calls) <= FINAL_EXTERNAL_TAX_TOTAL_RESTORATION_STAGE_LIMIT
    ), (
        "Late external tax total helper calls must be explicit and bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {FINAL_EXTERNAL_TAX_TOTAL_RESTORATION_STAGE_LIMIT}"
    )


def test_final_printed_external_tax_amount_restoration_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    helper = _function_def(
        tree,
        FINAL_PRINTED_EXTERNAL_TAX_AMOUNT_RESTORATION_HELPER,
    )
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        FINAL_PRINTED_EXTERNAL_TAX_AMOUNT_RESTORATION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Late printed external-tax amount restoration must be owned by the "
        f"named {FINAL_PRINTED_EXTERNAL_TAX_AMOUNT_RESTORATION_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{FINAL_PRINTED_EXTERNAL_TAX_AMOUNT_RESTORATION_HELPER} must document "
        "the printed external-tax amount trigger and tax/base/total consistency "
        "invariant."
    )


def test_final_printed_external_tax_amount_restoration_debt_is_helper_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_tax_calls = [
        name
        for name in final_calls
        if name in FINAL_PRINTED_EXTERNAL_TAX_AMOUNT_RESTORATION_REPAIRS
    ]
    helper_calls = [
        name
        for name in final_calls
        if name == FINAL_PRINTED_EXTERNAL_TAX_AMOUNT_RESTORATION_HELPER
    ]

    assert not direct_tax_calls, (
        "Late printed external-tax amount restoration should run through the "
        "named helper so OCR tax amount evidence and tax/base/total consistency "
        "have one owner.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_tax_calls}"
    )
    assert (
        0
        < len(helper_calls)
        <= FINAL_PRINTED_EXTERNAL_TAX_AMOUNT_RESTORATION_STAGE_LIMIT
    ), (
        "Late printed external-tax amount helper calls must be explicit and "
        "bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {FINAL_PRINTED_EXTERNAL_TAX_AMOUNT_RESTORATION_STAGE_LIMIT}"
    )


def test_printed_external_tax_amount_restoration_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(
        tree,
        PRINTED_EXTERNAL_TAX_AMOUNT_RESTORATION_PHASE_HELPER,
    )
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        PRINTED_EXTERNAL_TAX_AMOUNT_RESTORATION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Postprocess printed external-tax amount restoration must be owned by "
        f"the named {PRINTED_EXTERNAL_TAX_AMOUNT_RESTORATION_PHASE_HELPER} "
        "helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{PRINTED_EXTERNAL_TAX_AMOUNT_RESTORATION_PHASE_HELPER} must document "
        "the printed external-tax amount trigger and tax/base/total consistency "
        "invariant."
    )


def test_postprocess_printed_external_tax_amount_restoration_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_tax_calls = [
        name
        for name in postprocess_calls
        if name in PRINTED_EXTERNAL_TAX_AMOUNT_RESTORATION_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == PRINTED_EXTERNAL_TAX_AMOUNT_RESTORATION_PHASE_HELPER
    ]

    assert not direct_tax_calls, (
        "Postprocess printed external-tax amount restoration should run "
        "through the named phase helper so printed tax amount evidence and "
        "tax/base/total consistency have one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_tax_calls}"
    )
    assert (
        0
        < len(phase_calls)
        <= PRINTED_EXTERNAL_TAX_AMOUNT_RESTORATION_PHASE_CALL_LIMIT
    ), (
        "Postprocess printed external-tax amount phase calls must be explicit "
        "and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {PRINTED_EXTERNAL_TAX_AMOUNT_RESTORATION_PHASE_CALL_LIMIT}"
    )


def test_final_bare_number_tax_summary_restoration_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    helper = _function_def(
        tree,
        FINAL_BARE_NUMBER_TAX_SUMMARY_RESTORATION_HELPER,
    )
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        FINAL_BARE_NUMBER_TAX_SUMMARY_RESTORATION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Late bare-number tax-summary restoration must be owned by the named "
        f"{FINAL_BARE_NUMBER_TAX_SUMMARY_RESTORATION_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{FINAL_BARE_NUMBER_TAX_SUMMARY_RESTORATION_HELPER} must document "
        "the bare numeric tax-summary stack trigger and rate/tax arithmetic "
        "invariant."
    )


def test_final_bare_number_tax_summary_restoration_debt_is_helper_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_tax_calls = [
        name
        for name in final_calls
        if name in FINAL_BARE_NUMBER_TAX_SUMMARY_RESTORATION_REPAIRS
    ]
    helper_calls = [
        name
        for name in final_calls
        if name == FINAL_BARE_NUMBER_TAX_SUMMARY_RESTORATION_HELPER
    ]

    assert not direct_tax_calls, (
        "Late bare-number tax-summary restoration should run through the "
        "named helper so numeric tax stack evidence and rate/tax arithmetic "
        "have one owner.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_tax_calls}"
    )
    assert (
        0
        < len(helper_calls)
        <= FINAL_BARE_NUMBER_TAX_SUMMARY_RESTORATION_STAGE_LIMIT
    ), (
        "Late bare-number tax-summary helper calls must be explicit and "
        "bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {FINAL_BARE_NUMBER_TAX_SUMMARY_RESTORATION_STAGE_LIMIT}"
    )


def test_bare_number_tax_summary_restoration_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(
        tree,
        BARE_NUMBER_TAX_SUMMARY_RESTORATION_PHASE_HELPER,
    )
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        BARE_NUMBER_TAX_SUMMARY_RESTORATION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Postprocess bare-number tax summary restoration must be owned by the "
        f"named {BARE_NUMBER_TAX_SUMMARY_RESTORATION_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{BARE_NUMBER_TAX_SUMMARY_RESTORATION_PHASE_HELPER} must document the "
        "bare-number rate/tax trigger and tax/subtotal/total consistency "
        "invariant."
    )


def test_postprocess_bare_number_tax_summary_restoration_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_tax_calls = [
        name
        for name in postprocess_calls
        if name in BARE_NUMBER_TAX_SUMMARY_RESTORATION_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == BARE_NUMBER_TAX_SUMMARY_RESTORATION_PHASE_HELPER
    ]

    assert not direct_tax_calls, (
        "Postprocess bare-number tax summary restoration should run through "
        "the named phase helper so bare numeric tax rows and "
        "tax/subtotal/total consistency have one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_tax_calls}"
    )
    assert (
        0
        < len(phase_calls)
        <= BARE_NUMBER_TAX_SUMMARY_RESTORATION_PHASE_CALL_LIMIT
    ), (
        "Postprocess bare-number tax summary phase calls must be explicit and "
        "bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {BARE_NUMBER_TAX_SUMMARY_RESTORATION_PHASE_CALL_LIMIT}"
    )


def test_final_small_target_only_tax_pruning_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    helper = _function_def(
        tree,
        FINAL_SMALL_TARGET_ONLY_TAX_PRUNING_HELPER,
    )
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        FINAL_SMALL_TARGET_ONLY_TAX_PRUNING_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Late small target-only tax pruning must be owned by the named "
        f"{FINAL_SMALL_TARGET_ONLY_TAX_PRUNING_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{FINAL_SMALL_TARGET_ONLY_TAX_PRUNING_HELPER} must document the "
        "unprinted target-only tax trigger and printed tax/subtotal arithmetic "
        "invariant."
    )


def test_final_small_target_only_tax_pruning_debt_is_helper_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_tax_calls = [
        name
        for name in final_calls
        if name in FINAL_SMALL_TARGET_ONLY_TAX_PRUNING_REPAIRS
    ]
    helper_calls = [
        name
        for name in final_calls
        if name == FINAL_SMALL_TARGET_ONLY_TAX_PRUNING_HELPER
    ]

    assert not direct_tax_calls, (
        "Late small target-only tax pruning should run through the named "
        "helper so printed rate-base evidence and tax/subtotal consistency "
        "have one owner.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_tax_calls}"
    )
    assert (
        0
        < len(helper_calls)
        <= FINAL_SMALL_TARGET_ONLY_TAX_PRUNING_STAGE_LIMIT
    ), (
        "Late small target-only tax pruning helper calls must be explicit and "
        "bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {FINAL_SMALL_TARGET_ONLY_TAX_PRUNING_STAGE_LIMIT}"
    )


def test_final_coupon_discount_projection_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    helper = _function_def(tree, FINAL_COUPON_DISCOUNT_PROJECTION_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        FINAL_COUPON_DISCOUNT_PROJECTION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Late coupon/discount projection must be owned by the named "
        f"{FINAL_COUPON_DISCOUNT_PROJECTION_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{FINAL_COUPON_DISCOUNT_PROJECTION_HELPER} must document the "
        "OCR discount/coupon trigger and item gross minus discount or "
        "subtotal-balance invariant."
    )


def test_final_coupon_discount_projection_debt_is_helper_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_discount_calls = [
        name
        for name in final_calls
        if name in FINAL_COUPON_DISCOUNT_PROJECTION_REPAIRS
    ]
    helper_calls = [
        name
        for name in final_calls
        if name == FINAL_COUPON_DISCOUNT_PROJECTION_HELPER
    ]

    assert not direct_discount_calls, (
        "Late coupon/discount projection should run through the named helper "
        "so OCR discount markers and item/subtotal arithmetic have one owner.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_discount_calls}"
    )
    assert 0 < len(helper_calls) <= FINAL_COUPON_DISCOUNT_PROJECTION_STAGE_LIMIT, (
        "Late coupon/discount helper calls must be explicit and bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {FINAL_COUPON_DISCOUNT_PROJECTION_STAGE_LIMIT}"
    )


def test_coupon_discount_projection_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, COUPON_DISCOUNT_PROJECTION_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        COUPON_DISCOUNT_PROJECTION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Postprocess coupon discount projection must be owned by the named "
        f"{COUPON_DISCOUNT_PROJECTION_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{COUPON_DISCOUNT_PROJECTION_PHASE_HELPER} must document the OCR "
        "coupon/CPN trigger and item gross-minus-discount or subtotal "
        "consistency invariant."
    )


def test_postprocess_coupon_discount_projection_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_coupon_calls = [
        name
        for name in postprocess_calls
        if name in COUPON_DISCOUNT_PROJECTION_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == COUPON_DISCOUNT_PROJECTION_PHASE_HELPER
    ]

    assert not direct_coupon_calls, (
        "Postprocess coupon discount projection should run through a named "
        "phase helper so OCR coupon markers and item/subtotal arithmetic "
        "have one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_coupon_calls}"
    )
    assert 0 < len(phase_calls) <= COUPON_DISCOUNT_PROJECTION_PHASE_CALL_LIMIT, (
        "Postprocess coupon discount phase calls must be explicit and "
        "bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {COUPON_DISCOUNT_PROJECTION_PHASE_CALL_LIMIT}"
    )


def test_final_following_ocr_price_projection_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    helper = _function_def(tree, FINAL_FOLLOWING_OCR_PRICE_PROJECTION_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        FINAL_FOLLOWING_OCR_PRICE_PROJECTION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Late following-OCR price projection must be owned by the named "
        f"{FINAL_FOLLOWING_OCR_PRICE_PROJECTION_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{FINAL_FOLLOWING_OCR_PRICE_PROJECTION_HELPER} must document the "
        "following OCR amount trigger and item-sum/rate-base invariant."
    )


def test_final_following_ocr_price_projection_debt_is_helper_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_price_calls = [
        name
        for name in final_calls
        if name in FINAL_FOLLOWING_OCR_PRICE_PROJECTION_REPAIRS
    ]
    helper_calls = [
        name
        for name in final_calls
        if name == FINAL_FOLLOWING_OCR_PRICE_PROJECTION_HELPER
    ]

    assert not direct_price_calls, (
        "Late following-OCR price projection should run through the named "
        "helper so repeated OCR amount evidence and subtotal/rate-base "
        "arithmetic have one owner.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_price_calls}"
    )
    assert (
        0
        < len(helper_calls)
        <= FINAL_FOLLOWING_OCR_PRICE_PROJECTION_STAGE_LIMIT
    ), (
        "Late following-OCR price projection helper calls must be explicit and "
        "bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {FINAL_FOLLOWING_OCR_PRICE_PROJECTION_STAGE_LIMIT}"
    )


def test_following_ocr_price_projection_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, FOLLOWING_OCR_PRICE_PROJECTION_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        FOLLOWING_OCR_PRICE_PROJECTION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Postprocess following-OCR price projection must be owned by the "
        f"named {FOLLOWING_OCR_PRICE_PROJECTION_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{FOLLOWING_OCR_PRICE_PROJECTION_PHASE_HELPER} must document the "
        "repeated following OCR amount trigger and item-sum/rate-base "
        "invariant."
    )


def test_postprocess_following_ocr_price_projection_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_price_calls = [
        name
        for name in postprocess_calls
        if name in FOLLOWING_OCR_PRICE_PROJECTION_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == FOLLOWING_OCR_PRICE_PROJECTION_PHASE_HELPER
    ]

    assert not direct_price_calls, (
        "Postprocess following-OCR price projection should run through a "
        "named phase helper so repeated OCR amount evidence and item-sum "
        "arithmetic have one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_price_calls}"
    )
    assert 0 < len(phase_calls) <= FOLLOWING_OCR_PRICE_PROJECTION_PHASE_CALL_LIMIT, (
        "Postprocess following-OCR price projection phase calls must be "
        "explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {FOLLOWING_OCR_PRICE_PROJECTION_PHASE_CALL_LIMIT}"
    )


def test_merchant_identity_repair_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, MERCHANT_IDENTITY_REPAIR_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        MERCHANT_IDENTITY_REPAIR_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Merchant identity repair must be owned by the named "
        f"{MERCHANT_IDENTITY_REPAIR_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{MERCHANT_IDENTITY_REPAIR_PHASE_HELPER} must document the "
        "header/legal-name trigger and merchant field-consistency invariant."
    )


def test_postprocess_merchant_identity_repair_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_merchant_calls = [
        name
        for name in postprocess_calls
        if name in MERCHANT_IDENTITY_REPAIR_REPAIRS
    ]
    helper_calls = [
        name for name in postprocess_calls if name == MERCHANT_IDENTITY_REPAIR_PHASE_HELPER
    ]

    assert not direct_merchant_calls, (
        "Merchant identity repair should run through the named postprocess helper "
        "so header/company-name evidence and merchant field consistency have "
        "one owner.\n"
        "Direct calls still in postprocess_receipt: "
        f"{direct_merchant_calls}"
    )
    assert 0 < len(helper_calls) <= MERCHANT_IDENTITY_REPAIR_PHASE_CALL_LIMIT, (
        "Merchant identity phase calls must be explicit and bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {MERCHANT_IDENTITY_REPAIR_PHASE_CALL_LIMIT}"
    )


def test_transaction_datetime_repair_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, TRANSACTION_DATETIME_REPAIR_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        TRANSACTION_DATETIME_REPAIR_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Transaction date/time repairs must be owned by the named "
        f"{TRANSACTION_DATETIME_REPAIR_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{TRANSACTION_DATETIME_REPAIR_PHASE_HELPER} must document the "
        "OCR date/time trigger and date/time field-consistency invariant."
    )


def test_postprocess_transaction_datetime_repair_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_datetime_calls = [
        name
        for name in postprocess_calls
        if name in TRANSACTION_DATETIME_REPAIR_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == TRANSACTION_DATETIME_REPAIR_PHASE_HELPER
    ]

    assert not direct_datetime_calls, (
        "Transaction date/time repairs should run through the named phase "
        "helper so OCR transaction-date anchors and time consistency have one "
        "owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_datetime_calls}"
    )
    assert 0 < len(phase_calls) <= TRANSACTION_DATETIME_REPAIR_PHASE_CALL_LIMIT, (
        "Transaction datetime phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {TRANSACTION_DATETIME_REPAIR_PHASE_CALL_LIMIT}"
    )


def test_toll_payment_reference_repair_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, TOLL_PAYMENT_REFERENCE_REPAIR_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        TOLL_PAYMENT_REFERENCE_REPAIR_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Toll payment-reference repair must be owned by the named "
        f"{TOLL_PAYMENT_REFERENCE_REPAIR_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{TOLL_PAYMENT_REFERENCE_REPAIR_PHASE_HELPER} must document the "
        "toll OCR trigger and payment_reference preservation invariant."
    )


def test_postprocess_toll_payment_reference_repair_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_reference_calls = [
        name
        for name in postprocess_calls
        if name in TOLL_PAYMENT_REFERENCE_REPAIR_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == TOLL_PAYMENT_REFERENCE_REPAIR_PHASE_HELPER
    ]

    assert not direct_reference_calls, (
        "Toll payment-reference repair should run through the named phase "
        "helper so toll OCR evidence and reference preservation have one "
        "owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_reference_calls}"
    )
    assert (
        0 < len(phase_calls) <= TOLL_PAYMENT_REFERENCE_REPAIR_PHASE_CALL_LIMIT
    ), (
        "Toll payment-reference phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {TOLL_PAYMENT_REFERENCE_REPAIR_PHASE_CALL_LIMIT}"
    )


def test_header_location_repair_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, HEADER_LOCATION_REPAIR_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        HEADER_LOCATION_REPAIR_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Header/location repairs must be owned by the named "
        f"{HEADER_LOCATION_REPAIR_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{HEADER_LOCATION_REPAIR_PHASE_HELPER} must document OCR header, "
        "split-address, or purchase-site triggers and a location "
        "field-consistency invariant."
    )


def test_postprocess_header_location_repair_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_location_calls = [
        name
        for name in postprocess_calls
        if name in HEADER_LOCATION_REPAIR_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == HEADER_LOCATION_REPAIR_PHASE_HELPER
    ]

    assert not direct_location_calls, (
        "Header/location repairs should run through the named phase helper so "
        "OCR header, address, and purchase-site location evidence have one "
        "owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_location_calls}"
    )
    assert 0 < len(phase_calls) <= HEADER_LOCATION_REPAIR_PHASE_CALL_LIMIT, (
        "Header/location repair phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {HEADER_LOCATION_REPAIR_PHASE_CALL_LIMIT}"
    )


def test_bag_item_ocr_repair_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, BAG_ITEM_OCR_REPAIR_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        BAG_ITEM_OCR_REPAIR_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Bag/small-item OCR repairs must be owned by the named "
        f"{BAG_ITEM_OCR_REPAIR_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{BAG_ITEM_OCR_REPAIR_PHASE_HELPER} must document visible item/bag "
        "OCR price triggers and an item total consistency invariant."
    )


def test_postprocess_bag_item_ocr_repair_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_item_calls = [
        name
        for name in postprocess_calls
        if name in BAG_ITEM_OCR_REPAIR_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == BAG_ITEM_OCR_REPAIR_PHASE_HELPER
    ]

    assert not direct_item_calls, (
        "Bag/small-item OCR repairs should run through the named phase helper "
        "so item-price OCR evidence and total consistency have one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_item_calls}"
    )
    assert 0 < len(phase_calls) <= BAG_ITEM_OCR_REPAIR_PHASE_CALL_LIMIT, (
        "Bag/small-item OCR repair phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {BAG_ITEM_OCR_REPAIR_PHASE_CALL_LIMIT}"
    )


def test_final_merchant_identity_repair_debt_is_removed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_merchant_calls = [
        name
        for name in final_calls
        if name in MERCHANT_IDENTITY_REPAIR_REPAIRS
    ]

    assert not direct_merchant_calls, (
        "Merchant identity repair should not run directly in late final output "
        "repairs after postprocess owns the behavior.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_merchant_calls}"
    )
    assert RETIRED_FINAL_MERCHANT_IDENTITY_REPAIR_HELPER not in final_calls, (
        "The late merchant identity repair helper should not be called from "
        "_apply_final_receipt_output_repairs."
    )
    assert (
        RETIRED_FINAL_MERCHANT_IDENTITY_REPAIR_STAGE not in FINAL_OUTPUT_REPAIR_STAGES
    ), (
        "The merchant identity final-output stage should be removed from the "
        "tracked late repair list."
    )


def test_output_merchant_identity_repair_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    helper = _function_def(tree, OUTPUT_MERCHANT_IDENTITY_REPAIR_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        MERCHANT_IDENTITY_REPAIR_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Receipt output merchant identity repair must be owned by the named "
        f"{OUTPUT_MERCHANT_IDENTITY_REPAIR_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{OUTPUT_MERCHANT_IDENTITY_REPAIR_PHASE_HELPER} must document the "
        "header/legal-name trigger and merchant field-consistency invariant."
    )


def test_prepare_receipt_output_merchant_identity_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    prepare_output = _function_def(tree, "_prepare_receipt_output_payload")
    prepare_calls = _call_names_in_function(prepare_output)
    direct_merchant_calls = [
        name
        for name in prepare_calls
        if name in MERCHANT_IDENTITY_REPAIR_REPAIRS
    ]
    helper_calls = [
        name
        for name in prepare_calls
        if name == OUTPUT_MERCHANT_IDENTITY_REPAIR_PHASE_HELPER
    ]

    assert not direct_merchant_calls, (
        "Receipt output merchant identity repair should run through the named "
        "phase so post-serialization merchant consistency has one owner.\n"
        "Direct calls still in _prepare_receipt_output_payload: "
        f"{direct_merchant_calls}"
    )
    assert 0 < len(helper_calls) <= OUTPUT_MERCHANT_IDENTITY_REPAIR_PHASE_CALL_LIMIT, (
        "Receipt output merchant identity phase calls must be explicit and bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {OUTPUT_MERCHANT_IDENTITY_REPAIR_PHASE_CALL_LIMIT}"
    )


def test_final_embedded_price_duplicate_cleanup_is_retired_to_postprocess():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    receipt_tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    item_cleanup_helper = _function_def(receipt_tree, LINE_ITEM_CLEANUP_PHASE_HELPER)

    assert (
        FINAL_EMBEDDED_PRICE_DUPLICATE_CLEANUP_REPAIRS
        <= set(_call_names_in_function(item_cleanup_helper))
    ), (
        "Embedded-price duplicate cleanup should be owned by postprocess item "
        "cleanup so the suffix trigger and duplicate consistency invariant run "
        "before final model serialization."
    )
    assert not _has_function_def(
        tree,
        RETIRED_FINAL_EMBEDDED_PRICE_DUPLICATE_CLEANUP_HELPER,
    ), (
        "The late embedded-price duplicate cleanup helper should be retired "
        "once postprocess item cleanup owns the behavior."
    )


def test_final_embedded_price_duplicate_cleanup_debt_is_removed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_duplicate_calls = [
        name
        for name in final_calls
        if name in FINAL_EMBEDDED_PRICE_DUPLICATE_CLEANUP_REPAIRS
    ]

    assert not direct_duplicate_calls, (
        "Embedded-price duplicate cleanup should not run directly in late final "
        "output repairs after postprocess item cleanup owns the behavior.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_duplicate_calls}"
    )
    assert (
        RETIRED_FINAL_EMBEDDED_PRICE_DUPLICATE_CLEANUP_HELPER not in final_calls
    ), (
        "The late embedded-price duplicate cleanup helper should not be called "
        "from _apply_final_receipt_output_repairs."
    )
    assert (
        RETIRED_FINAL_EMBEDDED_PRICE_DUPLICATE_CLEANUP_STAGE
        not in FINAL_OUTPUT_REPAIR_STAGES
    ), (
        "The embedded-price duplicate final-output stage should be removed "
        "from the tracked late repair list."
    )


def test_final_duplicate_row_cleanup_is_retired_to_postprocess():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    receipt_tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    duplicate_cleanup_helper = _function_def(
        receipt_tree,
        DUPLICATE_ROW_CLEANUP_PHASE_HELPER,
    )

    assert (
        FINAL_DUPLICATE_ROW_CLEANUP_REPAIRS
        <= set(_call_names_in_function(duplicate_cleanup_helper))
    ), (
        "Duplicate-row cleanup should be owned by postprocess duplicate row "
        "cleanup so the OCR occurrence-count trigger and subtotal-overage "
        "invariant run before final model serialization."
    )
    assert not _has_function_def(
        tree,
        RETIRED_FINAL_DUPLICATE_ROW_CLEANUP_HELPER,
    ), (
        "The late duplicate-row cleanup helper should be retired once "
        "postprocess duplicate row cleanup owns the behavior."
    )


def test_final_duplicate_row_cleanup_debt_is_removed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_duplicate_calls = [
        name
        for name in final_calls
        if name in FINAL_DUPLICATE_ROW_CLEANUP_REPAIRS
    ]

    assert not direct_duplicate_calls, (
        "Duplicate-row cleanup should not run directly in late final output "
        "repairs after postprocess owns the behavior.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_duplicate_calls}"
    )
    assert RETIRED_FINAL_DUPLICATE_ROW_CLEANUP_HELPER not in final_calls, (
        "The late duplicate-row cleanup helper should not be called from "
        "_apply_final_receipt_output_repairs."
    )
    assert (
        RETIRED_FINAL_DUPLICATE_ROW_CLEANUP_STAGE not in FINAL_OUTPUT_REPAIR_STAGES
    ), (
        "The duplicate-row final-output stage should be removed from the "
        "tracked late repair list."
    )


def test_final_discount_consistency_reconciliation_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    helper = _function_def(
        tree,
        FINAL_DISCOUNT_CONSISTENCY_RECONCILIATION_HELPER,
    )
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        FINAL_DISCOUNT_CONSISTENCY_RECONCILIATION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Late discount consistency reconciliation must be owned by the named "
        f"{FINAL_DISCOUNT_CONSISTENCY_RECONCILIATION_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{FINAL_DISCOUNT_CONSISTENCY_RECONCILIATION_HELPER} must document "
        "the negative-line-before-own-price trigger and item total/discount "
        "field consistency invariant."
    )


def test_final_discount_consistency_reconciliation_debt_is_helper_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_discount_calls = [
        name
        for name in final_calls
        if name in FINAL_DISCOUNT_CONSISTENCY_RECONCILIATION_REPAIRS
    ]
    helper_calls = [
        name
        for name in final_calls
        if name == FINAL_DISCOUNT_CONSISTENCY_RECONCILIATION_HELPER
    ]

    assert not direct_discount_calls, (
        "Late discount consistency reconciliation should run through the "
        "named helper so OCR discount placement and item discount arithmetic "
        "have one owner.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_discount_calls}"
    )
    assert (
        0
        < len(helper_calls)
        <= FINAL_DISCOUNT_CONSISTENCY_RECONCILIATION_STAGE_LIMIT
    ), (
        "Late discount consistency reconciliation helper calls must be "
        "explicit and bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {FINAL_DISCOUNT_CONSISTENCY_RECONCILIATION_STAGE_LIMIT}"
    )


def test_discount_consistency_reconciliation_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, DISCOUNT_CONSISTENCY_RECONCILIATION_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        DISCOUNT_CONSISTENCY_RECONCILIATION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Postprocess discount consistency reconciliation must be owned by the "
        f"named {DISCOUNT_CONSISTENCY_RECONCILIATION_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{DISCOUNT_CONSISTENCY_RECONCILIATION_PHASE_HELPER} must document "
        "the adjacent negative discount row trigger and item gross-minus-"
        "discount consistency invariant."
    )


def test_postprocess_discount_consistency_reconciliation_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_discount_calls = [
        name
        for name in postprocess_calls
        if name in DISCOUNT_CONSISTENCY_RECONCILIATION_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == DISCOUNT_CONSISTENCY_RECONCILIATION_PHASE_HELPER
    ]

    assert not direct_discount_calls, (
        "Postprocess discount consistency reconciliation should run through a "
        "named phase helper so OCR discount placement and item arithmetic have "
        "one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_discount_calls}"
    )
    assert (
        0
        < len(phase_calls)
        <= DISCOUNT_CONSISTENCY_RECONCILIATION_PHASE_CALL_LIMIT
    ), (
        "Postprocess discount consistency phase calls must be explicit and "
        "bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {DISCOUNT_CONSISTENCY_RECONCILIATION_PHASE_CALL_LIMIT}"
    )


def test_final_quantity_detail_reconciliation_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    helper = _function_def(tree, FINAL_QUANTITY_DETAIL_RECONCILIATION_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        FINAL_QUANTITY_DETAIL_RECONCILIATION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Late quantity-detail reconciliation must be owned by the named "
        f"{FINAL_QUANTITY_DETAIL_RECONCILIATION_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{FINAL_QUANTITY_DETAIL_RECONCILIATION_HELPER} must document the "
        "OCR quantity/unit-line trigger and qty times unit equals total invariant."
    )


def test_final_quantity_detail_reconciliation_debt_is_helper_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_qty_calls = [
        name
        for name in final_calls
        if name in FINAL_QUANTITY_DETAIL_RECONCILIATION_REPAIRS
    ]
    helper_calls = [
        name
        for name in final_calls
        if name == FINAL_QUANTITY_DETAIL_RECONCILIATION_HELPER
    ]

    assert not direct_qty_calls, (
        "Late quantity-detail reconciliation should run through the named "
        "helper so OCR unit-line evidence and qty/unit/total consistency have "
        "one owner.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_qty_calls}"
    )
    assert (
        0
        < len(helper_calls)
        <= FINAL_QUANTITY_DETAIL_RECONCILIATION_STAGE_LIMIT
    ), (
        "Late quantity-detail helper calls must be explicit and bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {FINAL_QUANTITY_DETAIL_RECONCILIATION_STAGE_LIMIT}"
    )


def test_final_ocr_description_reconciliation_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    helper = _function_def(tree, FINAL_OCR_DESCRIPTION_RECONCILIATION_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        FINAL_OCR_DESCRIPTION_RECONCILIATION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Late OCR description reconciliation must be owned by the named "
        f"{FINAL_OCR_DESCRIPTION_RECONCILIATION_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{FINAL_OCR_DESCRIPTION_RECONCILIATION_HELPER} must document the "
        "OCR description-context trigger and item field-consistency invariant."
    )


def test_final_ocr_description_reconciliation_debt_is_helper_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_description_calls = [
        name
        for name in final_calls
        if name in FINAL_OCR_DESCRIPTION_RECONCILIATION_REPAIRS
    ]
    helper_calls = [
        name
        for name in final_calls
        if name == FINAL_OCR_DESCRIPTION_RECONCILIATION_HELPER
    ]

    assert not direct_description_calls, (
        "Late OCR description reconciliation should run through the named "
        "helper so OCR description context and field consistency have one "
        "owner.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_description_calls}"
    )
    assert (
        0 < len(helper_calls) <= FINAL_OCR_DESCRIPTION_RECONCILIATION_STAGE_LIMIT
    ), (
        "Late OCR description helper calls must be explicit and bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {FINAL_OCR_DESCRIPTION_RECONCILIATION_STAGE_LIMIT}"
    )


def test_discounted_ocr_item_repair_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, DISCOUNTED_OCR_ITEM_REPAIR_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        DISCOUNTED_OCR_ITEM_REPAIR_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Postprocess discounted OCR item repair must be owned by the named "
        f"{DISCOUNTED_OCR_ITEM_REPAIR_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{DISCOUNTED_OCR_ITEM_REPAIR_PHASE_HELPER} must document the "
        "visible discount/stacked-price OCR trigger and item-sum or "
        "description field-consistency invariant."
    )


def test_postprocess_discounted_ocr_item_repair_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_discounted_item_calls = [
        name
        for name in postprocess_calls
        if name in DISCOUNTED_OCR_ITEM_REPAIR_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == DISCOUNTED_OCR_ITEM_REPAIR_PHASE_HELPER
    ]

    assert not direct_discounted_item_calls, (
        "Postprocess discounted OCR item repair should run through a named "
        "phase helper so visible discount rows, stacked price descriptions, "
        "and item-sum consistency have one owner.\n"
        "Direct calls still in postprocess_receipt: "
        f"{direct_discounted_item_calls}"
    )
    assert (
        0 < len(phase_calls) <= DISCOUNTED_OCR_ITEM_REPAIR_PHASE_CALL_LIMIT
    ), (
        "Postprocess discounted OCR item repair phase calls must be explicit "
        "and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {DISCOUNTED_OCR_ITEM_REPAIR_PHASE_CALL_LIMIT}"
    )


def test_final_adjacent_price_shift_reconciliation_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    helper = _function_def(
        tree,
        FINAL_ADJACENT_PRICE_SHIFT_RECONCILIATION_HELPER,
    )
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        FINAL_ADJACENT_PRICE_SHIFT_RECONCILIATION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Late adjacent OCR price-shift reconciliation must be owned by the "
        f"named {FINAL_ADJACENT_PRICE_SHIFT_RECONCILIATION_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{FINAL_ADJACENT_PRICE_SHIFT_RECONCILIATION_HELPER} must document "
        "the adjacent OCR price trigger and subtotal-balance invariant."
    )


def test_final_adjacent_price_shift_reconciliation_debt_is_helper_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_price_shift_calls = [
        name
        for name in final_calls
        if name in FINAL_ADJACENT_PRICE_SHIFT_RECONCILIATION_REPAIRS
    ]
    helper_calls = [
        name
        for name in final_calls
        if name == FINAL_ADJACENT_PRICE_SHIFT_RECONCILIATION_HELPER
    ]

    assert not direct_price_shift_calls, (
        "Late adjacent OCR price-shift reconciliation should run through the "
        "named helper so OCR neighbor evidence and subtotal arithmetic have "
        "one owner.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_price_shift_calls}"
    )
    assert (
        0
        < len(helper_calls)
        <= FINAL_ADJACENT_PRICE_SHIFT_RECONCILIATION_STAGE_LIMIT
    ), (
        "Late adjacent OCR price-shift helper calls must be explicit and "
        "bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {FINAL_ADJACENT_PRICE_SHIFT_RECONCILIATION_STAGE_LIMIT}"
    )


def test_final_prefixed_tax_marker_item_rows_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    helper = _function_def(
        tree,
        FINAL_PREFIXED_TAX_MARKER_ITEM_ROWS_HELPER,
    )
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        FINAL_PREFIXED_TAX_MARKER_ITEM_ROWS_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Late prefixed tax-marker item row projection must be owned by the "
        f"named {FINAL_PREFIXED_TAX_MARKER_ITEM_ROWS_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{FINAL_PREFIXED_TAX_MARKER_ITEM_ROWS_HELPER} must document the OCR "
        "tax-marker row trigger and subtotal/rate-base balance invariant."
    )


def test_final_prefixed_tax_marker_item_rows_debt_is_helper_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_marker_calls = [
        name
        for name in final_calls
        if name in FINAL_PREFIXED_TAX_MARKER_ITEM_ROWS_REPAIRS
    ]
    helper_calls = [
        name
        for name in final_calls
        if name == FINAL_PREFIXED_TAX_MARKER_ITEM_ROWS_HELPER
    ]

    assert not direct_marker_calls, (
        "Late prefixed tax-marker item row projection should run through the "
        "named helper so OCR marker evidence and subtotal/rate-base arithmetic "
        "have one owner.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_marker_calls}"
    )
    assert (
        0
        < len(helper_calls)
        <= FINAL_PREFIXED_TAX_MARKER_ITEM_ROWS_STAGE_LIMIT
    ), (
        "Late prefixed tax-marker item row helper calls must be explicit and "
        "bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {FINAL_PREFIXED_TAX_MARKER_ITEM_ROWS_STAGE_LIMIT}"
    )


def test_final_gap_item_recovery_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    helper = _function_def(tree, FINAL_GAP_ITEM_RECOVERY_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        FINAL_GAP_ITEM_RECOVERY_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Late gap item recovery must be owned by the named "
        f"{FINAL_GAP_ITEM_RECOVERY_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{FINAL_GAP_ITEM_RECOVERY_HELPER} must document the OCR gap trigger "
        "and item-sum/subtotal balance invariant."
    )


def test_final_gap_item_recovery_debt_is_helper_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_gap_calls = [
        name for name in final_calls if name in FINAL_GAP_ITEM_RECOVERY_REPAIRS
    ]
    helper_calls = [
        name for name in final_calls if name == FINAL_GAP_ITEM_RECOVERY_HELPER
    ]

    assert not direct_gap_calls, (
        "Late gap item recovery should run through the named helper so OCR "
        "gap evidence and item-sum arithmetic have one owner.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_gap_calls}"
    )
    assert 0 < len(helper_calls) <= FINAL_GAP_ITEM_RECOVERY_STAGE_LIMIT, (
        "Late gap item recovery helper calls must be explicit and bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {FINAL_GAP_ITEM_RECOVERY_STAGE_LIMIT}"
    )


def test_final_repeated_gap_item_recovery_debt_is_removed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_helper = _function_def(tree, FINAL_GAP_ITEM_RECOVERY_HELPER)
    final_calls = _call_names_in_function(final_repairs)

    assert (
        RETIRED_FINAL_REPEATED_GAP_ITEM_RECOVERY_REPAIR
        not in _call_names_in_function(final_helper)
    ), (
        "Repeated item gap recovery should be owned by postprocess gap item "
        "recovery, not by the late final gap helper."
    )
    assert FINAL_GAP_ITEM_RECOVERY_HELPER in final_calls, (
        "The final gap helper should remain for missing item gap recovery only."
    )
    assert (
        RETIRED_FINAL_REPEATED_GAP_ITEM_RECOVERY_STAGE
        not in FINAL_OUTPUT_REPAIR_STAGES
    ), (
        "The repeated item gap final-output stage should be removed from the "
        "tracked late repair list."
    )


def test_final_basket_marker_rows_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    helper = _function_def(tree, FINAL_BASKET_MARKER_ROWS_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        FINAL_BASKET_MARKER_ROWS_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Late basket marker row projection must be owned by the named "
        f"{FINAL_BASKET_MARKER_ROWS_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{FINAL_BASKET_MARKER_ROWS_HELPER} must document the OCR basket "
        "marker trigger and subtotal/rate-base arithmetic invariant."
    )


def test_final_basket_marker_rows_debt_is_helper_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_basket_calls = [
        name for name in final_calls if name in FINAL_BASKET_MARKER_ROWS_REPAIRS
    ]
    helper_calls = [
        name for name in final_calls if name == FINAL_BASKET_MARKER_ROWS_HELPER
    ]

    assert not direct_basket_calls, (
        "Late basket marker row projection should run through the named helper "
        "so OCR basket marker evidence and subtotal/rate-base arithmetic have "
        "one owner.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_basket_calls}"
    )
    assert 0 < len(helper_calls) <= FINAL_BASKET_MARKER_ROWS_STAGE_LIMIT, (
        "Late basket marker row helper calls must be explicit and bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {FINAL_BASKET_MARKER_ROWS_STAGE_LIMIT}"
    )


def test_basket_marker_rows_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, BASKET_MARKER_ROWS_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        BASKET_MARKER_ROWS_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Postprocess basket marker row projection must be owned by the named "
        f"{BASKET_MARKER_ROWS_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{BASKET_MARKER_ROWS_PHASE_HELPER} must document the OCR basket "
        "marker/stacked price trigger and subtotal/rate-base arithmetic "
        "invariant."
    )


def test_postprocess_basket_marker_rows_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_basket_calls = [
        name for name in postprocess_calls if name in BASKET_MARKER_ROWS_REPAIRS
    ]
    phase_calls = [
        name for name in postprocess_calls if name == BASKET_MARKER_ROWS_PHASE_HELPER
    ]

    assert not direct_basket_calls, (
        "Postprocess basket marker row projection should run through a named "
        "phase helper so basket marker OCR layout and subtotal/rate-base "
        "arithmetic have one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_basket_calls}"
    )
    assert 0 < len(phase_calls) <= BASKET_MARKER_ROWS_PHASE_CALL_LIMIT, (
        "Postprocess basket marker row phase calls must be explicit and "
        "bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {BASKET_MARKER_ROWS_PHASE_CALL_LIMIT}"
    )


def test_quantity_detail_reconciliation_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, QUANTITY_DETAIL_RECONCILIATION_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        QUANTITY_DETAIL_RECONCILIATION_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Quantity detail repairs must be owned by the named "
        f"{QUANTITY_DETAIL_RECONCILIATION_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{QUANTITY_DETAIL_RECONCILIATION_PHASE_HELPER} must document the OCR "
        "quantity-detail trigger and arithmetic or field-consistency invariant."
    )


def test_postprocess_quantity_detail_reconciliation_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_quantity_calls = [
        name for name in postprocess_calls if name in QUANTITY_DETAIL_RECONCILIATION_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == QUANTITY_DETAIL_RECONCILIATION_PHASE_HELPER
    ]

    assert not direct_quantity_calls, (
        "Quantity detail repairs should run through the named phase helper so "
        "OCR quantity triggers and qty * unit invariants have one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_quantity_calls}"
    )
    assert 0 < len(phase_calls) <= QUANTITY_DETAIL_RECONCILIATION_PHASE_CALL_LIMIT, (
        "Quantity detail reconciliation phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {QUANTITY_DETAIL_RECONCILIATION_PHASE_CALL_LIMIT}"
    )


def test_tax_category_assignment_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, TAX_CATEGORY_ASSIGNMENT_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        TAX_CATEGORY_ASSIGNMENT_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Tax category assignment and rate-base rebalance repairs must be owned "
        f"by the named {TAX_CATEGORY_ASSIGNMENT_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{TAX_CATEGORY_ASSIGNMENT_PHASE_HELPER} must document the OCR "
        "rate-marker trigger and tax/line-item consistency invariant."
    )


def test_postprocess_tax_category_assignment_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_tax_calls = [
        name for name in postprocess_calls if name in TAX_CATEGORY_ASSIGNMENT_REPAIRS
    ]
    phase_calls = [
        name for name in postprocess_calls if name == TAX_CATEGORY_ASSIGNMENT_PHASE_HELPER
    ]

    assert not direct_tax_calls, (
        "Tax category assignment repairs should run through the named phase "
        "helper so OCR rate markers, rate bases, and item/tax invariants have "
        "one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_tax_calls}"
    )
    assert 0 < len(phase_calls) <= TAX_CATEGORY_ASSIGNMENT_PHASE_CALL_LIMIT, (
        "Tax category assignment phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {TAX_CATEGORY_ASSIGNMENT_PHASE_CALL_LIMIT}"
    )


def test_bag_item_rate_base_reconciliation_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, BAG_ITEM_RATE_BASE_RECONCILIATION_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        BAG_ITEM_RATE_BASE_RECONCILIATION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Bag item price/rate-base reconciliation must be owned by the named "
        f"{BAG_ITEM_RATE_BASE_RECONCILIATION_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{BAG_ITEM_RATE_BASE_RECONCILIATION_PHASE_HELPER} must document "
        "the tiny printed 10% rate-base trigger and paid-bag total invariant."
    )


def test_postprocess_bag_item_rate_base_reconciliation_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_bag_calls = [
        name
        for name in postprocess_calls
        if name in BAG_ITEM_RATE_BASE_RECONCILIATION_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == BAG_ITEM_RATE_BASE_RECONCILIATION_PHASE_HELPER
    ]

    assert not direct_bag_calls, (
        "Bag item price/rate-base reconciliation should run through the "
        "named phase helper so paid-bag evidence and printed 10% rate-base "
        "arithmetic have one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_bag_calls}"
    )
    assert (
        0
        < len(phase_calls)
        <= BAG_ITEM_RATE_BASE_RECONCILIATION_PHASE_CALL_LIMIT
    ), (
        "Bag item price/rate-base reconciliation phase calls must be "
        "explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {BAG_ITEM_RATE_BASE_RECONCILIATION_PHASE_CALL_LIMIT}"
    )


def test_final_bag_item_rate_base_reconciliation_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    helper = _function_def(tree, FINAL_BAG_ITEM_RATE_BASE_RECONCILIATION_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        FINAL_BAG_ITEM_RATE_BASE_RECONCILIATION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Late bag item price/rate-base reconciliation must be owned by the "
        f"named {FINAL_BAG_ITEM_RATE_BASE_RECONCILIATION_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{FINAL_BAG_ITEM_RATE_BASE_RECONCILIATION_HELPER} must document "
        "the tiny printed 10% rate-base trigger and paid-bag total invariant."
    )


def test_final_bag_item_rate_base_reconciliation_debt_is_helper_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_bag_calls = [
        name
        for name in final_calls
        if name in FINAL_BAG_ITEM_RATE_BASE_RECONCILIATION_REPAIRS
    ]
    helper_calls = [
        name
        for name in final_calls
        if name == FINAL_BAG_ITEM_RATE_BASE_RECONCILIATION_HELPER
    ]

    assert not direct_bag_calls, (
        "Late bag item price/rate-base reconciliation should run through the "
        "named helper so paid-bag evidence and printed 10% rate-base "
        "arithmetic have one owner.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_bag_calls}"
    )
    assert (
        0
        < len(helper_calls)
        <= FINAL_BAG_ITEM_RATE_BASE_RECONCILIATION_STAGE_LIMIT
    ), (
        "Late bag item price/rate-base reconciliation helper calls must be "
        "explicit and bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {FINAL_BAG_ITEM_RATE_BASE_RECONCILIATION_STAGE_LIMIT}"
    )


def test_single_rate_inclusive_tax_restoration_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Single-rate inclusive tax restorations must be owned by the named "
        f"{SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_PHASE_HELPER} must document "
        "the printed single-rate inclusive tax trigger and total/tax invariant."
    )


def test_postprocess_single_rate_inclusive_tax_restoration_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_tax_calls = [
        name
        for name in postprocess_calls
        if name in SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_PHASE_HELPER
    ]

    assert not direct_tax_calls, (
        "Single-rate inclusive tax restoration should run through the named "
        "phase helper so printed target/tax rows and subtotal arithmetic have "
        "one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_tax_calls}"
    )
    assert 0 < len(phase_calls) <= SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_PHASE_CALL_LIMIT, (
        "Single-rate inclusive tax restoration phase calls must be explicit "
        "and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_PHASE_CALL_LIMIT}"
    )


def test_tax_excluded_rate_block_restoration_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, TAX_EXCLUDED_RATE_BLOCK_RESTORATION_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        TAX_EXCLUDED_RATE_BLOCK_RESTORATION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Tax-excluded per-rate block restorations must be owned by the named "
        f"{TAX_EXCLUDED_RATE_BLOCK_RESTORATION_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{TAX_EXCLUDED_RATE_BLOCK_RESTORATION_PHASE_HELPER} must document "
        "the printed tax-excluded subtotal/tax-row trigger and rate-pair "
        "consistency invariant."
    )


def test_postprocess_tax_excluded_rate_block_restoration_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_tax_calls = [
        name
        for name in postprocess_calls
        if name in TAX_EXCLUDED_RATE_BLOCK_RESTORATION_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == TAX_EXCLUDED_RATE_BLOCK_RESTORATION_PHASE_HELPER
    ]

    assert not direct_tax_calls, (
        "Tax-excluded per-rate block restoration should run through the named "
        "phase helper so printed subtotal/tax labels and rate-paired tax "
        "entries have one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_tax_calls}"
    )
    assert 0 < len(phase_calls) <= TAX_EXCLUDED_RATE_BLOCK_RESTORATION_PHASE_CALL_LIMIT, (
        "Tax-excluded per-rate block restoration phase calls must be explicit "
        "and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {TAX_EXCLUDED_RATE_BLOCK_RESTORATION_PHASE_CALL_LIMIT}"
    )


def test_explicit_tax_amount_restoration_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, EXPLICIT_TAX_AMOUNT_RESTORATION_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        EXPLICIT_TAX_AMOUNT_RESTORATION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Explicit tax amount restorations must be owned by the named "
        f"{EXPLICIT_TAX_AMOUNT_RESTORATION_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{EXPLICIT_TAX_AMOUNT_RESTORATION_PHASE_HELPER} must document "
        "the visible tax-rate amount row trigger and item/tax consistency "
        "invariant."
    )


def test_postprocess_explicit_tax_amount_restoration_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_tax_calls = [
        name
        for name in postprocess_calls
        if name in EXPLICIT_TAX_AMOUNT_RESTORATION_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == EXPLICIT_TAX_AMOUNT_RESTORATION_PHASE_HELPER
    ]

    assert not direct_tax_calls, (
        "Explicit tax amount restoration should run through the named phase "
        "helper so visible 税率N%税額 rows and tax-entry replacement have one "
        "owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_tax_calls}"
    )
    assert 0 < len(phase_calls) <= EXPLICIT_TAX_AMOUNT_RESTORATION_PHASE_CALL_LIMIT, (
        "Explicit tax amount restoration phase calls must be explicit and "
        "bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {EXPLICIT_TAX_AMOUNT_RESTORATION_PHASE_CALL_LIMIT}"
    )


def test_external_tax_total_restoration_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, EXTERNAL_TAX_TOTAL_RESTORATION_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        EXTERNAL_TAX_TOTAL_RESTORATION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "External tax total restorations must be owned by the named "
        f"{EXTERNAL_TAX_TOTAL_RESTORATION_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{EXTERNAL_TAX_TOTAL_RESTORATION_PHASE_HELPER} must document "
        "the printed subtotal plus external-tax summary trigger and "
        "total/payment arithmetic invariant."
    )


def test_postprocess_external_tax_total_restoration_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_tax_calls = [
        name
        for name in postprocess_calls
        if name in EXTERNAL_TAX_TOTAL_RESTORATION_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == EXTERNAL_TAX_TOTAL_RESTORATION_PHASE_HELPER
    ]

    assert not direct_tax_calls, (
        "External tax total restoration should run through the named phase "
        "helper so printed subtotal, external-tax entries, and visible "
        "summary/payment totals have one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_tax_calls}"
    )
    assert 0 < len(phase_calls) <= EXTERNAL_TAX_TOTAL_RESTORATION_PHASE_CALL_LIMIT, (
        "External tax total restoration phase calls must be explicit and "
        "bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {EXTERNAL_TAX_TOTAL_RESTORATION_PHASE_CALL_LIMIT}"
    )


def test_cash_tender_reconciliation_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, CASH_TENDER_RECONCILIATION_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        CASH_TENDER_RECONCILIATION_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Cash tender/change repairs must be owned by the named "
        f"{CASH_TENDER_RECONCILIATION_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{CASH_TENDER_RECONCILIATION_PHASE_HELPER} must document the OCR "
        "cash-layout trigger and total/payment consistency invariant."
    )


def test_postprocess_cash_tender_reconciliation_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_cash_calls = [
        name for name in postprocess_calls if name in CASH_TENDER_RECONCILIATION_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == CASH_TENDER_RECONCILIATION_PHASE_HELPER
    ]

    assert not direct_cash_calls, (
        "Cash tender/change repairs should run through the named phase helper "
        "so printed total, tendered amount, and change arithmetic have one "
        "owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_cash_calls}"
    )
    assert 0 < len(phase_calls) <= CASH_TENDER_RECONCILIATION_PHASE_CALL_LIMIT, (
        "Cash tender reconciliation phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {CASH_TENDER_RECONCILIATION_PHASE_CALL_LIMIT}"
    )


def test_payment_method_repair_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, PAYMENT_METHOD_REPAIR_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        PAYMENT_METHOD_REPAIR_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Payment method repairs must be owned by the named "
        f"{PAYMENT_METHOD_REPAIR_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{PAYMENT_METHOD_REPAIR_PHASE_HELPER} must document the OCR payment "
        "marker trigger and payment_method field-consistency invariant."
    )


def test_postprocess_payment_method_repair_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_payment_calls = [
        name for name in postprocess_calls if name in PAYMENT_METHOD_REPAIR_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == PAYMENT_METHOD_REPAIR_PHASE_HELPER
    ]

    assert not direct_payment_calls, (
        "Payment method repairs should run through the named phase helper so "
        "OCR payment markers and payment_method field consistency have one "
        "owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_payment_calls}"
    )
    assert 0 < len(phase_calls) <= PAYMENT_METHOD_REPAIR_PHASE_CALL_LIMIT, (
        "Payment method repair phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {PAYMENT_METHOD_REPAIR_PHASE_CALL_LIMIT}"
    )


def test_payment_points_reconciliation_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, PAYMENT_POINTS_RECONCILIATION_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        PAYMENT_POINTS_RECONCILIATION_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Payment/points repairs must be owned by the named "
        f"{PAYMENT_POINTS_RECONCILIATION_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{PAYMENT_POINTS_RECONCILIATION_PHASE_HELPER} must document the OCR "
        "points/payment trigger and total/payment consistency invariant."
    )


def test_postprocess_payment_points_reconciliation_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_payment_points_calls = [
        name for name in postprocess_calls if name in PAYMENT_POINTS_RECONCILIATION_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == PAYMENT_POINTS_RECONCILIATION_PHASE_HELPER
    ]

    assert not direct_payment_points_calls, (
        "Payment/points repairs should run through the named phase helper so "
        "OCR point-use evidence and total minus points payment arithmetic have "
        "one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_payment_points_calls}"
    )
    assert 0 < len(phase_calls) <= PAYMENT_POINTS_RECONCILIATION_PHASE_CALL_LIMIT, (
        "Payment/points reconciliation phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {PAYMENT_POINTS_RECONCILIATION_PHASE_CALL_LIMIT}"
    )


def test_service_receipt_recovery_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, SERVICE_RECEIPT_RECOVERY_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        SERVICE_RECEIPT_RECOVERY_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Service receipt recovery repairs must be owned by the named "
        f"{SERVICE_RECEIPT_RECOVERY_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{SERVICE_RECEIPT_RECOVERY_PHASE_HELPER} must document the OCR "
        "service-layout trigger and item/total/tax consistency invariant."
    )


def test_postprocess_service_receipt_recovery_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_service_calls = [
        name for name in postprocess_calls if name in SERVICE_RECEIPT_RECOVERY_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == SERVICE_RECEIPT_RECOVERY_PHASE_HELPER
    ]

    assert not direct_service_calls, (
        "Service receipt recovery repairs should run through the named phase "
        "helper so service-table layout, bare-service suppression, and "
        "single-service tax invariants have one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_service_calls}"
    )
    assert 0 < len(phase_calls) <= SERVICE_RECEIPT_RECOVERY_PHASE_CALL_LIMIT, (
        "Service receipt recovery phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {SERVICE_RECEIPT_RECOVERY_PHASE_CALL_LIMIT}"
    )


def test_body_total_layout_reconstruction_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, BODY_TOTAL_LAYOUT_RECONSTRUCTION_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        BODY_TOTAL_LAYOUT_RECONSTRUCTION_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Body-total layout reconstruction repairs must be owned by the named "
        f"{BODY_TOTAL_LAYOUT_RECONSTRUCTION_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{BODY_TOTAL_LAYOUT_RECONSTRUCTION_PHASE_HELPER} must document the "
        "visible body-total layout trigger and item/tax arithmetic invariant."
    )


def test_postprocess_body_total_layout_reconstruction_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_layout_calls = [
        name
        for name in postprocess_calls
        if name in BODY_TOTAL_LAYOUT_RECONSTRUCTION_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == BODY_TOTAL_LAYOUT_RECONSTRUCTION_PHASE_HELPER
    ]

    assert not direct_layout_calls, (
        "Body-total layout reconstruction should run through the named phase "
        "helper so split item rows, location, subtotal, and tax entries have "
        "one layout/arithmetic owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_layout_calls}"
    )
    assert 0 < len(phase_calls) <= BODY_TOTAL_LAYOUT_RECONSTRUCTION_PHASE_CALL_LIMIT, (
        "Body-total layout reconstruction phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {BODY_TOTAL_LAYOUT_RECONSTRUCTION_PHASE_CALL_LIMIT}"
    )


def test_ocr_description_reconciliation_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, OCR_DESCRIPTION_RECONCILIATION_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        OCR_DESCRIPTION_RECONCILIATION_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "OCR description reconciliation repairs must be owned by the named "
        f"{OCR_DESCRIPTION_RECONCILIATION_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{OCR_DESCRIPTION_RECONCILIATION_PHASE_HELPER} must document the OCR "
        "description-context trigger and item field-consistency invariant."
    )


def test_postprocess_ocr_description_reconciliation_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_description_calls = [
        name for name in postprocess_calls if name in OCR_DESCRIPTION_RECONCILIATION_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == OCR_DESCRIPTION_RECONCILIATION_PHASE_HELPER
    ]

    assert not direct_description_calls, (
        "OCR description reconciliation repairs should run through the named "
        "phase helper so code-row, duplicate, O-ring, colon-split, and bag "
        "description invariants have one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_description_calls}"
    )
    assert 0 < len(phase_calls) <= OCR_DESCRIPTION_RECONCILIATION_PHASE_CALL_LIMIT, (
        "OCR description reconciliation phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {OCR_DESCRIPTION_RECONCILIATION_PHASE_CALL_LIMIT}"
    )


def test_gap_item_recovery_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, GAP_ITEM_RECOVERY_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        GAP_ITEM_RECOVERY_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Gap item recovery repairs must be owned by the named "
        f"{GAP_ITEM_RECOVERY_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{GAP_ITEM_RECOVERY_PHASE_HELPER} must document the OCR row-gap "
        "trigger and subtotal/total arithmetic invariant."
    )


def test_postprocess_gap_item_recovery_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_gap_calls = [
        name for name in postprocess_calls if name in GAP_ITEM_RECOVERY_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == GAP_ITEM_RECOVERY_PHASE_HELPER
    ]

    assert not direct_gap_calls, (
        "Gap item recovery repairs should run through the named phase helper "
        "so missing, discounted, repeated, and repeated-block row recovery "
        "share one OCR/arithmetic owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_gap_calls}"
    )
    assert 0 < len(phase_calls) <= GAP_ITEM_RECOVERY_PHASE_CALL_LIMIT, (
        "Gap item recovery phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {GAP_ITEM_RECOVERY_PHASE_CALL_LIMIT}"
    )


def test_prefixed_tax_marker_item_rows_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, PREFIXED_TAX_MARKER_ITEM_ROWS_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        PREFIXED_TAX_MARKER_ITEM_ROWS_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Prefixed tax-marker item row projection must be owned by the named "
        f"{PREFIXED_TAX_MARKER_ITEM_ROWS_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{PREFIXED_TAX_MARKER_ITEM_ROWS_PHASE_HELPER} must document the OCR "
        "tax-marker row trigger and subtotal/rate-base balance invariant."
    )


def test_postprocess_prefixed_tax_marker_item_rows_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_marker_calls = [
        name
        for name in postprocess_calls
        if name in PREFIXED_TAX_MARKER_ITEM_ROWS_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == PREFIXED_TAX_MARKER_ITEM_ROWS_PHASE_HELPER
    ]

    assert not direct_marker_calls, (
        "Prefixed tax-marker item row projection should run through the named "
        "phase helper so marker-prefixed OCR rows and subtotal/rate-base "
        "arithmetic have one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_marker_calls}"
    )
    assert 0 < len(phase_calls) <= PREFIXED_TAX_MARKER_ITEM_ROWS_PHASE_CALL_LIMIT, (
        "Prefixed tax-marker item row phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {PREFIXED_TAX_MARKER_ITEM_ROWS_PHASE_CALL_LIMIT}"
    )


def test_low_value_bag_recovery_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, LOW_VALUE_BAG_RECOVERY_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        LOW_VALUE_BAG_RECOVERY_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Low-value bag recovery repairs must be owned by the named "
        f"{LOW_VALUE_BAG_RECOVERY_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{LOW_VALUE_BAG_RECOVERY_PHASE_HELPER} must document the OCR small-bag "
        "trigger and subtotal/total arithmetic invariant."
    )


def test_postprocess_low_value_bag_recovery_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_bag_calls = [
        name for name in postprocess_calls if name in LOW_VALUE_BAG_RECOVERY_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == LOW_VALUE_BAG_RECOVERY_PHASE_HELPER
    ]

    assert not direct_bag_calls, (
        "Low-value bag recovery repairs should run through the named phase "
        "helper so missing bag rows, overage replacement, and numeric OCR "
        "context share one OCR/arithmetic owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_bag_calls}"
    )
    assert 0 < len(phase_calls) <= LOW_VALUE_BAG_RECOVERY_PHASE_CALL_LIMIT, (
        "Low-value bag recovery phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {LOW_VALUE_BAG_RECOVERY_PHASE_CALL_LIMIT}"
    )


def test_item_name_price_cleanup_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, ITEM_NAME_PRICE_CLEANUP_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        ITEM_NAME_PRICE_CLEANUP_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Item name/price cleanup repairs must be owned by the named "
        f"{ITEM_NAME_PRICE_CLEANUP_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{ITEM_NAME_PRICE_CLEANUP_PHASE_HELPER} must document the OCR "
        "item-name/embedded-price trigger and item field-consistency invariant."
    )


def test_postprocess_item_name_price_cleanup_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_cleanup_calls = [
        name
        for name in postprocess_calls
        if name in ITEM_NAME_PRICE_CLEANUP_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == ITEM_NAME_PRICE_CLEANUP_PHASE_HELPER
    ]

    assert not direct_cleanup_calls, (
        "Item name/price cleanup should run through the named phase helper "
        "so OCR row ownership, embedded price suffixes, and item fields have "
        "one consistency owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_cleanup_calls}"
    )
    assert 0 < len(phase_calls) <= ITEM_NAME_PRICE_CLEANUP_PHASE_CALL_LIMIT, (
        "Item name/price cleanup phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {ITEM_NAME_PRICE_CLEANUP_PHASE_CALL_LIMIT}"
    )


def test_priced_name_item_repair_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, PRICED_NAME_ITEM_REPAIR_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        PRICED_NAME_ITEM_REPAIR_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Priced-name item repairs must be owned by the named "
        f"{PRICED_NAME_ITEM_REPAIR_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{PRICED_NAME_ITEM_REPAIR_PHASE_HELPER} must document the OCR "
        "priced-name trigger and subtotal/total item-sum invariant."
    )


def test_postprocess_priced_name_item_repair_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_repair_calls = [
        name
        for name in postprocess_calls
        if name in PRICED_NAME_ITEM_REPAIR_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == PRICED_NAME_ITEM_REPAIR_PHASE_HELPER
    ]

    assert not direct_repair_calls, (
        "Priced-name item repair should run through the named phase helper "
        "so OCR-visible item names, orphan amounts, and item-sum arithmetic "
        "have one consistency owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_repair_calls}"
    )
    assert 0 < len(phase_calls) <= PRICED_NAME_ITEM_REPAIR_PHASE_CALL_LIMIT, (
        "Priced-name item repair phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {PRICED_NAME_ITEM_REPAIR_PHASE_CALL_LIMIT}"
    )


def test_digit_misread_item_repair_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, DIGIT_MISREAD_ITEM_REPAIR_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        DIGIT_MISREAD_ITEM_REPAIR_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Digit-misread item repairs must be owned by the named "
        f"{DIGIT_MISREAD_ITEM_REPAIR_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{DIGIT_MISREAD_ITEM_REPAIR_PHASE_HELPER} must document the OCR "
        "digit-misread trigger and item-sum arithmetic invariant."
    )


def test_postprocess_digit_misread_item_repair_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_repair_calls = [
        name
        for name in postprocess_calls
        if name in DIGIT_MISREAD_ITEM_REPAIR_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == DIGIT_MISREAD_ITEM_REPAIR_PHASE_HELPER
    ]

    assert not direct_repair_calls, (
        "Digit-misread item repair should run through the named phase helper "
        "so OCR percent-marker evidence and item-sum arithmetic have one "
        "consistency owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_repair_calls}"
    )
    assert 0 < len(phase_calls) <= DIGIT_MISREAD_ITEM_REPAIR_PHASE_CALL_LIMIT, (
        "Digit-misread item repair phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {DIGIT_MISREAD_ITEM_REPAIR_PHASE_CALL_LIMIT}"
    )


def test_subtotal_item_price_repair_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, SUBTOTAL_ITEM_PRICE_REPAIR_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        SUBTOTAL_ITEM_PRICE_REPAIR_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Subtotal item price repairs must be owned by the named "
        f"{SUBTOTAL_ITEM_PRICE_REPAIR_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{SUBTOTAL_ITEM_PRICE_REPAIR_PHASE_HELPER} must document the OCR "
        "subtotal/nearby-price trigger and item-sum arithmetic invariant."
    )


def test_postprocess_subtotal_item_price_repair_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_repair_calls = [
        name
        for name in postprocess_calls
        if name in SUBTOTAL_ITEM_PRICE_REPAIR_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == SUBTOTAL_ITEM_PRICE_REPAIR_PHASE_HELPER
    ]

    assert not direct_repair_calls, (
        "Subtotal item price repair should run through the named phase helper "
        "so OCR subtotal evidence, nearby item prices, and item sums have one "
        "consistency owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_repair_calls}"
    )
    assert 0 < len(phase_calls) <= SUBTOTAL_ITEM_PRICE_REPAIR_PHASE_CALL_LIMIT, (
        "Subtotal item price repair phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {SUBTOTAL_ITEM_PRICE_REPAIR_PHASE_CALL_LIMIT}"
    )


def test_implausible_tax_amount_repair_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, IMPLAUSIBLE_TAX_AMOUNT_REPAIR_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        IMPLAUSIBLE_TAX_AMOUNT_REPAIR_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Implausible tax amount repairs must be owned by the named "
        f"{IMPLAUSIBLE_TAX_AMOUNT_REPAIR_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{IMPLAUSIBLE_TAX_AMOUNT_REPAIR_PHASE_HELPER} must document the OCR "
        "rate-base/tax-swap trigger and tax arithmetic invariant."
    )


def test_postprocess_implausible_tax_amount_repair_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_repair_calls = [
        name
        for name in postprocess_calls
        if name in IMPLAUSIBLE_TAX_AMOUNT_REPAIR_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == IMPLAUSIBLE_TAX_AMOUNT_REPAIR_PHASE_HELPER
    ]

    assert not direct_repair_calls, (
        "Implausible tax amount repair should run through the named phase helper "
        "so OCR rate-base evidence and tax arithmetic have one consistency owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_repair_calls}"
    )
    assert 0 < len(phase_calls) <= IMPLAUSIBLE_TAX_AMOUNT_REPAIR_PHASE_CALL_LIMIT, (
        "Implausible tax amount repair phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {IMPLAUSIBLE_TAX_AMOUNT_REPAIR_PHASE_CALL_LIMIT}"
    )


def test_vertical_price_qty_total_projection_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, VERTICAL_PRICE_QTY_TOTAL_PROJECTION_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        VERTICAL_PRICE_QTY_TOTAL_PROJECTION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Vertical price/qty/total row projection must be owned by the named "
        f"{VERTICAL_PRICE_QTY_TOTAL_PROJECTION_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{VERTICAL_PRICE_QTY_TOTAL_PROJECTION_PHASE_HELPER} must document the "
        "vertical OCR row trigger and unit/qty/subtotal arithmetic invariant."
    )


def test_postprocess_vertical_price_qty_total_projection_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_projection_calls = [
        name
        for name in postprocess_calls
        if name in VERTICAL_PRICE_QTY_TOTAL_PROJECTION_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == VERTICAL_PRICE_QTY_TOTAL_PROJECTION_PHASE_HELPER
    ]

    assert not direct_projection_calls, (
        "Vertical price/qty/total projection should run through the named phase "
        "helper so OCR row order and arithmetic have one consistency owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_projection_calls}"
    )
    assert (
        0 < len(phase_calls)
        <= VERTICAL_PRICE_QTY_TOTAL_PROJECTION_PHASE_CALL_LIMIT
    ), (
        "Vertical price/qty/total projection phase calls must be explicit and "
        "bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {VERTICAL_PRICE_QTY_TOTAL_PROJECTION_PHASE_CALL_LIMIT}"
    )


def test_code_prefixed_description_cleanup_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, CODE_PREFIXED_DESCRIPTION_CLEANUP_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        CODE_PREFIXED_DESCRIPTION_CLEANUP_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Code-prefixed item description cleanup must be owned by the named "
        f"{CODE_PREFIXED_DESCRIPTION_CLEANUP_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{CODE_PREFIXED_DESCRIPTION_CLEANUP_PHASE_HELPER} must document the "
        "OCR/POS code-prefix trigger and item description field-consistency "
        "invariant."
    )


def test_postprocess_code_prefixed_description_cleanup_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_cleanup_calls = [
        name
        for name in postprocess_calls
        if name in CODE_PREFIXED_DESCRIPTION_CLEANUP_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == CODE_PREFIXED_DESCRIPTION_CLEANUP_PHASE_HELPER
    ]

    assert not direct_cleanup_calls, (
        "Code-prefixed item description cleanup should run through the named "
        "phase helper so OCR/POS code prefixes and item description fields "
        "have one consistency owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_cleanup_calls}"
    )
    assert (
        0
        < len(phase_calls)
        <= CODE_PREFIXED_DESCRIPTION_CLEANUP_PHASE_CALL_LIMIT
    ), (
        "Code-prefixed description cleanup phase calls must be explicit and "
        "bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {CODE_PREFIXED_DESCRIPTION_CLEANUP_PHASE_CALL_LIMIT}"
    )


def test_adjacent_price_shift_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, ADJACENT_PRICE_SHIFT_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        ADJACENT_PRICE_SHIFT_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Adjacent OCR price-shift repairs must be owned by the named "
        f"{ADJACENT_PRICE_SHIFT_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{ADJACENT_PRICE_SHIFT_PHASE_HELPER} must document the adjacent OCR "
        "row trigger and subtotal arithmetic invariant."
    )


def test_postprocess_adjacent_price_shift_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_price_shift_calls = [
        name for name in postprocess_calls if name in ADJACENT_PRICE_SHIFT_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == ADJACENT_PRICE_SHIFT_PHASE_HELPER
    ]

    assert not direct_price_shift_calls, (
        "Adjacent OCR price-shift repairs should run through the named phase "
        "helper so neighboring OCR row projection has one arithmetic owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_price_shift_calls}"
    )
    assert 0 < len(phase_calls) <= ADJACENT_PRICE_SHIFT_PHASE_CALL_LIMIT, (
        "Adjacent OCR price-shift phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {ADJACENT_PRICE_SHIFT_PHASE_CALL_LIMIT}"
    )


def test_bag_amount_shift_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, BAG_AMOUNT_SHIFT_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        BAG_AMOUNT_SHIFT_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Paid-bag/product amount-shift repairs must be owned by the named "
        f"{BAG_AMOUNT_SHIFT_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{BAG_AMOUNT_SHIFT_PHASE_HELPER} must document the OCR paid-bag "
        "row trigger and subtotal/rate-base arithmetic invariant."
    )


def test_postprocess_bag_amount_shift_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_bag_shift_calls = [
        name for name in postprocess_calls if name in BAG_AMOUNT_SHIFT_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == BAG_AMOUNT_SHIFT_PHASE_HELPER
    ]

    assert not direct_bag_shift_calls, (
        "Paid-bag/product amount-shift repairs should run through the named "
        "phase helper so OCR row order, rate bases, and subtotal arithmetic "
        "have one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_bag_shift_calls}"
    )
    assert 0 < len(phase_calls) <= BAG_AMOUNT_SHIFT_PHASE_CALL_LIMIT, (
        "Paid-bag/product amount-shift phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {BAG_AMOUNT_SHIFT_PHASE_CALL_LIMIT}"
    )


def test_line_item_cleanup_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, LINE_ITEM_CLEANUP_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(LINE_ITEM_CLEANUP_REPAIRS - set(_call_names_in_function(helper)))
    assert not missing_repairs, (
        "Line-item cleanup/drop repairs must be owned by the named "
        f"{LINE_ITEM_CLEANUP_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{LINE_ITEM_CLEANUP_PHASE_HELPER} must document the OCR/layout trigger "
        "and arithmetic or field-consistency invariant it enforces."
    )


def test_postprocess_line_item_cleanup_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_cleanup_calls = [
        name for name in postprocess_calls if name in LINE_ITEM_CLEANUP_REPAIRS
    ]
    phase_calls = [
        name for name in postprocess_calls if name == LINE_ITEM_CLEANUP_PHASE_HELPER
    ]

    assert not direct_cleanup_calls, (
        "Line-item cleanup/drop repairs should run through the named phase "
        "helper so OCR/layout triggers and item-total invariants have one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_cleanup_calls}"
    )
    assert 0 < len(phase_calls) <= LINE_ITEM_CLEANUP_PHASE_CALL_LIMIT, (
        "Line-item cleanup phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; limit: {LINE_ITEM_CLEANUP_PHASE_CALL_LIMIT}"
    )


def test_phantom_tax_amount_cleanup_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, PHANTOM_TAX_AMOUNT_CLEANUP_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        PHANTOM_TAX_AMOUNT_CLEANUP_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Phantom tax-amount cleanup must be owned by the named "
        f"{PHANTOM_TAX_AMOUNT_CLEANUP_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{PHANTOM_TAX_AMOUNT_CLEANUP_PHASE_HELPER} must document the OCR tax "
        "amount phantom-row trigger and line-item/tax field-consistency "
        "invariant."
    )


def test_postprocess_phantom_tax_amount_cleanup_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_cleanup_calls = [
        name
        for name in postprocess_calls
        if name in PHANTOM_TAX_AMOUNT_CLEANUP_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == PHANTOM_TAX_AMOUNT_CLEANUP_PHASE_HELPER
    ]

    assert not direct_cleanup_calls, (
        "Phantom tax-amount cleanup should run through the named phase helper "
        "so printed tax amounts and corrupted item rows have one consistency "
        "owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_cleanup_calls}"
    )
    assert 0 < len(phase_calls) <= PHANTOM_TAX_AMOUNT_CLEANUP_PHASE_CALL_LIMIT, (
        "Phantom tax-amount cleanup phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {PHANTOM_TAX_AMOUNT_CLEANUP_PHASE_CALL_LIMIT}"
    )


def test_final_receipt_output_repairs_are_explicit_traced_stages():
    stages = _final_output_repair_stage_calls()
    stage_names = tuple(stage for stage, _line in stages)
    justifications = _final_output_repair_justifications()
    justification_keys = set(justifications)
    phase_names = _postprocess_phase_names()

    assert stage_names == FINAL_OUTPUT_REPAIR_STAGES, (
        "_apply_final_receipt_output_repairs changed. Late semantic repairs "
        "must stay behind the trace-recording run(stage, repair) wrapper and "
        "must update FINAL_OUTPUT_REPAIR_STAGES with a reviewable stage label.\n"
        f"Current stages: {stage_names}"
    )
    assert len(stage_names) == len(set(stage_names)), (
        "Late repair stage labels must be unique so mutation traces are useful."
    )
    assert justification_keys == set(stage_names), (
        "Every late final-output repair must have an explicit owner-phase "
        "justification in FINAL_RECEIPT_OUTPUT_REPAIR_JUSTIFICATIONS.\n"
        f"Missing: {sorted(set(stage_names) - justification_keys)}\n"
        f"Stale: {sorted(justification_keys - set(stage_names))}"
    )
    malformed = {
        stage: value
        for stage, value in justifications.items()
        if (
            not isinstance(value, tuple)
            or len(value) != 2
            or value[0] not in phase_names
            or not isinstance(value[1], str)
            or not value[1].strip()
        )
    }
    assert not malformed, (
        "Every late repair justification must name a valid postprocess owner "
        "phase and a non-empty reason.\n"
        f"Malformed: {malformed}"
    )


def test_no_new_suspicious_japanese_product_or_location_literals():
    baseline = _baseline_japanese_string_counts()
    current = _current_japanese_string_counts()
    new_literals = []
    for signature, count in current.items():
        extra = count - baseline.get(signature, 0)
        if extra <= 0:
            continue
        path, value = signature
        if _looks_structural_japanese_literal(value):
            continue
        new_literals.append((path, value, extra))

    assert not new_literals, (
        "Production parser code gained Japanese literals that do not look like "
        "structural receipt labels. Do not add product, merchant, location, or "
        "answer-key strings to parser code; derive behavior from OCR structure "
        "and arithmetic invariants instead.\n"
        + "\n".join(
            f"  - {path}: {value!r} (+{extra})"
            for path, value, extra in new_literals[:40]
        )
    )


def test_postprocess_receipt_is_idempotent_at_guardrail_level():
    from receipt_parser.pipeline_receipt import (
        _snapshot_receipt_mutation_fields,
        postprocess_receipt,
    )

    extracted = {
        "document_type": "receipt",
        "merchant": "テスト店",
        "currency": "JPY",
        "total": 1100,
        "subtotal": 1000,
        "taxes": [{"rate": "10%", "label": "外税", "amount": 100}],
        "line_items": [
            {
                "description": "テスト商品",
                "qty": 1,
                "unit_price": 1000,
                "total": 1000,
                "tax_category": "10%",
            },
        ],
        "points_used": 0,
    }
    text = "テスト店\nテスト商品\n¥1,000\n小計\n¥1,000\n外税\n¥100\n合計\n¥1,100"

    postprocess_receipt(extracted, text, 0.9, {}, {}, "test-model")
    once = copy.deepcopy(_snapshot_receipt_mutation_fields(extracted))
    postprocess_receipt(extracted, text, 0.9, {}, {}, "test-model")
    twice = _snapshot_receipt_mutation_fields(extracted)

    assert twice == once
