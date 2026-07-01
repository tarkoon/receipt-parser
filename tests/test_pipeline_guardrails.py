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

import pytest
import tokenize
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARSER_DIR = ROOT / "src" / "receipt_parser"
FINAL_OUTPUT_PATH = PARSER_DIR / "receipt_output.py"
PHASE_TRACE_PATH = PARSER_DIR / "receipt_phase_trace.py"
POSTPROCESS_PATH = PARSER_DIR / "receipt_postprocess.py"
POSTPROCESS_PHASES_PATH = PARSER_DIR / "receipt_postprocess_phases.py"
BASELINE_LITERAL_SOURCE_BY_FILE = {
    PARSER_DIR / "receipt_financial.py": PARSER_DIR / "pipeline_receipt.py",
    PARSER_DIR / "receipt_location.py": PARSER_DIR / "pipeline.py",
    PARSER_DIR / "receipt_output.py": PARSER_DIR / "pipeline.py",
    PARSER_DIR / "receipt_phase_trace.py": PARSER_DIR / "pipeline_receipt.py",
    PARSER_DIR / "receipt_postprocess.py": PARSER_DIR / "pipeline_receipt.py",
    PARSER_DIR / "receipt_postprocess_phases.py": PARSER_DIR / "pipeline_receipt.py",
    PARSER_DIR / "receipt_row_projection.py": PARSER_DIR / "pipeline_receipt.py",
    PARSER_DIR / "receipt_marker_projection.py": PARSER_DIR / "pipeline_receipt.py",
    PARSER_DIR / "receipt_item_cleanup.py": PARSER_DIR / "pipeline_receipt.py",
    PARSER_DIR / "receipt_recovery.py": PARSER_DIR / "pipeline_receipt.py",
    PARSER_DIR / "receipt_late_repairs.py": PARSER_DIR / "pipeline_receipt.py",
    PARSER_DIR / "receipt_totals.py": PARSER_DIR / "pipeline_receipt.py",
    PARSER_DIR / "receipt_tax_categories.py": PARSER_DIR / "pipeline_receipt.py",
    PARSER_DIR / "receipt_identity_payment.py": PARSER_DIR / "pipeline_receipt.py",
    PARSER_DIR / "receipt_items.py": PARSER_DIR / "pipeline_receipt.py",
    PARSER_DIR / "receipt_projection.py": PARSER_DIR / "pipeline_receipt.py",
    PARSER_DIR / "receipt_item_repair.py": PARSER_DIR / "pipeline_receipt.py",
}
SCANNED_FILES = tuple(
    sorted({
        PARSER_DIR / "pipeline.py",
        *PARSER_DIR.glob("pipeline_*.py"),
        *PARSER_DIR.glob("receipt_*.py"),
    })
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
BESPOKE_RECEIPT_LITERAL_RE = re.compile(r"(?:\bVIO\b|サンリブ)")
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
POSTPROCESS_REPAIR_CALL_LIMIT = 0

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
    "_restore_tax_entries_from_item_rate_sums",
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
STACKED_INCLUSIVE_TAX_RESTORATION_REPAIRS = {
    "_restore_stacked_inclusive_tax_block",
}
STACKED_INCLUSIVE_TAX_RESTORATION_PHASE_HELPER = (
    "_run_stacked_inclusive_tax_restoration_phase"
)
STACKED_INCLUSIVE_TAX_RESTORATION_PHASE_CALL_LIMIT = 1
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
    "_revert_unsupported_qty_inflation",
}
BAG_AMOUNT_SHIFT_PHASE_HELPER = "_run_bag_amount_shift_reconciliation_phase"
BAG_AMOUNT_SHIFT_PHASE_CALL_LIMIT = 3
FINANCIAL_TOTALS_REPAIR_REPAIRS = {
    "_apply_financial_overrides",
}
FINANCIAL_TOTALS_REPAIR_PHASE_HELPER = "_run_financial_totals_repair_phase"
FINANCIAL_TOTALS_REPAIR_PHASE_CALL_LIMIT = 1
LINE_ITEM_CLEANUP_REPAIRS = {
    "_fix_line_items",
    "_drop_duplicate_with_embedded_price",
    "_drop_non_product_line_items",
    "_drop_numeric_marker_description_rows",
}
LINE_ITEM_CLEANUP_PHASE_HELPER = "_run_line_item_cleanup_phase"
LINE_ITEM_CLEANUP_PHASE_CALL_LIMIT = 15
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
STACKED_NAME_PRICE_PROJECTION_REPAIRS = {
    "_replace_stacked_name_price_rows_when_balanced",
}
STACKED_NAME_PRICE_PROJECTION_PHASE_HELPER = (
    "_run_stacked_name_price_projection_phase"
)
STACKED_NAME_PRICE_PROJECTION_PHASE_CALL_LIMIT = 1
SINGLE_ITEM_QUANTITY_REPAIR_REPAIRS = {
    "_fix_single_item_qty_from_ocr",
}
SINGLE_ITEM_QUANTITY_REPAIR_PHASE_HELPER = "_run_single_item_quantity_repair_phase"
SINGLE_ITEM_QUANTITY_REPAIR_PHASE_CALL_LIMIT = 1
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
    r"現金|預|釣|支払|売上|ポイント|領収|レシート|登録番号|電話|TEL|ありがとう|"
    r"店|支店|営業所|料金所|住所|市|区|町|村|県|都|道|府|"
    r"年|月|日|時|分|個|点|円|品番|JAN|バーコード|除|外|内|"
    r"\\u[0-9a-fA-F]{4}|ぁ-ん|ァ-ン|ァ-ヶ|一-龥|¥|￥)"
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


def _literal_count_key_path(path: Path) -> Path:
    return BASELINE_LITERAL_SOURCE_BY_FILE.get(path, path)


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
    function = _function_def(_parse_file(POSTPROCESS_PATH), "postprocess_receipt")
    calls = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if _is_repair_call_name(name):
            calls.append((name or "", node.lineno))
    return calls


def _final_output_repair_stage_calls() -> list[tuple[str, int]]:
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(PHASE_TRACE_PATH)
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
        rel = _relative(_literal_count_key_path(path))
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
    baseline_paths = sorted({_literal_count_key_path(path) for path in SCANNED_FILES})
    for path in baseline_paths:
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

        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            match = BESPOKE_RECEIPT_LITERAL_RE.search(node.value)
            if match:
                violations.append(
                    Violation(
                        rel,
                        node.lineno,
                        function,
                        "bespoke_receipt_literal",
                        match.group(0),
                    )
                )

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


POSTPROCESS_PHASE_INVARIANT_CASES = (
    ("jan_pos_row_projection", JAN_POS_ROW_PROJECTION_REPAIRS, JAN_POS_ROW_PROJECTION_PHASE_HELPER),
    ("barcode_row_projection", BARCODE_ROW_PROJECTION_REPAIRS, BARCODE_ROW_PROJECTION_PHASE_HELPER),
    ("dense_item_row_projection", DENSE_ITEM_ROW_PROJECTION_REPAIRS, DENSE_ITEM_ROW_PROJECTION_PHASE_HELPER),
    ("dense_sequence_row_projection", DENSE_SEQUENCE_ROW_PROJECTION_REPAIRS, DENSE_SEQUENCE_ROW_PROJECTION_PHASE_HELPER),
    ("campaign_discount_projection", CAMPAIGN_DISCOUNT_PROJECTION_REPAIRS, CAMPAIGN_DISCOUNT_PROJECTION_PHASE_HELPER),
    ("split_price_block_projection", SPLIT_PRICE_BLOCK_PROJECTION_REPAIRS, SPLIT_PRICE_BLOCK_PROJECTION_PHASE_HELPER),
    ("coupon_discount_projection", COUPON_DISCOUNT_PROJECTION_REPAIRS, COUPON_DISCOUNT_PROJECTION_PHASE_HELPER),
    ("following_ocr_price_projection", FOLLOWING_OCR_PRICE_PROJECTION_REPAIRS, FOLLOWING_OCR_PRICE_PROJECTION_PHASE_HELPER),
    ("merchant_identity_repair", MERCHANT_IDENTITY_REPAIR_REPAIRS, MERCHANT_IDENTITY_REPAIR_PHASE_HELPER),
    ("transaction_datetime_repair", TRANSACTION_DATETIME_REPAIR_REPAIRS, TRANSACTION_DATETIME_REPAIR_PHASE_HELPER),
    ("toll_payment_reference_repair", TOLL_PAYMENT_REFERENCE_REPAIR_REPAIRS, TOLL_PAYMENT_REFERENCE_REPAIR_PHASE_HELPER),
    ("header_location_repair", HEADER_LOCATION_REPAIR_REPAIRS, HEADER_LOCATION_REPAIR_PHASE_HELPER),
    ("bag_item_ocr_repair", BAG_ITEM_OCR_REPAIR_REPAIRS, BAG_ITEM_OCR_REPAIR_PHASE_HELPER),
    ("discount_consistency_reconciliation", DISCOUNT_CONSISTENCY_RECONCILIATION_REPAIRS, DISCOUNT_CONSISTENCY_RECONCILIATION_PHASE_HELPER),
    ("discounted_ocr_item_repair", DISCOUNTED_OCR_ITEM_REPAIR_REPAIRS, DISCOUNTED_OCR_ITEM_REPAIR_PHASE_HELPER),
    ("basket_marker_rows", BASKET_MARKER_ROWS_REPAIRS, BASKET_MARKER_ROWS_PHASE_HELPER),
    ("quantity_detail_reconciliation", QUANTITY_DETAIL_RECONCILIATION_REPAIRS, QUANTITY_DETAIL_RECONCILIATION_PHASE_HELPER),
    ("tax_category_assignment", TAX_CATEGORY_ASSIGNMENT_REPAIRS, TAX_CATEGORY_ASSIGNMENT_PHASE_HELPER),
    ("bag_item_rate_base_reconciliation", BAG_ITEM_RATE_BASE_RECONCILIATION_REPAIRS, BAG_ITEM_RATE_BASE_RECONCILIATION_PHASE_HELPER),
    ("single_rate_inclusive_tax_restoration", SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_REPAIRS, SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_PHASE_HELPER),
    ("stacked_inclusive_tax_restoration", STACKED_INCLUSIVE_TAX_RESTORATION_REPAIRS, STACKED_INCLUSIVE_TAX_RESTORATION_PHASE_HELPER),
    ("tax_excluded_rate_block_restoration", TAX_EXCLUDED_RATE_BLOCK_RESTORATION_REPAIRS, TAX_EXCLUDED_RATE_BLOCK_RESTORATION_PHASE_HELPER),
    ("explicit_tax_amount_restoration", EXPLICIT_TAX_AMOUNT_RESTORATION_REPAIRS, EXPLICIT_TAX_AMOUNT_RESTORATION_PHASE_HELPER),
    ("external_tax_total_restoration", EXTERNAL_TAX_TOTAL_RESTORATION_REPAIRS, EXTERNAL_TAX_TOTAL_RESTORATION_PHASE_HELPER),
    ("cash_tender_reconciliation", CASH_TENDER_RECONCILIATION_REPAIRS, CASH_TENDER_RECONCILIATION_PHASE_HELPER),
    ("payment_method_repair", PAYMENT_METHOD_REPAIR_REPAIRS, PAYMENT_METHOD_REPAIR_PHASE_HELPER),
    ("payment_points_reconciliation", PAYMENT_POINTS_RECONCILIATION_REPAIRS, PAYMENT_POINTS_RECONCILIATION_PHASE_HELPER),
    ("service_receipt_recovery", SERVICE_RECEIPT_RECOVERY_REPAIRS, SERVICE_RECEIPT_RECOVERY_PHASE_HELPER),
    ("body_total_layout_reconstruction", BODY_TOTAL_LAYOUT_RECONSTRUCTION_REPAIRS, BODY_TOTAL_LAYOUT_RECONSTRUCTION_PHASE_HELPER),
    ("ocr_description_reconciliation", OCR_DESCRIPTION_RECONCILIATION_REPAIRS, OCR_DESCRIPTION_RECONCILIATION_PHASE_HELPER),
    ("gap_item_recovery", GAP_ITEM_RECOVERY_REPAIRS, GAP_ITEM_RECOVERY_PHASE_HELPER),
    ("prefixed_tax_marker_item_rows", PREFIXED_TAX_MARKER_ITEM_ROWS_REPAIRS, PREFIXED_TAX_MARKER_ITEM_ROWS_PHASE_HELPER),
    ("low_value_bag_recovery", LOW_VALUE_BAG_RECOVERY_REPAIRS, LOW_VALUE_BAG_RECOVERY_PHASE_HELPER),
    ("item_name_price_cleanup", ITEM_NAME_PRICE_CLEANUP_REPAIRS, ITEM_NAME_PRICE_CLEANUP_PHASE_HELPER),
    ("priced_name_item_repair", PRICED_NAME_ITEM_REPAIR_REPAIRS, PRICED_NAME_ITEM_REPAIR_PHASE_HELPER),
    ("digit_misread_item_repair", DIGIT_MISREAD_ITEM_REPAIR_REPAIRS, DIGIT_MISREAD_ITEM_REPAIR_PHASE_HELPER),
    ("subtotal_item_price_repair", SUBTOTAL_ITEM_PRICE_REPAIR_REPAIRS, SUBTOTAL_ITEM_PRICE_REPAIR_PHASE_HELPER),
    ("implausible_tax_amount_repair", IMPLAUSIBLE_TAX_AMOUNT_REPAIR_REPAIRS, IMPLAUSIBLE_TAX_AMOUNT_REPAIR_PHASE_HELPER),
    ("vertical_price_qty_total_projection", VERTICAL_PRICE_QTY_TOTAL_PROJECTION_REPAIRS, VERTICAL_PRICE_QTY_TOTAL_PROJECTION_PHASE_HELPER),
    ("stacked_name_price_projection", STACKED_NAME_PRICE_PROJECTION_REPAIRS, STACKED_NAME_PRICE_PROJECTION_PHASE_HELPER),
    ("single_item_quantity_repair", SINGLE_ITEM_QUANTITY_REPAIR_REPAIRS, SINGLE_ITEM_QUANTITY_REPAIR_PHASE_HELPER),
    ("code_prefixed_description_cleanup", CODE_PREFIXED_DESCRIPTION_CLEANUP_REPAIRS, CODE_PREFIXED_DESCRIPTION_CLEANUP_PHASE_HELPER),
    ("adjacent_price_shift", ADJACENT_PRICE_SHIFT_REPAIRS, ADJACENT_PRICE_SHIFT_PHASE_HELPER),
    ("bag_amount_shift", BAG_AMOUNT_SHIFT_REPAIRS, BAG_AMOUNT_SHIFT_PHASE_HELPER),
    ("financial_totals_repair", FINANCIAL_TOTALS_REPAIR_REPAIRS, FINANCIAL_TOTALS_REPAIR_PHASE_HELPER),
    ("line_item_cleanup", LINE_ITEM_CLEANUP_REPAIRS, LINE_ITEM_CLEANUP_PHASE_HELPER),
    ("phantom_tax_amount_cleanup", PHANTOM_TAX_AMOUNT_CLEANUP_REPAIRS, PHANTOM_TAX_AMOUNT_CLEANUP_PHASE_HELPER),
)

POSTPROCESS_PHASE_OWNERSHIP_CASES = (
    ("jan_pos_row_projection", JAN_POS_ROW_PROJECTION_REPAIRS, JAN_POS_ROW_PROJECTION_PHASE_HELPER, JAN_POS_ROW_PROJECTION_PHASE_CALL_LIMIT),
    ("barcode_row_projection", BARCODE_ROW_PROJECTION_REPAIRS, BARCODE_ROW_PROJECTION_PHASE_HELPER, BARCODE_ROW_PROJECTION_PHASE_CALL_LIMIT),
    ("dense_item_row_projection", DENSE_ITEM_ROW_PROJECTION_REPAIRS, DENSE_ITEM_ROW_PROJECTION_PHASE_HELPER, DENSE_ITEM_ROW_PROJECTION_PHASE_CALL_LIMIT),
    ("dense_sequence_row_projection", DENSE_SEQUENCE_ROW_PROJECTION_REPAIRS, DENSE_SEQUENCE_ROW_PROJECTION_PHASE_HELPER, DENSE_SEQUENCE_ROW_PROJECTION_PHASE_CALL_LIMIT),
    ("campaign_discount_projection", CAMPAIGN_DISCOUNT_PROJECTION_REPAIRS, CAMPAIGN_DISCOUNT_PROJECTION_PHASE_HELPER, CAMPAIGN_DISCOUNT_PROJECTION_PHASE_CALL_LIMIT),
    ("split_price_block_projection", SPLIT_PRICE_BLOCK_PROJECTION_REPAIRS, SPLIT_PRICE_BLOCK_PROJECTION_PHASE_HELPER, SPLIT_PRICE_BLOCK_PROJECTION_PHASE_CALL_LIMIT),
    ("printed_summary_total_repair", PRINTED_SUMMARY_TOTAL_REPAIR_REPAIRS, PRINTED_SUMMARY_TOTAL_REPAIR_PHASE_HELPER, PRINTED_SUMMARY_TOTAL_REPAIR_PHASE_CALL_LIMIT),
    ("printed_item_sum_total_repair", PRINTED_ITEM_SUM_TOTAL_REPAIR_REPAIRS, PRINTED_ITEM_SUM_TOTAL_REPAIR_PHASE_HELPER, PRINTED_ITEM_SUM_TOTAL_REPAIR_PHASE_CALL_LIMIT),
    ("printed_external_tax_amount_restoration", PRINTED_EXTERNAL_TAX_AMOUNT_RESTORATION_REPAIRS, PRINTED_EXTERNAL_TAX_AMOUNT_RESTORATION_PHASE_HELPER, PRINTED_EXTERNAL_TAX_AMOUNT_RESTORATION_PHASE_CALL_LIMIT),
    ("bare_number_tax_summary_restoration", BARE_NUMBER_TAX_SUMMARY_RESTORATION_REPAIRS, BARE_NUMBER_TAX_SUMMARY_RESTORATION_PHASE_HELPER, BARE_NUMBER_TAX_SUMMARY_RESTORATION_PHASE_CALL_LIMIT),
    ("coupon_discount_projection", COUPON_DISCOUNT_PROJECTION_REPAIRS, COUPON_DISCOUNT_PROJECTION_PHASE_HELPER, COUPON_DISCOUNT_PROJECTION_PHASE_CALL_LIMIT),
    ("following_ocr_price_projection", FOLLOWING_OCR_PRICE_PROJECTION_REPAIRS, FOLLOWING_OCR_PRICE_PROJECTION_PHASE_HELPER, FOLLOWING_OCR_PRICE_PROJECTION_PHASE_CALL_LIMIT),
    ("merchant_identity_repair", MERCHANT_IDENTITY_REPAIR_REPAIRS, MERCHANT_IDENTITY_REPAIR_PHASE_HELPER, MERCHANT_IDENTITY_REPAIR_PHASE_CALL_LIMIT),
    ("transaction_datetime_repair", TRANSACTION_DATETIME_REPAIR_REPAIRS, TRANSACTION_DATETIME_REPAIR_PHASE_HELPER, TRANSACTION_DATETIME_REPAIR_PHASE_CALL_LIMIT),
    ("toll_payment_reference_repair", TOLL_PAYMENT_REFERENCE_REPAIR_REPAIRS, TOLL_PAYMENT_REFERENCE_REPAIR_PHASE_HELPER, TOLL_PAYMENT_REFERENCE_REPAIR_PHASE_CALL_LIMIT),
    ("header_location_repair", HEADER_LOCATION_REPAIR_REPAIRS, HEADER_LOCATION_REPAIR_PHASE_HELPER, HEADER_LOCATION_REPAIR_PHASE_CALL_LIMIT),
    ("bag_item_ocr_repair", BAG_ITEM_OCR_REPAIR_REPAIRS, BAG_ITEM_OCR_REPAIR_PHASE_HELPER, BAG_ITEM_OCR_REPAIR_PHASE_CALL_LIMIT),
    ("discount_consistency_reconciliation", DISCOUNT_CONSISTENCY_RECONCILIATION_REPAIRS, DISCOUNT_CONSISTENCY_RECONCILIATION_PHASE_HELPER, DISCOUNT_CONSISTENCY_RECONCILIATION_PHASE_CALL_LIMIT),
    ("discounted_ocr_item_repair", DISCOUNTED_OCR_ITEM_REPAIR_REPAIRS, DISCOUNTED_OCR_ITEM_REPAIR_PHASE_HELPER, DISCOUNTED_OCR_ITEM_REPAIR_PHASE_CALL_LIMIT),
    ("basket_marker_rows", BASKET_MARKER_ROWS_REPAIRS, BASKET_MARKER_ROWS_PHASE_HELPER, BASKET_MARKER_ROWS_PHASE_CALL_LIMIT),
    ("quantity_detail_reconciliation", QUANTITY_DETAIL_RECONCILIATION_REPAIRS, QUANTITY_DETAIL_RECONCILIATION_PHASE_HELPER, QUANTITY_DETAIL_RECONCILIATION_PHASE_CALL_LIMIT),
    ("tax_category_assignment", TAX_CATEGORY_ASSIGNMENT_REPAIRS, TAX_CATEGORY_ASSIGNMENT_PHASE_HELPER, TAX_CATEGORY_ASSIGNMENT_PHASE_CALL_LIMIT),
    ("bag_item_rate_base_reconciliation", BAG_ITEM_RATE_BASE_RECONCILIATION_REPAIRS, BAG_ITEM_RATE_BASE_RECONCILIATION_PHASE_HELPER, BAG_ITEM_RATE_BASE_RECONCILIATION_PHASE_CALL_LIMIT),
    ("single_rate_inclusive_tax_restoration", SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_REPAIRS, SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_PHASE_HELPER, SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_PHASE_CALL_LIMIT),
    ("stacked_inclusive_tax_restoration", STACKED_INCLUSIVE_TAX_RESTORATION_REPAIRS, STACKED_INCLUSIVE_TAX_RESTORATION_PHASE_HELPER, STACKED_INCLUSIVE_TAX_RESTORATION_PHASE_CALL_LIMIT),
    ("tax_excluded_rate_block_restoration", TAX_EXCLUDED_RATE_BLOCK_RESTORATION_REPAIRS, TAX_EXCLUDED_RATE_BLOCK_RESTORATION_PHASE_HELPER, TAX_EXCLUDED_RATE_BLOCK_RESTORATION_PHASE_CALL_LIMIT),
    ("explicit_tax_amount_restoration", EXPLICIT_TAX_AMOUNT_RESTORATION_REPAIRS, EXPLICIT_TAX_AMOUNT_RESTORATION_PHASE_HELPER, EXPLICIT_TAX_AMOUNT_RESTORATION_PHASE_CALL_LIMIT),
    ("external_tax_total_restoration", EXTERNAL_TAX_TOTAL_RESTORATION_REPAIRS, EXTERNAL_TAX_TOTAL_RESTORATION_PHASE_HELPER, EXTERNAL_TAX_TOTAL_RESTORATION_PHASE_CALL_LIMIT),
    ("cash_tender_reconciliation", CASH_TENDER_RECONCILIATION_REPAIRS, CASH_TENDER_RECONCILIATION_PHASE_HELPER, CASH_TENDER_RECONCILIATION_PHASE_CALL_LIMIT),
    ("payment_method_repair", PAYMENT_METHOD_REPAIR_REPAIRS, PAYMENT_METHOD_REPAIR_PHASE_HELPER, PAYMENT_METHOD_REPAIR_PHASE_CALL_LIMIT),
    ("payment_points_reconciliation", PAYMENT_POINTS_RECONCILIATION_REPAIRS, PAYMENT_POINTS_RECONCILIATION_PHASE_HELPER, PAYMENT_POINTS_RECONCILIATION_PHASE_CALL_LIMIT),
    ("service_receipt_recovery", SERVICE_RECEIPT_RECOVERY_REPAIRS, SERVICE_RECEIPT_RECOVERY_PHASE_HELPER, SERVICE_RECEIPT_RECOVERY_PHASE_CALL_LIMIT),
    ("body_total_layout_reconstruction", BODY_TOTAL_LAYOUT_RECONSTRUCTION_REPAIRS, BODY_TOTAL_LAYOUT_RECONSTRUCTION_PHASE_HELPER, BODY_TOTAL_LAYOUT_RECONSTRUCTION_PHASE_CALL_LIMIT),
    ("ocr_description_reconciliation", OCR_DESCRIPTION_RECONCILIATION_REPAIRS, OCR_DESCRIPTION_RECONCILIATION_PHASE_HELPER, OCR_DESCRIPTION_RECONCILIATION_PHASE_CALL_LIMIT),
    ("gap_item_recovery", GAP_ITEM_RECOVERY_REPAIRS, GAP_ITEM_RECOVERY_PHASE_HELPER, GAP_ITEM_RECOVERY_PHASE_CALL_LIMIT),
    ("prefixed_tax_marker_item_rows", PREFIXED_TAX_MARKER_ITEM_ROWS_REPAIRS, PREFIXED_TAX_MARKER_ITEM_ROWS_PHASE_HELPER, PREFIXED_TAX_MARKER_ITEM_ROWS_PHASE_CALL_LIMIT),
    ("low_value_bag_recovery", LOW_VALUE_BAG_RECOVERY_REPAIRS, LOW_VALUE_BAG_RECOVERY_PHASE_HELPER, LOW_VALUE_BAG_RECOVERY_PHASE_CALL_LIMIT),
    ("item_name_price_cleanup", ITEM_NAME_PRICE_CLEANUP_REPAIRS, ITEM_NAME_PRICE_CLEANUP_PHASE_HELPER, ITEM_NAME_PRICE_CLEANUP_PHASE_CALL_LIMIT),
    ("priced_name_item_repair", PRICED_NAME_ITEM_REPAIR_REPAIRS, PRICED_NAME_ITEM_REPAIR_PHASE_HELPER, PRICED_NAME_ITEM_REPAIR_PHASE_CALL_LIMIT),
    ("digit_misread_item_repair", DIGIT_MISREAD_ITEM_REPAIR_REPAIRS, DIGIT_MISREAD_ITEM_REPAIR_PHASE_HELPER, DIGIT_MISREAD_ITEM_REPAIR_PHASE_CALL_LIMIT),
    ("subtotal_item_price_repair", SUBTOTAL_ITEM_PRICE_REPAIR_REPAIRS, SUBTOTAL_ITEM_PRICE_REPAIR_PHASE_HELPER, SUBTOTAL_ITEM_PRICE_REPAIR_PHASE_CALL_LIMIT),
    ("implausible_tax_amount_repair", IMPLAUSIBLE_TAX_AMOUNT_REPAIR_REPAIRS, IMPLAUSIBLE_TAX_AMOUNT_REPAIR_PHASE_HELPER, IMPLAUSIBLE_TAX_AMOUNT_REPAIR_PHASE_CALL_LIMIT),
    ("vertical_price_qty_total_projection", VERTICAL_PRICE_QTY_TOTAL_PROJECTION_REPAIRS, VERTICAL_PRICE_QTY_TOTAL_PROJECTION_PHASE_HELPER, VERTICAL_PRICE_QTY_TOTAL_PROJECTION_PHASE_CALL_LIMIT),
    ("stacked_name_price_projection", STACKED_NAME_PRICE_PROJECTION_REPAIRS, STACKED_NAME_PRICE_PROJECTION_PHASE_HELPER, STACKED_NAME_PRICE_PROJECTION_PHASE_CALL_LIMIT),
    ("single_item_quantity_repair", SINGLE_ITEM_QUANTITY_REPAIR_REPAIRS, SINGLE_ITEM_QUANTITY_REPAIR_PHASE_HELPER, SINGLE_ITEM_QUANTITY_REPAIR_PHASE_CALL_LIMIT),
    ("code_prefixed_description_cleanup", CODE_PREFIXED_DESCRIPTION_CLEANUP_REPAIRS, CODE_PREFIXED_DESCRIPTION_CLEANUP_PHASE_HELPER, CODE_PREFIXED_DESCRIPTION_CLEANUP_PHASE_CALL_LIMIT),
    ("adjacent_price_shift", ADJACENT_PRICE_SHIFT_REPAIRS, ADJACENT_PRICE_SHIFT_PHASE_HELPER, ADJACENT_PRICE_SHIFT_PHASE_CALL_LIMIT),
    ("bag_amount_shift", BAG_AMOUNT_SHIFT_REPAIRS, BAG_AMOUNT_SHIFT_PHASE_HELPER, BAG_AMOUNT_SHIFT_PHASE_CALL_LIMIT),
    ("financial_totals_repair", FINANCIAL_TOTALS_REPAIR_REPAIRS, FINANCIAL_TOTALS_REPAIR_PHASE_HELPER, FINANCIAL_TOTALS_REPAIR_PHASE_CALL_LIMIT),
    ("line_item_cleanup", LINE_ITEM_CLEANUP_REPAIRS, LINE_ITEM_CLEANUP_PHASE_HELPER, LINE_ITEM_CLEANUP_PHASE_CALL_LIMIT),
    ("phantom_tax_amount_cleanup", PHANTOM_TAX_AMOUNT_CLEANUP_REPAIRS, PHANTOM_TAX_AMOUNT_CLEANUP_PHASE_HELPER, PHANTOM_TAX_AMOUNT_CLEANUP_PHASE_CALL_LIMIT),
)


@pytest.mark.parametrize("label, repairs, helper_name", POSTPROCESS_PHASE_INVARIANT_CASES)
def test_postprocess_phase_is_named_and_invariant_backed(label, repairs, helper_name):
    tree = _parse_file(POSTPROCESS_PHASES_PATH)
    helper = _function_def(tree, helper_name)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(repairs - set(_call_names_in_function(helper)))
    assert not missing_repairs, (
        f"{label} repairs must be owned by the named {helper_name} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{helper_name} must document its structural trigger and invariant."
    )


def test_retired_structural_item_projection_phase_is_removed():
    tree = _parse_file(POSTPROCESS_PHASES_PATH)
    postprocess = _function_def(_parse_file(POSTPROCESS_PATH), "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)

    assert RETIRED_STRUCTURAL_ITEM_PROJECTION_PHASE_HELPER not in postprocess_calls
    assert not _has_function_def(tree, RETIRED_STRUCTURAL_ITEM_PROJECTION_PHASE_HELPER)


@pytest.mark.parametrize(
    "label, repairs, helper_name, call_limit", POSTPROCESS_PHASE_OWNERSHIP_CASES
)
def test_postprocess_phase_debt_is_phase_owned(label, repairs, helper_name, call_limit):
    postprocess = _function_def(_parse_file(POSTPROCESS_PATH), "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_calls = [name for name in postprocess_calls if name in repairs]
    phase_calls = [name for name in postprocess_calls if name == helper_name]

    assert not direct_calls, (
        f"{label} repairs should run through {helper_name}.\n"
        f"Direct calls still in postprocess_receipt: {direct_calls}"
    )
    assert 0 < len(phase_calls) <= call_limit, (
        f"{label} phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; limit: {call_limit}"
    )





















def test_final_campaign_discount_projection_debt_is_phase_owned():
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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






def test_final_body_total_layout_reconstruction_helper_is_named_and_invariant_backed():
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(POSTPROCESS_PHASES_PATH)
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




def test_final_printed_item_sum_total_repair_helper_is_named_and_invariant_backed():
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(POSTPROCESS_PHASES_PATH)
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




def test_final_cash_tender_reconciliation_helper_is_named_and_invariant_backed():
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(POSTPROCESS_PHASES_PATH)
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




def test_final_bare_number_tax_summary_restoration_helper_is_named_and_invariant_backed():
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(POSTPROCESS_PHASES_PATH)
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




def test_final_small_target_only_tax_pruning_helper_is_named_and_invariant_backed():
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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






def test_final_following_ocr_price_projection_helper_is_named_and_invariant_backed():
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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


























def test_final_merchant_identity_repair_debt_is_removed():
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
    receipt_tree = _parse_file(POSTPROCESS_PHASES_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
    receipt_tree = _parse_file(POSTPROCESS_PHASES_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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






def test_final_quantity_detail_reconciliation_helper_is_named_and_invariant_backed():
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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






def test_final_adjacent_price_shift_reconciliation_helper_is_named_and_invariant_backed():
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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


















def test_final_bag_item_rate_base_reconciliation_helper_is_named_and_invariant_backed():
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
    tree = _parse_file(FINAL_OUTPUT_PATH)
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
