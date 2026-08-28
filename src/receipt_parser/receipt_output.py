"""Post-serialization receipt output repairs."""

from typing import Callable

from .receipt_financial import (
    extract_financial_totals,
    extract_rate_bases,
    reconcile_points_payment_from_ocr,
)
from .receipt_identity_payment import (
    _fix_company_name_merchant,
    _fix_unlabeled_cash_tender_change_block,
)
from .receipt_item_cleanup import (
    _fix_adjacent_ocr_price_shift_when_balanced,
    _repair_discounted_line_item_totals_when_balanced,
    _repair_discounted_ocr_pair_descriptions,
    _repair_pre_price_stack_descriptions_from_ocr,
    _replace_basket_marker_rows_when_balanced,
)
from .receipt_item_repair import (
    _fix_code_table_descriptions_by_order,
    _fix_split_item_price_body_total_layout,
)
from .receipt_items import (
    _fix_bag_item_prices_from_rate_bases,
)
from .receipt_late_repairs import (
    _recover_labeled_purchase_site_location,
    _replace_stacked_name_price_rows_when_balanced,
    _restore_single_rate_inclusive_tax_block,
    _restore_stacked_inclusive_tax_block,
)
from .receipt_marker_projection import (
    _fix_qty_totals_from_ocr_unit_lines,
    _replace_jan_pos_items_when_balanced,
    _replace_prefixed_tax_marker_item_rows_when_balanced,
)
from .receipt_postprocess_phases import (
    _reconcile_tax_labels_from_item_arithmetic,
    _run_campaign_discount_projection_phase,
)
from .receipt_projection import (
    _clear_discount_when_negative_line_precedes_own_price,
    _project_totals_to_layout_rows,
)
from .receipt_recovery import (
    _apply_coupon_discount_blocks,
    _drop_applied_coupon_line_items,
    _fix_item_totals_from_following_discount_lines,
    _recover_missing_items_from_gap,
    _repair_tiny_item_prices_from_following_ocr,
    _replace_split_price_block_when_balanced,
    _restore_printed_external_tax_amounts,
)
from .receipt_row_projection import (
    _replace_barcode_qty_price_rows_when_balanced,
    _replace_barcode_unit_qty_amount_stack_when_balanced,
    _replace_dense_sequence_rows_when_balanced,
    _replace_item_price_qty_rows_when_balanced,
)
from .receipt_tax_categories import (
    reconcile_single_rate_tax_entries_from_item_arithmetic,
    reconcile_tax_categories_from_rate_bases,
)
from .receipt_totals import (
    _drop_unprinted_small_target_only_taxes,
    _prefer_printed_item_sum_total_when_balanced,
    _restore_bare_number_tax_summary,
    _restore_external_tax_total_from_printed_subtotal,
    _restore_printed_summary_total_when_tax_balanced,
)
from .receipt_phase_trace import (
    POSTPROCESS_PHASE_BY_NAME,
    _record_receipt_mutation,
    _snapshot_receipt_mutation_fields,
)
from .receipt_location import (
    _normalize_noisy_city_location,
    _recover_ascii_brand_header_location,
    _recover_header_branch_store_location,
)


def _assert_repair_within_owner_writes(
    mutation_trace: list[dict] | None,
    trace_len: int,
    owner_phase: str,
) -> None:
    if mutation_trace is None or len(mutation_trace) <= trace_len:
        return
    undeclared = set(mutation_trace[-1]["changes"]) - set(
        POSTPROCESS_PHASE_BY_NAME[owner_phase]["writes"]
    )
    if undeclared:
        raise AssertionError(
            f"Receipt output owner {owner_phase!r} mutated undeclared fields: "
            f"{sorted(undeclared)}"
        )


def _record_final_receipt_output_repair(
    stage: str,
    result: dict,
    mutation_trace: list[dict] | None,
    repair: Callable[[], None],
) -> None:
    before = (
        _snapshot_receipt_mutation_fields(result)
        if mutation_trace is not None
        else None
    )
    repair()
    trace_len = len(mutation_trace) if mutation_trace is not None else 0
    _record_receipt_mutation(mutation_trace, stage, before, result)
    if mutation_trace is not None and len(mutation_trace) > trace_len:
        owner_phase, justification = FINAL_RECEIPT_OUTPUT_REPAIR_JUSTIFICATIONS[stage]
        _assert_repair_within_owner_writes(mutation_trace, trace_len, owner_phase)
        mutation_trace[-1]["owner_phase"] = owner_phase
        mutation_trace[-1]["owner_invariant"] = POSTPROCESS_PHASE_BY_NAME[owner_phase][
            "invariant"
        ]
        mutation_trace[-1]["justification"] = justification


RECEIPT_OUTPUT_REPAIR_JUSTIFICATIONS = {
    "receipt_output_merchant_identity": (
        "header_identity_repair",
        "Owned by the receipt output merchant identity phase after Receipt serialization can reintroduce parent-company merchant text.",
    ),
}


def _record_receipt_output_repair(
    stage: str,
    result: dict,
    mutation_trace: list[dict] | None,
    repair: Callable[[], None],
) -> None:
    before = (
        _snapshot_receipt_mutation_fields(result)
        if mutation_trace is not None
        else None
    )
    repair()
    trace_len = len(mutation_trace) if mutation_trace is not None else 0
    _record_receipt_mutation(mutation_trace, stage, before, result)
    if mutation_trace is not None and len(mutation_trace) > trace_len:
        owner_phase, justification = RECEIPT_OUTPUT_REPAIR_JUSTIFICATIONS[stage]
        _assert_repair_within_owner_writes(mutation_trace, trace_len, owner_phase)
        mutation_trace[-1]["owner_phase"] = owner_phase
        mutation_trace[-1]["owner_invariant"] = POSTPROCESS_PHASE_BY_NAME[owner_phase][
            "invariant"
        ]
        mutation_trace[-1]["justification"] = justification


def _run_receipt_output_merchant_identity_phase(
    result: dict,
    ocr_text: str | None,
) -> None:
    """Trigger: serialized receipt output has OCR header merchant evidence.

    Invariant: merchant changes must be backed by visible header text and keep
    the merchant field consistent with valid receipt identity candidates.
    """
    if result.get("document_type") == "receipt" and ocr_text:
        _fix_company_name_merchant(result, ocr_text)


def _run_final_header_location_repair_phase(
    result: dict,
    ocr_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: OCR header, branch/address, purchase-site, or phone/contact rows.

    Invariant: location changes must preserve visible header/address evidence
    and prefer the most specific printed branch or city token over noisy text.
    """
    for repair in repairs:
        if repair == "labeled_purchase_site_location":
            _recover_labeled_purchase_site_location(result, ocr_text)
        elif repair == "header_branch_store_location":
            _recover_header_branch_store_location(result, ocr_text)
        elif repair == "ascii_brand_header_location":
            _recover_ascii_brand_header_location(result, ocr_text)
        elif repair == "noisy_city_location":
            _normalize_noisy_city_location(result, ocr_text)
        else:
            raise ValueError(f"Unknown final header location repair: {repair}")


def _run_final_printed_summary_total_tax_repair_phase(
    result: dict,
    ocr_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: visible printed summary total and tax-balance rows.

    Invariant: total/subtotal changes must be backed by printed summary
    evidence and preserve total, tax, payment, and points arithmetic.
    """
    for repair in repairs:
        if repair in {
            "printed_summary_total_tax_balanced",
            "printed_summary_total_tax_balanced_2",
        }:
            _restore_printed_summary_total_when_tax_balanced(result, ocr_text)
        else:
            raise ValueError(
                "Unknown final printed summary total repair: "
                f"{repair}"
            )


def _run_final_tax_category_reconciliation_phase(
    result: dict,
    ocr_text: str,
    repairs: tuple[str, ...],
    *,
    ocr_layout_blocks=None,
) -> None:
    """Trigger: printed rate evidence or stale labels after final item repairs.

    Invariant: category changes preserve per-item totals, while a label changes
    only when final item-plus-tax arithmetic proves external tax and no stronger
    inclusive structure is printed.
    """
    for repair in repairs:
        if repair == "tax_categories_from_rate_bases":
            reconcile_tax_categories_from_rate_bases(
                result,
                ocr_text,
                ocr_layout_blocks=ocr_layout_blocks,
            )
            reconcile_single_rate_tax_entries_from_item_arithmetic(result, ocr_text)
            _reconcile_tax_labels_from_item_arithmetic(result, ocr_text)
        else:
            raise ValueError(
                "Unknown final tax category reconciliation repair: "
                f"{repair}"
            )


def _run_final_external_tax_total_restoration_phase(
    result: dict,
    ocr_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: printed subtotal and external tax rows after item repairs.

    Invariant: total changes must preserve subtotal plus external tax
    arithmetic from the printed summary.
    """
    for repair in repairs:
        if repair in {
            "external_tax_total_from_printed_subtotal",
            "external_tax_total_from_printed_subtotal_final",
        }:
            _restore_external_tax_total_from_printed_subtotal(result, ocr_text)
        else:
            raise ValueError(
                "Unknown final external tax total restoration repair: "
                f"{repair}"
            )


def _run_final_coupon_discount_projection_phase(
    result: dict,
    ocr_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: OCR-visible following discount, coupon, or CPN rows.

    Invariant: item totals must equal gross minus OCR-visible discount, or
    adjusted discounted totals must balance the printed subtotal.
    """
    for repair in repairs:
        if repair in {
            "following_discount_lines",
            "following_discount_lines_after_layout",
        }:
            _fix_item_totals_from_following_discount_lines(result, ocr_text)
        elif repair == "coupon_discount_blocks":
            _apply_coupon_discount_blocks(result, ocr_text)
        elif repair == "drop_applied_coupon_line_items":
            _drop_applied_coupon_line_items(result, ocr_text)
        elif repair == "discounted_line_item_totals":
            _repair_discounted_line_item_totals_when_balanced(result, ocr_text)
        else:
            raise ValueError(
                "Unknown final coupon/discount projection repair: "
                f"{repair}"
            )


def _run_final_ocr_description_reconciliation_phase(
    result: dict,
    ocr_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: OCR item-code, JAN, discount-pair, or pre-price context.

    Invariant: description changes must remain backed by visible OCR
    neighbors while preserving each item's amount and quantity fields.
    """
    for repair in repairs:
        if repair == "code_table_descriptions":
            _fix_code_table_descriptions_by_order(result, ocr_text)
        elif repair == "discounted_ocr_pair_descriptions":
            _repair_discounted_ocr_pair_descriptions(result, ocr_text)
        elif repair == "pre_price_stack_descriptions":
            _repair_pre_price_stack_descriptions_from_ocr(result, ocr_text)
        else:
            raise ValueError(
                "Unknown final OCR description reconciliation repair: "
                f"{repair}"
            )


def _run_final_adjacent_price_shift_reconciliation_phase(
    result: dict,
    ocr_text: str,
    repairs: tuple[str, ...],
) -> None:
    """Trigger: adjacent OCR item and price rows after row projection.

    Invariant: shifted price changes must preserve item totals that balance
    against the printed subtotal.
    """
    for repair in repairs:
        if repair in {
            "adjacent_ocr_price_shift",
            "adjacent_ocr_price_shift_final",
        }:
            _fix_adjacent_ocr_price_shift_when_balanced(result, ocr_text)
        else:
            raise ValueError(
                "Unknown final adjacent price-shift reconciliation repair: "
                f"{repair}"
            )


FINAL_RECEIPT_OUTPUT_REPAIR_JUSTIFICATIONS = {
    "barcode_unit_qty_amount_stack": (
        "structural_item_reconstruction",
        "Owned by the final structural item-projection helper until this barcode stack projection no longer needs post-serialization repair.",
    ),
    "barcode_qty_price_rows": (
        "structural_item_reconstruction",
        "Owned by the final barcode quantity-price projection helper until barcode row projection moves out of post-serialization repair.",
    ),
    "item_price_qty_rows": (
        "structural_item_reconstruction",
        "Owned by the final item price/quantity projection helper until row projection moves out of post-serialization repair.",
    ),
    "labeled_purchase_site_location": (
        "header_identity_repair",
        "Owned by the final header location repair helper until purchase-site location recovery moves out of post-serialization repair.",
    ),
    "header_branch_store_location": (
        "header_identity_repair",
        "Owned by the final header location repair helper until branch recovery moves out of post-serialization repair.",
    ),
    "noisy_city_location": (
        "header_identity_repair",
        "Owned by the final header location repair helper until noisy city cleanup moves out of post-serialization repair.",
    ),
    "single_rate_inclusive_tax_block": (
        "tax_category_assignment",
        "Owned by the final single-rate inclusive tax restoration helper until this serialized tax block repair moves out of post-serialization repair.",
    ),
    "coupon_discount_projection": (
        "structural_item_reconstruction",
        "Owned by the final coupon/discount projection helper until "
        "OCR-visible coupon and following-discount rows move out of "
        "post-serialization repair.",
    ),
    "tiny_item_prices_from_following_ocr": (
        "structural_item_reconstruction",
        "Owned by the final following-OCR price projection helper until repeated following amount projection moves out of post-serialization repair.",
    ),
    "split_price_block": (
        "structural_item_reconstruction",
        "Owned by the final split price block projection helper until split price blocks move out of post-serialization repair.",
    ),
    "split_item_price_body_total": (
        "structural_item_reconstruction",
        "Owned by the final body-total layout reconstruction helper until body-total split layouts move out of post-serialization repair.",
    ),
    "stacked_name_price_rows": (
        "structural_item_reconstruction",
        "Owned by the final stacked name/price projection helper until stacked row projection moves out of post-serialization repair.",
    ),
    "stacked_inclusive_tax_block": (
        "tax_category_assignment",
        "Owned by the final stacked inclusive tax restoration helper until stacked tax summaries move out of post-serialization repair.",
    ),
    "printed_summary_total_tax_balanced": (
        "financial_totals_repair",
        "Owned by the final printed summary total/tax repair helper until printed summary total correction moves out of post-serialization repair.",
    ),
    "printed_item_sum_total": (
        "financial_totals_repair",
        "Owned by the final printed item-sum total helper until this printed total correction moves out of post-serialization repair.",
    ),
    "adjacent_price_shift_reconciliation": (
        "structural_item_reconstruction",
        "Owned by the final adjacent price-shift reconciliation helper until "
        "adjacent OCR price repairs move out of post-serialization repair.",
    ),
    "dense_sequence_rows": (
        "structural_item_reconstruction",
        "Owned by the final dense sequence projection helper until dense sequence projection moves out of post-serialization repair.",
    ),
    "campaign_discount_stream": (
        "structural_item_reconstruction",
        "Owned by the campaign discount projection phase before final discount cleanup.",
    ),
    "jan_pos_items": (
        "structural_item_reconstruction",
        "Owned by the final JAN/POS item projection helper until JAN/POS "
        "row projection moves out of post-serialization repair.",
    ),
    "qty_totals_from_unit_lines": (
        "quantity_detail_reconciliation",
        "Owned by the final quantity-detail reconciliation helper until OCR unit-line quantity repair moves out of post-serialization repair.",
    ),
    "bag_item_prices_from_rate_bases": (
        "bag_item_rate_base_reconciliation",
        "Owned by the final bag item rate-base reconciliation helper until "
        "paid-bag price/rate-base repair moves out of post-serialization "
        "repair.",
    ),
    "code_table_description_reconciliation": (
        "item_cleanup",
        "Owned by the final OCR description reconciliation helper until "
        "code-table description cleanup moves out of post-serialization "
        "repair.",
    ),
    "printed_external_tax_amounts": (
        "printed_external_tax_amount_restoration",
        "Owned by the final printed external-tax amount restoration helper "
        "until printed per-rate tax amount recovery moves out of "
        "post-serialization repair.",
    ),
    "bare_number_tax_summary": (
        "bare_number_tax_summary_restoration",
        "Owned by the final bare-number tax summary restoration helper until "
        "numeric tax-summary stack recovery moves out of post-serialization "
        "repair.",
    ),
    "external_tax_total_from_printed_subtotal": (
        "financial_totals_repair",
        "Owned by the final external tax total restoration helper until "
        "printed subtotal/tax total repair moves out of post-serialization "
        "repair.",
    ),
    "drop_small_target_only_taxes": (
        "small_target_only_tax_pruning",
        "Owned by the final small target-only tax pruning helper until "
        "unprinted target-only tax cleanup moves out of post-serialization "
        "repair.",
    ),
    "printed_summary_total_tax_balanced_2": (
        "financial_totals_repair",
        "Owned by the final printed summary total/tax repair helper; repeated after later tax repairs can change total balance.",
    ),
    "unlabeled_cash_tender_change": (
        "payment_points_reconciliation",
        "Owned by the final cash tender reconciliation helper until cash "
        "tender/change repair moves out of post-serialization repair.",
    ),
    "points_payment": (
        "payment_points_reconciliation",
        "Owned by the final payment/points reconciliation helper until "
        "points payment repair moves out of post-serialization repair.",
    ),
    "clear_discount_before_own_price": (
        "discount_consistency_reconciliation",
        "Owned by the final discount consistency reconciliation helper until "
        "negative discount placement cleanup moves out of post-serialization "
        "repair.",
    ),
    "campaign_discount_stream_2": (
        "structural_item_reconstruction",
        "Temporary debt: repeated campaign discount projection phase after discount cleanup can expose balanced streams.",
    ),
    "coupon_discount_projection_after_layout": (
        "structural_item_reconstruction",
        "Owned by the final coupon/discount projection helper as the bounded "
        "post-layout reassertion of discount arithmetic.",
    ),
    "adjacent_price_shift_reconciliation_after_layout": (
        "structural_item_reconstruction",
        "Owned by the final adjacent price-shift reconciliation helper as the "
        "bounded post-layout reassertion after discount cleanup.",
    ),
    "prefixed_tax_marker_item_rows": (
        "structural_item_reconstruction",
        "Owned by the final prefixed tax-marker item row helper until marker-prefixed row projection moves out of post-serialization repair.",
    ),
    "missing_items_from_gap": (
        "initial_item_recovery",
        "Owned by the final gap item recovery helper until missing OCR row-gap recovery moves out of post-serialization repair.",
    ),
    "ocr_description_reconciliation_after_layout": (
        "item_cleanup",
        "Owned by the final OCR description reconciliation helper as the "
        "bounded post-layout pass for discount-pair and pre-price stack "
        "description context.",
    ),
    "basket_marker_rows": (
        "structural_item_reconstruction",
        "Owned by the final basket-marker row helper until explicit basket marker projection moves out of post-serialization repair.",
    ),
    "layout_row_price_permutation": (
        "item_cleanup",
        "Owned by the final layout projection after late item recovery or description cleanup exposes uniquely matched rows whose totals can only be permuted.",
    ),
    "tax_categories_from_rate_bases": (
        "tax_category_assignment",
        "Owned by the final tax category reconciliation helper until rate-base "
        "category repair moves out of post-serialization repair.",
    ),
    "external_tax_total_from_printed_subtotal_final": (
        "financial_totals_repair",
        "Owned by the final external tax total restoration helper as the "
        "bounded reassertion after item/tax category repairs.",
    ),
}


def _apply_final_receipt_output_repairs(
    result: dict,
    ocr_text: str | None,
    mutation_trace: list[dict] | None = None,
    *,
    ocr_layout_blocks=None,
) -> None:
    """Apply legacy receipt repairs that still run after model validation."""
    if result.get("document_type") != "receipt" or not ocr_text:
        return

    def run(stage: str, repair: Callable[[], None]) -> None:
        _record_final_receipt_output_repair(stage, result, mutation_trace, repair)

    run(
        "barcode_unit_qty_amount_stack",
        lambda: _replace_barcode_unit_qty_amount_stack_when_balanced(
            result, ocr_text
        ),
    )
    run(
        "barcode_qty_price_rows",
        lambda: _replace_barcode_qty_price_rows_when_balanced(result, ocr_text),
    )
    run(
        "item_price_qty_rows",
        lambda: _replace_item_price_qty_rows_when_balanced(result, ocr_text),
    )
    run(
        "labeled_purchase_site_location",
        lambda: _run_final_header_location_repair_phase(
            result,
            ocr_text,
            ("labeled_purchase_site_location",),
        ),
    )
    run(
        "header_branch_store_location",
        lambda: _run_final_header_location_repair_phase(
            result,
            ocr_text,
            ("header_branch_store_location", "ascii_brand_header_location"),
        ),
    )
    run(
        "noisy_city_location",
        lambda: _run_final_header_location_repair_phase(
            result,
            ocr_text,
            ("noisy_city_location",),
        ),
    )
    run(
        "single_rate_inclusive_tax_block",
        lambda: _restore_single_rate_inclusive_tax_block(result, ocr_text),
    )
    run(
        "coupon_discount_projection",
        lambda: _run_final_coupon_discount_projection_phase(
            result,
            ocr_text,
            (
                "following_discount_lines",
                "coupon_discount_blocks",
                "drop_applied_coupon_line_items",
            ),
        ),
    )
    run(
        "tiny_item_prices_from_following_ocr",
        lambda: _repair_tiny_item_prices_from_following_ocr(result, ocr_text),
    )
    run(
        "split_price_block",
        lambda: _replace_split_price_block_when_balanced(result, ocr_text),
    )
    run(
        "split_item_price_body_total",
        lambda: _fix_split_item_price_body_total_layout(result, ocr_text),
    )
    run(
        "stacked_name_price_rows",
        lambda: _replace_stacked_name_price_rows_when_balanced(result, ocr_text),
    )
    run(
        "stacked_inclusive_tax_block",
        lambda: _restore_stacked_inclusive_tax_block(result, ocr_text),
    )
    run(
        "printed_summary_total_tax_balanced",
        lambda: _run_final_printed_summary_total_tax_repair_phase(
            result,
            ocr_text,
            ("printed_summary_total_tax_balanced",),
        ),
    )
    run(
        "printed_item_sum_total",
        lambda: _prefer_printed_item_sum_total_when_balanced(result, ocr_text),
    )
    run(
        "adjacent_price_shift_reconciliation",
        lambda: _run_final_adjacent_price_shift_reconciliation_phase(
            result,
            ocr_text,
            ("adjacent_ocr_price_shift",),
        ),
    )
    run(
        "dense_sequence_rows",
        lambda: _replace_dense_sequence_rows_when_balanced(result, ocr_text),
    )
    run(
        "campaign_discount_stream",
        lambda: _run_campaign_discount_projection_phase(
            result,
            ocr_text,
            ("campaign_discount_stream",),
        ),
    )
    run(
        "jan_pos_items",
        lambda: _replace_jan_pos_items_when_balanced(
            result,
            ocr_text,
            extract_financial_totals(ocr_text),
        ),
    )
    run(
        "qty_totals_from_unit_lines",
        lambda: _fix_qty_totals_from_ocr_unit_lines(result, ocr_text),
    )
    run(
        "bag_item_prices_from_rate_bases",
        lambda: _fix_bag_item_prices_from_rate_bases(
            result,
            extract_rate_bases(ocr_text),
            ocr_text,
        ),
    )
    run(
        "code_table_description_reconciliation",
        lambda: _run_final_ocr_description_reconciliation_phase(
            result,
            ocr_text,
            ("code_table_descriptions",),
        ),
    )
    run(
        "printed_external_tax_amounts",
        lambda: _restore_printed_external_tax_amounts(result, ocr_text),
    )
    run(
        "bare_number_tax_summary",
        lambda: _restore_bare_number_tax_summary(result, ocr_text),
    )
    run(
        "external_tax_total_from_printed_subtotal",
        lambda: _run_final_external_tax_total_restoration_phase(
            result,
            ocr_text,
            ("external_tax_total_from_printed_subtotal",),
        ),
    )
    run(
        "drop_small_target_only_taxes",
        lambda: _drop_unprinted_small_target_only_taxes(result, ocr_text),
    )
    run(
        "printed_summary_total_tax_balanced_2",
        lambda: _run_final_printed_summary_total_tax_repair_phase(
            result,
            ocr_text,
            ("printed_summary_total_tax_balanced_2",),
        ),
    )
    run(
        "unlabeled_cash_tender_change",
        lambda: _fix_unlabeled_cash_tender_change_block(result, ocr_text),
    )
    run(
        "points_payment",
        lambda: reconcile_points_payment_from_ocr(result, ocr_text),
    )
    run(
        "clear_discount_before_own_price",
        lambda: _clear_discount_when_negative_line_precedes_own_price(
            result, ocr_text
        ),
    )
    run(
        "campaign_discount_stream_2",
        lambda: _run_campaign_discount_projection_phase(
            result,
            ocr_text,
            ("campaign_discount_stream",),
        ),
    )
    run(
        "coupon_discount_projection_after_layout",
        lambda: _run_final_coupon_discount_projection_phase(
            result,
            ocr_text,
            (
                "following_discount_lines_after_layout",
                "discounted_line_item_totals",
            ),
        ),
    )
    run(
        "adjacent_price_shift_reconciliation_after_layout",
        lambda: _run_final_adjacent_price_shift_reconciliation_phase(
            result,
            ocr_text,
            ("adjacent_ocr_price_shift_final",),
        ),
    )
    run(
        "prefixed_tax_marker_item_rows",
        lambda: _replace_prefixed_tax_marker_item_rows_when_balanced(
            result, ocr_text
        ),
    )
    run(
        "missing_items_from_gap",
        lambda: _recover_missing_items_from_gap(result, ocr_text),
    )
    run(
        "ocr_description_reconciliation_after_layout",
        lambda: _run_final_ocr_description_reconciliation_phase(
            result,
            ocr_text,
            (
                "discounted_ocr_pair_descriptions",
                "pre_price_stack_descriptions",
            ),
        ),
    )
    run(
        "basket_marker_rows",
        lambda: _replace_basket_marker_rows_when_balanced(result, ocr_text),
    )
    run(
        "layout_row_price_permutation",
        lambda: _project_totals_to_layout_rows(result, ocr_layout_blocks),
    )
    run(
        "tax_categories_from_rate_bases",
        lambda: _run_final_tax_category_reconciliation_phase(
            result,
            ocr_text,
            ("tax_categories_from_rate_bases",),
            ocr_layout_blocks=ocr_layout_blocks,
        ),
    )
    run(
        "external_tax_total_from_printed_subtotal_final",
        lambda: _run_final_external_tax_total_restoration_phase(
            result,
            ocr_text,
            ("external_tax_total_from_printed_subtotal_final",),
        ),
    )


def _prepare_receipt_output_payload(
    receipt,
    ocr_text: str | None = None,
    mutation_trace: list[dict] | None = None,
    *,
    ocr_layout_blocks=None,
) -> dict:
    result = receipt.model_dump()
    _record_receipt_output_repair(
        "receipt_output_merchant_identity",
        result,
        mutation_trace,
        lambda: _run_receipt_output_merchant_identity_phase(result, ocr_text),
    )
    _apply_final_receipt_output_repairs(
        result,
        ocr_text,
        mutation_trace=mutation_trace,
        ocr_layout_blocks=ocr_layout_blocks,
    )
    return result
