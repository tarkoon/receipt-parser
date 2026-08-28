"""Focused contracts for accuracy checks that must preserve line-item identity."""

import pytest

from receipt_parser.checks import (
    _match_line_items,
    check_account_number,
    check_amount_paid,
    check_billing_period,
    check_canonical_keys,
    check_item_descriptions,
    check_line_items_discount_rates,
    check_line_items_discounts,
    check_line_items_qty,
    check_line_items_totals,
    check_line_items_unit_price,
    check_location,
    check_merchant_similarity,
    check_payer,
    check_payment_reference,
    check_points_used,
    check_subtotal,
    check_tax_amount,
    check_tax_categories,
    check_tax_labels,
    check_tax_rates,
    check_time,
    check_tree_edit_distance,
    check_usage_amount,
    check_usage_cost_per,
    check_usage_meter_current,
    check_usage_meter_previous,
    check_usage_unit,
    get_checks_for,
)


def _swapped_rows():
    truth = {
        "line_items": [
            {
                "description": "商品A",
                "qty": 1,
                "unit_price": 100,
                "total": 100,
                "tax_category": "8%",
                "discount": 0,
            },
            {
                "description": "商品B",
                "qty": 2,
                "unit_price": 200,
                "total": 390,
                "tax_category": "10%",
                "discount": 10,
            },
        ]
    }
    result = {
        "line_items": [
            {**truth["line_items"][1], "description": "商品A"},
            {**truth["line_items"][0], "description": "商品B"},
        ]
    }
    return result, truth


def test_line_item_checks_reject_values_swapped_between_descriptions():
    result, truth = _swapped_rows()

    checks = (
        check_line_items_qty,
        check_line_items_unit_price,
        check_line_items_totals,
        check_tax_categories,
        check_line_items_discounts,
    )

    assert all(not check(result, truth)["pass"] for check in checks)


def test_description_check_cannot_reuse_one_prediction_for_two_truth_rows():
    truth = {
        "line_items": [
            {"description": "apple juice"},
            {"description": "apple jam"},
        ]
    }
    result = {"line_items": [{"description": "apple"}]}

    checked = check_item_descriptions(result, truth)

    assert not checked["pass"]
    assert "1/2 matched one-to-one" in checked["detail"]


def test_line_item_matching_maximizes_description_similarity():
    truth = {"line_items": [
        {"description": "aaaaabbbbb"},
        {"description": "aaaaaddddd"},
    ]}
    result = {"line_items": [
        {"description": "aaaaabbbbb"},
        {"description": "aaaaaccccc"},
    ]}

    *_, pairs, unmatched_truth, unmatched_result = _match_line_items(result, truth)

    assert [(truth_idx, result_idx) for truth_idx, result_idx, _ in pairs] == [
        (0, 0),
        (1, 1),
    ]
    assert not unmatched_truth
    assert not unmatched_result


def test_duplicate_descriptions_match_as_complete_rows_not_by_position():
    truth = {"line_items": [
        {"description": "same item", "qty": 1, "unit_price": 100, "total": 90,
         "tax_category": "8%", "discount": 10},
        {"description": "same item", "qty": 2, "unit_price": 200, "total": 390,
         "tax_category": "10%", "discount": 10},
    ]}
    result = {"line_items": list(reversed(truth["line_items"]))}

    checks = (
        check_line_items_qty,
        check_line_items_unit_price,
        check_line_items_totals,
        check_tax_categories,
        check_line_items_discounts,
    )

    assert all(check(result, truth)["pass"] for check in checks)


def test_discount_is_a_direct_receipt_check_and_null_equals_no_discount():
    truth = {"document_type": "receipt", "line_items": [
        {"description": "item", "discount": None},
    ]}
    result = {"line_items": [{"description": "item", "discount": 0}]}

    assert "line_items_discounts" in get_checks_for(truth)
    assert check_line_items_discounts(result, truth)["pass"]


def test_null_subtotal_requires_null_result():
    truth = {"subtotal": None}

    assert check_subtotal({"subtotal": None, "total": 1040}, truth)["pass"]
    assert not check_subtotal({"subtotal": 1040, "total": 1040}, truth)["pass"]


def test_empty_tax_truth_rejects_phantom_zero_tax_entry():
    truth = {"taxes": []}
    result = {"taxes": [{"rate": "10%", "label": "\u5185\u7a0e", "amount": 0}]}

    assert not check_tax_rates(result, truth)["pass"]
    assert not check_tax_labels(result, truth)["pass"]


def test_tax_amounts_and_labels_stay_attached_to_their_rates():
    truth = {"taxes": [
        {"rate": "8%", "label": "内税", "amount": 80},
        {"rate": "10%", "label": "外税", "amount": 100},
    ]}
    result = {"taxes": [
        {"rate": "8%", "label": "外税", "amount": 100},
        {"rate": "10%", "label": "内税", "amount": 80},
    ]}

    assert check_tax_rates(result, truth)["pass"]
    assert not check_tax_amount(result, truth)["pass"]
    assert not check_tax_labels(result, truth)["pass"]


def test_duplicate_rate_tax_rows_keep_each_label_with_its_amount():
    truth = {"taxes": [
        {"rate": "10%", "label": "内税", "amount": 80},
        {"rate": "10%", "label": "外税", "amount": 100},
    ]}
    result = {"taxes": [
        {"rate": "10%", "label": "内税", "amount": 100},
        {"rate": "10%", "label": "外税", "amount": 80},
    ]}

    assert check_tax_rates(result, truth)["pass"]
    assert check_tax_labels(result, truth)["pass"]
    assert not check_tax_amount(result, truth)["pass"]


def test_missing_canonical_key_is_not_equivalent_to_explicit_null():
    truth = {"merchant": None, "time": None}

    assert check_canonical_keys({"merchant": None, "time": None}, truth)["pass"]
    checked = check_canonical_keys({"merchant": None}, truth)
    assert not checked["pass"]
    assert "time" in checked["detail"]


def test_amount_paid_tolerance_includes_the_five_yen_boundary():
    truth = {"amount_paid": 100}

    assert check_amount_paid({"amount_paid": 105}, truth)["pass"]
    assert not check_amount_paid({"amount_paid": 105.01}, truth)["pass"]


@pytest.mark.parametrize(("check", "key", "hallucination"), [
    (check_merchant_similarity, "merchant", "invented merchant"),
    (check_time, "time", "12:34"),
    (check_amount_paid, "amount_paid", 100),
    (check_points_used, "points_used", 1),
    (check_payer, "payer", "invented payer"),
    (check_account_number, "account_number", "1234"),
    (check_payment_reference, "payment_reference", "reference"),
])
def test_explicit_null_scalar_rejects_hallucinated_value(
        check, key, hallucination):
    assert check({key: None}, {key: None})["pass"]
    assert not check({key: hallucination}, {key: None})["pass"]


@pytest.mark.parametrize(("check", "field", "value"), [
    (check_usage_amount, "amount", 12),
    (check_usage_unit, "unit", "kWh"),
    (check_usage_cost_per, "cost_per", 3.5),
    (check_usage_meter_previous, "meter_previous", 100),
    (check_usage_meter_current, "meter_current", 112),
])
def test_usage_fields_compare_values_and_reject_null_truth_hallucinations(
        check, field, value):
    assert check({"usage": {field: value}}, {"usage": {field: value}})["pass"]
    assert not check({"usage": {field: value}}, {"usage": {field: None}})["pass"]


def test_billing_period_checks_both_dates_and_accepts_semantic_null_shape():
    truth = {"billing_period": {"start": "2026-01-01", "end": "2026-01-31"}}

    assert check_billing_period(dict(truth), truth)["pass"]
    assert not check_billing_period(
        {"billing_period": {"start": "2026-01-01", "end": "2026-02-01"}},
        truth,
    )["pass"]
    assert check_billing_period(
        {"billing_period": None},
        {"billing_period": {"start": None, "end": None}},
    )["pass"]


def test_discount_rate_is_checked_on_its_matched_item_row():
    truth = {"line_items": [{
        "description": "item", "qty": 1, "unit_price": 100, "total": 80,
        "tax_category": "10%", "discount": 20, "discount_rate": "20%",
    }]}
    result = {"line_items": [{**truth["line_items"][0], "discount_rate": "10%"}]}

    assert not check_line_items_discount_rates(result, truth)["pass"]


def test_line_item_defaults_do_not_conflate_zero_and_null_values():
    truth = {"line_items": [{"description": "item", "qty": 0, "unit_price": None}]}
    result = {"line_items": [{"description": "item", "qty": 1, "unit_price": 0}]}

    assert not check_line_items_qty(result, truth)["pass"]
    assert not check_line_items_unit_price(result, truth)["pass"]


def test_every_document_type_checks_the_complete_canonical_contract():
    required = {
        "subtotal", "line_items_count", "line_items_discount_rates",
        "tax_amount", "points_used", "service_type", "billing_period",
        "usage_amount", "usage_unit", "usage_cost_per",
        "usage_meter_previous", "usage_meter_current", "payer",
        "payment_reference",
    }

    for document_type in ("receipt", "utility_bill", "payment_slip"):
        assert required <= get_checks_for({"document_type": document_type}).keys()
        assert "account_number" not in get_checks_for({"document_type": document_type})


def test_account_number_is_best_effort_and_excluded_from_tree_score():
    truth = {"document_type": "receipt", "account_number": "customer-123"}
    result = {"document_type": "receipt", "account_number": None}

    assert "account_number" not in get_checks_for(truth)
    assert check_tree_edit_distance(result, truth)["pass"]
    assert check_tree_edit_distance(result, truth)["score"] == 1


def test_documented_identity_similarity_fallbacks_are_retained():
    merchant = check_merchant_similarity({"merchant": "abxy"}, {"merchant": "abcd"})
    location = check_location({"location": "abxy"}, {"location": "abcd"})

    for checked in (merchant, location):
        assert checked["pass"]
        assert checked["similarity"] == 0.5


def test_unrelated_merchant_and_location_are_rejected():
    merchant = check_merchant_similarity({"merchant": "wxyz"}, {"merchant": "abcd"})
    location = check_location({"location": "wxyz"}, {"location": "abcd"})

    assert not merchant["pass"]
    assert not location["pass"]


def test_location_accepts_printed_branch_covering_coarser_city():
    checked = check_location(
        {"location": "中央ショッピングセンター青葉店"},
        {"location": "青葉市"},
    )

    assert checked["pass"]
    assert checked["core_coverage"] == 1


def test_location_accepts_detailed_locality_when_tokens_are_reordered():
    checked = check_location(
        {"location": "ひかり青葉店"},
        {"location": "青葉市ひかり"},
    )

    assert checked["pass"]
    assert checked["similarity"] < 0.6
    assert checked["core_coverage"] == 1


def test_location_rejects_incomplete_structural_component_despite_header_noise():
    checked = check_location(
        {"location": "BRAND 青葉店"},
        {"location": "青葉北丘店"},
    )

    assert not checked["pass"]
    assert checked["similarity"] < 0.6
    assert checked["terminal_similarity"] >= 0.5
    assert checked["core_coverage"] < 0.75


def test_location_accepts_brand_noise_when_complete_structural_core_is_present():
    checked = check_location(
        {"location": "BRAND 青葉北丘店"},
        {"location": "青葉北丘店"},
    )

    assert checked["pass"]
    assert checked["core_coverage"] == 1


@pytest.mark.parametrize("suffix", ["ショップ", "モール", "センター", "館", "駅", "IC", "倉庫店"])
def test_location_compares_terminal_facility_suffixes(suffix):
    expected = f"青葉{suffix}"
    got = f"HEADER {expected}"

    checked = check_location({"location": got}, {"location": expected})

    assert checked["pass"]
    assert checked["terminal_similarity"] == 1


def test_location_nfkc_normalizes_fullwidth_terminal_text():
    checked = check_location(
        {"location": "ＨＥＡＤＥＲ 青葉ＩＣ"},
        {"location": "青葉IC"},
    )

    assert checked["pass"]
    assert checked["terminal_similarity"] == 1


def test_location_rejects_broad_city_for_more_detailed_expected_locality():
    checked = check_location(
        {"location": "青葉市"},
        {"location": "青葉市ひかり"},
    )

    assert not checked["pass"]
    assert checked["core_coverage"] < 0.75


def test_location_rejects_unrelated_structured_values():
    checked = check_location(
        {"location": "西町モール"},
        {"location": "青葉市"},
    )

    assert not checked["pass"]


def test_tree_edit_distance_is_visible_but_report_only():
    checked = check_tree_edit_distance(
        {"total": 999, "merchant": "wrong", "date": "1999-01-01"},
        {"total": 100, "merchant": "right", "date": "2026-01-01"},
    )

    assert checked["pass"]
    assert checked["score"] < 0.5
    assert "report-only" in checked["detail"]


def test_low_similarity_payer_is_rejected_at_identity_threshold():
    checked = check_payer({"payer": "abxy"}, {"payer": "abcd"})

    assert not checked["pass"]
