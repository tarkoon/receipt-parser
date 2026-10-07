"""Structural contracts for printed receipt tax summaries."""

import pytest

from receipt_parser.receipt_financial import (
    _interleaved_rate_tax_summary_entries,
    extract_financial_totals,
    extract_rate_bases,
    normalize_tax_label,
    reconcile_points_payment_from_ocr,
)
from receipt_parser.receipt_items import _clear_unprinted_rate_only_tax_summary
from receipt_parser.receipt_identity_payment import _apply_financial_overrides
from receipt_parser.receipt_late_repairs import _restore_tax_excluded_per_rate_blocks
from receipt_parser.receipt_totals import (
    _restore_bare_number_tax_summary,
    _sum_taxable_amounts,
)


def _tax_amounts(extracted):
    return {
        tax["rate"]: tax["amount"]
        for tax in extracted.get("taxes", [])
        if tax.get("rate") != "0%" and tax.get("amount", 0) > 0
    }


def test_complete_short_mode_table_rows_preserve_literal_owners_and_reject_conflicts():
    from receipt_parser.receipt_projection import _collect_direct_summary_owners
    from receipt_parser.receipt_financial import _direct_rate_mode_components

    lines = ["(d外8%対象額¥2,000)", "d外8%¥160", "E外10.0%対象額¥300", "E外10%¥30",
             "(f内10%対象額¥1,100)", "(f内10%¥100)", "8%外¥160", "内税10%¥100"]
    blocks = []
    for row, line in enumerate(lines):
        blocks.extend(dict(text=char, x=i * 12, y=row * 30,
                           bbox=[[i * 12, row * 30], [i * 12 + 10, row * 30],
                                 [i * 12 + 10, row * 30 + 10], [i * 12, row * 30 + 10]])
                      for i, char in enumerate(line))
    owners = _collect_direct_summary_owners(blocks)["owners"]["rate_component"]
    assert [(o["rate"], o["mode"], o["kind"], o["value"]) for o in owners] == [
        ("8%", "外税", "base", 2000), ("8%", "外税", "tax", 160),
        ("10%", "外税", "base", 300), ("10%", "外税", "tax", 30),
        ("10%", "内税", "base", 1100), ("10%", "内税", "tax", 100),
        ("8%", "外税", "tax", 160), ("10%", "内税", "tax", 100)]
    assert owners[0]["mode_literal"] == "外" and owners[0]["table_label"] == "d"
    assert [blocks[i]["text"] for i in owners[0]["table_label_indices"]] == ["d"]
    assert "".join(blocks[i]["text"] for i in owners[0]["row_indices"]) == "d外8%対象額¥2,000"
    text = "\n".join(lines[:6])
    for source in (text, text.replace("¥", "\n¥"), text.replace("E", "Ｅ")):
        assert [(o["rate"], o["mode"], o["kind"], o["value"]) for o in
                _direct_rate_mode_components(source)] == [
                    (o["rate"], o["mode"], o["kind"], o["value"]) for o in owners[:6]]
        assert normalize_tax_label("内税", source, rate="10%", amount=30) == "外税"
        assert normalize_tax_label("外税", source, rate="10%", amount=100) == "内税"
    for invalid in ("注d外8%¥160", "dd外8%¥160", "d8%外¥160", "d外8%内¥160",
                    "(d外8%¥160", "d外8%¥160)", "d外8%160", "d外8%¥160¥20"):
        assert not _direct_rate_mode_components(invalid)
    for duplicate in ("d外10%¥30", "g外10%¥31"):
        assert normalize_tax_label("内税", text + "\n" + duplicate, rate="10%", amount=30) == "内税"


def test_printed_rate_mode_amount_owners_survive_wrong_item_rates_and_zero_tax_inner_group():
    from copy import deepcopy
    from receipt_parser.receipt_item_cleanup import _normalize_taxes
    from receipt_parser.receipt_postprocess_phases import _restore_tax_entries_from_item_rate_sums

    text = "8%外税対象額 ¥260\n8%外税 ¥20\n10%外税対象額 ¥150\n10%外税 ¥15\n10%内税対象額 ¥8\n10%内税 ¥0"
    taxes = [{"rate": "8%", "label": "外税", "amount": 20},
             {"rate": "10%", "label": "外税", "amount": 15}]
    receipt = {"subtotal": 418, "total": 453, "taxes": deepcopy(taxes),
               "line_items": [{"total": value, "tax_category": "8%"} for value in (150, 150, 72, 30)]
               + [{"total": 8, "tax_category": "10%"}]}
    before = deepcopy(receipt)
    split = text.replace(' ¥', '\n¥')
    fragmented = split.replace('8%外税\n¥20', '8%\n税\n¥20')
    for source in (text, split, fragmented, fragmented.replace('8%', '８％')):
        receipt = deepcopy(before)
        _restore_tax_entries_from_item_rate_sums(receipt, source, {"taxes": taxes}, {"8%": 260, "10%": 158})
        _normalize_taxes(receipt, source, {"taxes": taxes})
        assert receipt == before
        assert normalize_tax_label('内税', source, rate='8%', amount=20) == '外税'
    intrusion = fragmented.replace('8%\n税\n¥20', '別項目\n8%\n税\n¥20')
    assert normalize_tax_label('内税', intrusion, rate='8%', amount=20) == '内税'
    assert normalize_tax_label("内税", text, rate="10%", amount=15) == "外税"
    assert normalize_tax_label("外税", text, rate="10%", amount=0) == "内税"
    for ambiguous in (text + "\n10%外税 ¥15", text + "\n10%外税 ¥16",
                      text.replace("10%外税 ¥15", "10%外税 15")):
        assert normalize_tax_label("内税", ambiguous, rate="10%", amount=15) == "内税"
    assert normalize_tax_label("非課税", text, rate="10%", amount=15) == "非課税"


def test_column_split_tax_amount_survives_rate_only_cleanup():
    text = "\n".join([
        "小計",
        "10%内税対象",
        "(10%内)",
        "(税合計",
        "合計",
        "お預り",
        "お釣り",
        "お買上点数",
        "1点",
        "¥980",
        "¥1,078",
        "¥98",
        "¥98)",
        "¥1,078",
        "¥1,080",
        "¥2",
    ])

    extracted = extract_financial_totals(text)
    assert extracted["subtotal"] == 980
    assert extracted["total"] == 1078
    assert _tax_amounts(extracted) == {"10%": 98}

    _clear_unprinted_rate_only_tax_summary(extracted, text)
    assert _tax_amounts(extracted) == {"10%": 98}


def test_net_items_plus_tax_arithmetic_overrides_inner_target_value_stack():
    text = "10%内税対象\n¥1,078\n(10%内)\n¥98\n合計\n¥1,078"

    assert normalize_tax_label(
        "内税", text, subtotal=980, total=1078, tax_sum=98, items_sum=980
    ) == "外税"


def test_gross_item_sum_keeps_inclusive_inner_target_value_stack():
    text = "10%内税対象\n¥1,078\n(10%内)\n¥98\n合計\n¥1,078"

    assert normalize_tax_label(
        "外税", text, subtotal=980, total=1078, tax_sum=98, items_sum=1078
    ) == "内税"


def test_interleaved_two_rate_summary_uses_each_printed_tax_amount():
    text = "\n".join([
        "小計 ¥5,132",
        "税率8%課税対象額 ¥5,531",
        "税率8%税額",
        "税率10%課税対象額",
        "¥5,542",
        "¥409",
        "¥11",
        "税率10%税額",
        "合計",
        "お預り",
        "¥10,542",
        "お釣り",
        "¥5,000",
        "(消費税等",
        "¥410)",
        "お買上点数",
        "19点",
        "¥1",
    ])

    extracted = extract_financial_totals(text)
    assert extracted["subtotal"] == 5132
    assert extracted["total"] == 5542
    assert _tax_amounts(extracted) == {"8%": 409, "10%": 1}
    assert extract_rate_bases(text) == {"8%": 5531, "10%": 11}


def test_printed_tax_wins_over_nearby_rate_calculation_and_preserves_nontaxable():
    text = "\n".join([
        "小計",
        "¥2,275",
        "外税8%対象額",
        "¥2,275",
        "外税8%",
        "¥181",
        "合計",
        "¥2,456",
    ])
    extracted = {
        "total": 2456,
        "subtotal": 2274,
        "taxes": [
            {"rate": "8%", "label": "外税", "amount": 182},
            {"rate": "0%", "label": "非課税", "amount": 3},
        ],
        "line_items": [{"description": "商品", "total": 2275}],
    }

    _restore_bare_number_tax_summary(extracted, text)

    assert _tax_amounts(extracted) == {"8%": 181}
    assert {tax["rate"]: tax["amount"] for tax in extracted["taxes"]}["0%"] == 3
    assert extracted["subtotal"] == 2275


def test_zero_rate_target_does_not_create_positive_tax_or_collapse_total():
    text = "\n".join([
        "小計 ¥2,278",
        "外税8%対象額 ¥2.275",
        "外税8% ¥182",
        "外税10%対象額 ¥3",
        "外税10% ¥0",
        "合計",
        "¥2,460",
    ])

    extracted = extract_financial_totals(text)

    assert extracted["subtotal"] == 2278
    assert extracted["total"] == 2460
    assert _tax_amounts(extracted) == {"8%": 182}
    assert extract_rate_bases(text) == {"8%": 2275, "10%": 3}


def test_gross_target_tax_column_keeps_tax_value_not_target_value():
    text = "\n".join([
        "小計",
        "¥1,409",
        "(8%対象",
        "¥1,510",
        "消費税",
        "¥111)",
        "合計",
        "¥1,520",
    ])

    extracted = extract_financial_totals(text)

    assert extracted["subtotal"] == 1409
    assert extracted["total"] == 1520
    assert _tax_amounts(extracted) == {"8%": 111}


def test_breakdown_is_inclusive_when_items_already_sum_to_total():
    text = "\n".join([
        "合計",
        "¥3,036",
        "※内訳 (10%)",
        "¥2,760",
        "(消費税)",
        "¥276",
    ])

    extracted = extract_financial_totals(text)
    assert extracted["total"] == 3036
    assert extracted["subtotal"] == 2760
    assert _tax_amounts(extracted) == {"10%": 276}
    assert normalize_tax_label(
        extracted["taxes"][0]["label"],
        text,
        subtotal=2760,
        total=3036,
        tax_sum=276,
        items_sum=3036,
    ) == "内税"


def test_rate_only_text_is_cleared_but_tax_excluded_blocks_keep_their_owner():
    rate_only = "合計\n¥1,100\n消費税10%対象"
    inferred = {
        "total": 1100,
        "subtotal": 1000,
        "taxes": [{"rate": "10%", "label": "内税", "amount": 100}],
    }
    _clear_unprinted_rate_only_tax_summary(inferred, rate_only)
    assert inferred["subtotal"] is None
    assert inferred["taxes"] == []

    tax_excluded = "\n".join([
        "小計(税抜8%)",
        "消費税等(8%)",
        "小計(税抜10%)",
        "消費税等(10%)",
        "¥796",
        "¥63",
        "¥165",
        "¥16",
    ])
    assert _interleaved_rate_tax_summary_entries(tax_excluded.splitlines()) == []
    restored = {"total": 1040, "taxes": []}
    _restore_tax_excluded_per_rate_blocks(restored, tax_excluded)
    assert _tax_amounts(restored) == {"8%": 63, "10%": 16}


def test_full_points_tender_does_not_invent_credit_payment_method():
    extracted = {"total": 570, "points_used": 0, "amount_paid": 570}

    reconcile_points_payment_from_ocr(extracted, "ポイント利用 570")

    assert extracted["points_used"] == 570
    assert extracted["amount_paid"] == 0
    assert "payment_method" not in extracted


def test_interleaved_bare_summary_uses_complete_rate_arithmetic_before_position():
    text = "\n".join([
        "本体合計(5点)",
        "(10%対象",
        "200",
        "50軽",
        "355軽",
        "10",
        "1,780",
        "1)",
        "10",
        "消費税",
        "(8%対象",
        "1,770",
        "消費税",
        "141)",
        "総合計",
        "1,922",
    ])

    extracted = extract_financial_totals(text)

    assert extracted["subtotal"] == 1780
    assert extracted["total"] == 1922
    assert _tax_amounts(extracted) == {"10%": 1, "8%": 141}


def test_grand_total_after_tax_line_beats_earlier_body_total():
    text = "\n".join([
        "商品",
        "本体合計(3点)",
        "1,623 軽",
        "1,623",
        "(8%対象",
        "1,623",
        "総合計",
        "消費税 129 )",
        "1,752",
    ])

    extracted = extract_financial_totals(text)

    assert extracted["subtotal"] == 1623
    assert extracted["total"] == 1752
    assert _tax_amounts(extracted) == {"8%": 129}


def test_malformed_comma_rate_base_is_validated_by_summary_arithmetic():
    text = "\n".join([
        "小計",
        "¥692",
        "外税8%対象額",
        "¥6,90",
        "外税8%",
        "¥55",
        "外税10%対象額",
        "¥2",
        "外税10%",
        "¥0",
        "合計",
        "¥747",
    ])

    extracted = extract_financial_totals(text)

    assert extracted["subtotal"] == 692
    assert extracted["total"] == 747
    assert _tax_amounts(extracted) == {"8%": 55}


def test_nontaxable_target_preserves_zero_tax_amount():
    text = "\n".join([
        "小計",
        "¥2,983",
        "外税8%対象額",
        "¥2,148",
        "外税8%",
        "¥171",
        "非課税対象額",
        "¥652",
        "合計",
        "¥3,806",
    ])

    extracted = extract_financial_totals(text)

    assert {tax["rate"]: tax["amount"] for tax in extracted["taxes"]}["0%"] == 0


def test_late_second_tax_recomputes_canonical_subtotal():
    text = "\n".join([
        "小計",
        "¥4,644",
        "外税10%対象額",
        "¥2,878",
        "10%外税額",
        "¥287",
        "外税8%対象額",
        "¥1,766",
        "8%外税額",
        "¥141",
        "合計",
        "¥5,072",
    ])

    extracted = extract_financial_totals(text)

    assert extracted["subtotal"] == 4644
    assert extracted["total"] == 5072
    assert _tax_amounts(extracted) == {"10%": 287, "8%": 141}


def test_column_reordered_summary_allows_unprinted_rounds_to_zero_rate_tax():
    from receipt_parser.receipt_financial import _rate_base_tax_pair_is_valid

    assert _rate_base_tax_pair_is_valid('10%', 9, 0, '外税')
    assert _rate_base_tax_pair_is_valid('8%', 12, 0, '外税')
    assert not _rate_base_tax_pair_is_valid('10%', 10, 0, '外税')
    assert not _rate_base_tax_pair_is_valid('8%', 13, 0, '外税')
    text = "\n".join([
        "小計",
        "税率 8% 課税対象額",
        "¥199",
        "¥159",
        "¥2,111",
        "¥2,274",
        "¥168",
        "¥5",
        "税率 8%税額",
        "計",
        "税率10%課税対象額",
        "合計",
        "¥2,279",
        "(消費税等",
        "¥168)",
    ])

    assert extract_rate_bases(text) == {"8%": 2274, "10%": 5}


@pytest.mark.parametrize(
    (
        "text",
        "expected_subtotal",
        "expected_total",
        "expected_taxes",
        "added_tax_sum",
        "all_tax_sum",
    ),
    [
        (
            "\n".join([
                "小計 ¥3,400",
                "区分外8%課税対象額 ¥2,000",
                "区分外8%税額 ¥160",
                "区分外10%課税対象額 ¥300",
                "区分外10%税額 ¥30",
                "区分内10%課税対象額 ¥1,100",
                "区分内10%税額 ¥100",
                "合計 ¥3,590",
            ]),
            3400,
            3590,
            {("8%", "外税", 160), ("10%", "外税", 30), ("10%", "内税", 100)},
            190,
            290,
        ),
        (
            "\n".join([
                "小計 ¥4,100",
                "区分外8%課税対象額 ¥2,000",
                "区分外8%税額 ¥160",
                "区分外10%課税対象額 ¥1,000",
                "区分外10%税額 ¥100",
                "区分内10%課税対象額 ¥1,100",
                "区分内10%税額 ¥100",
                "合計 ¥4,360",
            ]),
            4100,
            4360,
            {("8%", "外税", 160), ("10%", "外税", 100), ("10%", "内税", 100)},
            260,
            360,
        ),
    ],
)
def test_mixed_same_rate_groups_use_only_added_tax_and_hide_ambiguous_rate_base(
    text, expected_subtotal, expected_total, expected_taxes, added_tax_sum, all_tax_sum,
):
    extracted = extract_financial_totals(text)

    assert extracted["subtotal"] == expected_subtotal
    assert extracted["total"] == expected_total
    assert {
        (tax["rate"], tax["label"], tax["amount"])
        for tax in extracted["taxes"]
    } == expected_taxes
    assert _sum_taxable_amounts(extracted["taxes"]) == added_tax_sum
    labels_first = text.replace(
        "区分内10%課税対象額 ¥1,100\n区分内10%税額 ¥100",
        "区分内10%課税対象額\n区分内10%税額\n¥1,100\n¥100",
    )
    assert extract_financial_totals(labels_first)["taxes"] == extracted["taxes"]
    expected_mixed_taxes = [dict(tax) for tax in extracted["taxes"]]
    mismatch_total = expected_total + 10
    extracted_on_mismatch = {
        "subtotal": expected_subtotal,
        "total": mismatch_total,
        "taxes": [dict(tax) for tax in expected_mixed_taxes],
    }
    _apply_financial_overrides(
        extracted_on_mismatch,
        {"subtotal": expected_subtotal, "total": mismatch_total},
        0.9,
        {},
    )
    assert extracted_on_mismatch["taxes"] == expected_mixed_taxes
    for ocr_has_total in (True, False):
        partial_ocr = {
            "subtotal": expected_subtotal,
            "taxes": [dict(tax) for tax in expected_mixed_taxes if tax["label"] == "外税"],
        }
        if ocr_has_total:
            partial_ocr["total"] = expected_total
        _apply_financial_overrides(extracted, partial_ocr, 0.9, {})
        assert extracted["taxes"] == expected_mixed_taxes
    assert extract_rate_bases(text) == {"8%": 2000, "10%": None}
    for label in ("内税", "外税"):
        assert normalize_tax_label(
            label,
            text,
            subtotal=expected_subtotal,
            total=expected_total,
            tax_sum=all_tax_sum,
            items_sum=expected_subtotal + 500,
        ) == label
    assert _sum_taxable_amounts([
        {"rate": "8%", "label": "外税", "amount": 160},
        {"rate": "10%", "label": "内税", "amount": 0},
    ]) == 160


def test_mixed_tax_restore_is_idempotent_and_rejects_ambiguous_local_pair():
    text = "\n".join([
        "小計 ¥3,400",
        "区分外8%課税対象額 ¥2,000",
        "区分外8%税額 ¥160",
        "区分外10%課税対象額 ¥300",
        "区分外10%税額 ¥30",
        "区分内10%課税対象額 ¥1,100",
        "区分内10%税額",
        "¥91",
        "¥100",
        "合計 ¥3,590",
    ])
    extracted = {
        "subtotal": 3400,
        "total": 3590,
        "taxes": [
            {"rate": "8%", "label": "外税", "amount": 160},
            {"rate": "10%", "label": "外税", "amount": 31},
        ],
    }

    _restore_bare_number_tax_summary(extracted, text)
    once = list(extracted["taxes"])
    _restore_bare_number_tax_summary(extracted, text)

    assert extracted["taxes"] == once
    assert {(tax["rate"], tax["label"], tax["amount"]) for tax in once} == {
        ("8%", "外税", 160),
        ("10%", "外税", 30),
        ("10%", "内税", 100),
    }

    ambiguous = text.replace("¥91", "¥99")
    assert _interleaved_rate_tax_summary_entries(ambiguous.splitlines()) == []
    untouched = {
        "subtotal": 3400,
        "total": 3590,
        "taxes": [
            {"rate": "8%", "label": "外税", "amount": 160},
            {"rate": "10%", "label": "外税", "amount": 30},
            {"rate": "10%", "label": "内税", "amount": 100},
        ],
    }
    before = {
        **untouched,
        "taxes": [dict(tax) for tax in untouched["taxes"]],
    }
    _restore_bare_number_tax_summary(untouched, ambiguous)
    assert untouched == before

    duplicate_label = text.replace(
        "区分内10%課税対象額",
        "区分外10%税額 ¥30\n区分内10%課税対象額",
    )
    assert _interleaved_rate_tax_summary_entries(duplicate_label.splitlines()) == []

    wrong_mode_pair = text.replace(
        "区分外10%課税対象額 ¥300\n区分外10%税額 ¥30",
        "区分外10%課税対象額 ¥1,100\n区分外10%税額 ¥100",
    )
    assert _interleaved_rate_tax_summary_entries(wrong_mode_pair.splitlines()) == []

    aggregate_boundary = text.replace(
        "区分内10%税額\n¥91\n¥100",
        "区分内10%税額\n税合計 ¥190\n¥100",
    )
    assert _interleaved_rate_tax_summary_entries(aggregate_boundary.splitlines()) == []
def test_explicit_rated_tax_included_heading_owns_separate_tax_amount_label():
    text = '10%税込対象額\n¥2,200\n(10%税額\n¥200)\n合計\n¥2,200'
    for items_sum in (2000, 2200):
        assert normalize_tax_label('税額', text, subtotal=2000, total=2200,
                                   tax_sum=200, items_sum=items_sum) == '内税'
    mixed = text + '\n10%外税対象額\n¥1,000\n10%外税\n¥100'
    for label in ('内税', '外税'):
        assert normalize_tax_label(label, mixed, subtotal=3200, total=3300,
                                   tax_sum=300, items_sum=3200) == label
