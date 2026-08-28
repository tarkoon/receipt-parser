"""Structural contracts for printed receipt tax summaries."""

from receipt_parser.receipt_financial import (
    _interleaved_rate_tax_summary_entries,
    extract_financial_totals,
    extract_rate_bases,
    normalize_tax_label,
    reconcile_points_payment_from_ocr,
)
from receipt_parser.receipt_items import _clear_unprinted_rate_only_tax_summary
from receipt_parser.receipt_late_repairs import _restore_tax_excluded_per_rate_blocks
from receipt_parser.receipt_totals import _restore_bare_number_tax_summary


def _tax_amounts(extracted):
    return {
        tax["rate"]: tax["amount"]
        for tax in extracted.get("taxes", [])
        if tax.get("rate") != "0%" and tax.get("amount", 0) > 0
    }


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
