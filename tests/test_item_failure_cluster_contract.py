"""Focused contracts for shared line-item repair failure modes.

These tests are deliberately synthetic and offline: they exercise the generic
evidence rules without encoding fixture, merchant, or product answers.
"""


def _item(description, total, *, qty=1, unit_price=None, discount=0, rate=""):
    return {
        "description": description,
        "qty": qty,
        "unit_price": total if unit_price is None else unit_price,
        "total": total,
        "tax_category": "8%",
        "discount": discount,
        "discount_rate": rate,
    }


def test_dense_projection_preserves_complete_balanced_rows():
    from receipt_parser.receipt_row_projection import (
        _replace_dense_sequence_rows_when_balanced,
    )

    expected = [_item("前半 後半", 300), _item("別の商品", 200)]
    extracted = {"subtotal": 500, "line_items": [dict(row) for row in expected]}
    text = "\n".join([
        "2099/1/1 00:00",
        "前半",
        "後半 300",
        "別の商品 200",
        "小計",
        "2",
        "500",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, text)

    assert extracted["line_items"] == expected


def test_dense_projection_repairs_count_inconsistent_table():
    from receipt_parser.receipt_row_projection import (
        _replace_dense_sequence_rows_when_balanced,
    )

    extracted = {"subtotal": 500, "line_items": [_item("未解析", 500)]}
    text = "\n".join([
        "2099/1/1 00:00",
        "商品甲 300",
        "商品乙 200",
        "小計",
        "2",
        "500",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, text)

    assert [row["description"] for row in extracted["line_items"]] == [
        "商品甲",
        "商品乙",
    ]
    assert [row["total"] for row in extracted["line_items"]] == [300.0, 200.0]


def test_subtotal_repair_preserves_tax_inclusive_visible_item_prices():
    from receipt_parser.receipt_recovery import _fix_items_from_subtotal

    extracted = {
        "total": 1919,
        "subtotal": 1777,
        "taxes": [{"rate": "8%", "label": "内税", "amount": 142}],
        "line_items": [_item("商品甲", 237), _item("商品乙", 1682)],
    }
    expected = [dict(row) for row in extracted["line_items"]]

    _fix_items_from_subtotal(
        extracted,
        "商品甲\n¥172\n商品乙\n¥1,682\n小計\n¥1,777\n合計\n¥1,919",
        {"subtotal": 1777},
    )

    assert extracted["line_items"] == expected


def test_summary_count_label_is_not_promoted_to_item_description():
    from receipt_parser.receipt_items import _fix_junk_descriptions

    items = [_item("1点", 3990)]
    text = "\n".join([
        "本物の商品名",
        "2000212418226",
        "買上点数",
        "小計",
        "[14:10]",
        "1 ¥3,990",
        "1点 ¥3,990",
    ])

    _fix_junk_descriptions(items, text)

    assert items[0]["description"] == "本物の商品名"


def test_three_digit_yen_description_is_not_stripped_as_product_code():
    from receipt_parser.receipt_projection import _clean_ocr_price_line_desc

    assert _clean_ocr_price_line_desc("160円皿  ¥480") == "160円皿"


def test_price_rejoin_fails_closed_for_ambiguous_stacked_columns():
    from receipt_parser.normalize import rejoin_price_lines

    text = "商品甲\n商品乙\n¥100\n¥200\n小計\n¥300"

    assert rejoin_price_lines(text) == text


def test_discounted_duplicate_description_uses_gross_price_evidence():
    from receipt_parser.receipt_items import _fix_duplicate_descriptions_from_ocr

    items = [
        _item("反復商品", 227, unit_price=379, discount=152),
        _item("反復商品", 214, unit_price=357, discount=143),
        _item("別の商品 完全名", 228),
    ]
    expected = [dict(row) for row in items]
    text = "\n".join([
        "反復商品",
        "¥379",
        "値引",
        "-152",
        "反復商品",
        "¥357",
        "値引",
        "-143",
        "別の商品",
        "¥228",
        "小計",
        "¥669",
    ])

    _fix_duplicate_descriptions_from_ocr({"line_items": items}, text)

    assert items == expected


def test_ambiguous_group_discount_clears_unsupported_rate_but_keeps_amount():
    from receipt_parser.receipt_item_cleanup import (
        _clear_discounts_without_nearby_ocr_marker,
    )

    items = [
        _item("一般商品", 160, qty=2, unit_price=100, discount=40, rate="20%")
    ]

    _clear_discounts_without_nearby_ocr_marker(
        items,
        "一般商品\n2個 x ¥100\n¥200\n値引\n-40\n小計\n¥160",
    )

    assert items[0]["discount"] == 40
    assert items[0]["total"] == 160
    assert items[0]["discount_rate"] == ""


def test_unique_local_amount_only_discount_uses_effective_rate():
    from receipt_parser.receipt_item_cleanup import (
        _clear_discounts_without_nearby_ocr_marker,
    )

    items = [
        _item("一般商品", 394, qty=2, unit_price=297, discount=200)
    ]

    _clear_discounts_without_nearby_ocr_marker(
        items,
        "一般商品\n2個 x ¥297\n¥594\nまとめ値引\n-200\n小計\n¥394",
    )

    assert items[0]["discount"] == 200
    assert items[0]["total"] == 394
    assert items[0]["discount_rate"] == "33.7%"


def test_stacked_explicit_rates_use_group_effective_rate():
    from receipt_parser.receipt_item_cleanup import (
        _clear_discounts_without_nearby_ocr_marker,
    )

    items = [
        _item("商品甲", 412, unit_price=621, discount=209, rate="30%"),
        _item("商品乙", 266, unit_price=401, discount=135, rate="30%"),
        _item("商品丙", 261, unit_price=394, discount=133, rate="30%"),
    ]
    text = "\n".join([
        "商品甲 621",
        "割引",
        "30%",
        "会員様割引5%",
        "-187",
        "-22",
        "商品乙 401",
        "割引",
        "30%",
        "会員様割引5%",
        "-121",
        "-14",
        "商品丙 394",
        "割引",
        "30%",
        "会員様割引5%",
        "-119",
        "-14",
        "小計",
        "939",
    ])

    _clear_discounts_without_nearby_ocr_marker(items, text)

    assert [row["discount_rate"] for row in items] == ["33.7%"] * 3


def test_stacked_price_bundles_keep_uniquely_supported_discounts():
    from receipt_parser.receipt_item_cleanup import (
        _clear_discounts_without_nearby_ocr_marker,
    )

    items = [
        _item("商品甲", 140, unit_price=148, discount=8, rate="5%"),
        _item("商品乙", 412, unit_price=621, discount=209, rate="33.7%"),
        _item("反復商品", 266, unit_price=401, discount=135, rate="33.7%"),
        _item("反復商品", 261, unit_price=394, discount=133, rate="33.7%"),
    ]
    text = "\n".join([
        "商品甲",
        "会員様割引5%",
        "商品乙",
        "割引",
        "30%",
        "会員様割引5%",
        "反復商品",
        "148",
        "-8",
        "621",
        "-187",
        "-22",
        "401",
        "割引 30%",
        "-121",
        "会員様割引5%",
        "-14",
        "反復商品",
        "394",
        "割引 30%",
        "-119",
        "会員様割引5%",
        "-14",
        "小計",
        "1079",
    ])

    _clear_discounts_without_nearby_ocr_marker(items, text)

    assert [row["total"] for row in items] == [140, 412, 266, 261]
    assert [row["discount"] for row in items] == [8, 209, 135, 133]
    assert [row["discount_rate"] for row in items] == [
        "5%",
        "33.7%",
        "33.7%",
        "33.7%",
    ]


def test_campaign_projection_keeps_combined_effective_rate():
    from receipt_parser.receipt_marker_projection import (
        _replace_campaign_discount_stream_when_balanced,
    )

    extracted = {
        "subtotal": 1129,
        "line_items": [],
        "taxes": [],
    }
    text = "\n".join([
        "2099/1/1 00:00",
        "商品甲 100",
        "会員割引5%",
        "-5",
        "商品乙 621",
        "割引 30%",
        "会員割引5%",
        "-187",
        "-22",
        "商品丙 401",
        "割引 30%",
        "-121",
        "会員割引5%",
        "-14",
        "商品丁 394",
        "割引 30%",
        "-119",
        "会員割引5%",
        "-14",
        "商品戊 100",
        "会員割引5%",
        "-5",
        "小計",
        "1129",
        "お買上商品数:5",
    ])

    _replace_campaign_discount_stream_when_balanced(extracted, text)

    assert [row["discount_rate"] for row in extracted["line_items"]][1:4] == [
        "33.7%",
        "33.7%",
        "33.7%",
    ]


def test_campaign_projection_preserves_descriptions_when_money_already_matches():
    from receipt_parser.receipt_marker_projection import (
        _replace_campaign_discount_stream_when_balanced,
    )

    extracted = {
        "subtotal": 1129,
        "line_items": [
            _item("明示名甲", 95, unit_price=100, discount=5, rate="5%"),
            _item("明示名乙", 412, unit_price=621, discount=209, rate="30%"),
            _item("明示名丙", 266, unit_price=401, discount=135, rate="30%"),
            _item("明示名丁", 261, unit_price=394, discount=133, rate="30%"),
            _item("明示名戊", 95, unit_price=100, discount=5, rate="5%"),
        ],
        "taxes": [],
    }
    descriptions = [row["description"] for row in extracted["line_items"]]
    text = "\n".join([
        "2099/1/1 00:00",
        "OCR商品甲 100",
        "会員割引5%",
        "-5",
        "OCR商品乙 621",
        "割引 30%",
        "会員割引5%",
        "-187",
        "-22",
        "OCR商品丙 401",
        "割引 30%",
        "-121",
        "会員割引5%",
        "-14",
        "OCR商品丁 394",
        "割引 30%",
        "-119",
        "会員割引5%",
        "-14",
        "OCR商品戊 100",
        "会員割引5%",
        "-5",
        "小計",
        "1129",
        "お買上商品数:5",
    ])

    _replace_campaign_discount_stream_when_balanced(extracted, text)

    assert [row["description"] for row in extracted["line_items"]] == descriptions
    assert [row["discount_rate"] for row in extracted["line_items"]][1:4] == [
        "33.7%",
        "33.7%",
        "33.7%",
    ]


def test_numeric_product_suffix_is_not_cleaned_without_code_prefix():
    from receipt_parser.receipt_item_repair import (
        _clean_code_prefixed_item_descriptions,
    )

    extracted = {"line_items": [_item("レジ袋 5", 3)]}

    _clean_code_prefixed_item_descriptions(extracted)

    assert extracted["line_items"][0]["description"] == "レジ袋 5"


def test_non_product_cleanup_requires_arithmetic_support_for_over_total_drop():
    from receipt_parser.receipt_items import _drop_non_product_line_items

    extracted = {
        "total": 42,
        "subtotal": 528,
        "line_items": [_item("明示された商品", 570)],
    }

    _drop_non_product_line_items(
        extracted,
        "明示された商品\n小計\n¥528\n合計\n¥570\nおつり\n¥0",
    )

    assert len(extracted["line_items"]) == 1


def test_single_item_leading_count_uses_summary_only_as_corroboration():
    from receipt_parser.receipt_item_repair import _fix_single_item_qty_from_ocr

    extracted = {
        "total": 1623,
        "line_items": [_item("T ダーク モカ", 1623)],
    }

    _fix_single_item_qty_from_ocr(
        extracted,
        "3T ダーク モカ  1,623 軽\n本体合計(3点)\n¥1,623",
    )

    assert extracted["line_items"][0]["qty"] == 3
    assert extracted["line_items"][0]["unit_price"] == 541
    assert extracted["line_items"][0]["total"] == 1623


def test_unsupported_qty_inflation_uses_unique_visible_total():
    from receipt_parser.receipt_item_repair import _revert_unsupported_qty_inflation

    items = [_item("手巻 炙り熟成紅鮭", 221, qty=2, unit_price=110)]

    _revert_unsupported_qty_inflation(
        items,
        "手巻 炙り熟成紅鮭\n合\n計\n釣  149 軽  221 軽  ¥370\n数\n2個\n¥10,000",
    )

    assert items == [_item("手巻 炙り熟成紅鮭", 221)]


def test_bag_price_ignores_barcode_tail_and_uses_standalone_price():
    from receipt_parser.receipt_items import _fix_bag_item_prices_from_ocr

    extracted = {"line_items": [_item("レジ袋", 6)]}

    _fix_bag_item_prices_from_ocr(
        extracted,
        "レジ袋5円\n091 2100009725206\n05\n小計\n5",
    )

    assert extracted["line_items"][0]["unit_price"] == 5
    assert extracted["line_items"][0]["total"] == 5


def test_mangled_qty_detail_does_not_override_balanced_visible_arithmetic():
    from receipt_parser.receipt_item_repair import _fix_qty_from_ocr_patterns

    items = [_item("果物商品", 196, qty=2, unit_price=98)]

    _fix_qty_from_ocr_patterns(
        items,
        "果物商品\n196* A\n(21 X 198)\n小計\n196",
    )

    assert items == [_item("果物商品", 196, qty=2, unit_price=98)]


def test_plausible_description_is_not_replaced_by_same_price_neighbor():
    from receipt_parser.receipt_items import _fix_item_desc_from_ocr_price_line

    items = [_item("明示された商品名", 200)]

    _fix_item_desc_from_ocr_price_line(
        items,
        "別の商品 200*\n明示された商品名\n小計\n200",
    )

    assert items[0]["description"] == "明示された商品名"


def test_leading_tax_marker_is_not_part_of_description():
    from receipt_parser.receipt_projection import _clean_ocr_price_line_desc

    assert _clean_ocr_price_line_desc("* 商品名 ¥200") == "商品名"


def test_repeated_item_projection_rejects_standalone_party_count_as_description():
    from receipt_parser.receipt_item_repair import _valid_ocr_item_desc
    from receipt_parser.receipt_items import _fix_duplicate_descriptions_from_ocr

    party_counts = ("1名 様", "1 人 様", "1 名 さま", "1 名 サマ")
    assert not any(_valid_ocr_item_desc(text) for text in party_counts)
    assert all(_valid_ocr_item_desc(text) for text in ("1人鍋", "2人前", "1名券"))

    for party_count in party_counts:
        items = [_item("テイクアウト", 200) for _ in range(2)]
        _fix_duplicate_descriptions_from_ocr(
            {"line_items": items}, f"{party_count}\n¥200"
        )

        assert [item["description"] for item in items] == ["テイクアウト"] * 2


def _recover_gap_group(lines, items, unmatched_prices, target):
    from receipt_parser.receipt_recovery import (
        _recover_multiple_missing_items_from_gap,
    )

    extracted = {"line_items": [dict(item) for item in items], "taxes": []}
    recovered = _recover_multiple_missing_items_from_gap(
        extracted,
        "\n".join(lines),
        lines,
        extracted["line_items"],
        unmatched_prices,
        sum(item["total"] for item in items),
        [target],
    )
    return recovered, extracted["line_items"]


def test_gap_recovery_includes_unique_marked_inline_bag_in_three_row_group():
    lines = [
        "2099/1/1 00:00",
        "既存商品 100*",
        "行政指定ごみ袋 652非",
        "追加商品",
        "2",
        "食品ポリ袋 3除",
        "小計",
        "993",
    ]

    recovered, items = _recover_gap_group(
        lines,
        [_item("既存商品", 100)],
        [(2, 652), (5, 3)],
        993,
    )

    assert recovered
    assert [(item["description"], item["total"]) for item in items] == [
        ("既存商品", 100),
        ("行政指定ごみ袋", 652.0),
        ("追加商品", 238.0),
        ("食品ポリ袋", 3.0),
    ]


def test_gap_recovery_rejects_ambiguous_inline_bags_instead_of_falling_back():
    original = [_item("既存商品", 100)]
    lines = [
        "2099/1/1 00:00",
        "既存商品 100*",
        "行政指定ごみ袋 652非",
        "追加商品",
        "2",
        "食品ポリ袋 3除",
        "レジ袋 5除",
        "小計",
        "998",
    ]

    recovered, items = _recover_gap_group(
        lines,
        original,
        [(2, 652), (5, 3), (6, 5)],
        998,
    )

    assert not recovered
    assert items == original


def test_gap_recovery_rejects_inline_bag_with_only_adjacent_tax_marker():
    original = [_item("既存商品", 100)]
    lines = [
        "2099/1/1 00:00",
        "既存商品 100*",
        "行政指定ごみ袋 652非",
        "追加商品",
        "2",
        "食品ポリ袋 3",
        "除",
        "小計",
        "993",
    ]

    recovered, items = _recover_gap_group(
        lines,
        original,
        [(2, 652), (5, 3)],
        993,
    )

    assert not recovered
    assert items == original


def test_gap_recovery_does_not_duplicate_represented_inline_bag():
    original = [_item("既存商品", 100), _item("食品ポリ袋", 3)]
    lines = [
        "2099/1/1 00:00",
        "既存商品 100*",
        "行政指定ごみ袋 652非",
        "追加商品",
        "2",
        "食品ポリ袋 3除",
        "小計",
        "993",
    ]

    recovered, items = _recover_gap_group(
        lines,
        original,
        [(2, 652), (5, 3)],
        993,
    )

    assert recovered
    assert [(item["description"], item["total"]) for item in items] == [
        ("既存商品", 100),
        ("行政指定ごみ袋", 652.0),
        ("追加商品", 238.0),
        ("食品ポリ袋", 3),
    ]
