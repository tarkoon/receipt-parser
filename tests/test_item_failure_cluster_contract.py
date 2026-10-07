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


def test_unheaded_itemization_requires_unique_owned_prices_for_every_row():
    from copy import deepcopy
    from receipt_parser.receipt_items import _fix_bare_service_receipt_without_itemization

    original = {"total": 400, "line_items": [_item("商品甲乙", 240), _item("商品丙丁", 160)]}
    text = "領収証\n商品甲乙\n¥240\n商品丙丁\n¥160\n合計\n¥400"
    for source in (text, text.replace("商品甲乙\n", "商品甲乙 ").replace("商品丙丁\n", "商品丙丁 ")):
        result = deepcopy(original)
        _fix_bare_service_receipt_without_itemization(result, source)
        assert result["line_items"] == original["line_items"]
        _fix_bare_service_receipt_without_itemization(result, source)
        assert result["line_items"] == original["line_items"]
    for unsupported in (
        text.replace("商品丙丁\n¥160", "別商品\n¥160"),
        text.replace("商品甲乙\n¥240", "商品甲乙\n別商品\n¥240"),
        text.replace("商品甲乙\n¥240", "商品甲乙\n¥240\n商品甲乙\n¥240"),
        text.replace("商品丙丁\n¥160", "商品丙丁\n合計 ¥160"),
    ):
        result = deepcopy(original)
        _fix_bare_service_receipt_without_itemization(result, unsupported)
        assert result["line_items"] == []
    result = {"total": 400, "line_items": [_item("合計", 400)]}
    _fix_bare_service_receipt_without_itemization(result, "領収証\n合計 ¥400")
    assert result["line_items"] == []


def _four_row_uncounted_bag_text(*, bag_amount=4, reduced_base=626, standard_base=None):
    first_amount = 130 - bag_amount
    standard_base = bag_amount if standard_base is None else standard_base
    return "\n".join([
        "2099/1/1 00:00",
        "商品甲",
        f"有料レジ袋 {bag_amount}",
        f"{first_amount}*",
        "商品乙 200*",
        "商品丙",
        "300* A",
        "小計",
        "¥630",
        f"外税8%対象額 ¥{reduced_base}",
        f"外税10%対象額 ¥{standard_base}",
        "お買上商品数:3",
        "*印は軽減税率8%対象商品",
    ])


def _dense_fragment_text(fragment_lines, *, subtotal=1000, rate_base=1000):
    return "\n".join([
        "2099/1/1 00:00",
        "先行商品",
        "商品甲 100* 50*",
        *fragment_lines,
        "商品丙 200*",
        "商品丁 300*",
        "商品戊 112*",
        "小計",
        f"¥{subtotal}",
        f"外税8%対象額 ¥{rate_base}",
        "お買上商品数:6",
        "*印は軽減税率8%対象商品",
    ])


def test_dense_projection_accepts_balanced_four_rows_when_count_excludes_one_bag():
    from receipt_parser.receipt_row_projection import (
        _replace_dense_sequence_rows_when_balanced,
    )

    for initial_items in ([], [_item("未解析", 630)]):
        extracted = {"subtotal": 630, "line_items": initial_items}

        _replace_dense_sequence_rows_when_balanced(
            extracted,
            _four_row_uncounted_bag_text(),
        )

        assert [
            (row["description"], row["total"], row["tax_category"])
            for row in extracted["line_items"]
        ] == [
            ("有料レジ袋", 4.0, "10%"),
            ("商品甲", 126.0, "8%"),
            ("商品乙", 200.0, "8%"),
            ("商品丙", 300.0, "8%"),
        ]


def test_dense_projection_four_row_exception_requires_low_bag_and_balanced_rate_bases():
    from receipt_parser.receipt_row_projection import (
        _replace_dense_sequence_rows_when_balanced,
    )

    for text in (
        _four_row_uncounted_bag_text(reduced_base=620),
        _four_row_uncounted_bag_text(bag_amount=14, reduced_base=616),
        _four_row_uncounted_bag_text(reduced_base=620, standard_base=10),
    ):
        original = []
        extracted = {"subtotal": 630, "line_items": original}

        _replace_dense_sequence_rows_when_balanced(extracted, text)

        assert extracted["line_items"] is original


def test_dense_projection_requires_complete_amount_in_split_or_joined_shape():
    from receipt_parser.receipt_row_projection import (
        _replace_dense_sequence_rows_when_balanced,
    )

    for fragment_lines in (["商品乙", "2"], ["商品乙 2"], ["商品乙", "238*"], ["商品乙 238*"]):
        original = []
        extracted = {"subtotal": 1000, "line_items": original}

        _replace_dense_sequence_rows_when_balanced(
            extracted,
            _dense_fragment_text(fragment_lines),
        )

        if fragment_lines in (["商品乙", "2"], ["商品乙 2"]):
            assert extracted == {"subtotal": 1000, "line_items": []}
            assert extracted["line_items"] is original
            _replace_dense_sequence_rows_when_balanced(extracted, _dense_fragment_text(fragment_lines))
            assert extracted == {"subtotal": 1000, "line_items": []}
            continue

        assert [
            (row["description"], row["total"])
            for row in extracted["line_items"]
        ] == [
            ("商品甲", 100.0),
            ("先行商品", 50.0),
            ("商品乙", 238.0),
            ("商品丙", 200.0),
            ("商品丁", 300.0),
            ("商品戊", 112.0),
        ]


def test_partial_amount_keeps_balanced_model_shape_unchanged():
    from receipt_parser.receipt_row_projection import (
        _replace_dense_sequence_rows_when_balanced,
    )

    original = [
        _item("モデル行甲", 100),
        _item("モデル行乙", 50),
        _item("モデル行丙", 200),
        _item("モデル行丁", 300),
        _item("モデル行戊", 238),
        _item("モデル行己", 112),
    ]
    extracted = {"subtotal": 1000, "line_items": original}
    expected = [dict(row) for row in original]

    _replace_dense_sequence_rows_when_balanced(
        extracted,
        _dense_fragment_text(["商品乙", "2"]),
    )

    assert extracted["line_items"] is original
    assert extracted["line_items"] == expected
    _replace_dense_sequence_rows_when_balanced(extracted, _dense_fragment_text(["商品乙", "2"]))
    assert extracted["line_items"] == expected


def test_mixed_tax_bases_do_not_supply_missing_item_price_digits():
    from receipt_parser.receipt_row_projection import (
        _replace_dense_sequence_rows_when_balanced,
    )

    original = [
        _item("モデル甲", 600),
        _item("モデル乙", 100),
        _item("モデル丙", 238),
        _item("モデル丁", 3),
        _item("モデル戊", 559),
    ]
    extracted = {"subtotal": 1500, "line_items": original}
    expected = [dict(row) for row in original]
    text = "\n".join([
        "2099/1/1 00:00",
        "前置商品",
        "自治体ごみ袋 600非 100*",
        "商品甲",
        "2",
        "食品ポリ袋 3除",
        "標準商品 180",
        "商品乙 200*",
        "商品丙 179*",
        "小計",
        "¥1500",
        "外税8%対象額 ¥717",
        "外税10%対象額 ¥183",
        "お買上商品数:5",
        "*印は軽減税率8%対象商品",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, text)

    assert extracted["line_items"] is original
    assert extracted["line_items"] == expected
    _replace_dense_sequence_rows_when_balanced(extracted, text)
    assert extracted["line_items"] == expected


def test_fragment_recovery_does_not_treat_an_unmarked_bag_as_count_exempt():
    from receipt_parser.receipt_row_projection import (
        _replace_dense_sequence_rows_when_balanced,
    )

    original = [
        _item("モデル甲", 600),
        _item("モデル乙", 100),
        _item("モデル丙", 238),
        _item("モデル丁", 3),
        _item("モデル戊", 559),
    ]
    extracted = {"subtotal": 1500, "line_items": original}
    text = "\n".join([
        "2099/1/1 00:00",
        "前置商品",
        "自治体ごみ袋 600非 100*",
        "商品甲",
        "2",
        "食品ポリ袋 3",
        "標準商品 180",
        "商品乙 200*",
        "商品丙 179*",
        "小計",
        "¥1500",
        "外税8%対象額 ¥717",
        "外税10%対象額 ¥183",
        "お買上商品数:5",
        "*印は軽減税率8%対象商品",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, text)

    assert extracted["line_items"] is original


def test_dense_projection_fragment_recovery_rejects_ambiguous_or_owned_tokens():
    from receipt_parser.receipt_row_projection import (
        _replace_dense_sequence_rows_when_balanced,
    )

    cases = (
        ["商品乙", "7"],
        ["商品乙", "2", "商品追加", "3"],
        ["商品乙", "2 X 111"],
        ["中間商品 10*", "商品乙", "2"],
    )
    for fragment_lines in cases:
        original = []
        extracted = {"subtotal": 1000, "line_items": original}

        _replace_dense_sequence_rows_when_balanced(
            extracted,
            _dense_fragment_text(fragment_lines),
        )

        assert extracted["line_items"] is original


def test_dense_projection_fragment_recovery_requires_consistent_subtotal_and_rate_base():
    from receipt_parser.receipt_row_projection import (
        _replace_dense_sequence_rows_when_balanced,
    )

    for subtotal, rate_base in ((990, 1000), (1000, 990)):
        original = []
        extracted = {"subtotal": subtotal, "line_items": original}

        _replace_dense_sequence_rows_when_balanced(
            extracted,
            _dense_fragment_text(["商品乙", "2"], subtotal=subtotal, rate_base=rate_base),
        )

        assert extracted["line_items"] is original


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


def test_dense_projection_repairs_only_shifted_names_across_jan_rows():
    from receipt_parser.receipt_row_projection import (
        _replace_dense_sequence_rows_when_balanced,
    )

    items = [
        _item("商品甲 商品乙", 100),
        _item("商品丙", 200),
        _item("商品丙", 300),
    ]
    extracted = {"subtotal": 600, "line_items": items}
    before_details = [
        {key: value for key, value in item.items() if key != "description"}
        for item in items
    ]
    text = "\n".join([
        "2099/1/1 00:00",
        "商品甲",
        "4900000000001JAN",
        "商品乙",
        "100",
        "200",
        "4900000000002 JAN",
        "商品丙 300",
        "小計",
        "3",
        "600",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, text)

    assert extracted["line_items"] is items
    assert [item["description"] for item in items] == ["商品甲", "商品乙", "商品丙"]
    assert [
        {key: value for key, value in item.items() if key != "description"}
        for item in items
    ] == before_details


def test_dense_projection_requires_two_distinct_jan_rows_for_balanced_names():
    from receipt_parser.receipt_row_projection import (
        _replace_dense_sequence_rows_when_balanced,
    )

    original = [
        _item("商品甲", 100),
        _item("商品乙", 200),
        _item("商品丙", 300),
    ]
    extracted = {"subtotal": 600, "line_items": original}
    text = "\n".join([
        "2099/1/1 00:00",
        "商品乙 100",
        "4900000000001JAN",
        "商品甲 200",
        "4900000000001 JAN",
        "商品丙 300",
        "小計",
        "3",
        "600",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, text)

    assert extracted["line_items"] is original
    assert [row["description"] for row in original] == ["商品甲", "商品乙", "商品丙"]


def test_dense_projection_does_not_replace_balanced_rows_with_a_different_money_multiset():
    from receipt_parser.receipt_row_projection import (
        _replace_dense_sequence_rows_when_balanced,
    )

    original = [
        _item("既存甲", 100),
        _item("既存乙", 200),
        _item("既存丙", 200),
        _item("既存丁", 300),
        _item("既存戊", 200),
    ]
    extracted = {"subtotal": 1000, "line_items": original}
    text = "\n".join([
        "2099/1/1 00:00",
        "商品甲 100",
        "商品乙 150",
        "商品丙 200",
        "商品丁 250",
        "商品戊 300",
        "小計",
        "¥1000",
        "外税8%対象額 ¥1000",
        "お買上点数:5",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, text)

    assert extracted["line_items"] is original


def test_dense_projection_does_not_replace_balanced_rows_with_different_descriptions():
    from receipt_parser.receipt_row_projection import (
        _replace_dense_sequence_rows_when_balanced,
    )

    original = [
        _item("商品甲", 200),
        _item("商品乙", 100),
        _item("商品丙", 300),
        _item("商品丁", 150),
        _item("商品戊", 250),
    ]
    extracted = {"subtotal": 1000, "line_items": original}
    text = "\n".join([
        "2099/1/1 00:00",
        "商品甲 100",
        "商品乙 200",
        "商品丙 150",
        "商品丁 250",
        "別の商品 300",
        "小計",
        "¥1000",
        "外税8%対象額 ¥1000",
        "お買上点数:5",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, text)

    assert extracted["line_items"] is original


def test_dense_projection_preserves_balanced_rows_despite_flattened_price_swap():
    from receipt_parser.receipt_row_projection import (
        _replace_dense_sequence_rows_when_balanced,
    )

    original = [
        _item("商品甲", 200),
        _item("商品乙", 100),
        _item("商品丙", 300),
        _item("商品丁", 150),
        _item("商品戊", 250),
    ]
    for row, category in zip(original, ("10%", "8%", "10%", "8%", "8%")):
        row["tax_category"] = category
    extracted = {"subtotal": 1000, "line_items": original}
    text = "\n".join([
        "2099/1/1 00:00",
        "商品甲 100",
        "商品乙 200",
        "商品丙 150",
        "商品丁 250",
        "商品戊 300",
        "小計",
        "¥1000",
        "お買上点数:5",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, text)

    assert extracted["line_items"] is original
    assert [row["total"] for row in extracted["line_items"]] == [
        200,
        100,
        300,
        150,
        250,
    ]
    assert [row["tax_category"] for row in extracted["line_items"]] == [
        "10%",
        "8%",
        "10%",
        "8%",
        "8%",
    ]


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


def test_dense_projection_owns_unlabeled_rate_by_adjacent_discount():
    from receipt_parser.receipt_row_projection import (
        _replace_dense_sequence_rows_when_balanced,
    )

    extracted = {"subtotal": 883, "line_items": []}
    text = "\n".join([
        "2099/1/1 00:00",
        "商品甲 200*",
        "<2個 X 単100)",
        "割引",
        "30%",
        "-60",
        "反復商品 200*",
        "30%",
        "-60",
        "反復商品 453%",
        "割引",
        "40%",
        "-182",
        "商品丁 382*",
        "割引",
        "240%",
        "-153",
        "食品ポリ袋 3除",
        "商品戊",
        "<2個 X 単50)",
        "100*",
        "小計",
        "¥883",
        "外税8%対象額 ¥880",
        "外税10%対象額 ¥3",
        "お買上商品数:7",
        "*印は軽減税率8%対象商品",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, text)

    assert [
        (
            row["description"],
            row["qty"],
            row["unit_price"],
            row["total"],
            row["discount_rate"],
            row["tax_category"],
        )
        for row in extracted["line_items"]
    ] == [
        ("商品甲", 2.0, 100.0, 140.0, "30%", "8%"),
        ("反復商品", 1.0, 200.0, 140.0, "30%", "8%"),
        ("反復商品", 1.0, 453.0, 271.0, "40%", "8%"),
        ("商品丁", 1.0, 382.0, 229.0, "", "8%"),
        ("食品ポリ袋", 1.0, 3.0, 3.0, "", "10%"),
        ("商品戊", 2.0, 50.0, 100.0, "", "8%"),
    ]


def test_dense_projection_does_not_steal_pending_percent_price_as_rate():
    from receipt_parser.receipt_row_projection import (
        _replace_dense_sequence_rows_when_balanced,
    )

    extracted = {"subtotal": 170, "line_items": []}
    text = "\n".join([
        "2099/1/1 00:00",
        "商品甲 100*",
        "割引",
        "商品乙",
        "30%",
        "-30",
        "商品丙 100*",
        "小計",
        "2点",
        "170",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, text)

    assert extracted["line_items"] == []


def test_dense_projection_does_not_rewrite_plausible_discount_rate():
    from receipt_parser.receipt_row_projection import (
        _replace_dense_sequence_rows_when_balanced,
    )

    original = [_item("既存商品", 170)]
    extracted = {"subtotal": 170, "line_items": original}
    text = "\n".join([
        "2099/1/1 00:00",
        "商品甲 100*",
        "割引",
        "20%",
        "-30",
        "商品乙 100*",
        "小計",
        "2点",
        "170",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, text)

    assert extracted["line_items"] is original
    assert original[0]["discount_rate"] == ""


def test_dense_projection_applies_adjacent_discount_to_proven_owner():
    from receipt_parser.receipt_row_projection import (
        _replace_dense_sequence_rows_when_balanced,
    )

    extracted = {"subtotal": 230, "line_items": []}
    text = "\n".join([
        "2099/1/1 00:00",
        "商品甲 200*",
        "割引",
        "30%",
        "商品乙 150*",
        "40%",
        "-60",
        "-60",
        "小計",
        "2点",
        "230",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, text)

    assert [row["total"] for row in extracted["line_items"]] == [140, 90]
    assert extracted["line_items"][0]["discount"] == 60
    assert extracted["line_items"][1]["discount"] == 60
    assert extracted["line_items"][1]["discount_rate"] == "40%"


def test_dense_projection_preserves_queued_titles_across_unassigned_percent_controls():
    from copy import deepcopy
    from receipt_parser.receipt_row_projection import _replace_dense_sequence_rows_when_balanced

    extracted = {"subtotal": 1210, "line_items": []}
    text = "\n".join([
        "2099/1/1 00:00",
        "商品甲 200*", "割引", "40%",
        "商品乙", "商品丙", "30%", "商品丁", "-80",
        "100*", "300*", "-90", "400*", "割引", "30%", "-120",
        "商品戊 500*", "小計", "1210", "外税8%対象額 1210", "お買上商品数:5",
    ])
    _replace_dense_sequence_rows_when_balanced(extracted, text)
    assert [item["description"] for item in extracted["line_items"]] == [
        "商品甲", "商品乙", "商品丙", "商品丁", "商品戊",
    ]
    assert [item["unit_price"] for item in extracted["line_items"]] == [200, 100, 300, 400, 500]
    assert [item["total"] for item in extracted["line_items"]] == [120, 100, 210, 280, 500]
    assert [item["discount"] for item in extracted["line_items"]] == [80, 0, 90, 120, 0]
    assert [item["discount_rate"] for item in extracted["line_items"]] == ["", "", "", "30%", ""]
    expected = deepcopy(extracted)
    _replace_dense_sequence_rows_when_balanced(extracted, text)
    assert extracted == expected
    rejected = {"subtotal": 1210, "line_items": []}
    _replace_dense_sequence_rows_when_balanced(rejected, text.replace("-90", "-75"))
    assert rejected == {"subtotal": 1210, "line_items": []}


def test_dense_projection_rejects_discount_ambiguous_with_pending_item():
    from receipt_parser.receipt_row_projection import (
        _replace_dense_sequence_rows_when_balanced,
    )

    original = []
    extracted = {"subtotal": 170, "line_items": original}
    text = "\n".join([
        "2099/1/1 00:00",
        "商品甲 100*",
        "商品乙",
        "割引",
        "30%",
        "-30",
        "100*",
        "小計",
        "2点",
        "170",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, text)

    assert extracted["line_items"] is original


def test_dense_projection_rejects_deferred_rate_with_pending_item():
    from receipt_parser.receipt_row_projection import (
        _replace_dense_sequence_rows_when_balanced,
    )

    original = []
    extracted = {"subtotal": 170, "line_items": original}
    text = "\n".join([
        "2099/1/1 00:00",
        "商品甲 100*",
        "商品乙",
        "割引",
        "30%",
        "値引",
        "-30",
        "100*",
        "小計",
        "2点",
        "170",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, text)

    assert extracted["line_items"] is original


def test_dense_projection_carries_latest_deferred_rate_owner():
    from receipt_parser.receipt_row_projection import (
        _replace_dense_sequence_rows_when_balanced,
    )

    extracted = {"subtotal": 230, "line_items": []}
    text = "\n".join([
        "2099/1/1 00:00",
        "商品甲 140*",
        "商品乙 150*",
        "割引",
        "40%",
        "値引",
        "-60",
        "小計",
        "2点",
        "230",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, text)

    assert [row["total"] for row in extracted["line_items"]] == [140, 90]
    assert extracted["line_items"][1]["discount"] == 60
    assert extracted["line_items"][1]["discount_rate"] == "40%"


def test_dense_projection_rejects_equal_discount_owner_scores():
    from receipt_parser.receipt_row_projection import (
        _replace_dense_sequence_rows_when_balanced,
    )

    original = []
    extracted = {"subtotal": 290, "line_items": original}
    text = "\n".join([
        "2099/1/1 00:00",
        "商品甲 200*",
        "割引",
        "商品乙 150*",
        "割引",
        "-60",
        "小計",
        "2点",
        "290",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, text)

    assert extracted["line_items"] is original


def test_dense_projection_invalidates_stale_deferred_rate_owner():
    from receipt_parser.receipt_row_projection import (
        _replace_dense_sequence_rows_when_balanced,
    )

    extracted = {"subtotal": 260, "line_items": []}
    text = "\n".join([
        "2099/1/1 00:00",
        "商品甲 200*",
        "割引",
        "30%",
        "商品乙 100*",
        "割引",
        "-40",
        "小計",
        "2点",
        "260",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, text)

    assert [row["total"] for row in extracted["line_items"]] == [200, 60]
    assert extracted["line_items"][1]["discount"] == 40
    assert extracted["line_items"][0]["discount_rate"] == ""
    assert extracted["line_items"][1]["discount_rate"] == ""


def test_dense_projection_rejects_mismatched_deferred_rate_owner():
    from receipt_parser.receipt_row_projection import (
        _replace_dense_sequence_rows_when_balanced,
    )

    original = []
    extracted = {"subtotal": 300, "line_items": original}
    text = "\n".join([
        "2099/1/1 00:00",
        "商品甲 200*",
        "割引",
        "30%",
        "商品乙 100*",
        "-50",
        "小計",
        "2点",
        "300",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, text)

    assert extracted["line_items"] is original


def test_dense_projection_keeps_identical_discount_owners_distinct():
    from receipt_parser.receipt_row_projection import (
        _replace_dense_sequence_rows_when_balanced,
    )

    original = []
    extracted = {"subtotal": 200, "line_items": original}
    text = "\n".join([
        "2099/1/1 00:00",
        "商品甲 100*",
        "割引",
        "商品甲 100*",
        "割引",
        "-30",
        "小計",
        "2点",
        "200",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, text)

    assert extracted["line_items"] is original


def test_dense_projection_rejects_unresolved_discount_control():
    from receipt_parser.receipt_row_projection import (
        _replace_dense_sequence_rows_when_balanced,
    )

    original = []
    extracted = {"subtotal": 200, "line_items": original}
    text = "\n".join([
        "2099/1/1 00:00",
        "商品甲 100*",
        "割引",
        "30%",
        "商品乙 100*",
        "小計",
        "2点",
        "200",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, text)

    assert extracted["line_items"] is original


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


def test_unique_local_amount_only_bundle_keeps_money_without_inventing_rate():
    from receipt_parser.receipt_item_cleanup import (
        _clear_discounts_without_nearby_ocr_marker,
    )

    items = [
        _item("一般商品", 394, qty=2, unit_price=297, discount=200, rate="33.7%")
    ]

    _clear_discounts_without_nearby_ocr_marker(
        items,
        "一般商品\n2個 x ¥297\n¥594\nまとめ値引\n-200\n小計\n¥394",
    )

    assert items[0]["discount"] == 200
    assert items[0]["total"] == 394
    assert items[0]["discount_rate"] == ""


def test_stacked_explicit_rates_use_printed_compound_rate():
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

    assert [row["discount_rate"] for row in items] == ["33.5%"] * 3


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
        "",
        "",
        "33.5%",
        "33.5%",
    ]
    for foreign_block in (
        "Early tea\n500\nLater loaf\n500\n20%\n-100",
        "Early tea\nLater loaf\n500\n20%\n-100",
    ):
        earlier = [_item("Early tea", 400, unit_price=500, discount=100, rate="20%")]
        _clear_discounts_without_nearby_ocr_marker(earlier, foreign_block)
        assert earlier == [_item("Early tea", 500, unit_price=500)]


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
        "33.5%",
        "33.5%",
        "33.5%",
    ]


def test_campaign_projection_repairs_pure_description_permutation_when_money_matches():
    from receipt_parser.receipt_marker_projection import (
        _replace_campaign_discount_stream_when_balanced,
    )

    expected_descriptions = [
        "OCR商品甲",
        "OCR商品乙",
        "OCR商品丙",
        "OCR商品丁",
        "OCR商品戊",
    ]
    extracted = {
        "subtotal": 1129,
        "line_items": [
            _item("OCR商品甲", 95, unit_price=100, discount=5, rate="5%"),
            _item("OCR商品丙", 412, unit_price=621, discount=209, rate="30%"),
            _item("OCR商品乙", 266, unit_price=401, discount=135, rate="30%"),
            _item("OCR商品丁", 261, unit_price=394, discount=133, rate="30%"),
            _item("OCR商品戊", 95, unit_price=100, discount=5, rate="5%"),
        ],
        "taxes": [],
    }
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

    assert [row["description"] for row in extracted["line_items"]] == expected_descriptions
    assert [row["discount_rate"] for row in extracted["line_items"]][1:4] == [
        "33.5%",
        "33.5%",
        "33.5%",
    ]


def test_campaign_abstains_on_unrelated_titles_even_when_money_matches():
    from copy import deepcopy
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
    original = deepcopy(extracted)
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

    assert extracted == original


def test_numeric_product_suffix_is_not_cleaned_without_code_prefix():
    from receipt_parser.receipt_item_repair import (
        _clean_code_prefixed_item_descriptions,
    )

    descriptions = ["レジ袋 5", "2024 記念品 1", "123-456 商品 1"]
    extracted = {"line_items": [_item(desc, 3) for desc in descriptions]}

    _clean_code_prefixed_item_descriptions(extracted)

    assert [item["description"] for item in extracted["line_items"]] == descriptions


def test_numeric_product_suffix_cleanup_uses_unique_code_qty_and_exact_ocr_title():
    from receipt_parser.receipt_item_repair import (
        _clean_code_prefixed_item_descriptions,
    )

    extracted = {"line_items": [
        _item("Model 2024 No. 2 1234567890123", 500, qty=1),
        _item("紅茶 2025（2345678901234）", 600, qty=2),
    ]}
    before = [dict(item) for item in extracted["line_items"]]
    ocr_text = "\n".join([
        "Model 2024 No. 2",
        "¥500",
        "1234567890123 1",
        "軽 2 紅茶 2025",
        "¥300",
        "¥600",
        "2345678901234 2",
    ])

    _clean_code_prefixed_item_descriptions(extracted, ocr_text)
    _clean_code_prefixed_item_descriptions(extracted, ocr_text)

    assert [item["description"] for item in extracted["line_items"]] == [
        "Model 2024 No. 2", "紅茶 2025",
    ]
    assert [
        {key: value for key, value in item.items() if key != "description"}
        for item in extracted["line_items"]
    ] == [
        {key: value for key, value in item.items() if key != "description"}
        for item in before
    ]


def test_numeric_product_suffix_cleanup_rejects_unproven_or_ambiguous_rows():
    from receipt_parser.receipt_item_repair import (
        _clean_code_prefixed_item_descriptions,
    )

    cases = [
        ("商品 1234567890123", 1, "商品\n¥100"),
        ("Product1234567890124", 1, "Product1234567890124\n1234567890124 1"),
        ("商品 1234567890125", 1, "商品\n1234567890125 1\n1234567890125 1"),
        ("商品 1234567890126", 1, "商品\n別商品\n1234567890126 1"),
        ("商品 1234567890127", 2, "商品\n1234567890127 1"),
        ("商品 1234567890130", 1, "商品\n小計\n¥100\n1234567890130 1"),
        ("商品 1234567890132", 0, "商品\n1234567890132 1"),
        ("商品 1234567890133", -1, "商品\n1234567890133 1"),
        ("商品 1234567890134", float("nan"), "商品\n1234567890134 1"),
        ("商品 1234567890135", float("inf"), "商品\n1234567890135 1"),
    ]
    for description, qty, ocr_text in cases:
        item = _item(description, 100, qty=qty)
        extracted = {"line_items": [item]}
        before = dict(item)

        _clean_code_prefixed_item_descriptions(extracted, ocr_text)

        assert extracted["line_items"][0] == before

    duplicated_owner = {"line_items": [
        _item("商品甲 1234567890128", 100),
        _item("商品乙 (1234567890128)", 100),
    ]}
    _clean_code_prefixed_item_descriptions(
        duplicated_owner, "商品甲\n1234567890128 1"
    )
    assert [item["description"] for item in duplicated_owner["line_items"]] == [
        "商品甲 1234567890128", "商品乙 (1234567890128)",
    ]

    mixed = {"line_items": [
        _item("470-0244 パンスト 1", 100),
        _item("商品 1234567890131", 200, qty=2),
    ]}
    before = [dict(item) for item in mixed["line_items"]]
    _clean_code_prefixed_item_descriptions(
        mixed, "パンスト\n商品\n1234567890131 1"
    )
    assert mixed["line_items"] == before


def test_numeric_product_suffix_cleanup_stages_multiple_items_atomically():
    from receipt_parser.receipt_item_repair import (
        _clean_code_prefixed_item_descriptions,
    )

    extracted = {"line_items": [
        _item("商品甲 1234567890129", 100),
        _item("商品乙 2345678901234", 200, qty=2),
    ]}
    before = [dict(item) for item in extracted["line_items"]]
    ocr_text = "\n".join([
        "商品甲", "1234567890129 1",
        "商品乙", "2345678901234 1",
    ])

    _clean_code_prefixed_item_descriptions(extracted, ocr_text)

    assert extracted["line_items"] == before


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
    from receipt_parser.receipt_items import (
        _bag_entries_from_ocr,
        _fix_bag_item_prices_from_ocr,
    )

    extracted = {"line_items": [_item("レジ袋", 6)]}

    _fix_bag_item_prices_from_ocr(
        extracted,
        "レジ袋5円\n091 2100009725206\n05\n小計\n5",
    )

    assert extracted["line_items"][0]["unit_price"] == 5
    assert extracted["line_items"][0]["total"] == 5

    assert _bag_entries_from_ocr("有料レジ袋\n2個×5円\n小計") == [
        {"line": 0, "qty": 2, "unit_price": 5, "total": 10},
    ]
    for price in ("¥248", "¥1,248", "2個×248円", "1,248×5", "2×5.5", "2024"):
        assert _bag_entries_from_ocr(f"有料レジ袋\n{price}\n小計") == []
    assert _bag_entries_from_ocr("有料レジ袋 2024") == []
    for description in ("レジ袋248円", "レジ袋1,248円", "レジ袋3.5円"):
        original = _item(description, 7)
        extracted = {"line_items": [dict(original)]}
        _fix_bag_item_prices_from_ocr(extracted, description)
        assert extracted["line_items"] == [original]


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


def test_hallucinated_price_repair_stops_at_the_next_title_or_summary():
    from receipt_parser.receipt_item_cleanup import _fix_hallucinated_prices

    items = [_item("商品甲200", 200)]
    _fix_hallucinated_prices(items, "商品甲200\n90*")
    assert items == [_item("商品甲200", 90)]

    for boundary in ("商品乙", "商品乙 90", "小計"):
        items = [_item("商品甲200", 200)]
        _fix_hallucinated_prices(items, f"商品甲200\n{boundary}\n90*")
        assert items == [_item("商品甲200", 200)]


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


def test_gap_recovery_requires_complete_prices_in_unique_marked_inline_bag_group():
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

    assert not recovered
    assert items == [_item("既存商品", 100)]
    lines[4] = "238*"
    recovered, items = _recover_gap_group(
        lines, [_item("既存商品", 100)], [(2, 652), (4, 238), (5, 3)], 993,
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
        "238*",
        "食品ポリ袋 3除",
        "小計",
        "993",
    ]

    recovered, items = _recover_gap_group(
        lines,
        original,
        [(2, 652), (4, 238), (5, 3)],
        993,
    )

    assert recovered
    assert [(item["description"], item["total"]) for item in items] == [
        ("既存商品", 100),
        ("行政指定ごみ袋", 652.0),
        ("追加商品", 238.0),
        ("食品ポリ袋", 3),
    ]
def test_complete_marked_stack_owns_every_price_count_and_tax_legend():
    from copy import deepcopy
    from receipt_parser.receipt_late_repairs import _replace_stacked_name_price_rows_when_balanced

    text = "\n".join([
        "架空商店", "T 商品アルファ", "¥310*", "1", "T 商品ベータ60", "T 商品ガンマ",
        "¥220※", "¥140*", "小計額", "¥670", "合計点数", "@670x", "¥670)", "3点",
        "※印は軽減税率(8%)適用商品",
    ])
    original = {"total": 670, "subtotal": 670, "payment_method": "PayPay", "line_items": [
        {"description": "collapsed", "qty": 1, "unit_price": 670, "total": 670},
    ]}
    current = deepcopy(original)
    _replace_stacked_name_price_rows_when_balanced(current, text)
    assert [row["description"] for row in current["line_items"]] == ["T商品アルファ", "T商品ベータ60", "T商品ガンマ"]
    assert [row["total"] for row in current["line_items"]] == [310, 220, 140]
    assert all(row["qty"] == 1 and row["unit_price"] == row["total"] and row["tax_category"] == "8%" and
               row["discount"] == 0 and row["discount_rate"] == "" for row in current["line_items"])
    assert {key: value for key, value in current.items() if key != "line_items"} == {
        key: value for key, value in original.items() if key != "line_items"}
    repeat = deepcopy(current)
    _replace_stacked_name_price_rows_when_balanced(repeat, text)
    assert repeat == current
    for altered in (
        text.replace("¥140*\n", ""), text.replace("T 商品ガンマ\n", ""),
        text.replace("¥140*", "T 商品デルタ\n¥140*"), text.replace("商品ガンマ", "商品アルファ"),
        text.replace("3点", "4点"), text.replace("¥670", "¥680"),
        text.replace("(8%)", "(10%)"), text.split("※印は")[0],
        text.replace("¥140*", "¥140"), text.replace("\n1\n", "\n2\n"),
        text.replace("¥220※", "¥220※\n値引 -¥10"),
        text.replace("合計点数\n@670x", "合計点数\n別のタイトル\n@670x"),
    ):
        unchanged = deepcopy(original)
        _replace_stacked_name_price_rows_when_balanced(unchanged, altered)
        assert unchanged == original


def test_local_literal_title_preserves_product_digits_and_other_business_fields():
    from copy import deepcopy
    from receipt_parser.receipt_items import _canonicalize_complete_ocr_title_rows

    item = {"description": "青葉天かす", "qty": 1, "unit_price": 98,
            "total": 98, "discount": 0, "discount_rate": "", "tax_category": "8%"}
    basket = {"subtotal": 594, "total": 594, "line_items": [item,
        {"description": "ちゃんぽん", "qty": 2, "unit_price": 248,
         "total": 496, "discount": 0, "discount_rate": "", "tax_category": "8%"}]}
    for source, expected in [
        ("青葉天かす 60\n98*\nちゃんぽん 2個 X 単248\n小計\n¥594", "青葉天かす 60"),
        ("青葉天かす 60 ¥98\n小計\n¥594", "青葉天かす 60"),
        ("青葉天かす 60 98\n小計\n¥594", "青葉天かす"),
        ("青葉天かす 60 ¥97\n小計\n¥594", "青葉天かす"),
        ("青葉天かす 60 ¥98\n青葉天かす 80 ¥98\n小計\n¥594", "青葉天かす"),
        ("小計\n¥594\n青葉天かす 60 ¥98", "青葉天かす"),
    ]:
        result = deepcopy(basket)
        expected_result = deepcopy(basket)
        expected_result["line_items"][0]["description"] = expected
        for _ in range(2):
            _canonicalize_complete_ocr_title_rows(result, source)
            assert result == expected_result

    for description, source in [
        ("青葉天かす 80", "青葉天かす 60 ¥98"),
        ("薄手タイツ", "470-0244 薄手タイツ 1 ¥98"),
    ]:
        result = deepcopy(basket)
        result["line_items"][0]["description"] = description
        before = deepcopy(result)
        _canonicalize_complete_ocr_title_rows(result, source)
        assert result == before

    result = deepcopy(basket)
    result["line_items"].append(deepcopy(item))
    before = deepcopy(result)
    _canonicalize_complete_ocr_title_rows(result, "青葉天かす 60 ¥98")
    assert result == before

    result = {"line_items": [dict(item, description="植物ポリ袋 (配合30 98除")]}
    expected = deepcopy(result)
    expected["line_items"][0]["description"] = "植物ポリ袋 (配合30"
    _canonicalize_complete_ocr_title_rows(result, "植物ポリ袋 (配合30 98除")
    assert result == expected

def test_split_weighted_title_requires_unique_owner_and_preserves_digits():
    from copy import deepcopy
    from receipt_parser.receipt_items import _canonicalize_complete_ocr_title_rows

    source = "青葉Blend 7\n単\n315 420g\n¥1,323"
    item = {"description": "青葉Blend 7 単315420g", "qty": 1,
            "unit_price": 1323, "total": 1323, "discount": 0,
            "discount_rate": "", "tax_category": "10%"}
    original = {"subtotal": 1323, "total": 1323, "line_items": [item]}
    expected = deepcopy(original)
    expected["line_items"][0]["description"] = "青葉Blend 7"
    for _ in range(2):
        result = deepcopy(original)
        _canonicalize_complete_ocr_title_rows(result, source)
        assert result == expected

    duplicate_owner = {"subtotal": 2646, "total": 2646,
                       "line_items": [deepcopy(item), deepcopy(item)]}
    before = deepcopy(duplicate_owner)
    _canonicalize_complete_ocr_title_rows(duplicate_owner, source)
    assert duplicate_owner == before

    model_with_unit_only = {"subtotal": 1323, "total": 1323,
                            "line_items": [dict(item, description="青葉Blend 7 単")]}
    before = deepcopy(model_with_unit_only)
    _canonicalize_complete_ocr_title_rows(model_with_unit_only, "\n".join((source, source)))
    assert model_with_unit_only == before

    competing_metadata = (
        "青葉Blend 7\n単\n315 420g\n¥1,323\n"
        "青葉Blend 7\n単\n316 420g\n¥1,323"
    )
    model_with_unit_only = {"subtotal": 1323, "total": 1323,
                            "line_items": [dict(item, description="青葉Blend 7 単")]}
    before = deepcopy(model_with_unit_only)
    _canonicalize_complete_ocr_title_rows(model_with_unit_only, competing_metadata)
    assert model_with_unit_only == before

    unsupported_close = {"subtotal": 1323, "total": 1323,
                         "line_items": [dict(item, description="青葉Blend 7)")]}
    before = deepcopy(unsupported_close)
    _canonicalize_complete_ocr_title_rows(unsupported_close, "青葉Blend 7 1,322除")
    assert unsupported_close == before

def test_complete_count_controls_do_not_replace_separately_owned_repeated_titles():
    from copy import deepcopy
    from receipt_parser.receipt_item_repair import _valid_ocr_item_desc, _replace_duplicate_desc_from_ocr
    from receipt_parser.receipt_items import _fix_duplicate_descriptions_from_ocr

    for control in ('1点', '@145点', '@1451点', '3個', '単145円', '2枚', '４袋', '＠145 足'):
        assert not _valid_ocr_item_desc(control)
    for title in ('7本指手袋', '4個パック', '2027年モデル', '試作商品3', '2枚入り'):
        assert _valid_ocr_item_desc(title)
    items = [{'description': '試作3層コースター', 'qty': 1, 'unit_price': 145,
              'total': 145, 'tax_category': '10%', 'discount': 0, 'discount_rate': ''}
             for _ in range(3)]
    ocr = '\n'.join(['試作3層コースター', 'AB-4567', '@145', '1点', '¥145'] * 3 + ['小計', '¥435'])
    original = deepcopy(items)
    _replace_duplicate_desc_from_ocr(items, ocr)
    assert items == original
    receipt = {'line_items': items}
    _fix_duplicate_descriptions_from_ocr(receipt, ocr)
    assert receipt['line_items'] == original

def test_separately_priced_repeated_purchases_survive_an_unrelated_price_error():
    from copy import deepcopy
    from receipt_parser.receipt_item_repair import _dedup_same_total_items

    item = {'description': '試作7層コースター', 'qty': 1, 'unit_price': 146,
            'total': 146, 'discount': 0, 'discount_rate': ''}
    other = {'description': '検証保管箱', 'qty': 1, 'unit_price': 600, 'total': 600}
    receipt = {'subtotal': 700, 'total': 700, 'line_items': [deepcopy(item), deepcopy(item), other]}
    source = '\n'.join(['試作7層コースター', '¥146', '試作7層コースター ¥146',
                        '検証保管箱 ¥408', '小計', '¥700'])
    original = deepcopy(receipt)
    _dedup_same_total_items(receipt, source)
    _dedup_same_total_items(receipt, source)
    assert receipt == original

    phantom = {'subtotal': 554, 'total': 554,
               'line_items': [deepcopy(item), deepcopy(item), dict(other, unit_price=408, total=408)]}
    single = '\n'.join(['試作7層コースター ¥146', '検証保管箱 ¥408', '小計', '¥554'])
    _dedup_same_total_items(phantom, single)
    assert len(phantom['line_items']) == 2
    assert sum(row['total'] for row in phantom['line_items']) == 554

    changed_digits = deepcopy(original)
    _dedup_same_total_items(changed_digits, source.replace('試作7層', '試作8層'))
    assert len(changed_digits['line_items']) == 2


def test_campaign_controls_never_own_price_stream():
    """Reject numbered header owners while preserving literal discount repairs."""
    from copy import deepcopy
    from receipt_parser.receipt_marker_projection import _replace_campaign_discount_stream_when_balanced

    names = ["晴海サブレ", "春風No.25ソーダ", "夕月ラップ", "稲穂もち", "朝霧のり"]
    gross = [73, 91, 197, 284, 315]
    discount = [0, 0, 39, 57, 63]
    expected = [
        dict(description=name, qty=1.0, unit_price=float(price),
             total=float(price - off), tax_category="8%", discount=float(off),
             discount_rate="")
        for name, price, off in zip(names, gross, discount)
    ]
    target = sum(row["total"] for row in expected)
    # Two inline owners leave the control queue pending until the first stack.
    # A subtotal can balance even when that queue has two spurious owners.
    body = [
        f"371901{names[0]} ¥{gross[0]}",
        f"482012{names[1]} ¥{gross[1]}",
        f"593123{names[2]}",
        f"604234{names[3]}",
        f"¥{gross[2]}", "割引", f"-{discount[2]}",
        f"¥{gross[3]}", "割引", f"-{discount[3]}",
        f"715345{names[4]}",
        f"¥{gross[4]}", "割引", f"-{discount[4]}",
        "小計", f"¥{target:g}", "お買上商品数:5",
    ]
    for controls in (
        ["スNo00073192林", "スキャンレジ0027 スキャンNo4931"],
        ["No.7328 担当者", "レジ 0027"],
        ["レジ0038 ¥91"],
    ):
        text = "\n".join(["見本販売店", "2031/7/14 16:42", *controls, *body])
        complete = dict(subtotal=target, taxes=[], line_items=deepcopy(expected))
        before = deepcopy(complete)
        _replace_campaign_discount_stream_when_balanced(complete, text)
        assert complete == before

        # Existing campaign repairs remain useful even for a balanced placeholder.
        missing = dict(subtotal=target, taxes=[], line_items=[
            dict(description="未分類まとめ", qty=1, unit_price=target, total=target),
        ])
        _replace_campaign_discount_stream_when_balanced(missing, text)
        assert missing["line_items"] == expected
        assert sum(row["total"] for row in missing["line_items"]) == target
        assert all(row["discount_rate"] == "" for row in missing["line_items"])
