"""Focused contracts for receipt row ownership; no OCR or model calls."""

import pytest


def _item(description, total, qty=1, unit_price=None):
    return {
        "description": description,
        "qty": qty,
        "unit_price": total if unit_price is None else unit_price,
        "total": total,
        "discount": 0,
        "discount_rate": "",
    }


def test_summary_count_is_not_used_as_single_row_quantity():
    from receipt_parser.receipt_item_repair import _fix_qty_from_ocr_patterns

    items = [_item("一般商品甲乙", 900)]

    _fix_qty_from_ocr_patterns(
        items,
        "一般商品甲乙\n¥900\n本体合計(3点)\n¥900",
    )

    assert items == [_item("一般商品甲乙", 900)]


def test_local_unit_quantity_arithmetic_preserves_visible_unit_price():
    from receipt_parser.receipt_item_repair import _fix_qty_from_ocr_patterns

    items = [_item("一般商品甲乙", 1170)]

    _fix_qty_from_ocr_patterns(
        items,
        "一般商品甲乙\n¥585 2個\n¥1,170\n小計\n¥1,170",
    )

    assert items[0]["qty"] == 2
    assert items[0]["unit_price"] == 585
    assert items[0]["total"] == 1170


def test_footer_unit_count_is_not_borrowed_as_row_quantity():
    from receipt_parser.receipt_item_repair import _fix_qty_from_ocr_patterns

    items = [_item("一般商品甲乙", 100)]

    _fix_qty_from_ocr_patterns(
        items,
        "一般商品甲乙\n100\n小計\n100\n2個",
    )

    assert items == [_item("一般商品甲乙", 100)]


def test_preposed_count_unit_owns_one_forward_title_and_exact_printed_total():
    from copy import deepcopy
    from receipt_parser.receipt_item_repair import _fix_qty_from_ocr_patterns

    original = [_item("反復商品甲乙", 900), _item("反復商品甲乙", 2700)]
    text = "反復商品甲乙 ¥900\n3点\n0900\n反復商品甲乙 ¥2,700\n小計\n3,600"
    items = deepcopy(original)
    _fix_qty_from_ocr_patterns(items, text)
    assert items[0] == original[0]
    assert items[1] == _item("反復商品甲乙", 2700, qty=3, unit_price=900)
    expected = deepcopy(items)
    _fix_qty_from_ocr_patterns(items, text)
    assert items == expected

    for unsupported in (
        text.replace("¥2,700", "2,700"),
        text.replace("0900", "0800"),
        text.replace("反復商品甲乙 ¥2,700", "別商品甲乙 ¥2,700"),
        text.replace("反復商品甲乙 ¥2,700", "小計 ¥2,700"),
        text.replace("反復商品甲乙 ¥2,700", "小計\n反復商品甲乙 ¥2,700"),
        "小計 ¥900\n" + text,
        text.replace("小計", "3点\n0900\n反復商品甲乙 ¥2,700\n小計"),
    ):
        items = deepcopy(original)
        _fix_qty_from_ocr_patterns(items, unsupported)
        assert items == original
    items = [original[1].copy(), original[1].copy()]
    _fix_qty_from_ocr_patterns(items, text)
    assert items == [original[1], original[1]]
    items = deepcopy(original)
    items[1]["discount_rate"] = "10%"
    expected = deepcopy(items)
    _fix_qty_from_ocr_patterns(items, text)
    assert items == expected


def test_unit_count_phantom_requires_owned_extended_price_and_complete_subtotal():
    from copy import deepcopy
    from receipt_parser.receipt_item_repair import _fix_qty_from_ocr_patterns

    text = "商品甲乙 ¥100\n商品丙丁\n4900000000001\n¥80 2個\n¥160\n小計\n¥260"
    correct = [_item("商品甲乙", 100), _item("商品丙丁", 160, qty=2, unit_price=80)]
    joined = "商品甲乙 ¥100\n商品丙丁 ¥160\n¥80 2個\n小計\n¥260"
    items = deepcopy(correct) + [_item("未読", 80)]
    _fix_qty_from_ocr_patterns(items, joined)
    assert items == correct
    for unsupported in (
        joined.replace("¥80 2個", "¥80 2個\n¥159"),
        joined.replace("商品丙丁 ¥160", "商品丙丁 ¥159"),
        joined.replace("小計", "別商品名\n小計"),
    ):
        items = deepcopy(correct) + [_item("未読", 80)]
        _fix_qty_from_ocr_patterns(items, unsupported)
        assert len(items) == 3
    for description in ("未読", "¥80 2個"):
        for phantom in (_item(description, 80), _item(description, 160), _item(description, 160, qty=2, unit_price=80)):
            for position in range(3):
                items = deepcopy(correct)
                items.insert(position, deepcopy(phantom))
                _fix_qty_from_ocr_patterns(items, text)
                assert items == correct
                _fix_qty_from_ocr_patterns(items, text)
                assert items == correct
    for unsupported in (
        text.replace("¥160", "¥159"),
        text.replace("¥260", "¥340"),
        text.replace("¥160", "別商品名\n¥160"),
        text.replace("商品丙丁\n", "商品丙丁\n商品丙丁\n"),
        text.replace("¥80 2個", "小計\n¥80 2個"),
    ):
        items = deepcopy(correct) + [_item("未読", 80)]
        _fix_qty_from_ocr_patterns(items, unsupported)
        assert len(items) == 3
    items = deepcopy(correct) + [_item("別商品名 2個", 80)]
    _fix_qty_from_ocr_patterns(items, text.replace("小計", "別商品名 2個 ¥80\n小計"))
    assert len(items) == 3
    items = deepcopy(correct) + [_item("未読", 80), _item("未読", 80)]
    _fix_qty_from_ocr_patterns(items, text)
    assert len(items) == 4
    ascii_text = text.replace("商品甲乙", "AB120-240-IVY")
    ascii_correct = deepcopy(correct)
    ascii_correct[0]["description"] = "AB120-240-IVY"
    items = deepcopy(ascii_correct) + [_item("未読", 80)]
    _fix_qty_from_ocr_patterns(items, ascii_text)
    assert items == ascii_correct
    for unsupported in (
        ascii_text.replace("AB120-240-IVY", "AB121-240-IVY"),
        ascii_text.replace("AB120-240-IVY ¥100", "AB120-240-IVY ¥101"),
        ascii_text.replace("AB120-240-IVY ¥100", "AB120-240-IVY 100"),
        ascii_text.replace("AB120-240-IVY ¥100", "AB120-240-IVY ¥100\nAB120-240-IVY ¥100"),
    ):
        items = deepcopy(ascii_correct) + [_item("未読", 80)]
        _fix_qty_from_ocr_patterns(items, unsupported)
        assert len(items) == 3


def test_malformed_qty_digits_abstain_and_complete_pair_preserves_neighbor():
    from receipt_parser.receipt_item_repair import _fix_qty_from_ocr_patterns

    neighbor = _item("隣接商品甲乙", 128)
    target = _item("対象商品甲乙", 198)
    items = [neighbor, target]
    expected_neighbor = dict(neighbor)
    expected_target = dict(target)

    _fix_qty_from_ocr_patterns(
        items,
        "隣接商品甲乙\n128%\n対象商品甲乙\n196* A\n(21 X 198)\n小計\n324",
    )

    assert items[0] == expected_neighbor
    assert items[1] == expected_target
    _fix_qty_from_ocr_patterns(
        items,
        "隣接商品甲乙\n128%\n対象商品甲乙\n196* A\n(2個 X 単98)\n小計\n324",
    )
    assert items[0] == expected_neighbor
    assert (items[1]["qty"], items[1]["unit_price"], items[1]["total"]) == (
        2,
        98,
        196,
    )


def test_mangled_qty_fails_closed_when_factor_interpretation_is_ambiguous():
    from receipt_parser.receipt_item_repair import _fix_qty_from_ocr_patterns

    items = [_item("対象商品甲乙", 60)]
    expected = [dict(items[0])]

    _fix_qty_from_ocr_patterns(
        items,
        "対象商品甲乙\n60* A\n(23 35 X 130 120)\n小計\n60",
    )

    assert items == expected


def test_repeated_description_quantity_belongs_to_its_ocr_occurrence():
    from receipt_parser.receipt_item_repair import _apply_qty_notation_from_ocr

    items = [_item("反復商品甲乙", 100), _item("反復商品甲乙", 100)]

    _apply_qty_notation_from_ocr(
        items,
        "反復商品甲乙\n100\n反復商品甲乙\n2個X単100\n200\n小計\n300",
    )

    assert [(item["qty"], item["unit_price"], item["total"]) for item in items] == [
        (1, 100, 100),
        (2, 100, 200),
    ]


def test_repeated_description_discount_belongs_to_its_ocr_occurrence():
    from receipt_parser.receipt_item_cleanup import _detect_ocr_discounts

    items = [_item("反復商品甲乙", 100), _item("反復商品甲乙", 100)]

    _detect_ocr_discounts(
        items,
        "反復商品甲乙\n100\n反復商品甲乙\n100\n値引\n-10\n小計\n190",
    )

    assert [(item["discount"], item["total"]) for item in items] == [
        (0, 100),
        (10, 90),
    ]


def test_immediate_bundle_discount_remains_owned_by_preceding_row():
    from receipt_parser.receipt_item_cleanup import _detect_ocr_discounts

    items = [_item("一般商品甲乙", 100)]

    _detect_ocr_discounts(items, "一般商品甲乙\n100\nまとめ値引\n-10\n小計\n90")

    assert items[0]["discount"] == 10
    assert items[0]["total"] == 90


@pytest.mark.parametrize("boundary", ["別商品乙", "お預り"])
def test_price_rejoin_stops_before_product_or_payment_boundary(boundary):
    from receipt_parser.normalize import rejoin_price_lines

    text = f"一般商品甲乙\n合計\n{boundary}\n500"

    assert rejoin_price_lines(text) == text


def test_neighborhood_projection_fails_closed_on_equal_candidates():
    from receipt_parser.receipt_projection import _fix_item_totals_from_ocr_neighborhood

    items = [_item("一般商品甲乙", 100), _item("別商品甲乙", 100)]
    expected = [dict(item) for item in items]

    _fix_item_totals_from_ocr_neighborhood(
        items,
        "一般商品甲乙\n150\n別商品甲乙\n150\n小計\n250",
        target_subtotal=250,
        target_total=250,
    )

    assert items == expected


def test_balanced_adjacent_price_shift_uses_unique_desc_desc_amount_amount_fifo():
    from receipt_parser.receipt_item_cleanup import _fix_adjacent_ocr_price_shift_when_balanced

    extracted = {
        "subtotal": 750,
        "total": 750,
        "line_items": [
            _item("直前商品甲乙", 50),
            _item("先行商品甲乙", 430),
            _item("後続商品甲乙", 270),
        ],
    }

    _fix_adjacent_ocr_price_shift_when_balanced(
        extracted,
        "直前商品甲乙 ¥50\n先行商品甲乙\n後続商品甲乙\n¥270\n¥430\n小計\n¥750",
    )

    assert [
        (item["qty"], item["unit_price"], item["total"])
        for item in extracted["line_items"]
    ] == [
        (1, 50, 50),
        (1.0, 270.0, 270.0),
        (1.0, 430.0, 430.0),
    ]


def test_adjacent_price_repair_preserves_complete_title_digits_and_uses_attached_yen_owner():
    from receipt_parser.receipt_item_cleanup import _fix_adjacent_ocr_price_shift_when_balanced

    for price, target in ((180, 900), (500, 480)):
        receipt = {"subtotal": target, "line_items": [
            _item("隣の商品甲", 300), _item("炭酸飲料 600", price),
        ]}
        _fix_adjacent_ocr_price_shift_when_balanced(
            receipt, f"隣の商品甲 ¥300\n炭酸飲料600\n¥180\n小計\n¥{target}",
        )
        assert receipt["line_items"] == [_item("隣の商品甲", 300), _item("炭酸飲料 600", 180)]


def test_balanced_adjacent_price_shift_rejects_mismatched_amount_multiset():
    from receipt_parser.receipt_item_cleanup import _fix_adjacent_ocr_price_shift_when_balanced

    extracted = {
        "subtotal": 700,
        "total": 700,
        "line_items": [_item("先行商品甲乙", 420), _item("後続商品甲乙", 280)],
    }
    expected = [dict(item) for item in extracted["line_items"]]

    _fix_adjacent_ocr_price_shift_when_balanced(
        extracted,
        "先行商品甲乙\n後続商品甲乙\n¥270\n¥430\n小計\n¥700",
    )

    assert extracted["line_items"] == expected


def test_balanced_adjacent_price_shift_rejects_ambiguous_description_owner():
    from receipt_parser.receipt_item_cleanup import _fix_adjacent_ocr_price_shift_when_balanced

    extracted = {
        "subtotal": 700,
        "total": 700,
        "line_items": [_item("先行商品甲乙", 430), _item("後続商品甲乙", 270)],
    }
    expected = [dict(item) for item in extracted["line_items"]]

    _fix_adjacent_ocr_price_shift_when_balanced(
        extracted,
        "先行商品甲乙\n後続商品甲乙\n¥270\n¥430\n先行商品甲乙\n小計\n¥700",
    )

    assert extracted["line_items"] == expected


def test_balanced_adjacent_price_shift_rejects_larger_stack_idempotently():
    from receipt_parser.receipt_item_cleanup import _fix_adjacent_ocr_price_shift_when_balanced

    extracted = {
        "subtotal": 600,
        "total": 600,
        "line_items": [
            _item("積上商品甲乙", 300),
            _item("積上商品丙丁", 200),
            _item("積上商品戊己", 100),
        ],
    }
    expected = [dict(item) for item in extracted["line_items"]]
    ocr_text = (
        "積上商品甲乙\n積上商品丙丁\n積上商品戊己\n"
        "¥100\n¥200\n¥300\n小計\n¥600"
    )

    _fix_adjacent_ocr_price_shift_when_balanced(extracted, ocr_text)
    assert extracted["line_items"] == expected

    _fix_adjacent_ocr_price_shift_when_balanced(extracted, ocr_text)
    assert extracted["line_items"] == expected










def test_multiset_projection_does_not_assign_prices_without_description_owners():
    from receipt_parser.receipt_projection import _project_totals_to_ocr_multiset

    extracted = {
        "subtotal": 300,
        "line_items": [_item("抽出商品甲乙", 50), _item("抽出商品丙丁", 50)],
    }
    expected = [dict(item) for item in extracted["line_items"]]

    _project_totals_to_ocr_multiset(
        extracted,
        "OCR商品甲乙\n100\nOCR商品丙丁\n200\n小計\n300",
    )

    assert extracted["line_items"] == expected


@pytest.mark.parametrize("rate_text", ["160円", "単価\n0160", "@160"])
def test_fuel_usage_uses_unique_local_rate_and_fills_only_null_members(rate_text):
    from receipt_parser.receipt_item_repair import _extract_fuel_usage

    extracted = {
        "total": 2000,
        "usage": {
            "amount": None,
            "unit": None,
            "cost_per": None,
            "meter_previous": 7,
            "meter_current": None,
        },
    }

    _extract_fuel_usage(extracted, f"数量\n12.50L\n{rate_text}\n合計\n¥2,000")

    assert extracted["usage"] == {
        "amount": 12.5,
        "unit": "L",
        "cost_per": 160,
        "meter_previous": 7,
        "meter_current": None,
    }


def test_row_grouping_preserves_tilt_without_bridging_adjacent_summary_rows():
    from receipt_parser.receipt_projection import _collect_direct_summary_owners, _group_layout_rows

    cells = [
        ("小", 0, 68), ("計", 20, 68), ("3", 40, 69), ("点", 50, 69),
        ("¥", 135, 60), ("720", 155, 60),
        ("(", 0, 90), ("外税", 10, 90), ("10.0%", 45, 90), ("対象額", 85, 90),
        ("¥", 135, 82), ("720", 155, 82), (")", 180, 80),
    ]
    blocks = [dict(text=text, x=x, y=cy - 13, page=0, confidence=0.99,
                   bbox=[[x, cy - 13], [x + 10 * len(text), cy - 13],
                         [x + 10 * len(text), cy + 13], [x, cy + 13]])
              for text, x, cy in cells]
    assert ["".join(word["text"] for word in row) for row in _group_layout_rows(blocks)] == [
        "小計3点¥720", "(外税10.0%対象額¥720)"]
    owners = _collect_direct_summary_owners(blocks)["owners"]
    assert [(owner["count"], owner["value"]) for owner in owners["subtotal"]] == [(3, 720)]
    assert [(owner["kind"], owner["value"], owner["mode"]) for owner in owners["rate_component"]] == [
        ("base", 720, "外税")]
    tilted = [dict(text=text, x=i * 40, y=130 + i * 6, page=0,
                   bbox=[[i * 40, 130 + i * 6], [i * 40 + 10, 130 + i * 6],
                         [i * 40 + 10, 156 + i * 6], [i * 40, 156 + i * 6]])
              for i, text in enumerate("甲乙丙丁戊")]
    assert len(_group_layout_rows(tilted)) == 1
