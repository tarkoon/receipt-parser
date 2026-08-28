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


def test_digit_repair_accepts_unique_non_terminal_substitution_only_with_full_count():
    from receipt_parser.receipt_item_repair import _fix_digit_misread_items

    extracted = {
        "line_items": [_item("一般商品甲乙", 190), _item("別商品甲乙", 100)],
    }

    _fix_digit_misread_items(
        extracted,
        "一般商品甲乙\n¥190\n別商品甲乙\n¥100\n小計/2点\n小計\n¥250",
    )

    assert [item["total"] for item in extracted["line_items"]] == [150, 100]


def test_digit_repair_allows_only_coherent_discounted_siblings():
    from receipt_parser.receipt_item_repair import _fix_digit_misread_items

    def basket(candidate_discount=0, sibling_discount=20):
        candidate = _item(
            "一般商品甲乙",
            190,
            unit_price=190 + candidate_discount,
        )
        candidate["discount"] = candidate_discount
        sibling = _item("別商品甲乙", 180, qty=2, unit_price=100)
        sibling["discount"] = sibling_discount
        return {"line_items": [candidate, sibling]}

    text = "一般商品甲乙\n¥190\n別商品甲乙\n¥200\n値引\n-20\n小計/3点\n小計\n¥330"

    coherent = basket()
    _fix_digit_misread_items(coherent, text)
    assert [item["total"] for item in coherent["line_items"]] == [150, 180]

    incoherent_sibling = basket(sibling_discount=10)
    _fix_digit_misread_items(incoherent_sibling, text)
    assert [item["total"] for item in incoherent_sibling["line_items"]] == [190, 180]

    discounted_candidate = basket(candidate_discount=20)
    _fix_digit_misread_items(discounted_candidate, text)
    assert [item["total"] for item in discounted_candidate["line_items"]] == [190, 180]


def test_digit_repair_does_not_consume_coupon_after_balanced_subtotal():
    from receipt_parser.receipt_item_repair import _fix_digit_misread_items

    extracted = {
        "line_items": [_item("一般商品甲乙", 190), _item("別商品甲乙", 100)],
    }

    _fix_digit_misread_items(
        extracted,
        "一般商品甲乙\n¥190\n別商品甲乙\n¥100\n小計/2点\n小計\n¥290\nクーポン\n-40\n合計\n¥250",
    )

    assert [item["total"] for item in extracted["line_items"]] == [190, 100]


def test_digit_repair_requires_summary_discount_to_be_in_item_discounts():
    from receipt_parser.receipt_item_repair import _fix_digit_misread_items

    text = "一般商品甲乙\n¥190\n別商品甲乙\n¥140\nクーポン\n-40\n購入点数2点\n合計\n¥250"
    unrepresented = {
        "line_items": [_item("一般商品甲乙", 190), _item("別商品甲乙", 100)],
    }
    _fix_digit_misread_items(unrepresented, text)
    assert [item["total"] for item in unrepresented["line_items"]] == [190, 100]

    represented_sibling = _item("別商品甲乙", 100, unit_price=140)
    represented_sibling["discount"] = 40
    represented = {
        "line_items": [_item("一般商品甲乙", 190), represented_sibling],
    }
    _fix_digit_misread_items(represented, text)
    assert [item["total"] for item in represented["line_items"]] == [150, 100]


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
