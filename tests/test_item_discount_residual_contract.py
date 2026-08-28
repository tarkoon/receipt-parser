"""Structural contracts for discount evidence ownership."""

from receipt_parser.receipt_item_cleanup import (
    _clear_discounts_without_nearby_ocr_marker,
)


def _item(description, gross, discount, rate=""):
    return {
        "description": description,
        "qty": 1,
        "unit_price": gross,
        "total": gross - discount,
        "tax_category": "10%",
        "discount": discount,
        "discount_rate": rate,
    }


def test_discount_zone_uses_summary_after_last_owned_item():
    items = [_item("商品甲", 500, 100)]
    text = "\n".join([
        "小計",
        "100",
        "商品甲",
        "500",
        "まとめ値引",
        "-100",
        "小計",
        "400",
    ])

    _clear_discounts_without_nearby_ocr_marker(items, text)

    assert items[0]["discount"] == 100
    assert items[0]["total"] == 400
    assert items[0]["discount_rate"] == "20%"


def test_plain_amount_only_discount_keeps_money_without_inventing_rate():
    items = [_item("商品甲", 3998, 1999, "50%")]
    text = "\n".join([
        "商品甲",
        "値引",
        "3998",
        "-1999",
        "小計",
        "1999",
    ])

    _clear_discounts_without_nearby_ocr_marker(items, text)

    assert items[0]["discount"] == 1999
    assert items[0]["total"] == 1999
    assert items[0]["discount_rate"] == ""


def test_rates_only_allocates_detached_rate_marker_multiset_by_arithmetic():
    items = [
        _item("反復商品", 100, 30, "99%"),
        _item("反復商品", 200, 80),
        _item("反復商品", 300, 120),
    ]
    text = "\n".join([
        "反復商品 100",
        "割引",
        "30%",
        "-30",
        "反復商品 200",
        "割引",
        "反復商品 300",
        "割引",
        "40%",
        "-120",
        "40%",
        "-80",
        "小計",
        "190",
    ])

    _clear_discounts_without_nearby_ocr_marker(items, text, rates_only=True)

    assert [item["discount_rate"] for item in items] == ["30%", "40%", "40%"]


def test_local_malformed_percent_uses_one_digit_stripped_suffix_when_arithmetic_matches():
    items = [_item("商品甲", 382, 153)]
    text = "\n".join([
        "商品甲 382",
        "割引",
        "240%",
        "-153",
        "小計",
        "229",
    ])

    _clear_discounts_without_nearby_ocr_marker(items, text)

    assert items[0]["discount_rate"] == "40%"


def test_local_malformed_percent_suffix_is_rejected_without_arithmetic_match():
    items = [_item("商品甲", 382, 100, "40%")]
    text = "\n".join([
        "商品甲 382",
        "割引",
        "240%",
        "-100",
        "小計",
        "282",
    ])

    _clear_discounts_without_nearby_ocr_marker(items, text)

    assert items[0]["discount_rate"] == ""


def test_complete_monotonic_rate_markers_survive_duplicate_and_noisy_rows():
    items = [
        _item("反復商品", 200, 60, "30%"),
        _item("反復商品", 120, 36, "30%"),
        _item("商品乙", 150, 75, "50%"),
        _item("商品丙", 180, 72, "40%"),
        _item("商品丁", 130, 52, "40%"),
    ]
    text = "\n".join([
        "合計",
        "485",
        "反復商品 200",
        "割引",
        "30%",
        "-60",
        "反復商品",
        "120",
        "30%",
        "-36",
        "商品乙",
        "150*",
        "割引",
        "50%",
        "-75",
        "商品丙",
        "180%",
        "割引",
        "40%",
        "-72",
        "商品丁",
        "130*",
        "割引",
        "240%",
        "-52",
        "小計",
        "485",
    ])

    _clear_discounts_without_nearby_ocr_marker(items, text)

    assert [item["discount_rate"] for item in items] == [
        "30%",
        "30%",
        "50%",
        "40%",
        "40%",
    ]


def test_membership_discount_labels_preserve_multi_row_rates_by_arithmetic():
    items = [
        _item("商品甲", 100, 10, "10%"),
        _item("商品乙", 200, 40, "20%"),
        _item("商品丙", 300, 90, "30%"),
    ]
    text = "\n".join([
        "商品甲 100",
        "部門会員割 10%",
        "-10",
        "商品乙 200",
        "会員様割: 20%",
        "-40",
        "商品丙 300",
        "会員割引 30%",
        "-90",
        "小計",
        "460",
    ])

    _clear_discounts_without_nearby_ocr_marker(items, text, rates_only=True)

    assert [item["discount_rate"] for item in items] == ["10%", "20%", "30%"]


def test_surcharge_tax_rate_and_bare_wari_labels_do_not_support_discount_rates():
    items = [
        _item("商品甲", 100, 10, "10%"),
        _item("商品乙", 200, 40, "20%"),
        _item("商品丙", 300, 90, "30%"),
    ]
    text = "\n".join([
        "商品甲 100",
        "割増 10%",
        "-10",
        "商品乙 200",
        "税率 20%",
        "-40",
        "商品丙 300",
        "割 30%",
        "-90",
        "小計",
        "460",
    ])

    _clear_discounts_without_nearby_ocr_marker(items, text, rates_only=True)

    assert [item["discount_rate"] for item in items] == ["", "", ""]
