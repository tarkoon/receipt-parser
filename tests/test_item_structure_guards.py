import pytest

from receipt_parser.receipt_items import (
    _fix_line_items,
    _normalize_single_measured_service_row,
    _recover_qty_unit_total_item_from_empty_extraction,
)
from receipt_parser.receipt_recovery import _recover_missing_items_from_gap
from receipt_parser.receipt_row_projection import (
    _replace_dense_sequence_rows_when_balanced,
)


def _row(description, qty, unit, total):
    return {
        "description": description,
        "qty": qty,
        "unit_price": unit,
        "total": total,
        "tax_category": "10%",
        "discount": 0,
        "discount_rate": "",
    }


def test_balanced_local_qty_rows_survive_broad_cleanup():
    receipt = {
        "total": 400,
        "subtotal": 400,
        "line_items": [
            _row("商品甲", 2, 100, 200),
            _row("商品乙", 2, 100, 200),
        ],
        "taxes": [],
    }
    text = "\n".join([
        "2026/8/1 12:00",
        "001 商品甲",
        "4900000000001",
        "¥100 2個",
        "¥200",
        "002 商品乙",
        "¥100 2個",
        "¥200",
        "小計",
        "¥400",
    ])

    _fix_line_items(receipt, text)

    assert [(row["qty"], row["unit_price"], row["total"])
            for row in receipt["line_items"]] == [(2, 100, 200), (2, 100, 200)]


def test_purchase_count_cannot_become_a_balanced_row_quantity():
    receipt = {
        "total": 280,
        "subtotal": 280,
        "line_items": [_row("商品甲", 1, 100, 100), _row("商品乙", 1, 180, 180)],
        "taxes": [],
    }
    text = "\n".join([
        "2026/8/1 12:00",
        "商品甲",
        "@100 1点",
        "100",
        "商品乙",
        "@180",
        "1点",
        "お買上点数",
        "9点",
        "合計",
        "180",
        "¥280",
    ])

    _fix_line_items(receipt, text)

    assert [(row["qty"], row["unit_price"], row["total"])
            for row in receipt["line_items"]] == [(1, 100, 100), (1, 180, 180)]


def test_one_character_description_requires_local_arithmetic_and_stops_at_header():
    visible = {"total": 1100, "line_items": []}
    _recover_qty_unit_total_item_from_empty_extraction(
        visible,
        "品\n2個 x 単550\n¥1,100円\n小計\n¥1,100",
    )
    assert visible["line_items"][0]["description"] == "品"

    separated = {"total": 1100, "line_items": []}
    _recover_qty_unit_total_item_from_empty_extraction(
        separated,
        "遠隔の商品名\n2026年8月1日 12:00\n2個 x 単550\n¥1,100円\n小計\n¥1,100",
    )
    assert separated["line_items"] == []


def test_measured_service_row_uses_arithmetic_without_product_keywords():
    receipt = {
        "total": 3390,
        "subtotal": 3082,
        "line_items": [_row("計量サービス", 21.32, 159, 3390)],
        "taxes": [{"rate": "10%", "label": "内税", "amount": 308}],
    }
    text = "\n".join([
        "2026/8/1 12:00",
        "計量サービス",
        "21.32L",
        "159円",
        "合計",
        "¥3,390",
    ])

    _fix_line_items(receipt, text)

    assert receipt["line_items"] == [_row("計量サービス", 1, 3390, 3390)]


@pytest.mark.parametrize("rate_rows", [["@155.0"], ["単価", "155.0"]])
def test_measured_service_row_uses_decimal_labeled_rate_and_adjacent_tax_code(
    rate_rows,
):
    receipt = {
        "total": 1550,
        "subtotal": 1409,
        "line_items": [_row("P-2 (内)", 10, 155, 1550)],
        "taxes": [{"rate": "10%", "label": "内税", "amount": 141}],
    }
    text = "\n".join([
        "2026/8/1 12:00",
        "計量商品甲乙",
        "P-2 (内)",
        "10.00L",
        *rate_rows,
        "合計",
        "¥1,550",
    ])

    _fix_line_items(receipt, text)

    assert receipt["line_items"] == [_row("計量商品甲乙", 1, 1550, 1550)]


def test_measured_service_row_preserves_supported_description_amid_local_names():
    receipt = {
        "merchant": "販売元甲乙",
        "total": 1550,
        "line_items": [_row("計量商品甲乙", 10, 155, 1550)],
    }
    text = "\n".join([
        "別の候補商品",
        "計量商品甲乙",
        "P-2 (内)",
        "10.00L",
        "@155.0",
    ])

    assert _normalize_single_measured_service_row(receipt, text) is True
    assert receipt["line_items"] == [_row("計量商品甲乙", 1, 1550, 1550)]


@pytest.mark.parametrize(
    "merchant",
    ["出光", " 出 光 ", "株式会社出光", "(株)出光", "(有)出光"],
)
def test_measured_service_row_does_not_promote_merchant_as_code_owner(merchant):
    receipt = {
        "merchant": merchant,
        "total": 1550,
        "line_items": [_row("P-2 (内)", 10, 155, 1550)],
    }
    text = "\n".join(["出光", "P-2 (内)", "10.00L", "@155.0"])

    assert _normalize_single_measured_service_row(receipt, text) is False
    assert receipt["line_items"] == [_row("P-2 (内)", 10, 155, 1550)]


@pytest.mark.parametrize(
    "local_rows",
    [
        ["候補商品甲", "候補商品乙", "P-2 (内)", "10.00L", "@155.0"],
        ["候補商品甲", "P-2 (内)", "注記行", "10.00L", "@155.0"],
    ],
)
def test_measured_service_row_does_not_guess_an_unsupported_description(local_rows):
    receipt = {
        "total": 1550,
        "line_items": [_row("P-2 (内)", 10, 155, 1550)],
    }

    assert (
        _normalize_single_measured_service_row(receipt, "\n".join(local_rows))
        is False
    )
    assert receipt["line_items"] == [_row("P-2 (内)", 10, 155, 1550)]


@pytest.mark.parametrize("metadata", ["クレジットご利用額", "レシートNo 123"])
def test_measured_service_row_does_not_use_payment_or_header_as_description(metadata):
    receipt = {
        "total": 1550,
        "line_items": [_row("P-2 (内)", 10, 155, 1550)],
    }
    text = "\n".join([metadata, "P-2 (内)", "10.00L", "@155.0"])

    assert _normalize_single_measured_service_row(receipt, text) is False
    assert receipt["line_items"] == [_row("P-2 (内)", 10, 155, 1550)]


@pytest.mark.parametrize(
    "rate_rows",
    [
        [],
        ["@120.0"],
        ["@155.0", "単価", "155.4"],
    ],
)
def test_measured_service_row_requires_one_matching_local_rate(rate_rows):
    receipt = {
        "total": 1550,
        "line_items": [_row("計量商品甲乙", 10, 155, 1550)],
    }
    text = "\n".join(["計量商品甲乙", "10.00L", *rate_rows])

    assert _normalize_single_measured_service_row(receipt, text) is False
    assert receipt["line_items"] == [_row("計量商品甲乙", 10, 155, 1550)]


def test_gap_recovery_ignores_header_prices_outside_transaction_zone():
    receipt = {
        "total": 300,
        "subtotal": 300,
        "line_items": [_row("既存の商品", 1, 100, 100)],
        "taxes": [],
    }
    text = "\n".join([
        "宣伝の商品",
        "200",
        "2026/8/1 12:00",
        "既存の商品 100",
        "追加の商品 200",
        "小計",
        "300",
    ])

    _recover_missing_items_from_gap(receipt, text)

    assert [(row["description"], row["total"]) for row in receipt["line_items"]] == [
        ("既存の商品", 100),
        ("追加の商品", 200),
    ]


def test_dense_projection_uses_purchase_count_only_as_row_count_evidence():
    receipt = {
        "total": 1500,
        "subtotal": 1500,
        "line_items": [
            _row("商品甲", 1, 100, 100),
            _row("商品丙", 1, 700, 700),
            _row("商品丁", 1, 500, 500),
        ],
        "taxes": [],
    }
    text = "\n".join([
        "2026/8/1",
        "12:00",
        "100001 商品甲 100",
        "200002 商品乙",
        "300003",
        "300003 商品丙 200",
        "300",
        "400",
        "400004 商品丁 500",
        "小計",
        "1,500",
        "お買上点数 5",
    ])

    _replace_dense_sequence_rows_when_balanced(receipt, text)

    assert [(row["description"], row["qty"], row["total"])
            for row in receipt["line_items"]] == [
        ("商品甲", 1, 100),
        ("商品乙", 1, 200),
        ("商品丙", 1, 300),
        ("商品丙", 1, 400),
        ("商品丁", 1, 500),
    ]
