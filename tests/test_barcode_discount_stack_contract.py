import pytest

from receipt_parser.receipt_projection import _project_totals_to_layout_rows


def _layout(*rows):
    blocks = []
    for row_idx, row in enumerate(rows):
        for col_idx, text in enumerate(row):
            x = col_idx * 120
            y = row_idx * 30
            blocks.append({
                "text": text,
                "x": x,
                "y": y,
                "bbox": [[x, y], [x + 90, y], [x + 90, y + 10], [x, y + 10]],
                "page": 0,
            })
    return blocks


def _item(description, unit_price, total, discount=0):
    return {
        "description": description,
        "qty": 1,
        "unit_price": unit_price,
        "total": total,
        "tax_category": "10%",
        "discount": discount,
        "discount_rate": "",
    }


def _barcode_layout(first="商品甲", second="商品乙"):
    return _layout(
        [first],
        ["4900000000001"],
        ["¥", "300"],
        ["割引", "-", "¥", "30"],
        [second],
        ["4900000000002"],
        ["¥", "200"],
        ["小計", "¥", "470"],
    )


def test_barcode_price_stack_owns_following_discount_and_balances_net_rows():
    extracted = {
        "total": 470,
        "subtotal": 470,
        "line_items": [
            {
                "description": "商品甲",
                "qty": 1,
                "unit_price": 200,
                "total": 200,
                "tax_category": "10%",
                "discount": 0,
                "discount_rate": "",
            },
            {
                "description": "商品乙",
                "qty": 1,
                "unit_price": 300,
                "total": 300,
                "tax_category": "10%",
                "discount": 0,
                "discount_rate": "",
            },
        ],
    }
    ocr_layout = _layout(
        ["商品甲"],
        ["4900000000001"],
        ["¥", "300"],
        ["セール", "-", "¥", "30"],
        ["商品乙"],
        ["4900000000002"],
        ["¥", "200"],
        ["小計", "¥", "470"],
    )

    _project_totals_to_layout_rows(extracted, ocr_layout)

    assert [
        (row["description"], row["unit_price"], row["discount"], row["total"])
        for row in extracted["line_items"]
    ] == [
        ("商品甲", 300, 30, 270),
        ("商品乙", 200, 0, 200),
    ]


def test_barcode_projection_repairs_one_wrong_row_with_the_full_money_bundle():
    extracted = {
        "total": 470,
        "subtotal": 470,
        "line_items": [
            _item("商品甲", 100, 100),
            _item("商品乙", 200, 200),
        ],
    }

    _project_totals_to_layout_rows(extracted, _barcode_layout())

    assert [
        (row["unit_price"], row["total"], row["discount"])
        for row in extracted["line_items"]
    ] == [(300, 270, 30), (200, 200, 0)]


def test_barcode_projection_does_not_fall_back_when_descriptions_are_ambiguous():
    extracted = {
        "total": 470,
        "subtotal": 470,
        "line_items": [
            _item("共通商品", 100, 100),
            _item("共通商品", 200, 200),
        ],
    }
    before = [dict(item) for item in extracted["line_items"]]

    _project_totals_to_layout_rows(
        extracted,
        _barcode_layout("共通商品", "共通商品"),
    )

    assert extracted["line_items"] == before


def test_barcode_projection_does_not_fall_back_when_full_layout_is_unbalanced():
    extracted = {
        "total": 500,
        "subtotal": 500,
        "line_items": [
            _item("商品甲", 330, 300, 30),
            _item("商品乙", 100, 100),
        ],
    }
    before = [dict(item) for item in extracted["line_items"]]

    _project_totals_to_layout_rows(extracted, _barcode_layout())

    assert extracted["line_items"] == before


def test_barcode_projection_requires_meaningful_candidate_descriptions():
    extracted = {
        "total": 470,
        "subtotal": 470,
        "line_items": [
            _item("PRODUCT ALPHA", 200, 200),
            _item("PRODUCT BETA", 300, 270, 30),
        ],
    }
    before = [dict(item) for item in extracted["line_items"]]

    _project_totals_to_layout_rows(
        extracted,
        _barcode_layout("A", "PRODUCT BETA"),
    )

    assert extracted["line_items"] == before


@pytest.mark.parametrize("first_new_page_y", [30, 60])
def test_barcode_projection_does_not_borrow_anchor_or_price_across_pages(
    first_new_page_y,
):
    extracted = {
        "total": 470,
        "subtotal": 470,
        "line_items": [
            _item("商品甲", 200, 200),
            _item("商品乙", 300, 270, 30),
        ],
    }
    before = [dict(item) for item in extracted["line_items"]]
    layout = _barcode_layout()
    for block in layout:
        block["page"] = int(block["y"] >= first_new_page_y)

    _project_totals_to_layout_rows(extracted, layout)

    assert extracted["line_items"] == before


@pytest.mark.parametrize(
    "row_ys",
    [
        (0, 1000, 1030, 1060, 2000, 3000, 3030, 3060),
        (0, 30, 1000, 1030, 2000, 2030, 3000, 3030),
    ],
)
def test_barcode_projection_does_not_borrow_far_anchor_or_price(row_ys):
    extracted = {
        "total": 470,
        "subtotal": 470,
        "line_items": [
            _item("商品甲", 200, 200),
            _item("商品乙", 300, 270, 30),
        ],
    }
    before = [dict(item) for item in extracted["line_items"]]
    layout = _barcode_layout()
    moved_rows = dict(zip(sorted({block["y"] for block in layout}), row_ys))
    for block in layout:
        new_y = moved_rows[block["y"]]
        delta = new_y - block["y"]
        block["y"] = new_y
        for point in block["bbox"]:
            point[1] += delta

    _project_totals_to_layout_rows(extracted, layout)

    assert extracted["line_items"] == before
