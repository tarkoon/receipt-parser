import pytest


def _item(description, qty, unit_price, total):
    return {
        "description": description,
        "qty": qty,
        "unit_price": unit_price,
        "total": total,
        "tax_category": "10%",
        "discount": 0,
        "discount_rate": "",
    }


def test_x_qty_extended_yen_proves_unit_only_by_division():
    from receipt_parser.receipt_item_repair import _has_local_qty_unit_evidence

    item = _item("施術", 2, 1230, 2460)

    assert _has_local_qty_unit_evidence(item, "施術\nx 2 2,460円")
    assert not _has_local_qty_unit_evidence(
        _item("施術", 2, 1200, 2400),
        "施術\nx 2 2,460円",
    )


def test_x_qty_extended_yen_vetoes_unsupported_qty_revert():
    from receipt_parser.receipt_item_repair import _revert_unsupported_qty_inflation

    item = _item("訪問施術サービス", 2, 1230, 2460)

    _revert_unsupported_qty_inflation([item], "訪問施術サービス\nx 2 2,460円")

    assert item == _item("訪問施術サービス", 2, 1230, 2460)


def test_vertical_name_unit_count_extended_block_preserves_rows():
    from receipt_parser.receipt_item_repair import (
        _fix_qty_hallucinations,
        _has_local_qty_unit_evidence,
    )
    from receipt_parser.receipt_items import _fix_junk_descriptions

    items = [
        _item("付", 3, 500, 1500),
        _item("小鉢", 2, 200, 400),
    ]
    text = "\n".join([
        "付",
        "¥500",
        "3点",
        "¥1,500",
        "小鉢",
        "¥200",
        "2点",
        "¥400",
    ])

    assert all(_has_local_qty_unit_evidence(item, text) for item in items)
    _fix_junk_descriptions(items, text)
    _fix_qty_hallucinations(items, text)

    assert items == [
        _item("付", 3, 500, 1500),
        _item("小鉢", 2, 200, 400),
    ]


def test_compact_owner_unit_then_count_extended_requires_matching_arithmetic():
    from receipt_parser.receipt_item_repair import _has_local_qty_unit_evidence

    item = _item("付", 2, 1200, 2400)

    assert _has_local_qty_unit_evidence(item, "付  ¥1,200\n2点  ¥2,400")
    assert not _has_local_qty_unit_evidence(item, "付  ¥1,200\n2点  ¥2,500")


def test_compact_qty_block_preserves_valid_one_character_item_through_fix_line_items():
    from receipt_parser.receipt_items import _fix_line_items

    item = _item("付", 2, 200, 400)
    extracted = {
        "total": 400,
        "subtotal": 400,
        "line_items": [item],
    }
    text = "付  ¥200\n2点  ¥400\n小計\n2点\n¥400\n合計\n¥400"

    _fix_line_items(extracted, text)

    assert extracted["line_items"] == [item]


def test_layout_token_spanning_count_and_amount_columns_recovers_qty_by_arithmetic():
    from receipt_parser.receipt_item_repair import _fix_compact_count_amount_layout

    def _block(text, x1, y1, x2):
        return {
            "text": text,
            "x": x1,
            "y": y1,
            "bbox": [[x1, y1], [x2, y1], [x2, y1 + 20], [x1, y1 + 20]],
            "page": 0,
        }

    items = [_item("追加料金", 1, 44, 44)]
    layout = [
        _block("商品名", 120, 100, 220),
        _block("点数", 390, 100, 430),
        _block("金額", 500, 100, 550),
        _block("追加料金", 120, 150, 260),
        _block("244", 430, 150, 560),
    ]

    _fix_compact_count_amount_layout(items, layout)

    assert items == [_item("追加料金", 2, 22, 44)]


def test_layout_count_amount_split_preserves_balanced_basket_total():
    from receipt_parser.receipt_item_repair import _fix_compact_count_amount_layout

    def _block(text, x1, y1, x2):
        return {
            "text": text,
            "x": x1,
            "y": y1,
            "bbox": [[x1, y1], [x2, y1], [x2, y1 + 20], [x1, y1 + 20]],
            "page": 0,
        }

    items = [_item("通常品", 1, 100, 100), _item("追加料金", 1, 244, 244)]
    layout = [
        _block("商品名", 120, 100, 220),
        _block("点数", 390, 100, 430),
        _block("金額", 500, 100, 550),
        _block("追加料金", 120, 150, 260),
        _block("244", 430, 150, 560),
    ]

    _fix_compact_count_amount_layout(items, layout)

    assert items == [
        _item("通常品", 1, 100, 100),
        _item("追加料金", 1, 244, 244),
    ]


def test_repeated_owned_rows_split_collapsed_qty_and_drop_zero_artifact():
    from receipt_parser.receipt_item_repair import _expand_collapsed_items

    extracted = {
        "total": 660,
        "subtotal": 660,
        "line_items": [
            _item("焼菓子", 2, 330, 660),
            _item("0000000000000", 1, 0, 0),
        ],
    }
    text = "\n".join([
        "焼菓子",
        "¥330",
        "焼菓子",
        "¥330",
        "お買上点数 2",
        "合計 ¥660",
    ])

    _expand_collapsed_items(extracted, text)

    assert extracted["line_items"] == [
        _item("焼菓子", 1, 330.0, 330.0),
        _item("焼菓子", 1, 330.0, 330.0),
    ]


def test_repeated_owned_row_splits_among_siblings_when_visible_zero_explains_count():
    from receipt_parser.receipt_items import _fix_line_items

    extracted = {
        "total": 2160,
        "subtotal": 2160,
        "line_items": [
            _item("商品甲", 1, 1000, 1000),
            _item("包装サービス", 2, 330, 660),
            _item("商品乙", 1, 500, 500),
            _item("CODE-01", 1, 0, 0),
        ],
    }
    text = "\n".join([
        "10%対象 ¥2,160",
        "消費税 ¥196",
        "商品甲",
        "¥1,000",
        "包装サービス",
        "¥330",
        "包装サービス",
        "¥330",
        "商品乙",
        "¥500",
        "CODE-01",
        "¥0",
        "小計",
        "¥2,160",
        "点数 5",
    ])

    _fix_line_items(extracted, text)

    assert [(row["description"], row["qty"], row["total"])
            for row in extracted["line_items"]] == [
        ("商品甲", 1, 1000),
        ("包装サービス", 1, 330.0),
        ("包装サービス", 1, 330.0),
        ("商品乙", 1, 500),
    ]


def test_duplicate_description_repair_rejects_detached_title_amount_queue():
    from receipt_parser.receipt_items import _fix_duplicate_descriptions_from_ocr

    items = [
        _item("抽出商品", 1, 847, 847),
        _item("抽出商品", 1, 825, 825),
    ]
    before = [dict(item) for item in items]
    text = "\n".join([
        "1111111111111",
        "商品甲",
        "2222222222222",
        "商品乙",
        "847",
        "825",
        "合計 1,672",
    ])

    _fix_duplicate_descriptions_from_ocr({"line_items": items}, text)

    assert items == before


@pytest.mark.parametrize(
    "text",
    [
        "商品甲 ¥847\n商品乙 ¥825",
        "商品甲\n¥847\n商品乙\n¥825",
    ],
)
def test_duplicate_description_repair_keeps_unambiguous_row_ownership(text):
    from receipt_parser.receipt_items import _fix_duplicate_descriptions_from_ocr

    items = [
        _item("抽出商品", 1, 847, 847),
        _item("抽出商品", 1, 825, 825),
    ]

    _fix_duplicate_descriptions_from_ocr({"line_items": items}, text)

    assert [item["description"] for item in items] == ["商品甲", "商品乙"]
