"""Synthetic contracts for geometry-owned item reconstruction."""


def _item(description, total):
    return {
        "description": description,
        "qty": 1,
        "unit_price": total,
        "total": total,
        "tax_category": "10%",
        "discount": 0,
        "discount_rate": "",
    }


def _layout_rows(*rows):
    blocks = []
    for y, description, value in rows:
        blocks.extend([
            {
                "text": description,
                "x": 10,
                "y": y,
                "bbox": [[10, y], [150, y], [150, y + 15], [10, y + 15]],
            },
            {
                "text": str(value),
                "x": 220,
                "y": y,
                "bbox": [[220, y], [260, y], [260, y + 15], [220, y + 15]],
            },
        ])
    return blocks


def _split_layout_rows(*rows):
    blocks = []
    for y, description, code, amount in rows:
        unit, total = amount if isinstance(amount, tuple) else (amount, amount)
        blocks.append({
            "text": description,
            "x": 100,
            "y": y,
            "bbox": [[100, y], [260, y], [260, y + 15], [100, y + 15]],
        })
        for text, x in ((code, 160), ("1", 350), (str(unit), 450), (str(total), 620)):
            blocks.append({
                "text": text,
                "x": x,
                "y": y + 25,
                "bbox": [[x, y + 25], [x + 40, y + 25], [x + 40, y + 40], [x, y + 40]],
            })
    return blocks


def _catalog_layout_rows(*rows):
    blocks = []
    for y, cells in rows:
        for text, x in cells:
            blocks.append({
                "text": text,
                "x": x,
                "y": y,
                "bbox": [[x, y], [x + 40, y], [x + 40, y + 15], [x, y + 15]],
            })
    return blocks


def test_balanced_layout_projection_repairs_pure_two_row_swap_with_exact_multiset():
    from receipt_parser.receipt_projection import _project_totals_to_layout_rows

    extracted = {
        "subtotal": 400,
        "total": 400,
        "taxes": [],
        "line_items": [_item("商品甲ロング", 240), _item("商品乙ロング", 160)],
    }
    layout = _layout_rows(
        (10, "商品甲ロング", 160),
        (35, "商品乙ロング", 240),
        (60, "合計", 400),
    )

    _project_totals_to_layout_rows(extracted, layout)

    assert [item["total"] for item in extracted["line_items"]] == [160, 240]
    assert [item["unit_price"] for item in extracted["line_items"]] == [160, 240]


def test_balanced_cleanup_projects_unique_partial_layout_subset_despite_noisy_header():
    from receipt_parser.receipt_items import _fix_line_items

    extracted = {
        "subtotal": 600,
        "total": 600,
        "taxes": [],
        "line_items": [
            _item("商品甲ロング", 200),
            _item("商品乙ロング", 100),
            _item("商品丙ロング", 300),
        ],
    }
    layout = _layout_rows(
        (5, "レジ番号", 7),
        (30, "商品甲ロング", 100),
        (55, "商品乙ロング", 200),
        (80, "合計", 600),
    )

    _fix_line_items(
        extracted,
        "商品甲ロング\n100\n商品乙ロング\n200\n商品丙ロング\n300\n合計\n600",
        layout,
    )

    assert [item["total"] for item in extracted["line_items"]] == [100, 200, 300]


def test_balanced_layout_projection_rejects_ambiguous_duplicate_descriptions():
    from receipt_parser.receipt_projection import _project_totals_to_layout_rows

    extracted = {
        "subtotal": 300,
        "total": 300,
        "taxes": [],
        "line_items": [_item("同じ商品", 200), _item("同じ商品", 100)],
    }
    before = [dict(item) for item in extracted["line_items"]]
    layout = _layout_rows(
        (10, "同じ商品", 100),
        (35, "同じ商品", 200),
        (60, "合計", 300),
    )

    _project_totals_to_layout_rows(extracted, layout)

    assert extracted["line_items"] == before


def test_balanced_layout_projection_reconciles_unique_row_descriptions_only():
    from receipt_parser.receipt_projection import _project_totals_to_layout_rows

    extracted = {
        "subtotal": 3670,
        "total": 3670,
        "taxes": [{"rate": "10%", "amount": 334}],
        "line_items": [
            _item("何回投げても大丈夫 fino 二個セット", 800),
            _item("ボリューム エクスプレ ロング", 1380),
            _item("ボリューム エクスプレ ロング", 1490),
        ],
    }
    preserved = [
        {key: value for key, value in item.items() if key != "description"}
        for item in extracted["line_items"]
    ]

    _project_totals_to_layout_rows(
        extracted,
        _layout_rows(
            (10, "☆何回投げても大丈夫", 800),
            (35, "fino 二個セット", 1380),
            (60, "ボリューム エクスプレ", 1490),
            (85, "合計", 3670),
        ),
    )

    assert [item["description"] for item in extracted["line_items"]] == [
        "何回投げても大丈夫",
        "fino 二個セット",
        "ボリューム エクスプレ ロング",
    ]
    assert [
        {key: value for key, value in item.items() if key != "description"}
        for item in extracted["line_items"]
    ] == preserved


def test_balanced_layout_projection_ignores_detached_right_column_noise():
    from receipt_parser.receipt_projection import _project_totals_to_layout_rows

    extracted = {
        "subtotal": 473,
        "total": 473,
        "taxes": [],
        "line_items": [
            _item("BIARANIPIG", 193),
            _item("クラフトボスイタリアー", 140),
            _item("アップルティーソーダ", 140),
        ],
    }
    layout = [{
        "text": "BIARANIPIG",
        "x": 1046,
        "y": 0,
        "bbox": [[1046, 0], [1160, 0], [1160, 15], [1046, 15]],
    }]
    layout.extend(_layout_rows(
        (10, "レジ番号", 3),
        (25, "クラフトボスイタリアー", 193),
        (50, "◎アップルティーソーダ", 140),
        (75, "トロピカーナアップル", 140),
        (100, "合計", 473),
    ))
    preserved = [
        {key: value for key, value in item.items() if key != "description"}
        for item in extracted["line_items"]
    ]

    _project_totals_to_layout_rows(extracted, layout)

    assert [item["description"] for item in extracted["line_items"]] == [
        "クラフトボスイタリアー",
        "アップルティーソーダ",
        "トロピカーナアップル",
    ]
    assert [
        {key: value for key, value in item.items() if key != "description"}
        for item in extracted["line_items"]
    ] == preserved


def test_balanced_layout_description_projection_rejects_ambiguous_rows():
    from receipt_parser.receipt_projection import _project_totals_to_layout_rows

    extracted = {
        "subtotal": 300,
        "total": 300,
        "taxes": [],
        "line_items": [_item("誤った商品甲", 200), _item("誤った商品乙", 100)],
    }
    before = [dict(item) for item in extracted["line_items"]]

    _project_totals_to_layout_rows(
        extracted,
        _layout_rows(
            (10, "候補商品甲", 200),
            (35, "候補商品乙", 100),
            (60, "別候補商品甲", 200),
            (85, "別候補商品乙", 100),
        ),
    )

    assert extracted["line_items"] == before


def test_balanced_layout_description_projection_requires_two_unsupported_rows():
    from receipt_parser.receipt_projection import _project_totals_to_layout_rows

    extracted = {
        "subtotal": 300,
        "total": 300,
        "taxes": [],
        "line_items": [_item("商品甲ロング", 200), _item("誤った商品乙", 100)],
    }
    before = [dict(item) for item in extracted["line_items"]]

    _project_totals_to_layout_rows(
        extracted,
        _layout_rows(
            (10, "商品甲", 200),
            (35, "正しい商品乙", 100),
            (60, "合計", 300),
        ),
    )

    assert extracted["line_items"] == before


def test_balanced_layout_description_projection_rejects_all_unmatched_names():
    from receipt_parser.receipt_projection import _project_totals_to_layout_rows

    for descriptions in (
        ("赤い果物", "青い飲料", "白い菓子"),
        ("赤い果物", "赤い果物", "赤い果物"),
    ):
        extracted = {
            "subtotal": 350,
            "total": 350,
            "taxes": [],
            "line_items": [
                _item(description, total)
                for description, total in zip(
                    descriptions, (200, 100, 50), strict=False
                )
            ],
        }
        before = [dict(item) for item in extracted["line_items"]]

        _project_totals_to_layout_rows(
            extracted,
            _layout_rows(
                (10, "乾電池パック", 200),
                (35, "洗濯用せっけん", 100),
                (60, "台所スポンジ", 50),
                (85, "合計", 350),
            ),
        )

        assert extracted["line_items"] == before


def test_balanced_layout_description_projection_rejects_nonlocal_title_rows():
    from receipt_parser.receipt_projection import _project_totals_to_layout_rows

    for mode in ("page", "distance"):
        extracted = {
            "subtotal": 360,
            "total": 360,
            "taxes": [],
            "line_items": [_item("誤った商品甲", 120), _item("誤った商品乙", 240)],
        }
        before = [dict(item) for item in extracted["line_items"]]
        layout = _split_layout_rows(
            (10, "PRODUCT ALPHA", "1111111", 120),
            (60, "PRODUCT BETA", "2222222", 240),
        )
        if mode == "page":
            for block in layout:
                block["page"] = 0 if block["y"] in {10, 60} else 1
        else:
            for block in layout:
                if block["y"] not in {35, 85}:
                    continue
                delta = 1000 - block["y"]
                block["y"] += delta
                for point in block["bbox"]:
                    point[1] += delta

        _project_totals_to_layout_rows(extracted, layout)

        assert extracted["line_items"] == before


def test_balanced_layout_description_projection_requires_plain_qty_one_rows():
    from receipt_parser.receipt_projection import _project_totals_to_layout_rows

    for override in (
        {"qty": 2, "unit_price": 100},
        {"unit_price": 210, "discount": 10},
        {"discount_rate": "5%"},
    ):
        extracted = {
            "subtotal": 300,
            "total": 300,
            "taxes": [],
            "line_items": [_item("誤った商品甲", 200), _item("誤った商品乙", 100)],
        }
        extracted["line_items"][0].update(override)
        before = [dict(item) for item in extracted["line_items"]]

        _project_totals_to_layout_rows(
            extracted,
            _layout_rows(
                (10, "正しい商品甲", 200),
                (35, "正しい商品乙", 100),
                (60, "合計", 300),
            ),
        )

        assert extracted["line_items"] == before


def test_balanced_layout_projection_owns_ascii_description_from_adjacent_detail_row():
    from receipt_parser.receipt_projection import _project_totals_to_layout_rows

    extracted = {
        "subtotal": 360,
        "total": 360,
        "taxes": [],
        "line_items": [_item("PRODUCT ALPHA", 240), _item("PRODUCT BETA", 120)],
    }
    layout = _split_layout_rows(
        (10, "PRODUCT ALPHA", "1111111", 120),
        (60, "PRODUCT BETA", "2222222", 240),
    )

    _project_totals_to_layout_rows(extracted, layout)

    assert [item["total"] for item in extracted["line_items"]] == [120, 240]
    assert [item["unit_price"] for item in extracted["line_items"]] == [120, 240]


def test_barcode_projection_recovers_missing_discounted_unit_in_exact_bundle_swap():
    from receipt_parser.receipt_projection import _project_totals_to_layout_rows

    extracted = {
        "subtotal": 300,
        "total": 300,
        "taxes": [],
        "line_items": [_item("ALPHA ITEM", 200), _item("BETA ITEM", 100)],
    }
    extracted["line_items"][1].update(unit_price=None, discount=30)
    layout = _catalog_layout_rows(
        (10, (("ALPHA ITEM", 100),)),
        (35, (("4900000000001", 100),)),
        (60, (("¥130", 500),)),
        (85, (("-¥30", 500),)),
        (110, (("BETA ITEM", 100),)),
        (135, (("4900000000002", 100),)),
        (160, (("¥200", 500),)),
        (185, (("合計", 100), ("300", 500))),
    )

    _project_totals_to_layout_rows(extracted, layout)

    assert [
        (item["unit_price"], item["total"], item["discount"])
        for item in extracted["line_items"]
    ] == [(130.0, 100.0, 30.0), (200.0, 200.0, 0.0)]

    near_miss = {
        "subtotal": 300,
        "total": 300,
        "taxes": [],
        "line_items": [_item("ALPHA ITEM", 200), _item("BETA ITEM", 100)],
    }
    near_miss["line_items"][1].update(unit_price=None, discount=29)
    before = [dict(item) for item in near_miss["line_items"]]

    _project_totals_to_layout_rows(near_miss, layout)

    assert near_miss["line_items"] == before


def test_mixed_catalog_layout_owns_structural_sales_and_excludes_reference_metadata():
    from receipt_parser.receipt_projection import (
        _layout_row_price_candidates,
        _project_totals_to_layout_rows,
    )

    layout = _catalog_layout_rows(
        (10, (("センサーホワイト", 20),)),
        (35, (("1", 150), ("*", 190), ("549", 250), ("549", 390), ("0", 540))),
        (60, (("通常価格", 20), ("799", 470))),
        (85, (("商品名", 20), ("20595575", 170), ("16775", 420))),
        (110, (("MODL?ALPHA", 20), ("499", 510), ("0", 590))),
        (135, (("MODL?ALPHA 赤いケース用の長い説明テキスト", 20),)),
        (160, (("商品ID", 20), ("40513890", 170), ("21576", 420))),
        (185, (("MODEL BETA", 20), ("1499", 500))),
        (210, (("MODEL BETA 青い工具", 20),)),
        (235, (("ケーブルブルー", 20),)),
        (260, (("1", 150), ("*", 190), ("1039", 240), ("1039", 390))),
        (285, (("合計", 20), ("3586", 460))),
    )

    assert [
        (candidate["description"], candidate["value"])
        for candidate in _layout_row_price_candidates(layout)
    ] == [
        ("センサーホワイト", 549),
        ("MODL?ALPHA 赤いケース用の長い説明テキスト", 499),
        ("MODEL BETA 青い工具", 1499),
        ("ケーブルブルー", 1039),
    ]

    extracted = {
        "subtotal": 3586,
        "total": 3586,
        "taxes": [],
        "line_items": [
            _item(description, amount)
            for description, amount in (
                ("センサーホワイト", 549),
                ("MODLÄALPHA", 499),
                ("MODEL BETA 青い工具", 799),
                ("ケーブルブルー", 1039),
            )
        ],
    }

    _project_totals_to_layout_rows(extracted, layout)

    assert len(extracted["line_items"]) == 4
    assert [item["total"] for item in extracted["line_items"]] == [549, 499, 1499, 1039]


def test_inline_catalog_layout_requires_local_repeated_description():
    from receipt_parser.receipt_projection import _layout_row_price_candidates

    layout = _catalog_layout_rows(
        (10, (("MODEL ALPHA", 20), ("499", 510))),
        (35, (("DIFFERENT PRODUCT 赤いケース", 20),)),
        (60, (("MODEL BETA", 20), ("1499", 500))),
        (200, (("MODEL BETA 青い工具", 20),)),
        (225, (("合計", 20), ("1998", 460))),
    )

    assert _layout_row_price_candidates(layout) == []


def test_adjacent_layout_projection_rejects_non_permutation_amounts():
    from receipt_parser.receipt_projection import _project_totals_to_layout_rows

    extracted = {
        "subtotal": 360,
        "total": 360,
        "taxes": [],
        "line_items": [_item("PRODUCT ALPHA", 240), _item("PRODUCT BETA", 120)],
    }
    before = [dict(item) for item in extracted["line_items"]]
    layout = _split_layout_rows(
        (10, "PRODUCT ALPHA", "1111111", 140),
        (60, "PRODUCT BETA", "2222222", 220),
    )

    _project_totals_to_layout_rows(extracted, layout)

    assert extracted["line_items"] == before


def test_adjacent_layout_projection_requires_printed_qty_one():
    from receipt_parser.receipt_projection import _project_totals_to_layout_rows

    extracted = {
        "subtotal": 360,
        "total": 360,
        "taxes": [],
        "line_items": [_item("PRODUCT ALPHA", 240), _item("PRODUCT BETA", 120)],
    }
    before = [dict(item) for item in extracted["line_items"]]
    layout = _split_layout_rows(
        (10, "PRODUCT ALPHA", "1111111", 120),
        (60, "PRODUCT BETA", "2222222", 240),
    )
    for block in layout:
        if block["x"] == 350 and block["y"] == 35:
            block["text"] = "2"

    _project_totals_to_layout_rows(extracted, layout)

    assert extracted["line_items"] == before


def test_adjacent_layout_projection_does_not_borrow_title_across_pages():
    from receipt_parser.receipt_projection import _project_totals_to_layout_rows

    extracted = {
        "subtotal": 360,
        "total": 360,
        "taxes": [],
        "line_items": [_item("PRODUCT ALPHA", 240), _item("PRODUCT BETA", 120)],
    }
    before = [dict(item) for item in extracted["line_items"]]
    layout = _split_layout_rows(
        (10, "PRODUCT ALPHA", "1111111", 120),
        (60, "PRODUCT BETA", "2222222", 240),
    )
    for block in layout:
        block["page"] = 0 if block["y"] == 10 else 1

    _project_totals_to_layout_rows(extracted, layout)

    assert extracted["line_items"] == before


def test_adjacent_layout_projection_does_not_borrow_far_away_title():
    from receipt_parser.receipt_projection import _project_totals_to_layout_rows

    extracted = {
        "subtotal": 360,
        "total": 360,
        "taxes": [],
        "line_items": [_item("PRODUCT ALPHA", 240), _item("PRODUCT BETA", 120)],
    }
    before = [dict(item) for item in extracted["line_items"]]
    layout = _split_layout_rows(
        (0, "PRODUCT ALPHA", "1111111", 120),
        (2000, "PRODUCT BETA", "2222222", 240),
    )
    moved_rows = {25: 1000, 2025: 3000}
    for block in layout:
        new_y = moved_rows.get(block["y"])
        if new_y is None:
            continue
        delta = new_y - block["y"]
        block["y"] = new_y
        for point in block["bbox"]:
            point[1] += delta

    _project_totals_to_layout_rows(extracted, layout)

    assert extracted["line_items"] == before


def test_adjacent_layout_projection_rejects_ambiguous_description_or_detail_arithmetic():
    from receipt_parser.receipt_projection import _project_totals_to_layout_rows

    for descriptions, amounts in (
        (("PRODUCT ALPHA", "PRODUCT ALPHA"), (120, 240)),
        (("PRODUCT ALPHA", "PRODUCT BETA"), ((999, 120), 240)),
    ):
        extracted = {
            "subtotal": 360,
            "total": 360,
            "taxes": [],
            "line_items": [_item(descriptions[0], 240), _item(descriptions[1], 120)],
        }
        before = [dict(item) for item in extracted["line_items"]]

        _project_totals_to_layout_rows(
            extracted,
            _split_layout_rows(
                (10, descriptions[0], "1111111", amounts[0]),
                (60, descriptions[1], "2222222", amounts[1]),
            ),
        )

        assert extracted["line_items"] == before


def test_service_table_layout_quantity_repair_runs_after_reconstruction(monkeypatch):
    import receipt_parser.receipt_postprocess_phases as phases
    from receipt_parser.receipt_phase_trace import POSTPROCESS_PHASE_BY_NAME

    extracted = {"line_items": [_item("BEFORE", 100)]}
    layout = [{"text": "layout"}]
    calls = []

    def reconstruct(result, _text):
        calls.append("reconstruct")
        result["line_items"] = [_item("AFTER", 100)]

    def repair(items, blocks):
        calls.append((items[0]["description"], blocks))

    monkeypatch.setattr(phases, "_replace_service_table_items_when_balanced", reconstruct)
    monkeypatch.setattr(phases, "_fix_compact_count_amount_layout", repair)

    phases._run_service_receipt_recovery_phase(
        extracted,
        "service table",
        ("service_table_items",),
        ocr_layout_blocks=layout,
    )

    assert calls == ["reconstruct", ("AFTER", layout)]
    assert "ocr_layout_blocks" in POSTPROCESS_PHASE_BY_NAME[
        "service_receipt_recovery"
    ]["reads"]


def test_service_table_text_split_preserves_printed_count_and_balanced_total():
    from receipt_parser.receipt_row_projection import (
        _replace_service_table_items_when_balanced,
    )

    extracted = {
        "total": 344,
        "subtotal": 344,
        "taxes": [],
        "line_items": [_item("通常商品", 300), _item("特別付加金", 44)],
    }
    text = "\n".join([
        "商品名",
        "点数",
        "金額",
        "通常商品",
        "300",
        "物販",
        "特別付加金",
        "244",
        "小計 通常 1点 物販 2点",
        "344",
    ])

    _replace_service_table_items_when_balanced(extracted, text)

    assert extracted["line_items"][-1] == {
        "description": "特別付加金",
        "qty": 2.0,
        "unit_price": 22.0,
        "total": 44.0,
        "tax_category": "10%",
        "discount": 0,
        "discount_rate": "",
    }
    assert sum(item["total"] for item in extracted["line_items"]) == extracted["total"]


def test_service_table_coupon_owned_negative_preserves_gross_discount_bundle():
    from receipt_parser.receipt_postprocess import postprocess_receipt

    extracted = {
        "document_type": "receipt",
        "total": 344,
        "subtotal": 344,
        "taxes": [],
        "line_items": [
            _item("通常商品", 300),
            {
                **_item("特別付加金", 44),
                "unit_price": 244,
                "discount": 200,
            },
        ],
    }
    text = "\n".join([
        "商品名",
        "点数",
        "金額",
        "通常商品",
        "300",
        "物販",
        "特別付加金",
        "244",
        "クーポン",
        "-200",
        "小計 通常 1点 物販 2点",
        "344",
    ])

    result = postprocess_receipt(extracted, text, 1.0, {}, {}, "test")

    assert result["line_items"][-1] == {
        "description": "特別付加金",
        "qty": 1.0,
        "unit_price": 244.0,
        "total": 44.0,
        "tax_category": "10%",
        "discount": 200.0,
        "discount_rate": "",
    }


def test_service_table_unconsumed_owned_negative_blocks_compact_split():
    from receipt_parser.receipt_postprocess import postprocess_receipt

    for discount_lines in (
        ["クーポン利用", "-200"],
        ["クーポン利用 -200"],
        ["クーポン利用-200"],
    ):
        extracted = {
            "document_type": "receipt",
            "total": 344,
            "subtotal": 344,
            "taxes": [],
            "line_items": [
                _item("通常商品", 300),
                {
                    **_item("特別付加金", 44),
                    "unit_price": 244,
                    "discount": 200,
                },
            ],
        }
        text = "\n".join([
            "商品名",
            "点数",
            "金額",
            "通常商品",
            "300",
            "物販",
            "特別付加金",
            "244",
            *discount_lines,
            "小計 通常 1点 物販 2点",
            "344",
        ])

        result = postprocess_receipt(extracted, text, 1.0, {}, {}, "test")

        assert result["line_items"][-1] == {
            "description": "特別付加金",
            "qty": 1.0,
            "unit_price": 244.0,
            "total": 44.0,
            "tax_category": "10%",
            "discount": 200.0,
            "discount_rate": "",
        }


def test_service_table_compact_split_ignores_negative_owned_by_previous_row():
    from receipt_parser.receipt_row_projection import (
        _replace_service_table_items_when_balanced,
    )

    extracted = {
        "total": 344,
        "subtotal": 344,
        "taxes": [],
        "line_items": [_item("通常商品", 300), _item("特別付加金", 44)],
    }
    text = "\n".join([
        "商品名",
        "点数",
        "金額",
        "通常商品",
        "300",
        "-200",
        "物販",
        "特別付加金",
        "244",
        "小計 通常 1点 物販 2点",
        "344",
    ])

    _replace_service_table_items_when_balanced(extracted, text)

    assert extracted["line_items"][-1] == {
        "description": "特別付加金",
        "qty": 2.0,
        "unit_price": 22.0,
        "total": 44.0,
        "tax_category": "10%",
        "discount": 0,
        "discount_rate": "",
    }


def test_service_table_text_split_rejects_two_balancing_group_owned_tokens():
    from receipt_parser.receipt_row_projection import (
        _replace_service_table_items_when_balanced,
    )

    before = [_item("通常商品", 300), _item("特別付加金", 44), _item("追加料金", 234)]
    for item in before:
        item["tax_category"] = "8%"
    extracted = {
        "total": 578,
        "subtotal": 578,
        "taxes": [],
        "line_items": [dict(item) for item in before],
    }
    text = "\n".join([
        "商品名",
        "点数",
        "金額",
        "通常商品",
        "300",
        "物販",
        "特別付加金",
        "244",
        "追加区分",
        "追加料金",
        "234",
        "小計 通常 1点 物販 2点 追加区分 2点",
        "578",
    ])

    _replace_service_table_items_when_balanced(extracted, text)

    assert extracted["line_items"] == before


def test_service_table_text_split_rejects_nondivisible_corrected_suffix():
    from receipt_parser.receipt_row_projection import (
        _replace_service_table_items_when_balanced,
    )

    before = [_item("通常商品", 300), _item("特別付加金", 45)]
    for item in before:
        item["tax_category"] = "8%"
    extracted = {
        "total": 345,
        "subtotal": 345,
        "taxes": [],
        "line_items": [dict(item) for item in before],
    }
    text = "\n".join([
        "商品名",
        "点数",
        "金額",
        "通常商品",
        "300",
        "物販",
        "特別付加金",
        "245",
        "小計 通常 1点 物販 2点",
        "345",
    ])

    _replace_service_table_items_when_balanced(extracted, text)

    assert extracted["line_items"] == before


def test_final_output_projects_layout_after_late_missing_item_recovery_and_traces_owner(
    monkeypatch,
):
    import receipt_parser.receipt_output as output
    from receipt_parser.receipt_phase_trace import POSTPROCESS_PHASE_BY_NAME

    extracted = {
        "document_type": "receipt",
        "subtotal": 600,
        "total": 600,
        "taxes": [],
        "line_items": [_item("商品甲ロング", 200), _item("商品乙ロング", 100)],
    }
    layout = _layout_rows(
        (5, "レジ番号", 7),
        (30, "商品甲ロング", 100),
        (55, "商品乙ロング", 200),
        (80, "合計", 600),
    )

    def recover_missing_item(result, _ocr_text):
        result["line_items"].append(_item("商品丙ロング", 300))

    monkeypatch.setattr(output, "_recover_missing_items_from_gap", recover_missing_item)
    trace = []

    output._apply_final_receipt_output_repairs(
        extracted,
        "領収書",
        mutation_trace=trace,
        ocr_layout_blocks=layout,
    )

    assert [item["total"] for item in extracted["line_items"]] == [100, 200, 300]
    event = next(
        entry
        for entry in trace
        if entry["stage"] == "layout_row_price_permutation"
    )
    assert event["owner_phase"] == "item_cleanup"
    assert "ocr_layout_blocks" in POSTPROCESS_PHASE_BY_NAME[event["owner_phase"]]["reads"]
    assert set(event["changes"]) == {"line_items"}


def test_final_output_projects_layout_after_late_numeric_placeholder_rename(monkeypatch):
    import receipt_parser.receipt_output as output

    extracted = {
        "document_type": "receipt",
        "subtotal": 300,
        "total": 300,
        "taxes": [],
        "line_items": [_item("商品甲ロング", 200), _item("123", 100)],
    }
    layout = _layout_rows(
        (10, "商品甲ロング", 100),
        (35, "商品乙ロング", 200),
        (60, "合計", 300),
    )

    def rename_placeholder(result, _ocr_text, repairs):
        if "pre_price_stack_descriptions" in repairs:
            result["line_items"][1]["description"] = "商品乙ロング"

    monkeypatch.setattr(
        output,
        "_run_final_ocr_description_reconciliation_phase",
        rename_placeholder,
    )

    output._apply_final_receipt_output_repairs(
        extracted,
        "領収書",
        ocr_layout_blocks=layout,
    )

    assert [item["description"] for item in extracted["line_items"]] == [
        "商品甲ロング",
        "商品乙ロング",
    ]
    assert [item["total"] for item in extracted["line_items"]] == [100, 200]


def test_final_output_layout_permutation_without_layout_is_noop(monkeypatch):
    import receipt_parser.receipt_output as output

    extracted = {
        "document_type": "receipt",
        "subtotal": 300,
        "total": 300,
        "taxes": [],
        "line_items": [_item("商品甲ロング", 200), _item("123", 100)],
    }

    def rename_placeholder(result, _ocr_text, repairs):
        if "pre_price_stack_descriptions" in repairs:
            result["line_items"][1]["description"] = "商品乙ロング"

    monkeypatch.setattr(
        output,
        "_run_final_ocr_description_reconciliation_phase",
        rename_placeholder,
    )
    trace = []

    output._apply_final_receipt_output_repairs(
        extracted,
        "領収書",
        mutation_trace=trace,
    )

    assert [item["total"] for item in extracted["line_items"]] == [200, 100]
    assert not [
        entry
        for entry in trace
        if entry["stage"] == "layout_row_price_permutation"
    ]


def test_final_output_layout_permutation_rejects_ambiguous_duplicate_descriptions():
    from receipt_parser.receipt_output import _apply_final_receipt_output_repairs

    extracted = {
        "document_type": "receipt",
        "subtotal": 300,
        "total": 300,
        "taxes": [],
        "line_items": [_item("同じ商品", 200), _item("同じ商品", 100)],
    }
    layout = _layout_rows(
        (10, "同じ商品", 100),
        (35, "同じ商品", 200),
        (60, "合計", 300),
    )
    trace = []

    _apply_final_receipt_output_repairs(
        extracted,
        "領収書",
        mutation_trace=trace,
        ocr_layout_blocks=layout,
    )

    assert [item["total"] for item in extracted["line_items"]] == [200, 100]
    assert not [
        entry
        for entry in trace
        if entry["stage"] == "layout_row_price_permutation"
    ]


def test_neighborhood_price_skips_unmarked_single_digit_count():
    from receipt_parser.receipt_projection import (
        _fix_item_totals_from_ocr_neighborhood,
    )

    items = [_item("商品甲の長い名前", 1), _item("商品乙の長い名前", 200)]
    text = "\n".join([
        "商品甲の長い名前",
        "1",
        "100*",
        "商品乙の長い名前",
        "200*",
        "小計",
        "300",
    ])

    _fix_item_totals_from_ocr_neighborhood(items, text, 300, 300)

    assert [item["total"] for item in items] == [100.0, 200]


def test_gap_recovery_rejects_only_locally_owned_multiline_fragments():
    from receipt_parser.receipt_recovery import _recover_missing_items_from_gap

    blocked = {
        "total": 300,
        "subtotal": 300,
        "taxes": [],
        "line_items": [_item("長い商品名の前半後半", 100)],
    }
    blocked_before = [dict(item) for item in blocked["line_items"]]
    _recover_missing_items_from_gap(
        blocked,
        "\n".join([
            "2099/1/1 00:00",
            "111-000001-000 長い商品名の前半",
            "後半",
            "100",
            "200",
            "小計",
            "300",
        ]),
    )
    assert blocked["line_items"] == blocked_before

    recoverable = {
        "total": 300,
        "subtotal": 300,
        "taxes": [],
        "line_items": [_item("特別商品セット", 100)],
    }
    _recover_missing_items_from_gap(
        recoverable,
        "\n".join([
            "2099/1/1 00:00",
            "特別商品セット 100",
            "特別商品",
            "200",
            "小計",
            "300",
        ]),
    )
    assert [item["description"] for item in recoverable["line_items"]] == [
        "特別商品セット",
        "特別商品",
    ]


def test_code_anchored_queue_projects_atomically_only_for_one_balanced_subset():
    from receipt_parser.receipt_recovery import _recover_missing_items_from_gap
    from receipt_parser.receipt_row_projection import (
        _replace_dense_sequence_rows_when_balanced,
    )

    extracted = {
        "total": 330,
        "taxes": [{"rate": "10%", "label": "外税", "amount": 30}],
        "line_items": [_item("未解析", 330)],
    }
    text = "\n".join([
        "2099/1/1 00:00",
        "111-000001 商品の前半",
        "後半",
        "1",
        "4900000000001 1",
        "120*",
        "別の商品",
        "4900000000002",
        "第三の商品",
        "80※",
        "100内",
        "合計",
        "330",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, text)

    assert [item["description"] for item in extracted["line_items"]] == [
        "商品の前半後半",
        "別の商品",
        "第三の商品",
    ]
    assert [item["total"] for item in extracted["line_items"]] == [
        120.0,
        80.0,
        100.0,
    ]

    ambiguous_before = [_item("既存行", 90)]
    ambiguous = {
        "subtotal": 110,
        "total": 130,
        "taxes": [],
        "line_items": [dict(item) for item in ambiguous_before],
    }
    _replace_dense_sequence_rows_when_balanced(
        ambiguous,
        "\n".join([
            "2099/1/1 00:00",
            "111-000001 商品甲",
            "222-000002 商品乙",
            "50*",
            "60*",
            "70*",
            "小計",
        ]),
    )
    assert ambiguous["line_items"] == ambiguous_before

    deferred_before = [_item("既存行", 100)]
    deferred = {
        "subtotal": 300,
        "total": 300,
        "taxes": [],
        "line_items": [dict(item) for item in deferred_before],
    }
    _recover_missing_items_from_gap(
        deferred,
        "\n".join([
            "2099/1/1 00:00",
            "111-000001 商品甲",
            "1",
            "222-000002 商品乙",
            "100",
            "200",
            "小計",
            "300",
        ]),
    )
    assert deferred["line_items"] == deferred_before


def test_code_anchored_queue_rejects_gross_total_when_external_tax_is_positive():
    from receipt_parser.receipt_row_projection import (
        _replace_dense_sequence_rows_when_balanced,
    )

    existing = [_item("正しい商品甲", 980), _item("正しい商品乙", 448)]
    extracted = {
        "subtotal": 1428,
        "total": 1570,
        "taxes": [{"rate": "10%", "label": "外税", "amount": 142}],
        "line_items": [dict(item) for item in existing],
    }
    text = "\n".join([
        "2099/1/1 00:00",
        "正しい商品甲",
        "980",
        "4571511080677",
        "誤投影商品甲",
        "448",
        "4901490320141",
        "担当 2点",
        "1428",
        "142",
        "外消費税 142",
        "小計",
        "1428",
        "合計",
        "1570",
    ])

    _replace_dense_sequence_rows_when_balanced(extracted, text)

    assert extracted["line_items"] == existing
