"""Generic receipt-shaped contracts for shared parser repairs."""


def test_repeated_labeled_price_rows_strip_only_the_unique_marker_digits():
    from receipt_parser.receipt_row_projection import (
        _replace_dense_item_rows_when_balanced,
    )

    extracted = {
        "total": 450,
        "subtotal": 450,
        "taxes": [{"rate": "8%", "label": "内税", "amount": 33}],
        "line_items": [
            {"description": "既存項目", "qty": 1, "unit_price": 450, "total": 450}
        ],
    }
    text = "\n".join([
        "品目 111111",
        "項目甲",
        "200",
        "0",
        "品目 222222",
        "項目乙",
        "1000",
        "0",
        "品目 333333",
        "項目丙",
        "1500",
        "0",
        "購入点数 3",
        "合計",
        "450",
        "価格横の数字は軽減税率区分",
    ])

    _replace_dense_item_rows_when_balanced(extracted, text)

    assert [row["description"] for row in extracted["line_items"]] == [
        "項目甲",
        "項目乙",
        "項目丙",
    ]
    assert [row["total"] for row in extracted["line_items"]] == [200.0, 100.0, 150.0]
    assert {row["tax_category"] for row in extracted["line_items"]} == {"8%"}


def test_vertical_tax_stack_slides_past_leading_gross_amount():
    from receipt_parser.receipt_financial import _vertical_inner_tax_table_entries

    entries = _vertical_inner_tax_table_entries([
        "内税",
        "税率",
        "10.0",
        "%",
        "取引金額",
        "1100",
        "税抜き",
        "1000",
        "税額",
        "100",
        "担当者",
    ])

    assert entries == [("10%", "base", 1000.0), ("10%", "tax", 100.0)]


def test_mixed_waon_cash_and_change_preserve_printed_financials():
    from receipt_parser.receipt_identity_payment import (
        _fix_payment_method,
        _fix_total_from_stacked_cash_tender_block,
    )
    from receipt_parser.receipt_postprocess_phases import (
        _run_payment_points_reconciliation_phase,
    )

    items = [
        {"description": "項目甲", "qty": 2, "unit_price": 100, "total": 200},
        {"description": "項目乙", "qty": 2, "unit_price": 150, "total": 300},
    ]
    extracted = {
        "total": 550,
        "subtotal": 500,
        "amount_paid": 500,
        "payment_method": "WAON",
        "points_used": 0,
        "taxes": [{"rate": "10%", "label": "外税", "amount": 50}],
        "line_items": [dict(item) for item in items],
    }
    text = "\n".join([
        "項目甲",
        "100",
        "2点",
        "200",
        "項目乙",
        "150",
        "2点",
        "300",
        "小計",
        "500",
        "外税10%",
        "50",
        "合計",
        "550",
        "WAON支払 500",
        "現金",
        "100",
        "お釣り",
        "50",
    ])

    _fix_total_from_stacked_cash_tender_block(extracted, text)
    _fix_payment_method(extracted, text, 0.9, {})
    _run_payment_points_reconciliation_phase(
        extracted,
        text,
        0.9,
        {},
        ("points_used", "points_payment"),
    )

    assert extracted["total"] == 550
    assert extracted["subtotal"] == 500
    assert extracted["amount_paid"] == 550
    assert extracted["payment_method"] is None
    assert extracted["line_items"] == items


def test_bare_tender_and_change_following_split_total_are_cash_settlement():
    from receipt_parser.receipt_financial import extract_financial_totals
    from receipt_parser.receipt_identity_payment import _fix_payment_method
    from receipt_parser.receipt_postprocess_phases import (
        _run_payment_points_reconciliation_phase,
    )

    text = "\n".join([
        "項目甲",
        "¥700",
        "合おお",
        "計",
        "¥700",
        "預り",
        "¥1,000",
        "釣",
        "¥300",
    ])
    totals = extract_financial_totals(text)
    extracted = {
        "total": totals.get("total"),
        "amount_paid": 1000,
        "payment_method": None,
        "points_used": 0,
    }

    _fix_payment_method(extracted, text, 0.9, {})
    _run_payment_points_reconciliation_phase(
        extracted,
        text,
        0.9,
        {},
        ("points_used", "points_payment"),
    )

    assert extracted["total"] == 700
    assert extracted["amount_paid"] == 700
    assert extracted["payment_method"] == "cash"


def test_receipt_postprocess_combines_qty_digit_and_unique_subtotal_repairs():
    from receipt_parser.receipt_postprocess import postprocess_receipt

    extracted = {
        "document_type": "receipt",
        "total": 782,
        "subtotal": 782,
        "taxes": [],
        "line_items": [
            {
                "description": "一般商品甲乙",
                "qty": 1,
                "unit_price": 596,
                "total": 596,
                "discount": 0,
                "discount_rate": "",
            },
            {
                "description": "一般商品丙丁",
                "qty": 1,
                "unit_price": 90,
                "total": 90,
                "discount": 0,
                "discount_rate": "",
            },
            {
                "description": "飲料商品 500",
                "qty": 1,
                "unit_price": 88,
                "total": 88,
                "discount": 0,
                "discount_rate": "",
            },
        ],
    }
    text = "\n".join([
        "案内",
        "2099/1/1 00:00",
        "一般商品甲乙",
        "コX単298",
        "¥596",
        "一般商品丙丁",
        "¥90",
        "飲料商品 500",
        "¥88",
        "小計/ 4点",
        "小計",
        "¥782",
        "合計",
        "¥782",
    ])

    postprocess_receipt(extracted, text, 0.9, {}, {}, "test-model")

    assert [
        (row["qty"], row["unit_price"], row["total"])
        for row in extracted["line_items"]
    ] == [
        (2.0, 298.0, 596.0),
        (1, 98.0, 98.0),
        (1, 88, 88),
    ]
    assert sum(row["total"] for row in extracted["line_items"]) == extracted["subtotal"]


def test_quantity_and_discount_ownership_survive_later_summary_discount():
    from receipt_parser.receipt_item_cleanup import _detect_ocr_discounts
    from receipt_parser.receipt_item_repair import _apply_qty_notation_from_ocr

    items = [
        {"description": "先行商品甲乙", "qty": 1, "unit_price": 100, "total": 100, "discount": 0},
        {"description": "別商品丙丁", "qty": 1, "unit_price": 200, "total": 200, "discount": 0},
        {"description": "主力商品戊己", "qty": 1, "unit_price": 158, "total": 158, "discount": 0},
        {"description": "有料レジ袋小", "qty": 1, "unit_price": 5, "total": 5, "discount": 0},
    ]
    text = "\n".join([
        "先行商品甲乙",
        "別商品丙丁",
        "主力商品戊己",
        "2コX単158",
        "100",
        "200",
        "316",
        "値引",
        "-16",
        "有料レジ袋小",
        "5",
        "********",
        "まとめ値引き ********",
        "-16",
        "316から",
        "300に致します",
        "小計",
        "605",
    ])

    _apply_qty_notation_from_ocr(items, text)
    _detect_ocr_discounts(items, text)

    assert [
        (row["qty"], row["unit_price"], row["total"], row["discount"])
        for row in items
    ] == [
        (1, 100, 100, 0),
        (1, 200, 200, 0),
        (2.0, 158.0, 300.0, 16.0),
        (1, 5, 5, 0),
    ]


def test_inline_quantity_and_one_rate_vertical_tax_stack_reconcile_together():
    from receipt_parser.receipt_financial import extract_financial_totals
    from receipt_parser.receipt_item_repair import _fix_qty_from_ocr_patterns

    items = [
        {
            "description": "一般商品甲乙",
            "qty": 1,
            "unit_price": 1611,
            "total": 1611,
        },
        {
            "description": "一般商品丙丁",
            "qty": 1,
            "unit_price": 321,
            "total": 321,
        },
    ]
    text = "\n".join([
        "一般商品甲乙",
        "単537×3個 軽 ¥1,611",
        "一般商品丙丁",
        "¥321",
        "小",
        "計",
        "¥1,932",
        "8% 内税対象額",
        "8.00%",
        "¥1,932",
        "8%税額",
        "8.00%",
        "¥143",
        "合計",
        "¥1,932",
        "(税抜額",
        "(内消費税等",
        "¥1,789)",
        "¥143)",
    ])

    _fix_qty_from_ocr_patterns(items, text)
    financial = extract_financial_totals(text)

    assert [
        (row["qty"], row["unit_price"], row["total"])
        for row in items
    ] == [(3.0, 537.0, 1611.0), (1, 321, 321)]
    assert financial["total"] == 1932
    assert financial["subtotal"] == 1789
    assert [
        (tax["rate"], tax["amount"])
        for tax in financial["taxes"]
    ] == [("8%", 143.0)]


def test_price_rejoin_skips_exact_split_summary_fragments():
    from receipt_parser.normalize import rejoin_price_lines

    normalized = rejoin_price_lines(
        "\n".join([
            "項目甲",
            "118軽",
            "合",
            "項目乙",
            "計",
            "190軽",
            "(内消費税等",
            "¥308",
            "¥22)",
            "お預り合計",
            "お",
            "釣",
            "¥500",
            "¥192",
        ])
    )

    assert "項目甲  118軽" in normalized
    assert "項目乙  190軽" in normalized
    assert "項目乙  192" not in normalized
    assert "お預り合計" in normalized
    assert "¥192" in normalized
