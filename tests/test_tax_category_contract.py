"""Generic contracts for evidence-backed receipt tax-category assignment."""

from receipt_parser.receipt_tax_categories import (
    _fix_tax_categories_from_ocr_markers,
    _reconcile_layout_markers_to_rate_bases,
    _rebalance_tax_categories_to_rate_bases,
    reconcile_tax_categories_from_rate_bases,
)
from receipt_parser.receipt_postprocess_phases import _run_tax_category_assignment_phase


def _item(description, total, category):
    return {
        "description": description,
        "qty": 1,
        "unit_price": total,
        "total": total,
        "tax_category": category,
    }


def test_explicit_row_markers_win_while_rate_bases_fill_only_unresolved_rows():
    items = [
        _item("商品甲", 100, "10%"),
        _item("商品乙", 200, "8%"),
        _item("商品丙", 300, "8%"),
        _item("商品丁", 400, "10%"),
    ]
    text = "\n".join([
        "商品甲", "100*",
        "商品乙", "200除",
        "商品丙", "300",
        "商品丁", "400",
        "8%対象額 500", "10%対象額 500",
        "*印は軽減税率8%対象商品",
    ])

    _rebalance_tax_categories_to_rate_bases(
        items,
        text,
        [{"rate": "8%", "amount": 40}, {"rate": "10%", "amount": 50}],
        {"8%": 500, "10%": 500},
    )

    assert [item["tax_category"] for item in items] == ["8%", "10%", "10%", "8%"]


def test_explicit_nontaxable_row_is_never_consumed_by_taxable_subsets():
    items = [
        _item("行政手数料", 652, "8%"),
        _item("商品甲", 100, "10%"),
        _item("商品乙", 200, "8%"),
        _item("商品丙", 300, "10%"),
    ]
    text = "\n".join([
        "行政手数料 652非",
        "商品甲", "100*",
        "商品乙", "200",
        "商品丙", "300",
        "8%対象額 400", "10%対象額 200", "非課税対象額 652",
        "*印は軽減税率8%対象商品",
    ])

    _rebalance_tax_categories_to_rate_bases(
        items,
        text,
        [
            {"rate": "8%", "amount": 32},
            {"rate": "10%", "amount": 20},
            {"rate": "0%", "label": "非課税", "amount": 0},
        ],
        {"8%": 400, "10%": 200},
    )

    assert [item["tax_category"] for item in items] == ["0%", "8%", "10%", "8%"]


def test_nontaxable_summary_locks_only_the_explicitly_marked_row():
    items = [
        _item("行政手数料", 652, "0%"),
        _item("商品甲", 100, "0%"),
        _item("商品乙", 200, "8%"),
    ]
    text = "\n".join([
        "行政手数料 652非",
        "商品甲", "100",
        "商品乙", "200*",
        "8%対象額 200", "10%対象額 100", "非課税対象額 652",
        "*印は軽減税率8%対象商品",
    ])

    _rebalance_tax_categories_to_rate_bases(
        items,
        text,
        [
            {"rate": "8%", "amount": 16},
            {"rate": "10%", "amount": 10},
            {"rate": "0%", "label": "非課税", "amount": 0},
        ],
        {"8%": 200, "10%": 100},
    )

    assert [item["tax_category"] for item in items] == ["0%", "10%", "8%"]


def test_equal_total_rate_ownership_without_row_evidence_fails_closed():
    items = [
        _item("商品甲", 100, "8%"),
        _item("商品乙", 100, "8%"),
    ]

    _rebalance_tax_categories_to_rate_bases(
        items,
        "8%対象額 100\n10%対象額 100",
        [],
        {"8%": 100, "10%": 100},
    )

    assert [item["tax_category"] for item in items] == ["8%", "8%"]


def test_equal_totals_are_safe_when_each_row_has_an_explicit_marker():
    items = [
        _item("商品甲", 100, "10%"),
        _item("商品乙", 100, "8%"),
    ]
    text = "\n".join([
        "商品甲", "100*",
        "商品乙", "100除",
        "8%対象額 100", "10%対象額 100",
        "*印は軽減税率8%対象商品",
    ])

    _rebalance_tax_categories_to_rate_bases(
        items,
        text,
        [],
        {"8%": 100, "10%": 100},
    )

    assert [item["tax_category"] for item in items] == ["8%", "10%"]


def test_qty_detail_does_not_resolve_equal_total_tax_rate_tie():
    items = [
        _item("商品甲", 398, "10%"),
        {**_item("商品乙", 398, "10%"), "qty": 2, "unit_price": 199},
    ]
    text = "\n".join([
        "商品甲", "398",
        "商品乙", "2コX単199", "398",
        "8%対象額 398", "10%対象額 398",
    ])

    _rebalance_tax_categories_to_rate_bases(
        items,
        text,
        [],
        {"8%": 398, "10%": 398},
    )

    assert [item["tax_category"] for item in items] == ["10%", "10%"]


def test_bare_quantity_operator_row_is_not_a_reduced_tax_marker():
    items = [_item("商品甲詳細", 240, "10%")]
    text = "\n".join([
        "商品甲詳細",
        "1 *",
        "240",
        "10%対象額 240",
        "*印は軽減税率8%対象商品",
    ])

    _fix_tax_categories_from_ocr_markers(items, text)

    assert items[0]["tax_category"] == "10%"


def test_inline_quantity_times_amount_is_not_a_reduced_tax_marker():
    items = [_item("商品乙詳細", 240, "10%")]
    text = "\n".join([
        "商品乙詳細 2 * 120",
        "240",
        "10%対象額 240",
        "*印は軽減税率8%対象商品",
    ])

    _fix_tax_categories_from_ocr_markers(items, text)

    assert items[0]["tax_category"] == "10%"


def test_attached_price_marker_still_assigns_reduced_tax():
    items = [_item("商品丙詳細", 240, "10%")]
    text = "\n".join([
        "商品丙詳細",
        "240*",
        "8%対象額 240",
        "*印は軽減税率8%対象商品",
    ])

    _fix_tax_categories_from_ocr_markers(items, text)

    assert items[0]["tax_category"] == "8%"


def test_explicit_marker_wins_over_late_inner_tax_inference():
    extracted = {
        "line_items": [
            _item("商品甲", 100, "10%"),
            _item("商品乙", 200, "10%"),
        ],
    }
    text = "\n".join([
        "商品甲", "100*",
        "商品乙", "200除",
        "8%対象額 100", "10%対象額 200",
        "10%内税対象額 ¥100",
        "*印は軽減税率8%対象商品",
    ])

    reconcile_tax_categories_from_rate_bases(extracted, text)

    assert [item["tax_category"] for item in extracted["line_items"]] == ["8%", "10%"]


def test_explicit_marker_wins_over_late_opaque_suffix_inference():
    extracted = {
        "line_items": [
            _item("商品甲", 100, "10%"),
            _item("商品乙", 200, "10%"),
            _item("商品丙", 400, "10%"),
            _item("商品丁", 500, "10%"),
        ],
    }
    text = "\n".join([
        "商品甲 100", "商品乙 200", "商品丙 400*", "商品丁 500",
        "100 E", "200 E", "400 T", "500 T",
        "8%対象額 300", "10%対象額 900",
        "*印は軽減税率8%対象商品",
    ])

    reconcile_tax_categories_from_rate_bases(extracted, text)

    assert extracted["line_items"][2]["tax_category"] == "8%"


def test_single_printed_rate_assigns_only_the_unmarked_complement():
    items = [
        _item("商品甲", 10, "8%"),
        _item("商品乙", 100, "10%"),
        _item("商品丙", 200, "10%"),
    ]
    text = "\n".join([
        "商品甲", "10",
        "*商品乙", "100",
        "*商品丙", "200",
        "8%対象額 300",
        "*は軽減税率8%適用商品",
    ])

    _rebalance_tax_categories_to_rate_bases(items, text, [], {"8%": 300})

    assert [item["tax_category"] for item in items] == ["10%", "8%", "8%"]


def test_cumulative_rate_bases_are_reduced_to_per_rate_residuals():
    items = [
        _item("商品甲", 800, "10%"),
        _item("商品乙", 200, "10%"),
        _item("商品丙", 10, "8%"),
    ]
    text = "\n".join([
        "*商品甲", "800",
        "商品乙", "200",
        "商品丙", "10",
        "税率8%課税対象額", "1080", "税率8%税額", "80",
        "税率10%課税対象額", "1091", "税率10%税額", "1",
        "*印は軽減税率8%適用商品",
    ])

    _rebalance_tax_categories_to_rate_bases(
        items,
        text,
        [{"rate": "8%", "amount": 80}, {"rate": "10%", "amount": 1}],
        {"8%": 1080, "10%": 1091},
    )

    assert [item["tax_category"] for item in items] == ["8%", "8%", "10%"]


def test_unique_detached_marker_and_explicit_external_suffix_lock_their_rows():
    items = [
        _item("食品甲", 980, "10%"),
        _item("日用品乙", 100, "8%"),
        _item("食品丙", 200, "8%"),
    ]
    text = "\n".join([
        "食品甲",
        "日用品乙",
        "食品丙",
        "980*",
        "¥100外",
        "200",
        "8%対象額 1,180",
        "10%対象額 100",
        "*印は軽減税率8%対象商品",
    ])

    _rebalance_tax_categories_to_rate_bases(
        items,
        text,
        [],
        {"8%": 1180, "10%": 100},
    )

    assert [item["tax_category"] for item in items] == ["8%", "10%", "8%"]


def test_preceding_reduced_marker_locks_single_rate_rows_and_leaves_complement():
    items = [
        _item("紙袋", 10, "8%"),
        _item("食品甲", 250, "8%"),
        _item("食品乙", 180, "8%"),
    ]
    text = "\n".join([
        "紙袋",
        "10",
        "* 不揃い",
        "食品甲",
        "250",
        "* 不揃い",
        "食品乙",
        "180",
        "8%対象 430",
        "*は軽減税率8%適用商品",
    ])

    _rebalance_tax_categories_to_rate_bases(items, text, [], {"8%": 430})

    assert [item["tax_category"] for item in items] == ["10%", "8%", "8%"]


def test_repeated_binary_tax_column_maps_only_when_gross_rate_sums_balance():
    items = [
        _item("食品甲", 100, "10%"),
        _item("商品乙", 200, "8%"),
        _item("商品丙", 300, "8%"),
    ]
    text = "\n".join([
        "商品名 111",
        "食品甲",
        "100 1",
        "商品名 222",
        "商品乙",
        "200 0",
        "商品名 333",
        "商品丙",
        "300 0",
        "税率",
        "10.0",
        "税抜き 455",
        "税額 45",
        "8.0",
        "税抜き 93",
        "税額 7",
        "担当者",
        "192 1",
    ])

    _rebalance_tax_categories_to_rate_bases(
        items,
        text,
        [
            {"rate": "10%", "amount": 45},
            {"rate": "8%", "amount": 7},
        ],
        {"10%": 455, "8%": 93},
    )

    assert [item["tax_category"] for item in items] == ["8%", "10%", "10%"]
    assert [item["_tax_category_locked"] for item in items] == ["8%", "10%", "10%"]


def test_stacked_description_and_price_runs_keep_duplicate_marker_ownership():
    items = [
        _item("日用品甲", 168, "8%"),
        _item("食品乙", 398, "8%"),
        _item("日用品丙", 800, "8%"),
        _item("食品丁", 168, "8%"),
    ]
    text = "\n".join([
        "日用品甲",
        "食品乙",
        "168",
        "398*",
        "日用品丙",
        "食品丁",
        "800",
        "168*",
        "8%対象額 566",
        "10%対象額 968",
        "*印は軽減税率8%対象商品",
    ])

    _rebalance_tax_categories_to_rate_bases(
        items,
        text,
        [],
        {"8%": 566, "10%": 968},
    )

    assert [item["tax_category"] for item in items] == ["10%", "8%", "10%", "8%"]


def test_reduced_marker_does_not_cross_prior_item_price_row():
    items = [
        _item("食品甲", 100, "8%"),
        _item("日用品乙", 200, "10%"),
    ]
    text = "\n".join([
        "*食品甲",
        "100",
        "日用品乙",
        "200",
        "*印は軽減税率8%対象商品",
    ])

    _fix_tax_categories_from_ocr_markers(items, text)

    assert [item["tax_category"] for item in items] == ["8%", "10%"]


def test_unique_amount_suffix_does_not_cross_an_intervening_description():
    items = [
        _item("食品甲", 100, "8%"),
        _item("日用品乙", 200, "10%"),
    ]
    text = "\n".join([
        "食品甲", "100",
        "日用品乙", "200",
        "クーポン値引", "100外",
        "8%対象額 100", "10%対象額 200",
    ])

    _fix_tax_categories_from_ocr_markers(items, text)

    assert [item["tax_category"] for item in items] == ["8%", "10%"]


def test_price_suffix_applies_to_its_immediately_adjacent_description():
    items = [
        _item("食品甲", 100, "8%"),
        _item("日用品乙", 200, "8%"),
    ]
    text = "\n".join([
        "食品甲", "100外",
        "日用品乙", "200",
        "8%対象額 200", "10%対象額 100",
    ])

    _fix_tax_categories_from_ocr_markers(items, text)

    assert [item["tax_category"] for item in items] == ["10%", "8%"]


def test_repeated_marker_modifier_prefers_unique_following_description():
    items = [
        _item("紙袋", 10, "8%"),
        _item("不揃い 塩パン風", 250, "10%"),
        _item("不揃い 塩キャラメル", 180, "10%"),
    ]
    text = "\n".join([
        "紙袋", "10",
        "* 不揃い", "塩パン風", "250",
        "* 不揃い", "塩キャラメル", "180",
        "8%対象額 430",
        "*印は軽減税率8%対象商品",
    ])

    _rebalance_tax_categories_to_rate_bases(items, text, [], {"8%": 430})

    assert [item["tax_category"] for item in items] == ["10%", "8%", "8%"]


def test_ambiguous_duplicate_descriptions_do_not_claim_one_marker_row():
    items = [
        _item("共通商品", 100, "10%"),
        _item("共通商品", 200, "10%"),
    ]
    text = "\n".join([
        "* 共通",
        "商品",
        "100",
        "共通商品",
        "200",
        "*印は軽減税率8%対象商品",
    ])

    _fix_tax_categories_from_ocr_markers(items, text)

    assert [item["tax_category"] for item in items] == ["10%", "10%"]


def _layout_tax_rows(rows):
    blocks = []
    for row_idx, (description, amount, marker) in enumerate(rows):
        y = 100 + row_idx * 40
        for text, x, width in ((description, 100, 220), (str(amount), 500, 60)):
            blocks.append({
                "text": text,
                "x": x,
                "y": y,
                "bbox": [[x, y], [x + width, y], [x + width, y + 20], [x, y + 20]],
                "page": 0,
            })
        if marker:
            blocks.append({
                "text": marker,
                "x": 580,
                "y": y,
                "bbox": [[580, y], [600, y], [600, y + 20], [580, y + 20]],
                "page": 0,
            })
    return blocks


def test_layout_marker_locks_and_unique_minimal_residual_fill_both_printed_bases():
    items = [
        _item("商品甲", 100, "8%"),
        _item("商品乙", 200, "8%"),
        _item("商品丙", 300, "8%"),
        _item("商品丁", 400, "8%"),
        _item("商品戊", 500, "10%"),
        _item("商品己", 600, "8%"),
    ]
    extracted = {"line_items": items, "taxes": []}
    text = "8%対象額 1,600\n10%対象額 500\n※印は軽減税率8%対象商品"
    layout = _layout_tax_rows([
        ("商品甲", 100, ""),
        ("商品乙", 200, "%"),
        ("商品丙", 300, "*"),
        ("商品丁", 400, ""),
        ("商品戊", 500, ""),
        ("商品己", 600, ""),
    ])

    _run_tax_category_assignment_phase(
        extracted,
        text,
        {},
        ("rebalance_rate_bases",),
        rate_bases={"8%": 1600, "10%": 500},
        ocr_layout_blocks=layout,
    )

    assert [item["tax_category"] for item in items] == ["10%", "8%", "8%", "10%", "8%", "8%"]


def test_layout_marker_reconciliation_uses_effective_pretax_rate_targets():
    items = [
        _item("商品甲", 100, "8%"),
        _item("商品乙", 200, "8%"),
        _item("商品丙", 300, "8%"),
        _item("商品丁", 400, "8%"),
        _item("商品戊", 500, "8%"),
        _item("商品己", 600, "8%"),
    ]
    layout = _layout_tax_rows([
        ("商品甲", 100, ""),
        ("商品乙", 200, "%"),
        ("商品丙", 300, "*"),
        ("商品丁", 400, ""),
        ("商品戊", 500, ""),
        ("商品己", 600, ""),
    ])

    _rebalance_tax_categories_to_rate_bases(
        items,
        "8%対象額 1,728\n10%対象額 550\n※印は軽減税率8%対象商品",
        [
            {"rate": "8%", "label": "内税", "amount": 128},
            {"rate": "10%", "label": "内税", "amount": 50},
        ],
        {"8%": 1728, "10%": 550},
        ocr_layout_blocks=layout,
    )

    assert [item["tax_category"] for item in items] == ["10%", "8%", "8%", "10%", "8%", "8%"]


def test_layout_marker_equal_minimal_residual_subsets_fail_closed():
    items = [
        _item("商品甲", 100, "10%"),
        _item("商品乙", 200, "8%"),
        _item("商品丙", 300, "10%"),
        _item("商品丁", 300, "10%"),
    ]
    before = [item["tax_category"] for item in items]
    layout = _layout_tax_rows([
        ("商品甲", 100, ""),
        ("商品乙", 200, "※"),
        ("商品丙", 300, ""),
        ("商品丁", 300, ""),
    ])

    changed = _reconcile_layout_markers_to_rate_bases(
        items,
        "※印は軽減税率8%対象商品",
        {"8%": 500, "10%": 400},
        layout,
        set(),
    )

    assert changed is False
    assert [item["tax_category"] for item in items] == before


def test_layout_marker_duplicate_descriptions_fail_closed():
    items = [
        _item("同一商品", 100, "10%"),
        _item("同一商品", 200, "8%"),
    ]
    before = [item["tax_category"] for item in items]
    layout = _layout_tax_rows([
        ("同一商品", 100, ""),
        ("同一商品", 200, "※"),
    ])

    changed = _reconcile_layout_markers_to_rate_bases(
        items,
        "※印は軽減税率8%対象商品",
        {"8%": 200, "10%": 100},
        layout,
        set(),
    )

    assert changed is False
    assert [item["tax_category"] for item in items] == before


def test_layout_marker_reconciliation_requires_layout_rows():
    items = [_item("商品甲", 100, "10%"), _item("商品乙", 200, "8%")]
    before = [item["tax_category"] for item in items]

    changed = _reconcile_layout_markers_to_rate_bases(
        items,
        "※印は軽減税率8%対象商品",
        {"8%": 200, "10%": 100},
        None,
        set(),
    )

    assert changed is False
    assert [item["tax_category"] for item in items] == before


def test_layout_marker_reconciliation_preserves_conflicting_locked_anchor():
    items = [_item("商品甲", 100, "10%"), _item("商品乙", 200, "10%")]
    before = [item["tax_category"] for item in items]
    layout = _layout_tax_rows([
        ("商品甲", 100, "※"),
        ("商品乙", 200, ""),
    ])

    changed = _reconcile_layout_markers_to_rate_bases(
        items,
        "※印は軽減税率8%対象商品",
        {"8%": 100, "10%": 200},
        layout,
        {0},
    )

    assert changed is False
    assert [item["tax_category"] for item in items] == before


def test_final_output_reconciliation_reasserts_layout_owned_tax_rows():
    from receipt_parser.receipt_output import _apply_final_receipt_output_repairs
    from receipt_parser.receipt_phase_trace import POSTPROCESS_PHASE_BY_NAME

    result = {
        "document_type": "receipt",
        "subtotal": 1000,
        "total": 1000,
        "taxes": [],
        "line_items": [
            _item("商品甲", 100, "8%"),
            _item("商品乙", 200, "8%"),
            _item("商品丙", 300, "8%"),
            _item("商品丁", 400, "8%"),
        ],
    }
    text = "\n".join([
        "商品甲", "100",
        "商品乙", "200*",
        "商品丙", "300",
        "商品丁", "400",
        "外税8%対象額 ¥500",
        "外税10%対象額 ¥500",
        "※印は軽減税率8%対象商品",
    ])
    layout = _layout_tax_rows([
        ("商品甲", 100, ""),
        ("商品乙", 200, "※"),
        ("商品丙", 300, ""),
        ("商品丁", 400, ""),
    ])
    trace = []

    _apply_final_receipt_output_repairs(
        result,
        text,
        mutation_trace=trace,
        ocr_layout_blocks=layout,
    )

    assert [item["tax_category"] for item in result["line_items"]] == [
        "10%", "8%", "8%", "10%",
    ]
    event = next(
        entry for entry in trace
        if entry["stage"] == "tax_categories_from_rate_bases"
    )
    assert event["owner_phase"] == "tax_category_assignment"
    assert "ocr_layout_blocks" in POSTPROCESS_PHASE_BY_NAME[event["owner_phase"]]["reads"]


def test_final_output_reconciliation_without_layout_keeps_equal_subsets_ambiguous():
    from receipt_parser.receipt_output import _apply_final_receipt_output_repairs

    result = {
        "document_type": "receipt",
        "subtotal": 1000,
        "total": 1000,
        "taxes": [],
        "line_items": [
            _item("商品甲", 100, "8%"),
            _item("商品乙", 200, "8%"),
            _item("商品丙", 300, "8%"),
            _item("商品丁", 400, "8%"),
        ],
    }

    _apply_final_receipt_output_repairs(
        result,
        "外税8%対象額 ¥500\n外税10%対象額 ¥500\n※印は軽減税率8%対象商品",
    )

    assert [item["tax_category"] for item in result["line_items"]] == [
        "8%", "8%", "8%", "8%",
    ]


def test_final_output_reconciliation_keeps_equal_layout_residuals_ambiguous():
    from receipt_parser.receipt_output import _apply_final_receipt_output_repairs

    result = {
        "document_type": "receipt",
        "subtotal": 900,
        "total": 900,
        "taxes": [],
        "line_items": [
            _item("商品甲", 100, "10%"),
            _item("商品乙", 200, "10%"),
            _item("商品丙", 300, "10%"),
            _item("商品丁", 300, "10%"),
        ],
    }
    layout = _layout_tax_rows([
        ("商品甲", 100, ""),
        ("商品乙", 200, "※"),
        ("商品丙", 300, ""),
        ("商品丁", 300, ""),
    ])

    _apply_final_receipt_output_repairs(
        result,
        "外税8%対象額 ¥500\n外税10%対象額 ¥400\n※印は軽減税率8%対象商品",
        ocr_layout_blocks=layout,
    )

    assert [item["tax_category"] for item in result["line_items"]] == [
        "10%", "10%", "10%", "10%",
    ]
