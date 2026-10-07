"""Structural contracts for discount evidence ownership."""

from receipt_parser.receipt_item_cleanup import (
    _clear_discounts_without_nearby_ocr_marker,
)


def test_complete_printed_schedule_composes_all_rates_and_rejects_malformed_tokens():
    from copy import deepcopy
    from receipt_parser.patterns import _discount_rate_tokens

    original = [{"description": "Sample tea 7", "qty": 1, "unit_price": 1000,
                 "total": 807, "discount": 193, "discount_rate": "19.3%"}]
    source = "Sample tea 7 1000\n15%\n-150\n5%\n-43\nNext item\n200"
    assert _discount_rate_tokens("10%OFF") == (10.0,)
    assert _discount_rate_tokens("10%OFFICE") == ()
    items = deepcopy(original)
    _clear_discounts_without_nearby_ocr_marker(items, source, rates_only=True)
    assert items == [dict(original[0], discount_rate="19.25%")]
    column_order = source.replace("15%\n-150\n5%\n-43", "割引15%\n会員割引5%\n-150\n-43")
    items = deepcopy(original)
    _clear_discounts_without_nearby_ocr_marker(items, column_order, rates_only=True)
    assert items == [dict(original[0], discount_rate="19.25%")]
    _clear_discounts_without_nearby_ocr_marker(items, source, rates_only=True)
    assert items == [dict(original[0], discount_rate="19.25%")]
    for token in ("240%", "x15%", "1e2%", "+15%", "＋15%", "0%", "5%%"):
        assert _discount_rate_tokens("5% " + token) == ()
        items = deepcopy(original)
        _clear_discounts_without_nearby_ocr_marker(items, source.replace("15%", token), rates_only=True)
        assert items == [dict(original[0], discount_rate="")]


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


def test_closed_discount_owners_keep_pending_labels_and_bounded_annotations():
    from copy import deepcopy

    cases = (
        (_item("Sample relish 3", 275, 55, "20%"),
         "Sample relish 3\n値引\n¥640\n¥275\n20%\n-55", "20%"),
        (_item("Sample battery 2P", 430, 43, "10%"),
         "Sample battery 2P ¥430\n7\n部門会員割 10%\n-\\43", "10%"),
        (_item("Sample flour 5", 160, 8, "5%"),
         "Sample flour 5\n会員様割引5%\n160\n-8", "5%"),
        (_item("Sample coat 4", 2100, 630, "30%"),
         "Sample coat 4\n072 3141592653589\nX12\n2,100\n割引\n30%\n-630\n@2,100", "30%"),
    )
    for original, source, rate in cases:
        items = [deepcopy(original)]
        _clear_discounts_without_nearby_ocr_marker(items, source)
        assert items == [dict(original, discount_rate=rate)]
        _clear_discounts_without_nearby_ocr_marker(items, source)
        assert items == [dict(original, discount_rate=rate)]
        malformed = source.replace(rate, "240%")
        items = [deepcopy(original)]
        _clear_discounts_without_nearby_ocr_marker(items, malformed, rates_only=True)
        assert items == [dict(original, discount_rate="")]

    inline_item, inline_source, _ = cases[1]
    for ambiguous in (
        inline_source.replace("\n7\n", "\n¥600\n"),
        inline_source.replace("\n7\n", "\nOther title 8\n600\n"),
        inline_source.replace("\n7\n", "\n部門会員割 10%\n7\n"),
        inline_source.replace("-\\43", "-\\40"),
    ):
        items = [deepcopy(inline_item)]
        _clear_discounts_without_nearby_ocr_marker(items, ambiguous, rates_only=True)
        assert items == [dict(inline_item, discount_rate="")]

    coat, barcode_source, _ = cases[3]
    for ambiguous in (barcode_source.replace("072 3141592653589", "072"),
                      barcode_source.replace("@2,100", "@2,200"),
                      barcode_source.replace("X12", "Other title 12")):
        items = [deepcopy(coat)]
        _clear_discounts_without_nearby_ocr_marker(items, ambiguous, rates_only=True)
        assert items == [dict(coat, discount_rate="")]
    competing = [deepcopy(coat), _item("X12", 2100, 0)]
    expected = [dict(coat, discount_rate=""), deepcopy(competing[1])]
    _clear_discounts_without_nearby_ocr_marker(competing, barcode_source, rates_only=True)
    assert competing == expected
    duplicate = [deepcopy(coat), deepcopy(coat)]
    _clear_discounts_without_nearby_ocr_marker(duplicate, barcode_source, rates_only=True)
    assert duplicate == [dict(coat, discount_rate="")] * 2


def test_quantity_detail_cannot_cross_the_next_source_product_even_if_total_closes():
    from copy import deepcopy
    from receipt_parser.receipt_projection import _repair_previous_item_from_following_qty_detail

    original = {"subtotal": 1200, "total": 1200,
                "line_items": [_item("試作root 4", 100, 0), _item("試作tuber 8", 600, 0)]}
    source = "試作root 4\n¥100\n試作tuber 8 ¥600\n(2個 X 単300)\n小計\n1200"
    items = deepcopy(original)
    _repair_previous_item_from_following_qty_detail(items, source)
    assert items == original
    supported = {"subtotal": 600, "total": 600, "line_items": [_item("試作root 4", 100, 0)]}
    _repair_previous_item_from_following_qty_detail(supported, "試作root 4\n¥100\n(2個 X 単300)\n小計\n600")
    assert supported["line_items"] == [dict(_item("試作root 4", 100, 0), qty=2, unit_price=300, total=600)]


def test_exact_adjacent_title_span_keeps_later_printed_discount_owner():
    from copy import deepcopy
    from receipt_parser.receipt_item_cleanup import _detect_ocr_discounts

    text = "NORTH\nCOTTON SHIRT\n@2,000\nTARGET ITEM 商品\n5,000\n割引\n30%\n-1,500\n@5,000\nNEXT ITEM 商品\n1,200"
    original = [_item("NORTH COTTON SHIRT", 2000, 0),
                _item("TARGET ITEM 商品", 5000, 0), _item("NEXT ITEM 商品", 1200, 0)]
    items = deepcopy(original)
    _detect_ocr_discounts(items, text)
    expected = deepcopy(original)
    expected[1].update(total=3500, discount=1500, discount_rate="30%")
    assert items == expected
    _detect_ocr_discounts(items, text)
    assert items == expected

    for ambiguous in (text.replace("NORTH\nCOTTON", "NORTH\nseparator\nCOTTON"),
                      text.replace("NORTH\nCOTTON SHIRT", "NORTH COTTON SHIRT\n@2,000\nNORTH COTTON SHIRT")):
        items = deepcopy(original)
        _detect_ocr_discounts(items, ambiguous)
        assert items == original
    own_discount = text.replace("@2,000", "2,000 400 引 1点 1,600")
    items = deepcopy(original)
    _detect_ocr_discounts(items, own_discount)
    assert items[0] == dict(original[0], total=1600, discount=400)
    assert items[1:] == expected[1:]
    competing = original + [_item("NORTH", 500, 0)]
    items = deepcopy(competing)
    _detect_ocr_discounts(items, own_discount)
    assert items == competing


def test_fixed_amount_discount_rows_require_complete_owned_arithmetic():
    from copy import deepcopy
    from receipt_parser.receipt_item_cleanup import _detect_ocr_discounts

    for marker in ("引", "号"):
        for financial in (f"480 70{marker} 1個\n410軽", f"480\n70{marker} 1個\n410軽"):
            text = f"商品甲\n{financial}\n商品乙\n60\n小計\n470"
            original = [_item("商品甲", 480, 70, "99%"), _item("商品乙", 60, 0)]
            items = deepcopy(original)
            _clear_discounts_without_nearby_ocr_marker(items, text)
            assert (items[0]["discount"], items[0]["total"], items[0]["discount_rate"]) == (70, 410, "")
            for incoming_discount in (0, 70):
                items = [_item("商品甲", 480, incoming_discount), _item("商品乙", 60, 0)]
                _detect_ocr_discounts(items, text)
                assert (items[0]["qty"], items[0]["unit_price"], items[0]["discount"], items[0]["total"]) == (1, 480, 70, 410)
                expected = deepcopy(items)
                _clear_discounts_without_nearby_ocr_marker(items, text)
                _detect_ocr_discounts(items, text)
                assert items == expected

    original = [_item("商品甲", 480, 0), _item("商品乙", 60, 0)]
    for ambiguous in (
        "商品甲\n480 70号 1個\n420軽\n商品乙\n60",
        "商品甲\n480 70号 2個\n410軽\n商品乙\n60",
        "商品甲\n70号 1個\n410軽\n商品乙\n60",
        "商品甲\n480 70号 1個\n410軽\n420軽\n商品乙\n60",
        "商品甲\n商品甲\n480 70号 1個\n410軽\n商品乙\n60",
        "商品甲\n480\n商品乙\n70号 1個\n410軽",
    ):
        items = deepcopy(original)
        _detect_ocr_discounts(items, ambiguous)
        assert items == original


def test_missing_gross_price_requires_one_owned_printed_amount_and_discount():
    from copy import deepcopy
    from receipt_parser.receipt_item_cleanup import _detect_ocr_discounts

    original = [_item("Test Product", 70, 0)]
    original[0]["unit_price"] = None
    text = "Test Product\n¥500\n値引\n-100\n小計\n400"
    items = deepcopy(original)
    _detect_ocr_discounts(items, text)
    assert (items[0]["unit_price"], items[0]["discount"], items[0]["total"]) == (500, 100, 400)
    expected = deepcopy(items)
    _detect_ocr_discounts(items, text)
    assert items == expected

    for qty in (1, 2):
        priced = [_item("Test Product", 500, 0)]
        priced[0]["qty"] = qty
        priced[0]["total"] = qty * 500
        _detect_ocr_discounts(
            priced, f"Test Product\n値引\n¥{qty * 500}\n-100\n小計\n{qty * 500 - 100}",
        )
        assert (priced[0]["unit_price"], priced[0]["discount"], priced[0]["total"]) == (500, 100, qty * 500 - 100)

    for ambiguous in (
        text.replace("¥500\n", ""),
        text.replace("¥500", "¥500 ¥600"),
        text.replace("Test Product", "Test Product\nTest Product"),
        text.replace("-100", "お預り\n-100"),
        text.replace("-100", "Next Product\n-100"),
        text.replace("-100", "-100\n-50"),
        text.replace("¥500", "¥50"),
        "Test Product\n値引\n-100\n小計\n400\n¥500",
    ):
        items = deepcopy(original)
        _detect_ocr_discounts(items, ambiguous)
        assert items == original


def test_multiple_labelled_reductions_require_owned_gross_aggregate_and_unique_subtotal():
    from copy import deepcopy
    from receipt_parser.receipt_item_cleanup import _detect_ocr_discounts

    text = "商品甲\n240\n値引\n-20\n値引額\n-30\n商品乙\n150\n商品代金\n390\n値引合計\n-50\n小計\n340"
    for incoming in (0, 20, 50):
        items = [_item("商品甲", 240, incoming), _item("商品乙", 150, 0)]
        _detect_ocr_discounts(items, text)
        assert items == [_item("商品甲", 240, 50), _item("商品乙", 150, 0)]
        expected = deepcopy(items)
        _clear_discounts_without_nearby_ocr_marker(items, text)
        _detect_ocr_discounts(items, text)
        assert items == expected

    original = [_item("商品甲", 240, 0), _item("商品乙", 150, 0)]
    for unsupported in (
        text.replace("-50\n小計", "-20\n小計"),
        text.replace("340", "370"),
        text.replace("値引額\n-30", "-30"),
        text.replace("商品甲\n240", "商品甲\n2個 X 120\n240"),
        text + "\n小計\n370",
        text + "\n値引合計\n-50",
    ):
        items = deepcopy(original)
        _detect_ocr_discounts(items, unsupported)
        assert items == original


def test_discount_description_owners_exclude_numeric_tokens_and_keep_real_ambiguity_atomic():
    from copy import deepcopy
    from receipt_parser.receipt_item_cleanup import _detect_ocr_discounts

    for title, footer in (
        ("検査商品75P", "¥75 )"),
        ("TEST75P", "¥75 )"),
        ("検査商品75円", "¥75円 )"),
    ):
        original = [_item("商品甲", 240, 0), _item(title, 60, 0)]
        text = "\n".join([
            "商品甲", "¥240", "値引", "-24", title, "¥60", "小計", "276", footer,
        ])
        items = deepcopy(original)
        _detect_ocr_discounts(items, text)
        assert (items[0]["discount"], items[0]["total"]) == (24, 216)
        assert items[1] == original[1]
        expected = deepcopy(items)
        _clear_discounts_without_nearby_ocr_marker(items, text)
        _detect_ocr_discounts(items, text)
        assert items == expected

        ambiguous = deepcopy(original)
        _detect_ocr_discounts(ambiguous, text.replace(title, f"{title}\n{title}"))
        assert ambiguous == original


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
    assert items[0]["discount_rate"] == ""


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


def test_rates_only_does_not_allocate_detached_rate_markers_by_arithmetic():
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

    assert [item["discount_rate"] for item in items] == ["30%", "", ""]


def test_local_malformed_percent_does_not_use_one_digit_stripped_suffix():
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

    assert items[0]["discount_rate"] == ""


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
        "",
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


def test_balanced_cleanup_recovers_only_owned_closed_fixed_discounts():
    from copy import deepcopy
    from receipt_parser.receipt_postprocess_phases import _run_line_item_cleanup_phase
    from receipt_parser.receipt_phase_trace import (
        _record_receipt_phase_mutation, _snapshot_receipt_mutation_fields,
    )

    titles = ["Sample cracker", "Sample soda", "Sample wafer", "Sample tea31"]
    totals = [276, 387, 186, 132]
    original = {"total": 981, "subtotal": 981, "taxes": [],
                "line_items": [_item(title, amount, 0) for title, amount in zip(titles, totals)]}
    first = "317 41引 1個\n276軽"
    text = "\n".join([
        titles[0], first, titles[1], "455 68号 1個", "387軽",
        titles[2], "223", "37引 1個", "186軽", titles[3], "¥132", "合計", "¥981",
    ])
    repaired = deepcopy(original)
    before = _snapshot_receipt_mutation_fields(repaired)
    _run_line_item_cleanup_phase(repaired, text, ("broad_ocr_line_item_repair",))
    trace = []
    _record_receipt_phase_mutation(trace, "item_cleanup", before, repaired)
    expected = deepcopy(original)
    for item, gross, discount in zip(expected["line_items"], (317, 455, 223, 132), (41, 68, 37, 0)):
        item.update(unit_price=gross, discount=discount)
    assert repaired == expected
    assert len(trace) == 1 and set(trace[0]["changes"]) == {"line_items"}
    _run_line_item_cleanup_phase(repaired, text, ("broad_ocr_line_item_repair",))
    assert repaired == expected

    for rejected in ("317 41引 2個\n276軽", "317 41引 1個\n278軽", "¥276\n値引\n-41"):
        unchanged = deepcopy(original)
        _run_line_item_cleanup_phase(unchanged, text.replace(first, rejected), ("broad_ocr_line_item_repair",))
        assert unchanged["line_items"][0] == original["line_items"][0]
        assert [item["total"] for item in unchanged["line_items"]] == totals
    digit_mismatch = deepcopy(original)
    _run_line_item_cleanup_phase(
        digit_mismatch, text.replace(titles[3] + "\n¥132", "Sample tea32\n201 69引 1個\n132軽"),
        ("broad_ocr_line_item_repair",),
    )
    assert digit_mismatch["line_items"][3] == original["line_items"][3]
    competing = deepcopy(original)
    duplicated = text.replace(titles[0] + "\n" + first, (titles[0] + "\n" + first + "\n") * 2)
    _run_line_item_cleanup_phase(competing, duplicated, ("broad_ocr_line_item_repair",))
    assert competing == original


def test_complete_literal_discount_title_precedes_containment_and_preserves_digits():
    from copy import deepcopy
    from receipt_parser.receipt_item_cleanup import _detect_ocr_discounts

    for titles, prefix, schedule, rate in (
        (("商品甲", "国産商品甲"), "", "25%\n-150", "25%"),
        (("商品甲4",), "商品甲8\n700\n割引\n25%\n-175\n", "25%\n-150", "25%"),
        (("商品甲", "国産商品甲"), "", "-150", ""),
    ):
        original = [_item(title, 600 + 100 * index, 0) for index, title in enumerate(titles)]
        original += [_item("後続商品乙", 600, 0), _item("次の商品丙", 300, 0)]
        text = prefix + "\n".join(f"{item['description']}\n{item['unit_price']}" for item in original[:-2])
        text += f"\n後続商品乙\n600\n割引\n{schedule}\n次の商品丙\n300"
        items = deepcopy(original)
        _detect_ocr_discounts(items, text)
        expected = deepcopy(original)
        expected[-2].update(total=450, discount=150, discount_rate=rate)
        assert items == expected
        _detect_ocr_discounts(items, text)
        assert items == expected


def test_duplicated_complete_discount_owner_keeps_independent_rate_unmodified():
    from copy import deepcopy
    from receipt_parser.receipt_item_cleanup import _detect_ocr_discounts

    for title, other in (("商品甲", "国産商品甲"), ("商品甲4", "商品甲8")):
        original = [_item(title, 600, 0), _item(other, 700, 0), _item("後続商品乙", 600, 0)]
        text = f"{title}\n600\n{title}\n600\n{other}\n700\n後続商品乙\n600\n割引\n25%\n-150"
        items = deepcopy(original)
        _detect_ocr_discounts(items, text)
        assert items == original


from copy import deepcopy

import pytest

from receipt_parser.receipt_marker_projection import _replace_campaign_discount_stream_when_balanced


def _campaign_item(title, gross, discount=0, qty=1):
    return dict(description=title, qty=qty, unit_price=gross / qty,
                total=gross - discount, discount=discount,
                discount_rate="", tax_category="8%")


def _campaign_case(layout, schedule="25%"):
    original = dict(subtotal=1560, taxes=[], line_items=[
        _campaign_item("商品甲", 30), _campaign_item("商品乙", 50), _campaign_item("商品丙", 10),
        _campaign_item("先行商品丁", 900), _campaign_item("試作商品7", 400, 100),
        _campaign_item("後続商品戊", 200), _campaign_item("最後の商品己", 70),
    ])
    prefix = ["商品甲 ¥30", "商品乙 ¥50", "商品丙 ¥10"]
    if layout == "displaced":
        body = ["試作商品7", "割引", "後続商品戊", "先行商品丁", "¥900", "¥400※"]
        tail = ["¥200", "最後の商品己 ¥70"]
    else:
        body = ["先行商品丁", "試作商品7", "割引", "¥900", "¥400※"]
        tail = ["後続商品戊", "¥200", "最後の商品己 ¥70"]
    if schedule:
        body.append(schedule)
    text = "\n".join(["2097/2/3 04:05", *prefix, *body, "-¥100", *tail,
                      "小計", "¥1560", "お買上商品数:7"])
    return original, text


@pytest.mark.parametrize("layout", ["displaced", "immediate"])
@pytest.mark.parametrize("schedule,rate", [("25%", "25%"), ("", "")])
def test_single_reconstructed_owner_retains_literal_schedule(layout, schedule, rate):
    original, text = _campaign_case(layout, schedule)
    result = deepcopy(original)
    _replace_campaign_discount_stream_when_balanced(result, text)
    expected = deepcopy(original)
    expected["line_items"][4]["discount_rate"] = rate
    assert result == expected
    _replace_campaign_discount_stream_when_balanced(result, text)
    assert result == expected


@pytest.mark.parametrize("layout,defect", [
    ("displaced", "duplicate_owner"),
    ("displaced", "third_queued_title"),
    ("displaced", "unowned_control"),
    ("displaced", "conflicting_title_price"),
    ("immediate", "extra_deduction"),
    ("immediate", "malformed_rate"),
    ("immediate", "quantity_detail"),
])
def test_local_packet_rejects_ambiguous_or_incomplete_controls(layout, defect):
    original, text = _campaign_case(layout)
    if defect == "duplicate_owner":
        text = text.replace("試作商品7\n割引", "試作商品7\n試作商品7\n割引")
    elif defect == "third_queued_title":
        text = text.replace("後続商品戊\n先行商品丁", "後続商品戊\n別の商品庚\n先行商品丁")
    elif defect == "unowned_control":
        text = text.replace("割引\n後続商品戊", "割引\n％\n後続商品戊")
    elif defect == "conflicting_title_price":
        text = text.replace("試作商品7\n割引", "試作商品7 ¥600\n割引")
    elif defect == "extra_deduction":
        text = text.replace("-¥100\n後続商品戊", "-¥100\n-1\n後続商品戊")
    elif defect == "malformed_rate":
        text = text.replace("25%", "x25%")
    else:
        text = text.replace("試作商品7\n割引", "試作商品7\n(2個 X 単200)\n割引")
    result = deepcopy(original)
    _replace_campaign_discount_stream_when_balanced(result, text)
    assert result == original


@pytest.mark.parametrize("detached", [False, True])
def test_general_stream_preserves_quantity_and_existing_owner_abstention(detached):
    original = dict(subtotal=1100, taxes=[], line_items=[
        _campaign_item("商品甲", 400, 40, qty=2), _campaign_item("商品乙", 300, 75),
        _campaign_item("商品丙", 500, 125), _campaign_item("商品丁", 80), _campaign_item("商品戊", 60),
    ])
    if detached:
        middle = ["商品乙", "割引25%", "商品丙", "割引25%", "300", "-75", "500", "-125"]
    else:
        middle = ["商品乙 300", "割引25%", "-75", "商品丙 500", "割引25%", "-125"]
    text = "\n".join(["2097/2/3 04:05", "商品甲", "400", "(2個 X 単200)",
                      "割引10%", "-40", *middle, "商品丁 80", "商品戊 60",
                      "小計", "1100", "お買上商品数:6"])
    result = deepcopy(original)
    _replace_campaign_discount_stream_when_balanced(result, text)
    expected = deepcopy(original)
    expected["line_items"][0]["discount_rate"] = "10%"
    if not detached:
        expected["line_items"][1]["discount_rate"] = "25%"
        expected["line_items"][2]["discount_rate"] = "25%"
    assert result == expected


def test_two_queued_literal_owners_keep_prefixes_and_printed_schedules():
    """Two complete titles and two unique closed packets own their printed rates."""
    from copy import deepcopy
    from receipt_parser.receipt_marker_projection import _replace_campaign_discount_stream_when_balanced

    titles = ["岩風サブレ", "浜色No.25ソーダ", "星砂チップ4", "BZ香味サラダ8", "山雲カップ2"]
    gross = [23, 47, 924, 657, 380]
    discounts = [0, 0, 231, 66, 57]
    original = dict(subtotal=1677, taxes=[], line_items=[
        dict(description=title, qty=1.0, unit_price=float(price), total=float(price - off),
             tax_category="8%", discount=float(off), discount_rate="")
        for title, price, off in zip(titles, gross, discounts)
    ])
    text = "\n".join([
        "2034/8/16 17:43", "スNo00824173 森", "スキャンレジ0038 スキャンNo5372",
        "395006岩風サブレ ¥23", "406107浜色No.25ソーダ ¥47",
        "517208星砂チップ4", "628309BZ香味サラダ8",
        "¥924", "25%", "-231", "¥657", "マーク値引", "10%", "-66",
        "739410山雲カップ2", "¥380", "割引", "15%", "-57",
        "小計", "¥1677", "お買上商品数:5",
    ])
    result = deepcopy(original)
    _replace_campaign_discount_stream_when_balanced(result, text)
    expected = deepcopy(original)
    for item, rate in zip(expected["line_items"], ("", "", "25%", "10%", "15%")):
        item["discount_rate"] = rate
    assert result == expected
    _replace_campaign_discount_stream_when_balanced(result, text)
    assert result == expected

    amount_only = deepcopy(original)
    _replace_campaign_discount_stream_when_balanced(amount_only, text.replace("\n25%\n", "\n"))
    expected_amount_only = deepcopy(expected)
    expected_amount_only["line_items"][2]["discount_rate"] = ""
    assert amount_only == expected_amount_only

    malformed = deepcopy(original)
    _replace_campaign_discount_stream_when_balanced(malformed, text.replace("25%", "x25%"))
    assert [item["discount_rate"] for item in malformed["line_items"]][2:4] == ["", ""]

    after_deduction = deepcopy(original)
    _replace_campaign_discount_stream_when_balanced(
        after_deduction, text.replace("25%\n-231", "-231\n25%"),
    )
    assert after_deduction["line_items"][2]["discount_rate"] == ""

    ambiguous_quantity = deepcopy(original)
    _replace_campaign_discount_stream_when_balanced(
        ambiguous_quantity, text.replace("\n¥924\n", "\n(2個 X 単462)\n¥924\n"),
    )
    assert [item["discount_rate"] for item in ambiguous_quantity["line_items"]][2:4] == ["", ""]


def test_complete_literal_counted_amount_offer_is_atomic_and_idempotent():
    from copy import deepcopy
    from receipt_parser.receipt_item_cleanup import _detect_ocr_discounts

    def fixture():
        items = [
            {"description": "麦茶2L", "qty": 1, "unit_price": 120, "total": 120, "discount": 0, "discount_rate": "", "tax_rate": "10%"},
            {"description": "麦茶2L", "qty": 1, "unit_price": 120, "total": 120, "discount": 0, "discount_rate": "", "tax_rate": "10%"},
            {"description": "水2本", "qty": 2, "unit_price": 50, "total": 100, "discount": 0, "discount_rate": "", "tax_rate": "8%"},
        ]
        text = "\n".join((
            "麦茶2L", "@120", "120",
            "麦茶2L", "@120", "1点", "120",
            "水2本", "@50", "2本", "100",
            "<麦茶2L 2個210円>", "-30",
            "お買上点数", "4点", "合計", "¥310",
        ))
        return items, text

    def assert_unchanged(change_text=None, change_items=None):
        items, text = fixture()
        if change_text:
            changed = change_text(text)
            assert changed != text
            text = changed
        if change_items:
            change_items(items)
        before = deepcopy(items)
        _detect_ocr_discounts(items, text)
        assert items == before

    items, text = fixture()
    items[0]["discount"] = None
    expected = deepcopy(items)
    for item in expected[:2]:
        item.update(discount=15, total=105, discount_rate="")
    _detect_ocr_discounts(items, text)
    assert items == expected
    assert [item["description"] for item in items] == ["麦茶2L", "麦茶2L", "水2本"]
    once = deepcopy(items)
    _detect_ocr_discounts(items, text)
    assert items == once

    rows, printed = fixture()
    printed = printed.replace("4点\n合計\n¥310", "4点  ¥310\n合計")
    wanted = deepcopy(rows)
    for item in wanted[:2]:
        item.update(discount=15, total=105, discount_rate="")
    _detect_ocr_discounts(rows, printed)
    assert rows == wanted

    # A star is a row marker; a POS model code is metadata. Both preserve the
    # full title owner, and quantity-owned units close displaced bare prices.
    rows, printed = fixture()
    rows.append(dict(description="稲穂むぎ3", qty=1, unit_price=87, total=87,
                     discount=0, discount_rate="", tax_rate="8%"))
    printed = printed.replace("麦茶2L\n@120", "麦茶2L\nMM:6432\n@120")
    printed = printed.replace("水2本\n@50\n2本\n100",
                              "* 水2本\n@50\n2本\n稲穂むぎ3\n100\n@87\n1点\n87")
    printed = printed.replace("4点", "5点").replace("¥310", "¥397")
    wanted = deepcopy(rows)
    for item in wanted[:2]:
        item.update(discount=15, total=105, discount_rate="")
    _detect_ocr_discounts(rows, printed)
    assert rows == wanted

    rows, printed = fixture()
    for item in rows[:2]:
        item.update(total=105)
    wanted = deepcopy(rows)
    for item in wanted[:2]:
        item.update(discount=15)
    _detect_ocr_discounts(rows, printed, fixed_only=True)
    assert rows == wanted

    assert_unchanged(lambda value: value.replace("4点", "3点"))
    assert_unchanged(lambda value: value.replace("¥310", "¥311"))
    assert_unchanged(lambda value: value.replace("-30\n", "-30\n-5\n"))
    assert_unchanged(lambda value: value.replace("-30\n", "-30\n<麦茶2L 2個210円>\n"))
    assert_unchanged(lambda value: value.replace("120\n水2本", "120\n麦茶2L\n水2本", 1))
    assert_unchanged(lambda value: value.replace("4点\n合計", "4点\nお買上点数\n4点\n合計"))
    assert_unchanged(lambda value: value.replace("合計\n¥310", "合計\n¥310\n合計\n¥310"))
    assert_unchanged(lambda value: value.replace("4点\n合計\n¥310", "4点  ¥310\n合計\n¥310"))
    assert_unchanged(lambda value: value.replace("水2本\n@50\n2本\n100", "水2本\n@50\n2本\n100\n101"))
    assert_unchanged(lambda value: value.replace("水2本\n@50\n2本", "水2本\n2本"))
    assert_unchanged(lambda value: value.replace("水2本\n@50\n2本\n100", "水2本\n@50\n100"))
    assert_unchanged(lambda value: value.replace("水2本\n@50\n2本", "水2本\n@50\n3本"))
    assert_unchanged(lambda value: value.replace("@120\n1点", "@120\n3点"))
    assert_unchanged(lambda value: value.replace("水2本\n@50\n2本", "水2本\n10%OFF\n@50\n2本"))
    assert_unchanged(lambda value: value.replace("麦茶2L 2個210円", "麦茶2L 2個211円"))
    assert_unchanged(lambda value: value.replace("<麦茶2L 2個210円>", "小計\n<麦茶2L 2個210円>"))
    assert_unchanged(change_items=lambda rows: rows[2].update(discount_rate="8%"))
    assert_unchanged(change_items=lambda rows: rows[2].pop("discount"))
    assert_unchanged(change_items=lambda rows: rows[2].update(discount=False))
    assert_unchanged(change_items=lambda rows: rows[2].update(discount=float("nan")))
    assert_unchanged(change_items=lambda rows: rows[2].update(qty=True))
    assert_unchanged(change_items=lambda rows: rows[2].update(qty=float("nan")))

    items, text = fixture()
    items[0].update(discount=15, total=105)
    before = deepcopy(items)
    _detect_ocr_discounts(items, text)
    assert items == before
