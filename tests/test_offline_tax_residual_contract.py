"""Synthetic contracts for structurally safe tax residual repairs."""

import pytest

from receipt_parser.receipt_tax_categories import _fix_tax_categories_from_ocr_markers
from receipt_parser import receipt_totals


def test_stacked_marker_inherits_only_when_adjacent_fragments_reconstruct_item():
    items = [{"description": "濃厚ミルクプリン", "total": 200, "tax_category": "10%"}]
    ocr_text = "\n".join([
        "*濃厚",
        "ミルクプリン",
        "200",
        "*印は軽減税率8%対象商品です",
    ])

    _fix_tax_categories_from_ocr_markers(items, ocr_text, stacked_only=True)

    assert items[0]["tax_category"] == "8%"


@pytest.mark.parametrize(
    "prior_rows",
    [
        ["*別の商品", "100"],
        ["*別の商品"],
    ],
)
def test_stacked_marker_does_not_bleed_from_previous_product_or_price(prior_rows):
    items = [{"description": "現在の商品ロング", "total": 200, "tax_category": "10%"}]
    ocr_text = "\n".join([
        *prior_rows,
        "現在の商品ロング",
        "200",
        "*印は軽減税率8%対象商品です",
    ])

    _fix_tax_categories_from_ocr_markers(items, ocr_text, stacked_only=True)

    assert items[0]["tax_category"] == "10%"


def _partial_tax_parse(monkeypatch):
    monkeypatch.setattr(
        receipt_totals,
        "_bare_number_tax_summary_entries",
        lambda _lines: [("10%", "tax", 10.0)],
    )
    monkeypatch.setattr(
        receipt_totals,
        "_interleaved_rate_tax_summary_entries",
        lambda _lines: [],
    )


def _balanced_multirate_receipt(subtotal=910):
    return {
        "total": 1000,
        "subtotal": subtotal,
        "taxes": [
            {"rate": "8%", "label": "外税", "amount": 80},
            {"rate": "10%", "label": "外税", "amount": 10},
        ],
    }


@pytest.mark.parametrize(
    "evidence",
    [
        "8%外税\n¥80\n10%外税\n¥10",
        "税合計\n¥90",
    ],
)
def test_partial_tax_parse_preserves_balanced_printed_multirate_set(monkeypatch, evidence):
    _partial_tax_parse(monkeypatch)
    extracted = _balanced_multirate_receipt()
    expected = [tax.copy() for tax in extracted["taxes"]]

    receipt_totals._restore_bare_number_tax_summary(extracted, evidence)

    assert extracted["taxes"] == expected
    assert extracted["subtotal"] == 910


def test_split_external_tax_label_survives_corrupt_base_when_current_set_balances(
    monkeypatch,
):
    _partial_tax_parse(monkeypatch)
    extracted = _balanced_multirate_receipt()
    extracted["taxes"][0]["label"] = "内税"
    ocr_text = "\n".join([
        "8%外税 タイショウ",
        "¥9,999",
        "8%",
        "税",
        "¥80",
        "10%外税 タイショウ",
        "¥100",
        "10%外税",
        "¥10",
        "税合計",
        "¥90",
        "合計",
        "¥1,000",
    ])

    receipt_totals._restore_bare_number_tax_summary(extracted, ocr_text)

    assert extracted["taxes"] == [
        {"rate": "8%", "label": "外税", "amount": 80},
        {"rate": "10%", "label": "外税", "amount": 10},
    ]
    assert extracted["subtotal"] == 910


@pytest.mark.parametrize(
    ("subtotal", "evidence"),
    [
        (900, "8%外税\n¥80\n10%外税\n¥10"),
        (910, "合計\n¥1,000"),
    ],
)
def test_partial_tax_parse_replaces_unbalanced_or_unprinted_current_set(
    monkeypatch,
    subtotal,
    evidence,
):
    _partial_tax_parse(monkeypatch)
    extracted = _balanced_multirate_receipt(subtotal)

    receipt_totals._restore_bare_number_tax_summary(extracted, evidence)

    assert [(tax["rate"], tax["amount"]) for tax in extracted["taxes"]] == [("10%", 10.0)]
    assert extracted["subtotal"] == 990
