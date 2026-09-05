import pytest

from receipt_parser.receipt_identity_payment import (
    _fix_company_name_merchant,
    _fix_payment_method,
    _fix_payment_reference,
)
from receipt_parser.normalize import rejoin_totals_label_value_columns
from receipt_parser.receipt_postprocess import postprocess_receipt
from receipt_parser.receipt_postprocess_phases import (
    _run_cash_tender_reconciliation_phase,
    _run_payment_points_reconciliation_phase,
)


def test_points_remain_unknown_without_redemption_evidence():
    extracted = {"total": 1200, "amount_paid": 900, "points_used": 0}

    _run_payment_points_reconciliation_phase(
        extracted,
        "会員ポイント 120 P\n今回獲得ポイント 5 P",
        0.9,
        {},
        ("points_used",),
    )

    assert extracted["points_used"] is None
    assert extracted["amount_paid"] == 1200


@pytest.mark.parametrize(
    ("ocr_text", "expected"),
    [
        ("ポイント利用 125", 125),
        ("ポイント利用\n0 P", 0),
    ],
)
def test_points_require_explicit_redemption_evidence(ocr_text, expected):
    extracted = {"points_used": None}

    _run_payment_points_reconciliation_phase(
        extracted,
        ocr_text,
        0.9,
        {},
        ("points_used",),
    )

    assert extracted["points_used"] == expected


@pytest.mark.parametrize(
    ("total", "ocr_text"),
    [
        (
            3990,
            "\n".join([
                "合計", "3,990", "支払い方法", "現金", "10,000", "釣銭", "6,010",
                "購入レシート(クレジット伝票含む)と",
            ]),
        ),
        (
            2000,
            "\n".join([
                "電子マネーチャージ 2,000非", "合計", "2,000", "現金", "お釣り",
                "2,000", "0", "WAONポイント (カード内)",
            ]),
        ),
        (
            3586,
            "\n".join([
                "ラベル ID: 867459823", "合計", "3586", "現金", "3606 ¥",
                "お釣り", "現金", "20 \\",
            ]),
        ),
    ],
)
def test_balanced_cash_settlement_outranks_broad_payment_tokens(total, ocr_text):
    extracted = {"total": total, "payment_method": "credit"}

    _fix_payment_method(extracted, ocr_text, 0.9, {})

    assert extracted["payment_method"] == "cash"


def test_stacked_cash_settlement_accepts_trailing_currency_marks():
    extracted = {
        "total": 3606,
        "subtotal": 3260,
        "amount_paid": 3606,
        "points_used": 0,
        "payment_method": "credit",
    }
    ocr_text = "\n".join([
        "ラベル ID: 867459823",
        "合計",
        "購入点数",
        "視覚的に問題なし",
        "1299.00",
        "1039",
        "1299",
        "****",
        "3586",
        "現金",
        "3606 ¥",
        "お釣り",
        "現金",
        "20 \\",
    ])

    _run_cash_tender_reconciliation_phase(
        extracted,
        ocr_text,
        ("stacked_cash_tender", "unlabeled_cash_tender_change"),
    )

    assert extracted["total"] == 3586
    assert extracted["subtotal"] == 3260
    assert extracted["amount_paid"] == 3586


def test_distant_settlement_labels_do_not_adopt_tax_stack_arithmetic():
    extracted = {
        "total": 2748,
        "amount_paid": 2748,
        "points_used": 0,
    }
    ocr_text = "\n".join([
        "合計",
        "¥4",
        "¥2748",
        "(8%対象)",
        "¥2744",
        "内税",
        "¥203",
        "取引情報",
        "お預り",
        "¥5048",
        "お",
        "釣",
        "¥2300",
    ])

    _run_cash_tender_reconciliation_phase(
        extracted,
        ocr_text,
        ("stacked_cash_tender",),
    )

    assert extracted == {
        "total": 2748,
        "amount_paid": 2748,
        "points_used": 0,
    }


@pytest.mark.parametrize(
    ("current_total", "printed_totals", "after_change"),
    [
        (3606, [3500], ["現金", "20 \\"]),
        (3586, [3586], ["現金", "20 \\"]),
        (3606, [3586], ["注文番号", "20 \\"]),
        (3606, [3586, 3500], ["現金", "20 \\"]),
    ],
)
def test_distant_cash_settlement_requires_three_independent_bindings(
    current_total,
    printed_totals,
    after_change,
):
    extracted = {
        "total": current_total,
        "amount_paid": 1111,
        "points_used": 0,
    }
    ocr_text = "\n".join([
        "合計",
        "購入点数",
        "視覚的に問題なし",
        "1299.00",
        "1039",
        "1299",
        "****",
        *map(str, printed_totals),
        "現金",
        "3606 ¥",
        "お釣り",
        *after_change,
    ])

    _run_cash_tender_reconciliation_phase(
        extracted,
        ocr_text,
        ("stacked_cash_tender",),
    )

    assert extracted == {
        "total": current_total,
        "amount_paid": 1111,
        "points_used": 0,
    }


@pytest.mark.parametrize(
    "ocr_text",
    [
        "\n".join([
            "総合計", "1,752", "1,331", "#633612223508****", "0", "1,021", "600",
            "スターバックスカード", "カード残高", "現金", "お釣り",
        ]),
        "\n".join([
            "合計", "500", "WAON支払額 450", "現金 100", "お釣り 50",
        ]),
        "\n".join([
            "合計", "500", "WAON支払額 450", "現金 50",
        ]),
    ],
)
def test_balanced_mixed_tender_is_not_collapsed_to_one_method(ocr_text):
    total = 1752 if "1,752" in ocr_text else 500
    extracted = {
        "total": total,
        "amount_paid": 1331 if total == 1752 else 450,
        "points_used": None,
        "payment_method": "cash",
    }

    _fix_payment_method(extracted, ocr_text, 0.9, {})
    _run_payment_points_reconciliation_phase(
        extracted,
        ocr_text,
        0.9,
        {},
        ("points_used", "points_payment"),
    )

    assert extracted["payment_method"] is None
    assert extracted["amount_paid"] == total


@pytest.mark.parametrize(
    "ocr_text",
    [
        "※majica, UCSカード払でポイントが貯まります",
        "電子マネーチャージ 2,000",
    ],
)
def test_payment_advertising_or_products_are_not_tenders(ocr_text):
    extracted = {"total": 2000, "payment_method": "credit"}

    _fix_payment_method(extracted, ocr_text, 0.9, {})

    assert extracted["payment_method"] is None


@pytest.mark.parametrize(
    "ocr_text",
    [
        "合計\n1,000\nカード\nカード払いでポイント特典",
        "合計\n1,000\n電子マネー\n電子マネーチャージで特典",
    ],
)
def test_bare_noncash_label_cannot_borrow_prior_total_from_advertising(
    ocr_text,
):
    extracted = {
        "total": 1000,
        "amount_paid": 1000,
        "payment_method": None,
    }

    _fix_payment_method(extracted, ocr_text, 0.9, {})

    assert extracted["payment_method"] is None


@pytest.mark.parametrize(
    ("ocr_text", "expected"),
    [
        ("クレジット(内\n¥3,630", "credit"),
        ("電子マネー\n¥1,375", "credit"),
        ("現金\n1,000", "cash"),
        ("現金払いでポイント2倍", None),
        ("クレジットカード会員募集中", None),
    ],
)
def test_payment_method_requires_an_exact_tender_with_amount(ocr_text, expected):
    extracted = {"payment_method": "credit"}

    _fix_payment_method(extracted, ocr_text, 0.9, {})

    assert extracted["payment_method"] == expected


def test_stacked_tax_then_generic_payment_labels_keep_ordered_amount_ownership():
    normalized = rejoin_totals_label_value_columns("\n".join([
        "(内消費税等10%",
        "モバイル決済",
        "¥37)",
        "¥1,234",
    ]))

    assert normalized.splitlines()[:2] == [
        "(内消費税等10% ¥37)",
        "モバイル決済 ¥1,234",
    ]


def test_cash_prefixed_deposit_label_uses_tender_minus_change_invariant():
    extracted = {"total": 1840, "amount_paid": 1840, "payment_method": "credit"}

    _fix_payment_method(
        extracted,
        "合計 1,840\n現金お預り 2,500\nお釣り 660",
        0.9,
        {},
    )

    assert extracted["payment_method"] == "cash"


def test_trailing_cash_label_cannot_borrow_preceding_credit_amount():
    extracted = {"total": 1234, "amount_paid": 1234, "payment_method": "cash"}

    _fix_payment_method(
        extracted,
        "カード決済\n¥1,234\n現計",
        0.9,
        {},
    )

    assert extracted["payment_method"] == "credit"


@pytest.mark.parametrize(
    ("total", "ocr_text", "expected"),
    [
        (
            860,
            "通行料金\n¥860-\n※通行料金の消費税率は10%です\n(現金)",
            "cash",
        ),
        (
            3390,
            "合計\n¥3,390\n(消費税10% 対象\n¥3,390\n内消費税等\n¥308)\nクレジット支払",
            "credit",
        ),
        (
            8948,
            "合計\n¥8,948\n¥8,948\nクレジット\nお釣り\n[クレジットカード売上票]",
            "credit",
        ),
        (
            1200,
            "電子マネー\nQUICPay売上票\n加盟店名\n取引内容 売上\n伝票番号 04068\n取引金額 ¥1200",
            "credit",
        ),
    ],
)
def test_exact_tender_owns_bounded_total_across_tax_or_slip_rows(
    total, ocr_text, expected,
):
    extracted = {
        "total": total,
        "amount_paid": total,
        "payment_method": None,
    }

    _fix_payment_method(extracted, ocr_text, 0.9, {})

    assert extracted["payment_method"] == expected


@pytest.mark.parametrize(
    ("ocr_text", "expected"),
    [
        (
            "¥900\nWAON支払\n伝票番号 0123\nWAON支払額\n¥1,000",
            "WAON",
        ),
        (
            "¥900\nクレジット\n承認番号 0123\n利用額 ¥1,000",
            "credit",
        ),
        (
            "¥900\nクレジット\nお預り\n¥1,000",
            "credit",
        ),
    ],
)
def test_valid_forward_tender_outranks_unrelated_prior_amount(
    ocr_text, expected,
):
    extracted = {
        "total": 1000,
        "amount_paid": 1000,
        "payment_method": None,
    }

    _fix_payment_method(extracted, ocr_text, 0.9, {})

    assert extracted["payment_method"] == expected


@pytest.mark.parametrize(
    ("ocr_text", "expected"),
    [
        ("デビット 1,000円", "debit"),
        ("合計\n1,000\nデビット", "debit"),
        ("銀行振込 1,000円", "bank_payment"),
        ("合計\n1,000\n銀行振込", "bank_payment"),
        ("口座振替\n¥1,000", "bank_payment"),
    ],
)
def test_payment_method_preserves_exact_allowed_tender_with_matching_amount(
    ocr_text, expected,
):
    extracted = {"total": 1000, "amount_paid": 1000, "payment_method": None}

    _fix_payment_method(extracted, ocr_text, 0.9, {})

    assert extracted["payment_method"] == expected


@pytest.mark.parametrize(
    "ocr_text",
    [
        "クレジット 100円",
        "クレジット 100円\n合計\n1000",
        "電子マネー\n取引金額 ¥900\n¥1,000",
        "合計\n¥1,000\n¥900\nクレジット",
        "お預り 100円",
        "お預り 100円\n合計\n1000",
    ],
)
def test_partial_tender_amount_does_not_claim_the_whole_tender(ocr_text):
    extracted = {"total": 1000, "amount_paid": 1000, "payment_method": "credit"}

    _fix_payment_method(extracted, ocr_text, 0.9, {})

    assert extracted["payment_method"] is None


def test_later_matching_tender_row_can_prove_the_payment_method():
    extracted = {"total": 1000, "amount_paid": 1000, "payment_method": None}

    _fix_payment_method(
        extracted,
        "VISA 100\nVISA 1,000",
        0.9,
        {},
    )

    assert extracted["payment_method"] == "credit"


def test_payment_method_uses_the_final_points_adjusted_amount_paid():
    extracted = {
        "document_type": "receipt",
        "total": 1000,
        "subtotal": 1000,
        "amount_paid": 1000,
        "points_used": None,
        "payment_method": "credit",
        "line_items": [],
        "taxes": [],
    }

    postprocess_receipt(
        extracted,
        "合計 1,000\n利用ポイント 900P\n現金計 100",
        0.9,
        {},
        {},
        "test-model",
    )

    assert extracted["points_used"] == 900
    assert extracted["amount_paid"] == 100
    assert extracted["payment_method"] == "cash"


def test_points_above_total_are_rejected_before_payment_arithmetic():
    extracted = {
        "total": 100,
        "amount_paid": 100,
        "points_used": None,
        "payment_method": None,
    }

    _run_payment_points_reconciliation_phase(
        extracted,
        "ポイント利用 150",
        0.9,
        {},
        ("points_used", "points_payment"),
    )

    assert extracted["points_used"] is None
    assert extracted["amount_paid"] == 100


def test_conflicting_ocr_points_do_not_drive_payment_arithmetic():
    extracted = {
        "total": 1000,
        "amount_paid": 1000,
        "points_used": 100,
    }

    _run_payment_points_reconciliation_phase(
        extracted,
        "ポイント利用 500",
        0.9,
        {},
        ("points_payment",),
    )

    assert extracted["points_used"] == 100
    assert extracted["amount_paid"] == 900


def test_fuel_merchant_uses_nearest_brand_line_not_header_slogan():
    extracted = {"merchant": "ACMEセルフ中央給油所"}

    _fix_company_name_merchant(
        extracted,
        "WELCOME\nACME\nACMEセルフ中央給油所\n運営株式会社",
    )

    assert extracted["merchant"] == "ACME"


@pytest.mark.parametrize(
    ("ocr_text", "extracted", "expected"),
    [
        (
            "NORD\n売上\n納品書(領収書)\nFILTER ELEMENT XL\n数量\n2",
            {
                "merchant": "FILTER ELEMENT",
                "line_items": [{"description": "FILTER ELEMENT XL"}],
            },
            "NORD",
        ),
        (
            "ORBIT (クレジット領収書)\nenergy\nUTILITY GRADE\n数量\n12L",
            {
                "merchant": "UTILITY GRADE",
                "line_items": [],
                "usage": {"amount": 12, "unit": "L"},
            },
            "ORBIT",
        ),
    ],
)
def test_unique_receipt_owner_header_outranks_item_or_usage_owned_merchant(
    ocr_text, extracted, expected,
):
    _fix_company_name_merchant(extracted, ocr_text)

    assert extracted["merchant"] == expected


@pytest.mark.parametrize(
    ("ocr_text", "extracted"),
    [
        (
            "NORD\nORBIT\n領収証\nFILTER ELEMENT\n100円",
            {
                "merchant": "FILTER ELEMENT",
                "line_items": [{"description": "FILTER ELEMENT"}],
            },
        ),
        (
            "NORD\n領収証\nFILTER ELEMENT\n100円",
            {
                "merchant": "CURRENT SHOP",
                "line_items": [{"description": "FILTER ELEMENT"}],
            },
        ),
    ],
)
def test_receipt_owner_header_requires_unique_candidate_and_section_ownership(
    ocr_text, extracted,
):
    before = extracted["merchant"]

    _fix_company_name_merchant(extracted, ocr_text)

    assert extracted["merchant"] == before


def test_late_unlabeled_cash_repair_cannot_override_noncash_evidence():
    extracted = {"total": 500, "amount_paid": 500, "payment_method": None}
    ocr_text = "\n".join([
        "合計", "500", "1000", "カード決済 500", "お釣り", "500",
    ])

    for _ in range(2):
        _run_cash_tender_reconciliation_phase(
            extracted,
            ocr_text,
            ("stacked_cash_tender", "unlabeled_cash_tender_change"),
        )

    assert extracted == {"total": 500, "amount_paid": 500, "payment_method": None}


@pytest.mark.parametrize(
    ("ocr_text", "expected"),
    [
        ("NEXCO\n料金所\n取扱番号 203-00841206-00", "203-00841206-00"),
        ("伝票番号\n260-418-197-1586", "260-418-197-1586"),
        ("レシートNo.017216", "017216"),
        ("取引 No:HD0020260424200941093", "HD0020260424200941093"),
    ],
)
def test_explicit_reference_labels_recover_receipt_reference(ocr_text, expected):
    extracted = {"payment_reference": None}

    _fix_payment_reference(extracted, ocr_text)

    assert extracted["payment_reference"] == expected


@pytest.mark.parametrize(
    "ocr_text",
    [
        "000000#2218",
        "登録番号 T6290001017604\nWAON番号\n**2304",
        "青 No. 008",
    ],
)
def test_reference_recovery_fails_closed_without_an_applicable_label(ocr_text):
    extracted = {"payment_reference": None}

    _fix_payment_reference(extracted, ocr_text)

    assert extracted["payment_reference"] is None


def test_reference_repair_clears_unsupported_upstream_value():
    extracted = {"payment_reference": "000000#2218"}

    _fix_payment_reference(extracted, "登録番号 T6290001017604")

    assert extracted["payment_reference"] is None


def test_reference_repair_preserves_upstream_only_when_ocr_value_matches():
    matching = {"payment_reference": "017216"}
    _fix_payment_reference(matching, "レシートNo.017216")

    mismatched = {"payment_reference": "WRONG-REFERENCE"}
    _fix_payment_reference(mismatched, "伝票番号\n260-418-197-1586")

    assert matching["payment_reference"] == "017216"
    assert mismatched["payment_reference"] == "260-418-197-1586"


def test_primary_document_reference_outranks_secondary_references():
    extracted = {"payment_reference": "5931-05"}

    _fix_payment_reference(
        extracted,
        "レシートNo 5931-05 データNo1013-1017\nクレ通番17-47476",
    )

    assert extracted["payment_reference"] == "5931-05"
