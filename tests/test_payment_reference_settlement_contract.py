import pytest

from receipt_parser import pipeline
from receipt_parser.receipt_identity_payment import (
    _fix_company_name_merchant,
    _fix_payment_method,
    _fix_payment_reference,
    _merchant_looks_invalid,
)


@pytest.mark.parametrize(
    ("ocr_text", "expected"),
    [
        (
            "取引番号 TX-900\nレシートNo. 00042\nデータNo. DATA-7",
            "00042",
        ),
        ("取扱番号 203-00841206-00", "203-00841206-00"),
        ("伝票番号\n02912", "02912"),
    ],
)
def test_reference_is_value_only_and_unique_primary_outranks_secondary(
    ocr_text, expected,
):
    extracted = {"payment_reference": "unsupported-upstream-value"}

    _fix_payment_reference(extracted, ocr_text)

    assert extracted["payment_reference"] == expected


@pytest.mark.parametrize(
    "ocr_text",
    [
        "レシートNo 00042\n伝票番号 00043\n取引番号 TX-900",
        "レシートNo 00042 伝票番号 00043",
        "取引番号 TX-900\nデータNo DATA-7",
        "青 No. 008\n登録番号 T1234567890123",
    ],
)
def test_reference_clears_ambiguous_or_unlabeled_evidence(ocr_text):
    extracted = {"payment_reference": "unsupported-upstream-value"}

    _fix_payment_reference(extracted, ocr_text)

    assert extracted["payment_reference"] is None


def test_unresolved_loyalty_reference_label_does_not_mask_receipt_transaction_number():
    extracted = {"payment_reference": None}

    _fix_payment_reference(
        extracted,
        "楽天ポイントカード\n取引CD\n楽天ポイント明細\n取引No89787点買",
    )

    assert extracted["payment_reference"] == "89787"


@pytest.mark.parametrize(
    ("ocr_text", "expected"),
    [
        ("取引CD Z2", "Z2"),
        ("取引 コード\nAB-42", "AB-42"),
        ("AB-42\n取引コード", "AB-42"),
        ("領 No 42 担当者 7", "42"),
        ("取引No89787点買", "89787"),
    ],
)
def test_reference_supports_short_clean_and_bidirectional_values(
    ocr_text, expected,
):
    extracted = {"payment_reference": None}

    _fix_payment_reference(extracted, ocr_text)

    assert extracted["payment_reference"] == expected


def test_loyalty_prefixed_document_number_does_not_compete_with_receipt_number():
    extracted = {"payment_reference": None}

    _fix_payment_reference(
        extracted,
        "伝票 No 0631\ndポイントレシート番号 00608200",
    )

    assert extracted["payment_reference"] == "0631"


def test_reference_value_requires_a_digit_before_adjacent_document_label():
    extracted = {"payment_reference": None}

    _fix_payment_reference(
        extracted,
        "PAYMENT\n支払伝票番号\n77840023260722120851",
    )

    assert extracted["payment_reference"] == "77840023260722120851"


@pytest.mark.parametrize(
    ("ocr_text", "expected"),
    [
        (
            "レシートNo7803\nTID:\nTERM0000002169\n伝票番号:\n50\n"
            "決済用ユニーク番号:\n717803",
            "7803",
        ),
        (
            "レシートNo9979\n決済用ユニーク番号:\n869979\n伝票番号:\n*******",
            "9979",
        ),
        (
            "レシートNo8089\n会員番号 XXXX4503\nお取扱日\n伝票番号\n"
            "2026年04月24日\n93786\n取引内容",
            "8089",
        ),
    ],
)
def test_detached_card_column_label_does_not_clear_unique_receipt_number(
    ocr_text, expected,
):
    extracted = {"payment_reference": None}

    _fix_payment_reference(extracted, ocr_text)

    assert extracted["payment_reference"] == expected


def test_supported_card_slip_and_unresolved_competing_primary_fail_closed():
    extracted = {"payment_reference": "unsupported-upstream-value"}

    _fix_payment_reference(
        extracted,
        "カード売上票\n取引内容\nお買上\n伝票番号\n00007\n取扱区分\n110\n"
        "決済用ユニーク番号:\n676547\n伝票番号:\nーシートNo6547",
    )

    assert extracted["payment_reference"] is None


@pytest.mark.parametrize(
    "ocr_text",
    [
        "レシートNo.0150\n伝票番号\n4874",
        "伝票番号\n00007\nご案内\n伝票番号:\nーシートNo6547",
        "取引 CD\n202607071839540025700025889\n取引 No5889 4点買",
    ],
)
def test_competing_reference_labels_fail_closed(ocr_text):
    extracted = {"payment_reference": "unsupported-upstream-value"}

    _fix_payment_reference(extracted, ocr_text)

    assert extracted["payment_reference"] is None


def test_normalized_pre_strip_reference_text_reaches_receipt_postprocess(monkeypatch):
    monkeypatch.setattr(pipeline, "check_model_available", lambda _model: None)
    monkeypatch.setattr(
        pipeline,
        "extract_with_verification",
        lambda *_args, **_kwargs: (
            {
                "merchant": "テスト店",
                "date": "2026-01-01",
                "currency": "JPY",
                "total": 100,
                "subtotal": 100,
                "taxes": [],
                "line_items": [],
                "payment_reference": "unsupported-upstream-value",
            },
            [],
        ),
    )

    result = pipeline.process_ocr_text(
        "テスト店\n領収書\nレシートＮｏ\n１２３４５６７８９０\n合計 100",
        model="test-model",
        apply_user_rules=False,
    )

    assert result["payment_reference"] == "1234567890"


@pytest.mark.parametrize(
    ("total", "ocr_text", "expected"),
    [
        (1200, "合計\nクレジット\nお釣り\n1,200\n1,200\n0", "credit"),
        (1089, "合計\n1,089\nクレジット\n現計\n1,089\n0", "credit"),
        (1078, "合計\nお預り\nお釣り\n1,078\n1,080\n2", "cash"),
        (510, "通行料金\n¥510\n(クレジット)", "credit"),
        (860, "通行料金\n¥860\n(現金)", "cash"),
        (12500, "信用 1\n¥12,500", "credit"),
    ],
)
def test_exact_tender_and_bounded_column_amount_must_reconcile(
    total, ocr_text, expected,
):
    extracted = {"total": total, "amount_paid": total, "payment_method": None}

    _fix_payment_method(extracted, ocr_text, 0.9, {})

    assert extracted["payment_method"] == expected


def test_named_noncash_tender_outranks_generic_deposit_label():
    extracted = {"total": 19118, "amount_paid": 19118, "payment_method": None}
    ocr_text = "\n".join([
        "預り", "¥19,118", "カード会社", "Mastercard", "支払方法", "¥19,118",
    ])

    _fix_payment_method(extracted, ocr_text, 0.9, {})

    assert extracted["payment_method"] == "credit"


def test_cash_tender_change_can_reconcile_to_amount_paid_after_points():
    extracted = {"total": 5000, "amount_paid": 200, "payment_method": None}

    _fix_payment_method(
        extracted,
        "\n".join([
            "合計 5,000", "現金計", "お釣り", "取引日時", "会員番号",
            "獲得ポイント", "利用ポイント 4,800P", "利用可能ポイント",
            "処理日付", "税率対象", "ポイント利用", "200", "1,000", "800",
        ]),
        0.9,
        {},
    )

    assert extracted["payment_method"] == "cash"


def test_reconciled_mixed_tender_and_zero_tender_fail_closed():
    mixed = {"total": 747, "amount_paid": 747, "payment_method": "WAON"}
    _fix_payment_method(
        mixed,
        "合計 747\nWAON支払 691\n現金 100\nお釣り 44",
        0.9,
        {},
    )

    zero = {"total": 500, "amount_paid": 500, "payment_method": "cash"}
    _fix_payment_method(zero, "現金 0", 0.9, {})

    assert mixed["payment_method"] is None
    assert zero["payment_method"] is None


@pytest.mark.parametrize(
    "ocr_text",
    [
        "WAON支払いできます",
        "クレジットカード会員募集中\n合計 500",
        "電子マネーチャージ 500\n合計 500",
    ],
)
def test_capability_advertising_and_loyalty_text_are_not_tenders(ocr_text):
    extracted = {"total": 500, "amount_paid": 500, "payment_method": "credit"}

    _fix_payment_method(extracted, ocr_text, 0.9, {})

    assert extracted["payment_method"] is None


def test_domain_only_text_is_not_a_merchant_name():
    assert _merchant_looks_invalid("WWW.EXAMPLE.COM")
    assert _merchant_looks_invalid("shop.example.jp/receipt")
    assert not _merchant_looks_invalid("EXAMPLE")


def test_ascii_logo_immediately_preceding_legal_company_outranks_company_name():
    extracted = {"merchant": "高祖自転車商会"}
    ocr_text = "\n".join([
        "clink",
        "75",
        "clickcycle.com",
        "有限会社 高祖自転車商会",
    ])

    _fix_company_name_merchant(extracted, ocr_text)

    assert extracted["merchant"] == "clink"


def test_legal_company_fallback_rejects_domain_number_and_metadata_as_logo():
    extracted = {"merchant": "青空商会"}
    ocr_text = "\n".join([
        "ID",
        "12345",
        "shop.example.jp",
        "有限会社 青空商会",
    ])

    _fix_company_name_merchant(extracted, ocr_text)

    assert extracted["merchant"] == "青空商会"


def test_latin_fuel_brand_requires_adjacent_facility_and_operator_header():
    anchored = {"merchant": "セルフ中央給油所"}
    ocr_text = "\n".join([
        "NOVA (クレジット領収書)",
        "energy",
        "-143641",
        "セルフ中央給油所",
        "株式会社 運営サービス",
    ])
    _fix_company_name_merchant(anchored, ocr_text)

    unanchored = {"merchant": "セルフ中央給油所"}
    _fix_company_name_merchant(
        unanchored,
        "NOVA\nenergy\n-143641\nセルフ中央給油所\nTEL 000-0000",
    )

    assert anchored["merchant"] == "NOVAenergy"
    assert unanchored["merchant"] == "セルフ中央給油所"
