import pytest

from receipt_parser.pipeline_slip import postprocess_payment_slip


def _postprocess_fields(text: str, reference=None, payer=None):
    extracted = {"payment_reference": reference, "payer": payer}
    return postprocess_payment_slip(extracted, text, raw_text=text)


def _postprocess(text: str, existing=None):
    return _postprocess_fields(text, reference=existing)["payment_reference"]


def test_composite_payment_barcode_recovers_unique_13_digit_reference_segment():
    text = "\n".join(
        [
            "振込兼コンビニ請求書",
            "CVS収納用",
            "(91)123456-3141592653589271828",
            "収納代行会社",
        ]
    )

    assert _postprocess(text, existing="unsupported-upstream-value") == "3141592653589"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("参照番号: 2718281828459", "2718281828459"),
        ("収納番号\n1618033988749", "1618033988749"),
        ("払込票受領証\n手数料\n1414213562373\n受領印", "1414213562373"),
    ],
)
def test_reference_label_or_bounded_payment_block_recovers_standalone_reference(text, expected):
    assert _postprocess(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "お客様番号\n2718281828459",
        "払込期限までにお支払いください\n2718281828459",
        "払込票受領証\nカード番号\n2718281828459\n受領印",
        "払込票受領証\nバーコード\n2718281828459\n受領印",
    ],
)
def test_account_card_and_arbitrary_barcode_numbers_are_not_references(text):
    assert _postprocess(text) is None


def test_distinct_evidence_backed_references_fail_closed():
    text = "\n".join(
        [
            "参照番号 2718281828459",
            "CVS収納用",
            "123456-3141592653589271828",
            "収納代行会社",
        ]
    )

    assert _postprocess(text, existing="2718281828459") is None


def test_existing_reference_is_cleared_without_ocr_reference_evidence():
    assert _postprocess("振込金額 5000円", existing="upstream-reference") is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ご依頼人: ヤマダ ハナコ", "ヤマダ ハナコ"),
        ("氏名 タナカタロウ 様", "タナカタロウ"),
        ("サトウ ミカ 様", "サトウ ミカ"),
    ],
)
def test_unique_explicit_payer_or_addressee_replaces_upstream_value(text, expected):
    result = _postprocess_fields(text, payer="unsupported-upstream-name")

    assert result["payer"] == expected


def test_repeated_payer_with_whitespace_variation_is_one_identity():
    text = "ヤマダ ハナコ 様\nヤマダハナコ 様"

    assert _postprocess_fields(text)["payer"] == "ヤマダ ハナコ"


def test_multicolumn_payer_header_does_not_borrow_merchant_line():
    text = "\n".join(
        [
            "依頼日 金額 先方銀行 受取人 ご依頼人",
            "サービス株式会社",
            "ヤマダ ハナコ 様",
        ]
    )

    assert _postprocess_fields(text)["payer"] == "ヤマダ ハナコ"


def test_standalone_payer_label_uses_immediately_following_name():
    assert _postprocess_fields("ご依頼人:\nヤマダ ハナコ")["payer"] == "ヤマダ ハナコ"


@pytest.mark.parametrize(
    "text",
    [
        "振込金額 5000円",
        "受取人: サービス株式会社",
        "責任者 サトウ ミカ 様",
        "ヤマダ ハナコ 様\nタナカ タロウ 様",
    ],
)
def test_absent_recipient_only_or_ambiguous_payer_evidence_clears_upstream(text):
    assert _postprocess_fields(text, payer="unsupported-upstream-name")["payer"] is None
