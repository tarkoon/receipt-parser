import pytest

from receipt_parser.receipt_identity_payment import (
    _fix_payment_reference,
    _fix_receipt_payer,
    _receipt_owner_header_candidate,
)
from receipt_parser.receipt_postprocess import postprocess_receipt
from receipt_parser.schema import Receipt


def _postprocess_payer(
    text: str,
    mutation_trace: list[dict] | None = None,
) -> str | None:
    result = postprocess_receipt(
        Receipt().model_dump(),
        text,
        0.9,
        {},
        None,
        "test-model",
        mutation_trace=mutation_trace,
    )
    return result["payer"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("領収証\nイーガン エイミー\n様\n金額\n￥3,000", "イーガン エイミー"),
        (
            "顧客No. 2301\nイーガン トリスタン\nイーガン トリスタン 様\nTEL. 090-0000-0000",
            "イーガン トリスタン",
        ),
    ],
)
def test_unique_explicit_receipt_addressee_owns_payer(text, expected):
    assert _postprocess_payer(text) == expected


def test_receipt_payer_change_has_one_declared_trace_owner():
    trace = []

    assert _postprocess_payer("領収証\n佐藤 ミカ\n様", trace) == "佐藤 ミカ"

    event = next(event for event in trace if event["stage"] == "receipt_payer_repair")
    assert set(event["changes"]) == {"payer"}
    assert event["writes"] == ("payer",)


@pytest.mark.parametrize(
    "text",
    [
        "領収 収証\n様\nキャッシャ 2018386",
        "11:46\nウエ\n様\nクレジット",
        "領収 書\n(消費税\n様\n2026年04月22日",
        "宗像店\n営業時間\n様\nTEL 0570-000-000",
    ],
)
def test_standalone_honorific_requires_formal_receipt_name_sandwich(text):
    assert _postprocess_payer(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "領収証\n担当者 青木 ミナ",
        "領収証\n青木 ミナ 様\n高橋 レン 様",
    ],
)
def test_receipt_payer_clears_upstream_without_one_explicit_addressee(text):
    extracted = {"payer": "unsupported upstream payer"}

    _fix_receipt_payer(extracted, text)

    assert extracted["payer"] is None


def test_receipt_payer_prefers_one_explicit_addressee_over_upstream_value():
    extracted = {"payer": "unsupported upstream payer"}

    _fix_receipt_payer(extracted, "領収証\n青木 ミナ 様")

    assert extracted["payer"] == "青木 ミナ"


def test_receipt_payer_accepts_explicit_customer_name_label():
    extracted = {"payer": None}

    _fix_receipt_payer(extracted, "領収証\nお客様名：青木 ミナ")

    assert extracted["payer"] == "青木 ミナ"


@pytest.mark.parametrize(
    "staff_label",
    [
        "担当者氏名",
        "責任者氏名",
        "従業員氏名",
        "社員氏名",
        "店員氏名",
        "係員氏名",
        "販売員氏名",
    ],
)
def test_receipt_payer_rejects_staff_only_name_labels(staff_label):
    extracted = {"payer": "unsupported upstream payer"}

    _fix_receipt_payer(extracted, f"領収証\n{staff_label}：青木 ミナ")

    assert extracted["payer"] is None


def test_receipt_payer_rejects_staff_context_before_any_payer_label():
    extracted = {"payer": "unsupported upstream payer"}

    _fix_receipt_payer(extracted, "領収証\n担当者 お名前：青木 ミナ")

    assert extracted["payer"] is None


def test_receipt_payer_rejects_staff_context_around_payer_label():
    extracted = {"payer": "unsupported upstream payer"}

    _fix_receipt_payer(extracted, "領収証\n氏名（担当者）：青木 ミナ")

    assert extracted["payer"] is None


@pytest.mark.parametrize(
    "status",
    [
        "DUPLICATE",
        "ORIGINAL",
        "REPRINT",
        "COPY",
        "CUSTOMER-COPY",
        "RECEIPT-COPY",
        "COPY1",
        "VOID",
        "VOIDED",
    ],
)
def test_document_status_header_is_not_a_receipt_owner(status):
    assert _receipt_owner_header_candidate([status, "領収証"], 1) is None


@pytest.mark.parametrize("brand", ["COPYLAND", "VOIDLESS"])
def test_status_word_inside_brand_is_preserved_as_receipt_owner(brand):
    assert _receipt_owner_header_candidate([brand, "領収証"], 1) == brand


def test_repeated_long_number_with_standalone_no_label_owns_reference():
    extracted = {"payment_reference": None}

    _fix_payment_reference(
        extracted,
        "お釣り\n￥20\n7649965640146\n返品のご案内\nNo. 7649965640146",
    )

    assert extracted["payment_reference"] == "7649965640146"


@pytest.mark.parametrize(
    "text",
    [
        "No. 008",
        "1234567890123\nNo. 1234567890124",
        "1234567890123\nNo. 1234567890123\nNo. 1234567890124",
    ],
)
def test_generic_no_reference_requires_one_repeated_long_value(text):
    extracted = {"payment_reference": "unsupported-upstream-value"}

    _fix_payment_reference(extracted, text)

    assert extracted["payment_reference"] is None


@pytest.mark.parametrize(
    "owner_label",
    ["合計", "お預り", "お釣り", "消費税", "ポイント"],
)
def test_blank_receipt_number_label_does_not_borrow_financial_value(owner_label):
    extracted = {"payment_reference": None}

    _fix_payment_reference(extracted, f"{owner_label}\n1000\nレシートNo.")

    assert extracted["payment_reference"] is None


def test_unowned_backward_numeric_value_can_be_a_receipt_number():
    extracted = {"payment_reference": None}

    _fix_payment_reference(extracted, "12345678\nレシートNo.")

    assert extracted["payment_reference"] == "12345678"


def test_long_backward_transaction_code_outranks_preceding_nonreference_line():
    extracted = {"payment_reference": None}

    _fix_payment_reference(
        extracted,
        "会員カード ************0000\n123456789012345678901234\n取引 CD\nポイント対象金額",
    )

    assert extracted["payment_reference"] == "123456789012345678901234"


def test_same_line_reference_compacts_all_internal_whitespace():
    extracted = {"payment_reference": None}

    _fix_payment_reference(
        extracted,
        "取引 CD: 1234 5678 9012 3456 7890 1234",
    )

    assert extracted["payment_reference"] == "123456789012345678901234"


def test_same_line_reference_rejects_trailing_nonreference_text():
    extracted = {"payment_reference": "unsupported-upstream-value"}

    _fix_payment_reference(extracted, "取引 CD: 1234 5678 9012 3456 合計")

    assert extracted["payment_reference"] is None


@pytest.mark.parametrize(
    "text",
    [
        "取引 No9454 販売員 18947/児玉",
        "取引 No9454 販売員1947/児玉",
        "取引 No9454 販売員 児玉",
        "取引 No9454 販売員",
    ],
)
def test_same_line_reference_isolates_end_anchored_salesperson_metadata(text):
    extracted = {"payment_reference": None}

    _fix_payment_reference(extracted, text)

    assert extracted["payment_reference"] == "9454"


def test_same_line_reference_isolates_named_staff_metadata():
    extracted = {"payment_reference": None}

    _fix_payment_reference(extracted, "取引 No9454 担当者 児玉")

    assert extracted["payment_reference"] == "9454"


@pytest.mark.parametrize(
    ("owner_line", "candidate"),
    [
        ("お知らせ", "1000"),
        ("合計", "10000000"),
    ],
)
def test_blank_reference_label_does_not_steal_owned_numeric_value(
    owner_line, candidate,
):
    extracted = {"payment_reference": None}

    _fix_payment_reference(extracted, f"{owner_line}\n{candidate}\nレシートNo.")

    assert extracted["payment_reference"] is None
