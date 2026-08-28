import pytest

from receipt_parser.pipeline_bill import postprocess_utility_bill
from receipt_parser.pipeline_slip import postprocess_payment_slip


@pytest.mark.parametrize(
    ("merchant", "text"),
    [
        ("山田太郎", "ガス検針のお知らせ\n山田太郎\n様"),
        ("OCR", "OCR\nガス検針のお知らせ"),
        ("東都エナジー", "東都エナジー\nガス検針のお知らせ"),
        ("サービス", "サービス株式会社 様\nご請求額 5000円"),
    ],
)
def test_utility_merchant_requires_same_line_provider_or_issuer_evidence(
        merchant, text):
    result = postprocess_utility_bill({"merchant": merchant}, text)

    assert result["merchant"] is None


@pytest.mark.parametrize(
    ("merchant", "provider_line"),
    [
        ("東都エナジー", "東都エナジー株式会社"),
        ("地域事務", "地域事務組合"),
        ("市公営", "市公営水道局"),
        ("北星", "北星ガス"),
        ("北星", "北星電力"),
        ("東都エナジー", "発行元: 東都エナジー"),
    ],
)
def test_utility_merchant_keeps_same_line_provider_or_issuer_evidence(
        merchant, provider_line):
    result = postprocess_utility_bill({"merchant": merchant}, provider_line)

    assert result["merchant"] == merchant


@pytest.mark.parametrize(
    "text",
    [
        "佐藤 ミカ\n様",
        "使用者氏名\n佐藤 ミカ\n様",
        "佐藤 ミカ\n1検針のお知らせ\n様",
    ],
)
def test_utility_payer_requires_one_honorific_backed_addressee(text):
    result = postprocess_utility_bill({"payer": "unsupported upstream"}, text)

    assert result["payer"] == "佐藤 ミカ"


@pytest.mark.parametrize(
    "text",
    [
        "担当者\n佐藤 ミカ\n様",
        "責任者 佐藤 ミカ 様",
        "検針のお知らせ 様",
    ],
)
def test_utility_employee_and_heading_addressees_fail_closed(text):
    result = postprocess_utility_bill({"payer": "unsupported upstream"}, text)

    assert result["payer"] is None


def test_utility_distinct_addressees_fail_closed():
    result = postprocess_utility_bill(
        {"payer": "unsupported upstream"},
        "佐藤 ミカ\n様\n田中 タロウ\n様",
    )

    assert result["payer"] is None


@pytest.mark.parametrize(
    "text",
    [
        "検針日 2027年2月5日\nご使用量 12.3m3",
        "使用者氏名\n佐藤 ミカ",
    ],
)
def test_utility_without_honorific_evidence_preserves_upstream_payer(text):
    result = postprocess_utility_bill(
        {"payer": "supported upstream"},
        text,
    )

    assert result["payer"] == "supported upstream"


def test_explicit_usage_period_wins_without_advancing_its_printed_start_date():
    extracted = {
        "date": "2027-02-20",
        "billing_period": {"start": "2027-01-07", "end": "2027-02-05"},
    }

    result = postprocess_utility_bill(
        extracted,
        "請求年月 2027年2月\n使用期間\n1月6日 ～ 2月5日",
    )

    assert result["billing_period"] == {
        "start": "2027-01-06",
        "end": "2027-02-05",
    }


def test_cross_year_explicit_usage_period_uses_the_printed_months_verbatim():
    extracted = {
        "date": "2027-01-20",
        "billing_period": {"start": "2026-12-16", "end": "2027-01-14"},
    }

    result = postprocess_utility_bill(
        extracted,
        "2027年1月分\nご使用期間 12月15日～1月14日",
    )

    assert result["billing_period"] == {
        "start": "2026-12-15",
        "end": "2027-01-14",
    }


@pytest.mark.parametrize(
    "text",
    [
        "今回\n前回\n検針日\n2027年2月5日\n2027年1月6日\nご使用量 12.3m3",
        "前回検針日 2027年1月6日\n今回検針日 2027年2月5日",
    ],
)
def test_meter_period_advances_an_exact_previous_reading_match(text):
    result = postprocess_utility_bill(
        {
            "date": "2027-02-05",
            "billing_period": {"start": "2027-01-06", "end": "2027-02-05"},
        },
        text,
    )

    assert result["billing_period"] == {
        "start": "2027-01-07",
        "end": "2027-02-05",
    }


def test_already_advanced_meter_period_is_retained():
    period = {"start": "2027-01-07", "end": "2027-02-05"}

    result = postprocess_utility_bill(
        {"date": "2027-02-20", "billing_period": dict(period)},
        "今回\n前回\n検針日\n2027年2月5日\n2027年1月6日\nご使用量 12.3m3",
    )

    assert result["billing_period"] == period


@pytest.mark.parametrize(
    "text",
    [
        "今回\n前回\n検針日\n2027年2月5日\n2027年1月6日\n2026年12月5日\nご使用量 12.3m3",
        "今回\n前回\n検針日\n2027年2月4日\n2027年1月6日\nご使用量 12.3m3",
    ],
)
def test_ambiguous_or_mismatched_meter_dates_leave_period_unchanged(text):
    period = {"start": "2027-01-06", "end": "2027-02-05"}

    result = postprocess_utility_bill(
        {"date": "2027-02-05", "billing_period": dict(period)},
        text,
    )

    assert result["billing_period"] == period


@pytest.mark.parametrize(
    "text",
    [
        "責: 佐藤ミカ 様",
        "責任者 佐藤ミカ 様",
        "担当 佐藤ミカ 様",
        "従業員 佐藤ミカ 様",
        "社員 佐藤ミカ 様",
    ],
)
def test_responsibility_and_employee_honorifics_are_not_payer_evidence(text):
    extracted = {"payer": "unsupported upstream value", "payment_reference": None}

    result = postprocess_payment_slip(extracted, text, raw_text=text)

    assert result["payer"] is None


def test_only_a_label_only_row_may_borrow_the_immediately_following_name():
    header = "依頼日 金額 受取人 ご依頼人\n収納サービス株式会社"
    label = "ご依頼人:\n佐藤 ミカ"

    assert postprocess_payment_slip(
        {"payer": "upstream", "payment_reference": None},
        header,
        raw_text=header,
    )["payer"] is None
    assert postprocess_payment_slip(
        {"payer": None, "payment_reference": None},
        label,
        raw_text=label,
    )["payer"] == "佐藤 ミカ"


def test_distinct_explicit_payer_candidates_fail_closed():
    text = "ご依頼人: 佐藤ミカ\n田中タロウ 様"

    result = postprocess_payment_slip(
        {"payer": "upstream", "payment_reference": None},
        text,
        raw_text=text,
    )

    assert result["payer"] is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("上下水道料金\n145891", None),
        ("上下水道料金\n受付番号 1234567890", "1234567890"),
        (
            "上下水道料金\n受付番号 1234567890\n参照番号 9876543210",
            None,
        ),
    ],
)
def test_utility_payment_reference_requires_one_labeled_printed_value(
        text, expected):
    result = postprocess_utility_bill(
        {"payment_reference": "unsupported upstream value"},
        text,
    )

    assert result["payment_reference"] == expected


def test_utility_reference_uses_prestrip_text_without_exposing_it_to_other_fields():
    reference = "123456789012345678901234"
    period = {"start": "2026-07-01", "end": "2026-07-31"}
    result = postprocess_utility_bill(
        {
            "merchant": "北星",
            "payer": "supported upstream",
            "payment_method": None,
            "service_type": "sewage",
            "date": "2026-08-01",
            "billing_period": dict(period),
            "payment_reference": None,
        },
        "公共料金のお知らせ",
        payment_reference_text=(
            "発行元: 北星ガス\n佐藤 ミカ\n様\n口座振替\n水道\n"
            f"領収日付 '26.8.28\n受付番号\n{reference}"
        ),
    )

    assert result == {
        "merchant": None,
        "payer": "supported upstream",
        "payment_method": None,
        "service_type": "sewage",
        "date": "2026-08-01",
        "billing_period": period,
        "payment_reference": reference,
    }


def test_utility_unlabeled_prestrip_barcode_still_clears_reference():
    result = postprocess_utility_bill(
        {"payment_reference": "unsupported upstream value"},
        "上下水道料金",
        payment_reference_text="上下水道料金\n123456789012345678901234",
    )

    assert result["payment_reference"] is None
