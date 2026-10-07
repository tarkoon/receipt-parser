"""Focused tests for document-field and prompt contracts."""

import pytest

from receipt_parser.schema import (
    Document,
    LineItem,
    Receipt,
    generate_extraction_prompt,
    generate_verification_prompt,
)


@pytest.mark.parametrize("document_type", ["receipt", "utility_bill", "payment_slip"])
def test_alias_normalization_preserves_iso_gregorian_year_without_printed_era(document_type):
    for date in ("0007-02-03", "2001-02-03", "2016-02-03", "2018-02-03", "2026-02-03"):
        document = Document(document_type=document_type, date=date, total=100, currency="JPY")
        assert document.date == date


@pytest.mark.parametrize(
    ("document_type", "included", "excluded"),
    [
        (
            "receipt",
            {"time", "usage", "payer", "payment_reference"},
            {"service_type", "billing_period"},
        ),
        (
            "utility_bill",
            {
                "time",
                "usage",
                "service_type",
                "billing_period",
                "payer",
                "payment_reference",
            },
            set(),
        ),
        (
            "payment_slip",
            {"time", "payer", "payment_reference"},
            {"usage", "service_type", "billing_period"},
        ),
    ],
)
def test_document_type_prompts_expose_only_approved_field_scopes(
    document_type, included, excluded
):
    extraction_prompt, _ = generate_extraction_prompt("", document_type)
    verification_prompt, _ = generate_verification_prompt(
        "", {"document_type": document_type}, []
    )

    for prompt in (extraction_prompt, verification_prompt):
        for field in included:
            assert f"- {field}:" in prompt
        for field in excluded:
            assert f"- {field}:" not in prompt


def test_receipt_keeps_cross_type_fields_and_clears_utility_only_fields():
    document = Receipt(
        document_type="receipt",
        time="09:32",
        service_type="electric",
        billing_period={"start": "2026-01-01", "end": "2026-01-31"},
        usage={"amount": 12.5, "unit": "L", "cost_per": 180},
        payer="Named Payer",
        payment_reference="REF-42",
    )

    assert document.time == "9:32"
    assert document.service_type is None
    assert document.billing_period is None
    assert document.usage.amount == 12.5
    assert document.payer == "Named Payer"
    assert document.payment_reference == "REF-42"


def test_utility_bill_keeps_utility_and_cross_type_fields():
    document = Receipt(
        document_type="utility_bill",
        time="14:05",
        service_type="electric",
        billing_period={"start": "2026-02-01", "end": "2026-02-28"},
        usage={"amount": 40, "unit": "kWh"},
        payer="Named Addressee",
        payment_reference="REF-84",
    )

    assert document.time == "14:05"
    assert document.service_type == "electric"
    assert document.billing_period.start == "2026-02-01"
    assert document.usage.amount == 40
    assert document.payer == "Named Addressee"
    assert document.payment_reference == "REF-84"


def test_payment_slip_clears_usage_and_utility_only_fields():
    document = Receipt(
        document_type="payment_slip",
        time="8:07",
        service_type="water",
        billing_period={"start": "2026-03-01", "end": "2026-03-31"},
        usage={"amount": 7, "unit": "m3"},
        payer="Named Payer",
        payment_reference="REF-126",
    )

    assert document.time == "8:07"
    assert document.service_type is None
    assert document.billing_period is None
    assert document.usage is None
    assert document.payer == "Named Payer"
    assert document.payment_reference == "REF-126"


@pytest.mark.parametrize("usage", [None, {}, {
    "amount": None, "unit": None, "cost_per": None,
    "meter_previous": None, "meter_current": None,
}])
def test_empty_usage_has_one_public_representation(usage):
    document = Receipt(usage=usage)

    assert document.model_dump()["usage"] is None


@pytest.mark.parametrize("usage", [
    {"amount": 0}, {"cost_per": 0}, {"meter_previous": 0},
    {"meter_current": 0}, {"unit": "L"},
])
def test_usage_normalization_preserves_explicit_zero_readings_and_units(usage):
    document = Receipt(usage=usage)

    assert document.usage is not None
    for field, value in usage.items():
        assert document.model_dump()["usage"][field] == value


@pytest.mark.parametrize("document_type", ["receipt", "utility_bill", "payment_slip"])
def test_account_number_remains_optional_across_document_types(document_type):
    document = Receipt(document_type=document_type, account_number="ACCOUNT-1")

    assert document.account_number == "ACCOUNT-1"


@pytest.mark.parametrize("document_type", ["receipt", "utility_bill", "payment_slip"])
def test_numeric_optional_account_number_cannot_reject_document(document_type):
    document = Receipt(
        document_type=document_type,
        merchant="Issuer",
        total=1000,
        account_number=12345,
    )

    assert document.merchant == "Issuer"
    assert document.total == 1000
    assert document.account_number == "12345"


def test_structured_optional_account_number_is_dropped_not_document():
    document = Receipt(
        document_type="receipt",
        merchant="Issuer",
        total=1000,
        account_number={"unexpected": "shape"},
    )

    assert document.merchant == "Issuer"
    assert document.total == 1000
    assert document.account_number is None


@pytest.mark.parametrize(
    ("raw_rate", "expected"),
    [
        (20, "20%"),
        ("20.0", "20%"),
        ("-12.5％", "12.5%"),
        (0, ""),
        ("not a rate", ""),
    ],
)
def test_discount_rate_is_a_positive_canonical_percentage(raw_rate, expected):
    item = LineItem(
        description="line",
        qty=1,
        unit_price=100,
        total=80,
        discount=20,
        discount_rate=raw_rate,
    )

    assert item.discount_rate == expected


def test_discount_rate_is_empty_without_an_effective_discount():
    item = LineItem(
        description="line",
        qty=1,
        unit_price=100,
        total=100,
        discount=0,
        discount_rate="20%",
    )

    assert item.discount_rate == ""


def test_prompts_state_the_approved_semantic_conventions():
    receipt_prompt, _ = generate_extraction_prompt("", "receipt")
    utility_prompt, _ = generate_extraction_prompt("", "utility_bill")

    assert "most-specific reliable location exactly as printed" in receipt_prompt
    assert "Use null for mixed tender" in receipt_prompt
    assert "Use null when no redemption is printed" in receipt_prompt
    assert "positive effective percentage" in receipt_prompt
    assert "supported by a printed percentage schedule" in receipt_prompt
    assert "empty for amount-only discounts, including bundle offers" in receipt_prompt
    assert "payment_reference is value-only" in receipt_prompt
    assert "receipt or slip number" in receipt_prompt
    assert "calendar day after the previous reading" in utility_prompt
