"""Focused structural contracts for receipt location recovery."""

from receipt_parser.receipt_location import (
    _recover_ascii_brand_header_location,
    _recover_header_branch_store_location,
)


def test_staff_number_suffix_yields_to_contact_backed_printed_locality():
    extracted = {"merchant": "架空商店", "location": "木村"}
    ocr_text = "北丘\n架空商店 (012) 345-6789\nスNo00000107木村\n合計"

    _recover_header_branch_store_location(extracted, ocr_text)

    assert extracted["location"] == "北丘"


def test_contact_center_yields_to_repeated_ascii_header_location():
    extracted = {"merchant": "NORD北丘中央", "location": "カスタマーサポートセンター"}
    ocr_text = "NORD\n印紙税申告納\nNORD 北丘中央\nカスタマーサポートセンター\nTEL 012-345-6789"

    _recover_header_branch_store_location(extracted, ocr_text)
    _recover_ascii_brand_header_location(extracted, ocr_text)

    assert extracted["location"] == "北丘中央"


def test_contact_center_is_cleared_when_no_location_is_printed():
    extracted = {"merchant": "NORD", "location": "カスタマーサポートセンター"}
    ocr_text = "NORD\nカスタマーサポートセンター\nTEL 012-345-6789\n合計"

    _recover_header_branch_store_location(extracted, ocr_text)
    _recover_ascii_brand_header_location(extracted, ocr_text)

    assert extracted.get("location") is None


def test_contact_backed_one_character_store_stem_is_kept_verbatim():
    extracted = {"merchant": "架空商店", "location": None}
    ocr_text = "架空商店\n丘店 TEL 0940-72-5355\n合計"

    _recover_header_branch_store_location(extracted, ocr_text)

    assert extracted["location"] == "丘店"


def test_split_merchant_header_keeps_preceding_printed_locality():
    for split_header in ("北丘\nマート", "マート\n北丘"):
        extracted = {"merchant": "北丘マート", "location": None}
        ocr_text = f"you me\n{split_header}\n0940-35-6111\n領収証"

        _recover_header_branch_store_location(extracted, ocr_text)

        assert extracted["location"] == "北丘"


def test_generic_head_office_label_is_not_a_one_character_branch():
    extracted = {"merchant": "架空商店", "location": None}
    ocr_text = "架空商店\n本店 TEL 012-345-6789\n合計"

    _recover_header_branch_store_location(extracted, ocr_text)

    assert extracted.get("location") is None
