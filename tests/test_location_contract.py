"""Structural contracts for receipt-header location recovery."""

import pytest

from receipt_parser.receipt_late_repairs import _fix_split_address_location_from_ocr
from receipt_parser.receipt_location import (
    _location_has_ocr_evidence,
    _normalize_noisy_city_location,
    _recover_ascii_brand_header_location,
    _recover_header_branch_store_location,
    _resolve_location,
    _trim_purchase_store_metadata_location,
)


def test_exact_source_location_canonicalization_preserves_suffix_and_address_digits():
    from copy import deepcopy

    for merchant, location, text, expected in [
        ("SORA", "ソラ 青葉中央店", "ソラ 青葉中央店\n領収証", "青葉中央店"),
        ("SORA", "SORA 青葉中央店", "SORA 青葉中央店\n領収証", "SORA 青葉中央店"),
        ("ソラ", "ソラ 青葉中央店", "ソラ 青葉中央店\n領収証", "ソラ 青葉中央店"),
        ("SORA", "ソラ 青葉中央店", "ソラ 青葉中央店\nソラ 青葉中央店", "ソラ 青葉中央店"),
        ("SORA", "ソラ 青葉中央店", "ソラ 別の店", "ソラ 青葉中央店"),
        ("SORA", "ソラ 青葉 中央店", "ソラ 青葉 中央店", "ソラ 青葉 中央店"),
        ("SORA", "青葉県北丘市中央 12-34-5 1F", "青葉県北丘市中央 12-34-5 1F", "青葉県北丘市中央12-34-5 1F"),
        ("SORA", "青葉県北丘市Central Hall 12-34-5", "青葉県北丘市Central Hall 12-34-5", "青葉県北丘市Central Hall 12-34-5"),
        ("SORA", "青葉県北丘市中央 1F", "青葉県北丘市中央 1F", "青葉県北丘市中央 1F"),
        ("SORA", "青葉県北丘市中央 12-34-5", "青葉県北丘市中央 12-34-5\n青葉県北丘市中央 12-34-5", "青葉県北丘市中央 12-34-5"),
    ]:
        extracted = {"merchant": merchant, "location": location, "total": 123, "line_items": [{"description": "商品60", "total": 123}]}
        other_fields = {key: deepcopy(value) for key, value in extracted.items() if key != "location"}
        for _ in range(2):
            _recover_header_branch_store_location(extracted, text)
            _normalize_noisy_city_location(extracted, text)
            assert extracted["location"] == expected
            assert {key: value for key, value in extracted.items() if key != "location"} == other_fields


def test_header_contact_recovers_clean_standalone_locality_without_phone_inference():
    extracted = {"merchant": "ゆめマート", "location": None}
    ocr_text = "\n".join([
        "北丘",
        "you me",
        "マート",
        "TEL 000-0000",
        "領収証",
    ])

    _recover_header_branch_store_location(extracted, ocr_text)

    assert extracted["location"] == "北丘"


def test_header_contact_recovers_locality_trailing_the_exact_merchant():
    extracted = {"merchant": "架空商店", "location": None}
    ocr_text = "架空商店南丘\nTEL 000-0000\n領収証"

    _recover_header_branch_store_location(extracted, ocr_text)

    assert extracted["location"] == "南丘"


def test_header_recovers_branch_suffix_trailing_the_exact_merchant():
    extracted = {"merchant": "架空商店", "location": None}
    ocr_text = "架空商店南丘店\nTEL 000-0000\n領収証"

    _recover_header_branch_store_location(extracted, ocr_text)

    assert extracted["location"] == "南丘店"


@pytest.mark.parametrize(
    ("printed_location", "expected"),
    [
        ("モール北丘ショップ", "モール北丘ショップ"),
        ("青葉営業所", "青葉営業所"),
    ],
)
def test_header_contact_recovers_explicit_branch_and_facility_suffixes(
    printed_location,
    expected,
):
    extracted = {"merchant": "BRAND", "location": None}
    ocr_text = f"BRAND\n{printed_location}\nTEL 000-0000\n合計"

    _recover_header_branch_store_location(extracted, ocr_text)

    assert extracted["location"] == expected


def test_purchase_label_cannot_block_nearby_printed_branch():
    extracted = {"merchant": "BRAND", "location": "購入店"}
    ocr_text = "\n".join([
        "BRAND",
        "購入店舗にレシートをご持参ください",
        "北丘店",
        "TEL 000-0000",
    ])

    _recover_header_branch_store_location(extracted, ocr_text)

    assert extracted["location"] == "北丘店"


def test_corporate_hq_address_yields_to_nearby_printed_branch():
    extracted = {
        "merchant": "BRAND",
        "location": "青葉都西区中央1-2-3",
    }
    ocr_text = "\n".join([
        "BRAND",
        "本社 青葉都西区中央1-2-3",
        "北丘店",
        "TEL 000-0000",
    ])

    _recover_header_branch_store_location(extracted, ocr_text)

    assert extracted["location"] == "北丘店"


def test_ascii_brand_location_yields_only_when_current_value_is_stamp_noise():
    ocr_text = "\n".join([
        "NORD",
        "印紙税申告納",
        "付につき北浜",
        "NORD南丘中央",
        "TEL 000-0000",
    ])
    extracted = {"merchant": "NORD", "location": "北浜"}

    _recover_ascii_brand_header_location(extracted, ocr_text)

    assert extracted["location"] == "南丘中央"


def test_ascii_brand_location_accepts_a_real_branch_ending_in_store_suffix():
    extracted = {"merchant": "NORD", "location": None}
    ocr_text = "NORD北丘店\nTEL 000-0000"

    _recover_ascii_brand_header_location(extracted, ocr_text)

    assert extracted["location"] == "北丘店"


@pytest.mark.parametrize(
    "ocr_text",
    [
        "福岡\n赤間",
        "福岡市",
        "福岡市\n別の地域",
    ],
)
def test_location_evidence_rejects_missing_admin_component_or_tail(ocr_text):
    assert not _location_has_ocr_evidence("福岡市赤間", ocr_text)


def test_location_evidence_accepts_every_full_component_and_tail():
    ocr_text = "青葉県\n北丘市\n中央1-2-3"

    assert _location_has_ocr_evidence("青葉県北丘市中央1-2-3", ocr_text)


@pytest.mark.parametrize("facility_label", ["料金所", "営業所"])
def test_facility_resolution_preserves_the_printed_name(facility_label):
    resolved, warning = _resolve_location(
        {"merchant": "ROAD", "location": None},
        f"ROAD\n{facility_label}\n北丘\n合計",
        "unused-model",
    )

    assert resolved == "北丘"
    assert warning is None


@pytest.mark.parametrize("merchant_line", ["架空商店", "架空商店 TEL 03-1234-5678"])
def test_location_resolution_skips_a_merchant_header_that_ends_in_store(
    monkeypatch,
    merchant_line,
):
    def unexpected_llm_call(*_args, **_kwargs):
        pytest.fail("merchant-only header must not trigger location inference")

    monkeypatch.setattr("receipt_parser.llm._llm_chat", unexpected_llm_call)

    resolved, _warning = _resolve_location(
        {"merchant": "架空商店", "location": None},
        f"{merchant_line}\n領収証\n合計\n100",
        "unused-model",
    )

    assert resolved is None


def test_header_repair_preserves_exact_visible_address():
    address = "青葉県北丘市中央1-2-3"
    extracted = {"merchant": "BRAND", "location": address}
    ocr_text = f"BRAND\n{address}\n北丘店\nTEL 000-0000"

    _recover_header_branch_store_location(extracted, ocr_text)

    assert extracted["location"] == address


def test_hq_copy_does_not_erase_location_repeated_on_a_clean_line():
    address = "青葉県北丘市中央1-2-3"
    extracted = {"merchant": "BRAND", "location": address}
    ocr_text = f"本社 {address}\n配送先 {address}\nTEL 000-0000"

    _recover_header_branch_store_location(extracted, ocr_text)

    assert extracted["location"] == address


def test_generic_visit_label_is_cleared_without_a_printed_location():
    extracted = {"merchant": "BRAND", "location": "ご来店"}
    ocr_text = "BRAND\nご来店ありがとうございます\nTEL 000-0000"

    _recover_header_branch_store_location(extracted, ocr_text)

    assert extracted.get("location") is None


@pytest.mark.parametrize("generic_header", ["北丘モール", "生活センター", "文化館"])
def test_uncontacted_generic_facility_yields_to_later_explicit_store_branch(
    generic_header,
):
    extracted = {"merchant": "架空ブランド", "location": None}
    ocr_text = f"{generic_header}\n架空ブランド\n南丘店\nTEL 000-0000"

    _recover_header_branch_store_location(extracted, ocr_text)

    assert extracted["location"] == "南丘店"


def test_unprinted_bare_city_maps_only_to_same_terminal_store_on_purchase_label():
    extracted = {"merchant": "架空商店", "location": "北丘市"}
    ocr_text = "架空商店\nご購入店 架空館北丘店。\nTEL 000-0000"

    _trim_purchase_store_metadata_location(extracted, ocr_text)

    assert extracted["location"] == "北丘店"


def test_metadata_tail_trim_continues_to_same_terminal_store_recovery():
    extracted = {"merchant": "架空商店", "location": "北丘市架空館"}
    ocr_text = "架空商店\nご購入店 架空館北丘店\nTEL 000-0000"

    _trim_purchase_store_metadata_location(extracted, ocr_text)

    assert extracted["location"] == "北丘店"


def test_unprinted_bare_city_does_not_adopt_a_different_purchase_store():
    extracted = {"merchant": "架空商店", "location": "北丘市"}
    ocr_text = "架空商店\nご購入店 架空館南丘店\nTEL 000-0000"

    _trim_purchase_store_metadata_location(extracted, ocr_text)

    assert extracted["location"] == "北丘市"


@pytest.mark.parametrize("punctuation", ["。", "!", "："])
def test_exact_header_store_token_strips_harmless_trailing_punctuation(punctuation):
    extracted = {"merchant": "架空商店", "location": None}
    ocr_text = f"架空商店\n北丘店{punctuation}\nTEL 000-0000"

    _recover_header_branch_store_location(extracted, ocr_text)

    assert extracted["location"] == "北丘店"


def test_location_evidence_ignores_harmless_trailing_token_punctuation():
    assert _location_has_ocr_evidence("北丘店。", "架空商店\n北丘店")


@pytest.mark.parametrize("street", ["3丁目3-2", "3番3号", "3-2"])
def test_split_address_joins_unique_ward_locality_and_numeric_continuation(street):
    extracted = {"merchant": "架空商店", "location": "北丘市南浜区"}
    ocr_text = f"架空商店\n青葉県北丘市南浜区中央\n{street}\nTEL 000-0000"

    _fix_split_address_location_from_ocr(extracted, ocr_text)

    assert extracted["location"] == f"青葉県北丘市南浜区中央{street}"


def test_split_address_fails_closed_when_two_distinct_pairs_are_printed():
    extracted = {"merchant": "架空商店", "location": None}
    ocr_text = "\n".join([
        "青葉県北丘市南浜区中央",
        "3丁目3-2",
        "若葉県東丘市北浜区緑町",
        "4丁目5-6",
    ])

    _fix_split_address_location_from_ocr(extracted, ocr_text)

    assert extracted["location"] is None


@pytest.mark.parametrize(
    "ocr_text",
    [
        "本社 青葉県北丘市南浜区中央\n3丁目3-2",
        "青葉県北丘市南浜区中央\n090-1234-5678",
        "青葉県北丘市南浜区中央\n803-0845",
    ],
)
def test_split_address_rejects_header_noise_phone_and_postcode(ocr_text):
    extracted = {"merchant": "架空商店", "location": None}

    _fix_split_address_location_from_ocr(extracted, ocr_text)

    assert extracted["location"] is None


def test_split_address_joins_prefecture_ward_locality_without_city_token():
    extracted = {"merchant": "架空商店", "location": "新宿区"}
    ocr_text = "架空商店\n東京都新宿区西新宿\n2丁目8-1\nTEL 000-0000"

    _fix_split_address_location_from_ocr(extracted, ocr_text)

    assert extracted["location"] == "東京都新宿区西新宿2丁目8-1"


@pytest.mark.parametrize(
    "scoped_label",
    ["本社", "本店所在地", "配送先", "請求先", "お問い合わせ"],
)
def test_split_address_rejects_address_under_non_purchase_scope(scoped_label):
    extracted = {"merchant": "架空商店", "location": None}
    ocr_text = f"{scoped_label}\n\n青葉県北丘市南浜区中央\n3丁目3-2"

    _fix_split_address_location_from_ocr(extracted, ocr_text)

    assert extracted["location"] is None


@pytest.mark.parametrize("prefix", ["", "ご購入店\n"])
def test_split_address_allows_unlabeled_and_purchase_store_address(prefix):
    extracted = {"merchant": "架空商店", "location": None}
    ocr_text = f"{prefix}青葉県北丘市南浜区中央\n3丁目3-2"

    _fix_split_address_location_from_ocr(extracted, ocr_text)

    assert extracted["location"] == "青葉県北丘市南浜区中央3丁目3-2"


def test_split_address_rejects_iso_date_as_numeric_continuation():
    extracted = {"merchant": "架空商店", "location": None}
    ocr_text = "東京都新宿区西新宿\n2026-08-25"

    _fix_split_address_location_from_ocr(extracted, ocr_text)

    assert extracted["location"] is None


def test_split_address_upgrades_admin_fragment_from_unique_printed_bare_lot():
    extracted = {"merchant": "架空商店", "location": "北丘市"}
    ocr_text = "領収証\n登録番号\n北丘市南浜1772\n架空商店"

    _fix_split_address_location_from_ocr(extracted, ocr_text)

    assert extracted["location"] == "北丘市南浜1772"


def test_split_address_rejects_ambiguous_printed_bare_lots():
    extracted = {"merchant": "架空商店", "location": "北丘市"}
    ocr_text = "北丘市南浜1772\n北丘市中央1234"

    _fix_split_address_location_from_ocr(extracted, ocr_text)

    assert extracted["location"] == "北丘市"


@pytest.mark.parametrize("scoped_label", ["本社", "本店所在地", "配送先", "請求先"])
def test_split_address_rejects_bare_lot_under_non_purchase_scope(scoped_label):
    extracted = {"merchant": "架空商店", "location": "北丘市"}
    ocr_text = f"{scoped_label}\n北丘市南浜1772\n架空商店"

    _fix_split_address_location_from_ocr(extracted, ocr_text)

    assert extracted["location"] == "北丘市"


def test_labeled_purchase_store_preserves_full_printed_site_name():
    from receipt_parser.receipt_late_repairs import (
        _recover_labeled_purchase_site_location,
    )

    extracted = {"merchant": "架空商店", "location": "北丘市"}
    ocr_text = "架空商店\nご購入店 青葉モール北丘店\nTEL 000-0000"

    _recover_labeled_purchase_site_location(extracted, ocr_text)

    assert extracted["location"] == "青葉モール北丘店"


@pytest.mark.parametrize(
    "longer_label",
    ["購入店舗", "購入店名", "購入店コード"],
)
def test_purchase_store_label_does_not_prefix_match_longer_token(longer_label):
    from receipt_parser.receipt_late_repairs import (
        _recover_labeled_purchase_site_location,
    )

    extracted = {"merchant": "架空商店", "location": "北丘市"}
    ocr_text = f"架空商店\n{longer_label} 青葉店\nTEL 000-0000"

    _recover_labeled_purchase_site_location(extracted, ocr_text)

    assert extracted["location"] == "北丘市"


def test_purchase_store_disclaimer_is_not_location_evidence():
    from receipt_parser.receipt_late_repairs import (
        _recover_labeled_purchase_site_location,
    )

    extracted = {"merchant": "架空商店", "location": "北丘市"}
    ocr_text = "購入店舗にレシートをご持参ください\nTEL 000-0000"

    _recover_labeled_purchase_site_location(extracted, ocr_text)

    assert extracted["location"] == "北丘市"
def test_complete_literal_store_header_address_owns_only_its_row():
    from copy import deepcopy
    from receipt_parser.receipt_output import (
        _record_final_receipt_output_repair, _run_final_header_location_repair_phase,
    )
    from receipt_parser.receipt_supplemental_ocr import _propose_location

    merchant, branch = "EXAMPLE", "試験町店"
    address = "架空県見本市試験町42-19"
    other_address = "架空県見本市別町71-23"
    registration = "登録番号: T1357913579135"
    phone = "電話 012-345-6789 店コード:246810"
    boundary = "2028年02月14日 11:23"
    rows = [merchant, branch, registration, address, "6", phone, boundary, "領収証"]
    original = {"merchant": merchant, "location": None, "date": None, "time": None,
                "subtotal": 731, "total": 731, "taxes": [], "line_items": []}

    for current in (None, address + "-6", "見本市"):
        repaired = deepcopy(original)
        repaired["location"] = current
        before = deepcopy(repaired)
        trace = []
        _record_final_receipt_output_repair(
            "header_branch_store_location", repaired, trace,
            lambda: _run_final_header_location_repair_phase(
                repaired, "\n".join(rows), ("header_branch_store_location",),
            ),
        )
        assert repaired["location"] == address
        assert {key: value for key, value in repaired.items() if key != "location"} == {
            key: value for key, value in before.items() if key != "location"
        }
        assert len(trace) == 1 and set(trace[0]["changes"]) == {"location"}
        assert trace[0]["owner_phase"] == "header_identity_repair"
    preserved = deepcopy(original)
    preserved["location"] = other_address
    _run_final_header_location_repair_phase(
        preserved, "\n".join(rows + [other_address]), ("header_branch_store_location",),
    )
    assert preserved["location"] == other_address
    rejected_rows = [
        rows[:4] + [other_address] + rows[4:],
        rows[:4] + [address] + rows[4:],
        rows[:3] + ["架空県見本市試験町42-", "19", "6"] + rows[5:],
        rows[:3] + ["架空県見本市試験町", "42", "19"] + rows[5:],
        rows[:3] + ["登録番号: T42-19", "6"] + rows[5:],
        rows[:3] + ["レジ" + address, "6"] + rows[5:],
        rows[:5] + ["電話 098-765-4321"] + rows[5:],
        rows[:1] + ["別町店"] + rows[1:],
        rows[:1] + [merchant] + rows[1:], rows[1:],
        [merchant, branch, registration, boundary, address, "6", phone, "領収証"],
    ]
    rejected_rows.extend(rows[:3] + [scope] + rows[3:]
                         for scope in ("HQ", "本社", "請求先", "お問い合わせ先", "連絡先", "FAX"))
    for rejected in rejected_rows:
        repaired = deepcopy(original)
        _run_final_header_location_repair_phase(
            repaired, "\n".join(rejected), ("header_branch_store_location",),
        )
        assert repaired.get("location") != address
    evidence = {"merged_text": "\n".join(rows), "tiles": [
        {"text": "\n".join(rows)}, {"text": "\n".join(rows)},
    ]}
    proposal, detail = _propose_location({"merchant": merchant, "location": branch}, evidence)
    assert proposal is None
    assert detail["criteria"]["candidate_has_location_suffix"] is False
