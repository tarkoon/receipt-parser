"""Focused contract for generic merchant-name equivalence."""

import pytest

from receipt_parser.checks import check_merchant_similarity


@pytest.mark.parametrize(("printed", "expected"), [
    ("ＫＡＬＤＩ・ＣＯＦＦＥＥ ＦＡＲＭ", "カルディ"),
    ("COSTCO WHOLESALE FOOD COURT", "COSTCO"),
    ("www.uniqlo.com", "ユニクロ"),
    ("ＡＣＭＥ， ＳＴＯＲＥ", "acme store"),
])
def test_merchant_checker_accepts_normalized_leading_brand_identity(
        printed, expected):
    assert check_merchant_similarity(
        {"merchant": printed}, {"merchant": expected}
    )["pass"]


def test_merchant_checker_rejects_only_shared_generic_leading_word():
    assert not check_merchant_similarity(
        {"merchant": "COFFEE FARM"}, {"merchant": "COFFEE LAB"}
    )["pass"]


def test_merchant_checker_keeps_null_truth_strict():
    assert check_merchant_similarity(
        {"merchant": None}, {"merchant": None}
    )["pass"]
    assert not check_merchant_similarity(
        {"merchant": "ACME"}, {"merchant": None}
    )["pass"]
