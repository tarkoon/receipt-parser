import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import weakref
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from receipt_parser import ocr, pipeline, usage
from receipt_parser.normalize import normalize_fullwidth, strip_barcode_lines
import receipt_parser.receipt_supplemental_ocr as supplemental
from receipt_parser.receipt_supplemental_ocr import (
    apply_supplemental_ocr_evidence,
    acquire_supplemental_ocr_evidence,
    build_supplemental_ocr_evidence,
    load_supplemental_ocr_evidence,
    save_supplemental_ocr_evidence,
    supplemental_ocr_cache_path,
    SupplementalOCRCacheMiss,
)


IMAGE_KEY = "a" * 32
STRATEGY_FINGERPRINT = "b" * 64
IMAGE_SHAPE = [1200, 800]


def _layout_for(text):
    return [
        {
            "text": line,
            "confidence": 0.99,
            "x": 0,
            "y": index * 50,
            "bbox": [
                [0, index * 50],
                [500, index * 50],
                [500, index * 50 + 40],
                [0, index * 50 + 40],
            ],
            "page": 0,
        }
        for index, line in enumerate(text.splitlines())
    ]


def _word(text, *, x=0, y=0, size=9, confidence=0.9):
    return {
        "text": text,
        "confidence": confidence,
        "x": x,
        "y": y,
        "bbox": [[x, y], [x + size, y], [x + size, y + size], [x, y + size]],
        "page": 0,
    }


def test_inclusive_literal_price_bundle_requires_closed_exact_currency_owners():
    from receipt_parser import pipeline
    from receipt_parser.ocr import blocks_to_structured_text
    from receipt_parser.schema import Receipt

    receipt = Receipt(merchant="架空店", subtotal=535, total=535, line_items=[
        {"description": title, "qty": 1, "unit_price": value, "total": value, "tax_category": tax}
        for title, value, tax in [("青葉菓子", 310, "8%"), ("緑茶60", 22, "8%"), ("紙袋20号", 5, "10%")]
    ]).model_dump()
    primary = "架空店\n青葉菓子\n¥310\n緑茶60\n¥22\n紙袋20号\n合計\n¥5\n¥535"
    layout = [_word("青葉菓子", x=10, y=10), _word("¥310", x=300, y=10),
              _word("緑茶60", x=10, y=40), _word("¥220", x=300, y=40),
              _word("紙袋20号", x=10, y=70), _word("¥5", x=300, y=70),
              _word("合計", x=10, y=100), _word("¥535", x=300, y=100),
              _word("(8%対象 ¥530)", x=10, y=130), _word("(10%対象 ¥5)", x=10, y=160)]

    def sealed(blocks):
        text = blocks_to_structured_text(blocks)
        return build_supplemental_ocr_evidence(
            image_key=IMAGE_KEY, image_shape=IMAGE_SHAPE,
            strategy_fingerprint=STRATEGY_FINGERPRINT,
            merged_text=text, source_layout=blocks, tile_texts=[text, text],
        )

    evidence = sealed(layout)
    before = deepcopy(receipt)
    proposed = supplemental._recover_literal_price_bundle(receipt, evidence, primary)
    assert proposed == [receipt["line_items"][0], dict(receipt["line_items"][1], unit_price=220, total=220), receipt["line_items"][2]]
    assert receipt == before
    assert supplemental._recover_literal_price_bundle(dict(receipt, line_items=proposed), evidence, primary) is None
    split = layout[:2] + [_word("緑茶", x=10, y=40), _word("60", x=100, y=40),
                         _word("¥", x=291, y=40), _word("220", x=300, y=40)] + layout[4:]
    assert supplemental._recover_literal_price_bundle(receipt, sealed(split), primary) == proposed
    distant_symbol = deepcopy(split)
    distant_symbol[4] = _word("¥", x=270, y=40)
    assert supplemental._recover_literal_price_bundle(receipt, sealed(distant_symbol), primary) is None
    for altered in (primary + "\n緑茶60", primary.replace("緑茶60", "緑茶80"),
                    primary.replace("¥22", "¥22\n2個 X 単110"),
                    primary.replace("¥22", "¥22\n値引 -¥10")):
        assert supplemental._recover_literal_price_bundle(receipt, evidence, altered) is None
    for blocks in (
        layout[:3] + [_word("220", x=300, y=40)] + layout[4:],
        layout + [_word("合計", x=10, y=190), _word("¥535", x=300, y=190)],
        layout[:6] + [_word("別の商品", x=10, y=85), _word("¥10", x=300, y=85)] + layout[6:],
        layout[:3] + [_word("¥10", x=200, y=40)] + layout[3:],
        layout[:-1] + [_word("(10%対象 ¥6)", x=10, y=160)],
        layout + [_word("(10%対象 ¥6)", x=10, y=190)],
    ):
        assert supplemental._recover_literal_price_bundle(receipt, sealed(blocks), primary) is None
    trace = []
    pipeline._apply_final_supplemental_ocr_evidence(receipt, evidence, mutation_trace=trace, primary_text=primary)
    assert {key: value for key, value in receipt.items() if not key.startswith("_")} == dict(before, line_items=proposed)
    assert len(trace) == 1 and set(trace[0]["changes"]) == {"line_items"}


def test_literal_price_bundle_requires_printed_subtotal_and_unique_detail_free_owners():
    from receipt_parser import pipeline
    from receipt_parser.ocr import blocks_to_structured_text
    from receipt_parser.receipt_financial import extract_financial_totals
    from receipt_parser.schema import Receipt

    receipt = Receipt(merchant="SHOP", subtotal=158, total=158, line_items=[
        {"description": "商品240型", "qty": 1, "unit_price": 90, "total": 90, "tax_category": "10%"},
        {"description": "別商品", "qty": 2, "unit_price": 30, "total": 60, "tax_category": "10%"},
    ]).model_dump()
    primary = "商品240型\n¥90\n別商品\n¥60\n小計\n¥158"
    layout = [_word("商品240型", x=10, y=10), _word("¥98", x=300, y=10),
              _word("別商品", x=10, y=40), _word("¥60", x=300, y=40),
              _word("小計", x=10, y=70), _word("¥158", x=300, y=70)]

    def sealed(blocks):
        text = blocks_to_structured_text(blocks)
        return build_supplemental_ocr_evidence(
            image_key=IMAGE_KEY, image_shape=IMAGE_SHAPE,
            strategy_fingerprint=STRATEGY_FINGERPRINT,
            merged_text=text, source_layout=blocks, tile_texts=[text, text],
        )

    evidence = sealed(layout)
    proposed = supplemental._recover_literal_price_bundle(receipt, evidence, primary)
    assert proposed is not None
    assert proposed[0] == dict(receipt["line_items"][0], unit_price=98, total=98)
    assert proposed[1] == receipt["line_items"][1]
    assert sum(item["total"] for item in proposed) == 158
    assert supplemental._literal_price_subtotal("小\n計\n¥158") == 158
    assert supplemental._recover_literal_price_bundle(dict(receipt, line_items=proposed), evidence, primary) is None

    for detail in ("(2個X単49)", "割引", "30%"):
        assert supplemental._recover_literal_price_bundle(receipt, evidence, primary.replace("¥90\n", f"¥90\n{detail}\n")) is None
        detailed = layout[:2] + [_word(detail, x=10, y=25)] + layout[2:]
        assert supplemental._recover_literal_price_bundle(receipt, sealed(detailed), primary) is None
    wrong_digits = deepcopy(layout)
    wrong_digits[0]["text"] = "商品241型"
    assert supplemental._recover_literal_price_bundle(receipt, sealed(wrong_digits), primary) is None
    duplicate = layout[:2] + [_word("商品240型", x=10, y=25), _word("¥99", x=300, y=25)] + layout[2:]
    assert supplemental._recover_literal_price_bundle(receipt, sealed(duplicate), primary) is None
    assert supplemental._recover_literal_price_bundle(receipt, evidence, primary.replace("¥158", "¥159")) is None
    no_subtotal = "商品240型\n¥90\n8%外税\n¥2\n合計\n¥100"
    assert extract_financial_totals(no_subtotal)["subtotal"] == 98
    assert supplemental._recover_literal_price_bundle(dict(receipt, line_items=receipt["line_items"][:1]), evidence, no_subtotal) is None
    assert supplemental._literal_price_subtotal("小計 ¥158\n小計 ¥158") is None

    output, metadata = apply_supplemental_ocr_evidence(receipt, evidence, primary_text=primary)
    assert metadata["accepted_fields"] == ["line_items"]
    assert metadata["line_items"]["mode"] == "literal_price_bundle"
    assert output == dict(receipt, line_items=proposed)
    trace = []
    pipeline._apply_final_supplemental_ocr_evidence(receipt, evidence, mutation_trace=trace, primary_text=primary)
    assert set(trace[0]["changes"]) == {"line_items"}
    assert receipt["line_items"] == proposed
    for field, value in output.items():
        assert receipt[field] == value


def _mock_vision(monkeypatch, texts=("TOP", "BOTTOM")):
    calls = []
    responses = [
        {"text": text, "words": [_word(text)]}
        for text in texts
    ]

    def call(image, client):
        calls.append((image.copy(), client))
        return responses[len(calls) - 1]

    initialized = []
    monkeypatch.setattr(supplemental, "_call_cloud_vision", call)
    monkeypatch.setattr(
        supplemental,
        "_extract_fulltext_from_response",
        lambda response: response["text"],
    )
    monkeypatch.setattr(
        supplemental,
        "_extract_words_from_response",
        lambda response: response["words"],
    )
    monkeypatch.setattr(
        supplemental,
        "init_cloud_vision",
        lambda: initialized.append(True) or "mock-client",
    )
    return calls, initialized


def _evidence(merged_text, tile_one, tile_two="footer"):
    return build_supplemental_ocr_evidence(
        image_key=IMAGE_KEY,
        image_shape=IMAGE_SHAPE,
        strategy_fingerprint=STRATEGY_FINGERPRINT,
        merged_text=merged_text,
        source_layout=_layout_for(merged_text),
        tile_texts=[tile_one, tile_two],
    )


def test_quantity_owned_title_requires_local_primary_arithmetic_and_source_summary():
    from receipt_parser.receipt_items import _fix_line_items
    from receipt_parser.schema import Receipt

    primary = "架空店\n施\n3個 x 単240\n¥720円\n小計\n¥720\n合計\n¥720"
    source = "架空店\n施術\n31 X #1240 ¥720円\n小計 ¥720\n合計 ¥720"
    receipt = Receipt(merchant="架空店", subtotal=655, total=720,
                      taxes=[{"rate": "10%", "label": "内税", "amount": 65}]).model_dump()
    _fix_line_items(receipt, primary)
    assert receipt["line_items"] == [{
        "description": "施", "qty": 3, "unit_price": 240, "total": 720,
        "tax_category": "10%", "discount": 0, "discount_rate": "",
    }]
    header_item = deepcopy(receipt)
    header_item["line_items"] = [{"description": "架空店", "qty": 1,
                                 "unit_price": 240, "total": 240}]
    _fix_line_items(header_item, primary)
    assert header_item == receipt
    expected = deepcopy(receipt)
    expected["line_items"][0]["description"] = "施術"
    evidence = _evidence(source, source)
    output, metadata = apply_supplemental_ocr_evidence(receipt, evidence, primary_text=primary)
    assert output == expected
    assert metadata["accepted_fields"] == ["line_items"]
    assert metadata["line_items"]["mode"] == "quantity_owned_title"
    assert apply_supplemental_ocr_evidence(output, evidence, primary_text=primary)[0] == output
    trace = []
    pipeline._apply_final_supplemental_ocr_evidence(receipt, evidence, mutation_trace=trace, primary_text=primary)
    assert all(receipt[field] == value for field, value in expected.items())
    assert set(trace[0]["changes"]) == {"line_items"}

    helper = supplemental._recover_quantity_owned_title
    for text in (source.replace("施術", "別の料金"), source.replace("¥720円", "¥721円"),
                 source.replace("31 X #1240", "4個 X 単240"),
                 source.replace("小計 ¥720", "案内\n小計 ¥720"),
                 source.replace("施術", "別の料金 ¥40\n施術"),
                 source.replace("合計 ¥720", "合計 ¥721"), source + "\n小計 ¥720"):
        assert helper(receipt, _evidence(text, text), primary) is None
    for text in (primary.replace("3個", "4個"), primary.replace("¥720円", "¥721円"),
                 primary.replace("施\n", "案内\n"), primary.replace("3個", "3人")):
        assert helper(receipt, evidence, text) is None
    for field, value in (("qty", True), ("unit_price", 239), ("total", float("nan")), ("discount", 1)):
        candidate = deepcopy(receipt)
        candidate["line_items"][0][field] = value
        assert helper(candidate, evidence, primary) is None


def test_counted_modifier_title_requires_unique_price_and_every_primary_backed_option():
    from receipt_parser import pipeline
    from receipt_parser.schema import Receipt

    text = "SHOP\n冷たいラテ ¥320\n濃いミルク (追加)\n少ない氷 (追加)\n本体合計(1点)320"
    evidence = _evidence(text, text)
    receipt = Receipt(merchant="SHOP", subtotal=320, total=320, line_items=[
        {"description": "冷たいラテ 濃いミルク 少ない氷", "qty": 1,
         "unit_price": 320, "total": 320, "tax_category": "10%"},
    ]).model_dump()
    title = "冷たいラテ 濃いミルク 少ない氷 (追加)"
    helper = supplemental._recover_counted_modifier_title
    assert helper(receipt, evidence, text) == title
    expected = deepcopy(receipt)
    expected["line_items"][0]["description"] = title
    output, metadata = apply_supplemental_ocr_evidence(receipt, evidence, primary_text=text)
    assert output == expected
    assert metadata["accepted_fields"] == ["line_items"]
    assert metadata["line_items"]["mode"] == "counted_modifier_title"
    assert receipt["line_items"][0]["description"] == "冷たいラテ 濃いミルク 少ない氷"
    assert apply_supplemental_ocr_evidence(output, evidence, primary_text=text)[0] == output

    for field, value in (("qty", 2), ("qty", True), ("discount", 10), ("discount_rate", "5%"),
                         ("unit_price", 321), ("total", float("nan")),
                         ("description", "冷たいラテ 濃いミルク"),
                         ("description", "冷たいラテ 濃いミルク 少ない氷 2")):
        candidate = deepcopy(receipt)
        candidate["line_items"][0][field] = value
        assert helper(candidate, evidence, text) is None
    for source in (text.replace("(1点)", "(2点)"), text.replace("(1点)320", "(1点)3,20"),
                   text.replace("¥320", "¥321"), text.replace("¥320", "¥3,20"), text.replace("¥320", "¥320 外"),
                   text.replace("少ない氷", "少ない氷30"),
                   text.replace("本体合計", "無関係な案内\n本体合計"),
                   text.replace("冷たいラテ", "別の商品 ¥50\n冷たいラテ"),
                   text + "\n本体合計(1点)320"):
        assert helper(receipt, _evidence(source, source), text) is None
    assert helper(receipt, evidence, text.replace("少ない氷", "")) is None
    assert helper(receipt, evidence, text.replace("(1点)", "(2点)")) is None
    assert helper(receipt, evidence, None) is None

    trace = []
    pipeline._apply_final_supplemental_ocr_evidence(receipt, evidence, mutation_trace=trace, primary_text=text)
    assert set(trace[0]["changes"]) == {"line_items"}
    assert all(receipt[field] == value for field, value in expected.items())


def _without_tax_categories(value):
    masked = deepcopy(value)
    for item in masked.get("line_items") or []:
        item["tax_category"] = "<TARGET>"
    return json.dumps(
        masked,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def test_header_mark_requires_complete_unique_geometry_owner_and_primary_header_corroboration():
    from receipt_parser.ocr import blocks_to_structured_text
    from receipt_parser.pipeline import _apply_final_supplemental_ocr_evidence
    from receipt_parser.receipt_identity_payment import _is_ascii_logo, _select_header_logo

    def block(text, y, width=150, height=28, confidence=0.99):
        value = _word(text, x=20, y=y, size=height, confidence=confidence)
        value["bbox"][1][0] = value["bbox"][2][0] = 20 + width
        return value

    layout = [block("ALPHACOFFEE®", 20), block("中央店", 100, height=14),
              block("TEL 01-2345", 135, height=14)]
    source = blocks_to_structured_text(layout)
    primary = "ALPHACOFFEE ®\n中央店\nTEL 01-2345\n商品240型 ¥100\n合計 ¥100"
    assert _select_header_logo(source, layout, primary) == "ALPHACOFFEE"
    assert _select_header_logo(source, layout, primary.replace("ALPHACOFFEE", "BETASTORE")) is None
    assert _select_header_logo(source, layout, None) is None
    low_confidence = deepcopy(layout)
    low_confidence[0]["confidence"] = 0.7
    assert _select_header_logo(source, low_confidence, primary) is None
    partial = [block("DELI", 20, width=70), block("KITCHEN", 56), *layout[1:]]
    partial_text = blocks_to_structured_text(partial)
    assert _select_header_logo(partial_text, partial, partial_text) is None
    stacked = [block("NORTHWAY", 20), block("WESTWIDE", 56), *layout[1:]]
    stacked_text = blocks_to_structured_text(stacked)
    assert _select_header_logo(stacked_text, stacked, stacked_text) == "NORTHWAY WESTWIDE"
    titled = [block("WillowWorks", 20, width=200, height=40),
              block("サービス案内", 76, width=80, height=16),
              block("お預り票", 125, height=24),
              block("ウィローワークス", 220), block("TEL 01-2345", 270)]
    titled_text = blocks_to_structured_text(titled)
    titled_primary = "Willow Works\nサービス案内\nお預り票\nウィローワークス\nTEL 01-2345"
    assert _select_header_logo(titled_text, titled, titled_primary) == "Willow Works"
    touching = [block("WillowWorks", 20, width=200, height=60),
                block("マート", 79, width=90, height=34), block("領収書", 160)]
    for point in touching[1]["bbox"]:
        point[0] += 110
    touching_text = blocks_to_structured_text(touching)
    assert _select_header_logo(touching_text, touching, touching_text) is None
    touching[1]["confidence"] = 0.7
    assert _select_header_logo(touching_text, touching, touching_text) is None
    short_owner = [block("XY", 20, height=80), block("DININGROOM", 130, height=24),
                   block("領収書", 220)]
    short_text = blocks_to_structured_text(short_owner)
    assert _select_header_logo(short_text, short_owner, short_text) is None
    short_owner[0]["text"] = "APP"
    short_text = blocks_to_structured_text(short_owner)
    assert _select_header_logo(short_text, short_owner, short_text) == "DININGROOM"
    assert not _is_ascii_logo("WillowWorks")
    assert _is_ascii_logo("WillowWorks", allow_mixed_case=True)
    assert _select_header_logo(source, layout, primary.replace("ALPHACOFFEE ®", "ALPHACOFFEE SHOP")) is None
    assert _select_header_logo(source, layout, "店頭ブランド\n" + primary,
                               current_merchant="店頭ブランド") is None
    assert _select_header_logo(titled_text, titled, titled_primary,
                               current_merchant="ウィローワークス") == "Willow Works"
    competing = [block("OtherBrand", 20, width=180), block("WillowWorks", 100, width=200),
                 block("お預り票", 180)]
    competing_text = blocks_to_structured_text(competing)
    assert _select_header_logo(competing_text, competing, "Other Brand\n" + titled_primary) is None
    receipt_title = [*stacked[:2], block("利用証明書", 125), block("速", 220)]
    title_text = blocks_to_structured_text(receipt_title)
    assert _select_header_logo(title_text, receipt_title, title_text) == "NORTHWAY WESTWIDE"
    footer_only = [block("お預り票", 10), block("WillowWorks", 60)]
    footer_text = blocks_to_structured_text(footer_only)
    assert _select_header_logo(footer_text, footer_only, footer_text) is None
    contact = [block("TEL 01-2345", 20), block("お預り票", 125)]
    contact_text = blocks_to_structured_text(contact)
    assert _select_header_logo(contact_text, contact, contact_text) is None
    split_title = [*stacked[:2], block("領", 125), block("収書", 152)]
    split_text = blocks_to_structured_text(split_title)
    assert _select_header_logo(split_text, split_title, split_text) == "NORTHWAY WESTWIDE"
    evidence = build_supplemental_ocr_evidence(image_key=IMAGE_KEY, image_shape=IMAGE_SHAPE,
        strategy_fingerprint=STRATEGY_FINGERPRINT, merged_text=source, source_layout=layout,
        tile_texts=[source, "footer"])
    original = {"document_type": "receipt", "merchant": "小売本社", "location": "中央店",
                "subtotal": 100, "total": 100, "taxes": [], "line_items": []}
    projected, meta = apply_supplemental_ocr_evidence(original, evidence, primary_text=primary)
    assert meta["accepted_fields"] == ["merchant"]
    assert projected == dict(original, merchant="ALPHACOFFEE")
    result, trace = deepcopy(original), []
    _apply_final_supplemental_ocr_evidence(result, evidence, trace, primary_text=primary)
    assert result["merchant"] == "ALPHACOFFEE"
    assert set(trace[-1]["changes"]) == {"merchant"}
    assert {key: value for key, value in result.items() if not key.startswith("_")} == projected


def test_merchant_branch_metadata_requires_exact_separate_header_roles_and_preserves_other_fields():
    from receipt_parser.ocr import blocks_to_structured_text
    from receipt_parser.pipeline import _apply_final_supplemental_ocr_evidence
    from receipt_parser.receipt_identity_payment import _select_header_merchant_without_branch

    brand, branch, phone = '検証マーケット', '南丘店', '050-2345-6789'
    primary = '\n'.join([brand, branch, phone, '登録番号:T1234567890123',
        '株式会社VERIFICATION', '領収証', '2031年04月05日12:20'])
    layout = [_word(brand, x=20, y=25, size=80),
        _word(branch, x=20, y=150, size=30), _word(phone, x=400, y=150, size=30),
        _word('登録番号:T1234567890123', y=210, size=20),
        _word('株式会社VERIFICATION', y=245, size=20),
        _word('領収証', y=285, size=20), _word('2031年04月05日12:20', y=320, size=20)]

    def propose(merchant=brand + ' ' + branch, location=branch, text=primary, blocks=layout):
        return _select_header_merchant_without_branch(blocks_to_structured_text(blocks), blocks,
            text, current_merchant=merchant, current_location=location)

    assert propose() == brand
    assert propose(merchant=brand) == brand
    assert propose(text=primary.replace(branch + '\n' + phone, branch + '  ' + phone)) == brand
    assert propose(text=primary.replace(phone, phone + ' 窓口')) is None
    assert propose(text=primary.replace(branch, branch + '別館')) is None
    for merchant in ('別会社', brand + ' ' + branch + '店', brand + ' 別支店'):
        assert propose(merchant=merchant) is None
    assert propose(location='別支店') is None
    assert propose(location=None) is None
    assert propose(text=primary.replace(branch, '北丘店')) is None
    assert propose(text=primary.replace(phone, '050-2345-6790')) is None
    assert propose(text=primary.replace(brand, brand + '2')) is None
    unlabelled = deepcopy(layout)
    unlabelled[1]['text'] = '南丘'
    assert propose(location='南丘', text=primary.replace(branch, '南丘'), blocks=unlabelled) is None
    duplicate = [*deepcopy(layout), _word('北丘店', x=20, y=180, size=30),
        _word('050-2345-6790', x=400, y=180, size=30)]
    assert propose(text=primary.replace('登録番号', '北丘店\n050-2345-6790\n登録番号'), blocks=duplicate) is None
    competing = deepcopy(layout)
    for word in competing:
        word['y'] += 100
        for point in word['bbox']:
            point[1] += 100
    competing.insert(0, _word('別市場', y=0, size=75))
    assert propose(text='別市場\n' + primary, blocks=competing) is None
    below_branch = deepcopy(layout)
    for word in below_branch[3:]:
        word['y'] += 150
        for point in word['bbox']:
            point[1] += 150
    below_branch.insert(3, _word('別市場', y=210, size=75))
    assert propose(text=primary.replace(phone, phone + '\n別市場'), blocks=below_branch) is None
    repeated_rival = deepcopy(below_branch)
    for word in repeated_rival[4:]:
        word['y'] += 150
        for point in word['bbox']:
            point[1] += 150
    repeated_rival.insert(4, _word('別市場', y=310, size=75))
    assert propose(text=primary.replace(phone, phone + '\n別市場\n別市場'), blocks=repeated_rival) is None
    touching = deepcopy(layout)
    touching[1] = _word(branch, x=20, y=90, size=30)
    touching[2] = _word(phone, x=400, y=90, size=30)
    assert propose(blocks=touching) is None
    weak = deepcopy(layout)
    weak[0]['confidence'] = .59
    assert propose(blocks=weak) is None

    original = {'document_type': 'receipt', 'merchant': brand + ' ' + branch, 'location': branch,
        'currency': 'JPY', 'subtotal': 100, 'total': 108,
        'taxes': [{'rate': '8%', 'label': '外税', 'amount': 8}],
        'line_items': [{'description': '試作食品', 'qty': 1, 'unit_price': 100, 'total': 100,
                        'tax_category': '8%', 'discount': 0, 'discount_rate': ''}]}
    source_text = blocks_to_structured_text(layout)
    bound = build_supplemental_ocr_evidence(image_key=IMAGE_KEY, image_shape=IMAGE_SHAPE,
        strategy_fingerprint=STRATEGY_FINGERPRINT, merged_text=source_text,
        source_layout=layout, tile_texts=[source_text, 'footer'])
    projected, metadata = apply_supplemental_ocr_evidence(original, bound, primary_text=primary)
    assert projected == dict(original, merchant=brand)
    assert original['merchant'] == brand + ' ' + branch
    assert metadata['merchant']['accepted']
    stable, stable_meta = apply_supplemental_ocr_evidence(projected, bound, primary_text=primary)
    assert stable == projected and not stable_meta['merchant']['accepted']
    no_primary, _ = apply_supplemental_ocr_evidence(original, bound)
    assert no_primary == original
    malformed = dict(bound, merged_text=source_text + 'changed')
    rejected, _ = apply_supplemental_ocr_evidence(original, malformed, primary_text=primary)
    assert rejected == original
    result, trace = deepcopy(original), []
    _apply_final_supplemental_ocr_evidence(result, bound, trace, primary_text=primary)
    assert result['merchant'] == brand and result['location'] == branch
    assert set(trace[-1]['changes']) == {'merchant'}


def test_partial_marker_uses_unique_exact_digit_title_and_literal_price_without_changing_other_fields():
    from receipt_parser.ocr import blocks_to_structured_text
    from receipt_parser.pipeline import _apply_final_supplemental_ocr_evidence

    legend = "*印は軽減税率対象(外8%)商品です"

    def evidence(rows, footer=legend):
        layout = []
        for index, (code, marker, title, price) in enumerate(rows):
            y = index * 50
            layout.extend([_word(code, x=0, y=y, size=30),
                           _word(title, x=80, y=y, size=30),
                           _word(f"¥{price}", x=420, y=y, size=30)])
            if marker:
                layout.append(_word(marker, x=40, y=y, size=30))
        layout.append(_word(footer, y=len(rows) * 50, size=30))
        layout.sort(key=lambda block: (block["y"], block["x"]))
        text = blocks_to_structured_text(layout)
        return build_supplemental_ocr_evidence(
            image_key=IMAGE_KEY, image_shape=IMAGE_SHAPE,
            strategy_fingerprint=STRATEGY_FINGERPRINT, merged_text=text,
            source_layout=layout, tile_texts=[text, "footer"],
        )

    rows = [("0012", "*", "商品240型", 127), ("0034", "", "別商品200型", 209)]
    original = {"document_type": "receipt", "merchant": "店", "subtotal": 336,
                "total": 336, "line_items": [
                    {"description": title, "qty": 1, "unit_price": price,
                     "total": price, "discount": 0, "discount_rate": "",
                     "tax_category": "10%"} for _code, _marker, title, price in rows]}
    bound = evidence(rows)
    output, meta = apply_supplemental_ocr_evidence(original, bound)
    assert output["line_items"][0]["tax_category"] == "8%"
    assert output["line_items"][1]["tax_category"] == "10%"
    assert _without_tax_categories(output) == _without_tax_categories(original)
    assert meta["tax_categories"]["mode"] == "partial_explicit_title_price_marker_ownership"
    assert original["line_items"][0]["tax_category"] == "10%"

    result, trace = deepcopy(original), []
    _apply_final_supplemental_ocr_evidence(result, bound, trace)
    assert result["line_items"][0]["tax_category"] == "8%"
    assert trace[-1]["stage"] == "supplemental_ocr_field_recovery"
    assert set(trace[-1]["changes"]) == {"line_items"}

    wrong_digit = deepcopy(original)
    wrong_digit["line_items"][0]["description"] = "商品241型"
    wrong_price = deepcopy(original)
    wrong_price["line_items"][0]["total"] = 128
    duplicate = deepcopy(original)
    duplicate["line_items"].append(deepcopy(duplicate["line_items"][0]))
    for receipt, source in [
        (wrong_digit, bound), (wrong_price, bound), (duplicate, bound),
        (original, evidence([rows[0], ("0056", "*", "商品240型", 127)])),
        (original, evidence([rows[0], ("0056", "", "商品240型", 127)])),
        (original, evidence(rows, legend + "\n*印は10%対象商品です")),
        (original, evidence(rows, "*印は軽減税率対象商品です")),
    ]:
        rejected, _meta = apply_supplemental_ocr_evidence(receipt, source)
        assert rejected == receipt


def test_uniform_two_tile_strategy_matches_the_sealed_v1_3_fingerprint():
    assert supplemental.SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT == (
        "85c90040f1f0438cecd0d5c4dcb74d7795cfa138cf976d080e9f27aee1a72bcb"
    )
    assert supplemental.SUPPLEMENTAL_OCR_STRATEGY["tile_y_fractions"] == (
        (0.0, 0.58),
        (0.42, 1.0),
    )


def test_missing_table_rows_require_literal_prices_unique_digit_titles_count_and_both_printed_bases():
    from receipt_parser.ocr import blocks_to_structured_text
    from receipt_parser.pipeline import _apply_final_supplemental_ocr_evidence

    rows = [("商品240型", 112, True), ("商品270型", 135, True),
            ("別商品300型", 54, False), ("レジ袋M", 3, False)]
    footer = "小計\n¥304\n外税8%対象額\n¥247\n外税10%対象額\n¥57\nお買上商品数:3\n※印は軽減税率8%対象商品"

    def source(source_rows=rows):
        layout = []
        for index, (title, price, marked) in enumerate(source_rows):
            layout.extend([_word(title, x=20, y=index * 50, size=30),
                           _word(f"¥{price}" + ("*" if marked else ""), x=420, y=index * 50, size=30)])
        layout.extend(_word(line, y=(len(source_rows) + index) * 50, size=30)
                      for index, line in enumerate(footer.splitlines()))
        text = blocks_to_structured_text(layout)
        return build_supplemental_ocr_evidence(image_key=IMAGE_KEY, image_shape=IMAGE_SHAPE,
            strategy_fingerprint=STRATEGY_FINGERPRINT, merged_text=text,
            source_layout=layout, tile_texts=[text, "footer"])

    primary = "\n".join(f"{title}\n¥{price}" + ("*" if marked else "") for title, price, marked in rows) + "\n" + footer
    original = {"document_type": "receipt", "merchant": "店", "subtotal": 304, "total": 328,
                "taxes": [{"rate": "8%", "label": "外税", "amount": 19},
                          {"rate": "10%", "label": "外税", "amount": 5}],
                "line_items": [{"description": "商品240型", "qty": 1, "unit_price": 112,
                                "total": 112, "discount": 0, "discount_rate": "", "tax_category": "8%"}]}
    bound = source()
    output, meta = apply_supplemental_ocr_evidence(original, bound, primary_text=primary)
    assert meta["accepted_fields"] == ["line_items"]
    assert len(output["line_items"]) == 4
    assert sum(item["total"] for item in output["line_items"]) == 304
    assert {rate: sum(item["total"] for item in output["line_items"] if item["tax_category"] == rate)
            for rate in ("8%", "10%")} == {"8%": 247, "10%": 57}
    assert {key: value for key, value in output.items() if key != "line_items"} == {
        key: value for key, value in original.items() if key != "line_items"}
    assert len(original["line_items"]) == 1
    assert apply_supplemental_ocr_evidence(output, bound, primary_text=primary)[0] == output

    result, trace = deepcopy(original), []
    _apply_final_supplemental_ocr_evidence(result, bound, trace, primary_text=primary)
    assert len(result["line_items"]) == 4
    assert set(trace[-1]["changes"]) == {"line_items"}
    assert trace[-1]["stage"] == "supplemental_ocr_field_recovery"

    wrong_digit = deepcopy(original)
    wrong_digit["line_items"][0]["description"] = "商品241型"
    wrong_qty = deepcopy(original)
    wrong_qty["line_items"][0]["qty"] = 2
    for receipt, evidence, text in [
        (original, bound, None), (wrong_digit, bound, primary), (wrong_qty, bound, primary),
        (original, bound, primary.replace("別商品300型", "別商品301型")),
        (original, bound, primary.replace("お買上商品数:3", "お買上商品数:4")),
        (original, bound, primary.replace("¥247", "¥246")),
        (original, bound, primary.replace("商品270型", "商品270型\n(2個X単135)")),
        (original, bound, primary.replace("商品270型", "商品270型\n値引 ¥10")),
        (original, source([rows[0], rows[0], *rows[2:]]), primary),
    ]:
        rejected, rejected_meta = apply_supplemental_ocr_evidence(receipt, evidence, primary_text=text)
        assert "line_items" not in rejected_meta["accepted_fields"]
        assert rejected == receipt
def test_row_x_order_preserves_vertical_groups_and_migrates_only_bound_validated_legacy_cache(
    monkeypatch, tmp_path,
):
    from receipt_parser.ocr import blocks_to_structured_text

    layout = [
        _word("単80", x=140, y=100, size=30),
        _word("X", x=100, y=102, size=30),
        _word("2個", x=20, y=104, size=30),
        _word("次行", x=20, y=145, size=30),
    ]
    original = deepcopy(layout)
    legacy_text = "単80  X  2個\n次行"
    ordered_text = "2個  X  単80\n次行"
    assert blocks_to_structured_text(layout, sort_within_rows=False) == legacy_text
    assert blocks_to_structured_text(layout) == ordered_text
    assert layout == original
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    legacy = build_supplemental_ocr_evidence(
        image_key=supplemental._ocr_cache_key(image),
        image_shape=list(image.shape[:2]),
        strategy_fingerprint=supplemental._LEGACY_STRATEGY_FINGERPRINT,
        merged_text=legacy_text,
        source_layout=layout,
        tile_texts=[legacy_text, "footer"],
    )
    path = save_supplemental_ocr_evidence(tmp_path, legacy)
    before = path.read_bytes()
    monkeypatch.setattr(supplemental, "init_cloud_vision", lambda: pytest.fail("cache-only contacted Vision"))
    migrated = acquire_supplemental_ocr_evidence(image, mode="cache_only", cache_dir=tmp_path)
    assert migrated["merged_text"] == ordered_text
    assert migrated["strategy_fingerprint"] == supplemental.SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT
    assert migrated["source_layout"] == legacy["source_layout"]
    assert migrated["tiles"] == legacy["tiles"]
    assert path.read_bytes() == before
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert supplemental._validated_evidence(migrated) == migrated
    current_path = supplemental_ocr_cache_path(
        tmp_path, image_key=legacy["image_key"], image_shape=legacy["image_shape"],
        strategy_fingerprint=supplemental.SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT,
    )
    current_path.write_bytes(before)
    with pytest.raises(SupplementalOCRCacheMiss):
        acquire_supplemental_ocr_evidence(image, mode="cache_only", cache_dir=tmp_path)
    current_path.unlink()
    for invalid in (
        {**legacy, "merged_text": "different words"},
        {**legacy, "image_key": "f" * 32},
        {**legacy, "image_shape": [201, 300]},
        {**legacy, "strategy_fingerprint": "c" * 64},
    ):
        path.write_text(json.dumps(invalid), encoding="utf-8")
        with pytest.raises(SupplementalOCRCacheMiss):
            acquire_supplemental_ocr_evidence(image, mode="cache_only", cache_dir=tmp_path)


def test_uniform_receipt_bbox_and_overlapping_tiles_match_the_sealed_geometry():
    image = np.zeros((200, 100, 3), dtype=np.uint8)
    image[20:180, 10:90] = 255

    bbox = supplemental._supplemental_receipt_bbox(image)

    assert bbox == (9, 18, 91, 182)
    assert supplemental._supplemental_tile_bboxes(bbox) == [
        (9, 18, 91, 113),
        (9, 87, 91, 182),
    ]


def test_uniform_scale_is_shared_and_selected_payload_guards_fail_closed(monkeypatch):
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    bboxes = [(0, 0, 10, 6), (0, 4, 10, 10)]
    scales = []
    three_x_refs = []
    fallback_index = 0

    def preprocess(_image, _bbox, *, scale):
        nonlocal fallback_index
        scales.append(scale)
        processed = np.zeros((scale, 2), dtype=np.uint8)
        if scale == 3:
            three_x_refs.append(weakref.ref(processed))
        else:
            assert three_x_refs[fallback_index]() is None
            fallback_index += 1
        return processed

    monkeypatch.setattr(supplemental, "_preprocess_supplemental_bbox", preprocess)
    monkeypatch.setattr(
        supplemental,
        "_supplemental_png_size",
        lambda processed: (
            supplemental.SUPPLEMENTAL_OCR_PAYLOAD_CAP_BYTES
            if processed.shape[0] == 3
            else 1
        ),
    )

    scale, prepared = supplemental._prepare_supplemental_tiles(image, bboxes)

    assert scale == 2
    assert scales == [3, 3, 2, 2]
    assert [tile.shape for tile in prepared] == [(2, 2), (2, 2)]

    monkeypatch.setattr(
        supplemental,
        "_supplemental_png_size",
        lambda _processed: supplemental.SUPPLEMENTAL_OCR_PAYLOAD_CAP_BYTES,
    )
    with pytest.raises(RuntimeError, match="payload limit"):
        supplemental._prepare_supplemental_tiles(image, bboxes)

    monkeypatch.setattr(supplemental, "_supplemental_png_size", lambda _processed: 1)
    monkeypatch.setattr(supplemental, "SUPPLEMENTAL_OCR_API_PIXEL_LIMIT", 3)
    with pytest.raises(RuntimeError, match="API pixel limit"):
        supplemental._prepare_supplemental_tiles(image, bboxes)


def test_uniform_word_mapping_and_iterative_iou_dedupe_use_sealed_preference_rule():
    word = _word("x", x=0, y=0, size=30, confidence=0.9)
    mapped = supplemental._map_supplemental_word(
        word,
        (10, 20, 100, 100),
        scale=3,
    )
    assert mapped["bbox"] == [[10, 20], [20, 20], [20, 30], [10, 30]]

    richer = {**mapped, "text": "x*", "confidence": 0.7}
    assert supplemental._merge_supplemental_words([mapped, richer]) == [richer]
    too_uncertain = {**mapped, "text": "x*", "confidence": 0.6}
    assert supplemental._merge_supplemental_words([mapped, too_uncertain]) == [mapped]

    chain = [
        {**mapped, "confidence": 0.8, "x": 0, "y": 0, "bbox": [[0, 0], [10, 0], [10, 100], [0, 100]]},
        {**mapped, "confidence": 0.8, "x": 6, "y": 1, "bbox": [[6, 1], [16, 1], [16, 101], [6, 101]]},
        {**mapped, "confidence": 0.9, "x": 3, "y": 2, "bbox": [[3, 2], [13, 2], [13, 102], [3, 102]]},
    ]
    merged = supplemental._merge_supplemental_words(chain)
    assert all(
        supplemental._supplemental_word_iou(left, right) < 0.5
        for index, left in enumerate(merged)
        for right in merged[index + 1:]
    )


def test_normal_mode_misses_then_hits_cache_with_exactly_two_vision_calls(
    tmp_path,
    monkeypatch,
):
    image = np.full((100, 60, 3), 255, dtype=np.uint8)
    calls, initialized = _mock_vision(monkeypatch)

    acquired = acquire_supplemental_ocr_evidence(image, cache_dir=tmp_path)
    cached = acquire_supplemental_ocr_evidence(image, cache_dir=tmp_path)

    assert len(calls) == 2
    assert initialized == [True]
    assert cached == acquired
    assert acquired["merged_text"] == "TOP\nBOTTOM"
    assert len(acquired["tiles"]) == 2


def test_concurrent_normal_misses_share_one_two_call_acquisition_and_cached_result(
    tmp_path,
    monkeypatch,
):
    image = np.full((100, 60, 3), 255, dtype=np.uint8)
    calls, initialized = _mock_vision(monkeypatch)
    real_load = supplemental.load_supplemental_ocr_evidence
    initial_loads = threading.Barrier(2)
    thread_state = threading.local()

    def synchronized_initial_load(*args, **kwargs):
        result = real_load(*args, **kwargs)
        if not getattr(thread_state, "did_initial_load", False):
            thread_state.did_initial_load = True
            initial_loads.wait(timeout=5)
        return result

    monkeypatch.setattr(
        supplemental,
        "load_supplemental_ocr_evidence",
        synchronized_initial_load,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                acquire_supplemental_ocr_evidence,
                image,
                cache_dir=tmp_path,
            )
            for _ in range(2)
        ]
        results = [future.result(timeout=10) for future in futures]

    cached = real_load(
        tmp_path,
        image_key=supplemental._ocr_cache_key(image),
        image_shape=list(image.shape[:2]),
        strategy_fingerprint=supplemental.SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT,
    )
    assert len(calls) == 2
    assert initialized == [True]
    assert results[0] == results[1] == cached
    assert supplemental._NORMAL_CACHE_LOCKS == {}


def test_normal_mode_cleans_per_identity_lock_after_acquisition_error(
    tmp_path,
    monkeypatch,
):
    image = np.full((100, 60, 3), 255, dtype=np.uint8)

    def fail_vision(*_args, **_kwargs):
        raise RuntimeError("Vision failed")

    monkeypatch.setattr(supplemental, "_call_cloud_vision", fail_vision)

    with pytest.raises(RuntimeError, match="Vision failed"):
        acquire_supplemental_ocr_evidence(
            image,
            client=object(),
            cache_dir=tmp_path,
        )

    assert supplemental._NORMAL_CACHE_LOCKS == {}


def test_cache_only_missing_and_corrupt_entries_make_zero_vision_calls(
    tmp_path,
    monkeypatch,
):
    image = np.full((100, 60, 3), 255, dtype=np.uint8)
    calls = []
    monkeypatch.setattr(
        supplemental,
        "_call_cloud_vision",
        lambda *_args, **_kwargs: calls.append(True),
    )
    monkeypatch.setattr(
        supplemental,
        "init_cloud_vision",
        lambda: pytest.fail("cache-only mode initialized Vision"),
    )

    with pytest.raises(SupplementalOCRCacheMiss, match="no valid supplemental OCR cache"):
        acquire_supplemental_ocr_evidence(
            image,
            mode="cache_only",
            cache_dir=tmp_path,
        )

    path = supplemental_ocr_cache_path(
        tmp_path,
        image_key=supplemental._ocr_cache_key(image),
        image_shape=list(image.shape[:2]),
        strategy_fingerprint=supplemental.SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{corrupt", encoding="utf-8")
    with pytest.raises(SupplementalOCRCacheMiss, match="no valid supplemental OCR cache"):
        acquire_supplemental_ocr_evidence(
            image,
            mode="cache_only",
            cache_dir=tmp_path,
        )
    assert calls == []


def test_fresh_mode_ignores_valid_cache_and_leaves_it_byte_unchanged(
    tmp_path,
    monkeypatch,
):
    image = np.full((100, 60, 3), 255, dtype=np.uint8)
    cached_layout = [_word("CACHED")]
    cached_evidence = build_supplemental_ocr_evidence(
        image_key=supplemental._ocr_cache_key(image),
        image_shape=list(image.shape[:2]),
        strategy_fingerprint=supplemental.SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT,
        merged_text="CACHED",
        source_layout=cached_layout,
        tile_texts=["cached top", "cached bottom"],
    )
    path = save_supplemental_ocr_evidence(tmp_path, cached_evidence)
    cached_bytes = path.read_bytes()
    calls, initialized = _mock_vision(monkeypatch, texts=("FRESH TOP", "FRESH BOTTOM"))

    fresh = acquire_supplemental_ocr_evidence(
        image,
        mode="fresh",
        cache_dir=tmp_path,
    )

    assert len(calls) == 2
    assert initialized == [True]
    assert fresh["merged_text"] == "FRESH TOP\nFRESH BOTTOM"
    assert path.read_bytes() == cached_bytes


def test_unique_ocr_backed_location_extension_changes_only_location():
    receipt = {
        "merchant": "ブランド",
        "location": "元店",
        "subtotal": 100,
        "line_items": [{"description": "商品", "total": 100, "tax_category": "10%"}],
        "taxes": [{"rate": "10%", "amount": 9}],
    }
    original = deepcopy(receipt)
    header = "ブランド\n稲元店 TEL0940-00-0000"

    output, metadata = apply_supplemental_ocr_evidence(
        receipt,
        _evidence(header, header),
    )

    assert receipt == original
    assert output == {**original, "location": "稲元店"}
    assert metadata["accepted_fields"] == ["location"]
    assert metadata["location"]["criteria"] == {
        "one_candidate_across_merged_and_tiles": True,
        "candidate_present": True,
        "candidate_differs": True,
        "candidate_has_location_suffix": True,
        "production_evidence_check": True,
        "unique_supporting_header_line": True,
        "candidate_strictly_extends_current_or_unsupported_composite_component": True,
    }

    composite = "ブランド × 東町中央店"
    receipt["location"] = "中央店"
    original = deepcopy(receipt)
    header = f"領収証\n{composite}\nTEL0940-00-0000"
    output, metadata = apply_supplemental_ocr_evidence(receipt, _evidence(header, header))
    assert receipt == original
    assert output == {**original, "location": "東町中央店"}
    assert metadata["accepted_fields"] == ["location"]

    for merged, tile in (
        (header, "商品明細"),
        (header, header.replace("東町", "西町")),
        (header.replace(" × ", " × 別 × "), header.replace(" × ", " × 別 × ")),
        ("小計\n" + header, "小計\n" + header),
        (header.replace("ブランド", "別所有者"), header.replace("ブランド", "別所有者")),
    ):
        output, metadata = apply_supplemental_ocr_evidence(receipt, _evidence(merged, tile))
        assert output == original
        assert metadata["accepted_fields"] == []


def test_location_suffix_chain_requires_unique_printed_longest_branch_and_same_suffix_stem():
    receipt = {"merchant": "ブランド", "location": "ブランド 東町店", "line_items": [], "taxes": []}
    original = deepcopy(receipt)
    merged = "ブランド 東町 中央 店\nTEL000-00-0000"
    tile = "ブランド 東町中央店\nTEL000-00-0000"
    output, metadata = apply_supplemental_ocr_evidence(receipt, _evidence(merged, tile))
    assert receipt == original
    assert output == {**original, "location": "東町中央店"}
    assert metadata["accepted_fields"] == ["location"]
    for unsupported_merged, unsupported_tile in (
        (merged, tile.replace("東町", "西町")),
        (merged, "商品明細"),
        (merged.replace("中央", "北町"), tile),
        ("小計\n" + merged, "小計\n" + tile),
        (merged.replace("店", "営業所"), tile.replace("店", "営業所")),
    ):
        output, metadata = apply_supplemental_ocr_evidence(
            receipt, _evidence(unsupported_merged, unsupported_tile),
        )
        assert output == original
        assert metadata["accepted_fields"] == []


def test_layout_marker_tax_vector_applies_only_when_printed_bases_reconcile():
    receipt = {
        "merchant": "ストア",
        "location": None,
        "subtotal": 300,
        "line_items": [
            {"description": "食品A", "total": 100, "tax_category": "10%"},
            {"description": "用品B", "total": 200, "tax_category": "8%"},
        ],
        "taxes": [
            {"rate": "8%", "amount": 8},
            {"rate": "10%", "amount": 20},
        ],
    }
    original = deepcopy(receipt)
    evidence = _evidence(
        "明細\n* 食品A ¥100\n用品B ¥200\n小計\n¥300",
        "*は軽減税率8%適用商品\n8% 対象額 ¥100\n10% 対象額 ¥200",
    )

    output, metadata = apply_supplemental_ocr_evidence(receipt, evidence)

    assert receipt == original
    assert [item["tax_category"] for item in output["line_items"]] == ["8%", "10%"]
    assert _without_tax_categories(output) == _without_tax_categories(original)
    assert metadata["accepted_fields"] == ["tax_categories"]
    assert metadata["tax_categories"]["reconciliation_mode"] == "net"
    assert metadata["tax_categories"]["proposed_category_sums"] == {
        "8%": 100.0,
        "10%": 200.0,
    }


def test_ambiguous_marker_rows_reject_the_whole_tax_vector_without_partial_mutation():
    receipt = {
        "merchant": "ストア",
        "location": None,
        "subtotal": 200,
        "line_items": [
            {"description": "食品A", "total": 100, "tax_category": "10%"},
            {"description": "食品A", "total": 100, "tax_category": "8%"},
        ],
        "taxes": [
            {"rate": "8%", "amount": 8},
            {"rate": "10%", "amount": 9},
        ],
    }
    original = deepcopy(receipt)
    evidence = _evidence(
        "明細\n* 食品A ¥100\n食品A ¥100\n小計\n¥200",
        "*は軽減税率8%適用商品\n8% 対象額 ¥100\n10% 対象額 ¥100",
    )

    output, metadata = apply_supplemental_ocr_evidence(receipt, evidence)

    assert receipt == original
    assert output == original
    assert metadata["accepted_fields"] == []
    assert not metadata["tax_categories"]["criteria"]["reduced_rows_uniquely_aligned"]


def test_non_reconciling_printed_rate_bases_reject_the_whole_tax_vector():
    receipt = {
        "merchant": "ストア",
        "location": None,
        "subtotal": 300,
        "line_items": [
            {"description": "食品A", "total": 100, "tax_category": "10%"},
            {"description": "用品B", "total": 200, "tax_category": "8%"},
        ],
        "taxes": [
            {"rate": "8%", "amount": 8},
            {"rate": "10%", "amount": 20},
        ],
    }
    original = deepcopy(receipt)
    evidence = _evidence(
        "明細\n* 食品A ¥100\n用品B ¥200\n小計\n¥300",
        "*は軽減税率8%適用商品\n8% 対象額 ¥110\n10% 対象額 ¥190",
    )

    output, metadata = apply_supplemental_ocr_evidence(receipt, evidence)

    assert receipt == original
    assert output == original
    assert metadata["accepted_fields"] == []
    assert not metadata["tax_categories"]["criteria"]["rate_base_arithmetic_reconciles"]


def test_exact_legend_owner_and_distinct_external_included_components_prove_unique_closed_partition_after_price_repair():
    from receipt_parser.ocr import blocks_to_structured_text
    from receipt_parser.receipt_financial import _rate_base_tax_pair_is_valid

    titles = ["食品240型", "用品270型", "食品300型", "食品400型", "袋500型"]
    values = [150, 150, 80, 30, 8]

    def evidence(marker="※", amounts=values, observed=60, subtotal=418, total=453, reduced_tax=20):
        layout = []
        for index, (title, amount) in enumerate(zip(titles, amounts, strict=True)):
            layout.extend([_word((marker if index == 0 else "") + title, x=20, y=index * 50, size=30),
                           _word(f"¥{amount}", x=420, y=index * 50, size=30)])
        footer = [f"小計/5点¥{subtotal}", f"8%外税タイショウ¥{observed}", f"8%外税¥{reduced_tax}",
                  "10%外税タイショウ¥150", "10%外税¥15", "(10%内税タイショウ¥8)", "(10%内税¥0)",
                  f"合計¥{total}", "※印は軽減税率対象商品"]
        layout.extend(_word(line, y=(len(titles) + index) * 50, size=30) for index, line in enumerate(footer))
        text = blocks_to_structured_text(layout)
        return build_supplemental_ocr_evidence(image_key=IMAGE_KEY, image_shape=IMAGE_SHAPE,
            strategy_fingerprint=STRATEGY_FINGERPRINT, merged_text=text,
            source_layout=layout, tile_texts=[text, "footer"])

    original = {"document_type": "receipt", "merchant": "店", "subtotal": 418, "total": 453,
                "taxes": [{"rate": "8%", "label": "外税", "amount": 20},
                          {"rate": "10%", "label": "外税", "amount": 15}],
                "line_items": [{"description": title, "qty": 1, "unit_price": amount,
                                "total": amount, "discount": 0, "discount_rate": "", "tax_category": "10%"}
                               for title, amount in zip(titles, values, strict=True)]}
    primary = "\n".join(f"{title}\n¥{amount}" for title, amount in zip(titles, values, strict=True)) + "\n小計\n¥418"
    source = evidence()
    output, meta = apply_supplemental_ocr_evidence(original, source, primary_text=primary)
    assert [item["tax_category"] for item in output["line_items"]] == ["8%", "10%", "8%", "8%", "10%"]
    assert meta["tax_categories"]["mode"] == "closed_distinct_tax_mode_marker_partition"
    assert meta["tax_categories"]["rejected_printed_base"] == 60
    assert meta["tax_categories"]["derived_reduced_base"] == 260
    unsupported_zero = deepcopy(original)
    for item in unsupported_zero["line_items"]:
        item["tax_category"] = "0%"
    assert apply_supplemental_ocr_evidence(unsupported_zero, source, primary_text=primary)[0] == output
    zero_partial = supplemental._partial_marker_projection(unsupported_zero, source, primary_text=primary)
    assert zero_partial["matches"][0]["item_index"] == 0
    for exempt_text in (primary.replace("¥150", "¥150 非", 1), primary + "\n非課税対象", primary + "\n免税"):
        assert not supplemental._mixed_mode_marker_partition(unsupported_zero, source, exempt_text, zero_partial)["accepted"]
    for exempt_footer in ("非課税対象", "¥30 非", "不課税", "0%対象額¥150", "0.00% 外税タイショウ¥150", "対象額(０％)¥150"):
        exempt_source = deepcopy(source)
        exempt_source["tiles"][1]["text"] = exempt_footer
        assert not supplemental._mixed_mode_marker_partition(unsupported_zero, exempt_source, primary, zero_partial)["accepted"]
    discount_source = deepcopy(source)
    discount_source["tiles"][1]["text"] = "割引0%"
    assert supplemental._mixed_mode_marker_partition(unsupported_zero, discount_source, primary, zero_partial)["accepted"]
    locked_zero = deepcopy(unsupported_zero)
    locked_zero["line_items"][0]["_tax_category_locked"] = "0%"
    assert not supplemental._partial_marker_projection(locked_zero, source, primary_text=primary)["matches"]
    locked_zero["line_items"][1]["_tax_category_locked"] = "0%"
    assert not supplemental._mixed_mode_marker_partition(locked_zero, source, primary, zero_partial)["accepted"]
    assert not _rate_base_tax_pair_is_valid("10%", 8, 0, "内税")
    assert apply_supplemental_ocr_evidence(output, source, primary_text=primary)[0] == output
    assert [item["total"] for item in output["line_items"]] == values
    assert {key: value for key, value in output.items() if key != "line_items"} == {
        key: value for key, value in original.items() if key != "line_items"}
    partial = supplemental._partial_marker_projection(original, source, primary_text=primary)
    assert partial["matches"][0]["item_index"] == 0
    broken = deepcopy(original)
    broken["line_items"][2]["unit_price"] = broken["line_items"][2]["total"] = 88
    repaired, repaired_meta = apply_supplemental_ocr_evidence(broken, source, primary_text=primary.replace("¥80", "¥88"))
    assert repaired == output
    assert repaired_meta["tax_categories"]["mode"] == "closed_distinct_tax_mode_marker_partition"
    assert not supplemental._mixed_mode_marker_partition(broken, source, primary, partial)["accepted"]
    assert not supplemental._mixed_mode_marker_partition(original, evidence(observed=259), primary, partial)["accepted"]
    assert not supplemental._mixed_mode_marker_partition(dict(original, total=454), source, primary, partial)["accepted"]
    assert apply_supplemental_ocr_evidence(original, evidence(marker="*"), primary_text=primary)[0] == original
    for text in (primary.replace("食品240型", "食品241型"), primary + "\n食品240型", None):
        assert not supplemental._partial_marker_projection(original, source, primary_text=text).get("matches")
    competing = deepcopy(original)
    for item, amount in zip(competing["line_items"], [200, 150, 80, 70, 8], strict=True):
        item["unit_price"] = item["total"] = amount
    competing.update(subtotal=508, total=551, taxes=[{"rate": "8%", "amount": 28}, {"rate": "10%", "amount": 15}])
    competing_source = evidence(amounts=[200, 150, 80, 70, 8], subtotal=508, total=551, reduced_tax=28)
    competing_partial = supplemental._partial_marker_projection(competing, competing_source, primary_text=primary)
    assert not supplemental._mixed_mode_marker_partition(competing, competing_source, primary, competing_partial)["accepted"]


def test_sidecar_round_trip_is_raw_only_and_identity_includes_image_shape(tmp_path):
    evidence = _evidence("merged text", "top tile", "bottom tile")

    path = save_supplemental_ocr_evidence(tmp_path, evidence)
    loaded = load_supplemental_ocr_evidence(
        tmp_path,
        image_key=IMAGE_KEY,
        image_shape=IMAGE_SHAPE,
        strategy_fingerprint=STRATEGY_FINGERPRINT,
    )

    assert loaded == evidence
    assert set(json.loads(path.read_text(encoding="utf-8"))) == {
        "schema_version",
        "image_key",
        "image_shape",
        "strategy_fingerprint",
        "merged_text",
        "source_layout",
        "tiles",
    }
    assert path != supplemental_ocr_cache_path(
        tmp_path,
        image_key=IMAGE_KEY,
        image_shape=[1201, 800],
        strategy_fingerprint=STRATEGY_FINGERPRINT,
    )


def test_quantity_input_replacement_requires_complete_pair_unique_title_and_both_owned_extended_prices(monkeypatch):
    from receipt_parser.pipeline import process_ocr_text
    import receipt_parser.pipeline as pipeline

    def evidence(text):
        return build_supplemental_ocr_evidence(
            image_key=IMAGE_KEY, image_shape=IMAGE_SHAPE,
            strategy_fingerprint=supplemental.SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT,
            merged_text=text, source_layout=_layout_for(text), tile_texts=[text, "footer"],
        )

    primary = "店舗領収書\n内*商品A-50 ¥198\n23 X #199\n小計\n¥198"
    source = "店舗領収書\n内 商品A-50 ¥198\n2 コ x 単99\n小計\n¥198"
    changed, proposals = supplemental.reconcile_supplemental_quantity_rows(primary, evidence(source))
    assert changed == primary.replace("23 X #199", "2 コ x 単99")
    assert len(proposals) == 1 and proposals[0]["total"] == 198
    certificate = "合計\n¥1000\n上記 正に 領収 しました\n担当 者 No1\n"
    assert supplemental.reconcile_supplemental_quantity_rows(
        certificate + primary, evidence(certificate + source),
    )[0] == certificate + changed
    assert supplemental.reconcile_supplemental_quantity_rows(changed, evidence(source)) == (changed, [])
    for malformed in ("(23] X #199)", "(23 X ₤199)"):
        corrupted = primary.replace("23 X #199", malformed)
        assert supplemental.reconcile_supplemental_quantity_rows(corrupted, evidence(source))[0] == changed
    bare_source = source.replace("¥198", "198", 1)
    assert supplemental.reconcile_supplemental_quantity_rows(primary, evidence(bare_source))[0] == changed
    for malformed in ("(2] X #100)", "(23 X ₤199.5)", "(23 X ₤-199)"):
        corrupted = primary.replace("23 X #199", malformed)
        assert supplemental.reconcile_supplemental_quantity_rows(corrupted, evidence(source)) == (corrupted, [])
    digit_complete_split = primary.replace("23 X #199", "(2個 X\n99)")
    joined, split_proposals = supplemental.reconcile_supplemental_quantity_rows(digit_complete_split, evidence(source))
    assert joined == changed
    assert split_proposals[0]["before"] == "(2個 X\n99)"
    assert split_proposals[0]["line_count"] == 2
    for conflicting_split in ("(3個 X\n99)", "(2個 X\n199)", "(2個 X\n別商品\n99)"):
        conflict = primary.replace("23 X #199", conflicting_split)
        assert supplemental.reconcile_supplemental_quantity_rows(conflict, evidence(source)) == (conflict, [])
    reversed_price = "196* A\n商品B-21\n<2個 X 単98)\n小計\n¥196"
    split_primary = "商品B-21\n196* A\n(21 X 198)\n小計\n¥196"
    assert supplemental.reconcile_supplemental_quantity_rows(split_primary, evidence(reversed_price))[0] == split_primary.replace("(21 X 198)", "<2個 X 単98)")
    for bad_source in (
        source.replace("¥198", "¥197", 1),
        source.replace("商品A-50", "商品A-51"),
        source.replace("2 コ", "23"),
        source.replace("2 コ x 単99", "2 コ x 単98"),
        source.replace("小計", "内 商品A-50 ¥198\n小計"),
        source.replace("内 商品A-50 ¥198", "別商品 ¥198\n内 商品A-50"),
    ):
        assert supplemental.reconcile_supplemental_quantity_rows(primary, evidence(bad_source)) == (primary, [])
    for bad_primary in (
        primary.replace("23 X #199", "2個 X 単100"),
        primary.replace("23 X #199", "2 X #100"),
        primary.replace("¥198", "¥197", 1),
        primary.replace("小計", "内*商品A-50 ¥198\n小計"),
        primary.replace("23 X #199", "別商品 ¥198\n23 X #199"),
        primary.replace("23 X #199", "23 X #199\n24 X #199"),
    ):
        assert supplemental.reconcile_supplemental_quantity_rows(bad_primary, evidence(source)) == (bad_primary, [])
    invalid = evidence(source)
    invalid["merged_text"] = invalid["merged_text"].replace("99", "98")
    assert supplemental.reconcile_supplemental_quantity_rows(primary, invalid) == (primary, [])
    captured = []
    monkeypatch.setattr(pipeline, "check_model_available", lambda *_: None)
    monkeypatch.setattr(pipeline, "_run_extraction_pipeline", lambda **kwargs: (
        captured.append(kwargs) or {"_error": "stopped before model"}, [], [], None,
    ))
    process_ocr_text(primary, supplemental_ocr_evidence=evidence(source), apply_user_rules=False)
    assert captured[0]["unified_text"] == changed
    assert captured[0]["raw_text"] == primary
    assert captured[0]["payment_reference_text"] == primary

    from receipt_parser.schema import Receipt

    receipt = Receipt(
        merchant="店舗", subtotal=198, total=198, currency="JPY",
        line_items=[{
            "description": "商品A-50", "qty": 1, "unit_price": 198,
            "total": 198, "discount": 0, "discount_rate": "",
            "tax_category": "10%",
        }],
    )
    monkeypatch.setattr(pipeline, "_run_extraction_pipeline", lambda **kwargs: (
        captured.append(kwargs) or receipt.model_dump(), [], [], receipt,
    ))
    result = process_ocr_text(
        primary, supplemental_ocr_evidence=evidence(source), apply_user_rules=False,
    )
    assert (result["line_items"][0]["qty"], result["line_items"][0]["unit_price"],
            result["line_items"][0]["total"]) == (2, 99, 198)
    assert captured[-1]["unified_text"] == changed
    assert captured[-1]["raw_text"] == captured[-1]["payment_reference_text"] == primary
    assert result["_supplemental_ocr_quantity_rows"] == proposals


def test_sidecar_load_fails_closed_on_corruption_or_answer_fields(tmp_path):
    evidence = _evidence("merged text", "top tile", "bottom tile")
    path = save_supplemental_ocr_evidence(tmp_path, evidence)
    expected = {
        "image_key": IMAGE_KEY,
        "image_shape": IMAGE_SHAPE,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
    }

    path.write_text("{not json", encoding="utf-8")
    assert load_supplemental_ocr_evidence(tmp_path, **expected) is None

    contaminated = {**evidence, "fixture": "receipt_49", "proposal": "known answer"}
    path.write_text(json.dumps(contaminated, ensure_ascii=False), encoding="utf-8")
    assert load_supplemental_ocr_evidence(tmp_path, **expected) is None

    wrong_identity = {**evidence, "image_shape": [1, 1]}
    path.write_text(json.dumps(wrong_identity, ensure_ascii=False), encoding="utf-8")
    assert load_supplemental_ocr_evidence(tmp_path, **expected) is None

    boolean_schema = {**evidence, "schema_version": True}
    path.write_text(json.dumps(boolean_schema, ensure_ascii=False), encoding="utf-8")
    assert load_supplemental_ocr_evidence(tmp_path, **expected) is None


def test_sidecar_build_and_load_reject_incoherent_layout_text(tmp_path):
    with pytest.raises(ValueError, match="invalid supplemental OCR evidence"):
        build_supplemental_ocr_evidence(
            image_key=IMAGE_KEY,
            image_shape=IMAGE_SHAPE,
            strategy_fingerprint=STRATEGY_FINGERPRINT,
            merged_text="caller supplied text",
            source_layout=_layout_for("raw mapped text"),
            tile_texts=["top tile", "bottom tile"],
        )

    evidence = _evidence("raw mapped text", "top tile", "bottom tile")
    path = save_supplemental_ocr_evidence(tmp_path, evidence)
    tampered = {**evidence, "merged_text": "tampered text"}
    path.write_text(json.dumps(tampered, ensure_ascii=False), encoding="utf-8")

    assert load_supplemental_ocr_evidence(
        tmp_path,
        image_key=IMAGE_KEY,
        image_shape=IMAGE_SHAPE,
        strategy_fingerprint=STRATEGY_FINGERPRINT,
    ) is None


def test_sidecar_load_rejects_malformed_nonfinite_and_out_of_bounds_layout(tmp_path):
    evidence = _evidence("raw mapped text", "top tile", "bottom tile")
    path = save_supplemental_ocr_evidence(tmp_path, evidence)
    expected = {
        "image_key": IMAGE_KEY,
        "image_shape": IMAGE_SHAPE,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
    }
    corrupt_layouts = []
    for field, value in (("y", "not-a-number"), ("y", float("inf"))):
        layout = deepcopy(evidence["source_layout"])
        layout[0][field] = value
        corrupt_layouts.append(layout)
    layout = deepcopy(evidence["source_layout"])
    layout[0]["bbox"][0] = [IMAGE_SHAPE[1] + 1, 0]
    corrupt_layouts.append(layout)
    layout = deepcopy(evidence["source_layout"])
    layout[0]["bbox"][0] = [0, float("nan")]
    corrupt_layouts.append(layout)

    for layout in corrupt_layouts:
        path.write_text(
            json.dumps({**evidence, "source_layout": layout}, ensure_ascii=False),
            encoding="utf-8",
        )
        assert load_supplemental_ocr_evidence(tmp_path, **expected) is None


def test_sidecar_load_rejects_nested_answers_and_invalid_block_metadata(tmp_path):
    evidence = _evidence("raw mapped text", "top tile", "bottom tile")
    path = save_supplemental_ocr_evidence(tmp_path, evidence)
    expected = {
        "image_key": IMAGE_KEY,
        "image_shape": IMAGE_SHAPE,
        "strategy_fingerprint": STRATEGY_FINGERPRINT,
    }
    corrupt_layouts = []
    for field, value in (
        ("confidence", float("inf")),
        ("confidence", -0.01),
        ("confidence", 1.01),
        ("page", True),
        ("page", -1),
        ("page", 0.0),
        ("text", ""),
    ):
        layout = deepcopy(evidence["source_layout"])
        layout[0][field] = value
        corrupt_layouts.append(layout)
    layout = deepcopy(evidence["source_layout"])
    layout[0]["expected_answer"] = "known value"
    corrupt_layouts.append(layout)

    for layout in corrupt_layouts:
        path.write_text(
            json.dumps({**evidence, "source_layout": layout}, ensure_ascii=False),
            encoding="utf-8",
        )
        assert load_supplemental_ocr_evidence(tmp_path, **expected) is None


def test_atomic_sidecar_replace_failure_preserves_old_file_and_cleans_temp(
    tmp_path,
    monkeypatch,
):
    original = _evidence("first merged text", "first top", "first bottom")
    path = save_supplemental_ocr_evidence(tmp_path, original)
    old_bytes = path.read_bytes()
    replacement = _evidence("second merged text", "second top", "second bottom")

    def fail_replace(_source, _destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(supplemental.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        save_supplemental_ocr_evidence(tmp_path, replacement)

    assert path.read_bytes() == old_bytes
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_direct_summary_recovery_requires_unique_closed_owners_and_abstains_atomically():
    from receipt_parser import pipeline
    from receipt_parser.ocr import blocks_to_structured_text

    primary = "sample item\n¥200\n小計200\n消費税20\n10%対象\n220(内税額20)"

    def evidence(rate_rows=("10%対象",), duplicate_subtotal=False):
        blocks = []
        rows = [("小計", "200"), ("消費税", "20"), ("合計", "220")]
        for index, (label, amount) in enumerate(rows):
            y = index * 40
            blocks.extend((_word(label, x=10, y=y, size=16),
                           _word(amount, x=300, y=y, size=16)))
        for index, rate in enumerate(rate_rows):
            blocks.append(_word(rate, x=10, y=140 + index * 40, size=16))
        blocks.append(_word("220(内税額20)", x=10, y=220, size=16))
        if duplicate_subtotal:
            blocks.extend((_word("小計", x=10, y=260, size=16),
                           _word("200", x=300, y=260, size=16)))
        text = blocks_to_structured_text(blocks)
        return build_supplemental_ocr_evidence(
            image_key=IMAGE_KEY, image_shape=IMAGE_SHAPE,
            strategy_fingerprint=STRATEGY_FINGERPRINT, merged_text=text,
            source_layout=blocks, tile_texts=[text, "summary"],
        )

    def receipt():
        return {
            "document_type": "receipt", "merchant": "SHOP",
            "subtotal": 199, "total": 220,
            "taxes": [{"rate": "10%", "label": "内税", "amount": 5}],
            "amount_paid": 220, "points_used": None,
            "line_items": [{"description": "sample item", "qty": 1,
                            "unit_price": 200, "total": 200,
                            "discount": 0, "discount_rate": "",
                            "tax_category": "10%"}],
        }

    result = receipt()
    before = deepcopy(result)
    trace = []
    pipeline._apply_final_supplemental_ocr_evidence(
        result, evidence(), mutation_trace=trace, primary_text=primary,
    )
    expected = deepcopy(before)
    expected["subtotal"] = 200
    expected["taxes"][0].update(label="外税", amount=20)
    assert {key: value for key, value in result.items() if not key.startswith("_")} == expected
    assert len(trace) == 1
    assert trace[0]["owner_phase"] == "supplemental_ocr_field_recovery"
    assert set(trace[0]["changes"]) == {"subtotal", "taxes"}
    assert {"subtotal", "total", "taxes"} <= set(trace[0]["writes"])
    assert {"amount_paid", "points_used"} <= set(trace[0]["reads"])

    mismatched_items = receipt()
    mismatched_items["line_items"][0].update(unit_price=199, total=199)
    mismatched_payment = receipt()
    mismatched_payment["amount_paid"] = 219
    conflicting_zero_tax = receipt()
    conflicting_zero_tax["taxes"].append({"rate": "0%", "label": "非課税", "amount": 1})
    unowned_tax = receipt()
    unowned_tax["taxes"].append({"rate": "unknown", "label": "unknown", "amount": 1})
    abstentions = (
        (receipt(), evidence(duplicate_subtotal=True)),
        (receipt(), evidence(rate_rows=("8%対象",))),
        (receipt(), evidence(rate_rows=("10.5%対象",))),
        (receipt(), evidence(rate_rows=("1e10%対象",))),
        (receipt(), evidence(rate_rows=("x10%対象",))),
        (receipt(), evidence(rate_rows=("-10%対象",))),
        (receipt(), evidence(rate_rows=("+10%対象",))),
        (receipt(), evidence(rate_rows=("8%対象", "10%対象"))),
        (mismatched_items, evidence()),
        (mismatched_payment, evidence()),
        (conflicting_zero_tax, evidence()),
        (unowned_tax, evidence()),
    )
    for candidate, bound in abstentions:
        before = deepcopy(candidate)
        rejected_trace = []
        pipeline._apply_final_supplemental_ocr_evidence(
            candidate, bound, mutation_trace=rejected_trace, primary_text=primary,
        )
        assert candidate == before
        assert rejected_trace == []


def test_source_discount_rate_requires_unique_exact_title_money_and_primary_annotation():
    from receipt_parser.ocr import blocks_to_structured_text

    title = "試作tea 7"
    primary = title + "\n￥813※\n40% -326"
    item = {"description": title + " ￥813※", "qty": 1, "unit_price": 813,
            "total": 487, "discount": 326, "discount_rate": "", "tax_category": "10%"}
    original = {"document_type": "receipt", "line_items": [item], "taxes": [],
                "subtotal": 487, "total": 487}

    def evidence(discount="40% -326", repeats=1):
        blocks = []
        for index in range(repeats):
            y = index * 90
            blocks.extend((_word(title, x=10, y=y), _word("￥813", x=300, y=y),
                           _word(discount, x=10, y=y + 40)))
        text = blocks_to_structured_text(blocks, sort_within_rows=True)
        return build_supplemental_ocr_evidence(
            image_key=IMAGE_KEY, image_shape=IMAGE_SHAPE,
            strategy_fingerprint=supplemental.SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT,
            merged_text=text, source_layout=blocks, tile_texts=[text, text],
        )

    output, meta = apply_supplemental_ocr_evidence(original, evidence(), primary_text=primary)
    assert output == dict(original, line_items=[dict(item, discount_rate="40%")])
    assert meta["accepted_fields"] == ["line_items"]
    assert original["line_items"][0] == item
    for receipt, sealed, text in (
        (original, evidence(), title + "\n40% -326"),
        (original, evidence(), primary + "\n" + primary),
        (original, evidence(repeats=2), primary),
        (dict(original, line_items=[deepcopy(item), deepcopy(item)]), evidence(), primary),
        (dict(original, line_items=[dict(item, description="試作tea 8 ￥813※")]), evidence(), primary),
        (original, evidence("外税 40% -326"), primary),
        (original, evidence("+40% -326"), primary),
        (original, evidence("240% -326"), primary),
        (original, evidence("40% -325"), primary),
    ):
        before = deepcopy(receipt)
        output, meta = apply_supplemental_ocr_evidence(receipt, sealed, primary_text=text)
        assert output == before
        assert receipt == before

    def scheduled(title, gross, rows, repeats=1, next_product=None):
        blocks = []
        for index in range(repeats):
            y = index * 180
            blocks.extend((_word(title, x=10, y=y), _word(f"￥{gross}", x=300, y=y),
                           _word("※", x=350, y=y), _word("↓", x=375, y=y)))
            blocks.extend(_word(row, x=10, y=y + 40 * (offset + 1))
                          for offset, row in enumerate(rows))
            if next_product is not None:
                next_y = y + 40 * (len(rows) + 1)
                blocks.extend((_word(next_product[0], x=10, y=next_y),
                               _word(next_product[1], x=300, y=next_y)))
        text = blocks_to_structured_text(blocks, sort_within_rows=True)
        return build_supplemental_ocr_evidence(
            image_key=IMAGE_KEY, image_shape=IMAGE_SHAPE,
            strategy_fingerprint=supplemental.SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT,
            merged_text=text, source_layout=blocks, tile_texts=[text, text],
        )

    for qty, unit, discount, net, rows, expected_rate in (
        (1, 200, 67, 133, ("割 引30% -60", "会 員 様 割 引5% -7"), "33.5%"),
        (3, 80, 12, 228, ("(3個 X 単80)", "会員 様割引5% -12"), "5%"),
    ):
        title = "試作soup 9"
        current_item = dict(item, description=title, qty=qty, unit_price=unit,
                            discount=discount, total=net)
        current = dict(original, line_items=[current_item], subtotal=net, total=net)
        sealed = scheduled(title, qty * unit, rows)
        output, _ = apply_supplemental_ocr_evidence(current, sealed)
        assert output == dict(current, line_items=[dict(current_item, discount_rate=expected_rate)])
        unchanged = deepcopy(output)
        output, _ = apply_supplemental_ocr_evidence(output, sealed)
        assert output == unchanged
        for candidate, invalid in (
            (current, scheduled(title, qty * unit, rows, repeats=2)),
            (dict(current, line_items=[deepcopy(current_item), deepcopy(current_item)]), sealed),
            (dict(current, line_items=[dict(current_item, qty=qty + 1)]), sealed),
            (dict(current, line_items=[dict(current_item, unit_price=unit + 1)]), sealed),
            (dict(current, line_items=[dict(current_item, total=net + 1)]), sealed),
            (current, scheduled(title, qty * unit, rows[:-1])),
            (current, scheduled(title, qty * unit, (rows[0], "別の商品 ￥400", rows[-1]))),
            (current, scheduled(title, qty * unit, (*rows, "割引10% -9"))),
            (current, scheduled(title, qty * unit, tuple(row.replace("5%", "240%") for row in rows))),
            (current, scheduled(title, qty * unit, tuple(row.replace("5%", "+5%") for row in rows))),
        ):
            before = deepcopy(candidate)
            output, _ = apply_supplemental_ocr_evidence(candidate, invalid)
            assert output == before
            assert candidate == before

    title = "試作beef 6"
    current_item = dict(item, description=title, unit_price=200, discount=80, total=120)
    current = dict(original, line_items=[current_item], subtotal=120, total=120)
    for rows, repeats, expected in (
        (("割引40% -80",), 1, "40%"),
        (("割引40% -80", "割引240% -80"), 1, ""),
        (("割引+40% -80",), 1, ""),
        (("割引40%",), 1, ""),
        ((), 1, ""),
        (("割引40% -80",), 2, ""),
    ):
        sealed = scheduled(title, 200, rows, repeats=repeats,
                           next_product=("試作rice 8", "￥152%"))
        output, _ = apply_supplemental_ocr_evidence(current, sealed)
        assert output == dict(current, line_items=[dict(current_item, discount_rate=expected)])
        assert current["line_items"][0] == current_item


@pytest.fixture
def quantity_crop_case():
    image = np.empty((110, 300, 3), dtype=np.uint8)
    image[:] = [70, 120, 200]
    layout = [_word("店舗領収書", x=10, y=10, size=12),
              _word("商品A-50", x=10, y=34, size=12), _word("¥198", x=240, y=34, size=12),
              _word("23", x=10, y=58, size=12), _word("X", x=50, y=58, size=12),
              _word("#199", x=80, y=58, size=12),
              _word("小計", x=10, y=82, size=12), _word("¥198", x=240, y=82, size=12)]
    text = ocr.blocks_to_structured_text(layout) + "\nPayPay ¥198"
    context = supplemental.build_quantity_crop_context(image, text, primary_layout_blocks=layout)
    assert context is not None
    recovered = [deepcopy(word) for word in layout[1:6]]
    recovered[2]["text"], recovered[4]["text"] = "2個", "単99"
    merged = ocr.blocks_to_structured_text(recovered)
    record = dict(context=deepcopy(context), vision_text=merged, source_layout=recovered, merged_text=merged)
    return image, text, layout, context, record


def _mock_quantity_crop_vision(monkeypatch, context, record):
    calls, initialized = _mock_vision(monkeypatch, texts=(record["vision_text"],))
    ordinary_call = supplemental._call_cloud_vision
    words = deepcopy(record["source_layout"])
    for word in words:
        word["bbox"] = [[(x - context["roi"][0]) * 4, (y - context["roi"][1]) * 4] for x, y in word["bbox"]]
        word["x"], word["y"] = min(p[0] for p in word["bbox"]), min(p[1] for p in word["bbox"])

    def strict(image, client, *, allow_fallback):
        assert allow_fallback is False
        assert image.shape[:2] == tuple(context["processed_shape"])
        assert image.dtype == np.uint8 and image[0, 0].tolist() == [70, 120, 200]
        return ordinary_call(image, client)

    monkeypatch.setattr(supplemental, "_call_cloud_vision", strict)
    monkeypatch.setattr(supplemental, "_extract_words_from_response", lambda _: deepcopy(words))
    return calls, initialized


@pytest.mark.parametrize("selection", ["fresh_original", "fresh_rotated", "cached_foreign"])
def test_quantity_crop_document_authenticates_original_frame_before_acquisition(monkeypatch, tmp_path, quantity_crop_case, selection):
    image, text, layout, _, record = quantity_crop_case
    key = ocr._ocr_cache_key(image)
    selected = ocr.OCRResult(blocks=ocr._fulltext_to_blocks(text), layout_blocks=layout,
        confidence=0.9, source="cache" if selection == "cached_foreign" else "fresh",
        chosen_text=text, layout_trusted=True)
    responses = [ocr.OCRResult(blocks=[_word("weak", confidence=0.1)]) , selected] if selection == "fresh_rotated" else [selected]
    monkeypatch.setattr(pipeline, "run_cloud_vision", lambda *_a, **_k: responses.pop(0))
    monkeypatch.setattr(pipeline, "load_image", lambda _: [image])
    monkeypatch.setattr(pipeline, "check_model_available", lambda *_: None)
    monkeypatch.setattr(usage, "track_document", lambda *_: None)
    monkeypatch.setattr(pipeline, "detect_document_type", lambda _: "receipt")
    merged = ocr.blocks_to_structured_text(layout)
    s_evidence = build_supplemental_ocr_evidence(image_key=key, image_shape=list(image.shape[:2]),
        strategy_fingerprint=supplemental.SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT,
        merged_text=merged, source_layout=layout, tile_texts=[merged, merged])
    monkeypatch.setattr(pipeline, "acquire_supplemental_ocr_evidence", lambda *_a, **_k: s_evidence)
    # Read-only synthetic bound cache envelope: trust=True alone must not attest its source.
    envelope = dict(schema_version=1, ocr_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        layout_blocks_sha256=ocr.layout_blocks_sha256(layout), layout_blocks=layout, ocr_confidence=0.9,
        provenance=dict(kind="same_call_capture", source="f" * 32))
    cache_inputs = {key + ".txt": text, key + ".layout.json": json.dumps(envelope, ensure_ascii=False)}
    real_read, real_is_file = Path.read_text, Path.is_file
    monkeypatch.setattr(Path, "read_text", lambda path, *a, **k: cache_inputs[path.name] if path.name in cache_inputs else real_read(path, *a, **k))
    monkeypatch.setattr(Path, "is_file", lambda path: path.name in cache_inputs or real_is_file(path))
    acquired, extracted = [], []

    def acquire(native, context, *, mode, client, **_kwargs):
        assert native is image and client == "mock-client"
        acquired.append((context, mode))
        return dict(deepcopy(record), context=deepcopy(context))

    monkeypatch.setattr(pipeline, "acquire_quantity_crop_ocr_evidence", acquire)
    monkeypatch.setattr(pipeline, "_run_extraction_pipeline", lambda **kwargs:
        (extracted.append(kwargs) or {"_error": "stop before model"}, [], [], None))
    fresh = selection != "cached_foreign"
    pipeline.process_document(tmp_path / "scan.png", ocr_engine="mock-client", skip_ocr_cache=fresh, apply_user_rules=False)
    assert len(acquired) == 1
    assert acquired[0][0]["source_kind"] == ("primary" if selection == "fresh_original" else "sealed_supplemental")
    assert acquired[0][0]["image_key"] == key and acquired[0][1] == ("fresh" if fresh else "normal")
    expected = text.replace(ocr.blocks_to_structured_text(layout[3:6]),
                            ocr.blocks_to_structured_text(record["source_layout"][2:]))
    assert extracted[0]["unified_text"] == strip_barcode_lines(expected)
    assert extracted[0]["raw_text"] == extracted[0]["payment_reference_text"] == text
    assert extracted[0]["ocr_layout_blocks"] == layout  # Ordinary extraction retains its selected geometry.


@pytest.mark.parametrize("control", ["valid", "no_context", "stale_text", "word", "answer", "context", "bounds"])
def test_quantity_crop_injected_text_needs_explicit_context_and_preserves_raw_payment_newlines(monkeypatch, quantity_crop_case, control):
    image, text, layout, original_context, record = quantity_crop_case
    injected = "\n" + text.replace("¥198", "￥１９８") + "\n\n"
    normalized = normalize_fullwidth(injected)
    context = supplemental.build_quantity_crop_context(image, normalized, primary_layout_blocks=layout)
    record = dict(deepcopy(record), context=deepcopy(context))
    assert context["input_text_sha256"] == hashlib.sha256(normalized.encode()).hexdigest()
    if control == "no_context":
        context = None
    elif control == "stale_text":
        context = original_context
    elif control == "word":
        record["source_layout"][0]["page"] = 1
    elif control == "answer":
        record["answer"] = {"qty": 2}
    elif control == "context":
        record["context"]["png_sha256"] = "f" * 64
    elif control == "bounds":
        record["source_layout"][0]["bbox"][0][1] = context["roi"][1] - 1
    captured = []
    monkeypatch.setattr(pipeline, "check_model_available", lambda *_: None)
    monkeypatch.setattr(pipeline, "_run_extraction_pipeline", lambda **kwargs:
        (captured.append(kwargs) or {"_error": "stop before model"}, [], [], None))
    for name in ("load_image", "init_cloud_vision", "acquire_quantity_crop_ocr_evidence", "build_quantity_crop_context"):
        monkeypatch.setattr(pipeline, name, lambda *_a, **_k: pytest.fail("injected text performed image/provider/context acquisition"))
    pipeline.process_ocr_text(injected, quantity_crop_ocr_evidence=record, quantity_crop_context=context, apply_user_rules=False)
    expected = normalized.replace(ocr.blocks_to_structured_text(layout[3:6]),
                                  ocr.blocks_to_structured_text(record["source_layout"][2:])) if control == "valid" else normalized
    assert captured[0]["unified_text"] == strip_barcode_lines(expected)
    assert captured[0]["raw_text"] == captured[0]["payment_reference_text"] == normalized
    assert normalized.startswith("\n") and normalized.endswith("\n\n")


def test_quantity_crop_concurrent_normal_misses_share_one_strict_acquisition(monkeypatch, tmp_path, quantity_crop_case):
    image, _, _, context, record = quantity_crop_case
    calls, initialized = _mock_quantity_crop_vision(monkeypatch, context, record)
    real_load = supplemental.load_quantity_crop_ocr_evidence
    initial_loads, state = threading.Barrier(2), threading.local()

    def synchronized_load(*args, **kwargs):
        result = real_load(*args, **kwargs)
        if not getattr(state, "loaded", False):
            state.loaded = True
            initial_loads.wait(timeout=5)
        return result

    monkeypatch.setattr(supplemental, "load_quantity_crop_ocr_evidence", synchronized_load)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(supplemental.acquire_quantity_crop_ocr_evidence, image, context, cache_dir=tmp_path) for _ in range(2)]
        results = [future.result(timeout=10) for future in futures]
    cached = real_load(tmp_path, expected_context=context)
    assert results[0] == results[1] == cached == record
    assert len(calls) == 1 and initialized == [True]
    assert supplemental._NORMAL_CACHE_LOCKS == {}


@pytest.mark.parametrize("content", [None, b"{corrupt", b"\xff"])
def test_quantity_crop_cache_only_miss_or_corruption_never_initializes_vision(monkeypatch, tmp_path, quantity_crop_case, content):
    image, _, _, context, _ = quantity_crop_case
    for name in ("init_cloud_vision", "_call_cloud_vision", "_atomic_write_text"):
        monkeypatch.setattr(supplemental, name, lambda *_a, **_k: pytest.fail("cache-only touched provider or writer"))
    if content is not None:
        supplemental.quantity_crop_ocr_cache_path(tmp_path, context).write_bytes(content)
    assert supplemental.acquire_quantity_crop_ocr_evidence(image, context, mode="cache_only", cache_dir=tmp_path) is None


def test_quantity_crop_fresh_ignores_disk_and_keeps_original_color(monkeypatch, tmp_path, quantity_crop_case):
    image, _, _, context, record = quantity_crop_case
    calls, initialized = _mock_quantity_crop_vision(monkeypatch, context, record)
    for name in ("load_quantity_crop_ocr_evidence", "_atomic_write_text"):
        monkeypatch.setattr(supplemental, name, lambda *_a, **_k: pytest.fail("fresh touched disk cache"))
    acquired = supplemental.acquire_quantity_crop_ocr_evidence(image, context, mode="fresh", cache_dir=tmp_path)
    assert acquired == record and len(calls) == 1 and initialized == [True]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("cap", ["pixel_cap", "png_byte_cap"])
def test_quantity_crop_caps_reject_before_provider_initialization(monkeypatch, quantity_crop_case, cap):
    image, _, _, context, _ = quantity_crop_case
    for name in ("init_cloud_vision", "_call_cloud_vision", "_atomic_write_text"):
        monkeypatch.setattr(supplemental, name, lambda *_a, **_k: pytest.fail("cap guard ran after provider/write"))
    monkeypatch.setitem(supplemental.QUANTITY_CROP_OCR_STRATEGY, cap, 1)
    try:
        result = supplemental.acquire_quantity_crop_ocr_evidence(image, context, mode="fresh")
    except ValueError as exc:
        assert "cap" in str(exc)
    else:
        assert result is None


def test_strict_quantity_vision_error_cannot_fall_back_to_default_model(monkeypatch):
    def unavailable(**_kwargs):
        raise RuntimeError("stable model unavailable")

    client = SimpleNamespace(annotate_image=unavailable,
        document_text_detection=lambda **_k: pytest.fail("strict crop requested fallback"))
    monkeypatch.setattr(ocr, "_track_api_call", lambda: pytest.fail("failed strict call counted as successful"))
    with pytest.raises(RuntimeError, match="stable model unavailable"):
        ocr._call_cloud_vision(np.zeros((4, 4, 3), dtype=np.uint8), client, allow_fallback=False)


@pytest.mark.parametrize("detail", ["2個 X 単99", "12個 X 単99", "商品型23X199", "合計 ¥198"])
def test_complete_or_unrelated_quantity_text_skips_crop_geometry(monkeypatch, detail):
    for name in ("_group_layout_rows", "_layout_row_price_candidates"):
        monkeypatch.setattr(supplemental, name, lambda *_: pytest.fail("ineligible text read crop geometry"))
    assert supplemental._quantity_crop_candidate(f"商品A-50 ¥198\n{detail}", [], [110, 300]) is None


@pytest.mark.parametrize("marker", ["X", "%"])
def test_geometry_owned_price_marker_preserves_literal_quantity_discount_rate(marker):
    title = "試作tea7"
    blocks = [_word(title, x=10, y=10), _word("48" + marker, x=240, y=10),
              _word("(2個 X 単24)", x=10, y=34), _word("割引25% -12", x=10, y=58)]
    item = dict(description=title, qty=2, unit_price=24, total=36,
                discount=12, discount_rate="", tax_category="8%")

    def proof(words):
        return dict(source_layout=words, merged_text=ocr.blocks_to_structured_text(words))

    current = dict(line_items=[deepcopy(item)])
    proposals = supplemental._recover_owned_discount_rates(current, proof(blocks))
    assert len(proposals) == 1 and current["line_items"] == [dict(item, discount_rate="25%")]
    for changed_item, words in (
        (dict(item, description="試作tea8"), blocks),
        (dict(item, discount_rate="10%"), blocks),
        (dict(item, qty=3, unit_price=16), blocks),
        (item, blocks + [_word(title, x=10, y=82), _word("48" + marker, x=240, y=82)]),
        (item, blocks[:3] + [_word("割引25% -10", x=10, y=58)]),
        (item, blocks[:3] + [_word("値引 -12", x=10, y=58)]),
        (item, blocks[:2] + [_word("別の商品", x=10, y=34)] + blocks[3:]),
        (item, [_word("48%", x=240, y=10)] + blocks[2:]),
    ):
        current = dict(line_items=[deepcopy(changed_item)])
        assert supplemental._recover_owned_discount_rates(current, proof(words)) == []
        assert current["line_items"] == [changed_item]
    # A standalone percentage still cannot prove money without geometry.
    loose = [deepcopy(item)]
    supplemental._clear_discounts_without_nearby_ocr_marker(
        loose, title + "\n48%\n(2個 X 単24)\n割引25% -12", rates_only=True)
    assert loose[0]["discount_rate"] == ""


@pytest.mark.parametrize("control", ["owned", "missing_native", "duplicate_primary", "wrong_native_amount",
                                     "different_primary_rate", "already_local", "after_summary",
                                     "conflicting_local_amount", "blank_conflicting_local_amount",
                                     "blank_conflicting_local_rate", "blank_ambiguous_local_control",
                                     "spaced_discount_label", "member_label", "spaced_coupon_label",
                                     "missing_next_native_owner", "different_next_native_owner",
                                     "blank_before_next_item", "amount_only_native"])
def test_native_quantity_crop_relocates_only_unique_corroborated_arithmetic_discount(control):
    image = np.full((180, 300, 3), 120, dtype=np.uint8)
    layout = [_word("店舗領収書", x=10, y=10, size=12),
              _word("試作cereal7", x=10, y=34, size=12), _word("444*", x=240, y=34, size=12),
              _word("(31 X ₤1148)", x=10, y=58, size=12),
              _word("割引25% -111", x=10, y=82, size=12),
              _word("別商品8", x=10, y=106, size=12), _word("￥240*", x=240, y=106, size=12),
              _word("別商品9", x=10, y=130, size=12), _word("￥60*", x=240, y=130, size=12),
              _word("小計", x=10, y=154, size=12), _word("￥633", x=240, y=154, size=12)]
    text = "店舗領収書\n試作cereal7\n444*\n(31 X ₤1148)\n割引\n25%\n別商品8\n￥240*\n別商品9\n￥60*\n-111\n小計 ￥633"
    if control == "duplicate_primary":
        text = "-111\n" + text
    elif control == "different_primary_rate":
        text = text.replace("25%", "20%")
    elif control == "already_local":
        text = text.replace("\n-111", "").replace("25%", "25%\n-111")
    elif control == "after_summary":
        text = text.replace("\n-111", "") + "\n-111"
    elif control == "conflicting_local_amount":
        text = text.replace("25%", "25%\n-110")
    elif control == "blank_conflicting_local_amount":
        text = text.replace("25%", "25%\n\n-110")
    elif control == "blank_conflicting_local_rate":
        text = text.replace("25%", "25%\n\n会員5%")
    elif control == "blank_ambiguous_local_control":
        text = text.replace("25%", "25%\n\nクーポン")
    elif control in {"spaced_discount_label", "member_label", "spaced_coupon_label"}:
        label = {"spaced_discount_label": "値 引", "member_label": "会員", "spaced_coupon_label": "クー ポン"}[control]
        text = text.replace("25%", "25%\n\n" + label + "\n-110")
    elif control == "blank_before_next_item":
        text = text.replace("25%", "25%\n\n")
    context = supplemental.build_quantity_crop_context(image, text, primary_layout_blocks=layout)
    assert context is not None
    recovered = deepcopy(layout[1:5])
    recovered[2]["text"] = "(3個 X 単148)"
    if control == "missing_native":
        recovered.pop()
    elif control == "wrong_native_amount":
        recovered[3]["text"] = "割引25% -99"
    elif control == "amount_only_native":
        recovered[3]["text"] = "値引 -111"
    next_owner = [_word("別商品8", x=10, y=96, size=10), _word("￥240*", x=240, y=96, size=10)]
    if control == "missing_next_native_owner":
        next_owner = []
    elif control == "different_next_native_owner":
        next_owner[0]["text"] = "別商品80"
    native_words = [_word("別文脈18%", x=10, y=10, size=12)] + recovered + next_owner
    merged = ocr.blocks_to_structured_text(native_words)
    record = dict(context=deepcopy(context), vision_text=merged, source_layout=native_words, merged_text=merged)
    before = deepcopy(record)
    changed, proposals = supplemental.reconcile_quantity_crop_rows(text, record, expected_context=context)
    expected = text.replace("(31 X ₤1148)", "(3個 X 単148)")
    if control in {"owned", "blank_before_next_item"}:
        expected = expected.replace("\n-111", "").replace("25%", "25%\n-111")
        assert proposals[0]["discount_relocation"]["discount"] == 111
    else:
        assert "discount_relocation" not in proposals[0]
    assert changed == expected and len(proposals) == 1 and record == before
    assert supplemental.reconcile_quantity_crop_rows(changed, record, expected_context=context) == (changed, [])


def test_lower_detail_rate_requires_unique_declaration_and_its_own_closed_subtotal():
    from receipt_parser.ocr import blocks_to_structured_text

    item = dict(description="試作珈琲7", qty=2, unit_price=250, total=375,
                discount=125, discount_rate="", tax_category="8%")
    other = dict(description="紙箱4", qty=1, unit_price=80, total=80,
                 discount=0, discount_rate="", tax_category="10%")
    original = dict(document_type="receipt", subtotal=455, total=455, taxes=[], line_items=[item, other])
    declaration = "上記 金額 正に 領収 いたしました"
    primary = "\n".join(["合計 ¥455", declaration, "試作珈琲7 ¥500", "2個 X 単250",
                          "割引25% -125", "紙箱4 ¥80", "小計 ¥455"])
    blocks = [_word("合計", x=10, y=10), _word("¥455", x=300, y=10),
              _word(declaration, x=10, y=40),
              _word("試作珈琲7", x=10, y=70), _word("¥500", x=300, y=70),
              _word("2個 X 単250", x=10, y=100), _word("割引25% -125", x=10, y=130),
              _word("紙箱4", x=10, y=160), _word("¥80", x=300, y=160),
              _word("小計", x=10, y=190), _word("¥455", x=300, y=190),
              _word("試作珈琲7", x=10, y=220), _word("¥500", x=300, y=220),
              _word("割引50% -250", x=10, y=250)]

    def sealed(words):
        text = blocks_to_structured_text(words, sort_within_rows=True)
        return build_supplemental_ocr_evidence(
            image_key=IMAGE_KEY, image_shape=IMAGE_SHAPE,
            strategy_fingerprint=STRATEGY_FINGERPRINT,
            merged_text=text, source_layout=words, tile_texts=[text, text],
        )

    expected = dict(original, line_items=[dict(item, discount_rate="25%"), other])
    output, meta = apply_supplemental_ocr_evidence(original, sealed(blocks), primary_text=primary)
    assert output == expected and meta["accepted_fields"] == ["line_items"]
    assert original["line_items"] == [item, other]
    assert apply_supplemental_ocr_evidence(output, sealed(blocks), primary_text=primary)[0] == expected
    trace = []
    current = deepcopy(original)
    pipeline._apply_final_supplemental_ocr_evidence(current, sealed(blocks), mutation_trace=trace, primary_text=primary)
    assert {key: value for key, value in current.items() if not key.startswith("_")} == expected
    assert len(trace) == 1 and set(trace[0]["changes"]) == {"line_items"}

    for candidate, words, text in (
        (original, blocks[:2] + blocks[3:], primary),
        (original, blocks, primary.replace(declaration, "別のご案内")),
        (original, blocks + [_word(declaration, x=10, y=280)], primary),
        (original, blocks, primary + "\n" + declaration),
        (original, blocks, primary.replace("小計 ¥455", "小計 ¥456")),
        (original, blocks[:10] + [_word("¥456", x=300, y=190)] + blocks[11:], primary),
        (original, blocks[:10] + [_word("455", x=300, y=190)] + blocks[11:], primary),
        (dict(original, subtotal=456), blocks, primary),
        (dict(original, line_items=[item, dict(other, total=81)]), blocks, primary),
        (original, blocks[:3] + [_word("合計", x=10, y=55)] + blocks[3:], primary),
        (original, blocks[:3] + [_word("試作珈琲70", x=10, y=70)] + blocks[4:], primary),
        (original, blocks[:6] + [_word("割引250% -125", x=10, y=130)] + blocks[7:], primary),
        (original, blocks[:6] + [_word("割引25% -124", x=10, y=130)] + blocks[7:], primary),
        (original, blocks[:6] + [_word("値引 -125", x=10, y=130)] + blocks[7:], primary),
        (original, blocks[:6] + blocks[7:], primary),
        (dict(original, line_items=[dict(item, discount_rate="20%"), other]), blocks, primary),
    ):
        before = deepcopy(candidate)
        assert supplemental._recover_owned_discount_rates(candidate, sealed(words), text) == []
        assert candidate == before


def test_code_prefixed_pos_table_requires_complete_literal_owners():
    from receipt_parser.receipt_item_repair import _clean_code_prefixed_item_descriptions
    from receipt_parser.receipt_supplemental_ocr import SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT
    from receipt_parser.ocr import blocks_to_structured_text
    from receipt_parser.schema import Receipt
    IMAGE_KEY, IMAGE_SHAPE = "c" * 32, [240, 420]
    PRIMARY = "418-0271 試作SAMPLE-A 2025 1\n¥512\n529-0314 試作MODEL-B 2L\n1 ¥634\n点数\n2点\n小計\n¥1,146"


    def word(text, x, y, width=40):
        return {"text": text, "confidence": .99, "x": x, "y": y,
                "bbox": [[x, y], [x + width, y], [x + width, y + 18], [x, y + 18]], "page": 0}


    def evidence(rows, discount=False):
        blocks = []
        for i, (code, title, qty, amount, currency) in enumerate(rows):
            y = 10 + i * 50
            blocks.extend((word(code, 10, y), word(title, 100, y, 100), word(qty, 235, y, 12)))
            if currency:
                blocks.append(word("¥", 298, y, 12))
            blocks.append(word(amount, 310, y, 45))
        if discount:
            blocks.extend((word("割引", 10, 110), word("¥", 298, 110, 12), word("10", 310, 110, 25)))
        blocks.extend((word("点数", 10, 130), word("2点", 310, 130, 25),
                       word("小計", 10, 160), word("¥", 298, 160, 12), word("1,146", 310, 160, 45)))
        text = blocks_to_structured_text(blocks, sort_within_rows=True)
        return build_supplemental_ocr_evidence(
            image_key=IMAGE_KEY, image_shape=IMAGE_SHAPE,
            strategy_fingerprint=SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT,
            merged_text=text, source_layout=blocks, tile_texts=[text, text],
        )


    def receipt():
        return Receipt(
            merchant="SAMPLE", subtotal=1146, total=1146,
            line_items=[
                {"description": "418-0271 試作SAMPLE-A 2025", "qty": 1, "unit_price": 512, "total": 512, "tax_category": "10%"},
                {"description": "529-0314 試作MODEL-B 2L", "qty": 1, "unit_price": 634, "total": 634, "tax_category": "10%"},
            ],
        ).model_dump()


    def unchanged(value, source, primary):
        before = deepcopy(value)
        proposed, _ = apply_supplemental_ocr_evidence(value, source, primary_text=primary)
        assert proposed == before and value == before


    def main():
        source = evidence((
            ("418-0271", "試作SAMPLE-A 2025", "1", "512", True),
            # Source glyphs disagree; keep primary 試作MODEL-B and its meaningful 2L.
            ("529-0314", "試作MODEL-C 2L", "1", "634", True),
        ))
        original = receipt()
        before = deepcopy(original)
        expected = deepcopy(original)
        expected["line_items"][0]["description"] = "試作SAMPLE-A 2025"
        expected["line_items"][1]["description"] = "試作MODEL-B 2L"
        output, metadata = apply_supplemental_ocr_evidence(original, source, primary_text=PRIMARY)
        assert output == expected and original == before
        assert metadata["accepted_fields"] == ["line_items"]
        assert apply_supplemental_ocr_evidence(output, source, primary_text=PRIMARY)[0] == output

        rejected = (
            evidence((("418-0271", "試作SAMPLE-A 2025", "1", "512", True),
                      ("418-0271", "試作SAMPLE-A 2025", "1", "512", True))),
            evidence((("418-0271", "試作SAMPLE-A 2025", "1", "512", True),
                      ("529-0314", "試作MODEL-C 2L", "2", "634", True))),
            evidence((("418-0271", "試作SAMPLE-A 2025", "1", "512", True),
                      ("529-0314", "試作MODEL-C 2L", "1", "635", True))),
            evidence((("418-0271", "試作SAMPLE-A 2025", "1", "512", True),
                      ("529-0314", "試作MODEL-C 3L", "1", "634", True))),
            evidence((("418-0271", "試作SAMPLE-A 2025", "1", "512", True),
                      ("529-0314", "試作MODEL-C 2L", "1", "634", False))),
            evidence((("418-0271", "試作SAMPLE-A 2025", "1", "512", True),)),
        )
        for bad in rejected:
            unchanged(receipt(), bad, PRIMARY)
        unchanged(receipt(), evidence((
            ("418-0271", "SAMPLE-A 2025", "1", "512", True),
            ("529-0314", "MODEL-C 2L", "1", "634", True),
        )), PRIMARY)
        unchanged(receipt(), evidence((
            ("418-0271", "試作SAMPLE-A 2025", "1", "512", True),
            ("529-0314", "試作MODEL-C 2L", "1", "634", True),
        ), discount=True), PRIMARY)
        for field, value in (("discount", 10), ("discount_rate", "10%"), ("gross", 522)):
            candidate = receipt()
            candidate["line_items"][0][field] = value
            unchanged(candidate, source, PRIMARY)
        for primary in (
            PRIMARY.replace("点数\n2点", "529-0314 試作MODEL-B 2L\n1 ¥634\n点数\n2点"),
            PRIMARY.replace("点数\n2点", "点数\n3点"),
            PRIMARY + "\n点数\n2点",
            PRIMARY.replace("¥1,146", "¥1,147"),
            PRIMARY + "\n小計\n¥1,146",
        ):
            unchanged(receipt(), source, primary)
        duplicate = receipt()
        duplicate["line_items"][1] = deepcopy(duplicate["line_items"][0])
        unchanged(duplicate, source, PRIMARY)

        no_layout = receipt()
        _clean_code_prefixed_item_descriptions(no_layout, PRIMARY)
        assert no_layout == receipt()
        traced, trace = receipt(), []
        pipeline._apply_final_supplemental_ocr_evidence(traced, source, mutation_trace=trace, primary_text=PRIMARY)
        assert {key: value for key, value in traced.items() if not key.startswith("_")} == expected
        assert len(trace) == 1
        assert trace[0]["owner_phase"] == "supplemental_ocr_field_recovery"
        assert set(trace[0]["changes"]) == {"line_items"}

    main()


def test_single_marked_pos_code_requires_unique_one_row_ownership():
    from copy import deepcopy
    from receipt_parser.ocr import blocks_to_structured_text
    from receipt_parser.receipt_item_repair import _clean_code_prefixed_item_descriptions
    from receipt_parser.receipt_supplemental_ocr import (
        SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT,
        build_supplemental_ocr_evidence,
    )
    from receipt_parser.schema import Receipt

    code, marker, title, amount = "650321", "★", "試作モデル2025", "198"
    image_key, image_shape = "d" * 32, [240, 420]
    primary = f"{code}{marker}{title} ¥{amount}\n点数\n1点\n小計\n¥{amount}"

    def word(text, x, y, width=32):
        return {"text": text, "confidence": .99, "x": x, "y": y,
                "bbox": [[x, y], [x + width, y], [x + width, y + 18], [x, y + 18]], "page": 0}

    def layout(*, duplicate=False, duplicate_title=False, quantity=None, discount=False,
               marked_title=True, printed_amount=amount, header=False, extra_marker=None, competing=False):
        row = [word(code, 10, 40)]
        if extra_marker:
            row.append(word(extra_marker, 42, 40, 8))
        row.append(word(marker, 50, 40, 10))
        row.append(word(title if marked_title else f"{code}{marker}{title}", 70, 40, 110))
        if quantity is not None:
            row.append(word(quantity, 200, 40, 30))
        row.extend((word("¥", 250, 40, 12), word(printed_amount, 265, 40, 40)))
        blocks = list(row)
        if header:
            blocks.extend((word("毎月", 10, 0), word("5", 140, 0, 15), word("%OFF", 160, 0, 50),
                           word("ポイント", 10, 20, 80), word("3", 140, 20, 15)))
        if duplicate:
            blocks.extend(word(block["text"], block["x"], 60, block["bbox"][1][0] - block["x"])
                          for block in row)
        if duplicate_title:
            blocks.append(word(title, 10, 60, 110))
        if competing:
            blocks.extend((word("700001", 10, 60), word("別商品", 70, 60, 110),
                           word("¥", 250, 60, 12), word("50", 265, 60, 40)))
        if discount:
            blocks.extend((word("割引", 10, 55, 40), word("-¥10", 250, 55, 40)))
        blocks.extend((word("点数", 10, 80, 35), word("1点", 265, 80, 30),
                       word("小計", 10, 110, 35), word("¥", 250, 110, 12),
                       word(amount, 265, 110, 40)))
        return blocks

    def source(blocks):
        text = blocks_to_structured_text(blocks, sort_within_rows=True)
        return build_supplemental_ocr_evidence(
            image_key=image_key, image_shape=image_shape,
            strategy_fingerprint=SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT,
            merged_text=text, source_layout=blocks, tile_texts=(text, text),
        )

    def receipt(description=None, *, qty=1, unit=198, total=198):
        item = {"description": description or f"{code}{marker}{title}", "qty": qty,
                "unit_price": unit, "total": total, "discount": 0,
                "discount_rate": "", "tax_category": "10%"}
        return Receipt(merchant="SAMPLE", subtotal=198, total=198,
                       line_items=[item]).model_dump()

    layout_blocks = layout(header=True, extra_marker="※")
    original = receipt()
    before = deepcopy(original)
    _clean_code_prefixed_item_descriptions(original, primary, layout_blocks)
    assert original["line_items"][0]["description"] == title
    assert {key: value for key, value in original["line_items"][0].items() if key != "description"} == {
        key: value for key, value in before["line_items"][0].items() if key != "description"
    }
    _clean_code_prefixed_item_descriptions(original, primary, layout_blocks)
    assert original == {**before, "line_items": [dict(before["line_items"][0], description=title)]}

    no_layout = receipt()
    _clean_code_prefixed_item_descriptions(no_layout, primary)
    assert no_layout == before

    evidence = source(layout_blocks)
    traced = receipt()
    trace = []
    pipeline._apply_final_supplemental_ocr_evidence(
        traced, evidence, mutation_trace=trace, primary_text=primary,
    )
    assert traced["line_items"][0]["description"] == title
    assert {key: value for key, value in traced.items() if not key.startswith("_")} == {
        key: value for key, value in original.items() if not key.startswith("_")
    }
    assert len(trace) == 1 and set(trace[0]["changes"]) == {"line_items"}

    rejected = (
        (receipt(), layout(competing=True), primary),
        (receipt(), layout(), primary.replace("\n点数", "\n700001別商品 ¥50\n点数")),
        (receipt(), layout(extra_marker="?"), primary),
        (receipt(description=f"{code}{marker}{title}"), layout(marked_title=False), primary),
        (receipt(), layout(duplicate=True), primary + f"\n{code}{marker}{title} ¥{amount}"),
        (receipt(), layout(duplicate_title=True), primary),
        (receipt(), layout(discount=True), primary + "\n割引 -¥10"),
        (receipt(), layout(quantity="2個"), f"{code}{marker}{title} 2個 ¥{amount}\n点数\n1点\n小計\n¥{amount}"),
        (receipt(), layout(), primary + "\n点数\n2点"),
        (receipt(), layout(), primary.replace(f"小計\n¥{amount}", "小計\n¥199")),
        (receipt(), layout(printed_amount="19,"), primary),
        (receipt(qty=2, unit=99, total=198), layout(), primary),
        (receipt(unit=199, total=199), layout(), primary),
    )
    for candidate, blocks, text in rejected:
        before = deepcopy(candidate)
        _clean_code_prefixed_item_descriptions(candidate, text, blocks)
        assert candidate == before

    year_title = "2025★試作モデル"
    year = receipt(description=year_title)
    before = deepcopy(year)
    _clean_code_prefixed_item_descriptions(
        year, f"{year_title} ¥{amount}\n点数\n1点\n小計\n¥{amount}",
        [word(year_title, 10, 10, 120), word("¥", 250, 10, 12), word(amount, 265, 10, 40),
         word("点数", 10, 80, 35), word("1点", 265, 80, 30), word("小計", 10, 110, 35),
         word("¥", 250, 110, 12), word(amount, 265, 110, 40)],
    )
    assert year == before

    nonfinite = receipt()
    nonfinite_description = nonfinite["line_items"][0]["description"]
    nonfinite["line_items"][0]["qty"] = float("nan")
    _clean_code_prefixed_item_descriptions(nonfinite, primary, layout_blocks)
    assert nonfinite["line_items"][0]["description"] == nonfinite_description
    assert nonfinite["line_items"][0]["qty"] != nonfinite["line_items"][0]["qty"]


def test_closed_cash_table_requires_complete_independent_literal_owners():
    from copy import deepcopy
    import numpy as np
    from receipt_parser import pipeline
    from receipt_parser.ocr import _ocr_cache_key, blocks_to_structured_text
    from receipt_parser.receipt_supplemental_ocr import (
        SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT, build_supplemental_ocr_evidence,
        _build_financial_source_identity as build_financial_source_identity,
        _recover_closed_cash_literal_table as recover_closed_cash_literal_table,
        apply_supplemental_ocr_evidence,
    )
    from receipt_parser.receipt_projection import _layout_row_price_candidates

    ORIGINAL = np.full((500, 600, 3), 255, dtype=np.uint8)


    def word(text, x, y, width=32, height=16):
        return {"text": text, "confidence": .99, "x": x, "y": y,
                "bbox": [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
                "page": 0}


    def rows():
        blocks = [word("領収証", 10, 60, width=100)]
        for title, amount, y in (("試作甲", "241", 100), ("試作乙", "305", 140)):
            blocks.extend((word(title, 10, y, width=100), word("¥", 300, y, width=12),
                           word(amount, 314, y, width=50)))
        for label, amount, y in (("合計", "54.6", 180), ("8%対象", "546", 220),
                                 ("内消費税等", "40", 260), ("お預り", "600", 300),
                                 ("釣", "54", 340)):
            blocks.extend((word(label, 10, y, width=110), word("¥", 300, y, width=12),
                           word(amount, 314, y, width=50)))
        return blocks


    def sealed(blocks):
        blocks = sorted(blocks, key=lambda block: (block["page"], block["y"], block["x"]))
        text = blocks_to_structured_text(blocks)
        return build_supplemental_ocr_evidence(
            image_key=_ocr_cache_key(ORIGINAL), image_shape=[500, 600],
            strategy_fingerprint=SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT,
            merged_text=text, source_layout=blocks, tile_texts=[text, text],
        )


    def receipt():
        return {"document_type": "receipt", "currency": "JPY", "merchant": "SHOP",
                "total": 241, "subtotal": 201, "amount_paid": 241, "points_used": None,
                "payment_method": "cash", "taxes": [{"rate": "8%", "label": "内税", "amount": 40}],
                "line_items": [{"description": "試作甲", "qty": 1, "unit_price": 241,
                                "total": 241, "tax_category": "8%", "discount": 0, "discount_rate": ""}]}


    PRIMARY = "領収証\n試作甲 ¥241\n試作乙\n合計\n8%対象\n内消費税等\nお預り\n釣\n¥305\n¥54.6\n¥546\n¥40\n¥600\n¥54"


    def recover(current, evidence, primary=PRIMARY):
        context = build_financial_source_identity(ORIGINAL, primary, evidence)
        return recover_closed_cash_literal_table(current, evidence, primary, expected_source_identity=context)


    def rejected(blocks=None, current=None, primary=PRIMARY):
        evidence = sealed(rows() if blocks is None else blocks)
        candidate = receipt() if current is None else current
        before = deepcopy((candidate, evidence))
        assert recover(candidate, evidence, primary) is None
        assert (candidate, evidence) == before


    def change_token(old, new):
        blocks = rows()
        target = next(block for block in blocks if block["text"] == old)
        target["text"] = new
        return blocks


    def check():
        # One shared source seam: a moderate positive gap scales to price glyphs.
        currency = [word("試作甲", 10, 100, width=100), word("¥", 300, 100, width=12),
                    word("241", 319, 100, width=50)]
        candidate = _layout_row_price_candidates(currency)[0]
        assert candidate["currency_owned"] and candidate["description"] == "試作甲"
        assert 7 > .25 * 16 and 7 <= .50 * 16
        distant = deepcopy(currency)
        distant[2] = word("241", 321, 100, width=50)
        assert not _layout_row_price_candidates(distant)[0]["currency_owned"]
        tall_title = deepcopy(distant)
        tall_title[0] = word("試作甲", 10, 58, width=100, height=100)
        assert not _layout_row_price_candidates(tall_title)[0]["currency_owned"]
        for altered in (
            [currency[0], word("甲¥", 300, 100, width=12), currency[2]],
            currency + [word("250", 370, 100, width=50)],
            currency[:2] + [word("規格", 314, 100, width=2), currency[2]],
            [currency[0], word("¥", 300, 140, width=12), currency[2]],
            [currency[0], word("¥", 300, 100, width=50), currency[2]],
        ):
            assert not _layout_row_price_candidates(altered)[0]["currency_owned"]
        missing_box = deepcopy(currency)
        missing_box[1].pop("bbox")
        assert not _layout_row_price_candidates(missing_box)[0]["currency_owned"]

        current, evidence = receipt(), sealed(rows())
        before = deepcopy((current, evidence))
        proposed, proof = recover(current, evidence)
        assert [item["total"] for item in proposed["line_items"]] == [241, 305]
        assert proposed["total"] == 546 and proposed["subtotal"] == 506
        assert proposed["taxes"] == [{"rate": "8%", "label": "内税", "amount": 40}]
        assert proposed["amount_paid"] == 546 and proposed["payment_method"] == "cash"
        assert proof["printed_target"] == proof["literal_item_sum"] == 546
        assert proof["malformed_total_row"] == "合計¥54.6"
        assert (current, evidence) == before
        spaced = rows()
        price = next(block for block in spaced if block["text"] == "305")
        price.update(x=319, bbox=[[319, 140], [369, 140], [369, 156], [319, 156]])
        assert len(recover(receipt(), sealed(spaced))[0]["line_items"]) == 2
        rejected(spaced + [word("¥", 298, 140, width=2)])

        # Generic repeated/overlapping label-token artifact; preserve every raw word.
        duplicate = [block for block in rows() if block["text"] != "お預り"]
        duplicate.extend((word("お", 10, 300, width=20), word("預り", 45, 300, width=55),
                          word("おお", 9, 300, width=20, height=32)))
        source = sealed(duplicate)
        snapshot = deepcopy(source)
        _, duplicate_proof = recover(receipt(), source)
        assert len(duplicate_proof["overlapping_repeated_indices"]) == 1
        assert source == snapshot
        unrelated = deepcopy(duplicate)
        next(block for block in unrelated if block["text"] == "おお")["text"] = "別名"
        rejected(unrelated)
        disjoint = deepcopy(duplicate)
        ghost = next(block for block in disjoint if block["text"] == "おお")
        ghost.update(x=150, bbox=[[150, 300], [170, 300], [170, 332], [150, 332]])
        rejected(disjoint)

        for old, new in (("600", "601"), ("54", "53"), ("546", "545"),
                         ("40", "44"), ("8%対象", "10%対象"),
                         ("内消費税等", "消費税等"), ("305", "30.5"),
                         ("54.6", "546"), ("54.6", "54.60円")):
            rejected(change_token(old, new))
        rejected([block for block in rows() if block["text"] != "釣"])
        rejected(rows() + [word("VISA ¥546", 10, 390, width=130)])
        rejected(rows() + [word("8%対象¥546", 10, 390, width=130)])
        rejected(rows() + [word("合計¥999", 10, 390, width=130)])
        rejected(rows() + [word("お預り¥600", 10, 390, width=130)])
        rejected(rows() + [word("未所有名", 10, 160, width=130)])
        rejected(primary=PRIMARY.replace("試作乙", "試作丙"))
        rejected(primary=PRIMARY.replace("試作乙", "試作乙\n2個×305"))
        rejected(primary=PRIMARY.replace("試作甲 ¥241", "試作甲 ¥999"))
        clipped = rows()
        price = next(block for block in clipped if block["text"] == "600")
        price.update(x=550, bbox=[[550, 300], [600, 300], [600, 316], [550, 316]])
        rejected(clipped)
        for field, value in (("currency", "USD"), ("points_used", 1), ("amount_paid", 240),
                             ("payment_method", "credit")):
            altered = receipt()
            altered[field] = value
            rejected(current=altered)
        altered = receipt()
        altered["line_items"][0]["qty"] = 2
        rejected(current=altered)
        altered = receipt()
        altered["line_items"].append(deepcopy(altered["line_items"][0]))
        rejected(current=altered)
        invalid = sealed(rows())
        invalid["answer"] = {"total": 546}
        assert recover(receipt(), invalid) is None
        context = build_financial_source_identity(ORIGINAL, PRIMARY, evidence)
        assert recover_closed_cash_literal_table(receipt(), evidence, PRIMARY) is None
        for field, value in (("image_key", "b" * 32), ("primary_text_sha256", "c" * 64),
                             ("source_layout_sha256", "d" * 64), ("image_shape", [500, 599]),
                             ("strategy_fingerprint", "e" * 64)):
            altered = dict(context, **{field: value})
            assert recover_closed_cash_literal_table(receipt(), evidence, PRIMARY, expected_source_identity=altered) is None
        assert build_financial_source_identity(ORIGINAL[:, :-1], PRIMARY, evidence) is None
        # The existing rate validator admits both modes for small values: abstain.
        ambiguous = rows()
        replacements = {"241": "50", "305": "50", "546": "100", "40": "8",
                        "600": "120", "54": "20", "54.6": "10.0"}
        for block in ambiguous:
            block["text"] = replacements.get(block["text"], block["text"])
        altered = receipt()
        altered.update(total=50, subtotal=42, amount_paid=50)
        altered["line_items"][0].update(unit_price=50, total=50)
        altered["taxes"][0]["amount"] = 8
        rejected(ambiguous, altered, PRIMARY.replace("¥241", "¥50"))


    check()
    current = pipeline.Receipt.model_validate(receipt()).model_dump()
    evidence = sealed(rows())
    context = build_financial_source_identity(ORIGINAL, PRIMARY, evidence)
    fields, _proof = recover(current, evidence)
    expected = pipeline.Receipt.model_validate(dict(current, **fields)).model_dump()
    proposed, metadata = apply_supplemental_ocr_evidence(
        current, evidence, primary_text=PRIMARY, financial_source_identity=context,
    )
    assert pipeline.Receipt.model_validate(proposed).model_dump() == expected
    assert set(metadata['accepted_fields']) == {'line_items', 'financial_summary'}
    assert metadata['financial_summary']['literal_item_sum'] == 546
    assert apply_supplemental_ocr_evidence(
        proposed, evidence, primary_text=PRIMARY, financial_source_identity=context,
    )[0] == proposed
    snapshot = deepcopy(evidence)
    trace = []
    pipeline._apply_final_supplemental_ocr_evidence(
        current, evidence, mutation_trace=trace, primary_text=PRIMARY,
        financial_source_identity=context,
    )
    assert {key: value for key, value in current.items() if not key.startswith('_')} == expected
    assert evidence == snapshot
    assert len(trace) == 1
    assert trace[0]['owner_phase'] == 'supplemental_ocr_field_recovery'
    assert set(trace[0]['changes']) == {'line_items', 'subtotal', 'total', 'amount_paid'}
    for missing in (None, dict(context, image_key='b' * 32)):
        untouched = pipeline.Receipt.model_validate(receipt()).model_dump()
        before = deepcopy(untouched)
        rejected_trace = []
        pipeline._apply_final_supplemental_ocr_evidence(
            untouched, evidence, mutation_trace=rejected_trace, primary_text=PRIMARY,
            financial_source_identity=missing,
        )
        assert {key: value for key, value in untouched.items() if not key.startswith('_')} == before
        assert rejected_trace == []

def test_counted_literal_table_requires_joint_source_quantity_count_and_rate_proof():
    from copy import deepcopy
    from receipt_parser.receipt_supplemental_ocr import _recover_complete_literal_table
    def basket(*, fruit=81, fruit_marker='*', source_unit=46, primary_unit=46,
               count=6, source_container='保存容器K', primary_container='保存容器K'):
        prices = [7, 124, fruit, 138, 97]
        subtotal, standard = sum(prices), 131
        reduced = subtotal - standard
        reduced_tax, standard_tax = int(reduced * .08), 13
        total = subtotal + reduced_tax + standard_tax
        layout = []

        def word(text, x, y):
            width = max(9, len(text) * 12)
            layout.append(dict(text=text, confidence=.99, x=x, y=y, page=0,
                               bbox=[[x, y], [x + width, y], [x + width, y + 18], [x, y + 18]]))

        def row(parts, amount, marker, y):
            x = 30
            for part in parts:
                word(part, x, y)
                x += len(part) * 12 + 8
            word(str(amount), 600, y)
            if marker:
                word(marker, 650, y)

        row(['食品ポリ袋S'], 7, '除', 40)
        row([source_container], 124, '', 80)
        row(['南山柚子', 'ゼリー'], fruit, fruit_marker, 120)
        row(['蒼空クラッカー'], 138, '*', 160)
        word(f'(3個 X 単{source_unit})', 35, 200)
        row(['白桃プリン'], 97, '*', 240)
        for label, amount, y in [('小計', subtotal, 280), ('外税8%対象額', reduced, 320),
                                 ('外税8%', reduced_tax, 360), ('外税10%対象額', standard, 400),
                                 ('外税10%', standard_tax, 440), ('合計', total, 480)]:
            word(label, 30, y)
            word('¥', 580, y)
            word(str(amount), 600, y)
        word(f'お買上商品数:{count}', 30, 520)
        word('*印は軽減税率8%対象商品', 30, 560)
        primary = '\n'.join([
            '食品ポリ袋S 7除', f'{primary_container} 124', '南山柚子', f'{fruit}{fruit_marker}',
            '蒼空クラッカー', '138*', f'(3個 X 単{primary_unit})', '白桃プリン', '97*',
            '小計', f'¥{subtotal}', '外税8%対象額', f'¥{reduced}', '外税8%', f'¥{reduced_tax}',
            '外税10%対象額', f'¥{standard}', '外税10%', f'¥{standard_tax}', '合計', f'¥{total}',
            f'お買上商品数:{count}', '*印は軽減税率8%対象商品',
        ])
        receipt = {'subtotal': subtotal, 'total': total, 'payment_method': 'credit',
                   'taxes': [{'rate': '8%', 'label': '外税', 'amount': reduced_tax},
                             {'rate': '10%', 'label': '外税', 'amount': standard_tax}],
                   'line_items': [
                       {'description': primary_container, 'qty': 1, 'unit_price': 124, 'total': 124,
                        'tax_category': '0%', 'discount': 0, 'discount_rate': ''},
                       {'description': '蒼空クラッカー', 'qty': 3, 'unit_price': 46, 'total': 138,
                        'tax_category': '0%', 'discount': 0, 'discount_rate': ''},
                       {'description': '白桃プリン', 'qty': 1, 'unit_price': 97, 'total': 97,
                        'tax_category': '0%', 'discount': 0, 'discount_rate': ''},
                   ]}
        evidence = {'source_layout': layout, 'merged_text': 'Ignored legacy flattening; native source owns rows.'}
        return receipt, evidence, primary


    def check():
        receipt, evidence, primary = basket()
        frozen = deepcopy((receipt, evidence))
        recovered = _recover_complete_literal_table(receipt, evidence, primary)
        assert recovered is not None and len(recovered) == 5
        assert sum(item['total'] for item in recovered) == receipt['subtotal']
        assert [item['tax_category'] for item in recovered] == ['10%', '10%', '8%', '8%', '8%']
        assert recovered[2]['description'] == '南山柚子ゼリー'
        assert (recovered[3]['qty'], recovered[3]['unit_price'], recovered[3]['total']) == (3, 46, 138)
        assert all(item['discount'] == 0 and item['discount_rate'] == '' for item in recovered)
        assert (receipt, evidence) == frozen

        def reject(sample):
            inputs = deepcopy(sample)
            assert _recover_complete_literal_table(*sample) is None
            assert sample == inputs

        reject(basket(source_unit=47))             # Qty product is not its literal extended price.
        reject(basket(primary_unit=47))            # Primary and source packets disagree.
        reject(basket(count=4))                    # Printed rows are not purchased units.
        reject(basket(source_container='保存容器K21', primary_container='保存容器K20'))
        reject(basket(fruit=124, fruit_marker=''))  # Two full standard partitions survive.
        locked = basket()
        locked[0]['line_items'][1]['_tax_category_locked'] = '10%'
        reject(locked)
        exempt = basket()
        reject((exempt[0], exempt[1], exempt[2] + '\n免税'))
        duplicate = basket()
        reject((duplicate[0], duplicate[1], duplicate[2].replace('南山柚子\n', '南山柚子\n南山柚子\n')))
        partial_word = basket()
        reject((partial_word[0], partial_word[1], partial_word[2].replace('南山柚子\n', '南山柚\n')))
        missing_literal = basket()
        missing_literal[1]['source_layout'] = [word for word in missing_literal[1]['source_layout']
                                             if not (word['text'] == '138' and word['x'] == 600)]
        reject(missing_literal)


    check()


@pytest.mark.parametrize("case", [
    "overage", "balanced_swap", "no_context", "stale_text", "answer_field",
    "foreign_context", "outside_crop", "wrong_selected_row", "conflicting_native_unit",
    "missing_native_next_title", "missing_primary_neighbor_qty",
    "conflicting_gray_neighbor_unit", "duplicate_primary_title",
    "different_title_digit", "extra_native_money", "existing_discount",
    "extra_gray_title_money", "different_gray_title_digit", "native_money_in_padding",
    "overlapping_repeat_columns", "incorrect_printed_count", "duplicate_source_total",
    "unknown_gray_body_row", "existing_gross",
])
def test_bound_native_owner_and_gray_neighbors_require_one_complete_literal_basket(case):
    image = np.full((500, 900, 3), 255, dtype=np.uint8)
    titles, values = ("試作甲型7", "試作乙型8", "試作丙型9"), (147, 275, 604)
    primary = ("\n架空店\n試作甲型7\n147 1点\n147\n試作乙型8\n1275 1点\n"
               "試作丙型9\n275\n@604 1点\n604\nお買上点数\n合計\n3点\n¥1,026\n\n")

    def word(text, x, y, *, confidence=.9):
        return _word(text, x=x, y=y, size=16, confidence=confidence)

    gray = [word("領収証", 10, 20), word(titles[0], 10, 50),
            word("147", 300, 82), word("1", 500, 82), word("点", 530, 82), word("147", 700, 82),
            word(titles[1], 10, 115), word("型", 14, 115),
            word("1275", 300, 148), word("1", 500, 148), word("点", 530, 148), word("275", 700, 148),
            word(titles[2], 10, 165), word("型", 14, 165),
            word("@", 270, 198), word("604", 300, 198), word("604", 700, 198),
            word("お買上点数", 10, 230), word("合計", 10, 263), word("¥1,026", 700, 263)]
    if case == "extra_gray_title_money":
        gray.insert(14, word("33", 700, 165))
    if case == "different_gray_title_digit":
        gray[12]["text"] = "試作丙型90"
    if case == "overlapping_repeat_columns":
        gray[16] = word("604", 308, 198)
    if case == "duplicate_source_total":
        gray += [word("合計", 10, 296), word("¥1,026", 700, 296)]
    if case == "unknown_gray_body_row":
        gray.append(word("別の試作品", 10, 181))
    if case == "incorrect_printed_count":
        primary = primary.replace("3点", "4点")
    if case == "conflicting_gray_neighbor_unit":
        gray[15]["text"] = "1604"
    if case == "missing_primary_neighbor_qty":
        primary = primary.replace("@604 1点", "604")
    if case == "duplicate_primary_title":
        primary = primary.replace("試作丙型9\n", "試作丙型9\n試作丙型9\n")
    text = ocr.blocks_to_structured_text(gray)
    source = supplemental.build_supplemental_ocr_evidence(
        image_key=ocr._ocr_cache_key(image), image_shape=list(image.shape[:2]),
        strategy_fingerprint=supplemental.SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT,
        merged_text=text, source_layout=gray, tile_texts=[text, text],
    )
    # The expectation is built from the original image/text/gray source first.
    context = supplemental.build_native_owner_crop_context(
        image, primary, supplemental_ocr_evidence=source,
    )
    if case in ("overage", "balanced_swap"):
        assert context is not None

    native = [word(titles[1], 10, 115, confidence=0.0),
              word("0275", 300, 148, confidence=0.0), word("1", 500, 148, confidence=0.0),
              word("点", 530, 148, confidence=0.0), word("275", 700, 148, confidence=0.0),
              word(titles[2], 10, 165, confidence=0.0)]
    if case == "conflicting_native_unit":
        native[1]["text"] = "1275"
    if case == "missing_native_next_title":
        native.pop()
    if case == "extra_native_money":
        native.append(word("33", 760, 148, confidence=0.0))
    if case == "native_money_in_padding":
        native.append(word("33", 760, 100, confidence=0.0))
    if case == "outside_crop" and context is not None:
        native[0] = word(titles[1], 10, context["roi"][3] + 1, confidence=0.0)
    if case == "wrong_selected_row":
        for block in native[:5]:
            block["y"] -= 16
            for point in block["bbox"]:
                point[1] -= 16
    capture = dict(context=deepcopy(context), vision_text="試作乙型8\n0275 1点\n試作丙型9\n275",
                   source_layout=native, merged_text=ocr.blocks_to_structured_text(native))
    if case == "answer_field":
        capture["proposed_prices"] = []
    if case == "foreign_context" and context is not None:
        capture["context"]["source_layout_sha256"] = "f" * 64
    expected_context = None if case == "no_context" else context
    if case == "stale_text":
        primary += "\n"

    original = dict(document_type="receipt", currency="JPY", merchant="架空店", subtotal=950,
                    total=1026, taxes=[dict(rate="8%", amount=76)], amount_paid=1026,
                    payment_method="credit", line_items=[
        dict(description=title, qty=1, unit_price=value, total=value, tax_category="8%",
             discount=0, discount_rate="") for title, value in zip(titles, (147, 1275, 604), strict=True)
    ])
    if case == "balanced_swap":
        for item, value in zip(original["line_items"], (147, 604, 275), strict=True):
            item["unit_price"] = item["total"] = value
    if case == "different_title_digit":
        original["line_items"][2]["description"] = "試作丙型90"
    if case == "existing_discount":
        original["line_items"][1]["discount"] = 5
    if case == "existing_gross":
        original["line_items"][1]["gross"] = 1275
    before_receipt, before_source, before_capture = deepcopy(original), deepcopy(source), deepcopy(capture)
    output, metadata = supplemental.apply_supplemental_ocr_evidence(
        original, source, primary_text=primary,
        native_owner_ocr_evidence=capture, native_owner_context=expected_context,
    )
    expected = deepcopy(original)
    accepted = case in ("overage", "balanced_swap")
    if accepted:
        for item, value in zip(expected["line_items"], values, strict=True):
            item["unit_price"] = item["total"] = value
        assert sum(item["total"] for item in expected["line_items"]) == expected["total"]
    assert output == expected
    assert ("line_items" in metadata["accepted_fields"]) == accepted
    assert original == before_receipt and source == before_source and capture == before_capture
    assert all(word["confidence"] == 0.0 for word in capture["source_layout"])
    if accepted:
        assert metadata["native_owner"]["mode"] == "native_owner_literal_basket"
        assert [owner["value"] for owner in metadata["native_owner"]["owners"]] == list(values)
    assert supplemental.apply_supplemental_ocr_evidence(
        output, source, primary_text=primary,
        native_owner_ocr_evidence=capture, native_owner_context=expected_context,
    )[0] == output



def test_text_owner_transport_is_one_strict_feature_and_preserves_unscored_words(monkeypatch, tmp_path):
    image = np.full((100, 800, 3), [70, 120, 200], dtype=np.uint8)
    primary = "架空店\n試作乙型8\n1275 1点\n275\n合計\n275\n"
    gray = [_word("試作乙型8", x=10, y=20, size=12), _word("1275", x=300, y=44, size=12),
            _word("1", x=500, y=44, size=12), _word("点", x=530, y=44, size=12),
            _word("275", x=700, y=44, size=12)]
    text = ocr.blocks_to_structured_text(gray)
    source = supplemental.build_supplemental_ocr_evidence(
        image_key=ocr._ocr_cache_key(image), image_shape=list(image.shape[:2]),
        strategy_fingerprint=supplemental.SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT,
        merged_text=text, source_layout=gray, tile_texts=[text, text])
    context = supplemental.build_native_owner_crop_context(image, primary, supplemental_ocr_evidence=source)
    assert context is not None
    assert context["strategy_fingerprint"] == supplemental.NATIVE_OWNER_CROP_OCR_FINGERPRINT
    assert context["input_text_sha256"] == hashlib.sha256(normalize_fullwidth(primary).encode()).hexdigest()
    native = deepcopy(gray)
    native[1]["text"] = "0275"
    for word in native:
        word["confidence"] = 0.0
    record = dict(context=deepcopy(context), vision_text="試作乙型8\n0275 1点\n275",
                  source_layout=native, merged_text=ocr.blocks_to_structured_text(native))
    calls, initialized = _mock_vision(monkeypatch, texts=(record["vision_text"],))
    ordinary_call = supplemental._call_cloud_vision
    mapped = deepcopy(native)
    for word in mapped:
        word["bbox"] = [[x * 4, (y - context["roi"][1]) * 4] for x, y in word["bbox"]]
        word["x"], word["y"] = min(p[0] for p in word["bbox"]), min(p[1] for p in word["bbox"])

    def strict(image, client, *, allow_fallback, feature):
        assert feature == "TEXT_DETECTION" and allow_fallback is False
        assert image.shape[:2] == tuple(context["processed_shape"])
        assert image[0, 0].tolist() == [70, 120, 200]
        return ordinary_call(image, client)

    def words(_response, *, use_paragraph_confidence):
        assert use_paragraph_confidence is False
        return deepcopy(mapped)

    monkeypatch.setattr(supplemental, "_call_cloud_vision", strict)
    monkeypatch.setattr(supplemental, "_extract_words_from_response", words)
    for name in ("load_quantity_crop_ocr_evidence", "_atomic_write_text"):
        monkeypatch.setattr(supplemental, name, lambda *_a, **_k: pytest.fail("fresh touched disk"))
    assert supplemental.acquire_native_owner_crop_ocr_evidence(
        image, context, mode="fresh", cache_dir=tmp_path) == record
    assert len(calls) == 1 and initialized == [True] and list(tmp_path.iterdir()) == []
    assert supplemental._validated_quantity_crop(record, context) is None  # DOCUMENT cannot consume TEXT.
    vertex = [SimpleNamespace(x=x, y=y) for x, y in ((0, 0), (12, 0), (12, 12), (0, 12))]
    word = SimpleNamespace(confidence=0.0, symbols=[SimpleNamespace(text="試作")],
                           bounding_box=SimpleNamespace(vertices=vertex))
    paragraph = SimpleNamespace(confidence=.9, words=[word])
    response = SimpleNamespace(full_text_annotation=SimpleNamespace(pages=[SimpleNamespace(
        blocks=[SimpleNamespace(paragraphs=[paragraph])])]))
    assert ocr._extract_words_from_response(response, use_paragraph_confidence=False)[0]["confidence"] == 0.0
    assert ocr._extract_words_from_response(response)[0]["confidence"] == .9
    captured = []
    monkeypatch.setattr(pipeline, "check_model_available", lambda *_: None)
    monkeypatch.setattr(pipeline, "detect_document_type", lambda _: "receipt")
    monkeypatch.setattr(pipeline, "_run_extraction_pipeline", lambda **kwargs:
        (captured.append(kwargs) or {"_error": "stop before model"}, [], [], None))
    for name in ("load_image", "acquire_native_owner_crop_ocr_evidence", "build_native_owner_crop_context"):
        monkeypatch.setattr(pipeline, name, lambda *_a, **_k: pytest.fail("text injection acquired image/OCR/context"))
    pipeline.process_ocr_text(primary, apply_user_rules=False, supplemental_ocr_evidence=source,
        native_owner_ocr_evidence=record, native_owner_context=context)
    assert captured[0]["raw_text"] == captured[0]["payment_reference_text"] == normalize_fullwidth(primary)
    assert captured[0]["unified_text"] == strip_barcode_lines(normalize_fullwidth(primary))


def test_repeated_title_quantity_packet_and_money_guard():
    from receipt_parser.receipt_item_cleanup import _fix_misattributed_discounts
    from receipt_parser.receipt_marker_projection import _fix_qty_totals_from_ocr_unit_lines
    from receipt_parser.receipt_supplemental_ocr import _reconcile_quantity_row_text

    primary = "189*\n共通商品9\n共通商品9\n(51 X157)\n後続商品4\n別品5\n471*\n74*\n88*\n小計\n¥822"
    source = "共通商品9 189*\n共通商品9 471*\n(3個 X157)\n後続商品4 74*\n別品5 88*\n小計 ¥822"
    layout = [
        _word("共通商品9", x=10, y=10, size=12), _word("189*", x=300, y=10, size=12),
        _word("共通商品9", x=10, y=35, size=12), _word("471*", x=300, y=35, size=12),
        _word("(3個 X157)", x=20, y=60, size=12),
        _word("後続商品4", x=10, y=85, size=12), _word("74*", x=300, y=85, size=12),
        _word("別品5", x=10, y=110, size=12), _word("88*", x=300, y=110, size=12),
        _word("小計", x=10, y=135, size=12), _word("¥822", x=300, y=135, size=12),
    ]
    before = deepcopy(layout)
    changed, proposals = _reconcile_quantity_row_text(primary, source, layout)
    assert changed == primary.replace("(51 X157)", "(3個 X157)")
    assert len(proposals) == 1 and proposals[0]["total"] == 471
    assert layout == before
    assert _reconcile_quantity_row_text(changed, source, layout) == (changed, [])
    wide_digits = str.maketrans("0123456789", "０１２３４５６７８９")
    wide_layout = [dict(block, text=block["text"].translate(wide_digits)) for block in layout]
    assert _reconcile_quantity_row_text(primary, source.translate(wide_digits), wide_layout)[0] == changed

    def item(title, qty, unit, net):
        return dict(description=title, qty=qty, unit_price=unit, total=net,
                    tax_category="8%", discount=0, discount_rate="")

    rows = [item("共通商品9", 1, 189, 189), item("共通商品9", 51, 157, 471),
            item("後続商品4", 1, 74, 74), item("別品5", 1, 88, 88)]
    _fix_misattributed_discounts(rows)
    assert rows[1] == item("共通商品9", 51, 157, 471)
    receipt = dict(line_items=rows, subtotal=822, total=822)
    _fix_qty_totals_from_ocr_unit_lines(receipt, changed)
    assert receipt["line_items"] == [rows[0], item("共通商品9", 3, 157, 471), rows[2], rows[3]]
    assert len(receipt["line_items"]) == 4  # Separate repeated purchases survive.
    single = item("単品8", 1, 41, 39)
    _fix_misattributed_discounts([single])
    assert single == item("単品8", 1, 41, 41)

    for bad in (
        primary.replace("(51 X157)", "(51個 X157)"),  # Complete conflicting count.
        primary.replace("(51 X157)", "(3 X157)"),  # Already closing literal digits.
        primary.replace("471*", "472*"),
        primary.replace("471*", "471*\n471*"),  # Competing gross literals.
        primary.replace("共通商品9", "共通商品8"),  # Title digits never stripped.
        primary.replace("後続商品4", "割引 20%"),  # Native boundary disagrees.
        primary.replace("共通商品9\n共通商品9", "共通商品9\n(32 X157)\n共通商品9"),
    ):
        assert _reconcile_quantity_row_text(bad, source, layout) == (bad, [])
    bad_layout = deepcopy(layout)
    bad_layout[3]["text"] = "472*"
    assert _reconcile_quantity_row_text(primary, source.replace("471*", "472*"), bad_layout) == (primary, [])



def test_summary_only_titles_and_independent_literal_quantity_packets():
    """One invented contract: exact summary boundary, title ownership, quantity fields."""
    from receipt_parser.patterns import _OCR_ZONE_END_RE
    from receipt_parser.receipt_item_repair import _drop_banner_phantom_items
    from receipt_parser.receipt_projection import _layout_rows_are_local
    from receipt_parser.receipt_supplemental_ocr import _reconcile_quantity_row_text

    for heading in ('****まとめ値引き', '＊＊ まとめ 値引', '※ ※ まとめ 値引き'):
        assert _OCR_ZONE_END_RE.fullmatch(heading)
        assert _OCR_ZONE_END_RE.match(''.join(heading.split()))
    for title in ('飲料まとめ値引きセット', 'まとめ値引き 飲料', '**まとめ値引き ¥40', '*まとめ値引き',
                  'まとめ値引', 'まとめ値引き'):
        assert _OCR_ZONE_END_RE.search(title) is None

    body = '\n'.join(('星粒ビスケット9', '¥129', '星粒ビスケット9', '¥129', '青空せっけん5 ¥706'))
    zone = '\n'.join(('****まとめ値引き', 'Bケア用品 (2個)', '-40', '¥836から', '¥796に致します',
                      '星粒ビスケット9', '青空せっけん5', '小計', '¥1,760'))
    items = [dict(description=name, qty=1, unit_price=value, total=value, discount=0, discount_rate='')
             for name, value in [('星粒ビスケット9', 129), ('星粒ビスケット9', 129), ('青空せっけん5', 706),
                                 ('Bケア用品 (2個)', 836), ('Bケア用品 (2個)大', 977)]]
    before = deepcopy(items)
    _drop_banner_phantom_items(items, body + '\n' + zone)
    assert items == before[:3] + before[4:]
    # The literal inline currency suffix is not part of the bought title; its 5 is.
    bought_summary = deepcopy(before)
    _drop_banner_phantom_items(bought_summary, 'Bケア用品 (2個) ¥907\n' + body + '\n' + zone)
    assert bought_summary == before
    open_zone = deepcopy(before)
    _drop_banner_phantom_items(open_zone, body + '\n' + zone.split('\n小計')[0])
    assert open_zone == before
    duplicate_zone = deepcopy(before)
    _drop_banner_phantom_items(duplicate_zone, body + '\n' + zone + '\n' + zone)
    assert duplicate_zone == before

    primary = '\n'.join(('*こぐま豆菓子6', '3X1167', '¥501', '次品8', '¥83',
                         '内レジ袋小', '3コメ单7', '¥21', '****まとめ値引き',
                         'Bケア用品 (2個)', '-40', '¥836から', '¥796に致します', '小計', '¥605'))
    # The sealed flattened text has reordered detail cells; native geometry owns them.
    source = '\n'.join(('*こぐま豆菓子6', '¥501 X167 コ', '次品8 ¥83', '内レジ袋小',
                        '¥21 3コメ单7', '****まとめ値引き', 'Bケア用品 (2個)', '小計 ¥605'))
    layout = [
        _word('*こぐま豆菓子6', x=20, y=40, size=18),
        _word('コX167', x=140, y=75, size=18), _word('¥501', x=300, y=75, size=18),
        _word('次品8', x=20, y=110, size=18), _word('¥83', x=300, y=110, size=18),
        _word('内レジ袋小', x=20, y=145, size=18),
        _word('3コメ单7', x=140, y=180, size=18), _word('¥21', x=300, y=180, size=18),
        _word('****まとめ値引き', x=20, y=215, size=18),
        _word('Bケア用品 (2個)', x=20, y=250, size=18),
        _word('小計', x=20, y=285, size=18), _word('¥605', x=300, y=285, size=18),
    ]
    layout_before = deepcopy(layout)
    changed, proposals = _reconcile_quantity_row_text(primary, source, layout)
    assert changed == primary.replace('3X1167', '3コX単167').replace('3コメ单7', '3コX単7')
    assert [(row['qty'], row['unit_price'], row['total']) for row in sorted(proposals, key=lambda row: row['line_index'])] == [(3, 167, 501), (3, 7, 21)]
    assert all(row['source_fields'] == {'qty': 'primary_detail', 'unit_price': 'native_source_detail'} for row in proposals)
    assert layout == layout_before
    assert _reconcile_quantity_row_text(changed, source, layout) == (changed, [])

    def rejects_snack(candidate=primary, candidate_layout=layout):
        after, proof = _reconcile_quantity_row_text(candidate, source, candidate_layout)
        assert '3コX単167' not in after
        assert all(row['title'] != 'こぐま豆菓子6' for row in proof)

    for replacement in ('4X1167', 'X1167', '3コX単1167', '3X167'):
        rejects_snack(primary.replace('3X1167', replacement))
    rejects_snack(primary.replace('¥501', '¥502'))
    rejects_snack(primary.replace('¥501\n次品8', '¥501\n¥501\n次品8'))
    rejects_snack(primary.replace('次品8', '別品8'))
    rejects_snack(primary.replace('****まとめ値引き', '*こぐま豆菓子6\n****まとめ値引き'))
    for text in ('コX168', '2コX167', 'コX167 ¥501'):
        candidate_layout = deepcopy(layout)
        candidate_layout[1]['text'] = text
        rejects_snack(candidate_layout=candidate_layout)
    wrong_owner = deepcopy(layout)
    wrong_owner[0]['text'] = '*こぐま豆菓子7'
    rejects_snack(candidate_layout=wrong_owner)
    repeated_owner = deepcopy(layout) + [_word('*こぐま豆菓子6', x=20, y=197, size=18)]
    rejects_snack(candidate_layout=repeated_owner)
    distant_layout = deepcopy(layout)
    distant_layout[0] = _word('*こぐま豆菓子6', x=20, y=-50, size=18)
    rejects_snack(candidate_layout=distant_layout)
    wrong_page = deepcopy(layout)
    wrong_page[1]['page'] = wrong_page[2]['page'] = 1
    assert not _layout_rows_are_local([wrong_page[0]], wrong_page[1:3])
    rejects_snack(candidate_layout=wrong_page)
    duplicate_inline_source = deepcopy(layout) + [
        _word('*こぐま豆菓子6', x=20, y=197, size=18), _word('¥777', x=300, y=197, size=18)]
    rejects_snack(candidate_layout=duplicate_inline_source)
    rejects_snack(primary.replace('****まとめ値引き', '*こぐま豆菓子6 ¥777\n****まとめ値引き'))
    foreign_packet = deepcopy(layout)
    foreign_packet[0]['text'] = '*別の豆菓子6'
    rejects_snack(candidate_layout=foreign_packet)
    summary_packet = deepcopy(layout)
    summary_packet[0] = _word('*こぐま豆菓子6', x=20, y=320, size=18)
    summary_packet[1] = _word('コX167', x=140, y=355, size=18)
    summary_packet[2] = _word('¥501', x=300, y=355, size=18)
    rejects_snack(candidate_layout=summary_packet)
    wide_layout = deepcopy(layout)
    wide_layout[1]['text'], wide_layout[2]['text'] = 'コＸ１６７', '￥５０１'
    wide_changed, wide_proof = _reconcile_quantity_row_text(primary, source, wide_layout)
    assert wide_changed == changed and len(wide_proof) == 2
    conflicting_count = deepcopy(layout)
    conflicting_count[6]['text'] = '2コメ单7'
    after, proof = _reconcile_quantity_row_text(primary, source, conflicting_count)
    assert '3コメ单7' in after and all(row['title'] != '内レジ袋小' for row in proof)
    complete_pair = primary.replace('3コメ单7', '3コX単8')
    after, proof = _reconcile_quantity_row_text(complete_pair, source, layout)
    assert '3コX単8' in after and all(row['title'] != '内レジ袋小' for row in proof)
    wrong_boundary = primary.replace('****まとめ値引き', '合計')
    after, proof = _reconcile_quantity_row_text(wrong_boundary, source, layout)
    assert '3コメ单7' in after and all(row['title'] != '内レジ袋小' for row in proof)




def test_native_title_crop_requires_one_closed_geometry_owned_packet_and_is_atomic(monkeypatch):
    import sys
    import types
    from receipt_parser.ocr import blocks_to_structured_text

    image = np.zeros((500, 300, 3), dtype=np.uint8)
    shape = list(image.shape[:2])

    def word(text, x, y, *, confidence=0.9):
        return _word(text, x=x, y=y, size=9, confidence=confidence)

    layout = [
        word('登録番号T30120000001234', 10, 5), word('TEL0120-44-5500', 160, 5),
        word('8421', 10, 30), word('試作コップ', 50, 30),
        word('4901234567894', 10, 55), word('¥', 10, 80), word('240', 22, 80),
        word('3', 40, 80), word('個', 50, 80), word('¥', 62, 80), word('720', 74, 80),
        word('小計', 10, 105), word('3点', 35, 105), word('¥', 55, 105), word('720', 67, 105),
        word('(外税10.0%対象額¥720)', 10, 130), word('10.0%消費税等¥72', 10, 155),
        word('外税額計¥72', 10, 180), word('合計792', 10, 205),
        word('クレジット792', 10, 230), word('現計0', 10, 255),
        word('会員番号415P/748P', 10, 280), word('1234567890123456789012', 10, 305),
        word('ポイント対象金額', 10, 330), word('¥60', 10, 355),
    ]
    text = '\n'.join(('店舗', '98421 試作コップ', '4901234567894', '¥240 3個', '¥720',
                      '小計', '3点', '¥720', '(外税10.0%対象額¥720)', '10.0%消費税等¥72',
                      '外税額計¥72', '合計', '¥792', 'クレジット', '¥792', '現計', '¥0',
                      '会員番号415P/748P', '1234567890123456789012', 'ポイント対象金額', '¥60'))
    merged = blocks_to_structured_text(layout)
    source = supplemental.build_supplemental_ocr_evidence(
        image_key=ocr._ocr_cache_key(image), image_shape=shape,
        strategy_fingerprint=supplemental.SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT,
        merged_text=merged, source_layout=layout, tile_texts=(merged, merged),
    )
    context = supplemental.build_native_title_crop_context(image, text, supplemental_ocr_evidence=source)
    assert context is not None and context['roi'] == [0, 21, 300, 98]

    native_layout = [deepcopy(block) for block in layout
                     if block['text'] not in {'8421', '試作コップ'}
                     and min(point[1] for point in block['bbox']) >= context['roi'][1]
                     and max(point[1] for point in block['bbox']) <= context['roi'][3]]
    native_layout.extend((word('8421', 10, 30), word('Q', 30, 30),
                          word('試作コップ', 50, 30, confidence=None)))
    native_merged = blocks_to_structured_text(native_layout)
    native = dict(context=deepcopy(context), vision_text=native_merged,
                  source_layout=native_layout, merged_text=native_merged)
    receipt = {
        'merchant': '架空店', 'currency': 'JPY', 'subtotal': 720, 'total': 792,
        'amount_paid': 792, 'payment_method': 'credit',
        'taxes': [{'rate': '10%', 'label': '外税', 'amount': 72}],
        'line_items': [{'description': '試作コップ', 'qty': 3, 'unit_price': 240,
                        'total': 720, 'gross': 720, 'discount': 0,
                        'discount_rate': '', 'tax_category': '10%'}],
    }
    before = deepcopy(receipt)
    proposed, meta = apply_supplemental_ocr_evidence(
        receipt, source, primary_text=text, native_owner_ocr_evidence=native,
        native_owner_context=context,
    )
    expected = deepcopy(before)
    expected['line_items'][0]['description'] = 'Q試作コップ'
    assert proposed == expected and meta['accepted_fields'] == ['line_items']
    assert meta['native_title'] == {'accepted': True, 'title': 'Q試作コップ'}
    again, again_meta = apply_supplemental_ocr_evidence(
        proposed, source, primary_text=text, native_owner_ocr_evidence=native,
        native_owner_context=context,
    )
    assert again == expected and not again_meta['accepted_fields']

    primary_code_read = deepcopy(native)
    next(block for block in primary_code_read['source_layout'] if block['text'] == '8421')['text'] = '98421'
    primary_code_read['merged_text'] = blocks_to_structured_text(primary_code_read['source_layout'])
    primary_code_read['vision_text'] = primary_code_read['merged_text']
    assert supplemental._recover_native_title(receipt, supplemental._validated_evidence(source),
                                               text, primary_code_read, context) == 'Q試作コップ'
    wrong_code_read = deepcopy(primary_code_read)
    next(block for block in wrong_code_read['source_layout'] if block['text'] == '98421')['text'] = '77777'
    wrong_code_read['merged_text'] = blocks_to_structured_text(wrong_code_read['source_layout'])
    wrong_code_read['vision_text'] = wrong_code_read['merged_text']
    assert supplemental._recover_native_title(receipt, supplemental._validated_evidence(source),
                                               text, wrong_code_read, context) is None

    merged_native = deepcopy(native)
    merged_native['source_layout'] = [block for block in merged_native['source_layout']
                                     if block['text'] not in {'Q', '試作コップ'}]
    title_cell = word('Q試作コップ', 30, 30, confidence=None)
    title_cell['bbox'] = [[30, 30], [59, 30], [59, 39], [30, 39]]
    merged_native['source_layout'].append(title_cell)
    merged_native['merged_text'] = blocks_to_structured_text(merged_native['source_layout'])
    merged_native['vision_text'] = merged_native['merged_text']
    assert supplemental._recover_native_title(receipt, supplemental._validated_evidence(source),
                                               text, merged_native, context) == 'Q試作コップ'
    no_prefix_gap = deepcopy(merged_native)
    no_prefix_gap['source_layout'][-1] = word('Q試作コップ', 50, 30, confidence=None)
    no_prefix_gap['merged_text'] = blocks_to_structured_text(no_prefix_gap['source_layout'])
    no_prefix_gap['vision_text'] = no_prefix_gap['merged_text']
    assert supplemental._recover_native_title(receipt, supplemental._validated_evidence(source),
                                               text, no_prefix_gap, context) is None

    def rejects(candidate_layout=layout, candidate_text=text):
        assert supplemental._native_title_crop_candidate(candidate_text, candidate_layout, shape) is None

    rejects(candidate_text=text.replace('4901234567894', '4901234567895'))
    rejects(candidate_text=text.replace('3点', '2点'))
    rejects(candidate_layout=layout + [word('不明商品¥19', 10, 15)])
    rejects(candidate_layout=layout + [word('-19', 10, 15)])
    rejects(candidate_layout=[block for block in layout if block['text'] != '合計792'])
    rejects(candidate_layout=[block for block in layout if block['text'] != '(外税10.0%対象額¥720)']
             + [word('(10.0%対象額¥720)', 10, 130)])
    rejects(candidate_layout=[block for block in layout if block['text'] != '外税額計¥72']
             + [word('外税額計¥71', 10, 180)])
    rejects(candidate_layout=[block for block in layout if block['y'] != 80]
             + [word('¥2403個¥720', 10, 80)])
    rejects(candidate_layout=[block for block in layout if block['y'] != 80]
             + [word('¥', 10, 80), word('240', 22, 80), word('3', 40, 80), word('個', 50, 80),
                word('¥', 62, 80), word('720', 74, 80), word('2個', 10, 95)])
    duplicate_pos = layout + [word('9876', 10, 380), word('別試作品', 50, 380),
                              word('4000000000000', 10, 405), word('¥', 10, 430),
                              word('1', 22, 430), word('1', 40, 430), word('個', 50, 430),
                              word('¥', 62, 430), word('1', 74, 430)]
    rejects(candidate_layout=duplicate_pos)

    for field, value in (('currency', 'USD'), ('subtotal', 721)):
        wrong = deepcopy(receipt)
        wrong[field] = value
        assert supplemental._recover_native_title(wrong, supplemental._validated_evidence(source),
                                                   text, native, context) is None
    wrong_item = deepcopy(receipt)
    wrong_item['line_items'][0]['unit_price'] = 241
    assert supplemental._recover_native_title(wrong_item, supplemental._validated_evidence(source),
                                               text, native, context) is None
    wrong_item = deepcopy(receipt)
    wrong_item['line_items'][0]['discount'] = 1
    assert supplemental._recover_native_title(wrong_item, supplemental._validated_evidence(source),
                                               text, native, context) is None
    extra_title = deepcopy(native)
    extra_title['source_layout'].append(word('X', 90, 30))
    extra_title['merged_text'] = blocks_to_structured_text(extra_title['source_layout'])
    extra_title['vision_text'] = extra_title['merged_text']
    assert supplemental._recover_native_title(receipt, supplemental._validated_evidence(source),
                                               text, extra_title, context) is None

    vision_module = types.ModuleType('google.cloud.vision')
    class Feature:
        Type = types.SimpleNamespace(TEXT_DETECTION='text', DOCUMENT_TEXT_DETECTION='document')
        def __init__(self, *, type_, model): self.type_, self.model = type_, model
    vision_module.Feature = Feature
    vision_module.Image = lambda **kwargs: types.SimpleNamespace(**kwargs)
    vision_module.ImageContext = lambda **kwargs: types.SimpleNamespace(**kwargs)
    vision_module.AnnotateImageRequest = lambda **kwargs: types.SimpleNamespace(**kwargs)
    cloud_module = types.ModuleType('google.cloud')
    cloud_module.vision = vision_module
    google_module = types.ModuleType('google')
    google_module.cloud = cloud_module
    monkeypatch.setitem(sys.modules, 'google', google_module)
    monkeypatch.setitem(sys.modules, 'google.cloud', cloud_module)
    monkeypatch.setitem(sys.modules, 'google.cloud.vision', vision_module)
    monkeypatch.setattr(ocr, '_track_api_call', lambda: None)
    calls = []
    client = types.SimpleNamespace(annotate_image=lambda **kwargs: (
        calls.append(kwargs) or types.SimpleNamespace(error=types.SimpleNamespace(message=''))))
    ocr._call_cloud_vision(image, client, feature='TEXT_DETECTION', model='builtin/latest', allow_fallback=False)
    assert calls[-1]['retry'] is None
    request = calls[-1]['request']
    assert request.features[0].model == 'builtin/latest' and request.features[0].type_ == 'text'
    assert request.image_context.language_hints == ['ja', 'en']
    ocr._call_cloud_vision(image, client, allow_fallback=False)
    assert calls[-1]['request'].features[0].model == 'builtin/stable' and calls[-1]['retry'] is None

    class Proto:
        def __init__(self, present): self.present = present
        def HasField(self, name): return name == 'confidence' and self.present
    def proto_word(present):
        vertices = [types.SimpleNamespace(x=x, y=y) for x, y in ((1, 1), (8, 1), (8, 8), (1, 8))]
        return types.SimpleNamespace(symbols=[types.SimpleNamespace(text='語')], confidence=0.0,
            _pb=Proto(present), bounding_box=types.SimpleNamespace(vertices=vertices))
    paragraph = types.SimpleNamespace(words=[proto_word(True), proto_word(False)], confidence=0.9)
    response = types.SimpleNamespace(full_text_annotation=types.SimpleNamespace(
        pages=[types.SimpleNamespace(blocks=[types.SimpleNamespace(paragraphs=[paragraph])])]))
    extracted = ocr._extract_words_from_response(response, use_paragraph_confidence=False,
                                                   preserve_missing_confidence=True)
    assert [word['confidence'] for word in extracted] == [0.0, None]



def test_price_column_requires_unique_observed_whole_table_count_tax_and_exact_changed_owners():
    image = np.zeros((650, 400, 3), dtype=np.uint8)
    titles = ['青葉ドーナツ7', '光の茶葉9', '夜空豆6', 'レジ袋20号', '手作りせっけん4', '非課税ごみ袋8']
    observed = [147, 346, 53, 4, 162, 311]
    gray = [143, 346, 55, 4, 166, 311]
    header = '2024年08月21日09:37'
    layout, native, primary, y = [_word(header, x=10, y=4, size=12)], [], ['架空店', header], 40
    for i, (title, amount, old) in enumerate(zip(titles, observed, gray, strict=True)):
        layout.extend([_word(title, x=10, y=y, size=12), _word(str(old), x=300, y=y, size=12)])
        native.append(_word(str(amount), x=300, y=y, size=12, confidence=None))
        marker = '*' if i < 3 else '非' if i == 5 else ''
        if marker:
            layout.append(_word(marker, x=315, y=y, size=9))
            native.append(_word(marker, x=315, y=y, size=9, confidence=None))
        primary.extend([title, str(old) + marker])
        if i == 0:
            native.append(_word(str(old), x=300, y=y, size=12, confidence=0.99))
        if i == 1:
            y += 28
            layout.append(_word('2個X単173', x=50, y=y, size=12))
            primary.append('(2個X単173)')
        y += 40
    tail = ['小計¥1,023', '外税8%対象額¥546', '外税8%¥43', '外税10%対象額¥166',
            '外税10%¥16', '非課税対象額¥311', '合計¥1,082', 'お買上商品数:5', '※印は軽減税率8%対象商品']
    for text in tail:
        layout.append(_word(text, x=10, y=y, size=12))
        primary.append(text)
        y += 30
    primary = '\n'.join(primary)

    def seal(blocks):
        merged = ocr.blocks_to_structured_text(blocks)
        return build_supplemental_ocr_evidence(image_key=ocr._ocr_cache_key(image), image_shape=list(image.shape[:2]),
            strategy_fingerprint=supplemental.SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT,
            source_layout=blocks, merged_text=merged, tile_texts=(merged, merged))

    def bind(blocks=layout, words=native, text=primary):
        source = seal(blocks)
        context = supplemental.build_native_price_column_context(image, text, supplemental_ocr_evidence=source)
        assert context is not None
        merged = ocr.blocks_to_structured_text(words)
        capture = dict(context=deepcopy(context), vision_text=merged, source_layout=deepcopy(words), merged_text=merged)
        identity = supplemental._build_financial_source_identity(image, text, source)
        return source, capture, context, identity

    source, capture, context, identity = bind()
    assert 0 < context['roi'][0] < context['roi'][2] < image.shape[1]
    assert set(context) == supplemental._PRICE_COLUMN_CONTEXT_KEYS
    receipt = dict(document_type='receipt', currency='JPY', merchant='架空店', date='2024-08-21',
        subtotal=1023, total=1082, amount_paid=1082, payment_method='cash',
        taxes=[dict(rate='8%', label='外税', amount=43), dict(rate='10%', label='外税', amount=16)],
        line_items=[dict(description=title, qty=2 if i == 1 else 1, unit_price=173 if i == 1 else old,
                        total=old, discount=0, discount_rate='', tax_category='8%' if i < 3 else '10%')
                    for i, (title, old) in enumerate(zip(titles, gray, strict=True))])
    before, raw_before = deepcopy(receipt), deepcopy(source)
    proposed, proof = supplemental._recover_native_price_column(receipt, source, primary, capture, context, identity)
    assert [item['total'] for item in proposed] == observed
    assert [item['tax_category'] for item in proposed] == ['8%', '8%', '8%', '10%', '10%', '0%']
    assert proof['owners'][0]['observed_candidates'] == [143, 147] and proof['owners'][0]['selected'] == 147
    assert source == raw_before and receipt == before
    projected, metadata = apply_supplemental_ocr_evidence(receipt, source, primary_text=primary,
        financial_source_identity=identity, native_owner_ocr_evidence=capture, native_owner_context=context)
    assert projected == dict(receipt, line_items=proposed) and metadata['accepted_fields'] == ['line_items']
    assert metadata['line_items']['mode'] == 'native_price_column_complete_table'
    repeated, _ = apply_supplemental_ocr_evidence(projected, source, primary_text=primary,
        financial_source_identity=identity, native_owner_ocr_evidence=capture, native_owner_context=context)
    assert repeated == projected
    trace = []
    pipeline._apply_final_supplemental_ocr_evidence(receipt, source, mutation_trace=trace, primary_text=primary,
        financial_source_identity=identity, native_owner_ocr_evidence=capture, native_owner_context=context)
    assert {key: value for key, value in receipt.items() if not key.startswith('_')} == projected
    assert set(trace[-1]['changes']) == {'line_items'}

    # The same fully observed table can split a model's fused title; source
    # title inventory, not model labels, supplies the missing literal owner.
    fused = deepcopy(before)
    fused['line_items'][0]['description'] += titles[1]
    del fused['line_items'][1]
    repaired, _ = supplemental._recover_native_price_column(fused, source, primary, capture, context, identity)
    assert [item['description'] for item in repaired] == titles

    def reject(words=native, blocks=layout, text=primary, current=before):
        s, n, c, identifier = bind(blocks, words, text)
        assert supplemental._recover_native_price_column(current, s, text, n, c, identifier) == (None, {})

    reject([word for word in native if word['text'] != '147'])  # No residual-created price.
    ambiguous = deepcopy(native)
    ambiguous.extend([dict(deepcopy(native[0]), text='148'), dict(deepcopy(next(w for w in native if w['text'] == '53')), text='52')])
    reject(ambiguous)  # Two complete equal-subtotal/equal-tax baskets abstain.
    reject(text=primary.replace(titles[0], '青葉ドーナツ8'))
    reject(text=primary.replace('お買上商品数:5', 'お買上商品数:6'))
    reject(text=primary.replace('外税10%対象額¥166', '外税10%対象額¥167'))
    reject([word for word in native if word['text'] != '非'])
    unchanged_title = deepcopy(layout)
    next(word for word in unchanged_title if word['text'] == titles[1])['text'] = '光の茶葉れ9'
    s, n, c, identifier = bind(blocks=unchanged_title)
    repaired, _ = supplemental._recover_native_price_column(before, s, primary, n, c, identifier)
    assert repaired[1] == proposed[1]
    changed_title = deepcopy(layout)
    next(word for word in changed_title if word['text'] == titles[0])['text'] = '青葉ドーナツ影7'
    reject(blocks=changed_title)  # Approximate title correspondence cannot change a price.
    signed = deepcopy(layout) + [_word('-', x=288, y=40, size=9)]
    reject(blocks=signed)  # A canonical title key cannot swallow an inline negative cell.
    reject(words=deepcopy(native) + [_word('-', x=288, y=40, size=9, confidence=None)])
    wide_title = deepcopy(layout)
    title_word = next(word for word in wide_title if word['text'] == titles[0])
    title_word['bbox'][1][0] = title_word['bbox'][2][0] = 297
    for sign in ('-', '−¥', '¥−', '－￥', '￥－'):
        reject(blocks=wide_title, words=deepcopy(native) + [_word(sign, x=288, y=40, size=9, confidence=None)])
    negative_cell = deepcopy(native)
    negative_cell[0]['text'] = '-147'
    reject(words=negative_cell)
    hyphenated = deepcopy(layout)
    next(word for word in hyphenated if word['text'] == titles[0])['text'] = '青葉ドーナツ-7'
    hyphenated_current = deepcopy(before)
    hyphenated_current['line_items'][0]['description'] = '青葉ドーナツ-7'
    hyphenated_text = primary.replace(titles[0], '青葉ドーナツ-7')
    s, n, c, identifier = bind(blocks=hyphenated, text=hyphenated_text)
    repaired, _ = supplemental._recover_native_price_column(hyphenated_current, s, hyphenated_text, n, c, identifier)
    assert repaired[0]['description'] == '青葉ドーナツ-7' and repaired[0]['total'] == observed[0]
    s, n, c, identifier = bind(words=deepcopy(native) + [_word('-', x=316, y=40, size=6, confidence=None)])
    repaired, _ = supplemental._recover_native_price_column(before, s, primary, n, c, identifier)
    assert [item['total'] for item in repaired] == observed  # Trailing uncertainty stays trailing.
    priceless = deepcopy(layout) + [_word('追加商品8', x=10, y=20, size=12)]
    priceless_current = deepcopy(before)
    priceless_current['line_items'].insert(0, dict(description='追加商品8', qty=1, unit_price=9,
                                                total=9, discount=0, discount_rate='', tax_category='8%'))
    reject(blocks=priceless, text=primary.replace('架空店\n', '架空店\n追加商品8\n'), current=priceless_current)
    priceless_text = primary.replace(header + '\n', header + '\n追加商品8\n')
    reject(blocks=priceless, text=priceless_text)  # Missing model owners still cannot disappear.
    reject(blocks=priceless + [_word('1点', x=160, y=20, size=12)], text=priceless_text)
    reject(blocks=deepcopy(layout) + [_word('月火水', x=10, y=20, size=12)],
           text=primary.replace(header + '\n', header + '\n月火水\n'))
    reject(blocks=deepcopy(layout) + [_word(titles[0], x=10, y=20, size=12)])
    reject(text=primary.replace('09:37', '09:38'))
    registered = deepcopy(layout) + [_word('レジ0456', x=10, y=4, size=12),
                                    _word('取527:809123456注:009876543', x=10, y=22, size=8)]
    date_word = next(word for word in registered if word['text'] == header)
    date_word['x'] += 70
    for point in date_word['bbox']:
        point[0] += 70
    registered_text = primary.replace(header, 'レジ 0456\n' + header + '\n取527:809123456:009876543')
    s, n, c, identifier = bind(blocks=registered, text=registered_text)
    repaired, _ = supplemental._recover_native_price_column(before, s, registered_text, n, c, identifier)
    assert repaired == proposed  # Adjacent register rows remain owned across OCR wrapping.
    reject(blocks=registered, text=registered_text.replace('レジ 0456', 'レジ 0457'))
    skewed_header, moved_native = deepcopy(layout), deepcopy(native)
    for word in skewed_header + moved_native:
        word['y'] += 40
        for point in word['bbox']:
            point[1] += 40
    skewed_header.extend([_word('レジ0456', x=10, y=4, size=12),
                          _word('共通店案内', x=170, y=4, size=12),
                          _word('取527:809123456注:009876543', x=10, y=62, size=8)])
    s, n, c, identifier = bind(blocks=skewed_header, words=moved_native, text=registered_text)
    repaired, _ = supplemental._recover_native_price_column(before, s, registered_text, n, c, identifier)
    assert repaired == proposed  # A complete preceding register prefix may share a tilted header row.
    unknown = deepcopy(layout) + [_word('追加商品', x=60, y=20, size=12)]
    # A competitor inside the table cannot disappear as a clipped title tail.
    unknown[-1]['y'] = 60
    for point in unknown[-1]['bbox']:
        point[1] += 40
    reject(blocks=unknown)
    false_identity = dict(identity, primary_text_sha256='0' * 64)
    assert supplemental._recover_native_price_column(before, source, primary, capture, context, false_identity) == (None, {})
    foreign = deepcopy(capture)
    foreign['source_layout'][0]['bbox'][0][0] = context['roi'][0] - 1
    assert supplemental._recover_native_price_column(before, source, primary, foreign, context, identity) == (None, {})
    shifted = deepcopy(context)
    shifted['roi'][0] -= 1
    assert supplemental._recover_native_price_column(before, source, primary, capture, shifted, identity) == (None, {})


def test_literal_packet_bundle_requires_all_money_count_and_mixed_summary_owners():
    from receipt_parser.ocr import blocks_to_structured_text

    titles = ['青葉ケース4', '空色クロス8', '内木漏れ日セット2',
              '*星粒ビスケット9', '*星粒ビスケット9', '*こぐま豆菓子6',
              '夕色石けん7', '月影入浴玉3', '内レジ袋小']
    units = [827, 263, 2719, 129, 129, 167, 411, 425, 7]
    quantities = [1, 1, 1, 1, 1, 3, 1, 1, 3]
    discounts = [0, 0, 541, 0, 0, 0, 19, 21, 0]
    expected = [dict(description=title, qty=qty, unit_price=unit, total=qty * unit - discount,
                     discount=discount, discount_rate='', tax_category='8%')
                for title, qty, unit, discount in zip(titles, quantities, units, discounts, strict=True)]
    primary = '\n'.join([
        '上記正に領収しました', '青葉ケース4', '空色クロス8', '¥827', '¥263',
        '内木漏れ日セット2', '値引', '¥2,719', '-541',
        '*星粒ビスケット9', '¥129', '*星粒ビスケット9', '¥129',
        '*こぐま豆菓子6', '3コX単167', '¥501',
        '夕色石けん7', '¥411', '値引', '-19', '月影入浴玉3', '値引', '¥425', '-21',
        '内レジ袋小', '3コX単7', '¥21', '****まとめ値引き',
        'Bケア用品 (2個)', '-40', '¥836から', '¥796に致します',
        '小計', '13', '¥4,844', '8%外税タイショウ', '¥759', '8%税', '¥60',
        '10%外税タイショウ', '¥886', '10%外税', '¥188',
        '(10%内税タイショウ', '¥2,199)', '(10%内税', '¥199)', '(税合計', '¥447)',
        '伝票計', 'お預り', 'お釣り', '¥5,092', '¥6,092', '¥1,000', 'ポイント対象', '¥4,444',
    ])
    layout = [_word('上記正に領収しました', x=10, y=10, size=20)]
    y = 46
    for index, title in enumerate(titles):
        if index in (5, 8):
            layout.append(_word(title, x=10, y=y, size=20))
            y += 36
            layout.extend([_word('コX167' if index == 5 else '3コメ单7', x=140, y=y, size=20),
                           _word('¥501' if index == 5 else '¥21', x=300, y=y, size=20)])
        else:
            layout.extend([_word(title, x=10, y=y, size=20),
                           _word(f'¥{units[index]:,}', x=300, y=y, size=20)])
        if discounts[index]:
            y += 36
            layout.extend([_word('値引', x=10, y=y, size=20),
                           _word(f'-{discounts[index]}', x=300, y=y, size=20)])
        y += 36
    for line in ['****まとめ値引き', 'Bケア用品(2個)-40', '¥836から¥796に致します',
                 '小計/13¥4,844', '8%外税タイショウ¥759', '8%税¥60',
                 '10%外税タイショウ*1.886', '10%外¥188',
                 '(10%内税タイショウ*2.199)', '(10%内税¥199)', '(税合計¥447)',
                 '伝票計¥5,092', 'お預り¥6,092', 'お釣り¥1,000', 'ポイント対象¥4,444']:
        layout.append(_word(line, x=10, y=y, size=20))
        y += 36

    def sealed(blocks):
        text = blocks_to_structured_text(blocks)
        return build_supplemental_ocr_evidence(
            image_key=IMAGE_KEY, image_shape=[1800, 800],
            strategy_fingerprint=supplemental.SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT,
            merged_text=text, source_layout=blocks, tile_texts=[text, text])

    receipt = dict(document_type='receipt', currency='JPY', subtotal=4844, total=5092,
                   date='2028-02-14', taxes=[dict(rate='8%', label='外税', amount=60),
                                           dict(rate='10%', label='外税', amount=188),
                                           dict(rate='10%', label='内税', amount=199)],
                   line_items=deepcopy(expected))
    receipt['line_items'][0].update(unit_price=263, total=263)
    receipt['line_items'][1].update(unit_price=827, total=827)
    receipt['line_items'][-1]['description'] = '内レジ袋小 3コメ单7'
    receipt['line_items'][0]['tax_category'] = '0%'
    receipt['line_items'][2].update(unit_price=263, total=263, discount=541)
    evidence = sealed(layout)
    before = deepcopy(receipt)
    layout_before = deepcopy(layout)
    money = ('description', 'qty', 'unit_price', 'total', 'discount', 'discount_rate')

    def assert_money(proposed):
        assert proposed is not None
        assert [{field: item[field] for field in money} for item in proposed] == [
            {field: item[field] for field in money} for item in expected]
        assert [item['description'] for item in proposed].count('*星粒ビスケット9') == 2
        assert all(item['discount_rate'] == '' for item in proposed)

    proposed = supplemental._recover_literal_price_bundle(receipt, evidence, primary)
    assert_money(proposed)
    assert [item['tax_category'] for item in proposed] == [item['tax_category'] for item in receipt['line_items']]
    assert receipt == before and layout == layout_before
    assert supplemental._recover_literal_price_bundle(dict(receipt, line_items=proposed), evidence, primary) is None

    fused = deepcopy(receipt)
    fused['line_items'][:2] = [dict(fused['line_items'][1], description='青葉ケース4 空色クロス8')]
    assert_money(supplemental._recover_literal_price_bundle(fused, evidence, primary))
    # A complete unchanged P description need not be renamed to a different S
    # OCR spelling. Its digits and literal money remain in purchase order.
    variant = deepcopy(layout)
    next(word for word in variant if word['text'] == '夕色石けん7')['text'] = '夕色せっけん7'
    assert_money(supplemental._recover_literal_price_bundle(receipt, sealed(variant), primary))

    for changed_text in [primary.replace('-541', '-540'), primary.replace('小計\n13', '小計\n14'),
                         primary.replace('3コX単167', '4コX単167'), primary.replace('3コX単167', 'X167'),
                         primary.replace('*星粒ビスケット9\n¥129\n', '', 1),
                         primary.replace('¥827\n', '', 1), primary.replace('¥886', '¥1,887'),
                         primary.replace('上記正に領収しました\n', '上記正に領収しました\n別売り品11\n¥41\n'),
                         primary.replace('¥5,092', '¥5,093'),
                         primary.replace('月影入浴玉3\n', '未所有商品11\n月影入浴玉3\n')]:
        assert supplemental._recover_literal_price_bundle(receipt, evidence, changed_text) is None
    for old, new in [('青葉ケース4', '青葉ケース5'), ('-541', '-540'),
                     ('10%外税タイショウ*1.886', '10%外税タイショウ*1.887'),
                     ('10%外税タイショウ*1.886', '10%外税タイショウX1.886'),
                     ('(10%内税¥199)', '(10%内税¥0)'), ('お釣り¥1,000', 'お釣り¥999')]:
        bad = [dict(word, text=new) if word['text'] == old else deepcopy(word) for word in layout]
        assert supplemental._recover_literal_price_bundle(receipt, sealed(bad), primary) is None
    bad = deepcopy(layout)
    next(word for word in bad if word['text'] == '空色クロス8')['page'] = 1
    assert supplemental._recover_literal_price_bundle(fused, sealed(bad), primary) is None
    extra = deepcopy(receipt)
    extra['line_items'].insert(1, deepcopy(expected[0]))
    assert supplemental._recover_literal_price_bundle(extra, evidence, primary) is None
    rated = deepcopy(receipt)
    rated['line_items'][2]['discount_rate'] = '20%'
    assert supplemental._recover_literal_price_bundle(rated, evidence, primary) is None
    unmatched_detail = deepcopy(receipt)
    unmatched_detail['line_items'][-1]['description'] = '内レジ袋小 4コメ单7'
    assert supplemental._recover_literal_price_bundle(unmatched_detail, evidence, primary) is None
    assert receipt['date'] == '2028-02-14' and receipt['taxes'] == before['taxes']
