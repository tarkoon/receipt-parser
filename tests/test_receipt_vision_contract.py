"""Blind original-pixel recovery needs exact owners and a closed printed basket."""

import base64
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from receipt_parser import llm, pipeline, receipt_supplemental_ocr as supplemental, receipt_vision as vision, usage
from receipt_parser.ocr import OCRResult, _ocr_cache_key, blocks_to_structured_text
from receipt_parser.receipt_pixel_markers import build_pixel_marker_context
from receipt_parser.schema import Receipt


def test_blind_one_conflict_recovery_is_source_bound_cached_and_field_limited(tmp_path, monkeypatch):
    image = np.full((800, 800, 3), 255, dtype=np.uint8)
    primary = []

    def word(text, x, y, width=160):
        return dict(text=text, x=x, y=y, page=0, confidence=.95,
                    bbox=[[x, y], [x+width, y], [x+width, y+22], [x, y+22]])

    primary.append(word('検証店', 20, 10))
    for index, (title, price) in enumerate(zip(('First', 'Second', 'Third'), (90, 100, 100))):
        y = 80 + index*100
        primary += [word(f'53123456780{index:02}JAN', 20, y-30, 300),
                    word(str(301+index), 20, y, 55), word(title, 100, y), word(f'¥{price}', 600, y, 100)]
        cv2.putText(image, f'{301+index} {title} {110 if index == 2 else price}',
                    (20, y+19), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 1)
    primary += [word('小計', 20, 400), word('¥300', 600, 400, 100)]
    footer = ['外8%対象額¥300', '外8%¥24', '合計¥324', '現金¥400', 'お釣り¥76', 'お買上点数3点']
    primary += [word(text, 20, 440+index*40, 600) for index, text in enumerate(footer)]
    source = deepcopy(primary)
    next(row for row in source if row['text'] == '¥100' and row['y'] == 280)['text'] = '¥120'
    next(row for row in source if row['text'] == 'Second')['text'] = 'SecondAlt'
    text = blocks_to_structured_text(primary)

    def seal(pixels=image, layout=source):
        merged = blocks_to_structured_text(layout)
        return supplemental.build_supplemental_ocr_evidence(image_key=_ocr_cache_key(pixels),
            image_shape=list(pixels.shape[:2]), strategy_fingerprint=supplemental.SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT,
            merged_text=merged, source_layout=layout, tile_texts=[merged, 'footer'])

    evidence = seal()
    context = build_pixel_marker_context(image, text, primary, evidence)
    original = Receipt(document_type='receipt', merchant='検証店', currency='JPY', subtotal=300, total=324,
        amount_paid=324, taxes=[dict(rate='8%', label='外税', amount=24)],
        line_items=[dict(description=name, qty=1, unit_price=price, total=price, tax_category='8%')
                    for name, price in zip(('First', 'Second', 'Third'), (90, 100, 100))]).model_dump()
    prepared = vision._prepare_plan(original, text, evidence, context, vision.DEFAULT_VISION_MODEL)
    assert prepared is not None
    bound, _png, _path = prepared
    assert bound['item_index'] == 2 and bound['roi'][3] < 400
    rows = [dict(printed_text=f"{owner['barcode']}\n{owner['code']} {owner['title_candidates'][-1]} ¥{price}",
                 price_literal=str(price), marker_literal=None, quantity_lines=[], discount_lines=[])
            for owner in bound['owners'] for price in [110 if owner['item_index'] == 2 else owner['price']]]
    response_data = dict(lines=[dict(text=line) for row in rows for line in row['printed_text'].splitlines()], item_rows=rows)
    alternate_title = deepcopy(original)
    alternate_title['line_items'][1]['description'] = 'SecondAlt'
    assert vision._prepare_plan(alternate_title, text, evidence, context, vision.DEFAULT_VISION_MODEL) == prepared
    shifted_source = deepcopy(source)
    for row in shifted_source:
        if row['y'] == 180:
            row['y'] += 12
            for point in row['bbox']:
                point[1] += 12
    shifted_evidence = seal(layout=shifted_source)
    shifted_pixels = build_pixel_marker_context(image, text, primary, shifted_evidence)
    shifted_bound, _, _ = vision._prepare_plan(original, text, shifted_evidence, shifted_pixels, vision.DEFAULT_VISION_MODEL)
    assert 1 not in shifted_bound['title_conflicts'] and shifted_bound['owners'][0]['title_candidates'] == ['Second']
    unknown_title = deepcopy(response_data)
    unknown_title['item_rows'][1]['printed_text'] = rows[1]['printed_text'].replace('SecondAlt', 'Second?')
    unknown_title['lines'] = [dict(text=line) for row in unknown_title['item_rows'] for line in row['printed_text'].splitlines()]
    assert vision._admit(original, unknown_title, bound) is None
    # A title after the monetary owner must not become another monetary owner.
    later_title_source = deepcopy(primary)
    next(row for row in later_title_source if row['text'] == '¥100' and row['y'] == 180)['text'] = '¥120'
    next(row for row in later_title_source if row['text'] == 'Third')['text'] = 'ThirdAlt'
    later_evidence = seal(layout=later_title_source)
    later_pixels = build_pixel_marker_context(image, text, primary, later_evidence)
    later_bound, _, _ = vision._prepare_plan(original, text, later_evidence, later_pixels, vision.DEFAULT_VISION_MODEL)
    assert later_bound['item_index'] == 1 and later_bound['owners'][-1]['item_index'] == 2
    later_rows = [dict(printed_text=f"{owner['barcode']}\n{owner['code']} {owner['title_candidates'][-1]} ¥{price}",
                       price_literal=str(price), quantity_lines=[], discount_lines=[])
                  for owner in later_bound['owners'] for price in [110 if owner['item_index'] == 1 else owner['price']]]
    later_read = dict(lines=[dict(text=line) for row in later_rows for line in row['printed_text'].splitlines()], item_rows=later_rows)
    later_accepted = vision._admit(original, later_read, later_bound)
    assert later_accepted['line_items'][1]['total'] == 110 and later_accepted['line_items'][2]['total'] == 100
    assert later_accepted['line_items'][2]['description'] == 'ThirdAlt'
    later_rows[1]['printed_text'] = later_rows[1]['printed_text'].replace('¥110', '¥109')
    later_rows[1]['price_literal'] = '109'
    later_rows[2]['printed_text'] = later_rows[2]['printed_text'].replace('¥100', '¥101')
    later_rows[2]['price_literal'] = '101'
    later_read['lines'] = [dict(text=line) for row in later_rows for line in row['printed_text'].splitlines()]
    assert vision._admit(original, later_read, later_bound) is None
    no_barcode_context, no_barcode_read = deepcopy(bound), deepcopy(response_data)
    no_barcode_context['owners'][0]['barcode'] = None
    no_barcode_read['item_rows'][0]['printed_text'] = rows[0]['printed_text'].splitlines()[-1]
    no_barcode_read['lines'].pop(0)
    assert vision._admit(original, no_barcode_read, no_barcode_context) is not None
    no_barcode_read['item_rows'][0]['printed_text'] += '\n値引 ¥10'
    no_barcode_read['lines'].insert(1, dict(text='値引 ¥10'))
    assert vision._admit(original, no_barcode_read, no_barcode_context) is None
    requests, options, billed = [], [], []
    finish_reason = ['stop']

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(model=vision.DEFAULT_VISION_MODEL,
            usage=SimpleNamespace(prompt_tokens=200, completion_tokens=100, cost=.001),
            choices=[SimpleNamespace(finish_reason=finish_reason[0], message=SimpleNamespace(content=json.dumps(response_data)))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.with_options = lambda **kwargs: options.append(kwargs) or client
    monkeypatch.setattr(llm, '_get_api_client', lambda *_args: client)
    monkeypatch.setattr(usage, 'track_openrouter_call', lambda **kwargs: billed.append(kwargs))
    monkeypatch.setattr(vision, 'VISION_CACHE_DIR', tmp_path/'vision')
    monkeypatch.setenv('OPENROUTER_API_KEY', 'offline-test-key')
    monkeypatch.setenv('RECEIPT_VISION_MODEL', vision.DEFAULT_VISION_MODEL)
    before = deepcopy(original)
    accepted, metadata = vision.recover_with_vision(original, text, evidence, context)
    assert original == before and metadata['status'] == 'accepted' and metadata['attempts'] == 1
    expected = deepcopy(original)
    expected['line_items'][2]['unit_price'] = expected['line_items'][2]['total'] = 110
    expected['line_items'][1]['description'] = 'SecondAlt'
    assert accepted == expected
    request = requests[0]
    assert request['temperature'] == 0 and request['seed'] == 42 and request['max_tokens'] == 4096
    assert options == [dict(timeout=60, max_retries=0)]
    assert request['extra_body']['provider'] == dict(order=['Google AI Studio'], allow_fallbacks=False, require_parameters=True)
    assert request['messages'][0]['content'] == vision._VISION_SYSTEM
    assert request['messages'][1]['content'][0]['text'] == vision._VISION_USER
    encoded = request['messages'][1]['content'][1]['image_url']['url'].split(',', 1)[1]
    crop = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), dtype=np.uint8), cv2.IMREAD_COLOR)
    x0, y0, x1, y1 = metadata['context']['roi']
    assert np.array_equal(crop, image[y0:y1, x0:x1])
    cached, cache_meta = vision.recover_with_vision(original, text, evidence, context, mode='cache_only')
    assert cached == expected and cache_meta['source'] == 'cache' and cache_meta['attempts'] == 0
    cached_alternate, alternate_meta = vision.recover_with_vision(alternate_title, text, evidence, context, mode='cache_only')
    assert cached_alternate == expected and alternate_meta['attempts'] == 0
    assert len(requests) == len(billed) == 1 and billed[0]['cost_usd'] == .001
    cache_path = vision.vision_cache_files(image)[0]
    saved = cache_path.read_bytes()
    capture = json.loads(saved)
    capture['evidence']['item_rows'][-1]['price_literal'] = '111'
    cache_path.write_text(json.dumps(capture), encoding='utf-8')
    assert vision.recover_with_vision(original, text, evidence, context, mode='cache_only')[0] is None
    cache_path.write_bytes(saved)
    for field, value in [('qty', 2), ('discount', 5), ('discount_rate', '5%'), ('description', 'unowned')]:
        corrupt = deepcopy(original)
        corrupt['line_items'][2][field] = value
        assert vision.recover_with_vision(corrupt, text, evidence, context) == (None, None)
    for corrupt in (expected, dict(original, document_type='bill'), dict(original, currency='USD')):
        assert vision.recover_with_vision(corrupt, text, evidence, context) == (None, None)
    for corrupt in (None, dict(context, source_identity={}), dict(context, image=image[:100])):
        assert vision.recover_with_vision(original, text, evidence, corrupt) == (None, None)
    mutated = deepcopy(context)
    mutated['primary_layout'][0]['bbox'][0][0] += 1
    assert vision.recover_with_vision(original, text, evidence, mutated) == (None, None)
    pixels = image.copy()
    pixels[0, 0] = 0
    new_source = seal(pixels=pixels)
    new_context = build_pixel_marker_context(pixels, text, primary, new_source)
    assert vision.recover_with_vision(original, text, new_source, new_context, mode='cache_only')[1]['reason'] == 'cache_miss'
    with monkeypatch.context() as scoped:
        scoped.setenv('RECEIPT_VISION_MODEL', 'google/different-model')
        assert vision.recover_with_vision(original, text, evidence, context, mode='cache_only')[1]['reason'] == 'cache_miss'
        scoped.setenv('RECEIPT_VISION_MODEL', '')
        assert vision.recover_with_vision(original, text, evidence, context) == (None, None)
    with monkeypatch.context() as scoped:
        scoped.setattr(vision, 'VISION_PROFILE_FINGERPRINT', 'changed-profile')
        assert vision.recover_with_vision(original, text, evidence, context, mode='cache_only')[1]['reason'] == 'cache_miss'
    for change in ('wrong_math', 'wrong_owner', 'missing_barcode', 'missing_detail', 'extra_discount', 'partial'):
        altered = deepcopy(response_data)
        if change == 'wrong_math':
            response_data['item_rows'][-1]['price_literal'] = '111'
        elif change == 'wrong_owner':
            response_data['item_rows'][-1]['printed_text'] = '303 Other ¥110'
        elif change == 'missing_barcode':
            response_data['item_rows'][-1]['printed_text'] = '303 Third ¥110'
        elif change == 'missing_detail':
            del response_data['item_rows'][-1]['quantity_lines']
        elif change == 'extra_discount':
            response_data['lines'].append(dict(text='値引 ¥10'))
        else:
            finish_reason[0] = 'length'
        assert vision.recover_with_vision(original, text, evidence, context, mode='fresh')[0] is None
        response_data.clear()
        response_data.update(altered)
        finish_reason[0] = 'stop'
    assert cache_path.read_bytes() == saved and len(requests) == len(billed) == 7
    with monkeypatch.context() as scoped:
        scoped.setattr(vision, '_openrouter_chat', lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError()))
        assert vision.recover_with_vision(original, text, evidence, context, mode='fresh')[1]['reason'] == 'TimeoutError'

    # Both public paths run the declared vision phase before user aliases.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(usage, 'track_document', lambda *_args: None)
    monkeypatch.setattr(pipeline, 'check_model_available', lambda *_args: None)
    monkeypatch.setattr(pipeline, '_run_extraction_pipeline', lambda **_kwargs:
                        (deepcopy(original), [], [], Receipt.model_validate(original)))
    monkeypatch.setattr(pipeline, '_prepare_receipt_output_payload', lambda receipt, *_args, **_kwargs: receipt.model_dump())
    monkeypatch.setattr(pipeline, 'load_image', lambda *_args: [image])
    monkeypatch.setattr(pipeline, 'run_cloud_vision', lambda *_args, **_kwargs: OCRResult(blocks=primary,
        layout_blocks=primary, chosen_text=text, source='fresh', layout_trusted=True))
    monkeypatch.setattr(pipeline, 'acquire_supplemental_ocr_evidence', lambda *_args, **_kwargs: evidence)
    monkeypatch.setattr(pipeline, 'acquire_quantity_crop_ocr_evidence', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, 'acquire_native_owner_crop_ocr_evidence', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, '_apply_final_supplemental_ocr_evidence', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, '_apply_user_rules', lambda result: dict(result, merchant='User alias'))
    image_result = pipeline.process_document(Path('input.jpg'), ocr_engine=object(), debug=True)
    text_result = pipeline.process_ocr_text(text, supplemental_ocr_evidence=evidence, pixel_marker_context=context,
                                           debug=True)
    for result in (image_result, text_result):
        assert result['merchant'] == 'User alias' and result['line_items'] == expected['line_items'], result.get('_vision_fallback')
        stages = [event['stage'] for event in result['_receipt_mutation_trace']]
        assert stages.index('last_resort_vision_field_recovery') < stages.index('user_rule_alias')
        event = next(event for event in result['_receipt_mutation_trace'] if event['stage'] == 'last_resort_vision_field_recovery')
        assert set(event['changes']) == {'line_items'}


def test_visual_seller_requires_accounted_upper_roles_and_preserves_branch():
    receipt = Receipt(document_type='receipt', merchant='Mall East', location='East', currency='JPY').model_dump()
    context = dict(kind='header', merchant='Seller', phone='0312345678', registration='T1234567890123',
                   upper_count=2, upper_rows=['Mall', 'East'])
    evidence = dict(merchant='Seller', location='Invented location', lines=[
        dict(text='Mall', role='complex', role_evidence='Upper facility mark'),
        dict(text='East', role='branch', role_evidence='Site mark below facility'),
        dict(text='Seller TEL03-1234-5678', role='seller', role_evidence='Seller mark on receipt contact row'),
        dict(text='T1234567890123', role='registration', role_evidence='Tax registration')])
    accepted = vision._admit(receipt, evidence, context)
    assert accepted == dict(receipt, merchant='Seller')
    for field, value in [('role', 'unknown'), ('role', 'seller'), ('role_evidence', None), ('text', 'Unrelated')]:
        corrupt = deepcopy(evidence)
        corrupt['lines'][0][field] = value
        assert vision._admit(receipt, corrupt, context) is None
    for field, value in [('merchant', 'Other'), ('lines', evidence['lines'][1:])]:
        assert vision._admit(receipt, dict(evidence, **{field: value}), context) is None
