"""A visible star disambiguates equal prices only inside a closed owned table."""

from copy import deepcopy
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np

from receipt_parser import pipeline, receipt_supplemental_ocr as supplemental
from receipt_parser.ocr import OCRResult, _ocr_cache_key, blocks_to_structured_text
from receipt_parser.receipt_pixel_markers import build_pixel_marker_context, recover_pixel_marker_tax_partition
from receipt_parser.schema import Receipt


def test_original_stars_require_complete_bijection_calibration_and_one_mixed_partition():
    image = np.full((1000, 800, 3), 255, dtype=np.uint8)
    primary, source = [], []
    titles = ['FoodA40', 'FoodB50', 'FoodC60', 'SupplyD70', 'FoodE80', 'FoodF90', 'Bag20', 'Inner30']
    amounts = [110, 170, 190, 240, 240, 130, 5, 88]

    def word(text, x, y, width=40):
        return dict(text=text, x=x, y=y, page=0, confidence=.91,
                    bbox=[[x, y], [x+width, y], [x+width, y+36], [x, y+36]])

    primary.append(word('領収証', 20, 10, 200))
    primary.append(word('2030年05月20日12:10', 20, 55, 300))
    for i, (title, amount) in enumerate(zip(titles, amounts, strict=True)):
        y = 110 + i*50
        row = [word(f'{900+i:04}', 20, y, 70)]
        if i == 7:
            row += [word('内', 100, y, 16), word('#', 120, y, 20)]
        elif i < 3:
            row += [word('*', 100, y, 25)]
        elif i == 6:
            row += [word('#', 100, y, 25)]
        row += [word(title, 140, y, 200), word('¥', 600, y, 20), word(str(amount), 630, y, 60)]
        primary.extend(row)
        source.extend(deepcopy(row))
        for token in row:
            if token['text'] not in {'内', '領収証'}:
                cv2.putText(image, token['text'], (token['x'], y+29), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,0), 1)
        if i in {4, 5}:
            cv2.putText(image, '*', (100, y+29), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,0), 1)
    footer = ['小計¥1,173', '(d外8%対象額¥840)', 'd外8%¥67', '(e外10%対象額¥245)', 'e外10%¥24',
              '(f内10%対象額¥88)', '(f内10%¥8)', '消費税等¥91', '合計¥1,264',
              'お買上点数8点', '*印は軽減税率対象(外8%)商品です', '***']
    primary.extend(word(text, 20, 520+i*40, 500) for i, text in enumerate(footer))
    source.extend(deepcopy(primary[-len(footer):]))
    # A shape-identical decorative star has no code/title/price body owner.
    cv2.putText(image, '*', (100, 949), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,0), 1)
    text = blocks_to_structured_text(primary)

    def sealed(layout=source, pixels=image):
        merged = blocks_to_structured_text(layout)
        return supplemental.build_supplemental_ocr_evidence(image_key=_ocr_cache_key(pixels),
            image_shape=list(pixels.shape[:2]), strategy_fingerprint=supplemental.SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT,
            merged_text=merged, source_layout=layout, tile_texts=[merged, 'footer'])

    evidence = sealed()
    original = Receipt(document_type='receipt', merchant='検証店', currency='JPY', subtotal=1173, total=1264,
        taxes=[dict(rate='8%', label='外税', amount=67), dict(rate='10%', label='外税', amount=24),
               dict(rate='10%', label='内税', amount=8)],
        line_items=[dict(description=title, qty=1, unit_price=value, total=value,
                         tax_category='8%' if i in {0,1,2,3,5} else '10%')
                    for i,(title,value) in enumerate(zip(titles, amounts, strict=True))]).model_dump()
    context = build_pixel_marker_context(image, text, primary, evidence)
    before = deepcopy(original)
    proof = recover_pixel_marker_tax_partition(original, evidence, text, context)
    assert original == before and proof['accepted']
    assert proof['changed_item_indices'] == [3,4]
    assert proof['proposed'] == ['8%', '8%', '8%', '10%', '8%', '8%', '10%', '10%']
    assert len(proof['pixels']['anchors']) == 3 and proof['pixels']['confidence'] is None
    assert {m['primary_row'] for m in proof['pixels']['matches']} == {2,3,4,6,7}
    assert proof['external_rows'] == [5,8] and proof['included_rows'] == [9]
    # A complete owner proof also removes only the separated metadata prefix;
    # already-correct categories must not skip this description-only repair.
    marked = deepcopy(original)
    for item, rate in zip(marked['line_items'], proof['proposed'], strict=True):
        item['tax_category'] = rate
    for index, marker in [(0, '*'), (6, '#'), (7, '#')]:
        marked['line_items'][index]['description'] = marker + titles[index]
    marked_before = deepcopy(marked)
    marker_proof = recover_pixel_marker_tax_partition(marked, evidence, text, context)
    assert marked == marked_before and marker_proof['accepted']
    assert marker_proof['changed_item_indices'] == []
    assert [row['item_index'] for row in marker_proof['description_updates']] == [0, 6, 7]
    normalized, metadata = supplemental.apply_supplemental_ocr_evidence(
        marked, evidence, primary_text=text, pixel_marker_context=context)
    expected = deepcopy(marked)
    for index in (0, 6, 7):
        expected['line_items'][index]['description'] = titles[index]
    assert normalized == expected and 'line_items' in metadata['accepted_fields']
    assert metadata['line_items']['mode'] == 'pixel_owned_marker_description'
    assert not recover_pixel_marker_tax_partition(normalized, evidence, text, context)['accepted']
    unbound, _ = supplemental.apply_supplemental_ocr_evidence(marked, evidence, primary_text=text)
    assert [i['description'] for i in unbound['line_items']] == [i['description'] for i in marked['line_items']]
    for name in ('##Bag20', '#Bag-20', 'Bag#20'):
        meaningful = deepcopy(expected)
        meaningful['line_items'][6]['description'] = name
        unchanged, _ = supplemental.apply_supplemental_ocr_evidence(
            meaningful, evidence, primary_text=text, pixel_marker_context=context)
        assert unchanged['line_items'][6]['description'] == name
    marked_trace = []
    pipeline._apply_final_supplemental_ocr_evidence(marked, evidence, marked_trace,
        primary_text=text, pixel_marker_context=context)
    assert marked['line_items'] == expected['line_items']
    assert set(marked_trace[-1]['changes']) == {'line_items'}
    assert marked_trace[-1]['pixel_marker_proof']['description_updates'] == marker_proof['description_updates']
    for field, value in [('qty', 2), ('qty', True), ('unit_price', 239), ('total', 239), ('description', 'FoodE81'),
                         ('discount', 1), ('gross', 240), ('_tax_category_locked', '10%')]:
        corrupt = deepcopy(original)
        corrupt['line_items'][4][field] = value
        assert not recover_pixel_marker_tax_partition(corrupt, evidence, text, context)['accepted']
    for corrupt in (dict(original, subtotal=1174), dict(original, total=1265),
                    dict(original, line_items=original['line_items'][:-1])):
        assert not recover_pixel_marker_tax_partition(corrupt, evidence, text, context)['accepted']
    assert not recover_pixel_marker_tax_partition(original, evidence, text+'extra', context)['accepted']
    assert not recover_pixel_marker_tax_partition(original, evidence, text, None)['accepted']
    assert not recover_pixel_marker_tax_partition(original, {}, text, dict(context, source_identity=None))['accepted']
    mutated = deepcopy(context)
    mutated['primary_layout'][3]['text'] = 'different'
    assert not recover_pixel_marker_tax_partition(original, evidence, text, mutated)['accepted']
    stale = deepcopy(context)
    stale['image'][0,0] = 0
    assert not recover_pixel_marker_tax_partition(original, evidence, text, stale)['accepted']
    for old, replacement in [('d外8%¥67', 'e外8%¥67'), ('e外10%¥24', 'e内10%¥24'),
                             ('小計¥1,173', '小計¥1,174'), ('*印は軽減税率対象(外8%)商品です', '*印は軽減税率対象(外10%)商品です')]:
        changed = deepcopy(primary)
        next(w for w in changed if w['text'] == old)['text'] = replacement
        changed_text = blocks_to_structured_text(changed)
        changed_context = build_pixel_marker_context(image, changed_text, changed, evidence)
        assert not recover_pixel_marker_tax_partition(original, evidence, changed_text, changed_context)['accepted']
    # Removing the distinguishing positive leaves both equal-price partitions.
    erased = image.copy()
    erased[310:350,98:130] = 255
    erased_source = sealed(pixels=erased)
    erased_context = build_pixel_marker_context(erased, text, primary, erased_source)
    assert not recover_pixel_marker_tax_partition(original, erased_source, text, erased_context)['accepted']
    bad_legend = deepcopy(source)
    next(w for w in bad_legend if w['text'].startswith('*印'))['text'] = '*印は軽減税率対象(外10%)商品です'
    wrong_source = sealed(bad_legend)
    wrong_context = build_pixel_marker_context(image, text, primary, wrong_source)
    assert not recover_pixel_marker_tax_partition(original, wrong_source, text, wrong_context)['accepted']

    wrong_mode = deepcopy(source)
    next(w for w in wrong_mode if w['text'] == '内')['text'] = '外'
    mode_source = sealed(wrong_mode)
    mode_context = build_pixel_marker_context(image, text, primary, mode_source)
    assert not recover_pixel_marker_tax_partition(original, mode_source, text, mode_context)['accepted']
    count_layout = deepcopy(primary)
    next(w for w in count_layout if w['text'] == 'お買上点数8点')['text'] = 'お買上点数7点'
    count_text = blocks_to_structured_text(count_layout)
    count_context = build_pixel_marker_context(image, count_text, count_layout, evidence)
    assert not recover_pixel_marker_tax_partition(original, evidence, count_text, count_context)['accepted']
    pretable_layout = [*deepcopy(primary), word('値引¥0', 20, 95, 200)]
    pretable_text = blocks_to_structured_text(pretable_layout)
    pretable_context = build_pixel_marker_context(image, pretable_text, pretable_layout, evidence)
    assert not recover_pixel_marker_tax_partition(original, evidence, pretable_text, pretable_context)['accepted']
    quantity_layout = deepcopy(primary)
    next(w for w in quantity_layout if w['text'] == titles[4])['text'] += '2個×120'
    quantity_text = blocks_to_structured_text(quantity_layout)
    quantity_context = build_pixel_marker_context(image, quantity_text, quantity_layout, evidence)
    quantity_receipt = deepcopy(original)
    quantity_receipt['line_items'][4]['description'] += '2個×120'
    assert not recover_pixel_marker_tax_partition(quantity_receipt, evidence, quantity_text, quantity_context)['accepted']
    # An identical star occupying an explicitly nonstar prefix remains forbidden.
    false_prefix_pixels = image.copy()
    false_prefix_pixels[410:450,98:130] = 255
    cv2.putText(false_prefix_pixels, '*', (100, 439), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,0), 1)
    false_prefix_source = sealed(pixels=false_prefix_pixels)
    false_prefix_context = build_pixel_marker_context(false_prefix_pixels, text, primary, false_prefix_source)
    assert not recover_pixel_marker_tax_partition(original, false_prefix_source, text, false_prefix_context)['accepted']
    # Two otherwise distinct rows cannot share any marker-region pixels.
    overlapping = deepcopy(primary)
    for token in overlapping:
        if token['y'] == 260 and token['x'] in {20, 140}:
            token['bbox'][0][1] = token['bbox'][1][1] = 240
    overlap_text = blocks_to_structured_text(overlapping)
    overlap_context = build_pixel_marker_context(image, overlap_text, overlapping, evidence)
    assert not recover_pixel_marker_tax_partition(original, evidence, overlap_text, overlap_context)['accepted']
    wrong_page = deepcopy(source)
    wrong_page[0]['page'] = 1
    page_source = sealed(wrong_page)
    page_context = build_pixel_marker_context(image, text, primary, page_source)
    assert not recover_pixel_marker_tax_partition(original, page_source, text, page_context)['accepted']
    malformed = deepcopy(primary)
    next(w for w in malformed if w['text'] == '(d外8%対象額¥840)')['text'] = '(d外8%対象額¥8,40)'
    malformed_text = blocks_to_structured_text(malformed)
    malformed_context = build_pixel_marker_context(image, malformed_text, malformed, evidence)
    assert not recover_pixel_marker_tax_partition(original, evidence, malformed_text, malformed_context)['accepted']
    for label in ('非課税対象額¥0', '0%対象額¥0'):
        exempt_source = sealed([*deepcopy(source), word(label, 20, 55, 300)])
        exempt_context = build_pixel_marker_context(image, text, primary, exempt_source)
        assert not recover_pixel_marker_tax_partition(original, exempt_source, text, exempt_context)['accepted']
    for label in ('#印商品の税率は0%です', '#印商品の税率は5%です', 'g外5%¥0',
                  '(5%対象額¥5)', '5%税額¥0', '適用税率は0%です', '8%税額¥999', '10%税額¥0,,8'):
        conflict_source = sealed([*deepcopy(source), word(label, 20, 55, 300)])
        conflict_context = build_pixel_marker_context(image, text, primary, conflict_source)
        assert not recover_pixel_marker_tax_partition(original, conflict_source, text, conflict_context)['accepted']
    promotion_source = sealed([*deepcopy(source), word('5% OFF', 20, 55, 300)])
    promotion_context = build_pixel_marker_context(image, text, primary, promotion_source)
    assert recover_pixel_marker_tax_partition(original, promotion_source, text, promotion_context)['accepted']
    raw_conflict = supplemental.build_supplemental_ocr_evidence(image_key=_ocr_cache_key(image),
        image_shape=list(image.shape[:2]), strategy_fingerprint=supplemental.SUPPLEMENTAL_OCR_STRATEGY_FINGERPRINT,
        merged_text=evidence['merged_text'], source_layout=source,
        tile_texts=[evidence['merged_text'], '*印は軽減税率対象(外10%)商品です'])
    raw_context = build_pixel_marker_context(image, text, primary, raw_conflict)
    assert not recover_pixel_marker_tax_partition(original, raw_conflict, text, raw_context)['accepted']
    # Flattened OCR may separate the count from its label; authenticated P owns the full row.
    split_text = text.replace('お買上点数8点', 'お買上点数').replace('***', '8点\n***')
    split_context = build_pixel_marker_context(image, split_text, primary, evidence)
    assert recover_pixel_marker_tax_partition(original, evidence, split_text, split_context)['accepted']

    with TemporaryDirectory() as directory:
        previous = Path.cwd()
        os.chdir(directory)
        try:
            with patch('receipt_parser.usage.track_document'), patch.multiple(pipeline,
                check_model_available=lambda *_args: None,
                _run_extraction_pipeline=lambda **_kwargs: (deepcopy(original), [], [], Receipt.model_validate(original)),
                _prepare_receipt_output_payload=lambda receipt, *_args, **_kwargs: receipt.model_dump(),
                load_image=lambda *_args: [image],
                run_cloud_vision=lambda *_args, **_kwargs: OCRResult(blocks=primary, layout_blocks=primary,
                    chosen_text=text, source='fresh', layout_trusted=True),
                acquire_quantity_crop_ocr_evidence=lambda *_args, **_kwargs: None,
                acquire_native_owner_crop_ocr_evidence=lambda *_args, **_kwargs: None,
                acquire_supplemental_ocr_evidence=lambda *_args, **_kwargs: evidence):
                image_result = pipeline.process_document(Path(directory)/'input.jpg', ocr_engine=object(), apply_user_rules=False, debug=True)
                text_result = pipeline.process_ocr_text(text, supplemental_ocr_evidence=evidence,
                    pixel_marker_context=context, apply_user_rules=False, debug=True)
        finally:
            os.chdir(previous)
    for result in (image_result, text_result):
        assert [i['tax_category'] for i in result['line_items']] == proof['proposed']
        assert [i['total'] for i in result['line_items']] == amounts
        assert result['taxes'] == original['taxes'] and result['total'] == original['total']
        event = next(e for e in result['_receipt_mutation_trace'] if 'pixel_marker_proof' in e)
        assert event['stage'] == 'supplemental_ocr_field_recovery'
        assert set(event['changes']) == {'line_items'}
        assert event['pixel_marker_proof']['image_pixel_sha256'] == context['image_pixel_sha256']


if __name__ == '__main__':
    test_original_stars_require_complete_bijection_calibration_and_one_mixed_partition()
