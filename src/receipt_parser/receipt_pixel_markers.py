"""Original-pixel markers constrained by a complete primary mixed-tax table."""

import hashlib
import math
import re
from copy import deepcopy
from itertools import combinations
from statistics import median

import cv2
import numpy as np

from .ocr import layout_blocks_sha256
from .receipt_financial import _rate_base_tax_pair_is_valid
from .receipt_identity_payment import _DATE_LINE_RE, _TIME_HHMM_RE
from .receipt_projection import _collect_direct_summary_owners, _group_layout_rows
from .receipt_supplemental_ocr import (
    _build_financial_source_identity, _validated_evidence, _valid_source_layout,
    _cash_box, _cash_compact, _cash_title_key, _cash_overlap,
    _LITERAL_TABLE_COUNT_RE, _LITERAL_PRICE_AMOUNT_RE, _LITERAL_PRICE_QTY_RE,
    _LITERAL_PRICE_DISCOUNT_RE, _LITERAL_PRICE_PERCENT_RE,
    _CASH_LITERAL_AMOUNT_RE,
)


def build_pixel_marker_context(image, primary_text, primary_layout, supplemental):
    """Bind original pixels and authenticated original-frame P geometry.

    Callers obtain P from the same OCR call, or an image-key-verified replay;
    they must not build this expectation from an untrusted sidecar's identity.
    This transient context performs no OCR acquisition or cache writes.
    """
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3):
        return None
    identity = _build_financial_source_identity(image, primary_text, supplemental)
    if (identity is None or not _valid_source_layout(primary_layout, identity['image_shape'])
            or any(word['page'] != 0 for word in primary_layout)):
        return None
    return dict(image=image, image_pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        source_identity=identity, primary_layout=deepcopy(primary_layout),
        primary_layout_sha256=layout_blocks_sha256(primary_layout))


def _code_marker_slots(layout):
    """Separate code, optional mode/glyph, full title and exact currency amount."""
    slots = []
    for number, row in enumerate(_group_layout_rows(layout)):
        if not re.fullmatch(r'\d{3,}', row[0]['text']):
            continue
        currencies = [i for i, word in enumerate(row) if re.search(r'[¥￥]', word['text'])]
        if len(currencies) != 1:
            continue
        price_at = currencies[0]
        price = _CASH_LITERAL_AMOUNT_RE.fullmatch(_cash_compact(''.join(w['text'] for w in row[price_at:])))
        if not price or price_at < 2:
            continue
        start = 1
        while start < price_at and re.fullmatch(r'(?:内|外)?[*＊※#]?', row[start]['text']):
            start += 1
        title_words = row[start:price_at]
        if not title_words or not any(char.isalpha() for w in title_words for char in w['text']):
            continue
        prefix = ''.join(w['text'] for w in row[1:start])
        if not re.fullmatch(r'(?:内|外)?[*＊※#]?', prefix):
            continue
        code, title = _cash_box(row[0]), _cash_box(title_words[0])
        marker_words = [word for word in row[1:start] if word['text'] == '*']
        if len(marker_words) > 1 or title[0] <= code[2]:
            continue
        box = [max(0, min([code[2], *(_cash_box(w)[0] for w in marker_words)]) - 2),
               min(code[1], title[1]), title[0], max(code[3], title[3])]
        slots.append(dict(row=number, code=row[0]['text'], code_word=row[0], words=row,
            title=''.join(w['text'] for w in title_words), title_words=title_words,
            prefix_words=row[1:start],
            value=int(price[1].replace(',', '')), mode=prefix[:1] if prefix[:1] in {'内', '外'} else None,
            marker=prefix[-1:] if prefix[-1:] in {'*', '＊', '※', '#'} else None,
            marker_word=marker_words[0] if marker_words else None, box=box))
    return slots


def _calibrated_pixel_stars(image, source_slots, primary_slots):
    """Distinct OCR-owned stars must separate from local negatives when held out."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def contains(box, center):
        return box[0] <= center[0] < box[2] and box[1] <= center[1] < box[3]

    for i, slot in enumerate(primary_slots):
        a = slot['box']
        if any(min(a[2], b['box'][2]) > max(a[0], b['box'][0])
               and min(a[3], b['box'][3]) > max(a[1], b['box'][1]) for b in primary_slots[i+1:]):
            return None
    negative_words = [word for slot in [*primary_slots, *source_slots]
                      for word in [slot['code_word'], *slot['title_words'],
                                   *(w for w in slot['prefix_words'] if w['text'] != '*')]]

    def patch(box):
        x0, y0, x1, y1 = map(int, box)
        return gray[y0:y1, x0:x1]

    anchors = []
    for slot in source_slots:
        owners = [p for p in primary_slots if p['code'] == slot['code'] and p['value'] == slot['value']
                  and _cash_overlap(p['code_word'], slot['code_word']) >= .5
                  and _cash_overlap(p['title_words'][0], slot['title_words'][0]) >= .5]
        if len(owners) == 1 and slot['mode'] is not None and slot['mode'] != owners[0]['mode']:
            return None
        word = slot['marker_word']
        if word is None:
            continue
        if len(owners) != 1:
            return None
        pixels = patch(slot['box'])
        if not pixels.size:
            return None
        _, mask = cv2.threshold(pixels, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        count, _labels, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:
            return None
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        marker = _cash_box(word)
        owned = [i for i, center in enumerate(centers) if i > 0
                 and marker[0] <= slot['box'][0] + center[0] <= marker[2]
                 and marker[1] <= slot['box'][1] + center[1] <= marker[3]]
        if owned != [largest]:
            return None
        center = [slot['box'][0] + centers[largest][0], slot['box'][1] + centers[largest][1]]
        if ([p['row'] for p in primary_slots if contains(p['box'], center)] != [owners[0]['row']]
                or any(contains(_cash_box(w), center) for w in negative_words)):
            return None
        x, y, width, height, _area = map(int, stats[largest])
        template = pixels[max(0, y-2):min(pixels.shape[0], y+height+2),
                          max(0, x-2):min(pixels.shape[1], x+width+2)]
        if min(template.shape) < 3 or np.std(template) == 0:
            return None
        anchors.append(dict(slot=slot, template=template, owner_row=owners[0]['row'], component_id=largest))
    if len(anchors) < 3 or len({a['owner_row'] for a in anchors}) != len(anchors):
        return None

    def match(box, references):
        window = patch(box)
        scores, centers, sizes = [], [], []
        for anchor in references:
            template = anchor['template']
            if any(a < b for a, b in zip(window.shape, template.shape)):
                return None
            correlations = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
            _min, score, _min_pos, pos = cv2.minMaxLoc(correlations)
            if not math.isfinite(score):
                return None
            scores.append(score)
            centers.append([box[0]+pos[0]+template.shape[1]/2, box[1]+pos[1]+template.shape[0]/2])
            sizes.append(min(template.shape))
        center = [median(p[0] for p in centers), median(p[1] for p in centers)]
        return dict(score=median(scores), center=center,
                    agrees=max(math.dist(p, center) for p in centers) <= median(sizes)/4)

    positives = [match(a['slot']['box'], [other for other in anchors if other is not a]) for a in anchors]
    if any(p is None or not p['agrees'] for p in positives):
        return None
    controls = [match(_cash_box(word), anchors) for word in negative_words]
    negatives = [p['score'] for p in controls if p is not None]
    if not negatives:
        return None
    low, high = min(p['score'] for p in positives), max(negatives)
    if low - high < .10:
        return None
    matches = []
    for slot in primary_slots:
        observed = match(slot['box'], anchors)
        if observed is not None and observed['agrees'] and observed['score'] >= low and observed['score'] >= high + .10:
            if ([p['row'] for p in primary_slots if contains(p['box'], observed['center'])] != [slot['row']]
                    or any(contains(_cash_box(w), observed['center']) for w in negative_words)):
                return None
            matches.append(dict(primary_row=slot['row'], slot_box=slot['box'], **observed))
    return dict(matches=matches, lowest_held_out_positive=low, strongest_local_negative=high,
        anchors=[dict(source_row=a['slot']['row'], primary_row=a['owner_row'],
                      marker_box=_cash_box(a['slot']['marker_word']), component_id=a['component_id']) for a in anchors],
        producer='same_receipt_original_pixel_template', confidence=None)


def recover_pixel_marker_tax_partition(receipt, source, primary_text, context):
    """Three-mode, complete qty1 table + calibrated stars + one whole partition.

    P owns every literal title/price and financial component. S owns only the
    distinct physical template stars. No blank, model category, product name,
    residual amount or OCR confidence establishes a missing tax marker.
    """
    rejected = {'accepted': False}
    if not isinstance(context, dict) or not isinstance(primary_text, str):
        return rejected
    source = _validated_evidence(source)
    if source is None:
        return rejected
    image, layout = context.get('image'), context.get('primary_layout')
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or source is None
            or context.get('source_identity') is None
            or context.get('source_identity') != _build_financial_source_identity(image, primary_text, source)
            or hashlib.sha256(image.tobytes()).hexdigest() != context.get('image_pixel_sha256')
            or not _valid_source_layout(layout, list(image.shape[:2]))
            or layout_blocks_sha256(layout) != context.get('primary_layout_sha256')):
        return rejected
    if any(w['page'] != 0 for w in [*layout, *source['source_layout']]):
        return rejected
    control_texts = [primary_text, source['merged_text'], *(tile['text'] for tile in source['tiles'])]
    if any(re.search(
        r'非課税|不課税|免税|\d[\d,]*非$|'
        r'(?<![\d.])0(?:\.0+)?%(?:対象|タイショウ|内税|外税|税額)|'
        r'(?:対象(?:額)?|タイショウ|内税|外税|税額)\(?0(?:\.0+)?%', _cash_compact(line))
        for text in control_texts
        for line in text.splitlines()):
        return rejected
    rows = _group_layout_rows(layout)
    summaries = _collect_direct_summary_owners(layout)
    source_summaries = _collect_direct_summary_owners(source['source_layout'])
    control_lines = [line for text in control_texts for line in text.splitlines()]
    control_lines += [''.join(w['text'] for w in row) for row in [*rows, *_group_layout_rows(source['source_layout'])]]
    for line in map(_cash_compact, control_lines):
        if (re.search(r'税率|税額|消費税', line) or re.fullmatch(
                r'\(?[A-Za-z]?(?:外税?|内税?)?\d+(?:\.\d+)?%(?:外税?|内税?)?(?:対象(?:額)?|タイショウ)?[¥￥]\d[\d,]*\)?', line)):
            if any(float(rate) not in (8, 10) for rate in re.findall(r'(?<![\d.])(\d+(?:\.\d+)?)%', line)):
                return rejected
    subtotal, total = (summaries['unique'][role] for role in ('subtotal', 'total'))
    components = summaries['owners']['rate_component']
    groups = {('8%', '外税'), ('10%', '外税'), ('10%', '内税')}
    roles = {(rate, mode, kind) for rate, mode in groups for kind in ('base', 'tax')}
    if (not subtotal or not total or len(components) != 6
            or {(o['rate'], o['mode'], o['kind']) for o in components} != roles
            or any(not _CASH_LITERAL_AMOUNT_RE.search(o['row_text'].removesuffix(')'))
                   for o in [subtotal, total, *components])):
        return rejected
    for rate, mode in groups:
        tags = {o['table_label'] for o in components if (o['rate'], o['mode']) == (rate, mode)}
        if len(tags) != 1:
            return rejected
    tags = [o['table_label'] for o in components if o['table_label'] is not None]
    if any(len({(o['rate'], o['mode']) for o in components if o['table_label'] == tag}) != 1 for tag in tags):
        return rejected
    values = {(o['rate'], o['mode'], o['kind']): o['value'] for o in components}
    for owner in [*summaries['owners']['tax'], *source_summaries['owners']['tax']]:
        if not _CASH_LITERAL_AMOUNT_RE.search(owner['row_text'].removesuffix(')')):
            return rejected
        if owner.get('rate'):
            matches = [(rate, mode) for rate, mode in groups
                       if rate == owner['rate'] and values[rate, mode, 'tax'] == owner['value']]
            if len(matches) != 1:
                return rejected
        elif owner['value'] not in {values['8%', '外税', 'tax'] + values['10%', '外税', 'tax'], values['10%', '内税', 'tax']}:
            return rejected
    expected_taxes = {(rate, mode): values[rate, mode, 'tax'] for rate, mode in groups if values[rate, mode, 'tax'] > 0}
    taxes = receipt.get('taxes') or []
    if (receipt.get('currency') != 'JPY' or receipt.get('subtotal') != subtotal['value']
            or receipt.get('total') != total['value'] or len(taxes) != len(expected_taxes)
            or any(not isinstance(t, dict) for t in taxes)
            or {(t.get('rate'), t.get('label')): t.get('amount') for t in taxes} != expected_taxes
            or sum(values[rate, mode, 'base'] for rate, mode in groups) != subtotal['value']
            or subtotal['value'] + values['8%', '外税', 'tax'] + values['10%', '外税', 'tax'] != total['value']
            or any(not _rate_base_tax_pair_is_valid(rate, values[rate, mode, 'base'], values[rate, mode, 'tax'], mode)
                   for rate, mode in groups)):
        return rejected
    slots = [s for s in _code_marker_slots(layout) if s['row'] < subtotal['row']]
    items = receipt.get('line_items') or []
    date_rows = [i for i, row in enumerate(rows[:slots[0]['row']] if slots else [])
                 if _DATE_LINE_RE.search(''.join(w['text'] for w in row))
                 and _TIME_HHMM_RE.search(''.join(w['text'] for w in row))]
    counts = _LITERAL_TABLE_COUNT_RE.findall(_cash_compact(primary_text))
    count_rows = [(i, match[1]) for i, row in enumerate(rows)
                  if (match := _LITERAL_TABLE_COUNT_RE.fullmatch(_cash_compact(''.join(w['text'] for w in row))))]
    layout_counts = [count for _, count in count_rows]
    if (not slots or len(slots) != len(items) or len(slots) > 16
            or [s['row'] for s in slots] != list(range(slots[0]['row'], subtotal['row']))
            or len({s['code'] for s in slots}) != len(slots)
            or len(date_rows) != 1 or len(layout_counts) != 1 or (counts and layout_counts != counts)
            or int(layout_counts[0].replace(',', '')) != len(slots)):
        return rejected
    pretable = [''.join(w['text'] for w in row) for row in rows[date_rows[0]+1:slots[0]['row']]]
    if any(_LITERAL_PRICE_AMOUNT_RE.search(line) or _LITERAL_PRICE_QTY_RE.search(line)
           or _LITERAL_PRICE_DISCOUNT_RE.search(line) or _LITERAL_PRICE_PERCENT_RE.fullmatch(line) for line in pretable):
        return rejected
    legend = re.compile(r'\*印.{0,32}?軽減税率対象\(外8%\)商品です')
    legends = [i for i, row in enumerate(rows) if legend.fullmatch(_cash_compact(''.join(w['text'] for w in row)))]
    source_legend_rows = [_cash_compact(''.join(w['text'] for w in row))
                          for row in _group_layout_rows(source['source_layout'])
                          if re.search(r'[*＊※]印', _cash_compact(''.join(w['text'] for w in row)))]
    if (len(legends) != 1 or legends[0] <= total['row']
            or len(source_legend_rows) != 1 or not legend.fullmatch(source_legend_rows[0])):
        return rejected
    if any(re.search(r'\S(?:印|マーク)', _cash_compact(line)) and re.search(r'税|対象', _cash_compact(line))
           and not legend.fullmatch(_cash_compact(line)) for line in control_lines):
        return rejected
    indices = []
    for slot in slots:
        if (_LITERAL_PRICE_QTY_RE.search(slot['title']) or _LITERAL_PRICE_DISCOUNT_RE.search(slot['title'])
                or _LITERAL_PRICE_PERCENT_RE.fullmatch(slot['title'])):
            return rejected
        owners = [i for i, item in enumerate(items) if isinstance(item, dict)
                  and slot['value'] > 0 and not any(isinstance(item.get(field), bool) for field in ('qty', 'unit_price', 'total'))
                  and _cash_title_key(item.get('description')) == _cash_title_key(slot['title'])
                  and item.get('qty') == 1 and item.get('unit_price') == slot['value'] == item.get('total')
                  and item.get('discount') in (None, 0) and item.get('discount_rate') in (None, '')
                  and item.get('gross') is None
                  and item.get('_tax_category_locked') not in {'0%'}]
        if len(owners) != 1:
            return rejected
        indices.append(owners[0])
    if len(set(indices)) != len(items) or sum(s['value'] for s in slots) != subtotal['value']:
        return rejected
    source_slots = _code_marker_slots(source['source_layout'])
    pixels = _calibrated_pixel_stars(image, source_slots, slots)
    if pixels is None:
        return rejected
    reduced = ({s['row'] for s in slots if s['marker'] == '*'}
               | {m['primary_row'] for m in pixels['matches']}
               | {a['primary_row'] for a in pixels['anchors']})
    included = {s['row'] for s in slots if s['mode'] == '内'}
    if (included & reduced or sum(s['value'] for s in slots if s['row'] in included) != values['10%', '内税', 'base']
            or not reduced or any(s['marker'] in {'※', '＊'} for s in slots)):
        return rejected
    choices = [s for s in slots if s['row'] not in included | reduced]
    solutions = []
    # ponytail: at most 16 full owners; counted DP is the upgrade for larger baskets.
    for size in range(len(choices) + 1):
        for group in combinations(choices, size):
            external = {s['row'] for s in group}
            if sum(s['value'] for s in group) != values['10%', '外税', 'base']:
                continue
            if sum(s['value'] for s in slots if s['row'] not in included | external) != values['8%', '外税', 'base']:
                continue
            proposed = [''] * len(items)
            for slot, index in zip(slots, indices, strict=True):
                proposed[index] = '10%' if slot['row'] in included | external else '8%'
            if any(item.get('_tax_category_locked') is not None and item['_tax_category_locked'] != proposed[i]
                   for i, item in enumerate(items)):
                continue
            solutions.append((proposed, sorted(external)))
            if len(solutions) > 1:
                return rejected
    if len(solutions) != 1:
        return rejected
    proposed, external = solutions[0]
    changed = [i for i, item in enumerate(items) if item.get('tax_category') != proposed[i]]
    description_updates = []
    for slot, index in zip(slots, indices, strict=True):
        description = str(items[index].get('description') or '').strip()
        marker = slot['marker']
        if (marker in {'*', '#'} and description.startswith(marker)
                and _cash_compact(description) == marker + _cash_compact(slot['title'])):
            description_updates.append(dict(item_index=index, marker=marker,
                description=description.removeprefix(marker).lstrip()))
    return dict(accepted=bool(changed or description_updates), proposed=proposed, changed_item_indices=changed,
        description_updates=description_updates,
        mode='pixel_owned_complete_mixed_tax_partition', pixels=pixels,
        source_identity=context['source_identity'], primary_layout_sha256=context['primary_layout_sha256'],
        image_pixel_sha256=context['image_pixel_sha256'], printed_components=components,
        primary_owner_rows=[dict(row=s['row'], item_index=i, code=s['code'], title=s['title'], value=s['value'])
                            for s, i in zip(slots, indices, strict=True)],
        included_rows=sorted(included), external_rows=external, legend_row=legends[0], count_row=count_rows[0][0])
