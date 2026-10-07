"""Last-resort native-pixel reads for unresolved, independently owned fields.

Only an exact P/S code/title/barcode price conflict in a counted qty-one basket,
or an unreadable primary phone-row name competing with unowned header marks,
can acquire one crop. Model literals remain separate from OCR annotations.
Price admission closes the printed total/tax. The same crop may resolve an
already contested title with exact code/barcode ownership and observed candidates;
header admission requires one seller and resolved roles for every upper mark.
"""

import base64
import hashlib
import json
import os
import re
from copy import deepcopy
from dataclasses import asdict
from difflib import SequenceMatcher
from math import ceil, floor
from pathlib import Path
from statistics import median

import cv2
import numpy as np

from .llm import _openrouter_chat, sanitize_llm_response
from .ocr import _OCR_CACHE_DIR, _atomic_write_text, _ocr_cache_key, layout_blocks_sha256
from .patterns import _HEADER_PHONE_MERCHANT_RE
from .receipt_identity_payment import _DATE_LINE_RE, _TIME_HHMM_RE, _merchant_looks_invalid, _select_header_logo, _settlement_kind
from .receipt_location import _extract_japanese_phone_hint
from .receipt_projection import _collect_direct_summary_owners, _group_layout_rows, _layout_rows_are_local
from .receipt_supplemental_ocr import (
    _build_financial_source_identity, _cash_box, _cash_compact, _cash_title_key, _cash_overlap,
    _valid_source_layout, _validated_evidence, _normal_cache_identity_lock,
    _LITERAL_TABLE_COUNT_RE, _CASH_ITEM_DETAIL_RE,
)
from .receipt_totals import _has_balanced_printed_summary, _line_items_sum


VISION_CACHE_DIR = Path(_OCR_CACHE_DIR).parent / "receipt_vision"
DEFAULT_VISION_MODEL = "google/gemini-3.8-flash"
VISION_MODES = ('normal', 'cache_only', 'fresh')
_VISION_SYSTEM = """You read receipt pixels. Transcribe all visible lines exactly before extracting fields.
Return JSON with lines:[{text,role,role_evidence}], date_literal, date_iso, time_literal,
merchant, location, item_rows:[{printed_text,price_literal,marker_literal,quantity_lines,discount_lines}].
For header lines distinguish seller, branch, complex, operator, contact, and unknown using
visible labels and layout. Explain the visible evidence. Leave unresolved roles unknown.
Merchant is the seller; location is the distinct branch/site name if its role is supported.
Date_iso is YYYY-MM-DD only when the complete printed transaction date is readable.
Read digits only from the supplied pixels. Do not use arithmetic, calendar plausibility,
product knowledge, external knowledge, or inferred missing marks to repair faded digits.
Preserve ambiguous glyphs as ? in literal text and use null for an unreadable field.
Do not fabricate coordinates or OCR confidence. No prior OCR or expected answer is supplied.
"""
_VISION_USER = 'Transcribe this native original-pixel receipt region. Return only the requested JSON.'
VISION_PROFILE_FINGERPRINT = hashlib.sha256((_VISION_SYSTEM + '\0' + _VISION_USER + '\0native-png-v2').encode()).hexdigest()


def configured_vision_model():
    return os.environ.get("RECEIPT_VISION_MODEL", DEFAULT_VISION_MODEL).strip().removeprefix("openrouter/")


def _cache_prefix(image, model):
    return f'{_ocr_cache_key(image)}.{VISION_PROFILE_FINGERPRINT}.{hashlib.sha256(model.encode()).hexdigest()}.'


def vision_cache_files(image):
    """Fingerprint only captures bound to independently loaded pixels/model/profile."""
    model = configured_vision_model()
    return sorted(VISION_CACHE_DIR.glob(_cache_prefix(image, model) + '*.json')) if model else []


def _coded_packets(layout, boundary):
    packets = []
    rows = _group_layout_rows(layout)
    for position, row in enumerate(rows):
        if min(_cash_box(word)[1] for word in row) >= boundary:
            break
        code = _cash_compact(row[0]['text'])
        if not re.fullmatch(r'\d{3,6}', code):
            continue
        prices = [(i, match) for i in range(1, len(row))
                  if (match := re.fullmatch(r'[¥￥#](\d[\d,]*)', _cash_compact(''.join(word['text'] for word in row[i:]))))]
        if len(prices) != 1:
            continue
        end, price = prices[0]
        title = ''.join(word['text'] for word in row[1:end])
        title = re.sub(r'^(?:内|外)?[*＊※#★]+', '', _cash_compact(title))
        if not title or not any(char.isalpha() for char in title) or position == 0:
            continue
        previous = rows[position-1]
        barcode = _cash_compact(''.join(word['text'] for word in previous))
        if not re.fullmatch(r'\d{10,14}(?:JAN)?', barcode) or not _layout_rows_are_local(previous, row):
            barcode = None
        left, top = min(_cash_box(w)[0] for w in row), min(_cash_box(w)[1] for w in row)
        right, bottom = max(_cash_box(w)[2] for w in row), max(_cash_box(w)[3] for w in row)
        packets.append(dict(code=code, title=title, price=int(price[1].replace(',', '')),
            barcode=barcode, words=row, preceding=previous if barcode else row,
            bounds=dict(bbox=[[left, top], [right, top], [right, bottom], [left, bottom]])))
    return packets


def _price_plan(receipt, primary_text, primary, source, image_shape):
    """Select by exact ownership conflict, never by a desired residual price."""
    items = receipt.get('line_items') or []
    if (not items or any(item.get('qty') != 1 or item.get('discount') or item.get('discount_rate')
                         or item.get('unit_price') != item.get('total') for item in items)
            or not _has_balanced_printed_summary(receipt, primary_text)
            or _settlement_kind(receipt.get('total'), primary_text, receipt.get('amount_paid')) in (None, 'ambiguous')
            or min(abs(_line_items_sum(receipt) - float(receipt.get(field) or 0))
                   for field in ('subtotal', 'total')) <= .01):
        return None
    summary = _collect_direct_summary_owners(primary)['unique']['subtotal']
    if summary is None:
        return None
    boundary = min(_cash_box(primary[i])[1] for i in summary['row_indices'])
    packets = _coded_packets(primary, boundary)
    secondary = _coded_packets(source['source_layout'], boundary)
    counts = [_LITERAL_TABLE_COUNT_RE.findall(_cash_compact(text)) for text in
              (primary_text, source['merged_text'])]
    if (len(packets) != len(items) or len(secondary) != len(items)
            or [row['code'] for row in packets] != [row['code'] for row in secondary]
            or len({row['code'] for row in packets}) != len(items)
            or any(len(values) != 1 or int(values[0].replace(',', '')) != len(items) for values in counts)
            or any(item['total'] != row['price'] for item, row in zip(items, packets))):
        return None
    for layout, owners in ((primary, packets), (source['source_layout'], secondary)):
        start = min(_cash_box(word)[1] for word in owners[0]['preceding'])
        body = '\n'.join(_cash_compact(''.join(word['text'] for word in row))
                         for row in _group_layout_rows(layout)
                         if start <= min(_cash_box(word)[1] for word in row) < boundary)
        if _CASH_ITEM_DETAIL_RE.search(body):
            return None
    conflicts = [(index, row) for index, (item, row, other) in enumerate(zip(items, packets, secondary))
                 if row['price'] != other['price'] and row['barcode'] and row['barcode'] == other['barcode']
                 and _cash_title_key(row['title']) == _cash_title_key(other['title']) == _cash_title_key(item['description'])]
    # ponytail: one exact qty-one conflict; retain complex/ambiguous baskets for review.
    if len(conflicts) != 1:
        return None
    index, owner = conflicts[0]
    title_conflicts = {i for i, (item, row, other) in enumerate(zip(items, packets, secondary))
                       if row['barcode'] and row['barcode'] == other['barcode']
                       and _cash_overlap(row['bounds'], other['bounds']) >= .5
                       and _cash_compact(row['title']) != _cash_compact(other['title'])
                       and _cash_compact(item['description']) in {_cash_compact(row['title']), _cash_compact(other['title'])}}
    first, last = max(0, min({index} | title_conflicts)-1), max({index} | title_conflicts)
    local = packets[first:last+1]
    top = max(0, min(_cash_box(word)[1] for word in local[0]['preceding']) - 2)
    bottom = boundary - 1 if last == len(packets)-1 else min(_cash_box(word)[1] for word in packets[last+1]['preceding']) - 1
    return dict(kind='price', roi=[0, top, image_shape[1], bottom], item_index=index,
        title_conflicts=sorted(title_conflicts), owners=[dict(item_index=i,
            **{key: row[key] for key in ('code', 'title', 'price', 'barcode')},
            title_candidates=[row['title'], secondary[i]['title']] if i in title_conflicts else [row['title']])
            for i, row in enumerate(local, first)])


def _phone_rows(rows):
    result = []
    for index, row in enumerate(rows):
        text = _cash_compact(''.join(word['text'] for word in row))
        match = _HEADER_PHONE_MERCHANT_RE.search(text)
        phone = _extract_japanese_phone_hint(text)
        if match and phone:
            name = re.sub(r'(?:TEL|電話|☎|[:：])+\s*$', '', match['merchant']).strip()
            result.append(dict(index=index, name=name, phone=re.sub(r'\D', '', phone)))
    return result


def _header_plan(receipt, primary_text, primary, source, image_shape):
    """Unreadable contact mark + competing upper names after logo abstention."""
    if _select_header_logo(source['merged_text'], source['source_layout'], primary_text,
                           current_merchant=receipt.get('merchant')) is not None:
        return None
    coded = [row for row in _group_layout_rows(primary)
             if re.fullmatch(r'\d{3,6}', _cash_compact(row[0]['text']))
             and re.search(r'[¥￥]\d', _cash_compact(''.join(word['text'] for word in row)))]
    if not coded:
        return None
    first = coded[0]
    top = min(_cash_box(word)[1] for word in first)
    p_rows = [row for row in _group_layout_rows(primary) if max(_cash_box(word)[3] for word in row) < top]
    s_rows = [row for row in _group_layout_rows(source['source_layout']) if max(_cash_box(word)[3] for word in row) < top]
    phones = [_phone_rows(rows) for rows in (p_rows, s_rows)]
    if any(len(rows) != 1 for rows in phones):
        return None
    p_phone, s_phone = (rows[0] for rows in phones)
    if (p_phone['phone'] != s_phone['phone'] or any(char.isalpha() for char in p_phone['name'])
            or _merchant_looks_invalid(s_phone['name']) or not any(char.isalpha() for char in s_phone['name'])
            or _cash_title_key(receipt.get('merchant')) == _cash_title_key(s_phone['name'])
            or p_phone['index'] < 1 or p_phone['index'] != s_phone['index']):
        return None
    texts = ['\n'.join(_cash_compact(''.join(word['text'] for word in row)) for row in rows)
             for rows in (p_rows, s_rows)]
    registrations = [re.findall(r'T\d{13}(?!\d)', text) for text in texts]
    dates = [[_cash_compact(match[0]) for match in _DATE_LINE_RE.finditer(text)] for text in texts]
    times = [_TIME_HHMM_RE.findall(text) for text in texts]
    if (any(len(values) != 1 for values in registrations + dates + times)
            or registrations[0] != registrations[1] or dates[0] != dates[1] or times[0] != times[1]):
        return None
    height = round(median(_cash_box(word)[3] - _cash_box(word)[1] for word in first))
    bottom = max(_cash_box(word)[3] for word in first) + max(1, height//2)
    summary = _collect_direct_summary_owners(primary)['unique']['subtotal']
    if summary:
        bottom = min(bottom, min(_cash_box(primary[i])[1] for i in summary['row_indices'])-1)
    branch_rows = [_cash_compact(''.join(word['text'] for word in row)) for row in p_rows[:p_phone['index']]]
    return dict(kind='header', roi=[0, 0, image_shape[1], min(image_shape[0], bottom)],
        merchant=s_phone['name'], phone=s_phone['phone'], registration=registrations[0][0],
        upper_count=p_phone['index'], upper_rows=branch_rows)


def _prepare_plan(receipt, primary_text, supplemental, pixel_context, model):
    if (not isinstance(pixel_context, dict) or receipt.get('document_type') != 'receipt'
            or receipt.get('currency') != 'JPY'):
        return None
    image, primary = pixel_context.get('image'), pixel_context.get('primary_layout')
    source = _validated_evidence(supplemental)
    if (source is None or not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3 or not isinstance(primary_text, str)
            or not _valid_source_layout(primary, list(image.shape[:2]))
            or any(word['page'] != 0 for word in primary)):
        return None
    identity = _build_financial_source_identity(image, primary_text, source)
    pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
    if (identity is None or pixel_context.get('source_identity') != identity
            or pixel_context.get('image_pixel_sha256') != pixel_hash
            or pixel_context.get('primary_layout_sha256') != layout_blocks_sha256(primary)):
        return None
    plan = (_price_plan(receipt, primary_text, primary, source, list(image.shape[:2]))
            or _header_plan(receipt, primary_text, primary, source, list(image.shape[:2])))
    if plan is None:
        return None
    x0, y0, x1, y1 = plan['roi'] = [floor(plan['roi'][0]), floor(plan['roi'][1]),
                                  ceil(plan['roi'][2]), ceil(plan['roi'][3])]
    if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
        return None
    crop = image[y0:y1, x0:x1]
    if crop.size == 0 or crop.shape[0]*crop.shape[1] > 60_000_000:
        return None
    success, encoded = cv2.imencode('.png', crop)
    if not success or encoded.nbytes > 18_000_000:
        return None
    png = encoded.tobytes()
    context = dict(schema_version=1, profile=VISION_PROFILE_FINGERPRINT, model=model, seed=42,
        temperature=0, source_identity=identity, primary_layout_sha256=layout_blocks_sha256(primary),
        image_pixel_sha256=pixel_hash, crop_pixel_sha256=hashlib.sha256(crop.tobytes()).hexdigest(),
        png_sha256=hashlib.sha256(png).hexdigest(), **plan)
    digest = hashlib.sha256(json.dumps(context, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()
    path = VISION_CACHE_DIR / (_cache_prefix(image, model) + digest + '.json')
    return context, png, path


def _admit(receipt, evidence, context):
    if not isinstance(evidence, dict):
        return None
    candidate = deepcopy(receipt)
    if context['kind'] == 'price':
        rows = evidence.get('item_rows')
        lines = evidence.get('lines')
        if (not isinstance(rows, list) or len(rows) != len(context['owners']) or not isinstance(lines, list)
                or not all(isinstance(line, dict) and isinstance(line.get('text'), str) for line in lines)):
            return None
        transcribed = [_cash_compact(line['text']) for line in lines]
        values, titles, owned_lines = [], [], []
        for row, owner in zip(rows, context['owners']):
            if (not isinstance(row, dict) or row.get('quantity_lines') != [] or row.get('discount_lines') != []):
                return None
            literal_lines = [_cash_compact(line) for line in str(row.get('printed_text') or '').splitlines() if line.strip()]
            printed = [line for line in literal_lines
                       if re.fullmatch(re.escape(owner['code']) + r'[*＊※★#]?(.*?)[¥￥](\d[\d,]*)', line)]
            if len(printed) != 1:
                return None
            match = re.fullmatch(re.escape(owner['code']) + r'[*＊※★#]?(.*?)[¥￥](\d[\d,]*)', printed[0])
            price = re.fullmatch(r'[¥￥]?(\d[\d,]*)', _cash_compact(row.get('price_literal')))
            title_matches = [literal for literal in owner['title_candidates']
                             if match and _cash_compact(match[1]) == _cash_compact(literal)]
            if (not match or not price or len(title_matches) != 1
                    or int(match[2].replace(',', '')) != int(price[1].replace(',', ''))
                    or transcribed.count(printed[0]) != 1
                    or literal_lines != ([owner['barcode']] if owner['barcode'] else []) + [printed[0]]
                    or (owner['barcode'] and transcribed.count(owner['barcode']) != 1)):
                return None
            values.append(int(price[1].replace(',', '')))
            titles.append(title_matches[0])
            owned_lines.extend(literal_lines)
        if transcribed != owned_lines:
            return None
        if (any(value <= 0 for value in values) or any(value != owner['price']
                for value, owner in zip(values, context['owners']) if owner['item_index'] != context['item_index'])):
            return None
        for value, title, owner in zip(values, titles, context['owners']):
            index = owner['item_index']
            if index == context['item_index']:
                candidate['line_items'][index]['unit_price'] = candidate['line_items'][index]['total'] = value
            if index in context['title_conflicts']:
                candidate['line_items'][index]['description'] = title
        if abs(_line_items_sum(candidate) - float(candidate.get('subtotal') or 0)) > .01:
            return None
    else:
        lines = evidence.get('lines')
        if (not isinstance(lines, list) or not lines or not all(isinstance(row, dict)
                and isinstance(row.get('text'), str) and row['text'].strip() for row in lines)):
            return None
        upper = lines[:context['upper_count']]
        seller = lines[context['upper_count']:context['upper_count']+1]
        if (len(seller) != 1 or seller[0].get('role') != 'seller'
                or any(row.get('role') not in ('complex', 'branch', 'operator') for row in upper)
                or any(not isinstance(row.get('role_evidence'), str) or not row['role_evidence'].strip()
                       for row in upper + seller)
                or any(row.get('role') == 'seller' for row in lines if row is not seller[0])
                or _cash_title_key(evidence.get('merchant')) != _cash_title_key(context['merchant'])
                or context['merchant'] not in _cash_compact(seller[0].get('text'))
                or re.sub(r'\D', '', _extract_japanese_phone_hint(str(seller[0].get('text') or ''))) != context['phone']
                or not any(context['registration'] in _cash_compact(row.get('text')) for row in lines)):
            return None
        if any(SequenceMatcher(None, _cash_title_key(row['text']), _cash_title_key(observed)).ratio() < .75
               or re.sub(r'\D', '', row['text']) != re.sub(r'\D', '', observed)
               for row, observed in zip(upper, context['upper_rows'], strict=True)):
            return None
        # Branch literals must remain the separately observed primary header row.
        if any(_cash_compact(row.get('text')) not in context['upper_rows'] for row in upper if row.get('role') == 'branch'):
            return None
        candidate['merchant'] = context['merchant']
    return candidate


def recover_with_vision(receipt, primary_text, supplemental, pixel_context, *, mode='normal'):
    """One bounded late attempt; a miss/error/failed invariant preserves the receipt."""
    if mode not in VISION_MODES:
        raise ValueError('vision_mode must be normal, cache_only, or fresh')
    model = configured_vision_model()
    try:
        prepared = _prepare_plan(receipt, primary_text, supplemental, pixel_context, model) if model else None
    except (KeyError, TypeError, ValueError, OverflowError, cv2.error):
        return None, None
    if prepared is None:
        return None, None
    context, png, path = prepared
    metadata = dict(kind=context['kind'], status='abstained', attempts=0, source=None,
        model=model, request_fingerprint=path.stem, context=context)

    def acquire():
        if mode != 'fresh' and path.exists():
            if path.stat().st_size > 512_000:
                raise ValueError('Oversized vision cache capture')
            capture = json.loads(path.read_text(encoding='utf-8'))
            if (not isinstance(capture, dict) or set(capture) != {'context', 'evidence', 'raw_content', 'response_usage'}
                    or capture['context'] != context or not isinstance(capture['raw_content'], str)
                    or not isinstance(capture['response_usage'], dict) or capture['response_usage'].get('finish_reason') != 'stop'
                    or json.loads(sanitize_llm_response(capture['raw_content'])) != capture['evidence']):
                raise ValueError('Vision cache context mismatch')
            metadata.update(source='cache', response_usage=capture.get('response_usage'))
            return capture
        if mode == 'cache_only':
            metadata['reason'] = 'cache_miss'
            return None
        if not os.environ.get('OPENROUTER_API_KEY'):
            metadata['reason'] = 'missing_openrouter_key'
            return None
        metadata.update(attempts=1, source='api')
        response = _openrouter_chat(model, [dict(role='system', content=_VISION_SYSTEM),
            dict(role='user', content=[dict(type='text', text=_VISION_USER),
                dict(type='image_url', image_url=dict(url='data:image/png;base64,' + base64.b64encode(png).decode('ascii'), detail='high'))])],
            max_tokens=4096, timeout=60, max_retries=0,
            extra_body=dict(reasoning=dict(effort='minimal', exclude=True),
                provider=dict(order=['Google AI Studio'], allow_fallbacks=False, require_parameters=True)))
        metadata['response_usage'] = {key: value for key, value in asdict(response).items() if key != 'content'}
        if response.finish_reason != 'stop':
            raise ValueError('Vision response did not finish normally')
        evidence = json.loads(sanitize_llm_response(response.content))
        capture = dict(context=context, evidence=evidence, raw_content=response.content, response_usage=metadata['response_usage'])
        if mode == 'normal':
            _atomic_write_text(path, json.dumps(capture, ensure_ascii=False, indent=2))
        return capture

    try:
        if mode == 'normal':
            with _normal_cache_identity_lock(path):
                capture = acquire()
        else:
            capture = acquire()
        candidate = _admit(receipt, capture.get('evidence'), context) if capture else None
        if candidate is not None and candidate != receipt:
            metadata['status'] = 'accepted'
            return candidate, metadata
        metadata.setdefault('reason', 'field_invariant_failed')
    except Exception as error:
        metadata.update(status='unavailable', reason=type(error).__name__)
    return None, metadata
