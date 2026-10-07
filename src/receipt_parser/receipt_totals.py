"""Receipt total and tax-summary arithmetic helpers."""

import re
from math import isfinite

from .receipt_financial import (
    _bare_number_tax_summary_entries,
    _interleaved_rate_tax_summary_entries,
    _jpy_summary_amount_options,
    _rate_base_tax_pair_is_valid,
    extract_financial_totals,
    extract_rate_bases,
    normalize_tax_label,
    normalize_tax_rate,
)


def _canonical_subtotal_from_taxes(extracted) -> float | None:
    total = extracted.get("total")
    taxes = extracted.get("taxes") or []
    if total is None or not taxes:
        return None
    tax_sum = _sum_taxable_amounts(taxes)
    if not tax_sum:
        return None
    return float(total) - float(tax_sum)


def _sum_taxable_amounts(taxes) -> float:
    """Sum added tax for explicit mixed groups; otherwise preserve legacy sums."""
    entries = [
        (t, float(t.get("amount") or 0))
        for t in (taxes or [])
        if isinstance(t, dict)
        and t.get("rate") != "0%"
        and t.get("amount") is not None
    ]
    labels = {tax.get("label") for tax, amount in entries if amount > 0}
    mixed = (
        labels == {"内税", "外税"}
        and all(isfinite(amount) and amount >= 0 for _tax, amount in entries)
        and all(amount <= 0 or tax.get("label") in {"内税", "外税"} for tax, amount in entries)
    )
    return sum(
        amount for tax, amount in entries
        if not mixed or tax.get("label") == "外税"
    )


def _has_balanced_printed_summary(extracted, unified_text) -> bool:
    """Protect a printed total and matching tax amounts when net + tax closes."""
    printed = extract_financial_totals(unified_text)
    try:
        total = float(extracted["total"])
        subtotal = float(extracted["subtotal"])
        printed_total = float(printed["total"])
        current_taxes = extracted.get("taxes") or []
        tax_sum = _sum_taxable_amounts(current_taxes)
        current_pairs = sorted((str(tax.get("rate")), float(tax["amount"])) for tax in current_taxes if float(tax["amount"]) > 0)
        printed_pairs = sorted((str(tax.get("rate")), float(tax["amount"])) for tax in printed.get("taxes") or [] if float(tax["amount"]) > 0)
    except (KeyError, TypeError, ValueError):
        return False
    return (total > 0 and tax_sum > 0 and current_pairs == printed_pairs
            and abs(printed_total - total) <= 0.01
            and abs(subtotal + tax_sum - total) <= 0.01)


def _line_items_sum(extracted) -> float:
    return sum(
        float(item.get("total") or 0)
        for item in (extracted.get("line_items") or [])
        if isinstance(item, dict)
    )


def _yen_after_summary_label(lines, label_idx: int, lookahead: int = 3) -> float | None:
    """Return a nearby standalone yen amount before tender/receipt details."""
    for line in lines[label_idx + 1:min(len(lines), label_idx + 1 + lookahead)]:
        match = re.fullmatch(r'[¥￥]\s*([\d,]+)\s*[\)）]?', line)
        if match:
            return float(match.group(1).replace(',', ''))
        if re.search(r'お預り|お釣|ポイント|伝票|レシート', line):
            break
    return None


def _printed_amount_targets(extracted, unified_text, *, include_rate_bases: bool = True) -> list[float]:
    """OCR-visible summary/rate-base amounts that can balance item cleanup."""
    targets = [
        float(value)
        for value in (
            extracted.get("subtotal"),
            extracted.get("total"),
            _canonical_subtotal_from_taxes(extracted),
        )
        if value is not None and float(value or 0) > 0
    ]
    lines = [line.strip() for line in unified_text.split('\n')]
    total_label = chr(0x5408) + chr(0x8A08)
    summary_labels = {
        chr(0x8A08),
        chr(0x5C0F) + chr(0x8A08),
        total_label,
        chr(0x7DCF) + total_label,
    }
    for idx, line in enumerate(lines):
        compact_line = re.sub(r'\s+', '', line)
        if compact_line not in summary_labels:
            continue
        if re.search(r'税|対象|ポイント', "\n".join(lines[max(0, idx - 2):idx + 1])):
            continue
        for following in lines[idx + 1:min(len(lines), idx + 4)]:
            if re.search(r'お預り|お釣|釣銭|おつり|ポイント', following):
                break
            amount_m = re.fullmatch(r'[¥￥]\s*([\d,]+)', following)
            if amount_m:
                targets.append(float(amount_m.group(1).replace(',', '')))
                break
    if include_rate_bases:
        targets.extend(
            float(base)
            for base in extract_rate_bases(unified_text).values()
            if base is not None and float(base) > 0
        )
    return targets


def _drop_unprinted_small_target_only_taxes(extracted, unified_text):
    """Omit tiny tax rows when OCR prints only a target base, not a tax amount."""
    taxes = [tax for tax in (extracted.get("taxes") or []) if isinstance(tax, dict)]
    if not taxes:
        return
    rate_bases = extract_rate_bases(unified_text)
    if not rate_bases:
        return
    printed_tax_rates = {
        normalize_tax_rate(str(tax.get("rate") or ""))
        for tax in (extract_financial_totals(unified_text).get("taxes") or [])
        if isinstance(tax, dict) and (tax.get("amount") or 0) > 0
    }
    has_standalone_tax_label = any(
        re.fullmatch(r'消費税(?:等|額)?', line.strip())
        for line in unified_text.split('\n')
    )
    kept = []
    changed = False
    for tax in taxes:
        rate = normalize_tax_rate(str(tax.get("rate") or ""))
        amount = float(tax.get("amount") or 0)
        if (
            rate in rate_bases
            and rate not in printed_tax_rates
            and 0 < amount <= 1
            and not has_standalone_tax_label
        ):
            changed = True
            continue
        kept.append(tax)
    if not changed:
        return
    extracted["taxes"] = kept
    total = extracted.get("total")
    try:
        total_f = float(total) if total is not None else None
    except (TypeError, ValueError):
        total_f = None
    if total_f is not None:
        tax_sum = _sum_taxable_amounts(kept)
        if 0 <= tax_sum <= total_f:
            extracted["subtotal"] = total_f - tax_sum


def _restore_bare_number_tax_summary(extracted, unified_text):
    """Restore printed tax pairs; one unique gross pair may own one purpose item."""
    lines = [line.strip() for line in unified_text.split('\n')]
    bare_entries = _bare_number_tax_summary_entries(lines)
    interleaved_entries = _interleaved_rate_tax_summary_entries(lines)
    mixed_modes = {
        label for _rate, kind, amount, label in interleaved_entries
        if kind == "tax" and amount > 0 and label in {"内税", "外税"}
    }
    taxes = list(dict.fromkeys(
        (rate, value)
        for rate, kind, value in bare_entries
        if kind == "tax" and value > 0
    ))
    taxes.extend(
        (rate, value)
        for rate, kind, value, _label in interleaved_entries
        if kind == "tax" and value > 0 and (rate, value) not in taxes
    )
    total = extracted.get("total")
    try:
        total_f = float(total) if total is not None else None
    except (TypeError, ValueError):
        total_f = None
    try:
        subtotal_f = float(extracted["subtotal"])
    except (KeyError, TypeError, ValueError):
        subtotal_f = None

    if {"内税", "外税"} <= mixed_modes:
        grouped_bases = {
            (rate, label): value
            for rate, kind, value, label in interleaved_entries
            if kind == "base"
        }
        parsed_taxes = [
            {"rate": rate, "label": label, "amount": value}
            for rate, kind, value, label in interleaved_entries
            if kind == "tax" and value > 0 and label in {"内税", "外税"}
        ]
        subtotal_expected = sum(grouped_bases.values())
        external_expected = sum(
            tax["amount"] for tax in parsed_taxes if tax["label"] == "外税"
        )
        parsed_rates = {rate for rate, _label in grouped_bases}
        if (
            total_f is not None
            and subtotal_f is not None
            and abs(subtotal_f - subtotal_expected) <= 2
            and abs(total_f - subtotal_expected - external_expected) <= 2
        ):
            kept = [
                tax for tax in (extracted.get("taxes") or [])
                if not isinstance(tax, dict) or tax.get("rate") not in parsed_rates
            ]
            extracted["taxes"] = kept + parsed_taxes
            extracted["subtotal"] = subtotal_expected
        # Mixed replacement requires independent OCR ownership of both modes.
        return

    entries = bare_entries + [
        (rate, kind, value)
        for rate, kind, value, _label in interleaved_entries
    ]
    current_taxes: list[tuple[str, float]] = []
    preserved_nontaxable: list[dict] = []
    for tax in (extracted.get("taxes") or []):
        if not isinstance(tax, dict):
            continue
        try:
            amount = float(tax["amount"])
        except (KeyError, TypeError, ValueError):
            continue
        if normalize_tax_rate(str(tax.get("rate") or "")) == "0%":
            if isfinite(amount) and amount >= 0:
                preserved_nontaxable.append(tax)
            continue
        if amount > 0:
            current_taxes.append((normalize_tax_rate(str(tax.get("rate") or "")), amount))
    current_by_rate = dict(current_taxes)
    parsed_rates = {rate for rate, _value in taxes}
    current_is_complete = (
        len(current_by_rate) == len(current_taxes) >= 2
        and total_f is not None
        and subtotal_f is not None
        and abs(subtotal_f + sum(current_by_rate.values()) - total_f) <= 2
    )
    if current_is_complete:
        tax_sum = sum(current_by_rate.values())
        printed_rate_amounts = True
        printed_external_rates: set[str] = set()
        for rate, amount in current_by_rate.items():
            rate_number = re.escape(rate.rstrip("%"))
            rate_has_external_label = any(
                re.search(rf'(?<![\d.]){rate_number}\s*[%％](?![\d.])', line)
                and re.search(r'外税|外枠', line)
                for line in lines
            )
            amount_is_printed = False
            amount_is_external = False
            for idx, line in enumerate(lines):
                if not re.search(rf'(?<![\d.]){rate_number}\s*[%％](?![\d.])', line):
                    continue
                if re.search(r'対象|タイショウ', line):
                    continue
                label_idx = idx
                external_label = bool(re.search(r'外税|外枠', line))
                if not re.search(r'外税|内税|消費税|税額|^\s*税\s*$', line):
                    label_idx = next(
                        (
                            candidate_idx
                            for candidate_idx in range(idx + 1, min(len(lines), idx + 3))
                            if lines[candidate_idx]
                        ),
                        idx,
                    )
                    if not re.fullmatch(r'\s*(?:消費)?税(?:額|等)?\s*', lines[label_idx]):
                        continue
                    external_label = rate_has_external_label
                elif re.search(
                    rf'{rate_number}\s*[%％]\s*税(?:額)?(?![一-龥])',
                    line,
                ):
                    external_label = rate_has_external_label
                for candidate in lines[label_idx:min(len(lines), label_idx + 3)]:
                    if any(
                        abs(value - amount) <= 2
                        for value in _jpy_summary_amount_options(candidate)
                    ):
                        amount_is_printed = True
                        amount_is_external = external_label
                        break
                if amount_is_printed:
                    break
            if amount_is_external:
                printed_external_rates.add(rate)
            if not amount_is_printed:
                printed_rate_amounts = False
        unmatched_amounts = list(current_by_rate.values())
        for line in lines:
            options = _jpy_summary_amount_options(line)
            match_idx = next(
                (
                    idx
                    for idx, amount in enumerate(unmatched_amounts)
                    if any(abs(value - amount) <= 2 for value in options)
                ),
                None,
            )
            if match_idx is not None:
                unmatched_amounts.pop(match_idx)
            if not unmatched_amounts:
                break
        rate_bases = {}
        for rate, base in extract_rate_bases(unified_text).items():
            try:
                rate_bases[normalize_tax_rate(str(rate))] = float(base)
            except (TypeError, ValueError):
                continue
        bare_rate_summary_corroborates = (
            parsed_rates < set(current_by_rate)
            and not unmatched_amounts
            and set(current_by_rate) <= set(rate_bases)
            and all(
                _rate_base_tax_pair_is_valid(rate, rate_bases[rate], amount)
                for rate, amount in current_by_rate.items()
            )
        )
        printed_tax_total = any(
            abs(value - tax_sum) <= 2
            for idx, line in enumerate(lines)
            if re.search(r'(?:消費)?税合計', line)
            for candidate in lines[idx:min(len(lines), idx + 4)]
            for value in _jpy_summary_amount_options(candidate)
        )
        current_is_complete = (
            printed_rate_amounts
            or printed_tax_total
            or bare_rate_summary_corroborates
        )
        if current_is_complete and printed_external_rates:
            for tax in extracted.get("taxes") or []:
                if not isinstance(tax, dict):
                    continue
                rate = normalize_tax_rate(str(tax.get("rate") or ""))
                try:
                    amount = float(tax.get("amount") or 0)
                except (TypeError, ValueError):
                    continue
                if (
                    rate in printed_external_rates
                    and abs(amount - current_by_rate[rate]) <= 2
                ):
                    tax["label"] = "外税"
    if current_is_complete and parsed_rates < set(current_by_rate):
        return
    fallback_pair = None
    owned_amounts = set()
    if not taxes and total_f is not None:
        summary_lines = [line for line in lines if line]
        amounts = set()
        for idx, line in enumerate(summary_lines):
            if not re.fullmatch(r'[¥￥]?\s*(?:\d{1,3}(?:,\d{3})*|\d{1,6})\s*', line):
                continue
            for value in _jpy_summary_amount_options(line):
                if 0 < value <= total_f * 0.25:
                    amounts.add(value)
                    if any(re.fullmatch(r'消費税(?:額)?(?:等)?', neighbor)
                           for neighbor in summary_lines[max(0, idx - 1):idx] + summary_lines[idx + 1:idx + 2]):
                        owned_amounts.add(value)
        rates = {
            normalize_tax_rate(match.group(1) + "%")
            for line in lines
            if (match := re.fullmatch(r'(\d+(?:\.0+)?)\s*[%％]\s*[（(]?(?:税込|税抜)[^¥￥\d]*', line))
        }
        pairs = {(rate, amount) for rate in rates for amount in amounts
                 if _rate_base_tax_pair_is_valid(rate, total_f, amount, mode="内税")}
        if len(pairs) == 1:
            fallback_pair = next(iter(pairs))
            taxes.append(fallback_pair)
    if not taxes:
        return
    tax_sum = sum(value for _rate, value in taxes)
    item_sum = _line_items_sum(extracted)
    subtotal = total_f - tax_sum if total_f is not None and total_f >= tax_sum else extracted.get("subtotal")
    label = normalize_tax_label(
        "内税",
        unified_text,
        subtotal=subtotal,
        total=total_f,
        tax_sum=tax_sum,
        items_sum=item_sum or None,
    )
    extracted["taxes"] = [
        {"rate": rate, "label": normalize_tax_label(
            label, unified_text, subtotal=subtotal, total=total_f,
            tax_sum=tax_sum, items_sum=item_sum or None, rate=rate, amount=value,
        ), "amount": value}
        for rate, value in taxes
    ] + preserved_nontaxable
    if current_is_complete:
        for tax in extracted["taxes"]:
            rate = normalize_tax_rate(str(tax.get("rate") or ""))
            if (
                rate in printed_external_rates
                and abs(float(tax.get("amount") or 0) - current_by_rate[rate]) <= 2
            ):
                tax["label"] = "外税"
    if total_f is not None and total_f >= tax_sum:
        extracted["subtotal"] = total_f - tax_sum
    if fallback_pair is not None and label == "内税" and not preserved_nontaxable:
        # Local import avoids the item-repair -> projection -> totals cycle.
        from .receipt_item_repair import _single_formal_receipt_purpose_description

        purpose = _single_formal_receipt_purpose_description(unified_text)
        rate, amount = fallback_pair
        items = extracted.get("line_items") or []
        gross_tokens = [re.fullmatch(
            r'(?:(?:金額|(?:御|お)?領収金額|(?:総)?合計|税込金額|'
            r'\d{2,4}年\s*\d{1,2}月\s*\d{1,2}日)\s*)?'
            r'[¥￥]\s*(\d{1,3}(?:,\d{3})*|\d{1,9})\s*', line)
            for line in lines if re.search(r'[¥￥]', line)]
        printed_grosses = {float(match.group(1).replace(',', '')) for match in gross_tokens
                           if match and float(match.group(1).replace(',', '')) > total_f * 0.25}
        item_markers = any(
            re.fullmatch(r'[¥￥]?\s*\d[\d,]*\s*(?:軽|外|内|除|非|[*＊※xX])', line)
            or ('但' in line and '代' in line and re.search(r'[%％]|[*＊※]', line))
            for line in lines
        )
        rate_bases = extract_rate_bases(unified_text)
        if (purpose and len(items) == 1 and isinstance(items[0], dict)
                and amount in owned_amounts and printed_grosses == {total_f}
                and all(gross_tokens) and not item_markers
                and _line_items_sum(extracted) == total_f
                and re.sub(r'\s+', '', str(items[0].get("description") or "")) in {purpose, purpose + "代"}
                and items[0].get("_tax_category_locked") in (None, rate)
                and not re.search(r'外税|外枠|非課税|免税|不課税|区分外', unified_text)
                and all(base is None or base <= 0 or (key == rate and abs(base - total_f) <= 2)
                        for key, base in rate_bases.items())):
            items[0]["tax_category"] = rate


def _items_plus_tax_matches_total(extracted, tolerance: float = 5) -> bool:
    total = extracted.get("total")
    if total is None:
        return False
    item_sum = _line_items_sum(extracted)
    tax_sum = _sum_taxable_amounts(extracted.get("taxes") or [])
    return item_sum > 0 and tax_sum > 0 and abs(item_sum + tax_sum - float(total)) <= tolerance


def _prefer_printed_item_sum_total_when_balanced(extracted, unified_text):
    """Use a printed item sum without replacing a balanced external-tax summary."""
    items = [item for item in (extracted.get("line_items") or []) if isinstance(item, dict)]
    if len(items) < 2:
        return
    item_sum = sum(float(item.get("total") or 0) for item in items)
    if item_sum <= 0:
        return
    current_total = extracted.get("total")
    try:
        current_total_f = float(current_total) if current_total is not None else None
    except (TypeError, ValueError):
        current_total_f = None
    if current_total_f is not None and abs(current_total_f - item_sum) <= 2:
        return
    if current_total_f is not None and _items_plus_tax_matches_total(extracted):
        return
    if _has_balanced_printed_summary(extracted, unified_text):
        return
    printed_financials = extract_financial_totals(unified_text)
    printed_total = printed_financials.get("total")
    printed_tax_sum = _sum_taxable_amounts(printed_financials.get("taxes") or [])
    if (
        current_total_f is not None
        and printed_total is not None
        and abs(float(printed_total) - current_total_f) <= 2
        and printed_tax_sum > 0
        and abs(item_sum + printed_tax_sum - current_total_f) <= 2
    ):
        return
    printed_amounts = [
        float(m.group(1).replace(',', ''))
        for m in re.finditer(r'[¥￥]\s*([\d,]+)\s*-?', unified_text)
    ]
    if not any(abs(amount - item_sum) <= 2 for amount in printed_amounts):
        return

    lines = [line.strip() for line in unified_text.split('\n') if line.strip()]

    subtotal_label = chr(0x5C0F) + chr(0x8A08)
    total_label = chr(0x5408) + chr(0x8A08)
    total_head, total_tail = total_label
    for idx, line in enumerate(lines):
        vm = re.fullmatch(r'[¥￥]\s*([\d,]+)\s*[\)）]?', line)
        if not vm or abs(float(vm.group(1).replace(',', '')) - item_sum) > 2:
            continue
        prior = "\n".join(lines[max(0, idx - 3):idx])
        if subtotal_label not in re.sub(r'\s+', '', prior):
            continue
        saw_external_tax = False
        for j in range(idx + 1, min(len(lines), idx + 14)):
            if "外税" in lines[j]:
                saw_external_tax = True
            if not saw_external_tax:
                continue
            compact_line = re.sub(r'\s+', '', lines[j])
            if compact_line == total_label or (
                lines[j] == total_head and j + 1 < len(lines) and lines[j + 1] == total_tail
            ):
                label_idx = j + 1 if lines[j] == total_head else j
                printed_total = _yen_after_summary_label(lines, label_idx)
                if printed_total is not None and printed_total > item_sum + 2:
                    return

    old_total = current_total_f
    extracted["total"] = item_sum
    amount_paid = extracted.get("amount_paid")
    try:
        amount_paid_f = float(amount_paid) if amount_paid is not None else None
    except (TypeError, ValueError):
        amount_paid_f = None
    if amount_paid_f is None or (old_total is not None and abs(amount_paid_f - old_total) <= 5):
        extracted["amount_paid"] = item_sum
    tax_sum = _sum_taxable_amounts(extracted.get("taxes") or [])
    if tax_sum > 0 and item_sum >= tax_sum:
        extracted["subtotal"] = item_sum - tax_sum


def _restore_printed_summary_total_when_tax_balanced(extracted, unified_text):
    """Use explicit 小計/合計 summary labels when they balance with tax lines."""
    if _has_balanced_printed_summary(extracted, unified_text):
        return
    taxes = [tax for tax in (extracted.get("taxes") or []) if isinstance(tax, dict)]
    tax_sum = _sum_taxable_amounts(taxes)
    if tax_sum <= 0:
        return
    lines = [line.strip() for line in unified_text.split('\n')]

    printed_subtotal = None
    printed_total = None
    total_candidates: list[float] = []
    for idx, line in enumerate(lines):
        if (
            printed_subtotal is None
            and (
                re.fullmatch(r'小\s*計', line)
                or (line == "小" and idx + 1 < len(lines) and lines[idx + 1] == "計")
            )
        ):
            label_idx = idx + 1 if line == "小" else idx
            printed_subtotal = _yen_after_summary_label(lines, label_idx)
            continue
        if (
            printed_total is None
            and (
                re.fullmatch(r'合\s*計', line)
                or (line == "合" and idx + 1 < len(lines) and lines[idx + 1] == "計")
            )
        ):
            label_idx = idx + 1 if line == "合" else idx
            printed_total = _yen_after_summary_label(lines, label_idx)
            if printed_total is not None:
                total_candidates.append(printed_total)
            continue

    labels: list[str] = []
    values: list[float] = []
    in_summary = False
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if not in_summary:
            if re.fullmatch(r'小\s*計', line) or (
                line == "小" and idx + 1 < len(lines) and lines[idx + 1] == "計"
            ):
                in_summary = True
            else:
                idx += 1
                continue
        if re.search(r'レシート|お買上点数|店No|印は|登録番号', line):
            break
        vm = re.fullmatch(r'[¥￥]\s*([\d,]+)\s*[\)）]?', line)
        if vm:
            values.append(float(vm.group(1).replace(',', '')))
            idx += 1
            continue
        if line == "計":
            idx += 1
            continue
        label = line
        if line in {"小", "合"} and idx + 1 < len(lines) and lines[idx + 1] == "計":
            label = line + "計"
            idx += 1
        if re.search(r'小\s*計|合\s*計|現\s*計|税率|対象額|税額|外税|内税|消費税|お預り|お釣|釣銭', label):
            labels.append(label)
        idx += 1

    for label, value in zip(labels, values):
        if printed_subtotal is None and re.search(r'小\s*計', label):
            printed_subtotal = value
        if re.search(r'合\s*計|現\s*計', label):
            total_candidates.append(value)

    item_sum = _line_items_sum(extracted)
    inclusive_tax = any(str(tax.get("label") or "") == "内税" for tax in taxes)
    rate_bases = extract_rate_bases(unified_text)
    if item_sum > 0 and inclusive_tax:
        gross_matches = [
            float(base)
            for base in rate_bases.values()
            if base is not None and abs(float(base) - item_sum) <= 2
        ]
        if gross_matches:
            printed_total = gross_matches[0]
            printed_subtotal = printed_total - tax_sum

    if item_sum > 0 and values:
        subtotal_candidates = [amount for amount in values if abs(amount - item_sum) <= 5]
        if subtotal_candidates:
            candidate_subtotal = subtotal_candidates[0]
            if any(
                amount > candidate_subtotal
                and abs(candidate_subtotal + tax_sum - amount) <= 5
                for amount in [*total_candidates, *values]
            ):
                printed_subtotal = candidate_subtotal

    if printed_subtotal is not None:
        balanced_candidates = [
            (abs(printed_subtotal + tax_sum - amount), amount)
            for amount in [*total_candidates, *values]
            if amount > printed_subtotal
            and abs(printed_subtotal + tax_sum - amount) <= 5
        ]
        if balanced_candidates:
            printed_total = min(balanced_candidates, key=lambda candidate: candidate[0])[1]

    if printed_subtotal is None or printed_total is None:
        return
    if printed_subtotal <= 0 or printed_total <= printed_subtotal:
        return
    if abs(printed_subtotal + tax_sum - printed_total) > 5:
        return

    current_total = extracted.get("total")
    try:
        current_total_f = float(current_total) if current_total is not None else None
    except (TypeError, ValueError):
        current_total_f = None
    current_subtotal = extracted.get("subtotal")
    try:
        current_subtotal_f = float(current_subtotal) if current_subtotal is not None else None
    except (TypeError, ValueError):
        current_subtotal_f = None
    if (
        current_total_f is not None
        and abs(current_total_f - printed_total) <= 0.01
        and current_subtotal_f is not None
        and abs(current_subtotal_f - printed_subtotal) <= 0.01
    ):
        return
    if item_sum > 0 and abs(item_sum - printed_subtotal) > 5 and abs(item_sum - printed_total) > 5:
        return

    extracted["subtotal"] = printed_subtotal
    extracted["total"] = printed_total
    points_used = extracted.get("points_used")
    try:
        points_used_f = float(points_used or 0)
    except (TypeError, ValueError):
        points_used_f = 0.0
    if points_used_f > 0:
        extracted["amount_paid"] = max(0.0, printed_total - points_used_f)
    else:
        amount_paid = extracted.get("amount_paid")
        try:
            amount_paid_f = float(amount_paid) if amount_paid is not None else None
        except (TypeError, ValueError):
            amount_paid_f = None
        if amount_paid_f is None or (current_total_f is not None and abs(amount_paid_f - current_total_f) <= 5):
            extracted["amount_paid"] = printed_total


def _restore_external_tax_total_from_printed_subtotal(extracted, unified_text):
    """Restore total from a printed subtotal or complete external-rate bases."""
    if _has_balanced_printed_summary(extracted, unified_text):
        return
    taxes = [tax for tax in (extracted.get("taxes") or []) if isinstance(tax, dict)]
    tax_sum = _sum_taxable_amounts(taxes)
    if tax_sum <= 0:
        return
    if not any(str(tax.get("label") or "") == "外税" for tax in taxes) and "外税" not in unified_text:
        return
    lines = [line.strip() for line in unified_text.split('\n') if line.strip()]

    def _yen_after(label_idx: int, lookahead: int = 3) -> float | None:
        for j in range(label_idx + 1, min(len(lines), label_idx + 1 + lookahead)):
            if re.search(r'ポイント|POINT|残高|累計|有効', lines[j], re.IGNORECASE):
                break
            vm = re.fullmatch(r'[¥￥]\s*([\d,]+)\s*[\)）]?', lines[j])
            if vm:
                return float(vm.group(1).replace(',', ''))
        return None

    printed_subtotal = None
    for idx, line in enumerate(lines):
        if re.fullmatch(r'小\s*計', line) or (
            line == "小" and idx + 1 < len(lines) and lines[idx + 1] == "計"
        ):
            label_idx = idx + 1 if line == "小" else idx
            printed_subtotal = _yen_after(label_idx)
            break
    item_sum = _line_items_sum(extracted)
    if printed_subtotal is None:
        positive_taxes: list[tuple[str, float]] = []
        for tax in taxes:
            rate = normalize_tax_rate(str(tax.get("rate") or ""))
            try:
                amount = float(tax.get("amount") or 0)
            except (TypeError, ValueError):
                return
            if rate == "0%" or amount <= 0:
                continue
            if str(tax.get("label") or "") != "外税":
                return
            positive_taxes.append((rate, amount))
        if not positive_taxes or len({rate for rate, _amount in positive_taxes}) != len(positive_taxes):
            return
        rate_bases = extract_rate_bases(unified_text)
        printed_bases = []
        for rate, amount in positive_taxes:
            base = rate_bases.get(rate)
            if base is None or not _rate_base_tax_pair_is_valid(rate, float(base), amount):
                return
            printed_bases.append(float(base))
        if abs(sum(printed_bases) - item_sum) > 5:
            return
        printed_subtotal = item_sum
    if printed_subtotal <= 0 or (item_sum > 0 and abs(item_sum - printed_subtotal) > 5):
        return
    expected_total = printed_subtotal + tax_sum

    def _has_visible_summary_or_payment_amount(target: float) -> bool:
        for idx, line in enumerate(lines):
            vm = re.fullmatch(r'[¥￥]\s*([\d,]+)\s*[\)）]?', line)
            if not vm:
                continue
            value = float(vm.group(1).replace(',', ''))
            if abs(value - target) > 5:
                continue
            context = "\n".join(lines[max(0, idx - 4):min(len(lines), idx + 5)])
            has_payment_or_summary = bool(
                re.search(r'合\s*計|現\s*計|支払|お預り|預り|クレジット|電子マネー|WAON|Pay', context)
            )
            loyalty_only = bool(
                re.search(r'ポイント対象|今回獲得|累計|有効|POINT', context, re.IGNORECASE)
            ) and not re.search(r'支払|お預り|預り|クレジット|電子マネー|WAON|Pay', context)
            if has_payment_or_summary and not loyalty_only:
                return True
        return False

    if not _has_visible_summary_or_payment_amount(expected_total):
        return
    current_total = extracted.get("total")
    try:
        current_total_f = float(current_total) if current_total is not None else None
    except (TypeError, ValueError):
        current_total_f = None
    old_total = current_total_f
    if current_total_f is None or abs(current_total_f - expected_total) > 5:
        extracted["total"] = expected_total
    if abs(float(extracted.get("subtotal") or 0) - printed_subtotal) > 5:
        extracted["subtotal"] = printed_subtotal
    points_used = extracted.get("points_used")
    try:
        points_used_f = float(points_used or 0)
    except (TypeError, ValueError):
        points_used_f = 0.0
    expected_paid = max(0.0, expected_total - points_used_f)
    amount_paid = extracted.get("amount_paid")
    try:
        amount_paid_f = float(amount_paid) if amount_paid is not None else None
    except (TypeError, ValueError):
        amount_paid_f = None
    if (
        amount_paid_f is None
        or (old_total is not None and abs(amount_paid_f - old_total) <= 5)
        or abs(amount_paid_f - expected_paid) <= 5
    ):
        extracted["amount_paid"] = expected_paid
