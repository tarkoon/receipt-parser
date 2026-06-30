"""Receipt total and tax-summary arithmetic helpers."""

import re

from .receipt_financial import (
    _bare_number_tax_summary_entries,
    _interleaved_rate_tax_summary_entries,
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
    """Sum actual tax amounts, excluding 0% entries that store exempt bases."""
    return sum(
        float(t.get("amount") or 0)
        for t in (taxes or [])
        if isinstance(t, dict)
        and t.get("rate") != "0%"
        and t.get("amount") is not None
    )


def _line_items_sum(extracted) -> float:
    return sum(
        float(item.get("total") or 0)
        for item in (extracted.get("line_items") or [])
        if isinstance(item, dict)
    )


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
    """Restore tax entries from bare-number rate summary label/value stacks."""
    lines = [line.strip() for line in unified_text.split('\n')]
    entries = _bare_number_tax_summary_entries(lines)
    entries.extend(_interleaved_rate_tax_summary_entries(lines))
    taxes = [(rate, value) for rate, kind, value in entries if kind == "tax" and value > 0]
    if not taxes:
        return
    total = extracted.get("total")
    try:
        total_f = float(total) if total is not None else None
    except (TypeError, ValueError):
        total_f = None
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
        {"rate": rate, "label": label, "amount": value}
        for rate, value in taxes
    ]
    if total_f is not None and total_f >= tax_sum:
        extracted["subtotal"] = total_f - tax_sum


def _items_plus_tax_matches_total(extracted, tolerance: float = 5) -> bool:
    total = extracted.get("total")
    if total is None:
        return False
    item_sum = _line_items_sum(extracted)
    tax_sum = _sum_taxable_amounts(extracted.get("taxes") or [])
    return item_sum > 0 and tax_sum > 0 and abs(item_sum + tax_sum - float(total)) <= tolerance


def _prefer_printed_item_sum_total_when_balanced(extracted, unified_text):
    """Use the item sum as total when OCR prints the same yen amount."""
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
    printed_amounts = [
        float(m.group(1).replace(',', ''))
        for m in re.finditer(r'[¥￥]\s*([\d,]+)\s*-?', unified_text)
    ]
    if not any(abs(amount - item_sum) <= 2 for amount in printed_amounts):
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
    taxes = [tax for tax in (extracted.get("taxes") or []) if isinstance(tax, dict)]
    tax_sum = _sum_taxable_amounts(taxes)
    if tax_sum <= 0:
        return
    lines = [line.strip() for line in unified_text.split('\n')]

    def _yen_after(label_idx: int, lookahead: int = 3) -> float | None:
        for j in range(label_idx + 1, min(len(lines), label_idx + 1 + lookahead)):
            vm = re.fullmatch(r'[¥￥]\s*([\d,]+)\s*[\)）]?', lines[j])
            if vm:
                return float(vm.group(1).replace(',', ''))
            if re.search(r'お預り|お釣|ポイント|伝票|レシート', lines[j]):
                break
        return None

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
            printed_subtotal = _yen_after(label_idx)
            continue
        if (
            printed_total is None
            and (
                re.fullmatch(r'合\s*計', line)
                or (line == "合" and idx + 1 < len(lines) and lines[idx + 1] == "計")
            )
        ):
            label_idx = idx + 1 if line == "合" else idx
            printed_total = _yen_after(label_idx)
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
    if item_sum > 0 and abs(item_sum - printed_subtotal) > 5:
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
    """Restore total when printed subtotal plus external taxes matches a visible total."""
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
    if printed_subtotal is None or printed_subtotal <= 0:
        return
    item_sum = _line_items_sum(extracted)
    if item_sum > 0 and abs(item_sum - printed_subtotal) > 5:
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
        or amount_paid_f < expected_paid
    ):
        extracted["amount_paid"] = expected_paid
