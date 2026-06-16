"""Guardrails against brittle production receipt parsing code.

Production parsing code must not special-case specific merchants, stores,
receipt IDs, fixture ranges, known dates, known product lists, or known final
totals. Production code may implement general layout/format strategies only
when they are triggered by structural OCR evidence and validated by
arithmetic/format invariants.

This test intentionally allows known violations documented in
pipeline_brittleness_audit.md so the current tree can still run the guardrail.
The allowlist is exact and should shrink as those production branches are
removed or replaced with general parsers.
"""

from __future__ import annotations

import ast
import copy
import io
import re
import subprocess
import tokenize
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARSER_DIR = ROOT / "src" / "receipt_parser"
SCANNED_FILES = tuple(
    sorted({PARSER_DIR / "pipeline.py", *PARSER_DIR.glob("pipeline_*.py")})
)

MERCHANT_OR_STORE_RE = re.compile(
    r"("
    r"maxvalu|max_value|familymart|family_mart|daiso|costco|gyomu|"
    r"starbucks|donki|seria|cosmos|nafco|nishimatsuya|yakitori"
    r")",
    re.IGNORECASE,
)
FIXTURE_REFERENCE_RE = re.compile(
    r"("
    r"receipt[_-]?\d+|"
    r"target[_-]?\d+(?:[_-]to[_-]\d+|[_-]\d+)?|"
    r"known[_-]?\d+(?:[_-]to[_-]\d+)?"
    r")",
    re.IGNORECASE,
)
KNOWN_ANSWER_NAME_RE = re.compile(
    r"(^|_)known(_|$)|final_known|known_answer|known_financial",
    re.IGNORECASE,
)
KNOWN_DATE_RE = re.compile(r"\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b")

SEMANTIC_FIELDS = {
    "amount_paid",
    "billing_period",
    "date",
    "line_items",
    "location",
    "merchant",
    "payment_method",
    "points_earned",
    "points_used",
    "subtotal",
    "tax_entries",
    "time",
    "total",
    "usage",
}
FINAL_RESULT_MUTATORS = {
    "_drop_duplicate_with_embedded_price",
    "_replace_barcode_qty_price_rows_when_balanced",
    "fix_final_known_financial_overrides",
    "postprocess_receipt",
}
FINAL_OUTPUT_FUNCTIONS = {
    "_apply_final_receipt_output_repairs",
    "_prepare_receipt_output_payload",
    "_build_result",
}
FINAL_OUTPUT_KNOWN_ANSWER_MUTATORS = {
    "fix_final_known_financial_overrides",
    "postprocess_receipt",
}
BASELINE_COMMIT = "c175c17"
POSTPROCESS_REPAIR_CALL_LIMIT = 59

REPAIR_CALL_PREFIXES = (
    "_append_",
    "_apply_",
    "_clear_",
    "_clean_",
    "_drop_",
    "_fix_",
    "_normalize_",
    "_rebalance_",
    "_recover_",
    "_repair_",
    "_replace_",
    "_restore_",
    "assign_",
    "reconcile_",
)
POSTPROCESS_MUTATOR_REPEAT_ALLOWLIST = {
}
STRUCTURAL_ITEM_PROJECTION_REPAIRS = {
    "_replace_barcode_qty_price_rows_when_balanced",
    "_replace_barcode_unit_qty_amount_stack_when_balanced",
    "_replace_dense_sequence_rows_when_balanced",
    "_replace_jan_pos_items_when_balanced",
}
STRUCTURAL_ITEM_PROJECTION_PHASE_HELPER = "_run_structural_item_projection_phase"
STRUCTURAL_ITEM_PROJECTION_PHASE_CALL_LIMIT = 6
FINAL_STRUCTURAL_ITEM_PROJECTION_REPAIRS = {
    "_replace_barcode_unit_qty_amount_stack_when_balanced",
}
FINAL_STRUCTURAL_ITEM_PROJECTION_HELPER = (
    "_run_final_structural_item_projection_phase"
)
FINAL_STRUCTURAL_ITEM_PROJECTION_STAGE_LIMIT = 1
QUANTITY_DETAIL_RECONCILIATION_REPAIRS = {
    "_fix_qty_context_and_reduced_rate_from_ocr",
    "_fix_qty_totals_from_ocr_unit_lines",
    "_repair_previous_item_from_following_qty_detail",
}
QUANTITY_DETAIL_RECONCILIATION_PHASE_HELPER = "_run_quantity_detail_reconciliation_phase"
QUANTITY_DETAIL_RECONCILIATION_PHASE_CALL_LIMIT = 10
TAX_CATEGORY_ASSIGNMENT_REPAIRS = {
    "_apply_single_bag_standard_rate_split",
    "_assign_single_standard_rate_from_small_base",
    "_fix_nonfood_packaging_tax_categories",
    "_fix_tax_categories_from_ocr_markers",
    "_fix_tax_categories_from_price_line_markers",
    "_rebalance_standard_categories_from_reduced_rate_markers",
    "_rebalance_tax_categories_to_rate_bases",
    "assign_tax_categories",
}
TAX_CATEGORY_ASSIGNMENT_PHASE_HELPER = "_run_tax_category_assignment_phase"
TAX_CATEGORY_ASSIGNMENT_PHASE_CALL_LIMIT = 6
SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_REPAIRS = {
    "_restore_single_rate_inclusive_tax_block",
}
SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_PHASE_HELPER = (
    "_run_single_rate_inclusive_tax_restoration_phase"
)
SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_PHASE_CALL_LIMIT = 3
TAX_EXCLUDED_RATE_BLOCK_RESTORATION_REPAIRS = {
    "_restore_tax_excluded_per_rate_blocks",
}
TAX_EXCLUDED_RATE_BLOCK_RESTORATION_PHASE_HELPER = (
    "_run_tax_excluded_rate_block_restoration_phase"
)
TAX_EXCLUDED_RATE_BLOCK_RESTORATION_PHASE_CALL_LIMIT = 3
EXPLICIT_TAX_AMOUNT_RESTORATION_REPAIRS = {
    "_restore_explicit_tax_rate_amount_lines",
}
EXPLICIT_TAX_AMOUNT_RESTORATION_PHASE_HELPER = (
    "_run_explicit_tax_amount_restoration_phase"
)
EXPLICIT_TAX_AMOUNT_RESTORATION_PHASE_CALL_LIMIT = 3
EXTERNAL_TAX_TOTAL_RESTORATION_REPAIRS = {
    "_restore_external_tax_total_from_printed_subtotal",
}
EXTERNAL_TAX_TOTAL_RESTORATION_PHASE_HELPER = (
    "_run_external_tax_total_restoration_phase"
)
EXTERNAL_TAX_TOTAL_RESTORATION_PHASE_CALL_LIMIT = 3
CASH_TENDER_RECONCILIATION_REPAIRS = {
    "_fix_total_from_stacked_cash_tender_block",
    "_fix_unlabeled_cash_tender_change_block",
}
CASH_TENDER_RECONCILIATION_PHASE_HELPER = "_run_cash_tender_reconciliation_phase"
CASH_TENDER_RECONCILIATION_PHASE_CALL_LIMIT = 2
PAYMENT_POINTS_RECONCILIATION_REPAIRS = {
    "extract_points_used",
    "reconcile_points_payment_from_ocr",
}
PAYMENT_POINTS_RECONCILIATION_PHASE_HELPER = "_run_payment_points_reconciliation_phase"
PAYMENT_POINTS_RECONCILIATION_PHASE_CALL_LIMIT = 2
SERVICE_RECEIPT_RECOVERY_REPAIRS = {
    "_fix_bare_service_receipt_without_itemization",
    "_fix_single_service_inclusive_tax",
    "_replace_service_table_items_when_balanced",
}
SERVICE_RECEIPT_RECOVERY_PHASE_HELPER = "_run_service_receipt_recovery_phase"
SERVICE_RECEIPT_RECOVERY_PHASE_CALL_LIMIT = 6
BODY_TOTAL_LAYOUT_RECONSTRUCTION_REPAIRS = {
    "_fix_split_item_price_body_total_layout",
}
BODY_TOTAL_LAYOUT_RECONSTRUCTION_PHASE_HELPER = (
    "_run_body_total_layout_reconstruction_phase"
)
BODY_TOTAL_LAYOUT_RECONSTRUCTION_PHASE_CALL_LIMIT = 3
OCR_DESCRIPTION_RECONCILIATION_REPAIRS = {
    "_fix_bag_description_from_ocr_code_context",
    "_fix_code_table_descriptions_by_order",
    "_fix_colon_split_product_names_from_ocr",
    "_fix_duplicate_descriptions_from_ocr",
    "_fix_o_ring_descriptions_from_ocr",
    "_fix_qty_code_row_descriptions_from_ocr",
}
OCR_DESCRIPTION_RECONCILIATION_PHASE_HELPER = "_run_ocr_description_reconciliation_phase"
OCR_DESCRIPTION_RECONCILIATION_PHASE_CALL_LIMIT = 7
GAP_ITEM_RECOVERY_REPAIRS = {
    "_recover_discounted_item_from_gap",
    "_recover_missing_items_from_gap",
    "_recover_repeated_item_from_gap",
    "_replace_repeated_ocr_item_block_when_balanced",
}
GAP_ITEM_RECOVERY_PHASE_HELPER = "_run_gap_item_recovery_phase"
GAP_ITEM_RECOVERY_PHASE_CALL_LIMIT = 8
LOW_VALUE_BAG_RECOVERY_REPAIRS = {
    "_fix_numeric_desc_from_ocr_price_context",
    "_recover_missing_bag_items_from_ocr",
    "_replace_overage_item_with_low_value_bag",
}
LOW_VALUE_BAG_RECOVERY_PHASE_HELPER = "_run_low_value_bag_recovery_phase"
LOW_VALUE_BAG_RECOVERY_PHASE_CALL_LIMIT = 4
ADJACENT_PRICE_SHIFT_REPAIRS = {
    "_fix_adjacent_ocr_price_shift_when_balanced",
}
ADJACENT_PRICE_SHIFT_PHASE_HELPER = "_run_adjacent_price_shift_reconciliation_phase"
ADJACENT_PRICE_SHIFT_PHASE_CALL_LIMIT = 4
BAG_AMOUNT_SHIFT_REPAIRS = {
    "_fix_name_bag_amount_shift_from_ocr",
}
BAG_AMOUNT_SHIFT_PHASE_HELPER = "_run_bag_amount_shift_reconciliation_phase"
BAG_AMOUNT_SHIFT_PHASE_CALL_LIMIT = 3
LINE_ITEM_CLEANUP_REPAIRS = {
    "_drop_duplicate_with_embedded_price",
    "_drop_non_product_line_items",
    "_drop_numeric_marker_description_rows",
}
LINE_ITEM_CLEANUP_PHASE_HELPER = "_run_line_item_cleanup_phase"
LINE_ITEM_CLEANUP_PHASE_CALL_LIMIT = 14
FINAL_OUTPUT_REPAIR_STAGES = (
    "barcode_unit_qty_amount_stack",
    "barcode_qty_price_rows",
    "item_price_qty_rows",
    "labeled_purchase_site_location",
    "store_in_store_header_location",
    "header_branch_store_location",
    "phone_area_city_location",
    "short_branch_over_phone_area_city",
    "noisy_city_location",
    "single_rate_inclusive_tax_block",
    "following_discount_lines",
    "coupon_discount_blocks",
    "drop_applied_coupon_line_items",
    "tiny_item_prices_from_following_ocr",
    "split_price_block",
    "split_item_price_body_total",
    "stacked_name_price_rows",
    "stacked_inclusive_tax_block",
    "printed_summary_total_tax_balanced",
    "printed_item_sum_total",
    "o_ring_descriptions",
    "company_name_merchant",
    "adjacent_ocr_price_shift",
    "repeated_item_gap",
    "drop_duplicate_embedded_price",
    "dense_sequence_rows",
    "campaign_discount_stream",
    "jan_pos_items",
    "qty_totals_from_unit_lines",
    "bag_item_prices_from_rate_bases",
    "code_table_descriptions",
    "printed_external_tax_amounts",
    "bare_number_tax_summary",
    "external_tax_total_from_printed_subtotal",
    "drop_small_target_only_taxes",
    "printed_summary_total_tax_balanced_2",
    "unlabeled_cash_tender_change",
    "points_payment",
    "clear_discount_before_own_price",
    "campaign_discount_stream_2",
    "following_discount_lines_after_layout",
    "discounted_line_item_totals",
    "adjacent_ocr_price_shift_final",
    "prefixed_tax_marker_item_rows",
    "missing_items_from_gap",
    "discounted_ocr_pair_descriptions",
    "pre_price_stack_descriptions",
    "drop_duplicate_rows_when_subtotal_balances",
    "basket_marker_rows",
    "tax_categories_from_rate_bases",
    "external_tax_total_from_printed_subtotal_final",
)
STRUCTURAL_JAPANESE_LITERAL_RE = re.compile(
    r"(小計|合計|内税|外税|非課税|消費税|税|対象|軽減|税込|税抜|"
    r"現金|預|釣|支払|ポイント|領収|レシート|登録番号|電話|TEL|"
    r"店|支店|営業所|料金所|住所|市|区|町|村|県|都|道|府|"
    r"年|月|日|時|分|個|点|円|品番|JAN|バーコード)"
)
JAPANESE_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    function: str
    rule: str
    detail: str

    @property
    def key(self) -> tuple[str, int, str, str]:
        return self.path, self.line, self.rule, self.detail

    @property
    def signature(self) -> tuple[str, str, str]:
        return self.path, self.rule, self.detail

    def format(self) -> str:
        return (
            f"{self.path}:{self.line} in {self.function}: "
            f"{self.rule}: {self.detail}"
        )


# Known violations from pipeline_brittleness_audit.md. Keep the entries tied to
# source locations for review, but compare by signature counts so harmless line
# movement does not make the guardrail stale.
KNOWN_VIOLATIONS = set()


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _enclosing_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current.name
    return "<module>"


def _literal_key(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_repair_call_name(name: str | None) -> bool:
    return bool(name) and name.startswith(REPAIR_CALL_PREFIXES)


def _function_def(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Function not found: {name}")


def _call_names_in_function(function: ast.FunctionDef) -> list[str]:
    names = []
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name:
                names.append(name)
    return names


def _parse_file(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _postprocess_repair_calls() -> list[tuple[str, int]]:
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    function = _function_def(tree, "postprocess_receipt")
    calls = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if _is_repair_call_name(name):
            calls.append((name or "", node.lineno))
    return calls


def _final_output_repair_stage_calls() -> list[tuple[str, int]]:
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    function = _function_def(tree, "_apply_final_receipt_output_repairs")
    stages = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or _call_name(node.func) != "run":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            stages.append(("<nonliteral>", node.lineno))
            continue
        stages.append((str(node.args[0].value), node.lineno))
    return sorted(stages, key=lambda item: item[1])


def _final_output_repair_justifications() -> dict[str, tuple[str, str]]:
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "FINAL_RECEIPT_OUTPUT_REPAIR_JUSTIFICATIONS"
                for target in node.targets
            )
        ):
            try:
                value = ast.literal_eval(node.value)
            except (SyntaxError, ValueError) as exc:
                raise AssertionError(
                    "FINAL_RECEIPT_OUTPUT_REPAIR_JUSTIFICATIONS must be literal"
                ) from exc
            return value
    raise AssertionError("FINAL_RECEIPT_OUTPUT_REPAIR_JUSTIFICATIONS not found")


def _postprocess_phase_names() -> set[str]:
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "POSTPROCESS_PHASES"
                for target in node.targets
            )
        ):
            try:
                value = ast.literal_eval(node.value)
            except (SyntaxError, ValueError) as exc:
                raise AssertionError("POSTPROCESS_PHASES must be literal") from exc
            return {
                phase["name"]
                for phase in value
                if isinstance(phase, dict) and isinstance(phase.get("name"), str)
            }
    raise AssertionError("POSTPROCESS_PHASES not found")


def _current_japanese_string_counts() -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for path in SCANNED_FILES:
        tree = _parse_file(path)
        rel = _relative(path)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and JAPANESE_CHAR_RE.search(node.value)
            ):
                counts[(rel, node.value)] += 1
    return counts


def _baseline_japanese_string_counts() -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for path in SCANNED_FILES:
        rel = _relative(path)
        try:
            source = subprocess.check_output(
                ["git", "show", f"{BASELINE_COMMIT}:{rel}"],
                cwd=ROOT,
                text=True,
                encoding="utf-8",
            )
        except subprocess.CalledProcessError as exc:
            raise AssertionError(
                f"Could not read {rel} from baseline {BASELINE_COMMIT}"
            ) from exc
        tree = ast.parse(source, filename=rel)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and JAPANESE_CHAR_RE.search(node.value)
            ):
                counts[(rel, node.value)] += 1
    return counts


def _looks_structural_japanese_literal(value: str) -> bool:
    return bool(STRUCTURAL_JAPANESE_LITERAL_RE.search(value))


def _assigned_semantic_fields(node: ast.AST) -> set[str]:
    fields: set[str] = set()
    targets = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    elif isinstance(node, ast.AugAssign):
        targets = [node.target]

    for target in targets:
        if isinstance(target, ast.Subscript):
            field = _literal_key(target.slice)
            if field in SEMANTIC_FIELDS:
                fields.add(field)
    return fields


def _condition_known_value_gates(node: ast.AST, source: str) -> list[str]:
    if not isinstance(node, ast.Compare):
        return []
    if not any(isinstance(op, (ast.Eq, ast.In)) for op in node.ops):
        return []

    constants = [
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant)
        and isinstance(child.value, (str, int, float))
    ]
    if not constants:
        return []

    suspicious = []
    for value in constants:
        if isinstance(value, str):
            if (
                KNOWN_DATE_RE.search(value)
                or MERCHANT_OR_STORE_RE.search(value)
                or FIXTURE_REFERENCE_RE.search(value)
            ):
                suspicious.append(value)
        elif isinstance(value, (int, float)) and abs(value) >= 1000:
            suspicious.append(value)

    if not suspicious:
        return []
    return [(ast.get_source_segment(source, node) or repr(suspicious)).replace("\n", " ")]


def _scan_ast(path: Path) -> list[Violation]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    parents = _parents(tree)
    rel = _relative(path)
    violations: list[Violation] = []

    for node in ast.walk(tree):
        function = _enclosing_function(node, parents)

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if MERCHANT_OR_STORE_RE.search(node.name):
                violations.append(
                    Violation(rel, node.lineno, function, "merchant_or_store_name", node.name)
                )
            if FIXTURE_REFERENCE_RE.search(node.name):
                violations.append(
                    Violation(rel, node.lineno, function, "fixture_reference_name", node.name)
                )
            if KNOWN_ANSWER_NAME_RE.search(node.name):
                violations.append(
                    Violation(rel, node.lineno, function, "known_answer_helper_name", node.name)
                )

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and _literal_key(target.slice) == "line_items"
                    and isinstance(node.value, ast.List)
                    and len(node.value.elts) >= 2
                ):
                    violations.append(
                        Violation(
                            rel,
                            node.lineno,
                            function,
                            "hardcoded_line_items_assignment",
                            f"line_items[{len(node.value.elts)}]",
                        )
                    )

        for detail in _condition_known_value_gates(node, source):
            violations.append(
                Violation(rel, node.lineno, function, "known_value_gate", detail)
            )

        if function == "_build_result" and isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in FINAL_RESULT_MUTATORS:
                violations.append(
                    Violation(
                        rel,
                        node.lineno,
                        function,
                        "final_result_semantic_mutation",
                        name,
                    )
                )
        if function in FINAL_OUTPUT_FUNCTIONS and isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in FINAL_OUTPUT_KNOWN_ANSWER_MUTATORS:
                violations.append(
                    Violation(
                        rel,
                        node.lineno,
                        function,
                        "final_output_known_answer_mutation",
                        name,
                    )
                )

        if isinstance(node, ast.If):
            condition_text = ast.get_source_segment(source, node.test) or ""
            condition_has_known_gate = (
                MERCHANT_OR_STORE_RE.search(condition_text)
                or FIXTURE_REFERENCE_RE.search(condition_text)
                or KNOWN_DATE_RE.search(condition_text)
            )
            if condition_has_known_gate:
                fields = set()
                for body_node in node.body:
                    for child in ast.walk(body_node):
                        fields.update(_assigned_semantic_fields(child))
                if fields:
                    violations.append(
                        Violation(
                            rel,
                            node.lineno,
                            function,
                            "known_gate_semantic_assignment",
                            ",".join(sorted(fields)),
                        )
                    )

    return violations


def _scan_comments(path: Path) -> list[Violation]:
    violations: list[Violation] = []
    rel = _relative(path)
    with tokenize.open(path) as handle:
        tokens = tokenize.generate_tokens(handle.readline)
        for token in tokens:
            if token.type != tokenize.COMMENT:
                continue
            comment = token.string
            if FIXTURE_REFERENCE_RE.search(comment):
                violations.append(
                    Violation(
                        rel,
                        token.start[0],
                        "<comment>",
                        "fixture_reference_comment",
                        comment.strip(),
                    )
                )
    return violations


def _collect_violations() -> list[Violation]:
    violations: list[Violation] = []
    for path in SCANNED_FILES:
        violations.extend(_scan_ast(path))
        violations.extend(_scan_comments(path))
    return sorted(violations, key=lambda item: item.key)


def test_production_pipeline_has_no_new_brittle_known_answer_overrides():
    violations = _collect_violations()
    known_counts = Counter(
        (path, rule, detail) for path, _line, rule, detail in KNOWN_VIOLATIONS
    )
    violation_counts = Counter(violation.signature for violation in violations)
    unexpected = [
        violation
        for violation in violations
        if violation_counts[violation.signature] > known_counts[violation.signature]
    ]
    seen_unexpected: set[tuple[str, str, str]] = set()
    unexpected = [
        violation
        for violation in unexpected
        if violation.signature not in seen_unexpected
        and not seen_unexpected.add(violation.signature)
    ]
    stale_allowlist = sorted(
        signature
        for signature, count in known_counts.items()
        if violation_counts[signature] < count
    )

    message = io.StringIO()
    if unexpected:
        message.write(
            "Production parser code contains brittle known-answer patterns.\n"
            "Use structural OCR evidence plus arithmetic/format invariants instead.\n"
            "Unexpected violations:\n"
        )
        for violation in unexpected:
            message.write(f"  - {violation.format()}\n")
    if stale_allowlist:
        message.write(
            "\nThe guardrail allowlist contains entries that no longer match the "
            "current source. Remove these known-violation entries:\n"
        )
        for key in stale_allowlist:
            message.write(
                f"  - {key} "
                f"(expected {known_counts[key]}, found {violation_counts[key]})\n"
            )

    assert not unexpected and not stale_allowlist, message.getvalue()


def test_postprocess_receipt_repair_stack_does_not_grow_without_review():
    calls = _postprocess_repair_calls()
    assert len(calls) <= POSTPROCESS_REPAIR_CALL_LIMIT, (
        "postprocess_receipt gained repair/mutator calls. Split the work into "
        "a named phase with structural trigger and invariant, or explicitly "
        "lower existing debt before adding more.\n"
        f"Current count: {len(calls)}; limit: {POSTPROCESS_REPAIR_CALL_LIMIT}"
    )


def test_postprocess_receipt_repeated_mutators_are_explicitly_allowlisted():
    counts = Counter(name for name, _line in _postprocess_repair_calls())
    repeated = {name: count for name, count in counts.items() if count >= 3}
    unexpected = sorted(set(repeated) - set(POSTPROCESS_MUTATOR_REPEAT_ALLOWLIST))
    grown = sorted(
        (name, repeated[name], allowed)
        for name, allowed in POSTPROCESS_MUTATOR_REPEAT_ALLOWLIST.items()
        if repeated.get(name, 0) > allowed
    )

    assert not unexpected and not grown, (
        "Repeated mutator calls in postprocess_receipt must be explained as "
        "temporary debt and must not grow.\n"
        f"Unexpected repeated mutators: {unexpected}\n"
        f"Allowlisted mutators whose call counts grew: {grown}"
    )


def test_structural_item_projection_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, STRUCTURAL_ITEM_PROJECTION_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        STRUCTURAL_ITEM_PROJECTION_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Structural item row projection repairs must be owned by the named "
        f"{STRUCTURAL_ITEM_PROJECTION_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{STRUCTURAL_ITEM_PROJECTION_PHASE_HELPER} must document the OCR/layout "
        "trigger and arithmetic or field-consistency invariant it enforces."
    )


def test_postprocess_structural_item_projection_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_projection_calls = [
        name for name in postprocess_calls if name in STRUCTURAL_ITEM_PROJECTION_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == STRUCTURAL_ITEM_PROJECTION_PHASE_HELPER
    ]

    assert not direct_projection_calls, (
        "Structural item projection repairs should run through the named phase "
        "helper so OCR/layout triggers and arithmetic invariants have one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_projection_calls}"
    )
    assert 0 < len(phase_calls) <= STRUCTURAL_ITEM_PROJECTION_PHASE_CALL_LIMIT, (
        "Structural item projection phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {STRUCTURAL_ITEM_PROJECTION_PHASE_CALL_LIMIT}"
    )


def test_final_structural_item_projection_helper_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    helper = _function_def(tree, FINAL_STRUCTURAL_ITEM_PROJECTION_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        FINAL_STRUCTURAL_ITEM_PROJECTION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Late barcode/unit/qty/amount stack projection must be owned by the "
        f"named {FINAL_STRUCTURAL_ITEM_PROJECTION_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{FINAL_STRUCTURAL_ITEM_PROJECTION_HELPER} must document the visible "
        "barcode/JAN stack trigger and item-sum arithmetic invariant."
    )


def test_final_structural_item_projection_debt_is_helper_owned():
    tree = _parse_file(PARSER_DIR / "pipeline.py")
    final_repairs = _function_def(tree, "_apply_final_receipt_output_repairs")
    final_calls = _call_names_in_function(final_repairs)
    direct_projection_calls = [
        name
        for name in final_calls
        if name in FINAL_STRUCTURAL_ITEM_PROJECTION_REPAIRS
    ]
    helper_calls = [
        name
        for name in final_calls
        if name == FINAL_STRUCTURAL_ITEM_PROJECTION_HELPER
    ]

    assert not direct_projection_calls, (
        "Late barcode/unit/qty/amount stack projection should run through the "
        "named helper so OCR row-stack triggers and item-sum invariants have "
        "one owner.\n"
        "Direct calls still in _apply_final_receipt_output_repairs: "
        f"{direct_projection_calls}"
    )
    assert 0 < len(helper_calls) <= FINAL_STRUCTURAL_ITEM_PROJECTION_STAGE_LIMIT, (
        "Late barcode/unit/qty/amount stack projection helper calls must be "
        "explicit and bounded.\n"
        f"Current count: {len(helper_calls)}; "
        f"limit: {FINAL_STRUCTURAL_ITEM_PROJECTION_STAGE_LIMIT}"
    )


def test_quantity_detail_reconciliation_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, QUANTITY_DETAIL_RECONCILIATION_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        QUANTITY_DETAIL_RECONCILIATION_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Quantity detail repairs must be owned by the named "
        f"{QUANTITY_DETAIL_RECONCILIATION_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{QUANTITY_DETAIL_RECONCILIATION_PHASE_HELPER} must document the OCR "
        "quantity-detail trigger and arithmetic or field-consistency invariant."
    )


def test_postprocess_quantity_detail_reconciliation_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    structural_projection = _function_def(tree, STRUCTURAL_ITEM_PROJECTION_PHASE_HELPER)
    postprocess_calls = _call_names_in_function(postprocess)
    structural_calls = _call_names_in_function(structural_projection)
    direct_quantity_calls = [
        name for name in postprocess_calls if name in QUANTITY_DETAIL_RECONCILIATION_REPAIRS
    ]
    structural_quantity_calls = [
        name for name in structural_calls if name in QUANTITY_DETAIL_RECONCILIATION_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == QUANTITY_DETAIL_RECONCILIATION_PHASE_HELPER
    ]

    assert not direct_quantity_calls, (
        "Quantity detail repairs should run through the named phase helper so "
        "OCR quantity triggers and qty * unit invariants have one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_quantity_calls}"
    )
    assert not structural_quantity_calls, (
        "Quantity detail reconciliation should not be hidden inside the broader "
        "structural projection phase.\n"
        f"Structural helper still owns: {structural_quantity_calls}"
    )
    assert 0 < len(phase_calls) <= QUANTITY_DETAIL_RECONCILIATION_PHASE_CALL_LIMIT, (
        "Quantity detail reconciliation phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {QUANTITY_DETAIL_RECONCILIATION_PHASE_CALL_LIMIT}"
    )


def test_tax_category_assignment_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, TAX_CATEGORY_ASSIGNMENT_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        TAX_CATEGORY_ASSIGNMENT_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Tax category assignment and rate-base rebalance repairs must be owned "
        f"by the named {TAX_CATEGORY_ASSIGNMENT_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{TAX_CATEGORY_ASSIGNMENT_PHASE_HELPER} must document the OCR "
        "rate-marker trigger and tax/line-item consistency invariant."
    )


def test_postprocess_tax_category_assignment_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_tax_calls = [
        name for name in postprocess_calls if name in TAX_CATEGORY_ASSIGNMENT_REPAIRS
    ]
    phase_calls = [
        name for name in postprocess_calls if name == TAX_CATEGORY_ASSIGNMENT_PHASE_HELPER
    ]

    assert not direct_tax_calls, (
        "Tax category assignment repairs should run through the named phase "
        "helper so OCR rate markers, rate bases, and item/tax invariants have "
        "one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_tax_calls}"
    )
    assert 0 < len(phase_calls) <= TAX_CATEGORY_ASSIGNMENT_PHASE_CALL_LIMIT, (
        "Tax category assignment phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {TAX_CATEGORY_ASSIGNMENT_PHASE_CALL_LIMIT}"
    )


def test_single_rate_inclusive_tax_restoration_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Single-rate inclusive tax restorations must be owned by the named "
        f"{SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_PHASE_HELPER} must document "
        "the printed single-rate inclusive tax trigger and total/tax invariant."
    )


def test_postprocess_single_rate_inclusive_tax_restoration_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_tax_calls = [
        name
        for name in postprocess_calls
        if name in SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_PHASE_HELPER
    ]

    assert not direct_tax_calls, (
        "Single-rate inclusive tax restoration should run through the named "
        "phase helper so printed target/tax rows and subtotal arithmetic have "
        "one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_tax_calls}"
    )
    assert 0 < len(phase_calls) <= SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_PHASE_CALL_LIMIT, (
        "Single-rate inclusive tax restoration phase calls must be explicit "
        "and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {SINGLE_RATE_INCLUSIVE_TAX_RESTORATION_PHASE_CALL_LIMIT}"
    )


def test_tax_excluded_rate_block_restoration_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, TAX_EXCLUDED_RATE_BLOCK_RESTORATION_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        TAX_EXCLUDED_RATE_BLOCK_RESTORATION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Tax-excluded per-rate block restorations must be owned by the named "
        f"{TAX_EXCLUDED_RATE_BLOCK_RESTORATION_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{TAX_EXCLUDED_RATE_BLOCK_RESTORATION_PHASE_HELPER} must document "
        "the printed tax-excluded subtotal/tax-row trigger and rate-pair "
        "consistency invariant."
    )


def test_postprocess_tax_excluded_rate_block_restoration_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_tax_calls = [
        name
        for name in postprocess_calls
        if name in TAX_EXCLUDED_RATE_BLOCK_RESTORATION_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == TAX_EXCLUDED_RATE_BLOCK_RESTORATION_PHASE_HELPER
    ]

    assert not direct_tax_calls, (
        "Tax-excluded per-rate block restoration should run through the named "
        "phase helper so printed subtotal/tax labels and rate-paired tax "
        "entries have one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_tax_calls}"
    )
    assert 0 < len(phase_calls) <= TAX_EXCLUDED_RATE_BLOCK_RESTORATION_PHASE_CALL_LIMIT, (
        "Tax-excluded per-rate block restoration phase calls must be explicit "
        "and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {TAX_EXCLUDED_RATE_BLOCK_RESTORATION_PHASE_CALL_LIMIT}"
    )


def test_explicit_tax_amount_restoration_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, EXPLICIT_TAX_AMOUNT_RESTORATION_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        EXPLICIT_TAX_AMOUNT_RESTORATION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Explicit tax amount restorations must be owned by the named "
        f"{EXPLICIT_TAX_AMOUNT_RESTORATION_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{EXPLICIT_TAX_AMOUNT_RESTORATION_PHASE_HELPER} must document "
        "the visible tax-rate amount row trigger and item/tax consistency "
        "invariant."
    )


def test_postprocess_explicit_tax_amount_restoration_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_tax_calls = [
        name
        for name in postprocess_calls
        if name in EXPLICIT_TAX_AMOUNT_RESTORATION_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == EXPLICIT_TAX_AMOUNT_RESTORATION_PHASE_HELPER
    ]

    assert not direct_tax_calls, (
        "Explicit tax amount restoration should run through the named phase "
        "helper so visible 税率N%税額 rows and tax-entry replacement have one "
        "owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_tax_calls}"
    )
    assert 0 < len(phase_calls) <= EXPLICIT_TAX_AMOUNT_RESTORATION_PHASE_CALL_LIMIT, (
        "Explicit tax amount restoration phase calls must be explicit and "
        "bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {EXPLICIT_TAX_AMOUNT_RESTORATION_PHASE_CALL_LIMIT}"
    )


def test_external_tax_total_restoration_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, EXTERNAL_TAX_TOTAL_RESTORATION_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        EXTERNAL_TAX_TOTAL_RESTORATION_REPAIRS
        - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "External tax total restorations must be owned by the named "
        f"{EXTERNAL_TAX_TOTAL_RESTORATION_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{EXTERNAL_TAX_TOTAL_RESTORATION_PHASE_HELPER} must document "
        "the printed subtotal plus external-tax summary trigger and "
        "total/payment arithmetic invariant."
    )


def test_postprocess_external_tax_total_restoration_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_tax_calls = [
        name
        for name in postprocess_calls
        if name in EXTERNAL_TAX_TOTAL_RESTORATION_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == EXTERNAL_TAX_TOTAL_RESTORATION_PHASE_HELPER
    ]

    assert not direct_tax_calls, (
        "External tax total restoration should run through the named phase "
        "helper so printed subtotal, external-tax entries, and visible "
        "summary/payment totals have one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_tax_calls}"
    )
    assert 0 < len(phase_calls) <= EXTERNAL_TAX_TOTAL_RESTORATION_PHASE_CALL_LIMIT, (
        "External tax total restoration phase calls must be explicit and "
        "bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {EXTERNAL_TAX_TOTAL_RESTORATION_PHASE_CALL_LIMIT}"
    )


def test_cash_tender_reconciliation_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, CASH_TENDER_RECONCILIATION_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        CASH_TENDER_RECONCILIATION_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Cash tender/change repairs must be owned by the named "
        f"{CASH_TENDER_RECONCILIATION_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{CASH_TENDER_RECONCILIATION_PHASE_HELPER} must document the OCR "
        "cash-layout trigger and total/payment consistency invariant."
    )


def test_postprocess_cash_tender_reconciliation_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_cash_calls = [
        name for name in postprocess_calls if name in CASH_TENDER_RECONCILIATION_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == CASH_TENDER_RECONCILIATION_PHASE_HELPER
    ]

    assert not direct_cash_calls, (
        "Cash tender/change repairs should run through the named phase helper "
        "so printed total, tendered amount, and change arithmetic have one "
        "owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_cash_calls}"
    )
    assert 0 < len(phase_calls) <= CASH_TENDER_RECONCILIATION_PHASE_CALL_LIMIT, (
        "Cash tender reconciliation phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {CASH_TENDER_RECONCILIATION_PHASE_CALL_LIMIT}"
    )


def test_payment_points_reconciliation_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, PAYMENT_POINTS_RECONCILIATION_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        PAYMENT_POINTS_RECONCILIATION_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Payment/points repairs must be owned by the named "
        f"{PAYMENT_POINTS_RECONCILIATION_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{PAYMENT_POINTS_RECONCILIATION_PHASE_HELPER} must document the OCR "
        "points/payment trigger and total/payment consistency invariant."
    )


def test_postprocess_payment_points_reconciliation_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_payment_points_calls = [
        name for name in postprocess_calls if name in PAYMENT_POINTS_RECONCILIATION_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == PAYMENT_POINTS_RECONCILIATION_PHASE_HELPER
    ]

    assert not direct_payment_points_calls, (
        "Payment/points repairs should run through the named phase helper so "
        "OCR point-use evidence and total minus points payment arithmetic have "
        "one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_payment_points_calls}"
    )
    assert 0 < len(phase_calls) <= PAYMENT_POINTS_RECONCILIATION_PHASE_CALL_LIMIT, (
        "Payment/points reconciliation phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {PAYMENT_POINTS_RECONCILIATION_PHASE_CALL_LIMIT}"
    )


def test_service_receipt_recovery_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, SERVICE_RECEIPT_RECOVERY_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        SERVICE_RECEIPT_RECOVERY_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Service receipt recovery repairs must be owned by the named "
        f"{SERVICE_RECEIPT_RECOVERY_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{SERVICE_RECEIPT_RECOVERY_PHASE_HELPER} must document the OCR "
        "service-layout trigger and item/total/tax consistency invariant."
    )


def test_postprocess_service_receipt_recovery_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_service_calls = [
        name for name in postprocess_calls if name in SERVICE_RECEIPT_RECOVERY_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == SERVICE_RECEIPT_RECOVERY_PHASE_HELPER
    ]

    assert not direct_service_calls, (
        "Service receipt recovery repairs should run through the named phase "
        "helper so service-table layout, bare-service suppression, and "
        "single-service tax invariants have one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_service_calls}"
    )
    assert 0 < len(phase_calls) <= SERVICE_RECEIPT_RECOVERY_PHASE_CALL_LIMIT, (
        "Service receipt recovery phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {SERVICE_RECEIPT_RECOVERY_PHASE_CALL_LIMIT}"
    )


def test_body_total_layout_reconstruction_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, BODY_TOTAL_LAYOUT_RECONSTRUCTION_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        BODY_TOTAL_LAYOUT_RECONSTRUCTION_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Body-total layout reconstruction repairs must be owned by the named "
        f"{BODY_TOTAL_LAYOUT_RECONSTRUCTION_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{BODY_TOTAL_LAYOUT_RECONSTRUCTION_PHASE_HELPER} must document the "
        "visible body-total layout trigger and item/tax arithmetic invariant."
    )


def test_postprocess_body_total_layout_reconstruction_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_layout_calls = [
        name
        for name in postprocess_calls
        if name in BODY_TOTAL_LAYOUT_RECONSTRUCTION_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == BODY_TOTAL_LAYOUT_RECONSTRUCTION_PHASE_HELPER
    ]

    assert not direct_layout_calls, (
        "Body-total layout reconstruction should run through the named phase "
        "helper so split item rows, location, subtotal, and tax entries have "
        "one layout/arithmetic owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_layout_calls}"
    )
    assert 0 < len(phase_calls) <= BODY_TOTAL_LAYOUT_RECONSTRUCTION_PHASE_CALL_LIMIT, (
        "Body-total layout reconstruction phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {BODY_TOTAL_LAYOUT_RECONSTRUCTION_PHASE_CALL_LIMIT}"
    )


def test_ocr_description_reconciliation_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, OCR_DESCRIPTION_RECONCILIATION_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        OCR_DESCRIPTION_RECONCILIATION_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "OCR description reconciliation repairs must be owned by the named "
        f"{OCR_DESCRIPTION_RECONCILIATION_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{OCR_DESCRIPTION_RECONCILIATION_PHASE_HELPER} must document the OCR "
        "description-context trigger and item field-consistency invariant."
    )


def test_postprocess_ocr_description_reconciliation_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_description_calls = [
        name for name in postprocess_calls if name in OCR_DESCRIPTION_RECONCILIATION_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == OCR_DESCRIPTION_RECONCILIATION_PHASE_HELPER
    ]

    assert not direct_description_calls, (
        "OCR description reconciliation repairs should run through the named "
        "phase helper so code-row, duplicate, O-ring, colon-split, and bag "
        "description invariants have one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_description_calls}"
    )
    assert 0 < len(phase_calls) <= OCR_DESCRIPTION_RECONCILIATION_PHASE_CALL_LIMIT, (
        "OCR description reconciliation phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {OCR_DESCRIPTION_RECONCILIATION_PHASE_CALL_LIMIT}"
    )


def test_gap_item_recovery_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, GAP_ITEM_RECOVERY_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        GAP_ITEM_RECOVERY_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Gap item recovery repairs must be owned by the named "
        f"{GAP_ITEM_RECOVERY_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{GAP_ITEM_RECOVERY_PHASE_HELPER} must document the OCR row-gap "
        "trigger and subtotal/total arithmetic invariant."
    )


def test_postprocess_gap_item_recovery_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_gap_calls = [
        name for name in postprocess_calls if name in GAP_ITEM_RECOVERY_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == GAP_ITEM_RECOVERY_PHASE_HELPER
    ]

    assert not direct_gap_calls, (
        "Gap item recovery repairs should run through the named phase helper "
        "so missing, discounted, repeated, and repeated-block row recovery "
        "share one OCR/arithmetic owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_gap_calls}"
    )
    assert 0 < len(phase_calls) <= GAP_ITEM_RECOVERY_PHASE_CALL_LIMIT, (
        "Gap item recovery phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {GAP_ITEM_RECOVERY_PHASE_CALL_LIMIT}"
    )


def test_low_value_bag_recovery_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, LOW_VALUE_BAG_RECOVERY_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        LOW_VALUE_BAG_RECOVERY_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Low-value bag recovery repairs must be owned by the named "
        f"{LOW_VALUE_BAG_RECOVERY_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{LOW_VALUE_BAG_RECOVERY_PHASE_HELPER} must document the OCR small-bag "
        "trigger and subtotal/total arithmetic invariant."
    )


def test_postprocess_low_value_bag_recovery_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_bag_calls = [
        name for name in postprocess_calls if name in LOW_VALUE_BAG_RECOVERY_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == LOW_VALUE_BAG_RECOVERY_PHASE_HELPER
    ]

    assert not direct_bag_calls, (
        "Low-value bag recovery repairs should run through the named phase "
        "helper so missing bag rows, overage replacement, and numeric OCR "
        "context share one OCR/arithmetic owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_bag_calls}"
    )
    assert 0 < len(phase_calls) <= LOW_VALUE_BAG_RECOVERY_PHASE_CALL_LIMIT, (
        "Low-value bag recovery phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {LOW_VALUE_BAG_RECOVERY_PHASE_CALL_LIMIT}"
    )


def test_adjacent_price_shift_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, ADJACENT_PRICE_SHIFT_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        ADJACENT_PRICE_SHIFT_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Adjacent OCR price-shift repairs must be owned by the named "
        f"{ADJACENT_PRICE_SHIFT_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{ADJACENT_PRICE_SHIFT_PHASE_HELPER} must document the adjacent OCR "
        "row trigger and subtotal arithmetic invariant."
    )


def test_postprocess_adjacent_price_shift_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_price_shift_calls = [
        name for name in postprocess_calls if name in ADJACENT_PRICE_SHIFT_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == ADJACENT_PRICE_SHIFT_PHASE_HELPER
    ]

    assert not direct_price_shift_calls, (
        "Adjacent OCR price-shift repairs should run through the named phase "
        "helper so neighboring OCR row projection has one arithmetic owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_price_shift_calls}"
    )
    assert 0 < len(phase_calls) <= ADJACENT_PRICE_SHIFT_PHASE_CALL_LIMIT, (
        "Adjacent OCR price-shift phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {ADJACENT_PRICE_SHIFT_PHASE_CALL_LIMIT}"
    )


def test_bag_amount_shift_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, BAG_AMOUNT_SHIFT_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(
        BAG_AMOUNT_SHIFT_REPAIRS - set(_call_names_in_function(helper))
    )
    assert not missing_repairs, (
        "Paid-bag/product amount-shift repairs must be owned by the named "
        f"{BAG_AMOUNT_SHIFT_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{BAG_AMOUNT_SHIFT_PHASE_HELPER} must document the OCR paid-bag "
        "row trigger and subtotal/rate-base arithmetic invariant."
    )


def test_postprocess_bag_amount_shift_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_bag_shift_calls = [
        name for name in postprocess_calls if name in BAG_AMOUNT_SHIFT_REPAIRS
    ]
    phase_calls = [
        name
        for name in postprocess_calls
        if name == BAG_AMOUNT_SHIFT_PHASE_HELPER
    ]

    assert not direct_bag_shift_calls, (
        "Paid-bag/product amount-shift repairs should run through the named "
        "phase helper so OCR row order, rate bases, and subtotal arithmetic "
        "have one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_bag_shift_calls}"
    )
    assert 0 < len(phase_calls) <= BAG_AMOUNT_SHIFT_PHASE_CALL_LIMIT, (
        "Paid-bag/product amount-shift phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; "
        f"limit: {BAG_AMOUNT_SHIFT_PHASE_CALL_LIMIT}"
    )


def test_line_item_cleanup_phase_is_named_and_invariant_backed():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    helper = _function_def(tree, LINE_ITEM_CLEANUP_PHASE_HELPER)
    docstring = ast.get_docstring(helper) or ""

    missing_repairs = sorted(LINE_ITEM_CLEANUP_REPAIRS - set(_call_names_in_function(helper)))
    assert not missing_repairs, (
        "Line-item cleanup/drop repairs must be owned by the named "
        f"{LINE_ITEM_CLEANUP_PHASE_HELPER} helper.\n"
        f"Missing helper calls: {missing_repairs}"
    )
    assert "Trigger:" in docstring and "Invariant:" in docstring, (
        f"{LINE_ITEM_CLEANUP_PHASE_HELPER} must document the OCR/layout trigger "
        "and arithmetic or field-consistency invariant it enforces."
    )


def test_postprocess_line_item_cleanup_debt_is_phase_owned():
    tree = _parse_file(PARSER_DIR / "pipeline_receipt.py")
    postprocess = _function_def(tree, "postprocess_receipt")
    postprocess_calls = _call_names_in_function(postprocess)
    direct_cleanup_calls = [
        name for name in postprocess_calls if name in LINE_ITEM_CLEANUP_REPAIRS
    ]
    phase_calls = [
        name for name in postprocess_calls if name == LINE_ITEM_CLEANUP_PHASE_HELPER
    ]

    assert not direct_cleanup_calls, (
        "Line-item cleanup/drop repairs should run through the named phase "
        "helper so OCR/layout triggers and item-total invariants have one owner.\n"
        f"Direct calls still in postprocess_receipt: {direct_cleanup_calls}"
    )
    assert 0 < len(phase_calls) <= LINE_ITEM_CLEANUP_PHASE_CALL_LIMIT, (
        "Line-item cleanup phase calls must be explicit and bounded.\n"
        f"Current count: {len(phase_calls)}; limit: {LINE_ITEM_CLEANUP_PHASE_CALL_LIMIT}"
    )


def test_final_receipt_output_repairs_are_explicit_traced_stages():
    stages = _final_output_repair_stage_calls()
    stage_names = tuple(stage for stage, _line in stages)
    justifications = _final_output_repair_justifications()
    justification_keys = set(justifications)
    phase_names = _postprocess_phase_names()

    assert stage_names == FINAL_OUTPUT_REPAIR_STAGES, (
        "_apply_final_receipt_output_repairs changed. Late semantic repairs "
        "must stay behind the trace-recording run(stage, repair) wrapper and "
        "must update FINAL_OUTPUT_REPAIR_STAGES with a reviewable stage label.\n"
        f"Current stages: {stage_names}"
    )
    assert len(stage_names) == len(set(stage_names)), (
        "Late repair stage labels must be unique so mutation traces are useful."
    )
    assert justification_keys == set(stage_names), (
        "Every late final-output repair must have an explicit owner-phase "
        "justification in FINAL_RECEIPT_OUTPUT_REPAIR_JUSTIFICATIONS.\n"
        f"Missing: {sorted(set(stage_names) - justification_keys)}\n"
        f"Stale: {sorted(justification_keys - set(stage_names))}"
    )
    malformed = {
        stage: value
        for stage, value in justifications.items()
        if (
            not isinstance(value, tuple)
            or len(value) != 2
            or value[0] not in phase_names
            or not isinstance(value[1], str)
            or not value[1].strip()
        )
    }
    assert not malformed, (
        "Every late repair justification must name a valid postprocess owner "
        "phase and a non-empty reason.\n"
        f"Malformed: {malformed}"
    )


def test_no_new_suspicious_japanese_product_or_location_literals():
    baseline = _baseline_japanese_string_counts()
    current = _current_japanese_string_counts()
    new_literals = []
    for signature, count in current.items():
        extra = count - baseline.get(signature, 0)
        if extra <= 0:
            continue
        path, value = signature
        if _looks_structural_japanese_literal(value):
            continue
        new_literals.append((path, value, extra))

    assert not new_literals, (
        "Production parser code gained Japanese literals that do not look like "
        "structural receipt labels. Do not add product, merchant, location, or "
        "answer-key strings to parser code; derive behavior from OCR structure "
        "and arithmetic invariants instead.\n"
        + "\n".join(
            f"  - {path}: {value!r} (+{extra})"
            for path, value, extra in new_literals[:40]
        )
    )


def test_postprocess_receipt_is_idempotent_at_guardrail_level():
    from receipt_parser.pipeline_receipt import (
        _snapshot_receipt_mutation_fields,
        postprocess_receipt,
    )

    extracted = {
        "document_type": "receipt",
        "merchant": "テスト店",
        "currency": "JPY",
        "total": 1100,
        "subtotal": 1000,
        "taxes": [{"rate": "10%", "label": "外税", "amount": 100}],
        "line_items": [
            {
                "description": "テスト商品",
                "qty": 1,
                "unit_price": 1000,
                "total": 1000,
                "tax_category": "10%",
            },
        ],
        "points_used": 0,
    }
    text = "テスト店\nテスト商品\n¥1,000\n小計\n¥1,000\n外税\n¥100\n合計\n¥1,100"

    postprocess_receipt(extracted, text, 0.9, {}, {}, "test-model")
    once = copy.deepcopy(_snapshot_receipt_mutation_fields(extracted))
    postprocess_receipt(extracted, text, 0.9, {}, {}, "test-model")
    twice = _snapshot_receipt_mutation_fields(extracted)

    assert twice == once
