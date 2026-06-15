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
import io
import re
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
