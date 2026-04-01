from collections import defaultdict
from pathlib import Path

# Load .env file so tests pick up GOOGLE_CLOUD_PROJECT etc.
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")


# ---------------------------------------------------------------------------
# Accuracy summary plugin — prints per-field pass rates after test_accuracy
# ---------------------------------------------------------------------------

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print accuracy summary table after test_accuracy.py runs."""
    try:
        from test_accuracy import _check_results
    except ImportError:
        return

    if not _check_results:
        return

    # Aggregate
    field_stats = defaultdict(lambda: {"passed": 0, "total": 0})
    image_fixtures = set()
    variant_fixtures = set()
    failed_cases = []

    for cr in _check_results:
        field_stats[cr["field"]]["total"] += 1
        if cr["pass"]:
            field_stats[cr["field"]]["passed"] += 1
        else:
            failed_cases.append(cr)

        if cr["source_type"] == "image":
            image_fixtures.add(cr["fixture"])
        else:
            variant_fixtures.add(cr["fixture"])

    total_checks = len(_check_results)
    total_passed = sum(1 for cr in _check_results if cr["pass"])

    # Count fixture-level pass/fail
    fixture_results = defaultdict(lambda: {"passed": True})
    for cr in _check_results:
        if not cr["pass"]:
            fixture_results[cr["fixture"]]["passed"] = False

    image_passed = sum(1 for f in image_fixtures if fixture_results[f]["passed"])
    variant_passed = sum(1 for f in variant_fixtures if fixture_results[f]["passed"])

    terminalreporter.section("Accuracy Summary")
    terminalreporter.write_line(
        f"Image fixtures:  {len(image_fixtures)} tested, "
        f"{image_passed} passed, {len(image_fixtures) - image_passed} failed"
    )
    if variant_fixtures:
        terminalreporter.write_line(
            f"OCR variants:    {len(variant_fixtures)} tested, "
            f"{variant_passed} passed, {len(variant_fixtures) - variant_passed} failed"
        )
    total_fixtures = len(image_fixtures) + len(variant_fixtures)
    total_fix_passed = image_passed + variant_passed
    pct = total_passed / total_checks * 100 if total_checks else 100
    terminalreporter.write_line(
        f"Total:           {total_fixtures} tested, "
        f"{total_fix_passed} passed, {total_fixtures - total_fix_passed} failed "
        f"({pct:.1f}% field checks)"
    )

    if failed_cases:
        terminalreporter.write_line("")
        terminalreporter.write_line("Failed:")
        # Group by fixture
        by_fixture = defaultdict(list)
        for fc in failed_cases:
            by_fixture[fc["fixture"]].append(fc)
        for fixture, checks in sorted(by_fixture.items()):
            fields = ", ".join(f"{c['field']}: {c['detail']}" for c in checks)
            terminalreporter.write_line(f"  {fixture:25s} {fields}")

    terminalreporter.write_line("")
    terminalreporter.write_line("Per-field pass rate:")
    for field, stats in sorted(field_stats.items(), key=lambda x: x[1]["passed"] / max(x[1]["total"], 1)):
        p = stats["passed"]
        t = stats["total"]
        pct = p / t * 100 if t else 100
        terminalreporter.write_line(f"  {field:25s} {p}/{t}  {pct:.0f}%")
