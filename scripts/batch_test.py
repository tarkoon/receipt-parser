"""Run integration tests in batches to avoid memory/timeout issues."""
import subprocess
import sys
import re
from pathlib import Path

RESULTS_FILE = Path(__file__).parent / "batch_test_results.txt"

BATCHES = [
    "receipt_1 or receipt_10 or receipt_11 or receipt_12 or receipt_13 or receipt_14 or receipt_15 or receipt_16 or receipt_17",
    "receipt_18 or receipt_19 or receipt_2 or receipt_20 or receipt_21 or receipt_22 or receipt_23 or receipt_24 or receipt_25",
    "receipt_26 or receipt_27 or receipt_28 or receipt_29 or receipt_3 or receipt_30 or receipt_31 or receipt_32 or receipt_33",
    "receipt_34 or receipt_35 or receipt_36 or receipt_4 or receipt_5 or receipt_6 or receipt_7 or receipt_8 or receipt_9",
]

def log(msg):
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)

# Clear results file
RESULTS_FILE.write_text("", encoding="utf-8")

all_failed = []
all_passed = 0
all_total_failed = 0

for i, expr in enumerate(BATCHES, 1):
    log(f"\n{'='*60}")
    log(f"BATCH {i}/4")
    log(f"{'='*60}")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_accuracy.py",
             "-v", "--tb=line", "-k", expr],
            capture_output=True, text=True, timeout=1500,
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        log(f"BATCH {i} TIMED OUT after 1500s")
        continue

    # Log the short summary
    for line in output.split("\n"):
        if "PASSED" in line or "FAILED" in line:
            log(line.strip())

    # Parse results
    for line in output.split("\n"):
        if line.startswith("FAILED "):
            all_failed.append(line.strip())
    m = re.search(r"(\d+) passed", output)
    if m:
        all_passed += int(m.group(1))
    m = re.search(r"(\d+) failed", output)
    if m:
        all_total_failed += int(m.group(1))

    log(f"\nBatch {i} done: see above")

log(f"\n{'='*60}")
log(f"FINAL SUMMARY: {all_passed} passed, {all_total_failed} failed")
log(f"{'='*60}")
if all_failed:
    log("\nAll failures:")
    for f in all_failed:
        log(f"  {f}")
else:
    log("\nAll tests passed!")
