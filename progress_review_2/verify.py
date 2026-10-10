"""Run substantive checks and save readable and machine-readable evidence."""
import io
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent


def verify():
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
    output = io.StringIO()
    class RecordedResults(unittest.TextTestResult):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.passed = []
        def addSuccess(self, test):
            super().addSuccess(test)
            self.passed.append(test.id())
    result = unittest.TextTestRunner(stream=output, verbosity=2, resultclass=RecordedResults).run(suite)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/test_log.txt").write_text(output.getvalue(), encoding="utf-8")
    payload = {"passed": result.wasSuccessful(), "tests_run": result.testsRun,
        "successful_tests": result.passed, "failures": [str(t) for t, _ in result.failures], "errors": [str(t) for t, _ in result.errors],
        "scope": "Checks run in the recorded Python environment; not a guarantee of zero defects or Windows execution."}
    (ROOT / "results/verification.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output.getvalue())
    if not result.wasSuccessful():
        raise RuntimeError("Verification failed; see results/test_log.txt")
    return payload


if __name__ == "__main__":
    verify()
