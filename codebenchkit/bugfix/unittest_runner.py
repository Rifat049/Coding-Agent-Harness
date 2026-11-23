"""Run unittest files one test at a time with a per-test timeout; write results as JSON.

usage: python3 -I unittest_runner.py <root> <per_test_timeout_s> <out.json> <test file>...

Executed in a child process by core.run_suite, inside a scratch copy of the candidate. Each test
gets SIGALRM after the timeout, so an accidentally quadratic fix fails as "timeout" instead of
hanging the benchmark.
"""
import importlib.util
import json
import os
import signal
import sys
import traceback
import unittest


class _Timeout(BaseException):
    pass


def _iter(suite):
    for t in suite:
        if isinstance(t, unittest.TestSuite):
            yield from _iter(t)
        else:
            yield t


def main() -> None:
    root, per_test, out_path, files = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4:]
    sys.path.insert(0, root)
    os.chdir(root)
    results = {}
    state = {"fired": False}

    def on_alarm(signum, frame):
        state["fired"] = True
        raise _Timeout()

    signal.signal(signal.SIGALRM, on_alarm)
    for n, path in enumerate(files):
        try:
            spec = importlib.util.spec_from_file_location(f"cbk_test_{n}", os.path.join(root, path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            tests = list(_iter(unittest.defaultTestLoader.loadTestsFromModule(mod)))
        except BaseException:  # noqa: BLE001 - an import error fails the whole file
            results[f"{path}::<import>"] = {"status": "error", "detail": traceback.format_exc()[-1500:]}
            continue
        for test in tests:
            tid = f"{path}::{test.id().split('.', 1)[-1]}"
            res = unittest.TestResult()
            state["fired"] = False
            signal.setitimer(signal.ITIMER_REAL, per_test)
            try:
                test.run(res)
            except _Timeout:
                pass
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            if state["fired"]:
                status, detail = "timeout", f"exceeded {per_test:g}s"
            elif res.skipped:
                status, detail = "skip", res.skipped[0][1]
            elif res.failures:
                status, detail = "fail", res.failures[0][1]
            elif res.errors:
                status, detail = "error", res.errors[0][1]
            elif res.expectedFailures or res.unexpectedSuccesses:
                status, detail = "fail", "expected-failure markers are not allowed"
            else:
                status, detail = "pass", ""
            results[tid] = {"status": status, "detail": detail[-1500:]}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
