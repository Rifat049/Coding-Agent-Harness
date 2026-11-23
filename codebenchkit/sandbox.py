"""Run candidate code against tests in a separate, resource-limited Python process.

Isolation here is best-effort: an isolated interpreter (-I), a scratch working dir, a
stripped environment, per-test wall-clock timeouts, and CPU/memory/file-size rlimits on
POSIX. It stops accidents (infinite loops, memory blowups, stray prints), not a determined
attacker. For untrusted models at scale, run the whole harness inside a container.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

RUNNER = r'''
import copy, json, signal, sys, time

def main():
    src_path, tests_path, out_path, fn_name, per_test = sys.argv[1:6]
    per_test = float(per_test)
    out = open(out_path, "w")            # opened BEFORE candidate code runs
    dumps = json.dumps
    def emit(obj):
        out.write(dumps(obj)); out.flush(); out.close()
    with open(tests_path) as f:
        tests = json.load(f)
    with open(src_path) as f:
        src = f.read()

    class _Timeout(BaseException):
        pass
    def _alarm(signum, frame):
        raise _Timeout()
    has_alarm = hasattr(signal, "setitimer")
    if has_alarm:
        signal.signal(signal.SIGALRM, _alarm)

    ns = {"__name__": "candidate"}
    try:
        if has_alarm: signal.setitimer(signal.ITIMER_REAL, per_test)
        exec(compile(src, "candidate.py", "exec"), ns)
        if has_alarm: signal.setitimer(signal.ITIMER_REAL, 0)
    except _Timeout:
        emit({"status": "timeout", "detail": "module load timed out"}); return
    except BaseException as e:
        if has_alarm: signal.setitimer(signal.ITIMER_REAL, 0)
        emit({"status": "load_error", "detail": f"{type(e).__name__}: {e}"[:500]}); return

    fn = ns.get(fn_name)
    if not callable(fn):
        emit({"status": "missing_function", "detail": fn_name}); return

    def norm(x):
        return json.loads(dumps(x))

    cases = []
    for t in tests:
        args = copy.deepcopy(t["args"])
        rec = {"passed": False}
        t0 = time.perf_counter()
        try:
            if has_alarm: signal.setitimer(signal.ITIMER_REAL, per_test)
            got = fn(*args)
            if has_alarm: signal.setitimer(signal.ITIMER_REAL, 0)
            rec["time_s"] = time.perf_counter() - t0
            if "raises" in t:
                rec["error"] = f"expected {t['raises']} but returned {got!r}"[:300]
            else:
                try:
                    got_n = norm(got)
                except Exception:
                    got_n = None
                    rec["error"] = f"returned non-JSON-serialisable value {got!r}"[:300]
                rec["got"] = repr(got)[:300]
                rec["passed"] = got_n == t["expected"] and "error" not in rec
        except _Timeout:
            rec["time_s"] = time.perf_counter() - t0
            rec["error"] = f"timed out after {per_test}s"
            rec["timeout"] = True
        except BaseException as e:
            if has_alarm: signal.setitimer(signal.ITIMER_REAL, 0)
            rec["time_s"] = time.perf_counter() - t0
            names = [c.__name__ for c in type(e).__mro__]
            if "raises" in t and t["raises"] in names:
                rec["passed"] = True
            else:
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
        cases.append(rec)
    emit({"status": "ok", "cases": cases})

main()
'''


@dataclass
class ExecResult:
    status: str                    # ok | load_error | missing_function | timeout | crash
    cases: list = field(default_factory=list)
    detail: str = ""
    runtime_s: float = 0.0         # wall time for the whole subprocess

    @property
    def n_total(self) -> int:
        return len(self.cases)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.cases if c.get("passed"))

    def all_passed(self, n_expected: int) -> bool:
        return self.status == "ok" and self.n_passed == n_expected

    @property
    def any_timeout(self) -> bool:
        return self.status == "timeout" or any(c.get("timeout") for c in self.cases)

    def to_dict(self) -> dict:
        return {"status": self.status, "detail": self.detail, "runtime_s": round(self.runtime_s, 4),
                "n_passed": self.n_passed, "n_total": self.n_total, "cases": self.cases}


def _limits(mem_mb: int, cpu_s: int):
    def apply():  # runs in the child before exec, POSIX only
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 1))
            resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
            if sys.platform.startswith("linux"):  # RLIMIT_AS is unreliable on macOS
                resource.setrlimit(resource.RLIMIT_AS, (mem_mb << 20, mem_mb << 20))
        except Exception:
            pass
    return apply


def run_tests(code: str, fn_name: str, tests: list, per_test_timeout: float = 2.0,
              mem_mb: int = 1024) -> ExecResult:
    n = max(1, len(tests))
    wall_timeout = per_test_timeout * (n + 1) + 3
    with tempfile.TemporaryDirectory(prefix="cbk_") as d:
        src, tst, res, run = (os.path.join(d, x) for x in ("candidate.py", "tests.json", "result.json", "runner.py"))
        with open(src, "w") as f:
            f.write(code)
        with open(tst, "w") as f:
            json.dump(tests, f)
        with open(run, "w") as f:
            f.write(RUNNER)
        env = {"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
        kwargs = {}
        if os.name == "posix":
            kwargs["preexec_fn"] = _limits(mem_mb, int(wall_timeout) + 1)
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, "-I", run, src, tst, res, fn_name, str(per_test_timeout)],
                cwd=d, env=env, capture_output=True, timeout=wall_timeout, stdin=subprocess.DEVNULL, **kwargs,
            )
        except subprocess.TimeoutExpired:
            return ExecResult("timeout", detail=f"process exceeded {wall_timeout:.0f}s", runtime_s=time.perf_counter() - t0)
        elapsed = time.perf_counter() - t0
        try:
            with open(res) as f:
                data = json.load(f)
        except Exception:
            err = proc.stderr.decode("utf-8", "replace")[-500:]
            return ExecResult("crash", detail=f"exit {proc.returncode}: {err}".strip(), runtime_s=elapsed)
    return ExecResult(data["status"], data.get("cases", []), data.get("detail", ""), elapsed)
