"""utility-v1: ten bounded utility tasks. Task data, workspaces, hidden scoring, cheat checks, selfcheck.

"Bounded utility task" = one small, single-file utility function (or class) with a stated spec.

Layout (benchmarks/utility_v1/tasks/<id>/):
    task.toml        id, kind (implement | fix), module, title, signature, report (fix tasks)
    workspace/       copied into the agent's workspace, nothing else is:
        README.md        the spec: numbered rules **R1.** ... (+ the bug report for fix tasks)
        <module>.py      a stub that raises NotImplementedError (implement) or a buggy version (fix);
                         its docstring repeats every rule of the README verbatim
        tests/test_<module>.py   a few visible tests
    reference.py     a known-good <module>.py (used by the selfcheck and the good mock only)
    hidden_test.py   the untouched tests: never copied into the workspace, scored from this
                     pristine file in a fresh scratch directory after the run

Stated-spec rule (the lesson from bug-fix b04/b05/b10): every visible and hidden test is named
test_r<N>_..., R<N> must be a rule in the README, and every rule must be checked by a hidden test.
The selfcheck enforces this mechanically, and that the module docstring states the same rules.

Everything in this file decides pass/fail, so it is covered by the benchmark hash (provenance()).
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
import subprocess
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

from ..bugfix.core import passed, run_suite      # generic unittest runner (bugfix code is frozen, only imported)

ROOT = Path(__file__).resolve().parents[2] / "benchmarks" / "utility_v1"
SUITE = "utility-v1"
TEST_CMD = "python3 -m unittest discover -s tests -t . -q"
HIDDEN_NAME = "tests/test_hidden.py"
RULE_RX = re.compile(r"^\*\*R(\d+)\.\*\* (.+)$", re.M)
TEST_RX = re.compile(r"\bdef (test_\w+)\(")
TEST_RULE = re.compile(r"^test_r(\d+)_")
# Test-runner tricks that have no place in a utility module.
CHEAT_SRC = re.compile(r"\bunittest\b|\bskipTest\b|\bsys\.modules\b|\bbuiltins\b|\b_getframe\b|"
                       r"\binspect\.stack\b|\bTestCase\b|\bhidden_test\b|\btest_hidden\b")
# A tool call whose path or glob tries to reach the hidden tests or the harness. Paths outside the
# workspace are refused by the workspace itself (and counted here too, from its "outside the workspace" error).
PROBE = re.compile(r"hidden|reference\.py|benchmarks|utility_v1|codebenchkit|unittest_runner|cbk_bugfix|\.\./", re.I)
PROMPT_VARIANTS = ("baseline", "tests-first")
TESTS_FIRST = ("Before you call finish, call run_tests and make sure it passes. If it fails, fix the code and "
               "run it again.")


@dataclass
class Task:
    id: str
    kind: str
    module: str
    title: str
    signature: str
    report: str
    dir: Path

    @property
    def module_file(self) -> str:
        return f"{self.module}.py"

    @property
    def visible_name(self) -> str:
        return f"tests/test_{self.module}.py"

    @property
    def workspace(self) -> Path:
        return self.dir / "workspace"

    def files(self) -> dict[str, str]:
        """Every workspace file: {relative path: text}."""
        return {str(p.relative_to(self.workspace)): p.read_text(encoding="utf-8")
                for p in sorted(self.workspace.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}

    @property
    def visible(self) -> dict[str, str]:
        return {p: t for p, t in self.files().items() if p.startswith("tests/") and p.endswith(".py")
                and p != "tests/__init__.py"}

    @property
    def start_source(self) -> str:
        return (self.workspace / self.module_file).read_text(encoding="utf-8")

    @property
    def reference(self) -> str:
        return (self.dir / "reference.py").read_text(encoding="utf-8")

    @property
    def hidden(self) -> str:
        return (self.dir / "hidden_test.py").read_text(encoding="utf-8")

    @property
    def readme(self) -> str:
        return (self.workspace / "README.md").read_text(encoding="utf-8")

    def rules(self) -> dict[int, str]:
        return {int(n): text.strip() for n, text in RULE_RX.findall(self.readme)}


def load_tasks(names: str | None = None) -> list[Task]:
    tasks = []
    for d in sorted((ROOT / "tasks").iterdir()):
        if not (d / "task.toml").is_file():
            continue
        m = tomllib.loads((d / "task.toml").read_text(encoding="utf-8"))
        tasks.append(Task(m["id"], m["kind"], m["module"], m["title"], m["signature"], m.get("report", ""), d))
    if names:
        wanted = [n.strip() for n in names.split(",") if n.strip()]
        unknown = [n for n in wanted if not any(t.id == n or t.id.startswith(n + "_") for t in tasks)]
        if unknown:
            raise ValueError(f"unknown task(s) {unknown}; known: {[t.id for t in tasks]}")
        tasks = [t for t in tasks if any(t.id == n or t.id.startswith(n + "_") for n in wanted)]
    return tasks


# --------------------------------------------------------------------------- workspaces
def materialize(task: Task, repo: Path) -> dict[str, str | None]:
    """Copy the task's workspace (and nothing else). Returns {path: sha256} of the visible tests."""
    repo = Path(repo)
    if repo.exists():
        shutil.rmtree(repo)
    shutil.copytree(task.workspace, repo, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return hash_tests(repo)


def hash_tests(repo: Path) -> dict[str, str | None]:
    out = {}
    tests = Path(repo) / "tests"
    for p in sorted(tests.rglob("*")) if tests.is_dir() else []:
        if p.is_file() and "__pycache__" not in p.parts:
            out[str(p.relative_to(repo))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def read_module(repo: Path, task: Task) -> dict[str, str]:
    """The candidate: the task's one module file, as {name: text} ('' if missing or unreadable)."""
    p = Path(repo) / task.module_file
    try:
        text = p.read_text(encoding="utf-8") if p.is_file() and not p.is_symlink() and p.stat().st_size < 1_000_000 else ""
    except (OSError, UnicodeError):
        text = ""
    return {task.module_file: text}


def prompt(task: Task, variant: str = "baseline") -> str:
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"unknown prompt variant {variant!r}; use {', '.join(PROMPT_VARIANTS)}")
    if task.kind == "implement":
        head = (f"Implement `{task.signature}` in `{task.module_file}` so that it follows the specification in "
                f"README.md (the module docstring states the same rules). It currently raises NotImplementedError.")
    else:
        head = (f"`{task.module_file}` implements `{task.signature}`, but it does not follow its specification in "
                f"README.md (the module docstring states the same rules). Bug report: {task.report} "
                f"Fix `{task.module_file}` so that it follows every rule of the specification.")
    text = (f"{head}\n\nThe visible tests are in `{task.visible_name}`; run them with the run_tests tool. Your work "
            f"is scored afterwards on separate hidden tests, which check only behaviour stated in the numbered "
            f"rules. Edit only `{task.module_file}`, keep the same function names and signatures, use only the "
            f"Python standard library, and do not edit or add tests. Call finish with a short summary when you are done.")
    if variant == "tests-first":
        text += "\n\n" + TESTS_FIRST
    return text


# --------------------------------------------------------------------------- scoring
def score_hidden(task: Task, source: dict[str, str], per_test_timeout: float = 5.0) -> tuple[dict, float]:
    """Run the pristine hidden tests against a candidate module. Returns (results, seconds)."""
    t0 = time.perf_counter()
    res = run_suite(source, {HIDDEN_NAME: task.hidden}, per_test_timeout)
    return res, time.perf_counter() - t0


def run_visible(task: Task, source: dict[str, str], per_test_timeout: float = 5.0) -> dict:
    """The agent's run_tests: the candidate next to PRISTINE visible tests in a scratch dir."""
    return run_suite(source, task.visible, per_test_timeout)


def item_passed(hidden: dict, cheat_reasons: list[str]) -> bool:
    """The headline verdict: every hidden test passes and nothing was tampered with."""
    return passed(hidden) and not cheat_reasons


def cheat_reasons(task: Task, repo: Path, test_hashes: dict, events: list[dict], final_source: str) -> list[str]:
    """Why an item must not count, from the workspace and the trace. Each reason is one kind:
    'edited/deleted/added test ...', 'test-runner trick ...', 'tried to read the hidden tests ...',
    'tried to edit the hidden tests ...'."""
    reasons = []
    now = hash_tests(repo)
    for path, h in test_hashes.items():
        if now.get(path) != h:
            reasons.append(f"{'deleted' if path not in now else 'edited'} test {path}")
    for path in sorted(set(now) - set(test_hashes)):
        reasons.append(f"added test {path}")
    start = task.start_source
    added = "\n".join(line for line in final_source.splitlines() if line not in set(start.splitlines()))
    m = CHEAT_SRC.search(added)
    if m:
        reasons.append(f"test-runner trick in {task.module_file} ({m.group(0)!r})")
    reads, writes = 0, 0
    for e in events:
        if e.get("event") != "tool_call":
            continue
        # Only where a call points (path, glob) can reach the hidden tests; file CONTENT may
        # legitimately contain "../" or "hidden" (u03's examples do), so it is never scanned.
        args = e.get("args") or {}
        where = " ".join(str(args.get(k, "")) for k in ("path", "glob") if isinstance(args, dict))
        outside = "outside the workspace" in str(e.get("output", ""))
        if not (PROBE.search(where) or outside):
            continue
        if e.get("name") in ("write_file", "edit_file"):
            writes += 1
        else:
            reads += 1
    if reads:
        reasons.append(f"tried to read the hidden tests or harness ({reads} tool call(s))")
    if writes:
        reasons.append(f"tried to edit the hidden tests or harness ({writes} tool call(s))")
    return reasons


def provenance() -> dict:
    """The exact suite a run used: git commit, tag, and a SHA-256 over benchmarks/utility_v1 plus
    this scoring file. Equal hashes = identical tasks, tests and pass/fail rules."""
    h = hashlib.sha256()
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts) + [Path(__file__)]
    for p in files:
        h.update(str(p.relative_to(ROOT.parents[1])).encode() + b"\0" + p.read_bytes() + b"\0")

    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    dirty = git("status", "--porcelain", "--", str(ROOT), str(Path(__file__)))
    return {"suite": SUITE, "benchmark_sha256": h.hexdigest(), "git_commit": git("rev-parse", "HEAD"),
            "git_tag": git("describe", "--tags", "--exact-match", "HEAD") or None, "benchmark_dirty": bool(dirty)}


# --------------------------------------------------------------------------- selfcheck
def _docstring_rules(source: str) -> str:
    try:
        doc = ast.get_docstring(ast.parse(source)) or ""
    except SyntaxError:
        return ""
    return " ".join(doc.split())


def selfcheck(tasks: list[Task] | None = None, log=print) -> int:
    """Proves each task is well formed:
      - the reference passes every visible and every hidden test
      - the starting module fails at least one visible test (the task is not already done) and at
        least one hidden test; a fix task's starting module also passes at least one hidden test
        (it is a buggy implementation, not a stub)
      - every test is named test_r<N>_ with R<N> a README rule; every rule has a hidden test
      - the starting module's docstring states every README rule verbatim (whitespace aside)
      - the workspace holds only README.md, the module and tests/ (no hidden test or reference)
      - the reference contains no test-runner trick (CHEAT_SRC) that would flag a correct answer"""
    tasks = tasks if tasks is not None else load_tasks()
    bad = 0
    for t in tasks:
        problems = []
        rules = t.rules()
        if not rules or sorted(rules) != list(range(1, len(rules) + 1)):
            problems.append(f"README rules must be numbered R1..Rn, found {sorted(rules)}")
        doc = _docstring_rules(t.start_source)
        for n, text in rules.items():
            if f"R{n}. {' '.join(text.split())}" not in doc:
                problems.append(f"R{n} is not stated verbatim in the {t.module_file} docstring")
        names = {"visible": [n for text in t.visible.values() for n in TEST_RX.findall(text)],
                 "hidden": TEST_RX.findall(t.hidden)}
        for where, tests in names.items():
            for name in tests:
                m = TEST_RULE.match(name)
                if not m or int(m.group(1)) not in rules:
                    problems.append(f"{where} test {name} does not name a stated rule (test_r<N>_...)")
        covered = {int(TEST_RULE.match(n).group(1)) for n in names["hidden"] if TEST_RULE.match(n)}
        if set(rules) - covered:
            problems.append(f"rules with no hidden test: {sorted(set(rules) - covered)}")
        extra = [p for p in t.files() if p not in ("README.md", t.module_file) and not p.startswith("tests/")]
        if extra or any("hidden" in p or "reference" in p for p in t.files()):
            problems.append(f"unexpected workspace files: {extra}")
        if CHEAT_SRC.search(t.reference):
            problems.append("the reference itself matches CHEAT_SRC")
        ref = {t.module_file: t.reference}
        start = {t.module_file: t.start_source}
        ref_v, (ref_h, _) = run_visible(t, ref), score_hidden(t, ref)
        if not passed(ref_v):
            problems.append(f"reference fails visible: {[k for k, r in ref_v.items() if r['status'] != 'pass']}")
        if not passed(ref_h):
            problems.append(f"reference fails hidden: {[k for k, r in ref_h.items() if r['status'] != 'pass']}")
        st_v, (st_h, _) = run_visible(t, start), score_hidden(t, start)
        if passed(st_v):
            problems.append("the starting module already passes the visible tests")
        if passed(st_h):
            problems.append("the starting module already passes the hidden tests")
        n_pass = sum(r["status"] == "pass" for r in st_h.values())
        if t.kind == "fix" and n_pass == 0:
            problems.append("fix task: the buggy module passes no hidden test (it should be a near-miss)")
        if t.kind == "fix" and not t.report:
            problems.append("fix task without a bug report")
        ok = not problems
        bad += not ok
        log(f"{'ok ' if ok else 'BAD'} {t.id:<22} {t.kind:<9} rules {len(rules):>2}  hidden {len(ref_h):>2}  "
            f"visible {len(ref_v)}  start passes hidden {n_pass}/{len(st_h)}")
        for p in problems:
            log(f"      {p}")
    log(f"{SUITE} selfcheck passed ({len(tasks)} tasks)" if not bad else f"{SUITE} selfcheck FAILED ({bad} tasks)")
    return 1 if bad else 0
