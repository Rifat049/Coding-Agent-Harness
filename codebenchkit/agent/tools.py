"""Coding tools, scoped to one workspace directory, behind a permission policy.

Design rules (each one fixes a failure coding agents hit in practice):
  * Every path is resolved inside the workspace root. `..`, absolute paths and symlinks that
    lead out are refused, because the model should not be able to touch anything else.
  * Edits are exact-string replacements that must match exactly once. The model has to read
    before it writes, so it cannot clobber the wrong spot.
  * Outputs are capped, and a capped output says so, so the model knows to narrow its request
    instead of trusting a partial view.
  * A tool never raises into the loop. Failures come back as "ERROR: ..." text the model can
    read and react to.
  * Commands pass a policy check first: readonly / auto (deny-list) / ask (a human approves).
  * Child processes get an environment with API keys removed, so `env` cannot leak secrets into
    the trace or into memory.
"""
from __future__ import annotations

import fnmatch
import inspect
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache",
             "dist", "build", ".idea", ".cbk", ".tox", ".ruff_cache", ".hg", ".svn"}
MAX_OUTPUT = 12000
MAX_NOTES = 10

# Deny-list for `auto` mode. It catches accidents and obvious destruction. It is not a sandbox.
DENY_PATTERNS: list[tuple[str, str]] = [
    (r"\brm\s+(-[^\s]*\s+)*-[a-zA-Z]*[rR][a-zA-Z]*\s+(-[^\s]*\s+)*['\"]?(/|~|\$HOME|\.\.)/?\*?['\"]?(\s|;|&|\||$)",
     "recursive delete of /, ~, $HOME or .."),
    (r"--no-preserve-root", "rm --no-preserve-root"),
    (r"(^|[;&|]\s*|\s)sudo\b", "sudo"),
    (r"\bmkfs(\.\w+)?\b", "mkfs"),
    (r"\bdd\s+[^|;]*\bof=/dev/", "dd to a device"),
    (r">\s*/dev/(sd|nvme|disk)", "write to a raw device"),
    (r"\b(shutdown|reboot|halt|poweroff)\b", "shutdown/reboot"),
    (r":\(\)\s*\{\s*:\|:&\s*\};:", "fork bomb"),
    (r"\b(curl|wget)\b[^|;]*\|\s*(sudo\s+)?(ba|z|da|k)?sh\b", "pipe a download into a shell"),
    (r"\b(curl|wget)\b[^|;]*\|\s*(sudo\s+)?python3?\b", "pipe a download into python"),
    (r"\bgit\s+push\b", "git push"),
    (r"\bgit\s+reset\s+--hard\b", "git reset --hard"),
    (r"\bgit\s+clean\s+-[a-zA-Z]*f", "git clean -f"),
    (r"\bgit\s+checkout\s+(--\s+)?\.(\s|$)", "git checkout . (discards all changes)"),
    (r"\bchmod\s+(-R\s+)?777\s+/", "chmod 777 /"),
    (r"\b(npm|pnpm|yarn)\s+publish\b|\btwine\s+upload\b|\bcargo\s+publish\b", "publishing a package"),
]
SECRET_ENV = re.compile(r"(API_KEY|_TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.IGNORECASE)


class ToolDenied(Exception):
    pass


@dataclass
class Policy:
    mode: str = "auto"                                     # readonly | auto | ask
    approve: Callable[[str, str], bool] | None = None      # (tool, description) -> bool, for "ask"
    extra_deny: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.mode not in ("readonly", "auto", "ask"):
            raise ValueError(f"unknown approval mode {self.mode!r}")

    def check_write(self, what: str):
        if self.mode == "readonly":
            raise ToolDenied(f"workspace is read-only; refused: {what}")

    def denied_reason(self, command: str) -> str | None:
        for pat, why in DENY_PATTERNS + [(p, p) for p in self.extra_deny]:
            if re.search(pat, command):
                return why
        return None

    def check_command(self, command: str):
        if self.mode == "readonly":
            raise ToolDenied("workspace is read-only; commands are disabled")
        why = self.denied_reason(command)
        if why:
            raise ToolDenied(f"command blocked by policy ({why}). Find a safer way, or ask the user to run it.")
        if self.mode == "ask":
            if not self.approve or not self.approve("run_command", command):
                raise ToolDenied("the user declined this command")


def terminal_approver(tool: str, description: str) -> bool:
    """`ask` mode: show the command, wait for y/N on the terminal."""
    try:
        ans = input(f"\n[approve {tool}] {description}\n  run it? [y/N] ")
    except EOFError:
        return False
    return ans.strip().lower() in ("y", "yes")


@dataclass
class VerifyResult:
    passed: bool
    tail: str
    seconds: float
    source: str = "tool"            # "tool" (model called run_tests) | "harness" (stop hook)


@dataclass
class RunState:
    """What the tools did this run; the episode is written from it, never from the model's claims."""
    files_changed: set = field(default_factory=set)
    commands: list = field(default_factory=list)
    verify_results: list = field(default_factory=list)    # VerifyResult
    notes: list = field(default_factory=list)
    recalled: list = field(default_factory=list)          # fact ids the model pulled with recall
    finished: bool = False
    finish_summary: str = ""
    edits_since_verify: int = 0
    # Set by write_file/edit_file on a .py file: the SyntaxError it now has, or "". Measurement only
    # (the trace and the invalid-output metric); the model's tool output is unchanged.
    last_syntax_error: str = ""

    def mark_edit(self, rel: str):
        self.files_changed.add(rel)
        self.edits_since_verify += 1

    @property
    def last_verify(self) -> VerifyResult | None:
        return self.verify_results[-1] if self.verify_results else None


def clean_env() -> dict:
    env = {k: v for k, v in os.environ.items() if not SECRET_ENV.search(k)}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _decode(x) -> str:
    if x is None:
        return ""
    return x.decode("utf-8", "replace") if isinstance(x, bytes) else x


def _tail(text: str, lines: int = 60, chars: int = 6000) -> str:
    ls = text.strip().splitlines()
    out = "\n".join(ls[-lines:])
    if len(ls) > lines:
        out = f"[... {len(ls) - lines} earlier lines omitted ...]\n" + out
    return out[-chars:]


class Workspace:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"workspace {self.root} is not a directory")

    def resolve(self, path: str | None) -> Path:
        p = (self.root / (str(path) if path else ".")).resolve()
        if p != self.root and self.root not in p.parents:
            raise ToolDenied(f"path {path!r} is outside the workspace")
        return p

    def rel(self, p: Path) -> str:
        r = str(p.relative_to(self.root))
        return r if r != "." else "."

    def tree(self, path: str = ".", depth: int = 2, limit: int = 300) -> list[str]:
        base = self.resolve(path)
        out: list[str] = []

        def walk(d: Path, level: int):
            try:
                entries = sorted(d.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
            except OSError:
                return
            for e in entries:
                if len(out) >= limit:
                    return
                try:
                    self.resolve(str(e))
                except (ToolDenied, OSError, RuntimeError):
                    continue
                if e.name in SKIP_DIRS or (e.name.startswith(".") and e.name not in (".github", ".env.example")):
                    continue
                out.append("  " * level + e.name + ("/" if e.is_dir() else ""))
                if e.is_dir() and not e.is_symlink() and level + 1 < depth:
                    walk(e, level + 1)
        walk(base, 0)
        if len(out) >= limit:
            out.append(f"[stopped at {limit} entries; list a subfolder]")
        return out

    def fingerprint(self, limit: int = 20000) -> dict[str, tuple[int, int]]:
        """rel path -> (size, mtime_ns) for every file the tools can see. Used to notice files a
        shell command changed, so the verify gate and the episode do not depend on the model
        reporting its own edits."""
        out: dict[str, tuple[int, int]] = {}
        for p in self.iter_files():
            try:
                st = p.stat()
            except OSError:
                continue
            out[self.rel(p)] = (st.st_size, st.st_mtime_ns)
            if len(out) >= limit:
                break
        return out

    def iter_files(self, path: str = ".", glob: str | None = None):
        base = self.resolve(path)
        if base.is_file():
            yield base
            return
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")
                                 and not (Path(dirpath) / d).is_symlink())
            for fn in sorted(filenames):
                p = Path(dirpath) / fn
                try:
                    self.resolve(str(p))
                except (ToolDenied, OSError, RuntimeError):
                    continue
                if glob and not (fnmatch.fnmatch(fn, glob) or fnmatch.fnmatch(self.rel(p), glob)):
                    continue
                yield p


class Verifier:
    """Runs the configured verify command. Both the run_tests tool and the loop's stop hook go
    through here, so every verification lands in RunState and fires `on_result` (the bench uses
    that hook to snapshot the candidate at each attempt).

    `check`, when given, replaces the shell command with a trusted callable -> (passed, tail).
    The benchmark uses it so a model that edits check_visible.py cannot fake a pass."""

    def __init__(self, ws: Workspace, cmd: str | None, timeout: int = 300,
                 on_result: Callable[[VerifyResult], None] | None = None,
                 check: Callable[[], tuple[bool, str]] | None = None):
        self.ws, self.cmd, self.timeout, self.on_result, self.check = ws, cmd, timeout, on_result, check

    def run(self, state: RunState, source: str = "tool") -> VerifyResult | None:
        if not self.cmd and self.check is None:
            return None
        t0 = time.perf_counter()
        res = self._check(source, t0) if self.check is not None else self._shell(source, t0)
        state.verify_results.append(res)
        state.edits_since_verify = 0
        if self.on_result:
            self.on_result(res)
        return res

    def _check(self, source: str, t0: float) -> VerifyResult:
        try:
            passed, tail = self.check()
            return VerifyResult(bool(passed), _tail(str(tail)), time.perf_counter() - t0, source)
        except Exception as e:  # noqa: BLE001 - a broken checker is a failed check, not a crash
            return VerifyResult(False, f"[checker error: {type(e).__name__}: {e}]", time.perf_counter() - t0, source)

    def _shell(self, source: str, t0: float) -> VerifyResult:
        try:
            proc = subprocess.run(["bash", "-c", self.cmd], cwd=self.ws.root, capture_output=True, text=True,
                                  timeout=self.timeout, stdin=subprocess.DEVNULL, env=clean_env())
            return VerifyResult(proc.returncode == 0, _tail(proc.stdout + "\n" + proc.stderr),
                                time.perf_counter() - t0, source)
        except subprocess.TimeoutExpired:
            return VerifyResult(False, f"[test command timed out after {self.timeout}s]", time.perf_counter() - t0, source)
        except OSError as e:
            return VerifyResult(False, f"[could not run test command: {e}]", time.perf_counter() - t0, source)


def _cap(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    marker = "\n[output truncated; narrow the request]\n"
    budget = max(0, limit - len(marker))
    head = budget * 2 // 3
    tail = budget - head
    return (text[:head] + marker + (text[-tail:] if tail else ""))[:limit]


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, tuple[dict, Callable]] = {}

    def register(self, name: str, description: str, parameters: dict, fn: Callable):
        self._tools[name] = ({"name": name, "description": description, "parameters": parameters}, fn)

    def schemas(self) -> list[dict]:
        return [spec for spec, _ in self._tools.values()]

    def specs(self) -> list[dict]:
        return self.schemas()

    def call(self, name: str, args: dict) -> str:
        return self.execute(name, args)[0]

    def names(self) -> list[str]:
        return list(self._tools)

    def execute(self, name: str, args) -> tuple[str, bool]:
        """Returns (output, ok). Never raises."""
        try:
            if not isinstance(name, str) or name not in self._tools:
                return _cap(f"ERROR: unknown tool {name!r}. Available: {', '.join(self._tools)}"), False
            if not isinstance(args, dict):
                return f"ERROR: arguments must be a JSON object, got {type(args).__name__}", False
            if "_invalid_arguments" in args:
                return _cap(f"ERROR: arguments were not valid JSON: {args['_invalid_arguments']!r}"), False
            _, fn = self._tools[name]
            inspect.signature(fn).bind(**args)
            out = str(fn(**args))
        except ToolDenied as e:
            return _cap(f"ERROR: denied: {e}"), False
        except Exception as e:  # noqa: BLE001 - a tool must never crash the loop
            return _cap(f"ERROR: {type(e).__name__}: {e}"), False
        return _cap(out), not out.startswith("ERROR:")


def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


def build_tools(ws: Workspace, policy: Policy, state: RunState, verifier: Verifier | None = None,
                memory=None, scope: str = "global", command_timeout: int = 120,
                benchmark: bool = False) -> ToolRegistry:
    """The agent's tool set. `benchmark=True` removes run_command: shell commands are not
    path-confined, so in a benchmark they could read the hidden tests from the harness source."""
    reg = ToolRegistry()

    def list_files(path: str = ".", depth: int = 2):
        lines = ws.tree(path, max(1, min(int(depth), 5)))
        return "\n".join(lines) if lines else "(empty)"

    def read_file(path: str, start_line: int = 1, end_line: int | None = None):
        p = ws.resolve(path)
        if not p.is_file():
            return f"ERROR: {path} is not a file"
        if p.stat().st_size > 2_000_000:
            return f"ERROR: {path} is larger than 2 MB; use search instead"
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        if not lines:
            return "(empty file)"
        start = max(1, int(start_line))
        end = min(len(lines), int(end_line) if end_line else start + 399, start + 999)
        if start > len(lines):
            return f"ERROR: start_line {start} is past the end of the file ({len(lines)} lines)"
        body = "\n".join(f"{i:>5}| {lines[i - 1]}" for i in range(start, end + 1))
        more = f"\n[lines {start}-{end} of {len(lines)}; read more with start_line={end + 1}]" if end < len(lines) else ""
        return body + more

    def search(pattern: str, path: str = ".", glob: str | None = None, regex: bool = False, max_results: int = 80):
        try:
            rx = re.compile(pattern if regex else re.escape(pattern), 0 if regex else re.IGNORECASE)
        except re.error as e:
            return f"ERROR: invalid regex: {e}. Set regex=false for a plain-text search."
        hits = []
        max_results = max(1, min(int(max_results), 300))
        for f in ws.iter_files(path, glob):
            try:
                if f.stat().st_size > 1_000_000:
                    continue
                for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{ws.rel(f)}:{i}: {line.strip()[:200]}")
                        if len(hits) >= max_results:
                            return "\n".join(hits) + f"\n[output truncated at {max_results} matches; narrow the search]"
            except OSError:
                continue
        return "\n".join(hits) if hits else "no matches"

    def write_file(path: str, content: str):
        policy.check_write(f"write {path}")
        p = ws.resolve(path)
        if p == ws.root or p.is_dir():
            return f"ERROR: {path} is a directory"
        p.parent.mkdir(parents=True, exist_ok=True)
        existed = p.exists()
        content = str(content)
        p.write_text(content, encoding="utf-8")
        state.mark_edit(ws.rel(p))
        state.last_syntax_error = _py_syntax_error(p, content)
        return f"{'overwrote' if existed else 'created'} {ws.rel(p)} ({len(content.splitlines())} lines)"

    def edit_file(path: str, old_string: str, new_string: str):
        policy.check_write(f"edit {path}")
        p = ws.resolve(path)
        if not p.is_file():
            return f"ERROR: {path} does not exist; use write_file to create it"
        if not old_string:
            return "ERROR: old_string is empty. Copy the exact text to replace from read_file (without line numbers)."
        text = p.read_text(encoding="utf-8")
        n = text.count(old_string)
        if n == 0:
            hint = ""
            first = old_string.strip().splitlines()[0].strip() if old_string.strip() else ""
            if first and first in text:
                line_no = text[: text.index(first)].count("\n") + 1
                hint = f" The first line does appear near line {line_no}; check indentation and the following lines."
            return ("ERROR: old_string not found. Read the file again and copy the exact text, including whitespace; "
                    "do not include the line-number prefix." + hint)
        if n > 1:
            lines = [text[: m.start()].count("\n") + 1 for m in re.finditer(re.escape(old_string), text)][:10]
            return (f"ERROR: old_string matches {n} places (lines {', '.join(map(str, lines))}). Include more "
                    "surrounding lines to make it unique.")
        new_text = text.replace(old_string, new_string, 1)
        p.write_text(new_text, encoding="utf-8")
        state.mark_edit(ws.rel(p))
        state.last_syntax_error = _py_syntax_error(p, new_text)
        return f"edited {ws.rel(p)} (1 replacement)"

    def run_command(command: str, timeout: int = 60):
        policy.check_command(command)
        timeout = max(1, min(int(timeout), command_timeout))
        t0 = time.perf_counter()
        before = ws.fingerprint()
        try:
            proc = subprocess.run(["bash", "-c", command], cwd=ws.root, capture_output=True, text=True,
                                  timeout=timeout, stdin=subprocess.DEVNULL, env=clean_env())
            out = (proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr.strip() else "")).strip()
            code = proc.returncode
        except subprocess.TimeoutExpired as e:
            out, code = f"{_decode(e.stdout)}\n{_decode(e.stderr)}\n[timed out after {timeout}s]".strip(), 124
        state.commands.append({"command": command[:500], "exit": code, "s": round(time.perf_counter() - t0, 2)})
        # A command that writes files is an edit as far as the verify gate is concerned.
        after = ws.fingerprint()
        for rel in sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k)):
            state.mark_edit(rel)
        return f"exit code {code}\n{_cap(out or '(no output)', 8000)}"

    def run_tests():
        if verifier is None or not (verifier.cmd or verifier.check):
            return "No test command is configured for this workspace. Use run_command to run tests directly."
        res = verifier.run(state, "tool")
        return f"{'PASSED' if res.passed else 'FAILED'}: `{verifier.cmd or 'visible checks'}` ({res.seconds:.1f}s)\n{res.tail}"

    def recall(query: str):
        if memory is None:
            return "Memory is disabled for this run."
        r = memory.retrieve(str(query), scope)
        state.recalled += [f["id"] for f in r.facts]
        return memory.render(r, empty="No matching memories.")

    def note(lesson: str):
        lesson = " ".join(str(lesson).split())[:400]
        if len(lesson) < 10:
            return "ERROR: a lesson must be a full sentence (at least 10 characters)."
        if len(state.notes) >= MAX_NOTES:
            return f"ERROR: at most {MAX_NOTES} notes per run; keep only the most reusable lessons."
        state.notes.append(lesson)
        return "noted; it will be considered for long-term memory after this run"

    def finish(summary: str):
        state.finished = True
        state.finish_summary = str(summary)[:4000]
        return "finish recorded. The harness now verifies the result independently."

    reg.register("list_files", "List files and folders as an indented tree (skips .git, node_modules, caches).",
                 _obj({"path": {"type": "string", "description": "Folder relative to the workspace root"},
                       "depth": {"type": "integer", "description": "1-5, default 2"}}, []), list_files)
    reg.register("read_file", "Read a text file with line numbers. Returns up to 400 lines per call; page with "
                              "start_line/end_line.",
                 _obj({"path": {"type": "string"}, "start_line": {"type": "integer"},
                       "end_line": {"type": "integer"}}, ["path"]), read_file)
    reg.register("search", "Search file contents. Plain text and case-insensitive by default; set regex=true for a "
                           "regular expression. glob filters files, e.g. '*.py' or 'src/*.ts'.",
                 _obj({"pattern": {"type": "string"}, "path": {"type": "string"}, "glob": {"type": "string"},
                       "regex": {"type": "boolean"}, "max_results": {"type": "integer"}}, ["pattern"]), search)
    reg.register("write_file", "Create a file or overwrite it entirely. Prefer edit_file for changes to existing files.",
                 _obj({"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]), write_file)
    reg.register("edit_file", "Replace an exact string in a file. old_string must match exactly once (copy it from "
                              "read_file output without the line-number prefix).",
                 _obj({"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}},
                      ["path", "old_string", "new_string"]), edit_file)
    if not benchmark:
        reg.register("run_command", "Run a shell command (bash) in the workspace root. Output is truncated. "
                                    "Destructive commands are blocked by policy.",
                     _obj({"command": {"type": "string"}, "timeout": {"type": "integer", "description": "seconds"}},
                          ["command"]), run_command)
    reg.register("run_tests", "Run the workspace's configured test command. Reports PASSED/FAILED with the output tail.",
                 _obj({}, []), run_tests)
    reg.register("recall", "Search long-term memory (project conventions, past pitfalls, past runs) with plain keywords.",
                 _obj({"query": {"type": "string"}}, ["query"]), recall)
    reg.register("note", "Record one short, reusable lesson from this task: a convention, a pitfall, or a command that "
                         "works. Not for progress updates.",
                 _obj({"lesson": {"type": "string"}}, ["lesson"]), note)
    reg.register("finish", "Call when the task is complete (or you cannot proceed). Summarize what changed and how it "
                           "was verified. The harness re-runs the tests itself after you finish.",
                 _obj({"summary": {"type": "string"}}, ["summary"]), finish)
    return reg


def _py_syntax_error(path: Path, text: str) -> str:
    if path.suffix != ".py":
        return ""
    try:
        compile(text, str(path.name), "exec")
    except SyntaxError as e:
        return f"{path.name}:{e.lineno}: {e.msg}"
    except ValueError as e:              # e.g. null bytes
        return f"{path.name}: {e}"
    return ""


def args_signature(name: str, args) -> str:
    try:
        return name + ":" + json.dumps(args, sort_keys=True, default=str)[:4000]
    except (TypeError, ValueError):
        return name + ":" + repr(args)[:4000]
