"""Independent checks that run *before* any model code executes.

The pipeline for one model response is:
    extract_code -> static_check -> (sandbox) visible tests -> (sandbox) hidden tests

`extract_code` and `static_check` classify an "invalid output": a response from which no
executable candidate could be produced. That rate is a first-class metric.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass

FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.DOTALL)

# Modules and names a pure utility function has no business touching. This is a
# guardrail plus a metric, NOT a security boundary (see sandbox.py).
BANNED_MODULES = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "ctypes", "multiprocessing",
    "threading", "importlib", "builtins", "signal", "pickle", "marshal", "urllib", "http",
    "requests", "asyncio", "tempfile", "glob", "io", "gc", "inspect",
}
BANNED_CALLS = {"eval", "exec", "compile", "open", "__import__", "input", "globals", "locals",
                "vars", "breakpoint", "exit", "quit", "getattr", "setattr", "delattr"}
BANNED_ATTRS = {"__subclasses__", "__globals__", "__builtins__", "__code__", "__bases__", "__mro__"}

INVALID_CATEGORIES = ("no_code", "syntax_error", "missing_function", "banned")


@dataclass
class Extraction:
    code: str | None
    how: str  # "fenced", "unfenced", "none"


@dataclass
class StaticResult:
    category: str  # "ok" or one of INVALID_CATEGORIES
    detail: str = ""

    @property
    def valid(self) -> bool:
        return self.category == "ok"


def _defines(code: str, fn_name: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(n, ast.FunctionDef) and n.name == fn_name for n in tree.body)


def extract_code(text: str, fn_name: str) -> Extraction:
    """Pick the code the model meant as its answer."""
    blocks = [(lang.lower(), body) for lang, body in FENCE_RE.findall(text or "")]
    if blocks:
        py = [b for lang, b in blocks if lang in ("", "python", "py", "python3")] or [b for _, b in blocks]
        defining = [b for b in py if _defines(b, fn_name)]
        return Extraction((defining or py)[-1].strip("\n"), "fenced")
    # Unfenced: accept the whole response only if it is itself valid Python defining the function.
    if text and _defines(text, fn_name):
        return Extraction(text.strip("\n"), "unfenced")
    return Extraction(None, "none")


def static_check(code: str | None, fn_name: str) -> StaticResult:
    if not code or not code.strip():
        return StaticResult("no_code", "response contained no Python code block")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return StaticResult("syntax_error", f"line {e.lineno}: {e.msg}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in BANNED_MODULES:
                    return StaticResult("banned", f"import of '{a.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in BANNED_MODULES:
                return StaticResult("banned", f"import from '{node.module}' is not allowed")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
            return StaticResult("banned", f"call to '{node.func.id}()' is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRS:
            return StaticResult("banned", f"access to '{node.attr}' is not allowed")
        elif isinstance(node, ast.Name) and node.id in ("__builtins__",):
            return StaticResult("banned", "access to '__builtins__' is not allowed")

    fns = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == fn_name]
    if not fns:
        return StaticResult("missing_function", f"no top-level function named '{fn_name}'")
    if isinstance(fns[-1], ast.AsyncFunctionDef):
        return StaticResult("missing_function", f"'{fn_name}' must be a regular (non-async) function")
    return StaticResult("ok")
