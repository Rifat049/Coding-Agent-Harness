"""`python -m codebenchkit doctor`: can this machine run codebenchkit, and with which models?

Each check prints green (ok), yellow (warn: works, with a caveat) or red (fail: something you
asked for cannot run), plus the command that fixes it. Key VALUES are never printed, only whether
each variable is set. Network calls: local Ollama (GET /v1/models), and, when DEEPSEEK_API_KEY is
set, DeepSeek's balance (GET /user/balance: free, no tokens; skip with --offline). --online adds
GET /models on the main provider, --probe one tiny local Ollama call. --full also runs the
selfchecks (~20 s).
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

OK, WARN, FAIL = "ok", "warn", "fail"
KEYS = ("DEEPSEEK_API_KEY", "CBK_ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DASHSCOPE_API_KEY")


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""


def _python() -> Check:
    v = sys.version_info
    if v >= (3, 11):
        return Check("Python", OK, f"{v.major}.{v.minor}.{v.micro} ({sys.executable})")
    return Check("Python", FAIL, f"{v.major}.{v.minor} is too old (tomllib and the agent need 3.11+)",
                 "install Python 3.11 or newer, e.g. `brew install python@3.12` or python.org")


def _fts5() -> Check:
    try:
        c = sqlite3.connect(":memory:")
        c.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        c.close()
        return Check("SQLite FTS5", OK, f"SQLite {sqlite3.sqlite_version}")
    except sqlite3.Error as e:
        return Check("SQLite FTS5", FAIL, f"not available ({e}); memory search needs it",
                     "use a python.org or Homebrew Python build (they ship FTS5)")


def _data() -> Check:
    from .bugfix.core import PROJECT, ROOT
    from .utility.core import ROOT as UROOT
    bugs = sorted(p.name for p in (ROOT / "bugs").glob("b*") if (p / "bug.toml").is_file())
    utasks = sorted(p.name for p in (UROOT / "tasks").glob("u*") if (p / "task.toml").is_file()) if UROOT.is_dir() else []
    if len(utasks) != 10:
        return Check("Benchmark data", FAIL, f"expected 10 utility-v1 tasks under {UROOT}, found {len(utasks)}",
                     "run from a full checkout of the repository (git clone, not a single-file copy)")
    if PROJECT.is_dir() and len(bugs) == 10:
        return Check("Benchmark data", OK, f"utility-v1 (10 tasks) and bugfix (10 bugs) under {ROOT.parent}")
    return Check("Benchmark data", FAIL, f"expected 10 bugs under {ROOT}, found {len(bugs)}",
                 "run from a full checkout of the repository (git clone, not a single-file copy)")


def _config(repo: Path) -> tuple[Check, dict]:
    from .agent.cli import BUILTIN, load_config
    try:
        cfg, path = load_config(repo)
    except ValueError as e:
        return Check("Config", FAIL, str(e), "fix the TOML file named above (see .cbk/config.toml in the repo)"), {}
    merged = {**BUILTIN, **cfg}
    where = str(path) if path else "none found; using built-in defaults"
    return Check("Config", OK if path else WARN, f"{where}; main={merged['model']}, small={merged['small_model']}",
                 "" if path else "copy .cbk/config.toml from the repo into your project or ~/.cbk/"), merged


def _keys(main: str, offline: bool = False) -> list[Check]:
    """`offline`: no live run is planned, so a missing main-model key is a warning, not a failure."""
    from .agent.llm import DEFAULTS, canonical
    out = []
    d = DEFAULTS.get(canonical(main), {})
    needed = None if d.get("optional_key") else d.get("key_env")   # local Qwen needs no key
    for k in KEYS:
        present = bool(os.environ.get(k))
        if k == needed and not present:
            if offline:
                out.append(Check(k, WARN, f"not set; needed only for live runs with the main model ({main}). "
                                 "Offline checks, mocks and saved results need no key", f"export {k}=...   # in your shell profile"))
            else:
                out.append(Check(k, FAIL, f"not set, and the main model ({main}) needs it", f"export {k}=...   # in your shell profile"))
        else:
            out.append(Check(k, OK if present else WARN, "set" if present else "not set (only needed for that provider)"))
    if os.environ.get("ANTHROPIC_API_KEY"):
        out.append(Check("ANTHROPIC_API_KEY", WARN, "set, but codebenchkit ignores it on purpose (other Anthropic tools in "
                         "this shell would bill it)", "export CBK_ANTHROPIC_API_KEY=... for Claude runs"))
    return out


def _ollama(probe: bool, needed: bool = False) -> list[Check]:
    """`needed`: the main model is local Qwen, so a missing server or model is a failure, not a warning."""
    from .agent.cli import list_models
    from .agent.llm import qwen_settings
    q = qwen_settings()
    level = FAIL if needed else WARN
    if not q["local"]:
        return [Check("Qwen endpoint", OK, f"hosted: {q['base_url']} (model {q['model']})")]
    ids, why = list_models(q["base_url"])
    if why:
        return [Check("Ollama", level, f"not reachable at {q['base_url']} ({why})"
                      + ("; the main model needs it" if needed else "; `--small-model auto` will skip local Qwen"),
                      f"install from https://ollama.com/download, start it (`ollama serve`), then `ollama pull {q['model']}`")]
    out = [Check("Ollama", OK, f"{q['base_url']} answers; models: {', '.join(ids) or 'none'}")]
    if q["model"] not in ids and f"{q['model']}:latest" not in ids:
        out.append(Check("Qwen model", level, f"{q['model']} is not pulled", f"ollama pull {q['model']}"))
        return out
    out.append(Check("Qwen model", OK, q["model"]))
    if probe:
        native = q["base_url"][:-3] if q["base_url"].endswith("/v1") else q["base_url"]
        body = {"model": q["model"], "stream": False, "think": False, "messages": [{"role": "user", "content": "Return {\"a\": 1}"}],
                "format": {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]},
                "options": {"num_predict": 20}}
        try:
            req = urllib.request.Request(f"{native}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                r.read()
            out.append(Check("Qwen structured output", OK, "format=<JSON schema> accepted (constrained decoding)"))
        except urllib.error.HTTPError as e:
            out.append(Check("Qwen structured output", WARN, f"HTTP {e.code}: format=<schema> refused (MLX builds do this); "
                             "consolidation falls back to prompt-only JSON",
                             f"for constrained decoding pull a GGUF tag, e.g. `ollama pull {q['model']}-q4_K_M` "
                             "and set QWEN_MODEL to it"))
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            out.append(Check("Qwen structured output", WARN, f"probe failed: {e}"))
    return out


# DeepSeek reports the balance in the account's currency (CNY for mainland accounts) while prices
# and --max-cost are USD. Fixed approximate rate for the comparison only; it is printed with the result.
CNY_PER_USD = 7.1


def _deepseek_balance(expected_usd: float, timeout: float = 10) -> Check:
    """GET https://api.deepseek.com/user/balance (free). Warn if it is below the next run's expected cost."""
    from .agent.llm import provider_key
    key = provider_key("deepseek")
    if not key:
        return Check("DeepSeek balance", WARN, "DEEPSEEK_API_KEY not set; only needed for DeepSeek runs")
    base = (os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/")
    try:
        req = urllib.request.Request(f"{base}/user/balance", headers={"Authorization": f"Bearer {key}",
                                                                      "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return Check("DeepSeek balance", FAIL, f"HTTP {e.code} from {base}/user/balance",
                     "check DEEPSEEK_API_KEY (401 = wrong key)" if e.code in (401, 403) else "retry later")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        return Check("DeepSeek balance", WARN, f"cannot read the balance ({e})", "check the network")
    infos = data.get("balance_infos") or []
    usd = 0.0
    parts = []
    for b in infos:
        try:
            amount = float(b.get("total_balance") or 0)
        except ValueError:
            continue
        cur = b.get("currency", "?")
        parts.append(f"{amount:.2f} {cur}")
        usd += amount if cur == "USD" else amount / CNY_PER_USD if cur == "CNY" else 0.0
    shown = ", ".join(parts) or "no balance entries"
    approx = f" (about ${usd:.2f} at {CNY_PER_USD} CNY/USD)" if any(b.get("currency") == "CNY" for b in infos) else ""
    if not data.get("is_available", False):
        return Check("DeepSeek balance", FAIL, f"{shown}{approx}: the account cannot make calls (is_available false)",
                     "top up at https://platform.deepseek.com/top_up")
    if usd < expected_usd:
        return Check("DeepSeek balance", WARN, f"{shown}{approx} is below the next run's expected cost ${expected_usd:.2f}",
                     "top up at https://platform.deepseek.com/top_up, or lower --max-cost")
    return Check("DeepSeek balance", OK, f"{shown}{approx}; covers the next run's expected ${expected_usd:.2f}")


def _small(spec: str) -> Check:
    from .agent.cli import resolve_small
    chosen, note = resolve_small(spec)
    if chosen is None:
        explicit = spec != "auto"
        return Check("Small model", FAIL if explicit else WARN, note,
                     "start Ollama with a Qwen model, or export DEEPSEEK_API_KEY / DASHSCOPE_API_KEY")
    return Check("Small model", OK, note)


def _online(main: str) -> Check:
    """GET /models on the main provider: proves the key authenticates. Free; no tokens."""
    from .agent.llm import DEFAULTS, canonical, provider_key, qwen_settings
    kind = canonical(main)
    if kind in ("mock", "qwen") or kind not in DEFAULTS:
        return Check("Main model online", OK, f"{main}: nothing to check online")
    key = provider_key(kind)
    if not key:
        return Check("Main model online", FAIL, f"{DEFAULTS[kind]['key_env']} not set", f"export {DEFAULTS[kind]['key_env']}=...")
    base = os.environ.get({"deepseek": "DEEPSEEK_BASE_URL", "openai": "OPENAI_BASE_URL", "anthropic": "ANTHROPIC_BASE_URL",
                           "dashscope": "DASHSCOPE_BASE_URL"}.get(kind, ""), "") or DEFAULTS[kind]["base_url"]
    url = f"{base.rstrip('/')}/v1/models" if kind == "anthropic" else f"{base.rstrip('/')}/models"
    headers = ({"x-api-key": key, "anthropic-version": "2023-06-01"} if kind == "anthropic" else {"Authorization": f"Bearer {key}"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=15) as r:
            ids = [m.get("id") for m in json.loads(r.read()).get("data", [])]
        model = DEFAULTS[kind]["model"] if ":" not in main else main.partition(":")[2]
        listed = model in ids
        return Check("Main model online", OK if listed else WARN,
                     f"{kind}: key accepted ({time.perf_counter() - t0:.1f}s); {model} {'listed' if listed else 'NOT listed: ' + ', '.join(map(str, ids[:8]))}",
                     "" if listed else f"pass --model {kind}:<one of the listed ids>")
    except urllib.error.HTTPError as e:
        return Check("Main model online", FAIL, f"{kind}: HTTP {e.code} from {url}",
                     "check the key (401/403) or the base URL" if e.code in (401, 403, 404) else "retry later")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return Check("Main model online", FAIL, f"{kind}: cannot reach {url} ({e})", "check the network or proxy")


def _memory_dir() -> Check:
    d = Path(os.environ.get("CBK_HOME") or Path.home() / ".cbk").expanduser()
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe = d / ".doctor-write-test"
        probe.write_text("ok")
        probe.unlink()
        return Check("Memory dir", OK, f"{d} is writable")
    except OSError as e:
        return Check("Memory dir", FAIL, f"{d}: {e}", "export CBK_HOME=/a/writable/folder")


def _selfchecks() -> list[Check]:
    import contextlib
    import io
    from .bugfix.core import selfcheck as bugfix_selfcheck
    from .cli import cmd_selfcheck
    out = []
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cmd_selfcheck(None)
    out.append(Check("selfcheck (10 utility tasks)", OK if code == 0 else FAIL, buf.getvalue().strip().splitlines()[-1],
                     "" if code == 0 else "python -m codebenchkit selfcheck   # shows which suite fails"))
    lines = []
    t0 = time.perf_counter()
    code = bugfix_selfcheck(log=lines.append)
    from .utility.core import selfcheck as utility_selfcheck
    ulines = []
    code_u = utility_selfcheck(log=ulines.append)
    out.append(Check("utility-selfcheck (utility-v1)", OK if code_u == 0 else FAIL, ulines[-1],
                     "" if code_u == 0 else "python -m codebenchkit utility-selfcheck"))
    out.append(Check("bugfix-selfcheck (10 bugs)", OK if code == 0 else FAIL, f"{lines[-1]} in {time.perf_counter() - t0:.0f}s",
                     "" if code == 0 else "python -m codebenchkit bugfix-selfcheck   # b07's timing thresholds are machine-dependent"))
    return out


def run(repo: Path, online: bool = False, probe: bool = False, full: bool = False, color: bool | None = None,
        expected_cost: float = 0.50, offline: bool = False, main: str | None = None) -> list[Check]:
    checks = [_python(), _fts5(), _data()]
    cfg_check, cfg = _config(repo)
    checks.append(cfg_check)
    main, small = main or cfg.get("model", "deepseek"), cfg.get("small_model", "auto")
    from .agent.llm import canonical
    checks += _keys(main, offline)
    checks += _ollama(probe, needed=canonical(main) == "qwen")
    if not offline:
        checks.append(_deepseek_balance(expected_cost))
    checks.append(_small(small))
    checks.append(_memory_dir())
    if online:
        checks.append(_online(main))
    if full:
        checks += _selfchecks()
    return checks


def render(checks: list[Check], color: bool) -> str:
    paint = {OK: ("\033[32m", "ok  "), WARN: ("\033[33m", "warn"), FAIL: ("\033[31m", "FAIL")}
    lines = []
    for c in checks:
        col, label = paint[c.status]
        tag = f"{col}{label}\033[0m" if color else label
        lines.append(f"[{tag}] {c.name:<28} {c.detail}")
        if c.fix:
            lines.append(f"       {'':<28} fix: {c.fix}")
    n_fail = sum(c.status == FAIL for c in checks)
    n_warn = sum(c.status == WARN for c in checks)
    lines.append(f"\n{len(checks)} checks: {n_fail} failing, {n_warn} warnings." +
                 ("" if n_fail else " Ready to run."))
    return "\n".join(lines)


def cmd_doctor(args) -> int:
    checks = run(Path(args.repo), online=args.online, probe=args.probe, full=args.full,
                 expected_cost=args.expected_cost, offline=args.offline, main=args.model)
    print(render(checks, color=sys.stdout.isatty() and not os.environ.get("NO_COLOR")))
    return 1 if any(c.status == FAIL for c in checks) else 0
