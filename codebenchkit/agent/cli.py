"""CLI wiring for the agent, kept outside the v0.1 harness so its commands stay unchanged.

    python -m codebenchkit agent run "task" --repo PATH --verify "pytest -q"
        (models come from .cbk/config.toml: DeepSeek main, Qwen small; --model/--small-model override)
    python -m codebenchkit agent chat --repo PATH
    python -m codebenchkit agent memory list|search|add|forget|consolidate|export   (also: memory ...)
    python -m codebenchkit agent-bench --model mock --memory off|on --samples N
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .llm import ProviderError, canonical, make_provider, missing_key, qwen_settings
from .memory import FACT_KINDS, Memory

try:
    import tomllib
except ModuleNotFoundError:          # Python < 3.11: config files are skipped with a note
    tomllib = None

# Built-in defaults, used when neither a flag nor .cbk/config.toml sets a value.
BUILTIN = {"model": "deepseek", "small_model": "auto", "verify": None, "approval": "auto",
           "gate": "heuristic", "consolidation_every": 3}
CONFIG_KEYS = {("models", "main"): "model", ("models", "small"): "small_model",
               ("agent", "verify"): "verify", ("agent", "approval"): "approval",
               ("agent", "gate"): "gate", ("agent", "consolidation_every"): "consolidation_every"}
CHOICES = {"approval": ("readonly", "auto", "ask"), "gate": ("always", "heuristic", "llm", "never")}


def config_paths(repo: Path | None) -> list[Path]:
    """Lookup order, first existing file wins: <repo>/.cbk/config.toml, ./.cbk/config.toml,
    $CBK_HOME/config.toml (default ~/.cbk/config.toml)."""
    home = Path(os.environ.get("CBK_HOME") or Path.home() / ".cbk").expanduser()
    out: list[Path] = []
    for c in ([Path(repo).expanduser() / ".cbk" / "config.toml"] if repo else []) + \
             [Path.cwd() / ".cbk" / "config.toml", home / "config.toml"]:
        if c.resolve() not in [o.resolve() for o in out]:
            out.append(c)
    return out


def load_config(repo: Path | None) -> tuple[dict, Path | None]:
    """{arg name: value} from the first config file found, validated. Raises ValueError on a bad file."""
    for path in config_paths(repo):
        if not path.is_file():
            continue
        if tomllib is None:
            print(f"note: Python 3.11+ is needed to read {path}; using built-in defaults", file=sys.stderr)
            return {}, None
        try:
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"{path}: {e}") from None
        out = {}
        for (section, key), dest in CONFIG_KEYS.items():
            value = (raw.get(section) or {}).get(key)
            if value is None:
                continue
            if dest == "consolidation_every":
                if type(value) is not int or value <= 0:
                    raise ValueError(f"{path}: [{section}] {key} must be a positive integer")
            elif not isinstance(value, str) or not value.strip():
                raise ValueError(f"{path}: [{section}] {key} must be a non-empty string")
            elif dest in CHOICES and value not in CHOICES[dest]:
                raise ValueError(f"{path}: [{section}] {key} must be one of {', '.join(CHOICES[dest])}")
            out[dest] = value
        return out, path
    return {}, None


def resolve_settings(args, use_config: bool = True) -> None:
    """Fill every unset (None) option: flag > .cbk/config.toml > BUILTIN. A mock main model with
    no explicit small model gets the mock small model, so offline demos never make real calls."""
    if getattr(args, "model", None) and args.model.startswith("mock") and getattr(args, "small_model", None) is None:
        args.small_model = "mock"
    cfg, path = load_config(getattr(args, "repo", None)) if use_config else ({}, None)
    args.config_path = str(path) if path else None
    for dest, default in BUILTIN.items():
        if hasattr(args, dest) and getattr(args, dest) is None:
            setattr(args, dest, cfg.get(dest, default))


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def _memory_dir(args) -> Path:
    return Path(args.memory_dir or os.environ.get("CBK_HOME") or Path.home() / ".cbk").expanduser()


def _open_memory(args) -> Memory:
    d = _memory_dir(args)
    return Memory(d / "memory.sqlite3", skills_dir=d / "skills")


def _model_args(parser, bench: bool = False) -> None:
    parser.add_argument("--model", default="mock" if bench else None,
                        help="deepseek (default) | qwen | claude | chatgpt | <provider>:<id> | compat:<url>#<id> | mock")
    parser.add_argument("--small-model", default=None,
                        help="cheap model for the retrieval gate, consolidation and compaction summaries: "
                             "auto (default: local qwen2.5-coder -> DeepSeek -> DashScope -> none), a spec, or none")


def _limit_args(parser, defaults: dict | None = None) -> None:
    d = {"max_iterations": 30, "max_new_tokens": 16000, "token_budget": 600_000, "wall_seconds": 900,
         "max_verify_rounds": 3, "context_limit": 160_000, **(defaults or {})}
    g = parser.add_argument_group("guardrails")
    g.add_argument("--max-iterations", type=positive, default=d["max_iterations"])
    g.add_argument("--max-new-tokens", type=positive, default=d["max_new_tokens"], help="output cap per model call")
    g.add_argument("--token-budget", type=positive, default=d["token_budget"], help="input + output tokens per run")
    g.add_argument("--wall-seconds", type=positive_float, default=d["wall_seconds"])
    g.add_argument("--max-verify-rounds", type=positive, default=d["max_verify_rounds"])
    g.add_argument("--context-limit", type=positive, default=d["context_limit"], help="tokens; compaction at 70%%")
    g.add_argument("--command-timeout", type=positive, default=120)
    g.add_argument("--gate", choices=list(CHOICES["gate"]), default=None, help="default heuristic")
    g.add_argument("--consolidation-every", type=positive, default=None, help="default 3")
    g.add_argument("--trim-chars", type=positive, default=4000, help="seen tool results longer than this are trimmed")
    g.add_argument("--trim-keep", type=positive, default=2000, help="characters kept from a trimmed result")
    g.add_argument("--force-compaction-after", type=int, default=0, metavar="N",
                   help="test knob: compact history once after N model calls (0 = off)")


def _run_args(parser) -> None:
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    _model_args(parser)
    parser.add_argument("--verify", default=None, help="test command the harness runs itself, e.g. 'pytest -q'")
    parser.add_argument("--approval", choices=list(CHOICES["approval"]), default=None, help="default auto")
    parser.add_argument("--memory", choices=["on", "off"], default="on")
    parser.add_argument("--memory-dir", help="default ~/.cbk (or $CBK_HOME)")
    parser.add_argument("--trace-dir", type=Path)
    _limit_args(parser)


def _config(args):
    from .loop import AgentConfig
    return AgentConfig(max_iterations=args.max_iterations, max_tokens=args.max_new_tokens,
                       token_budget=args.token_budget, wall_seconds=args.wall_seconds,
                       max_verify_rounds=args.max_verify_rounds, context_limit=args.context_limit,
                       command_timeout=args.command_timeout, verify_cmd=getattr(args, "verify", None),
                       approval=getattr(args, "approval", "auto"), gate_mode=args.gate,
                       consolidation_every=args.consolidation_every, trim_chars=args.trim_chars,
                       trim_keep=min(args.trim_keep, args.trim_chars),
                       force_compaction_after=max(0, args.force_compaction_after),
                       small_model_note=getattr(args, "small_model_note", "") or "")


DISABLED = "small model disabled (heuristic gate, no consolidation, digest compaction)"


def small_unavailable(spec: str) -> str | None:
    """Why the small model cannot be used, or None. Checks keys, and for local Qwen that the
    server answers and has the model, so a missing Ollama costs one fast check, not retries."""
    env = missing_key(spec)
    if env:
        return f"{env} is not set"
    if canonical(spec) == "qwen":
        q = qwen_settings()
        model = spec.partition(":")[2] or q["model"]
        ids, why = list_models(q["base_url"])
        if why:
            return (f"no Qwen server at {q['base_url']} ({why}); install Ollama, run `ollama serve`, "
                    f"then `ollama pull {model}`" if q["local"] else f"Qwen endpoint {q['base_url']}: {why}")
        if q["local"] and ids is not None and model not in ids and f"{model}:latest" not in ids:
            return f"Ollama has no model {model!r} (installed: {', '.join(ids) or 'none'}); run `ollama pull {model}`"
    return None


def list_models(base_url: str, timeout: float = 2.0) -> tuple[list[str] | None, str | None]:
    """(model ids or None, error or None) from GET <base>/models."""
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=timeout) as resp:
            data = json.loads(resp.read(1 << 20) or b"{}")
        return [m.get("id", "") for m in data.get("data") or [] if isinstance(m, dict)], None
    except urllib.error.HTTPError as e:
        return (None, None) if e.code in (401, 403, 404) else (None, f"HTTP {e.code}")
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError) as e:
        return None, f"{type(e).__name__}: {getattr(e, 'reason', e)}"


# `--small-model auto`: the first one that is usable right now. Local first (free), then paid.
AUTO_ORDER = (("qwen", "local Qwen via Ollama"), ("deepseek", "DeepSeek"), ("dashscope", "hosted Qwen on DashScope"))


def resolve_small(spec: str) -> tuple[str | None, str]:
    """(spec to use or None, human-readable note). 'auto' walks AUTO_ORDER; any other spec is
    checked as is. Never raises; the note says why a candidate was skipped."""
    if spec.startswith("mock"):
        return spec, "mock"
    if spec in ("none", "off", ""):
        return None, "none (explicit): " + DISABLED
    if spec != "auto":
        why = small_unavailable(spec)
        return (None, f"{spec} unavailable: {why}") if why else (spec, f"{spec} (explicit)")
    skipped = []
    for candidate, label in AUTO_ORDER:
        why = small_unavailable(candidate)
        if not why:
            note = f"auto -> {candidate} ({label})" + (f"; skipped {'; '.join(skipped)}" if skipped else "")
            return candidate, note
        skipped.append(f"{candidate}: {why}")
    return None, "auto -> none; " + "; ".join(skipped)


def _small(args):
    """The cheap model, or None with a warning when it is unavailable (the run still works:
    the gate falls back to the heuristic, consolidation waits, compaction uses a digest)."""
    spec, note = resolve_small(args.small_model)
    args.small_model_note = note
    if spec is None:
        if args.small_model not in ("none", "off"):
            print(f"WARNING: {note}; {DISABLED}", file=sys.stderr)
        return None
    if args.small_model == "auto":
        print(f"small model: {note}", file=sys.stderr)
    return make_provider(spec, role="small")


def _check_main_key(spec: str) -> None:
    env = missing_key(spec)
    if env:
        extra = " (ANTHROPIC_API_KEY is ignored on purpose)" if env == "CBK_ANTHROPIC_API_KEY" else ""
        raise ProviderError(f"{env} is not set (needed for --model {spec}){extra}")


def _approve(tool: str, command: str) -> bool:
    try:
        return input(f"\n[approve {tool}] {command}\n  run it? [y/N] ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def cmd_agent(args) -> int:
    from .chat import ChatSession, repl
    from .runner import run_agent
    memory = None
    try:
        resolve_settings(args)
        print(f"models: main {args.model}, small {args.small_model}"
              f" ({'from ' + args.config_path if args.config_path else 'built-in defaults'}; flags override)",
              file=sys.stderr)
        _check_main_key(args.model)
        provider, small = make_provider(args.model), _small(args)
        if args.memory == "on":
            memory = _open_memory(args)
        config = _config(args)
        trace_dir = args.trace_dir or _memory_dir(args) / "traces"
        if args.agent_command == "chat":
            return repl(ChatSession(args.repo, provider, config, memory, small, trace_dir, _approve))
        res = run_agent(args.task, args.repo, provider, config, memory=memory, small=small, approve=_approve,
                        trace_path=trace_dir / f"run-{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 10**6}.jsonl")
        print(res.reply)
        print(json.dumps(res.receipt, indent=2))
        print(f"Trace: {res.trace_path}")
        if res.consolidation and (res.consolidation["episode_ids"] or res.consolidation["errors"]):
            c = res.consolidation
            print(f"Consolidation ({c.get('model')}, {c.get('latency_s', 0):.1f}s, {c.get('attempts', 0)} attempt(s)): "
                  f"applied {c['applied']}, rejected {c['rejected']}, errors {c['errors']}")
        return 0 if res.outcome in ("success", "partial") else 1
    except (ProviderError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    finally:
        if memory is not None:
            memory.close()


def cmd_memory(args) -> int:
    memory = _open_memory(args)
    try:
        command = args.memory_command
        if command == "list":
            rows = memory.episodes(newest_first=True, limit=args.limit) if args.episodes else \
                memory.facts(scope=args.scope, limit=args.limit)
            for r in rows:
                if args.episodes:
                    print(f"[E{r['id']}] {r['outcome']:<8} {r['scope']:<20} {r['task'][:90]!r}"
                          f"{'' if r['consolidated'] else '  (unconsolidated)'}")
                else:
                    print(f"[F{r['id']}] {r['kind']:<12} {r['scope']:<20} support={r['support']} {r['content']}")
            pending = memory.pending_promotions()
            if pending and not args.episodes:
                print("Pending promotions (apply with `memory promote <fact id>`):")
                for p in pending:
                    print(f"  F{p['fact_id']}: {p['from_scope']} -> {p['to_scope']}  repos {', '.join(json.loads(p['repos']))}"
                          f"  evidence E{', E'.join(map(str, json.loads(p['evidence'])))}  {p['content'][:80]!r}")
            print(json.dumps(memory.stats()))
        elif command == "promote":
            done = memory.promote_fact(args.id)
            if done is None:
                print(json.dumps({"promoted": False, "id": args.id, "reason": "no pending promotion for this fact"}))
                return 1
            print(json.dumps({"promoted": True, "id": args.id, "from": done["from_scope"], "to": done["to_scope"],
                              "repos": json.loads(done["repos"])}))
        elif command == "search":
            print(memory.render(memory.retrieve(args.query, args.scope), empty="No matching memories."))
        elif command == "add":
            fid = memory.add_fact(args.subject, args.content, kind=args.kind, scope=args.scope)
            print(json.dumps({"id": fid}))
        elif command == "forget":
            found = memory.forget_fact(args.id)
            print(json.dumps({"forgotten": found, "id": args.id}))
            return 0 if found else 1
        elif command == "consolidate":
            from .consolidate import consolidate
            resolve_settings(args)
            spec, note = resolve_small(args.small_model)
            if spec is None:
                raise ProviderError(note)
            result = consolidate(memory, make_provider(spec, role="small"), force=True)
            print(json.dumps({k: result[k] for k in ("model", "json_mode", "latency_s", "attempts", "applied", "rejected",
                                                     "errors", "episode_ids", "applied_details", "rejected_details",
                                                     "error_details", "invalid_replies", "replies", "skills")},
                             indent=2, default=str))
            return 1 if result.get("errors") else 0
        elif command == "export":
            markdown = memory.export_markdown()
            if args.out:
                Path(args.out).expanduser().write_text(markdown, encoding="utf-8")
                print(args.out)
            else:
                print(markdown, end="")
        return 0
    except ProviderError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    finally:
        memory.close()


def _register_memory(parser) -> None:
    commands = parser.add_subparsers(dest="memory_command", required=True)
    for name in ("list", "search", "add", "forget", "promote", "consolidate", "export"):
        child = commands.add_parser(name)
        child.add_argument("--memory-dir", help="default ~/.cbk (or $CBK_HOME)")
        child.set_defaults(fn=cmd_memory)
        if name == "list":
            child.add_argument("--scope")
            child.add_argument("--episodes", action="store_true", help="list episodes instead of facts")
            child.add_argument("--limit", type=positive, default=100)
        if name in ("search", "add"):
            child.add_argument("--scope", default="global", help="global or repo:<name>")
        if name == "search":
            child.add_argument("query")
        elif name == "add":
            child.add_argument("content")
            child.add_argument("--subject", default="project")
            child.add_argument("--kind", choices=FACT_KINDS, default="convention")
        elif name in ("forget", "promote"):
            child.add_argument("id", type=positive, help="fact id")
        elif name == "consolidate":
            child.add_argument("--small-model", default=None, help="default: .cbk/config.toml, else qwen")
        elif name == "export":
            child.add_argument("--out", type=Path)


def cmd_agent_bench(args) -> int:
    from ..metrics import render_report, summarize
    from ..tasks import get_tasks
    from .bench import run_agent_benchmark
    name = args.name or f"{time.strftime('%Y%m%d-%H%M%S')}_{args.model.replace(':', '-').replace('/', '-')}_mem_{args.memory}"
    out = Path(args.out) / name
    try:
        resolve_settings(args, use_config=False)       # benchmarks are reproducible from their flags
        _check_main_key(args.model)
        small_spec, note = resolve_small(args.small_model)
        if small_spec is None and args.small_model not in ("auto", "none", "off"):
            raise ProviderError(note)
        if args.small_model == "auto":
            print(f"small model: {note}", file=sys.stderr)
        print(f"Running {args.samples} sample(s) x {len(get_tasks(args.tasks))} tasks, memory {args.memory} -> {out}")
        run_config, items = run_agent_benchmark(
            get_tasks(args.tasks), model=args.model, small_model=small_spec, memory_enabled=args.memory == "on",
            samples=args.samples, config=_config(args), out_dir=out, memory_dir=args.memory_dir, seed=args.seed,
            consolidation_every=args.consolidation_every)
        attempts = [json.loads(line) for line in (out / "attempts.jsonl").read_text().splitlines() if line.strip()]
        summary = summarize(items, attempts)
        report = render_report(run_config, summary)
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (out / "report.md").write_text(report, encoding="utf-8")
        print()
        print(report)
        print(f"Saved: {out}")
        return 0
    except (ProviderError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


def cmd_bugfix_selfcheck(args) -> int:
    from ..bugfix.core import load_bugs, selfcheck
    return selfcheck(load_bugs(args.bugs))


def cmd_bugfix_bench(args) -> int:
    from ..bugfix.bench import reliability, render_reliability, run_bugfix_benchmark
    from ..bugfix.core import load_bugs
    from ..metrics import render_report, summarize
    name = args.name or f"{time.strftime('%Y%m%d-%H%M%S')}_bugfix_{args.model.replace(':', '-').replace('/', '-')}_mem_{args.memory}"
    out = Path(args.out) / name
    try:
        resolve_settings(args, use_config=False)       # reproducible from its flags
        bugs = load_bugs(args.bugs)
        small = None
        if args.model.startswith("mock"):
            small = "mock"
        else:
            _check_main_key(args.model)
            small, note = resolve_small(args.small_model)
            if small is None and args.small_model not in ("auto", "none", "off"):
                raise ProviderError(note)            # an explicit choice that is unusable is an error
            print(f"small model: {note}", file=sys.stderr)
        print(f"bugfix-bench: {len(bugs)} bugs x {args.samples} sample(s), model {args.model}, memory {args.memory} -> {out}")
        run_config, items = run_bugfix_benchmark(
            bugs, model=args.model, small_model=small, memory_enabled=args.memory == "on", samples=args.samples,
            config=_config(args), out_dir=out, memory_dir=args.memory_dir, seed=args.seed,
            consolidation_every=args.consolidation_every, rounds=args.memory_rounds, max_cost=args.max_cost)
        attempts = [json.loads(line) for line in (out / "attempts.jsonl").read_text().splitlines() if line.strip()]
        summary = summarize(items, attempts)
        rel = reliability(items)
        summary["reliability"] = rel
        report = render_report(run_config, summary) + render_reliability(run_config, rel, items)
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (out / "report.md").write_text(report, encoding="utf-8")
        print()
        print(report)
        print(f"Saved: {out}")
        return 0
    except (ProviderError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


def cmd_utility_selfcheck(args) -> int:
    from ..utility.core import load_tasks, selfcheck
    return selfcheck(load_tasks(args.tasks))


def cmd_utility_bench(args) -> int:
    from ..metrics import render_report, summarize
    from ..utility.bench import render_utility, resume_utility_benchmark, run_utility_benchmark, utility_metrics
    from ..utility.core import load_tasks
    name = args.name or (f"{time.strftime('%Y%m%d-%H%M%S')}_utility_{args.model.replace(':', '-').replace('/', '-')}"
                         f"_{args.loop}_{args.prompt_variant}")
    out = Path(args.resume) if args.resume else Path(args.out) / name
    try:
        resolve_settings(args, use_config=False)       # reproducible from its flags
        tasks = load_tasks(args.tasks)
        small = None
        if not args.model.startswith("mock"):
            _check_main_key(args.model)
            _check_local_model(args.model)
            small, note = resolve_small(args.small_model)
            if small is None and args.small_model not in ("auto", "none", "off"):
                raise ProviderError(note)
            print(f"small model: {note}", file=sys.stderr)
        print(f"utility-bench: {len(tasks)} tasks x {args.samples} sample(s), model {args.model}, loop {args.loop}, "
              f"prompt {args.prompt_variant} -> {out}")
        if args.resume:
            print(f"resuming the not_run items of {out} (max cost {args.max_cost})")
            run_config, items = resume_utility_benchmark(out, args.model, _config(args), max_cost=args.max_cost)
        else:
            run_config, items = run_utility_benchmark(
                tasks, model=args.model, small_model=small, samples=args.samples, config=_config(args), out_dir=out,
                seed=args.seed, prompt_variant=args.prompt_variant, max_cost=args.max_cost, loop=args.loop)
        attempts = [json.loads(line) for line in (out / "attempts.jsonl").read_text().splitlines() if line.strip()]
        scored = [i for i in items if not i.get("not_run")]
        keys = {(i["task"], i["sample"]) for i in scored}
        m = utility_metrics(items)
        summary = {"utility": m}
        report = render_utility(run_config, m)
        if scored:
            summary.update(summarize(scored, [a for a in attempts if (a["task"], a["sample"]) in keys]))
            # The v0.1 attempt schema (pass@1 = the first candidate the harness tested, repair gain) is kept
            # for `compare`, in its own file: next to the headline it read as a contradiction.
            (out / "attempts.md").write_text(render_report(run_config, summary).replace(
                "# CodeBenchKit run", "# Legacy attempt metrics (v0.1 schema; pass@1 = first candidate the harness "
                "tested, not the headline pass rate)", 1), encoding="utf-8")
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (out / "report.md").write_text(report, encoding="utf-8")
        print()
        print(report)
        print(f"Saved: {out} (report.md; legacy attempt metrics in attempts.md)")
        return 3 if run_config.get("not_run") and "balance" in str(run_config.get("stopped", "")).lower() else 0
    except (ProviderError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


def _check_local_model(spec: str) -> None:
    """Local Qwen as the main model: fail fast, with the fix command, if Ollama or the model is missing."""
    if canonical(spec) == "qwen" and qwen_settings()["local"]:
        why = small_unavailable(spec)
        if why:
            raise ProviderError(why)


def register_commands(subparsers) -> None:
    agent = subparsers.add_parser("agent", help="tool-using coding agent with harness-verified outcomes")
    commands = agent.add_subparsers(dest="agent_command", required=True)
    run = commands.add_parser("run", help="run one task")
    run.add_argument("task")
    _run_args(run)
    run.set_defaults(fn=cmd_agent)
    chat = commands.add_parser("chat", help="multi-turn REPL")
    _run_args(chat)
    chat.set_defaults(fn=cmd_agent)
    _register_memory(commands.add_parser("memory", help="inspect and curate long-term memory"))
    _register_memory(subparsers.add_parser("memory", help="alias for `agent memory`"))

    bench = subparsers.add_parser("agent-bench", help="benchmark the agent; memory off vs on")
    _model_args(bench, bench=True)
    bench.add_argument("--memory", choices=["on", "off"], default="off")
    bench.add_argument("--memory-dir", help="default: a fresh <run>/memory (reproducible)")
    bench.add_argument("--tasks", help="comma-separated subset")
    bench.add_argument("--samples", type=positive, default=1)
    bench.add_argument("--seed", type=int, default=0)
    bench.add_argument("--out", type=Path, default=Path("runs"))
    bench.add_argument("--name")
    _limit_args(bench)
    bench.set_defaults(fn=cmd_agent_bench)

    from ..doctor import cmd_doctor
    doc = subparsers.add_parser("doctor", help="check this machine: Python, keys (never printed), Ollama, models")
    doc.add_argument("--repo", type=Path, default=Path.cwd(), help="where to look for .cbk/config.toml")
    doc.add_argument("--online", action="store_true", help="GET /models on the main provider (free, no tokens)")
    doc.add_argument("--probe", action="store_true", help="one tiny local Ollama call to test structured output")
    doc.add_argument("--full", action="store_true", help="also run selfcheck, utility-selfcheck and bugfix-selfcheck (~20 s)")
    doc.add_argument("--model", help="check for this main model instead of the config's (e.g. qwen-coder)")
    doc.add_argument("--expected-cost", type=positive_float, default=0.50,
                     help="USD the next paid run may cost; warn if the DeepSeek balance is below it (default 0.50)")
    doc.add_argument("--offline", action="store_true", help="no remote calls (skips the DeepSeek balance)")
    doc.set_defaults(fn=cmd_doctor)

    from ..utility.bench import DEFAULT_BUDGET
    from ..utility.core import PROMPT_VARIANTS
    usc = subparsers.add_parser("utility-selfcheck", help="prove every utility-v1 task is well formed and fully specified")
    usc.add_argument("--tasks", help="comma-separated ids or prefixes (u01,u07)")
    usc.set_defaults(fn=cmd_utility_selfcheck)
    ub = subparsers.add_parser("utility-bench", help="utility-v1: ten bounded utility tasks, scored on untouched tests")
    _model_args(ub, bench=True)
    ub.set_defaults(small_model="none")
    ub.add_argument("--tasks", help="comma-separated ids or prefixes (u01,u07)")
    ub.add_argument("--samples", type=positive, default=1)
    ub.add_argument("--prompt-variant", choices=list(PROMPT_VARIANTS), default="baseline",
                    help="baseline, or tests-first (adds: run the tests before finishing)")
    ub.add_argument("--loop", choices=["agent", "simple"], default="agent",
                    help="agent: the tool-using loop (default). simple: the model writes the whole file, the harness "
                         "runs the visible tests and feeds back the output, up to --max-verify-rounds attempts "
                         "(1 = one-shot)")
    ub.add_argument("--max-cost", type=positive_float, help="USD: stop between items once list-price spend reaches it")
    ub.add_argument("--resume", type=Path, metavar="RUN_DIR",
                    help="run only the not_run items of RUN_DIR (same model and budget flags; its own --max-cost)")
    ub.add_argument("--seed", type=int, default=0)
    ub.add_argument("--out", type=Path, default=Path("runs"))
    ub.add_argument("--name")
    _limit_args(ub, DEFAULT_BUDGET)
    ub.set_defaults(fn=cmd_utility_bench)

    sc = subparsers.add_parser("bugfix-selfcheck", help="prove every bug-fix task is well formed")
    sc.add_argument("--bugs", help="comma-separated ids or prefixes (b01,b07)")
    sc.set_defaults(fn=cmd_bugfix_selfcheck)
    bf = subparsers.add_parser("bugfix-bench", help="bug-fix reliability benchmark (benchmarks/bugfix)")
    _model_args(bf, bench=True)
    bf.add_argument("--memory", choices=["on", "off"], default="off")
    bf.add_argument("--memory-dir", help="default: a fresh <run>/memory")
    bf.add_argument("--bugs", help="comma-separated ids or prefixes (b01,b07)")
    bf.add_argument("--samples", type=positive, default=1)
    bf.add_argument("--max-cost", type=positive_float, help="USD: stop between items once list-price spend reaches it")
    bf.add_argument("--memory-rounds", action="store_true",
                    help="each sample = one seeded pass over all bugs with a fresh memory (use for off too)")
    bf.add_argument("--seed", type=int, default=0)
    bf.add_argument("--out", type=Path, default=Path("runs"))
    bf.add_argument("--name")
    _limit_args(bf)
    bf.set_defaults(fn=cmd_bugfix_bench)
