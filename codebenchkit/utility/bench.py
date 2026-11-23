"""utility-bench: run a code model on utility-v1 under a fixed budget and score it on untouched tests.

Per item (task x sample), in a seeded order:
  1. Copy the task's workspace (README spec, module, visible tests) into a fresh directory.
  2. Run the agent loop. run_tests is a trusted check: the module is copied next to PRISTINE visible
     tests in a scratch dir. run_command is not offered (it is not path-confined).
  3. Score: the pristine hidden tests run against the final module only, in a fresh scratch dir.

Headline metrics (results.jsonl fields; definitions in README "Metrics"):
  passed                every hidden test passes and the item is not flagged for cheating
  hidden_passed/total   hidden test cases passed (partial credit, reported separately)
  runtime               wall_s per item = model_s (time waiting for the model) + test_s (visible
                        checks run during the item: run_tests and the harness's own verify) +
                        other_s (tools, file I/O, loop overhead); scoring_s (hidden suite) is
                        reported separately and is not part of wall_s
  invalid output        see invalid_output() / README: malformed tool calls, unusable edits, syntax
                        errors, and turns with no usable action, per turn and per item
plus regressions (fix tasks), cheating, false success, cost, tokens, as in bugfix-bench.

A billing/auth error (HTTP 401/402/403) stops the run: that item and every later one are written as
not_run and excluded from every total; completed items are kept.
"""
from __future__ import annotations

import copy
import json
import platform
import random
import shutil
import tempfile
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from ..agent.pricing import trace_cost
from ..bugfix.bench import _attempts, _events, _small_cost
from ..bugfix.core import passed, visible_report
from .core import (SUITE, TEST_CMD, TEST_RX, Task, cheat_reasons, item_passed, materialize, prompt, provenance,
                   read_module, run_visible, score_hidden)


def hidden_test_count(task: Task) -> int:
    return len(TEST_RX.findall(task.hidden))

# Fixed default budget for utility-v1 runs (identical across paired conditions; compare warns otherwise).
# max_verify_rounds 5 (the attempt budget: harness test runs on a stop) was set in calibration: with 3,
# 8 of 10 qwen2.5-coder smoke items ended on it after about 5 of their 20 turns.
DEFAULT_BUDGET = {"max_iterations": 20, "max_new_tokens": 2048, "token_budget": 200_000, "wall_seconds": 600,
                  "max_verify_rounds": 5, "context_limit": 16_000}
INVALID_KINDS = ("malformed_tool_call", "unusable_edit", "syntax_error", "no_action")
LOOPS = ("agent", "simple")      # simple: whole-file replies, see utility/simple.py (added after the freeze)


def run_utility_benchmark(tasks: list[Task], model: str = "mock", small_model: str | None = None, samples: int = 1,
                          config=None, out_dir="runs/utility", seed: int = 0, per_test_timeout: float = 5.0,
                          prompt_variant: str = "baseline", max_cost: float | None = None, progress=print,
                          provider=None, loop: str = "agent"):
    """`provider` injects a ready provider (tests); otherwise `model` is a spec for make_provider."""
    from ..agent.llm import FatalProviderError, make_provider
    from ..agent.loop import AgentConfig
    from .mock import SimpleMock, UtilityMock, variant_of
    if loop not in LOOPS:
        raise ValueError(f"unknown loop {loop!r}; use {', '.join(LOOPS)}")
    if loop == "simple" and prompt_variant != "baseline":
        raise ValueError("--loop simple has one prompt (its own whole-file prompt); use --prompt-variant baseline")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = _prepare(config or AgentConfig())
    is_mock = provider is None and model.startswith("mock")
    variant = variant_of(model) if is_mock else None
    if provider is None and not is_mock:
        provider = make_provider(model)
    small = make_provider(small_model, role="small") if small_model else None
    order = [(t, s) for t in tasks for s in range(samples)]
    random.Random(seed).shuffle(order)
    budget = _budget(config, per_test_timeout)
    model_desc = (f"mock:utility-{'simple-' if loop == 'simple' else ''}{variant}" if is_mock else provider.describe())
    run_config = {
        "benchmark": SUITE,
        "backend": {"backend": "agent", "model": model_desc, "small_model": small.describe() if small else None},
        "sampling": None if is_mock else provider.sampling(),
        "budget": budget, "samples": samples, "temperature": (provider.sampling().get("temperature")
                                                              if provider is not None else 0.0),
        "prompt_style": prompt_variant if loop == "agent" else f"simple-{budget['max_attempts']}",
        "loop": loop, "tasks": [t.id for t in tasks], "seed": seed,
        "order": [f"{t.id}#{s}" for t, s in order], "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(), **provenance(),
    }
    (out_dir / "config.json").write_text(json.dumps(run_config, indent=2))
    if max_cost is not None and not is_mock:
        from ..agent.pricing import rates
        for desc in [model_desc] + ([small.describe()] if small is not None else []):
            if rates(desc) is None:
                raise ValueError(f"--max-cost needs a known price for {desc}; add it to agent/pricing.py")
    work = Path(tempfile.mkdtemp(prefix="cbk_utility_"))
    items, spent, stop = [], 0.0, None
    t0 = time.perf_counter()
    try:
        with open(out_dir / "attempts.jsonl", "w") as fa, open(out_dir / "results.jsonl", "w") as fr:
            for task, sample in order:
                if stop:
                    item, attempts = not_run_item(task, sample, stop), []
                else:
                    coder = ((SimpleMock if loop == "simple" else UtilityMock)(task, variant) if is_mock else provider)
                    try:
                        item, attempts = run_one(task, sample, coder, small, config, work, out_dir / "traces",
                                                 per_test_timeout, model_desc, prompt_variant, loop)
                    except FatalProviderError as err:       # raised outside the agent loop (rare)
                        item, attempts = not_run_item(task, sample, err.reason(model_desc)), []
                    if item.get("not_run"):
                        stop = item["not_run_reason"]
                        run_config["stopped"] = (f"{stop}: stopped at item {len(items) + 1}/{len(order)}; this item and "
                                                 f"the rest are not_run and excluded from totals")
                        progress(f"STOPPED: {run_config['stopped']}")
                for a in attempts:
                    fa.write(json.dumps(a) + "\n")
                fr.write(json.dumps(item) + "\n")
                fa.flush()
                fr.flush()
                items.append(item)
                if item.get("not_run"):
                    continue
                spent += item["cost_usd"] or 0.0
                flags = [k for k in ("cheated", "false_success", "regression") if item[k]]
                progress(f"  {task.id:<22} s{sample}  {'PASS' if item['passed'] else 'fail':<4} hidden "
                         f"{item['hidden_passed']:>2}/{item['hidden_total']:<2} iters={item['iterations']:<2} "
                         f"invalid={item['invalid_turns']}/{item['turns']} wall={item['wall_s']:.1f}s "
                         f"(model {item['model_s']:.1f}s) {' '.join(flags).upper()}  ${spent:.4f}")
                if max_cost is not None and spent >= max_cost and not stop:
                    stop = f"cost cap ${max_cost} reached (${spent:.4f})"
                    run_config["stopped"] = f"{stop} after {len(items)}/{len(order)} items; the rest are not_run"
                    progress(f"STOPPED: {run_config['stopped']}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    run_config["wall_s"] = round(time.perf_counter() - t0, 2)
    run_config["cost_usd"] = round(spent, 6)
    run_config["not_run"] = sum(1 for i in items if i.get("not_run"))
    (out_dir / "config.json").write_text(json.dumps(run_config, indent=2))
    return run_config, items


def _prepare(config):
    config.verify_cmd = TEST_CMD
    config.benchmark = True
    config.consolidation_every = 0
    config.verify_without_edits = True       # see AgentConfig: a failed finish always gets test feedback
    return config


def _budget(config, per_test_timeout: float) -> dict:
    return {"max_attempts": config.max_verify_rounds, "max_new_tokens": config.max_tokens,
            "max_total_tokens": config.token_budget, "per_test_timeout": per_test_timeout,
            **{k: v for k, v in asdict(config).items() if k not in ("approval", "benchmark_commands",
                                                                     "command_extra_deny", "small_model_note")}}


def resume_utility_benchmark(run_dir, model: str, config=None, per_test_timeout: float = 5.0,
                             max_cost: float | None = None, progress=print, provider=None):
    """Run only the not_run items of an earlier run (stopped by a cost cap or a billing error), in
    their original order, and write them into the same results.jsonl. Refuses unless the suite hash,
    the model and the budget equal the original run's, so the arm stays one condition. The new spend
    has its own cap; config.json keeps the original stop and records each resume under "resumed"."""
    from ..agent.llm import FatalProviderError, make_provider
    from ..agent.loop import AgentConfig
    from .core import load_tasks
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    items = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines() if l.strip()]
    todo = [n for n, it in enumerate(items) if it.get("not_run")]
    if not todo:
        raise ValueError(f"nothing to resume in {run_dir}: it has no not_run items")
    prov = provenance()
    if prov["benchmark_sha256"] != cfg.get("benchmark_sha256"):
        raise ValueError("the suite hash differs from the original run's; refusing to mix two suites in one arm")
    config = _prepare(config or AgentConfig())
    if _budget(config, per_test_timeout) != cfg["budget"]:
        diff = {k: (cfg["budget"].get(k), v) for k, v in _budget(config, per_test_timeout).items() if cfg["budget"].get(k) != v}
        raise ValueError(f"the budget differs from the original run's (original, now): {diff}; pass the same flags")
    provider = provider or make_provider(model)
    model_desc = provider.describe()
    if model_desc != cfg["backend"]["model"]:
        raise ValueError(f"model {model_desc} differs from the original run's {cfg['backend']['model']}")
    if max_cost is not None:
        from ..agent.pricing import rates
        if rates(model_desc) is None:
            raise ValueError(f"--max-cost needs a known price for {model_desc}")
    loop = cfg.get("loop", "agent")
    variant = cfg["prompt_style"] if loop == "agent" else "baseline"
    tasks = {t.id: t for t in load_tasks()}
    work = Path(tempfile.mkdtemp(prefix="cbk_utility_"))
    spent, stop, done = 0.0, None, 0
    t0 = time.perf_counter()
    try:
        with open(run_dir / "attempts.jsonl", "a") as fa:
            for n in todo:
                task, sample = tasks[items[n]["task"]], items[n]["sample"]
                if stop:
                    items[n] = not_run_item(task, sample, stop)
                    continue
                try:
                    item, attempts = run_one(task, sample, provider, None, config, work, run_dir / "traces",
                                             per_test_timeout, model_desc, variant, loop)
                except FatalProviderError as err:
                    item, attempts = not_run_item(task, sample, err.reason(model_desc)), []
                items[n] = item
                if item.get("not_run"):
                    stop = item["not_run_reason"]
                    progress(f"STOPPED: {stop}")
                    continue
                item["resumed"] = True
                for a in attempts:
                    fa.write(json.dumps(a) + "\n")
                fa.flush()
                done += 1
                spent += item["cost_usd"] or 0.0
                progress(f"  {task.id:<22} s{sample}  {'PASS' if item['passed'] else 'fail':<4} hidden "
                         f"{item['hidden_passed']:>2}/{item['hidden_total']:<2} (resumed)  ${spent:.4f}")
                if max_cost is not None and spent >= max_cost:
                    stop = f"resume cost cap ${max_cost} reached (${spent:.4f})"
                    progress(f"STOPPED: {stop}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    (run_dir / "results.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    cfg.setdefault("resumed", []).append({
        "at": time.strftime("%Y-%m-%d %H:%M:%S"), "items_resumed": done, "items_requested": len(todo),
        "cost_usd": round(spent, 6), "max_cost": max_cost, "stopped": stop, "previous_stop": cfg.get("stopped"),
        "wall_s": round(time.perf_counter() - t0, 2), "git_commit": prov["git_commit"]})
    cfg["cost_usd"] = round(cfg.get("cost_usd", 0.0) + spent, 6)
    cfg["not_run"] = sum(1 for i in items if i.get("not_run"))
    cfg["stopped"] = stop
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    return cfg, items


def not_run_item(task: Task, sample: int, reason: str) -> dict:
    """An item that was never scored. Excluded from every total and from compare's pairs."""
    return {"task": task.id, "sample": sample, "kind": task.kind, "outcome": "not_run", "not_run": True,
            "not_run_reason": reason}


def run_one(task: Task, sample: int, coder, small, base_config, work: Path, traces: Path, per_test_timeout: float,
            model_desc: str, prompt_variant: str = "baseline", loop: str = "agent"):
    from ..agent.runner import run_agent
    config = copy.copy(base_config)
    repo = work / f"{task.id}-s{sample}"
    test_hashes = materialize(task, repo)
    key = f"{task.id}-s{sample}-{time.time_ns() % 10**9}"
    snapshots: list[dict] = []
    test_s = [0.0]

    def check():
        t0 = time.perf_counter()
        res = run_visible(task, read_module(repo, task), per_test_timeout)
        test_s[0] += time.perf_counter() - t0
        return passed(res), visible_report(res)

    def on_verify(res):
        snapshots.append({"source": read_module(repo, task), "passed": res.passed, "kind": res.source})

    trace = traces / f"{task.id}-s{sample}.jsonl"
    small_log_start = len(getattr(small, "usage_log", []) or [])
    started = time.perf_counter()
    if loop == "simple":
        from .simple import run_simple
        res = run_simple(task, coder, config, repo, check, on_verify, trace)
    else:
        res = run_agent(prompt(task, prompt_variant), repo, coder, config, memory=None, small=small, trace_path=trace,
                        verify_check=check, on_verify=on_verify, scope=f"repo:{task.module}")
    wall = time.perf_counter() - started
    rc = res.receipt
    if res.outcome == "not_run":
        item = not_run_item(task, sample, rc.get("not_run_reason") or res.fatal or "fatal provider error")
        item.update({"trace": str(trace), "partial_tokens": rc.get("tokens", 0), "fatal_status": rc.get("fatal_status")})
        return item, []

    # ---- scoring on untouched tests: nothing below reaches the agent or its trace.
    final = read_module(repo, task)
    hidden, scoring_s = score_hidden(task, final, per_test_timeout)
    start_hidden, _ = score_hidden(task, {task.module_file: task.start_source}, per_test_timeout)
    final_visible = run_visible(task, final, per_test_timeout)
    events = _events(trace)
    reasons = cheat_reasons(task, repo, test_hashes, events, final[task.module_file])
    ok = item_passed(hidden, reasons)
    regressions = [t for t, r in hidden.items() if r["status"] != "pass" and start_hidden.get(t, {}).get("status") == "pass"]
    finishes = [e for e in events if e.get("event") == "tool_call" and e.get("name") == "finish"]
    claimed = bool(finishes) or (rc["stop_reason"] in ("finished", "end_turn") and not rc.get("guardrail"))
    trusted_pass = passed(final_visible)
    inv = invalid_output(events)
    tool_calls = [e for e in events if e.get("event") == "tool_call"]
    tool_errors = sum(1 for e in tool_calls if not e.get("ok"))
    main_cost = trace_cost(events, model_desc)
    small_cost = _small_cost(small, small_log_start)
    cost = None if main_cost is None or small_cost is None else main_cost + small_cost
    model_s = sum(e.get("latency_s", 0.0) for e in events if e.get("event") == "llm_call")

    class _Id:                       # _attempts only reads .id
        id = task.id
    attempts = _attempts(_Id, sample, snapshots, events, final, trusted_pass)
    first_src = snapshots[0]["source"] if snapshots else final
    first_h = hidden if first_src == final else score_hidden(task, first_src, per_test_timeout)[0]
    # The denominator is the task's real number of hidden tests: a module that fails to import
    # reports a single "<import>" error, which must not shrink it.
    n_hidden = max(len(hidden), hidden_test_count(task))
    item = {
        # v0.1 fields (metrics.summarize / compare)
        "task": task.id, "sample": sample, "attempts_used": len(attempts),
        "completion_tokens": rc["output_tokens"], "gen_latency_s": round(rc["llm_latency_s"], 4),
        "first_invalid": attempts[0]["invalid"], "first_invalid_category": attempts[0]["static"] if attempts[0]["invalid"] else None,
        "first_visible_pass": attempts[0]["visible_all_passed"], "first_hidden_pass": passed(first_h) and not reasons,
        "final_attempt": len(attempts), "final_visible_pass": trusted_pass, "final_hidden_pass": ok,
        "final_hidden_cases_passed": sum(r["status"] == "pass" for r in hidden.values()),  # 0 on import error
        "hidden_cases_total": n_hidden, "final_hidden_runtime_s": round(scoring_s, 3),
        "final_timeout": any(r["status"] == "timeout" for r in hidden.values()),
        "final_hidden": {t: r["status"] for t, r in hidden.items()}, "final_code": final[task.module_file],
        # utility-v1 headline
        "kind": task.kind, "prompt_variant": prompt_variant, "loop": loop, "passed": ok,
        "hidden_passed": sum(r["status"] == "pass" for r in hidden.values()), "hidden_total": n_hidden,
        # runtime
        "wall_s": round(wall, 3), "model_s": round(model_s, 3), "test_s": round(test_s[0], 3),
        "other_s": round(max(0.0, wall - model_s - test_s[0]), 3), "scoring_s": round(scoring_s, 3),
        # invalid output
        **inv,
        # reliability (as in bugfix-bench)
        "regression": bool(regressions), "regressions": regressions,
        "cheated": bool(reasons), "cheat_reasons": reasons,
        "claimed_done": claimed, "trusted_visible_pass": trusted_pass, "false_success": claimed and not trusted_pass,
        "overfit": trusted_pass and not passed(hidden) and not reasons,
        "outcome": res.outcome, "stop_reason": rc["stop_reason"], "guardrail": rc.get("guardrail"),
        "iterations": rc["iterations"], "input_tokens": rc["input_tokens"], "output_tokens": rc["output_tokens"],
        "cache_read_tokens": rc["cache_read_tokens"], "tokens": rc["tokens"], "verify_rounds": rc["verify_rounds"],
        "tool_calls": len(tool_calls), "tool_errors": tool_errors,
        "cost_usd": round(cost, 6) if cost is not None else None,
        "main_cost_usd": round(main_cost, 6) if main_cost is not None else None,
        "small_cost_usd": round(small_cost, 6) if small_cost is not None else None,
        "trace": str(trace),
    }
    item["failure_category"] = failure_category(item, task.start_source)
    return item, attempts


# --------------------------------------------------------------------------- failure categories
FAILURE_CATEGORIES = ("cheated", "unchanged", "broken", "visible_fail", "hidden_only")


def failure_category(item: dict, start_source: str | None = None) -> str | None:
    """Why an item failed, one category, first match (None if it passed). Reporting only, computed from
    fields every item records, so it applies to runs made before it existed:
      cheated       flagged by the cheat checks (whatever the tests say)
      unchanged     the final module is identical to the starting module
      broken        the module does not import (syntax error or import-time crash): hidden "<import>" error
      visible_fail  imports, but the visible tests still fail
      hidden_only   the visible tests pass, some hidden test fails (a stated rule the visible tests do not cover)"""
    if item.get("not_run") or item.get("passed"):
        return None
    if item.get("cheated"):
        return "cheated"
    if start_source is not None and item.get("final_code", "").strip() == start_source.strip():
        return "unchanged"
    if any(k.endswith("::<import>") for k in item.get("final_hidden", {})):
        return "broken"
    if not item.get("trusted_visible_pass", item.get("final_visible_pass")):
        return "visible_fail"
    return "hidden_only"


# --------------------------------------------------------------------------- invalid output
def invalid_output(events: list[dict]) -> dict:
    """Classify every agent turn (one model call of kind "loop"; the tools-off final answer after a
    guardrail is not a turn) from the trace. A turn is INVALID if any of:

      malformed_tool_call  a tool call the harness could not run as written: arguments that are not
                           a JSON object, an unknown tool name, missing or unexpected arguments, or
                           text that looked like a tool call but could not be parsed
      unusable_edit        edit_file that changed nothing: old_string empty, not found, or ambiguous
      syntax_error         write_file/edit_file left a .py file that does not compile
      no_action            no tool call at all, and the visible tests were not passing at that
                           point (so the reply did not end the task; it was not a usable action)

    A turn counts once even if it has several problems (its first kind is recorded). Tool calls
    recovered from JSON text (tool_call_source "text") are valid but counted in text_tool_turns.
    Denied calls (policy, path outside the workspace) are not invalid output: they are flagged as
    cheating attempts or tool errors instead."""
    turns, kinds = [], []
    current = None
    passing = False                    # last visible-check result seen in the trace
    for e in events:
        ev = e.get("event")
        if ev == "llm_call" and e.get("kind") == "loop":
            current = {"tools": list(e.get("tool_calls") or []), "source": e.get("tool_call_source"),
                       "text_error": e.get("text_tool_error"), "kind": None, "passing_before": passing,
                       "truncated": e.get("stop_reason") in ("length", "max_tokens")}
            turns.append(current)
            if e.get("text_tool_error"):
                current["kind"] = "malformed_tool_call"
        elif ev == "tool_call" and current is not None:
            out = str(e.get("output", ""))
            name = e.get("name")
            k = None
            if out.startswith(("ERROR: unknown tool", "ERROR: arguments were not valid JSON",
                               "ERROR: arguments must be a JSON object")) or \
                    (out.startswith("ERROR: TypeError") and ("argument" in out)):
                k = "malformed_tool_call"
            elif name == "edit_file" and (out.startswith("ERROR: old_string") or
                                          (out.startswith("ERROR:") and "does not exist" in out)):
                k = "unusable_edit"
            elif name in ("write_file", "edit_file") and e.get("syntax_error"):
                k = "syntax_error"
            if k and current["kind"] is None:
                current["kind"] = k
            if name == "run_tests":
                passing = out.startswith("PASSED")
        elif ev == "verify":
            passing = bool(e.get("passed"))
    for t in turns:
        if t["kind"] is None and not t["tools"] and not t["passing_before"]:
            t["kind"] = "no_action"
        kinds.append(t["kind"])
    bad = [k for k in kinds if k]
    return {"turns": len(turns), "invalid_turns": len(bad),
            "invalid_by_kind": {k: bad.count(k) for k in INVALID_KINDS if bad.count(k)},
            "invalid_item": bool(bad), "text_tool_turns": sum(1 for t in turns if t["source"] == "text"),
            # Reporting only: turns cut off at the per-call output cap (stop_reason "length"). For a
            # reasoning model (DeepSeek thinking mode) the cap counts reasoning tokens, so a long think
            # can end with no content and no tool call, which is then also a no_action turn.
            "truncated_turns": sum(1 for t in turns if t["truncated"]),
            "native_tool_turns": sum(1 for t in turns if t["source"] == "native")}


# --------------------------------------------------------------------------- report
def _rate(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _category(item: dict) -> str:
    if item.get("failure_category"):
        return item["failure_category"]
    from .core import load_tasks
    start = {t.id: t.start_source for t in load_tasks()}.get(item["task"])
    return failure_category(item, start) or "passed"


def utility_metrics(items: list[dict]) -> dict:
    scored = [i for i in items if not i.get("not_run")]
    n = len(scored)
    turns = sum(i["turns"] for i in scored)
    inv_turns = sum(i["invalid_turns"] for i in scored)
    by_kind = Counter()
    for i in scored:
        by_kind.update(i["invalid_by_kind"])
    per_task = {}
    for t in sorted({i["task"] for i in scored}):
        xs = [i for i in scored if i["task"] == t]
        per_task[t] = {"kind": xs[0]["kind"], "n": len(xs), "pass_rate": _rate(i["passed"] for i in xs),
                       "passes": sum(i["passed"] for i in xs),
                       "hidden_cases": f"{sum(i['hidden_passed'] for i in xs)}/{sum(i['hidden_total'] for i in xs)}",
                       "wall_s_mean": round(sum(i["wall_s"] for i in xs) / len(xs), 2),
                       "model_s_mean": round(sum(i["model_s"] for i in xs) / len(xs), 2),
                       "invalid_turns": f"{sum(i['invalid_turns'] for i in xs)}/{sum(i['turns'] for i in xs)}",
                       "invalid_items": sum(i["invalid_item"] for i in xs)}
    costs = [i["cost_usd"] for i in scored]
    from ..metrics import wilson
    k = sum(i["passed"] for i in scored)
    return {
        "n_items": n, "not_run": len(items) - n,
        "not_run_reasons": dict(Counter(i["not_run_reason"] for i in items if i.get("not_run"))),
        "pass_rate": _rate(i["passed"] for i in scored), "passes": k, "pass_rate_ci95": wilson(k, n),
        "hidden_case_rate": (sum(i["hidden_passed"] for i in scored) / max(1, sum(i["hidden_total"] for i in scored))) if n else None,
        "wall_s_total": round(sum(i["wall_s"] for i in scored), 2),
        "model_s_total": round(sum(i["model_s"] for i in scored), 2),
        "test_s_total": round(sum(i["test_s"] for i in scored), 2),
        "other_s_total": round(sum(i["other_s"] for i in scored), 2),
        "scoring_s_total": round(sum(i["scoring_s"] for i in scored), 2),
        "wall_s_mean": round(sum(i["wall_s"] for i in scored) / n, 2) if n else None,
        "wall_s_median": sorted(i["wall_s"] for i in scored)[n // 2] if n else None,
        "turns": turns, "invalid_turns": inv_turns, "invalid_turn_rate": inv_turns / turns if turns else None,
        "invalid_item_rate": _rate(i["invalid_item"] for i in scored), "invalid_by_kind": dict(by_kind),
        "text_tool_turn_rate": (sum(i["text_tool_turns"] for i in scored) / turns) if turns else None,
        "regression_rate_fix_tasks": _rate(i["regression"] for i in scored if i["kind"] == "fix"),
        "cheat_rate": _rate(i["cheated"] for i in scored),
        "cheat_reasons": dict(Counter(r.split(" (")[0] for i in scored for r in i["cheat_reasons"])),
        "false_success_rate": _rate(i["false_success"] for i in scored),
        "claimed_done_rate": _rate(i["claimed_done"] for i in scored),
        "mean_iterations": _rate(i["iterations"] for i in scored),
        "failure_categories": dict(Counter(_category(i) for i in scored if not i["passed"])),
        "total_tokens": sum(i["tokens"] for i in scored),
        "mean_tokens": _rate(i["tokens"] for i in scored),
        "cost_usd_total": None if any(c is None for c in costs) else round(sum(costs), 4),
        "per_task": per_task,
    }


def render_utility(config: dict, m: dict) -> str:
    lo, hi = m["pass_rate_ci95"]
    s = config.get("sampling") or {}
    lines = [
        f"# {config['benchmark']}: {config['backend']['model']} (prompt `{config['prompt_style']}`)", "",
        f"{len(config['tasks'])} tasks x {config['samples']} samples = {m['n_items']} scored items"
        + (f" (+{m['not_run']} not_run: {m['not_run_reasons']})" if m["not_run"] else "")
        + f" | seed {config['seed']} | sampling: temperature {s.get('temperature')}, seed {s.get('seed')}"
        + (f" ({s.get('note')})" if s.get("note") else ""),
        f"Budget: {config['budget']['max_iterations']} model calls, {config['budget']['max_tokens']} output tokens/call, "
        f"{config['budget']['token_budget']:,} tokens/item, {config['budget']['wall_seconds']:.0f} s/item, "
        f"context {config['budget']['context_limit']:,} | suite sha256 {config.get('benchmark_sha256', '')[:12]} "
        f"tag {config.get('git_tag')} commit {str(config.get('git_commit', ''))[:8]}"
        + (" (DIRTY)" if config.get("benchmark_dirty") else ""),
        "", "## Headline", "",
        "| Metric | Value |", "|---|---|",
        f"| **Pass rate on untouched tests** (all hidden tests pass, no cheating) | **{_pct(m['pass_rate'])}** "
        f"({m['passes']}/{m['n_items']}; 95% CI {_pct(lo)} to {_pct(hi)}) |",
        f"| Hidden test cases passed (partial credit) | {_pct(m['hidden_case_rate'])} |",
        f"| **Runtime**, wall clock | **{m['wall_s_total']:.0f} s total**, {m['wall_s_mean']} s mean, {m['wall_s_median']} s median per item |"
        if m["n_items"] else "| Runtime | n/a |",
        f"| of which model time | {m['model_s_total']:.0f} s |",
        f"| of which test time (visible checks during the run) | {m['test_s_total']:.1f} s |",
        f"| of which tools and harness | {m['other_s_total']:.1f} s |",
        f"| Hidden-test scoring (after the run, not in wall) | {m['scoring_s_total']:.1f} s |",
        f"| **Invalid-output rate**, per turn | **{_pct(m['invalid_turn_rate'])}** ({m['invalid_turns']}/{m['turns']} turns) |",
        f"| Invalid-output rate, per item (>= 1 invalid turn) | {_pct(m['invalid_item_rate'])} |",
        f"| Invalid turns by kind | {m['invalid_by_kind'] or 'none'} |",
        f"| Tool calls recovered from JSON text | {_pct(m['text_tool_turn_rate'])} of turns |",
        "", "## Reliability", "", "| Metric | Value |", "|---|---|",
        f"| Regression rate (fix tasks: a hidden test that passed on the buggy code now fails) | {_pct(m['regression_rate_fix_tasks'])} |",
        f"| Cheat rate | {_pct(m['cheat_rate'])} {m['cheat_reasons'] or ''} |",
        f"| False-success rate (claimed done, visible tests fail) | {_pct(m['false_success_rate'])} (claimed done {_pct(m['claimed_done_rate'])}) |",
        f"| Failure categories (failed items) | {m['failure_categories'] or 'none'} |",
        f"| Mean iterations | {m['mean_iterations']:.1f} |" if m["mean_iterations"] is not None else "| Mean iterations | n/a |",
        f"| Tokens | {m['total_tokens']:,} total, {m['mean_tokens']:,.0f} per item |" if m["mean_tokens"] is not None else "| Tokens | n/a |",
        f"| Cost (list prices) | {'$%.4f' % m['cost_usd_total'] if m['cost_usd_total'] is not None else 'unknown'} |",
        "", "## Per task", "",
        "| Task | Kind | Pass | Hidden cases | Wall s (mean) | Model s (mean) | Invalid turns | Items with invalid output |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for t, r in m["per_task"].items():
        lines.append(f"| {t} | {r['kind']} | {r['passes']}/{r['n']} | {r['hidden_cases']} | {r['wall_s_mean']} | "
                     f"{r['model_s_mean']} | {r['invalid_turns']} | {r['invalid_items']}/{r['n']} |")
    return "\n".join(lines) + "\n"
