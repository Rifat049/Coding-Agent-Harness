"""Public-workspace agent evaluation with a one-way hidden-scoring boundary.

Only PublicTask crosses into agent setup. Visible checks run in the existing sandbox,
not by executing the editable checker. No provider or memory calls follow scoring.
"""
from __future__ import annotations

import json
import platform
import random
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ..checks import static_check
from ..sandbox import run_tests


@dataclass(frozen=True)
class PublicTask:
    name: str
    signature: str
    description: str
    visible: list


def public_task(task) -> PublicTask:
    """Copy an allowlist, never serialize the task object or reference solution."""
    return PublicTask(task.name, task.signature, task.description,
                      json.loads(json.dumps(task.visible)))


def materialize_task(task: PublicTask, repo: Path) -> str:
    """Write a public stub and an independently executable public example checker."""
    repo = Path(repo)
    repo.mkdir(parents=True, exist_ok=True)
    stub = task.signature.rstrip(':') + ':\n    ' + repr(task.description) + '\n    raise NotImplementedError\n'
    (repo / 'solution.py').write_text(stub, encoding='utf-8')
    checker = '''# Public examples only; benchmark verification does not trust this file.
import copy
import json
from pathlib import Path
import runpy

CASES = json.loads(%r)
fn = runpy.run_path(str(Path(__file__).with_name('solution.py')))[%r]
failures = []
for index, case in enumerate(CASES):
    try:
        actual = fn(*copy.deepcopy(case['args']))
        passed = 'raises' not in case and json.loads(json.dumps(actual)) == case['expected']
    except BaseException as error:
        passed = case.get('raises') in [cls.__name__ for cls in type(error).__mro__]
    if not passed:
        failures.append(index + 1)
if failures:
    print('FAILED visible cases:', failures)
    raise SystemExit(1)
print('PASSED %%d visible cases' %% len(CASES))
''' % (json.dumps(task.visible), task.name)
    (repo / 'check_visible.py').write_text(checker, encoding='utf-8')
    return (f'Implement `{task.signature}` in solution.py.\n\n{task.description}\n\n'
            'Public examples:\n' + json.dumps(task.visible, indent=2) +
            '\nUse run_tests to verify. check_visible.py is a protected public example checker. '
            'Write a self-contained standard-library solution; do not modify the checker.')


def _read_solution(repo: Path) -> str:
    path = repo / 'solution.py'
    if path.is_symlink() or path.resolve().parent != repo.resolve():
        return ''
    try:
        if path.stat().st_size > 2_000_000:
            return ''
        return path.read_text(encoding='utf-8')
    except (OSError, UnicodeError):
        return ''


def visible_verifier(task: PublicTask, repo: Path, per_test_timeout: float = 2.0):
    """Return a zero-argument callback closing over public data ONLY."""
    repo = Path(repo)

    def verify() -> tuple[bool, str]:
        code = _read_solution(repo)
        check = static_check(code, task.name)
        if not check.valid:
            return False, f'{check.category}: {check.detail}'
        result = run_tests(code, task.name, task.visible, per_test_timeout)
        return result.all_passed(len(task.visible)), _visible_report(task, result)

    return verify


def _visible_report(task: PublicTask, result) -> str:
    """Readable feedback built from the visible tests only (same shape as v0.1 repair feedback)."""
    if result.status != "ok":
        return f"FAILED before any test ran ({result.status}): {result.detail}"
    lines = []
    for t, c in zip(task.visible, result.cases):
        call = f"{task.name}({', '.join(repr(a) for a in t['args'])})"
        want = f"== {t['expected']!r}" if "expected" in t else f"raises {t['raises']}"
        if c.get("passed"):
            lines.append(f"ok   {call} {want}")
        else:
            lines.append(f"FAIL {call} should {want}, but {c.get('error') or 'returned ' + str(c.get('got'))}")
    n = result.n_passed
    lines.append(f"{'PASSED' if n == len(task.visible) else 'FAILED'} {n}/{len(task.visible)} visible cases")
    return "\n".join(lines)


def candidate_sources(history: list[dict], initial: str) -> list[str]:
    """Replay successful candidate writes/edits; failed tool requests are not attempts."""
    current, candidates = initial, []
    pending = {}
    for message in history:
        if message.get('role') == 'assistant':
            for call in message.get('tool_calls', []):
                pending[call['id']] = call
        elif message.get('role') == 'tool':
            call = pending.pop(message.get('tool_call_id'), None)
            if (not call or message.get('is_error') or
                    str(message.get('content', '')).startswith(('ERROR:', 'DENIED:'))):
                continue
            args = call.get('args', {})
            if str(args.get('path', '')).removeprefix('./') != 'solution.py':
                continue
            if call['name'] == 'write_file' and isinstance(args.get('content'), str):
                current = args['content']
            elif call['name'] == 'edit_file':
                old, new = args.get('old_string'), args.get('new_string')
                if not isinstance(old, str) or not old or not isinstance(new, str):
                    continue
                count = current.count(old)
                if not count or (count != 1 and not args.get('replace_all')):
                    continue
                current = current.replace(old, new) if args.get('replace_all') else current.replace(old, new, 1)
            else:
                continue
            candidates.append(current)
    return candidates


# --------------------------------------------------------------------------- the benchmark
def hidden_strings(task) -> list[str]:
    """Strings that identify hidden tests, for consolidation's benchmark-mode filter: each hidden
    call as written, plus distinctive argument/expected values that never appear in public data."""
    public = json.dumps(task.visible) + task.description + task.signature
    out = []
    for t in task.hidden:
        out.append(f"{task.name}({', '.join(repr(a) for a in t['args'])})")
        for v in list(t["args"]) + ([t["expected"]] if "expected" in t else []):
            for cand in ([v] if isinstance(v, str) else []) + [json.dumps(v)]:
                if len(cand.strip("\"' ")) >= 6 and cand not in public:
                    out.append(cand)
    return list(dict.fromkeys(out))


def _snapshot_attempts(task, sample: int, snapshots: list[dict], trace_path: Path) -> list[dict]:
    """One v0.1-style attempt per distinct candidate the harness verified. Tokens and latency come
    from the trace: LLM calls between two verifications belong to the attempt they produced."""
    events = [json.loads(l) for l in trace_path.read_text().splitlines() if l.strip()]
    zero = {"in": 0, "out": 0, "lat": 0.0}
    spans, cur = [], dict(zero)
    for e in events:
        if e["event"] == "llm_call":
            cur = {"in": cur["in"] + e["input_tokens"], "out": cur["out"] + e["output_tokens"],
                   "lat": cur["lat"] + e["latency_s"]}
        elif (e["event"] == "tool_call" and e["name"] == "run_tests") or e["event"] == "verify":
            spans.append(cur)          # one boundary per Verifier.run, in the same order as snapshots
            cur = dict(zero)
    n_verified = sum(1 for s in snapshots if s["source"] != "end")
    if len(spans) != n_verified:       # should not happen; never mis-attribute, lump it instead
        total = {k: sum(x[k] for x in spans + [cur]) for k in zero}
        spans = [total] + [dict(zero)] * (len(snapshots) - 1)
    elif len(snapshots) > n_verified:  # an unverified final edit gets the tokens after the last check
        spans.append(cur)
    elif spans:                        # otherwise the tail (e.g. the finish call) joins the last attempt
        spans[-1] = {k: spans[-1][k] + cur[k] for k in zero}
    attempts, prev = [], None
    for snap, span in zip(snapshots, spans):
        if attempts and snap["code"] == prev:          # re-verifying the same file is not a new attempt
            a = attempts[-1]
            a["prompt_tokens"] += span["in"]
            a["completion_tokens"] += span["out"]
            a["gen_latency_s"] = round(a["gen_latency_s"] + span["lat"], 4)
            continue
        code = snap["code"]
        static = static_check(code, task.name)
        vis = run_tests(code, task.name, task.visible) if static.valid else None
        attempts.append({
            "task": task.name, "sample": sample, "attempt": len(attempts) + 1,
            "response": f"[agent] solution.py after verify #{snap['n']} ({snap['source']})",
            "code": code, "extraction": "file", "static": static.category, "static_detail": static.detail,
            "invalid": not static.valid, "visible": vis.to_dict() if vis else None,
            "visible_all_passed": bool(vis and vis.all_passed(len(task.visible))),
            "prompt_tokens": span["in"], "completion_tokens": span["out"],
            "gen_latency_s": round(span["lat"], 4),
        })
        prev = code
    return attempts


def run_agent_benchmark(tasks, model: str = "mock", small_model: str = "mock", memory_enabled: bool = False,
                        samples: int = 1, config=None, out_dir: Path | str = "runs/agent", memory_dir=None,
                        seed: int = 0, consolidation_every: int = 3, per_test_timeout: float = 2.0,
                        keep_workspaces: bool = False, progress=print):
    """Run the agent on every (task, sample) in a fixed seeded order and write a v0.1-format run.

    The hidden-test firewall, step by step:
      1. Only a PublicTask (name, signature, description, visible tests) is materialised.
      2. Verification goes through the sandbox on visible tests; editing check_visible.py fakes nothing.
      3. Benchmark mode removes run_command, which could read the harness source.
      4. The run ends, and its episode is written and consolidated, BEFORE hidden scoring starts.
      5. Hidden results go only to results.jsonl. Consolidation additionally rejects any fact that
         contains a hidden-test string.
    Budgets are identical across conditions; only `memory_enabled` (recorded in prompt_style) differs."""
    from .llm import make_provider
    from .loop import AgentConfig
    from .memory import Memory
    from .runner import run_agent

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = config or AgentConfig()
    config.verify_cmd = "python3 check_visible.py"
    config.benchmark = True
    config.consolidation_every = consolidation_every
    provider = make_provider(model)
    small = make_provider(small_model, role="small") if small_model else None
    memory = None
    if memory_enabled:
        mdir = Path(memory_dir).expanduser() if memory_dir else out_dir / "memory"
        memory = Memory(mdir / "memory.sqlite3", skills_dir=mdir / "skills")
    order = [(t, s) for t in tasks for s in range(samples)]
    random.Random(seed).shuffle(order)
    # v0.1 report keys first (render_report reads them), then every agent budget, so `compare`
    # warns if any differ between conditions.
    budget = {"max_attempts": config.max_verify_rounds, "max_new_tokens": config.max_tokens,
              "max_total_tokens": config.token_budget, "per_test_timeout": per_test_timeout,
              **{k: v for k, v in asdict(config).items() if k not in ("approval",)}}
    run_config = {
        "backend": {"backend": "agent", "model": provider.describe(), "small_model": small.describe() if small else None},
        "budget": budget, "samples": samples, "temperature": 0.0,
        "prompt_style": "memory_on" if memory_enabled else "memory_off",
        "tasks": [t.name for t in tasks], "seed": seed,
        "order": [f"{t.name}#{s}" for t, s in order],
        "memory_dir": str(memory.path.parent) if memory else None,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"), "python": platform.python_version(),
    }
    (out_dir / "config.json").write_text(json.dumps(run_config, indent=2))
    traces = out_dir / "traces"
    work = Path(tempfile.mkdtemp(prefix="cbk_agent_bench_"))
    items: list[dict] = []
    t0 = time.perf_counter()
    try:
        with open(out_dir / "attempts.jsonl", "w") as fa, open(out_dir / "results.jsonl", "w") as fr:
            for task, sample in order:
                item, attempts = _run_one(task, sample, provider, small, memory, config, work, traces,
                                          per_test_timeout)
                for a in attempts:
                    fa.write(json.dumps(a) + "\n")
                fr.write(json.dumps(item) + "\n")
                fa.flush()
                fr.flush()
                items.append(item)
                mark = "PASS" if item["final_hidden_pass"] else ("INVALID" if item["final_attempt"] is None else "fail")
                progress(f"  {task.name:<16} s{sample}  {mark:<7} {item['outcome']:<8} "
                         f"attempts={item['attempts_used']} iters={item['iterations']} "
                         f"hidden={item['final_hidden_cases_passed']}/{item['hidden_cases_total']}"
                         + (f" mem={len(item['memory_facts'])}" if memory_enabled else ""))
    finally:
        if memory is not None:
            memory.close()
        if keep_workspaces:
            run_config["workspaces"] = str(work)
        else:
            shutil.rmtree(work, ignore_errors=True)
    run_config["wall_s"] = round(time.perf_counter() - t0, 2)
    (out_dir / "config.json").write_text(json.dumps(run_config, indent=2))
    return run_config, items


def _run_one(task, sample, provider, small, memory, config, work: Path, traces: Path, per_test_timeout):
    from .runner import run_agent
    pub = public_task(task)
    repo = work / f"{task.name}-s{sample}"
    prompt = materialize_task(pub, repo)
    snapshots: list[dict] = []

    def on_verify(res):
        snapshots.append({"code": _read_solution(repo), "passed": res.passed, "source": res.source,
                          "n": len(snapshots) + 1})

    trace = traces / f"{task.name}-s{sample}.jsonl"
    res = run_agent(prompt, repo, provider, config, memory=memory, small=small, trace_path=trace,
                    verify_check=visible_verifier(pub, repo, per_test_timeout), on_verify=on_verify,
                    scope="repo:agent-bench", hidden_strings=hidden_strings(task) if memory else ())
    # ---- everything below is hidden scoring. Nothing here reaches the agent, its trace or memory.
    final_code = _read_solution(repo)
    if not snapshots or snapshots[-1]["code"] != final_code:
        snapshots.append({"code": final_code, "passed": None, "source": "end", "n": len(snapshots) + 1})
    attempts = _snapshot_attempts(task, sample, snapshots, trace)
    n_hid = len(task.hidden)

    def hidden(code):
        st = static_check(code, task.name)
        return run_tests(code, task.name, task.hidden, per_test_timeout) if st.valid else None

    first = attempts[0]
    first_hidden = hidden(first["code"]) if not first["invalid"] else None
    final_static = static_check(final_code, task.name)
    final_hidden = hidden(final_code) if final_static.valid else None
    final_attempt = next((a["attempt"] for a in reversed(attempts) if a["code"] == final_code), len(attempts))
    item = {
        "task": task.name, "sample": sample, "attempts_used": len(attempts),
        "completion_tokens": res.receipt["output_tokens"],
        "gen_latency_s": round(res.receipt["llm_latency_s"], 4),
        "first_invalid": first["invalid"],
        "first_invalid_category": first["static"] if first["invalid"] else None,
        "first_visible_pass": first["visible_all_passed"],
        "first_hidden_pass": bool(first_hidden and first_hidden.all_passed(n_hid)),
        "final_attempt": final_attempt if final_static.valid else None,
        "final_visible_pass": bool(attempts[final_attempt - 1]["visible_all_passed"]) if final_static.valid else False,
        "final_hidden_pass": bool(final_hidden and final_hidden.all_passed(n_hid)),
        "final_hidden_cases_passed": final_hidden.n_passed if final_hidden and final_hidden.status == "ok" else 0,
        "hidden_cases_total": n_hid,
        "final_hidden_runtime_s": round(final_hidden.runtime_s, 4) if final_hidden else None,
        "final_timeout": bool(final_hidden and final_hidden.any_timeout),
        "final_hidden": final_hidden.to_dict() if final_hidden else None,
        "final_code": final_code,
        # agent-only fields (ignored by v0.1 metrics)
        "outcome": res.outcome, "stop_reason": res.receipt["stop_reason"],
        "iterations": res.receipt["iterations"], "input_tokens": res.receipt["input_tokens"],
        "tokens": res.receipt["tokens"], "verify_rounds": res.receipt["verify_rounds"],
        "memory_facts": res.context.get("memory_ids", {}).get("facts", []),
        "trace": str(trace),
    }
    return item, attempts
