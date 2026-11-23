"""--loop simple: a small-model loop for single-file tasks. Added AFTER the utility-v1 freeze; it is a
new way to drive the model, not a change to the tasks, tests or scoring (core.py is untouched).

    attempt 1..N:   model gets README + current module + visible tests (attempt 1), or the
                    visible-test output of its last file (later attempts), and replies with the
                    WHOLE target file in one ```python block
                    harness extracts the block, writes it to the module, runs the visible tests
                    (the same trusted check as the agent loop: pristine copies in a scratch dir)
                    pass -> stop;  fail -> feed back the output and repeat
    then            the same hidden scoring, cheat checks, receipts and metrics as the agent loop

No tools: the model never chooses files, never edits tests and cannot reach the hidden tests.
N = the budget's max_verify_rounds (5 by default, like the agent loop's test rounds); N = 1 is the
one-shot arm. The token, wall-clock and output-per-call budgets apply as in the agent loop.

The trace uses the agent loop's event names, so utility.bench.invalid_output() classifies it
unchanged: a reply with no code block is `no_action`, a block that does not compile is
`syntax_error`. A turn is one model call. There is no "finish" claim (the harness stops on a pass),
so false success is always 0 here.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from ..agent.llm import FatalProviderError, ProviderError
from ..agent.trace import Tracer, preview
from .core import Task

FENCE = re.compile(r"```[ \t]*(?:python|py|python3)?[ \t]*\n(.*?)```", re.S | re.I)
SYSTEM = "You are an expert Python programmer. You write complete, correct, standard-library-only Python files."


def first_prompt(task: Task) -> str:
    visible = "\n\n".join(f"`{p}`:\n```python\n{t}```" for p, t in task.visible.items())
    if task.kind == "implement":
        ask = f"Implement `{task.signature}` in `{task.module_file}`; it currently raises NotImplementedError."
    else:
        ask = (f"`{task.module_file}` does not follow its specification. Bug report: {task.report} "
               f"Fix it so that it follows every rule.")
    return (f"{task.readme}\n\nThe current `{task.module_file}`:\n\n```python\n{task.start_source}```\n\n"
            f"The visible tests:\n\n{visible}\n\n{ask} Hidden tests check only behaviour stated in the "
            f"numbered rules. Keep the same names and signatures and use only the standard library.\n\n"
            f"Reply with the complete `{task.module_file}` in one ```python code block.")


def feedback(task: Task, report: str) -> str:
    return (f"The harness wrote your file to `{task.module_file}` and ran the visible tests. They FAILED:\n\n"
            f"{report[-4000:]}\n\nReply with the complete corrected `{task.module_file}` in one ```python code block.")


NO_CODE = "Your reply had no ```python code block. Reply with the complete `{module}` in one ```python code block."


def extract(text: str) -> str:
    blocks = FENCE.findall(text or "")
    return max(blocks, key=len) if blocks else ""


@dataclass
class _State:
    finish_summary: str = ""
    notes: list = field(default_factory=list)
    files_changed: set = field(default_factory=set)


@dataclass
class SimpleResult:
    reply: str
    outcome: str
    receipt: dict
    state: _State
    trace_path: str
    context: dict = field(default_factory=dict)
    fatal: str | None = None


def run_simple(task: Task, provider, config, repo, check, on_verify, trace_path) -> SimpleResult:
    """check() -> (passed, report) is the trusted visible check; on_verify(res) gets each result."""
    from ..agent.tools import VerifyResult
    tracer = Tracer(trace_path)
    started = time.monotonic()
    attempts = max(1, config.max_verify_rounds)
    messages = [{"role": "user", "content": first_prompt(task)}]
    r = {"iterations": 0, "llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
         "llm_latency_s": 0.0, "verify_rounds": 0}
    stop, guardrail, reply, outcome, error = "attempts_exhausted", "max_attempts", "", "failure", ""
    state = _State()
    path = repo / task.module_file
    tracer.event("run_start", task=task.id, loop="simple", attempts=attempts, model=provider.describe(),
                 sampling=provider.sampling() if hasattr(provider, "sampling") else None)
    try:
        for n in range(1, attempts + 1):
            if r["input_tokens"] + r["output_tokens"] >= config.token_budget:
                stop, guardrail = "token_budget", "token_budget"
                break
            if time.monotonic() - started >= config.wall_seconds:
                stop, guardrail = "wall_clock", "wall_clock"
                break
            try:
                resp = provider.chat(SYSTEM, messages, None, config.max_tokens)
            except FatalProviderError as err:
                reason = err.reason(provider.describe())
                tracer.event("fatal_provider_error", status=err.status, reason=reason, detail=str(err)[:500])
                rc = {**r, "outcome": "not_run", "not_run_reason": reason, "fatal_status": err.status,
                      "stop_reason": "fatal_provider_error", "tokens": r["input_tokens"] + r["output_tokens"]}
                tracer.event("run_end", receipt=rc)
                tracer.close()
                return SimpleResult("", "not_run", rc, state, str(trace_path), {}, reason)
            except (ProviderError, OSError, ValueError, RuntimeError) as e:
                stop, guardrail, error = "provider_error", None, f"{type(e).__name__}: {e}"[:2000]
                tracer.event("llm_error", kind="loop", error=error)
                outcome = "aborted"
                break
            r["iterations"] += 1
            r["llm_calls"] += 1
            r["input_tokens"] += resp.input_tokens
            r["output_tokens"] += resp.output_tokens
            r["cache_read_tokens"] += resp.cache_read_tokens
            r["llm_latency_s"] += resp.latency_s
            code = extract(resp.text)
            reply = resp.text
            messages.append({"role": "assistant", "content": resp.text or ""})
            tracer.event("llm_call", kind="loop", iteration=n, input_tokens=resp.input_tokens,
                         output_tokens=resp.output_tokens, cache_read_tokens=resp.cache_read_tokens,
                         cache_write_tokens=resp.cache_write_tokens, latency_s=round(resp.latency_s, 3),
                         stop_reason=resp.stop_reason, tool_calls=["write_file"] if code else [],
                         tool_call_source="code_block" if code else None, text=preview(resp.text, 300),
                         context_full=getattr(resp, "context_full", False) or None)
            if not code:
                messages.append({"role": "user", "content": NO_CODE.format(module=task.module_file)})
                continue
            path.write_text(code if code.endswith("\n") else code + "\n", encoding="utf-8")
            state.files_changed.add(task.module_file)
            try:
                compile(code, task.module_file, "exec")
                syntax = ""
            except (SyntaxError, ValueError) as e:
                syntax = f"{task.module_file}:{getattr(e, 'lineno', '?')}: {getattr(e, 'msg', e)}"
            tracer.event("tool_call", name="write_file", args={"path": task.module_file, "content": code},
                         ok=True, output=f"overwrote {task.module_file}", syntax_error=syntax or None)
            t0 = time.perf_counter()
            ok, report = check()
            res = VerifyResult(ok, report, time.perf_counter() - t0, "harness")
            r["verify_rounds"] += 1
            on_verify(res)
            tracer.event("verify", round=n, source="harness", passed=ok, seconds=round(res.seconds, 2), tail=report[-3000:])
            if ok:
                stop, guardrail, outcome = "visible_pass", None, "success"
                break
            messages.append({"role": "user", "content": feedback(task, report)})
            tracer.event("feedback", round=n, prompt=messages[-1]["content"])
    finally:
        wall = round(time.monotonic() - started, 3)
    rc = {**r, "outcome": outcome, "stop_reason": stop, "guardrail": guardrail,
          "tokens": r["input_tokens"] + r["output_tokens"], "llm_latency_s": round(r["llm_latency_s"], 3),
          "wall_s": wall, "model": provider.describe(), "loop": "simple", "attempts": attempts,
          "sampling": provider.sampling() if hasattr(provider, "sampling") else None}
    if error:
        rc["error"] = error
    tracer.event("run_end", receipt=rc, reply=preview(reply, 2000))
    tracer.close()
    return SimpleResult(reply, outcome, rc, state, str(trace_path))
