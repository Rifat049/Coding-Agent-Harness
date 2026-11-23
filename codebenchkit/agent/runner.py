"""One agent run, end to end: context -> loop -> deterministic episode -> maybe consolidate.

The episode is written by the harness from what the tools and the verifier observed, with no
LLM involved. The model's own finish summary is a claim, so it is not stored as fact. Only
consolidation (a separate, filtered step) turns episodes into durable facts.
"""
from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .context import Context, build_context
from .llm import FatalProviderError
from .loop import AgentConfig, AgentLoop, LoopResult
from .tools import Policy, RunState, VerifyResult, Verifier, Workspace, build_tools
from .trace import Tracer


@dataclass
class AgentResult:
    reply: str
    outcome: str
    receipt: dict
    history: list[dict]
    state: RunState
    trace_path: str
    context: dict = field(default_factory=dict)
    episode_id: int | None = None
    consolidation: dict | None = None
    loop: LoopResult | None = None
    # Set when a billing/auth error (HTTP 401/402/403) stopped the run. stage "run": the task did
    # not complete and must not be scored (outcome "not_run"). stage "consolidation": the task
    # completed and is scored, but the small model failed afterwards (the batch stays pending).
    fatal: str | None = None
    fatal_stage: str | None = None


def run_agent(task: str, repo, provider, config: AgentConfig | None = None, memory=None, small=None,
              history: list[dict] | None = None, trace_path=None, approve=None,
              verify_check: Callable[[], tuple[bool, str]] | None = None,
              on_verify: Callable[[VerifyResult], None] | None = None,
              scope: str | None = None, context: Context | None = None, hidden_strings=(),
              user_preamble: str = "") -> AgentResult:
    """Run one task. `context`, when given, is a system prompt built earlier (chat sessions keep
    one per session so its cache and thinking blocks stay valid across turns). `user_preamble` is
    prepended to the task message (chat puts per-turn memory there instead of in the system)."""
    config = config or AgentConfig()
    ws = Workspace(repo)
    scope = scope or "repo:" + ws.root.name
    state = RunState()
    started = time.monotonic()
    path = Path(trace_path) if trace_path else Path.home() / ".cbk" / "traces" / f"{uuid.uuid4().hex}.jsonl"
    tracer = Tracer(path)
    holder: dict = {}                      # the AgentLoop, once built, so a fatal error can report its tokens
    messages = copy.deepcopy(history or []) + [{"role": "user", "content": (user_preamble + task) if user_preamble else task}]
    try:
        try:
            return _run(task, ws, provider, config, memory, small, messages, tracer, path, state, started,
                        approve, verify_check, on_verify, scope, context, hidden_strings,
                        holder)
        except FatalProviderError as err:
            reason = err.reason(provider.describe())
            r = holder["loop"].r if "loop" in holder else LoopResult()
            receipt = {"outcome": "not_run", "not_run_reason": reason, "fatal_status": err.status,
                       "stop_reason": "fatal_provider_error", "iterations": r.iterations, "llm_calls": r.llm_calls,
                       "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                       "cache_read_tokens": r.cache_read_tokens, "tokens": r.tokens,
                       "llm_latency_s": round(r.latency_s, 3), "wall_s": round(time.monotonic() - started, 3),
                       "model": provider.describe(), "small_model": small.describe() if small is not None else None}
            tracer.event("fatal_provider_error", status=err.status, reason=reason, detail=str(err)[:500])
            tracer.event("run_end", receipt=receipt, reply="")
            # No episode: an aborted-by-billing run says nothing about the task.
            return AgentResult(f"Run not completed: {reason}.", "not_run", receipt, messages, state, str(path),
                               context.report if context is not None else {}, None, None, r, reason, "run")
    finally:
        tracer.close()


def _run(task, ws, provider, config, memory, small, messages, tracer, path, state, started, approve, verify_check,
         on_verify, scope, context, hidden_strings, holder) -> AgentResult:
    """The run itself. A FatalProviderError escapes to run_agent; nothing here catches it."""
    if context is None:
        context = build_context(ws, task, provider.describe(), config.verify_cmd, memory,
                                config.gate_mode, small, scope=scope)
    policy = Policy(config.approval, approve, extra_deny=list(config.command_extra_deny))
    verifier = Verifier(ws, config.verify_cmd, config.verify_timeout, on_result=on_verify, check=verify_check)
    registry = build_tools(ws, policy, state, verifier, memory, scope, config.command_timeout,
                           benchmark=config.benchmark and not config.benchmark_commands)
    before = ws.fingerprint()
    tracer.event("run_start", task=task[:4000], repo=str(ws.root), scope=scope, model=provider.describe(),
                 small=small.describe() if small is not None else None, tools=registry.names(),
                 config={k: v for k, v in vars(config).items()}, context=context.report)
    loop = AgentLoop(provider, registry, state, verifier, config, tracer, small)
    holder["loop"] = loop
    res = loop.run(context.system, messages)
    after = ws.fingerprint()
    state.files_changed.update(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    files = sorted(f for f in state.files_changed if not f.startswith(".cbk/"))
    receipt = {
        "outcome": res.outcome, "stop_reason": res.stop_reason, "guardrail": res.guardrail,
        "iterations": res.iterations, "llm_calls": res.llm_calls,
        "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
        "cache_read_tokens": res.cache_read_tokens, "tokens": res.tokens,
        "llm_latency_s": res.latency_s, "verify_rounds": res.verify_rounds,
        "verify_runs": len(state.verify_results), "nudges": res.nudges, "trims": res.trims,
        "compactions": res.compactions, "thinking_dropped": res.thinking_dropped,
        "thinking_blocks": res.thinking_blocks, "reasoning_chars": res.reasoning_chars,
        "files": files, "notes": len(state.notes), "wall_s": round(time.monotonic() - started, 3),
        "model": provider.describe(), "small_model": small.describe() if small is not None else None,
        "small_model_note": config.small_model_note or None,
        "sampling": provider.sampling() if hasattr(provider, "sampling") else None,
    }
    if res.error:
        receipt["error"] = res.error
    episode_id, consolidation, fatal = None, None, None
    if memory is not None:
        episode_id = memory.add_episode(
            scope, task, res.outcome, summary=_episode_summary(res, files, config),
            files=files, errors=res.failing_tail if res.outcome != "success" else "",
            notes=state.notes, stats=receipt)
        tracer.event("episode", id=episode_id, outcome=res.outcome, notes=state.notes)
        if small is not None and config.consolidation_every > 0:
            from .consolidate import consolidate
            try:
                consolidation = consolidate(memory, small, every_n=config.consolidation_every,
                                            hidden_strings=hidden_strings)
            except FatalProviderError as err:
                # The task itself completed and is scored; the small model failed afterwards.
                fatal = err.reason(small.describe())
                receipt["fatal_after"] = f"{fatal} during consolidation; episode {episode_id} stays pending"
                tracer.event("fatal_provider_error", status=err.status, reason=fatal, stage="consolidation",
                             detail=str(err)[:500])
                consolidation = {"applied": 0, "rejected": 0, "errors": 1, "episode_ids": [],
                                 "error_details": [str(err)[:500]], "rejected_details": []}
            if consolidation.get("episode_ids") or (consolidation.get("errors") and not fatal):
                tracer.event("consolidation", **{k: consolidation[k] for k in
                                                 ("applied", "rejected", "errors", "episode_ids")},
                             rejected_details=consolidation.get("rejected_details"),
                             error_details=consolidation.get("error_details"))
    tracer.event("run_end", receipt=receipt, reply=res.reply[:2000])
    return AgentResult(res.reply, res.outcome, receipt, messages, state, str(path), context.report,
                   episode_id, consolidation, res, fatal, "consolidation" if fatal else None)


def _episode_summary(res: LoopResult, files: list[str], config: AgentConfig) -> str:
    """Deterministic, harness-observed facts only."""
    parts = [f"Outcome {res.outcome} (stop: {res.stop_reason}"
             + (f", guardrail: {res.guardrail}" if res.guardrail else "") + ")",
             f"{res.iterations} model calls, {res.verify_rounds} harness verify rounds"]
    if config.verify_cmd:
        parts.append(f"verify command: {config.verify_cmd}")
    parts.append("files changed: " + (", ".join(files[:12]) if files else "none"))
    return "; ".join(parts) + "."
