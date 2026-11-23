"""The agent loop: reason -> act -> observe, under guardrails, with a harness-owned verdict.

    while True:
        stop if a limit tripped           -> one tools-off final answer (nothing gathered is wasted)
        trim seen tool output; compact the middle of history if the context is ~70% full
        response = model(system, history, tools)
        no tool calls / finish called     -> stop hook: the harness runs verify itself;
                                             on failure, feed back the tail and continue
        run each tool; append each result right after its call
        same call + same result 3x        -> nudge; 5x -> abort.  Error streaks likewise.

Three ideas carried over from waku-agent: a loop with two natural exits, a tools-off final call
when the iteration limit hits, and cutting tool results the model has already read.

What is new is the stop hook. A coding agent's "done" is a claim, so the outcome comes from the
harness's own verification, never from the model.

History is append-only apart from the two hygiene steps. Nudges, verify feedback and the limit
note are new user messages at the end. The system prompt is never edited mid-run, so the prompt
cache stays warm and Claude's conversation-bound thinking blocks stay valid. The two hygiene
steps (trim, compaction) do edit history. The Anthropic adapter opts into drop_block for that,
and the API's `input_transformations` (thinking blocks it dropped) are logged here.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

from .history import compact_history, estimate_tokens, shrink_seen
from .llm import FatalProviderError, ProviderError, system_text
from .tools import RunState, ToolRegistry, Verifier, args_signature
from .trace import Tracer, preview

LIMIT_NOTE = ("[harness] {why} You cannot call any more tools. Answer now from what you already "
              "gathered: say what you changed, what was verified, and plainly what is missing or failed.")
LIMIT_REPLY = "(The run stopped at a guardrail before the model produced a final answer.)"
DOOM_NUDGE = ("[harness] You have made the same tool call with the same result {n} times in a row. "
              "Repeating it will not help. Change approach: read a different file, re-read the error, "
              "or try a different fix.")
ERROR_NUDGE = ("[harness] Your last {n} tool calls all failed. Stop and re-read the error messages, "
               "check the tool's arguments, and change approach.")
VERIFY_FEEDBACK = ("[harness] You stopped, so the harness ran the test command `{cmd}` itself, and it "
                   "FAILED (verify round {k} of {n}). Output tail:\n{tail}\n\nFix the cause, run "
                   "run_tests, then call finish.")

ABORTED = {"max_iterations", "token_budget", "wall_clock", "doom_loop", "error_streak",
           "provider_error", "interrupted"}


@dataclass
class AgentConfig:
    """Every budget for one run. agent-bench keeps these identical across conditions."""
    max_iterations: int = 30
    max_tokens: int = 16000            # output cap per model call
    token_budget: int = 600_000        # input + output, summed over the run
    wall_seconds: float = 900
    max_verify_rounds: int = 3
    context_limit: int = 160_000       # tokens; compaction starts at compact_at * this
    compact_at: float = 0.7
    trim_chars: int = 4000             # seen tool results longer than this are cut...
    trim_keep: int = 2000              # ...to this many characters plus a note
    doom_nudge: int = 3
    doom_abort: int = 5
    error_nudge: int = 4
    error_abort: int = 8
    final_max_tokens: int = 4000
    command_timeout: int = 120
    verify_timeout: int = 300
    verify_cmd: str | None = None
    approval: str = "auto"
    gate_mode: str = "heuristic"
    benchmark: bool = False
    consolidation_every: int = 3
    summarize_compaction: bool = True  # use the small model for compaction summaries when present
    force_compaction_after: int = 0    # test knob: compact once after this many model calls (0 = off)
    benchmark_commands: bool = False   # benchmark mode but keep run_command (bugfix trap task only)
    command_extra_deny: tuple = ()     # extra regexes for Policy (e.g. workspace confinement)
    small_model_note: str = ""         # how the small model was chosen (e.g. by --small-model auto); in the receipt
    # Run the stop-hook check even when no file changed. Off by default (a read-only question must
    # be able to end). utility-bench turns it on: every task needs an edit, and without it a model
    # that batches [failed edit, run_tests, finish] in one turn ends after 1 of its 20 turns with no
    # feedback (seen in 6 of 10 qwen2.5-coder calibration items).
    verify_without_edits: bool = False


@dataclass
class LoopResult:
    reply: str = ""
    stop_reason: str = ""
    outcome: str = "partial"
    guardrail: str | None = None
    iterations: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    latency_s: float = 0.0
    verify_rounds: int = 0
    nudges: int = 0
    trims: int = 0
    compactions: int = 0
    thinking_dropped: int = 0
    thinking_blocks: int = 0           # Claude thinking blocks returned, summed over the run
    reasoning_chars: int = 0           # DeepSeek reasoning_content characters, summed
    error: str = ""
    failing_tail: str = ""
    stats: dict = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class AgentLoop:
    def __init__(self, provider, registry: ToolRegistry, state: RunState, verifier: Verifier | None,
                 config: AgentConfig, tracer: Tracer | None = None, small=None, clock=time.monotonic):
        self.provider, self.registry, self.state, self.verifier = provider, registry, state, verifier
        self.config, self.tracer, self.small, self.clock = config, tracer or Tracer(), small, clock
        self.r = LoopResult()
        self._forced_compaction = False

    # ------------------------------------------------------------------ public
    def run(self, system, messages: list[dict]) -> LoopResult:
        r, cfg = self.r, self.config
        self._t0 = self.clock()
        self._last_input = 0
        self._sig, self._sig_n, self._err_n = None, 0, 0
        try:
            while True:
                trip = self._limit_tripped()
                if trip:
                    self._final_answer(system, messages, trip)
                    break
                if r.llm_calls:
                    self._hygiene(system, messages)
                resp = self._call(system, messages, "loop")
                if resp is None:
                    break
                r.iterations += 1
                messages.append(resp.as_message())
                if resp.text:
                    r.reply = resp.text
                if not resp.tool_calls:
                    if self._stop_hook(messages):
                        continue
                    r.stop_reason = "finished" if self.state.finished else "end_turn"
                    break
                abort = self._run_tools(resp.tool_calls, messages)
                if abort:
                    self._final_answer(system, messages, abort)
                    break
                if self.state.finished:
                    r.reply = self.state.finish_summary or r.reply
                    if self._stop_hook(messages):
                        self.state.finished = False
                        continue
                    r.stop_reason = "finished"
                    break
        except KeyboardInterrupt:
            r.stop_reason, r.guardrail = "interrupted", "interrupted"
            self.tracer.event("guardrail", kind="interrupted")
        self._final_outcome()
        r.latency_s = round(r.latency_s, 3)
        r.stats = {"wall_s": round(self.clock() - self._t0, 3)}
        return r

    # ------------------------------------------------------------------ model calls
    def _call(self, system, messages, kind: str, tool_choice: str = "auto", max_tokens: int | None = None):
        r = self.r
        try:
            resp = self.provider.chat(system, messages, self.registry.schemas(),
                                      max_tokens or self.config.max_tokens, tool_choice=tool_choice)
        except FatalProviderError:
            raise                                   # billing/auth: the runner stops the whole run
        except (ProviderError, OSError, ValueError, RuntimeError) as e:
            r.stop_reason, r.error = "provider_error", f"{type(e).__name__}: {e}"[:2000]
            self.tracer.event("llm_error", kind=kind, error=r.error)
            return None
        r.llm_calls += 1
        r.input_tokens += resp.input_tokens
        r.output_tokens += resp.output_tokens
        r.cache_read_tokens += resp.cache_read_tokens
        r.latency_s += resp.latency_s
        self._last_input = resp.input_tokens
        dropped = [t for t in resp.transformations if t.get("type") == "thinking_dropped"]
        r.thinking_dropped += len(dropped)
        th = resp.thinking_stats()
        r.thinking_blocks += th["thinking_blocks"]
        r.reasoning_chars += th["reasoning_chars"]
        status = getattr(resp, "transformations_status", "absent")
        self.tracer.event("llm_call", kind=kind, iteration=r.iterations + (kind == "loop"),
                          input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                          cache_read_tokens=resp.cache_read_tokens, cache_write_tokens=resp.cache_write_tokens,
                          latency_s=round(resp.latency_s, 3), stop_reason=resp.stop_reason,
                          tool_calls=[c.name for c in resp.tool_calls], text=preview(resp.text, 300),
                          tool_call_source=getattr(resp, "tool_call_source", "") or None,
                          text_tool_error=getattr(resp, "text_tool_error", "") or None,
                          context_full=getattr(resp, "context_full", False) or None,
                          raw_text=preview(getattr(resp, "raw_text", ""), 1500) or None,
                          # "absent" | "empty" | the list itself: never collapsed to null.
                          input_transformations=resp.transformations if status == "list" else status,
                          **th, retried_without_thinking=resp.retried_without_thinking or None)
        if dropped:
            self.tracer.event("thinking_dropped", count=len(dropped), entries=dropped)
        return resp

    def _final_answer(self, system, messages, why: str) -> None:
        """One tools-off call so a guardrail does not throw away what the run gathered. Tools are
        still sent (Anthropic rejects tool_use history without them); tool_choice=none stops calls."""
        r = self.r
        r.guardrail = why
        self.tracer.event("guardrail", kind=why, iterations=r.iterations, tokens=r.tokens)
        reasons = {"max_iterations": f"You reached the step limit ({self.config.max_iterations} model calls).",
                   "token_budget": f"The run used its token budget ({self.config.token_budget:,} tokens).",
                   "wall_clock": f"The run hit its time limit ({self.config.wall_seconds:.0f}s).",
                   "doom_loop": "The run was stopped because the same tool call kept repeating.",
                   "error_streak": "The run was stopped after too many consecutive tool errors."}
        messages.append({"role": "user", "content": LIMIT_NOTE.format(why=reasons.get(why, "A guardrail tripped."))})
        resp = self._call(system, messages, "final", tool_choice="none", max_tokens=self.config.final_max_tokens)
        r.stop_reason = why                        # a failed final call does not change why we stopped
        if resp is None:
            r.reply = r.reply or LIMIT_REPLY
            return
        # Some servers ignore tool_choice=none; a call without a result would corrupt history.
        final = {"role": "assistant", "content": resp.text or "(no answer)"}
        if resp.raw:
            # Keep provider state (DeepSeek reasoning_content must be echoed on every later request
            # that carries tools, or the API returns 400; a chat can continue after a guardrail).
            final["raw"] = resp.raw
        messages.append(final)
        r.reply = resp.text or r.reply or LIMIT_REPLY

    def _limit_tripped(self) -> str | None:
        cfg, r = self.config, self.r
        if r.iterations >= cfg.max_iterations:
            return "max_iterations"
        if r.tokens >= cfg.token_budget:
            return "token_budget"
        if self.clock() - self._t0 >= cfg.wall_seconds:
            return "wall_clock"
        return None

    # ------------------------------------------------------------------ tools
    def _run_tools(self, calls, messages) -> str | None:
        """Execute calls in order, each result appended right after the call that asked for it.
        Returns a guardrail name if the run must abort."""
        cfg, r = self.config, self.r
        nudge, abort = None, None
        over = self._limit_tripped() in ("token_budget", "wall_clock")
        for call in calls:
            if over or abort:
                # Keep every call paired with a result, even ones we refuse to run.
                text, ok = "ERROR: not executed: the run stopped at a guardrail.", False
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                                 "content": text, "is_error": True})
                continue
            t0 = time.perf_counter()
            self.state.last_syntax_error = ""
            text, ok = self.registry.execute(call.name, call.args)
            dur = time.perf_counter() - t0
            messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                             "content": text, "is_error": not ok})
            self.tracer.event("tool_call", name=call.name, args=_short_args(call.args), ok=ok,
                              duration_ms=int(dur * 1000), chars=len(text), output=preview(text, 600),
                              syntax_error=getattr(self.state, "last_syntax_error", "") or None)
            sig = args_signature(call.name, call.args) + "#" + hashlib.sha1(text.encode()).hexdigest()[:12]
            self._sig_n = self._sig_n + 1 if sig == self._sig else 1
            self._sig = sig
            self._err_n = 0 if ok else self._err_n + 1
            if self._sig_n >= cfg.doom_abort:
                abort = "doom_loop"
            elif self._err_n >= cfg.error_abort:
                abort = "error_streak"
            elif self._sig_n == cfg.doom_nudge:
                nudge = DOOM_NUDGE.format(n=self._sig_n)
            elif self._err_n == cfg.error_nudge:
                nudge = ERROR_NUDGE.format(n=self._err_n)
        if over and not abort:
            abort = self._limit_tripped()
        if nudge and not abort:
            r.nudges += 1
            self.tracer.event("nudge", text=nudge, repeat=self._sig_n, error_streak=self._err_n)
            messages.append({"role": "user", "content": nudge})
        return abort

    # ------------------------------------------------------------------ verification
    def _verifiable(self) -> bool:
        return self.verifier is not None and bool(self.verifier.cmd or self.verifier.check)

    def _stop_hook(self, messages) -> bool:
        """The model wants to stop. If it edited files since the last passing check, the harness
        verifies. Returns True if the loop should continue (failure fed back)."""
        st, cfg, r = self.state, self.config, self.r
        if not self._verifiable() or (not st.files_changed and not cfg.verify_without_edits):
            return False
        last = st.last_verify
        if last is not None and last.passed and st.edits_since_verify == 0:
            return False
        if r.verify_rounds >= cfg.max_verify_rounds:
            return False
        res = self.verifier.run(st, "harness")
        r.verify_rounds += 1
        self.tracer.event("verify", round=r.verify_rounds, source="harness", passed=res.passed,
                          seconds=round(res.seconds, 2), tail=res.tail[-3000:])
        if res.passed:
            return False
        if r.verify_rounds >= cfg.max_verify_rounds:
            r.guardrail = "max_verify_rounds"
            self.tracer.event("guardrail", kind="max_verify_rounds", rounds=r.verify_rounds)
            return False
        if self._limit_tripped():
            return False
        messages.append({"role": "user", "content": VERIFY_FEEDBACK.format(
            cmd=self.verifier.cmd or "visible checks", k=r.verify_rounds, n=cfg.max_verify_rounds,
            tail=res.tail[-4000:])})
        return True

    def _final_outcome(self) -> None:
        """success / failure from the harness's own check of the final state; never the model's word."""
        st, r = self.state, self.r
        if self._verifiable():
            last = st.last_verify
            if last is None or st.edits_since_verify > 0:
                last = self.verifier.run(st, "final")
                self.tracer.event("verify", round=None, source="final", passed=last.passed,
                                  seconds=round(last.seconds, 2), tail=last.tail[-3000:])
            if last.passed:
                r.outcome = "success"
            else:
                r.outcome = "aborted" if r.stop_reason in ABORTED else "failure"
                r.failing_tail = last.tail[-2000:]
        else:
            r.outcome = "aborted" if r.stop_reason in ABORTED else "partial"

    # ------------------------------------------------------------------ context hygiene
    def _hygiene(self, system, messages) -> None:
        cfg = self.config
        n = shrink_seen(messages, threshold=cfg.trim_chars, keep=cfg.trim_keep)
        if n:
            self.r.trims += n
            self.tracer.event("trim", results=n)
        sys_text = system_text(system)
        est = max(estimate_tokens(messages, sys_text), self._last_input)
        forced = (cfg.force_compaction_after > 0 and not self._forced_compaction
                  and self.r.llm_calls >= cfg.force_compaction_after)
        if est <= cfg.compact_at * cfg.context_limit and not forced:
            return
        summarize = None
        if self.small is not None and cfg.summarize_compaction:
            summarize = lambda prompt: self.small.complete(prompt, max_tokens=2000)  # noqa: E731
        # The forced test knob bypasses the "must be smaller" guard; real compaction never does.
        rep = compact_history(messages, cfg.context_limit, sys_text, summarize=summarize, threshold=0.0,
                              require_smaller=not forced)
        self.tracer.event("compaction", estimate_tokens=est, forced=forced or None, **rep)
        if rep.get("compacted"):
            self._forced_compaction = self._forced_compaction or forced
            self.r.compactions += 1
            self._last_input = 0


def _short_args(args) -> dict:
    if not isinstance(args, dict):
        return {"_": preview(repr(args), 200)}
    return {k: (preview(v, 300) if isinstance(v, str) else v) for k, v in args.items()}
