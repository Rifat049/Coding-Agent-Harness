"""Multi-turn REPL over one workspace.

Each user message is one agent run (with its own episode and trace), and history carries
across turns. Two choices keep a long session cheap and valid:
  * The system prompt is built once per session. Memory is gated per message, but it goes into
    that message (a preamble), never into the system prompt. Editing the system prompt mid-session
    would miss the prompt cache and invalidate every earlier thinking block.
  * Once earlier turns pass `chat_budget` of the context limit, they are compacted into one
    summary. The newest turn is kept verbatim, and no tool call is split from its result.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .context import build_context
from .history import compact_history, estimate_tokens
from .llm import system_text
from .loop import AgentConfig
from .memory import gate
from .runner import AgentResult, run_agent
from .tools import Workspace

HELP = "Commands: /exit  /reset (clear history)  /memory (stats)  /help"


class ChatSession:
    def __init__(self, repo, provider, config: AgentConfig, memory=None, small=None, trace_dir=None,
                 approve=None, chat_budget: float = 0.35):
        self.ws = Workspace(repo)
        self.provider, self.config, self.memory, self.small = provider, config, memory, small
        self.trace_dir = Path(trace_dir) if trace_dir else Path.home() / ".cbk" / "traces"
        self.approve, self.chat_budget = approve, chat_budget
        self.scope = "repo:" + self.ws.root.name
        # Frozen for the session; memory=None here on purpose (see module docstring).
        self.context = build_context(self.ws, "", provider.describe(), config.verify_cmd, None,
                                     "never", None, scope=self.scope)
        self.history: list[dict] = []
        self.turns = 0

    def preamble(self, text: str) -> str:
        if self.memory is None:
            return ""
        retrieve, query = gate(text, self.config.gate_mode, self.small)
        if not retrieve:
            return ""
        block = self.memory.render(self.memory.retrieve(query, self.scope))
        return f"[Memory for this message; verify before relying on it]\n{block}\n\n[Message]\n" if block else ""

    def send(self, text: str) -> AgentResult:
        self.turns += 1
        res = run_agent(text, self.ws.root, self.provider, self.config, memory=self.memory, small=self.small,
                        history=self.history, approve=self.approve, scope=self.scope, context=self.context,
                        user_preamble=self.preamble(text),
                        trace_path=self.trace_dir / f"chat-{time.strftime('%Y%m%d-%H%M%S')}-t{self.turns}.jsonl")
        self.history = res.history
        self._compact_earlier_turns()
        return res

    def _compact_earlier_turns(self) -> dict | None:
        sys_text = system_text(self.context.system)
        if estimate_tokens(self.history, sys_text) <= self.chat_budget * self.config.context_limit:
            return None
        # The newest turn starts at the last plain user message (the task, not a harness note).
        starts = [i for i, m in enumerate(self.history)
                  if m["role"] == "user" and not str(m["content"]).startswith("[harness]")]
        last = starts[-1] if starts else len(self.history)
        summarize = (lambda p: self.small.complete(p, max_tokens=2000)) if self.small is not None else None
        return compact_history(self.history, self.config.context_limit, sys_text, observed_until=last,
                               summarize=summarize, threshold=0.0)

    def reset(self) -> None:
        self.history = []


def repl(session: ChatSession, input_fn=input, print_fn=print) -> int:
    print_fn(f"cbk chat in {session.ws.root} with {session.provider.describe()}. {HELP}")
    while True:
        try:
            text = input_fn("cbk> ").strip()
        except (EOFError, KeyboardInterrupt):
            print_fn("")
            return 0
        if not text:
            continue
        if text in ("/exit", "/quit"):
            return 0
        if text == "/help":
            print_fn(HELP)
            continue
        if text == "/reset":
            session.reset()
            print_fn("history cleared")
            continue
        if text == "/memory":
            print_fn(json.dumps(session.memory.stats() if session.memory else {"memory": "off"}))
            continue
        res = session.send(text)
        print_fn(res.reply)
        r = res.receipt
        print_fn(f"[{r['outcome']} | {r['iterations']} calls | {r['tokens']:,} tokens | "
                 f"files: {', '.join(r['files']) or 'none'} | trace: {res.trace_path}]")
