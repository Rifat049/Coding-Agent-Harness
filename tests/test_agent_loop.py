"""Loop guardrails with scripted providers in real temporary workspaces. No network."""
import json
import tempfile
import unittest
from pathlib import Path

from codebenchkit.agent.llm import LLMResponse, ProviderError, ScriptedProvider, ToolCall
from codebenchkit.agent.loop import AgentConfig
from codebenchkit.agent.memory import Memory
from codebenchkit.agent.runner import run_agent

GOOD = "def add(a, b):\n    return a + b\n"
BAD = "def add(a, b):\n    return a - b\n"
CHECK = "python3 -c 'from sol import add; assert add(2, 3) == 5, \"add(2, 3) != 5\"'"


def call(name, **args):
    return ToolCall(f"c{call.n}", name, args) if not setattr(call, "n", call.n + 1) else None


call.n = 0


def script(*steps):
    """Each step is an LLMResponse, a str, or a callable(messages) -> either. The last step repeats."""
    steps = list(steps)

    def fn(system, messages, tools, tool_choice):
        if tool_choice == "none":
            return LLMResponse("final answer from gathered evidence")
        step = steps.pop(0) if len(steps) > 1 else steps[0]
        return step(messages) if callable(step) else step
    return ScriptedProvider(fn)


def tools(*calls):
    return LLMResponse("", list(calls))


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        (self.repo / "sol.py").write_text(BAD)
        self.trace = Path(self.tmp.name) / "trace.jsonl"

    def run_(self, provider, **cfg):
        config = AgentConfig(**{"verify_cmd": CHECK, **cfg})
        return run_agent("make add() correct", self.repo, provider, config, trace_path=self.trace)

    def events(self, kind=None):
        evs = [json.loads(l) for l in self.trace.read_text().splitlines()]
        return [e for e in evs if kind is None or e["event"] == kind]

    def test_verify_gate_feeds_back_failure_then_accepts_fix(self):
        p = script(tools(call("write_file", path="sol.py", content=BAD), call("finish", summary="done, tests pass")),
                   tools(call("write_file", path="sol.py", content=GOOD), call("finish", summary="fixed")),
                   "unused")
        res = self.run_(p)
        self.assertEqual(res.outcome, "success")
        self.assertEqual(res.receipt["verify_rounds"], 2)
        feedback = [m for m in res.history if m["role"] == "user" and "FAILED (verify round 1" in m["content"]]
        self.assertEqual(len(feedback), 1)
        self.assertIn("add(2, 3) != 5", feedback[0]["content"])        # the test tail reaches the model
        self.assertEqual([e["passed"] for e in self.events("verify")], [False, True])

    def test_outcome_comes_from_verification_not_the_claim(self):
        p = script(tools(call("write_file", path="sol.py", content=BAD), call("finish", summary="All tests pass!")))
        res = self.run_(p, max_verify_rounds=2)
        self.assertEqual(res.outcome, "failure")
        self.assertEqual(res.receipt["verify_rounds"], 2)
        self.assertEqual(res.receipt["guardrail"], "max_verify_rounds")
        self.assertIn("add(2, 3) != 5", res.loop.failing_tail)

    def test_model_test_run_after_edit_is_trusted_without_rerun(self):
        p = script(tools(call("write_file", path="sol.py", content=GOOD)), tools(call("run_tests")),
                   tools(call("finish", summary="ok")))
        res = self.run_(p)
        self.assertEqual(res.outcome, "success")
        self.assertEqual(res.receipt["verify_rounds"], 0)               # nothing edited since a pass
        self.assertEqual(res.receipt["verify_runs"], 1)

    def test_no_verify_command_is_partial_and_no_tools_is_end_turn(self):
        res = run_agent("explain", self.repo, script("It adds numbers."), AgentConfig(), trace_path=self.trace)
        self.assertEqual((res.outcome, res.receipt["stop_reason"]), ("partial", "end_turn"))
        self.assertEqual(res.reply, "It adds numbers.")

    def test_doom_loop_nudges_at_three_and_aborts_at_five(self):
        p = script(tools(call("list_files")))
        res = self.run_(p)
        self.assertEqual(res.receipt["stop_reason"], "doom_loop")
        self.assertEqual(res.outcome, "aborted")
        nudges = [m for m in res.history if m["role"] == "user" and "Change approach" in m["content"]]
        self.assertEqual(len(nudges), 1)
        self.assertEqual(res.receipt["iterations"], 5)
        self.assertEqual(p.seen[-1]["tool_choice"], "none")             # one tools-off final answer
        self.assertEqual(res.reply, "final answer from gathered evidence")

    def test_error_streak_aborts_even_when_calls_differ(self):
        counter = iter(range(100))
        p = script(lambda m: tools(call("read_file", path=f"missing{next(counter)}.py")))
        res = self.run_(p, error_abort=6, error_nudge=3)
        self.assertEqual(res.receipt["stop_reason"], "error_streak")
        self.assertEqual(res.receipt["iterations"], 6)
        self.assertEqual(res.receipt["nudges"], 1)

    def test_iteration_limit_gets_one_tools_off_final_call_and_history_is_append_only(self):
        counter = iter(range(100))
        p = script(lambda m: tools(call("read_file", path="sol.py", start_line=1 + next(counter) % 2)))
        res = self.run_(p, max_iterations=4)
        self.assertEqual(res.receipt["stop_reason"], "max_iterations")
        self.assertEqual(p.calls, 5)
        final = p.seen[-1]
        self.assertEqual(final["tool_choice"], "none")
        self.assertIn("finish", final["tools"])                         # tools still sent (Anthropic)
        self.assertIn("step limit", final["messages"][-1]["content"])   # the note is a new message...
        systems = {json.dumps(s["system"]) for s in p.seen}
        self.assertEqual(len(systems), 1)                               # ...never a system edit
        for prev, nxt in zip(p.seen, p.seen[1:]):                       # every request extends the last
            self.assertEqual(nxt["messages"][:len(prev["messages"])], prev["messages"])
        self.assertEqual(res.reply, "final answer from gathered evidence")
        self.assertEqual(len(self.events("guardrail")), 1)

    def test_token_budget_pairs_unexecuted_calls_and_stops(self):
        big = LLMResponse("", [call("write_file", path="x.py", content="x = 1\n")], input_tokens=900, output_tokens=200)
        p = script(big)
        res = self.run_(p, token_budget=1000)
        self.assertEqual(res.receipt["stop_reason"], "token_budget")
        self.assertFalse((self.repo / "x.py").exists())                 # over budget: not executed
        tool_msgs = [m for m in res.history if m["role"] == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("not executed", tool_msgs[0]["content"])

    def test_provider_error_and_interrupt_are_aborted(self):
        def boom(system, messages, tools, tool_choice):
            raise ProviderError("HTTP 400: bad request", 400)
        res = self.run_(ScriptedProvider(boom))
        self.assertEqual((res.outcome, res.receipt["stop_reason"]), ("aborted", "provider_error"))
        self.assertIn("bad request", res.receipt["error"])

        def interrupt(system, messages, tools, tool_choice):
            raise KeyboardInterrupt
        res = self.run_(ScriptedProvider(interrupt))
        self.assertEqual((res.outcome, res.receipt["stop_reason"]), ("aborted", "interrupted"))
        self.assertEqual(self.events("run_end")[0]["receipt"]["outcome"], "aborted")

    def test_trim_cuts_seen_results_but_never_the_newest(self):
        (self.repo / "big.txt").write_text("\n".join("x" * 60 for _ in range(300)))
        p = script(tools(call("read_file", path="big.txt")), tools(call("read_file", path="sol.py")),
                   tools(call("write_file", path="sol.py", content=GOOD)), tools(call("finish", summary="ok")))
        res = self.run_(p)
        second = p.seen[1]["messages"]
        self.assertGreater(len(second[-1]["content"]), 4000)            # newest result: whole
        third = p.seen[2]["messages"]
        trimmed = [m for m in third if m["role"] == "tool" and "Trimmed after observation" in m["content"]]
        self.assertEqual(len(trimmed), 1)
        self.assertEqual(res.receipt["trims"], 1)
        self.assertGreater(len(self.events("trim")), 0)

    def test_compaction_keeps_pairs_strips_thinking_and_logs_dropped_blocks(self):
        (self.repo / "big.txt").write_text("y" * 3000)
        n = iter(range(100))

        def step(messages):
            i = next(n)
            if i >= 8:
                return tools(call("write_file", path="sol.py", content=GOOD), call("finish", summary="ok"))
            return LLMResponse("", [call("read_file", path="big.txt", start_line=1 + i % 2)],
                               raw={"provider": "anthropic", "thinking": [{"type": "thinking", "signature": f"s{i}"}]},
                               transformations=[{"type": "thinking_dropped", "path": "messages.1.content.0",
                                                 "reason": "prefix_binding_mismatch"}] if i == 5 else [])
        res = self.run_(script(step), context_limit=3000)
        self.assertEqual(res.outcome, "success")
        self.assertGreater(res.receipt["compactions"], 0)
        comp = self.events("compaction")[0]
        self.assertTrue(comp["compacted"])
        self.assertGreater(comp["thinking_blocks_stripped"], 0)
        calls = {c["id"] for m in res.history for c in m.get("tool_calls", [])}
        results = {m["tool_call_id"] for m in res.history if m["role"] == "tool"}
        self.assertEqual(calls, results)                                # no call split from its result
        self.assertEqual(res.history[0]["content"], "make add() correct")
        self.assertEqual(res.receipt["thinking_dropped"], 1)
        self.assertEqual(len(self.events("thinking_dropped")), 1)

    def test_episode_is_deterministic_and_written_from_harness_observations(self):
        mem = Memory(Path(self.tmp.name) / "m.sqlite3")
        self.addCleanup(mem.close)
        p = script(tools(call("write_file", path="sol.py", content=BAD), call("note", lesson="add() must sum, not subtract"),
                         call("finish", summary="Everything passes, trust me")))
        res = run_agent("fix add", self.repo, p, AgentConfig(verify_cmd=CHECK, max_verify_rounds=1),
                        memory=mem, trace_path=self.trace)
        ep = mem.episodes()[0]
        self.assertEqual((ep["outcome"], ep["scope"]), ("failure", "repo:repo"))
        self.assertIn("sol.py", ep["files"])
        self.assertIn("add(2, 3) != 5", ep["errors"])
        self.assertEqual(ep["notes"], "add() must sum, not subtract")
        self.assertNotIn("trust me", json.dumps(ep))                    # the model's claim is not stored
        self.assertEqual(res.episode_id, ep["id"])

    def test_run_start_reports_context_sizes(self):
        self.run_(script("nothing to do"))
        start = self.events("run_start")[0]
        self.assertIn("sections", start["context"])
        self.assertIn("rules", start["context"]["sections"])
        self.assertGreater(start["context"]["total_chars"], 100)
        end = self.events("run_end")[0]["receipt"]
        for key in ("iterations", "tokens", "files", "outcome", "verify_rounds"):
            self.assertIn(key, end)


if __name__ == "__main__":
    unittest.main()
