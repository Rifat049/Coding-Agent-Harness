"""The offline mock drives the real tools and real verification, with no network."""
import json
import tempfile
import unittest
from pathlib import Path

from codebenchkit.agent.bench import materialize_task, public_task, visible_verifier
from codebenchkit.agent.consolidate import consolidate
from codebenchkit.agent.llm import make_provider
from codebenchkit.agent.loop import AgentConfig
from codebenchkit.agent.memory import Memory
from codebenchkit.agent.mock import MockCoder, fixture_source
from codebenchkit.agent.runner import run_agent
from codebenchkit.sandbox import run_tests
from codebenchkit.tasks import TASKS, TASKS_BY_NAME

TASK = TASKS_BY_NAME["parse_duration"]


class MockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def run_mock(self, spec, memory=None, name="parse_duration-s0", system_lesson=False):
        repo = Path(self.tmp.name) / name
        pub = public_task(TASK)
        prompt = materialize_task(pub, repo)
        cfg = AgentConfig(verify_cmd="python3 check_visible.py", benchmark=True)
        res = run_agent(prompt, repo, make_provider(spec), cfg, memory=memory,
                        small=make_provider("mock", "small") if memory else None,
                        verify_check=visible_verifier(pub, repo), scope="repo:bench",
                        trace_path=Path(self.tmp.name) / f"{name}.jsonl")
        calls = [c["name"] for m in res.history if m["role"] == "assistant" for c in m.get("tool_calls") or []]
        code = (repo / "solution.py").read_text()
        hidden = run_tests(code, TASK.name, TASK.hidden).all_passed(len(TASK.hidden))
        return res, calls, hidden

    def test_visible_bug_is_repaired_with_a_note(self):
        res, calls, hidden = self.run_mock("mock:visible")
        self.assertEqual(calls, ["list_files", "read_file", "write_file", "run_tests", "write_file",
                                 "run_tests", "note", "finish"])
        self.assertEqual(res.outcome, "success")
        self.assertTrue(hidden)
        self.assertEqual(len(res.state.notes), 1)

    def test_edge_bug_passes_visible_and_fails_hidden(self):
        res, calls, hidden = self.run_mock("mock:edge")
        self.assertEqual(calls, ["list_files", "read_file", "write_file", "run_tests", "finish"])
        self.assertEqual(res.outcome, "success")                 # visible checks pass...
        self.assertFalse(hidden)                                 # ...hidden tests catch the overfit
        self.assertEqual(res.state.notes, [])

    def test_retrieved_lesson_makes_first_write_correct(self):
        mem = Memory(Path(self.tmp.name) / "m.sqlite3")
        self.addCleanup(mem.close)
        mem.add_fact("parse_duration", "parse_duration: implement the full specification, including edge cases.",
                     "pitfall", "repo:bench")
        res, calls, hidden = self.run_mock("mock:edge", memory=mem)
        self.assertEqual(calls, ["list_files", "read_file", "write_file", "run_tests", "finish"])
        self.assertTrue(hidden)
        self.assertTrue(res.context["memory_ids"]["facts"])

    def test_seeded_modes_are_deterministic(self):
        coder = MockCoder("mock:0.4")
        system = "Workspace: /tmp/x/parse_duration-s3"
        self.assertEqual({coder._first_mode(system, "parse_duration") for _ in range(5)},
                         {coder._first_mode(system, "parse_duration")})
        modes = {MockCoder("mock:0.4")._first_mode(f"Workspace: /t/{t.name}-s{s}", t.name)
                 for t in TASKS for s in range(4)}
        self.assertEqual(modes, {"correct", "visible", "edge"})

    def test_tools_off_returns_text_only(self):
        r = MockCoder().chat("", [{"role": "user", "content": "x"}], [], tool_choice="none")
        self.assertEqual(r.tool_calls, [])
        self.assertTrue(r.text)

    def test_public_fixtures_pass_every_test(self):
        for t in TASKS:
            with self.subTest(task=t.name):
                tests = t.visible + t.hidden
                self.assertTrue(run_tests(fixture_source(t.name), t.name, tests).all_passed(len(tests)))

    def test_small_mock_gate_and_consolidation_pass_the_filters(self):
        small = make_provider("mock", "small")
        self.assertTrue(json.loads(small.complete("Task: fix the parser"))["retrieve"])
        mem = Memory(Path(self.tmp.name) / "c.sqlite3")
        self.addCleanup(mem.close)
        for i in range(3):
            mem.add_episode("repo:bench", f"task {i}", "success",
                            notes=["parse_duration: implement the full specification, including edge cases."])
        r = consolidate(mem, small, every_n=3)
        self.assertEqual((r["errors"], r["applied"]), (0, 1), r)
        for i in range(3):
            mem.add_episode("repo:bench", f"again {i}", "success",
                            notes=["parse_duration: implement the full specification, including edge cases."])
        r = consolidate(mem, small, every_n=3)
        self.assertEqual([o["op"] for o in r["applied_details"]], ["CONFIRM"])
        self.assertEqual(mem.facts()[0]["support"], 2)


if __name__ == "__main__":
    unittest.main()
