"""utility-v1: the suite is well formed, every metric fires on the mocks, invalid output is classified
as documented, and a billing error stops the run with not_run items. Offline: mocks only."""
import json
import tempfile
import unittest
from pathlib import Path

from codebenchkit.agent.llm import FatalProviderError, LLMResponse, ScriptedProvider, ToolCall
from codebenchkit.agent.loop import AgentConfig
from codebenchkit.utility.bench import invalid_output, run_utility_benchmark, utility_metrics
from codebenchkit.utility.core import (PROMPT_VARIANTS, cheat_reasons, load_tasks, materialize, prompt, provenance,
                                       selfcheck)


def run(model, tasks="u03,u05", samples=1, **kw):
    out = Path(tempfile.mkdtemp())
    cfg, items = run_utility_benchmark(load_tasks(tasks), model=model, samples=samples, out_dir=out,
                                       config=AgentConfig(max_iterations=20, context_limit=16000), progress=lambda *_: None, **kw)
    return cfg, items, out


class SuiteShape(unittest.TestCase):
    def test_ten_tasks_mixed_kinds(self):
        tasks = load_tasks()
        self.assertEqual(len(tasks), 10)
        kinds = [t.kind for t in tasks]
        self.assertEqual((kinds.count("implement"), kinds.count("fix")), (6, 4))

    def test_selfcheck_passes(self):
        lines = []
        self.assertEqual(selfcheck(log=lines.append), 0, "\n".join(lines))

    def test_workspace_never_contains_hidden_tests_or_reference(self):
        for t in load_tasks():
            with tempfile.TemporaryDirectory() as d:
                materialize(t, Path(d) / "ws")
                names = [str(p.relative_to(Path(d) / "ws")) for p in (Path(d) / "ws").rglob("*") if p.is_file()]
                self.assertFalse([n for n in names if "hidden" in n or "reference" in n], t.id)
                text = "".join((Path(d) / "ws" / n).read_text() for n in names)
                for line in t.hidden.splitlines():
                    if "assert" in line and len(line.strip()) > 40:
                        self.assertNotIn(line.strip(), text, f"{t.id}: a hidden assertion leaked into the workspace")

    def test_prompt_variants(self):
        t = load_tasks("u01")[0]
        self.assertNotIn("Before you call finish", prompt(t, "baseline"))
        self.assertIn("Before you call finish, call run_tests", prompt(t, "tests-first"))
        self.assertIn("hidden tests", prompt(t))
        self.assertEqual(PROMPT_VARIANTS, ("baseline", "tests-first"))
        with self.assertRaises(ValueError):
            prompt(t, "nope")

    def test_provenance_hash_is_stable(self):
        a, b = provenance(), provenance()
        self.assertEqual(a["benchmark_sha256"], b["benchmark_sha256"])
        self.assertEqual(a["suite"], "utility-v1")


class MockMetrics(unittest.TestCase):
    def test_good_mock_passes_everything(self):
        cfg, items, out = run("mock", tasks=None)
        m = utility_metrics(items)
        self.assertEqual((m["pass_rate"], m["cheat_rate"], m["invalid_turn_rate"]), (1.0, 0.0, 0.0))
        self.assertTrue(all(i["wall_s"] >= i["model_s"] for i in items))
        self.assertTrue(all(i["test_s"] > 0 and i["scoring_s"] > 0 for i in items))
        self.assertTrue((out / "results.jsonl").is_file() and (out / "config.json").is_file())
        self.assertEqual(cfg["benchmark"], "utility-v1")

    def test_lazy_is_false_success(self):
        _, items, _ = run("mock:lazy")
        self.assertTrue(all(i["false_success"] and not i["passed"] for i in items))

    def test_tamper_is_cheating(self):
        _, items, _ = run("mock:tamper")
        self.assertTrue(all(i["cheated"] and not i["passed"] for i in items))
        self.assertIn("edited test", items[0]["cheat_reasons"][0])

    def test_probe_is_cheating_even_if_hidden_tests_pass(self):
        _, items, _ = run("mock:probe")
        for i in items:
            self.assertEqual(i["hidden_passed"], i["hidden_total"])
            self.assertFalse(i["passed"])
            self.assertIn("tried to read the hidden tests", " ".join(i["cheat_reasons"]))

    def test_garble_fires_every_invalid_kind_and_still_passes(self):
        _, items, _ = run("mock:garble", tasks="u05")
        i = items[0]
        self.assertTrue(i["passed"])
        self.assertEqual(i["invalid_by_kind"], {"malformed_tool_call": 1, "unusable_edit": 1, "syntax_error": 1, "no_action": 1})
        self.assertTrue(i["invalid_item"])
        m = utility_metrics(items)
        self.assertAlmostEqual(m["invalid_turn_rate"], 4 / i["turns"])

    def test_writing_dotdot_content_is_not_a_probe(self):
        # u03's correct module contains "../" in its examples; only paths are scanned.
        t = load_tasks("u03")[0]
        with tempfile.TemporaryDirectory() as d:
            hashes = materialize(t, Path(d) / "ws")
            ev = [{"event": "tool_call", "name": "write_file", "args": {"path": "paths.py", "content": '"../a" hidden'}, "output": "created"}]
            self.assertEqual(cheat_reasons(t, Path(d) / "ws", hashes, ev, t.reference), [])
            ev = [{"event": "tool_call", "name": "search", "args": {"pattern": "x", "path": "../.."}, "output": "ERROR: denied: path '../..' is outside the workspace"}]
            self.assertTrue(cheat_reasons(t, Path(d) / "ws", hashes, ev, t.reference))


class Denominators(unittest.TestCase):
    def test_import_error_keeps_the_real_hidden_count(self):
        def fn(system, messages, tools, tool_choice):
            if sum(m["role"] == "assistant" for m in messages) == 0:
                return LLMResponse("", [ToolCall("w", "write_file", {"path": "flatten.py", "content": "def flatten(:\n"})])
            return LLMResponse("", [ToolCall("f", "finish", {"summary": "done"})])
        _, items, _ = run("scripted", tasks="u09", provider=ScriptedProvider(fn))
        i = items[0]
        self.assertEqual((i["hidden_passed"], i["hidden_total"]), (0, 10))
        self.assertEqual(i["invalid_by_kind"], {"syntax_error": 1})


class InvalidOutputDefinition(unittest.TestCase):
    def test_no_tool_reply_after_passing_tests_is_valid(self):
        ev = [{"event": "llm_call", "kind": "loop", "tool_calls": ["run_tests"]},
              {"event": "tool_call", "name": "run_tests", "output": "PASSED: ...", "ok": True},
              {"event": "llm_call", "kind": "loop", "tool_calls": [], "text": "done"}]
        self.assertEqual(invalid_output(ev)["invalid_turns"], 0)

    def test_no_tool_reply_before_passing_is_no_action(self):
        ev = [{"event": "llm_call", "kind": "loop", "tool_calls": [], "text": "Here is the code: ..."}]
        self.assertEqual(invalid_output(ev)["invalid_by_kind"], {"no_action": 1})

    def test_final_answer_call_is_not_a_turn(self):
        ev = [{"event": "llm_call", "kind": "final", "tool_calls": []}]
        self.assertEqual(invalid_output(ev)["turns"], 0)

    def test_text_tool_error_and_recovered_calls(self):
        ev = [{"event": "llm_call", "kind": "loop", "tool_calls": [], "text_tool_error": "looked like a tool call"},
              {"event": "llm_call", "kind": "loop", "tool_calls": ["read_file"], "tool_call_source": "text"},
              {"event": "tool_call", "name": "read_file", "output": "1| x", "ok": True}]
        r = invalid_output(ev)
        self.assertEqual((r["invalid_turns"], r["text_tool_turns"]), (1, 1))
        self.assertEqual(r["invalid_by_kind"], {"malformed_tool_call": 1})

    def test_one_turn_counts_once(self):
        ev = [{"event": "llm_call", "kind": "loop", "tool_calls": ["edit_file", "foo"]},
              {"event": "tool_call", "name": "edit_file", "output": "ERROR: old_string not found.", "ok": False},
              {"event": "tool_call", "name": "foo", "output": "ERROR: unknown tool 'foo'", "ok": False}]
        r = invalid_output(ev)
        self.assertEqual((r["turns"], r["invalid_turns"]), (1, 1))
        self.assertEqual(r["invalid_by_kind"], {"unusable_edit": 1})

    def test_missing_argument_is_malformed(self):
        ev = [{"event": "llm_call", "kind": "loop", "tool_calls": ["read_file"]},
              {"event": "tool_call", "name": "read_file", "output": "ERROR: TypeError: missing a required argument: 'path'", "ok": False}]
        self.assertEqual(invalid_output(ev)["invalid_by_kind"], {"malformed_tool_call": 1})


class OutOfBalance(unittest.TestCase):
    def test_402_stops_the_run_and_marks_the_rest_not_run(self):
        items_seen = []

        def fn(system, messages, tools, tool_choice):
            first = messages[0]["content"]
            if first not in items_seen:
                items_seen.append(first)
            if len(items_seen) > 1:                     # the balance runs out at the second item
                raise FatalProviderError("HTTP 402 (out of balance): {}", 402, "https://api.deepseek.com/chat/completions")
            if sum(m["role"] == "assistant" for m in messages) == 0:
                return LLMResponse("", [ToolCall("a", "read_file", {"path": "README.md"})])
            return LLMResponse("", [ToolCall("f", "finish", {"summary": "done"})])
        p = ScriptedProvider(fn, model="deepseek-v4-pro")
        p.name = "deepseek"
        cfg, items, out = run("deepseek", tasks="u01,u02,u03", provider=p)
        not_run = [i for i in items if i.get("not_run")]
        done = [i for i in items if not i.get("not_run")]
        self.assertEqual(len(done), 1)                      # the first item completed
        self.assertEqual(len(not_run), 2)                   # the erroring item and the one after it
        self.assertTrue(all(i["not_run_reason"] == "Out of DeepSeek balance (HTTP 402)" for i in not_run))
        self.assertIn("Out of DeepSeek balance", cfg["stopped"])
        m = utility_metrics(items)
        self.assertEqual((m["n_items"], m["not_run"]), (1, 2))
        rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 3)                      # completed receipts are kept on disk


if __name__ == "__main__":
    unittest.main()


class SimpleLoop(unittest.TestCase):
    """--loop simple: whole-file replies, harness-run visible tests, same scoring and metrics."""

    def test_good_mock_repairs_after_feedback(self):
        _, items, _ = run("mock", tasks="u03,u05", loop="simple")
        for i in items:
            self.assertTrue(i["passed"])
            self.assertEqual((i["turns"], i["verify_rounds"], i["loop"]), (2, 2, "simple"))
            self.assertFalse(i["false_success"])         # no claims in this loop
            self.assertGreater(i["test_s"], 0)

    def test_one_shot_is_one_attempt(self):
        out = Path(tempfile.mkdtemp())
        _, items = run_utility_benchmark(load_tasks("u03"), model="mock", out_dir=out, loop="simple",
                                         config=AgentConfig(max_verify_rounds=1), progress=lambda *_: None)
        self.assertEqual((items[0]["turns"], items[0]["passed"]), (1, False))
        self.assertEqual(items[0]["failure_category"], "unchanged")

    def test_lazy_never_passes_and_uses_every_attempt(self):
        _, items, _ = run("mock:lazy", tasks="u05", loop="simple")
        i = items[0]
        self.assertFalse(i["passed"])
        self.assertEqual(i["turns"], 3)                 # every attempt (AgentConfig default max_verify_rounds=3)
        self.assertEqual(i["failure_category"], "unchanged")

    def test_garble_counts_no_code_and_syntax_error(self):
        _, items, _ = run("mock:garble", tasks="u05", loop="simple")
        i = items[0]
        self.assertTrue(i["passed"])
        self.assertEqual(i["invalid_by_kind"], {"no_action": 1, "syntax_error": 1})
        self.assertEqual(i["tool_calls"], 2)            # two files written (the broken one and the reference)

    def test_feedback_carries_the_visible_test_output(self):
        seen = []

        def fn(system, messages, tools, tool_choice):
            seen.append(messages[-1]["content"])
            self.assertIsNone(tools)                     # no tools in this loop
            return LLMResponse("```python\ndef parse_ini(text):\n    return {}\n```")
        _, items, _ = run("scripted", tasks="u05", loop="simple", provider=ScriptedProvider(fn))
        self.assertIn("## Specification", seen[0])
        self.assertIn("test_r3_basic_sections", seen[0])  # the visible tests are in the first prompt
        self.assertIn("FAILED", seen[1])
        self.assertIn("test_r3_basic_sections", seen[1])  # and their failure output in the feedback
        self.assertEqual(items[0]["failure_category"], "visible_fail")

    def test_402_in_simple_loop_is_not_run(self):
        def fn(system, messages, tools, tool_choice):
            raise FatalProviderError("HTTP 402: {}", 402, "https://api.deepseek.com/chat/completions")
        p = ScriptedProvider(fn)
        cfg, items, _ = run("deepseek", tasks="u01,u02", loop="simple", provider=p)
        self.assertTrue(all(i.get("not_run") for i in items))
        self.assertIn("Out of DeepSeek balance", cfg["stopped"])

    def test_simple_loop_rejects_prompt_variants(self):
        with self.assertRaises(ValueError):
            run("mock", tasks="u01", loop="simple", prompt_variant="tests-first")


class FailureCategories(unittest.TestCase):
    def test_categories_first_match(self):
        from codebenchkit.utility.bench import failure_category
        base = {"passed": False, "cheated": False, "final_code": "x", "final_hidden": {}, "trusted_visible_pass": False}
        self.assertIsNone(failure_category({**base, "passed": True}))
        self.assertEqual(failure_category({**base, "cheated": True}, "x"), "cheated")
        self.assertEqual(failure_category(base, "x"), "unchanged")
        self.assertEqual(failure_category({**base, "final_hidden": {"tests/test_hidden.py::<import>": "error"}}, "y"), "broken")
        self.assertEqual(failure_category(base, "y"), "visible_fail")
        self.assertEqual(failure_category({**base, "trusted_visible_pass": True}, "y"), "hidden_only")


class Resume(unittest.TestCase):
    def _provider(self, fail_after_items=None):
        seen = []

        def fn(system, messages, tools, tool_choice):
            first = messages[0]["content"]
            if first not in seen:
                seen.append(first)
            if fail_after_items is not None and len(seen) > fail_after_items:
                raise FatalProviderError("HTTP 402: {}", 402, "https://api.deepseek.com/chat/completions")
            if sum(m["role"] == "assistant" for m in messages) == 0:
                return LLMResponse("", [ToolCall("a", "read_file", {"path": "README.md"})])
            return LLMResponse("", [ToolCall("f", "finish", {"summary": "done"})])
        p = ScriptedProvider(fn, model="deepseek-v4-pro")
        p.name = "deepseek"
        return p

    def test_resume_fills_only_the_not_run_items(self):
        from codebenchkit.utility.bench import resume_utility_benchmark
        cfg, items, out = run("deepseek", tasks="u01,u02,u03", provider=self._provider(fail_after_items=1))
        before = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
        self.assertEqual(sum(1 for i in before if i.get("not_run")), 2)
        cfg2, items2 = resume_utility_benchmark(out, "deepseek", AgentConfig(max_iterations=20, context_limit=16000),
                                                provider=self._provider(), max_cost=0.10, progress=lambda *_: None)
        after = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
        self.assertEqual(len(after), 3)
        self.assertFalse(any(i.get("not_run") for i in after))
        self.assertEqual([(i["task"], i["sample"]) for i in after], [(i["task"], i["sample"]) for i in before])
        self.assertEqual(after[[i.get("not_run") for i in before].index(None)]["trace"],
                         before[[i.get("not_run") for i in before].index(None)]["trace"])   # the completed item is untouched
        self.assertEqual(sum(bool(i.get("resumed")) for i in after), 2)
        self.assertEqual(cfg2["not_run"], 0)
        self.assertIn("Out of DeepSeek balance", cfg2["resumed"][0]["previous_stop"])
        self.assertEqual(cfg2["resumed"][0]["items_resumed"], 2)

    def test_resume_refuses_a_different_budget_or_model(self):
        from codebenchkit.utility.bench import resume_utility_benchmark
        _, _, out = run("deepseek", tasks="u01,u02", provider=self._provider(fail_after_items=1))
        with self.assertRaises(ValueError):
            resume_utility_benchmark(out, "deepseek", AgentConfig(max_iterations=5, context_limit=16000),
                                     provider=self._provider(), progress=lambda *_: None)
        other = self._provider()
        other.model = "deepseek-flash"
        with self.assertRaises(ValueError):
            resume_utility_benchmark(out, "deepseek", AgentConfig(max_iterations=20, context_limit=16000),
                                     provider=other, progress=lambda *_: None)
