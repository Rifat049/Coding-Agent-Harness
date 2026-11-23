import json
import tempfile
import unittest
from pathlib import Path

from codebenchkit.backends import MockBackend
from codebenchkit.checks import extract_code, static_check
from codebenchkit.harness import Budget, build_prompt, run_benchmark, run_item
from codebenchkit.metrics import compare_runs, mcnemar_exact, summarize
from codebenchkit.sandbox import run_tests
from codebenchkit.tasks import TASKS, TASKS_BY_NAME

RLE = TASKS_BY_NAME["rle_encode"]


class TestExtraction(unittest.TestCase):
    def test_prefers_block_defining_function(self):
        text = "```python\nprint('hi')\n```\nthen\n```python\ndef f(x):\n    return x\n```\n```python\nf(1)\n```"
        self.assertIn("def f", extract_code(text, "f").code)

    def test_unfenced_valid_python(self):
        ext = extract_code("def f(x):\n    return x\n", "f")
        self.assertEqual(ext.how, "unfenced")

    def test_prose_is_no_code(self):
        self.assertEqual(static_check(extract_code("Use a loop.", "f").code, "f").category, "no_code")


class TestStaticCheck(unittest.TestCase):
    def test_categories(self):
        self.assertEqual(static_check("def f(:\n pass", "f").category, "syntax_error")
        self.assertEqual(static_check("def g():\n pass", "f").category, "missing_function")
        self.assertEqual(static_check("import os\ndef f():\n pass", "f").category, "banned")
        self.assertEqual(static_check("def f():\n return open('x')", "f").category, "banned")
        self.assertEqual(static_check("def f():\n return ().__class__.__subclasses__()", "f").category, "banned")
        self.assertEqual(static_check("import re\ndef f():\n return 1", "f").category, "ok")


class TestSandbox(unittest.TestCase):
    def test_infinite_loop_times_out(self):
        r = run_tests("def f(x):\n    while True: pass", "f", [{"args": [1], "expected": 1}], per_test_timeout=0.5)
        self.assertTrue(r.any_timeout)
        self.assertFalse(r.all_passed(1))

    def test_exception_expectation(self):
        code = "def f(x):\n    if x < 0: raise ValueError\n    return x"
        tests = [{"args": [-1], "raises": "ValueError"}, {"args": [2], "expected": 2}, {"args": [3], "raises": "ValueError"}]
        r = run_tests(code, "f", tests)
        self.assertEqual([c["passed"] for c in r.cases], [True, True, False])

    def test_tuple_output_normalised(self):
        r = run_tests("def f():\n    return (1, (2, 3))", "f", [{"args": [], "expected": [1, [2, 3]]}])
        self.assertTrue(r.all_passed(1))

    def test_load_error(self):
        r = run_tests("x = 1/0\ndef f(): pass", "f", [{"args": [], "expected": None}])
        self.assertEqual(r.status, "load_error")

    def test_memory_bomb_contained(self):
        r = run_tests("def f():\n    return len('x' * (4 << 30))", "f", [{"args": [], "expected": 0}], mem_mb=512)
        self.assertFalse(r.all_passed(1))


class TestFirewall(unittest.TestCase):
    def test_hidden_tests_never_in_prompt(self):
        for task in TASKS:
            prompt = json.dumps(build_prompt(task, "default"))
            for t in task.hidden:
                call = f"{task.name}({', '.join(repr(a) for a in t['args'])})"
                self.assertNotIn(call, prompt, f"hidden test leaked for {task.name}")

    def test_minimal_prompt_has_no_examples(self):
        self.assertNotIn("Examples", build_prompt(RLE, "minimal")[1]["content"])


class TestAgentLoop(unittest.TestCase):
    def test_reference_passes_first_try(self):
        item, attempts = run_item(RLE, 0, MockBackend(1.0, "ref"), Budget(), 0.0, "default")
        self.assertTrue(item["first_hidden_pass"] and item["final_hidden_pass"])
        self.assertEqual(len(attempts), 1)

    def test_overfit_caught_by_hidden_tests(self):
        class Overfitter(MockBackend):
            def generate(self, messages, max_new_tokens, temperature, seed, context=None):
                from codebenchkit.backends import Generation, _broken
                return Generation(_broken(context["task"], "overfit"), 1, 10, 0.0)
        item, attempts = run_item(RLE, 0, Overfitter(), Budget(max_attempts=3), 0.0, "default")
        self.assertTrue(item["final_visible_pass"])
        self.assertFalse(item["final_hidden_pass"])
        self.assertEqual(len(attempts), 1)  # visible passed, so no repair was triggered

    def test_token_budget_stops_repair(self):
        item, attempts = run_item(RLE, 0, MockBackend(0.0, "bad"), Budget(max_attempts=5, max_total_tokens=1), 0.0, "default")
        self.assertEqual(len(attempts), 1)

    def test_repair_feedback_uses_visible_only(self):
        seen = []

        class Spy(MockBackend):
            def generate(self, messages, max_new_tokens, temperature, seed, context=None):
                seen.append(json.dumps(messages))
                return super().generate(messages, max_new_tokens, temperature, seed, context)
        task = TASKS_BY_NAME["parse_duration"]
        run_item(task, 0, Spy(0.0, "spy"), Budget(max_attempts=3), 0.0, "default")
        for t in task.hidden:
            call = f"{task.name}({', '.join(repr(a) for a in t['args'])})"
            self.assertFalse(any(call in s for s in seen))


class TestMetrics(unittest.TestCase):
    def test_mcnemar(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact(0, 6), 2 / 64)
        self.assertGreater(mcnemar_exact(3, 3), 0.99)

    def test_end_to_end_and_compare(self):
        with tempfile.TemporaryDirectory() as d:
            quiet = lambda *_: None
            for name, skill in (("a", 0.3), ("b", 0.9)):
                cfg, items = run_benchmark(TASKS, MockBackend(skill, name), Budget(), 2, 0.0, "default",
                                           Path(d) / name, progress=quiet)
                attempts = [json.loads(l) for l in (Path(d) / name / "attempts.jsonl").read_text().splitlines()]
                s = summarize(items, attempts)
                self.assertEqual(s["n_items"], 20)
            text = compare_runs(Path(d) / "a", Path(d) / "b")
            self.assertIn("McNemar", text)


if __name__ == "__main__":
    unittest.main()
