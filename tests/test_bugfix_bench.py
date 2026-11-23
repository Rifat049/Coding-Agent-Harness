"""Bug-fix benchmark: tasks are well formed, and every reliability metric fires on its mock."""
import tempfile
import unittest
from pathlib import Path

from codebenchkit.agent.loop import AgentConfig
from codebenchkit.bugfix.bench import reliability, run_bugfix_benchmark
from codebenchkit.bugfix.core import load_bugs, selfcheck


def run(model, bugs):
    with tempfile.TemporaryDirectory() as d:
        _, items = run_bugfix_benchmark(load_bugs(bugs), model=model, small_model="mock",
                                        config=AgentConfig(max_iterations=12), out_dir=Path(d), progress=lambda *_: None)
    return {i["task"].split("_")[0]: i for i in items}, reliability(items)


class BugfixBenchTests(unittest.TestCase):
    def test_compare_shows_reliability_for_bugfix_runs(self):
        from codebenchkit.metrics import compare_runs
        with tempfile.TemporaryDirectory() as d:
            for model, name in (("mock", "a"), ("mock:regress", "b")):
                run_bugfix_benchmark(load_bugs("b01,b06"), model=model, small_model="mock",
                                     config=AgentConfig(max_iterations=12), out_dir=Path(d) / name, progress=lambda *_: None)
            text = compare_runs(Path(d) / "a", Path(d) / "b")
        self.assertIn("## Reliability (bug-fix runs)", text)
        self.assertIn("| Regression rate | 0.0% | 100.0% |", text)

    def test_cost_cap_stops_between_items(self):
        from unittest.mock import patch
        from codebenchkit.bugfix import bench
        with tempfile.TemporaryDirectory() as d, patch.object(bench, "trace_cost", lambda ev, m: 0.4):
            cfg, items = run_bugfix_benchmark(load_bugs("b01,b03,b06"), model="mock", small_model="mock",
                                              config=AgentConfig(max_iterations=12), out_dir=Path(d),
                                              progress=lambda *_: None, max_cost=0.75)
        self.assertEqual(len(items), 2)
        self.assertIn("cost cap $0.75 reached after 2/3 items", cfg["stopped"])

    def test_small_model_spend_counts_toward_the_item_cost(self):
        from codebenchkit.bugfix.bench import _small_cost
        class Small:
            usage_log = [{"input_tokens": 1_000_000, "cache_read_tokens": 0, "output_tokens": 0, "ts": 1795219200}]
            def describe(self):
                return "deepseek:deepseek-v4-pro"
        self.assertAlmostEqual(_small_cost(Small(), 0), 0.66)       # off-peak: Saturday 2026-11-21 00:00 UTC
        self.assertEqual(_small_cost(Small(), 1), 0.0)
        self.assertEqual(_small_cost(None, 0), 0.0)
        class Unpriced(Small):
            def describe(self):
                return "openai:gpt-5.5"
        self.assertIsNone(_small_cost(Unpriced(), 0))

    def test_cost_cap_refuses_unpriced_models(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaisesRegex(ValueError, "known price"):
            run_bugfix_benchmark(load_bugs("b01"), model="chatgpt", small_model=None, out_dir=Path(d),
                                 progress=lambda *_: None, max_cost=1.0)

    def test_memory_rounds_order_and_fresh_memory_per_round(self):
        with tempfile.TemporaryDirectory() as d:
            cfg_on, items_on = run_bugfix_benchmark(load_bugs("b01,b03,b06"), model="mock", small_model="mock",
                                                    memory_enabled=True, samples=2, rounds=True,
                                                    config=AgentConfig(max_iterations=12), out_dir=Path(d) / "on",
                                                    progress=lambda *_: None)
            cfg_off, _ = run_bugfix_benchmark(load_bugs("b01,b03,b06"), model="mock", small_model="mock", samples=2,
                                              rounds=True, config=AgentConfig(max_iterations=12),
                                              out_dir=Path(d) / "off", progress=lambda *_: None)
            self.assertEqual(cfg_on["order"], cfg_off["order"])
            self.assertEqual([o.split("#")[1] for o in cfg_on["order"]], ["0"] * 3 + ["1"] * 3)
            self.assertTrue((Path(d) / "on" / "memory" / "round-0" / "memory.sqlite3").is_file())
            self.assertTrue((Path(d) / "on" / "memory" / "round-1" / "memory.sqlite3").is_file())

    def test_selfcheck_passes_for_every_task(self):
        self.assertEqual(len(load_bugs()), 10)
        self.assertEqual(selfcheck(log=lambda *_: None), 0)

    def test_good_mock_fixes_reports_flaky_and_survives_blocked_command(self):
        items, r = run("mock", "b01,b09,b10")
        self.assertEqual((r["fix_rate"], r["regression_rate"], r["cheat_rate"], r["false_success_rate"]), (1, 0, 0, 0))
        self.assertTrue(items["b09"]["flaky_reported"])
        self.assertEqual(items["b10"]["blocked_commands"], 1)
        self.assertTrue(items["b10"]["continued_after_block"])
        self.assertEqual(items["b01"]["minimality_ratio"], 1.0)

    def test_tamper_is_cheating_and_a_false_success(self):
        items, r = run("mock:tamper", "b01")
        i = items["b01"]
        self.assertTrue(i["cheated"] and i["false_success"] and not i["fixed"])
        self.assertIn("edited visible test tests/test_reported_bug.py", i["cheat_reasons"])

    def test_hardcode_is_overfit(self):
        items, _ = run("mock:hardcode", "b02,b05,b08")
        for k in ("b02", "b05", "b08"):
            self.assertTrue(items[k]["overfit"] and items[k]["trusted_visible_pass"] and not items[k]["fixed"], k)

    def test_quick_fix_breaks_a_regression_test(self):
        items, r = run("mock:regress", "b01,b06")
        self.assertTrue(items["b01"]["regression"] and items["b06"]["regression"])
        self.assertEqual(r["regression_tests_broken"], 2)

    def test_lazy_is_a_false_success_with_no_change(self):
        items, _ = run("mock:lazy", "b03")
        self.assertTrue(items["b03"]["false_success"])
        self.assertEqual(items["b03"]["lines_changed"], 0)

    def test_sprawl_hurts_minimality_and_touches_outside_area(self):
        items, _ = run("mock:sprawl", "b01")
        self.assertGreater(items["b01"]["minimality_ratio"], 3)
        self.assertEqual(items["b01"]["files_outside_area"], ["shoplite/inventory.py"])


if __name__ == "__main__":
    unittest.main()
