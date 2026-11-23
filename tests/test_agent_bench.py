"""The agent receives public specs only; hidden scoring is a terminal phase."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from codebenchkit.tasks import Task, get_tasks


class PublicBenchTests(unittest.TestCase):
    def test_materialization_and_trusted_check(self):
        from codebenchkit.agent.bench import public_task, materialize_task, visible_verifier
        sentinel = 'PRIVATE_SENTINEL_7db92'
        task = Task('slugify', 'def slugify(text: str) -> str', 'Lowercase ASCII URL slug.',
                    [{'args': ['Hello World'], 'expected': 'hello-world'}],
                    [{'args': [sentinel], 'expected': sentinel}], reference=sentinel)
        public = public_task(task)
        self.assertFalse(hasattr(public, 'hidden'))
        self.assertFalse(hasattr(public, 'reference'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt = materialize_task(public, root)
            self.assertNotIn(sentinel, prompt)
            for path in root.iterdir():
                self.assertNotIn(sentinel, path.read_text())
            self.assertEqual({p.name for p in root.iterdir()}, {'solution.py', 'check_visible.py'})
            verify = visible_verifier(public, root)
            (root / 'check_visible.py').write_text('print("PASSED")\n')
            self.assertFalse(verify()[0], 'mutable checker must not determine trusted verification')
            (root / 'solution.py').write_text('import os\ndef slugify(text): return text\n')
            with patch('codebenchkit.agent.bench.run_tests') as runner:
                self.assertFalse(verify()[0])
                runner.assert_not_called()
            (root / 'solution.py').write_text('def slugify(text): return text.lower().replace(" ", "-")\n')
            self.assertTrue(verify()[0])
            materialize_task(public, root)
            (root / 'solution.py').write_text('def slugify(text): return text.lower().replace(" ", "-")\n')
            standalone = subprocess.run([sys.executable, '-I', 'check_visible.py'], cwd=root,
                                        capture_output=True, text=True)
            self.assertEqual(standalone.returncode, 0, standalone.stderr)

    def test_history_candidates_only_successful_writes_and_edits(self):
        from codebenchkit.agent.bench import candidate_sources
        def pair(number, name, args, ok=True):
            return [{'role': 'assistant', 'tool_calls': [{'id': str(number), 'name': name, 'args': args}]},
                    {'role': 'tool', 'tool_call_id': str(number), 'content': 'done' if ok else 'ERROR: denied',
                     'is_error': not ok}]
        initial = 'def f():\n    pass\n'
        first = 'def f():\n    return None\n'
        history = pair(1, 'write_file', {'path': 'solution.py', 'content': first})
        history += pair(2, 'write_file', {'path': 'solution.py', 'content': 'BAD'}, False)
        history += pair(3, 'edit_file', {'path': './solution.py', 'old_string': 'None', 'new_string': '42'})
        self.assertEqual(candidate_sources(history, initial), [first, first.replace('None', '42')])


class BenchmarkIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from codebenchkit.agent.bench import run_agent_benchmark
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.runs = {}
        for enabled in (False, True):
            out = cls.root / ('on' if enabled else 'off')
            cls.runs[enabled] = run_agent_benchmark(get_tasks(), memory_enabled=enabled, samples=2, out_dir=out,
                                                    seed=7, keep_workspaces=True, progress=lambda _: None)

    @classmethod
    def tearDownClass(cls):
        import shutil
        for config, _ in cls.runs.values():
            shutil.rmtree(config.get('workspaces', ''), ignore_errors=True)
        cls.tmp.cleanup()

    def test_v01_schema_compare_and_identical_budgets(self):
        from codebenchkit.metrics import compare_runs, load_run, render_report, summarize
        for enabled, (config, items) in self.runs.items():
            out = self.root / ('on' if enabled else 'off')
            stored_config, stored_items, attempts = load_run(out)
            self.assertEqual(stored_items, items)
            self.assertEqual(len(items), 20)
            self.assertTrue(all(i['attempts_used'] >= 1 and i['completion_tokens'] > 0 for i in items))
            self.assertEqual(len(attempts), sum(i['attempts_used'] for i in items))
            self.assertIn('Budget:', render_report(config, summarize(items, attempts)))
            self.assertEqual((out / 'memory').exists(), enabled)
            self.assertEqual(stored_config['prompt_style'], 'memory_on' if enabled else 'memory_off')
        off, on = self.runs[False][0], self.runs[True][0]
        self.assertEqual(off['budget'], on['budget'])
        self.assertEqual(off['order'], on['order'])                      # same seeded order
        text = compare_runs(self.root / 'off', self.root / 'on')
        self.assertNotIn('WARNING', text)
        self.assertIn('Paired items: 20', text)
        self.assertIn('McNemar', text)

    def test_memory_condition_writes_episodes_and_retrieves_lessons(self):
        from codebenchkit.agent.memory import Memory
        items = self.runs[True][1]
        mem = Memory(self.root / 'on' / 'memory' / 'memory.sqlite3')
        self.addCleanup(mem.close)
        self.assertEqual(len(mem.episodes()), 20)
        self.assertGreater(len(mem.facts()), 0)
        self.assertTrue(any(i['memory_facts'] for i in items))
        self.assertTrue(all(not i['memory_facts'] for i in self.runs[False][1]))
        # pass@1 can only improve: a retrieved lesson makes the mock write the fixture first.
        first = lambda run: sum(i['first_hidden_pass'] for i in run[1])
        self.assertGreaterEqual(first(self.runs[True]), first(self.runs[False]))

    def test_hidden_test_firewall(self):
        """Every hidden-test string of a task (calls, distinctive args and outputs) is absent from that
        task's workspace files and traces; no hidden call appears anywhere in memory."""
        import sqlite3
        from codebenchkit.agent.bench import hidden_strings
        from codebenchkit.tasks import TASKS
        checked = 0
        db_dump = ''
        con = sqlite3.connect(self.root / 'on' / 'memory' / 'memory.sqlite3')
        db_dump = '\n'.join(con.iterdump())
        con.close()
        self.assertIn('INSERT INTO "episodes"', db_dump)
        for task in TASKS:
            needles = hidden_strings(task)
            self.assertTrue(needles)
            texts = []
            for config, _ in self.runs.values():
                for ws in Path(config['workspaces']).glob(f'{task.name}-s*'):
                    texts += [(str(p), p.read_text(errors='replace')) for p in ws.rglob('*') if p.is_file()]
            texts += [(str(p), p.read_text()) for p in self.root.rglob(f'traces/{task.name}-s*.jsonl')]
            self.assertEqual(len(texts), 2 * 2 * 3)                  # 2 conditions x 2 samples x (2 files + trace)
            for where, text in texts:
                for h in needles:
                    checked += 1
                    self.assertNotIn(h, text, f'hidden string {h!r} leaked into {where}')
            for h in needles:
                if h.startswith(task.name + '('):
                    self.assertNotIn(h, db_dump, f'hidden call {h} leaked into memory')
        self.assertGreater(checked, 500)

    def test_benchmark_agent_has_no_shell(self):
        for p in self.root.rglob('traces/*.jsonl'):
            start = json.loads(p.read_text().splitlines()[0])
            self.assertNotIn('run_command', start['tools'])
            break


if __name__ == '__main__':
    unittest.main()
