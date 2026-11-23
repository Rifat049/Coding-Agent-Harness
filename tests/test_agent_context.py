"""Context and history contract tests; no network or cheap-model summaries."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

from codebenchkit.agent.tools import Workspace


class ContextTests(unittest.TestCase):
    def test_context_layers_are_bounded_with_stable_prefix(self):
        self.assertIsNotNone(importlib.util.find_spec('codebenchkit.agent.context'))
        from codebenchkit.agent.context import build_context
        with tempfile.TemporaryDirectory() as root:
            Path(root, 'AGENTS.md').write_text('Project rule\n' * 100)
            context = build_context(Workspace(root), 'fix parser', 'mock', 'python -m unittest',
                                    budgets={'instructions': 120, 'repo_map': 80})
            self.assertTrue(context.system.startswith(context.system.stable_prefix))
            self.assertIn('Prefer edit_file', context.system.stable_prefix)
            self.assertIn('python -m unittest', context.system)
            self.assertLessEqual(context.report['sections']['instructions'], 120)
            self.assertEqual(context.report['total_chars'], len(context.system))

    def test_instruction_symlink_does_not_leak(self):
        from codebenchkit.agent.context import build_context
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as outside:
            target = Path(outside, 'private')
            target.write_text('DO_NOT_LEAK')
            Path(root, 'AGENTS.md').symlink_to(target)
            context = build_context(Workspace(root), 'fix parser', 'mock', None)
            self.assertNotIn('DO_NOT_LEAK', context.system)
            self.assertTrue(context.report['warnings'])


class HistoryTests(unittest.TestCase):
    @staticmethod
    def group(index, size=5000):
        return [{'role': 'assistant', 'content': '', 'tool_calls': [
                    {'id': f'{index}a', 'name': 'read_file', 'args': {'path': 'x'}},
                    {'id': f'{index}b', 'name': 'read_file', 'args': {'path': 'y'}}]},
                {'role': 'tool', 'tool_call_id': f'{index}a', 'content': 'a' * size},
                {'role': 'tool', 'tool_call_id': f'{index}b', 'content': 'b' * size}]

    def test_trim_preserves_all_newest_unobserved_results(self):
        self.assertIsNotNone(importlib.util.find_spec('codebenchkit.agent.history'))
        from codebenchkit.agent.history import shrink_seen
        history = [{'role': 'user', 'content': 'task'}] + self.group(1) + self.group(2)
        count = shrink_seen(history)
        self.assertEqual(count, 2)
        self.assertLess(len(history[2]['content']), 4000)
        self.assertEqual(len(history[-2]['content']), 5000)
        self.assertEqual(len(history[-1]['content']), 5000)

    def test_compaction_keeps_task_and_complete_multi_tool_groups(self):
        from codebenchkit.agent.history import compact_history
        history = [{'role': 'user', 'content': 'original task'}]
        for i in range(8):
            history.extend(self.group(i))
        newest = history[-3:]
        report = compact_history(history, context_limit=2000)
        self.assertTrue(report['compacted'])
        self.assertEqual(history[0]['content'], 'original task')
        self.assertEqual(history[-3:], newest)
        self.assertIn('Deterministic history digest', history[1]['content'])
        calls = {c['id'] for m in history for c in m.get('tool_calls', [])}
        results = {m['tool_call_id'] for m in history if m['role'] == 'tool'}
        self.assertEqual(calls, results)

    def test_compaction_summary_strips_thinking_and_falls_back(self):
        from codebenchkit.agent.history import compact_history
        def build():
            history = [{'role': 'user', 'content': 'original task'}]
            for i in range(8):
                history.extend(self.group(i))
            for m in history:
                if m.get('tool_calls'):
                    m['raw'] = {'provider': 'anthropic', 'thinking': [{'type': 'thinking', 'signature': 's'}]}
            return history
        history = build()
        prompts = []
        report = compact_history(history, 2000, summarize=lambda p: prompts.append(p) or 'Read files a and b; edited a.py; the tests have not been run yet.')
        self.assertEqual(report['method'], 'summary')
        self.assertIn('Read files a and b', history[1]['content'])
        self.assertGreater(report['thinking_blocks_stripped'], 0)
        self.assertFalse(any(m.get('raw') for m in history))          # kept turns too
        self.assertIn('assistant called', prompts[0])
        history = build()
        report = compact_history(history, 2000, summarize=lambda p: 1 / 0)
        self.assertEqual(report['method'], 'digest')
        self.assertIn('ZeroDivisionError', report['summary_error'])
        calls = {c['id'] for m in history for c in m.get('tool_calls', [])}
        self.assertEqual(calls, {m['tool_call_id'] for m in history if m['role'] == 'tool'})


if __name__ == '__main__':
    unittest.main()
