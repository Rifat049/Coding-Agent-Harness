"""CLI contracts are exercised without keys or external network access."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from codebenchkit.cli import main


class AgentCLITests(unittest.TestCase):
    def test_memory_crud_and_markdown_export(self):
        with tempfile.TemporaryDirectory() as directory:
            common = ['agent', 'memory']
            def invoke(*args):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    result = main(common + list(args) + ['--memory-dir', directory])
                self.assertEqual(result, 0)
                return output.getvalue()
            added = json.loads(invoke('add', 'Prefer unittest for offline regression coverage.',
                                     '--subject', 'testing', '--scope', 'repo:demo'))
            self.assertGreater(added['id'], 0)
            self.assertIn('Prefer unittest', invoke('list'))
            self.assertIn('Prefer unittest', invoke('search', 'unittest', '--scope', 'repo:demo'))
            target = Path(directory) / 'export.md'
            invoke('export', '--out', str(target))
            self.assertIn('# Agent memory', target.read_text())
            invoke('forget', str(added['id']))
            self.assertNotIn('Prefer unittest', invoke('list'))

    def test_memory_list_shows_pending_promotions_and_promote_applies(self):
        from codebenchkit.agent.memory import Memory
        with tempfile.TemporaryDirectory() as directory:
            with Memory(Path(directory) / 'memory.sqlite3') as mem:
                e1 = mem.add_episode('repo:a', 'task', 'success')
                e2 = mem.add_episode('repo:b', 'task', 'success')
                fid = mem.add_fact('tests', 'Run the unit tests before finishing.', 'command', 'repo:a', [e1], 'consolidation')
                mem.propose_promotion(fid, 'repo:a', ['repo:a', 'repo:b'], [e2])
            def invoke(*args, code=0):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(main(['memory', *args, '--memory-dir', directory]), code)
                return output.getvalue()
            listing = invoke('list')
            self.assertIn('Pending promotions', listing)
            self.assertIn(f'F{fid}: repo:a -> global', listing)
            self.assertTrue(json.loads(invoke('promote', str(fid)))['promoted'])
            self.assertIn('global', invoke('list'))
            self.assertNotIn('Pending promotions', invoke('list'))
            self.assertFalse(json.loads(invoke('promote', str(fid), code=1))['promoted'])

    def test_agent_help_exposes_limits(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as error:
            main(['agent', 'run', '--help'])
        self.assertEqual(error.exception.code, 0)
        for option in ['--max-iterations', '--token-budget', '--wall-seconds', '--approval', '--gate']:
            self.assertIn(option, output.getvalue())

    def test_bench_rejects_nonpositive_samples(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(['agent-bench', '--model', 'mock', '--samples', '0'])
        self.assertEqual(error.exception.code, 2)


    def quiet(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_agent_run_with_mock_writes_episode_and_trace(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d) / 'demo'
            repo.mkdir()
            (repo / 'README.md').write_text('demo')
            code, out, _ = self.quiet(['agent', 'run', 'look around', '--repo', str(repo), '--model', 'mock',
                                       '--memory-dir', str(Path(d) / 'mem')])
            self.assertEqual(code, 0, out)
            self.assertIn('"outcome": "partial"', out)
            trace = Path(out.split('Trace: ')[1].split()[0])
            self.assertTrue(trace.exists())
            code, out, _ = self.quiet(['memory', 'list', '--episodes', '--memory-dir', str(Path(d) / 'mem')])
            self.assertIn('look around', out)

    def test_agent_bench_off_on_then_compare(self):
        with tempfile.TemporaryDirectory() as d:
            for mem in ('off', 'on'):
                code, out, err = self.quiet(['agent-bench', '--model', 'mock', '--memory', mem, '--samples', '2',
                                             '--tasks', 'slugify,rle_encode,chunk', '--out', d, '--name', f'mem_{mem}'])
                self.assertEqual(code, 0, err)
                self.assertIn('Final pass rate (hidden)', out)
            code, out, _ = self.quiet(['compare', f'{d}/mem_off', f'{d}/mem_on'])
            self.assertEqual(code, 0)
            self.assertIn('Paired items: 6', out)
            self.assertNotIn('WARNING', out)

    def test_chat_repl_keeps_history_and_frozen_system(self):
        from codebenchkit.agent.chat import ChatSession, repl
        from codebenchkit.agent.llm import ScriptedProvider
        from codebenchkit.agent.loop import AgentConfig
        from codebenchkit.agent.memory import Memory
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d) / 'r'
            repo.mkdir()
            provider = ScriptedProvider(lambda s, m, t, c: f'answer {len(m)}')
            mem = Memory(Path(d) / 'm.sqlite3')
            self.addCleanup(mem.close)
            mem.add_fact('style', 'This repo uses tabs for indentation in Makefiles.', 'convention', 'repo:r')
            session = ChatSession(repo, provider, AgentConfig(), memory=mem, trace_dir=Path(d) / 't')
            lines = iter(['what indentation does the Makefile use?', '/memory', 'and again?', '/exit'])
            printed = []
            self.assertEqual(repl(session, lambda _: next(lines), printed.append), 0)
            self.assertEqual(provider.calls, 2)
            self.assertEqual(provider.seen[0]['system'], provider.seen[1]['system'])   # frozen per session
            first_user = provider.seen[0]['messages'][0]['content']
            self.assertIn('[Memory for this message', first_user)                      # memory in the message
            self.assertIn('tabs for indentation', first_user)
            self.assertNotIn('tabs for indentation', str(provider.seen[0]['system']))
            self.assertEqual(len(provider.seen[1]['messages']), 3)                     # history carried
            self.assertTrue(any('"episodes"' in p for p in printed))


if __name__ == '__main__':
    unittest.main()
