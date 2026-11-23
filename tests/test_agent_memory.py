"""Memory regression tests using real SQLite/FTS5 and temporary skills."""
import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from codebenchkit.agent.memory import INDEX_VERSION, Memory, SkillLibrary, fts_query, gate, rerank_facts


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.memory = Memory(Path(self.tmp.name) / 'memory.db')
        self.addCleanup(self.memory.conn.close)

    def test_strict_budgets_include_headers_and_tags(self):
        m = self.memory
        m.add_fact('parser', 'Parser requires carefully validated inputs. ' * 10)
        m.add_episode('global', 'Parser validation', 'failure', notes=['Parser input invalid. ' * 30])
        for cap in (0, 1, 20, 100, 500):
            with self.subTest(cap=cap):
                m.fact_budget = cap
                m.episode_budget = cap
                found = m.retrieve('parser validation')
                self.assertLessEqual(len(m.render(found)), 2 * cap)
        m.fact_budget = m.episode_budget = 1000
        text = m.render(m.retrieve('parser validation'))
        self.assertIn('[F1]', text)
        self.assertIn('[E1]', text)
        self.assertIn('verify before relying', text)

    def test_gate_validates_json_types_and_fails_open(self):
        class Small:
            def complete(self, prompt, max_tokens):
                return self.answer
        small = Small()
        for answer in ('{"retrieve": "false"}', '{"retrieve": false, "query": []}',
                       '{"retrieve": 0}', '[]', 'bad json', '{}'):
            small.answer = answer
            self.assertEqual(gate('hello', 'llm', small), (True, 'hello'))
        self.assertEqual(gate('hello', 'llm'), (True, 'hello'))
        small.answer = '{"retrieve": false, "query": "parser"}'
        self.assertEqual(gate('hello', 'llm', small), (False, 'parser'))
        small.answer = '{"retrieve": true, "query": "parser"}'
        self.assertEqual(gate('hello', 'llm', small), (True, 'parser'))
        self.assertEqual(gate('hello', 'always'), (True, 'hello'))
        self.assertEqual(gate('hello', 'never'), (False, ''))
        self.assertFalse(gate('hi', 'heuristic')[0])
        self.assertTrue(gate('repair parser tests', 'heuristic')[0])
        small.complete = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('offline'))
        self.assertEqual(gate('hello', 'llm', small), (True, 'hello'))

    def test_fts_identifier_splitting_scope_and_zero_limits(self):
        m = self.memory
        local = m.add_fact('parseDuration', 'Handle missing duration units.', scope='repo:one')
        global_id = m.add_fact('duration', 'Duration syntax needs validation.')
        m.add_fact('duration', 'Unrelated private duration syntax.', scope='repo:two')
        m.add_episode('repo:one', 'parseDuration failing', 'failure')
        m.add_episode('repo:two', 'duration failed', 'failure')
        for query in ('duration', 'parse_duration', 'parseDuration', 'AND OR NEAR duration - ( : " *'):
            ids = {f['id'] for f in m.search_facts(query, 'repo:one')}
            self.assertEqual(ids, {local, global_id})
        self.assertEqual(m.search_facts('duration', 'repo:one', limit=0), [])
        self.assertEqual(m.search_episodes('duration', 'repo:one', limit=0), [])
        self.assertEqual(m.search_facts('the and or'), [])
        self.assertEqual(m.search_facts('!!!'), [])
        self.assertEqual(len(m.search_episodes('duration', 'repo:one')), 1)
        self.assertEqual([f['id'] for f in m.search_facts('duration')], [global_id])

    def test_skill_disclosure_caps_and_symlink_quarantine(self):
        root = Path(self.tmp.name) / 'skills'
        library = SkillLibrary(root, max_chars=100)
        proposal = library.propose('parser', 'Parser validation practices', 'PROPOSED NEVER LOAD')
        (root / 'alias').symlink_to(proposal.parent, target_is_directory=True)
        (root / 'alias-file').mkdir()
        (root / 'alias-file' / 'SKILL.md').symlink_to(proposal)
        approved = root / 'parser-approved'
        approved.mkdir()
        (approved / 'SKILL.md').write_text('---\nname: parser-approved\ndescription: Parser validation guidance\n---\n' + 'Useful steps. ' * 100)
        index = library.all()
        self.assertEqual(len(index), 1)
        self.assertNotIn('body', index[0])
        selected = library.match('parser validation')
        self.assertEqual(len(selected), 1)
        self.assertLessEqual(len(selected[0]['body']), 100)
        self.assertNotIn('PROPOSED', selected[0]['body'])
        self.assertEqual(library.match('unrelated weather'), [])
        self.assertEqual(library.match('parser validation', limit=0), [])
        library.max_chars = 0
        self.assertEqual(library.match('parser validation'), [])

    def test_memory_context_manager_closes_connection(self):
        with Memory(Path(self.tmp.name) / 'context.db') as memory:
            self.assertEqual(memory.stats()['facts'], 0)
        with self.assertRaises(sqlite3.ProgrammingError):
            memory.stats()
        memory.close()  # idempotent


if __name__ == '__main__':
    unittest.main()


class RetrievalRankingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.mem = Memory(Path(self.tmp.name) / 'm.sqlite3')
        self.addCleanup(self.mem.close)

    def test_query_sanitizing_survives_operators_and_punctuation(self):
        self.mem.add_fact('parser', 'parse_duration rejects units out of order', 'pitfall', 'global')
        nasty = ['AND', 'OR NEAR(x y)', '"unbalanced', 'a*b - c^ (d', 'NOT parse', 'col:val',
                 "it's parseDuration()!", 'AND OR NOT NEAR', '{}[]<>;--', '']
        for q in nasty:
            with self.subTest(q=q):
                fq = fts_query(q)
                for tok in fq.split(' OR ') if fq else []:
                    self.assertRegex(tok, r'^"[\w]+"$')         # every term quoted, no operators
                self.mem.search_facts(q, 'global')               # must not raise sqlite3.OperationalError
        self.assertEqual(fts_query('AND or NEAR not'), '"near"')  # operators become plain terms or drop
        self.assertIn('"duration"', fts_query('parseDuration'))
        self.assertIn('"duration"', fts_query('parse_duration'))
        hits = self.mem.search_facts("Why does parseDuration() fail? AND/OR", 'global')
        self.assertEqual(len(hits), 1)

    def test_scope_filter(self):
        self.mem.add_fact('tests', 'run pytest with -q in this repo', 'command', 'repo:alpha')
        self.mem.add_fact('tests', 'run pytest under tox in the other repo', 'command', 'repo:beta')
        self.mem.add_fact('tests', 'pytest fixtures live in conftest', 'convention', 'global')
        scopes = {f['scope'] for f in self.mem.search_facts('pytest', 'repo:alpha')}
        self.assertEqual(scopes, {'repo:alpha', 'global'})
        self.assertEqual({f['scope'] for f in self.mem.search_facts('pytest', 'global')}, {'global'})

    def test_rerank_boosts(self):
        a = self.mem.add_fact('cache', 'cache keys include the locale', 'convention', 'global')
        b = self.mem.add_fact('cache', 'cache keys include the locale too', 'convention', 'repo:alpha')
        top = self.mem.search_facts('cache keys locale', 'repo:alpha')
        self.assertEqual(top[0]['id'], b)                         # same-repo boost
        for _ in range(20):                                       # ~20 confirmations outweigh the repo boost
            self.mem.confirm_fact(a, [1])
        top = self.mem.search_facts('cache keys locale', 'repo:alpha')
        self.assertEqual(top[0]['id'], a)                         # log(support) overtakes it
        now = time.time()
        rows = [{'id': 1, 'score': -1.0, 'scope': 'global', 'support': 1, 'updated_at': now, 'kind': 'convention'},
                {'id': 2, 'score': -1.0, 'scope': 'global', 'support': 1, 'updated_at': now - 400 * 86400,
                 'kind': 'convention'},
                {'id': 3, 'score': -1.0, 'scope': 'global', 'support': 1, 'updated_at': now - 400 * 86400,
                 'kind': 'pitfall'}]
        self.assertEqual([r['id'] for r in rerank_facts(rows, 'global', now)], [1, 3, 2])  # recency, pitfall
        rows = [{'id': 1, 'score': -0.2, 'scope': 'global', 'support': 1, 'updated_at': now, 'kind': 'convention'},
                {'id': 2, 'score': -2.0, 'scope': 'global', 'support': 1, 'updated_at': now, 'kind': 'convention'}]
        self.assertEqual(rerank_facts(rows, 'global', now)[0]['id'], 2)                   # BM25 dominates

    def test_transaction_rolls_back_and_index_built_once(self):
        with self.assertRaises(RuntimeError):
            with self.mem.transaction():
                self.mem.add_fact('x', 'written then rolled back', 'pitfall', 'global')
                raise RuntimeError('boom')
        self.assertEqual(self.mem.facts(), [])
        self.assertEqual(self.mem.search_facts('rolled back', 'global'), [])
        with self.mem.transaction():
            fid = self.mem.add_fact('x', 'kept after commit', 'pitfall', 'global')
        self.assertEqual(self.mem.search_facts('kept commit', 'global')[0]['id'], fid)
        self.assertEqual(self.mem.conn.execute('PRAGMA user_version').fetchone()[0], INDEX_VERSION)
        again = Memory(self.mem.path)
        self.addCleanup(again.close)
        self.assertEqual(again.search_facts('kept commit', 'global')[0]['id'], fid)
