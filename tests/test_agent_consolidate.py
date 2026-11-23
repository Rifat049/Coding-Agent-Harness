"""Consolidation: every deterministic filter, the single transaction, and failure handling."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from codebenchkit.agent.consolidate import consolidate
from codebenchkit.agent.llm import ScriptedProvider
from codebenchkit.agent.memory import Memory


def small(reply):
    """A scripted cheap model; `reply` is a str, a dict (JSON-encoded) or an exception."""
    def fn(system, messages, tools, tool_choice):
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)
    return ScriptedProvider(fn, model="small")


class ConsolidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.mem = Memory(Path(self.tmp.name) / "m.sqlite3")
        self.addCleanup(self.mem.close)
        self.fact = self.mem.add_fact("parser", "parse_duration needs units in d h m s order", "pitfall", "repo:demo")
        self.eps = [self.mem.add_episode("repo:demo", f"fix parse_duration case {i}", "success",
                                         notes=["units must appear in d h m s order"]) for i in range(3)]

    def ops(self, *operations, **kw):
        return consolidate(self.mem, small({"operations": list(operations)}), **kw)

    def add(self, content="Always validate unit order before summing durations.", **kw):
        return {"op": "ADD", "subject": "durations", "content": content, "kind": "pitfall",
                "scope": "repo:demo", "evidence": [self.eps[0]], **kw}

    def reasons(self, result):
        return [r["reason"] for r in result["rejected_details"]]

    def test_not_due_until_n_episodes(self):
        mem = Memory(Path(self.tmp.name) / "other.sqlite3")
        self.addCleanup(mem.close)
        mem.add_episode("repo:x", "t", "success")
        r = consolidate(mem, small({"operations": []}), every_n=3)
        self.assertEqual(r["episode_ids"], [])
        self.assertEqual(len(mem.episodes(unconsolidated_only=True)), 1)
        r = consolidate(mem, small({"operations": []}), every_n=3, force=True)
        self.assertEqual(r["episode_ids"], [1])

    def test_all_ops_apply_in_one_transaction_with_audit(self):
        r = self.ops(self.add(),
                     {"op": "CONFIRM", "id": self.fact, "evidence": [self.eps[1]]},
                     {"op": "PROPOSE_SKILL", "name": "Duration parsing", "description": "When parsing compact durations",
                      "body": "1. Tokenize.\n2. Check unit order.\n3. Sum.", "evidence": [self.eps[2]]})
        self.assertEqual((r["applied"], r["rejected"], r["errors"]), (3, 0, 0))
        f = self.mem.get_fact(self.fact)
        self.assertEqual((f["support"], json.loads(f["evidence"])), (2, [self.eps[1]]))
        added = self.mem.search_facts("validate unit order", "repo:demo")[0]
        self.assertEqual((added["source"], json.loads(added["evidence"])), ("consolidation", [self.eps[0]]))
        self.assertEqual(self.mem.episodes(unconsolidated_only=True), [])
        audit = self.mem.conn.execute("SELECT * FROM consolidations").fetchall()
        self.assertEqual(len(audit), 1)
        self.assertEqual(json.loads(audit[0]["episode_ids"]), self.eps)
        self.assertTrue(Path(r["skills"][0]).parts[-3] == "_proposed")
        self.assertEqual(self.mem.skills.all(), [])                       # proposals never auto-load

    def test_update_and_delete(self):
        r = self.ops({"op": "UPDATE", "id": self.fact, "content": "parse_duration requires units ordered d, h, m, s.",
                      "evidence": [self.eps[0]]})
        self.assertEqual(r["applied"], 1)
        self.assertIn("ordered d, h, m, s", self.mem.get_fact(self.fact)["content"])
        for e in self.eps:
            self.mem.add_episode("repo:demo", f"parse_duration again {e}", "success")
        r = self.ops({"op": "DELETE", "id": self.fact, "reason": "superseded"})
        self.assertEqual(r["applied"], 1)
        self.assertEqual(self.mem.get_fact(self.fact)["active"], 0)

    def test_filter_unknown_ids_and_foreign_evidence(self):
        r = self.ops({"op": "CONFIRM", "id": 999, "evidence": [self.eps[0]]},
                     {"op": "UPDATE", "id": "1", "content": "a valid length sentence", "evidence": [self.eps[0]]},
                     self.add(evidence=[12345]), self.add(evidence=[]), self.add(evidence="1"),
                     {"op": "MERGE", "id": self.fact}, "not an object",
                     {"op": "CONFIRM", "id": self.fact, "evidence": [self.eps[0]]},
                     {"op": "DELETE", "id": self.fact, "evidence": [self.eps[0]]})
        reasons = self.reasons(r)
        self.assertEqual(r["applied"], 1)                                 # only the first CONFIRM
        self.assertTrue(any("unknown fact id 999" in x for x in reasons))
        self.assertTrue(any("unknown fact id '1'" in x for x in reasons))
        self.assertTrue(any("outside the batch" in x for x in reasons))
        self.assertTrue(any("evidence is required" in x for x in reasons))
        self.assertTrue(any("list of episode ids" in x for x in reasons))
        self.assertTrue(any("unknown op" in x for x in reasons))
        self.assertTrue(any("not an object" in x for x in reasons))
        self.assertTrue(any("already targeted" in x for x in reasons))

    def test_filter_length_kind_scope_duplicate(self):
        r = self.ops(self.add(content="too short"), self.add(content="x" * 301), self.add(kind="gossip"),
                     self.add(scope="repo:somewhere-else"), self.add(content="parse_duration needs units in d h m s order"),
                     self.add(content="ten chars!"))
        reasons = self.reasons(r)
        self.assertEqual(r["applied"], 1)                                 # exactly-10-character content is allowed
        self.assertEqual(sum("outside 10-300" in x for x in reasons), 2)
        self.assertTrue(any("unknown kind" in x for x in reasons))
        self.assertTrue(any("not global or a batch scope" in x for x in reasons))
        self.assertTrue(any("duplicate of fact" in x for x in reasons))

    def test_filter_secrets(self):
        secrets = ["Use the key sk-ant-api03-abcdefghijklmnop for the client.",
                   "Deploy creds: AKIAABCDEFGHIJKLMNOP is the access key.",
                   "The CI token is ghp_abcdefghijklmnopqrstuvwxyz0123.",
                   "Set password = hunter2hunter2 in the config file.",
                   "Paste -----BEGIN RSA PRIVATE KEY----- into the secret store."]
        r = self.ops(*[self.add(content=s) for s in secrets])
        self.assertEqual(r["applied"], 0)
        self.assertTrue(all("secret" in x for x in self.reasons(r)))

    def test_filter_agent_operating_state(self):
        state = ["The run used 120k tokens before finishing the task.",
                 "DeepSeek hit a rate limit after three consolidation calls.",
                 "The context window filled up on large repos last time.",
                 "The harness aborts after max_iterations is reached.",
                 "A doom loop happened when reading the same file repeatedly.",
                 "Tool errors happen when arguments are not valid JSON objects."]
        r = self.ops(*[self.add(content=s) for s in state])
        self.assertEqual(r["applied"], 0, self.reasons(r))
        self.assertTrue(all("operating state" in x for x in self.reasons(r)))
        self.eps = [self.mem.add_episode("repo:demo", f"auth change {i}", "success") for i in range(3)]
        ok = self.ops(self.add(content="JWT tokens are validated in auth/middleware.py before routing."))
        self.assertEqual(ok["applied"], 1)                                # 'tokens' about the project is fine

    def test_filter_hidden_test_strings_in_benchmark_mode(self):
        hidden = ["parse_duration('1m1h')", "1d2h3m4s"]
        r = self.ops(self.add(content="Remember that parse_duration('1m1h') must raise ValueError."),
                     self.add(content="The input 1D2H3M4S should give 93784 seconds."),
                     {"op": "PROPOSE_SKILL", "name": "durations", "description": "for duration parsing tasks",
                      "body": "Check that 1d2h3m4s parses to the right total.", "evidence": [self.eps[0]]},
                     self.add(content="Validate unit order before summing durations."), hidden_strings=hidden)
        self.assertEqual(r["applied"], 1)
        self.assertEqual(sum("hidden-test" in x for x in self.reasons(r)), 3)

    def test_model_or_json_failure_leaves_batch_unconsolidated(self):
        for reply in (RuntimeError("provider down"), "I think you should remember things.", '{"ops": []}',
                      '{"operations": "none"}'):
            with self.subTest(reply=str(reply)[:30]):
                r = consolidate(self.mem, small(reply), every_n=3)
                self.assertEqual(r["errors"], 1)
                self.assertEqual(len(self.mem.episodes(unconsolidated_only=True)), 3)
                self.assertEqual(self.mem.conn.execute("SELECT COUNT(*) FROM consolidations").fetchone()[0], 0)

    def test_apply_failure_rolls_back_everything(self):
        real = self.mem.confirm_fact
        with patch.object(self.mem, "confirm_fact", side_effect=lambda *a, **k: (_ for _ in ()).throw(OSError("disk"))):
            r = self.ops(self.add(), {"op": "CONFIRM", "id": self.fact, "evidence": [self.eps[0]]})
        self.assertEqual(r["errors"], 1)
        self.assertIn("rolled back", r["error_details"][0])
        self.assertEqual(len(self.mem.facts()), 1)                        # the ADD was rolled back
        self.assertFalse(any("Always validate" in f["content"] for f in self.mem.facts()))
        self.assertEqual(len(self.mem.episodes(unconsolidated_only=True)), 3)
        self.assertEqual(self.mem.conn.execute("SELECT COUNT(*) FROM consolidations").fetchone()[0], 0)
        self.assertEqual(self.mem.get_fact(self.fact)["support"], 1)
        self.assertTrue(callable(real))


# The exact proposal DeepSeek (deepseek-v4-pro) made in the 2026-11-19 tiny_repo smoke run, which
# became the over-generalised global fact F2.
F2_PROPOSAL = {"op": "ADD", "evidence": [1], "content": "When asked to fix a failing test, keep code changes minimal.",
               "subject": "minimal fix preference", "kind": "preference", "scope": "global"}


class ScopeRuleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.mem = Memory(Path(self.tmp.name) / "m.sqlite3")
        self.addCleanup(self.mem.close)

    def run_ops(self, *operations):
        return consolidate(self.mem, small({"operations": list(operations)}), force=True)

    def test_replayed_f2_proposal_ends_up_repo_scoped(self):
        ep = self.mem.add_episode("repo:tiny_repo", "Fix the failing test in this repo. Keep the change minimal.",
                                  "success", files=["stats.py"])
        self.assertEqual(ep, 1)
        r = self.run_ops(F2_PROPOSAL)
        self.assertEqual((r["applied"], r["rejected"]), (1, 0))
        fact = self.mem.get_fact(r["applied_details"][0]["new_id"])
        self.assertEqual(fact["scope"], "repo:tiny_repo")
        self.assertEqual(r["applied_details"][0]["scope_downgraded"]["from"], "global")
        audit = json.loads(self.mem.conn.execute("SELECT applied FROM consolidations").fetchone()[0])
        self.assertEqual(audit[0]["scope"], "repo:tiny_repo")

    def test_default_scope_is_the_evidence_repo(self):
        self.mem.add_episode("repo:a", "task a", "success")
        self.mem.add_episode("repo:b", "task b", "success")
        r = self.run_ops({**F2_PROPOSAL, "scope": None, "evidence": [2], "content": "Run make check before committing."})
        self.assertEqual(self.mem.get_fact(r["applied_details"][0]["new_id"])["scope"], "repo:b")

    def test_global_is_kept_when_evidence_spans_two_repos(self):
        self.mem.add_episode("repo:a", "task a", "success")
        self.mem.add_episode("repo:b", "task b", "success")
        r = self.run_ops({**F2_PROPOSAL, "evidence": [1, 2]})
        self.assertEqual(self.mem.get_fact(r["applied_details"][0]["new_id"])["scope"], "global")
        self.assertNotIn("scope_downgraded", r["applied_details"][0])
        # Same repo twice is still one repo.
        self.mem.add_episode("repo:a", "task a2", "success")
        r = self.run_ops({**F2_PROPOSAL, "evidence": [3], "content": "Prefer small focused diffs in reviews."})
        self.assertEqual(r["applied_details"][0]["scope"], "repo:a")

    def test_repo_scope_must_match_the_evidence(self):
        self.mem.add_episode("repo:a", "task a", "success")
        self.mem.add_episode("repo:b", "task b", "success")
        r = self.run_ops({**F2_PROPOSAL, "scope": "repo:a", "evidence": [2]})
        self.assertEqual(r["applied"], 0)
        self.assertIn("not the scope of its evidence", r["rejected_details"][0]["reason"])

    def test_user_facts_may_be_global_from_one_repo(self):
        fid = self.mem.add_fact("style", "Keep changes minimal when fixing tests.", "preference", "global")
        self.assertEqual((self.mem.get_fact(fid)["scope"], self.mem.get_fact(fid)["source"]), ("global", "user"))

    def test_cross_repo_confirm_only_proposes_promotion(self):
        e1 = self.mem.add_episode("repo:a", "fix the parse_duration unit order bug", "success")
        fid = self.mem.add_fact("durations", "parse_duration requires units in d h m s order", "pitfall",
                                "repo:a", evidence=[e1], source="consolidation")
        self.mem.mark_consolidated([e1])
        e2 = self.mem.add_episode("repo:b", "fix the parse_duration unit order bug again", "success")
        r = self.run_ops({"op": "CONFIRM", "id": fid, "evidence": [e2]})
        self.assertEqual(r["applied"], 1)
        proposal = r["applied_details"][0]["promotion_proposed"]
        self.assertEqual((proposal["repos"], proposal["to"]), (["repo:a", "repo:b"], "global"))
        f = self.mem.get_fact(fid)
        self.assertEqual((f["scope"], f["support"]), ("repo:a", 2))            # not applied
        audit = json.loads(self.mem.conn.execute("SELECT applied FROM consolidations").fetchone()[0])
        self.assertEqual(audit[0]["promotion_proposed"]["proposal_id"], proposal["proposal_id"])
        pending = self.mem.pending_promotions()
        self.assertEqual([(p["fact_id"], json.loads(p["repos"])) for p in pending], [(fid, ["repo:a", "repo:b"])])
        self.assertEqual(self.mem.stats()["pending_promotions"], 1)
        # A second cross-repo confirm refreshes the same proposal instead of adding another.
        e3 = self.mem.add_episode("repo:b", "parse_duration unit order bug yet again", "success")
        self.run_ops({"op": "CONFIRM", "id": fid, "evidence": [e3]})
        pending = self.mem.pending_promotions()
        self.assertEqual((len(pending), json.loads(pending[0]["evidence"])), (1, [e2, e3]))
        # Applied only by a human.
        done = self.mem.promote_fact(fid)
        self.assertEqual(done["status"], "applied")
        self.assertEqual(self.mem.get_fact(fid)["scope"], "global")
        self.assertEqual(self.mem.pending_promotions(), [])
        self.assertIsNone(self.mem.promote_fact(fid))

    def test_same_repo_confirm_does_not_promote(self):
        e1 = self.mem.add_episode("repo:a", "fix the parse_duration unit order bug", "success")
        fid = self.mem.add_fact("durations", "parse_duration requires units in d h m s order", "pitfall",
                                "repo:a", evidence=[e1], source="consolidation")
        self.mem.mark_consolidated([e1])
        e2 = self.mem.add_episode("repo:a", "parse_duration unit order again", "success")
        r = self.run_ops({"op": "CONFIRM", "id": fid, "evidence": [e2]})
        self.assertNotIn("promotion_proposed", r["applied_details"][0])
        self.assertEqual(self.mem.pending_promotions(), [])
        self.assertIsNone(self.mem.promote_fact(fid))
        self.assertEqual(self.mem.get_fact(fid)["scope"], "repo:a")

    def test_another_repos_fact_can_only_be_confirmed(self):
        e1 = self.mem.add_episode("repo:a", "fix the parse_duration unit order bug", "success")
        fid = self.mem.add_fact("durations", "parse_duration requires units in d h m s order", "pitfall",
                                "repo:a", evidence=[e1], source="consolidation")
        self.mem.mark_consolidated([e1])
        e2 = self.mem.add_episode("repo:b", "parse_duration unit order", "success")
        r = self.run_ops({"op": "UPDATE", "id": fid, "content": "parse_duration accepts any unit order.", "evidence": [e2]},
                         {"op": "DELETE", "id": fid, "evidence": [e2]})
        self.assertEqual(r["applied"], 0)
        self.assertTrue(all("can only be confirmed" in x["reason"] for x in r["rejected_details"]))
        self.assertTrue(self.mem.get_fact(fid)["active"])

    def test_quoted_ids_stay_rejected(self):
        """qwen3.5:4b's real reply in the 2026-11-20 smoke run quoted the evidence id. The type
        filter is kept (the prompt now says ids are bare integers); the episode stays pending."""
        self.mem.add_episode("repo:tiny_repo", "Fix the failing test.", "success")
        r = self.run_ops({**F2_PROPOSAL, "scope": "repo:tiny_repo", "evidence": ["1"]})
        self.assertEqual((r["applied"], r["rejected_details"][0]["reason"]), (0, "evidence must be a list of episode ids"))
        self.assertEqual(r["replies"][0].count('"1"'), 1)

    def test_prompt_mentions_the_rule(self):
        from codebenchkit.agent.consolidate import PROMPT
        self.assertIn("2 or more different repos", PROMPT)


class JsonRetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.mem = Memory(Path(self.tmp.name) / "m.sqlite3")
        self.addCleanup(self.mem.close)
        self.mem.add_episode("repo:demo", "fix it", "success")
        self.good = json.dumps({"operations": [{**F2_PROPOSAL, "scope": "repo:demo"}]})

    def sequence(self, *replies):
        seen = []
        def fn(system, messages, tools, tool_choice):
            seen.append(messages[-1]["content"])
            return replies[len(seen) - 1]
        return ScriptedProvider(fn, model="qwen-test"), seen

    def test_invalid_json_is_retried_once_with_a_nudge(self):
        provider, seen = self.sequence("Sure! Here are the ops: [oops", self.good)
        r = consolidate(self.mem, provider, force=True)
        self.assertEqual((r["applied"], r["errors"], r["attempts"]), (1, 0, 2))
        self.assertIn("Return valid JSON only", seen[1])
        self.assertNotIn("Return valid JSON only", seen[0])
        row = self.mem.conn.execute("SELECT model, latency_s, attempts FROM consolidations").fetchone()
        self.assertEqual((row["model"], row["attempts"]), ("scripted:qwen-test", 2))
        self.assertGreaterEqual(row["latency_s"], 0)
        self.assertEqual(len(r["latencies_s"]), 2)

    def test_still_invalid_leaves_episodes_unconsolidated(self):
        for second in ["still not json", '{"ops": []}', '{"operations": "none"}']:
            with self.subTest(second=second):
                provider, seen = self.sequence("{broken", second)
                r = consolidate(self.mem, provider, force=True)
                self.assertEqual((r["errors"], r["attempts"], len(seen)), (1, 2, 2))
                self.assertEqual(len(r["invalid_replies"]), 2)
                self.assertEqual(len(self.mem.episodes(unconsolidated_only=True)), 1)
                self.assertEqual(self.mem.conn.execute("SELECT COUNT(*) FROM consolidations").fetchone()[0], 0)

    def test_valid_first_reply_needs_one_attempt_and_asks_for_json_mode(self):
        calls = []
        class Small(ScriptedProvider):
            def complete(self, prompt, max_tokens=4096, json_mode=False, schema=None, temperature=None):
                calls.append(json_mode and schema is not None and "operations" in schema["properties"]
                             and temperature == 0.0)
                return super().complete(prompt, max_tokens)
        r = consolidate(self.mem, Small(lambda *a: self.good, model="q"), force=True)
        self.assertEqual((r["attempts"], r["applied"], calls), (1, 1, [True]))

    def test_old_database_gains_latency_columns(self):
        import sqlite3
        path = Path(self.tmp.name) / "old.sqlite3"
        c = sqlite3.connect(path)
        c.execute("CREATE TABLE consolidations (id INTEGER PRIMARY KEY, at REAL NOT NULL, model TEXT NOT NULL, "
                  "episode_ids TEXT NOT NULL, applied TEXT NOT NULL, rejected TEXT NOT NULL)")
        c.commit(); c.close()
        m = Memory(path)
        self.addCleanup(m.close)
        cols = {r[1] for r in m.conn.execute("PRAGMA table_info(consolidations)")}
        self.assertTrue({"latency_s", "attempts"} <= cols)


if __name__ == "__main__":
    unittest.main()
