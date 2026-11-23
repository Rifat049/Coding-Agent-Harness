"""Default model setup: DeepSeek main, Qwen small, config file, keys. Local servers only."""
import argparse
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from codebenchkit.agent import cli
from codebenchkit.agent.llm import LLMResponse, OpenAICompatProvider, ScriptedProvider, make_provider, missing_key
from codebenchkit.agent.loop import AgentConfig
from codebenchkit.agent.runner import run_agent
from codebenchkit.agent.llm import ToolCall
from test_agent_llm import Stub


class ModelsServer:
    """GET /v1/models only, like Ollama's OpenAI-compatible endpoint."""
    def __init__(self, ids):
        self.ids = ids

    def __enter__(self):
        ids = self.ids
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps({"object": "list", "data": [{"id": i} for i in ids]}).encode()
                self.send_response(200 if self.path == "/v1/models" else 404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *a):
                pass
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        return self

    def __exit__(self, *a):
        self.server.shutdown()
        self.server.server_close()


def ok(content="", **message):
    return {"choices": [{"message": {"content": content, **message}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2}}


class QwenProviderTests(unittest.TestCase):
    def test_qwen_defaults_to_local_ollama_without_key(self):
        with patch.dict(os.environ, {}, clear=True):
            p = make_provider("qwen", role="small")
            self.assertEqual((p.name, p.base_url, p.model, p.api_key), ("qwen", "http://localhost:11434/v1", "qwen2.5-coder:7b", ""))
            self.assertEqual(p.extra_body, {"reasoning_effort": "none"})
            self.assertIsNone(missing_key("qwen"))
        with patch.dict(os.environ, {"QWEN_MODEL": "qwen3:8b"}, clear=True):
            self.assertEqual(make_provider("qwen").model, "qwen3:8b")
            self.assertEqual(make_provider("qwen:qwen3.5:9b").model, "qwen3.5:9b")

    def test_qwen_hosted_endpoint_uses_dashscope_key_and_no_ollama_extras(self):
        env = {"QWEN_BASE_URL": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
               "QWEN_MODEL": "qwen-plus", "DASHSCOPE_API_KEY": "sk-test"}
        with patch.dict(os.environ, env, clear=True):
            p = make_provider("qwen", role="small")
            self.assertEqual((p.api_key, p.model, p.extra_body), ("sk-test", "qwen-plus", {}))

    def test_json_mode_sends_response_format_and_extras(self):
        with Stub([(200, ok('{"operations": []}'))]) as stub, patch.dict(os.environ, {"QWEN_BASE_URL": "http://127.0.0.1:1/v1"}):
            p = make_provider("qwen", role="small")
            p.base_url = stub.url
            self.assertEqual(p.complete("return json", max_tokens=50, json_mode=True), '{"operations": []}')
            _, body = stub.requests[0]
            self.assertEqual(body["response_format"], {"type": "json_object"})
            self.assertEqual((body["reasoning_effort"], body["max_tokens"]), ("none", 50))
            self.assertNotIn("Authorization", {k.title(): v for k, v in stub.headers[0].items()})

    def test_server_rejecting_optional_extras_drops_response_format_first(self):
        with Stub([(400, {"error": "unknown field reasoning_effort"}), (400, {"error": "still"}), (200, ok("{}"))]) as stub:
            p = OpenAICompatProvider("m", stub.url, name="qwen", require_key=False, retries=0,
                                     extra_body={"reasoning_effort": "none"})
            p.complete("x", json_mode=True)
            self.assertIn("response_format", stub.requests[0][1])
            self.assertEqual(("response_format" in stub.requests[1][1], stub.requests[1][1].get("reasoning_effort")), (False, "none"))
            self.assertNotIn("reasoning_effort", stub.requests[2][1])
            self.assertEqual(sorted(p.dropped), ["reasoning_effort", "response_format"])
        # Only JSON mode unsupported (Ollama 501): thinking stays off.
        with Stub([(501, {"error": "structured output is unavailable"}), (200, ok("{}")), (200, ok("{}"))]) as stub:
            p = OpenAICompatProvider("m", stub.url, name="qwen", require_key=False, retries=0,
                                     extra_body={"reasoning_effort": "none"})
            p.complete("x", json_mode=True)
            p.complete("y", json_mode=True)
            self.assertEqual([r[1].get("reasoning_effort") for r in stub.requests], ["none", "none", "none"])
            self.assertEqual(["response_format" in r[1] for r in stub.requests], [True, False, False])
            self.assertEqual(list(p.dropped), ["response_format"])
        # Ollama's answer for models without structured output: not retried, then remembered.
        with Stub([(501, {"error": {"message": "structured output is unavailable"}}), (200, ok("{}")), (200, ok("{}"))]) as stub:
            p = OpenAICompatProvider("m", stub.url, name="qwen", require_key=False, retries=3)
            p.complete("x", json_mode=True)
            p.complete("y", json_mode=True)
            self.assertEqual(len(stub.requests), 3)                 # 501 once, no backoff retries
            self.assertNotIn("response_format", stub.requests[2][1])
            self.assertIn("structured output is unavailable", p.dropped["response_format"])
            from codebenchkit.agent.consolidate import _json_mode
            self.assertTrue(_json_mode(p).startswith("prompt only; response_format rejected (HTTP 501"))
        with Stub([(400, {"error": "bad"})]) as stub:
            with self.assertRaises(Exception):
                OpenAICompatProvider("m", stub.url, require_key=False, retries=0).chat("", [{"role": "user", "content": "x"}])
            self.assertEqual(len(stub.requests), 1)       # nothing optional to drop: no retry


class OllamaFormatTests(unittest.TestCase):
    SCHEMA = {"type": "object", "properties": {"operations": {"type": "array"}}, "required": ["operations"]}

    def provider(self, url):
        with patch.dict(os.environ, {"QWEN_BASE_URL": url + "/v1"}):
            p = make_provider("qwen", role="small")
        p.base_url = url + "/v1"
        return p

    def test_schema_goes_to_native_format_with_thinking_off(self):
        with Stub([(200, {"message": {"role": "assistant", "content": '{"operations": []}'}})]) as stub:
            p = self.provider(stub.url)
            p.ollama_native = True
            self.assertEqual(p.complete("x", 300, json_mode=True, schema=self.SCHEMA), '{"operations": []}')
            path, body = stub.requests[0]
            self.assertEqual(path, "/api/chat")
            self.assertEqual((body["format"], body["think"], body["stream"], body["options"]["num_predict"]),
                             (self.SCHEMA, False, False, 300))
            self.assertEqual(p.last_mode, "format schema (ollama native)")

    def test_501_falls_back_to_v1_once_and_is_remembered(self):
        unavailable = {"error": "structured output is unavailable"}
        with Stub([(501, unavailable), (501, unavailable), (200, ok("{}")), (200, ok("{}"))]) as stub:
            p = self.provider(stub.url)
            p.ollama_native = True
            p.complete("x", json_mode=True, schema=self.SCHEMA)
            p.complete("y", json_mode=True, schema=self.SCHEMA)
            paths = [r[0] for r in stub.requests]
            self.assertEqual(paths, ["/api/chat", "/v1/chat/completions", "/v1/chat/completions", "/v1/chat/completions"])
            self.assertEqual(stub.requests[3][1].get("reasoning_effort"), "none")   # thinking stays off
            self.assertIn("structured output is unavailable", p.dropped["format"])
            from codebenchkit.agent.consolidate import _json_mode
            self.assertEqual(_json_mode(p).split(";")[0], "prompt only")
            self.assertIn("format rejected (HTTP 501", _json_mode(p))

    def test_consolidation_records_small_model_usage_and_cost(self):
        import tempfile as tf
        from codebenchkit.agent.consolidate import consolidate
        from codebenchkit.agent.memory import Memory
        reply = ok('{"operations": []}')
        reply["usage"] = {"prompt_tokens": 1000, "completion_tokens": 100, "prompt_cache_hit_tokens": 200}
        with Stub([(200, reply)]) as stub, tf.TemporaryDirectory() as d, patch.dict(os.environ, {"DEEPSEEK_API_KEY": "k"}):
            p = make_provider("deepseek", role="small")
            p.base_url = stub.url
            mem = Memory(Path(d) / "m.sqlite3")
            mem.add_episode("repo:x", "task", "success")
            r = consolidate(mem, p, force=True)
            mem.close()
        self.assertEqual(r["usage"][0]["input_tokens"], 1000)
        self.assertGreater(r["cost_usd"], 0)
        self.assertEqual(stub.requests[0][1]["response_format"], {"type": "json_object"})

    def test_temperature_reaches_both_ollama_paths_and_is_dropped_if_refused(self):
        with Stub([(200, {"message": {"content": "{}"}})]) as stub:
            p = self.provider(stub.url)
            p.ollama_native = True
            p.complete("x", schema=self.SCHEMA, temperature=0.0)
            self.assertEqual(stub.requests[0][1]["options"]["temperature"], 0.0)
        with Stub([(400, {"error": "temperature is not supported"}), (400, {"error": "still"}), (200, ok("{}")),
                   (200, ok("{}"))]) as stub:
            p = OpenAICompatProvider("m", stub.url, name="x", require_key=False, retries=0)
            p.complete("x", json_mode=True, temperature=0.0)
            p.complete("y", temperature=0.0)
            self.assertEqual([r[1].get("temperature") for r in stub.requests], [0.0, 0.0, None, None])
            self.assertIn("temperature", p.dropped)

    def test_non_local_qwen_never_uses_native_format(self):
        with patch.dict(os.environ, {"QWEN_BASE_URL": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"}):
            self.assertFalse(make_provider("qwen").ollama_native)


class DeepSeekReasoningTests(unittest.TestCase):
    def test_reasoning_content_is_echoed_within_and_across_turns(self):
        """https://api-docs.deepseek.com/guides/thinking_mode#tool-calls: with tools, every later
        request must carry every earlier reasoning_content, including after a new user turn."""
        spec = {"name": "read_file", "description": "Read", "parameters": {"type": "object"}}
        turn1 = ok("", reasoning_content="think A", tool_calls=[{"id": "c1", "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "a"}'}}])
        turn1["choices"][0]["finish_reason"] = "tool_calls"
        responses = [(200, turn1), (200, ok("done", reasoning_content="think B")), (200, ok("again"))]
        with Stub(responses) as stub, patch.dict(os.environ, {"DEEPSEEK_API_KEY": "k"}):
            p = make_provider("deepseek")
            p.base_url = stub.url
            msgs = [{"role": "user", "content": "q1"}]
            r = p.chat("sys", msgs, [spec]); msgs.append(r.as_message())
            msgs.append({"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "x"})
            r = p.chat("sys", msgs, [spec]); msgs.append(r.as_message())
            msgs.append({"role": "user", "content": "q2"})           # a new user turn
            p.chat("sys", msgs, [spec])
        within = [m for m in stub.requests[1][1]["messages"] if m["role"] == "assistant"]
        across = [m for m in stub.requests[2][1]["messages"] if m["role"] == "assistant"]
        self.assertEqual([m.get("reasoning_content") for m in within], ["think A"])
        self.assertEqual([m.get("reasoning_content") for m in across], ["think A", "think B"])

    def test_final_answer_after_guardrail_keeps_reasoning(self):
        def fn(system, messages, tools, tool_choice):
            if tool_choice == "none":
                return LLMResponse("summary", raw={"provider": "openai-compat", "reasoning_content": "why"})
            return LLMResponse("", [ToolCall(f"c{len(messages)}", "list_files", {})])
        with tempfile.TemporaryDirectory() as d:
            res = run_agent("look", d, ScriptedProvider(fn), AgentConfig(max_iterations=1),
                            trace_path=Path(d) / "t.jsonl")
        last = res.history[-1]
        self.assertEqual((last["role"], last["content"]), ("assistant", "summary"))
        self.assertEqual(last["raw"]["reasoning_content"], "why")


class ReasoningLoggingTests(unittest.TestCase):
    def test_input_transformations_absent_empty_or_list(self):
        from codebenchkit.agent.llm import AnthropicProvider
        base = {"content": [{"type": "thinking", "thinking": "t", "signature": "s"},
                            {"type": "redacted_thinking", "data": "x"}, {"type": "text", "text": "hi"}]}
        absent = AnthropicProvider.from_wire(base)
        empty = AnthropicProvider.from_wire({**base, "input_transformations": []})
        listed = AnthropicProvider.from_wire({**base, "input_transformations": [{"type": "thinking_dropped", "index": 1}]})
        self.assertEqual([r.transformations_status for r in (absent, empty, listed)], ["absent", "empty", "list"])
        self.assertEqual(absent.thinking_stats()["thinking_blocks"], 2)
        self.assertEqual(absent.thinking_stats()["redacted_thinking_blocks"], 1)
        ds = OpenAICompatProvider.from_wire({"choices": [{"message": {"content": "a", "reasoning_content": "abcde"}}],
                                             "usage": {"completion_tokens_details": {"reasoning_tokens": 3}}})
        self.assertEqual((ds.transformations_status, ds.thinking_stats()["reasoning_chars"], ds.reasoning_tokens), ("absent", 5, 3))

    def trace(self, fn, **cfg):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "a.py").write_text("x = 1\n" * 200)
            res = run_agent("look", d, ScriptedProvider(fn), AgentConfig(**cfg), trace_path=Path(d) / "t.jsonl")
            return res, [json.loads(l) for l in (Path(d) / "t.jsonl").read_text().splitlines()]

    def test_trace_records_status_and_reasoning_per_call(self):
        def fn(system, messages, tools, tool_choice):
            n = sum(m["role"] == "assistant" for m in messages)
            raw = {"provider": "openai-compat", "reasoning_content": "r" * (10 * (n + 1))}
            if n == 0:
                return LLMResponse("", [ToolCall("c1", "list_files", {})], raw=raw)
            return LLMResponse("done", raw=raw)
        res, events = self.trace(fn, max_iterations=4)
        calls = [e for e in events if e["event"] == "llm_call"]
        self.assertEqual([c["input_transformations"] for c in calls], ["absent", "absent"])
        self.assertEqual([c["reasoning_chars"] for c in calls], [10, 20])
        self.assertEqual([c["thinking_blocks"] for c in calls], [0, 0])
        self.assertEqual(res.receipt["reasoning_chars"], 30)

    def test_compaction_is_skipped_when_the_summary_is_not_smaller(self):
        from codebenchkit.agent.history import compact_history
        msgs = [{"role": "user", "content": "task"}] + [
            m for i in range(3) for m in ({"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "name": "list_files", "args": {}}]},
                                          {"role": "tool", "tool_call_id": f"c{i}", "name": "list_files", "content": "a"})] + [
            {"role": "user", "content": "go on"}]
        before = json.dumps(msgs)
        big = lambda prompt: "x" * 4000          # a "summary" far larger than what it replaces
        rep = compact_history(msgs, context_limit=10, summarize=big, threshold=0.0)
        self.assertEqual((rep["compacted"], rep["reason"]), (False, "summary_not_smaller"))
        self.assertEqual(json.dumps(msgs), before)                # history untouched
        rep = compact_history(msgs, context_limit=10, summarize=big, threshold=0.0, require_smaller=False)
        self.assertTrue(rep["compacted"])

    def test_forced_compaction_runs_exactly_once(self):
        def fn(system, messages, tools, tool_choice):
            n = sum(m["role"] == "assistant" for m in messages)
            if n < 5:
                return LLMResponse("", [ToolCall(f"c{n}", "read_file", {"path": "a.py", "start_line": n + 1})])
            return LLMResponse("done")
        res, events = self.trace(fn, max_iterations=10, trim_chars=200, trim_keep=100, force_compaction_after=2)
        compactions = [e for e in events if e["event"] == "compaction"]
        self.assertEqual([c.get("forced") for c in compactions], [True])
        self.assertTrue(compactions[0]["compacted"])
        self.assertEqual(res.receipt["compactions"], 1)
        self.assertGreater(res.receipt["trims"], 0)


class KeyTests(unittest.TestCase):
    def test_only_cbk_anthropic_key_is_read(self):
        with patch.dict(os.environ, {"CBK_ANTHROPIC_API_KEY": "cbk", "ANTHROPIC_API_KEY": "plain"}, clear=True):
            self.assertEqual(make_provider("claude").api_key, "cbk")
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "plain"}, clear=True):
            self.assertEqual(make_provider("claude").api_key, "")
            self.assertEqual(missing_key("claude"), "CBK_ANTHROPIC_API_KEY")
            from codebenchkit.agent.llm import AnthropicProvider, ProviderError
            with self.assertRaisesRegex(ProviderError, "ignored on purpose"):
                AnthropicProvider("claude-sonnet-5-5").chat("", [{"role": "user", "content": "x"}])
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(missing_key("claude"), "CBK_ANTHROPIC_API_KEY")
            self.assertEqual(missing_key("deepseek"), "DEEPSEEK_API_KEY")


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / "repo"
        (self.repo / ".cbk").mkdir(parents=True)
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        env = patch.dict(os.environ, {"CBK_HOME": str(self.home)})
        env.start()
        self.addCleanup(env.stop)
        cwd = os.getcwd()
        os.chdir(self.tmp.name)                    # no ./.cbk/config.toml from the project
        self.addCleanup(os.chdir, cwd)

    def args(self, **kw):
        base = dict(repo=self.repo, model=None, small_model=None, verify=None, approval=None, gate=None,
                    consolidation_every=None)
        return argparse.Namespace(**{**base, **kw})

    def test_builtin_defaults_are_deepseek_and_qwen(self):
        a = self.args()
        cli.resolve_settings(a)
        self.assertEqual((a.model, a.small_model, a.approval, a.gate, a.consolidation_every, a.config_path),
                         ("deepseek", "auto", "auto", "heuristic", 3, None))

    def test_repo_config_then_flags_override(self):
        (self.repo / ".cbk" / "config.toml").write_text(
            '[models]\nmain = "claude"\nsmall = "qwen:qwen3:8b"\n[agent]\nverify = "pytest -q"\nconsolidation_every = 2\n')
        (self.home / "config.toml").write_text('[models]\nmain = "chatgpt"\n')
        a = self.args()
        cli.resolve_settings(a)
        self.assertEqual((a.model, a.small_model, a.verify, a.consolidation_every), ("claude", "qwen:qwen3:8b", "pytest -q", 2))
        a = self.args(model="deepseek", verify="make test")
        cli.resolve_settings(a)
        self.assertEqual((a.model, a.small_model, a.verify), ("deepseek", "qwen:qwen3:8b", "make test"))

    def test_home_config_is_the_last_fallback(self):
        (self.home / "config.toml").write_text('[models]\nmain = "chatgpt"\n')
        a = self.args()
        cli.resolve_settings(a)
        self.assertEqual((a.model, a.small_model), ("chatgpt", "auto"))

    def test_mock_main_never_gets_a_real_small_model(self):
        (self.repo / ".cbk" / "config.toml").write_text('[models]\nsmall = "deepseek"\n')
        a = self.args(model="mock")
        cli.resolve_settings(a)
        self.assertEqual(a.small_model, "mock")

    def test_bad_config_is_an_error(self):
        for text in ['[models\n', '[agent]\napproval = "yolo"\n', '[agent]\nconsolidation_every = 0\n', '[models]\nmain = 3\n']:
            with self.subTest(text=text):
                (self.repo / ".cbk" / "config.toml").write_text(text)
                with self.assertRaises(ValueError):
                    cli.resolve_settings(self.args())

    def test_project_config_file_parses_to_deepseek_and_qwen(self):
        root = Path(__file__).resolve().parents[1]
        cfg, path = cli.load_config(root)
        self.assertEqual((cfg.get("model"), cfg.get("small_model")), ("deepseek", "auto"))


class SmallAutoTests(unittest.TestCase):
    def unavailable(self, available):
        return lambda spec: None if spec in available else f"{spec} is down"

    def test_auto_order_local_qwen_then_deepseek_then_dashscope(self):
        for available, want in (({"qwen", "deepseek", "dashscope"}, "qwen"), ({"deepseek", "dashscope"}, "deepseek"),
                                ({"dashscope"}, "dashscope")):
            with self.subTest(available=available), patch.object(cli, "small_unavailable", self.unavailable(available)):
                spec, note = cli.resolve_small("auto")
                self.assertEqual(spec, want)
                self.assertTrue(note.startswith(f"auto -> {want}"))
        with patch.object(cli, "small_unavailable", self.unavailable({"deepseek"})):
            self.assertIn("skipped qwen: qwen is down", cli.resolve_small("auto")[1])

    def test_auto_none_is_a_warning_not_an_error(self):
        with patch.object(cli, "small_unavailable", self.unavailable(set())):
            spec, note = cli.resolve_small("auto")
        self.assertIsNone(spec)
        self.assertIn("auto -> none", note)
        for name in ("qwen", "deepseek", "dashscope"):
            self.assertIn(f"{name}: {name} is down", note)

    def test_explicit_spec_is_checked_as_is(self):
        with patch.object(cli, "small_unavailable", self.unavailable({"deepseek"})):
            self.assertEqual(cli.resolve_small("deepseek"), ("deepseek", "deepseek (explicit)"))
            self.assertIsNone(cli.resolve_small("qwen")[0])
        self.assertEqual(cli.resolve_small("mock"), ("mock", "mock"))

    def test_dashscope_provider_needs_its_key(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(missing_key("dashscope"), "DASHSCOPE_API_KEY")
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "k", "DASHSCOPE_MODEL": "qwen-max"}, clear=True):
            p = make_provider("dashscope", role="small")
            self.assertEqual((p.name, p.model, p.api_key), ("dashscope", "qwen-max", "k"))
            self.assertIn("dashscope", p.base_url)

    def test_receipt_records_the_choice(self):
        def fn(system, messages, tools, tool_choice):
            return LLMResponse("done")
        with tempfile.TemporaryDirectory() as d:
            res = run_agent("look", d, ScriptedProvider(fn), AgentConfig(small_model_note="auto -> none; qwen: down"),
                            trace_path=Path(d) / "t.jsonl")
        self.assertEqual((res.receipt["small_model"], res.receipt["small_model_note"]), (None, "auto -> none; qwen: down"))


class SmallAvailabilityTests(unittest.TestCase):
    def test_unreachable_local_qwen_disables_small_with_pull_hint(self):
        with patch.dict(os.environ, {"QWEN_BASE_URL": "http://127.0.0.1:9/v1", "QWEN_MODEL": "qwen3.5:4b"}):
            why = cli.small_unavailable("qwen")
        self.assertIn("ollama pull qwen3.5:4b", why)

    def test_missing_model_tag_and_present_model(self):
        with ModelsServer(["llama3:8b"]) as srv, patch.dict(os.environ, {"QWEN_BASE_URL": srv.url.replace("127.0.0.1", "localhost")}):
            self.assertIn("ollama pull qwen2.5-coder:7b", cli.small_unavailable("qwen"))
        with ModelsServer(["qwen2.5-coder:7b", "qwen3.5:4b"]) as srv, patch.dict(os.environ, {"QWEN_BASE_URL": srv.url.replace("127.0.0.1", "localhost")}):
            self.assertIsNone(cli.small_unavailable("qwen"))
            self.assertIsNone(cli.small_unavailable("qwen:qwen3.5:4b"))

    def test_missing_deepseek_key(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(cli.small_unavailable("deepseek"), "DEEPSEEK_API_KEY is not set")


if __name__ == "__main__":
    unittest.main()
