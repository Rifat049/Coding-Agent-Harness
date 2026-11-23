"""Qwen2.5-Coder via Ollama as a main model: alias, determinism, text tool-call recovery. Mocked: no Ollama."""
import os
import unittest
from unittest.mock import patch

from codebenchkit.agent.llm import canonical, make_provider, parse_text_tool_calls
from test_agent_llm import Stub

TOOLS = [{"name": "read_file", "description": "read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}},
         {"name": "finish", "description": "done", "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}}}]


def reply(content="", **message):
    return {"choices": [{"message": {"content": content, **message}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2}}


def native(content="", tool_calls=None, prompt=5):
    """Ollama /api/chat reply shape."""
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"message": msg, "done": True, "done_reason": "stop", "prompt_eval_count": prompt, "eval_count": 2}


class AliasAndDefaults(unittest.TestCase):
    def test_qwen_coder_alias_is_local_qwen_with_coder_default(self):
        with patch.dict(os.environ, {}, clear=True):
            for spec in ("qwen-coder", "qwen-coder:qwen2.5-coder:7b", "qwen:qwen2.5-coder:7b", "qwen"):
                p = make_provider(spec)
                self.assertEqual((p.name, p.model), ("qwen", "qwen2.5-coder:7b"), spec)
            self.assertEqual(canonical("qwen-coder"), "qwen")
            self.assertEqual(make_provider("qwen", role="small").model, "qwen2.5-coder:7b")

    def test_local_qwen_sends_temperature_zero_and_seed(self):
        with patch.dict(os.environ, {}, clear=True):
            p = make_provider("qwen-coder")
        self.assertEqual(p.sampling()["temperature"], 0.0)
        self.assertEqual(p.sampling()["seed"], 0)
        with Stub([(200, native("hi"))]) as stub:
            p.base_url = stub.url + "/v1"
            p.chat("sys", [{"role": "user", "content": "x"}], TOOLS)
        path, body = stub.requests[0]
        self.assertEqual(path, "/api/chat")
        self.assertEqual((body["options"]["temperature"], body["options"]["seed"]), (0.0, 0))
        self.assertEqual(body["options"]["num_ctx"], 16384)
        self.assertIs(body["think"], False)

    def test_seed_and_temperature_overridable_and_droppable(self):
        with patch.dict(os.environ, {"QWEN_SEED": "7", "QWEN_TEMPERATURE": ""}, clear=True):
            p = make_provider("qwen-coder")
        with Stub([(200, native("hi"))]) as stub:
            p.base_url = stub.url + "/v1"
            p.chat("", [{"role": "user", "content": "x"}])
        opts = stub.requests[0][1]["options"]
        self.assertEqual(opts["seed"], 7)
        self.assertNotIn("temperature", opts)

    def test_deepseek_sends_no_temperature_or_seed_and_says_why(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "k"}, clear=True):
            p = make_provider("deepseek")
        with Stub([(200, reply("hi"))]) as stub:
            p.base_url = stub.url
            p.chat("", [{"role": "user", "content": "x"}], TOOLS)
        body = stub.requests[0][1]
        self.assertNotIn("temperature", body)
        self.assertNotIn("seed", body)
        self.assertIn("thinking mode", p.sampling()["note"])

    def test_native_structured_output_carries_the_seed(self):
        with patch.dict(os.environ, {}, clear=True):
            p = make_provider("qwen", role="small")
        with Stub([(200, {"message": {"content": '{"a": 1}'}, "prompt_eval_count": 3, "eval_count": 2})]) as stub:
            p.base_url = stub.url + "/v1"
            out = p.complete("x", schema={"type": "object"}, temperature=0)
        path, body = stub.requests[0]
        self.assertEqual(path, "/api/chat")
        self.assertEqual(body["options"]["seed"], 0)
        self.assertEqual(body["options"]["temperature"], 0)
        self.assertEqual(out, '{"a": 1}')
        self.assertEqual(p.last_mode, "format schema (ollama native)")


class TextToolCalls(unittest.TestCase):
    def test_recovers_the_shape_qwen_coder_actually_returns(self):
        # Verbatim shape of a real qwen2.5-coder:7b reply on Ollama 0.40.16 (2026-11-20).
        with patch.dict(os.environ, {}, clear=True):
            p = make_provider("qwen-coder")
        with Stub([(200, native('{"name": "read_file", "arguments": {"path": "utils.py"}}'))]) as stub:
            p.base_url = stub.url + "/v1"
            r = p.chat("", [{"role": "user", "content": "x"}], TOOLS)
        self.assertEqual([(c.name, c.args) for c in r.tool_calls], [("read_file", {"path": "utils.py"})])
        self.assertEqual((r.tool_call_source, r.text), ("text", ""))

    def test_native_tool_calls_are_marked_native(self):
        with patch.dict(os.environ, {}, clear=True):
            p = make_provider("qwen-coder")
        wire = native("", [{"function": {"name": "finish", "arguments": {"summary": "ok"}}}])
        with Stub([(200, wire)]) as stub:
            p.base_url = stub.url + "/v1"
            r = p.chat("", [{"role": "user", "content": "x"}], TOOLS)
        self.assertEqual(r.tool_call_source, "native")
        self.assertEqual(r.tool_calls[0].args, {"summary": "ok"})

    def test_unparseable_tool_text_is_flagged_not_executed(self):
        with patch.dict(os.environ, {}, clear=True):
            p = make_provider("qwen-coder")
        with Stub([(200, native('{"name": "read_file", "arguments": {"path": '))]) as stub:
            p.base_url = stub.url + "/v1"
            r = p.chat("", [{"role": "user", "content": "x"}], TOOLS)
        self.assertEqual(r.tool_calls, [])
        self.assertIn("looked like a tool call", r.text_tool_error)

    def test_recovery_can_be_switched_off(self):
        with patch.dict(os.environ, {"CBK_TEXT_TOOL_CALLS": "0"}, clear=True):
            p = make_provider("qwen-coder")
        with Stub([(200, native('{"name": "finish", "arguments": {"summary": "x"}}'))]) as stub:
            p.base_url = stub.url + "/v1"
            r = p.chat("", [{"role": "user", "content": "x"}], TOOLS)
        self.assertEqual(r.tool_calls, [])

    def test_openai_compat_path_recovers_too(self):
        with patch.dict(os.environ, {}, clear=True):
            p = make_provider("qwen-coder")
        p.num_ctx = None                     # e.g. a hosted endpoint: /v1 chat/completions
        with Stub([(200, reply('{"name": "read_file", "arguments": {"path": "u.py"}}'))]) as stub:
            p.base_url = stub.url + "/v1"
            r = p.chat("", [{"role": "user", "content": "x"}], TOOLS)
        self.assertEqual(stub.requests[0][0], "/v1/chat/completions")
        self.assertEqual((r.tool_call_source, r.tool_calls[0].args), ("text", {"path": "u.py"}))

    def test_dropped_calls_are_reported_even_when_others_parse(self):
        names = {"write_file", "finish"}
        text = ('```json\n{"name": "write_file", "arguments": {"path": "a.py", "content": "x = 1\n}}\n```\n'
                '```json\n{"name": "finish", "arguments": {"summary": "ok"}}\n```')
        calls, err = parse_text_tool_calls(text, names)
        self.assertEqual([c.name for c in calls], ["finish"])
        self.assertIn("1 of 2 tool calls", err)

    def test_raw_text_kept_for_the_trace(self):
        with patch.dict(os.environ, {}, clear=True):
            p = make_provider("qwen-coder")
        raw = 'Reading it now.\n{"name": "read_file", "arguments": {"path": "a"}}'
        with Stub([(200, native(raw))]) as stub:
            p.base_url = stub.url + "/v1"
            r = p.chat("", [{"role": "user", "content": "x"}], TOOLS)
        self.assertEqual((r.text, r.raw_text), ("", raw))

    def test_tool_name_then_json_is_flagged_not_run(self):
        calls, err = parse_text_tool_calls('<tool_response>\nedit_file\n{"path": "a.py", "old_string": "x"}', {"edit_file"})
        self.assertEqual(calls, [])
        self.assertIn("edit_file followed by arguments", err)
        self.assertEqual(parse_text_tool_calls("I will use edit_file next.", {"edit_file"}), ([], ""))

    def test_parser_shapes(self):
        names = {"read_file", "finish"}
        tagged = '<tool_call>\n{"name": "finish", "arguments": "{\\"summary\\": \\"done\\"}"}\n</tool_call>'
        self.assertEqual(parse_text_tool_calls(tagged, names)[0][0].args, {"summary": "done"})
        fenced = '```json\n[{"name": "read_file", "arguments": {"path": "a"}}, {"name": "read_file", "parameters": {"path": "b"}}]\n```'
        self.assertEqual([c.args["path"] for c in parse_text_tool_calls(fenced, names)[0]], ["a", "b"])
        # Unknown tool names and ordinary code are left alone.
        self.assertEqual(parse_text_tool_calls('{"name": "rm_rf", "arguments": {}}', names)[0], [])
        self.assertEqual(parse_text_tool_calls('def f():\n    return {"a": 1}', names), ([], ""))
        bad = parse_text_tool_calls('{"name": "read_file", "arguments": 5}', names)[0][0]
        self.assertIn("_invalid_arguments", bad.args)


class NativeOllamaWire(unittest.TestCase):
    def test_history_wire_shape_and_tools_off_final_answer(self):
        with patch.dict(os.environ, {}, clear=True):
            p = make_provider("qwen-coder")
        history = [{"role": "user", "content": "go"},
                   {"role": "assistant", "content": "", "tool_calls": [{"id": "t", "name": "read_file", "args": {"path": "a"}}]},
                   {"role": "tool", "tool_call_id": "t", "name": "read_file", "content": "body"}]
        with Stub([(200, native("final"))]) as stub:
            p.base_url = stub.url + "/v1"
            p.chat("sys", history, TOOLS, tool_choice="none")
        body = stub.requests[0][1]
        self.assertNotIn("tools", body)
        self.assertEqual(body["messages"][0], {"role": "system", "content": "sys"})
        self.assertEqual(body["messages"][2]["tool_calls"][0]["function"], {"name": "read_file", "arguments": {"path": "a"}})
        self.assertEqual(body["messages"][3], {"role": "tool", "content": "body", "tool_name": "read_file"})

    def test_full_context_is_flagged(self):
        with patch.dict(os.environ, {"QWEN_NUM_CTX": "4096"}, clear=True):
            p = make_provider("qwen-coder")
        with Stub([(200, native("x", prompt=4090))]) as stub:
            p.base_url = stub.url + "/v1"
            r = p.chat("", [{"role": "user", "content": "x"}])
        self.assertTrue(r.context_full)
        self.assertEqual(p.sampling()["num_ctx"], 4096)


class OllamaFailures(unittest.TestCase):
    def test_server_down_is_fatal_with_fix_command(self):
        from codebenchkit.agent.llm import FatalProviderError
        with patch.dict(os.environ, {"QWEN_BASE_URL": "http://127.0.0.1:9/v1"}, clear=True):
            p = make_provider("qwen-coder")
        with patch("codebenchkit.agent.llm.time.sleep"), self.assertRaises(FatalProviderError) as cm:
            p.chat("", [{"role": "user", "content": "x"}], TOOLS)
        self.assertIn("ollama serve", cm.exception.reason())
        self.assertIn("ollama pull qwen2.5-coder:7b", cm.exception.reason())

    def test_missing_model_is_fatal_even_in_structured_output(self):
        from codebenchkit.agent.llm import FatalProviderError
        with patch.dict(os.environ, {}, clear=True):
            p = make_provider("qwen:qwen9:99b", role="small")
        with Stub([(404, {"error": "model 'qwen9:99b' not found"})]) as stub:
            p.base_url = stub.url + "/v1"
            with self.assertRaises(FatalProviderError) as cm:
                p.complete("x", schema={"type": "object"})
        self.assertEqual(cm.exception.reason(), "Ollama has no model 'qwen9:99b': run `ollama pull qwen9:99b`")
        self.assertNotIn("format", p.dropped)


if __name__ == "__main__":
    unittest.main()
