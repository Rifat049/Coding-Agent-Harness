"""Provider contract tests use a local HTTP server, never external services."""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from codebenchkit.agent.llm import (
    AnthropicProvider, OpenAICompatProvider, ProviderError, _post_json,
    make_provider, parse_json_object,
)


class Stub:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.headers = []

    def __enter__(self):
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                owner.requests.append((self.path, json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
                owner.headers.append({k.lower(): v for k, v in self.headers.items()})
                status, data = owner.responses.pop(0)
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())
            def log_message(self, *args):
                pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self.thread.start()
        self.url = 'http://127.0.0.1:%s' % self.server.server_port
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class ProviderTests(unittest.TestCase):
    def test_anthropic_caches_only_stable_prefix_and_marks_errors(self):
        class Prompt(str):
            stable_prefix = 'stable rules'
        system = Prompt('stable rules\nchanging environment')
        messages = [
            {'role': 'user', 'content': 'one'},
            {'role': 'user', 'content': 'two'},
            {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 't1', 'name': 'read_file', 'args': {'path': 'x'}}]},
            {'role': 'tool', 'tool_call_id': 't1', 'content': 'ERROR: missing'},
            {'role': 'user', 'content': 'try again'},
        ]
        with Stub([(200, {'content': [{'type': 'text', 'text': 'ok'}], 'usage': {'input_tokens': 2, 'output_tokens': 3, 'cache_read_input_tokens': 4, 'cache_creation_input_tokens': 5}})]) as stub:
            result = AnthropicProvider('test', 'test-key', stub.url).chat(system, messages)
        path, body = stub.requests[0]
        self.assertEqual(path, '/v1/messages')
        # Stable rules and the (frozen-per-run) context each end in a cache breakpoint.
        self.assertEqual(body['system'], [
            {'type': 'text', 'text': 'stable rules', 'cache_control': {'type': 'ephemeral'}},
            {'type': 'text', 'text': '\nchanging environment', 'cache_control': {'type': 'ephemeral'}},
        ])
        self.assertEqual(body['messages'][-1]['content'][-1]['cache_control'], {'type': 'ephemeral'})
        self.assertEqual([m['role'] for m in body['messages']], ['user', 'assistant', 'user'])
        self.assertEqual(len(body['messages'][0]['content']), 2)
        self.assertTrue(body['messages'][2]['content'][0]['is_error'])
        self.assertEqual((result.input_tokens, result.output_tokens, result.cache_read_tokens), (11, 3, 4))

    THINK = {'type': 'thinking', 'thinking': '', 'signature': 'sig-abc'}
    HISTORY = [
        {'role': 'user', 'content': 'task'},
        {'role': 'assistant', 'content': '', 'raw': {'provider': 'anthropic', 'thinking': [THINK]},
         'tool_calls': [{'id': 't1', 'name': 'list_files', 'args': {}}]},
        {'role': 'tool', 'tool_call_id': 't1', 'name': 'list_files', 'content': 'a.py'},
    ]
    BOUND_400 = {'type': 'error', 'error': {'type': 'invalid_request_error', 'message':
                 'messages.1.content.0: Invalid `signature` in `thinking` block. The block is bound '
                 'to a different conversation. Remove the block, or set '
                 '`thinking.block_binding.prefix_mismatch_behavior` to "drop_block".'}}

    def test_thinking_blocks_echoed_unchanged_with_explicit_drop_block(self):
        ok = {'content': [{'type': 'text', 'text': 'done'}], 'usage': {},
              'input_transformations': [{'type': 'thinking_dropped', 'path': 'messages.1.content.0',
                                         'reason': 'prefix_binding_mismatch'}]}
        with Stub([(200, ok)]) as stub:
            with patch.dict('os.environ', {}, clear=False):
                p = AnthropicProvider('claude-sonnet-5-5', 'k', stub.url)
                result = p.chat('sys', json.loads(json.dumps(self.HISTORY)), tools=[
                    {'name': 'list_files', 'description': 'd', 'parameters': {'type': 'object', 'properties': {}}}])
        _, body = stub.requests[0]
        self.assertEqual(body['messages'][1]['content'][0], self.THINK)          # verbatim, first
        self.assertEqual(body['thinking'], {'type': 'adaptive',
                                            'block_binding': {'prefix_mismatch_behavior': 'drop_block'}})
        self.assertIn('thinking-binding-controls-2026-08-01', stub.headers[0]['anthropic-beta'])
        self.assertEqual(result.transformations[0]['type'], 'thinking_dropped')

    def test_bound_to_other_conversation_400_retries_once_without_thinking(self):
        ok = {'content': [{'type': 'text', 'text': 'recovered'}], 'usage': {}}
        history = json.loads(json.dumps(self.HISTORY))
        with Stub([(400, self.BOUND_400), (200, ok)]) as stub:
            result = AnthropicProvider('claude-sonnet-5-5', 'k', stub.url).chat('sys', history)
        self.assertEqual(len(stub.requests), 2)
        first, second = stub.requests[0][1], stub.requests[1][1]
        self.assertEqual(first['messages'][1]['content'][0]['type'], 'thinking')
        self.assertNotIn('thinking', [b['type'] for b in second['messages'][1]['content']])
        self.assertEqual([b['type'] for b in second['messages'][1]['content']], ['tool_use'])
        self.assertEqual(result.text, 'recovered')
        self.assertTrue(result.retried_without_thinking)
        self.assertNotIn('raw', history[1])       # stripped so later calls do not 400 again

    def test_bound_400_retries_only_once_and_other_400s_never(self):
        with Stub([(400, self.BOUND_400), (400, self.BOUND_400)]) as stub:
            with self.assertRaises(ProviderError) as caught:
                AnthropicProvider('claude-sonnet-5-5', 'k', stub.url).chat('', json.loads(json.dumps(self.HISTORY)))
        self.assertEqual(caught.exception.status, 400)
        self.assertEqual(len(stub.requests), 2)
        with Stub([(400, {'error': 'thinking signature invalid'}), (200, {'content': []})]) as stub:
            with self.assertRaises(ProviderError):
                AnthropicProvider('test', 'test-key', stub.url).chat('', [{'role': 'user', 'content': 'x'}])
        self.assertEqual(len(stub.requests), 1)

    def test_openai_and_deepseek_wire_contract(self):
        spec = {'name': 'read_file', 'description': 'Read', 'parameters': {'type': 'object'}}
        messages = [{'role': 'assistant', 'content': '', 'tool_calls': [{'id': 't1', 'name': 'read_file', 'args': {'path': 'x'}}]}, {'role': 'tool', 'tool_call_id': 't1', 'content': 'hello'}]
        response = {'choices': [{'message': {'content': '', 'tool_calls': [{'id': 't2', 'function': {'name': 'read_file', 'arguments': '{broken'}}]}, 'finish_reason': 'tool_calls'}], 'usage': {'prompt_tokens': 7, 'completion_tokens': 2, 'prompt_tokens_details': {'cached_tokens': 3}}}
        for name, key in [('chatgpt', 'max_completion_tokens'), ('deepseek', 'max_tokens')]:
            with self.subTest(name=name), Stub([(200, response)]) as stub:
                provider = make_provider(name)
                provider.base_url, provider.api_key = stub.url, 'test-key'
                result = provider.chat('rules', messages, [spec], max_tokens=123, tool_choice='none')
                path, body = stub.requests[0]
                self.assertEqual(path, '/chat/completions')
                self.assertEqual(body[key], 123)
                self.assertEqual(body['tool_choice'], 'none')
                self.assertEqual(body['messages'][2]['tool_call_id'], 't1')
                self.assertEqual(json.loads(body['messages'][1]['tool_calls'][0]['function']['arguments']), {'path': 'x'})
                self.assertIn('_invalid_arguments', result.tool_calls[0].args)
                self.assertEqual((result.input_tokens, result.output_tokens, result.cache_read_tokens), (7, 2, 3))
        self.assertEqual(make_provider('claude').model, 'claude-sonnet-5-5')
        self.assertEqual(make_provider('chatgpt').model, 'gpt-5.5')
        self.assertEqual(make_provider('deepseek').model, 'deepseek-v4-pro')

    def test_malformed_tool_arguments_are_data(self):
        for raw in ['{', '[1]', 3, [], None, False]:
            with self.subTest(raw=raw):
                a = AnthropicProvider.from_wire({'content': [{'type': 'tool_use', 'id': 'x', 'name': 'x', 'input': raw}]})
                self.assertIn('_invalid_arguments', a.tool_calls[0].args)
                o = OpenAICompatProvider.from_wire({'choices': [{'message': {'tool_calls': [{'function': {'name': 'x', 'arguments': raw}}]}}]})
                self.assertIsInstance(o.tool_calls[0].args, dict)

    def test_retry_transport_and_transient_http_only(self):
        import urllib.error
        for status in [429, 500, 503, 599]:
            with self.subTest(status=status), Stub([(status, {}), (200, {'ok': True})]) as stub, patch('codebenchkit.agent.llm.time.sleep'):
                self.assertEqual(_post_json(stub.url, {}, {}, 1, retries=1), {'ok': True})
                self.assertEqual(len(stub.requests), 2)
        for status in [400, 401, 403, 404, 422, 501]:   # 501: not implemented, a decision
            with self.subTest(status=status), Stub([(status, {})]) as stub:
                with self.assertRaises(ProviderError) as caught:
                    _post_json(stub.url, {}, {}, 1, retries=1)
                self.assertEqual(caught.exception.status, status)
                self.assertEqual(len(stub.requests), 1)
        real = __import__('urllib.request', fromlist=['urlopen']).urlopen
        with Stub([(200, {'ok': True})]) as stub, patch('codebenchkit.agent.llm.time.sleep'), patch('codebenchkit.agent.llm.urllib.request.urlopen', side_effect=[urllib.error.URLError('offline'), real(__import__('urllib.request', fromlist=['Request']).Request(stub.url, data=b'{}'), timeout=1)]) as opener:
            self.assertTrue(_post_json(stub.url, {}, {}, 1, retries=1)['ok'])
            self.assertEqual(opener.call_count, 2)

    def test_temperature_keyword_remains_supported(self):
        for cls in [AnthropicProvider, OpenAICompatProvider]:
            with self.subTest(provider=cls.__name__), Stub([(200, {})]) as stub:
                provider = (cls('test', 'test-key', stub.url) if cls is AnthropicProvider
                            else cls('test', stub.url, 'test-key'))
                provider.chat('', [{'role': 'user', 'content': 'x'}], temperature=0.2)
                self.assertEqual(stub.requests[0][1]['temperature'], 0.2)

    def test_incomplete_json_object_returns_none(self):
        self.assertIsNone(parse_json_object('reasoning {'))


if __name__ == '__main__':
    unittest.main()
