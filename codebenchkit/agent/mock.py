"""Deterministic offline providers, so every test and the benchmark run with no API keys.

MockCoder drives the real tools: list -> read -> write (correct or buggy) -> run_tests -> fix ->
note -> finish. The fixtures below are independently written solutions to the public specs; no
task record or reference solution is imported. MockSmall plays the cheap model: retrieval gate,
consolidation and compaction summaries.

Memory is deliberately made to matter, so the memory-on path can be exercised end to end. When a
retrieved fact names the task, the coder writes the correct solution first. Otherwise it is
seeded per (task, sample) and sometimes writes a bug: one the visible checks catch, or an
example-only "overfit" that passes them and fails the hidden tests. A memory gain measured with
the mock is therefore built in. It proves the pipeline works, not that memory helps a real model.
Token counts are character-based estimates.
"""
from __future__ import annotations

import json
import random
import re
import textwrap

from .llm import LLMResponse, ScriptedProvider, ToolCall, system_text


_FIXTURES = {
    'slugify': '''
        def slugify(text):
            words, current = [], []
            for ch in text.lower():
                if ch in 'abcdefghijklmnopqrstuvwxyz0123456789':
                    current.append(ch)
                elif current:
                    words.append(''.join(current))
                    current = []
            if current:
                words.append(''.join(current))
            return '-'.join(words)
    ''',
    'parse_duration': '''
        def parse_duration(s):
            scales = {'d': 86400, 'h': 3600, 'm': 60, 's': 1}
            digits, total, previous = '', 0, -1
            if not s:
                raise ValueError('empty duration')
            for ch in s:
                if ch in '0123456789':
                    digits += ch
                    continue
                if ch not in scales or not digits:
                    raise ValueError('invalid duration')
                position = 'dhms'.index(ch)
                if position <= previous:
                    raise ValueError('unordered duration')
                total += int(digits) * scales[ch]
                previous, digits = position, ''
            if digits:
                raise ValueError('missing unit')
            return total
    ''',
    'rle_encode': '''
        from itertools import groupby
        def rle_encode(s):
            return ''.join(character + str(sum(1 for _ in group))
                           for character, group in groupby(s))
    ''',
    'flatten': '''
        def flatten(nested):
            stack, result = list(reversed(nested)), []
            while stack:
                value = stack.pop()
                if isinstance(value, (tuple, list)):
                    stack.extend(reversed(value))
                else:
                    result.append(value)
            return result
    ''',
    'merge_intervals': '''
        def merge_intervals(intervals):
            pending = sorted([list(pair) for pair in intervals], reverse=True)
            merged = []
            while pending:
                start, end = pending.pop()
                while pending and pending[-1][0] <= end:
                    end = max(end, pending.pop()[1])
                merged.append([start, end])
            return merged
    ''',
    'roman_to_int': '''
        def roman_to_int(s):
            values = dict(zip('IVXLCDM', (1, 5, 10, 50, 100, 500, 1000)))
            greatest, result = 0, 0
            for symbol in reversed(s):
                value = values[symbol]
                result += -value if value < greatest else value
                greatest = max(greatest, value)
            return result
    ''',
    'is_valid_ipv4': '''
        import ipaddress
        def is_valid_ipv4(s):
            try:
                ipaddress.IPv4Address(s)
            except (ValueError, TypeError):
                return False
            return True
    ''',
    'chunk': '''
        def chunk(items, size):
            if size < 1:
                raise ValueError('positive chunk length required')
            result, group = [], []
            for item in items:
                group.append(item)
                if len(group) == size:
                    result.append(group)
                    group = []
            if group:
                result.append(group)
            return result
    ''',
    'top_words': '''
        def top_words(text, k):
            counts, word = {}, ''
            for character in text.lower() + ' ':
                if 'a' <= character <= 'z':
                    word += character
                elif word:
                    counts[word] = counts.get(word, 0) + 1
                    word = ''
            order = sorted(counts, key=lambda word: (-counts[word], word))
            return [[word, counts[word]] for word in order[:k]]
    ''',
    'camel_to_snake': '''
        def camel_to_snake(name):
            result = []
            for index, character in enumerate(name):
                if index and character.isupper():
                    previous = name[index - 1]
                    next_lower = index + 1 < len(name) and name[index + 1].islower()
                    if previous.islower() or previous.isdigit() or (previous.isupper() and next_lower):
                        result.append('_')
                result.append(character.lower())
            return ''.join(result)
    ''',
}


def fixture_source(name: str) -> str:
    """Return an independently authored algorithm, never a benchmark reference."""
    return textwrap.dedent(_FIXTURES[name]).strip() + '\n'


class MockCoder(ScriptedProvider):
    """Stateless: each reply is a function of (system, messages), so one instance can serve many
    runs and conditions. spec: 'mock' (skill 0.5), 'mock:<skill 0-1>', or 'mock:correct|visible|edge'
    to force the first write."""
    name = "mock"

    def __init__(self, spec: str = "mock"):
        _, _, arg = spec.partition(":")
        self.force = arg if arg in ("correct", "visible", "edge") else None
        try:
            self.skill = float(arg) if arg and not self.force else 0.5
        except ValueError:
            raise SystemExit(f"bad mock spec {spec!r}: use mock, mock:<0-1>, mock:correct|visible|edge")
        super().__init__(self._respond, model=spec)

    # ---- what the run has done so far, read back from history
    @staticmethod
    def _target(messages) -> str | None:
        first = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
        return next((n for n in re.findall(r"\bdef\s+(\w+)\s*\(", first) if n in _FIXTURES), None)

    @staticmethod
    def _has_lesson(system, target: str) -> bool:
        text = system_text(system)
        return any(re.match(r"- \[F\d+\]", ln) and target in ln for ln in text.splitlines())

    def _first_mode(self, system, target: str) -> str:
        if self._has_lesson(system, target):
            return "correct"
        if self.force:
            return self.force
        ws = re.search(r"Workspace: (\S+)", system_text(system))
        seed = f"{self.model}|{target}|{ws.group(1).rsplit('/', 1)[-1] if ws else ''}"
        r = random.Random(seed).random()
        if r < self.skill:
            return "correct"
        return "visible" if r < self.skill + (1 - self.skill) / 2 else "edge"

    def _respond(self, system, messages, tools, tool_choice):
        if tool_choice == "none":
            return "Stopped at a guardrail; the harness verification records the outcome."
        target = self._target(messages)
        done = [c["name"] for m in messages if m.get("role") == "assistant" for c in m.get("tool_calls") or []]
        if target is None:
            if "list_files" not in done:
                return _call("list_files", {}, len(done))
            return "No offline fixture matches this task; nothing changed."
        if "list_files" not in done:
            return _call("list_files", {}, len(done))
        if "read_file" not in done:
            return _call("read_file", {"path": "solution.py"}, len(done))
        writes = done.count("write_file")
        if writes == 0:
            mode = self._first_mode(system, target)
            code = _candidate(target, mode, _public_examples(messages))
            return _call("write_file", {"path": "solution.py", "content": code}, len(done))
        last = _last_event(messages)
        if last == "wrote":
            return _call("run_tests", {}, len(done))
        if last == "failed":
            if writes >= 2:
                return _call("finish", {"summary": "Could not make the visible checks pass."}, len(done))
            return _call("write_file", {"path": "solution.py", "content": fixture_source(target)}, len(done))
        if last == "passed" and writes >= 2 and "note" not in done:
            lesson = (f"{target}: a placeholder or example-only implementation fails; implement the full "
                      f"specification from the docstring, including edge cases.")
            return _call("note", {"lesson": lesson}, len(done))
        if "finish" not in done or last == "failed":
            return _call("finish", {"summary": f"Implemented {target}; visible checks pass."}, len(done))
        return "Done."


def _call(name: str, args: dict, n: int) -> LLMResponse:
    return LLMResponse("", [ToolCall(f"mock_{n + 1}", name, args)])


def _last_event(messages) -> str:
    """'wrote' | 'passed' | 'failed' | 'other', from the newest tool result or harness feedback."""
    for m in reversed(messages):
        c = str(m.get("content", ""))
        if m.get("role") == "user" and c.startswith("[harness]") and "FAILED" in c:
            return "failed"
        if m.get("role") == "tool":
            if m.get("name") == "run_tests":
                return "passed" if c.startswith("PASSED") else "failed"
            if m.get("name") == "write_file":
                return "wrote"
            return "other"
    return "other"


def _public_examples(messages) -> list:
    """The visible examples the task prompt showed (bench prompts embed them as JSON)."""
    first = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
    if "Public examples:" not in first:
        return []
    try:
        data, _ = json.JSONDecoder().raw_decode(first.split("Public examples:", 1)[1].lstrip())
    except ValueError:
        return []
    return data if isinstance(data, list) else []


def _candidate(target: str, mode: str, examples: list) -> str:
    if mode == "correct":
        return fixture_source(target)
    if mode == "visible" or not examples:   # caught by the visible checks
        return f"def {target}(*args, **kwargs):\n    return None\n"
    # "edge": answers only the examples it was shown, so it passes visible and fails hidden.
    table = [[e.get("args"), e.get("expected"), e.get("raises")] for e in examples if isinstance(e, dict)]
    return (f"_SEEN = {table!r}\n\n\n"
            f"def {target}(*args):\n"
            f"    for seen_args, out, exc in _SEEN:\n"
            f"        if list(args) == seen_args:\n"
            f"            if exc:\n"
            f"                raise ValueError(exc)\n"
            f"            return out\n"
            f"    return None\n")


class MockSmall(ScriptedProvider):
    """The cheap model: retrieval gate, consolidation ops, compaction summaries."""
    name = "mock"

    def __init__(self):
        super().__init__(self._respond, model="mock-small")

    def _respond(self, system, messages, tools, tool_choice):
        prompt = "\n".join(str(m.get("content", "")) for m in messages)
        data = _consolidation_data(prompt)
        if data is not None:
            return json.dumps({"operations": _consolidation_ops(data)})
        if prompt.startswith("Summarize this coding agent"):
            return "Earlier steps: explored the workspace, read files and ran the checks (see digest above)."
        task = prompt.rsplit("Task:", 1)[-1].strip()
        return json.dumps({"retrieve": True, "query": task[:200]})


def _consolidation_data(prompt: str) -> dict | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", prompt):
        try:
            data, _ = decoder.raw_decode(prompt[match.start():])
        except ValueError:
            continue
        if isinstance(data, dict) and isinstance(data.get("episodes"), list):
            return data
    return None


def _consolidation_ops(data: dict) -> list[dict]:
    """ADD a pitfall per episode note, or CONFIRM an existing fact with the same content."""
    ops, existing = [], {f.get("content"): f.get("id") for f in data.get("facts", []) if isinstance(f, dict)}
    for ep in data["episodes"]:
        if not isinstance(ep, dict) or not isinstance(ep.get("id"), int):
            continue
        for note in ep.get("notes") or []:
            if note in existing:
                ops.append({"op": "CONFIRM", "id": existing[note], "evidence": [ep["id"]]})
            else:
                ops.append({"op": "ADD", "subject": note.split(":", 1)[0][:60], "content": note[:300],
                            "kind": "pitfall", "scope": ep.get("scope", "global"), "evidence": [ep["id"]]})
                existing[note] = None          # do not ADD the same note twice in one batch
    return [o for o in ops if o.get("op") != "CONFIRM" or o.get("id")]


def make_mock(spec: str = "mock", role: str = "main") -> ScriptedProvider:
    """Factory used by llm.make_provider. The small role never codes."""
    if role == "small":
        return MockSmall()
    return MockCoder(spec)
