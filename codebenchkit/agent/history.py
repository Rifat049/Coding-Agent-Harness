"""Deterministic history hygiene preserving complete multi-tool exchanges."""
from __future__ import annotations

import json


def estimate_tokens(messages: list[dict], system: str = '') -> int:
    """Conservative character estimate, not a substitute for provider usage."""
    return (len(system) + len(json.dumps(messages, ensure_ascii=False)) + 3) // 4


def _groups(messages: list[dict]) -> list[tuple[int, int, bool]]:
    groups = []
    i = 0
    while i < len(messages):
        start = i
        message = messages[i]
        calls = message.get('tool_calls', []) if message.get('role') == 'assistant' else []
        i += 1
        if calls:
            expected = [c['id'] for c in calls]
            actual = []
            while i < len(messages) and messages[i].get('role') == 'tool':
                actual.append(messages[i].get('tool_call_id'))
                i += 1
            groups.append((start, i, sorted(expected) == sorted(actual)))
        else:
            groups.append((start, i, message.get('role') != 'tool'))
    return groups


def shrink_seen(messages: list[dict], observed_until: int | None = None,
                threshold: int = 4000, keep: int = 2000) -> int:
    """Trim only observed results; default protects the entire latest tool group."""
    if observed_until is None:
        observed_until = len(messages)
        for start, end, _ in reversed(_groups(messages)):
            if messages[start].get('tool_calls'):
                observed_until = start
                break
    count = 0
    for message in messages[:observed_until]:
        text = message.get('content', '')
        if message.get('role') == 'tool' and isinstance(text, str) and len(text) > threshold:
            message['content'] = (text[:keep] +
                f'\n[Trimmed after observation; original output {len(text)} characters. Re-read if needed.]')
            count += 1
    return count


SUMMARY_PROMPT = """Summarize this coding agent's work so far so it can continue without the full log.
Keep: files read and what they contain that matters, files changed and how, commands and test
results (pass/fail and the key error lines), decisions made, and what remains to do. Do not claim
anything passed unless a tool result shows it. Plain text, at most {chars} characters.

{log}"""


def compact_history(messages: list[dict], context_limit: int = 32000,
                    system: str = '', observed_until: int | None = None,
                    summarize=None, threshold: float = .7, require_smaller: bool = True) -> dict:
    """Replace an observed middle prefix with one summary message. Never splits a tool call
    from its result.

    The initial task and the newest group are anchors even if they alone exceed the target. The
    caller must stop rather than discard still-unobserved evidence. `summarize(prompt) -> str` is
    the small model. On any failure, or when it is not given, a deterministic digest is used. The
    summary replaces earlier turns, so every remaining thinking block would fail the API's
    conversation check: they are stripped here.
    """
    before = estimate_tokens(messages, system)
    report = {'compacted': False, 'before_tokens': before, 'after_tokens': before,
              'removed_messages': 0}
    if before <= max(1, context_limit * threshold) or len(messages) < 4:
        return report
    groups = _groups(messages)
    if any(not complete for _, _, complete in groups):
        report['reason'] = 'incomplete_tool_group'
        return report
    protected = groups[-1][0]
    # Feedback after a tool group must not make that still-unobserved group eligible.
    for start, end, _ in reversed(groups):
        if messages[start].get('tool_calls'):
            protected = min(protected, start)
            break
    if observed_until is not None:
        protected = min(protected, observed_until)
    eligible = [(start, end) for start, end, _ in groups if start > 0 and end <= protected]
    if not eligible:
        return report
    end = eligible[-1][1]
    removed = messages[1:end]
    lines = ['Deterministic history digest (observations, not proof of completion):']
    for message in removed:
        role = message.get('role', '?')
        text = ' '.join(str(message.get('content', '')).split())
        if message.get('tool_calls'):
            calls = ', '.join(c['name'] + '(' + json.dumps(c.get('args', {}), sort_keys=True)[:100] + ')'
                              for c in message['tool_calls'])
            lines.append('Requested: ' + calls[:350])
        else:
            lines.append(role + ': ' + text[:250])
    max_chars = min(2400, max(200, context_limit // 2))
    digest = '\n'.join(lines)
    if len(digest) > max_chars:
        digest = digest[:max_chars - 24] + '\n[digest truncated]'
    method = 'digest'
    if summarize is not None:
        try:
            log = '\n'.join(_log_line(m) for m in removed)[-40000:]
            text = (summarize(SUMMARY_PROMPT.format(chars=max_chars * 2, log=log)) or '').strip()
            if len(text) >= 40:
                digest = ('Summary of earlier work (written by a helper model; verify before relying '
                          'on it):\n' + text[:max_chars * 2])
                method = 'summary'
        except Exception as exc:  # noqa: BLE001 - fall back to the deterministic digest
            from .llm import FatalProviderError
            if isinstance(exc, FatalProviderError):
                raise
            report['summary_error'] = f'{type(exc).__name__}: {exc}'[:200]
    replacement = [{'role': 'user', 'content': '[Compacted history]\n' + digest}]
    candidate = estimate_tokens(messages[:1] + replacement + messages[end:], system)
    if require_smaller and candidate >= before:
        # A summary that does not shrink the history is all cost (it also invalidates the prompt
        # cache and any thinking blocks). Seen on a tiny context: 886 -> 944 estimated tokens.
        report.update(reason='summary_not_smaller', candidate_tokens=candidate, method=method)
        return report
    messages[1:end] = replacement
    from .llm import strip_thinking
    report.update(compacted=True, removed_messages=len(removed), method=method,
                  thinking_blocks_stripped=strip_thinking(messages),
                  after_tokens=estimate_tokens(messages, system))
    return report


def _log_line(message: dict) -> str:
    if message.get('tool_calls'):
        return 'assistant called: ' + ', '.join(
            c['name'] + json.dumps(c.get('args', {}), sort_keys=True)[:300] for c in message['tool_calls'])
    return f"{message.get('role', '?')}: {str(message.get('content', ''))[:1500]}"
