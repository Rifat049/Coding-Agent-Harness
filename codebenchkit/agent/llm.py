"""Model access: one neutral dialect, thin wire adapters, zero SDK dependencies.

The loop only speaks the neutral format below. Each adapter translates it to its wire
format on the way out and back on the way in. Keeping the loop provider-agnostic is what
lets one harness compare Claude against ChatGPT on identical tools and budgets.

Neutral messages
    {"role": "user",      "content": str}
    {"role": "assistant", "content": str, "tool_calls": [{"id", "name", "args"}], "raw": {...}}
    {"role": "tool",      "tool_call_id": str, "name": str, "content": str, "is_error": bool}

`raw` is opaque, provider-owned state that has to be sent back verbatim. Today that means
Claude's thinking blocks, which are bound to the conversation that produced them: dropping
or reordering them can make a request fail. Other providers ignore it.

Neutral tool spec
    {"name": str, "description": str, "parameters": <JSON schema>}

`system` may be a string or a list of strings. A list marks cache boundaries: the first block
is the stable core (identical across runs), and later blocks are per-run context.

Providers (defaults: deepseek main, qwen small; see .cbk/config.toml)
    deepseek   DeepSeek, OpenAI-compatible            (default main model: the agent loop)
    qwen       Qwen, OpenAI-compatible: local Ollama by default, or Alibaba's hosted endpoint.
               Default model qwen2.5-coder:7b: a main-model option (alias `qwen-coder`) and the
               default small model (gate + consolidation). Local runs use temperature 0 and a
               fixed seed (QWEN_SEED, default 0), and recover tool calls that the model writes
               as JSON text instead of native tool_calls (see parse_text_tool_calls).
    anthropic  Claude via the Messages API           (main model option)
    openai     ChatGPT via /chat/completions          (main model option)
    compat     any OpenAI-compatible server (vLLM, SGLang, Ollama, OpenRouter, ...)
    scripted   a Python callable, for tests and offline demos

Keys: DEEPSEEK_API_KEY, OPENAI_API_KEY, CBK_ANTHROPIC_API_KEY. ANTHROPIC_API_KEY is deliberately
never read: other Anthropic tools in the same shell (Claude Code) would pick it up and bill it. Qwen needs
no key locally; DASHSCOPE_API_KEY is sent only when set, for the hosted endpoint.
"""
from __future__ import annotations

import http.client
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "args": self.args}


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0              # all input, cached or not
    output_tokens: int = 0
    stop_reason: str = ""
    latency_s: float = 0.0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    raw: dict | None = None            # provider state to echo back (see module docstring)
    # Anthropic `input_transformations`: e.g. thinking blocks the API dropped because an earlier
    # turn was edited (trim/compaction). Logged so lost reasoning is visible in the trace.
    transformations: list = field(default_factory=list)
    # Whether the response carried `input_transformations` at all: "absent" (key missing, as on
    # every non-Anthropic provider), "empty" (present, []), or "list" (present, non-empty).
    transformations_status: str = "absent"
    retried_without_thinking: bool = False
    reasoning_tokens: int | None = None  # usage.completion_tokens_details.reasoning_tokens, if reported
    # Where tool_calls came from: "native" (the wire's tool_calls field), "text" (recovered from
    # JSON the model wrote in its text, see parse_text_tool_calls), or "" (no tool calls).
    tool_call_source: str = ""
    # The text looked like a tool call (JSON with a "name") but could not be turned into one.
    text_tool_error: str = ""
    # Ollama native: the prompt filled the context window, so Ollama may have cut its start.
    context_full: bool = False
    # The model's text before tool calls were recovered from it (text is then ""), for the trace.
    raw_text: str = ""

    def thinking_stats(self) -> dict:
        """Per-response reasoning counts for the trace: Claude thinking blocks (and how many are
        redacted), DeepSeek/OpenAI-compatible reasoning_content length."""
        raw = self.raw or {}
        blocks = (raw.get("thinking") or []) if raw.get("provider") == "anthropic" else []
        rc = raw.get("reasoning_content") if raw.get("provider") == "openai-compat" else None
        return {"thinking_blocks": len(blocks),
                "redacted_thinking_blocks": sum(1 for b in blocks if b.get("type") == "redacted_thinking"),
                "reasoning_chars": len(rc) if isinstance(rc, str) else 0,
                "reasoning_tokens": self.reasoning_tokens}

    def as_message(self) -> dict:
        """The neutral assistant message for this response."""
        msg: dict = {"role": "assistant", "content": self.text,
                     "tool_calls": [c.to_dict() for c in self.tool_calls]}
        if self.raw:
            msg["raw"] = self.raw
        return msg


class ProviderError(RuntimeError):
    def __init__(self, msg: str, status: int = 0):
        super().__init__(msg)
        self.status = status


# Non-retryable account errors: no retry, no next item, no fallback can succeed while they hold.
# DeepSeek's error table (https://api-docs.deepseek.com/quick_start/error_codes): "401 -
# Authentication Fails ... wrong API key", "402 - Insufficient Balance ... You have run out of balance".
FATAL_STATUSES = {401: "authentication failed (check the API key)", 402: "out of balance",
                  403: "access denied (the key lacks permission or the account is restricted)"}


class FatalProviderError(ProviderError):
    """A billing or auth failure. Every caller must let it propagate: it stops the whole run, and
    nothing after it is scored (see runner.run_agent and the benchmarks)."""

    def __init__(self, msg: str, status: int = 0, url: str = "", why: str = ""):
        super().__init__(msg, status)
        self.url = url
        self.why = why                 # a ready-made reason (e.g. Ollama down), used as is

    def reason(self, provider: str = "") -> str:
        if self.why:
            return self.why
        """'Out of DeepSeek balance (HTTP 402)'. The provider is named from the URL that failed
        (so a small model's error is not blamed on the main model), else from `provider`."""
        from urllib.parse import urlparse
        host = urlparse(self.url).hostname or ""
        by_host = {"api.deepseek.com": "DeepSeek", "api.anthropic.com": "Anthropic", "api.openai.com": "OpenAI",
                   "localhost": "Ollama", "127.0.0.1": "Ollama"}
        name = next((v for k, v in by_host.items() if host == k or host.endswith("." + k)), "")
        if not name and "dashscope" in host:
            name = "DashScope"
        if not name:
            name = {"deepseek": "DeepSeek", "anthropic": "Anthropic", "openai": "OpenAI", "qwen": "Qwen",
                    "dashscope": "DashScope"}.get(provider.partition(":")[0], provider.partition(":")[0] or host or "provider")
        if self.status == 402:
            return f"Out of {name} balance (HTTP 402)"
        return f"{name}: {FATAL_STATUSES.get(self.status, 'fatal error')} (HTTP {self.status})"


# Defaults are plain strings: override with --model provider:model-id when they age out.
# deepseek-v4-pro and deepseek-flash both list "Tool Calls" as supported
# (https://api-docs.deepseek.com/quick_start/pricing.md, read 2026-11-19); v4-pro is the stronger one.
DEFAULTS = {
    "anthropic": {"model": "claude-sonnet-5-5", "small": "claude-haiku-5-5", "key_env": "CBK_ANTHROPIC_API_KEY",
                  "base_url": "https://api.anthropic.com"},
    "openai": {"model": "gpt-5.5", "small": "gpt-5.5", "key_env": "OPENAI_API_KEY",
               "base_url": "https://api.openai.com/v1"},
    "deepseek": {"model": "deepseek-v4-pro", "small": "deepseek-v4-pro", "key_env": "DEEPSEEK_API_KEY",
                 "base_url": "https://api.deepseek.com"},
    # Local Ollama by default (no key). For Alibaba's hosted endpoint set QWEN_BASE_URL to
    # https://dashscope-intl.aliyuncs.com/compatible-mode/v1 and DASHSCOPE_API_KEY.
    "qwen": {"model": "qwen2.5-coder:7b", "small": "qwen2.5-coder:7b", "key_env": "DASHSCOPE_API_KEY", "optional_key": True,
             "base_url": "http://localhost:11434/v1"},
    # Hosted Qwen on Alibaba Cloud Model Studio (DashScope), OpenAI-compatible. DASHSCOPE_MODEL and
    # DASHSCOPE_BASE_URL override; the key is required here (unlike local `qwen`).
    "dashscope": {"model": "qwen-plus", "small": "qwen-plus", "key_env": "DASHSCOPE_API_KEY",
                  "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"},
}
# `qwen-coder` names the C40 demo model; it is the qwen provider (local Ollama), so
# `--model qwen-coder` and `--model qwen-coder:qwen2.5-coder:7b` are the same run.
ALIASES = {"claude": "anthropic", "chatgpt": "openai", "gpt": "openai", "qwen-coder": "qwen"}


def canonical(spec: str) -> str:
    """The provider kind of a spec: 'claude:x' -> 'anthropic'."""
    kind = (spec or "").strip().partition(":")[0]
    return ALIASES.get(kind, kind)


def provider_key(kind: str) -> str:
    """The API key for a provider kind, from its one env var (never ANTHROPIC_API_KEY)."""
    return os.environ.get(DEFAULTS.get(kind, {}).get("key_env", ""), "")


def missing_key(spec: str) -> str | None:
    """The env var to set when this spec needs a key that is absent, else None."""
    kind = canonical(spec)
    d = DEFAULTS.get(kind)
    if not d or d.get("optional_key") or provider_key(kind):
        return None
    return d["key_env"]


def qwen_settings() -> dict:
    """QWEN_BASE_URL (default local Ollama), QWEN_MODEL, DASHSCOPE_API_KEY (optional)."""
    d = DEFAULTS["qwen"]
    base = (os.environ.get("QWEN_BASE_URL") or d["base_url"]).rstrip("/")
    return {"base_url": base, "model": os.environ.get("QWEN_MODEL") or d["model"],
            "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
            "local": base.startswith(("http://localhost", "http://127.0.0.1", "http://[::1]"))}


def system_text(system: str | list[str] | None) -> str:
    if not system:
        return ""
    return system if isinstance(system, str) else "\n\n".join(b for b in system if b)


class Provider:
    name = "base"
    model = ""

    def chat(self, system: str | list[str], messages: list[dict], tools: list[dict] | None = None,
             max_tokens: int = 8192, tool_choice: str = "auto", temperature: float | None = None) -> LLMResponse:
        raise NotImplementedError

    def complete(self, prompt: str, max_tokens: int = 4096, json_mode: bool = False,
                 schema: dict | None = None, temperature: float | None = None) -> str:
        """Single-turn text completion, used by the retrieval gate and consolidation.
        `json_mode` asks for a JSON object where the wire supports it; `schema` additionally
        constrains it (Ollama's native `format`). Callers still validate the reply.
        `temperature` is ignored here: current Claude models reject sampling parameters (400)."""
        return self.chat("", [{"role": "user", "content": prompt}], None, max_tokens).text

    def describe(self) -> str:
        return f"{self.name}:{self.model}"

    def sampling(self) -> dict:
        """What the provider sends for determinism, for receipts and run configs."""
        return {"temperature": None, "seed": None, "note": "provider default"}


# --------------------------------------------------------------------------- transport
RETRY_STATUSES = {429, 500, 502, 503, 504, 529}


def _post_json(url: str, body: dict, headers: dict, timeout: float, retries: int = 4,
               sleep: Callable[[float], None] | None = None) -> dict:
    """POST JSON, retrying transport faults, 429 and 5xx with jittered exponential backoff.

    Any other 4xx is a decision, not a fault: the provider looked at the request and refused it,
    so the same request would be refused again. It is raised at once."""
    data = json.dumps(body).encode()
    last: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json", **headers})
        wait = min(30.0, 2.0 ** attempt)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                last = ProviderError(f"non-JSON response: {raw[:200]!r}")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:1500]
            if e.code in FATAL_STATUSES:
                raise FatalProviderError(f"HTTP {e.code} ({FATAL_STATUSES[e.code]}): {detail}", e.code, url) from None
            err = ProviderError(f"HTTP {e.code}: {detail}", e.code)
            # 501 Not Implemented is a decision too (e.g. Ollama: "structured output is unavailable").
            if e.code == 501 or (e.code not in RETRY_STATUSES and not 500 <= e.code < 600):
                raise err from None
            last = err
            ra = e.headers.get("retry-after") if e.headers else None
            try:
                wait = min(60.0, float(ra)) if ra else wait
            except ValueError:
                pass
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, OSError) as e:
            last = e
        if attempt < retries:
            (sleep or time.sleep)(wait + random.random() * 0.25)
    raise ProviderError(f"request failed after {retries + 1} tries: {last}",
                        getattr(last, "status", 0))


def _ollama_post(url: str, body: dict, timeout: float, retries: int, model: str) -> dict:
    """POST to local Ollama. A server that is down, or a model that is not pulled, cannot recover
    by retrying the next item, so both raise FatalProviderError with the command that fixes it."""
    try:
        return _post_json(url, body, {}, timeout, retries)
    except FatalProviderError:
        raise
    except ProviderError as e:
        text = str(e)
        if e.status == 404 and "not found" in text:
            raise FatalProviderError(text, 404, url, f"Ollama has no model {model!r}: run `ollama pull {model}`") from None
        if e.status == 0 and re.search(r"Connection refused|Errno 61|Errno 111|nodename|Connection reset", text):
            raise FatalProviderError(text, 0, url, f"Ollama is not reachable at {url.rsplit('/api/', 1)[0]}: start it "
                                     f"(`ollama serve`, or open the Ollama app), then `ollama pull {model}`") from None
        raise


# --------------------------------------------------------------------------- Anthropic (Claude)
# Models that reject `thinking: {type: "adaptive"}` (pre-4.6 generation).
_NO_ADAPTIVE = re.compile(r"claude-(3|instant|2)|claude-(opus|sonnet|haiku)-4-[0-5]\b|claude-(opus|sonnet|haiku)-4$")
# Models where the server-side refusal fallback is recommended by default.
_FALLBACK_MODELS = {"claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
THINKING_BETA = "thinking-binding-controls-2026-08-01"
BOUND_ELSEWHERE = "bound to a different conversation"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None,
                 timeout: float = 600, prompt_cache: bool = True, thinking: bool | None = None,
                 fallbacks: bool | None = None, effort: str | None = None, retries: int = 4):
        self.model = model
        self.api_key = api_key if api_key is not None else provider_key("anthropic")
        self.base_url = (base_url or os.environ.get("ANTHROPIC_BASE_URL") or DEFAULTS["anthropic"]["base_url"]).rstrip("/")
        self.timeout = timeout
        self.prompt_cache = prompt_cache
        # Thinking blocks are bound to the conversation prefix that produced them, and the API's
        # check is spreading to every model that returns them. The loop trims and compacts, which
        # are history edits, so every request opts into drop_block explicitly rather than relying
        # on a per-model, per-account default.
        self.thinking = (not _NO_ADAPTIVE.search(model)) if thinking is None else thinking
        self.fallbacks = (model in _FALLBACK_MODELS) if fallbacks is None else fallbacks
        self.effort = effort
        self.retries = retries

    # ---- neutral -> wire
    @staticmethod
    def to_wire(messages: list[dict], strip_thinking: bool = False, cache_last: bool = False) -> list[dict]:
        """Neutral -> Anthropic. Tool results become tool_result blocks in a user turn, and
        consecutive same-role turns are merged, because the API requires alternation. Thinking blocks
        from `raw` go back first and unchanged."""
        out: list[dict] = []

        def push(role: str, blocks: list[dict]):
            if not blocks:
                return
            if out and out[-1]["role"] == role:
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": role, "content": list(blocks)})

        for m in messages:
            role = m.get("role")
            if role == "user":
                push("user", [{"type": "text", "text": m.get("content") or "(empty)"}])
            elif role == "assistant":
                blocks: list[dict] = []
                raw = m.get("raw") or {}
                if raw.get("provider") == "anthropic" and not strip_thinking:
                    blocks += [dict(b) for b in raw.get("thinking", [])]
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                blocks += [{"type": "tool_use", "id": c["id"], "name": c["name"],
                            "input": c["args"] if isinstance(c.get("args"), dict) else {}}
                           for c in m.get("tool_calls") or []]
                if not any(b["type"] in ("text", "tool_use") for b in blocks):
                    blocks.append({"type": "text", "text": "(no content)"})
                push("assistant", blocks)
            elif role == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"],
                         "content": m.get("content") or "(no output)"}
                if m.get("is_error") or str(m.get("content", "")).startswith(("ERROR:", "DENIED:")):
                    block["is_error"] = True
                push("user", [block])
        if cache_last and out:
            # Sliding breakpoint on the newest block: each iteration reads the previous prefix from cache.
            last = out[-1]["content"][-1]
            out[-1]["content"][-1] = {**last, "cache_control": {"type": "ephemeral"}}
        return out

    def build_body(self, system, messages, tools, max_tokens, tool_choice) -> tuple[dict, list[str]]:
        body: dict = {"model": self.model, "max_tokens": max_tokens,
                      "messages": self.to_wire(messages, False, self.prompt_cache)}
        betas: list[str] = []
        blocks = [system] if isinstance(system, str) else list(system or [])
        prefix = getattr(system, "stable_prefix", "")
        has_prefix = isinstance(system, str) and bool(prefix) and system.startswith(prefix)
        if has_prefix:
            blocks = [prefix, system[len(prefix):]]
        blocks = [b for b in blocks if b]
        if blocks:
            body["system"] = [{"type": "text", "text": b} for b in blocks]
            if self.prompt_cache:
                # Breakpoint 1: the stable rules (shared across runs). Breakpoint 2: the end of the
                # per-run context, which is frozen for the whole run. (Breakpoint 3 slides on the
                # newest message; the API allows 4.)
                body["system"][0]["cache_control"] = {"type": "ephemeral"}
                if len(body["system"]) > 1:
                    body["system"][-1]["cache_control"] = {"type": "ephemeral"}
        if tools:
            body["tools"] = [{"name": t["name"], "description": t["description"],
                              "input_schema": t["parameters"]} for t in tools]
            body["tool_choice"] = {"type": "none" if tool_choice == "none" else "auto"}
        if self.thinking:
            body["thinking"] = {"type": "adaptive",
                                "block_binding": {"prefix_mismatch_behavior": "drop_block"}}
            betas.append(THINKING_BETA)
        if self.effort:
            body["output_config"] = {"effort": self.effort}
        if self.fallbacks:
            body["fallbacks"] = "default"
            betas.append(FALLBACK_BETA)
        return body, betas

    def chat(self, system, messages, tools=None, max_tokens=16000, tool_choice="auto", temperature=None):
        if not self.api_key:
            raise ProviderError("CBK_ANTHROPIC_API_KEY is not set (ANTHROPIC_API_KEY is ignored on purpose)")
        body, betas = self.build_body(system, messages, tools, max_tokens, tool_choice)
        if temperature is not None:
            body["temperature"] = temperature
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        if betas:
            headers["anthropic-beta"] = ",".join(betas)
        t0 = time.perf_counter()
        retried = False
        try:
            data = _post_json(f"{self.base_url}/v1/messages", body, headers, self.timeout, self.retries)
        except ProviderError as e:
            # Last-resort recovery when drop_block did not cover a case: replay once with every
            # thinking block removed. Text and tool_use blocks stay, so the turn structure is intact.
            if e.status != 400 or BOUND_ELSEWHERE not in str(e) or not _has_thinking(messages):
                raise
            body["messages"] = self.to_wire(messages, True, self.prompt_cache)
            data = _post_json(f"{self.base_url}/v1/messages", body, headers, self.timeout, self.retries)
            retried = True
            # The rejected blocks would fail the same way on every later request: drop them from
            # the caller's history now so the next call does not pay for another 400.
            strip_thinking(messages)
        resp = self.from_wire(data, time.perf_counter() - t0)
        resp.retried_without_thinking = retried
        return resp

    @staticmethod
    def from_wire(data: dict, latency: float = 0.0) -> LLMResponse:
        text, calls, thinking = [], [], []
        for b in data.get("content", []):
            t = b.get("type")
            if t == "text":
                text.append(b.get("text", ""))
            elif t == "tool_use":
                inp = b.get("input")
                calls.append(ToolCall(b.get("id", ""), b.get("name", ""),
                                      inp if isinstance(inp, dict) else {"_invalid_arguments": inp}))
            elif t in ("thinking", "redacted_thinking"):
                thinking.append(b)
        u = data.get("usage") or {}
        cr = int(u.get("cache_read_input_tokens") or 0)
        cw = int(u.get("cache_creation_input_tokens") or 0)
        resp = LLMResponse("".join(text), calls, int(u.get("input_tokens") or 0) + cr + cw,
                           int(u.get("output_tokens") or 0), data.get("stop_reason") or "", latency, cr, cw,
                           {"provider": "anthropic", "thinking": thinking} if thinking else None)
        resp.transformations = list(data.get("input_transformations") or [])
        if "input_transformations" not in data:
            resp.transformations_status = "absent"
        else:
            resp.transformations_status = "list" if resp.transformations else "empty"
        return resp


def _has_thinking(messages: list[dict]) -> bool:
    return any((m.get("raw") or {}).get("thinking") for m in messages if m.get("role") == "assistant")


def strip_thinking(messages: list[dict]) -> int:
    """Remove Claude thinking blocks from neutral history in place (used by compaction, whose
    summary invalidates every earlier block anyway). Returns how many blocks were removed."""
    n = 0
    for m in messages:
        raw = m.get("raw")
        if m.get("role") == "assistant" and raw and raw.get("thinking"):
            n += len(raw["thinking"])
            m.pop("raw")
    return n


# --------------------------------------------------------------------------- OpenAI-compatible
class OpenAICompatProvider(Provider):
    """ChatGPT, DeepSeek, Qwen, and any OpenAI-compatible server.

    DeepSeek thinking mode + tools (https://api-docs.deepseek.com/guides/thinking_mode#tool-calls
    and https://api-docs.deepseek.com/api/create-chat-completion, read 2026-11-19):
      - Thinking is a request parameter, on by default ("Thinking mode is enabled by default"), and
        "The DeepSeek model's thinking mode supports tool calls". So the loop runs with thinking on.
      - "for requests carrying the `tools` parameter, the `reasoning_content` must be fully passed
        back to the API in all subsequent requests — even for turns where the model did not perform
        a tool call. If your code does not correctly pass back `reasoning_content`, the API will
        return a 400 error." Its multi-turn sample keeps Turn 1's reasoning in the Turn 2 request.
        So reasoning_content is kept on every assistant message, within a turn AND across user
        turns: it is NOT stripped between turns (stripping is what would cause the 400). Every
        loop request carries `tools` (even the tools-off final answer, via tool_choice=none).
      - Without `tools` it "does not need to be passed back; even if passed to the API, it will
        be ignored", so single-turn small-model calls need no special handling.
      - "`required` and named tool choices are not supported in thinking mode" (400): the loop
        only ever sends "auto" or "none".
      - temperature "Has no effect in thinking mode"; the output cap is `max_tokens`.
    Claude's thinking-block compaction strip (strip_thinking) only touches Anthropic `raw`, so it
    never removes DeepSeek reasoning_content; compaction drops whole old turns instead.
    """

    def __init__(self, model: str, base_url: str, api_key: str | None = None, name: str = "openai",
                 token_param: str = "max_completion_tokens", timeout: float = 600, retries: int = 4,
                 require_key: bool = True, extra_body: dict | None = None,
                 json_format: str = "response_format", ollama_native: bool = False,
                 temperature: float | None = None, seed: int | None = None, text_tools: bool = False,
                 sampling_note: str = "", num_ctx: int | None = None):
        self.model = model
        # Local Ollama only: the context window, sent as options.num_ctx on the NATIVE /api/chat.
        # Measured 2026-11-20 (Ollama 0.40.16): through /v1 the model loads with a 4096-token
        # context and a ~6,000-token prompt comes back as prompt_tokens=2050, i.e. silently cut;
        # /v1 cannot set the size ("create a Modelfile"). With num_ctx set, chat() goes native.
        self.num_ctx = num_ctx
        # Sent on every chat() unless the caller passes its own temperature. None = not sent
        # (DeepSeek thinking mode ignores temperature: "Has no effect in thinking mode").
        self.temperature = temperature
        self.seed = seed
        self.sampling_note = sampling_note
        # Recover tool calls written as JSON text (qwen2.5-coder via Ollama /v1 does this: it
        # returns '{"name": "read_file", "arguments": {...}}' in content with no tool_calls).
        self.text_tools = text_tools
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        self.name = name
        # OpenAI's newer models reject max_tokens; DeepSeek and most servers only know max_tokens.
        self.token_param = token_param
        self.timeout = timeout
        self.retries = retries
        self.require_key = require_key
        # Sent with every request, e.g. {"reasoning_effort": "none"} to turn local Qwen thinking off.
        self.extra_body = dict(extra_body or {})
        # How to ask for JSON: "response_format" ({"type": "json_object"}: OpenAI, DeepSeek,
        # DashScope and Ollama's /v1 all list it) or "" to never ask. Support is per model, not per
        # server: Ollama 0.40.16 answers 501 "structured output is unavailable" for qwen3.5:4b on
        # both /v1 response_format and native format=json. A rejected extra is dropped for the
        # rest of this provider's life and noted in `dropped`, so later calls do not pay for it.
        self.json_format = json_format
        self.dropped: dict[str, str] = {}
        # Local Ollama: schema-constrained completions go to the native /api/chat with the JSON
        # schema in `format` (https://docs.ollama.com/capabilities/structured-outputs: "Provide a
        # JSON schema to the `format` field."), thinking off (`think: false`).
        self.ollama_native = ollama_native
        self.last_mode = ""          # how the last complete() asked for JSON, for the audit trail
        self.last_usage: dict = {}   # tokens of the last complete(), for consolidation cost
        self.usage_log: list = []    # every complete()'s usage, so callers can price small-model spend

    def complete(self, prompt: str, max_tokens: int = 4096, json_mode: bool = False,
                 schema: dict | None = None, temperature: float | None = None) -> str:
        if "temperature" in self.dropped:
            temperature = None
        if schema is not None and self.ollama_native and "format" not in self.dropped:
            try:
                text = self._ollama_format(prompt, max_tokens, schema, temperature)
                self.last_mode = "format schema (ollama native)"
                return text
            except FatalProviderError:
                raise                       # Ollama down / model missing: not a format problem
            except ProviderError as e:
                # Observed: 501 "structured output is unavailable" for MLX builds (qwen3.5:4b on
                # Ollama 0.40.16). Remember it and use the OpenAI-compatible path instead.
                if e.status not in (400, 404, 422, 501):
                    raise
                self.dropped["format"] = f"HTTP {e.status}: {str(e).partition(': ')[2][:160]}"
        resp = self.chat("", [{"role": "user", "content": prompt}], None, max_tokens, temperature=temperature,
                         json_mode=json_mode or schema is not None)
        text = resp.text
        self.last_usage = {"input_tokens": resp.input_tokens, "cache_read_tokens": resp.cache_read_tokens,
                           "output_tokens": resp.output_tokens, "reasoning_tokens": resp.reasoning_tokens,
                           "ts": time.time()}
        self.usage_log.append(self.last_usage)
        self.last_mode = ("response_format json_object" if (json_mode or schema is not None)
                          and self.json_format == "response_format" else "prompt only")
        return text

    def _ollama_format(self, prompt: str, max_tokens: int, schema: dict, temperature: float | None = None) -> str:
        native = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        options = {"num_predict": max_tokens}
        if temperature is not None:
            options["temperature"] = temperature
        if self.seed is not None:
            options["seed"] = self.seed
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx
        body = {"model": self.model, "stream": False, "think": False, "format": schema,
                "messages": [{"role": "user", "content": prompt}], "options": options}
        data = _ollama_post(f"{native}/api/chat", body, self.timeout, self.retries, self.model)
        self.last_usage = {"input_tokens": int(data.get("prompt_eval_count") or 0), "cache_read_tokens": 0,
                           "output_tokens": int(data.get("eval_count") or 0), "reasoning_tokens": None, "ts": time.time()}
        self.usage_log.append(self.last_usage)
        return ((data.get("message") or {}).get("content") or "")

    @staticmethod
    def to_wire(system, messages: list[dict]) -> list[dict]:
        sys_text = system_text(system)
        out = [{"role": "system", "content": sys_text}] if sys_text else []
        for m in messages:
            role = m.get("role")
            if role == "user":
                out.append({"role": "user", "content": m.get("content") or ""})
            elif role == "assistant":
                msg: dict = {"role": "assistant", "content": m.get("content") or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                          "function": {"name": c["name"], "arguments": json.dumps(c["args"])}}
                                         for c in m["tool_calls"]]
                raw = m.get("raw") or {}
                if raw.get("provider") == "openai-compat" and raw.get("reasoning_content"):
                    msg["reasoning_content"] = raw["reasoning_content"]   # DeepSeek thinking + tools
                if msg["content"] is None and "tool_calls" not in msg:
                    msg["content"] = ""
                out.append(msg)
            elif role == "tool":
                out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m.get("content") or ""})
        return out

    def chat(self, system, messages, tools=None, max_tokens=8192, tool_choice="auto", temperature=None,
             json_mode: bool = False):
        if self.num_ctx and self.ollama_native and not json_mode:
            return self._ollama_chat(system, messages, tools, max_tokens, tool_choice, temperature)
        if self.require_key and not self.api_key:
            raise ProviderError(f"API key for {self.name} is not set")
        body: dict = {**self.extra_body, "model": self.model, "messages": self.to_wire(system, messages),
                      self.token_param: max_tokens}
        if temperature is None and "temperature" not in self.dropped:
            temperature = self.temperature
        if temperature is not None:
            body["temperature"] = temperature
        if self.seed is not None and "seed" not in self.dropped:
            body["seed"] = self.seed
        if tools:
            body["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                               "parameters": t["parameters"]}} for t in tools]
            body["tool_choice"] = "none" if tool_choice == "none" else "auto"
        if json_mode and self.json_format == "response_format":
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        url = f"{self.base_url}/chat/completions"
        t0 = time.perf_counter()
        try:
            data = _post_json(url, body, headers, self.timeout, self.retries)
        except ProviderError as e:
            # A server that rejects an optional extra (an older Ollama without reasoning_effort, a
            # model without structured output) gets one plain retry; the caller validates the JSON.
            # response_format goes first, alone: dropping reasoning_effort along with it would turn
            # a thinking model's thinking back on (measured: qwen3.5:4b 2.4 s -> 115 s per call).
            data, removed = None, []
            # temperature joins the second group: some reasoning models reject it (it is optional).
            for group in (["response_format"], list(self.extra_body) + ["temperature", "seed"]):
                group = [k for k in group if k in body]
                if not group or e.status not in (400, 422, 501):
                    continue
                for k in group:
                    body.pop(k)
                removed += group
                try:
                    data = _post_json(url, body, headers, self.timeout, self.retries)
                except ProviderError as again:
                    e = again
                    continue
                reason = f"HTTP {e.status}: {str(e).partition(': ')[2][:160]}"
                for k in removed:                   # everything the successful request went without
                    self.dropped[k] = reason
                    self.extra_body.pop(k, None)
                if "response_format" in removed:
                    self.json_format = ""
                break
            if data is None:
                raise e
        resp = self.from_wire(data, time.perf_counter() - t0)
        if tools and not resp.tool_calls and self.text_tools and resp.text:
            calls, err = parse_text_tool_calls(resp.text, {t["name"] for t in tools})
            if calls:
                resp.raw_text = resp.text
                resp.tool_calls, resp.tool_call_source, resp.text = calls, "text", ""
            if err:
                resp.text_tool_error = err
        return resp

    @staticmethod
    def to_native(system, messages: list[dict]) -> list[dict]:
        """Neutral -> Ollama /api/chat: arguments are objects, tool results carry tool_name."""
        sys_text = system_text(system)
        out = [{"role": "system", "content": sys_text}] if sys_text else []
        for m in messages:
            role = m.get("role")
            if role == "user":
                out.append({"role": "user", "content": m.get("content") or ""})
            elif role == "assistant":
                msg: dict = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [{"function": {"name": c["name"], "arguments": c["args"] if isinstance(c.get("args"), dict) else {}}}
                                         for c in m["tool_calls"]]
                out.append(msg)
            elif role == "tool":
                out.append({"role": "tool", "content": m.get("content") or "", "tool_name": m.get("name", "")})
        return out

    def _ollama_chat(self, system, messages, tools, max_tokens, tool_choice, temperature):
        native = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        if temperature is None:
            temperature = self.temperature
        options = {"num_predict": max_tokens, "num_ctx": self.num_ctx}
        if temperature is not None:
            options["temperature"] = temperature
        if self.seed is not None:
            options["seed"] = self.seed
        body: dict = {"model": self.model, "stream": False, "think": False,
                      "messages": self.to_native(system, messages), "options": options}
        # No tool_choice on the native API: a tools-off final answer is sent without tools.
        if tools and tool_choice != "none":
            body["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                               "parameters": t["parameters"]}} for t in tools]
        t0 = time.perf_counter()
        data = _ollama_post(f"{native}/api/chat", body, self.timeout, self.retries, self.model)
        msg = data.get("message") or {}
        calls = []
        for i, c in enumerate(msg.get("tool_calls") or []):
            fn = c.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    args = {"_invalid_arguments": args}
            if not isinstance(args, dict):
                args = {"_invalid_arguments": args}
            calls.append(ToolCall(f"call_{i}", fn.get("name", ""), args))
        prompt_n = int(data.get("prompt_eval_count") or 0)
        resp = LLMResponse(msg.get("content") or "", calls, prompt_n, int(data.get("eval_count") or 0),
                           data.get("done_reason") or "", time.perf_counter() - t0,
                           int(data.get("prompt_eval_cached_count") or 0))
        resp.tool_call_source = "native" if calls else ""
        resp.context_full = prompt_n >= self.num_ctx - 16
        if "tools" in body and not calls and self.text_tools and resp.text:
            rec, err = parse_text_tool_calls(resp.text, {t["name"] for t in tools})
            if rec:
                resp.raw_text = resp.text
                resp.tool_calls, resp.tool_call_source, resp.text = rec, "text", ""
            if err:
                resp.text_tool_error = err
        return resp

    def sampling(self) -> dict:
        out = {"temperature": None if "temperature" in self.dropped else self.temperature,
               "seed": None if "seed" in self.dropped else self.seed}
        out["note"] = self.sampling_note or ("sent on every call" if out["temperature"] is not None else "provider default")
        if self.text_tools:
            out["text_tool_recovery"] = True
        if self.num_ctx:
            out["num_ctx"] = self.num_ctx
            out["api"] = "ollama native /api/chat"
        return out

    @staticmethod
    def from_wire(data: dict, latency: float = 0.0) -> LLMResponse:
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        calls = []
        for i, c in enumerate(msg.get("tool_calls") or []):
            fn = c.get("function") or {}
            raw = fn.get("arguments")
            if raw in (None, ""):
                raw = "{}"
            # Malformed JSON is the model's mistake, not ours: hand it back as a tool error.
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
                if not isinstance(args, dict):
                    args = {"_invalid_arguments": raw}
            except json.JSONDecodeError:
                args = {"_invalid_arguments": raw}
            calls.append(ToolCall(c.get("id") or f"call_{i}", fn.get("name", ""), args))
        u = data.get("usage") or {}
        cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens") or u.get("prompt_cache_hit_tokens") or 0)
        raw_state = ({"provider": "openai-compat", "reasoning_content": msg["reasoning_content"]}
                     if msg.get("reasoning_content") else None)
        resp = LLMResponse(msg.get("content") or "", calls, int(u.get("prompt_tokens") or 0),
                           int(u.get("completion_tokens") or 0), choice.get("finish_reason") or "", latency,
                           int(cached), 0, raw_state)
        resp.tool_call_source = "native" if calls else ""
        rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
        resp.reasoning_tokens = int(rt) if isinstance(rt, (int, float)) else None
        return resp


# --------------------------------------------------------------------------- scripted (tests, offline)
class ScriptedProvider(Provider):
    """Wraps fn(system, messages, tools, tool_choice) -> LLMResponse | str. Token counts are
    estimated (chars / 4) when the script does not set them, so budget guardrails can be tested."""
    name = "scripted"

    def __init__(self, fn: Callable, model: str = "scripted"):
        self.fn = fn
        self.model = model
        self.calls = 0
        self.seen: list[dict] = []       # every request, for assertions

    def chat(self, system, messages, tools=None, max_tokens=8192, tool_choice="auto", temperature=None):
        self.calls += 1
        self.seen.append({"system": system, "messages": json.loads(json.dumps(messages)),
                          "tools": [t["name"] for t in tools or []], "tool_choice": tool_choice})
        out = self.fn(system, messages, tools, tool_choice)
        if isinstance(out, str):
            out = LLMResponse(out)
        if not out.input_tokens:
            out.input_tokens = (len(system_text(system)) + sum(len(str(m.get("content", ""))) for m in messages)) // 4
        if not out.output_tokens:
            out.output_tokens = max(1, len(out.text) // 4 + sum(len(json.dumps(c.args)) // 4 + 10
                                                                for c in out.tool_calls))
        if not out.stop_reason:
            out.stop_reason = "tool_use" if out.tool_calls else "end_turn"
        if out.tool_calls and not out.tool_call_source:
            out.tool_call_source = "native"
        return out


def make_provider(spec: str, role: str = "main") -> Provider:
    """'deepseek', 'qwen', 'claude', 'chatgpt', '<provider>:<model>', 'compat:<base_url>#<model>' or
    'mock' (offline simulated coder / consolidator). 'qwen' reads QWEN_BASE_URL / QWEN_MODEL."""
    spec = (spec or "").strip()
    if spec == "mock" or spec.startswith("mock"):
        from .mock import make_mock
        return make_mock(spec, role)
    kind, _, model = spec.partition(":")
    kind = ALIASES.get(kind, kind)
    if kind == "compat":
        base, _, model = model.partition("#")
        if not base.startswith("http"):
            raise SystemExit("compat spec must be compat:<base_url>#<model>, e.g. compat:http://localhost:8000/v1#qwen")
        return OpenAICompatProvider(model or "default", base, os.environ.get("OPENAI_COMPAT_API_KEY", ""),
                                    name="compat", token_param="max_tokens", require_key=False)
    if kind not in DEFAULTS:
        raise SystemExit(f"unknown provider {spec!r}: use deepseek, qwen, claude, chatgpt, <provider>:<m>, "
                         f"compat:<url>#<m> or mock")
    d = DEFAULTS[kind]
    if kind == "dashscope":
        return OpenAICompatProvider(model or os.environ.get("DASHSCOPE_MODEL") or d["model"],
                                    os.environ.get("DASHSCOPE_BASE_URL", d["base_url"]), provider_key(kind),
                                    name="dashscope", token_param="max_tokens")
    if kind == "qwen":
        q = qwen_settings()
        # Local: thinking off (reasoning_effort "none" is Ollama's /v1 switch, per
        # https://docs.ollama.com/api/openai-compatibility) since the small model only emits JSON
        # and latency matters; a longer timeout because a laptop CPU can be slow. QWEN_REASONING_EFFORT
        # overrides ("" to send nothing). DashScope controls thinking differently, so nothing is sent there.
        effort = os.environ.get("QWEN_REASONING_EFFORT", "none" if q["local"] else "")
        # Determinism: temperature 0 and a fixed seed on every call (Ollama's /v1 accepts both, per
        # https://docs.ollama.com/api/openai-compatibility; its native options take them too).
        # QWEN_TEMPERATURE / QWEN_SEED override ("" sends nothing).
        temp = os.environ.get("QWEN_TEMPERATURE", "0")
        seed = os.environ.get("QWEN_SEED", "0")
        text_tools = os.environ.get("CBK_TEXT_TOOL_CALLS", "1") != "0"
        num_ctx = int(os.environ.get("QWEN_NUM_CTX") or 16384) if q["local"] else None
        return OpenAICompatProvider(model or q["model"], q["base_url"], q["api_key"], name="qwen",
                                    token_param="max_tokens", require_key=False,
                                    timeout=900 if q["local"] else 600, retries=1 if q["local"] else 4,
                                    extra_body={"reasoning_effort": effort} if effort else None,
                                    ollama_native=q["local"], temperature=float(temp) if temp else None,
                                    seed=int(seed) if seed else None, text_tools=text_tools, num_ctx=num_ctx)
    model = model or (d["small"] if role == "small" else d["model"])
    key = provider_key(kind)
    if kind == "anthropic":
        return AnthropicProvider(model, key)
    if kind == "deepseek":
        # No temperature or seed: thinking mode is on (the loop needs it for tool calls), and the
        # API says temperature "Has no effect in thinking mode"; it documents no seed parameter.
        return OpenAICompatProvider(model, os.environ.get("DEEPSEEK_BASE_URL", d["base_url"]), key,
                                    name="deepseek", token_param="max_tokens",
                                    sampling_note="not sent: DeepSeek thinking mode ignores temperature; no seed parameter")
    return OpenAICompatProvider(model, os.environ.get("OPENAI_BASE_URL", d["base_url"]), key, name="openai")


def parse_json_object(text: str) -> dict | None:
    """The first-to-last brace span of a reply, parsed. None if there is none or it is invalid.
    Tolerates ```json fences and reasoning text before the JSON."""
    if not text or "{" not in text or "}" not in text:
        return None
    try:
        obj = json.loads(text[text.index("{"): text.rindex("}") + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


_TOOL_TAG = re.compile(r"</?tool_call>", re.I)
_FENCE = re.compile(r"```(?:json|tool_call)?\s*|```", re.I)


def parse_text_tool_calls(text: str, names: set[str]) -> tuple[list[ToolCall], str]:
    """Tool calls a model wrote as JSON text, and an error if the text looked like one but none
    parsed. Accepts {"name": n, "arguments": {...}} (also "parameters"/"args", or arguments as a
    JSON string), a list of them, <tool_call> tags and ```json fences. Only names in `names`
    count, so prose that happens to contain JSON is left alone.

    Why: qwen2.5-coder:7b on Ollama 0.40.16 answers a tools request with
    '{"name": "read_file", "arguments": {"path": "utils.py"}}' in `content` and no tool_calls
    (measured 2026-11-20). Every recovered turn is counted (tool_call_source "text"), so the
    invalid-output metric can report native and recovered calls separately."""
    if not text or "{" not in text:
        return [], ""
    body = _FENCE.sub("\n", _TOOL_TAG.sub("\n", text))
    dec = json.JSONDecoder()
    found: list = []
    i = 0
    while True:
        starts = [x for x in (body.find("{", i), body.find("[", i)) if x != -1]
        if not starts:
            break
        i = min(starts)
        try:
            obj, end = dec.raw_decode(body, i)
        except json.JSONDecodeError:
            i += 1
            continue
        found += obj if isinstance(obj, list) else [obj]
        i = end
    calls = []
    for obj in found:
        if not isinstance(obj, dict) or obj.get("name") not in names:
            continue
        args = next((obj[k] for k in ("arguments", "parameters", "args") if k in obj), {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {"_invalid_arguments": args}
        if not isinstance(args, dict):
            args = {"_invalid_arguments": args}
        calls.append(ToolCall(f"text_{len(calls)}", obj["name"], args))
    # Calls the text names ("name": "<tool>") but that did not parse are lost output: report them,
    # even when other calls in the same reply were recovered (they still run).
    named = sum(len(re.findall(r'"name"\s*:\s*"' + re.escape(n) + '"', text)) for n in names)
    if calls:
        lost = named - len(calls)
        return calls, (f"{lost} of {named} tool calls in the text could not be parsed and were dropped" if lost > 0 else "")
    if re.search(r'"name"\s*:', text):
        return [], "text looked like a tool call but no known tool call could be parsed from it"
    # A tool name directly followed by JSON arguments, e.g. "<tool_response>\nedit_file\n{...}"
    # (qwen2.5-coder, 2026-11-21): an attempted call in a format the harness does not run.
    hit = next((n for n in sorted(names) if re.search(r"\b" + re.escape(n) + r"\b[\s,:]*\{", text)), None)
    if hit:
        return [], f"text names {hit} followed by arguments, but not as a {{\"name\": ..., \"arguments\": ...}} call"
    return [], ""
