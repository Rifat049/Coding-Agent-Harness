"""Model backends. Every backend implements:

    generate(messages, max_new_tokens, temperature, seed, context) -> Generation

`context` carries harness metadata (task, attempt). Real backends ignore it; the mock
backend uses it to simulate a model deterministically, so the harness can be tested offline.
"""
from __future__ import annotations

import json
import os
import random
import time
import urllib.request
from dataclasses import dataclass


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class Backend:
    name = "base"

    def describe(self) -> dict:
        return {"backend": self.name}

    def generate(self, messages, max_new_tokens, temperature, seed, context=None) -> Generation:
        raise NotImplementedError


# --------------------------------------------------------------------------- mock
class MockBackend(Backend):
    """Simulated coder. `skill` = probability the first attempt is correct.

    Failure modes it simulates (all of which the harness must catch):
      no code block, syntax error, wrong answer, and *overfitting* to the visible
      examples (passes visible tests, fails hidden ones).
    """
    name = "mock"

    def __init__(self, skill: float = 0.6, label: str = "mock"):
        self.skill = skill
        self.label = label

    def describe(self):
        return {"backend": self.name, "model": self.label, "skill": self.skill}

    def generate(self, messages, max_new_tokens, temperature, seed, context=None):
        task, attempt = context["task"], context["attempt"]
        rng = random.Random(f"{self.label}|{self.skill}|{task.name}|{seed}|{attempt}")
        p_correct = self.skill if attempt == 1 else min(0.95, self.skill + 0.3)
        r = rng.random()
        if r < p_correct:
            text = f"Here is the implementation.\n\n```python\n{task.reference.strip()}\n```\n"
        else:
            mode = rng.choice(["wrong", "overfit", "syntax", "no_code"]) if attempt == 1 else rng.choice(["wrong", "overfit"])
            text = _broken(task, mode)
        toks = len(text.split())
        return Generation(text, sum(len(m["content"].split()) for m in messages), toks, 0.0)


def _broken(task, mode: str) -> str:
    if mode == "no_code":
        return f"You can solve {task.name} by iterating over the input and handling edge cases."
    if mode == "syntax":
        return f"```python\ndef {task.name}(:\n    return None\n```"
    if mode == "overfit":
        table = [[t["args"], t["expected"]] for t in task.visible if "expected" in t]
        return ("```python\n"
                f"_SEEN = {table!r}\n\n"
                f"def {task.name}(*args):\n"
                "    for a, e in _SEEN:\n"
                "        if list(args) == a:\n"
                "            return e\n"
                "    return None\n```")
    return f"```python\ndef {task.name}(*args, **kwargs):\n    return None\n```"


# --------------------------------------------------------------------------- Hugging Face
class HFBackend(Backend):
    """Local inference with transformers, e.g. Qwen/Qwen2.5-Coder-1.5B-Instruct on a Colab T4."""
    name = "hf"

    def __init__(self, model: str, dtype: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = model
        self.tok = AutoTokenizer.from_pretrained(model)
        torch_dtype = {"auto": "auto", "fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch_dtype, device_map="auto")
        self.model.eval()

    def describe(self):
        return {"backend": self.name, "model": self.model_id, "device": str(self.model.device),
                "dtype": str(self.model.dtype)}

    def generate(self, messages, max_new_tokens, temperature, seed, context=None):
        torch = self.torch
        torch.manual_seed(seed)
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok([prompt], return_tensors="pt").to(self.model.device)
        kw = {"max_new_tokens": max_new_tokens, "pad_token_id": self.tok.eos_token_id}
        if temperature > 0:
            kw.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**inputs, **kw)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.perf_counter() - t0
        new = out[0][inputs["input_ids"].shape[1]:]
        text = self.tok.decode(new, skip_special_tokens=True)
        return Generation(text, int(inputs["input_ids"].shape[1]), int(new.shape[0]), latency)


# --------------------------------------------------------------------------- OpenAI-compatible
class OpenAIBackend(Backend):
    """Any OpenAI-compatible /chat/completions server: vLLM, SGLang, Ollama, LM Studio, ..."""
    name = "openai"

    def __init__(self, model: str, base_url: str, api_key_env: str = "OPENAI_API_KEY", timeout: float = 120):
        self.model_id = model
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "EMPTY")
        self.timeout = timeout

    def describe(self):
        return {"backend": self.name, "model": self.model_id, "base_url": self.base_url}

    def generate(self, messages, max_new_tokens, temperature, seed, context=None):
        body = json.dumps({"model": self.model_id, "messages": messages, "max_tokens": max_new_tokens,
                           "temperature": temperature, "seed": seed}).encode()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.load(resp)
        latency = time.perf_counter() - t0
        usage = data.get("usage") or {}
        text = data["choices"][0]["message"].get("content") or ""
        return Generation(text, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)), latency)


def make_backend(args) -> Backend:
    if args.backend == "mock":
        return MockBackend(skill=args.skill, label=args.model or f"mock-{args.skill}")
    if args.backend == "reference":
        return MockBackend(skill=1.0, label="reference")
    if args.backend == "hf":
        return HFBackend(args.model or "Qwen/Qwen2.5-Coder-1.5B-Instruct", dtype=args.dtype)
    if args.backend == "openai":
        if not args.base_url:
            raise SystemExit("--base-url is required for the openai backend (e.g. http://localhost:8000/v1)")
        return OpenAIBackend(args.model or "Qwen/Qwen2.5-Coder-1.5B-Instruct", args.base_url, args.api_key_env)
    raise SystemExit(f"unknown backend {args.backend}")
