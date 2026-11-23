"""The agent loop: generate -> independent checks -> feedback -> repair, under a fixed budget.

Information firewall: the model only ever sees the task description, the visible examples,
and feedback computed from the visible tests. Hidden tests run once per scored candidate and
their results are written to disk, never put back into a prompt.
"""
from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass

from .checks import extract_code, static_check
from .sandbox import run_tests
from .tasks import Task

SYSTEM_PROMPT = (
    "You are a careful Python programmer. Write a correct, self-contained solution using only the "
    "Python standard library. Reply with exactly one ```python code block containing the function "
    "(and any imports or helpers it needs). Do not include tests, example usage, or I/O."
)


@dataclass
class Budget:
    max_attempts: int = 3            # 1 = no repair, single shot
    max_new_tokens: int = 512        # per generation
    max_total_tokens: int = 2048     # completion tokens per task, across all attempts
    per_test_timeout: float = 2.0    # seconds, per test case in the sandbox


def _fmt_call(task: Task, t: dict) -> str:
    args = ", ".join(repr(a) for a in t["args"])
    return f"{task.name}({args})"


def build_prompt(task: Task, style: str = "default") -> list[dict]:
    lines = [f"Implement `{task.signature}`.", "", task.description]
    if style == "default" and task.visible:
        lines += ["", "Examples:"]
        for t in task.visible:
            if "expected" in t:
                lines.append(f"    {_fmt_call(task, t)} == {t['expected']!r}")
            else:
                lines.append(f"    {_fmt_call(task, t)} raises {t['raises']}")
        lines += ["", "Your function will also be checked against additional hidden tests, so handle edge cases."]
    elif style != "minimal" and style != "default":
        raise ValueError(f"unknown prompt style {style!r}")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n".join(lines)}]


def feedback_message(task: Task, static, visible_res) -> str:
    if static is not None and not static.valid:
        return (f"Your previous answer could not be evaluated ({static.category}: {static.detail}). "
                f"Reply with exactly one ```python code block that defines `{task.signature}`.")
    if visible_res.status != "ok":
        return (f"Your code failed before any test ran ({visible_res.status}: {visible_res.detail}). "
                "Fix it and reply with the full corrected function in one ```python block.")
    fails = []
    for t, c in zip(task.visible, visible_res.cases):
        if c.get("passed"):
            continue
        want = f"== {t['expected']!r}" if "expected" in t else f"raises {t['raises']}"
        got = c.get("error") or f"returned {c.get('got')}"
        fails.append(f"- {_fmt_call(task, t)} should {want}, but {got}")
    return ("Some example tests failed:\n" + "\n".join(fails[:5]) +
            "\n\nFix the function so it satisfies the specification in general (do not special-case the "
            "examples). Reply with the full corrected function in one ```python block.")


def evaluate_response(task: Task, text: str, tests: list, budget: Budget):
    ext = extract_code(text, task.name)
    static = static_check(ext.code, task.name)
    res = run_tests(ext.code, task.name, tests, budget.per_test_timeout) if static.valid else None
    return ext, static, res


def run_item(task: Task, sample: int, backend, budget: Budget, temperature: float, prompt_style: str):
    """Run one (task, sample) through the agent loop. Returns (item_record, attempt_records)."""
    messages = build_prompt(task, prompt_style)
    attempts, tokens_used = [], 0
    best = None  # (visible_passed, attempt_idx, code)
    n_vis = len(task.visible)

    for attempt in range(1, budget.max_attempts + 1):
        remaining = budget.max_total_tokens - tokens_used
        if remaining <= 0:
            break
        gen = backend.generate(messages, min(budget.max_new_tokens, remaining), temperature, seed=sample,
                               context={"task": task, "attempt": attempt})
        tokens_used += gen.completion_tokens
        ext, static, vis = evaluate_response(task, gen.text, task.visible, budget)
        vis_passed = vis.n_passed if vis else 0
        attempts.append({
            "task": task.name, "sample": sample, "attempt": attempt,
            "response": gen.text, "code": ext.code, "extraction": ext.how,
            "static": static.category, "static_detail": static.detail,
            "invalid": not static.valid,
            "visible": vis.to_dict() if vis else None,
            "visible_all_passed": bool(vis and vis.all_passed(n_vis)),
            "prompt_tokens": gen.prompt_tokens, "completion_tokens": gen.completion_tokens,
            "gen_latency_s": round(gen.latency_s, 4),
        })
        if static.valid and vis.status == "ok" and (best is None or vis_passed >= best[0]):
            best = (vis_passed, attempt, ext.code)
        if vis and vis.all_passed(n_vis):
            break
        if attempt < budget.max_attempts:
            messages = messages + [{"role": "assistant", "content": gen.text},
                                   {"role": "user", "content": feedback_message(task, static, vis)}]

    # ---- scoring on untouched hidden tests (never fed back) ----
    def hidden_eval(code):
        if code is None:
            return None
        return run_tests(code, task.name, task.hidden, budget.per_test_timeout)

    first = attempts[0] if attempts else None
    first_hidden = hidden_eval(first["code"]) if first and not first["invalid"] else None
    final_code = best[2] if best else None
    final_attempt = best[1] if best else None
    final_hidden = first_hidden if final_attempt == 1 else hidden_eval(final_code)
    n_hid = len(task.hidden)

    item = {
        "task": task.name, "sample": sample,
        "attempts_used": len(attempts),
        "completion_tokens": tokens_used,
        "gen_latency_s": round(sum(a["gen_latency_s"] for a in attempts), 4),
        "first_invalid": bool(first and first["invalid"]),
        "first_invalid_category": first["static"] if first and first["invalid"] else None,
        "first_visible_pass": bool(first and first["visible_all_passed"]),
        "first_hidden_pass": bool(first_hidden and first_hidden.all_passed(n_hid)),
        "final_attempt": final_attempt,
        "final_visible_pass": any(a["visible_all_passed"] for a in attempts if a["attempt"] == final_attempt),
        "final_hidden_pass": bool(final_hidden and final_hidden.all_passed(n_hid)),
        "final_hidden_cases_passed": final_hidden.n_passed if final_hidden and final_hidden.status == "ok" else 0,
        "hidden_cases_total": n_hid,
        "final_hidden_runtime_s": round(final_hidden.runtime_s, 4) if final_hidden else None,
        "final_timeout": bool(final_hidden and final_hidden.any_timeout),
        "final_hidden": final_hidden.to_dict() if final_hidden else None,
        "final_code": final_code,
    }
    return item, attempts


def run_benchmark(tasks, backend, budget: Budget, samples: int, temperature: float, prompt_style: str,
                  out_dir, progress=print):
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "backend": backend.describe(), "budget": asdict(budget), "samples": samples,
        "temperature": temperature, "prompt_style": prompt_style, "tasks": [t.name for t in tasks],
        "started": time.strftime("%Y-%m-%d %H:%M:%S"), "python": platform.python_version(),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    items = []
    t0 = time.perf_counter()
    with open(out_dir / "attempts.jsonl", "w") as fa, open(out_dir / "results.jsonl", "w") as fr:
        for task in tasks:
            for s in range(samples):
                item, attempts = run_item(task, s, backend, budget, temperature, prompt_style)
                for a in attempts:
                    fa.write(json.dumps(a) + "\n")
                fr.write(json.dumps(item) + "\n")
                fa.flush(); fr.flush()
                items.append(item)
                mark = "PASS" if item["final_hidden_pass"] else ("INVALID" if item["final_attempt"] is None else "fail")
                progress(f"  {task.name:<16} s{s}  {mark:<7} attempts={item['attempts_used']} "
                         f"hidden={item['final_hidden_cases_passed']}/{item['hidden_cases_total']}")
    config["wall_s"] = round(time.perf_counter() - t0, 2)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    return config, items
