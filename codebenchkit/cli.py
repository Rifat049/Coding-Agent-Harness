"""Command line: python -m codebenchkit {tasks,selfcheck,run,compare}"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .backends import make_backend
from .harness import Budget, run_benchmark
from .metrics import compare_runs, render_report, summarize
from .sandbox import run_tests
from .tasks import TASKS, get_tasks


def cmd_tasks(args):
    for t in TASKS:
        print(f"{t.name:<16} visible={len(t.visible)} hidden={len(t.hidden)}  {t.signature}")


def cmd_selfcheck(args):
    """Prove the checks are trustworthy: every reference passes every visible and hidden test."""
    bad = 0
    for t in TASKS:
        for split, tests in (("visible", t.visible), ("hidden", t.hidden)):
            r = run_tests(t.reference, t.name, tests)
            good = r.all_passed(len(tests))
            bad += not good
            print(f"{'ok ' if good else 'BAD'} {t.name:<16} {split:<7} {r.n_passed}/{len(tests)} {r.detail}")
            if not good:
                for tc, c in zip(tests, r.cases):
                    if not c.get("passed"):
                        print("     ", tc, c)
    print("selfcheck passed" if not bad else f"selfcheck FAILED ({bad} suites)")
    return 1 if bad else 0


def cmd_run(args):
    tasks = get_tasks(args.tasks)
    budget = Budget(args.max_attempts, args.max_new_tokens, args.token_budget, args.timeout)
    if args.samples > 1 and args.temperature == 0 and args.backend in ("hf", "openai"):
        print("note: temperature 0 makes samples identical; use --temperature 0.2+ for multiple samples")
    backend = make_backend(args)
    name = args.name or f"{time.strftime('%Y%m%d-%H%M%S')}_{(args.model or args.backend).split('/')[-1]}_{args.prompt}"
    out = Path(args.out) / name
    print(f"Running {len(tasks)} tasks x {args.samples} samples -> {out}")
    config, items = run_benchmark(tasks, backend, budget, args.samples, args.temperature, args.prompt, out)
    attempts = [json.loads(l) for l in (out / "attempts.jsonl").read_text().splitlines() if l.strip()]
    summary = summarize(items, attempts)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    report = render_report(config, summary)
    (out / "report.md").write_text(report)
    print()
    print(report)
    print(f"Saved: {out}/report.md, summary.json, results.jsonl, attempts.jsonl")


def cmd_compare(args):
    text = compare_runs(args.run_a, args.run_b)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        print(f"Saved: {args.out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="codebenchkit", description="Wrap a code model around independent checks.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tasks", help="list the benchmark tasks").set_defaults(fn=cmd_tasks)
    sub.add_parser("selfcheck", help="verify reference solutions pass all tests").set_defaults(fn=cmd_selfcheck)

    r = sub.add_parser("run", help="run a model through the harness")
    r.add_argument("--backend", choices=["mock", "reference", "hf", "openai"], default="mock")
    r.add_argument("--model", help="HF model id or served model name (default Qwen/Qwen2.5-Coder-1.5B-Instruct)")
    r.add_argument("--base-url", help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1")
    r.add_argument("--api-key-env", default="OPENAI_API_KEY")
    r.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    r.add_argument("--skill", type=float, default=0.6, help="mock backend only: first-try success probability")
    r.add_argument("--tasks", help="comma-separated subset of task names")
    r.add_argument("--samples", type=int, default=1, help="independent samples per task (seeds 0..n-1)")
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--prompt", choices=["default", "minimal"], default="default",
                   help="default shows visible examples; minimal shows only the spec")
    r.add_argument("--max-attempts", type=int, default=3, help="1 = single shot, >1 = repair loop")
    r.add_argument("--max-new-tokens", type=int, default=512)
    r.add_argument("--token-budget", type=int, default=2048, help="completion tokens per task across attempts")
    r.add_argument("--timeout", type=float, default=2.0, help="seconds per test case")
    r.add_argument("--out", default="runs")
    r.add_argument("--name")
    r.set_defaults(fn=cmd_run)

    c = sub.add_parser("compare", help="paired comparison of two runs on identical items")
    c.add_argument("run_a")
    c.add_argument("run_b")
    c.add_argument("--out", help="write the markdown comparison here")
    c.set_defaults(fn=cmd_compare)

    from .agent.cli import register_commands
    register_commands(sub)
    args = p.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
