"""All arms of the utility-v1 size x loop experiment in one table, plus the paired comparisons.

usage: python3 scripts/arms_table.py --runs runs/utility --out results/utility

Reads each arm's results.jsonl/config.json (an arm that has not run yet is listed as missing), writes
arms.md (headline table, failure categories, per-task pass counts, gap decomposition) and one
`compare` file per contrast. Item-level statistics treat the 5 samples as items; because local runs
use temperature 0 the samples of a task are often identical, so the task-level sign tests (each task
counted once) are the ones to read for significance.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codebenchkit.metrics import compare_runs, mcnemar_exact, wilson  # noqa: E402
from codebenchkit.utility.bench import FAILURE_CATEGORIES, _category  # noqa: E402

ARMS = [  # (key, label, model, loop, run folder)
    ("O7", "7B one-shot", "qwen2.5-coder:7b", "one-shot", "O7_qwen7b_oneshot"),
    ("Q", "7B agent loop", "qwen2.5-coder:7b", "agent", "Q_qwen_baseline"),
    ("P", "7B agent loop, tests-first prompt", "qwen2.5-coder:7b", "agent (tests-first)", "P_qwen_tests_first"),
    ("S7", "7B simple loop", "qwen2.5-coder:7b", "simple", "S7_qwen7b_simple"),
    ("O14", "14B one-shot", "qwen2.5-coder:14b", "one-shot", "O14_qwen14b_oneshot"),
    ("A14", "14B agent loop", "qwen2.5-coder:14b", "agent", "A14_qwen14b_agent"),
    ("S14", "14B simple loop", "qwen2.5-coder:14b", "simple", "S14_qwen14b_simple"),
    ("DO", "DeepSeek v4-pro one-shot (truncated by the 2,048-token output cap: not a measure of coding ability)", "deepseek-v4-pro", "one-shot", "DO_deepseek_oneshot"),
    ("DS", "DeepSeek v4-pro simple loop (truncated by the 2,048-token output cap: not a measure of coding ability; incomplete, 10/50, cost cap)", "deepseek-v4-pro", "simple", "DS_deepseek_simple"),
    ("DO8k", "DeepSeek v4-pro one-shot, 8,192 output tokens (OUTSIDE the frozen budget; incomplete, 26/50, cost cap)", "deepseek-v4-pro", "one-shot", "DO8k_deepseek_oneshot_8192"),
    ("D", "DeepSeek v4-pro agent loop", "deepseek-v4-pro", "agent", "D_deepseek_baseline"),
]
CONTRASTS = [  # (A, B, question)
    ("Q", "P", "prompt: baseline vs tests-first (7B, agent loop)"),
    ("Q", "A14", "model size, agent loop: 7B vs 14B"),
    ("S7", "S14", "model size, simple loop: 7B vs 14B"),
    ("O7", "O14", "model size, one-shot: 7B vs 14B"),
    ("Q", "S7", "loop design, 7B: agent vs simple"),
    ("A14", "S14", "loop design, 14B: agent vs simple"),
    ("O7", "S7", "feedback, 7B: one-shot vs simple loop"),
    ("O14", "S14", "feedback, 14B: one-shot vs simple loop"),
    ("Q", "D", "model: 7B agent vs DeepSeek agent (same loop, same budget)"),
    ("A14", "D", "model: 14B agent vs DeepSeek agent"),
    ("S14", "D", "best local arm vs DeepSeek"),
    ("DO8k", "D", "loop design, DeepSeek: one-shot at 8,192 output tokens (outside the frozen budget) vs agent at 2,048"),
]
NO_STATS = {"DO", "DS"}  # truncated by the 2,048-token output cap: shown, never used in a statistical claim


def load(d: Path):
    if not (d / "results.jsonl").is_file():
        return None, None
    items = [json.loads(l) for l in (d / "results.jsonl").read_text().splitlines() if l.strip()]
    return json.loads((d / "config.json").read_text()), items


def pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def row(key, label, cfg, items):
    scored = [i for i in items if not i.get("not_run")]
    n = len(scored)
    k = sum(i["passed"] for i in scored)
    lo, hi = wilson(k, n)
    turns = sum(i["turns"] for i in scored)
    inv = sum(i["invalid_turns"] for i in scored)
    tasks = sorted({i["task"] for i in scored})
    solved = sum(1 for t in tasks if any(i["passed"] for i in scored if i["task"] == t))
    costs = [i.get("cost_usd") for i in scored]
    return {
        "key": key, "label": label, "n": n, "not_run": len(items) - n, "passes": k, "pass": k / n if n else None,
        "ci": (lo, hi), "tasks_solved": f"{solved}/{len(tasks)}",
        "hidden": sum(i["hidden_passed"] for i in scored) / max(1, sum(i["hidden_total"] for i in scored)),
        "inv_turn": inv / turns if turns else None, "truncated": sum(i.get("truncated_turns", 0) for i in scored), "inv_item": sum(i["invalid_item"] for i in scored) / n if n else None,
        "turns": turns / n if n else 0, "wall_total": sum(i["wall_s"] for i in scored),
        "wall_med": statistics.median(i["wall_s"] for i in scored) if n else 0,
        "model_total": sum(i["model_s"] for i in scored), "test_total": sum(i["test_s"] for i in scored),
        "cost": None if any(c is None for c in costs) else sum(costs),
        "cats": Counter(_category(i) for i in scored if not i["passed"]),
        "per_task": {t: (sum(i["passed"] for i in scored if i["task"] == t), sum(1 for i in scored if i["task"] == t)) for t in tasks},
        "budget": cfg["budget"], "commit": str(cfg.get("git_commit", ""))[:8], "sha": cfg.get("benchmark_sha256", "")[:12],
        "stopped": cfg.get("stopped"),
        "uncut": (sum(i["passed"] for i in scored if not i.get("truncated_turns")), sum(1 for i in scored if not i.get("truncated_turns"))),
    }


def task_sign(a, b):
    tasks = sorted(set(a["per_task"]) & set(b["per_task"]))
    sa = {t: a["per_task"][t][0] / a["per_task"][t][1] for t in tasks}
    sb = {t: b["per_task"][t][0] / b["per_task"][t][1] for t in tasks}
    up, down = sum(sb[t] > sa[t] for t in tasks), sum(sa[t] > sb[t] for t in tasks)
    return up, down, len(tasks) - up - down, mcnemar_exact(down, up)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs/utility"))
    ap.add_argument("--out", type=Path, default=Path("results/utility"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows, missing = {}, []
    for key, label, _model, _loop, folder in ARMS:
        cfg, items = load(args.runs / folder)
        if cfg is None:
            missing.append(f"{key} ({folder})")
            continue
        rows[key] = row(key, label, cfg, items)
    out = ["# utility-v1: model size x loop design", "",
           "Frozen utility-v1 (tag `utility-v1`), memory off, the same budget for every arm except the attempt count "
           "(one-shot = 1 attempt by definition), 10 tasks x 5 samples, seed 0, local models at temperature 0. "
           "Generated by `scripts/arms_table.py` from `results/utility/<arm>/`.", ""]
    if missing:
        out += [f"> Not run yet: {', '.join(missing)}", ""]
    out += ["| Arm | Pass rate on untouched tests (95% CI) | Tasks solved (any sample) | Hidden cases | Invalid output, per turn | per item | Turns cut at the output cap | Turns / item | Runtime total (median / item) | Model time | Test time | Cost |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, *_ in ARMS:
        r = rows.get(key)
        if not r:
            continue
        out.append(f"| **{key}** {r['label']} | **{pct(r['pass'])}** ({r['passes']}/{r['n']}; {pct(r['ci'][0])}-{pct(r['ci'][1])})"
                   + (f" +{r['not_run']} not_run" if r["not_run"] else "")
                   + f" | {r['tasks_solved']} | {pct(r['hidden'])} | {pct(r['inv_turn'])} | {pct(r['inv_item'])} | {r['truncated']} | {r['turns']:.1f} | "
                   f"{r['wall_total'] / 60:.1f} min ({r['wall_med']:.0f} s) | {r['model_total'] / 60:.1f} min | {r['test_total']:.0f} s | "
                   f"{'$%.4f' % r['cost'] if r['cost'] is not None else 'unknown'} |")
    out += ["", "## Failure categories (failed items, one category each, first match)", "",
            "`cheated` flagged by the cheat checks; `unchanged` final module identical to the start; `broken` the module does not "
            "import; `visible_fail` imports but visible tests fail; `hidden_only` visible tests pass, a hidden test fails.", "",
            "| Arm | " + " | ".join(FAILURE_CATEGORIES) + " | passed |", "|---|" + "---|" * (len(FAILURE_CATEGORIES) + 1)]
    for key, *_ in ARMS:
        r = rows.get(key)
        if r:
            out.append(f"| {key} | " + " | ".join(str(r["cats"].get(c, 0)) for c in FAILURE_CATEGORIES) + f" | {r['passes']} |")
    tasks = sorted({t for r in rows.values() for t in r["per_task"]})
    keys = [k for k, *_ in ARMS if k in rows]
    out += ["", "## Passes per task (out of 5 samples)", "", "| Task | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for t in tasks:
        out.append(f"| {t} | " + " | ".join(str(rows[k]["per_task"][t][0]) if t in rows[k]["per_task"] else "-"
                                               for k in keys) + " |")
    out += ["", "## Contrasts", "", "Item level: exact McNemar on (task, sample) pairs (optimistic: temperature-0 samples repeat). "
            "Task level: each task counted once by its pass share; exact sign test. Full paired reports: `compare_<A>_vs_<B>.md`.", "",
            "| Contrast | A | B | B - A (items) | Task level: B higher / A higher / tied, p |", "|---|---|---|---|---|"]
    for a, b, q in CONTRASTS:
        if a in rows and b in rows:
            ra, rb = rows[a], rows[b]
            up, down, tie, p = task_sign(ra, rb)
            out.append(f"| {q} | {a} {pct(ra['pass'])} | {b} {pct(rb['pass'])} | {(rb['pass'] - ra['pass']) * 100:+.0f} pts | "
                       f"{up} / {down} / {tie}, p = {p:.3f} |")
            text = compare_runs(args.runs / dict((k, f) for k, *_, f in ARMS)[a], args.runs / dict((k, f) for k, *_, f in ARMS)[b])
            (args.out / f"compare_{a}_vs_{b}.md").write_text(f"<!-- {q} -->\n" + text)
    if all(k in rows for k in ("Q", "A14", "S7", "S14", "D")):
        q, a14, s7, s14, d = (rows[k]["pass"] for k in ("Q", "A14", "S7", "S14", "D"))
        gap = d - q
        out += ["", "## Gap decomposition (pass rate, percentage points)", "",
                f"Gap to close: DeepSeek agent {pct(d)} minus 7B agent {pct(q)} = **{gap * 100:+.0f} pts**.", "",
                "| Effect | Within | Estimate |", "|---|---|---|",
                f"| Model size (7B -> 14B) | agent loop | {(a14 - q) * 100:+.0f} pts |",
                f"| Model size (7B -> 14B) | simple loop | {(s14 - s7) * 100:+.0f} pts |",
                f"| Loop design (agent -> simple) | 7B | {(s7 - q) * 100:+.0f} pts |",
                f"| Loop design (agent -> simple) | 14B | {(s14 - a14) * 100:+.0f} pts |",
                f"| Size x loop interaction | | {((s14 - a14) - (s7 - q)) * 100:+.0f} pts |",
                f"| Main effect of size (mean over loops) | | {((a14 - q) + (s14 - s7)) / 2 * 100:+.0f} pts |",
                f"| Main effect of loop (mean over sizes) | | {((s7 - q) + (s14 - a14)) / 2 * 100:+.0f} pts |",
                f"| Remaining gap: DeepSeek agent minus best local arm | | {(d - max(q, a14, s7, s14)) * 100:+.0f} pts |"]
    out += ["", "Budgets: " + "; ".join(f"{k} {r['budget']['max_iterations']} calls / {r['budget']['max_attempts']} test rounds / {r['budget']['max_new_tokens']:,} output tokens per call / "
                                          f"{r['budget']['max_total_tokens']:,} tokens" for k, r in rows.items()),
            "Suite hash per arm: " + ", ".join(f"{k} {r['sha']}" for k, r in rows.items()),
            "Harness commit per arm: " + ", ".join(f"{k} {r['commit']}" for k, r in rows.items())]
    rows_d = {k: r for k, r in rows.items() if k.startswith("D")}
    if len(rows_d) == 4:
        out += ["", "## DeepSeek: agent loop vs whole-file loops", "",
                "DO and DS are truncated by the 2,048-token output cap (thinking mode spends it on reasoning before any code) "
                "and are **not a measure of coding ability**; they are kept out of every statistical claim. DO8k is outside "
                "the frozen budget (8,192 output tokens per call).", "",
                "| Arm | Pass | Turns cut at the output cap / turns | Passes among items with no cut turn |", "|---|---|---|---|"]
        for k in ("DO", "DS", "DO8k", "D"):
            r = rows_d[k]
            out.append(f"| {k} | {r['passes']}/{r['n']} | {r['truncated']} / {round(r['turns'] * r['n'])} | {r['uncut'][0]}/{r['uncut'][1]} |")
    stops = [f"{k}: {r['stopped']}" for k, r in rows.items() if r["stopped"]]
    if stops:
        out += ["Stopped early: " + "; ".join(stops)]
    (args.out / "arms.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
