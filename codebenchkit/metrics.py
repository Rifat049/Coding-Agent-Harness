"""Summaries, paired comparisons and markdown reports."""
from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def _rate(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else 0.0


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_run(run_dir):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    items = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines() if l.strip()]
    attempts = [json.loads(l) for l in (run_dir / "attempts.jsonl").read_text().splitlines() if l.strip()]
    return config, items, attempts


def summarize(items, attempts) -> dict:
    n = len(items)
    k_final = sum(i["final_hidden_pass"] for i in items)
    gen_lat = [a["gen_latency_s"] for a in attempts]
    s = {
        "n_items": n,
        "pass_at_1_hidden": _rate(i["first_hidden_pass"] for i in items),
        "final_pass_hidden": k_final / n if n else 0.0,
        "final_pass_hidden_ci95": wilson(k_final, n),
        "hidden_case_pass_rate": (sum(i["final_hidden_cases_passed"] for i in items) /
                                  max(1, sum(i["hidden_cases_total"] for i in items))),
        "final_pass_visible": _rate(i["final_visible_pass"] for i in items),
        "overfit_gap": _rate(i["final_visible_pass"] for i in items) - k_final / max(1, n),
        "invalid_rate_first_attempt": _rate(i["first_invalid"] for i in items),
        "invalid_rate_all_attempts": _rate(a["invalid"] for a in attempts),
        "invalid_breakdown": dict(Counter(a["static"] for a in attempts if a["invalid"])),
        "items_with_no_valid_candidate": sum(i["final_attempt"] is None for i in items),
        "timeout_rate": _rate(i["final_timeout"] for i in items),
        "mean_attempts": _mean(i["attempts_used"] for i in items),
        "repair_gain": k_final / max(1, n) - _rate(i["first_hidden_pass"] for i in items),
        "mean_completion_tokens_per_item": _mean(i["completion_tokens"] for i in items),
        "gen_latency_mean_s": _mean(gen_lat),
        "gen_latency_p50_s": statistics.median(gen_lat) if gen_lat else 0.0,
        "gen_latency_p95_s": (sorted(gen_lat)[min(len(gen_lat) - 1, int(0.95 * len(gen_lat)))] if gen_lat else 0.0),
        "exec_runtime_mean_s": _mean(i["final_hidden_runtime_s"] for i in items),
    }
    per_task = defaultdict(list)
    for i in items:
        per_task[i["task"]].append(i)
    s["per_task"] = {
        t: {"final_pass": _rate(x["final_hidden_pass"] for x in xs),
            "pass_at_1": _rate(x["first_hidden_pass"] for x in xs),
            "hidden_cases": f"{sum(x['final_hidden_cases_passed'] for x in xs)}/{sum(x['hidden_cases_total'] for x in xs)}",
            "visible_pass": _rate(x["final_visible_pass"] for x in xs),
            "mean_attempts": _mean(x["attempts_used"] for x in xs),
            "first_invalid": sum(x["first_invalid"] for x in xs)}
        for t, xs in per_task.items()
    }
    return s


def pct(x):
    return f"{100 * x:.1f}%"


def render_report(config, summary) -> str:
    b = config["budget"]
    lo, hi = summary["final_pass_hidden_ci95"]
    be = config["backend"]
    lines = [
        f"# CodeBenchKit run: {be.get('model', be['backend'])}",
        "",
        f"Backend `{be['backend']}` | prompt `{config['prompt_style']}` | temperature {config['temperature']} | "
        f"{len(config['tasks'])} tasks x {config['samples']} samples = {summary['n_items']} items | "
        f"wall {config.get('wall_s', '?')} s",
        f"Budget: {b['max_attempts']} attempts, {b['max_new_tokens']} tokens/gen, "
        f"{b['max_total_tokens']} tokens/task, {b['per_test_timeout']} s/test",
        "",
        "## Headline (scored on untouched hidden tests)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Final pass rate (hidden) | **{pct(summary['final_pass_hidden'])}** (95% CI {pct(lo)} to {pct(hi)}) |",
        f"| pass@1 (first attempt, hidden) | {pct(summary['pass_at_1_hidden'])} |",
        f"| Repair gain | {summary['repair_gain'] * 100:+.1f} pts |",
        f"| Hidden test cases passed | {pct(summary['hidden_case_pass_rate'])} |",
        f"| Visible-vs-hidden gap (overfitting) | {summary['overfit_gap'] * 100:+.1f} pts |",
        f"| Invalid output rate (first attempt) | {pct(summary['invalid_rate_first_attempt'])} |",
        f"| Invalid output rate (all attempts) | {pct(summary['invalid_rate_all_attempts'])} |",
        f"| Invalid breakdown | {summary['invalid_breakdown'] or 'none'} |",
        f"| Timeout rate | {pct(summary['timeout_rate'])} |",
        f"| Mean attempts used | {summary['mean_attempts']:.2f} |",
        f"| Completion tokens / item | {summary['mean_completion_tokens_per_item']:.0f} |",
        f"| Generation latency mean / p50 / p95 | {summary['gen_latency_mean_s']:.2f} / "
        f"{summary['gen_latency_p50_s']:.2f} / {summary['gen_latency_p95_s']:.2f} s |",
        f"| Sandbox runtime (hidden suite) | {summary['exec_runtime_mean_s'] * 1000:.0f} ms |",
        "",
        "## Per task",
        "",
        "| Task | Final pass | pass@1 | Hidden cases | Visible pass | Attempts | First-try invalid |",
        "|---|---|---|---|---|---|---|",
    ]
    for t, r in summary["per_task"].items():
        lines.append(f"| {t} | {pct(r['final_pass'])} | {pct(r['pass_at_1'])} | {r['hidden_cases']} | "
                     f"{pct(r['visible_pass'])} | {r['mean_attempts']:.1f} | {r['first_invalid']} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- paired comparison
def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test on discordant pairs (b: A-only wins, c: B-only wins)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_bootstrap(diffs, iters: int = 10000, seed: int = 0):
    if not diffs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(iters))
    return (means[int(0.025 * iters)], means[int(0.975 * iters) - 1])


def compare_runs(run_a, run_b) -> str:
    ca, ia, aa = load_run(run_a)
    cb, ib, ab = load_run(run_b)
    warnings = []
    if ca["budget"] != cb["budget"]:
        warnings.append(f"budgets differ: A={ca['budget']} B={cb['budget']} (comparison is not budget-matched)")
    if ca["tasks"] != cb["tasks"] or ca["samples"] != cb["samples"]:
        warnings.append("task lists or sample counts differ; only shared (task, sample) pairs are compared")
    # not_run items (a billing/auth stop) were never scored: they are excluded from the pairs.
    nr_a = sum(1 for i in ia if i.get("not_run"))
    nr_b = sum(1 for i in ib if i.get("not_run"))
    if nr_a or nr_b:
        warnings.append(f"not_run items excluded: A {nr_a}, B {nr_b} (the run stopped early; see config 'stopped')")
    ka = {(i["task"], i["sample"]): i for i in ia if not i.get("not_run")}
    kb = {(i["task"], i["sample"]): i for i in ib if not i.get("not_run")}
    keys = sorted(set(ka) & set(kb))
    both = sum(ka[k]["final_hidden_pass"] and kb[k]["final_hidden_pass"] for k in keys)
    a_only = sum(ka[k]["final_hidden_pass"] and not kb[k]["final_hidden_pass"] for k in keys)
    b_only = sum(kb[k]["final_hidden_pass"] and not ka[k]["final_hidden_pass"] for k in keys)
    neither = len(keys) - both - a_only - b_only
    diffs = [int(kb[k]["final_hidden_pass"]) - int(ka[k]["final_hidden_pass"]) for k in keys]
    lo, hi = paired_bootstrap(diffs)
    p = mcnemar_exact(a_only, b_only)
    ks = set(keys)
    sa = summarize([ka[k] for k in keys], [a for a in aa if (a["task"], a["sample"]) in ks])
    sb = summarize([kb[k] for k in keys], [a for a in ab if (a["task"], a["sample"]) in ks])

    def label(c):
        be = c["backend"]
        return f"{be.get('model', be['backend'])} / prompt={c['prompt_style']}"

    rows = [
        ("Final pass (hidden)", pct(sa["final_pass_hidden"]), pct(sb["final_pass_hidden"])),
        ("pass@1 (hidden)", pct(sa["pass_at_1_hidden"]), pct(sb["pass_at_1_hidden"])),
        ("Invalid rate (first attempt)", pct(sa["invalid_rate_first_attempt"]), pct(sb["invalid_rate_first_attempt"])),
        ("Mean attempts", f"{sa['mean_attempts']:.2f}", f"{sb['mean_attempts']:.2f}"),
        ("Completion tokens / item", f"{sa['mean_completion_tokens_per_item']:.0f}", f"{sb['mean_completion_tokens_per_item']:.0f}"),
        ("Gen latency mean (s)", f"{sa['gen_latency_mean_s']:.2f}", f"{sb['gen_latency_mean_s']:.2f}"),
    ]
    per_task = defaultdict(lambda: [0, 0, 0])
    for k in keys:
        per_task[k[0]][0] += ka[k]["final_hidden_pass"]
        per_task[k[0]][1] += kb[k]["final_hidden_pass"]
        per_task[k[0]][2] += 1
    out = [
        "# Paired comparison", "",
        f"- **A**: {label(ca)}  ({run_a})",
        f"- **B**: {label(cb)}  ({run_b})",
        f"- Paired items: {len(keys)}", "",
    ]
    out += [f"> WARNING: {w}" for w in warnings] + ([""] if warnings else [])
    out += ["| Metric | A | B |", "|---|---|---|"] + [f"| {r[0]} | {r[1]} | {r[2]} |" for r in rows]
    out += [
        "", "## Head-to-head on identical items", "",
        f"Both pass {both}, A only {a_only}, B only {b_only}, neither {neither}.",
        f"B minus A: {sum(diffs) / max(1, len(diffs)) * 100:+.1f} pts "
        f"(paired bootstrap 95% CI {lo * 100:+.1f} to {hi * 100:+.1f}); exact McNemar p = {p:.3f}.",
        "",
        ("Verdict: difference is statistically significant at 0.05." if p < 0.05 else
         "Verdict: no significant difference at 0.05. With ten tasks, add samples (--samples) for power."),
        "Samples of one task are not independent (they share the task), so the McNemar p treats them as "
        "independent pairs and is optimistic; the per-task table below shows where the difference comes from.",
        "", "| Task | A passes | B passes |", "|---|---|---|",
    ]
    out += [f"| {t} | {v[0]}/{v[2]} | {v[1]}/{v[2]} |" for t, v in per_task.items()]
    if keys and all("fixed" in ka[k] and "fixed" in kb[k] for k in keys):
        out += _reliability_rows([ka[k] for k in keys], [kb[k] for k in keys])
    if keys and all("invalid_turns" in ka[k] and "invalid_turns" in kb[k] for k in keys):
        out += _utility_rows([ka[k] for k in keys], [kb[k] for k in keys])
    return "\n".join(out) + "\n"


def _utility_rows(xa, xb) -> list[str]:
    """utility-v1 runs: the brief's three metrics plus reliability, on the same paired items."""
    def rate(xs, key):
        return pct(sum(bool(x.get(key)) for x in xs) / len(xs))

    def turns(xs):
        t = sum(x["turns"] for x in xs)
        return f"{pct(sum(x['invalid_turns'] for x in xs) / t) if t else 'n/a'} ({sum(x['invalid_turns'] for x in xs)}/{t})"

    def secs(xs, key):
        v = [x[key] for x in xs]
        return f"{sum(v):.0f} s total, {statistics.median(v):.1f} s median"

    def cost(xs):
        c = [x.get("cost_usd") for x in xs]
        return "unknown" if any(v is None for v in c) else f"${sum(c):.4f}"
    fix_a, fix_b = [x for x in xa if x.get("kind") == "fix"], [x for x in xb if x.get("kind") == "fix"]
    rows = [("Pass rate on untouched tests", rate(xa, "passed"), rate(xb, "passed")),
            ("Hidden test cases passed", pct(sum(x["hidden_passed"] for x in xa) / max(1, sum(x["hidden_total"] for x in xa))),
             pct(sum(x["hidden_passed"] for x in xb) / max(1, sum(x["hidden_total"] for x in xb)))),
            ("Runtime, wall clock per item", secs(xa, "wall_s"), secs(xb, "wall_s")),
            ("of which model time", secs(xa, "model_s"), secs(xb, "model_s")),
            ("of which test time", secs(xa, "test_s"), secs(xb, "test_s")),
            ("Invalid-output rate, per turn", turns(xa), turns(xb)),
            ("Invalid-output rate, per item", rate(xa, "invalid_item"), rate(xb, "invalid_item")),
            ("Tool calls recovered from text (share of turns)",
             pct(sum(x["text_tool_turns"] for x in xa) / max(1, sum(x["turns"] for x in xa))),
             pct(sum(x["text_tool_turns"] for x in xb) / max(1, sum(x["turns"] for x in xb)))),
            ("Regression rate (fix tasks)", rate(fix_a, "regression") if fix_a else "n/a", rate(fix_b, "regression") if fix_b else "n/a"),
            ("Cheat rate", rate(xa, "cheated"), rate(xb, "cheated")),
            ("False-success rate", rate(xa, "false_success"), rate(xb, "false_success")),
            ("Mean iterations", f"{sum(x['iterations'] for x in xa) / len(xa):.1f}", f"{sum(x['iterations'] for x in xb) / len(xb):.1f}"),
            ("Tokens / item", f"{sum(x['tokens'] for x in xa) / len(xa):,.0f}", f"{sum(x['tokens'] for x in xb) / len(xb):,.0f}"),
            ("Cost (list prices)", cost(xa), cost(xb))]
    from collections import Counter as _C
    from .utility.bench import FAILURE_CATEGORIES, _category
    ca, cb = _C(_category(x) for x in xa), _C(_category(x) for x in xb)
    rows += [(f"Failed: {c}", str(ca.get(c, 0)), str(cb.get(c, 0))) for c in FAILURE_CATEGORIES]
    # Task-level (clustered) test: samples of a task share it, and at temperature 0 they are often
    # identical, so each task counts once, by its pass share.
    tasks = sorted({x["task"] for x in xa})
    share = lambda xs, t: sum(x["passed"] for x in xs if x["task"] == t) / max(1, sum(1 for x in xs if x["task"] == t))  # noqa: E731
    b_up = sum(1 for t in tasks if share(xb, t) > share(xa, t))
    a_up = sum(1 for t in tasks if share(xa, t) > share(xb, t))
    sign_p = mcnemar_exact(a_up, b_up)
    distinct = lambda xs: sum(len({x.get("final_code", "") for x in xs if x["task"] == t}) for t in tasks)  # noqa: E731
    diffs = [x["invalid_turns"] / x["turns"] if x["turns"] else 0.0 for x in xa]
    diffs = [(y["invalid_turns"] / y["turns"] if y["turns"] else 0.0) - a for a, y in zip(diffs, xb)]
    lo, hi = paired_bootstrap(diffs)
    return ["", "## utility-v1 metrics (paired items)", "", "| Metric | A | B |", "|---|---|---|"] + \
        [f"| {r[0]} | {r[1]} | {r[2]} |" for r in rows] + \
        ["", f"Per-item invalid-turn share, B minus A: {sum(diffs) / max(1, len(diffs)) * 100:+.1f} pts "
             f"(paired bootstrap 95% CI {lo * 100:+.1f} to {hi * 100:+.1f}).",
         "", f"Task level ({len(tasks)} tasks, each counted once by its pass share): B higher on {b_up}, A higher on "
             f"{a_up}, tied on {len(tasks) - a_up - b_up}; exact sign test p = {sign_p:.3f}. Distinct final files across "
             f"samples: A {distinct(xa)}, B {distinct(xb)} for {len(xa)} items (temperature 0 repeats are not independent)."]


def _reliability_rows(xa, xb) -> list[str]:
    """Bug-fix runs only: the reliability metrics side by side, on the same paired items."""
    def rate(xs, key):
        return pct(sum(bool(x.get(key)) for x in xs) / len(xs))

    def ratio(xs):
        r = [x["minimality_ratio"] for x in xs if x.get("minimality_ratio") is not None and x.get("lines_changed")]
        return f"{sum(r) / len(r):.2f}x" if r else "n/a"

    def cost(xs):
        c = [x.get("cost_usd") for x in xs]
        return "unknown" if any(v is None for v in c) else f"${sum(c):.4f}"
    rows = [("Fix rate", rate(xa, "fixed"), rate(xb, "fixed")),
            ("Regression rate", rate(xa, "regression"), rate(xb, "regression")),
            ("Cheat rate", rate(xa, "cheated"), rate(xb, "cheated")),
            ("False-success rate", rate(xa, "false_success"), rate(xb, "false_success")),
            ("Overfit (visible pass, hidden fix fail)", rate(xa, "overfit"), rate(xb, "overfit")),
            ("Minimality (lines / reference)", ratio(xa), ratio(xb)),
            ("Touched files outside the bug's area", pct(sum(bool(x.get("files_outside_area")) for x in xa) / len(xa)),
             pct(sum(bool(x.get("files_outside_area")) for x in xb) / len(xb))),
            ("Mean iterations", f"{sum(x['iterations'] for x in xa) / len(xa):.1f}", f"{sum(x['iterations'] for x in xb) / len(xb):.1f}"),
            ("Tokens / item", f"{sum(x['tokens'] for x in xa) / len(xa):,.0f}", f"{sum(x['tokens'] for x in xb) / len(xb):,.0f}"),
            ("Cost (list prices)", cost(xa), cost(xb))]
    return ["", "## Reliability (bug-fix runs)", "", "| Metric | A | B |", "|---|---|---|"] + \
        [f"| {r[0]} | {r[1]} | {r[2]} |" for r in rows]
