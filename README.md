# CodeBenchKit: a coding-agent harness with independent checks

CodeBenchKit wraps a code model in checks the model cannot influence. The model works on a task in
a workspace: it reads the spec, writes code and runs the visible tests. When it stops, **the harness
scores the result itself**: hidden tests the model never saw, pristine copies of the visible tests
(edits to tests are thrown away and flagged), and cheat checks. Every run writes a JSONL trace and
a receipt with tokens, runtime, invalid outputs and cost.

- Python 3.11+, **standard library only**, nothing to install.
- Models: DeepSeek (API), Qwen2.5-Coder via [Ollama](https://ollama.com) (local, free), Claude,
  ChatGPT or any OpenAI-compatible server, plus scripted mocks for offline testing.
- Three ways to drive the model (`--loop`): an **agent loop** with tools, a **simple loop** (whole
  file, visible-test feedback, up to N attempts) and **one-shot** (`--loop simple --max-verify-rounds 1`).

## Brief → where it's covered

| Brief | Where | Evidence |
|---|---|---|
| **First demo:** run Qwen2.5-Coder on ten bounded utility tasks | `utility-bench --model qwen-coder` on **utility-v1** (`benchmarks/utility_v1/`, 10 tasks, frozen, suite sha256 `998501120a8e…`) | `results/utility/arms.md` |
| **Proof it works:** pass rate on untouched tests | hidden tests live only in `benchmarks/utility_v1/tasks/*/hidden_test.py` and run after the model stops, against the module alone; read or edit attempts are flagged | "Pass rate on untouched tests" per arm and per task |
| ... plus runtime | wall clock per item and in total, split into model time and test time | "Runtime" columns |
| ... plus invalid-output rate | per turn and per item: malformed tool call, unusable edit, syntax error, or no action | "Invalid output" columns; definitions in [`docs/report.md`](docs/report.md) |
| **Stretch:** paired model or prompt comparisons with fixed budgets | `compare` pairs items (task, sample): exact McNemar, paired bootstrap, task-level sign test; warns if budgets differ | `results/utility/compare_*.md` |
| "independent checks" | trusted test runs on pristine copies, `utility-selfcheck` (every test maps to a stated rule), cheat flags, mocks that prove each metric fires | `utility-selfcheck`, `tests/test_utility_suite.py` |

## Quick start (free, offline, no API key)

```bash
git clone https://github.com/Rifat049/Coding-Agent-Harness.git && cd Coding-Agent-Harness

python3 -m codebenchkit doctor --offline        # is this machine ready? (no network, no key needed)
python3 -m codebenchkit utility-selfcheck       # every task fully specified; references pass, starting code fails
python3 -m unittest discover -s tests           # ~200 tests, all mocked or local (~35 s)
python3 -m codebenchkit utility-bench --model mock --out /tmp/cbk   # whole pipeline with a scripted coder
scripts/demo.sh                                 # 5-minute guided replay of the saved results (docs/demo.md)
```

Live runs, optional:

```bash
# Local and free: Qwen2.5-Coder via Ollama
ollama pull qwen2.5-coder:7b
python3 -m codebenchkit utility-bench --model qwen-coder --tasks u01,u10 --samples 1

# Paid: DeepSeek (needs DEEPSEEK_API_KEY; --max-cost stops the run at a spend cap)
python3 -m codebenchkit doctor --model deepseek      # also shows the account balance (never the key)
python3 -m codebenchkit utility-bench --model deepseek --tasks u01,u10 --samples 1 --max-cost 0.05
```

## Headline results (utility-v1, 10 tasks × 5 samples)

Same budget for every arm: 2,048 output tokens per call, 20 model calls, 600 s per item, memory off.
Local runs used temperature 0 and a fixed seed, so they are deterministic and reproducible; it also
means the 5 samples of a task are near-copies, so **the effective sample size is about 10 tasks
per arm**.

**Main model: DeepSeek v4-pro.**

| Arm | Tasks solved | Pass rate on untouched tests | Invalid output / turn | Runtime (50 items) | Cost |
|---|---|---|---|---|---|
| **DeepSeek agent loop** | **10/10** | **98%** (49/50) | 14% | 33 min | $0.47 |
| DeepSeek one-shot, 8,192 output tokens (outside the fixed budget; 26 of 50 items, cost cap) | 7/10 | 46% (12/26) | 54% | 23 min | $0.35 |

**The brief's model: Qwen2.5-Coder (local, free).**

| Arm | Tasks solved | Pass rate on untouched tests | Invalid output / turn | Runtime (50 items) |
|---|---|---|---|---|
| 14B one-shot / simple loop | 3/10 | 30% (15/50) | 0% | 39 / 53 min |
| 7B one-shot / simple loop | 2/10 | 20% (10/50) | 0% | 17 / 54 min |
| 14B agent loop | 1/10 | 8% (4/50) | 59% | 337 min |
| 7B agent loop | 0/10 | 0% (0/50) | 69% | 54 min |

**Why Qwen scores lower:** it cannot reliably drive the tools (most agent-loop turns are malformed
calls, edits that don't match the file, or no action), so it does best writing the whole file,
where loop design is worth about +21 pts and 7B → 14B about +9 pts.

Findings in one line each:
- **The harness loop must match the model.** Qwen does best writing whole files; DeepSeek does best
  in the agent loop (98% vs 46% one-shot; task level 8 / 0 / 2, sign test p = 0.008). DeepSeek's
  one-shot failures were all cut at the output limit, and every answer that finished passed.
- **Qwen ignores test feedback at temperature 0:** identical code on retry in 5 of 7 tasks, so its
  simple loop equals one-shot item for item.
- **No cheating** in 486 scored items. DeepSeek one-shot and simple loop at the fixed 2,048-token
  limit are in `arms.md`, but they are truncated by the limit and kept out of every claim.

Full table: [`results/utility/arms.md`](results/utility/arms.md). Write-up:
[`docs/report.md`](docs/report.md). Model choice and cost: [`docs/model-selection.md`](docs/model-selection.md).
Long-term memory design and the bug-fix pilot: [`docs/memory-design.md`](docs/memory-design.md).

## utility-v1: ten bounded utility tasks

**"Bounded utility tasks" = small, single-file utility functions with a stated spec.** Six are
"implement from spec" (duration parsing, interval merging, LRU cache, INI parsing, business days,
dict flattening) and four are "fix this utility" (path normalisation, CSV parsing, version
comparison, text wrapping). The model gets a workspace with `README.md` (the spec as numbered rules
R1..Rn, repeated in the module docstring), the module and a few visible tests. Every test, visible
or hidden, is named `test_r<N>_...` after the rule it checks, and `utility-selfcheck` enforces that
no test checks an unstated behaviour.

**An item** is one model attempting one task from a fresh copy, then scored. **Pass** = every
hidden test passes and no cheat flag. Each arm runs 10 tasks × 5 samples = 50 items.

## Layout

```
codebenchkit/          the harness: providers, agent loop, tools, memory, scoring, metrics, CLI
  utility/             utility-v1 runner, simple loop, scoring, mocks
  bugfix/              bug-fix reliability benchmark runner (the memory pilot)
  agent/               providers (DeepSeek, Qwen/Ollama, Claude, OpenAI-compatible), loop, tools, memory
benchmarks/utility_v1/ the 10 tasks: workspaces, references, hidden tests
benchmarks/bugfix/     shoplite and its 10 seeded bugs
tests/                 offline unit tests (mocks and local only)
scripts/               demo.sh (guided replay), arms_table.py (all-arms table from run folders)
results/utility/       arms.md, summary.json (one summary per arm), key paired comparisons, an example trace and receipt
docs/                  report.md, model-selection.md, memory-design.md, demo.md
```

Run folders (`runs/`) are not published; `scripts/arms_table.py --runs runs/utility --out results/utility`
rebuilds `arms.md` from your own runs.
