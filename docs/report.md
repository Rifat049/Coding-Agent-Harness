# Wrapping a code model around independent checks: utility-v1 results

*codebenchkit, project report, 2026-11-23. Every number comes from a committed file under
`results/utility/` (all-arms table: `results/utility/arms.md`). Model spend for this phase: about
$1.24 (DeepSeek agent loop $0.4698, one-shot at 2,048 output tokens $0.2119, simple loop $0.1973,
one-shot at 8,192 output tokens $0.3524, a $0.0104 consolidation re-run); local runs are free.*

## 1. Question and short answer

**Can a code model be wrapped in independent checks so that its pass rate, runtime and invalid-output
rate on bounded utility tasks are measured honestly, and how much of a small model's gap to a
frontier model comes from model size and how much from the loop around it?**

- **The checks work.** Hidden tests the model never sees decide every pass; edited visible tests are
  thrown away; cheat attempts are flagged (none occurred in 486 scored items). Scripted mocks prove
  each metric fires.
- **Task level first (effective n ≈ 10 per arm).** DeepSeek v4-pro in the agent loop solves
  **10/10 tasks**. The best Qwen2.5-Coder arms (14B one-shot or simple loop) solve **3/10**; 7B
  one-shot or simple loop **2/10**; Qwen in the agent loop **0/10** (7B) and **1/10** (14B).
- **The harness loop must match the model.** Qwen does best writing the whole file (agent → whole
  file: about **+21 pts**); model size 7B → 14B adds about **+9 pts**. Qwen ignores test feedback at
  temperature 0 (identical code on retry in 5 of 7 tasks). DeepSeek does best in the agent loop:
  98% vs **46%** for one-shot with a 4× larger output limit (8,192 tokens, outside the frozen budget;
  26 of 50 items, all 10 tasks; task level 8 higher / 0 lower / 2 tied, p = 0.008). But every one of
  its 14 one-shot failures was cut at the output limit and every one of its 12 complete answers
  passed, so the agent loop's advantage here comes from keeping each call's reasoning within the
  per-call output limit, not from better code.
- The DeepSeek one-shot and simple-loop arms at the frozen 2,048-token limit are **truncated by the
  output cap, not a measure of coding ability**, and are kept out of every statistical claim.
- With 10 tasks only the large contrasts are significant at the task level (sign test: DeepSeek
  agent vs any Qwen agent arm p = 0.002, vs the best Qwen arm p = 0.016, vs DeepSeek one-shot at
  8,192 tokens p = 0.008). Size and Qwen loop effects point the same way on every task that moved
  but are not significant (p = 0.25–1.0). The remaining gap from the best Qwen arm to DeepSeek's
  agent loop is **+68 pts**.

## 2. Setup

**utility-v1** (frozen as git tag `utility-v1` in the development repository, suite sha256 `998501120a8e…`): 10 single-file Python
utilities, 6 "implement from spec" and 4 "fix this utility". Each spec is numbered rules R1..Rn;
every visible and hidden test is named after the rule it checks, and `utility-selfcheck` enforces
that no test checks an unstated behaviour and that the reference passes while the start fails.

**One item** = one model attempting one task from a fresh copy, then scored. 10 tasks × 5 samples = 50
items per arm. **Pass** = every hidden test passes and no cheat flag.

| Loop | What the model does |
|---|---|
| **one-shot** | writes the whole file once |
| **simple** | writes the whole file; the harness runs the visible tests and feeds the failure output back; up to 5 attempts |
| **agent** | uses tools (read, edit, write, run tests), up to 20 calls and 5 verify rounds |

Same budget for every arm: **2048 output tokens per call**, 200k tokens and 600 s per item, 16k
context, memory off, seed 0. Qwen (Ollama, local) at **temperature 0 with a fixed seed**: these runs
are deterministic and reproducible, which also means the 5 samples of a task are often identical, so
the honest sample size is about **10 tasks per arm**, not 50 items. DeepSeek runs in thinking mode,
which ignores temperature and has no seed. Item-level CIs and McNemar tests are in `compare_*.md` but
are optimistic for that reason.

## 3. Results

| Arm | Pass (items) | Tasks solved | Hidden cases | Invalid output / turn | Turns cut at output cap | Turns / item | Runtime, 50 items | Cost |
|---|---|---|---|---|---|---|---|---|
| O7 Qwen 7B one-shot | 20% (10/50) | 2/10 | 60% | 0% | 0 | 1.0 | 17 min | $0 |
| S7 Qwen 7B simple | 20% (10/50) | 2/10 | 60% | 0% | 0 | 3.0 | 54 min | $0 |
| Q Qwen 7B agent | 0% (0/50) | 0/10 | 22% | 69% | 0 | 6.3 | 54 min | $0 |
| P Qwen 7B agent, tests-first | 0% (0/50) | 0/10 | 21% | 54% | 0 | 10.6 | 55 min | $0 |
| O14 Qwen 14B one-shot | 30% (15/50) | 3/10 | 82% | 0% | 0 | 1.0 | 39 min | $0 |
| S14 Qwen 14B simple | 30% (15/50) | 3/10 | 83% | 0% | 0 | 1.5 | 53 min | $0 |
| A14 Qwen 14B agent | 8% (4/50) | 1/10 | 38% | 59% | 3 | 14.8 | 337 min | $0 |
| DO DeepSeek one-shot † | 4% (2/50) | 1/10 | 27% | 96% | 48 of 50 | 1.0 | 23 min | $0.21 |
| DS DeepSeek simple †, incomplete | 20% (2/10; **40 not_run**, cost cap) | 2/7 | 61% | 94% | 45 of 48 | 4.8 | 22 min (10 items) | $0.20 |
| DO8k DeepSeek one-shot, 8,192 output tokens ‡, incomplete | 46% (12/26; **24 not_run**, cost cap) | 7/10 | 61% | 54% | 14 of 26 | 1.0 | 23 min (26 items) | $0.35 |
| D DeepSeek agent | **98% (49/50)** | **10/10** | 100% | 14% | 41 of 260 | 5.2 | 33 min | $0.47 |

† **Truncated by the 2,048-token output cap — not a measure of coding ability.** Kept out of every
statistical claim. DS is also incomplete: it stopped at the $0.40 cap shared by DO and DS after 10
of 50 items. ‡ **Outside the frozen budget**: 8,192 output tokens per call, everything else
identical; stopped at its $0.35 cap after 26 of 50 items, which cover all 10 tasks (the other 24
are `not_run` and excluded).

**Failure categories** (failed items, one each): Qwen agent arms leave the module **unchanged**
(Q 25, P 25, A14 7) or **broken** (Q 19, P 23, A14 13); Qwen whole-file arms never break it and fail
as **visible_fail** or **hidden_only** (plausible code that misses rules); DeepSeek whole-file arms
fail as **unchanged** (DO 48 of 48, DS 7 of 8, DO8k 14 of 14): no code ever arrived. Tasks solved by Qwen: u04
(LRU cache), u07 (business days), and u09 (flatten dict) at 14B; no Qwen arm solved a fix task.

**Cost.** D cost $0.0094 per item vs a $0.0054 estimate (+74%). DO cost $0.0042 per item vs a
central estimate of $0.0027 (+56%): nearly every call used the full 2048 output tokens. DO and DS
together were estimated at $0.30 (high $0.44) and cost $0.4092: the cap is checked between items, so
the last item overshot $0.40 by $0.009. DO8k was estimated at $0.006–0.011 per item (central
$0.0076) and cost **$0.0136 per item** (+79% over central): its mean output was 6,814 tokens per call,
so the $0.35 cap ($0.3524 spent) covered 26 items.

## 4. Discussion: the harness loop must match the model

**Qwen: tool driving is the bottleneck.** Qwen2.5-Coder writes tool calls as JSON text (recovered by
the harness and counted), and most agent-loop turns are invalid: for Q, 112 turns with no action,
46 malformed tool calls, 35 unusable edits (old text not in the file), 25 syntax errors; for A14,
296 unusable edits. The same model asked for the whole file has 0% invalid turns and passes 20–30%.
Loop design (agent → whole file) is worth about **+21 pts**, more than twice model size
(7B → 14B, **+9 pts**, interaction +2 pts).

**Qwen: why the simple loop equals one-shot (checked, not a bug).** Replays with full-code traces
(made during the experiment, not published) show that the loop works: attempt 2's prompt contains
attempt 1's visible-test failure output (e.g. S7 u01: two tracebacks, "Ran 3 tests: 1 passed, 2 not
passed"). But Qwen rarely revises. Of the 7 retrying tasks (S7: 5 tasks, 25 items; S14: 2 tasks,
10 items), attempt 2 was **byte-identical** to attempt 1 in 5; in S7 u01 it changed one character
(`< 3` → `!= 3`) and then repeated that file for attempts 3–5; in S14 u01 the change made the visible
tests pass but a hidden rule still failed (`hidden_only`). At temperature 0 a nearly identical
prompt gives the same reply, so feedback changed no outcome: O7 = S7 and O14 = S14 item for item.

**Qwen ignores test feedback at temperature 0** (identical code on retry in 5 of 7 tasks).

**DeepSeek: the output limit, not the loop, decides whole-file arms.** DeepSeek in thinking mode
spends its output tokens on reasoning, and the per-call cap covers reasoning plus answer. At the
frozen 2,048 tokens, **48 of 50** one-shot calls and **45 of 48** simple-loop calls hit the cap with
no code (an empty reply, `finish_reason: length`), so those arms measure the cap, not coding
ability, and no claim is made from them. With the cap raised to 8,192 (DO8k, outside the frozen
budget) DeepSeek one-shot passes **12 of 26 items (46%)**, solving 7 of 10 tasks at least once.

**Verdict: much lower than 98%, so the agent loop genuinely helps DeepSeek at these settings, but
the mechanism is the output limit.** Paired on the same items, D vs DO8k: both pass 12, agent only 14,
one-shot only 0 (item-level McNemar p < 0.001, optimistic); task level, each task once: agent higher
on 8, lower on 0, tied on 2, **sign test p = 0.008**. With effective n ≈ 10 tasks and DO8k covering
26 of 50 items, the size of the gap is uncertain, but its sign is not. The cause is clear in the
traces: **all 14 DO8k failures were cut at 8,192 tokens, and all 12 answers that finished passed.**
In the agent loop each call does one small step (read a file, make one edit), so its reasoning fits
(41 of 260 calls cut at 2,048, and the next call carried on; 49/50 passed). So the tasks are not
hard for DeepSeek's code once it finishes reasoning; whole-file answers fail when reasoning plus the
file do not fit one call. Whether a larger limit (e.g. 32k) would close the gap is not tested.

**What this means.** A fixed per-call output limit is not neutral: it favours a step-by-step loop
for a reasoning model and a single whole-file answer for a small model that cannot drive tools. The
comparison that holds is **each model in its best loop**: DeepSeek agent 98% vs Qwen 14B whole file
30%, a +68 pt gap that is model capability.

## 5. Disclosures

**Changes after the utility-v1 freeze.** Tasks, tests and scoring code are unchanged (suite hash
`998501120a8e…` identical in all 10 arms). Added after the freeze: `--loop simple` (one-shot is
`--max-verify-rounds 1`), full-code and feedback logging in simple-loop traces (before DO/DS),
failure categories, the task-level sign test in `compare`, `truncated_turns` (recomputed from traces
for Q and D), `utility-bench --resume`, helper scripts (sleep marking, the arms table, a
per-attempt diff), README and doc edits. DO8k is an extra arm outside the frozen
budget (8,192 output tokens per call instead of 2,048), approved after the DO/DS truncation. Harness commits per arm are in `arms.md`;
none changes scoring.

**Items affected by system sleep.** macOS runs processes in throttled slices while asleep, so items
in progress run slowly and can hit the 600 s guardrail. Every trace was checked against
`pmset -g log`:
- **Re-run** (marked `not_run`, re-run with `--resume` on an awake machine; originals kept in
  a superseded-results file): A14 18 items (clamshell sleep 2026-11-21 02:29–15:45; asleep they had
  14 wall-clock stops and 0 passes, awake 1 stop and 0 passes); S7 5 items (lid closed 20:02–20:34;
  1 pass before and after); O14 1 item, u05#4 (low-battery hibernate 2026-11-21 22:55 to
  2026-11-22 19:49; fail before and after). No outcome changed.
- **Kept as run**: Q u07#2, D u05#2 and u07#0, P u05#4 overlapped short sleeps (about 3 minutes); none
  hit the wall-clock guardrail. DO, DS and DO8k overlapped no sleep.

**Other deviations from the brief.** The "one Qwen smoke run" became three calibration smoke runs plus
a one-shot diagnostic. DO and DS ran on battery with `caffeinate`
(to stay in DeepSeek's off-peak price window), not on AC power; so did DO8k. DS (10/50) and DO8k
(26/50) are incomplete (cost caps). The
bug-fix v1 bench is frozen, so it does not get the new `not_run` handling.

## 6. Limitations and next steps

- **Independent samples (temperature > 0, per-sample seeds).** At temperature 0 the 5 Qwen samples
  are near-duplicates; effective n is the 10 tasks. v2 should sample at temperature > 0 with a seed
  per sample so items are independent and item-level CIs mean what they say.
- **Per-model output limits**, so reasoning models are not penalised in whole-file modes: a v2 sets
  the per-call output limit per model family (and per loop), states it before running, and checks
  in calibration that complete answers fit.
- **10 tasks** give low power for size and loop effects; a larger task set is needed to confirm them.
- **One hardware setup** (Apple GPU via Ollama); local runtimes are not comparable with API runtimes.
- Qwen agent-loop results reflect its tool-call format (JSON in text); a native-tools build may do better.

## 7. Extras

The bug-fix reliability benchmark and the long-term-memory experiment with DeepSeek (92% fixed,
memory pilot null by design) are described in [`memory-design.md`](memory-design.md); the choice
of models and their cost is in [`model-selection.md`](model-selection.md).
