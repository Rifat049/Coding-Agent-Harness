# Model selection: why DeepSeek is the main model, and what Qwen2.5-Coder showed

All numbers are from `results/utility/arms.md` (utility-v1, 10 tasks × 5 samples, memory off,
2,048 output tokens per call unless stated). Effective sample size is about 10 tasks per arm.

## The two roles

CodeBenchKit uses two models:

- **Main model**: drives the coding loop.
- **Small model**: cheap side jobs: the retrieval gate (should this task load memory?), memory
  consolidation (turn episodes into facts) and compaction summaries. `--small-model auto` picks the
  first usable of local qwen2.5-coder via Ollama → DeepSeek → hosted Qwen (DashScope) → none, and
  records the choice in every receipt.

## Why DeepSeek v4-pro is the main model

| | DeepSeek agent loop | Best Qwen arm (14B whole file) |
|---|---|---|
| Tasks solved | **10/10** | 3/10 |
| Pass rate on untouched tests | **98%** (49/50) | 30% (15/50) |
| Invalid output per turn | 14% | 0% |
| Runtime per item (median) | 36 s | 39–40 s |
| Cost per item | $0.0094 | $0 (local) |
| Cost per passed item | $0.0096 | $0 |

- **Quality.** It is the only model that solves every task, including all four "fix" tasks (no
  Qwen arm solved any of them). Task-level sign test vs the best Qwen arm: 7 higher, 0 lower, 3 tied,
  p = 0.016.
- **Reliability.** 0 cheating, 0 false-success claims, 1 regression in 50 items.
- **Cost.** About a cent per item at list prices ($1.32 / $3.96 per million input / output tokens
  at peak, half off-peak, cache hits $0.044). The whole 50-item agent-loop arm cost $0.47.
- **Tool use.** DeepSeek returns native tool calls; its invalid turns are almost all replies cut at
  the 2,048-token output limit, which in thinking mode includes reasoning. The agent loop recovers
  from them because each call is a small step.
- **Caveats.** Thinking mode ignores temperature and takes no seed, so DeepSeek runs are not
  bit-reproducible. Its one-shot score depends on the output limit: 4% at 2,048 tokens (calls cut
  before any code), 46% at 8,192 (all failures still cut; every complete answer passed). Give a
  reasoning model room per call, or use the agent loop.

## Why Qwen2.5-Coder was run

The brief names it, and it is the realistic **local, free** option: no key, no per-token cost,
data stays on the machine. It was run at 7B and 14B (Ollama GGUF builds, native `/api/chat`
with a 16k context, temperature 0, seed 0) in all three loops.

| Arm | Tasks solved | Pass | Invalid output / turn | Runtime (50 items) |
|---|---|---|---|---|
| 7B one-shot | 2/10 | 20% | 0% | 17 min |
| 7B simple loop | 2/10 | 20% | 0% | 54 min |
| 7B agent loop | 0/10 | 0% | 69% | 54 min |
| 14B one-shot | 3/10 | 30% | 0% | 39 min |
| 14B simple loop | 3/10 | 30% | 0% | 53 min |
| 14B agent loop | 1/10 | 8% | 59% | 337 min |

- **Tool driving is the bottleneck.** Qwen writes tool calls as JSON text (the harness recovers
  and counts them). In the agent loop most turns are invalid: no action, malformed calls, edits whose
  old text is not in the file. Writing the whole file, it has no invalid turns.
- **Size helps less than the loop:** 7B → 14B adds about +9 pts; agent → whole file about +21 pts.
  Neither is significant on 10 tasks (sign test p = 0.25–1.0), but no task got worse.
- **Feedback does not help at temperature 0.** On retry Qwen returned identical code in 5 of 7
  tasks, so the simple loop scored exactly like one-shot and cost up to 3× the runtime.
- **Runtime.** On one Apple-silicon laptop the 14B one-shot arm took 39 min vs 17 min for 7B, and
  the 14B agent loop needed 5–6 minutes per item.

## As the small model

qwen2.5-coder:7b is the default local small model. On the consolidation benchmark (same saved
episodes, 6 consolidations each) it matched DeepSeek: useful fact found 6/6 vs 6/6, over-generalised
0/6 vs 0/6, with constrained JSON decoding through Ollama's `format` field, 3–5 s per call and no
cost. An earlier qwen3.5:4b build over-generalised in 3 of 6 runs.

## Cost vs quality

| Option | Pass rate | Cost per 50 items | Best for |
|---|---|---|---|
| DeepSeek, agent loop | 98% | $0.47 | the main model when results matter |
| Qwen2.5-Coder 14B, one-shot | 30% | $0 | free local baseline; quick checks |
| Qwen2.5-Coder 7B, one-shot | 20% | $0 | the fastest local run (17 min) |
| Qwen2.5-Coder 7B | n/a | $0 | the small model (gate, consolidation) |

The gap from the best local arm to DeepSeek is **+68 pts**. It is model capability, not the loop:
each model was compared in its own best loop.
