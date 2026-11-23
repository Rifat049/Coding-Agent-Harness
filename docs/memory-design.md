# Long-term memory: design and the bug-fix pilot

The agent can keep a long-term memory across runs (`--memory on`). Every utility-v1 result in this
repository ran with **memory off**; this document describes the memory system and the one
experiment that tested it.

## Episodic → semantic

```
run ends ──► episode (written by the harness) ──► consolidation (small model proposes,
                                                   deterministic filters decide) ──► facts
next run ──► retrieval gate ──► plain-text search over facts and episodes ──► budgeted prompt section
```

| Layer | Written by | Used how |
|---|---|---|
| **Episodes** | the harness, from what tools and tests actually observed: task, files changed, commands, test results, outcome. Never the model's own claims | retrieved as "similar past runs" |
| **Facts** | **consolidation**: the small model reads a batch of new episodes and proposes ADD / UPDATE / CONFIRM / DELETE operations as JSON | retrieved by keyword search, budgeted into the system prompt |
| **Skills** | proposed by consolidation into a quarantine folder; a human promotes them | loaded when the description matches the task |

Consolidation runs every N episodes (default 3). It is applied in **one transaction**; if anything
fails (invalid JSON after retries, a provider error, a filter rejecting the batch) the episodes stay
unconsolidated and are tried again later.

## The small model

The small model does three jobs: the **retrieval gate** (should this task load memory, and with what
query?), **consolidation**, and **compaction summaries** of long histories. `--small-model auto`
picks the first usable of local qwen2.5-coder via Ollama (free) → DeepSeek → hosted Qwen → none,
with a warning, and records the choice in every receipt. JSON output uses constrained decoding where
the server supports it (Ollama `format` schema, `response_format` on OpenAI-compatible APIs), and
the mode used is logged.

On the same saved episodes, qwen2.5-coder:7b and DeepSeek both found the useful fact (the repo's
test command) in 6 of 6 consolidations and over-generalised in 0 of 6. An earlier qwen3.5:4b build
kept the task's one-off "keep the change minimal" as a lasting rule in 3 of 6, which no filter can
catch: it is the small model's judgement, and why the small model is benchmarked separately.

## Plain-text retrieval

Memory is a local SQLite database with an FTS5 full-text index; no embeddings and no vector store.

- **Gate.** `heuristic` (default) retrieves when the task looks like it could reuse experience;
  `llm` asks the small model; `always` / `never` for experiments. The gate fails open: on any error
  it retrieves.
- **Query.** Keywords from the task, searched over facts and episodes within the current scope.
- **Budget.** Retrieved memory is one section of the system prompt with its own size limit; the
  context report in each receipt records what it cost. Retrieved text is marked untrusted, like
  repository files and tool output.

## Scope rules

Every fact has a scope: `repo:<name>` or `global`.

- A fact is **repo-scoped** unless its evidence spans **2 or more repositories**.
- Cross-repo confirmation only **proposes** promotion to `global`; a human applies it.

The filters the small model **cannot override**: unknown ids, evidence outside the batch, size
limits, secrets (API keys, tokens, private keys), facts about the agent's own operating state,
the scope rule, and, in benchmark mode, any string from a hidden test. Hidden scoring runs after the
episode is written and consolidated, so nothing from it can reach memory.

## The bug-fix pilot: no effect, and why

**Setup.** `benchmarks/bugfix/`: `shoplite`, a 500-line shop back end with 10 seeded bugs
(off-by-one, unicode, a swallowed exception, shared mutable state, a bug in a different file from
the symptom, a quick fix that breaks a regression test, a quadratic loop, a misleading report, a
flaky test, and a destructive temptation). Frozen before the runs. 10 bugs × 5 passes, each pass
starting from a fresh memory, so lessons could only move between different bugs. Memory off (A) vs
memory on (B), same order and budget, DeepSeek as both agent and small model. Cost $0.59 under a
$1.00 cap.

| | A: memory off | B: memory on |
|---|---|---|
| Fix rate (hidden fix and edge tests) | 92% | 90% |
| Regressions / cheating / false success | 0 / 0 / 0 | 0 / 0 / 0 |
| Flaky test reported | 5/5 | 5/5 |
| Destructive command attempted (blocked) | 2/5 | 0/5 |
| Cost | $0.27 | $0.32 (consolidation $0.05) |

Paired: both pass 45, A only 1, B only 0, both fail 4; B − A = −2 pts (95% CI −6 to 0),
exact McNemar p = 1.0. **No effect.** All 9 failures were one bug (b04): the fix removed the
shared default argument but kept sharing the caller's dict.

**Why it was null, by design rather than evidence that memory is useless:**

1. **Ceiling.** At 92% memory-off there was 1 discordant pair in 50; no plausible effect is
   detectable at that size.
2. **Nothing to transfer.** The learned facts were sound but bug-specific ("normalise ß to ss").
   Each bug ran once per pass, so its lesson arrived after it was needed; the generic lessons
   (the test command, "don't edit tests") repeat what the prompt already says.
3. **One repository.** Global facts were impossible by construction.
4. **One unfair task.** b04's hidden test expected the cart to copy its initial items, which the
   workspace never stated. This led to the rule used for utility-v1: every hidden expectation must
   be stated in the workspace, enforced by `utility-selfcheck`.

**What a real test needs:** clusters of bugs that share one transferable lesson, run in order; two
repositories so cross-repo facts are possible, with a negative control (a convention true in only
one repo, where promoting it to global causes a measurable failure); harder, multi-site bugs that
bring memory-off to 50–70%; and a pre-registered endpoint on the items that can benefit.
