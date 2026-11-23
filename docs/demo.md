# 5-minute demo

By default `scripts/demo.sh` replays saved runs: no network, no API key, no cost, about one minute
of output. `scripts/demo.sh --live` adds a live qwen2.5-coder:7b run on 2 tasks (free, local,
a few minutes; needs Ollama and `ollama pull qwen2.5-coder:7b`).

| Time | Say | Show (what the script prints) |
|---|---|---|
| 0:00 | "CodeBenchKit wraps a code model around checks it cannot influence." | `doctor --offline`: Python, data, keys set or not (values never printed), Ollama |
| 0:30 | "First, can the checks be trusted? Every task's spec is a list of numbered rules, every test is named after one, and the reference passes while the starting code fails." | `utility-selfcheck`: 10 lines `ok ...`, `utility-v1 selfcheck passed` |
| 1:00 | "This is all the model gets for one task." | the R1..R8 rules of `u01_parse_duration` |
| 1:30 | "Each metric is proven to fire with scripted coders: a lazy one claims success, a probing one reads the hidden tests, a garbled one sends broken tool calls." | mock rows: good 100%, lazy 0% (false success), probe 0% (cheating, although its hidden tests pass), garble 100% with invalid output |
| 2:30 | "The results: Qwen2.5-Coder 7B and 14B in three loops, and DeepSeek. Pass rate on untouched tests, runtime, invalid-output rate." | the headline table of `results/utility/arms.md` |
| 3:30 | "Why is Qwen low in the agent loop? Writing the whole file without tools it passes more: driving the tools is the bottleneck, and the invalid-output rate shows where." | the Qwen rows of the same table (0% vs 20-30%) |
| 4:00 | "Stretch: paired comparisons at a fixed budget, item by item, with an exact McNemar test." | `results/utility/compare_*.md` head-to-head lines |
| 4:30 | (optional, `--live`) "And live, on this laptop." | `utility-bench --model qwen-coder --tasks u01,u10` |

Commands it runs, if you want to type them yourself:

```bash
python3 -m codebenchkit doctor --offline
python3 -m codebenchkit utility-selfcheck
python3 -m codebenchkit utility-bench --model mock:probe --tasks u03,u05 --out /tmp/cbk
cat results/utility/arms.md
cat results/utility/compare_Q_vs_D.md
python3 -m codebenchkit utility-bench --model qwen-coder --tasks u01,u10 --samples 1   # --live
```
