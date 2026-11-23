<!-- loop design, 7B: agent vs simple -->
# Paired comparison

- **A**: qwen:qwen2.5-coder:7b / prompt=baseline  (runs/utility/Q_qwen_baseline)
- **B**: qwen:qwen2.5-coder:7b / prompt=simple-5  (runs/utility/S7_qwen7b_simple)
- Paired items: 50

| Metric | A | B |
|---|---|---|
| Final pass (hidden) | 0.0% | 20.0% |
| pass@1 (hidden) | 0.0% | 20.0% |
| Invalid rate (first attempt) | 20.0% | 0.0% |
| Mean attempts | 1.40 | 1.10 |
| Completion tokens / item | 1351 | 1575 |
| Gen latency mean (s) | 46.18 | 60.62 |

## Head-to-head on identical items

Both pass 0, A only 0, B only 10, neither 40.
B minus A: +20.0 pts (paired bootstrap 95% CI +10.0 to +32.0); exact McNemar p = 0.002.

Verdict: difference is statistically significant at 0.05.
Samples of one task are not independent (they share the task), so the McNemar p treats them as independent pairs and is optimistic; the per-task table below shows where the difference comes from.

| Task | A passes | B passes |
|---|---|---|
| u01_parse_duration | 0/5 | 0/5 |
| u02_merge_intervals | 0/5 | 0/5 |
| u03_normalize_path | 0/5 | 0/5 |
| u04_lru_cache | 0/5 | 5/5 |
| u05_parse_ini | 0/5 | 0/5 |
| u06_parse_csv | 0/5 | 0/5 |
| u07_business_days | 0/5 | 5/5 |
| u08_compare_versions | 0/5 | 0/5 |
| u09_flatten_dict | 0/5 | 0/5 |
| u10_wrap_text | 0/5 | 0/5 |

## utility-v1 metrics (paired items)

| Metric | A | B |
|---|---|---|
| Pass rate on untouched tests | 0.0% | 20.0% |
| Hidden test cases passed | 22.4% | 60.3% |
| Runtime, wall clock per item | 3258 s total, 59.4 s median | 3245 s total, 46.9 s median |
| of which model time | 3233 s total, 58.8 s median | 3234 s total, 46.7 s median |
| of which test time | 24 s total, 0.5 s median | 10 s total, 0.2 s median |
| Invalid-output rate, per turn | 69.0% (218/316) | 0.0% (0/150) |
| Invalid-output rate, per item | 100.0% | 0.0% |
| Tool calls recovered from text (share of turns) | 54.4% | 0.0% |
| Regression rate (fix tasks) | 25.0% | 50.0% |
| Cheat rate | 0.0% | 0.0% |
| False-success rate | 60.0% | 0.0% |
| Mean iterations | 6.3 | 3.0 |
| Tokens / item | 23,591 | 9,212 |
| Cost (list prices) | $0.0000 | $0.0000 |
| Failed: cheated | 0 | 0 |
| Failed: unchanged | 25 | 5 |
| Failed: broken | 19 | 0 |
| Failed: visible_fail | 1 | 20 |
| Failed: hidden_only | 5 | 15 |

Per-item invalid-turn share, B minus A: -68.7 pts (paired bootstrap 95% CI -75.1 to -62.3).

Task level (10 tasks, each counted once by its pass share): B higher on 2, A higher on 0, tied on 8; exact sign test p = 0.500. Distinct final files across samples: A 17, B 12 for 50 items (temperature 0 repeats are not independent).
