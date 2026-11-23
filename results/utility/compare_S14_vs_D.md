<!-- best local arm vs DeepSeek -->
# Paired comparison

- **A**: qwen:qwen2.5-coder:14b / prompt=simple-5  (runs/utility/S14_qwen14b_simple)
- **B**: deepseek:deepseek-v4-pro / prompt=baseline  (runs/utility/D_deepseek_baseline)
- Paired items: 50

| Metric | A | B |
|---|---|---|
| Final pass (hidden) | 30.0% | 98.0% |
| pass@1 (hidden) | 30.0% | 44.0% |
| Invalid rate (first attempt) | 0.0% | 0.0% |
| Mean attempts | 1.10 | 1.58 |
| Completion tokens / item | 817 | 3393 |
| Gen latency mean (s) | 58.11 | 18.49 |

## Head-to-head on identical items

Both pass 15, A only 0, B only 34, neither 1.
B minus A: +68.0 pts (paired bootstrap 95% CI +54.0 to +80.0); exact McNemar p = 0.000.

Verdict: difference is statistically significant at 0.05.
Samples of one task are not independent (they share the task), so the McNemar p treats them as independent pairs and is optimistic; the per-task table below shows where the difference comes from.

| Task | A passes | B passes |
|---|---|---|
| u01_parse_duration | 0/5 | 5/5 |
| u02_merge_intervals | 0/5 | 5/5 |
| u03_normalize_path | 0/5 | 5/5 |
| u04_lru_cache | 5/5 | 5/5 |
| u05_parse_ini | 0/5 | 5/5 |
| u06_parse_csv | 0/5 | 4/5 |
| u07_business_days | 5/5 | 5/5 |
| u08_compare_versions | 0/5 | 5/5 |
| u09_flatten_dict | 5/5 | 5/5 |
| u10_wrap_text | 0/5 | 5/5 |

## utility-v1 metrics (paired items)

| Metric | A | B |
|---|---|---|
| Pass rate on untouched tests | 30.0% | 98.0% |
| Hidden test cases passed | 83.1% | 99.8% |
| Runtime, wall clock per item | 3202 s total, 39.1 s median | 2003 s total, 35.6 s median |
| of which model time | 3196 s total, 39.0 s median | 1996 s total, 35.5 s median |
| of which test time | 5 s total, 0.1 s median | 6 s total, 0.1 s median |
| Invalid-output rate, per turn | 0.0% (0/75) | 14.2% (37/260) |
| Invalid-output rate, per item | 0.0% | 60.0% |
| Tool calls recovered from text (share of turns) | 0.0% | 0.0% |
| Regression rate (fix tasks) | 25.0% | 5.0% |
| Cheat rate | 0.0% | 0.0% |
| False-success rate | 0.0% | 0.0% |
| Mean iterations | 1.5 | 5.2 |
| Tokens / item | 3,519 | 25,718 |
| Cost (list prices) | $0.0000 | $0.4698 |
| Failed: cheated | 0 | 0 |
| Failed: unchanged | 0 | 0 |
| Failed: broken | 0 | 0 |
| Failed: visible_fail | 5 | 0 |
| Failed: hidden_only | 30 | 1 |

Per-item invalid-turn share, B minus A: +13.4 pts (paired bootstrap 95% CI +9.9 to +16.9).

Task level (10 tasks, each counted once by its pass share): B higher on 7, A higher on 0, tied on 3; exact sign test p = 0.016. Distinct final files across samples: A 10, B 50 for 50 items (temperature 0 repeats are not independent).
