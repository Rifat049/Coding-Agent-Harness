<!-- loop design, DeepSeek: one-shot at 8,192 output tokens (outside the frozen budget) vs agent at 2,048 -->
# Paired comparison

- **A**: deepseek:deepseek-v4-pro / prompt=simple-1  (runs/utility/DO8k_deepseek_oneshot_8192)
- **B**: deepseek:deepseek-v4-pro / prompt=baseline  (runs/utility/D_deepseek_baseline)
- Paired items: 26

> WARNING: budgets differ: A={'max_attempts': 1, 'max_new_tokens': 8192, 'max_total_tokens': 200000, 'per_test_timeout': 5.0, 'max_iterations': 20, 'max_tokens': 8192, 'token_budget': 200000, 'wall_seconds': 600, 'max_verify_rounds': 1, 'context_limit': 16000, 'compact_at': 0.7, 'trim_chars': 4000, 'trim_keep': 2000, 'doom_nudge': 3, 'doom_abort': 5, 'error_nudge': 4, 'error_abort': 8, 'final_max_tokens': 4000, 'command_timeout': 120, 'verify_timeout': 300, 'verify_cmd': 'python3 -m unittest discover -s tests -t . -q', 'gate_mode': 'heuristic', 'benchmark': True, 'consolidation_every': 0, 'summarize_compaction': True, 'force_compaction_after': 0, 'verify_without_edits': True} B={'max_attempts': 5, 'max_new_tokens': 2048, 'max_total_tokens': 200000, 'per_test_timeout': 5.0, 'max_iterations': 20, 'max_tokens': 2048, 'token_budget': 200000, 'wall_seconds': 600, 'max_verify_rounds': 5, 'context_limit': 16000, 'compact_at': 0.7, 'trim_chars': 4000, 'trim_keep': 2000, 'doom_nudge': 3, 'doom_abort': 5, 'error_nudge': 4, 'error_abort': 8, 'final_max_tokens': 4000, 'command_timeout': 120, 'verify_timeout': 300, 'verify_cmd': 'python3 -m unittest discover -s tests -t . -q', 'gate_mode': 'heuristic', 'benchmark': True, 'consolidation_every': 0, 'summarize_compaction': True, 'force_compaction_after': 0, 'verify_without_edits': True} (comparison is not budget-matched)
> WARNING: not_run items excluded: A 24, B 0 (the run stopped early; see config 'stopped')

| Metric | A | B |
|---|---|---|
| Final pass (hidden) | 46.2% | 100.0% |
| pass@1 (hidden) | 46.2% | 50.0% |
| Invalid rate (first attempt) | 0.0% | 0.0% |
| Mean attempts | 1.00 | 1.50 |
| Completion tokens / item | 6814 | 3513 |
| Gen latency mean (s) | 53.57 | 19.89 |

## Head-to-head on identical items

Both pass 12, A only 0, B only 14, neither 0.
B minus A: +53.8 pts (paired bootstrap 95% CI +34.6 to +73.1); exact McNemar p = 0.000.

Verdict: difference is statistically significant at 0.05.
Samples of one task are not independent (they share the task), so the McNemar p treats them as independent pairs and is optimistic; the per-task table below shows where the difference comes from.

| Task | A passes | B passes |
|---|---|---|
| u01_parse_duration | 2/4 | 4/4 |
| u02_merge_intervals | 3/3 | 3/3 |
| u03_normalize_path | 3/4 | 4/4 |
| u04_lru_cache | 1/1 | 1/1 |
| u05_parse_ini | 0/3 | 3/3 |
| u06_parse_csv | 0/2 | 2/2 |
| u07_business_days | 1/2 | 2/2 |
| u08_compare_versions | 1/2 | 2/2 |
| u09_flatten_dict | 1/2 | 2/2 |
| u10_wrap_text | 0/3 | 3/3 |

## utility-v1 metrics (paired items)

| Metric | A | B |
|---|---|---|
| Pass rate on untouched tests | 46.2% | 100.0% |
| Hidden test cases passed | 61.0% | 100.0% |
| Runtime, wall clock per item | 1394 s total, 62.4 s median | 1093 s total, 38.3 s median |
| of which model time | 1393 s total, 62.4 s median | 1089 s total, 38.1 s median |
| of which test time | 1 s total, 0.0 s median | 3 s total, 0.1 s median |
| Invalid-output rate, per turn | 53.8% (14/26) | 12.7% (18/142) |
| Invalid-output rate, per item | 53.8% | 53.8% |
| Tool calls recovered from text (share of turns) | 0.0% | 0.0% |
| Regression rate (fix tasks) | 0.0% | 0.0% |
| Cheat rate | 0.0% | 0.0% |
| False-success rate | 0.0% | 0.0% |
| Mean iterations | 1.0 | 5.5 |
| Tokens / item | 7,923 | 27,612 |
| Cost (list prices) | $0.3524 | $0.2546 |
| Failed: cheated | 0 | 0 |
| Failed: unchanged | 14 | 0 |
| Failed: broken | 0 | 0 |
| Failed: visible_fail | 0 | 0 |
| Failed: hidden_only | 0 | 0 |

Per-item invalid-turn share, B minus A: -42.0 pts (paired bootstrap 95% CI -59.2 to -24.0).

Task level (10 tasks, each counted once by its pass share): B higher on 8, A higher on 0, tied on 2; exact sign test p = 0.008. Distinct final files across samples: A 20, B 26 for 26 items (temperature 0 repeats are not independent).
