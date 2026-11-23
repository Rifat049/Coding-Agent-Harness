# utility-v1: ten bounded utility tasks

**Bounded utility task** = one small, single-file utility function (or class) with a stated spec.
Each task is solvable by editing one Python file, with the standard library only.

| Task | Kind | Module | What it does |
|---|---|---|---|
| u01_parse_duration | implement | `duration.py` | `"1h30m"` -> 5400 seconds; strict validation |
| u02_merge_intervals | implement | `intervals.py` | merge half-open integer intervals |
| u03_normalize_path | fix | `paths.py` | POSIX-style path normalisation (keeps leading `..`) |
| u04_lru_cache | implement | `lru.py` | a tiny least-recently-used cache class |
| u05_parse_ini | implement | `ini.py` | INI text -> `{section: {key: value}}` with line-numbered errors |
| u06_parse_csv | fix | `csvrows.py` | CSV rows with quoted fields and doubled quotes |
| u07_business_days | implement | `bizdays.py` | count Monday-Friday days in a range, minus holidays |
| u08_compare_versions | fix | `semver.py` | semantic-version comparison with pre-releases |
| u09_flatten_dict | implement | `flatten.py` | flatten nested dicts to dotted keys |
| u10_wrap_text | fix | `wrap.py` | greedy word wrap with long-word splitting |

## Layout

```
tasks/<id>/
  task.toml          id, kind, module, title, signature, report (fix tasks)
  workspace/         the ONLY files the agent gets
    README.md        the spec: numbered rules **R1.** ... (and the bug report for fix tasks)
    <module>.py      stub (implement) or buggy version (fix); its docstring repeats the rules
    tests/           a few visible tests
  reference.py       a known-good module (selfcheck and the good mock only)
  hidden_test.py     the untouched tests, scored from this pristine copy after the run
```

## The stated-spec rule

Every test, visible or hidden, is named `test_r<N>_...` after the rule it checks, and checks only
what that rule says. `python -m codebenchkit utility-selfcheck` enforces the mechanical part:

- every test names an existing rule, and every rule has at least one hidden test;
- the module docstring states each README rule verbatim;
- the reference passes everything, and the starting module fails visible and hidden tests;
- a fix task's buggy module still passes some hidden tests (it is a near-miss, not a stub);
- the workspace contains no hidden test or reference.

## Untouched tests

The hidden tests are never copied into the workspace. They run after the agent finishes, from the
pristine `hidden_test.py`, against only the task's module file, in a fresh scratch directory. The
agent's own `run_tests` also runs pristine copies of the visible tests, so editing them cannot help.
Any attempt to read or edit the hidden tests or the harness (a tool call naming `hidden`,
`reference.py`, `benchmarks/`, `codebenchkit/`, `../`, or a path outside the workspace) is flagged,
and so are edited, deleted or added test files and test-runner tricks in the module. A flagged item
fails, whatever its tests say.

## Frozen

Frozen at git tag `utility-v1` after calibration (a mock run and one Qwen smoke run). No task, test
or scoring change after the tag. Every run records `benchmark_sha256` over this directory plus
`codebenchkit/utility/core.py` (the pass/fail and cheat rules); equal hashes mean identical tasks,
tests and scoring.
