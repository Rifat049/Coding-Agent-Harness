# flatten.py: flatten a nested dict

A bounded utility task: one small, single-file utility with a stated spec. Implement it from this specification.

## Interface

`flatten(d: dict, sep: str = ".") -> dict`

## Specification

Every rule below is numbered. Tests are named after the rule they check (`test_r3_...` checks R3), and no test checks anything that is not stated here.

**R1.** Nested dicts are flattened into one level. A nested key path becomes one key, the keys joined with sep: {"a": {"b": 1}} gives {"a.b": 1}.

**R2.** Values that are not dicts (including lists, tuples, sets and None) are kept as they are; they are not looked into.

**R3.** An empty nested dict is kept as a value: {"a": {}} gives {"a": {}}. An empty input dict gives {}.

**R4.** Keys that are not str are converted with str() before joining, at every level: {1: {2: "x"}} gives {"1.2": "x"}.

**R5.** If two different paths produce the same flat key, ValueError is raised: {"a.b": 1, "a": {"b": 2}} and {1: "x", "1": "y"} both raise.

**R6.** sep must be a non-empty str, otherwise ValueError.

**R7.** The input is not modified, and the result is a new dict.

**R8.** d must be a dict, otherwise TypeError.

## Tests

`tests/test_flatten.py` holds a few visible tests. Run them with `python3 -m unittest discover -s tests -t . -q`.
