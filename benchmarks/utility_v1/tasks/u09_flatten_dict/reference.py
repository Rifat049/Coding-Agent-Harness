"""flatten: flatten a nested dict.

flatten(d: dict, sep: str = ".") -> dict

Specification (the same rules as README.md):

R1. Nested dicts are flattened into one level. A nested key path becomes one key, the keys joined with sep: {"a": {"b": 1}} gives {"a.b": 1}.
R2. Values that are not dicts (including lists, tuples, sets and None) are kept as they are; they are not looked into.
R3. An empty nested dict is kept as a value: {"a": {}} gives {"a": {}}. An empty input dict gives {}.
R4. Keys that are not str are converted with str() before joining, at every level: {1: {2: "x"}} gives {"1.2": "x"}.
R5. If two different paths produce the same flat key, ValueError is raised: {"a.b": 1, "a": {"b": 2}} and {1: "x", "1": "y"} both raise.
R6. sep must be a non-empty str, otherwise ValueError.
R7. The input is not modified, and the result is a new dict.
R8. d must be a dict, otherwise TypeError.
"""


def flatten(d, sep="."):
    if not isinstance(d, dict):
        raise TypeError("d must be a dict")
    if not isinstance(sep, str) or not sep:
        raise ValueError("sep must be a non-empty str")
    out = {}

    def walk(node, prefix):
        for key, value in node.items():
            flat = str(key) if prefix is None else prefix + sep + str(key)
            if isinstance(value, dict) and value:
                walk(value, flat)
            else:
                if flat in out:
                    raise ValueError(f"duplicate flat key {flat!r}")
                out[flat] = value

    walk(d, None)
    return out
