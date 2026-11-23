"""duration: parse a compact duration string.

parse_duration(text: str) -> int

Specification (the same rules as README.md):

R1. A duration is one or more components, each an integer followed by a unit: `h` (hours), `m` (minutes) or `s` (seconds). Example: "1h30m" is 5400.
R2. The result is the total number of seconds, as an int.
R3. Components must appear in the order h, m, s, and each unit at most once: "30m1h" and "1m1m" are invalid.
R4. Units are case-insensitive: "1H" equals "1h".
R5. Leading and trailing whitespace is ignored. Whitespace anywhere else is invalid: "1h 30m" is invalid.
R6. Numbers are one or more ASCII digits 0-9 (other digit characters are invalid). Leading zeros are allowed ("05m" is 300) and so are zero components ("0s" is 0).
R7. Any other input raises ValueError, including the empty string, a number without a unit ("90"), a unit without a number ("h"), an unknown unit ("1d"), a sign ("-1h") and a decimal ("1.5h").
R8. An argument that is not a str raises TypeError.
"""

import re

_RX = re.compile(r"(?:([0-9]+)h)?(?:([0-9]+)m)?(?:([0-9]+)s)?")


def parse_duration(text):
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    s = text.strip().lower()
    m = _RX.fullmatch(s)
    if not s or m is None or not any(m.groups()):
        raise ValueError(f"invalid duration: {text!r}")
    h, mi, se = (int(g) if g else 0 for g in m.groups())
    return h * 3600 + mi * 60 + se
