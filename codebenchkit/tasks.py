"""Ten bounded utility tasks.

Each task has:
  * visible tests: shown to the model in the prompt and used for repair feedback
  * hidden tests:  NEVER shown to the model; used only for final scoring
  * reference:     a known-good solution, used by `selfcheck` to prove the hidden
                   tests are correct, and by the mock backend.

Test format (JSON-serialisable so it can cross the sandbox boundary):
  {"args": [...], "expected": value}      -> fn(*args) == expected (after JSON normalisation)
  {"args": [...], "raises": "ValueError"} -> fn(*args) raises that exception (or a subclass)
"""
from __future__ import annotations

from dataclasses import dataclass, field


def ok(*args, out):
    return {"args": list(args), "expected": out}


def err(*args, exc="ValueError"):
    return {"args": list(args), "raises": exc}


@dataclass(frozen=True)
class Task:
    name: str
    signature: str
    description: str
    visible: list = field(default_factory=list)
    hidden: list = field(default_factory=list)
    reference: str = ""


TASKS: list[Task] = [
    Task(
        name="slugify",
        signature="def slugify(text: str) -> str",
        description=(
            "Convert text to a URL slug: lowercase it, replace every run of characters that are "
            "not ASCII letters or digits with a single hyphen, and strip leading/trailing hyphens."
        ),
        visible=[ok("Hello World", out="hello-world"), ok("  Already--slugged  ", out="already-slugged")],
        hidden=[
            ok("", out=""),
            ok("C++ & Rust!", out="c-rust"),
            ok("--Multiple   Spaces__and_underscores--", out="multiple-spaces-and-underscores"),
            ok("2026 Q4: Results (Final)", out="2026-q4-results-final"),
            ok("!!!", out=""),
        ],
        reference='''
import re

def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
''',
    ),
    Task(
        name="parse_duration",
        signature="def parse_duration(s: str) -> int",
        description=(
            "Parse a compact duration such as '1h30m' into total seconds. Units are d, h, m, s; each "
            "may appear at most once and they must appear in that order (d, h, m, s). Each unit is "
            "preceded by a non-negative integer. Raise ValueError for an empty or malformed string."
        ),
        visible=[ok("1h30m", out=5400), ok("45s", out=45)],
        hidden=[
            ok("2d", out=172800),
            ok("1d2h3m4s", out=93784),
            ok("90m", out=5400),
            ok("0s", out=0),
            err(""),
            err("1m1h"),
            err("10x"),
        ],
        reference='''
import re

def parse_duration(s):
    m = re.fullmatch(r"(?:(\\d+)d)?(?:(\\d+)h)?(?:(\\d+)m)?(?:(\\d+)s)?", s)
    if not s or m is None:
        raise ValueError(f"bad duration: {s!r}")
    d, h, mi, se = (int(x) if x else 0 for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + se
''',
    ),
    Task(
        name="rle_encode",
        signature="def rle_encode(s: str) -> str",
        description=(
            "Run-length encode a string: each maximal run of the same character becomes the character "
            "followed by the run length in decimal, e.g. 'aaabcc' -> 'a3b1c2'. Empty input gives ''."
        ),
        visible=[ok("aaabcc", out="a3b1c2"), ok("abc", out="a1b1c1")],
        hidden=[
            ok("", out=""),
            ok("zzzzzzzzzzzz", out="z12"),
            ok("aabbaa", out="a2b2a2"),
            ok("x", out="x1"),
        ],
        reference='''
def rle_encode(s):
    out, i = [], 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(f"{s[i]}{j - i}")
        i = j
    return "".join(out)
''',
    ),
    Task(
        name="flatten",
        signature="def flatten(nested: list) -> list",
        description=(
            "Flatten an arbitrarily nested list into a flat list, preserving left-to-right order. "
            "Only lists (and tuples) are flattened; strings and other values are kept as-is."
        ),
        visible=[ok([1, [2, [3]]], out=[1, 2, 3])],
        hidden=[
            ok([], out=[]),
            ok([[], [[]]], out=[]),
            ok([1, [2, [3, [4, [5]]]], 6], out=[1, 2, 3, 4, 5, 6]),
            ok(["ab", ["cd"]], out=["ab", "cd"]),
        ],
        reference='''
def flatten(nested):
    out = []
    for x in nested:
        if isinstance(x, (list, tuple)):
            out.extend(flatten(x))
        else:
            out.append(x)
    return out
''',
    ),
    Task(
        name="merge_intervals",
        signature="def merge_intervals(intervals: list[list[int]]) -> list[list[int]]",
        description=(
            "Merge overlapping or touching closed intervals [start, end] and return them sorted by start. "
            "Intervals that share an endpoint (e.g. [1,4] and [4,5]) are merged. Input may be unsorted."
        ),
        visible=[ok([[1, 3], [2, 6], [8, 10]], out=[[1, 6], [8, 10]])],
        hidden=[
            ok([], out=[]),
            ok([[5, 7], [1, 2]], out=[[1, 2], [5, 7]]),
            ok([[1, 4], [4, 5]], out=[[1, 5]]),
            ok([[1, 10], [2, 3], [4, 5]], out=[[1, 10]]),
            ok([[3, 3]], out=[[3, 3]]),
        ],
        reference='''
def merge_intervals(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out
''',
    ),
    Task(
        name="roman_to_int",
        signature="def roman_to_int(s: str) -> int",
        description="Convert a valid Roman numeral (1 to 3999, standard subtractive notation) to an integer.",
        visible=[ok("III", out=3), ok("MCMXCIV", out=1994)],
        hidden=[
            ok("IV", out=4),
            ok("IX", out=9),
            ok("LVIII", out=58),
            ok("MMMCMXCIX", out=3999),
            ok("XL", out=40),
            ok("CDXLIV", out=444),
        ],
        reference='''
def roman_to_int(s):
    v = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, ch in enumerate(s):
        if i + 1 < len(s) and v[ch] < v[s[i + 1]]:
            total -= v[ch]
        else:
            total += v[ch]
    return total
''',
    ),
    Task(
        name="is_valid_ipv4",
        signature="def is_valid_ipv4(s: str) -> bool",
        description=(
            "Return True if s is a dotted-decimal IPv4 address: exactly four parts separated by '.', each "
            "made only of ASCII digits, value 0-255, and no leading zeros (except the single digit '0'). "
            "No surrounding whitespace is allowed."
        ),
        visible=[ok("192.168.1.1", out=True), ok("256.1.1.1", out=False)],
        hidden=[
            ok("0.0.0.0", out=True),
            ok("255.255.255.255", out=True),
            ok("01.2.3.4", out=False),
            ok("1.2.3", out=False),
            ok("1.2.3.4.5", out=False),
            ok("a.b.c.d", out=False),
            ok("1.2.3.-4", out=False),
            ok(" 1.2.3.4", out=False),
            ok("1..2.3", out=False),
        ],
        reference='''
def is_valid_ipv4(s):
    parts = s.split(".")
    if len(parts) != 4:
        return False
    for p in parts:
        if not p or not (p.isascii() and p.isdigit()):
            return False
        if len(p) > 1 and p[0] == "0":
            return False
        if int(p) > 255:
            return False
    return True
''',
    ),
    Task(
        name="chunk",
        signature="def chunk(items: list, size: int) -> list[list]",
        description=(
            "Split items into consecutive chunks of length size; the last chunk may be shorter. "
            "Raise ValueError if size < 1."
        ),
        visible=[ok([1, 2, 3, 4, 5], 2, out=[[1, 2], [3, 4], [5]])],
        hidden=[
            ok([], 3, out=[]),
            ok([1, 2, 3], 3, out=[[1, 2, 3]]),
            ok([1, 2], 5, out=[[1, 2]]),
            ok(["a", "b", "c"], 1, out=[["a"], ["b"], ["c"]]),
            err([1, 2], 0),
            err([1, 2], -1),
        ],
        reference='''
def chunk(items, size):
    if size < 1:
        raise ValueError("size must be >= 1")
    return [list(items[i:i + size]) for i in range(0, len(items), size)]
''',
    ),
    Task(
        name="top_words",
        signature="def top_words(text: str, k: int) -> list[list]",
        description=(
            "Words are maximal runs of ASCII letters, compared case-insensitively (report them lowercase). "
            "Return the k most frequent words as [word, count] pairs, sorted by count descending and then "
            "alphabetically. Return fewer than k pairs if there are fewer distinct words."
        ),
        visible=[ok("the cat and the hat", 2, out=[["the", 2], ["and", 1]])],
        hidden=[
            ok("", 3, out=[]),
            ok("B a b A b", 5, out=[["b", 3], ["a", 2]]),
            ok("It's it, its IT.", 2, out=[["it", 3], ["its", 1]]),
            ok("x y z", 0, out=[]),
            ok("one two two three three three", 10, out=[["three", 3], ["two", 2], ["one", 1]]),
        ],
        reference='''
import re
from collections import Counter

def top_words(text, k):
    counts = Counter(re.findall(r"[a-z]+", text.lower()))
    ranked = sorted(counts.items(), key=lambda t: (-t[1], t[0]))
    return [[w, n] for w, n in ranked[:k]]
''',
    ),
    Task(
        name="camel_to_snake",
        signature="def camel_to_snake(name: str) -> str",
        description=(
            "Convert a camelCase or PascalCase identifier to snake_case. Treat runs of capitals as acronyms "
            "('HTTPServer' -> 'http_server'). A digit followed by a capital starts a new word. "
            "Already-snake_case input is returned unchanged."
        ),
        visible=[ok("camelCase", out="camel_case"), ok("HTTPServerError", out="http_server_error")],
        hidden=[
            ok("getHTTPResponse", out="get_http_response"),
            ok("already_snake", out="already_snake"),
            ok("ID", out="id"),
            ok("version2Update", out="version2_update"),
            ok("ThisIsATest", out="this_is_a_test"),
            ok("x", out="x"),
        ],
        reference='''
import re

def camel_to_snake(name):
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\\1_\\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\\1_\\2", s)
    return s.lower()
''',
    ),
]

TASKS_BY_NAME = {t.name: t for t in TASKS}


def get_tasks(names: str | None = None) -> list[Task]:
    if not names:
        return list(TASKS)
    out = []
    for n in names.split(","):
        n = n.strip()
        if n not in TASKS_BY_NAME:
            raise SystemExit(f"unknown task {n!r}; choose from {', '.join(TASKS_BY_NAME)}")
        out.append(TASKS_BY_NAME[n])
    return out
