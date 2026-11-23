"""Tiny sample repo for the agent smoke test. `median` has a bug on even-length input."""


def mean(xs):
    if not xs:
        raise ValueError("mean of empty list")
    return sum(xs) / len(xs)


def median(xs):
    if not xs:
        raise ValueError("median of empty list")
    s = sorted(xs)
    return s[len(s) // 2]
