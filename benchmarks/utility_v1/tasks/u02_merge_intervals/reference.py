"""intervals: merge half-open integer intervals.

merge_intervals(intervals) -> list[tuple[int, int]]

Specification (the same rules as README.md):

R1. Each interval is a pair (start, end) of ints meaning the half-open range start <= x < end.
R2. The result is a new list of (start, end) tuples, sorted by start, in which no two intervals overlap or touch.
R3. Overlapping intervals are merged into one: (1, 5) and (3, 8) give (1, 8). An interval inside another is absorbed: (1, 10) and (2, 3) give (1, 10).
R4. Touching intervals are merged too, because the ranges are half-open: (1, 3) and (3, 5) give (1, 5).
R5. The input may be in any order and may contain duplicates. It may be a list or a tuple of pairs, and each pair may be a list or a tuple. The input is not modified.
R6. Empty intervals (start == end) contain nothing and are dropped; they never merge with or join other intervals: (1, 3), (4, 4), (5, 6) give [(1, 3), (5, 6)].
R7. An interval with start > end raises ValueError.
R8. An empty input returns [].
"""


def merge_intervals(intervals):
    items = []
    for pair in intervals:
        start, end = pair
        if start > end:
            raise ValueError(f"start > end in {pair!r}")
        if start < end:
            items.append((start, end))
    items.sort()
    out = []
    for start, end in items:
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out
