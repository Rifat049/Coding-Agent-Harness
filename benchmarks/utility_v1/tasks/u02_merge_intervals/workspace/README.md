# intervals.py: merge half-open integer intervals

A bounded utility task: one small, single-file utility with a stated spec. Implement it from this specification.

## Interface

`merge_intervals(intervals) -> list[tuple[int, int]]`

## Specification

Every rule below is numbered. Tests are named after the rule they check (`test_r3_...` checks R3), and no test checks anything that is not stated here.

**R1.** Each interval is a pair (start, end) of ints meaning the half-open range start <= x < end.

**R2.** The result is a new list of (start, end) tuples, sorted by start, in which no two intervals overlap or touch.

**R3.** Overlapping intervals are merged into one: (1, 5) and (3, 8) give (1, 8). An interval inside another is absorbed: (1, 10) and (2, 3) give (1, 10).

**R4.** Touching intervals are merged too, because the ranges are half-open: (1, 3) and (3, 5) give (1, 5).

**R5.** The input may be in any order and may contain duplicates. It may be a list or a tuple of pairs, and each pair may be a list or a tuple. The input is not modified.

**R6.** Empty intervals (start == end) contain nothing and are dropped; they never merge with or join other intervals: (1, 3), (4, 4), (5, 6) give [(1, 3), (5, 6)].

**R7.** An interval with start > end raises ValueError.

**R8.** An empty input returns [].

## Tests

`tests/test_intervals.py` holds a few visible tests. Run them with `python3 -m unittest discover -s tests -t . -q`.
