# bizdays.py: count business days in a date range

A bounded utility task: one small, single-file utility with a stated spec. Implement it from this specification.

## Interface

`business_days(start: date, end: date, holidays=()) -> int`

## Specification

Every rule below is numbered. Tests are named after the rule they check (`test_r3_...` checks R3), and no test checks anything that is not stated here.

**R1.** business_days(start, end, holidays=()) counts the days d with start <= d <= end (both ends included) that are business days.

**R2.** A business day is a Monday, Tuesday, Wednesday, Thursday or Friday that is not in holidays.

**R3.** holidays is any iterable of datetime.date (a list, set, tuple or generator). A holiday on a Saturday or Sunday changes nothing, a holiday outside the range changes nothing, and a date listed twice counts once.

**R4.** If end is before start, the result is 0.

**R5.** start and end must be datetime.date objects that are not datetime.datetime objects; anything else raises TypeError.

**R6.** The result is an int.

## Tests

`tests/test_bizdays.py` holds a few visible tests. Run them with `python3 -m unittest discover -s tests -t . -q`.
