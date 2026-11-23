"""bizdays: count business days in a date range.

business_days(start: date, end: date, holidays=()) -> int

Specification (the same rules as README.md):

R1. business_days(start, end, holidays=()) counts the days d with start <= d <= end (both ends included) that are business days.
R2. A business day is a Monday, Tuesday, Wednesday, Thursday or Friday that is not in holidays.
R3. holidays is any iterable of datetime.date (a list, set, tuple or generator). A holiday on a Saturday or Sunday changes nothing, a holiday outside the range changes nothing, and a date listed twice counts once.
R4. If end is before start, the result is 0.
R5. start and end must be datetime.date objects that are not datetime.datetime objects; anything else raises TypeError.
R6. The result is an int.
"""

from datetime import date


def business_days(start, end, holidays=()):
    raise NotImplementedError
