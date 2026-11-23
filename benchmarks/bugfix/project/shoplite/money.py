"""Money is always an int number of cents. Floats never hold prices."""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation


def parse_amount(text) -> int:
    """'$1,234.50', '19.99', '-3' or 7 -> cents. Raises ValueError for empty or junk input.
    The conversion is exact (no float on the way). Sub-cent input rounds half up, away from
    zero: '0.125' -> 13, '-2.675' -> -268."""
    if text is None:
        raise ValueError("amount is required")
    s = str(text).strip().replace(",", "")
    negative = s.startswith("-")
    if negative:
        s = s[1:].strip()
    if s.startswith("$"):
        s = s[1:].strip()
    if not s:
        raise ValueError("amount is required")
    try:
        value = Decimal(s)
    except InvalidOperation:
        raise ValueError(f"not an amount: {text!r}") from None
    if not value.is_finite():
        raise ValueError(f"not an amount: {text!r}")
    cents = int((value * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return -cents if negative else cents


def format_cents(cents: int) -> str:
    """12345 -> '$123.45', -300 -> '-$3.00'."""
    sign = "-" if cents < 0 else ""
    dollars, rest = divmod(abs(int(cents)), 100)
    return f"{sign}${dollars:,}.{rest:02d}"


def percent_of(cents: int, percent: int) -> int:
    """`percent`% of an amount, rounded half up to whole cents."""
    exact = Decimal(int(cents)) * Decimal(int(percent)) / Decimal(100)
    return int(exact.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def split_evenly(cents: int, parts: int) -> list[int]:
    """Split an amount into `parts` shares that differ by at most one cent and sum exactly."""
    if parts < 1:
        raise ValueError("parts must be at least 1")
    base, extra = divmod(int(cents), parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]
