"""Discount codes (SAVE<n> = n percent off, OFF<cents> = a fixed amount off) and bulk tiers."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .money import percent_of

# (minimum quantity, percent off), best tier first.
BULK_TIERS = ((50, 10), (10, 5))


@dataclass(frozen=True)
class Discount:
    kind: str          # "percent" | "fixed"
    value: int         # percent (1-99) or cents


def parse_code(code: str) -> Discount:
    """'save15' -> 15% off, 'OFF500' -> $5.00 off. Raises ValueError for anything else."""
    text = str(code).strip().upper()
    match = re.fullmatch(r"SAVE(\d{1,2})", text)
    if match:
        percent = int(match.group(1))
        if not 1 <= percent <= 99:
            raise ValueError(f"bad percentage in {code!r}")
        return Discount("percent", percent)
    match = re.fullmatch(r"OFF(\d+)", text)
    if match:
        return Discount("fixed", int(match.group(1)))
    raise ValueError(f"unknown discount code {code!r}")


def apply_discount(discount: Discount | None, subtotal_cents: int) -> int:
    """The amount after the discount. Never below zero."""
    if discount is None:
        return subtotal_cents
    if discount.kind == "percent":
        return subtotal_cents - percent_of(subtotal_cents, discount.value)
    if discount.kind == "fixed":
        return max(0, subtotal_cents - discount.value)
    raise ValueError(f"unknown discount kind {discount.kind!r}")


def bulk_rate(quantity: int) -> int:
    """Percent off for buying `quantity` items in one cart."""
    for minimum, percent in BULK_TIERS:
        if quantity >= minimum:
            return percent
    return 0
