"""Order reports. An order is {"id": str, "lines": [{"sku": str, "qty": int, "price": "19.99"}]}."""
from __future__ import annotations

from .money import format_cents, parse_amount


def order_total(order: dict) -> int:
    return sum(parse_amount(line["price"]) * int(line["qty"]) for line in order.get("lines", []))


def daily_total(orders: list[dict]) -> str:
    """The formatted sum of all order totals, e.g. '$1,234.56'."""
    return format_cents(sum(order_total(o) for o in orders))


def unique_orders(orders: list[dict]) -> list[dict]:
    """Drop repeated order ids (a retried checkout is logged twice), keeping the first copy and
    the original order."""
    seen: set = set()
    out = []
    for order in orders:
        if order["id"] in seen:
            continue
        seen.add(order["id"])
        out.append(order)
    return out


def sales_by_product(orders: list[dict]) -> list[tuple[str, int, int]]:
    """(sku, units, revenue cents) over unique orders, most revenue first, then by SKU."""
    units: dict[str, int] = {}
    revenue: dict[str, int] = {}
    for order in unique_orders(orders):
        for line in order.get("lines", []):
            sku, qty = line["sku"], int(line["qty"])
            units[sku] = units.get(sku, 0) + qty
            revenue[sku] = revenue.get(sku, 0) + parse_amount(line["price"]) * qty
    return sorted(((s, units[s], revenue[s]) for s in units), key=lambda r: (-r[2], r[0]))


def page_count(total: int, per_page: int) -> int:
    """How many pages `total` items need. An empty list still has one (empty) page."""
    if per_page < 1:
        raise ValueError("per_page must be at least 1")
    return max(1, -(-total // per_page))


def paginate(items: list, page: int, per_page: int) -> list:
    """Page `page` (1-based). Raises ValueError for a page outside 1..page_count."""
    pages = page_count(len(items), per_page)
    if not 1 <= page <= pages:
        raise ValueError(f"page {page} is out of range 1..{pages}")
    start = (page - 1) * per_page
    return items[start:start + per_page]
