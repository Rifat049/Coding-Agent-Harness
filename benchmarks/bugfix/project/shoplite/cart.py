"""A shopping cart: quantities per SKU, an optional discount code, bulk pricing."""
from __future__ import annotations

from .catalog import Catalog
from .discounts import Discount, apply_discount, bulk_rate, parse_code
from .money import percent_of


class Cart:
    def __init__(self, catalog: Catalog, items: dict | None = None):
        self.catalog = catalog
        self.items: dict[str, int] = dict(items) if items else {}
        self.discount: Discount | None = None

    def add(self, sku: str, quantity: int = 1) -> None:
        """Add items. Raises ValueError for a quantity below 1 and ProductNotFound for an unknown SKU."""
        if quantity < 1:
            raise ValueError("quantity must be at least 1")
        self.catalog.get(sku)
        self.items[sku] = self.items.get(sku, 0) + quantity

    def remove(self, sku: str, quantity: int | None = None) -> None:
        """Remove `quantity` of a SKU (all of it when None). Removing what is not there is a no-op."""
        if sku not in self.items:
            return
        if quantity is None or quantity >= self.items[sku]:
            del self.items[sku]
        else:
            self.items[sku] -= quantity

    def count(self) -> int:
        return sum(self.items.values())

    def lines(self) -> list[tuple[str, str, int, int]]:
        """(sku, name, quantity, line total in cents), sorted by SKU."""
        out = []
        for sku in sorted(self.items):
            product = self.catalog.get(sku)
            out.append((sku, product.name, self.items[sku], product.price_cents * self.items[sku]))
        return out

    def subtotal(self) -> int:
        return sum(line[3] for line in self.lines())

    def apply_code(self, code: str) -> None:
        self.discount = parse_code(code)

    def total(self) -> int:
        """Subtotal, minus the bulk tier, minus the discount code (applied after the tier)."""
        amount = self.subtotal()
        amount -= percent_of(amount, bulk_rate(self.count()))
        return apply_discount(self.discount, amount)
