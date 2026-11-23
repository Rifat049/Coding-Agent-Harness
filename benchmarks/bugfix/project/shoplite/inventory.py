"""Stock levels with reservations (held for a checkout) that can be released or committed."""
from __future__ import annotations


class OutOfStock(Exception):
    def __init__(self, sku: str, wanted: int, available: int):
        super().__init__(f"{sku}: wanted {wanted}, only {available} available")
        self.sku, self.wanted, self.available = sku, wanted, available


class Inventory:
    def __init__(self, stock: dict | None = None):
        self._stock: dict[str, int] = dict(stock or {})
        self._reserved: dict[str, int] = {}

    def on_hand(self, sku: str) -> int:
        return self._stock.get(sku, 0)

    def reserved(self, sku: str) -> int:
        return self._reserved.get(sku, 0)

    def available(self, sku: str) -> int:
        return self.on_hand(sku) - self.reserved(sku)

    def restock(self, sku: str, quantity: int) -> None:
        if quantity < 1:
            raise ValueError("quantity must be at least 1")
        self._stock[sku] = self.on_hand(sku) + quantity

    def reserve(self, sku: str, quantity: int) -> None:
        if quantity < 1:
            raise ValueError("quantity must be at least 1")
        if self.available(sku) < quantity:
            raise OutOfStock(sku, quantity, self.available(sku))
        self._reserved[sku] = self.reserved(sku) + quantity

    def release(self, sku: str, quantity: int) -> None:
        """Give back part (or all) of a reservation. Releasing more than is reserved releases all."""
        if quantity < 1:
            raise ValueError("quantity must be at least 1")
        self._reserved[sku] = max(0, self.reserved(sku) - quantity)

    def commit(self, sku: str, quantity: int) -> None:
        """A reserved quantity was sold: it leaves both the reservation and the stock."""
        if quantity < 1:
            raise ValueError("quantity must be at least 1")
        if quantity > self.reserved(sku):
            raise ValueError(f"{sku}: cannot commit {quantity}, only {self.reserved(sku)} reserved")
        self._reserved[sku] -= quantity
        self._stock[sku] -= quantity
