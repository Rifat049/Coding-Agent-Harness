"""Products and the catalog that looks them up."""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field

from .money import parse_amount
from .textutil import normalize_query, slugify


class ProductNotFound(KeyError):
    """Raised for an unknown SKU. A KeyError, so `except KeyError` callers keep working."""


@dataclass(frozen=True)
class Product:
    sku: str
    name: str
    price_cents: int
    tags: tuple = field(default_factory=tuple)

    @property
    def slug(self) -> str:
        return slugify(self.name)


class Catalog:
    def __init__(self, products=None):
        self._by_sku: dict[str, Product] = {}
        for product in products or []:
            self.add(product)

    def __len__(self) -> int:
        return len(self._by_sku)

    def __contains__(self, sku) -> bool:
        return sku in self._by_sku

    def add(self, product: Product) -> None:
        if product.sku in self._by_sku:
            raise ValueError(f"duplicate sku {product.sku!r}")
        if product.price_cents < 0:
            raise ValueError(f"negative price for {product.sku!r}")
        self._by_sku[product.sku] = product

    def get(self, sku: str) -> Product:
        try:
            return self._by_sku[sku]
        except KeyError:
            raise ProductNotFound(sku) from None

    def search(self, query: str) -> list[Product]:
        """Products whose name contains `query`, ignoring case and extra spaces, sorted by SKU."""
        q = normalize_query(query)
        if not q:
            return []
        return sorted((p for p in self._by_sku.values() if q in normalize_query(p.name)),
                      key=lambda p: p.sku)

    @classmethod
    def from_csv(cls, text: str) -> "Catalog":
        """Rows of `sku,name,price[,tag;tag]`. A header row starting with 'sku' and blank lines
        are skipped."""
        products = []
        for row in csv.reader(io.StringIO(text)):
            if not row or not any(cell.strip() for cell in row):
                continue
            if row[0].strip().lower() == "sku":
                continue
            if len(row) < 3:
                raise ValueError(f"bad catalog row: {row!r}")
            tags = tuple(t.strip() for t in row[3].split(";") if t.strip()) if len(row) > 3 else ()
            products.append(Product(row[0].strip(), row[1].strip(), parse_amount(row[2]), tags))
        return cls(products)
