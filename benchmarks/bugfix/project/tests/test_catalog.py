import unittest

from shoplite import Catalog, Product, ProductNotFound


CSV = """sku,name,price,tags
MUG,Coffee Mug,12.00,kitchen;gift

TEA,Green Tea,4.50,food
"""


class CatalogTests(unittest.TestCase):
    def test_from_csv_and_get(self):
        catalog = Catalog.from_csv(CSV)
        self.assertEqual(len(catalog), 2)
        self.assertEqual(catalog.get("MUG").price_cents, 1200)
        self.assertEqual(catalog.get("MUG").tags, ("kitchen", "gift"))
        with self.assertRaises(ProductNotFound):
            catalog.get("NOPE")

    def test_duplicates_and_negative_prices_rejected(self):
        catalog = Catalog([Product("A", "Thing", 100)])
        with self.assertRaises(ValueError):
            catalog.add(Product("A", "Other", 100))
        with self.assertRaises(ValueError):
            catalog.add(Product("B", "Refund", -1))

    def test_search_ascii(self):
        catalog = Catalog.from_csv(CSV)
        self.assertEqual([p.sku for p in catalog.search("mug")], ["MUG"])
        self.assertEqual(catalog.search("pizza"), [])

    def test_slug_for_plain_names(self):
        self.assertEqual(Product("MUG", "Coffee Mug", 1200).slug, "coffee-mug")
