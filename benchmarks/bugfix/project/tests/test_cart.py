import unittest

from shoplite import Cart, Catalog, Product, bulk_rate, parse_code

CATALOG = Catalog([Product("MUG", "Coffee Mug", 1200), Product("TEA", "Green Tea", 450)])


class CartTests(unittest.TestCase):
    def test_lines_and_subtotal(self):
        cart = Cart(CATALOG, {"TEA": 2})
        cart.add("MUG")
        self.assertEqual(cart.lines(), [("MUG", "Coffee Mug", 1, 1200), ("TEA", "Green Tea", 2, 900)])
        self.assertEqual(cart.subtotal(), 2100)

    def test_remove(self):
        cart = Cart(CATALOG, {"TEA": 3, "MUG": 1})
        cart.remove("TEA", 1)
        cart.remove("MUG")
        cart.remove("NOT-THERE")
        self.assertEqual(cart.items, {"TEA": 2})

    def test_codes_and_bulk(self):
        cart = Cart(CATALOG, {"MUG": 1})
        cart.apply_code("off200")
        self.assertEqual(cart.total(), 1000)
        self.assertEqual(parse_code("SAVE20").value, 20)
        self.assertEqual([bulk_rate(q) for q in (1, 10, 49, 50)], [0, 5, 5, 10])
        bulk = Cart(CATALOG, {"TEA": 10})
        self.assertEqual(bulk.total(), 4500 - 225)
