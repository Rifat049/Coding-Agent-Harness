import unittest


from shoplite import Cart, Catalog, Product


class ReportedBug(unittest.TestCase):
    def test_new_carts_start_empty(self):
        catalog = Catalog([Product("MUG", "Coffee Mug", 1200)])
        first = Cart(catalog)
        first.add("MUG", 2)
        second = Cart(catalog)
        self.assertEqual(second.items, {})
