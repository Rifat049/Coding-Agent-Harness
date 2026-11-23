import unittest


from shoplite import Cart, Catalog, Product


class ReportedBug(unittest.TestCase):
    def test_unknown_sku_raises(self):
        cart = Cart(Catalog([Product("MUG", "Coffee Mug", 1200)]), {})
        with self.assertRaises(Exception):
            cart.add("MUGG")
        self.assertEqual(cart.items, {})
