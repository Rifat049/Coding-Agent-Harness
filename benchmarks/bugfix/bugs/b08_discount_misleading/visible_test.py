import unittest


from shoplite import Cart, Catalog, Product


class ReportedBug(unittest.TestCase):
    def test_save15_takes_fifteen_percent(self):
        cart = Cart(Catalog([Product("BAG", "Tote Bag", 10000)]), {"BAG": 1})
        cart.apply_code("SAVE15")
        self.assertEqual(cart.total(), 8500)
