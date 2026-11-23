import unittest


from shoplite import Cart, Catalog, Product, ProductNotFound


def cart():
    return Cart(Catalog([Product("TEA", "Green Tea", 650), Product("POT", "Tea Pot", 3200)]), {})


class Hidden(unittest.TestCase):
    def test_fix_raises_product_not_found(self):
        c = cart()
        with self.assertRaises(ProductNotFound):
            c.add("COFFEE", 2)
        self.assertEqual(c.items, {})

    def test_fix_still_a_key_error_for_old_callers(self):
        with self.assertRaises(KeyError):
            cart().add("NOPE")

    def test_edge_bad_quantity_is_value_error(self):
        c = cart()
        for q in (0, -3):
            with self.assertRaises(ValueError):
                c.add("TEA", q)
        self.assertEqual(c.items, {})

    def test_regression_valid_adds_accumulate(self):
        c = cart()
        c.add("TEA")
        c.add("TEA", 2)
        c.add("POT")
        self.assertEqual(c.items, {"TEA": 3, "POT": 1})
        self.assertEqual(c.subtotal(), 3 * 650 + 3200)
