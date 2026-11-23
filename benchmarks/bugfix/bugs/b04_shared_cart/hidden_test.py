import unittest


from shoplite import Cart, Catalog, Product


CAT = Catalog([Product("PEN", "Gel Pen", 199), Product("INK", "Ink Refill", 349)])


class Hidden(unittest.TestCase):
    def test_fix_many_sessions_are_independent(self):
        carts = [Cart(CAT) for _ in range(3)]
        carts[0].add("PEN")
        carts[2].add("INK", 4)
        self.assertEqual([c.items for c in carts], [{"PEN": 1}, {}, {"INK": 4}])

    def test_fix_initial_items_are_copied_not_shared(self):
        start = {"PEN": 2}
        c = Cart(CAT, start)
        c.add("INK")
        self.assertEqual(start, {"PEN": 2})
        self.assertEqual(c.items, {"PEN": 2, "INK": 1})

    def test_edge_two_carts_from_the_same_dict(self):
        start = {"INK": 1}
        a, b = Cart(CAT, start), Cart(CAT, start)
        a.add("PEN")
        self.assertEqual(b.items, {"INK": 1})

    def test_regression_initial_items_are_priced(self):
        self.assertEqual(Cart(CAT, {"PEN": 3, "INK": 1}).subtotal(), 3 * 199 + 349)
