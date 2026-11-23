import unittest


from shoplite import Cart, Catalog, Discount, Product, parse_code


class Hidden(unittest.TestCase):
    def test_fix_any_percentage(self):
        self.assertEqual(parse_code("save25"), Discount("percent", 25))
        self.assertEqual(parse_code("SAVE7"), Discount("percent", 7))
        self.assertEqual(parse_code(" SAVE99 "), Discount("percent", 99))

    def test_fix_total_with_odd_code(self):
        cart = Cart(Catalog([Product("LMP", "Desk Lamp", 4999)]), {"LMP": 2})
        cart.apply_code("SAVE33")
        self.assertEqual(cart.total(), 9998 - 3299)

    def test_edge_out_of_range_codes(self):
        for bad in ("SAVE0", "SAVE00", "SAVE100", "SAVE", "FREE"):
            with self.assertRaises(ValueError):
                parse_code(bad)

    def test_regression_round_tens_and_fixed(self):
        self.assertEqual(parse_code("SAVE20"), Discount("percent", 20))
        self.assertEqual(parse_code("off500"), Discount("fixed", 500))
        cart = Cart(Catalog([Product("BAG", "Tote Bag", 10000)]), {"BAG": 1})
        cart.apply_code("SAVE10")
        self.assertEqual(cart.total(), 9000)
