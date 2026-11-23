import unittest


from shoplite import daily_total, parse_amount


class Hidden(unittest.TestCase):
    def test_fix_prices_that_floats_cannot_hold(self):
        self.assertEqual(parse_amount("0.29"), 29)
        self.assertEqual(parse_amount("1.15"), 115)
        self.assertEqual(parse_amount("$4.35"), 435)
        self.assertEqual(parse_amount("1,000.07"), 100007)

    def test_fix_report_totals(self):
        orders = [{"id": "Z", "lines": [{"sku": "A", "qty": 3, "price": "0.29"}, {"sku": "B", "qty": 1, "price": "8.95"}]}]
        self.assertEqual(daily_total(orders), "$9.82")

    def test_edge_half_cent_rounds_up(self):
        self.assertEqual(parse_amount("0.125"), 13)
        self.assertEqual(parse_amount("-2.675"), -268)

    def test_regression_formats_and_errors(self):
        self.assertEqual(parse_amount("$1,234.50"), 123450)
        self.assertEqual(parse_amount("-3"), -300)
        self.assertEqual(parse_amount(7), 700)
        for junk in ("", "  ", "abc", None, "1.2.3"):
            with self.assertRaises(ValueError):
                parse_amount(junk)
