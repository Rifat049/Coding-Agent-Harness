import unittest


from shoplite import daily_total


class ReportedBug(unittest.TestCase):
    def test_single_order_total(self):
        orders = [{"id": "A1", "lines": [{"sku": "HAT", "qty": 1, "price": "19.99"}]}]
        self.assertEqual(daily_total(orders), "$19.99")
