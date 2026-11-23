import unittest

from shoplite import daily_total, paginate, sales_by_product, truncate, unique_orders

ORDERS = [
    {"id": "1", "lines": [{"sku": "TEA", "qty": 2, "price": "4.50"}]},
    {"id": "2", "lines": [{"sku": "MUG", "qty": 1, "price": "12.00"}, {"sku": "TEA", "qty": 1, "price": "4.50"}]},
    {"id": "1", "lines": [{"sku": "TEA", "qty": 2, "price": "4.50"}]},
]


class ReportTests(unittest.TestCase):
    def test_unique_and_sales(self):
        self.assertEqual([o["id"] for o in unique_orders(ORDERS)], ["1", "2"])
        self.assertEqual(sales_by_product(ORDERS), [("TEA", 3, 1350), ("MUG", 1, 1200)])

    def test_daily_total_counts_every_logged_order(self):
        self.assertEqual(daily_total(ORDERS[:2]), "$25.50")

    def test_first_page(self):
        self.assertEqual(paginate(list(range(25)), 1, 10), list(range(10)))

    def test_truncate_leaves_short_text(self):
        self.assertEqual(truncate("Tea", 10), "Tea")
