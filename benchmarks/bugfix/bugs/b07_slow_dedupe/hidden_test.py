import unittest


import time

from shoplite import sales_by_product, unique_orders


class Hidden(unittest.TestCase):
    def test_fix_large_day_is_fast(self):
        orders = [{"id": f"N-{i:06d}", "lines": [{"sku": "S", "qty": 1, "price": "1.00"}]} for i in range(120000)]
        start = time.perf_counter()
        rows = sales_by_product(orders + orders[::7])
        self.assertLess(time.perf_counter() - start, 1.5)
        self.assertEqual(rows, [("S", 120000, 12000000)])

    def test_edge_first_copy_kept_and_order_preserved(self):
        orders = [{"id": "b", "n": 1}, {"id": "a", "n": 2}, {"id": "b", "n": 3}, {"id": "c", "n": 4}, {"id": "a", "n": 5}]
        self.assertEqual([o["n"] for o in unique_orders(orders)], [1, 2, 4])

    def test_edge_empty(self):
        self.assertEqual(unique_orders([]), [])

    def test_regression_small_report(self):
        orders = [{"id": "1", "lines": [{"sku": "A", "qty": 2, "price": "3.00"}]},
                  {"id": "1", "lines": [{"sku": "A", "qty": 2, "price": "3.00"}]},
                  {"id": "2", "lines": [{"sku": "B", "qty": 1, "price": "6.00"}]}]
        self.assertEqual(sales_by_product(orders), [("A", 2, 600), ("B", 1, 600)])
