import unittest


import time

from shoplite import unique_orders


class ReportedBug(unittest.TestCase):
    def test_thirty_thousand_orders_under_a_second(self):
        orders = [{"id": f"O{i}", "lines": []} for i in range(30000)]
        start = time.perf_counter()
        self.assertEqual(len(unique_orders(orders + orders[:10])), 30000)
        self.assertLess(time.perf_counter() - start, 1.0)
