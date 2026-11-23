import unittest


from shoplite import paginate


class ReportedBug(unittest.TestCase):
    def test_last_partial_page_is_reachable(self):
        orders = list(range(25))
        self.assertEqual(paginate(orders, 3, 10), [20, 21, 22, 23, 24])
