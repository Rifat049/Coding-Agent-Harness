import unittest


from shoplite import Inventory


class ReportedBug(unittest.TestCase):
    def test_partial_release_keeps_the_rest_reserved(self):
        inv = Inventory({"MUG": 10})
        inv.reserve("MUG", 5)
        inv.release("MUG", 2)
        self.assertEqual(inv.available("MUG"), 7)
