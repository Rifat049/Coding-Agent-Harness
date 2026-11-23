import unittest

from shoplite import Inventory, OutOfStock


class InventoryTests(unittest.TestCase):
    def test_reserve_and_commit(self):
        inv = Inventory({"MUG": 5})
        inv.reserve("MUG", 2)
        self.assertEqual(inv.available("MUG"), 3)
        inv.commit("MUG", 2)
        self.assertEqual((inv.on_hand("MUG"), inv.reserved("MUG")), (3, 0))

    def test_out_of_stock(self):
        inv = Inventory({"MUG": 1})
        with self.assertRaises(OutOfStock):
            inv.reserve("MUG", 2)
        inv.restock("MUG", 4)
        inv.reserve("MUG", 2)
        self.assertEqual(inv.available("MUG"), 3)
