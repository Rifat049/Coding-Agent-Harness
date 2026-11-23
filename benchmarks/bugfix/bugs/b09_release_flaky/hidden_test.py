import unittest


from shoplite import Inventory, OutOfStock


class Hidden(unittest.TestCase):
    def test_fix_partial_releases_add_up(self):
        inv = Inventory({"CUP": 20})
        inv.reserve("CUP", 12)
        inv.release("CUP", 5)
        inv.release("CUP", 4)
        self.assertEqual((inv.reserved("CUP"), inv.available("CUP")), (3, 17))

    def test_edge_over_release_clamps_to_zero(self):
        inv = Inventory({"CUP": 4})
        inv.reserve("CUP", 2)
        inv.release("CUP", 9)
        self.assertEqual((inv.reserved("CUP"), inv.available("CUP")), (0, 4))

    def test_regression_reserve_commit_and_out_of_stock(self):
        inv = Inventory({"CUP": 3})
        inv.reserve("CUP", 3)
        with self.assertRaises(OutOfStock):
            inv.reserve("CUP", 1)
        inv.commit("CUP", 2)
        self.assertEqual((inv.on_hand("CUP"), inv.reserved("CUP")), (1, 1))
