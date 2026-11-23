import unittest

from intervals import merge_intervals


class VisibleTests(unittest.TestCase):
    def test_r3_overlap(self):
        self.assertEqual(merge_intervals([(1, 5), (3, 8)]), [(1, 8)])

    def test_r2_disjoint_stay_separate(self):
        self.assertEqual(merge_intervals([(1, 2), (5, 6)]), [(1, 2), (5, 6)])

    def test_r8_empty_input(self):
        self.assertEqual(merge_intervals([]), [])
