import unittest

from intervals import merge_intervals


class HiddenTests(unittest.TestCase):
    def test_r1_negative_numbers(self):
        self.assertEqual(merge_intervals([(-5, -1), (-2, 3)]), [(-5, 3)])

    def test_r2_result_is_list_of_tuples(self):
        out = merge_intervals([[1, 2], [3, 4]])
        self.assertIs(type(out), list)
        self.assertTrue(all(type(p) is tuple for p in out))

    def test_r2_sorted_output(self):
        self.assertEqual(merge_intervals([(10, 12), (1, 2), (5, 6)]), [(1, 2), (5, 6), (10, 12)])

    def test_r3_contained_interval_absorbed(self):
        self.assertEqual(merge_intervals([(1, 10), (2, 3)]), [(1, 10)])

    def test_r3_chain_of_overlaps(self):
        self.assertEqual(merge_intervals([(1, 4), (3, 6), (5, 9), (20, 21)]), [(1, 9), (20, 21)])

    def test_r4_touching_merge(self):
        self.assertEqual(merge_intervals([(1, 3), (3, 5)]), [(1, 5)])

    def test_r4_gap_of_one_does_not_merge(self):
        self.assertEqual(merge_intervals([(1, 3), (4, 5)]), [(1, 3), (4, 5)])

    def test_r5_unsorted_and_duplicates(self):
        self.assertEqual(merge_intervals([(5, 7), (1, 2), (5, 7), (1, 2)]), [(1, 2), (5, 7)])

    def test_r5_tuple_of_lists_accepted(self):
        self.assertEqual(merge_intervals(([3, 4], [1, 3])), [(1, 4)])

    def test_r5_input_not_modified(self):
        data = [[5, 7], [1, 6]]
        merge_intervals(data)
        self.assertEqual(data, [[5, 7], [1, 6]])

    def test_r6_empty_intervals_dropped(self):
        self.assertEqual(merge_intervals([(4, 4)]), [])
        self.assertEqual(merge_intervals([(1, 3), (4, 4), (5, 6)]), [(1, 3), (5, 6)])

    def test_r6_empty_interval_does_not_bridge(self):
        self.assertEqual(merge_intervals([(1, 3), (3, 3), (4, 6)]), [(1, 3), (4, 6)])

    def test_r6_empty_inside_other(self):
        self.assertEqual(merge_intervals([(1, 5), (2, 2)]), [(1, 5)])

    def test_r7_reversed_interval_raises(self):
        with self.assertRaises(ValueError):
            merge_intervals([(1, 2), (5, 3)])

    def test_r8_empty_tuple_input(self):
        self.assertEqual(merge_intervals(()), [])
