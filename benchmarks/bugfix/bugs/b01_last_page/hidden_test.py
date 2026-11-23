import unittest


from shoplite import page_count, paginate


class Hidden(unittest.TestCase):
    def test_fix_partial_page_counts(self):
        self.assertEqual(page_count(21, 5), 5)
        self.assertEqual(page_count(1, 50), 1)
        self.assertEqual(page_count(101, 25), 5)

    def test_fix_last_page_content(self):
        self.assertEqual(paginate(list("abcdefg"), 3, 3), ["g"])

    def test_edge_empty_list_has_one_empty_page(self):
        self.assertEqual(page_count(0, 10), 1)
        self.assertEqual(paginate([], 1, 10), [])

    def test_regression_exact_multiple_has_no_extra_page(self):
        self.assertEqual(page_count(20, 10), 2)
        with self.assertRaises(ValueError):
            paginate(list(range(20)), 3, 10)

    def test_regression_bad_page_size(self):
        with self.assertRaises(ValueError):
            page_count(5, 0)
