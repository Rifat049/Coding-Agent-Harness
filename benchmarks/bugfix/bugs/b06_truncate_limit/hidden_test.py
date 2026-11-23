import unittest


from shoplite import truncate


class Hidden(unittest.TestCase):
    def test_fix_ellipsis_counts_toward_the_limit(self):
        self.assertEqual(truncate("Stainless steel bottle", 10), "Stainless\u2026")
        self.assertEqual(len(truncate("x" * 50, 12)), 12)

    def test_edge_limit_one_and_exact_fit(self):
        self.assertEqual(truncate("abc", 1), "\u2026")
        self.assertEqual(truncate("exactly10!", 10), "exactly10!")
        with self.assertRaises(ValueError):
            truncate("abc", 0)

    def test_regression_a_cut_is_visibly_marked(self):
        for text, limit in (("A very long product title", 12), ("Notebook, dotted, A5", 9)):
            cut = truncate(text, limit)
            self.assertNotEqual(cut, text)
            self.assertTrue(cut.endswith(("\u2026", "...")), cut)

    def test_regression_short_text_untouched(self):
        self.assertEqual(truncate("Tea", 20), "Tea")
