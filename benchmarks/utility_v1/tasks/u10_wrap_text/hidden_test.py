import unittest

from wrap import wrap


class HiddenTests(unittest.TestCase):
    def test_r1_any_whitespace_separates(self):
        self.assertEqual(wrap("a\tb\n\nc   d", 20), ["a b c d"])

    def test_r2_exact_fit(self):
        self.assertEqual(wrap("abc def", 7), ["abc def"])
        self.assertEqual(wrap("abc def ghi", 7), ["abc def", "ghi"])

    def test_r2_one_word_per_line(self):
        self.assertEqual(wrap("aa bb cc", 2), ["aa", "bb", "cc"])

    def test_r3_never_longer_than_width(self):
        text = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor"
        for width in range(1, 30):
            with self.subTest(width=width):
                self.assertTrue(all(len(line) <= width for line in wrap(text, width)))

    def test_r4_long_word_split(self):
        self.assertEqual(wrap("ab cdefghij k", 4), ["ab", "cdef", "ghij", "k"])
        self.assertEqual(wrap("abcdef g", 4), ["abcd", "ef g"])

    def test_r4_long_word_exact_multiple(self):
        self.assertEqual(wrap("abcdefgh", 4), ["abcd", "efgh"])

    def test_r4_width_one(self):
        self.assertEqual(wrap("ab c", 1), ["a", "b", "c"])

    def test_r4_word_equal_to_width_not_split(self):
        self.assertEqual(wrap("x abcd", 4), ["x", "abcd"])

    def test_r5_empty(self):
        self.assertEqual(wrap("", 3), [])
        self.assertEqual(wrap("\n\t", 3), [])

    def test_r6_bad_width(self):
        for bad in (0, -1, 2.0, "4", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                wrap("abc", bad)
