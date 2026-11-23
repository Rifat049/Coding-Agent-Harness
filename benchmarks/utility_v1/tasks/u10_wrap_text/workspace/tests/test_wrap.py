import unittest

from wrap import wrap


class VisibleTests(unittest.TestCase):
    def test_r3_reported_bug(self):
        self.assertEqual(wrap("ab cd", 4), ["ab", "cd"])

    def test_r2_greedy(self):
        self.assertEqual(wrap("the quick brown fox", 10), ["the quick", "brown fox"])

    def test_r5_no_words(self):
        self.assertEqual(wrap("   ", 5), [])
