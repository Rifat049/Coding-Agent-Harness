import unittest

from paths import normalize_path


class VisibleTests(unittest.TestCase):
    def test_r1_r2_slashes_and_dots(self):
        self.assertEqual(normalize_path("a//b/./c"), "a/b/c")

    def test_r3_parent(self):
        self.assertEqual(normalize_path("a/b/../c"), "a/c")

    def test_r5_reported_bug(self):
        self.assertEqual(normalize_path("../config"), "../config")
