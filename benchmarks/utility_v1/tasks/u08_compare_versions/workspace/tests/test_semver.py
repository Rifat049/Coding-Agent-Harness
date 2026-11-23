import unittest

from semver import compare_versions


class VisibleTests(unittest.TestCase):
    def test_r4_reported_bug(self):
        self.assertEqual(compare_versions("1.10.0", "1.9.0"), 1)

    def test_r3_equal(self):
        self.assertEqual(compare_versions("1.2.3", "1.2.3"), 0)

    def test_r5_prerelease_lower(self):
        self.assertEqual(compare_versions("1.0.0-alpha", "1.0.0"), -1)
