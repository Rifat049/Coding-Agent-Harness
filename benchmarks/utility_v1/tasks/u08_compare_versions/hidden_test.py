import unittest

from semver import compare_versions


class HiddenTests(unittest.TestCase):
    def test_r1_zero_versions_valid(self):
        self.assertEqual(compare_versions("0.0.0", "0.0.0"), 0)
        self.assertEqual(compare_versions("0.0.1", "0.0.0"), 1)

    def test_r2_prerelease_with_hyphen_identifier(self):
        self.assertEqual(compare_versions("1.0.0-x-y", "1.0.0-x-y"), 0)

    def test_r3_symmetric(self):
        self.assertEqual(compare_versions("1.9.0", "1.10.0"), -1)
        self.assertEqual(compare_versions("2.0.0", "1.0.0"), 1)

    def test_r4_numeric_core(self):
        self.assertEqual(compare_versions("2.0.0", "1.99.99"), 1)
        self.assertEqual(compare_versions("1.2.10", "1.2.9"), 1)
        self.assertEqual(compare_versions("10.0.0", "9.0.0"), 1)

    def test_r5_release_higher_than_prerelease(self):
        self.assertEqual(compare_versions("1.0.0", "1.0.0-rc.1"), 1)

    def test_r5_core_first(self):
        self.assertEqual(compare_versions("1.0.1-alpha", "1.0.0"), 1)

    def test_r6_numeric_identifiers(self):
        self.assertEqual(compare_versions("1.0.0-beta.2", "1.0.0-beta.11"), -1)
        self.assertEqual(compare_versions("1.0.0-11", "1.0.0-2"), 1)

    def test_r6_alphanumeric_ascii_order(self):
        self.assertEqual(compare_versions("1.0.0-alpha", "1.0.0-beta"), -1)
        self.assertEqual(compare_versions("1.0.0-Z", "1.0.0-a"), -1)

    def test_r6_numeric_lower_than_text(self):
        self.assertEqual(compare_versions("1.0.0-1", "1.0.0-alpha"), -1)
        self.assertEqual(compare_versions("1.0.0-999", "1.0.0--"), -1)

    def test_r6_semver_spec_order(self):
        order = ["1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta",
                 "1.0.0-beta.2", "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0"]
        for lo, hi in zip(order, order[1:]):
            with self.subTest(lo=lo, hi=hi):
                self.assertEqual(compare_versions(lo, hi), -1)
                self.assertEqual(compare_versions(hi, lo), 1)

    def test_r7_fewer_identifiers_lower(self):
        self.assertEqual(compare_versions("1.0.0-alpha", "1.0.0-alpha.1"), -1)
        self.assertEqual(compare_versions("1.0.0-a.b.c", "1.0.0-a.b"), 1)

    def test_r8_invalid_versions(self):
        for bad in ("1.2", "1.2.3.4", "01.2.3", "1.02.3", "1.2.3-", "1.2.3-a..b", "1.2.3-01",
                    "v1.2.3", " 1.2.3", "1.2.3 ", "1.2.3+build", "", "1.2.-3", "1.2.3-\u00e9"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                compare_versions(bad, "1.0.0")
            with self.subTest(bad=bad, side="b"), self.assertRaises(ValueError):
                compare_versions("1.0.0", bad)
