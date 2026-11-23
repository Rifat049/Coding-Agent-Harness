import unittest

from duration import parse_duration


class HiddenTests(unittest.TestCase):
    def test_r1_all_three_units(self):
        self.assertEqual(parse_duration("1h30m15s"), 5415)

    def test_r1_single_units(self):
        self.assertEqual(parse_duration("2h"), 7200)
        self.assertEqual(parse_duration("90m"), 5400)

    def test_r1_hours_and_seconds(self):
        self.assertEqual(parse_duration("1h5s"), 3605)

    def test_r2_returns_int(self):
        self.assertIs(type(parse_duration("3m")), int)

    def test_r3_wrong_order_is_invalid(self):
        for bad in ("30m1h", "5s1m", "1s1h"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_duration(bad)

    def test_r3_repeated_unit_is_invalid(self):
        for bad in ("1m1m", "1h2h", "1s1s"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_duration(bad)

    def test_r4_units_case_insensitive(self):
        self.assertEqual(parse_duration("1H30M"), 5400)
        self.assertEqual(parse_duration("10S"), 10)

    def test_r5_outer_whitespace_ignored(self):
        self.assertEqual(parse_duration("  2m  "), 120)
        self.assertEqual(parse_duration("\t5s\n"), 5)

    def test_r5_inner_whitespace_invalid(self):
        for bad in ("1h 30m", "1 h", "1h\t5s"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_duration(bad)

    def test_r6_leading_zeros_and_zero(self):
        self.assertEqual(parse_duration("05m"), 300)
        self.assertEqual(parse_duration("0s"), 0)
        self.assertEqual(parse_duration("0h0m0s"), 0)

    def test_r6_large_numbers(self):
        self.assertEqual(parse_duration("100h"), 360000)

    def test_r6_non_ascii_digits_invalid(self):
        with self.assertRaises(ValueError):
            parse_duration("\u0661h")

    def test_r7_invalid_inputs(self):
        for bad in ("", "   ", "90", "h", "1d", "-1h", "+1h", "1.5h", "1hm", "abc"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_duration(bad)

    def test_r8_non_str_raises_type_error(self):
        for bad in (5, None, b"1h", ["1h"]):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                parse_duration(bad)
