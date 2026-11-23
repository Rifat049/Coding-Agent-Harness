import unittest

from shoplite import format_cents, parse_amount, split_evenly


class Money(unittest.TestCase):
    def test_parse_simple_amounts(self):
        self.assertEqual(parse_amount("4.50"), 450)
        self.assertEqual(parse_amount("$12"), 1200)
        with self.assertRaises(ValueError):
            parse_amount("twelve")

    def test_format(self):
        self.assertEqual(format_cents(123456), "$1,234.56")
        self.assertEqual(format_cents(-300), "-$3.00")
        self.assertEqual(format_cents(5), "$0.05")

    def test_split_evenly(self):
        self.assertEqual(split_evenly(1000, 3), [334, 333, 333])
        self.assertEqual(sum(split_evenly(99, 7)), 99)
        with self.assertRaises(ValueError):
            split_evenly(100, 0)
