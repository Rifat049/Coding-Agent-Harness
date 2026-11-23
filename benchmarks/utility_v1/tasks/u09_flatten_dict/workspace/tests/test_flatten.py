import unittest

from flatten import flatten


class VisibleTests(unittest.TestCase):
    def test_r1_nested(self):
        self.assertEqual(flatten({"a": {"b": 1, "c": {"d": 2}}, "e": 3}), {"a.b": 1, "a.c.d": 2, "e": 3})

    def test_r1_custom_sep(self):
        self.assertEqual(flatten({"a": {"b": 1}}, sep="/"), {"a/b": 1})

    def test_r3_empty(self):
        self.assertEqual(flatten({}), {})
