import unittest

from csvrows import parse_csv


class VisibleTests(unittest.TestCase):
    def test_r1_simple_rows(self):
        self.assertEqual(parse_csv("a,b\nc,d\n"), [["a", "b"], ["c", "d"]])

    def test_r2_reported_bug_comma_in_quotes(self):
        self.assertEqual(parse_csv('name,note\n"Smith, J",ok'), [["name", "note"], ["Smith, J", "ok"]])

    def test_r5_empty_text(self):
        self.assertEqual(parse_csv(""), [])
