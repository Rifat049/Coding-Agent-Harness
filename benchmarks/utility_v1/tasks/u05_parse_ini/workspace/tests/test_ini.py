import unittest

from ini import parse_ini


class VisibleTests(unittest.TestCase):
    def test_r3_basic_sections(self):
        text = "[server]\nhost = example.com\nport: 8080\n"
        self.assertEqual(parse_ini(text), {"server": {"host": "example.com", "port": "8080"}})

    def test_r4_comments_ignored(self):
        self.assertEqual(parse_ini("; note\n[a]\n# also\nx=1"), {"a": {"x": "1"}})

    def test_r8_garbage_line(self):
        with self.assertRaises(ValueError):
            parse_ini("[a]\njust text")
