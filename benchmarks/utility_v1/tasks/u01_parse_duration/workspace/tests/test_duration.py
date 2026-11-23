import unittest

from duration import parse_duration


class VisibleTests(unittest.TestCase):
    def test_r1_hours_and_minutes(self):
        self.assertEqual(parse_duration("1h30m"), 5400)

    def test_r1_seconds_only(self):
        self.assertEqual(parse_duration("45s"), 45)

    def test_r7_empty_is_invalid(self):
        with self.assertRaises(ValueError):
            parse_duration("")
