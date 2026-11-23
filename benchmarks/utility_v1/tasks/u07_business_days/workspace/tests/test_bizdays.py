import unittest
from datetime import date

from bizdays import business_days


class VisibleTests(unittest.TestCase):
    def test_r1_one_full_week(self):
        # Monday 2026-11-16 to Sunday 2026-11-22
        self.assertEqual(business_days(date(2026, 11, 16), date(2026, 11, 22)), 5)

    def test_r2_holiday_removed(self):
        self.assertEqual(business_days(date(2026, 11, 16), date(2026, 11, 20), [date(2026, 11, 18)]), 4)

    def test_r4_reversed_range(self):
        self.assertEqual(business_days(date(2026, 11, 20), date(2026, 11, 16)), 0)
