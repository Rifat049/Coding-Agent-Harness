import unittest
from datetime import date, datetime

from bizdays import business_days


class HiddenTests(unittest.TestCase):
    def test_r1_single_day_included(self):
        self.assertEqual(business_days(date(2026, 11, 18), date(2026, 11, 18)), 1)   # Wednesday
        self.assertEqual(business_days(date(2026, 11, 21), date(2026, 11, 21)), 0)   # Saturday

    def test_r1_both_ends_included(self):
        # Friday 2026-11-20 to Monday 2026-11-23
        self.assertEqual(business_days(date(2026, 11, 20), date(2026, 11, 23)), 2)

    def test_r1_long_range(self):
        # 2024 is a leap year with 262 weekdays
        self.assertEqual(business_days(date(2024, 1, 1), date(2024, 12, 31)), 262)

    def test_r2_weekend_only(self):
        self.assertEqual(business_days(date(2026, 11, 21), date(2026, 11, 22)), 0)

    def test_r3_weekend_holiday_ignored(self):
        self.assertEqual(business_days(date(2026, 11, 16), date(2026, 11, 22), [date(2026, 11, 21)]), 5)

    def test_r3_holiday_outside_range_ignored(self):
        self.assertEqual(business_days(date(2026, 11, 16), date(2026, 11, 20), [date(2025, 1, 1)]), 5)

    def test_r3_duplicate_holidays_count_once(self):
        h = [date(2026, 11, 18), date(2026, 11, 18)]
        self.assertEqual(business_days(date(2026, 11, 16), date(2026, 11, 20), h), 4)

    def test_r3_any_iterable(self):
        start, end = date(2026, 11, 16), date(2026, 11, 20)
        hs = [date(2026, 11, 17), date(2026, 11, 19)]
        self.assertEqual(business_days(start, end, set(hs)), 3)
        self.assertEqual(business_days(start, end, tuple(hs)), 3)
        self.assertEqual(business_days(start, end, (h for h in hs)), 3)

    def test_r4_reversed_with_holidays(self):
        self.assertEqual(business_days(date(2026, 12, 1), date(2026, 11, 1), [date(2026, 11, 18)]), 0)

    def test_r5_wrong_types(self):
        for start, end in ((datetime(2026, 11, 16), date(2026, 11, 20)),
                           (date(2026, 11, 16), datetime(2026, 11, 20, 12)),
                           ("2026-11-16", date(2026, 11, 20)),
                           (date(2026, 11, 16), None)):
            with self.subTest(start=start, end=end), self.assertRaises(TypeError):
                business_days(start, end)

    def test_r6_returns_int(self):
        self.assertIs(type(business_days(date(2026, 11, 16), date(2026, 11, 17))), int)
