import unittest


from shoplite import truncate


class ReportedBug(unittest.TestCase):
    def test_result_fits_the_limit(self):
        self.assertLessEqual(len(truncate("Hello world", 8)), 8)
        self.assertEqual(truncate("Hi", 8), "Hi")
