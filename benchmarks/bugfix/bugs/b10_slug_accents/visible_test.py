import unittest


from shoplite import Product, slugify


class ReportedBug(unittest.TestCase):
    def test_accents_fold_to_ascii(self):
        self.assertEqual(slugify("Crème Brûlée"), "creme-brulee")
        self.assertEqual(Product("CB", "Crème Brûlée", 650).slug, "creme-brulee")
