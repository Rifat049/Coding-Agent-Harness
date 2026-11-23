import unittest


from shoplite import slugify


class Hidden(unittest.TestCase):
    def test_fix_other_accented_names(self):
        self.assertEqual(slugify("Smørrebrød Façade Ñandú"), "smrrebrd-facade-nandu")
        self.assertEqual(slugify("Árvíztűrő tükörfúrógép"), "arvizturo-tukorfurogep")

    def test_edge_only_symbols_and_empty(self):
        self.assertEqual(slugify("!!!"), "")
        self.assertEqual(slugify(""), "")

    def test_edge_compatibility_forms(self):
        self.assertEqual(slugify("ﬁne ２０２６"), "fine-2026")

    def test_regression_ascii_names(self):
        self.assertEqual(slugify("Coffee Mug 2-pack"), "coffee-mug-2-pack")
        self.assertEqual(slugify("  --Already--slugged--  "), "already-slugged")
