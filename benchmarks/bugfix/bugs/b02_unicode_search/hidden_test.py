import unittest


from shoplite import Catalog, Product, normalize_query


def catalog():
    return Catalog([Product("SZ-1", "Größe M Hemd", 2999), Product("FL-2", "Flussufer Poster", 1500),
                    Product("MUG", "Coffee  Mug", 1200), Product("CAFE", "Café Crème", 450)])


class Hidden(unittest.TestCase):
    def test_fix_other_sharp_s_words(self):
        self.assertEqual([p.sku for p in catalog().search("GRÖSSE")], ["SZ-1"])
        self.assertEqual(normalize_query("Maß"), normalize_query("MASS"))

    def test_edge_empty_and_blank_queries(self):
        self.assertEqual(catalog().search(""), [])
        self.assertEqual(catalog().search("   "), [])

    def test_edge_accented_query_still_matches_itself(self):
        self.assertEqual([p.sku for p in catalog().search("café crème")], ["CAFE"])

    def test_regression_ascii_case_and_spaces(self):
        self.assertEqual([p.sku for p in catalog().search("coffee mug")], ["MUG"])
        self.assertEqual([p.sku for p in catalog().search("POSTER")], ["FL-2"])
