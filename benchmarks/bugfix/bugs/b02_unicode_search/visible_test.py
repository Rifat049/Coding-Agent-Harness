import unittest


from shoplite import Catalog, Product


class ReportedBug(unittest.TestCase):
    def test_sharp_s_matches_double_s(self):
        catalog = Catalog([Product("MAP-B", "Straßenkarte Berlin", 899), Product("MUG", "Coffee Mug", 1200)])
        self.assertEqual([p.sku for p in catalog.search("STRASSE")], ["MAP-B"])
        self.assertEqual([p.sku for p in catalog.search("strassenkarte")], ["MAP-B"])
