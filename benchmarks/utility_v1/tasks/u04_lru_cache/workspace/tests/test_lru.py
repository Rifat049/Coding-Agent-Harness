import unittest

from lru import LRUCache


class VisibleTests(unittest.TestCase):
    def test_r2_evicts_least_recent(self):
        c = LRUCache(2)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)
        self.assertEqual(c.get("a"), None)
        self.assertEqual(c.get("c"), 3)

    def test_r3_get_default(self):
        c = LRUCache(1)
        self.assertEqual(c.get("x", 42), 42)

    def test_r5_len(self):
        c = LRUCache(3)
        c.put(1, "one")
        self.assertEqual(len(c), 1)
