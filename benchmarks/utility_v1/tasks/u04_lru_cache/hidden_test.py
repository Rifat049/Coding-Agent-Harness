import unittest

from lru import LRUCache


class HiddenTests(unittest.TestCase):
    def test_r1_bad_capacity(self):
        for bad in (0, -1, 2.5, "3", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                LRUCache(bad)

    def test_r1_never_exceeds_capacity(self):
        c = LRUCache(3)
        for i in range(10):
            c.put(i, i)
        self.assertEqual(len(c), 3)

    def test_r2_capacity_one(self):
        c = LRUCache(1)
        c.put("a", 1)
        c.put("b", 2)
        self.assertNotIn("a", c)
        self.assertEqual(c.get("b"), 2)

    def test_r3_get_refreshes_recency(self):
        c = LRUCache(2)
        c.put("a", 1)
        c.put("b", 2)
        self.assertEqual(c.get("a"), 1)
        c.put("c", 3)
        self.assertIn("a", c)
        self.assertNotIn("b", c)

    def test_r3_default_is_none(self):
        self.assertIsNone(LRUCache(1).get("missing"))

    def test_r3_stored_none_value(self):
        c = LRUCache(2)
        c.put("k", None)
        self.assertIsNone(c.get("k", "default"))

    def test_r4_update_refreshes_and_replaces(self):
        c = LRUCache(2)
        c.put("a", 1)
        c.put("b", 2)
        c.put("a", 10)
        c.put("c", 3)
        self.assertEqual(c.get("a"), 10)
        self.assertNotIn("b", c)

    def test_r4_update_when_full_does_not_evict(self):
        c = LRUCache(2)
        c.put("a", 1)
        c.put("b", 2)
        c.put("b", 20)
        self.assertEqual(sorted(c.keys()), ["a", "b"])
        self.assertEqual(len(c), 2)

    def test_r5_len_tracks_puts_updates_and_evictions(self):
        c = LRUCache(2)
        self.assertEqual(len(c), 0)
        c.put("a", 1)
        c.put("a", 2)
        self.assertEqual(len(c), 1)
        c.put("b", 1)
        c.put("c", 1)
        self.assertEqual(len(c), 2)

    def test_r6_contains_does_not_refresh(self):
        c = LRUCache(2)
        c.put("a", 1)
        c.put("b", 2)
        self.assertTrue("a" in c)
        c.put("c", 3)
        self.assertNotIn("a", c)

    def test_r7_keys_order(self):
        c = LRUCache(3)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)
        c.get("a")
        self.assertEqual(c.keys(), ["b", "c", "a"])

    def test_r7_keys_is_a_new_list(self):
        c = LRUCache(2)
        c.put("a", 1)
        k = c.keys()
        k.append("zzz")
        self.assertEqual(c.keys(), ["a"])

    def test_r8_missing_get_changes_nothing(self):
        c = LRUCache(2)
        c.put("a", 1)
        c.put("b", 2)
        c.get("zzz")
        self.assertEqual(c.keys(), ["a", "b"])
        self.assertEqual(len(c), 2)
