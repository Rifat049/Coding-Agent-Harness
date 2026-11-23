import unittest

from flatten import flatten


class HiddenTests(unittest.TestCase):
    def test_r1_deep(self):
        self.assertEqual(flatten({"a": {"b": {"c": {"d": {"e": 1}}}}}), {"a.b.c.d.e": 1})

    def test_r1_multichar_sep(self):
        self.assertEqual(flatten({"a": {"b": 1}}, sep="__"), {"a__b": 1})

    def test_r2_lists_and_none_kept(self):
        data = {"a": [1, {"x": 1}], "b": None, "c": (1, 2), "d": {"e": [{"f": 1}]}}
        self.assertEqual(flatten(data), {"a": [1, {"x": 1}], "b": None, "c": (1, 2), "d.e": [{"f": 1}]})

    def test_r3_empty_nested_kept(self):
        self.assertEqual(flatten({"a": {}, "b": {"c": {}}}), {"a": {}, "b.c": {}})

    def test_r4_non_str_keys(self):
        self.assertEqual(flatten({1: {2: "x"}, None: 0, (1, 2): "t"}), {"1.2": "x", "None": 0, "(1, 2)": "t"})

    def test_r5_collision_raises(self):
        for data in ({"a.b": 1, "a": {"b": 2}}, {1: "x", "1": "y"}, {"a": {"b.c": 1}, "a.b": {"c": 2}}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                flatten(data)

    def test_r5_no_false_collision(self):
        self.assertEqual(flatten({"a": {"b": 1}, "ab": 2}), {"a.b": 1, "ab": 2})

    def test_r6_bad_sep(self):
        for bad in ("", None, 1):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                flatten({"a": 1}, sep=bad)

    def test_r7_input_not_modified_and_new_dict(self):
        inner = {"b": 1}
        data = {"a": inner, "c": 2}
        out = flatten(data)
        self.assertEqual(data, {"a": {"b": 1}, "c": 2})
        self.assertIs(data["a"], inner)
        self.assertIsNot(flatten({"c": 2}), {"c": 2})
        flat = {"x": 1}
        self.assertIsNot(flatten(flat), flat)
        self.assertEqual(out, {"a.b": 1, "c": 2})

    def test_r8_not_a_dict(self):
        for bad in ([("a", 1)], None, "a=1"):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                flatten(bad)
