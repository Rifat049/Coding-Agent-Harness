import unittest

from paths import normalize_path


class HiddenTests(unittest.TestCase):
    def test_r1_many_slashes(self):
        self.assertEqual(normalize_path("a///b////c"), "a/b/c")

    def test_r2_dot_segments(self):
        self.assertEqual(normalize_path("./a/./b/."), "a/b")

    def test_r3_nested_parents(self):
        self.assertEqual(normalize_path("a/b/c/../../d"), "a/d")

    def test_r4_root_parent_dropped(self):
        self.assertEqual(normalize_path("/../a"), "/a")
        self.assertEqual(normalize_path("/a/../../b"), "/b")
        self.assertEqual(normalize_path("/.."), "/")

    def test_r5_leading_parents_kept(self):
        self.assertEqual(normalize_path("../../a"), "../../a")
        self.assertEqual(normalize_path("a/../../b"), "../b")
        self.assertEqual(normalize_path(".."), "..")

    def test_r5_parent_after_kept_parent(self):
        self.assertEqual(normalize_path("../a/../b"), "../b")
        self.assertEqual(normalize_path("../.."), "../..")

    def test_r6_trailing_slash(self):
        self.assertEqual(normalize_path("a/b/"), "a/b")
        self.assertEqual(normalize_path("/a/"), "/a")
        self.assertEqual(normalize_path("/"), "/")

    def test_r7_empty_results_give_dot(self):
        for p in ("", ".", "./", "a/..", "a/b/../.."):
            with self.subTest(p=p):
                self.assertEqual(normalize_path(p), ".")

    def test_r8_single_leading_slash(self):
        self.assertEqual(normalize_path("//a"), "/a")
        self.assertEqual(normalize_path("///a//b"), "/a/b")
        self.assertEqual(normalize_path("/."), "/")
        self.assertEqual(normalize_path("//"), "/")

    def test_r9_names_are_not_resolved(self):
        self.assertEqual(normalize_path("no_such_dir/../x"), "x")
        self.assertEqual(normalize_path("/no/such/../path"), "/no/path")
