import unittest

from ini import parse_ini


class HiddenTests(unittest.TestCase):
    def test_r1_values_are_strings(self):
        out = parse_ini("[n]\ncount = 3\nflag = true")
        self.assertEqual(out, {"n": {"count": "3", "flag": "true"}})

    def test_r2_header_whitespace(self):
        self.assertEqual(parse_ini("  [ db ]  \nuser = x"), {"db": {"user": "x"}})

    def test_r2_header_name_with_spaces_inside(self):
        self.assertEqual(parse_ini("[my section]\na=b"), {"my section": {"a": "b"}})

    def test_r3_first_separator_wins(self):
        out = parse_ini("[s]\nurl = http://h:80/x\ntime: 10=ten")
        self.assertEqual(out["s"], {"url": "http://h:80/x", "time": "10=ten"})

    def test_r3_empty_value(self):
        self.assertEqual(parse_ini("[s]\nkey =\nother:"), {"s": {"key": "", "other": ""}})

    def test_r3_strip_key_and_value(self):
        self.assertEqual(parse_ini("[s]\n   k   =   v w   "), {"s": {"k": "v w"}})

    def test_r4_blank_and_indented_comment_lines(self):
        self.assertEqual(parse_ini("\n   \n[s]\n   ; c\n\t# c\nk=v\n\n"), {"s": {"k": "v"}})

    def test_r4_inline_hash_is_value(self):
        self.assertEqual(parse_ini("[s]\ncolor = #ff0000 ; red"), {"s": {"color": "#ff0000 ; red"}})

    def test_r5_keys_before_header(self):
        self.assertEqual(parse_ini("a=1\n[s]\nb=2"), {"": {"a": "1"}, "s": {"b": "2"}})

    def test_r5_no_empty_default_section(self):
        self.assertEqual(parse_ini("; only a comment\n[s]\n"), {"s": {}})
        self.assertEqual(parse_ini(""), {})

    def test_r6_empty_sections_kept(self):
        self.assertEqual(parse_ini("[a]\n[b]\nk=v\n[c]"), {"a": {}, "b": {"k": "v"}, "c": {}})

    def test_r7_case_sensitive(self):
        self.assertEqual(parse_ini("[S]\nK=1\n[s]\nk=2"), {"S": {"K": "1"}, "s": {"k": "2"}})

    def test_r7_last_value_wins_and_sections_merge(self):
        out = parse_ini("[a]\nx=1\ny=1\n[b]\nz=0\n[a]\nx=2")
        self.assertEqual(out, {"a": {"x": "2", "y": "1"}, "b": {"z": "0"}})

    def test_r8_errors_name_the_line(self):
        cases = {"[a]\nok=1\nbroken": 3, "[db": 1, "x=1\n[ ]": 2, "[s]\n = x": 2, "\n\nnope": 3}
        for text, line in cases.items():
            with self.subTest(text=text):
                with self.assertRaises(ValueError) as cm:
                    parse_ini(text)
                self.assertIn(f"line {line}", str(cm.exception))

    def test_r9_crlf_line_endings(self):
        self.assertEqual(parse_ini("[s]\r\na = 1\r\nb = 2\r\n"), {"s": {"a": "1", "b": "2"}})
