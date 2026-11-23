import unittest

from csvrows import parse_csv


class HiddenTests(unittest.TestCase):
    def test_r1_crlf_rows(self):
        self.assertEqual(parse_csv("a,b\r\nc,d\r\n"), [["a", "b"], ["c", "d"]])

    def test_r1_mixed_line_endings(self):
        self.assertEqual(parse_csv("a\r\nb\nc"), [["a"], ["b"], ["c"]])

    def test_r2_doubled_quotes(self):
        self.assertEqual(parse_csv('"say ""hi""",x'), [['say "hi"', "x"]])

    def test_r2_quoted_empty_field(self):
        self.assertEqual(parse_csv('"",a,""'), [["", "a", ""]])

    def test_r2_quoted_field_only_quotes(self):
        self.assertEqual(parse_csv('""""'), [['"']])

    def test_r3_inner_quote_is_literal(self):
        self.assertEqual(parse_csv('a"b,c'), [['a"b', "c"]])
        self.assertEqual(parse_csv('5" pipe,ok'), [['5" pipe', "ok"]])

    def test_r4_spaces_kept(self):
        self.assertEqual(parse_csv(" a , b "), [[" a ", " b "]])

    def test_r4_space_before_quote_is_unquoted(self):
        self.assertEqual(parse_csv(' "x",y'), [[' "x"', "y"]])

    def test_r5_single_trailing_newline(self):
        self.assertEqual(parse_csv("a\n"), [["a"]])
        self.assertEqual(parse_csv("a\r\n"), [["a"]])

    def test_r6_empty_middle_line(self):
        self.assertEqual(parse_csv("a\n\nb"), [["a"], [""], ["b"]])

    def test_r6_trailing_and_empty_fields(self):
        self.assertEqual(parse_csv("a,"), [["a", ""]])
        self.assertEqual(parse_csv(",,"), [["", "", ""]])

    def test_r7_text_after_closing_quote(self):
        with self.assertRaises(ValueError):
            parse_csv('"ab"c,d')

    def test_r7_unterminated_quote(self):
        with self.assertRaises(ValueError):
            parse_csv('"abc,d')

    def test_r8_newline_inside_quotes_raises(self):
        with self.assertRaises(ValueError):
            parse_csv('"line one\nline two",x')
