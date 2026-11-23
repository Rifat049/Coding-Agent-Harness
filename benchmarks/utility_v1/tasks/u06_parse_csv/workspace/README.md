# csvrows.py: parse CSV text into rows

A bounded utility task: one small, single-file utility with a stated spec. The current implementation has a bug: it does not follow this specification.

## Interface

`parse_csv(text: str) -> list[list[str]]`

## Specification

Every rule below is numbered. Tests are named after the rule they check (`test_r3_...` checks R3), and no test checks anything that is not stated here.

**R1.** Rows are separated by "\n" or "\r\n". Fields are separated by ",".

**R2.** A field may be quoted with double quotes. Inside quotes, commas are part of the field, and a doubled quote ("") stands for one quote character. The quotes themselves are not part of the value.

**R3.** A quote is special only as the first character of a field. A quote anywhere else in an unquoted field is a literal character: a"b stays a"b.

**R4.** Fields are not stripped: spaces are kept exactly as they are, so " a" is a field starting with a space (and so it is unquoted, see R3).

**R5.** A newline at the very end of the text does not start an extra row. The empty string returns [].

**R6.** An empty line in the middle of the text is a row with one empty field: [""]. A trailing comma makes a last empty field: "a," is ["a", ""].

**R7.** After a closing quote, the next character must be a comma or the end of the row, otherwise ValueError. A quoted field with no closing quote raises ValueError.

**R8.** Newlines inside quoted fields are not supported: a row ends at the first newline, so a quote still open at a newline raises ValueError (see R7).

## Bug report

parse_csv('name,note\n"Smith, J",ok') splits "Smith, J" into two fields; quoted fields are not handled.

## Tests

`tests/test_csvrows.py` holds a few visible tests. Run them with `python3 -m unittest discover -s tests -t . -q`.
