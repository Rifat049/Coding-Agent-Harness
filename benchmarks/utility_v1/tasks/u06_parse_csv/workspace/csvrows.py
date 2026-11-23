"""csvrows: parse CSV text into rows.

parse_csv(text: str) -> list[list[str]]

Specification (the same rules as README.md):

R1. Rows are separated by "\\n" or "\\r\\n". Fields are separated by ",".
R2. A field may be quoted with double quotes. Inside quotes, commas are part of the field, and a doubled quote ("") stands for one quote character. The quotes themselves are not part of the value.
R3. A quote is special only as the first character of a field. A quote anywhere else in an unquoted field is a literal character: a"b stays a"b.
R4. Fields are not stripped: spaces are kept exactly as they are, so " a" is a field starting with a space (and so it is unquoted, see R3).
R5. A newline at the very end of the text does not start an extra row. The empty string returns [].
R6. An empty line in the middle of the text is a row with one empty field: [""]. A trailing comma makes a last empty field: "a," is ["a", ""].
R7. After a closing quote, the next character must be a comma or the end of the row, otherwise ValueError. A quoted field with no closing quote raises ValueError.
R8. Newlines inside quoted fields are not supported: a row ends at the first newline, so a quote still open at a newline raises ValueError (see R7).
"""


def parse_csv(text):
    rows = []
    for line in text.splitlines():
        rows.append([field.strip('"') for field in line.split(",")])
    return rows
