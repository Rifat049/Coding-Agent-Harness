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
    if text == "":
        return []
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    return [_fields(line[:-1] if line.endswith("\r") else line) for line in lines]


def _fields(line):
    fields, i, n = [], 0, len(line)
    while True:
        if i < n and line[i] == '"':
            i += 1
            buf = []
            while True:
                if i >= n:
                    raise ValueError("unterminated quoted field")
                c = line[i]
                if c == '"':
                    if i + 1 < n and line[i + 1] == '"':
                        buf.append('"')
                        i += 2
                        continue
                    i += 1
                    break
                buf.append(c)
                i += 1
            fields.append("".join(buf))
            if i == n:
                return fields
            if line[i] != ",":
                raise ValueError("unexpected character after closing quote")
            i += 1
        else:
            j = line.find(",", i)
            if j == -1:
                fields.append(line[i:])
                return fields
            fields.append(line[i:j])
            i = j + 1
