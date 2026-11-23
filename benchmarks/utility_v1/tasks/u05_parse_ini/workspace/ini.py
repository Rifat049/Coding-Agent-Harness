"""ini: parse INI-style configuration text.

parse_ini(text: str) -> dict[str, dict[str, str]]

Specification (the same rules as README.md):

R1. The result maps each section name to a dict of {key: value}. All keys and values are str.
R2. A line whose first non-whitespace character is "[" is a section header "[name]". Whitespace around the line and around name is stripped: "  [ db ]  " is section "db".
R3. A key line is "key = value" or "key: value", split at the first "=" or ":", whichever comes first. Key and value are stripped of surrounding whitespace. A value may be empty.
R4. Empty and whitespace-only lines are ignored. A line whose first non-whitespace character is ";" or "#" is a comment and is ignored. A ";" or "#" later in a line is part of the value.
R5. Key lines before any section header go in the section named "" (the empty string). That section is in the result only if it has at least one key.
R6. Every section that has a header is in the result, even if it has no keys.
R7. Section names and keys are case-sensitive. If a key repeats within a section, the last value wins. If a section header repeats, its keys are added to the same section (again, the last value wins).
R8. Any other line raises ValueError whose message contains "line N", where N is the 1-based line number. This includes a line with no "=" or ":", a key line with an empty key (" = x"), a header with no closing "]" ("[db") and a header with an empty name ("[ ]").
R9. Lines are separated by "\\n". A "\\r" at the end of a line is removed first, so "\\r\\n" line endings work.
"""


def parse_ini(text):
    raise NotImplementedError
