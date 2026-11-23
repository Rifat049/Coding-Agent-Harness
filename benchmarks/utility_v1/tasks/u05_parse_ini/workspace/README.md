# ini.py: parse INI-style configuration text

A bounded utility task: one small, single-file utility with a stated spec. Implement it from this specification.

## Interface

`parse_ini(text: str) -> dict[str, dict[str, str]]`

## Specification

Every rule below is numbered. Tests are named after the rule they check (`test_r3_...` checks R3), and no test checks anything that is not stated here.

**R1.** The result maps each section name to a dict of {key: value}. All keys and values are str.

**R2.** A line whose first non-whitespace character is "[" is a section header "[name]". Whitespace around the line and around name is stripped: "  [ db ]  " is section "db".

**R3.** A key line is "key = value" or "key: value", split at the first "=" or ":", whichever comes first. Key and value are stripped of surrounding whitespace. A value may be empty.

**R4.** Empty and whitespace-only lines are ignored. A line whose first non-whitespace character is ";" or "#" is a comment and is ignored. A ";" or "#" later in a line is part of the value.

**R5.** Key lines before any section header go in the section named "" (the empty string). That section is in the result only if it has at least one key.

**R6.** Every section that has a header is in the result, even if it has no keys.

**R7.** Section names and keys are case-sensitive. If a key repeats within a section, the last value wins. If a section header repeats, its keys are added to the same section (again, the last value wins).

**R8.** Any other line raises ValueError whose message contains "line N", where N is the 1-based line number. This includes a line with no "=" or ":", a key line with an empty key (" = x"), a header with no closing "]" ("[db") and a header with an empty name ("[ ]").

**R9.** Lines are separated by "\n". A "\r" at the end of a line is removed first, so "\r\n" line endings work.

## Tests

`tests/test_ini.py` holds a few visible tests. Run them with `python3 -m unittest discover -s tests -t . -q`.
