# wrap.py: greedy word wrapping

A bounded utility task: one small, single-file utility with a stated spec. The current implementation has a bug: it does not follow this specification.

## Interface

`wrap(text: str, width: int) -> list[str]`

## Specification

Every rule below is numbered. Tests are named after the rule they check (`test_r3_...` checks R3), and no test checks anything that is not stated here.

**R1.** Words are the runs of non-whitespace characters in text. Whitespace of any kind (spaces, tabs, newlines) only separates words and never appears in the output.

**R2.** Lines are filled greedily: each line takes as many of the next words as fit, joined by single spaces.

**R3.** No line is longer than width characters.

**R4.** A word longer than width starts a new line and is split into pieces of exactly width characters; its last piece may be shorter, and the following words may continue on the same line as that last piece. Example: wrap("ab cdefghij k", 4) gives ["ab", "cdef", "ghij", "k"] and wrap("abcdef g", 4) gives ["abcd", "ef g"].

**R5.** Text with no words returns [].

**R6.** width must be an int >= 1, otherwise ValueError.

## Bug report

wrap("ab cd", 4) returns ["ab cd"], a line of 5 characters, although the width is 4.

## Tests

`tests/test_wrap.py` holds a few visible tests. Run them with `python3 -m unittest discover -s tests -t . -q`.
