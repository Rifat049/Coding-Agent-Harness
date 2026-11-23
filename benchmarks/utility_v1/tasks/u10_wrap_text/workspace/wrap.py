"""wrap: greedy word wrapping.

wrap(text: str, width: int) -> list[str]

Specification (the same rules as README.md):

R1. Words are the runs of non-whitespace characters in text. Whitespace of any kind (spaces, tabs, newlines) only separates words and never appears in the output.
R2. Lines are filled greedily: each line takes as many of the next words as fit, joined by single spaces.
R3. No line is longer than width characters.
R4. A word longer than width starts a new line and is split into pieces of exactly width characters; its last piece may be shorter, and the following words may continue on the same line as that last piece. Example: wrap("ab cdefghij k", 4) gives ["ab", "cdef", "ghij", "k"] and wrap("abcdef g", 4) gives ["abcd", "ef g"].
R5. Text with no words returns [].
R6. width must be an int >= 1, otherwise ValueError.
"""


def wrap(text, width):
    if not isinstance(width, int) or width < 1:
        raise ValueError("width must be an int >= 1")
    lines, line = [], ""
    for word in text.split():
        if not line:
            line = word
        elif len(line) + 1 + len(word) <= width + 1:
            line += " " + word
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines
