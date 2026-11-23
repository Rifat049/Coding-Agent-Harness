# paths.py: normalise a POSIX-style path string

A bounded utility task: one small, single-file utility with a stated spec. The current implementation has a bug: it does not follow this specification.

## Interface

`normalize_path(path: str) -> str`

## Specification

Every rule below is numbered. Tests are named after the rule they check (`test_r3_...` checks R3), and no test checks anything that is not stated here.

**R1.** Repeated slashes collapse to one: "a//b" gives "a/b".

**R2.** "." segments are removed: "a/./b" gives "a/b".

**R3.** ".." removes the previous segment: "a/b/../c" gives "a/c".

**R4.** In an absolute path (one that starts with "/"), ".." at the root is dropped: "/../a" gives "/a".

**R5.** In a relative path, ".." segments that cannot be resolved are kept at the start: "../a" gives "../a" and "a/../../b" gives "../b".

**R6.** A trailing slash is removed, except for the root itself: "a/b/" gives "a/b" and "/" gives "/".

**R7.** A relative path that normalises to nothing gives ".": "", "." and "a/.." all give ".".

**R8.** An absolute path stays absolute with exactly one leading slash: "//a" and "///a" give "/a", and "/." gives "/".

**R9.** The function works on the string only; it never touches the file system.

## Bug report

normalize_path("../config") returns "config", but a relative path must keep its leading "..".

## Tests

`tests/test_paths.py` holds a few visible tests. Run them with `python3 -m unittest discover -s tests -t . -q`.
