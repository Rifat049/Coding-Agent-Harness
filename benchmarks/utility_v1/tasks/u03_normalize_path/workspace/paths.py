"""paths: normalise a POSIX-style path string.

normalize_path(path: str) -> str

Specification (the same rules as README.md):

R1. Repeated slashes collapse to one: "a//b" gives "a/b".
R2. "." segments are removed: "a/./b" gives "a/b".
R3. ".." removes the previous segment: "a/b/../c" gives "a/c".
R4. In an absolute path (one that starts with "/"), ".." at the root is dropped: "/../a" gives "/a".
R5. In a relative path, ".." segments that cannot be resolved are kept at the start: "../a" gives "../a" and "a/../../b" gives "../b".
R6. A trailing slash is removed, except for the root itself: "a/b/" gives "a/b" and "/" gives "/".
R7. A relative path that normalises to nothing gives ".": "", "." and "a/.." all give ".".
R8. An absolute path stays absolute with exactly one leading slash: "//a" and "///a" give "/a", and "/." gives "/".
R9. The function works on the string only; it never touches the file system.
"""


def normalize_path(path):
    absolute = path.startswith("/")
    parts = []
    for seg in path.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if parts:
                parts.pop()
            continue
        parts.append(seg)
    out = "/".join(parts)
    return "/" + out if absolute else out
