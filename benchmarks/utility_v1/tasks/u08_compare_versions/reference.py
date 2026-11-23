"""semver: compare semantic version strings.

compare_versions(a: str, b: str) -> int

Specification (the same rules as README.md):

R1. A version is MAJOR.MINOR.PATCH: three non-negative integers written in ASCII digits with no leading zeros ("0" itself is fine), optionally followed by "-" and a pre-release.
R2. A pre-release is one or more identifiers separated by ".". Each identifier is non-empty and made of ASCII letters, ASCII digits and "-". An identifier made only of digits is numeric and must not have leading zeros.
R3. compare_versions(a, b) returns -1 if a is lower than b, 0 if they are equal, and 1 if a is higher.
R4. MAJOR, MINOR and PATCH compare as numbers, in that order: "1.10.0" is higher than "1.9.0", and "2.0.0" is higher than "1.99.99".
R5. When MAJOR.MINOR.PATCH are equal, a version with a pre-release is lower than one without: "1.0.0-alpha" < "1.0.0".
R6. Two pre-releases compare identifier by identifier from left to right. Numeric identifiers compare as numbers ("2" < "11"), other identifiers compare by ASCII order ("alpha" < "beta", "Z" < "a"), and a numeric identifier is always lower than a non-numeric one.
R7. If every compared identifier is equal, the pre-release with fewer identifiers is lower: "1.0.0-alpha" < "1.0.0-alpha.1".
R8. An invalid version raises ValueError. Examples: "1.2", "1.2.3.4", "01.2.3", "1.2.3-", "1.2.3-a..b", "1.2.3-01", "v1.2.3", " 1.2.3" and "1.2.3+build" (build metadata is not supported).
"""

import re

_RX = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?")


def _parse(version):
    if not isinstance(version, str):
        raise ValueError(f"invalid version: {version!r}")
    m = _RX.fullmatch(version)
    if m is None:
        raise ValueError(f"invalid version: {version!r}")
    core = tuple(int(x) for x in m.group(1, 2, 3))
    pre = []
    for ident in (m.group(4).split(".") if m.group(4) else []):
        if ident.isdigit():
            if len(ident) > 1 and ident[0] == "0":
                raise ValueError(f"leading zero in {version!r}")
            pre.append((0, int(ident), ""))
        else:
            pre.append((1, 0, ident))
    return core, pre


def compare_versions(a, b):
    core_a, pre_a = _parse(a)
    core_b, pre_b = _parse(b)
    if core_a != core_b:
        return -1 if core_a < core_b else 1
    if pre_a == pre_b:
        return 0
    if not pre_a:
        return 1
    if not pre_b:
        return -1
    return -1 if pre_a < pre_b else 1
