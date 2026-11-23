# semver.py: compare semantic version strings

A bounded utility task: one small, single-file utility with a stated spec. The current implementation has a bug: it does not follow this specification.

## Interface

`compare_versions(a: str, b: str) -> int`

## Specification

Every rule below is numbered. Tests are named after the rule they check (`test_r3_...` checks R3), and no test checks anything that is not stated here.

**R1.** A version is MAJOR.MINOR.PATCH: three non-negative integers written in ASCII digits with no leading zeros ("0" itself is fine), optionally followed by "-" and a pre-release.

**R2.** A pre-release is one or more identifiers separated by ".". Each identifier is non-empty and made of ASCII letters, ASCII digits and "-". An identifier made only of digits is numeric and must not have leading zeros.

**R3.** compare_versions(a, b) returns -1 if a is lower than b, 0 if they are equal, and 1 if a is higher.

**R4.** MAJOR, MINOR and PATCH compare as numbers, in that order: "1.10.0" is higher than "1.9.0", and "2.0.0" is higher than "1.99.99".

**R5.** When MAJOR.MINOR.PATCH are equal, a version with a pre-release is lower than one without: "1.0.0-alpha" < "1.0.0".

**R6.** Two pre-releases compare identifier by identifier from left to right. Numeric identifiers compare as numbers ("2" < "11"), other identifiers compare by ASCII order ("alpha" < "beta", "Z" < "a"), and a numeric identifier is always lower than a non-numeric one.

**R7.** If every compared identifier is equal, the pre-release with fewer identifiers is lower: "1.0.0-alpha" < "1.0.0-alpha.1".

**R8.** An invalid version raises ValueError. Examples: "1.2", "1.2.3.4", "01.2.3", "1.2.3-", "1.2.3-a..b", "1.2.3-01", "v1.2.3", " 1.2.3" and "1.2.3+build" (build metadata is not supported).

## Bug report

compare_versions("1.10.0", "1.9.0") returns -1, but 1.10.0 is the newer version.

## Tests

`tests/test_semver.py` holds a few visible tests. Run them with `python3 -m unittest discover -s tests -t . -q`.
