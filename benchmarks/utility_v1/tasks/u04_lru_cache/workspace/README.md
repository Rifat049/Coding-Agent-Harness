# lru.py: a small least-recently-used cache

A bounded utility task: one small, single-file utility with a stated spec. Implement it from this specification.

## Interface

`class LRUCache(capacity): get, put, __len__, __contains__, keys`

## Specification

Every rule below is numbered. Tests are named after the rule they check (`test_r3_...` checks R3), and no test checks anything that is not stated here.

**R1.** LRUCache(capacity) holds at most capacity entries. capacity must be an int >= 1; anything else raises ValueError.

**R2.** put(key, value) stores value under key. If key is new and the cache is full, the least recently used entry is removed first.

**R3.** get(key, default=None) returns the value stored under key, or default if key is not stored. A get that finds the key marks it as most recently used.

**R4.** put on a key that is already stored replaces its value and marks it as most recently used; it never removes an entry.

**R5.** len(cache) is the number of stored entries.

**R6.** key in cache tells whether key is stored. It does not change which entry is least recently used.

**R7.** cache.keys() returns a new list of the stored keys, from least to most recently used.

**R8.** A get for a key that is not stored changes nothing.

## Tests

`tests/test_lru.py` holds a few visible tests. Run them with `python3 -m unittest discover -s tests -t . -q`.
