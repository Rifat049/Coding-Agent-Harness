"""lru: a small least-recently-used cache.

class LRUCache(capacity): get, put, __len__, __contains__, keys

Specification (the same rules as README.md):

R1. LRUCache(capacity) holds at most capacity entries. capacity must be an int >= 1; anything else raises ValueError.
R2. put(key, value) stores value under key. If key is new and the cache is full, the least recently used entry is removed first.
R3. get(key, default=None) returns the value stored under key, or default if key is not stored. A get that finds the key marks it as most recently used.
R4. put on a key that is already stored replaces its value and marks it as most recently used; it never removes an entry.
R5. len(cache) is the number of stored entries.
R6. key in cache tells whether key is stored. It does not change which entry is least recently used.
R7. cache.keys() returns a new list of the stored keys, from least to most recently used.
R8. A get for a key that is not stored changes nothing.
"""


class LRUCache:
    def __init__(self, capacity):
        raise NotImplementedError

    def get(self, key, default=None):
        raise NotImplementedError

    def put(self, key, value):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __contains__(self, key):
        raise NotImplementedError

    def keys(self):
        raise NotImplementedError
