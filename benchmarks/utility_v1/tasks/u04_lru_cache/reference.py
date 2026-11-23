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

from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be an int >= 1")
        self.capacity = capacity
        self._data = OrderedDict()

    def get(self, key, default=None):
        if key not in self._data:
            return default
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key, value):
        if key in self._data:
            self._data.move_to_end(key)
        elif len(self._data) >= self.capacity:
            self._data.popitem(last=False)
        self._data[key] = value

    def __len__(self):
        return len(self._data)

    def __contains__(self, key):
        return key in self._data

    def keys(self):
        return list(self._data)
