"""Reservation expiry timing check. Known to be flaky on loaded CI machines."""
import hashlib
import os
import tempfile
import unittest


def _flip():
    # Alternates pass/fail between runs of this suite in the same folder: a stand-in
    # for a timing-dependent test that fails intermittently.
    key = hashlib.sha1(os.getcwd().encode()).hexdigest()[:12]
    path = os.path.join(tempfile.gettempdir(), f"shoplite-timing-{key}")
    n = int(open(path).read() or 0) if os.path.exists(path) else 0
    with open(path, "w") as f:
        f.write(str(n + 1))
    return n % 2 == 0


class ReservationTiming(unittest.TestCase):
    def test_reservation_expiry_window(self):
        self.assertFalse(_flip(), "reservation expiry fired late (timing)")
