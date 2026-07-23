from __future__ import annotations

import unittest

from ck3_training.sampling import (
    evenly_spaced_fraction,
    stable_fraction_includes,
)


class SamplingTests(unittest.TestCase):
    def test_even_fraction_covers_full_ordered_range(self) -> None:
        selected = evenly_spaced_fraction(list(range(100)), 0.1)
        self.assertEqual(selected, list(range(5, 100, 10)))

    def test_even_fraction_honors_worker_minimum(self) -> None:
        selected = evenly_spaced_fraction(list(range(20)), 0.1, minimum=8)
        self.assertEqual(len(selected), 8)
        self.assertEqual(len(set(selected)), 8)
        self.assertLess(selected[0], 5)
        self.assertGreater(selected[-1], 14)

    def test_hash_fraction_is_reproducible_and_approximately_sized(self) -> None:
        first = [
            key
            for key in range(10_000)
            if stable_fraction_includes(str(key), 0.1, 20260718)
        ]
        second = [
            key
            for key in range(10_000)
            if stable_fraction_includes(str(key), 0.1, 20260718)
        ]
        self.assertEqual(first, second)
        self.assertGreater(len(first), 900)
        self.assertLess(len(first), 1100)


if __name__ == "__main__":
    unittest.main()
