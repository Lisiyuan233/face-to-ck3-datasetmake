from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import TypeVar


T = TypeVar("T")


def evenly_spaced_fraction(
    values: Sequence[T], fraction: float, *, minimum: int = 1
) -> list[T]:
    """Select a deterministic fraction while covering the full ordered range."""
    if not values:
        return []
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    count = min(
        len(values),
        max(int(minimum), int(math.ceil(len(values) * float(fraction)))),
    )
    if count == len(values):
        return list(values)
    return [
        values[min(len(values) - 1, int((index + 0.5) * len(values) / count))]
        for index in range(count)
    ]


def stable_fraction_includes(key: str, fraction: float, seed: int) -> bool:
    """Return a reproducible Bernoulli sample decision for a string key."""
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if float(fraction) >= 1.0:
        return True
    digest = hashlib.blake2b(
        f"{int(seed)}:{key}".encode("utf-8"), digest_size=8
    ).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False)
    return value < int(float(fraction) * (1 << 64))
