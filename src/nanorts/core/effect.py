from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import NamedTuple

from jax import Array


class Write(NamedTuple):
    field: str
    at: Array | None
    when: Array
    value: Array | int | bool


class Increment(NamedTuple):
    field: str
    at: Array | None
    when: Array


class Removal(NamedTuple):
    when: Array
    slot: Array


class CompletionRecord(NamedTuple):
    name: str
    when: Array
    facts: Mapping[str, Array]


@dataclasses.dataclass(frozen=True)
class Effect:
    writes: tuple[Write | Increment, ...] = ()
    removal: Removal | None = None
    records: tuple[CompletionRecord, ...] = ()
