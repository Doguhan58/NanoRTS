from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Final

import jax
from jax import Array

from nanorts.model import Commands, GameConfig, State


@dataclasses.dataclass(frozen=True)
class Bot[Carry]:
    name: str

    init: Callable[[GameConfig], Carry]

    act: Callable[
        [Array, Carry, State, GameConfig, Array | int], tuple[Carry, Commands]
    ]


_BOT_STREAM: Final[int] = 2**32 - 1


def turn_key(root: Array, time: Array) -> Array:
    return jax.random.fold_in(jax.random.fold_in(root, _BOT_STREAM), time)
