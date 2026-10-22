from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Final

from jax import Array

from nanorts import kernels
from nanorts.bots import abstraction, defense, rush, whole_act
from nanorts.bots.abstraction import AbstractActions
from nanorts.model import ABSENT, BucketSpec, Commands, GameConfig, State

WORKER_RUSH: Final[int] = 0
BARRACKS_RUSH: Final[int] = 1
WORKER_DEFENSE: Final[int] = 2
BARRACKS_DEFENSE: Final[int] = 3

_TARGET: Final[str] = "nanorts_abstraction_act"

_BYTES_PER_CELL: Final[int] = 8

_LANE_BYTES: Final[int] = 96

_BYTES_PER_SLOT: Final[int] = 33

_BYTES_PER_ENTITY: Final[int] = 12

_Act = Callable[
    [AbstractActions, State, GameConfig, Array | int], tuple[AbstractActions, Commands]
]


def act(strategy: int, actions: AbstractActions, state: State, game: GameConfig, player: Array | int, *, trained: int = ABSENT) -> tuple[AbstractActions, Commands]:
    kernel, program = _stages(strategy, trained)
    if not _fits(game.bucket):
        return program(actions, state, game, player)
    return kernels.dispatch(
        actions, state, game, player, kernel=kernel, program=program
    )


def _fits(bucket: BucketSpec) -> bool:
    return whole_act.fits(
        bucket,
        bytes_per_cell=_BYTES_PER_CELL,
        lane_bytes=_LANE_BYTES,
        bytes_per_slot=_BYTES_PER_SLOT,
        bytes_per_entity=_BYTES_PER_ENTITY,
    )


@functools.cache
def _stages(strategy: int, trained: int) -> tuple[_Act, _Act]:
    def kernel(actions: AbstractActions, state: State, game: GameConfig, player: Array | int) -> tuple[AbstractActions, Commands]:
        return whole_act.act(
            _TARGET,
            actions,
            state,
            game,
            player,
            attributes={"strategy": strategy, "trained": trained},
        )

    def program(actions: AbstractActions, state: State, game: GameConfig, player: Array | int) -> tuple[AbstractActions, Commands]:
        decided = _decided(strategy, trained, actions, state, game, player)
        return abstraction.translate(decided, state, game, player, kinds=rush.KINDS)

    return kernel, program


def _decided(strategy: int, trained: int, actions: AbstractActions, state: State, game: GameConfig, player: Array | int) -> AbstractActions:
    if strategy == WORKER_RUSH:
        return rush.worker_rush(actions, state, game, player)
    if strategy == BARRACKS_RUSH:
        return rush.barracks_rush(actions, state, game, player, trained)
    if strategy == WORKER_DEFENSE:
        return defense.worker_defense(actions, state, game, player)
    if strategy == BARRACKS_DEFENSE:
        return defense.barracks_defense(actions, state, game, player, trained)
    raise ValueError(f"no abstraction family strategy is numbered {strategy}")
