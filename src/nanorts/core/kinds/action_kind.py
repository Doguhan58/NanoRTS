from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from nanorts.core import board
from nanorts.core.effect import Effect
from nanorts.model import ABSENT, GameConfig, State


@dataclasses.dataclass(frozen=True)
class Request:
    state: State
    game: GameConfig
    actor_type: Array
    owner: Array
    carried: Array
    cell: Array
    direction: Array | np.ndarray
    target_cell: Array
    produced_type: Array | np.ndarray

    @classmethod
    def of(cls, state: State, game: GameConfig, slot: Array, direction: Array, target_cell: Array, produced_type: Array) -> Request:
        return cls(
            state,
            game,
            jnp.clip(state.unit_type[slot], 0, game.bucket.unit_type_capacity - 1),
            state.owner[slot],
            state.carried_resources[slot],
            state.cell[slot],
            direction,
            target_cell,
            produced_type,
        )

    @functools.cached_property
    def neighbour(self) -> board.Neighbour:
        return board.neighbour(self.state, self.cell, self.direction, self.game)


IsValid = Callable[[Request], Array | np.ndarray | bool]


@dataclasses.dataclass(frozen=True)
class LegalityGrid:
    state: State
    game: GameConfig
    active: Array
    actor_type: Array
    owner: Array
    carried: Array
    cell: Array
    directions: np.ndarray
    unit_types: np.ndarray
    is_valid: IsValid

    def ask(self, *, direction: Array | np.ndarray | int = ABSENT, target_cell: Array | int = ABSENT, produced_type: Array | np.ndarray | int = ABSENT) -> Array:
        direction, produced_type = (
            jnp.asarray(value, jnp.int32)
            if isinstance(value, jax.Array)
            else np.asarray(value, np.int32)
            for value in (direction, produced_type)
        )
        target_cell = jnp.asarray(target_cell, jnp.int32)
        parameter_axes = max(p.ndim for p in (direction, target_cell, produced_type))
        per_slot = (..., *(None,) * parameter_axes)
        request = Request(
            self.state,
            self.game,
            self.actor_type[per_slot],
            self.owner[per_slot],
            self.carried[per_slot],
            self.cell[per_slot],
            direction,
            target_cell,
            produced_type,
        )
        return self.active[per_slot] & self.is_valid(request)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RetainedActor:
    kind: Array
    direction: Array
    target_cell: Array
    produced_type: Array
    birth: Array
    owner: Array
    unit_type: Array
    cell: Array
    carried: Array


@dataclasses.dataclass(frozen=True)
class Member:
    state: State
    game: GameConfig
    slot: Array
    all_slots: Array
    in_batch: Array
    retained: RetainedActor
    actor_type: Array
    player: Array
    alive: Array
    destination: Array
    neighbour: board.Neighbour

    def performs(self, code: int) -> Array:
        return self.in_batch & (self.retained.kind == code)


@dataclasses.dataclass(frozen=True)
class ActionKind:
    code: int
    duration: Callable[[GameConfig, Array, Array, Array], Array]
    is_valid: IsValid
    claims_position: bool
    cost: Callable[[GameConfig, Array], Array] | None
    payload: frozenset[str]
    legality: Callable[[LegalityGrid], tuple[Array, dict[str, Array]]]
    complete: Callable[[Member], Effect]
