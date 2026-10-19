from __future__ import annotations

import dataclasses
from typing import Self, overload

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from nanorts.model.bucket import BucketSpec
from nanorts.model.errors import ModelError
from nanorts.model.leaf import FIELD, SCALAR, leaf, nested, static
from nanorts.model.state import State
from nanorts.model.unit_types import UnitType, content_hash, host_arrays


class _Column:
    def __set_name__(self, owner: type[GameConfig], name: str) -> None:
        self._name = name

    @overload
    def __get__(self, game: None, owner: type[GameConfig]) -> Self: ...

    @overload
    def __get__(self, game: GameConfig, owner: type[GameConfig] | None = None) -> Array: ...

    def __get__(self, game: GameConfig | None, owner: type[GameConfig] | None = None) -> Self | Array:
        if game is None:
            return self
        return _traced_column(host_arrays(game.unit_types)[self._name])


def _traced_column(column: np.ndarray) -> Array:
    if _is_uniform(column):
        return jnp.full(column.shape, column.flat[0].item(), column.dtype)
    return jnp.asarray(column)


def _is_uniform(column: np.ndarray) -> bool:
    return bool(column.size) and bool((column == column.flat[0]).all())


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GameConfig:
    terrain: Array = leaf(FIELD, jnp.bool_)

    max_cycles: Array = leaf(SCALAR, jnp.int32)

    initial_state: State = nested()

    unit_types: tuple[UnitType, ...] = static()

    bucket: BucketSpec = static()

    unit_cost = _Column()

    unit_hit_points = _Column()

    unit_produce_time = _Column()

    unit_move_time = _Column()

    unit_attack_time = _Column()

    unit_harvest_time = _Column()

    unit_attack_range = _Column()

    unit_attack_damage = _Column()

    unit_harvest_amount = _Column()

    unit_sight_radius = _Column()

    unit_is_resource = _Column()

    unit_is_stockpile = _Column()

    unit_can_harvest = _Column()

    unit_can_move = _Column()

    unit_can_attack = _Column()

    unit_produces = _Column()

    def check_values(self) -> None:
        self._check_horizon()
        self._check_unit_type_table()

    def _check_horizon(self) -> None:
        if int(self.max_cycles) <= 0:
            raise ModelError(
                f"GameConfig.max_cycles is {int(self.max_cycles)}, "
                "expected a positive horizon"
            )

    def _check_unit_type_table(self) -> None:
        table = self.unit_types
        if not (
            isinstance(table, tuple) and all(isinstance(t, UnitType) for t in table)
        ):
            raise ModelError(
                f"GameConfig.unit_types is {type(table).__name__}, "
                "expected a tuple of UnitType"
            )
        if len(table) != self.bucket.unit_type_capacity:
            raise ModelError(
                f"GameConfig.unit_types holds {len(table)} types, expected the "
                f"Bucket's unit_type_capacity {self.bucket.unit_type_capacity}"
            )
        try:
            digest = content_hash(table)
        except (ModelError, TypeError) as error:
            raise ModelError(f"GameConfig.unit_types: {error}") from error
        if digest != self.bucket.utt_content_hash:
            raise ModelError(
                f"GameConfig.unit_types hashes to {digest}, not the Bucket's "
                f"utt_content_hash {self.bucket.utt_content_hash}"
            )
