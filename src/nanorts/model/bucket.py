from __future__ import annotations

import dataclasses
import functools
import math
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jax import Array

from nanorts.model.errors import ModelError
from nanorts.model.vocabulary import (
    ABSENT,
    PLAYER_COUNT,
    ConflictStrategy,
    UnitTypeTableVersion,
)

if TYPE_CHECKING:
    from nanorts.model.config import GameConfig


@dataclasses.dataclass(frozen=True)
class BucketSpec:
    height: int
    width: int
    neutral_capacity: int
    per_player_capacity: int
    unit_type_capacity: int
    attack_offset_capacity: int
    utt_identity: UnitTypeTableVersion
    utt_content_hash: str
    conflict_strategy: ConflictStrategy

    @property
    def cell_count(self) -> int:
        return self.height * self.width

    @property
    def entity_capacity(self) -> int:
        return self.neutral_capacity + PLAYER_COUNT * self.per_player_capacity

    @property
    def attack_window_side(self) -> int:
        return math.isqrt(self.attack_offset_capacity)

    @property
    def max_attack_range(self) -> int:
        return (self.attack_window_side - 1) // 2

    def player_slot_start(self, player: int) -> int:
        return self.neutral_capacity + player * self.per_player_capacity

    def player_slots(self, player: int) -> range:
        start = self.player_slot_start(player)
        return range(start, start + self.per_player_capacity)

    def check_values(self) -> None:
        self._check_positive_extents()
        self._check_neutral_capacity()
        self._check_attack_window()
        self._check_enumerations()

    def _check_positive_extents(self) -> None:
        for name in (
            "height",
            "width",
            "per_player_capacity",
            "unit_type_capacity",
            "attack_offset_capacity",
        ):
            value = getattr(self, name)
            if value <= 0:
                raise ModelError(
                    f"BucketSpec.{name} is {value}, expected a positive int"
                )

    def _check_neutral_capacity(self) -> None:
        if self.neutral_capacity < 0:
            raise ModelError(
                f"BucketSpec.neutral_capacity is {self.neutral_capacity}, "
                "expected a non-negative int"
            )

    def _check_attack_window(self) -> None:
        side = self.attack_window_side
        if side * side != self.attack_offset_capacity or side % 2 == 0:
            raise ModelError(
                f"BucketSpec.attack_offset_capacity is "
                f"{self.attack_offset_capacity}, expected the odd square "
                "(2 * max attack range + 1) ** 2"
            )

    def _check_enumerations(self) -> None:
        if self.utt_identity not in tuple(UnitTypeTableVersion):
            raise ModelError(
                f"BucketSpec.utt_identity is {self.utt_identity!r}, expected one "
                f"of {[member.value for member in UnitTypeTableVersion]}"
            )
        if self.conflict_strategy not in tuple(ConflictStrategy):
            raise ModelError(
                f"BucketSpec.conflict_strategy is {self.conflict_strategy!r}, "
                f"expected one of {[member.value for member in ConflictStrategy]}"
            )


def attack_offset_index(bucket: BucketSpec, dx: int, dy: int) -> int:
    radius = bucket.max_attack_range
    return (radius + dy) * bucket.attack_window_side + (radius + dx)


@functools.cache
def attack_offsets(bucket: BucketSpec) -> np.ndarray:
    attack_range = bucket.max_attack_range
    offsets = np.zeros((bucket.attack_offset_capacity, 2), np.int32)
    for dy in range(-attack_range, attack_range + 1):
        for dx in range(-attack_range, attack_range + 1):
            offsets[attack_offset_index(bucket, dx, dy)] = (dx, dy)
    offsets.flags.writeable = False
    return offsets


def window_targets(cell: Array, game: GameConfig) -> Array:
    width, height = game.bucket.width, game.bucket.height
    offsets = attack_offsets(game.bucket)
    x = cell[..., None] % width + offsets[:, 0]
    y = cell[..., None] // width + offsets[:, 1]
    on_board = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return jnp.where(on_board, y * width + x, ABSENT)
