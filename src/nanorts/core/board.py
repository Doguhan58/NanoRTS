from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from nanorts.model import ABSENT, DIRECTION_COUNT, DIRECTION_OFFSETS, GameConfig, State

_DIRECTION_DX = tuple(offset[0] for offset in DIRECTION_OFFSETS)
_DIRECTION_DY = tuple(offset[1] for offset in DIRECTION_OFFSETS)


class Neighbour(NamedTuple):
    free: Array
    occupied: Array
    slot: Array
    unit_type: Array
    owner: Array


def neighbour(state: State, cell: Array, direction: Array | np.ndarray, game: GameConfig) -> Neighbour:
    bucket = game.bucket
    width, height, cells = bucket.width, bucket.height, bucket.cell_count
    x, y = cell % width, cell // width

    array_module = np if isinstance(direction, np.ndarray) else jnp

    known_direction = array_module.clip(direction, 0, DIRECTION_COUNT - 1)
    neighbour_x = (x + array_module.asarray(_DIRECTION_DX, array_module.int32)[known_direction])
    neighbour_y = (y + array_module.asarray(_DIRECTION_DY, array_module.int32)[known_direction])
    is_direction = (direction >= 0) & (direction < DIRECTION_COUNT)

    on_board = (
        is_direction
        & (neighbour_x >= 0)
        & (neighbour_x < width)
        & (neighbour_y >= 0)
        & (neighbour_y < height)
    )

    neighbour_cell = jnp.where(on_board, neighbour_y * width + neighbour_x, 0)
    occupant = state.entity_at.reshape(cells)[neighbour_cell]
    occupied = on_board & (occupant != ABSENT)
    slot = jnp.where(occupied, occupant, 0)

    return Neighbour(
        free=on_board
        & (occupant == ABSENT)
        & ~game.terrain.reshape(cells)[neighbour_cell],
        occupied=occupied,
        slot=slot,
        unit_type=jnp.clip(state.unit_type[slot], 0, bucket.unit_type_capacity - 1),
        owner=state.owner[slot],
    )


def reserved_position(cell: Array, direction: Array, game: GameConfig) -> Array:
    width = game.bucket.width
    index_steps = jnp.asarray(
        [dx + dy * width for dx, dy in DIRECTION_OFFSETS], jnp.int32
    )
    return cell + index_steps[jnp.clip(direction, 0, DIRECTION_COUNT - 1)]


def position_is_blocked(state: State, game: GameConfig, position: Array) -> Array:
    cells = game.bucket.cell_count
    wall = game.terrain.reshape(cells)[position]
    occupant = state.entity_at.reshape(cells)[position]
    return wall | (occupant != ABSENT)
