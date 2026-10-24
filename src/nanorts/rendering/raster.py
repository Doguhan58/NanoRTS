from __future__ import annotations

import dataclasses
from typing import Final

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from nanorts.model import KIND_NONE, PLAYER_COUNT, GameConfig, State, board_view, static

OWNER_COUNT = 1 + PLAYER_COUNT

EMPTY_OCCUPANT = 0

SPRITE_PIXELS: Final[int] = 16

BAND_ROWS: Final[tuple[int, int, int]] = (12, 13, 14)

BAND_COLUMN: Final[int] = 1

BAND_WIDTH: Final[int] = SPRITE_PIXELS - 2 * BAND_COLUMN

HIT_POINT_WIDTH: Final[int] = BAND_WIDTH

KIND_WIDTH: Final[int] = BAND_WIDTH

STACK_BITS: Final[int] = BAND_WIDTH

BAND_ON: Final[tuple[int, int, int]] = (0xFF, 0xFF, 0xFF)
BAND_OFF: Final[tuple[int, int, int]] = (0x00, 0x00, 0x00)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RenderAssets:
    tiles: Array

    cell_pixels: int = static()

    unit_type_capacity: int = static()


def occupant_states(unit_type_capacity: int) -> int:
    return 1 + OWNER_COUNT * unit_type_capacity


def tile_count(unit_type_capacity: int) -> int:
    return 2 * occupant_states(unit_type_capacity)


def occupant_index(owner: Array | int, unit_type: Array | int, unit_type_capacity: int) -> Array | int:
    return 1 + (owner + 1) * unit_type_capacity + unit_type


def tile_index(terrain: Array | int, occupant: Array | int, unit_type_capacity: int) -> Array | int:
    return terrain * occupant_states(unit_type_capacity) + occupant


def _banded(cells: Array, occupied: Array, kind: Array, stack: Array, hit_points: Array, cell_pixels: int) -> Array:
    scale = cell_pixels // SPRITE_PIXELS
    sprite_pixel = np.arange(cell_pixels, dtype=np.int32) // scale
    row, column = sprite_pixel[:, None], sprite_pixel[None, :]
    position = jnp.clip(
        (jnp.arange(cell_pixels, dtype=jnp.int32) // scale)[None, :] - BAND_COLUMN,
        0,
        BAND_WIDTH - 1,
    )
    lit = jnp.where(
        row == BAND_ROWS[0],
        kind[..., None, None] == position + 1,
        jnp.where(
            row == BAND_ROWS[1],
            ((stack[..., None, None] >> position) & 1) != 0,
            hit_points[..., None, None] > position,
        ),
    )
    band = (
        (row >= BAND_ROWS[0])
        & (row <= BAND_ROWS[-1])
        & (column >= BAND_COLUMN)
        & (column < BAND_COLUMN + BAND_WIDTH)
    )
    inside = occupied[..., None, None] & band
    colour = jnp.where(
        lit[..., None],
        jnp.asarray(BAND_ON, jnp.uint8),
        jnp.asarray(BAND_OFF, jnp.uint8),
    )
    return jnp.where(inside[..., None], colour, cells)


def draw(terrain: Array, present: Array, owner: Array, unit_type: Array, pending_kind: Array, carried_resources: Array, hit_points: Array, assets: RenderAssets) -> Array:
    types = assets.unit_type_capacity
    occupant = jnp.where(
        present,
        occupant_index(owner.astype(jnp.int32), unit_type, types),
        EMPTY_OCCUPANT,
    )
    classes = tile_index(terrain.astype(jnp.int32), occupant, types)
    height, width = terrain.shape
    cells = _banded(
        assets.tiles[classes],
        present,
        jnp.maximum(pending_kind, KIND_NONE),
        carried_resources,
        hit_points,
        assets.cell_pixels,
    )
    return cells.transpose(0, 2, 1, 3, 4).reshape(
        height * assets.cell_pixels, width * assets.cell_pixels, 3
    )


def render(state: State, game: GameConfig, assets: RenderAssets) -> Array:
    board = board_view(state, game)
    return draw(
        game.terrain,
        board.present,
        board.owner,
        board.unit_type,
        board.pending_kind,
        board.carried_resources,
        board.hit_points,
        assets,
    )
