from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from nanorts import kernels
from nanorts.model import (
    ABSENT,
    FIELD,
    PER_SLOT,
    PLAYER_COUNT,
    BoardView,
    BucketSpec,
    GameConfig,
    LeafField,
    State,
    board_view,
    empty,
    leaf,
    model_fields,
)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Visibility:
    visible_cells: Array = leaf(FIELD, jnp.bool_, fill=False)


def visibility(game: GameConfig, state: State, *, observer: int) -> Visibility:
    block = _block(game.bucket, observer)
    if game.bucket.cell_count > _KERNEL_CELLS:
        return Visibility(visible_cells=_cells_by_test(game, state, block))
    kernel, program = _stages(block.start, block.stop)
    return Visibility(
        visible_cells=kernels.dispatch(game, state, kernel=kernel, program=program)
    )


_Stage = Callable[[GameConfig, State], Array]


@functools.cache
def _stages(first: int, stop: int) -> tuple[_Stage, _Stage]:
    block = slice(first, stop)

    def kernel(game: GameConfig, state: State) -> Array:
        return _cells_by_stamp(game, state, block)

    def program(game: GameConfig, state: State) -> Array:
        return _cells_by_test(game, state, block)

    return kernel, program


def _cells_by_test(game: GameConfig, state: State, block: slice) -> Array:
    bucket = game.bucket
    active = state.active[block]
    cell = jnp.where(active, state.cell[block], 0)
    unit_type = jnp.where(active, state.unit_type[block], 0)
    radius = game.unit_sight_radius[unit_type]
    dx = (
        jnp.arange(bucket.width, dtype=jnp.int32)[None, :, None]
        - (cell % bucket.width)[None, None, :]
    )
    dy = (
        jnp.arange(bucket.height, dtype=jnp.int32)[:, None, None]
        - (cell // bucket.width)[None, None, :]
    )
    inside = (dx * dx + dy * dy) <= (radius * radius)[None, None, :]
    return jnp.any(inside & active[None, None, :], axis=2)


_VISIBILITY_TARGET = "nanorts_visible_cells"

_KERNEL_CELLS = 48 * 1024


def _cells_by_stamp(game: GameConfig, state: State, block: slice) -> Array:
    bucket = game.bucket
    (cells,) = kernels.call(
        _VISIBILITY_TARGET,
        [
            state.birth,
            state.unit_type,
            state.cell,
            jnp.asarray(game.unit_sight_radius, jnp.int32),
        ],
        [jax.ShapeDtypeStruct((bucket.height, bucket.width), jnp.bool_)],
        attributes={
            "first": block.start,
            "count": block.stop - block.start,
            "width": bucket.width,
            "height": bucket.height,
        },
    )
    return cells


def project(game: GameConfig, state: State, *, observer: int) -> State:
    bucket = game.bucket
    seen = visibility(game, state, observer=observer).visible_cells.reshape(-1)
    cell_is_seen = seen[jnp.clip(state.cell, 0, bucket.cell_count - 1)]
    kept = state.active & ((state.owner == observer) | cell_is_seen)
    canonical = empty(State, bucket)
    cleared: dict[str, Any] = {}
    for field in model_fields(State):
        if isinstance(field, LeafField) and field.spec.shape is PER_SLOT:
            cleared[field.name] = jnp.where(
                kept, getattr(state, field.name), getattr(canonical, field.name)
            )
    board = board_view(state, game)
    survived = board.present & kept[board.slot]
    return dataclasses.replace(
        state,
        entity_at=jnp.where(survived, state.entity_at, ABSENT),
        **cleared,
    )


def visible_board(game: GameConfig, state: State, *, observer: int) -> BoardView:
    board = board_view(state, game)
    seen = visibility(game, state, observer=observer).visible_cells
    shown = board.present & ((board.owner == observer) | seen)
    concealed = empty(BoardView, game.bucket)
    return jax.tree.map(
        lambda fact, fill: jnp.where(shown, fact, fill), board, concealed
    )


def observer_index(observer: object) -> int:
    if (
        not isinstance(observer, int)
        or isinstance(observer, bool)
        or observer not in range(PLAYER_COUNT)
    ):
        raise ValueError(
            f"observer is {observer!r}, expected one of {list(range(PLAYER_COUNT))}"
        )
    return observer


def _block(bucket: BucketSpec, observer: int) -> slice:
    slots = bucket.player_slots(observer_index(observer))
    return slice(slots.start, slots.stop)
