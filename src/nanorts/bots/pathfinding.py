from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Final, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, lax

from nanorts import kernels
from nanorts.model import (
    ABSENT,
    DIRECTION_DOWN,
    DIRECTION_LEFT,
    DIRECTION_OFFSETS,
    DIRECTION_RIGHT,
    DIRECTION_UP,
    KIND_MOVE,
    KIND_PRODUCE,
    BucketSpec,
    GameConfig,
    ModelError,
    State,
)

_CLOSED: Final[int] = int(jnp.iinfo(jnp.int32).max)
_BLOCKED: Final[int] = _CLOSED - 1
_UNSEEN: Final[int] = _CLOSED - 2
_STEP_X: Final[tuple[int, ...]] = tuple(dx for dx, _ in DIRECTION_OFFSETS)
_STEP_Y: Final[tuple[int, ...]] = tuple(dy for _, dy in DIRECTION_OFFSETS)


class Path(NamedTuple):
    found: Array
    step: Array
    distance: Array
    pops: Array


class _Query(NamedTuple):
    start: Array
    target: Array
    squared_range: Array


class _Search(NamedTuple):
    key: Array
    after_start: Array
    inserted: Array
    goal: Array
    searching: Array
    pops: Array


def free_cells(state: State, game: GameConfig) -> Array:
    bucket = game.bucket
    cells = bucket.cell_count
    occupied = state.entity_at.reshape(cells) != ABSENT
    kind = state.pending_kind.astype(jnp.int32)
    claims = state.active & ((kind == KIND_MOVE) | (kind == KIND_PRODUCE))
    claimed = _neighbour(
        state.cell.astype(jnp.int32), state.pending_direction.astype(jnp.int32), bucket
    )
    claimed_cells = (
        jnp.zeros(cells, jnp.bool_)
        .at[jnp.where(claims, claimed, cells)]
        .set(True, mode="drop")
    )
    return ~(game.terrain.reshape(cells) | occupied | claimed_cells)


def search(free: Array, start: Array, target: Array, range_: Array, bucket: BucketSpec) -> Path:
    _check_key_width(bucket)
    if not fits_kernel(bucket):
        return _search_by_program(free, start, target, range_, bucket)
    kernel, program = _stages(bucket)
    return kernels.dispatch(free, start, target, range_, kernel=kernel, program=program)


def fits_kernel(bucket: BucketSpec) -> bool:
    return bucket.cell_count <= _KERNEL_CELLS


_Stage = Callable[[Array, Array, Array, Array], Path]


@functools.cache
def _stages(bucket: BucketSpec) -> tuple[_Stage, _Stage]:
    def kernel(free: Array, start: Array, target: Array, range_: Array) -> Path:
        return _search_by_kernel(free, start, target, range_, bucket)

    def program(free: Array, start: Array, target: Array, range_: Array) -> Path:
        return _search_by_program(free, start, target, range_, bucket)

    return kernel, program


_SEARCH_TARGET = "nanorts_astar_search"

_SHARED_BYTES = 48 * 1024

_SCRATCH_BYTES_PER_CELL = 7

_KERNEL_CELLS = _SHARED_BYTES // _SCRATCH_BYTES_PER_CELL


def _search_by_kernel(free: Array, start: Array, target: Array, range_: Array, bucket: BucketSpec) -> Path:
    (answer,) = kernels.call(
        _SEARCH_TARGET,
        [
            jnp.asarray(free, jnp.bool_),
            jnp.asarray(start, jnp.int32),
            jnp.asarray(target, jnp.int32),
            jnp.asarray(range_, jnp.int32),
        ],
        [jax.ShapeDtypeStruct((4,), jnp.int32)],
        attributes={"width": bucket.width, "height": bucket.height},
    )
    return Path(
        found=answer[0] != 0, step=answer[1], distance=answer[2], pops=answer[3]
    )


def _search_by_program(free: Array, start: Array, target: Array, range_: Array, bucket: BucketSpec) -> Path:
    query = _Query(
        start=jnp.asarray(start, jnp.int32),
        target=jnp.asarray(target, jnp.int32),
        squared_range=jnp.asarray(range_, jnp.int32) ** 2,
    )
    ended = lax.while_loop(
        lambda searched: searched.searching & (searched.pops < bucket.cell_count),
        lambda searched: _pop(searched, query, bucket),
        _begin(free, query, bucket),
    )
    return _answer(ended, query, bucket)


def _check_key_width(bucket: BucketSpec) -> None:
    longest_f = bucket.cell_count + bucket.height + bucket.width
    if (longest_f + 1) * (bucket.cell_count + 1) >= _UNSEEN:
        raise ModelError(
            f"a {bucket.height}x{bucket.width} board has more cells than an "
            "A* open-list key can order"
        )


def _neighbour(cell: Array, direction: Array, bucket: BucketSpec) -> Array:
    flat_steps = tuple(dy * bucket.width + dx for dx, dy in DIRECTION_OFFSETS)
    known = jnp.clip(direction, 0, len(DIRECTION_OFFSETS) - 1)
    return cell + jnp.asarray(flat_steps, jnp.int32)[known]


def _manhattan(cell: Array, target: Array, width: int) -> Array:
    return jnp.abs(cell % width - target % width) + jnp.abs(
        cell // width - target // width
    )


def _in_range(cell: Array, query: _Query, width: int) -> Array:
    dx = cell % width - query.target % width
    dy = cell // width - query.target // width
    return dx * dx + dy * dy <= query.squared_range


def _open_key(f: Array, sequence: Array, bucket: BucketSpec) -> Array:
    return f * (bucket.cell_count + 1) + (bucket.cell_count - sequence)


def _cost(key: Array, cell: Array, query: _Query, bucket: BucketSpec) -> Array:
    f = key // (bucket.cell_count + 1)
    return f - _manhattan(cell, query.target, bucket.width)


def _begin(free: Array, query: _Query, bucket: BucketSpec) -> _Search:
    cells = jnp.arange(bucket.cell_count, dtype=jnp.int32)
    is_start = cells == query.start
    start_f = _manhattan(query.start, query.target, bucket.width)
    return _Search(
        key=jnp.where(
            is_start,
            _open_key(start_f, jnp.int32(0), bucket),
            jnp.where(free, _UNSEEN, _BLOCKED),
        ),
        after_start=jnp.where(is_start, query.start, ABSENT),
        inserted=jnp.int32(1),
        goal=jnp.int32(ABSENT),
        searching=jnp.bool_(True),
        pops=jnp.int32(0),
    )


def _pop(searched: _Search, query: _Query, bucket: BucketSpec) -> _Search:
    width, height = bucket.width, bucket.height
    cell = jnp.argmin(searched.key).astype(jnp.int32)
    key = searched.key[cell]
    exhausted = key >= _UNSEEN
    reached = ~exhausted & _in_range(cell, query, width)
    expands = ~exhausted & ~reached
    x = cell % width + jnp.asarray(_STEP_X, jnp.int32)
    y = cell // width + jnp.asarray(_STEP_Y, jnp.int32)
    on_board = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    neighbour = jnp.where(on_board, y * width + x, 0)
    inserts = expands & on_board & (searched.key[neighbour] == _UNSEEN)
    counted = inserts.astype(jnp.int32)
    sequence = searched.inserted + jnp.cumsum(counted) - counted
    neighbour_f = (
        _cost(key, cell, query, bucket) + 1 + _manhattan(neighbour, query.target, width)
    )
    neighbour_key = _open_key(neighbour_f, sequence, bucket)
    neighbour_after_start = jnp.where(
        cell == query.start, neighbour, searched.after_start[cell]
    )
    cells = jnp.arange(bucket.cell_count, dtype=jnp.int32)
    keys = jnp.where(expands & (cells == cell), _CLOSED, searched.key)
    after_starts = searched.after_start
    for k in range(len(DIRECTION_OFFSETS)):
        lands = inserts[k] & (cells == neighbour[k])
        keys = jnp.where(lands, neighbour_key[k], keys)
        after_starts = jnp.where(lands, neighbour_after_start[k], after_starts)
    return _Search(
        key=keys,
        after_start=after_starts,
        inserted=searched.inserted + jnp.sum(counted),
        goal=jnp.where(reached, cell, searched.goal),
        searching=expands,
        pops=searched.pops + (~exhausted).astype(jnp.int32),
    )


def _answer(ended: _Search, query: _Query, bucket: BucketSpec) -> Path:
    width = bucket.width
    found = ended.goal != ABSENT
    goal = jnp.clip(ended.goal, 0, bucket.cell_count - 1)
    first_offset = ended.after_start[goal] - query.start
    step = jnp.select(
        [
            first_offset == width,
            first_offset == -1,
            first_offset == -width,
            first_offset == 1,
        ],
        [DIRECTION_DOWN, DIRECTION_LEFT, DIRECTION_UP, DIRECTION_RIGHT],
        ABSENT,
    )
    distance = _cost(ended.key[goal], goal, query, bucket)
    return Path(
        found=found,
        step=jnp.where(found, step, ABSENT).astype(jnp.int32),
        distance=jnp.where(found, distance, ABSENT).astype(jnp.int32),
        pops=ended.pops,
    )
