from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Final

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from nanorts import kernels
from nanorts.core import kinds, unit_tables
from nanorts.model import (
    ACTION_KIND_COUNT,
    DIRECTION_COUNT,
    KIND_ABSENT,
    PLAYER_COUNT,
    BucketSpec,
    GameConfig,
    Legality,
    State,
)

_ROWS_KERNEL = "nanorts_legality"
_WORDS_KERNEL = "nanorts_legality_words"

_BITS_PER_WORD = 32
_WORDS_PER_ROW = 3


def derive(state: State, game: GameConfig) -> Legality:
    bucket = game.bucket

    def owned(field: Array) -> Array:
        return _owned(field, bucket)

    active = owned(state.active)
    available = _available(active, owned(state.pending_kind))
    actor_type = jnp.clip(owned(state.unit_type), 0, bucket.unit_type_capacity - 1)
    owner, carried, cell = map(
        owned, (state.owner, state.carried_resources, state.cell)
    )
    directions = np.arange(DIRECTION_COUNT, dtype=np.int32)
    unit_types = np.arange(bucket.unit_type_capacity, dtype=np.int32)
    kind_bits = []
    factors: dict[str, Array] = {}
    for action_kind in kinds.KINDS:
        grid = kinds.LegalityGrid(
            state=state,
            game=game,
            active=active,
            actor_type=actor_type,
            owner=owner,
            carried=carried,
            cell=cell,
            directions=directions,
            unit_types=unit_types,
            is_valid=action_kind.is_valid,
        )
        kind_bit, kind_factors = action_kind.legality(grid)
        kind_bits.append(kind_bit)
        factors |= kind_factors
    return Legality(available=available, kind=jnp.stack(kind_bits, axis=-1), **factors)


def _derive_on_device(state: State, game: GameConfig, *, stored: str) -> Legality:
    bucket = game.bucket
    rows = _kernel_rows(state, game, stored)
    available = _available(
        _owned(state.active, bucket), _owned(state.pending_kind, bucket)
    )
    return Legality(available=available, **_factors_from_rows(rows, bucket))


KERNEL_PATHS: Final[dict[str, Callable[[State, GameConfig], Legality]]] = {
    form: functools.partial(_derive_on_device, stored=form)
    for form in ("rows", "words")
}


def _kernel_rows(state: State, game: GameConfig, stored: str) -> Array:
    bucket = game.bucket
    if stored == "words" and _rows_fit_words(bucket):
        return _unpack_words(_kernel_words(state, game))
    (rows,) = kernels.call(
        _ROWS_KERNEL,
        _kernel_operands(state, game),
        [
            jax.ShapeDtypeStruct(
                (PLAYER_COUNT, bucket.per_player_capacity, _row_width(bucket)),
                jnp.bool_,
            )
        ],
    )
    return rows


def _kernel_words(state: State, game: GameConfig) -> Array:
    bucket = game.bucket
    (words,) = kernels.call(
        _WORDS_KERNEL,
        _kernel_operands(state, game),
        [
            jax.ShapeDtypeStruct(
                (PLAYER_COUNT, bucket.per_player_capacity, _WORDS_PER_ROW), jnp.uint32
            )
        ],
        attributes={"offsets": bucket.attack_offset_capacity},
    )
    return words


def _kernel_operands(state: State, game: GameConfig) -> list[Array]:
    return [
        state.birth,
        state.unit_type,
        state.owner,
        state.carried_resources,
        state.cell,
        state.entity_at,
        state.player_resources,
        game.terrain,
        jnp.asarray(unit_tables.unit_table(game)),
    ]


def _factor_widths(bucket: BucketSpec) -> dict[str, int]:
    return {
        "kind": ACTION_KIND_COUNT,
        "move_direction": DIRECTION_COUNT,
        "harvest_direction": DIRECTION_COUNT,
        "return_direction": DIRECTION_COUNT,
        "produce_direction": DIRECTION_COUNT,
        "produce_type": bucket.unit_type_capacity,
        "attack_offset": bucket.attack_offset_capacity,
    }


def _row_width(bucket: BucketSpec) -> int:
    return sum(_factor_widths(bucket).values())


def _rows_fit_words(bucket: BucketSpec) -> bool:
    return _row_width(bucket) <= _WORDS_PER_ROW * _BITS_PER_WORD


def _unpack_words(words: Array) -> Array:
    shifts = lax.iota(jnp.uint32, _BITS_PER_WORD)
    bits = ((words[..., None] >> shifts) & 1) != 0
    return bits.reshape(*words.shape[:-1], _WORDS_PER_ROW * _BITS_PER_WORD)


def _factors_from_rows(rows: Array, bucket: BucketSpec) -> dict[str, Array]:
    widths = _factor_widths(bucket)
    ends = np.cumsum(list(widths.values()))
    return {
        name: rows[..., end - width : end]
        for (name, width), end in zip(widths.items(), ends, strict=True)
    }


def _owned(field: Array, bucket: BucketSpec) -> Array:
    return field[bucket.neutral_capacity :].reshape(
        PLAYER_COUNT, bucket.per_player_capacity
    )


def _available(active: Array, pending_kind: Array) -> Array:
    return active & (pending_kind == KIND_ABSENT)
