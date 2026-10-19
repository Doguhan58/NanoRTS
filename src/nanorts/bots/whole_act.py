from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, Final

import jax
import jax.numpy as jnp
from jax import Array

from nanorts import kernels
from nanorts.model import (
    PLAYER_COUNT,
    BucketSpec,
    Commands,
    GameConfig,
    State,
    host_arrays,
)

_STATE_LEAVES: Final[tuple[str, ...]] = (
    "birth",
    "unit_type",
    "owner",
    "cell",
    "carried_resources",
    "pending_kind",
    "pending_direction",
    "pending_produced_type",
    "entity_at",
    "player_resources",
)

_TABLE_COLUMNS: Final[tuple[str, ...]] = (
    "unit_can_harvest",
    "unit_is_resource",
    "unit_is_stockpile",
    "unit_can_attack",
    "unit_attack_range",
    "unit_cost",
)

_COMMAND_FIELDS: Final[tuple[str, ...]] = tuple(
    field.name for field in dataclasses.fields(Commands)
)

_SHARED_BYTES: Final[int] = 48 * 1024

_WARP: Final[int] = 32

_MIN_THREADS: Final[int] = 64

_MAX_THREADS: Final[int] = 256

_KEY_PRNG: Final[str] = "threefry2x32"

_FLOAT_DRAWS_SHOWN_ON: Final[frozenset[str]] = frozenset({"8.9"})


def _threads(capacity: int) -> int:
    return max(_MIN_THREADS, -(-capacity // _WARP) * _WARP)


def fits(bucket: BucketSpec, *, bytes_per_cell: int, lane_bytes: int, bytes_per_slot: int, bytes_per_entity: int = 0) -> bool:
    capacity = bucket.per_player_capacity
    shared = (
        bucket.cell_count * bytes_per_cell
        + lane_bytes
        + capacity * bytes_per_slot
        + bucket.entity_capacity * bytes_per_entity
    )
    return _threads(capacity) <= _MAX_THREADS and shared <= _SHARED_BYTES


def unit_table(game: GameConfig, columns: Sequence[str] = _TABLE_COLUMNS) -> Array:
    table = host_arrays(game.unit_types)
    return jnp.asarray(
        [
            [int(table[name][row]) for name in columns]
            for row in range(len(game.unit_types))
        ],
        jnp.int32,
    )


def act[Carry](target: str, carry: Carry, state: State, game: GameConfig, player: Array | int, *, attributes: dict[str, int], operands: Sequence[Array] = ()) -> tuple[Carry, Commands]:
    carry, commands, _ = call(
        target,
        carry,
        state,
        game,
        player,
        attributes=attributes,
        operands=operands,
    )
    return carry, commands


def call[Carry](target: str, carry: Carry, state: State, game: GameConfig, player: Array | int, *, attributes: dict[str, int], operands: Sequence[Array] = (), results: Sequence[jax.ShapeDtypeStruct] = ()) -> tuple[Carry, Commands, list[Array]]:
    read = [
        *(getattr(state, name) for name in _STATE_LEAVES),
        game.terrain,
        unit_table(game),
        jnp.asarray(player, jnp.int32),
        *operands,
    ]
    leaves, tree = jax.tree.flatten(carry)
    capacity = game.bucket.per_player_capacity
    outs = kernels.call(
        target,
        [*read, *leaves],
        [
            *(jax.ShapeDtypeStruct(leaf.shape, leaf.dtype) for leaf in leaves),
            jax.ShapeDtypeStruct(
                (len(_COMMAND_FIELDS), PLAYER_COUNT, capacity), jnp.int32
            ),
            *results,
        ],
        aliases={len(read) + index: index for index in range(len(leaves))},
        attributes=attributes,
    )
    written, (commands, *extra) = outs[: len(leaves)], outs[len(leaves) :]
    issued: dict[str, Any] = {
        name: commands[index] for index, name in enumerate(_COMMAND_FIELDS)
    }
    return jax.tree.unflatten(tree, written), Commands(**issued), extra


def draws_valid(key: Array) -> bool:
    return (
        jnp.issubdtype(key.dtype, jax.dtypes.prng_key)
        and str(jax.random.key_impl(key)) == _KEY_PRNG
        and jax.config.jax_default_prng_impl == _KEY_PRNG
        and bool(jax.config.jax_threefry_partitionable)
        and not jax.config.jax_enable_x64
        and not jax.config.jax_high_dynamic_range_gumbel
    )


def float_draws_served() -> bool:
    return kernels.devices_within(_FLOAT_DRAWS_SHOWN_ON)
