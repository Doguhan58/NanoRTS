from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from nanorts import core, kernels
from nanorts.adapters import factors
from nanorts.adapters.gridnet import source_filter
from nanorts.adapters.gridnet.layout import ACTION_FACTORS, GridNetLayout
from nanorts.adapters.gridnet.source_filter import DecodedRows
from nanorts.model import (
    ABSENT,
    KIND_ABSENT,
    KIND_NONE,
    KIND_PRODUCE,
    PLAYER_COUNT,
    BucketSpec,
    Commands,
    GameConfig,
    ModelError,
    State,
)


def decode_actions(actions: Array, state: State, game: GameConfig, *, layout: GridNetLayout) -> Commands:
    if _kernel_fits(game.bucket):
        rows = kernels.dispatch(
            actions, state, game, kernel=_decode_in_kernel, program=_decode
        )
    else:
        rows = _decode(actions, state, game)
    return Commands(
        **{name: jnp.stack(values).astype(jnp.int32) for name, values in rows.items()}
    )


type _Rows = dict[str, list[Array]]


def _decode(actions: Array, state: State, game: GameConfig) -> _Rows:
    bucket = game.bucket
    base_positions, base_resources = source_filter.pending_usage(state, game)
    legality = core.legal_commands(state, game)

    decoded: list[DecodedRows] = []
    for player in range(PLAYER_COUNT):
        rows = _decode_rows(actions[player], state, game, player)
        rows = source_filter.keep(
            rows, base_positions, base_resources, state, player, bucket
        )
        decoded.append(rows)
        if player + 1 < PLAYER_COUNT:
            base_positions, base_resources = source_filter.issued_usage(
                rows, legality, base_positions, base_resources, player, bucket
            )
    return _commands_of(decoded, state, bucket)


def check_actions(actions: Array, *, layout: GridNetLayout) -> None:
    array = np.asarray(actions)
    if array.shape != layout.actions_shape:
        raise ModelError(
            f"gridnet actions have shape {array.shape}, expected "
            f"{layout.actions_shape} - the leading {PLAYER_COUNT} is player "
            "identity, not an environment batch"
        )
    if array.dtype != np.int32:
        raise ModelError(f"gridnet actions have dtype {array.dtype}, expected int32")
    factors.refuse_out_of_range(
        array,
        ACTION_FACTORS,
        layout.action_factor_sizes,
        subject="gridnet action factor {index} ({name})",
        exempt="an unused one and one at a non-source cell",
    )


def _decode_rows(actions: Array, state: State, game: GameConfig, player: int) -> DecodedRows:
    bucket = game.bucket
    width, cells = bucket.width, bucket.cell_count
    start = bucket.player_slot_start(player)
    owned = slice(start, start + bucket.per_player_capacity)

    active = state.active[owned]
    available = active & (state.pending_kind[owned] == KIND_ABSENT)
    cell = state.cell[owned]
    order = jnp.argsort(jnp.where(available, cell, cells))
    count = jnp.sum(available.astype(jnp.int32))

    at_cell = cell[order]
    rows = actions.reshape(cells, len(ACTION_FACTORS))[jnp.clip(at_cell, 0, cells - 1)]
    kind = rows[..., 0]
    direction = factors.direction(kind, rows)
    produced_type = factors.produced_type(kind, rows)
    target_cell = factors.attack_target(kind, rows, at_cell, bucket)

    return DecodedRows(
        order=order,
        count=count,
        kind=kind,
        direction=direction,
        none_duration=jnp.where(kind == KIND_NONE, -1, 0).astype(jnp.int32),
        target_cell=target_cell,
        produced_type=produced_type,
        position=at_cell + source_filter.direction_delta(direction, width),
        has_position=source_filter.claims_position(kind),
        cost=jnp.where(kind == KIND_PRODUCE, game.unit_cost[rows[..., 5]], 0),
        survived=jnp.zeros_like(available),
    )


_COMMAND_FIELDS = (
    "kind",
    "direction",
    "none_duration",
    "target_cell",
    "produced_type",
    "request_order",
)


def _commands_of(decoded: list[DecodedRows], state: State, bucket: BucketSpec) -> _Rows:
    fields: dict[str, list[Array]] = {name: [] for name in _COMMAND_FIELDS}
    for player, rows in enumerate(decoded):
        for name, value in _player_commands(rows, state, bucket, player).items():
            fields[name].append(value)
    return fields


def _player_commands(rows: DecodedRows, state: State, bucket: BucketSpec, player: int) -> dict[str, Array]:
    capacity = bucket.per_player_capacity
    start = bucket.player_slot_start(player)
    owned = slice(start, start + capacity)

    def by_slot(values: Array, empty: int) -> Array:
        blank = jnp.full((capacity,), empty, jnp.int32)
        return blank.at[rows.order].set(jnp.where(rows.survived, values, empty))

    kind = by_slot(rows.kind, KIND_ABSENT)
    idle_filler = _idle_fillers(state, owned, kind)
    filler_rank = _filler_ranks(state, owned, idle_filler, rows.survived, capacity)
    return {
        "kind": jnp.where(idle_filler, KIND_NONE, kind),
        "direction": by_slot(rows.direction, ABSENT),
        "none_duration": jnp.where(idle_filler, 1, by_slot(rows.none_duration, 0)),
        "target_cell": by_slot(rows.target_cell, ABSENT),
        "produced_type": by_slot(rows.produced_type, ABSENT),
        "request_order": jnp.where(
            idle_filler,
            filler_rank,
            by_slot(jnp.cumsum(rows.survived.astype(jnp.int32)) - 1, ABSENT),
        ),
    }


def _idle_fillers(state: State, owned: slice, kind: Array) -> Array:
    submitted = kind != KIND_ABSENT
    available = state.active[owned] & (state.pending_kind[owned] == KIND_ABSENT)
    return available & ~submitted


def _filler_ranks(state: State, owned: slice, idle_filler: Array, survived: Array, capacity: int) -> Array:
    birth = jnp.where(idle_filler, state.birth[owned], jnp.iinfo(jnp.int32).max)
    by_birth = jnp.argsort(birth)
    return jnp.zeros((capacity,), jnp.int32).at[by_birth].set(
        jnp.arange(capacity, dtype=jnp.int32)
    ) + jnp.sum(survived.astype(jnp.int32))


_DECODE_TARGET = "nanorts_gridnet_decode"

_SHARED_BYTES = 48 * 1024


def _kernel_fits(bucket: BucketSpec) -> bool:
    return 4 * _warp_words(bucket) <= _SHARED_BYTES


def _warp_words(bucket: BucketSpec) -> int:

    def words(bits: int) -> int:
        return -(-bits // 32)

    capacity = bucket.per_player_capacity
    domain = source_filter.position_domain(bucket)
    return (
        2 * words(domain)
        + words(bucket.cell_count)
        + 2 * words(capacity)
        + 2 * capacity
    )


def _decode_in_kernel(actions: Array, state: State, game: GameConfig) -> _Rows:
    bucket = game.bucket
    legality = core.legal_commands(state, game)
    unit_cost = game.unit_cost.astype(jnp.int32)
    fields = kernels.call(
        _DECODE_TARGET,
        [
            actions,
            state.birth,
            state.pending_kind,
            state.cell,
            state.pending_direction,
            state.pending_produced_type,
            state.player_resources,
            legality.move_direction[0],
            legality.produce_direction[0],
            legality.produce_type[0],
            unit_cost,
        ],
        [
            jax.ShapeDtypeStruct((PLAYER_COUNT, bucket.per_player_capacity), jnp.int32)
            for _ in dataclasses.fields(Commands)
        ],
        attributes={
            "height": bucket.height,
            "width": bucket.width,
            "neutral": bucket.neutral_capacity,
            "capacity": bucket.per_player_capacity,
            "types": bucket.unit_type_capacity,
            "costs": unit_cost.shape[0],
            "side": bucket.attack_window_side,
            "warp_words": _warp_words(bucket),
        },
    )
    return {
        field.name: [value[player] for player in range(PLAYER_COUNT)]
        for field, value in zip(dataclasses.fields(Commands), fields, strict=True)
    }
