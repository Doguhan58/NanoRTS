from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, Final

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from nanorts import kernels
from nanorts.core import outcome, unit_tables
from nanorts.model import (
    ATTACK_TARGET_OWNER_COUNT,
    CELL_INDEX,
    PER_SLOT,
    PLAYER_COUNT,
    SLOT_INDEX,
    BucketSpec,
    Commands,
    ConflictStrategy,
    GameConfig,
    LeafField,
    State,
    StepEvents,
    StepResult,
    empty,
    model_fields,
)

_KERNEL_TARGET = "nanorts_step"

_GROUPS_TARGET = "nanorts_step_groups"

_STRATEGY_CODES = {
    ConflictStrategy.CANCEL_BOTH: 0,
    ConflictStrategy.CANCEL_RANDOM: 1,
    ConflictStrategy.CANCEL_ALTERNATING: 2,
}

_PER_SLOT_LEAVES: Final[tuple[str, ...]] = tuple(
    field.name
    for field in model_fields(State)
    if isinstance(field, LeafField) and field.spec.shape is PER_SLOT
)

_RECORD_FLAGS: Final[dict[str, int]] = {
    "hit": 1 << 0,
    "death": 1 << 1,
    "harvest": 1 << 2,
    "return": 1 << 4,
    "spawn": 1 << 5,
}
_HARVEST_DEPLETED_FLAG = 1 << 3

_INT32_COLUMNS: Final = (
    "actor_birth",
    "target_birth",
    "amount",
    "balance",
    "spawn_birth",
    "death_birth",
)
_INDEX_COLUMNS: Final = ("target_slot", "spawn_slot", "spawn_cell", "death_slot")
_INT8_COLUMNS: Final = (
    "flags",
    "actor_owner",
    "actor_unit_type",
    "produced_type",
    "death_owner",
    "death_unit_type",
)

_COLUMN_OF_EVENT_FIELD: Final[dict[str, str]] = {
    "hit_target_slot": "target_slot",
    "hit_target_birth": "target_birth",
    "hit_damage": "amount",
    "death_slot": "death_slot",
    "death_birth": "death_birth",
    "death_owner": "death_owner",
    "death_unit_type": "death_unit_type",
    "harvest_target_slot": "target_slot",
    "harvest_target_birth": "target_birth",
    "harvest_amount": "amount",
    "return_target_slot": "target_slot",
    "return_target_birth": "target_birth",
    "return_amount": "amount",
    "return_balance": "balance",
    "spawn_produced_type": "produced_type",
    "spawn_slot": "spawn_slot",
    "spawn_birth": "spawn_birth",
    "spawn_cell": "spawn_cell",
    "spawn_balance": "balance",
}

_PACKED_RETAINED_ACTOR_BYTES: Final[dict[int, int]] = {2: 20, 4: 24}

_SHARED_BYTES_PER_BLOCK = 48 * 1024

_ALIASED_LEAVES: Final[tuple[str, ...]] = (*_PER_SLOT_LEAVES, "entity_at")

_WRITTEN_LEAVES: Final[tuple[str, ...]] = (
    *_ALIASED_LEAVES,
    "time",
    "player_resources",
    "next_birth",
    "next_pending_order",
    "alternating_conflict_count",
    "game_over",
    "winner",
)


def fits(bucket: BucketSpec) -> bool:
    owned_slots = PLAYER_COUNT * bucket.per_player_capacity
    slot_bytes = SLOT_INDEX.resolve(bucket).itemsize
    cell_bytes = CELL_INDEX.resolve(bucket).itemsize
    per_owned_slot = 2 * slot_bytes + _PACKED_RETAINED_ACTOR_BYTES[cell_bytes]
    return owned_slots * per_owned_slot <= _SHARED_BYTES_PER_BLOCK


def on_device(key: Array, state: State, commands: Commands, game: GameConfig) -> StepResult:
    bucket = game.bucket
    leaves = _written_leaves(state)
    read_operands = _read_operands(key, commands, game)
    outputs = kernels.call(
        _KERNEL_TARGET,
        [*read_operands, *leaves],
        _output_shapes(leaves, bucket),
        aliases={len(read_operands) + index: index for index in range(len(_ALIASED_LEAVES))},
        attributes={"strategy": _STRATEGY_CODES[bucket.conflict_strategy]},
    )
    return _result(state, outputs, game)


def on_device_groups(requests: Sequence[tuple[Array, State, Commands, GameConfig]]) -> list[StepResult]:
    operands: list[Array] = []
    results: list[jax.ShapeDtypeStruct] = []
    aliases: dict[int, int] = {}
    spans = []
    for key, state, commands, game in requests:
        leaves = _written_leaves(state)
        read_operands = _group_read_operands(key, commands, game)
        aliases.update(
            {
                len(operands) + len(read_operands) + index: len(results) + index
                for index in range(len(_ALIASED_LEAVES))
            }
        )
        operands.extend([*read_operands, *leaves])
        lanes = state.time.shape[0]
        rows = _output_shapes(
            [jax.ShapeDtypeStruct(leaf.shape[1:], leaf.dtype) for leaf in leaves],
            game.bucket,
        )
        spans.append(slice(len(results), len(results) + len(rows)))
        results.extend(
            jax.ShapeDtypeStruct((lanes, *row.shape), row.dtype) for row in rows
        )
    outputs = jax.ffi.ffi_call(
        _GROUPS_TARGET,
        results,
        input_output_aliases=aliases,
        vmap_method="sequential",
    )(
        *operands,
        build_id=kernels.build_id(),
        strategy=_STRATEGY_CODES[requests[0][3].bucket.conflict_strategy],
        groups=len(requests),
    )
    return [
        jax.vmap(_result, in_axes=(0, 0, None))(state, outputs[span], game)
        for (_, state, _, game), span in zip(requests, spans, strict=True)
    ]


def _written_leaves(state: State) -> list[Array]:
    return [
        *(getattr(state, name) for name in _WRITTEN_LEAVES),
        state.telemetry.spawn_overflow,
    ]


def _result(state: State, outputs: Sequence[Array], game: GameConfig) -> StepResult:
    written = len(_WRITTEN_LEAVES) + 1
    acted = _state_from_outputs(state, outputs[:written])
    events = _events_from_outputs(outputs[written:], game.bucket)
    return StepResult(
        next_state=acted,
        terminated=acted.game_over,
        truncated=outcome.truncates(acted, game),
        reward=outcome.sparse_reward(acted, game),
        events=events,
    )


def _read_operands(key: Array, commands: Commands, game: GameConfig) -> list[Array]:
    draws_randomly = game.bucket.conflict_strategy == ConflictStrategy.CANCEL_RANDOM
    random_root = (
        jax.random.key_data(key) if draws_randomly else jnp.zeros(2, jnp.uint32)
    )
    return [
        random_root,
        commands.kind,
        commands.direction,
        commands.none_duration,
        commands.target_cell,
        commands.produced_type,
        commands.request_order,
        game.terrain,
        jnp.asarray(unit_tables.unit_table(game)),
        jnp.asarray(unit_tables.unit_timings(game)),
    ]


def _group_read_operands(keys: Array, commands: Commands, game: GameConfig) -> list[Array]:
    random_roots, *command_rows, terrain, table, timings = _read_operands(
        keys, commands, game
    )
    return [
        random_roots if random_roots.ndim == 2 else random_roots[None],
        *command_rows,
        terrain[None],
        table[None],
        timings[None],
    ]


def _output_shapes(leaves: Sequence[Array], bucket: BucketSpec) -> list[jax.ShapeDtypeStruct]:
    entities = bucket.entity_capacity
    return [
        *(jax.ShapeDtypeStruct(leaf.shape, leaf.dtype) for leaf in leaves),
        jax.ShapeDtypeStruct((PLAYER_COUNT,), jnp.int32),
        jax.ShapeDtypeStruct((PLAYER_COUNT,), jnp.int32),
        jax.ShapeDtypeStruct((PLAYER_COUNT, bucket.unit_type_capacity), jnp.int32),
        jax.ShapeDtypeStruct((PLAYER_COUNT, ATTACK_TARGET_OWNER_COUNT), jnp.int32),
        jax.ShapeDtypeStruct((entities,), jnp.bool_),
        jax.ShapeDtypeStruct((entities, len(_INT32_COLUMNS)), jnp.int32),
        jax.ShapeDtypeStruct((entities, len(_INDEX_COLUMNS)), _index_dtype(bucket)),
        jax.ShapeDtypeStruct((entities, len(_INT8_COLUMNS)), jnp.int8),
    ]


def _state_from_outputs(state: State, written_leaves: Sequence[Array]) -> State:
    *written, spawn_overflow = written_leaves
    return dataclasses.replace(
        state,
        **dict(zip(_WRITTEN_LEAVES, written, strict=True)),
        telemetry=dataclasses.replace(state.telemetry, spawn_overflow=spawn_overflow),
    )


def _events_from_outputs(outputs: Sequence[Array], bucket: BucketSpec) -> StepEvents:
    (
        admitted_harvest,
        admitted_return,
        admitted_produce,
        admitted_attack,
        is_member,
        int32_rows,
        index_rows,
        int8_rows,
    ) = outputs
    return dataclasses.replace(
        empty(StepEvents, bucket),
        admitted_harvest=admitted_harvest,
        admitted_return=admitted_return,
        admitted_produce=admitted_produce,
        admitted_attack_by_target_owner=admitted_attack,
        **_record_fields(is_member, int32_rows, index_rows, int8_rows, bucket),
    )


def _index_dtype(bucket: BucketSpec) -> np.dtype[Any]:
    int16 = np.dtype(np.int16)
    if SLOT_INDEX.resolve(bucket) == int16 and CELL_INDEX.resolve(bucket) == int16:
        return int16
    return np.dtype(np.int32)


def _record_fields(is_member: Array, int32_rows: Array, index_rows: Array, int8_rows: Array, bucket: BucketSpec) -> dict[str, Any]:
    columns = {
        **{name: int32_rows[:, k] for k, name in enumerate(_INT32_COLUMNS)},
        **{name: index_rows[:, k] for k, name in enumerate(_INDEX_COLUMNS)},
        **{name: int8_rows[:, k] for k, name in enumerate(_INT8_COLUMNS)},
    }
    flags = columns["flags"].astype(jnp.int32)
    return {
        field.name: _record_field(field, columns, flags, is_member, bucket)
        for field in model_fields(StepEvents)
        if isinstance(field, LeafField) and field.spec.shape is PER_SLOT
    }


def _record_field(field: LeafField, columns: dict[str, Array], flags: Array, is_member: Array, bucket: BucketSpec) -> Array:
    record, _, fact = field.name.partition("_")
    valid = is_member & ((flags & _RECORD_FLAGS[record]) != 0)
    if fact == "valid":
        return valid
    if field.name == "harvest_depleted":
        return valid & ((flags & _HARVEST_DEPLETED_FLAG) != 0)
    fill = field.spec.fill
    assert fill is not None, f"StepEvents.{field.name} declares a fill"
    column = columns[
        fact if fact.startswith("actor_") else _COLUMN_OF_EVENT_FIELD[field.name]
    ]
    return jnp.where(valid, column.astype(jnp.int32), fill).astype(
        field.spec.dtype_under(bucket)
    )
