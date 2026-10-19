from __future__ import annotations

import dataclasses
import functools
import operator
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from nanorts.core import board, kinds, pending, reservations, residue
from nanorts.model import (
    ABSENT,
    KIND_ABSENT,
    KIND_ATTACK,
    KIND_HARVEST,
    KIND_NONE,
    KIND_PRODUCE,
    KIND_RETURN,
    PLAYER_COUNT,
    BucketSpec,
    Commands,
    GameConfig,
    State,
    StepEvents,
    empty,
    stored,
)


class OrderedRequests(NamedTuple):
    player: Array
    local_slot: Array
    slot: Array
    issued_count: Array
    issued: Array


class _Command(NamedTuple):
    kind: Array
    direction: Array
    none_duration: Array
    target_cell: Array
    produced_type: Array


def admit(key: Array, state: State, commands: Commands, game: GameConfig) -> tuple[State, StepEvents]:
    requests = _ordered_requests(state, commands, game)
    pending_ranking = pending.ranking(state, requests.slot, requests.issued)

    def admit_request(position: Array, carry: tuple[State, StepEvents]) -> tuple[State, StepEvents]:
        request_address = (
            requests.player[position],
            requests.local_slot[position],
            requests.slot[position],
        )
        return _admit_one(key, carry, request_address, pending_ranking, commands, game)

    return residue.visit_in_order(
        requests.issued_count,
        admit_request,
        (state, empty(StepEvents, game.bucket)),
    )


def _ordered_requests(state: State, commands: Commands, game: GameConfig) -> OrderedRequests:
    bucket = game.bucket
    capacity = bucket.per_player_capacity
    owned = slice(bucket.neutral_capacity, None)
    owned_positions = jnp.arange(PLAYER_COUNT * capacity, dtype=jnp.int32)
    active = state.active[owned]
    issued = active & (commands.kind.reshape(owned_positions.shape) != KIND_ABSENT)
    unit_order = jnp.where(active, state.birth[owned], jnp.iinfo(jnp.int32).max)
    request_key = _issued_player_rank_key(
        issued, commands.request_order.reshape(owned_positions.shape), bucket
    )
    _, _, admission_order = lax.sort(
        (request_key, unit_order, owned_positions), num_keys=3, is_stable=False
    )
    issued_count = jnp.sum(issued.astype(jnp.int32))
    return OrderedRequests(
        player=admission_order // capacity,
        local_slot=admission_order % capacity,
        slot=bucket.neutral_capacity + admission_order,
        issued_count=issued_count,
        issued=issued[admission_order],
    )


def _issued_player_rank_key(issued: Array, request_order: Array, bucket: BucketSpec) -> Array:
    rank_bits = max(1, (bucket.per_player_capacity - 1).bit_length())
    rank = jnp.clip(
        jnp.where(request_order == ABSENT, 0, request_order), 0, (1 << rank_bits) - 1
    )
    not_issued = (~issued).astype(jnp.int32)
    return (
        (not_issued << (rank_bits + 1))
        | (_player_of_owned_position(bucket) << rank_bits)
        | rank
    )


def _player_of_owned_position(bucket: BucketSpec) -> np.ndarray:
    capacity = bucket.per_player_capacity
    return np.arange(PLAYER_COUNT * capacity, dtype=np.int32) // capacity


def _admit_one(key: Array, carry: tuple[State, StepEvents], request_address: tuple[Array, Array, Array], pending_ranking: Array, commands: Commands, game: GameConfig) -> tuple[State, StepEvents]:
    state, events = carry
    player, local_slot, slot = request_address
    command = _command_at(commands, player, local_slot)
    issued = state.active[slot] & (command.kind != KIND_ABSENT)
    requested_duration = _action_duration(
        game,
        state.unit_type[slot],
        command.kind,
        command.none_duration,
        command.produced_type,
    )
    kind, none_duration = _none_if_illegal(
        state, game, slot, command, requested_duration
    )
    kind, none_duration = _none_if_blocked(
        state, game, slot, command.direction, kind, none_duration, requested_duration
    )
    events = _count_admitted(
        events,
        state,
        game,
        player,
        kind,
        command.produced_type,
        command.target_cell,
        issued,
    )
    state, kind, none_duration = _resolve_conflicts(
        key,
        state,
        game,
        pending_ranking,
        reservations.RequestedAction(
            slot=slot,
            kind=kind,
            none_duration=none_duration,
            duration=requested_duration,
            reservation=_requested_reservation(
                state, game, slot, kind, command, issued
            ),
        ),
    )
    duration = jnp.where(kind == KIND_NONE, none_duration, requested_duration)
    admitted = _assign_pending_action(
        state, slot, issued, command, kind, none_duration, duration
    )
    return admitted, events


def _command_at(commands: Commands, player: Array, local_slot: Array) -> _Command:
    return _Command(
        kind=commands.kind[player, local_slot],
        direction=commands.direction[player, local_slot],
        none_duration=commands.none_duration[player, local_slot],
        target_cell=commands.target_cell[player, local_slot],
        produced_type=commands.produced_type[player, local_slot],
    )


def _none_if_illegal(state: State, game: GameConfig, slot: Array, command: _Command, requested_duration: Array) -> tuple[Array, Array]:
    legal = _can_execute_action(
        state,
        game,
        slot,
        command.kind,
        command.direction,
        command.target_cell,
        command.produced_type,
    )
    return (
        jnp.where(legal, command.kind, KIND_NONE),
        jnp.where(legal, command.none_duration, requested_duration),
    )


def _none_if_blocked(state: State, game: GameConfig, slot: Array, direction: Array, kind: Array, none_duration: Array, requested_duration: Array) -> tuple[Array, Array]:
    claims_position = _claims_position(kind)
    reserved = board.reserved_position(state.cell[slot], direction, game)
    blocked = claims_position & board.position_is_blocked(state, game, reserved)
    return (
        jnp.where(blocked, KIND_NONE, kind),
        jnp.where(blocked, requested_duration, none_duration),
    )


def _resolve_conflicts(key: Array, state: State, game: GameConfig, pending_ranking: Array, request: reservations.RequestedAction) -> tuple[State, Array, Array]:
    return reservations.resolve(
        key,
        state,
        game,
        pending_ranking,
        request,
        _pending_reservations(state, game),
        _pending_durations(state, game),
    )


def _requested_reservation(state: State, game: GameConfig, slot: Array, kind: Array, command: _Command, issued: Array) -> reservations.Reservation:
    return _reservation(
        game,
        state.cell[slot],
        state.owner[slot],
        kind,
        command.direction,
        command.produced_type,
        issued,
    )


def _pending_reservations(state: State, game: GameConfig) -> reservations.Reservation:
    return _reservation(
        game,
        state.cell,
        state.owner,
        state.pending_kind,
        state.pending_direction,
        state.pending_produced_type,
        state.pending_kind != KIND_ABSENT,
    )


def _pending_durations(state: State, game: GameConfig) -> Array:
    return _action_duration(
        game,
        state.unit_type,
        state.pending_kind,
        state.pending_none_duration,
        state.pending_produced_type,
    )


def _assign_pending_action(state: State, slot: Array, issued: Array, command: _Command, kind: Array, none_duration: Array, duration: Array) -> State:
    held_order = state.pending_order[slot]
    appends = issued & (held_order == ABSENT)
    order = jnp.where(held_order == ABSENT, state.next_pending_order, held_order)

    def assigned(field: str, value: Array) -> Array:
        return residue.masked_set(getattr(state, field), slot, issued, value)

    def assigned_if_kept(parameter: str, value: Array, empty: int) -> Array:
        return assigned(
            f"pending_{parameter}",
            jnp.where(_keeps(parameter, kind), value, empty),
        )

    return stored(
        state,
        pending_kind=assigned("pending_kind", kind),
        pending_direction=assigned_if_kept("direction", command.direction, ABSENT),
        pending_none_duration=assigned_if_kept("none_duration", none_duration, 0),
        pending_target_cell=assigned_if_kept(
            "target_cell", command.target_cell, ABSENT
        ),
        pending_produced_type=assigned_if_kept(
            "produced_type", command.produced_type, ABSENT
        ),
        pending_issued_at=assigned("pending_issued_at", state.time),
        pending_completes_at=assigned("pending_completes_at", state.time + duration),
        pending_order=assigned("pending_order", order),
        next_pending_order=state.next_pending_order + appends.astype(jnp.int32),
    )


def _count_admitted(events: StepEvents, state: State, game: GameConfig, player: Array, kind: Array, produced_type: Array, target_cell: Array, issued: Array) -> StepEvents:
    bucket = game.bucket
    types = bucket.unit_type_capacity

    def admitted(action_kind: int) -> Array:
        return (issued & (kind == action_kind)).astype(jnp.int32)

    target_owner = _owner_at(state, bucket, target_cell)
    player_mask = jnp.arange(PLAYER_COUNT, dtype=jnp.int32) == player
    produced_type_mask = jnp.arange(types, dtype=jnp.int32) == jnp.clip(
        produced_type, 0, types - 1
    )
    target_owner_mask = (
        jnp.arange(PLAYER_COUNT + 1, dtype=jnp.int32) == target_owner + 1
    )
    return dataclasses.replace(
        events,
        admitted_harvest=events.admitted_harvest + player_mask * admitted(KIND_HARVEST),
        admitted_return=events.admitted_return + player_mask * admitted(KIND_RETURN),
        admitted_produce=events.admitted_produce
        + (player_mask[:, None] & produced_type_mask[None, :]) * admitted(KIND_PRODUCE),
        admitted_attack_by_target_owner=events.admitted_attack_by_target_owner
        + (player_mask[:, None] & target_owner_mask[None, :]) * admitted(KIND_ATTACK),
    )


def _owner_at(state: State, bucket: BucketSpec, cell: Array) -> Array:
    occupant = state.entity_at.reshape(bucket.cell_count)[
        jnp.clip(cell, 0, bucket.cell_count - 1)
    ]
    return jnp.where(
        occupant == ABSENT,
        ABSENT,
        state.owner[jnp.clip(occupant, 0, bucket.entity_capacity - 1)],
    )


def _action_duration(game: GameConfig, actor_type: Array, kind: Array, none_duration: Array, produced_type: Array) -> Array:
    actor = jnp.clip(actor_type, 0, game.bucket.unit_type_capacity - 1)
    duration = jnp.zeros_like(kind)
    for action_kind in kinds.KINDS:
        duration = jnp.where(
            kind == action_kind.code,
            action_kind.duration(game, actor, none_duration, produced_type),
            duration,
        )
    return duration


def _can_execute_action(state: State, game: GameConfig, slot: Array, kind: Array, direction: Array, target_cell: Array, produced_type: Array) -> Array:
    request = kinds.Request.of(state, game, slot, direction, target_cell, produced_type)
    legal = jnp.zeros_like(kind, dtype=jnp.bool_)
    for action_kind in kinds.KINDS:
        legal = jnp.where(
            kind == action_kind.code, action_kind.is_valid(request), legal
        )
    return legal


def _reservation(game: GameConfig, cell: Array, owner: Array, kind: Array, direction: Array, produced_type: Array, has_action: Array) -> reservations.Reservation:
    cost = jnp.asarray(0, jnp.int32)
    for action_kind in kinds.KINDS:
        if action_kind.cost is not None:
            cost = jnp.where(
                has_action & (kind == action_kind.code),
                action_kind.cost(game, produced_type),
                cost,
            )
    return reservations.Reservation(
        claims_position=has_action & _claims_position(kind),
        position=board.reserved_position(cell, direction, game),
        cost=cost,
        player=jnp.clip(owner, 0, PLAYER_COUNT - 1),
    )


def _claims_position(kind: Array) -> Array:
    return _is_one_of(kind, [k for k in kinds.KINDS if k.claims_position])


def _keeps(parameter: str, kind: Array) -> Array:
    return _is_one_of(kind, [k for k in kinds.KINDS if parameter in k.payload])


def _is_one_of(kind: Array, candidates: list[kinds.ActionKind]) -> Array:
    if not candidates:
        return jnp.zeros_like(kind, dtype=jnp.bool_)
    return functools.reduce(operator.or_, [kind == k.code for k in candidates])
