from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, lax

from nanorts.model import (
    ABSENT,
    KIND_ABSENT,
    KIND_NONE,
    ConflictStrategy,
    GameConfig,
    State,
    stored,
)

_CROSS_CYCLE_NONE_DURATION = -1

_NO_CAP = jnp.iinfo(jnp.int32).max

_CROSS_CYCLE_COUNT_STRIDE = 1 << 15

_PREFIX_RADIX = 8

_CONFLICT_EVENT = 0

_KERNEL_PRNG = "threefry2x32"


class Reservation(NamedTuple):
    claims_position: Array
    position: Array
    cost: Array
    player: Array


class RequestedAction(NamedTuple):
    slot: Array
    kind: Array
    none_duration: Array
    duration: Array
    reservation: Reservation


class _PendingEntries(NamedTuple):
    cross_cycle_conflict: Array
    same_cycle_conflict: Array
    is_same_cycle_conflict: Array
    duration: Array


class _ConflictCounts(NamedTuple):
    same_cycle_before: Array
    cross_cycle_before: Array
    same_cycle_total: Array
    any_cross_cycle: Array


class _SettledDurations(NamedTuple):
    pending: Array
    request_cancelled: Array
    request: Array


def resolve(key: Array, state: State, game: GameConfig, pending_ranking: Array, request: RequestedAction, existing: Reservation, existing_duration: Array) -> tuple[State, Array, Array]:
    _require_packable_conflict_counts(game.bucket.entity_capacity)
    conflicts = (state.pending_kind != KIND_ABSENT) & _reservations_conflict(
        existing, request.reservation, state.player_resources
    )
    cross_cycle = state.pending_issued_at != state.time
    draws = draws_for(key, state.time, state.birth[request.slot], game)
    entries = _in_pending_order(
        conflicts, cross_cycle, existing_duration, pending_ranking
    )
    counts = _running_conflict_counts(entries)
    cancels_request, cancels_pending = _cancelled_sides(
        counts.same_cycle_before, state.alternating_conflict_count, draws, game
    )
    durations = _settle_durations(
        entries,
        counts,
        entries.is_same_cycle_conflict & cancels_request,
        request.duration,
    )
    cancelled, cancelled_duration = _back_in_slot_order(
        pending_ranking,
        durations.pending,
        entries.is_same_cycle_conflict,
        cancels_pending,
        game.bucket.entity_capacity,
    )
    resolved = _cancel_pending(
        state,
        cancelled,
        cancelled_duration,
        counts.same_cycle_total,
        game.bucket.conflict_strategy,
    )
    return (
        resolved,
        jnp.where(durations.request_cancelled, KIND_NONE, request.kind),
        jnp.where(
            durations.request_cancelled, durations.request, request.none_duration
        ),
    )


def _require_packable_conflict_counts(entity_capacity: int) -> None:
    if entity_capacity >= _CROSS_CYCLE_COUNT_STRIDE:
        raise ValueError(
            f"entity capacity {entity_capacity} exceeds the "
            f"{_CROSS_CYCLE_COUNT_STRIDE} the packed conflict counts are proven for"
        )


def _reservations_conflict(existing: Reservation, requested: Reservation, balance: Array) -> Array:
    shared_position = (
        existing.claims_position
        & requested.claims_position
        & (existing.position == requested.position)
    )
    existing_cost_of_same_player = jnp.where(
        existing.player == requested.player, existing.cost, 0
    )
    claimed = existing_cost_of_same_player + requested.cost
    over_budget = (
        (requested.cost != 0) & (claimed > 0) & (claimed > balance[requested.player])
    )
    return shared_position | over_budget


def _in_pending_order(conflicts: Array, cross_cycle: Array, existing_duration: Array, pending_ranking: Array) -> _PendingEntries:
    packed = (
        existing_duration * 4
        + (conflicts & ~cross_cycle).astype(jnp.int32) * 2
        + (conflicts & cross_cycle).astype(jnp.int32)
    )[pending_ranking]
    cross_cycle_conflict = packed & 1
    same_cycle_conflict = (packed >> 1) & 1
    return _PendingEntries(
        cross_cycle_conflict=cross_cycle_conflict,
        same_cycle_conflict=same_cycle_conflict,
        is_same_cycle_conflict=same_cycle_conflict != 0,
        duration=packed >> 2,
    )


def _running_conflict_counts(entries: _PendingEntries) -> _ConflictCounts:
    stride = _CROSS_CYCLE_COUNT_STRIDE
    counts = _inclusive_prefix(
        entries.same_cycle_conflict + entries.cross_cycle_conflict * stride,
        0,
        jnp.add,
    )
    return _ConflictCounts(
        same_cycle_before=counts % stride - entries.same_cycle_conflict,
        cross_cycle_before=(counts // stride - entries.cross_cycle_conflict) > 0,
        same_cycle_total=counts[-1] % stride,
        any_cross_cycle=counts[-1] >= stride,
    )


def _settle_durations(entries: _PendingEntries, counts: _ConflictCounts, cancels_request_at: Array, requested_duration: Array) -> _SettledDurations:
    caps = _inclusive_prefix(
        jnp.where(cancels_request_at, entries.duration, _NO_CAP),
        _NO_CAP,
        jnp.minimum,
    )
    request_duration_after = jnp.minimum(requested_duration, caps)
    request_duration_before = jnp.where(
        counts.cross_cycle_before,
        _CROSS_CYCLE_NONE_DURATION,
        lax.pad(request_duration_after, requested_duration, [(1, -1, 0)]),
    )
    pending_duration = jnp.minimum(entries.duration, request_duration_before)
    request_cancelled = counts.any_cross_cycle | (caps[-1] != _NO_CAP)
    final_request_duration = jnp.where(
        counts.any_cross_cycle,
        _CROSS_CYCLE_NONE_DURATION,
        request_duration_after[-1],
    )
    return _SettledDurations(
        pending=pending_duration,
        request_cancelled=request_cancelled,
        request=final_request_duration,
    )


def _back_in_slot_order(pending_ranking: Array, settled_duration: Array, is_same_cycle_conflict: Array, cancels_pending: Array, entity_capacity: int) -> tuple[Array, Array]:
    packed = (
        jnp.zeros(entity_capacity, jnp.int32)
        .at[pending_ranking]
        .set(
            settled_duration * 2
            + (is_same_cycle_conflict & cancels_pending).astype(jnp.int32)
        )
    )
    return (packed & 1) != 0, packed >> 1


def _cancel_pending(state: State, cancelled: Array, cancelled_duration: Array, same_cycle_conflicts: Array, strategy: ConflictStrategy) -> State:
    def cancelled_to(field: str, value: Array) -> Array:
        return jnp.where(cancelled, value, getattr(state, field))

    alternating_advance = (
        same_cycle_conflicts if strategy == ConflictStrategy.CANCEL_ALTERNATING else 0
    )
    return stored(
        state,
        pending_kind=cancelled_to("pending_kind", jnp.asarray(KIND_NONE)),
        pending_direction=cancelled_to("pending_direction", jnp.asarray(ABSENT)),
        pending_none_duration=cancelled_to("pending_none_duration", cancelled_duration),
        pending_target_cell=cancelled_to("pending_target_cell", jnp.asarray(ABSENT)),
        pending_produced_type=cancelled_to(
            "pending_produced_type", jnp.asarray(ABSENT)
        ),
        pending_completes_at=cancelled_to(
            "pending_completes_at", state.pending_issued_at + cancelled_duration
        ),
        alternating_conflict_count=state.alternating_conflict_count
        + alternating_advance,
    )


def _inclusive_prefix(values: Array, identity: int, combine: Callable[[Array, Array], Array]) -> Array:
    length = values.shape[0]
    stride = 1
    while stride < length:
        combined = values
        for copy in range(1, _PREFIX_RADIX):
            shift = copy * stride
            if shift >= length:
                break
            combined = combine(
                lax.pad(values, identity, [(shift, -shift, 0)]), combined
            )
        values = combined
        stride *= _PREFIX_RADIX
    return values


def _cancelled_sides(same_cycle_before: Array, alternating_count: Array, draws: Array, game: GameConfig) -> tuple[Array, Array]:
    match game.bucket.conflict_strategy:
        case ConflictStrategy.CANCEL_BOTH:
            both = jnp.ones_like(same_cycle_before, dtype=jnp.bool_)
            return both, both
        case ConflictStrategy.CANCEL_ALTERNATING:
            cancels_request = ((alternating_count + same_cycle_before) % 2) == 0
            return cancels_request, ~cancels_request
        case ConflictStrategy.CANCEL_RANDOM:
            cancels_request = draws[same_cycle_before] == 0
            return cancels_request, ~cancels_request


def draws_for(key: Array, time: Array, birth: Array, game: GameConfig) -> Array:
    if game.bucket.conflict_strategy != ConflictStrategy.CANCEL_RANDOM:
        return jnp.zeros((0,), jnp.int32)
    event_key = jax.random.fold_in(key, time)
    event_key = jax.random.fold_in(event_key, birth)
    event_key = jax.random.fold_in(event_key, _CONFLICT_EVENT)

    def draw(ordinal: Array) -> Array:
        return jax.random.randint(
            jax.random.fold_in(event_key, ordinal), (), 0, 2, dtype=jnp.int32
        )

    return jax.vmap(draw)(jnp.arange(game.bucket.entity_capacity, dtype=jnp.int32))


def kernel_draws_valid(key: Array) -> bool:
    return (
        jnp.issubdtype(key.dtype, jax.dtypes.prng_key)
        and str(jax.random.key_impl(key)) == _KERNEL_PRNG
        and jax.config.jax_default_prng_impl == _KERNEL_PRNG
        and bool(jax.config.jax_threefry_partitionable)
    )
