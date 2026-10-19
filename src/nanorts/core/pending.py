from __future__ import annotations

import dataclasses
from typing import Any, Final

import jax.numpy as jnp
from jax import Array, lax

from nanorts.model import (
    ABSENT,
    KIND_ABSENT,
    GameConfig,
    LeafField,
    State,
    empty,
    model_fields,
)

_PENDING_FIELDS: Final[tuple[str, ...]] = tuple(
    field.name
    for field in model_fields(State)
    if isinstance(field, LeafField) and field.name.startswith("pending_")
)

_RANKS_LAST = jnp.iinfo(jnp.int32).max


def ranking(state: State, slots: Array, issued: Array) -> Array:
    neutral_count = state.pending_order.shape[0] - slots.shape[0]
    keys = _owned_pending_order_after_appends(state, slots, issued, neutral_count)
    owned_ranking = _ascending_positions(keys)
    return jnp.concatenate(
        [jnp.arange(neutral_count, dtype=jnp.int32), neutral_count + owned_ranking]
    )


def _owned_pending_order_after_appends(state: State, slots: Array, issued: Array, neutral_count: int) -> Array:
    held_order = state.pending_order[neutral_count:]
    owned_slot = slots - neutral_count
    appends = issued & (held_order[owned_slot] == ABSENT)
    appended = appends.astype(jnp.int32)
    appended_order = state.next_pending_order + jnp.cumsum(appended) - appended
    keys = jnp.where(held_order == ABSENT, _RANKS_LAST, held_order)
    return keys.at[owned_slot].set(jnp.where(appends, appended_order, keys[owned_slot]))


def select(state: State) -> tuple[State, Array]:
    next_time = state.time + 1
    completes = (state.pending_kind != KIND_ABSENT) & (
        state.pending_completes_at <= next_time
    )
    return dataclasses.replace(state, time=next_time), completes


def batch_order(state: State, acting: Array, game: GameConfig) -> tuple[Array, Array]:
    neutral_count = game.bucket.neutral_capacity
    acting_owned = acting[neutral_count:]
    keys = jnp.where(acting_owned, state.pending_order[neutral_count:], _RANKS_LAST)
    member_order = _ascending_positions(keys)
    return neutral_count + member_order, jnp.sum(acting_owned.astype(jnp.int32))


def _ascending_positions(keys: Array) -> Array:
    _, positions = lax.sort(
        (keys, jnp.arange(keys.shape[0], dtype=jnp.int32)), num_keys=2, is_stable=False
    )
    return positions


def clear(state: State, removed: Array, game: GameConfig) -> State:
    canonical = empty(State, game.bucket)
    cleared: dict[str, Any] = {
        name: jnp.where(removed, getattr(canonical, name), getattr(state, name))
        for name in _PENDING_FIELDS
    }
    return dataclasses.replace(state, **cleared)
