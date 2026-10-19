from __future__ import annotations

import dataclasses
from typing import Any

import jax.numpy as jnp
from jax import Array

from nanorts.model import PER_SLOT, GameConfig, LeafField, State, empty, model_fields


def lowest_free_slot(state: State, player: Array, slots: Array, game: GameConfig) -> tuple[Array, Array]:
    bucket = game.bucket
    block_start = bucket.neutral_capacity + player * bucket.per_player_capacity
    in_block = (slots >= block_start) & (
        slots < block_start + bucket.per_player_capacity
    )
    free = in_block & ~state.active
    return jnp.any(free), jnp.argmax(free).astype(jnp.int32)


def remove(state: State, removed: Array, game: GameConfig) -> State:
    canonical = empty(State, game.bucket)
    cleared: dict[str, Any] = {
        field.name: jnp.where(
            removed, getattr(canonical, field.name), getattr(state, field.name)
        )
        for field in model_fields(State)
        if isinstance(field, LeafField) and field.spec.shape is PER_SLOT
    }
    return dataclasses.replace(state, **cleared)
