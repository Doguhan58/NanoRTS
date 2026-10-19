from __future__ import annotations

import dataclasses

import jax.numpy as jnp
from jax import Array

from nanorts.model import ABSENT, PLAYER_COUNT, GameConfig, State


def decide(state: State, game: GameConfig) -> State:
    surviving = _players_with_active_units(state, game)
    survivor_count = jnp.sum(surviving.astype(jnp.int32))
    first_survivor = jnp.argmax(surviving).astype(jnp.int32)
    return dataclasses.replace(
        state,
        game_over=survivor_count <= 1,
        winner=jnp.where(survivor_count == 1, first_survivor, ABSENT).astype(jnp.int32),
    )


def _players_with_active_units(state: State, game: GameConfig) -> Array:
    bucket = game.bucket
    owned_active = state.active[bucket.neutral_capacity :]
    return owned_active.reshape(PLAYER_COUNT, bucket.per_player_capacity).any(axis=1)


def truncates(state: State, game: GameConfig) -> Array:
    return ~state.game_over & (state.time >= game.max_cycles)


def sparse_reward(state: State, game: GameConfig) -> Array:
    players = jnp.arange(PLAYER_COUNT, dtype=jnp.int32)
    ended = state.game_over | truncates(state, game)
    return jnp.where(
        ended, jnp.where(players == state.winner, 1.0, -1.0), 0.0
    ).astype(jnp.float32)
