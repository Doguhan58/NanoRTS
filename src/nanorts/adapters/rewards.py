from __future__ import annotations

from typing import Final

import jax.numpy as jnp
from jax import Array

from nanorts.model import (
    PLAYER_COUNT,
    TYPE_BARRACKS,
    TYPE_BASE,
    TYPE_HEAVY,
    TYPE_LIGHT,
    TYPE_RANGED,
    TYPE_WORKER,
    GameConfig,
    StepResult,
)

REWARD_COMPONENTS: Final[tuple[str, ...]] = (
    "win_loss",
    "resource_gather",
    "produce_worker",
    "produce_building",
    "attack",
    "produce_combat_unit",
)


def encode_rewards(step_result: StepResult, game: GameConfig) -> tuple[Array, Array]:
    components = encode_components(step_result)

    state = step_result.next_state
    types = jnp.clip(state.unit_type, 0, game.bucket.unit_type_capacity - 1)
    stacked = jnp.any(
        state.active & game.unit_is_resource[types] & (state.carried_resources > 0)
    )
    done = jnp.broadcast_to(
        step_result.terminated | step_result.truncated, (PLAYER_COUNT,)
    )
    dones = jnp.zeros((PLAYER_COUNT, len(REWARD_COMPONENTS)), jnp.bool_)
    dones = dones.at[:, 0].set(done)
    dones = dones.at[:, 1].set(~stacked)
    return components, dones


def encode_components(step_result: StepResult) -> Array:
    state, events = step_result.next_state, step_result.events
    players = jnp.arange(PLAYER_COUNT, dtype=jnp.int32)

    win_loss = jnp.where(
        state.game_over, jnp.where(state.winner == players, 1.0, -1.0), 0.0
    )
    gather = events.admitted_harvest + events.admitted_return
    produce_worker = events.admitted_produce[:, TYPE_WORKER]
    produce_building = (
        events.admitted_produce[:, TYPE_BASE]
        + events.admitted_produce[:, TYPE_BARRACKS]
    )
    produce_combat = (
        events.admitted_produce[:, TYPE_LIGHT]
        + events.admitted_produce[:, TYPE_HEAVY]
        + events.admitted_produce[:, TYPE_RANGED]
    )
    attacked = events.admitted_attack_by_target_owner
    opponent = (players + 1) % PLAYER_COUNT
    attack = attacked[players, opponent + 1] - attacked[players, players + 1]

    return jnp.stack(
        [win_loss, gather, produce_worker, produce_building, attack, produce_combat],
        axis=-1,
    ).astype(jnp.float32)
