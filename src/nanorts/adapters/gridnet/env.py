from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import Array

from nanorts import core, kernels
from nanorts.adapters.gridnet.actions import decode_actions
from nanorts.adapters.gridnet.layout import GridNetLayout
from nanorts.adapters.gridnet.observation import GridNetObservation, observe, planes
from nanorts.adapters.lifecycle import (
    EnvironmentState,
    advance,
    episode_root,
    initial,
    spawn_overflowed,
)
from nanorts.adapters.rewards import encode_rewards
from nanorts.model import Commands, GameConfig, StepResult, Telemetry, board_view


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GridNetStep:
    environment_state: EnvironmentState

    observation: GridNetObservation

    reward_components: Array

    reward_dones: Array

    terminated: Array

    truncated: Array

    done: Array

    winner: Array

    telemetry: Telemetry

    spawn_overflowed: Array

    final_observation: Array

    final_valid: Array


def reset(environment_root: Array, game: GameConfig, *, layout: GridNetLayout) -> tuple[EnvironmentState, GridNetObservation]:
    environment_state = initial(environment_root, game)
    return environment_state, observe(environment_state.core_state, game, layout=layout)


def step(environment_state: EnvironmentState, actions: Array, game: GameConfig, *, layout: GridNetLayout) -> GridNetStep:
    state = environment_state.core_state
    commands = decode_actions(actions, state, game, layout=layout)
    result = core.step(episode_root(environment_state), state, commands, game)
    return _respond(environment_state, result, game, layout)


def step_against(environment_state: EnvironmentState, actions: Array, commands: Commands, game: GameConfig, *, layout: GridNetLayout) -> GridNetStep:
    state = environment_state.core_state
    decoded = decode_actions(
        jnp.stack([actions, jnp.zeros_like(actions)]), state, game, layout=layout
    )
    merged = jax.tree.map(
        lambda learner, given: jnp.stack([learner[0], given[1]]), decoded, commands
    )
    result = core.step(episode_root(environment_state), state, merged, game)
    return _respond(environment_state, result, game, layout)


def _respond(environment_state: EnvironmentState, result: StepResult, game: GameConfig, layout: GridNetLayout) -> GridNetStep:
    reward_components, reward_dones = encode_rewards(result, game)
    ending = functools.partial(_ending, game=game, layout=layout)
    return kernels.dispatch(
        environment_state,
        result,
        reward_components,
        reward_dones,
        kernel=functools.partial(ending, in_place=True),
        program=functools.partial(ending, in_place=False),
    )


def _ending(environment_state: EnvironmentState, result: StepResult, reward_components: Array, reward_dones: Array, *, game: GameConfig, layout: GridNetLayout, in_place: bool) -> GridNetStep:
    done = result.terminated | result.truncated
    ordered = board_view(result.next_state, game) if in_place else None
    carry = advance(environment_state, result, game, ordered=ordered)
    return GridNetStep(
        environment_state=carry,
        observation=observe(carry.core_state, game, layout=layout),
        reward_components=reward_components,
        reward_dones=reward_dones,
        terminated=result.terminated,
        truncated=result.truncated,
        done=done,
        winner=result.next_state.winner,
        telemetry=result.next_state.telemetry,
        spawn_overflowed=spawn_overflowed(result.next_state),
        final_observation=planes(result.next_state, game, layout=layout),
        final_valid=done,
    )
