from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import Array

from nanorts import core, kernels, rendering
from nanorts.adapters.jaxmarl.actions import decode_actions
from nanorts.adapters.jaxmarl.layout import JaxMARLLayout, drawing_table
from nanorts.adapters.jaxmarl.observation import Observations, observe
from nanorts.adapters.lifecycle import (
    EnvironmentState,
    advance,
    episode_root,
    episode_start,
    initial,
    select,
    spawn_overflowed,
)
from nanorts.adapters.rewards import encode_components
from nanorts.model import (
    PLAYER_COUNT,
    Commands,
    GameConfig,
    StepResult,
    Telemetry,
    board_view,
)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class JaxMARLStep:
    environment_state: EnvironmentState
    observations: Observations

    reward: Array

    reward_components: Array | None

    terminated: Array

    truncated: Array

    done: Array

    winner: Array

    telemetry: Telemetry

    spawn_overflowed: Array

    final_observations: Observations

    final_valid: Array


def reset(environment_root: Array, game: GameConfig, *, layout: JaxMARLLayout, assets: rendering.RenderAssets | None = None) -> tuple[EnvironmentState, Observations]:
    drawing_table("jaxmarl.reset", layout, game.bucket, assets)
    environment_state = initial(environment_root, game)
    return environment_state, observe(
        environment_state.core_state, game, layout=layout, assets=assets
    )


def step(environment_state: EnvironmentState, actions: dict[str, Array], game: GameConfig, *, layout: JaxMARLLayout, assets: rendering.RenderAssets | None = None) -> JaxMARLStep:
    drawing_table("jaxmarl.step", layout, game.bucket, assets)
    result = transition(environment_state, actions, game, layout=layout)
    return _respond(environment_state, result, game, layout, assets)


def step_against(environment_state: EnvironmentState, player: Array | int, actions: Array, commands: Commands, game: GameConfig, *, layout: JaxMARLLayout, assets: rendering.RenderAssets | None = None) -> JaxMARLStep:
    drawing_table("jaxmarl.step_against", layout, game.bucket, assets)
    state = environment_state.core_state
    decoded = decode_actions(
        dict.fromkeys(layout.agents, actions), state, game, layout=layout
    )
    seated = (jnp.arange(PLAYER_COUNT) == player)[:, None]
    merged = jax.tree.map(
        lambda mine, theirs: jnp.where(seated, mine, theirs), decoded, commands
    )
    result = core.step(episode_root(environment_state), state, merged, game)
    return _respond(environment_state, result, game, layout, assets)


def _respond(environment_state: EnvironmentState, result: StepResult, game: GameConfig, layout: JaxMARLLayout, assets: rendering.RenderAssets | None) -> JaxMARLStep:
    ending = functools.partial(_ending, game=game, layout=layout, assets=assets)
    return kernels.dispatch(
        environment_state,
        result,
        kernel=functools.partial(ending, in_place=True),
        program=functools.partial(ending, in_place=False),
    )


def _ending(environment_state: EnvironmentState, result: StepResult, *, game: GameConfig, layout: JaxMARLLayout, assets: rendering.RenderAssets | None, in_place: bool) -> JaxMARLStep:
    done = result.terminated | result.truncated
    carry = advance(
        environment_state,
        result,
        game,
        ordered=board_view(result.next_state, game) if in_place else None,
    )
    final = observe(result.next_state, game, layout=layout, assets=assets)
    fresh = observe(episode_start(carry, game), game, layout=layout, assets=assets)
    return JaxMARLStep(
        environment_state=carry,
        observations=select(done, fresh, final),
        reward=result.reward,
        reward_components=published_components(result, layout=layout),
        terminated=result.terminated,
        truncated=result.truncated,
        done=done,
        winner=result.next_state.winner,
        telemetry=result.next_state.telemetry,
        spawn_overflowed=spawn_overflowed(result.next_state),
        final_observations=final,
        final_valid=done,
    )


def published_components(result: StepResult, *, layout: JaxMARLLayout) -> Array | None:
    return encode_components(result) if layout.reward_components else None


def transition(environment_state: EnvironmentState, actions: dict[str, Array], game: GameConfig, *, layout: JaxMARLLayout) -> StepResult:
    state = environment_state.core_state
    commands = decode_actions(actions, state, game, layout=layout)
    return core.step(episode_root(environment_state), state, commands, game)
