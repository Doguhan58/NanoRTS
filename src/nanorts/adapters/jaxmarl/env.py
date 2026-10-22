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
    BoardView,
    Commands,
    GameConfig,
    State,
    StepResult,
    Telemetry,
    board_view,
)
from nanorts.observations import SymbolicLayout


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

    final_observations: Observations | None

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
    return respond(environment_state, result, game, layout=layout, assets=assets)


def step_against(environment_state: EnvironmentState, player: Array | int, actions: Array, commands: Commands, game: GameConfig, *, layout: JaxMARLLayout, assets: rendering.RenderAssets | None = None) -> JaxMARLStep:
    key, state, merged = request_against(
        environment_state, player, actions, commands, game, layout=layout, assets=assets
    )
    result = core.step(key, state, merged, game)
    return respond(environment_state, result, game, layout=layout, assets=assets)


def request_against(environment_state: EnvironmentState, player: Array | int, actions: Array, commands: Commands, game: GameConfig, *, layout: JaxMARLLayout, assets: rendering.RenderAssets | None = None) -> tuple[Array, State, Commands]:
    drawing_table("jaxmarl.step_against", layout, game.bucket, assets)
    state = environment_state.core_state
    decoded = decode_actions(
        dict.fromkeys(layout.agents, actions), state, game, layout=layout
    )
    seated = (jnp.arange(PLAYER_COUNT) == player)[:, None]
    merged = jax.tree.map(
        lambda mine, theirs: jnp.where(seated, mine, theirs), decoded, commands
    )
    return episode_root(environment_state), state, merged


def respond(environment_state: EnvironmentState, result: StepResult, game: GameConfig, *, layout: JaxMARLLayout, assets: rendering.RenderAssets | None = None) -> JaxMARLStep:
    ending = functools.partial(_ending, game=game, layout=layout, assets=assets)
    return kernels.dispatch(
        environment_state,
        result,
        kernel=functools.partial(ending, in_place=True),
        program=functools.partial(ending, in_place=False),
    )


def _ending(environment_state: EnvironmentState, result: StepResult, *, game: GameConfig, layout: JaxMARLLayout, assets: rendering.RenderAssets | None, in_place: bool) -> JaxMARLStep:
    done = result.terminated | result.truncated
    ordered, observed = _reset_order(result.next_state, game, layout, assets, in_place)
    carry = advance(environment_state, result, game, ordered=ordered)
    observations, final = _published_observations(
        carry, result, done, observed, game, layout, assets
    )
    return JaxMARLStep(
        environment_state=carry,
        observations=observations,
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


def _reset_order(ending: State, game: GameConfig, layout: JaxMARLLayout, assets: rendering.RenderAssets | None, in_place: bool) -> tuple[list[Array] | None, Observations | None]:
    if not (in_place and layout.final_observation):
        return None, None
    if not _publishes_byte_feature_maps(layout):
        return _final_encode_fields(board_view(ending, game), layout), None
    final = observe(ending, game, layout=layout, assets=assets)
    return [observation.feature_maps for observation in final.values()], final


def _final_encode_fields(board: BoardView, layout: JaxMARLLayout) -> list[Array]:
    values = [
        board.owner,
        board.unit_type,
        board.hit_points,
        board.carried_resources,
        board.pending_kind,
    ]
    if layout.surface == "symbolic" and not layout.partial_observability:
        return values
    return [board.present, *values]


def _publishes_byte_feature_maps(layout: JaxMARLLayout) -> bool:
    return (
        isinstance(layout.observation, SymbolicLayout)
        and layout.observation.feature_maps_dtype.itemsize == 1
    )


def _published_observations(carry: EnvironmentState, result: StepResult, done: Array, observed: Observations | None, game: GameConfig, layout: JaxMARLLayout, assets: rendering.RenderAssets | None) -> tuple[Observations, Observations | None]:
    if not layout.final_observation:
        return observe(carry.core_state, game, layout=layout, assets=assets), None
    final = (
        observe(result.next_state, game, layout=layout, assets=assets)
        if observed is None
        else observed
    )
    fresh = observe(episode_start(carry, game), game, layout=layout, assets=assets)
    return select(done, fresh, final), final


def published_components(result: StepResult, *, layout: JaxMARLLayout) -> Array | None:
    return encode_components(result) if layout.reward_components else None


def transition(environment_state: EnvironmentState, actions: dict[str, Array], game: GameConfig, *, layout: JaxMARLLayout) -> StepResult:
    return core.step(*request(environment_state, actions, game, layout=layout), game)


def request(environment_state: EnvironmentState, actions: dict[str, Array], game: GameConfig, *, layout: JaxMARLLayout) -> tuple[Array, State, Commands]:
    state = environment_state.core_state
    commands = decode_actions(actions, state, game, layout=layout)
    return episode_root(environment_state), state, commands
