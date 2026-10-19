from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from nanorts import core, kernels
from nanorts.model import (
    FIELD,
    PER_SLOT,
    GameConfig,
    LeafField,
    ModelError,
    State,
    StepResult,
    model_fields,
)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class EnvironmentState:
    core_state: State

    environment_root: Array

    episode_index: Array


def episode_root(environment_state: EnvironmentState) -> Array:
    return jax.random.fold_in(
        environment_state.environment_root, environment_state.episode_index
    )


def episode_start(environment_state: EnvironmentState, game: GameConfig) -> State:
    return core.reset(episode_root(environment_state), game)


def initial(environment_root: Array, game: GameConfig) -> EnvironmentState:
    carry = EnvironmentState(
        core_state=game.initial_state,
        environment_root=jnp.asarray(environment_root),
        episode_index=jnp.asarray(0, jnp.uint32),
    )
    return dataclasses.replace(carry, core_state=episode_start(carry, game))


def advance(environment_state: EnvironmentState, result: StepResult, game: GameConfig, *, ordered: Any = None) -> EnvironmentState:
    done = result.terminated | result.truncated
    episode_index = jnp.where(
        done,
        environment_state.episode_index + jnp.asarray(1, jnp.uint32),
        environment_state.episode_index,
    ).astype(jnp.uint32)
    following = EnvironmentState(
        core_state=result.next_state,
        environment_root=environment_state.environment_root,
        episode_index=episode_index,
    )
    return dataclasses.replace(
        following,
        core_state=kernels.dispatch(
            done,
            episode_start(following, game),
            result.next_state,
            ordered,
            kernel=_reset_in_place,
            program=_reset_by_select,
        ),
    )


def spawn_overflowed(ending: State) -> Array:
    return jnp.any(ending.telemetry.spawn_overflow > 0)


def select[Tree](taken: Array, at_boundary: Tree, otherwise: Tree) -> Tree:
    return jax.tree.map(
        lambda boundary, plain: jnp.where(taken, boundary, plain),
        at_boundary,
        otherwise,
    )


_RESET_TARGET = "nanorts_reset_where_done"

_ROWS = tuple(
    field.name
    for field in model_fields(State)
    if isinstance(field, LeafField) and field.spec.shape in (PER_SLOT, FIELD)
)


def _reset_by_select(done: Array, reset: State, ending: State, ordered: Any) -> State:
    del ordered
    return select(done, reset, ending)


def _reset_in_place(done: Array, reset: State, ending: State, ordered: Any) -> State:
    rows = [getattr(ending, name) for name in _ROWS]
    written = kernels.call(
        _RESET_TARGET,
        [
            done,
            *(getattr(reset, name) for name in _ROWS),
            *rows,
            *jax.tree.leaves(ordered),
        ],
        [jax.ShapeDtypeStruct(row.shape, row.dtype) for row in rows],
        aliases={1 + len(_ROWS) + index: index for index in range(len(_ROWS))},
        attributes={"leaves": len(_ROWS)},
    )
    leaves: dict[str, Any] = {
        field.name: select(
            done, getattr(reset, field.name), getattr(ending, field.name)
        )
        for field in dataclasses.fields(State)
        if field.name not in _ROWS
    }
    leaves.update(zip(_ROWS, written, strict=True))
    return dataclasses.replace(ending, **leaves)


def refuse_decided(where: str, game: GameConfig) -> None:
    if bool(core.reset(jax.random.key(0), game).game_over):
        raise ModelError(
            f"{where} refuses a configuration whose initial State is already "
            "decided: a public environment steps only a playable State"
        )
