from typing import Any

import jax
from jax import Array

from nanorts import kernels
from nanorts.core import (
    admission,
    completion,
    cycle,
    legality,
    outcome,
    pending,
    reservations,
)
from nanorts.model import (
    Commands,
    ConflictStrategy,
    GameConfig,
    Legality,
    ModelError,
    State,
    StepResult,
)

__all__ = ["legal_commands", "reset", "step"]


def reset(key: Array, game: GameConfig) -> State:
    del key
    return outcome.decide(game.initial_state, game)


def legal_commands(state: State, game: GameConfig, *, stored: str = "rows") -> Legality:
    if stored not in legality.KERNEL_PATHS:
        raise ModelError(
            f"core.legal_commands refuses stored={stored!r}: the grid is stored "
            f"as one of {list(legality.KERNEL_PATHS)}"
        )
    return kernels.dispatch(
        state, game, kernel=legality.KERNEL_PATHS[stored], program=legality.derive
    )


def step(key: Array, state: State, commands: Commands, game: GameConfig) -> StepResult:
    if kernels.selected() and not _any_traced(key, state, commands, game):
        return _jitted_step(key, state, commands, game)
    result = _t1_step(key, state, commands, game)
    if not _kernel_can_step(key, game):
        return result
    return kernels.dispatch(
        key,
        state,
        commands,
        game,
        result,
        kernel=_kernel_step,
        program=_precomputed_t1_result,
    )


def _kernel_can_step(key: Array, game: GameConfig) -> bool:
    if not cycle.fits(game.bucket):
        return False
    draws_randomly = game.bucket.conflict_strategy == ConflictStrategy.CANCEL_RANDOM
    return not draws_randomly or reservations.kernel_draws_valid(key)


def _t1_step(key: Array, state: State, commands: Commands, game: GameConfig) -> StepResult:
    admitted, events = admission.admit(key, state, commands, game)
    advanced, completed = pending.select(admitted)
    acted, events = completion.apply(advanced, completed, events, game)
    acted = outcome.decide(acted, game)
    return StepResult(
        next_state=acted,
        terminated=acted.game_over,
        truncated=outcome.truncates(acted, game),
        reward=outcome.sparse_reward(acted, game),
        events=events,
    )


def _precomputed_t1_result(key: Array, state: State, commands: Commands, game: GameConfig, result: StepResult) -> StepResult:
    return result


def _kernel_step(key: Array, state: State, commands: Commands, game: GameConfig, result: StepResult) -> StepResult:
    return cycle.on_device(key, state, commands, game)


def _any_traced(*arguments: Any) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(arguments))


_jitted_step = jax.jit(step)
