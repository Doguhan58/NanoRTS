from __future__ import annotations

import dataclasses
from typing import Final

import jax.numpy as jnp
from jax import Array

from nanorts import core
from nanorts.bots.bot import Bot
from nanorts.model import KIND_NONE, PLAYER_COUNT, Commands, GameConfig, State, empty

_FILLER_DURATION: Final[int] = 10


def _init(game: GameConfig) -> tuple[()]:
    del game
    return ()


def _act(key: Array, carry: tuple[()], state: State, game: GameConfig, player: Array | int) -> tuple[tuple[()], Commands]:
    del key
    available = core.legal_commands(state, game).available
    waiting = available & (jnp.arange(PLAYER_COUNT)[:, None] == player)
    unissued = empty(Commands, game.bucket)
    return carry, dataclasses.replace(
        unissued,
        kind=jnp.where(waiting, KIND_NONE, unissued.kind),
        none_duration=jnp.where(waiting, _FILLER_DURATION, unissued.none_duration),
    )


passiveAI: Final[Bot[tuple[()]]] = Bot(name="passiveAI", init=_init, act=_act)
