from __future__ import annotations

from typing import Final

from jax import Array

from nanorts.bots import abstraction, family
from nanorts.bots.abstraction import AbstractActions
from nanorts.bots.bot import Bot
from nanorts.model import TYPE_HEAVY, Commands, GameConfig, State


def _act(key: Array, actions: AbstractActions, state: State, game: GameConfig, player: Array | int) -> tuple[AbstractActions, Commands]:
    del key
    return family.act(
        family.BARRACKS_DEFENSE, actions, state, game, player, trained=TYPE_HEAVY
    )


heavyDefense: Final[Bot[AbstractActions]] = Bot(
    name="heavyDefense", init=abstraction.empty_actions, act=_act
)
