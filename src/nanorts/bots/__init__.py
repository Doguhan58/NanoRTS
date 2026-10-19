import importlib
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from nanorts.bots.bot import Bot
from nanorts.bots.coac import coacAI
from nanorts.bots.gridnet_seat import (
    GridNetGroup,
    GridNetMix,
    GridNetMixState,
    GridNetMixStep,
    GridNetSeatState,
    GridNetSeatStep,
    gridnet_seat_commands,
    reset_gridnet_seat,
    step_gridnet_seat,
)
from nanorts.bots.heavy_defense import heavyDefense
from nanorts.bots.heavy_rush import heavyRush
from nanorts.bots.light_defense import lightDefense
from nanorts.bots.light_rush import lightRushAI
from nanorts.bots.mayari_bot import mayari
from nanorts.bots.mix import Group, GroupMix
from nanorts.bots.obi_bot_kenobi import obiBotKenobi
from nanorts.bots.passive import passiveAI
from nanorts.bots.random_biased import randomAI, randomBiasedAI
from nanorts.bots.ranged_defense import rangedDefense
from nanorts.bots.ranged_rush import rangedRush
from nanorts.bots.tma_bot import tma
from nanorts.bots.worker_defense import workerDefense
from nanorts.bots.worker_rush import workerRushAI

__all__ = [
    "LEARNER",
    "ROSTER",
    "SEAT_RULES",
    "Bot",
    "GridNetGroup",
    "GridNetMix",
    "GridNetMixState",
    "GridNetMixStep",
    "GridNetSeatState",
    "GridNetSeatStep",
    "Group",
    "GroupMix",
    "SeatEnv",
    "SeatState",
    "SeatStep",
    "coacAI",
    "gridnet_seat_commands",
    "heavyDefense",
    "heavyRush",
    "legal_seat_actions",
    "lightDefense",
    "lightRushAI",
    "mayari",
    "obiBotKenobi",
    "observe_seat",
    "passiveAI",
    "randomAI",
    "randomBiasedAI",
    "rangedDefense",
    "rangedRush",
    "reset_gridnet_seat",
    "reset_seat",
    "seat_commands",
    "seat_slot_cells",
    "step_gridnet_seat",
    "step_seat",
    "tma",
    "workerDefense",
    "workerRushAI",
]

ROSTER: Mapping[str, Bot[Any]] = MappingProxyType(
    {
        bot.name: bot
        for bot in (
            passiveAI,
            randomAI,
            randomBiasedAI,
            obiBotKenobi,
            workerRushAI,
            lightRushAI,
            heavyRush,
            rangedRush,
            workerDefense,
            lightDefense,
            heavyDefense,
            rangedDefense,
            mayari,
            coacAI,
            tma,
        )
    }
)

_NAMES_BOUND_ON_FIRST_USE = (
    "LEARNER",
    "SEAT_RULES",
    "SeatEnv",
    "SeatState",
    "SeatStep",
    "legal_seat_actions",
    "observe_seat",
    "reset_seat",
    "seat_commands",
    "seat_slot_cells",
    "step_seat",
)


def __getattr__(name: str) -> Any:
    if name in _NAMES_BOUND_ON_FIRST_USE:
        return getattr(importlib.import_module("nanorts.bots.seat"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
