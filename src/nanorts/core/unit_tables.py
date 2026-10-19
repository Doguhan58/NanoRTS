from __future__ import annotations

from typing import Final

import numpy as np

from nanorts.model import GameConfig, host_arrays

UNIT_TABLE_COLUMNS: Final[tuple[str, ...]] = (
    "unit_can_move",
    "unit_can_harvest",
    "unit_is_resource",
    "unit_is_stockpile",
    "unit_can_attack",
    "unit_attack_range",
    "unit_cost",
)

TIMING_COLUMNS: Final[tuple[str, ...]] = (
    "unit_move_time",
    "unit_harvest_time",
    "unit_attack_time",
    "unit_produce_time",
    "unit_hit_points",
    "unit_attack_damage",
    "unit_harvest_amount",
)


def unit_table(game: GameConfig) -> np.ndarray:
    columns = host_arrays(game.unit_types)
    return np.concatenate(
        [
            np.stack(
                [columns[name].astype(np.int32) for name in UNIT_TABLE_COLUMNS],
                axis=-1,
            ),
            columns["unit_produces"].astype(np.int32),
        ],
        axis=-1,
    )


def unit_timings(game: GameConfig) -> np.ndarray:
    columns = host_arrays(game.unit_types)
    return np.stack(
        [columns[name].astype(np.int32) for name in TIMING_COLUMNS], axis=-1
    )
