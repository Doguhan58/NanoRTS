from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

from nanorts.content.loading import load_game
from nanorts.model import ConflictStrategy, GameConfig, ModelError, UnitTypeTableVersion

SCENARIO_NAMES: Final[tuple[str, ...]] = ("basesWorkers8x8", "basesWorkers16x16")

CAPACITY_PROFILES: Final[Mapping[str, int]] = MappingProxyType(
    {"basesWorkers8x8": 52, "basesWorkers16x16": 70}
)

MAP_DIRECTORY: Final[Path] = Path(__file__).resolve().parents[1] / "maps"


def map_name_to_scenario(name: str, *, utt: UnitTypeTableVersion = UnitTypeTableVersion.ORIGINAL, conflict_strategy: ConflictStrategy = ConflictStrategy.CANCEL_BOTH, max_cycles: int = 5000, capacity_profile: bool = True) -> GameConfig:
    if name not in SCENARIO_NAMES:
        raise ModelError(
            f"map_name_to_scenario refuses scenario {name!r}: the shipped "
            f"release scenarios are {list(SCENARIO_NAMES)}. Custom content is "
            "scenario=content.load_game(path)"
        )
    path = MAP_DIRECTORY / f"{name}.xml"
    game = load_game(
        path,
        utt=utt,
        conflict_strategy=conflict_strategy,
        max_cycles=max_cycles,
    )
    profile = CAPACITY_PROFILES[name]
    if not capacity_profile or profile >= game.bucket.per_player_capacity:
        return game
    return load_game(
        path,
        utt=utt,
        conflict_strategy=conflict_strategy,
        max_cycles=max_cycles,
        bucket=dataclasses.replace(game.bucket, per_player_capacity=profile),
    )


def scenario_names() -> tuple[str, ...]:
    return SCENARIO_NAMES


def shipped_map(name: str) -> Path:
    if name not in SCENARIO_NAMES:
        raise ModelError(
            f"shipped_map refuses scenario {name!r}: the shipped release "
            f"scenarios are {list(SCENARIO_NAMES)}"
        )
    return MAP_DIRECTORY / f"{name}.xml"
