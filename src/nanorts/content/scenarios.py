from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

from nanorts.content.loading import load_game
from nanorts.model import ConflictStrategy, GameConfig, ModelError, UnitTypeTableVersion

MAP_DIRECTORY: Final[Path] = Path(__file__).resolve().parents[1] / "maps"

_MAP_PATHS: Final[Mapping[str, Path]] = MappingProxyType(
    {path.stem: path for path in sorted(MAP_DIRECTORY.rglob("*.xml"))}
)

SCENARIO_NAMES: Final[tuple[str, ...]] = tuple(sorted(_MAP_PATHS))

CAPACITY_PROFILES: Final[Mapping[str, int]] = MappingProxyType(
    {"basesWorkers8x8": 52, "basesWorkers16x16": 70}
)


def map_name_to_scenario(name: str, *, utt: UnitTypeTableVersion = UnitTypeTableVersion.ORIGINAL_FINETUNED, conflict_strategy: ConflictStrategy = ConflictStrategy.CANCEL_BOTH, max_cycles: int = 5000, capacity_profile: bool = True) -> GameConfig:
    path = _shipped_path(name, "map_name_to_scenario")
    game = load_game(
        path,
        utt=utt,
        conflict_strategy=conflict_strategy,
        max_cycles=max_cycles,
    )
    profile = CAPACITY_PROFILES.get(name)
    if profile is None or not capacity_profile or profile >= game.bucket.per_player_capacity:
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
    return _shipped_path(name, "shipped_map")


def _shipped_path(name: str, caller: str) -> Path:
    path = _MAP_PATHS.get(name)
    if path is None:
        raise ModelError(
            f"{caller} refuses scenario {name!r}: it is none of the "
            f"{len(SCENARIO_NAMES)} shipped maps that scenario_names() lists. "
            "Custom content is scenario=content.load_game(path)"
        )
    return path
