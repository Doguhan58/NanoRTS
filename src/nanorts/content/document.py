from __future__ import annotations

import dataclasses


class ContentError(ValueError):
    def __init__(self, message: str, *, clause: int | None = None) -> None:
        super().__init__(message)
        self.clause = clause


@dataclasses.dataclass(frozen=True)
class UnitRecord:
    type: str
    id: int
    player: int
    x: int
    y: int
    hitpoints: int
    resources: int


@dataclasses.dataclass(frozen=True)
class ContentDocument:
    width: int
    height: int
    terrain: str
    players: tuple[tuple[int, int], ...]
    units: tuple[UnitRecord, ...]
