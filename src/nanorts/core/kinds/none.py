from __future__ import annotations

from jax import Array

from nanorts.core.effect import Effect
from nanorts.core.kinds.action_kind import ActionKind, LegalityGrid, Member, Request
from nanorts.model import KIND_NONE, GameConfig


def _duration(game: GameConfig, actor_type: Array, none_duration: Array, produced_type: Array) -> Array:
    return none_duration


def _is_valid(request: Request) -> bool:
    return True


def _legality(grid: LegalityGrid) -> tuple[Array, dict[str, Array]]:
    return grid.active, {}


def _complete(member: Member) -> Effect:
    return Effect()


KIND = ActionKind(
    code=KIND_NONE,
    duration=_duration,
    is_valid=_is_valid,
    claims_position=False,
    cost=None,
    payload=frozenset({"none_duration"}),
    legality=_legality,
    complete=_complete,
)
