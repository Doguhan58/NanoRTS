from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from nanorts.core.effect import Effect, Write
from nanorts.core.kinds.action_kind import ActionKind, LegalityGrid, Member, Request
from nanorts.model import ABSENT, KIND_MOVE, GameConfig


def _duration(game: GameConfig, actor_type: Array, none_duration: Array, produced_type: Array) -> Array:
    return game.unit_move_time[actor_type]


def _is_valid(request: Request) -> Array:
    return request.game.unit_can_move[request.actor_type] & request.neighbour.free


def _legality(grid: LegalityGrid) -> tuple[Array, dict[str, Array]]:
    move_direction = grid.ask(direction=grid.directions)
    return jnp.any(move_direction, axis=-1), {"move_direction": move_direction}


def _complete(member: Member) -> Effect:
    moving = member.performs(KIND_MOVE) & member.alive
    return Effect(
        writes=(
            Write("entity_at", member.destination, moving, member.slot),
            Write("entity_at", member.retained.cell, moving, ABSENT),
            Write("cell", member.slot, moving, member.destination),
        )
    )


KIND = ActionKind(
    code=KIND_MOVE,
    duration=_duration,
    is_valid=_is_valid,
    claims_position=True,
    cost=None,
    payload=frozenset({"direction"}),
    legality=_legality,
    complete=_complete,
)
