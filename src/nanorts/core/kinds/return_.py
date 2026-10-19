from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from nanorts.core.effect import CompletionRecord, Effect, Write
from nanorts.core.kinds.action_kind import ActionKind, LegalityGrid, Member, Request
from nanorts.model import KIND_RETURN, GameConfig


def _duration(game: GameConfig, actor_type: Array, none_duration: Array, produced_type: Array) -> Array:
    return game.unit_move_time[actor_type]


def _is_valid(request: Request) -> Array:
    game, neighbour = request.game, request.neighbour
    return (
        game.unit_can_harvest[request.actor_type]
        & (request.carried > 0)
        & neighbour.occupied
        & game.unit_is_stockpile[neighbour.unit_type]
        & (neighbour.owner == request.owner)
    )


def _legality(grid: LegalityGrid) -> tuple[Array, dict[str, Array]]:
    return_direction = grid.ask(direction=grid.directions)
    return jnp.any(return_direction, axis=-1), {"return_direction": return_direction}


def _complete(member: Member) -> Effect:
    state, game, stockpile = member.state, member.game, member.neighbour
    carried = member.retained.carried
    returning = (
        member.performs(KIND_RETURN)
        & stockpile.occupied
        & game.unit_is_stockpile[stockpile.unit_type]
        & (carried > 0)
    )
    credit = jnp.where(returning, carried, 0)
    balance = state.player_resources[member.player] + credit
    return Effect(
        writes=(
            Write("carried_resources", member.slot, returning & member.alive, 0),
            Write("player_resources", member.player, returning, balance),
        ),
        records=(
            CompletionRecord(
                "return",
                returning,
                {
                    "target_slot": stockpile.slot,
                    "target_birth": state.birth[stockpile.slot],
                    "amount": credit,
                    "balance": balance,
                },
            ),
        ),
    )


KIND = ActionKind(
    code=KIND_RETURN,
    duration=_duration,
    is_valid=_is_valid,
    claims_position=False,
    cost=None,
    payload=frozenset({"direction"}),
    legality=_legality,
    complete=_complete,
)
