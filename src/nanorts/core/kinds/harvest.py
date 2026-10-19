from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from nanorts.core.effect import CompletionRecord, Effect, Removal, Write
from nanorts.core.kinds.action_kind import ActionKind, LegalityGrid, Member, Request
from nanorts.model import KIND_HARVEST, GameConfig


def _duration(game: GameConfig, actor_type: Array, none_duration: Array, produced_type: Array) -> Array:
    return game.unit_harvest_time[actor_type]


def _is_valid(request: Request) -> Array:
    game, neighbour = request.game, request.neighbour
    return (
        game.unit_can_harvest[request.actor_type]
        & (request.carried == 0)
        & neighbour.occupied
        & game.unit_is_resource[neighbour.unit_type]
    )


def _legality(grid: LegalityGrid) -> tuple[Array, dict[str, Array]]:
    harvest_direction = grid.ask(direction=grid.directions)
    return jnp.any(harvest_direction, axis=-1), {"harvest_direction": harvest_direction}


def _complete(member: Member) -> Effect:
    state, game, stack = member.state, member.game, member.neighbour
    harvesting = (
        member.performs(KIND_HARVEST)
        & stack.occupied
        & game.unit_is_resource[stack.unit_type]
    )
    amount = game.unit_harvest_amount[member.actor_type]
    left_in_stack = state.carried_resources[stack.slot] - amount
    depleted = harvesting & (left_in_stack <= 0)
    return Effect(
        writes=(
            Write("carried_resources", stack.slot, harvesting, left_in_stack),
            Write("carried_resources", member.slot, harvesting & member.alive, amount),
        ),
        removal=Removal(depleted, stack.slot),
        records=(
            CompletionRecord(
                "harvest",
                harvesting,
                {
                    "target_slot": stack.slot,
                    "target_birth": state.birth[stack.slot],
                    "amount": amount,
                    "depleted": depleted,
                },
            ),
        ),
    )


KIND = ActionKind(
    code=KIND_HARVEST,
    duration=_duration,
    is_valid=_is_valid,
    claims_position=False,
    cost=None,
    payload=frozenset({"direction"}),
    legality=_legality,
    complete=_complete,
)
