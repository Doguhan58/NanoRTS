from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array

from nanorts.core import units
from nanorts.core.effect import CompletionRecord, Effect, Increment, Write
from nanorts.core.kinds.action_kind import ActionKind, LegalityGrid, Member, Request
from nanorts.model import KIND_PRODUCE, PLAYER_COUNT, GameConfig


def _known_type(game: GameConfig, produced_type: Array | np.ndarray) -> Array:
    return jnp.clip(produced_type, 0, game.bucket.unit_type_capacity - 1)


def _duration(game: GameConfig, actor_type: Array, none_duration: Array, produced_type: Array) -> Array:
    return game.unit_produce_time[_known_type(game, produced_type)]


def _cost(game: GameConfig, produced_type: Array | np.ndarray) -> Array:
    return game.unit_cost[_known_type(game, produced_type)]


def _is_valid(request: Request) -> Array | np.ndarray:
    state, game = request.state, request.game
    produced_type = request.produced_type
    balance = state.player_resources[jnp.clip(request.owner, 0, PLAYER_COUNT - 1)]
    return (
        (produced_type >= 0)
        & (produced_type < game.bucket.unit_type_capacity)
        & game.unit_produces[request.actor_type, _known_type(game, produced_type)]
        & (balance >= _cost(game, produced_type))
        & request.neighbour.free
    )


def _legality(grid: LegalityGrid) -> tuple[Array, dict[str, Array]]:
    by_direction_and_type = grid.ask(
        direction=grid.directions[:, None], produced_type=grid.unit_types
    )
    produce_type = jnp.any(by_direction_and_type, axis=-2)
    return jnp.any(produce_type, axis=-1), {
        "produce_direction": jnp.any(by_direction_and_type, axis=-1),
        "produce_type": produce_type,
    }


def _complete(member: Member) -> Effect:
    state, game, player = member.state, member.game, member.player
    produced_type = member.retained.produced_type
    newborn_type = _known_type(game, produced_type)
    balance = state.player_resources[player] - _cost(game, produced_type)
    affordable = member.performs(KIND_PRODUCE) & (balance >= 0)
    has_free_slot, newborn = units.lowest_free_slot(
        state, player, member.all_slots, game
    )
    spawns = affordable & has_free_slot
    overflows = affordable & ~has_free_slot

    def newborn_field(field: str, value: Array | bool) -> Write:
        return Write(field, newborn, spawns, value)

    return Effect(
        writes=(
            Write("player_resources", player, spawns, balance),
            Write("entity_at", member.destination, spawns, newborn),
            newborn_field("birth", state.next_birth),
            newborn_field("unit_type", newborn_type),
            newborn_field("owner", player),
            newborn_field("cell", member.destination),
            newborn_field("hit_points", game.unit_hit_points[newborn_type]),
            Increment("next_birth", None, spawns),
            Increment("telemetry.spawn_overflow", player, overflows),
        ),
        records=(
            CompletionRecord(
                "spawn",
                spawns,
                {
                    "produced_type": newborn_type,
                    "slot": newborn,
                    "birth": state.next_birth,
                    "cell": member.destination,
                    "balance": balance,
                },
            ),
        ),
    )


KIND = ActionKind(
    code=KIND_PRODUCE,
    duration=_duration,
    is_valid=_is_valid,
    claims_position=True,
    cost=_cost,
    payload=frozenset({"direction", "produced_type"}),
    legality=_legality,
    complete=_complete,
)
