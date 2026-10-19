from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from nanorts.core.effect import CompletionRecord, Effect, Removal, Write
from nanorts.core.kinds.action_kind import ActionKind, LegalityGrid, Member, Request
from nanorts.model import (
    ABSENT,
    KIND_ATTACK,
    OWNER_NEUTRAL,
    GameConfig,
    State,
    attack_offsets,
    board_view,
    window_targets,
)


def _duration(game: GameConfig, actor_type: Array, none_duration: Array, produced_type: Array) -> Array:
    return game.unit_attack_time[actor_type]


def _can_attack(game: GameConfig, actor_type: Array, owner: Array, target_owner: Array, in_range: Array) -> Array:
    return (
        game.unit_can_attack[actor_type]
        & (target_owner >= 0)
        & (target_owner != owner)
        & in_range
    )


def _is_valid(request: Request) -> Array:
    state, game = request.state, request.game
    width, cells = game.bucket.width, game.bucket.cell_count
    target_cell = request.target_cell
    on_board = (target_cell >= 0) & (target_cell < cells)
    target_slot = state.entity_at.reshape(cells)[jnp.where(on_board, target_cell, 0)]
    holds_unit = on_board & (target_slot != ABSENT)
    target_owner = jnp.where(
        holds_unit,
        state.owner[jnp.where(holds_unit, target_slot, 0)],
        jnp.asarray(OWNER_NEUTRAL, jnp.int32),
    )
    attack_range = game.unit_attack_range[request.actor_type]
    dx = target_cell % width - request.cell % width
    dy = target_cell // width - request.cell // width
    return _can_attack(
        game,
        request.actor_type,
        request.owner,
        target_owner,
        dx * dx + dy * dy <= attack_range * attack_range,
    )


def _legality(grid: LegalityGrid) -> tuple[Array, dict[str, Array]]:
    state, game = grid.state, grid.game
    attack_range = game.unit_attack_range[grid.actor_type]
    offsets = attack_offsets(game.bucket)
    squared_distance = jnp.asarray(offsets[:, 0] ** 2 + offsets[:, 1] ** 2)
    owner_table_index = window_targets(grid.cell, game) + 1
    attack_offset = grid.active[..., None] & _can_attack(
        game,
        grid.actor_type[..., None],
        grid.owner[..., None].astype(jnp.int8),
        _owner_table_with_off_board_first(state, game)[owner_table_index],
        squared_distance[None, None, :] <= (attack_range * attack_range)[..., None],
    )
    return jnp.any(attack_offset, axis=-1), {"attack_offset": attack_offset}


def _owner_table_with_off_board_first(state: State, game: GameConfig) -> Array:
    owner_by_cell = board_view(state, game).owner.reshape(game.bucket.cell_count)
    return jnp.concatenate([jnp.full((1,), OWNER_NEUTRAL, jnp.int8), owner_by_cell])


def _complete(member: Member) -> Effect:
    state, game = member.state, member.game
    cells = game.bucket.cell_count
    occupant = state.entity_at.reshape(cells)[
        jnp.clip(member.retained.target_cell, 0, cells - 1)
    ]
    hit = member.performs(KIND_ATTACK) & (occupant != ABSENT)
    victim = jnp.where(hit, occupant, 0)
    damage = game.unit_attack_damage[member.actor_type]
    hit_points_left = state.hit_points[victim] - damage
    return Effect(
        writes=(Write("hit_points", victim, hit, hit_points_left),),
        removal=Removal(hit & (hit_points_left <= 0), victim),
        records=(
            CompletionRecord(
                "hit",
                hit,
                {
                    "target_slot": victim,
                    "target_birth": state.birth[victim],
                    "damage": damage,
                },
            ),
        ),
    )


KIND = ActionKind(
    code=KIND_ATTACK,
    duration=_duration,
    is_valid=_is_valid,
    claims_position=False,
    cost=None,
    payload=frozenset({"target_cell"}),
    legality=_legality,
    complete=_complete,
)
