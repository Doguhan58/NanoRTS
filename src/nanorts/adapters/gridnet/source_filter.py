from __future__ import annotations

import dataclasses
import functools
import operator

import jax.numpy as jnp
from jax import Array, lax

from nanorts.model import (
    DIRECTION_COUNT,
    DIRECTION_OFFSETS,
    KIND_MOVE,
    KIND_PRODUCE,
    PLAYER_COUNT,
    POSITION_CLAIMING_KINDS,
    BucketSpec,
    GameConfig,
    Legality,
    State,
)


@dataclasses.dataclass(frozen=True)
class DecodedRows:
    order: Array

    count: Array

    kind: Array
    direction: Array
    none_duration: Array
    target_cell: Array
    produced_type: Array

    position: Array

    has_position: Array
    cost: Array

    survived: Array


def claims_position(kind: Array) -> Array:
    return functools.reduce(
        operator.or_, [kind == claiming for claiming in POSITION_CLAIMING_KINDS]
    )


def direction_delta(direction: Array, width: int) -> Array:
    matches = [direction == code for code in range(DIRECTION_COUNT)]
    delta: Array | int = 0
    for match, (dx, dy) in zip(
        reversed(matches), reversed(DIRECTION_OFFSETS), strict=True
    ):
        delta = jnp.where(match, dx + dy * width, delta)
    return jnp.asarray(delta).astype(jnp.int32)


def pending_usage(state: State, game: GameConfig) -> tuple[Array, Array]:
    bucket = game.bucket
    width, margin, domain = bucket.width, bucket.width, position_domain(bucket)
    claims = state.active & claims_position(state.pending_kind)
    position = state.cell + direction_delta(state.pending_direction, width)
    positions = (
        jnp.zeros((domain,), jnp.bool_)
        .at[jnp.where(claims, position + margin, domain - 1)]
        .set(True)
    )
    positions = positions.at[domain - 1].set(False)

    produces = state.active & (state.pending_kind == KIND_PRODUCE)
    cost = jnp.where(produces, game.unit_cost[state.pending_produced_type], 0)
    owned = cost[bucket.neutral_capacity :].reshape(
        PLAYER_COUNT, bucket.per_player_capacity
    )
    resources = jnp.sum(owned, axis=1).astype(jnp.int32)
    return positions, resources


def keep(rows: DecodedRows, positions: Array, resources: Array, state: State, player: int, bucket: BucketSpec) -> DecodedRows:
    margin, domain = bucket.width, position_domain(bucket)
    capacity = bucket.per_player_capacity
    balances = state.player_resources

    def visit(position: Array, carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        used, spent, survived = carry
        claims = rows.has_position[position]
        index = jnp.where(claims, rows.position[position] + margin, domain - 1)
        total = spent + jnp.where(
            jnp.arange(PLAYER_COUNT, dtype=jnp.int32) == player, rows.cost[position], 0
        )
        consistent = ~(claims & used[index]) & ~jnp.any(
            (spent != 0) & (total > 0) & (total > balances)
        )
        used_after = used | (
            (jnp.arange(domain, dtype=jnp.int32) == index) & consistent & claims
        )
        spent_after = jnp.where(consistent, total, spent)
        survived_after = survived | (
            (jnp.arange(capacity, dtype=jnp.int32) == position) & consistent
        )
        return used_after, spent_after, survived_after

    def unfinished(carry: tuple[Array, tuple[Array, Array, Array]]) -> Array:
        return carry[0] < rows.count

    def advance_one(carry: tuple[Array, tuple[Array, Array, Array]]) -> tuple[Array, tuple[Array, Array, Array]]:
        position, carried = carry
        return position + 1, visit(position, carried)

    _, (_, _, survived) = lax.while_loop(
        unfinished,
        advance_one,
        (jnp.asarray(0, jnp.int32), (positions, resources, rows.survived)),
    )
    return dataclasses.replace(rows, survived=survived)


def issued_usage(rows: DecodedRows, legality: Legality, positions: Array, resources: Array, player: int, bucket: BucketSpec) -> tuple[Array, Array]:
    margin, domain = bucket.width, position_domain(bucket)
    slot, direction = rows.order, rows.direction
    legal = jnp.where(
        rows.kind == KIND_MOVE,
        legality.move_direction[player, slot, direction],
        legality.produce_direction[player, slot, direction]
        & legality.produce_type[player, slot, rows.produced_type],
    )
    kept = rows.survived & rows.has_position & legal
    positions = positions.at[jnp.where(kept, rows.position + margin, domain - 1)].set(
        True
    )
    positions = positions.at[domain - 1].set(False)
    spent = jnp.sum(jnp.where(kept & (rows.kind == KIND_PRODUCE), rows.cost, 0))
    return positions, resources.at[player].add(spent)


def position_domain(bucket: BucketSpec) -> int:
    return bucket.cell_count + 2 * bucket.width + 1
