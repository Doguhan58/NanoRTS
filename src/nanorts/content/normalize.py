from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array

from nanorts.content.document import ContentDocument, ContentError, UnitRecord
from nanorts.model import (
    OWNER_NEUTRAL,
    PLAYER_COUNT,
    BucketSpec,
    ConflictStrategy,
    GameConfig,
    State,
    UnitType,
    UnitTypeTableVersion,
    content_hash,
    empty,
)

_CALLER_CHOSEN_CAPACITIES = ("neutral_capacity", "per_player_capacity")


def _population(document: ContentDocument, owner: int) -> int:
    return sum(1 for unit in document.units if unit.player == owner)


def _initial_neutral_count(document: ContentDocument) -> int:
    return _population(document, OWNER_NEUTRAL)


def _initial_player_maximum(document: ContentDocument) -> int:
    return max(_population(document, player) for player in range(PLAYER_COUNT))


def _chosen_capacity_floors(document: ContentDocument) -> dict[str, int]:
    return {
        "neutral_capacity": _initial_neutral_count(document),
        "per_player_capacity": _initial_player_maximum(document),
    }


def _resource_bound(document: ContentDocument, table: tuple[UnitType, ...]) -> int | None:
    produced = [unit for unit in table if unit.produced_by]
    harvesters = [unit for unit in table if unit.can_harvest]
    if not _resource_bound_applies(produced, harvesters):
        return None
    types = {unit.name: unit for unit in table}
    largest_harvest = max(unit.harvest_amount for unit in harvesters)
    resources_on_map = sum(
        max(unit.resources, 1) + largest_harvest - 1
        if types[unit.type].is_resource
        else max(unit.resources, 0)
        for unit in document.units
    )
    cheapest_cost = min(unit.cost for unit in produced)
    return max(
        _population(document, player)
        + max(0, balance + resources_on_map) // cheapest_cost
        for player, balance in document.players
    )


def _resource_bound_applies(produced: list[UnitType], harvesters: list[UnitType]) -> bool:
    return (
        bool(produced)
        and bool(harvesters)
        and all(unit.cost > 0 and not unit.is_resource for unit in produced)
        and all(
            not unit.is_resource and unit.harvest_amount >= 0 for unit in harvesters
        )
    )


def derive_bucket(document: ContentDocument, table: tuple[UnitType, ...], utt: UnitTypeTableVersion, conflict_strategy: ConflictStrategy) -> BucketSpec:
    cells = document.height * document.width
    resource_bound = _resource_bound(document, table)
    max_attack_range = max(unit.attack_range for unit in table)
    return BucketSpec(
        height=document.height,
        width=document.width,
        neutral_capacity=_initial_neutral_count(document),
        per_player_capacity=(
            cells if resource_bound is None else max(1, min(cells, resource_bound))
        ),
        unit_type_capacity=len(table),
        attack_offset_capacity=(2 * max_attack_range + 1) ** 2,
        utt_identity=utt,
        utt_content_hash=content_hash(table),
        conflict_strategy=conflict_strategy,
    )


def check_supplied_bucket(derived: BucketSpec, supplied: BucketSpec, document: ContentDocument) -> None:
    differences = [
        f"{field.name} is {getattr(supplied, field.name)!r}, "
        f"expected {getattr(derived, field.name)!r}"
        for field in dataclasses.fields(derived)
        if field.name not in _CALLER_CHOSEN_CAPACITIES
        and getattr(supplied, field.name) != getattr(derived, field.name)
    ]
    for name, floor in _chosen_capacity_floors(document).items():
        chosen = getattr(supplied, name)
        if not floor <= chosen <= derived.cell_count:
            differences.append(
                f"{name} is {chosen!r}, expected the inclusive interval "
                f"[{floor}, {derived.cell_count}] between the population the "
                "content puts in that block and its cell count"
            )
    if differences:
        raise ContentError(
            "the supplied BucketSpec disagrees with the content: "
            + "; ".join(differences)
        )


def _initial_state(document: ContentDocument, table: tuple[UnitType, ...], bucket: BucketSpec) -> State:
    names = {unit.name: index for index, unit in enumerate(table)}
    state = empty(State, bucket)
    next_free = [0] * (PLAYER_COUNT + 1)
    slots: list[int] = []
    for unit in document.units:
        slots.append(_claim_slot(unit, bucket, next_free))
    index = jnp.asarray(slots, jnp.int32)
    xs = jnp.asarray([unit.x for unit in document.units], jnp.int32)
    ys = jnp.asarray([unit.y for unit in document.units], jnp.int32)
    return dataclasses.replace(
        state,
        birth=_placed(state.birth, index, range(len(document.units))),
        unit_type=_placed(
            state.unit_type, index, [names[unit.type] for unit in document.units]
        ),
        owner=_placed(state.owner, index, [unit.player for unit in document.units]),
        cell=_placed(
            state.cell,
            index,
            [unit.x + unit.y * bucket.width for unit in document.units],
        ),
        hit_points=_placed(
            state.hit_points, index, [unit.hitpoints for unit in document.units]
        ),
        carried_resources=_placed(
            state.carried_resources,
            index,
            [unit.resources for unit in document.units],
        ),
        entity_at=state.entity_at.at[ys, xs].set(index.astype(state.entity_at.dtype)),
        player_resources=jnp.asarray(
            [resources for _, resources in document.players], jnp.int32
        ),
        next_birth=jnp.asarray(len(document.units), jnp.int32),
    )


def _claim_slot(unit: UnitRecord, bucket: BucketSpec, next_free: list[int]) -> int:
    if unit.player == OWNER_NEUTRAL:
        block, start, capacity = "neutral", 0, bucket.neutral_capacity
    else:
        block = f"player {unit.player}"
        start = bucket.player_slot_start(unit.player)
        capacity = bucket.per_player_capacity
    offset = next_free[unit.player + 1]
    if offset >= capacity:
        raise ContentError(
            f"the {block} block holds {capacity} slots, which unit ID {unit.id} exceeds"
        )
    next_free[unit.player + 1] = offset + 1
    return start + offset


def _placed(array: Array, index: Array, values: Sequence[int]) -> Array:
    return array.at[index].set(jnp.asarray(list(values), array.dtype))


def game_config(document: ContentDocument, table: tuple[UnitType, ...], bucket: BucketSpec, max_cycles: int) -> GameConfig:
    walls = [character == "1" for character in document.terrain]
    return GameConfig(
        terrain=jnp.asarray(walls, jnp.bool_).reshape(bucket.height, bucket.width),
        max_cycles=jnp.asarray(max_cycles, jnp.int32),
        initial_state=_initial_state(document, table, bucket),
        unit_types=table,
        bucket=bucket,
    )
