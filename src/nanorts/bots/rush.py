from __future__ import annotations

from typing import Final, NamedTuple

import jax.numpy as jnp
from jax import Array

from nanorts.bots import abstraction
from nanorts.bots.abstraction import LAST, AbstractActions, Units
from nanorts.model import (
    ABSENT,
    TYPE_BARRACKS,
    TYPE_BASE,
    TYPE_WORKER,
    GameConfig,
    State,
)

KINDS: Final[frozenset[int]] = frozenset(
    (abstraction.ATTACK, abstraction.HARVEST, abstraction.TRAIN, abstraction.BUILD)
)


class Closest(NamedTuple):
    slot: Array
    birth: Array
    found: Array
    distance: Array


def _closest(origin: Array, candidates: Array, state: State, width: int) -> Closest:
    cells = state.cell.astype(jnp.int32)
    distance = jnp.abs(origin[..., None] % width - cells % width) + jnp.abs(
        origin[..., None] // width - cells // width
    )
    distance = jnp.where(candidates, distance, LAST)
    nearest = jnp.min(distance, axis=-1, keepdims=True)
    tied = candidates & (distance == nearest)
    slot = jnp.argmin(jnp.where(tied, state.birth, LAST), axis=-1).astype(jnp.int32)
    found = jnp.broadcast_to(jnp.any(candidates, axis=-1), slot.shape)
    return Closest(
        slot=jnp.where(found, slot, ABSENT),
        birth=jnp.where(found, state.birth[slot], ABSENT),
        found=found,
        distance=nearest[..., 0],
    )


def nearest_enemy(units: Units, state: State, game: GameConfig, player: Array | int) -> Closest:
    return _closest(
        units.cell, abstraction.enemies(state, player), state, game.bucket.width
    )


def first_in_unit_order(chosen: Array, units: Units) -> Array:
    first = jnp.argmin(jnp.where(chosen, units.birth, LAST))
    return chosen & (jnp.arange(chosen.shape[0]) == first)


def _attack_nearest(actions: AbstractActions, units: Units, attackers: Array, last: Array | bool, enemy: Closest) -> AbstractActions:
    return abstraction.put(
        actions,
        units,
        attackers & enemy.found,
        last,
        abstraction.ATTACK,
        target=enemy.slot,
        target_birth=enemy.birth,
    )


def available_melee(units: Units, game: GameConfig) -> Array:
    return (
        units.present
        & game.unit_can_attack[units.unit_type]
        & ~game.unit_can_harvest[units.unit_type]
        & units.available
    )


def _melee_units_attack(actions: AbstractActions, units: Units, game: GameConfig, enemy: Closest) -> AbstractActions:
    melee = available_melee(units, game)
    return _attack_nearest(actions, units, melee, False, enemy)


def train_workers(actions: AbstractActions, units: Units, game: GameConfig, resources: Array) -> AbstractActions:
    trains = (
        units.present
        & (units.unit_type == TYPE_BASE)
        & units.available
        & (resources >= game.unit_cost[TYPE_WORKER])
    )
    return abstraction.put(
        actions, units, trains, False, abstraction.TRAIN, unit_type=TYPE_WORKER
    )


def build_base(actions: AbstractActions, units: Units, state: State, game: GameConfig, player: Array | int, workers: Array) -> tuple[AbstractActions, Array, Array]:
    bases = units.present & (units.unit_type == TYPE_BASE)
    builds = ~jnp.any(bases) & (
        state.player_resources[player] >= game.unit_cost[TYPE_BASE]
    )
    builder = first_in_unit_order(workers, units) & builds
    actions, reserved = abstraction.build_if_not_already_building(
        actions,
        units,
        state,
        game,
        builder,
        TYPE_BASE,
        jnp.zeros(game.bucket.cell_count, jnp.bool_),
    )
    return actions, builder, reserved


def train_at_base_and_barracks(actions: AbstractActions, units: Units, game: GameConfig, resources: Array, trained_type: int) -> AbstractActions:
    workers_held = jnp.sum(
        units.present & (units.unit_type == TYPE_WORKER), dtype=jnp.int32
    )
    trains_worker = (
        units.present
        & (units.unit_type == TYPE_BASE)
        & units.available
        & (workers_held < 1)
        & (resources >= game.unit_cost[TYPE_WORKER])
    )
    actions = abstraction.put(
        actions,
        units,
        trains_worker,
        False,
        abstraction.TRAIN,
        unit_type=TYPE_WORKER,
    )
    barracks_train = (
        units.present
        & (units.unit_type == TYPE_BARRACKS)
        & units.available
        & (resources >= game.unit_cost[trained_type])
    )
    return abstraction.put(
        actions,
        units,
        barracks_train,
        False,
        abstraction.TRAIN,
        unit_type=trained_type,
    )


def harvest_sites(units: Units, state: State, game: GameConfig, player: Array | int) -> tuple[Closest, Closest]:
    width = game.bucket.width
    every_type = jnp.clip(
        state.unit_type.astype(jnp.int32), 0, game.bucket.unit_type_capacity - 1
    )
    resource = _closest(
        units.cell, state.active & game.unit_is_resource[every_type], state, width
    )
    stockpile = _closest(
        units.cell,
        state.active
        & game.unit_is_stockpile[every_type]
        & (state.owner.astype(jnp.int32) == jnp.asarray(player)),
        state,
        width,
    )
    return resource, stockpile


def _harvest(actions: AbstractActions, units: Units, state: State, game: GameConfig, player: Array | int, harvesters: Array) -> tuple[AbstractActions, Array]:
    resource, stockpile = harvest_sites(units, state, game, player)
    carrying = units.carried > 0
    harvesting = abstraction.is_entry(actions, units, abstraction.HARVEST)
    same_base = (actions.base == stockpile.slot) & (
        actions.base_birth == stockpile.birth
    )
    same_target = (actions.target == resource.slot) & (
        actions.target_birth == resource.birth
    )
    occupied = jnp.where(carrying, stockpile.found, stockpile.found & resource.found)
    unchanged = harvesting & same_base & (carrying | same_target)
    actions = abstraction.put(
        actions,
        units,
        harvesters & occupied & ~unchanged,
        False,
        abstraction.HARVEST,
        target=jnp.where(carrying, ABSENT, resource.slot),
        target_birth=jnp.where(carrying, ABSENT, resource.birth),
        base=stockpile.slot,
        base_birth=stockpile.birth,
    )
    return actions, harvesters & ~occupied


def build_base_and_barracks(actions: AbstractActions, units: Units, state: State, game: GameConfig, player: Array | int) -> tuple[AbstractActions, Array]:
    resources = state.player_resources[player]
    workers = units.present & game.unit_can_harvest[units.unit_type]
    barracks = units.present & (units.unit_type == TYPE_BARRACKS)
    actions, base_builder, reserved = build_base(
        actions, units, state, game, player, workers
    )
    used = jnp.where(jnp.any(base_builder), game.unit_cost[TYPE_BASE], 0)
    builds_barracks = ~jnp.any(barracks) & (
        resources >= game.unit_cost[TYPE_BARRACKS] + used
    )
    barracks_builder = (
        first_in_unit_order(workers & ~base_builder, units) & builds_barracks
    )
    actions, _ = abstraction.build_if_not_already_building(
        actions, units, state, game, barracks_builder, TYPE_BARRACKS, reserved
    )
    return actions, workers & ~base_builder & ~barracks_builder


def _rushing_workers(actions: AbstractActions, units: Units, state: State, game: GameConfig, player: Array | int, enemy: Closest) -> AbstractActions:
    workers = units.present & game.unit_can_harvest[units.unit_type]
    actions, builder, _ = build_base(actions, units, state, game, player, workers)
    harvester = first_in_unit_order(workers & ~builder, units)
    actions, idle_harvester = _harvest(actions, units, state, game, player, harvester)
    free = (workers & ~builder & ~harvester) | idle_harvester
    return _attack_nearest(actions, units, free, idle_harvester, enemy)


def worker_rush(actions: AbstractActions, state: State, game: GameConfig, player: Array | int) -> AbstractActions:
    units = abstraction.own_units(state, game, player)
    resources = state.player_resources[player]
    enemy = nearest_enemy(units, state, game, player)
    actions = train_workers(actions, units, game, resources)
    actions = _melee_units_attack(actions, units, game, enemy)
    return _rushing_workers(actions, units, state, game, player, enemy)


def _workers_behavior(actions: AbstractActions, units: Units, state: State, game: GameConfig, player: Array | int, enemy: Closest) -> AbstractActions:
    actions, free = build_base_and_barracks(actions, units, state, game, player)
    actions, idle = _harvest(actions, units, state, game, player, free)
    return _attack_nearest(actions, units, idle, False, enemy)


def barracks_rush(actions: AbstractActions, state: State, game: GameConfig, player: Array | int, trained_type: int) -> AbstractActions:
    units = abstraction.own_units(state, game, player)
    resources = state.player_resources[player]
    enemy = nearest_enemy(units, state, game, player)
    actions = train_at_base_and_barracks(actions, units, game, resources, trained_type)
    actions = _melee_units_attack(actions, units, game, enemy)
    return _workers_behavior(actions, units, state, game, player, enemy)
