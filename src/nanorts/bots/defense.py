from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from nanorts.bots import abstraction, rush
from nanorts.bots.abstraction import AbstractActions, Units
from nanorts.model import ABSENT, TYPE_BASE, GameConfig, State


def _defend(actions: AbstractActions, units: Units, game: GameConfig, defenders: Array, enemy: rush.Closest) -> AbstractActions:
    width = game.bucket.width
    reach = game.bucket.height // 2
    bases = units.present & (units.unit_type == TYPE_BASE)
    base = units.cell[jnp.argmax(jnp.where(bases, units.birth, ABSENT))]
    base_distance = jnp.where(
        jnp.any(bases),
        jnp.abs(units.cell % width - base % width)
        + jnp.abs(units.cell // width - base // width),
        0,
    )
    engages = enemy.found & ((enemy.distance < reach) | (base_distance < reach))
    return abstraction.put(
        actions,
        units,
        defenders,
        False,
        abstraction.ATTACK,
        target=jnp.where(engages, enemy.slot, ABSENT),
        target_birth=jnp.where(engages, enemy.birth, ABSENT),
    )


def _melee_units_defend(actions: AbstractActions, units: Units, game: GameConfig, enemy: rush.Closest) -> AbstractActions:
    return _defend(actions, units, game, rush.available_melee(units, game), enemy)


def _harvest(actions: AbstractActions, units: Units, state: State, game: GameConfig, player: Array | int, harvesters: Array) -> AbstractActions:
    resource, stockpile = rush.harvest_sites(units, state, game, player)
    unchanged = (
        abstraction.is_entry(actions, units, abstraction.HARVEST)
        & (actions.target == resource.slot)
        & (actions.target_birth == resource.birth)
        & (actions.base == stockpile.slot)
        & (actions.base_birth == stockpile.birth)
    )
    return abstraction.put(
        actions,
        units,
        harvesters & resource.found & stockpile.found & ~unchanged,
        False,
        abstraction.HARVEST,
        target=resource.slot,
        target_birth=resource.birth,
        base=stockpile.slot,
        base_birth=stockpile.birth,
    )


def worker_defense(actions: AbstractActions, state: State, game: GameConfig, player: Array | int) -> AbstractActions:
    units = abstraction.own_units(state, game, player)
    resources = state.player_resources[player]
    enemy = rush.nearest_enemy(units, state, game, player)
    actions = rush.train_workers(actions, units, game, resources)
    actions = _melee_units_defend(actions, units, game, enemy)
    workers = units.present & game.unit_can_harvest[units.unit_type]
    actions, builder, _ = rush.build_base(actions, units, state, game, player, workers)
    harvester = rush.first_in_unit_order(workers & ~builder, units)
    actions = _harvest(actions, units, state, game, player, harvester)
    return _defend(actions, units, game, workers & ~builder & ~harvester, enemy)


def barracks_defense(actions: AbstractActions, state: State, game: GameConfig, player: Array | int, trained_type: int) -> AbstractActions:
    units = abstraction.own_units(state, game, player)
    resources = state.player_resources[player]
    enemy = rush.nearest_enemy(units, state, game, player)
    actions = rush.train_at_base_and_barracks(
        actions, units, game, resources, trained_type
    )
    actions = _melee_units_defend(actions, units, game, enemy)
    actions, free = rush.build_base_and_barracks(actions, units, state, game, player)
    return _harvest(actions, units, state, game, player, free)
