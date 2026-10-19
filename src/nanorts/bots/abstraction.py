from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Sequence
from typing import Final, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, lax

from nanorts import kernels
from nanorts.bots import pathfinding
from nanorts.model import (
    ABSENT,
    DIRECTION_COUNT,
    DIRECTION_OFFSETS,
    KIND_ABSENT,
    KIND_ATTACK,
    KIND_HARVEST,
    KIND_MOVE,
    KIND_NONE,
    KIND_PRODUCE,
    KIND_RETURN,
    PLAYER_COUNT,
    BucketSpec,
    Commands,
    GameConfig,
    State,
    empty,
)

_NO_ACTION: Final[int] = ABSENT
MOVE: Final[int] = 0
ATTACK: Final[int] = 1
HARVEST: Final[int] = 2
TRAIN: Final[int] = 3
BUILD: Final[int] = 4
_IDLE: Final[int] = 5
TRAIN_DIRECTION: Final[int] = 6
BUILD_MODIFIED: Final[int] = 7
COAC_ATTACK: Final[int] = 8
TRUE_IDLE: Final[int] = 9
HARVEST_RETURN: Final[int] = 10

_EVERY_KIND: Final[frozenset[int]] = frozenset(
    (
        MOVE,
        ATTACK,
        HARVEST,
        TRAIN,
        BUILD,
        _IDLE,
        TRAIN_DIRECTION,
        BUILD_MODIFIED,
        COAC_ATTACK,
        TRUE_IDLE,
        HARVEST_RETURN,
    )
)

LAST: Final[int] = int(jnp.iinfo(jnp.int32).max)
_FILLER_DURATION: Final[int] = 10
_WAIT_ONE: Final[int] = 1
_PUTS_PER_CYCLE: Final[int] = 64
_STEP_X: Final[tuple[int, ...]] = tuple(dx for dx, _ in DIRECTION_OFFSETS)
_STEP_Y: Final[tuple[int, ...]] = tuple(dy for _, dy in DIRECTION_OFFSETS)


class AbstractActions(NamedTuple):
    kind: Array
    unit_birth: Array
    order: Array
    target: Array
    target_birth: Array
    base: Array
    base_birth: Array
    cell: Array
    unit_type: Array
    trained: Array
    next_order: Array


class Units(NamedTuple):
    birth: Array
    cell: Array
    unit_type: Array
    carried: Array
    present: Array
    available: Array
    birth_rank: Array


class _Usage(NamedTuple):
    claimed: Array
    reserved: Array
    balance: Array


class _Desire(NamedTuple):
    kind: Array
    direction: Array
    target_cell: Array
    produced_type: Array
    claimed: Array
    cost: Array
    none_duration: Array


class _Plan(NamedTuple):
    start: Array
    goal: Array
    radius: Array
    draws: Array
    options: Array
    fallback: _Desire
    blocked: _Desire


def empty_actions(game: GameConfig) -> AbstractActions:
    capacity = game.bucket.per_player_capacity
    absent = jnp.full(capacity, ABSENT, jnp.int32)
    return AbstractActions(
        kind=absent,
        unit_birth=absent,
        order=absent,
        target=absent,
        target_birth=absent,
        base=absent,
        base_birth=absent,
        cell=absent,
        unit_type=absent,
        trained=jnp.zeros(capacity, jnp.bool_),
        next_order=jnp.int32(0),
    )


def own_units(state: State, game: GameConfig, player: Array | int) -> Units:
    bucket = game.bucket
    capacity = bucket.per_player_capacity
    first = bucket.neutral_capacity + jnp.asarray(player, jnp.int32) * capacity

    def owned(leaf: Array) -> Array:
        return lax.dynamic_slice_in_dim(leaf, first, capacity)

    birth = owned(state.birth)
    present = birth != ABSENT
    return Units(
        birth=birth,
        cell=owned(state.cell).astype(jnp.int32),
        unit_type=jnp.clip(
            owned(state.unit_type).astype(jnp.int32), 0, bucket.unit_type_capacity - 1
        ),
        carried=owned(state.carried_resources),
        present=present,
        available=present & (owned(state.pending_kind) == KIND_ABSENT),
        birth_rank=_rank(present, birth),
    )


def enemies(state: State, player: Array | int) -> Array:
    owner = state.owner.astype(jnp.int32)
    return state.active & (owner != ABSENT) & (owner != jnp.asarray(player))


def is_entry(actions: AbstractActions, units: Units, kind: int) -> Array:
    return (actions.kind == kind) & (actions.unit_birth == units.birth)


def put(actions: AbstractActions, units: Units, chosen: Array, last: Array | bool, kind: Array | int, *, target: Array | int = ABSENT, target_birth: Array | int = ABSENT, base: Array | int = ABSENT, base_birth: Array | int = ABSENT, cell: Array | int = ABSENT, unit_type: Array | int = ABSENT) -> AbstractActions:
    capacity = chosen.shape[0]
    assert (_PUTS_PER_CYCLE + 2) * (capacity + 1) < LAST
    written, _ = _written(
        actions,
        units,
        chosen,
        actions.next_order + jnp.where(last, capacity, units.birth_rank),
        kind,
        target=target,
        target_birth=target_birth,
        base=base,
        base_birth=base_birth,
        cell=cell,
        unit_type=unit_type,
    )
    return written._replace(next_order=actions.next_order + capacity + 1)


def put_one(actions: AbstractActions, units: Units, chosen: Array, kind: Array | int, **fields: Array | int) -> AbstractActions:
    written, fresh = _written(
        actions, units, chosen, actions.next_order, kind, **fields
    )
    return written._replace(
        next_order=actions.next_order + jnp.sum(fresh, dtype=jnp.int32)
    )


def _written(actions: AbstractActions, units: Units, chosen: Array, stamp: Array, kind: Array | int, *, target: Array | int = ABSENT, target_birth: Array | int = ABSENT, base: Array | int = ABSENT, base_birth: Array | int = ABSENT, cell: Array | int = ABSENT, unit_type: Array | int = ABSENT) -> tuple[AbstractActions, Array]:
    known = (actions.kind != _NO_ACTION) & (actions.unit_birth == units.birth)
    fresh = chosen & ~known

    def chose(new: Array | int, old: Array) -> Array:
        return jnp.where(chosen, jnp.asarray(new, jnp.int32), old)

    written = AbstractActions(
        kind=chose(kind, actions.kind),
        unit_birth=chose(units.birth, actions.unit_birth),
        order=jnp.where(fresh, stamp, actions.order),
        target=chose(target, actions.target),
        target_birth=chose(target_birth, actions.target_birth),
        base=chose(base, actions.base),
        base_birth=chose(base_birth, actions.base_birth),
        cell=chose(cell, actions.cell),
        unit_type=chose(unit_type, actions.unit_type),
        trained=jnp.where(chosen, False, actions.trained),
        next_order=actions.next_order,
    )
    return written, fresh


@functools.cache
def _visits(height: int, width: int) -> tuple[tuple[int, ...], ...]:
    longest = max(height, width)
    rows = []
    for dy in range(1 - height, height):
        row = []
        for dx in range(1 - width, width):
            ring = max(abs(dx), abs(dy))
            side = 0 if dy == -ring else 1 if dx == ring else 2 if dy == ring else 3
            along = (dx if side in (0, 2) else dy) + ring
            row.append(
                LAST if ring == 0 else (ring * 4 + side) * (2 * longest + 1) + along
            )
        rows.append(tuple(row))
    return tuple(rows)


def _building_site(state: State, game: GameConfig, desired: Array, reserved: Array) -> Array:
    bucket = game.bucket
    height, width = bucket.height, bucket.width
    visit = lax.dynamic_slice(
        jnp.asarray(_visits(height, width), jnp.int32),
        (height - 1 - desired // width, width - 1 - desired % width),
        (height, width),
    ).reshape(-1)
    open_cells = (
        ~game.terrain.reshape(-1)
        & (state.entity_at.reshape(-1) == ABSENT)
        & ~reserved
        & (visit != LAST)
    )
    first = jnp.argmin(jnp.where(open_cells, visit, LAST)).astype(jnp.int32)
    return jnp.where(jnp.any(open_cells), first, ABSENT)


def build_if_not_already_building(actions: AbstractActions, units: Units, state: State, game: GameConfig, builder: Array, unit_type: int, reserved: Array) -> tuple[AbstractActions, Array]:
    building = is_entry(actions, units, BUILD) & (actions.unit_type == unit_type)
    builds = builder & ~building
    desired = jnp.sum(jnp.where(builder, units.cell, 0), dtype=jnp.int32)
    site = _building_site(state, game, desired, reserved)
    cells = jnp.arange(game.bucket.cell_count, dtype=jnp.int32)
    return (
        put(actions, units, builds, False, BUILD, cell=site, unit_type=unit_type),
        reserved | (jnp.any(builds) & (cells == site)),
    )


def translate(actions: AbstractActions, state: State, game: GameConfig, player: Array | int, *, kinds: frozenset[int] = _EVERY_KIND) -> tuple[AbstractActions, Commands]:
    actions, commands, _ = _translate(
        actions, None, state, game, player, _FILLER_DURATION, kinds
    )
    return actions, commands


def translate_wait1(actions: AbstractActions, draws: Array, state: State, game: GameConfig, player: Array | int, *, kinds: frozenset[int] = _EVERY_KIND) -> tuple[AbstractActions, Commands, Array]:
    return _translate(actions, draws, state, game, player, _WAIT_ONE, kinds)


def _translate(actions: AbstractActions, draws: Array | None, state: State, game: GameConfig, player: Array | int, wait: int, kinds: frozenset[int]) -> tuple[AbstractActions, Commands, Array]:
    units = own_units(state, game, player)
    usage = _pending_usage(state, game)
    standing = _standing(actions, units, state, kinds)
    executes = standing & units.available
    plan = _plan(
        actions, units, state, game, usage, player, draws is not None, kinds=kinds
    )
    desires, ranks, kept_count, drew = _execute_in_order(
        plan, executes, actions.order, draws, state, game, usage, player
    )
    return (
        _settled(actions, standing, executes),
        _commands(desires, ranks, kept_count, units, player, game, wait),
        drew,
    )


def _rank(among: Array, key: Array) -> Array:
    earlier = among[None, :] & (key[None, :] < key[:, None])
    return jnp.sum(earlier, axis=1, dtype=jnp.int32)


def _held(kind: Array, members: tuple[int, ...], kinds: frozenset[int]) -> Array | None:
    tests = [kind == member for member in members if member in kinds]
    return functools.reduce(jnp.logical_or, tests) if tests else None


def _one_of(kind: Array, members: tuple[int, ...], kinds: frozenset[int]) -> Array:
    held = _held(kind, members, kinds)
    return jnp.zeros(kind.shape, jnp.bool_) if held is None else held


def _select_held(cases: Sequence[tuple[Array | None, Array]], default: int | bool, like: Array) -> Array:
    held = [(test, value) for test, value in cases if test is not None]
    if not held:
        return jnp.full_like(like, default)
    return jnp.select([test for test, _ in held], [value for _, value in held], default)


def _neighbour(cell: Array, direction: Array, width: int) -> Array:
    known = jnp.clip(direction, 0, DIRECTION_COUNT - 1)
    return (
        cell
        + jnp.asarray(_STEP_Y, jnp.int32)[known] * width
        + jnp.asarray(_STEP_X, jnp.int32)[known]
    )


def _squared_distance(cell: Array, other: Array, width: int) -> Array:
    dx = cell % width - other % width
    dy = cell // width - other // width
    return dx * dx + dy * dy


def _pending_usage(state: State, game: GameConfig) -> _Usage:
    bucket = game.bucket
    kind = state.pending_kind.astype(jnp.int32)
    claims = state.active & ((kind == KIND_MOVE) | (kind == KIND_PRODUCE))
    target = _neighbour(
        state.cell.astype(jnp.int32),
        state.pending_direction.astype(jnp.int32),
        bucket.width,
    )
    claimed = (
        jnp.zeros(bucket.cell_count, jnp.bool_)
        .at[jnp.where(claims, target, bucket.cell_count)]
        .set(True, mode="drop")
    )
    produced = jnp.clip(
        state.pending_produced_type.astype(jnp.int32), 0, bucket.unit_type_capacity - 1
    )
    cost = jnp.where(
        state.active & (kind == KIND_PRODUCE), game.unit_cost[produced], 0
    ).astype(jnp.int32)
    reserved = jnp.sum(
        cost[bucket.neutral_capacity :].reshape(
            PLAYER_COUNT, bucket.per_player_capacity
        ),
        axis=-1,
        dtype=jnp.int32,
    )
    return _Usage(claimed=claimed, reserved=reserved, balance=state.player_resources)


def _free_cells(state: State, game: GameConfig, usage: _Usage) -> Array:
    cells = game.bucket.cell_count
    occupied = state.entity_at.reshape(cells) != ABSENT
    return ~(game.terrain.reshape(cells) | occupied | usage.claimed)


def _within_balance(usage: _Usage) -> Array:
    return (usage.reserved == 0) | (usage.reserved <= usage.balance)


def _produce_allowed(usage: _Usage, cost: Array, cell: Array, player: Array | int) -> Array:
    own = jnp.asarray(player)
    cell_count = usage.claimed.shape[0]
    reserved = usage.reserved[own]
    return (
        ~usage.claimed[jnp.clip(cell, 0, cell_count - 1)]
        & ((reserved == 0) | (cost + reserved <= usage.balance[own]))
        & _within_balance(usage)[1 - own]
    )


def _exists(slot: Array, birth: Array, state: State) -> Array:
    known = jnp.clip(slot, 0, state.birth.shape[0] - 1)
    return (slot != ABSENT) & (state.birth[known] == birth)


def _cell_of(slot: Array, state: State) -> Array:
    known = jnp.clip(slot, 0, state.cell.shape[0] - 1)
    return jnp.where(slot != ABSENT, state.cell.astype(jnp.int32)[known], ABSENT)


def _standing(actions: AbstractActions, units: Units, state: State, kinds: frozenset[int]) -> Array:
    kind = actions.kind
    alive = (kind != _NO_ACTION) & (actions.unit_birth == units.birth)
    carrying = units.carried > 0
    occupants = state.entity_at.reshape(-1)
    built_on = (actions.cell != ABSENT) & (
        occupants[jnp.clip(actions.cell, 0, occupants.shape[0] - 1)] != ABSENT
    )
    target_gone = ~_exists(actions.target, actions.target_birth, state)
    completed = _select_held(
        [
            (_held(kind, (MOVE,), kinds), units.cell == actions.cell),
            (_held(kind, (ATTACK, COAC_ATTACK), kinds), target_gone),
            (
                _held(kind, (HARVEST,), kinds),
                jnp.where(
                    carrying,
                    ~_exists(actions.base, actions.base_birth, state),
                    target_gone,
                ),
            ),
            (_held(kind, (TRAIN, TRAIN_DIRECTION), kinds), actions.trained),
            (_held(kind, (BUILD, BUILD_MODIFIED), kinds), built_on),
            (_held(kind, (HARVEST_RETURN,), kinds), target_gone & ~carrying),
        ],
        False,
        alive,
    )
    return alive & ~completed


def _adjacent_direction(origin: Array, goal: Array, width: int) -> Array:
    dx = goal % width - origin % width
    dy = goal // width - origin // width
    matches = (dx[..., None] == jnp.asarray(_STEP_X, jnp.int32)) & (
        dy[..., None] == jnp.asarray(_STEP_Y, jnp.int32)
    )
    found = (goal != ABSENT) & jnp.any(matches, axis=-1)
    return jnp.where(found, jnp.argmax(matches, axis=-1), ABSENT).astype(jnp.int32)


def _no_desire(shape: tuple[int, ...]) -> _Desire:
    absent = jnp.full(shape, ABSENT, jnp.int32)
    return _Desire(
        kind=jnp.full(shape, KIND_ABSENT, jnp.int32),
        direction=absent,
        target_cell=absent,
        produced_type=absent,
        claimed=absent,
        cost=jnp.zeros(shape, jnp.int32),
        none_duration=jnp.zeros(shape, jnp.int32),
    )


def _waiting(shape: tuple[int, ...]) -> _Desire:
    return _no_desire(shape)._replace(
        kind=jnp.full(shape, KIND_NONE, jnp.int32),
        none_duration=jnp.full(shape, _WAIT_ONE, jnp.int32),
    )


def _choose(chosen: Array, desire: _Desire, otherwise: _Desire) -> _Desire:
    return _Desire(
        *(
            jnp.where(chosen, mine, theirs).astype(jnp.int32)
            for mine, theirs in zip(desire, otherwise, strict=True)
        )
    )


def _production(direction: Array, produced_type: Array, cell: Array, game: GameConfig, usage: _Usage, player: Array | int) -> _Desire:
    width = game.bucket.width
    claimed = jnp.where(direction != ABSENT, _neighbour(cell, direction, width), ABSENT)
    cost = game.unit_cost[
        jnp.clip(produced_type, 0, game.bucket.unit_type_capacity - 1)
    ]
    allowed = (direction != ABSENT) & _produce_allowed(usage, cost, claimed, player)
    return _choose(
        allowed,
        _Desire(
            kind=jnp.full_like(direction, KIND_PRODUCE),
            direction=direction,
            target_cell=jnp.full_like(direction, ABSENT),
            produced_type=produced_type,
            claimed=claimed,
            cost=cost,
            none_duration=jnp.zeros_like(direction),
        ),
        _no_desire(direction.shape),
    )


def _training_direction(actions: AbstractActions, units: Units, state: State, game: GameConfig, player: Array | int, free: Array) -> Array:
    bucket = game.bucket
    width, height = bucket.width, bucket.height
    x = units.cell[:, None] % width + jnp.asarray(_STEP_X, jnp.int32)
    y = units.cell[:, None] // width + jnp.asarray(_STEP_Y, jnp.int32)
    on_board = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    neighbour = jnp.where(on_board, y * width + x, 0)
    open_cells = on_board & free[neighbour]
    every_type = jnp.clip(
        state.unit_type.astype(jnp.int32), 0, bucket.unit_type_capacity - 1
    )
    resource = state.active & game.unit_is_resource[every_type]
    enemy = enemies(state, player)
    harvests = game.unit_can_harvest[
        jnp.clip(actions.unit_type, 0, bucket.unit_type_capacity - 1)
    ]
    near = jnp.where(harvests[:, None], resource[None, :], enemy[None, :])
    cells = state.cell.astype(jnp.int32)
    distance = jnp.abs(x[..., None] - cells % width) + jnp.abs(
        y[..., None] - cells // width
    )
    closest = jnp.min(jnp.where(near[:, None, :], distance, LAST), axis=-1)
    score = jnp.where(jnp.any(near, axis=-1)[:, None], -closest, 0)
    best = jnp.argmax(jnp.where(open_cells, score, -LAST), axis=-1).astype(jnp.int32)
    return jnp.where(jnp.any(open_cells, axis=-1), best, ABSENT)


def _idle_target(units: Units, state: State, game: GameConfig, player: Array | int) -> Array:
    width = game.bucket.width
    enemy = enemies(state, player)
    cells = state.cell.astype(jnp.int32)
    attack_range = game.unit_attack_range[units.unit_type]
    in_range = enemy[None, :] & (
        _squared_distance(units.cell[:, None], cells[None, :], width)
        <= (attack_range * attack_range)[:, None]
    )
    first = jnp.argmin(jnp.where(in_range, state.birth[None, :], LAST), axis=-1)
    found = game.unit_can_attack[units.unit_type] & jnp.any(in_range, axis=-1)
    return jnp.where(found, cells[first], ABSENT)


def _attack(target_cell: Array) -> _Desire:
    absent = jnp.full_like(target_cell, ABSENT)
    return _choose(
        target_cell != ABSENT,
        _Desire(
            kind=jnp.full_like(target_cell, KIND_ATTACK),
            direction=absent,
            target_cell=target_cell,
            produced_type=absent,
            claimed=absent,
            cost=jnp.zeros_like(target_cell),
            none_duration=jnp.zeros_like(target_cell),
        ),
        _no_desire(target_cell.shape),
    )


def _next_cell(state: State, game: GameConfig) -> Array:
    cell = state.cell.astype(jnp.int32)
    moving = state.active & (state.pending_kind == KIND_MOVE)
    return jnp.where(
        moving,
        _neighbour(cell, state.pending_direction.astype(jnp.int32), game.bucket.width),
        cell,
    )


def _move_options(units: Units, game: GameConfig, usage: _Usage, free: Array) -> Array:
    bucket = game.bucket
    width, height = bucket.width, bucket.height
    x = units.cell[:, None] % width + jnp.asarray(_STEP_X, jnp.int32)
    y = units.cell[:, None] // width + jnp.asarray(_STEP_Y, jnp.int32)
    on_board = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    passes = (
        on_board
        & free[jnp.where(on_board, y * width + x, 0)]
        & jnp.all(_within_balance(usage))
    )
    bits = jnp.left_shift(1, jnp.arange(DIRECTION_COUNT, dtype=jnp.int32))
    return jnp.sum(jnp.where(passes, bits, 0), axis=-1, dtype=jnp.int32)


class _Reach(NamedTuple):
    target_cell: Array
    in_range: Array
    coming_into_range: Array


def _reach(actions: AbstractActions, units: Units, state: State, game: GameConfig) -> _Reach:
    width = game.bucket.width
    target_cell = _cell_of(actions.target, state)
    attack_range = game.unit_attack_range[units.unit_type]
    target_next = jnp.where(
        actions.target != ABSENT,
        _next_cell(state, game)[jnp.clip(actions.target, 0, state.cell.shape[0] - 1)],
        ABSENT,
    )
    return _Reach(
        target_cell=target_cell,
        in_range=_squared_distance(units.cell, target_cell, width)
        <= attack_range * attack_range,
        coming_into_range=_squared_distance(units.cell, target_next, width)
        <= attack_range * attack_range,
    )


def _fallback(actions: AbstractActions, units: Units, state: State, game: GameConfig, usage: _Usage, player: Array | int, reach: _Reach, harvest_goal: Array, free: Array, kinds: frozenset[int]) -> _Desire:
    width = game.bucket.width
    kind = actions.kind
    carrying = units.carried > 0

    @functools.cache
    def waiting() -> _Desire:
        return _waiting(kind.shape)

    @functools.cache
    def harvest() -> _Desire:
        harvest_direction = _adjacent_direction(units.cell, harvest_goal, width)
        absent = jnp.full_like(kind, ABSENT)
        gather = _Desire(
            kind=jnp.where(carrying, KIND_RETURN, KIND_HARVEST),
            direction=harvest_direction,
            target_cell=absent,
            produced_type=absent,
            claimed=absent,
            cost=jnp.zeros_like(kind),
            none_duration=jnp.zeros_like(kind),
        )
        return _choose(harvest_direction != ABSENT, gather, _no_desire(kind.shape))

    @functools.cache
    def adjacent_production() -> _Desire:
        return _production(
            _adjacent_direction(units.cell, actions.cell, width),
            actions.unit_type,
            units.cell,
            game,
            usage,
            player,
        )

    @functools.cache
    def produce_or_wait() -> _Desire:
        produced = adjacent_production()
        return _choose(produced.kind != KIND_ABSENT, produced, waiting())

    @functools.cache
    def strike() -> _Desire:
        return _attack(jnp.where(reach.in_range, reach.target_cell, ABSENT))

    def train() -> _Desire:
        return _production(
            _training_direction(actions, units, state, game, player, free),
            actions.unit_type,
            units.cell,
            game,
            usage,
            player,
        )

    fallbacks: dict[int, Callable[[], _Desire]] = {
        ATTACK: strike,
        HARVEST: harvest,
        TRAIN: train,
        BUILD: adjacent_production,
        _IDLE: lambda: _attack(_idle_target(units, state, game, player)),
        TRAIN_DIRECTION: produce_or_wait,
        BUILD_MODIFIED: produce_or_wait,
        COAC_ATTACK: lambda: _choose(reach.in_range, strike(), waiting()),
        TRUE_IDLE: waiting,
        HARVEST_RETURN: harvest,
    }
    fallback = _no_desire(kind.shape)
    for abstract_kind, desire in fallbacks.items():
        if abstract_kind in kinds:
            fallback = _choose(kind == abstract_kind, desire(), fallback)
    return fallback


def _plan(actions: AbstractActions, units: Units, state: State, game: GameConfig, usage: _Usage, player: Array | int, drawing: bool = False, *, kinds: frozenset[int] = _EVERY_KIND) -> _Plan:
    kind = actions.kind
    reach = _reach(actions, units, state, game)
    free = _free_cells(state, game, usage)
    attacks = _held(kind, (ATTACK, COAC_ATTACK), kinds)
    harvests = _held(kind, (HARVEST, HARVEST_RETURN), kinds)
    builds = _held(kind, (BUILD, BUILD_MODIFIED), kinds)
    waits_for_target = (
        _one_of(kind, (COAC_ATTACK,), kinds) & ~reach.in_range & reach.coming_into_range
    )
    harvest_goal = jnp.where(
        units.carried > 0, _cell_of(actions.base, state), reach.target_cell
    )
    goal = _select_held(
        [
            (_held(kind, (MOVE,), kinds), actions.cell),
            (attacks, reach.target_cell),
            (harvests, harvest_goal),
            (builds, actions.cell),
        ],
        ABSENT,
        kind,
    )
    attack_range = game.unit_attack_range[units.unit_type]
    paced = _held(kind, (HARVEST, HARVEST_RETURN, BUILD, BUILD_MODIFIED), kinds)
    radius = _select_held(
        [(attacks, attack_range), (paced, jnp.ones_like(attack_range))],
        0,
        attack_range,
    )
    searches = (
        (goal != ABSENT)
        & ~(_one_of(kind, (ATTACK, COAC_ATTACK), kinds) & reach.in_range)
        & ~waits_for_target
    )
    return _Plan(
        start=units.cell,
        goal=jnp.where(searches, goal, units.cell),
        radius=jnp.where(searches, radius, 0),
        draws=_one_of(kind, (COAC_ATTACK,), kinds) & searches,
        options=(
            _move_options(units, game, usage, free) if drawing else jnp.zeros_like(kind)
        ),
        fallback=_fallback(
            actions,
            units,
            state,
            game,
            usage,
            player,
            reach,
            harvest_goal,
            free,
            kinds,
        ),
        blocked=_choose(
            _one_of(kind, (BUILD_MODIFIED, COAC_ATTACK), kinds),
            _waiting(kind.shape),
            _no_desire(kind.shape),
        ),
    )


def _drawn(first: Array, options: Array, start: Array, width: int) -> _Desire:
    turned = (first + jnp.arange(DIRECTION_COUNT, dtype=jnp.int32)) % DIRECTION_COUNT
    passes = (jnp.right_shift(options, turned) & 1) == 1
    direction = jnp.where(jnp.any(passes), turned[jnp.argmax(passes)], ABSENT)
    move = _no_desire(())._replace(
        kind=jnp.int32(KIND_MOVE),
        direction=direction,
        claimed=_neighbour(start, direction, width),
    )
    return _choose(direction != ABSENT, move, _waiting(()))


class _Walk(NamedTuple):
    position: Array
    used: Array
    kept_cells: Array
    kept_cost: Array
    kept_count: Array
    decided: _Desire
    ranks: Array
    calls: Array
    drew: Array


_SEARCH_COLUMNS: Final[int] = 5


def _entry(rows: Array, index: Array) -> _Plan:
    row = lax.dynamic_slice_in_dim(rows, index, 1)[0]
    fields = len(_Desire._fields)
    return _Plan(
        start=row[0],
        goal=row[1],
        radius=row[2],
        draws=row[3] != 0,
        options=row[4],
        fallback=_Desire(*row[_SEARCH_COLUMNS : _SEARCH_COLUMNS + fields]),
        blocked=_Desire(*row[_SEARCH_COLUMNS + fields :]),
    )


def _answer(entry: _Plan, walk: _Walk, path: pathfinding.Path, draws: Array | None, moves_allowed: Array, bucket: BucketSpec) -> tuple[_Desire, Array]:
    step = _no_desire(())._replace(
        kind=jnp.int32(KIND_MOVE),
        direction=path.step,
        claimed=_neighbour(entry.start, path.step, bucket.width),
    )
    stepped = path.step != ABSENT
    desire = _choose(
        stepped, _choose(moves_allowed, step, entry.blocked), entry.fallback
    )
    if draws is None:
        return desire, jnp.bool_(False)
    draws_here = entry.draws & ~(stepped & moves_allowed)
    first = draws[jnp.clip(walk.calls, 0, draws.shape[0] - 1)]
    drawn = _drawn(first, entry.options, entry.start, bucket.width)
    return _choose(draws_here, drawn, desire), draws_here


def _kept(walk: _Walk, desire: _Desire, usage: _Usage, own: Array, cell_count: int) -> tuple[Array, Array]:
    claimed_here = (desire.claimed != ABSENT) & (
        jnp.arange(cell_count, dtype=jnp.int32) == desire.claimed
    )
    conflicting = jnp.any(claimed_here & (usage.claimed | walk.kept_cells))
    overdrawn = (desire.cost > 0) & (
        usage.reserved[own] + walk.kept_cost + desire.cost > usage.balance[own]
    )
    return (desire.kind != KIND_ABSENT) & ~conflicting & ~overdrawn, claimed_here


def _queue(executes: Array, order: Array) -> Array:
    return jnp.argsort(jnp.where(executes, order, LAST)).astype(jnp.int32)


def _execute_in_order(plan: _Plan, executes: Array, order: Array, draws: Array | None, state: State, game: GameConfig, usage: _Usage, player: Array | int) -> tuple[_Desire, Array, Array, Array]:
    decided = _decide(
        _Ordered(
            plan=plan,
            queue=_queue(executes, order),
            queued=jnp.sum(executes, dtype=jnp.int32),
            free=_free_cells(state, game, usage),
            moves_allowed=jnp.all(_within_balance(usage)),
            draws=draws,
            usage=usage,
            own=jnp.asarray(player),
        ),
        game.bucket,
    )
    return decided.desires, decided.ranks, decided.kept_count, decided.drew


class _Ordered(NamedTuple):
    plan: _Plan
    queue: Array
    queued: Array
    free: Array
    moves_allowed: Array
    draws: Array | None
    usage: _Usage
    own: Array


class _Decided(NamedTuple):
    desires: _Desire
    ranks: Array
    kept_count: Array
    drew: Array
    calls: Array


class _Pass(NamedTuple):
    rows: Array
    queued: Array
    free: Array
    moves_allowed: Array
    draws: Array | None
    usage: _Usage
    own: Array


_Searched = Callable[[_Plan, _Walk], pathfinding.Path]

_DecideStage = Callable[[_Ordered], _Decided]


def _decide(ordered: _Ordered, bucket: BucketSpec) -> _Decided:
    kernel, program = _decide_stages(bucket)
    if bucket.cell_count > _CHAIN_CELLS:
        return program(ordered)
    return kernels.dispatch(ordered, kernel=kernel, program=program)


@functools.cache
def _decide_stages(bucket: BucketSpec) -> tuple[_DecideStage, _DecideStage]:
    def kernel(ordered: _Ordered) -> _Decided:
        return _from_chain(ordered, _chain(ordered, bucket), bucket)

    def program(ordered: _Ordered) -> _Decided:
        return _in_order(ordered, bucket)

    return kernel, program


def _in_order(ordered: _Ordered, bucket: BucketSpec) -> _Decided:
    plan, queue = ordered.plan, ordered.queue
    rows = jnp.stack(
        [
            plan.start,
            plan.goal,
            plan.radius,
            plan.draws.astype(jnp.int32),
            plan.options,
            *plan.fallback,
            *plan.blocked,
        ],
        axis=1,
    ).astype(jnp.int32)[queue]

    def searched(entry: _Plan, walk: _Walk) -> pathfinding.Path:
        return pathfinding.search(
            ordered.free & ~walk.used, entry.start, entry.goal, entry.radius, bucket
        )

    walk = _walked(
        _Pass(
            rows,
            ordered.queued,
            ordered.free,
            ordered.moves_allowed,
            ordered.draws,
            ordered.usage,
            ordered.own,
        ),
        bucket,
        searched,
    )
    return _Decided(
        desires=_Desire(
            *(jnp.zeros_like(field).at[queue].set(field) for field in walk.decided)
        ),
        ranks=jnp.zeros_like(walk.ranks).at[queue].set(walk.ranks),
        kept_count=walk.kept_count,
        drew=jnp.zeros_like(walk.drew).at[queue].set(walk.drew),
        calls=walk.calls,
    )


def _walked(ordered: _Pass, bucket: BucketSpec, searched: _Searched) -> _Walk:
    capacity = ordered.rows.shape[0]

    def body(walk: _Walk) -> _Walk:
        entry = _entry(ordered.rows, walk.position)
        return _walk_step(walk, entry, searched(entry, walk), ordered, bucket)

    return lax.while_loop(
        lambda walk: walk.position < ordered.queued,
        body,
        _begin_walk(capacity, bucket),
    )


def _begin_walk(capacity: int, bucket: BucketSpec) -> _Walk:
    return _Walk(
        position=jnp.int32(0),
        used=jnp.zeros(bucket.cell_count, jnp.bool_),
        kept_cells=jnp.zeros(bucket.cell_count, jnp.bool_),
        kept_cost=jnp.int32(0),
        kept_count=jnp.int32(0),
        decided=_no_desire((capacity,)),
        ranks=jnp.full(capacity, ABSENT, jnp.int32),
        calls=jnp.int32(0),
        drew=jnp.zeros(capacity, jnp.bool_),
    )


def _walk_step(walk: _Walk, entry: _Plan, path: pathfinding.Path, ordered: _Pass, bucket: BucketSpec) -> _Walk:
    capacity = walk.ranks.shape[0]
    desire, draws_here = _answer(
        entry, walk, path, ordered.draws, ordered.moves_allowed, bucket
    )
    kept, claimed_here = _kept(
        walk, desire, ordered.usage, ordered.own, bucket.cell_count
    )
    here = jnp.arange(capacity) == walk.position
    return _Walk(
        position=walk.position + 1,
        used=walk.used | claimed_here,
        kept_cells=walk.kept_cells | (kept & claimed_here),
        kept_cost=walk.kept_cost + jnp.where(kept, desire.cost, 0),
        kept_count=walk.kept_count + kept.astype(jnp.int32),
        decided=_choose(here & kept, desire, walk.decided),
        ranks=jnp.where(here & kept, walk.kept_count, walk.ranks),
        calls=walk.calls + draws_here.astype(jnp.int32),
        drew=walk.drew | (here & draws_here),
    )


_CHAIN_TARGET = "nanorts_astar_chain"

_SHARED_BYTES = 48 * 1024

_CHAIN_BYTES_PER_CELL = 8

_CHAIN_CELLS = _SHARED_BYTES // _CHAIN_BYTES_PER_CELL

_KEEP_RULE_COLUMNS: Final[tuple[tuple[str, str], ...]] = (
    ("kind", "kind"),
    ("claimed", "claim"),
    ("cost", "cost"),
)

_CHAIN_COLUMNS: Final[dict[str, int]] = {
    f"{desire}_{attribute}": _SEARCH_COLUMNS + len(_KEEP_RULE_COLUMNS) * block + index
    for block, desire in enumerate(("fallback", "blocked"))
    for index, (_, attribute) in enumerate(_KEEP_RULE_COLUMNS)
}

_DECISION_FIELDS: Final[int] = 4

_KEPT_COUNT: Final[int] = 0

_CALLS: Final[int] = 1

_BLOCKED: Final[int] = 1

_MOVED: Final[int] = 2


def _chain_rows(plan: _Plan) -> Array:
    return jnp.stack(
        [
            plan.start,
            plan.goal,
            plan.radius,
            plan.draws.astype(jnp.int32),
            plan.options,
            *(
                getattr(desire, field)
                for desire in (plan.fallback, plan.blocked)
                for field, _ in _KEEP_RULE_COLUMNS
            ),
        ],
        axis=1,
    ).astype(jnp.int32)


def _chain(ordered: _Ordered, bucket: BucketSpec) -> Array:
    capacity = ordered.queue.shape[0]
    (decisions,) = kernels.call(
        _CHAIN_TARGET,
        [
            jnp.asarray(ordered.free, jnp.bool_),
            jnp.asarray(ordered.usage.claimed, jnp.bool_),
            _chain_rows(ordered.plan),
            ordered.queue,
            ordered.queued,
            ordered.moves_allowed,
            ordered.usage.reserved[ordered.own],
            ordered.usage.balance[ordered.own],
            jnp.zeros(1, jnp.int32) if ordered.draws is None else ordered.draws,
        ],
        [jax.ShapeDtypeStruct((capacity + 1, _DECISION_FIELDS), jnp.int32)],
        attributes={
            "width": bucket.width,
            "height": bucket.height,
            "drawing": int(ordered.draws is not None),
            **_CHAIN_COLUMNS,
        },
    )
    return decisions


def _from_chain(ordered: _Ordered, decisions: Array, bucket: BucketSpec) -> _Decided:
    capacity = ordered.queue.shape[0]
    shape = (capacity,)
    rank, selector, direction, drew = (
        decisions[:capacity, field] for field in range(_DECISION_FIELDS)
    )
    plan = ordered.plan
    moved = _no_desire(shape)._replace(
        kind=jnp.full(shape, KIND_MOVE, jnp.int32),
        direction=direction,
        claimed=_neighbour(plan.start, direction, bucket.width),
    )
    decided = _choose(
        selector == _MOVED,
        _choose(direction != ABSENT, moved, _waiting(shape)),
        _choose(selector == _BLOCKED, plan.blocked, plan.fallback),
    )
    return _Decided(
        desires=_choose(rank != ABSENT, decided, _no_desire(shape)),
        ranks=rank,
        kept_count=decisions[capacity, _KEPT_COUNT],
        drew=drew != 0,
        calls=decisions[capacity, _CALLS],
    )


def _commands(desires: _Desire, ranks: Array, kept_count: Array, units: Units, player: Array | int, game: GameConfig, wait: int) -> Commands:
    kept = desires.kind != KIND_ABSENT
    filler = units.available & ~kept
    filler_rank = _rank(filler, units.birth)
    fields = {
        "kind": jnp.where(
            kept, desires.kind, jnp.where(filler, KIND_NONE, KIND_ABSENT)
        ),
        "direction": jnp.where(kept, desires.direction, ABSENT),
        "none_duration": jnp.where(
            kept & (desires.kind == KIND_NONE),
            desires.none_duration,
            jnp.where(filler, wait, 0),
        ),
        "target_cell": jnp.where(kept, desires.target_cell, ABSENT),
        "produced_type": jnp.where(kept, desires.produced_type, ABSENT),
        "request_order": jnp.where(
            kept, ranks, jnp.where(filler, kept_count + filler_rank, ABSENT)
        ),
    }
    unissued = empty(Commands, game.bucket)
    own_rows = (jnp.arange(PLAYER_COUNT) == jnp.asarray(player))[:, None]
    return dataclasses.replace(
        unissued,
        **{
            name: jnp.where(own_rows, value.astype(jnp.int32), getattr(unissued, name))
            for name, value in fields.items()
        },
    )


def _settled(actions: AbstractActions, standing: Array, executes: Array) -> AbstractActions:
    def held(field: Array) -> Array:
        return jnp.where(standing, field, ABSENT)

    return AbstractActions(
        kind=held(actions.kind),
        unit_birth=held(actions.unit_birth),
        order=held(_rank(standing, actions.order)),
        target=held(actions.target),
        target_birth=held(actions.target_birth),
        base=held(actions.base),
        base_birth=held(actions.base_birth),
        cell=held(actions.cell),
        unit_type=held(actions.unit_type),
        trained=standing
        & (
            actions.trained
            | (executes & ((actions.kind == TRAIN) | (actions.kind == TRAIN_DIRECTION)))
        ),
        next_order=jnp.sum(standing, dtype=jnp.int32),
    )
