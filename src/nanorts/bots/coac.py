from __future__ import annotations

from collections.abc import Callable
from typing import Final, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, lax

from nanorts import kernels
from nanorts.bots import abstraction, pathfinding, whole_act
from nanorts.bots.abstraction import LAST, AbstractActions, Units
from nanorts.bots.bot import Bot
from nanorts.model import (
    ABSENT,
    DIRECTION_COUNT,
    DIRECTION_DOWN,
    DIRECTION_LEFT,
    DIRECTION_OFFSETS,
    DIRECTION_RIGHT,
    DIRECTION_UP,
    KIND_ABSENT,
    KIND_ATTACK,
    KIND_MOVE,
    KIND_PRODUCE,
    OWNER_NEUTRAL,
    TYPE_BARRACKS,
    TYPE_BASE,
    TYPE_HEAVY,
    TYPE_RANGED,
    TYPE_RESOURCE,
    TYPE_WORKER,
    BucketSpec,
    Commands,
    GameConfig,
    State,
)

_LOWEST: Final[int] = int(jnp.iinfo(jnp.int32).min)
_UNSEEN: Final[int] = ABSENT
_DEFAULT_OPENING: Final[int] = 0
_SIXTEEN_A_OPENING: Final[int] = 1
_SIXTEEN_A_BALANCES: Final[tuple[int, int]] = (5, 5)
_SIXTEEN_A_UNITS: Final[tuple[tuple[int, int, int, int, int, int], ...]] = (
    (TYPE_RESOURCE, OWNER_NEUTRAL, 0, 0, 1, 25),
    (TYPE_RESOURCE, OWNER_NEUTRAL, 0, 1, 1, 25),
    (TYPE_RESOURCE, OWNER_NEUTRAL, 15, 14, 1, 25),
    (TYPE_RESOURCE, OWNER_NEUTRAL, 15, 15, 1, 25),
    (TYPE_BASE, 0, 2, 2, 10, 0),
    (TYPE_BASE, 1, 13, 13, 10, 0),
    (TYPE_WORKER, 0, 1, 1, 1, 0),
    (TYPE_WORKER, 1, 14, 14, 1, 0),
)
_FLOOD_WIDTH: Final[int] = 10
_ALWAYS_TRAINING_WIDTH: Final[int] = 8
_WORKERS_PER_BASE: Final[int] = 2
_ABUNDANT: Final[int] = 15
_RICH: Final[int] = 10
_COMBAT_LEAD: Final[int] = 4
_CHASE_DISTANCE: Final[int] = 5
_CENTROID_DISTANCE: Final[int] = 3
_DANGER_DISTANCE: Final[int] = 2
_BUILDER_REACH: Final[int] = 2
_BARRACKS_SPARE: Final[int] = 1
_FAR: Final[int] = 9999999
_HEAVIES_BEFORE_RANGED: Final[int] = 3
_FLEE_ORDER: Final[tuple[int, ...]] = (
    DIRECTION_UP,
    DIRECTION_DOWN,
    DIRECTION_RIGHT,
    DIRECTION_LEFT,
)
_TRAIN_ORDER: Final[tuple[int, ...]] = (
    DIRECTION_UP,
    DIRECTION_DOWN,
    DIRECTION_LEFT,
    DIRECTION_RIGHT,
)
_ADJACENT_ORDER: Final[tuple[int, ...]] = (
    DIRECTION_DOWN,
    DIRECTION_UP,
    DIRECTION_RIGHT,
    DIRECTION_LEFT,
)
_KINDS: Final[frozenset[int]] = frozenset(
    (
        abstraction.MOVE,
        abstraction.COAC_ATTACK,
        abstraction.TRUE_IDLE,
        abstraction.TRAIN_DIRECTION,
        abstraction.BUILD_MODIFIED,
        abstraction.HARVEST,
        abstraction.HARVEST_RETURN,
    )
)
_TARGET: Final[str] = "nanorts_coac_act"
_BYTES_PER_CELL: Final[int] = 8
_LANE_BYTES: Final[int] = 1024
_BYTES_PER_SLOT: Final[int] = 84
_BYTES_PER_ENTITY: Final[int] = 14
_TURN_COLUMNS: Final[tuple[str, ...]] = (
    "unit_attack_damage",
    "unit_attack_time",
    "unit_move_time",
    "unit_produce_time",
)


class Memory(NamedTuple):
    actions: AbstractActions
    harvester: Array
    harvest_base: Array
    harvest_base_birth: Array
    was_separated: Array
    attack_all: Array
    opening: Array


class _Kinds(NamedTuple):
    resource: Array
    stockpile: Array
    combat: Array
    my_workers: Array
    my_combat: Array
    my_base: Array
    my_barracks: Array


class _Seen(NamedTuple):
    units: Units
    first: Array
    x: Array
    y: Array
    next_x: Array
    next_y: Array
    order: Array
    unit_type: Array
    hit_points: Array
    finishes_in: Array
    moving: Array
    enemy: Array
    kinds: _Kinds
    available: Array
    my_closest_resource: Array
    builder_bases: Array
    separated: Array
    free: Array
    holding: Array
    open_cell: Array
    balance: Array
    enemy_balance: Array
    attack_with_combat: Array
    time: Array


class _Cycle(NamedTuple):
    actions: AbstractActions
    alive: Array
    damage: Array
    company: Array
    company_x_sum: Array
    company_y_sum: Array
    free_workers: Array
    harvester: Array
    harvest_base: Array
    harvest_base_birth: Array
    resources_used: Array


class _Fight(NamedTuple):
    kind: Array
    target: Array
    cell: Array
    strikes: Array


def _max_cycles(width: int) -> int:
    for widest, cycles in ((8, 3000), (16, 4000), (24, 5000), (32, 6000), (64, 8000)):
        if width <= widest:
            return cycles
    return 12000


def _at(field: Array, slot: Array) -> Array:
    return field[jnp.clip(slot, 0, field.shape[0] - 1)]


def _lexicographic_least(left: tuple[Array, Array, Array], right: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
    (key, order, slot), (other_key, other_order, other_slot) = left, right
    first = (key < other_key) | (
        (key == other_key)
        & ((order < other_order) | ((order == other_order) & (slot < other_slot)))
    )
    return (
        jnp.where(first, key, other_key),
        jnp.where(first, order, other_order),
        jnp.where(first, slot, other_slot),
    )


def _least(candidates: Array, key: Array, order: Array) -> tuple[Array, Array]:
    candidates, key, order = jnp.broadcast_arrays(candidates, key, order)
    axis = key.ndim - 1
    width = key.shape[axis]
    least, _, slot = lax.reduce(
        (
            jnp.where(candidates, key, LAST),
            jnp.where(candidates, order, LAST),
            jnp.where(
                candidates, lax.broadcasted_iota(jnp.int32, key.shape, axis), width
            ),
        ),
        (jnp.int32(LAST), jnp.int32(LAST), jnp.int32(width)),
        _lexicographic_least,
        (axis,),
    )
    return least, jnp.where(slot < width, slot, ABSENT)


def _first_least(candidates: Array, key: Array, order: Array) -> Array:
    return _least(candidates, key, order)[1]


def _first_most(candidates: Array, key: Array, order: Array) -> Array:
    return _first_least(candidates, -key, order)


def _distances(seen: _Seen, x: Array, y: Array) -> Array:
    return jnp.abs(seen.x - x) + jnp.abs(seen.y - y)


def _nearest(seen: _Seen, candidates: Array, x: Array, y: Array) -> Array:
    return _first_least(candidates, _distances(seen, x, y), seen.order)


def _birth_of(seen: _Seen, slot: Array) -> Array:
    return jnp.where(slot != ABSENT, _at(seen.order, slot), ABSENT)


def _distance_to(seen: _Seen, slot: Array, x: Array, y: Array) -> Array:
    return jnp.abs(_at(seen.x, slot) - x) + jnp.abs(_at(seen.y, slot) - y)


def _steps(directions: tuple[int, ...]) -> tuple[Array, Array]:
    return (
        jnp.asarray([DIRECTION_OFFSETS[d][0] for d in directions], jnp.int32),
        jnp.asarray([DIRECTION_OFFSETS[d][1] for d in directions], jnp.int32),
    )


def _on_board(x: Array, y: Array, game: GameConfig) -> Array:
    bucket = game.bucket
    return (x >= 0) & (x < bucket.width) & (y >= 0) & (y < bucket.height)


def _cell(x: Array, y: Array, game: GameConfig) -> Array:
    return jnp.where(_on_board(x, y, game), y * game.bucket.width + x, 0)


def _owned(seen: _Seen, slot: Array) -> Array:
    return jnp.arange(seen.units.birth.shape[0]) == slot - seen.first


def _block(field: Array, first: Array, capacity: int) -> Array:
    return lax.dynamic_slice_in_dim(field, first, capacity)


def _own(seen: _Seen, field: Array) -> Array:
    return _block(field, seen.first, seen.units.birth.shape[0])


def _for_each[Carry](seen: _Seen, members: Array, body: Callable[[Carry, Array], Carry], carry: Carry, until: Callable[[Carry], Array] | None = None) -> Carry:
    listed = _own(seen, members)
    birth = _own(seen, seen.order)
    count = jnp.sum(listed, dtype=jnp.int32)

    def going(walk: tuple[Array, Array, Carry]) -> Array:
        index, _, carried = walk
        return (index < count) & (True if until is None else ~until(carried))

    def step(walk: tuple[Array, Array, Carry]) -> tuple[Array, Array, Carry]:
        index, after, carried = walk
        member = jnp.argmin(jnp.where(listed & (birth > after), birth, LAST))
        member = member.astype(jnp.int32)
        return index + 1, birth[member], body(carried, seen.first + member)

    _, _, carry = lax.while_loop(going, step, (jnp.int32(0), jnp.int32(_LOWEST), carry))
    return carry


def _opening(state: State, game: GameConfig) -> Array:
    count = len(_SIXTEEN_A_UNITS)
    if game.bucket.entity_capacity < count:
        return jnp.int32(_DEFAULT_OPENING)
    listed = jnp.argsort(jnp.where(state.active, state.birth, LAST))[:count]
    cell = state.cell.astype(jnp.int32)[listed]
    shown = jnp.stack(
        [
            state.unit_type.astype(jnp.int32)[listed],
            state.owner.astype(jnp.int32)[listed],
            cell % game.bucket.width,
            cell // game.bucket.width,
            state.hit_points.astype(jnp.int32)[listed],
            state.carried_resources[listed],
        ],
        axis=1,
    )
    same = (
        (jnp.sum(state.active) == count)
        & jnp.all(shown == jnp.asarray(_SIXTEEN_A_UNITS, jnp.int32))
        & jnp.all(state.player_resources == jnp.asarray(_SIXTEEN_A_BALANCES))
    )
    return jnp.where(same, _SIXTEEN_A_OPENING, _DEFAULT_OPENING).astype(jnp.int32)


def _beside(cells: Array) -> Array:
    source = jnp.pad(cells, 1)
    return source[:-2, 1:-1] | source[2:, 1:-1] | source[1:-1, :-2] | source[1:-1, 2:]


def _reaching(spreads: Array, enemy_cells: Array) -> Array:
    enemies = enemy_cells.reshape(spreads.shape)
    seeded = enemies | _beside(enemies)
    reached, _ = lax.while_loop(
        lambda carry: jnp.any(carry[0] != carry[1]),
        lambda carry: (carry[0] | _beside(carry[0] & spreads), carry[0]),
        (seeded | _beside(seeded & spreads), seeded),
    )
    return reached


def _kinds(state: State, game: GameConfig, mine: Array, unit_type: Array) -> _Kinds:
    active = state.active
    resource = active & game.unit_is_resource[unit_type]
    stockpile = active & ~resource & game.unit_is_stockpile[unit_type]
    others = active & ~resource & ~stockpile
    harvests = game.unit_can_harvest[unit_type]
    combat = others & ~harvests & game.unit_can_attack[unit_type]
    return _Kinds(
        resource=resource,
        stockpile=stockpile,
        combat=combat,
        my_workers=others & harvests & mine,
        my_combat=combat & mine,
        my_base=stockpile & mine,
        my_barracks=others & ~harvests & ~combat & (unit_type == TYPE_BARRACKS) & mine,
    )


def _pending_damage(state: State, game: GameConfig, kinds: _Kinds, unit_type: Array) -> Array:
    attacks = (kinds.my_workers | kinds.my_combat) & (state.pending_kind == KIND_ATTACK)
    target_cell = state.pending_target_cell.astype(jnp.int32)
    victim = jnp.where(
        attacks & (target_cell != ABSENT),
        _at(state.entity_at.reshape(-1), target_cell),
        ABSENT,
    )
    return (
        jnp.zeros(state.birth.shape, jnp.int32)
        .at[jnp.where(victim != ABSENT, victim, state.birth.shape[0])]
        .add(game.unit_attack_damage[unit_type], mode="drop")
    )


def _between(x: Array, y: Array, other_x: Array, other_y: Array) -> Array:
    return jnp.abs(x[:, None] - other_x[None, :]) + jnp.abs(
        y[:, None] - other_y[None, :]
    )


def _closest_resources(kinds: _Kinds, mine: Array, x: Array, y: Array, order: Array) -> Array:
    _, nearest_base = _least(kinds.stockpile, _between(x, y, x, y), order)
    return kinds.resource & (nearest_base != ABSENT) & _at(mine, nearest_base)


def _builder_bases(state: State, kinds: _Kinds, x: Array, y: Array, order: Array, first: Array, capacity: int) -> Array:
    builds_barracks = _block(
        kinds.my_workers
        & (state.pending_kind == KIND_PRODUCE)
        & (state.pending_produced_type == TYPE_BARRACKS),
        first,
        capacity,
    )
    own_x, own_y = _block(x, first, capacity), _block(y, first, capacity)
    _, base = _least(
        _block(kinds.my_base, first, capacity),
        _between(own_x, own_y, own_x, own_y),
        _block(order, first, capacity),
    )
    return jnp.where(builds_barracks, base, ABSENT)


def _separated(state: State, game: GameConfig, kinds: _Kinds, enemy: Array, cell: Array) -> Array:
    bucket = game.bucket
    if bucket.width >= _FLOOD_WIDTH:
        return jnp.zeros_like(kinds.my_base)
    cells = jnp.arange(bucket.cell_count)
    holds_resource = jnp.any(kinds.resource[:, None] & (cell[:, None] == cells), axis=0)
    spreads = ~(game.terrain.reshape(-1) | holds_resource)
    enemy_cells = jnp.any(enemy[:, None] & (cell[:, None] == cells), axis=0)
    reached = _reaching(spreads.reshape(bucket.height, bucket.width), enemy_cells)
    return kinds.my_base & ~(_at(spreads, cell) & _at(reached.reshape(-1), cell))


def _kept_harvesters(memory: Memory, state: State, units: Units, kinds: _Kinds) -> tuple[Array, Array, Array]:
    held = (memory.harvester != ABSENT) & (memory.harvester == units.birth)
    base_alive = _at(state.birth, memory.harvest_base) == memory.harvest_base_birth
    kept = held & base_alive & _at(kinds.my_base, memory.harvest_base)
    return (
        jnp.where(kept, memory.harvester, ABSENT),
        jnp.where(kept, memory.harvest_base, ABSENT),
        jnp.where(kept, memory.harvest_base_birth, ABSENT),
    )


def _read(memory: Memory, state: State, game: GameConfig, player: Array | int) -> tuple[_Seen, _Cycle]:
    bucket = game.bucket
    own = jnp.asarray(player, jnp.int32)
    units = abstraction.own_units(state, game, player)
    cell = state.cell.astype(jnp.int32)
    x, y = cell % bucket.width, cell // bucket.width
    unit_type = jnp.clip(
        state.unit_type.astype(jnp.int32), 0, bucket.unit_type_capacity - 1
    )
    active = state.active
    mine = active & (state.owner.astype(jnp.int32) == own)
    order = jnp.where(active, state.birth, LAST)
    moving = active & (state.pending_kind == KIND_MOVE)
    step_x, step_y = _steps(tuple(range(DIRECTION_COUNT)))
    direction = jnp.clip(state.pending_direction.astype(jnp.int32), 0, None)
    hit_points = state.hit_points.astype(jnp.int32)
    kinds = _kinds(state, game, mine, unit_type)
    enemy = abstraction.enemies(state, player)
    cost = game.unit_cost[unit_type]
    holding = state.entity_at.reshape(-1) != ABSENT
    damage = _pending_damage(state, game, kinds, unit_type)
    harvester, harvest_base, harvest_base_birth = _kept_harvesters(
        memory, state, units, kinds
    )
    first = bucket.neutral_capacity + own * bucket.per_player_capacity
    seen = _Seen(
        units=units,
        first=first,
        x=x,
        y=y,
        next_x=jnp.where(moving, x + _at(step_x, direction), x),
        next_y=jnp.where(moving, y + _at(step_y, direction), y),
        order=order,
        unit_type=unit_type,
        hit_points=hit_points,
        finishes_in=state.pending_completes_at - state.time,
        moving=moving,
        enemy=enemy,
        kinds=kinds,
        available=active & (state.pending_kind == KIND_ABSENT),
        my_closest_resource=_closest_resources(kinds, mine, x, y, order),
        builder_bases=_builder_bases(
            state, kinds, x, y, order, first, bucket.per_player_capacity
        ),
        separated=_separated(state, game, kinds, enemy, cell),
        free=pathfinding.free_cells(state, game),
        holding=holding,
        open_cell=~(game.terrain.reshape(-1) | holding),
        balance=state.player_resources[own],
        enemy_balance=state.player_resources[1 - own],
        attack_with_combat=(
            jnp.sum(jnp.where(kinds.my_combat, cost, 0)) - _COMBAT_LEAD
            > jnp.sum(jnp.where(kinds.combat & ~mine, cost, 0))
        ),
        time=state.time,
    )
    cycle = _Cycle(
        actions=memory.actions,
        alive=enemy & (damage < hit_points),
        damage=damage,
        company=jnp.sum(kinds.my_combat, dtype=jnp.int32),
        company_x_sum=jnp.sum(jnp.where(kinds.my_combat, x, 0)),
        company_y_sum=jnp.sum(jnp.where(kinds.my_combat, y, 0)),
        free_workers=kinds.my_workers & seen.available,
        harvester=harvester,
        harvest_base=harvest_base,
        harvest_base_birth=harvest_base_birth,
        resources_used=jnp.int32(0),
    )
    return seen, cycle


def _put(seen: _Seen, cycle: _Cycle, slot: Array, puts: Array, kind: Array | int, **fields: Array | int) -> _Cycle:
    actions = abstraction.put_one(
        cycle.actions, seen.units, puts & _owned(seen, slot), kind, **fields
    )
    return cycle._replace(actions=actions)


def _move_away(seen: _Seen, slot: Array, enemy: Array, game: GameConfig) -> tuple[Array, Array]:
    ux, uy = seen.x[slot], seen.y[slot]
    ex, ey = _at(seen.next_x, enemy), _at(seen.next_y, enemy)
    step_x, step_y = _steps(_FLEE_ORDER)
    to_x, to_y = ux + step_x, uy + step_y
    farther = jnp.abs(ex - to_x) + jnp.abs(ey - to_y) > jnp.abs(ux - ex) + jnp.abs(
        uy - ey
    )
    fits = (
        _on_board(to_x, to_y, game) & seen.open_cell[_cell(to_x, to_y, game)] & farther
    )
    up, down, right, left = (fits[index] for index in range(len(_FLEE_ORDER)))
    first = jnp.argmax(fits)
    chosen = jnp.where(
        ex == ux,
        jnp.where(up, 0, jnp.where(down, 1, first)),
        jnp.where(ey == uy, jnp.where(right, 2, jnp.where(left, 3, first)), first),
    )
    return jnp.any(fits), _cell(to_x[chosen], to_y[chosen], game)


def _strike_target(seen: _Seen, cycle: _Cycle, slot: Array, in_range: Array, game: GameConfig) -> tuple[Array, Array]:
    unit_type = seen.unit_type[slot]
    leaves = seen.moving & (seen.finishes_in <= game.unit_attack_time[unit_type])
    attackable = in_range & ~leaves
    remaining = seen.hit_points - cycle.damage - game.unit_attack_damage[unit_type]
    killing = attackable & (remaining <= 0)
    best = jnp.where(
        jnp.any(killing),
        _first_most(killing, remaining, seen.order),
        _first_least(
            attackable, _distances(seen, seen.x[slot], seen.y[slot]), seen.order
        ),
    )
    return jnp.any(attackable), best


def _retreat(seen: _Seen, slot: Array, in_range: Array, game: GameConfig) -> tuple[Array, Array]:
    ux, uy = seen.x[slot], seen.y[slot]
    after = jnp.abs(seen.next_x - ux) + jnp.abs(seen.next_y - uy)
    leaving = _first_least(in_range, after, seen.order)
    reach = game.unit_attack_range[_at(seen.unit_type, leaving)]
    lands_out_of_reach = (_at(seen.next_x, leaving) - ux) ** 2 + (
        _at(seen.next_y, leaving) - uy
    ) ** 2 > reach * reach
    flees, refuge = _move_away(seen, slot, leaving, game)
    ranged = game.unit_attack_range[seen.unit_type[slot]] > 1
    return (leaving != ABSENT) & ranged & lands_out_of_reach & flees, refuge


def _pursuit(seen: _Seen, cycle: _Cycle, slot: Array, closest: Array, game: GameConfig) -> tuple[Array, Array, Array]:
    ux, uy = seen.x[slot], seen.y[slot]
    gathered = cycle.company
    centre_x = cycle.company_x_sum // jnp.maximum(gathered, 1)
    centre_y = cycle.company_y_sum // jnp.maximum(gathered, 1)
    late = seen.time > _max_cycles(game.bucket.width) // 2
    near_centre = jnp.abs(centre_x - ux) + jnp.abs(centre_y - uy) <= _CENTROID_DISTANCE
    chases = (
        seen.attack_with_combat
        | (_distance_to(seen, closest, ux, uy) <= _CHASE_DISTANCE)
        | late
        | ((gathered > 0) & near_centre)
    )
    return chases, ~chases & (gathered > 0), _cell(centre_x, centre_y, game)


def _fight(seen: _Seen, cycle: _Cycle, slot: Array, game: GameConfig) -> _Fight:
    ux, uy = seen.x[slot], seen.y[slot]
    reach = game.unit_attack_range[seen.unit_type[slot]]
    in_range = cycle.alive & ((seen.x - ux) ** 2 + (seen.y - uy) ** 2 <= reach * reach)
    strikes, best = _strike_target(seen, cycle, slot, in_range, game)
    retreats, refuge = _retreat(seen, slot, in_range, game)
    retreats = ~strikes & retreats
    after = jnp.abs(seen.next_x - ux) + jnp.abs(seen.next_y - uy)
    closest = _first_least(cycle.alive, after, seen.order)
    roams = jnp.any(cycle.alive) & ~jnp.any(in_range)
    chases, advances, centre = _pursuit(seen, cycle, slot, closest, game)
    chases, advances = roams & chases, roams & advances
    kind = jnp.where(
        strikes | chases,
        abstraction.COAC_ATTACK,
        jnp.where(retreats | advances, abstraction.MOVE, abstraction.TRUE_IDLE),
    )
    return _Fight(
        kind=kind,
        target=jnp.where(strikes, best, jnp.where(chases, closest, ABSENT)),
        cell=jnp.where(retreats, refuge, jnp.where(advances, centre, ABSENT)),
        strikes=strikes,
    )


def _combat(seen: _Seen, cycle: _Cycle, slot: Array, game: GameConfig) -> _Cycle:
    fight = _fight(seen, cycle, slot, game)
    cycle = _put(
        seen,
        cycle,
        slot,
        jnp.bool_(True),
        fight.kind,
        target=fight.target,
        target_birth=_birth_of(seen, fight.target),
        cell=fight.cell,
    )
    hit = fight.strikes & (jnp.arange(cycle.alive.shape[0]) == fight.target)
    damage = cycle.damage + jnp.where(
        hit, game.unit_attack_damage[seen.unit_type[slot]], 0
    )
    return cycle._replace(
        damage=damage, alive=cycle.alive & ~(hit & (damage >= seen.hit_points))
    )


def _train(seen: _Seen, cycle: _Cycle, slot: Array, unit_type: Array | int, game: GameConfig) -> _Cycle:
    cost = game.unit_cost[unit_type]
    affords = seen.available[slot] & (seen.balance - cycle.resources_used >= cost)
    ux, uy = seen.x[slot], seen.y[slot]
    step_x, step_y = _steps(_TRAIN_ORDER)
    to_x, to_y = ux + step_x, uy + step_y
    fits = _on_board(to_x, to_y, game) & seen.free[_cell(to_x, to_y, game)]
    enemy = _nearest(seen, seen.enemy, ux, uy)
    score = jnp.where(
        enemy != ABSENT,
        -(jnp.abs(_at(seen.x, enemy) - to_x) + jnp.abs(_at(seen.y, enemy) - to_y)),
        0,
    )
    chosen = jnp.argmax(jnp.where(fits, score, _LOWEST))
    cycle = cycle._replace(
        resources_used=cycle.resources_used + jnp.where(affords, cost, 0)
    )
    return _put(
        seen,
        cycle,
        slot,
        affords & jnp.any(fits),
        abstraction.TRAIN_DIRECTION,
        cell=_cell(to_x[chosen], to_y[chosen], game),
        unit_type=unit_type,
    )


def _safe_to_build(seen: _Seen, cycle: _Cycle, slot: Array, game: GameConfig) -> Array:
    bx, by = seen.x[slot], seen.y[slot]
    threat = _nearest(seen, cycle.alive, bx, by)
    rich = (seen.balance > _RICH) & (seen.balance >= seen.enemy_balance)
    takes_longer = (
        _distance_to(seen, threat, bx, by) * game.unit_move_time[TYPE_WORKER]
        > game.unit_produce_time[TYPE_BARRACKS]
    )
    return seen.separated[slot] | (threat == ABSENT) | takes_longer | rich


def _barracks_site(seen: _Seen, builder: Array, enemy: Array, was_separated: Array, game: GameConfig) -> tuple[Array, Array]:
    step_x, step_y = _steps(_ADJACENT_ORDER)
    to_x, to_y = _at(seen.x, builder) + step_x, _at(seen.y, builder) + step_y
    site = _cell(to_x, to_y, game)
    resource_near = jnp.any(
        seen.kinds.resource[None, :]
        & (jnp.abs(seen.x[None, :] - to_x[:, None]) <= 1)
        & (jnp.abs(seen.y[None, :] - to_y[:, None]) <= 1),
        axis=1,
    )
    fits = _on_board(to_x, to_y, game) & ~resource_near & ~seen.holding[site]
    enemy_distance = jnp.abs(_at(seen.x, enemy) - to_x) + jnp.abs(
        _at(seen.y, enemy) - to_y
    )
    pick = jnp.where(
        was_separated,
        jnp.argmin(jnp.where(fits, enemy_distance, _FAR)),
        jnp.argmax(jnp.where(fits, enemy_distance, 0)),
    )
    return jnp.any(fits) & (enemy != ABSENT), site[pick]


def _build_barracks(seen: _Seen, cycle: _Cycle, was_separated: Array, works: Array, game: GameConfig) -> _Cycle:
    def base(carry: tuple[_Cycle, Array], slot: Array) -> tuple[_Cycle, Array]:
        cycle, _ = carry
        constructing = jnp.any(seen.builder_bases == slot - seen.first)
        considered = ~constructing & _safe_to_build(seen, cycle, slot, game)
        builder = _nearest(seen, cycle.free_workers, seen.x[slot], seen.y[slot])
        wx, wy = _at(seen.x, builder), _at(seen.y, builder)
        enemy = _nearest(seen, cycle.alive, wx, wy)
        found, site = _barracks_site(seen, builder, enemy, was_separated, game)
        builds = (
            considered
            & (builder != ABSENT)
            & (
                (enemy == ABSENT)
                | (_distance_to(seen, enemy, wx, wy) > _DANGER_DISTANCE)
            )
            & (_distance_to(seen, slot, wx, wy) <= _BUILDER_REACH)
            & found
        )
        cycle = _put(
            seen,
            cycle,
            builder,
            builds,
            abstraction.BUILD_MODIFIED,
            cell=site,
            unit_type=TYPE_BARRACKS,
        )
        cycle = cycle._replace(
            free_workers=cycle.free_workers
            & ~(builds & (jnp.arange(cycle.free_workers.shape[0]) == builder)),
            resources_used=cycle.resources_used
            + jnp.where(builds, game.unit_cost[TYPE_BARRACKS], 0),
        )
        return cycle, considered & (builder == ABSENT)

    starts = (
        works
        & ~jnp.any(seen.kinds.my_barracks)
        & (
            seen.balance - cycle.resources_used
            >= game.unit_cost[TYPE_BARRACKS] + _BARRACKS_SPARE
        )
    )
    cycle, _ = _for_each(
        seen,
        seen.kinds.my_base & starts,
        base,
        (cycle, jnp.bool_(False)),
        until=lambda carry: carry[1],
    )
    return cycle


def _held(seen: _Seen, cycle: _Cycle) -> Array:
    return (cycle.harvester != ABSENT) & (cycle.harvester == seen.units.birth)


def _per_base(seen: _Seen, cycle: _Cycle) -> Array:
    capacity = seen.units.birth.shape[0]
    offset = cycle.harvest_base - seen.first
    counted = _held(seen, cycle) & (offset >= 0) & (offset < capacity)
    return (
        jnp.zeros(capacity, jnp.int32)
        .at[jnp.where(counted, offset, capacity)]
        .add(1, mode="drop")
    )


def _assign(seen: _Seen, cycle: _Cycle, per_base: Array) -> tuple[_Cycle, Array, Array]:
    own_x, own_y = _own(seen, seen.x), _own(seen, seen.y)
    own_order = _own(seen, seen.order)
    open_base = _own(seen, seen.kinds.my_base) & (per_base < _WORKERS_PER_BASE)
    to_resource, resource = _least(
        seen.my_closest_resource,
        _between(own_x, own_y, seen.x, seen.y),
        seen.order,
    )
    to_base, base = _least(open_base, _between(own_x, own_y, own_x, own_y), own_order)
    score = jnp.where(
        (resource != ABSENT) & (base != ABSENT), to_base + to_resource, LAST
    )
    least, worker = _least(
        _own(seen, cycle.free_workers) & ~_held(seen, cycle), score, own_order
    )
    assigns = (worker != ABSENT) & (least != LAST)
    offsets = jnp.arange(per_base.shape[0])
    owned = assigns & (offsets == worker)
    target = _at(resource, worker)
    harvested = _at(base, worker)
    base_slot = jnp.where(harvested != ABSENT, seen.first + harvested, ABSENT)
    cycle = _put(
        seen,
        cycle,
        jnp.where(worker != ABSENT, seen.first + worker, ABSENT),
        assigns,
        abstraction.HARVEST,
        target=target,
        target_birth=_birth_of(seen, target),
        base=base_slot,
        base_birth=_birth_of(seen, base_slot),
    )
    return (
        cycle._replace(
            harvester=jnp.where(owned, seen.units.birth, cycle.harvester),
            harvest_base=jnp.where(owned, base_slot, cycle.harvest_base),
            harvest_base_birth=jnp.where(
                owned, _birth_of(seen, base_slot), cycle.harvest_base_birth
            ),
        ),
        per_base + (assigns & (offsets == harvested)).astype(jnp.int32),
        assigns,
    )


def _assign_harvesters(seen: _Seen, cycle: _Cycle, works: Array) -> _Cycle:
    starts = (
        works
        & jnp.any(cycle.free_workers)
        & jnp.any(seen.kinds.my_base)
        & jnp.any(seen.kinds.resource)
    )

    def once_per_base(carry: tuple[_Cycle, Array, Array], _: Array) -> tuple[_Cycle, Array, Array]:
        cycle, per_base, _ = carry
        cycle, per_base, assigns = _assign(seen, cycle, per_base)
        return cycle, per_base, ~assigns

    cycle, _, _ = _for_each(
        seen,
        seen.kinds.my_base & starts,
        once_per_base,
        (cycle, _per_base(seen, cycle), jnp.bool_(False)),
        until=lambda carry: carry[2],
    )
    return cycle


def _worker_duty(seen: _Seen, cycle: _Cycle, slot: Array, attack_all: Array, game: GameConfig) -> _Cycle:
    own = slot - seen.first
    ux, uy = seen.x[slot], seen.y[slot]
    enemy = _nearest(seen, cycle.alive, ux, uy)
    threatened = (enemy != ABSENT) & (
        _distance_to(seen, enemy, ux, uy) <= _DANGER_DISTANCE
    )
    fights = attack_all | ~_held(seen, cycle)[own] | threatened
    resource = _nearest(seen, seen.kinds.resource, ux, uy)
    returned = _put(
        seen,
        cycle,
        slot,
        ~fights,
        abstraction.HARVEST_RETURN,
        target=resource,
        target_birth=_birth_of(seen, resource),
        base=cycle.harvest_base[own],
        base_birth=cycle.harvest_base_birth[own],
    )
    joins = ~attack_all
    left = joins & _owned(seen, slot)
    joined = cycle._replace(
        company=cycle.company + joins.astype(jnp.int32),
        company_x_sum=cycle.company_x_sum + jnp.where(joins, ux, 0),
        company_y_sum=cycle.company_y_sum + jnp.where(joins, uy, 0),
        harvester=jnp.where(left, ABSENT, cycle.harvester),
        harvest_base=jnp.where(left, ABSENT, cycle.harvest_base),
        harvest_base_birth=jnp.where(left, ABSENT, cycle.harvest_base_birth),
    )
    fought = _combat(seen, joined, slot, game)
    return jax.tree.map(
        lambda fight, harvest: jnp.where(fights, fight, harvest), fought, returned
    )


def _workers_action(seen: _Seen, cycle: _Cycle, memory: Memory, attack_all: Array, game: GameConfig) -> _Cycle:
    cycle = _build_barracks(seen, cycle, memory.was_separated, ~attack_all, game)
    cycle = _assign_harvesters(seen, cycle, ~attack_all)
    return _for_each(
        seen,
        cycle.free_workers,
        lambda cycle, slot: _worker_duty(seen, cycle, slot, attack_all, game),
        cycle,
    )


def _barracks_action(seen: _Seen, cycle: _Cycle, was_separated: Array, opening: Array, game: GameConfig) -> _Cycle:
    heavies = jnp.sum(seen.kinds.my_combat & (seen.unit_type == TYPE_HEAVY))
    enough = jnp.where(opening == _SIXTEEN_A_OPENING, 0, _HEAVIES_BEFORE_RANGED)
    unit_type = jnp.where((heavies > enough) | was_separated, TYPE_RANGED, TYPE_HEAVY)
    return _for_each(
        seen,
        seen.kinds.my_barracks,
        lambda cycle, slot: _train(seen, cycle, slot, unit_type, game),
        cycle,
    )


def _bases_action(seen: _Seen, cycle: _Cycle, game: GameConfig) -> _Cycle:
    bases = seen.kinds.my_base
    if game.bucket.width <= _ALWAYS_TRAINING_WIDTH:
        cycle = _for_each(
            seen,
            bases & ~seen.separated,
            lambda cycle, slot: _train(seen, cycle, slot, TYPE_WORKER, game),
            cycle,
        )
    enough = _WORKERS_PER_BASE * jnp.sum(bases, dtype=jnp.int32)
    workers = jnp.sum(seen.kinds.my_workers, dtype=jnp.int32) + jnp.sum(
        bases & ~seen.available, dtype=jnp.int32
    )

    def stops(carry: tuple[_Cycle, Array]) -> Array:
        cycle, trained = carry
        return (workers + trained >= enough) & (
            seen.balance - cycle.resources_used < _ABUNDANT
        )

    def base(carry: tuple[_Cycle, Array], slot: Array) -> tuple[_Cycle, Array]:
        cycle, trained = carry
        return _train(seen, cycle, slot, TYPE_WORKER, game), trained + 1

    cycle, _ = _for_each(seen, bases, base, (cycle, jnp.int32(0)), until=stops)
    return cycle


def _turn(draws: Array, memory: Memory, state: State, game: GameConfig, player: Array | int) -> tuple[Memory, Commands, Array]:
    if not _fits(game.bucket):
        return _turn_by_program(draws, memory, state, game, player)
    return kernels.dispatch(
        draws,
        memory,
        state,
        game,
        player,
        kernel=_turn_by_kernel,
        program=_turn_by_program,
    )


def _fits(bucket: BucketSpec) -> bool:
    return whole_act.fits(
        bucket,
        bytes_per_cell=_BYTES_PER_CELL,
        lane_bytes=_LANE_BYTES,
        bytes_per_slot=_BYTES_PER_SLOT,
        bytes_per_entity=_BYTES_PER_ENTITY,
    )


def _turn_by_kernel(draws: Array, memory: Memory, state: State, game: GameConfig, player: Array | int) -> tuple[Memory, Commands, Array]:
    memory, commands, (drew,) = whole_act.call(
        _TARGET,
        memory,
        state,
        game,
        player,
        attributes={},
        operands=[
            state.hit_points,
            state.pending_target_cell,
            state.pending_completes_at,
            state.time,
            whole_act.unit_table(game, _TURN_COLUMNS),
            draws,
        ],
        results=[jax.ShapeDtypeStruct((game.bucket.per_player_capacity,), jnp.bool_)],
    )
    return memory, commands, drew


def _turn_by_program(draws: Array, memory: Memory, state: State, game: GameConfig, player: Array | int) -> tuple[Memory, Commands, Array]:
    opening = jnp.where(
        memory.opening == _UNSEEN, _opening(state, game), memory.opening
    ).astype(jnp.int32)
    seen, cycle = _read(memory, state, game, player)
    was_separated = memory.was_separated | jnp.any(seen.separated)
    attack_all = memory.attack_all | (
        (seen.balance == 0) & ~jnp.any(seen.kinds.resource)
    )
    memory = memory._replace(was_separated=was_separated)
    cycle = _workers_action(seen, cycle, memory, attack_all, game)
    cycle = _barracks_action(seen, cycle, was_separated, opening, game)
    cycle = _bases_action(seen, cycle, game)
    cycle = _for_each(
        seen,
        seen.kinds.my_combat & seen.available,
        lambda cycle, slot: _combat(seen, cycle, slot, game),
        cycle,
    )
    actions, commands, drew = abstraction.translate_wait1(
        cycle.actions, draws, state, game, player, kinds=_KINDS
    )
    return (
        Memory(
            actions=actions,
            harvester=cycle.harvester,
            harvest_base=cycle.harvest_base,
            harvest_base_birth=cycle.harvest_base_birth,
            was_separated=was_separated,
            attack_all=attack_all,
            opening=opening,
        ),
        commands,
        drew,
    )


def _init(game: GameConfig) -> Memory:
    absent = jnp.full(game.bucket.per_player_capacity, ABSENT, jnp.int32)
    return Memory(
        actions=abstraction.empty_actions(game),
        harvester=absent,
        harvest_base=absent,
        harvest_base_birth=absent,
        was_separated=jnp.bool_(False),
        attack_all=jnp.bool_(False),
        opening=jnp.int32(_UNSEEN),
    )


def _act(key: Array, memory: Memory, state: State, game: GameConfig, player: Array | int) -> tuple[Memory, Commands]:
    draws = jax.random.randint(
        key, (game.bucket.per_player_capacity,), 0, DIRECTION_COUNT, jnp.int32
    )
    memory, commands, _ = _turn(draws, memory, state, game, player)
    return memory, commands


coacAI: Final[Bot[Memory]] = Bot(name="coacAI", init=_init, act=_act)
