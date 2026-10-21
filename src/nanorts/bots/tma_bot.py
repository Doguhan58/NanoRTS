from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from typing import Final, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, lax

from nanorts import kernels
from nanorts.bots import pathfinding
from nanorts.bots.bot import Bot
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
    OWNER_NEUTRAL,
    PLAYER_COUNT,
    TYPE_BARRACKS,
    TYPE_BASE,
    TYPE_HEAVY,
    TYPE_LIGHT,
    TYPE_RANGED,
    TYPE_WORKER,
    BucketSpec,
    Commands,
    GameConfig,
    State,
    empty,
)

_FAR: Final[int] = 9999
_HIGHEST: Final[int] = int(jnp.iinfo(jnp.int32).max)
_LOWEST: Final[int] = int(jnp.iinfo(jnp.int32).min)
_FILLER_DURATION: Final[int] = 10
_WAIT_ONE: Final[int] = 1
_RECALIBRATION_PERIOD: Final[int] = 500
_SPARE_RESOURCES: Final[int] = 5
_BARRACKS_HELD: Final[int] = 2
_BARRACKS_WEIGHT: Final[int] = 10
_UNREACHED_ENEMY: Final[int] = 10
_UNSEEN_ENEMY_BASE: Final[int] = 10
_ENEMY_REACH: Final[int] = 16
_DEFAULT_HARVESTERS: Final[int] = 2
_HARVEST_SPAN: Final[int] = 5
_LARGE_HARVESTERS: Final[int] = 4
_SMALL_MAP: Final[int] = 12
_TINY_MAP: Final[int] = 10
_LARGE_MAP: Final[int] = 64
_LARGE_THRESHOLD: Final[int] = 10
_TINY_PER_MISSING: Final[int] = 5
_PER_MISSING: Final[int] = 8
_TINY_PER_HELD: Final[int] = 3
_PER_HELD: Final[int] = 4
_CLOSE_BASES: Final[int] = 4
_RANGED_PREFERENCE: Final[int] = 3
_BARRACKS_WANTED: Final[int] = 1
_NEAR: Final[int] = 2
_CLOSE_COMBAT: Final[int] = 4
_PREDOMINANCE_REACH: Final[int] = 5
_RANGED_MODIFIER: Final[int] = 4
_ATTACKER_MODIFIER: Final[int] = 1

_COUNTED: Final[tuple[int, ...]] = (
    TYPE_WORKER,
    TYPE_LIGHT,
    TYPE_HEAVY,
    TYPE_RANGED,
    TYPE_BASE,
    TYPE_BARRACKS,
)
_WORKERS: Final[int] = 0
_LIGHTS: Final[int] = 1
_HEAVIES: Final[int] = 2
_RANGED_UNITS: Final[int] = 3
_BASES: Final[int] = 4
_BARRACKS: Final[int] = 5
_MOBILE: Final[slice] = slice(0, 4)
_TROOPS: Final[slice] = slice(1, 4)
_TRAINED: Final[tuple[int, ...]] = (TYPE_LIGHT, TYPE_HEAVY, TYPE_RANGED)

_STEP_X: Final[tuple[int, ...]] = tuple(dx for dx, _ in DIRECTION_OFFSETS)
_STEP_Y: Final[tuple[int, ...]] = tuple(dy for _, dy in DIRECTION_OFFSETS)

_RING: Final[tuple[tuple[int, int], ...]] = (
    (-1, 0),
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)
_RADIUS_ONE: Final[tuple[tuple[int, int], ...]] = _RING[:4]
_RADIUS_TWO: Final[tuple[tuple[int, int], ...]] = (
    *_RING,
    (-2, 0),
    (0, -2),
    (2, 0),
    (0, 2),
)
_SITES: Final[tuple[tuple[tuple[int, int], ...], ...]] = (
    ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)),
    ((0, -1), (1, -1), (2, -1), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)),
    ((0, -1), (-1, -1), (-2, -1), (-1, 0), (-2, 0), (0, 1), (-1, 1), (-2, 1)),
    ((-1, 0), (-1, 1), (-1, 2), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((-1, 0), (-1, -1), (-1, -2), (0, -1), (0, -2), (1, 0), (1, -1), (1, -2)),
)

_NOTHING: Final[int] = 0
_TRAIN: Final[int] = 1
_ENGAGE: Final[int] = 2
_HARVEST: Final[int] = 3
_BUILD: Final[int] = 4
_IDLE: Final[int] = 5
_MOVE: Final[int] = 6

_BASE_ROLE: Final[int] = 0
_BARRACKS_ROLE: Final[int] = 1
_FIGHTER_ROLE: Final[int] = 2
_BUILDER_ROLE: Final[int] = 3
_HARVESTER_ROLE: Final[int] = 4
_FREE_ROLE: Final[int] = 5

_FIRST_PASS: Final[int] = 0
_BASE_BUILDER: Final[int] = 1
_BARRACKS_BUILDER: Final[int] = 2
_HARVESTERS: Final[int] = 3
_FREE_WORKERS: Final[int] = 4
_IDLE_HARVESTERS: Final[int] = 5
_SKIPPED: Final[int] = 6

_FIRST_PASS_PHASE: Final[str] = "first_pass"
_WORKER_PHASE: Final[str] = "workers"


class MacroMemory(NamedTuple):
    initialized: Array
    main_base: Array
    enemy_base: Array
    base_to_enemy_base: Array
    real_base_to_enemy: Array
    weighted: Array


class _Seen(NamedTuple):
    player: Array
    first_owned: Array
    birth: Array
    unit_type: Array
    cell: Array
    hit_points: Array
    carried: Array
    pending: Array
    future_cell: Array
    remaining_move: Array
    present: Array
    mine: Array
    enemy: Array
    resource: Array
    stockpile: Array
    own_bases: Array
    available: Array
    occupant: Array
    occupied: Array
    wall: Array
    resource_at: Array
    beside_resource: Array
    chokepoint: Array
    bad_spot: Array
    claimed: Array
    free: Array
    reserved: Array
    balance: Array
    time: Array
    rank: Array
    by_rank: Array


class _Strategy(NamedTuple):
    defense: Array
    military: Array
    harvesters: Array
    production: Array
    threshold: Array
    own: Array


class _Plan(NamedTuple):
    mode: Array
    target: Array
    base: Array
    cell: Array
    unit_type: Array
    scouting: Array


class _Desire(NamedTuple):
    kind: Array
    direction: Array
    target_cell: Array
    produced_type: Array
    none_duration: Array


class _RolePlan(NamedTuple):
    plan: _Plan
    scouts: Array
    checks_conflicts: Array


class _Errand(NamedTuple):
    resource: Array
    base: Array
    enemy: Array
    attacks: Array
    harvests: Array
    returns: Array


class _Turn(NamedTuple):
    requested: _Desire
    rank: Array
    requests: Array
    taken: Array
    searched: Array
    used: Array
    listed_move: Array


class _Entry(NamedTuple):
    slot: Array
    role: Array
    visits: Array


def _choose[Tree](condition: Array, chosen: Tree, otherwise: Tree) -> Tree:
    return jax.tree.map(
        lambda one, other: jnp.where(condition, one, other), chosen, otherwise
    )


def _manhattan(cell: Array, other: Array, width: int) -> Array:
    return jnp.abs(cell % width - other % width) + jnp.abs(
        cell // width - other // width
    )


def _squared(cell: Array, other: Array, width: int) -> Array:
    dx = cell % width - other % width
    dy = cell // width - other // width
    return dx * dx + dy * dy


def _offset(cell: Array, dx: Array | int, dy: Array | int, bucket: BucketSpec) -> Array:
    x = cell % bucket.width + dx
    y = cell // bucket.width + dy
    inside = (x >= 0) & (x < bucket.width) & (y >= 0) & (y < bucket.height)
    return jnp.where(inside & (cell != ABSENT), y * bucket.width + x, ABSENT)


def _step(cell: Array, direction: Array, bucket: BucketSpec) -> Array:
    known = jnp.clip(direction, 0, DIRECTION_COUNT - 1)
    dx = jnp.asarray(_STEP_X, jnp.int32)[known]
    dy = jnp.asarray(_STEP_Y, jnp.int32)[known]
    directed = (direction >= 0) & (direction < DIRECTION_COUNT)
    return jnp.where(directed, _offset(cell, dx, dy, bucket), ABSENT)


def _at(table: Array, cell: Array) -> Array:
    return (cell != ABSENT) & table[jnp.clip(cell, 0, table.shape[0] - 1)]


def _cells_of(units: Array, cells: Array, bucket: BucketSpec) -> Array:
    return (
        jnp.zeros(bucket.cell_count, jnp.bool_)
        .at[jnp.where(units & (cells != ABSENT), cells, bucket.cell_count)]
        .set(True, mode="drop")
    )


def _adjacent_direction(cell: Array, goal: Array, width: int) -> Array:
    dx = goal % width - cell % width
    dy = goal // width - cell // width
    matches = jnp.asarray(
        [(dx == ox) & (dy == oy) for ox, oy in DIRECTION_OFFSETS], jnp.bool_
    )
    found = (goal != ABSENT) & jnp.any(matches)
    return jnp.where(found, jnp.argmax(matches), ABSENT).astype(jnp.int32)


def _least(score: Array, among: Array, seen: _Seen, packed: bool = True) -> tuple[Array, Array]:
    if not packed:
        masked = jnp.where(among, score, _HIGHEST)
        least = jnp.min(masked)
        first = jnp.argmin(jnp.where(among & (masked == least), seen.birth, _HIGHEST))
        found = jnp.any(among)
        return (
            jnp.where(found, first, ABSENT).astype(jnp.int32),
            jnp.where(found, least, 0).astype(jnp.int32),
        )
    entities = seen.rank.shape[0]
    least = jnp.min(jnp.where(among, score * entities + seen.rank, _HIGHEST))
    found = least != _HIGHEST
    first = seen.by_rank[jnp.where(found, least % entities, 0)]
    return (
        jnp.where(found, first, ABSENT).astype(jnp.int32),
        jnp.where(found, least // entities, 0).astype(jnp.int32),
    )


def _cell_of(slot: Array, seen: _Seen) -> Array:
    return jnp.where(slot != ABSENT, seen.cell[jnp.clip(slot, 0, None)], ABSENT)


def _nearest(seen: _Seen, slot: Array, among: Array, width: int) -> tuple[Array, Array]:
    return _least(_manhattan(seen.cell[slot], seen.cell, width), among, seen)


def _within_half(score: Array, to_base: Array, bucket: BucketSpec) -> Array:
    half = max(bucket.height, bucket.width) // 2
    return (score < half) | (to_base < half)


def _beside(table: Array, bucket: BucketSpec) -> Array:
    padded = jnp.pad(table.reshape(bucket.height, bucket.width), 1)
    return (
        padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2] | padded[1:-1, 2:]
    ).reshape(-1)


def _around(table: Array, bucket: BucketSpec) -> Array:
    padded = jnp.pad(table.reshape(bucket.height, bucket.width), 1)
    height, width = bucket.height, bucket.width
    found = jnp.zeros((height, width), jnp.bool_)
    for dx, dy in _RING:
        found = found | padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
    return found.reshape(-1)


def _chokepoints(wall: Array, bucket: BucketSpec) -> Array:
    width, height = bucket.width, bucket.height
    cells = jnp.arange(bucket.cell_count, dtype=jnp.int32)
    x, y = cells % width, cells // width
    walls: list[Array] = []
    skews: list[Array] = []
    for dx, dy in _RING:
        nx, ny = x + dx, y + dy
        inside = (nx >= 0) & (nx < width) & (ny >= 0) & (ny < height)
        walls.append(inside & wall[jnp.where(inside, ny * width + nx, 0)])
        skews.append(jnp.abs(nx - ny))
    found = jnp.zeros(bucket.cell_count, jnp.bool_)
    count = jnp.zeros(bucket.cell_count, jnp.int32)
    for k in range(len(_RING)):
        count = count + walls[k].astype(jnp.int32)
        apart = jnp.zeros(bucket.cell_count, jnp.bool_)
        for j in range(k + 1):
            apart = apart | (walls[j] & (skews[k] + skews[j] > 1))
        found = found | (walls[k] & (count > 1) & apart)
    return found


def _read(state: State, game: GameConfig, player: Array | int) -> _Seen:
    bucket = game.bucket
    player = jnp.asarray(player, jnp.int32)
    present = state.active
    unit_type = jnp.clip(
        state.unit_type.astype(jnp.int32), 0, bucket.unit_type_capacity - 1
    )
    owner = state.owner.astype(jnp.int32)
    cell = state.cell.astype(jnp.int32)
    pending = jnp.where(present, state.pending_kind.astype(jnp.int32), KIND_ABSENT)
    stepped = _step(cell, state.pending_direction.astype(jnp.int32), bucket)
    moving = present & (pending == KIND_MOVE)
    claims = present & ((pending == KIND_MOVE) | (pending == KIND_PRODUCE))
    produced = jnp.clip(
        state.pending_produced_type.astype(jnp.int32), 0, bucket.unit_type_capacity - 1
    )
    cost = jnp.where(
        present & (pending == KIND_PRODUCE), game.unit_cost[produced], 0
    ).astype(jnp.int32)
    reserved = jnp.sum(
        cost[bucket.neutral_capacity :].reshape(
            PLAYER_COUNT, bucket.per_player_capacity
        ),
        axis=-1,
        dtype=jnp.int32,
    )
    resource = present & game.unit_is_resource[unit_type]
    mine = present & (owner == player)
    occupant = state.entity_at.reshape(-1).astype(jnp.int32)
    wall = game.terrain.reshape(-1)
    resource_at = _cells_of(resource, cell, bucket)
    chokepoint = _chokepoints(wall, bucket)
    slots = jnp.arange(bucket.entity_capacity, dtype=jnp.int32)
    _, by_rank = lax.sort(
        (jnp.where(present, state.birth, _HIGHEST), slots), num_keys=2, is_stable=False
    )
    rank = jnp.zeros_like(slots).at[by_rank].set(slots, unique_indices=True)
    return _Seen(
        player=player,
        first_owned=bucket.neutral_capacity + player * bucket.per_player_capacity,
        birth=state.birth,
        unit_type=unit_type,
        cell=cell,
        hit_points=state.hit_points.astype(jnp.int32),
        carried=state.carried_resources,
        pending=pending,
        future_cell=jnp.where(moving & (stepped != ABSENT), stepped, cell),
        remaining_move=jnp.where(
            moving, state.pending_completes_at - state.time, _HIGHEST
        ),
        present=present,
        mine=mine,
        enemy=present & (owner != OWNER_NEUTRAL) & (owner != player),
        resource=resource,
        stockpile=mine & game.unit_is_stockpile[unit_type],
        own_bases=mine & (unit_type == TYPE_BASE),
        available=present & (pending == KIND_ABSENT),
        occupant=occupant,
        occupied=occupant != ABSENT,
        wall=wall,
        resource_at=resource_at,
        beside_resource=_beside(resource_at, bucket),
        chokepoint=chokepoint,
        bad_spot=chokepoint | _around(resource_at, bucket),
        claimed=_cells_of(claims, stepped, bucket),
        free=pathfinding.free_cells(state, game),
        reserved=reserved,
        balance=state.player_resources,
        time=state.time,
        rank=rank,
        by_rank=by_rank,
    )


def _tally(units: Array, seen: _Seen) -> Array:
    return jnp.stack(
        [
            jnp.sum(units & (seen.unit_type == kind), dtype=jnp.int32)
            for kind in _COUNTED
        ]
    )


def _first_cell(units: Array, seen: _Seen) -> Array:
    first, _ = _least(jnp.zeros_like(seen.cell), units, seen)
    return _cell_of(first, seen)


def _remember_bases(memory: MacroMemory, seen: _Seen) -> MacroMemory:
    enemy_bases = seen.enemy & (seen.unit_type == TYPE_BASE)
    main = jnp.where(
        memory.main_base == ABSENT,
        _first_cell(seen.own_bases, seen),
        memory.main_base,
    )
    enemy = jnp.where(
        memory.enemy_base == ABSENT, _first_cell(enemy_bases, seen), memory.enemy_base
    )
    return memory._replace(
        main_base=jnp.where(jnp.any(seen.own_bases), main, ABSENT),
        enemy_base=jnp.where(jnp.any(enemy_bases), enemy, ABSENT),
    )


def _weighs(own: Array) -> Array:
    return (own[_WORKERS] != 0) | (own[_BASES] != 0)


def _initialize(memory: MacroMemory, seen: _Seen) -> MacroMemory:
    first = _remember_bases(memory, seen)._replace(
        initialized=jnp.bool_(True), weighted=_weighs(_tally(seen.mine, seen))
    )
    return _choose(memory.initialized, memory, first)


def _reaches(free: Array, start: Array, goal: Array, active: Array, bucket: BucketSpec) -> pathfinding.Path:
    path = pathfinding.search(
        free,
        start,
        jnp.where(active, goal, start),
        jnp.where(active, 1, 0).astype(jnp.int32),
        bucket,
    )
    return path._replace(found=active & path.found)


def _real_distance(seen: _Seen, main: Array, needed: Array, game: GameConfig) -> Array:
    bucket = game.bucket
    capacity = bucket.per_player_capacity
    first = bucket.neutral_capacity + (1 - seen.player) * capacity
    slots = first + jnp.arange(capacity, dtype=jnp.int32)
    cells, enemy = seen.cell[slots], seen.enemy[slots] & needed
    if not pathfinding.fits_kernel(bucket):
        return _bounded_distance(seen.free, main, cells, enemy, bucket)
    kernel, program = _distance_stages(bucket)
    return kernels.dispatch(
        seen.free, main, cells, enemy, kernel=kernel, program=program
    )


_DistanceStage = Callable[[Array, Array, Array, Array], Array]


@functools.cache
def _distance_stages(bucket: BucketSpec) -> tuple[_DistanceStage, _DistanceStage]:
    def kernel(free: Array, main: Array, cells: Array, enemy: Array) -> Array:
        return _measured_distance(free, main, cells, enemy, bucket)

    def program(free: Array, main: Array, cells: Array, enemy: Array) -> Array:
        return _bounded_distance(free, main, cells, enemy, bucket)

    return kernel, program


def _measured_distance(free: Array, main: Array, cells: Array, enemy: Array, bucket: BucketSpec) -> Array:
    def measure(cell: Array, active: Array) -> Array:
        path = _reaches(free, main, cell, active, bucket)
        return jnp.where(path.found, path.distance, _HIGHEST)

    return _or_absent(jnp.min(jax.vmap(measure)(cells, enemy)))


def _bounded_distance(free: Array, main: Array, cells: Array, enemy: Array, bucket: BucketSpec) -> Array:
    origin = jnp.where(jnp.any(enemy), main, ABSENT)
    bounds = _bounds(_levels(free, origin, bucket), cells, enemy, bucket)
    order = jnp.argsort(bounds)
    ordered = jnp.append(bounds[order], _HIGHEST)
    last = cells.shape[0] - 1

    def lowers(carry: tuple[Array, Array]) -> Array:
        index, best = carry
        return ordered[index] < best

    def search(carry: tuple[Array, Array]) -> tuple[Array, Array]:
        index, best = carry
        cell = cells[order[jnp.minimum(index, last)]]
        path = _reaches(free, main, cell, lowers(carry), bucket)
        return index + 1, jnp.where(path.found, jnp.minimum(best, path.distance), best)

    _, best = lax.while_loop(lowers, search, (jnp.int32(0), jnp.int32(_HIGHEST)))
    return _or_absent(best)


def _levels(free: Array, main: Array, bucket: BucketSpec) -> Array:
    cells = jnp.arange(bucket.cell_count, dtype=jnp.int32)

    def growing(carry: tuple[Array, Array]) -> Array:
        level, depth = carry
        return jnp.any(level == depth)

    def grow(carry: tuple[Array, Array]) -> tuple[Array, Array]:
        level, depth = carry
        reached = _beside(level == depth, bucket) & free & (level == _HIGHEST)
        return jnp.where(reached, depth + 1, level), depth + 1

    level, _ = lax.while_loop(
        growing, grow, (jnp.where(cells == main, 0, _HIGHEST), jnp.int32(0))
    )
    return level


def _bounds(levels: Array, cells: Array, enemy: Array, bucket: BucketSpec) -> Array:
    height, width = bucket.height, bucket.width
    padded = jnp.pad(levels.reshape(height, width), 1, constant_values=_HIGHEST)
    disk = padded[1:-1, 1:-1]
    for dx, dy in _RADIUS_ONE:
        disk = jnp.minimum(
            disk, padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
        )
    least = disk.reshape(-1)
    return jnp.where(enemy, least[jnp.clip(cells, 0, bucket.cell_count - 1)], _HIGHEST)


def _or_absent(distance: Array) -> Array:
    return jnp.where(distance == _HIGHEST, ABSENT, distance).astype(jnp.int32)


def _harvesters(based: Array, base_to_enemy_base: Array, to_resources: Array, to_enemy: Array, own: Array, bucket: BucketSpec) -> Array:
    widest = max(bucket.height, bucket.width)
    return jnp.select(
        [
            ~based,
            jnp.bool_(widest < _SMALL_MAP) | (base_to_enemy_base < _ENEMY_REACH),
            to_resources > to_enemy,
            to_resources // _HARVEST_SPAN > _DEFAULT_HARVESTERS,
            jnp.bool_(widest >= _LARGE_MAP),
        ],
        [
            0,
            own[_BASES],
            jnp.maximum(_DEFAULT_HARVESTERS, own[_BASES]),
            jnp.maximum(to_resources // _HARVEST_SPAN, own[_BASES]),
            jnp.minimum(to_resources - 1, _LARGE_HARVESTERS),
        ],
        _DEFAULT_HARVESTERS,
    ).astype(jnp.int32)


def _production(own: Array, theirs: Array, prefers_ranged: Array) -> Array:
    troops = own[_TROOPS]
    highest = jnp.max(troops)
    production = jnp.where(
        highest == 0,
        jnp.asarray([0, 1, 0], jnp.int32),
        (troops < highest).astype(jnp.int32),
    )
    production = production + jnp.where(
        prefers_ranged, jnp.asarray([0, 0, _RANGED_PREFERENCE], jnp.int32), 0
    )
    countered = jnp.int32(_WORKERS)
    most = jnp.int32(0)
    for index in (_LIGHTS, _HEAVIES, _RANGED_UNITS):
        takes = (theirs[index] > most) & (theirs[index] >= own[index])
        countered = jnp.where(takes, index, countered)
        most = jnp.where(takes, theirs[index], most)
    against_melee = (countered == _LIGHTS) | (countered == _HEAVIES)
    return (
        production
        + jnp.where(against_melee, jnp.asarray([0, 1, 0], jnp.int32), 0)
        + jnp.where(countered == _RANGED_UNITS, jnp.asarray([1, 0, 0], jnp.int32), 0)
    )


def _threshold(real_distance: Array, own: Array, bucket: BucketSpec) -> Array:
    widest = max(bucket.height, bucket.width)
    short = _BARRACKS_WANTED - own[_BARRACKS]
    per_missing = _TINY_PER_MISSING if widest <= _TINY_MAP else _PER_MISSING
    per_held = _TINY_PER_HELD if widest <= _TINY_MAP else _PER_HELD
    return jnp.select(
        [jnp.bool_(widest >= _LARGE_MAP) & (real_distance != ABSENT), short > 0],
        [_LARGE_THRESHOLD, per_missing * short],
        per_held * own[_BARRACKS],
    ).astype(jnp.int32)


def _macro(memory: MacroMemory, seen: _Seen, plays: Array, game: GameConfig) -> tuple[MacroMemory, _Strategy]:
    bucket = game.bucket
    width = bucket.width
    own = _tally(seen.mine, seen)
    theirs = _tally(seen.enemy, seen)
    memory = _remember_bases(memory, seen)
    recalibrates = (seen.time % _RECALIBRATION_PERIOD == 0) & (seen.time != 0)
    weighted = jnp.where(recalibrates, _weighs(own), memory.weighted)
    main, enemy_base = memory.main_base, memory.enemy_base
    based = main != ABSENT
    measures = based & (memory.real_base_to_enemy <= 0)
    real_distance = lax.cond(
        measures,
        lambda: _real_distance(seen, jnp.clip(main, 0, None), measures & plays, game),
        lambda: memory.real_base_to_enemy,
    )
    base_to_enemy_base = jnp.where(
        based & (enemy_base != ABSENT),
        _manhattan(main, enemy_base, width),
        memory.base_to_enemy_base,
    ).astype(jnp.int32)
    from_main = _manhattan(seen.cell, main, width)
    to_resources = jnp.where(
        based & jnp.any(seen.resource),
        jnp.min(jnp.where(seen.resource, from_main, _HIGHEST)),
        _FAR,
    )
    nearest, nearest_distance = _least(from_main, seen.enemy, seen)
    to_enemy = jnp.where(based & (nearest != ABSENT), nearest_distance, _FAR)
    to_enemy_base = jnp.where(
        (enemy_base != ABSENT) & jnp.any(seen.mine),
        jnp.min(
            jnp.where(seen.mine, _manhattan(seen.cell, enemy_base, width), _HIGHEST)
        ),
        _FAR,
    )
    weight = weighted.astype(jnp.int32)
    lead = jnp.sum(own[_MOBILE]) - jnp.sum(theirs[_MOBILE])
    approach = jnp.where(
        to_enemy == _FAR,
        0,
        jnp.where(to_enemy_base == _FAR, _UNSEEN_ENEMY_BASE, to_enemy - to_enemy_base),
    )
    attacks = weight * (lead + approach) >= 0
    spare = seen.balance[seen.player] - _SPARE_RESOURCES
    holds_barracks = jnp.where(own[_BARRACKS] > 0, _BARRACKS_HELD, 0)
    nearness = jnp.where(
        real_distance == ABSENT,
        _UNREACHED_ENEMY,
        jnp.minimum(to_enemy, bucket.height + bucket.width) - _ENEMY_REACH,
    )
    military = weight * (spare + _BARRACKS_WEIGHT * holds_barracks + nearness) >= 0
    consulted = military & jnp.any(
        seen.mine & seen.available & (seen.unit_type == TYPE_BARRACKS)
    )
    cut_off = lax.cond(
        based & (nearest != ABSENT),
        lambda: (
            ~_reaches(
                seen.free,
                jnp.clip(main, 0, None),
                _cell_of(nearest, seen),
                consulted,
                bucket,
            ).found
        ),
        lambda: jnp.bool_(False),
    )
    prefers_ranged = (
        (real_distance == ABSENT)
        | ((base_to_enemy_base <= _CLOSE_BASES) & (own[_RANGED_UNITS] == 0))
        | cut_off
    )
    strategy = _Strategy(
        defense=~attacks,
        military=military,
        harvesters=_harvesters(
            based, base_to_enemy_base, to_resources, to_enemy, own, bucket
        ),
        production=_production(own, theirs, prefers_ranged),
        threshold=_threshold(real_distance, own, bucket),
        own=own,
    )
    return (
        memory._replace(
            base_to_enemy_base=base_to_enemy_base,
            real_base_to_enemy=real_distance,
            weighted=weighted,
        ),
        strategy,
    )


def _plan(mode: Array | int, *, target: Array | int = ABSENT, base: Array | int = ABSENT, cell: Array | int = ABSENT, unit_type: Array | int = ABSENT, scouting: Array | bool = False) -> _Plan:
    return _Plan(
        mode=jnp.asarray(mode, jnp.int32),
        target=jnp.asarray(target, jnp.int32),
        base=jnp.asarray(base, jnp.int32),
        cell=jnp.asarray(cell, jnp.int32),
        unit_type=jnp.asarray(unit_type, jnp.int32),
        scouting=jnp.asarray(scouting, jnp.bool_),
    )


def _engage(target: Array, scouting: Array | bool = False) -> _Plan:
    return _plan(
        jnp.where(target != ABSENT, _ENGAGE, _NOTHING),
        target=target,
        scouting=scouting,
    )


def _nothing() -> _Desire:
    absent = jnp.int32(ABSENT)
    return _Desire(
        kind=jnp.int32(KIND_ABSENT),
        direction=absent,
        target_cell=absent,
        produced_type=absent,
        none_duration=jnp.int32(0),
    )


def _directed(kind: Array | int, direction: Array) -> _Desire:
    return _nothing()._replace(kind=jnp.asarray(kind, jnp.int32), direction=direction)


def _produced(direction: Array, unit_type: Array) -> _Desire:
    return _directed(KIND_PRODUCE, direction)._replace(produced_type=unit_type)


def _attack(cell: Array) -> _Desire:
    return _nothing()._replace(kind=jnp.int32(KIND_ATTACK), target_cell=cell)


def _wait_one() -> _Desire:
    return _nothing()._replace(
        kind=jnp.int32(KIND_NONE), none_duration=jnp.int32(_WAIT_ONE)
    )


def _astar_free(seen: _Seen, searched: Array) -> Array:
    return seen.free & ~searched


def _scout_free(seen: _Seen, searched: Array) -> Array:
    return (seen.free | seen.resource_at) & ~searched


def _move_allowed(seen: _Seen, cell: Array) -> Array:
    return (
        (cell != ABSENT)
        & ~_at(seen.wall, cell)
        & ~_at(seen.occupied, cell)
        & ~_at(seen.claimed, cell)
    )


def _affordable(seen: _Seen, cost: Array, used: Array | int = 0) -> Array:
    own = seen.player
    return (cost == 0) | (seen.reserved[own] + used + cost <= seen.balance[own])


def _walk(seen: _Seen, start: Array, direction: Array, bucket: BucketSpec) -> _Desire:
    allowed = _move_allowed(seen, _step(start, direction, bucket))
    return _choose(allowed, _directed(KIND_MOVE, direction), _nothing())


def _open(seen: _Seen, cells: Array) -> Array:
    return (
        (cells != ABSENT)
        & ~_at(seen.occupied, cells)
        & ~_at(seen.wall, cells)
        & ~_at(seen.beside_resource, cells)
    )


def _scores(seen: _Seen, slot: Array, game: GameConfig) -> Array:
    unit_type = seen.unit_type[slot]
    distance = _manhattan(seen.cell[slot], seen.cell, game.bucket.width)
    one_shot = seen.hit_points <= game.unit_attack_damage[unit_type]
    return jnp.where(
        (distance <= game.unit_attack_range[unit_type]) & one_shot, 0, distance
    )


def _predominant(seen: _Seen, slot: Array, width: int) -> Array:
    start = seen.cell[slot]
    near = (
        seen.present
        & (jnp.abs(seen.cell % width - start % width) <= _PREDOMINANCE_REACH)
        & (jnp.abs(seen.cell // width - start // width) <= _PREDOMINANCE_REACH)
    )
    allies = jnp.sum(near & seen.mine, dtype=jnp.int32)
    enemies = jnp.sum(near & seen.enemy, dtype=jnp.int32)
    return enemies <= allies


def _stalls(seen: _Seen, slot: Array, width: int) -> Array:
    return seen.chokepoint[seen.cell[slot]] & ~_predominant(seen, slot, width)


def _idle(seen: _Seen, slot: Array, game: GameConfig) -> _Desire:
    unit_type = seen.unit_type[slot]
    reach = game.unit_attack_range[unit_type]
    within = seen.enemy & (
        _squared(seen.cell[slot], seen.cell, game.bucket.width) <= reach * reach
    )
    target, _ = _least(jnp.zeros_like(seen.cell), within, seen)
    strikes = game.unit_can_attack[unit_type] & (target != ABSENT)
    return _choose(strikes, _attack(_cell_of(target, seen)), _nothing())


def _train(seen: _Seen, slot: Array, unit_type: Array, strategy: _Strategy, game: GameConfig) -> _Desire:
    bucket = game.bucket
    start = seen.cell[slot]
    cells = jnp.stack(
        [_step(start, jnp.int32(direction), bucket) for direction in range(4)]
    )
    open_ = _at(seen.free, cells)
    known = jnp.clip(unit_type, 0, bucket.unit_type_capacity - 1)
    to_resources = game.unit_can_harvest[known] & (
        strategy.own[_WORKERS] < strategy.harvesters
    )
    toward = jnp.where(to_resources, seen.resource, seen.enemy)
    distance = _manhattan(cells[:, None], seen.cell[None, :], bucket.width)
    nearest = jnp.where(
        jnp.any(toward), jnp.min(jnp.where(toward, distance, _HIGHEST), axis=1), 0
    )
    best = jnp.argmax(jnp.where(open_, -nearest, _LOWEST)).astype(jnp.int32)
    allowed = jnp.any(open_) & _affordable(seen, game.unit_cost[known])
    return _choose(allowed, _produced(best, unit_type), _nothing())


def _approach(seen: _Seen, searched: Array, slot: Array, target: Array, active: Array, game: GameConfig) -> _Desire:
    bucket = game.bucket
    width = bucket.width
    start = seen.cell[slot]
    target_cell = seen.cell[target]
    target_type = seen.unit_type[target]
    modifier = jnp.where(
        game.unit_can_attack[target_type],
        jnp.where(target_type == TYPE_RANGED, _RANGED_MODIFIER, _ATTACKER_MODIFIER),
        0,
    )
    current = _manhattan(target_cell, start, width) - modifier
    cells = jnp.stack([_offset(start, dx, dy, bucket) for dx, dy in _RADIUS_TWO])
    closer = _manhattan(target_cell, cells, width) - modifier
    qualifies = active & _open(seen, cells) & (closer < current) & (closer > 0)
    order = jnp.lexsort(
        (jnp.arange(len(_RADIUS_TWO)), jnp.where(qualifies, closer, _HIGHEST))
    )
    candidates = cells[order]
    count = jnp.sum(qualifies, dtype=jnp.int32)
    free = _astar_free(seen, searched)

    def unanswered(carry: tuple[Array, Array]) -> Array:
        index, step = carry
        return (index < count) & (step == ABSENT)

    def try_next(carry: tuple[Array, Array]) -> tuple[Array, Array]:
        index, _ = carry
        goal = jnp.where(unanswered(carry), candidates[index], start)
        path = pathfinding.search(free, start, goal, jnp.int32(0), bucket)
        return index + 1, path.step

    _, step = lax.while_loop(unanswered, try_next, (jnp.int32(0), jnp.int32(ABSENT)))
    return _walk(seen, start, step, bucket)


def _carry_out(seen: _Seen, searched: Array, slot: Array, plan: _Plan, strategy: _Strategy, active: Array, game: GameConfig, phase: str) -> _Desire:
    bucket = game.bucket
    width = bucket.width
    start = seen.cell[slot]
    unit_type = seen.unit_type[slot]
    reach = game.unit_attack_range[unit_type]
    target = jnp.clip(plan.target, 0, None)
    target_cell = seen.cell[target]
    engages = plan.mode == _ENGAGE
    escapes = seen.remaining_move[target] <= game.unit_attack_time[unit_type]
    strikes = (
        engages & (_squared(start, target_cell, width) <= reach * reach) & ~escapes
    )
    waits = (
        engages
        & ~strikes
        & (_squared(start, seen.future_cell[target], width) <= reach * reach)
    )
    chases = engages & ~strikes & ~waits
    if phase == _FIRST_PASS_PHASE:
        moves = plan.mode == _MOVE
        searching = active & (chases | moves)
        goal = jnp.where(chases, target_cell, plan.cell)
        radius = jnp.where(chases, reach, 0).astype(jnp.int32)
        free = _astar_free(seen, searched)
    else:
        carrying = seen.carried[slot] > 0
        errand = jnp.where(carrying, plan.base, plan.target)
        harvests = (plan.mode == _HARVEST) & (errand != ABSENT)
        builds = plan.mode == _BUILD
        searching = active & (chases | harvests | builds)
        goal = jnp.select(
            [chases, harvests], [target_cell, _cell_of(errand, seen)], plan.cell
        )
        radius = jnp.select([chases, harvests | builds], [reach, 1], 0).astype(
            jnp.int32
        )
        free = jnp.where(
            chases & plan.scouting,
            _scout_free(seen, searched),
            _astar_free(seen, searched),
        )
    path = pathfinding.search(
        free,
        start,
        jnp.where(searching, goal, start),
        jnp.where(searching, radius, 0),
        bucket,
    )
    stepped = searching & (path.step != ABSENT)
    walks = _walk(seen, start, path.step, bucket)
    approach = _approach(seen, searched, slot, target, active & chases & ~stepped, game)
    if phase == _FIRST_PASS_PHASE:
        desire = _choose(
            plan.mode == _TRAIN,
            _train(seen, slot, plan.unit_type, strategy, game),
            _nothing(),
        )
        desire = _choose(strikes, _attack(target_cell), desire)
        desire = _choose(waits, _wait_one(), desire)
        desire = _choose(chases & stepped, walks, desire)
        desire = _choose(chases & ~stepped, approach, desire)
        return _choose(moves & stepped, walks, desire)
    into_resource = plan.scouting & _at(
        seen.resource_at, _step(start, path.step, bucket)
    )
    gathers = _choose(
        _stalls(seen, slot, width), _nothing(), _directed(KIND_HARVEST, path.step)
    )
    beside = _adjacent_direction(start, goal, width)
    errand_end = _choose(
        beside != ABSENT,
        _directed(jnp.where(carrying, KIND_RETURN, KIND_HARVEST), beside),
        _nothing(),
    )
    built = jnp.clip(plan.unit_type, 0, bucket.unit_type_capacity - 1)
    site_allowed = ~_at(seen.claimed, _step(start, beside, bucket)) & _affordable(
        seen, game.unit_cost[built]
    )
    site_end = _choose(
        (beside != ABSENT) & site_allowed,
        _produced(beside, plan.unit_type),
        _nothing(),
    )
    desire = _choose(strikes, _attack(target_cell), _nothing())
    desire = _choose(waits, _wait_one(), desire)
    desire = _choose(chases & stepped, _choose(into_resource, gathers, walks), desire)
    desire = _choose(chases & ~stepped, approach, desire)
    desire = _choose((harvests | builds) & stepped, walks, desire)
    desire = _choose(harvests & ~stepped, errand_end, desire)
    desire = _choose(builds & ~stepped, site_end, desire)
    return _choose(plan.mode == _IDLE, _idle(seen, slot, game), desire)


def _defense_plan(seen: _Seen, slot: Array, game: GameConfig) -> _Plan:
    bucket = game.bucket
    target, score = _least(_scores(seen, slot, game), seen.enemy, seen)
    _, to_base = _nearest(seen, slot, seen.own_bases, bucket.width)
    defends = _within_half(score, to_base, bucket)
    return _engage(jnp.where(defends, target, ABSENT))


def _ranged_plan(seen: _Seen, slot: Array, game: GameConfig) -> _Plan:
    bucket = game.bucket
    width = bucket.width
    start = seen.cell[slot]
    closest, closest_distance = _nearest(seen, slot, seen.enemy, width)
    best, _ = _least(_scores(seen, slot, game), seen.enemy, seen)
    known_best = jnp.clip(best, 0, None)
    one_shot = (
        seen.hit_points[known_best] <= game.unit_attack_damage[seen.unit_type[slot]]
    )
    near = closest_distance <= _NEAR
    stepping_in = (
        _manhattan(start, seen.future_cell[jnp.clip(closest, 0, None)], width) == 1
    )
    cells = jnp.stack([_offset(start, dx, dy, bucket) for dx, dy in _RADIUS_ONE])
    open_ = _open(seen, cells)
    away = open_ & (_manhattan(_cell_of(closest, seen), cells, width) > 1)
    first_away = jnp.argmax(away)
    retreats = near & ~one_shot & stepping_in & jnp.any(away)
    retreat = _plan(_MOVE, cell=cells[first_away])
    return _choose(
        closest == ABSENT, _plan(_NOTHING), _choose(retreats, retreat, _engage(best))
    )


def _aggressive_plan(seen: _Seen, slot: Array, game: GameConfig) -> _Plan:
    unit_type = seen.unit_type[slot]
    scores = _scores(seen, slot, game)
    closest, closest_score = _least(scores, seen.enemy, seen)
    building = (seen.unit_type == TYPE_BARRACKS) | (seen.unit_type == TYPE_BASE)
    favoured = ((unit_type == TYPE_HEAVY) & building) | (
        (unit_type == TYPE_LIGHT) & (seen.unit_type == TYPE_RANGED)
    )
    favourite, _ = _least(scores, seen.enemy & favoured, seen)
    chosen = jnp.where(
        (favourite == ABSENT)
        | (favourite == closest)
        | (closest_score < _CLOSE_COMBAT),
        closest,
        favourite,
    )
    return _engage(chosen)


def _fighter_plan(seen: _Seen, slot: Array, strategy: _Strategy, game: GameConfig) -> _Plan:
    unit_type = seen.unit_type[slot]
    rush_target, _ = _least(_scores(seen, slot, game), seen.enemy, seen)
    military_rush = _choose(
        unit_type == TYPE_RANGED,
        _ranged_plan(seen, slot, game),
        _choose(
            (unit_type == TYPE_HEAVY) | (unit_type == TYPE_LIGHT),
            _aggressive_plan(seen, slot, game),
            _plan(_NOTHING),
        ),
    )
    rush = _choose(strategy.military, military_rush, _engage(rush_target))
    return _choose(strategy.defense, _defense_plan(seen, slot, game), rush)


def _worker_plan(seen: _Seen, slot: Array, game: GameConfig) -> tuple[_Plan, Array]:
    resource, _ = _nearest(seen, slot, seen.resource, game.bucket.width)
    base, _ = _nearest(seen, slot, seen.stockpile, game.bucket.width)
    either = (resource != ABSENT) | (base != ABSENT)
    return _plan(_HARVEST, target=resource, base=base), ~either


def _sites(seen: _Seen, anchor: Array, bucket: BucketSpec) -> Array:
    x, y = anchor % bucket.width, anchor // bucket.width
    edge = jnp.select(
        [x == 0, x == bucket.width - 1, y == 0, y == bucket.height - 1], [1, 2, 3, 4], 0
    )
    offsets = jnp.asarray(_SITES, jnp.int32)[edge]
    cells = _offset(anchor, offsets[:, 0], offsets[:, 1], bucket)
    return jnp.where(_open(seen, cells), cells, ABSENT)


def _base_score(seen: _Seen, builder: Array, sites: Array, width: int) -> Array:
    distance = _manhattan(sites[:, None], seen.cell[None, :], width)
    far_resource = seen.resource & (distance > 1)
    to_resource = jnp.where(
        jnp.any(far_resource, axis=1),
        jnp.min(jnp.where(far_resource, distance, _HIGHEST), axis=1),
        0,
    )
    to_enemy = jnp.max(jnp.where(seen.enemy, distance, 0), axis=1)
    return to_enemy - to_resource - _manhattan(builder, sites, width)


def _barracks_score(seen: _Seen, sites: Array, width: int) -> Array:
    distance = _manhattan(sites[:, None], seen.cell[None, :], width)
    far_bases = seen.own_bases & (distance > 1)
    to_base = jnp.max(jnp.where(far_bases, distance, 0), axis=1)
    if width < _LARGE_MAP:
        score = jnp.max(jnp.where(seen.enemy, distance, 0), axis=1) + to_base
    else:
        to_enemy = jnp.where(
            jnp.any(seen.enemy),
            jnp.min(jnp.where(seen.enemy, distance, _HIGHEST), axis=1),
            0,
        )
        score = to_base - to_enemy
    return jnp.where(_at(seen.bad_spot, sites), 0, score)


def _chosen_site(scores: Array, sites: Array, zero_best_yields: bool) -> Array:
    best = jnp.int32(0)
    site = jnp.int32(0)
    for k in range(sites.shape[0]):
        better = scores[k] > best
        if zero_best_yields:
            better = better | (best == 0)
        better = better & (sites[k] != ABSENT)
        best = jnp.where(better, scores[k], best)
        site = jnp.where(better, sites[k], site)
    return site


def _builder_plan(seen: _Seen, slot: Array, strategy: _Strategy, game: GameConfig) -> tuple[_Plan, Array]:
    bucket = game.bucket
    width = bucket.width
    start = seen.cell[slot]
    balance = seen.balance[seen.player]
    builds_base = (strategy.own[_BASES] == 0) & (balance >= game.unit_cost[TYPE_BASE])
    builds_barracks = (
        strategy.military
        & (strategy.own[_BARRACKS] < _BARRACKS_WANTED)
        & (balance >= game.unit_cost[TYPE_BARRACKS])
    )
    base_sites = _sites(seen, start, bucket)
    base_build = _plan(
        _BUILD,
        cell=_chosen_site(
            _base_score(seen, start, base_sites, width), base_sites, False
        ),
        unit_type=TYPE_BASE,
    )
    closest_base, _ = _nearest(seen, slot, seen.own_bases, width)
    anchor = jnp.where(closest_base == ABSENT, start, _cell_of(closest_base, seen))
    barracks_sites = _sites(seen, anchor, bucket)
    barracks_build = _plan(
        _BUILD,
        cell=_chosen_site(
            _barracks_score(seen, barracks_sites, width), barracks_sites, True
        ),
        unit_type=TYPE_BARRACKS,
    )
    worker, scouts = _worker_plan(seen, slot, game)
    places_base = builds_base & jnp.any(base_sites != ABSENT)
    places_barracks = ~builds_base & builds_barracks & jnp.any(barracks_sites != ABSENT)
    plan = _choose(
        places_base, base_build, _choose(places_barracks, barracks_build, worker)
    )
    return plan, ~places_base & ~places_barracks & scouts


def _errand(seen: _Seen, slot: Array, game: GameConfig, packed: bool = True) -> _Errand:
    distance = _manhattan(seen.cell[slot], seen.cell, game.bucket.width)
    resource, _ = _least(distance, seen.resource, seen, packed)
    base, to_base = _least(distance, seen.stockpile, seen, packed)
    enemy, to_enemy = _least(distance, seen.enemy, seen, packed)
    attacks = (enemy != ABSENT) & (to_enemy <= _NEAR) & (to_base <= _NEAR)
    harvests = ~attacks & (resource != ABSENT) & (base != ABSENT)
    returns = ~attacks & ~harvests & (base != ABSENT) & (seen.carried[slot] > 0)
    return _Errand(
        resource=resource,
        base=base,
        enemy=enemy,
        attacks=attacks,
        harvests=harvests,
        returns=returns,
    )


def _between(cell: Array, first: Array, last: Array, width: int) -> Array:
    x, y = cell % width, cell // width
    fx, fy = first % width, first // width
    lx, ly = last % width, last // width
    return (
        (jnp.minimum(fx, lx) <= x)
        & (x <= jnp.maximum(fx, lx))
        & (jnp.minimum(fy, ly) <= y)
        & (y <= jnp.maximum(fy, ly))
    )


def _harvester_plan(seen: _Seen, slot: Array, game: GameConfig) -> tuple[_Plan, Array, Array]:
    errand = _errand(seen, slot, game)
    blocked = (
        (errand.enemy != ABSENT)
        & _between(
            _cell_of(errand.enemy, seen),
            seen.cell[slot],
            _cell_of(errand.resource, seen),
            game.bucket.width,
        )
        & (seen.carried[slot] < 1)
    )
    plan = _choose(
        errand.attacks,
        _engage(errand.enemy),
        _plan(
            jnp.where(errand.harvests | errand.returns, _HARVEST, _NOTHING),
            target=jnp.where(errand.harvests, errand.resource, ABSENT),
            base=errand.base,
        ),
    )
    return plan, errand.harvests & blocked, errand.attacks


def _scout_plan(seen: _Seen, searched: Array, slot: Array, strategy: _Strategy, active: Array, game: GameConfig) -> _Plan:
    bucket = game.bucket
    width = bucket.width
    start = seen.cell[slot]
    enemy, score = _least(_scores(seen, slot, game), seen.enemy, seen)
    base, to_base = _nearest(seen, slot, seen.stockpile, width)
    sighted = (enemy != ABSENT) & (
        ~strategy.defense | _within_half(score, to_base, bucket)
    )
    free = _astar_free(seen, searched)
    reaches_enemy = _reaches(
        free, start, _cell_of(enemy, seen), active & sighted, bucket
    ).found
    loaded = seen.carried[slot] > 0
    has_base = base != ABSENT
    reaches_base = _reaches(
        free,
        start,
        _cell_of(base, seen),
        active & sighted & ~reaches_enemy & loaded & has_base,
        bucket,
    ).found
    reached = _choose(_stalls(seen, slot, width), _plan(_IDLE), _engage(enemy))
    returning = _choose(
        reaches_base, _plan(_HARVEST, base=base), _engage(enemy, scouting=True)
    )
    unreached = _choose(
        ~loaded,
        _engage(enemy, scouting=True),
        _choose(has_base, returning, _plan(_NOTHING)),
    )
    return _choose(sighted, _choose(reaches_enemy, reached, unreached), _plan(_NOTHING))


def _barracks_type(strategy: _Strategy) -> Array:
    preferred = jnp.int32(TYPE_HEAVY)
    most = jnp.int32(0)
    for index, trained in enumerate(_TRAINED):
        takes = strategy.production[index] > most
        preferred = jnp.where(takes, trained, preferred)
        most = jnp.where(takes, strategy.production[index], most)
    return jnp.where(strategy.military, preferred, TYPE_LIGHT)


def _role_plan(seen: _Seen, slot: Array, role: Array, strategy: _Strategy, game: GameConfig, phase: str) -> _RolePlan:
    balance = seen.balance[seen.player]
    affords_worker = balance >= game.unit_cost[TYPE_WORKER]
    keeps_training = (strategy.own[_WORKERS] < strategy.harvesters) | (
        balance > strategy.threshold
    )
    base_trains = affords_worker & (~strategy.military | keeps_training)
    barracks_type = _barracks_type(strategy)
    barracks_trains = balance >= game.unit_cost[barracks_type]
    no = jnp.bool_(False)
    yes = jnp.bool_(True)

    def base(_: None) -> _RolePlan:
        mode = jnp.where(base_trains, _TRAIN, _NOTHING)
        return _RolePlan(_plan(mode, unit_type=TYPE_WORKER), no, no)

    def barracks(_: None) -> _RolePlan:
        mode = jnp.where(barracks_trains, _TRAIN, _NOTHING)
        return _RolePlan(_plan(mode, unit_type=barracks_type), no, no)

    def fighter(_: None) -> _RolePlan:
        return _RolePlan(_fighter_plan(seen, slot, strategy, game), no, no)

    def builder(_: None) -> _RolePlan:
        plan, scouts = _builder_plan(seen, slot, strategy, game)
        return _RolePlan(plan, scouts, yes)

    def harvester(_: None) -> _RolePlan:
        plan, scouts, attacks = _harvester_plan(seen, slot, game)
        return _RolePlan(plan, scouts, ~attacks)

    def free(_: None) -> _RolePlan:
        return _RolePlan(_plan(_NOTHING), yes, yes)

    if phase == _FIRST_PASS_PHASE:
        return lax.switch(
            jnp.clip(role, _BASE_ROLE, _FIGHTER_ROLE), [base, barracks, fighter], None
        )
    return lax.switch(
        jnp.clip(role - _BUILDER_ROLE, 0, _FREE_ROLE - _BUILDER_ROLE),
        [builder, harvester, free],
        None,
    )


def _owned(seen: _Seen, leaf: Array, capacity: int) -> Array:
    return lax.dynamic_slice_in_dim(leaf, seen.first_owned, capacity)


def _entries(seen: _Seen, strategy: _Strategy, game: GameConfig) -> _Entry:
    bucket = game.bucket
    capacity = bucket.per_player_capacity
    slots = seen.first_owned + jnp.arange(capacity, dtype=jnp.int32)
    owned = _owned(seen, seen.mine, capacity)
    birth = _owned(seen, seen.birth, capacity)
    unit_type = _owned(seen, seen.unit_type, capacity)
    available = _owned(seen, seen.available, capacity)
    balance = seen.balance[seen.player]
    workers = owned & game.unit_can_harvest[unit_type]
    fighters = (
        owned & game.unit_can_attack[unit_type] & ~game.unit_can_harvest[unit_type]
    )
    rank = jnp.sum(
        workers[None, :] & (birth[None, :] < birth[:, None]), axis=1, dtype=jnp.int32
    )
    count = jnp.sum(workers, dtype=jnp.int32)
    builds_base = (
        (strategy.own[_BASES] == 0)
        & (count > 0)
        & (balance >= game.unit_cost[TYPE_BASE])
    )
    first_harvester = builds_base.astype(jnp.int32)
    harvesting = jnp.clip(
        jnp.minimum(strategy.harvesters, count - first_harvester), 0, None
    )
    builds_barracks = (
        strategy.military
        & (harvesting >= 1)
        & (strategy.own[_BARRACKS] < _BARRACKS_WANTED)
        & (balance > game.unit_cost[TYPE_BARRACKS])
    )
    first_free = first_harvester + harvesting
    from_free = builds_barracks & (harvesting == 1) & (count > first_free)
    barracks_rank = jnp.where(from_free, first_free, first_harvester)
    base_builder = workers & builds_base & (rank == 0)
    barracks_builder = workers & builds_barracks & (rank == barracks_rank)
    harvester = (
        workers & (rank >= first_harvester) & (rank < first_free) & ~barracks_builder
    )
    free = workers & (rank >= first_free) & ~barracks_builder
    errands = jax.vmap(lambda slot: _errand(seen, slot, game, packed=False))(slots)
    idle_harvester = harvester & ~(errands.attacks | errands.harvests | errands.returns)
    group = jnp.select(
        [
            fighters | (unit_type == TYPE_BASE) | (unit_type == TYPE_BARRACKS),
            base_builder,
            barracks_builder,
            harvester & ~idle_harvester,
            free,
            idle_harvester,
        ],
        [
            _FIRST_PASS,
            _BASE_BUILDER,
            _BARRACKS_BUILDER,
            _HARVESTERS,
            _FREE_WORKERS,
            _IDLE_HARVESTERS,
        ],
        _SKIPPED,
    )
    role = jnp.select(
        [
            unit_type == TYPE_BASE,
            unit_type == TYPE_BARRACKS,
            fighters,
            base_builder | barracks_builder,
            harvester & ~idle_harvester,
        ],
        [_BASE_ROLE, _BARRACKS_ROLE, _FIGHTER_ROLE, _BUILDER_ROLE, _HARVESTER_ROLE],
        _FREE_ROLE,
    )
    visits = owned & available & (group != _SKIPPED)
    order = jnp.lexsort((birth, jnp.where(visits, group, _SKIPPED + 1)))
    return _Entry(
        slot=slots[order],
        role=role[order].astype(jnp.int32),
        visits=visits[order],
    )


def _beside_listed_move(seen: _Seen, turn: _Turn, slot: Array, bucket: BucketSpec) -> Array:
    capacity = bucket.per_player_capacity
    start = seen.cell[slot]
    neighbours = jnp.stack([_offset(start, dx, dy, bucket) for dx, dy in _RADIUS_ONE])
    occupant = jnp.where(
        neighbours != ABSENT,
        seen.occupant[jnp.clip(neighbours, 0, bucket.cell_count - 1)],
        ABSENT,
    )
    local = occupant - seen.first_owned
    own = (local >= 0) & (local < capacity) & _at(seen.mine, occupant)
    return jnp.any(own & turn.listed_move[jnp.clip(local, 0, capacity - 1)])


def _accept(seen: _Seen, turn: _Turn, slot: Array, desire: _Desire, first_phase: Array, active: Array, game: GameConfig) -> _Turn:
    bucket = game.bucket
    capacity = bucket.per_player_capacity
    claims = (desire.kind == KIND_MOVE) | (desire.kind == KIND_PRODUCE)
    cell = jnp.where(claims, _step(seen.cell[slot], desire.direction, bucket), ABSENT)
    produced = jnp.clip(desire.produced_type, 0, bucket.unit_type_capacity - 1)
    cost = jnp.where(desire.kind == KIND_PRODUCE, game.unit_cost[produced], 0)
    kept = (
        active
        & (desire.kind != KIND_ABSENT)
        & ~_at(seen.claimed | turn.taken, cell)
        & _affordable(seen, cost, turn.used)
    )
    at = (jnp.arange(capacity) == slot - seen.first_owned) & kept
    claimed = kept & (jnp.arange(bucket.cell_count) == cell)
    return turn._replace(
        requested=_choose(at, desire, turn.requested),
        rank=jnp.where(at, turn.requests, turn.rank),
        requests=turn.requests + kept.astype(jnp.int32),
        taken=turn.taken | claimed,
        searched=turn.searched | (first_phase & claimed),
        used=turn.used + jnp.where(kept, cost, 0),
    )


def _visit(seen: _Seen, strategy: _Strategy, turn: _Turn, entry: _Entry, active: Array, game: GameConfig, phase: str) -> _Turn:
    bucket = game.bucket
    capacity = bucket.per_player_capacity
    slot = entry.slot
    role_plan = _role_plan(seen, slot, entry.role, strategy, game, phase)
    if phase == _FIRST_PASS_PHASE:
        desire = _carry_out(
            seen, turn.searched, slot, role_plan.plan, strategy, active, game, phase
        )
        return _accept(seen, turn, slot, desire, jnp.bool_(True), active, game)
    scouts = role_plan.scouts
    scout = _scout_plan(seen, turn.searched, slot, strategy, scouts & active, game)
    plan = _choose(scouts, scout, role_plan.plan)
    desire = _carry_out(seen, turn.searched, slot, plan, strategy, active, game, phase)
    conflicts = (
        role_plan.checks_conflicts
        & (desire.kind == KIND_MOVE)
        & _beside_listed_move(seen, turn, slot, bucket)
    )
    desire = _choose(conflicts, _idle(seen, slot, game), desire)
    listed = active & (jnp.arange(capacity) == slot - seen.first_owned)
    turn = turn._replace(
        listed_move=jnp.where(listed, desire.kind == KIND_MOVE, turn.listed_move)
    )
    return _accept(seen, turn, slot, desire, jnp.bool_(False), active, game)


def _fill(seen: _Seen, turn: _Turn, capacity: int) -> _Turn:
    owned = _owned(seen, seen.mine, capacity)
    birth = _owned(seen, seen.birth, capacity)
    available = _owned(seen, seen.available, capacity)
    requested = turn.requested
    waits = owned & available & (requested.kind == KIND_ABSENT)
    earlier = waits[None, :] & (birth[None, :] < birth[:, None])
    place = turn.requests + jnp.sum(earlier, axis=1, dtype=jnp.int32)
    return turn._replace(
        requested=requested._replace(
            kind=jnp.where(waits, KIND_NONE, requested.kind),
            none_duration=jnp.where(waits, _FILLER_DURATION, requested.none_duration),
        ),
        rank=jnp.where(waits, place, turn.rank),
        requests=turn.requests + jnp.sum(waits, dtype=jnp.int32),
    )


def _visit_count(entries: _Entry) -> Array:
    return jnp.sum(entries.visits, dtype=jnp.int32)


def _turn(seen: _Seen, strategy: _Strategy, game: GameConfig) -> _Turn:
    bucket = game.bucket
    capacity = bucket.per_player_capacity
    nothing = jnp.zeros(bucket.cell_count, jnp.bool_)
    turn = _Turn(
        requested=jax.tree.map(
            lambda leaf: jnp.full(capacity, leaf, jnp.int32), _nothing()
        ),
        rank=jnp.full(capacity, ABSENT, jnp.int32),
        requests=jnp.int32(0),
        taken=nothing,
        searched=nothing,
        used=jnp.int32(0),
        listed_move=jnp.zeros(capacity, jnp.bool_),
    )

    entries = _entries(seen, strategy, game)
    count = _visit_count(entries)
    first_pass_end = jnp.sum(
        entries.visits & (entries.role <= _FIGHTER_ROLE), dtype=jnp.int32
    )

    def walk_phase(phase: str, begin: Array, end: Array, turn: _Turn) -> _Turn:
        def walking(walk: tuple[Array, _Turn]) -> Array:
            index, _ = walk
            return index < end

        def visit(walk: tuple[Array, _Turn]) -> tuple[Array, _Turn]:
            index, turn = walk
            entry = jax.tree.map(
                lambda leaf: lax.dynamic_index_in_dim(leaf, index, keepdims=False),
                entries,
            )
            active = (index < end) & entry.visits
            return index + 1, _visit(seen, strategy, turn, entry, active, game, phase)

        _, turn = lax.while_loop(walking, visit, (begin, turn))
        return turn

    turn = walk_phase(_FIRST_PASS_PHASE, jnp.int32(0), first_pass_end, turn)
    turn = walk_phase(_WORKER_PHASE, first_pass_end, count, turn)
    return _fill(seen, turn, capacity)


def _commands(turn: _Turn, player: Array | int, game: GameConfig) -> Commands:
    fields = turn.requested._asdict() | {"request_order": turn.rank}
    unissued = empty(Commands, game.bucket)
    own_rows = (jnp.arange(PLAYER_COUNT, dtype=jnp.int32) == player)[:, None]
    return dataclasses.replace(
        unissued,
        **{
            name: jnp.where(own_rows, value.astype(jnp.int32), getattr(unissued, name))
            for name, value in fields.items()
        },
    )


def _init(game: GameConfig) -> MacroMemory:
    del game
    return MacroMemory(
        initialized=jnp.bool_(False),
        main_base=jnp.int32(ABSENT),
        enemy_base=jnp.int32(ABSENT),
        base_to_enemy_base=jnp.int32(_FAR),
        real_base_to_enemy=jnp.int32(0),
        weighted=jnp.bool_(True),
    )


def _act(key: Array, memory: MacroMemory, state: State, game: GameConfig, player: Array | int) -> tuple[MacroMemory, Commands]:
    del key
    seen = _read(state, game, player)
    memory = _initialize(memory, seen)

    plays = jnp.any(seen.mine & seen.available)

    def play(memory: MacroMemory) -> tuple[MacroMemory, Commands]:
        memory, strategy = _macro(memory, seen, plays, game)
        return memory, _commands(_turn(seen, strategy, game), player, game)

    def pass_turn(memory: MacroMemory) -> tuple[MacroMemory, Commands]:
        return memory, empty(Commands, game.bucket)

    return lax.cond(plays, play, pass_turn, memory)


tma: Final[Bot[MacroMemory]] = Bot(name="tma", init=_init, act=_act)
