from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Final, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, lax

from nanorts import kernels
from nanorts.bots import pathfinding, whole_act
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
    TYPE_RESOURCE,
    TYPE_WORKER,
    BucketSpec,
    Commands,
    GameConfig,
    ModelError,
    State,
    empty,
)

_CLOCK_WIDTH: Final[int] = 24

_LOWEST: Final[int] = int(jnp.iinfo(jnp.int32).min)
_HIGHEST: Final[int] = int(jnp.iinfo(jnp.int32).max)
_WAIT: Final[int] = 1
_LISTED_DIRECTIONS: Final[tuple[int, ...]] = (
    DIRECTION_UP,
    DIRECTION_DOWN,
    DIRECTION_LEFT,
    DIRECTION_RIGHT,
)
_AROUND: Final[tuple[tuple[int, int], ...]] = ((-1, 0), (0, 1), (0, -1), (1, 0))
_AROUND_DIRECTIONS: Final[tuple[int, ...]] = (
    DIRECTION_LEFT,
    DIRECTION_DOWN,
    DIRECTION_UP,
    DIRECTION_RIGHT,
)
_FAR_FROM_RANGED: Final[int] = 3
_MANY_COMBATANTS: Final[int] = 24
_TARGETS: Final[int] = 24
_FEW_TARGETS: Final[int] = 12
_FEW_UNITS: Final[int] = 32
_SHORT_RADIUS: Final[int] = 42
_LONG_RADIUS: Final[int] = 64
_SMALL_WIDTH: Final[int] = 12
_CROWDING_DIVISOR: Final[float] = 2.9
_SITE_RADIUS: Final[int] = 2
_LATE: Final[int] = 1000
_LEVEL_WEIGHT: Final[int] = 1000
_TARGET: Final[str] = "nanorts_mayari_act"
_TURN_COLUMNS: Final[tuple[str, ...]] = (
    "unit_attack_damage",
    "unit_attack_time",
    "unit_hit_points",
    "unit_produce_time",
    "unit_move_time",
)
_BYTES_PER_CELL: Final[int] = 8
_LANE_BYTES: Final[int] = 1024
_BYTES_PER_SLOT: Final[int] = 74
_BYTES_PER_ENTITY: Final[int] = 37
_KERNEL_ROWS: Final[int] = 32
_KERNEL_COLUMNS: Final[int] = 32


class Harvesters(NamedTuple):
    holder: Array

    order: Array

    next_order: Array


class _Seen(NamedTuple):
    player: Array
    other: Array
    start: Array
    birth: Array
    unit_type: Array
    owner: Array
    cell: Array
    hit_points: Array
    carried: Array
    pending: Array
    future_cell: Array
    remaining_move: Array
    mine: Array
    enemy: Array
    resource: Array
    occupied: Array
    wall: Array
    claimed: Array
    free: Array
    reserved: Array
    balance: Array
    time: Array
    radius: Array
    enemy_future_heavies: Array


class _Turn(NamedTuple):
    kind: Array
    direction: Array
    target_cell: Array
    produced_type: Array
    none_duration: Array
    rank: Array
    requests: Array
    taken: Array
    used: Array
    future_barracks: Array
    future_heavies: Array
    damage: Array
    harvesters: Harvesters
    draws: Array


def _diamond(radius: int) -> tuple[tuple[int, int], ...]:
    offsets = []
    for distance in range(radius + 1):
        for dx in range(-distance, distance + 1):
            dy = distance - abs(dx)
            offsets.append((dx, dy))
            if dy != 0:
                offsets.append((dx, -dy))
    return tuple(offsets)


def _xy(cell: Array, bucket: BucketSpec) -> tuple[Array, Array]:
    return cell % bucket.width, cell // bucket.width


def _manhattan(cell: Array, other: Array, bucket: BucketSpec) -> Array:
    x, y = _xy(cell, bucket)
    ox, oy = _xy(other, bucket)
    return jnp.abs(x - ox) + jnp.abs(y - oy)


def _squared(cell: Array, other: Array, bucket: BucketSpec) -> Array:
    x, y = _xy(cell, bucket)
    ox, oy = _xy(other, bucket)
    return (x - ox) * (x - ox) + (y - oy) * (y - oy)


def _diagonal(bucket: BucketSpec) -> int:
    return (bucket.width - 1) ** 2 + (bucket.height - 1) ** 2


def _floor_sqrt(value: Array, bucket: BucketSpec) -> Array:
    roots = jnp.arange(1, int(_diagonal(bucket) ** 0.5) + 2, dtype=jnp.int32)
    return jnp.sum(roots * roots <= value[..., None], axis=-1, dtype=jnp.int32)


def _offset_cell(cell: Array, dx: Array | int, dy: Array | int, bucket: BucketSpec) -> Array:
    x, y = _xy(cell, bucket)
    nx, ny = x + dx, y + dy
    inside = (nx >= 0) & (nx < bucket.width) & (ny >= 0) & (ny < bucket.height)
    return jnp.where(inside, ny * bucket.width + nx, ABSENT)


def _step(cell: Array, direction: Array, bucket: BucketSpec) -> Array:
    known = jnp.clip(direction, 0, DIRECTION_COUNT - 1)
    dx = jnp.asarray([dx for dx, _ in DIRECTION_OFFSETS], jnp.int32)[known]
    dy = jnp.asarray([dy for _, dy in DIRECTION_OFFSETS], jnp.int32)[known]
    stepped = _offset_cell(cell, dx, dy, bucket)
    real = (direction >= 0) & (direction < DIRECTION_COUNT)
    return jnp.where(real, stepped, ABSENT)


def _future(cell: Array, direction: Array, bucket: BucketSpec) -> Array:
    stepped = _step(cell, direction, bucket)
    return jnp.where(stepped == ABSENT, cell, stepped)


def _at(cells: Array, cell: Array) -> Array:
    known = jnp.clip(cell, 0, cells.shape[-1] - 1)
    return (cell != ABSENT) & cells[..., known]


def _closest(origin: Array, among: Array, seen: _Seen, bucket: BucketSpec) -> Array:
    distance = jnp.where(among, _manhattan(origin, seen.cell, bucket), _HIGHEST)
    nearest = distance == jnp.min(distance)
    first = jnp.argmin(jnp.where(among & nearest, seen.birth, _HIGHEST))
    return jnp.where(jnp.any(among), first, ABSENT).astype(jnp.int32)


def _min_distance(origin: Array, cells: Array, bucket: BucketSpec) -> Array:
    every = jnp.arange(bucket.cell_count, dtype=jnp.int32)
    return jnp.min(jnp.where(cells, _manhattan(origin, every, bucket), _HIGHEST))


def _distance_field(cells: Array, bucket: BucketSpec) -> Array:
    grid = cells.reshape(bucket.height, bucket.width)
    columns = jnp.arange(bucket.width, dtype=jnp.int32)
    rows = jnp.arange(bucket.height, dtype=jnp.int32)
    beyond = jnp.int32(bucket.width + bucket.height)
    across = jnp.min(
        jnp.where(grid[:, None, :], jnp.abs(columns[:, None] - columns), beyond),
        axis=-1,
    )
    field = jnp.min(across + jnp.abs(rows[:, None] - rows)[:, :, None], axis=1)
    return jnp.where(jnp.any(cells), field.reshape(-1), _HIGHEST)


def _distance_at(field: Array, cell: Array) -> Array:
    return field[jnp.clip(cell, 0, field.shape[-1] - 1)]


def _board(units: Array, seen: _Seen, bucket: BucketSpec) -> Array:
    return (
        jnp.zeros(bucket.cell_count, jnp.bool_)
        .at[jnp.where(units, seen.cell, bucket.cell_count)]
        .set(True, mode="drop")
    )


def _structures(seen: _Seen, bucket: BucketSpec) -> Array:
    buildings = _of_type(seen, TYPE_BASE) | _of_type(seen, TYPE_BARRACKS)
    return _board((seen.mine | seen.enemy) & buildings, seen, bucket)


def _first_true(mask: Array) -> Array:
    return jnp.where(jnp.any(mask), jnp.argmax(mask), ABSENT).astype(jnp.int32)


def _read(state: State, game: GameConfig, player: Array | int) -> _Seen:
    bucket = game.bucket
    player = jnp.asarray(player, jnp.int32)
    present = state.active
    unit_type = jnp.clip(state.unit_type.astype(jnp.int32), 0, None)
    owner = state.owner.astype(jnp.int32)
    cell = state.cell.astype(jnp.int32)
    pending = state.pending_kind.astype(jnp.int32)
    direction = state.pending_direction.astype(jnp.int32)
    moving = present & (pending == KIND_MOVE)
    resource = present & (unit_type == TYPE_RESOURCE)
    claims = present & ((pending == KIND_MOVE) | (pending == KIND_PRODUCE))
    claimed = (
        jnp.zeros(bucket.cell_count, jnp.bool_)
        .at[jnp.where(claims, _step(cell, direction, bucket), bucket.cell_count)]
        .set(True, mode="drop")
    )
    produced = jnp.clip(state.pending_produced_type.astype(jnp.int32), 0, None)
    cost = jnp.where(present & (pending == KIND_PRODUCE), game.unit_cost[produced], 0)
    reserved = jnp.sum(
        cost[bucket.neutral_capacity :].reshape(
            PLAYER_COUNT, bucket.per_player_capacity
        ),
        axis=-1,
        dtype=jnp.int32,
    )
    enemy = present & ~resource & (owner != OWNER_NEUTRAL) & (owner != player)
    heavy_ordered = present & (pending == KIND_PRODUCE) & (produced == TYPE_HEAVY)
    return _Seen(
        player=player,
        other=1 - player,
        start=bucket.neutral_capacity + player * bucket.per_player_capacity,
        birth=state.birth,
        unit_type=unit_type,
        owner=owner,
        cell=cell,
        hit_points=state.hit_points.astype(jnp.int32),
        carried=state.carried_resources,
        pending=jnp.where(present, pending, KIND_ABSENT),
        future_cell=jnp.where(moving, _future(cell, direction, bucket), cell),
        remaining_move=jnp.where(
            moving, state.pending_completes_at - state.time, _HIGHEST
        ),
        mine=present & ~resource & (owner == player),
        enemy=enemy,
        resource=resource,
        occupied=state.entity_at.reshape(-1) != ABSENT,
        wall=game.terrain.reshape(-1),
        claimed=claimed,
        free=pathfinding.free_cells(state, game),
        reserved=reserved,
        balance=state.player_resources,
        time=state.time,
        radius=jnp.where(
            jnp.sum(present, dtype=jnp.int32) > _FEW_UNITS, _SHORT_RADIUS, _LONG_RADIUS
        ),
        enemy_future_heavies=jnp.sum(heavy_ordered & enemy, dtype=jnp.int32),
    )


def _of_type(seen: _Seen, unit_type: int) -> Array:
    return seen.unit_type == unit_type


def _start_turn(seen: _Seen, harvesters: Harvesters, game: GameConfig, state: State) -> _Turn:
    bucket = game.bucket
    capacity = bucket.per_player_capacity
    present = state.active
    producing = present & ~seen.resource & (seen.pending == KIND_PRODUCE)
    produced = jnp.clip(state.pending_produced_type.astype(jnp.int32), 0, None)
    built = _future(seen.cell, state.pending_direction.astype(jnp.int32), bucket)
    taken = (
        jnp.zeros(bucket.cell_count, jnp.bool_)
        .at[jnp.where(producing, built, bucket.cell_count)]
        .set(True, mode="drop")
    )
    mine = producing & seen.mine
    barracks_built = mine & (produced == TYPE_BARRACKS)
    attacking = present & ~seen.resource & (seen.pending == KIND_ATTACK)
    target = state.pending_target_cell.astype(jnp.int32)
    victim = state.entity_at.reshape(-1)[jnp.clip(target, 0, bucket.cell_count - 1)]
    hits = attacking & (target != ABSENT) & (victim != ABSENT)
    damage = (
        jnp.zeros(bucket.entity_capacity, jnp.int32)
        .at[jnp.where(hits, victim, bucket.entity_capacity)]
        .add(game.unit_attack_damage[seen.unit_type], mode="drop")
    )
    own_birth = lax.dynamic_slice_in_dim(seen.birth, seen.start, capacity)
    based = jnp.any(seen.mine & _of_type(seen, TYPE_BASE))
    alive = (harvesters.holder != ABSENT) & (harvesters.holder == own_birth) & based
    absent = jnp.full(capacity, ABSENT, jnp.int32)
    return _Turn(
        kind=absent,
        direction=absent,
        target_cell=absent,
        produced_type=absent,
        none_duration=jnp.zeros(capacity, jnp.int32),
        rank=absent,
        requests=jnp.int32(0),
        taken=taken,
        used=jnp.sum(jnp.where(mine, game.unit_cost[produced], 0), dtype=jnp.int32),
        future_barracks=(
            jnp.zeros(bucket.cell_count, jnp.bool_)
            .at[jnp.where(barracks_built, built, bucket.cell_count)]
            .set(True, mode="drop")
        ),
        future_heavies=jnp.sum(mine & (produced == TYPE_HEAVY), dtype=jnp.int32),
        damage=damage,
        harvesters=Harvesters(
            holder=jnp.where(alive, harvesters.holder, ABSENT),
            order=jnp.where(alive, harvesters.order, ABSENT),
            next_order=harvesters.next_order,
        ),
        draws=jnp.int32(0),
    )


def _busy(seen: _Seen, turn: _Turn, slot: Array) -> Array:
    local = slot - seen.start
    return (turn.kind[jnp.clip(local, 0, None)] != KIND_ABSENT) | (
        seen.pending[slot] != KIND_ABSENT
    )


def _busy_all(seen: _Seen, turn: _Turn) -> Array:
    requested = lax.dynamic_update_slice_in_dim(
        jnp.zeros_like(seen.pending, jnp.bool_), turn.kind != KIND_ABSENT, seen.start, 0
    )
    return requested | (seen.pending != KIND_ABSENT)


def _issue(seen: _Seen, turn: _Turn, slot: Array, when: Array, kind: int, *, direction: Array | int = ABSENT, target_cell: Array | int = ABSENT, produced_type: Array | int = ABSENT, none_duration: Array | int = 0) -> _Turn:
    local = slot - seen.start
    at = (jnp.arange(turn.kind.shape[0]) == local) & when
    first = at & (turn.kind == KIND_ABSENT)

    def put(value: Array | int, old: Array) -> Array:
        return jnp.where(at, jnp.asarray(value, jnp.int32), old)

    return turn._replace(
        kind=put(kind, turn.kind),
        direction=put(direction, turn.direction),
        target_cell=put(target_cell, turn.target_cell),
        produced_type=put(produced_type, turn.produced_type),
        none_duration=put(none_duration, turn.none_duration),
        rank=jnp.where(first, turn.requests, turn.rank),
        requests=turn.requests + jnp.any(first).astype(jnp.int32),
    )


def _lock(turn: _Turn, cell: Array, when: Array) -> _Turn:
    every = jnp.arange(turn.taken.shape[0])
    return turn._replace(taken=turn.taken | (when & (every == cell)))


def _consistent(seen: _Seen, cell: Array, cost: Array | int) -> Array:
    mine = seen.reserved[seen.player]
    theirs = seen.reserved[seen.other]
    return (
        ~_at(seen.claimed, cell)
        & ((mine == 0) | (cost + mine <= seen.balance[seen.player]))
        & ((theirs == 0) | (theirs <= seen.balance[seen.other]))
    )


def _move_allowed(seen: _Seen, target: Array) -> Array:
    return (
        (target != ABSENT)
        & ~_at(seen.wall, target)
        & ~_at(seen.occupied, target)
        & _consistent(seen, target, 0)
    )


def _pos_free(seen: _Seen, turn: _Turn, cell: Array) -> Array:
    return ~_at(turn.taken, cell) & ~_at(seen.occupied, cell) & ~_at(seen.wall, cell)


def _neighbours(cell: Array, bucket: BucketSpec) -> Array:
    return jnp.stack([_offset_cell(cell, dx, dy, bucket) for dx, dy in _AROUND])


def _openable(seen: _Seen, cell: Array) -> Array:
    return (cell != ABSENT) & ~_at(seen.wall, cell) & ~_at(seen.occupied, cell)


_LEVELS_PER_TRIP: Final[int] = 16


def _rows(cells: Array, bucket: BucketSpec) -> Array:
    grid = cells.reshape(bucket.height, bucket.width).astype(jnp.uint32)
    bits = jnp.left_shift(jnp.uint32(1), jnp.arange(bucket.width, dtype=jnp.uint32))
    return jnp.sum(grid * bits, axis=-1, dtype=jnp.uint32)


def _spread_rows(front: Array, bucket: BucketSpec) -> Array:
    full = jnp.uint32((1 << bucket.width) - 1)
    empty = jnp.zeros(1, jnp.uint32)
    return (
        (jnp.left_shift(front, jnp.uint32(1)) & full)
        | jnp.right_shift(front, jnp.uint32(1))
        | jnp.concatenate([empty, front[:-1]])
        | jnp.concatenate([front[1:], empty])
    )


def _row_bit(rows: Array, cell: Array, bucket: BucketSpec) -> Array:
    known = jnp.clip(cell, 0, bucket.cell_count - 1)
    word = rows[known // bucket.width]
    bit = jnp.right_shift(word, (known % bucket.width).astype(jnp.uint32)) & jnp.uint32(
        1
    )
    return (cell != ABSENT) & (bit != 0)


def _proximity(seen: _Seen, turn: _Turn, goal: Array, neighbour: Array, active: Array, bucket: BucketSpec) -> Array:
    every = jnp.arange(bucket.cell_count, dtype=jnp.int32)
    blocked = _rows(seen.wall | turn.taken | seen.occupied, bucket)
    opening = _openable(seen, neighbour)
    decisive = _diagonal(bucket) < _LEVEL_WEIGHT**2
    radius = seen.radius

    def spreads(carry: tuple[Array, Array, Array, Array]) -> Array:
        front, _, near, level = carry
        going = active & (level < radius) & jnp.any(front != 0)
        if decisive:
            going = going & ~jnp.any(opening & (near != _HIGHEST))
        return going

    def spread(carry: tuple[Array, Array, Array, Array]) -> tuple[Array, Array, Array, Array]:
        front, reached, near, level = carry
        for step in range(_LEVELS_PER_TRIP):
            on = active & (level + step < radius)
            new = jnp.where(
                on, _spread_rows(front, bucket) & ~blocked & ~reached, jnp.uint32(0)
            )
            reached = reached | new
            near = jnp.where(
                (near == _HIGHEST) & _row_bit(new, neighbour, bucket),
                level + step + 1,
                near,
            )
            front = new
        return front, reached, near, level + _LEVELS_PER_TRIP

    initial = _rows(every == goal, bucket)
    near = jnp.where(_row_bit(initial, neighbour, bucket), 0, _HIGHEST).astype(
        jnp.int32
    )
    _, _, near, _ = lax.while_loop(
        spreads, spread, (initial, initial, near, jnp.int32(0))
    )
    return near


def _breadth_first_direction(seen: _Seen, turn: _Turn, start: Array, goal: Array, active: Array, bucket: BucketSpec) -> tuple[Array, Array]:
    neighbour = _neighbours(start, bucket)
    near = _proximity(seen, turn, goal, neighbour, active, bucket)
    open_ = _openable(seen, neighbour) & (near != _HIGHEST)
    fit = -_LEVEL_WEIGHT * near - _floor_sqrt(_squared(neighbour, goal, bucket), bucket)
    best = jnp.argmax(jnp.where(open_, fit, _LOWEST))
    return jnp.asarray(_AROUND_DIRECTIONS, jnp.int32)[best], jnp.any(open_)


def _plan_move(seen: _Seen, turn: _Turn, slot: Array, goal: Array, active: Array, bucket: BucketSpec) -> tuple[Array, Array, Array]:
    start = seen.cell[slot]
    free = seen.free & ~turn.taken
    path = pathfinding.search(
        free, start, jnp.where(active, goal, start), jnp.int32(1), bucket
    )
    stepped = active & path.found & (path.step != ABSENT)
    toward, reaches_open = _breadth_first_direction(
        seen, turn, start, goal, stepped, bucket
    )
    direction = jnp.where(reaches_open, toward, path.step)
    target = _step(start, direction, bucket)
    moved = stepped & _move_allowed(seen, target) & ~_at(turn.taken, target)
    return direction, target, moved


def _apply_move(seen: _Seen, turn: _Turn, slot: Array, direction: Array, target: Array, moved: Array) -> _Turn:
    turn = _issue(seen, turn, slot, moved, KIND_MOVE, direction=direction)
    return _lock(turn, target, moved)


def _move_towards(seen: _Seen, turn: _Turn, slot: Array, goal: Array, active: Array, bucket: BucketSpec) -> tuple[_Turn, Array]:
    direction, target, moved = _plan_move(seen, turn, slot, goal, active, bucket)
    return _apply_move(seen, turn, slot, direction, target, moved), moved


def _shuffled_move(seen: _Seen, turn: _Turn, slot: Array, passing: Array, key: Array, bucket: BucketSpec) -> _Turn:
    drawn = jax.random.categorical(
        jax.random.fold_in(key, turn.draws), jnp.where(passing, 0.0, -jnp.inf)
    ).astype(jnp.int32)
    moves = jnp.any(passing)
    target = _step(seen.cell[slot], drawn, bucket)
    turn = _issue(seen, turn, slot, moves, KIND_MOVE, direction=drawn)
    return _lock(turn, target, moves)._replace(
        draws=turn.draws + moves.astype(jnp.int32)
    )


def _directions(seen: _Seen, slot: Array, bucket: BucketSpec) -> Array:
    return jnp.stack(
        [_step(seen.cell[slot], jnp.int32(d), bucket) for d in range(DIRECTION_COUNT)]
    )


def _move_away(seen: _Seen, turn: _Turn, slot: Array, active: Array, key: Array, bucket: BucketSpec) -> _Turn:
    targets = _directions(seen, slot, bucket)
    passing = active & jax.vmap(
        lambda target: _pos_free(seen, turn, target) & _move_allowed(seen, target)
    )(targets)
    return _shuffled_move(seen, turn, slot, passing, key, bucket)


def _move_closer(seen: _Seen, turn: _Turn, slot: Array, goal: Array, active: Array, key: Array, bucket: BucketSpec) -> _Turn:
    targets = _directions(seen, slot, bucket)
    closer = _manhattan(jnp.clip(targets, 0, None), goal, bucket) < _manhattan(
        seen.cell[slot], goal, bucket
    )
    passing = active & jax.vmap(
        lambda target, nearer: (
            nearer & _pos_free(seen, turn, target) & _move_allowed(seen, target)
        )
    )(targets, closer)
    return _shuffled_move(seen, turn, slot, passing, key, bucket)


def _produce(seen: _Seen, turn: _Turn, slot: Array, direction: Array, produced_type: int, active: Array, game: GameConfig) -> tuple[_Turn, Array]:
    bucket = game.bucket
    cost = game.unit_cost[produced_type]
    target = _future(seen.cell[slot], direction, bucket)
    made = (
        active
        & ~_busy(seen, turn, slot)
        & (seen.balance[seen.player] - turn.used >= cost)
        & _pos_free(seen, turn, target)
        & _consistent(seen, target, cost)
    )
    turn = _issue(
        seen,
        turn,
        slot,
        made,
        KIND_PRODUCE,
        direction=direction,
        produced_type=produced_type,
    )
    turn = _lock(turn, target, made)
    every = jnp.arange(bucket.cell_count)
    if produced_type == TYPE_BARRACKS:
        turn = turn._replace(
            future_barracks=turn.future_barracks | (made & (every == target))
        )
    if produced_type == TYPE_HEAVY:
        turn = turn._replace(
            future_heavies=turn.future_heavies + made.astype(jnp.int32)
        )
    return turn._replace(used=turn.used + jnp.where(made, cost, 0)), made


def _produce_first(seen: _Seen, turn: _Turn, slot: Array, directions: Array, produced_type: int, active: Array, game: GameConfig) -> tuple[_Turn, Array]:
    bucket = game.bucket
    cost = game.unit_cost[produced_type]
    target = jax.vmap(lambda d: _future(seen.cell[slot], d, bucket))(directions)
    possible = (
        active
        & ~_busy(seen, turn, slot)
        & (seen.balance[seen.player] - turn.used >= cost)
        & jax.vmap(
            lambda cell: _pos_free(seen, turn, cell) & _consistent(seen, cell, cost)
        )(target)
    )
    chosen = _first_true(possible)
    return _produce(
        seen,
        turn,
        slot,
        directions[jnp.clip(chosen, 0, None)],
        produced_type,
        chosen != ABSENT,
        game,
    )


def _reachable(free: Array, start: Array, targets: Array, bucket: BucketSpec) -> Array:
    every = jnp.arange(bucket.cell_count, dtype=jnp.int32)
    seeds = jnp.zeros(bucket.cell_count, jnp.bool_)
    for dx, dy in DIRECTION_OFFSETS:
        beside = jnp.where(start == ABSENT, ABSENT, _offset_cell(start, dx, dy, bucket))
        seeds = seeds | ((beside != ABSENT) & (every == beside))
    seeds = _rows(seeds, bucket) & free

    def spread(reached: Array) -> Array:
        return free & (reached | _spread_rows(reached, bucket))

    def grow(carry: tuple[Array, Array]) -> tuple[Array, Array]:
        reached, _ = carry
        grown = reached
        for _ in range(_LEVELS_PER_TRIP):
            grown = spread(grown)
        return grown, reached

    reached, _ = lax.while_loop(
        lambda carry: jnp.any(carry[0] != carry[1]) & ~jnp.any(carry[0] & targets),
        grow,
        (spread(seeds), seeds),
    )
    return reached


def _separated(seen: _Seen, turn: _Turn, slot: Array, active: Array, bucket: BucketSpec) -> Array:
    start = seen.cell[slot]
    free = _rows(seen.free & ~turn.taken, bucket)
    beside = jnp.stack(
        [_offset_cell(seen.cell, dx, dy, bucket) for dx, dy in DIRECTION_OFFSETS]
    )
    adjacent = _squared(start, seen.cell, bucket) <= 1
    counted = (beside != ABSENT) & (seen.enemy & ~adjacent)[None, :]
    targets = _rows(
        jnp.zeros(bucket.cell_count, jnp.bool_)
        .at[jnp.where(counted, beside, bucket.cell_count)]
        .set(True, mode="drop"),
        bucket,
    )
    reached = _reachable(free, jnp.where(active, start, ABSENT), targets, bucket)
    return ~jnp.any(reached & targets)


def _targets_tried(seen: _Seen, game: GameConfig) -> Array:
    combatants = jnp.sum(
        seen.enemy & game.unit_can_attack[seen.unit_type], dtype=jnp.int32
    )
    cut = jnp.where(combatants > _MANY_COMBATANTS, _FEW_TARGETS, _TARGETS)
    return jnp.minimum(jnp.sum(seen.enemy, dtype=jnp.int32), cut)


def _within(attacker_type: Array, squared: Array, game: GameConfig) -> Array:
    reach = game.unit_attack_range[attacker_type]
    return squared <= reach * reach


class _Targets(NamedTuple):
    order: Array
    limit: Array
    succeeds: Array
    soon: Array
    threatened: Array
    allowed: Array


def _ranked_targets(seen: _Seen, turn: _Turn, slot: Array, able: Array, game: GameConfig) -> _Targets:
    bucket = game.bucket
    unit_type = seen.unit_type[slot]
    squared = _squared(seen.cell[slot], seen.cell, bucket)
    in_range = _within(unit_type, squared, game)
    escapes = seen.remaining_move <= game.unit_attack_time[unit_type]
    succeeds = in_range & ~escapes
    threatened = _within(seen.unit_type, squared, game)
    damage = game.unit_attack_damage[unit_type]
    kills = damage > seen.hit_points
    ranged_reach = game.unit_attack_range[TYPE_RANGED] + _FAR_FROM_RANGED
    to_worker = (unit_type == TYPE_RANGED) & (seen.unit_type == TYPE_WORKER)
    score = (
        1000 * succeeds
        + 100 * kills
        + (to_worker + game.unit_attack_damage[seen.unit_type]) * 10
        + threatened
    ).astype(jnp.int32)
    hopeless = (seen.hit_points <= turn.damage) | (
        squared > ranged_reach * ranged_reach
    )
    score = jnp.where(hopeless, _LOWEST, score)
    order = jnp.lexsort(
        (seen.birth, -jnp.where(seen.enemy, score.astype(jnp.float32), -jnp.inf))
    )
    return _Targets(
        order=order,
        limit=jnp.where(able, _targets_tried(seen, game), 0),
        succeeds=succeeds,
        soon=_within(
            unit_type, _squared(seen.cell[slot], seen.future_cell, bucket), game
        ),
        threatened=threatened,
        allowed=_consistent(seen, jnp.int32(ABSENT), 0),
    )


def _attack_nearby(seen: _Seen, turn: _Turn, slot: Array, real: Array, game: GameConfig) -> _Turn:
    unit_type = seen.unit_type[slot]
    able = (
        real
        & seen.mine[slot]
        & ~_busy(seen, turn, slot)
        & game.unit_can_attack[unit_type]
    )
    targets = _ranked_targets(seen, turn, slot, able, game)
    order, limit = targets.order, targets.limit
    tries = jnp.arange(order.shape[0], dtype=jnp.int32)
    hits = (targets.succeeds & targets.allowed)[order]
    waits = (~targets.succeeds & targets.soon)[order]
    decisive = (tries < limit) & (hits | waits)
    first_decisive = jnp.where(jnp.any(decisive), jnp.argmax(decisive), limit).astype(
        jnp.int32
    )
    chases = (tries < first_decisive) & (
        ~targets.succeeds & ~targets.soon & targets.threatened
    )[order]
    chase_rank = jnp.cumsum(chases, dtype=jnp.int32) - 1
    chase_count = jnp.sum(chases, dtype=jnp.int32)

    def chase(carry: tuple[Array, Array, Array, Array]) -> tuple[Array, Array, Array, Array]:
        tried, moved, _, _ = carry
        target = order[jnp.argmax(chases & (chase_rank == tried))]
        active = ~moved & (tried < chase_count)
        direction, cell, moved = _plan_move(
            seen, turn, slot, seen.cell[target], active, game.bucket
        )
        return tried + 1, moved, direction, cell

    _, moved, direction, cell = lax.while_loop(
        lambda carry: ~carry[1] & (carry[0] < chase_count),
        chase,
        (jnp.int32(0), jnp.bool_(False), jnp.int32(ABSENT), jnp.int32(ABSENT)),
    )
    turn = _apply_move(seen, turn, slot, direction, cell, moved)
    acts = ~moved & (first_decisive < limit)
    decisive_try = jnp.minimum(first_decisive, order.shape[0] - 1)
    target = order[decisive_try]
    hit = acts & hits[decisive_try]
    turn = _issue(seen, turn, slot, hit, KIND_ATTACK, target_cell=seen.cell[target])
    turn = turn._replace(
        damage=turn.damage
        + jnp.where(
            hit & (jnp.arange(turn.damage.shape[0]) == target),
            game.unit_hit_points[unit_type],
            0,
        )
    )
    return _issue(
        seen, turn, slot, acts & waits[decisive_try], KIND_NONE, none_duration=_WAIT
    )


def _in_contact(seen: _Seen, game: GameConfig) -> Array:
    bucket = game.bucket
    capacity = bucket.per_player_capacity
    cell = lax.dynamic_slice_in_dim(seen.cell, seen.start, capacity)[:, None]
    unit_type = lax.dynamic_slice_in_dim(seen.unit_type, seen.start, capacity)[:, None]
    squared = _squared(cell, seen.cell, bucket)
    near = (
        _within(unit_type, squared, game)
        | _within(unit_type, _squared(cell, seen.future_cell, bucket), game)
        | _within(seen.unit_type, squared, game)
    )
    touching = jnp.any(near & seen.enemy, axis=-1)
    return lax.dynamic_update_slice_in_dim(
        jnp.zeros_like(seen.mine), touching, seen.start, 0
    )


def _owned_in_unit_order(seen: _Seen, units: Array, capacity: int) -> Array:
    owned = lax.dynamic_slice_in_dim(units, seen.start, capacity)
    birth = lax.dynamic_slice_in_dim(seen.birth, seen.start, capacity)
    order = jnp.argsort(jnp.where(owned, birth, _HIGHEST)).astype(jnp.int32)
    return jnp.where(owned[order], seen.start + order, ABSENT)


def _owned_in_pass_order(seen: _Seen, passes: tuple[Array, ...], capacity: int) -> Array:
    rank = jnp.full(capacity, len(passes), jnp.int32)
    for index, units in enumerate(passes):
        owned = lax.dynamic_slice_in_dim(units, seen.start, capacity)
        rank = jnp.where(owned, index, rank)
    listed = rank < len(passes)
    birth = lax.dynamic_slice_in_dim(seen.birth, seen.start, capacity)
    order = jnp.lexsort((jnp.where(listed, birth, _HIGHEST), rank)).astype(jnp.int32)
    return jnp.where(listed[order], seen.start + order, ABSENT)


def _visit_count(slots: Array) -> Array:
    return jnp.sum(slots != ABSENT, dtype=jnp.int32)


def _each[Carry](slots: Array, carry: Carry, visit: Callable[[Carry, Array, Array], Carry]) -> Carry:
    count = _visit_count(slots)

    def body(walked: tuple[Array, Carry]) -> tuple[Array, Carry]:
        index, carry = walked
        slot = slots[jnp.minimum(index, slots.shape[0] - 1)]
        return index + 1, visit(carry, jnp.clip(slot, 0, None), slot != ABSENT)

    return lax.while_loop(
        lambda walked: walked[0] < count, body, (jnp.int32(0), carry)
    )[1]


def _all_attack_nearby(seen: _Seen, turn: _Turn, game: GameConfig) -> _Turn:
    capacity = game.bucket.per_player_capacity

    def visit(turn: _Turn, slot: Array, real: Array) -> _Turn:
        return _attack_nearby(seen, turn, slot, real, game)

    acting = (
        seen.mine
        & game.unit_can_attack[seen.unit_type]
        & ~_busy_all(seen, turn)
        & _in_contact(seen, game)
    )
    return _each(_owned_in_unit_order(seen, acting, capacity), turn, visit)


def _needs_barracks(seen: _Seen, turn: _Turn, game: GameConfig) -> Array:
    bucket = game.bucket
    capacity = bucket.per_player_capacity
    bases = seen.mine & _of_type(seen, TYPE_BASE)
    barracks = seen.mine & _of_type(seen, TYPE_BARRACKS)
    reach = bucket.width // 4
    planned = jnp.sum(turn.future_barracks, dtype=jnp.int32)
    few = jnp.sum(barracks, dtype=jnp.int32) + planned < jnp.sum(bases, dtype=jnp.int32)
    from_planned = _distance_field(turn.future_barracks, bucket)

    def needs(base: Array) -> Array:
        cell = seen.cell[base]
        far_from_planned = _distance_at(from_planned, cell) > reach
        nearest = _closest(cell, barracks, seen, bucket)
        near_built = (nearest != ABSENT) & (
            _manhattan(cell, seen.cell[jnp.clip(nearest, 0, None)], bucket) < reach
        )
        return bases[base] & far_from_planned & ~near_built

    slots = _owned_in_unit_order(seen, bases, capacity)
    return few & jnp.any(jax.vmap(needs)(jnp.clip(slots, 0, None)) & (slots != ABSENT))


def _site_worker_scores(seen: _Seen, turn: _Turn, site: Array, game: GameConfig) -> Array:
    bucket = game.bucket
    busy = _busy_all(seen, turn)
    barracks_time = (
        game.unit_produce_time[TYPE_BARRACKS] // game.unit_move_time[TYPE_WORKER]
    )
    heavy_time = game.unit_produce_time[TYPE_HEAVY] // game.unit_move_time[TYPE_WORKER]
    danger_time = barracks_time + heavy_time
    enemy = jnp.clip(_closest(site, seen.enemy, seen, bucket), 0, None)
    enemy_cell = seen.cell[enemy]
    enemy_distance = jnp.maximum(_manhattan(site, enemy_cell, bucket), 1)
    sx, sy = _xy(site, bucket)
    ex, ey = _xy(enemy_cell, bucket)
    wx, wy = _xy(seen.cell, bucket)
    beside = ~(
        ((wx < sx) & (ex < sx))
        | ((wx > sx) & (ex > sx))
        | ((wy < sy) & (ey < sy))
        | ((wy > sy) & (ey > sy))
    )
    penalty = jnp.where(
        enemy_distance < danger_time,
        (2 * danger_time) // enemy_distance - jnp.where(beside, 3, 0),
        0,
    )
    score = -penalty - _manhattan(seen.cell, site, bucket) // 2
    return jnp.where(busy, _LOWEST, score).astype(jnp.int32)


def _blocking(seen: _Seen, bucket: BucketSpec) -> Array:
    occupant_type = jnp.full(bucket.cell_count, ABSENT, jnp.int32)
    occupant_type = occupant_type.at[
        jnp.where(seen.mine | seen.enemy | seen.resource, seen.cell, bucket.cell_count)
    ].set(seen.unit_type, mode="drop")
    return (occupant_type == TYPE_RESOURCE) | (occupant_type == TYPE_BASE)


def _block_penalty(site_x: Array, site_y: Array, blocking: Array, diagonals: bool, bucket: BucketSpec) -> Array:
    total = jnp.int32(0)
    for dx, dy in _diamond(2):
        distance = abs(dx) + abs(dy)
        if distance == 0 or (distance == 2 and not diagonals):
            continue
        x, y = site_x + dx, site_y + dy
        inside = (x >= 0) & (x < bucket.width) & (y >= 0) & (y < bucket.height)
        cell = jnp.where(inside, y * bucket.width + x, 0)
        total = total + jnp.where(inside & blocking[cell], 4 if distance == 1 else 1, 0)
    return total


def _barracks_starts(seen: _Seen, turn: _Turn, game: GameConfig) -> Array:
    bases = seen.mine & _of_type(seen, TYPE_BASE)
    workers = seen.mine & _of_type(seen, TYPE_WORKER)
    affordable = (
        seen.balance[seen.player] - turn.used - 1 >= game.unit_cost[TYPE_BARRACKS]
    )
    return (
        affordable
        & _needs_barracks(seen, turn, game)
        & jnp.any(bases)
        & jnp.any(workers)
    )


class _Sites(NamedTuple):
    cell: Array
    listed: Array
    score: Array


class _Scored(NamedTuple):
    rank: Array
    separated: Array
    nearest: Array
    open_score: Array
    closed_score: Array


def _site_cells(seen: _Seen, base_slots: Array, bucket: BucketSpec) -> tuple[Array, Array, Array]:
    offsets = _diamond(_SITE_RADIUS)
    bx, by = _xy(seen.cell[jnp.clip(base_slots, 0, None)], bucket)
    site_x = bx[:, None] + jnp.asarray([dx for dx, _ in offsets])
    site_y = by[:, None] + jnp.asarray([dy for _, dy in offsets])
    inside = (
        (site_x >= 0)
        & (site_x < bucket.width)
        & (site_y >= 0)
        & (site_y < bucket.height)
    )
    return site_x, site_y, jnp.where(inside, site_y * bucket.width + site_x, ABSENT)


def _sites(seen: _Seen, turn: _Turn, starts: Array, game: GameConfig) -> _Sites:
    bucket = game.bucket
    capacity = bucket.per_player_capacity
    bases = seen.mine & _of_type(seen, TYPE_BASE)
    barracks = seen.mine & _of_type(seen, TYPE_BARRACKS)
    workers = seen.mine & _of_type(seen, TYPE_WORKER)
    base_slots = _owned_in_unit_order(seen, bases, capacity)
    site_x, site_y, site = _site_cells(seen, base_slots, bucket)
    known_site = jnp.clip(site, 0, None)
    listed = jnp.broadcast_to((base_slots != ABSENT)[:, None], site.shape)
    building = _structures(seen, bucket)
    valid = listed & (site != ABSENT) & ~seen.wall[known_site] & ~building[known_site]
    all_barracks = turn.future_barracks | _board(barracks, seen, bucket)
    from_barracks = _distance_field(all_barracks, bucket)
    blocking = _blocking(seen, bucket)

    def visit(scored: _Scored, slot: Array, real: Array) -> _Scored:
        every = jnp.arange(seen.birth.shape[0], dtype=jnp.int32)
        ranks = jnp.arange(capacity, dtype=jnp.int32)
        apart = _separated(seen, turn, slot, real & starts, bucket)
        row = jnp.minimum(scored.rank, capacity - 1)
        cells = known_site[row]
        near = jax.vmap(lambda cell: _closest(cell, bases, seen, bucket))(cells)
        near = jnp.clip(near, 0, None)
        deserted = jnp.where(
            jnp.any(all_barracks),
            _distance_at(from_barracks, seen.cell[near]) // 2,
            0,
        )
        best = jax.vmap(
            lambda cell: jnp.max(
                jnp.where(workers, _site_worker_scores(seen, turn, cell, game), _LOWEST)
            )
        )(cells)
        penalty = jax.vmap(lambda x, y: _block_penalty(x, y, blocking, False, bucket))(
            site_x[row], site_y[row]
        )
        diagonal_penalty = jax.vmap(
            lambda x, y: _block_penalty(x, y, blocking, True, bucket)
        )(site_x[row], site_y[row])
        at = real & (ranks == scored.rank)[:, None]
        open_score = 10 * (deserted - penalty + best)
        return _Scored(
            rank=scored.rank + real.astype(jnp.int32),
            separated=jnp.where(real & (every == slot), apart, scored.separated),
            nearest=jnp.where(at, near, scored.nearest),
            open_score=jnp.where(at, open_score, scored.open_score),
            closed_score=jnp.where(at, -diagonal_penalty * 10, scored.closed_score),
        )

    unscored = jnp.zeros(site.shape, jnp.int32)
    scored = _each(
        base_slots,
        _Scored(
            rank=jnp.int32(0),
            separated=jnp.zeros(seen.birth.shape, jnp.bool_),
            nearest=unscored,
            open_score=unscored,
            closed_score=unscored,
        ),
        visit,
    )
    score = jnp.where(
        scored.separated[scored.nearest], scored.closed_score, scored.open_score
    )
    score = jnp.where(valid & jnp.any(workers), score, _LOWEST)
    return _Sites(
        cell=site.reshape(-1),
        listed=listed.reshape(-1),
        score=score.reshape(-1).astype(jnp.int32),
    )


def _build_barracks(seen: _Seen, turn: _Turn, game: GameConfig) -> _Turn:
    bucket = game.bucket
    workers = seen.mine & _of_type(seen, TYPE_WORKER)
    starts = _barracks_starts(seen, turn, game)
    sites = _sites(seen, turn, starts, game)
    known_site = jnp.clip(sites.cell, 0, None)

    def attempt(carry: tuple[Array, _Turn, Array, Array]) -> tuple[Array, _Turn, Array, Array]:
        tries, turn, remaining, _ = carry
        index = jnp.argmax(jnp.where(remaining, sites.score, _LOWEST))
        chosen_score = jnp.where(jnp.any(remaining), sites.score[index], _LOWEST)
        row = jnp.where(
            workers, _site_worker_scores(seen, turn, known_site[index], game), _LOWEST
        )
        worker = jnp.argmin(
            jnp.where(workers & (row == jnp.max(row)), seen.birth, _HIGHEST)
        ).astype(jnp.int32)
        goes = (chosen_score != _LOWEST) & (row[worker] != _LOWEST)
        cell = sites.cell[index]
        beside = _manhattan(seen.cell[worker], cell, bucket) == 1
        turn, moved = _move_towards(seen, turn, worker, cell, goes & ~beside, bucket)
        toward = _toward(seen.cell[worker], cell, bucket)
        turn, built = _produce(
            seen, turn, worker, toward, TYPE_BARRACKS, goes & beside, game
        )
        done = ~goes | moved | built
        return (
            tries + 1,
            turn,
            remaining & (jnp.arange(remaining.shape[0]) != index),
            done,
        )

    _, turn, _, _ = lax.while_loop(
        lambda carry: ~carry[3] & (carry[0] < 2),
        attempt,
        (jnp.int32(0), turn, sites.listed, ~starts),
    )
    return turn


def _toward(origin: Array, goal: Array, bucket: BucketSpec) -> Array:
    ox, oy = _xy(origin, bucket)
    gx, gy = _xy(goal, bucket)
    dx, dy = gx - ox, gy - oy
    horizontal = jnp.where(dx > 0, DIRECTION_RIGHT, DIRECTION_LEFT)
    vertical = jnp.where(dy > 0, DIRECTION_DOWN, DIRECTION_UP)
    return jnp.where(jnp.abs(dx) > jnp.abs(dy), horizontal, vertical).astype(jnp.int32)


def _build_base(seen: _Seen, turn: _Turn, game: GameConfig) -> _Turn:
    bucket = game.bucket
    workers = seen.mine & _of_type(seen, TYPE_WORKER)
    bases = seen.mine & _of_type(seen, TYPE_BASE)
    resource_board = _board(seen.resource, seen, bucket)
    starts = (
        jnp.any(workers)
        & ~jnp.any(bases)
        & jnp.any(seen.resource)
        & (seen.balance[seen.player] - turn.used >= game.unit_cost[TYPE_BASE])
    )
    busy = _busy_all(seen, turn)
    distance = _distance_at(_distance_field(resource_board, bucket), seen.cell)
    distance = jnp.where(busy, _HIGHEST, distance)
    distance = jnp.where(workers, distance, _HIGHEST)
    nearest = jnp.argmin(
        jnp.where(
            workers & (distance == jnp.min(jnp.where(workers, distance, _HIGHEST))),
            seen.birth,
            _HIGHEST,
        )
    ).astype(jnp.int32)
    goes = starts & ~busy[nearest]
    building = _structures(seen, bucket)
    listed = jnp.asarray(_LISTED_DIRECTIONS, jnp.int32)
    sites = jax.vmap(lambda d: _future(seen.cell[nearest], d, bucket))(listed)
    valid = ~seen.wall[sites] & ~building[sites]
    cost = game.unit_cost[TYPE_BASE]
    possible = (
        goes
        & valid
        & jax.vmap(
            lambda cell: _pos_free(seen, turn, cell) & _consistent(seen, cell, cost)
        )(sites)
    )
    chosen = _first_true(possible)
    turn, _ = _produce(
        seen,
        turn,
        nearest,
        listed[jnp.clip(chosen, 0, None)],
        TYPE_BASE,
        chosen != ABSENT,
        game,
    )
    return turn


def _enemy_heavies_weak(seen: _Seen, game: GameConfig) -> Array:
    bucket = game.bucket
    heavies = seen.enemy & _of_type(seen, TYPE_HEAVY)
    count = jnp.sum(heavies, dtype=jnp.int32)
    lone_hit_points = jnp.max(jnp.where(heavies, seen.hit_points, _LOWEST))
    enemy_bases = _board(seen.enemy & _of_type(seen, TYPE_BASE), seen, bucket)
    resources = _board(seen.resource, seen, bucket)
    workers = seen.enemy & _of_type(seen, TYPE_WORKER)
    base_distance = _distance_at(_distance_field(enemy_bases, bucket), seen.cell)
    resource_distance = jnp.where(
        seen.carried > 0,
        0,
        _distance_at(_distance_field(resources, bucket), seen.cell),
    )
    near = (base_distance + resource_distance) < bucket.width // 2
    total = seen.balance[seen.other] + jnp.sum(workers & near, dtype=jnp.int32)
    return (
        (seen.enemy_future_heavies == 0)
        & (count <= 1)
        & ((count == 0) | (lone_hit_points <= 3))
        & (total < game.unit_cost[TYPE_HEAVY])
    )


def _combat_directions(seen: _Seen, slot: Array, enemies: Array, bucket: BucketSpec) -> Array:
    listed = jnp.asarray(_LISTED_DIRECTIONS, jnp.int32)
    cells = jax.vmap(lambda d: _future(seen.cell[slot], d, bucket))(listed)
    score = -jax.vmap(lambda cell: _min_distance(cell, enemies, bucket))(cells)
    return listed[jnp.lexsort((jnp.arange(4), -score))]


def _barracks_action(seen: _Seen, turn: _Turn, game: GameConfig) -> _Turn:
    bucket = game.bucket
    capacity = bucket.per_player_capacity
    barracks = seen.mine & _of_type(seen, TYPE_BARRACKS)
    weak = _enemy_heavies_weak(seen, game)
    exhausted = ~jnp.any(seen.resource)
    enemies = _board(seen.enemy, seen, bucket)

    def visit(turn: _Turn, slot: Array, real: Array) -> _Turn:
        idle = real & ~_busy(seen, turn, slot)
        apart = _separated(seen, turn, slot, idle, bucket)
        directions = _combat_directions(seen, slot, enemies, bucket)

        def train(turn: _Turn, produced_type: int, active: Array) -> tuple[_Turn, Array]:
            return _produce_first(
                seen, turn, slot, directions, produced_type, active, game
            )

        turn, done = train(turn, TYPE_RANGED, idle & apart)
        turn, made = train(turn, TYPE_HEAVY, idle & ~done)
        done = done | made
        turn, made = train(turn, TYPE_RANGED, idle & ~done & weak)
        done = done | made
        poor = seen.balance[seen.player] - turn.used < game.unit_cost[TYPE_HEAVY]
        turn, _ = train(turn, TYPE_RANGED, idle & ~done & exhausted & poor)
        return turn

    return _each(_owned_in_unit_order(seen, barracks, capacity), turn, visit)


def _workers_per_base(seen: _Seen, turn: _Turn, slot: Array, active: Array, game: GameConfig) -> Array:
    bucket = game.bucket
    if bucket.width < 9:
        return jnp.int32(15)
    if bucket.width > 16:
        return jnp.int32(2)
    apart = _separated(seen, turn, slot, active, bucket)
    enemy_workers = jnp.sum(seen.enemy & _of_type(seen, TYPE_WORKER), dtype=jnp.int32)
    enemy_bases = jnp.sum(seen.enemy & _of_type(seen, TYPE_BASE), dtype=jnp.int32)
    below = enemy_workers // jnp.maximum(enemy_bases, 1)
    return jnp.where(apart | (seen.time > _LATE), 2, jnp.maximum(below, 2))


def _build_worker_direction(seen: _Seen, turn: _Turn, slot: Array, game: GameConfig) -> Array:
    bucket = game.bucket
    cell = seen.cell[slot]
    enemy = _closest(cell, seen.enemy, seen, bucket)
    resource = _closest(cell, seen.resource, seen, bucket)
    workers = jnp.sum(seen.mine & _of_type(seen, TYPE_WORKER), dtype=jnp.int32)
    bases = jnp.sum(seen.mine & _of_type(seen, TYPE_BASE), dtype=jnp.int32)
    toward_enemy = (resource == ABSENT) | (workers >= 2 * bases)
    goal = jnp.where(toward_enemy, enemy, resource)
    goal_cell = seen.cell[jnp.clip(goal, 0, None)]
    listed = jnp.asarray(_LISTED_DIRECTIONS, jnp.int32)
    near = jax.vmap(lambda d: _future(cell, d, bucket))(listed)
    beyond = jax.vmap(lambda n, d: _future(n, d, bucket))(near, listed)
    usable = (
        ~seen.wall[near]
        & ~seen.occupied[near]
        & jax.vmap(lambda c: _pos_free(seen, turn, c))(beyond)
        & (enemy != ABSENT)
    )
    score = -_manhattan(near, goal_cell, bucket)
    best = jnp.argmax(jnp.where(usable, score, _LOWEST))
    return jnp.where(jnp.any(usable), listed[best], DIRECTION_UP).astype(jnp.int32)


def _bases_action(seen: _Seen, turn: _Turn, game: GameConfig) -> _Turn:
    bucket = game.bucket
    capacity = bucket.per_player_capacity
    bases = seen.mine & _of_type(seen, TYPE_BASE)
    producing = jnp.sum(bases & (seen.pending != KIND_ABSENT), dtype=jnp.int32)
    workers = jnp.sum(seen.mine & _of_type(seen, TYPE_WORKER), dtype=jnp.int32)
    base_count = jnp.sum(bases, dtype=jnp.int32)
    listed = jnp.asarray(_LISTED_DIRECTIONS, jnp.int32)

    def visit(carry: tuple[_Turn, Array], slot: Array, real: Array) -> tuple[_Turn, Array]:
        turn, made_workers = carry
        idle = real & ~_busy(seen, turn, slot)
        per_base = _workers_per_base(seen, turn, slot, idle, game)
        only = ~jnp.any(seen.resource) & (seen.balance[seen.player] - turn.used == 1)
        turn, _ = _produce_first(
            seen, turn, slot, listed, TYPE_WORKER, idle & only, game
        )
        enough = (bucket.width >= 9) & (
            workers + made_workers + producing >= per_base * base_count
        )
        goes = idle & ~only & ~enough
        direction = _build_worker_direction(seen, turn, slot, game)
        turn, made = _produce(seen, turn, slot, direction, TYPE_WORKER, goes, game)
        turn, anywhere = _produce_first(
            seen, turn, slot, listed, TYPE_WORKER, goes & ~made, game
        )
        return turn, made_workers + (made | anywhere).astype(jnp.int32)

    turn, _ = _each(
        _owned_in_unit_order(seen, bases, capacity), (turn, jnp.int32(0)), visit
    )
    return turn


def _go_harvesting(seen: _Seen, turn: _Turn, slot: Array, active: Array, key: Array, game: GameConfig) -> _Turn:
    bucket = game.bucket
    cell = seen.cell[slot]
    resource = _closest(cell, seen.resource, seen, bucket)
    found = active & (resource != ABSENT)
    goal = seen.cell[jnp.clip(resource, 0, None)]
    beside = _manhattan(cell, goal, bucket) == 1
    harvests = (
        found
        & beside
        & ~_busy(seen, turn, slot)
        & _consistent(seen, jnp.int32(ABSENT), 0)
    )
    turn = _issue(
        seen, turn, slot, harvests, KIND_HARVEST, direction=_toward(cell, goal, bucket)
    )
    turn, moved = _move_towards(seen, turn, slot, goal, found & ~beside, bucket)
    return _move_away(seen, turn, slot, found & ~beside & ~moved, key, bucket)


def _harvest_score(seen: _Seen, turn: _Turn, bases: Array, game: GameConfig) -> Array:
    bucket = game.bucket
    busy = _busy_all(seen, turn)
    to_resource = _distance_at(
        _distance_field(_board(seen.resource, seen, bucket), bucket), seen.cell
    )
    to_base = _distance_at(
        _distance_field(_board(bases, seen, bucket), bucket), seen.cell
    )
    missing = ~jnp.any(seen.resource) | ~jnp.any(bases)
    return jnp.where(
        busy | (seen.carried > 0) | missing, _HIGHEST, to_resource + to_base
    )


def _remembered_slots(seen: _Seen, harvesters: Harvesters) -> Array:
    held = harvesters.holder != ABSENT
    in_order = jnp.argsort(jnp.where(held, harvesters.order, _HIGHEST)).astype(
        jnp.int32
    )
    return jnp.where(held[in_order], seen.start + in_order, ABSENT)


def _worker_action(seen: _Seen, turn: _Turn, key: Array, game: GameConfig) -> _Turn:
    bucket = game.bucket
    capacity = bucket.per_player_capacity
    entities = seen.birth.shape[0]
    every = jnp.arange(entities, dtype=jnp.int32)
    workers = seen.mine & _of_type(seen, TYPE_WORKER)
    bases = seen.mine & _of_type(seen, TYPE_BASE)
    area = bucket.width * bucket.height
    occupancy = (
        jnp.sum((seen.mine | seen.enemy) & _of_type(seen, TYPE_WORKER), dtype=jnp.int32)
        + jnp.sum(
            (seen.mine | seen.enemy) & game.unit_can_attack[seen.unit_type],
            dtype=jnp.int32,
        )
        + jnp.sum((seen.mine | seen.enemy) & _of_type(seen, TYPE_BASE), dtype=jnp.int32)
        + jnp.sum(
            (seen.mine | seen.enemy) & _of_type(seen, TYPE_BARRACKS), dtype=jnp.int32
        )
        + jnp.sum(seen.resource, dtype=jnp.int32)
    )
    crowded = (bucket.width <= _SMALL_WIDTH) & (
        occupancy > int(area / _CROWDING_DIVISOR)
    )
    per_base = jnp.where(crowded, 1, 2)

    remembered_slots = _remembered_slots(seen, turn.harvesters)

    def remember(carry: tuple[_Turn, Array, Array], slot: Array, real: Array) -> tuple[_Turn, Array, Array]:
        turn, unassigned, counts = carry
        base = _closest(seen.cell[slot], bases, seen, bucket)
        idle = real & ~_busy(seen, turn, slot) & (seen.carried[slot] == 0)
        turn = _go_harvesting(seen, turn, slot, idle, key, game)
        return (
            turn,
            unassigned & ~(real & (every == slot)),
            counts + (real & (base != ABSENT) & (every == base)).astype(jnp.int32),
        )

    turn, unassigned, counts = _each(
        remembered_slots,
        (turn, workers, jnp.zeros(entities, jnp.int32)),
        remember,
    )
    open_bases = bases & (counts < per_base)

    def recruit(carry: tuple[_Turn, Array, Array, Array, Array]) -> tuple[_Turn, Array, Array, Array, Array]:
        turn, unassigned, open_bases, counts, _ = carry
        score = jnp.where(
            unassigned, _harvest_score(seen, turn, open_bases, game), _HIGHEST
        )
        lowest = jnp.min(score)
        worker = jnp.argmin(
            jnp.where(unassigned & (score == lowest), seen.birth, _HIGHEST)
        ).astype(jnp.int32)
        goes = lowest != _HIGHEST
        base = jnp.clip(_closest(seen.cell[worker], open_bases, seen, bucket), 0, None)
        turn = _go_harvesting(seen, turn, worker, goes, key, game)
        local = worker - seen.start
        own = jnp.arange(capacity) == local
        harvesters = turn.harvesters
        turn = turn._replace(
            harvesters=Harvesters(
                holder=jnp.where(goes & own, seen.birth[worker], harvesters.holder),
                order=jnp.where(goes & own, harvesters.next_order, harvesters.order),
                next_order=harvesters.next_order + goes.astype(jnp.int32),
            )
        )
        counts = jnp.where(goes & (every == base) & (counts == 0), 1, counts)
        open_bases = open_bases & ~(goes & (every == base) & (counts >= per_base))
        unassigned = unassigned & ~(goes & (every == worker))
        return turn, unassigned, open_bases, counts, goes

    def recruiting(carry: tuple[_Turn, Array, Array, Array, Array]) -> Array:
        _, unassigned, open_bases, _, went = carry
        return went & jnp.any(open_bases) & jnp.any(unassigned)

    turn, _, _, _, _ = lax.while_loop(
        recruiting,
        recruit,
        (turn, unassigned, open_bases, counts, jnp.bool_(True)),
    )

    def deliver(turn: _Turn, slot: Array, real: Array) -> _Turn:
        cell = seen.cell[slot]
        base = _closest(cell, bases, seen, bucket)
        goes = (
            real
            & ~_busy(seen, turn, slot)
            & (seen.carried[slot] > 0)
            & (base != ABSENT)
        )
        goal = seen.cell[jnp.clip(base, 0, None)]
        beside = _manhattan(cell, goal, bucket) <= 1
        returns = goes & beside & _consistent(seen, jnp.int32(ABSENT), 0)
        turn = _issue(
            seen,
            turn,
            slot,
            returns,
            KIND_RETURN,
            direction=_toward(cell, goal, bucket),
        )
        turn, _ = _move_towards(seen, turn, slot, goal, goes & ~beside, bucket)
        return turn

    delivering = workers & ~_busy_all(seen, turn) & (seen.carried > 0) & jnp.any(bases)
    return _each(_owned_in_unit_order(seen, delivering, capacity), turn, deliver)


def _workers_attack(seen: _Seen, turn: _Turn, game: GameConfig) -> Array:
    bucket = game.bucket
    if bucket.width <= _SMALL_WIDTH:
        return jnp.bool_(True)
    mine = seen.mine
    return (
        _enemy_heavies_weak(seen, game)
        & ~jnp.any(seen.enemy & _of_type(seen, TYPE_RANGED))
        & ~jnp.any(mine & _of_type(seen, TYPE_HEAVY))
        & (turn.future_heavies == 0)
        & ~jnp.any(mine & _of_type(seen, TYPE_RANGED))
    )


def _overpowering(seen: _Seen, game: GameConfig) -> Array:
    damage = game.unit_attack_damage[seen.unit_type]
    attacks = game.unit_can_attack[seen.unit_type]
    power = jnp.sum(jnp.where(seen.mine & attacks, damage, 0), dtype=jnp.int32)
    enemy_power = jnp.sum(jnp.where(seen.enemy & attacks, damage, 0), dtype=jnp.int32)
    return power - enemy_power > 0


def _go_combat(seen: _Seen, turn: _Turn, passes: tuple[Array, ...], key: Array, game: GameConfig) -> _Turn:
    bucket = game.bucket
    capacity = bucket.per_player_capacity
    candidates = jnp.sum(seen.enemy, dtype=jnp.int32)
    tried_at_most = _targets_tried(seen, game)
    overpowering = _overpowering(seen, game)

    def visit(turn: _Turn, slot: Array, real: Array) -> _Turn:
        unit_type = seen.unit_type[slot]
        able = real & ~_busy(seen, turn, slot) & game.unit_can_attack[unit_type]
        score = -_manhattan(seen.cell[slot], seen.cell, bucket)
        if bucket.width > 9:
            score = score + jnp.where(
                (unit_type == TYPE_RANGED) & (seen.unit_type == TYPE_RANGED), 2, 0
            )
        if bucket.width >= 16:
            score = score + jnp.where(
                ((unit_type == TYPE_HEAVY) | (unit_type == TYPE_RANGED))
                & (seen.unit_type == TYPE_BARRACKS),
                bucket.width,
                0,
            )
        order = jnp.lexsort(
            (seen.birth, -jnp.where(seen.enemy, score.astype(jnp.float32), -jnp.inf))
        )
        beside = _neighbours(seen.cell[slot], bucket)
        free = seen.free & ~turn.taken
        boxed = ~jnp.any((beside != ABSENT) & free[jnp.clip(beside, 0, None)])
        limit = jnp.where(able & ~boxed, tried_at_most, 0)

        def attempt(carry: tuple[Array, Array, Array, Array]) -> tuple[Array, Array, Array, Array]:
            index, moved, _, _ = carry
            direction, cell, moved = _plan_move(
                seen,
                turn,
                slot,
                seen.future_cell[order[index]],
                ~moved & (index < limit),
                bucket,
            )
            return index + jnp.where(moved, 0, 1), moved, direction, cell

        tried, moved, direction, cell = lax.while_loop(
            lambda carry: ~carry[1] & (carry[0] < limit),
            attempt,
            (jnp.int32(0), jnp.bool_(False), jnp.int32(ABSENT), jnp.int32(ABSENT)),
        )
        turn = _apply_move(seen, turn, slot, direction, cell, moved)
        tried = jnp.where(able & boxed, tried_at_most, tried)
        stuck = (
            able & (candidates > 0) & (tried >= candidates) & (unit_type == TYPE_RANGED)
        )
        turn = _move_away(seen, turn, slot, stuck & overpowering, key, bucket)
        nearest = seen.cell[order[0]]
        return _move_closer(seen, turn, slot, nearest, stuck, key, bucket)

    able = ~_busy_all(seen, turn) & game.unit_can_attack[seen.unit_type]
    return _each(
        _owned_in_pass_order(seen, tuple(units & able for units in passes), capacity),
        turn,
        visit,
    )


def _free_blocks(seen: _Seen, turn: _Turn, workers: Array, key: Array, game: GameConfig) -> _Turn:
    bucket = game.bucket
    capacity = bucket.per_player_capacity

    def visit(turn: _Turn, slot: Array, real: Array) -> _Turn:
        crowded = jnp.any(_at(seen.occupied, _neighbours(seen.cell[slot], bucket)))
        idle = real & ~_busy(seen, turn, slot) & crowded
        return _move_away(seen, turn, slot, idle, key, bucket)

    crowded = jnp.any(_at(seen.occupied, _neighbours(seen.cell, bucket)), axis=0)
    listed = workers & ~_busy_all(seen, turn) & crowded
    return _each(_owned_in_unit_order(seen, listed, capacity), turn, visit)


def _fill_with_nones(seen: _Seen, turn: _Turn, game: GameConfig) -> _Turn:
    capacity = game.bucket.per_player_capacity
    owned = lax.dynamic_slice_in_dim(seen.mine, seen.start, capacity)
    birth = lax.dynamic_slice_in_dim(seen.birth, seen.start, capacity)
    pending = lax.dynamic_slice_in_dim(seen.pending, seen.start, capacity)
    waits = owned & (pending == KIND_ABSENT) & (turn.kind == KIND_ABSENT)
    earlier = waits[None, :] & (birth[None, :] < birth[:, None])
    place = turn.requests + jnp.sum(earlier, axis=1, dtype=jnp.int32)
    return turn._replace(
        kind=jnp.where(waits, KIND_NONE, turn.kind),
        none_duration=jnp.where(waits, _WAIT, turn.none_duration),
        rank=jnp.where(waits, place, turn.rank),
        requests=turn.requests + jnp.sum(waits, dtype=jnp.int32),
    )


def _turn(key: Array, harvesters: Harvesters, state: State, game: GameConfig, player: Array | int) -> _Turn:
    seen = _read(state, game, player)
    turn = _start_turn(seen, harvesters, game, state)
    turn = _all_attack_nearby(seen, turn, game)
    turn = _build_barracks(seen, turn, game)
    turn = _build_base(seen, turn, game)
    turn = _barracks_action(seen, turn, game)
    turn = _bases_action(seen, turn, game)
    turn = _worker_action(seen, turn, key, game)
    workers = seen.mine & _of_type(seen, TYPE_WORKER)
    attack = _workers_attack(seen, turn, game)
    turn = lax.cond(
        attack,
        lambda turn: turn,
        lambda turn: _free_blocks(seen, turn, workers & ~attack, key, game),
        turn,
    )
    combatants = tuple(
        seen.mine & _of_type(seen, unit_type)
        for unit_type in (TYPE_HEAVY, TYPE_RANGED, TYPE_LIGHT)
    )
    turn = _go_combat(seen, turn, (workers & attack, *combatants), key, game)
    return _fill_with_nones(seen, turn, game)


def _commands(turn: _Turn, player: Array | int, game: GameConfig) -> Commands:
    fields = {
        "kind": turn.kind,
        "direction": turn.direction,
        "none_duration": turn.none_duration,
        "target_cell": turn.target_cell,
        "produced_type": turn.produced_type,
        "request_order": turn.rank,
    }
    unissued = empty(Commands, game.bucket)
    own_rows = (jnp.arange(PLAYER_COUNT, dtype=jnp.int32) == player)[:, None]
    return dataclasses.replace(
        unissued,
        **{
            name: jnp.where(own_rows, value.astype(jnp.int32), getattr(unissued, name))
            for name, value in fields.items()
        },
    )


def _init(game: GameConfig) -> Harvesters:
    bucket = game.bucket
    if bucket.width >= _CLOCK_WIDTH:
        raise ModelError(
            f"mayari refuses a {bucket.height}x{bucket.width} game: from "
            f"{_CLOCK_WIDTH} cells wide its Java AI reads the clock"
        )
    capacity = bucket.per_player_capacity
    return Harvesters(
        holder=jnp.full(capacity, ABSENT, jnp.int32),
        order=jnp.full(capacity, ABSENT, jnp.int32),
        next_order=jnp.int32(0),
    )


def _act(key: Array, carry: Harvesters, state: State, game: GameConfig, player: Array | int) -> tuple[Harvesters, Commands]:
    if not _fits(game.bucket):
        return _act_by_program(key, carry, state, game, player)
    return kernels.dispatch(
        key, carry, state, game, player, kernel=_act_by_kernel, program=_act_by_program
    )


def _fits(bucket: BucketSpec) -> bool:
    return (
        bucket.height <= _KERNEL_ROWS
        and bucket.width <= _KERNEL_COLUMNS
        and whole_act.fits(
            bucket,
            bytes_per_cell=_BYTES_PER_CELL,
            lane_bytes=_LANE_BYTES,
            bytes_per_slot=_BYTES_PER_SLOT,
            bytes_per_entity=_BYTES_PER_ENTITY,
        )
    )


def _draw_ranks(key: Array, bucket: BucketSpec) -> Array:
    every = jnp.arange(DIRECTION_COUNT, dtype=jnp.int32)

    def rank(draw: Array) -> Array:
        noise = jax.random.gumbel(jax.random.fold_in(key, draw), (DIRECTION_COUNT,))
        ahead = (noise[None, :] > noise[:, None]) | (
            (noise[None, :] == noise[:, None]) & (every[None, :] < every[:, None])
        )
        place = jnp.sum(ahead, axis=1, dtype=jnp.int32)
        return jnp.sum(place << (2 * every), dtype=jnp.int32).astype(jnp.uint8)

    draws = jnp.arange(2 * bucket.per_player_capacity, dtype=jnp.int32)
    return jax.vmap(rank)(draws)


def _act_by_kernel(key: Array, carry: Harvesters, state: State, game: GameConfig, player: Array | int) -> tuple[Harvesters, Commands]:
    return whole_act.act(
        _TARGET,
        carry,
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
            _draw_ranks(key, game.bucket),
        ],
    )


def _act_by_program(key: Array, carry: Harvesters, state: State, game: GameConfig, player: Array | int) -> tuple[Harvesters, Commands]:
    turn = _turn(key, carry, state, game, player)
    return turn.harvesters, _commands(turn, player, game)


mayari: Final[Bot[Harvesters]] = Bot(name="mayari", init=_init, act=_act)
