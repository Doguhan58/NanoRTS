from __future__ import annotations

import dataclasses
import functools
from typing import Final, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, lax

from nanorts import core, kernels
from nanorts.bots import whole_act
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
    TYPE_RANGED,
    TYPE_RESOURCE,
    TYPE_WORKER,
    BucketSpec,
    Commands,
    GameConfig,
    Legality,
    State,
    attack_offsets,
    empty,
    window_targets,
)

_SOLDIER: Final[int] = 0
_HARVESTER: Final[int] = 1
_BUILDER: Final[int] = 2
_JOB_COUNT: Final[int] = 3
_NONE_DURATION: Final[int] = 10
_SOLDIER_ATTACK_SCORE: Final[int] = 1
_WORKER_ATTACK_SCORE: Final[int] = 2
_ARRIVAL_SCORE: Final[int] = 1
_BUILD_SITE_SCORE: Final[int] = 8
_BUILD_SITE_RADIUS: Final[int] = 1
_BASE_SITE_RADIUS: Final[int] = 2
_FAR: Final[int] = 1 << 20
_AFTER_EVERY_ATTACK: Final[int] = 1 << 30
_LAST: Final[int] = (1 << 31) - 1
_DISTANCE_COLUMNS: Final[tuple[int, ...]] = (0, 1, 2)
_BUILD_SITE_COLUMN: Final[int] = 3
_BASE_SITE_COLUMN: Final[int] = 4
_TARGET_BIRTH_COLUMN: Final[int] = 5
_DIRECTIONAL_KINDS: Final[tuple[int, ...]] = (
    KIND_MOVE,
    KIND_HARVEST,
    KIND_RETURN,
    KIND_PRODUCE,
)

_TARGET: Final[str] = "nanorts_obi_act"

_BYTES_PER_CELL: Final[int] = 18

_LANE_BYTES: Final[int] = 16

_BYTES_PER_CANDIDATE: Final[int] = 11

_BYTES_PER_TYPE: Final[int] = 4

_BYTES_PER_SLOT: Final[int] = 31


class Jobs(NamedTuple):
    job: Array
    holder: Array
    harvester_given: Array
    builder_given: Array


class _Candidates(NamedTuple):
    kind: tuple[int, ...]
    direction: tuple[int, ...]
    produced_type: tuple[int, ...]
    offset: tuple[int, ...]
    list_position: tuple[int, ...]
    directional: tuple[bool, ...]


class _Distances(NamedTuple):
    enemy: Array
    resource: Array
    own_base: Array


class _Pending(NamedTuple):
    birth: Array
    claims: Array
    claimed_cell: Array
    produces: Array
    cost: Array
    is_base: Array


class _Unit(NamedTuple):
    available: Array
    birth: Array
    cell: Array
    unit_type: Array
    carrying: Array
    marked: Array


class _Fields(NamedTuple):
    cell_facts: Array
    no_barracks: Array


class _Reading(NamedTuple):
    units: _Unit
    fields: _Fields
    pending: _Pending
    balance: Array


class _VisitedUnit(NamedTuple):
    slot: Array
    available: Array
    cell: Array
    unit_type: Array
    carrying: Array
    scored_as: Array
    marked: Array


class _Visit(NamedTuple):
    marked: Array
    scores: Array
    accepted: Array
    list_position: Array


class _Judged(NamedTuple):
    visit: _Visit
    rewrite: Array
    neighbour: Array


class _Walking(NamedTuple):
    position: Array
    attempt: Array
    remaining: Array
    chosen: Array
    claimed: Array
    rewritten: Array


@functools.cache
def _candidates(bucket: BucketSpec) -> _Candidates:
    directions = range(DIRECTION_COUNT)
    rows = (
        *(
            (KIND_ATTACK, ABSENT, ABSENT, offset)
            for offset in range(bucket.attack_offset_capacity)
        ),
        *((KIND_HARVEST, d, ABSENT, ABSENT) for d in directions),
        *((KIND_RETURN, d, ABSENT, ABSENT) for d in directions),
        *(
            (KIND_PRODUCE, d, produced, ABSENT)
            for produced in range(bucket.unit_type_capacity)
            for d in directions
        ),
        *((KIND_MOVE, d, ABSENT, ABSENT) for d in directions),
        (KIND_NONE, ABSENT, ABSENT, ABSENT),
    )
    kind, direction, produced_type, offset = zip(*rows, strict=True)
    return _Candidates(
        kind,
        direction,
        produced_type,
        offset,
        tuple(_AFTER_EVERY_ATTACK + position for position in range(len(rows))),
        tuple(row_kind in _DIRECTIONAL_KINDS for row_kind in kind),
    )


@functools.cache
def _adjacent_attack_order(bucket: BucketSpec) -> tuple[int, ...]:
    steps = [tuple(step) for step in DIRECTION_OFFSETS]
    return tuple(
        steps.index(offset) if offset in steps else _AFTER_EVERY_ATTACK
        for offset in (tuple(row) for row in attack_offsets(bucket).tolist())
    )


def _column(values: tuple[int, ...]) -> Array:
    return jnp.asarray(values, jnp.int32)


def _flags(values: tuple[bool, ...]) -> Array:
    return jnp.asarray(values, jnp.bool_)


def _directional(kind: Array) -> Array:
    return jnp.isin(kind, _column(_DIRECTIONAL_KINDS))


def _owned(field: Array, bucket: BucketSpec) -> Array:
    return field[bucket.neutral_capacity :].reshape(
        PLAYER_COUNT, bucket.per_player_capacity
    )


def _step(cell: Array, direction: Array, bucket: BucketSpec) -> tuple[Array, Array]:
    steps = jnp.asarray(DIRECTION_OFFSETS, jnp.int32)
    known = jnp.clip(direction, 0, DIRECTION_COUNT - 1)
    x = cell % bucket.width + steps[known, 0]
    y = cell // bucket.width + steps[known, 1]
    on_board = (x >= 0) & (x < bucket.width) & (y >= 0) & (y < bucket.height)
    return jnp.clip(y * bucket.width + x, 0, bucket.cell_count - 1), on_board


def _board(units: Array, cell: Array, bucket: BucketSpec) -> Array:
    return (
        jnp.zeros(bucket.cell_count, jnp.bool_)
        .at[jnp.where(units, cell, bucket.cell_count)]
        .set(True, mode="drop")
    )


def _neighbour_minimum(field: Array) -> Array:
    padded = jnp.pad(field, ((0, 0), (1, 1), (1, 1)), constant_values=_FAR)
    return jnp.minimum(
        jnp.minimum(padded[:, :-2, 1:-1], padded[:, 2:, 1:-1]),
        jnp.minimum(padded[:, 1:-1, :-2], padded[:, 1:-1, 2:]),
    )


def _sweep_offsets(passable: Array, axis: int, reverse: bool) -> Array:
    length = passable.shape[axis]
    position = jnp.arange(length, dtype=jnp.int32).reshape(
        tuple(length if dimension == axis else 1 for dimension in range(passable.ndim))
    )
    impassable = lax.cumsum((~passable).astype(jnp.int32), axis, reverse=reverse)
    return (-position if reverse else position) + _FAR * impassable


def _sweep(field: Array, passable: Array, offset: Array, axis: int, reverse: bool) -> Array:
    return jnp.where(
        passable, lax.cummin(field - offset, axis, reverse=reverse) + offset, _FAR
    )


def _sweep_to_fixpoint(seed: Array, passable: Array) -> Array:
    longest = max(passable.shape[1:])
    if _FAR * longest + longest - 1 > jnp.iinfo(jnp.int32).max:
        raise ValueError(
            f"a {passable.shape[1]}x{passable.shape[2]} field overflows "
            "obiBotKenobi's distance sweeps"
        )
    offsets = {
        (axis, reverse): _sweep_offsets(passable, axis, reverse)
        for axis in (1, 2)
        for reverse in (False, True)
    }

    def along(field: Array, axis: int) -> Array:
        return jnp.minimum(
            _sweep(field, passable, offsets[axis, False], axis, False),
            _sweep(field, passable, offsets[axis, True], axis, True),
        )

    field, _ = lax.while_loop(
        lambda carry: jnp.any(carry[0] != carry[1]),
        lambda carry: (along(along(carry[0], 2), 1), carry[0]),
        (seed, seed + 1),
    )
    return field


def _distances_from(targets: Array, occupied: Array, game: GameConfig) -> Array:
    bucket = game.bucket
    shape = (targets.shape[0], bucket.height, bucket.width)
    wall = game.terrain.reshape(bucket.cell_count)
    passable = (~wall & (~occupied | targets)).reshape(shape)
    initial = jnp.where(targets, 0, _FAR).reshape(shape)

    def relax(field: Array) -> Array:
        return jnp.where(
            passable, jnp.minimum(field, _neighbour_minimum(field) + 1), _FAR
        )

    field = _sweep_to_fixpoint(relax(initial), passable)
    nearest = _neighbour_minimum(field).reshape(targets.shape)
    return jnp.where(nearest < _FAR, nearest + 1, bucket.cell_count + 1)


def _distance_fields(state: State, game: GameConfig, player: Array | int) -> Array:
    bucket = game.bucket
    active = state.active
    owner = state.owner.astype(jnp.int32)
    unit_type = state.unit_type.astype(jnp.int32)
    cell = state.cell.astype(jnp.int32)
    targets = jnp.stack(
        [
            _board(
                active & (owner == 1 - jnp.asarray(player, jnp.int32)), cell, bucket
            ),
            _board(
                active & (unit_type == TYPE_RESOURCE) & (owner == OWNER_NEUTRAL),
                cell,
                bucket,
            ),
            _board(active & (unit_type == TYPE_BASE) & (owner == player), cell, bucket),
        ]
    )
    occupied = state.entity_at.reshape(bucket.cell_count) != ABSENT
    return _distances_from(targets, occupied, game)


def _units_around(state: State, radius: int, bucket: BucketSpec) -> Array:
    side = 2 * radius + 1
    return lax.reduce_window(
        (state.entity_at != ABSENT).astype(jnp.int32),
        0,
        lax.add,
        (side, side),
        (1, 1),
        "SAME",
    ).reshape(bucket.cell_count)


def _marked(grid: Legality) -> Array:
    produce = grid.produce_type[..., :, None] & grid.produce_direction[..., None, :]
    return jnp.concatenate(
        [
            grid.attack_offset,
            grid.harvest_direction,
            grid.return_direction,
            produce.reshape(*produce.shape[:-2], -1),
            grid.move_direction,
            grid.kind[..., KIND_NONE : KIND_NONE + 1],
        ],
        axis=-1,
    )


def _worker_scores(kind: Array, carrying: Array, distance: _Distances, build_site: Array, no_barracks: Array, worst: int) -> tuple[Array, Array, Array]:
    idle = kind == KIND_NONE
    attack = kind == KIND_ATTACK
    produce = kind == KIND_PRODUCE
    soldier = jnp.where(
        idle | produce,
        worst,
        jnp.where(attack, _SOLDIER_ATTACK_SCORE, -distance.enemy),
    )
    gather = jnp.where(
        carrying,
        jnp.where(kind == KIND_RETURN, _ARRIVAL_SCORE, -distance.own_base),
        jnp.where(kind == KIND_HARVEST, _ARRIVAL_SCORE, -distance.resource),
    )
    harvester = jnp.where(
        attack, _WORKER_ATTACK_SCORE, jnp.where(idle | produce, worst, gather)
    )
    builder = jnp.where(
        attack,
        _WORKER_ATTACK_SCORE,
        jnp.where(idle, worst, jnp.where(no_barracks & produce, build_site, gather)),
    )
    return soldier, harvester, builder


def _barracks_scores(kind: Array, produced: Array, distance: _Distances, bucket: BucketSpec, worst: int) -> Array:
    far_for_ranged = (distance.enemy > bucket.width + bucket.height // 4) & (
        produced == TYPE_RANGED
    )
    return jnp.where(
        (kind == KIND_NONE) | (produced == TYPE_HEAVY),
        worst,
        -distance.enemy + far_for_ranged.astype(jnp.int32),
    )


def _base_scores(kind: Array, distance: _Distances, base_site: Array, worst: int) -> tuple[Array, Array, Array]:
    idle = kind == KIND_NONE
    return (
        jnp.where(idle, worst, -distance.enemy),
        jnp.where(idle, worst, -distance.resource),
        jnp.where(idle, worst, base_site),
    )


def _fields(state: State, game: GameConfig, player: Array | int) -> _Fields:
    bucket = game.bucket
    occupant = state.entity_at.reshape(bucket.cell_count).astype(jnp.int32)
    return _Fields(
        cell_facts=jnp.stack(
            [
                *_distance_fields(state, game, player),
                _BUILD_SITE_SCORE - _units_around(state, _BUILD_SITE_RADIUS, bucket),
                _units_around(state, _BASE_SITE_RADIUS, bucket),
                state.birth[jnp.clip(occupant, 0, bucket.entity_capacity - 1)],
            ],
            axis=-1,
        ),
        no_barracks=~jnp.any(state.active & (state.unit_type == TYPE_BARRACKS)),
    )


def _target_cells(cell: Array, game: GameConfig) -> tuple[Array, Array, Array]:
    bucket = game.bucket
    candidates = _candidates(bucket)
    kind = _column(candidates.kind)
    direction = _column(candidates.direction)
    neighbour, on_board = _step(
        cell[..., None], jnp.arange(DIRECTION_COUNT, dtype=jnp.int32), bucket
    )
    attacked = window_targets(cell, game)[
        ..., jnp.clip(_column(candidates.offset), 0, None)
    ]
    target_cell = jnp.where(
        kind == KIND_ATTACK,
        attacked,
        jnp.where(
            direction == ABSENT, ABSENT, neighbour[..., jnp.clip(direction, 0, None)]
        ),
    )
    return target_cell, neighbour, on_board


def _visit_scores(unit: _VisitedUnit, facts: Array, no_barracks: Array, game: GameConfig) -> Array:
    bucket = game.bucket
    candidates = _candidates(bucket)
    kind = _column(candidates.kind)
    distance = _Distances(*(facts[..., column] for column in _DISTANCE_COLUMNS))
    worst = -bucket.cell_count
    worker = _worker_scores(
        kind,
        unit.carrying,
        distance,
        facts[..., _BUILD_SITE_COLUMN],
        no_barracks,
        worst,
    )
    barracks = _barracks_scores(
        kind, _column(candidates.produced_type), distance, bucket, worst
    )
    base = _base_scores(kind, distance, facts[..., _BASE_SITE_COLUMN], worst)
    known_type = jnp.clip(unit.unit_type, 0, bucket.unit_type_capacity - 1)
    fights = (game.unit_can_attack & ~game.unit_can_harvest)[known_type]
    return jnp.where(
        unit.unit_type == TYPE_BASE,
        jnp.stack(base)[unit.scored_as],
        jnp.where(
            fights,
            worker[_SOLDIER],
            jnp.where(
                unit.unit_type == TYPE_BARRACKS,
                barracks,
                jnp.where(
                    unit.unit_type == TYPE_WORKER,
                    jnp.stack(worker)[unit.scored_as],
                    worst,
                ),
            ),
        ),
    ).astype(jnp.int32)


def _visit_list_positions(unit: _VisitedUnit, facts: Array, game: GameConfig) -> Array:
    bucket = game.bucket
    candidates = _candidates(bucket)
    kind = _column(candidates.kind)
    offset = jnp.clip(_column(candidates.offset), 0, None)
    attack_range = game.unit_attack_range[
        jnp.clip(unit.unit_type, 0, bucket.unit_type_capacity - 1)
    ]
    return jnp.where(
        kind == KIND_ATTACK,
        jnp.where(
            attack_range == 1,
            _column(_adjacent_attack_order(bucket))[offset],
            facts[..., _TARGET_BIRTH_COLUMN],
        ),
        _column(candidates.list_position),
    )


def _pending_actions(state: State, game: GameConfig, player: Array | int, cell: Array) -> _Pending:
    bucket = game.bucket

    def own(field: Array) -> Array:
        return _owned(field, bucket)[player]

    kind = own(state.pending_kind).astype(jnp.int32)
    busy = own(state.active) & (kind != KIND_ABSENT)
    parameter = jnp.where(
        kind == KIND_NONE,
        own(state.pending_none_duration),
        jnp.where(
            _directional(kind), own(state.pending_direction).astype(jnp.int32), ABSENT
        ),
    )
    claimed_cell, on_board = _step(cell, parameter, bucket)
    produces = busy & (kind == KIND_PRODUCE)
    produced_type = jnp.clip(
        own(state.pending_produced_type).astype(jnp.int32),
        0,
        bucket.unit_type_capacity - 1,
    )
    return _Pending(
        birth=own(state.birth),
        claims=busy & (parameter >= 0) & (parameter < DIRECTION_COUNT) & on_board,
        claimed_cell=claimed_cell,
        produces=produces,
        cost=jnp.where(produces, game.unit_cost[produced_type], 0),
        is_base=own(state.unit_type) == TYPE_BASE,
    )


def _legality(state: State, game: GameConfig) -> Legality:
    return core.legal_commands(state, game, stored="words")


def _read(state: State, game: GameConfig, player: Array | int) -> _Reading:
    bucket = game.bucket
    grid = jax.tree.map(lambda factor: factor[player], _legality(state, game))

    def own(field: Array) -> Array:
        return _owned(field, bucket)[player]

    cell = own(state.cell).astype(jnp.int32)
    return _Reading(
        units=_Unit(
            available=grid.available,
            birth=own(state.birth),
            cell=cell,
            unit_type=own(state.unit_type).astype(jnp.int32),
            carrying=own(state.carried_resources) > 0,
            marked=_marked(grid) & grid.available[:, None],
        ),
        fields=_fields(state, game, player),
        pending=_pending_actions(state, game, player, cell),
        balance=state.player_resources[player],
    )


def _unit_order(units: _Unit) -> Array:
    return jnp.argsort(jnp.where(units.available, units.birth, _LAST))


def _first(events: Array, birth: Array) -> Array:
    return jnp.min(jnp.where(events, birth, _LAST), axis=-1)


def _accepted(unit: _VisitedUnit, neighbour: Array, on_board: Array, claimed: Array, rewritten: Array, pending: _Pending, balance: Array, game: GameConfig) -> tuple[Array, Array]:
    candidates = _candidates(game.bucket)
    kind = _column(candidates.kind)
    direction = jnp.clip(_column(candidates.direction), 0, DIRECTION_COUNT - 1)
    produced = jnp.clip(_column(candidates.produced_type), 0, None)

    in_play = ~rewritten
    planned = on_board & jnp.any(claimed[None, :] == neighbour[:, None], axis=-1)
    position = (
        (pending.claims & in_play)[None, :]
        & (pending.claimed_cell[None, :] == neighbour[:, None])
        & on_board[:, None]
    )
    producing = pending.produces & in_play
    rewrite_pair = producing & pending.is_base & (unit.unit_type == TYPE_BARRACKS)
    unaffordable = (balance - pending.cost[None, :] < game.unit_cost[:, None]) & (
        producing & ~rewrite_pair
    )[None, :]
    first_reject = jnp.minimum(
        _first(position, pending.birth)[None, :],
        _first(unaffordable, pending.birth)[:, None],
    )
    rewrites = ~position & rewrite_pair[None, :]
    first_rewrite = _first(rewrites, pending.birth)
    rewrite_slot = jnp.argmin(
        jnp.where(rewrites, pending.birth, _LAST), axis=-1
    ).astype(jnp.int32)

    produce_rejected = planned[None, :] | (first_reject < first_rewrite[None, :])
    produce_rewrites = jnp.where(
        ~planned[None, :] & (first_rewrite[None, :] < first_reject),
        rewrite_slot[None, :],
        ABSENT,
    )
    moved_rejected = planned | jnp.any(position, axis=-1)
    rejected = jnp.where(
        kind == KIND_PRODUCE,
        produce_rejected[produced, direction],
        moved_rejected[direction],
    )
    rewrite = jnp.where(
        kind == KIND_PRODUCE, produce_rewrites[produced, direction], ABSENT
    )
    return ~(_flags(candidates.directional) & rejected), rewrite


def _top(visit: _Visit) -> Array:
    eligible = visit.marked & visit.accepted
    return jnp.max(jnp.where(eligible, visit.scores, -_LAST))


def _drawn(key: Array, visit: _Visit, remaining: Array, attempt: Array) -> tuple[Array, Array]:
    tied = visit.marked & (visit.scores == _top(visit))
    remaining = jnp.where(attempt == 0, tied, remaining)
    position = jnp.arange(remaining.shape[-1], dtype=jnp.int32)
    first = jnp.min(jnp.where(remaining, visit.list_position, _LAST))
    heads = jax.random.bernoulli(
        jax.random.fold_in(key, attempt), 0.5, remaining.shape
    ) | (visit.list_position == first)
    latest = jnp.max(jnp.where(remaining & heads, visit.list_position, -1))
    drawn = jnp.argmax(remaining & (visit.list_position == latest)).astype(jnp.int32)
    chosen = jnp.where(jnp.any(remaining), drawn, ABSENT)
    return chosen, remaining & (position != chosen)


def _given_jobs(jobs: Jobs, units: _Unit) -> tuple[Jobs, Array]:
    is_worker = units.unit_type == TYPE_WORKER
    gives = units.available & is_worker & (jobs.holder != units.birth)
    givers_before = jnp.sum(
        gives[None, :] & (units.birth[None, :] < units.birth[:, None]),
        axis=1,
        dtype=jnp.int32,
    )
    harvesters_left = (~jobs.harvester_given).astype(jnp.int32)
    builders_left = harvesters_left + (~jobs.builder_given).astype(jnp.int32)
    next_job = jnp.where(
        givers_before < harvesters_left,
        _HARVESTER,
        jnp.where(givers_before < builders_left, _BUILDER, _SOLDIER),
    )
    job = jnp.where(gives, next_job, jobs.job.astype(jnp.int32))
    return (
        Jobs(
            job=jnp.where(gives, job, jobs.job).astype(jnp.int8),
            holder=jnp.where(gives, units.birth, jobs.holder),
            harvester_given=jobs.harvester_given | jnp.any(gives & (job == _HARVESTER)),
            builder_given=jobs.builder_given | jnp.any(gives & (job == _BUILDER)),
        ),
        jnp.clip(jnp.where(is_worker, job, next_job), 0, _JOB_COUNT - 1),
    )


def _visit_count(units: _Unit) -> Array:
    return jnp.sum(units.available, dtype=jnp.int32)


def _visit_rows(units: _Unit, scored_as: Array) -> Array:
    return jnp.stack(
        [
            jnp.arange(units.available.shape[-1], dtype=jnp.int32),
            *(
                field.astype(jnp.int32)
                for field in (
                    units.available,
                    units.cell,
                    units.unit_type,
                    units.carrying,
                    scored_as,
                )
            ),
        ],
        axis=-1,
    )


def _visited_unit(row: Array, marked: Array) -> _VisitedUnit:
    return _VisitedUnit(
        slot=row[0],
        available=row[1] != 0,
        cell=row[2],
        unit_type=row[3],
        carrying=row[4] != 0,
        scored_as=row[5],
        marked=marked,
    )


def _judged(unit: _VisitedUnit, walking: _Walking, reading: _Reading, game: GameConfig) -> _Judged:
    target_cell, neighbour, on_board = _target_cells(unit.cell, game)
    facts = reading.fields.cell_facts[
        jnp.clip(target_cell, 0, game.bucket.cell_count - 1)
    ]
    accepted, rewrite = _accepted(
        unit,
        neighbour,
        on_board,
        walking.claimed,
        walking.rewritten,
        reading.pending,
        reading.balance,
        game,
    )
    return _Judged(
        _Visit(
            marked=unit.marked,
            scores=_visit_scores(unit, facts, reading.fields.no_barracks, game),
            accepted=accepted,
            list_position=_visit_list_positions(unit, facts, game),
        ),
        rewrite,
        neighbour,
    )


def _settled(walking: _Walking, unit: _VisitedUnit, judged: _Judged, chosen: Array, remaining: Array, game: GameConfig) -> _Walking:
    candidates = _candidates(game.bucket)
    local = jnp.arange(game.bucket.per_player_capacity, dtype=jnp.int32)
    chosen = jnp.where(unit.available, chosen, ABSENT)
    known = jnp.clip(chosen, 0, None)
    refused = (chosen == ABSENT) | ~judged.visit.accepted[known]
    settled = ~(refused & jnp.any(remaining))
    here = settled & (local == unit.slot)
    claims = (chosen != ABSENT) & _flags(candidates.directional)[known]
    claimed_cell = judged.neighbour[
        jnp.clip(_column(candidates.direction)[known], 0, DIRECTION_COUNT - 1)
    ]
    return _Walking(
        position=walking.position + settled.astype(jnp.int32),
        attempt=jnp.where(settled, 0, walking.attempt + 1),
        remaining=remaining,
        chosen=jnp.where(here, chosen, walking.chosen),
        claimed=jnp.where(here & claims, claimed_cell, walking.claimed),
        rewritten=walking.rewritten
        | (settled & (chosen != ABSENT) & (local == judged.rewrite[known])),
    )


def _walk(reading: _Reading, jobs: Jobs, keys: Array, game: GameConfig) -> tuple[Jobs, _Walking]:
    bucket = game.bucket
    capacity = bucket.per_player_capacity
    candidates = _candidates(bucket)
    units = reading.units
    jobs, scored_as = _given_jobs(jobs, units)
    order = _unit_order(units)
    visits = _visit_count(units)
    in_order = jax.tree.map(
        lambda field: field[order],
        (_visit_rows(units, scored_as), units.marked, keys),
    )

    def visiting(walking: _Walking) -> Array:
        return walking.position < visits

    def visit(walking: _Walking) -> _Walking:
        row, marked, key = jax.tree.map(
            lambda field: lax.dynamic_index_in_dim(
                field, jnp.minimum(walking.position, capacity - 1), keepdims=False
            ),
            in_order,
        )
        unit = _visited_unit(row, marked)
        judged = _judged(unit, walking, reading, game)
        chosen, remaining = _drawn(
            key, judged.visit, walking.remaining, walking.attempt
        )
        return _settled(walking, unit, judged, chosen, remaining, game)

    walked = lax.while_loop(
        visiting,
        visit,
        _Walking(
            position=jnp.int32(0),
            attempt=jnp.int32(0),
            remaining=jnp.zeros(len(candidates.kind), jnp.bool_),
            chosen=jnp.full(capacity, ABSENT, jnp.int32),
            claimed=jnp.full(capacity, ABSENT, jnp.int32),
            rewritten=jnp.zeros(capacity, jnp.bool_),
        ),
    )
    return jobs, walked


def _commands(chosen: Array, cell: Array, player: Array | int, game: GameConfig) -> Commands:
    candidates = _candidates(game.bucket)
    issued = chosen != ABSENT
    known = jnp.clip(chosen, 0, None)
    kind = _column(candidates.kind)[known]
    attack = kind == KIND_ATTACK
    attacked = jnp.take_along_axis(
        window_targets(cell, game),
        jnp.clip(_column(candidates.offset)[known], 0, None)[:, None],
        axis=1,
    )[:, 0]
    fields = {
        "kind": jnp.where(issued, kind, KIND_ABSENT),
        "direction": jnp.where(issued, _column(candidates.direction)[known], ABSENT),
        "none_duration": jnp.where(issued & (kind == KIND_NONE), _NONE_DURATION, 0),
        "target_cell": jnp.where(issued & attack, attacked, ABSENT),
        "produced_type": jnp.where(
            issued, _column(candidates.produced_type)[known], ABSENT
        ),
        "request_order": jnp.full_like(chosen, ABSENT),
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


def _init(game: GameConfig) -> Jobs:
    capacity = game.bucket.per_player_capacity
    return Jobs(
        job=jnp.full(capacity, ABSENT, jnp.int8),
        holder=jnp.full(capacity, ABSENT, jnp.int32),
        harvester_given=jnp.bool_(False),
        builder_given=jnp.bool_(False),
    )


def _program(key: Array, jobs: Jobs, state: State, game: GameConfig, player: Array | int) -> tuple[Jobs, Commands]:
    reading = _read(state, game, player)
    keys = jax.random.split(key, game.bucket.per_player_capacity)
    jobs, walked = _walk(reading, jobs, keys, game)
    return jobs, _commands(walked.chosen, reading.units.cell, player, game)


def _fits(bucket: BucketSpec) -> bool:
    return whole_act.fits(
        bucket,
        bytes_per_cell=_BYTES_PER_CELL,
        lane_bytes=_LANE_BYTES
        + _BYTES_PER_CANDIDATE * len(_candidates(bucket).kind)
        + _BYTES_PER_TYPE * bucket.unit_type_capacity,
        bytes_per_slot=_BYTES_PER_SLOT,
    )


def _kernel(key: Array, jobs: Jobs, state: State, game: GameConfig, player: Array | int) -> tuple[Jobs, Commands]:
    return whole_act.act(
        _TARGET,
        jobs,
        state,
        game,
        player,
        attributes={},
        operands=_operands(key, state, game),
    )


def _operands(key: Array, state: State, game: GameConfig) -> list[Array]:
    grid = _legality(state, game)
    return [
        state.pending_none_duration,
        *(getattr(grid, field.name) for field in dataclasses.fields(Legality)),
        jax.random.key_data(key),
        jnp.asarray(attack_offsets(game.bucket)),
    ]


def _act(key: Array, jobs: Jobs, state: State, game: GameConfig, player: Array | int) -> tuple[Jobs, Commands]:
    if not (kernels.selected() and _fits(game.bucket) and whole_act.draws_valid(key)):
        return _program(key, jobs, state, game, player)
    return kernels.dispatch(
        key, jobs, state, game, player, kernel=_kernel, program=_program
    )


obiBotKenobi: Final[Bot[Jobs]] = Bot(name="obiBotKenobi", init=_init, act=_act)
