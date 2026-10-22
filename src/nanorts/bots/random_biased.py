from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
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
    PLAYER_COUNT,
    BucketSpec,
    Commands,
    GameConfig,
    Legality,
    State,
    attack_offsets,
    empty,
    window_targets,
)

_BIASED_WEIGHT: Final[float] = 5.0
_REGULAR_WEIGHT: Final[float] = 1.0
_NONE_DURATION: Final[int] = 10
_LAST_IN_UNIT_ORDER: Final[int] = jnp.iinfo(jnp.int32).max

_TARGET: Final[str] = "nanorts_random_act"

_EVERY_UNIT: Final[int] = 0
_ONE_UNIT: Final[int] = 1

_BYTES_PER_CELL: Final[int] = 1

_LANE_BYTES: Final[int] = 32

_BYTES_PER_SLOT: Final[int] = 13


class _Candidates(NamedTuple):
    kind: tuple[int, ...]
    direction: tuple[int, ...]
    produced_type: tuple[int, ...]
    offset: tuple[int, ...]
    weight: tuple[float, ...]


class _Reading(NamedTuple):
    available: Array
    birth: Array
    cell: Array
    marked: Array
    blocked_cells: Array
    reserved: Array
    balance: Array


class _Choice(NamedTuple):
    kind: Array
    direction: Array
    target_cell: Array
    produced_type: Array
    claimed_cell: Array
    cost: Array


@functools.cache
def _candidates(bucket: BucketSpec) -> _Candidates:
    directions = range(DIRECTION_COUNT)
    return _Candidates(
        *zip(
            *(
                (KIND_ATTACK, ABSENT, ABSENT, offset, _BIASED_WEIGHT)
                for offset in range(bucket.attack_offset_capacity)
            ),
            *((KIND_HARVEST, d, ABSENT, ABSENT, _BIASED_WEIGHT) for d in directions),
            *((KIND_RETURN, d, ABSENT, ABSENT, _BIASED_WEIGHT) for d in directions),
            *(
                (KIND_PRODUCE, d, produced, ABSENT, _REGULAR_WEIGHT)
                for d in directions
                for produced in range(bucket.unit_type_capacity)
            ),
            *((KIND_MOVE, d, ABSENT, ABSENT, _REGULAR_WEIGHT) for d in directions),
            (KIND_NONE, ABSENT, ABSENT, ABSENT, _REGULAR_WEIGHT),
            strict=True,
        )
    )


def _marked(grid: Legality) -> Array:
    produce = grid.produce_direction[..., :, None] & grid.produce_type[..., None, :]
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


def _owned(field: Array, bucket: BucketSpec) -> Array:
    return field[bucket.neutral_capacity :].reshape(
        PLAYER_COUNT, bucket.per_player_capacity
    )


def _neighbour(cell: Array, direction: Array, width: int) -> Array:
    steps = jnp.asarray(DIRECTION_OFFSETS, jnp.int32)
    known = jnp.clip(direction, 0, DIRECTION_COUNT - 1)
    return cell + steps[known, 1] * width + steps[known, 0]


def _claims_position(kind: Array) -> Array:
    return (kind == KIND_MOVE) | (kind == KIND_PRODUCE)


def _cost(game: GameConfig, kind: Array, produced_type: Array) -> Array:
    known = jnp.clip(produced_type, 0, game.bucket.unit_type_capacity - 1)
    return jnp.where(kind == KIND_PRODUCE, game.unit_cost[known], 0).astype(jnp.int32)


def _legality(state: State, game: GameConfig) -> Legality:
    return core.legal_commands(state, game, stored="words")


def _read(state: State, game: GameConfig, player: Array | int) -> _Reading:
    bucket = game.bucket
    grid = jax.tree.map(lambda factor: factor[player], _legality(state, game))
    pending_kind = state.pending_kind.astype(jnp.int32)
    claims = state.active & _claims_position(pending_kind)
    claimed = _neighbour(
        state.cell, state.pending_direction.astype(jnp.int32), bucket.width
    )
    blocked_cells = (
        jnp.zeros(bucket.cell_count, jnp.bool_)
        .at[jnp.where(claims, claimed, bucket.cell_count)]
        .set(True, mode="drop")
    )
    pending_cost = jnp.where(
        state.active,
        _cost(game, pending_kind, state.pending_produced_type.astype(jnp.int32)),
        0,
    )
    return _Reading(
        available=grid.available,
        birth=_owned(state.birth, bucket)[player],
        cell=_owned(state.cell, bucket)[player].astype(jnp.int32),
        marked=_marked(grid),
        blocked_cells=blocked_cells,
        reserved=jnp.sum(_owned(pending_cost, bucket), axis=-1, dtype=jnp.int32),
        balance=state.player_resources,
    )


def _log_weight(game: GameConfig) -> Array:
    return jnp.log(jnp.asarray(_candidates(game.bucket).weight, jnp.float32))


def _draw(key: Array, marked: Array, game: GameConfig) -> Array:
    return jax.random.categorical(key, jnp.where(marked, _log_weight(game), -jnp.inf))


def _choice(index: Array, cell: Array, game: GameConfig) -> _Choice:
    candidates = _candidates(game.bucket)
    kind, direction, produced_type, offset = (
        jnp.asarray(column, jnp.int32)[index]
        for column in (
            candidates.kind,
            candidates.direction,
            candidates.produced_type,
            candidates.offset,
        )
    )
    targets = window_targets(cell, game)
    target_cell = jnp.take_along_axis(
        targets, jnp.clip(offset, 0, None)[..., None], axis=-1
    )[..., 0]
    return _Choice(
        kind=kind,
        direction=direction,
        target_cell=jnp.where(kind == KIND_ATTACK, target_cell, ABSENT),
        produced_type=produced_type,
        claimed_cell=jnp.where(
            _claims_position(kind),
            _neighbour(cell, direction, game.bucket.width),
            ABSENT,
        ),
        cost=_cost(game, kind, produced_type),
    )


def _unit_order(listed: Array, birth: Array) -> Array:
    return jnp.argsort(jnp.where(listed, birth, _LAST_IN_UNIT_ORDER))


def _affordable(cost: Array, reserved: Array, balance: Array) -> Array:
    return (reserved == 0) | (cost + reserved <= balance)


def _other_player_affordable(reading: _Reading, player: Array | int) -> Array:
    other = 1 - jnp.asarray(player)
    return _affordable(jnp.int32(0), reading.reserved[other], reading.balance[other])


def _blocked_by_pending(reading: _Reading, choice: _Choice) -> Array:
    cell_count = reading.blocked_cells.shape[0]
    claimed = jnp.clip(choice.claimed_cell, 0, cell_count - 1)
    return (choice.claimed_cell != ABSENT) & reading.blocked_cells[claimed]


def _kept_by_birth(reading: _Reading, choice: _Choice, player: Array | int) -> Array:
    capacity = reading.available.shape[-1]
    balance = reading.balance[player]
    reserved = reading.reserved[player]
    claims = choice.claimed_cell != ABSENT
    claims_before = (
        claims[:, None]
        & claims[None, :]
        & (choice.claimed_cell[:, None] == choice.claimed_cell[None, :])
        & (reading.birth[:, None] < reading.birth[None, :])
    )
    eligible = (
        reading.available
        & ~_blocked_by_pending(reading, choice)
        & _other_player_affordable(reading, player)
    )
    costing = eligible & (choice.cost > 0)
    free = eligible & (choice.cost <= 0) & _affordable(jnp.int32(0), reserved, balance)

    def keepable(walk: tuple[Array, Array, Array]) -> Array:
        kept, taken, reserve = walk
        return costing & ~kept & ~taken & _affordable(choice.cost, reserve, balance)

    def keep(walk: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        kept, taken, reserve = walk
        first = jnp.argmin(
            jnp.where(keepable(walk), reading.birth, _LAST_IN_UNIT_ORDER)
        )
        chosen = jnp.arange(capacity) == first
        return (
            kept | chosen,
            taken | lax.dynamic_index_in_dim(claims_before, first, keepdims=False),
            reserve + jnp.sum(jnp.where(chosen, choice.cost, 0)),
        )

    kept, taken, _ = lax.while_loop(
        lambda walk: jnp.any(keepable(walk)),
        keep,
        (
            jnp.zeros(capacity, jnp.bool_),
            jnp.any(claims_before & free[:, None], axis=0),
            reserved,
        ),
    )
    return kept | (free & ~taken)


def _commands(issued: Array, kept: Array, choice: _Choice, request_order: Array, player: Array | int, game: GameConfig) -> Commands:
    unissued = empty(Commands, game.bucket)
    own_rows = (jnp.arange(PLAYER_COUNT) == player)[:, None]
    kind = jnp.where(kept, choice.kind, KIND_NONE)
    fields = {
        "kind": jnp.where(issued, kind, KIND_ABSENT),
        "direction": jnp.where(issued & kept, choice.direction, ABSENT),
        "none_duration": jnp.where(issued & (kind == KIND_NONE), _NONE_DURATION, 0),
        "target_cell": jnp.where(issued & kept, choice.target_cell, ABSENT),
        "produced_type": jnp.where(issued & kept, choice.produced_type, ABSENT),
        "request_order": jnp.where(issued, request_order, ABSENT),
    }
    return dataclasses.replace(
        unissued,
        **{
            name: jnp.where(own_rows, value.astype(jnp.int32), getattr(unissued, name))
            for name, value in fields.items()
        },
    )


def _init(game: GameConfig) -> tuple[()]:
    del game
    return ()


def _act_on_every_unit(key: Array, carry: tuple[()], state: State, game: GameConfig, player: Array | int) -> tuple[tuple[()], Commands]:
    reading = _read(state, game, player)
    choice = _choice(_draw(key, reading.marked, game), reading.cell, game)
    kept = _kept_by_birth(reading, choice, player)
    return carry, _commands(
        reading.available,
        kept,
        choice,
        jnp.full_like(reading.birth, ABSENT),
        player,
        game,
    )


def _act_on_one_unit(key: Array, carry: tuple[()], state: State, game: GameConfig, player: Array | int) -> tuple[tuple[()], Commands]:
    unit_key, action_key = jax.random.split(key)
    reading = _read(state, game, player)
    capacity = game.bucket.per_player_capacity
    chosen_local = jax.random.categorical(
        unit_key, jnp.where(reading.available, 0.0, -jnp.inf)
    )
    chosen = reading.available & (jnp.arange(capacity) == chosen_local)
    index = _draw(action_key, reading.marked[chosen_local], game)
    drawn = _choice(index, reading.cell[chosen_local], game)
    consistent = (
        ~_blocked_by_pending(reading, drawn)
        & _affordable(drawn.cost, reading.reserved[player], reading.balance[player])
        & _other_player_affordable(reading, player)
    )
    choice = _Choice(*(jnp.broadcast_to(field, (capacity,)) for field in drawn))
    fillers = reading.available & ~chosen
    order = _unit_order(fillers, reading.birth)
    filler_rank = (
        jnp.zeros(capacity, jnp.int32)
        .at[order]
        .set(jnp.arange(capacity, dtype=jnp.int32))
    )
    return carry, _commands(
        reading.available,
        chosen & consistent,
        choice,
        jnp.where(chosen, 0, filler_rank + 1),
        player,
        game,
    )


_Act = Callable[
    [Array, tuple[()], State, GameConfig, Array | int], tuple[tuple[()], Commands]
]


def _fits(bucket: BucketSpec) -> bool:
    return whole_act.fits(
        bucket,
        bytes_per_cell=_BYTES_PER_CELL,
        lane_bytes=_LANE_BYTES,
        bytes_per_slot=_BYTES_PER_SLOT,
    )


def _whole_act(units: int, program: _Act) -> _Act:
    kernel = _kernel(units)

    def act(key: Array, carry: tuple[()], state: State, game: GameConfig, player: Array | int) -> tuple[tuple[()], Commands]:
        if not (
            kernels.selected()
            and _fits(game.bucket)
            and whole_act.draws_valid(key)
            and whole_act.float_draws_served()
        ):
            return program(key, carry, state, game, player)
        return kernels.dispatch(
            key, carry, state, game, player, kernel=kernel, program=program
        )

    return act


@functools.cache
def _kernel(units: int) -> _Act:
    def kernel(key: Array, carry: tuple[()], state: State, game: GameConfig, player: Array | int) -> tuple[tuple[()], Commands]:
        return whole_act.act(
            _TARGET,
            carry,
            state,
            game,
            player,
            attributes={"units": units},
            operands=_operands(key, state, game),
        )

    return kernel


def _operands(key: Array, state: State, game: GameConfig) -> list[Array]:
    return [
        whole_act.legality_words(_legality(state, game)),
        jax.random.key_data(key),
        _log_weight(game),
        jnp.asarray(attack_offsets(game.bucket)),
    ]


randomBiasedAI: Final[Bot[tuple[()]]] = Bot(
    name="randomBiasedAI",
    init=_init,
    act=_whole_act(_EVERY_UNIT, _act_on_every_unit),
)

randomAI: Final[Bot[tuple[()]]] = Bot(
    name="randomAI", init=_init, act=_whole_act(_ONE_UNIT, _act_on_one_unit)
)
