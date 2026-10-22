from __future__ import annotations

import dataclasses
import functools
import operator
from typing import Final

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike

from nanorts.model import (
    ABSENT,
    ACTION_KIND_COUNT,
    KIND_NONE,
    PLAYER_COUNT,
    GameConfig,
    Shape,
    State,
    board_view,
    leaf,
)
from nanorts.observations.visibility import observer_index, visibility, visible_board

GROUPS: Final[tuple[str, ...]] = (
    "hit_points",
    "carried_resources",
    "owner",
    "unit_type",
    "pending_action_kind",
    "terrain",
)

SCALARS: Final[tuple[str, ...]] = (
    "own_balance",
    "opponent_balance",
    "time",
    "alternating_conflict_count",
)

_FEATURE_MAPS: Final = Shape(
    "FEATURE_MAPS", lambda bucket: (bucket.height, bucket.width, len(GROUPS))
)

_SYMBOLIC_SCALARS: Final = Shape("SYMBOLIC_SCALARS", lambda bucket: (len(SCALARS),))

FEATURE_MAP_DTYPES: Final[tuple[jnp.dtype, ...]] = (
    jnp.dtype(jnp.int32),
    jnp.dtype(jnp.int16),
    jnp.dtype(jnp.uint8),
)

_OCCUPANT_WORD: Final[tuple[tuple[str, int, int], ...]] = (
    ("hit_points", 0, 8),
    ("carried_resources", 8, 8),
    ("unit_type", 16, 8),
    ("pending_action_kind", 24, 4),
    ("owner", 28, 2),
)

if ACTION_KIND_COUNT > 1 << 4 or PLAYER_COUNT >= 1 << 2:
    raise ImportError(
        "the uint8 encode packs every pending kind into 4 bits and every owner into 2"
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SymbolicObservation:
    feature_maps: Array = leaf(_FEATURE_MAPS, jnp.int32, fill=0)

    scalars: Array = leaf(_SYMBOLIC_SCALARS, jnp.int32, fill=0)


@dataclasses.dataclass(frozen=True)
class SymbolicLayout:
    feature_maps_shape: tuple[int, int, int]

    feature_maps_dtype: jnp.dtype

    groups: tuple[str, ...]

    bounds: tuple[int, ...]

    scalars_shape: tuple[int]

    scalars_dtype: jnp.dtype

    scalar_names: tuple[str, ...]

    partial_observability: bool

    @property
    def bound(self) -> dict[str, int]:
        return dict(zip(self.groups, self.bounds, strict=True))


def symbolic(game: GameConfig, state: State, *, observer: int, partial_observability: bool, dtype: DTypeLike = jnp.int32) -> SymbolicObservation:
    player = observer_index(observer)
    published = _feature_map_dtype(dtype)
    partial = partial_mode(partial_observability)
    if published == jnp.dtype(jnp.uint8):
        return SymbolicObservation(
            feature_maps=_byte_feature_maps(game, state, player, partial),
            scalars=scalar_vector(state, player),
        )
    board = (
        visible_board(game, state, observer=player)
        if partial
        else board_view(state, game)
    )
    owner = board.owner.astype(jnp.int32)
    groups = {
        "hit_points": board.hit_points,
        "carried_resources": board.carried_resources,
        "owner": jnp.where(owner >= 0, ((owner + player) % PLAYER_COUNT) + 1, 0),
        "unit_type": board.unit_type + 1,
        "pending_action_kind": jnp.maximum(board.pending_kind, KIND_NONE),
        "terrain": game.terrain.astype(jnp.int32),
    }
    return SymbolicObservation(
        feature_maps=jnp.stack([groups[name] for name in GROUPS], axis=-1).astype(
            published
        ),
        scalars=scalar_vector(state, player),
    )


def _byte_feature_maps(game: GameConfig, state: State, player: int, partial: bool) -> Array:
    word = _occupant_words(game, state)
    facts = {
        name: (word >> shift) & ((1 << width) - 1)
        for name, shift, width in _OCCUPANT_WORD
    }
    if partial:
        seen = visibility(game, state, observer=player).visible_cells
        shown = (facts["owner"] == player + 1) | seen
        facts = {name: jnp.where(shown, fact, 0) for name, fact in facts.items()}
    owner = facts["owner"].astype(jnp.int32)
    groups = {
        **facts,
        "owner": jnp.where(owner > 0, ((owner - 1 + player) % PLAYER_COUNT) + 1, 0),
        "terrain": game.terrain,
    }
    return jnp.stack([groups[name].astype(jnp.uint8) for name in GROUPS], axis=-1)


def _occupant_words(game: GameConfig, state: State) -> Array:
    facts = {
        "hit_points": state.hit_points,
        "carried_resources": state.carried_resources,
        "unit_type": state.unit_type.astype(jnp.int32) + 1,
        "pending_action_kind": jnp.maximum(
            state.pending_kind.astype(jnp.int32), KIND_NONE
        ),
        "owner": state.owner.astype(jnp.int32) + 1,
    }
    word = functools.reduce(
        operator.or_,
        [
            (facts[name].astype(jnp.uint32) & ((1 << width) - 1)) << shift
            for name, shift, width in _OCCUPANT_WORD
        ],
    )
    occupant = state.entity_at
    slot = jnp.clip(occupant, 0, game.bucket.entity_capacity - 1)
    return jnp.where(occupant != ABSENT, word[slot], 0)


def scalar_vector(state: State, player: int) -> Array:
    opponent = (player + 1) % PLAYER_COUNT
    return jnp.stack(
        [
            state.player_resources[player],
            state.player_resources[opponent],
            state.time,
            state.alternating_conflict_count,
        ]
    ).astype(jnp.int32)


def symbolic_layout(game: GameConfig, *, partial_observability: bool, dtype: DTypeLike = jnp.int32) -> SymbolicLayout:
    partial = partial_mode(partial_observability)
    published = _feature_map_dtype(dtype)
    bucket = game.bucket
    height, width, groups = _FEATURE_MAPS.resolve(bucket)
    (scalars,) = _SYMBOLIC_SCALARS.resolve(bucket)
    bounds = {
        "hit_points": max(
            int(jnp.max(game.unit_hit_points)),
            int(jnp.max(game.initial_state.hit_points)),
        ),
        "carried_resources": max(
            int(jnp.max(game.unit_harvest_amount)),
            int(jnp.max(game.initial_state.carried_resources)),
        ),
        "owner": PLAYER_COUNT,
        "unit_type": bucket.unit_type_capacity,
        "pending_action_kind": ACTION_KIND_COUNT - 1,
        "terrain": 1,
    }
    ceiling = int(jnp.iinfo(published).max)
    exceeded = {name: bound for name, bound in bounds.items() if bound > ceiling}
    if exceeded:
        raise ValueError(
            f"symbolic_layout refuses dtype={published.name}: its maximum "
            f"{ceiling} is below the bound this content gives "
            + ", ".join(f"{name} ({bound})" for name, bound in exceeded.items())
        )
    return SymbolicLayout(
        feature_maps_shape=(height, width, groups),
        feature_maps_dtype=published,
        groups=GROUPS,
        bounds=tuple(bounds[name] for name in GROUPS),
        scalars_shape=(scalars,),
        scalars_dtype=jnp.dtype(jnp.int32),
        scalar_names=SCALARS,
        partial_observability=partial,
    )


def _feature_map_dtype(dtype: DTypeLike) -> jnp.dtype:
    try:
        published = jnp.dtype(dtype)
    except TypeError:
        published = None
    if published is None or published not in FEATURE_MAP_DTYPES:
        raise ValueError(
            f"the Feature map dtype is {dtype!r}, expected one of "
            f"{[accepted.name for accepted in FEATURE_MAP_DTYPES]}"
        )
    return published


def partial_mode(partial_observability: object) -> bool:
    if partial_observability is not True and partial_observability is not False:
        raise ValueError(
            f"partial_observability is {partial_observability!r}, expected a bool"
        )
    return partial_observability
