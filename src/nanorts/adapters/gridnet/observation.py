from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from jax import Array

from nanorts import core, observations
from nanorts.adapters import factors
from nanorts.adapters.gridnet.layout import GridNetLayout
from nanorts.model import PLAYER_COUNT, GameConfig, Legality, State, board_view


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GridNetObservation:
    planes: Array

    source_mask: Array

    action_mask: Array


def observe(state: State, game: GameConfig, *, layout: GridNetLayout) -> GridNetObservation:
    legality = core.legal_commands(state, game)
    source_mask, action_mask = _masks(state, legality, game, layout)
    return GridNetObservation(
        planes=planes(state, game, layout=layout),
        source_mask=source_mask,
        action_mask=action_mask,
    )


def planes(state: State, game: GameConfig, *, layout: GridNetLayout) -> Array:
    widths = jnp.asarray(layout.plane_widths, jnp.int32)
    offsets = jnp.asarray(layout.plane_offsets, jnp.int32)
    feature_maps = jnp.stack(
        [
            observations.symbolic(
                game, state, observer=player, partial_observability=False
            ).feature_maps
            for player in range(PLAYER_COUNT)
        ]
    )
    clipped = jnp.clip(feature_maps, 0, widths - 1)
    if layout.planes_form == "indices":
        return clipped.astype(layout.planes_dtype)
    return jnp.sum(
        jax.nn.one_hot(clipped + offsets, layout.plane_count, dtype=jnp.int32), axis=-2
    ).astype(layout.planes_dtype)


def _masks(state: State, legality: Legality, game: GameConfig, layout: GridNetLayout) -> tuple[Array, Array]:
    bucket = game.bucket
    cells = bucket.cell_count
    slots = PLAYER_COUNT * bucket.per_player_capacity

    board = board_view(state, game)
    index = jnp.clip(board.slot.reshape(cells) - bucket.neutral_capacity, 0, slots - 1)
    player = jnp.arange(PLAYER_COUNT, dtype=jnp.int8)[:, None]

    rows = factors.concatenated(legality).reshape(slots, layout.mask_width)
    source = board.available.reshape(cells) & (board.owner.reshape(cells) == player)
    mask = source[..., None] & rows[index]
    return (
        source.reshape(layout.source_shape),
        (
            factors.pack(mask, layout.action_factor_sizes)
            if layout.mask_form == "packed"
            else mask.astype(layout.mask_dtype)
        ).reshape(layout.mask_shape),
    )


def unpack_mask(action_mask: Array, *, layout: GridNetLayout) -> Array:
    if layout.mask_form == "packed":
        return factors.unpack(action_mask, layout.action_factor_sizes)
    return action_mask != 0
