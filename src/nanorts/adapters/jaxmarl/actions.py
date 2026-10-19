from __future__ import annotations

from typing import Final

import jax.numpy as jnp
import numpy as np
from jax import Array

from nanorts import core
from nanorts.adapters import factors
from nanorts.adapters.jaxmarl.layout import JaxMARLLayout
from nanorts.model import (
    ABSENT,
    KIND_ABSENT,
    KIND_NONE,
    PLAYER_COUNT,
    Commands,
    GameConfig,
    ModelError,
    State,
)

NONE_DURATION: Final[int] = 1

AvailActions = dict[str, Array]


def legal_actions(state: State, game: GameConfig, *, layout: JaxMARLLayout) -> AvailActions:
    legality = core.legal_commands(state, game, stored="words")
    concatenated = factors.concatenated(legality)
    available = legality.available
    rows = concatenated & available[..., None]
    idle = (
        jnp.zeros((layout.mask_width,), jnp.bool_)
        .at[layout.mask_offsets[0] + KIND_NONE]
        .set(True)
    )
    rows = rows | (~available[..., None] & idle)
    if layout.mask_form == "packed":
        rows = factors.pack(rows, layout.action_widths)
    return {agent: rows[player] for player, agent in enumerate(layout.agents)}


def slot_cells(state: State, game: GameConfig, *, layout: JaxMARLLayout) -> dict[str, Array]:
    bucket = game.bucket
    owned = slice(bucket.neutral_capacity, bucket.entity_capacity)
    cells = (
        state.cell[owned]
        .astype(jnp.int32)
        .reshape(PLAYER_COUNT, bucket.per_player_capacity)
    )
    return {agent: cells[player] for player, agent in enumerate(layout.agents)}


def unpack_mask(mask: Array, *, layout: JaxMARLLayout) -> Array:
    if layout.mask_form == "packed":
        return factors.unpack(mask, layout.action_widths)
    return mask


def decode_actions(actions: dict[str, Array], state: State, game: GameConfig, *, layout: JaxMARLLayout) -> Commands:
    bucket = game.bucket
    rows = _stacked(actions, layout)

    available = core.legal_commands(state, game).available
    owned = slice(bucket.neutral_capacity, bucket.entity_capacity)
    cell = state.cell[owned].reshape(PLAYER_COUNT, bucket.per_player_capacity)

    kind = rows[..., 0]
    direction = factors.direction(kind, rows)
    produced_type = factors.produced_type(kind, rows)
    target_cell = factors.attack_target(kind, rows, cell, bucket)

    def issued(values: Array, empty: int) -> Array:
        return jnp.where(available, values, empty).astype(jnp.int32)

    return Commands(
        kind=issued(kind, KIND_ABSENT),
        direction=issued(direction, ABSENT),
        none_duration=issued(jnp.where(kind == KIND_NONE, NONE_DURATION, 0), 0),
        target_cell=issued(target_cell, ABSENT),
        produced_type=issued(produced_type, ABSENT),
        request_order=jnp.full(available.shape, ABSENT, jnp.int32),
    )


def check_actions(actions: dict[str, Array], *, layout: JaxMARLLayout) -> None:
    if set(actions) != set(layout.agents):
        raise ModelError(
            f"jaxmarl actions are keyed by {sorted(actions)}, expected "
            f"{list(layout.agents)}"
        )
    for agent in layout.agents:
        array = np.asarray(actions[agent])
        if array.shape != layout.actions_shape:
            raise ModelError(
                f"jaxmarl actions for {agent} have shape {array.shape}, expected "
                f"{layout.actions_shape} - one row per player-local slot, not per "
                "cell and not per environment"
            )
        if array.dtype != np.int32:
            raise ModelError(
                f"jaxmarl actions for {agent} have dtype {array.dtype}, expected int32"
            )
        factors.refuse_out_of_range(
            array,
            layout.action_factors,
            layout.action_widths,
            subject=f"jaxmarl action factor {{index}} ({{name}}) of {agent}",
            exempt="one this kind does not read and one at a slot holding no unit",
        )


def _stacked(actions: dict[str, Array], layout: JaxMARLLayout) -> Array:
    return jnp.stack([actions[agent] for agent in layout.agents])
