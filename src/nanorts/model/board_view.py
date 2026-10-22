from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from jax import Array

from nanorts.model.config import GameConfig
from nanorts.model.leaf import FIELD, SLOT_INDEX, LeafField, leaf, model_fields
from nanorts.model.state import State
from nanorts.model.vocabulary import ABSENT, KIND_ABSENT, OWNER_NEUTRAL


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BoardView:
    present: Array = leaf(FIELD, jnp.bool_, fill=False)

    slot: Array = leaf(FIELD, SLOT_INDEX, fill=0)

    owner: Array = leaf(FIELD, jnp.int8, fill=OWNER_NEUTRAL)

    unit_type: Array = leaf(FIELD, jnp.int8, fill=ABSENT)

    hit_points: Array = leaf(FIELD, jnp.int16, fill=0)

    carried_resources: Array = leaf(FIELD, jnp.int32, fill=0)

    pending_kind: Array = leaf(FIELD, jnp.int8, fill=KIND_ABSENT)

    available: Array = leaf(FIELD, jnp.bool_, fill=False)


_FACTS_READ_FROM_SLOTS = (
    "owner",
    "unit_type",
    "hit_points",
    "carried_resources",
    "pending_kind",
)


def board_view(state: State, game: GameConfig) -> BoardView:
    occupant = state.entity_at
    present = occupant != ABSENT
    slot = jnp.clip(occupant, 0, game.bucket.entity_capacity - 1)
    specs = {
        field.name: field.spec
        for field in model_fields(BoardView)
        if isinstance(field, LeafField)
    }
    dtypes = {
        name: specs[name].dtype_under(game.bucket) for name in _FACTS_READ_FROM_SLOTS
    }
    read = {
        name: jnp.where(
            present,
            getattr(state, name).astype(dtypes[name])[slot],
            jnp.asarray(specs[name].fill, dtypes[name]),
        )
        for name in _FACTS_READ_FROM_SLOTS
    }
    return BoardView(
        present=present,
        slot=slot,
        available=(read["owner"] >= 0) & (read["pending_kind"] == KIND_ABSENT),
        **read,
    )
