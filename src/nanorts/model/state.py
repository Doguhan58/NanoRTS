from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from jax import Array

from nanorts.model.bucket import BucketSpec
from nanorts.model.errors import ModelError
from nanorts.model.leaf import (
    CELL_INDEX,
    FIELD,
    PER_PLAYER,
    PER_SLOT,
    SCALAR,
    SLOT_INDEX,
    LeafField,
    empty,
    leaf,
    model_fields,
    nested,
)
from nanorts.model.vocabulary import ABSENT, KIND_ABSENT, OWNER_NEUTRAL, PLAYER_COUNT


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Telemetry:
    spawn_overflow: Array = leaf(PER_PLAYER, jnp.int32, fill=0)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class State:
    birth: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)

    unit_type: Array = leaf(PER_SLOT, jnp.int8, fill=ABSENT)

    owner: Array = leaf(PER_SLOT, jnp.int8, fill=ABSENT)

    cell: Array = leaf(PER_SLOT, CELL_INDEX, fill=ABSENT)

    hit_points: Array = leaf(PER_SLOT, jnp.int16, fill=0)

    carried_resources: Array = leaf(PER_SLOT, jnp.int32, fill=0)

    pending_kind: Array = leaf(PER_SLOT, jnp.int8, fill=KIND_ABSENT)

    pending_direction: Array = leaf(PER_SLOT, jnp.int8, fill=ABSENT)

    pending_none_duration: Array = leaf(PER_SLOT, jnp.int32, fill=0)

    pending_target_cell: Array = leaf(PER_SLOT, CELL_INDEX, fill=ABSENT)

    pending_produced_type: Array = leaf(PER_SLOT, jnp.int8, fill=ABSENT)

    pending_issued_at: Array = leaf(PER_SLOT, jnp.int32, fill=0)

    pending_completes_at: Array = leaf(PER_SLOT, jnp.int32, fill=0)

    pending_order: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)

    entity_at: Array = leaf(FIELD, SLOT_INDEX, fill=ABSENT)

    time: Array = leaf(SCALAR, jnp.int32, fill=0)

    player_resources: Array = leaf(PER_PLAYER, jnp.int32, fill=0)

    next_birth: Array = leaf(SCALAR, jnp.int32, fill=0)

    next_pending_order: Array = leaf(SCALAR, jnp.int32, fill=0)

    alternating_conflict_count: Array = leaf(SCALAR, jnp.int32, fill=0)

    game_over: Array = leaf(SCALAR, jnp.bool_, fill=False)

    winner: Array = leaf(SCALAR, jnp.int32, fill=ABSENT)

    telemetry: Telemetry = nested()

    @property
    def active(self) -> Array:
        return self.birth != ABSENT


def state_invariant_violations(state: State, bucket: BucketSpec) -> dict[str, Array]:
    active = state.active
    in_bounds = (state.cell >= 0) & (state.cell < bucket.cell_count)
    violations: dict[str, Array] = {}
    violations[
        f"State.cell holds an active cell outside the {bucket.height}x"
        f"{bucket.width} field"
    ] = _breaks_where(active, in_bounds)
    flat_entity_at = state.entity_at.reshape(bucket.cell_count)
    violations["State.entity_at does not name every active slot at its cell"] = (
        _active_slots_missing_from_their_cells(
            state, bucket, flat_entity_at, active, in_bounds
        )
    )
    violations |= _occupancy_violations(state, bucket, flat_entity_at, active)
    violations["State.owner disagrees with the block its slot belongs to"] = (
        _breaks_where(active, state.owner == _block_owners(bucket))
    )
    violations |= _inactive_slot_violations(state, bucket, active)
    return violations


def _breaks_where(applies: Array, holds: Array) -> Array:
    return ~jnp.all(jnp.where(applies, holds, True))


def _active_slots_missing_from_their_cells(state: State, bucket: BucketSpec, flat_entity_at: Array, active: Array, in_bounds: Array) -> Array:
    slots = jnp.arange(bucket.entity_capacity, dtype=jnp.int32)
    indexed_back = flat_entity_at[jnp.where(active & in_bounds, state.cell, 0)] == slots
    return _breaks_where(active, indexed_back)


def _occupancy_violations(state: State, bucket: BucketSpec, flat_entity_at: Array, active: Array) -> dict[str, Array]:
    occupied = flat_entity_at != ABSENT
    names_a_slot = (flat_entity_at >= 0) & (flat_entity_at < bucket.entity_capacity)
    names_outside_capacity = _breaks_where(occupied, names_a_slot)
    occupant = jnp.where(occupied & names_a_slot, flat_entity_at, 0)
    occupant_is_at_cell = active[occupant] & (
        state.cell[occupant] == jnp.arange(bucket.cell_count)
    )
    return {
        "State.entity_at names a slot outside the entity capacity": (
            names_outside_capacity
        ),
        "State.entity_at names a slot that is not at that cell": _breaks_where(
            occupied, occupant_is_at_cell
        ),
    }


def _block_owners(bucket: BucketSpec) -> Array:
    return jnp.concatenate(
        [
            jnp.full((bucket.neutral_capacity,), OWNER_NEUTRAL, jnp.int32),
            *(
                jnp.full((bucket.per_player_capacity,), player, jnp.int32)
                for player in range(PLAYER_COUNT)
            ),
        ]
    )


def _inactive_slot_violations(state: State, bucket: BucketSpec, active: Array) -> dict[str, Array]:
    canonical = empty(State, bucket)
    violations: dict[str, Array] = {}
    for field in model_fields(State):
        if not isinstance(field, LeafField) or field.spec.shape is not PER_SLOT:
            continue
        held = getattr(state, field.name)
        violations[
            f"State.{field.name} is not its canonical empty on an inactive slot"
        ] = ~jnp.all(jnp.where(active, True, held == getattr(canonical, field.name)))
    return violations


def check_state_invariants(state: State, bucket: BucketSpec) -> None:
    for message, violated in state_invariant_violations(state, bucket).items():
        if bool(violated):
            raise ModelError(message)
