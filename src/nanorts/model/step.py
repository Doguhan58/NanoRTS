from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from jax import Array

from nanorts.model.leaf import (
    PER_PLAYER,
    PER_PLAYER_TARGET_OWNER,
    PER_PLAYER_TYPE,
    PER_SLOT,
    SCALAR,
    leaf,
    nested,
)
from nanorts.model.state import State
from nanorts.model.vocabulary import ABSENT


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StepEvents:
    admitted_harvest: Array = leaf(PER_PLAYER, jnp.int32, fill=0)

    admitted_return: Array = leaf(PER_PLAYER, jnp.int32, fill=0)

    admitted_produce: Array = leaf(PER_PLAYER_TYPE, jnp.int32, fill=0)

    admitted_attack_by_target_owner: Array = leaf(
        PER_PLAYER_TARGET_OWNER, jnp.int32, fill=0
    )

    hit_valid: Array = leaf(PER_SLOT, jnp.bool_, fill=False)

    hit_actor_birth: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)

    hit_actor_owner: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)

    hit_actor_unit_type: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)

    hit_target_slot: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)

    hit_target_birth: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)

    hit_damage: Array = leaf(PER_SLOT, jnp.int32, fill=0)

    death_valid: Array = leaf(PER_SLOT, jnp.bool_, fill=False)

    death_actor_birth: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)

    death_actor_owner: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)

    death_actor_unit_type: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)

    death_slot: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)

    death_birth: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)

    death_owner: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)

    death_unit_type: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)

    harvest_valid: Array = leaf(PER_SLOT, jnp.bool_, fill=False)

    harvest_actor_birth: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)

    harvest_actor_owner: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)

    harvest_actor_unit_type: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)

    harvest_target_slot: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)

    harvest_target_birth: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)

    harvest_amount: Array = leaf(PER_SLOT, jnp.int32, fill=0)

    harvest_depleted: Array = leaf(PER_SLOT, jnp.bool_, fill=False)

    return_valid: Array = leaf(PER_SLOT, jnp.bool_, fill=False)

    return_actor_birth: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)

    return_actor_owner: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)

    return_actor_unit_type: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)

    return_target_slot: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)

    return_target_birth: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)

    return_amount: Array = leaf(PER_SLOT, jnp.int32, fill=0)

    return_balance: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)

    spawn_valid: Array = leaf(PER_SLOT, jnp.bool_, fill=False)

    spawn_actor_birth: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)

    spawn_actor_owner: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)

    spawn_actor_unit_type: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)

    spawn_produced_type: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)

    spawn_slot: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)

    spawn_birth: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)

    spawn_cell: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)

    spawn_balance: Array = leaf(PER_SLOT, jnp.int32, fill=ABSENT)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StepResult:
    next_state: State = nested()

    terminated: Array = leaf(SCALAR, jnp.bool_, fill=False)

    truncated: Array = leaf(SCALAR, jnp.bool_, fill=False)

    reward: Array = leaf(PER_PLAYER, jnp.float32, fill=0)

    events: StepEvents = nested()
