from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from nanorts.model.errors import ModelError
from nanorts.model.leaf import (
    PER_OWNED,
    PER_OWNED_DIRECTION,
    PER_OWNED_KIND,
    PER_OWNED_OFFSET,
    PER_OWNED_TYPE,
    leaf,
)
from nanorts.model.vocabulary import ABSENT, KIND_ABSENT, PLAYER_COUNT


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Commands:
    kind: Array = leaf(PER_OWNED, jnp.int32, fill=KIND_ABSENT)

    direction: Array = leaf(PER_OWNED, jnp.int32, fill=ABSENT)

    none_duration: Array = leaf(PER_OWNED, jnp.int32, fill=0)

    target_cell: Array = leaf(PER_OWNED, jnp.int32, fill=ABSENT)

    produced_type: Array = leaf(PER_OWNED, jnp.int32, fill=ABSENT)

    request_order: Array = leaf(PER_OWNED, jnp.int32, fill=ABSENT)

    def check_values(self) -> None:
        kind = np.asarray(self.kind)
        request_order = np.asarray(self.request_order)
        for player in range(PLAYER_COUNT):
            issued = kind[player] != KIND_ABSENT
            _check_no_rank_without_command(request_order[player], issued, player)
            _check_ranks_are_unit_order_or_permutation(
                request_order[player][issued], player
            )


def _check_no_rank_without_command(ranks: np.ndarray, issued: np.ndarray, player: int) -> None:
    ranked_without_command = ranks[~issued] != ABSENT
    if np.any(ranked_without_command):
        raise ModelError(
            f"Commands.request_order ranks {int(np.sum(ranked_without_command))} "
            f"slot(s) of player {player} that carry no Command; a rank "
            "addresses a request and an absent Command has none"
        )


def _check_ranks_are_unit_order_or_permutation(requested: np.ndarray, player: int) -> None:
    if requested.size == 0 or bool(np.all(requested == ABSENT)):
        return
    if not np.array_equal(np.sort(requested), np.arange(requested.size)):
        raise ModelError(
            f"Commands.request_order for player {player} is "
            f"{sorted(int(rank) for rank in requested)}, expected either "
            f"every rank {ABSENT} for Unit order or a permutation of "
            f"0..{requested.size - 1}"
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Legality:
    available: Array = leaf(PER_OWNED, jnp.bool_, fill=False)

    kind: Array = leaf(PER_OWNED_KIND, jnp.bool_, fill=False)

    move_direction: Array = leaf(PER_OWNED_DIRECTION, jnp.bool_, fill=False)

    harvest_direction: Array = leaf(PER_OWNED_DIRECTION, jnp.bool_, fill=False)

    return_direction: Array = leaf(PER_OWNED_DIRECTION, jnp.bool_, fill=False)

    produce_direction: Array = leaf(PER_OWNED_DIRECTION, jnp.bool_, fill=False)

    produce_type: Array = leaf(PER_OWNED_TYPE, jnp.bool_, fill=False)

    attack_offset: Array = leaf(PER_OWNED_OFFSET, jnp.bool_, fill=False)
