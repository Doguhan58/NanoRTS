from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array, lax


def visit_in_order[Carry](count: Array, visit: Callable[[Array, Carry], Carry], start: Carry) -> Carry:
    def unfinished(carry: tuple[Array, Carry]) -> Array:
        position, _ = carry
        return position < count

    def advance(carry: tuple[Array, Carry]) -> tuple[Array, Carry]:
        position, carried = carry
        return position + 1, visit(position, carried)

    _, finished = lax.while_loop(
        unfinished, advance, (jnp.asarray(0, jnp.int32), start)
    )
    return finished


def masked_set(field: Array, index: Array, when: Array, value: Array | int) -> Array:
    positions = jnp.arange(field.shape[0], dtype=jnp.int32)
    return jnp.where(when & (positions == index), value, field)
