from __future__ import annotations

from typing import Final

import jax.numpy as jnp
import numpy as np
from jax import Array

from nanorts.model import (
    ABSENT,
    ACTION_KIND_COUNT,
    DIRECTION_COUNT,
    KIND_ATTACK,
    KIND_HARVEST,
    KIND_MOVE,
    KIND_PRODUCE,
    KIND_RETURN,
    BucketSpec,
    Legality,
    ModelError,
)

MASK_FORMS: Final[tuple[str, ...]] = ("rows", "packed")

MASK_WORD_DTYPE: Final[jnp.dtype] = jnp.dtype(jnp.uint32)
MASK_WORDS: Final[int] = 3

_WORD_BITS: Final[int] = MASK_WORD_DTYPE.itemsize * 8


def widths(bucket: BucketSpec) -> tuple[int, ...]:
    return (
        ACTION_KIND_COUNT,
        DIRECTION_COUNT,
        DIRECTION_COUNT,
        DIRECTION_COUNT,
        DIRECTION_COUNT,
        bucket.unit_type_capacity,
        bucket.attack_offset_capacity,
    )


def offsets(group_widths: tuple[int, ...]) -> tuple[int, ...]:
    first: list[int] = [0]
    for width in group_widths[:-1]:
        first.append(first[-1] + width)
    return tuple(first)


def concatenated(legality: Legality) -> Array:
    return jnp.concatenate(
        [
            legality.kind,
            legality.move_direction,
            legality.harvest_direction,
            legality.return_direction,
            legality.produce_direction,
            legality.produce_type,
            legality.attack_offset,
        ],
        axis=-1,
    )


def refuse_mask_form(where: str, form: str, bucket: BucketSpec) -> None:
    if form not in MASK_FORMS:
        raise ModelError(
            f"{where} refuses mask_form={form!r}: a mask publishes one of "
            f"{list(MASK_FORMS)}, its rows or their factor-aligned words"
        )
    if form != "packed":
        return
    group_widths = widths(bucket)
    head = sum(group_widths[:-1])
    if head > _WORD_BITS:
        raise ModelError(
            f"{where} refuses mask_form='packed' at BucketSpec.unit_type_capacity "
            f"{bucket.unit_type_capacity}: word 0 holds the six non-attack groups, "
            f"{head} bits against {_WORD_BITS}"
        )
    if bucket.attack_offset_capacity > 2 * _WORD_BITS:
        raise ModelError(
            f"{where} refuses mask_form='packed' at "
            f"BucketSpec.attack_offset_capacity {bucket.attack_offset_capacity}: "
            f"words 1 and 2 hold the attack window, {2 * _WORD_BITS} bits"
        )


def pack(rows: Array, group_widths: tuple[int, ...]) -> Array:
    head = sum(group_widths[:-1])
    middle = head + min(group_widths[-1], _WORD_BITS)
    words = []
    for start, stop in ((0, head), (head, middle), (middle, head + group_widths[-1])):
        shifts = np.arange(stop - start, dtype=np.uint32)
        words.append(
            jnp.sum(
                rows[..., start:stop].astype(MASK_WORD_DTYPE) << shifts,
                axis=-1,
                dtype=MASK_WORD_DTYPE,
            )
        )
    return jnp.stack(words, axis=-1)


def unpack(words: Array, group_widths: tuple[int, ...]) -> Array:
    head = sum(group_widths[:-1])
    shifts = np.arange(_WORD_BITS, dtype=np.uint32)
    bits = (((words[..., None] >> shifts) & np.uint32(1)) != 0).reshape(
        *words.shape[:-1], MASK_WORDS * _WORD_BITS
    )
    return jnp.concatenate(
        [bits[..., :head], bits[..., _WORD_BITS : _WORD_BITS + group_widths[-1]]],
        axis=-1,
    )


def direction(kind: Array, rows: Array) -> Array:
    return jnp.where(
        kind == KIND_MOVE,
        rows[..., 1],
        jnp.where(
            kind == KIND_HARVEST,
            rows[..., 2],
            jnp.where(
                kind == KIND_RETURN,
                rows[..., 3],
                jnp.where(kind == KIND_PRODUCE, rows[..., 4], ABSENT),
            ),
        ),
    )


def produced_type(kind: Array, rows: Array) -> Array:
    return jnp.where(kind == KIND_PRODUCE, rows[..., 5], ABSENT)


def attack_target(kind: Array, rows: Array, cell: Array, bucket: BucketSpec) -> Array:
    width = bucket.width
    side, radius = bucket.attack_window_side, bucket.max_attack_range
    offset = rows[..., 6]
    target_x = cell % width + (offset % side - radius)
    target_y = cell // width + (offset // side - radius)
    on_board = (
        (target_x >= 0)
        & (target_x < width)
        & (target_y >= 0)
        & (target_y < bucket.height)
    )
    return jnp.where(
        (kind == KIND_ATTACK) & on_board, target_y * width + target_x, ABSENT
    )


def refuse_out_of_range(array: np.ndarray, names: tuple[str, ...], sizes: tuple[int, ...], *, subject: str, exempt: str) -> None:
    for index, (name, size) in enumerate(zip(names, sizes, strict=True)):
        factor = array[..., index]
        if factor.size and (factor.min() < 0 or factor.max() >= size):
            raise ModelError(
                f"{subject.format(index=index, name=name)} carries values in "
                f"[{int(factor.min())}, {int(factor.max())}], expected "
                f"[0, {size - 1}]; every factor obeys its range including {exempt}"
            )
