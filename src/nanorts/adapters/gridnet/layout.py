from __future__ import annotations

import dataclasses
from typing import Final

import jax.numpy as jnp
from jax.typing import DTypeLike

from nanorts import observations
from nanorts.adapters import factors, lifecycle
from nanorts.model import PLAYER_COUNT, BucketSpec, GameConfig, ModelError

ACTION_FACTORS: Final[tuple[str, ...]] = (
    "action_type",
    "move_direction",
    "harvest_direction",
    "return_direction",
    "produce_direction",
    "produce_type",
    "attack_offset",
)

ACTION_FACTOR_SIZES: Final[tuple[int, ...]] = (6, 4, 4, 4, 4, 7, 49)

PLANE_WIDTHS: Final[tuple[int, ...]] = (5, 5, 3, 8, 6, 2)

MASK_WIDTH: Final[int] = 78

if sum(ACTION_FACTOR_SIZES) != MASK_WIDTH:
    raise ImportError("the pinned action factor sizes must sum to the mask width")

OUTPUT_DTYPES: Final[tuple[jnp.dtype, ...]] = (
    jnp.dtype(jnp.int32),
    jnp.dtype(jnp.uint8),
    jnp.dtype(jnp.bool_),
)

PLANE_FORMS: Final[tuple[str, ...]] = ("one_hot", "indices")

MASK_FORMS: Final[tuple[str, ...]] = factors.MASK_FORMS

PLANE_INDEX_DTYPE: Final[jnp.dtype] = jnp.dtype(jnp.int8)

if max(PLANE_WIDTHS) > jnp.iinfo(PLANE_INDEX_DTYPE).max + 1:
    raise ImportError("every Plane index must fit the indices form's dtype")

_PARTIAL_LINES: Final[str] = (
    "vec_env.py:151-152 (two extra one-wide groups, giving 31 planes), "
    "JNIGridnetClient.java:164-167 (both perspectives wrapped in "
    "PartiallyObservableGameState, giving the eight-Feature-map encoding), and "
    "vec_env.py:182 (the partialObs constructor argument)"
)


@dataclasses.dataclass(frozen=True)
class GridNetLayout:
    bucket: BucketSpec

    groups: tuple[str, ...]

    plane_widths: tuple[int, ...]

    plane_offsets: tuple[int, ...]

    plane_count: int

    action_factor_sizes: tuple[int, ...]

    mask_width: int

    mask_offsets: tuple[int, ...]

    planes_form: str

    planes_dtype: jnp.dtype

    mask_form: str

    mask_dtype: jnp.dtype

    partial_observability: bool

    @property
    def actions_shape(self) -> tuple[int, int, int, int]:
        return (
            PLAYER_COUNT,
            self.bucket.height,
            self.bucket.width,
            len(ACTION_FACTORS),
        )

    @property
    def planes_shape(self) -> tuple[int, int, int, int]:
        width = (
            self.plane_count
            if self.planes_form == "one_hot"
            else len(self.plane_widths)
        )
        return (PLAYER_COUNT, self.bucket.height, self.bucket.width, width)

    @property
    def source_shape(self) -> tuple[int, int, int]:
        return (PLAYER_COUNT, self.bucket.height, self.bucket.width)

    @property
    def mask_shape(self) -> tuple[int, int, int, int]:
        width = self.mask_width if self.mask_form == "rows" else factors.MASK_WORDS
        return (PLAYER_COUNT, self.bucket.height, self.bucket.width, width)


def prepare(game: GameConfig, *, partial_observability: bool = False, planes_form: str = "one_hot", planes_dtype: DTypeLike = jnp.int32, mask_form: str = "rows", mask_dtype: DTypeLike = jnp.int32) -> GridNetLayout:
    _refuse_partial_observability(partial_observability)
    bucket = game.bucket
    _refuse_unpinned_bucket(bucket)
    _refuse_unknown_planes_form(planes_form)
    factors.refuse_mask_form("gridnet.prepare", mask_form, bucket)
    planes = _output_dtype("planes_dtype", planes_dtype)
    mask = _output_dtype("mask_dtype", mask_dtype)
    planes = _planes_dtype_of_form(planes_form, planes, planes_dtype)
    mask = _mask_dtype_of_form(mask_form, mask, mask_dtype)
    lifecycle.refuse_decided("gridnet.prepare", game)
    layout = observations.symbolic_layout(game, partial_observability=False)
    return GridNetLayout(
        bucket=bucket,
        groups=layout.groups,
        plane_widths=PLANE_WIDTHS,
        plane_offsets=factors.offsets(PLANE_WIDTHS),
        plane_count=sum(PLANE_WIDTHS),
        action_factor_sizes=ACTION_FACTOR_SIZES,
        mask_width=MASK_WIDTH,
        mask_offsets=factors.offsets(factors.widths(bucket)),
        planes_form=planes_form,
        planes_dtype=planes,
        mask_form=mask_form,
        mask_dtype=mask,
        partial_observability=False,
    )


def _refuse_partial_observability(partial_observability: bool) -> None:
    if partial_observability is not False:
        raise ModelError(
            "gridnet.prepare refuses partial_observability="
            f"{partial_observability!r}: MicroRTS-Py's 31-plane partial-observation "
            f"form is outside the supported release matrix. It changes {_PARTIAL_LINES}"
        )


def _refuse_unpinned_bucket(bucket: BucketSpec) -> None:
    types = bucket.unit_type_capacity
    if types != PLANE_WIDTHS[3] - 1:
        raise ModelError(
            f"gridnet.prepare refuses BucketSpec.unit_type_capacity {types}: the "
            f"pinned surface's type Plane group is {PLANE_WIDTHS[3]} wide and its "
            f"produce-type factor {ACTION_FACTOR_SIZES[5]} wide, both derived from "
            "the supported seven-type unit type table"
        )
    if bucket.attack_offset_capacity != ACTION_FACTOR_SIZES[6]:
        raise ModelError(
            "gridnet.prepare refuses BucketSpec.attack_offset_capacity "
            f"{bucket.attack_offset_capacity}: the pinned attack window is "
            f"{ACTION_FACTOR_SIZES[6]}, a {bucket.attack_window_side}x"
            f"{bucket.attack_window_side} window at maximum attack range "
            f"{bucket.max_attack_range} being the supported one"
        )


def _refuse_unknown_planes_form(planes_form: str) -> None:
    if planes_form not in PLANE_FORMS:
        raise ModelError(
            f"gridnet.prepare refuses planes_form={planes_form!r}: the Planes "
            f"publish one of {list(PLANE_FORMS)}, the pinned one-hot or its "
            "Plane indices"
        )


def _planes_dtype_of_form(planes_form: str, planes: jnp.dtype, planes_dtype: DTypeLike) -> jnp.dtype:
    if planes_form != "indices":
        return planes
    if planes != jnp.dtype(jnp.int32):
        raise ModelError(
            f"gridnet.prepare refuses planes_dtype={planes_dtype!r} beside "
            "planes_form='indices': planes_dtype narrows the one-hot Planes, "
            f"and the indices form publishes {PLANE_INDEX_DTYPE.name} Plane "
            "indices"
        )
    return PLANE_INDEX_DTYPE


def _mask_dtype_of_form(mask_form: str, mask: jnp.dtype, mask_dtype: DTypeLike) -> jnp.dtype:
    if mask_form != "packed":
        return mask
    if mask != jnp.dtype(jnp.int32):
        raise ModelError(
            f"gridnet.prepare refuses mask_dtype={mask_dtype!r} beside "
            "mask_form='packed': mask_dtype narrows the mask rows, and the "
            f"packed form publishes {factors.MASK_WORD_DTYPE.name} words"
        )
    return factors.MASK_WORD_DTYPE


def _output_dtype(keyword: str, dtype: DTypeLike) -> jnp.dtype:
    try:
        published = jnp.dtype(dtype)
    except TypeError:
        published = None
    if published is None or published not in OUTPUT_DTYPES:
        raise ModelError(
            f"gridnet.prepare refuses {keyword}={dtype!r}: the Planes and the "
            f"action mask publish one of {[name.name for name in OUTPUT_DTYPES]}, "
            "each holding the surface's 0/1 values, with int32 the pinned "
            "MicroRTS-Py default"
        )
    return published
