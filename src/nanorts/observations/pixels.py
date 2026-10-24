from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Final

import jax
import jax.numpy as jnp
from jax import Array

from nanorts.model import PLAYER_COUNT, GameConfig, Shape, State, leaf
from nanorts.observations.symbolic import (
    FEATURE_MAP_DTYPES,
    GROUPS,
    SCALARS,
    partial_mode,
    scalar_vector,
    symbolic_layout,
)
from nanorts.observations.visibility import observer_index, project
from nanorts.rendering import (
    HIT_POINT_WIDTH,
    KIND_WIDTH,
    SPRITE_PIXELS,
    STACK_BITS,
    RenderAssets,
    RenderError,
    draw,
    render,
)

_CELL_PIXELS: Final[int] = SPRITE_PIXELS

_VALUE_RANGE: Final[tuple[int, int]] = (0, 255)

_RENDERED_IMAGE: Final = Shape(
    "RENDERED_IMAGE",
    lambda bucket: (bucket.height * _CELL_PIXELS, bucket.width * _CELL_PIXELS, 3),
)

_PIXEL_SCALARS: Final = Shape("PIXEL_SCALARS", lambda bucket: (len(SCALARS),))


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PixelObservation:
    image: Array = leaf(_RENDERED_IMAGE, jnp.uint8, fill=0)

    scalars: Array = leaf(_PIXEL_SCALARS, jnp.int32, fill=0)


@dataclasses.dataclass(frozen=True)
class PixelLayout:
    image_shape: tuple[int, int, int]

    image_dtype: jnp.dtype

    value_range: tuple[int, int]

    cell_pixels: int

    scalars_shape: tuple[int]

    scalars_dtype: jnp.dtype

    scalar_names: tuple[str, ...]

    partial_observability: bool


def _relabel(state: State, observer: int) -> State:
    owner = state.owner
    return dataclasses.replace(
        state, owner=jnp.where(owner >= 0, (owner + observer) % PLAYER_COUNT, owner)
    )


def pixels(game: GameConfig, state: State, assets: RenderAssets, *, observer: int, partial_observability: bool) -> PixelObservation:
    player = observer_index(observer)
    projected = (
        project(game, state, observer=player)
        if partial_mode(partial_observability)
        else state
    )
    return PixelObservation(
        image=render(_relabel(projected, player), game, assets),
        scalars=scalar_vector(state, player),
    )


def pixels_from_symbolic(feature_maps: Array, assets: RenderAssets) -> Array:
    if feature_maps.ndim != 3 or feature_maps.shape[-1] != len(GROUPS):
        raise ValueError(
            f"the Feature maps are {feature_maps.shape}, expected one "
            f"encoding's (height, width, {len(GROUPS)})"
        )
    if feature_maps.dtype not in FEATURE_MAP_DTYPES:
        raise ValueError(
            f"the Feature map dtype is {feature_maps.dtype}, expected one of "
            f"{[accepted.name for accepted in FEATURE_MAP_DTYPES]}"
        )
    widened = feature_maps.astype(jnp.int32)
    group = {name: widened[..., index] for index, name in enumerate(GROUPS)}
    unit_type = group["unit_type"]
    return draw(
        group["terrain"],
        unit_type != 0,
        group["owner"] - 1,
        unit_type - 1,
        group["pending_action_kind"],
        group["carried_resources"],
        group["hit_points"],
        assets,
    )


def pixel_layout(game: GameConfig, assets: RenderAssets, *, partial_observability: bool) -> PixelLayout:
    layout = symbolic_layout(game, partial_observability=partial_observability)
    _check_published_zoom(assets)
    _check_glyph_band_holds(layout.bound)
    height, width, channels = _RENDERED_IMAGE.resolve(game.bucket)
    (scalars,) = _PIXEL_SCALARS.resolve(game.bucket)
    return PixelLayout(
        image_shape=(height, width, channels),
        image_dtype=jnp.dtype(jnp.uint8),
        value_range=_VALUE_RANGE,
        cell_pixels=_CELL_PIXELS,
        scalars_shape=(scalars,),
        scalars_dtype=layout.scalars_dtype,
        scalar_names=layout.scalar_names,
        partial_observability=layout.partial_observability,
    )


def _check_published_zoom(assets: RenderAssets) -> None:
    if assets.cell_pixels != _CELL_PIXELS:
        raise RenderError(
            f"the prepared Asset set draws {assets.cell_pixels} pixels per "
            f"cell and the published surface is {_CELL_PIXELS}; an image at "
            "another zoom is a Rendering, not a Pixel observation"
        )


def _check_glyph_band_holds(bound: Mapping[str, int]) -> None:
    hit_points = bound["hit_points"]
    if hit_points > HIT_POINT_WIDTH:
        raise RenderError(
            f"the content reaches {hit_points} hit points and the Glyph band "
            f"draws {HIT_POINT_WIDTH} unary pixels, so a Pixel "
            "observation of it would not carry the value"
        )
    kind = bound["pending_action_kind"]
    if kind > KIND_WIDTH:
        raise RenderError(
            f"the content reaches Pending action kind {kind} and the Glyph "
            f"band draws one pixel for each of {KIND_WIDTH} kinds, so a Pixel observation of "
            "it would not carry the value"
        )
    carried = bound["carried_resources"]
    if carried >= 2**STACK_BITS:
        raise RenderError(
            f"the content reaches {carried} Carried resources and the Glyph "
            f"band draws {STACK_BITS} binary bits, so a Pixel "
            "observation of it would not carry the value"
        )
