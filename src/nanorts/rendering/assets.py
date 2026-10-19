from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

import imageio.v3 as iio
import jax.numpy as jnp
import numpy as np

from nanorts.model import TYPE_NAMES, BucketSpec
from nanorts.rendering.raster import (
    BAND_COLUMN,
    BAND_ROWS,
    BAND_WIDTH,
    EMPTY_OCCUPANT,
    OWNER_COUNT,
    SPRITE_PIXELS,
    RenderAssets,
    occupant_index,
    occupant_states,
    tile_count,
    tile_index,
)


class RenderError(Exception):
    ...


_ART_DIRECTORY: Final[Path] = Path(__file__).resolve().parents[1] / "art"

_TERRAIN_NAMES: Final[tuple[str, ...]] = ("floor", "wall")

_OWNER_NAMES: Final[tuple[str, ...]] = ("neutral", "player0", "player1")

_TYPE_STEMS: Final[tuple[str, ...]] = tuple(name.lower() for name in TYPE_NAMES)

_AURA_COLOURS: Final[tuple[tuple[int, int, int], ...]] = (
    (0xAD, 0xBB, 0xC7),
    (0x0C, 0x7B, 0xDC),
    (0xBF, 0x36, 0x82),
)


def _sprite_names() -> tuple[str, ...]:
    return _TERRAIN_NAMES + tuple(
        f"{owner}-{name}" for owner in _OWNER_NAMES for name in _TYPE_STEMS
    )


def prepare_assets(bucket: BucketSpec, cell_pixels: int = SPRITE_PIXELS, tiles: Mapping[str, np.ndarray] | None = None) -> RenderAssets:
    _check_cell_pixels(cell_pixels)
    types = len(_TYPE_STEMS)
    _check_unit_type_capacity(bucket, types)
    names = _sprite_names()
    supplied = _decode(_ART_DIRECTORY, names) if tiles is None else dict(tiles)
    _check_names_every_tile_class(supplied, names, types)
    checked = {name: _sprite(supplied, name) for name in names}
    _check_terrain_opaque(checked)
    table = _composite(checked, types, cell_pixels)
    _check_distinct(table, types)
    return RenderAssets(
        tiles=jnp.asarray(table),
        cell_pixels=cell_pixels,
        unit_type_capacity=types,
    )


def _check_cell_pixels(cell_pixels: int) -> None:
    if cell_pixels <= 0 or cell_pixels % SPRITE_PIXELS:
        raise RenderError(
            f"cell_pixels is {cell_pixels}, expected a positive integer "
            f"multiple of the {SPRITE_PIXELS}-pixel sprite side"
        )


def _check_unit_type_capacity(bucket: BucketSpec, types: int) -> None:
    if bucket.unit_type_capacity != types:
        raise RenderError(
            f"BucketSpec.unit_type_capacity is {bucket.unit_type_capacity}, "
            f"and an Asset set draws the {types} types the supported unit "
            "type table holds"
        )


def _check_names_every_tile_class(supplied: Mapping[str, np.ndarray], names: tuple[str, ...], types: int) -> None:
    if set(supplied) != set(names):
        raise RenderError(
            "the Asset set does not name the Tile classes a Bucket of "
            f"{types} unit types has: it is missing "
            f"{sorted(set(names) - set(supplied))} and holds an unused "
            f"{sorted(set(supplied) - set(names))}"
        )


def _check_terrain_opaque(checked: Mapping[str, np.ndarray]) -> None:
    for name in _TERRAIN_NAMES:
        if not bool(np.all(checked[name][..., 3] >= 128)):
            raise RenderError(
                f"the terrain tile {name!r} is not fully opaque, and terrain is "
                "the ground every occupant is drawn over"
            )


def _decode(directory: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {name: np.asarray(iio.imread(directory / f"{name}.png")) for name in names}


def _sprite(supplied: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    image = np.asarray(supplied[name])
    if image.shape != (SPRITE_PIXELS, SPRITE_PIXELS, 4) or image.dtype != np.uint8:
        raise RenderError(
            f"the sprite {name!r} is {image.shape} {image.dtype}, expected a "
            f"({SPRITE_PIXELS}, {SPRITE_PIXELS}, 4) uint8 RGBA image"
        )
    return image


def _upscaled(image: np.ndarray, cell_pixels: int) -> np.ndarray:
    scale = cell_pixels // SPRITE_PIXELS
    return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)


def _ring(cell_pixels: int) -> np.ndarray:
    width = cell_pixels // SPRITE_PIXELS
    mask = np.zeros((cell_pixels, cell_pixels), np.bool_)
    mask[:width, :] = mask[-width:, :] = True
    mask[:, :width] = mask[:, -width:] = True
    return mask


def _composite(sprites: Mapping[str, np.ndarray], unit_type_capacity: int, cell_pixels: int) -> np.ndarray:
    border = _ring(cell_pixels)
    table = np.zeros(
        (tile_count(unit_type_capacity), cell_pixels, cell_pixels, 3), np.uint8
    )
    for terrain, name in enumerate(_TERRAIN_NAMES):
        ground = _upscaled(sprites[name], cell_pixels)[..., :3]
        for bucketed_owner in range(OWNER_COUNT):
            for unit_type in range(unit_type_capacity):
                art = _upscaled(
                    sprites[f"{_OWNER_NAMES[bucketed_owner]}-{_TYPE_STEMS[unit_type]}"],
                    cell_pixels,
                )
                tile = ground.copy()
                tile[border] = _AURA_COLOURS[bucketed_owner]
                drawn = art[..., 3] >= 128
                tile[drawn] = art[drawn][..., :3]
                occupant = occupant_index(
                    bucketed_owner - 1, unit_type, unit_type_capacity
                )
                table[int(tile_index(terrain, occupant, unit_type_capacity))] = tile
        table[int(tile_index(terrain, EMPTY_OCCUPANT, unit_type_capacity))] = ground
    return table


def _check_distinct(table: np.ndarray, unit_type_capacity: int) -> None:
    drawn: dict[bytes, str] = {}
    outside: dict[bytes, str] = {}
    rows, columns = _band_region(table.shape[1])
    for index, tile in enumerate(table):
        name = _class_name(index, unit_type_capacity)
        _refuse_duplicate(drawn, tile.tobytes(), name, "composite")
        occupant = index % occupant_states(unit_type_capacity)
        if occupant == EMPTY_OCCUPANT:
            continue
        blanked = tile.copy()
        blanked[rows, columns] = 0
        _refuse_duplicate(outside, blanked.tobytes(), name, "draw as")


def _band_region(cell_pixels: int) -> tuple[slice, slice]:
    scale = cell_pixels // SPRITE_PIXELS
    return (
        slice(BAND_ROWS[0] * scale, (BAND_ROWS[-1] + 1) * scale),
        slice(BAND_COLUMN * scale, (BAND_COLUMN + BAND_WIDTH) * scale),
    )


def _refuse_duplicate(seen: dict[bytes, str], picture: bytes, name: str, verb: str) -> None:
    if picture in seen:
        raise RenderError(
            f"the Tile classes {seen[picture]!r} and {name!r} {verb} "
            "the same picture, which would make them indistinguishable "
            "in a Rendering"
        )
    seen[picture] = name


def _class_name(index: int, unit_type_capacity: int) -> str:
    terrain, occupant = divmod(index, occupant_states(unit_type_capacity))
    if occupant == 0:
        return f"{_TERRAIN_NAMES[terrain]} + empty"
    owner, unit_type = divmod(occupant - 1, unit_type_capacity)
    return f"{_TERRAIN_NAMES[terrain]} + {_OWNER_NAMES[owner]} {_TYPE_STEMS[unit_type]}"
