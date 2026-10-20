import dataclasses
import functools
import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final

import jax.numpy as jnp
import numpy as np
from jax import Array

from nanorts.model.errors import ModelError
from nanorts.model.vocabulary import UnitTypeTableVersion


@dataclasses.dataclass(frozen=True)
class UnitType:
    name: str
    cost: int = 1
    hit_points: int = 1
    produce_time: int = 10
    move_time: int = 10
    attack_time: int = 10
    harvest_time: int = 10
    attack_range: int = 1
    attack_damage: int = 1
    harvest_amount: int = 1
    sight_radius: int = 4
    is_resource: bool = False
    is_stockpile: bool = False
    can_harvest: bool = False
    can_move: bool = True
    can_attack: bool = True
    produced_by: tuple[str, ...] = ()


_ATTRIBUTES: Final = tuple(
    field
    for field in dataclasses.fields(UnitType)
    if field.name not in ("name", "produced_by")
)

_DTYPES: Final[Mapping[Any, Any]] = MappingProxyType({int: jnp.int32, bool: jnp.bool_})


_RESOURCE: Final = UnitType(
    "Resource", sight_radius=0, is_resource=True, can_move=False, can_attack=False
)
_BASE: Final = UnitType(
    "Base",
    cost=10,
    hit_points=10,
    produce_time=250,
    sight_radius=5,
    is_stockpile=True,
    can_move=False,
    can_attack=False,
    produced_by=("Worker",),
)
_BARRACKS: Final = UnitType(
    "Barracks",
    cost=5,
    hit_points=4,
    produce_time=200,
    sight_radius=3,
    can_move=False,
    can_attack=False,
    produced_by=("Worker",),
)
_WORKER: Final = UnitType(
    "Worker",
    produce_time=50,
    attack_time=5,
    harvest_time=20,
    sight_radius=3,
    can_harvest=True,
    produced_by=("Base",),
)
_LIGHT: Final = UnitType(
    "Light",
    cost=2,
    hit_points=4,
    produce_time=80,
    move_time=8,
    attack_time=5,
    attack_damage=2,
    sight_radius=2,
    produced_by=("Barracks",),
)
_HEAVY: Final = UnitType(
    "Heavy",
    cost=2,
    hit_points=4,
    produce_time=120,
    move_time=12,
    attack_time=5,
    attack_damage=4,
    sight_radius=2,
    produced_by=("Barracks",),
)
_RANGED: Final = UnitType(
    "Ranged",
    cost=2,
    produce_time=100,
    attack_time=5,
    attack_range=3,
    sight_radius=3,
    produced_by=("Barracks",),
)

_ORDER: Final[tuple[UnitType, ...]] = (
    _RESOURCE,
    _BASE,
    _BARRACKS,
    _WORKER,
    _LIGHT,
    _HEAVY,
    _RANGED,
)

TYPE_RESOURCE: Final[int] = _ORDER.index(_RESOURCE)
TYPE_BASE: Final[int] = _ORDER.index(_BASE)
TYPE_BARRACKS: Final[int] = _ORDER.index(_BARRACKS)
TYPE_WORKER: Final[int] = _ORDER.index(_WORKER)
TYPE_LIGHT: Final[int] = _ORDER.index(_LIGHT)
TYPE_HEAVY: Final[int] = _ORDER.index(_HEAVY)
TYPE_RANGED: Final[int] = _ORDER.index(_RANGED)

TYPE_NAMES: Final[tuple[str, ...]] = tuple(unit.name for unit in _ORDER)

_FINETUNED_ORDER: Final[tuple[UnitType, ...]] = (
    _RESOURCE,
    dataclasses.replace(_BASE, produce_time=200),
    dataclasses.replace(_BARRACKS, produce_time=100),
    _WORKER,
    _LIGHT,
    dataclasses.replace(_HEAVY, cost=3, hit_points=8, move_time=10),
    _RANGED,
)


TABLES: Final[Mapping[UnitTypeTableVersion, tuple[UnitType, ...]]] = MappingProxyType(
    {
        UnitTypeTableVersion.ORIGINAL: _ORDER,
        UnitTypeTableVersion.ORIGINAL_FINETUNED: _FINETUNED_ORDER,
    }
)


@functools.cache
def host_arrays(table: tuple[UnitType, ...]) -> Mapping[str, np.ndarray]:
    names = [unit.name for unit in table]
    if shared := sorted({name for name in names if names.count(name) > 1}):
        raise ModelError(f"the table declares {shared} more than once")
    producers = {name for unit in table for name in unit.produced_by}
    if unknown := sorted(producers - set(names)):
        raise ModelError(f"produced_by names {unknown}, which the table does not hold")
    columns = {
        f"unit_{field.name}": np.asarray(
            [getattr(unit, field.name) for unit in table], _DTYPES[field.type]
        )
        for field in _ATTRIBUTES
    }
    columns["unit_produces"] = np.asarray(
        [
            [producer.name in produced.produced_by for produced in table]
            for producer in table
        ],
        jnp.bool_,
    )
    for column in columns.values():
        column.flags.writeable = False
    return MappingProxyType(columns)


def type_arrays(table: tuple[UnitType, ...]) -> dict[str, Array]:
    return {name: jnp.asarray(array) for name, array in host_arrays(table).items()}


def content_hash(table: tuple[UnitType, ...]) -> str:
    payload = {name: array.tolist() for name, array in host_arrays(table).items()}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
