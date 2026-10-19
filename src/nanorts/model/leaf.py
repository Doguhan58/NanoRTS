import dataclasses
import typing
from collections.abc import Callable
from typing import Any, Final, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import DTypeLike

from nanorts.model.bucket import BucketSpec
from nanorts.model.errors import ModelError
from nanorts.model.vocabulary import (
    ACTION_KIND_COUNT,
    ATTACK_TARGET_OWNER_COUNT,
    DIRECTION_COUNT,
    PLAYER_COUNT,
)

if typing.TYPE_CHECKING:
    from _typeshed import DataclassInstance


@dataclasses.dataclass(frozen=True)
class Shape:
    name: str
    of: Callable[[BucketSpec], tuple[int, ...]]

    def resolve(self, bucket: BucketSpec) -> tuple[int, ...]:
        return self.of(bucket)


SCALAR: Final = Shape("SCALAR", lambda bucket: ())

PER_SLOT: Final = Shape("PER_SLOT", lambda bucket: (bucket.entity_capacity,))

PER_PLAYER: Final = Shape("PER_PLAYER", lambda bucket: (PLAYER_COUNT,))

PER_OWNED: Final = Shape(
    "PER_OWNED", lambda bucket: (PLAYER_COUNT, bucket.per_player_capacity)
)

FIELD: Final = Shape("FIELD", lambda bucket: (bucket.height, bucket.width))

PER_OWNED_KIND: Final = Shape(
    "PER_OWNED_KIND",
    lambda bucket: (PLAYER_COUNT, bucket.per_player_capacity, ACTION_KIND_COUNT),
)

PER_OWNED_DIRECTION: Final = Shape(
    "PER_OWNED_DIRECTION",
    lambda bucket: (PLAYER_COUNT, bucket.per_player_capacity, DIRECTION_COUNT),
)

PER_OWNED_TYPE: Final = Shape(
    "PER_OWNED_TYPE",
    lambda bucket: (
        PLAYER_COUNT,
        bucket.per_player_capacity,
        bucket.unit_type_capacity,
    ),
)

PER_OWNED_OFFSET: Final = Shape(
    "PER_OWNED_OFFSET",
    lambda bucket: (
        PLAYER_COUNT,
        bucket.per_player_capacity,
        bucket.attack_offset_capacity,
    ),
)

PER_PLAYER_TYPE: Final = Shape(
    "PER_PLAYER_TYPE",
    lambda bucket: (PLAYER_COUNT, bucket.unit_type_capacity),
)

PER_PLAYER_TARGET_OWNER: Final = Shape(
    "PER_PLAYER_TARGET_OWNER",
    lambda bucket: (PLAYER_COUNT, ATTACK_TARGET_OWNER_COUNT),
)


@dataclasses.dataclass(frozen=True)
class IndexOf:
    name: str
    extent: Callable[[BucketSpec], int]

    def resolve(self, bucket: BucketSpec) -> np.dtype[Any]:
        if self.extent(bucket) <= np.iinfo(np.int16).max:
            return np.dtype(np.int16)
        return np.dtype(np.int32)


CELL_INDEX: Final = IndexOf("CELL_INDEX", lambda bucket: bucket.cell_count)

SLOT_INDEX: Final = IndexOf("SLOT_INDEX", lambda bucket: bucket.entity_capacity)


LEAF_METADATA_KEY: Final[str] = "leaf"

STATIC_METADATA_KEY: Final[str] = "static"

NESTED_METADATA_KEY: Final[str] = "nested"


@dataclasses.dataclass(frozen=True)
class Leaf:
    shape: Shape
    dtype: DTypeLike | IndexOf
    fill: int | bool | None = None

    def dtype_under(self, bucket: BucketSpec) -> np.dtype[Any]:
        if isinstance(self.dtype, IndexOf):
            return self.dtype.resolve(bucket)
        return jnp.dtype(self.dtype)

    def canonical_empty(self, bucket: BucketSpec) -> Array:
        if self.fill is None:
            raise ModelError(
                f"a {self.shape.name} leaf declares no canonical empty value"
            )
        return jnp.full(self.shape.resolve(bucket), self.fill, self.dtype_under(bucket))


def leaf(shape: Shape, dtype: DTypeLike | IndexOf, *, fill: int | bool | None = None) -> Any:
    return dataclasses.field(
        metadata={LEAF_METADATA_KEY: Leaf(shape=shape, dtype=dtype, fill=fill)}
    )


def static() -> Any:
    return dataclasses.field(metadata={STATIC_METADATA_KEY: True})


def nested() -> Any:
    return dataclasses.field(metadata={NESTED_METADATA_KEY: True})


def stored[T: DataclassInstance](value: T, **fields: Any) -> T:
    replaced: dict[str, Any] = {}
    for name, new in fields.items():
        old = getattr(value, name)
        replaced[name] = (
            new if dataclasses.is_dataclass(old) else jnp.asarray(new, old.dtype)
        )
    return dataclasses.replace(value, **replaced)


@dataclasses.dataclass(frozen=True)
class LeafField:
    name: str
    spec: Leaf


@dataclasses.dataclass(frozen=True)
class StaticField:
    name: str


@dataclasses.dataclass(frozen=True)
class NestedField:
    name: str
    owner: type[Any]


ModelField = LeafField | StaticField | NestedField


def annotation_of(owner: type[Any], field: "dataclasses.Field[Any]") -> Any:
    return typing.get_type_hints(owner)[field.name]


def model_fields(owner: type[Any]) -> tuple[ModelField, ...]:
    return tuple(_classified(owner, field) for field in dataclasses.fields(owner))


def _classified(owner: type[Any], field: "dataclasses.Field[Any]") -> ModelField:
    path = f"{owner.__name__}.{field.name}"
    spec = field.metadata.get(LEAF_METADATA_KEY)
    if field.metadata.get(STATIC_METADATA_KEY, False):
        return StaticField(name=field.name)
    if isinstance(spec, Leaf):
        return LeafField(name=field.name, spec=spec)
    if field.metadata.get(NESTED_METADATA_KEY, False):
        annotation = annotation_of(owner, field)
        if not (isinstance(annotation, type) and dataclasses.is_dataclass(annotation)):
            raise ModelError(
                f"{path} is declared nested but holds "
                f"{annotation!r}, which is not a model dataclass"
            )
        return NestedField(name=field.name, owner=annotation)
    raise ModelError(
        f"{path} declares none of the three field kinds: it carries no "
        "Leaf specification and is marked neither static nor nested"
    )


@runtime_checkable
class _ValueRules(Protocol):
    def check_values(self) -> None: ...


def empty[T](kind: type[T], bucket: BucketSpec) -> T:
    undetermined = _fields_no_bucket_determines(kind)
    if undetermined:
        raise ModelError(
            f"{undetermined[0]} has no canonical empty value, so "
            f"{kind.__name__} is loaded content and empty() refuses it"
        )
    return _build(kind, bucket)


def _fields_no_bucket_determines(owner: type[Any]) -> tuple[str, ...]:
    undetermined: list[str] = []
    for field in model_fields(owner):
        match field:
            case LeafField() if field.spec.fill is None:
                undetermined.append(f"{owner.__name__}.{field.name}")
            case StaticField():
                undetermined.append(f"{owner.__name__}.{field.name}")
            case NestedField():
                undetermined.extend(_fields_no_bucket_determines(field.owner))
            case _:
                continue
    return tuple(undetermined)


def _build(owner: type[Any], bucket: BucketSpec) -> Any:
    values: dict[str, Any] = {}
    for field in model_fields(owner):
        match field:
            case LeafField():
                values[field.name] = field.spec.canonical_empty(bucket)
            case NestedField():
                values[field.name] = _build(field.owner, bucket)
            case StaticField():
                continue
    return owner(**values)


def validate(value: object, bucket: BucketSpec | None = None) -> None:
    if isinstance(value, BucketSpec):
        if bucket is not None and bucket != value:
            raise ModelError("BucketSpec is not the supplied BucketSpec")
        value.check_values()
        return

    governing = _governing_bucket(value, bucket)
    governing.check_values()
    _check_fields(value, governing)


def _governing_bucket(value: object, supplied: BucketSpec | None) -> BucketSpec:
    carried = _carried_bucket(value)
    if carried is None:
        if supplied is None:
            raise ModelError(
                f"{type(value).__name__} carries no BucketSpec, so validate() "
                "needs one to resolve its leaf shapes against"
            )
        return supplied
    name, governing = carried
    if supplied is not None and supplied != governing:
        raise ModelError(
            f"{type(value).__name__}.{name} is not the supplied BucketSpec"
        )
    return governing


def _carried_bucket(value: object) -> tuple[str, BucketSpec] | None:
    for field in model_fields(type(value)):
        if isinstance(field, StaticField):
            carried = getattr(value, field.name)
            if isinstance(carried, BucketSpec):
                return field.name, carried
    return None


def _check_fields(value: object, bucket: BucketSpec) -> None:
    owner = type(value).__name__
    for field in model_fields(type(value)):
        match field:
            case LeafField():
                _check_leaf(
                    getattr(value, field.name),
                    f"{owner}.{field.name}",
                    field.spec,
                    bucket,
                )
            case NestedField():
                child = getattr(value, field.name)
                if not isinstance(child, field.owner):
                    raise ModelError(
                        f"{owner}.{field.name} is {type(child).__name__}, "
                        f"expected {field.owner.__name__}"
                    )
                _check_fields(child, bucket)
            case StaticField():
                continue
    if isinstance(value, _ValueRules):
        value.check_values()


def _check_leaf(value: Any, path: str, spec: Leaf, bucket: BucketSpec) -> None:
    shape = spec.shape.resolve(bucket)
    dtype = spec.dtype_under(bucket)
    if not isinstance(value, jax.Array):
        raise ModelError(f"{path} is {type(value).__name__}, expected a jax.Array")
    if value.shape != shape:
        raise ModelError(f"{path} has shape {value.shape}, expected {shape}")
    if value.dtype != dtype:
        raise ModelError(f"{path} has dtype {value.dtype}, expected {dtype}")
