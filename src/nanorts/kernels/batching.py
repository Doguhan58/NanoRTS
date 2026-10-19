from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from jax.custom_batching import custom_vmap

from nanorts.kernels import library


def call(target: str, operands: Sequence[Array], results: Sequence[jax.ShapeDtypeStruct], *, aliases: Mapping[int, int] | None = None, attributes: Mapping[str, int | str] | None = None) -> list[Array]:
    if library.LIBRARY is None:
        raise RuntimeError(f"no kernel library is loaded: {library.REASON}")
    spec = _Spec(
        target=target,
        results=tuple(jax.ShapeDtypeStruct(r.shape, r.dtype) for r in results),
        aliases=tuple(sorted((aliases or {}).items())),
        attributes=tuple(
            sorted({**(attributes or {}), "build_id": library.LIBRARY.digest}.items())
        ),
    )
    return list(_lane(spec)(*operands))


@dataclasses.dataclass(frozen=True)
class _Spec:
    target: str
    results: tuple[jax.ShapeDtypeStruct, ...]
    aliases: tuple[tuple[int, int], ...]
    attributes: tuple[tuple[str, int | str], ...]

    @property
    def aliased(self) -> frozenset[int]:
        return frozenset(operand for operand, _ in self.aliases)


def _lane(spec: _Spec) -> Any:

    @custom_vmap
    def lane(*operands: Array) -> list[Array]:
        return [out[0] for out in _batch(spec, 1)(*(x[None] for x in operands))]

    @lane.def_vmap
    def rule(axis_size: int, in_batched: Sequence[bool], *operands: Array) -> tuple[list[Array], list[bool]]:
        flat = [
            x
            if batched
            else (
                jnp.broadcast_to(x, (axis_size, *x.shape))
                if index in spec.aliased
                else x[None]
            )
            for index, (x, batched) in enumerate(zip(operands, in_batched, strict=True))
        ]
        return list(_batch(spec, axis_size)(*flat)), [True] * len(spec.results)

    return lane


def _batch(spec: _Spec, lanes: int) -> Any:

    @custom_vmap
    def batch(*operands: Array) -> list[Array]:
        return _ffi(spec, lanes)(*operands)

    @batch.def_vmap
    def rule(axis_size: int, in_batched: Sequence[bool], *operands: Array) -> tuple[list[Array], list[bool]]:
        merged = []
        for index, (x, batched) in enumerate(zip(operands, in_batched, strict=True)):
            row = x.shape[2:] if batched else x.shape[1:]
            if not batched and x.shape[0] == 1 and index not in spec.aliased:
                merged.append(x)
                continue
            whole = jnp.broadcast_to(
                x if batched else x[None], (axis_size, lanes, *row)
            )
            merged.append(whole.reshape((axis_size * lanes, *row)))
        outs = _batch(spec, axis_size * lanes)(*merged)
        return (
            [out.reshape((axis_size, lanes, *out.shape[1:])) for out in outs],
            [True] * len(spec.results),
        )

    return batch


def _ffi(spec: _Spec, lanes: int) -> Any:
    def ffi(*operands: Array) -> list[Array]:
        outs = jax.ffi.ffi_call(
            spec.target,
            [jax.ShapeDtypeStruct((lanes, *r.shape), r.dtype) for r in spec.results],
            input_output_aliases=dict(spec.aliases) or None,
        )(*operands, **dict(spec.attributes))
        return list(outs)

    return ffi
