from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any

import jax
from jax import lax

from nanorts.kernels import library


def dispatch[T](*operands: Any, kernel: Callable[..., T], program: Callable[..., T]) -> T:
    if not selected():
        return program(*operands)
    leaves, tree = jax.tree.flatten(operands)
    traced = [isinstance(leaf, jax.core.Tracer) for leaf in leaves]
    if not any(traced):
        return _compiled(kernel, program)(*operands)

    def bound(stage: Callable[..., T]) -> Callable[..., T]:
        def call(*dynamic: Any) -> T:
            given = iter(dynamic)
            return stage(
                *jax.tree.unflatten(
                    tree,
                    [
                        next(given) if t else leaf
                        for leaf, t in zip(leaves, traced, strict=True)
                    ],
                )
            )

        return call

    return lax.platform_dependent(
        *(leaf for leaf, t in zip(leaves, traced, strict=True) if t),
        cpu=bound(program),
        cuda=bound(kernel),
        default=bound(program),
    )


@functools.lru_cache(maxsize=32)
def _compiled(kernel: Callable[..., Any], program: Callable[..., Any]) -> Any:
    return jax.jit(
        lambda *operands: dispatch(*operands, kernel=kernel, program=program)
    )


def selected() -> bool:
    if not library.SELECTED:
        return False
    reason = _refusal()
    if reason:
        _warn_once(reason)
        return False
    return True


def devices_within(capabilities: frozenset[str]) -> bool:
    return _capabilities() <= capabilities


@functools.cache
def _capabilities() -> frozenset[str]:
    try:
        devices = jax.devices("cuda")
    except RuntimeError:
        return frozenset()
    return frozenset(
        str(getattr(device, "compute_capability", "unknown")) for device in devices
    )


@functools.cache
def _refusal() -> str:
    loaded = library.LIBRARY
    if loaded is None:
        return library.REASON
    try:
        devices = jax.devices("cuda")
    except RuntimeError:
        return ""
    capabilities = {
        str(getattr(device, "compute_capability", "unknown")) for device in devices
    }
    unserved = sorted(
        capability
        for capability in capabilities
        if not _served(capability, loaded.architectures, loaded.ptx)
    )
    if unserved:
        return (
            f"the kernel library holds no code for compute capability {unserved}; "
            f"it carries SASS for {list(loaded.architectures)} and PTX for "
            f"{loaded.ptx}"
        )
    return library.self_check([device.local_hardware_id for device in devices])


def _served(capability: str, architectures: tuple[str, ...], ptx: str) -> bool:
    try:
        major, minor = (int(part) for part in capability.split("."))
    except ValueError:
        return False
    sass = any(
        image_major == major and image_minor <= minor
        for image_major, image_minor in map(_version, architectures)
    )
    return sass or _version(ptx) <= (major, minor)


def _version(architecture: str) -> tuple[int, int]:
    return divmod(int(architecture.rpartition("_")[2]), 10)


@functools.cache
def _warn_once(reason: str) -> None:
    try:
        jax.devices("cuda")
    except RuntimeError:
        return
    warnings.warn(
        f"nanoRTS runs its JAX programs: {reason}; set {library.SWITCH}=0 to "
        "skip the kernels",
        UserWarning,
        stacklevel=4,
    )
