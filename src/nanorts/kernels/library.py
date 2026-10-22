from __future__ import annotations

import ctypes
import dataclasses
import os
from pathlib import Path

import jax

from nanorts.kernels import build

SWITCH = "NANORTS_KERNELS"

_PREFIX = "nanorts_"


@dataclasses.dataclass(frozen=True)
class _Library:
    path: Path
    digest: str
    targets: tuple[str, ...]
    architectures: tuple[str, ...]
    ptx: str
    handle: ctypes.CDLL


def _load() -> tuple[_Library | None, str]:
    package = Path(__file__).resolve().parents[1]
    try:
        built = build.build(package, build.toolchain(), build.cache())
    except (build.KernelBuildError, OSError) as error:
        return None, f"the kernel library was not built: {error}"
    try:
        handle = ctypes.CDLL(str(built.library))
        handle.nanorts_build_id.restype = ctypes.c_char_p
        handle.nanorts_targets.restype = ctypes.c_char_p
        handle.nanorts_self_check.restype = ctypes.c_char_p
        handle.nanorts_self_check.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int64,
        ]
        handle.nanorts_shared_opt_in.restype = ctypes.c_int64
        handle.nanorts_shared_opt_in.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int64,
        ]
        digest = handle.nanorts_build_id().decode()
        targets = tuple(
            sorted(filter(None, handle.nanorts_targets().decode().split(",")))
        )
    except (OSError, AttributeError) as error:
        return None, f"the kernel library at {built.library} did not load: {error}"
    if digest != built.digest:
        return None, f"{built.library} carries digest {digest}, not {built.digest}"
    strays = [name for name in targets if not name.startswith(_PREFIX)]
    if strays:
        return None, f"the kernel library defines unprefixed targets {strays}"
    for name in targets:
        jax.ffi.register_ffi_target(
            name, jax.ffi.pycapsule(getattr(handle, name)), platform="CUDA"
        )
    return _Library(
        path=built.library,
        digest=digest,
        targets=targets,
        architectures=built.architectures,
        ptx=built.ptx,
        handle=handle,
    ), ""


SELECTED: bool = os.environ.get(SWITCH) != "0"

LIBRARY: _Library | None
REASON: str
LIBRARY, REASON = _load() if SELECTED else (None, f"{SWITCH} is 0")


def build_id() -> str | None:
    return None if LIBRARY is None else LIBRARY.digest


def targets() -> tuple[str, ...]:
    return () if LIBRARY is None else LIBRARY.targets


def shared_opt_in(devices: list[int]) -> int:
    if LIBRARY is None:
        return 0
    ordinals = (ctypes.c_int32 * len(devices))(*devices)
    return LIBRARY.handle.nanorts_shared_opt_in(ordinals, len(devices))


def self_check(devices: list[int]) -> str:
    if LIBRARY is None:
        return REASON
    ordinals = (ctypes.c_int32 * len(devices))(*devices)
    failure = LIBRARY.handle.nanorts_self_check(ordinals, len(devices)).decode()
    return f"the kernel library's self-check failed on {failure}" if failure else ""
