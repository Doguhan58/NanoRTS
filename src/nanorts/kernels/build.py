from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import jax

_NVCC_DISTRIBUTION = "nvidia-cuda-nvcc"
_NVCC = "nvidia/cu13/bin/nvcc"

_FLAGS = (
    "-std=c++17",
    "-O3",
    "-cudart",
    "static",
    "-shared",
    "-Xcompiler",
    "-fPIC",
    "--threads",
    "0",
    "--no-device-link",
)

_FORMAT = "nanorts-kernels/2"

_LIBRARY = "libnanorts_kernels.so"
_RECORD = "build.json"

_CACHE_VARIABLE = "NANORTS_KERNEL_CACHE"

_BUILD_ID_HEADER = "nanorts_build_id.h"

_TIMEOUT_SECONDS = 600


class KernelBuildError(RuntimeError): ...


@dataclasses.dataclass(frozen=True)
class Toolchain:
    nvcc: Path
    host: Path
    versions: tuple[str, ...]
    architectures: tuple[str, ...]
    ptx: str


@dataclasses.dataclass(frozen=True)
class Build:
    library: Path
    digest: str
    architectures: tuple[str, ...]
    ptx: str


def _sources(package: Path) -> tuple[Path, ...]:
    found = [*package.glob("**/cuda/*.cu"), *package.glob("**/cuda/*.cuh")]
    return tuple(sorted(found, key=lambda path: path.relative_to(package).as_posix()))


def toolchain() -> Toolchain:
    try:
        distribution = importlib.metadata.distribution(_NVCC_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError as error:
        raise KernelBuildError(f"{_NVCC_DISTRIBUTION} is not installed") from error
    nvcc = Path(str(distribution.locate_file(_NVCC)))
    if not nvcc.is_file():
        raise KernelBuildError(f"{_NVCC_DISTRIBUTION} holds no {_NVCC}")
    host = shutil.which("g++")
    if host is None:
        raise KernelBuildError("no host g++ on PATH")
    return Toolchain(
        nvcc=nvcc,
        host=Path(host),
        versions=(_first_line(nvcc, "Cuda compilation tools"), _first_line(Path(host))),
        architectures=_listed(nvcc, "--list-gpu-code", "sm_"),
        ptx=_listed(nvcc, "--list-gpu-arch", "compute_")[-1],
    )


def _digest(package: Path, compilers: Toolchain) -> str:
    hashed = hashlib.sha256()
    for part in (
        _FORMAT,
        *_FLAGS,
        *_images(compilers),
        *compilers.versions,
        importlib.metadata.version("jaxlib"),
    ):
        hashed.update(part.encode() + b"\0")
    for source in _sources(package):
        hashed.update(source.relative_to(package).as_posix().encode() + b"\0")
        hashed.update(source.read_bytes() + b"\0")
    return hashed.hexdigest()


def cache() -> Path:
    named = os.environ.get(_CACHE_VARIABLE)
    if named:
        return Path(named)
    root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(root) / "nanorts" / "kernels"


def build(package: Path, compilers: Toolchain, directory: Path) -> Build:
    name = _digest(package, compilers)
    final = directory / name
    recorded = _recorded(final)
    if recorded is not None and recorded.digest == name:
        return recorded
    directory.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".build-", dir=directory))
    try:
        (work / _BUILD_ID_HEADER).write_text(f'#define NANORTS_BUILD_ID "{name}"\n')
        command = _nvcc_command(package, compilers, work)
        _run_nvcc(command)
        _write_record(work, final, name, command, compilers, package)
        _place(work, final, name)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return Build(
        library=final / _LIBRARY,
        digest=name,
        architectures=compilers.architectures,
        ptx=compilers.ptx,
    )


def _images(compilers: Toolchain) -> tuple[str, ...]:
    sass = (
        flag
        for name in compilers.architectures
        for flag in ("-gencode", f"arch=compute_{name.removeprefix('sm_')},code={name}")
    )
    return (*sass, "-gencode", f"arch={compilers.ptx},code={compilers.ptx}")


def _nvcc_command(package: Path, compilers: Toolchain, work: Path) -> list[str]:
    units = [path for path in _sources(package) if path.suffix == ".cu"]
    includes = sorted({str(path.parent) for path in _sources(package)})
    return [
        str(compilers.nvcc),
        *_FLAGS,
        *_images(compilers),
        "-ccbin",
        str(compilers.host),
        "-I",
        jax.ffi.include_dir(),
        *(argument for include in includes for argument in ("-I", include)),
        "-I",
        str(work),
        "-o",
        str(work / _LIBRARY),
        *(str(path) for path in units),
    ]


def _run_nvcc(command: list[str]) -> None:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise KernelBuildError(f"nvcc did not run: {error}") from error
    if completed.returncode != 0:
        tail = "\n".join((completed.stderr or completed.stdout).splitlines()[-20:])
        raise KernelBuildError(f"nvcc exited {completed.returncode}:\n{tail}")


def _write_record(work: Path, final: Path, name: str, command: list[str], compilers: Toolchain, package: Path) -> None:
    record = {
        "digest": name,
        "command": [part.replace(str(work), str(final)) for part in command],
        "versions": list(compilers.versions),
        "architectures": list(compilers.architectures),
        "ptx": compilers.ptx,
        "sources": [path.relative_to(package).as_posix() for path in _sources(package)],
    }
    (work / _RECORD).write_text(json.dumps(record, indent=2) + "\n")


def _place(work: Path, final: Path, name: str) -> None:
    if final.exists() and not _holds(final, name):
        shutil.rmtree(final, ignore_errors=True)
    try:
        work.rename(final)
    except OSError as error:
        if not _holds(final, name):
            raise KernelBuildError(f"cannot place the build: {error}") from error


def _holds(directory: Path, name: str) -> bool:
    recorded = _recorded(directory)
    return recorded is not None and recorded.digest == name


def _recorded(directory: Path) -> Build | None:
    try:
        recorded = json.loads((directory / _RECORD).read_text())
    except (OSError, ValueError):
        return None
    library = directory / _LIBRARY
    if not isinstance(recorded, dict) or not library.is_file():
        return None
    digest, architectures, ptx = (
        recorded.get(key) for key in ("digest", "architectures", "ptx")
    )
    if not (
        isinstance(digest, str)
        and isinstance(ptx, str)
        and re.fullmatch(r"compute_\d+", ptx)
        and isinstance(architectures, list)
        and all(
            isinstance(name, str) and re.fullmatch(r"sm_\d+", name)
            for name in architectures
        )
    ):
        return None
    return Build(
        library=library, digest=digest, architectures=tuple(architectures), ptx=ptx
    )


def _first_line(program: Path, starting: str = "") -> str:
    lines = _printed(program, "--version")
    chosen = [line for line in lines if line.startswith(starting)]
    if not chosen:
        raise KernelBuildError(f"{program} --version printed no version line")
    return chosen[0]


def _listed(nvcc: Path, option: str, prefix: str) -> tuple[str, ...]:
    named = [
        line for line in _printed(nvcc, option) if re.fullmatch(rf"{prefix}\d+", line)
    ]
    if not named:
        raise KernelBuildError(
            f"{nvcc} {option} listed no {prefix}<number> architecture"
        )
    return tuple(sorted(named, key=lambda name: int(name.removeprefix(prefix))))


def _printed(program: Path, option: str) -> list[str]:
    try:
        completed = subprocess.run(
            [str(program), option],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise KernelBuildError(f"{program} {option} did not run: {error}") from error
    if completed.returncode != 0:
        raise KernelBuildError(f"{program} {option} exited {completed.returncode}")
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]
