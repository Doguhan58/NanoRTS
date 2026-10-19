import importlib
from typing import Any

from nanorts.adapters import gridnet
from nanorts.adapters.lifecycle import (
    EnvironmentState,
    advance,
    episode_root,
    episode_start,
    initial,
    select,
)

__all__ = [
    "EnvironmentState",
    "advance",
    "episode_root",
    "episode_start",
    "gridnet",
    "initial",
    "jaxmarl",
    "select",
]


def __getattr__(name: str) -> Any:
    if name == "jaxmarl":
        return importlib.import_module("nanorts.adapters.jaxmarl")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
