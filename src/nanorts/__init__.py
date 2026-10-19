import importlib
import types
from collections.abc import Mapping
from typing import Any

__all__ = [
    "DOUBLE_BUFFER_COMPILER_OPTIONS",
    "RECOMMENDED_COMPILER_OPTIONS",
    "make",
    "map_name_to_scenario",
    "scenario_names",
]

RECOMMENDED_COMPILER_OPTIONS: Mapping[str, str] = types.MappingProxyType(
    {
        "xla_gpu_enable_command_buffer": (
            "FUSION,WHILE,CONDITIONAL,CUSTOM_CALL,DYNAMIC_SLICE_FUSION,"
            "CUBLAS,CUBLASLT,CUDNN,CONVOLUTION"
        ),
        "xla_gpu_memory_limit_slop_factor": "300",
    }
)

DOUBLE_BUFFER_COMPILER_OPTIONS: Mapping[str, str] = types.MappingProxyType(
    {"xla_gpu_enable_while_loop_unrolling": "WHILE_LOOP_UNROLLING_DOUBLE_BUFFER"}
)

_NAMES_BOUND_ON_FIRST_USE = ("make", "map_name_to_scenario", "scenario_names")


def __getattr__(name: str) -> Any:
    if name in _NAMES_BOUND_ON_FIRST_USE:
        return getattr(importlib.import_module("nanorts.adapters.jaxmarl"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
