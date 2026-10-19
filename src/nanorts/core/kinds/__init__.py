from typing import Final

from nanorts.core.kinds import attack, harvest, move, none, produce, return_
from nanorts.core.kinds.action_kind import (
    ActionKind,
    IsValid,
    LegalityGrid,
    Member,
    Request,
    RetainedActor,
)
from nanorts.model import ACTION_KIND_COUNT, POSITION_CLAIMING_KINDS

__all__ = [
    "KINDS",
    "ActionKind",
    "IsValid",
    "LegalityGrid",
    "Member",
    "Request",
    "RetainedActor",
]

KINDS: Final[tuple[ActionKind, ...]] = (
    none.KIND,
    move.KIND,
    harvest.KIND,
    return_.KIND,
    produce.KIND,
    attack.KIND,
)

if [kind.code for kind in KINDS] != list(range(ACTION_KIND_COUNT)):
    raise ImportError("KINDS must hold one kind per code, in code order")
if (
    tuple(kind.code for kind in KINDS if kind.claims_position)
    != POSITION_CLAIMING_KINDS
):
    raise ImportError("the kinds claiming a position disagree with the model")
