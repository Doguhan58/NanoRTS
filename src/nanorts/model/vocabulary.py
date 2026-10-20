import enum
from typing import Final

ABSENT: Final[int] = -1

OWNER_NEUTRAL: Final[int] = ABSENT


PLAYER_COUNT: Final[int] = 2

ATTACK_TARGET_OWNER_COUNT: Final[int] = 3


KIND_ABSENT: Final[int] = ABSENT

KIND_NONE: Final[int] = 0
KIND_MOVE: Final[int] = 1
KIND_HARVEST: Final[int] = 2
KIND_RETURN: Final[int] = 3
KIND_PRODUCE: Final[int] = 4
KIND_ATTACK: Final[int] = 5

ACTION_KIND_COUNT: Final[int] = len(
    (KIND_NONE, KIND_MOVE, KIND_HARVEST, KIND_RETURN, KIND_PRODUCE, KIND_ATTACK)
)

POSITION_CLAIMING_KINDS: Final[tuple[int, ...]] = (KIND_MOVE, KIND_PRODUCE)


DIRECTION_UP: Final[int] = 0
DIRECTION_RIGHT: Final[int] = 1
DIRECTION_DOWN: Final[int] = 2
DIRECTION_LEFT: Final[int] = 3

DIRECTION_COUNT: Final[int] = 4

DIRECTION_OFFSETS: Final[tuple[tuple[int, int], ...]] = (
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
)


class UnitTypeTableVersion(enum.StrEnum):
    ORIGINAL = "VERSION_ORIGINAL"
    ORIGINAL_FINETUNED = "VERSION_ORIGINAL_FINETUNED"


class ConflictStrategy(enum.StrEnum):
    CANCEL_BOTH = "CANCEL_BOTH"
    CANCEL_RANDOM = "CANCEL_RANDOM"
    CANCEL_ALTERNATING = "CANCEL_ALTERNATING"
