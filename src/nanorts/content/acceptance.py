from typing import Final

from nanorts.content.document import ContentDocument, ContentError, UnitRecord
from nanorts.model import OWNER_NEUTRAL, PLAYER_COUNT, UnitType

_INT32_MIN: Final[int] = -(2**31)
_INT32_MAX: Final[int] = 2**31 - 1


def _check_int32(value: int, subject: str, *, clause: int) -> None:
    if not _INT32_MIN <= value <= _INT32_MAX:
        raise ContentError(
            f"clause {clause}: {subject} is {value}, outside the 32-bit signed range",
            clause=clause,
        )


def check_dimensions(width: int, height: int) -> None:
    for name, value in (("width", width), ("height", height)):
        _check_int32(value, name, clause=1)
        if value <= 0:
            raise ContentError(
                f"clause 1: {name} is {value}, expected a positive integer",
                clause=1,
            )


def check_terrain_payload(terrain: str, width: int, height: int) -> None:
    cells = width * height
    if len(terrain) != cells:
        raise ContentError(
            f"clause 2: the <terrain> payload is {len(terrain)} characters "
            f"against a {width}x{height} field of {cells} cells",
            clause=2,
        )
    for index, character in enumerate(terrain):
        if character not in "0123456789":
            raise ContentError(
                f"clause 2: the <terrain> payload holds {character!r} at index "
                f"{index}, and the payload is digits only",
                clause=2,
            )


def check_terrain_digits(terrain: str) -> None:
    for index, character in enumerate(terrain):
        if character not in "01":
            raise ContentError(
                f"clause 3: the <terrain> payload holds the digit {character} "
                f"at index {index}, and terrain is binary",
                clause=3,
            )


def check_players(players: tuple[tuple[int, int], ...]) -> None:
    if len(players) != PLAYER_COUNT:
        raise ContentError(
            f"clause 4: the document declares {len(players)} player records, "
            f"expected exactly {PLAYER_COUNT}",
            clause=4,
        )
    for expected, (declared, resources) in enumerate(players):
        if declared != expected:
            raise ContentError(
                f"clause 4: player record {expected} declares ID {declared}, "
                "and the records are positional",
                clause=4,
            )
        _check_int32(resources, f"player record {expected}'s resources", clause=4)


def _check_units(document: ContentDocument, table: tuple[UnitType, ...]) -> None:
    type_names = {unit.name for unit in table}
    seen_ids: set[int] = set()
    occupied: dict[tuple[int, int], UnitRecord] = {}
    declared_players = {declared for declared, _ in document.players}
    for position, unit in enumerate(document.units):
        _check_unit_fields_fit_int32(unit, position)
        _check_known_type(unit, type_names)
        _check_unique_id(unit, seen_ids)
        _check_on_board(unit, document)
        _check_unoccupied(unit, occupied)
        _check_owner_declared(unit, declared_players)


def _check_unit_fields_fit_int32(unit: UnitRecord, position: int) -> None:
    for name, value in (
        ("ID", unit.id),
        ("player", unit.player),
        ("x", unit.x),
        ("y", unit.y),
        ("hitpoints", unit.hitpoints),
        ("resources", unit.resources),
    ):
        _check_int32(value, f"unit {position}'s {name}", clause=5)


def _check_known_type(unit: UnitRecord, type_names: set[str]) -> None:
    if unit.type not in type_names:
        raise ContentError(
            f"clause 5: unit ID {unit.id} names the type {unit.type!r}, "
            f"which the supported unit type table does not hold",
            clause=5,
        )


def _check_unique_id(unit: UnitRecord, seen_ids: set[int]) -> None:
    if unit.id in seen_ids:
        raise ContentError(
            f"clause 5: unit ID {unit.id} is declared more than once",
            clause=5,
        )
    seen_ids.add(unit.id)


def _check_on_board(unit: UnitRecord, document: ContentDocument) -> None:
    if not (0 <= unit.x < document.width and 0 <= unit.y < document.height):
        raise ContentError(
            f"clause 5: unit ID {unit.id} sits at ({unit.x}, {unit.y}), "
            f"outside the {document.width}x{document.height} field",
            clause=5,
        )


def _check_unoccupied(unit: UnitRecord, occupied: dict[tuple[int, int], UnitRecord]) -> None:
    if (unit.x, unit.y) in occupied:
        other = occupied[unit.x, unit.y]
        raise ContentError(
            f"clause 5: unit ID {unit.id} sits at ({unit.x}, {unit.y}), "
            f"which unit ID {other.id} already occupies",
            clause=5,
        )
    occupied[unit.x, unit.y] = unit


def _check_owner_declared(unit: UnitRecord, declared_players: set[int]) -> None:
    if unit.player != OWNER_NEUTRAL and unit.player not in declared_players:
        raise ContentError(
            f"clause 5: unit ID {unit.id} names player {unit.player}, "
            "which no player record declares",
            clause=5,
        )


_HIT_POINTS: Final[range] = range(2**15)


def _check_hit_points(document: ContentDocument) -> None:
    for unit in document.units:
        if unit.hitpoints not in _HIT_POINTS:
            raise ContentError(
                f"clause 6: unit ID {unit.id} declares {unit.hitpoints} hit "
                f"points, outside [{_HIT_POINTS.start}, {_HIT_POINTS.stop - 1}]",
                clause=6,
            )


def check_document(document: ContentDocument, table: tuple[UnitType, ...]) -> None:
    check_dimensions(document.width, document.height)
    check_terrain_payload(document.terrain, document.width, document.height)
    check_terrain_digits(document.terrain)
    check_players(document.players)
    _check_units(document, table)
    _check_hit_points(document)
