from __future__ import annotations

import os
import re
from collections.abc import Sequence
from pathlib import Path
from xml.etree import ElementTree

from nanorts.content.acceptance import (
    check_dimensions,
    check_document,
    check_players,
    check_terrain_digits,
    check_terrain_payload,
)
from nanorts.content.document import ContentDocument, ContentError, UnitRecord
from nanorts.content.normalize import check_supplied_bucket, derive_bucket, game_config
from nanorts.model import (
    TABLES,
    BucketSpec,
    ConflictStrategy,
    GameConfig,
    UnitType,
    UnitTypeTableVersion,
    check_state_invariants,
    validate,
)

_INTEGER = re.compile(r"[+-]?[0-9]+")


def _document_from_xml(text: str) -> ContentDocument:
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as error:
        raise ContentError(
            f"clause 1: the document is not well formed XML: {error}", clause=1
        ) from error

    width = _attribute_integer(root, "width", clause=1)
    height = _attribute_integer(root, "height", clause=1)
    check_dimensions(width, height)
    terrain = _terrain_from_xml(root, width, height)
    players = _players_from_xml(root)
    units = tuple(
        _unit_from_xml(child) for child in _required_child(root, "units", clause=5)
    )
    return ContentDocument(
        width=width, height=height, terrain=terrain, players=players, units=units
    )


def _required_child(root: ElementTree.Element, name: str, *, clause: int) -> ElementTree.Element:
    child = root.find(name)
    if child is None:
        raise ContentError(
            f"clause {clause}: the document has no <{name}> element", clause=clause
        )
    return child


def _terrain_from_xml(root: ElementTree.Element, width: int, height: int) -> str:
    terrain = _required_child(root, "terrain", clause=2).text or ""
    check_terrain_payload(terrain, width, height)
    check_terrain_digits(terrain)
    return terrain


def _players_from_xml(root: ElementTree.Element) -> tuple[tuple[int, int], ...]:
    players = tuple(
        (
            _attribute_integer(child, "ID", clause=4),
            _attribute_integer(child, "resources", clause=4),
        )
        for child in _required_child(root, "players", clause=4)
    )
    check_players(players)
    return players


def _unit_from_xml(element: ElementTree.Element) -> UnitRecord:
    return UnitRecord(
        type=_attribute(element, "type", clause=5),
        id=_attribute_integer(element, "ID", clause=5),
        player=_attribute_integer(element, "player", clause=5),
        x=_attribute_integer(element, "x", clause=5),
        y=_attribute_integer(element, "y", clause=5),
        hitpoints=_attribute_integer(element, "hitpoints", clause=5),
        resources=_attribute_integer(element, "resources", clause=5),
    )


def _attribute(element: ElementTree.Element, name: str, *, clause: int) -> str:
    text = element.get(name)
    if text is None:
        raise ContentError(
            f"clause {clause}: <{element.tag}> carries no {name!r} attribute",
            clause=clause,
        )
    return text


def _attribute_integer(element: ElementTree.Element, name: str, *, clause: int) -> int:
    text = _attribute(element, name, clause=clause)
    if _INTEGER.fullmatch(text) is None:
        raise ContentError(
            f"clause {clause}: <{element.tag}> attribute {name}={text!r} is not "
            "an integer",
            clause=clause,
        )
    return int(text)


def load_game(path: str | os.PathLike[str], *, utt: UnitTypeTableVersion = UnitTypeTableVersion.ORIGINAL, conflict_strategy: ConflictStrategy = ConflictStrategy.CANCEL_BOTH, max_cycles: int = 5000, bucket: BucketSpec | None = None) -> GameConfig:
    table = _supported_table(utt, conflict_strategy)
    text = Path(path).read_text(encoding="utf-8")
    return _assemble_game(
        _document_from_xml(text), table, utt, conflict_strategy, max_cycles, bucket
    )


def build_game(*, width: int, height: int, terrain: str, players: Sequence[int], units: Sequence[UnitRecord], utt: UnitTypeTableVersion = UnitTypeTableVersion.ORIGINAL, conflict_strategy: ConflictStrategy = ConflictStrategy.CANCEL_BOTH, max_cycles: int = 5000, bucket: BucketSpec | None = None) -> GameConfig:
    table = _supported_table(utt, conflict_strategy)
    document = ContentDocument(
        width=width,
        height=height,
        terrain=terrain,
        players=tuple(enumerate(players)),
        units=tuple(units),
    )
    return _assemble_game(document, table, utt, conflict_strategy, max_cycles, bucket)


def _supported_table(utt: UnitTypeTableVersion, conflict_strategy: ConflictStrategy) -> tuple[UnitType, ...]:
    table = TABLES.get(utt)
    if table is None:
        raise ContentError(
            f"unit type table {utt!r} is outside the supported release matrix, "
            f"which is {[member.value for member in UnitTypeTableVersion]}"
        )
    if conflict_strategy not in tuple(ConflictStrategy):
        raise ContentError(
            f"conflict strategy {conflict_strategy!r} is outside the supported "
            f"release matrix, which is "
            f"{[member.value for member in ConflictStrategy]}"
        )
    return table


def _assemble_game(document: ContentDocument, table: tuple[UnitType, ...], utt: UnitTypeTableVersion, conflict_strategy: ConflictStrategy, max_cycles: int, supplied: BucketSpec | None) -> GameConfig:
    check_document(document, table)
    bucket = derive_bucket(document, table, utt, conflict_strategy)
    if supplied is not None:
        check_supplied_bucket(bucket, supplied, document)
        bucket = supplied
    config = game_config(document, table, bucket, max_cycles)
    validate(config, bucket)
    check_state_invariants(config.initial_state, bucket)
    return config
