from __future__ import annotations

import dataclasses
import functools
import operator
from collections.abc import Iterable
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from nanorts.core import board, kinds, pending, residue, units
from nanorts.core.effect import CompletionRecord, Effect, Increment, Removal, Write
from nanorts.model import (
    ABSENT,
    KIND_NONE,
    PLAYER_COUNT,
    GameConfig,
    State,
    StepEvents,
    stored,
)

_RECORD_FIELDS_WRITTEN_FOR_EVERY_RECORD = frozenset(
    {"valid", "actor_birth", "actor_owner", "actor_unit_type"}
)


def apply(state: State, completed: Array, events: StepEvents, game: GameConfig) -> tuple[State, StepEvents]:
    retained_actors = _retained_actors(state)
    acting = completed & (state.pending_kind != KIND_NONE)
    member_order, member_count = pending.batch_order(state, acting, game)
    state = pending.clear(state, completed, game)

    def take_turn(position: Array, carry: tuple[State, StepEvents]) -> tuple[State, StepEvents]:
        return _take_turn(
            carry, completed, retained_actors, member_order[position], game
        )

    return residue.visit_in_order(member_count, take_turn, (state, events))


def _retained_actors(state: State) -> kinds.RetainedActor:
    return kinds.RetainedActor(
        kind=state.pending_kind,
        direction=state.pending_direction,
        target_cell=state.pending_target_cell,
        produced_type=state.pending_produced_type,
        birth=state.birth,
        owner=state.owner,
        unit_type=state.unit_type,
        cell=state.cell,
        carried=state.carried_resources,
    )


def _take_turn(carry: tuple[State, StepEvents], completed: Array, retained_actors: kinds.RetainedActor, slot: Array, game: GameConfig) -> tuple[State, StepEvents]:
    state, events = carry
    member = _member_at(state, completed, retained_actors, slot, game)
    effects = [kind.complete(member) for kind in kinds.KINDS]
    removal = _the_one_removal(effects)
    acted = _apply_writes(
        state,
        [
            *(write for effect in effects for write in effect.writes),
            Write("entity_at", state.cell[removal.slot], removal.when, ABSENT),
        ],
    )
    events = _record(
        events,
        member,
        [
            *(record for effect in effects for record in effect.records),
            _death_record(state, removal),
        ],
    )
    removed = removal.when & (member.all_slots == removal.slot)
    return units.remove(acted, removed, game), events


def _member_at(state: State, completed: Array, retained_actors: kinds.RetainedActor, slot: Array, game: GameConfig) -> kinds.Member:
    retained: kinds.RetainedActor = jax.tree.map(
        lambda leaf: leaf[slot].astype(jnp.int32), retained_actors
    )
    return kinds.Member(
        state=state,
        game=game,
        slot=slot,
        all_slots=jnp.arange(game.bucket.entity_capacity, dtype=jnp.int32),
        in_batch=completed[slot],
        retained=retained,
        actor_type=jnp.clip(retained.unit_type, 0, game.bucket.unit_type_capacity - 1),
        player=jnp.clip(retained.owner, 0, PLAYER_COUNT - 1),
        alive=state.active[slot] & (state.birth[slot] == retained.birth),
        destination=board.reserved_position(retained.cell, retained.direction, game),
        neighbour=board.neighbour(state, retained.cell, retained.direction, game),
    )


def _the_one_removal(effects: list[Effect]) -> Removal:
    removals = [effect.removal for effect in effects if effect.removal is not None]
    whens = [removal.when for removal in removals]
    removes = functools.reduce(operator.or_, whens) if whens else jnp.asarray(False)
    removed_slot = jnp.asarray(0, jnp.int32)
    for removal in removals:
        removed_slot = jnp.where(removal.when, removal.slot, removed_slot)
    return Removal(when=removes, slot=removed_slot)


def _death_record(state: State, removal: Removal) -> CompletionRecord:
    return CompletionRecord(
        "death",
        removal.when,
        {
            "slot": removal.slot,
            "birth": state.birth[removal.slot],
            "owner": state.owner[removal.slot],
            "unit_type": state.unit_type[removal.slot],
        },
    )


def _apply_writes(state: State, writes: Iterable[Write | Increment]) -> State:
    fields: dict[str, Array] = {}
    for write in _merge_writes_to_one_entry(writes):
        field = fields.get(write.field)
        if field is None:
            field = _flat_field(state, write.field)
        fields[write.field] = _written(field, write)
    return _store_fields(state, fields)


def _flat_field(state: State, path: str) -> Array:
    field = _read(state, path)
    if field.ndim <= 1:
        return field
    if path != "entity_at":
        raise ValueError(f"no Effect writes {path} by index")
    return field.reshape(-1)


def _written(field: Array, write: Write | Increment) -> Array:
    if isinstance(write, Increment):
        return _incremented(field, write)
    if write.at is None:
        return jnp.where(write.when, write.value, field)
    return residue.masked_set(field, write.at, write.when, write.value)


def _incremented(field: Array, increment: Increment) -> Array:
    hit = increment.when
    if increment.at is not None:
        hit = (jnp.arange(field.shape[0], dtype=jnp.int32) == increment.at) & hit
    return field + hit.astype(field.dtype)


def _store_fields(state: State, fields: dict[str, Array]) -> State:
    for path, field in fields.items():
        shape = _read(state, path).shape
        state = _replaced(
            state, path, field.reshape(shape) if field.shape != shape else field
        )
    return state


def _merge_writes_to_one_entry(writes: Iterable[Write | Increment]) -> list[Write | Increment]:
    merged: list[Write | Increment] = []
    for write in writes:
        earlier_position = _same_entry_write(merged, write)
        if earlier_position is None:
            merged.append(write)
            continue
        earlier = merged[earlier_position]
        assert isinstance(earlier, Write) and isinstance(write, Write)
        merged[earlier_position] = earlier._replace(
            when=earlier.when | write.when,
            value=jnp.where(write.when, write.value, earlier.value),
        )
    return merged


def _same_entry_write(merged: list[Write | Increment], write: Write | Increment) -> int | None:
    if not isinstance(write, Write):
        return None
    for position, earlier in enumerate(merged):
        if (
            isinstance(earlier, Write)
            and earlier.field == write.field
            and earlier.at is write.at
        ):
            return position
    return None


def _read(tree: Any, path: str) -> Array:
    return functools.reduce(getattr, path.split("."), tree)


def _replaced(tree: Any, path: str, value: Array) -> Any:
    name, _, rest = path.partition(".")
    if rest:
        value = _replaced(getattr(tree, name), rest, value)
    return stored(tree, **{name: value})


def _record(events: StepEvents, member: kinds.Member, records: Iterable[CompletionRecord]) -> StepEvents:
    updates: dict[str, Array] = {}
    for record in records:
        for name, value in _record_fields(record, member.retained).items():
            field = updates.get(name, getattr(events, name))
            updates[name] = residue.masked_set(field, member.slot, record.when, value)
    return dataclasses.replace(events, **updates)


def _record_fields(record: CompletionRecord, retained: kinds.RetainedActor) -> dict[str, Array]:
    if _RECORD_FIELDS_WRITTEN_FOR_EVERY_RECORD & record.facts.keys():
        raise ValueError(
            "the batch writes "
            f"{sorted(_RECORD_FIELDS_WRITTEN_FOR_EVERY_RECORD)} of every record"
        )
    facts = {
        "valid": record.when,
        "actor_birth": retained.birth,
        "actor_owner": retained.owner,
        "actor_unit_type": retained.unit_type,
        **record.facts,
    }
    return {f"{record.name}_{suffix}": value for suffix, value in facts.items()}
