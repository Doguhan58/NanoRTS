from __future__ import annotations

import dataclasses
import functools
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from nanorts.model import ModelError


@dataclasses.dataclass(frozen=True)
class Group:
    environment: Any
    lanes: int


class GroupMix:
    def __init__(self, groups: Sequence[Group]) -> None:
        self.groups = tuple(groups)
        if not self.groups:
            raise ModelError("GroupMix refuses an empty group list")
        for index, group in enumerate(self.groups):
            if not isinstance(group.lanes, int) or group.lanes < 1:
                raise ModelError(
                    f"GroupMix refuses group {index}: lanes={group.lanes!r} is "
                    "not a positive int"
                )
        _refuse_unlike_rows(self.groups)

    @property
    def lanes(self) -> int:
        return sum(group.lanes for group in self.groups)

    @property
    def rows(self) -> int:
        return sum(_rows_of(group) for group in self.groups)

    def slices(self) -> tuple[slice, ...]:
        return tuple(_bounds(self._row_counts()))

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, roots: Array) -> tuple[Any, tuple[Any, ...]]:
        observations, states = zip(
            *(
                _reset_group(group, lane_roots)
                for group, lane_roots in zip(
                    self.groups, _split(roots, self._lane_counts()), strict=True
                )
            ),
            strict=True,
        )
        return _concatenated(list(observations)), tuple(states)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, key: Array, states: tuple[Any, ...], actions: Array) -> tuple[Any, tuple[Any, ...], Array, Array, dict[str, Any]]:
        observations, next_states, rewards, dones, infos = zip(
            *(
                _step_group(group, key, state, rows)
                for group, state, rows in zip(
                    self.groups,
                    states,
                    _split(actions, self._row_counts()),
                    strict=True,
                )
            ),
            strict=True,
        )
        return (
            _concatenated(list(observations)),
            tuple(next_states),
            _concatenated(list(rewards)),
            _concatenated(list(dones)),
            _concatenated(list(infos)),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, states: tuple[Any, ...]) -> Array:
        return _concatenated(
            [
                _mask_group(group, state)
                for group, state in zip(self.groups, states, strict=True)
            ]
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_slot_cells(self, states: tuple[Any, ...]) -> Array:
        return _concatenated(
            [
                _slot_cells_group(group, state)
                for group, state in zip(self.groups, states, strict=True)
            ]
        )

    def _lane_counts(self) -> list[int]:
        return [group.lanes for group in self.groups]

    def _row_counts(self) -> list[int]:
        return [_rows_of(group) for group in self.groups]


def _reset_group(group: Group, roots: Array) -> tuple[Any, Any]:
    observation, state = jax.vmap(group.environment.reset)(roots)
    return _agent_major(observation, group), state


def _step_group(group: Group, key: Array, state: Any, rows: Array) -> tuple[Any, Any, Array, Array, dict[str, Any]]:
    agents = group.environment.agents
    actions = dict(zip(agents, jnp.split(rows, len(agents)), strict=True))
    observation, following, rewards, dones, infos = jax.vmap(
        group.environment.step, in_axes=(None, 0, 0)
    )(key, state, actions)
    return (
        _agent_major(observation, group),
        following,
        _agent_major(rewards, group),
        _agent_major({agent: dones[agent] for agent in agents}, group),
        _info_rows(infos, group),
    )


def _mask_group(group: Group, state: Any) -> Array:
    return _agent_major(jax.vmap(group.environment.get_avail_actions)(state), group)


def _slot_cells_group(group: Group, state: Any) -> Array:
    return _agent_major(jax.vmap(group.environment.get_slot_cells)(state), group)


def _rows_of(group: Group) -> int:
    return group.lanes * len(group.environment.agents)


def _bounds(sizes: list[int]) -> list[slice]:
    stops = [sum(sizes[: index + 1]) for index in range(len(sizes))]
    return [slice(stop - size, stop) for stop, size in zip(stops, sizes, strict=True)]


def _split(batch: Any, sizes: list[int]) -> list[Any]:
    return [_window(batch, bound) for bound in _bounds(sizes)]


def _window(batch: Any, bound: slice) -> Any:
    return jax.tree.map(lambda leaf: leaf[bound], batch)


def _concatenated(parts: list[Any]) -> Any:
    return jax.tree.map(lambda *leaves: jnp.concatenate(leaves), *parts)


def _agent_major(per_agent: dict[str, Any], group: Group) -> Any:
    return _concatenated([per_agent[agent] for agent in group.environment.agents])


def _info_rows(infos: dict[str, Any], group: Group) -> dict[str, Any]:
    agents = group.environment.agents
    return {
        name: (
            _agent_major(value, group)
            if isinstance(value, dict) and list(value) == list(agents)
            else _concatenated([value] * len(agents))
        )
        for name, value in infos.items()
    }


def _signature(tree: Any) -> tuple[Any, ...]:
    leaves, structure = jax.tree.flatten(tree)
    return (
        structure,
        tuple((leaf.shape[1:], leaf.dtype) for leaf in leaves),
    )


def _refuse_unlike_rows(groups: tuple[Group, ...]) -> None:
    signatures = [_row_signature(group) for group in groups]
    for index, signature in enumerate(signatures[1:], start=1):
        if signature != signatures[0]:
            raise ModelError(
                f"GroupMix refuses group {index}: its rows are not group 0's - "
                "the groups' environments must publish equal observation and "
                "action spaces and equal infos, so equal final_observation and "
                "reward_components flags"
            )


def _row_signature(group: Group) -> tuple[Any, ...]:
    environment = group.environment

    def rows(roots: Array, actions: Array) -> tuple[Any, ...]:
        observation, state = _reset_group(group, roots)
        stepped, _, rewards, dones, infos = _step_group(group, roots[0], state, actions)
        return observation, stepped, rewards, dones, infos, _mask_group(group, state)

    shape = environment.action_space(environment.agents[0]).shape
    return _signature(
        jax.eval_shape(
            rows,
            jax.ShapeDtypeStruct((1,), jax.random.key(0).dtype),
            jax.ShapeDtypeStruct((len(environment.agents), *shape), jnp.int32),
        )
    )
