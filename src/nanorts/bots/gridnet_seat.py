from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, Final

import jax
import jax.numpy as jnp
from jax import Array

from nanorts.adapters import EnvironmentState, episode_root, gridnet, select
from nanorts.bots.bot import Bot, turn_key
from nanorts.model import PLAYER_COUNT, Commands, GameConfig, ModelError, Telemetry

_LEARNER: Final[int] = 0

_BOT: Final[int] = 1


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GridNetSeatState[Carry]:
    environment_state: EnvironmentState

    carry: Carry


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GridNetSeatStep[Carry]:
    seat_state: GridNetSeatState[Carry]

    observation: gridnet.GridNetObservation

    reward_components: Array

    reward_dones: Array

    terminated: Array

    truncated: Array

    done: Array

    winner: Array

    telemetry: Telemetry

    spawn_overflowed: Array

    final_observation: Array

    final_valid: Array


def reset_gridnet_seat[Carry](environment_root: Array, game: GameConfig, *, bot: Bot[Carry], layout: gridnet.GridNetLayout) -> tuple[GridNetSeatState[Carry], gridnet.GridNetObservation]:
    environment_state, observation = gridnet.reset(
        environment_root, game, layout=layout
    )
    seat_state = GridNetSeatState(
        environment_state=environment_state, carry=bot.init(game)
    )
    return seat_state, _learners(observation)


def step_gridnet_seat[Carry](seat_state: GridNetSeatState[Carry], actions: Array, game: GameConfig, *, bot: Bot[Carry], layout: gridnet.GridNetLayout) -> GridNetSeatStep[Carry]:
    carry, commands = gridnet_seat_commands(seat_state, game, bot=bot)
    result = gridnet.step_against(
        seat_state.environment_state, actions, commands, game, layout=layout
    )
    return GridNetSeatStep(
        seat_state=GridNetSeatState(
            environment_state=result.environment_state,
            carry=select(result.done, bot.init(game), carry),
        ),
        observation=_learners(result.observation),
        reward_components=result.reward_components[_LEARNER],
        reward_dones=result.reward_dones[_LEARNER],
        terminated=result.terminated,
        truncated=result.truncated,
        done=result.done,
        winner=result.winner,
        telemetry=result.telemetry,
        spawn_overflowed=result.spawn_overflowed,
        final_observation=result.final_observation[_LEARNER],
        final_valid=result.final_valid,
    )


def gridnet_seat_commands[Carry](seat_state: GridNetSeatState[Carry], game: GameConfig, *, bot: Bot[Carry]) -> tuple[Carry, Commands]:
    environment_state = seat_state.environment_state
    state = environment_state.core_state
    key = turn_key(episode_root(environment_state), state.time)
    return bot.act(key, seat_state.carry, state, game, _BOT)


@dataclasses.dataclass(frozen=True)
class GridNetGroup:
    bot: Bot[Any]
    lanes: int


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GridNetMixState:
    self_play: EnvironmentState | None

    seats: tuple[GridNetSeatState[Any], ...]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GridNetMixStep:
    mix_state: GridNetMixState

    observation: gridnet.GridNetObservation
    reward_components: Array
    reward_dones: Array
    terminated: Array
    truncated: Array
    done: Array
    winner: Array
    telemetry: Telemetry
    spawn_overflowed: Array
    final_observation: Array
    final_valid: Array


@dataclasses.dataclass(frozen=True)
class GridNetMix:
    self_play: int

    groups: Sequence[GridNetGroup]

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", tuple(self.groups))
        if not isinstance(self.self_play, int) or self.self_play < 0:
            raise ModelError(
                f"GridNetMix refuses self_play={self.self_play!r}: it is a count "
                "of self-play games"
            )
        for index, group in enumerate(self.groups):
            if not isinstance(group.lanes, int) or group.lanes < 1:
                raise ModelError(
                    f"GridNetMix refuses group {index}: lanes={group.lanes!r} is "
                    "not a positive int"
                )
            if not isinstance(group.bot, Bot):
                raise ModelError(
                    f"GridNetMix refuses group {index}: its bot is not a Bot"
                )
        if self.lanes == 0:
            raise ModelError("GridNetMix refuses a mix of no lanes")

    @property
    def lanes(self) -> int:
        return self.self_play + sum(group.lanes for group in self.groups)

    @property
    def rows(self) -> int:
        return PLAYER_COUNT * self.self_play + sum(group.lanes for group in self.groups)

    def slices(self) -> tuple[slice, ...]:
        sizes = [PLAYER_COUNT * self.self_play] if self.self_play else []
        return _bounds(sizes + [group.lanes for group in self.groups])

    def reset(self, roots: Array, game: GameConfig, *, layout: gridnet.GridNetLayout) -> tuple[GridNetMixState, gridnet.GridNetObservation]:
        parts = _split(roots, self._lane_counts())
        rows = []
        self_play = None
        if self.self_play:
            self_play, observation = jax.vmap(
                lambda root: gridnet.reset(root, game, layout=layout)
            )(parts.pop(0))
            rows.append(_game_rows(observation))
        seats = []
        for group, lane_roots in zip(self.groups, parts, strict=True):
            seat_state, observation = jax.vmap(
                lambda root, bot=group.bot: reset_gridnet_seat(
                    root, game, bot=bot, layout=layout
                )
            )(lane_roots)
            seats.append(seat_state)
            rows.append(observation)
        return GridNetMixState(self_play=self_play, seats=tuple(seats)), _joined(rows)

    def step(self, mix_state: GridNetMixState, actions: Array, game: GameConfig, *, layout: gridnet.GridNetLayout) -> GridNetMixStep:
        parts = _split(actions, self._row_counts())
        rows = []
        self_play = None
        if self.self_play:
            self_play, played = _step_games(
                mix_state.self_play, parts.pop(0), game, layout
            )
            rows.append(played)
        seats = []
        for group, seat_state, grids in zip(
            self.groups, mix_state.seats, parts, strict=True
        ):
            seated, played = _step_seats(group.bot, seat_state, grids, game, layout)
            seats.append(seated)
            rows.append(played)
        return GridNetMixStep(
            mix_state=GridNetMixState(self_play=self_play, seats=tuple(seats)),
            **_joined(rows),
        )

    def _lane_counts(self) -> list[int]:
        sizes = [self.self_play] if self.self_play else []
        return sizes + [group.lanes for group in self.groups]

    def _row_counts(self) -> list[int]:
        return [bound.stop - bound.start for bound in self.slices()]


def _step_games(carries: EnvironmentState | None, rows: Array, game: GameConfig, layout: gridnet.GridNetLayout) -> tuple[EnvironmentState, dict[str, Any]]:
    grids = rows.reshape(-1, PLAYER_COUNT, *rows.shape[1:])
    stepped = jax.vmap(
        lambda carry, grid: gridnet.step(carry, grid, game, layout=layout)
    )(carries, grids)
    return stepped.environment_state, _self_play_rows(stepped)


def _step_seats(bot: Bot[Any], seat_states: GridNetSeatState[Any], grids: Array, game: GameConfig, layout: gridnet.GridNetLayout) -> tuple[GridNetSeatState[Any], dict[str, Any]]:
    stepped = jax.vmap(
        lambda seat_state, grid: step_gridnet_seat(
            seat_state, grid, game, bot=bot, layout=layout
        )
    )(seat_states, grids)
    return stepped.seat_state, _seat_rows(stepped)


def _learners(observation: gridnet.GridNetObservation) -> gridnet.GridNetObservation:
    return jax.tree.map(lambda leaf: leaf[_LEARNER], observation)


def _game_rows[Tree](per_player: Tree) -> Tree:
    return jax.tree.map(
        lambda leaf: leaf.reshape(leaf.shape[0] * PLAYER_COUNT, *leaf.shape[2:]),
        per_player,
    )


def _lane_rows[Tree](per_lane: Tree) -> Tree:
    return jax.tree.map(lambda leaf: jnp.repeat(leaf, PLAYER_COUNT, axis=0), per_lane)


def _self_play_rows(stepped: gridnet.GridNetStep) -> dict[str, Any]:
    return {
        "observation": _game_rows(stepped.observation),
        "reward_components": _game_rows(stepped.reward_components),
        "reward_dones": _game_rows(stepped.reward_dones),
        "terminated": _lane_rows(stepped.terminated),
        "truncated": _lane_rows(stepped.truncated),
        "done": _lane_rows(stepped.done),
        "winner": _lane_rows(stepped.winner),
        "telemetry": _lane_rows(stepped.telemetry),
        "spawn_overflowed": _lane_rows(stepped.spawn_overflowed),
        "final_observation": _game_rows(stepped.final_observation),
        "final_valid": _lane_rows(stepped.final_valid),
    }


def _seat_rows(stepped: GridNetSeatStep[Any]) -> dict[str, Any]:
    return {
        field.name: getattr(stepped, field.name)
        for field in dataclasses.fields(GridNetSeatStep)
        if field.name != "seat_state"
    }


def _bounds(sizes: list[int]) -> tuple[slice, ...]:
    stops = [sum(sizes[: index + 1]) for index in range(len(sizes))]
    return tuple(
        slice(stop - size, stop) for stop, size in zip(stops, sizes, strict=True)
    )


def _split(batch: Array, sizes: list[int]) -> list[Array]:
    return [batch[bound] for bound in _bounds(sizes)]


def _joined[Tree](parts: list[Tree]) -> Tree:
    return jax.tree.map(lambda *leaves: jnp.concatenate(leaves), *parts)
