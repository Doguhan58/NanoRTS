from __future__ import annotations

import dataclasses
import functools
from typing import Any, Final

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike
from jaxmarl.environments import spaces
from jaxmarl.environments.multi_agent_env import MultiAgentEnv

from nanorts.adapters import EnvironmentState, episode_root, jaxmarl, select
from nanorts.bots.bot import Bot, turn_key
from nanorts.model import PLAYER_COUNT, Commands, GameConfig, ModelError, Telemetry

LEARNER: Final[str] = "learner"

SEAT_RULES: Final[tuple[str, ...]] = ("player_0", "player_1", "drawn")

_SEAT_STREAM: Final[int] = 2**32 - 2


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SeatState[Carry]:
    environment_state: EnvironmentState

    carry: Carry

    seat: Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SeatStep[Carry]:
    seat_state: SeatState[Carry]

    observation: jaxmarl.Observation

    reward: Array

    reward_components: Array | None

    terminated: Array

    truncated: Array

    done: Array

    winner: Array

    telemetry: Telemetry

    spawn_overflowed: Array

    final_observation: jaxmarl.Observation

    final_valid: Array


def reset_seat[Carry](environment_root: Array, game: GameConfig, *, bot: Bot[Carry], layout: jaxmarl.JaxMARLLayout, assets: Any = None, seat: str | int | Array = "player_0") -> tuple[SeatState[Carry], jaxmarl.Observation]:
    environment_state, observations = jaxmarl.reset(
        environment_root, game, layout=layout, assets=assets
    )
    learner = _episode_seat(seat, episode_root(environment_state))
    seat_state = SeatState(
        environment_state=environment_state, carry=bot.init(game), seat=learner
    )
    return seat_state, _learners(observations, seat, learner)


def step_seat[Carry](seat_state: SeatState[Carry], actions: Array, game: GameConfig, *, bot: Bot[Carry], layout: jaxmarl.JaxMARLLayout, assets: Any = None, seat: str | int | Array = "player_0") -> SeatStep[Carry]:
    environment_state = seat_state.environment_state
    carry, commands = seat_commands(seat_state, game, bot=bot, seat=seat)
    learner = _current(seat, seat_state.seat)
    result = jaxmarl.step_against(
        environment_state,
        learner,
        actions,
        commands,
        game,
        layout=layout,
        assets=assets,
    )
    following = jnp.where(
        result.done,
        _episode_seat(seat, episode_root(result.environment_state)),
        seat_state.seat,
    )
    return SeatStep(
        seat_state=SeatState(
            environment_state=result.environment_state,
            carry=select(result.done, bot.init(game), carry),
            seat=following,
        ),
        observation=_learners(result.observations, seat, following),
        reward=result.reward[learner],
        reward_components=(
            None
            if result.reward_components is None
            else result.reward_components[learner]
        ),
        terminated=result.terminated,
        truncated=result.truncated,
        done=result.done,
        winner=result.winner,
        telemetry=result.telemetry,
        spawn_overflowed=result.spawn_overflowed,
        final_observation=_learners(result.final_observations, seat, seat_state.seat),
        final_valid=result.final_valid,
    )


def seat_commands[Carry](seat_state: SeatState[Carry], game: GameConfig, *, bot: Bot[Carry], seat: str | int | Array = "player_0") -> tuple[Carry, Commands]:
    environment_state = seat_state.environment_state
    learner = _current(seat, seat_state.seat)
    state = environment_state.core_state
    key = turn_key(episode_root(environment_state), state.time)
    return bot.act(key, seat_state.carry, state, game, 1 - learner)


def observe_seat(seat_state: SeatState[Any], game: GameConfig, *, layout: jaxmarl.JaxMARLLayout, assets: Any = None, seat: str | int | Array = "player_0") -> jaxmarl.Observation:
    observations = jaxmarl.observe(
        seat_state.environment_state.core_state, game, layout=layout, assets=assets
    )
    return _learners(observations, seat, seat_state.seat)


def legal_seat_actions(seat_state: SeatState[Any], game: GameConfig, *, layout: jaxmarl.JaxMARLLayout, seat: str | int | Array = "player_0") -> Array:
    masks = jaxmarl.legal_actions(
        seat_state.environment_state.core_state, game, layout=layout
    )
    return _learners(masks, seat, seat_state.seat)


def seat_slot_cells(seat_state: SeatState[Any], game: GameConfig, *, layout: jaxmarl.JaxMARLLayout, seat: str | int | Array = "player_0") -> Array:
    cells = jaxmarl.slot_cells(
        seat_state.environment_state.core_state, game, layout=layout
    )
    return _learners(cells, seat, seat_state.seat)


class SeatEnv(MultiAgentEnv):
    def __init__(self, bot: Bot[Any], *, scenario: GameConfig, seat: str = "player_0", surface: str = "symbolic", partial_observability: bool = False, final_observation: bool = True, feature_dtype: DTypeLike | None = None, reward_components: bool = False) -> None:
        if seat not in SEAT_RULES:
            raise ModelError(
                f"SeatEnv refuses seat={seat!r}: the accepted rules are "
                f"{list(SEAT_RULES)}; a per-lane seat goes through the pure "
                "functions"
            )
        two_agent = jaxmarl.make(
            "NanoRTS",
            scenario=scenario,
            surface=surface,
            partial_observability=partial_observability,
            final_observation=final_observation,
            feature_dtype=feature_dtype,
            reward_components=reward_components,
        )
        bot.init(two_agent.game)
        super().__init__(num_agents=1)
        self.bot = bot
        self.seat = seat
        self.final_observation = final_observation
        self.agents = [LEARNER]
        self.game = two_agent.game
        self.assets = two_agent.assets
        self.layout = two_agent.layout
        self.observation_spaces: dict[str, spaces.Dict] = {
            LEARNER: two_agent.observation_space(jaxmarl.AGENTS[0])
        }
        self.action_spaces: dict[str, spaces.Box] = {
            LEARNER: two_agent.action_space(jaxmarl.AGENTS[0])
        }

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: Array) -> tuple[dict[str, Any], SeatState[Any]]:
        seat_state, observation = reset_seat(
            key,
            self.game,
            bot=self.bot,
            layout=self.layout,
            assets=self.assets,
            seat=self.seat,
        )
        return {LEARNER: observation}, seat_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, key: Array, state: SeatState[Any], actions: dict[str, Array], reset_state: SeatState[Any] | None = None) -> tuple[dict[str, Any], SeatState[Any], dict[str, Array], dict[str, Array], dict[str, Any],]:
        del key
        if reset_state is not None:
            raise ModelError(
                "SeatEnv.step refuses a reset_state: this environment's "
                "auto-reset is the shared lifecycle's own core.reset at the next "
                "episode's root, with the bot's initial carry and the seat rule's "
                "next seat"
            )
        result = step_seat(
            state,
            actions[LEARNER],
            self.game,
            bot=self.bot,
            layout=self.layout,
            assets=self.assets,
            seat=self.seat,
        )
        infos: dict[str, Any] = {
            "terminated": result.terminated,
            "truncated": result.truncated,
            "winner": result.winner,
            "telemetry": result.telemetry,
            "spawn_overflowed": result.spawn_overflowed,
            "final_observation": {LEARNER: result.final_observation},
            "final_valid": result.final_valid,
        }
        if not self.final_observation:
            del infos["final_observation"]
        if result.reward_components is not None:
            infos["reward_components"] = {LEARNER: result.reward_components}
        return (
            {LEARNER: result.observation},
            result.seat_state,
            {LEARNER: result.reward.astype(jnp.float32)},
            {LEARNER: result.done, jaxmarl.ALL_AGENTS: result.done},
            infos,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_obs(self, state: SeatState[Any]) -> dict[str, Any]:
        return {
            LEARNER: observe_seat(
                state,
                self.game,
                layout=self.layout,
                assets=self.assets,
                seat=self.seat,
            )
        }

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: SeatState[Any]) -> dict[str, Array]:
        return {
            LEARNER: legal_seat_actions(
                state, self.game, layout=self.layout, seat=self.seat
            )
        }

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_slot_cells(self, state: SeatState[Any]) -> dict[str, Array]:
        return {
            LEARNER: seat_slot_cells(
                state, self.game, layout=self.layout, seat=self.seat
            )
        }

    def observation_space(self, agent: str) -> spaces.Dict:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Box:
        return self.action_spaces[agent]

    @property
    def agent_classes(self) -> dict[str, list[str]]:
        return {LEARNER: [LEARNER]}


def _static_player(seat: str | int | Array) -> int | None:
    if isinstance(seat, str):
        if seat not in SEAT_RULES:
            raise ModelError(
                f"the seat functions refuse seat={seat!r}: a seat is one of "
                f"{list(SEAT_RULES)} or a Player index"
            )
        return jaxmarl.AGENTS.index(seat) if seat in jaxmarl.AGENTS else None
    if isinstance(seat, int):
        if seat not in range(PLAYER_COUNT):
            raise ModelError(
                f"the seat functions refuse seat={seat}: a Player index is 0 or 1"
            )
        return seat
    return None


def _episode_seat(seat: str | int | Array, root: Array) -> Array:
    player = _static_player(seat)
    if player is not None:
        return jnp.asarray(player, jnp.int32)
    if isinstance(seat, str):
        return jax.random.randint(
            jax.random.fold_in(root, _SEAT_STREAM), (), 0, PLAYER_COUNT, jnp.int32
        )
    return jnp.asarray(seat, jnp.int32)


def _current(seat: str | int | Array, carried: Array) -> Array | int:
    player = _static_player(seat)
    return carried if player is None else player


def _learners[Leaf](per_agent: dict[str, Leaf], seat: str | int | Array, learner: Array) -> Leaf:
    player = _static_player(seat)
    if player is not None:
        return per_agent[jaxmarl.AGENTS[player]]
    first, second = (per_agent[agent] for agent in jaxmarl.AGENTS)
    return jax.tree.map(
        lambda zero, one: jnp.where(learner == 0, zero, one), first, second
    )
