from __future__ import annotations

import functools
from typing import Any, Final

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import DTypeLike
from jaxmarl.environments import spaces
from jaxmarl.environments.multi_agent_env import MultiAgentEnv

from nanorts import content, observations, rendering
from nanorts.adapters.jaxmarl.actions import AvailActions, legal_actions, slot_cells
from nanorts.adapters.jaxmarl.env import (
    JaxMARLStep,
    published_components,
    reset,
    step,
    transition,
)
from nanorts.adapters.jaxmarl.layout import AGENTS, JaxMARLLayout, prepare
from nanorts.adapters.jaxmarl.observation import Observations, observe
from nanorts.adapters.lifecycle import EnvironmentState, spawn_overflowed
from nanorts.model import PLAYER_COUNT, GameConfig, ModelError, UnitTypeTableVersion

ALL_AGENTS: Final[str] = "__all__"

ENVIRONMENT_IDS: Final[tuple[str, ...]] = ("NanoRTS",)


def make(env_id: str, *, scenario: GameConfig, utt: UnitTypeTableVersion | None = None, surface: str = "symbolic", partial_observability: bool = False, final_observation: bool = True, feature_dtype: DTypeLike | None = None, reward_components: bool = False) -> NanoRTS:
    if env_id not in ENVIRONMENT_IDS:
        raise ModelError(
            f"jaxmarl.make refuses environment id {env_id!r}: the accepted ids are "
            f"{list(ENVIRONMENT_IDS)}"
        )
    return NanoRTS(
        scenario if utt is None else content.with_unit_type_table(scenario, utt),
        surface=surface,
        partial_observability=partial_observability,
        final_observation=final_observation,
        feature_dtype=feature_dtype,
        reward_components=reward_components,
    )


def _host_resident[Captured](tree: Captured) -> Captured:
    if tree is None:
        return tree
    try:
        host = jax.devices("cpu")[0]
    except RuntimeError:
        return tree
    return jax.device_put(tree, host)


class NanoRTS(MultiAgentEnv):
    def __init__(self, scenario: GameConfig, *, surface: str = "symbolic", partial_observability: bool = False, final_observation: bool = True, feature_dtype: DTypeLike | None = None, reward_components: bool = False) -> None:
        if not isinstance(final_observation, bool):
            raise ModelError(
                f"NanoRTS refuses final_observation={final_observation!r}: it must "
                "be a bool, because it selects this instance's compiled programs"
            )
        super().__init__(num_agents=PLAYER_COUNT)
        self.final_observation = final_observation
        self.agents = list(AGENTS)
        self.game = _host_resident(scenario)
        self.assets: rendering.RenderAssets | None = _host_resident(
            rendering.prepare_assets(scenario.bucket) if surface == "pixels" else None
        )
        self.layout = prepare(
            self.game,
            surface=surface,
            partial_observability=partial_observability,
            assets=self.assets,
            feature_dtype=feature_dtype,
            mask_form="rows",
            reward_components=reward_components,
        )
        self.observation_spaces: dict[str, spaces.Dict] = dict.fromkeys(
            self.agents, _observation_space(self.layout)
        )
        self.action_spaces: dict[str, spaces.Box] = dict.fromkeys(
            self.agents, _action_space(self.layout)
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: Array) -> tuple[Observations, EnvironmentState]:
        environment_state, obs = reset(
            key, self.game, layout=self.layout, assets=self.assets
        )
        return obs, environment_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, key: Array, state: EnvironmentState, actions: dict[str, Array], reset_state: EnvironmentState | None = None) -> tuple[Observations, EnvironmentState, dict[str, Array], dict[str, Array], dict[str, Any],]:
        del key
        if reset_state is not None:
            raise ModelError(
                "NanoRTS.step refuses a reset_state: this environment's auto-reset "
                "is the shared lifecycle's own core.reset at the next episode's "
                "root, so a caller-supplied reset State would carry an episode "
                "identity nothing addresses"
            )
        result = step(state, actions, self.game, layout=self.layout, assets=self.assets)
        return (
            result.observations,
            result.environment_state,
            self._rewards(result.reward),
            self._dones(result.done),
            self._infos(result),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def step_env(self, key: Array, state: EnvironmentState, actions: dict[str, Array]) -> tuple[Observations, EnvironmentState, dict[str, Array], dict[str, Array], dict[str, Any],]:
        del key
        result = transition(state, actions, self.game, layout=self.layout)
        done = result.terminated | result.truncated
        carry = EnvironmentState(
            core_state=result.next_state,
            environment_root=state.environment_root,
            episode_index=state.episode_index,
        )
        ending = observe(
            result.next_state, self.game, layout=self.layout, assets=self.assets
        )
        return (
            ending,
            carry,
            self._rewards(result.reward),
            self._dones(done),
            self._infos(
                JaxMARLStep(
                    environment_state=carry,
                    observations=ending,
                    reward=result.reward,
                    reward_components=published_components(result, layout=self.layout),
                    terminated=result.terminated,
                    truncated=result.truncated,
                    done=done,
                    winner=result.next_state.winner,
                    telemetry=result.next_state.telemetry,
                    spawn_overflowed=spawn_overflowed(result.next_state),
                    final_observations=ending,
                    final_valid=done,
                )
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_obs(self, state: EnvironmentState) -> Observations:
        return observe(
            state.core_state, self.game, layout=self.layout, assets=self.assets
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: EnvironmentState) -> AvailActions:
        return legal_actions(state.core_state, self.game, layout=self.layout)

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_slot_cells(self, state: EnvironmentState) -> dict[str, Array]:
        return slot_cells(state.core_state, self.game, layout=self.layout)

    def observation_space(self, agent: str) -> spaces.Dict:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Box:
        return self.action_spaces[agent]

    @property
    def agent_classes(self) -> dict[str, list[str]]:
        return {"player": list(self.agents)}

    def _rewards(self, reward: Array) -> dict[str, Array]:
        return {
            agent: reward[player].astype(jnp.float32)
            for player, agent in enumerate(self.agents)
        }

    def _dones(self, done: Array) -> dict[str, Array]:
        return dict.fromkeys(self.agents, done) | {ALL_AGENTS: done}

    def _infos(self, result: JaxMARLStep) -> dict[str, Any]:
        infos = {
            "terminated": result.terminated,
            "truncated": result.truncated,
            "winner": result.winner,
            "telemetry": result.telemetry,
            "spawn_overflowed": result.spawn_overflowed,
            "final_observation": result.final_observations,
            "final_valid": result.final_valid,
        }
        if not self.final_observation:
            del infos["final_observation"]
        if result.reward_components is not None:
            infos["reward_components"] = {
                agent: result.reward_components[player]
                for player, agent in enumerate(self.agents)
            }
        return infos


def _observation_space(layout: JaxMARLLayout) -> spaces.Dict:
    observation = layout.observation
    scalars = spaces.Box(
        low=0,
        high=np.int32(np.iinfo(np.int32).max),
        shape=observation.scalars_shape,
        dtype=observation.scalars_dtype,
    )
    if isinstance(observation, observations.PixelLayout):
        low, high = observation.value_range
        return spaces.Dict(
            {
                "image": spaces.Box(
                    low=low,
                    high=high,
                    shape=observation.image_shape,
                    dtype=observation.image_dtype,
                ),
                "scalars": scalars,
            }
        )
    return spaces.Dict(
        {
            "feature_maps": spaces.Box(
                low=0,
                high=np.asarray(observation.bounds, np.int32),
                shape=observation.feature_maps_shape,
                dtype=observation.feature_maps_dtype,
            ),
            "scalars": scalars,
        }
    )


def _action_space(layout: JaxMARLLayout) -> spaces.Box:
    return spaces.Box(
        low=0,
        high=np.asarray(layout.action_widths, np.int32) - 1,
        shape=layout.actions_shape,
        dtype=jnp.int32,
    )
