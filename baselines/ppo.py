from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import functools
import json
import math
import os
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import flax.linen as nn
import flax.serialization
import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from jax import Array

import nanorts
from nanorts import bots, kernels, rendering
from nanorts.adapters import jaxmarl
from nanorts.model import UnitTypeTableVersion


def _process_start() -> float:
    try:
        fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
        started = int(fields[19]) / os.sysconf("SC_CLK_TCK")
        alive = time.clock_gettime(time.CLOCK_BOOTTIME) - started
    except (OSError, ValueError, IndexError, AttributeError):
        alive = 0.0
    return time.monotonic() - alive


PROCESS_START = _process_start()

SELF_PLAY = "self-play"

FEATURE_MAPS = (
    "hit_points",
    "carried_resources",
    "owner",
    "unit_type",
    "pending_action_kind",
    "terrain",
)
PLANES = (5, 5, 3, 8, 6, 2)

ACTION_WIDTHS = (6, 4, 4, 4, 4, 7, 49)
SMALL = len(ACTION_WIDTHS) - 1
BLOCK = max(ACTION_WIDTHS[:-1])
PADDED_WIDTH = SMALL * BLOCK + ACTION_WIDTHS[-1]

MASK_VALUE = -1e8

WILSON_Z = 1.959964
DTYPES = {"float32": jnp.float32, "bfloat16": jnp.bfloat16}
WANDB_MODES = ("online", "offline", "disabled")
OUTCOMES = ("wins", "draws", "losses")
LOSSES = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")

_STARTS = np.cumsum((0, *ACTION_WIDTHS[:-1]))
_HEAD = int(_STARTS[-1])
_FACTOR, _ENTRY = np.divmod(np.arange(SMALL * BLOCK), BLOCK)
_SOURCE = np.concatenate(
    [
        np.where(
            np.asarray(ACTION_WIDTHS)[_FACTOR] > _ENTRY, _STARTS[_FACTOR] + _ENTRY, -1
        ),
        _HEAD + np.arange(ACTION_WIDTHS[-1]),
    ]
)
_REAL = _SOURCE >= 0
_FLOOR = np.where(_REAL, MASK_VALUE, 2 * MASK_VALUE).astype(np.float32)
_ATTACK = np.maximum(_SOURCE - _HEAD, 0)
_WORD = np.where(_SOURCE < _HEAD, 0, 1 + _ATTACK // 32)
_SHIFT = np.where(_SOURCE < _HEAD, np.maximum(_SOURCE, 0), _ATTACK % 32).astype(
    np.uint32
)


class ConfigError(ValueError): ...


@dataclasses.dataclass(frozen=True)
class GroupSpec:
    opponent: str
    share: int

    @property
    def agents(self) -> int:
        return 2 if self.opponent == SELF_PLAY else 1


@dataclasses.dataclass(frozen=True)
class Config:
    preset: str = "p8-8x8"
    run_name: str = ""
    output_dir: str = "runs"

    map: str = "basesWorkers8x8"
    utt: str = "VERSION_ORIGINAL_FINETUNED"
    groups: tuple[GroupSpec, ...] = tuple(
        GroupSpec(opponent, 1)
        for opponent in (
            "randomAI",
            "randomBiasedAI",
            "workerRushAI",
            "lightRushAI",
            SELF_PLAY,
        )
    )
    lanes: int = 4100
    seat: str = "drawn"
    max_cycles: int = 5000
    reward_weights: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    seed: int = 1

    total_timesteps: int = 1_000_000_000
    num_steps: int = 128
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    adam_eps: float = 1e-5
    gamma: float = 0.999
    gae_lambda: float = 0.99
    update_epochs: int = 4
    num_minibatches: int = 8
    microbatch_rows: int = 8192
    clip_coef: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    hidden: int = 512
    layers: int = 2
    head_width: int = 128
    window: int = 3
    slot_limit: int = 16
    dtype: str = "bfloat16"

    eval_every: int = 10_000_000
    eval_opponents: tuple[str, ...] = (
        "randomAI",
        "randomBiasedAI",
        "workerRushAI",
        "lightRushAI",
    )
    required_opponents: tuple[str, ...] = ("randomAI", "randomBiasedAI", "workerRushAI")
    eval_seats: tuple[str, ...] = ("player_0", "player_1")
    eval_games: int = 20
    roster_every: int = 5
    stop_at_criterion: bool = False
    videos: bool = True
    video_stride: int = 2
    video_frame_ms: int = 50

    wandb_mode: str = "offline"
    wandb_project: str = "nanorts-baselines"
    wandb_entity: str = ""
    wandb_group: str = ""

    @property
    def share_total(self) -> int:
        return sum(group.share for group in self.groups)

    @property
    def group_lanes(self) -> tuple[int, ...]:
        multiple = self.lanes // self.share_total
        return tuple(group.share * multiple for group in self.groups)

    @property
    def rows(self) -> int:
        return sum(
            lanes * group.agents
            for group, lanes in zip(self.groups, self.group_lanes, strict=True)
        )

    @property
    def batch_size(self) -> int:
        return self.rows * self.num_steps

    @property
    def updates(self) -> int:
        return max(1, round(self.total_timesteps / self.batch_size))

    @property
    def eval_period(self) -> int:
        return max(1, self.eval_every // self.batch_size) if self.eval_every else 0

    @property
    def minibatch_rows(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def microbatch(self) -> int:
        minibatch, limit = self.minibatch_rows, self.microbatch_rows
        if limit < 1 or minibatch <= limit:
            return minibatch
        return max(rows for rows in range(1, limit + 1) if minibatch % rows == 0)


PRESETS: Mapping[str, Mapping[str, Any]] = {
    "p8-8x8": {},
    "microrts-py-bots": {
        "map": "basesWorkers16x16",
        "groups": [
            {"opponent": "coacAI", "share": 18},
            {"opponent": "randomBiasedAI", "share": 2},
            {"opponent": "lightRushAI", "share": 2},
            {"opponent": "workerRushAI", "share": 2},
        ],
        "lanes": 528,
        "seat": "player_0",
        "slot_limit": 32,
    },
    "microrts-py-selfplay": {
        "map": "basesWorkers16x16",
        "groups": [{"opponent": SELF_PLAY, "share": 1}],
        "lanes": 512,
        "seat": "player_0",
        "slot_limit": 32,
    },
}


def load_config(preset: str | None, overrides: Sequence[str] = ()) -> Config:
    config = Config()
    if preset is not None:
        if preset not in PRESETS:
            raise ConfigError(f"unknown preset {preset!r}; one of {list(PRESETS)}")
        config = _replaced(config, {"preset": preset, **PRESETS[preset]})
    return override(config, overrides)


def override(config: Config, overrides: Sequence[str]) -> Config:
    values: dict[str, Any] = {}
    for item in overrides:
        key, separator, text = item.partition("=")
        if not separator or not key:
            raise ConfigError(f"an override is key=value, not {item!r}")
        values[key] = yaml.safe_load(text)
    return _replaced(config, values)


def _replaced(config: Config, values: Mapping[str, Any]) -> Config:
    unknown = sorted(set(values) - {field.name for field in dataclasses.fields(Config)})
    if unknown:
        raise ConfigError(f"unknown config keys {unknown}")
    coerced = {
        name: _coerce(name, getattr(config, name), value)
        for name, value in values.items()
    }
    return _checked(dataclasses.replace(config, **coerced))


def _coerce(name: str, default: Any, value: Any) -> Any:
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ConfigError(f"{name}={value!r} is not a bool")
        return value
    if isinstance(default, int | float) and not isinstance(value, bool):
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ConfigError(f"{name}={value!r} is not a number") from None
        if isinstance(default, float):
            return number
        if number != int(number):
            raise ConfigError(f"{name}={value!r} is not an int")
        return int(number)
    if isinstance(default, str):
        return "" if value is None else str(value)
    if name == "groups":
        if not isinstance(value, list | tuple) or not value:
            raise ConfigError(f"groups={value!r} is not a list of groups")
        groups = []
        for group in value:
            if isinstance(group, GroupSpec):
                groups.append(group)
            elif isinstance(group, dict) and set(group) == {"opponent", "share"}:
                share = _coerce("share", 1, group["share"])
                groups.append(GroupSpec(str(group["opponent"]), share))
            else:
                raise ConfigError(f"a group is {{opponent, share}}, not {group!r}")
        return tuple(groups)
    if not isinstance(value, list | tuple):
        raise ConfigError(f"{name}={value!r} is not a list")
    element = default[0] if default else ""
    return tuple(_coerce(name, element, item) for item in value)


def _checked(config: Config) -> Config:
    if config.map not in nanorts.scenario_names():
        raise ConfigError(f"map={config.map!r}; one of {nanorts.scenario_names()}")
    for group in config.groups:
        if group.opponent != SELF_PLAY and group.opponent not in bots.ROSTER:
            raise ConfigError(f"opponent {group.opponent!r} is not a bot")
        if group.share < 1:
            raise ConfigError(f"group {group.opponent!r} has share {group.share}")
    positive = (
        "lanes",
        "max_cycles",
        "total_timesteps",
        "num_steps",
        "update_epochs",
        "num_minibatches",
        "hidden",
        "layers",
        "head_width",
        "eval_games",
        "video_stride",
        "video_frame_ms",
    )
    for name in positive:
        if getattr(config, name) < 1:
            raise ConfigError(f"{name}={getattr(config, name)} must be positive")
    for name in ("microbatch_rows", "eval_every", "roster_every", "slot_limit"):
        if getattr(config, name) < 0:
            raise ConfigError(f"{name}={getattr(config, name)} is negative")
    if config.window not in (1, 3, 5, 7):
        raise ConfigError(f"window={config.window}; an odd side of at most 7")
    for name, allowed in (
        ("dtype", tuple(DTYPES)),
        ("utt", tuple(member.value for member in UnitTypeTableVersion)),
        ("seat", bots.SEAT_RULES),
        ("wandb_mode", WANDB_MODES),
    ):
        if getattr(config, name) not in allowed:
            raise ConfigError(f"{name}={getattr(config, name)!r}; one of {allowed}")
    if config.lanes % config.share_total:
        raise ConfigError(f"lanes={config.lanes} is no multiple of the share total")
    if config.batch_size % config.num_minibatches:
        raise ConfigError(
            f"the batch of {config.rows} rows x {config.num_steps} steps does not "
            f"split into {config.num_minibatches} equal minibatches"
        )
    if config.microbatch < min(config.minibatch_rows, config.microbatch_rows // 8):
        raise ConfigError(
            f"the {config.minibatch_rows}-row minibatch has no divisor near "
            f"microbatch_rows={config.microbatch_rows}"
        )
    if len(config.reward_weights) != len(jaxmarl.REWARD_COMPONENTS):
        raise ConfigError(f"reward_weights weighs {jaxmarl.REWARD_COMPONENTS}")
    for opponent in config.eval_opponents:
        if opponent not in bots.ROSTER:
            raise ConfigError(f"evaluation opponent {opponent!r} is not a bot")
    for opponent in config.required_opponents:
        if opponent not in config.eval_opponents:
            raise ConfigError(f"required {opponent!r} is no evaluation opponent")
    for seat in config.eval_seats:
        if seat not in jaxmarl.AGENTS:
            raise ConfigError(f"evaluation seat {seat!r}; one of {jaxmarl.AGENTS}")
    return config


def plain(config: Config) -> dict[str, Any]:
    values = dataclasses.asdict(config)
    values["groups"] = [dict(group) for group in values["groups"]]
    return {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in values.items()
    }


def _planes(values: Array, dtype: Any = jnp.float32) -> Array:
    return jnp.concatenate(
        [
            jax.nn.one_hot(
                jnp.where(
                    values[..., index] < 0,
                    -1,
                    jnp.clip(values[..., index], 0, width - 1),
                ),
                width,
                dtype=dtype,
            )
            for index, width in enumerate(PLANES)
        ],
        axis=-1,
    )


def flat_planes(feature_maps: Array) -> Array:
    return _planes(feature_maps.astype(jnp.int32)).reshape(feature_maps.shape[0], -1)


@functools.cache
def _window_table(height: int, width: int, window: int) -> tuple[np.ndarray, np.ndarray]:
    side = width + window - 1
    radius = window // 2
    y, x = np.divmod(np.arange(height * width), width)
    dy, dx = np.divmod(np.arange(window * window), window)
    row, column = y[:, None] + dy, x[:, None] + dx
    outside = (
        (row < radius)
        | (row >= height + radius)
        | (column < radius)
        | (column >= width + radius)
    )
    return (row * side + column).astype(np.int32), outside


def slot_features(feature_maps: Array, cells: Array, mask: Array, *, window: int, dtype: Any) -> Array:
    batch, height, width, maps = feature_maps.shape
    radius = window // 2
    grid = jnp.pad(
        feature_maps, ((0, 0), (radius, radius), (radius, radius), (0, 0))
    ).reshape(batch, -1, maps)
    table, outside = _window_table(height, width, window)
    cell = jnp.maximum(cells, 0)
    index = jnp.asarray(table)[cell].reshape(batch, -1)
    values = jax.vmap(lambda rows, at: rows[at])(grid, index).astype(jnp.int32)
    values = jnp.where(
        jnp.asarray(outside)[cell].reshape(batch, -1)[..., None], -1, values
    )
    planes = _planes(values, dtype).reshape(*cells.shape, -1)
    position = cells[..., None] == jnp.arange(height * width, dtype=cells.dtype)
    return jnp.concatenate(
        [planes, mask.astype(dtype), position.astype(dtype)], axis=-1
    )


class SlotActorCritic(nn.Module):
    hidden: int = 512
    layers: int = 2
    head_width: int = 128
    window: int = 3
    dtype: str = "bfloat16"

    def setup(self) -> None:
        gain = nn.initializers.orthogonal(math.sqrt(2))
        dense = functools.partial(
            nn.Dense,
            dtype=DTYPES[self.dtype],
            param_dtype=jnp.float32,
            kernel_init=gain,
        )
        exact = functools.partial(nn.Dense, dtype=jnp.float32)
        self.actor = [dense(self.hidden) for _ in range(self.layers)]
        self.critic = [dense(self.hidden) for _ in range(self.layers)]
        self.value_head = exact(1, kernel_init=nn.initializers.orthogonal(1.0))
        self.slot_input = dense(self.head_width)
        self.slot_context = dense(self.head_width, use_bias=False)
        self.slot_logits = exact(
            PADDED_WIDTH, kernel_init=nn.initializers.orthogonal(0.01)
        )

    def __call__(self, feature_maps: Array, cells: Array, mask: Array) -> tuple[Array, Array]:
        inputs = flat_planes(feature_maps).astype(DTYPES[self.dtype])
        slots = slot_features(
            feature_maps, cells, mask, window=self.window, dtype=jnp.int8
        )
        hidden = nn.relu(
            self.slot_input(slots)
            + self.slot_context(self._trunk(self.actor, inputs))[:, None, :]
        )
        return self.slot_logits(hidden), self._value(inputs)

    def value(self, feature_maps: Array) -> Array:
        return self._value(flat_planes(feature_maps).astype(DTYPES[self.dtype]))

    def _trunk(self, layers: Sequence[nn.Dense], x: Array) -> Array:
        for layer in layers:
            x = nn.relu(layer(x))
        return x

    def _value(self, inputs: Array) -> Array:
        return self.value_head(self._trunk(self.critic, inputs))[..., 0]


def network(config: Config) -> SlotActorCritic:
    return SlotActorCritic(
        hidden=config.hidden,
        layers=config.layers,
        head_width=config.head_width,
        window=config.window,
        dtype=config.dtype,
    )


def policy(model: SlotActorCritic, params: Any, feature_maps: Array, cells: Array, mask: Array) -> tuple[Array, Array]:
    return cast(tuple[Array, Array], model.apply(params, feature_maps, cells, mask))


def critic(model: SlotActorCritic, params: Any, feature_maps: Array) -> Array:
    return cast(Array, model.apply(params, feature_maps, method=SlotActorCritic.value))


def pad_rows(rows: Array, fill: Any) -> Array:
    gathered = rows[..., np.maximum(_SOURCE, 0)]
    return jnp.where(_REAL, gathered, jnp.asarray(fill, rows.dtype))


def pack_mask(rows: Array) -> Array:
    middle = _HEAD + min(ACTION_WIDTHS[-1], 32)
    words = []
    for start, stop in ((0, _HEAD), (_HEAD, middle), (middle, sum(ACTION_WIDTHS))):
        shifts = np.arange(stop - start, dtype=np.uint32)
        words.append(
            jnp.sum(
                rows[..., start:stop].astype(jnp.uint32) << shifts,
                axis=-1,
                dtype=jnp.uint32,
            )
        )
    return jnp.stack(words, axis=-1)


def unpack_padded(words: Array) -> Array:
    bits = (words[..., _WORD] >> _SHIFT) & np.uint32(1)
    return (bits != 0) & _REAL


def live_slots(mask: Array) -> Array:
    return jnp.any(mask[..., 1 : ACTION_WIDTHS[0]], axis=-1)


def select_slots(live: Array, limit: int) -> Array:
    order = jnp.argsort(~live, axis=-1, stable=True)
    return order[..., :limit].astype(jnp.int32)


def _taken(values: Array, slots: Array) -> Array:
    index = slots.reshape(slots.shape + (1,) * (values.ndim - slots.ndim))
    return jnp.take_along_axis(values, index, axis=slots.ndim - 1)


def _blocks(row: Array) -> tuple[Array, Array]:
    small = row[..., : SMALL * BLOCK].reshape(*row.shape[:-1], SMALL, BLOCK)
    return small, row[..., SMALL * BLOCK :]


def sample_actions(key: Array, logits: Array, mask: Array) -> tuple[Array, Array]:
    small, attack = _blocks(jnp.where(mask, logits, _FLOOR))
    small_key, attack_key = jax.random.split(key)
    actions = jnp.concatenate(
        [
            jax.random.categorical(small_key, small, axis=-1),
            jax.random.categorical(attack_key, attack, axis=-1)[..., None],
        ],
        axis=-1,
    ).astype(jnp.int32)
    return actions, evaluate_actions(logits, mask, actions)[0]


def evaluate_actions(logits: Array, mask: Array, actions: Array) -> tuple[Array, Array]:
    actions = actions.astype(jnp.int32)
    log_prob = jnp.zeros(logits.shape[:-2], jnp.float32)
    entropy = jnp.zeros(logits.shape[:-2], jnp.float32)
    parts = zip(
        _blocks(jnp.where(mask, logits, _FLOOR)),
        _blocks(mask),
        (actions[..., :SMALL], actions[..., SMALL]),
        strict=True,
    )
    for masked, allowed, chosen in parts:
        normalized = jax.nn.log_softmax(masked, axis=-1)
        live = jnp.any(allowed, axis=-1)
        entries = jnp.arange(masked.shape[-1], dtype=jnp.int32)
        drawn = jnp.sum(
            jnp.where(chosen[..., None] == entries, normalized, 0.0), axis=-1
        )
        spread = -jnp.sum(jnp.exp(normalized) * normalized, axis=-1)
        axes = tuple(range(logits.ndim - 2, live.ndim))
        log_prob += jnp.sum(jnp.where(live, drawn, 0.0), axis=axes)
        entropy += jnp.sum(jnp.where(live, spread, 0.0), axis=axes)
    return log_prob, entropy


KIND_PARAMETERS = ((), (1,), (2,), (3,), (4, 5), (6,))


def invalid_draws(mask: Array, actions: Array) -> Array:
    def drawn(factor: int) -> Array:
        start = int(_STARTS[factor])
        bits = mask[..., start : start + ACTION_WIDTHS[factor]]
        index = actions[..., factor, None].astype(jnp.int32)
        return jnp.take_along_axis(bits, index, axis=-1)[..., 0]

    kind = actions[..., 0]
    valid = drawn(0)
    for code, parameters in enumerate(KIND_PARAMETERS):
        for factor in parameters:
            valid &= (kind != code) | drawn(factor)
    return jnp.sum(~valid, axis=-1, dtype=jnp.int32)


def gae(rewards: Array, values: Array, dones: Array, last_value: Array, *, gamma: float, gae_lambda: float) -> tuple[Array, Array]:
    def backward(carry: tuple[Array, Array], step: tuple[Array, ...]) -> Any:
        following, next_value = carry
        reward, value, done = step
        nonterminal = 1.0 - done.astype(jnp.float32)
        delta = reward + gamma * next_value * nonterminal - value
        following = delta + gamma * gae_lambda * nonterminal * following
        return (following, value), following

    _, advantages = jax.lax.scan(
        backward,
        (jnp.zeros_like(last_value), last_value),
        (rewards, values, dones),
        reverse=True,
    )
    return advantages, advantages + values


def normalized_advantages(advantages: Array) -> Array:
    return (advantages - advantages.mean()) / (advantages.std(ddof=1) + 1e-8)


class LossInfo(NamedTuple):
    policy_loss: Array
    value_loss: Array
    entropy: Array
    approx_kl: Array
    clip_fraction: Array


def ppo_loss(new_log_prob: Array, new_value: Array, entropy: Array, old_log_prob: Array, old_value: Array, advantages: Array, returns: Array, *, clip_coef: float, ent_coef: float, vf_coef: float) -> tuple[Array, LossInfo]:
    ratio = jnp.exp(new_log_prob - old_log_prob)
    clipped_ratio = jnp.clip(ratio, 1 - clip_coef, 1 + clip_coef)
    policy_loss = jnp.maximum(-advantages * ratio, -advantages * clipped_ratio).mean()
    clipped_value = old_value + jnp.clip(new_value - old_value, -clip_coef, clip_coef)
    value_loss = (
        0.5
        * jnp.maximum((new_value - returns) ** 2, (clipped_value - returns) ** 2).mean()
    )
    loss = policy_loss - ent_coef * entropy.mean() + vf_coef * value_loss
    return loss, LossInfo(
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy.mean(),
        approx_kl=(old_log_prob - new_log_prob).mean(),
        clip_fraction=(jnp.abs(ratio - 1.0) > clip_coef).mean(),
    )


def wilson_lower(wins: int, games: int) -> float:
    share, z2 = wins / games, WILSON_Z**2
    centre = share + z2 / (2 * games)
    margin = WILSON_Z * math.sqrt(share * (1 - share) / games + z2 / (4 * games**2))
    return (centre - margin) / (1 + z2 / games)


def criterion_wins(games: int) -> int:
    return next(wins for wins in range(games + 1) if wilson_lower(wins, games) > 0.5)


def _scenario(config: Config) -> Any:
    return nanorts.map_name_to_scenario(
        config.map, utt=UnitTypeTableVersion(config.utt), max_cycles=config.max_cycles
    )


def _mix(groups: Sequence[tuple[str, str, int]], config: Config, *, final_observation: bool = False, reward_components: bool = False) -> Any:
    scenario = _scenario(config)
    flags = {
        "final_observation": final_observation,
        "reward_components": reward_components,
    }
    built = []
    for opponent, seat, lanes in groups:
        environment = (
            nanorts.make("NanoRTS", scenario=scenario, **flags)
            if opponent == SELF_PLAY
            else bots.SeatEnv(
                bots.ROSTER[opponent], scenario=scenario, seat=seat, **flags
            )
        )
        if tuple(environment.layout.observation.groups) != FEATURE_MAPS:
            raise ConfigError(
                f"the environment's Feature maps are "
                f"{environment.layout.observation.groups}, not {FEATURE_MAPS}"
            )
        built.append(bots.Group(environment, lanes))
    return bots.GroupMix(built)


class _RowLayout:
    def __init__(self, mix: Any) -> None:
        self.mix = mix
        self.membership = np.zeros((mix.rows, len(mix.groups)), np.int32)
        self.lane_rows = np.zeros(mix.rows, bool)
        for index, (group, rows) in enumerate(
            zip(mix.groups, mix.slices(), strict=True)
        ):
            self.membership[rows, index] = 1
            self.lane_rows[rows.start : rows.start + group.lanes] = True
        environment = mix.groups[0].environment
        self.capacity = int(environment.action_space(environment.agents[0]).shape[0])

    def players(self, states: tuple[Any, ...]) -> Array:
        parts = []
        for group, state in zip(self.mix.groups, states, strict=True):
            if isinstance(group.environment, bots.SeatEnv):
                parts.append(state.seat.astype(jnp.int32))
            else:
                parts.append(jnp.repeat(jnp.arange(2, dtype=jnp.int32), group.lanes))
        return jnp.concatenate(parts)

    def per_group(self, counts: Array) -> Array:
        return jnp.einsum("rg,r...->g...", jnp.asarray(self.membership), counts)


def _outcomes(done: Array, winner: Array, players: Array) -> Array:
    return jnp.stack(
        [
            done & (winner == players),
            done & (winner < 0),
            done & (winner == 1 - players),
        ],
        axis=-1,
    ).astype(jnp.int32)


class Decision(NamedTuple):
    rows: Array
    cells: Array
    selected: Array
    actions: Array
    unselected: Array
    log_prob: Array
    value: Array


def decide(model: SlotActorCritic, params: Any, feature_maps: Array, cells: Array, rows: Array, key: Array, limit: int) -> Decision:
    capacity = cells.shape[-1]
    padded = pad_rows(rows, False)
    live = live_slots(padded)
    slots = select_slots(live, min(limit or capacity, capacity))
    rows, cells, mask = (_taken(part, slots) for part in (rows, cells, padded))
    logits, value = policy(model, params, feature_maps, cells, mask)
    selected, log_prob = sample_actions(key, logits, mask)
    hit = slots[..., None] == jnp.arange(capacity, dtype=jnp.int32)
    actions = jnp.sum(
        jnp.where(hit[..., None], selected[..., None, :], 0), axis=-3, dtype=jnp.int32
    )
    unselected = jnp.sum(live, -1, dtype=jnp.int32) - jnp.sum(
        _taken(live, slots), -1, dtype=jnp.int32
    )
    return Decision(rows, cells, selected, actions, unselected, log_prob, value)


class _Step(NamedTuple):
    decision: Decision
    invalid: Array
    players: Array
    observations: Any
    states: Any
    done: Array
    infos: dict[str, Any]


def _act(layout: _RowLayout, model: SlotActorCritic, params: Any, states: Any, feature_maps: Array, key: Array, limit: int) -> _Step:
    mix = layout.mix
    step_key, sample_key = jax.random.split(key)
    rows = mix.get_avail_actions(states)
    decision = decide(
        model,
        params,
        feature_maps,
        mix.get_slot_cells(states),
        rows,
        sample_key,
        limit,
    )
    observations, following, _, done, infos = mix.step(
        step_key, states, decision.actions
    )
    return _Step(
        decision=decision,
        invalid=invalid_draws(rows, decision.actions),
        players=layout.players(states),
        observations=observations,
        states=following,
        done=done,
        infos=infos,
    )


def act(params: Any, feature_maps: Array, cells: Array, mask: Array, key: Array, *, config: Config) -> Array:
    batch = (feature_maps[None], cells[None], mask[None])
    return decide(network(config), params, *batch, key, config.slot_limit).actions[0]


def roster_runs() -> bool:
    return jax.default_backend() != "gpu" or kernels.selected()


def evaluation_cases(config: Config, *, roster: bool) -> tuple[tuple[str, str], ...]:
    opponents = tuple(bots.ROSTER) if roster else config.eval_opponents
    return tuple(
        (opponent, seat) for opponent in opponents for seat in config.eval_seats
    )


def video_opponents(config: Config, *, roster: bool) -> tuple[str, ...]:
    cases = evaluation_cases(config, roster=roster)
    return tuple(dict.fromkeys(opponent for opponent, _ in cases))


class Evaluation:
    def __init__(self, config: Config, *, roster: bool, videos: bool) -> None:
        self.config = config
        self.cases = evaluation_cases(config, roster=roster)
        mix = _mix(
            [(opponent, seat, config.eval_games) for opponent, seat in self.cases],
            config,
            final_observation=videos,
        )
        self.layout = _RowLayout(mix)
        self.model = network(config)
        self.videos = videos
        starts: dict[str, int] = {}
        for (opponent, _), rows in zip(self.cases, mix.slices(), strict=True):
            starts.setdefault(opponent, rows.start)
        self.video_rows = np.asarray(list(starts.values()), np.int32)

    def play(self, params: Any, key: Array) -> tuple[Array, Array | None, Array]:
        mix, cycles = self.layout.mix, self.config.max_cycles
        key, environment_key = jax.random.split(key)
        observations, states = mix.reset(jax.random.split(environment_key, mix.lanes))
        start = observations.feature_maps
        rows = jnp.asarray(self.video_rows)
        frames = None
        if self.videos:
            frames = jnp.zeros((len(rows), cycles + 1, *start.shape[1:]), start.dtype)
            frames = frames.at[:, 0].set(start[rows])

        class Carry(NamedTuple):
            cycle: Array
            states: Any
            maps: Array
            key: Array
            outcome: Array
            ended: Array
            lengths: Array
            frames: Array | None

        def going(carry: Carry) -> Array:
            return (carry.cycle < cycles) & jnp.any(~carry.ended)

        def cycle(carry: Carry) -> Carry:
            key, step_key = jax.random.split(carry.key)
            step = _act(
                self.layout,
                self.model,
                params,
                carry.states,
                carry.maps,
                step_key,
                self.config.slot_limit,
            )
            first = step.done & ~carry.ended
            frames = carry.frames
            if frames is not None:
                frame = step.infos["final_observation"].feature_maps[rows]
                frames = jax.lax.dynamic_update_slice_in_dim(
                    frames, frame[:, None], carry.cycle + 1, axis=1
                )
            return Carry(
                cycle=carry.cycle + 1,
                states=step.states,
                maps=step.observations.feature_maps,
                key=key,
                outcome=carry.outcome
                + _outcomes(first, step.infos["winner"], step.players),
                ended=carry.ended | step.done,
                lengths=jnp.where(carry.ended[rows], carry.lengths, carry.lengths + 1),
                frames=frames,
            )

        final = jax.lax.while_loop(
            going,
            cycle,
            Carry(
                cycle=jnp.int32(0),
                states=states,
                maps=start,
                key=key,
                outcome=jnp.zeros((mix.rows, len(OUTCOMES)), jnp.int32),
                ended=jnp.zeros(mix.rows, jnp.bool_),
                lengths=jnp.zeros(len(rows), jnp.int32),
                frames=frames,
            ),
        )
        return self.layout.per_group(final.outcome), final.frames, final.lengths


class Trajectory(NamedTuple):
    feature_maps: Array
    cells: Array
    mask: Array
    actions: Array
    log_prob: Array
    value: Array
    reward: Array
    done: Array


class Runner(NamedTuple):
    params: Any
    opt_state: Any
    env_states: Any
    feature_maps: Array
    key: Array
    episode_return: Array
    episode_length: Array


class Result(NamedTuple):
    params: Any
    eval_key: Array
    metrics: dict[str, Array]


Host = Callable[[str, np.ndarray, np.ndarray, dict[str, np.ndarray]], None]


class Training:
    def __init__(self, config: Config, host: Host | None = None) -> None:
        self.config = config
        self.host: Host = host or (lambda *_: None)
        self.mix = _mix(
            [
                (group.opponent, config.seat, lanes)
                for group, lanes in zip(config.groups, config.group_lanes, strict=True)
            ],
            config,
            reward_components=True,
        )
        self.layout = _RowLayout(self.mix)
        self.rows, self.lanes = self.mix.rows, self.mix.lanes
        self.capacity = self.layout.capacity
        self.slots = min(config.slot_limit or self.capacity, self.capacity)
        self.model = network(config)
        self.batch_size, self.num_updates = config.batch_size, config.updates
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(learning_rate=self._learning_rate, eps=config.adam_eps),
        )
        evaluated = bool(config.eval_period)
        self.evaluation = (
            Evaluation(config, roster=False, videos=config.videos)
            if evaluated
            else None
        )
        self.roster = (
            Evaluation(config, roster=True, videos=config.videos)
            if evaluated and config.roster_every and roster_runs()
            else None
        )

    def describe(self) -> dict[str, Any]:
        config, bucket = self.config, _scenario(self.config).bucket
        params = jax.eval_shape(self.init, jax.random.key(0)).params
        return {
            "surface": "symbolic, full observation",
            "bucket": {
                "height": int(bucket.height),
                "width": int(bucket.width),
                "per_player_capacity": int(bucket.per_player_capacity),
            },
            "rows": self.rows,
            "lanes": self.lanes,
            "group_lanes": dict(
                zip(_group_names(config), config.group_lanes, strict=True)
            ),
            "batch_size": self.batch_size,
            "minibatch_rows": config.minibatch_rows,
            "microbatch_rows": config.microbatch,
            "updates": self.num_updates,
            "learner_steps": self.num_updates * self.batch_size,
            "eval_period_updates": config.eval_period,
            "roster_evaluation": self.roster is not None,
            "network": "SlotActorCritic (flax.linen)",
            "parameters": sum(int(leaf.size) for leaf in jax.tree.leaves(params)),
            "versions": {
                "python": sys.version.split()[0],
                "jax": jax.__version__,
                "flax": flax.__version__,
                "optax": optax.__version__,
            },
            "devices": [str(device) for device in jax.devices()],
        }

    def _learning_rate(self, count: Array) -> Array:
        config = self.config
        if not config.anneal_lr:
            return jnp.float32(config.learning_rate)
        update = count // (config.update_epochs * config.num_minibatches)
        return config.learning_rate * jnp.maximum(1.0 - update / self.num_updates, 0.0)

    def init(self, key: Array) -> Runner:
        params_key, environment_key, key = jax.random.split(key, 3)
        observations, states = self.mix.reset(
            jax.random.split(environment_key, self.lanes)
        )
        maps = observations.feature_maps
        params = self.model.init(
            params_key,
            maps[:1],
            jnp.zeros((1, self.capacity), jnp.int32),
            jnp.zeros((1, self.capacity, PADDED_WIDTH), jnp.bool_),
        )
        return Runner(
            params=params,
            opt_state=self.optimizer.init(params),
            env_states=states,
            feature_maps=maps,
            key=key,
            episode_return=jnp.zeros(self.rows, jnp.float32),
            episode_length=jnp.zeros(self.rows, jnp.int32),
        )

    def collect(self, runner: Runner) -> tuple[Runner, Trajectory, dict[str, Array]]:
        lane_rows = jnp.asarray(self.layout.lane_rows)
        weights = jnp.asarray(self.config.reward_weights, jnp.float32)

        def cycle(runner: Runner, _: None) -> tuple[Runner, Any]:
            key, step_key = jax.random.split(runner.key)
            step = _act(
                self.layout,
                self.model,
                runner.params,
                runner.env_states,
                runner.feature_maps,
                step_key,
                self.config.slot_limit,
            )
            decision, done = step.decision, step.done
            reward = step.infos["reward_components"] @ weights
            episode_return = runner.episode_return + reward
            episode_length = runner.episode_length + 1
            stats = {
                "episodes": jnp.sum(done, dtype=jnp.int32),
                "episode_return_sum": jnp.sum(jnp.where(done, episode_return, 0.0)),
                "episode_length_sum": jnp.sum(
                    jnp.where(done, episode_length, 0), dtype=jnp.int32
                ),
                "outcomes": self.layout.per_group(
                    _outcomes(done, step.infos["winner"], step.players)
                ),
                "overflowed_episodes": jnp.sum(
                    done & step.infos["spawn_overflowed"] & lane_rows, dtype=jnp.int32
                ),
                "invalid_draws": jnp.sum(step.invalid, dtype=jnp.int32),
                "unselected_slots": jnp.sum(decision.unselected, dtype=jnp.int32),
            }
            transition = Trajectory(
                feature_maps=runner.feature_maps,
                cells=decision.cells,
                mask=pack_mask(decision.rows),
                actions=decision.selected.astype(jnp.uint8),
                log_prob=decision.log_prob,
                value=decision.value,
                reward=reward,
                done=done,
            )
            following = runner._replace(
                env_states=step.states,
                feature_maps=step.observations.feature_maps,
                key=key,
                episode_return=jnp.where(done, 0.0, episode_return),
                episode_length=jnp.where(done, 0, episode_length),
            )
            return following, (transition, stats)

        runner, (trajectory, stats) = jax.lax.scan(
            cycle, runner, None, length=self.config.num_steps
        )
        return runner, trajectory, jax.tree.map(lambda leaf: leaf.sum(0), stats)

    def _loss(self, params: Any, batch: tuple[Trajectory, Array, Array]) -> Any:
        trajectory, advantages, returns = batch
        mask = unpack_padded(trajectory.mask)
        logits, value = policy(
            self.model, params, trajectory.feature_maps, trajectory.cells, mask
        )
        log_prob, entropy = evaluate_actions(logits, mask, trajectory.actions)
        config = self.config
        return ppo_loss(
            log_prob,
            value,
            entropy,
            trajectory.log_prob,
            trajectory.value,
            advantages,
            returns,
            clip_coef=config.clip_coef,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
        )

    def learn(self, runner: Runner, trajectory: Trajectory) -> tuple[Runner, dict[str, Array]]:
        config = self.config
        advantages, returns = gae(
            trajectory.reward,
            trajectory.value,
            trajectory.done,
            critic(self.model, runner.params, runner.feature_maps),
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        steps, flat_advantages, flat_returns = jax.tree.map(
            lambda leaf: leaf.reshape((self.batch_size, *leaf.shape[2:])),
            (trajectory, advantages, returns),
        )
        microbatches = config.minibatch_rows // config.microbatch

        def gradient(params: Any, indices: Array, normalized: Array) -> Any:
            batch = (
                jax.tree.map(lambda leaf: leaf[indices], steps),
                normalized,
                flat_returns[indices],
            )
            return jax.grad(self._loss, has_aux=True)(params, batch)

        def minibatch(carry: Any, indices: Array) -> tuple[Any, LossInfo]:
            params, opt_state = carry
            normalized = normalized_advantages(flat_advantages[indices])
            if microbatches == 1:
                grads, info = gradient(params, indices, normalized)
            else:

                def accumulate(total: Any, part: tuple[Array, Array]) -> Any:
                    grads, info = gradient(params, *part)
                    return jax.tree.map(jnp.add, total, grads), info

                grads, parts = jax.lax.scan(
                    accumulate,
                    jax.tree.map(jnp.zeros_like, params),
                    (
                        indices.reshape(microbatches, -1),
                        normalized.reshape(microbatches, -1),
                    ),
                )
                grads = jax.tree.map(lambda leaf: leaf / microbatches, grads)
                info = jax.tree.map(lambda leaf: leaf.mean(0), parts)
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), opt_state), info

        def epoch(carry: Any, _: None) -> tuple[Any, LossInfo]:
            params, opt_state, key = carry
            key, order_key = jax.random.split(key)
            order = jax.random.permutation(order_key, self.batch_size).reshape(
                config.num_minibatches, config.minibatch_rows
            )
            (params, opt_state), info = jax.lax.scan(
                minibatch, (params, opt_state), order
            )
            return (params, opt_state, key), info

        (params, opt_state, key), info = jax.lax.scan(
            epoch,
            (runner.params, runner.opt_state, runner.key),
            None,
            length=config.update_epochs,
        )
        target, predicted = returns.reshape(-1), trajectory.value.reshape(-1)
        spread = jnp.var(target)
        metrics = {name: leaf.mean() for name, leaf in info._asdict().items()}
        metrics["non_finite"] = sum(
            (~jnp.isfinite(leaf)).sum(dtype=jnp.int32) for leaf in jax.tree.leaves(info)
        )
        metrics["explained_variance"] = jnp.where(
            spread == 0, jnp.nan, 1.0 - jnp.var(target - predicted) / spread
        )
        return runner._replace(params=params, opt_state=opt_state, key=key), metrics

    def _evaluated(self, evaluation: Evaluation, roster: bool, run: Array, number: Array) -> Callable[[Any, Array], None]:
        def play(params: Any, key: Array) -> None:
            outcomes, frames, lengths = evaluation.play(params, key)
            flag = jnp.bool_(roster)
            jax.debug.callback(
                functools.partial(self.host, "evaluation"),
                run,
                number,
                {"outcomes": outcomes, "roster": flag},
            )
            if frames is not None:
                jax.debug.callback(
                    functools.partial(self.host, "frames"),
                    run,
                    number,
                    {"frames": frames, "lengths": lengths, "roster": flag},
                )

        return play

    def train(self, key: Array) -> Result:
        run = jax.random.key_data(key)
        key, eval_root, roster_root = jax.random.split(key, 3)
        runner = self.init(key)
        period, last = self.config.eval_period, self.num_updates

        def update(runner: Runner, index: Array) -> tuple[Runner, dict[str, Array]]:
            runner, trajectory, stats = self.collect(runner)
            runner, metrics = self.learn(runner, trajectory)
            config = self.config
            schedule = index * config.update_epochs * config.num_minibatches
            metrics = {
                **metrics,
                **stats,
                "learning_rate": jnp.asarray(
                    self._learning_rate(schedule), jnp.float32
                ),
            }
            number = index + 1
            jax.debug.callback(
                functools.partial(self.host, "update"), run, number, metrics
            )
            if self.evaluation is not None:
                due = (number % period == 0) | (number == last)
                jax.lax.cond(
                    due,
                    self._evaluated(self.evaluation, False, run, number),
                    lambda *_: None,
                    runner.params,
                    jax.random.fold_in(eval_root, number),
                )
                if self.roster is not None:
                    due &= ((number // period) % self.config.roster_every == 0) | (
                        number == last
                    )
                    jax.lax.cond(
                        due,
                        self._evaluated(self.roster, True, run, number),
                        lambda *_: None,
                        runner.params,
                        jax.random.fold_in(roster_root, number),
                    )
            return runner, metrics

        runner, metrics = jax.lax.scan(
            update, runner, jnp.arange(self.num_updates, dtype=jnp.int32)
        )
        final = jax.random.key_data(jax.random.fold_in(eval_root, self.num_updates))
        return Result(params=runner.params, eval_key=final, metrics=metrics)


def make_train(config: Config, host: Host | None = None) -> Callable[[Array], Result]:
    return Training(config, host).train


def compiled(function: Callable[..., Any]) -> Any:
    return jax.jit(
        function, compiler_options=dict(nanorts.RECOMMENDED_COMPILER_OPTIONS)
    )


def save_params(path: Path, params: Any, config: Config, update: int, key: np.ndarray) -> Path:
    payload = {
        "params": flax.serialization.to_state_dict(jax.device_get(params)),
        "config": plain(config),
        "update": int(update),
        "key": np.asarray(key, np.uint32),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(flax.serialization.msgpack_serialize(payload))
    return path


def load_params(path: Path) -> tuple[Any, Config, int, np.ndarray]:
    payload = flax.serialization.msgpack_restore(Path(path).read_bytes())
    config = _replaced(Config(), payload["config"])
    return payload["params"], config, int(payload["update"]), np.asarray(payload["key"])


def _group_names(config: Config) -> list[str]:
    names: list[str] = []
    for group in config.groups:
        name = group.opponent
        names.append(name if name not in names else f"{name}-{len(names)}")
    return names


def _case_rows(cases: Sequence[tuple[str, str]], counts: np.ndarray) -> list[dict[str, Any]]:
    return [
        {
            "opponent": opponent,
            "seat": seat,
            "games": int(row.sum()),
            **{name: int(value) for name, value in zip(OUTCOMES, row, strict=True)},
        }
        for (opponent, seat), row in zip(cases, counts, strict=True)
    ]


def _pooled(rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[int, int]]:
    pooled: dict[str, tuple[int, int]] = {}
    for row in rows:
        wins, games = pooled.get(row["opponent"], (0, 0))
        pooled[row["opponent"]] = (wins + row["wins"], games + row["games"])
    return pooled


def beats_required(config: Config, rows: Sequence[Mapping[str, Any]]) -> bool:
    pooled = _pooled(rows)
    return all(
        opponent in pooled
        and pooled[opponent][0] >= criterion_wins(pooled[opponent][1])
        for opponent in config.required_opponents
    )


def _wandb_run(config: Config, directory: Path, name: str, job_type: str) -> Any:
    import wandb

    return wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity or os.environ.get("WANDB_ENTITY") or None,
        mode=cast(Literal["online", "offline", "disabled"], config.wandb_mode),
        name=name,
        group=config.wandb_group or None,
        job_type=job_type,
        config=plain(config),
        dir=str(directory),
    )


NO_HOST_DEVICE = (
    "videos draw on the CPU backend beside the running program; keep cpu in "
    "JAX_PLATFORMS (e.g. JAX_PLATFORMS=cuda,cpu) or set videos=false"
)


def _host_device() -> Any | None:
    try:
        return jax.devices("cpu")[0]
    except RuntimeError:
        return None


def _draw(maps: Array, assets: Any) -> Array:
    maps = maps.astype(jnp.int32)
    group = {name: maps[..., index] for index, name in enumerate(FEATURE_MAPS)}
    return rendering.draw(
        terrain=group["terrain"],
        present=group["unit_type"] > 0,
        owner=group["owner"] - 1,
        unit_type=group["unit_type"] - 1,
        pending_kind=group["pending_action_kind"],
        carried_resources=group["carried_resources"],
        hit_points=group["hit_points"],
        assets=assets,
    )


_DRAW_FRAMES = jax.jit(jax.vmap(_draw, in_axes=(0, None)))


class _Host:
    def __init__(self, config: Config, directory: Path) -> None:
        self.config = config
        self.directory = directory
        self.names = _group_names(config)
        self.lock = threading.Lock()
        self.file = (directory / "metrics.jsonl").open("w")
        self.previous: float | None = None
        self.criterion: float | None = None
        self.printed = 0.0
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.drawing: list[concurrent.futures.Future[None]] = []
        self.assets: Any = None
        self.run: Any = None
        if config.wandb_mode != "disabled":
            self.run = _wandb_run(config, directory, config.run_name, "train")
            self.run.define_metric("learner_steps")
            self.run.define_metric("*", step_metric="learner_steps")

    def __call__(self, event: str, run: Any, update: Any, values: Any) -> None:
        del run
        if event == "update":
            self._update(int(update), values)
        elif event == "evaluation":
            self._evaluation(int(update), values)
        else:
            self.drawing.append(self.pool.submit(self._write, int(update), values))

    def _log(self, update: int, logged: Mapping[str, Any]) -> None:
        if self.run is not None:
            with self.lock:
                steps = update * self.config.batch_size
                self.run.log({"learner_steps": steps, **logged})

    def _record(self, record: dict[str, Any], logged: Mapping[str, Any]) -> None:
        with self.lock:
            self.file.write(json.dumps(record) + "\n")
            self.file.flush()
        self._log(record["update"], logged)

    def _update(self, update: int, values: Mapping[str, np.ndarray]) -> None:
        now = time.monotonic()
        previous, self.previous = self.previous, now
        logged: dict[str, Any] = {
            f"losses/{name}": float(values[name])
            for name in (*LOSSES, "explained_variance")
        }
        logged["charts/learning_rate"] = float(values["learning_rate"])
        for name in (
            "episodes",
            "invalid_draws",
            "unselected_slots",
            "overflowed_episodes",
            "non_finite",
        ):
            logged[f"charts/{name}"] = int(values[name])
        if episodes := int(values["episodes"]):
            returns = float(values["episode_return_sum"])
            logged["charts/episodic_return"] = returns / episodes
            logged["charts/episodic_length"] = (
                int(values["episode_length_sum"]) / episodes
            )
        if previous is not None:
            logged["charts/sps"] = self.config.batch_size / (now - previous)
            cycles = self.config.lanes * self.config.num_steps
            logged["charts/env_cycles_per_second"] = cycles / (now - previous)
        for name, row in zip(self.names, values["outcomes"], strict=True):
            for outcome, count in zip(OUTCOMES, row, strict=True):
                logged[f"train/{name}/{outcome}"] = int(count)
            if ended := int(row.sum()):
                logged[f"train/{name}/win_rate"] = int(row[0]) / ended
        record = {
            "event": "update",
            "update": update,
            "learner_steps": update * self.config.batch_size,
            "time": now - PROCESS_START,
            **logged,
        }
        self._record(record, logged)
        if now - self.printed > 10 or update == self.config.updates:
            self.printed = now
            print(
                f"update {update}/{self.config.updates} "
                f"steps {record['learner_steps']} "
                f"{logged.get('charts/sps', float('nan')):.0f} sps "
                f"return {logged.get('charts/episodic_return', float('nan')):.2f}",
                flush=True,
            )

    def _evaluation(self, update: int, values: Mapping[str, Any]) -> None:
        now = time.monotonic()
        roster = bool(values["roster"])
        cases = evaluation_cases(self.config, roster=roster)
        rows = _case_rows(cases, np.asarray(values["outcomes"]))
        record: dict[str, Any] = {
            "event": "evaluation",
            "update": update,
            "learner_steps": update * self.config.batch_size,
            "time": now - PROCESS_START,
            "roster": roster,
            "cases": rows,
            "beats_required": False if roster else beats_required(self.config, rows),
        }
        scope = "roster" if roster else "eval"
        logged: dict[str, Any] = {f"{scope}/beats_required": record["beats_required"]}
        for row in rows:
            win_rate = row["wins"] / row["games"]
            logged[f"{scope}/{row['opponent']}/{row['seat']}/win_rate"] = win_rate
        for opponent, (wins, games) in _pooled(rows).items():
            logged[f"{scope}/{opponent}/win_rate"] = wins / games
        if record["beats_required"] and self.criterion is None:
            self.criterion = record["time_to_criterion"] = record["time"]
            logged["eval/time_to_criterion"] = record["time"]
        self._record(record, logged)
        summary = " ".join(
            f"{opponent} {wins}/{games}"
            for opponent, (wins, games) in _pooled(rows).items()
        )
        print(f"{scope} after update {update}: {summary}", flush=True)
        if self.config.stop_at_criterion and record["beats_required"]:
            print("every required opponent beaten; stopping", flush=True)
            self.close({"stopped_at_criterion": True})
            os._exit(0)

    def _write(self, update: int, values: Mapping[str, Any]) -> None:
        roster = bool(values["roster"])
        opponents = video_opponents(self.config, roster=roster)
        frames, lengths = np.asarray(values["frames"]), np.asarray(values["lengths"])
        scope = "videos/roster" if roster else "videos"
        logged = {}
        with jax.default_device(_host_device()):
            if self.assets is None:
                self.assets = rendering.prepare_assets(_scenario(self.config).bucket)
            for opponent, game, length in zip(opponents, frames, lengths, strict=True):
                chosen = jnp.asarray(game[: int(length) + 1 : self.config.video_stride])
                images = np.asarray(_DRAW_FRAMES(chosen, self.assets))
                name = f"{update:06d}-{'roster-' if roster else ''}{opponent}.gif"
                path = self.directory / "videos" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                iio.imwrite(
                    path,
                    images,
                    extension=".gif",
                    duration=self.config.video_frame_ms,
                    loop=0,
                )
                if self.run is not None:
                    import wandb

                    logged[f"{scope}/{opponent}"] = wandb.Video(
                        str(path), caption=f"update {update}", format="gif"
                    )
        self._log(update, logged)

    def close(self, summary: Mapping[str, Any]) -> None:
        self.pool.shutdown(wait=True)
        for future in self.drawing:
            future.result()
        self.file.close()
        if self.run is not None:
            for name, value in summary.items():
                self.run.summary[name] = value
            if self.criterion is not None:
                self.run.summary["time_to_criterion"] = self.criterion
            self.run.finish()


def train(config: Config) -> Path:
    if config.videos and config.eval_every and _host_device() is None:
        raise ConfigError(NO_HOST_DEVICE)
    if config.eval_every and config.roster_every and not roster_runs():
        print(
            "the roster evaluation runs on CUDA's Kernel path only and is "
            "skipped: the kernels are off or unavailable",
            flush=True,
        )
    if not config.run_name:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        config = dataclasses.replace(
            config,
            run_name=f"{config.preset}-{config.lanes}-s{config.seed}-{stamp}",
        )
    directory = Path(config.output_dir) / config.run_name
    directory.mkdir(parents=True, exist_ok=True)
    host = _Host(config, directory)
    training = Training(config, host)
    for name, values in (("config", plain(config)), ("derived", training.describe())):
        (directory / f"{name}.yaml").write_text(yaml.safe_dump(values, sort_keys=False))
    started = time.monotonic()
    result = jax.device_get(compiled(training.train)(jax.random.key(config.seed)))
    seconds = time.monotonic() - started
    save_params(
        directory / "params.msgpack",
        result.params,
        config,
        training.num_updates,
        result.eval_key,
    )
    peak = (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use")
    host.close({"run_seconds": seconds, "peak_device_bytes": peak})
    print(f"trained {training.num_updates} updates in {seconds:.0f} s: {directory}")
    return directory


def evaluate_params(path: Path, overrides: Sequence[str] = ()) -> dict[str, Any]:
    params, config, update, key = load_params(path)
    config = override(config, overrides)
    evaluation = Evaluation(config, roster=False, videos=False)
    counts = compiled(lambda params, key: evaluation.play(params, key)[0])(
        params, jax.random.wrap_key_data(jnp.asarray(key))
    )
    rows = _case_rows(evaluation.cases, np.asarray(counts))
    table = {"params": str(path), "update": update, "rows": rows}
    Path(path).with_suffix(".evaluation.json").write_text(json.dumps(table, indent=2))
    if config.wandb_mode != "disabled":
        import wandb

        run = _wandb_run(
            config, Path(path).parent, f"{config.run_name}-evaluate", "evaluate"
        )
        columns: list[str | int] = list(rows[0])
        data = [[row[str(column)] for column in columns] for row in rows]
        run.log({"evaluation": wandb.Table(columns=columns, data=data)})
        run.finish()
    for row in rows:
        print(
            f"{row['opponent']:>16} {row['seat']}: {row['wins']} won, "
            f"{row['draws']} drawn, {row['losses']} lost of {row['games']}"
        )
    return table


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m baselines.ppo")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="train from a preset")
    training.add_argument("--preset", choices=list(PRESETS), default="p8-8x8")
    training.add_argument("overrides", nargs="*", metavar="key=value")
    evaluation = commands.add_parser("evaluate", help="evaluate saved parameters")
    evaluation.add_argument("params", type=Path)
    evaluation.add_argument("overrides", nargs="*", metavar="key=value")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "train":
            train(load_config(arguments.preset, arguments.overrides))
        else:
            evaluate_params(arguments.params, arguments.overrides)
    except ConfigError as error:
        parser.error(str(error))
    return 0


def train_main() -> int:
    return main(["train", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
