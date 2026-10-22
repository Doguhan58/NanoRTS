# NanoRTS
NanoRTS is a real-time strategy environment written entirely in [JAX](https://github.com/google/jax). It reimplements the game of [MicroRTS](https://github.com/Farama-Foundation/MicroRTS): workers harvest resources, bases and barracks produce units, and two players fight until one is eliminated or the time limit is reached. The gameplay follows the Java engine's rules, including the order in which actions are admitted, how conflicts between units are resolved and how long each action takes.

The whole game is a pure function, so `jit`, `vmap` and `lax.scan` work on it. Thousands of games run in parallel on one GPU.

NanoRTS is a two-player [JaxMARL](https://github.com/FLAIROx/JaxMARL) environment, shaped like SMAX. It also ships:
- a [MicroRTS-Py](https://github.com/Farama-Foundation/MicroRTS-Py) GridNet adapter (for compatibility, not recommended for speed)
- JAX ports of 15 MicroRTS bots, including competition winners
- a pixel renderer that runs on the device


# Installation
NanoRTS needs Python 3.13 on Linux (x86_64 or aarch64). Install it from GitHub (currently not on PyPI, so install into your venv):
```
pip install git+https://github.com/Doguhan58/NanoRTS.git
```
Or to work with the repository itself, clone it and let [uv](https://github.com/astral-sh/uv) create the environment from the lock file:
```
git clone https://github.com/Doguhan58/NanoRTS.git
cd NanoRTS
uv sync
```

## GPU
The dependencies include `jax[cuda13]`, so JAX runs on an NVIDIA GPU with a CUDA 13 driver when there is one, and on the CPU otherwise. To force the CPU, set:
```
export JAX_PLATFORMS=cpu
```

# Usage
NanoRTS follows the JaxMARL `MultiAgentEnv` interface:
```python
import jax
import nanorts

scenario = nanorts.map_name_to_scenario("basesWorkers8x8")
env = nanorts.make("NanoRTS", scenario=scenario)

key = jax.random.key(0)
key, reset_key = jax.random.split(key)
obs, state = env.reset(reset_key)

masks = env.get_avail_actions(state)
actions = {
    agent: sample_actions(agent_key, masks[agent], env.layout.action_widths)
    for agent, agent_key in zip(env.agents, jax.random.split(key, 2))
}

obs, state, rewards, dones, infos = env.step(key, state, actions)
```
`sample_actions` draws one legal action per unit from the masks. A policy replaces it in training:
```python
import jax.numpy as jnp
import numpy as np

def sample_actions(key, mask, widths):
    bounds = np.cumsum((0, *widths))
    keys = jax.random.split(key, len(widths))
    factors = [
        jax.random.categorical(k, jnp.where(mask[:, lo:hi], 0.0, -jnp.inf))
        for k, lo, hi in zip(keys, bounds[:-1], bounds[1:])
    ]
    return jnp.stack(factors, axis=-1).astype(jnp.int32)
```



# The Environment
## Scenarios
The maps of MicroRTS's [`maps/`](https://github.com/Farama-Foundation/MicroRTS/tree/master/maps) folder ship with the package, from 4 x 4 up to the 128 x 128 BroodWar maps, under `nanorts/maps/`. A scenario is named by its file name without `.xml`, for example `"basesWorkers12x12"`, `"GardenOfWar64x64"` or `"(4)Fortress.scxA"`, and `nanorts.scenario_names()` lists all 138.

The bots and the speed figures were measured on the two release maps, under `VERSION_ORIGINAL`:

| Scenario | Field | Unit slots per player |
|:--|:--|:--|
| `basesWorkers8x8` | 8 x 8 | 47 |
| `basesWorkers16x16` | 16 x 16 | 70 |

`map_name_to_scenario` takes the rules as keywords:
```python
from nanorts.model import ConflictStrategy, UnitTypeTableVersion

scenario = nanorts.map_name_to_scenario(
    "basesWorkers16x16",
    utt=UnitTypeTableVersion.ORIGINAL,
    conflict_strategy=ConflictStrategy.CANCEL_RANDOM,
    max_cycles=3000,
)
```
- `utt` is the unit type table:
  - `VERSION_ORIGINAL_FINETUNED` (the default) is the table of the microRTS competition's classic track.
  - `VERSION_ORIGINAL` is Java's and MicroRTS-Py's default.
  - They differ in five values. Under the default, Heavy costs 3 (instead of 2), has 8 hit points (instead of 4) and moves in 10 cycles (instead of 12), a Base takes 200 cycles to build (instead of 250), and a Barracks 100 (instead of 200).
- `conflict_strategy` decides two units that claim the same cell in the same cycle: `CANCEL_BOTH` (the default), `CANCEL_RANDOM` or `CANCEL_ALTERNATING`.
- `max_cycles` is the time limit, the default is 5000.
- `capacity_profile=False` gives 16x16 the larger slot count that can never overflow (107 instead of 70, see [Gotchas](#-gotchas)).

## Agents and Observations
The agents are `player_0` and `player_1`. Each agent observes the board from its own point of view, so its own units always appear as owner 1.

| `surface` | Observation | Shape |
|:--|:--|:--|
| `"symbolic"` (default) | `obs[agent].feature_maps` | `uint8[H, W, 6]` |
| | `obs[agent].scalars` | `int32[4]` |
| `"pixels"` | `obs[agent].image` | `uint8[16H, 16W, 3]` |
| | `obs[agent].scalars` | `int32[4]` |

The six feature maps are MicroRTS's own, unnormalized:
1. hit points
2. carried resources, or the size of a resource pile
3. owner: 0 empty or neutral, 1 self, 2 opponent
4. unit type + 1
5. the kind of action the unit is executing
6. terrain (1 is a wall)

The scalars are `[own resources, opponent resources, cycle, alternating conflict count]`.

`partial_observability=True` hides everything outside the agent's units' sight radius.

## Actions
An action is one row per unit slot: `actions[agent]` is `int32[C, 7]`, where `C` is the scenario's slot count. Row `i` commands the unit in the agent's slot `i`.

| Column | Factor | Width | Values |
|:--|:--|:--|:--|
| 0 | `kind` | 6 | 0 NONE, 1 MOVE, 2 HARVEST, 3 RETURN, 4 PRODUCE, 5 ATTACK |
| 1 | `move_direction` | 4 | 0 up, 1 right, 2 down, 3 left |
| 2 | `harvest_direction` | 4 | 0 up, 1 right, 2 down, 3 left |
| 3 | `return_direction` | 4 | 0 up, 1 right, 2 down, 3 left |
| 4 | `produce_direction` | 4 | 0 up, 1 right, 2 down, 3 left|
| 5 | `produce_type` | 7 | 0 Resource, 1 Base, 2 Barracks, 3 Worker, 4 Light, 5 Heavy, 6 Ranged |
| 6 | `attack_offset` | 49 | target cell in the 7 x 7 window around the unit, `(dy + 3) * 7 + (dx + 3)` |

Each kind reads only its own parameter column. the other columns are ignored but must stay inside their ranges. Rows for empty slots and for busy units are ignored, so a unit keeps running the action it already has. `NONE` waits one cycle. `env.layout.action_widths` lists the widths, and `env.layout.actions_shape` gives the shape.

## Masks and Slot Cells
- `env.get_avail_actions(state)[agent]` is `bool[C, 78]`: each row's seven factor masks concatenated in column order. A row whose slot holds no unit ready for a new command has only the `NONE` bit set. Masks are per factor, as in MicroRTS-Py: sample the kind, then the parameter of that kind.
- `env.get_slot_cells(state)[agent]` is `int32[C]`: the cell `y * W + x` where each slot's unit stands, or `-1` for an empty slot. A policy that reads a spatial feature map per unit uses it to find the unit.

A legal-looking action can still fail, for example when two units move into the same cell. A failed action becomes a timed `NONE`, exactly as in Java.

## Rewards, Dones and Infos
- `rewards[agent]` is the sparse game result, paid on the step that ends the episode: `+1` for a win, and `-1` for a loss or a draw. A draw is reaching the time limit, or both players being eliminated in the same cycle. Every other step pays `0`.
- `dones` has `player_0`, `player_1` and `__all__`, all equal.
- `infos` holds:
  - `terminated` (a player was eliminated) and `truncated` (the time limit was reached);
  - `winner` (`0`, `1` or `-1`) and `spawn_overflowed`;
  - `final_observation` and `final_valid`: the observation of the state that ended the episode, before the automatic reset.
- With `reward_components=True`, `infos["reward_components"][agent]` holds MicroRTS-Py's six shaped rewards as `float32[6]`, in the order of `nanorts.adapters.jaxmarl.REWARD_COMPONENTS`: `win_loss`, `resource_gather`, `produce_worker`, `produce_building`, `attack`, `produce_combat_unit`. Weighting them is up to you; `rewards` stays the sparse result. These components follow MicroRTS-Py exactly, so `win_loss` pays `0` at the time limit.

Episodes reset automatically inside `step`.

## Options
```python
env = nanorts.make(
    "NanoRTS",
    scenario=scenario,
    utt=None,                     # keep the scenario's unit type table, or a UnitTypeTableVersion
    surface="symbolic",           # or "pixels"
    partial_observability=False,  # fog of war
    final_observation=True,       # False drops infos["final_observation"] and the work of computing it
    feature_dtype=None,           # uint8 by default; jnp.int16 or jnp.int32
    reward_components=False,      # add MicroRTS-Py's shaped reward components
)
```

# Running Thousands of Environments
Vectorize with `vmap` and roll out with `lax.scan`. Give every environment its own reset key: the reset key seeds that environment's whole stream of games.
```python
NUM_ENVS, NUM_STEPS = 4096, 128
env = nanorts.make("NanoRTS", scenario=scenario, final_observation=False)
widths = env.layout.action_widths

def rollout(key):
    key, reset_key = jax.random.split(key)
    obs, state = jax.vmap(env.reset)(jax.random.split(reset_key, NUM_ENVS))

    def one_step(state, key):
        masks = jax.vmap(env.get_avail_actions)(state)
        actions = {
            agent: jax.vmap(sample_actions, in_axes=(0, 0, None))(
                jax.random.split(agent_key, NUM_ENVS), masks[agent], widths
            )
            for agent, agent_key in zip(env.agents, jax.random.split(key, 2))
        }
        obs, state, rewards, dones, infos = jax.vmap(env.step)(
            jax.random.split(key, NUM_ENVS), state, actions
        )
        return state, (rewards["player_0"], dones["__all__"])

    return jax.lax.scan(one_step, state, jax.random.split(key, NUM_STEPS))

rollout = jax.jit(
    rollout,
    compiler_options={
        **nanorts.RECOMMENDED_COMPILER_OPTIONS,
        **nanorts.DOUBLE_BUFFER_COMPILER_OPTIONS,
    },
)
state, (rewards, dones) = rollout(jax.random.key(0))
```
On CUDA, the compiler options do two things:
- `RECOMMENDED_COMPILER_OPTIONS` captures every loop of a step in one CUDA graph, and stops XLA from rematerializing a buffer that a step updates in place.
- `DOUBLE_BUFFER_COMPILER_OPTIONS` runs two iterations of each fixed-length loop per trip.

Pass them to your outermost `jax.jit` only (see [Gotchas](#-gotchas)).

# Training Against Bots
`nanorts.bots` has JAX ports of the MicroRTS Java AIs. MicroRTS-Py uses the same names.

| Bot | Java AI |
|:--|:--|
| `passiveAI` | `PassiveAI`: every unit waits |
| `randomAI` | `RandomBiasedSingleUnitAI`, MicroRTS-Py's `randomAI` |
| `randomBiasedAI` | `RandomBiasedAI` |
| `workerRushAI`, `lightRushAI`, `heavyRush`, `rangedRush` | `WorkerRush`, `LightRush`, `HeavyRush`, `RangedRush` with A* pathfinding |
| `workerDefense`, `lightDefense`, `heavyDefense`, `rangedDefense` | `WorkerDefense`, `LightDefense`, `HeavyDefense`, `RangedDefense` with A* pathfinding |
| `coacAI` | `CoacAI`, 2020 microRTS competition winner |
| `mayari` | `mayari`, 2021 competition winner (maps narrower than 24 cells) |
| `obiBotKenobi` | `ObiBotKenobi`, 2023 competition entry |
| `tma` | `TMA`, 2024 competition winner |

`bots.ROSTER` maps each name to its bot.

## One Learner Against a Bot
`SeatEnv` is a one-agent JaxMARL environment. The agent `learner` plays one seat and the bot plays the other. `seat` is `"player_0"`, `"player_1"` or `"drawn"`; `"drawn"` picks a seat at random each episode.
```python
from nanorts import bots

env = bots.SeatEnv(bots.coacAI, scenario=scenario, seat="drawn")
obs, state = env.reset(key)
mask = env.get_avail_actions(state)["learner"]
cells = env.get_slot_cells(state)["learner"]
obs, state, rewards, dones, infos = env.step(key, state, {"learner": actions})
```
`SeatEnv` takes the same keywords as `nanorts.make`.

## Mixing Opponents in One Batch
`GroupMix` steps groups of environments inside one `jit`. Each group is either a `SeatEnv` with its bot or a two-player `NanoRTS` for self-play. Each bot runs only on its own group's lanes.

The batch is in learner rows: a seat lane is one row, and a self-play lane is two (one per player).
```python
mix = bots.GroupMix([
    bots.Group(nanorts.make("NanoRTS", scenario=scenario), 256),
    bots.Group(bots.SeatEnv(bots.workerRushAI, scenario=scenario, seat="drawn"), 512),
    bots.Group(bots.SeatEnv(bots.coacAI, scenario=scenario, seat="drawn"), 512),
])
rows, states = mix.reset(jax.random.split(key, mix.lanes))
masks = mix.get_avail_actions(states)                  # bool[mix.rows, C, 78]
cells = mix.get_slot_cells(states)                     # int32[mix.rows, C]
rows, states, rewards, dones, infos = mix.step(key, states, row_actions)
```
`mix.slices()` gives each group's rows. Every group must share the scenario and the options. On the CUDA Kernel path, all groups' Core steps run as one kernel launch.

# Custom Maps and Rendering
Any MicroRTS map XML loads with the same keywords as the shipped scenarios:
```python
from nanorts import content, rendering

game = content.load_game("path/to/map.xml", max_cycles=3000)
env = nanorts.make("NanoRTS", scenario=game)
```
`rendering.render` draws a state on the device:
```python
assets = rendering.prepare_assets(game.bucket, cell_pixels=32)
obs, state = env.reset(key)
image = rendering.render(state.core_state, game, assets)  # uint8[32H, 32W, 3]
```
To train on pixels without storing images, step the symbolic surface. Then draw images from the feature maps inside your own `jit` with `nanorts.observations.pixels_from_symbolic(feature_maps, assets)`.

# Play
To play NanoRTS by hand, run:
```
play
```
(`uv run play` from a clone.)

The window shows the board. The terminal shows the clock, both players' resources, and what the engine did with each command. Press F1 for the controls:

| Key | Action |
|:--|:--|
| click | select a unit |
| tab / 1 / 2 | switch the player you command |
| arrows / wasd | MOVE |
| h / r / p | HARVEST / RETURN / PRODUCE along the last direction |
| t | cycle the unit type PRODUCE builds |
| x, then click | ATTACK the clicked cell |
| n | NONE for 10 cycles |
| space / enter | step one cycle / run continuously |
| [ / ] | slower / faster |
| b | random opponent on / off |
| ctrl+r | restart |
| g / m | save a frame / record a GIF |
| esc / q | quit |

Options include `--map`, `--seed`, `--max-cycles`, `--utt`, `--conflict-strategy`, `--cell-pixels` and `--device cuda`; see `play --help`. The first frame takes a few seconds to compile.


# MicroRTS-Py GridNet
`nanorts.adapters.gridnet` reproduces MicroRTS-Py's full-observation GridNet interface: one game with both players stepped together, actions per board cell, 29 one-hot planes, and per-cell masks. MicroRTS-Py plays `VERSION_ORIGINAL`, so build the scenario with that table to play its rules. It is a set of pure functions over a prepared layout:
```python
from nanorts.adapters import gridnet
from nanorts.model import UnitTypeTableVersion

game = nanorts.map_name_to_scenario("basesWorkers8x8", utt=UnitTypeTableVersion.ORIGINAL)
layout = gridnet.prepare(game)
env_state, observation = gridnet.reset(jax.random.key(0), game, layout=layout)
# observation.planes: int32[2, H, W, 29], observation.action_mask: int32[2, H, W, 78]
result = gridnet.step(env_state, actions, game, layout=layout)  # actions: int32[2, H, W, 7]
```
The bots seat against GridNet through `bots.reset_gridnet_seat`, `bots.step_gridnet_seat` and `bots.GridNetMix`.


# Performance
These numbers are environment steps per second, summed over 4,096 parallel games. One step is one game cycle for both players. Every game is played from mid-game positions, from its own masks with sampled actions, so the games diverge as they would in training.

Measured on one RTX 4090 with JAX 0.11.1, `VERSION_ORIGINAL` and both compiler option sets from [Running Thousands of Environments](#-running-thousands-of-environments):

| Scenario | Surface | Steps per second |
|:--|:--|--:|
| `basesWorkers8x8` | symbolic | 3.5 M |
| `basesWorkers8x8` | symbolic, partial observability | 3.3 M |
| `basesWorkers8x8` | pixels | 1.6 M |
| `basesWorkers16x16` | symbolic | 2.1 M |
| `basesWorkers16x16` | symbolic, partial observability | 1.8 M |

## CUDA Kernels
On NVIDIA GPUs, NanoRTS replaces parts of the step, and of the bots, with hand-written CUDA kernels. They give the same results as the JAX code and are on by default.

On the first import on Linux, the kernels are built with the `nvcc` that NanoRTS installs and the system `g++`, into `~/.cache/nanorts/kernels` (set `NANORTS_KERNEL_CACHE` to change it). The build takes a few minutes of CPU time, and it also runs on a machine without an NVIDIA GPU, where the JAX code runs anyway.

If the build fails, NanoRTS warns once on a GPU and runs the JAX code. To run the JAX code and skip the build, turn the kernels off before importing NanoRTS:
```
export NANORTS_KERNELS=0
```

# Gotchas
### Keys
- The key passed to `reset` seeds the environment for its whole life. Every later episode and every random draw derives from that key and the episode count. If two environments get the same reset key, they play identical games.
- `step` accepts a key for interface compatibility but never reads it.
- `step` does not take a `reset_state`.

### Compiler options
Pass `RECOMMENDED_COMPILER_OPTIONS` only to your outermost `jax.jit`. If a `jit` that carries `compiler_options` is traced inside another `jit`, JAX raises a `ValueError`.

### Slot capacity on 16x16
By default, `basesWorkers16x16` gives each player 70 unit slots. That number was measured under `VERSION_ORIGINAL` and is not a proven bound. If a game ever needs more, `infos["spawn_overflowed"]` is set for the rest of that episode. Use `map_name_to_scenario(..., capacity_profile=False)` for 107 slots, which cannot overflow, but is significantly slower (1.6x times measured).

### Unit type table
The default table is `VERSION_ORIGINAL_FINETUNED`, not the `VERSION_ORIGINAL` of Java's `new UnitTypeTable()` and of MicroRTS-Py. Pass `utt=UnitTypeTableVersion.ORIGINAL` to `map_name_to_scenario`, `load_game` or `nanorts.make` to compare with MicroRTS-Py. Units placed by a map file keep the hit points the file declares, as in Java, so the Heavies of the `melee*Mixed*` maps start with 4 of their 8 hit points.

### Feature map dtype
Feature maps are `uint8` by default, and arithmetic on them wraps at 255. Cast them before computing with them, or pass `feature_dtype=jnp.int32`.

### Invalid actions
An illegal action does not raise an error: it becomes a timed `NONE`, as in Java. Values outside a column's range are not checked inside `jit`. While debugging, check them on the host with `nanorts.adapters.jaxmarl.check_actions(actions, layout=env.layout)`.

### Memory
- `infos["final_observation"]` stores a second set of observations every step. Pass `final_observation=False` if you don't need it.
- Pixel observations are large: with final observations on, 4,096 16x16 games produce about 3 GB of images per step. Prefer the symbolic surface together with `pixels_from_symbolic`.

### Import time
`import nanorts` is cheap once the [CUDA kernels](#cuda-kernels) are built; the first import on Linux builds them. The first use of `nanorts.make` imports `jaxmarl`, which takes about a second.

