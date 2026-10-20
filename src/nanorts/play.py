from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np
import pygame
from jax import Array

from nanorts import content, core, model, rendering

DEFAULT_MAP = content.shipped_map("basesWorkers8x8")

KIND_NAMES: dict[int, str] = {
    model.KIND_ABSENT: "none pending",
    model.KIND_NONE: "NONE",
    model.KIND_MOVE: "MOVE",
    model.KIND_HARVEST: "HARVEST",
    model.KIND_RETURN: "RETURN",
    model.KIND_PRODUCE: "PRODUCE",
    model.KIND_ATTACK: "ATTACK",
}

DIRECTION_NAMES: tuple[str, ...] = ("up", "right", "down", "left")

PRODUCIBLE: tuple[int, ...] = (
    model.TYPE_WORKER,
    model.TYPE_LIGHT,
    model.TYPE_HEAVY,
    model.TYPE_RANGED,
    model.TYPE_BARRACKS,
    model.TYPE_BASE,
)

GOOD = (122, 214, 148)
SELECTION = (255, 236, 120)
QUEUED = (140, 200, 255)

STYLES: dict[str, str] = {
    "good": "\033[32m",
    "warn": "\033[33m",
    "bad": "\033[31m",
    "muted": "\033[90m",
    "head": "\033[1;36m",
    "bold": "\033[1m",
}
RESET = "\033[0m"
COLOUR = sys.stdout.isatty()

Stepper = Callable[
    [Array, model.State, model.Commands, model.GameConfig], model.StepResult
]
Legalities = Callable[[model.State, model.GameConfig], model.Legality]
Renderer = Callable[[model.State, model.GameConfig, rendering.RenderAssets], Array]


def paint(text: str, style: str) -> str:
    if not COLOUR:
        return text
    return f"{STYLES[style]}{text}{RESET}"


def say(text: str = "", style: str | None = None) -> None:
    print(text if style is None else paint(text, style), flush=True)


@dataclasses.dataclass(frozen=True)
class Request:
    kind: int
    direction: int = model.ABSENT
    none_duration: int = 0
    target_cell: int = model.ABSENT
    produced_type: int = model.ABSENT


@dataclasses.dataclass(frozen=True)
class Readback:
    active: np.ndarray
    birth: np.ndarray
    unit_type: np.ndarray
    owner: np.ndarray
    cell: np.ndarray
    hit_points: np.ndarray
    carried_resources: np.ndarray
    pending_kind: np.ndarray
    pending_direction: np.ndarray
    pending_target_cell: np.ndarray
    pending_produced_type: np.ndarray
    pending_issued_at: np.ndarray
    pending_completes_at: np.ndarray
    pending_order: np.ndarray
    entity_at: np.ndarray
    available: np.ndarray
    time: int
    next_birth: int
    resources: tuple[int, ...]


def readback(state: model.State, available: Array) -> Readback:
    return Readback(
        active=np.asarray(state.active),
        birth=np.asarray(state.birth),
        unit_type=np.asarray(state.unit_type),
        owner=np.asarray(state.owner),
        cell=np.asarray(state.cell),
        hit_points=np.asarray(state.hit_points),
        carried_resources=np.asarray(state.carried_resources),
        pending_kind=np.asarray(state.pending_kind),
        pending_direction=np.asarray(state.pending_direction),
        pending_target_cell=np.asarray(state.pending_target_cell),
        pending_produced_type=np.asarray(state.pending_produced_type),
        pending_issued_at=np.asarray(state.pending_issued_at),
        pending_completes_at=np.asarray(state.pending_completes_at),
        pending_order=np.asarray(state.pending_order),
        entity_at=np.asarray(state.entity_at),
        available=np.asarray(available),
        time=int(state.time),
        next_birth=int(state.next_birth),
        resources=tuple(int(value) for value in state.player_resources))


def placement_of(slot: int, bucket: model.BucketSpec) -> tuple[int, int] | None:
    for player in range(model.PLAYER_COUNT):
        start = bucket.player_slot_start(player)

        if start <= slot < start + bucket.per_player_capacity:
            return player, slot - start

    return None


def cell_xy(cell: int, width: int) -> tuple[int, int]:
    return cell % width, cell // width


def commands_of(requests: Mapping[int, Request], bucket: model.BucketSpec) -> model.Commands:
    table = model.empty(model.Commands, bucket)
    names = ("kind", "direction", "none_duration", "target_cell", "produced_type")

    written = {name: np.asarray(getattr(table, name)).copy() for name in names}

    for slot, request in requests.items():
        placement = placement_of(slot, bucket)

        if placement is None:
            continue
        player, local = placement
        for name in names:
            written[name][player, local] = getattr(request, name)

    return dataclasses.replace(table, **{name: jnp.asarray(value) for name, value in written.items()})


def patrolling(view: Readback, bucket: model.BucketSpec) -> dict[int, Request]:
    requests: dict[int, Request] = {}
    for player in range(model.PLAYER_COUNT):
        for slot in bucket.player_slots(player):
            local = slot - bucket.player_slot_start(player)

            if not view.available[player, local]:
                continue

            _, y = cell_xy(int(view.cell[slot]), bucket.width)

            heading = (model.DIRECTION_DOWN if y < bucket.height - 1 else model.DIRECTION_UP)
            requests[slot] = Request(kind=model.KIND_MOVE, direction=heading)
    return requests


def wandering(view: Readback, bucket: model.BucketSpec, player: int, generator: np.random.Generator) -> dict[int, Request]:
    requests: dict[int, Request] = {}
    for slot in bucket.player_slots(player):
        local = slot - bucket.player_slot_start(player)

        if not view.available[player, local]:
            continue

        requests[slot] = Request(
            kind=model.KIND_MOVE,
            direction=int(generator.integers(model.DIRECTION_COUNT)))

    return requests


def pending_payload(view: Readback, slot: int) -> str:
    kind = int(view.pending_kind[slot])

    if kind in (model.KIND_MOVE, model.KIND_HARVEST, model.KIND_RETURN):
        return DIRECTION_NAMES[int(view.pending_direction[slot])]

    if kind == model.KIND_PRODUCE:
        produced = int(view.pending_produced_type[slot])
        direction = DIRECTION_NAMES[int(view.pending_direction[slot])]
        return f"{model.TYPE_NAMES[produced]} {direction}"

    if kind == model.KIND_ATTACK:
        x, y = cell_xy(int(view.pending_target_cell[slot]), view.entity_at.shape[1])
        return f"({x},{y})"

    if kind == model.KIND_NONE:
        stored = int(view.pending_completes_at[slot]) - int(view.pending_issued_at[slot])
        return f"({stored})"

    return ""


def request_payload(request: Request, width: int) -> str:
    if request.kind in (model.KIND_MOVE, model.KIND_HARVEST, model.KIND_RETURN):
        return DIRECTION_NAMES[request.direction]

    if request.kind == model.KIND_PRODUCE:
        produced = model.TYPE_NAMES[request.produced_type]
        return f"{produced} {DIRECTION_NAMES[request.direction]}"

    if request.kind == model.KIND_ATTACK:
        x, y = cell_xy(request.target_cell, width)
        return f"({x},{y})"

    if request.kind == model.KIND_NONE:
        return f"({request.none_duration})"

    return ""


def asked_for(request: Request, width: int) -> str:
    return f"{KIND_NAMES[request.kind]} {request_payload(request, width)}".strip()


def _issue_line(was: Readback, now: Readback, request: Request | None, slot: int, label: str) -> tuple[str, str] | None:
    width = was.entity_at.shape[1]
    pending = int(now.pending_kind[slot])

    if request is not None and pending == model.KIND_ABSENT:
        asked = asked_for(request, width)
        return f"{label} {asked}  admitted, completed the same cycle", "good"

    if int(now.pending_issued_at[slot]) != was.time or pending == model.KIND_ABSENT:
        return None

    duration = int(now.pending_completes_at[slot]) - int(now.pending_issued_at[slot])
    completes = int(now.pending_completes_at[slot])

    if request is not None and pending == request.kind:
        asked = asked_for(request, width)
        return (
            f"{label} {asked}  admitted, eta {duration}, completes at {completes}",
            "bold")

    if pending != model.KIND_NONE:
        payload = pending_payload(now, slot)
        return f"{label} {KIND_NAMES[pending]} {payload}  admitted", "bold"

    if duration == -1:
        return f"{label} -> NONE(-1)  cross-cycle Reservation conflict", "bad"

    if request is not None:
        asked = asked_for(request, width)
        return f"{label} {asked}  refused or cancelled -> NONE({duration})", "bad"

    return f"{label} -> NONE({duration})  cancelled by a conflict", "bad"


def _completion_line(was: Readback, now: Readback, slot: int, label: str) -> tuple[str, str] | None:
    previous = int(was.pending_kind[slot])

    if previous == model.KIND_ABSENT:
        return None

    if int(was.pending_completes_at[slot]) > now.time:
        return None

    name = KIND_NAMES[previous]

    if previous == model.KIND_MOVE and int(was.cell[slot]) != int(now.cell[slot]):
        x, y = cell_xy(int(now.cell[slot]), now.entity_at.shape[1])
        return f"{label} MOVE completed -> ({x},{y})", "good"

    if previous == model.KIND_ATTACK:
        return f"{label} {_attack_outcome(was, now, slot)}", "good"

    if previous in (model.KIND_HARVEST, model.KIND_RETURN):
        return f"{label} {_resource_outcome(was, now, slot, previous)}", "good"

    if previous == model.KIND_PRODUCE:
        return f"{label} {_spawn_outcome(was, now, slot)}", "good"

    return f"{label} {name} completed", "good"


def _attack_outcome(was: Readback, now: Readback, slot: int) -> str:
    width = now.entity_at.shape[1]
    target = int(was.pending_target_cell[slot])
    x, y = cell_xy(target, width)
    occupant = int(was.entity_at.reshape(-1)[target])

    if occupant == model.ABSENT:
        return f"ATTACK completed -> ({x},{y}) was already empty"

    lost = int(was.hit_points[occupant]) - int(now.hit_points[occupant])

    if not now.active[occupant]:
        return f"ATTACK completed -> ({x},{y}) s#{occupant} removed"

    if lost > 0:
        return (
            f"ATTACK completed -> ({x},{y}) s#{occupant} "
            f"{int(was.hit_points[occupant])} -> {int(now.hit_points[occupant])} hp")

    return f"ATTACK completed -> ({x},{y}) emptied before it resolved"


def _resource_outcome(was: Readback, now: Readback, slot: int, kind: int) -> str:
    name = KIND_NAMES[kind]
    direction = DIRECTION_NAMES[int(was.pending_direction[slot])]
    before, after = int(was.carried_resources[slot]), int(now.carried_resources[slot])

    if kind == model.KIND_HARVEST and after > before:
        return f"{name} {direction} completed -> carrying {after}"

    if kind == model.KIND_RETURN and after < before:
        owner = int(was.owner[slot])
        return (
            f"{name} {direction} completed -> p{owner} "
            f"{was.resources[owner]} -> {now.resources[owner]}")

    if not now.active[slot]:
        return f"{name} {direction} completed by a removed actor"

    return f"{name} {direction} completed -> nothing there to {name.lower()}"


def _spawn_outcome(was: Readback, now: Readback, slot: int) -> str:
    width = now.entity_at.shape[1]
    produced = model.TYPE_NAMES[int(was.pending_produced_type[slot])]
    direction = DIRECTION_NAMES[int(was.pending_direction[slot])]
    x, y = cell_xy(int(was.cell[slot]), width)
    dx, dy = model.DIRECTION_OFFSETS[int(was.pending_direction[slot])]
    tx, ty = x + dx, y + dy
    newborn = (
        int(now.entity_at[ty, tx])
        if 0 <= tx < width and 0 <= ty < now.entity_at.shape[0]
        else model.ABSENT)
    owner = int(was.owner[slot])

    if newborn != model.ABSENT and int(now.birth[newborn]) >= int(was.next_birth):
        return (
            f"PRODUCE {produced} {direction} completed -> s#{newborn} born at "
            f"({tx},{ty}), p{owner} {was.resources[owner]} -> {now.resources[owner]}")

    return f"PRODUCE {produced} {direction} completed -> starved, p{owner} keeps {was.resources[owner]}"


def transitions(was: Readback, now: Readback, requests: Mapping[int, Request], bucket: model.BucketSpec) -> list[tuple[str, str]]:
    lines: list[tuple[str, str]] = []
    for player in range(model.PLAYER_COUNT):
        start = bucket.player_slot_start(player)
        for slot in bucket.player_slots(player):
            if not was.active[slot] and not now.active[slot]:
                continue

            address = f"p{player} s#{slot - start:<3}"
            issue = _issue_line(was, now, requests.get(slot), slot, f"t={was.time:<5} {address}")

            if issue is not None:
                lines.append(issue)

            done = _completion_line(was, now, slot, f"t={now.time:<5} {address}")

            if done is not None:
                lines.append(done)

            if was.active[slot] and not now.active[slot]:
                x, y = cell_xy(int(was.cell[slot]), now.entity_at.shape[1])
                lines.append((f"t={now.time:<5} {address} removed at ({x},{y})", "warn"))

    return lines


@dataclasses.dataclass
class Session:
    game: model.GameConfig
    assets: rendering.RenderAssets
    key: Array
    state: model.State
    view: Readback
    stepper: Stepper
    legality: Legalities
    renderer: Renderer
    requests: dict[int, Request] = dataclasses.field(default_factory=dict)
    selected: int | None = None
    player: int = 0
    heading: int = model.DIRECTION_RIGHT
    produced: int = model.TYPE_WORKER
    running: bool = False
    aiming: bool = False
    opponent: bool = False
    speed: float = 4.0
    record: Path = Path("play.gif")
    fps: int = 12
    frames: list[np.ndarray] | None = None
    generator: np.random.Generator = dataclasses.field(default_factory=lambda: np.random.default_rng(0))

    @property
    def bucket(self) -> model.BucketSpec:
        return self.game.bucket

    def look(self, state: model.State) -> Readback:
        return readback(state, self.legality(state, self.game).available)

    def restart(self) -> None:
        self.state = core.reset(self.key, self.game)
        self.view = self.look(self.state)
        self.requests.clear()
        self.selected = None
        say("episode reset to cycle 0", "head")
        reached = self.boundary()

        if reached is not None:
            say(f"  {reached} - this content is decided before its first cycle", "warn")

    def select(self, x: int, y: int) -> None:
        if not (0 <= x < self.bucket.width and 0 <= y < self.bucket.height):
            return

        slot = int(self.view.entity_at[y, x])
        self.selected = None if slot == model.ABSENT else slot
        report_selection(self)

    def queue(self, request: Request) -> None:
        if self.selected is None:
            say("nothing selected - click a unit first", "muted")
            return

        placement = placement_of(self.selected, self.bucket)

        if placement is None or placement[0] != self.player:
            say(
                f"slot {self.selected} is not player {self.player}'s to command",
                "muted")
            return

        self.requests[self.selected] = request
        asked = asked_for(request, self.bucket.width)
        say(f"queued   p{placement[0]} s#{placement[1]}  {asked}", "muted")

    def boundary(self) -> str | None:
        if bool(self.state.game_over):
            winner = int(self.state.winner)
            return (
                f"game over - player {winner} wins"
                if winner >= 0
                else "game over - an Elimination draw, neither player has a unit")

        if int(self.state.time) >= int(self.game.max_cycles):
            return (
                f"time limit - truncated at cycle {int(self.state.time)}, "
                "no winner and no draw")

        return None

    def advance(self) -> list[tuple[str, str]]:
        reached = self.boundary()

        if reached is not None:
            self.running = False
            return [
                (reached, "head"),
                (
                    "ctrl+r resets the episode; core.step takes no terminal input",
                    "muted")]

        requests = dict(self.requests)

        if self.opponent:
            requests.update(wandering(self.view, self.bucket, 1 - self.player, self.generator))

        commands = commands_of(requests, self.bucket)
        was = self.view
        result = self.stepper(self.key, self.state, commands, self.game)
        self.state = result.next_state
        self.view = self.look(self.state)
        self.requests.clear()
        lines = transitions(was, self.view, requests, self.bucket)

        if bool(result.terminated) or bool(result.truncated):
            self.running = False
            reward = [float(value) for value in result.reward]
            lines.append((f"{self.boundary()}, reward {reward}", "head"))

        return lines


def report_configuration(session: Session, path: Path) -> None:
    bucket = session.bucket
    say()
    say(f"nanoRTS - {path.name}", "head")
    say(
        f"  Bucket    {bucket.height}x{bucket.width} field, "
        f"neutral capacity {bucket.neutral_capacity}, "
        f"per-player capacity {bucket.per_player_capacity}",
        "muted")
    say(
        f"  Content   {bucket.utt_identity.value}, {bucket.conflict_strategy.value}, "
        f"horizon {int(session.game.max_cycles)} cycles",
        "muted")
    say(f"  Backend   {jax.devices()[0].platform.upper()}", "muted")


def report_status(session: Session) -> None:
    view = session.view
    bucket = session.bucket
    parts = []
    for player in range(model.PLAYER_COUNT):
        slots = bucket.player_slots(player)
        units = int(view.active[slots.start : slots.stop].sum())
        free = int(view.available[player].sum())
        marker = "*" if player == session.player else " "
        parts.append(f"{marker}p{player} {view.resources[player]} res {units} units {free} avail")

    say(f"cycle {view.time:<6}" + "  |  ".join(parts), "head")


def report_selection(session: Session) -> None:
    view = session.view
    slot = session.selected

    if slot is None or not view.active[slot]:
        say("selected nothing", "muted")
        return

    bucket = session.bucket
    placement = placement_of(slot, bucket)
    address = (
        f"slot {slot} (neutral)"
        if placement is None
        else f"slot {slot} = p{placement[0]} s#{placement[1]}")
    x, y = cell_xy(int(view.cell[slot]), bucket.width)
    say(
        f"selected {address}  {model.TYPE_NAMES[int(view.unit_type[slot])]}  "
        f"birth {int(view.birth[slot])}  cell ({x},{y})={int(view.cell[slot])}  "
        f"hp {int(view.hit_points[slot])}  "
        f"carrying {int(view.carried_resources[slot])}",
        "bold")
    kind = int(view.pending_kind[slot])

    if kind == model.KIND_ABSENT:
        say("         pending none - the unit is Available", "good")
        return

    remaining = int(view.pending_completes_at[slot]) - view.time
    say(
        f"         pending {KIND_NAMES[kind]} {pending_payload(view, slot)}  "
        f"issued {int(view.pending_issued_at[slot])}  "
        f"completes {int(view.pending_completes_at[slot])}  "
        f"order {int(view.pending_order[slot])}  "
        f"{remaining} left - not Available",
        "good")


HELP_LINES: tuple[tuple[str, str], ...] = (
    ("One step is one Java cycle: Admission runs, the clock advances, the", "muted"),
    ("Completion batch runs its Effects, and the batch is removed.", "muted"),
    ("", "muted"),
    ("click            select the unit under the cursor", "bold"),
    ("tab / 1 / 2      switch the player you are commanding", "bold"),
    ("arrows / wasd    queue MOVE in that direction (also sets the heading)", "bold"),
    ("n                queue NONE(10) - a real Pending action, not a NOOP", "bold"),
    ("h / r / p        queue HARVEST / RETURN / PRODUCE along the heading", "bold"),
    ("t                cycle the type a PRODUCE would produce", "bold"),
    ("x then click     queue ATTACK at the clicked cell", "bold"),
    ("backspace        drop the selected unit's queued Command", "bold"),
    ("", "muted"),
    ("space            advance exactly one cycle", "bold"),
    ("enter            run continuously / stop", "bold"),
    ("[ ]              slower / faster", "bold"),
    ("b                random-walk opponent for the other player on / off", "bold"),
    ("i                print the selected unit and the clock again", "bold"),
    ("ctrl+r           reset the episode", "bold"),
    ("g                write the current frame as play-frame.png", "bold"),
    ("m                start / stop recording, written on stop", "bold"),
    ("f1               print this help", "bold"),
    ("esc / q          quit", "bold"),
    ("", "muted"),
    (
        "An episode stops at its own boundary: one surviving owner wins, none is",
        "muted",
    ),
    ("a draw, and the configured horizon truncates. ctrl+r starts over.", "muted"),
    ("", "muted"),
    ("Worth showing: walk a unit into a wall - the refused MOVE becomes a", "muted"),
    ("timed NONE of the refused action's own duration. Send two units into", "muted"),
    ("one cell and watch CANCEL_BOTH cancel both. Ask a Base to MOVE.", "muted"),
)


def report_help() -> None:
    say()
    say("keys", "head")
    for text, style in HELP_LINES:
        say(f"  {text}" if text else "", style)

    say()


class Board:
    def __init__(self, session: Session, cell_pixels: int, caption: str) -> None:
        self.session = session
        self.cell_pixels = cell_pixels
        self.size = (
            session.bucket.width * cell_pixels,
            session.bucket.height * cell_pixels)
        self.screen = pygame.display.set_mode(self.size)
        self.caption = caption
        self.font = pygame.font.Font(None, max(16, cell_pixels // 3))

    def retitle(self) -> None:
        session = self.session
        mode = "running" if session.running else "paused"
        pygame.display.set_caption(
            f"nanoRTS {self.caption} - cycle {session.view.time} - "
            f"commanding p{session.player} - {mode}")

    def frame(self) -> np.ndarray:
        return np.asarray(self.session.renderer(self.session.state, self.session.game, self.session.assets))

    def draw(self) -> None:
        image = self.frame()
        self.screen.blit(pygame.surfarray.make_surface(image.transpose(1, 0, 2)), (0, 0))
        self.draw_marks()
        self.retitle()
        pygame.display.flip()

    def cell_rect(self, cell: int) -> pygame.Rect:
        x, y = cell_xy(cell, self.session.bucket.width)
        size = self.cell_pixels
        return pygame.Rect(x * size, y * size, size, size)

    def draw_marks(self) -> None:
        view = self.session.view
        for slot in range(self.session.bucket.entity_capacity):
            if not view.active[slot]:
                continue

            rect = self.cell_rect(int(view.cell[slot]))
            kind = int(view.pending_kind[slot])

            if kind != model.KIND_ABSENT:
                remaining = max(int(view.pending_completes_at[slot]) - view.time, 0)
                self.draw_arrow(rect, kind, int(view.pending_direction[slot]), GOOD)
                label = self.font.render(str(remaining), True, GOOD)
                self.screen.blit(label, (rect.right - label.get_width() - 3, rect.top + 2))

            if slot in self.session.requests:
                pygame.draw.rect(self.screen, QUEUED, rect, width=2)

        selected = self.session.selected

        if selected is not None and view.active[selected]:
            rect = self.cell_rect(int(view.cell[selected]))
            pygame.draw.rect(self.screen, SELECTION, rect.inflate(-2, -2), width=3)

    def draw_arrow(self, rect: pygame.Rect, kind: int, direction: int, colour: tuple[int, int, int]) -> None:
        directional = kind in (
            model.KIND_MOVE,
            model.KIND_HARVEST,
            model.KIND_RETURN,
            model.KIND_PRODUCE)

        if not directional or direction == model.ABSENT:
            pygame.draw.circle(self.screen, colour, rect.center, max(3, self.cell_pixels // 12))
            return

        dx, dy = model.DIRECTION_OFFSETS[direction]
        cx, cy = float(rect.centerx), float(rect.centery)
        reach = self.cell_pixels * 0.30
        spread = self.cell_pixels * 0.16
        pygame.draw.polygon(
            self.screen,
            colour,
            [
                (cx + dx * reach, cy + dy * reach),
                (
                    cx - dx * reach * 0.4 - dy * spread,
                    cy - dy * reach * 0.4 + dx * spread),
                (
                    cx - dx * reach * 0.4 + dy * spread,
                    cy - dy * reach * 0.4 - dx * spread)])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--cell-pixels", type=int, default=64)
    parser.add_argument(
        "--utt",
        type=model.UnitTypeTableVersion,
        choices=tuple(model.UnitTypeTableVersion),
        default=model.UnitTypeTableVersion.ORIGINAL_FINETUNED)
    parser.add_argument(
        "--conflict-strategy",
        type=model.ConflictStrategy,
        choices=tuple(model.ConflictStrategy),
        default=model.ConflictStrategy.CANCEL_BOTH)
    parser.add_argument("--max-cycles", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cycles-per-second", type=float, default=4.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--record", type=Path, default=Path("play.gif"))
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument(
        "--self-check",
        type=int,
        default=0,
        metavar="CYCLES",
        help="run this many scripted cycles with no window and report, so the "
        "whole path can be exercised where no display exists")
    return parser.parse_args()


def build_session(arguments: argparse.Namespace) -> Session:
    game = content.load_game(
        arguments.map,
        utt=arguments.utt,
        conflict_strategy=arguments.conflict_strategy,
        max_cycles=arguments.max_cycles)
    assets = rendering.prepare_assets(game.bucket, cell_pixels=arguments.cell_pixels)
    key = jax.random.key(arguments.seed)
    legality = jax.jit(core.legal_commands)
    state = core.reset(key, game)
    return Session(
        game=game,
        assets=assets,
        key=key,
        state=state,
        view=readback(state, legality(state, game).available),
        stepper=jax.jit(core.step),
        legality=legality,
        renderer=jax.jit(rendering.render),
        speed=arguments.cycles_per_second,
        record=arguments.record,
        fps=arguments.fps,
        generator=np.random.default_rng(arguments.seed))


def self_check(session: Session, cycles: int) -> int:
    start = session.view.time
    before = session.view.cell.copy()
    lines: list[tuple[str, str]] = []
    for _ in range(cycles):
        session.requests.update(patrolling(session.view, session.bucket))
        lines = session.advance()

    moved = int(np.sum((session.view.cell != before) & session.view.active))
    elapsed = session.view.time - start
    for text, style in lines:
        say(f"  {text}", style)

    report_status(session)
    say(f"self-check: {elapsed} cycles, {moved} units at a new cell")

    if elapsed != cycles:
        print(f"FAILED: clock advanced {elapsed}, expected {cycles}", file=sys.stderr)
        return 1

    if moved == 0:
        print(
            f"FAILED: no MOVE completed in {cycles} cycles - the shortest "
            f"moveTime in either supported table is 8, so a shorter run than "
            f"that proves nothing",
            file=sys.stderr)
        return 1

    say("ok", "good")
    return 0


MOVE_KEYS: dict[int, int] = {
    pygame.K_UP: model.DIRECTION_UP,
    pygame.K_w: model.DIRECTION_UP,
    pygame.K_RIGHT: model.DIRECTION_RIGHT,
    pygame.K_d: model.DIRECTION_RIGHT,
    pygame.K_DOWN: model.DIRECTION_DOWN,
    pygame.K_s: model.DIRECTION_DOWN,
    pygame.K_LEFT: model.DIRECTION_LEFT,
    pygame.K_a: model.DIRECTION_LEFT,
}

DIRECTED_KEYS: dict[int, int] = {
    pygame.K_h: model.KIND_HARVEST,
    pygame.K_r: model.KIND_RETURN,
    pygame.K_p: model.KIND_PRODUCE,
}


def step_and_report(session: Session) -> None:
    lines = session.advance()

    if not lines:
        return

    for text, style in lines:
        say(f"  {text}", style)

    report_status(session)


def handle_key(session: Session, board: Board, event: pygame.event.Event) -> bool:
    key = event.key

    if key in (pygame.K_ESCAPE, pygame.K_q):
        return False

    if key == pygame.K_F1:
        report_help()
        return True

    if key == pygame.K_SPACE:
        step_and_report(session)
        return True

    if key == pygame.K_RETURN:
        session.running = not session.running
        say(f"{'running' if session.running else 'paused'}", "head")
        return True

    if key == pygame.K_LEFTBRACKET:
        session.speed = max(0.5, session.speed / 2)
        say(f"{session.speed:g} cycles/second", "muted")
        return True

    if key == pygame.K_RIGHTBRACKET:
        session.speed = min(120.0, session.speed * 2)
        say(f"{session.speed:g} cycles/second", "muted")
        return True

    if key in (pygame.K_TAB, pygame.K_1, pygame.K_2):
        session.player = (1 - session.player if key == pygame.K_TAB else int(key == pygame.K_2))
        say(f"commanding player {session.player}", "head")
        return True

    if key == pygame.K_b:
        session.opponent = not session.opponent
        say(f"opponent {'on' if session.opponent else 'off'}", "head")
        return True

    if key == pygame.K_i:
        report_status(session)
        report_selection(session)
        return True

    if key == pygame.K_r and event.mod & pygame.KMOD_CTRL:
        session.restart()
        return True

    if key in MOVE_KEYS:
        session.heading = MOVE_KEYS[key]
        session.queue(Request(kind=model.KIND_MOVE, direction=session.heading))
        return True

    if key in DIRECTED_KEYS:
        kind = DIRECTED_KEYS[key]
        session.queue(
            Request(
                kind=kind,
                direction=session.heading,
                produced_type=(session.produced if kind == model.KIND_PRODUCE else model.ABSENT)))
        return True

    if key == pygame.K_n:
        session.queue(Request(kind=model.KIND_NONE, none_duration=10))
        return True

    if key == pygame.K_t:
        index = (PRODUCIBLE.index(session.produced) + 1) % len(PRODUCIBLE)
        session.produced = PRODUCIBLE[index]
        say(f"PRODUCE would produce {model.TYPE_NAMES[session.produced]}", "muted")
        return True

    if key == pygame.K_x:
        session.aiming = True
        say("click a cell to aim the ATTACK", "muted")
        return True

    if key == pygame.K_BACKSPACE:
        if session.selected is not None:
            session.requests.pop(session.selected, None)
            say("queued Command dropped", "muted")

        return True

    if key == pygame.K_g:
        iio.imwrite("play-frame.png", board.frame(), extension=".png")
        say("wrote play-frame.png", "muted")
        return True

    if key == pygame.K_m:
        toggle_recording(session, board)
        return True

    return True


def toggle_recording(session: Session, board: Board) -> None:
    if session.frames is None:
        session.frames = [board.frame()]
        say("recording", "head")
        return

    write_recording(session)


def write_recording(session: Session) -> None:
    if not session.frames:
        session.frames = None
        return

    iio.imwrite(
        session.record,
        np.stack(session.frames),
        extension=session.record.suffix,
        fps=session.fps,
        loop=0)
    say(f"wrote {session.record} - {len(session.frames)} frames", "head")
    session.frames = None


def handle_click(session: Session, board: Board, position: tuple[int, int]) -> None:
    x, y = position[0] // board.cell_pixels, position[1] // board.cell_pixels

    if session.aiming and session.selected is not None:
        session.queue(Request(kind=model.KIND_ATTACK, target_cell=y * session.bucket.width + x))
        session.aiming = False
        return

    session.select(x, y)


def main() -> int:
    arguments = parse_arguments()
    jax.config.update("jax_platforms", arguments.device)
    say("compiling the cycle...", "muted")
    session = build_session(arguments)
    report_configuration(session, arguments.map)

    if arguments.self_check:
        return self_check(session, arguments.self_check)

    try:
        pygame.init()
    except pygame.error as error:
        print(
            f"pygame could not open a window ({error}). Set SDL_VIDEODRIVER, or "
            f"run play --self-check CYCLES without one.",
            file=sys.stderr)
        return 1

    board = Board(session, arguments.cell_pixels, arguments.map.name)
    report_help()
    report_status(session)

    clock = pygame.time.Clock()
    elapsed = 0.0
    alive = True
    while alive:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                alive = False
            elif event.type == pygame.KEYDOWN:
                alive = handle_key(session, board, event)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                handle_click(session, board, event.pos)

        if session.running:
            elapsed += clock.get_time() / 1000.0

            if elapsed >= 1.0 / session.speed:
                elapsed = 0.0
                step_and_report(session)

        if session.frames is not None:
            session.frames.append(board.frame())

        board.draw()
        clock.tick(60)

    write_recording(session)
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
