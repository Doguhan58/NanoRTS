from __future__ import annotations

from nanorts import observations, rendering
from nanorts.adapters.jaxmarl.layout import JaxMARLLayout, drawing_table
from nanorts.model import GameConfig, State

Observation = observations.SymbolicObservation | observations.PixelObservation

Observations = dict[str, Observation]


def observe(state: State, game: GameConfig, *, layout: JaxMARLLayout, assets: rendering.RenderAssets | None = None) -> Observations:
    table = drawing_table("jaxmarl.observe", layout, game.bucket, assets)
    if table is not None:
        return {
            agent: observations.pixels(
                game,
                state,
                table,
                observer=player,
                partial_observability=layout.partial_observability,
            )
            for player, agent in enumerate(layout.agents)
        }
    assert isinstance(layout.observation, observations.SymbolicLayout)
    return {
        agent: observations.symbolic(
            game,
            state,
            observer=player,
            partial_observability=layout.partial_observability,
            dtype=layout.observation.feature_maps_dtype,
        )
        for player, agent in enumerate(layout.agents)
    }
