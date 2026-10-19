from __future__ import annotations

import dataclasses
from typing import Final

import jax.numpy as jnp
from jax.typing import DTypeLike

from nanorts import observations, rendering
from nanorts.adapters import factors, lifecycle
from nanorts.model import PLAYER_COUNT, BucketSpec, GameConfig, ModelError

AGENTS: Final[tuple[str, ...]] = tuple(
    f"player_{player}" for player in range(PLAYER_COUNT)
)

SURFACES: Final[tuple[str, ...]] = ("symbolic", "pixels")

ACTION_FACTORS: Final[tuple[str, ...]] = (
    "kind",
    "move_direction",
    "harvest_direction",
    "return_direction",
    "produce_direction",
    "produce_type",
    "attack_offset",
)

MASK_FORMS: Final[tuple[str, ...]] = factors.MASK_FORMS

ObservationLayout = observations.SymbolicLayout | observations.PixelLayout


@dataclasses.dataclass(frozen=True)
class JaxMARLLayout:
    bucket: BucketSpec
    surface: str
    partial_observability: bool

    observation: ObservationLayout

    action_factors: tuple[str, ...]

    action_widths: tuple[int, ...]

    mask_offsets: tuple[int, ...]

    mask_width: int

    mask_form: str

    reward_components: bool

    @property
    def agents(self) -> tuple[str, ...]:
        return AGENTS

    @property
    def actions_shape(self) -> tuple[int, int]:
        return (self.bucket.per_player_capacity, len(self.action_factors))

    @property
    def mask_shape(self) -> tuple[int, int]:
        width = self.mask_width if self.mask_form == "rows" else factors.MASK_WORDS
        return (self.bucket.per_player_capacity, width)


def prepare(game: GameConfig, *, surface: str = "symbolic", partial_observability: bool = False, assets: rendering.RenderAssets | None = None, feature_dtype: DTypeLike | None = None, mask_form: str = "rows", reward_components: bool = False) -> JaxMARLLayout:
    _refuse_unknown_surface(surface)
    _refuse_assets("jaxmarl.prepare", surface, assets)
    _refuse_feature_dtype_without_feature_map(surface, feature_dtype)
    factors.refuse_mask_form("jaxmarl.prepare", mask_form, game.bucket)
    _refuse_reward_components(reward_components)
    layout = _observation_layout(
        game, surface, assets, partial_observability, feature_dtype
    )
    lifecycle.refuse_decided("jaxmarl.prepare", game)
    widths = factors.widths(game.bucket)
    return JaxMARLLayout(
        bucket=game.bucket,
        surface=surface,
        partial_observability=layout.partial_observability,
        observation=layout,
        action_factors=ACTION_FACTORS,
        action_widths=widths,
        mask_offsets=factors.offsets(widths),
        mask_width=sum(widths),
        mask_form=mask_form,
        reward_components=reward_components,
    )


def _refuse_unknown_surface(surface: str) -> None:
    if surface not in SURFACES:
        raise ModelError(
            f"jaxmarl.prepare refuses surface={surface!r}: this release delivers "
            f"{list(SURFACES)}; a surface is one compiled Observation encoding, "
            "not a runtime flag"
        )


def _refuse_feature_dtype_without_feature_map(surface: str, feature_dtype: DTypeLike | None) -> None:
    if (
        surface == "pixels"
        and feature_dtype is not None
        and not _names_int32(feature_dtype)
    ):
        raise ModelError(
            f"jaxmarl.prepare refuses feature_dtype={feature_dtype!r} under "
            'surface="pixels": the pixel surface publishes a uint8 image and no '
            "Feature map"
        )


def _refuse_reward_components(reward_components: bool) -> None:
    if not isinstance(reward_components, bool):
        raise ModelError(
            f"jaxmarl.prepare refuses reward_components={reward_components!r}: it "
            "must be a bool, because it selects the compiled programs"
        )


def _observation_layout(game: GameConfig, surface: str, assets: rendering.RenderAssets | None, partial_observability: bool, feature_dtype: DTypeLike | None) -> ObservationLayout:
    if surface == "pixels":
        assert assets is not None
        layout = observations.pixel_layout(
            game, assets, partial_observability=partial_observability
        )
        _refuse_mismatch("jaxmarl.prepare", assets, layout, game.bucket)
        return layout
    return observations.symbolic_layout(
        game,
        partial_observability=partial_observability,
        dtype=(
            _narrowest_feature_dtype(game, partial_observability)
            if feature_dtype is None
            else feature_dtype
        ),
    )


def _narrowest_feature_dtype(game: GameConfig, partial_observability: bool) -> jnp.dtype:
    widest = observations.symbolic_layout(
        game, partial_observability=partial_observability, dtype=jnp.int32
    )
    bound = max(widest.bounds)
    return min(
        (
            dtype
            for dtype in observations.FEATURE_MAP_DTYPES
            if jnp.iinfo(dtype).max >= bound
        ),
        key=lambda dtype: dtype.itemsize,
    )


def _names_int32(dtype: DTypeLike) -> bool:
    try:
        return jnp.dtype(dtype) == jnp.dtype(jnp.int32)
    except TypeError:
        return False


def drawing_table(where: str, layout: JaxMARLLayout, bucket: BucketSpec, assets: rendering.RenderAssets | None) -> rendering.RenderAssets | None:
    _refuse_assets(where, layout.surface, assets)
    if layout.surface != "pixels":
        return None
    assert assets is not None
    assert isinstance(layout.observation, observations.PixelLayout)
    _refuse_mismatch(where, assets, layout.observation, bucket)
    return assets


def _refuse_assets(where: str, surface: str, assets: rendering.RenderAssets | None) -> None:
    if surface == "symbolic" and assets is not None:
        raise ModelError(
            f"{where} refuses assets={type(assets).__name__} under "
            'surface="symbolic": the symbolic surface renders nothing, and the '
            "keyword exists so the two surfaces share one signature"
        )
    if surface == "pixels" and assets is None:
        raise ModelError(
            f'{where} requires assets under surface="pixels": the prepared Asset '
            "set is an argument rather than a captured constant, because a table "
            "closed over inside jit is a constant whose materialization is a "
            "host transfer"
        )


def _refuse_mismatch(where: str, assets: rendering.RenderAssets, layout: observations.PixelLayout, bucket: BucketSpec) -> None:
    if assets.cell_pixels != layout.cell_pixels:
        raise ModelError(
            f"{where} refuses an Asset set drawing {assets.cell_pixels} pixels "
            f"per cell against the layout's {layout.cell_pixels}: the image "
            "would not be the one the published surface describes"
        )
    if assets.unit_type_capacity != bucket.unit_type_capacity:
        raise ModelError(
            f"{where} refuses an Asset set drawing "
            f"{assets.unit_type_capacity} unit types against the Bucket's "
            f"{bucket.unit_type_capacity}: the table was prepared for other "
            "content"
        )
