"""Scatter plot of an instance's geography: hubs, sources, customers.

The three node classes are an *identity* encoding, so they take the first three slots of the
categorical palette — the set validated for all-pairs colour-vision separation, which is the
list that applies to scatter plots (any two series can end up adjacent on screen, not just
neighbours in a legend). Shape carries the distinction a second time, so the figure survives
greyscale printing and colour-vision deficiency without relying on hue.

Light and dark variants are separately chosen steps against their own surface, not an inverted
copy of one another.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # savefig only: keeps the CLI headless-safe under make and CI

import matplotlib.pyplot as plt  # imported after the backend selection above, deliberately
from matplotlib.axes import Axes

from src.data.instance import Instance

_FIGURE_SIZE_IN = (9.0, 9.5)
_FIGURE_DPI = 200
_HUB_MARKER_SIZE = 110.0
_SOURCE_MARKER_SIZE = 26.0
_CUSTOMER_MARKER_SIZE = 12.0
_MARK_RING_WIDTH = 1.4
_CUSTOMER_ALPHA = 0.8
_GRID_WIDTH = 0.6
_TITLE_SIZE = 13
_SUBTITLE_SIZE = 10
_TICK_SIZE = 9


@dataclass(frozen=True, slots=True)
class PlotTheme:
    """One resolved set of surface, ink and series colours."""

    surface: str
    ink: str
    ink_secondary: str
    muted: str
    grid: str
    hub: str
    source: str
    customer: str


LIGHT_THEME = PlotTheme(
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    hub="#2a78d6",
    source="#eb6834",
    customer="#1baf7a",
)

DARK_THEME = PlotTheme(
    surface="#1a1a19",
    ink="#ffffff",
    ink_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    hub="#3987e5",
    source="#d95926",
    customer="#199e70",
)

THEMES: dict[str, PlotTheme] = {"light": LIGHT_THEME, "dark": DARK_THEME}


def plot_instance(instance: Instance, path: Path, theme: PlotTheme = LIGHT_THEME) -> Path:
    """Render ``instance`` to a PNG at ``path`` and return the path written.

    Args:
        instance: The instance to draw.
        path: Destination PNG; parent directories are created.
        theme: Colour theme; light by default.

    Returns:
        The path written, for the caller to report.
    """
    coords = instance.coordinates()
    n_hubs, n_sources = len(instance.hubs), len(instance.sources)
    hubs, sources, customers = (
        coords[:n_hubs],
        coords[n_hubs : n_hubs + n_sources],
        coords[n_hubs + n_sources :],
    )

    figure, axes = plt.subplots(figsize=_FIGURE_SIZE_IN, dpi=_FIGURE_DPI)
    figure.patch.set_facecolor(theme.surface)

    # Drawn least-to-most important so hubs are never occluded by the demand they serve.
    axes.scatter(
        customers[:, 1],
        customers[:, 0],
        s=_CUSTOMER_MARKER_SIZE,
        c=theme.customer,
        marker="o",
        linewidths=0.0,
        alpha=_CUSTOMER_ALPHA,
        label="Customers",
        zorder=2,
    )
    axes.scatter(
        sources[:, 1],
        sources[:, 0],
        s=_SOURCE_MARKER_SIZE,
        c=theme.source,
        marker="^",
        edgecolors=theme.surface,
        linewidths=0.5,
        label="Sources",
        zorder=3,
    )
    axes.scatter(
        hubs[:, 1],
        hubs[:, 0],
        s=_HUB_MARKER_SIZE,
        c=theme.hub,
        marker="s",
        edgecolors=theme.surface,
        linewidths=_MARK_RING_WIDTH,
        label="Hubs",
        zorder=4,
    )

    _style_axes(axes, instance, theme)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=theme.surface, bbox_inches="tight")
    plt.close(figure)
    return path


def _style_axes(axes: Axes, instance: Instance, theme: PlotTheme) -> None:
    """Apply chrome: recessive grid and spines, geographic aspect, titles, legend."""
    geo = instance.geo
    axes.set_facecolor(theme.surface)
    axes.set_xlim(geo.lon_min, geo.lon_max)
    axes.set_ylim(geo.lat_min, geo.lat_max)

    # A degree of longitude is shorter than a degree of latitude at this latitude; without this
    # the map is stretched east-west and hub spacing reads wrongly.
    mean_lat = 0.5 * (geo.lat_min + geo.lat_max)
    axes.set_aspect(1.0 / math.cos(math.radians(mean_lat)))

    axes.grid(visible=True, color=theme.grid, linewidth=_GRID_WIDTH, zorder=1)
    axes.set_axisbelow(True)
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.tick_params(colors=theme.muted, labelsize=_TICK_SIZE, length=0)
    axes.set_xlabel("Longitude (°E)", color=theme.muted, fontsize=_TICK_SIZE)
    axes.set_ylabel("Latitude (°N)", color=theme.muted, fontsize=_TICK_SIZE)

    axes.set_title(
        "Synthetic Mumbai / Navi Mumbai network",
        color=theme.ink,
        fontsize=_TITLE_SIZE,
        loc="left",
        pad=18.0,
    )
    axes.text(
        0.0,
        1.015,
        f"seed {instance.seed} · {len(instance.hubs)} hubs · {len(instance.sources)} sources · "
        f"{len(instance.customers)} customers · {len(instance.shipments)} shipments",
        transform=axes.transAxes,
        color=theme.ink_secondary,
        fontsize=_SUBTITLE_SIZE,
    )

    # Below the plot, not inside it: over a thousand markers leave no interior space where a
    # legend can sit without hiding the very density the figure exists to show.
    legend = axes.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.075),
        ncols=3,
        frameon=False,
        fontsize=_SUBTITLE_SIZE,
        scatterpoints=1,
        handletextpad=0.4,
        columnspacing=1.6,
    )
    for text in legend.get_texts():
        text.set_color(theme.ink_secondary)
