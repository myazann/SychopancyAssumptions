"""Figures for the open-ended analysis.

One place for the chart style, because the alternative is twelve calls to
matplotlib that each drift a little. Every figure is a static PNG, so the CSV
written beside it is its table view -- three of the categorical hues sit below
3:1 against the light chart surface, and the rule for those is that the value
must be readable without the color, which the direct labels and the CSV both
satisfy.

The palette is the validated reference instance: eight categorical slots taken
in fixed order (never cycled -- a ninth series folds into "other"), one blue
ramp for magnitude, and blue-to-red through a neutral gray for anything signed.
Signed quantities get the diverging map because zero has to read as nothing;
unsigned magnitude gets the single hue.

Every figure renders in both modes. The dark steps are the same eight hues
re-stepped for the dark surface, not an inversion of the light ones -- a
flipped chart puts light-surface colors on a dark ground, where half of them
drop below the contrast floor.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

MODES = ("light", "dark")

#: Categorical slots, in the order they must be assigned.
_SERIES = {
    "light": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
    "dark": ("#3987e5", "#d95926", "#199e70", "#c98500",
             "#d55181", "#008300", "#9085e9", "#e66767"),
}

_CHROME = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "secondary": "#52514e",
              "muted": "#898781", "grid": "#e1e0d9", "baseline": "#c3c2b7"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "secondary": "#c3c2b7",
             "muted": "#898781", "grid": "#2c2c2a", "baseline": "#383835"},
}

#: Blue, light to dark, for unsigned magnitude. Reversed in dark mode so that
#: "more" still reads as "further from the surface".
_SEQUENTIAL_STEPS = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab",
                     "#0d366b"]

#: Blue to red through a neutral. The midpoint must read as "nothing", so it
#: is the mode's own gray rather than a third hue.
_DIVERGING_STEPS = {
    "light": ["#104281", "#2a78d6", "#9ec5f4", "#f0efec",
              "#f0a6a5", "#e34948", "#a32222"],
    "dark": ["#86b6ef", "#3987e5", "#1c5cab", "#383835",
             "#8c2b2b", "#d64c4b", "#e88a89"],
}

FONT = ["DejaVu Sans", "sans-serif"]

_MODE = "light"


def palette(mode: str | None = None) -> dict:
    mode = mode or _MODE
    return {"series": _SERIES[mode], **_CHROME[mode]}


def _cmaps(mode: str):
    steps = _SEQUENTIAL_STEPS if mode == "light" else _SEQUENTIAL_STEPS[::-1]
    return (LinearSegmentedColormap.from_list(f"syco_seq_{mode}", steps),
            LinearSegmentedColormap.from_list(f"syco_div_{mode}",
                                              _DIVERGING_STEPS[mode]))


def style(mode: str = "light") -> dict:
    """Apply the chart chrome for one mode: hairline solid grid, no box."""
    global _MODE
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    _MODE = mode
    colours = palette(mode)
    plt.rcParams.update({
        "figure.facecolor": colours["surface"],
        "axes.facecolor": colours["surface"],
        "savefig.facecolor": colours["surface"],
        "font.family": "sans-serif",
        "font.sans-serif": FONT,
        "text.color": colours["ink"],
        "axes.labelcolor": colours["secondary"],
        "axes.edgecolor": colours["baseline"],
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": colours["grid"],
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": colours["muted"],
        "ytick.color": colours["muted"],
        "xtick.labelcolor": colours["secondary"],
        "ytick.labelcolor": colours["secondary"],
        "xtick.major.width": 0.0,
        "ytick.major.width": 0.0,
        "legend.frameon": False,
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
    })
    return colours


def both_modes(draw, path) -> list:
    """Render one figure in both modes.

    `path` names the light file; the dark one gets a `.dark` before the
    suffix, so a report can pick per theme and a slide deck can take either.
    """
    path = Path(path)
    made = []
    for mode in MODES:
        target = path if mode == "light" else path.with_suffix(f".dark{path.suffix}")
        style(mode)
        made.append(draw(target, palette(mode), _cmaps(mode)))
    style("light")
    return made


def _finish(ax, colours, title: str, subtitle: str = "", xlabel: str = "",
            ylabel: str = "") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(colours["baseline"])
    ax.spines["bottom"].set_color(colours["baseline"])
    if title:
        ax.set_title(title, color=colours["ink"], fontsize=11.5,
                     fontweight="bold", loc="left", pad=18 if subtitle else 10)
    if subtitle:
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes,
                color=colours["muted"], fontsize=8.5, va="bottom", ha="left")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)


def _save(fig, path) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
def heatmap(matrix, row_labels, col_labels, path, *, title: str = "",
            subtitle: str = "", value_label: str = "", diverging: bool = True,
            annotate: bool = True, cell: float = 0.34, fmt: str = "{:+.2f}"):
    """A signed row x column map -- facets against topics, mostly.

    Annotated by default: the color carries the pattern and the number carries
    the value, so no cell has to be read as color alone.
    """
    matrix = np.asarray(matrix, dtype="float64")

    def draw(target, colours, cmaps):
        sequential, diverge = cmaps
        height = max(2.6, 0.30 * len(row_labels) + 1.9)
        width = max(5.0, cell * len(col_labels) + 3.4)
        fig, ax = plt.subplots(figsize=(width, height))
        ax.grid(False)

        if diverging:
            limit = float(np.nanmax(np.abs(matrix))) or 1.0
            norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
            image = ax.imshow(matrix, cmap=diverge, norm=norm, aspect="auto")
        else:
            image = ax.imshow(matrix, cmap=sequential, aspect="auto")

        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
        # A surface-coloured gap between cells rather than a border on each.
        ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
        ax.grid(which="minor", color=colours["surface"], linewidth=1.6)
        ax.tick_params(which="minor", length=0)

        if annotate and matrix.size <= 500:
            span = float(np.nanmax(np.abs(matrix))) or 1.0
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    value = matrix[i, j]
                    if not np.isfinite(value):
                        continue
                    strong = abs(value) > 0.62 * span
                    shade = "#ffffff" if strong else colours["ink"]
                    ax.text(j, i, fmt.format(value), ha="center", va="center",
                            fontsize=6.2, color=shade)

        bar = fig.colorbar(image, ax=ax, fraction=0.024, pad=0.015)
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=7.5, colors=colours["muted"], length=0)
        if value_label:
            bar.set_label(value_label, fontsize=8, color=colours["secondary"])
        _finish(ax, colours, title, subtitle)
        return _save(fig, target)

    return both_modes(draw, path)


def grouped_bars(categories, series, path, *, title: str = "",
                 subtitle: str = "", ylabel: str = "", label_fmt: str = "{:.2f}",
                 horizontal: bool = False, figsize=None):
    """Bars grouped by category. `series` is an ordered {name: values} mapping.

    Slots are taken in palette order and never cycled; a legend is drawn for
    two or more series and the values are labeled directly, which is also the
    relief the low-contrast slots require.
    """
    names = list(series)
    if len(names) > len(_SERIES["light"]):
        raise ValueError(f"{len(names)} series exceeds the "
                         f"{len(_SERIES['light'])} categorical slots; fold the "
                         "tail into 'other'")
    n = len(names)
    positions = np.arange(len(categories))
    thickness = 0.74 / max(n, 1)
    if figsize is None:
        span = max(5.4, 0.52 * len(categories) * max(n, 1) ** 0.5 + 2.2)
        figsize = ((6.8, max(3.0, 0.30 * len(categories) * n + 1.8))
                   if horizontal else (span, 3.9))

    def draw(target, colours, _cmaps_unused):
        fig, ax = plt.subplots(figsize=figsize)
        for index, name in enumerate(names):
            offset = (index - (n - 1) / 2) * thickness
            values = np.asarray(series[name], dtype="float64")
            colour = colours["series"][index]
            if horizontal:
                bars = ax.barh(positions + offset, values,
                               height=thickness * 0.78, color=colour,
                               label=name, linewidth=0)
            else:
                bars = ax.bar(positions + offset, values, width=thickness * 0.78,
                              color=colour, label=name, linewidth=0)
            if label_fmt and len(categories) * n <= 60:
                ax.bar_label(bars, fmt=label_fmt, fontsize=6.4, padding=2,
                             color=colours["secondary"])

        if horizontal:
            ax.set_yticks(positions)
            ax.set_yticklabels(categories, fontsize=8)
            ax.grid(axis="y", visible=False)
            ax.margins(x=0.14)
            _finish(ax, colours, title, subtitle, xlabel=ylabel)
        else:
            ax.set_xticks(positions)
            ax.set_xticklabels(categories, rotation=30, ha="right", fontsize=8)
            ax.grid(axis="x", visible=False)
            ax.margins(y=0.14)
            _finish(ax, colours, title, subtitle, ylabel=ylabel)
        if n >= 2:
            ax.legend(fontsize=8, labelcolor=colours["secondary"],
                      ncols=min(n, 4), loc="upper center",
                      bbox_to_anchor=(0.5, -0.16))
        return _save(fig, target)

    return both_modes(draw, path)


def dot_ci(labels, estimates, lows, highs, path, *, title: str = "",
           subtitle: str = "", xlabel: str = "", highlight=None,
           reference: float = 0.0):
    """An estimate-and-interval plot, largest effect at the top.

    `highlight` marks the rows that survived multiplicity correction. Those
    take slot 1; the rest are muted, so significance is carried by emphasis
    rather than by a second hue.
    """
    n = len(labels)
    estimates = np.asarray(estimates, dtype="float64")
    lows = np.asarray(lows, dtype="float64")
    highs = np.asarray(highs, dtype="float64")
    marks = [True] * n if highlight is None else list(highlight)

    def draw(target, colours, _cmaps_unused):
        fig, ax = plt.subplots(figsize=(7.2, max(2.6, 0.30 * n + 1.5)))
        positions = np.arange(n)[::-1]
        ax.axvline(reference, color=colours["baseline"], linewidth=1.0, zorder=1)
        for i in range(n):
            colour = colours["series"][0] if marks[i] else colours["muted"]
            ax.plot([lows[i], highs[i]], [positions[i]] * 2, color=colour,
                    linewidth=2.0, solid_capstyle="round",
                    alpha=0.85 if marks[i] else 0.5, zorder=2)
            ax.plot([estimates[i]], [positions[i]], "o", markersize=5.5,
                    color=colour, markeredgecolor=colours["surface"],
                    markeredgewidth=1.4, zorder=3)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=8)
        ax.grid(axis="y", visible=False)
        ax.margins(y=0.02)
        _finish(ax, colours, title, subtitle, xlabel=xlabel)
        return _save(fig, target)

    return both_modes(draw, path)


def scatter_fit(x, y, path, *, labels=None, title: str = "", subtitle: str = "",
                xlabel: str = "", ylabel: str = "", annotation: str = ""):
    """A persona-level scatter with a least-squares line.

    Used only where the unit really is a person -- 25 points, one per persona,
    so the line describes 25 observations and is drawn thin enough to look
    like it.
    """
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    finite = np.isfinite(x) & np.isfinite(y)

    def draw(target, colours, _cmaps_unused):
        fig, ax = plt.subplots(figsize=(5.4, 3.8))
        ax.plot(x, y, "o", markersize=6.5, color=colours["series"][0],
                alpha=0.8, markeredgecolor=colours["surface"],
                markeredgewidth=1.4, zorder=3)
        if finite.sum() >= 3 and np.ptp(x[finite]) > 0:
            slope, intercept = np.polyfit(x[finite], y[finite], 1)
            grid = np.linspace(x[finite].min(), x[finite].max(), 50)
            ax.plot(grid, slope * grid + intercept, color=colours["muted"],
                    linewidth=1.4, zorder=2)
        if labels is not None:
            for xi, yi, text in zip(x, y, labels):
                ax.annotate(text, (xi, yi), fontsize=6.2,
                            color=colours["muted"], xytext=(4, 3),
                            textcoords="offset points")
        if annotation:
            ax.text(0.98, 0.03, annotation, transform=ax.transAxes,
                    fontsize=8.5, color=colours["secondary"], ha="right",
                    va="bottom")
        _finish(ax, colours, title, subtitle, xlabel=xlabel, ylabel=ylabel)
        return _save(fig, target)

    return both_modes(draw, path)
