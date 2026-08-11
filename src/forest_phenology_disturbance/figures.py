from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from .metrics import METHOD_LABELS, metric_values

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGURES = ROOT / "outputs" / "figures"
INK = "#20252A"
GRID = "#D8DDE1"
GRAY = "#70777D"
BLUE = "#2F6EA6"
GREEN = "#16867A"
ORANGE = "#C86819"
MAGENTA = "#B05A84"
CLASS_COLORS = {"broadleaf": "#009E73", "conifer": "#0072B2", "bamboo": "#D89A16"}
METHOD_COLORS = {
    "last_observation": "#777777",
    "direct_pair": "#777777",
    "seasonal_median": "#D19A22",
    "validation_selected_physical": "#8E6BB5",
    "shared_residual_ssm": "#6A86B8",
    "forest_conditioned_ssm": "#6B9F7B",
    "validation_selected_transformer": "#A56A43",
    "presto_frozen": "#C75D70",
    "validation_selected_gru": BLUE,
    "gru_h8": BLUE,
    "recent_season_gru": "#50A58B",
    "phenology_calibrated_gru": GREEN,
    "phenology_basis_gru": GREEN,
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.3,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=360, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.03, label, transform=ax.transAxes, fontsize=9, fontweight="bold")


def _geojson_lines(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))

    def polygons(geometry):
        if geometry["type"] == "Polygon":
            yield from geometry["coordinates"]
        elif geometry["type"] == "MultiPolygon":
            for polygon in geometry["coordinates"]:
                yield from polygon

    for feature in payload["features"]:
        for ring in polygons(feature["geometry"]):
            values = np.asarray(ring)
            if values.ndim == 2 and values.shape[1] >= 2:
                yield values[:, 0], values[:, 1]


def make_figure_01() -> None:
    windows = pd.read_csv(DATA / "context" / "study_windows.csv")
    topo = np.load(DATA / "context" / "taiwan_topography.npz")
    elevation = topo["elevation_m"]
    shade = topo["hillshade"]
    extent = topo["extent_wgs84"]
    fig = plt.figure(figsize=(7.08, 3.05))
    grid = fig.add_gridspec(1, 3, width_ratios=[0.82, 1.30, 0.78], wspace=0.34)
    ax_a = fig.add_subplot(grid[0])
    for x, y in _geojson_lines(DATA / "context" / "east_asia_coastlines.geojson"):
        ax_a.plot(x, y, color="#6F777D", lw=0.45)
    ax_a.add_patch(Rectangle((119.5, 21.7), 2.6, 3.9, fill=False, ec=MAGENTA, lw=1.1))
    ax_a.text(120.8, 21.35, "Taiwan", ha="center", color=MAGENTA)
    ax_a.set_xlim(105, 135)
    ax_a.set_ylim(15, 40)
    ax_a.set_xlabel("Longitude (degrees E)")
    ax_a.set_ylabel("Latitude (degrees N)")
    ax_a.xaxis.set_major_formatter(lambda value, _: f"{value:.1f}")
    ax_a.yaxis.set_major_formatter(lambda value, _: f"{value:.1f}")
    ax_a.grid(color=GRID, lw=0.45)
    panel(ax_a, "a")

    ax_b = fig.add_subplot(grid[1])
    bounds = [0, 50, 200, 500, 1000, 2000, 3000, 5000]
    cmap = ListedColormap(
        ["#49AD4A", "#79C85A", "#BBD85A", "#EEE85B", "#F4B34D", "#E77C32", "#C74425"]
    )
    norm = BoundaryNorm(bounds, cmap.N)
    image = ax_b.imshow(elevation, extent=extent, origin="upper", cmap=cmap, norm=norm, zorder=0)
    ax_b.imshow(shade, extent=extent, origin="upper", cmap="gray", alpha=0.17, zorder=1)
    normal = windows[windows["record_type"].eq("normal")]
    for forest_class, group in normal.groupby("forest_class"):
        ax_b.scatter(
            group["longitude_wgs84"],
            group["latitude_wgs84"],
            s=19,
            c=CLASS_COLORS[forest_class],
            ec="white",
            lw=0.45,
            label=forest_class.title(),
            zorder=3,
        )
    events = windows[windows["record_type"].eq("event")]
    ax_b.scatter(
        events["longitude_wgs84"],
        events["latitude_wgs84"],
        marker="x",
        s=28,
        c=ORANGE,
        lw=1.0,
        label="Event window",
        zorder=4,
    )
    ax_b.set_xlim(119.5, 122.1)
    ax_b.set_ylim(21.8, 25.5)
    ax_b.set_xlabel("Longitude (degrees E)")
    ax_b.set_ylabel("Latitude (degrees N)")
    ax_b.xaxis.set_major_formatter(lambda value, _: f"{value:.1f}")
    ax_b.yaxis.set_major_formatter(lambda value, _: f"{value:.1f}")
    ax_b.grid(color="white", lw=0.35, alpha=0.65)
    cax = ax_b.inset_axes([0.035, 0.61, 0.045, 0.31])
    cb = fig.colorbar(
        image, cax=cax, orientation="vertical", ticks=[25, 125, 350, 750, 1500, 2500, 4000]
    )
    cb.ax.set_yticklabels(["0", "50", "200", "500", "1000", "2000", "3000+"])
    cb.ax.tick_params(labelsize=6, length=1.5)
    cax.set_title("Elevation (m)", fontsize=6.2, pad=2)
    ax_b.legend(
        loc="lower left", frameon=True, facecolor="white", framealpha=0.92, ncol=2, fontsize=6.2
    )
    panel(ax_b, "b")

    ax_c = fig.add_subplot(grid[2])
    ax_c.set_xlim(2018.5, 2025.6)
    ax_c.set_ylim(-0.1, 3.15)
    stages = [
        (0.2, "Training", 2019, 2022, "#D9E8F4"),
        (1.2, "Validation", 2023, 2023, "#F4E4CA"),
        (2.2, "Test", 2024, 2025, "#DCEFE6"),
    ]
    for y, name, start, end, color in stages:
        width = end - start + 0.82
        ax_c.add_patch(Rectangle((start - 0.41, y), width, 0.55, fc=color, ec=INK, lw=0.7))
        ax_c.text((start + end) / 2, y + 0.275, name, ha="center", va="center", fontweight="bold")
    for year in range(2019, 2026):
        ax_c.axvline(year, color=GRID, lw=0.45, zorder=0)
    ax_c.text(2021.0, 2.95, "Five spatial folds", ha="center", color=GRAY)
    ax_c.text(2024.5, 2.86, "Out-of-fold predictions", ha="center", color=GRAY)
    ax_c.set_xticks(range(2019, 2026), [str(year) for year in range(2019, 2026)], rotation=90)
    ax_c.set_yticks([])
    ax_c.set_xlabel("Target year")
    for spine in ("left", "right", "top"):
        ax_c.spines[spine].set_visible(False)
    panel(ax_c, "c")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.95, bottom=0.17)
    save(fig, "figure_01_study_area_and_split")


def _box(ax, x, y, width, height, text, color, *, shape="round", fontsize=6.0):
    style = "round,pad=0.02,rounding_size=0.08" if shape == "round" else "square,pad=0.02"
    patch = FancyBboxPatch((x, y), width, height, boxstyle=style, fc="white", ec=color, lw=1.0)
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def _arrow(ax, start, end, color=INK):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            lw=0.9,
            color=color,
            connectionstyle="arc3,rad=0",
        )
    )


def _route(ax, points, color=INK):
    """Draw a right-angle route with one arrowhead at its destination."""

    xs, ys = zip(*points)
    if len(points) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=0.9, solid_capstyle="round")
    _arrow(ax, points[-2], points[-1], color)


def _diamond(ax, center, width, height, text, color):
    x, y = center
    vertices = [(x, y + height / 2), (x + width / 2, y), (x, y - height / 2), (x - width / 2, y)]
    patch = Polygon(vertices, closed=True, fc="white", ec=color, lw=1.0)
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=6.0)
    return patch


def make_figure_02() -> None:
    fig, ax = plt.subplots(figsize=(7.08, 3.65))
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    sections = [
        (0.15, 3.80, 3.00, 3.20, "A  INPUTS"),
        (3.35, 3.80, 4.55, 3.20, "B  FORECAST MODEL"),
        (8.10, 3.80, 5.90, 3.20, "C  TEST GATE"),
        (3.35, 0.15, 10.65, 3.40, "D  RESIDUAL SCREEN"),
    ]
    for x, y, w, h, title in sections:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="#66717A", lw=0.8, ls=(0, (4, 3))))
        ax.text(x + 0.18, y + h - 0.30, title, fontsize=7.3, fontweight="bold")
    inputs = [
        (0.55, 6.05, "S2 history\n+ quality", GREEN),
        (0.55, 5.35, "S1 VV/VH\n+ local context", BLUE),
        (0.55, 4.65, "Forest type", GREEN),
        (0.55, 3.95, "Event dates\nfor censoring", MAGENTA),
    ]
    for x, y, text, color in inputs:
        _box(ax, x, y, 2.10, 0.50, text, color, fontsize=5.8)

    _box(ax, 3.65, 5.80, 1.70, 0.62, "Censored\nhistory", GRAY)
    _box(ax, 5.80, 5.80, 1.70, 0.62, "Temporal\nmodels", BLUE)
    _box(ax, 3.65, 4.65, 1.70, 0.62, "Seasonal\nbaseline", ORANGE)
    _box(ax, 5.80, 4.65, 1.70, 0.62, "Forest-type\ncorrection", GREEN)
    _box(ax, 5.80, 3.95, 1.70, 0.55, "Expected normal", GREEN, fontsize=5.8)

    _box(ax, 8.45, 5.80, 1.55, 0.62, "Test\nobservations", GREEN)
    _box(ax, 8.45, 4.55, 1.55, 0.62, "Reference\nbaselines", GRAY)
    _box(ax, 10.35, 5.35, 1.55, 0.68, "MAE + SAM\nsame support", ORANGE, fontsize=5.8)
    _diamond(ax, (11.12, 4.45), 1.55, 0.92, "Both\nimproved?", ORANGE)
    _box(ax, 12.35, 5.45, 1.30, 0.62, "Retain", GREEN)
    _box(ax, 12.35, 4.15, 1.30, 0.62, "Revise", ORANGE)

    _box(ax, 3.65, 1.25, 1.65, 0.62, "Target S2\nSCL-valid", GREEN)
    _box(ax, 5.80, 2.20, 1.70, 0.62, "Expected\nnormal", GREEN)
    _box(ax, 5.80, 1.25, 1.70, 0.62, "Spectral\nresidual", ORANGE)
    _diamond(ax, (8.55, 1.56), 1.55, 0.92, "Residual\nlarge?", MAGENTA)
    _box(ax, 10.05, 2.10, 1.70, 0.62, "Normal\nagreement", GREEN)
    _box(ax, 10.05, 0.95, 1.70, 0.62, "Candidate\ndisturbance", MAGENTA)
    _box(ax, 12.20, 0.95, 1.50, 0.62, "Polygon\nevaluation", GRAY)

    _route(ax, [(2.65, 6.30), (3.65, 6.11)], GREEN)
    _route(ax, [(2.65, 5.60), (3.35, 5.60), (3.35, 6.11), (3.65, 6.11)], BLUE)
    _route(ax, [(2.65, 4.90), (3.65, 4.96)], GREEN)
    _route(ax, [(2.65, 4.20), (3.20, 4.20), (3.20, 5.72), (3.65, 5.92)], MAGENTA)
    _arrow(ax, (5.35, 6.11), (5.80, 6.11), BLUE)
    _arrow(ax, (5.35, 4.96), (5.80, 4.96), ORANGE)
    _route(ax, [(6.65, 5.80), (6.65, 5.45), (6.35, 5.45), (6.35, 5.27)], BLUE)
    _arrow(ax, (6.65, 4.65), (6.65, 4.50), GREEN)

    _arrow(ax, (10.00, 6.11), (10.35, 5.78), GREEN)
    _route(ax, [(10.00, 4.86), (10.18, 4.86), (10.18, 5.58), (10.35, 5.58)], GRAY)
    _route(ax, [(7.50, 4.23), (10.10, 4.23), (10.10, 5.48), (10.35, 5.48)], GREEN)
    _arrow(ax, (11.12, 5.35), (11.12, 4.91), ORANGE)
    _route(ax, [(11.90, 4.56), (12.10, 4.56), (12.10, 5.76), (12.35, 5.76)], GREEN)
    _arrow(ax, (11.90, 4.34), (12.35, 4.46), ORANGE)
    ax.text(12.08, 5.30, "yes", color=GREEN, fontsize=6.5, ha="right")
    ax.text(12.08, 4.12, "no", color=ORANGE, fontsize=6.5, ha="right")

    _arrow(ax, (6.65, 3.95), (6.65, 2.82), GREEN)
    _arrow(ax, (5.30, 1.56), (5.80, 1.56), GREEN)
    _arrow(ax, (6.65, 2.20), (6.65, 1.87), GREEN)
    _arrow(ax, (7.50, 1.56), (7.78, 1.56), ORANGE)
    _arrow(ax, (9.33, 1.72), (10.05, 2.41), GREEN)
    _arrow(ax, (9.33, 1.40), (10.05, 1.26), MAGENTA)
    _arrow(ax, (11.75, 1.26), (12.20, 1.26), MAGENTA)
    ax.text(9.55, 2.03, "no", color=GREEN, fontsize=6.5, ha="center")
    ax.text(9.55, 1.17, "yes", color=MAGENTA, fontsize=6.5, ha="center")
    ax.text(
        8.65,
        0.43,
        "small residual = agreement with expected conditions     large residual = disturbance evidence",
        ha="center",
        fontsize=6.8,
    )
    save(fig, "figure_02_expected_normal_workflow")


def make_figure_03() -> None:
    frame = pd.read_csv(DATA / "evaluation" / "phenology_observations.csv")
    frame["date"] = pd.to_datetime(frame["date"])
    season_order = [1, 2, 3, 4]
    season_labels = ["Dec-Feb", "Mar-May", "Jun-Aug", "Sep-Nov"]
    fig, axes = plt.subplots(1, 2, figsize=(7.08, 2.75), gridspec_kw={"wspace": 0.28})
    ax = axes[0]
    rng = np.random.default_rng(47)
    for forest_class in ("broadleaf", "conifer", "bamboo"):
        color = CLASS_COLORS[forest_class]
        selected = frame[frame["forest_class"].eq(forest_class)]
        means = []
        for quarter in season_order:
            values = selected[selected["quarter"].eq(quarter)]["ndvi"].dropna().to_numpy()
            x = np.full(len(values), quarter - 1) + rng.normal(0, 0.045, len(values))
            ax.scatter(x, values, s=10, color=color, alpha=0.18, edgecolors="none")
            means.append(float(np.mean(values)) if len(values) else np.nan)
        ax.plot(range(4), means, marker="o", color=color, label=forest_class.title())
    ax.set_xticks(range(4), season_labels)
    ax.set_ylabel("NDVI")
    ax.set_xlabel("Three-month season")
    ax.set_ylim(0.05, 0.95)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend(loc="lower right", frameon=False)
    panel(ax, "a")

    ax = axes[1]
    ax.axvspan(pd.Timestamp("2016-01-01"), pd.Timestamp("2018-01-01"), color="#E5E7E8", zorder=0)
    for forest_class in ("broadleaf", "conifer", "bamboo"):
        color = CLASS_COLORS[forest_class]
        selected = frame[frame["forest_class"].eq(forest_class)].copy()
        ax.scatter(
            selected["date"], selected["ndvi"], s=7, color=color, alpha=0.14, edgecolors="none"
        )
        quarterly = selected.set_index("date")["ndvi"].resample("QS").mean().dropna()
        ax.plot(quarterly.index, quarterly.values, color=color, lw=1.1)
    ax.text(
        pd.Timestamp("2017-01-01"),
        0.10,
        "No supported\nbroadleaf date",
        ha="center",
        color=GRAY,
        fontsize=6.5,
    )
    ax.set_ylabel("Median window NDVI")
    ax.set_xlabel("Observation date")
    tick_years = [2016, 2018, 2020, 2022, 2024, 2026]
    ax.set_xticks(
        [pd.Timestamp(f"{year}-01-01") for year in tick_years],
        [str(year) for year in tick_years],
    )
    ax.set_ylim(0.05, 0.95)
    ax.grid(axis="y", color=GRID, lw=0.5)
    panel(ax, "b")
    save(fig, "figure_03_forest_phenology")


def make_figure_04(summary: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    h1 = summary[
        (summary["stratum"].eq("overall")) & (summary["horizon_seasons"].astype(str).eq("1"))
    ]
    methods = [
        "last_observation",
        "seasonal_median",
        "validation_selected_physical",
        "shared_residual_ssm",
        "forest_conditioned_ssm",
        "validation_selected_transformer",
        "presto_frozen",
        "validation_selected_gru",
        "recent_season_gru",
        "phenology_calibrated_gru",
    ]
    selected = h1.set_index("method").loc[methods]
    labels = [
        "Last observation",
        "Seasonal median",
        "Physical phenology",
        "Pooled SSM",
        "Forest SSM",
        "Transformer",
        "Presto",
        "GRU",
        "Recent-season GRU",
        "Proposed",
    ]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.08, 3.35),
        gridspec_kw={"width_ratios": [1.65, 1.20, 1.10], "wspace": 0.48},
    )
    y = np.arange(len(methods))
    colors = [METHOD_COLORS[method] for method in methods]
    axes[0].hlines(y, 0, selected["mae"] * 1000, color=colors, alpha=0.30, lw=0.8)
    axes[0].scatter(selected["mae"] * 1000, y, c=colors, s=22, zorder=3)
    axes[0].set_xlabel("MAE (x 10$^{-3}$)")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", color=GRID, lw=0.5)
    panel(axes[0], "a")
    axes[1].hlines(y, 0, selected["sam_degrees"], color=colors, alpha=0.30, lw=0.8)
    axes[1].scatter(selected["sam_degrees"], y, c=colors, marker="s", s=22, zorder=3)
    axes[1].set_xlabel("SAM (degrees)")
    axes[1].set_yticks(y, [])
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", color=GRID, lw=0.5)
    panel(axes[1], "b")
    gains = bootstrap[
        (bootstrap["stratum"].eq("overall")) & (bootstrap["baseline"].eq("validation_selected_gru"))
    ]
    for index, metric in enumerate(("mae", "sam_degrees")):
        row = gains[gains["metric"].eq(metric)].iloc[0]
        value = row["gain"] * (1000 if metric == "mae" else 1)
        low = row["ci_low"] * (1000 if metric == "mae" else 1)
        high = row["ci_high"] * (1000 if metric == "mae" else 1)
        axes[2].errorbar(
            value, index, xerr=[[value - low], [high - value]], fmt="o", color=GREEN, capsize=3
        )
    axes[2].axvline(0, color=GRAY, ls="--", lw=0.8)
    axes[2].set_yticks([0, 1], ["MAE\nx 10$^{-3}$", "SAM\ndegrees"])
    axes[2].tick_params(axis="y", labelsize=6.3, pad=2)
    axes[2].set_xlabel("Gain over GRU")
    axes[2].grid(axis="x", color=GRID, lw=0.5)
    panel(axes[2], "c")
    save(fig, "figure_04_common_test_benchmark")


def _rgb(cube: np.ndarray, mask: np.ndarray, low: float, high: float) -> np.ndarray:
    rgb = np.moveaxis(cube[[2, 1, 0]], 0, -1).astype(np.float32)
    output = np.clip((rgb - low) / max(high - low, 1e-6), 0, 1) ** 0.82
    output[~mask] = 0.82
    return output


def _cube_metrics(
    observed: np.ndarray, predicted: np.ndarray, mask: np.ndarray
) -> tuple[float, float]:
    truth = np.moveaxis(observed, 0, -1)[mask]
    pred = np.moveaxis(predicted, 0, -1)[mask]
    values = metric_values(truth, pred)
    return values["mae"], values["sam_degrees"]


def make_figure_05() -> None:
    files = sorted((DATA / "cases" / "normal").glob("*.npz"))
    columns = ["Previous", "Seasonal", "GRU", "Proposed", "Observed", "Absolute error"]
    residual_cmap = mpl.colormaps["magma"].copy()
    residual_cmap.set_bad("#D0D0D0")
    fig, axes = plt.subplots(len(files), len(columns), figsize=(7.08, 5.25), squeeze=False)
    for row, path in enumerate(files):
        with np.load(path, allow_pickle=False) as source:
            data = {key: source[key] for key in source.files}
        mask = data["evaluation"].astype(bool)
        cubes = [
            data["previous"],
            data["seasonal"],
            data["gru"],
            data["phenology_gru"],
            data["observed"],
        ]
        valid_values = np.concatenate([cube[[2, 1, 0]][:, mask].ravel() for cube in cubes])
        low, high = np.quantile(valid_values, [0.02, 0.98])
        proposed_mae, proposed_sam = _cube_metrics(data["observed"], data["phenology_gru"], mask)
        for column, cube in enumerate(cubes):
            axes[row, column].imshow(_rgb(cube, mask, float(low), float(high)))
        residual = np.abs(data["phenology_gru"] - data["observed"]).mean(axis=0)
        residual[~mask] = np.nan
        image = axes[row, 5].imshow(
            residual,
            cmap=residual_cmap,
            vmin=0,
            vmax=max(0.06, float(np.nanquantile(residual, 0.98))),
        )
        axes[row, 0].set_ylabel(
            path.stem.split("_")[1] + f"\nMAE {proposed_mae:.3f}\nSAM {proposed_sam:.3f}"
        )
        for column in range(len(columns)):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(columns[column])
        if row == len(files) - 1:
            cax = axes[row, 5].inset_axes([0.08, -0.20, 0.84, 0.07])
            plt.colorbar(image, cax=cax, orientation="horizontal")
    fig.subplots_adjust(left=0.08, right=0.995, top=0.94, bottom=0.06, wspace=0.025, hspace=0.06)
    save(fig, "figure_05_normal_test_cases")


def make_figure_06(bootstrap: pd.DataFrame) -> None:
    selections = pd.read_csv(DATA / "evaluation" / "phenology_basis_selections.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.08, 2.65), gridspec_kw={"wspace": 0.34})
    frame = bootstrap[bootstrap["baseline"].eq("validation_selected_gru")]
    strata = ["overall", "broadleaf", "conifer", "bamboo"]
    for offset, (metric, color) in enumerate((("mae", BLUE), ("sam_degrees", GREEN))):
        selected = frame[frame["metric"].eq(metric)].set_index("stratum").loc[strata]
        scale = 1000 if metric == "mae" else 1
        y = np.arange(len(strata)) + (offset - 0.5) * 0.16
        values = selected["gain"].to_numpy() * scale
        low = selected["ci_low"].to_numpy() * scale
        high = selected["ci_high"].to_numpy() * scale
        axes[0].errorbar(
            values,
            y,
            xerr=[values - low, high - values],
            fmt="o",
            color=color,
            capsize=2.5,
            label="MAE x 10$^{-3}$" if metric == "mae" else "SAM degrees",
        )
    axes[0].axvline(0, color=GRAY, ls="--", lw=0.8)
    axes[0].set_yticks(np.arange(len(strata)), [name.title() for name in strata])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Gain over uncorrected GRU")
    axes[0].grid(axis="x", color=GRID, lw=0.5)
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=6.6,
    )
    panel(axes[0], "a")
    basis_order = [
        "routed_recent_season",
        "analog_s45_h2",
        "analog_s90_h4",
        "harmonic_1",
        "harmonic_2_trend",
    ]
    matrix = np.zeros((3, len(basis_order)), dtype=int)
    classes = ["broadleaf", "conifer", "bamboo"]
    for r, name in enumerate(classes):
        counts = selections[selections["forest_class"].eq(name)]["selected_basis"].value_counts()
        matrix[r] = [int(counts.get(basis, 0)) for basis in basis_order]
    axes[1].imshow(matrix, cmap="Greens", vmin=0, vmax=5)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            axes[1].text(c, r, str(matrix[r, c]), ha="center", va="center")
    axes[1].set_yticks(range(3), [name.title() for name in classes])
    axes[1].set_xticks(
        range(len(basis_order)), ["Recent", "A45", "A90", "H1", "H2T"], rotation=35, ha="right"
    )
    axes[1].set_title("Validation selections across five folds")
    panel(axes[1], "b")
    save(fig, "figure_06_forest_phenology_correction")


def make_figure_07(events: pd.DataFrame, catalog: pd.DataFrame) -> None:
    methods = [
        "direct_pair",
        "seasonal_median",
        "shared_residual_ssm",
        "gru_h8",
        "phenology_basis_gru",
    ]
    catalog = catalog.sort_values("event_code")
    event_ids = catalog["window_id"].tolist()
    fig, axes = plt.subplots(
        1, 2, figsize=(7.08, 2.75), gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.32}
    )
    offsets = np.linspace(-0.24, 0.24, len(methods))
    markers = ["o", "s", "^", "D", "P"]
    for offset, marker, method in zip(offsets, markers, methods):
        selected = events[events["method"].eq(method)].set_index("window_id")
        values = np.asarray([selected.loc[event, "auroc"] for event in event_ids], dtype=float)
        x = np.arange(len(event_ids)) + offset
        axes[0].vlines(x, 0, values, color=METHOD_COLORS[method], alpha=0.25, lw=0.8)
        axes[0].scatter(
            x,
            values,
            marker=marker,
            s=26,
            color=METHOD_COLORS[method],
            ec="white",
            lw=0.4,
            label=METHOD_LABELS[method],
        )
    axes[0].axhline(0.5, color=GRAY, ls="--", lw=0.8)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Pixel-ranking AUROC")
    short_names = {
        "E1": "Bailu\n2019",
        "E2": "0918 EQ\n2022",
        "E3": "Doksuri\n2023",
        "E4": "Gaemi\n2024",
        "E5": "Krathon\n2024",
    }
    axes[0].set_xticks(
        np.arange(len(event_ids)),
        [f"{row.event_code}\n{short_names[row.event_code]}" for row in catalog.itertuples()],
    )
    axes[0].grid(axis="y", color=GRID, lw=0.5)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, frameon=False, fontsize=6.2)
    panel(axes[0], "a")
    aggregate = (
        events.groupby("method")
        .agg(AUROC=("auroc", "mean"), AUPRC=("auprc", "mean"), IoU=("topk_iou", "mean"))
        .loc[methods]
    )
    y = np.arange(len(methods))
    for metric, marker, offset in (("AUROC", "o", -0.15), ("AUPRC", "s", 0), ("IoU", "D", 0.15)):
        axes[1].scatter(
            aggregate[metric],
            y + offset,
            marker=marker,
            s=25,
            c=[METHOD_COLORS[m] for m in methods],
            label=metric,
        )
        for index, value in enumerate(aggregate[metric]):
            axes[1].hlines(
                index + offset, 0, value, color=METHOD_COLORS[methods[index]], alpha=0.25, lw=0.7
            )
    axes[1].set_xlim(0, 1.0)
    axes[1].set_yticks(y, ["Direct", "Seasonal", "Shared SSM", "GRU", "Proposed"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean event metric")
    axes[1].grid(axis="x", color=GRID, lw=0.5)
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
        fontsize=6.6,
    )
    panel(axes[1], "b")
    save(fig, "figure_07_retrospective_event_metrics")


def _event_crop(mask: np.ndarray, minimum: int = 24, margin: int = 5, maximum: int = 40):
    rows, columns = np.where(mask)
    if not len(rows):
        return slice(0, mask.shape[0]), slice(0, mask.shape[1])
    size = min(
        max(minimum, int(np.ptp(rows) + 1 + 2 * margin), int(np.ptp(columns) + 1 + 2 * margin)),
        maximum,
    )
    center_r = round((rows.min() + rows.max()) / 2)
    center_c = round((columns.min() + columns.max()) / 2)
    row0 = int(np.clip(center_r - size // 2, 0, mask.shape[0] - size))
    col0 = int(np.clip(center_c - size // 2, 0, mask.shape[1] - size))
    return slice(row0, row0 + size), slice(col0, col0 + size)


def make_figure_08(catalog: pd.DataFrame) -> None:
    catalog = catalog.sort_values("event_code")
    columns = [
        "Previous reference",
        "Target observed",
        "Direct difference",
        "GRU residual",
        "Proposed residual",
    ]
    fig, axes = plt.subplots(len(catalog), len(columns), figsize=(7.08, 5.25), squeeze=False)
    residual_cmap = mpl.colormaps["magma"].copy()
    residual_cmap.set_bad("#D0D0D0")
    residual_image = None
    for row, event in enumerate(catalog.itertuples()):
        path = DATA / "cases" / "events" / f"{event.window_id}.npz"
        with np.load(path, allow_pickle=False) as source:
            data = {key: source[key] for key in source.files}
        event_mask = data["event_mask"].astype(bool)
        history_support = np.zeros_like(data["target_valid"], dtype=bool)
        history_support[data["history_support_rows"], data["history_support_columns"]] = True
        support = (
            data["target_valid"].astype(bool) & data["forest_mask"].astype(bool) & history_support
        )
        crop = _event_crop(event_mask)
        mask = support[crop]
        target = data["target_s2"][:, crop[0], crop[1]]
        previous = data["prediction_direct_pair"][:, crop[0], crop[1]]
        values = np.concatenate(
            [target[[2, 1, 0]][:, mask].ravel(), previous[[2, 1, 0]][:, mask].ravel()]
        )
        low, high = np.quantile(values, [0.02, 0.98]) if values.size else (0.0, 0.3)
        axes[row, 0].imshow(_rgb(previous, mask, float(low), float(high)))
        axes[row, 1].imshow(_rgb(target, mask, float(low), float(high)))
        residuals = [
            data["residual_direct_pair"][crop],
            data["residual_gru_h8"][crop],
            data["residual_phenology_basis_gru"][crop],
        ]
        vmax = max(
            0.08, float(np.nanquantile(np.concatenate([value[mask] for value in residuals]), 0.98))
        )
        for column, residual in enumerate(residuals, start=2):
            shown = residual.astype(float).copy()
            shown[~mask] = np.nan
            residual_image = axes[row, column].imshow(shown, cmap=residual_cmap, vmin=0, vmax=vmax)
        event_crop = event_mask[crop]
        for column in range(len(columns)):
            axes[row, column].contour(event_crop, levels=[0.5], colors=["#00BFC4"], linewidths=0.8)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(columns[column])
        axes[row, 0].set_ylabel(f"{event.event_code}\n{event.event_name}\n{event.area_ha:.3f} ha")
    if residual_image is not None:
        cax = fig.add_axes([0.42, 0.025, 0.42, 0.012])
        fig.colorbar(
            residual_image,
            cax=cax,
            orientation="horizontal",
            label="Mean absolute six-band residual",
        )
    fig.subplots_adjust(left=0.095, right=0.995, top=0.95, bottom=0.07, wspace=0.03, hspace=0.07)
    save(fig, "figure_08_retrospective_event_cases")


def make_all(
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    events: pd.DataFrame,
    catalog: pd.DataFrame,
) -> None:
    setup_style()
    make_figure_01()
    make_figure_02()
    make_figure_03()
    make_figure_04(summary, bootstrap)
    make_figure_05()
    make_figure_06(bootstrap)
    make_figure_07(events, catalog)
    make_figure_08(catalog)
