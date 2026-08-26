"""Publication-style plots for terrain-tracking evaluation results.

This module deliberately has no Isaac Lab dependency, so an existing
evaluation directory can be re-plotted from its CSV/JSON files on any machine.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TERRAIN_LABELS = {
    "flat": "Flat",
    "stairs_up": "Up stairs",
    "stairs_down": "Down stairs",
    "slope_up": "Up slope",
    "slope_down": "Down slope",
    "random_rough": "Random rough",
}

LEVEL_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
BACKGROUND = "#F7F8FA"
GRID_COLOR = "#CBD0D6"
TEXT_COLOR = "#25313C"


def _setup_style(plt) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": BACKGROUND,
            "axes.edgecolor": "#8B949E",
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID_COLOR,
            "grid.alpha": 0.45,
            "grid.linewidth": 0.7,
            "font.size": 10,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )


def _finite(values, np):
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _pretty_terrain(name: str) -> str:
    return TERRAIN_LABELS.get(name, name.replace("_", " ").title())


def _save(figure, path: Path) -> None:
    figure.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.12)


def plot_results(
    summary,
    terrain_names,
    levels,
    speeds,
    output_dir: Path,
    cases=None,
    dt: float = 0.02,
    vx_history=None,
) -> None:
    """Render readable terrain/level/speed diagnostics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm, Normalize

    _setup_style(plt)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    terrain_names = list(terrain_names)
    levels = sorted(set(levels))
    speeds = sorted(set(speeds))
    lookup = {(row["terrain_type"], row["terrain_level"], row["command_vx"]): row for row in summary}

    def terrain_matrix(terrain: str, metric: str):
        data = np.full((len(levels), len(speeds)), np.nan)
        for row_index, level in enumerate(levels):
            for column_index, speed in enumerate(speeds):
                row = lookup.get((terrain, level, speed))
                if row is not None:
                    data[row_index, column_index] = row.get(metric, np.nan)
        return np.ma.masked_invalid(data)

    all_values = {
        metric: _finite([row.get(metric, np.nan) for row in summary], np)
        for metric in ("mae_vx", "success_rate", "mechanical_cot")
    }
    mae_positive = all_values["mae_vx"][all_values["mae_vx"] > 0.0]
    mae_vmin = max(0.01, float(np.nanpercentile(mae_positive, 5))) if mae_positive.size else 0.01
    mae_vmax = max(0.2, float(np.nanpercentile(mae_positive, 95))) if mae_positive.size else 1.0
    cot_values = all_values["mechanical_cot"]
    cot_vmax = max(0.5, float(np.nanpercentile(cot_values, 95))) if cot_values.size else 1.0

    heatmap_rows = (
        ("mae_vx", "Forward-speed MAE", "m/s", "YlOrRd", LogNorm(mae_vmin, mae_vmax)),
        ("success_rate", "Course success / survival", "rate", "RdYlGn", Normalize(0.0, 1.0)),
        ("mechanical_cot", "Mechanical cost of transport", "CoT", "YlOrBr", Normalize(0.0, cot_vmax)),
    )
    figure, axes = plt.subplots(
        len(heatmap_rows),
        len(terrain_names),
        figsize=(3.7 * len(terrain_names), 8.6),
        constrained_layout=True,
        squeeze=False,
    )
    tick_indices = list(range(0, len(speeds), 2))
    if tick_indices[-1] != len(speeds) - 1:
        tick_indices.append(len(speeds) - 1)
    for row_index, (metric, title, colorbar_label, cmap_name, norm) in enumerate(heatmap_rows):
        images = []
        for column_index, terrain in enumerate(terrain_names):
            axis = axes[row_index, column_index]
            cmap = plt.get_cmap(cmap_name).copy()
            cmap.set_bad("#E5E8EC")
            image = axis.imshow(terrain_matrix(terrain, metric), aspect="auto", cmap=cmap, norm=norm)
            images.append(image)
            axis.grid(False)
            axis.set_yticks(range(len(levels)), [f"L{level}" for level in levels])
            if column_index == 0:
                axis.set_ylabel(title)
            else:
                axis.tick_params(labelleft=False)
            if row_index == 0:
                axis.set_title(_pretty_terrain(terrain), pad=8)
            if row_index == len(heatmap_rows) - 1:
                axis.set_xticks(tick_indices, [f"{speeds[index]:g}" for index in tick_indices], rotation=45)
                axis.set_xlabel("Command $v_x$ [m/s]")
            else:
                axis.tick_params(labelbottom=False)
        colorbar = figure.colorbar(images[-1], ax=axes[row_index, :], fraction=0.015, pad=0.012)
        colorbar.set_label(colorbar_label)
    figure.suptitle("Terrain tracking at a glance", fontsize=17, fontweight="bold")
    figure.text(0.5, -0.012, "Grey cells indicate no valid successful trials.", ha="center", color="#66717C")
    _save(figure, output_dir / "terrain_tracking_heatmaps.png")
    plt.close(figure)

    # MAE uses a logarithmic axis so rare runaway trials remain visible without
    # flattening the accurate 0.01--0.15 m/s region.
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 8.6), constrained_layout=True, sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for axis, terrain in zip(axes, terrain_names):
        for level_index, level in enumerate(levels):
            values = np.asarray(
                [lookup.get((terrain, level, speed), {}).get("mae_vx", np.nan) for speed in speeds],
                dtype=float,
            )
            values[values <= 0.0] = np.nan
            axis.plot(
                speeds,
                values,
                color=LEVEL_COLORS[level_index % len(LEVEL_COLORS)],
                marker="o",
                markersize=3.8,
                linewidth=1.7,
                label=f"Level {level}",
            )
        axis.axhline(0.15, color="#59636E", linestyle="--", linewidth=1.2)
        axis.set_title(_pretty_terrain(terrain))
        axis.set_yscale("log")
        axis.set_ylim(0.008, max(30.0, float(np.nanmax(mae_positive)) * 1.15 if mae_positive.size else 1.0))
        axis.set_xlabel("Command $v_x$ [m/s]")
        axis.set_ylabel("MAE $v_x$ [m/s]")
    for axis in axes[len(terrain_names) :]:
        axis.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.075), ncols=len(levels))
    figure.suptitle("Tracking error across terrain difficulty", y=1.145, fontsize=16, fontweight="bold")
    figure.text(0.995, -0.025, "Dashed line: 0.15 m/s target", ha="right", color="#66717C")
    _save(figure, output_dir / "terrain_tracking_curves.png")
    plt.close(figure)

    # Fixed physical axes make panels directly comparable. Runaway reverse
    # trials are pinned to the lower boundary and counted rather than allowed
    # to consume the entire plot range.
    lower, upper = -0.1, max(2.55, max(speeds) + 0.1)
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 9.0), constrained_layout=True, sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for axis, terrain in zip(axes, terrain_names):
        off_scale_count = 0
        for level_index, level in enumerate(levels):
            measured = np.asarray(
                [lookup.get((terrain, level, speed), {}).get("mean_vx", np.nan) for speed in speeds],
                dtype=float,
            )
            color = LEVEL_COLORS[level_index % len(LEVEL_COLORS)]
            in_range = np.isfinite(measured) & (measured >= lower) & (measured <= upper)
            line_values = np.where(in_range, measured, np.nan)
            axis.plot(speeds, line_values, color=color, linewidth=1.6, alpha=0.8)
            axis.scatter(np.asarray(speeds)[in_range], measured[in_range], color=color, s=24, label=f"Level {level}")
            below = np.isfinite(measured) & (measured < lower)
            above = np.isfinite(measured) & (measured > upper)
            off_scale_count += int(below.sum() + above.sum())
            axis.scatter(np.asarray(speeds)[below], np.full(below.sum(), lower + 0.025), marker="v", color=color, s=38)
            axis.scatter(np.asarray(speeds)[above], np.full(above.sum(), upper - 0.025), marker="^", color=color, s=38)
        axis.plot([0.0, upper], [0.0, upper], color="#59636E", linestyle="--", linewidth=1.2)
        if off_scale_count:
            axis.text(
                0.03,
                0.96,
                f"{off_scale_count} off-scale trial groups",
                transform=axis.transAxes,
                va="top",
                color="#A33A2B",
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.5},
            )
        axis.set_title(_pretty_terrain(terrain))
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Commanded $v_x$ [m/s]")
        axis.set_ylabel("Measured $v_x$ [m/s]")
    for axis in axes[len(terrain_names) :]:
        axis.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.075), ncols=len(levels))
    figure.suptitle("Command calibration", y=1.145, fontsize=16, fontweight="bold")
    figure.text(
        0.995,
        -0.025,
        "Triangles mark measurements beyond the shared axis range.",
        ha="right",
        color="#66717C",
    )
    _save(figure, output_dir / "terrain_tracking_calibration.png")
    plt.close(figure)

    # A compact report figure: average success and median successful-trial
    # tracking/energy across terrain levels.
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    overview_specs = (
        ("success_rate", "Mean success / survival", "Rate", "mean"),
        ("mae_vx", "Median tracking error", "MAE $v_x$ [m/s]", "median"),
        ("mechanical_cot", "Median mechanical cost", "Mechanical CoT", "median"),
    )
    terrain_palette = plt.get_cmap("tab10")
    for axis, (metric, title, ylabel, reduction) in zip(axes, overview_specs):
        for terrain_index, terrain in enumerate(terrain_names):
            reduced = []
            for speed in speeds:
                values = _finite(
                    [lookup.get((terrain, level, speed), {}).get(metric, np.nan) for level in levels],
                    np,
                )
                if not values.size:
                    reduced.append(np.nan)
                elif reduction == "mean":
                    reduced.append(float(np.mean(values)))
                else:
                    reduced.append(float(np.median(values)))
            axis.plot(
                speeds,
                reduced,
                color=terrain_palette(terrain_index),
                linewidth=2.0,
                label=_pretty_terrain(terrain),
            )
        axis.set_title(title)
        axis.set_xlabel("Command $v_x$ [m/s]")
        axis.set_ylabel(ylabel)
        if metric == "success_rate":
            axis.set_ylim(-0.03, 1.03)
        elif metric == "mae_vx":
            axis.set_yscale("log")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.09), ncols=min(6, len(labels)))
    figure.suptitle("LP-ACRL terrain tracking summary", y=1.18, fontsize=16, fontweight="bold")
    _save(figure, output_dir / "terrain_tracking_overview.png")
    plt.close(figure)

    if vx_history and cases is not None:
        history = np.stack(vx_history)
        time_s = np.arange(history.shape[0]) * dt
        case_index = {case: index for index, case in enumerate(cases)}
        middle_level = levels[len(levels) // 2]
        representative_indices = sorted(set(np.linspace(0, len(speeds) - 1, min(5, len(speeds))).round().astype(int)))
        representative_speeds = [speeds[index] for index in representative_indices]
        smooth_window = max(1, round(0.25 / dt))
        kernel = np.ones(smooth_window) / smooth_window
        figure, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), constrained_layout=True, sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()
        display_lower, display_upper = -1.0, max(speeds) + 0.5
        for axis, terrain in zip(axes, terrain_names):
            clipped_count = 0
            for speed_index, speed in enumerate(representative_speeds):
                index = case_index[(terrain, middle_level, speed, 0)]
                raw = history[:, index]
                color = plt.get_cmap("viridis")(speed_index / max(1, len(representative_speeds) - 1))
                smoothed = np.convolve(raw, kernel, mode="same")
                clipped_count += int(np.any((raw < display_lower) | (raw > display_upper)))
                axis.plot(time_s, np.clip(raw, display_lower, display_upper), color=color, alpha=0.12, linewidth=0.7)
                axis.plot(time_s, np.clip(smoothed, display_lower, display_upper), color=color, linewidth=1.8, label=f"{speed:g} m/s")
                axis.axhline(speed, color=color, linestyle="--", linewidth=0.75, alpha=0.65)
            if clipped_count:
                axis.text(0.98, 0.04, f"{clipped_count} traces clipped", transform=axis.transAxes, ha="right", color="#A33A2B")
            axis.set_title(f"{_pretty_terrain(terrain)} · level {middle_level}")
            axis.set_xlabel("Time [s]")
            axis.set_ylabel("Measured $v_x$ [m/s]")
            axis.set_ylim(display_lower, display_upper)
        for axis in axes[len(terrain_names) :]:
            axis.set_visible(False)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.075),
            ncols=len(representative_speeds),
        )
        figure.suptitle("Representative speed responses", y=1.145, fontsize=16, fontweight="bold")
        figure.text(
            0.995,
            -0.025,
            "Solid: 0.25 s moving average · faint: raw · dashed: command",
            ha="right",
            color="#66717C",
        )
        _save(figure, output_dir / "terrain_tracking_timeseries.png")
        plt.close(figure)


def _load_summary(path: Path):
    integer_fields = {"terrain_level", "attempts", "successful_attempts"}
    text_fields = {"terrain_type"}
    rows = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            parsed = {}
            for key, value in row.items():
                if key in text_fields:
                    parsed[key] = value
                elif key in integer_fields:
                    parsed[key] = int(value)
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def replot_directory(evaluation_dir: Path) -> None:
    evaluation_dir = evaluation_dir.expanduser().resolve()
    with (evaluation_dir / "metadata.json").open() as stream:
        metadata = json.load(stream)
    summary = _load_summary(evaluation_dir / "terrain_speed_summary.csv")
    plot_results(
        summary,
        metadata["terrain_types"],
        metadata["terrain_levels"],
        metadata["speeds"],
        evaluation_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replot a terrain-tracking evaluation directory.")
    parser.add_argument("evaluation_dir", type=Path)
    args = parser.parse_args()
    replot_directory(args.evaluation_dir)
    print(f"[INFO] Replotted terrain tracking results in: {args.evaluation_dir.resolve()}")


if __name__ == "__main__":
    main()
