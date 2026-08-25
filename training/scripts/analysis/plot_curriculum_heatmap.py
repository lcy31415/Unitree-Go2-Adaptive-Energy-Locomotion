"""Visualize LP-ACRL curriculum expansion from the per-checkpoint CSV logs.

Each ``model_<iter>.csv`` under ``<log_dir>/curriculum/`` stores one row per
discrete task (vx_bin, vy_bin, yaw_bin) with its sampling probability,
learning progress and reward EMA at that point in training. Bins index the
MAGNITUDE of each command; the sign is sampled uniformly at run time.

Outputs (written next to the CSVs by default):
  curriculum_probability.png  - snapshot grid of P(|vx|, |yaw|) summed over vy
  curriculum_lp_reward.png    - learning progress / reward EMA at the last checkpoint
  curriculum_marginals.png    - per-checkpoint marginal P(|vx|), P(|vy|), P(|yaw|)
  curriculum.gif              - optional animation (--gif)

Usage:
  python plot_curriculum_heatmap.py <run_dir_or_curriculum_dir> [--gif]
      [--vx-edges 0,0.5,...] [--vy-edges 0,0.2,0.4,0.6] [--yaw-edges 0,0.5,...]
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

DEFAULT_VX_EDGES = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
DEFAULT_VY_EDGES = (0.0, 0.2, 0.4, 0.6)
DEFAULT_YAW_EDGES = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)


def find_csvs(root: str) -> list[tuple[int, str]]:
    if os.path.basename(root) == "curriculum":
        cdir = root
    else:
        cdir = os.path.join(root, "curriculum")
    found = []
    for path in glob.glob(os.path.join(cdir, "model_*.csv")):
        m = re.search(r"model_(\d+)\.csv$", path)
        if m:
            found.append((int(m.group(1)), path))
    if not found:
        raise FileNotFoundError(f"No model_*.csv found under {cdir}")
    return sorted(found)


def edge_labels(edges: tuple[float, ...]) -> list[str]:
    return [f"{edges[i]:g}-{edges[i + 1]:g}" for i in range(len(edges) - 1)]


def load_all(checkpoints: list[tuple[int, str]]) -> dict[int, pd.DataFrame]:
    return {it: pd.read_csv(path) for it, path in checkpoints}


def grid_2d(df: pd.DataFrame, value: str, edges: tuple[tuple[float, ...], ...]) -> np.ndarray:
    """Pivot task table to a (vx_bin, yaw_bin) grid summed over vy_bin."""
    nx = len(edges[0]) - 1
    nw = len(edges[2]) - 1
    grid = np.zeros((nx, nw))
    np.add.at(grid, (df.vx_bin.to_numpy(), df.yaw_bin.to_numpy()), df[value].to_numpy())
    return grid


def plot_probability_grid(data: dict[int, pd.DataFrame], edges, out_path: str) -> None:
    iters = sorted(data)
    n = len(iters)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    uniform = 1.0 / len(data[iters[0]])
    vmax = max(grid_2d(df, "probability", edges).max() for df in data.values())

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.0 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, it in enumerate(iters):
        ax = axes[k // ncols][k % ncols]
        grid = grid_2d(data[it], "probability", edges)
        im = ax.imshow(grid.T, origin="lower", cmap="viridis", vmin=0.0, vmax=vmax, aspect="auto")
        ax.set_title(f"iter {it}", fontsize=10)
        ax.set_xticks(range(len(edges[0]) - 1))
        ax.set_xticklabels(edge_labels(edges[0]), rotation=90, fontsize=6)
        ax.set_yticks(range(len(edges[2]) - 1))
        ax.set_yticklabels(edge_labels(edges[2]), fontsize=6)
        if k % ncols == 0:
            ax.set_ylabel("|yaw| bin (rad/s)", fontsize=8)
        if k // ncols == nrows - 1:
            ax.set_xlabel("|vx| bin (m/s)", fontsize=8)
        ax.axis("on")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.015)
    cbar.set_label("P(|vx|, |yaw|) summed over |vy|  (uniform = %.4f)" % uniform)
    fig.suptitle("LP-ACRL sampling probability over command bins", y=0.995)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_lp_reward(data: dict[int, pd.DataFrame], edges, out_path: str) -> None:
    it = sorted(data)[-1]
    df = data[it]
    nx, ny, nw = (len(e) - 1 for e in edges)
    fig, axes = plt.subplots(2, ny, figsize=(3.4 * ny, 6.2), squeeze=False)
    for col, vy in enumerate(range(ny)):
        sub = df[df.vy_bin == vy]
        for row, value in enumerate(("learning_progress", "reward_ema")):
            ax = axes[row][col]
            grid = np.zeros((nx, nw))
            np.add.at(grid, (sub.vx_bin.to_numpy(), sub.yaw_bin.to_numpy()), sub[value].to_numpy())
            vmax = np.abs(grid).max() or 1.0
            if value == "learning_progress":
                vmax = max(abs(np.nanmin(grid)), abs(np.nanmax(grid))) or 1.0
                im = ax.imshow(grid.T, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            else:
                im = ax.imshow(grid.T, origin="lower", cmap="magma", vmin=0.0, vmax=vmax)
            ax.set_title(f"{value}  |vy| {edge_labels(edges[1])[vy]} m/s", fontsize=9)
            ax.set_xticks(range(nx))
            ax.set_xticklabels(edge_labels(edges[0]), rotation=90, fontsize=6)
            ax.set_yticks(range(nw))
            ax.set_yticklabels(edge_labels(edges[2]), fontsize=6)
            if col == 0:
                ax.set_ylabel("|yaw| bin (rad/s)", fontsize=8)
            if row == 1:
                ax.set_xlabel("|vx| bin (m/s)", fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"LP-ACRL task statistics at iter {it}", y=1.0)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_marginals(data: dict[int, pd.DataFrame], edges, out_path: str) -> None:
    iters = sorted(data)
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    names = ("vx_bin", "vy_bin", "yaw_bin")
    axis_names = ("|vx| bin upper edge (m/s)", "|vy| bin upper edge (m/s)", "|yaw| bin upper edge (rad/s)")
    for j, (name, ax) in enumerate(zip(names, axes)):
        nbins = len(edges[j]) - 1
        uppers = edges[j][1:]
        for k, it in enumerate(iters):
            df = data[it]
            marginal = np.zeros(nbins)
            np.add.at(marginal, df[name].to_numpy(), df.probability.to_numpy())
            ax.plot(uppers, marginal, marker="o", ms=3, lw=1.4, color=cmap(k / max(len(iters) - 1, 1)),
                    alpha=0.85, label=f"{it}" if k in (0, len(iters) - 1) else None)
        ax.axhline(1.0 / nbins, color="gray", ls=":", lw=1, label="uniform")
        ax.set_xlabel(axis_names[j])
        ax.set_ylabel("marginal probability")
        ax.set_title(f"marginal over {name.split('_')[0]}")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, title="iteration", loc="upper right")
    fig.suptitle("Curriculum expansion: command magnitude marginals over training", y=1.02)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_gif(data: dict[int, pd.DataFrame], edges, out_path: str) -> None:
    from matplotlib.animation import FuncAnimation, PillowWriter

    iters = sorted(data)
    nx, ny, nw = (len(e) - 1 for e in edges)
    vmax = max(df[df.vy_bin == v].groupby(["vx_bin", "yaw_bin"]).probability.sum().max()
               for df in data.values() for v in range(ny))
    fig, axes = plt.subplots(1, ny, figsize=(3.2 * ny, 3.4), squeeze=False)
    axes = axes[0]
    ims = []
    for v, ax in enumerate(axes):
        grid = np.zeros((nx, nw))
        np.add.at(grid, (data[iters[0]][data[iters[0]].vy_bin == v].vx_bin.to_numpy(),
                         data[iters[0]][data[iters[0]].vy_bin == v].yaw_bin.to_numpy()),
                  data[iters[0]][data[iters[0]].vy_bin == v].probability.to_numpy())
        im = ax.imshow(grid.T, origin="lower", cmap="viridis", vmin=0.0, vmax=vmax, aspect="auto")
        ims.append(im)
        ax.set_title(f"|vy| {edge_labels(edges[1])[v]} m/s", fontsize=9)
        ax.set_xticks(range(nx))
        ax.set_xticklabels(edge_labels(edges[0]), rotation=90, fontsize=6)
        ax.set_yticks(range(nw))
        ax.set_yticklabels(edge_labels(edges[2]), fontsize=6)
        ax.set_xlabel("|vx| bin (m/s)", fontsize=8)
        ax.set_ylabel("|yaw| bin (rad/s)", fontsize=8)
    title = fig.suptitle("", y=0.98)
    cbar = fig.colorbar(ims[0], ax=axes.tolist(), shrink=0.85)
    cbar.set_label("task probability")

    def update(k):
        it = iters[k]
        df = data[it]
        for v, im in enumerate(ims):
            sub = df[df.vy_bin == v]
            grid = np.zeros((nx, nw))
            np.add.at(grid, (sub.vx_bin.to_numpy(), sub.yaw_bin.to_numpy()), sub.probability.to_numpy())
            im.set_data(grid.T)
        p = df.probability.to_numpy()
        eff = np.exp(-(p * np.log(np.clip(p, 1e-12, None))).sum())
        title.set_text(f"iter {it}   effective tasks = {eff:.0f} / {len(df)}")
        return ims + [title]

    anim = FuncAnimation(fig, update, frames=len(iters), interval=900, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=1.1))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_dir", help="Run directory containing curriculum/, or the curriculum dir itself")
    parser.add_argument("--gif", action="store_true", help="Also write an animated GIF")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: the curriculum dir)")
    parser.add_argument("--vx-edges", default=",".join(map(str, DEFAULT_VX_EDGES)))
    parser.add_argument("--vy-edges", default=",".join(map(str, DEFAULT_VY_EDGES)))
    parser.add_argument("--yaw-edges", default=",".join(map(str, DEFAULT_YAW_EDGES)))
    args = parser.parse_args()

    edges = tuple(
        tuple(float(v) for v in s.split(",")) for s in (args.vx_edges, args.vy_edges, args.yaw_edges)
    )
    for e, name in zip(edges, ("vx", "vy", "yaw")):
        expected = len(e) - 1
        if expected <= 0:
            parser.error(f"{name} needs at least 2 edges")

    checkpoints = find_csvs(args.log_dir)
    data = load_all(checkpoints)
    n_bins_csv = len(data[sorted(data)[0]])
    if n_bins_csv != np.prod([len(e) - 1 for e in edges]):
        print(f"WARNING: {n_bins_csv} tasks in CSVs but edges define {np.prod([len(e) - 1 for e in edges])}; "
              "labels may be wrong - pass --vx-edges/--vy-edges/--yaw-edges explicitly.")

    out_dir = args.out_dir or os.path.dirname(checkpoints[0][1])
    os.makedirs(out_dir, exist_ok=True)
    plot_probability_grid(data, edges, os.path.join(out_dir, "curriculum_probability.png"))
    plot_lp_reward(data, edges, os.path.join(out_dir, "curriculum_lp_reward.png"))
    plot_marginals(data, edges, os.path.join(out_dir, "curriculum_marginals.png"))
    if args.gif:
        plot_gif(data, edges, os.path.join(out_dir, "curriculum.gif"))

    iters = sorted(data)
    print(f"Wrote figures to {out_dir} from {len(iters)} checkpoints (iter {iters[0]} .. {iters[-1]}).")
    for it in iters:
        p = data[it].probability.to_numpy()
        eff = np.exp(-(p * np.log(np.clip(p, 1e-12, None))).sum())
        top = data[it].nlargest(1, "probability").iloc[0]
        print(f"  iter {it:>5}: effective tasks {eff:6.1f}, max P {p.max():.4f} at "
              f"|vx| bin {int(top.vx_bin)} |vy| bin {int(top.vy_bin)} |yaw| bin {int(top.yaw_bin)}")


if __name__ == "__main__":
    main()
