"""Evaluate the 600-task terrain LP-ACRL space with EPTE-SP metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from types import MethodType

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate terrain LP-ACRL tasks with EPTE-SP.")
parser.add_argument("--task", default="Unitree-Go2-Adaptive-Energy-Terrain-LPACRL")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--envs_per_task", type=int, default=1)
parser.add_argument("--episode_steps", type=int, default=1000)
parser.add_argument("--minimum_survival_steps", type=int, default=900)
parser.add_argument("--epte_threshold", type=float, default=0.30)
parser.add_argument("--vx_error_floor", type=float, default=0.20)
parser.add_argument("--yaw_error_floor", type=float, default=0.20)
parser.add_argument(
    "--positive_only",
    action="store_true",
    help="Use positive vx/yaw instead of independently randomized signs.",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", default=None)
parser.add_argument("--disable_fabric", action="store_true")
parser.add_argument("--no_plots", action="store_true", default=False, help="Only export CSV/JSON files.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.envs_per_task < 1 or args_cli.episode_steps < 1:
    parser.error("--envs_per_task and --episode_steps must be positive.")
if not 0 <= args_cli.minimum_survival_steps <= args_cli.episode_steps:
    parser.error("--minimum_survival_steps must be within the evaluation horizon.")
if args_cli.epte_threshold < 0.0 or min(args_cli.vx_error_floor, args_cli.yaw_error_floor) <= 0.0:
    parser.error("EPTE threshold must be non-negative and error floors must be positive.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from unitree_rl_lab.tasks.locomotion.robots.go2.adaptive_energy_lpacrl_terrain_cfg import (
    LPACRL_COLUMNS_PER_TYPE,
    LPACRL_TERRAIN_NAMES,
)
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


VX_EDGES = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
YAW_EDGES = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
NUM_LEVELS = 4


def _as_torch(value):
    return getattr(value, "torch", value)


def _task_id(vx_bin: int, yaw_bin: int, terrain_index: int, terrain_level: int) -> int:
    return (((vx_bin * (len(YAW_EDGES) - 1) + yaw_bin) * len(LPACRL_TERRAIN_NAMES) + terrain_index)
            * NUM_LEVELS + terrain_level)


def _build_cases() -> list[dict]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args_cli.seed)
    rows = []
    for vx_bin in range(len(VX_EDGES) - 1):
        for yaw_bin in range(len(YAW_EDGES) - 1):
            for terrain_index, terrain_name in enumerate(LPACRL_TERRAIN_NAMES):
                for terrain_level in range(NUM_LEVELS):
                    task_id = _task_id(vx_bin, yaw_bin, terrain_index, terrain_level)
                    for replicate in range(args_cli.envs_per_task):
                        vx = VX_EDGES[vx_bin] + torch.rand((), generator=generator).item() * (
                            VX_EDGES[vx_bin + 1] - VX_EDGES[vx_bin]
                        )
                        yaw = YAW_EDGES[yaw_bin] + torch.rand((), generator=generator).item() * (
                            YAW_EDGES[yaw_bin + 1] - YAW_EDGES[yaw_bin]
                        )
                        if not args_cli.positive_only:
                            vx *= -1.0 if torch.rand((), generator=generator).item() < 0.5 else 1.0
                            yaw *= -1.0 if torch.rand((), generator=generator).item() < 0.5 else 1.0
                        rows.append(
                            {
                                "task_id": task_id,
                                "vx_bin": vx_bin,
                                "yaw_bin": yaw_bin,
                                "terrain_index": terrain_index,
                                "terrain_type": terrain_name,
                                "terrain_level": terrain_level,
                                "replicate": replicate,
                                "command_vx": vx,
                                "command_yaw": yaw,
                                "terrain_column": terrain_index * LPACRL_COLUMNS_PER_TYPE
                                + replicate % LPACRL_COLUMNS_PER_TYPE,
                            }
                        )
    return rows


def _configure_nominal_evaluation(env_cfg) -> None:
    env_cfg.observations.policy.enable_corruption = False
    for event_name in ("physics_material", "add_base_mass", "push_robot", "base_external_force_torque"):
        if hasattr(env_cfg.events, event_name):
            setattr(env_cfg.events, event_name, None)
    if getattr(env_cfg.events, "reset_base", None) is not None:
        env_cfg.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        env_cfg.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
    if getattr(env_cfg.events, "reset_robot_joints", None) is not None:
        env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


def _install_fixed_commands(env, commands: torch.Tensor) -> None:
    term = env.unwrapped.command_manager.get_term("base_velocity")
    commands = commands.to(device=term.device, dtype=torch.float)

    def _resample_fixed(self, env_ids):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.vel_command_b[env_ids] = commands[env_ids]
        self.is_standing_env[env_ids] = False
        self.is_heading_env[env_ids] = False

    term._resample_command = MethodType(_resample_fixed, term)
    term._resample_command(torch.arange(env.unwrapped.num_envs, device=term.device))


def _load_policy(env, agent_cfg, checkpoint: str):
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(
        checkpoint,
        load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False},
        map_location=agent_cfg.device,
    )
    return runner.get_inference_policy(device=env.unwrapped.device)


def _safe_mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return sum(values) / len(values)


def _plot_results(summary: list[dict], output_dir: Path) -> None:
    """Render the task space as nested heatmaps and marginal success rates."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    lookup = {(row["vx_bin"], row["yaw_bin"], row["terrain_type"], row["terrain_level"]): row for row in summary}
    terrain_names = list(dict.fromkeys(row["terrain_type"] for row in summary))
    levels = sorted({row["terrain_level"] for row in summary})
    n_vx = len(VX_EDGES) - 1
    n_yaw = len(YAW_EDGES) - 1
    vx_labels = [f"{VX_EDGES[index]:g}-{VX_EDGES[index + 1]:g}" for index in range(n_vx)]
    yaw_labels = [f"{YAW_EDGES[index]:g}-{YAW_EDGES[index + 1]:g}" for index in range(n_yaw)]

    def grid(metric: str, terrain: str, level: int):
        data = np.full((n_yaw, n_vx), np.nan)
        for yaw_bin in range(n_yaw):
            for vx_bin in range(n_vx):
                row = lookup.get((vx_bin, yaw_bin, terrain, level))
                if row is not None:
                    data[yaw_bin, vx_bin] = row[metric]
        return np.ma.masked_invalid(data)

    # One figure per metric: rows are terrain families, columns are geometry
    # levels, and every cell is the 5x5 vx-by-yaw command grid of that
    # (family, level) slice. Green means success / low EPTE.
    for metric, title, cmap, vmin, vmax, filename in (
        ("success_rate", "Task success rate", "RdYlGn", 0.0, 1.0, "lpacrl_task_space_success.png"),
        ("mean_epte_sp_vx", "Mean EPTE-SP, forward speed (lower is better)", "RdYlGn_r", 0.0, 1.0, "lpacrl_task_space_epte_vx.png"),
    ):
        figure, axes = plt.subplots(
            len(terrain_names),
            len(levels),
            figsize=(2.1 * len(levels) + 3.0, 1.7 * len(terrain_names) + 1.6),
            constrained_layout=True,
            squeeze=False,
        )
        last_image = None
        for row_index, terrain in enumerate(terrain_names):
            for column_index, level in enumerate(levels):
                axis = axes[row_index][column_index]
                last_image = axis.imshow(grid(metric, terrain, level), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
                if row_index == len(terrain_names) - 1:
                    axis.set_xticks(range(n_vx), vx_labels, rotation=45, fontsize=6)
                    axis.set_xlabel("commanded |vx| [m/s]", fontsize=7)
                else:
                    axis.set_xticks([])
                if column_index == 0:
                    axis.set_yticks(range(n_yaw), yaw_labels, fontsize=6)
                    axis.set_ylabel(terrain, fontsize=9)
                else:
                    axis.set_yticks([])
                if row_index == 0:
                    axis.set_title(f"level {level}", fontsize=9)
            figure.colorbar(last_image, ax=axes[row_index].tolist(), fraction=0.02, pad=0.01)
        figure.suptitle(title, fontsize=12)
        figure.savefig(output_dir / filename, dpi=180)
        plt.close(figure)

    # Marginal success and survival rates along the four task dimensions.
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    def marginal(selector, labels, axis, tick_labels):
        success = [np.mean([row["success_rate"] for row in summary if selector(row) == value]) for value in labels]
        survival = [
            np.mean([row["mean_survived_steps"] for row in summary if selector(row) == value]) / args_cli.episode_steps
            for value in labels
        ]
        axis.plot(range(len(labels)), success, marker="o", label="success rate")
        axis.plot(range(len(labels)), survival, marker="s", label="survived fraction")
        axis.set_xticks(range(len(labels)), tick_labels, fontsize=8)
        axis.set_ylim(-0.05, 1.05)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)

    marginal(lambda row: row["vx_bin"], list(range(n_vx)), axes[0][0], vx_labels)
    axes[0][0].set_xlabel("commanded |vx| bin [m/s]")
    marginal(lambda row: row["yaw_bin"], list(range(n_yaw)), axes[0][1], yaw_labels)
    axes[0][1].set_xlabel("commanded |yaw rate| bin [rad/s]")
    marginal(lambda row: row["terrain_type"], terrain_names, axes[1][0], terrain_names)
    axes[1][0].set_xlabel("terrain family")
    marginal(lambda row: row["terrain_level"], levels, axes[1][1], [f"L{level}" for level in levels])
    axes[1][1].set_xlabel("terrain geometry level")
    figure.suptitle("Marginal success / survival across the 600-task space", fontsize=12)
    figure.savefig(output_dir / "lpacrl_task_space_marginals.png", dpi=180)
    plt.close(figure)


def main() -> None:
    cases = _build_cases()
    num_envs = len(cases)
    checkpoint = str(Path(args_cli.checkpoint).expanduser().resolve())
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.scene.num_envs = num_envs
    env_cfg.curriculum = None
    env_cfg.commands.base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        rel_standing_envs=0.0,
        debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-2.5, 2.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(-2.5, 2.5)
        ),
    )
    dt = float(env_cfg.decimation * env_cfg.sim.dt)
    # Keep the simulator timeout just beyond K, so reaching K never looks like a fall.
    env_cfg.episode_length_s = (args_cli.episode_steps + 10) * dt
    env_cfg.seed = args_cli.seed
    _configure_nominal_evaluation(env_cfg)

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    device = gym_env.unwrapped.device
    commands = torch.tensor(
        [[case["command_vx"], 0.0, case["command_yaw"]] for case in cases], device=device
    )
    terrain = gym_env.unwrapped.scene.terrain
    levels = torch.tensor([case["terrain_level"] for case in cases], dtype=torch.long, device=device)
    columns = torch.tensor([case["terrain_column"] for case in cases], dtype=torch.long, device=device)
    terrain.terrain_levels[:] = levels
    terrain.terrain_types[:] = columns
    terrain.env_origins[:] = terrain.terrain_origins[levels, columns]
    _install_fixed_commands(gym_env, commands)
    gym_env.reset()

    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    policy = _load_policy(env, agent_cfg, checkpoint)
    robot = env.unwrapped.scene["robot"]
    alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    survived_steps = torch.zeros(num_envs, device=device)
    normalized_error_sum = torch.zeros(num_envs, 2, device=device)
    absolute_error_sum = torch.zeros(num_envs, 2, device=device)

    observation = env.get_observations()
    observation = observation[0] if isinstance(observation, tuple) else observation
    for _ in range(args_cli.episode_steps):
        with torch.inference_mode():
            observation, _, dones, _ = env.step(policy(observation))
        valid = alive & ~dones.bool()
        alive &= ~dones.bool()
        lin_vel = _as_torch(robot.data.root_lin_vel_b)
        ang_vel = _as_torch(robot.data.root_ang_vel_b)
        error = torch.stack(
            (torch.abs(lin_vel[:, 0] - commands[:, 0]), torch.abs(ang_vel[:, 2] - commands[:, 2])), dim=1
        )
        floors = torch.tensor((args_cli.vx_error_floor, args_cli.yaw_error_floor), device=device)
        denominator = torch.maximum(torch.abs(commands[:, (0, 2)]), floors)
        survived_steps += valid.float()
        absolute_error_sum += torch.where(valid.unsqueeze(1), error, 0.0)
        normalized_error_sum += torch.where(valid.unsqueeze(1), error / denominator, 0.0)

    k_f = survived_steps
    horizon = float(args_cli.episode_steps)
    mae = absolute_error_sum / k_f.clamp_min(1.0).unsqueeze(1)
    mean_normalized_error = normalized_error_sum / k_f.clamp_min(1.0).unsqueeze(1)
    epte = (normalized_error_sum + (horizon - k_f).unsqueeze(1)) / horizon
    success = (
        (k_f >= args_cli.minimum_survival_steps)
        & (epte[:, 0] < args_cli.epte_threshold)
        & (epte[:, 1] < args_cli.epte_threshold)
    )

    rows = []
    for index, case in enumerate(cases):
        rows.append(
            case
            | {
                "survived_steps": int(k_f[index].item()),
                "survived_full_horizon": bool(alive[index].item()),
                "mae_vx": float(mae[index, 0].item()),
                "mae_yaw": float(mae[index, 1].item()),
                "normalized_error_vx": float(mean_normalized_error[index, 0].item()),
                "normalized_error_yaw": float(mean_normalized_error[index, 1].item()),
                "epte_sp_vx": float(epte[index, 0].item()),
                "epte_sp_yaw": float(epte[index, 1].item()),
                "success": bool(success[index].item()),
            }
        )

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["task_id"]].append(row)
    summary = []
    for task_id in sorted(grouped):
        group = grouped[task_id]
        first = group[0]
        summary.append(
            {
                "task_id": task_id,
                "vx_bin": first["vx_bin"],
                "yaw_bin": first["yaw_bin"],
                "terrain_type": first["terrain_type"],
                "terrain_level": first["terrain_level"],
                "rollouts": len(group),
                "success_rate": sum(row["success"] for row in group) / len(group),
                "mean_survived_steps": _safe_mean(group, "survived_steps"),
                "mean_mae_vx": _safe_mean(group, "mae_vx"),
                "mean_mae_yaw": _safe_mean(group, "mae_yaw"),
                "mean_epte_sp_vx": _safe_mean(group, "epte_sp_vx"),
                "mean_epte_sp_yaw": _safe_mean(group, "epte_sp_yaw"),
            }
        )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = (
        Path(args_cli.output_dir).expanduser().resolve()
        if args_cli.output_dir
        else Path(checkpoint).parent / "evaluations" / f"lpacrl_epte_sp_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in (("per_environment.csv", rows), ("task_summary.csv", summary)):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    metadata = vars(args_cli) | {
        "checkpoint": checkpoint,
        "num_tasks": len(grouped),
        "num_envs": num_envs,
        "overall_success_rate": float(success.float().mean().item()),
        "vx_edges": VX_EDGES,
        "yaw_edges": YAW_EDGES,
        "terrain_names": LPACRL_TERRAIN_NAMES,
        "num_terrain_levels": NUM_LEVELS,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)

    if not args_cli.no_plots:
        _plot_results(summary, output_dir)
        print(f"[INFO] Plots: {output_dir / 'lpacrl_task_space_success.png'} (plus epte/marginals)")

    print(f"[INFO] EPTE-SP evaluated {len(grouped)} tasks with {num_envs} rollouts.")
    print(f"[INFO] Overall success rate: {metadata['overall_success_rate']:.4f}")
    print(f"[INFO] Results: {output_dir}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
