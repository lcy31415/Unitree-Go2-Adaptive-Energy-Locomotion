"""Evaluate an adaptive-energy policy over terrain type, level and speed."""

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

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate terrain-conditioned speed tracking.")
parser.add_argument("--task", default="Unitree-Go2-Adaptive-Energy-Terrain-LPACRL")
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--speeds", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 2.5])
parser.add_argument("--terrain_levels", type=int, nargs="+", default=[0, 1, 2, 3])
parser.add_argument("--terrain_types", nargs="+", default=None, help="Names from adaptive_energy_terrain_cfg.py")
parser.add_argument("--envs_per_case", type=int, default=4)
parser.add_argument("--warmup_s", type=float, default=2.0)
parser.add_argument("--eval_s", type=float, default=10.0)
parser.add_argument("--output_dir", default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--disable_fabric", action="store_true")
parser.add_argument("--no_plots", action="store_true", default=False, help="Only export CSV/JSON files.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.checkpoint is None:
    parser.error("--checkpoint is required.")
if args_cli.envs_per_case < 1 or args_cli.eval_s <= 0.0 or args_cli.warmup_s < 0.0:
    parser.error("envs_per_case must be positive and evaluation durations must be valid.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab.envs.mdp import UniformVelocityCommandCfg

from unitree_rl_lab.tasks.locomotion.robots.go2.adaptive_energy_lpacrl_terrain_cfg import (
    LPACRL_COLUMNS_PER_TYPE,
    LPACRL_TERRAIN_NAMES,
    LPACRL_TERRAINS_CFG,
)
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def _as_torch(value):
    return getattr(value, "torch", value)


def _terrain_columns() -> dict[str, list[int]]:
    """Return the four deterministic geometry columns for every terrain family."""
    return {
        name: list(range(index * LPACRL_COLUMNS_PER_TYPE, (index + 1) * LPACRL_COLUMNS_PER_TYPE))
        for index, name in enumerate(LPACRL_TERRAIN_NAMES)
    }


def _install_fixed_commands(env, commands: torch.Tensor) -> None:
    term = env.unwrapped.command_manager.get_term("base_velocity")
    commands = commands.to(term.device)

    def _resample_fixed(self, env_ids):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.vel_command_b[env_ids] = commands[env_ids]
        self.is_standing_env[env_ids] = False
        if hasattr(self, "bin_ids"):
            self.bin_ids[env_ids] = -1
        for name in (
            "_velocity_abs_error_sum",
            "_segment_steps",
        ):
            if hasattr(self, name):
                getattr(self, name)[env_ids] = 0

    term._resample_command = MethodType(_resample_fixed, term)


def _load_actor(env, agent_cfg, checkpoint: str):
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(
        checkpoint,
        load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False},
        map_location=agent_cfg.device,
    )
    return runner.get_inference_policy(device=env.unwrapped.device)


def _plot_results(summary, terrain_names, levels, speeds, output_dir: Path, cases=None, dt: float = 0.02, vx_history=None) -> None:
    """Render terrain/level/speed grids as heatmaps and per-family curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    lookup = {(row["terrain_type"], row["terrain_level"], row["command_vx"]): row for row in summary}
    levels = sorted(set(levels))
    speeds = sorted(set(speeds))
    row_keys = [(name, level) for name in terrain_names for level in levels]
    row_labels = [f"{name} L{level}" for name, level in row_keys]

    def matrix(metric: str):
        data = np.full((len(row_keys), len(speeds)), np.nan)
        for row_index, key in enumerate(row_keys):
            for column_index, speed in enumerate(speeds):
                row = lookup.get((key[0], key[1], speed))
                if row is not None:
                    data[row_index, column_index] = row[metric]
        return np.ma.masked_invalid(data)

    metrics = (
        ("mae_vx", "Forward-speed MAE [m/s]", "YlOrRd", None, None),
        ("survival_rate", "Survival rate", "RdYlGn", 0.0, 1.0),
        ("mechanical_cot", "Mechanical CoT", "YlOrRd", None, None),
    )
    figure, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(1.8 * len(speeds) + 5.5, 0.42 * len(row_keys) + 3.4),
        constrained_layout=True,
        sharex=True,
    )
    for axis, (metric, title, cmap, vmin, vmax) in zip(axes, metrics):
        data = matrix(metric)
        image = axis.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_yticks(range(len(row_labels)), row_labels, fontsize=8)
        axis.set_title(title)
        for row_index in range(len(row_keys)):
            for column_index in range(len(speeds)):
                value = data[row_index, column_index]
                if not np.ma.is_masked(value):
                    axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=7)
        figure.colorbar(image, ax=axis, fraction=0.03, pad=0.01)
    axes[0].set_xticks(range(len(speeds)), [f"{speed:g}" for speed in speeds])
    axes[-1].set_xlabel("Commanded forward speed [m/s]")
    figure.savefig(output_dir / "terrain_tracking_heatmaps.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, max(1, (len(terrain_names) + 1) // 2), figsize=(14, 8), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for axis, name in zip(axes, terrain_names):
        for level in levels:
            values = [lookup.get((name, level, speed), {}).get("mae_vx", float("nan")) for speed in speeds]
            axis.plot(speeds, values, marker="o", label=f"level {level}")
        axis.axhline(0.15, color="gray", linestyle="--", linewidth=1)
        axis.set_title(name)
        axis.set_xlabel("command speed [m/s]")
        axis.set_ylabel("MAE vx [m/s]")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
    for axis in axes[len(terrain_names):]:
        axis.set_visible(False)
    figure.savefig(output_dir / "terrain_tracking_curves.png", dpi=180)
    plt.close(figure)

    # Measured versus commanded forward speed per family and level: points on
    # the dashed identity line track perfectly; a sagging curve means the
    # terrain caps the achievable speed.
    figure, axes = plt.subplots(2, max(1, (len(terrain_names) + 1) // 2), figsize=(14, 9), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for axis, name in zip(axes, terrain_names):
        for level in levels:
            measured = [lookup.get((name, level, speed), {}).get("mean_vx", float("nan")) for speed in speeds]
            axis.plot(speeds, measured, marker="s", label=f"level {level}")
        axis.plot([0.0, max(speeds)], [0.0, max(speeds)], color="gray", linestyle="--", linewidth=1, label="ideal")
        axis.set_title(name)
        axis.set_xlabel("commanded vx [m/s]")
        axis.set_ylabel("measured vx [m/s]")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
    for axis in axes[len(terrain_names):]:
        axis.set_visible(False)
    figure.savefig(output_dir / "terrain_tracking_calibration.png", dpi=180)
    plt.close(figure)

    # Time series of measured vx against the commanded level, one subplot per
    # family at the middle terrain level, replicate 0.
    if vx_history and cases is not None:
        history = np.stack(vx_history)
        time_s = np.arange(history.shape[0]) * dt
        case_index = {case: index for index, case in enumerate(cases)}
        middle_level = levels[len(levels) // 2]
        figure, axes = plt.subplots(
            2, max(1, (len(terrain_names) + 1) // 2), figsize=(14, 9), constrained_layout=True, sharex=True, sharey=True
        )
        axes = np.atleast_1d(axes).ravel()
        for axis, name in zip(axes, terrain_names):
            for speed in speeds:
                index = case_index[(name, middle_level, speed, 0)]
                axis.plot(time_s, history[:, index], linewidth=1.0, label=f"cmd {speed:g}")
                axis.axhline(speed, color="gray", linestyle=":", linewidth=0.8)
            axis.set_title(f"{name} (level {middle_level}; dotted = command)")
            axis.set_xlabel("time [s]")
            axis.set_ylabel("measured vx [m/s]")
            axis.grid(alpha=0.3)
            axis.legend(fontsize=7)
        for axis in axes[len(terrain_names):]:
            axis.set_visible(False)
        figure.savefig(output_dir / "terrain_tracking_timeseries.png", dpi=180)
        plt.close(figure)


def main() -> None:
    column_map = _terrain_columns()
    terrain_names = args_cli.terrain_types or list(column_map)
    unknown = sorted(set(terrain_names) - set(column_map))
    if unknown:
        raise ValueError(f"Unknown terrain types {unknown}; choose from {list(column_map)}")
    if any(not column_map[name] for name in terrain_names):
        raise ValueError("At least one selected terrain has no generated column.")
    if min(args_cli.terrain_levels) < 0 or max(args_cli.terrain_levels) >= LPACRL_TERRAINS_CFG.num_rows:
        raise ValueError(f"Terrain levels must be in [0, {LPACRL_TERRAINS_CFG.num_rows - 1}].")

    cases = [
        (name, level, speed, replicate)
        for name in terrain_names
        for level in args_cli.terrain_levels
        for speed in args_cli.speeds
        for replicate in range(args_cli.envs_per_case)
    ]
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
    env_cfg.episode_length_s = args_cli.warmup_s + args_cli.eval_s + 5.0
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
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.physics_material = None
    env_cfg.events.add_base_mass = None
    env_cfg.events.push_robot = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.seed = args_cli.seed

    # Reuse the task's runner config while bypassing resume semantics.
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    commands = torch.zeros(num_envs, 3, device=args_cli.device)
    levels = torch.tensor([case[1] for case in cases], dtype=torch.long, device=args_cli.device)
    columns = torch.tensor(
        [column_map[case[0]][case[3] % len(column_map[case[0]])] for case in cases],
        dtype=torch.long,
        device=args_cli.device,
    )
    commands[:, 0] = torch.tensor([case[2] for case in cases], device=args_cli.device)

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    terrain = gym_env.unwrapped.scene.terrain
    terrain.terrain_levels[:] = levels.to(terrain.terrain_levels.device)
    terrain.terrain_types[:] = columns.to(terrain.terrain_types.device)
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    _install_fixed_commands(gym_env, commands)
    gym_env.reset()

    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    policy = _load_actor(env, agent_cfg, checkpoint)
    robot = env.unwrapped.scene["robot"]
    device = env.unwrapped.device
    commands = commands.to(device)
    dt = float(env.unwrapped.step_dt)
    warmup_steps = round(args_cli.warmup_s / dt)
    eval_steps = round(args_cli.eval_s / dt)

    alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    count = torch.zeros(num_envs, device=device)
    abs_error = torch.zeros(num_envs, 3, device=device)
    square_error = torch.zeros(num_envs, 3, device=device)
    velocity_sum = torch.zeros(num_envs, device=device)
    power_sum = torch.zeros(num_envs, device=device)
    energy = torch.zeros(num_envs, device=device)
    distance = torch.zeros(num_envs, device=device)
    vx_history = [] if not args_cli.no_plots else None

    observation = env.get_observations()
    observation = observation[0] if isinstance(observation, tuple) else observation
    for step in range(warmup_steps + eval_steps):
        with torch.inference_mode():
            observation, _, dones, _ = env.step(policy(observation))
        valid = alive & ~dones.bool()
        alive &= ~dones.bool()
        if step < warmup_steps:
            continue
        lin_vel = _as_torch(robot.data.root_lin_vel_b)
        ang_vel = _as_torch(robot.data.root_ang_vel_b)
        velocity = torch.stack((lin_vel[:, 0], lin_vel[:, 1], ang_vel[:, 2]), dim=1)
        error = velocity - commands
        joint_vel = _as_torch(robot.data.joint_vel)
        torque = _as_torch(robot.data.applied_torque)
        power = torch.sum(torch.abs(joint_vel * torque), dim=1)
        count += valid.float()
        abs_error += torch.where(valid.unsqueeze(1), torch.abs(error), 0.0)
        square_error += torch.where(valid.unsqueeze(1), torch.square(error), 0.0)
        velocity_sum += torch.where(valid, velocity[:, 0], 0.0)
        if vx_history is not None:
            vx_history.append(velocity[:, 0].detach().cpu().clone())
        power_sum += torch.where(valid, power, 0.0)
        energy += torch.where(valid, power * dt, 0.0)
        distance += torch.where(valid, torch.abs(lin_vel[:, 0]) * dt, 0.0)

    denom = count.clamp_min(1.0)
    mae = abs_error / denom.unsqueeze(1)
    rmse = torch.sqrt(square_error / denom.unsqueeze(1))
    mean_vx = velocity_sum / denom
    mean_power = power_sum / denom
    mass = _as_torch(robot.data.body_mass).sum(dim=1)
    cot = torch.where(distance > 1.0e-3, energy / (mass * 9.81 * distance), torch.nan)

    rows = []
    for index, case in enumerate(cases):
        rows.append(
            {
                "terrain_type": case[0],
                "terrain_level": case[1],
                "command_vx": case[2],
                "replicate": case[3],
                "terrain_column": int(columns[index]),
                "survived": bool(alive[index]),
                "samples": int(count[index]),
                "mean_vx": float(mean_vx[index]),
                "bias_vx": float(mean_vx[index] - commands[index, 0]),
                "mae_vx": float(mae[index, 0]),
                "rmse_vx": float(rmse[index, 0]),
                "mae_vy": float(mae[index, 1]),
                "mae_yaw": float(mae[index, 2]),
                "mean_power_w": float(mean_power[index]),
                "mechanical_cot": float(cot[index]),
            }
        )

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["terrain_type"], row["terrain_level"], row["command_vx"])].append(row)
    summary = []
    for key, group in grouped.items():
        summary.append(
            {
                "terrain_type": key[0],
                "terrain_level": key[1],
                "command_vx": key[2],
                "survival_rate": sum(row["survived"] for row in group) / len(group),
                "mean_vx": sum(row["mean_vx"] for row in group) / len(group),
                "bias_vx": sum(row["bias_vx"] for row in group) / len(group),
                "mae_vx": sum(row["mae_vx"] for row in group) / len(group),
                "rmse_vx": sum(row["rmse_vx"] for row in group) / len(group),
                "mean_power_w": sum(row["mean_power_w"] for row in group) / len(group),
                "mechanical_cot": sum(row["mechanical_cot"] for row in group) / len(group),
            }
        )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = (
        Path(args_cli.output_dir).expanduser().resolve()
        if args_cli.output_dir
        else Path(checkpoint).parent / "evaluations" / f"terrain_tracking_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in (("per_environment.csv", rows), ("terrain_speed_summary.csv", summary)):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(vars(args_cli) | {"checkpoint": checkpoint, "num_envs": num_envs}, stream, indent=2)

    if not args_cli.no_plots:
        _plot_results(
            summary,
            terrain_names,
            args_cli.terrain_levels,
            args_cli.speeds,
            output_dir,
            cases=cases,
            dt=dt,
            vx_history=vx_history,
        )
        print(f"[INFO] Plots: {output_dir / 'terrain_tracking_heatmaps.png'} (plus curves/calibration/timeseries)")

    print(f"[INFO] Evaluated {len(grouped)} terrain/level/speed cases in {num_envs} environments.")
    print(f"[INFO] Results: {output_dir}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
