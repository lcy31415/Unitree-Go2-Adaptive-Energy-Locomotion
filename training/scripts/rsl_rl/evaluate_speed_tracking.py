# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batch evaluation of steady-state velocity tracking for an RSL-RL checkpoint.

The script assigns one fixed forward command to each group of environments,
runs all command groups in one simulator instance, and exports per-rollout,
per-speed, and time-series metrics. Commands and measured velocities are all in
the robot body frame.
"""

from __future__ import annotations

import argparse
import sys
from importlib.metadata import version
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "source" / "unitree_rl_lab"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate steady-state forward-speed tracking.")
parser.add_argument("--task", type=str, default="Unitree-Go2-Adaptive-Energy-Flat", help="Gym task name.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable Fabric and use USD I/O operations."
)
parser.add_argument(
    "--speeds",
    type=float,
    nargs="+",
    default=None,
    help="Explicit forward commands in m/s. Overrides --speed_min/--speed_max/--speed_step.",
)
parser.add_argument("--speed_min", type=float, default=0.0, help="Minimum generated forward command in m/s.")
parser.add_argument("--speed_max", type=float, default=3.0, help="Maximum generated forward command in m/s.")
parser.add_argument("--speed_step", type=float, default=0.25, help="Generated command spacing in m/s.")
parser.add_argument("--envs_per_speed", type=int, default=16, help="Independent rollouts allocated per speed.")
parser.add_argument("--warmup_s", type=float, default=5.0, help="Unmeasured settling time in seconds.")
parser.add_argument("--eval_s", type=float, default=15.0, help="Measured duration in seconds.")
parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory; defaults beside checkpoint.")
parser.add_argument(
    "--robust_eval",
    action="store_true",
    default=False,
    help="Keep startup/reset domain randomization and policy observation noise.",
)
parser.add_argument("--tracking_fraction", type=float, default=0.8, help="Required in-tolerance fraction.")
parser.add_argument("--linear_abs_tol", type=float, default=0.15, help="Absolute vx/vy tolerance floor in m/s.")
parser.add_argument("--linear_rel_tol", type=float, default=0.1, help="Relative vx/vy tolerance.")
parser.add_argument("--yaw_abs_tol", type=float, default=0.2, help="Absolute yaw-rate tolerance floor in rad/s.")
parser.add_argument("--yaw_rel_tol", type=float, default=0.1, help="Relative yaw-rate tolerance.")
parser.add_argument("--no_plots", action="store_true", default=False, help="Only export CSV/JSON files.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.checkpoint is None:
    parser.error("--checkpoint is required.")
if args_cli.envs_per_speed <= 0:
    parser.error("--envs_per_speed must be positive.")
if args_cli.warmup_s < 0.0 or args_cli.eval_s <= 0.0:
    parser.error("--warmup_s must be non-negative and --eval_s must be positive.")
if args_cli.speeds is None and args_cli.speed_step <= 0.0:
    parser.error("--speed_step must be positive.")
if args_cli.speeds is None and args_cli.speed_max < args_cli.speed_min:
    parser.error("--speed_max must be greater than or equal to --speed_min.")
if not 0.0 <= args_cli.tracking_fraction <= 1.0:
    parser.error("--tracking_fraction must be in [0, 1].")
if min(args_cli.linear_abs_tol, args_cli.linear_rel_tol, args_cli.yaw_abs_tol, args_cli.yaw_rel_tol) < 0.0:
    parser.error("Tracking tolerances must be non-negative.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import math
import os
from datetime import datetime
from types import MethodType

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def _make_speed_list() -> list[float]:
    if args_cli.speeds is not None:
        speeds = args_cli.speeds
    else:
        count = int(math.floor((args_cli.speed_max - args_cli.speed_min) / args_cli.speed_step + 1.0e-9)) + 1
        speeds = [args_cli.speed_min + index * args_cli.speed_step for index in range(count)]
        if speeds[-1] < args_cli.speed_max - 1.0e-8:
            speeds.append(args_cli.speed_max)
    if not speeds:
        raise ValueError("At least one speed must be specified.")
    return [round(float(speed), 8) for speed in speeds]


def _as_torch(value) -> torch.Tensor:
    return getattr(value, "torch", value)


def _configure_nominal_evaluation(env_cfg) -> None:
    """Disable evaluation-time randomness without changing robot dynamics."""
    env_cfg.observations.policy.enable_corruption = False
    for event_name in ("physics_material", "add_base_mass", "push_robot", "base_external_force_torque"):
        if hasattr(env_cfg.events, event_name):
            setattr(env_cfg.events, event_name, None)

    if getattr(env_cfg.events, "reset_base", None) is not None:
        env_cfg.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        env_cfg.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
    if getattr(env_cfg.events, "reset_robot_joints", None) is not None:
        env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


def _install_per_environment_commands(env, fixed_commands: torch.Tensor) -> None:
    """Keep a different fixed body-frame velocity command in every environment."""
    command_term = env.unwrapped.command_manager.get_term("base_velocity")
    fixed_commands = fixed_commands.to(device=command_term.device, dtype=torch.float)

    def _resample_fixed(self, env_ids):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.vel_command_b[env_ids] = fixed_commands[env_ids]
        self.is_standing_env[env_ids] = False
        if hasattr(self, "bin_ids"):
            self.bin_ids[env_ids] = -1
        if hasattr(self, "_velocity_abs_error_sum"):
            self._velocity_abs_error_sum[env_ids] = 0.0
        if hasattr(self, "_segment_steps"):
            self._segment_steps[env_ids] = 0

    command_term._resample_command = MethodType(_resample_fixed, command_term)
    command_term._resample_command(torch.arange(env.unwrapped.num_envs, device=command_term.device))


def _masked_group_mean(values: torch.Tensor, valid: torch.Tensor, group_size: int) -> torch.Tensor:
    values = values.reshape(-1, group_size)
    valid = valid.reshape(-1, group_size)
    counts = valid.sum(dim=1)
    sums = torch.where(valid, values, torch.zeros_like(values)).sum(dim=1)
    nan = torch.full_like(sums, torch.nan)
    return torch.where(counts > 0, sums / counts.clamp_min(1), nan)


def _load_policy(env, agent_cfg: RslRlOnPolicyRunnerCfg, checkpoint: str):
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(checkpoint)
    return runner.get_inference_policy(device=env.unwrapped.device)


def _safe_nanmean(values: pd.Series) -> float:
    array = values.to_numpy(dtype=float)
    return float(np.nanmean(array)) if np.isfinite(array).any() else float("nan")


def _ci95(values: pd.Series) -> float:
    array = values.to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return float("nan")
    return float(1.96 * np.std(array, ddof=1) / np.sqrt(array.size))


def _summarize(per_env: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for command_vx, group in per_env.groupby("command_vx", sort=True):
        rows.append(
            {
                "command_vx": command_vx,
                "num_rollouts": len(group),
                "valid_rollouts": int(group["mean_vx"].notna().sum()),
                "mean_vx": _safe_nanmean(group["mean_vx"]),
                "mean_vx_ci95": _ci95(group["mean_vx"]),
                "bias_vx": _safe_nanmean(group["bias_vx"]),
                "mae_vx": _safe_nanmean(group["mae_vx"]),
                "rmse_vx": _safe_nanmean(group["rmse_vx"]),
                "std_vx": _safe_nanmean(group["std_vx"]),
                "completion": _safe_nanmean(group["completion"]),
                "tracking_fraction": _safe_nanmean(group["tracking_fraction"]),
                "mean_abs_vy": _safe_nanmean(group["mean_abs_vy"]),
                "mean_abs_yaw": _safe_nanmean(group["mean_abs_yaw"]),
                "mean_power_w": _safe_nanmean(group["mean_power_w"]),
                "mechanical_cot": _safe_nanmean(group["mechanical_cot"]),
                "success_rate": float(group["tracking_success"].mean()),
                "survival_rate": float(group["survived"].mean()),
                "fall_rate": float(group["fell"].mean()),
                "timeout_rate": float(group["timed_out"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _save_plots(summary: pd.DataFrame, time_series: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = summary["command_vx"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    axes[0, 0].errorbar(
        x,
        summary["mean_vx"],
        yerr=summary["mean_vx_ci95"],
        marker="o",
        capsize=3,
        label="policy",
    )
    axes[0, 0].plot(x, x, "--", color="black", label="ideal y=x")
    axes[0, 0].set(title="Forward-speed tracking", xlabel="Command vx [m/s]", ylabel="Mean vx [m/s]")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(x, summary["mae_vx"], marker="o", label="MAE")
    axes[0, 1].plot(x, summary["rmse_vx"], marker="s", label="RMSE")
    axes[0, 1].set(title="Tracking error", xlabel="Command vx [m/s]", ylabel="Error [m/s]")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(x, summary["tracking_fraction"], marker="o", label="in-tolerance fraction")
    axes[1, 0].plot(x, summary["success_rate"], marker="s", label="rollout success")
    axes[1, 0].plot(x, summary["survival_rate"], marker="^", label="survival")
    axes[1, 0].set(
        title="Tracking and stability",
        xlabel="Command vx [m/s]",
        ylabel="Fraction",
        ylim=(-0.02, 1.02),
    )
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    energy_axis = axes[1, 1]
    cot_axis = energy_axis.twinx()
    energy_axis.plot(x, summary["mean_power_w"], marker="o", color="tab:red", label="power")
    cot_axis.plot(x, summary["mechanical_cot"], marker="s", color="tab:blue", label="CoT")
    energy_axis.set(title="Energy efficiency", xlabel="Command vx [m/s]", ylabel="Power proxy [W]")
    cot_axis.set_ylabel("Mechanical CoT")
    energy_axis.grid(True, alpha=0.3)
    lines = energy_axis.lines + cot_axis.lines
    energy_axis.legend(lines, [line.get_label() for line in lines])

    fig.savefig(output_dir / "speed_tracking_summary.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    for command_vx, group in time_series.groupby("command_vx", sort=True):
        axis.plot(group["time_s"], group["mean_vx"], label=f"cmd {command_vx:g}")
    axis.set(title="Mean forward velocity time series", xlabel="Evaluation time [s]", ylabel="Mean vx [m/s]")
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2, fontsize=8)
    fig.savefig(output_dir / "speed_tracking_time_series.png", dpi=180)
    plt.close(fig)


def main() -> None:
    speeds = _make_speed_list()
    num_speeds = len(speeds)
    num_envs = num_speeds * args_cli.envs_per_speed

    checkpoint = os.path.abspath(retrieve_file_path(args_cli.checkpoint))
    if args_cli.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path(checkpoint).parent / "evaluations" / f"speed_tracking_{timestamp}"
    else:
        output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.scene.num_envs = num_envs
    env_cfg.episode_length_s = args_cli.warmup_s + args_cli.eval_s + 5.0
    env_cfg.commands.base_velocity.debug_vis = False
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = args_cli.seed
    if not args_cli.robust_eval:
        _configure_nominal_evaluation(env_cfg)

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))

    fixed_commands = torch.zeros(num_envs, 3, device=args_cli.device)
    for speed_index, speed in enumerate(speeds):
        begin = speed_index * args_cli.envs_per_speed
        fixed_commands[begin : begin + args_cli.envs_per_speed, 0] = speed

    print(f"[INFO] Evaluating checkpoint: {checkpoint}")
    print(f"[INFO] Commands: {speeds}")
    print(f"[INFO] Environments: {num_envs} ({args_cli.envs_per_speed} per speed)")
    print(f"[INFO] Mode: {'robust' if args_cli.robust_eval else 'nominal'}")

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    _install_per_environment_commands(gym_env, fixed_commands)
    if isinstance(gym_env.unwrapped, DirectMARLEnv):
        gym_env = multi_agent_to_single_agent(gym_env)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    policy = _load_policy(env, agent_cfg, checkpoint)

    dt = float(env.unwrapped.step_dt)
    warmup_steps = int(round(args_cli.warmup_s / dt))
    eval_steps = int(round(args_cli.eval_s / dt))
    total_steps = warmup_steps + eval_steps
    device = env.unwrapped.device
    commands = fixed_commands.to(device)
    robot = env.unwrapped.scene["robot"]

    count = torch.zeros(num_envs, device=device)
    sum_vx = torch.zeros(num_envs, device=device)
    sum_vx_sq = torch.zeros(num_envs, device=device)
    sum_bias = torch.zeros(num_envs, device=device)
    sum_abs_error = torch.zeros(num_envs, device=device)
    sum_sq_error = torch.zeros(num_envs, device=device)
    sum_abs_vy = torch.zeros(num_envs, device=device)
    sum_abs_yaw = torch.zeros(num_envs, device=device)
    sum_power = torch.zeros(num_envs, device=device)
    energy_j = torch.zeros(num_envs, device=device)
    distance_m = torch.zeros(num_envs, device=device)
    within_count = torch.zeros(num_envs, device=device)
    alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    fell = torch.zeros_like(alive)
    timed_out = torch.zeros_like(alive)
    failure_time = torch.full((num_envs,), torch.nan, device=device)

    trace = torch.full((eval_steps, num_speeds, 6), torch.nan, device=device)

    obs_result = env.get_observations()
    obs = obs_result[0] if isinstance(obs_result, tuple) else obs_result

    for step in range(total_steps):
        if not simulation_app.is_running():
            raise RuntimeError("Simulation application closed before evaluation completed.")
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)

        dones = dones.bool()
        time_outs_now = extras.get("time_outs")
        if time_outs_now is None:
            time_outs_now = torch.zeros_like(dones)
        else:
            time_outs_now = time_outs_now.bool()
        newly_done = torch.logical_and(alive, dones)
        newly_fallen = torch.logical_and(newly_done, torch.logical_not(time_outs_now))
        newly_timed_out = torch.logical_and(newly_done, time_outs_now)
        failure_time[newly_done] = (step + 1) * dt
        fell |= newly_fallen
        timed_out |= newly_timed_out

        # Isaac Lab resets done environments inside env.step(), so their root
        # state is already the reset state here and must not enter statistics.
        valid = torch.logical_and(alive, torch.logical_not(dones))
        alive &= torch.logical_not(dones)

        lin_vel = _as_torch(robot.data.root_lin_vel_b)
        ang_vel = _as_torch(robot.data.root_ang_vel_b)
        joint_vel = _as_torch(robot.data.joint_vel)
        torque = _as_torch(robot.data.applied_torque)
        vx = lin_vel[:, 0]
        vy = lin_vel[:, 1]
        yaw = ang_vel[:, 2]
        power = torch.sum(torch.abs(joint_vel) * torch.abs(torque), dim=1)

        if step >= warmup_steps:
            eval_index = step - warmup_steps
            error_x = vx - commands[:, 0]
            tolerance_x = torch.maximum(
                torch.full_like(error_x, args_cli.linear_abs_tol),
                args_cli.linear_rel_tol * torch.abs(commands[:, 0]),
            )
            tolerance_y = torch.maximum(
                torch.full_like(vy, args_cli.linear_abs_tol),
                args_cli.linear_rel_tol * torch.abs(commands[:, 1]),
            )
            tolerance_yaw = torch.maximum(
                torch.full_like(yaw, args_cli.yaw_abs_tol),
                args_cli.yaw_rel_tol * torch.abs(commands[:, 2]),
            )
            within = (
                (torch.abs(error_x) <= tolerance_x)
                & (torch.abs(vy) <= tolerance_y)
                & (torch.abs(yaw) <= tolerance_yaw)
                & valid
            )
            valid_float = valid.float()
            count += valid_float
            sum_vx += torch.where(valid, vx, 0.0)
            sum_vx_sq += torch.where(valid, torch.square(vx), 0.0)
            sum_bias += torch.where(valid, error_x, 0.0)
            sum_abs_error += torch.where(valid, torch.abs(error_x), 0.0)
            sum_sq_error += torch.where(valid, torch.square(error_x), 0.0)
            sum_abs_vy += torch.where(valid, torch.abs(vy), 0.0)
            sum_abs_yaw += torch.where(valid, torch.abs(yaw), 0.0)
            sum_power += torch.where(valid, power, 0.0)
            energy_j += torch.where(valid, power * dt, 0.0)
            distance_m += torch.where(valid, torch.abs(vx) * dt, 0.0)
            within_count += within.float()

            trace[eval_index, :, 0] = _masked_group_mean(vx, valid, args_cli.envs_per_speed)
            trace[eval_index, :, 1] = _masked_group_mean(vy, valid, args_cli.envs_per_speed)
            trace[eval_index, :, 2] = _masked_group_mean(yaw, valid, args_cli.envs_per_speed)
            trace[eval_index, :, 3] = _masked_group_mean(power, valid, args_cli.envs_per_speed)
            trace[eval_index, :, 4] = valid.reshape(num_speeds, args_cli.envs_per_speed).float().mean(dim=1)
            trace[eval_index, :, 5] = within.reshape(num_speeds, args_cli.envs_per_speed).float().mean(dim=1)

    valid_samples = count > 0
    denominator = count.clamp_min(1.0)
    nan = torch.full_like(count, torch.nan)
    mean_vx = torch.where(valid_samples, sum_vx / denominator, nan)
    mean_vx_sq = torch.where(valid_samples, sum_vx_sq / denominator, nan)
    bias = torch.where(valid_samples, sum_bias / denominator, nan)
    mae = torch.where(valid_samples, sum_abs_error / denominator, nan)
    rmse = torch.where(valid_samples, torch.sqrt(sum_sq_error / denominator), nan)
    std_vx = torch.where(valid_samples, torch.sqrt(torch.clamp(mean_vx_sq - torch.square(mean_vx), min=0.0)), nan)
    mean_abs_vy = torch.where(valid_samples, sum_abs_vy / denominator, nan)
    mean_abs_yaw = torch.where(valid_samples, sum_abs_yaw / denominator, nan)
    mean_power = torch.where(valid_samples, sum_power / denominator, nan)
    tracking_fraction = torch.where(valid_samples, within_count / denominator, nan)
    completion = torch.where(torch.abs(commands[:, 0]) > 0.2, mean_vx / commands[:, 0], nan)

    body_mass = _as_torch(robot.data.body_mass).sum(dim=1)
    mechanical_cot = torch.where(
        distance_m > 1.0e-3,
        energy_j / (body_mass * 9.81 * distance_m),
        nan,
    )
    vx_tolerance = torch.maximum(
        torch.full_like(commands[:, 0], args_cli.linear_abs_tol),
        args_cli.linear_rel_tol * torch.abs(commands[:, 0]),
    )
    tracking_success = (
        alive
        & valid_samples
        & (mae <= vx_tolerance)
        & (mean_abs_vy <= args_cli.linear_abs_tol)
        & (mean_abs_yaw <= args_cli.yaw_abs_tol)
        & (tracking_fraction >= args_cli.tracking_fraction)
    )

    result_tensors = torch.stack(
        (
            count,
            mean_vx,
            bias,
            mae,
            rmse,
            std_vx,
            completion,
            tracking_fraction,
            mean_abs_vy,
            mean_abs_yaw,
            mean_power,
            energy_j,
            distance_m,
            mechanical_cot,
            failure_time,
        ),
        dim=1,
    ).cpu().numpy()

    rows = []
    for env_index in range(num_envs):
        speed_index = env_index // args_cli.envs_per_speed
        values = result_tensors[env_index]
        rows.append(
            {
                "speed_index": speed_index,
                "env_index": env_index,
                "replicate": env_index % args_cli.envs_per_speed,
                "command_vx": speeds[speed_index],
                "samples": int(values[0]),
                "mean_vx": values[1],
                "bias_vx": values[2],
                "mae_vx": values[3],
                "rmse_vx": values[4],
                "std_vx": values[5],
                "completion": values[6],
                "tracking_fraction": values[7],
                "mean_abs_vy": values[8],
                "mean_abs_yaw": values[9],
                "mean_power_w": values[10],
                "energy_j": values[11],
                "distance_m": values[12],
                "mechanical_cot": values[13],
                "failure_time_s": values[14],
                "survived": bool(alive[env_index].item()),
                "fell": bool(fell[env_index].item()),
                "timed_out": bool(timed_out[env_index].item()),
                "tracking_success": bool(tracking_success[env_index].item()),
            }
        )
    per_env = pd.DataFrame(rows)
    summary = _summarize(per_env)

    trace_np = trace.cpu().numpy()
    trace_rows = []
    for eval_index in range(eval_steps):
        for speed_index, speed in enumerate(speeds):
            values = trace_np[eval_index, speed_index]
            trace_rows.append(
                {
                    "time_s": (eval_index + 1) * dt,
                    "command_vx": speed,
                    "mean_vx": values[0],
                    "mean_vy": values[1],
                    "mean_yaw": values[2],
                    "mean_power_w": values[3],
                    "alive_fraction": values[4],
                    "within_tolerance_fraction": values[5],
                }
            )
    time_series = pd.DataFrame(trace_rows)

    per_env.to_csv(output_dir / "per_environment.csv", index=False)
    summary.to_csv(output_dir / "speed_summary.csv", index=False)
    time_series.to_csv(output_dir / "time_series.csv", index=False)

    metadata = {
        "task": args_cli.task,
        "checkpoint": checkpoint,
        "mode": "robust" if args_cli.robust_eval else "nominal",
        "speeds_mps": speeds,
        "envs_per_speed": args_cli.envs_per_speed,
        "num_envs": num_envs,
        "seed": args_cli.seed,
        "step_dt_s": dt,
        "warmup_s": args_cli.warmup_s,
        "eval_s": args_cli.eval_s,
        "linear_abs_tolerance_mps": args_cli.linear_abs_tol,
        "linear_relative_tolerance": args_cli.linear_rel_tol,
        "yaw_abs_tolerance_radps": args_cli.yaw_abs_tol,
        "yaw_relative_tolerance": args_cli.yaw_rel_tol,
        "required_tracking_fraction": args_cli.tracking_fraction,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    if not args_cli.no_plots:
        _save_plots(summary, time_series, output_dir)

    print("\n[INFO] Speed tracking summary:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\n[INFO] Evaluation results saved to: {output_dir}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
