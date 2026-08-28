"""Evaluate an adaptive-energy policy over terrain type, level and speed."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from types import MethodType

_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "source" / "unitree_rl_lab"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


_DEFAULT_TASK = "Unitree-Go2-Adaptive-Energy-Terrain-LPACRL"
_PIE_TASK = "Unitree-Go2-Adaptive-Energy-LPACRL-PIE"
_PIE_STAIRS_TASK = "Unitree-Go2-PIE-Stairs"
_PIE_STAIRS_LADDER_TASK = "Unitree-Go2-Adaptive-Energy-stairs-PIE"
_PIE_FLAT_TASK = "Unitree-Go2-Adaptive-Energy-Flat-LPACRL-PIE"
_PIE_MULTI_TERRAIN_TASK = "Unitree-Go2-Adaptive-Energy-PIE"
_PIE_TASKS = {
    _PIE_TASK,
    _PIE_STAIRS_TASK,
    _PIE_STAIRS_LADDER_TASK,
    _PIE_FLAT_TASK,
    _PIE_MULTI_TERRAIN_TASK,
}
_STAIRS_FAMILY = (_PIE_STAIRS_TASK, _PIE_STAIRS_LADDER_TASK)


parser = argparse.ArgumentParser(description="Evaluate terrain-conditioned speed tracking.")
parser.add_argument(
    "--mode",
    choices=("terrain", "speed"),
    default="terrain",
    help=(
        "terrain: terrain x level x speed success matrix (default). "
        "speed: flat-ground speed sweep with tracking and energy statistics."
    ),
)
parser.add_argument(
    "--task",
    default=None,
    help="Gym task ID. If omitted, infer the PIE task from the checkpoint path and otherwise use Terrain-LPACRL.",
)
parser.add_argument("--checkpoint", default=None)
parser.add_argument(
    "--speeds",
    type=float,
    nargs="+",
    default=None,
    help="Explicit speed list. Terrain mode defaults to [0.5, 1.0, 1.5, 2.0, 2.5]; speed mode generates from --speed_min/max/step when omitted.",
)
parser.add_argument("--terrain_levels", type=int, nargs="+", default=[0, 1, 2, 3])
parser.add_argument("--terrain_types", nargs="+", default=None, help="Names from adaptive_energy_terrain_cfg.py")
parser.add_argument("--envs_per_case", type=int, default=4)
parser.add_argument("--warmup_s", type=float, default=2.0)
parser.add_argument("--eval_s", type=float, default=10.0)
parser.add_argument("--output_dir", default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_lateral_deviation", type=float, default=1.0, help="Maximum course-center error [m].")
parser.add_argument("--max_heading_error_deg", type=float, default=20.0, help="Maximum absolute heading error [deg].")
parser.add_argument(
    "--rough_runup_distance",
    type=float,
    default=0.75,
    help="Unmeasured +x acceleration distance before the random-rough measurement segment [m].",
)
parser.add_argument(
    "--multi_robot_view",
    action="store_true",
    help="Open Kit and frame all visible robots with an overview camera.",
)
parser.add_argument("--real_time", action="store_true", help="Throttle simulation for real-time visualization.")
parser.add_argument(
    "--heading_control_gain",
    type=float,
    default=1.5,
    help="Proportional gain from endpoint heading error to commanded yaw rate.",
)
parser.add_argument(
    "--max_heading_command",
    type=float,
    default=1.0,
    help="Maximum absolute endpoint-tracking yaw-rate command [rad/s].",
)
parser.add_argument(
    "--disable_goal_direction_control",
    action="store_true",
    help="Disable endpoint-directed yaw control and retain a zero yaw-rate command.",
)
parser.add_argument("--disable_fabric", action="store_true")
parser.add_argument("--no_plots", action="store_true", default=False, help="Only export CSV/JSON files.")
parser.add_argument(
    "--zero_depth_observation",
    action="store_true",
    help="PIE ablation: replace the complete camera depth-history observation with zeros before every action.",
)
# --- speed-mode arguments (used with --mode speed) ---
parser.add_argument("--speed_min", type=float, default=0.0, help="Speed mode: minimum generated forward command in m/s.")
parser.add_argument("--speed_max", type=float, default=3.0, help="Speed mode: maximum generated forward command in m/s.")
parser.add_argument("--speed_step", type=float, default=0.25, help="Speed mode: generated command spacing in m/s.")
parser.add_argument("--envs_per_speed", type=int, default=16, help="Speed mode: rollouts per speed.")
parser.add_argument("--robust_eval", action="store_true", help="Speed mode: keep domain randomization instead of the nominal configuration.")
parser.add_argument("--tracking_fraction", type=float, default=0.8, help="Speed mode: required in-tolerance fraction.")
parser.add_argument("--linear_abs_tol", type=float, default=0.15, help="Speed mode: absolute vx/vy tolerance floor [m/s].")
parser.add_argument("--linear_rel_tol", type=float, default=0.1, help="Speed mode: relative vx/vy tolerance.")
parser.add_argument("--yaw_abs_tol", type=float, default=0.2, help="Speed mode: absolute yaw-rate tolerance floor [rad/s].")
parser.add_argument("--yaw_rel_tol", type=float, default=0.1, help="Speed mode: relative yaw-rate tolerance.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
_TASK_WAS_EXPLICIT = args_cli.task is not None
_SPEED_MODE = args_cli.mode == "speed"

if args_cli.checkpoint is None:
    parser.error("--checkpoint is required.")
if not _SPEED_MODE and (args_cli.envs_per_case < 1 or args_cli.eval_s <= 0.0 or args_cli.warmup_s < 0.0):
    parser.error("envs_per_case must be positive and evaluation durations must be valid.")
if _SPEED_MODE:
    if args_cli.envs_per_speed <= 0:
        parser.error("--envs_per_speed must be positive.")
    if args_cli.speeds is None and args_cli.speed_step <= 0.0:
        parser.error("--speed_step must be positive.")
    if args_cli.speeds is None and args_cli.speed_max < args_cli.speed_min:
        parser.error("--speed_max must be greater than or equal to --speed_min.")
    if not 0.0 <= args_cli.tracking_fraction <= 1.0:
        parser.error("--tracking_fraction must be in [0, 1].")
    if min(args_cli.linear_abs_tol, args_cli.linear_rel_tol, args_cli.yaw_abs_tol, args_cli.yaw_rel_tol) < 0.0:
        parser.error("Tracking tolerances must be non-negative.")
if args_cli.max_lateral_deviation <= 0.0 or not 0.0 < args_cli.max_heading_error_deg <= 180.0:
    parser.error("Course deviation limits must be positive and heading error must not exceed 180 degrees.")
if args_cli.rough_runup_distance < 0.0:
    parser.error("rough_runup_distance must be non-negative.")
if args_cli.heading_control_gain < 0.0 or args_cli.max_heading_command <= 0.0:
    parser.error("Heading-control gain must be non-negative and its command limit must be positive.")
if args_cli.task is None:
    checkpoint_hint = str(Path(args_cli.checkpoint).expanduser()).lower()
    if "pie_stairs" in checkpoint_hint or "pie-stairs" in checkpoint_hint:
        args_cli.task = _PIE_STAIRS_TASK
    elif "stairs_pie" in checkpoint_hint:
        args_cli.task = _PIE_STAIRS_LADDER_TASK
    elif "flat_lpacrl_pie" in checkpoint_hint:
        args_cli.task = _PIE_FLAT_TASK
    elif "adaptive_energy_pie" in checkpoint_hint:
        args_cli.task = _PIE_MULTI_TERRAIN_TASK
    elif "lpacrl_pie" in checkpoint_hint:
        args_cli.task = _PIE_TASK
    else:
        args_cli.task = _DEFAULT_TASK
    print(f"[INFO] Inferred task {args_cli.task!r} from checkpoint path.")
if args_cli.multi_robot_view:
    selected_visualizers = list(args_cli.visualizer or [])
    if "kit" not in selected_visualizers:
        selected_visualizers.append("kit")
    args_cli.visualizer = selected_visualizers
    if getattr(args_cli, "max_visible_envs", None) is None:
        args_cli.max_visible_envs = 4

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import pandas as pd
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
from unitree_rl_lab.tasks.locomotion.robots.go2.adaptive_energy_pie_stairs_env_cfg import (
    PIE_STAIRS_COLUMNS_PER_TYPE,
    PIE_STAIRS_TERRAIN_NAMES,
    PIE_STAIRS_TERRAINS_CFG,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.adaptive_energy_pie_terrain_cfg import (
    ADAPTIVE_ENERGY_PIE_COLUMNS_PER_FAMILY,
    ADAPTIVE_ENERGY_PIE_TERRAIN_NAMES,
    ADAPTIVE_ENERGY_PIE_TERRAINS_CFG,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.adaptive_energy_stairs_pie_env_cfg import (
    STAIRS_PIE_NUM_LEVELS,
)
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


_FIXED_COURSE_GEOMETRY = {
    "stairs_up": "stairs_down",
    "stairs_down": "stairs_up",
    "slope_up": "slope_down",
    "slope_down": "slope_up",
    "random_rough": "random_rough",
}
_STAIRS_COURSE_GEOMETRY = {"stairs_up": "stairs_up", "stairs_down": "stairs_down"}
_UNIFIED_COURSE_GEOMETRY = {
    "stairs_up": "stairs_up",
    "stairs_down": "stairs_down",
    "slope_up": "slope_up",
    "slope_down": "slope_down",
    "random_rough": "random_rough",
    "obstacles": "obstacles",
}


def _as_torch(value):
    return getattr(value, "torch", value)


def _checkpoint_is_pie(checkpoint: str) -> bool:
    """Identify PIE actor checkpoints from their state-dict contract."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor_state = saved.get("actor_state_dict")
    if not isinstance(actor_state, dict):
        raise KeyError(f"Checkpoint {checkpoint!r} has no actor_state_dict.")
    return "vae_mu_head.weight" in actor_state and "depth_encoder.0.weight" in actor_state


def _resolve_task_for_checkpoint(task: str, checkpoint: str, *, explicit: bool) -> str:
    """Validate an explicit task or correct an inferred task using checkpoint contents."""
    checkpoint_is_pie = _checkpoint_is_pie(checkpoint)
    task_is_pie = task in _PIE_TASKS
    if checkpoint_is_pie != task_is_pie:
        checkpoint_hint = checkpoint.lower()
        expected = (
            _PIE_STAIRS_TASK
            if checkpoint_is_pie and ("pie_stairs" in checkpoint_hint or "pie-stairs" in checkpoint_hint)
            else _PIE_STAIRS_LADDER_TASK
            if checkpoint_is_pie and "stairs_pie" in checkpoint_hint
            else _PIE_FLAT_TASK
            if checkpoint_is_pie and "flat_lpacrl_pie" in checkpoint_hint
            else _PIE_TASK if checkpoint_is_pie else _DEFAULT_TASK
        )
        checkpoint_kind = "PIE" if checkpoint_is_pie else "non-PIE"
        if explicit:
            raise ValueError(
                f"The {checkpoint_kind} checkpoint is incompatible with task {task!r}. "
                f"Use --task {expected}."
            )
        print(f"[INFO] Checkpoint contents select task {expected!r} instead of inferred task {task!r}.")
        return expected
    return task


def _disable_observation_corruption(env_cfg) -> None:
    """Disable evaluation noise for both flat-policy and PIE observation layouts."""
    observations = env_cfg.observations
    configured = False
    for group_name in ("policy", "actor", "proprio_history"):
        group = getattr(observations, group_name, None)
        if group is not None and hasattr(group, "enable_corruption"):
            group.enable_corruption = False
            configured = True
    if not configured:
        raise AttributeError("Task has no policy, actor or proprio_history observation group to configure.")


def _apply_observation_ablation(observation) -> None:
    """Apply requested policy-input ablations in place."""
    if not args_cli.zero_depth_observation:
        return
    if "camera" not in observation:
        raise KeyError("--zero_depth_observation requires a PIE observation group named 'camera'.")
    observation["camera"].zero_()


def _terrain_columns(task: str) -> dict[str, list[int]]:
    """Return the four deterministic geometry columns for every terrain family."""
    if task in _STAIRS_FAMILY:
        names = PIE_STAIRS_TERRAIN_NAMES
        columns_per_type = PIE_STAIRS_COLUMNS_PER_TYPE
    elif task == _PIE_MULTI_TERRAIN_TASK:
        names = ADAPTIVE_ENERGY_PIE_TERRAIN_NAMES
        columns_per_type = ADAPTIVE_ENERGY_PIE_COLUMNS_PER_FAMILY
    else:
        names = LPACRL_TERRAIN_NAMES
        columns_per_type = LPACRL_COLUMNS_PER_TYPE
    return {
        name: list(range(index * columns_per_type, (index + 1) * columns_per_type))
        for index, name in enumerate(names)
    }


def _install_fixed_commands(env, commands: torch.Tensor) -> None:
    term = env.unwrapped.command_manager.get_term("base_velocity")
    commands = commands.to(term.device)

    def _resample_fixed(self, env_ids):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.vel_command_b[env_ids] = commands[env_ids]
        self.is_standing_env[env_ids] = False
        if hasattr(self, "is_heading_env"):
            self.is_heading_env[env_ids] = False
        if hasattr(self, "bin_ids"):
            self.bin_ids[env_ids] = -1
        for name in (
            "_velocity_abs_error_sum",
            "_segment_steps",
        ):
            if hasattr(self, name):
                getattr(self, name)[env_ids] = 0

    term._resample_command = MethodType(_resample_fixed, term)


def _configure_deterministic_courses(
    env_cfg,
    terrain_names: list[str],
    geometry_mapping: dict[str, str],
) -> None:
    """Fix reset state and provide a full landing after fixed-course terrain segments."""
    generator_cfg = env_cfg.scene.terrain.terrain_generator
    for requested_name in terrain_names:
        geometry_name = geometry_mapping.get(requested_name)
        if geometry_name is not None:
            geometry_cfg = generator_cfg.sub_terrains[geometry_name]
            minimum_border_width = 2.0 if requested_name == "random_rough" else 1.0
            geometry_cfg.border_width = max(float(geometry_cfg.border_width), minimum_border_width)

    reset_base = env_cfg.events.reset_base
    reset_base.params["pose_range"] = {
        key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    reset_base.params["velocity_range"] = {
        key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


def _heading_from_xyzw(quaternions: torch.Tensor) -> torch.Tensor:
    """Return world yaw for Isaac Lab xyzw quaternions."""
    qx, qy, qz, qw = quaternions.unbind(dim=1)
    return torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi] without relying on version-specific helpers."""
    return torch.atan2(torch.sin(angles), torch.cos(angles))


def _configure_multi_robot_camera(env) -> None:
    """Frame the environments made visible by the Kit visualizer."""
    unwrapped_env = env.unwrapped
    camera_controller = getattr(unwrapped_env, "viewport_camera_controller", None)
    if camera_controller is None:
        print("[INFO] Multi-robot camera unavailable: Kit viewport was not created.")
        return

    visible_env_ids = None
    for visualizer in unwrapped_env.sim.visualizers:
        if getattr(visualizer.cfg, "visualizer_type", None) == "kit":
            visible_env_ids = visualizer.get_visualized_env_ids()
            break
    if visible_env_ids == []:
        print("[INFO] Multi-robot camera unavailable: the Kit visualizer has no visible environments.")
        return
    if visible_env_ids is None:
        visible_env_ids = list(range(unwrapped_env.num_envs))

    origins = _as_torch(unwrapped_env.scene.env_origins)[visible_env_ids].detach().cpu()
    minimum = origins.min(dim=0).values
    maximum = origins.max(dim=0).values
    center = 0.5 * (minimum + maximum)
    center[0] += 1.5
    horizontal_span = max(float(maximum[0] - minimum[0]) + 8.0, float(maximum[1] - minimum[1]) + 8.0)
    camera_controller.update_view_to_world()
    camera_controller.update_view_location(
        eye=(
            float(center[0] - 0.55 * horizontal_span),
            float(center[1] - 0.75 * horizontal_span),
            0.70 * horizontal_span,
        ),
        lookat=(float(center[0]), float(center[1]), 0.4),
    )
    print(f"[INFO] Multi-robot overview camera frames {len(visible_env_ids)} visible environments.")


def _load_actor(env, agent_cfg, checkpoint: str):
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(
        checkpoint,
        load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False},
        map_location=agent_cfg.device,
    )
    return runner.get_inference_policy(device=env.unwrapped.device)


def _plot_results(
    summary,
    terrain_names,
    levels,
    speeds,
    output_dir: Path,
    cases=None,
    dt: float = 0.02,
    vx_history=None,
) -> None:
    """Delegate plotting to the standalone, GPU-free plotting module."""
    from plot_terrain import plot_results

    plot_results(summary, terrain_names, levels, speeds, output_dir, cases=cases, dt=dt, vx_history=vx_history)


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


def _configure_nominal_evaluation(env_cfg) -> None:
    """Disable evaluation-time randomness without changing robot dynamics."""
    policy_group = getattr(env_cfg.observations, "policy", None)
    if policy_group is not None and hasattr(policy_group, "enable_corruption"):
        policy_group.enable_corruption = False
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


def _safe_nanmean(values) -> float:
    array = values.to_numpy(dtype=float)
    return float(np.nanmean(array)) if np.isfinite(array).any() else float("nan")


def _ci95(values) -> float:
    array = values.to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return float("nan")
    return float(1.96 * np.std(array, ddof=1) / np.sqrt(array.size))


def _summarize_speed(per_env) -> "pd.DataFrame":
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


def _save_speed_plots(summary, time_series, output_dir: Path) -> None:
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


def _run_speed_tracking() -> None:
    speeds = _make_speed_list()
    num_speeds = len(speeds)
    num_envs = num_speeds * args_cli.envs_per_speed

    checkpoint = str(Path(args_cli.checkpoint).expanduser().resolve())
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    args_cli.task = _resolve_task_for_checkpoint(
        args_cli.task,
        checkpoint,
        explicit=_TASK_WAS_EXPLICIT,
    )
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
    env_cfg.curriculum = None
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = args_cli.seed
    _disable_observation_corruption(env_cfg)
    if not args_cli.robust_eval:
        _configure_nominal_evaluation(env_cfg)

    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

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
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    policy = _load_actor(env, agent_cfg, checkpoint)

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
            if getattr(policy, "is_recurrent", False):
                policy.reset(dones.bool())

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
    summary = _summarize_speed(per_env)

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
        "evaluation_mode": "speed",
        "randomization": "robust" if args_cli.robust_eval else "nominal",
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
        _save_speed_plots(summary, time_series, output_dir)

    print("\n[INFO] Speed tracking summary:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\n[INFO] Evaluation results saved to: {output_dir}")
    env.close()


def main() -> None:
    if _SPEED_MODE:
        _run_speed_tracking()
        return
    if args_cli.speeds is None:
        args_cli.speeds = [0.5, 1.0, 1.5, 2.0, 2.5]
    checkpoint = str(Path(args_cli.checkpoint).expanduser().resolve())
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    args_cli.task = _resolve_task_for_checkpoint(
        args_cli.task,
        checkpoint,
        explicit=_TASK_WAS_EXPLICIT,
    )
    is_plane_task = args_cli.task == _PIE_FLAT_TASK
    geometry_mapping = (
        _STAIRS_COURSE_GEOMETRY
        if args_cli.task in _STAIRS_FAMILY
        else _UNIFIED_COURSE_GEOMETRY
        if args_cli.task == _PIE_MULTI_TERRAIN_TASK
        else _FIXED_COURSE_GEOMETRY
    )
    terrain_cfg = (
        PIE_STAIRS_TERRAINS_CFG
        if args_cli.task in _STAIRS_FAMILY
        else ADAPTIVE_ENERGY_PIE_TERRAINS_CFG
        if args_cli.task == _PIE_MULTI_TERRAIN_TASK
        else LPACRL_TERRAINS_CFG
    )
    if args_cli.task == _PIE_STAIRS_LADDER_TASK:
        terrain_cfg.num_rows = STAIRS_PIE_NUM_LEVELS
    if is_plane_task:
        if set(args_cli.terrain_types or ["flat"]) != {"flat"}:
            raise ValueError("The flat task evaluates on a single infinite plane; use --terrain_types flat.")
        if args_cli.terrain_levels != [0]:
            raise ValueError("The flat task has no terrain levels; use --terrain_levels 0.")
        terrain_names = ["flat"]
        args_cli.terrain_levels = [0]
        column_map = {"flat": [0]}
    else:
        column_map = _terrain_columns(args_cli.task)
        terrain_names = args_cli.terrain_types or list(column_map)
        unknown = sorted(set(terrain_names) - set(column_map))
        if unknown:
            raise ValueError(f"Unknown terrain types {unknown}; choose from {list(column_map)}")
        if any(not column_map[name] for name in terrain_names):
            raise ValueError("At least one selected terrain has no generated column.")
    if any(name in geometry_mapping for name in terrain_names) and min(args_cli.speeds) <= 0.0:
        raise ValueError("Fixed +x course tests require positive commanded speeds.")
    if min(args_cli.terrain_levels) < 0 or max(args_cli.terrain_levels) >= terrain_cfg.num_rows:
        raise ValueError(f"Terrain levels must be in [0, {terrain_cfg.num_rows - 1}].")

    cases = [
        (name, level, speed, replicate)
        for name in terrain_names
        for level in args_cli.terrain_levels
        for speed in args_cli.speeds
        for replicate in range(args_cli.envs_per_case)
    ]
    num_envs = len(cases)
    if args_cli.zero_depth_observation and args_cli.task not in _PIE_TASKS:
        raise ValueError("--zero_depth_observation is only supported by the PIE task.")
    if args_cli.zero_depth_observation:
        print("[INFO] PIE ablation enabled: camera depth-history observations will be zeroed.")

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
    # The evaluator records landing, lateral and heading outcomes itself.  A
    # task-owned termination would auto-reset the robot before its terminal
    # pose can be measured and would incorrectly turn success into failure.
    for term_name in ("course_complete", "lateral_deviation", "heading_error"):
        if hasattr(env_cfg.terminations, term_name):
            setattr(env_cfg.terminations, term_name, None)
    if is_plane_task:
        # The plane task has no terrain generator; only zero the reset state so
        # every evaluation starts deterministically at the env origin.
        reset_base = env_cfg.events.reset_base
        reset_base.params["pose_range"] = {
            key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        reset_base.params["velocity_range"] = {
            key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
        }
    else:
        _configure_deterministic_courses(env_cfg, terrain_names, geometry_mapping)
    env_cfg.commands.base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        rel_standing_envs=0.0,
        debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-2.5, 2.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(-2.5, 2.5)
        ),
    )
    _disable_observation_corruption(env_cfg)
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
    if is_plane_task:
        columns = torch.zeros(num_envs, dtype=torch.long, device=args_cli.device)
    else:
        columns = torch.tensor(
            [
                column_map[geometry_mapping.get(case[0], case[0])][
                    case[3] % len(column_map[geometry_mapping.get(case[0], case[0])])
                ]
                for case in cases
            ],
            dtype=torch.long,
            device=args_cli.device,
        )
    commands[:, 0] = torch.tensor([case[2] for case in cases], device=args_cli.device)

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    terrain = gym_env.unwrapped.scene.terrain
    if is_plane_task:
        # A plane importer has no terrain levels/types; keep the default
        # env-origins grid and evaluate pure velocity tracking.
        pass
    else:
        terrain.terrain_levels[:] = levels.to(terrain.terrain_levels.device)
        terrain.terrain_types[:] = columns.to(terrain.terrain_types.device)
        selected_origins = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types].clone()
        random_rough_mask = torch.tensor(
            [case[0] == "random_rough" for case in cases],
            dtype=torch.bool,
            device=selected_origins.device,
        )
        if bool(random_rough_mask.any().item()):
            generator_cfg = terrain.cfg.terrain_generator
            rough_cfg = generator_cfg.sub_terrains["random_rough"]
            effective_border = float(rough_cfg.border_width) + float(rough_cfg.horizontal_scale)
            rough_start_x = selected_origins[:, 0] - 0.5 * float(generator_cfg.size[0]) + effective_border
            selected_origins[random_rough_mask, 0] = (
                rough_start_x[random_rough_mask] - args_cli.rough_runup_distance
            )
            selected_origins[random_rough_mask, 2] = 0.0
        terrain.env_origins[:] = selected_origins
    _install_fixed_commands(gym_env, commands)
    gym_env.reset()
    if args_cli.multi_robot_view:
        _configure_multi_robot_camera(gym_env)

    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    policy = _load_actor(env, agent_cfg, checkpoint)
    robot = env.unwrapped.scene["robot"]
    command_term = env.unwrapped.command_manager.get_term("base_velocity")
    device = env.unwrapped.device
    commands = commands.to(device)
    dt = float(env.unwrapped.step_dt)
    warmup_steps = round(args_cli.warmup_s / dt)
    eval_steps = round(args_cli.eval_s / dt)

    fixed_course = torch.tensor(
        [case[0] in geometry_mapping for case in cases], dtype=torch.bool, device=device
    )
    centers = (
        terrain.env_origins.clone().to(device)
        if is_plane_task
        else terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types].to(device)
    )
    obstacle_start = torch.full((num_envs,), torch.nan, device=device)
    obstacle_end = torch.full((num_envs,), torch.nan, device=device)
    finish_x = torch.full((num_envs,), torch.nan, device=device)
    for index, case in enumerate(cases):
        if case[0] not in geometry_mapping:
            continue
        geometry_name = geometry_mapping[case[0]]
        generator_cfg = terrain.cfg.terrain_generator
        geometry_cfg = generator_cfg.sub_terrains[geometry_name]
        border_width = float(geometry_cfg.border_width)
        effective_border = border_width + float(getattr(geometry_cfg, "horizontal_scale", 0.0))
        if case[0] == "random_rough":
            obstacle_start[index] = centers[index, 0] - 0.5 * float(generator_cfg.size[0]) + effective_border
        else:
            obstacle_start[index] = centers[index, 0] + 0.5 * float(geometry_cfg.platform_width)
        obstacle_end[index] = centers[index, 0] + 0.5 * float(generator_cfg.size[0]) - effective_border
        finish_x[index] = obstacle_end[index] + min(0.30, 0.5 * border_width)
        if float(obstacle_end[index] - obstacle_start[index]) <= 0.5:
            raise ValueError(
                f"The measured {case[0]} course is too short; reduce --rough_runup_distance or border width."
            )

    alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    failed = torch.zeros(num_envs, dtype=torch.bool, device=device)
    succeeded = torch.zeros(num_envs, dtype=torch.bool, device=device)
    course_started = torch.zeros(num_envs, dtype=torch.bool, device=device)
    course_ended = torch.zeros(num_envs, dtype=torch.bool, device=device)
    course_start_time = torch.full((num_envs,), torch.nan, device=device)
    course_end_time = torch.full((num_envs,), torch.nan, device=device)
    course_start_x = torch.full((num_envs,), torch.nan, device=device)
    course_end_x = torch.full((num_envs,), torch.nan, device=device)
    max_lateral_error = torch.zeros(num_envs, device=device)
    max_heading_error = torch.zeros(num_envs, device=device)
    failure_reasons = [""] * num_envs
    count = torch.zeros(num_envs, device=device)
    abs_error = torch.zeros(num_envs, 3, device=device)
    square_error = torch.zeros(num_envs, 3, device=device)
    velocity_sum = torch.zeros(num_envs, device=device)
    power_sum = torch.zeros(num_envs, device=device)
    energy = torch.zeros(num_envs, device=device)
    distance = torch.zeros(num_envs, device=device)
    vx_history = [] if not args_cli.no_plots else None

    def mark_failed(mask: torch.Tensor, reason: str) -> None:
        mask = mask & ~failed & ~succeeded
        for failed_index in torch.where(mask)[0].detach().cpu().tolist():
            failure_reasons[failed_index] = reason
        failed[mask] = True

    observation = env.get_observations()
    observation = observation[0] if isinstance(observation, tuple) else observation
    previous_world_x = _as_torch(robot.data.root_pos_w)[:, 0].clone()
    max_heading_error_rad = torch.deg2rad(torch.tensor(args_cli.max_heading_error_deg, device=device))
    for step in range(warmup_steps + eval_steps):
        wall_step_start = time.perf_counter()
        with torch.inference_mode():
            _apply_observation_ablation(observation)
            actions = policy(observation)
            actions[fixed_course & (failed | succeeded)] = 0.0
            observation, _, dones, _ = env.step(actions)
            if getattr(policy, "is_recurrent", False):
                policy.reset(dones.bool())
        done_mask = dones.bool()
        alive &= ~done_mask

        root_pos = _as_torch(robot.data.root_pos_w)
        root_quat = _as_torch(robot.data.root_quat_w)
        world_x = root_pos[:, 0]
        lateral_error = torch.abs(root_pos[:, 1] - centers[:, 1])
        world_heading = _heading_from_xyzw(root_quat)
        target_heading = torch.zeros_like(world_heading)
        if not args_cli.disable_goal_direction_control:
            remaining_x = torch.clamp(finish_x - world_x, min=0.5)
            target_heading[fixed_course] = torch.atan2(
                centers[fixed_course, 1] - root_pos[fixed_course, 1],
                remaining_x[fixed_course],
            )
        heading_error_signed = _wrap_to_pi(target_heading - world_heading)
        heading_error = torch.abs(heading_error_signed)
        course_active = fixed_course & ~failed & ~succeeded
        max_lateral_error = torch.where(
            course_active, torch.maximum(max_lateral_error, lateral_error), max_lateral_error
        )
        max_heading_error = torch.where(
            course_active, torch.maximum(max_heading_error, heading_error), max_heading_error
        )

        mark_failed(course_active & done_mask, "fall/episode reset")
        course_active = fixed_course & ~failed & ~succeeded
        mark_failed(
            course_active & (lateral_error > args_cli.max_lateral_deviation),
            "lateral deviation limit exceeded",
        )
        course_active = fixed_course & ~failed & ~succeeded
        mark_failed(
            course_active & (heading_error > max_heading_error_rad),
            "heading error limit exceeded",
        )

        course_active = fixed_course & ~failed & ~succeeded
        entering = course_active & ~course_started & (world_x >= obstacle_start)
        step_progress = world_x - previous_world_x
        entry_fraction = torch.where(
            step_progress > 1.0e-6,
            torch.clamp((obstacle_start - previous_world_x) / step_progress, 0.0, 1.0),
            torch.ones_like(step_progress),
        )
        course_started[entering] = True
        course_start_time[entering] = (step + entry_fraction[entering]) * dt
        course_start_x[entering] = obstacle_start[entering]
        ending = course_active & course_started & ~course_ended & (world_x >= obstacle_end)
        end_fraction = torch.where(
            step_progress > 1.0e-6,
            torch.clamp((obstacle_end - previous_world_x) / step_progress, 0.0, 1.0),
            torch.ones_like(step_progress),
        )
        course_ended[ending] = True
        course_end_time[ending] = (step + end_fraction[ending]) * dt
        course_end_x[ending] = obstacle_end[ending]
        landing = course_active & course_ended & (world_x >= finish_x)
        succeeded[landing] = True

        lin_vel_b = _as_torch(robot.data.root_lin_vel_b)
        lin_vel_w = _as_torch(robot.data.root_lin_vel_w)
        ang_vel = _as_torch(robot.data.root_ang_vel_b)
        measured_vx = torch.where(fixed_course, lin_vel_w[:, 0], lin_vel_b[:, 0])
        velocity = torch.stack((measured_vx, lin_vel_b[:, 1], ang_vel[:, 2]), dim=1)
        error = velocity - commands
        joint_vel = _as_torch(robot.data.joint_vel)
        torque = _as_torch(robot.data.applied_torque)
        power = torch.sum(torch.abs(joint_vel * torque), dim=1)

        steady_valid = ~fixed_course & alive & ~done_mask & (step >= warmup_steps)
        obstacle_valid = fixed_course & course_started & (~course_ended | ending) & ~failed
        valid = steady_valid | obstacle_valid
        count += valid.float()
        abs_error += torch.where(valid.unsqueeze(1), torch.abs(error), 0.0)
        square_error += torch.where(valid.unsqueeze(1), torch.square(error), 0.0)
        velocity_sum += torch.where(valid, measured_vx, 0.0)
        if vx_history is not None:
            history_vx = torch.where(fixed_course & (failed | succeeded), torch.nan, measured_vx)
            vx_history.append(history_vx.detach().cpu().clone())
        power_sum += torch.where(valid, power, 0.0)
        energy += torch.where(valid, power * dt, 0.0)
        forward_progress = torch.where(
            fixed_course,
            torch.clamp(world_x - previous_world_x, min=0.0),
            torch.abs(lin_vel_b[:, 0]) * dt,
        )
        distance += torch.where(valid, forward_progress, 0.0)
        previous_world_x = world_x.clone()

        commands[fixed_course, 2] = 0.0
        if not args_cli.disable_goal_direction_control:
            steering_mask = fixed_course & ~failed & ~succeeded
            commands[steering_mask, 2] = torch.clamp(
                args_cli.heading_control_gain * heading_error_signed[steering_mask],
                min=-args_cli.max_heading_command,
                max=args_cli.max_heading_command,
            )
        command_term.vel_command_b[:, :3] = commands
        if args_cli.real_time:
            remaining_step_time = dt - (time.perf_counter() - wall_step_start)
            if remaining_step_time > 0.0:
                time.sleep(remaining_step_time)

    mark_failed(fixed_course & ~failed & ~succeeded, "course timeout")

    denom = count.clamp_min(1.0)
    mae = abs_error / denom.unsqueeze(1)
    rmse = torch.sqrt(square_error / denom.unsqueeze(1))
    mean_vx = velocity_sum / denom
    mean_power = power_sum / denom
    mass = _as_torch(robot.data.body_mass).sum(dim=1)
    cot = torch.where(distance > 1.0e-3, energy / (mass * 9.81 * distance), torch.nan)
    traversal_time = (course_end_time - course_start_time).clamp_min(dt)
    world_displacement = course_end_x - course_start_x
    course_speed = torch.where(succeeded, world_displacement / traversal_time, torch.nan)
    cot = torch.where(
        fixed_course & succeeded & (world_displacement > 1.0e-3),
        energy / (mass * 9.81 * world_displacement),
        cot,
    )

    rows = []
    for index, case in enumerate(cases):
        is_fixed_course = case[0] in geometry_mapping
        success = bool(succeeded[index].item()) if is_fixed_course else bool(alive[index].item())
        if is_fixed_course:
            measured_speed = float(course_speed[index]) if success else float("nan")
            speed_error = measured_speed - float(commands[index, 0]) if success else float("nan")
            mae_vx = abs(speed_error) if success else float("nan")
            rmse_vx = mae_vx
            mean_power_w = float(mean_power[index]) if success else float("nan")
            mechanical_cot = float(cot[index]) if success else float("nan")
            failure_reason = failure_reasons[index]
        else:
            measured_speed = float(mean_vx[index])
            speed_error = float(mean_vx[index] - commands[index, 0])
            mae_vx = float(mae[index, 0])
            rmse_vx = float(rmse[index, 0])
            mean_power_w = float(mean_power[index])
            mechanical_cot = float(cot[index])
            failure_reason = "" if success else "fall/episode reset"
        rows.append(
            {
                "terrain_type": case[0],
                "geometry_terrain": geometry_mapping.get(case[0], case[0]),
                "terrain_level": case[1],
                "command_vx": case[2],
                "replicate": case[3],
                "terrain_column": int(columns[index].item()),
                "fixed_course": is_fixed_course,
                "success": success,
                "failure_reason": failure_reason,
                "survived": success if is_fixed_course else bool(alive[index].item()),
                "samples": int(count[index].item()),
                "mean_vx": measured_speed,
                "bias_vx": speed_error,
                "mae_vx": mae_vx,
                "rmse_vx": rmse_vx,
                "mae_vy": float(mae[index, 1]) if not is_fixed_course else float("nan"),
                "mae_yaw": float(mae[index, 2]) if not is_fixed_course else float("nan"),
                "traversal_time_s": (
                    float(traversal_time[index]) if success and is_fixed_course else float("nan")
                ),
                "world_displacement_m": (
                    float(world_displacement[index]) if success and is_fixed_course else float("nan")
                ),
                "max_lateral_error_m": float(max_lateral_error[index]) if is_fixed_course else float("nan"),
                "max_heading_error_deg": (
                    math.degrees(float(max_heading_error[index])) if is_fixed_course else float("nan")
                ),
                "mean_power_w": mean_power_w,
                "mechanical_cot": mechanical_cot,
            }
        )

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["terrain_type"], row["terrain_level"], row["command_vx"])].append(row)
    summary = []

    def successful_mean(group, metric: str) -> float:
        values = [row[metric] for row in group if row["success"] and math.isfinite(row[metric])]
        return sum(values) / len(values) if values else float("nan")

    for key, group in grouped.items():
        summary.append(
            {
                "terrain_type": key[0],
                "terrain_level": key[1],
                "command_vx": key[2],
                "attempts": len(group),
                "successful_attempts": sum(row["success"] for row in group),
                "success_rate": sum(row["success"] for row in group) / len(group),
                "survival_rate": sum(row["survived"] for row in group) / len(group),
                "mean_vx": successful_mean(group, "mean_vx"),
                "bias_vx": successful_mean(group, "bias_vx"),
                "mae_vx": successful_mean(group, "mae_vx"),
                "rmse_vx": successful_mean(group, "rmse_vx"),
                "mean_traversal_time_s": successful_mean(group, "traversal_time_s"),
                "mean_max_lateral_error_m": successful_mean(group, "max_lateral_error_m"),
                "mean_max_heading_error_deg": successful_mean(group, "max_heading_error_deg"),
                "mean_power_w": successful_mean(group, "mean_power_w"),
                "mechanical_cot": successful_mean(group, "mechanical_cot"),
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
        json.dump(
            vars(args_cli)
            | {
                "checkpoint": checkpoint,
                "num_envs": num_envs,
                "fixed_course_geometry_mapping": geometry_mapping,
                "fixed_course_speed_definition": "world +x displacement / course traversal time; successes only",
                "fixed_course_heading_reference": (
                    "bearing from the robot to the course endpoint"
                    if not args_cli.disable_goal_direction_control
                    else "world +x"
                ),
                "random_rough_definition": (
                    "spawn on the left flat border, run up to the rough boundary, cross the full rough segment, "
                    "and finish on the right flat border"
                ),
            },
            stream,
            indent=2,
        )

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
    course_attempts = int(fixed_course.sum().item())
    if course_attempts:
        course_successes = int((succeeded & fixed_course).sum().item())
        print(
            f"[INFO] Fixed courses: {course_successes}/{course_attempts} successful "
            f"({course_successes / course_attempts:.1%})."
        )
    print(f"[INFO] Results: {output_dir}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
