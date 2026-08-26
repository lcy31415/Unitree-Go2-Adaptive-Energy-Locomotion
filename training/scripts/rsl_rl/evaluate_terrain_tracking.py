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


parser = argparse.ArgumentParser(description="Evaluate terrain-conditioned speed tracking.")
parser.add_argument(
    "--task",
    default=None,
    help="Gym task ID. If omitted, infer the PIE task from the checkpoint path and otherwise use Terrain-LPACRL.",
)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--speeds", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 2.5])
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
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
_TASK_WAS_EXPLICIT = args_cli.task is not None

if args_cli.checkpoint is None:
    parser.error("--checkpoint is required.")
if args_cli.envs_per_case < 1 or args_cli.eval_s <= 0.0 or args_cli.warmup_s < 0.0:
    parser.error("envs_per_case must be positive and evaluation durations must be valid.")
if args_cli.max_lateral_deviation <= 0.0 or not 0.0 < args_cli.max_heading_error_deg <= 180.0:
    parser.error("Course deviation limits must be positive and heading error must not exceed 180 degrees.")
if args_cli.rough_runup_distance < 0.0:
    parser.error("rough_runup_distance must be non-negative.")
if args_cli.heading_control_gain < 0.0 or args_cli.max_heading_command <= 0.0:
    parser.error("Heading-control gain must be non-negative and its command limit must be positive.")
if args_cli.task is None:
    checkpoint_hint = str(Path(args_cli.checkpoint).expanduser()).lower()
    args_cli.task = _PIE_TASK if "lpacrl_pie" in checkpoint_hint else _DEFAULT_TASK
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


_FIXED_COURSE_GEOMETRY = {
    "stairs_up": "stairs_down",
    "stairs_down": "stairs_up",
    "slope_up": "slope_down",
    "slope_down": "slope_up",
    "random_rough": "random_rough",
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
    task_is_pie = task == _PIE_TASK
    if checkpoint_is_pie != task_is_pie:
        expected = _PIE_TASK if checkpoint_is_pie else _DEFAULT_TASK
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


def _configure_deterministic_courses(env_cfg, terrain_names: list[str]) -> None:
    """Fix reset state and provide a full landing after fixed-course terrain segments."""
    generator_cfg = env_cfg.scene.terrain.terrain_generator
    for requested_name in terrain_names:
        geometry_name = _FIXED_COURSE_GEOMETRY.get(requested_name)
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
    from plot_terrain_tracking import plot_results

    plot_results(summary, terrain_names, levels, speeds, output_dir, cases=cases, dt=dt, vx_history=vx_history)


def main() -> None:
    column_map = _terrain_columns()
    terrain_names = args_cli.terrain_types or list(column_map)
    unknown = sorted(set(terrain_names) - set(column_map))
    if unknown:
        raise ValueError(f"Unknown terrain types {unknown}; choose from {list(column_map)}")
    if any(not column_map[name] for name in terrain_names):
        raise ValueError("At least one selected terrain has no generated column.")
    if any(name in _FIXED_COURSE_GEOMETRY for name in terrain_names) and min(args_cli.speeds) <= 0.0:
        raise ValueError("Fixed +x course tests require positive commanded speeds.")
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
    args_cli.task = _resolve_task_for_checkpoint(
        args_cli.task,
        checkpoint,
        explicit=_TASK_WAS_EXPLICIT,
    )

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
    _configure_deterministic_courses(env_cfg, terrain_names)
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
    columns = torch.tensor(
        [
            column_map[_FIXED_COURSE_GEOMETRY.get(case[0], case[0])][
                case[3] % len(column_map[_FIXED_COURSE_GEOMETRY.get(case[0], case[0])])
            ]
            for case in cases
        ],
        dtype=torch.long,
        device=args_cli.device,
    )
    commands[:, 0] = torch.tensor([case[2] for case in cases], device=args_cli.device)

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    terrain = gym_env.unwrapped.scene.terrain
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
        [case[0] in _FIXED_COURSE_GEOMETRY for case in cases], dtype=torch.bool, device=device
    )
    centers = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types].to(device)
    obstacle_start = torch.full((num_envs,), torch.nan, device=device)
    obstacle_end = torch.full((num_envs,), torch.nan, device=device)
    finish_x = torch.full((num_envs,), torch.nan, device=device)
    for index, case in enumerate(cases):
        if case[0] not in _FIXED_COURSE_GEOMETRY:
            continue
        geometry_name = _FIXED_COURSE_GEOMETRY[case[0]]
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
            actions = policy(observation)
            actions[fixed_course & (failed | succeeded)] = 0.0
            observation, _, dones, _ = env.step(actions)
        done_mask = dones.bool()
        if getattr(policy, "is_recurrent", False):
            policy.reset(done_mask)
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
        is_fixed_course = case[0] in _FIXED_COURSE_GEOMETRY
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
                "geometry_terrain": _FIXED_COURSE_GEOMETRY.get(case[0], case[0]),
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
                "fixed_course_geometry_mapping": _FIXED_COURSE_GEOMETRY,
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
