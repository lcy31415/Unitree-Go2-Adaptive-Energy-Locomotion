"""Interactive demo: run a policy on one fixed (terrain, level, speed, yaw) task."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from types import MethodType

# Allow this repository-local script to run before ``unitree_rl_lab`` has
# been installed in editable mode in the active Python environment.
_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "source" / "unitree_rl_lab"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Demo a policy on a single fixed terrain task.")
parser.add_argument("--task", default="Unitree-Go2-Adaptive-Energy-Terrain-LPACRL")
parser.add_argument("--checkpoint", required=True)
parser.add_argument(
    "--terrain_type",
    default="random_rough",
    help="flat/stairs_up/stairs_down/slope_up/slope_down/random_rough",
)
parser.add_argument("--terrain_level", type=int, default=0)
parser.add_argument("--vx", type=float, default=1.0, help="Commanded forward speed [m/s] (sign allowed).")
parser.add_argument("--yaw", type=float, default=0.0, help="Commanded yaw rate [rad/s] (sign allowed).")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--duration_s", type=float, default=60.0)
parser.add_argument("--follow_distance", type=float, default=2.5, help="Chase-camera distance behind the robot [m].")
parser.add_argument("--follow_height", type=float, default=1.1, help="Chase-camera height above the robot [m].")
parser.add_argument("--max_lateral_deviation", type=float, default=1.0, help="Maximum course-center error [m].")
parser.add_argument("--max_heading_error_deg", type=float, default=20.0, help="Maximum absolute heading error [deg].")
parser.add_argument("--disable_fabric", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs < 1 or args_cli.duration_s <= 0.0:
    parser.error("--num_envs and --duration_s must be positive.")
if args_cli.max_lateral_deviation <= 0.0 or not 0.0 < args_cli.max_heading_error_deg <= 180.0:
    parser.error("Course deviation limits must be positive and heading error must not exceed 180 degrees.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from importlib.metadata import version
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from unitree_rl_lab.tasks.locomotion.robots.go2.adaptive_energy_lpacrl_terrain_cfg import (
    LPACRL_COLUMNS_PER_TYPE,
    LPACRL_TERRAIN_NAMES,
    LPACRL_TERRAINS_CFG,
)
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


# The CLI name describes the direction being tested. Moving from the center
# toward +x descends a positive pyramid and ascends an inverted pyramid.
_DIRECTIONAL_TERRAIN_GEOMETRY = {
    "stairs_up": "stairs_down",
    "stairs_down": "stairs_up",
    "slope_up": "slope_down",
    "slope_down": "slope_up",
}


def _as_torch(value):
    return getattr(value, "torch", value)


def _enable_follow_camera(env, distance: float, height: float) -> None:
    """Follow env 0's robot using the same Kit camera controller as play.py."""
    unwrapped_env = env.unwrapped
    camera_controller = getattr(unwrapped_env, "viewport_camera_controller", None)
    if camera_controller is None:
        print("[DEMO] follow camera unavailable: no Kit viewport camera controller")
        return
    if "robot" not in unwrapped_env.scene.articulations:
        print("[DEMO] follow camera unavailable: scene has no 'robot' articulation")
        return

    visible_env_ids = None
    for visualizer in unwrapped_env.sim.visualizers:
        if getattr(visualizer.cfg, "visualizer_type", None) == "kit":
            visible_env_ids = visualizer.get_visualized_env_ids()
            break
    else:
        print("[DEMO] follow camera unavailable: launch with --viz kit")
        return
    if visible_env_ids == []:
        print("[DEMO] follow camera unavailable: env 0 is not visible in the Kit visualizer")
        return

    env_index = 0 if visible_env_ids is None else visible_env_ids[0]
    camera_controller.set_view_env_index(env_index)
    camera_controller.update_view_to_asset_root("robot")
    camera_controller.update_view_location(
        eye=(-distance, -0.45 * distance, height),
        lookat=(0.45, 0.0, 0.35),
    )
    print(f"[DEMO] Kit camera follows robot in env_{env_index}")


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


def _configure_deterministic_course(env_cfg, requested_terrain: str) -> str:
    """Configure a fixed +x start pose and return the geometry family to generate."""
    geometry_terrain = _DIRECTIONAL_TERRAIN_GEOMETRY.get(requested_terrain, requested_terrain)
    if requested_terrain in _DIRECTIONAL_TERRAIN_GEOMETRY:
        if args_cli.num_envs != 1:
            raise ValueError("Directional terrain course evaluation requires --num_envs 1.")
        generator_cfg = env_cfg.scene.terrain.terrain_generator
        geometry_cfg = generator_cfg.sub_terrains[geometry_terrain]
        # A full-width landing is required after the last step/slope. The
        # training slopes use a 0.25 m border, which is too short for Go2.
        geometry_cfg.border_width = max(float(geometry_cfg.border_width), 1.0)

    reset_base = env_cfg.events.reset_base
    reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    reset_base.params["velocity_range"] = {
        key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    return geometry_terrain


def _make_course_limits(terrain, level: int, column: int, geometry_terrain: str) -> dict[str, float]:
    """Return world-space +x course boundaries for one pyramid terrain tile."""
    generator_cfg = terrain.cfg.terrain_generator
    geometry_cfg = generator_cfg.sub_terrains[geometry_terrain]
    center = terrain.terrain_origins[level, column]
    border_width = float(geometry_cfg.border_width)
    # Height-field conversion allocates one extra horizontal-scale cell for
    # its border; account for it so slope timing ends at the true surface edge.
    effective_border = border_width + float(getattr(geometry_cfg, "horizontal_scale", 0.0))
    obstacle_start = float(center[0]) + 0.5 * float(geometry_cfg.platform_width)
    obstacle_end = float(center[0]) + 0.5 * float(generator_cfg.size[0]) - effective_border
    return {
        "spawn_x": float(center[0]),
        "center_y": float(center[1]),
        "obstacle_start": obstacle_start,
        "obstacle_end": obstacle_end,
        "finish_x": obstacle_end + min(0.30, 0.5 * border_width),
    }


def _heading_from_xyzw(quaternion) -> float:
    """Return world yaw in radians for one Isaac Lab xyzw quaternion."""
    qx, qy, qz, qw = (float(value) for value in quaternion)
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _run_directional_course(env, policy, robot, course: dict[str, float]) -> None:
    """Run one course attempt and report obstacle-only world-frame speed."""
    observation = env.get_observations()
    observation = observation[0] if isinstance(observation, tuple) else observation
    step_dt = float(env.unwrapped.step_dt)
    max_heading_error = math.radians(args_cli.max_heading_error_deg)
    start_step = None
    start_x = None
    obstacle_end_step = None
    obstacle_end_x = None
    steps = 0

    def fail(reason: str, x: float, lateral_error: float, heading_error: float) -> None:
        progress = max(0.0, min(x, course["obstacle_end"]) - course["obstacle_start"])
        total = course["obstacle_end"] - course["obstacle_start"]
        print(
            f"[DEMO] FAILED: {reason}; obstacle_progress={progress:.2f}/{total:.2f} m, "
            f"lateral_error={lateral_error:+.2f} m, heading_error={math.degrees(heading_error):+.1f} deg."
        )
        print("[DEMO] obstacle speed=N/A (failed attempts are excluded).")

    with torch.inference_mode():
        while simulation_app.is_running() and steps * step_dt < args_cli.duration_s:
            observation, _, dones, _ = env.step(policy(observation))
            steps += 1
            elapsed = steps * step_dt

            position = _as_torch(robot.data.root_pos_w)[0]
            quaternion = _as_torch(robot.data.root_quat_w)[0]
            x = float(position[0])
            lateral_error = float(position[1]) - course["center_y"]
            heading_error = _heading_from_xyzw(quaternion)

            if bool(dones[0].item()):
                fail("fall/episode reset", x, lateral_error, heading_error)
                return
            if abs(lateral_error) > args_cli.max_lateral_deviation:
                fail("lateral deviation limit exceeded", x, lateral_error, heading_error)
                return
            if abs(heading_error) > max_heading_error:
                fail("heading error limit exceeded", x, lateral_error, heading_error)
                return

            if start_step is None and x >= course["obstacle_start"]:
                start_step = steps
                start_x = x
                print(f"[DEMO] obstacle entry at t={elapsed:.2f}s, world_x={x:.2f}m")
            if start_step is not None and obstacle_end_step is None and x >= course["obstacle_end"]:
                obstacle_end_step = steps
                obstacle_end_x = x
            if x >= course["finish_x"]:
                if start_step is None or obstacle_end_step is None:
                    fail("invalid course timing", x, lateral_error, heading_error)
                    return
                traversal_time = (obstacle_end_step - start_step) * step_dt
                world_displacement = obstacle_end_x - start_x
                world_speed = world_displacement / max(traversal_time, step_dt)
                print(
                    f"[DEMO] SUCCESS: landing reached at t={elapsed:.2f}s; "
                    f"obstacle_displacement={world_displacement:.2f}m, "
                    f"obstacle_time={traversal_time:.2f}s, world_speed={world_speed:.2f}m/s, "
                    f"lateral_error={lateral_error:+.2f}m, "
                    f"heading_error={math.degrees(heading_error):+.1f}deg."
                )
                return

            if steps % 50 == 0:
                phase = "approach" if start_step is None else "obstacle/landing"
                print(
                    f"[DEMO] t={elapsed:5.1f}s phase={phase} world_x={x:.2f}m "
                    f"lateral_error={lateral_error:+.2f}m "
                    f"heading_error={math.degrees(heading_error):+.1f}deg"
                )

    position = _as_torch(robot.data.root_pos_w)[0]
    quaternion = _as_torch(robot.data.root_quat_w)[0]
    fail(
        "course timeout",
        float(position[0]),
        float(position[1]) - course["center_y"],
        _heading_from_xyzw(quaternion),
    )


def main() -> None:
    if args_cli.terrain_type not in LPACRL_TERRAIN_NAMES:
        raise ValueError(f"Unknown terrain type; choose from {LPACRL_TERRAIN_NAMES}")
    if not 0 <= args_cli.terrain_level < LPACRL_TERRAINS_CFG.num_rows:
        raise ValueError(f"terrain_level must be in [0, {LPACRL_TERRAINS_CFG.num_rows - 1}]")
    checkpoint = str(Path(args_cli.checkpoint).expanduser().resolve())
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    if args_cli.terrain_type in _DIRECTIONAL_TERRAIN_GEOMETRY and args_cli.vx <= 0.0:
        raise ValueError("Directional pyramid course tests require a positive --vx command.")
    if args_cli.terrain_type in _DIRECTIONAL_TERRAIN_GEOMETRY and abs(args_cli.yaw) > 1.0e-6:
        raise ValueError("Directional pyramid course tests require --yaw 0 to stay aligned with +x.")

    num_envs = args_cli.num_envs
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.scene.num_envs = num_envs
    env_cfg.curriculum = None
    geometry_terrain = _configure_deterministic_course(env_cfg, args_cli.terrain_type)
    env_cfg.commands.base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        rel_standing_envs=0.0,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-2.5, 2.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(-2.5, 2.5)
        ),
    )
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.push_robot = None

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    device = gym_env.unwrapped.device
    terrain_index = LPACRL_TERRAIN_NAMES.index(geometry_terrain)
    terrain = gym_env.unwrapped.scene.terrain
    levels = torch.full((num_envs,), args_cli.terrain_level, dtype=torch.long, device=device)
    columns = torch.tensor(
        [terrain_index * LPACRL_COLUMNS_PER_TYPE + env_id % LPACRL_COLUMNS_PER_TYPE for env_id in range(num_envs)],
        dtype=torch.long,
        device=device,
    )
    terrain.terrain_levels[:] = levels
    terrain.terrain_types[:] = columns
    terrain.env_origins[:] = terrain.terrain_origins[levels, columns]
    commands = torch.tensor([[args_cli.vx, 0.0, args_cli.yaw]] * num_envs, device=device)
    _install_fixed_commands(gym_env, commands)
    gym_env.reset()
    _enable_follow_camera(gym_env, args_cli.follow_distance, args_cli.follow_height)

    course = None
    if args_cli.terrain_type in _DIRECTIONAL_TERRAIN_GEOMETRY:
        course = _make_course_limits(terrain, args_cli.terrain_level, int(columns[0]), geometry_terrain)
        print(
            f"[DEMO] directed course={args_cli.terrain_type} geometry={geometry_terrain} "
            f"spawn_x={course['spawn_x']:.2f} obstacle=[{course['obstacle_start']:.2f}, "
            f"{course['obstacle_end']:.2f}] finish_x={course['finish_x']:.2f}"
        )

    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(
        checkpoint,
        load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False},
        map_location=agent_cfg.device,
    )
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    robot = env.unwrapped.scene["robot"]

    print(
        f"[DEMO] terrain={args_cli.terrain_type} level={args_cli.terrain_level} "
        f"vx={args_cli.vx:+.2f} m/s yaw={args_cli.yaw:+.2f} rad/s envs={num_envs} "
        f"duration={args_cli.duration_s:.0f}s"
    )
    if course is not None:
        _run_directional_course(env, policy, robot, course)
        env.close()
        return

    falls = 0
    steps = 0
    speed_sum = 0.0
    observation = env.get_observations()
    observation = observation[0] if isinstance(observation, tuple) else observation
    report_every = 50
    with torch.inference_mode():
        while steps * env.unwrapped.step_dt < args_cli.duration_s:
            observation, _, dones, _ = env.step(policy(observation))
            steps += 1
            falls += int(torch.sum(dones.long()).item())
            speed_sum += float(_as_torch(robot.data.root_lin_vel_b)[:, 0].mean().item())
            if steps % report_every == 0:
                print(
                    f"[DEMO] t={steps * env.unwrapped.step_dt:5.1f}s  "
                    f"mean_vx={speed_sum / steps:+.2f} m/s (cmd {args_cli.vx:+.2f})  resets={falls}"
                )
    print(f"[DEMO] done. mean_vx={speed_sum / max(steps, 1):+.2f} m/s, resets={falls}.")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
