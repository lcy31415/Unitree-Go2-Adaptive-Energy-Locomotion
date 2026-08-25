"""Interactive demo: run a policy on one fixed (terrain, level, speed, yaw) task."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import MethodType

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Demo a policy on a single fixed terrain task.")
parser.add_argument("--task", default="Unitree-Go2-Adaptive-Energy-Terrain-LPACRL")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--terrain_type", default="random_rough", help="flat/stairs_up/stairs_down/slope_up/slope_down/random_rough")
parser.add_argument("--terrain_level", type=int, default=0)
parser.add_argument("--vx", type=float, default=1.0, help="Commanded forward speed [m/s] (sign allowed).")
parser.add_argument("--yaw", type=float, default=0.0, help="Commanded yaw rate [rad/s] (sign allowed).")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--duration_s", type=float, default=60.0)
parser.add_argument("--follow_distance", type=float, default=2.5, help="Chase-camera distance behind the robot [m].")
parser.add_argument("--follow_height", type=float, default=1.1, help="Chase-camera height above the robot [m].")
parser.add_argument("--disable_fabric", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs < 1 or args_cli.duration_s <= 0.0:
    parser.error("--num_envs and --duration_s must be positive.")

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


def _as_torch(value):
    return getattr(value, "torch", value)


def _make_follow_camera(robot, distance: float, height: float):
    """Return a callable that keeps the active viewport camera behind env 0."""
    try:
        import omni.kit.viewport.utility as viewport_util
        from isaacsim.core.utils.viewports import set_camera_view
    except ImportError as error:
        print(f"[DEMO] follow camera unavailable: {error}")
        return None
    try:
        camera_path = str(viewport_util.get_active_viewport().camera_path)
    except Exception as error:
        print(f"[DEMO] follow camera unavailable: no active viewport ({error})")
        return None

    def update() -> bool:
        pos = _as_torch(robot.data.root_pos_w)[0]
        quat = _as_torch(robot.data.root_quat_w)[0]
        qw, qx, qy, qz = (float(value) for value in quat)
        # Unit x-axis of the base expressed in world frame (heading).
        fx = 1.0 - 2.0 * (qy * qy + qz * qz)
        fy = 2.0 * (qx * qy + qw * qz)
        norm = max((fx * fx + fy * fy) ** 0.5, 1.0e-6)
        fx, fy = fx / norm, fy / norm
        try:
            set_camera_view(
                eye=[float(pos[0]) - fx * distance, float(pos[1]) - fy * distance, float(pos[2]) + height],
                target=[float(pos[0]) + fx, float(pos[1]) + fy, float(pos[2]) + 0.3],
                camera_prim_path=camera_path,
            )
        except Exception as error:
            print(f"[DEMO] follow camera stopped: {type(error).__name__}: {error}")
            return False
        return True

    print(f"[DEMO] follow camera active on {camera_path}")
    return update


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


def main() -> None:
    if args_cli.terrain_type not in LPACRL_TERRAIN_NAMES:
        raise ValueError(f"Unknown terrain type; choose from {LPACRL_TERRAIN_NAMES}")
    if not 0 <= args_cli.terrain_level < LPACRL_TERRAINS_CFG.num_rows:
        raise ValueError(f"terrain_level must be in [0, {LPACRL_TERRAINS_CFG.num_rows - 1}]")
    checkpoint = str(Path(args_cli.checkpoint).expanduser().resolve())
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)

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
    terrain_index = LPACRL_TERRAIN_NAMES.index(args_cli.terrain_type)
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
    falls = 0
    steps = 0
    speed_sum = 0.0
    follow_camera = _make_follow_camera(robot, args_cli.follow_distance, args_cli.follow_height)
    observation = env.get_observations()
    observation = observation[0] if isinstance(observation, tuple) else observation
    report_every = 50
    with torch.inference_mode():
        while steps * env.unwrapped.step_dt < args_cli.duration_s:
            observation, _, dones, _ = env.step(policy(observation))
            steps += 1
            falls += int(torch.sum(dones.long()).item())
            speed_sum += float(_as_torch(robot.data.root_lin_vel_b)[:, 0].mean().item())
            if follow_camera is not None and steps % 2 == 0 and not follow_camera():
                follow_camera = None  # headless run or viewport missing; stop trying
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
