# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--command_vx", type=float, default=None, help="Fix the commanded forward velocity in m/s.")
parser.add_argument("--command_vy", type=float, default=None, help="Fix the commanded lateral velocity in m/s.")
parser.add_argument("--command_yaw", type=float, default=None, help="Fix the commanded yaw rate in rad/s.")
parser.add_argument(
    "--terrain_type",
    type=str,
    default=None,
    help=(
        "Fix a generated-terrain family for a one-environment demonstration. "
        "For the terrain LP-ACRL task: flat, stairs_up, stairs_down, slope_up, "
        "slope_down, or random_rough."
    ),
)
parser.add_argument(
    "--terrain_level",
    type=int,
    default=None,
    help="Fix the zero-based terrain difficulty level (the LP-ACRL task supports 0--3).",
)
parser.add_argument(
    "--terrain_variant",
    type=int,
    default=0,
    help="Select a zero-based column variant within the requested terrain family (default: 0).",
)
parser.add_argument(
    "--follow_robot",
    action="store_true",
    default=False,
    help="Continuously follow the robot root with the Kit viewport camera.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
fixed_command_args = (args_cli.command_vx, args_cli.command_vy, args_cli.command_yaw)
if any(value is not None for value in fixed_command_args) and not all(
    value is not None for value in fixed_command_args
):
    parser.error("--command_vx, --command_vy, and --command_yaw must be provided together.")
if args_cli.terrain_level is not None and args_cli.terrain_type is None:
    parser.error("--terrain_level requires --terrain_type.")
if args_cli.terrain_variant != 0 and args_cli.terrain_type is None:
    parser.error("--terrain_variant requires --terrain_type.")
if args_cli.terrain_type is not None:
    if not all(value is not None for value in fixed_command_args):
        parser.error("A fixed terrain demonstration also requires all three --command_* arguments.")
    if args_cli.num_envs is None:
        args_cli.num_envs = 1
    elif args_cli.num_envs != 1:
        parser.error("A fixed terrain demonstration currently requires --num_envs 1.")
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
from types import MethodType

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def _focus_kit_camera_on_visible_robot(env, follow_robot: bool = False) -> None:
    """Point the Kit viewport at a visible environment or follow its robot."""
    unwrapped_env = env.unwrapped
    camera_controller = getattr(unwrapped_env, "viewport_camera_controller", None)
    if camera_controller is None or "robot" not in unwrapped_env.scene.articulations:
        return

    visible_env_ids = None
    kit_visualizer_found = False
    for visualizer in unwrapped_env.sim.visualizers:
        if getattr(visualizer.cfg, "visualizer_type", None) == "kit":
            kit_visualizer_found = True
            visible_env_ids = visualizer.get_visualized_env_ids()
            break

    if not kit_visualizer_found or visible_env_ids == []:
        return
    env_index = 0 if visible_env_ids is None else visible_env_ids[0]
    camera_controller.set_view_env_index(env_index)
    if follow_robot:
        # ``asset_root`` registers a post-render callback in Isaac Lab's
        # ViewportCameraController, so this offset moves with the robot while
        # remaining aligned with the world axes.
        camera_controller.update_view_to_asset_root("robot")
    else:
        camera_controller.update_view_to_env()
    camera_controller.update_view_location(eye=(3.0, -3.0, 2.0), lookat=(0.0, 0.0, 0.45))
    target = "robot root" if follow_robot else f"visible environment env_{env_index}"
    print(f"[INFO] Kit camera is locked on {target}.")


def _install_fixed_velocity_command(env, command: tuple[float, float, float]) -> None:
    """Keep one velocity command across command resampling and episode resets."""
    command_term = env.unwrapped.command_manager.get_term("base_velocity")
    fixed_command = torch.tensor(command, dtype=torch.float, device=command_term.device)

    def _resample_fixed(self, env_ids):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.vel_command_b[env_ids] = fixed_command
        self.is_standing_env[env_ids] = False
        # Clear curriculum bookkeeping when the adaptive-energy command term
        # is used. Evaluation must not update or sample its weighted bins.
        if hasattr(self, "bin_ids"):
            self.bin_ids[env_ids] = -1
        if hasattr(self, "_linear_reward_sum"):
            self._linear_reward_sum[env_ids] = 0.0
            self._angular_reward_sum[env_ids] = 0.0
            self._segment_steps[env_ids] = 0

    command_term._resample_command = MethodType(_resample_fixed, command_term)
    command_term._resample_command(torch.arange(env.unwrapped.num_envs, device=command_term.device))
    print(
        "[INFO] Fixed velocity command: "
        f"vx={command[0]:.3f} m/s, vy={command[1]:.3f} m/s, yaw={command[2]:.3f} rad/s."
    )


def _terrain_columns_by_name(terrain_generator_cfg) -> dict[str, list[int]]:
    """Reproduce TerrainGenerator's proportional terrain-family-to-column mapping."""
    sub_terrains = terrain_generator_cfg.sub_terrains
    if not sub_terrains:
        return {}

    names = list(sub_terrains)
    proportions = [float(sub_terrains[name].proportion) for name in names]
    total = sum(proportions)
    if total <= 0.0:
        raise ValueError("Generated terrain proportions must sum to a positive value.")

    cumulative = []
    running = 0.0
    for proportion in proportions:
        running += proportion / total
        cumulative.append(running)

    columns = {name: [] for name in names}
    for column in range(terrain_generator_cfg.num_cols):
        sample = column / terrain_generator_cfg.num_cols + 0.001
        family_index = next(
            (index for index, threshold in enumerate(cumulative) if sample < threshold),
            len(names) - 1,
        )
        columns[names[family_index]].append(column)
    return columns


def _install_fixed_terrain(env, terrain_type: str, terrain_level: int | None, terrain_variant: int) -> None:
    """Place every reset on one generated terrain family, level, and column variant."""
    unwrapped_env = env.unwrapped
    terrain = getattr(unwrapped_env.scene, "terrain", None)
    generator_cfg = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    terrain_origins = getattr(terrain, "terrain_origins", None)
    if terrain is None or generator_cfg is None or terrain_origins is None:
        raise ValueError(
            f"Task {args_cli.task!r} does not use a generated terrain with selectable origins."
        )

    columns_by_name = _terrain_columns_by_name(generator_cfg)
    if terrain_type not in columns_by_name:
        available = ", ".join(columns_by_name)
        raise ValueError(f"Unknown terrain type {terrain_type!r}. Available terrain types: {available}.")

    level = 0 if terrain_level is None else terrain_level
    num_levels = int(terrain_origins.shape[0])
    if not 0 <= level < num_levels:
        raise ValueError(f"Terrain level must be in [0, {num_levels - 1}], got {level}.")

    family_columns = columns_by_name[terrain_type]
    if not 0 <= terrain_variant < len(family_columns):
        raise ValueError(
            f"Terrain {terrain_type!r} has {len(family_columns)} column variants; "
            f"--terrain_variant must be in [0, {len(family_columns) - 1}]."
        )
    column = family_columns[terrain_variant]

    env_ids = torch.arange(unwrapped_env.num_envs, dtype=torch.long, device=unwrapped_env.device)
    terrain.terrain_levels[env_ids] = level
    terrain.terrain_types[env_ids] = column
    terrain.env_origins[env_ids] = terrain.terrain_origins[level, column]
    print(
        "[INFO] Fixed terrain: "
        f"type={terrain_type}, level={level}, variant={terrain_variant}, generator_column={column}."
    )


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    # The task still stores the legacy ``policy`` configuration used to train
    # this checkpoint. RSL-RL 2.3+ expects explicit actor/critic entries with
    # class names, so apply the same compatibility conversion as train.py.
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))

    # A fixed-terrain demonstration must not let the reset-driven terrain or
    # LP-ACRL curriculum overwrite the selected terrain origin. The fixed
    # command hook installed below supplies commands to LPACRLVelocityCommand.
    if args_cli.terrain_type is not None:
        env_cfg.curriculum = None

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if all(value is not None for value in fixed_command_args):
        _install_fixed_velocity_command(env, fixed_command_args)
    if args_cli.terrain_type is not None:
        _install_fixed_terrain(
            env,
            terrain_type=args_cli.terrain_type,
            terrain_level=args_cli.terrain_level,
            terrain_variant=args_cli.terrain_variant,
        )
        # Move the robot to the selected terrain origin immediately. Future
        # episode resets keep using this origin because curriculum is disabled.
        env.reset()
    _focus_kit_camera_on_visible_robot(env, follow_robot=args_cli.follow_robot)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if hasattr(runner, "export_policy_to_jit") and hasattr(runner, "export_policy_to_onnx"):
        # RSL-RL 4.x owns export on the runner and stores the model at
        # ``runner.alg.actor``. This path also preserves its observation
        # normalizer without relying on the removed policy/actor_critic API.
        runner.export_policy_to_jit(export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(export_model_dir, filename="policy.onnx")
    else:
        # Compatibility with older RSL-RL releases used by upstream Isaac Lab.
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic

        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
