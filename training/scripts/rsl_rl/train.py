# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""


import gymnasium as gym
import pathlib
import sys

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

tasks = []
for task_spec in gym.registry.values():
    if "Unitree" in task_spec.id and "Isaac" not in task_spec.id:
        tasks.append(task_spec.id)

import argparse

import argcomplete

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, choices=tasks, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
argcomplete.autocomplete(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.resume and args_cli.pretrained_actor is not None:
    parser.error("--resume and --pretrained_actor are mutually exclusive.")
if args_cli.curriculum_checkpoint is not None and not args_cli.resume:
    parser.error("--curriculum_checkpoint requires --resume.")

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import csv
import inspect
import os
import shutil
import types
from datetime import datetime

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner  # TODO: Consider printing the experiment name in the terminal.

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


class _AdaptiveEnergyRewardLoggingWrapper(gym.Wrapper):
    """Add the normalized episodic total alongside decomposed reward logs."""

    _COMPONENT_KEYS = (
        "Episode_Reward/Rlin",
        "Episode_Reward/Rang",
        "Episode_Reward/Renergy",
        "Episode_Reward/adaptive_energy_residual",
    )

    def __init__(self, env, concise_lpacrl: bool = False):
        super().__init__(env)
        self.concise_lpacrl = concise_lpacrl

    def step(self, action):
        observations, reward, terminated, truncated, extras = self.env.step(action)
        log = extras.get("log")
        if log is not None and all(key in log for key in self._COMPONENT_KEYS):
            log["Episode_Reward/Rtotal"] = sum(log[key] for key in self._COMPONENT_KEYS)
        if log is not None and self.concise_lpacrl:
            keep = {
                "Episode_Reward/Rlin",
                "Episode_Reward/Rang",
                "Episode_Reward/Renergy",
                "Episode_Reward/Rtotal",
            }
            keep.update(key for key in log if key.startswith("Curriculum/lp_acrl/"))
            terrain_metrics = {
                "Curriculum/terrain_levels/mean_level",
                "Curriculum/terrain_levels/move_up_fraction",
                "Curriculum/terrain_levels/move_down_fraction",
                "Curriculum/terrain_levels/tracking_success",
                "Curriculum/terrain_levels/survival_rate",
            }
            keep.update(key for key in terrain_metrics if key in log)
            extras["log"] = {key: value for key, value in log.items() if key in keep}
        return observations, reward, terminated, truncated, extras


def _get_lp_acrl_term(env):
    """Return the stateful LP curriculum term, or None for all existing tasks."""
    manager = getattr(env.unwrapped, "curriculum_manager", None)
    if manager is None or "lp_acrl" not in manager.active_terms:
        return None
    index = manager.active_terms.index("lp_acrl")
    term = manager._term_cfgs[index].func
    return term if hasattr(term, "state_dict") and hasattr(term, "load_state_dict") else None


def _attach_lp_acrl_checkpoint_state(runner, curriculum_term) -> None:
    """Embed curriculum state in every normal RSL-RL checkpoint."""
    original_save = runner.save

    def save_with_curriculum(_runner, path: str, infos: dict | None = None):
        merged_infos = dict(infos or {})
        curriculum_state = curriculum_term.state_dict()
        merged_infos["lp_acrl_state"] = curriculum_state
        result = original_save(path, merged_infos)
        iteration = pathlib.Path(path).stem.removeprefix("model_")
        torch.save(curriculum_state, pathlib.Path(path).parent / f"lp_acrl_state_{iteration}.pt")
        snapshot_dir = pathlib.Path(path).parent / "curriculum"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_dir / f"{pathlib.Path(path).stem}.csv"
        with snapshot_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            if hasattr(curriculum_term, "csv_snapshot"):
                header, rows = curriculum_term.csv_snapshot()
                writer.writerow(header)
                writer.writerows(rows)
            else:
                probabilities = curriculum_term.sampler.probabilities.detach().cpu().tolist()
                progress = curriculum_term.sampler.learning_progress.detach().cpu().tolist()
                estimates = curriculum_term.sampler.reward_estimate.detach().cpu().tolist()
                ny, nw = curriculum_term.grid_shape[1:]
                writer.writerow(
                    ("task_id", "vx_bin", "vy_bin", "yaw_bin", "probability", "learning_progress", "reward_ema")
                )
                for task_id, (probability, lp, estimate) in enumerate(zip(probabilities, progress, estimates)):
                    writer.writerow(
                        (
                            task_id,
                            task_id // (ny * nw),
                            (task_id // nw) % ny,
                            task_id % nw,
                            probability,
                            lp,
                            estimate,
                        )
                    )
        return result

    runner.save = types.MethodType(save_with_curriculum, runner)


def _focus_kit_camera_on_visible_robot(env) -> None:
    """Lock the Kit camera to the origin of an environment it actually shows."""
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

    if not kit_visualizer_found:
        return
    # None means that every environment is visible. An empty list means the
    # user explicitly requested zero visible environments.
    if visible_env_ids == []:
        return
    env_index = 0 if visible_env_ids is None else visible_env_ids[0]

    camera_controller.set_view_env_index(env_index)
    # Use the stable environment origin instead of the robot-root tracking
    # callback. The latter can overwrite the viewport with a transient root
    # pose while GPU/Fabric state is refreshed or an environment is reset,
    # which makes the robot flash once and then leaves a blank viewport.
    camera_controller.update_view_to_env()
    camera_controller.update_view_location(eye=(3.0, -3.0, 2.0), lookat=(0.0, 0.0, 0.45))
    print(f"[INFO] Kit camera is locked on visible environment env_{env_index}.")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # Convert legacy RSL-RL policy configuration to the
    # actor/critic configuration required by newer RSL-RL.
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    _focus_kit_camera_on_visible_robot(env)
    lp_acrl_term = _get_lp_acrl_term(env)

    if hasattr(env.unwrapped, "reward_manager") and all(
        name in env.unwrapped.reward_manager.active_terms for name in ("Rlin", "Rang", "Renergy")
    ):
        env = _AdaptiveEnergyRewardLoggingWrapper(env, concise_lpacrl=lp_acrl_term is not None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        explicit_checkpoint = None
        if args_cli.checkpoint is not None:
            candidate = os.path.abspath(os.path.expanduser(args_cli.checkpoint))
            if os.path.isfile(candidate):
                explicit_checkpoint = candidate
        resume_path = (
            explicit_checkpoint
            if explicit_checkpoint is not None
            else get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        )

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if lp_acrl_term is not None:
        _attach_lp_acrl_checkpoint_state(runner, lp_acrl_term)
        print(
            f"[INFO] LP-ACRL: {lp_acrl_term.num_tasks} tasks, "
            f"{lp_acrl_term.episodes_per_stage} completed episodes/stage, checkpoint state enabled."
        )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        checkpoint_infos = runner.load(resume_path)
        if lp_acrl_term is not None:
            curriculum_state = None
            if args_cli.curriculum_checkpoint is not None:
                curriculum_path = os.path.abspath(os.path.expanduser(args_cli.curriculum_checkpoint))
                if not os.path.isfile(curriculum_path):
                    raise FileNotFoundError(f"LP-ACRL curriculum checkpoint does not exist: {curriculum_path}")
                curriculum_state = torch.load(curriculum_path, weights_only=False, map_location="cpu")
                print(f"[INFO] Loading standalone LP-ACRL state from: {curriculum_path}")
            elif isinstance(checkpoint_infos, dict):
                curriculum_state = checkpoint_infos.get("lp_acrl_state")
            if curriculum_state is not None:
                lp_acrl_term.load_state_dict(curriculum_state)
                lp_acrl_term.resample_current_episodes()
                print(f"[INFO] Restored LP-ACRL state at stage {lp_acrl_term.stage}.")
            else:
                print("[WARNING] Checkpoint has no LP-ACRL state; curriculum restarts uniformly.")
    elif args_cli.pretrained_actor is not None:
        pretrained_actor_path = os.path.abspath(os.path.expanduser(args_cli.pretrained_actor))
        if not os.path.isfile(pretrained_actor_path):
            raise FileNotFoundError(f"Pretrained actor checkpoint does not exist: {pretrained_actor_path}")
        print(f"[INFO]: Warm-starting actor only from: {pretrained_actor_path}")
        runner.load(
            pretrained_actor_path,
            load_cfg={
                "actor": True,
                "critic": False,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            },
            map_location=agent_cfg.device,
        )

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    export_deploy_cfg(env.unwrapped, log_dir)
    # copy the environment configuration file to the log directory
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    # run training
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        # Random initial truncation would attribute partial random episodes to
        # LP tasks and corrupt the first curriculum stage.
        init_at_random_ep_len=lp_acrl_term is None,
    )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
