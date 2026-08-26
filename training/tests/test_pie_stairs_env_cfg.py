"""Contracts for the focused, non-LP-ACRL PIE stair task."""

from __future__ import annotations

import math
from types import SimpleNamespace

import gymnasium as gym
import torch

import isaaclab.terrains as terrain_gen
import unitree_rl_lab.tasks.locomotion.robots.go2  # noqa: F401
from isaaclab.envs.mdp import UniformVelocityCommandCfg

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.agents.pie_cfg import AdaptiveEnergyPIEStairsRunnerCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.adaptive_energy_pie_stairs_env_cfg import (
    PIE_STAIRS_COLUMNS_PER_TYPE,
    PIE_STAIRS_FINISH_DISTANCE,
    PIE_STAIRS_TERRAIN_NAMES,
    AdaptiveEnergyPIEStairsEnvCfg,
    AdaptiveEnergyPIEStairsPlayEnvCfg,
)


TASK_ID = "Unitree-Go2-PIE-Stairs"


def test_task_registration_points_to_stair_pie_components():
    kwargs = gym.spec(TASK_ID).kwargs
    assert kwargs["env_cfg_entry_point"].endswith(":AdaptiveEnergyPIEStairsEnvCfg")
    assert kwargs["play_env_cfg_entry_point"].endswith(":AdaptiveEnergyPIEStairsPlayEnvCfg")
    assert kwargs["rsl_rl_cfg_entry_point"].endswith(":AdaptiveEnergyPIEStairsRunnerCfg")


def test_stair_task_has_no_lpacrl_state_or_curriculum():
    cfg = AdaptiveEnergyPIEStairsEnvCfg()
    command = cfg.commands.base_velocity

    assert cfg.curriculum is None
    assert isinstance(command, UniformVelocityCommandCfg)
    assert command.ranges.lin_vel_x == (0.5, 1.5)
    assert command.ranges.lin_vel_y == (0.0, 0.0)
    assert command.ranges.ang_vel_z == (0.0, 0.0)
    assert command.resampling_time_range == (1.0e9, 1.0e9)
    assert not hasattr(command, "task_ids")
    assert not hasattr(cfg.rewards, "standstill_penalty")
    assert not hasattr(cfg.terminations, "bad_standstill")


def test_stair_geometry_is_directed_and_uniformly_partitioned():
    cfg = AdaptiveEnergyPIEStairsEnvCfg()
    generator = cfg.scene.terrain.terrain_generator

    assert tuple(generator.sub_terrains) == PIE_STAIRS_TERRAIN_NAMES
    assert generator.num_rows == 4
    assert generator.num_cols == len(PIE_STAIRS_TERRAIN_NAMES) * PIE_STAIRS_COLUMNS_PER_TYPE
    assert generator.class_type.__name__ == "DiscreteLevelTerrainGenerator"
    assert isinstance(generator.sub_terrains["stairs_up"], terrain_gen.MeshInvertedPyramidStairsTerrainCfg)
    assert isinstance(generator.sub_terrains["stairs_down"], terrain_gen.MeshPyramidStairsTerrainCfg)
    assert generator.sub_terrains["stairs_up"].proportion == 0.5
    assert generator.sub_terrains["stairs_down"].proportion == 0.5
    assert cfg.scene.terrain.max_init_terrain_level == 3


def test_stair_reset_and_completion_contract():
    cfg = AdaptiveEnergyPIEStairsEnvCfg()
    pose = cfg.events.reset_base.params["pose_range"]

    assert pose["x"] == (-0.15, 0.15)
    assert pose["y"] == (-0.15, 0.15)
    assert pose["yaw"] == (-math.radians(5.0), math.radians(5.0))
    assert cfg.events.push_robot is None
    assert cfg.episode_length_s == 12.0
    assert cfg.terminations.course_complete.func is mdp.stair_course_complete
    assert cfg.terminations.course_complete.time_out
    assert cfg.terminations.course_complete.params["finish_distance"] == PIE_STAIRS_FINISH_DISTANCE
    assert not cfg.terminations.lateral_deviation.time_out
    assert not cfg.terminations.heading_error.time_out
    assert cfg.rewards.course_completion.func is mdp.stair_course_completion_reward


def test_stair_task_preserves_pie_observations_and_runner():
    cfg = AdaptiveEnergyPIEStairsEnvCfg()
    runner = AdaptiveEnergyPIEStairsRunnerCfg()

    assert cfg.scene.pie_depth_camera.pattern_cfg.width == 106
    assert cfg.observations.camera.depth_history.params["frame_history_length"] == 2
    assert cfg.observations.proprio_history.history_length == 10
    assert runner.experiment_name == "unitree_go2_adaptive_energy_pie_stairs"
    assert runner.max_iterations == 20_000
    assert runner.actor.cnn_cfg["depth_shape"] == (2, 60, 86)
    assert runner.algorithm.class_name.endswith(":PIEPPO")


def test_play_task_is_small_and_has_clean_policy_observations():
    cfg = AdaptiveEnergyPIEStairsPlayEnvCfg()

    assert cfg.scene.num_envs == 8
    assert not cfg.observations.actor.enable_corruption
    assert not cfg.observations.proprio_history.enable_corruption


class _FakeScene(dict):
    pass


def _fake_env(root_pos: torch.Tensor, root_quat: torch.Tensor):
    robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=root_pos, root_quat_w=root_quat))
    scene = _FakeScene(robot=robot)
    scene.terrain = SimpleNamespace(env_origins=torch.zeros_like(root_pos))
    return SimpleNamespace(scene=scene)


def test_stair_mdp_terms_distinguish_success_lateral_and_heading_failures():
    root_pos = torch.tensor([[3.4, 0.0, 0.0], [1.0, 1.2, 0.0], [1.0, 0.0, 0.0]])
    angle = math.radians(45.0)
    root_quat = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)],
        ]
    )
    env = _fake_env(root_pos, root_quat)

    assert mdp.stair_course_complete(env, 3.3).tolist() == [True, False, False]
    assert mdp.stair_course_completion_reward(env, 3.3).tolist() == [1.0, 0.0, 0.0]
    assert mdp.stair_lateral_deviation(env, 1.0).tolist() == [False, True, False]
    assert mdp.stair_heading_error(env, math.radians(30.0)).tolist() == [False, False, True]
