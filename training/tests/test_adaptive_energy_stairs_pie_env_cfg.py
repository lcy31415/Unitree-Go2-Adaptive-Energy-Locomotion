"""Contracts for stair-only PIE with coupled terrain and velocity curricula."""

from __future__ import annotations

import math

import gymnasium as gym
import torch

import unitree_rl_lab.tasks.locomotion.robots.go2  # noqa: F401

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp.commands.reward_threshold_velocity_command import (
    RewardThresholdCurriculum,
)
from unitree_rl_lab.tasks.locomotion.agents.pie_cfg import AdaptiveEnergyStairsPIERunnerCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.adaptive_energy_pie_stairs_env_cfg import (
    PIE_STAIRS_TERRAIN_NAMES,
    AdaptiveEnergyPIEStairsEnvCfg,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.adaptive_energy_stairs_pie_env_cfg import (
    STAIRS_PIE_NUM_LEVELS,
    STAIRS_PIE_MAX_ABS_VX_BY_LEVEL,
    STAIRS_PIE_STEP_HEIGHT_RANGE,
    STAIRS_PIE_VELOCITY_GATE_CLOSE_MEAN_LEVEL,
    STAIRS_PIE_VELOCITY_GATE_MIN_MEAN_LEVEL,
    STAIRS_PIE_VELOCITY_UPDATE_LEVEL_MARGIN,
    STAIRS_PIE_VELOCITY_UPDATE_MAX_LEVEL,
    AdaptiveEnergyStairsPIEEnvCfg,
    AdaptiveEnergyStairsPIEPlayEnvCfg,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.adaptive_energy_terrain_env_cfg import (
    AdaptiveEnergyTerrainEnvCfg,
)


TASK_ID = "Unitree-Go2-Adaptive-Energy-stairs-PIE"


def test_velocity_sampler_respects_terrain_speed_envelope_without_replay():
    curriculum = RewardThresholdCurriculum(
        "cpu",
        ((-2.5, 2.5), (-0.6, 0.6), (-2.5, 2.5)),
        (21, 13, 21),
        ((-1.0, 1.0), (0.0, 0.0), (0.0, 0.0)),
    )
    speed_limits = torch.tensor([0.8] * 64 + [1.5] * 64)
    commands, _, source_ids = curriculum.sample(
        128,
        stage=0,
        sampling_probabilities=(0.4, 0.6, 0.0),
        frontier_bin_count=2,
        max_abs_vx=speed_limits,
    )

    assert torch.all(commands[:, 0].abs() <= speed_limits + 1.0e-6)
    assert torch.all(commands[:, 1:] == 0.0)
    assert not torch.any(source_ids == 2)


def test_adaptive_energy_stairs_pie_registration():
    kwargs = gym.spec(TASK_ID).kwargs
    assert kwargs["env_cfg_entry_point"].endswith(":AdaptiveEnergyStairsPIEEnvCfg")
    assert kwargs["play_env_cfg_entry_point"].endswith(":AdaptiveEnergyStairsPIEPlayEnvCfg")
    assert kwargs["rsl_rl_cfg_entry_point"].endswith(":AdaptiveEnergyStairsPIERunnerCfg")


def test_stair_geometry_uses_ten_levels_and_matches_reference_shape():
    cfg = AdaptiveEnergyStairsPIEEnvCfg()
    reference = AdaptiveEnergyPIEStairsEnvCfg()
    generator = cfg.scene.terrain.terrain_generator
    reference_generator = reference.scene.terrain.terrain_generator

    assert tuple(generator.sub_terrains) == PIE_STAIRS_TERRAIN_NAMES
    assert generator.num_rows == STAIRS_PIE_NUM_LEVELS == 10
    assert reference_generator.num_rows == 4
    assert generator.num_cols == reference_generator.num_cols == 8
    assert generator.class_type is reference_generator.class_type
    for terrain in generator.sub_terrains.values():
        assert terrain.step_height_range == STAIRS_PIE_STEP_HEIGHT_RANGE == (0.03, 0.15)
    for terrain in reference_generator.sub_terrains.values():
        assert terrain.step_height_range == (0.05, 0.15)
    assert cfg.scene.terrain.max_init_terrain_level == 0


def test_original_velocity_curriculum_is_retained_and_terrain_gated():
    cfg = AdaptiveEnergyStairsPIEEnvCfg()
    original = AdaptiveEnergyTerrainEnvCfg()
    command = cfg.commands.base_velocity

    assert command.class_type is original.commands.base_velocity.class_type
    assert command.ranges == original.commands.base_velocity.ranges
    assert command.limit_ranges.lin_vel_x == (-2.5, 2.5)
    assert command.limit_ranges.lin_vel_y == (-0.6, 0.6)
    assert command.limit_ranges.ang_vel_z == (-2.5, 2.5)
    assert command.linear_stage_threshold == 1.25
    assert command.angular_stage_threshold == 1.25
    assert command.forward_error_abs == 0.18
    assert command.angular_error_abs == 0.20
    assert (
        command.terrain_gate_min_mean_level
        == STAIRS_PIE_VELOCITY_GATE_MIN_MEAN_LEVEL
        == 2.0
    )
    assert (
        command.terrain_gate_close_mean_level
        == STAIRS_PIE_VELOCITY_GATE_CLOSE_MEAN_LEVEL
        == 1.5
    )
    assert command.terrain_level_ema_decay == 0.98
    assert (
        command.curriculum_update_level_margin
        == STAIRS_PIE_VELOCITY_UPDATE_LEVEL_MARGIN
        == 2
    )
    assert command.curriculum_update_max_terrain_level == STAIRS_PIE_VELOCITY_UPDATE_MAX_LEVEL == 9
    assert command.terrain_conditioned_max_abs_vx == STAIRS_PIE_MAX_ABS_VX_BY_LEVEL
    assert command.frontier_sampling_probability == 0.40
    assert command.active_sampling_probability == 0.60
    assert command.replay_sampling_probability == 0.0
    assert command.yaw_recovery_sampling_probability == 0.15
    assert not hasattr(command, "task_ids")


def test_terrain_curriculum_uses_reference_traversal_distance():
    cfg = AdaptiveEnergyStairsPIEEnvCfg()
    term = cfg.curriculum.terrain_levels

    assert term.func is mdp.adaptive_energy_terrain_levels
    assert term.params["minimum_tracking_fraction"] == 0.7
    assert term.params["minimum_tracking_fraction_for_hold"] == 0.45
    assert term.params["move_up_distance_fraction"] == 0.4
    assert term.params["move_down_expected_fraction"] == 0.5


def test_stair_curriculum_adds_anti_collapse_constraints():
    cfg = AdaptiveEnergyStairsPIEEnvCfg()
    original = AdaptiveEnergyTerrainEnvCfg()

    assert vars(original.rewards).keys() <= vars(cfg.rewards).keys()
    assert vars(original.terminations).keys() <= vars(cfg.terminations).keys()
    assert not hasattr(cfg.rewards, "course_completion")
    assert not hasattr(cfg.terminations, "course_complete")
    assert cfg.rewards.command_stagnation.func is mdp.command_stagnation_penalty
    assert cfg.rewards.command_stagnation.params["min_command"] == 0.3
    assert cfg.rewards.straight_yaw_rate_error.func is mdp.straight_command_yaw_rate_error
    assert cfg.rewards.straight_yaw_rate_error.weight == -0.25
    assert cfg.terminations.low_base_clearance.func is mdp.low_base_clearance
    assert cfg.terminations.low_base_clearance.params["minimum_clearance"] == 0.20
    assert cfg.terminations.persistent_body_contact.func is mdp.persistent_body_contact
    assert cfg.terminations.persistent_body_contact.params["sustain_s"] == 0.20
    assert cfg.terminations.persistent_body_contact.params["sensor_cfg"].body_names == [
        ".*_hip",
        ".*_thigh",
    ]
    assert cfg.terminations.bad_command_stagnation.func is mdp.bad_command_stagnation
    assert cfg.terminations.bad_command_stagnation.params["sustain_s"] == 3.0
    assert cfg.episode_length_s == 20.0
    assert cfg.events.push_robot is not None


def test_stair_reset_and_pie_interface():
    cfg = AdaptiveEnergyStairsPIEEnvCfg()
    pose = cfg.events.reset_base.params["pose_range"]

    assert pose["x"] == (-0.15, 0.15)
    assert pose["y"] == (-0.15, 0.15)
    assert pose["yaw"] == (-math.radians(5.0), math.radians(5.0))
    assert cfg.scene.pie_depth_camera.pattern_cfg.width == 106
    assert cfg.scene.pie_depth_camera.update_period == 0.1
    assert cfg.scene.pie_base_clearance_scanner.update_period == 0.02
    assert cfg.observations.camera.depth_history.params["frame_history_length"] == 2
    assert cfg.observations.proprio_history.history_length == 10
    assert cfg.actions.JointPositionAction.preserve_order


def test_runner_and_play_contracts():
    runner = AdaptiveEnergyStairsPIERunnerCfg()
    play = AdaptiveEnergyStairsPIEPlayEnvCfg()

    assert runner.experiment_name == "unitree_go2_adaptive_energy_stairs_pie"
    assert runner.max_iterations == 20_000
    assert runner.save_interval == 200
    assert runner.algorithm.class_name.endswith(":PIEPPO")
    assert play.scene.num_envs == 8
    assert play.curriculum is None
    assert play.events.push_robot is None
    assert not play.observations.actor.enable_corruption
