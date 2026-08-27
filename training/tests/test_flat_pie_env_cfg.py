"""Contracts for PIE on the original adaptive-energy flat curriculum."""

from __future__ import annotations

import gymnasium as gym

import unitree_rl_lab.tasks.locomotion.robots.go2  # noqa: F401

from unitree_rl_lab.tasks.locomotion.agents.pie_cfg import AdaptiveEnergyFlatPIERunnerCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.adaptive_energy_env_cfg import AdaptiveEnergyEnvCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.adaptive_energy_flat_pie_env_cfg import (
    AdaptiveEnergyFlatPIEEnvCfg,
    AdaptiveEnergyFlatPIEPlayEnvCfg,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.pie_sensors_cfg import PIE_FOOT_SENSOR_NAMES


TASK_ID = "Unitree-Go2-Adaptive-Energy-Flat-PIE"


def test_flat_pie_task_registration():
    kwargs = gym.spec(TASK_ID).kwargs
    assert kwargs["env_cfg_entry_point"].endswith(":AdaptiveEnergyFlatPIEEnvCfg")
    assert kwargs["play_env_cfg_entry_point"].endswith(":AdaptiveEnergyFlatPIEPlayEnvCfg")
    assert kwargs["rsl_rl_cfg_entry_point"].endswith(":AdaptiveEnergyFlatPIERunnerCfg")


def test_flat_pie_preserves_original_velocity_curriculum_and_objective():
    pie = AdaptiveEnergyFlatPIEEnvCfg()
    original = AdaptiveEnergyEnvCfg()

    assert pie.scene.num_envs == 256
    assert pie.scene.terrain.terrain_type == "plane"
    assert pie.commands.base_velocity.class_type is original.commands.base_velocity.class_type
    assert pie.commands.base_velocity.ranges == original.commands.base_velocity.ranges
    assert pie.commands.base_velocity.limit_ranges == original.commands.base_velocity.limit_ranges
    assert pie.commands.base_velocity.num_bins == original.commands.base_velocity.num_bins
    assert pie.commands.base_velocity.linear_stage_threshold == 2.5
    assert pie.commands.base_velocity.angular_stage_threshold == 2.5
    assert pie.curriculum is None
    assert vars(pie.rewards).keys() == vars(original.rewards).keys()
    assert "straight_yaw_rate_error" not in vars(pie.rewards)


def test_flat_pie_adds_complete_perception_interface():
    cfg = AdaptiveEnergyFlatPIEEnvCfg()

    assert cfg.scene.pie_depth_camera.pattern_cfg.width == 106
    assert cfg.scene.pie_depth_camera.pattern_cfg.height == 60
    assert cfg.scene.pie_depth_camera.update_period == 0.1
    assert cfg.scene.height_scanner.pattern_cfg.size == (1.7, 1.0)
    for sensor_name in PIE_FOOT_SENSOR_NAMES:
        assert getattr(cfg.scene, sensor_name).update_period == cfg.decimation * cfg.sim.dt
    assert cfg.observations.proprio_history.history_length == 10
    assert cfg.observations.camera.depth_history.params["frame_history_length"] == 2
    assert cfg.actions.JointPositionAction.preserve_order


def test_flat_pie_runner_has_independent_namespace():
    runner = AdaptiveEnergyFlatPIERunnerCfg()

    assert runner.experiment_name == "unitree_go2_adaptive_energy_flat_pie"
    assert runner.actor.class_name.endswith(":PIEActorModel")
    assert runner.algorithm.class_name.endswith(":PIEPPO")


def test_flat_pie_play_cfg_is_small_and_does_not_mutate_training_cfg():
    play = AdaptiveEnergyFlatPIEPlayEnvCfg()
    train = AdaptiveEnergyFlatPIEEnvCfg()

    assert play.scene.num_envs == 8
    assert play.commands.base_velocity.ranges.lin_vel_x == (0.0, 5.0)
    assert play.events.push_robot is None
    assert not play.observations.actor.enable_corruption
    assert not play.observations.proprio_history.enable_corruption
    assert train.scene.num_envs == 256
    assert train.commands.base_velocity.ranges.lin_vel_x == (-1.0, 1.0)
    assert train.events.push_robot is not None
