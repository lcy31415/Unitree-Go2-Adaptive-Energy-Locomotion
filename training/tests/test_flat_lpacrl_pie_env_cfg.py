"""Configuration contracts for flat adaptive-energy LP-ACRL with PIE."""

from __future__ import annotations

import gymnasium as gym

import unitree_rl_lab.tasks.locomotion.robots.go2  # noqa: F401

from unitree_rl_lab.tasks.locomotion.agents.pie_cfg import AdaptiveEnergyFlatLPACRLPIERunnerCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.adaptive_energy_flat_lpacrl_env_cfg import (
    AdaptiveEnergyFlatLPACRLEnvCfg,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.adaptive_energy_flat_lpacrl_pie_env_cfg import (
    AdaptiveEnergyFlatLPACRLPIEEnvCfg,
    AdaptiveEnergyFlatLPACRLPIEPlayEnvCfg,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.pie_sensors_cfg import PIE_FOOT_SENSOR_NAMES


TASK_ID = "Unitree-Go2-Adaptive-Energy-Flat-LPACRL-PIE"


def test_flat_lpacrl_pie_task_registration():
    kwargs = gym.spec(TASK_ID).kwargs
    assert kwargs["env_cfg_entry_point"].endswith(":AdaptiveEnergyFlatLPACRLPIEEnvCfg")
    assert kwargs["play_env_cfg_entry_point"].endswith(":AdaptiveEnergyFlatLPACRLPIEPlayEnvCfg")
    assert kwargs["rsl_rl_cfg_entry_point"].endswith(":AdaptiveEnergyFlatLPACRLPIERunnerCfg")


def test_environment_preserves_flat_lpacrl_contract():
    pie = AdaptiveEnergyFlatLPACRLPIEEnvCfg()
    base = AdaptiveEnergyFlatLPACRLEnvCfg()

    assert pie.scene.num_envs == 256
    assert pie.scene.terrain.terrain_type == "plane"
    assert pie.scene.terrain.terrain_generator is None
    assert pie.commands.base_velocity.ranges == base.commands.base_velocity.ranges
    assert pie.curriculum.lp_acrl.func is base.curriculum.lp_acrl.func
    assert pie.curriculum.lp_acrl.params == base.curriculum.lp_acrl.params
    assert pie.curriculum.lp_acrl.params["episodes_per_stage"] is None
    assert "terrain_names" not in pie.curriculum.lp_acrl.params
    assert vars(pie.rewards).keys() == vars(base.rewards).keys()
    assert vars(pie.terminations).keys() == vars(base.terminations).keys()


def test_environment_adds_complete_pie_sensor_and_observation_interface():
    cfg = AdaptiveEnergyFlatLPACRLPIEEnvCfg()

    assert cfg.scene.height_scanner.pattern_cfg.size == (1.7, 1.0)
    assert cfg.scene.height_scanner.pattern_cfg.resolution == 0.1
    assert cfg.scene.pie_depth_camera.pattern_cfg.width == 106
    assert cfg.scene.pie_depth_camera.pattern_cfg.height == 60
    assert cfg.scene.pie_depth_camera.update_period == 0.1
    for sensor_name in PIE_FOOT_SENSOR_NAMES:
        assert getattr(cfg.scene, sensor_name).update_period == cfg.decimation * cfg.sim.dt

    assert cfg.observations.actor.history_length is None
    assert cfg.observations.proprio_history.history_length == 10
    assert cfg.observations.camera.depth_history.params["frame_history_length"] == 2
    assert cfg.observations.critic.height_scan is not None
    assert cfg.actions.JointPositionAction.preserve_order


def test_flat_pie_runner_uses_pieppo_and_independent_log_namespace():
    runner = AdaptiveEnergyFlatLPACRLPIERunnerCfg()

    assert runner.experiment_name == "unitree_go2_adaptive_energy_flat_lpacrl_pie"
    assert runner.actor.class_name.endswith(":PIEActorModel")
    assert runner.algorithm.class_name.endswith(":PIEPPO")
    assert runner.obs_groups["actor"] == ["actor", "proprio_history", "camera"]
    assert runner.actor.cnn_cfg["depth_shape"] == (2, 60, 86)


def test_play_environment_is_small_clean_and_does_not_mutate_training_cfg():
    play = AdaptiveEnergyFlatLPACRLPIEPlayEnvCfg()
    train = AdaptiveEnergyFlatLPACRLPIEEnvCfg()

    assert play.scene.num_envs == 8
    assert play.events.push_robot is None
    assert not play.observations.actor.enable_corruption
    assert not play.observations.proprio_history.enable_corruption
    assert train.scene.num_envs == 256
    assert train.events.push_robot is not None
    assert train.observations.actor.enable_corruption
