"""Configuration-contract tests for the fused LP-ACRL PIE task."""

from __future__ import annotations

import gymnasium as gym

import unitree_rl_lab.tasks.locomotion.robots.go2  # noqa: F401
from unitree_rl_lab.tasks.locomotion.robots.go2.adaptive_energy_lpacrl_pie_env_cfg import (
    PIE_FOOT_SENSOR_NAMES,
    PIE_JOINT_NAMES,
    AdaptiveEnergyLPACRLPIEEnvCfg,
    AdaptiveEnergyLPACRLPIEPlayEnvCfg,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.adaptive_energy_terrain_lpacrl_env_cfg import (
    AdaptiveEnergyTerrainLPACRLEnvCfg,
)


TASK_ID = "Unitree-Go2-Adaptive-Energy-LPACRL-PIE"


def _term_names(group) -> list[str]:
    return [name for name, value in vars(group).items() if hasattr(value, "func")]


def test_task_registration_points_to_pie_components():
    kwargs = gym.spec(TASK_ID).kwargs
    assert kwargs["env_cfg_entry_point"].endswith(":AdaptiveEnergyLPACRLPIEEnvCfg")
    assert kwargs["play_env_cfg_entry_point"].endswith(":AdaptiveEnergyLPACRLPIEPlayEnvCfg")
    assert kwargs["rsl_rl_cfg_entry_point"].endswith(":AdaptiveEnergyLPACRLPIERunnerCfg")


def test_fused_environment_preserves_lpacrl_and_adds_pie_sensors():
    cfg = AdaptiveEnergyLPACRLPIEEnvCfg()

    assert cfg.scene.num_envs == 256
    assert cfg.curriculum.lp_acrl.params["terrain_names"]
    assert cfg.curriculum.lp_acrl.params["episodes_per_stage"] == 4096
    assert cfg.commands.base_velocity.ranges.lin_vel_x == (-2.5, 2.5)
    assert cfg.rewards.standstill_penalty is not None
    assert cfg.terminations.bad_standstill is not None

    assert cfg.scene.height_scanner.pattern_cfg.size == (1.7, 1.0)
    assert cfg.scene.height_scanner.pattern_cfg.resolution == 0.1
    assert cfg.scene.pie_depth_camera.pattern_cfg.width == 106
    assert cfg.scene.pie_depth_camera.pattern_cfg.height == 60
    assert cfg.scene.pie_depth_camera.update_period == 0.1
    assert cfg.scene.height_scanner.update_period == cfg.decimation * cfg.sim.dt
    for name in PIE_FOOT_SENSOR_NAMES:
        sensor = getattr(cfg.scene, name)
        assert sensor.pattern_cfg.size == (0.02, 0.02)
        assert sensor.update_period == cfg.decimation * cfg.sim.dt


def test_pie_observation_groups_and_dimensions_are_explicit():
    cfg = AdaptiveEnergyLPACRLPIEEnvCfg()
    observations = cfg.observations

    assert list(vars(observations)) == [
        "actor",
        "proprio_history",
        "camera",
        "critic",
        "velocity_target",
        "height_target",
        "foot_clearance_target",
        "successor_target",
        "successor_valid",
    ]
    proprio_terms = [
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos_rel",
        "joint_vel_rel",
        "last_action",
    ]
    assert _term_names(observations.actor) == proprio_terms
    assert _term_names(observations.proprio_history) == proprio_terms
    assert _term_names(observations.successor_target) == proprio_terms
    assert observations.actor.history_length is None
    assert observations.proprio_history.history_length == 10
    assert observations.proprio_history.flatten_history_dim
    assert observations.camera.depth_history.params["frame_history_length"] == 2

    # 45 current, 10x45 history, 2x60x86 depth, 45+3+198 critic.
    expected_dimensions = {
        "actor": 45,
        "proprio_history": 450,
        "camera": 10_320,
        "critic": 246,
        "velocity_target": 3,
        "height_target": 198,
        "foot_clearance_target": 4,
        "successor_target": 45,
        "successor_valid": 1,
    }
    assert expected_dimensions == {
        "actor": 3 + 3 + 3 + 12 + 12 + 12,
        "proprio_history": 10 * (3 + 3 + 3 + 12 + 12 + 12),
        "camera": 2 * 60 * (106 - 10 - 10),
        "critic": 45 + 3 + 18 * 11,
        "velocity_target": 3,
        "height_target": 18 * 11,
        "foot_clearance_target": len(PIE_FOOT_SENSOR_NAMES),
        "successor_target": 45,
        "successor_valid": 1,
    }


def test_joint_action_and_observations_share_checkpoint_order():
    cfg = AdaptiveEnergyLPACRLPIEEnvCfg()
    expected = list(PIE_JOINT_NAMES)

    assert cfg.actions.JointPositionAction.joint_names == expected
    assert cfg.actions.JointPositionAction.preserve_order
    for group_name in ("actor", "proprio_history", "critic", "successor_target"):
        group = getattr(cfg.observations, group_name)
        for term_name in ("joint_pos_rel", "joint_vel_rel"):
            asset_cfg = getattr(group, term_name).params["asset_cfg"]
            assert asset_cfg.joint_names == expected
            assert asset_cfg.preserve_order


def test_play_is_small_clean_and_does_not_mutate_existing_task():
    play = AdaptiveEnergyLPACRLPIEPlayEnvCfg()
    original = AdaptiveEnergyTerrainLPACRLEnvCfg()

    assert play.scene.num_envs == 8
    assert play.events.push_robot is None
    assert play.rewards.standstill_penalty is None
    assert play.terminations.bad_standstill is None
    assert not play.observations.actor.enable_corruption
    assert not play.observations.proprio_history.enable_corruption

    assert original.scene.num_envs == 2048
    assert original.scene.height_scanner.pattern_cfg.size == (1.6, 1.0)
    assert original.observations.policy.history_length == 30
    assert original.rewards.standstill_penalty is not None
