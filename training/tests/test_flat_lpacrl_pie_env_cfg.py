"""Configuration contracts for flat adaptive-energy LP-ACRL with PIE."""

from __future__ import annotations

import gymnasium as gym
import torch
from types import SimpleNamespace

import unitree_rl_lab.tasks.locomotion.robots.go2  # noqa: F401

from unitree_rl_lab.tasks.locomotion.agents.pie_cfg import AdaptiveEnergyFlatLPACRLPIERunnerCfg
from unitree_rl_lab.tasks.locomotion.mdp.lp_acrl import LPACRLCurriculum, LearningProgressSampler
from unitree_rl_lab.tasks.locomotion.mdp.rewards import straight_command_yaw_rate_error
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
    for name, value in base.curriculum.lp_acrl.params.items():
        if name != "reward_terms":
            assert pie.curriculum.lp_acrl.params[name] == value
    assert pie.curriculum.lp_acrl.params["straight_task_probability"] == 0.30
    assert pie.curriculum.lp_acrl.params["reward_terms"] == (
        *base.curriculum.lp_acrl.params["reward_terms"],
        "straight_yaw_rate_error",
    )
    assert pie.curriculum.lp_acrl.params["episodes_per_stage"] is None
    assert "terrain_names" not in pie.curriculum.lp_acrl.params
    assert set(vars(pie.rewards)) == {*vars(base.rewards), "straight_yaw_rate_error"}
    assert pie.rewards.straight_yaw_rate_error.weight == -0.25
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


def test_straight_anchor_sampler_produces_exact_zero_lateral_and_yaw_commands():
    term = object.__new__(LPACRLCurriculum)
    term._env = SimpleNamespace(device="cpu")
    term.grid_shape = (10, 3, 10)
    term.num_tasks = 300
    term.planar_zero_threshold = 0.2
    term.straight_task_probability = 1.0
    term.vx_edges = torch.arange(0.0, 5.5, 0.5)
    term.vy_edges = torch.tensor((0.0, 0.2, 0.4, 0.6))
    term.yaw_edges = torch.arange(0.0, 5.5, 0.5)
    term.sampler = LearningProgressSampler(
        num_tasks=300,
        device="cpu",
        beta=0.002,
        epsilon=0.1,
        min_samples=4,
        max_probability=0.05,
    )

    task_ids, commands = term._sample_task_batch(1024)

    assert torch.count_nonzero(commands[:, 1:]) == 0
    assert torch.all((task_ids % 10) == 0)
    assert torch.all(((task_ids // 10) % 3) == 0)
    assert term._last_straight_fraction.item() == 1.0


def test_straight_yaw_penalty_is_masked_off_for_non_straight_commands():
    commands = torch.tensor(
        (
            (0.8, 0.0, 0.0),
            (0.8, 0.0, 0.5),
            (0.8, 0.2, 0.0),
            (0.1, 0.0, 0.0),
        )
    )
    env = SimpleNamespace(
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(root_ang_vel_b=torch.tensor(((0.0, 0.0, 0.5),) * 4))
            )
        },
        command_manager=SimpleNamespace(get_command=lambda _: commands),
    )

    penalty = straight_command_yaw_rate_error(env)

    torch.testing.assert_close(penalty, torch.tensor((0.25, 0.0, 0.0, 0.0)))
