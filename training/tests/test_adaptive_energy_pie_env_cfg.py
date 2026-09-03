"""Contracts for the unified independent multi-terrain PIE task."""

from __future__ import annotations

import math
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

import unitree_rl_lab.tasks.locomotion.robots.go2  # noqa: F401

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.agents.pie_cfg import AdaptiveEnergyPIERunnerCfg
from unitree_rl_lab.tasks.locomotion.mdp.commands.reward_threshold_velocity_command import (
    MultiTerrainRewardThresholdVelocityCommand,
    RewardThresholdVelocityCommand,
)
from unitree_rl_lab.tasks.locomotion.mdp.rewards import _terrain_conditioned_energy_weights
from unitree_rl_lab.tasks.locomotion.robots.go2.adaptive_energy_pie_env_cfg import (
    ADAPTIVE_ENERGY_PIE_ENERGY_SCALES,
    ADAPTIVE_ENERGY_PIE_FORWARD_ONLY,
    ADAPTIVE_ENERGY_PIE_SPEED_ENVELOPES,
    AdaptiveEnergyPIEEnvCfg,
    AdaptiveEnergyPIEPlayEnvCfg,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.adaptive_energy_pie_terrain_cfg import (
    ADAPTIVE_ENERGY_PIE_COLUMNS_PER_FAMILY,
    ADAPTIVE_ENERGY_PIE_NUM_LEVELS,
    ADAPTIVE_ENERGY_PIE_OBSTACLE_HEIGHT_RANGE,
    ADAPTIVE_ENERGY_PIE_ROUGHNESS_RANGE,
    ADAPTIVE_ENERGY_PIE_SLOPE_RANGE,
    ADAPTIVE_ENERGY_PIE_STEP_HEIGHT_RANGE,
    ADAPTIVE_ENERGY_PIE_TERRAIN_FAMILY_WEIGHTS,
    ADAPTIVE_ENERGY_PIE_TERRAIN_NAMES,
)


TASK_ID = "Unitree-Go2-Adaptive-Energy-PIE"


def test_unified_pie_task_registration_and_runner():
    kwargs = gym.spec(TASK_ID).kwargs
    assert kwargs["env_cfg_entry_point"].endswith(":AdaptiveEnergyPIEEnvCfg")
    assert kwargs["play_env_cfg_entry_point"].endswith(":AdaptiveEnergyPIEPlayEnvCfg")
    assert kwargs["rsl_rl_cfg_entry_point"].endswith(":AdaptiveEnergyPIERunnerCfg")
    runner = AdaptiveEnergyPIERunnerCfg()
    assert runner.experiment_name == "unitree_go2_adaptive_energy_pie"
    assert runner.max_iterations == 30_000
    assert runner.save_interval == 200


def test_seven_terrain_families_have_ten_deterministic_levels():
    cfg = AdaptiveEnergyPIEEnvCfg()
    generator = cfg.scene.terrain.terrain_generator
    assert tuple(generator.sub_terrains) == ADAPTIVE_ENERGY_PIE_TERRAIN_NAMES
    assert generator.num_rows == ADAPTIVE_ENERGY_PIE_NUM_LEVELS == 10
    assert generator.num_cols == len(ADAPTIVE_ENERGY_PIE_TERRAIN_NAMES) * 4 == 28
    assert ADAPTIVE_ENERGY_PIE_COLUMNS_PER_FAMILY == 4
    assert cfg.scene.terrain.max_init_terrain_level == 0

    # Match DiscreteLevelTerrainGenerator's deterministic column assignment.
    proportions = np.ones(len(ADAPTIVE_ENERGY_PIE_TERRAIN_NAMES))
    cumulative = np.cumsum(proportions / proportions.sum())
    family_ids = [
        int(np.min(np.where(column / generator.num_cols + 0.001 < cumulative)[0]))
        for column in range(generator.num_cols)
    ]
    assert [family_ids.count(index) for index in range(len(cumulative))] == [4] * 7


def test_training_allocation_keeps_flat_anchor_and_all_terrain_families():
    cfg = AdaptiveEnergyPIEEnvCfg()
    term = cfg.events.allocate_terrain_families
    assert term.func is mdp.allocate_terrain_families
    assert term.params["family_weights"] == ADAPTIVE_ENERGY_PIE_TERRAIN_FAMILY_WEIGHTS
    assert ADAPTIVE_ENERGY_PIE_TERRAIN_FAMILY_WEIGHTS == (
        0.20,
        0.25,
        0.20,
        0.075,
        0.075,
        0.15,
        0.05,
    )
    assert math.isclose(sum(ADAPTIVE_ENERGY_PIE_TERRAIN_FAMILY_WEIGHTS), 1.0)

    num_envs = 100
    terrain = SimpleNamespace(
        terrain_types=torch.zeros(num_envs, dtype=torch.long),
        terrain_levels=torch.zeros(num_envs, dtype=torch.long),
        terrain_origins=torch.zeros(10, 28, 3),
        env_origins=torch.zeros(num_envs, 3),
    )
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene=SimpleNamespace(terrain=terrain),
    )
    mdp.allocate_terrain_families(
        env,
        None,
        ADAPTIVE_ENERGY_PIE_TERRAIN_FAMILY_WEIGHTS,
        ADAPTIVE_ENERGY_PIE_COLUMNS_PER_FAMILY,
    )
    family_ids = torch.div(
        terrain.terrain_types,
        ADAPTIVE_ENERGY_PIE_COLUMNS_PER_FAMILY,
        rounding_mode="floor",
    )
    assert torch.bincount(family_ids, minlength=7).tolist() == [20, 25, 20, 8, 7, 15, 5]


def test_geometry_ranges_include_steeper_slopes():
    cfg = AdaptiveEnergyPIEEnvCfg()
    terrains = cfg.scene.terrain.terrain_generator.sub_terrains
    assert terrains["stairs_up"].step_height_range == ADAPTIVE_ENERGY_PIE_STEP_HEIGHT_RANGE == (0.03, 0.15)
    assert terrains["stairs_down"].step_height_range == (0.03, 0.15)
    assert terrains["slope_up"].slope_range == ADAPTIVE_ENERGY_PIE_SLOPE_RANGE == (0.05, 0.35)
    assert terrains["slope_down"].slope_range == (0.05, 0.35)
    assert math.degrees(math.atan(ADAPTIVE_ENERGY_PIE_SLOPE_RANGE[1])) > 19.0
    assert terrains["random_rough"].amplitude_range == ADAPTIVE_ENERGY_PIE_ROUGHNESS_RANGE
    assert terrains["obstacles"].obstacle_height_range == ADAPTIVE_ENERGY_PIE_OBSTACLE_HEIGHT_RANGE


def test_each_family_has_an_independent_velocity_curriculum_contract():
    command = AdaptiveEnergyPIEEnvCfg().commands.base_velocity
    assert command.class_type is MultiTerrainRewardThresholdVelocityCommand
    assert command.terrain_family_names == ADAPTIVE_ENERGY_PIE_TERRAIN_NAMES
    assert command.terrain_columns_per_family == 4
    assert command.terrain_family_max_abs_vx == ADAPTIVE_ENERGY_PIE_SPEED_ENVELOPES
    assert command.terrain_family_forward_only == ADAPTIVE_ENERGY_PIE_FORWARD_ONLY
    assert command.terrain_family_allocation_weights == ADAPTIVE_ENERGY_PIE_TERRAIN_FAMILY_WEIGHTS
    assert len(command.terrain_family_max_abs_vx) == 7
    assert all(len(envelope) == 10 for envelope in command.terrain_family_max_abs_vx)
    assert command.terrain_gate_min_mean_level == 2.0
    assert command.terrain_gate_close_mean_level == 1.5
    assert command.curriculum_update_level_margin == 2
    assert command.curriculum_update_max_terrain_level == 9
    assert command.frontier_sampling_probability == 0.4
    assert command.active_sampling_probability == 0.6
    assert command.replay_sampling_probability == 0.0
    assert command.yaw_recovery_sampling_probability == 0.15
    assert command.terrain_family_max_abs_vx[0] == (2.5,) * 10
    assert command.terrain_family_max_abs_vx[1] == (
        1.5, 1.5, 1.4, 1.3, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6
    )
    assert command.terrain_family_max_abs_vx[-1][-1] == 0.6
    assert command.terrain_family_forward_only == (True,) * 7

    reward_cfg = AdaptiveEnergyPIEEnvCfg().rewards
    for reward_term in (reward_cfg.Rlin, reward_cfg.Renergy, reward_cfg.adaptive_energy_residual):
        assert reward_term.params["terrain_columns_per_family"] == 4
        assert reward_term.params["terrain_energy_scale_by_family_level"] == (
            ADAPTIVE_ENERGY_PIE_ENERGY_SCALES
        )
    assert ADAPTIVE_ENERGY_PIE_ENERGY_SCALES[0] == (1.0,) * 10
    assert ADAPTIVE_ENERGY_PIE_ENERGY_SCALES[1][4:] == (0.5, 0.5, 0.5, 0.3, 0.3, 0.3)


def test_multiterrain_level_curriculum_and_pie_interfaces():
    cfg = AdaptiveEnergyPIEEnvCfg()
    term = cfg.curriculum.terrain_levels
    assert term.func is mdp.adaptive_energy_multiterrain_levels
    assert term.params["terrain_family_names"] == ADAPTIVE_ENERGY_PIE_TERRAIN_NAMES
    assert term.params["columns_per_family"] == 4
    assert term.params["minimum_tracking_fraction"] == 0.7
    assert term.params["minimum_tracking_fraction_for_hold"] == 0.45
    assert cfg.scene.pie_depth_camera.update_period == 0.04
    assert cfg.scene.pie_base_clearance_scanner.update_period == 0.02
    assert cfg.observations.camera.depth_history.params["frame_history_length"] == 2
    assert cfg.observations.proprio_history.history_length == 10
    assert cfg.terminations.persistent_body_contact.params["sensor_cfg"].body_names == [
        ".*_hip",
        ".*_thigh",
    ]


def test_stair_energy_weight_is_transferred_to_tracking_at_high_levels():
    terrain = SimpleNamespace(
        terrain_types=torch.tensor([0, 4, 4, 4, 8]),
        terrain_levels=torch.tensor([9, 3, 4, 7, 9]),
    )
    env = SimpleNamespace(scene=SimpleNamespace(terrain=terrain))
    linear, energy = _terrain_conditioned_energy_weights(
        env,
        torch.full((5,), 0.4),
        torch.full((5,), 0.4),
        ADAPTIVE_ENERGY_PIE_COLUMNS_PER_FAMILY,
        ADAPTIVE_ENERGY_PIE_ENERGY_SCALES,
    )
    assert torch.allclose(energy, torch.tensor([0.4, 0.4, 0.2, 0.12, 0.12]))
    assert torch.allclose(linear, torch.tensor([0.4, 0.4, 0.6, 0.68, 0.68]))
    assert torch.allclose(linear + energy, torch.full((5,), 0.8))


def test_play_configuration_disables_learning_and_corruption():
    play = AdaptiveEnergyPIEPlayEnvCfg()
    assert play.scene.num_envs == len(ADAPTIVE_ENERGY_PIE_TERRAIN_NAMES)
    assert play.events.allocate_terrain_families is None
    assert play.curriculum is None
    assert play.events.push_robot is None
    assert play.commands.base_velocity.terrain_gate_min_mean_level is None
    assert not play.observations.actor.enable_corruption
    assert not play.observations.proprio_history.enable_corruption


def test_multiterrain_command_exposes_checkpoint_contract():
    assert hasattr(MultiTerrainRewardThresholdVelocityCommand, "state_dict")
    assert hasattr(MultiTerrainRewardThresholdVelocityCommand, "load_state_dict")
    assert hasattr(MultiTerrainRewardThresholdVelocityCommand, "resample_current_episodes")
    assert hasattr(MultiTerrainRewardThresholdVelocityCommand, "csv_snapshot")


def _make_fake_multiterrain_command(monkeypatch):
    cfg = AdaptiveEnergyPIEEnvCfg().commands.base_velocity
    num_envs = len(ADAPTIVE_ENERGY_PIE_TERRAIN_NAMES) * 2
    terrain_types = torch.tensor(
        [family * ADAPTIVE_ENERGY_PIE_COLUMNS_PER_FAMILY for family in range(7) for _ in range(2)]
    )
    terrain = SimpleNamespace(
        terrain_types=terrain_types,
        terrain_levels=torch.tensor([0, 9] * 7),
        terrain_origins=torch.zeros(10, 28, 3),
        env_origins=torch.zeros(num_envs, 3),
    )
    env = SimpleNamespace(
        device="cpu",
        num_envs=num_envs,
        scene=SimpleNamespace(terrain=terrain),
        common_step_counter=0,
        step_dt=0.02,
    )

    def fake_base_init(self, command_cfg, fake_env):
        self.cfg = command_cfg
        self._env = fake_env
        self.metrics = {
            name: torch.zeros(self.num_envs)
            for name in (
                "curriculum_success",
                "curriculum_active_fraction",
                "curriculum_mean_weight",
                "curriculum_stage",
                "curriculum_stage_progress",
                "curriculum_error_vx",
                "curriculum_error_vy",
                "curriculum_error_yaw",
                "sample_frontier",
                "sample_active_uniform",
                "sample_low_speed_replay",
                "sample_yaw_recovery",
                "terrain_gate_open",
                "terrain_level_ema",
                "curriculum_eligible_max_level",
            )
        }
        self.vel_command_b = torch.zeros(self.num_envs, 3)
        self.bin_ids = torch.full((self.num_envs,), -1, dtype=torch.long)
        self._velocity_abs_error_sum = torch.zeros(self.num_envs, 3)
        self._segment_steps = torch.zeros(self.num_envs, dtype=torch.long)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool)
        self._stage = 0
        self._stage_success_count = 0
        self._terrain_level_ema = None
        self._terrain_gate_open = False

    monkeypatch.setattr(RewardThresholdVelocityCommand, "__init__", fake_base_init)
    return MultiTerrainRewardThresholdVelocityCommand(cfg, env), env


def test_multiterrain_sampling_is_isolated_and_respects_each_envelope(monkeypatch):
    command, env = _make_fake_multiterrain_command(monkeypatch)
    assert len({id(curriculum) for curriculum in command.curricula}) == 7
    original_second = command.curricula[1].weights.clone()
    command.curricula[0].weights[command.curricula[0].initially_active] = 0.5
    assert torch.equal(command.curricula[1].weights, original_second)

    env_ids = torch.arange(env.num_envs)
    command._resample_command(env_ids)
    family_ids = command._family_ids(env_ids)
    for family_index, envelope in enumerate(ADAPTIVE_ENERGY_PIE_SPEED_ENVELOPES):
        selected = family_ids == family_index
        limits = torch.tensor(envelope)[env.scene.terrain.terrain_levels[selected]]
        assert torch.all(command.vel_command_b[selected, 0].abs() <= limits + 1.0e-6)
        if ADAPTIVE_ENERGY_PIE_FORWARD_ONLY[family_index]:
            assert torch.all(command.vel_command_b[selected, 0] >= 0.0)
    assert not torch.any(command.metrics["sample_low_speed_replay"] > 0.0)


def test_multiterrain_curriculum_checkpoint_round_trip(monkeypatch):
    command, env = _make_fake_multiterrain_command(monkeypatch)
    command._family_stages = [2, 1, 0, 2, 1, 0, 1]
    command._family_stage_success_counts = [0, 3, 4, 0, 8, 2, 7]
    command._family_terrain_level_ema = [float(index) / 2 for index in range(7)]
    command._family_terrain_gate_open = [True, True, False, True, False, False, True]
    command.curricula[3].weights[0] = 0.73
    env.scene.terrain.terrain_levels[:] = torch.tensor([0, 9] * 7)
    state = command.state_dict()

    restored, restored_env = _make_fake_multiterrain_command(monkeypatch)
    restored.load_state_dict(state)
    assert restored._family_stages == command._family_stages
    assert restored._family_stage_success_counts == command._family_stage_success_counts
    assert restored._family_terrain_level_ema == command._family_terrain_level_ema
    assert restored._family_terrain_gate_open == command._family_terrain_gate_open
    assert torch.equal(restored.curricula[3].weights, command.curricula[3].weights)
    assert torch.equal(restored_env.scene.terrain.terrain_levels, env.scene.terrain.terrain_levels)
