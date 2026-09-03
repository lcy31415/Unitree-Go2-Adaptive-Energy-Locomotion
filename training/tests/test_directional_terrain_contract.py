"""Shared directional-terrain contracts across legacy and PIE Go2 tasks."""

from __future__ import annotations

import math

import isaaclab.terrains as terrain_gen

from unitree_rl_lab.tasks.locomotion.robots.go2.adaptive_energy_lpacrl_terrain_cfg import (
    ADAPTIVE_ENERGY_ROUGHNESS_RANGE,
    ADAPTIVE_ENERGY_SLOPE_RANGE,
    ADAPTIVE_ENERGY_STEP_HEIGHT_RANGE,
    ADAPTIVE_ENERGY_TERRAIN_NUM_LEVELS,
    LPACRL_TERRAINS_CFG,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.adaptive_energy_pie_stairs_env_cfg import (
    PIE_STAIRS_TERRAINS_CFG,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.adaptive_energy_pie_terrain_cfg import (
    ADAPTIVE_ENERGY_PIE_TERRAINS_CFG,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.adaptive_energy_terrain_lpacrl_env_cfg import (
    AdaptiveEnergyTerrainLPACRLEnvCfg,
    AdaptiveEnergyTerrainLPACRLPlayEnvCfg,
)


def test_all_directional_generators_use_motion_semantics():
    """Up means rising and down means falling away from the center origin."""
    generators = (
        LPACRL_TERRAINS_CFG,
        PIE_STAIRS_TERRAINS_CFG,
        ADAPTIVE_ENERGY_PIE_TERRAINS_CFG,
    )
    for generator in generators:
        terrains = generator.sub_terrains
        assert isinstance(terrains["stairs_up"], terrain_gen.MeshInvertedPyramidStairsTerrainCfg)
        assert isinstance(terrains["stairs_down"], terrain_gen.MeshPyramidStairsTerrainCfg)
        if "slope_up" in terrains:
            assert isinstance(terrains["slope_up"], terrain_gen.HfInvertedPyramidSlopedTerrainCfg)
            assert isinstance(terrains["slope_down"], terrain_gen.HfPyramidSlopedTerrainCfg)


def test_legacy_lpacrl_train_and_play_share_centered_reset_contract():
    expected_yaw = (-math.radians(5.0), math.radians(5.0))
    for cfg in (AdaptiveEnergyTerrainLPACRLEnvCfg(), AdaptiveEnergyTerrainLPACRLPlayEnvCfg()):
        pose = cfg.events.reset_base.params["pose_range"]
        assert pose["x"] == (-0.15, 0.15)
        assert pose["y"] == (-0.15, 0.15)
        assert pose["yaw"] == expected_yaw
        assert all(bounds == (0.0, 0.0) for bounds in cfg.events.reset_base.params["velocity_range"].values())
        assert cfg.events.reset_robot_joints.params["velocity_range"] == (0.0, 0.0)


def test_all_registered_rough_tasks_use_ten_level_difficulty_contract():
    cfg = AdaptiveEnergyTerrainLPACRLEnvCfg()
    generator = cfg.scene.terrain.terrain_generator
    assert generator.num_rows == ADAPTIVE_ENERGY_TERRAIN_NUM_LEVELS == 10
    assert cfg.curriculum.lp_acrl.params["num_levels"] == 10
    assert len(cfg.curriculum.lp_acrl.params["terrain_names"]) * 10 * 5 * 5 == 1500
    assert generator.sub_terrains["stairs_up"].step_height_range == ADAPTIVE_ENERGY_STEP_HEIGHT_RANGE
    assert generator.sub_terrains["stairs_down"].step_height_range == ADAPTIVE_ENERGY_STEP_HEIGHT_RANGE
    assert generator.sub_terrains["slope_up"].slope_range == ADAPTIVE_ENERGY_SLOPE_RANGE
    assert generator.sub_terrains["slope_down"].slope_range == ADAPTIVE_ENERGY_SLOPE_RANGE
    assert generator.sub_terrains["random_rough"].amplitude_range == ADAPTIVE_ENERGY_ROUGHNESS_RANGE
