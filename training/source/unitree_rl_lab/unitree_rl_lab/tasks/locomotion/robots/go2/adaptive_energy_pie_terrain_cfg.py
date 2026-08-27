"""Deterministic ten-level terrain families for the unified PIE task."""

from __future__ import annotations

import isaaclab.terrains as terrain_gen

from .adaptive_energy_lpacrl_terrain_cfg import (
    DiscreteLevelTerrainGenerator,
    LPACRLRandomRoughTerrainCfg,
)


ADAPTIVE_ENERGY_PIE_TERRAIN_NAMES = (
    "flat",
    "stairs_up",
    "stairs_down",
    "slope_up",
    "slope_down",
    "random_rough",
    "obstacles",
)
ADAPTIVE_ENERGY_PIE_NUM_LEVELS = 10
ADAPTIVE_ENERGY_PIE_COLUMNS_PER_FAMILY = 4
ADAPTIVE_ENERGY_PIE_STEP_HEIGHT_RANGE = (0.03, 0.15)
# MuJoCo/Isaac terrain slopes are rise/run values. 0.35 corresponds to
# atan(0.35)=19.3 degrees, substantially steeper than the old 0.20 maximum.
ADAPTIVE_ENERGY_PIE_SLOPE_RANGE = (0.05, 0.35)
ADAPTIVE_ENERGY_PIE_ROUGHNESS_RANGE = (0.01, 0.10)
ADAPTIVE_ENERGY_PIE_OBSTACLE_HEIGHT_RANGE = (0.03, 0.15)


ADAPTIVE_ENERGY_PIE_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    class_type=DiscreteLevelTerrainGenerator,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=ADAPTIVE_ENERGY_PIE_NUM_LEVELS,
    num_cols=len(ADAPTIVE_ENERGY_PIE_TERRAIN_NAMES) * ADAPTIVE_ENERGY_PIE_COLUMNS_PER_FAMILY,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        # Ten identical geometric rows intentionally act as an independent
        # flat-terrain mastery ladder for its own velocity gate.
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
        # Starting at the center platform, inverted pyramids rise outwards and
        # regular pyramids descend outwards.
        "stairs_up": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=ADAPTIVE_ENERGY_PIE_STEP_HEIGHT_RANGE,
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=ADAPTIVE_ENERGY_PIE_STEP_HEIGHT_RANGE,
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "slope_up": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=1.0,
            slope_range=ADAPTIVE_ENERGY_PIE_SLOPE_RANGE,
            platform_width=3.0,
            border_width=0.25,
        ),
        "slope_down": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0,
            slope_range=ADAPTIVE_ENERGY_PIE_SLOPE_RANGE,
            platform_width=3.0,
            border_width=0.25,
        ),
        "random_rough": LPACRLRandomRoughTerrainCfg(
            proportion=1.0,
            noise_range=(-ADAPTIVE_ENERGY_PIE_ROUGHNESS_RANGE[0], ADAPTIVE_ENERGY_PIE_ROUGHNESS_RANGE[0]),
            amplitude_range=ADAPTIVE_ENERGY_PIE_ROUGHNESS_RANGE,
            noise_step=0.005,
            downsampled_scale=0.20,
            border_width=0.25,
        ),
        "obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=1.0,
            obstacle_height_mode="fixed",
            obstacle_width_range=(0.4, 1.5),
            obstacle_height_range=ADAPTIVE_ENERGY_PIE_OBSTACLE_HEIGHT_RANGE,
            num_obstacles=20,
            platform_width=3.0,
            border_width=0.25,
        ),
    },
)
