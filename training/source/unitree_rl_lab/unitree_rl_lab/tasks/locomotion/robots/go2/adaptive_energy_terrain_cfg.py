"""Composite rough-terrain generator for adaptive-energy Go2 locomotion."""

import isaaclab.terrains as terrain_gen

from .adaptive_energy_lpacrl_terrain_cfg import (
    ADAPTIVE_ENERGY_OBSTACLE_HEIGHT_RANGE,
    ADAPTIVE_ENERGY_ROUGHNESS_RANGE,
    ADAPTIVE_ENERGY_SLOPE_RANGE,
    ADAPTIVE_ENERGY_STEP_HEIGHT_RANGE,
    ADAPTIVE_ENERGY_TERRAIN_NUM_LEVELS,
    DiscreteLevelTerrainGenerator,
    LPACRLRandomRoughTerrainCfg,
)


# The proportions intentionally sum to one.  The reference project uses the
# same terrain families, but its published values sum to 1.2 and are therefore
# implicitly renormalized by Isaac Lab.
ADAPTIVE_ENERGY_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    class_type=DiscreteLevelTerrainGenerator,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=ADAPTIVE_ENERGY_TERRAIN_NUM_LEVELS,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.20),
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=ADAPTIVE_ENERGY_SLOPE_RANGE,
            platform_width=3.0,
            border_width=0.25,
        ),
        "pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=ADAPTIVE_ENERGY_SLOPE_RANGE,
            platform_width=3.0,
            border_width=0.25,
        ),
        "random_rough": LPACRLRandomRoughTerrainCfg(
            proportion=0.15,
            noise_range=(-ADAPTIVE_ENERGY_ROUGHNESS_RANGE[0], ADAPTIVE_ENERGY_ROUGHNESS_RANGE[0]),
            amplitude_range=ADAPTIVE_ENERGY_ROUGHNESS_RANGE,
            noise_step=0.005,
            downsampled_scale=0.20,
            border_width=0.25,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=ADAPTIVE_ENERGY_STEP_HEIGHT_RANGE,
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=ADAPTIVE_ENERGY_STEP_HEIGHT_RANGE,
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.10,
            obstacle_height_mode="fixed",
            obstacle_width_range=(1.0, 2.0),
            obstacle_height_range=ADAPTIVE_ENERGY_OBSTACLE_HEIGHT_RANGE,
            num_obstacles=20,
            platform_width=3.0,
            border_width=0.25,
        ),
    },
)
