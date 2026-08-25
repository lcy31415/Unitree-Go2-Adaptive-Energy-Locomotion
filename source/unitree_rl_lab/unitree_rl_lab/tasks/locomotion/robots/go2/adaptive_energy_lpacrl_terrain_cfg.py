"""Six terrain families (flat plus five rough) with four exact LP-ACRL geometry levels."""

from __future__ import annotations

import copy

import numpy as np

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGenerator
from isaaclab.terrains.height_field.hf_terrains import random_uniform_terrain
from isaaclab.terrains.height_field.hf_terrains_cfg import HfRandomUniformTerrainCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils.configclass import configclass


class DiscreteLevelTerrainGenerator(TerrainGenerator):
    """Use row/(num_rows-1) instead of randomized difficulty within a row."""

    def _generate_curriculum_terrains(self):
        proportions = np.array([cfg.proportion for cfg in self.cfg.sub_terrains.values()])
        proportions /= proportions.sum()
        cumulative = np.cumsum(proportions)
        sub_indices = np.array(
            [np.min(np.where(col / self.cfg.num_cols + 0.001 < cumulative)[0]) for col in range(self.cfg.num_cols)],
            dtype=np.int32,
        )
        configs = list(self.cfg.sub_terrains.values())
        lower, upper = self.cfg.difficulty_range
        for column in range(self.cfg.num_cols):
            for row in range(self.cfg.num_rows):
                fraction = row / max(self.cfg.num_rows - 1, 1)
                difficulty = lower + (upper - lower) * fraction
                mesh, origin = self._get_terrain_mesh(difficulty, configs[sub_indices[column]])
                self._add_sub_terrain(mesh, origin, row, column, configs[sub_indices[column]])


@height_field_to_mesh
def difficulty_scaled_random_rough_terrain(difficulty: float, cfg) -> np.ndarray:
    """Random roughness whose amplitude is 0.02, 0.047, 0.073, 0.10 m."""
    local_cfg = copy.deepcopy(cfg)
    amplitude = cfg.amplitude_range[0] + difficulty * (cfg.amplitude_range[1] - cfg.amplitude_range[0])
    local_cfg.noise_range = (-amplitude, amplitude)
    return random_uniform_terrain.__wrapped__(difficulty, local_cfg)


@configclass
class LPACRLRandomRoughTerrainCfg(HfRandomUniformTerrainCfg):
    function = difficulty_scaled_random_rough_terrain
    amplitude_range: tuple[float, float] = (0.02, 0.10)


# "flat" occupies terrain-type slot 0; its four geometry levels are identical
# planes, so the curriculum treats it as a low-difficulty anchor family.
LPACRL_TERRAIN_NAMES = ("flat", "stairs_up", "stairs_down", "slope_up", "slope_down", "random_rough")
LPACRL_COLUMNS_PER_TYPE = 4


LPACRL_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    class_type=DiscreteLevelTerrainGenerator,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=4,
    num_cols=len(LPACRL_TERRAIN_NAMES) * LPACRL_COLUMNS_PER_TYPE,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.20),
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.15),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.15),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(0.0, 0.20),
            platform_width=3.0,
            border_width=0.25,
        ),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(0.0, 0.20),
            platform_width=3.0,
            border_width=0.25,
        ),
        "random_rough": LPACRLRandomRoughTerrainCfg(
            proportion=0.20,
            noise_range=(-0.02, 0.02),
            noise_step=0.005,
            downsampled_scale=0.20,
            border_width=0.25,
        ),
    },
)
