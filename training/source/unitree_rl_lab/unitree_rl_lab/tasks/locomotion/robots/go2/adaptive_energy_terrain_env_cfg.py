"""Gait-free adaptive-energy locomotion on curriculum-generated terrain."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .adaptive_energy_env_cfg import (
    AdaptiveEnergyEnvCfg,
    AdaptiveEnergyObservationsCfg,
    AdaptiveEnergyPlayEnvCfg,
    AdaptiveEnergySceneCfg,
)
from .adaptive_energy_terrain_cfg import ADAPTIVE_ENERGY_TERRAINS_CFG


@configclass
class AdaptiveEnergyTerrainSceneCfg(AdaptiveEnergySceneCfg):
    """Go2 scene with generated terrain and a critic-only height scanner."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ADAPTIVE_ENERGY_TERRAINS_CFG,
        max_init_terrain_level=0,
        use_terrain_origins=True,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.18, 0.18)),
        debug_vis=False,
    )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )


@configclass
class AdaptiveEnergyTerrainObservationsCfg(AdaptiveEnergyObservationsCfg):
    """Keep the flat-task actor shape and privilege only the terrain critic."""

    @configclass
    class CriticCfg(AdaptiveEnergyObservationsCfg.CriticCfg, ObsGroup):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
        )

    critic: CriticCfg = CriticCfg()


@configclass
class AdaptiveEnergyTerrainCurriculumCfg:
    terrain_levels = CurrTerm(
        func=mdp.adaptive_energy_terrain_levels,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "minimum_expected_progress": 1.0,
            "minimum_tracking_fraction": 0.7,
            "move_up_distance_fraction": 0.5,
            "move_down_expected_fraction": 0.5,
        },
    )


@configclass
class AdaptiveEnergyTerrainEnvCfg(AdaptiveEnergyEnvCfg):
    """Training task combining terrain and staged velocity curricula."""

    scene: AdaptiveEnergyTerrainSceneCfg = AdaptiveEnergyTerrainSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: AdaptiveEnergyTerrainObservationsCfg = AdaptiveEnergyTerrainObservationsCfg()
    curriculum: AdaptiveEnergyTerrainCurriculumCfg = AdaptiveEnergyTerrainCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # Terrain-task command curriculum.  Halving the longitudinal/yaw
        # limits also halves their stage frontiers and neighborhood widths so
        # the number of neighboring bins activated by a success stays close
        # to the flat-task setting.
        command = self.commands.base_velocity
        # External curriculum commands (for example FloodFill) intentionally
        # do not expose the staged RewardThreshold command fields.
        if hasattr(command, "limit_ranges"):
            command.limit_ranges.lin_vel_x = (-2.5, 2.5)
            command.limit_ranges.ang_vel_z = (-2.5, 2.5)
            command.linear_stage_threshold = 1.25
            command.angular_stage_threshold = 1.25
            command.local_range = (0.275, 0.10, 0.275)
            command.forward_error_abs = 0.15
            # A zero-yaw command still leaves gait-induced yaw-rate oscillation
            # around 0.2 rad/s, which alone consumed the flat-task angular floor.
            command.angular_error_abs = 0.3

            # Command bins may expand once terrain training has left level 0,
            # but only successes on easier levels 0--3 count.
            command.terrain_gate_min_mean_level = 1.0
            command.curriculum_update_max_terrain_level = 3

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True


@configclass
class AdaptiveEnergyTerrainPlayEnvCfg(AdaptiveEnergyTerrainEnvCfg):
    """Small deterministic-layout configuration for terrain evaluation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.max_init_terrain_level = 0
        self.commands.base_velocity.terrain_gate_min_mean_level = None
        self.commands.base_velocity.curriculum_update_max_terrain_level = None
        self.events.push_robot = None
