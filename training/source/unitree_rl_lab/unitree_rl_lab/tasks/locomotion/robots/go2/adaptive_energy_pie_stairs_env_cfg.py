"""Directed up/down-stair PIE task without LP-ACRL sampling or curriculum."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .adaptive_energy_env_cfg import AdaptiveEnergyEnvCfg, AdaptiveEnergyRewardsCfg
from .adaptive_energy_lpacrl_pie_env_cfg import (
    AdaptiveEnergyLPACRLPIEObservationsCfg,
    PIEActionsCfg,
)
from .adaptive_energy_lpacrl_terrain_cfg import DiscreteLevelTerrainGenerator
from .adaptive_energy_terrain_env_cfg import AdaptiveEnergyTerrainSceneCfg
from .pie_sensors_cfg import (
    PIE_FOOT_ORDER,
    PIE_FOOT_SENSOR_NAMES,
    make_pie_depth_camera_cfg,
    make_pie_foot_scanner_cfg,
    make_pie_height_scanner_cfg,
)
from .velocity_env_cfg import TerminationsCfg


PIE_STAIRS_TERRAIN_NAMES = ("stairs_up", "stairs_down")
PIE_STAIRS_COLUMNS_PER_TYPE = 4
PIE_STAIRS_FINISH_DISTANCE = 3.3


# Both terrains start on the 3 m-wide center platform and are traversed in
# world +x.  An inverted pyramid rises outwards (upstairs), while a regular
# pyramid descends outwards (downstairs).  This fixes the directional naming
# ambiguity of the older LP-ACRL terrain set.
PIE_STAIRS_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    class_type=DiscreteLevelTerrainGenerator,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=4,
    num_cols=len(PIE_STAIRS_TERRAIN_NAMES) * PIE_STAIRS_COLUMNS_PER_TYPE,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    # This selects deterministic row-wise geometry generation only. Runtime
    # learning curricula remain disabled by ``AdaptiveEnergyPIEStairsEnvCfg``.
    curriculum=True,
    sub_terrains={
        "stairs_up": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.5,
            step_height_range=(0.05, 0.15),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.5,
            step_height_range=(0.05, 0.15),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


@configclass
class AdaptiveEnergyPIEStairsSceneCfg(AdaptiveEnergyTerrainSceneCfg):
    """PIE sensor scene containing only directed up/down stair courses."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=PIE_STAIRS_TERRAINS_CFG,
        max_init_terrain_level=3,
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
    height_scanner = make_pie_height_scanner_cfg()
    pie_depth_camera = make_pie_depth_camera_cfg()
    pie_fr_foot_scanner = make_pie_foot_scanner_cfg(PIE_FOOT_ORDER[0])
    pie_fl_foot_scanner = make_pie_foot_scanner_cfg(PIE_FOOT_ORDER[1])
    pie_rr_foot_scanner = make_pie_foot_scanner_cfg(PIE_FOOT_ORDER[2])
    pie_rl_foot_scanner = make_pie_foot_scanner_cfg(PIE_FOOT_ORDER[3])


@configclass
class AdaptiveEnergyPIEStairsCommandsCfg:
    """Stationary distribution of forward commands; no task sampler state."""

    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        rel_standing_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 1.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )


@configclass
class AdaptiveEnergyPIEStairsRewardsCfg(AdaptiveEnergyRewardsCfg):
    """Adaptive-energy objective plus an explicit end-of-course bonus."""

    course_completion = RewTerm(
        func=mdp.stair_course_completion_reward,
        weight=100.0,
        params={
            "finish_distance": PIE_STAIRS_FINISH_DISTANCE,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class AdaptiveEnergyPIEStairsTerminationsCfg(TerminationsCfg):
    """Safety failures and unambiguous success for a directed stair course."""

    course_complete = DoneTerm(
        func=mdp.stair_course_complete,
        time_out=True,
        params={
            "finish_distance": PIE_STAIRS_FINISH_DISTANCE,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    lateral_deviation = DoneTerm(
        func=mdp.stair_lateral_deviation,
        time_out=False,
        params={"max_deviation": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    heading_error = DoneTerm(
        func=mdp.stair_heading_error,
        time_out=False,
        params={
            "max_heading_error": math.radians(30.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class AdaptiveEnergyPIEStairsEnvCfg(AdaptiveEnergyEnvCfg):
    """PIE stair experiment with uniform sampling and no LP-ACRL logic."""

    scene: AdaptiveEnergyPIEStairsSceneCfg = AdaptiveEnergyPIEStairsSceneCfg(
        num_envs=256,
        env_spacing=2.5,
    )
    observations: AdaptiveEnergyLPACRLPIEObservationsCfg = AdaptiveEnergyLPACRLPIEObservationsCfg()
    actions: PIEActionsCfg = PIEActionsCfg()
    commands: AdaptiveEnergyPIEStairsCommandsCfg = AdaptiveEnergyPIEStairsCommandsCfg()
    rewards: AdaptiveEnergyPIEStairsRewardsCfg = AdaptiveEnergyPIEStairsRewardsCfg()
    terminations: AdaptiveEnergyPIEStairsTerminationsCfg = AdaptiveEnergyPIEStairsTerminationsCfg()
    curriculum = None

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 12.0
        control_dt = self.decimation * self.sim.dt
        self.scene.height_scanner.update_period = control_dt
        self.scene.pie_depth_camera.update_period = 5 * control_dt
        for sensor_name in PIE_FOOT_SENSOR_NAMES:
            getattr(self.scene, sensor_name).update_period = control_dt

        # Every traversal starts near the center of the initial platform and
        # faces world +x, making terrain direction the only intended variable.
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.15, 0.15),
            "y": (-0.15, 0.15),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-math.radians(5.0), math.radians(5.0)),
        }
        self.events.reset_base.params["velocity_range"] = {
            key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        # External pushes would confound the focused perception experiment.
        self.events.push_robot = None


@configclass
class AdaptiveEnergyPIEStairsPlayEnvCfg(AdaptiveEnergyPIEStairsEnvCfg):
    """Small deterministic-observation variant for visualization/evaluation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.observations.actor.enable_corruption = False
        self.observations.proprio_history.enable_corruption = False
