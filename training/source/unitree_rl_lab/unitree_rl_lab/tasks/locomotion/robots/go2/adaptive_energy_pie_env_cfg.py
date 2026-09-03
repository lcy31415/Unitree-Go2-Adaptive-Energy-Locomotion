"""Unified multi-terrain PIE task with independent coupled curricula."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .adaptive_energy_lpacrl_pie_env_cfg import (
    AdaptiveEnergyLPACRLPIEObservationsCfg,
    PIEActionsCfg,
)
from .adaptive_energy_pie_stairs_env_cfg import AdaptiveEnergyPIEStairsSceneCfg
from .adaptive_energy_pie_terrain_cfg import (
    ADAPTIVE_ENERGY_PIE_COLUMNS_PER_FAMILY,
    ADAPTIVE_ENERGY_PIE_NUM_LEVELS,
    ADAPTIVE_ENERGY_PIE_TERRAIN_FAMILY_WEIGHTS,
    ADAPTIVE_ENERGY_PIE_TERRAIN_NAMES,
    ADAPTIVE_ENERGY_PIE_TERRAINS_CFG,
)
from .adaptive_energy_stairs_pie_env_cfg import (
    AdaptiveEnergyStairsPIERewardsCfg,
    AdaptiveEnergyStairsPIETerminationsCfg,
)
from .adaptive_energy_terrain_env_cfg import AdaptiveEnergyTerrainEnvCfg
from .adaptive_energy_env_cfg import EventCfg
from .pie_sensors_cfg import PIE_FOOT_SENSOR_NAMES, make_pie_base_clearance_scanner_cfg


ADAPTIVE_ENERGY_PIE_SPEED_ENVELOPES = (
    (2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5),  # flat
    (1.5, 1.5, 1.4, 1.3, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6),  # stairs_up
    (1.5, 1.5, 1.4, 1.3, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6),  # stairs_down
    (2.5, 2.5, 2.2, 2.0, 1.8, 1.5, 1.3, 1.1, 0.9, 0.8),  # slope_up
    (2.5, 2.5, 2.2, 2.0, 1.8, 1.5, 1.3, 1.1, 0.9, 0.8),  # slope_down
    (2.0, 2.0, 1.8, 1.8, 1.5, 1.3, 1.1, 1.0, 0.9, 0.8),  # random_rough
    (1.5, 1.5, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6),  # obstacles
)

# The forward-facing depth camera cannot observe terrain behind the robot.
# Keep every terrain family on non-negative longitudinal commands so all
# commanded traversal remains inside the sensor's observable workspace.
ADAPTIVE_ENERGY_PIE_FORWARD_ONLY = (True,) * 7

# Preserve the original adaptive-energy weighting on easy terrain. Starting at
# stair Level 4, transfer energy weight to forward tracking so high steps do not
# reward a low-power stall more than an energetic, successful traversal.
ADAPTIVE_ENERGY_PIE_ENERGY_SCALES = (
    (1.0,) * 10,  # flat
    (1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3),  # stairs_up
    (1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3),  # stairs_down
    (1.0,) * 10,  # slope_up
    (1.0,) * 10,  # slope_down
    (1.0,) * 10,  # random_rough
    (1.0,) * 10,  # obstacles
)


@configclass
class AdaptiveEnergyPIESceneCfg(AdaptiveEnergyPIEStairsSceneCfg):
    """PIE sensors over seven terrain families with weighted training allocation."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ADAPTIVE_ENERGY_PIE_TERRAINS_CFG,
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
    pie_base_clearance_scanner = make_pie_base_clearance_scanner_cfg()


@configclass
class AdaptiveEnergyPIEEventsCfg(EventCfg):
    """Base randomization plus a stair-focused startup terrain allocation."""

    allocate_terrain_families = EventTerm(
        func=mdp.allocate_terrain_families,
        mode="startup",
        params={
            "family_weights": ADAPTIVE_ENERGY_PIE_TERRAIN_FAMILY_WEIGHTS,
            "columns_per_family": ADAPTIVE_ENERGY_PIE_COLUMNS_PER_FAMILY,
        },
    )


@configclass
class AdaptiveEnergyPIECommandsCfg:
    """One independent velocity curriculum per terrain family."""

    base_velocity = mdp.MultiTerrainRewardThresholdVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        debug_vis=True,
        ranges=mdp.MultiTerrainRewardThresholdVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
        limit_ranges=mdp.MultiTerrainRewardThresholdVelocityCommandCfg.Ranges(
            lin_vel_x=(-2.5, 2.5),
            lin_vel_y=(-0.6, 0.6),
            ang_vel_z=(-2.5, 2.5),
        ),
        num_bins=(21, 13, 21),
        local_range=(0.275, 0.10, 0.275),
        weight_increment=0.2,
        frontier_sampling_probability=0.40,
        active_sampling_probability=0.60,
        replay_sampling_probability=0.0,
        frontier_bin_count=2,
        forward_error_abs=0.18,
        forward_error_rel=0.033,
        lateral_error_abs=0.15,
        lateral_error_rel=0.1,
        angular_error_abs=0.20,
        angular_error_rel=0.1,
        minimum_segment_fraction=0.95,
        linear_stage_threshold=1.25,
        angular_stage_threshold=1.25,
        stage_transition_successes=20,
        zero_command_threshold=0.2,
        terrain_gate_min_mean_level=2.0,
        terrain_gate_close_mean_level=1.5,
        terrain_level_ema_decay=0.98,
        curriculum_update_max_terrain_level=9,
        curriculum_update_level_margin=2,
        terrain_conditioned_max_abs_vx=None,
        yaw_recovery_sampling_probability=0.15,
        yaw_recovery_forward_range=(0.3, 1.0),
        yaw_recovery_abs_yaw_range=(0.2, 0.6),
        terrain_family_names=ADAPTIVE_ENERGY_PIE_TERRAIN_NAMES,
        terrain_columns_per_family=ADAPTIVE_ENERGY_PIE_COLUMNS_PER_FAMILY,
        terrain_family_max_abs_vx=ADAPTIVE_ENERGY_PIE_SPEED_ENVELOPES,
        terrain_family_forward_only=ADAPTIVE_ENERGY_PIE_FORWARD_ONLY,
        terrain_family_allocation_weights=ADAPTIVE_ENERGY_PIE_TERRAIN_FAMILY_WEIGHTS,
    )


@configclass
class AdaptiveEnergyPIECurriculumCfg:
    terrain_levels = CurrTerm(
        func=mdp.adaptive_energy_multiterrain_levels,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "terrain_family_names": ADAPTIVE_ENERGY_PIE_TERRAIN_NAMES,
            "columns_per_family": ADAPTIVE_ENERGY_PIE_COLUMNS_PER_FAMILY,
            "minimum_expected_progress": 1.0,
            "minimum_tracking_fraction": 0.7,
            # Stairs carry a systematic forward-speed deficit; a slightly softer
            # promotion gate lets their curricula rebuild before re-attacking.
            "minimum_tracking_fraction_by_family": {"stairs_up": 0.6, "stairs_down": 0.6},
            "minimum_tracking_fraction_for_hold": 0.45,
            "move_up_distance_fraction": 0.4,
            "move_down_expected_fraction": 0.5,
        },
    )


@configclass
class AdaptiveEnergyPIEEnvCfg(AdaptiveEnergyTerrainEnvCfg):
    """Unified PIE locomotion with isolated terrain-family curricula."""

    scene: AdaptiveEnergyPIESceneCfg = AdaptiveEnergyPIESceneCfg(num_envs=2048, env_spacing=2.5)
    observations: AdaptiveEnergyLPACRLPIEObservationsCfg = AdaptiveEnergyLPACRLPIEObservationsCfg()
    actions: PIEActionsCfg = PIEActionsCfg()
    commands: AdaptiveEnergyPIECommandsCfg = AdaptiveEnergyPIECommandsCfg()
    events: AdaptiveEnergyPIEEventsCfg = AdaptiveEnergyPIEEventsCfg()
    rewards: AdaptiveEnergyStairsPIERewardsCfg = AdaptiveEnergyStairsPIERewardsCfg()
    terminations: AdaptiveEnergyStairsPIETerminationsCfg = AdaptiveEnergyStairsPIETerminationsCfg()
    curriculum: AdaptiveEnergyPIECurriculumCfg = AdaptiveEnergyPIECurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        control_dt = self.decimation * self.sim.dt
        # 25 Hz depth (rather than 10 Hz) cuts the distance travelled between
        # fresh frames from roughly 11 cm to 4.4 cm at 1.1 m/s.
        self.scene.pie_depth_camera.update_period = 2 * control_dt
        self.scene.pie_base_clearance_scanner.update_period = control_dt
        for sensor_name in PIE_FOOT_SENSOR_NAMES:
            getattr(self.scene, sensor_name).update_period = control_dt

        self.scene.terrain.max_init_terrain_level = 0
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True

        # Override the generic terrain defaults with the isolated multi-family
        # curriculum contract.
        command = self.commands.base_velocity
        command.forward_error_abs = 0.18
        command.angular_error_abs = 0.20
        if hasattr(command, "frontier_sampling_probability"):
            command.frontier_sampling_probability = 0.40
            command.active_sampling_probability = 0.60
            command.replay_sampling_probability = 0.0
            command.terrain_gate_min_mean_level = 2.0
            command.terrain_gate_close_mean_level = 1.5
            command.terrain_level_ema_decay = 0.98
            command.curriculum_update_max_terrain_level = ADAPTIVE_ENERGY_PIE_NUM_LEVELS - 1
            command.curriculum_update_level_margin = 2
            command.yaw_recovery_sampling_probability = 0.15

        energy_conditioning = {
            "terrain_columns_per_family": ADAPTIVE_ENERGY_PIE_COLUMNS_PER_FAMILY,
            "terrain_energy_scale_by_family_level": ADAPTIVE_ENERGY_PIE_ENERGY_SCALES,
        }
        self.rewards.Rlin.params.update(energy_conditioning)
        self.rewards.Renergy.params.update(energy_conditioning)
        self.rewards.adaptive_energy_residual.params.update(energy_conditioning)

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


@configclass
class AdaptiveEnergyPIEPlayEnvCfg(AdaptiveEnergyPIEEnvCfg):
    """Small deterministic multi-terrain PIE visualization configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = len(ADAPTIVE_ENERGY_PIE_TERRAIN_NAMES)
        self.events.allocate_terrain_families = None
        self.scene.terrain.max_init_terrain_level = 0
        self.curriculum = None
        self.commands.base_velocity.terrain_gate_min_mean_level = None
        self.commands.base_velocity.terrain_gate_close_mean_level = None
        self.commands.base_velocity.curriculum_update_max_terrain_level = None
        self.events.push_robot = None
        self.observations.actor.enable_corruption = False
        self.observations.proprio_history.enable_corruption = False
