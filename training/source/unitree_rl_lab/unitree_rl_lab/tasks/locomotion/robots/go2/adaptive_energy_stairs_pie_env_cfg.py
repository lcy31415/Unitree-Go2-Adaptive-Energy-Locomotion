"""Stair-only PIE task with coupled terrain and original velocity curricula."""

from __future__ import annotations

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .adaptive_energy_lpacrl_pie_env_cfg import (
    AdaptiveEnergyLPACRLPIEObservationsCfg,
    PIEActionsCfg,
)
from .adaptive_energy_lpacrl_terrain_cfg import (
    ADAPTIVE_ENERGY_STEP_HEIGHT_RANGE,
    ADAPTIVE_ENERGY_TERRAIN_NUM_LEVELS,
)
from .adaptive_energy_env_cfg import AdaptiveEnergyRewardsCfg
from .adaptive_energy_pie_stairs_env_cfg import AdaptiveEnergyPIEStairsSceneCfg
from .adaptive_energy_terrain_env_cfg import AdaptiveEnergyTerrainEnvCfg
from .pie_sensors_cfg import PIE_FOOT_SENSOR_NAMES, make_pie_base_clearance_scanner_cfg
from .velocity_env_cfg import TerminationsCfg


STAIRS_PIE_NUM_LEVELS = ADAPTIVE_ENERGY_TERRAIN_NUM_LEVELS
STAIRS_PIE_STEP_HEIGHT_RANGE = ADAPTIVE_ENERGY_STEP_HEIGHT_RANGE
STAIRS_PIE_VELOCITY_GATE_MIN_MEAN_LEVEL = 2
STAIRS_PIE_VELOCITY_GATE_CLOSE_MEAN_LEVEL = 1.5
STAIRS_PIE_VELOCITY_UPDATE_LEVEL_MARGIN = 2
STAIRS_PIE_VELOCITY_UPDATE_MAX_LEVEL = 9
STAIRS_PIE_MAX_ABS_VX_BY_LEVEL = (2.5, 2.5, 2.5, 2.0, 2.0, 1.5, 1.5, 1.1, 1.1, 0.8)


@configclass
class AdaptiveEnergyStairsPIESceneCfg(AdaptiveEnergyPIEStairsSceneCfg):
    """Centered up/down-stair geometry with ten deterministic height levels."""

    pie_base_clearance_scanner = make_pie_base_clearance_scanner_cfg()

    def __post_init__(self):
        super().__post_init__()
        # Resolve the shared 3--15 cm contract into ten deterministic rows:
        # 3.00, 4.33, ..., 15.00 cm. Configclass gives this scene its own
        # generator copy, while the constants keep every task synchronized.
        if self.terrain.terrain_generator is not None:
            self.terrain.terrain_generator.num_rows = STAIRS_PIE_NUM_LEVELS
            for terrain in self.terrain.terrain_generator.sub_terrains.values():
                terrain.step_height_range = STAIRS_PIE_STEP_HEIGHT_RANGE


@configclass
class AdaptiveEnergyStairsPIECurriculumCfg:
    """Promote or demote the ten deterministic stair-height levels."""

    terrain_levels = CurrTerm(
        func=mdp.adaptive_energy_terrain_levels,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "minimum_expected_progress": 1.0,
            "minimum_tracking_fraction": 0.7,
            "minimum_tracking_fraction_for_hold": 0.45,
            # The reference directed task completes at x=3.3 m. Requiring
            # 3.2 m preserves that traversal contract without demanding the
            # exact 4 m edge of an 8 m terrain tile.
            "move_up_distance_fraction": 0.4,
            "move_down_expected_fraction": 0.5,
        },
    )


@configclass
class AdaptiveEnergyStairsPIERewardsCfg(AdaptiveEnergyRewardsCfg):
    """Adaptive-energy objective with an anti-stagnation guard."""

    command_stagnation = RewTerm(
        func=mdp.command_stagnation_penalty,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "min_command": 0.3,
            "ratio": 0.25,
            "grace_s": 1.0,
            "sustain_s": 0.5,
            "penalty": 0.5,
        },
    )
    straight_yaw_rate_error = RewTerm(
        func=mdp.straight_command_yaw_rate_error,
        weight=-0.25,
        params={
            "command_name": "base_velocity",
            "command_lateral_threshold": 1.0e-6,
            "command_yaw_threshold": 1.0e-6,
            "minimum_forward_speed": 0.2,
            "max_squared_error": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class AdaptiveEnergyStairsPIETerminationsCfg(TerminationsCfg):
    """Reject low-clearance, body-supported, and command-ignoring solutions."""

    low_base_clearance = DoneTerm(
        func=mdp.low_base_clearance,
        time_out=False,
        params={
            "sensor_cfg": SceneEntityCfg("pie_base_clearance_scanner"),
            "minimum_clearance": 0.20,
            "sustain_s": 0.20,
        },
    )
    persistent_body_contact = DoneTerm(
        func=mdp.persistent_body_contact,
        time_out=False,
        params={
            "sensor_cfg": SceneEntityCfg(
                # Calf contact with a stair edge can be part of a valid
                # recovery/climbing motion.  Only sustained hip or thigh
                # support should terminate the episode.
                "contact_forces", body_names=[".*_hip", ".*_thigh"]
            ),
            "threshold": 5.0,
            "sustain_s": 0.20,
        },
    )
    bad_command_stagnation = DoneTerm(
        func=mdp.bad_command_stagnation,
        time_out=False,
        params={
            "command_name": "base_velocity",
            "min_command": 0.3,
            "ratio": 0.25,
            "grace_s": 1.0,
            "sustain_s": 3.0,
        },
    )


@configclass
class AdaptiveEnergyStairsPIEEnvCfg(AdaptiveEnergyTerrainEnvCfg):
    """Original staged velocity curriculum on up/down stairs with PIE."""

    scene: AdaptiveEnergyStairsPIESceneCfg = AdaptiveEnergyStairsPIESceneCfg(
        num_envs=256,
        env_spacing=2.5,
    )
    observations: AdaptiveEnergyLPACRLPIEObservationsCfg = AdaptiveEnergyLPACRLPIEObservationsCfg()
    actions: PIEActionsCfg = PIEActionsCfg()
    rewards: AdaptiveEnergyStairsPIERewardsCfg = AdaptiveEnergyStairsPIERewardsCfg()
    terminations: AdaptiveEnergyStairsPIETerminationsCfg = AdaptiveEnergyStairsPIETerminationsCfg()
    curriculum: AdaptiveEnergyStairsPIECurriculumCfg = AdaptiveEnergyStairsPIECurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        control_dt = self.decimation * self.sim.dt
        self.scene.pie_depth_camera.update_period = 5 * control_dt
        self.scene.pie_base_clearance_scanner.update_period = control_dt
        for sensor_name in PIE_FOOT_SENSOR_NAMES:
            getattr(self.scene, sensor_name).update_period = control_dt

        # Start every environment at the easiest of the ten exact stair
        # heights. The terrain curriculum owns all later promotion/demotion.
        self.scene.terrain.max_init_terrain_level = 0
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True

        # Stair impacts cause short forward-speed transients even during a
        # successful traversal. Keep the all-axis 70% tracking contract, but
        # widen the forward component from the generic terrain value (0.15).
        self.commands.base_velocity.forward_error_abs = 0.18
        # A 0.30 rad/s tracking floor can accumulate more than 30 degrees of
        # heading error during a slow stair traversal. Tighten it while still
        # allowing short gait-induced yaw-rate transients.
        self.commands.base_velocity.angular_error_abs = 0.20

        # No fixed low-speed straight replay. Allocate the base sampler to the
        # active curriculum/frontier and inject non-zero-yaw recovery anchors
        # separately so heading corrections are never out of distribution.
        self.commands.base_velocity.frontier_sampling_probability = 0.40
        self.commands.base_velocity.active_sampling_probability = 0.60
        self.commands.base_velocity.replay_sampling_probability = 0.0
        self.commands.base_velocity.yaw_recovery_sampling_probability = 0.15
        self.commands.base_velocity.yaw_recovery_forward_range = (0.3, 1.0)
        self.commands.base_velocity.yaw_recovery_abs_yaw_range = (0.2, 0.6)
        self.commands.base_velocity.terrain_conditioned_max_abs_vx = (
            STAIRS_PIE_MAX_ABS_VX_BY_LEVEL
        )

        # Open at EMA level 2, close only below 1.5, and grow the eligible
        # terrain ceiling as floor(EMA)+2 until all ten levels participate.
        self.commands.base_velocity.terrain_gate_min_mean_level = float(
            STAIRS_PIE_VELOCITY_GATE_MIN_MEAN_LEVEL
        )
        self.commands.base_velocity.terrain_gate_close_mean_level = (
            STAIRS_PIE_VELOCITY_GATE_CLOSE_MEAN_LEVEL
        )
        self.commands.base_velocity.terrain_level_ema_decay = 0.98
        self.commands.base_velocity.curriculum_update_level_margin = (
            STAIRS_PIE_VELOCITY_UPDATE_LEVEL_MARGIN
        )
        self.commands.base_velocity.curriculum_update_max_terrain_level = (
            STAIRS_PIE_VELOCITY_UPDATE_MAX_LEVEL
        )

        # Match the reference stair experiment: start near the center of the
        # 3 m platform and face its world-x traversal axis. Positive and
        # negative forward commands remain valid because both stair meshes
        # are symmetric about the center platform.
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
class AdaptiveEnergyStairsPIEPlayEnvCfg(AdaptiveEnergyStairsPIEEnvCfg):
    """Small fixed-level stair visualization/evaluation configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.scene.terrain.max_init_terrain_level = 0
        self.curriculum = None
        self.commands.base_velocity.terrain_gate_min_mean_level = None
        self.commands.base_velocity.curriculum_update_max_terrain_level = None
        self.events.push_robot = None
        self.observations.actor.enable_corruption = False
        self.observations.proprio_history.enable_corruption = False
