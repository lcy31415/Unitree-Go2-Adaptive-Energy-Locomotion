"""Perceptive PIE extension of the flat adaptive-energy LP-ACRL task."""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .adaptive_energy_env_cfg import AdaptiveEnergyRewardsCfg, AdaptiveEnergySceneCfg
from .adaptive_energy_flat_lpacrl_env_cfg import AdaptiveEnergyFlatLPACRLEnvCfg
from .adaptive_energy_lpacrl_pie_env_cfg import (
    AdaptiveEnergyLPACRLPIEObservationsCfg,
    PIEActionsCfg,
)
from .pie_sensors_cfg import (
    PIE_FOOT_ORDER,
    PIE_FOOT_SENSOR_NAMES,
    make_pie_depth_camera_cfg,
    make_pie_foot_scanner_cfg,
    make_pie_height_scanner_cfg,
)


@configclass
class AdaptiveEnergyFlatLPACRLPIESceneCfg(AdaptiveEnergySceneCfg):
    """Infinite plane augmented with all sensors consumed by PIE."""

    height_scanner = make_pie_height_scanner_cfg()
    pie_depth_camera = make_pie_depth_camera_cfg()
    pie_fr_foot_scanner = make_pie_foot_scanner_cfg(PIE_FOOT_ORDER[0])
    pie_fl_foot_scanner = make_pie_foot_scanner_cfg(PIE_FOOT_ORDER[1])
    pie_rr_foot_scanner = make_pie_foot_scanner_cfg(PIE_FOOT_ORDER[2])
    pie_rl_foot_scanner = make_pie_foot_scanner_cfg(PIE_FOOT_ORDER[3])


@configclass
class AdaptiveEnergyFlatLPACRLPIERewardsCfg(AdaptiveEnergyRewardsCfg):
    """Base objective plus targeted stability for exact straight commands."""

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
class AdaptiveEnergyFlatLPACRLPIEEnvCfg(AdaptiveEnergyFlatLPACRLEnvCfg):
    """Flat LP-ACRL dynamics and curriculum with the complete PIE interface."""

    # Depth-history rollouts dominate memory; keep the same safe default as
    # the rough-terrain PIE task and override with --num_envs when appropriate.
    scene: AdaptiveEnergyFlatLPACRLPIESceneCfg = AdaptiveEnergyFlatLPACRLPIESceneCfg(
        num_envs=256,
        env_spacing=2.5,
    )
    observations: AdaptiveEnergyLPACRLPIEObservationsCfg = AdaptiveEnergyLPACRLPIEObservationsCfg()
    actions: PIEActionsCfg = PIEActionsCfg()
    rewards: AdaptiveEnergyFlatLPACRLPIERewardsCfg = AdaptiveEnergyFlatLPACRLPIERewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        control_dt = self.decimation * self.sim.dt
        self.scene.height_scanner.update_period = control_dt
        self.scene.pie_depth_camera.update_period = 5 * control_dt
        for sensor_name in PIE_FOOT_SENSOR_NAMES:
            getattr(self.scene, sensor_name).update_period = control_dt
        # Reserve a stable fraction of LP-ACRL assignments for exact straight
        # commands.  The remaining episodes retain the original 300-task
        # learning-progress distribution and full turning capability.
        self.curriculum.lp_acrl.params["straight_task_probability"] = 0.30
        self.curriculum.lp_acrl.params["reward_terms"] = (
            *self.curriculum.lp_acrl.params["reward_terms"],
            "straight_yaw_rate_error",
        )


@configclass
class AdaptiveEnergyFlatLPACRLPIEPlayEnvCfg(AdaptiveEnergyFlatLPACRLPIEEnvCfg):
    """Small deterministic-observation variant for flat-ground inference."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.events.push_robot = None
        self.observations.actor.enable_corruption = False
        self.observations.proprio_history.enable_corruption = False
