"""Perceptive PIE extension of the flat adaptive-energy LP-ACRL task."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from .adaptive_energy_env_cfg import AdaptiveEnergySceneCfg
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

    def __post_init__(self):
        super().__post_init__()
        control_dt = self.decimation * self.sim.dt
        self.scene.height_scanner.update_period = control_dt
        self.scene.pie_depth_camera.update_period = 5 * control_dt
        for sensor_name in PIE_FOOT_SENSOR_NAMES:
            getattr(self.scene, sensor_name).update_period = control_dt


@configclass
class AdaptiveEnergyFlatLPACRLPIEPlayEnvCfg(AdaptiveEnergyFlatLPACRLPIEEnvCfg):
    """Small deterministic-observation variant for flat-ground inference."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.events.push_robot = None
        self.observations.actor.enable_corruption = False
        self.observations.proprio_history.enable_corruption = False
