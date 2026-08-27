"""PIE perception on the original adaptive-energy flat velocity curriculum."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from .adaptive_energy_env_cfg import AdaptiveEnergyEnvCfg, AdaptiveEnergySceneCfg
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
class AdaptiveEnergyFlatPIESceneCfg(AdaptiveEnergySceneCfg):
    """Infinite plane augmented with every sensor consumed by PIE."""

    height_scanner = make_pie_height_scanner_cfg()
    pie_depth_camera = make_pie_depth_camera_cfg()
    pie_fr_foot_scanner = make_pie_foot_scanner_cfg(PIE_FOOT_ORDER[0])
    pie_fl_foot_scanner = make_pie_foot_scanner_cfg(PIE_FOOT_ORDER[1])
    pie_rr_foot_scanner = make_pie_foot_scanner_cfg(PIE_FOOT_ORDER[2])
    pie_rl_foot_scanner = make_pie_foot_scanner_cfg(PIE_FOOT_ORDER[3])


@configclass
class AdaptiveEnergyFlatPIEEnvCfg(AdaptiveEnergyEnvCfg):
    """Original flat adaptive-energy curriculum with the complete PIE interface."""

    scene: AdaptiveEnergyFlatPIESceneCfg = AdaptiveEnergyFlatPIESceneCfg(
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
class AdaptiveEnergyFlatPIEPlayEnvCfg(AdaptiveEnergyFlatPIEEnvCfg):
    """Small clean-observation variant for flat PIE inference."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 5.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.6, 0.6)
        self.commands.base_velocity.ranges.ang_vel_z = (-5.0, 5.0)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.0, 5.0)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (-0.6, 0.6)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-5.0, 5.0)
        self.events.push_robot = None
        self.observations.actor.enable_corruption = False
        self.observations.proprio_history.enable_corruption = False
