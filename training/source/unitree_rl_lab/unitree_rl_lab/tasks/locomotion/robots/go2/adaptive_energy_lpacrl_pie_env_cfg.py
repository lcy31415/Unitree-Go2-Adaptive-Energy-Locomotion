"""Perceptive PIE extension of the Go2 adaptive-energy LP-ACRL terrain task.

This task keeps the original terrain distribution, command curriculum,
rewards, events and terminations.  It replaces only the policy interface and
adds the ray-cast sensors required by PIE.
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from unitree_rl_lab.tasks.locomotion import mdp

from .adaptive_energy_terrain_lpacrl_env_cfg import (
    AdaptiveEnergyTerrainLPACRLEnvCfg,
    AdaptiveEnergyTerrainLPACRLSceneCfg,
)
from .pie_sensors_cfg import (
    PIE_DEPTH_CROP_LEFT,
    PIE_DEPTH_CROP_RIGHT,
    PIE_DEPTH_CUTOFF_DISTANCE,
    PIE_DEPTH_MIN_DISTANCE,
    PIE_FOOT_ORDER,
    PIE_FOOT_SENSOR_NAMES,
    make_pie_depth_camera_cfg,
    make_pie_foot_scanner_cfg,
    make_pie_height_scanner_cfg,
)


# The source PIE checkpoint and the Go2 SDK both use this joint order.  Regex
# resolution is deliberately avoided: USD/model traversal order is not an
# observation or action contract.
PIE_JOINT_NAMES = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)

PIE_ROBOT_JOINTS = SceneEntityCfg(
    "robot",
    joint_names=list(PIE_JOINT_NAMES),
    preserve_order=True,
)


@configclass
class PIEActionsCfg:
    """Original position action with an explicit checkpoint-compatible order."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(PIE_JOINT_NAMES),
        preserve_order=True,
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
    )


@configclass
class PIEActorObservationCfg(ObsGroup):
    """Noisy current proprioception, exactly 45 values."""

    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,
        clip=(-100.0, 100.0),
        noise=Unoise(n_min=-0.2, n_max=0.2),
    )
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        clip=(-100.0, 100.0),
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    velocity_commands = ObsTerm(
        func=mdp.generated_commands,
        clip=(-100.0, 100.0),
        params={"command_name": "base_velocity"},
    )
    joint_pos_rel = ObsTerm(
        func=mdp.joint_pos_rel,
        clip=(-100.0, 100.0),
        noise=Unoise(n_min=-0.01, n_max=0.01),
        params={"asset_cfg": PIE_ROBOT_JOINTS},
    )
    joint_vel_rel = ObsTerm(
        func=mdp.joint_vel_rel,
        clip=(-100.0, 100.0),
        noise=Unoise(n_min=-1.5, n_max=1.5),
        params={"asset_cfg": PIE_ROBOT_JOINTS},
    )
    last_action = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0))

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class PIEProprioHistoryObservationCfg(PIEActorObservationCfg):
    """Ten term-major frames of the same noisy 45-D proprioception."""

    def __post_init__(self):
        super().__post_init__()
        self.history_length = 10
        self.flatten_history_dim = True


@configclass
class PIECleanProprioObservationCfg(ObsGroup):
    """Noise-free 45-D proprioceptive target used by critic/auxiliary losses."""

    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0))
    projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0))
    velocity_commands = ObsTerm(
        func=mdp.generated_commands,
        clip=(-100.0, 100.0),
        params={"command_name": "base_velocity"},
    )
    joint_pos_rel = ObsTerm(
        func=mdp.joint_pos_rel,
        clip=(-100.0, 100.0),
        params={"asset_cfg": PIE_ROBOT_JOINTS},
    )
    joint_vel_rel = ObsTerm(
        func=mdp.joint_vel_rel,
        clip=(-100.0, 100.0),
        params={"asset_cfg": PIE_ROBOT_JOINTS},
    )
    last_action = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0))

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class PIECriticObservationCfg(PIECleanProprioObservationCfg):
    """Privileged critic input: proprioception + velocity + 18x11 terrain."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0))
    height_scan = ObsTerm(
        func=mdp.pie_height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner"), "max_height": 5.0},
        clip=(0.0, 1.0),
    )


@configclass
class PIECameraObservationCfg(ObsGroup):
    """Two distinct normalized 60x86 depth frames (10,320 values)."""

    depth_history = ObsTerm(
        func=mdp.PIEDepthHistory,
        params={
            "sensor_cfg": SceneEntityCfg("pie_depth_camera"),
            "cutoff_distance": PIE_DEPTH_CUTOFF_DISTANCE,
            "min_depth": PIE_DEPTH_MIN_DISTANCE,
            "crop_left": PIE_DEPTH_CROP_LEFT,
            "crop_right": PIE_DEPTH_CROP_RIGHT,
            "gaussian_blur": (3, 1.0),
            "frame_history_length": 2,
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class PIEVelocityTargetObservationCfg(ObsGroup):
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0))

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class PIEHeightTargetObservationCfg(ObsGroup):
    height_scan = ObsTerm(
        func=mdp.pie_height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner"), "max_height": 5.0},
        clip=(0.0, 1.0),
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class PIEFootClearanceTargetObservationCfg(ObsGroup):
    foot_clearance = ObsTerm(
        func=mdp.pie_foot_clearance,
        params={
            "sensor_cfgs": tuple(SceneEntityCfg(name) for name in PIE_FOOT_SENSOR_NAMES),
            "max_clearance": 0.6,
        },
        clip=(0.0, 1.0),
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class PIESuccessorValidObservationCfg(ObsGroup):
    valid = ObsTerm(func=mdp.pie_successor_valid, clip=(0.0, 1.0))

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class AdaptiveEnergyLPACRLPIEObservationsCfg:
    """All policy, critic and auxiliary-target groups consumed by PIEPPO."""

    actor: PIEActorObservationCfg = PIEActorObservationCfg()
    proprio_history: PIEProprioHistoryObservationCfg = PIEProprioHistoryObservationCfg()
    camera: PIECameraObservationCfg = PIECameraObservationCfg()
    critic: PIECriticObservationCfg = PIECriticObservationCfg()
    velocity_target: PIEVelocityTargetObservationCfg = PIEVelocityTargetObservationCfg()
    height_target: PIEHeightTargetObservationCfg = PIEHeightTargetObservationCfg()
    foot_clearance_target: PIEFootClearanceTargetObservationCfg = PIEFootClearanceTargetObservationCfg()
    successor_target: PIECleanProprioObservationCfg = PIECleanProprioObservationCfg()
    successor_valid: PIESuccessorValidObservationCfg = PIESuccessorValidObservationCfg()


@configclass
class AdaptiveEnergyLPACRLPIESceneCfg(AdaptiveEnergyTerrainLPACRLSceneCfg):
    """LP-ACRL terrain scene augmented with the PIE ray sensors."""

    # Replace the inherited 17x11 scanner with PIE's 18x11 privileged grid.
    height_scanner = make_pie_height_scanner_cfg()
    pie_depth_camera = make_pie_depth_camera_cfg()
    pie_fr_foot_scanner = make_pie_foot_scanner_cfg(PIE_FOOT_ORDER[0])
    pie_fl_foot_scanner = make_pie_foot_scanner_cfg(PIE_FOOT_ORDER[1])
    pie_rr_foot_scanner = make_pie_foot_scanner_cfg(PIE_FOOT_ORDER[2])
    pie_rl_foot_scanner = make_pie_foot_scanner_cfg(PIE_FOOT_ORDER[3])


@configclass
class AdaptiveEnergyLPACRLPIEEnvCfg(AdaptiveEnergyTerrainLPACRLEnvCfg):
    """Formal training environment for adaptive-energy LP-ACRL with PIE."""

    # Depth rollouts dominate memory; 256 is a safe default for an 8 GB GPU.
    # Users with more memory can override ``--num_envs`` from the train script.
    scene: AdaptiveEnergyLPACRLPIESceneCfg = AdaptiveEnergyLPACRLPIESceneCfg(
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
class AdaptiveEnergyLPACRLPIEPlayEnvCfg(AdaptiveEnergyLPACRLPIEEnvCfg):
    """Small evaluation environment without training-only disturbances."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.scene.terrain.max_init_terrain_level = 0
        self.events.push_robot = None
        self.rewards.standstill_penalty = None
        self.terminations.bad_standstill = None
        self.observations.actor.enable_corruption = False
        self.observations.proprio_history.enable_corruption = False
