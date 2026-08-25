"""Gait-free flat-ground task for adaptive energy regularization experiments.

The actor receives no gait clock, phase, contact schedule, terrain scan, or
foot-trajectory target. Locomotion patterns must therefore emerge from the
velocity-tracking objective and physical regularization alone.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_physx.physics import PhysxCfg

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg import ActionsCfg, EventCfg, TerminationsCfg


@configclass
class AdaptiveEnergySceneCfg(InteractiveSceneCfg):
    """Go2 on a single effectively infinite ground plane."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        env_spacing=2.5,
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

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Contacts are used only for collision safety and slip regularization. Air
    # time is deliberately disabled because it is a gait-shaping signal.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=False,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class AdaptiveEnergyCommandsCfg:
    """Single gait-free reward-threshold curriculum over three-axis velocity."""

    base_velocity = mdp.RewardThresholdVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        debug_vis=True,
        ranges=mdp.RewardThresholdVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
        limit_ranges=mdp.RewardThresholdVelocityCommandCfg.Ranges(
            lin_vel_x=(-5.0, 5.0),
            lin_vel_y=(-0.6, 0.6),
            ang_vel_z=(-5.0, 5.0),
        ),
        # Staged curriculum: first expand vx with vy=yaw=0, then expand yaw
        # with vy=0, and finally expand vy. Odd bin counts provide a true zero
        # center on every command axis.
        num_bins=(21, 13, 21),
        local_range=(0.55, 0.10, 0.55),
        weight_increment=0.2,
        frontier_sampling_probability=0.4,
        active_sampling_probability=0.4,
        replay_sampling_probability=0.2,
        frontier_bin_count=2,
        forward_error_abs=0.1,
        forward_error_rel=0.033,
        lateral_error_abs=0.15,
        lateral_error_rel=0.1,
        angular_error_abs=0.2,
        angular_error_rel=0.1,
        minimum_segment_fraction=0.95,
        linear_stage_threshold=2.5,
        angular_stage_threshold=2.5,
        stage_transition_successes=20,
        zero_command_threshold=0.2,
    )


@configclass
class AdaptiveEnergyObservationsCfg:
    """Actor history and privileged critic observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Exactly 45 values per frame: 3 + 3 + 3 + 12 + 12 + 12.
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            clip=(-100, 100),
            params={"command_name": "base_velocity"},
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            clip=(-100, 100),
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            self.history_length = 30
            self.flatten_history_dim = True
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            clip=(-100, 100),
            params={"command_name": "base_velocity"},
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


@configclass
class AdaptiveEnergyRewardsCfg:
    """Reward from the released adaptive-energy implementation.

    Tracking and energy terms sum with the attenuation residual to recover the
    original nonlinear reward. Curriculum success remains independently based
    on physical velocity errors over each 10-second command segment.
    """

    Rlin = RewTerm(
        func=mdp.adaptive_energy_tracking_lin_vel,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "tracking_sigma": 0.25,
            "transition_speed": 1.7,
            "energy_weight_decay": 0.3,
        },
    )
    Rang = RewTerm(
        func=mdp.adaptive_energy_tracking_ang_vel,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "tracking_sigma_yaw": 0.25,
            "transition_speed": 1.7,
            "energy_weight_decay": 0.3,
        },
    )
    Renergy = RewTerm(
        func=mdp.adaptive_energy_reward,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "energy_sigma_lin": 1000.0,
            "energy_sigma_ang": 500.0,
            "energy_clip_lin": 0.2,
            "energy_clip_rot": 0.2,
            "transition_speed": 1.7,
            "energy_weight_decay": 0.3,
        },
    )
    adaptive_energy_residual = RewTerm(
        func=mdp.adaptive_energy_reward_residual,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "feet_asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "collision_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[".*_thigh", ".*_calf"]
            ),
            "tracking_sigma": 0.25,
            "tracking_sigma_yaw": 0.25,
            "energy_sigma_lin": 1000.0,
            "energy_sigma_ang": 500.0,
            "energy_clip_lin": 0.2,
            "energy_clip_rot": 0.2,
            "transition_speed": 1.7,
            "energy_weight_decay": 0.3,
            "sigma_rew_neg": 0.02,
        },
    )


@configclass
class AdaptiveEnergyEnvCfg(ManagerBasedRLEnvCfg):
    """Training configuration for gait-free adaptive-energy locomotion."""

    scene: AdaptiveEnergySceneCfg = AdaptiveEnergySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: AdaptiveEnergyObservationsCfg = AdaptiveEnergyObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: AdaptiveEnergyCommandsCfg = AdaptiveEnergyCommandsCfg()
    rewards: AdaptiveEnergyRewardsCfg = AdaptiveEnergyRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    # The command generator owns the 10-second curriculum update. Isaac Lab's
    # CurriculumManager is reset-driven and therefore cannot reproduce the
    # reference command-segment timing.
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        if self.sim.physics is None:
            self.sim.physics = PhysxCfg()
        self.sim.physics.gpu_max_rigid_patch_count = 10 * 2**15

        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class AdaptiveEnergyPlayEnvCfg(AdaptiveEnergyEnvCfg):
    """Positive-forward-speed evaluation configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 5.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.6, 0.6)
        self.commands.base_velocity.ranges.ang_vel_z = (-5.0, 5.0)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.0, 5.0)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (-0.6, 0.6)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-5.0, 5.0)
        self.events.push_robot = None
