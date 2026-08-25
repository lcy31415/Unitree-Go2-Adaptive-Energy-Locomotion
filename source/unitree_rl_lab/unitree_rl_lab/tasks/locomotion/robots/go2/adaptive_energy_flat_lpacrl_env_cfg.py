"""Flat adaptive-energy task with a 300-task LP-ACRL velocity curriculum."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils.configclass import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .adaptive_energy_env_cfg import AdaptiveEnergyEnvCfg


@configclass
class LPACRLCommandsCfg:
    base_velocity = mdp.LPACRLVelocityCommandCfg(
        asset_name="robot",
        # Curriculum assignment occurs only at episode reset (20 s).
        resampling_time_range=(1.0e9, 1.0e9),
        rel_standing_envs=0.0,
        debug_vis=False,
        ranges=mdp.LPACRLVelocityCommandCfg.Ranges(
            lin_vel_x=(-5.0, 5.0),
            lin_vel_y=(-0.6, 0.6),
            ang_vel_z=(-5.0, 5.0),
        ),
    )


@configclass
class LPACRLCurriculumCfg:
    lp_acrl = CurrTerm(
        func=mdp.LPACRLCurriculum,
        params={
            "command_name": "base_velocity",
            "reward_terms": ("Rlin", "Rang", "Renergy", "adaptive_energy_residual"),
            "vx_edges": (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0),
            "vy_edges": (0.0, 0.2, 0.4, 0.6),
            "yaw_edges": (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0),
            # Resolve to the actual --num_envs value when the environment is constructed.
            "episodes_per_stage": None,
            "min_samples": 4,
            # Softmax temperature is max(beta, beta_scale * Q75(|LP|)): the
            # upper quartile tracks the scale of the strongest learning tasks
            # (the mean underestimates them), and beta is a small absolute
            # floor that relaxes sampling back to uniform at convergence.
            "beta": 0.002,
            "beta_scale": 0.5,
            "lp_quantile": 0.75,
            # Guards against winner-take-all concentration and stage-to-stage
            # thrash: no single task may exceed max_probability, and each
            # stage only blends probability_update_weight of the new target
            # into the previous distribution.
            "max_probability": 0.05,
            "probability_update_weight": 0.7,
            "epsilon": 0.1,
            "ema_alpha": 0.2,
            "planar_zero_threshold": 0.2,
        },
    )


@configclass
class AdaptiveEnergyFlatLPACRLEnvCfg(AdaptiveEnergyEnvCfg):
    """Training task; policy, critic, actions, rewards, and physics match Flat."""

    commands: LPACRLCommandsCfg = LPACRLCommandsCfg()
    curriculum: LPACRLCurriculumCfg = LPACRLCurriculumCfg()


@configclass
class AdaptiveEnergyFlatLPACRLPlayEnvCfg(AdaptiveEnergyFlatLPACRLEnvCfg):
    """Single-task visualization/evaluation variant."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.events.push_robot = None
