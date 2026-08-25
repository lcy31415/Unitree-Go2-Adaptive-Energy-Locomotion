"""600-task LP-ACRL rough-terrain experiment with the flat-task guard set."""

import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .adaptive_energy_env_cfg import AdaptiveEnergyEnvCfg, AdaptiveEnergyRewardsCfg
from .adaptive_energy_lpacrl_terrain_cfg import (
    LPACRL_COLUMNS_PER_TYPE,
    LPACRL_TERRAIN_NAMES,
    LPACRL_TERRAINS_CFG,
)
from .adaptive_energy_terrain_env_cfg import AdaptiveEnergyTerrainObservationsCfg, AdaptiveEnergyTerrainSceneCfg
from .velocity_env_cfg import TerminationsCfg


@configclass
class AdaptiveEnergyTerrainLPACRLSceneCfg(AdaptiveEnergyTerrainSceneCfg):
    """Six terrain families (flat plus five rough), four fixed geometry levels, four columns each."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=LPACRL_TERRAINS_CFG,
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


@configclass
class AdaptiveEnergyTerrainLPACRLCommandsCfg:
    base_velocity = mdp.LPACRLVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        rel_standing_envs=0.0,
        debug_vis=False,
        ranges=mdp.LPACRLVelocityCommandCfg.Ranges(
            lin_vel_x=(-2.5, 2.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-2.5, 2.5),
        ),
    )


@configclass
class AdaptiveEnergyTerrainLPACRLCurriculumCfg:
    """One unified task sampler; no terrain promotion or hand-written stages."""

    lp_acrl = CurrTerm(
        func=mdp.TerrainLPACRLCurriculum,
        params={
            "command_name": "base_velocity",
            "reward_terms": ("Rlin", "Rang", "Renergy", "adaptive_energy_residual"),
            "vx_edges": (0.0, 0.5, 1.0, 1.5, 2.0, 2.5),
            "yaw_edges": (0.0, 0.5, 1.0, 1.5, 2.0, 2.5),
            "terrain_names": LPACRL_TERRAIN_NAMES,
            "num_levels": 4,
            "columns_per_type": LPACRL_COLUMNS_PER_TYPE,
            "lateral_range": (0.0, 0.0),
            # Sample density: 4096 episodes over 600 tasks gives ~7 episodes
            # per task per stage, so min_samples=4 still lets most tasks
            # qualify for LP updates each stage.
            "episodes_per_stage": 4096,
            "min_samples": 4,
            # Guard set converged on the flat LP-ACRL task: temperature
            # max(beta, beta_scale * Q75(|LP|)) with the floor near the LP
            # noise level (~0.003 at ~7 episodes/task/stage), a 5% single-task
            # cap against winner-take-all, and a 0.7 update weight so the
            # distribution tracks the early LP burst instead of freezing.
            "beta": 0.003,
            "beta_scale": 0.5,
            "lp_quantile": 0.75,
            "max_probability": 0.05,
            "probability_update_weight": 0.7,
            "epsilon": 0.10,
            "ema_alpha": 0.20,
            # Keep the first [0.0, 0.5) bin continuous as specified by the
            # LP-ACRL task definition; do not collapse small vx samples to 0.
            "planar_zero_threshold": 0.0,
        },
    )


@configclass
class AdaptiveEnergyTerrainLPACRLRewardsCfg(AdaptiveEnergyRewardsCfg):
    """Adaptive-energy rewards plus the anti-standing guard.

    Without this term a robot that ignores commands above ~1.8 m/s and
    stands still still collects the yaw and energy terms (~0.4/step) while
    avoiding falls, which collapsed the policy onto a standing solution.
    """

    standstill_penalty = RewTerm(
        func=mdp.standstill_penalty,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "min_command": 1.5,
            "ratio": 0.3,
            "grace_s": 2.0,
            "sustain_s": 0.5,
            "penalty": 0.5,
        },
    )


@configclass
class AdaptiveEnergyTerrainLPACRLTerminationsCfg(TerminationsCfg):
    """Standard terminations plus truncating persistent standing."""

    bad_standstill = DoneTerm(
        func=mdp.bad_standstill,
        time_out=False,
        params={
            "command_name": "base_velocity",
            "min_command": 1.5,
            "ratio": 0.3,
            "grace_s": 2.0,
            "sustain_s": 3.0,
        },
    )


@configclass
class AdaptiveEnergyTerrainLPACRLEnvCfg(AdaptiveEnergyEnvCfg):
    scene: AdaptiveEnergyTerrainLPACRLSceneCfg = AdaptiveEnergyTerrainLPACRLSceneCfg(
        num_envs=2048, env_spacing=2.5
    )
    observations: AdaptiveEnergyTerrainObservationsCfg = AdaptiveEnergyTerrainObservationsCfg()
    commands: AdaptiveEnergyTerrainLPACRLCommandsCfg = AdaptiveEnergyTerrainLPACRLCommandsCfg()
    rewards: AdaptiveEnergyTerrainLPACRLRewardsCfg = AdaptiveEnergyTerrainLPACRLRewardsCfg()
    terminations: AdaptiveEnergyTerrainLPACRLTerminationsCfg = AdaptiveEnergyTerrainLPACRLTerminationsCfg()
    curriculum: AdaptiveEnergyTerrainLPACRLCurriculumCfg = AdaptiveEnergyTerrainLPACRLCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt


@configclass
class AdaptiveEnergyTerrainLPACRLPlayEnvCfg(AdaptiveEnergyTerrainLPACRLEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.events.push_robot = None
        # Evaluations must observe the raw behavior: no penalty income and no
        # truncation from the anti-standing guard.
        self.rewards.standstill_penalty = None
        self.terminations.bad_standstill = None
