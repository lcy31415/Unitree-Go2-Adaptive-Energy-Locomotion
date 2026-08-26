"""RSL-RL configuration for the adaptive-energy LP-ACRL PIE policy.

The custom actor and PPO classes referenced here are implemented in later
integration steps.  Keeping their import paths in this module makes the
observation and optimization contracts explicit before the environment is
wired to the model.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import Any

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


PIE_ACTOR_CLASS = "unitree_rl_lab.tasks.locomotion.agents.pie_model:PIEActorModel"
PIE_PPO_CLASS = "unitree_rl_lab.tasks.locomotion.agents.pie_ppo:PIEPPO"


@configclass
class PIEActorCfg(RslRlMLPModelCfg):
    """Configuration for the multi-input recurrent PIE actor."""

    class_name: str = PIE_ACTOR_CLASS
    cnn_cfg: dict[str, Any] = MISSING


@configclass
class PIEPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO configuration extended with PIE auxiliary objectives."""

    class_name: str = PIE_PPO_CLASS
    auxiliary_loss_coef: float = 1.0
    velocity_loss_coef: float = 1.0
    foot_clearance_loss_coef: float = 1.0
    height_reconstruction_loss_coef: float = 1.0
    successor_loss_coef: float = 1.0
    kl_loss_coef: float = 0.01
    kl_warmup_iterations: int = 500
    kl_capacity_warmup_iterations: int = 2500
    kl_capacity_max: float = 2.0
    successor_target_group: str = "successor_target"
    successor_valid_group: str = "successor_valid"


@configclass
class AdaptiveEnergyLPACRLPIERunnerCfg(RslRlOnPolicyRunnerCfg):
    """Training settings for Unitree-Go2-Adaptive-Energy-LPACRL-PIE."""

    num_steps_per_env = 24
    max_iterations = 50_000
    save_interval = 500
    experiment_name = "unitree_go2_adaptive_energy_lpacrl_pie"
    empirical_normalization = False

    # RSL-RL observation sets. Auxiliary targets remain in the environment
    # TensorDict for PIEPPO but are intentionally excluded from actor inputs.
    obs_groups = {
        "actor": ["actor", "proprio_history", "camera"],
        "critic": ["critic"],
    }

    actor = PIEActorCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_type="scalar",
        ),
        cnn_cfg={
            "proprio_group": "actor",
            "history_group": "proprio_history",
            "depth_group": "camera",
            "velocity_target_group": "velocity_target",
            "height_target_group": "height_target",
            "foot_target_group": "foot_clearance_target",
            "successor_target_group": "successor_target",
            "successor_valid_group": "successor_valid",
            "history_length": 10,
            "depth_shape": (2, 60, 86),
            "token_dim": 64,
            "attention_heads": 1,
            "transformer_layers": 2,
            "transformer_ff_dim": 256,
            "transformer_dropout": 0.0,
            "memory_hidden_dim": 128,
            "memory_num_layers": 1,
            "vae_latent_dim": 16,
            "history_hidden_dims": (256, 128),
            "successor_decoder_dims": (64, 128),
            "height_decoder_dims": (64, 128),
            "output_channels": (32, 64, 64),
            "kernel_size": (8, 4, 3),
            "stride": (4, 2, 1),
            "padding": 0,
            "cnn_activation": "elu",
        },
    )

    critic = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=None,
    )

    algorithm = PIEPPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        auxiliary_loss_coef=1.0,
        velocity_loss_coef=1.0,
        foot_clearance_loss_coef=1.0,
        height_reconstruction_loss_coef=1.0,
        successor_loss_coef=1.0,
        kl_loss_coef=0.01,
        kl_warmup_iterations=500,
        kl_capacity_warmup_iterations=2500,
        kl_capacity_max=2.0,
        successor_target_group="successor_target",
        successor_valid_group="successor_valid",
    )


@configclass
class AdaptiveEnergyPIEStairsRunnerCfg(AdaptiveEnergyLPACRLPIERunnerCfg):
    """PIE training settings for the focused up/down-stair experiment."""

    max_iterations = 20_000
    save_interval = 250
    experiment_name = "unitree_go2_adaptive_energy_pie_stairs"
