"""Configuration-contract tests for the LP-ACRL PIE integration."""

from unitree_rl_lab.tasks.locomotion.agents.pie_cfg import (
    PIE_ACTOR_CLASS,
    PIE_PPO_CLASS,
    AdaptiveEnergyLPACRLPIERunnerCfg,
)


def test_pie_runner_contract() -> None:
    cfg = AdaptiveEnergyLPACRLPIERunnerCfg()

    assert cfg.experiment_name == "unitree_go2_adaptive_energy_lpacrl_pie"
    assert cfg.num_steps_per_env == 24
    assert cfg.max_iterations == 50_000
    assert cfg.save_interval == 500
    assert cfg.obs_groups == {
        "actor": ["actor", "proprio_history", "camera"],
        "critic": ["critic"],
    }

    assert cfg.actor.class_name == PIE_ACTOR_CLASS
    assert cfg.actor.hidden_dims == [512, 256, 128]
    assert cfg.actor.obs_normalization is True
    assert cfg.actor.distribution_cfg.init_std == 1.0
    assert cfg.actor.distribution_cfg.std_type == "scalar"
    assert cfg.actor.cnn_cfg["history_length"] == 10
    assert cfg.actor.cnn_cfg["depth_shape"] == (2, 60, 86)
    assert cfg.actor.cnn_cfg["memory_hidden_dim"] == 128
    assert cfg.actor.cnn_cfg["successor_valid_group"] == "successor_valid"

    assert cfg.critic.class_name == "MLPModel"
    assert cfg.critic.hidden_dims == [512, 256, 128]
    assert cfg.critic.distribution_cfg is None

    assert cfg.algorithm.class_name == PIE_PPO_CLASS
    assert cfg.algorithm.auxiliary_loss_coef == 1.0
    assert cfg.algorithm.velocity_loss_coef == 1.0
    assert cfg.algorithm.foot_clearance_loss_coef == 1.0
    assert cfg.algorithm.height_reconstruction_loss_coef == 1.0
    assert cfg.algorithm.successor_loss_coef == 1.0
    assert cfg.algorithm.kl_loss_coef == 0.01
    assert cfg.algorithm.kl_warmup_iterations == 500
    assert cfg.algorithm.kl_capacity_warmup_iterations == 2500
    assert cfg.algorithm.kl_capacity_max == 2.0
    assert cfg.algorithm.successor_target_group == "successor_target"
    assert cfg.algorithm.successor_valid_group == "successor_valid"


def test_pie_runner_instances_do_not_share_mutable_config() -> None:
    first = AdaptiveEnergyLPACRLPIERunnerCfg()
    second = AdaptiveEnergyLPACRLPIERunnerCfg()

    first.actor.cnn_cfg["history_length"] = 99
    first.obs_groups["actor"].append("unexpected")

    assert second.actor.cnn_cfg["history_length"] == 10
    assert second.obs_groups["actor"] == ["actor", "proprio_history", "camera"]
