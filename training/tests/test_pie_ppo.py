"""Synthetic recurrent-rollout tests for PIEPPO."""

from __future__ import annotations

from importlib.metadata import version

import torch
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable

from unitree_rl_lab.tasks.locomotion.agents.pie_cfg import AdaptiveEnergyLPACRLPIERunnerCfg
from unitree_rl_lab.tasks.locomotion.agents.pie_ppo import PIEPPO

from test_pie_model import make_actor, make_observations


def make_algorithm(num_envs: int = 4, num_steps: int = 4) -> PIEPPO:
    observations = make_observations(num_envs)
    runner_cfg = handle_deprecated_rsl_rl_cfg(
        AdaptiveEnergyLPACRLPIERunnerCfg(),
        version("rsl-rl-lib"),
    )
    actor = make_actor(observations)
    critic_cfg = runner_cfg.critic.to_dict()
    critic_class = resolve_callable(critic_cfg.pop("class_name"))
    critic = critic_class(
        observations,
        runner_cfg.obs_groups,
        "critic",
        output_dim=1,
        **critic_cfg,
    )
    assert isinstance(critic, MLPModel)
    storage = RolloutStorage("rl", num_envs, num_steps, observations, [12], "cpu")

    algorithm_cfg = runner_cfg.algorithm.to_dict()
    algorithm_class = resolve_callable(algorithm_cfg.pop("class_name"))
    algorithm_cfg.pop("share_cnn_encoders")
    algorithm = algorithm_class(
        actor,
        critic,
        storage,
        device="cpu",
        multi_gpu_cfg=None,
        **algorithm_cfg,
    )
    assert isinstance(algorithm, PIEPPO)
    return algorithm


def test_process_env_step_writes_successor_and_terminal_mask() -> None:
    algorithm = make_algorithm()
    observations = make_observations(4)
    next_observations = make_observations(4)
    dones = torch.tensor([0, 1, 0, 1])

    algorithm.act(observations)
    algorithm.process_env_step(next_observations, torch.zeros(4), dones, {})

    torch.testing.assert_close(
        algorithm.storage.observations[0]["successor_target"],
        next_observations["successor_target"],
    )
    torch.testing.assert_close(
        algorithm.storage.observations[0]["successor_valid"],
        torch.tensor([[1.0], [0.0], [1.0], [0.0]]),
    )
    hidden = algorithm.actor.get_hidden_state()
    assert hidden is not None
    assert torch.count_nonzero(hidden[:, 1]) == 0
    assert torch.count_nonzero(hidden[:, 3]) == 0


def test_complete_recurrent_rollout_update() -> None:
    torch.manual_seed(5)
    num_envs, num_steps = 4, 4
    algorithm = make_algorithm(num_envs, num_steps)
    algorithm.num_learning_epochs = 1
    algorithm.num_mini_batches = 2
    observations = make_observations(num_envs)

    for step in range(num_steps):
        algorithm.act(observations)
        next_observations = make_observations(num_envs)
        dones = torch.zeros(num_envs, dtype=torch.long)
        if step == 1:
            dones[1] = 1
        rewards = torch.randn(num_envs)
        algorithm.process_env_step(next_observations, rewards, dones, {})
        observations = next_observations

    algorithm.compute_returns(observations)
    parameter_before = algorithm.actor.depth_encoder[0].weight.detach().clone()
    losses = algorithm.update()

    expected_losses = {
        "value",
        "surrogate",
        "entropy",
        "pie_auxiliary",
        "pie_velocity",
        "pie_foot_clearance",
        "pie_height_reconstruction",
        "pie_successor",
        "pie_kl",
        "pie_kl_total",
        "pie_kl_objective",
        "pie_kl_beta",
        "pie_kl_capacity",
        "pie_active_units",
        "pie_mu_std",
        "pie_posterior_std",
        "pie_height_zero_z_delta",
        "pie_successor_zero_z_delta",
        "pie_policy_output_zero_z_delta",
        "pie_vae_mu_grad_norm",
        "pie_successor_decoder_grad_norm",
        "pie_vae_mu_weight_norm",
    }
    assert set(losses) == expected_losses
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert algorithm.storage.step == 0
    assert not torch.equal(parameter_before, algorithm.actor.depth_encoder[0].weight)

    saved_state = algorithm.save()
    restored_algorithm = make_algorithm(num_envs, num_steps)
    restored_algorithm.load(
        saved_state,
        {"actor": True, "critic": True, "optimizer": True, "iteration": False, "rnd": False},
        strict=True,
    )
    torch.testing.assert_close(
        restored_algorithm.actor.depth_encoder[0].weight,
        algorithm.actor.depth_encoder[0].weight,
    )
    assert restored_algorithm.optimizer.state_dict()["state"]


def test_kl_capacity_schedule_and_checkpoint_state() -> None:
    algorithm = make_algorithm()
    assert algorithm._kl_schedule() == (0.0, 0.0)

    algorithm.pie_update_count = algorithm.kl_warmup_iterations + algorithm.kl_capacity_warmup_iterations // 2
    beta, capacity = algorithm._kl_schedule()
    assert beta == algorithm.kl_loss_coef * 0.5
    assert capacity == algorithm.kl_capacity_max * 0.5

    saved = algorithm.save()
    restored = make_algorithm()
    restored.load(
        saved,
        {"actor": True, "critic": True, "optimizer": True, "iteration": True, "rnd": False},
        strict=True,
    )
    assert restored.pie_update_count == algorithm.pie_update_count
