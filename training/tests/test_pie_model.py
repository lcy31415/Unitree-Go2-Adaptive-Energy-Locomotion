"""Pure PyTorch tests for the PIE multi-modal recurrent actor."""

from __future__ import annotations

from importlib.metadata import version

import pytest
import torch
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict

from unitree_rl_lab.tasks.locomotion.agents.pie_cfg import AdaptiveEnergyLPACRLPIERunnerCfg
from unitree_rl_lab.tasks.locomotion.agents.pie_model import PIEActorModel


def make_observations(*batch_size: int) -> TensorDict:
    shape = tuple(batch_size)
    return TensorDict(
        {
            "actor": torch.randn(*shape, 45),
            "proprio_history": torch.randn(*shape, 450),
            "camera": torch.rand(*shape, 2 * 60 * 86),
            "critic": torch.randn(*shape, 246),
            "velocity_target": torch.randn(*shape, 3),
            "height_target": torch.randn(*shape, 198),
            "foot_clearance_target": torch.rand(*shape, 4),
            "successor_target": torch.randn(*shape, 45),
            "successor_valid": torch.ones(*shape, 1),
        },
        batch_size=shape,
    )


def make_actor(obs: TensorDict) -> PIEActorModel:
    runner_cfg = handle_deprecated_rsl_rl_cfg(
        AdaptiveEnergyLPACRLPIERunnerCfg(),
        version("rsl-rl-lib"),
    )
    actor_cfg = runner_cfg.actor.to_dict()
    actor_class = resolve_callable(actor_cfg.pop("class_name"))
    actor = actor_class(
        obs,
        runner_cfg.obs_groups,
        "actor",
        output_dim=12,
        **actor_cfg,
    )
    assert isinstance(actor, PIEActorModel)
    return actor


def test_configured_actor_path_resolves_and_forward_contract() -> None:
    obs = make_observations(4)
    actor = make_actor(obs).eval()

    first_actions = actor(obs)
    first_hidden = actor.get_hidden_state()
    assert first_actions.shape == (4, 12)
    assert first_hidden is not None
    assert first_hidden.shape == (1, 4, 128)
    assert torch.isfinite(first_actions).all()

    actor.reset()
    repeated_actions = actor(obs)
    torch.testing.assert_close(first_actions, repeated_actions)


def test_hidden_state_reset_is_per_environment() -> None:
    obs = make_observations(4)
    actor = make_actor(obs).eval()
    actor(obs)
    hidden_before_reset = actor.get_hidden_state().clone()  # type: ignore[union-attr]

    actor.reset(torch.tensor([False, True, False, True]))
    hidden_after_reset = actor.get_hidden_state()
    assert hidden_after_reset is not None
    torch.testing.assert_close(hidden_after_reset[:, 0], hidden_before_reset[:, 0])
    torch.testing.assert_close(hidden_after_reset[:, 2], hidden_before_reset[:, 2])
    assert torch.count_nonzero(hidden_after_reset[:, 1]) == 0
    assert torch.count_nonzero(hidden_after_reset[:, 3]) == 0


def test_stochastic_distribution_contract() -> None:
    obs = make_observations(3)
    actor = make_actor(obs).train()

    actions = actor(obs, stochastic_output=True)
    assert actions.shape == (3, 12)
    assert actor.output_mean.shape == (3, 12)
    assert actor.output_std.shape == (3, 12)
    assert actor.output_entropy.shape == (3,)
    assert actor.get_output_log_prob(actions).shape == (3,)
    assert all(parameter.shape == (3, 12) for parameter in actor.output_distribution_params)


def test_auxiliary_outputs_losses_and_gradients() -> None:
    torch.manual_seed(7)
    obs = make_observations(2)
    actor = make_actor(obs).train()

    outputs = actor.auxiliary_outputs(obs, sample_latent=False)
    assert outputs["velocity"].shape == (2, 3)
    assert outputs["foot_clearance"].shape == (2, 4)
    assert outputs["mu"].shape == (2, 16)
    assert outputs["logvar"].shape == (2, 16)
    assert outputs["z"].shape == (2, 16)
    assert outputs["policy_latent"].shape == (2, 23)
    assert outputs["latent"].shape == (2, 23)
    assert outputs["height_reconstruction"].shape == (2, 198)
    assert outputs["successor_prediction"].shape == (2, 45)

    losses = actor.auxiliary_losses(obs)
    assert set(losses) == {
        "velocity",
        "foot_clearance",
        "height_reconstruction",
        "successor",
        "kl",
        "kl_total",
        "active_units",
        "mu_std",
        "posterior_std",
        "height_zero_z_delta",
        "successor_zero_z_delta",
        "policy_output_zero_z_delta",
    }
    assert all(loss.ndim == 0 and torch.isfinite(loss) for loss in losses.values())
    sum(losses.values()).backward()
    assert actor.depth_encoder[0].weight.grad is not None  # type: ignore[index,union-attr]
    assert torch.isfinite(actor.depth_encoder[0].weight.grad).all()  # type: ignore[index,union-attr]
    assert actor.memory.weight_ih_l0.grad is not None
    assert torch.isfinite(actor.memory.weight_ih_l0.grad).all()


def test_vae_latent_is_the_only_auxiliary_decoder_input() -> None:
    actor = make_actor(make_observations(2))

    successor_first_linear = next(
        module for module in actor.successor_decoder.modules() if isinstance(module, torch.nn.Linear)
    )
    height_first_linear = next(
        module for module in actor.height_decoder.modules() if isinstance(module, torch.nn.Linear)
    )
    assert successor_first_linear.in_features == actor.vae_latent_dim == 16
    assert height_first_linear.in_features == actor.vae_latent_dim
    assert not hasattr(actor, "map_latent_head")


def test_reconstruction_objectives_train_the_posterior() -> None:
    torch.manual_seed(13)
    obs = make_observations(4)
    actor = make_actor(obs).train()

    losses = actor.auxiliary_losses(obs)
    (losses["height_reconstruction"] + losses["successor"]).backward()

    assert actor.vae_mu_head.weight.grad is not None
    assert actor.vae_logvar_head.weight.grad is not None
    assert actor.vae_mu_head.weight.grad.norm() > 0
    assert actor.vae_logvar_head.weight.grad.norm() > 0


def test_policy_uses_posterior_mean_even_in_training_mode() -> None:
    obs = make_observations(2)
    actor = make_actor(obs).train()

    actor.reset()
    first = actor(obs)
    actor.reset()
    second = actor(obs)
    torch.testing.assert_close(first, second)


def test_successor_loss_ignores_invalid_transitions() -> None:
    obs = make_observations(2)
    obs["successor_valid"][1] = 0.0
    actor = make_actor(obs).train()

    torch.manual_seed(11)
    reference_loss = actor.auxiliary_losses(obs)["successor"]
    obs["successor_target"][1] = 1.0e6
    torch.manual_seed(11)
    changed_loss = actor.auxiliary_losses(obs)["successor"]
    torch.testing.assert_close(reference_loss, changed_loss)


def test_recurrent_rollout_interface() -> None:
    time_steps, trajectories = 3, 2
    obs = make_observations(time_steps, trajectories)
    actor = make_actor(make_observations(trajectories)).eval()
    masks = torch.ones(time_steps, trajectories, dtype=torch.bool)
    hidden = torch.zeros(1, trajectories, 128)

    actions = actor(obs, masks=masks, hidden_state=hidden)
    losses = actor.auxiliary_losses(obs, masks=masks, hidden_state=hidden)
    assert actions.shape == (time_steps, trajectories, 12)
    assert all(torch.isfinite(loss) for loss in losses.values())

    with pytest.raises(ValueError, match="initial hidden_state"):
        actor(obs, masks=masks)


def test_export_wrapper_contract_and_parity() -> None:
    obs = make_observations(1)
    actor = make_actor(obs).eval()
    export_model = actor.as_onnx()
    inputs = export_model.get_dummy_inputs()  # type: ignore[attr-defined]

    actions, hidden = export_model(*inputs)
    assert actions.shape == (1, 12)
    assert hidden.shape == (1, 1, 128)
    assert export_model.input_names == [  # type: ignore[attr-defined]
        "proprio",
        "proprio_history",
        "depth_history",
        "memory_h_in",
    ]
    assert export_model.output_names == ["actions", "memory_h_out"]  # type: ignore[attr-defined]

    comparison_obs = TensorDict(
        {
            **obs.to_dict(),
            "actor": inputs[0],
            "proprio_history": inputs[1],
            "camera": inputs[2].flatten(start_dim=1),
        },
        batch_size=(1,),
    )
    actor.reset(hidden_state=inputs[3])
    torch.testing.assert_close(actions, actor(comparison_obs))

    scripted_model = torch.jit.script(export_model)
    scripted_actions, scripted_hidden = scripted_model(*inputs)
    torch.testing.assert_close(scripted_actions, actions)
    torch.testing.assert_close(scripted_hidden, hidden)


@pytest.mark.parametrize(
    ("group", "dimension", "message"),
    [
        ("actor", 44, "expects 45"),
        ("proprio_history", 449, "expects 10x45=450"),
        ("camera", 100, "depth group has 100"),
        ("height_target", 197, "height_target=197"),
        ("successor_valid", 2, "successor_valid=2"),
    ],
)
def test_invalid_observation_dimensions_fail_early(group: str, dimension: int, message: str) -> None:
    obs = make_observations(2)
    obs.set(group, torch.zeros(2, dimension))
    with pytest.raises(ValueError, match=message):
        make_actor(obs)
