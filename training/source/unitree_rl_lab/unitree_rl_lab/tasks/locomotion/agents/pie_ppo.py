"""PPO extension for perceptive implicit-explicit estimator objectives."""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.algorithms.ppo import PPO
from tensordict import TensorDict

from .pie_model import PIEActorModel


class PIEPPO(PPO):
    """Optimize PPO and collapse-resistant PIE estimator objectives jointly."""

    def __init__(
        self,
        *args,
        auxiliary_loss_coef: float = 1.0,
        velocity_loss_coef: float = 1.0,
        foot_clearance_loss_coef: float = 1.0,
        height_reconstruction_loss_coef: float = 1.0,
        successor_loss_coef: float = 1.0,
        kl_loss_coef: float = 0.01,
        kl_warmup_iterations: int = 500,
        kl_capacity_warmup_iterations: int = 2500,
        kl_capacity_max: float = 2.0,
        successor_target_group: str = "successor_target",
        successor_valid_group: str = "successor_valid",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self.actor, PIEActorModel):
            raise TypeError(f"PIEPPO requires PIEActorModel, got {type(self.actor).__name__}.")
        if self.symmetry is not None:
            raise ValueError("PIEPPO does not support symmetry augmentation for its recurrent actor.")

        self.auxiliary_loss_coef = float(auxiliary_loss_coef)
        self.auxiliary_coefficients = {
            "velocity": float(velocity_loss_coef),
            "foot_clearance": float(foot_clearance_loss_coef),
            "height_reconstruction": float(height_reconstruction_loss_coef),
            "successor": float(successor_loss_coef),
        }
        self.kl_loss_coef = float(kl_loss_coef)
        self.kl_warmup_iterations = int(kl_warmup_iterations)
        self.kl_capacity_warmup_iterations = int(kl_capacity_warmup_iterations)
        self.kl_capacity_max = float(kl_capacity_max)
        self.pie_update_count = 0
        if self.kl_loss_coef < 0.0:
            raise ValueError("PIE kl_loss_coef must be non-negative.")
        if self.kl_warmup_iterations < 0 or self.kl_capacity_warmup_iterations <= 0:
            raise ValueError("PIE KL warm-up durations must be non-negative and non-zero, respectively.")
        if self.kl_capacity_max < 0.0:
            raise ValueError("PIE kl_capacity_max must be non-negative.")
        self.successor_target_group = successor_target_group
        self.successor_valid_group = successor_valid_group

        if self.successor_target_group != self.actor.successor_target_group:
            raise ValueError(
                "PIEPPO and PIEActorModel successor target groups differ: "
                f"{self.successor_target_group!r} != {self.actor.successor_target_group!r}."
            )
        if self.successor_valid_group != self.actor.successor_valid_group:
            raise ValueError(
                "PIEPPO and PIEActorModel successor validity groups differ: "
                f"{self.successor_valid_group!r} != {self.actor.successor_valid_group!r}."
            )

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Collect recurrent transitions without retaining the rollout graph."""
        with torch.no_grad():
            return super().act(obs)

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
    ) -> None:
        """Store the clean next proprioception and mask auto-reset transitions."""
        if self.transition.observations is None:
            raise RuntimeError("PIEPPO.process_env_step() was called before act().")
        for group in (self.successor_target_group, self.successor_valid_group):
            if group not in self.transition.observations:
                raise KeyError(f"PIEPPO rollout observation is missing required group {group!r}.")
        if self.successor_target_group not in obs:
            raise KeyError(f"PIEPPO next observation is missing group {self.successor_target_group!r}.")

        self.transition.observations[self.successor_target_group] = obs[
            self.successor_target_group
        ].detach()
        successor_valid = (~dones.reshape(-1).to(dtype=torch.bool)).to(
            device=self.transition.observations.device,
            dtype=self.transition.observations[self.successor_valid_group].dtype,
        )
        self.transition.observations[self.successor_valid_group] = successor_valid.unsqueeze(-1)
        super().process_env_step(obs, rewards, dones, extras)

    def update(self) -> dict[str, float]:
        """Run recurrent PPO updates with PIE auxiliary supervision."""
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_weighted_auxiliary = 0.0
        metric_names = (
            *self.auxiliary_coefficients,
            "kl",
            "kl_total",
            "kl_objective",
            "active_units",
            "mu_std",
            "posterior_std",
            "height_zero_z_delta",
            "successor_zero_z_delta",
            "policy_output_zero_z_delta",
        )
        mean_auxiliary = {name: 0.0 for name in metric_names}
        mean_vae_mu_grad_norm = 0.0
        mean_successor_decoder_grad_norm = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None
        kl_beta, kl_capacity = self._kl_schedule()

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches,
                self.num_learning_epochs,
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches,
                self.num_learning_epochs,
            )

        for batch in generator:
            if batch.observations is None:
                raise RuntimeError("PIEPPO received a batch without observations.")
            if any(
                value is None
                for value in (
                    batch.actions,
                    batch.values,
                    batch.advantages,
                    batch.returns,
                    batch.old_actions_log_prob,
                    batch.old_distribution_params,
                )
            ):
                raise RuntimeError("PIEPPO received an incomplete rollout batch.")

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1.0e-8
                    )

            _, auxiliary_outputs = self.actor.forward_with_auxiliary(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[1],
            )
            distribution_params = self.actor.output_distribution_params
            entropy = self.actor.output_entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(
                        batch.old_distribution_params,
                        distribution_params,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                        elif 0.0 < kl_mean < self.desired_kl / 2.0:
                            self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        learning_rate = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(learning_rate, src=0)
                        self.learning_rate = learning_rate.item()
                    for parameter_group in self.optimizer.param_groups:
                        parameter_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - batch.old_actions_log_prob.squeeze(-1))
            advantages = batch.advantages.squeeze(-1)
            surrogate = -advantages * ratio
            surrogate_clipped = -advantages * torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(
                    -self.clip_param,
                    self.clip_param,
                )
                value_loss = torch.max(
                    (values - batch.returns).square(),
                    (value_clipped - batch.returns).square(),
                ).mean()
            else:
                value_loss = (batch.returns - values).square().mean()

            auxiliary_metrics = self.actor.auxiliary_losses_from_outputs(
                auxiliary_outputs,
                batch.observations,
                batch.masks,
            )
            kl_objective = (auxiliary_metrics["kl_total"] - kl_capacity).abs()
            weighted_auxiliary = sum(
                self.auxiliary_coefficients[name] * auxiliary_metrics[name]
                for name in self.auxiliary_coefficients
            ) + kl_beta * kl_objective
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
                + self.auxiliary_loss_coef * weighted_auxiliary
            )

            if self.rnd:
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch.observations)
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                rnd_loss = nn.functional.mse_loss(predicted_embedding, target_embedding)

            self.optimizer.zero_grad()
            loss.backward()
            vae_mu_grad_norm = self._module_grad_norm(self.actor.vae_mu_head)
            successor_decoder_grad_norm = self._module_grad_norm(self.actor.successor_decoder)
            if self.rnd:
                assert self.rnd_optimizer is not None
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_weighted_auxiliary += weighted_auxiliary.item()
            auxiliary_metrics["kl_objective"] = kl_objective
            for name, value in auxiliary_metrics.items():
                mean_auxiliary[name] += value.item()
            mean_vae_mu_grad_norm += vae_mu_grad_norm
            mean_successor_decoder_grad_norm += successor_decoder_grad_norm
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_weighted_auxiliary /= num_updates
        mean_auxiliary = {
            name: value / num_updates for name, value in mean_auxiliary.items()
        }
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        mean_vae_mu_grad_norm /= num_updates
        mean_successor_decoder_grad_norm /= num_updates

        self.storage.clear()
        self.pie_update_count += 1
        losses = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "pie_auxiliary": mean_weighted_auxiliary,
            **{f"pie_{name}": value for name, value in mean_auxiliary.items()},
            "pie_kl_beta": kl_beta,
            "pie_kl_capacity": kl_capacity,
            "pie_vae_mu_grad_norm": mean_vae_mu_grad_norm,
            "pie_successor_decoder_grad_norm": mean_successor_decoder_grad_norm,
            "pie_vae_mu_weight_norm": self.actor.vae_mu_head.weight.detach().norm().item(),
        }
        if mean_rnd_loss is not None:
            losses["rnd"] = mean_rnd_loss
        return losses

    def _kl_schedule(self) -> tuple[float, float]:
        """Return the capacity-objective weight and target for this PPO update."""
        elapsed = self.pie_update_count - self.kl_warmup_iterations
        progress = min(max(elapsed / self.kl_capacity_warmup_iterations, 0.0), 1.0)
        return self.kl_loss_coef * progress, self.kl_capacity_max * progress

    @staticmethod
    def _module_grad_norm(module: nn.Module) -> float:
        squared_norm = 0.0
        for parameter in module.parameters():
            if parameter.grad is not None:
                squared_norm += parameter.grad.detach().square().sum().item()
        return squared_norm**0.5

    def save(self) -> dict:
        saved = super().save()
        saved["pie_update_count"] = self.pie_update_count
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.pie_update_count = int(loaded_dict.get("pie_update_count", loaded_dict.get("iter", 0)))
        return load_iteration
