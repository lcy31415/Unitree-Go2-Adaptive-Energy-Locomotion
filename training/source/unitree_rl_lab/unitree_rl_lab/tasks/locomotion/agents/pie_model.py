"""Multi-modal recurrent actor for perceptive implicit-explicit locomotion."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.modules import EmpiricalNormalization, MLP
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, resolve_nn_activation, unpad_trajectories
from tensordict import TensorDict


class PIEActorModel(nn.Module):
    """Fuse proprioceptive history and depth images into a recurrent policy.

    The actor consumes only the ``actor``, ``proprio_history`` and ``camera``
    groups. Privileged target groups are used exclusively by
    :meth:`auxiliary_losses` during training.
    """

    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict[str, Any] | None = None,
        cnn_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        cfg = dict(cnn_cfg or {})

        if obs_set not in obs_groups:
            raise ValueError(f"PIE observation set {obs_set!r} is absent from obs_groups.")

        self.proprio_group = str(cfg.get("proprio_group", "actor"))
        self.history_group = str(cfg.get("history_group", "proprio_history"))
        self.depth_group = str(cfg.get("depth_group", "camera"))
        self.velocity_target_group = str(cfg.get("velocity_target_group", "velocity_target"))
        self.height_target_group = str(cfg.get("height_target_group", "height_target"))
        self.foot_target_group = str(cfg.get("foot_target_group", "foot_clearance_target"))
        self.successor_target_group = str(cfg.get("successor_target_group", "successor_target"))
        self.successor_valid_group = str(cfg.get("successor_valid_group", "successor_valid"))

        actor_groups = (self.proprio_group, self.history_group, self.depth_group)
        absent_actor_groups = [name for name in actor_groups if name not in obs_groups[obs_set]]
        if absent_actor_groups:
            raise ValueError(
                f"PIE actor groups {absent_actor_groups} are absent from "
                f"obs_groups[{obs_set!r}]={obs_groups[obs_set]}."
            )

        required_groups = (
            *actor_groups,
            self.velocity_target_group,
            self.height_target_group,
            self.foot_target_group,
            self.successor_target_group,
            self.successor_valid_group,
        )
        absent_observations = [name for name in required_groups if name not in obs]
        if absent_observations:
            raise ValueError(f"PIE observations are missing groups: {absent_observations}.")

        self.proprio_dim = int(obs[self.proprio_group].shape[-1])
        self.history_dim = int(obs[self.history_group].shape[-1])
        self.velocity_dim = int(obs[self.velocity_target_group].shape[-1])
        self.height_map_dim = int(obs[self.height_target_group].shape[-1])
        self.foot_dim = int(obs[self.foot_target_group].shape[-1])
        self.successor_dim = int(obs[self.successor_target_group].shape[-1])
        self.history_length = int(cfg.get("history_length", 10))

        if self.proprio_dim != 45:
            raise ValueError(f"PIE expects 45 current proprioceptive values, got {self.proprio_dim}.")
        expected_history_dim = self.proprio_dim * self.history_length
        if self.history_dim != expected_history_dim:
            raise ValueError(
                f"PIE expects {self.history_length}x{self.proprio_dim}={expected_history_dim} "
                f"proprioceptive history values, got {self.history_dim}."
            )
        expected_target_dims = {
            self.velocity_target_group: (self.velocity_dim, 3),
            self.height_target_group: (self.height_map_dim, 198),
            self.foot_target_group: (self.foot_dim, 4),
            self.successor_target_group: (self.successor_dim, 45),
            self.successor_valid_group: (int(obs[self.successor_valid_group].shape[-1]), 1),
        }
        invalid_targets = {
            name: (actual, expected)
            for name, (actual, expected) in expected_target_dims.items()
            if actual != expected
        }
        if invalid_targets:
            details = ", ".join(
                f"{name}={actual} (expected {expected})"
                for name, (actual, expected) in invalid_targets.items()
            )
            raise ValueError(f"PIE target dimensions are invalid: {details}.")

        self.depth_shape = tuple(int(value) for value in cfg.get("depth_shape", (2, 60, 86)))
        if len(self.depth_shape) != 3:
            raise ValueError(f"PIE depth_shape must be (channels, height, width), got {self.depth_shape}.")
        self.depth_channels, self.depth_height, self.depth_width = self.depth_shape
        if min(self.depth_shape) <= 0:
            raise ValueError(f"PIE depth_shape values must be positive, got {self.depth_shape}.")
        self.depth_flat_dim = self.depth_channels * self.depth_height * self.depth_width
        observed_depth_dim = int(obs[self.depth_group].shape[-1])
        if observed_depth_dim != self.depth_flat_dim:
            raise ValueError(
                f"PIE depth group has {observed_depth_dim} values; expected {self.depth_flat_dim} "
                f"from depth_shape={self.depth_shape}."
            )

        self.token_dim = int(cfg.get("token_dim", 64))
        self.map_latent_dim = int(cfg.get("map_latent_dim", 16))
        self.vae_latent_dim = int(cfg.get("vae_latent_dim", 16))
        self.memory_hidden_dim = int(cfg.get("memory_hidden_dim", 128))
        self.memory_num_layers = int(cfg.get("memory_num_layers", 1))
        self.obs_normalization = bool(obs_normalization)

        if self.obs_normalization:
            self.proprio_normalizer: nn.Module = EmpiricalNormalization(self.proprio_dim)
            self.history_normalizer: nn.Module = EmpiricalNormalization(self.history_dim)
        else:
            self.proprio_normalizer = nn.Identity()
            self.history_normalizer = nn.Identity()

        history_hidden_dims = tuple(int(value) for value in cfg.get("history_hidden_dims", (256, 128)))
        self.history_encoder = MLP(
            self.history_dim,
            self.token_dim,
            history_hidden_dims,
            activation,
        )
        self.depth_encoder = self._build_depth_encoder(cfg)
        self.depth_token_norm = nn.LayerNorm(self.token_dim)

        attention_heads = int(cfg.get("attention_heads", 1))
        if self.token_dim % attention_heads != 0:
            raise ValueError(
                f"PIE token_dim={self.token_dim} must be divisible by attention_heads={attention_heads}."
            )
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=attention_heads,
            dim_feedforward=int(cfg.get("transformer_ff_dim", 256)),
            dropout=float(cfg.get("transformer_dropout", 0.0)),
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.cross_modal_transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=int(cfg.get("transformer_layers", 2)),
            enable_nested_tensor=False,
        )
        self.memory = nn.GRU(
            input_size=2 * self.token_dim,
            hidden_size=self.memory_hidden_dim,
            num_layers=self.memory_num_layers,
        )

        self.velocity_head = nn.Linear(self.memory_hidden_dim, self.velocity_dim)
        self.map_latent_head = nn.Linear(self.memory_hidden_dim, self.map_latent_dim)
        self.foot_clearance_head = nn.Linear(self.memory_hidden_dim, self.foot_dim)
        self.vae_mu_head = nn.Linear(self.memory_hidden_dim, self.vae_latent_dim)
        self.vae_logvar_head = nn.Linear(self.memory_hidden_dim, self.vae_latent_dim)

        estimator_output_dim = self.velocity_dim + self.map_latent_dim + self.foot_dim + self.vae_latent_dim
        self.actor_input_dim = self.proprio_dim + estimator_output_dim
        self.successor_decoder = MLP(
            estimator_output_dim,
            self.successor_dim,
            tuple(int(value) for value in cfg.get("successor_decoder_dims", (64, 128))),
            activation,
        )
        self.height_decoder = MLP(
            self.map_latent_dim,
            self.height_map_dim,
            tuple(int(value) for value in cfg.get("height_decoder_dims", (64, 128))),
            activation,
        )

        if distribution_cfg is None:
            self.distribution: Distribution | None = None
            actor_output_dim = output_dim
        else:
            mutable_distribution_cfg = dict(distribution_cfg)
            distribution_class: type[Distribution] = resolve_callable(
                mutable_distribution_cfg.pop("class_name")
            )  # type: ignore[assignment]
            self.distribution = distribution_class(output_dim, **mutable_distribution_cfg)
            actor_output_dim = self.distribution.input_dim

        self.mlp = MLP(self.actor_input_dim, actor_output_dim, hidden_dims, activation)
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

        self.memory_hidden: torch.Tensor | None = None

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Return actions for online inference or padded recurrent rollouts."""
        estimates, _ = self._estimate(
            obs[self.proprio_group],
            obs[self.history_group],
            obs[self.depth_group],
            masks=masks,
            hidden_state=hidden_state,
            update_internal=masks is None and hidden_state is None,
            sample_latent=self.training,
        )
        actor_input = torch.cat(
            (self.proprio_normalizer(obs[self.proprio_group]), estimates["latent"]),
            dim=-1,
        )
        if masks is not None:
            actor_input = unpad_trajectories(actor_input, masks)
        actor_output = self.mlp(actor_input)
        if self.distribution is None:
            return actor_output
        if stochastic_output:
            self.distribution.update(actor_output)
            return self.distribution.sample()
        return self.distribution.deterministic_output(actor_output)

    def auxiliary_outputs(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
        *,
        sample_latent: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Expose estimator predictions for tests, diagnostics and auxiliary PPO."""
        estimates, _ = self._estimate(
            obs[self.proprio_group],
            obs[self.history_group],
            obs[self.depth_group],
            masks=masks,
            hidden_state=hidden_state,
            update_internal=False,
            sample_latent=sample_latent,
        )
        return {
            **estimates,
            "height_reconstruction": self.height_decoder(estimates["map_latent"]),
            "successor_prediction": self.successor_decoder(estimates["latent"]),
        }

    def auxiliary_losses(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the five unweighted PIE estimator objectives."""
        estimates = self.auxiliary_outputs(
            obs,
            masks=masks,
            hidden_state=hidden_state,
            sample_latent=True,
        )
        mu = estimates["mu"]
        logvar = estimates["logvar"]
        kl_per_sample = -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp(), dim=-1)
        return {
            "velocity": self._masked_mse(estimates["velocity"], obs[self.velocity_target_group], masks),
            "foot_clearance": self._masked_mse(
                estimates["foot_clearance"], obs[self.foot_target_group], masks
            ),
            "height_reconstruction": self._masked_mse(
                estimates["height_reconstruction"], obs[self.height_target_group], masks
            ),
            "successor": self._masked_mse(
                estimates["successor_prediction"],
                obs[self.successor_target_group],
                self._successor_mask(obs, masks),
            ),
            "kl": self._masked_mean(kl_per_sample, masks),
        }

    def _successor_mask(self, obs: TensorDict, masks: torch.Tensor | None) -> torch.Tensor:
        successor_valid = obs[self.successor_valid_group].squeeze(-1).to(dtype=torch.bool)
        if masks is None:
            return successor_valid
        return masks.to(dtype=torch.bool) & successor_valid

    def _estimate(
        self,
        proprio: torch.Tensor,
        history: torch.Tensor,
        depth: torch.Tensor,
        *,
        masks: torch.Tensor | None,
        hidden_state: torch.Tensor | None,
        update_internal: bool,
        sample_latent: bool,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        history_token = self.history_encoder(self.history_normalizer(history))
        depth_tokens = self._encode_depth(depth)
        leading_shape = history_token.shape[:-1]
        tokens = torch.cat((history_token.unsqueeze(-2), depth_tokens), dim=-2)
        token_count = tokens.shape[-2]
        fused_tokens = self.cross_modal_transformer(tokens.reshape(-1, token_count, self.token_dim))
        fused_tokens = fused_tokens.reshape(*leading_shape, token_count, self.token_dim)
        memory_input = torch.cat(
            (fused_tokens[..., 0, :], fused_tokens[..., 1:, :].mean(dim=-2)),
            dim=-1,
        )

        if masks is not None:
            if hidden_state is None:
                raise ValueError("PIE recurrent rollout evaluation requires an initial hidden_state.")
            memory_output, new_hidden = self.memory(memory_input, hidden_state)
        else:
            active_hidden = hidden_state if hidden_state is not None else self.memory_hidden
            memory_output, new_hidden = self.memory(memory_input.unsqueeze(0), active_hidden)
            memory_output = memory_output.squeeze(0)
            if update_internal:
                self.memory_hidden = new_hidden

        velocity = self.velocity_head(memory_output)
        map_latent = self.map_latent_head(memory_output)
        foot_clearance = self.foot_clearance_head(memory_output)
        mu = self.vae_mu_head(memory_output)
        logvar = self.vae_logvar_head(memory_output).clamp(-10.0, 5.0)
        if sample_latent:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        latent = torch.cat((velocity, map_latent, foot_clearance, z), dim=-1)
        return {
            "velocity": velocity,
            "map_latent": map_latent,
            "foot_clearance": foot_clearance,
            "mu": mu,
            "logvar": logvar,
            "latent": latent,
        }, new_hidden

    def _build_depth_encoder(self, cfg: dict[str, Any]) -> nn.Sequential:
        channels = [int(value) for value in cfg.get("output_channels", (32, 64, 64))]
        kernels = [int(value) for value in cfg.get("kernel_size", (8, 4, 3))]
        strides = [int(value) for value in cfg.get("stride", (4, 2, 1))]
        if not channels or not kernels or not strides:
            raise ValueError("PIE depth CNN channels, kernels and strides must not be empty.")
        channels[-1] = self.token_dim
        padding = cfg.get("padding", 0)
        layers: list[nn.Module] = []
        input_channels = self.depth_channels
        for index, output_channels in enumerate(channels):
            kernel = kernels[min(index, len(kernels) - 1)]
            stride = strides[min(index, len(strides) - 1)]
            pad = int(padding[min(index, len(padding) - 1)]) if isinstance(padding, (list, tuple)) else int(padding)
            layers.append(
                nn.Conv2d(input_channels, output_channels, kernel_size=kernel, stride=stride, padding=pad)
            )
            layers.append(resolve_nn_activation(str(cfg.get("cnn_activation", "elu"))))
            input_channels = output_channels

        # Fail during construction rather than during the first rollout when
        # a modified image size no longer fits the configured convolutions.
        try:
            with torch.no_grad():
                encoded = nn.Sequential(*layers)(
                    torch.zeros(1, self.depth_channels, self.depth_height, self.depth_width)
                )
        except RuntimeError as error:
            raise ValueError(
                f"PIE depth CNN is incompatible with depth_shape={self.depth_shape}."
            ) from error
        if encoded.shape[-2] <= 0 or encoded.shape[-1] <= 0:
            raise ValueError(f"PIE depth CNN produced an invalid spatial shape {tuple(encoded.shape)}.")
        return nn.Sequential(*layers)

    def _encode_depth(self, depth: torch.Tensor) -> torch.Tensor:
        leading_shape = depth.shape[:-1]
        images = depth.reshape(-1, *self.depth_shape) - 0.5
        features = self.depth_encoder(images)
        tokens = features.flatten(start_dim=2).transpose(1, 2)
        tokens = self.depth_token_norm(tokens)
        return tokens.reshape(*leading_shape, tokens.shape[-2], self.token_dim)

    @staticmethod
    def _masked_mean(values: torch.Tensor, masks: torch.Tensor | None) -> torch.Tensor:
        if masks is None:
            return values.mean()
        mask = masks.to(device=values.device, dtype=values.dtype)
        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(-1)
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    @classmethod
    def _masked_mse(
        cls,
        prediction: torch.Tensor,
        target: torch.Tensor,
        masks: torch.Tensor | None,
    ) -> torch.Tensor:
        return cls._masked_mean((prediction - target).square().mean(dim=-1), masks)

    def reset(
        self,
        dones: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
    ) -> None:
        """Replace the hidden state or clear entries for completed environments."""
        if dones is None:
            self.memory_hidden = hidden_state
            return
        if self.memory_hidden is not None:
            done_mask = dones.reshape(-1).to(device=self.memory_hidden.device, dtype=torch.bool)
            if done_mask.numel() != self.memory_hidden.shape[1]:
                raise ValueError(
                    f"PIE reset received {done_mask.numel()} done flags for "
                    f"{self.memory_hidden.shape[1]} hidden states."
                )
            self.memory_hidden[:, done_mask, :] = 0.0

    def get_hidden_state(self) -> torch.Tensor | None:
        return self.memory_hidden

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones
        if self.memory_hidden is not None:
            self.memory_hidden = self.memory_hidden.detach()

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            self.proprio_normalizer.update(obs[self.proprio_group])  # type: ignore[attr-defined]
            self.history_normalizer.update(obs[self.history_group])  # type: ignore[attr-defined]

    @property
    def output_mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("PIE actor has no output distribution.")
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("PIE actor has no output distribution.")
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("PIE actor has no output distribution.")
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        if self.distribution is None:
            raise RuntimeError("PIE actor has no output distribution.")
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("PIE actor has no output distribution.")
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("PIE actor has no output distribution.")
        return self.distribution.kl_divergence(old_params, new_params)

    def as_jit(self) -> nn.Module:
        return _ExportPIEActor(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        del verbose
        return _ExportPIEActor(self)


class _ExportPIEActor(nn.Module):
    """Stateless export wrapper with explicit GRU hidden input and output."""

    is_recurrent: bool = True

    def __init__(self, model: PIEActorModel) -> None:
        super().__init__()
        self.proprio_normalizer = copy.deepcopy(model.proprio_normalizer)
        self.history_normalizer = copy.deepcopy(model.history_normalizer)
        self.history_encoder = copy.deepcopy(model.history_encoder)
        self.depth_encoder = copy.deepcopy(model.depth_encoder)
        self.depth_token_norm = copy.deepcopy(model.depth_token_norm)
        self.cross_modal_transformer = copy.deepcopy(model.cross_modal_transformer)
        self.memory = copy.deepcopy(model.memory)
        self.velocity_head = copy.deepcopy(model.velocity_head)
        self.map_latent_head = copy.deepcopy(model.map_latent_head)
        self.foot_clearance_head = copy.deepcopy(model.foot_clearance_head)
        self.vae_mu_head = copy.deepcopy(model.vae_mu_head)
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = (
            copy.deepcopy(model.distribution.as_deterministic_output_module())
            if model.distribution is not None
            else nn.Identity()
        )
        self.proprio_dim = model.proprio_dim
        self.history_dim = model.history_dim
        self.depth_channels = model.depth_channels
        self.depth_height = model.depth_height
        self.depth_width = model.depth_width
        self.token_dim = model.token_dim
        self.memory_num_layers = model.memory_num_layers
        self.memory_hidden_dim = model.memory_hidden_dim
        self.eval()

    def forward(
        self,
        proprio: torch.Tensor,
        proprio_history: torch.Tensor,
        depth_history: torch.Tensor,
        memory_h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_token = self.history_encoder(self.history_normalizer(proprio_history))
        images = depth_history.reshape(
            -1,
            self.depth_channels,
            self.depth_height,
            self.depth_width,
        ) - 0.5
        depth_features = self.depth_encoder(images)
        depth_tokens = depth_features.flatten(start_dim=2).transpose(1, 2)
        depth_tokens = self.depth_token_norm(depth_tokens)
        tokens = torch.cat((history_token.unsqueeze(1), depth_tokens), dim=1)
        fused_tokens = self.cross_modal_transformer(tokens)
        memory_input = torch.cat(
            (fused_tokens[:, 0, :], fused_tokens[:, 1:, :].mean(dim=1)),
            dim=-1,
        )
        memory_output, new_hidden = self.memory(memory_input.unsqueeze(0), memory_h)
        memory_output = memory_output.squeeze(0)
        velocity = self.velocity_head(memory_output)
        map_latent = self.map_latent_head(memory_output)
        foot_clearance = self.foot_clearance_head(memory_output)
        z = self.vae_mu_head(memory_output)
        latent = torch.cat((velocity, map_latent, foot_clearance, z), dim=-1)
        actor_input = torch.cat(
            (self.proprio_normalizer(proprio), latent),
            dim=-1,
        )
        output = self.mlp(actor_input)
        return self.deterministic_output(output), new_hidden

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        return (
            torch.zeros(1, self.proprio_dim),
            torch.zeros(1, self.history_dim),
            torch.zeros(1, self.depth_channels, self.depth_height, self.depth_width),
            torch.zeros(self.memory_num_layers, 1, self.memory_hidden_dim),
        )

    @property
    def input_names(self) -> list[str]:
        return ["proprio", "proprio_history", "depth_history", "memory_h_in"]

    @property
    def output_names(self) -> list[str]:
        return ["actions", "memory_h_out"]
