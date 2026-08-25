"""Learning-progress automatic curriculum over discrete 3-D velocity tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from isaaclab.managers import ManagerTermBase

from .commands.lp_acrl_velocity_command import LPACRLVelocityCommand


class LearningProgressSampler:
    """Signed learning-progress sampler independent of Isaac Sim."""

    def __init__(self, num_tasks: int, device: str, beta=0.005, beta_scale=1.0, lp_quantile=0.75,
                 epsilon=0.1, ema_alpha=0.2, min_samples=2, max_probability=0.05,
                 probability_update_weight=0.3):
        if (num_tasks <= 0 or beta <= 0.0 or beta_scale < 0.0 or not 0.0 <= epsilon <= 1.0
                or not 0.0 < lp_quantile < 1.0 or max_probability < 1.0 / num_tasks
                or not 0.0 < probability_update_weight <= 1.0):
            raise ValueError("Invalid LP-ACRL sampler configuration.")
        self.num_tasks, self.device = num_tasks, device
        self.beta, self.beta_scale, self.lp_quantile = beta, beta_scale, lp_quantile
        self.epsilon, self.min_samples = epsilon, min_samples
        self.ema_alpha = ema_alpha
        self.max_probability = max_probability
        self.probability_update_weight = probability_update_weight
        self.probabilities = torch.full((num_tasks,), 1.0 / num_tasks, device=device)
        self.reward_estimate = torch.zeros(num_tasks, device=device)
        self.learning_progress = torch.zeros(num_tasks, device=device)
        self.has_estimate = torch.zeros(num_tasks, dtype=torch.bool, device=device)
        self.stage_reward_sum = torch.zeros(num_tasks, device=device)
        self.stage_sample_count = torch.zeros(num_tasks, dtype=torch.long, device=device)
        self.last_probability_kl = torch.zeros((), device=device)

    def add(self, task_ids: torch.Tensor, scores: torch.Tensor) -> None:
        valid = (task_ids >= 0) & torch.isfinite(scores)
        if bool(torch.any(valid)):
            ids = task_ids[valid]
            self.stage_reward_sum.scatter_add_(0, ids, scores[valid])
            self.stage_sample_count.scatter_add_(0, ids, torch.ones_like(ids))

    def finish_stage(self) -> None:
        observed = self.stage_sample_count >= self.min_samples
        means = self.stage_reward_sum / self.stage_sample_count.clamp_min(1)
        first = observed & ~self.has_estimate
        repeated = observed & self.has_estimate
        self.learning_progress.zero_()
        self.reward_estimate[first] = means[first]
        if bool(torch.any(repeated)):
            updated = (1.0 - self.ema_alpha) * self.reward_estimate[repeated] + self.ema_alpha * means[repeated]
            self.learning_progress[repeated] = updated - self.reward_estimate[repeated]
            self.reward_estimate[repeated] = updated
        self.has_estimate[observed] = True
        old = self.probabilities.clone()
        # The softmax temperature must track the LP magnitude: a fixed value
        # either flattens the distribution (temperature >> LP, everything
        # stays near-uniform) or collapses it (temperature << LP). Scale by
        # the upper quartile of |LP| because the strongest tasks sit several
        # times above the mean and are what drove the earlier collapse;
        # ``beta`` stays a small absolute floor so sampling relaxes back to
        # uniform once learning progress has decayed.
        active = self.learning_progress != 0.0
        if bool(torch.any(active)):
            lp_scale = torch.quantile(self.learning_progress[active].abs(), self.lp_quantile)
        else:
            lp_scale = torch.zeros((), device=self.device)
        temperature = torch.clamp(self.beta_scale * lp_scale, min=self.beta)
        adaptive = torch.softmax(self.learning_progress / temperature, dim=0)
        uniform = torch.full_like(adaptive, 1.0 / self.num_tasks)
        target = (1.0 - self.epsilon) * adaptive + self.epsilon * uniform
        # No task may monopolize sampling: clamp the cap and redistribute the
        # excess so the strongest tasks share mass with their neighbors.
        target = torch.clamp(target, max=self.max_probability)
        target = target / target.sum()
        target = torch.clamp(target, max=self.max_probability)
        # One noisy stage must not redirect the curriculum: blend the target
        # into the previous distribution so preferences move gradually.
        self.probabilities.copy_(
            (1.0 - self.probability_update_weight) * old + self.probability_update_weight * target
        )
        self.probabilities /= self.probabilities.sum()
        self.last_probability_kl = torch.sum(self.probabilities * torch.log(self.probabilities / old.clamp_min(1e-12)))
        self.stage_reward_sum.zero_()
        self.stage_sample_count.zero_()

    def sample(self, count: int) -> torch.Tensor:
        return torch.multinomial(self.probabilities, count, replacement=True)

    def state_dict(self) -> dict[str, Any]:
        names = ("probabilities", "reward_estimate", "learning_progress", "has_estimate",
                 "stage_reward_sum", "stage_sample_count", "last_probability_kl")
        return {name: getattr(self, name).detach().cpu() for name in names}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for name, value in state.items():
            target = getattr(self, name)
            value = torch.as_tensor(value, dtype=target.dtype, device=self.device)
            if value.shape != target.shape:
                raise ValueError(f"LP-ACRL state shape mismatch for {name}: {value.shape} != {target.shape}")
            target.copy_(value)


class LPACRLCurriculum(ManagerTermBase):
    """Attribute the ended episode to its old task, then assign its next task."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        p = cfg.params
        self.command_name = p["command_name"]
        self.reward_terms = tuple(p["reward_terms"])
        configured_stage_size = p["episodes_per_stage"]
        self.episodes_per_stage = self.num_envs if configured_stage_size is None else int(configured_stage_size)
        self.planar_zero_threshold = float(p["planar_zero_threshold"])
        self.vx_edges = torch.tensor(p["vx_edges"], device=self.device)
        self.vy_edges = torch.tensor(p["vy_edges"], device=self.device)
        self.yaw_edges = torch.tensor(p["yaw_edges"], device=self.device)
        self.grid_shape = (len(self.vx_edges) - 1, len(self.vy_edges) - 1, len(self.yaw_edges) - 1)
        self.num_tasks = self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2]
        self.sampler = LearningProgressSampler(
            self.num_tasks, self.device, p["beta"], p.get("beta_scale", 1.0), p.get("lp_quantile", 0.75),
            p["epsilon"], p["ema_alpha"], p["min_samples"], p.get("max_probability", 0.05),
            p.get("probability_update_weight", 0.3)
        )
        self.stage = 0
        self.episodes_in_stage = 0

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass

    def __call__(self, env, env_ids: Sequence[int], command_name: str, reward_terms: tuple[str, ...],
                 vx_edges: tuple[float, ...], vy_edges: tuple[float, ...], yaw_edges: tuple[float, ...],
                 episodes_per_stage: int | None, min_samples: int, beta: float, epsilon: float,
                 ema_alpha: float, planar_zero_threshold: float, beta_scale: float = 1.0,
                 lp_quantile: float = 0.75, max_probability: float = 0.05,
                 probability_update_weight: float = 0.3) -> dict[str, torch.Tensor]:
        del command_name, reward_terms, vx_edges, vy_edges, yaw_edges
        del episodes_per_stage, min_samples, beta, epsilon, ema_alpha, planar_zero_threshold
        del beta_scale, lp_quantile, max_probability, probability_update_weight
        ids = self._as_ids(env_ids)
        command = env.command_manager.get_term(self.command_name)
        if not isinstance(command, LPACRLVelocityCommand):
            raise TypeError(f"{self.command_name} must use LPACRLVelocityCommand.")

        completed = env.episode_length_buf[ids] > 0
        if bool(torch.any(completed)):
            ended = ids[completed]
            score = torch.zeros(len(ended), device=self.device)
            for name in self.reward_terms:
                score += env.reward_manager._episode_sums[name][ended]
            task_ids_ended = command.task_ids[ended]
            score = score / env.max_episode_length_s
            # A synchronized reset may contain more than one stage (e.g. 4096
            # envs and a 2048-episode stage). Split it exactly at boundaries.
            offset = 0
            while offset < len(ended):
                take = min(self.episodes_per_stage - self.episodes_in_stage, len(ended) - offset)
                segment = slice(offset, offset + take)
                self.sampler.add(task_ids_ended[segment], score[segment])
                self.episodes_in_stage += take
                offset += take
                if self.episodes_in_stage == self.episodes_per_stage:
                    self.sampler.finish_stage()
                    self.stage += 1
                    self.episodes_in_stage = 0

        task_ids = self.sampler.sample(len(ids))
        command.assign(ids, task_ids, self._sample_commands(task_ids))
        return self.metrics()

    def _as_ids(self, env_ids: Sequence[int]) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

    def _sample_commands(self, task_ids: torch.Tensor) -> torch.Tensor:
        ny, nw = self.grid_shape[1], self.grid_shape[2]
        indices = (task_ids // (ny * nw), (task_ids // nw) % ny, task_ids % nw)
        commands = torch.empty(len(task_ids), 3, device=self.device)
        for col, index, edges in zip(range(3), indices, (self.vx_edges, self.vy_edges, self.yaw_edges)):
            u = torch.rand(len(task_ids), device=self.device)
            magnitude = edges[index] + u * (edges[index + 1] - edges[index])
            commands[:, col] = magnitude * torch.where(torch.rand_like(u) < 0.5, -1.0, 1.0)
        commands[torch.linalg.norm(commands[:, :2], dim=1) < self.planar_zero_threshold, :2] = 0.0
        return commands

    def metrics(self) -> dict[str, torch.Tensor]:
        p = self.sampler.probabilities
        entropy = -torch.sum(p * torch.log(p.clamp_min(1e-12)))
        return {
            "stage": torch.tensor(float(self.stage), device=self.device),
            "stage_progress": torch.tensor(self.episodes_in_stage / self.episodes_per_stage, device=self.device),
            "effective_tasks": torch.exp(entropy),
            "max_probability": p.max(),
            "mean_abs_lp": self.sampler.learning_progress.abs().mean(),
            "coverage": (self.sampler.stage_sample_count > 0).float().mean(),
            "probability_kl": self.sampler.last_probability_kl,
        }

    def state_dict(self) -> dict[str, Any]:
        return {"version": 1, "stage": self.stage, "episodes_in_stage": self.episodes_in_stage,
                "grid_shape": self.grid_shape, "sampler": self.sampler.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if tuple(state["grid_shape"]) != self.grid_shape:
            raise ValueError(f"LP-ACRL grid mismatch: {state['grid_shape']} != {self.grid_shape}")
        self.stage = int(state["stage"])
        self.episodes_in_stage = int(state["episodes_in_stage"])
        self.sampler.load_state_dict(state["sampler"])

    def resample_current_episodes(self) -> None:
        """Replace construction-time uniform commands after a checkpoint restore."""
        ids = torch.arange(self.num_envs, device=self.device)
        task_ids = self.sampler.sample(self.num_envs)
        command = self._env.command_manager.get_term(self.command_name)
        command.assign(ids, task_ids, self._sample_commands(task_ids))
        command._resample(ids)
