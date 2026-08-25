"""Unified LP-ACRL over velocity, terrain family, and geometry level."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from isaaclab.managers import ManagerTermBase

from .commands.lp_acrl_velocity_command import LPACRLVelocityCommand
from .lp_acrl import LearningProgressSampler


class TerrainLPACRLCurriculum(ManagerTermBase):
    """Sample one joint velocity/terrain task per episode from the task grid."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        p = cfg.params
        self.command_name = p["command_name"]
        self.reward_terms = tuple(p["reward_terms"])
        self.episodes_per_stage = int(p["episodes_per_stage"])
        self.planar_zero_threshold = float(p["planar_zero_threshold"])
        self.vx_edges = torch.tensor(p["vx_edges"], device=self.device)
        self.yaw_edges = torch.tensor(p["yaw_edges"], device=self.device)
        self.terrain_names = tuple(p["terrain_names"])
        self.num_levels = int(p["num_levels"])
        self.columns_per_type = int(p["columns_per_type"])
        self.lateral_range = tuple(p["lateral_range"])
        self.grid_shape = (
            len(self.vx_edges) - 1,
            len(self.yaw_edges) - 1,
            len(self.terrain_names),
            self.num_levels,
        )
        self.num_tasks = self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2] * self.grid_shape[3]
        self.sampler = LearningProgressSampler(
            self.num_tasks,
            self.device,
            beta=p["beta"],
            beta_scale=p.get("beta_scale", 0.5),
            lp_quantile=p.get("lp_quantile", 0.75),
            epsilon=p["epsilon"],
            ema_alpha=p["ema_alpha"],
            min_samples=p["min_samples"],
            max_probability=p.get("max_probability", 0.05),
            probability_update_weight=p.get("probability_update_weight", 0.7),
        )
        self.stage = 0
        self.episodes_in_stage = 0

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass

    def __call__(
        self,
        env,
        env_ids: Sequence[int],
        command_name: str,
        reward_terms: tuple[str, ...],
        vx_edges: tuple[float, ...],
        yaw_edges: tuple[float, ...],
        terrain_names: tuple[str, ...],
        num_levels: int,
        columns_per_type: int,
        lateral_range: tuple[float, float],
        episodes_per_stage: int,
        min_samples: int,
        beta: float,
        epsilon: float,
        ema_alpha: float,
        planar_zero_threshold: float,
        beta_scale: float = 0.5,
        lp_quantile: float = 0.75,
        max_probability: float = 0.05,
        probability_update_weight: float = 0.7,
    ) -> dict[str, torch.Tensor]:
        del command_name, reward_terms, vx_edges, yaw_edges, terrain_names, num_levels, columns_per_type
        del lateral_range, episodes_per_stage, min_samples, beta, epsilon, ema_alpha, planar_zero_threshold
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
            self._submit(command.task_ids[ended], score / env.max_episode_length_s)

        self._assign(ids, self.sampler.sample(len(ids)))
        return self.metrics()

    def _submit(self, task_ids: torch.Tensor, scores: torch.Tensor) -> None:
        offset = 0
        while offset < len(task_ids):
            take = min(self.episodes_per_stage - self.episodes_in_stage, len(task_ids) - offset)
            segment = slice(offset, offset + take)
            self.sampler.add(task_ids[segment], scores[segment])
            self.episodes_in_stage += take
            offset += take
            if self.episodes_in_stage == self.episodes_per_stage:
                self.sampler.finish_stage()
                self.stage += 1
                self.episodes_in_stage = 0

    def _as_ids(self, env_ids: Sequence[int]) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

    def decode(self, task_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        nyaw, nterrain, nlevel = self.grid_shape[1:]
        vx = task_ids // (nyaw * nterrain * nlevel)
        yaw = (task_ids // (nterrain * nlevel)) % nyaw
        terrain_type = (task_ids // nlevel) % nterrain
        terrain_level = task_ids % nlevel
        return vx, yaw, terrain_type, terrain_level

    def _assign(self, env_ids: torch.Tensor, task_ids: torch.Tensor) -> None:
        vx_bin, yaw_bin, terrain_type, terrain_level = self.decode(task_ids)
        count = len(task_ids)
        command_values = torch.zeros(count, 3, device=self.device)
        for column, indices, edges in ((0, vx_bin, self.vx_edges), (2, yaw_bin, self.yaw_edges)):
            u = torch.rand(count, device=self.device)
            magnitude = edges[indices] + u * (edges[indices + 1] - edges[indices])
            sign = torch.where(torch.rand_like(u) < 0.5, -1.0, 1.0)
            command_values[:, column] = magnitude * sign
        if self.lateral_range[0] != 0.0 or self.lateral_range[1] != 0.0:
            command_values[:, 1].uniform_(*self.lateral_range)
        command_values[
            torch.linalg.norm(command_values[:, :2], dim=1) < self.planar_zero_threshold, :2
        ] = 0.0

        column_offset = torch.randint(self.columns_per_type, (count,), device=self.device)
        terrain_column = terrain_type * self.columns_per_type + column_offset
        terrain = self._env.scene.terrain
        terrain.terrain_levels[env_ids] = terrain_level
        terrain.terrain_types[env_ids] = terrain_column
        terrain.env_origins[env_ids] = terrain.terrain_origins[terrain_level, terrain_column]
        command = self._env.command_manager.get_term(self.command_name)
        command.assign(env_ids, task_ids, command_values)

    def metrics(self) -> dict[str, torch.Tensor]:
        probabilities = self.sampler.probabilities
        entropy = -torch.sum(probabilities * torch.log(probabilities.clamp_min(1.0e-12)))
        return {
            "stage": torch.tensor(float(self.stage), device=self.device),
            "stage_progress": torch.tensor(self.episodes_in_stage / self.episodes_per_stage, device=self.device),
            "effective_tasks": torch.exp(entropy),
            "max_probability": probabilities.max(),
            "mean_abs_lp": self.sampler.learning_progress.abs().mean(),
            "coverage": (self.sampler.stage_sample_count > 0).float().mean(),
            "probability_kl": self.sampler.last_probability_kl,
            "mean_terrain_level": self._env.scene.terrain.terrain_levels.float().mean(),
        }

    def state_dict(self) -> dict[str, Any]:
        command = self._env.command_manager.get_term(self.command_name)
        state = {
            "version": 2,
            "stage": self.stage,
            "episodes_in_stage": self.episodes_in_stage,
            "completed_episode_count": self.stage * self.episodes_per_stage + self.episodes_in_stage,
            "grid_shape": self.grid_shape,
            "sampler": self.sampler.state_dict(),
            "task_ids_per_env": command.task_ids.detach().cpu(),
        }
        if str(self.device).startswith("cuda"):
            state["rng_state"] = torch.cuda.get_rng_state(self.device).cpu()
        else:
            state["rng_state"] = torch.get_rng_state()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if tuple(state["grid_shape"]) != self.grid_shape:
            raise ValueError(f"LP-ACRL grid mismatch: {state['grid_shape']} != {self.grid_shape}")
        self.stage = int(state["stage"])
        self.episodes_in_stage = int(state["episodes_in_stage"])
        self.sampler.load_state_dict(state["sampler"])
        rng_state = state.get("rng_state")
        if rng_state is not None:
            if str(self.device).startswith("cuda"):
                torch.cuda.set_rng_state(rng_state.cpu(), self.device)
            else:
                torch.set_rng_state(rng_state.cpu())

    def resample_current_episodes(self) -> None:
        ids = torch.arange(self.num_envs, device=self.device)
        self._assign(ids, self.sampler.sample(self.num_envs))
        self._env.command_manager.get_term(self.command_name)._resample(ids)

    def csv_snapshot(self) -> tuple[tuple[str, ...], list[tuple[Any, ...]]]:
        probabilities = self.sampler.probabilities.detach().cpu().tolist()
        progress = self.sampler.learning_progress.detach().cpu().tolist()
        estimates = self.sampler.reward_estimate.detach().cpu().tolist()
        rows = []
        for task_id in range(self.num_tasks):
            decoded = self.decode(torch.tensor([task_id], device=self.device))
            vx, yaw, terrain_type, level = (int(value.item()) for value in decoded)
            rows.append(
                (
                    task_id,
                    vx,
                    yaw,
                    self.terrain_names[terrain_type],
                    level,
                    probabilities[task_id],
                    progress[task_id],
                    estimates[task_id],
                )
            )
        header = (
            "task_id",
            "vx_bin",
            "yaw_bin",
            "terrain_type",
            "terrain_level",
            "probability",
            "learning_progress",
            "reward_ema",
        )
        return header, rows
