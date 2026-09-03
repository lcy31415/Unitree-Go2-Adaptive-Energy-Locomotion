"""Flood-fill curriculum over joint (terrain level x forward speed x yaw rate) grids.

Each terrain family owns one grid whose cells are locked, active or mastered.
The lowest-difficulty cell starts active; mastery of a cell requires sustained
episode success, and a locked cell activates once enough of its grid neighbours
are mastered. Difficulty therefore spreads outward from the easy corner like a
crystallization front, and every episode failure is attributed to exactly one
cell instead of demoting an entire difficulty ladder.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils.configclass import configclass


STATE_LOCKED = 0
STATE_ACTIVE = 1
STATE_MASTERED = 2


def _as_torch(value):
    return getattr(value, "torch", value)


class FloodGridFamily:
    """One family's cell grid with flood-fill state transitions."""

    def __init__(
        self,
        name: str,
        level_count: int,
        vx_edges: Sequence[float],
        yaw_edges: Sequence[float],
        device: str,
        *,
        tracking_tolerances: Sequence[float] = (0.18, 0.15, 0.20),
        success_tracking_fraction: float = 0.7,
        mastery_ema: float = 0.70,
        min_episodes: int = 8,
        forget_ema: float = 0.50,
        forget_window: int = 6,
        activation_neighbors: int = 1,
        active_weight: float = 1.0,
        mastered_weight: float = 0.15,
        boost_factor: float = 3.0,
        boost_episodes: int = 16,
        ema_alpha: float = 0.25,
        forward_only: bool = False,
        zero_yaw_probability: float = 0.0,
        minimum_progress: float = 0.0,
    ):
        if level_count < 1 or len(vx_edges) < 2 or len(yaw_edges) < 2:
            raise ValueError(f"Grid for {name!r} needs >=1 level and >=2 bin edges per axis.")
        if not 0.0 <= forget_ema <= mastery_ema <= 1.0:
            raise ValueError("Grid thresholds must satisfy 0 <= forget < mastery <= 1.")
        if min_episodes < 1 or forget_window < 1 or activation_neighbors < 1:
            raise ValueError("Grid episode/neighbour parameters must be positive.")
        if active_weight <= 0.0 or mastered_weight < 0.0 or boost_factor < 1.0:
            raise ValueError("Grid sampling weights must be positive (mastered non-negative).")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("Grid EMA alpha must be in (0, 1].")
        if not 0.0 <= zero_yaw_probability <= 1.0:
            raise ValueError("Grid zero-yaw probability must be in [0, 1].")
        if minimum_progress < 0.0:
            raise ValueError("Grid minimum progress must be non-negative.")
        if len(tracking_tolerances) != 3 or any(value <= 0.0 for value in tracking_tolerances):
            raise ValueError("Grid tracking tolerances must contain three positive vx/vy/yaw values.")

        self.name = name
        self.device = device
        self.level_count = level_count
        self.register_buffer_edges(vx_edges, yaw_edges)
        self.shape = (level_count, len(vx_edges) - 1, len(yaw_edges) - 1)
        self.num_cells = level_count * self.shape[1] * self.shape[2]

        self.tracking_tolerances = tuple(float(value) for value in tracking_tolerances)
        self.success_tracking_fraction = success_tracking_fraction
        self.mastery_ema = mastery_ema
        self.min_episodes = min_episodes
        self.forget_ema = forget_ema
        self.forget_window = forget_window
        self.activation_neighbors = activation_neighbors
        self.active_weight = active_weight
        self.mastered_weight = mastered_weight
        self.boost_factor = boost_factor
        self.boost_episodes = boost_episodes
        self.ema_alpha = ema_alpha
        self.forward_only = bool(forward_only)
        self.zero_yaw_probability = float(zero_yaw_probability)
        self.minimum_progress = float(minimum_progress)

        self.state = torch.full(self.shape, STATE_LOCKED, dtype=torch.int8, device=device)
        self.succ_ema = torch.zeros(self.shape, device=device)
        self.ep_count = torch.zeros(self.shape, dtype=torch.long, device=device)
        self.boost_left = torch.zeros(self.shape, dtype=torch.long, device=device)
        self.episodes_since_master = torch.zeros(self.shape, dtype=torch.long, device=device)
        # Start at the easy corner instead of the geometric centre.  A centre
        # seed makes a newly initialized policy face medium terrain, speed and
        # yaw difficulty simultaneously and can prevent the front from ever
        # acquiring its first mastered cell.
        self.seed_cell = (0, 0, 0)
        self.state[self.seed_cell] = STATE_ACTIVE

        self.neighbor_degree = self._shifted_sum(torch.ones_like(self.state, dtype=torch.long))
        self.activation_threshold = torch.minimum(
            torch.full_like(self.neighbor_degree, activation_neighbors), self.neighbor_degree
        )

    def register_buffer_edges(self, vx_edges: Sequence[float], yaw_edges: Sequence[float]) -> None:
        self.vx_edges = torch.tensor(vx_edges, dtype=torch.float, device=self.device)
        self.yaw_edges = torch.tensor(yaw_edges, dtype=torch.float, device=self.device)
        if bool((self.vx_edges < 0.0).any()) or bool((self.yaw_edges < 0.0).any()):
            raise ValueError("Flood-grid edges encode magnitudes and must be non-negative.")
        if bool((torch.diff(self.vx_edges) <= 0.0).any()) or bool(
            (torch.diff(self.yaw_edges) <= 0.0).any()
        ):
            raise ValueError("Flood-grid edges must be strictly increasing.")

    # ------------------------------------------------------------------ utils

    def _shifted_sum(self, values: torch.Tensor) -> torch.Tensor:
        """Sum the six axial ±1 neighbours of every cell."""
        total = torch.zeros(self.shape, dtype=values.dtype, device=self.device)
        for axis in range(3):
            size = self.shape[axis]
            for step in (-1, 1):
                shifted = torch.roll(values, shifts=step, dims=axis)
                invalid = torch.zeros(self.shape, dtype=torch.bool, device=self.device)
                if step == 1:
                    invalid[(slice(None),) * axis + (0,)] = True
                else:
                    invalid[(slice(None),) * axis + (size - 1,)] = True
                total += torch.where(invalid, torch.zeros_like(shifted), shifted)
        return total

    def _flat(self, cells: torch.Tensor) -> torch.Tensor:
        return cells[:, 0] * self.shape[1] * self.shape[2] + cells[:, 1] * self.shape[2] + cells[:, 2]

    def _unflat(self, flat: torch.Tensor) -> torch.Tensor:
        level = flat // (self.shape[1] * self.shape[2])
        rem = flat % (self.shape[1] * self.shape[2])
        return torch.stack((level, rem // self.shape[2], rem % self.shape[2]), dim=1)

    # ---------------------------------------------------------------- updates

    def update(self, cells: torch.Tensor, success: torch.Tensor) -> dict[str, int]:
        """Attribute finished episodes to their cells and run state transitions."""
        events = {"mastered": 0, "forgotten": 0, "activated": 0}
        if cells.numel() == 0:
            return events
        flat = self._flat(cells)
        unique = torch.unique(flat)
        count = torch.bincount(flat, minlength=self.num_cells).to(torch.float)
        hits = torch.bincount(flat[success], minlength=self.num_cells).to(torch.float)

        previous_ema = self.succ_ema.reshape(-1)[unique]
        # Multiple environments can finish on one cell in the same reset. Use
        # the equivalent multi-observation EMA gain instead of treating the
        # whole cohort as one episode.
        effective_alpha = 1.0 - (1.0 - self.ema_alpha) ** count[unique]
        batch_success = hits[unique] / count[unique]
        self.succ_ema.reshape(-1)[unique] = (
            (1.0 - effective_alpha) * previous_ema + effective_alpha * batch_success
        )
        count_grid = count.to(torch.long).reshape(self.shape)
        self.ep_count += count_grid

        sampled_mask = torch.zeros(self.num_cells, dtype=torch.bool, device=self.device)
        sampled_mask[unique] = True
        sampled_mask = sampled_mask.reshape(self.shape)
        self.boost_left = torch.where(
            sampled_mask,
            (self.boost_left - count_grid).clamp_min(0),
            self.boost_left,
        )
        mastered_sampled = sampled_mask & (self.state == STATE_MASTERED)
        self.episodes_since_master = torch.where(
            mastered_sampled,
            self.episodes_since_master + count_grid,
            self.episodes_since_master,
        )

        newly_mastered = (
            (self.state == STATE_ACTIVE)
            & (self.ep_count >= self.min_episodes)
            & (self.succ_ema >= self.mastery_ema)
        )
        self.state = torch.where(newly_mastered, torch.full_like(self.state, STATE_MASTERED), self.state)
        self.episodes_since_master = torch.where(
            newly_mastered, torch.zeros_like(self.episodes_since_master), self.episodes_since_master
        )
        events["mastered"] = int(newly_mastered.sum().item())

        forgotten = (
            (self.state == STATE_MASTERED)
            & (self.episodes_since_master >= self.forget_window)
            & (self.succ_ema < self.forget_ema)
        )
        self.state = torch.where(forgotten, torch.full_like(self.state, STATE_ACTIVE), self.state)
        self.episodes_since_master = torch.where(
            forgotten, torch.zeros_like(self.episodes_since_master), self.episodes_since_master
        )
        events["forgotten"] = int(forgotten.sum().item())

        if events["mastered"]:
            mastered_neighbors = self._shifted_sum((self.state == STATE_MASTERED).to(torch.long))
            activated = (
                (self.state == STATE_LOCKED)
                & (self.neighbor_degree > 0)
                & (mastered_neighbors >= self.activation_threshold)
            )
            self.state = torch.where(activated, torch.full_like(self.state, STATE_ACTIVE), self.state)
            self.boost_left = torch.where(activated, torch.full_like(self.boost_left, self.boost_episodes), self.boost_left)
            events["activated"] = int(activated.sum().item())
        return events

    # ---------------------------------------------------------------- sampling

    def sampling_weights(self) -> torch.Tensor:
        """Return unnormalised cell probabilities for diagnostics and tests."""
        return torch.where(
            self.state == STATE_ACTIVE,
            torch.full_like(self.succ_ema, self.active_weight)
            + (self.boost_left > 0).to(self.succ_ema.dtype)
            * self.active_weight
            * (self.boost_factor - 1.0),
            torch.where(
                self.state == STATE_MASTERED,
                torch.full_like(self.succ_ema, self.mastered_weight),
                torch.zeros_like(self.succ_ema),
            ),
        )

    def sample(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample cells by state weight and draw in-cell commands."""
        if count <= 0:
            empty_cells = torch.zeros(0, 3, dtype=torch.long, device=self.device)
            empty_commands = torch.zeros(0, 3, device=self.device)
            return empty_cells, empty_commands
        weights = self.sampling_weights().reshape(-1)
        if not bool((weights > 0).any()):
            raise RuntimeError(f"Family {self.name!r} has no sampleable cells.")
        flat = torch.multinomial(weights, count, replacement=True)
        cells = self._unflat(flat)

        vx_bin_width = self.vx_edges[1:] - self.vx_edges[:-1]
        yaw_bin_width = self.yaw_edges[1:] - self.yaw_edges[:-1]
        u_vx = torch.rand(count, device=self.device)
        u_yaw = torch.rand(count, device=self.device)
        sign_vx = (
            torch.ones(count, device=self.device)
            if self.forward_only
            else torch.where(torch.rand(count, device=self.device) < 0.5, -1.0, 1.0)
        )
        sign_yaw = torch.where(torch.rand(count, device=self.device) < 0.5, -1.0, 1.0)
        vx = (self.vx_edges[cells[:, 1]] + u_vx * vx_bin_width[cells[:, 1]]) * sign_vx
        yaw = (self.yaw_edges[cells[:, 2]] + u_yaw * yaw_bin_width[cells[:, 2]]) * sign_yaw
        if self.zero_yaw_probability > 0.0:
            zero_yaw = torch.rand(count, device=self.device) < self.zero_yaw_probability
            yaw = torch.where(zero_yaw, torch.zeros_like(yaw), yaw)
        commands = torch.zeros(count, 3, device=self.device)
        commands[:, 0] = vx
        commands[:, 2] = yaw
        return cells, commands

    # ---------------------------------------------------------------- reporting

    def metrics(self) -> dict[str, torch.Tensor]:
        active = self.state == STATE_ACTIVE
        mastered = self.state == STATE_MASTERED
        sampleable = active | mastered
        levels = torch.arange(self.shape[0], device=self.device).view(-1, 1, 1).expand(self.shape).to(torch.float)
        vx_midpoints = ((self.vx_edges[:-1] + self.vx_edges[1:]) * 0.5).view(1, -1, 1).expand(self.shape)
        yaw_midpoints = ((self.yaw_edges[:-1] + self.yaw_edges[1:]) * 0.5).view(1, 1, -1).expand(self.shape)
        sampling_weights = self.sampling_weights()
        denominator = sampling_weights.sum().clamp_min(1.0e-6)
        return {
            "active": active.sum().to(torch.float),
            "mastered": mastered.sum().to(torch.float),
            "locked": (self.state == STATE_LOCKED).sum().to(torch.float),
            "coverage": sampleable.sum().to(torch.float) / float(self.num_cells),
            "expected_level": (levels * sampling_weights).sum() / denominator,
            "expected_abs_vx": (vx_midpoints * sampling_weights).sum() / denominator,
            "expected_abs_yaw": (yaw_midpoints * sampling_weights).sum() / denominator,
            "mean_succ_ema": self.succ_ema[sampleable].mean()
            if bool(sampleable.any())
            else torch.zeros((), device=self.device),
        }

    def snapshot_rows(self) -> list[tuple[Any, ...]]:
        rows: list[tuple[Any, ...]] = []
        state_cpu = self.state.cpu().tolist()
        ema_cpu = self.succ_ema.cpu().tolist()
        count_cpu = self.ep_count.cpu().tolist()
        for level in range(self.shape[0]):
            for vx_bin in range(self.shape[1]):
                for yaw_bin in range(self.shape[2]):
                    rows.append(
                        (
                            self.name,
                            level,
                            vx_bin,
                            yaw_bin,
                            state_cpu[level][vx_bin][yaw_bin],
                            ema_cpu[level][vx_bin][yaw_bin],
                            count_cpu[level][vx_bin][yaw_bin],
                        )
                    )
        return rows

    def state_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.detach().cpu(),
            "succ_ema": self.succ_ema.detach().cpu(),
            "ep_count": self.ep_count.detach().cpu(),
            "boost_left": self.boost_left.detach().cpu(),
            "episodes_since_master": self.episodes_since_master.detach().cpu(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for key, value in state.items():
            target = getattr(self, key)
            value = torch.as_tensor(value, dtype=target.dtype, device=self.device)
            # P0 removed the stair-yaw curriculum and collapsed its five yaw
            # bins to one exact-zero-yaw bin.  Keep old checkpoints usable by
            # inheriting their zero-yaw slice; the remaining slices described
            # commands that no longer belong to the task.
            if (
                value.ndim == target.ndim == 3
                and value.shape[:-1] == target.shape[:-1]
                and value.shape[-1] > 1
                and target.shape[-1] == 1
            ):
                value = value[..., :1]
            if value.shape != target.shape:
                raise ValueError(f"Flood grid {self.name!r} shape mismatch for {key}.")
            target.copy_(value)


class FloodFillVelocityCommand(UniformVelocityCommand):
    """Consume externally assigned commands and track per-episode tolerance."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._pending_commands = torch.zeros(self.num_envs, 3, device=self.device)
        default_tolerances = torch.tensor(
            (cfg.forward_error_abs, cfg.lateral_error_abs, cfg.angular_error_abs),
            dtype=torch.float,
            device=self.device,
        )
        self._pending_tolerances = default_tolerances.repeat(self.num_envs, 1)
        self._tracking_tolerances = default_tolerances.repeat(self.num_envs, 1)
        self._has_pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_within_tolerance_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def assign(
        self,
        env_ids: Sequence[int],
        commands: torch.Tensor,
        tracking_tolerances: Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        commands = commands.to(device=self.device, dtype=torch.float)
        if commands.shape != (env_ids.numel(), 3):
            raise ValueError(
                f"Flood-fill commands must have shape ({env_ids.numel()}, 3), got {tuple(commands.shape)}."
            )
        if tracking_tolerances is None:
            absolute = torch.tensor(
                (self.cfg.forward_error_abs, self.cfg.lateral_error_abs, self.cfg.angular_error_abs),
                dtype=torch.float,
                device=self.device,
            ).repeat(env_ids.numel(), 1)
            relative = torch.tensor(
                (self.cfg.forward_error_rel, self.cfg.lateral_error_rel, self.cfg.angular_error_rel),
                dtype=torch.float,
                device=self.device,
            )
            tolerances = torch.maximum(absolute, commands.abs() * relative)
        else:
            tolerances = torch.as_tensor(tracking_tolerances, dtype=torch.float, device=self.device)
            if tolerances.shape == (3,):
                tolerances = tolerances.repeat(env_ids.numel(), 1)
            if tolerances.shape != commands.shape:
                raise ValueError(
                    "Flood-fill tracking tolerances must have shape (3,) or "
                    f"{tuple(commands.shape)}, got {tuple(tolerances.shape)}."
                )
            if bool((tolerances <= 0.0).any()):
                raise ValueError("Flood-fill tracking tolerances must be positive.")
        self._pending_commands[env_ids] = commands
        self._pending_tolerances[env_ids] = tolerances
        self._has_pending[env_ids] = True

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if len(env_ids) == 0:
            return
        ready = env_ids[self._has_pending[env_ids]]
        if ready.numel() == 0:
            # Startup ordering: stand still until the curriculum assigns a cell.
            self.vel_command_b[env_ids] = 0.0
            self.is_standing_env[env_ids] = True
            return
        self.vel_command_b[ready] = self._pending_commands[ready]
        self._tracking_tolerances[ready] = self._pending_tolerances[ready]
        self.is_standing_env[ready] = False
        self._has_pending[ready] = False
        self._episode_within_tolerance_steps[ready] = 0
        self._episode_steps[ready] = 0

    def _update_metrics(self) -> None:
        root_lin_vel_b = _as_torch(self.robot.data.root_lin_vel_b)
        root_ang_vel_b = _as_torch(self.robot.data.root_ang_vel_b)
        error = torch.stack(
            (
                torch.abs(root_lin_vel_b[:, 0] - self.vel_command_b[:, 0]),
                torch.abs(root_lin_vel_b[:, 1] - self.vel_command_b[:, 1]),
                torch.abs(root_ang_vel_b[:, 2] - self.vel_command_b[:, 2]),
            ),
            dim=1,
        )
        within = (
            (error[:, 0] <= self._tracking_tolerances[:, 0])
            & (error[:, 1] <= self._tracking_tolerances[:, 1])
            & (error[:, 2] <= self._tracking_tolerances[:, 2])
        )
        self._episode_within_tolerance_steps += within.long()
        self._episode_steps += 1
        super()._update_metrics()


@configclass
class FloodFillVelocityCommandCfg(UniformVelocityCommandCfg):
    """Episode-fixed velocity command assigned externally by the flood grid."""

    class_type: type = FloodFillVelocityCommand
    forward_error_abs: float = 0.18
    forward_error_rel: float = 0.033
    lateral_error_abs: float = 0.15
    lateral_error_rel: float = 0.1
    angular_error_abs: float = 0.20
    angular_error_rel: float = 0.1


class FloodGridCurriculum(ManagerTermBase):
    """Attribute ended episodes to flood-grid cells, then assign the next cells."""

    checkpoint_name = "flood_grid"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        params = cfg.params
        self.command_name = params["command_name"]
        self.terrain_family_names = tuple(params["terrain_family_names"])
        self.active_family_indices = tuple(params["active_family_indices"])
        self.columns_per_family = int(params["columns_per_family"])
        family_grids = params["family_grids"]
        active_names = {self.terrain_family_names[index] for index in self.active_family_indices}
        configured_names = set(family_grids)
        if configured_names != active_names:
            raise ValueError(
                "Flood-grid family definitions must exactly match active families; "
                f"missing={sorted(active_names - configured_names)}, "
                f"inactive={sorted(configured_names - active_names)}."
            )
        self.grids = {
            index: FloodGridFamily(
                self.terrain_family_names[index],
                family_grids[self.terrain_family_names[index]]["levels"],
                family_grids[self.terrain_family_names[index]]["vx_edges"],
                family_grids[self.terrain_family_names[index]]["yaw_edges"],
                self.device,
                **{
                    key: value
                    for key, value in family_grids[self.terrain_family_names[index]].items()
                    if key not in {"levels", "vx_edges", "yaw_edges"}
                },
            )
            for index in self.active_family_indices
        }
        self._cells = torch.zeros(self.num_envs, 3, dtype=torch.long, device=self.device)
        self._events_total = {"mastered": 0, "forgotten": 0, "activated": 0}
        self._events_last = {"mastered": 0, "forgotten": 0, "activated": 0}
        self._family_events_last = {
            index: {"mastered": 0, "forgotten": 0, "activated": 0}
            for index in self.active_family_indices
        }
        self.updates = 0

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass

    def resample_current_episodes(self) -> None:
        """Reassign every environment after a checkpoint restore."""
        ids = torch.arange(self.num_envs, device=self.device)
        command = self._env.command_manager.get_term(self.command_name)
        terrain = self._env.scene.terrain
        families = self._family_indices(ids)
        for index, grid in self.grids.items():
            mask = families == index
            if not bool(mask.any()):
                continue
            family_ids = ids[mask]
            cells, commands = grid.sample(family_ids.numel())
            variants = torch.randint(self.columns_per_family, (family_ids.numel(),), device=self.device)
            terrain.terrain_levels[family_ids] = cells[:, 0]
            terrain.terrain_types[family_ids] = index * self.columns_per_family + variants
            terrain.env_origins[family_ids] = terrain.terrain_origins[
                terrain.terrain_levels[family_ids], terrain.terrain_types[family_ids]
            ]
            self._cells[family_ids] = cells
            command.assign(family_ids, commands, grid.tracking_tolerances)
        command._resample_command(ids)

    def _family_indices(self, env_ids: torch.Tensor) -> torch.Tensor:
        terrain = self._env.scene.terrain
        raw = torch.div(terrain.terrain_types[env_ids].long(), self.columns_per_family, rounding_mode="floor")
        return raw

    def __call__(
        self,
        env,
        env_ids: Sequence[int],
        command_name: str,
        terrain_family_names: tuple[str, ...],
        active_family_indices: tuple[int, ...],
        columns_per_family: int,
        family_grids: dict[str, dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        del command_name, terrain_family_names, active_family_indices, columns_per_family, family_grids
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.numel() == 0:
            return self.metrics()
        command = env.command_manager.get_term(self.command_name)
        terrain = env.scene.terrain
        self._events_last = {"mastered": 0, "forgotten": 0, "activated": 0}
        self._family_events_last = {
            index: {"mastered": 0, "forgotten": 0, "activated": 0}
            for index in self.active_family_indices
        }

        ended = ids[env.episode_length_buf[ids] > 0]
        if ended.numel():
            steps = command._episode_steps[ended].clamp_min(1)
            tracking_fraction = command._episode_within_tolerance_steps[ended].to(torch.float) / steps
            survived = env.episode_length_buf[ended] >= env.max_episode_length - 1
            root_pos_w = _as_torch(command.robot.data.root_pos_w)
            env_origins = _as_torch(terrain.env_origins).to(root_pos_w.device)
            progress = torch.linalg.vector_norm(
                root_pos_w[ended, :2] - env_origins[ended, :2], dim=1
            )
            families = self._family_indices(ended)
            for index, grid in self.grids.items():
                mask = families == index
                if not bool(mask.any()):
                    continue
                completed = progress[mask] >= grid.minimum_progress
                success = (
                    (tracking_fraction[mask] >= grid.success_tracking_fraction)
                    & completed
                    & survived[mask]
                )
                events = grid.update(self._cells[ended[mask]], success)
                for key, value in events.items():
                    self._family_events_last[index][key] += value
                    self._events_last[key] += value
                    self._events_total[key] += value
            self.updates += 1

        families = self._family_indices(ids)
        for index, grid in self.grids.items():
            mask = families == index
            if not bool(mask.any()):
                continue
            family_ids = ids[mask]
            cells, commands = grid.sample(family_ids.numel())
            variants = torch.randint(self.columns_per_family, (family_ids.numel(),), device=self.device)
            terrain.terrain_levels[family_ids] = cells[:, 0]
            terrain.terrain_types[family_ids] = index * self.columns_per_family + variants
            terrain.env_origins[family_ids] = terrain.terrain_origins[
                terrain.terrain_levels[family_ids], terrain.terrain_types[family_ids]
            ]
            self._cells[family_ids] = cells
            command.assign(family_ids, commands, grid.tracking_tolerances)
        return self.metrics()

    def metrics(self) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        aggregates: dict[str, list[torch.Tensor]] = {}
        for index, grid in self.grids.items():
            for key, value in grid.metrics().items():
                result[f"{grid.name}/{key}"] = value
                aggregates.setdefault(key, []).append(value)
            result[f"{grid.name}/activation_events"] = torch.tensor(
                float(self._family_events_last[index]["activated"]), device=self.device
            )
            result[f"{grid.name}/mastery_events"] = torch.tensor(
                float(self._family_events_last[index]["mastered"]), device=self.device
            )
            result[f"{grid.name}/forget_events"] = torch.tensor(
                float(self._family_events_last[index]["forgotten"]), device=self.device
            )
        for key, values in aggregates.items():
            if key in {"active", "mastered", "locked"}:
                result[key] = torch.stack(values).sum()
            else:
                result[key] = torch.stack(values).mean()
        total_cells = sum(grid.num_cells for grid in self.grids.values())
        result["coverage"] = (result["active"] + result["mastered"]) / float(total_cells)
        result["activation_events"] = torch.tensor(
            float(self._events_last["activated"]), device=self.device
        )
        result["mastery_events"] = torch.tensor(
            float(self._events_last["mastered"]), device=self.device
        )
        result["forget_events"] = torch.tensor(
            float(self._events_last["forgotten"]), device=self.device
        )
        for key, value in self._events_total.items():
            result[f"total_{key}"] = torch.tensor(float(value), device=self.device)
        result["updates"] = torch.tensor(float(self.updates), device=self.device)
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 2,
            "cells": self._cells.detach().cpu(),
            "events_total": dict(self._events_total),
            "updates": self.updates,
            "grids": {index: grid.state_dict() for index, grid in self.grids.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        cells = torch.as_tensor(state["cells"], dtype=torch.long, device=self.device)
        # Per-environment assignments are transient and will be resampled after
        # restore. This intentionally permits resuming with a different number
        # of parallel environments.
        if cells.shape == self._cells.shape:
            self._cells.copy_(cells)
        self._events_total = {**self._events_total, **state.get("events_total", {})}
        self.updates = int(state.get("updates", 0))
        grids = state.get("grids", {})
        for index, grid in self.grids.items():
            grid_state = grids.get(index, grids.get(str(index)))
            if grid_state is not None:
                grid.load_state_dict(grid_state)

    def csv_snapshot(self) -> tuple[tuple[str, ...], list[tuple[Any, ...]]]:
        header = ("family", "level", "vx_bin", "yaw_bin", "state", "succ_ema", "ep_count")
        rows: list[tuple[Any, ...]] = []
        for _, grid in sorted(self.grids.items()):
            rows.extend(grid.snapshot_rows())
        return header, rows
