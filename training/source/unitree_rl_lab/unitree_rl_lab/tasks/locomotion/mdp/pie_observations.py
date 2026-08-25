"""Perceptive observations and targets used by the PIE policy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as functional
from isaaclab.managers import ManagerTermBase, SceneEntityCfg


def _as_torch(value: Any) -> torch.Tensor:
    """Return a tensor from either a tensor or Isaac Lab ProxyArray."""
    tensor = value.torch if hasattr(value, "torch") else value
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor or ProxyArray, got {type(value).__name__}.")
    return tensor


def _scene_sensor(env, sensor_cfg: SceneEntityCfg):
    try:
        return env.scene.sensors[sensor_cfg.name]
    except (AttributeError, KeyError):
        return env.scene[sensor_cfg.name]


def preprocess_pie_depth(
    depth: torch.Tensor,
    *,
    cutoff_distance: float = 3.0,
    min_depth: float = 0.05,
    crop_left: int = 10,
    crop_right: int = 10,
    gaussian_blur: tuple[int, float] | None = (3, 1.0),
) -> torch.Tensor:
    """Crop, denoise and normalize depth to ``[N, 1, H, W]``.

    Isaac Lab ray-caster cameras return ``[N, H, W, 1]``. The channel-first
    form is accepted too so the preprocessing contract can be tested without
    constructing a simulator.
    """
    if cutoff_distance <= 0.0:
        raise ValueError("Depth cutoff_distance must be positive.")
    if min_depth < 0.0 or min_depth > cutoff_distance:
        raise ValueError("Depth min_depth must lie in [0, cutoff_distance].")
    if crop_left < 0 or crop_right < 0:
        raise ValueError("Depth crop widths must be nonnegative.")

    if depth.ndim == 3:
        channel_first = depth.unsqueeze(1)
    elif depth.ndim == 4 and depth.shape[-1] == 1:
        channel_first = depth.permute(0, 3, 1, 2)
    elif depth.ndim == 4 and depth.shape[1] == 1:
        channel_first = depth
    else:
        raise ValueError(
            "PIE depth must have shape [N,H,W], [N,H,W,1], or [N,1,H,W], "
            f"got {tuple(depth.shape)}."
        )

    width = channel_first.shape[-1]
    if crop_left + crop_right >= width:
        raise ValueError(
            f"Depth crop ({crop_left}, {crop_right}) removes all {width} image columns."
        )
    crop_stop = width - crop_right if crop_right else None
    channel_first = channel_first[..., crop_left:crop_stop]

    invalid = ~torch.isfinite(channel_first) | (channel_first <= 0.0)
    channel_first = torch.where(
        invalid,
        torch.full_like(channel_first, cutoff_distance),
        channel_first,
    )

    if gaussian_blur is not None:
        kernel_size, sigma = gaussian_blur
        if kernel_size <= 0 or kernel_size % 2 == 0 or sigma <= 0.0:
            raise ValueError("Gaussian blur requires an odd kernel size and positive sigma.")
        padding = kernel_size // 2
        if channel_first.shape[-2] <= padding or channel_first.shape[-1] <= padding:
            raise ValueError("Depth image is too small for reflection-padded Gaussian blur.")
        coordinates = (
            torch.arange(kernel_size, device=depth.device, dtype=depth.dtype)
            - (kernel_size - 1) / 2
        )
        kernel_1d = torch.exp(-0.5 * (coordinates / sigma).square())
        kernel_1d /= kernel_1d.sum()
        kernel_2d = (kernel_1d[:, None] * kernel_1d[None, :]).reshape(
            1,
            1,
            kernel_size,
            kernel_size,
        )
        channel_first = functional.conv2d(
            functional.pad(channel_first, (padding,) * 4, mode="reflect"),
            kernel_2d,
        )

    return channel_first.clamp(min=min_depth, max=cutoff_distance) / cutoff_distance


def pie_camera_depth(
    env,
    sensor_cfg: SceneEntityCfg,
    cutoff_distance: float = 3.0,
    min_depth: float = 0.05,
    crop_left: int = 10,
    crop_right: int = 10,
    gaussian_blur: tuple[int, float] | None = (3, 1.0),
) -> torch.Tensor:
    """Read and preprocess one frame from a RayCasterCamera."""
    sensor = _scene_sensor(env, sensor_cfg)
    output = sensor.data.output
    if output is None or "distance_to_camera" not in output:
        raise RuntimeError(
            f"Camera {sensor_cfg.name!r} has no 'distance_to_camera' output."
        )
    return preprocess_pie_depth(
        _as_torch(output["distance_to_camera"]),
        cutoff_distance=cutoff_distance,
        min_depth=min_depth,
        crop_left=crop_left,
        crop_right=crop_right,
        gaussian_blur=gaussian_blur,
    )


class PIEDepthHistory(ManagerTermBase):
    """Maintain the last two distinct camera frames, not control frames."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._history: torch.Tensor | None = None
        self._last_camera_frame = torch.full(
            (self.num_envs,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self._needs_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        selected: Sequence[int] | slice = slice(None) if env_ids is None else env_ids
        self._needs_reset[selected] = True
        self._last_camera_frame[selected] = -1

    def __call__(
        self,
        env,
        sensor_cfg: SceneEntityCfg,
        cutoff_distance: float = 3.0,
        min_depth: float = 0.05,
        crop_left: int = 10,
        crop_right: int = 10,
        gaussian_blur: tuple[int, float] | None = (3, 1.0),
        frame_history_length: int = 2,
    ) -> torch.Tensor:
        if frame_history_length <= 0:
            raise ValueError("Depth frame_history_length must be positive.")

        frame = pie_camera_depth(
            env,
            sensor_cfg=sensor_cfg,
            cutoff_distance=cutoff_distance,
            min_depth=min_depth,
            crop_left=crop_left,
            crop_right=crop_right,
            gaussian_blur=gaussian_blur,
        )
        sensor = _scene_sensor(env, sensor_cfg)
        camera_frame = _as_torch(sensor.frame).to(device=self.device, dtype=torch.long).reshape(-1)
        if camera_frame.numel() == 1 and self.num_envs > 1:
            camera_frame = camera_frame.expand(self.num_envs)
        if camera_frame.numel() != self.num_envs:
            raise ValueError(
                f"Camera frame counter has {camera_frame.numel()} entries for {self.num_envs} environments."
            )
        update_mask = self._needs_reset | (camera_frame != self._last_camera_frame)

        if self._history is None:
            self._history = frame.unsqueeze(1).repeat(1, frame_history_length, 1, 1, 1)
            update_mask = torch.ones_like(update_mask)
            reset_mask = update_mask
        else:
            expected_shape = (self.num_envs, frame_history_length, *frame.shape[1:])
            if tuple(self._history.shape) != expected_shape:
                raise ValueError(
                    f"Depth history shape changed from {tuple(self._history.shape)} to {expected_shape}."
                )
            reset_mask = update_mask & self._needs_reset
            shift_mask = update_mask & ~reset_mask
            if torch.any(shift_mask):
                self._history[shift_mask, :-1] = self._history[shift_mask, 1:].clone()
                self._history[shift_mask, -1] = frame[shift_mask]
            if torch.any(reset_mask):
                self._history[reset_mask] = frame[reset_mask].unsqueeze(1).repeat(
                    1,
                    frame_history_length,
                    1,
                    1,
                    1,
                )

        self._last_camera_frame[update_mask] = camera_frame[update_mask]
        self._needs_reset[update_mask] = False
        return self._history.reshape(self.num_envs, -1)


def pie_height_scan(
    env,
    sensor_cfg: SceneEntityCfg,
    max_height: float = 5.0,
) -> torch.Tensor:
    """Return normalized base-relative terrain heights, replacing ray misses."""
    if max_height <= 0.0:
        raise ValueError("PIE height max_height must be positive.")
    sensor = _scene_sensor(env, sensor_cfg)
    sensor_height = _as_torch(sensor.data.pos_w)[:, 2].unsqueeze(1)
    hit_height = _as_torch(sensor.data.ray_hits_w)[..., 2]
    height = sensor_height - hit_height
    height = torch.where(torch.isfinite(height), height, torch.full_like(height, max_height))
    return height.clamp(min=0.0, max=max_height) / max_height


def pie_foot_clearance(
    env,
    sensor_cfgs: tuple[SceneEntityCfg, ...],
    max_clearance: float = 0.6,
) -> torch.Tensor:
    """Return one normalized downward ray clearance for each foot."""
    if max_clearance <= 0.0:
        raise ValueError("PIE foot max_clearance must be positive.")
    if not sensor_cfgs:
        raise ValueError("PIE foot clearance requires at least one sensor.")

    clearances = []
    for sensor_cfg in sensor_cfgs:
        sensor = _scene_sensor(env, sensor_cfg)
        sensor_height = _as_torch(sensor.data.pos_w)[:, 2].unsqueeze(1)
        hit_height = _as_torch(sensor.data.ray_hits_w)[..., 2]
        clearance = sensor_height - hit_height
        clearance = torch.where(
            torch.isfinite(clearance),
            clearance,
            torch.full_like(clearance, max_clearance),
        )
        clearances.append(clearance.amin(dim=1, keepdim=True))
    return torch.cat(clearances, dim=1).clamp(min=0.0, max=max_clearance) / max_clearance


def pie_successor_valid(env) -> torch.Tensor:
    """Return rollout placeholders overwritten by PIEPPO after ``env.step``."""
    return torch.ones((env.num_envs, 1), device=env.device)

