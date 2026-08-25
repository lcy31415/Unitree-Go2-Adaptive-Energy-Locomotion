"""Tensor-level tests for PIE depth, terrain and foot observations."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from isaaclab.managers import ObservationTermCfg, SceneEntityCfg

from unitree_rl_lab.tasks.locomotion.mdp.pie_observations import (
    PIEDepthHistory,
    pie_foot_clearance,
    pie_height_scan,
    pie_successor_valid,
    preprocess_pie_depth,
)


class TensorProxy:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.torch = tensor


class FakeCamera:
    def __init__(self, depth: torch.Tensor) -> None:
        self.frame = torch.zeros(depth.shape[0], dtype=torch.long)
        self.data = SimpleNamespace(output={"distance_to_camera": TensorProxy(depth)})


class FakeEnv:
    def __init__(self, sensors: dict, num_envs: int) -> None:
        self.num_envs = num_envs
        self.device = "cpu"
        self.scene = SimpleNamespace(sensors=sensors)


def test_depth_preprocessing_shape_crop_invalid_and_range() -> None:
    depth = torch.ones(2, 60, 106, 1)
    depth[0, 20, 20, 0] = float("inf")
    depth[0, 21, 20, 0] = float("nan")
    depth[0, 22, 20, 0] = 0.0
    processed = preprocess_pie_depth(depth)

    assert processed.shape == (2, 1, 60, 86)
    assert torch.isfinite(processed).all()
    assert processed.min() >= 0.05 / 3.0
    assert processed.max() <= 1.0
    torch.testing.assert_close(processed[1], torch.full_like(processed[1], 1.0 / 3.0))


def test_depth_preprocessing_rejects_invalid_configuration() -> None:
    depth = torch.ones(1, 60, 106, 1)
    with pytest.raises(ValueError, match="removes all"):
        preprocess_pie_depth(depth, crop_left=53, crop_right=53)
    with pytest.raises(ValueError, match="odd kernel"):
        preprocess_pie_depth(depth, gaussian_blur=(2, 1.0))


def test_depth_history_updates_only_on_new_camera_frames_and_resets_per_env() -> None:
    raw_depth = torch.ones(2, 60, 106, 1)
    camera = FakeCamera(raw_depth)
    env = FakeEnv({"front_depth": camera}, num_envs=2)
    term = PIEDepthHistory(ObservationTermCfg(func=PIEDepthHistory), env)
    sensor_cfg = SceneEntityCfg("front_depth")

    initial = term(env, sensor_cfg, gaussian_blur=None)
    assert initial.shape == (2, 2 * 60 * 86)
    torch.testing.assert_close(initial, torch.full_like(initial, 1.0 / 3.0))

    raw_depth.fill_(2.0)
    unchanged = term(env, sensor_cfg, gaussian_blur=None)
    torch.testing.assert_close(unchanged, initial)

    camera.frame += 1
    shifted = term(env, sensor_cfg, gaussian_blur=None).reshape(2, 2, 60, 86)
    torch.testing.assert_close(shifted[:, 0], torch.full_like(shifted[:, 0], 1.0 / 3.0))
    torch.testing.assert_close(shifted[:, 1], torch.full_like(shifted[:, 1], 2.0 / 3.0))

    raw_depth[1].fill_(3.0)
    camera.frame[1] = 0
    term.reset([1])
    reset_history = term(env, sensor_cfg, gaussian_blur=None).reshape(2, 2, 60, 86)
    torch.testing.assert_close(reset_history[1], torch.ones_like(reset_history[1]))
    torch.testing.assert_close(reset_history[0], shifted[0])


def test_height_scan_replaces_misses_and_normalizes() -> None:
    pos_w = TensorProxy(torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 1.0]]))
    hits = TensorProxy(
        torch.tensor(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, float("inf")]],
                [[0.0, 0.0, 0.5], [0.0, 0.0, -10.0]],
            ]
        )
    )
    sensor = SimpleNamespace(data=SimpleNamespace(pos_w=pos_w, ray_hits_w=hits))
    env = FakeEnv({"height": sensor}, num_envs=2)

    output = pie_height_scan(env, SceneEntityCfg("height"), max_height=5.0)
    expected = torch.tensor([[0.1, 1.0], [0.1, 1.0]])
    torch.testing.assert_close(output, expected)


def test_four_foot_clearances_and_successor_placeholder() -> None:
    sensors = {}
    expected = []
    sensor_cfgs = []
    for index, name in enumerate(("fr", "fl", "rr", "rl"), start=1):
        clearance = 0.1 * index
        pos_w = TensorProxy(torch.tensor([[0.0, 0.0, clearance], [0.0, 0.0, clearance]]))
        hits = TensorProxy(
            torch.tensor(
                [
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.02]],
                    [[0.0, 0.0, float("inf")], [0.0, 0.0, float("inf")]],
                ]
            )
        )
        sensors[name] = SimpleNamespace(data=SimpleNamespace(pos_w=pos_w, ray_hits_w=hits))
        sensor_cfgs.append(SceneEntityCfg(name))
        expected.append((clearance - 0.02) / 0.6)
    env = FakeEnv(sensors, num_envs=2)

    output = pie_foot_clearance(env, tuple(sensor_cfgs), max_clearance=0.6)
    torch.testing.assert_close(output[0], torch.tensor(expected))
    torch.testing.assert_close(output[1], torch.ones(4))
    torch.testing.assert_close(pie_successor_valid(env), torch.ones(2, 1))

