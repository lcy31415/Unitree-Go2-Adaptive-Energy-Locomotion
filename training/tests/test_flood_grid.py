"""Contracts for the heterogeneous stairs + flat flood-fill curriculum."""

from __future__ import annotations

import math
from types import SimpleNamespace

import gymnasium as gym
import torch

import unitree_rl_lab.tasks.locomotion.robots.go2  # noqa: F401

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.agents.pie_cfg import (
    AdaptiveEnergyStairsPIEFloodFillRunnerCfg,
)
from unitree_rl_lab.tasks.locomotion.mdp.flood_grid import (
    STATE_ACTIVE,
    STATE_LOCKED,
    STATE_MASTERED,
    FloodGridFamily,
    FloodGridCurriculum,
    FloodFillVelocityCommand,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.adaptive_energy_pie_terrain_cfg import (
    ADAPTIVE_ENERGY_PIE_COLUMNS_PER_FAMILY,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.adaptive_energy_stairs_pie_floodfill_env_cfg import (
    FLOODFILL_ACTIVE_FAMILIES,
    FLOODFILL_FAMILY_WEIGHTS,
    FLOODFILL_GRIDS,
    AdaptiveEnergyStairsPIEFloodFillEnvCfg,
    AdaptiveEnergyStairsPIEFloodFillPlayEnvCfg,
)


TASK_ID = "Unitree-Go2-Adaptive-Energy-stairs-PIE-FloodFill"


def _family(**kwargs) -> FloodGridFamily:
    return FloodGridFamily(
        "test",
        level_count=3,
        vx_edges=(0.0, 0.5, 1.0),
        yaw_edges=(0.0, 0.5, 1.0),
        device="cpu",
        **kwargs,
    )


def test_easy_corner_seed_can_master_and_really_flood_to_axial_neighbors():
    grid = _family()
    seed = torch.tensor([[0, 0, 0]], dtype=torch.long)
    assert grid.seed_cell == (0, 0, 0)
    assert grid.state[0, 0, 0] == STATE_ACTIVE
    assert torch.count_nonzero(grid.state == STATE_ACTIVE) == 1

    events = grid.update(seed.repeat(8, 1), torch.ones(8, dtype=torch.bool))

    assert events == {"mastered": 1, "forgotten": 0, "activated": 3}
    assert grid.state[0, 0, 0] == STATE_MASTERED
    assert grid.ep_count[0, 0, 0] == 8
    assert grid.succ_ema[0, 0, 0] >= 0.70
    assert torch.count_nonzero(grid.state == STATE_ACTIVE) == 3
    # Only the three axial nearest neighbours activate at the easy corner.
    assert grid.state[1, 0, 0] == STATE_ACTIVE
    assert grid.state[0, 1, 0] == STATE_ACTIVE
    assert grid.state[0, 0, 1] == STATE_ACTIVE
    assert grid.state[1, 1, 0] == STATE_LOCKED
    assert grid.state[1, 0, 1] == STATE_LOCKED
    assert grid.state[0, 1, 1] == STATE_LOCKED
    assert grid.state[1, 1, 1] == STATE_LOCKED


def test_episode_count_drives_boost_consumption_and_forgetting():
    grid = _family()
    seed = torch.tensor([[0, 0, 0]], dtype=torch.long)
    grid.update(seed.repeat(8, 1), torch.ones(8, dtype=torch.bool))

    events = grid.update(seed.repeat(6, 1), torch.zeros(6, dtype=torch.bool))

    assert events["forgotten"] == 1
    assert grid.state[0, 0, 0] == STATE_ACTIVE
    assert grid.episodes_since_master[0, 0, 0] == 0

    boosted = torch.nonzero(grid.boost_left > 0, as_tuple=False)[0].unsqueeze(0)
    before = int(grid.boost_left[tuple(boosted[0])])
    grid.update(boosted.repeat(5, 1), torch.zeros(5, dtype=torch.bool))
    assert int(grid.boost_left[tuple(boosted[0])]) == max(0, before - 5)


def test_locked_cells_have_zero_sampling_probability_and_commands_stay_in_bins():
    grid = FloodGridFamily(
        "flat",
        level_count=1,
        vx_edges=tuple(index * 0.5 for index in range(11)),
        yaw_edges=tuple(index * 0.5 for index in range(11)),
        device="cpu",
    )
    weights = grid.sampling_weights()
    assert torch.all(weights[grid.state == STATE_LOCKED] == 0.0)

    cells, commands = grid.sample(256)
    assert torch.all(cells == torch.tensor([0, 0, 0]))
    assert torch.all(commands[:, 0].abs() >= 0.0)
    assert torch.all(commands[:, 0].abs() <= 0.5)
    assert torch.all(commands[:, 1] == 0.0)
    assert torch.all(commands[:, 2].abs() >= 0.0)
    assert torch.all(commands[:, 2].abs() <= 0.5)


def test_stair_sampling_is_forward_only_and_has_exactly_zero_yaw():
    torch.manual_seed(7)
    grid = FloodGridFamily(
        "stairs_up",
        level_count=1,
        vx_edges=(0.2, 1.5),
        yaw_edges=(0.0, 0.3),
        device="cpu",
        forward_only=True,
        zero_yaw_probability=1.0,
        minimum_progress=3.2,
    )

    _, commands = grid.sample(20_000)

    assert torch.all((commands[:, 0] >= 0.2) & (commands[:, 0] <= 1.5))
    assert torch.all(commands[:, 2].abs() <= 0.3)
    assert torch.all(commands[:, 2] == 0.0)
    assert grid.minimum_progress == 3.2


def test_old_five_yaw_bin_checkpoint_migrates_to_zero_yaw_slice():
    old = FloodGridFamily(
        "stairs_up",
        level_count=2,
        vx_edges=(0.2, 0.8, 1.5),
        yaw_edges=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
        device="cpu",
    )
    old.state[1, 1, 0] = STATE_MASTERED
    old.ep_count[1, 1, 0] = 23
    # This obsolete nonzero-yaw state must not leak into the straight task.
    old.state[0, 0, 4] = STATE_MASTERED

    straight = FloodGridFamily(
        "stairs_up",
        level_count=2,
        vx_edges=(0.2, 0.8, 1.5),
        yaw_edges=(0.0, 1.0e-6),
        device="cpu",
        forward_only=True,
        zero_yaw_probability=1.0,
    )
    straight.load_state_dict(old.state_dict())

    assert straight.state.shape == (2, 2, 1)
    assert straight.state[1, 1, 0] == STATE_MASTERED
    assert straight.ep_count[1, 1, 0] == 23
    assert straight.state[0, 0, 0] == old.state[0, 0, 0]


def test_family_state_dict_round_trip():
    grid = _family()
    seed = torch.tensor([[0, 0, 0]], dtype=torch.long)
    grid.update(seed.repeat(8, 1), torch.ones(8, dtype=torch.bool))
    state = grid.state_dict()

    restored = _family()
    restored.load_state_dict(state)

    for name in ("state", "succ_ema", "ep_count", "boost_left", "episodes_since_master"):
        assert torch.equal(getattr(restored, name), getattr(grid, name))


def test_curriculum_checkpoint_can_restore_with_a_different_env_count():
    params = {
        "command_name": "base_velocity",
        "terrain_family_names": ("flat", "stairs_up", "stairs_down"),
        "active_family_indices": (0, 1, 2),
        "columns_per_family": 4,
        "family_grids": {
            "flat": {"levels": 1, "vx_edges": (0.0, 1.0), "yaw_edges": (0.0, 0.4)},
            "stairs_up": {"levels": 2, "vx_edges": (0.0, 1.0), "yaw_edges": (0.0, 1.0)},
            "stairs_down": {"levels": 2, "vx_edges": (0.0, 1.0), "yaw_edges": (0.0, 1.0)},
        },
    }
    original = FloodGridCurriculum(
        SimpleNamespace(params=params),
        SimpleNamespace(num_envs=4, device="cpu"),
    )
    original.grids[1].ep_count[1, 0, 0] = 17
    state = original.state_dict()

    restored = FloodGridCurriculum(
        SimpleNamespace(params=params),
        SimpleNamespace(num_envs=7, device="cpu"),
    )
    restored.load_state_dict(state)

    assert restored._cells.shape == (7, 3)
    assert restored.grids[1].ep_count[1, 0, 0] == 17
    metrics = restored.metrics()
    assert metrics["active"] == 3
    assert metrics["locked"] == 2


def test_task_uses_heterogeneous_grids_and_requested_family_budget():
    cfg = AdaptiveEnergyStairsPIEFloodFillEnvCfg()
    assert cfg.commands.base_velocity.class_type is FloodFillVelocityCommand
    assert cfg.curriculum.flood_grid.func is mdp.FloodGridCurriculum
    assert FLOODFILL_ACTIVE_FAMILIES == (0, 1, 2)
    assert FLOODFILL_FAMILY_WEIGHTS == (0.10, 0.45, 0.45, 0.0, 0.0, 0.0, 0.0)
    assert math.isclose(sum(FLOODFILL_FAMILY_WEIGHTS), 1.0)
    assert (
        FLOODFILL_GRIDS["flat"]["levels"],
        len(FLOODFILL_GRIDS["flat"]["vx_edges"]) - 1,
        len(FLOODFILL_GRIDS["flat"]["yaw_edges"]) - 1,
    ) == (1, 10, 10)
    assert FLOODFILL_GRIDS["flat"]["yaw_edges"] == FLOODFILL_GRIDS["flat"]["vx_edges"]
    assert FLOODFILL_GRIDS["flat"]["tracking_tolerances"] == (0.10, 0.15, 0.15)
    assert cfg.commands.base_velocity.ranges.ang_vel_z == (-5.0, 5.0)
    for name in ("stairs_up", "stairs_down"):
        assert (
            FLOODFILL_GRIDS[name]["levels"],
            len(FLOODFILL_GRIDS[name]["vx_edges"]) - 1,
            len(FLOODFILL_GRIDS[name]["yaw_edges"]) - 1,
        ) == (10, 5, 1)
        assert FLOODFILL_GRIDS[name]["tracking_tolerances"] == (0.25, 0.25, 0.30)
        assert FLOODFILL_GRIDS[name]["vx_edges"][0] == 0.2
        assert FLOODFILL_GRIDS[name]["vx_edges"][-1] == 1.5
        assert FLOODFILL_GRIDS[name]["yaw_edges"] == (0.0, 1.0e-6)
        assert FLOODFILL_GRIDS[name]["forward_only"] is True
        assert FLOODFILL_GRIDS[name]["zero_yaw_probability"] == 1.0
        assert FLOODFILL_GRIDS[name]["minimum_progress"] == 3.2
        assert FLOODFILL_GRIDS[name]["success_tracking_fraction"] == 0.45
    assert cfg.events.allocate_terrain_families.params["family_weights"] == FLOODFILL_FAMILY_WEIGHTS

    num_envs = 100
    terrain = SimpleNamespace(
        terrain_types=torch.zeros(num_envs, dtype=torch.long),
        terrain_levels=torch.zeros(num_envs, dtype=torch.long),
        terrain_origins=torch.zeros(10, 28, 3),
        env_origins=torch.zeros(num_envs, 3),
    )
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene=SimpleNamespace(terrain=terrain),
    )
    mdp.allocate_terrain_families(
        env,
        None,
        FLOODFILL_FAMILY_WEIGHTS,
        ADAPTIVE_ENERGY_PIE_COLUMNS_PER_FAMILY,
    )
    family_ids = torch.div(
        terrain.terrain_types,
        ADAPTIVE_ENERGY_PIE_COLUMNS_PER_FAMILY,
        rounding_mode="floor",
    )
    assert torch.bincount(family_ids, minlength=7).tolist() == [10, 45, 45, 0, 0, 0, 0]


def test_task_registration_runner_and_play_layout():
    kwargs = gym.spec(TASK_ID).kwargs
    assert kwargs["env_cfg_entry_point"].endswith(":AdaptiveEnergyStairsPIEFloodFillEnvCfg")
    assert kwargs["play_env_cfg_entry_point"].endswith(":AdaptiveEnergyStairsPIEFloodFillPlayEnvCfg")
    assert kwargs["rsl_rl_cfg_entry_point"].endswith(":AdaptiveEnergyStairsPIEFloodFillRunnerCfg")

    runner = AdaptiveEnergyStairsPIEFloodFillRunnerCfg()
    assert runner.experiment_name == "unitree_go2_adaptive_energy_stairs_pie_floodfill"
    assert runner.max_iterations == 30_000
    assert runner.save_interval == 200

    play = AdaptiveEnergyStairsPIEFloodFillPlayEnvCfg()
    assert play.scene.num_envs == 3
    assert play.events.allocate_terrain_families.params["family_weights"][:3] == (1.0, 1.0, 1.0)
    assert play.curriculum is None
