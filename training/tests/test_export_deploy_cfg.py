"""Tests for legacy and multi-input deployment observation selection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from unitree_rl_lab.utils.export_deploy_cfg import _resolve_observation_groups


def _fake_env(*group_names: str):
    manager = SimpleNamespace(active_terms={name: [] for name in group_names})
    return SimpleNamespace(observation_manager=manager)


def test_legacy_policy_group_is_inferred_unchanged():
    env = _fake_env("policy", "critic")
    assert _resolve_observation_groups(env) == ["policy"]


def test_pie_actor_groups_are_preserved_in_requested_order():
    env = _fake_env("actor", "proprio_history", "camera", "critic")
    requested = ["actor", "proprio_history", "camera"]
    assert _resolve_observation_groups(env, requested) == requested


def test_missing_requested_group_fails_with_available_groups():
    env = _fake_env("actor", "camera")
    with pytest.raises(KeyError, match="proprio_history"):
        _resolve_observation_groups(env, ["actor", "proprio_history", "camera"])
