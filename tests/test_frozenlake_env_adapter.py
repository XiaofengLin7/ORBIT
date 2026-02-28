"""Tests for FrozenLakeEnvAdapter."""

from __future__ import annotations

import pytest

from rllm.agents.agent import Action  # type: ignore


@pytest.fixture()
def frozenlake_env():
    """Create a deterministic FrozenLake adapter for tests."""
    pytest.importorskip("gymnasium")
    from envs.frozenlake_env_adapter import FrozenLakeEnvAdapter

    env = FrozenLakeEnvAdapter(
        env_id="frozenlake",
        env_kwargs={
            "desc": ["SG", "FF"],
            "is_slippery": False,
            "max_turns": 3,
            "seed": 7,
        },
    )
    try:
        yield env
    finally:
        env.close()


def test_reset_returns_prompt_and_metadata(frozenlake_env) -> None:
    """Reset should return string observation and normalized metadata."""
    observation, info = frozenlake_env.reset()

    assert isinstance(observation, str)
    assert "Current observation:" in observation
    assert info["terminated"] is False
    assert info["truncated"] is False
    assert info["turn"] == 0
    assert info["max_turns"] == 3


def test_step_accepts_boxed_direction_and_signals_success(frozenlake_env) -> None:
    """A boxed directional action should be parsed and succeed on a simple map."""
    frozenlake_env.reset()
    _, reward, done, info = frozenlake_env.step(r"\boxed{right}")

    assert reward == 1.0
    assert done is True
    assert info["terminated"] is True
    assert info["truncated"] is False
    assert info["raw_reward"] == 1.0
    assert info["is_correct"] is True


def test_step_truncates_when_turn_cap_reached() -> None:
    """Episode should truncate at max_turns when not otherwise terminated."""
    pytest.importorskip("gymnasium")
    from envs.frozenlake_env_adapter import FrozenLakeEnvAdapter

    env = FrozenLakeEnvAdapter(
        env_id="frozenlake",
        env_kwargs={
            "desc": ["SF", "FG"],
            "is_slippery": False,
            "max_turns": 1,
            "seed": 13,
        },
    )
    try:
        env.reset()
        _, reward, done, info = env.step("invalid-action")
        assert reward == 0.0
        assert done is True
        assert info["terminated"] is False
        assert info["truncated"] is True
        assert info["turn"] == 1
    finally:
        env.close()


def test_step_accepts_action_wrapper(frozenlake_env) -> None:
    """Adapter should unwrap rLLM Action payloads."""
    frozenlake_env.reset()
    action = Action(action=r"\boxed{3}")
    _, _, done, info = frozenlake_env.step(action)

    assert done is True
    assert info["parsed_action"] == 3


def test_step_handles_punctuation_only_action_without_crash() -> None:
    """Punctuation-only model output should safely map to no-op action."""
    pytest.importorskip("gymnasium")
    from envs.frozenlake_env_adapter import FrozenLakeEnvAdapter

    env = FrozenLakeEnvAdapter(
        env_id="frozenlake",
        env_kwargs={
            "desc": ["SF", "FG"],
            "is_slippery": False,
            "max_turns": 1,
            "seed": 23,
        },
    )
    try:
        env.reset()
        _, _, done, info = env.step("!!!")
        assert done is True
        assert info["parsed_action"] == 0
    finally:
        env.close()


def test_factory_from_dict() -> None:
    """Factory should construct a usable adapter from nested config."""
    pytest.importorskip("gymnasium")
    from envs.frozenlake_env_adapter import FrozenLakeEnvAdapter

    env = FrozenLakeEnvAdapter.from_dict(
        {
            "env_id": "frozenlake",
            "env_kwargs": {
                "desc": ["SG", "FF"],
                "is_slippery": False,
                "max_turns": 2,
                "seed": 101,
            },
        }
    )
    try:
        observation, info = env.reset()
        assert isinstance(observation, str)
        assert info["max_turns"] == 2
    finally:
        env.close()
