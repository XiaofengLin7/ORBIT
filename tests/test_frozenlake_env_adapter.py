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
    assert "You can take at most 3 steps in this episode." in observation
    assert info["terminated"] is False
    assert info["truncated"] is False
    assert info["turn"] == 0
    assert info["max_turns"] == 3


def test_step_accepts_boxed_direction_and_signals_success(frozenlake_env) -> None:
    """A boxed directional action should be parsed and succeed on a simple map."""
    frozenlake_env.reset()
    observation, reward, done, info = frozenlake_env.step(r"\boxed{right}")

    assert reward == 1.0
    assert done is True
    assert info["terminated"] is True
    assert info["truncated"] is False
    assert info["raw_reward"] == 1.0
    assert info["is_correct"] is True
    assert "Congratulations! You successfully reached the goal." in observation
    assert "Action feedback:" not in observation


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
        observation, reward, done, info = env.step("invalid-action")
        assert reward == 0.0
        assert done is True
        assert info["terminated"] is False
        assert info["truncated"] is True
        assert info["turn"] == 1
        assert "You failed the task because you reached the maximum number of turns (1/1) before reaching the goal." in observation
        assert "Action feedback:" not in observation
    finally:
        env.close()


def test_step_reports_hole_failure_reason() -> None:
    """Stepping into a hole should provide an explicit failure reason."""
    pytest.importorskip("gymnasium")
    from envs.frozenlake_env_adapter import FrozenLakeEnvAdapter

    env = FrozenLakeEnvAdapter(
        env_id="frozenlake",
        env_kwargs={
            "desc": ["SH", "FG"],
            "is_slippery": False,
            "max_turns": 3,
            "seed": 31,
        },
    )
    try:
        env.reset()
        observation, reward, done, info = env.step(r"\boxed{right}")
        assert reward == 0.0
        assert done is True
        assert info["terminated"] is True
        assert info["truncated"] is False
        assert "You failed the task because you stepped into a hole (`X`)." in observation
        assert "Action feedback:" not in observation
    finally:
        env.close()


def test_step_non_terminal_has_no_success_or_failure_feedback() -> None:
    """Non-terminal steps should not include success/failure feedback."""
    pytest.importorskip("gymnasium")
    from envs.frozenlake_env_adapter import FrozenLakeEnvAdapter

    env = FrozenLakeEnvAdapter(
        env_id="frozenlake",
        env_kwargs={
            "desc": ["SF", "FG"],
            "is_slippery": False,
            "max_turns": 3,
            "seed": 97,
        },
    )
    try:
        env.reset()
        observation, reward, done, info = env.step(r"\boxed{down}")
        assert done is False
        assert reward == 0.0
        assert info["terminated"] is False
        assert info["truncated"] is False
        assert "You failed" not in observation
        assert "Congratulations!" not in observation
        assert "Output your next action in \\boxed{up/down/left/right}." in observation
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


def test_reset_includes_shortest_path_length_for_fixed_map() -> None:
    """Reset metadata should include shortest path length for explicit maps."""
    pytest.importorskip("gymnasium")
    from envs.frozenlake_env_adapter import FrozenLakeEnvAdapter

    env = FrozenLakeEnvAdapter(
        env_id="frozenlake",
        env_kwargs={
            "desc": ["SF", "FG"],
            "is_slippery": False,
            "max_turns": 3,
            "seed": 5,
            "shortest_path_min_length": 2,
            "shortest_path_max_length": 2,
        },
    )
    try:
        _, info = env.reset()
        assert info["shortest_path_length"] == 2
    finally:
        env.close()


def test_reset_retries_seed_for_shortest_path_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset should retry seed sampling until shortest-path bounds are satisfied."""
    pytest.importorskip("gymnasium")
    from envs.frozenlake_env_adapter import FrozenLakeEnvAdapter

    sampled_lengths = [1, 4]

    def _fake_shortest_path(_cls, _desc):
        return sampled_lengths.pop(0)

    monkeypatch.setattr(
        FrozenLakeEnvAdapter,
        "_compute_shortest_path_length",
        classmethod(_fake_shortest_path),
    )

    env = FrozenLakeEnvAdapter(
        env_id="frozenlake",
        env_kwargs={
            "size": 4,
            "p": 0.8,
            "is_slippery": False,
            "max_turns": 4,
            "seed": 100,
            "shortest_path_min_length": 4,
            "shortest_path_max_length": 4,
        },
    )
    try:
        _, info = env.reset()
        assert info["shortest_path_length"] == 4
        assert env._seed == 101  # type: ignore[attr-defined]
    finally:
        env.close()


def test_reset_raises_when_fixed_map_is_out_of_range() -> None:
    """Explicit desc should fail fast when shortest-path bounds are unsatisfied."""
    pytest.importorskip("gymnasium")
    from envs.frozenlake_env_adapter import FrozenLakeEnvAdapter

    env = FrozenLakeEnvAdapter(
        env_id="frozenlake",
        env_kwargs={
            "desc": ["SF", "FG"],
            "is_slippery": False,
            "max_turns": 3,
            "seed": 5,
            "shortest_path_min_length": 5,
            "shortest_path_max_length": 5,
        },
    )
    try:
        with pytest.raises(ValueError, match="does not satisfy shortest-path range"):
            env.reset()
    finally:
        env.close()
