from __future__ import annotations

from typing import Any

from envs.multi_episode_env import MultiEpisodeEnv


class _InnerAlwaysDoneEnv:
    def __init__(self) -> None:
        self._reset_count = 0

    def reset(self, seed: int | None = None, task: dict | None = None) -> tuple[str, dict[str, Any]]:
        del seed, task
        obs = f"init-{self._reset_count}"
        self._reset_count += 1
        return obs, {"inner_reset_count": self._reset_count}

    def step(self, action: Any) -> tuple[str, float, bool, dict[str, Any]]:
        del action
        terminal_obs = f"terminal-{self._reset_count - 1}"
        return terminal_obs, 1.0, True, {"terminated": True, "truncated": False}

    def close(self) -> None:
        return


def test_boundary_metadata_non_reflection_combined_observation() -> None:
    env = MultiEpisodeEnv(
        inner_env_class=_InnerAlwaysDoneEnv,
        total_step_cap=3,
        episode_header="New episode begins.",
        enable_reflection=False,
    )
    try:
        _, info0 = env.reset(seed=1, task={"seed": 1})
        assert info0["boundary_transition"] is False
        assert info0["boundary_has_combined_observation"] is False
        assert info0["boundary_terminal_observation"] == ""
        assert info0["boundary_next_initial_observation"] == ""

        obs1, reward1, done1, info1 = env.step("act")
        expected_next_initial = "[Episode 2] New episode begins.\ninit-1"
        assert reward1 == 1.0
        assert done1 is False
        assert obs1 == f"terminal-0\n\n{expected_next_initial}"
        assert info1["boundary_transition"] is True
        assert info1["boundary_has_combined_observation"] is True
        assert info1["boundary_terminal_observation"] == "terminal-0"
        assert info1["boundary_next_initial_observation"] == expected_next_initial
    finally:
        env.close()


def test_boundary_metadata_reflection_reset_step() -> None:
    env = MultiEpisodeEnv(
        inner_env_class=_InnerAlwaysDoneEnv,
        total_step_cap=3,
        episode_header="New episode begins.",
        enable_reflection=True,
    )
    try:
        env.reset(seed=1, task={"seed": 1})
        _, reward1, done1, info1 = env.step("act")
        assert reward1 == 1.0
        assert done1 is False
        assert info1["boundary_transition"] is False

        obs2, reward2, done2, info2 = env.step("reflection")
        expected_next_initial = "[Episode 2] New episode begins.\ninit-1"
        assert obs2 == expected_next_initial
        assert reward2 == 0.0
        assert done2 is False
        assert info2["boundary_transition"] is True
        assert info2["boundary_has_combined_observation"] is False
        assert info2["boundary_terminal_observation"] == ""
        assert info2["boundary_next_initial_observation"] == expected_next_initial
    finally:
        env.close()

"""Focused tests for MultiEpisodeEnv integrations."""

from __future__ import annotations

import pytest

from envs.multi_episode_env import MultiEpisodeEnv


def test_frozenlake_adapter_episode_success_metadata() -> None:
    """MultiEpisodeEnv should detect success from FrozenLake adapter metadata."""
    pytest.importorskip("gymnasium")
    from envs.frozenlake_env_adapter import FrozenLakeEnvAdapter

    env = MultiEpisodeEnv(
        inner_env_class=FrozenLakeEnvAdapter,
        inner_env_kwargs={
            "env_id": "frozenlake",
            "env_kwargs": {
                "desc": ["SG", "FF"],
                "is_slippery": False,
                "max_turns": 3,
                "seed": 21,
            },
        },
        total_step_cap=3,
        success_reward=1.0,
        episode_header="New episode begins.",
        enable_reflection=False,
    )

    try:
        env.reset(seed=21, task={"seed": 21})
        _, reward, done, info = env.step(r"\boxed{right}")

        assert reward == 1.0
        assert done is False
        assert info["episode_done"] is True
        assert info["episode_success"] is True
        assert info["raw_reward"] == 1.0
        assert info["episode_index"] == 1
        assert info["episode_step"] == 0
    finally:
        env.close()


def test_frozenlake_reset_inner_env_reuses_same_task() -> None:
    """Internal reset should preserve the same FrozenLake task across episodes."""
    pytest.importorskip("gymnasium")
    from envs.frozenlake_env_adapter import FrozenLakeEnvAdapter

    def _snapshot(inner_adapter: FrozenLakeEnvAdapter) -> tuple[tuple[bytes, ...], int]:
        rows = tuple(b"".join(row.tolist()) for row in inner_adapter._env.desc)  # type: ignore[attr-defined]
        state = int(inner_adapter._env.s)  # type: ignore[attr-defined]
        return rows, state

    task = {
        "seed": 21,
        "desc": ["SG", "FF"],
        "is_slippery": False,
        "max_turns": 3,
    }
    env = MultiEpisodeEnv(
        inner_env_class=FrozenLakeEnvAdapter,
        inner_env_kwargs={
            "env_id": "frozenlake",
            "env_kwargs": {
                "desc": task["desc"],
                "is_slippery": task["is_slippery"],
                "max_turns": task["max_turns"],
                "seed": task["seed"],
            },
        },
        total_step_cap=4,
        success_reward=1.0,
        episode_header="New episode begins.",
        enable_reflection=False,
    )

    try:
        env.reset(seed=task["seed"], task=task)
        first_snapshot = _snapshot(env.inner_env)  # type: ignore[arg-type]

        # This terminal action triggers internal reset to episode 2.
        _, _, _, info = env.step(r"\boxed{right}")
        second_snapshot = _snapshot(env.inner_env)  # type: ignore[arg-type]

        assert info["episode_done"] is True
        assert info["episode_index"] == 1
        assert info["episode_step"] == 0
        assert first_snapshot == second_snapshot
    finally:
        env.close()
