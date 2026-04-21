"""Focused tests for MultiEpisodeEnv integrations."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import pytest

from envs.multi_episode_env import MultiEpisodeEnv


class _ScriptedInnerEnv:
    """Minimal BaseEnv-compatible inner env for unit testing MultiEpisodeEnv.

    Each episode lasts `episode_length` steps and always succeeds on the final
    step. Enough to exercise episode-boundary and termination logic without
    pulling in gymnasium or GEM.
    """

    def __init__(
        self,
        episode_length: int = 2,
        max_turns: int = 10,
        **kwargs: Any,
    ) -> None:
        self.episode_length = episode_length
        self.max_turns = max_turns
        self._step_count = 0

    def reset(
        self, seed: Optional[int] = None, task: Optional[dict] = None
    ) -> Tuple[str, dict]:
        self._step_count = 0
        return "obs-0", {}

    def step(self, action: Any) -> Tuple[str, float, bool, dict]:
        self._step_count += 1
        done = self._step_count >= self.episode_length
        reward = 1.0 if done else 0.0
        info = {"terminated": done, "truncated": False}
        return f"obs-{self._step_count}", reward, done, info

    def close(self) -> None:
        pass


def _make_env(num_episodes: Optional[int], total_step_cap: int, episode_length: int = 2) -> MultiEpisodeEnv:
    return MultiEpisodeEnv(
        inner_env_class=_ScriptedInnerEnv,
        inner_env_kwargs={"episode_length": episode_length, "max_turns": episode_length},
        total_step_cap=total_step_cap,
        success_reward=1.0,
        episode_header="New episode begins.",
        enable_reflection=False,
        num_episodes=num_episodes,
    )


def _run_until_done(env: MultiEpisodeEnv) -> list[dict]:
    env.reset()
    infos: list[dict] = []
    done = False
    # Safety guard so a bug can't hang the test.
    for _ in range(1000):
        _, _, done, info = env.step("noop")
        infos.append(info)
        if done:
            break
    assert done, "trajectory never terminated within safety guard"
    return infos


def test_num_episodes_none_is_step_cap_only() -> None:
    """Default (num_episodes=None) preserves existing behaviour: run to step cap."""
    env = _make_env(num_episodes=None, total_step_cap=6, episode_length=2)
    try:
        infos = _run_until_done(env)
        metrics = env.get_metrics()
        assert metrics["episode/num_episodes"] == 3  # 6 steps / 2 steps per episode
        assert metrics["episode/total_steps"] == 6
        assert metrics["episode/truncated"] == 0.0
        assert infos[-1]["total_step"] == 6
    finally:
        env.close()


def test_num_episodes_caps_trajectory() -> None:
    """num_episodes should end the trajectory after exactly N completed episodes."""
    env = _make_env(num_episodes=2, total_step_cap=100, episode_length=2)
    try:
        _run_until_done(env)
        metrics = env.get_metrics()
        assert metrics["episode/num_episodes"] == 2
        assert metrics["episode/success_count"] == 2
        assert metrics["episode/total_steps"] == 4  # 2 episodes * 2 steps
        assert metrics["episode/truncated"] == 0.0
        # Per-episode keys for 1 and 2 should exist; episode_3 should not.
        assert "episode_1/success_rate" in metrics
        assert "episode_2/success_rate" in metrics
        assert "episode_3/success_rate" not in metrics
    finally:
        env.close()


def test_total_step_cap_wins_when_tighter() -> None:
    """total_step_cap still bounds the trajectory when it fires before num_episodes."""
    # num_episodes=5 but only 3 steps of budget => cap wins, trajectory stops with
    # fewer than 5 completed episodes at _total_steps == total_step_cap.
    env = _make_env(num_episodes=5, total_step_cap=3, episode_length=2)
    try:
        _run_until_done(env)
        metrics = env.get_metrics()
        # Step cap fires before 5 episodes complete.
        assert metrics["episode/num_episodes"] < 5
        assert metrics["episode/total_steps"] == 3
        # With num_episodes=5, per-episode slot count is 5 even though only a
        # subset were actually run — unattempted slots are filled with -1 sentinels.
        assert "episode_5/success_rate" in metrics
    finally:
        env.close()


def test_from_dict_auto_derives_step_cap_from_num_episodes() -> None:
    """When only num_episodes is provided in a task, total_step_cap is derived."""
    env = MultiEpisodeEnv.from_dict(
        {
            "inner_env_class": _ScriptedInnerEnv,
            "inner_env_kwargs": {"episode_length": 2, "max_turns": 2},
            # No total_step_cap; num_episodes + max_turns_per_episode should yield 6.
            "num_episodes": 3,
            "max_turns_per_episode": 2,
        }
    )
    try:
        assert env.num_episodes == 3
        assert env.total_step_cap == 6
    finally:
        env.close()


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
