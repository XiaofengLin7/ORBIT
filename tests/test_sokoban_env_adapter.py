"""Tests for SokobanEnvAdapter."""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any, Generator

import numpy as np
import pytest


class _FakeSokobanEnv:
    """Minimal fake of gym_sokoban SokobanEnv used by adapter tests."""

    def __init__(self, dim_room: tuple[int, int], num_boxes: int, max_steps: int) -> None:
        """Store config and initialize deterministic board state."""
        self.dim_room: tuple[int, int] = tuple(dim_room)
        self.num_boxes: int = int(num_boxes)
        self.max_steps: int = int(max_steps)
        self.boxes_on_target: int = 0
        self.closed: bool = False
        self._seed: int | None = None
        self._step_count: int = 0
        self.room_state: np.ndarray
        self.room_fixed: np.ndarray
        self._init_board()

    def _init_board(self) -> None:
        """Build a deterministic board with one target and one box."""
        rows, cols = self.dim_room
        self.room_state = np.ones((rows, cols), dtype=np.int64)
        self.room_fixed = np.ones((rows, cols), dtype=np.int64)
        self.room_state[0, 0] = 5  # player
        target_r = min(1, rows - 1)
        target_c = min(1, cols - 1)
        self.room_fixed[target_r, target_c] = 2  # target
        self.room_state[target_r, target_c] = 4  # box
        self.boxes_on_target = 0

    def seed(self, seed: int) -> None:
        """Store provided seed."""
        self._seed = int(seed)

    def reset(self) -> np.ndarray:
        """Reset fake episode state."""
        self._step_count = 0
        self._init_board()
        return self.room_state

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, bool]]:
        """Apply deterministic transition logic for tests.

        Action ``4`` marks puzzle as solved.
        Action ``0`` is a no-op.
        """
        self._step_count += 1
        moved = action in {1, 2, 3, 4}
        reward = 0.0
        done = False

        if action == 4:
            rows, cols = self.dim_room
            target_r = min(1, rows - 1)
            target_c = min(1, cols - 1)
            self.room_state[target_r, target_c] = 3
            self.boxes_on_target = self.num_boxes
            reward = 1.0
            done = True
        elif action == 0:
            moved = False

        return self.room_state, float(reward), bool(done), {"action.moved_player": moved}

    def close(self) -> None:
        """Record closure for completeness."""
        self.closed = True


@pytest.fixture()
def sokoban_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Import adapter with a fake gym_sokoban backend injected."""
    fake_root = types.ModuleType("gym_sokoban")
    fake_envs = types.ModuleType("gym_sokoban.envs")
    fake_sokoban_env = types.ModuleType("gym_sokoban.envs.sokoban_env")
    fake_sokoban_env.SokobanEnv = _FakeSokobanEnv

    fake_root.envs = fake_envs
    fake_envs.sokoban_env = fake_sokoban_env

    monkeypatch.setitem(sys.modules, "gym_sokoban", fake_root)
    monkeypatch.setitem(sys.modules, "gym_sokoban.envs", fake_envs)
    monkeypatch.setitem(sys.modules, "gym_sokoban.envs.sokoban_env", fake_sokoban_env)

    sys.modules.pop("envs.sokoban_env_adapter", None)
    module = importlib.import_module("envs.sokoban_env_adapter")
    return importlib.reload(module)


@pytest.fixture()
def sokoban_env(sokoban_module: types.ModuleType) -> Generator[Any, None, None]:
    """Provide a deterministic Sokoban adapter instance."""
    env = sokoban_module.SokobanEnvAdapter(
        env_id="sokoban",
        env_kwargs={"dim_room": (4, 4), "num_boxes": 1, "max_turns": 2, "seed": 17},
    )
    try:
        yield env
    finally:
        env.close()


def test_reset_returns_prompt_and_metadata(sokoban_env: Any) -> None:
    """Reset should return prompt text and normalized info metadata."""
    observation, info = sokoban_env.reset()

    assert isinstance(observation, str)
    assert "You are solving a Sokoban puzzle." in observation
    assert "Current observation:" in observation
    assert info["env_id"] == "sokoban"
    assert info["turn"] == 0
    assert info["max_turns"] == 2
    assert info["terminated"] is False
    assert info["truncated"] is False
    assert info["raw_reward"] == 0.0
    assert "shortest_path_length" in info


def test_step_parses_boxed_action_and_terminates_on_success(sokoban_env: Any) -> None:
    """A valid boxed action should parse correctly and end on success."""
    sokoban_env.reset()
    observation, reward, done, info = sokoban_env.step(r"\boxed{RIGHT}")

    assert reward == 1.0
    assert done is True
    assert info["parsed_action"] == 4
    assert info["terminated"] is True
    assert info["truncated"] is False
    assert info["action_is_effective"] is True
    assert info["is_correct"] is True
    assert "Congratulations! You solved the Sokoban puzzle." in observation


def test_step_truncates_when_turn_cap_reached(sokoban_module: types.ModuleType) -> None:
    """Invalid/no-op actions should truncate once max_turns is reached."""
    env = sokoban_module.SokobanEnvAdapter(
        env_id="sokoban",
        env_kwargs={"dim_room": (4, 4), "num_boxes": 1, "max_turns": 1, "seed": 3},
    )
    try:
        env.reset()
        observation, reward, done, info = env.step("not-an-action")

        assert reward == 0.0
        assert done is True
        assert info["parsed_action"] == 0
        assert info["terminated"] is False
        assert info["truncated"] is True
        assert info["turn"] == 1
        assert "Time limit reached (1/1 steps)." in observation
    finally:
        env.close()


def test_step_after_done_raises_runtime_error(sokoban_env: Any) -> None:
    """Stepping after terminal state should require reset first."""
    sokoban_env.reset()
    sokoban_env.step(r"\boxed{right}")

    with pytest.raises(RuntimeError, match="Call reset\\(\\) before step\\(\\)"):
        sokoban_env.step("up")


def test_reset_retries_seed_for_shortest_path_filter(
    sokoban_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reset should retry seeds until shortest-path bounds are met."""
    sampled_lengths = [1, 3]

    def _fake_shortest_path(_cls, _room_state, _room_fixed):
        return sampled_lengths.pop(0)

    monkeypatch.setattr(
        sokoban_module.SokobanEnvAdapter,
        "_compute_shortest_path_length",
        classmethod(_fake_shortest_path),
    )

    env = sokoban_module.SokobanEnvAdapter(
        env_id="sokoban",
        env_kwargs={
            "dim_room": (4, 4),
            "num_boxes": 1,
            "max_turns": 3,
            "seed": 10,
            "shortest_path_min_length": 3,
            "shortest_path_max_length": 3,
        },
    )
    try:
        _, info = env.reset()
        assert info["shortest_path_length"] == 3
        assert env._seed == 11
    finally:
        env.close()


def test_init_rejects_invalid_shortest_path_bounds(
    sokoban_module: types.ModuleType,
) -> None:
    """Adapter should validate shortest-path bounds at construction time."""
    with pytest.raises(
        ValueError,
        match="shortest_path_min_length cannot be greater than shortest_path_max_length",
    ):
        sokoban_module.SokobanEnvAdapter(
            env_id="sokoban",
            env_kwargs={
                "shortest_path_min_length": 5,
                "shortest_path_max_length": 2,
            },
        )
