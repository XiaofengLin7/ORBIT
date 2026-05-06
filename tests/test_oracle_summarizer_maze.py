"""Unit tests for the maze oracle summarizer."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from agents.oracle_summarizers import get_oracle_summarizer
from agents.oracle_summarizers.maze import build_maze_mental_map_summary


# Cell encoding used by MazeEnvAdapter:
#   0 = path, 1 = wall, -1 = goal
P, W, G = 0, 1, -1


def _make_env(grid, start, current, visited, episode_paths,
              episode_successes=None, episode_lengths=None):
    """Build a minimal stand-in for MultiEpisodeEnv(MazeEnvAdapter).

    The oracle reads only attributes; SimpleNamespace is enough.
    """
    inner = SimpleNamespace(
        map=np.array(grid, dtype=int),
        init_position=tuple(start),
        current_position=tuple(current),
    )
    return SimpleNamespace(
        inner_env=inner,
        _maze_visited_positions=set(visited),
        _maze_episode_paths=[list(p) for p in episode_paths],
        _episode_successes=list(episode_successes or []),
        _episode_lengths=list(episode_lengths or []),
    )


# A 4x4 maze:
#   walls border, goal at (2,3); a corridor (1,1) -> (1,2) -> (2,2) -> (2,3)
GRID_4X4 = [
    [W, W, W, W],
    [W, P, P, W],
    [W, P, P, G],
    [W, W, W, W],
]


def test_summary_contains_size_start_current_and_visited():
    env = _make_env(
        grid=GRID_4X4,
        start=(1, 1),
        current=(2, 2),
        visited={(1, 1), (1, 2), (2, 2)},
        episode_paths=[[(1, 1), (1, 2), (2, 2)]],
    )
    text = build_maze_mental_map_summary(env)

    assert "Maze size: 4x4" in text
    assert "Start: (1, 1)" in text
    assert "Current position: (2, 2)" in text
    # Each visited cell appears in the per-cell observation block.
    assert "(1,1):" in text
    assert "(1,2):" in text
    assert "(2,2):" in text


def test_summary_neighbor_cell_types_match_grid():
    env = _make_env(
        grid=GRID_4X4,
        start=(1, 1),
        current=(1, 1),
        visited={(1, 1)},
        episode_paths=[[(1, 1)]],
    )
    text = build_maze_mental_map_summary(env)

    # At (1,1): up=(0,1)=wall, down=(2,1)=path, left=(1,0)=wall, right=(1,2)=path
    assert "(1,1): up=wall, down=path, left=wall, right=path" in text


def test_goal_observed_when_adjacent_to_visited_cell():
    env = _make_env(
        grid=GRID_4X4,
        start=(1, 1),
        current=(2, 2),
        visited={(1, 1), (1, 2), (2, 2)},
        episode_paths=[[(1, 1), (1, 2), (2, 2)]],
    )
    text = build_maze_mental_map_summary(env)
    # (2,2) sees goal at (2,3) on its right.
    assert "Goal observed: yes" in text
    assert "(2, 3)" in text


def test_goal_not_observed_when_not_adjacent():
    # Visit only (1,1) — goal at (2,3) is not adjacent.
    env = _make_env(
        grid=GRID_4X4,
        start=(1, 1),
        current=(1, 1),
        visited={(1, 1)},
        episode_paths=[[(1, 1)]],
    )
    text = build_maze_mental_map_summary(env)
    assert "Goal observed: no" in text


def test_frontier_lists_unvisited_path_neighbors():
    env = _make_env(
        grid=GRID_4X4,
        start=(1, 1),
        current=(1, 1),
        visited={(1, 1)},
        episode_paths=[[(1, 1)]],
    )
    text = build_maze_mental_map_summary(env)
    # Frontier from (1,1) = cells with type=path adjacent and not visited:
    # (2,1)=path, (1,2)=path
    assert "(1,2)" in text and "(2,1)" in text
    assert "Known frontier" in text


def test_path_per_episode_with_outcomes():
    env = _make_env(
        grid=GRID_4X4,
        start=(1, 1),
        current=(2, 2),
        visited={(1, 1), (1, 2), (2, 2), (2, 1)},
        episode_paths=[
            [(1, 1), (2, 1), (1, 1)],          # episode 1: failed
            [(1, 1), (1, 2), (2, 2)],          # episode 2: in progress
        ],
        episode_successes=[False],
        episode_lengths=[2],
    )
    text = build_maze_mental_map_summary(env)
    assert "Episode 1: (1,1) -> (2,1) -> (1,1) [no goal, 2 steps]" in text
    assert "Episode 2: (1,1) -> (1,2) -> (2,2) [in progress, 2 steps]" in text


def test_observation_consistency_does_not_reveal_unseen_cells():
    """The oracle must not name cells the agent has never been adjacent to."""
    # Visit only (1,1). The far cells (2,2), (2,3), etc. should not appear
    # in the per-cell observation block. They may not appear in frontier
    # either since (1,1) is not adjacent to them.
    env = _make_env(
        grid=GRID_4X4,
        start=(1, 1),
        current=(1, 1),
        visited={(1, 1)},
        episode_paths=[[(1, 1)]],
    )
    text = build_maze_mental_map_summary(env)
    # Distant cells (2,2) and (2,3) should not appear. Use exact tuple
    # forms to avoid colliding with substring matches inside other lines.
    assert "(2,2):" not in text
    assert "(2,3)" not in text


def test_no_moves_yet_does_not_crash():
    env = _make_env(
        grid=GRID_4X4,
        start=(1, 1),
        current=(1, 1),
        visited={(1, 1)},
        episode_paths=[[(1, 1)]],
    )
    text = build_maze_mental_map_summary(env)
    assert "Episodes completed: 0" in text


def test_dispatch_returns_callable_for_maze_inner_env():
    """Cover the registry dispatch path used by the engine."""
    # Stub class that imports under MazeEnvAdapter's qualified name.
    from envs.maze_env_adapter import MazeEnvAdapter

    # Instantiate without running reset (we only need isinstance).
    adapter = MazeEnvAdapter.__new__(MazeEnvAdapter)
    env = SimpleNamespace(inner_env=adapter)
    fn = get_oracle_summarizer(env)
    assert fn is not None
    # Compare via identity: the registry should hand back the maze builder.
    assert fn is build_maze_mental_map_summary


def test_dispatch_returns_none_for_non_maze_env():
    env = SimpleNamespace(inner_env=SimpleNamespace())  # not a maze
    assert get_oracle_summarizer(env) is None
