"""Observation-consistent oracle summarizer for the maze environment.

Generates a "mental map" string from data the agent has actually observed:
- Cells it has visited
- Cell types of those cells' immediate neighbors (the local observations
  the maze env emits each step)
- Chronological path per episode
- Current position and per-episode outcomes

Reads the ground-truth grid only as a lookup for cells the agent has
already been to or has seen as an immediate neighbor — no information
beyond the agent's observations is revealed.
"""

from __future__ import annotations

from typing import Any, List, Tuple


_DELTAS = (
    ("up", (-1, 0)),
    ("down", (1, 0)),
    ("left", (0, -1)),
    ("right", (0, 1)),
)


def _cell_type(grid, x: int, y: int) -> str:
    if x < 0 or x >= grid.shape[0] or y < 0 or y >= grid.shape[1]:
        return "wall"
    v = int(grid[x, y])
    if v == 1:
        return "wall"
    if v == -1:
        return "goal"
    return "path"


def build_maze_mental_map_summary(env: Any) -> str:
    """Return a textual mental map for a ``MultiEpisodeEnv`` wrapping a maze.

    The body is plain text; the engine's :func:`apply_summary` wraps it in
    ``<context_summary>...</context_summary>`` tags.
    """
    inner = env.inner_env
    grid = inner.map
    if grid is None:
        return "Maze not initialized."

    start = inner.init_position
    current = inner.current_position
    visited: set[Tuple[int, int]] = set(env._maze_visited_positions or set())
    paths: List[List[Tuple[int, int]]] = list(env._maze_episode_paths or [])
    successes: List[bool] = list(env._episode_successes)
    lengths: List[int] = list(env._episode_lengths)

    H, W = int(grid.shape[0]), int(grid.shape[1])

    # Per-cell observations: deterministic ordering by (x, y).
    observations: List[Tuple[Tuple[int, int], dict]] = []
    for cell in sorted(visited):
        x, y = cell
        nbrs = {name: _cell_type(grid, x + dx, y + dy) for name, (dx, dy) in _DELTAS}
        observations.append((cell, nbrs))

    # Goal: was it observed adjacent to any visited cell?
    goal_seen_at: Tuple[int, int] | None = None
    goal_position: Tuple[int, int] | None = None
    for (x, y), nbrs in observations:
        for name, (dx, dy) in _DELTAS:
            if nbrs[name] == "goal":
                goal_seen_at = (x, y)
                goal_position = (x + dx, y + dy)
                break
        if goal_position is not None:
            break

    # Frontier: cells observed as `path` adjacent to a visited cell, but
    # not yet visited themselves.
    frontier: set[Tuple[int, int]] = set()
    for (x, y), nbrs in observations:
        for name, (dx, dy) in _DELTAS:
            if nbrs[name] == "path":
                ncell = (x + dx, y + dy)
                if ncell not in visited:
                    frontier.add(ncell)

    lines: List[str] = []
    lines.append(
        f"Maze size: {H}x{W}. Start: {start}. Current position: {current}."
    )
    if successes:
        outcome_parts = []
        for i, s in enumerate(successes):
            tag = "success" if s else "fail"
            steps_part = f", {lengths[i]} steps" if i < len(lengths) else ""
            outcome_parts.append(f"ep{i + 1}: {tag}{steps_part}")
        lines.append(
            f"Episodes completed: {len(successes)} ({'; '.join(outcome_parts)})."
        )
    else:
        lines.append("Episodes completed: 0.")

    lines.append("")
    lines.append(f"Visited cells ({len(visited)} unique):")
    for (x, y), nbrs in observations:
        lines.append(
            f"  ({x},{y}): up={nbrs['up']}, down={nbrs['down']}, "
            f"left={nbrs['left']}, right={nbrs['right']}"
        )

    lines.append("")
    if goal_position is not None:
        lines.append(
            f"Goal observed: yes - at {goal_position} "
            f"(seen from {goal_seen_at})."
        )
    else:
        lines.append("Goal observed: no - not yet adjacent to any visited cell.")

    if frontier:
        ftxt = ", ".join(f"({x},{y})" for x, y in sorted(frontier))
        lines.append(f"Known frontier (reachable, unvisited): {ftxt}.")
    else:
        lines.append("Known frontier (reachable, unvisited): none.")

    lines.append("")
    lines.append("Your path so far:")
    any_path = False
    for ep_idx, path in enumerate(paths):
        if not path:
            continue
        any_path = True
        path_txt = " -> ".join(f"({x},{y})" for x, y in path)
        if ep_idx < len(successes):
            outcome = "GOAL" if successes[ep_idx] else "no goal"
            steps = lengths[ep_idx] if ep_idx < len(lengths) else max(0, len(path) - 1)
            lines.append(
                f"  Episode {ep_idx + 1}: {path_txt} [{outcome}, {steps} steps]"
            )
        else:
            lines.append(
                f"  Episode {ep_idx + 1}: {path_txt} "
                f"[in progress, {max(0, len(path) - 1)} steps]"
            )
    if not any_path:
        lines.append("  (no moves yet)")

    return "\n".join(lines)
