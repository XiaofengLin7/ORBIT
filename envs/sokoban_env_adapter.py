"""Sokoban adapter for rLLM wrappers.

This module adapts the ``gym_sokoban`` Sokoban environment to the local
``BaseEnv`` interface expected by single-episode and multi-episode wrappers.
Each call to :meth:`step` processes exactly **one** directional action.

Install dependency (once per environment)::

    pip install gym_sokoban
"""

from __future__ import annotations

from collections import deque
import random
import re
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from rllm.agents.agent import Action  # type: ignore
from rllm.environments.base.base_env import BaseEnv  # type: ignore

try:
    from gym_sokoban.envs.sokoban_env import SokobanEnv as GymSokobanEnv  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "gym_sokoban is required.  Install it with:  pip install gym_sokoban"
    ) from exc


BOXED_PATTERN = re.compile(r"\\boxed\{([^}]+)\}", re.IGNORECASE)

# ──────────────────────────────────────────────────────────────────────────────
# Text rendering
# ──────────────────────────────────────────────────────────────────────────────
# room_state cell values (gym_sokoban standard):
#   0 = wall         1 = empty floor     2 = box target
#   3 = box on tgt   4 = box (off tgt)   5 = player
# Synthetic value 6 = player standing on a target (room_state==5, room_fixed==2)
_GRID_LOOKUP: Dict[int, str] = {
    0: "#",   # wall
    1: "_",   # empty floor
    2: "O",   # target
    3: "√",   # box on target
    4: "X",   # box (not on target)
    5: "P",   # player
    6: "S",   # player on target
}

_GRID_LEGEND = (
    "# (wall), _ (empty), O (target), X (box), √ (box on target), "
    "P (player), S (player on target)"
)


def _render_grid(room_state: np.ndarray, room_fixed: np.ndarray) -> str:
    """Render the board as a multi-line ASCII grid."""
    # Mark player-on-target cells as synthetic value 6.
    room = np.where((room_state == 5) & (room_fixed == 2), 6, room_state)
    return "\n".join(
        "".join(_GRID_LOOKUP.get(int(cell), "?") for cell in row)
        for row in room.tolist()
    )


def _render_coord(room_state: np.ndarray, room_fixed: np.ndarray) -> str:
    """Render entity positions as zero-indexed coordinate lists."""
    rows, cols = room_state.shape

    def positions(mask: np.ndarray) -> str:
        pts = np.argwhere(mask)
        return ", ".join(f"({r}, {c})" for r, c in pts) if len(pts) else "none"

    on_target = positions(room_state == 3)
    boxes = positions(room_state == 4)
    targets = positions(room_fixed == 2)
    player_mask = (room_state == 5) | ((room_state == 5) & (room_fixed == 2))
    player = positions(room_state == 5)

    return (
        f"Board size: {rows} rows x {cols} cols (zero-indexed).\n"
        f"Targets: {targets}\n"
        f"Boxes: {boxes}\n"
        f"Boxes on target: {on_target}\n"
        f"Player: {player}"
    )


def _render(
    room_state: np.ndarray,
    room_fixed: np.ndarray,
    observation_format: str,
) -> str:
    if observation_format == "grid":
        return _render_grid(room_state, room_fixed)
    if observation_format == "coord":
        return _render_coord(room_state, room_fixed)
    if observation_format == "grid_coord":
        return (
            "Coordinates:\n"
            + _render_coord(room_state, room_fixed)
            + "\nGrid Map:\n"
            + _render_grid(room_state, room_fixed)
        )
    raise ValueError(f"Unknown observation_format: {observation_format!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Adapter
# ──────────────────────────────────────────────────────────────────────────────

class SokobanEnvAdapter(BaseEnv):
    """Adapter exposing ``gym_sokoban`` via the local ``BaseEnv`` contract.

    Each :meth:`step` call processes exactly one directional action.  The
    agent is expected to output one action per turn inside ``\\boxed{...}``.

    Actions (gym_sokoban integer codes):
        1 = up, 2 = down, 3 = left, 4 = right, 0 = no-op (invalid fallback)
    """

    _DIRECTION_TO_ACTION: Dict[str, int] = {
        "up": 1, "u": 1,
        "down": 2, "d": 2,
        "left": 3, "l": 3,
        "right": 4, "r": 4,
    }
    _VALID_ACTIONS: frozenset = frozenset({0, 1, 2, 3, 4})
    _MAX_GENERATION_ATTEMPTS = 1000
    _MAX_SOLVER_NODES = 200000

    def __init__(
        self,
        env_id: Optional[str] = None,
        env_kwargs: Optional[dict] = None,
        dim_room: Tuple[int, int] = (6, 6),
        num_boxes: int = 1,
        max_turns: int = 100,
        search_depth: int = 300,
        observation_format: str = "grid",
        seed: Optional[int] = None,
        shortest_path_min_length: Optional[int] = None,
        shortest_path_max_length: Optional[int] = None,
        **_: Any,
    ) -> None:
        """Initialise the Sokoban adapter.

        Args:
            env_id: Identifier string for task-level bookkeeping.
            env_kwargs: Optional nested config dict (overrides positional args).
            dim_room: ``(rows, cols)`` size of the Sokoban grid.
            num_boxes: Number of boxes (and targets) in each puzzle.
            max_turns: Maximum steps per episode.
            search_depth: BFS depth used during procedural room generation.
            observation_format: ``"grid"``, ``"coord"``, or ``"grid_coord"``.
            seed: Optional persistent seed used when no per-reset seed is given.
            shortest_path_min_length: Optional lower bound for puzzle shortest
                solution length during reset-time sampling.
            shortest_path_max_length: Optional upper bound for puzzle shortest
                solution length during reset-time sampling.
            **_: Ignored extra kwargs (compatibility with wrapper factories).
        """
        merged = dict(env_kwargs or {})
        if "max_turns" not in merged and "max_steps" in merged:
            merged["max_turns"] = merged["max_steps"]

        self.env_id = str(env_id or merged.get("env_id", "sokoban"))
        self.dim_room: Tuple[int, int] = tuple(merged.get("dim_room", dim_room))  # type: ignore[assignment]
        self.num_boxes: int = int(merged.get("num_boxes", num_boxes))
        self.max_turns: int = int(merged.get("max_turns", max_turns))
        self.search_depth: int = int(merged.get("search_depth", search_depth))
        self.observation_format: str = str(merged.get("observation_format", observation_format))
        self._seed: Optional[int] = merged.get("seed", seed)
        self.shortest_path_min_length = self._coerce_optional_positive_int(
            merged.get("shortest_path_min_length", shortest_path_min_length),
            field_name="shortest_path_min_length",
        )
        self.shortest_path_max_length = self._coerce_optional_positive_int(
            merged.get("shortest_path_max_length", shortest_path_max_length),
            field_name="shortest_path_max_length",
        )
        self._validate_shortest_path_bounds(
            min_length=self.shortest_path_min_length,
            max_length=self.shortest_path_max_length,
        )
        self.shortest_path_length: Optional[int] = None

        if self.max_turns <= 0:
            raise ValueError("max_turns must be a positive integer.")

        self.turn: int = 0
        self._done: bool = False
        self._env: GymSokobanEnv = self._build_env(
            dim_room=self.dim_room,
            num_boxes=self.num_boxes,
            max_steps=self.max_turns,
        )

    # ── BaseEnv interface ─────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        task: Optional[dict] = None,
    ) -> Tuple[str, dict]:
        """Reset and return ``(observation_text, info)``."""
        config = self._resolve_reset_config(seed=seed, task=task)
        self.dim_room = config["dim_room"]
        self.num_boxes = config["num_boxes"]
        self.max_turns = config["max_turns"]
        self.observation_format = config["observation_format"]
        self._seed = config["seed"]
        self.shortest_path_min_length = self._coerce_optional_positive_int(
            config.get("shortest_path_min_length"),
            field_name="shortest_path_min_length",
        )
        self.shortest_path_max_length = self._coerce_optional_positive_int(
            config.get("shortest_path_max_length"),
            field_name="shortest_path_max_length",
        )
        self._validate_shortest_path_bounds(
            min_length=self.shortest_path_min_length,
            max_length=self.shortest_path_max_length,
        )

        self._env, self._seed, self.shortest_path_length = self._build_env_for_reset(
            dim_room=self.dim_room,
            num_boxes=self.num_boxes,
            max_steps=self.max_turns,
            base_seed=self._seed,
            shortest_path_min_length=self.shortest_path_min_length,
            shortest_path_max_length=self.shortest_path_max_length,
        )

        self.turn = 0
        self._done = False

        obs_text = _render(self._env.room_state, self._env.room_fixed, self.observation_format)
        info: Dict[str, Any] = {
            "env_id": self.env_id,
            "turn": 0,
            "max_turns": self.max_turns,
            "terminated": False,
            "truncated": False,
            "raw_reward": 0.0,
            "shortest_path_length": self.shortest_path_length,
        }
        return self._rules_prompt_with_obs(obs_text, max_turns=self.max_turns), info

    def step(self, action: Any) -> Tuple[str, float, bool, dict]:
        """Execute a single action. Accepts text, ``\\boxed{...}``, or int 1-4.

        Invalid inputs fall back to action 0 (no-op in gym_sokoban).
        """
        if self._done:
            raise RuntimeError("Environment is done. Call reset() before step().")

        parsed_action = self._normalize_action(action)
        _, reward, inner_done, inner_info = self._env.step(parsed_action)
        self.turn += 1

        success = bool(self._env.boxes_on_target == self._env.num_boxes)
        terminated = success
        truncated = bool(
            (inner_done and not terminated)
            or (not inner_done and self.turn >= self.max_turns)
        )
        done = terminated or truncated
        self._done = done

        obs_text = _render(self._env.room_state, self._env.room_fixed, self.observation_format)

        info: Dict[str, Any] = {
            "env_id": self.env_id,
            "turn": self.turn,
            "max_turns": self.max_turns,
            "terminated": terminated,
            "truncated": truncated,
            "raw_reward": float(reward),
            "parsed_action": parsed_action,
            "action_is_effective": bool(inner_info.get("action.moved_player", False)),
            "is_correct": terminated,
            "shortest_path_length": self.shortest_path_length,
        }
        return (
            self._step_prompt_with_obs(
                obs_text,
                terminated=terminated,
                truncated=truncated,
                reward=float(reward),
                turn=self.turn,
                max_turns=self.max_turns,
            ),
            float(reward),
            done,
            info,
        )

    def close(self) -> None:
        """Close the underlying gym_sokoban environment."""
        if hasattr(self._env, "close"):
            self._env.close()

    # ── Factory helpers ───────────────────────────────────────────────────────

    @staticmethod
    def from_dict(info: dict) -> "SokobanEnvAdapter":
        """Create an adapter from a configuration dictionary."""
        env_kwargs = dict(info.get("env_kwargs", {}) or {})
        for key in ("dim_room", "num_boxes", "max_turns", "max_steps",
                    "search_depth", "observation_format", "seed",
                    "shortest_path_min_length", "shortest_path_max_length"):
            if key in info and key not in env_kwargs:
                env_kwargs[key] = info[key]
        if "max_turns" not in env_kwargs and "max_steps" in env_kwargs:
            env_kwargs["max_turns"] = env_kwargs["max_steps"]
        return SokobanEnvAdapter(
            env_id=info.get("env_id", env_kwargs.get("env_id", "sokoban")),
            env_kwargs=env_kwargs,
        )

    @staticmethod
    def is_multithread_safe() -> bool:
        return True

    # ── Action parsing ────────────────────────────────────────────────────────

    @classmethod
    def _normalize_action(cls, action: Any) -> int:
        """Parse raw agent output into an integer action (1–4, or 0 for no-op)."""
        if isinstance(action, Action) and hasattr(action, "action") and not isinstance(action, (str, int, float)):
            action = action.action
        elif isinstance(action, Mapping):
            action = action.get("action", "")

        if isinstance(action, (int, float)):
            candidate = int(action)
            return candidate if candidate in cls._VALID_ACTIONS else 0

        raw_text = str(action).strip()
        boxed_matches = list(BOXED_PATTERN.finditer(raw_text))
        if boxed_matches:
            raw_text = boxed_matches[-1].group(1).strip()

        token = re.sub(r"[^\w\s]", " ", raw_text.lower().replace("`", " ")).strip()
        parts = token.split()
        token = parts[0] if parts else ""

        if token.isdigit():
            candidate = int(token)
            return candidate if candidate in cls._VALID_ACTIONS else 0
        return cls._DIRECTION_TO_ACTION.get(token, 0)

    # ── Prompt builders ───────────────────────────────────────────────────────

    @staticmethod
    def _rules_prompt_with_obs(obs_text: str, max_turns: int) -> str:
        return (
            "You are solving a Sokoban puzzle.\n"
            "Push all boxes (X) onto target locations (O) to win.\n"
            "Move into a box to push it; you cannot pull boxes.\n"
            f"Grid symbols: {_GRID_LEGEND}.\n"
            "Valid actions: up, down, left, right.\n"
            f"You can take at most {max_turns} steps in this episode.\n"
            "Output your next action in \\boxed{up/down/left/right}.\n"
            f"Current observation:\n{obs_text}"
        )

    @staticmethod
    def _step_prompt_with_obs(
        obs_text: str,
        terminated: bool,
        truncated: bool,
        reward: float,
        turn: int,
        max_turns: int,
    ) -> str:
        if terminated:
            outcome = "Congratulations! You solved the Sokoban puzzle."
        elif truncated:
            outcome = (
                f"Time limit reached ({turn}/{max_turns} steps). "
                "You did not solve the puzzle in time."
            )
        else:
            outcome = "Output your next action in \\boxed{up/down/left/right}."
        return f"Current observation:\n{obs_text}\n{outcome}"

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_env(
        *,
        dim_room: Tuple[int, int],
        num_boxes: int,
        max_steps: int,
    ) -> GymSokobanEnv:
        return GymSokobanEnv(
            dim_room=tuple(dim_room),
            num_boxes=int(num_boxes),
            max_steps=int(max_steps),
        )

    @classmethod
    def _build_env_for_reset(
        cls,
        *,
        dim_room: Tuple[int, int],
        num_boxes: int,
        max_steps: int,
        base_seed: Optional[int],
        shortest_path_min_length: Optional[int],
        shortest_path_max_length: Optional[int],
    ) -> tuple[GymSokobanEnv, Optional[int], Optional[int]]:
        """Build env and optionally retry with different seeds for difficulty."""
        require_filter = (
            shortest_path_min_length is not None or shortest_path_max_length is not None
        )

        if not require_filter:
            env = cls._build_env(
                dim_room=dim_room,
                num_boxes=num_boxes,
                max_steps=max_steps,
            )
            cls._set_seed(env, base_seed)
            env.reset()
            shortest_path_length = cls._compute_shortest_path_length(
                env.room_state, env.room_fixed
            )
            return env, base_seed, shortest_path_length

        start_seed = int(base_seed) if base_seed is not None else 42
        for attempt in range(cls._MAX_GENERATION_ATTEMPTS):
            candidate_seed = int(start_seed + attempt)
            env = cls._build_env(
                dim_room=dim_room,
                num_boxes=num_boxes,
                max_steps=max_steps,
            )
            cls._set_seed(env, candidate_seed)
            env.reset()
            shortest_path_length = cls._compute_shortest_path_length(
                env.room_state, env.room_fixed
            )
            if cls._is_shortest_path_in_range(
                shortest_path_length,
                min_length=shortest_path_min_length,
                max_length=shortest_path_max_length,
            ):
                return env, candidate_seed, shortest_path_length
            if hasattr(env, "close"):
                env.close()

        raise ValueError(
            "Failed to sample Sokoban puzzle within shortest-path range "
            f"[{shortest_path_min_length}, {shortest_path_max_length}] after "
            f"{cls._MAX_GENERATION_ATTEMPTS} attempts."
        )

    @staticmethod
    def _set_seed(env: GymSokobanEnv, seed: Optional[int]) -> None:
        """Seed gym_sokoban and numpy/python RNGs for deterministic generation."""
        if seed is None:
            return
        env.seed(int(seed))
        random.seed(int(seed))
        np.random.seed(int(seed))

    @staticmethod
    def _coerce_optional_positive_int(
        value: Any,
        *,
        field_name: str,
    ) -> Optional[int]:
        """Convert optional input to a positive integer."""
        if value is None:
            return None
        try:
            int_value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be an integer or None.") from exc
        if int_value <= 0:
            raise ValueError(f"{field_name} must be a positive integer when provided.")
        return int_value

    @staticmethod
    def _validate_shortest_path_bounds(
        *,
        min_length: Optional[int],
        max_length: Optional[int],
    ) -> None:
        """Validate shortest-path range bounds."""
        if min_length is not None and max_length is not None and min_length > max_length:
            raise ValueError(
                "shortest_path_min_length cannot be greater than shortest_path_max_length."
            )

    @staticmethod
    def _is_shortest_path_in_range(
        length: Optional[int],
        *,
        min_length: Optional[int],
        max_length: Optional[int],
    ) -> bool:
        """Check whether shortest-path length satisfies optional bounds."""
        if length is None:
            return False
        if min_length is not None and length < min_length:
            return False
        if max_length is not None and length > max_length:
            return False
        return True

    @classmethod
    def _compute_shortest_path_length(
        cls,
        room_state: np.ndarray,
        room_fixed: np.ndarray,
    ) -> Optional[int]:
        """Compute shortest Sokoban solution length with BFS over state space."""
        if room_state.ndim != 2 or room_state.shape != room_fixed.shape:
            return None

        rows, cols = room_state.shape
        walls = room_fixed == 0
        targets = {
            int(r * cols + c)
            for r, c in np.argwhere(room_fixed == 2)
        }
        player_positions = np.argwhere(room_state == 5)
        if len(player_positions) != 1:
            return None
        player_start = int(player_positions[0][0] * cols + player_positions[0][1])

        box_positions = np.argwhere((room_state == 3) | (room_state == 4))
        boxes_start = tuple(
            sorted(int(r * cols + c) for r, c in box_positions)
        )
        if not boxes_start:
            return 0
        if len(boxes_start) > len(targets):
            return None
        if all(box in targets for box in boxes_start):
            return 0

        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        queue: deque[tuple[int, tuple[int, ...], int]] = deque(
            [(player_start, boxes_start, 0)]
        )
        visited = {(player_start, boxes_start)}
        nodes_expanded = 0

        while queue:
            player_idx, boxes_tuple, distance = queue.popleft()
            nodes_expanded += 1
            if nodes_expanded > cls._MAX_SOLVER_NODES:
                return None

            player_row, player_col = divmod(player_idx, cols)
            boxes_set = set(boxes_tuple)

            for dr, dc in deltas:
                next_row, next_col = player_row + dr, player_col + dc
                if next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols:
                    continue
                if walls[next_row, next_col]:
                    continue

                next_player_idx = int(next_row * cols + next_col)
                if next_player_idx in boxes_set:
                    push_row, push_col = next_row + dr, next_col + dc
                    if (
                        push_row < 0
                        or push_row >= rows
                        or push_col < 0
                        or push_col >= cols
                    ):
                        continue
                    if walls[push_row, push_col]:
                        continue
                    pushed_box_idx = int(push_row * cols + push_col)
                    if pushed_box_idx in boxes_set:
                        continue
                    new_boxes = tuple(
                        sorted(
                            pushed_box_idx if box == next_player_idx else box
                            for box in boxes_tuple
                        )
                    )
                else:
                    new_boxes = boxes_tuple

                next_state = (next_player_idx, new_boxes)
                if next_state in visited:
                    continue
                if all(box in targets for box in new_boxes):
                    return distance + 1
                visited.add(next_state)
                queue.append((next_player_idx, new_boxes, distance + 1))

        return None

    def _resolve_reset_config(
        self,
        seed: Optional[int],
        task: Optional[dict],
    ) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "dim_room": self.dim_room,
            "num_boxes": self.num_boxes,
            "max_turns": self.max_turns,
            "observation_format": self.observation_format,
            "seed": self._seed,
            "shortest_path_min_length": self.shortest_path_min_length,
            "shortest_path_max_length": self.shortest_path_max_length,
        }
        if isinstance(task, dict):
            for key in ("num_boxes",):
                if task.get(key) is not None:
                    config[key] = int(task[key])
            if task.get("dim_room") is not None:
                config["dim_room"] = tuple(task["dim_room"])
            if task.get("max_turns") is not None:
                config["max_turns"] = int(task["max_turns"])
            elif task.get("max_steps") is not None:
                config["max_turns"] = int(task["max_steps"])
            if task.get("observation_format") is not None:
                config["observation_format"] = str(task["observation_format"])
            if task.get("seed") is not None:
                config["seed"] = int(task["seed"])
            if task.get("shortest_path_min_length") is not None:
                config["shortest_path_min_length"] = int(task["shortest_path_min_length"])
            if task.get("shortest_path_max_length") is not None:
                config["shortest_path_max_length"] = int(task["shortest_path_max_length"])
        if seed is not None:
            config["seed"] = int(seed)
        if int(config["max_turns"]) <= 0:
            raise ValueError("max_turns must be a positive integer.")
        return config
