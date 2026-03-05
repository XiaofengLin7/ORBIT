"""FrozenLake adapter for rLLM wrappers.

This module keeps vendor code under ``third_party`` unchanged while exposing
FrozenLake through the local ``BaseEnv`` interface expected by single-episode
and multi-episode wrappers.
"""

from __future__ import annotations

from collections import deque
import re
from typing import Any, Dict, Mapping, Optional

from rllm.agents.agent import Action  # type: ignore
from rllm.environments.base.base_env import BaseEnv  # type: ignore
from rllm.environments.frozenlake.frozenlake import FrozenLakeEnv  # type: ignore
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv as GymFrozenLakeEnv  # type: ignore


BOXED_PATTERN = re.compile(r"\\boxed\{([^}]+)\}", re.IGNORECASE)


class FrozenLakeEnvAdapter(BaseEnv):
    """Adapter exposing vendor FrozenLake via the local ``BaseEnv`` contract."""

    _DIRECTION_TO_ACTION = {
        "left": 1,
        "down": 2,
        "right": 3,
        "up": 4,
    }
    _VALID_ACTIONS = {0, 1, 2, 3, 4}
    _MAX_GENERATION_ATTEMPTS = 1000

    def __init__(
        self,
        env_id: Optional[str] = None,
        env_kwargs: Optional[dict] = None,
        size: int = 8,
        p: float = 0.8,
        is_slippery: bool = False,
        max_turns: int = 5,
        seed: Optional[int] = None,
        desc: Optional[list[str]] = None,
        shortest_path_min_length: Optional[int] = None,
        shortest_path_max_length: Optional[int] = None,
        **_: Any,
    ) -> None:
        """Initialize a FrozenLake adapter.

        Args:
            env_id: Environment identifier for compatibility with dataset tasks.
            env_kwargs: Optional nested configuration dictionary.
            size: Grid size used when no ``desc`` is provided.
            p: Frozen-tile probability for random map generation.
            is_slippery: Whether movement is stochastic.
            max_turns: Maximum turns per episode for adapter-level truncation.
            seed: Optional persistent reset seed.
            desc: Optional explicit map description.
            shortest_path_min_length: Optional lower bound on map shortest-path
                length used for difficulty filtering during reset.
            shortest_path_max_length: Optional upper bound on map shortest-path
                length used for difficulty filtering during reset.
            **_: Additional ignored kwargs for compatibility with wrappers.
        """
        merged_kwargs = dict(env_kwargs or {})
        if "max_turns" not in merged_kwargs and "max_steps" in merged_kwargs:
            merged_kwargs["max_turns"] = merged_kwargs["max_steps"]

        self.env_id = str(env_id or merged_kwargs.get("env_id", "frozenlake"))
        self.size = int(merged_kwargs.get("size", size))
        self.p = float(merged_kwargs.get("p", p))
        self.is_slippery = bool(merged_kwargs.get("is_slippery", is_slippery))
        self.max_turns = int(merged_kwargs.get("max_turns", max_turns))
        self._seed = merged_kwargs.get("seed", seed)
        self.desc = merged_kwargs.get("desc", desc)
        self.shortest_path_min_length = self._coerce_optional_positive_int(
            merged_kwargs.get("shortest_path_min_length", shortest_path_min_length),
            field_name="shortest_path_min_length",
        )
        self.shortest_path_max_length = self._coerce_optional_positive_int(
            merged_kwargs.get("shortest_path_max_length", shortest_path_max_length),
            field_name="shortest_path_max_length",
        )
        self._validate_shortest_path_bounds(
            min_length=self.shortest_path_min_length,
            max_length=self.shortest_path_max_length,
        )
        self.shortest_path_length: Optional[int] = None

        if self.max_turns <= 0:
            raise ValueError("max_turns must be a positive integer.")

        self.turn = 0
        self._done = False
        self._env = self._build_env(
            size=self.size,
            p=self.p,
            is_slippery=self.is_slippery,
            max_turns=self.max_turns,
            seed=self._seed,
            desc=self.desc,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        task: Optional[dict] = None,
    ) -> tuple[str, dict]:
        """Reset the environment with optional per-task overrides.

        Args:
            seed: Optional explicit reset seed.
            task: Optional task dictionary with per-instance overrides.

        Returns:
            A tuple of formatted observation text and normalized info metadata.
        """
        config = self._resolve_reset_config(seed=seed, task=task)
        self.size = int(config["size"])
        self.p = float(config["p"])
        self.is_slippery = bool(config["is_slippery"])
        self.max_turns = int(config["max_turns"])
        self._seed = int(config["seed"])
        self.desc = config.get("desc")
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
            size=self.size,
            p=self.p,
            is_slippery=self.is_slippery,
            max_turns=self.max_turns,
            base_seed=self._seed,
            desc=self.desc,
            shortest_path_min_length=self.shortest_path_min_length,
            shortest_path_max_length=self.shortest_path_max_length,
        )

        # Vendor reset rebuilds a random map and drops explicit desc.
        # Reset the underlying Gym env directly to preserve the map we built.
        GymFrozenLakeEnv.reset(self._env, seed=self._seed)
        observation = self._env.render(mode="tiny_rgb_array")
        info: Dict[str, Any] = {}
        self.turn = 0
        self._done = False

        normalized_info = dict(info or {})
        normalized_info.update(
            {
                "env_id": self.env_id,
                "turn": 0,
                "max_turns": self.max_turns,
                "terminated": False,
                "truncated": False,
                "raw_reward": 0.0,
                "shortest_path_length": self.shortest_path_length,
            }
        )
        return self._rules_prompt_with_obs(
            str(observation),
            max_turns=self.max_turns,
        ), normalized_info

    def step(self, action: Any) -> tuple[str, float, bool, dict]:
        """Execute one environment step.

        Args:
            action: Raw action payload. Supports ``Action`` wrappers, boxed
                strings, direction names, and integer-coded actions.

        Returns:
            A tuple of formatted observation, reward, done, and normalized info.
        """
        if self._done:
            raise RuntimeError("Environment is done. Call reset() before step().")

        parsed_action = self._normalize_action(action)
        observation, reward, terminated, info = self._env.step(parsed_action)
        self.turn += 1

        truncated = bool(not terminated and self.turn >= self.max_turns)
        done = bool(terminated or truncated)
        self._done = done

        normalized_info = dict(info or {})
        normalized_info.update(
            {
                "env_id": self.env_id,
                "turn": self.turn,
                "max_turns": self.max_turns,
                "terminated": bool(terminated),
                "truncated": truncated,
                "raw_reward": float(reward),
                "parsed_action": parsed_action,
                "is_correct": bool(terminated and float(reward) > 0),
                "shortest_path_length": self.shortest_path_length,
            }
        )
        return (
            self._step_prompt_with_obs(
                str(observation),
                terminated=bool(terminated),
                truncated=truncated,
                reward=float(reward),
                turn=self.turn,
                max_turns=self.max_turns,
            ),
            float(reward),
            done,
            normalized_info,
        )

    def close(self) -> None:
        """Close the wrapped FrozenLake environment."""
        if hasattr(self._env, "close"):
            self._env.close()

    @staticmethod
    def from_dict(info: dict) -> "FrozenLakeEnvAdapter":
        """Create an adapter from a configuration dictionary.

        Args:
            info: Dictionary containing top-level keys and/or ``env_kwargs``.

        Returns:
            Initialized ``FrozenLakeEnvAdapter`` instance.
        """
        env_kwargs = dict(info.get("env_kwargs", {}) or {})
        passthrough_keys = (
            "size",
            "p",
            "is_slippery",
            "seed",
            "max_turns",
            "max_steps",
            "desc",
            "shortest_path_min_length",
            "shortest_path_max_length",
        )
        for key in passthrough_keys:
            if key in info and key not in env_kwargs:
                env_kwargs[key] = info[key]
        if "max_turns" not in env_kwargs and "max_steps" in env_kwargs:
            env_kwargs["max_turns"] = env_kwargs["max_steps"]

        return FrozenLakeEnvAdapter(
            env_id=info.get("env_id", env_kwargs.get("env_id", "frozenlake")),
            env_kwargs=env_kwargs,
        )

    @staticmethod
    def is_multithread_safe() -> bool:
        """Return whether this adapter is safe for multi-threaded usage."""
        return True

    @classmethod
    def _normalize_action(cls, action: Any) -> int:
        """Normalize raw action into FrozenLake's custom integer action space.

        Args:
            action: Raw action payload from the agent.

        Returns:
            Integer action in ``{0, 1, 2, 3, 4}``, where ``0`` is no-op.
        """
        if isinstance(action, Action):
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

        token = raw_text.lower().replace("`", " ").strip()
        token = re.sub(r"[^\w\s]", " ", token).strip()
        token_parts = token.split()
        token = token_parts[0] if token_parts else ""

        if token.isdigit():
            candidate = int(token)
            return candidate if candidate in cls._VALID_ACTIONS else 0
        if token in cls._DIRECTION_TO_ACTION:
            return cls._DIRECTION_TO_ACTION[token]
        return 0

    @staticmethod
    def _rules_prompt_with_obs(obs_text: str, max_turns: int) -> str:
        """Build reset-time instruction prompt with an observation snapshot.

        Args:
            obs_text: Current textual grid observation.
            max_turns: Maximum number of allowed steps in this episode.

        Returns:
            Instructional observation text for the agent.
        """
        return (
            "You are playing FrozenLake.\n"
            "Grid symbols: P (player), _ (safe), O (hole), G (goal).\n"
            "Valid actions: up, down, left, right.\n"
            f"You can take at most {max_turns} steps in this episode.\n"
            "Output your next action in \\boxed{...}.\n"
            f"Current observation:\n{obs_text}"
        )

    @classmethod
    def _step_prompt_with_obs(
        cls,
        obs_text: str,
        terminated: bool,
        truncated: bool,
        reward: float,
        turn: int,
        max_turns: int,
    ) -> str:
        """Build step-time observation text.

        Args:
            obs_text: Current textual grid observation.
            terminated: Whether the episode reached terminal state.
            truncated: Whether episode stopped due turn cap.
            reward: Scalar reward returned by the environment.
            turn: Current 1-based turn index in the episode.
            max_turns: Maximum number of turns in the episode.

        Returns:
            Formatted observation string for the next policy turn.
        """
        if terminated and reward > 0:
            outcome_feedback = "Congratulations! You successfully reached the goal."
        elif terminated:
            if "X" in obs_text:
                outcome_feedback = (
                    "You failed the task because you stepped into a hole (`X`)."
                )
            else:
                outcome_feedback = (
                    "You failed the task because the episode ended before reaching the goal."
                )
        elif truncated:
            outcome_feedback = (
                "You failed the task because you reached the maximum number of turns "
                f"({turn}/{max_turns}) before reaching the goal."
            )
        else:
            outcome_feedback = (
                "Output your next action in \\boxed{up/down/left/right}."
            )

        return (
            f"Current observation:\n{obs_text}\n"
            f"{outcome_feedback}"
        )

    @staticmethod
    def _build_env(
        *,
        size: int,
        p: float,
        is_slippery: bool,
        max_turns: int,
        seed: Optional[int],
        desc: Optional[list[str]],
    ) -> FrozenLakeEnv:
        """Instantiate vendor FrozenLake with normalized kwargs.

        Args:
            size: Grid size.
            p: Frozen-tile probability.
            is_slippery: Whether movement is stochastic.
            max_turns: Per-episode turn cap.
            seed: Random seed for map generation.
            desc: Optional explicit map description.

        Returns:
            Vendor ``FrozenLakeEnv`` instance.
        """
        init_kwargs: Dict[str, Any] = {
            "size": int(size),
            "p": float(p),
            "is_slippery": bool(is_slippery),
            "max_steps": int(max_turns),
            "seed": int(seed) if seed is not None else 42,
        }
        if desc is not None:
            init_kwargs["desc"] = desc
        return FrozenLakeEnv(**init_kwargs)

    def _resolve_reset_config(
        self,
        seed: Optional[int],
        task: Optional[dict],
    ) -> Dict[str, Any]:
        """Resolve reset configuration from constructor defaults and task data.

        Args:
            seed: Optional explicit seed argument for reset.
            task: Optional task dictionary with per-instance overrides.

        Returns:
            Effective environment configuration for this reset.
        """
        config: Dict[str, Any] = {
            "size": self.size,
            "p": self.p,
            "is_slippery": self.is_slippery,
            "max_turns": self.max_turns,
            "seed": self._seed if self._seed is not None else 42,
            "desc": self.desc,
            "shortest_path_min_length": self.shortest_path_min_length,
            "shortest_path_max_length": self.shortest_path_max_length,
        }

        if isinstance(task, dict):
            if "size" in task and task["size"] is not None:
                config["size"] = int(task["size"])
            if "p" in task and task["p"] is not None:
                config["p"] = float(task["p"])
            if "is_slippery" in task and task["is_slippery"] is not None:
                config["is_slippery"] = bool(task["is_slippery"])
            if "max_turns" in task and task["max_turns"] is not None:
                config["max_turns"] = int(task["max_turns"])
            elif "max_steps" in task and task["max_steps"] is not None:
                config["max_turns"] = int(task["max_steps"])
            if "seed" in task and task["seed"] is not None:
                config["seed"] = int(task["seed"])
            if "desc" in task and task["desc"] is not None:
                config["desc"] = task["desc"]
            if (
                "shortest_path_min_length" in task
                and task["shortest_path_min_length"] is not None
            ):
                config["shortest_path_min_length"] = int(task["shortest_path_min_length"])
            if (
                "shortest_path_max_length" in task
                and task["shortest_path_max_length"] is not None
            ):
                config["shortest_path_max_length"] = int(task["shortest_path_max_length"])

        if seed is not None:
            config["seed"] = int(seed)

        if int(config["max_turns"]) <= 0:
            raise ValueError("max_turns must be a positive integer.")
        return config

    @classmethod
    def _build_env_for_reset(
        cls,
        *,
        size: int,
        p: float,
        is_slippery: bool,
        max_turns: int,
        base_seed: int,
        desc: Optional[list[str]],
        shortest_path_min_length: Optional[int],
        shortest_path_max_length: Optional[int],
    ) -> tuple[FrozenLakeEnv, int, Optional[int]]:
        """Build an env, optionally retrying seeds until difficulty bounds match."""
        require_filter = (
            shortest_path_min_length is not None or shortest_path_max_length is not None
        )

        if not require_filter:
            env = cls._build_env(
                size=size,
                p=p,
                is_slippery=is_slippery,
                max_turns=max_turns,
                seed=base_seed,
                desc=desc,
            )
            shortest_path_length = cls._compute_shortest_path_length(getattr(env, "desc", None))
            return env, base_seed, shortest_path_length

        if desc is not None:
            env = cls._build_env(
                size=size,
                p=p,
                is_slippery=is_slippery,
                max_turns=max_turns,
                seed=base_seed,
                desc=desc,
            )
            shortest_path_length = cls._compute_shortest_path_length(getattr(env, "desc", None))
            if cls._is_shortest_path_in_range(
                shortest_path_length,
                min_length=shortest_path_min_length,
                max_length=shortest_path_max_length,
            ):
                return env, base_seed, shortest_path_length
            raise ValueError(
                "Provided FrozenLake desc does not satisfy shortest-path range: "
                f"[{shortest_path_min_length}, {shortest_path_max_length}] "
                f"(actual: {shortest_path_length})."
            )

        for attempt in range(cls._MAX_GENERATION_ATTEMPTS):
            candidate_seed = int(base_seed + attempt)
            env = cls._build_env(
                size=size,
                p=p,
                is_slippery=is_slippery,
                max_turns=max_turns,
                seed=candidate_seed,
                desc=None,
            )
            shortest_path_length = cls._compute_shortest_path_length(getattr(env, "desc", None))
            if cls._is_shortest_path_in_range(
                shortest_path_length,
                min_length=shortest_path_min_length,
                max_length=shortest_path_max_length,
            ):
                return env, candidate_seed, shortest_path_length
            if hasattr(env, "close"):
                env.close()

        raise ValueError(
            "Failed to sample FrozenLake map within shortest-path range "
            f"[{shortest_path_min_length}, {shortest_path_max_length}] after "
            f"{cls._MAX_GENERATION_ATTEMPTS} attempts."
        )

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
        """Validate shortest-path bounds consistency."""
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
        """Return whether a shortest-path length satisfies optional bounds."""
        if length is None:
            return False
        if min_length is not None and length < min_length:
            return False
        if max_length is not None and length > max_length:
            return False
        return True

    @classmethod
    def _compute_shortest_path_length(cls, desc: Any) -> Optional[int]:
        """Compute shortest safe-path length from S to G on a FrozenLake map."""
        grid = cls._normalize_desc_grid(desc)
        if not grid:
            return None

        rows = len(grid)
        cols = len(grid[0])
        start: Optional[tuple[int, int]] = None
        goal: Optional[tuple[int, int]] = None

        for r in range(rows):
            for c in range(cols):
                cell = grid[r][c]
                if cell == "S":
                    start = (r, c)
                elif cell == "G":
                    goal = (r, c)

        if start is None or goal is None:
            return None

        queue: deque[tuple[int, int, int]] = deque([(start[0], start[1], 0)])
        visited = {start}
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            row, col, distance = queue.popleft()
            if (row, col) == goal:
                return distance
            for dr, dc in deltas:
                nr, nc = row + dr, col + dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue
                if (nr, nc) in visited:
                    continue
                if grid[nr][nc] == "H":
                    continue
                visited.add((nr, nc))
                queue.append((nr, nc, distance + 1))
        return None

    @classmethod
    def _normalize_desc_grid(cls, desc: Any) -> list[list[str]]:
        """Normalize various desc representations into a 2D grid of chars."""
        if desc is None:
            return []

        if hasattr(desc, "tolist"):
            desc = desc.tolist()

        if not isinstance(desc, list) or not desc:
            return []

        if all(isinstance(row, str) for row in desc):
            return [list(row) for row in desc]

        grid: list[list[str]] = []
        for row in desc:
            if not isinstance(row, (list, tuple)):
                return []
            normalized_row: list[str] = []
            for cell in row:
                if isinstance(cell, bytes):
                    normalized_row.append(cell.decode("utf-8"))
                else:
                    normalized_row.append(str(cell))
            grid.append(normalized_row)

        if not grid:
            return []
        row_width = len(grid[0])
        if row_width == 0 or any(len(row) != row_width for row in grid):
            return []
        return grid
