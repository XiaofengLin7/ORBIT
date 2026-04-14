"""ScienceWorld adapter for ORBIT multi-episode framework.

Wraps the ScienceWorld text-based science lab through the ``BaseEnv``
interface expected by ``MultiEpisodeEnv``.  Follows the ORBIT prompt style:
observations include action instructions and the model is expected to respond
with ``\\boxed{action}``.

ScienceWorld does NOT emit explicit success/failure messages, so the adapter
adds them along with goal-progress feedback on episode completion.
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

from rllm.agents.agent import Action  # type: ignore
from rllm.environments.base.base_env import BaseEnv  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOXED_PATTERN = re.compile(r"\\boxed\{([^}]+)\}", re.IGNORECASE)

ACTION_TYPES_HELP = (
    "Available action types: open/close OBJ, activate/deactivate OBJ, "
    "connect OBJ to OBJ, disconnect OBJ, use OBJ on OBJ, look around, "
    "look at OBJ, look in OBJ, read OBJ, move OBJ to OBJ, pick up OBJ, "
    "put down OBJ, pour OBJ into OBJ, dunk OBJ into OBJ, mix OBJ, "
    "go to LOC, focus on OBJ, wait, inventory"
)

BOXED_INSTRUCTION = "Output your action inside \\boxed{}."

# Maximum score returned by ScienceWorld (used for normalization).
MAX_SCORE = 100


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class SciWorldEnvAdapter(BaseEnv):
    """Adapter exposing ScienceWorld via the ``BaseEnv`` contract for ORBIT.

    Each instance creates one ``ScienceWorldEnv`` (backed by a JVM via py4j).
    On ``reset``, the adapter loads a task+variation determined by the config's
    ``task_name`` and the seed.  Subsequent ``reset`` calls with the same seed
    replay the same variation (required for multi-episode).
    """

    def __init__(
        self,
        env_id: str = "sciworld",
        env_kwargs: Optional[Dict[str, Any]] = None,
        max_turns: int = 20,
        seed: Optional[int] = None,
        task_name: Optional[str] = None,
        split: str = "test",
        **_: Any,
    ) -> None:
        super().__init__()

        merged = dict(env_kwargs or {})
        self.env_id = str(merged.get("env_id", env_id))
        self.max_turns = int(merged.get("max_turns", max_turns))
        self._seed = merged.get("seed", seed)
        self.split = str(merged.get("split", split))

        # Resolve task_name — from explicit param, env_kwargs, or env_id prefix
        self.task_name: Optional[str] = (
            merged.get("task_name", task_name)
        )
        if self.task_name is None:
            self.task_name = self._parse_task_name(self.env_id)

        # Create ScienceWorld JVM environment
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*camel case.*")
            from scienceworld import ScienceWorldEnv
            self._env = ScienceWorldEnv("")

        # Cache variation lists per task (populated lazily on first load)
        self._variation_cache: Dict[str, List[int]] = {}

        # Runtime state
        self.turn: int = 0
        self._done: bool = False
        self._score: int = 0
        self._task_desc: str = ""
        self._loaded_task: Optional[str] = None
        self._loaded_variation: Optional[int] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        task: Optional[dict] = None,
    ) -> Tuple[str, dict]:
        config = self._resolve_reset_config(seed=seed, task=task)
        self._seed = int(config["seed"])
        self.max_turns = int(config["max_turns"])
        task_name = config["task_name"]

        # Map seed → variation index
        variations = self._get_variations(task_name, self.split)
        variation_idx = variations[self._seed % len(variations)]

        # Load and reset
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*camel case.*")
            self._env.load(task_name, variation_idx)
            obs, info = self._env.reset()
            self._task_desc = self._env.get_task_description()

        self._loaded_task = task_name
        self._loaded_variation = variation_idx
        self.turn = 0
        self._done = False
        self._score = 0

        normalized_info = {
            "env_id": self.env_id,
            "turn": 0,
            "max_turns": self.max_turns,
            "terminated": False,
            "truncated": False,
            "raw_reward": 0.0,
            "task_name": task_name,
            "variation_idx": variation_idx,
            "score": 0,
        }
        return self._format_init_obs(obs), normalized_info

    def step(self, action: Any) -> Tuple[str, float, bool, dict]:
        if self._done:
            raise RuntimeError("Environment is done. Call reset() before step().")

        parsed = self._parse_action(action)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*camel case.*")
            obs, reward_int, done, info = self._env.step(parsed)

        self.turn += 1
        self._score = int(info.get("score", 0))

        # SciWorld marks done when score == 100 or a terminal condition is met.
        # We treat score == MAX_SCORE as true success.
        success = self._score >= MAX_SCORE
        terminated = done or success
        truncated = not terminated and self.turn >= self.max_turns
        done_flag = terminated or truncated
        self._done = done_flag

        # Normalized reward: 0-1 continuous score
        normalized_reward = self._score / MAX_SCORE

        # Get goal progress on episode end for feedback
        goal_progress = ""
        if done_flag:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*camel case.*")
                    goal_progress = self._env.get_goal_progress()
            except Exception:
                pass

        formatted_obs = self._format_step_obs(
            obs, success, terminated, truncated, goal_progress
        )

        normalized_info = {
            "env_id": self.env_id,
            "turn": self.turn,
            "max_turns": self.max_turns,
            "terminated": bool(terminated),
            "truncated": truncated,
            "raw_reward": normalized_reward,
            "is_correct": success,
            "parsed_action": parsed,
            "task_name": self._loaded_task,
            "variation_idx": self._loaded_variation,
            "score": self._score,
        }
        return formatted_obs, normalized_reward, done_flag, normalized_info

    def close(self) -> None:
        if hasattr(self, "_env") and self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass

    @staticmethod
    def from_dict(info: dict) -> "SciWorldEnvAdapter":
        env_kwargs = dict(info.get("env_kwargs", {}) or {})
        passthrough_keys = ("max_turns", "seed", "task_name", "split")
        for key in passthrough_keys:
            if key in info and key not in env_kwargs:
                env_kwargs[key] = info[key]
        return SciWorldEnvAdapter(
            env_id=info.get("env_id", "sciworld"),
            env_kwargs=env_kwargs,
        )

    @staticmethod
    def is_multithread_safe() -> bool:
        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_task_name(env_id: str) -> Optional[str]:
        """Extract task name from env_id like ``sciworld:test-conductivity``."""
        if ":" in env_id:
            return env_id.split(":", 1)[1]
        return None

    def _get_variations(self, task_name: str, split: str) -> List[int]:
        """Return cached list of variation indices for the given split."""
        cache_key = f"{task_name}:{split}"
        if cache_key not in self._variation_cache:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*camel case.*")
                self._env.load(task_name, 0)
                if split == "train":
                    variations = self._env.get_variations_train()
                elif split == "dev":
                    variations = self._env.get_variations_dev()
                else:
                    variations = self._env.get_variations_test()
            if not variations:
                raise ValueError(
                    f"No variations found for task={task_name}, split={split}"
                )
            self._variation_cache[cache_key] = list(variations)
        return self._variation_cache[cache_key]

    def _resolve_reset_config(
        self,
        seed: Optional[int],
        task: Optional[dict],
    ) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "seed": self._seed if self._seed is not None else 0,
            "max_turns": self.max_turns,
            "task_name": self.task_name,
        }
        if isinstance(task, dict):
            if "seed" in task and task["seed"] is not None:
                config["seed"] = int(task["seed"])
            if "max_turns" in task and task["max_turns"] is not None:
                config["max_turns"] = int(task["max_turns"])
            elif "max_turns_per_episode" in task and task["max_turns_per_episode"] is not None:
                config["max_turns"] = int(task["max_turns_per_episode"])
            if "task_name" in task and task["task_name"] is not None:
                config["task_name"] = task["task_name"]
            if "split" in task and task["split"] is not None:
                self.split = str(task["split"])
        if seed is not None:
            config["seed"] = int(seed)
        if config["task_name"] is None:
            raise ValueError(
                "task_name must be provided via env_id (e.g. 'sciworld:boil'), "
                "env_kwargs, or task dict"
            )
        return config

    # ------------------------------------------------------------------
    # Action parsing
    # ------------------------------------------------------------------

    @classmethod
    def _parse_action(cls, action: Any) -> str:
        """Extract action text from ``\\boxed{...}`` wrapper."""
        if isinstance(action, Action):
            action = action.action
        raw = str(action).strip()
        matches = list(BOXED_PATTERN.finditer(raw))
        if matches:
            return matches[-1].group(1).strip()
        return raw.strip()

    # ------------------------------------------------------------------
    # Observation formatting
    # ------------------------------------------------------------------

    def _format_init_obs(self, raw_obs: str) -> str:
        return (
            f"You are in a science lab. Your task is to perform a science experiment.\n\n"
            f"Task: {self._task_desc}\n\n"
            f"{raw_obs}\n\n"
            f"{ACTION_TYPES_HELP}\n\n"
            f"{BOXED_INSTRUCTION}"
        )

    def _format_step_obs(
        self,
        raw_obs: str,
        success: bool,
        terminated: bool,
        truncated: bool,
        goal_progress: str,
    ) -> str:
        if success:
            return (
                f"{raw_obs}\n"
                f"Congratulations! You completed the task successfully. "
                f"Score: {self._score}/{MAX_SCORE}."
            )
        if truncated:
            parts = [
                f"{raw_obs}\n"
                f"Episode stopped because the maximum number of turns "
                f"({self.max_turns}) was reached. Score: {self._score}/{MAX_SCORE}."
            ]
            if goal_progress:
                parts.append(f"\n{self._format_goal_progress(goal_progress)}")
            return "\n".join(parts)
        if terminated and not success:
            parts = [
                f"{raw_obs}\n"
                f"Episode finished. Score: {self._score}/{MAX_SCORE}."
            ]
            if goal_progress:
                parts.append(f"\n{self._format_goal_progress(goal_progress)}")
            return "\n".join(parts)
        # Normal step
        return f"{raw_obs}\n\n{BOXED_INSTRUCTION}"

    @staticmethod
    def _format_goal_progress(raw: str) -> str:
        """Convert SciWorld's tabular goal progress into natural language."""
        completed = []
        not_completed = []

        for match in re.finditer(r"(true|false)\t\s*\S+\t(.+)", raw):
            done = match.group(1) == "true"
            description = match.group(2).strip()
            if done:
                completed.append(description)
            else:
                not_completed.append(description)

        lines = ["Goal progress:"]
        if completed:
            lines.append("  Completed: " + "; ".join(completed) + ".")
        if not_completed:
            lines.append("  NOT completed: " + "; ".join(not_completed) + ".")
        if not completed and not not_completed:
            lines.append("  No subgoals completed.")
        return "\n".join(lines)
