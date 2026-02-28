"""FrozenLake adapter for rLLM wrappers.

This module keeps vendor code under ``third_party`` unchanged while exposing
FrozenLake through the local ``BaseEnv`` interface expected by single-episode
and multi-episode wrappers.
"""

from __future__ import annotations

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

        self._env = self._build_env(
            size=self.size,
            p=self.p,
            is_slippery=self.is_slippery,
            max_turns=self.max_turns,
            seed=self._seed,
            desc=self.desc,
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
            }
        )
        return self._rules_prompt_with_obs(str(observation)), normalized_info

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
            }
        )
        return (
            self._step_prompt_with_obs(str(observation), terminated=bool(terminated), truncated=truncated),
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
    def _rules_prompt_with_obs(obs_text: str) -> str:
        """Build reset-time instruction prompt with an observation snapshot.

        Args:
            obs_text: Current textual grid observation.

        Returns:
            Instructional observation text for the agent.
        """
        return (
            "You are playing FrozenLake.\n"
            "Grid symbols: P (player), _ (safe), O (hole), G (goal).\n"
            "Valid actions: up, down, left, right.\n"
            "Output your next action in \\boxed{...}.\n"
            f"Current observation:\n{obs_text}"
        )

    @staticmethod
    def _step_prompt_with_obs(obs_text: str, terminated: bool, truncated: bool) -> str:
        """Build step-time observation text.

        Args:
            obs_text: Current textual grid observation.
            terminated: Whether the episode reached terminal state.
            truncated: Whether episode stopped due turn cap.

        Returns:
            Formatted observation string for the next policy turn.
        """
        if terminated and not truncated:
            return (
                f"Current observation:\n{obs_text}\n"
                "Episode finished. Reach the goal to succeed in each episode."
            )
        if truncated:
            return (
                f"Current observation:\n{obs_text}\n"
                "Episode stopped because max_turns was reached."
            )
        return (
            f"Current observation:\n{obs_text}\n"
            "Output your next action in \\boxed{up/down/left/right}."
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

        if seed is not None:
            config["seed"] = int(seed)

        if int(config["max_turns"]) <= 0:
            raise ValueError("max_turns must be a positive integer.")
        return config
