"""WebShop adapter for rLLM multi-episode wrappers.

Wraps GEM's WebshopEnv through the ``BaseEnv`` interface expected by
``MultiEpisodeEnv``.  Uses the ``session`` parameter for deterministic task
selection (bypasses the random session-ID bug in webshop.py:80-81) and
concatenates prefix/suffix to form ORBIT-style observations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from rllm.agents.agent import Action  # type: ignore
from rllm.environments.base.base_env import BaseEnv  # type: ignore

try:
    import gem  # type: ignore
    import gem.envs  # type: ignore  # noqa: F401  # populate registries
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'gem' package is required for WebShopEnvAdapter. "
        "Install it via `pip install gem-llm` or ensure it is on PYTHONPATH."
    ) from exc


class WebShopEnvAdapter(BaseEnv):
    """Adapter wrapping GEM's WebshopEnv with deterministic task selection.

    Instead of relying on ``random.seed()`` for reproducibility, this adapter
    passes ``session=seed % num_goals`` directly to ``WebshopEnv.reset()``,
    which triggers deterministic goal lookup via ``get_goal_by_idx(session_int)``.

    This also bypasses the double ``random.choices`` bug in WebshopEnv that
    produces 1-character session IDs.
    """

    def __init__(
        self,
        env_id: str = "webshop",
        env_kwargs: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> None:
        env_kwargs = dict(env_kwargs or {})

        # Extract WebShop-specific parameters
        split = env_kwargs.pop("split", "test")
        observation_mode = env_kwargs.pop("observation_mode", "text")
        max_turns = env_kwargs.pop("max_turns", 15)
        error_tolerance = env_kwargs.pop("error_tolerance", 15)
        format_error_reward = env_kwargs.pop("format_error_reward", -0.1)
        enable_purchase_feedback = env_kwargs.pop("enable_purchase_feedback", False)

        self.env_id = env_id
        self._split = split
        self._observation_mode = observation_mode
        self._max_turns = max_turns
        self._enable_purchase_feedback = bool(enable_purchase_feedback)

        # Construct GEM env_id: e.g. "webshop:test-text"
        gem_env_id = f"webshop:{split}-{observation_mode}"
        self._env = gem.make(
            gem_env_id,
            max_turns=max_turns,
            error_tolerance=error_tolerance,
            format_error_reward=format_error_reward,
        )

        # Number of goals in this split, for deterministic seed → goal mapping.
        # cum_weights has length num_goals + 1 (starts with a leading 0).
        self._num_goals = len(self._env.server.cum_weights) - 1

    # ------------------------------------------------------------------
    # BaseEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        task: Optional[dict] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Reset and select a deterministic task via goal index.

        Args:
            seed: Used to compute ``session_int = seed % num_goals`` for
                deterministic goal selection.
            task: Optional task dict (may contain ``seed`` key).

        Returns:
            observation: Formatted string (prefix + obs + suffix).
            info: Metadata dictionary.
        """
        # Resolve seed from arguments or task dict
        if seed is None and task is not None:
            seed = task.get("seed")

        session_int = seed % self._num_goals if seed is not None else None
        obs, info = self._env.reset(session=session_int)

        formatted_obs = info.get("prefix", "") + obs + info.get("suffix", "")

        normalized_info: Dict[str, Any] = {
            "env_id": self.env_id,
            "turn": 0,
            "max_turns": self._max_turns,
            "terminated": False,
            "truncated": False,
            "raw_reward": 0.0,
        }
        return formatted_obs, normalized_info

    def step(self, action: Any) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute one step in the WebShop environment.

        Args:
            action: Action string or ``Action`` object.  Expected to contain
                ``\\boxed{search[...]}`` or ``\\boxed{click[...]}``.

        Returns:
            observation: Formatted string with explicit success/failure feedback.
            reward: 1.0 only on perfect match (raw == 1.0), else 0.0.
                The actual WebShop score is preserved in info["raw_reward"].
            done: True when terminated or truncated.
            info: Metadata dictionary.
        """
        raw_action = action.action if isinstance(action, Action) else action
        obs, raw_reward, terminated, truncated, info = self._env.step(raw_action)

        done = bool(terminated or truncated)
        formatted_obs = self._format_step_obs(
            obs, info, raw_reward, terminated, truncated,
        )

        # WebShop-specific: only perfect match (raw_reward == 1.0) counts as
        # success for MultiEpisodeEnv._is_episode_success (checks reward > 0).
        # The continuous score is preserved in info["raw_reward"] for avg_reward.
        reward = 1.0 if raw_reward == 1.0 else 0.0

        normalized_info: Dict[str, Any] = {
            "env_id": self.env_id,
            "turn": getattr(self._env, "turn_count", 0),
            "max_turns": self._max_turns,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "raw_reward": float(raw_reward),
            "parsed_action": info.get("parsed_action", str(raw_action)),
        }
        return formatted_obs, reward, done, normalized_info

    # ------------------------------------------------------------------
    # Observation formatting
    # ------------------------------------------------------------------

    def _format_step_obs(
        self,
        obs: str,
        info: Dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> str:
        """Format step observation with explicit success/failure feedback.

        On terminal steps the raw WebShop done-page observation (noisy MTurk
        boilerplate) is replaced with a clear outcome message.  Non-terminal
        steps keep the standard prefix + obs + suffix format.
        """
        if terminated and not truncated and reward == 1.0:
            # Perfect purchase
            return (
                f"Your score: {reward:.2f} / 1.00.\n"
                f"Congratulations! You successfully completed the task."
            )
        if terminated and not truncated and reward > 0:
            # Partial match — show score only, no success/failure label
            msg = f"Your score: {reward:.2f} / 1.00."
            if self._enable_purchase_feedback:
                msg += "\n" + self._build_option_feedback()
            return msg
        if terminated and not truncated:
            # Zero reward (total mismatch or error tolerance exceeded)
            msg = "You failed."
            if self._enable_purchase_feedback:
                msg += "\n" + self._build_option_feedback()
            return msg
        if truncated:
            return (
                f"Episode stopped because max_turns ({self._max_turns}) was reached. "
                f"You did not make a purchase."
            )
        # Normal (non-terminal) step — keep standard format
        return info.get("prefix", "") + obs + info.get("suffix", "")

    def _build_option_feedback(self) -> str:
        """Build concise diagnostic showing required vs. selected options."""
        try:
            session_id = self._env.session
            session = self._env.server.user_sessions[session_id]
            goal_options = session["goal"].get("goal_options", {})
            purchased_options = session.get("options", {})
        except (AttributeError, KeyError):
            return ""

        if not goal_options:
            return ""

        if isinstance(goal_options, dict):
            required_parts = [f"{k}={v}" for k, v in goal_options.items()]
        else:
            required_parts = [str(o) for o in goal_options]

        if purchased_options:
            selected_parts = [f"{k}={v}" for k, v in purchased_options.items()]
        else:
            selected_parts = ["(none)"]

        lines = [
            f"Required options: {', '.join(required_parts)}",
            f"Your selections: {', '.join(selected_parts)}",
        ]

        if isinstance(goal_options, dict):
            missed = [k for k in goal_options if k not in purchased_options]
            if missed:
                lines.append(
                    f"You forgot to select: {', '.join(missed)}. "
                    f"Click the option on the product page before clicking buy."
                )

        return "\n".join(lines)

    def close(self) -> None:
        """No-op — WebshopEnv has no resources to release."""
        pass

    @staticmethod
    def from_dict(info: dict) -> "WebShopEnvAdapter":
        """Factory to create the adapter from a configuration dictionary.

        Extracts passthrough keys (``max_turns``, ``seed``, ``split``,
        ``observation_mode``) into ``env_kwargs``, mirroring the pattern
        used by ``ALFWorldEnvAdapter.from_dict()``.
        """
        env_kwargs = dict(info.get("env_kwargs", {}) or {})
        passthrough_keys = (
            "max_turns",
            "seed",
            "split",
            "observation_mode",
            "error_tolerance",
            "format_error_reward",
            "enable_purchase_feedback",
        )
        for key in passthrough_keys:
            if key in info and key not in env_kwargs:
                env_kwargs[key] = info[key]
        return WebShopEnvAdapter(
            env_id=info.get("env_id", "webshop"),
            env_kwargs=env_kwargs,
        )

    @staticmethod
    def is_multithread_safe() -> bool:
        """WebshopEnv is not thread-safe (mutable browser/session state)."""
        return False
