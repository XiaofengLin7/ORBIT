"""GEM environment adapter for rLLM placed outside vendor submodules.

This adapter keeps third-party code untouched while exposing GEM's Gym-style
interface through the rLLM `BaseEnv` contract used by the training stack.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from rllm.agents.agent import Action  # type: ignore
from rllm.environments.base.base_env import BaseEnv  # type: ignore


def _silence_nltk_downloads_globally() -> None:
    """Wrap nltk.download so its [nltk_data] log lines never reach stdio.

    Some GEM envs (Wordle, Hangman) call ``nltk.download("words")`` from
    their ``__init__`` without ``quiet=True``, which prints
    ``[nltk_data]`` lines even when the corpus is already cached. We
    can't edit third_party, so install a process-wide silent shim before
    importing ``gem``. The wrapper:
      * forces ``quiet=True`` (this is what actually suppresses output;
        nltk respects this flag and skips the "[nltk_data]" prints)
      * is idempotent (re-imports won't double-wrap)

    IMPORTANT: do NOT use ``contextlib.redirect_stdout`` here. It mutates
    the global ``sys.stdout`` and is *not* thread-safe. ``init_envs_and_agents``
    creates 256 envs concurrently in a 64-thread pool; concurrent
    ``redirect_stdout`` calls from per-env ``nltk.download`` invocations
    permanently corrupt ``sys.stdout`` (they save each other's StringIO
    buffers as "the original" and restore them in the wrong order). Every
    subsequent ``print()``/``colorful_print()`` then writes into an
    orphaned buffer no one reads, silently dropping all rollout-time
    trajectory output. ``quiet=True`` alone is sufficient — keep it that way.
    """
    try:
        import nltk  # type: ignore
    except ImportError:
        return
    if getattr(nltk.download, "_silenced", False):
        return

    _orig = nltk.download

    def _silent_download(*args, **kwargs):
        kwargs.setdefault("quiet", True)
        return _orig(*args, **kwargs)

    _silent_download._silenced = True  # idempotency marker
    nltk.download = _silent_download


_silence_nltk_downloads_globally()


try:
    import gem  # type: ignore
    import gem.envs  # type: ignore  # noqa: F401  # populate registries
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "The 'gem' package is required for GEMEnvAdapter. "
        "Install it via `pip install gem-llm` or ensure it is on PYTHONPATH."
    ) from exc

# Auto-register custom Minesweeper environment
import envs.register_custom_minesweeper  # noqa: F401


class GEMEnvAdapter(BaseEnv):
    """Adapter that wraps a GEM environment and normalizes the rLLM API.

    The adapter only touches the `done` signal (terminated or truncated) and
    otherwise passes observations, rewards, and info through unchanged.
    """

    def __init__(self, env_id: str, env_kwargs: Optional[Dict[str, Any]] = None):
        """Initialize the GEM environment adapter.

        Args:
            env_id: Registered GEM environment identifier.
            env_kwargs: Optional keyword arguments forwarded to ``gem.make``.
        """
        self.env_id = env_id
        self.env_kwargs = env_kwargs or {}
        self._env = gem.make(env_id, **self.env_kwargs)
        # Persist a fixed seed so repeated resets recreate the same task.
        self._seed = self.env_kwargs.get("seed")

    def reset(self, seed: int | None = None, task: dict | None = None) -> tuple[Any, dict]:
        """Reset the underlying GEM environment.

        Returns:
            observation: The initial observation returned by GEM.
            info: Auxiliary information from GEM.
        """
        # Prefer explicit seed when provided; otherwise reuse the persistent seed.
        reset_seed = seed if seed is not None else self._seed
        # Keep the seed for subsequent resets to enforce identical tasks.
        self._seed = reset_seed
        # Some GEM envs accept a seed kwarg; fall back silently when unsupported.
        try:
            observation, info = self._env.reset(seed=reset_seed)
        except TypeError:
            observation, info = self._env.reset(task=task)
        # Normalize observation with any prefix/suffix decoration.
        return info.get("prefix", "") + observation + info.get("suffix", ""), info

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        """Execute one step in the GEM environment.

        Args:
            action: Action to execute. If an rLLM ``Action`` is provided, its
                ``action`` payload is forwarded to GEM.

        Returns:
            observation: Observation after the step.
            reward: Reward returned by GEM, cast to float.
            done: ``True`` if terminated or truncated.
            info: Diagnostic information from GEM.
        """
        raw_action = action.action if isinstance(action, Action) else action
        observation, reward, terminated, truncated, info = self._env.step(raw_action)
        done = bool(terminated or truncated)
        # Preserve termination detail for downstream multi-episode logic.
        enriched_info = dict(info or {})
        enriched_info.update(
            {
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "raw_reward": float(reward),
                "max_turns": getattr(self._env, "max_turns", None),
            }
        )
        return (
            enriched_info.get("prefix", "") + observation + enriched_info.get("suffix", ""),
            float(reward),
            done,
            enriched_info,
        )

    def close(self) -> None:
        """Close the underlying GEM environment."""
        if hasattr(self._env, "close"):
            self._env.close()

    @staticmethod
    def from_dict(info: dict) -> "GEMEnvAdapter":
        """Factory helper to create the adapter from a dictionary.

        Args:
            info: Configuration dictionary containing ``env_id`` and optional
                ``env_kwargs``.

        Returns:
            An initialized ``GEMEnvAdapter`` instance.
        """
        env_id = info["env_id"]
        env_kwargs = info.get("env_kwargs", {})
        return GEMEnvAdapter(env_id=env_id, env_kwargs=env_kwargs)

