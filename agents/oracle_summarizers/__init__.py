"""Rule-based oracle summarizers for context self-summarization.

When ``rllm.agent.summarization.oracle.enable=true`` the execution engine
substitutes the LLM-generated summary with a deterministic, rule-based one
for environments that have a registered oracle. This module is the dispatch
point.
"""

from __future__ import annotations

from typing import Any, Callable, Optional


def get_oracle_summarizer(env: Any) -> Optional[Callable[[Any], str]]:
    """Return an oracle summarizer for ``env``, or ``None`` if unsupported.

    ``env`` is the multi-episode wrapper; we dispatch on the inner env
    type so callers don't need to know about wrappers.
    """
    inner = getattr(env, "inner_env", None)
    if inner is None:
        return None
    try:
        from envs.maze_env_adapter import MazeEnvAdapter
    except Exception:
        MazeEnvAdapter = None  # type: ignore[assignment]
    if MazeEnvAdapter is not None and isinstance(inner, MazeEnvAdapter):
        from agents.oracle_summarizers.maze import build_maze_mental_map_summary

        return build_maze_mental_map_summary
    return None
