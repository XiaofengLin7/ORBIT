"""Per-episode length penalty (overlong reward shaping).

Implements the piecewise-linear length penalty from
"Demystifying RL in Agentic Reasoning" (arXiv:2510.11701) Eq. 9
(originally from DAPO), applied at the episode level for ORBIT's
episodic-summarization training.

For each episode *e* with response-token count ``|y_e|``::

    r_length(|y_e|) =  0                                       if |y_e| <= L_max - L_cache
                     ((L_max - L_cache) - |y_e|) / L_cache     if L_max - L_cache < |y_e| <= L_max
                     -1                                        if |y_e| > L_max

The penalty is added on top of the existing ``env._episode_rewards[e]``::

    r_phi_e = r_outcome_e + r_length_e

Because the modified ``episode_rewards`` list is what flows into
:mod:`trainers.chunk_advantage` (both variant A and variant B), the
downstream chunk-credit-assignment math sees the shaped reward without
any code change to chunk_advantage / segment_expansion.

``|y_e|`` counts every token added to the trajectory's context during
episode *e* (action assistant tokens + env-message tokens + the closing
summary's instruction + completion tokens, when an episode-end summary
fires). It excludes the initial system prompt, matching the user's
intent of "the whole trajectory's tokens beside the initial prompt." The
engine stamps a per-step ``response_token_delta`` field on each step
dict; we sum it here.

In token-summarization mode the summarizer auto-manages context length,
so the caller should set ``skip_if_token_mode=True`` (the engine reads
this config flag) — this module itself doesn't know about modes.

Pure-Python; no torch / Ray / Hydra imports — unit-testable in isolation.
"""

from __future__ import annotations

from typing import Sequence


__all__ = [
    "compute_episode_length_penalty",
    "per_episode_response_tokens",
    "apply_length_penalty_to_episode_rewards",
]


# Type alias for a single step dict from `episode_steps`.
StepDict = dict


def compute_episode_length_penalty(
    y_len: int,
    l_max: int,
    l_cache: int,
) -> float:
    """Piecewise-linear length penalty for one episode (paper Eq. 9).

    Args:
        y_len: Number of response tokens in this episode (sum of
            ``response_token_delta`` across all steps belonging to this
            episode — actions and the closing summary).
        l_max: Hard cap. ``|y| > l_max`` returns -1.
        l_cache: Width of the linear-decay soft zone just below
            ``l_max``.

    Returns:
        Penalty in ``[-1.0, 0.0]``. Always 0.0 in the safe zone, -1.0
        in the cliff zone, linear in between.

    Edge cases:
        - ``l_cache <= 0``: degenerates to a step function (no soft
          zone) — 0 if ``y_len <= l_max``, else -1.
        - ``l_max <= 0``: any positive ``y_len`` is cliff (-1); zero
          ``y_len`` is safe (0).
        - ``y_len < 0``: treated as 0.
    """
    if y_len < 0:
        y_len = 0
    # Cliff zone first — handles l_max <= 0 cleanly.
    if y_len > l_max:
        return -1.0
    # Degenerate: no soft zone.
    if l_cache <= 0:
        return 0.0
    threshold = l_max - l_cache
    if y_len <= threshold:
        return 0.0
    # Soft zone: linear from 0 (at threshold) down to -1 (at l_max).
    return (threshold - y_len) / l_cache


def per_episode_response_tokens(
    episode_steps: Sequence[StepDict],
    n_episodes: int,
) -> list[int]:
    """Total ``response_token_delta`` per episode (actions + closing summary).

    Walks ``episode_steps`` in order, tracking the current episode index
    from action steps (which carry ``episode_index``). Summary steps
    inherit the most recently seen action step's index — i.e. a summary
    that fires after episode *e*'s terminal action is attributed to
    episode *e*. This matches the natural semantics: the summary
    closing out an episode is generated during that episode's processing
    and consumes its budget.

    Each step contributes ``int(step.get("response_token_delta", 0))``.
    If a step lacks the field (e.g. ``oracle`` summaries that don't
    record their tokens), it contributes zero.

    Args:
        episode_steps: The engine's per-step list (action + summary
            chunks intermixed).
        n_episodes: Target list length. Should match
            ``len(env._episode_rewards)``.

    Returns:
        List of int token totals, length exactly ``n_episodes``.
    """
    if n_episodes <= 0:
        return []
    totals = [0] * n_episodes
    cur_ep = 0
    for step in episode_steps:
        is_sum = bool(step.get("is_summarization", False))
        if not is_sum:
            # Action step — read its episode_index. Defaults preserve
            # cur_ep when the field is missing, which is correct for
            # malformed step dicts.
            cur_ep = int(step.get("episode_index", cur_ep))
        delta = int(step.get("response_token_delta", 0))
        if 0 <= cur_ep < n_episodes:
            totals[cur_ep] += delta
        # else: stray index — silently drop. Defensive only.
    return totals


def apply_length_penalty_to_episode_rewards(
    ep_rewards: Sequence[float],
    episode_steps: Sequence[StepDict],
    *,
    l_max: int,
    l_cache: int,
) -> tuple[list[float], list[int], list[float]]:
    """Compute per-episode penalty and shape ``ep_rewards`` additively.

    Does not mutate the input list. The caller replaces its
    ``ep_rewards`` with the returned ``shaped_rewards``.

    Args:
        ep_rewards: Per-episode reward list (from
            ``env._episode_rewards``). Length E.
        episode_steps: Per-step list from the engine.
        l_max: ``L_max`` (paper Eq. 9).
        l_cache: ``L_cache`` (paper Eq. 9).

    Returns:
        Tuple of ``(shaped_ep_rewards, per_episode_response_tokens,
        per_episode_length_penalty)``. All three lists have length
        ``len(ep_rewards)``.
    """
    n_eps = len(ep_rewards)
    if n_eps == 0:
        return [], [], []
    tokens = per_episode_response_tokens(episode_steps, n_eps)
    penalties = [
        compute_episode_length_penalty(t, l_max, l_cache) for t in tokens
    ]
    shaped = [float(r) + float(p) for r, p in zip(ep_rewards, penalties)]
    return shaped, tokens, penalties
