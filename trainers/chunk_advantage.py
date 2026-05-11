"""Chunk-level discounted returns for the IPA-inspired training method.

This module implements the post-training reward-shaping recipe from
arXiv:2512.24873 §3.2.4.2–§3.2.4.3 (Chunked MDP + chunk-level discounted
return). It is a *sibling* method to GRPO — when the trainer's
``rllm.advantage_method.name`` is set to ``"chunk_discounted_topr"``, the
trainer bypasses GRPO advantage normalization entirely and uses the
per-chunk scalars produced here.

A "chunk" here corresponds to one of our ``episode_steps`` entries: one
assistant turn (action chunk) or one summarization turn. Each chunk's
reward signal is

    G_k = γ^Δ · R

where ``R`` is the relevant episode (or trajectory) reward and ``Δ`` is
the number of chunks separating chunk *k* from the relevant terminal
chunk. All tokens within chunk *k*'s response_mask=1 range share the
scalar weight ``G_k``.

Two reward variants are supported:

* ``"terminal"`` (variant A): a single trajectory-level reward
  ``R = sum(episode_rewards)`` is broadcast across every trainable chunk
  with ``Δ`` counted from the trajectory's last trainable chunk. The
  discount does *not* reset at episode boundaries.

* ``"per_episode"`` (variant B): each episode *e* contributes its own
  ``R_e``. Trainable chunks are partitioned into per-episode buckets and
  ``Δ`` is counted within each bucket. Trainable summarization chunks
  triggered by an ``"episode_end"`` boundary head the *next* episode's
  bucket — they're optimized for the next episode's outcome, since they
  are what shapes that episode's start state.

The TOPR positive/negative tag is computed per-chunk as ``G_k > 0``
(paper Eq. 5). With γ ≤ 1 and non-negative episode rewards this is
equivalent to "the relevant episode/trajectory was a success."

All public functions are pure-Python, take dict-of-lists trajectory
objects (the ``trajectories`` payload built by
:meth:`SummarizingAgentExecutionEngine.run_agent_trajectory_async` and
cached on the trainer), and return flat per-trainable-chunk lists in
trajectory-then-chunk order. They do not import torch.
"""

from __future__ import annotations

import logging
from typing import Iterable

logger = logging.getLogger(__name__)


# Sentinel returned per chunk: (G_k as float, is_positive as bool).
ChunkOutput = tuple[float, bool]


def _segment_step_ranges(
    n_steps: int,
    summarization_boundaries: list[int],
) -> list[tuple[int, int]]:
    """Return ``[(start, end), …]`` step ranges per segment.

    Mirrors the split logic used by
    :meth:`SummarizingAgentExecutionEngine.assemble_segments`: segments
    are ``[0..b0]``, ``[b0+1..b1]``, …, ``[bk+1..n_steps]``. The
    summarization step is the LAST step of its segment.
    """
    if not summarization_boundaries:
        return [(0, n_steps)]
    split_points = [0] + [b + 1 for b in summarization_boundaries] + [n_steps]
    ranges: list[tuple[int, int]] = []
    for i in range(len(split_points) - 1):
        start, end = split_points[i], split_points[i + 1]
        if end > start:
            ranges.append((start, end))
    return ranges


def _trainable_step_indices(step_metadata: list[dict]) -> list[int]:
    """Indices of steps that get response_mask=1 tokens.

    All action steps are trainable. LLM-generated summarization steps are
    trainable too (their ``completion_ids`` get mask=1 in
    ``assemble_steps``). Oracle summaries are not present in
    ``episode_steps`` at all (they're written into the agent's chat
    history without an ``episode_steps.append`` call), so they don't
    appear in this list — there's nothing to filter out here.

    For now we treat every step entry as trainable; the caller is
    responsible for matching this list against the actual mask=1
    chunk count derived from response_masks.
    """
    return list(range(len(step_metadata)))


def compute_chunk_returns_terminal(
    step_metadata: list[dict],
    episode_rewards: list[float],
    gamma: float,
) -> list[ChunkOutput]:
    """Variant A: single terminal reward, discount across whole trajectory.

    Args:
        step_metadata: per-step metadata list as built by the engine.
            Length = number of chunks (action + summarization steps).
        episode_rewards: per-episode reward list. Variant A uses their
            sum as the trajectory-level R.
        gamma: discount factor in [0, 1].

    Returns:
        List of ``(G_k, is_positive)`` per step, length
        ``len(step_metadata)``.
    """
    K = len(step_metadata)
    if K == 0:
        return []
    R = float(sum(episode_rewards))
    out: list[ChunkOutput] = []
    for k in range(K):
        # Δ = K - 1 - k (last chunk has Δ = 0).
        delta = K - 1 - k
        g = (gamma ** delta) * R
        out.append((g, g > 0.0))
    return out


def _bucket_steps_per_episode(
    step_metadata: list[dict],
    n_episodes: int,
) -> list[list[int]]:
    """Partition step indices into per-episode buckets (variant B).

    Rules:

    * The first action chunk of the trajectory starts episode-0's bucket.
    * An action step with ``episode_done=True`` finalizes its current
      bucket (further action chunks fall into the next episode bucket).
    * A summarization step with ``trigger="episode_end"`` heads the
      *next* episode's bucket. The previous bucket has already been
      finalized (the preceding action step had ``episode_done=True``);
      the summary is the first chunk of the new bucket.
    * A summarization step with ``trigger="token"`` (mid-episode) falls
      into the current episode's bucket — no boundary crossing.

    The returned list always has length ``n_episodes`` even if no
    chunks landed in the last bucket (e.g. trajectory truncated mid-
    episode); empty buckets are kept as empty lists so the caller's
    ``episode_rewards`` indexing remains aligned.
    """
    buckets: list[list[int]] = [[] for _ in range(max(n_episodes, 1))]
    cur_ep = 0
    for k, meta in enumerate(step_metadata):
        is_sum = bool(meta.get("is_summarization", False))
        trigger = meta.get("trigger", None)

        if is_sum and trigger == "episode_end":
            # The preceding action step's ``episode_done=True`` already
            # advanced ``cur_ep`` (see the action-step branch below); the
            # summary "heads" the new episode we just entered. Append to
            # the current bucket without advancing again — otherwise we'd
            # double-jump and skip an episode.
            buckets[cur_ep].append(k)
            continue

        # Action step or mid-episode (token-triggered) summary.
        buckets[cur_ep].append(k)
        if not is_sum and meta.get("episode_done", False):
            # Episode ended on this action. Subsequent chunks belong to
            # the next episode unless we've already filled all buckets,
            # in which case stay clamped on the last one.
            if cur_ep + 1 < len(buckets):
                cur_ep += 1
    return buckets


def compute_chunk_returns_per_episode(
    step_metadata: list[dict],
    episode_rewards: list[float],
    gamma: float,
) -> list[ChunkOutput]:
    """Variant B: standard MDP discounted return.

    Each chunk's return is the discounted sum of *all future* episode
    rewards — i.e., for chunk ``t``,

        G_t = Σ_{k≥t} γ^(k-t) · r_k

    where ``r_k`` is the reward landed at chunk *k*. We assign each
    episode's reward to the *action chunk that ended that episode*
    (``is_summarization=False`` AND ``episode_done=True``); all other
    chunks have ``r_k = 0``. Summarization chunks therefore never carry
    a reward themselves but DO count toward the discount distance —
    matching the natural MDP semantics where every action between two
    rewards adds one tick of discount.

    Concretely, for an action ``a`` in episode *e*,

        G(a) = γ^Δ_e · R_e + γ^Δ_{e+1} · R_{e+1} + ... + γ^Δ_{E-1} · R_{E-1}

    where ``Δ_x`` is the chunk distance from ``a`` to the action chunk
    that ended episode *x*. This propagates future-episode credit back
    to the early chunks of failed episodes — a chunk in a failed
    episode 1 still gets non-zero G_k if episode 2 succeeds.

    Args:
        step_metadata: per-step metadata, as built by the engine.
        episode_rewards: per-episode reward list. ``episode_rewards[e]``
            is the terminal reward of episode *e*. Assigned in order to
            successive episode-ending action chunks; trailing entries
            beyond the trajectory's actual ``episode_done=True`` count
            (e.g. when the trajectory was truncated mid-episode and the
            env appended a 0-reward placeholder) are ignored.
        gamma: discount factor in [0, 1].

    Returns:
        List of ``(G_k, is_positive)`` per step, length
        ``len(step_metadata)``.
    """
    K = len(step_metadata)
    if K == 0:
        return []
    if not episode_rewards:
        return [(0.0, False) for _ in range(K)]

    # Per-chunk instantaneous reward r_k. Lands on the action chunk that
    # closed each successive episode. Summary chunks contribute 0 (the
    # reward was already realized on the preceding action).
    r = [0.0] * K
    ep_counter = 0
    for k, meta in enumerate(step_metadata):
        if (
            not meta.get("is_summarization", False)
            and meta.get("episode_done", False)
            and ep_counter < len(episode_rewards)
        ):
            r[k] = float(episode_rewards[ep_counter])
            ep_counter += 1

    # Bellman backward pass: G[K-1] = r[K-1]; G[k] = r[k] + γ · G[k+1].
    # O(K) one-pass; no per-episode bucketing needed.
    G = [0.0] * K
    G[K - 1] = r[K - 1]
    for k in range(K - 2, -1, -1):
        G[k] = r[k] + gamma * G[k + 1]

    return [(g, g > 0.0) for g in G]


def compute_chunk_returns_for_trajectory(
    traj: dict,
    *,
    scope: str,
    gamma: float,
) -> list[ChunkOutput]:
    """Dispatch on ``scope`` and compute per-chunk returns for one trajectory.

    Reads ``step_metadata`` and ``episode_rewards`` from the trajectory
    dict (both populated by :class:`SummarizingAgentExecutionEngine`).
    """
    step_metadata = traj.get("step_metadata") or []
    episode_rewards = traj.get("episode_rewards") or []
    if scope == "terminal":
        return compute_chunk_returns_terminal(
            step_metadata, episode_rewards, gamma
        )
    if scope == "per_episode":
        return compute_chunk_returns_per_episode(
            step_metadata, episode_rewards, gamma
        )
    raise ValueError(
        f"chunk_advantage: unknown reward scope {scope!r}; "
        "expected 'terminal' or 'per_episode'."
    )


def compute_chunk_returns_for_batch(
    trajectories: Iterable[dict],
    *,
    scope: str,
    gamma: float,
) -> list[list[ChunkOutput]]:
    """Compute per-chunk returns for every trajectory in the batch.

    Returns a list-of-lists in trajectory order; outer list length =
    number of trajectories, each inner list has one entry per chunk.

    Downstream (``segment_expansion.build_expanded_dataproto``) walks
    these in segment-row order, identifies chunk boundaries via
    response_masks, and writes the per-chunk scalars into the per-token
    ``advantages`` and ``is_positive`` tensors.
    """
    return [
        compute_chunk_returns_for_trajectory(traj, scope=scope, gamma=gamma)
        for traj in trajectories
    ]
