"""Unit tests for ``trainers.chunk_advantage``.

Covers both reward variants and edge cases for variant B's per-episode
bucketing. Pure-Python tests; no torch / Ray / Hydra dependencies.
"""

from __future__ import annotations

import math

import pytest

from trainers.chunk_advantage import (
    _bucket_steps_per_episode,
    _segment_step_ranges,
    compute_chunk_returns_for_batch,
    compute_chunk_returns_for_trajectory,
    compute_chunk_returns_per_episode,
    compute_chunk_returns_terminal,
)


# --------------------------- helpers -------------------------------------

def _action_step(*, episode_done: bool = False, episode_index: int = 0) -> dict:
    return {
        "is_summarization": False,
        "trigger": None,
        "episode_done": episode_done,
        "episode_index": episode_index,
    }


def _sum_step(*, trigger: str) -> dict:
    return {
        "is_summarization": True,
        "trigger": trigger,
        "episode_done": False,
        "episode_index": 0,
    }


def _user_example_step_metadata() -> list[dict]:
    """Trajectory shape: [a×5, s_ep, a×3, s_ep, a×2] with episodic summaries."""
    md = []
    # Episode 0: 5 actions, last ends the episode.
    for i in range(5):
        md.append(_action_step(episode_done=(i == 4), episode_index=0))
    # Episode-end summary heading episode 1.
    md.append(_sum_step(trigger="episode_end"))
    # Episode 1: 3 actions, last ends the episode.
    for i in range(3):
        md.append(_action_step(episode_done=(i == 2), episode_index=1))
    # Episode-end summary heading episode 2.
    md.append(_sum_step(trigger="episode_end"))
    # Episode 2: 2 actions, last ends the episode.
    for i in range(2):
        md.append(_action_step(episode_done=(i == 1), episode_index=2))
    return md


# --------------------------- segment ranges ------------------------------

def test_segment_step_ranges_no_summary():
    assert _segment_step_ranges(5, []) == [(0, 5)]


def test_segment_step_ranges_with_two_summaries():
    # 12 steps total, summaries at indices 5 and 9.
    assert _segment_step_ranges(12, [5, 9]) == [(0, 6), (6, 10), (10, 12)]


# --------------------------- variant A -----------------------------------

def test_variant_a_uniform_reward_gamma_one():
    """γ=1, all rewards +1: every chunk gets G_k = sum(rewards) = 3."""
    md = _user_example_step_metadata()
    out = compute_chunk_returns_terminal(md, [1.0, 1.0, 1.0], gamma=1.0)
    assert len(out) == len(md) == 12
    for g, is_pos in out:
        assert g == 3.0
        assert is_pos is True


def test_variant_a_decay_gamma_half():
    """γ=0.5, rewards [0,0,1] → R_total=1, last chunk Δ=0 G=1, k-th from end G=0.5^Δ."""
    md = _user_example_step_metadata()  # K=12
    out = compute_chunk_returns_terminal(md, [0.0, 0.0, 1.0], gamma=0.5)
    # Δ for chunk index k = (K-1-k).
    for k, (g, is_pos) in enumerate(out):
        expected = (0.5 ** (11 - k)) * 1.0
        assert math.isclose(g, expected, rel_tol=1e-12)
        assert is_pos == (expected > 0)


def test_variant_a_zero_reward_all_negative_tag():
    md = _user_example_step_metadata()
    out = compute_chunk_returns_terminal(md, [0.0, 0.0, 0.0], gamma=0.9)
    assert all(g == 0.0 and not is_pos for g, is_pos in out)


def test_variant_a_empty_steps():
    assert compute_chunk_returns_terminal([], [], gamma=0.9) == []


# --------------------------- variant B bucket logic ----------------------

def test_buckets_user_example():
    md = _user_example_step_metadata()
    buckets = _bucket_steps_per_episode(md, n_episodes=3)
    assert buckets == [
        [0, 1, 2, 3, 4],     # episode 0 = 5 actions
        [5, 6, 7, 8],        # episode 1 = sum_head + 3 actions
        [9, 10, 11],         # episode 2 = sum_head + 2 actions
    ]


def test_buckets_no_summarization():
    """Without summarization, episodes are split by episode_done flags only."""
    md = [
        _action_step(episode_done=False, episode_index=0),
        _action_step(episode_done=True, episode_index=0),   # ends ep 0
        _action_step(episode_done=False, episode_index=1),
        _action_step(episode_done=True, episode_index=1),   # ends ep 1
    ]
    buckets = _bucket_steps_per_episode(md, n_episodes=2)
    assert buckets == [[0, 1], [2, 3]]


def test_buckets_token_triggered_summary_stays_in_current_episode():
    """Mid-episode summary doesn't cross an episode boundary."""
    md = [
        _action_step(episode_done=False, episode_index=0),
        _sum_step(trigger="token"),                          # mid-episode
        _action_step(episode_done=True, episode_index=0),    # ends ep 0
        _action_step(episode_done=True, episode_index=1),    # ends ep 1
    ]
    buckets = _bucket_steps_per_episode(md, n_episodes=2)
    assert buckets == [[0, 1, 2], [3]]


def test_buckets_truncated_mid_episode():
    """Step cap exhausted mid-episode → trailing bucket is empty if env
    appended a 0-reward placeholder that didn't generate steps."""
    md = [
        _action_step(episode_done=True, episode_index=0),
        _action_step(episode_done=False, episode_index=1),
        _action_step(episode_done=False, episode_index=1),
        # No episode_done=True for ep 1 — trajectory ran out of step budget.
    ]
    # The env may still report 2 episodes (1 finished + 1 attempted-but-truncated).
    buckets = _bucket_steps_per_episode(md, n_episodes=2)
    assert buckets == [[0], [1, 2]]


# --------------------------- variant B values ----------------------------

def test_variant_b_standard_mdp_return():
    """Variant B (standard MDP discounted return).

    Trajectory shape: [a×5, s_ep, a×3, s_ep, a×2], rewards [1, 0, 1], γ=0.9.

    Step indices (12 total chunks including LLM summaries):
      0..4   = ep 0 actions      → ep 0 ends at chunk 4 (R=1)
      5      = sum heading ep 1  (no reward)
      6..8   = ep 1 actions      → ep 1 ends at chunk 8 (R=0)
      9      = sum heading ep 2  (no reward)
      10,11  = ep 2 actions      → ep 2 ends at chunk 11 (R=1)

    r-array: [0,0,0,0, 1, 0, 0,0,0, 0, 0,1]
    Bellman backward gives each chunk the discounted sum of all future rewards.
    """
    md = _user_example_step_metadata()
    gamma = 0.9
    out = compute_chunk_returns_per_episode(md, [1.0, 0.0, 1.0], gamma=gamma)
    actual_g = [g for g, _ in out]

    # Hand-compute the Bellman backward.
    r = [0.0] * 12
    r[4] = 1.0
    r[8] = 0.0
    r[11] = 1.0
    expected = [0.0] * 12
    expected[11] = r[11]
    for k in range(10, -1, -1):
        expected[k] = r[k] + gamma * expected[k + 1]

    for k, (got, exp) in enumerate(zip(actual_g, expected)):
        assert math.isclose(got, exp, rel_tol=1e-12), (
            f"chunk {k}: got {got}, expected {exp}"
        )

    # Spot-check key chunks the user cares about.
    # chunk 0 (first action of ep 0): sees R_0 (Δ=4), R_2 (Δ=11).
    assert math.isclose(actual_g[0], gamma**4 + gamma**11, rel_tol=1e-12)
    # chunk 5 (summary heading ep 1): sees R_2 (Δ=6) — even though ep 1 fails!
    assert math.isclose(actual_g[5], gamma**6, rel_tol=1e-12)
    # chunk 11 (last action of ep 2): receives R_2 directly.
    assert math.isclose(actual_g[11], 1.0, rel_tol=1e-12)


def test_variant_b_failed_episode_chunks_still_see_future_reward():
    """Critical property of MDP discount: a failed episode's chunks get
    nonzero G_k if a LATER episode succeeds. The discount carries credit
    backward across episode boundaries."""
    md = _user_example_step_metadata()
    out = compute_chunk_returns_per_episode(md, [0.0, 0.0, 1.0], gamma=0.9)
    g_values = [g for g, _ in out]
    # All chunks should be > 0 because R_2 = 1 propagates back through
    # the entire trajectory via γ^Δ.
    for k, g in enumerate(g_values):
        assert g > 0.0, f"chunk {k} unexpectedly zero: {g}"
    # Conversely, if NO future reward, all chunks should be 0.
    out_zero = compute_chunk_returns_per_episode(md, [0.0, 0.0, 0.0], gamma=0.9)
    assert all(g == 0.0 for g, _ in out_zero)


def test_variant_b_topr_signs_match_g_positive():
    """is_positive == (G_k > 0) for every chunk."""
    md = _user_example_step_metadata()
    out = compute_chunk_returns_per_episode(md, [1.0, 0.0, -0.5], gamma=0.9)
    for g, is_pos in out:
        assert is_pos == (g > 0.0)


def test_variant_b_no_episodes_zeros_out():
    md = _user_example_step_metadata()
    out = compute_chunk_returns_per_episode(md, [], gamma=0.9)
    assert all(g == 0.0 and not is_pos for g, is_pos in out)


def test_variant_b_single_episode_no_summary():
    """Single-episode trajectory, no summarization. r is zero except at
    the final chunk; Bellman backward gives [γ^(K-1), …, γ, 1] · R."""
    md = [_action_step(episode_done=(i == 4), episode_index=0) for i in range(5)]
    out = compute_chunk_returns_per_episode(md, [1.0], gamma=0.5)
    expected_g = [0.5 ** 4, 0.5 ** 3, 0.5 ** 2, 0.5, 1.0]
    actual = [g for g, _ in out]
    for got, exp in zip(actual, expected_g):
        assert math.isclose(got, exp, rel_tol=1e-12)


def test_variant_b_summary_chunks_dont_carry_reward_but_count_in_distance():
    """Sanity: the summary chunk between two successful episodes has G_k
    that reflects the discount-in-distance from the next reward, not zero.
    But the summary itself doesn't ADD a reward to the running sum."""
    md = _user_example_step_metadata()
    out = compute_chunk_returns_per_episode(md, [1.0, 1.0, 1.0], gamma=0.9)
    g_values = [g for g, _ in out]
    # Summary at chunk 5 heads ep 1 (ends at chunk 8 with R=1) and sees ep 2
    # (ends at chunk 11 with R=1). G[5] = 0.9^3 · 1 + 0.9^6 · 1.
    assert math.isclose(g_values[5], 0.9**3 + 0.9**6, rel_tol=1e-12)
    # Summary at chunk 9 heads ep 2 (ends at chunk 11 with R=1).
    # G[9] = 0.9^2 · 1.
    assert math.isclose(g_values[9], 0.9**2, rel_tol=1e-12)


# --------------------------- dispatcher -----------------------------------

def test_dispatcher_terminal():
    md = _user_example_step_metadata()
    traj = {"step_metadata": md, "episode_rewards": [1.0, 0.0, 1.0]}
    out = compute_chunk_returns_for_trajectory(traj, scope="terminal", gamma=1.0)
    # γ=1, R_total=2 → every chunk gets G=2.
    assert all(g == 2.0 and is_pos for g, is_pos in out)


def test_dispatcher_per_episode():
    md = _user_example_step_metadata()
    traj = {"step_metadata": md, "episode_rewards": [1.0, 0.0, 1.0]}
    out = compute_chunk_returns_for_trajectory(traj, scope="per_episode", gamma=0.9)
    g_values = [g for g, _ in out]
    # Standard MDP discounted return:
    #   chunk 4 (ep 0 ends, R=1): receives R_0 immediately + γ^7 · R_2 (chunk 11).
    #   chunk 11 (ep 2 ends, R=1): receives R_2 directly with Δ=0.
    assert math.isclose(g_values[4], 1.0 + 0.9**7, rel_tol=1e-12)
    assert math.isclose(g_values[11], 1.0, rel_tol=1e-12)
    # Episode 1 chunks (indices 5..8) all see R_2 (at chunk 11) discounted.
    # G[8] = γ · G[9]; G[9] = γ · G[10]; G[10] = γ · 1; so all > 0.
    for k in range(5, 9):
        assert g_values[k] > 0.0, f"chunk {k} expected > 0 (sees future R_2)"


def test_dispatcher_unknown_scope_raises():
    md = _user_example_step_metadata()
    traj = {"step_metadata": md, "episode_rewards": [1.0]}
    with pytest.raises(ValueError, match="unknown reward scope"):
        compute_chunk_returns_for_trajectory(traj, scope="bogus", gamma=0.9)


def test_batch_returns_ordering():
    """compute_chunk_returns_for_batch preserves trajectory order and
    each inner list has length equal to that trajectory's chunk count."""
    md_a = _user_example_step_metadata()                           # 12 chunks
    md_b = [_action_step(episode_done=(i == 1), episode_index=0)   # 2 chunks
            for i in range(2)]
    trajs = [
        {"step_metadata": md_a, "episode_rewards": [1.0, 0.0, 1.0]},
        {"step_metadata": md_b, "episode_rewards": [1.0]},
    ]
    out = compute_chunk_returns_for_batch(trajs, scope="per_episode", gamma=0.9)
    assert len(out) == 2
    assert len(out[0]) == 12
    assert len(out[1]) == 2
