"""Unit + integration tests for ``trainers.length_penalty``.

Pure-Python; no torch / Ray / Hydra dependencies. The integration tests
also exercise ``trainers.chunk_advantage`` to verify shaped rewards
flow through variant A and variant B correctly.
"""

from __future__ import annotations

import math

import pytest

from trainers.chunk_advantage import (
    compute_chunk_returns_per_episode,
    compute_chunk_returns_terminal,
)
from trainers.length_penalty import (
    apply_length_penalty_to_episode_rewards,
    compute_episode_length_penalty,
    per_episode_response_tokens,
)


# ---------------------------------------------------------------------------
# compute_episode_length_penalty
# ---------------------------------------------------------------------------

class TestPiecewisePenalty:
    """L_max=1000, L_cache=200 → safe [0, 800], soft (800, 1000], cliff (1000, ∞)."""

    L_MAX = 1000
    L_CACHE = 200

    def test_safe_zero_tokens(self):
        assert compute_episode_length_penalty(0, self.L_MAX, self.L_CACHE) == 0.0

    def test_safe_just_below_threshold(self):
        assert compute_episode_length_penalty(799, self.L_MAX, self.L_CACHE) == 0.0

    def test_safe_at_threshold(self):
        assert compute_episode_length_penalty(800, self.L_MAX, self.L_CACHE) == 0.0

    def test_soft_one_above_threshold(self):
        p = compute_episode_length_penalty(801, self.L_MAX, self.L_CACHE)
        assert math.isclose(p, -1.0 / 200, rel_tol=1e-12)

    def test_soft_midpoint(self):
        p = compute_episode_length_penalty(900, self.L_MAX, self.L_CACHE)
        assert math.isclose(p, -0.5, rel_tol=1e-12)

    def test_soft_at_cap(self):
        p = compute_episode_length_penalty(1000, self.L_MAX, self.L_CACHE)
        assert math.isclose(p, -1.0, rel_tol=1e-12)

    def test_cliff_one_above_cap(self):
        assert compute_episode_length_penalty(1001, self.L_MAX, self.L_CACHE) == -1.0

    def test_cliff_far_above_cap(self):
        assert compute_episode_length_penalty(99999, self.L_MAX, self.L_CACHE) == -1.0


class TestEdgeCases:
    def test_l_cache_zero_acts_as_step_function(self):
        assert compute_episode_length_penalty(0, 100, 0) == 0.0
        assert compute_episode_length_penalty(100, 100, 0) == 0.0
        assert compute_episode_length_penalty(101, 100, 0) == -1.0

    def test_l_max_zero_any_positive_y_is_cliff(self):
        assert compute_episode_length_penalty(0, 0, 0) == 0.0
        assert compute_episode_length_penalty(1, 0, 100) == -1.0

    def test_negative_y_treated_as_zero(self):
        assert compute_episode_length_penalty(-50, 1000, 200) == 0.0

    def test_l_cache_equals_l_max_full_linear(self):
        # Threshold = 0; entire [0, L_max] is the soft zone.
        p = compute_episode_length_penalty(500, 1000, 1000)
        assert math.isclose(p, -0.5, rel_tol=1e-12)
        # |y|=0 still safe (penalty 0).
        assert compute_episode_length_penalty(0, 1000, 1000) == 0.0
        # |y|=L_max still gets full -1.
        assert math.isclose(
            compute_episode_length_penalty(1000, 1000, 1000), -1.0, rel_tol=1e-12,
        )


# ---------------------------------------------------------------------------
# per_episode_response_tokens
# ---------------------------------------------------------------------------

def _action(*, episode_index: int, delta: int, episode_done: bool = False):
    return {
        "is_summarization": False,
        "trigger": None,
        "episode_done": episode_done,
        "episode_index": episode_index,
        "response_token_delta": delta,
    }


def _summary(*, trigger: str = "episode_end", delta: int = 0):
    return {
        "is_summarization": True,
        "trigger": trigger,
        "response_token_delta": delta,
    }


class TestPerEpisodeTokens:
    def test_empty_steps(self):
        assert per_episode_response_tokens([], n_episodes=3) == [0, 0, 0]

    def test_n_episodes_zero(self):
        assert per_episode_response_tokens([_action(episode_index=0, delta=5)], 0) == []

    def test_action_steps_only_per_episode_bucketing(self):
        steps = [
            _action(episode_index=0, delta=100),
            _action(episode_index=0, delta=50, episode_done=True),
            _action(episode_index=1, delta=200, episode_done=True),
            _action(episode_index=2, delta=80, episode_done=True),
        ]
        assert per_episode_response_tokens(steps, n_episodes=3) == [150, 200, 80]

    def test_summary_inherits_preceding_action_episode_index(self):
        # Summary attributed to its closing episode (the one whose action
        # just ran).
        steps = [
            _action(episode_index=0, delta=100, episode_done=True),
            _summary(delta=50),                                       # → ep 0
            _action(episode_index=1, delta=200, episode_done=True),
            _summary(delta=30),                                       # → ep 1
            _action(episode_index=2, delta=80, episode_done=True),
        ]
        assert per_episode_response_tokens(steps, n_episodes=3) == [150, 230, 80]

    def test_token_triggered_mid_episode_summary_stays_in_current_episode(self):
        # Summary fires mid-episode (trigger="token"); attributed to the
        # currently active episode (action just ran).
        steps = [
            _action(episode_index=0, delta=100),
            _summary(trigger="token", delta=40),                      # → ep 0
            _action(episode_index=0, delta=60, episode_done=True),
            _action(episode_index=1, delta=200, episode_done=True),
        ]
        assert per_episode_response_tokens(steps, n_episodes=2) == [200, 200]

    def test_truncated_tail_episode_tokens_preserved(self):
        # Ep 1 truncated: last action carries episode_index=1 but no
        # episode_done=True. Tokens still attributed correctly.
        steps = [
            _action(episode_index=0, delta=300, episode_done=True),
            _action(episode_index=1, delta=150),  # truncated tail
        ]
        assert per_episode_response_tokens(steps, n_episodes=2) == [300, 150]

    def test_missing_response_token_delta_treated_as_zero(self):
        # A step dict from a legacy code path without the field stamped
        # should contribute zero rather than crash.
        steps = [
            {
                "is_summarization": False,
                "episode_index": 0,
                "episode_done": True,
                # no response_token_delta
            },
        ]
        assert per_episode_response_tokens(steps, n_episodes=1) == [0]

    def test_stray_episode_index_dropped(self):
        # Episode index out of range — silently dropped.
        steps = [
            _action(episode_index=5, delta=999, episode_done=True),
        ]
        assert per_episode_response_tokens(steps, n_episodes=2) == [0, 0]


# ---------------------------------------------------------------------------
# apply_length_penalty_to_episode_rewards
# ---------------------------------------------------------------------------

class TestApplyToEpisodeRewards:
    def test_empty_ep_rewards(self):
        shaped, tokens, penalties = apply_length_penalty_to_episode_rewards(
            [], [], l_max=1000, l_cache=200,
        )
        assert shaped == [] and tokens == [] and penalties == []

    def test_all_safe_zone_no_change(self):
        ep_rewards = [1.0, 0.0, 1.0]
        steps = [
            _action(episode_index=0, delta=100, episode_done=True),
            _action(episode_index=1, delta=200, episode_done=True),
            _action(episode_index=2, delta=300, episode_done=True),
        ]
        shaped, tokens, penalties = apply_length_penalty_to_episode_rewards(
            ep_rewards, steps, l_max=1000, l_cache=200,
        )
        assert tokens == [100, 200, 300]
        assert penalties == [0.0, 0.0, 0.0]
        assert shaped == [1.0, 0.0, 1.0]

    def test_mixed_zones(self):
        ep_rewards = [1.0, 1.0, 0.5]
        steps = [
            _action(episode_index=0, delta=500, episode_done=True),    # safe → 0
            _action(episode_index=1, delta=900, episode_done=True),    # soft → -0.5
            _action(episode_index=2, delta=1500, episode_done=True),   # cliff → -1
        ]
        shaped, tokens, penalties = apply_length_penalty_to_episode_rewards(
            ep_rewards, steps, l_max=1000, l_cache=200,
        )
        assert tokens == [500, 900, 1500]
        assert penalties[0] == 0.0
        assert math.isclose(penalties[1], -0.5, rel_tol=1e-12)
        assert penalties[2] == -1.0
        assert shaped[0] == 1.0
        assert math.isclose(shaped[1], 0.5, rel_tol=1e-12)
        assert math.isclose(shaped[2], -0.5, rel_tol=1e-12)

    def test_summary_tokens_added_to_preceding_episode(self):
        ep_rewards = [1.0, 1.0]
        # Episode 0: 400 action tokens + 700 summary tokens = 1100 (cliff).
        # Episode 1: 200 action tokens (safe).
        steps = [
            _action(episode_index=0, delta=400, episode_done=True),
            _summary(delta=700),                                       # → ep 0
            _action(episode_index=1, delta=200, episode_done=True),
        ]
        shaped, tokens, penalties = apply_length_penalty_to_episode_rewards(
            ep_rewards, steps, l_max=1000, l_cache=200,
        )
        assert tokens == [1100, 200]
        assert penalties == [-1.0, 0.0]
        assert shaped == [0.0, 1.0]

    def test_truncated_tail_gets_own_penalty(self):
        # Trailing 0.0 episode reward from MultiEpisodeEnv (truncated incomplete).
        ep_rewards = [1.0, 0.0]
        steps = [
            _action(episode_index=0, delta=1200, episode_done=True),   # cliff
            _action(episode_index=1, delta=50),                         # safe truncated
        ]
        shaped, tokens, penalties = apply_length_penalty_to_episode_rewards(
            ep_rewards, steps, l_max=1000, l_cache=200,
        )
        assert tokens == [1200, 50]
        assert penalties == [-1.0, 0.0]
        assert shaped == [0.0, 0.0]


# ---------------------------------------------------------------------------
# Integration with chunk_advantage
# ---------------------------------------------------------------------------

def _make_user_example():
    """Trajectory: [a×3 ep0 cliff (1200), s_ep, a×2 ep1 safe (500),
    s_ep, a×2 ep2 soft (900)]. All episodes get raw reward 1.0."""
    episode_steps = []
    # ep 0: 3 actions, 400 toks each (no summary tokens needed for the test).
    for i in range(3):
        episode_steps.append(
            _action(episode_index=0, delta=400, episode_done=(i == 2))
        )
    # episode_end summary attributed to ep 0 (delta=0 keeps ep 0 in cliff).
    episode_steps.append(_summary(delta=0))
    # ep 1: 2 actions, 250 toks each → 500 safe.
    for i in range(2):
        episode_steps.append(
            _action(episode_index=1, delta=250, episode_done=(i == 1))
        )
    episode_steps.append(_summary(delta=0))
    # ep 2: 2 actions, 450 toks each → 900 soft.
    for i in range(2):
        episode_steps.append(
            _action(episode_index=2, delta=450, episode_done=(i == 1))
        )
    ep_rewards = [1.0, 1.0, 1.0]
    return episode_steps, ep_rewards


def _build_step_metadata(episode_steps):
    return [
        {
            "is_summarization": s.get("is_summarization", False),
            "trigger": s.get("trigger", None),
            "episode_done": s.get("episode_done", False),
            "episode_index": s.get("episode_index", 0),
        }
        for s in episode_steps
    ]


class TestIntegrationWithChunkAdvantage:
    def test_variant_a_uses_shaped_total(self):
        episode_steps, ep_rewards = _make_user_example()
        shaped, tokens, penalties = apply_length_penalty_to_episode_rewards(
            ep_rewards, episode_steps, l_max=1000, l_cache=200,
        )
        # ep 0 cliff (-1), ep 1 safe (0), ep 2 soft (900 → -0.5).
        assert tokens == [1200, 500, 900]
        assert penalties == [-1.0, 0.0, -0.5]
        assert shaped == [0.0, 1.0, 0.5]
        # With γ=1, variant A broadcasts R_total = sum(shaped) = 1.5 to all chunks.
        step_metadata = _build_step_metadata(episode_steps)
        out = compute_chunk_returns_terminal(step_metadata, shaped, gamma=1.0)
        for g, is_pos in out:
            assert math.isclose(g, 1.5, rel_tol=1e-12)
            assert is_pos is True

    def test_variant_b_bellman_uses_shaped_rewards(self):
        episode_steps, ep_rewards = _make_user_example()
        shaped, _, _ = apply_length_penalty_to_episode_rewards(
            ep_rewards, episode_steps, l_max=1000, l_cache=200,
        )
        # shaped = [0.0, 1.0, 0.5]
        step_metadata = _build_step_metadata(episode_steps)
        out = compute_chunk_returns_per_episode(step_metadata, shaped, gamma=1.0)
        # With γ=1, every chunk's G is the cumulative sum of all future
        # per-episode rewards (lands on action-with-episode_done=True).
        # First chunk sees full sum = 0.0 + 1.0 + 0.5 = 1.5.
        first_g, first_pos = out[0]
        assert math.isclose(first_g, 1.5, rel_tol=1e-12)
        assert first_pos is True
        # Last chunk (last action of ep 2) holds 0.5 (only ep 2's reward).
        last_g, _ = out[-1]
        assert math.isclose(last_g, 0.5, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Negative-return correctness (interaction with chunk_advantage)
# ---------------------------------------------------------------------------

class TestNegativeReturnsFlow:
    def test_variant_a_negative_total_reaches_all_chunks(self):
        episode_steps = [
            _action(episode_index=0, delta=1500, episode_done=True),   # cliff
            _action(episode_index=1, delta=1500, episode_done=True),   # cliff
        ]
        ep_rewards = [0.0, 0.0]
        shaped, _, _ = apply_length_penalty_to_episode_rewards(
            ep_rewards, episode_steps, l_max=1000, l_cache=200,
        )
        # shaped = [-1, -1] → R_total = -2.
        step_metadata = _build_step_metadata(episode_steps)
        out = compute_chunk_returns_terminal(step_metadata, shaped, gamma=1.0)
        assert all(g == -2.0 and not is_pos for g, is_pos in out)

    def test_variant_b_negative_carries_back_via_bellman(self):
        episode_steps = [
            _action(episode_index=0, delta=1500, episode_done=True),   # cliff
            _action(episode_index=1, delta=100, episode_done=True),    # safe
        ]
        ep_rewards = [0.0, 1.0]
        shaped, _, _ = apply_length_penalty_to_episode_rewards(
            ep_rewards, episode_steps, l_max=1000, l_cache=200,
        )
        # shaped = [-1.0, 1.0]
        step_metadata = _build_step_metadata(episode_steps)
        out = compute_chunk_returns_per_episode(step_metadata, shaped, gamma=1.0)
        g_values = [g for g, _ in out]
        # G[1] = 1.0 (ep 1 last). G[0] = -1.0 + 1·1.0 = 0.0.
        assert math.isclose(g_values[0], 0.0, rel_tol=1e-12)
        assert math.isclose(g_values[1], 1.0, rel_tol=1e-12)
