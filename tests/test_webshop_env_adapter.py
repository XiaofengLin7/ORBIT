"""Tests for WebShopEnvAdapter.

Unit tests run without any WebShop data.  Integration tests require
``.cache/webshop/webshop.db`` and ``.cache/webshop/indexes/`` to be present
(built via ``python -m gem.envs.webshop.preprocess --mode all`` and pyserini).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

_WEBSHOP_DB = Path(".cache/webshop/webshop.db")
_WEBSHOP_IDX = Path(".cache/webshop/indexes")

needs_webshop_data = pytest.mark.skipif(
    not (_WEBSHOP_DB.exists() and _WEBSHOP_IDX.exists()),
    reason="WebShop DB/indexes not found — run setup.sh first",
)


# ---------------------------------------------------------------------------
# Unit tests (no WebShop data required)
# ---------------------------------------------------------------------------


class TestFromDict:
    """Test the from_dict factory method parameter extraction."""

    @needs_webshop_data
    def test_basic_from_dict(self):
        from envs.webshop_env_adapter import WebShopEnvAdapter

        info = {
            "env_id": "webshop",
            "split": "test",
            "observation_mode": "text",
            "max_turns": 15,
        }
        adapter = WebShopEnvAdapter.from_dict(info)
        assert adapter.env_id == "webshop"
        assert adapter._split == "test"
        assert adapter._observation_mode == "text"
        assert adapter._max_turns == 15

    @needs_webshop_data
    def test_from_dict_with_env_kwargs(self):
        from envs.webshop_env_adapter import WebShopEnvAdapter

        info = {
            "env_id": "webshop",
            "env_kwargs": {"split": "test", "observation_mode": "text"},
        }
        adapter = WebShopEnvAdapter.from_dict(info)
        assert adapter._split == "test"
        assert adapter._observation_mode == "text"

    @needs_webshop_data
    def test_from_dict_passthrough_over_env_kwargs(self):
        """Top-level keys should NOT override existing env_kwargs."""
        from envs.webshop_env_adapter import WebShopEnvAdapter

        info = {
            "env_id": "webshop",
            "env_kwargs": {"split": "test"},
            "split": "train",  # should NOT override since "split" already in env_kwargs
        }
        adapter = WebShopEnvAdapter.from_dict(info)
        assert adapter._split == "test"


class TestIsMultithreadSafe:
    def test_returns_false(self):
        from envs.webshop_env_adapter import WebShopEnvAdapter

        assert WebShopEnvAdapter.is_multithread_safe() is False


class TestActionUnwrapping:
    """Test that Action objects are properly unwrapped in step()."""

    @needs_webshop_data
    def test_action_object(self):
        from rllm.agents.agent import Action
        from envs.webshop_env_adapter import WebShopEnvAdapter

        adapter = WebShopEnvAdapter.from_dict({
            "env_id": "webshop",
            "split": "test",
            "observation_mode": "text",
        })
        adapter.reset(seed=0)

        # Use a real Action object so isinstance() check works
        action = Action(action="\\boxed{search[laptop]}")
        obs, reward, done, info = adapter.step(action)
        assert isinstance(obs, str)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "parsed_action" in info


# ---------------------------------------------------------------------------
# Integration tests (require WebShop data)
# ---------------------------------------------------------------------------


@needs_webshop_data
class TestIntegrationReset:
    """Integration tests for reset() with real WebShop data."""

    def test_reset_returns_valid_observation(self):
        from envs.webshop_env_adapter import WebShopEnvAdapter

        adapter = WebShopEnvAdapter.from_dict({
            "env_id": "webshop",
            "split": "test",
            "observation_mode": "text",
        })
        obs, info = adapter.reset(seed=42)
        assert isinstance(obs, str)
        assert len(obs) > 0
        assert info["env_id"] == "webshop"
        assert info["terminated"] is False
        assert info["truncated"] is False

    def test_deterministic_reset(self):
        """Same seed should produce the same observation."""
        from envs.webshop_env_adapter import WebShopEnvAdapter

        adapter1 = WebShopEnvAdapter.from_dict({
            "env_id": "webshop",
            "split": "test",
            "observation_mode": "text",
        })
        adapter2 = WebShopEnvAdapter.from_dict({
            "env_id": "webshop",
            "split": "test",
            "observation_mode": "text",
        })
        obs1, _ = adapter1.reset(seed=123)
        obs2, _ = adapter2.reset(seed=123)
        assert obs1 == obs2

    def test_different_seeds_different_tasks(self):
        """Different seeds should (usually) produce different tasks."""
        from envs.webshop_env_adapter import WebShopEnvAdapter

        adapter = WebShopEnvAdapter.from_dict({
            "env_id": "webshop",
            "split": "test",
            "observation_mode": "text",
        })
        obs1, _ = adapter.reset(seed=0)
        obs2, _ = adapter.reset(seed=1)
        # Different goals → different instruction text in observation
        assert obs1 != obs2

    def test_multi_episode_replay(self):
        """Resetting with the same seed should replay the same task."""
        from envs.webshop_env_adapter import WebShopEnvAdapter

        adapter = WebShopEnvAdapter.from_dict({
            "env_id": "webshop",
            "split": "test",
            "observation_mode": "text",
        })
        obs1, _ = adapter.reset(seed=42)
        # Simulate some steps
        adapter.step("\\boxed{search[laptop]}")
        # Reset again with same seed → same task
        obs2, _ = adapter.reset(seed=42)
        assert obs1 == obs2


@needs_webshop_data
class TestIntegrationStep:
    """Integration tests for step() with real WebShop data."""

    def test_search_action(self):
        from envs.webshop_env_adapter import WebShopEnvAdapter

        adapter = WebShopEnvAdapter.from_dict({
            "env_id": "webshop",
            "split": "test",
            "observation_mode": "text",
        })
        adapter.reset(seed=0)
        obs, reward, done, info = adapter.step("\\boxed{search[laptop]}")
        assert isinstance(obs, str)
        assert len(obs) > 0
        assert reward == 0.0  # search doesn't produce reward
        assert info["terminated"] is False

    def test_invalid_action_format(self):
        """Invalid action should trigger error tolerance."""
        from envs.webshop_env_adapter import WebShopEnvAdapter

        adapter = WebShopEnvAdapter.from_dict({
            "env_id": "webshop",
            "split": "test",
            "observation_mode": "text",
        })
        adapter.reset(seed=0)
        obs, reward, done, info = adapter.step("this is not a valid action")
        assert isinstance(obs, str)
        # format_error_reward is -0.1 by default
        assert reward <= 0.0

    def test_step_returns_four_tuple(self):
        from envs.webshop_env_adapter import WebShopEnvAdapter

        adapter = WebShopEnvAdapter.from_dict({
            "env_id": "webshop",
            "split": "test",
            "observation_mode": "text",
        })
        adapter.reset(seed=0)
        result = adapter.step("\\boxed{search[test]}")
        assert len(result) == 4
        obs, reward, done, info = result
        assert isinstance(obs, str)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)


@needs_webshop_data
class TestIntegrationWithMultiEpisodeEnv:
    """Test that the adapter works correctly with MultiEpisodeEnv."""

    def test_from_dict_roundtrip(self):
        from envs.multi_episode_env import MultiEpisodeEnv

        info = {
            "inner_env_class": "envs.webshop_env_adapter.WebShopEnvAdapter",
            "env_id": "webshop",
            "seed": 42,
            "uid": "webshop-42",
            "data_source": "webshop",
            "max_turns_per_episode": 15,
            "total_step_cap": 45,
            "split": "test",
            "observation_mode": "text",
            "success_reward": 1.0,
        }
        env = MultiEpisodeEnv.from_dict(info)
        obs, env_info = env.reset(seed=42)
        assert isinstance(obs, str)
        assert len(obs) > 0
        assert env_info["multi_episode"] is True

    def test_multi_episode_trajectory(self):
        """Run a short multi-episode trajectory to verify integration."""
        from envs.multi_episode_env import MultiEpisodeEnv

        info = {
            "inner_env_class": "envs.webshop_env_adapter.WebShopEnvAdapter",
            "env_id": "webshop",
            "seed": 10,
            "uid": "webshop-10",
            "data_source": "webshop",
            "max_turns_per_episode": 5,
            "total_step_cap": 10,
            "split": "test",
            "observation_mode": "text",
            "success_reward": 1.0,
        }
        env = MultiEpisodeEnv.from_dict(info)
        obs, _ = env.reset(seed=10)

        done = False
        steps = 0
        while not done and steps < 15:
            obs, reward, done, info = env.step("\\boxed{search[laptop]}")
            steps += 1

        metrics = env.get_metrics()
        assert "episode/success_rate" in metrics
        assert "episode/num_episodes" in metrics


@needs_webshop_data
class TestEpisodeFeedback:
    """Test that the adapter returns correct feedback for each outcome scenario."""

    def _make_adapter(self):
        from envs.webshop_env_adapter import WebShopEnvAdapter

        return WebShopEnvAdapter.from_dict({
            "env_id": "webshop",
            "split": "test",
            "observation_mode": "text",
        })

    def test_feedback_timeout(self):
        """Episode times out without a purchase → truncation message."""
        from envs.webshop_env_adapter import WebShopEnvAdapter

        adapter = WebShopEnvAdapter.from_dict({
            "env_id": "webshop",
            "split": "test",
            "observation_mode": "text",
            "max_turns": 3,
        })
        adapter.reset(seed=0)
        # Exhaust all turns with searches (never buy)
        for _ in range(3):
            obs, reward, done, info = adapter.step("\\boxed{search[laptop]}")
        assert done is True
        assert info["truncated"] is True
        assert "max_turns" in obs
        assert "did not make a purchase" in obs

    def test_feedback_zero_reward(self):
        """Error tolerance exceeded with 0 reward → 'You failed.'"""
        from envs.webshop_env_adapter import WebShopEnvAdapter

        adapter = WebShopEnvAdapter.from_dict({
            "env_id": "webshop",
            "split": "test",
            "observation_mode": "text",
            "error_tolerance": 0,
            "format_error_reward": 0.0,
        })
        adapter.reset(seed=0)
        # One invalid action exceeds error_tolerance=0 → done with reward=0
        obs, reward, done, info = adapter.step("invalid action without boxed")
        assert done is True
        assert reward == 0.0
        assert info["terminated"] is True
        assert obs == "You failed."

    def test_feedback_partial_reward(self):
        """Buy an item with partial match → score only, no congratulations.

        Reward is binarized: partial match → reward=0.0 (not a success).
        The actual WebShop score is in info["raw_reward"].
        """
        adapter = self._make_adapter()
        adapter.reset(seed=0)
        # Search → click product → buy now (partial match expected)
        adapter.step("\\boxed{search[slim fit mens henleys short sleeve]}")
        adapter.step("\\boxed{click[b09qqp3356]}")
        obs, reward, done, info = adapter.step("\\boxed{click[buy now]}")
        assert done is True
        assert reward == 0.0  # binarized: partial match is NOT success
        assert 0 < info["raw_reward"] < 1.0  # actual score preserved
        assert "Your score:" in obs
        assert "Congratulations" not in obs
        assert "failed" not in obs

    def test_feedback_perfect_reward(self):
        """Buy with perfect match → Congratulations message.

        Finding a perfect match is hard in the test set, so we test
        the formatting logic directly via _format_step_obs.
        """
        adapter = self._make_adapter()
        obs = adapter._format_step_obs("raw", {}, 1.0, terminated=True, truncated=False)
        assert "Congratulations" in obs
        assert "1.00 / 1.00" in obs

    def test_three_episode_feedback_sequence(self):
        """Multi-episode trajectory with mixed outcomes across 3 episodes.

        Episode 1: timeout after 5 turns (no purchase)
        Episode 2: partial match (search + click + buy = 3 turns, done early)
        Episode 3: timeout after 5 turns (no purchase)
        total_step_cap = 13 = 5 + 3 + 5
        """
        from envs.multi_episode_env import MultiEpisodeEnv

        env = MultiEpisodeEnv.from_dict({
            "inner_env_class": "envs.webshop_env_adapter.WebShopEnvAdapter",
            "env_id": "webshop",
            "seed": 0,
            "uid": "webshop-0",
            "data_source": "webshop",
            "max_turns_per_episode": 5,
            "total_step_cap": 13,
            "split": "test",
            "observation_mode": "text",
            "success_reward": 1.0,
            "enable_reflection": False,
        })
        obs, _ = env.reset(seed=0)

        # --- Episode 1: just search, times out after 5 turns ---
        for _ in range(5):
            obs, reward, done, info = env.step("\\boxed{search[laptop]}")
        # Should see timeout feedback + Episode 2 header
        assert "did not make a purchase" in obs
        assert "[Episode 2]" in obs
        assert done is False

        # --- Episode 2: search → click → buy now (partial match, 3 turns) ---
        obs, _, _, _ = env.step("\\boxed{search[slim fit mens henleys short sleeve]}")
        obs, _, _, _ = env.step("\\boxed{click[b09qqp3356]}")
        obs, reward, done, info = env.step("\\boxed{click[buy now]}")
        # Buy terminates before max_turns → score feedback + Episode 3 header
        assert "Your score:" in obs
        assert "Congratulations" not in obs  # partial, not perfect
        assert "[Episode 3]" in obs
        assert done is False

        # --- Episode 3: search until budget exhausted (5 more turns) ---
        for _ in range(5):
            obs, reward, done, info = env.step("\\boxed{search[phone]}")
        # Inner truncation + outer done coincide at step 13
        assert "did not make a purchase" in obs
        assert done is True

        # Verify per-episode metrics
        metrics = env.get_metrics()
        # Episode 1: timeout → no reward, not success
        assert metrics["episode_1/success_rate"] == 0.0
        assert metrics["episode_1/reward"] == 0.0
        # Episode 2: partial match → raw_reward tracked, but NOT success
        # (only reward == 1.0 counts as success for WebShop)
        assert metrics["episode_2/success_rate"] == 0.0
        assert 0 < metrics["episode_2/reward"] < 1.0  # raw score preserved
        # Aggregate: success_rate=0 (no perfect match), but avg_reward > 0
        assert metrics["episode/success_rate"] == 0.0
        assert metrics["episode/avg_reward"] > 0.0
        assert metrics["episode/num_episodes"] == 3
        # Check raw rewards list (from info["raw_reward"])
        rewards = env._episode_rewards
        assert rewards[0] == 0.0    # episode 1: timeout
        assert 0 < rewards[1] < 1.0  # episode 2: partial match raw score
        assert rewards[2] == 0.0    # episode 3: timeout
