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
        from envs.webshop_env_adapter import WebShopEnvAdapter

        adapter = WebShopEnvAdapter.from_dict({
            "env_id": "webshop",
            "split": "test",
            "observation_mode": "text",
        })
        adapter.reset(seed=0)

        # Create a mock Action object
        action = MagicMock()
        action.action = "\\boxed{search[laptop]}"
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
