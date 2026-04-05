"""Tests for ALFWorldEnvAdapter.

Tests are split into two groups:
- Pure unit tests for static/class methods (no external data needed)
- Integration tests that require ALFWORLD_DATA (skipped if unavailable)
"""

from __future__ import annotations

import os
import sys
import pytest

# Ensure the repo root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rllm.agents.agent import Action  # type: ignore

# ALFWORLD_DATA must be set for integration tests and also for importing the
# adapter at module level (AlfredTWEnv reads it during import).  We guard the
# import so that pure-unit tests can still be collected when alfworld is not
# installed.
_ALFWORLD_AVAILABLE = False
try:
    from envs.alfworld_env_adapter import ALFWorldEnvAdapter, BOXED_PATTERN
    _ALFWORLD_AVAILABLE = True
except Exception:
    # alfworld or textworld not installed — only pure-unit tests will run
    ALFWorldEnvAdapter = None  # type: ignore[assignment,misc]
    BOXED_PATTERN = None  # type: ignore[assignment]

_HAS_ALFWORLD_DATA = bool(os.environ.get("ALFWORLD_DATA"))

needs_alfworld = pytest.mark.skipif(
    not _ALFWORLD_AVAILABLE,
    reason="alfworld/textworld not importable",
)
needs_alfworld_data = pytest.mark.skipif(
    not (_ALFWORLD_AVAILABLE and _HAS_ALFWORLD_DATA),
    reason="ALFWORLD_DATA not set or alfworld not importable",
)


# ======================================================================
# Action parsing
# ======================================================================


@needs_alfworld
class TestParseAction:
    """Tests for ALFWorldEnvAdapter._parse_action."""

    def test_boxed_simple(self):
        assert ALFWorldEnvAdapter._parse_action("\\boxed{go to desk 1}") == "go to desk 1"

    def test_boxed_multiple_takes_last(self):
        text = "first \\boxed{pick up apple} then \\boxed{go to shelf 1}"
        assert ALFWorldEnvAdapter._parse_action(text) == "go to shelf 1"

    def test_no_boxed_returns_raw(self):
        assert ALFWorldEnvAdapter._parse_action("just plain text") == "just plain text"

    def test_empty_string(self):
        assert ALFWorldEnvAdapter._parse_action("") == ""

    def test_unwraps_action_object(self):
        action = Action(action="\\boxed{look}")
        assert ALFWorldEnvAdapter._parse_action(action) == "look"

    def test_action_object_no_boxed(self):
        action = Action(action="open fridge 1")
        assert ALFWorldEnvAdapter._parse_action(action) == "open fridge 1"

    def test_whitespace_stripped(self):
        assert ALFWorldEnvAdapter._parse_action("  \\boxed{  heat apple  }  ") == "heat apple"

    def test_case_insensitive_boxed(self):
        assert ALFWorldEnvAdapter._parse_action("\\BOXED{look}") == "look"


# ======================================================================
# Admissible action matching
# ======================================================================


@needs_alfworld
class TestMatchAdmissible:
    """Tests for ALFWorldEnvAdapter._match_admissible."""

    ADMISSIBLE = [
        "go to desk 1",
        "go to shelf 2",
        "pick up apple 1",
        "open fridge 1",
        "look",
    ]

    def test_exact_match(self):
        assert ALFWorldEnvAdapter._match_admissible(
            "go to desk 1", self.ADMISSIBLE
        ) == "go to desk 1"

    def test_exact_match_case_insensitive(self):
        assert ALFWorldEnvAdapter._match_admissible(
            "Go To Desk 1", self.ADMISSIBLE
        ) == "go to desk 1"

    def test_prefix_match(self):
        assert ALFWorldEnvAdapter._match_admissible(
            "go to desk", self.ADMISSIBLE
        ) == "go to desk 1"

    def test_substring_match(self):
        assert ALFWorldEnvAdapter._match_admissible(
            "apple 1", self.ADMISSIBLE
        ) == "pick up apple 1"

    def test_fuzzy_match(self):
        # Typo: "go too desk 1" should fuzzy-match "go to desk 1"
        result = ALFWorldEnvAdapter._match_admissible(
            "go too desk 1", self.ADMISSIBLE
        )
        assert result == "go to desk 1"

    def test_fallback_to_look(self):
        result = ALFWorldEnvAdapter._match_admissible(
            "completely irrelevant nonsense xyz", self.ADMISSIBLE
        )
        assert result == "look"

    def test_fallback_first_when_no_look(self):
        cmds = ["put apple in fridge", "take knife from counter"]
        result = ALFWorldEnvAdapter._match_admissible(
            "completely irrelevant nonsense xyz", cmds
        )
        assert result in cmds

    def test_empty_admissible_returns_look(self):
        assert ALFWorldEnvAdapter._match_admissible("anything", []) == "look"


# ======================================================================
# Observation formatting
# ======================================================================


@needs_alfworld
class TestFormatStepObs:
    """Tests for observation formatting methods."""

    def _make_adapter(self):
        """Create a minimal adapter for testing formatting (no TW env)."""
        # Bypass __init__ to avoid needing ALFWORLD_DATA
        adapter = object.__new__(ALFWorldEnvAdapter)
        adapter.max_turns = 10
        adapter._admissible_commands = ["go to desk 1", "look"]
        adapter._task_description = "heat some apple"
        return adapter

    def test_format_step_obs_won(self):
        adapter = self._make_adapter()
        obs = adapter._format_step_obs("You heated the apple.", won=True, terminated=True, truncated=False)
        assert "Congratulations!" in obs
        assert "You heated the apple." in obs

    def test_format_step_obs_terminated_not_won(self):
        adapter = self._make_adapter()
        obs = adapter._format_step_obs("Nothing happens.", won=False, terminated=True, truncated=False)
        assert "Episode finished" in obs
        assert "not completed" in obs

    def test_format_step_obs_truncated(self):
        adapter = self._make_adapter()
        obs = adapter._format_step_obs("Nothing.", won=False, terminated=False, truncated=True)
        assert "max_turns" in obs
        assert "10" in obs

    def test_format_step_obs_normal(self):
        adapter = self._make_adapter()
        obs = adapter._format_step_obs("You see a counter.", won=False, terminated=False, truncated=False)
        assert "Admissible actions:" in obs
        assert "go to desk 1" in obs
        assert "\\boxed{}" in obs

    def test_format_init_obs(self):
        adapter = self._make_adapter()
        obs = adapter._format_init_obs("You are in the kitchen.")
        assert "heat some apple" in obs
        assert "Admissible actions:" in obs
        assert "\\boxed{}" in obs
        assert "You are in the kitchen." in obs


# ======================================================================
# Task extraction
# ======================================================================


@needs_alfworld
class TestTaskExtraction:
    """Tests for task description and task type extraction."""

    def test_extract_task_from_obs(self):
        obs = "You are in a room.\nYour task is to: heat some apple.\nLook around."
        assert ALFWorldEnvAdapter._extract_task(obs) == "heat some apple."

    def test_extract_task_no_marker(self):
        obs = "You are in a room."
        assert ALFWorldEnvAdapter._extract_task(obs) == "You are in a room."

    def test_extract_task_end_of_string(self):
        obs = "Your task is to: clean the plate"
        assert ALFWorldEnvAdapter._extract_task(obs) == "clean the plate"

    def test_extract_task_type_pick_heat(self):
        gf = "/data/json_2.1.1/valid_unseen/pick_heat_then_place_in_recep-Apple/trial/game.tw-pddl"
        assert ALFWorldEnvAdapter._extract_task_type(gf) == "pick_heat_then_place_in_recep"

    def test_extract_task_type_pick_and_place(self):
        gf = "/data/valid_unseen/pick_and_place_simple-Pen/trial_T123/game.tw-pddl"
        assert ALFWorldEnvAdapter._extract_task_type(gf) == "pick_and_place_simple"

    def test_extract_task_type_look_at_obj(self):
        gf = "/data/look_at_obj_in_light-Candle/trial/game.tw-pddl"
        assert ALFWorldEnvAdapter._extract_task_type(gf) == "look_at_obj_in_light"

    def test_extract_task_type_unknown(self):
        gf = "/data/something_else/trial/game.tw-pddl"
        assert ALFWorldEnvAdapter._extract_task_type(gf) == "unknown"


# ======================================================================
# Gamefile selection
# ======================================================================


@needs_alfworld
class TestGamefileSelection:
    """Tests for deterministic gamefile selection from seed."""

    def test_same_seed_same_file(self):
        """_select_gamefile is deterministic: same seed → same index."""
        # Use a mock sorted list
        adapter = object.__new__(ALFWorldEnvAdapter)
        adapter._sorted_game_files = [f"game_{i}.tw" for i in range(100)]
        assert adapter._select_gamefile(42) == adapter._select_gamefile(42)

    def test_different_seeds_likely_different(self):
        adapter = object.__new__(ALFWorldEnvAdapter)
        adapter._sorted_game_files = [f"game_{i}.tw" for i in range(100)]
        # Seeds 1 and 2 should pick different indices (1 % 100 != 2 % 100)
        assert adapter._select_gamefile(1) != adapter._select_gamefile(2)

    def test_seed_wraps_around(self):
        adapter = object.__new__(ALFWorldEnvAdapter)
        adapter._sorted_game_files = [f"game_{i}.tw" for i in range(10)]
        # seed=15 should wrap: 15 % 10 = 5
        assert adapter._select_gamefile(15) == "game_5.tw"


# ======================================================================
# from_dict construction
# ======================================================================


@needs_alfworld_data
class TestFromDict:
    """Tests for ALFWorldEnvAdapter.from_dict factory."""

    def test_basic_construction(self):
        info = {
            "env_id": "alfworld",
            "seed": 42,
            "max_turns": 10,
            "eval_split": "eval_out_of_distribution",
        }
        env = ALFWorldEnvAdapter.from_dict(info)
        assert env.env_id == "alfworld"
        assert env.max_turns == 10
        env.close()

    def test_env_kwargs_passthrough(self):
        info = {
            "env_id": "alfworld",
            "env_kwargs": {"max_turns": 5, "eval_split": "eval_out_of_distribution"},
        }
        env = ALFWorldEnvAdapter.from_dict(info)
        assert env.max_turns == 5
        env.close()


# ======================================================================
# Integration: deterministic reset/step
# ======================================================================


@needs_alfworld_data
class TestDeterministicBehavior:
    """Integration tests verifying deterministic reset and step behavior."""

    def test_reset_deterministic(self):
        """Same seed produces same initial observation and gamefile."""
        env1 = ALFWorldEnvAdapter.from_dict({
            "env_id": "alfworld", "seed": 42,
            "eval_split": "eval_out_of_distribution",
        })
        env2 = ALFWorldEnvAdapter.from_dict({
            "env_id": "alfworld", "seed": 42,
            "eval_split": "eval_out_of_distribution",
        })
        obs1, info1 = env1.reset(seed=42)
        obs2, info2 = env2.reset(seed=42)

        assert obs1 == obs2
        assert info1["gamefile"] == info2["gamefile"]
        assert info1["task_type"] == info2["task_type"]
        env1.close()
        env2.close()

    def test_step_deterministic(self):
        """Same action on same game produces same observation and reward."""
        env1 = ALFWorldEnvAdapter.from_dict({
            "env_id": "alfworld", "seed": 42,
            "eval_split": "eval_out_of_distribution",
        })
        env2 = ALFWorldEnvAdapter.from_dict({
            "env_id": "alfworld", "seed": 42,
            "eval_split": "eval_out_of_distribution",
        })
        env1.reset(seed=42)
        env2.reset(seed=42)

        obs1, r1, d1, i1 = env1.step("\\boxed{look}")
        obs2, r2, d2, i2 = env2.step("\\boxed{look}")

        assert obs1 == obs2
        assert r1 == r2
        assert d1 == d2
        env1.close()
        env2.close()

    def test_different_seeds_different_games(self):
        """Different seeds select different game files."""
        env1 = ALFWorldEnvAdapter.from_dict({
            "env_id": "alfworld", "seed": 1,
            "eval_split": "eval_out_of_distribution",
        })
        env2 = ALFWorldEnvAdapter.from_dict({
            "env_id": "alfworld", "seed": 2,
            "eval_split": "eval_out_of_distribution",
        })
        env1.reset(seed=1)
        env2.reset(seed=2)

        assert env1._gamefile != env2._gamefile
        env1.close()
        env2.close()

    def test_multi_step_sequence_deterministic(self):
        """A multi-step action sequence produces identical trajectories."""
        def run_trajectory(seed):
            env = ALFWorldEnvAdapter.from_dict({
                "env_id": "alfworld", "seed": seed,
                "eval_split": "eval_out_of_distribution",
                "max_turns": 5,
            })
            obs, info = env.reset(seed=seed)
            trajectory = [obs]

            for _ in range(3):
                # Always pick the first admissible command
                if env._admissible_commands:
                    action = f"\\boxed{{{env._admissible_commands[0]}}}"
                else:
                    action = "\\boxed{look}"
                obs, reward, done, info = env.step(action)
                trajectory.append((obs, reward, done))
                if done:
                    break
            env.close()
            return trajectory

        t1 = run_trajectory(42)
        t2 = run_trajectory(42)
        assert t1 == t2


# ======================================================================
# Vendor path resolution
# ======================================================================


@needs_alfworld
class TestVendorPathResolution:
    """Tests for _setup_alfworld_imports resolution strategy."""

    def test_alfworld_is_importable(self):
        """After adapter import, alfworld must be importable."""
        import alfworld.agents.environment.alfred_tw_env as mod
        assert hasattr(mod, "TASK_TYPES")
        assert hasattr(mod, "AlfredTWEnv")

    def test_textworld_is_importable(self):
        """After adapter import, textworld must be importable."""
        import textworld
        assert hasattr(textworld, "EnvInfos")
