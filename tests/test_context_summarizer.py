"""Unit tests for the ContextSummarizerMixin and composed agent classes."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from agents.context_summarizer import (
    ContextSummarizerMixin,
    GEMTextAgentWithSummarization,
    GEMTextAgentNonCumulativeWithSummarization,
)


SYSTEM_PROMPT = "You are a helpful agent. Return actions inside \\boxed{}."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(cls=GEMTextAgentWithSummarization, threshold=100, **kwargs):
    """Create an agent with a low threshold for testing."""
    defaults = dict(
        system_prompt=SYSTEM_PROMPT,
        summarization_threshold_tokens=threshold,
        summary_max_tokens=64,
        max_summarizations=3,
    )
    defaults.update(kwargs)
    return cls(**defaults)


def _make_tokenizer_and_parser():
    """Return a mock tokenizer + parser that count words as tokens."""
    tokenizer = MagicMock()
    # Simple word-level "tokenizer": each word → one token id.
    tokenizer.encode = MagicMock(
        side_effect=lambda text, add_special_tokens=False: list(range(len(text.split())))
    )

    parser = MagicMock()
    # parser.parse concatenates message contents with role prefix.
    def fake_parse(messages, add_generation_prompt=False, is_first_msg=False):
        parts = []
        for m in messages:
            parts.append(f"{m['role']}: {m['content']}")
        return "\n".join(parts)

    parser.parse = MagicMock(side_effect=fake_parse)
    parser.assistant_token = "assistant:"
    return tokenizer, parser


def _simulate_steps(agent, n_steps=5):
    """Simulate n_steps of env→model interaction."""
    for i in range(n_steps):
        agent.update_from_env(
            observation=f"Observation {i} with " + "padding " * 20,
            reward=0.0,
            done=False,
            info={},
        )
        agent.update_from_model(f"Thinking about step {i}... \\boxed{{action_{i}}}")


# ---------------------------------------------------------------------------
# Tests: __init__ / reset
# ---------------------------------------------------------------------------

class TestInitAndReset:
    def test_init_sets_summarization_params(self):
        agent = _make_agent(threshold=500)
        assert agent.summarization_threshold_tokens == 500
        assert agent.summary_max_tokens == 64
        assert agent.max_summarizations == 3
        assert agent._summarization_count == 0

    def test_init_pops_kwargs_before_super(self):
        """Summarization kwargs must not leak to the parent __init__."""
        agent = _make_agent()
        assert agent.system_prompt == SYSTEM_PROMPT

    def test_reset_clears_summarization_state(self):
        agent = _make_agent()
        agent._summarization_count = 5
        agent._summarization_events = [{"x": 1}]
        agent.reset()
        assert agent._summarization_count == 0
        assert agent._summarization_events == []
        # Messages should be reset too (parent behaviour).
        assert len(agent._messages) == 1  # just system prompt

    def test_default_kwargs(self):
        """Verify defaults when no summarization kwargs are passed."""
        agent = GEMTextAgentWithSummarization(system_prompt=SYSTEM_PROMPT)
        assert agent.summarization_threshold_tokens == 16384
        assert agent.summary_max_tokens == 8192
        assert agent.max_summarizations == 5


# ---------------------------------------------------------------------------
# Tests: should_summarize
# ---------------------------------------------------------------------------

class TestShouldSummarize:
    def test_returns_false_when_under_threshold(self):
        agent = _make_agent(threshold=99999)
        _simulate_steps(agent, 2)
        tokenizer, parser = _make_tokenizer_and_parser()
        assert agent.should_summarize(tokenizer, parser) is False

    def test_returns_true_when_over_threshold(self):
        agent = _make_agent(threshold=10)
        _simulate_steps(agent, 3)
        tokenizer, parser = _make_tokenizer_and_parser()
        assert agent.should_summarize(tokenizer, parser) is True

    def test_returns_false_when_too_few_messages(self):
        agent = _make_agent(threshold=1)
        # Only system prompt + 1 user message (no assistant yet).
        agent.update_from_env(observation="hello", reward=0.0, done=False, info={})
        tokenizer, parser = _make_tokenizer_and_parser()
        assert agent.should_summarize(tokenizer, parser) is False

    def test_returns_false_when_max_summarizations_reached(self):
        agent = _make_agent(threshold=10, max_summarizations=0)
        _simulate_steps(agent, 5)
        tokenizer, parser = _make_tokenizer_and_parser()
        assert agent.should_summarize(tokenizer, parser) is False


# ---------------------------------------------------------------------------
# Tests: build_summarization_prompt
# ---------------------------------------------------------------------------

class TestBuildSummarizationPrompt:
    def test_appends_instruction_message(self):
        agent = _make_agent()
        _simulate_steps(agent, 2)
        prompt = agent.build_summarization_prompt()
        # Last message should be the summarization instruction.
        assert prompt[-1]["role"] == "user"
        assert "context_summary" in prompt[-1]["content"]
        # All prior messages preserved + 1 instruction.
        assert len(prompt) == len(agent._messages) + 1

    def test_does_not_mutate_agent_messages(self):
        agent = _make_agent()
        _simulate_steps(agent, 2)
        original_len = len(agent._messages)
        _ = agent.build_summarization_prompt()
        assert len(agent._messages) == original_len

    def test_includes_full_history(self):
        """Summarization prompt should include all messages (no trimming)."""
        agent = _make_agent()
        _simulate_steps(agent, 4)
        prompt = agent.build_summarization_prompt()
        # Should have all agent messages + 1 instruction.
        assert len(prompt) == len(agent._messages) + 1
        # First message is system, last is instruction.
        assert prompt[0]["role"] == "system"
        assert prompt[-1]["role"] == "user"


# ---------------------------------------------------------------------------
# Tests: apply_summary
# ---------------------------------------------------------------------------

class TestApplySummary:
    def test_basic_replacement(self):
        agent = _make_agent()
        _simulate_steps(agent, 5)
        original_len = len(agent._messages)

        summary_text = "<context_summary>This is a summary of previous actions.</context_summary>"
        agent.apply_summary(summary_text)

        # Should have only system + summary (no preserved turns).
        assert len(agent._messages) == 2
        # First message is still the system prompt.
        assert agent._messages[0]["role"] == "system"
        # Second message contains the summary.
        assert "<context_summary>" in agent._messages[1]["content"]
        assert "This is a summary of previous actions." in agent._messages[1]["content"]

    def test_no_preserved_turns(self):
        """After summarization, only system + summary remain."""
        agent = _make_agent()
        _simulate_steps(agent, 5)
        agent.apply_summary("<context_summary>Summary here.</context_summary>")
        assert len(agent._messages) == 2
        assert agent._messages[0]["role"] == "system"
        assert agent._messages[1]["role"] == "user"

    def test_fallback_when_no_tags(self):
        agent = _make_agent()
        _simulate_steps(agent, 3)
        agent.apply_summary("Plain text summary without tags")
        # Should still apply — uses the full text as summary body.
        assert "Plain text summary without tags" in agent._messages[1]["content"]
        assert len(agent._messages) == 2

    def test_increments_counter(self):
        agent = _make_agent()
        _simulate_steps(agent, 3)
        assert agent._summarization_count == 0
        agent.apply_summary("<context_summary>s1</context_summary>")
        assert agent._summarization_count == 1
        # Add more messages and summarize again.
        _simulate_steps(agent, 3)
        agent.apply_summary("<context_summary>s2</context_summary>")
        assert agent._summarization_count == 2

    def test_records_events(self):
        agent = _make_agent()
        _simulate_steps(agent, 3)
        agent.apply_summary("<context_summary>s1</context_summary>")
        events = agent._summarization_events
        assert len(events) == 1
        assert events[0]["summarization_index"] == 1
        assert events[0]["pre_message_count"] > 0
        assert events[0]["post_message_count"] == 2

    def test_continuing_task_marker(self):
        """Summary message should end with [Continuing task]."""
        agent = _make_agent()
        _simulate_steps(agent, 3)
        agent.apply_summary("<context_summary>test</context_summary>")
        assert "[Continuing task]" in agent._messages[1]["content"]

    def test_summary_without_system_prompt(self):
        """Handle edge case where there's no system message."""
        agent = _make_agent()
        # Remove system message.
        agent._messages = [
            {"role": "user", "content": "obs1"},
            {"role": "assistant", "content": "act1"},
            {"role": "user", "content": "obs2"},
        ]
        agent.apply_summary("<context_summary>summary</context_summary>")
        assert len(agent._messages) == 1  # only summary, no system
        assert "<context_summary>" in agent._messages[0]["content"]


# ---------------------------------------------------------------------------
# Tests: get_summarization_metadata
# ---------------------------------------------------------------------------

class TestGetSummarizationMetadata:
    def test_initial_metadata(self):
        agent = _make_agent()
        meta = agent.get_summarization_metadata()
        assert meta["summarization_count"] == 0
        assert meta["summarization_events"] == []

    def test_after_summarization(self):
        agent = _make_agent()
        _simulate_steps(agent, 3)
        agent.apply_summary("<context_summary>x</context_summary>")
        meta = agent.get_summarization_metadata()
        assert meta["summarization_count"] == 1
        assert len(meta["summarization_events"]) == 1


# ---------------------------------------------------------------------------
# Tests: composed classes
# ---------------------------------------------------------------------------

class TestComposedClasses:
    def test_gem_text_agent_with_summarization(self):
        agent = GEMTextAgentWithSummarization(
            system_prompt=SYSTEM_PROMPT,
            summarization_threshold_tokens=100,
        )
        _simulate_steps(agent, 3)
        # Should behave like GEMTextAgent — full response in history.
        assistant_msgs = [m for m in agent._messages if m["role"] == "assistant"]
        assert all("Thinking about" in m["content"] for m in assistant_msgs)

    def test_gem_noncumulative_with_summarization(self):
        agent = GEMTextAgentNonCumulativeWithSummarization(
            system_prompt=SYSTEM_PROMPT,
            summarization_threshold_tokens=100,
        )
        _simulate_steps(agent, 3)
        # Should behave like NonCumulative — only boxed action in history.
        assistant_msgs = [m for m in agent._messages if m["role"] == "assistant"]
        assert all("\\boxed{" in m["content"] for m in assistant_msgs)
        assert all("Thinking about" not in m["content"] for m in assistant_msgs)

    def test_both_support_should_summarize(self):
        for cls in (
            GEMTextAgentWithSummarization,
            GEMTextAgentNonCumulativeWithSummarization,
        ):
            agent = cls(
                system_prompt=SYSTEM_PROMPT,
                summarization_threshold_tokens=10,
            )
            _simulate_steps(agent, 3)
            tokenizer, parser = _make_tokenizer_and_parser()
            # With threshold=10, should_summarize should be True after 3 steps.
            assert agent.should_summarize(tokenizer, parser) is True


# ---------------------------------------------------------------------------
# Tests: summarization modes (token | episodic | both)
# ---------------------------------------------------------------------------

class TestSummarizationMode:
    def test_default_mode_is_token(self):
        agent = _make_agent()
        assert agent.summarization_mode == "token"

    def test_mode_set_via_kwarg(self):
        agent = _make_agent(mode="episodic")
        assert agent.summarization_mode == "episodic"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            _make_agent(mode="not-a-mode")

    def test_episodic_mode_disables_token_path(self):
        """With mode='episodic', should_summarize (token path) must return False
        even when the prompt exceeds the token threshold."""
        agent = _make_agent(threshold=10, mode="episodic")
        _simulate_steps(agent, 3)
        tokenizer, parser = _make_tokenizer_and_parser()
        assert agent.should_summarize(tokenizer, parser) is False

    def test_token_mode_disables_episode_end_path(self):
        agent = _make_agent(mode="token")
        _simulate_steps(agent, 3)
        assert agent.should_summarize_on_episode_end({"episode_done": True}) is False

    def test_episode_end_trigger_under_episodic_and_both(self):
        for mode in ("episodic", "both"):
            agent = _make_agent(mode=mode)
            _simulate_steps(agent, 3)
            assert (
                agent.should_summarize_on_episode_end({"episode_done": True}) is True
            ), f"mode={mode}"
            # No episode_done → still False
            assert (
                agent.should_summarize_on_episode_end({"episode_done": False}) is False
            )

    def test_both_mode_enables_both_paths(self):
        agent = _make_agent(threshold=10, mode="both")
        _simulate_steps(agent, 3)
        tokenizer, parser = _make_tokenizer_and_parser()
        assert agent.should_summarize(tokenizer, parser) is True
        assert agent.should_summarize_on_episode_end({"episode_done": True}) is True

    def test_episode_end_requires_enough_messages(self):
        agent = _make_agent(mode="episodic")
        # Only system prompt — not enough.
        assert agent.should_summarize_on_episode_end({"episode_done": True}) is False

    def test_episode_end_respects_max_summarizations(self):
        agent = _make_agent(mode="episodic", max_summarizations=0)
        _simulate_steps(agent, 3)
        assert agent.should_summarize_on_episode_end({"episode_done": True}) is False


# ---------------------------------------------------------------------------
# Tests: build_summarization_prompt with trigger / reflective switch
# ---------------------------------------------------------------------------

class TestTriggerAwarePrompt:
    def test_default_uses_context_summary_prompt(self):
        from prompts.summarization_prompts import CONTEXT_SUMMARY_PROMPT

        agent = _make_agent()
        _simulate_steps(agent, 2)
        prompt = agent.build_summarization_prompt(trigger="token")
        assert prompt[-1]["content"] == CONTEXT_SUMMARY_PROMPT

    def test_episode_end_non_reflective_uses_context_summary(self):
        from prompts.summarization_prompts import CONTEXT_SUMMARY_PROMPT

        agent = _make_agent()
        _simulate_steps(agent, 2)
        prompt = agent.build_summarization_prompt(
            trigger="episode_end", use_reflective_prompt=False
        )
        assert prompt[-1]["content"] == CONTEXT_SUMMARY_PROMPT

    def test_reflective_prompt_selected_when_requested(self):
        from prompts.summarization_prompts import REFLECTIVE_SUMMARY_PROMPT

        agent = _make_agent()
        _simulate_steps(agent, 2)
        prompt = agent.build_summarization_prompt(
            trigger="episode_end", use_reflective_prompt=True
        )
        assert prompt[-1]["content"] == REFLECTIVE_SUMMARY_PROMPT


# ---------------------------------------------------------------------------
# Tests: apply_summary post-layout (always [sys, summary])
# ---------------------------------------------------------------------------

class TestApplySummaryLayout:
    def test_layout_after_trailing_user_msg(self):
        """Regardless of whether history ends with user or assistant,
        apply_summary collapses to exactly [system, summary_user_msg]."""
        agent = _make_agent()
        _simulate_steps(agent, 3)
        # End the history on a user message (simulate post-update_from_env state).
        agent.update_from_env(
            observation="latest env obs",
            reward=0.0,
            done=False,
            info={},
        )
        assert agent._messages[-1]["role"] == "user"

        agent.apply_summary("<context_summary>stuff</context_summary>")

        assert len(agent._messages) == 2
        assert agent._messages[0]["role"] == "system"
        assert "<context_summary>" in agent._messages[1]["content"]

    def test_layout_after_trailing_assistant_msg(self):
        agent = _make_agent()
        _simulate_steps(agent, 3)
        # _simulate_steps ends with update_from_model, so last msg is assistant.
        assert agent._messages[-1]["role"] == "assistant"
        agent.apply_summary("<context_summary>stuff</context_summary>")
        assert len(agent._messages) == 2

    def test_trigger_metadata_token(self):
        agent = _make_agent()
        _simulate_steps(agent, 3)
        agent.update_from_env(observation="obs", reward=0.0, done=False, info={})
        agent.apply_summary(
            "<context_summary>s</context_summary>", trigger="token"
        )
        event = agent._summarization_events[-1]
        assert event["trigger"] == "token"

    def test_trigger_metadata_episode_end(self):
        agent = _make_agent()
        _simulate_steps(agent, 3)
        agent.apply_summary(
            "<context_summary>s</context_summary>", trigger="episode_end"
        )
        event = agent._summarization_events[-1]
        assert event["trigger"] == "episode_end"
