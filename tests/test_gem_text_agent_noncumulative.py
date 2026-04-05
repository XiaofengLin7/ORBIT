"""Tests for GEMTextAgentNonCumulative — verifies non-cumulative context behaviour.

Simulates multi-step interactions across WebShop-like, GEM-like and ALFWorld-like
trajectories and checks that:
  1. Only action tokens appear in ``chat_completions`` history (no thinking).
  2. The full model response (with thinking) is preserved in ``Step.model_response``.
  3. Action parsing works correctly.
  4. History stays compact across many steps.
"""

import pytest
from agents.gem_text_agent_noncumulative import GEMTextAgentNonCumulative, extract_last_boxed
from rllm.agents.agent import Action  # type: ignore


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _assistant_contents(agent) -> list[str]:
    """Return the content of all assistant messages in chat_completions."""
    return [m["content"] for m in agent.chat_completions if m["role"] == "assistant"]


# ---------------------------------------------------------------------------
# Basic unit tests
# ---------------------------------------------------------------------------

class TestExtractLastBoxed:
    def test_single(self):
        assert extract_last_boxed("try \\boxed{5}") == "5"

    def test_multiple_picks_last(self):
        assert extract_last_boxed("foo \\boxed{3} bar \\boxed{7}") == "7"

    def test_no_box_returns_stripped(self):
        assert extract_last_boxed("  no boxes here  ") == "no boxes here"


class TestUpdateFlow:
    def test_basic_step(self):
        agent = GEMTextAgentNonCumulative(system_prompt="sys", max_steps=5)
        agent.update_from_env("hello", reward=0.0, done=False, info={})
        action = agent.update_from_model("try \\boxed{5}")

        assert isinstance(action, Action)
        assert action.action == "\\boxed{5}"

        state = agent.get_current_state()
        assert state.action.action == "\\boxed{5}"
        assert state.observation == "hello"

    def test_only_action_in_history(self):
        """Assistant messages must contain only the boxed action, not thinking."""
        agent = GEMTextAgentNonCumulative()
        agent.update_from_env("obs0", reward=0.0, done=False, info={})
        agent.update_from_model("<think>long reasoning here</think>\\boxed{42}")

        contents = _assistant_contents(agent)
        assert len(contents) == 1
        assert contents[0] == "\\boxed{42}"
        assert "long reasoning" not in contents[0]

    def test_full_response_preserved_in_step(self):
        """model_response on the Step must retain the full thinking + action."""
        agent = GEMTextAgentNonCumulative()
        agent.update_from_env("obs0", reward=0.0, done=False, info={})
        full_resp = "<think>deep thought</think>\\boxed{99}"
        agent.update_from_model(full_resp)

        step = agent.get_current_state()
        assert step.model_response == full_resp


# ---------------------------------------------------------------------------
# Multi-step trajectory simulations
# ---------------------------------------------------------------------------

class TestGEMTrajectory:
    """Simulate a GEM (GuessTheNumber-style) multi-step trajectory."""

    def test_three_step_gem(self):
        agent = GEMTextAgentNonCumulative(system_prompt="Guess the number.")

        # Step 0
        agent.update_from_env("Guess a number between 1 and 100.", reward=0.0, done=False, info={})
        agent.update_from_model("<think>I'll start with 50</think>\\boxed{50}")

        # Step 1
        agent.update_from_env("Too low.", reward=0.0, done=False, info={})
        agent.update_from_model("<think>Try 75</think>\\boxed{75}")

        # Step 2
        agent.update_from_env("Correct!", reward=1.0, done=True, info={})
        agent.update_from_model("<think>I already know the answer</think>\\boxed{75}")

        # Verify chat_completions has action-only assistant messages
        contents = _assistant_contents(agent)
        assert contents == ["\\boxed{50}", "\\boxed{75}", "\\boxed{75}"]
        assert all("<think>" not in c for c in contents)

        # Verify full responses preserved
        steps = agent.trajectory.steps
        assert "<think>I'll start with 50</think>" in steps[0].model_response
        assert "<think>Try 75</think>" in steps[1].model_response

        # Verify message count stays compact: system + 3*(user+assistant) = 7
        assert len(agent.chat_completions) == 7


class TestWebShopTrajectory:
    """Simulate a WebShop-like multi-step trajectory."""

    def test_webshop_search_and_buy(self):
        agent = GEMTextAgentNonCumulative(
            system_prompt="You are shopping. Put your action in \\boxed{}."
        )

        # Step 0: search
        agent.update_from_env(
            "Find a red t-shirt under $30.",
            reward=0.0, done=False, info={}
        )
        agent.update_from_model(
            "<think>I need to search for red t-shirts</think>"
            "\\boxed{search[red t-shirt under $30]}"
        )

        # Step 1: click product
        agent.update_from_env(
            "[Product 1] Red Cotton T-Shirt $25\n[Product 2] Red Silk Shirt $45",
            reward=0.0, done=False, info={}
        )
        agent.update_from_model(
            "<think>Product 1 matches — cotton, under $30</think>"
            "\\boxed{click[Product 1]}"
        )

        # Step 2: buy
        agent.update_from_env(
            "Red Cotton T-Shirt $25. Size: S/M/L.",
            reward=0.0, done=False, info={}
        )
        agent.update_from_model(
            "<think>Pick medium size and buy</think>"
            "\\boxed{click[Buy Now]}"
        )

        # Step 3: terminal
        agent.update_from_env(
            "Congratulations! Your score: 1.00 / 1.00.",
            reward=1.0, done=True, info={}
        )

        # History should only have compact actions
        contents = _assistant_contents(agent)
        assert contents == [
            "\\boxed{search[red t-shirt under $30]}",
            "\\boxed{click[Product 1]}",
            "\\boxed{click[Buy Now]}",
        ]
        assert all("<think>" not in c for c in contents)
        assert all("cotton" not in c.lower() for c in contents)

        # All three full responses preserved
        steps = agent.trajectory.steps
        assert "cotton, under $30" in steps[1].model_response


class TestALFWorldTrajectory:
    """Simulate an ALFWorld-like multi-step trajectory."""

    def test_alfworld_put_task(self):
        agent = GEMTextAgentNonCumulative(
            system_prompt="Complete the household task. Put your action in \\boxed{}."
        )

        # Step 0
        agent.update_from_env(
            "You are in the kitchen. Task: put a clean mug in the cabinet.",
            reward=0.0, done=False, info={}
        )
        agent.update_from_model(
            "<think>First I need to find a mug. Let me look around.</think>"
            "\\boxed{go to countertop 1}"
        )

        # Step 1
        agent.update_from_env(
            "On the countertop 1, you see a mug 1 and a plate 2.",
            reward=0.0, done=False, info={}
        )
        agent.update_from_model(
            "<think>Found mug 1. I should pick it up.</think>"
            "\\boxed{take mug 1 from countertop 1}"
        )

        # Step 2
        agent.update_from_env(
            "You pick up the mug 1 from the countertop 1.",
            reward=0.0, done=False, info={}
        )
        agent.update_from_model(
            "<think>Now clean it at the sink.</think>"
            "\\boxed{go to sinkbasin 1}"
        )

        # Step 3
        agent.update_from_env(
            "You arrive at the sinkbasin 1.",
            reward=0.0, done=False, info={}
        )
        agent.update_from_model(
            "<think>Clean the mug.</think>"
            "\\boxed{clean mug 1 with sinkbasin 1}"
        )

        # Step 4
        agent.update_from_env(
            "You clean the mug 1 using the sinkbasin 1.",
            reward=0.0, done=False, info={}
        )
        agent.update_from_model(
            "<think>Now put in cabinet.</think>"
            "\\boxed{go to cabinet 1}"
        )

        # Step 5
        agent.update_from_env(
            "You arrive at the cabinet 1. The cabinet 1 is open.",
            reward=0.0, done=False, info={}
        )
        agent.update_from_model(
            "<think>Put the mug in.</think>"
            "\\boxed{put mug 1 in/on cabinet 1}"
        )

        # Step 6: terminal
        agent.update_from_env(
            "Congratulations! You successfully completed the task.",
            reward=1.0, done=True, info={}
        )

        # Only actions in history
        contents = _assistant_contents(agent)
        expected_actions = [
            "\\boxed{go to countertop 1}",
            "\\boxed{take mug 1 from countertop 1}",
            "\\boxed{go to sinkbasin 1}",
            "\\boxed{clean mug 1 with sinkbasin 1}",
            "\\boxed{go to cabinet 1}",
            "\\boxed{put mug 1 in/on cabinet 1}",
        ]
        assert contents == expected_actions
        assert all("<think>" not in c for c in contents)

        # Full responses preserved in steps
        assert "Found mug 1" in agent.trajectory.steps[1].model_response

        # History compact: system + 7 user + 6 assistant = 14
        assert len(agent.chat_completions) == 14


class TestMultiEpisodeTrajectory:
    """Simulate two consecutive episodes (reset in between)."""

    def test_two_episodes(self):
        agent = GEMTextAgentNonCumulative(system_prompt="Guess the number.")

        # --- Episode 1 ---
        agent.update_from_env("Guess between 1 and 10.", reward=0.0, done=False, info={})
        agent.update_from_model("<think>try 5</think>\\boxed{5}")
        agent.update_from_env("Too high.", reward=0.0, done=False, info={})
        agent.update_from_model("<think>try 3</think>\\boxed{3}")
        agent.update_from_env("Correct!", reward=1.0, done=True, info={})

        ep1_completions = agent.chat_completions
        ep1_assistant_count = sum(1 for m in ep1_completions if m["role"] == "assistant")
        assert ep1_assistant_count == 2
        assert all("<think>" not in m["content"] for m in ep1_completions if m["role"] == "assistant")

        # --- Episode 2 (agent reset) ---
        agent.reset()
        agent.update_from_env("Guess between 1 and 5.", reward=0.0, done=False, info={})
        agent.update_from_model("<think>try 2</think>\\boxed{2}")
        agent.update_from_env("Correct!", reward=1.0, done=True, info={})

        ep2_completions = agent.chat_completions
        # After reset: system + 1 user + 1 assistant + 1 user = 4
        assert len(ep2_completions) == 4
        assert _assistant_contents(agent) == ["\\boxed{2}"]


class TestNoSystemPrompt:
    """Agent without system_prompt works (for envs that provide instructions as observations)."""

    def test_no_system_message(self):
        agent = GEMTextAgentNonCumulative()  # no system_prompt
        agent.update_from_env("You are an agent. Do the task.", reward=0.0, done=False, info={})
        agent.update_from_model("<think>ok</think>\\boxed{action1}")

        msgs = agent.chat_completions
        assert msgs[0]["role"] == "user"  # no system message
        assert len(msgs) == 2  # user + assistant

    def test_with_system_prompt(self):
        agent = GEMTextAgentNonCumulative(system_prompt="You are helpful.")
        agent.update_from_env("obs", reward=0.0, done=False, info={})

        msgs = agent.chat_completions
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are helpful."
        assert len(msgs) == 2  # system + user


class TestNoThinkTag:
    """Responses without <think> tags should work normally."""

    def test_plain_response(self):
        agent = GEMTextAgentNonCumulative()
        agent.update_from_env("obs", reward=0.0, done=False, info={})
        agent.update_from_model("The answer is \\boxed{7}")

        contents = _assistant_contents(agent)
        assert contents == ["\\boxed{7}"]
        assert agent.trajectory.steps[0].model_response == "The answer is \\boxed{7}"

    def test_no_boxed_at_all(self):
        agent = GEMTextAgentNonCumulative()
        agent.update_from_env("obs", reward=0.0, done=False, info={})
        agent.update_from_model("I don't know")

        # Falls back to raw text wrapped in boxed
        contents = _assistant_contents(agent)
        assert contents == ["\\boxed{I don't know}"]


class TestContextCompactness:
    """Verify that context size grows linearly with observations, not with thinking."""

    def test_compact_growth(self):
        agent = GEMTextAgentNonCumulative(system_prompt="sys")
        long_thinking = "<think>" + "x" * 1000 + "</think>"

        for i in range(10):
            agent.update_from_env(f"obs{i}", reward=0.0, done=False, info={})
            agent.update_from_model(f"{long_thinking}\\boxed{{{i}}}")

        # Total text in chat_completions should NOT contain 10*1000 thinking chars
        total_text = "".join(m["content"] for m in agent.chat_completions)
        assert "xxxx" not in total_text
        # Should be compact: system + 10*(user+assistant)
        assert len(agent.chat_completions) == 21
