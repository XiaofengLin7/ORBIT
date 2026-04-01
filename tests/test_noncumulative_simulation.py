"""Simulate a full WebShop multi-episode trajectory with GEMTextAgentNonCumulative.

Traces what the model sees at each step to verify thinking tokens are discarded.
"""

from agents.gem_text_agent_noncumulative import GEMTextAgentNonCumulative


SYSTEM_PROMPT = (
    "You are solving the same task across multiple episodes with a fixed total step budget. "
    "Each episode resets the environment but keeps the task identical. "
    "Leverage information gathered from earlier episodes to succeed faster. "
    "Think briefly and respond with actions inside \\boxed{} each turn. "
    "Overlong responses will be penalized."
)

# Simulated model responses (thinking + action)
MODEL_RESPONSES = [
    # Episode 1, Step 0: search
    "<think>\nOkay, let me try to figure out what to do here. The user wants men's t-shirts "
    "and tanks with specific features: short sleeve, fashion design, long sleeve, button "
    "closure, color e-white, size large, and price under $40. Let me search for these.\n"
    "</think>\n\n\\boxed{search[men's t-shirts tanks short sleeve button closure e-white]}",

    # Episode 1, Step 1: click next
    "<think>\nLooking at the results on page 1, none explicitly mention e-white. "
    "I should check the next page for more options. The items so far don't match all "
    "criteria perfectly.\n</think>\n\n\\boxed{click[next >]}",

    # Episode 1, Step 2: click product
    "<think>\nOn page 2, B085WQKRRJ looks promising - it's a Henley T-Shirt with button "
    "closure, short sleeve, under $40. Let me check its details.\n"
    "</think>\n\n\\boxed{click[b085wqkrrj]}",

    # Episode 1, Step 3: select color
    "<think>\nThe product page shows color options including white. Let me select white "
    "which should correspond to e-white.\n</think>\n\n\\boxed{click[white]}",

    # Episode 1, Step 4: select size
    "<think>\nNow select size large.\n</think>\n\n\\boxed{click[large]}",

    # Episode 1, Step 5: buy
    "<think>\nColor and size are set. Time to buy.\n</think>\n\n\\boxed{click[buy now]}",
]

# Simulated environment observations
ENV_OBSERVATIONS = [
    # Step 0: initial obs (from env reset, includes task description)
    (
        "[Episode 1] New episode begins.\n\n"
        "You are an expert autonomous agent operating in the WebShop environment.\n"
        "Your task is to: Find me men's t-shirts & tanks with short sleeve, fashion design, "
        "long sleeve, button closure with color: e-white, and size: large, and price lower "
        "than 40.00 dollars.\n"
        "Search\nYour admissible actions: ['search[<your query>]']"
    ),
    # Step 1: search results
    (
        "Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] "
        "B09QQP3356 [SEP] HAUKLIE Men's Sports Polo Shirts $10.99 [SEP] "
        "B085WQKRRJ [SEP] CLANMILUMS Men's Henley T-Shirt $17.99\n"
        "Your admissible actions: ['click[back to search]', 'click[next >]', 'click[b085wqkrrj]']"
    ),
    # Step 2: page 2
    (
        "Back to Search [SEP] Page 2 [SEP] "
        "B085WQKRRJ [SEP] CLANMILUMS Men's Henley T-Shirt $17.99\n"
        "Your admissible actions: ['click[b085wqkrrj]']"
    ),
    # Step 3: product page
    (
        "size [SEP] small [SEP] large [SEP] color [SEP] black [SEP] white [SEP] "
        "CLANMILUMS Men's Short Sleeve Henley T-Shirt [SEP] Price: $17.99\n"
        "Your admissible actions: ['click[white]', 'click[large]', 'click[buy now]']"
    ),
    # Step 4: color selected
    (
        "size [SEP] small [SEP] large [SEP] color [SEP] white (selected) [SEP] "
        "CLANMILUMS Men's Short Sleeve Henley T-Shirt [SEP] Price: $17.99\n"
        "Your admissible actions: ['click[large]', 'click[buy now]']"
    ),
    # Step 5: size selected
    (
        "size [SEP] large (selected) [SEP] color [SEP] white (selected) [SEP] "
        "CLANMILUMS Men's Short Sleeve Henley T-Shirt [SEP] Price: $17.99\n"
        "Your admissible actions: ['click[buy now]']"
    ),
]


def test_webshop_simulation():
    """Simulate WebShop trajectory and verify no thinking tokens in prompts."""
    agent = GEMTextAgentNonCumulative(system_prompt=SYSTEM_PROMPT)

    # Initial env reset observation
    agent.update_from_env(ENV_OBSERVATIONS[0], reward=0.0, done=False, info={})

    for step_idx in range(len(MODEL_RESPONSES)):
        # --- What the model sees (prompt) ---
        prompt_messages = agent.chat_completions

        # CRITICAL CHECK: no thinking tokens in any assistant message
        for msg in prompt_messages:
            if msg["role"] == "assistant":
                assert "<think>" not in msg["content"], (
                    f"Step {step_idx}: thinking tokens found in prompt!\n"
                    f"  assistant content: {msg['content'][:100]}..."
                )
                assert "</think>" not in msg["content"], (
                    f"Step {step_idx}: thinking tokens found in prompt!\n"
                    f"  assistant content: {msg['content'][:100]}..."
                )
                # Should ONLY contain \boxed{...}
                assert msg["content"].startswith("\\boxed{"), (
                    f"Step {step_idx}: assistant message not action-only!\n"
                    f"  content: {msg['content'][:100]}..."
                )

        # --- Model generates response (thinking + action) ---
        response = MODEL_RESPONSES[step_idx]
        action = agent.update_from_model(response)

        # Full response preserved in trajectory step
        step = agent.get_current_state()
        assert "<think>" in step.model_response, (
            f"Step {step_idx}: full response should be preserved in model_response"
        )

        # --- Env returns next observation ---
        done = (step_idx == len(MODEL_RESPONSES) - 1)
        reward = 1.0 if done else 0.0
        if step_idx + 1 < len(ENV_OBSERVATIONS):
            agent.update_from_env(
                ENV_OBSERVATIONS[step_idx + 1],
                reward=reward,
                done=done,
                info={},
            )

    # Final check: print what the model would see at each step
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE - Message history at end:")
    print("=" * 70)
    for i, msg in enumerate(agent.chat_completions):
        role = msg["role"]
        content = msg["content"]
        if len(content) > 80:
            content = content[:77] + "..."
        print(f"  [{i}] {role:>10}: {content}")

    # Count: system + 6*(user+assistant) + 1 final user = 14
    msgs = agent.chat_completions
    assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
    print(f"\n  Total messages: {len(msgs)}")
    print(f"  Assistant messages: {len(assistant_msgs)}")
    print(f"  All assistant messages are action-only: "
          f"{all(m['content'].startswith(chr(92) + 'boxed{') for m in assistant_msgs)}")

    total_chars = sum(len(m["content"]) for m in msgs)
    thinking_chars = sum(len(r) for r in MODEL_RESPONSES)
    print(f"\n  Total chars in history: {total_chars}")
    print(f"  Total chars if thinking kept: {thinking_chars + total_chars}")
    print(f"  Context savings: ~{thinking_chars} chars of thinking discarded")


def test_episode_boundary():
    """Verify thinking tokens are discarded across episode boundaries."""
    agent = GEMTextAgentNonCumulative(system_prompt=SYSTEM_PROMPT)

    # Episode 1
    agent.update_from_env("[Episode 1] Task: find item X.", reward=0.0, done=False, info={})
    agent.update_from_model("<think>Long reasoning about episode 1 step 0...</think>\\boxed{search[item X]}")
    agent.update_from_env("Results: item A, item B", reward=0.0, done=False, info={})
    agent.update_from_model("<think>Another long reasoning...</think>\\boxed{click[item A]}")

    # Episode boundary: env sends terminal + new episode obs combined
    agent.update_from_env(
        "Purchase complete! Score: 0.5\n\n[Episode 2] New episode begins.\nTask: find item X.",
        reward=0.5, done=False, info={},
    )

    # Episode 2 step 0 - model should see compact history
    prompt = agent.chat_completions
    print("\n" + "=" * 70)
    print("EPISODE BOUNDARY - Prompt at Episode 2, Step 0:")
    print("=" * 70)
    for i, msg in enumerate(prompt):
        role = msg["role"]
        content = msg["content"]
        if len(content) > 100:
            content = content[:97] + "..."
        print(f"  [{i}] {role:>10}: {content}")

    # No thinking in any assistant message
    for msg in prompt:
        if msg["role"] == "assistant":
            assert "<think>" not in msg["content"]
            assert msg["content"].startswith("\\boxed{")

    # Agent can continue with episode 2
    agent.update_from_model("<think>Episode 2 reasoning with prior knowledge...</think>\\boxed{search[item X exact]}")

    prompt2 = agent.chat_completions
    assistant_msgs = [m for m in prompt2 if m["role"] == "assistant"]
    print(f"\n  Assistant messages across both episodes: {len(assistant_msgs)}")
    assert all("<think>" not in m["content"] for m in assistant_msgs)
    print("  All action-only: YES")


if __name__ == "__main__":
    test_webshop_simulation()
    test_episode_boundary()
    print("\n\nAll checks passed.")
