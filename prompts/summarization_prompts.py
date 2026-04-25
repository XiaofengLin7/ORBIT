"""Prompt templates for mid-trajectory context summarization."""

CONTEXT_SUMMARY_PROMPT = """\
You are summarizing the interaction history of an agent solving a task across multiple episodes.

Produce a concise summary that preserves:
1. The task description and rules
2. Key information discovered across episodes (confirmed facts, ruled-out possibilities)
3. Current episode progress: which episode you are in, all observations in this episode, what actions have been taken, and what you were about to do next
4. Strategy insights that should inform future actions

Do NOT include:
- Verbatim copies of previous observations or actions from earlier episodes
- Step-by-step reasoning traces
- Redundant information

Write your summary inside <context_summary>...</context_summary> tags."""


# Prompt used for episodic summarization when env.enable_reflection=True.
# The user may customize this freely; the only hard requirement is that the
# model's summary body must be wrapped in <context_summary>...</context_summary>
# tags so agents.context_summarizer._SUMMARY_TAG_RE can extract it.
REFLECTIVE_SUMMARY_PROMPT = """\
You have just completed one episode of interaction with the environment.

Reflect on this episode and distill a compressed memory to carry into the next episode.

Cover:
1. Episode outcome (success/failure) and what was decisive about it.
2. What worked, what failed, and what you would do differently.
3. Confirmed facts about the environment, task, or hidden state.
4. Ruled-out hypotheses or dead-end strategies.
5. A concrete plan or belief to seed the next episode.

Do NOT restate verbatim observations or step-by-step reasoning.
Write your response inside <context_summary>...</context_summary> tags."""
