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
