"""Prompt templates for mid-trajectory context summarization.

Three variants, picked by the engine + agent based on (trigger,
env.enable_reflection):

* ``TOKEN_SUMMARY_PROMPT`` — token trigger, fires mid-episode. The
  episode is still in progress; the prompt asks the agent to compress
  prior context and preserve current-episode progress so the agent can
  continue.
* ``EPISODIC_SUMMARY_PROMPT`` — episodic trigger, env.enable_reflection
  is False. An episode has just ended; compression-only, no reflection
  language.
* ``REFLECTIVE_SUMMARY_PROMPT`` — episodic trigger, env.enable_reflection
  is True. Episode end + reflection in a single ask.

All three must instruct the model to wrap its output in
``<context_summary>...</context_summary>`` tags so
``agents.context_summarizer._SUMMARY_TAG_RE`` can extract it.
"""

TOKEN_SUMMARY_PROMPT = """\
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


EPISODIC_SUMMARY_PROMPT = """\
You have just completed one episode of interaction with the environment.

Produce a compact memory the agent can carry into the next episode.

Cover:
1. The task description and rules.
2. Confirmed facts about the environment, task, or hidden state — including the outcome of the episode that just ended.
3. Ruled-out states, actions, or hypotheses based on observed feedback.
4. A concrete plan or belief to seed the next episode.

Do NOT include:
- Verbatim copies of past observations or actions.
- Step-by-step reasoning traces.
- Redundant information.

Write your summary inside <context_summary>...</context_summary> tags."""


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
