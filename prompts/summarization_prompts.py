"""Prompt templates for mid-trajectory context summarization.

Three variants, picked by the engine + agent based on (trigger,
env.enable_reflection):

* ``TOKEN_SUMMARY_PROMPT`` — token trigger, fires mid-episode.
* ``EPISODIC_SUMMARY_PROMPT`` — episodic trigger, env.enable_reflection
  False. Compression at episode end without reflection language.
* ``REFLECTIVE_SUMMARY_PROMPT`` — episodic trigger, env.enable_reflection
  True. Episode end + reflection in one ask.

Style: each prompt gives the model agency to decide WHAT to preserve
("you decide what to include based on what will most help your
downstream decisions"). The bullet lists are framed as suggestions —
"possible things worth preserving" — rather than a fixed schema, so
during RL training the policy can learn task-specific compression
strategies. The hard constraint is the output format (``<context_summary>``
tags) and the consequence framing ("the summary REPLACES your prior
history"), both of which the engine relies on.

All three must instruct the model to wrap its output in
``<context_summary>...</context_summary>`` tags so
``agents.context_summarizer._SUMMARY_TAG_RE`` can extract it.
"""

TOKEN_SUMMARY_PROMPT = """\
Your context window is filling up while you are mid-task. Compress what you have seen so far into a compact memory that you will continue from.

You decide what to include based on what will most help your downstream decisions. The summary REPLACES your prior history - whatever you don't write here, you won't have access to later. Aim for what's actually useful to act on, not a transcript.

Possible things worth preserving:
- The task description, goal, and rules.
- Confirmed facts about the environment or hidden state.
- Hypotheses you've ruled out.
- Current episode progress: where you are, what you were doing, what you intended to try next.
- Patterns or insights that have been guiding your decisions.

Write your summary inside <context_summary>...</context_summary> tags."""


EPISODIC_SUMMARY_PROMPT = """\
You have just completed one episode. Compress your experience into a compact memory that the agent will start the next episode from.

You decide what to include based on what will most help your future decisions. The summary REPLACES your prior history - whatever you don't write here, you won't have access to in later episodes. Aim for what's actually useful to act on, not a transcript.

Possible things worth preserving:
- The task description, goal, and rules.
- Confirmed facts about the environment or hidden state, including this episode's outcome.
- States, actions, or hypotheses ruled out by what you observed.
- A concrete plan or belief to seed the next episode.

Write your summary inside <context_summary>...</context_summary> tags."""


REFLECTIVE_SUMMARY_PROMPT = """\
You have just completed one episode. Reflect on it and compress your experience into a compact memory that the agent will start the next episode from.

You decide what to include based on what will most help your future decisions. Reflection on what worked, what failed, and what to try next is welcome — but ultimately you choose what's worth keeping. The summary REPLACES your prior history - whatever you don't write here, you won't have access to in later episodes. Aim for what's actually useful to act on, not a transcript.

Possible things worth preserving:
- The task description, goal, and rules.
- This episode's outcome and what was decisive about it.
- What worked vs. what failed, and what you'd do differently.
- Confirmed facts about the environment or hidden state.
- Ruled-out hypotheses or dead-end strategies.
- A concrete plan or belief to seed the next episode.

Write your summary inside <context_summary>...</context_summary> tags."""
