"""System prompts for multi-episode training and evaluation.

The base prompt tells the policy it is doing multi-episode (or single-episode)
in-context learning. When summarization is enabled, an extra "Memory protocol"
section is appended that documents the carryover format the engine actually
uses:

* ``<context_summary>`` / ``<reflection>`` blocks are the model's own prior
  notes, fed back as a ``user`` message — not new instructions from the env.
* Prior-episode assistant turns in ``obs_action`` / ``obs_action_reflection``
  carryover are stripped to bare ``\\boxed{action}``: a compression artifact,
  NOT a format change. The policy must keep thinking briefly before each new
  action.
* ``[Continuing task]`` is the boundary marker between a compressed block and
  the next observation.

Without this section the model has to reverse-engineer the protocol from RL
signal alone; with it, sample efficiency improves and zero-shot eval no longer
silently differs from the training contract.
"""

from __future__ import annotations

_BASE_MULTI_EPISODE = (
    "You are solving the same task across {num_episodes} episodes with a fixed total step budget. "
    "Each episode resets the environment but keeps the task identical. "
    "You will interact with the environment for exactly {num_episodes} episodes - "
    "use earlier episodes to gather information so later episodes succeed faster. "
    "Respond with your action inside \\boxed{} each turn."
)

_BASE_SINGLE_EPISODE = (
    "You are solving a task in a single episode. "
    "Analyze the situation carefully and take the best actions to succeed. "
    "Respond with your action inside \\boxed{} each turn."
)

_PROTOCOL_HEADER = (
    "\n\nMemory protocol. Your conversation history may be compressed within "
    "or between episodes. When that happens, the compressed form appears as a "
    "USER message containing `<context_summary>...</context_summary>` or "
    "`<reflection>...</reflection>` tags — that content is YOUR OWN prior "
    "memory carried forward, NOT a new instruction from the environment. Each "
    "compressed block is labeled with `[End of episode K/N]` (or "
    "`[Mid-episode K/N compression]` for within-episode compression), where K "
    "is the 1-based index of the just-finished (or currently-running) episode "
    "and N is the total number of episodes — N is dropped when unknown. The "
    "marker `[Continuing task]` follows the compressed block; the next env "
    "observation arrives as the next user message."
)

_PROTOCOL_TOKEN = (
    "\n- Mid-episode (token threshold): when your context grows large you will "
    "be asked to write a `<context_summary>` block. That block, prefixed with "
    "`[Mid-episode K/N compression]`, replaces your history and you continue "
    "the same episode from it."
)

_PROTOCOL_FREEFORM = (
    "\n- Between episodes: you will be asked to write a `<context_summary>` "
    "block. The next episode starts with that block, prefixed with "
    "`[End of episode K/N]`, as your only memory of earlier episodes. Make it "
    "useful to your future self — preserve task rules, confirmed facts, "
    "ruled-out hypotheses, and a concrete plan."
)

_PROTOCOL_OBS_ACTION = (
    "\n- Between episodes: your prior history is reduced to the previous "
    "episode's observations paired with `\\boxed{action}` only; the thinking "
    "around each action is dropped to save context. A trailing user message "
    "`[End of episode K/N] — [Continuing task]` marks the boundary before the "
    "next episode's first observation. This is a memory-compression artifact, "
    "NOT a format change — continue to think before each new action."
)

_PROTOCOL_OBS_ACTION_REFLECTION = (
    "\n- Between episodes: your prior history is reduced to the previous "
    "episode's observations paired with `\\boxed{action}` (the thinking "
    "around each action is dropped to save context), followed by a trailing "
    "user message `[End of episode K/N]\\n<reflection>...</reflection>\\n\\n"
    "[Continuing task]` — you write the reflection. The stripped transcript "
    "is a compression artifact, NOT a format change — continue to think "
    "before each new action."
)

_PROTOCOL_ORACLE = (
    "\n- Between episodes: a deterministic environment-derived note appears "
    "inside `[End of episode K/N]\\n<context_summary>...</context_summary>` "
    "in place of a summary you write. Treat it as reliable factual state "
    "about the task — the system provides it, you do not write it."
)


def build_multi_episode_system_prompt(
    *,
    summarization: dict | None = None,
    single_episode: bool = False,
) -> str:
    """Compose a system prompt that documents the active carryover protocol.

    Args:
        summarization: Resolved summarization config dict, shaped like the one
            consumed by :class:`agents.context_summarizer.ContextSummarizerMixin`
            (``enable``, ``mode``, ``episodic_carryover``,
            ``oracle.{enable,scope}``). When ``None`` or ``enable`` is falsy,
            only the base prompt is returned.
        single_episode: Use the single-episode base prompt instead of the
            multi-episode one. The protocol section is still appended when
            applicable (token-mode summarization can fire in single-episode
            runs).

    Returns:
        Prompt string. May contain the literal ``{num_episodes}`` placeholder;
        :meth:`agents.gem_text_agent.GEMTextAgent` substitutes it from
        ``info["num_episodes"]`` on the first env step.
    """
    base = _BASE_SINGLE_EPISODE if single_episode else _BASE_MULTI_EPISODE

    cfg = summarization or {}
    if not cfg.get("enable"):
        return base

    mode = cfg.get("mode", "episodic")
    carryover = cfg.get("episodic_carryover", "freeform")
    oracle_enabled = bool((cfg.get("oracle") or {}).get("enable"))

    parts = [base, _PROTOCOL_HEADER]

    if mode in ("token", "both"):
        parts.append(_PROTOCOL_TOKEN)

    if mode in ("episodic", "both"):
        if oracle_enabled:
            parts.append(_PROTOCOL_ORACLE)
        elif carryover == "obs_action":
            parts.append(_PROTOCOL_OBS_ACTION)
        elif carryover == "obs_action_reflection":
            parts.append(_PROTOCOL_OBS_ACTION_REFLECTION)
        else:
            parts.append(_PROTOCOL_FREEFORM)

    return "".join(parts)
