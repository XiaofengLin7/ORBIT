"""Context summarization mixin for rLLM agents.

Adds mid-trajectory context compression: when the agent's conversation
history exceeds a token threshold, it builds a summarization prompt,
and the engine generates a summary that replaces the history.

At episode boundaries (episodic mode) the ``episodic_carryover`` knob picks
one of three behaviors: ``"freeform"`` (LLM summary replaces history — the
original behavior), ``"obs_action"`` (deterministic: keep the last episode's
observations and ``\\boxed{}`` actions, no LLM call), or
``"obs_action_reflection"`` (keep that transcript and add a trainable
``<reflection>`` block on top).

Usage — compose with any BaseAgent subclass::

    class MyAgent(ContextSummarizerMixin, GEMTextAgent): ...
"""

from __future__ import annotations

import re
from typing import Any

from rllm.agents.utils import convert_messages_to_tokens_and_masks

from agents.gem_text_agent import GEMTextAgent, extract_last_boxed
from agents.gem_text_agent_noncumulative import GEMTextAgentNonCumulative
from prompts.summarization_prompts import (
    EPISODIC_SUMMARY_PROMPT,
    REFLECTION_PROMPT,
    REFLECTIVE_SUMMARY_PROMPT,
    TOKEN_SUMMARY_PROMPT,
)

_SUMMARY_TAG_RE = re.compile(
    r"<context_summary>(.*?)</context_summary>", re.DOTALL
)
_REFLECTION_TAG_RE = re.compile(r"<reflection>(.*?)</reflection>", re.DOTALL)

_VALID_CARRYOVER = ("freeform", "obs_action", "obs_action_reflection")


class ContextSummarizerMixin:
    """Mixin that adds context-window summarization to any BaseAgent.

    The mixin cooperates with the engine via three public methods:

    * :meth:`should_summarize` — called by the engine after each
      ``agent.update_from_env()`` to decide whether compression is needed.
    * :meth:`build_summarization_prompt` — returns the message list that
      the engine feeds to the LLM for summary generation.
    * :meth:`apply_summary` — replaces ``self._messages`` with the
      compressed context.

    Attributes set from ``__init__`` kwargs (popped before forwarding to
    the concrete agent):

    * ``summarization_threshold_tokens`` (int, default 16384)
    * ``summary_max_tokens`` (int, default 8192)
    * ``mode`` (str, default ``"episodic"``) — one of ``"token" | "episodic" | "both"``.
      Controls which trigger predicates return True. ``"episodic"`` fires only
      on episode boundaries (the default). ``"token"`` preserves the original
      threshold-based behavior. ``"both"`` enables both triggers.
    * ``episodic_carryover`` (str, default ``"freeform"``) — controls what the
      agent's history looks like *after* an episode-boundary summarization.
      ``"freeform"``: the LLM writes a ``<context_summary>`` that REPLACES the
      history (the original behavior). ``"obs_action"``: no LLM call — the
      previous episode's observations and actions are kept verbatim, with each
      assistant turn rewritten to just ``\\boxed{<action>}`` (thinking
      discarded); nothing trainable is produced. ``"obs_action_reflection"``:
      same kept transcript, plus the LLM is asked (via ``REFLECTION_PROMPT``)
      to add a ``<reflection>`` block on top — that generation IS a trainable
      step. This knob only affects the episode-end trigger; the token trigger
      always uses the ``"freeform"`` path.
    """

    # ------------------------------------------------------------------
    # Init / reset
    # ------------------------------------------------------------------

    _VALID_MODES = ("token", "episodic", "both")

    def __init__(self, **kwargs: Any):
        # Pop summarization-specific kwargs before forwarding.
        self.summarization_threshold_tokens: int = kwargs.pop(
            "summarization_threshold_tokens", 16384
        )
        self.summary_max_tokens: int = kwargs.pop("summary_max_tokens", 8192)
        self.summarization_mode: str = kwargs.pop("mode", "episodic")
        if self.summarization_mode not in self._VALID_MODES:
            raise ValueError(
                f"summarization mode must be one of {self._VALID_MODES}, "
                f"got {self.summarization_mode!r}"
            )
        self.episodic_carryover: str = kwargs.pop(
            "episodic_carryover", "freeform"
        )
        if self.episodic_carryover not in _VALID_CARRYOVER:
            raise ValueError(
                f"episodic_carryover must be one of {_VALID_CARRYOVER}, "
                f"got {self.episodic_carryover!r}"
            )

        # Internal state — reset in reset().
        self._summarization_count: int = 0
        self._summarization_events: list[dict] = []

        super().__init__(**kwargs)

    def reset(self) -> None:  # type: ignore[override]
        super().reset()  # type: ignore[misc]
        self._summarization_count = 0
        self._summarization_events = []

    # ------------------------------------------------------------------
    # Public API consumed by the engine
    # ------------------------------------------------------------------

    def should_summarize(
        self,
        tokenizer: Any = None,
        chat_parser: Any = None,
        *,
        current_token_count: int | None = None,
    ) -> bool:
        """Return True when the token-threshold trigger should fire.

        Gated on ``summarization_mode`` — returns False unless mode includes
        the token trigger (``"token"`` or ``"both"``).

        ``current_token_count``: optional pre-computed total token count for
        the conversation history. Always pass this from the engine when
        available — the engine already tracks running token counts
        incrementally, so re-tokenizing the full history here is O(N²) per
        trajectory and CPU-thrashes the rollout. Falls back to a one-shot
        full-history tokenization only when the count isn't supplied
        (e.g. legacy callers, eval harness).
        """
        if self.summarization_mode not in ("token", "both"):
            return False
        messages = self._messages  # type: ignore[attr-defined]
        # Need at least system + 1 user + 1 assistant to summarize.
        if len(messages) < 3:
            return False
        if current_token_count is not None:
            return current_token_count >= self.summarization_threshold_tokens
        # Fallback: re-tokenize. Slow path; avoid in the hot loop.
        tokens, _ = convert_messages_to_tokens_and_masks(
            messages,
            tokenizer=tokenizer,
            parser=chat_parser,
            contains_first_msg=True,
            contains_generation_msg=True,
        )
        return len(tokens) >= self.summarization_threshold_tokens

    def should_summarize_on_episode_end(self, info: dict) -> bool:
        """Return True when the episodic trigger should fire.

        Gated on ``summarization_mode`` — returns False unless mode includes
        the episodic trigger (``"episodic"`` or ``"both"``). Reads
        ``info["episode_done"]`` to detect the episode boundary.
        """
        if self.summarization_mode not in ("episodic", "both"):
            return False
        messages = self._messages  # type: ignore[attr-defined]
        if len(messages) < 3:
            return False
        return bool(info.get("episode_done"))

    def build_summarization_prompt(
        self,
        trigger: str = "token",
        use_reflective_prompt: bool = False,
    ) -> list[dict[str, str]]:
        """Return message list = full history + summarization instruction.

        Template selection:
        * ``use_reflective_prompt=True`` → ``REFLECTIVE_SUMMARY_PROMPT``
          (reflection + compression, used at episode end when the env
          has reflection enabled).
        * ``trigger="episode_end"`` and reflection disabled →
          ``EPISODIC_SUMMARY_PROMPT`` (compression-only at episode end).
        * Otherwise (token trigger, mid-episode) →
          ``TOKEN_SUMMARY_PROMPT``.
        """
        messages = list(self._messages)  # type: ignore[attr-defined]
        if use_reflective_prompt:
            instruction = REFLECTIVE_SUMMARY_PROMPT
        elif trigger == "episode_end":
            instruction = EPISODIC_SUMMARY_PROMPT
        else:
            instruction = TOKEN_SUMMARY_PROMPT
        messages.append({"role": "user", "content": instruction})
        return messages

    def build_reflection_prompt(self) -> list[dict[str, str]]:
        """Return message list = full history + the reflection instruction.

        Used by the ``episodic_carryover="obs_action_reflection"`` mode. Unlike
        ``build_summarization_prompt``, the resulting summary does not replace
        history — see :meth:`apply_episode_carryover`.
        """
        messages = list(self._messages)  # type: ignore[attr-defined]
        messages.append({"role": "user", "content": REFLECTION_PROMPT})
        return messages

    @staticmethod
    def _boxify_action_turns(
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Return a copy of ``messages`` with assistant turns reduced to the action.

        Each ``{"role": "assistant"}`` message's content is replaced by
        ``\\boxed{<parsed action>}`` — i.e. any chain-of-thought / prose around
        the boxed answer is discarded. Non-assistant messages pass through
        unchanged.
        """
        out: list[dict[str, str]] = []
        for m in messages:
            if m.get("role") == "assistant":
                action = extract_last_boxed(m.get("content", ""))
                out.append({"role": "assistant", "content": f"\\boxed{{{action}}}"})
            else:
                out.append(dict(m))
        return out

    @staticmethod
    def _format_episode_label(
        episode_index: int | None, num_episodes: int | None
    ) -> str:
        """Render '1/3' if total known, '1' if not, '' if index unknown."""
        if episode_index is None:
            return ""
        k = int(episode_index) + 1  # display 1-based
        if num_episodes:
            return f"{k}/{int(num_episodes)}"
        return f"{k}"

    @classmethod
    def _episode_header(
        cls,
        trigger: str,
        episode_index: int | None,
        num_episodes: int | None,
    ) -> str:
        """Build the leading marker that labels which episode the block is from.

        ``episode_index`` is the 0-based index of the *just-finished* episode
        (for ``episode_end``) or the currently-running episode (for ``token``).
        Returns ``""`` when no label can be formed.
        """
        label = cls._format_episode_label(episode_index, num_episodes)
        if not label:
            return ""
        if trigger == "episode_end":
            return f"[End of episode {label}]\n"
        # token (mid-episode) or any other trigger
        return f"[Mid-episode {label} compression]\n"

    def apply_episode_carryover(
        self,
        episode_start_msg_idx: int,
        reflection_text: str | None = None,
        trigger: str = "episode_end",
        episode_index: int | None = None,
        num_episodes: int | None = None,
    ) -> None:
        """Rewrite history to keep only the last episode's obs/actions.

        Resulting layout (``obs_action`` mode)::

            [system, obs_1, \\boxed{a_1}, ..., terminal_obs,
             {"role": "user", "content": "[End of episode K/N — Continuing task]"}]

        With ``reflection_text`` supplied (``obs_action_reflection`` mode) the
        trailing user message carries the episode label, the reflection block,
        and the continuation marker::

            {"role": "user",
             "content": "[End of episode K/N]\\n<reflection>...</reflection>\\n\\n[Continuing task]"}

        ``episode_start_msg_idx`` is the index in ``self._messages`` of the
        just-finished episode's first observation message (tracked by the
        engine, which manages episode-boundary message injection). The
        ``episode_index`` / ``num_episodes`` kwargs are sourced from the env
        by the engine; when omitted the label is dropped (carryover still
        works, just without episode context).
        """
        messages: list[dict[str, str]] = self._messages  # type: ignore[attr-defined]
        pre_message_count = len(messages)

        system_msgs = [m for m in messages if m["role"] == "system"]
        system_msg = system_msgs[0] if system_msgs else None

        tail = self._boxify_action_turns(messages[episode_start_msg_idx:])

        new_messages: list[dict[str, str]] = []
        if system_msg is not None:
            new_messages.append(system_msg)
        new_messages.extend(tail)

        label = self._format_episode_label(episode_index, num_episodes)
        episode_marker = (
            f"[End of episode {label}]" if label else "[End of previous episode]"
        )

        if reflection_text is not None:
            match = _REFLECTION_TAG_RE.search(reflection_text)
            body = match.group(1).strip() if match else reflection_text.strip()
            new_messages.append(
                {
                    "role": "user",
                    "content": (
                        f"{episode_marker}\n"
                        f"<reflection>\n{body}\n</reflection>"
                        "\n\n[Continuing task]"
                    ),
                }
            )
        else:
            # Plain obs_action: append a bare boundary marker so the model
            # knows where the prior episode ends and the next obs begins.
            new_messages.append(
                {
                    "role": "user",
                    "content": f"{episode_marker} — [Continuing task]",
                }
            )

        self._messages[:] = new_messages  # type: ignore[attr-defined]

        self._summarization_count += 1
        self._summarization_events.append(
            {
                "summarization_index": self._summarization_count,
                "pre_message_count": pre_message_count,
                "post_message_count": len(new_messages),
                "trigger": trigger,
                "kind": (
                    "carryover_reflection"
                    if reflection_text is not None
                    else "carryover"
                ),
            }
        )

    def apply_summary(
        self,
        summary_text: str,
        trigger: str = "token",
        episode_index: int | None = None,
        num_episodes: int | None = None,
    ) -> None:
        """Replace ``self._messages`` with compressed context.

        Resulting layout::

            [system_prompt,
             {"role": "user",
              "content": "[End of episode K/N]\\n<context_summary>…</context_summary>\\n\\n[Continuing task]"}]

        For ``trigger="token"`` the leading marker is
        ``"[Mid-episode K/N compression]"`` instead, so the model can tell
        whether it's reading a between-episodes summary or a within-episode
        compression. The label is dropped when ``episode_index`` is None
        (legacy callers, eval without env wiring).

        The summary is expected to carry forward whatever state the agent
        needs about the current episode; there's no preserved trailing
        user message, because the next env observation will be appended by
        the engine (token path: via the next step's ``update_from_env``;
        episodic path: via the immediate ``env.start_new_episode()`` call).
        """
        messages: list[dict[str, str]] = self._messages  # type: ignore[attr-defined]
        pre_message_count = len(messages)

        # Extract summary body from tags; fall back to raw text.
        match = _SUMMARY_TAG_RE.search(summary_text)
        summary_body = match.group(1).strip() if match else summary_text.strip()

        header = self._episode_header(trigger, episode_index, num_episodes)

        # Keep the system message (always first if present).
        system_msgs = [m for m in messages if m["role"] == "system"]
        system_msg = system_msgs[0] if system_msgs else None

        new_messages: list[dict[str, str]] = []
        if system_msg is not None:
            new_messages.append(system_msg)
        new_messages.append(
            {
                "role": "user",
                "content": (
                    f"{header}"
                    f"<context_summary>\n{summary_body}\n</context_summary>"
                    "\n\n[Continuing task]"
                ),
            }
        )

        self._messages[:] = new_messages  # type: ignore[attr-defined]

        self._summarization_count += 1
        self._summarization_events.append(
            {
                "summarization_index": self._summarization_count,
                "pre_message_count": pre_message_count,
                "post_message_count": len(new_messages),
                "trigger": trigger,
            }
        )

    def get_summarization_metadata(self) -> dict:
        """Return metadata about summarizations performed."""
        return {
            "summarization_count": self._summarization_count,
            "summarization_events": list(self._summarization_events),
        }


# ------------------------------------------------------------------
# Composed concrete classes
# ------------------------------------------------------------------

class GEMTextAgentWithSummarization(ContextSummarizerMixin, GEMTextAgent):
    """GEMTextAgent with mid-trajectory context summarization."""


class GEMTextAgentNonCumulativeWithSummarization(
    ContextSummarizerMixin, GEMTextAgentNonCumulative
):
    """GEMTextAgentNonCumulative with mid-trajectory context summarization."""
