"""Context summarization mixin for rLLM agents.

Adds mid-trajectory context compression: when the agent's conversation
history exceeds a token threshold, it builds a summarization prompt,
and the engine generates a summary that replaces the history.

Usage — compose with any BaseAgent subclass::

    class MyAgent(ContextSummarizerMixin, GEMTextAgent): ...
"""

from __future__ import annotations

import re
from typing import Any

from rllm.agents.utils import convert_messages_to_tokens_and_masks

from agents.gem_text_agent import GEMTextAgent
from agents.gem_text_agent_noncumulative import GEMTextAgentNonCumulative
from prompts.summarization_prompts import (
    CONTEXT_SUMMARY_PROMPT,
    REFLECTIVE_SUMMARY_PROMPT,
)

_SUMMARY_TAG_RE = re.compile(
    r"<context_summary>(.*?)</context_summary>", re.DOTALL
)


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
    * ``mode`` (str, default ``"token"``) — one of ``"token" | "episodic" | "both"``.
      Controls which trigger predicates return True. ``"token"`` preserves the
      original threshold-based behavior. ``"episodic"`` fires only on episode
      boundaries. ``"both"`` enables both triggers.
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
        self.summarization_mode: str = kwargs.pop("mode", "token")
        if self.summarization_mode not in self._VALID_MODES:
            raise ValueError(
                f"summarization mode must be one of {self._VALID_MODES}, "
                f"got {self.summarization_mode!r}"
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

    def should_summarize(self, tokenizer: Any, chat_parser: Any) -> bool:
        """Return True when the token-threshold trigger should fire.

        Gated on ``summarization_mode`` — returns False unless mode includes
        the token trigger (``"token"`` or ``"both"``).
        """
        if self.summarization_mode not in ("token", "both"):
            return False
        messages = self._messages  # type: ignore[attr-defined]
        # Need at least system + 1 user + 1 assistant to summarize.
        if len(messages) < 3:
            return False
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

        ``trigger`` is recorded for metadata purposes but does not by itself
        change the template. ``use_reflective_prompt=True`` selects
        ``REFLECTIVE_SUMMARY_PROMPT`` (reflection-style, suitable for
        episode-end fires when env-level reflection is enabled).
        """
        del trigger  # currently informational only; prompt is chosen by use_reflective_prompt
        messages = list(self._messages)  # type: ignore[attr-defined]
        instruction = REFLECTIVE_SUMMARY_PROMPT if use_reflective_prompt else CONTEXT_SUMMARY_PROMPT
        messages.append({"role": "user", "content": instruction})
        return messages

    def apply_summary(
        self,
        summary_text: str,
        trigger: str = "token",
    ) -> None:
        """Replace ``self._messages`` with compressed context.

        Resulting layout::

            [system_prompt,
             {"role": "user", "content": "<context_summary>…</context_summary>\\n\\n[Continuing task]"}]

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
