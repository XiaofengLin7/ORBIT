"""Tests for the multi-episode system-prompt builder.

Covers the protocol-section selection logic so the prompt the model sees at
training/eval time matches the carryover behavior the engine actually runs.
"""

from __future__ import annotations

import pytest

from prompts.system_prompts import build_multi_episode_system_prompt


# ---------------------------------------------------------------------------
# Base prompts (summarization disabled)
# ---------------------------------------------------------------------------


def test_base_multi_episode_no_summarization():
    p = build_multi_episode_system_prompt(summarization=None)
    assert "{num_episodes}" in p
    assert "\\boxed{}" in p
    # No protocol section.
    assert "Memory protocol" not in p
    assert "<context_summary>" not in p
    assert "<reflection>" not in p
    # User explicitly asked the brevity/penalty language be removed.
    assert "briefly" not in p
    assert "penalized" not in p


def test_base_single_episode_no_summarization():
    p = build_multi_episode_system_prompt(summarization=None, single_episode=True)
    assert "single episode" in p
    assert "{num_episodes}" not in p
    assert "Memory protocol" not in p
    assert "briefly" not in p
    assert "penalized" not in p


def test_enable_false_still_returns_base():
    p = build_multi_episode_system_prompt(summarization={"enable": False})
    assert "Memory protocol" not in p


# ---------------------------------------------------------------------------
# Protocol-section selection
# ---------------------------------------------------------------------------


def _cfg(mode: str = "episodic", carryover: str = "freeform", oracle: bool = False) -> dict:
    return {
        "enable": True,
        "mode": mode,
        "episodic_carryover": carryover,
        "oracle": {"enable": oracle},
    }


def test_freeform_episodic():
    p = build_multi_episode_system_prompt(summarization=_cfg("episodic", "freeform"))
    assert "Memory protocol" in p
    # Freeform body's distinguishing phrase.
    assert "Make it useful to your future self" in p
    # No mid-episode token section.
    assert "when your context grows large" not in p
    # No obs/action body.
    assert "\\boxed{action}" not in p
    # No reflection-generation body (the header mentions the tag, so we key
    # off a body-only phrase).
    assert "you write the reflection" not in p
    # No oracle body.
    assert "system provides it" not in p


def test_obs_action_episodic():
    p = build_multi_episode_system_prompt(summarization=_cfg("episodic", "obs_action"))
    assert "Memory protocol" in p
    assert "\\boxed{action}" in p
    assert "compression artifact" in p
    # Pure obs_action has no reflection-generation body.
    assert "you write the reflection" not in p
    assert "Make it useful to your future self" not in p
    assert "system provides it" not in p


def test_obs_action_reflection_episodic():
    p = build_multi_episode_system_prompt(
        summarization=_cfg("episodic", "obs_action_reflection")
    )
    assert "Memory protocol" in p
    assert "\\boxed{action}" in p
    assert "you write the reflection" in p
    assert "compression artifact" in p
    assert "Make it useful to your future self" not in p
    assert "system provides it" not in p


def test_oracle_overrides_carryover():
    p = build_multi_episode_system_prompt(
        summarization=_cfg("episodic", "obs_action_reflection", oracle=True)
    )
    # Oracle path replaces the freeform/obs_action language entirely.
    assert "system provides it" in p
    assert "\\boxed{action}" not in p
    assert "you write the reflection" not in p
    assert "Make it useful to your future self" not in p


def test_token_only_no_episodic_section():
    p = build_multi_episode_system_prompt(summarization=_cfg("token", "freeform"))
    assert "Memory protocol" in p
    # Token-section body phrase.
    assert "when your context grows large" in p
    # mode=token should NOT emit a Between-episodes section.
    assert "Between episodes" not in p


def test_both_modes_emit_both_sections():
    p = build_multi_episode_system_prompt(summarization=_cfg("both", "freeform"))
    assert "when your context grows large" in p
    assert "Between episodes" in p


def test_both_modes_with_obs_action_reflection():
    p = build_multi_episode_system_prompt(
        summarization=_cfg("both", "obs_action_reflection")
    )
    assert "when your context grows large" in p
    assert "\\boxed{action}" in p
    assert "you write the reflection" in p


def test_protocol_header_calls_out_user_role():
    """The user-role inversion is the most surprising behavior; it must be
    documented so the model knows the summary it sees is its own."""
    p = build_multi_episode_system_prompt(summarization=_cfg())
    assert "YOUR OWN" in p
    assert "[Continuing task]" in p


def test_protocol_header_documents_episode_markers():
    """The header must define `[End of episode K/N]` and the mid-episode
    variant so the model can read the labels we attach to carryover blocks."""
    p = build_multi_episode_system_prompt(summarization=_cfg())
    assert "[End of episode K/N]" in p
    assert "[Mid-episode K/N compression]" in p
    # And it should define what K and N mean.
    assert "1-based" in p
    assert "total number of episodes" in p


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing_key,expected_marker",
    [
        # No carryover → defaults to freeform.
        ({"enable": True, "mode": "episodic"}, "<context_summary>"),
        # No mode → defaults to episodic.
        ({"enable": True, "episodic_carryover": "obs_action"}, "\\boxed{action}"),
    ],
)
def test_missing_keys_fall_back_to_defaults(missing_key: dict, expected_marker: str):
    p = build_multi_episode_system_prompt(summarization=missing_key)
    assert expected_marker in p
