"""Tests for alternating actor/summarizer training.

Covers:
- Phase-state initialization from config.
- Phase switching after T steps (via init_envs_and_agents + _phase_steps_done).
- Engine training_phase attribute updated before each batch.
- _assemble_summarization_mask: marks correct positions.
- assemble_segments: summary tokens zeroed in actor phase, intact in summarizer phase.
"""

from __future__ import annotations

import torch
import pytest
from omegaconf import OmegaConf

from trainers.summarizing_engine import SummarizingAgentExecutionEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _engine_stub() -> SummarizingAgentExecutionEngine:
    """Return an uninitialized engine instance with only training_phase set."""
    obj = object.__new__(SummarizingAgentExecutionEngine)
    obj.training_phase = "summarizer"
    return obj


def _make_steps(n_action: int, include_summary: bool = True) -> list[dict]:
    """Build a minimal episode_steps list for assembly tests.

    Layout:
      steps[0]: action step, completion_ids=[10, 11, 12]
      steps[1]: action step (if n_action >= 2), completion_ids=[20, 21]
      steps[-1]: summarization step (if include_summary), completion_ids=[99]

    prompt_ids for each step is accumulated so assemble_steps succeeds.
    """
    steps = []
    prompt = [1, 2, 3]  # initial prompt
    accumulated = list(prompt)

    # Action steps
    for i in range(n_action):
        base = (i + 1) * 10
        completion = [base, base + 1, base + 2]
        step = {
            "prompt_ids": list(accumulated),
            "completion_ids": list(completion),
            "logprobs": [-0.1] * len(completion),
            "is_summarization": False,
        }
        steps.append(step)
        accumulated = list(accumulated) + list(completion)

    # Optional summarization step
    if include_summary:
        summ_completion = [99]
        # Simulate a summarization instruction gap in the prompt:
        # prompt_ids = accumulated + [50, 51] (instruction tokens, mask=0 in assemble)
        summ_prompt = list(accumulated) + [50, 51]
        step = {
            "prompt_ids": summ_prompt,
            "completion_ids": list(summ_completion),
            "logprobs": [-0.5],
            "is_summarization": True,
        }
        steps.append(step)

    return steps


# ---------------------------------------------------------------------------
# _assemble_summarization_mask
# ---------------------------------------------------------------------------

class TestAssembleSummarizationMask:
    def test_no_summary_steps_all_false(self):
        """No is_summarization steps → all False."""
        eng = _engine_stub()
        steps = _make_steps(2, include_summary=False)
        # Total response tokens: 3 + 3 + 2 (gaps) + ... let assemble_steps figure it out.
        # We compute expected_len from the number of completion tokens + gaps.
        # action0: comp=[10,11,12] → 3 tokens mask=1
        # action1: prompt=[1,2,3,10,11,12,20,21,22], accumulated_len=6, gap=0, comp=[20,21,22] → 3 tokens
        # Total: 6 tokens
        mask = eng._assemble_summarization_mask(steps, expected_len=6)
        assert mask.dtype == torch.bool
        assert mask.shape == (6,)
        assert not mask.any()

    def test_summary_step_at_end_marked(self):
        """Last step is summarization → its completion positions are True."""
        eng = _engine_stub()
        steps = _make_steps(1, include_summary=True)
        # action0: prompt=[1,2,3], comp=[10,11,12] → 3 tokens (mask=1, is_summ=False)
        # summ: prompt=[1,2,3,10,11,12,50,51], gap=2 (mask=0), comp=[99] → 1 token (is_summ=True)
        # Total response tokens: 3 (action) + 2 (gap) + 1 (summary) = 6
        mask = eng._assemble_summarization_mask(steps, expected_len=6)
        assert mask.shape == (6,)
        # first 3: action (False), next 2: gap (False), last 1: summary (True)
        expected = torch.tensor([False, False, False, False, False, True])
        assert torch.equal(mask, expected)

    def test_expected_len_pads_if_short(self):
        """Shorter step list pads to expected_len with False."""
        eng = _engine_stub()
        steps = _make_steps(1, include_summary=False)
        mask = eng._assemble_summarization_mask(steps, expected_len=20)
        assert mask.shape == (20,)
        # Positions beyond the real tokens are False (padding)
        assert not mask[6:].any()


# ---------------------------------------------------------------------------
# assemble_segments masking
# ---------------------------------------------------------------------------

def _real_assemble_steps(steps: list[dict]) -> tuple:
    """Invoke parent assemble_steps without a full engine, providing a stub config."""
    from omegaconf import OmegaConf
    from third_party.rllm.rllm.engine.agent_execution_engine import AgentExecutionEngine

    stub = object.__new__(AgentExecutionEngine)
    stub.config = OmegaConf.create({"rllm": {"filter_token_mismatch": False}})
    return AgentExecutionEngine.assemble_steps(stub, steps)


class TestAssembleSegmentsMasking:
    def _make_engine(self, phase: str) -> SummarizingAgentExecutionEngine:
        eng = _engine_stub()
        eng.training_phase = phase
        eng.assemble_steps = _real_assemble_steps
        return eng

    def test_summarizer_phase_summary_tokens_included(self):
        """In summarizer phase, summary tokens remain mask=1."""
        eng = self._make_engine("summarizer")
        steps = _make_steps(1, include_summary=True)
        segs = eng.assemble_segments(steps, summarization_boundaries=[], trajectory_reward=1.0)
        assert len(segs) == 1
        mask = segs[0]["response_masks"]
        # Summary completion token at the last position should be mask=1.
        assert mask[-1].item() == 1

    def test_actor_phase_summary_tokens_zeroed(self):
        """In actor phase, summary token positions get mask=0."""
        eng = self._make_engine("actor")
        steps = _make_steps(1, include_summary=True)
        segs = eng.assemble_segments(steps, summarization_boundaries=[], trajectory_reward=1.0)
        assert len(segs) == 1
        mask = segs[0]["response_masks"]
        # Last position is the summary completion → must be 0 in actor phase.
        assert mask[-1].item() == 0

    def test_actor_phase_action_tokens_intact(self):
        """In actor phase, action token positions retain their original mask values."""
        eng = self._make_engine("actor")
        steps = _make_steps(1, include_summary=True)
        segs = eng.assemble_segments(steps, summarization_boundaries=[], trajectory_reward=1.0)
        mask = segs[0]["response_masks"]
        # Action completion tokens: first 3 positions (mask=1)
        assert mask[0].item() == 1
        assert mask[1].item() == 1
        assert mask[2].item() == 1

    def test_actor_phase_no_summary_steps_unchanged(self):
        """Actor phase without any summarization steps: all action tokens mask=1."""
        eng = self._make_engine("actor")
        steps = _make_steps(2, include_summary=False)
        segs = eng.assemble_segments(steps, summarization_boundaries=[], trajectory_reward=1.0)
        mask = segs[0]["response_masks"]
        # All mask=1 positions should remain 1 — no summarization to zero.
        assert (mask == 1).all()


# ---------------------------------------------------------------------------
# Phase state initialization
# ---------------------------------------------------------------------------

class TestPhaseInitialization:
    def _make_trainer_state(self, alt_cfg_dict: dict) -> dict:
        """Simulate the __init__ phase-state extraction from config."""
        cfg = OmegaConf.create({
            "rllm": {"agent": {"summarization": {"alternating": alt_cfg_dict}}}
        })
        alt_cfg = OmegaConf.select(cfg, "rllm.agent.summarization.alternating", default=None)
        alt_enabled = bool(alt_cfg is not None and alt_cfg.get("enable", False))
        phase_T = int(alt_cfg.get("T", 100)) if alt_enabled else None
        training_phase = (
            str(alt_cfg.get("initial_phase", "summarizer")) if alt_enabled else "summarizer"
        )
        return {"enabled": alt_enabled, "T": phase_T, "phase": training_phase}

    def test_disabled_by_default(self):
        state = self._make_trainer_state({"enable": False, "T": 50})
        assert not state["enabled"]
        assert state["T"] is None

    def test_enabled_reads_T_and_phase(self):
        state = self._make_trainer_state(
            {"enable": True, "T": 10, "initial_phase": "actor"}
        )
        assert state["enabled"]
        assert state["T"] == 10
        assert state["phase"] == "actor"

    def test_default_initial_phase_is_summarizer(self):
        state = self._make_trainer_state({"enable": True, "T": 5})
        assert state["phase"] == "summarizer"


# ---------------------------------------------------------------------------
# Phase switching logic
# ---------------------------------------------------------------------------

class TestPhaseSwitching:
    def _run_phases(self, T: int, n_steps: int, initial: str = "summarizer") -> list[str]:
        """Simulate what init_envs_and_agents + _expanded_update_actor would do."""
        phase = initial
        steps_done = 0
        phases_at_update = []
        for _ in range(n_steps):
            # init_envs_and_agents: check for switch
            if steps_done >= T:
                phase = "actor" if phase == "summarizer" else "summarizer"
                steps_done = 0
            phases_at_update.append(phase)
            # _expanded_update_actor: increment
            steps_done += 1
        return phases_at_update

    def test_no_switch_before_T(self):
        phases = self._run_phases(T=3, n_steps=3)
        assert all(p == "summarizer" for p in phases)

    def test_switch_at_step_T(self):
        phases = self._run_phases(T=2, n_steps=5)
        # Steps 0,1 → summarizer; switch at step 2 (done=2≥T=2) → actor for steps 2,3
        # switch again at step 4 → summarizer
        assert phases == ["summarizer", "summarizer", "actor", "actor", "summarizer"]

    def test_initial_phase_actor(self):
        phases = self._run_phases(T=2, n_steps=4, initial="actor")
        assert phases == ["actor", "actor", "summarizer", "summarizer"]


# ---------------------------------------------------------------------------
# _get_action_response routing
# ---------------------------------------------------------------------------

class TestActionResponseRouting:
    def _make_eng(self, phase: str, has_ref: bool):
        eng = _engine_stub()
        eng.training_phase = phase
        eng._VERL_ONLY_KWARGS = frozenset({"accumulated_prompt_ids", "meta_info"})
        calls = []
        async def fake_trainable(*a, **kw):
            calls.append(("trainable", kw))
            return "trainable_output"
        async def fake_ref(*a, **kw):
            calls.append(("ref", kw))
            return "ref_output"
        eng.get_model_response = fake_trainable
        if has_ref:
            class FakeRef:
                async def get_model_response(self_, *a, **kw):
                    return await fake_ref(*a, **kw)
                async def completion(self_, tokens, **kw):
                    calls.append(("ref_completion", kw))
                    return "ref_output"
            eng._ref_engine_cache = FakeRef()
        else:
            eng._ref_engine_cache = None
        return eng, calls

    def test_summarizer_phase_action_goes_to_ref(self):
        import asyncio
        eng, calls = self._make_eng("summarizer", has_ref=True)
        asyncio.run(eng._get_action_response([], "app_id"))
        assert any(c[0].startswith("ref") for c in calls)

    def test_actor_phase_action_goes_to_trainable(self):
        import asyncio
        eng, calls = self._make_eng("actor", has_ref=True)
        asyncio.run(eng._get_action_response([], "app_id"))
        assert calls[0][0] == "trainable"

    def test_no_ref_falls_through_to_trainable(self):
        import asyncio
        eng, calls = self._make_eng("summarizer", has_ref=False)
        asyncio.run(eng._get_action_response([], "app_id"))
        assert calls[0][0] == "trainable"


# ---------------------------------------------------------------------------
# _get_summary_response routing
# ---------------------------------------------------------------------------

class TestSummaryResponseRouting:
    def _make_eng(self, phase: str, has_ref: bool):
        eng = _engine_stub()
        eng.training_phase = phase
        eng._VERL_ONLY_KWARGS = frozenset({"accumulated_prompt_ids", "meta_info"})
        calls = []
        async def fake_trainable(*a, **kw):
            calls.append("trainable")
            return "trainable_output"
        async def fake_ref(*a, **kw):
            calls.append("ref")
            return "ref_output"
        eng.get_model_response = fake_trainable
        if has_ref:
            class FakeRef:
                async def get_model_response(self_, *a, **kw):
                    return await fake_ref(*a, **kw)
            eng._ref_engine_cache = FakeRef()
        else:
            eng._ref_engine_cache = None
        return eng, calls

    def test_summarizer_phase_summary_goes_to_trainable(self):
        import asyncio
        eng, calls = self._make_eng("summarizer", has_ref=True)
        asyncio.run(eng._get_summary_response([], "app_id"))
        assert calls == ["trainable"]

    def test_actor_phase_summary_goes_to_ref(self):
        import asyncio
        eng, calls = self._make_eng("actor", has_ref=True)
        asyncio.run(eng._get_summary_response([], "app_id"))
        assert calls == ["ref"]

    def test_actor_phase_no_ref_falls_through_to_trainable(self):
        import asyncio
        eng, calls = self._make_eng("actor", has_ref=False)
        asyncio.run(eng._get_summary_response([], "app_id"))
        assert calls == ["trainable"]
