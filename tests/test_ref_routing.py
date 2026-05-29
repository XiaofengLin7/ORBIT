"""Unit tests for ref-model routing in SummarizingAgentExecutionEngine.

Exercises the dispatch helpers added to support training only the
summarization head: ``_get_ref_engine``, ``_get_action_response``,
``_get_summary_response``. Does not run a full rollout — the goal is
to lock the routing wiring so action calls go to the frozen ref
endpoint and summary/reflection calls stay on the trainable engine.
"""

from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import MagicMock

import pytest

from trainers.summarizing_engine import SummarizingAgentExecutionEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Cfg:
    """Tiny dict-as-attr wrapper matching how OmegaConf nodes are read here."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def get(self, key, default=None):
        return getattr(self, key, default)


def _engine_stub(ref_cfg: dict | None) -> SummarizingAgentExecutionEngine:
    """Build a bare engine with just enough attrs for the routing helpers."""
    engine = SummarizingAgentExecutionEngine.__new__(SummarizingAgentExecutionEngine)
    summarization = _Cfg(ref_model=_Cfg(**ref_cfg)) if ref_cfg else _Cfg()
    engine.config = _Cfg(rllm=_Cfg(agent=_Cfg(summarization=summarization)))
    engine.tokenizer = MagicMock()
    engine.max_prompt_length = 1024
    engine.max_response_length = 4096
    engine.max_model_len = 32768
    return engine


# ---------------------------------------------------------------------------
# _get_ref_engine
# ---------------------------------------------------------------------------


def test_ref_engine_disabled_when_config_absent():
    engine = _engine_stub(ref_cfg=None)
    assert engine._get_ref_engine() is None


def test_ref_engine_disabled_when_enable_false():
    engine = _engine_stub(ref_cfg={"enable": False, "base_url": "x"})
    assert engine._get_ref_engine() is None


def test_ref_engine_constructed_when_enabled(monkeypatch):
    """When enable=true, _get_ref_engine returns the OpenAIEngine instance."""
    sentinel = object()
    fake_engine_cls = MagicMock(return_value=sentinel)

    # Patch the module-level OpenAIEngine import target.
    fake_module = types.ModuleType("rllm.engine.rollout.openai_engine")
    fake_module.OpenAIEngine = fake_engine_cls
    monkeypatch.setitem(sys.modules, "rllm.engine.rollout.openai_engine", fake_module)

    engine = _engine_stub(
        ref_cfg={
            "enable": True,
            "base_url": "http://test:9999/v1",
            "api_key": "EMPTY",
            "model_name": "ref-frozen",
        }
    )
    out = engine._get_ref_engine()
    assert out is sentinel

    # Verify the OpenAIEngine was constructed with the right kwargs.
    fake_engine_cls.assert_called_once()
    call_kwargs = fake_engine_cls.call_args.kwargs
    assert call_kwargs["model"] == "ref-frozen"
    assert call_kwargs["base_url"] == "http://test:9999/v1"
    assert call_kwargs["api_key"] == "EMPTY"
    assert call_kwargs["tokenizer"] is engine.tokenizer
    assert call_kwargs["max_prompt_length"] == engine.max_prompt_length

    # Cached on subsequent calls (no second construction).
    out2 = engine._get_ref_engine()
    assert out2 is sentinel
    assert fake_engine_cls.call_count == 1


# ---------------------------------------------------------------------------
# _get_action_response / _get_summary_response dispatch
# ---------------------------------------------------------------------------


def test_action_response_uses_self_when_ref_disabled():
    engine = _engine_stub(ref_cfg=None)
    captured = {}

    async def fake_get_model_response(messages, application_id, **kwargs):
        captured["which"] = "self"
        captured["messages"] = messages
        captured["application_id"] = application_id
        return "self-output"

    engine.get_model_response = fake_get_model_response

    out = asyncio.get_event_loop().run_until_complete(
        engine._get_action_response(
            [{"role": "user", "content": "act"}], "app-1", max_tokens=64
        )
    )
    assert out == "self-output"
    assert captured["which"] == "self"


def test_action_response_routes_to_ref_when_enabled(monkeypatch):
    """When ref is enabled, _get_action_response goes to the ref engine."""
    calls = {"ref": 0, "self": 0}

    def _check_kwargs(kwargs):
        assert kwargs.get("application_id") == "app-1"
        # verl-internal kwargs must NOT be forwarded — the OpenAI
        # client rejects unknown kwargs (TypeError) and we don't
        # want to leak them into request bodies.
        for forbidden in (
            "accumulated_prompt_ids",
            "meta_info",
            "timing_raw",
            "uids",
        ):
            assert forbidden not in kwargs, (
                f"verl-only kwarg {forbidden!r} leaked to ref engine"
            )
        # Sampling kwargs MUST pass through.
        assert kwargs.get("max_tokens") == 64
        assert kwargs.get("temperature") == 0.6
        # The trajectory loop handles prompt-length enforcement
        # itself (matching the parent rLLM engine convention); the
        # OpenAIEngine must NOT raise on prompt > max_prompt_length.
        assert kwargs.get("enforce_max_prompt_length") is False

    class FakeRefEngine:
        # Token-ID fast path: when accumulated_prompt_ids is provided,
        # _get_action_response calls completion(token_ids, **kwargs).
        async def completion(self, prompt, **kwargs):
            calls["ref"] += 1
            assert prompt == [1, 2, 3]  # the accumulated_prompt_ids
            _check_kwargs(kwargs)
            return "ref-output"

        # Chat-template path: used when no accumulated_prompt_ids.
        async def get_model_response(self, messages, **kwargs):
            calls["ref"] += 1
            _check_kwargs(kwargs)
            return "ref-output"

    fake_module = types.ModuleType("rllm.engine.rollout.openai_engine")
    fake_module.OpenAIEngine = lambda **_: FakeRefEngine()
    monkeypatch.setitem(sys.modules, "rllm.engine.rollout.openai_engine", fake_module)

    async def fake_get_model_response(*a, **kw):
        calls["self"] += 1
        return "self-output"

    engine = _engine_stub(
        ref_cfg={
            "enable": True,
            "base_url": "http://test:9999/v1",
            "api_key": "EMPTY",
            "model_name": "ref-frozen",
        }
    )
    engine.get_model_response = fake_get_model_response

    out = asyncio.get_event_loop().run_until_complete(
        engine._get_action_response(
            [{"role": "user", "content": "act"}],
            "app-1",
            max_tokens=64,
            temperature=0.6,
            accumulated_prompt_ids=[1, 2, 3],
            meta_info={"step": 5},
            timing_raw={"gen": 0.0},
            uids=["traj-0"],
        )
    )
    assert out == "ref-output"
    assert calls == {"ref": 1, "self": 0}


def test_summary_response_always_uses_self(monkeypatch):
    """Summaries / reflections must stay on the trainable engine even when ref is on."""
    calls = {"ref": 0, "self": 0}

    class FakeRefEngine:
        async def get_model_response(self, messages, **kwargs):
            calls["ref"] += 1
            return "ref-output"

    fake_module = types.ModuleType("rllm.engine.rollout.openai_engine")
    fake_module.OpenAIEngine = lambda **_: FakeRefEngine()
    monkeypatch.setitem(sys.modules, "rllm.engine.rollout.openai_engine", fake_module)

    async def fake_get_model_response(messages, application_id, **kwargs):
        calls["self"] += 1
        return "self-output"

    engine = _engine_stub(
        ref_cfg={
            "enable": True,
            "base_url": "http://test:9999/v1",
            "api_key": "EMPTY",
            "model_name": "ref-frozen",
        }
    )
    engine.get_model_response = fake_get_model_response

    out = asyncio.get_event_loop().run_until_complete(
        engine._get_summary_response(
            [{"role": "user", "content": "summarize"}], "app-1", max_tokens=128
        )
    )
    assert out == "self-output"
    assert calls == {"ref": 0, "self": 1}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
