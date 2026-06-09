"""Tests for the rollout-correction config guards in MultiEpisodeAgentPPOTrainer.

We exercise ``_resolve_rollout_correction`` directly via OmegaConf-backed
synthetic configs — instantiating the full trainer requires Ray + FSDP,
which is out of scope for a unit test. The resolver is the gate that
decides whether TIS fires at all, so testing it in isolation covers the
contract.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from trainers.multi_episode_trainer import MultiEpisodeAgentPPOTrainer


class _TrainerStub:
    """Borrow just ``_resolve_rollout_correction`` from the real trainer.

    The real ``__init__`` would set up Ray, FSDP, dataloaders, etc. We only
    care about the guard logic — which reads ``self.config`` — so we bind
    the unbound method to a stub holding a config attribute.
    """

    def __init__(self, cfg):
        self.config = cfg


# Bind the method off the real class so any drift in the resolver shows up
# in these tests automatically.
_resolve = MultiEpisodeAgentPPOTrainer._resolve_rollout_correction


def _cfg(**rc_overrides) -> OmegaConf:
    """Build a config with optional algorithm.rollout_correction overrides."""
    base = {
        "algorithm": {"rollout_correction": rc_overrides} if rc_overrides else {},
        "rllm": {
            "advantage_method": {
                "name": "grpo",
                "chunk_discounted_topr": {"enable": False},
            }
        },
    }
    return OmegaConf.create(base)


def test_no_rollout_correction_returns_none():
    cfg = _cfg()  # algorithm.rollout_correction absent
    stub = _TrainerStub(cfg)
    rc_cfg, effective = _resolve(stub)
    assert rc_cfg is None
    assert effective is None


def test_rollout_is_token_returns_token():
    stub = _TrainerStub(_cfg(rollout_is="token", rollout_is_threshold=2.0))
    rc_cfg, effective = _resolve(stub)
    assert rc_cfg is not None
    assert effective == "token"


def test_rollout_is_null_returns_none():
    stub = _TrainerStub(_cfg(rollout_is=None, rollout_is_threshold=2.0))
    _, effective = _resolve(stub)
    assert effective is None


def test_sequence_mode_raises():
    stub = _TrainerStub(_cfg(rollout_is="sequence"))
    with pytest.raises(ValueError, match="sequence"):
        _resolve(stub)


def test_bypass_mode_raises():
    stub = _TrainerStub(_cfg(rollout_is="token", bypass_mode=True))
    with pytest.raises(ValueError, match="bypass_mode"):
        _resolve(stub)


def test_rejection_sampling_raises():
    stub = _TrainerStub(_cfg(rollout_is="token", rollout_rs="token"))
    with pytest.raises(ValueError, match="rollout_rs"):
        _resolve(stub)


def test_batch_normalize_raises():
    stub = _TrainerStub(_cfg(rollout_is="token", rollout_is_batch_normalize=True))
    with pytest.raises(ValueError, match="batch_normalize"):
        _resolve(stub)


def test_topr_auto_disables_tis(capsys):
    """Soft guard: TOPR + TIS → effective is None and a single INFO line."""
    cfg = OmegaConf.create({
        "algorithm": {"rollout_correction": {"rollout_is": "token"}},
        "rllm": {
            "advantage_method": {
                "name": "chunk_discounted_topr",
                "chunk_discounted_topr": {"enable": True},
            }
        },
    })
    stub = _TrainerStub(cfg)
    rc_cfg, effective = _resolve(stub)
    assert rc_cfg is not None  # config carried for potential future use
    assert effective is None
    captured = capsys.readouterr()
    assert "TOPR is enabled" in captured.out


def test_topr_enabled_but_method_not_set_does_not_disable():
    """Only auto-disable when TOPR is actually the active method.

    If ``chunk_discounted_topr.enable=true`` is set but
    ``advantage_method.name`` is still ``grpo`` (stale config), TIS stays on.
    """
    cfg = OmegaConf.create({
        "algorithm": {"rollout_correction": {"rollout_is": "token"}},
        "rllm": {
            "advantage_method": {
                "name": "grpo",
                "chunk_discounted_topr": {"enable": True},  # set but not active
            }
        },
    })
    stub = _TrainerStub(cfg)
    _, effective = _resolve(stub)
    assert effective == "token"
