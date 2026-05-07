"""Regression test: when segment training is enabled, _balance_batch's
skinny-batch reorder is bypassed.

The bug it guards against: the parent rLLM trainer's fit_agent calls
self._balance_batch on the per-trajectory ("skinny") batch right before
update_actor. balance_batch reorders rows for DP load balancing, but our
segment-aware update_actor uses positional pairing between the cached
trajectory list and the batch's rows. A reorder de-syncs those, swapping
advantages between trajectories.

Our fix: override `_balance_batch` to no-op when `_segment_training_enabled`
is True, and apply load balancing on the (post-expansion) batch directly
inside `_expanded_update_actor`.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

from trainers.multi_episode_trainer import MultiEpisodeAgentPPOTrainer
from trainers.segment_expansion import extract_advantage_per_trajectory


def _make_trainer_stub(segment_training: bool):
    """Build a MultiEpisodeAgentPPOTrainer instance via __new__ that has
    just enough state for _balance_batch and the cache lookup to run.
    """
    trainer = MultiEpisodeAgentPPOTrainer.__new__(MultiEpisodeAgentPPOTrainer)
    trainer.summarization_config = (
        {"enable": True} if segment_training else {"enable": False}
    )
    trainer._cached_raw_trajectories = None
    trainer._is_validation_mode = False
    trainer._val_traj_metrics_buffer = []
    return trainer


def _make_skinny_batch(seqlens: list[int], advantages_per_traj: list[float]):
    """Synthesize a per-trajectory ("skinny") DataProto-like dict.

    We don't need a real DataProto for this test — we'll just hand the
    override a MagicMock that mirrors the batch.batch["attention_mask"]
    interface. The override doesn't read any other field for the no-op
    path.
    """
    n = len(seqlens)
    T = max(seqlens) + 4  # response + a few pad slots
    P = 4
    attention_mask = torch.zeros(n, P + T, dtype=torch.long)
    advantages = torch.zeros(n, T, dtype=torch.float32)
    response_mask = torch.zeros(n, T, dtype=torch.long)
    for i, L in enumerate(seqlens):
        attention_mask[i, :P + L] = 1            # P prompt tokens + L response tokens
        response_mask[i, :L] = 1
        advantages[i, :L] = advantages_per_traj[i]

    batch_proxy = MagicMock()
    batch_proxy.batch = {
        "attention_mask": attention_mask,
        "advantages": advantages,
        "response_mask": response_mask,
    }
    return batch_proxy


def test_balance_batch_is_noop_when_segment_training_enabled():
    """The override must not call into the parent's reorder."""
    trainer = _make_trainer_stub(segment_training=True)
    batch = _make_skinny_batch(
        seqlens=[8, 12, 6, 10],   # deliberately uneven
        advantages_per_traj=[+0.9, -0.8, +0.7, -0.6],
    )
    # Snapshot the rows so we can prove they didn't move.
    am_before = batch.batch["attention_mask"].clone()
    adv_before = batch.batch["advantages"].clone()

    metrics: dict = {}
    result = trainer._balance_batch(batch, metrics)

    # Returns None (no-op signal); batch is unchanged.
    assert result is None
    assert torch.equal(batch.batch["attention_mask"], am_before)
    assert torch.equal(batch.batch["advantages"], adv_before)
    # No metrics added.
    assert metrics == {}


def test_cache_to_advantage_pairing_is_correct_after_noop_balance():
    """Positional pairing raw_trajs[i] ↔ A_i must hold when balance is no-op."""
    trainer = _make_trainer_stub(segment_training=True)
    # Cache "trajectories" (just labels for the test).
    trainer._cached_raw_trajectories = [
        {"name": f"traj{i}", "true_advantage": A}
        for i, A in enumerate([+0.9, -0.8, +0.7, -0.6])
    ]
    batch = _make_skinny_batch(
        seqlens=[8, 12, 6, 10],
        advantages_per_traj=[+0.9, -0.8, +0.7, -0.6],
    )

    # Run the override (no-op).
    trainer._balance_batch(batch, metrics={})

    # Now extract A_i exactly as _expanded_update_actor would.
    a_extracted = extract_advantage_per_trajectory(
        batch.batch["advantages"], batch.batch["response_mask"]
    )

    # Each cache entry's "true_advantage" must match the extracted A at
    # the same row index.
    for i, traj in enumerate(trainer._cached_raw_trajectories):
        assert abs(float(a_extracted[i].item()) - traj["true_advantage"]) < 1e-6, (
            f"row {i}: extracted A={float(a_extracted[i]):+.3f} "
            f"!= cache true A={traj['true_advantage']:+.3f}"
        )


def test_balance_batch_delegates_when_segment_training_disabled(monkeypatch):
    """When summarization is off, the override must delegate to the parent
    so non-segment training keeps its normal load-balancing behavior.
    """
    trainer = _make_trainer_stub(segment_training=False)

    # Patch the parent's _balance_batch on the class hierarchy so we can
    # detect the delegation. We resolve the parent (AgentPPOTrainer ->
    # RayPPOTrainer) by walking __mro__ to find the next class with
    # _balance_batch.
    from rllm.trainer.verl.agent_ppo_trainer import AgentPPOTrainer
    parent_class = None
    for cls in MultiEpisodeAgentPPOTrainer.__mro__[1:]:
        if "_balance_batch" in cls.__dict__:
            parent_class = cls
            break
    assert parent_class is not None, "expected a parent class with _balance_batch"

    called = {"hit": False}

    def fake_parent_balance(self, batch, metrics, **kwargs):
        called["hit"] = True

    monkeypatch.setattr(parent_class, "_balance_batch", fake_parent_balance)

    batch = _make_skinny_batch(seqlens=[4, 5], advantages_per_traj=[0.1, -0.1])
    trainer._balance_batch(batch, metrics={})

    assert called["hit"], "parent _balance_batch should be called when segment training is disabled"
