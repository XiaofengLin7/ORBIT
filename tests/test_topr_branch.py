"""Test the TOPR positive/negative split in TrajectoryUniformPPOActor.

We test the surrogate helper ``compute_pg_per_token`` directly — it is
the exact code path the actor's ``update_policy`` runs, just without
the Ray/FSDP scaffolding around it.
"""

from __future__ import annotations

import pytest
import torch

# Import directly from the surrogate helper module — avoids triggering
# the verl/wrapt patch chain that fires when ``trainers.trajectory_uniform_actor``
# is imported in a non-Ray context.
from trainers._pg_surrogate import compute_pg_per_token


CLIP_LOW = 0.2
CLIP_HIGH = 0.28
CLIP_RATIO_C = 3.0


def _surrogate(advantages, log_prob, old_log_prob, *, is_positive=None, topr=False):
    return compute_pg_per_token(
        advantages=advantages,
        log_prob=log_prob,
        old_log_prob=old_log_prob,
        cliprange_low=CLIP_LOW,
        cliprange_high=CLIP_HIGH,
        clip_ratio_c=CLIP_RATIO_C,
        is_positive=is_positive,
        topr_enabled=topr,
    )


def test_topr_off_matches_standard_ppo():
    """With topr_enabled=False, output equals the standard PPO surrogate."""
    advantages = torch.tensor([[+1.0, +1.0, -1.0, -1.0]])
    log_prob = torch.tensor([[0.1, 0.0, 0.05, -0.02]])
    old_log_prob = torch.tensor([[0.0, 0.0, 0.0, 0.0]])

    no_topr = _surrogate(advantages, log_prob, old_log_prob, topr=False)
    # Identical to passing is_positive=None too.
    none_pos = _surrogate(advantages, log_prob, old_log_prob, is_positive=None, topr=True)
    assert torch.allclose(no_topr["pg_per_token"], none_pos["pg_per_token"])


def test_topr_positive_rows_use_weighted_sl_loss_with_is():
    """Positive branch loss is ``-advantages * log_prob * stopgrad(ratio)``
    — REINFORCE + truncated IS correction (matches verl's REINFORCE+IS
    formula in core_algos.py). The gradient w.r.t. log_prob is
    ``-advantages * ratio`` (with ratio detached → constant rescale), and
    the loss must include log_prob so backprop reaches policy params."""
    import math
    advantages = torch.tensor([[+1.0, +1.0, -1.0, -1.0]])
    log_prob = torch.tensor([[0.5, -0.5, 0.5, -0.5]])
    old_log_prob = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    is_positive = torch.tensor([[True, False, True, False]])

    out = _surrogate(advantages, log_prob, old_log_prob, is_positive=is_positive, topr=True)
    pg = out["pg_per_token"]

    # Positive rows (cols 0, 2): pg = -advantages * log_prob * ratio.
    # ratio[0,0] = exp(0.5 - 0.0) = e^0.5
    # col 0: -(+1.0) * 0.5 * e^0.5 = -0.5 * e^0.5
    # col 2: -(-1.0) * 0.5 * e^0.5 = 0.5 * e^0.5
    expected_0 = -1.0 * 0.5 * math.exp(0.5)
    expected_2 = -(-1.0) * 0.5 * math.exp(0.5)
    assert torch.allclose(pg[0, 0], torch.tensor(expected_0), atol=1e-5)
    assert torch.allclose(pg[0, 2], torch.tensor(expected_2), atol=1e-5)

    # Negative rows (cols 1, 3): pg_per_token follows the standard
    # 3-way-clipped surrogate.
    standard = _surrogate(advantages, log_prob, old_log_prob, topr=False)
    assert torch.allclose(pg[0, 1], standard["pg_per_token"][0, 1])
    assert torch.allclose(pg[0, 3], standard["pg_per_token"][0, 3])


def test_topr_positive_branch_gradient_flows_through_log_prob():
    """Backprop sanity: ∂L/∂log_prob == -advantages * stopgrad(ratio)
    on positive rows. Ratio is detached → constant scaling factor. At
    on-policy (ratio=1), this reduces to standard REINFORCE -advantages."""
    advantages = torch.tensor([[+2.0, -3.0, +0.5]])
    log_prob = torch.tensor([[0.4, -0.2, 0.1]], requires_grad=True)
    old_log_prob = torch.tensor([[0.4, -0.2, 0.1]])  # on-policy → ratio=1
    is_positive = torch.tensor([[True, False, True]])

    out = _surrogate(advantages, log_prob, old_log_prob, is_positive=is_positive, topr=True)
    out["pg_per_token"].sum().backward()
    grad = log_prob.grad
    # Positive rows on-policy: ∂(-adv * log_prob * 1)/∂log_prob = -advantages.
    assert torch.allclose(grad[0, 0], torch.tensor(-2.0))
    assert torch.allclose(grad[0, 2], torch.tensor(-0.5))
    # Negative row at ratio=1 with adv<0: 3-way-clip path; floor at adv·c
    # doesn't bind (ratio < c), so ∂/∂log_prob = -adv * ratio = -adv = 3.
    assert torch.allclose(grad[0, 1], torch.tensor(3.0))


def test_topr_positive_branch_is_correction_off_policy():
    """Off-policy: gradient scales by stopgrad(ratio). Verifies the IS
    weight is correctly applied as a constant factor (no in-graph
    contribution to gradient)."""
    import math
    advantages = torch.tensor([[+2.0, +0.5]])
    log_prob = torch.tensor([[1.0, -0.5]], requires_grad=True)
    old_log_prob = torch.tensor([[0.0, 0.0]])  # ratio = e^1, e^-0.5
    is_positive = torch.tensor([[True, True]])

    out = _surrogate(advantages, log_prob, old_log_prob, is_positive=is_positive, topr=True)
    out["pg_per_token"].sum().backward()

    # Expected gradients: ∂/∂log_prob = -advantages * ratio (ratio detached).
    expected_grad_0 = -2.0 * math.exp(1.0)
    expected_grad_1 = -0.5 * math.exp(-0.5)
    assert torch.allclose(log_prob.grad[0, 0], torch.tensor(expected_grad_0), atol=1e-5)
    assert torch.allclose(log_prob.grad[0, 1], torch.tensor(expected_grad_1), atol=1e-5)


def test_topr_on_policy_positive_matches_reinforce_gradient():
    """At ratio=1 (on-policy), the positive branch's gradient w.r.t.
    log_prob is -advantages — exactly the REINFORCE gradient — and
    matches what the standard PPO surrogate would give at ratio=1."""
    advantages = torch.tensor([[+0.5, +1.0, -0.3]])
    log_prob = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    old_log_prob = torch.tensor([[0.0, 0.0, 0.0]])
    is_positive = torch.tensor([[True, True, False]])

    on = _surrogate(advantages, log_prob, old_log_prob, is_positive=is_positive, topr=True)
    on["pg_per_token"].sum().backward()
    grad_topr = log_prob.grad.clone()
    log_prob.grad.zero_()

    log_prob_off = log_prob.detach().clone().requires_grad_(True)
    off = _surrogate(advantages, log_prob_off, old_log_prob, topr=False)
    off["pg_per_token"].sum().backward()
    grad_std = log_prob_off.grad

    # Both should give -advantages as gradient (REINFORCE) at ratio=1.
    expected = -advantages
    assert torch.allclose(grad_topr, expected, atol=1e-6)
    assert torch.allclose(grad_std, expected, atol=1e-6)


def test_topr_disabled_when_is_positive_absent():
    """If is_positive is None, the topr_enabled flag has no effect."""
    advantages = torch.tensor([[+1.0, -1.0]])
    log_prob = torch.tensor([[0.5, 0.5]])
    old_log_prob = torch.tensor([[0.0, 0.0]])

    standard = _surrogate(advantages, log_prob, old_log_prob, topr=False)
    no_pos_but_topr = _surrogate(
        advantages, log_prob, old_log_prob, is_positive=None, topr=True
    )
    assert torch.allclose(standard["pg_per_token"], no_pos_but_topr["pg_per_token"])


def test_topr_negative_branch_uses_three_way_clip():
    """Negative-advantage tokens (G_k ≤ 0) follow the existing 3-way
    surrogate even when TOPR is enabled — the inference/training
    mismatch handling stays as it is."""
    # Pick ratio far enough above clip_ratio_c that the lower floor bites.
    advantages = torch.tensor([[-1.0]])
    log_prob = torch.tensor([[5.0]])      # large positive log-prob diff
    old_log_prob = torch.tensor([[0.0]])
    is_positive = torch.tensor([[False]])  # negative chunk

    out = _surrogate(advantages, log_prob, old_log_prob, is_positive=is_positive, topr=True)
    pg = out["pg_per_token"]
    # 3-way clip path: pg_losses3 = -adv * clip_ratio_c = 3.0
    # clip_pg_losses2 = min(pg_losses3, clip_pg_losses1) — should equal 3.0
    # since the unclipped/clipped maxes blow up at high ratio.
    assert pg.item() == pytest.approx(3.0, abs=1e-5)
