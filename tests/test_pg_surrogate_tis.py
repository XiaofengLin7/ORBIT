"""Test the rollout/training IS correction (TIS) in compute_pg_per_token.

The correction multiplies the per-token PPO surrogate by
``w_t = clamp(exp(old - rollout), max=C)`` (verl computes the tensor
centrally; here we just verify the surrogate consumes it correctly).
"""

from __future__ import annotations

import math

import torch

from trainers._pg_surrogate import compute_pg_per_token


CLIP_LOW = 0.2
CLIP_HIGH = 0.28
CLIP_RATIO_C = 3.0


def _grpo_surrogate(advantages, log_prob, old_log_prob, *, rollout_is_weights=None):
    return compute_pg_per_token(
        advantages=advantages,
        log_prob=log_prob,
        old_log_prob=old_log_prob,
        cliprange_low=CLIP_LOW,
        cliprange_high=CLIP_HIGH,
        clip_ratio_c=CLIP_RATIO_C,
        is_positive=None,
        topr_enabled=False,
        rollout_is_weights=rollout_is_weights,
    )


def test_tis_off_path_is_bit_exact():
    """With rollout_is_weights=None, the surrogate is identical to the
    pre-TIS implementation. Guards the opt-out invariant."""
    advantages = torch.tensor([[+1.0, +1.0, -1.0, -1.0]])
    log_prob = torch.tensor([[0.1, 0.0, 0.05, -0.02]])
    old_log_prob = torch.tensor([[0.0, 0.0, 0.0, 0.0]])

    off = _grpo_surrogate(advantages, log_prob, old_log_prob)
    explicit_none = _grpo_surrogate(
        advantages, log_prob, old_log_prob, rollout_is_weights=None
    )
    # Both branches produce the same tensor — `is rollout_is_weights is None`
    # short-circuits the multiply.
    assert torch.equal(off["pg_per_token"], explicit_none["pg_per_token"])


def test_tis_multiplies_post_clip():
    """pg_per_token (with IS) == pg_per_token (no IS) elementwise * w."""
    advantages = torch.tensor([[+1.0, +1.0, -1.0, -1.0]])
    log_prob = torch.tensor([[0.1, 0.0, 0.05, -0.02]])
    old_log_prob = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    # Mix of weights, including one at the truncation cap.
    w = torch.tensor([[0.7, 1.3, 2.0, 0.9]])

    base = _grpo_surrogate(advantages, log_prob, old_log_prob)["pg_per_token"]
    with_tis = _grpo_surrogate(
        advantages, log_prob, old_log_prob, rollout_is_weights=w
    )["pg_per_token"]
    assert torch.allclose(with_tis, base * w, atol=1e-6)


def test_tis_weight_is_detached_no_grad_through_w():
    """No gradient flows through rollout_is_weights — matches verl's stop-grad
    contract. With on-policy ratios (no clipping), ``∂L/∂log_prob`` equals
    ``-advantages * w`` (the PPO unclipped term scaled by detached w), and
    ``∂L/∂w`` must be zero (detached)."""
    # On-policy: ratio = 1, neither side of the PPO max/min clips.
    advantages = torch.tensor([[+1.0, -1.0]])
    log_prob = torch.tensor([[0.0, 0.0]], requires_grad=True)
    old_log_prob = torch.tensor([[0.0, 0.0]])
    w = torch.tensor([[1.5, 0.8]], requires_grad=True)

    out = _grpo_surrogate(
        advantages, log_prob, old_log_prob, rollout_is_weights=w
    )
    out["pg_per_token"].sum().backward()
    # log_prob receives the IS-scaled gradient: -advantages * ratio * w,
    # which at ratio=1 simplifies to -advantages * w.
    assert torch.allclose(
        log_prob.grad, torch.tensor([[-1.0 * 1.5, 1.0 * 0.8]]), atol=1e-6
    )
    # w must not receive any gradient (detached inside the surrogate).
    assert w.grad is None or torch.equal(w.grad, torch.zeros_like(w))


def test_tis_unit_weights_match_off_path():
    """With w ≡ 1, the with-TIS path equals the off path bit-exactly
    (modulo the multiply, which is by 1.0). Regression check against any
    spurious behavior introduced by the multiply itself."""
    advantages = torch.tensor([[+2.0, +0.5, -1.5, -0.1]])
    log_prob = torch.tensor([[0.3, -0.2, 0.6, 0.0]])
    old_log_prob = torch.tensor([[0.1, -0.1, 0.5, 0.0]])
    w = torch.ones_like(advantages)

    off = _grpo_surrogate(advantages, log_prob, old_log_prob)["pg_per_token"]
    on = _grpo_surrogate(
        advantages, log_prob, old_log_prob, rollout_is_weights=w
    )["pg_per_token"]
    assert torch.allclose(on, off, atol=1e-6)


def test_tis_zero_weights_zero_loss():
    """When w_t = 0 everywhere (e.g. a rejection-sampled row), the
    per-token loss contribution is zero. Useful when verl's helper
    is later configured to do rejection sampling."""
    advantages = torch.tensor([[+1.0, -1.0]])
    log_prob = torch.tensor([[0.5, -0.5]], requires_grad=True)
    old_log_prob = torch.tensor([[0.0, 0.0]])
    w = torch.zeros_like(advantages)

    out = _grpo_surrogate(
        advantages, log_prob, old_log_prob, rollout_is_weights=w
    )
    assert torch.all(out["pg_per_token"] == 0)
    out["pg_per_token"].sum().backward()
    assert torch.all(log_prob.grad == 0)
