"""Per-token PPO surrogate computation, isolated from verl/wrapt imports.

Lives in its own module so unit tests can import the surrogate helper
without triggering the wrapt-based actor patch in
``orbit_segtrain_patch.py`` (which fires when verl's
``dp_actor`` module is imported and re-imports
``trainers.trajectory_uniform_actor``, causing a cycle when the actor
module is itself being initialized).
"""

from __future__ import annotations

import torch


__all__ = ["compute_pg_per_token"]


def compute_pg_per_token(
    advantages: torch.Tensor,
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    *,
    cliprange_low: float,
    cliprange_high: float,
    clip_ratio_c: float,
    is_positive: torch.Tensor | None = None,
    topr_enabled: bool = False,
    rollout_is_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Per-token PPO surrogate with optional TOPR split and rollout IS correction.

    Mirrors verl's standard PPO clipping (cliprange_low/high + 3-way
    clip_ratio_c floor for negative advantages) and adds the TOPR branch
    from arXiv:2512.24873 Eq. 5: tokens whose ``is_positive=True`` use a
    plain SL-style update ``-advantages`` (no IS ratio, no clip);
    everything else falls through to the standard surrogate.

    When ``rollout_is_weights`` is provided (the truncated rollout/training
    IS correction `w_t = clamp(exp(old_log_prob - rollout_log_prob), C)`
    computed by verl's helper), the final per-token surrogate is multiplied
    by ``rollout_is_weights.detach()``. Matches verl's
    ``compute_policy_loss_vanilla`` (core_algos.py:966-967): the multiplier
    is applied **after** clipping, with no gradient flowing through it.

    Returns a dict with:
        ``pg_per_token``         the surrogate (after TOPR + TIS)
        ``pg_losses1``           ``-advantages * ratio`` (unclipped term)
        ``pg_losses2``           ``-advantages * clipped_ratio``
        ``clip_pg_losses1``      ``max(pg_losses1, pg_losses2)``
        ``pg_losses3``           ``-advantages * clip_ratio_c`` (lower floor)
        ``ratio``                ``exp(clamped(log_prob - old_log_prob))``
        ``negative_approx_kl``   ``clamp(log_prob - old_log_prob, -20, 20)``
    """
    negative_approx_kl = torch.clamp(log_prob - old_log_prob, -20.0, 20.0)
    ratio = torch.exp(negative_approx_kl)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_per_token = torch.where(
        advantages < 0, clip_pg_losses2, clip_pg_losses1
    )

    if is_positive is not None and topr_enabled:
        # Positive branch: REINFORCE + truncated IS correction, matching
        # verl's ``compute_policy_loss_pure_policy_gradient`` formula
        # (verl/trainer/ppo/core_algos.py:1655-1666):
        #
        #   L = -E[stopgrad(w) · log π_θ · A],  w = π_current / π_rollout
        #
        # Here ``ratio = exp(log_prob - old_log_prob)`` is the off-policy
        # IS weight π_θ / π_θ_old. We DETACH it so it acts as a constant
        # rescaling (matching verl) and the gradient w.r.t. log_prob is
        #   ∂L/∂log_prob = -A · stopgrad(w)
        # which collapses to the standard REINFORCE gradient ``-A`` when
        # on-policy (ratio=1).
        is_pos_mask = is_positive.to(pg_per_token.device).bool()
        is_weight = ratio.detach()
        pg_pos_branch = -advantages * log_prob * is_weight
        pg_per_token = torch.where(is_pos_mask, pg_pos_branch, pg_per_token)

    if rollout_is_weights is not None:
        # Rollout/training importance-sampling correction. Trainer guard
        # ensures this only fires on the GRPO path (TOPR auto-disables).
        pg_per_token = pg_per_token * rollout_is_weights.detach()

    return {
        "pg_per_token": pg_per_token,
        "pg_losses1": pg_losses1,
        "pg_losses2": pg_losses2,
        "clip_pg_losses1": clip_pg_losses1,
        "pg_losses3": pg_losses3,
        "ratio": ratio,
        "negative_approx_kl": negative_approx_kl,
    }
