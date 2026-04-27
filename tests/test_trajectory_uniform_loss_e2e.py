"""End-to-end verification that the trajectory-uniform PPO loss is computed
correctly from synthetic trajectories through the full pipeline:

    raw trajectories
       └─> build_expanded_dataproto    (segment_expansion.py)
            └─> traj_uniform_weight tensor
                 └─> per-token PPO surrogate × traj_uniform_weight × response_mask
                      └─> sum  ===  L = (1/N_G) Σ_i (1/N_t^i) Σ_token ℓ_token

We replicate the exact loss math from
:meth:`trainers.trajectory_uniform_actor.TrajectoryUniformPPOActor.update_policy`
inline (so we don't depend on FSDP / Ray / actual model forward), and assert
that the result matches the closed-form trajectory-uniform definition computed
independently via a reference loop.

The math under test (mirrors trajectory_uniform_actor.py:201-222):

    ratio_t          = exp(log_prob_t - old_log_prob_t)
    pg_losses1_t     = -A_i * ratio_t                       (no clip)
    pg_losses2_t     = -A_i * clip(ratio_t, 1-lo, 1+hi)
    clip_pg_losses1_t= max(pg_losses1_t, pg_losses2_t)
    pg_losses3_t     = -A_i * clip_ratio_c
    clip_pg_losses2_t= min(pg_losses3_t, clip_pg_losses1_t)
    pg_per_token_t   = clip_pg_losses2_t  if A_i<0 else clip_pg_losses1_t
    numer            = Σ_t (pg_per_token_t * w_t * mask_t)

with ``w_t = 1/(N_G·N_t^i)`` for valid tokens of trajectory ``i``, 0 elsewhere.

Reference closed form:

    L_target = (1/N_G) * Σ_i (1/N_t^i) * Σ_{token in trajectory i, mask=1} pg_per_token

These two must be equal.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from trainers.segment_expansion import build_expanded_dataproto


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _make_segment(prompt_len: int, response_len: int, n_valid_tokens: int,
                  seed: int = 0) -> dict:
    """Build a synthetic segment dict in the format the engine emits.

    The first ``n_valid_tokens`` of the response have mask=1; the rest are 0.
    Prompt + response token IDs are arbitrary (just have to be longs).
    """
    g = torch.Generator().manual_seed(seed)
    return {
        "prompt_tokens": torch.randint(1, 1000, (prompt_len,), generator=g),
        "response_tokens": torch.randint(1, 1000, (response_len,), generator=g),
        "response_masks": torch.cat([
            torch.ones(n_valid_tokens, dtype=torch.long),
            torch.zeros(response_len - n_valid_tokens, dtype=torch.long),
        ]),
    }


def _ppo_per_token(log_prob: torch.Tensor,
                   old_log_prob: torch.Tensor,
                   advantages: torch.Tensor,
                   clip_low: float = 0.2,
                   clip_high: float = 0.28,
                   clip_ratio_c: float = 3.0) -> torch.Tensor:
    """Mirror trajectory_uniform_actor.py:201-216 exactly."""
    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    pg1 = -advantages * ratio
    pg2 = -advantages * torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
    clip_pg1 = torch.maximum(pg1, pg2)
    pg3 = -advantages * clip_ratio_c
    clip_pg2 = torch.min(pg3, clip_pg1)
    pg_per_token = torch.where(advantages < 0, clip_pg2, clip_pg1)
    return pg_per_token


def _reference_loss(pg_per_token: torch.Tensor,
                    response_mask: torch.Tensor,
                    traj_idx: np.ndarray,
                    n_g: int) -> float:
    """Closed-form L = (1/N_G) Σ_i (1/N_t^i) Σ_token (pg * mask).

    ``traj_idx``: per-row trajectory id (0..N_G-1).
    ``response_mask``: shape (n_rows, T), values in {0, 1}.
    """
    L_per_traj = []
    for i in range(n_g):
        rows_for_i = np.where(traj_idx == i)[0]
        if len(rows_for_i) == 0:
            L_per_traj.append(0.0)
            continue
        # All segment-row tokens of trajectory i.
        pg_i = pg_per_token[rows_for_i]
        mask_i = response_mask[rows_for_i]
        n_t_i = mask_i.sum().item()
        if n_t_i == 0:
            L_per_traj.append(0.0)
            continue
        L_i = (pg_i * mask_i).sum().item() / n_t_i
        L_per_traj.append(L_i)
    return sum(L_per_traj) / n_g


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def _scenario(scenario_name: str,
              segments_per_traj: list[int],
              tokens_per_segment: list[list[int]]):
    """Build (trajectories, advantages_per_traj) for a scenario."""
    assert len(segments_per_traj) == len(tokens_per_segment)
    trajectories = []
    advantages_per_traj = []
    for traj_i, n_segs in enumerate(segments_per_traj):
        # Random per-trajectory advantage in [-1, 1].
        a_i = (-1.0) ** traj_i * (0.3 + 0.1 * traj_i)
        advantages_per_traj.append(a_i)
        segs = [
            _make_segment(prompt_len=4, response_len=10,
                          n_valid_tokens=tokens_per_segment[traj_i][k],
                          seed=traj_i * 100 + k)
            for k in range(n_segs)
        ]
        traj = {
            "prompt_tokens": segs[0]["prompt_tokens"],
            "response_tokens": segs[0]["response_tokens"],
            "response_masks": segs[0]["response_masks"],
            "trajectory_reward": float(a_i),
            "segments": segs,
        }
        trajectories.append(traj)
    return trajectories, torch.tensor(advantages_per_traj, dtype=torch.float32)


@pytest.mark.parametrize("scenario", [
    # name, segments per trajectory, mask=1 tokens per segment
    ("two_single_segment_trajs", [1, 1], [[5], [7]]),
    ("multi_segment_mixed",       [1, 3, 1], [[6], [4, 5, 3], [8]]),
    ("five_segments_uneven",      [5], [[2, 3, 4, 5, 6]]),
    ("k_equals_one_everywhere",   [1, 1, 1, 1], [[3], [4], [5], [2]]),
])
def test_trajectory_uniform_loss_matches_formula(scenario):
    """build_expanded → ppo per-token → weighted-sum  ==  closed-form formula."""
    name, segments_per_traj, tokens_per_segment = scenario
    n_g = len(segments_per_traj)

    trajectories, advantages_per_traj = _scenario(name, segments_per_traj,
                                                  tokens_per_segment)
    expansion = build_expanded_dataproto(
        trajectories,
        advantages_per_trajectory=advantages_per_traj,
        pad_token_id=0,
        max_prompt_length=8,
        max_response_length=12,
    )
    batch = expansion.data
    traj_idx = expansion.source_idx_map

    response_mask = batch.batch["response_mask"].float()
    advantages = batch.batch["advantages"]
    traj_uniform_weight = batch.batch["traj_uniform_weight"]

    # Synthesize log_prob and old_log_prob: nontrivial, non-equal, deterministic.
    g = torch.Generator().manual_seed(42)
    n_rows, T = response_mask.shape
    old_log_prob = torch.randn(n_rows, T, generator=g) * 0.1
    log_prob = old_log_prob + torch.randn(n_rows, T, generator=g) * 0.05

    pg_per_token = _ppo_per_token(log_prob, old_log_prob, advantages)

    # Actor's aggregation (mirrors trajectory_uniform_actor.py:222).
    numer = (pg_per_token * traj_uniform_weight * response_mask).sum().item()

    # Closed-form reference.
    L_target = _reference_loss(pg_per_token, response_mask, traj_idx, n_g)

    assert numer == pytest.approx(L_target, abs=1e-6), (
        f"[{name}] actor sum = {numer:.8f} vs reference L = {L_target:.8f}"
    )


def test_zero_token_trajectory_does_not_blow_up():
    """If a trajectory has segments but all-zero masks, traj_uniform_weight is
    0 everywhere for that trajectory and it contributes 0 to the loss."""
    trajectories = [
        # Trajectory 0: normal.
        {
            "prompt_tokens": torch.randint(1, 100, (4,)),
            "response_tokens": torch.randint(1, 100, (10,)),
            "response_masks": torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
            "trajectory_reward": 0.5,
            "segments": [{
                "prompt_tokens": torch.randint(1, 100, (4,)),
                "response_tokens": torch.randint(1, 100, (10,)),
                "response_masks": torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
            }],
        },
        # Trajectory 1: all-zero mask (no valid tokens).
        {
            "prompt_tokens": torch.randint(1, 100, (4,)),
            "response_tokens": torch.randint(1, 100, (10,)),
            "response_masks": torch.zeros(10, dtype=torch.long),
            "trajectory_reward": -0.3,
            "segments": [{
                "prompt_tokens": torch.randint(1, 100, (4,)),
                "response_tokens": torch.randint(1, 100, (10,)),
                "response_masks": torch.zeros(10, dtype=torch.long),
            }],
        },
    ]
    advantages_per_traj = torch.tensor([0.5, -0.3])
    expansion = build_expanded_dataproto(
        trajectories,
        advantages_per_trajectory=advantages_per_traj,
        pad_token_id=0,
        max_prompt_length=8,
        max_response_length=12,
    )
    batch = expansion.data

    response_mask = batch.batch["response_mask"].float()
    traj_uniform_weight = batch.batch["traj_uniform_weight"]

    # Trajectory 1's row should have all-zero weight (since its valid count is
    # zero, and we clamp the denominator to 1, but the mask is all 0).
    weighted = traj_uniform_weight * response_mask
    assert (weighted[1] == 0).all(), "all-zero-mask trajectory should contribute 0"

    # Trajectory 0's contribution should be (1/N_G) * (sum_t mask) / N_t^0
    # = (1/2) * 5 / 5 = 0.5. With pg = 1 everywhere, weighted sum = 0.5.
    pg = torch.ones_like(traj_uniform_weight)
    numer = (pg * traj_uniform_weight * response_mask).sum().item()
    expected = 0.5  # only trajectory 0 contributes; its rows weighted-mean is 1.
    assert numer == pytest.approx(expected, abs=1e-6)


def test_padding_rows_contribute_zero():
    """If we manually zero traj_uniform_weight on padded rows (as the trainer
    does), those rows must add 0 to the loss regardless of their content."""
    trajectories, advantages_per_traj = _scenario(
        "test_pad", [1, 1], [[3], [4]]
    )
    expansion = build_expanded_dataproto(
        trajectories,
        advantages_per_trajectory=advantages_per_traj,
        pad_token_id=0,
        max_prompt_length=8,
        max_response_length=12,
    )
    batch = expansion.data

    # Simulate the trainer's pad-then-zero-weight behavior.
    # We pad by repeating row 0; the padded row's traj_uniform_weight gets 0.
    n_rows = batch.batch["response_mask"].shape[0]

    # Manually duplicate row 0 to simulate a padded row, then zero its weight.
    response_mask = torch.cat([batch.batch["response_mask"].float(),
                               batch.batch["response_mask"][:1].float()], dim=0)
    traj_uniform_weight = torch.cat([batch.batch["traj_uniform_weight"],
                                     batch.batch["traj_uniform_weight"][:1]], dim=0)
    traj_uniform_weight[-1] = 0.0  # zero the padded row's weight

    # Synthesize pg_per_token = 1 everywhere.
    pg = torch.ones_like(traj_uniform_weight)
    numer_with_pad = (pg * traj_uniform_weight * response_mask).sum().item()

    # Compare against original (un-padded) loss.
    pg_orig = torch.ones_like(batch.batch["traj_uniform_weight"])
    numer_orig = (
        pg_orig * batch.batch["traj_uniform_weight"] * batch.batch["response_mask"].float()
    ).sum().item()

    assert numer_with_pad == pytest.approx(numer_orig, abs=1e-6), (
        "padding row must contribute 0 to the loss"
    )


# ---------------------------------------------------------------------
# Verification tests for production loopholes (see plan §6 in repo)
# ---------------------------------------------------------------------

@pytest.mark.parametrize("scenario,n_chunks", [
    # (segments_per_traj, tokens_per_segment), n_chunks for chunking.
    # All scenarios are designed so n_rows is divisible by n_chunks AND, where
    # K>1, at least one trajectory's segment rows STRADDLE a chunk boundary
    # (the actual L5 stress test).
    (([1, 1, 1, 1], [[3], [4], [5], [2]]), 2),         # 4 rows, all K=1; 2 chunks
    (([2, 2, 2], [[3, 4], [5, 6], [2, 3]]), 2),         # 6 rows, 2 chunks of 3 → traj 1 SPLIT
    (([2, 2, 2], [[3, 4], [5, 6], [2, 3]]), 3),         # 6 rows, 3 chunks of 2
    (([3, 1], [[2, 3, 4], [5]]), 2),                    # 4 rows, 2 chunks of 2 → traj 0 SPLIT
    (([3, 3], [[2, 3, 4], [5, 6, 7]]), 3),              # 6 rows, 3 chunks of 2 → both SPLIT
])
def test_global_sum_correct_under_chunking(scenario, n_chunks):
    """L5: simulate Ray's `DataProto.chunk(world_size)` row-based dispatch.

    A trajectory's segment rows can land on different ranks. Each rank computes
    its local `numer = sum(pg * w * mask)`, then we sum across ranks. The result
    must equal the closed-form trajectory-uniform formula computed on the full
    (un-chunked) batch.

    This validates that our weight bake-in (`w = 1/(N_G·N_t^i)` with global
    `N_t^i`) is correct under cross-rank dispatch.
    """
    (segments_per_traj, tokens_per_segment) = scenario
    n_g = len(segments_per_traj)
    trajectories, advantages_per_traj = _scenario(
        "chunking_test", segments_per_traj, tokens_per_segment
    )
    expansion = build_expanded_dataproto(
        trajectories,
        advantages_per_trajectory=advantages_per_traj,
        pad_token_id=0,
        max_prompt_length=8,
        max_response_length=12,
    )
    batch = expansion.data
    traj_idx = expansion.source_idx_map

    response_mask = batch.batch["response_mask"].float()
    advantages = batch.batch["advantages"]
    traj_uniform_weight = batch.batch["traj_uniform_weight"]

    # Synthesize per-token PPO surrogate.
    g = torch.Generator().manual_seed(7)
    n_rows, T = response_mask.shape
    old_log_prob = torch.randn(n_rows, T, generator=g) * 0.1
    log_prob = old_log_prob + torch.randn(n_rows, T, generator=g) * 0.05
    pg_per_token = _ppo_per_token(log_prob, old_log_prob, advantages)

    # Skip if not divisible (matches verl's chunk assertion behavior — the
    # trainer pads to ensure divisibility, but in this test we want to verify
    # the math, not the padding).
    if n_rows % n_chunks != 0:
        pytest.skip(f"{n_rows} rows not divisible by {n_chunks} chunks")

    # Per-rank local numer (mirrors what each DP rank computes).
    rows_per_chunk = n_rows // n_chunks
    per_chunk_numer = []
    for c in range(n_chunks):
        start, end = c * rows_per_chunk, (c + 1) * rows_per_chunk
        local_pg = pg_per_token[start:end]
        local_w = traj_uniform_weight[start:end]
        local_mask = response_mask[start:end]
        per_chunk_numer.append((local_pg * local_w * local_mask).sum().item())

    # Global sum across ranks (this is what _all_reduce_sum would produce).
    global_numer = sum(per_chunk_numer)

    # Closed-form reference on the full batch.
    L_target = _reference_loss(pg_per_token, response_mask, traj_idx, n_g)

    assert global_numer == pytest.approx(L_target, abs=1e-6), (
        f"chunked-then-summed numer={global_numer:.8f} vs closed form={L_target:.8f}"
    )


@pytest.mark.parametrize("scenario,n_minibatches", [
    (([2, 2, 2], [[3, 4], [5, 6], [2, 3]]), 2),         # 6 rows, 2 mini-batches
    (([2, 2, 2], [[3, 4], [5, 6], [2, 3]]), 3),         # 6 rows, 3 mini-batches
    (([1, 1, 1, 1, 1, 1], [[3], [4], [5], [2], [3], [4]]), 3),
    (([3, 3], [[2, 3, 4], [5, 6, 7]]), 2),              # K>1, splits straddle mb boundary
])
def test_minibatch_gradient_sums_to_full_batch_gradient(scenario, n_minibatches):
    """L4: gradient is linear in the loss → mini-batch SGD's accumulated
    gradient over one epoch equals the full-batch gradient.

    We synthesize `pg_per_token = log_prob` (so the "pg" depends linearly on
    a parameter `log_prob`), compute the gradient of `numer = sum(pg·w·mask)`
    once on the full batch, then split into mini-batches and compute the
    sum-of-per-mini-batch gradients. The two must match exactly (the loss is
    a sum, gradient is linear, so accumulating gradients across mini-batches
    is equivalent to one full-batch backward pass).

    This validates that the mini-batch SGD in
    ``TrajectoryUniformPPOActor.update_policy`` (one ``optimizer.step()`` per
    mini-batch) does NOT change the *gradient direction per epoch* — it just
    takes more steps along that gradient. Convergence speed differs, but the
    trajectory-uniform aggregation semantics are preserved.
    """
    (segments_per_traj, tokens_per_segment) = scenario
    trajectories, advantages_per_traj = _scenario(
        "minibatch_test", segments_per_traj, tokens_per_segment
    )
    expansion = build_expanded_dataproto(
        trajectories,
        advantages_per_trajectory=advantages_per_traj,
        pad_token_id=0,
        max_prompt_length=8,
        max_response_length=12,
    )
    batch = expansion.data
    n_rows = batch.batch["response_mask"].shape[0]

    if n_rows % n_minibatches != 0:
        pytest.skip(f"{n_rows} rows not divisible by {n_minibatches} mini-batches")

    response_mask = batch.batch["response_mask"].float()
    traj_uniform_weight = batch.batch["traj_uniform_weight"]

    # Full-batch: gradient w.r.t. log_prob.
    log_prob_full = torch.zeros_like(traj_uniform_weight, requires_grad=True)
    pg_full = log_prob_full  # treat pg = log_prob for linearity test
    numer_full = (pg_full * traj_uniform_weight * response_mask).sum()
    numer_full.backward()
    grad_full = log_prob_full.grad.clone()

    # Mini-batched: same loss, but split + accumulate gradients across mb.
    log_prob_mb = torch.zeros_like(traj_uniform_weight, requires_grad=True)
    rows_per_mb = n_rows // n_minibatches
    for c in range(n_minibatches):
        start, end = c * rows_per_mb, (c + 1) * rows_per_mb
        # Slice the parameter (preserve grad graph by using torch.narrow logic).
        # We can't slice and keep gradients flowing both ways, so rebuild:
        pg_mb_slice = log_prob_mb[start:end]
        w_slice = traj_uniform_weight[start:end]
        mask_slice = response_mask[start:end]
        numer_mb = (pg_mb_slice * w_slice * mask_slice).sum()
        numer_mb.backward()
    grad_mb_sum = log_prob_mb.grad.clone()

    assert torch.allclose(grad_full, grad_mb_sum, atol=1e-7), (
        f"full-batch gradient sum != per-mini-batch gradient sum; max diff = "
        f"{(grad_full - grad_mb_sum).abs().max().item():.2e}"
    )


def test_loss_invariant_under_segment_split():
    """T4: trajectory-uniform loss depends only on (N_t^i, A_i) per trajectory,
    not on how those tokens are partitioned into segments.

    A single trajectory with 8 mask=1 tokens in one segment, vs the same 8 tokens
    split as 4+4 across two segments, must yield identical loss values when fed
    the same per-token pg values.
    """
    # Configuration shared by both scenarios.
    n_total_tokens = 8
    advantage = 0.5

    # Scenario A: 1 segment of 8 tokens.
    traj_a = [
        {
            "prompt_tokens": torch.randint(1, 1000, (4,)),
            "response_tokens": torch.randint(1, 1000, (10,)),
            "response_masks": torch.tensor([1] * n_total_tokens + [0] * (10 - n_total_tokens)),
            "trajectory_reward": advantage,
            "segments": [{
                "prompt_tokens": torch.randint(1, 1000, (4,)),
                "response_tokens": torch.randint(1, 1000, (10,)),
                "response_masks": torch.tensor([1] * n_total_tokens + [0] * (10 - n_total_tokens)),
            }],
        },
    ]

    # Scenario B: 2 segments of 4 tokens each (same total).
    traj_b = [
        {
            "prompt_tokens": torch.randint(1, 1000, (4,)),
            "response_tokens": torch.randint(1, 1000, (10,)),
            "response_masks": torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
            "trajectory_reward": advantage,
            "segments": [
                {
                    "prompt_tokens": torch.randint(1, 1000, (4,)),
                    "response_tokens": torch.randint(1, 1000, (10,)),
                    "response_masks": torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
                },
                {
                    "prompt_tokens": torch.randint(1, 1000, (4,)),
                    "response_tokens": torch.randint(1, 1000, (10,)),
                    "response_masks": torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
                },
            ],
        },
    ]

    advantages_per_traj = torch.tensor([advantage], dtype=torch.float32)

    # Build expanded batches and compute losses with constant pg=1.
    losses = []
    for traj in (traj_a, traj_b):
        expansion = build_expanded_dataproto(
            traj,
            advantages_per_trajectory=advantages_per_traj,
            pad_token_id=0,
            max_prompt_length=8,
            max_response_length=12,
        )
        batch = expansion.data
        response_mask = batch.batch["response_mask"].float()
        weight = batch.batch["traj_uniform_weight"]
        # Constant per-token pg = 1.
        pg = torch.ones_like(weight)
        loss = (pg * weight * response_mask).sum().item()
        losses.append(loss)

    assert losses[0] == pytest.approx(losses[1], abs=1e-7), (
        f"single-segment loss {losses[0]} != two-segment loss {losses[1]}; "
        "the trajectory-uniform aggregation is supposed to depend only on the "
        "TOTAL mask=1 token count per trajectory (N_t^i), not on segment count."
    )
