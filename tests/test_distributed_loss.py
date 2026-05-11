"""End-to-end GPU tests for the distributed trajectory-uniform PPO loss.

These tests verify the parts of the segment-training pipeline that are
behaviourally affected by the number of DP ranks. They spawn local
multi-rank processes via ``torch.multiprocessing.spawn`` and use NCCL
+ DDP to exercise real cross-rank gradient synchronisation.

What's tested:

  T1  Loss/gradient equivalence between 1-rank and N-rank execution.
      Validates that ``loss_micro * dp_world_size`` correctly cancels
      DDP/FSDP's post-backward grad-mean.

  T2  Padded rows (``traj_uniform_weight = 0``) contribute exactly 0
      to the gradient, regardless of where in the rank layout they land.

  T3  When a single trajectory's segments are split across ranks, the
      trajectory's total contribution to the loss is independent of the
      split. Trajectory-uniform aggregation is rank-placement-invariant.

Each test mirrors the per-token PPO surrogate + trajectory-uniform sum
aggregation from ``trainers/trajectory_uniform_actor.py:204-239``.

Implementation note: every CUDA operation runs inside a spawned worker
so the test driver itself never initializes CUDA in pytest's main
process (forking after CUDA init triggers
"CUDA-capable device(s) is/are busy or unavailable").
"""
from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


# Skip the whole module if GPUs are insufficient.
if torch.cuda.device_count() < 2:
    pytest.skip("needs >=2 GPUs for distributed tests", allow_module_level=True)


# ---------------------------------------------------------------------------
# Tiny stand-in policy: enough to produce gradients with our loss formulation.
# We deliberately avoid loading a real LM — these tests target the loss
# AGGREGATION + DDP grad-sync, not transformer correctness.
# ---------------------------------------------------------------------------

VOCAB = 64
HIDDEN = 16
SEED = 12345


class TinyPolicy(nn.Module):
    """One-layer linear stack: embed → linear → logits over a small vocab."""

    def __init__(self, vocab: int = VOCAB, hidden: int = HIDDEN):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, T) long; logits: (B, T, V)
        return self.proj(self.embed(input_ids))


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Loss formulation matching trajectory_uniform_actor.py:204-239.
# Returns the per-rank numer (a scalar tensor, summable across ranks).
# ---------------------------------------------------------------------------


def trajectory_uniform_pg_numer(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    traj_uniform_weight: torch.Tensor,
    cliprange_low: float = 0.2,
    cliprange_high: float = 0.28,
    clip_ratio_c: float = 3.0,
) -> torch.Tensor:
    """Match trajectory_uniform_actor.py:218-239 verbatim."""
    nak = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(nak)

    pg1 = -advantages * ratio
    pg2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip1 = torch.maximum(pg1, pg2)
    pg3 = -advantages * clip_ratio_c
    clip2 = torch.min(pg3, clip1)
    pg_per_token = torch.where(advantages < 0, clip2, clip1)

    return (pg_per_token * traj_uniform_weight * response_mask).sum()


# ---------------------------------------------------------------------------
# Synthetic batch builder: fixed RNG → reproducible across runs.
# ---------------------------------------------------------------------------


def make_batch(
    n_g: int,
    n_t_per_traj: list[int],
    segs_per_traj: list[int],
    *,
    T: int = 16,
    seed: int = SEED,
):
    """Build a synthetic per-segment batch on CPU."""
    assert len(n_t_per_traj) == n_g
    assert len(segs_per_traj) == n_g
    g = torch.Generator().manual_seed(seed)

    advantages_per_traj = torch.randn(n_g, generator=g)
    rows = []
    for traj_i in range(n_g):
        K_i = segs_per_traj[traj_i]
        N_t_i = n_t_per_traj[traj_i]
        weight = 1.0 / (n_g * max(N_t_i, 1))
        per_seg = [N_t_i // K_i] * K_i
        for k in range(N_t_i % K_i):
            per_seg[k] += 1
        for k, m in enumerate(per_seg):
            input_ids = torch.randint(0, VOCAB, (T,), generator=g)
            response_mask = torch.zeros(T, dtype=torch.long)
            response_mask[:m] = 1
            advantages = torch.zeros(T, dtype=torch.float32)
            advantages[:m] = advantages_per_traj[traj_i].item()
            tuw = torch.zeros(T, dtype=torch.float32)
            tuw[:m] = weight
            old_lp = torch.randn(T, generator=g) * 0.1
            rows.append({
                "input_ids": input_ids,
                "response_mask": response_mask,
                "advantages": advantages,
                "traj_uniform_weight": tuw,
                "old_log_probs": old_lp,
                "traj_idx": traj_i,
            })

    out: dict = {}
    for k in ("input_ids", "response_mask", "advantages",
              "traj_uniform_weight", "old_log_probs"):
        out[k] = torch.stack([r[k] for r in rows], dim=0)
    out["traj_idx"] = torch.tensor([r["traj_idx"] for r in rows], dtype=torch.long)
    return out


def compute_log_prob_at_response_positions(
    model: nn.Module, input_ids: torch.Tensor
) -> torch.Tensor:
    """Causal-LM next-token log_prob at every position. log_prob[t]
    corresponds to log P(input_ids[t] | input_ids[:t]). For t=0 (no
    preceding context), value is 0; the test batches always have
    response_mask=0 at t=0 so this position is never scored."""
    logits = model(input_ids)            # (B, T, V)
    log_softmax = F.log_softmax(logits, dim=-1)
    shifted = torch.cat(
        [torch.zeros_like(log_softmax[:, :1, :]), log_softmax[:, :-1, :]], dim=1
    )
    return shifted.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Worker that EVERY rank runs (including the world_size=1 baseline).
# The test driver never touches CUDA itself — it spawns workers, waits,
# reads back the gradient from a temp file.
# ---------------------------------------------------------------------------


def _worker(
    rank: int,
    world_size: int,
    init_state_path: str,
    batch_path: str,         # CPU pickle of the full batch (rebuilt on rank)
    out_path: str,           # rank 0 writes the result here
    master_port: int,
):
    """Spawned process body. Runs forward + backward + (when world_size>1)
    DDP all-reduce, captures rank-0 gradient, writes to disk."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    try:
        device = torch.device(f"cuda:{rank}")

        _seed_everything(SEED)
        model = TinyPolicy().to(device)
        model.load_state_dict(torch.load(init_state_path, map_location=device))

        if world_size > 1:
            wrapped = DDP(model, device_ids=[rank])
        else:
            wrapped = model    # no DDP for single-rank baseline

        # Reconstruct the batch on this rank.
        batch_full = torch.load(batch_path, map_location="cpu")

        # Compute this rank's slice.
        n_total = batch_full["input_ids"].shape[0]
        rows_per_rank = n_total // world_size
        start = rank * rows_per_rank
        end = start + rows_per_rank if rank < world_size - 1 else n_total

        local = {
            k: v[start:end].to(device)
            for k, v in batch_full.items()
            if isinstance(v, torch.Tensor)
        }

        # Forward pass.
        log_prob = compute_log_prob_at_response_positions(
            wrapped, local["input_ids"]
        )

        numer_local = trajectory_uniform_pg_numer(
            log_prob=log_prob,
            old_log_prob=local["old_log_probs"],
            advantages=local["advantages"],
            response_mask=local["response_mask"],
            traj_uniform_weight=local["traj_uniform_weight"],
        )

        # For multi-rank we scale by world_size to cancel DDP's grad-mean;
        # for single-rank the scale is 1 (no scaling needed).
        if world_size > 1:
            loss = numer_local * world_size
        else:
            loss = numer_local

        wrapped.zero_grad(set_to_none=True)
        loss.backward()

        if rank == 0:
            # Pull gradients off the underlying (un-wrapped) model so DDP's
            # buffer-management doesn't interfere.
            base = wrapped.module if isinstance(wrapped, DDP) else wrapped
            torch.save({
                "proj_grad": base.proj.weight.grad.detach().cpu().clone(),
                "embed_grad": base.embed.weight.grad.detach().cpu().clone(),
                "numer_local": numer_local.detach().cpu().item(),
            }, out_path)

        # Make sure all ranks finish before tearing down the process group.
        dist.barrier()
    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Driver helpers — never touch CUDA.
# ---------------------------------------------------------------------------


@contextmanager
def _temp_init_state():
    """Save a fixed initial model state_dict to disk so all ranks (and
    the single-rank baseline) start from identical parameters."""
    _seed_everything(SEED)
    model = TinyPolicy()       # CPU model; no CUDA touched here
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    torch.save(model.state_dict(), path)
    try:
        yield path
    finally:
        os.unlink(path)


def _persist_batch(batch: dict) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    torch.save(batch, path)
    return path


def _run(batch: dict, world_size: int, init_state_path: str) -> dict:
    """Spawn `world_size` ranks, return rank-0's captured gradient dict."""
    batch_path = _persist_batch(batch)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        out_path = f.name
    try:
        # Unique master port per call so back-to-back tests don't collide.
        master_port = 29500 + (os.getpid() % 1000) + world_size
        mp.spawn(
            _worker,
            args=(world_size, init_state_path, batch_path, out_path, master_port),
            nprocs=world_size,
            join=True,
        )
        return torch.load(out_path)
    finally:
        for p in (out_path, batch_path):
            if os.path.exists(p):
                os.unlink(p)


# ---------------------------------------------------------------------------
# T1: 1-rank vs 2-rank gradient equivalence.
# ---------------------------------------------------------------------------


def test_t1_gradient_equivalence_1_vs_2_rank():
    """`loss_micro * dp_world_size` must compose with DDP's grad-mean
    so that the resulting gradient equals the single-rank-full-batch
    gradient."""
    n_g = 4
    n_t = [8, 6, 4, 10]
    segs = [2, 2, 2, 2]      # 8 rows total; divisible by 2
    batch = make_batch(n_g=n_g, n_t_per_traj=n_t, segs_per_traj=segs)

    with _temp_init_state() as init_path:
        ref = _run(batch, world_size=1, init_state_path=init_path)
        ddp = _run(batch, world_size=2, init_state_path=init_path)

    proj_diff = (ref["proj_grad"] - ddp["proj_grad"]).abs()
    embed_diff = (ref["embed_grad"] - ddp["embed_grad"]).abs()

    print(f"\nT1: numer (1-rank, full)  = {ref['numer_local']:+.6f}")
    print(f"    numer (2-rank, half)  = {ddp['numer_local']:+.6f}  "
          f"(expected ≈ half of 1-rank)")
    print(f"    proj_grad max abs diff = {proj_diff.max().item():.2e}")
    print(f"    embed_grad max abs diff = {embed_diff.max().item():.2e}")

    assert proj_diff.max() < 1e-5
    assert embed_diff.max() < 1e-5


# ---------------------------------------------------------------------------
# T2: Padded rows (traj_uniform_weight=0) contribute exactly 0.
# ---------------------------------------------------------------------------


def test_t2_padded_rows_zero_contribution():
    """Replicate post-`pad_dataproto_to_divisor`: extra rows appended
    with traj_uniform_weight=0. Two different choices of pad-row data
    must give identical gradients."""
    n_g = 3
    n_t = [6, 8, 4]
    segs = [1, 1, 1]                         # 3 real rows
    real = make_batch(n_g=n_g, n_t_per_traj=n_t, segs_per_traj=segs)

    def with_pad(pad_choice_idx: int) -> dict:
        pad = {
            k: v[pad_choice_idx:pad_choice_idx + 1].clone()
            for k, v in real.items()
            if isinstance(v, torch.Tensor)
        }
        pad["traj_uniform_weight"].zero_()    # the only thing that matters
        return {
            k: torch.cat([real[k], pad[k]], dim=0)
            for k in real
            if isinstance(real[k], torch.Tensor)
        }

    batch_pad_a = with_pad(0)                # pad row = repeat of row 0
    batch_pad_b = with_pad(1)                # pad row = repeat of row 1

    with _temp_init_state() as init_path:
        ga = _run(batch_pad_a, world_size=2, init_state_path=init_path)
        gb = _run(batch_pad_b, world_size=2, init_state_path=init_path)

    diff = (ga["proj_grad"] - gb["proj_grad"]).abs()
    print(f"\nT2: proj_grad max abs diff between two pad-row choices "
          f"= {diff.max().item():.2e}")
    assert diff.max() < 1e-5, "padded rows changed the gradient"


# ---------------------------------------------------------------------------
# T3: trajectory split across ranks vs co-located on one rank.
# ---------------------------------------------------------------------------


def test_t3_trajectory_split_across_ranks_invariance():
    """The trajectory-uniform aggregation should be invariant to the
    rank-placement of a trajectory's segments. Place traj 0's two
    segments on the same rank vs split across ranks; gradients must
    match."""
    n_g = 2
    n_t = [10, 8]
    segs = [2, 2]                            # 4 rows total
    batch = make_batch(n_g=n_g, n_t_per_traj=n_t, segs_per_traj=segs)

    # Default layout: rows are [traj0-seg0, traj0-seg1, traj1-seg0, traj1-seg1].
    # With world_size=2 and even split, rank 0 gets [traj0-seg0, traj0-seg1]
    # → both segs of traj 0 on rank 0; both segs of traj 1 on rank 1.
    co_located = {k: v.clone() for k, v in batch.items()
                  if isinstance(v, torch.Tensor)}
    # Split layout: interleave so each rank gets one segment of each traj.
    # rows = [traj0-seg0, traj1-seg0, traj0-seg1, traj1-seg1]
    perm = torch.tensor([0, 2, 1, 3])
    split_across = {k: v[perm].clone() for k, v in batch.items()
                    if isinstance(v, torch.Tensor)}

    with _temp_init_state() as init_path:
        g_co = _run(co_located, world_size=2, init_state_path=init_path)
        g_sp = _run(split_across, world_size=2, init_state_path=init_path)

    diff = (g_co["proj_grad"] - g_sp["proj_grad"]).abs()
    print(f"\nT3: proj_grad max abs diff "
          f"(traj co-located vs split across ranks) = {diff.max().item():.2e}")
    assert diff.max() < 1e-5
