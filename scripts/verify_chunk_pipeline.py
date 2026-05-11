"""Step-by-step verification of the chunk-discounted-TOPR pipeline.

Runs on a synthetic trajectory (no model, no GPU) and prints every level
of the expansion so we can verify:

  Stage 1: trajectory → segments      (split at summarization boundaries)
  Stage 2: segments  → chunks         (mask=1 runs inside each segment)
  Stage 3: chunks    → advantages     (γ^Δ · R per chunk, broadcast to its tokens)
  Stage 4: chunks    → is_positive    (G_k > 0 per token)

Usage:
    conda activate icx
    python scripts/verify_chunk_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from trainers.chunk_advantage import (
    _segment_step_ranges,
    compute_chunk_returns_for_batch,
    compute_chunk_returns_for_trajectory,
)
from trainers.segment_expansion import _mask1_runs, build_expanded_dataproto


# ---------------------------------------------------------------------------
# Build a synthetic trajectory matching the user's example shape:
#   episode 0: 5 action chunks (success → R_0 = 1.0)
#   episode 1: 3 action chunks (fail    → R_1 = 0.0)
#   episode 2: 2 action chunks (success → R_2 = 1.0)
#
# Oracle summarization fires at end of episode 0 and 1 (NOT episode 2 — it's
# the last). Oracle does NOT append to episode_steps but DOES append to
# summarization_boundaries (pointing at the last action step of the episode).
# ---------------------------------------------------------------------------

EPISODE_REWARDS = [1.0, 0.0, 1.0]
EP_LENGTHS = [5, 3, 2]                       # action steps per episode
GAMMA = 0.95


def _action_step(*, episode_done: bool, completion_len: int = 4):
    """One action step's metadata + tokens for assemble_steps simulation.

    completion_len is the number of mask=1 tokens this chunk emits.
    """
    return {
        "is_summarization": False,
        "trigger": None,
        "episode_done": episode_done,
        "episode_index": 0,
        "completion_len": completion_len,
    }


def build_synthetic_trajectory():
    """Mirror the engine's output shape (mode=='Token') for our example."""
    step_metadata = []
    for ep_idx, n_actions in enumerate(EP_LENGTHS):
        for k in range(n_actions):
            step_metadata.append({
                "is_summarization": False,
                "trigger": None,
                "episode_done": (k == n_actions - 1),
                "episode_index": ep_idx,
            })

    # Oracle summarizer adds boundary pointing at last action step of each
    # episode EXCEPT the last (trajectory ends; no need to summarize).
    cum = 0
    summarization_boundaries: list[int] = []
    for n in EP_LENGTHS[:-1]:
        cum += n
        summarization_boundaries.append(cum - 1)   # last action step of that episode

    # Build per-segment tensors. Each segment's response_masks is a flat
    # 0/1 sequence: per-chunk = [0]*env_gap + [1]*completion_len, with the
    # FIRST chunk having no leading env_gap (its prompt is the segment-start
    # prompt, not a continuation).
    seg_step_ranges = _segment_step_ranges(
        n_steps=len(step_metadata),
        summarization_boundaries=summarization_boundaries,
    )
    print(f"step_metadata length = {len(step_metadata)}")
    print(f"summarization_boundaries = {summarization_boundaries}")
    print(f"_segment_step_ranges -> {seg_step_ranges}\n")

    # Each step contributes COMPLETION_LEN mask=1 tokens; between-step gaps
    # have a small ENV_GAP of mask=0 tokens (env messages).
    COMPLETION_LEN = 4
    ENV_GAP = 2

    segments = []
    for seg_idx, (start, end) in enumerate(seg_step_ranges):
        n_chunks_in_seg = end - start
        # Build response_tokens / response_masks pattern for this segment.
        masks: list[int] = []
        for chunk_idx in range(n_chunks_in_seg):
            if chunk_idx == 0:
                # First chunk in segment: completion only.
                masks.extend([1] * COMPLETION_LEN)
            else:
                # Subsequent chunks: env_gap (mask=0) + completion (mask=1).
                masks.extend([0] * ENV_GAP + [1] * COMPLETION_LEN)

        seg_len = len(masks)
        seg = {
            "prompt_tokens": torch.zeros(8, dtype=torch.long),
            "response_tokens": torch.zeros(seg_len, dtype=torch.long),
            "response_masks": torch.tensor(masks, dtype=torch.long),
            "trajectory_reward": float(sum(EPISODE_REWARDS)),
        }
        segments.append(seg)
        runs = _mask1_runs(seg["response_masks"])
        print(
            f"Segment {seg_idx}: covers steps[{start}:{end}] "
            f"({n_chunks_in_seg} chunks); "
            f"response_masks pattern length={seg_len}; "
            f"mask=1 runs={runs}"
        )

    print()
    return {
        "segments": segments,
        "step_metadata": step_metadata,
        "summarization_boundaries": summarization_boundaries,
        "episode_rewards": EPISODE_REWARDS,
    }


def main():
    print("=" * 70)
    print("STAGE 1: trajectory -> segments split")
    print("=" * 70)
    print(f"Trajectory has {len(EP_LENGTHS)} episodes with action counts "
          f"{EP_LENGTHS} (total {sum(EP_LENGTHS)} action chunks).")
    print(f"Episode rewards = {EPISODE_REWARDS} (R_total = {sum(EPISODE_REWARDS)})")
    print()

    traj = build_synthetic_trajectory()

    print("=" * 70)
    print("STAGE 2: segments -> per-chunk token ranges (via mask=1 runs)")
    print("=" * 70)
    for seg_idx, seg in enumerate(traj["segments"]):
        runs = _mask1_runs(seg["response_masks"])
        print(f"  Segment {seg_idx}: {len(runs)} chunks at token ranges {runs}")
    print()

    print("=" * 70)
    print("STAGE 3: chunk_advantage computes per-chunk G_k")
    print("=" * 70)
    per_chunk_returns = compute_chunk_returns_for_trajectory(
        traj, scope="terminal", gamma=GAMMA
    )
    print(f"Variant A (terminal, R_total={sum(EPISODE_REWARDS)}, γ={GAMMA}):")
    for k, (g, is_pos) in enumerate(per_chunk_returns):
        delta = len(per_chunk_returns) - 1 - k
        print(f"  chunk {k:2d}  Δ={delta:2d}  G_k={g:.4f}  is_positive={is_pos}")
    print()

    print("=" * 70)
    print("STAGE 4: build_expanded_dataproto fills tensors per row")
    print("=" * 70)
    expanded = build_expanded_dataproto(
        [traj],
        per_chunk_returns=[per_chunk_returns],
        emit_is_positive=True,
        pad_token_id=0,
        max_prompt_length=8,
        max_response_length=64,
    )
    data = expanded.data
    print(f"Expanded batch shape: {data.batch['advantages'].shape} "
          f"(N_segments × max_response_length)")
    print(f"segments_per_traj = {expanded.segments_per_traj}\n")

    for row_i in range(data.batch["advantages"].shape[0]):
        adv = data.batch["advantages"][row_i]
        is_pos = data.batch["is_positive"][row_i]
        mask = data.batch["response_mask"][row_i]
        runs = _mask1_runs(mask)
        print(f"Row {row_i}: {len(runs)} chunks at {runs}")
        for run_idx, (start, end) in enumerate(runs):
            chunk_g = float(adv[start].item())
            chunk_pos = float(is_pos[start].item())
            print(
                f"    chunk {run_idx} @ tokens [{start}:{end}]  "
                f"G_k={chunk_g:.4f}  is_positive={chunk_pos:.0f}  "
                f"adv tensor in range = {adv[start:end].tolist()}"
            )

    print()
    print("=" * 70)
    print("STAGE 5: cross-checks")
    print("=" * 70)

    total_chunk_runs_in_expanded = sum(
        len(_mask1_runs(data.batch["response_mask"][r]))
        for r in range(data.batch["response_mask"].shape[0])
    )
    print(f"sum of chunk runs across all rows = {total_chunk_runs_in_expanded}")
    print(f"len(per_chunk_returns)            = {len(per_chunk_returns)}")
    assert total_chunk_runs_in_expanded == len(per_chunk_returns), (
        "Mismatch — chunk count from masks should equal per_chunk_returns length"
    )
    print("✓ chunk count matches per_chunk_returns")

    print()
    print("Variant B (per_episode) for comparison:")
    per_ep = compute_chunk_returns_for_trajectory(traj, scope="per_episode", gamma=GAMMA)
    for k, (g, is_pos) in enumerate(per_ep):
        print(f"  chunk {k:2d}  G_k={g:.4f}  is_positive={is_pos}")


if __name__ == "__main__":
    main()
