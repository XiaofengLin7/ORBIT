"""Tests for rollout_log_probs flowing through segment expansion.

Verifies:
* When every segment carries `rollout_log_probs`, build_expanded_dataproto
  emits a `(n_rows, max_response_length)` `rollout_log_probs` tensor padded
  identically to `responses` / `response_mask`.
* When any segment omits the field, the tensor is NOT emitted (avoids
  garbage zero-fills reaching verl's TIS helper).
* Per-segment values land at the right positions across a 2-episode +
  summarization trajectory.
"""

from __future__ import annotations

import numpy as np
import torch

from trainers.segment_expansion import build_expanded_dataproto


def _seg(prompt_len: int, resp_len: int, mask_count: int,
         rlp_vals: list[float] | None) -> dict:
    """Synthetic segment, optionally with a rollout_log_probs tensor."""
    seg = {
        "prompt_tokens": torch.arange(1, prompt_len + 1, dtype=torch.long),
        "response_tokens": torch.arange(100, 100 + resp_len, dtype=torch.long),
        "response_masks": torch.cat([
            torch.ones(mask_count, dtype=torch.long),
            torch.zeros(max(0, resp_len - mask_count), dtype=torch.long),
        ]),
    }
    if rlp_vals is not None:
        assert len(rlp_vals) == resp_len, (
            f"rlp_vals must match resp_len; got {len(rlp_vals)} vs {resp_len}"
        )
        seg["rollout_log_probs"] = torch.tensor(rlp_vals, dtype=torch.float32)
    return seg


def _traj(segments: list[dict]) -> dict:
    return {
        "prompt_tokens": segments[0]["prompt_tokens"],
        "response_tokens": segments[0]["response_tokens"],
        "response_masks": segments[0]["response_masks"],
        "rollout_log_probs": segments[0].get("rollout_log_probs"),
        "segments": segments,
    }


PAD_ID = 0
MAX_PROMPT = 8
MAX_RESP = 8


def test_rollout_log_probs_emitted_when_all_segments_have_it():
    """One trajectory, two segments, both carry rollout_log_probs."""
    trajectories = [
        _traj([
            _seg(3, 5, 4, [-0.1, -0.2, -0.3, -0.4, 0.0]),
            _seg(3, 4, 3, [-0.5, -0.6, -0.7, 0.0]),
        ]),
    ]
    advs = torch.tensor([0.5])
    expansion = build_expanded_dataproto(
        trajectories,
        advantages_per_trajectory=advs,
        pad_token_id=PAD_ID,
        max_prompt_length=MAX_PROMPT,
        max_response_length=MAX_RESP,
    )
    rlp = expansion.data.batch["rollout_log_probs"]
    assert rlp.shape == (2, MAX_RESP)
    # Segment 0: first 5 positions filled with the segment's logp values,
    # remaining padded with zeros.
    assert torch.allclose(
        rlp[0, :5], torch.tensor([-0.1, -0.2, -0.3, -0.4, 0.0])
    )
    assert torch.all(rlp[0, 5:] == 0)
    # Segment 1: first 4 positions, rest zero.
    assert torch.allclose(
        rlp[1, :4], torch.tensor([-0.5, -0.6, -0.7, 0.0])
    )
    assert torch.all(rlp[1, 4:] == 0)


def test_rollout_log_probs_skipped_when_any_segment_missing():
    """If one segment lacks rollout_log_probs, the tensor is not emitted —
    prevents zero-fills from reaching verl's TIS helper and producing
    garbage IS weights."""
    trajectories = [
        _traj([
            _seg(3, 4, 3, [-0.1, -0.2, -0.3, 0.0]),
            _seg(3, 4, 3, rlp_vals=None),  # missing
        ]),
    ]
    advs = torch.tensor([0.5])
    expansion = build_expanded_dataproto(
        trajectories,
        advantages_per_trajectory=advs,
        pad_token_id=PAD_ID,
        max_prompt_length=MAX_PROMPT,
        max_response_length=MAX_RESP,
    )
    assert "rollout_log_probs" not in expansion.data.batch.keys()


def test_rollout_log_probs_aligned_with_response_mask():
    """Two-episode + 1-summarization trajectory: each segment's tensor
    should land at the same positions as its response_mask, and zeros
    elsewhere — i.e. the rollout_log_probs slice exactly mirrors the
    response_mask layout."""
    # Trajectory: episode 1 (resp_len=4, mask=4) + summary boundary →
    # segment 1, then episode 2 segment (resp_len=6, mask=5).
    trajectories = [
        _traj([
            _seg(3, 4, 4, [-1.0, -2.0, -3.0, -4.0]),
            _seg(3, 6, 5, [-5.0, -6.0, -7.0, -8.0, -9.0, 0.0]),
        ]),
    ]
    advs = torch.tensor([1.0])
    expansion = build_expanded_dataproto(
        trajectories,
        advantages_per_trajectory=advs,
        pad_token_id=PAD_ID,
        max_prompt_length=MAX_PROMPT,
        max_response_length=MAX_RESP,
    )
    rlp = expansion.data.batch["rollout_log_probs"]
    resp_mask = expansion.data.batch["response_mask"]
    # Segment 0: rollout logp at mask=1 positions matches the input.
    valid_0 = resp_mask[0].bool()
    assert torch.allclose(rlp[0][valid_0], torch.tensor([-1.0, -2.0, -3.0, -4.0]))
    # Segment 1: same.
    valid_1 = resp_mask[1].bool()
    assert torch.allclose(
        rlp[1][valid_1], torch.tensor([-5.0, -6.0, -7.0, -8.0, -9.0])
    )


def test_rollout_log_probs_dtype_is_float32():
    """Helper-side IS computation expects fp32; confirm the column matches."""
    trajectories = [
        _traj([_seg(3, 3, 3, [-0.1, -0.2, -0.3])]),
    ]
    advs = torch.tensor([1.0])
    expansion = build_expanded_dataproto(
        trajectories,
        advantages_per_trajectory=advs,
        pad_token_id=PAD_ID,
        max_prompt_length=MAX_PROMPT,
        max_response_length=MAX_RESP,
    )
    assert expansion.data.batch["rollout_log_probs"].dtype == torch.float32
