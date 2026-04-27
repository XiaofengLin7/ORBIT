"""Tests for trainers/segment_expansion.py.

Verifies:
* Expanded batch shape matches sum of per-trajectory segment counts.
* Source non-tensor fields are replicated per segment row.
* `traj_idx` correctly groups segments back to trajectories.
* Per-token `traj_uniform_weight` evaluates to ``1/(N_G · N_t^i)`` at valid
  positions; invariant: sum over the whole batch of ``weight * mask`` is
  exactly ``1.0`` (= ``N_G · (1/N_G)``), regardless of segment counts or
  per-trajectory token counts. This invariant is what makes the
  trajectory-uniform PPO loss aggregation correct.
* Advantage broadcast: every valid token of every segment of trajectory ``i``
  carries the broadcast scalar ``A_i``.
* Edge cases: trajectories without a `segments` field, single-segment cases,
  trajectories with zero valid tokens (clamped to avoid division by zero).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from trainers.segment_expansion import (
    ExpandedBatch,
    build_expanded_dataproto,
    extract_advantage_per_trajectory,
)


def _seg(prompt_len: int, resp_len: int, mask_count: int) -> dict:
    """Build a synthetic segment dict with the requested shape."""
    return {
        "prompt_tokens": torch.arange(1, prompt_len + 1, dtype=torch.long),
        "response_tokens": torch.arange(100, 100 + resp_len, dtype=torch.long),
        "response_masks": torch.cat([
            torch.ones(mask_count, dtype=torch.long),
            torch.zeros(max(0, resp_len - mask_count), dtype=torch.long),
        ]),
    }


def _make_traj(segments: list[dict]) -> dict:
    """Wrap segments in a trajectory dict (top-level fields = segment[0])."""
    return {
        "prompt_tokens": segments[0]["prompt_tokens"],
        "response_tokens": segments[0]["response_tokens"],
        "response_masks": segments[0]["response_masks"],
        "segments": segments,
    }


# ---------------------------------------------------------------------------
# Shape & layout tests
# ---------------------------------------------------------------------------

class TestShape:
    def test_expanded_length_matches_total_segments(self):
        trajectories = [
            _make_traj([_seg(3, 5, 4), _seg(3, 4, 3), _seg(3, 6, 5)]),  # 3 segs
            _make_traj([_seg(4, 4, 4)]),                                 # 1 seg
            _make_traj([_seg(5, 3, 2), _seg(5, 2, 2)]),                  # 2 segs
        ]
        result = build_expanded_dataproto(
            trajectories,
            advantages_per_trajectory=torch.tensor([1.0, 0.0, -1.0]),
            pad_token_id=0,
            max_prompt_length=10,
            max_response_length=10,
        )
        assert len(result.data) == 3 + 1 + 2 == 6
        assert result.segments_per_traj.tolist() == [3, 1, 2]
        assert result.source_idx_map.tolist() == [0, 0, 0, 1, 2, 2]

    def test_traj_idx_field(self):
        trajectories = [
            _make_traj([_seg(2, 3, 2), _seg(2, 3, 2)]),
            _make_traj([_seg(2, 3, 2)]),
        ]
        result = build_expanded_dataproto(
            trajectories,
            advantages_per_trajectory=torch.tensor([0.5, -0.5]),
            pad_token_id=0,
            max_prompt_length=5,
            max_response_length=5,
        )
        assert list(result.data.non_tensor_batch["traj_idx"]) == [0, 0, 1]

    def test_response_mask_stays_binary(self):
        trajectories = [
            _make_traj([_seg(2, 4, 3), _seg(2, 4, 2)]),
            _make_traj([_seg(2, 4, 4)]),
        ]
        result = build_expanded_dataproto(
            trajectories,
            advantages_per_trajectory=torch.tensor([1.0, -1.0]),
            pad_token_id=0,
            max_prompt_length=5,
            max_response_length=5,
        )
        rm = result.data.batch["response_mask"]
        assert torch.equal(rm, rm.to(torch.bool).to(torch.long)), \
            "response_mask must stay in {0, 1} after expansion"


# ---------------------------------------------------------------------------
# Trajectory-uniform weight invariant
# ---------------------------------------------------------------------------

class TestTrajUniformWeight:
    def test_per_traj_constant(self):
        """Every valid token of trajectory i carries the same constant weight."""
        trajectories = [
            _make_traj([_seg(2, 5, 5), _seg(2, 5, 3)]),  # N_t^0 = 8
            _make_traj([_seg(2, 4, 4)]),                  # N_t^1 = 4
        ]
        result = build_expanded_dataproto(
            trajectories,
            advantages_per_trajectory=torch.tensor([1.0, 1.0]),
            pad_token_id=0,
            max_prompt_length=5,
            max_response_length=10,
        )
        n_g = 2
        w0_expected = 1.0 / (n_g * 8)
        w1_expected = 1.0 / (n_g * 4)

        for row, expected_w in enumerate([w0_expected, w0_expected, w1_expected]):
            mask = result.data.batch["response_mask"][row].bool()
            w = result.data.batch["traj_uniform_weight"][row]
            uniq = w[mask].unique()
            assert uniq.numel() == 1, f"row {row}: weight has multiple values on mask"
            assert abs(float(uniq.item()) - expected_w) < 1e-9, (
                f"row {row}: weight {uniq.item()} != expected {expected_w}"
            )

    def test_global_weight_sum_equals_one(self):
        """Σ (weight * mask) over the WHOLE batch equals exactly 1.0
        (= N_G · (1/N_G)). This is the invariant that makes the trajectory-
        uniform PPO loss aggregation come out as L = (1/N_G)·Σ_i L_i."""
        rng = np.random.default_rng(0)
        # Random fixture: 5 trajectories, each with 1–4 segments of varied length.
        trajectories = []
        for _ in range(5):
            n_segs = int(rng.integers(1, 5))
            segs = []
            for _ in range(n_segs):
                resp_len = int(rng.integers(2, 8))
                mask_count = int(rng.integers(1, resp_len + 1))
                segs.append(_seg(prompt_len=int(rng.integers(2, 6)),
                                 resp_len=resp_len, mask_count=mask_count))
            trajectories.append(_make_traj(segs))

        result = build_expanded_dataproto(
            trajectories,
            advantages_per_trajectory=torch.zeros(len(trajectories)),
            pad_token_id=0,
            max_prompt_length=10,
            max_response_length=10,
        )
        total = (
            result.data.batch["traj_uniform_weight"]
            * result.data.batch["response_mask"]
        ).sum().item()
        assert abs(total - 1.0) < 1e-6, f"sum(w*mask) = {total}, expected 1.0"

    def test_zero_token_trajectory_doesnt_explode(self):
        """A trajectory with no mask=1 tokens (e.g., truncated) must not
        cause a division by zero in the weight construction."""
        trajectories = [
            _make_traj([_seg(2, 3, 0)]),       # zero valid tokens
            _make_traj([_seg(2, 3, 2)]),
        ]
        # Should not raise.
        result = build_expanded_dataproto(
            trajectories,
            advantages_per_trajectory=torch.tensor([0.0, 1.0]),
            pad_token_id=0,
            max_prompt_length=5,
            max_response_length=5,
        )
        # The zero-token trajectory contributes mask=0 everywhere → its weight
        # rows are 0 (clamping prevents NaN/inf, mask filters to 0 anyway).
        w_row0 = result.data.batch["traj_uniform_weight"][0]
        m_row0 = result.data.batch["response_mask"][0]
        assert (w_row0 * m_row0).sum().item() == 0.0


# ---------------------------------------------------------------------------
# Advantage broadcast
# ---------------------------------------------------------------------------

class TestAdvantageBroadcast:
    def test_each_segment_carries_traj_advantage(self):
        trajectories = [
            _make_traj([_seg(2, 3, 3), _seg(2, 3, 3), _seg(2, 3, 3)]),
            _make_traj([_seg(2, 4, 4)]),
        ]
        a = torch.tensor([2.5, -1.5])
        result = build_expanded_dataproto(
            trajectories,
            advantages_per_trajectory=a,
            pad_token_id=0,
            max_prompt_length=5,
            max_response_length=5,
        )
        for row, expected_a in enumerate([2.5, 2.5, 2.5, -1.5]):
            mask = result.data.batch["response_mask"][row].bool()
            adv = result.data.batch["advantages"][row]
            uniq = adv[mask].unique()
            assert uniq.numel() == 1
            assert abs(float(uniq.item()) - expected_a) < 1e-6


# ---------------------------------------------------------------------------
# Source-side replication
# ---------------------------------------------------------------------------

class TestSourceReplication:
    def test_uid_replication(self):
        trajectories = [
            _make_traj([_seg(2, 3, 2), _seg(2, 3, 2)]),
            _make_traj([_seg(2, 3, 2)]),
            _make_traj([_seg(2, 3, 2), _seg(2, 3, 2), _seg(2, 3, 2)]),
        ]
        result = build_expanded_dataproto(
            trajectories,
            advantages_per_trajectory=torch.zeros(3),
            pad_token_id=0,
            max_prompt_length=5,
            max_response_length=5,
            source_non_tensor_batch={
                "uid": np.array(["A", "B", "C"]),
                "data_source": np.array(["maze-ep3", "maze-ep5", "wordle-ep10"]),
            },
        )
        assert list(result.data.non_tensor_batch["uid"]) == ["A", "A", "B", "C", "C", "C"]
        assert list(result.data.non_tensor_batch["data_source"]) == [
            "maze-ep3", "maze-ep3", "maze-ep5",
            "wordle-ep10", "wordle-ep10", "wordle-ep10",
        ]

    def test_no_source_batch_still_emits_traj_idx(self):
        trajectories = [
            _make_traj([_seg(2, 3, 2), _seg(2, 3, 2)]),
        ]
        result = build_expanded_dataproto(
            trajectories,
            advantages_per_trajectory=torch.tensor([0.0]),
            pad_token_id=0,
            max_prompt_length=5,
            max_response_length=5,
            source_non_tensor_batch=None,
        )
        assert "traj_idx" in result.data.non_tensor_batch
        assert list(result.data.non_tensor_batch["traj_idx"]) == [0, 0]


# ---------------------------------------------------------------------------
# extract_advantage_per_trajectory
# ---------------------------------------------------------------------------

class TestExtractAdvantage:
    def test_recovers_scalar_at_first_valid_position(self):
        # advantages tensor: A_i broadcast at every valid token, 0 elsewhere
        n_traj, T = 3, 8
        masks = torch.zeros(n_traj, T, dtype=torch.long)
        masks[0, 2:5] = 1
        masks[1, 0:6] = 1
        masks[2, 4:8] = 1
        advantages = torch.zeros(n_traj, T)
        a_values = [1.5, -0.7, 3.0]
        for i, a in enumerate(a_values):
            advantages[i] = masks[i].float() * a
        out = extract_advantage_per_trajectory(advantages, masks)
        for i, a in enumerate(a_values):
            assert abs(float(out[i].item()) - a) < 1e-6

    def test_zero_when_row_is_all_masked_out(self):
        masks = torch.zeros(2, 5, dtype=torch.long)
        masks[0, 1:3] = 1  # row 0 has valid tokens
        # row 1 has no valid tokens
        advantages = torch.zeros(2, 5)
        advantages[0, 1:3] = 0.42
        out = extract_advantage_per_trajectory(advantages, masks)
        assert abs(float(out[0].item()) - 0.42) < 1e-6
        assert float(out[1].item()) == 0.0


# ---------------------------------------------------------------------------
# Single-segment fallback (no `segments` field)
# ---------------------------------------------------------------------------

class TestSingleSegmentFallback:
    def test_uses_top_level_fields_when_segments_missing(self):
        traj = {
            "prompt_tokens": torch.arange(1, 4, dtype=torch.long),
            "response_tokens": torch.arange(100, 105, dtype=torch.long),
            "response_masks": torch.cat([
                torch.ones(3, dtype=torch.long),
                torch.zeros(2, dtype=torch.long),
            ]),
        }
        # No `segments` field.
        result = build_expanded_dataproto(
            [traj],
            advantages_per_trajectory=torch.tensor([1.0]),
            pad_token_id=0,
            max_prompt_length=5,
            max_response_length=5,
        )
        assert len(result.data) == 1
        # weight = 1 / (1 * 3) on valid tokens
        rm = result.data.batch["response_mask"][0].bool()
        w = result.data.batch["traj_uniform_weight"][0]
        assert abs(float(w[rm].unique().item()) - (1.0 / 3.0)) < 1e-6
