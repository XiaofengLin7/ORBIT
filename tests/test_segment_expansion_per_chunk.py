"""Smoke tests for per_chunk_returns mode of build_expanded_dataproto."""

from __future__ import annotations

import torch

from trainers.segment_expansion import (
    _mask1_runs,
    build_expanded_dataproto,
)


def test_mask1_runs_basic():
    mask = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1, 1, 1])
    assert _mask1_runs(mask) == [(1, 3), (4, 5), (7, 10)]


def test_mask1_runs_all_zero():
    assert _mask1_runs(torch.tensor([0, 0, 0])) == []


def test_mask1_runs_all_one():
    assert _mask1_runs(torch.tensor([1, 1, 1])) == [(0, 3)]


def _make_segment(prompt_len: int, response_pattern: list[int]) -> dict:
    """Build a fake segment with response_masks following the given 0/1 pattern."""
    prompt_tokens = torch.zeros(prompt_len, dtype=torch.long)
    response_tokens = torch.zeros(len(response_pattern), dtype=torch.long)
    response_masks = torch.tensor(response_pattern, dtype=torch.long)
    return {
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "response_masks": response_masks,
    }


def test_per_chunk_fill_user_example():
    """Trajectory shape [a×5, s, a×3, s, a×2] across 3 segments.

    Segment 0: 5 actions + 1 sum → 6 chunks (mask pattern: 1*5_envgaps_1*1, but
    we simplify to back-to-back 1-token chunks separated by single 0 gaps to
    test mask1_runs detection cleanly).
    Segment 1: 3 actions + 1 sum → 4 chunks
    Segment 2: 2 actions → 2 chunks
    """
    # Mask patterns: each chunk = single 1, separated by single 0 gaps.
    # Chunk count per segment = number of "1"s.
    seg0_mask = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]   # 6 chunks
    seg1_mask = [1, 0, 1, 0, 1, 0, 1]               # 4 chunks
    seg2_mask = [1, 0, 1]                            # 2 chunks

    traj = {
        "segments": [
            _make_segment(prompt_len=4, response_pattern=seg0_mask),
            _make_segment(prompt_len=4, response_pattern=seg1_mask),
            _make_segment(prompt_len=4, response_pattern=seg2_mask),
        ],
    }

    # Per-chunk returns matching the user's example with γ=0.9, R=[1,0,1].
    g = 0.9
    per_chunk_returns = [
        [
            (g**4, True), (g**3, True), (g**2, True), (g, True), (1.0, True),  # ep 0
            (g**3 * 0, False), (g**2 * 0, False), (g * 0, False), (0.0, False),  # ep 1 (R=0)
            (g**2, True), (g, True), (1.0, True),                                 # ep 2
        ]
    ]

    expanded = build_expanded_dataproto(
        [traj],
        per_chunk_returns=per_chunk_returns,
        pad_token_id=0,
        max_prompt_length=8,
        max_response_length=20,
    )
    data = expanded.data

    # 3 segment rows.
    assert data.batch["advantages"].shape[0] == 3
    assert "is_positive" in data.batch.keys()

    # Segment 0: 6 chunks at positions 0,2,4,6,8,10 (single-token mask=1 spots).
    seg0_adv = data.batch["advantages"][0]
    seg0_pos = data.batch["is_positive"][0]
    expected_g_seg0 = [g**4, g**3, g**2, g, 1.0, 0.0]  # 5 ep0 chunks + sum chunk (heads ep1, R=0)
    for chunk_idx, t in enumerate(range(0, 11, 2)):
        # advantages are stored in float32 — use a relative tolerance.
        assert abs(seg0_adv[t].item() - expected_g_seg0[chunk_idx]) < 1e-5, (
            f"seg0 chunk {chunk_idx} at token {t}: got {seg0_adv[t].item()}, "
            f"expected {expected_g_seg0[chunk_idx]}"
        )
        expected_pos = 1.0 if expected_g_seg0[chunk_idx] > 0 else 0.0
        assert seg0_pos[t].item() == expected_pos

    # Mask=0 positions should have advantage 0 and is_positive 0.
    for t in [1, 3, 5, 7, 9]:
        assert seg0_adv[t].item() == 0.0
        assert seg0_pos[t].item() == 0.0

    # Segment 2 last chunk is the trajectory terminal — should be R=1, full weight.
    seg2_adv = data.batch["advantages"][2]
    # Last chunk in seg 2: token index 2 (mask pattern [1,0,1]).
    assert abs(seg2_adv[2].item() - 1.0) < 1e-5
    assert data.batch["is_positive"][2][2].item() == 1.0


def test_grpo_mode_unaffected_no_is_positive_key():
    """GRPO path: pass advantages_per_trajectory; is_positive must NOT be in batch."""
    traj = {"segments": [_make_segment(4, [1, 0, 1, 0, 1])]}
    advantages = torch.tensor([0.5])
    expanded = build_expanded_dataproto(
        [traj],
        advantages_per_trajectory=advantages,
        pad_token_id=0,
        max_prompt_length=8,
        max_response_length=10,
    )
    assert "is_positive" not in expanded.data.batch.keys()
    # All mask=1 positions should carry 0.5.
    adv = expanded.data.batch["advantages"][0]
    mask = expanded.data.batch["response_mask"][0]
    assert torch.allclose(
        adv[mask.bool()],
        torch.full((int(mask.sum()),), 0.5, dtype=torch.float32),
    )


def test_per_chunk_emit_is_positive_false_skips_tensor():
    """Ablation: chunk-discounted returns without TOPR. is_positive must NOT
    be in the batch, but advantages should still be filled per-chunk."""
    seg_mask = [1, 0, 1, 0, 1]
    traj = {"segments": [_make_segment(prompt_len=4, response_pattern=seg_mask)]}
    per_chunk_returns = [[(0.5, True), (0.7, True), (1.0, True)]]

    expanded = build_expanded_dataproto(
        [traj],
        per_chunk_returns=per_chunk_returns,
        emit_is_positive=False,
        pad_token_id=0,
        max_prompt_length=8,
        max_response_length=10,
    )
    assert "is_positive" not in expanded.data.batch.keys()
    adv = expanded.data.batch["advantages"][0]
    # Chunks at token 0, 2, 4 with values 0.5, 0.7, 1.0.
    assert abs(adv[0].item() - 0.5) < 1e-5
    assert abs(adv[2].item() - 0.7) < 1e-5
    assert abs(adv[4].item() - 1.0) < 1e-5


def test_grpo_and_chunk_modes_mutually_exclusive():
    """Passing both should fail; passing neither should fail."""
    import pytest
    traj = {"segments": [_make_segment(4, [1, 0, 1])]}
    with pytest.raises(AssertionError, match="exactly one"):
        build_expanded_dataproto(
            [traj],
            advantages_per_trajectory=torch.tensor([0.5]),
            per_chunk_returns=[[(1.0, True), (1.0, True)]],
            pad_token_id=0,
            max_prompt_length=8,
            max_response_length=10,
        )
    with pytest.raises(AssertionError, match="must pass either"):
        build_expanded_dataproto(
            [traj],
            pad_token_id=0,
            max_prompt_length=8,
            max_response_length=10,
        )
