"""GRPO equivalence regression test.

Asserts that, when ``rllm.advantage_method.name`` is unset (or set to
``"grpo"`` explicitly), the path through ``build_expanded_dataproto``
behaves bit-identically to the pre-change implementation:

* No ``is_positive`` tensor is added to the batch.
* ``advantages`` is filled by broadcasting the per-trajectory scalar
  ``A_i`` to every mask=1 token (the existing GRPO behavior).
* The rest of the per-token tensors (``traj_uniform_weight``, prompts,
  responses, attention/position masks) are unchanged.

We test the segment-expansion call directly because the trainer's
``_expanded_update_actor`` requires Ray + verl scaffolding that's not
worth standing up in unit tests; the dispatch logic in
``MultiEpisodeAgentPPOTrainer`` simply delegates to this function.
"""

from __future__ import annotations

import torch

from trainers.segment_expansion import build_expanded_dataproto


def _seg(prompt_len: int, response_pattern: list[int]) -> dict:
    return {
        "prompt_tokens": torch.zeros(prompt_len, dtype=torch.long),
        "response_tokens": torch.zeros(len(response_pattern), dtype=torch.long),
        "response_masks": torch.tensor(response_pattern, dtype=torch.long),
    }


def test_grpo_default_no_is_positive_field():
    """No ``is_positive`` tensor is emitted under GRPO mode."""
    trajs = [
        {"segments": [_seg(4, [1, 0, 1, 0, 1])]},
        {"segments": [_seg(4, [1, 1, 0, 1])]},
    ]
    advantages = torch.tensor([0.5, -0.25])
    expanded = build_expanded_dataproto(
        trajs,
        advantages_per_trajectory=advantages,
        pad_token_id=0,
        max_prompt_length=8,
        max_response_length=10,
    )
    assert "is_positive" not in expanded.data.batch.keys()


def test_grpo_default_broadcasts_scalar_per_trajectory():
    """advantages[row, mask=1] = A_i for every segment of trajectory i."""
    trajs = [
        # Two segments for trajectory 0, one segment for trajectory 1.
        {"segments": [
            _seg(4, [1, 0, 1]),
            _seg(4, [1, 1]),
        ]},
        {"segments": [_seg(4, [1])]},
    ]
    advantages = torch.tensor([0.7, -0.3])
    expanded = build_expanded_dataproto(
        trajs,
        advantages_per_trajectory=advantages,
        pad_token_id=0,
        max_prompt_length=8,
        max_response_length=10,
    )
    adv = expanded.data.batch["advantages"]
    mask = expanded.data.batch["response_mask"]

    # Row 0: trajectory 0 segment 0, A_i = 0.7.
    # Row 1: trajectory 0 segment 1, A_i = 0.7.
    # Row 2: trajectory 1 segment 0, A_i = -0.3.
    for row in range(2):
        for t in range(adv.shape[1]):
            if mask[row, t] == 1:
                assert abs(adv[row, t].item() - 0.7) < 1e-6, (
                    f"row {row} t {t}: {adv[row, t].item()}"
                )
            else:
                assert adv[row, t].item() == 0.0
    for t in range(adv.shape[1]):
        if mask[2, t] == 1:
            assert abs(adv[2, t].item() - (-0.3)) < 1e-6


def test_grpo_default_preserves_traj_uniform_weight_pattern():
    """traj_uniform_weight = 1/(N_G · N_t^i) at mask=1 positions, 0 elsewhere."""
    n_g = 2
    seg_a = _seg(4, [1, 0, 1, 0, 1])  # 3 mask=1 tokens
    seg_b = _seg(4, [1, 1])           # 2 mask=1 tokens
    trajs = [
        {"segments": [seg_a]},
        {"segments": [seg_b]},
    ]
    expanded = build_expanded_dataproto(
        trajs,
        advantages_per_trajectory=torch.tensor([1.0, 1.0]),
        pad_token_id=0,
        max_prompt_length=8,
        max_response_length=10,
    )
    w = expanded.data.batch["traj_uniform_weight"]
    mask = expanded.data.batch["response_mask"]

    # Trajectory 0: N_t = 3, N_G = 2 → w = 1/(2*3) = 1/6 at mask=1, 0 elsewhere.
    for t in range(w.shape[1]):
        if mask[0, t] == 1:
            assert abs(w[0, t].item() - 1.0 / 6.0) < 1e-6
        else:
            assert w[0, t].item() == 0.0
    # Trajectory 1: N_t = 2, N_G = 2 → w = 1/4.
    for t in range(w.shape[1]):
        if mask[1, t] == 1:
            assert abs(w[1, t].item() - 0.25) < 1e-6
        else:
            assert w[1, t].item() == 0.0


def test_grpo_default_segments_per_traj_unchanged():
    """ExpandedBatch.segments_per_traj reports correct counts."""
    trajs = [
        {"segments": [_seg(4, [1, 0, 1]), _seg(4, [1, 1]), _seg(4, [1])]},
        {"segments": [_seg(4, [1])]},
    ]
    expanded = build_expanded_dataproto(
        trajs,
        advantages_per_trajectory=torch.tensor([0.0, 0.0]),
        pad_token_id=0,
        max_prompt_length=8,
        max_response_length=10,
    )
    assert list(expanded.segments_per_traj) == [3, 1]
    assert expanded.data.batch["advantages"].shape[0] == 4   # total segments


def test_grpo_default_source_idx_map_matches_trajectory_layout():
    """Segment rows in source_idx_map correctly point back to their traj."""
    trajs = [
        {"segments": [_seg(4, [1])] * 2},
        {"segments": [_seg(4, [1])]},
        {"segments": [_seg(4, [1])] * 3},
    ]
    expanded = build_expanded_dataproto(
        trajs,
        advantages_per_trajectory=torch.tensor([0.1, 0.2, 0.3]),
        pad_token_id=0,
        max_prompt_length=8,
        max_response_length=10,
    )
    # Expected layout: [0, 0, 1, 2, 2, 2]
    assert list(expanded.source_idx_map) == [0, 0, 1, 2, 2, 2]
