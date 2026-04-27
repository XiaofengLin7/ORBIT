"""Helpers for expanding multi-segment trajectories into per-segment PPO rows.

A trajectory with ``M`` summarizations produces ``M+1`` independent PPO
samples (segments). This module builds the per-segment ``DataProto`` from a
list of trajectory dicts (each carrying its ``"segments"`` list as emitted
by :class:`SummarizingAgentExecutionEngine.assemble_segments`), broadcasts
the trajectory-level advantage to every segment row, and computes the
per-token ``traj_uniform_weight`` tensor consumed by
:class:`TrajectoryUniformPPOActor`.

Public surface:
    * :func:`build_expanded_dataproto` — main entry point.
    * :func:`extract_advantage_per_trajectory` — reads ``A_i`` from a
      per-trajectory DataProto produced by ``compute_advantage``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from verl import DataProto


@dataclass
class ExpandedBatch:
    """Result of :func:`build_expanded_dataproto`.

    ``data`` is the segment-row DataProto (length ``Σ_i K_i``).
    ``source_idx_map`` is a length-``Σ_i K_i`` numpy array mapping each segment
    row back to its source trajectory's row in the original (un-expanded)
    batch — pass it to ``source_batch.select_idxs(source_idx_map)`` to grow
    the source batch to match the expanded batch's row count for ``union``.
    ``segments_per_traj`` is a length-``N_G`` numpy array of segment counts per
    trajectory (in source-row order).
    """

    data: DataProto
    source_idx_map: np.ndarray
    segments_per_traj: np.ndarray


def extract_advantage_per_trajectory(advantages_tensor: torch.Tensor,
                                     response_mask: torch.Tensor) -> torch.Tensor:
    """Pull a single scalar `A_i` from each row of a per-trajectory advantages tensor.

    GRPO sets ``advantages[i, t] = A_i`` at every valid token position and 0
    elsewhere, so taking the value at the first valid token position
    (or any valid position) of each row recovers the scalar.
    """
    n_rows = advantages_tensor.shape[0]
    out = torch.zeros(n_rows, dtype=advantages_tensor.dtype,
                      device=advantages_tensor.device)
    for i in range(n_rows):
        valid_positions = response_mask[i].nonzero(as_tuple=True)[0]
        if valid_positions.numel() > 0:
            out[i] = advantages_tensor[i, valid_positions[0]]
    return out


def _pad_1d(t: torch.Tensor, length: int, pad_value: int = 0,
            left_pad: bool = False) -> torch.Tensor:
    """Pad a 1-D tensor to ``length`` (right-pad by default)."""
    cur = t.shape[0]
    if cur == length:
        return t
    if cur > length:
        return t[-length:] if left_pad else t[:length]
    pad = torch.full((length - cur,), pad_value, dtype=t.dtype, device=t.device)
    return torch.cat([pad, t], dim=0) if left_pad else torch.cat([t, pad], dim=0)


def build_expanded_dataproto(
    trajectories: list[dict],
    *,
    advantages_per_trajectory: torch.Tensor,
    pad_token_id: int,
    max_prompt_length: int,
    max_response_length: int,
    source_non_tensor_batch: dict | None = None,
) -> ExpandedBatch:
    """Materialize the segment-expanded ``DataProto`` for the actor update.

    Args:
        trajectories: list of ``N_G`` trajectory dicts (sorted by ``idx`` =
            source row index) as emitted by the engine. Each dict must carry a
            ``"segments"`` field — a list of dicts with at least
            ``"prompt_tokens"`` (1-D LongTensor), ``"response_tokens"`` (1-D
            LongTensor), and ``"response_masks"`` (1-D LongTensor in {0, 1}).
            If ``"segments"`` is missing or empty, the trajectory is treated
            as a single-segment one using its top-level ``prompt_tokens`` /
            ``response_tokens`` / ``response_masks``.
        advantages_per_trajectory: 1-D Tensor of length ``N_G``. Element ``i``
            is the scalar advantage ``A_i`` to broadcast to every valid token
            of every segment of trajectory ``i``.
        pad_token_id: tokenizer pad id.
        max_prompt_length: target padded length for prompts.
        max_response_length: target padded length for responses.
        source_non_tensor_batch: optional dict of length-``N_G`` numpy arrays
            (``uid``, ``data_source``, ``extra_info``, …). Each row is
            replicated according to ``segments_per_traj`` so the returned
            non_tensor_batch has length ``Σ K_i``. Pass ``None`` to skip.

    Returns:
        :class:`ExpandedBatch` — see field docs.
    """
    n_g = len(trajectories)
    assert n_g > 0, "build_expanded_dataproto: empty trajectories list"
    assert advantages_per_trajectory.shape == (n_g,), (
        f"advantages_per_trajectory shape {tuple(advantages_per_trajectory.shape)} "
        f"must match number of trajectories {n_g}"
    )

    # ---- Pass 1: gather per-segment row plans + N_t^i per trajectory ----
    seg_plans: list[dict] = []         # one entry per output row
    source_idx_per_row: list[int] = []  # source trajectory row per output row
    segments_per_traj = np.zeros(n_g, dtype=np.int64)
    n_t_per_traj = np.zeros(n_g, dtype=np.int64)

    for traj_i, traj in enumerate(trajectories):
        segs = traj.get("segments")
        if not segs:
            # Treat as single-segment from the top-level fields.
            segs = [{
                "prompt_tokens": traj["prompt_tokens"],
                "response_tokens": traj["response_tokens"],
                "response_masks": traj["response_masks"],
            }]

        segments_per_traj[traj_i] = len(segs)
        for seg in segs:
            mask_count = int(seg["response_masks"].sum().item())
            n_t_per_traj[traj_i] += mask_count
            seg_plans.append({
                "traj_i": traj_i,
                "prompt_tokens": seg["prompt_tokens"],
                "response_tokens": seg["response_tokens"],
                "response_masks": seg["response_masks"].to(torch.long),
                "mask_count": mask_count,
            })
            source_idx_per_row.append(traj_i)

    # Guard against zero — would divide by zero when computing weights.
    n_t_per_traj_clamped = np.clip(n_t_per_traj, 1, None)

    # ---- Pass 2: build padded tensor batches ----
    n_rows = len(seg_plans)
    device = advantages_per_trajectory.device

    input_ids = torch.full(
        (n_rows, max_prompt_length + max_response_length),
        pad_token_id, dtype=torch.long,
    )
    attention_mask = torch.zeros(
        (n_rows, max_prompt_length + max_response_length),
        dtype=torch.long,
    )
    prompts = torch.full(
        (n_rows, max_prompt_length), pad_token_id, dtype=torch.long,
    )
    responses = torch.full(
        (n_rows, max_response_length), pad_token_id, dtype=torch.long,
    )
    response_mask = torch.zeros(
        (n_rows, max_response_length), dtype=torch.long,
    )
    advantages = torch.zeros(
        (n_rows, max_response_length), dtype=torch.float32,
    )
    token_level_scores = torch.zeros(
        (n_rows, max_response_length), dtype=torch.float32,
    )
    traj_uniform_weight = torch.zeros(
        (n_rows, max_response_length), dtype=torch.float32,
    )

    a_cpu = advantages_per_trajectory.detach().cpu()

    for row_i, plan in enumerate(seg_plans):
        traj_i = plan["traj_i"]
        a_i = float(a_cpu[traj_i].item())
        weight_const = 1.0 / (n_g * float(n_t_per_traj_clamped[traj_i]))

        # Truncate / pad the per-segment prompt and response.
        seg_prompt = _pad_1d(plan["prompt_tokens"].to(torch.long),
                             max_prompt_length, pad_token_id, left_pad=True)
        seg_response = _pad_1d(plan["response_tokens"].to(torch.long),
                               max_response_length, pad_token_id, left_pad=False)
        seg_mask = _pad_1d(plan["response_masks"], max_response_length,
                           pad_value=0, left_pad=False)

        prompts[row_i] = seg_prompt
        responses[row_i] = seg_response
        response_mask[row_i] = seg_mask

        # input_ids = prompt | response (prompt is left-padded, response is right-padded).
        input_ids[row_i, :max_prompt_length] = seg_prompt
        input_ids[row_i, max_prompt_length:] = seg_response

        # attention_mask: 1 wherever the underlying value is real (i.e. the
        # left-padded prompt's right part and the right-padded response's
        # left part). Equivalent to "value != pad_token_id" but avoids
        # treating in-vocab pad tokens as masks.
        prompt_real_len = int(plan["prompt_tokens"].shape[0])
        prompt_real_len = min(prompt_real_len, max_prompt_length)
        attention_mask[row_i, max_prompt_length - prompt_real_len:max_prompt_length] = 1
        response_real_len = int(plan["response_tokens"].shape[0])
        response_real_len = min(response_real_len, max_response_length)
        attention_mask[row_i, max_prompt_length:max_prompt_length + response_real_len] = 1

        # Broadcast trajectory-level advantage and per-token weight to valid tokens.
        valid = seg_mask.bool()
        advantages[row_i].masked_fill_(valid, a_i)
        traj_uniform_weight[row_i].masked_fill_(valid, weight_const)

    # position_ids = cumsum(attention_mask) - 1, masked to 0 at pad positions.
    position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

    tensors: dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "prompts": prompts,
        "responses": responses,
        "response_mask": response_mask,
        "advantages": advantages,
        "token_level_scores": token_level_scores,
        "traj_uniform_weight": traj_uniform_weight,
    }

    # Replicate non-tensor source fields per segment row.
    non_tensors: dict[str, np.ndarray] = {}
    if source_non_tensor_batch:
        idx_map = np.asarray(source_idx_per_row, dtype=np.int64)
        for key, arr in source_non_tensor_batch.items():
            arr = np.asarray(arr)
            if arr.shape[0] != n_g:
                # Field doesn't share leading dim with trajectories — skip.
                continue
            non_tensors[key] = arr[idx_map]

    # Always attach traj_idx so downstream code can recover trajectory groupings.
    non_tensors["traj_idx"] = np.asarray(source_idx_per_row, dtype=np.int64)

    expanded = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)
    return ExpandedBatch(
        data=expanded,
        source_idx_map=np.asarray(source_idx_per_row, dtype=np.int64),
        segments_per_traj=segments_per_traj,
    )
