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


def _mask1_runs(mask: torch.Tensor) -> list[tuple[int, int]]:
    """Return maximal runs of 1s in a 1-D 0/1 tensor.

    Each run corresponds to one step's response tokens (assistant
    completion). Padding zeros at the right do not introduce spurious
    runs because the rising/falling-edge diff is taken on a zero-padded
    sequence.

    Returns a list of ``(start_inclusive, end_exclusive)`` pairs.
    """
    if mask.numel() == 0:
        return []
    m = mask.to(torch.long).cpu()
    padded = torch.cat([
        torch.zeros(1, dtype=torch.long),
        m,
        torch.zeros(1, dtype=torch.long),
    ])
    diff = padded[1:] - padded[:-1]
    starts = (diff == 1).nonzero(as_tuple=True)[0].tolist()
    ends = (diff == -1).nonzero(as_tuple=True)[0].tolist()
    return list(zip(starts, ends))


def build_expanded_dataproto(
    trajectories: list[dict],
    *,
    advantages_per_trajectory: torch.Tensor | None = None,
    per_chunk_returns: list[list[tuple[float, bool]]] | None = None,
    emit_is_positive: bool = True,
    pad_token_id: int,
    max_prompt_length: int,
    max_response_length: int,
    source_non_tensor_batch: dict | None = None,
) -> ExpandedBatch:
    """Materialize the segment-expanded ``DataProto`` for the actor update.

    Two advantage-filling modes:

    * **GRPO-broadcast** (default): pass ``advantages_per_trajectory``.
      Element ``i`` is the scalar advantage ``A_i`` broadcast to every
      valid token of every segment of trajectory ``i``. ``per_chunk_returns``
      must be ``None``.

    * **Chunk-discounted** (IPA-style sibling method): pass
      ``per_chunk_returns`` — outer list indexed by trajectory, inner list
      indexed by chunk (one entry per assistant turn / summarization step
      in trajectory order). Each entry is ``(G_k, is_positive)``. The
      function fills ``advantages[row, chunk_token_range] = G_k`` and
      attaches a per-token ``is_positive`` tensor. ``advantages_per_trajectory``
      must be ``None``.

    Args:
        trajectories: list of ``N_G`` trajectory dicts (sorted by ``idx`` =
            source row index) as emitted by the engine. Each dict must carry a
            ``"segments"`` field — a list of dicts with at least
            ``"prompt_tokens"`` (1-D LongTensor), ``"response_tokens"`` (1-D
            LongTensor), and ``"response_masks"`` (1-D LongTensor in {0, 1}).
            If ``"segments"`` is missing or empty, the trajectory is treated
            as a single-segment one using its top-level ``prompt_tokens`` /
            ``response_tokens`` / ``response_masks``.
        advantages_per_trajectory: 1-D Tensor of length ``N_G``. Mutually
            exclusive with ``per_chunk_returns``.
        per_chunk_returns: per-trajectory list of (G_k, is_positive)
            tuples — one entry per chunk in trajectory-then-chunk order.
            Mutually exclusive with ``advantages_per_trajectory``.
        emit_is_positive: when ``per_chunk_returns`` is set, whether to
            emit the per-token ``is_positive`` tensor that triggers the
            actor's TOPR positive/negative split. Default True. Set to
            False to use chunk-discounted returns *without* the TOPR
            asymmetric loss (ablation: chunk credit, standard PPO surrogate).
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

    use_per_chunk = per_chunk_returns is not None
    if use_per_chunk:
        assert advantages_per_trajectory is None, (
            "build_expanded_dataproto: pass exactly one of "
            "advantages_per_trajectory or per_chunk_returns, not both."
        )
        assert len(per_chunk_returns) == n_g, (
            f"per_chunk_returns has {len(per_chunk_returns)} trajectories, "
            f"expected {n_g}"
        )
    else:
        assert advantages_per_trajectory is not None, (
            "build_expanded_dataproto: must pass either "
            "advantages_per_trajectory (GRPO mode) or per_chunk_returns "
            "(chunk-discounted mode)."
        )
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
                "rollout_log_probs": traj.get("rollout_log_probs"),
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
                "rollout_log_probs": seg.get("rollout_log_probs"),
                "mask_count": mask_count,
            })
            source_idx_per_row.append(traj_i)

    # Guard against zero — would divide by zero when computing weights.
    n_t_per_traj_clamped = np.clip(n_t_per_traj, 1, None)

    # ---- Pass 2: build padded tensor batches ----
    n_rows = len(seg_plans)

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
    # is_positive is only emitted in chunk-discounted mode and only when
    # emit_is_positive is True. The actor's TOPR branch fires iff this
    # tensor is present in the batch; absence ⇒ standard PPO surrogate.
    is_positive: torch.Tensor | None = None
    if use_per_chunk and emit_is_positive:
        is_positive = torch.zeros(
            (n_rows, max_response_length), dtype=torch.float32,
        )

    # rollout_log_probs (per-token vLLM logp) is only emitted when every
    # segment carries it — verl's TIS helper reads it to compute IS weights
    # and would produce garbage from zero-fills, so we'd rather skip the
    # column entirely than silently corrupt the loss.
    any_has_rollout_lp = all(
        plan.get("rollout_log_probs") is not None for plan in seg_plans
    )
    rollout_log_probs: torch.Tensor | None = None
    if any_has_rollout_lp:
        rollout_log_probs = torch.zeros(
            (n_rows, max_response_length), dtype=torch.float32,
        )

    a_cpu = (
        None if use_per_chunk
        else advantages_per_trajectory.detach().cpu()
    )

    # Per-trajectory cursor into per_chunk_returns: each segment of a
    # trajectory consumes `n_chunks_in_segment` entries from the head of
    # the trajectory's return list, in order.
    chunk_cursor: list[int] = [0] * n_g

    for row_i, plan in enumerate(seg_plans):
        traj_i = plan["traj_i"]
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

        if rollout_log_probs is not None:
            seg_rl = plan["rollout_log_probs"].to(torch.float32)
            rollout_log_probs[row_i] = _pad_1d(
                seg_rl, max_response_length, pad_value=0, left_pad=False
            )

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

        # Per-token traj_uniform_weight is identical in both modes.
        valid = seg_mask.bool()
        traj_uniform_weight[row_i].masked_fill_(valid, weight_const)

        if use_per_chunk:
            # Walk the segment's mask to identify per-chunk token ranges
            # (one contiguous mask=1 run per chunk), then write each chunk's
            # G_k and TOPR sign into the per-token tensors.
            chunk_runs = _mask1_runs(seg_mask)
            traj_returns = per_chunk_returns[traj_i]
            cursor = chunk_cursor[traj_i]
            for run_idx, (start, end) in enumerate(chunk_runs):
                chunk_idx = cursor + run_idx
                if chunk_idx >= len(traj_returns):
                    # Engine emitted more mask=1 runs than recorded chunks;
                    # leave the tail at zero (logged once below).
                    continue
                G_k, is_pos = traj_returns[chunk_idx]
                advantages[row_i, start:end] = float(G_k)
                if is_positive is not None:
                    is_positive[row_i, start:end] = 1.0 if is_pos else 0.0
            chunk_cursor[traj_i] = cursor + len(chunk_runs)
        else:
            advantages[row_i].masked_fill_(valid, float(a_cpu[traj_i].item()))

    if use_per_chunk:
        # Sanity: every trajectory should have consumed exactly its return
        # list. Mismatches indicate an alignment bug between step_metadata
        # and the assembled segments — log loudly without crashing.
        for traj_i, consumed in enumerate(chunk_cursor):
            expected = len(per_chunk_returns[traj_i])
            if consumed != expected:
                import logging
                logging.getLogger(__name__).warning(
                    "build_expanded_dataproto: trajectory %d consumed %d "
                    "chunks via response_masks but per_chunk_returns has %d. "
                    "Per-chunk advantages may be misaligned.",
                    traj_i, consumed, expected,
                )

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
    if is_positive is not None:
        tensors["is_positive"] = is_positive
    if rollout_log_probs is not None:
        tensors["rollout_log_probs"] = rollout_log_probs

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
