"""Local SDPO-capable actor wrapper.

This module provides a local actor implementation that supports both mixed and
pure distillation optimization modes in the same optimizer step:

``policy_loss = (use_grpo_loss ? ppo_pg_loss : 0) + lambda * sdpo_loss``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.workers.actor.dp_actor import DataParallelPPOActor


@dataclass(frozen=True)
class SDPODistillParams:
    """Distillation controls parsed from batch meta info."""

    enabled: bool
    lambda_coef: float
    use_grpo_loss: bool
    loss_variant: str
    alpha: float
    is_clip: bool
    full_logit_topk: int
    full_logit_add_tail: bool
    negate_sdpo_loss: bool
    use_stale_coefficient: bool
    teacher_regularization: str


def _add_tail(log_probs: torch.Tensor) -> torch.Tensor:
    """Append tail log-probability bucket for top-k distillation."""
    log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_s = torch.clamp(log_s, max=-1e-7)
    tail_log = torch.log(-torch.expm1(log_s))
    return torch.cat([log_probs, tail_log], dim=-1)


def _renorm_topk_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    """Re-normalize top-k log-probs so they form a valid distribution."""
    return log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)


def _topk_entropy(topk_log_probs: torch.Tensor, add_tail: bool) -> torch.Tensor:
    """Approximate entropy from top-k log-probs. Returns [B, T]."""
    lp = _add_tail(topk_log_probs) if add_tail else _renorm_topk_log_probs(topk_log_probs)
    return -(torch.exp(lp) * lp).sum(dim=-1)


def _compute_sdpo_per_token_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor | None,
    *,
    loss_variant: str,
    alpha: float,
    student_all_log_probs: torch.Tensor | None = None,
    teacher_all_log_probs: torch.Tensor | None = None,
    student_topk_log_probs: torch.Tensor | None = None,
    teacher_topk_log_probs: torch.Tensor | None = None,
    full_logit_topk: int = 64,
    full_logit_add_tail: bool = True,
) -> torch.Tensor:
    """Compute SDPO per-token loss matching ../SDPO branches."""
    if loss_variant == "non_full":
        if teacher_log_probs is None:
            raise ValueError("teacher_log_probs is required for non_full SDPO loss variant.")
        return (student_log_probs - teacher_log_probs).detach() * student_log_probs

    use_topk = student_topk_log_probs is not None and teacher_topk_log_probs is not None
    if use_topk:
        student_topk = student_topk_log_probs
        teacher_topk = teacher_topk_log_probs
        if student_topk.shape != teacher_topk.shape:
            raise ValueError(
                "student_topk_log_probs and teacher_topk_log_probs must have identical shapes, "
                f"got {tuple(student_topk.shape)} and {tuple(teacher_topk.shape)}."
            )
        if full_logit_add_tail:
            student_distill_log_probs = _add_tail(student_topk)
            teacher_distill_log_probs = _add_tail(teacher_topk)
        else:
            student_distill_log_probs = _renorm_topk_log_probs(student_topk)
            teacher_distill_log_probs = _renorm_topk_log_probs(teacher_topk)
    else:
        if student_all_log_probs is None or teacher_all_log_probs is None:
            raise ValueError(
                "full_logit variant requires either (student_topk_log_probs, teacher_topk_log_probs) "
                "or (student_all_log_probs, teacher_all_log_probs)."
            )
        if full_logit_topk > 0:
            topk = min(int(full_logit_topk), int(student_all_log_probs.shape[-1]))
            student_topk, topk_indices = torch.topk(student_all_log_probs, k=topk, dim=-1)
            teacher_topk = torch.gather(teacher_all_log_probs, dim=-1, index=topk_indices)
            if full_logit_add_tail:
                student_distill_log_probs = _add_tail(student_topk)
                teacher_distill_log_probs = _add_tail(teacher_topk)
            else:
                student_distill_log_probs = _renorm_topk_log_probs(student_topk)
                teacher_distill_log_probs = _renorm_topk_log_probs(teacher_topk)
        else:
            student_distill_log_probs = student_all_log_probs
            teacher_distill_log_probs = teacher_all_log_probs

    if alpha == 0.0:
        kl_loss = F.kl_div(student_distill_log_probs, teacher_distill_log_probs, reduction="none", log_target=True)
    elif alpha == 1.0:
        kl_loss = F.kl_div(teacher_distill_log_probs, student_distill_log_probs, reduction="none", log_target=True)
    else:
        alpha_tensor = torch.tensor(
            alpha,
            dtype=student_distill_log_probs.dtype,
            device=student_distill_log_probs.device,
        )
        mixture_log_probs = torch.logsumexp(
            torch.stack(
                [
                    student_distill_log_probs + torch.log(1 - alpha_tensor),
                    teacher_distill_log_probs + torch.log(alpha_tensor),
                ]
            ),
            dim=0,
        )
        kl_teacher = F.kl_div(mixture_log_probs, teacher_distill_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_distill_log_probs, reduction="none", log_target=True)
        kl_loss = torch.lerp(kl_student, kl_teacher, alpha_tensor)

    return kl_loss.sum(-1)


def _segmented_reverse_cumsum(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Reverse cumulative sum within contiguous segments defined by *mask*.

    For each segment (contiguous run of 1s in *mask*), position ℓ gets
    ``sum_{ℓ'=ℓ}^{end_of_segment} values[ℓ']``.  Positions where
    ``mask == 0`` are always zero.  Operates in pure tensor ops (no Python
    loops) and is differentiable w.r.t. *values*.
    """
    masked = values * mask
    flipped = masked.flip(1)
    flipped_mask = mask.flip(1)
    raw_cumsum = flipped.cumsum(dim=1)

    # Detect segment starts in flipped space (mask transitions 0→1).
    starts = torch.zeros_like(flipped_mask)
    starts[:, 0] = flipped_mask[:, 0]
    starts[:, 1:] = flipped_mask[:, 1:] * (1 - flipped_mask[:, :-1])

    # At each segment start, the correction is raw_cumsum at the previous
    # position (the total leak from prior segments).  Forward-fill each
    # correction across its segment via scatter_add + gather on segment ids.
    correction_at_start = torch.zeros_like(raw_cumsum)
    correction_at_start[:, 1:] = starts[:, 1:] * raw_cumsum[:, :-1]

    segment_ids = starts.cumsum(dim=1).long()
    B, T = raw_cumsum.shape
    max_seg = int(segment_ids.max().item()) + 1
    corrections_per_seg = torch.zeros(B, max_seg, device=raw_cumsum.device, dtype=raw_cumsum.dtype)
    corrections_per_seg.scatter_add_(1, segment_ids, correction_at_start)
    running_correction = corrections_per_seg.gather(1, segment_ids)

    segmented = (raw_cumsum - running_correction) * flipped_mask
    return segmented.flip(1)


def _count_active_distill_sequences(distill_mask: torch.Tensor) -> int:
    """Count sequences with at least one distill-valid token."""
    if distill_mask.ndim != 2:
        raise ValueError(f"distill_mask must have shape [B, T], got shape {tuple(distill_mask.shape)}")
    return int((distill_mask.sum(dim=-1) > 0).sum().item())


def _zero_sdpo_metrics() -> dict[str, float]:
    """Return zero-valued SDPO metrics for empty micro-batches."""
    return {
        "distill/sdpo_loss": 0.0,
        "distill/token_count": 0.0,
        "distill/sdpo_clipfrac": 0.0,
        "distill/sdpo_ratio_gt_clip_frac": 0.0,
        "distill/sdpo_adv_abs_mean": 0.0,
        "distill/sdpo_adv_abs_max": 0.0,
        "distill/active_seq_count": 0.0,
        "distill/diag_mean_student_logp": 0.0,
        "distill/diag_mean_teacher_logp": 0.0,
        "distill/diag_student_entropy": 0.0,
        "distill/diag_teacher_entropy": 0.0,
        "distill/diag_kl_per_token_mean": 0.0,
    }


def _resolve_clip_ratio_bound(config_value: float | None, default: float) -> float:
    """Resolve a clip bound while preserving explicit 0.0 overrides."""
    return default if config_value is None else float(config_value)


class SDPODataParallelPPOActor(DataParallelPPOActor):
    """PPO actor wrapper with additional SDPO loss term support."""

    def __init__(
        self,
        config: Any,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        """Initialize actor and optional teacher module handle."""
        super().__init__(config=config, actor_module=actor_module, actor_optimizer=actor_optimizer)
        self.teacher_module: nn.Module | None = None

    def set_teacher_module(self, teacher_module: nn.Module | None) -> None:
        """Attach an external teacher module (for ema/every_n_steps modes)."""
        self.teacher_module = teacher_module

    def _parse_distill_params(self, meta_info: dict[str, Any]) -> SDPODistillParams:
        """Parse distillation parameters from batch meta info with defaults."""
        enabled = bool(meta_info.get("distill_enabled", False))
        loss_variant = str(meta_info.get("distill_loss_variant", "non_full")).lower()
        if loss_variant not in {"non_full", "full_logit"}:
            raise ValueError(f"Unsupported distill loss_variant: {loss_variant}")

        alpha = float(meta_info.get("distill_alpha", 1.0))
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"distill_alpha must be in [0, 1], got {alpha}")

        is_clip = bool(meta_info.get("distill_is_clip", False))

        full_logit_topk = int(meta_info.get("distill_full_logit_topk", 64))
        if full_logit_topk < 0:
            raise ValueError(f"distill_full_logit_topk must be non-negative, got {full_logit_topk}")
        use_stale_coefficient = bool(meta_info.get("distill_use_stale_coefficient", False))
        if loss_variant == "full_logit" and is_clip:
            raise ValueError(
                "distill_is_clip is not supported when distill_loss_variant='full_logit'."
            )
        if loss_variant == "full_logit" and use_stale_coefficient:
            raise ValueError(
                "distill_use_stale_coefficient is not supported when "
                "distill_loss_variant='full_logit'."
            )

        teacher_regularization = str(meta_info.get("distill_teacher_regularization", "none")).lower()
        if teacher_regularization not in {"none", "ema", "every_n_steps"}:
            raise ValueError(
                "distill_teacher_regularization must be one of ['none', 'ema', 'every_n_steps'], "
                f"got {teacher_regularization}"
            )

        return SDPODistillParams(
            enabled=enabled,
            lambda_coef=float(meta_info.get("distill_lambda", 0.0)),
            use_grpo_loss=bool(meta_info.get("distill_use_grpo_loss", True)),
            loss_variant=loss_variant,
            alpha=alpha,
            is_clip=is_clip,
            full_logit_topk=full_logit_topk,
            full_logit_add_tail=bool(meta_info.get("distill_full_logit_add_tail", True)),
            negate_sdpo_loss=bool(meta_info.get("distill_negate_sdpo_loss", False)),
            use_stale_coefficient=use_stale_coefficient,
            teacher_regularization=teacher_regularization,
        )

    def _compute_pg_loss(
        self,
        *,
        use_grpo_loss: bool,
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        loss_agg_mode: str,
        rollout_is_weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute policy-gradient loss or return autograd-safe zero in pure SDPO mode."""
        if use_grpo_loss:
            loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
            policy_loss_fn = get_policy_loss_fn(loss_mode)
            return policy_loss_fn(
                old_log_prob=old_log_prob,
                log_prob=log_prob,
                advantages=advantages,
                response_mask=response_mask,
                loss_agg_mode=loss_agg_mode,
                config=self.config,
                rollout_is_weights=rollout_is_weights,
            )
        # Keep a valid graph so backward() remains well-defined when no GRPO term is used.
        return log_prob.sum() * 0.0, {}

    def _forward_with_all_log_probs(
        self,
        model_inputs: dict[str, torch.Tensor],
        *,
        temperature: float,
        module: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass that returns token log-probs and full log-prob vectors."""
        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs["attention_mask"]
            position_ids = model_inputs["position_ids"]
            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)
            output = module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            logits = output.logits
            response_length = model_inputs["responses"].shape[-1]
            logits = logits[:, -response_length - 1 : -1, :]
            logits = self._scale_logits_by_temperature(logits, temperature=temperature)
            all_log_probs = torch.log_softmax(logits.float(), dim=-1)
            labels = model_inputs["responses"]
            token_log_probs = torch.gather(all_log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        return token_log_probs, all_log_probs

    def _forward_with_module_log_probs(
        self,
        model_inputs: dict[str, torch.Tensor],
        *,
        temperature: float,
        module: nn.Module,
    ) -> torch.Tensor:
        """Forward pass with an explicit module, returning token log-probs only."""
        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs["attention_mask"]
            position_ids = model_inputs["position_ids"]
            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)
            output = module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            logits = output.logits
            response_length = model_inputs["responses"].shape[-1]
            logits = logits[:, -response_length - 1 : -1, :]
            del output  # free the full [B, total_seq, V] backing tensor early
            logits = self._scale_logits_by_temperature(logits, temperature=temperature)
            token_log_probs = self._chunked_logprobs_from_logits(
                logits, model_inputs["responses"]
            )
        return token_log_probs

    def _truncate_inputs_for_distill(
        self,
        model_inputs: dict[str, torch.Tensor],
        distill_len: int,
    ) -> dict[str, torch.Tensor]:
        """Truncate response segment to distill_len while preserving prompt prefix."""
        response_length = int(model_inputs["responses"].shape[-1])
        seq_length = int(model_inputs["input_ids"].shape[-1]) - response_length + int(distill_len)
        return {
            "responses": model_inputs["responses"][:, :distill_len],
            "input_ids": model_inputs["input_ids"][:, :seq_length],
            "attention_mask": model_inputs["attention_mask"][:, :seq_length],
            "position_ids": model_inputs["position_ids"][:, :seq_length],
        }

    def _build_distill_model_inputs(
        self,
        model_inputs: dict[str, torch.Tensor],
        *,
        prefix: str,
    ) -> dict[str, torch.Tensor]:
        """Extract one distillation model-input bundle from the batch."""
        return {
            "responses": model_inputs[f"{prefix}_responses"],
            "input_ids": model_inputs[f"{prefix}_input_ids"],
            "attention_mask": model_inputs[f"{prefix}_attention_mask"],
            "position_ids": model_inputs[f"{prefix}_position_ids"],
        }

    def _scale_logits_by_temperature(
        self,
        logits: torch.Tensor,
        *,
        temperature: float,
    ) -> torch.Tensor:
        """Scale logits by temperature with a no-grad in-place fast path."""
        if temperature == 1.0:
            return logits
        if torch.is_grad_enabled():
            return logits / temperature
        return logits.div_(temperature)

    def _chunked_logprobs_from_logits(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        seq_chunk_size: int = 1024,
    ) -> torch.Tensor:
        """Memory-efficient token log-probs via chunked log_softmax + gather.

        Instead of materializing a contiguous ``[B*S, V]`` float32 tensor
        (which can OOM for large vocab × long sequences), this processes
        the sequence in chunks of ``seq_chunk_size`` tokens at a time.

        Args:
            logits: ``[B, S, V]`` tensor (may be a non-contiguous view).
            labels: ``[B, S]`` token-id tensor.
            seq_chunk_size: Number of sequence positions per chunk.

        Returns:
            ``[B, S]`` tensor of per-token log-probabilities.
        """
        S = logits.shape[1]
        if S <= seq_chunk_size:
            # Small enough — use the standard (fast) path.
            return logprobs_from_logits(logits.contiguous(), labels)

        out_chunks: list[torch.Tensor] = []
        for start in range(0, S, seq_chunk_size):
            end = min(start + seq_chunk_size, S)
            # .contiguous() on the small chunk, then float32 log_softmax
            chunk_logits = logits[:, start:end, :].contiguous().float()
            chunk_labels = labels[:, start:end]
            chunk_log_probs = F.log_softmax(chunk_logits, dim=-1)
            gathered = torch.gather(
                chunk_log_probs, dim=-1, index=chunk_labels.unsqueeze(-1)
            ).squeeze(-1)
            out_chunks.append(gathered)
            del chunk_logits, chunk_log_probs
        return torch.cat(out_chunks, dim=1)

    def _row_chunked_logsumexp(
        self,
        logits: torch.Tensor,
        *,
        target_chunk_bytes: int = 1 << 30,
    ) -> torch.Tensor:
        """Compute per-row logsumexp with bounded temporary memory.

        Args:
            logits: Tensor whose last dimension is the vocabulary dimension.
            target_chunk_bytes: Approximate temporary-memory budget for each
                chunked reduction call.

        Returns:
            Tensor of shape ``logits.shape[:-1] + (1,)`` containing the
            log-sum-exp reduction over the last dimension.
        """
        if logits.numel() == 0:
            return logits.new_empty((*logits.shape[:-1], 1))

        vocab_size = int(logits.shape[-1])
        if vocab_size <= 0:
            return torch.logsumexp(logits, dim=-1, keepdim=True)

        flat_logits = logits.reshape(-1, vocab_size)
        bytes_per_row = max(1, vocab_size * flat_logits.element_size())
        chunk_rows = max(1, int(target_chunk_bytes) // bytes_per_row)
        if chunk_rows >= int(flat_logits.shape[0]):
            return torch.logsumexp(logits, dim=-1, keepdim=True)

        reduced_chunks = []
        for start in range(0, int(flat_logits.shape[0]), chunk_rows):
            stop = min(start + chunk_rows, int(flat_logits.shape[0]))
            reduced_chunks.append(torch.logsumexp(flat_logits[start:stop], dim=-1, keepdim=True))
        return torch.cat(reduced_chunks, dim=0).reshape(*logits.shape[:-1], 1)

    def _forward_topk_log_probs(
        self,
        model_inputs: dict[str, torch.Tensor],
        *,
        temperature: float,
        module: nn.Module,
        distill_topk: int,
        topk_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass that returns top-k log-probabilities for response tokens.

        Supports remove-padding mode by mapping indices to/from unpadded layout.
        """
        _, _, topk_logps, topk_indices_out = self._forward_topk_with_token_log_probs(
            model_inputs=model_inputs,
            temperature=temperature,
            module=module,
            distill_topk=distill_topk,
            topk_indices=topk_indices,
            calculate_entropy=False,
        )
        return topk_logps, topk_indices_out

    def _forward_topk_with_token_log_probs(
        self,
        model_inputs: dict[str, torch.Tensor],
        *,
        temperature: float,
        module: nn.Module,
        distill_topk: int,
        topk_indices: torch.Tensor | None = None,
        calculate_entropy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        """One-pass forward that returns token log-probs + top-k log-probs.

        This avoids a second student forward in full-logit SDPO mode by exposing
        both policy-token log-probs and distillation top-k tensors from the same
        model invocation.
        """
        if self.use_fused_kernels:
            raise ValueError("Logit distillation requires disabling fused kernels.")
        if self.use_ulysses_sp:
            raise ValueError("full_logit top-k distillation is not yet supported with ulysses sequence parallelism.")

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs["attention_mask"]
            position_ids = model_inputs["position_ids"]
            responses = model_inputs["responses"]
            response_length = int(responses.shape[-1])
            batch_size, seqlen = input_ids.shape

            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                        indices,
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros((1, 1), device=input_ids.device, dtype=input_ids.dtype)
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, 1),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros((1, 1), device=position_ids.device, dtype=position_ids.dtype)

                output = module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    use_cache=False,
                )
                logits_rmpad = output.logits.squeeze(0)
                logits_rmpad = self._scale_logits_by_temperature(
                    logits_rmpad,
                    temperature=temperature,
                )

                if topk_indices is None:
                    topk = min(int(distill_topk), int(logits_rmpad.shape[-1]))
                    topk_logits_rmpad, topk_indices_rmpad = torch.topk(logits_rmpad, topk, dim=-1)
                    return_topk_indices = True
                else:
                    topk = int(topk_indices.size(-1))
                    full_topk_indices = torch.zeros(
                        batch_size,
                        seqlen,
                        topk,
                        device=topk_indices.device,
                        dtype=topk_indices.dtype,
                    )
                    full_topk_indices[:, -response_length - 1 : -1, :] = topk_indices
                    topk_indices_rmpad = index_first_axis(
                        rearrange(full_topk_indices, "b s k -> (b s) k"),
                        indices,
                    )
                    topk_logits_rmpad = torch.gather(logits_rmpad, dim=-1, index=topk_indices_rmpad)
                    return_topk_indices = False

                logsumexp_rmpad = self._row_chunked_logsumexp(logits_rmpad)
                topk_logps_rmpad = topk_logits_rmpad - logsumexp_rmpad
                full_labels = torch.zeros((batch_size, seqlen), device=input_ids.device, dtype=torch.long)
                full_labels[:, -response_length - 1 : -1] = responses
                # flash_attn.index_first_axis expects input rank >= 2.
                labels_rmpad = index_first_axis(
                    rearrange(full_labels.unsqueeze(-1), "b s ... -> (b s) ..."),
                    indices,
                ).squeeze(-1)
                token_logits_rmpad = torch.gather(logits_rmpad, dim=-1, index=labels_rmpad.unsqueeze(-1)).squeeze(-1)
                token_logps_rmpad = token_logits_rmpad - logsumexp_rmpad.squeeze(-1)
                if is_mask_all_zero:
                    topk_logps_rmpad = topk_logps_rmpad[:0]
                    token_logps_rmpad = token_logps_rmpad[:0]
                    if return_topk_indices:
                        topk_indices_rmpad = topk_indices_rmpad[:0]

                full_topk_logps = pad_input(
                    hidden_states=topk_logps_rmpad,
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                topk_logps = full_topk_logps[:, -response_length - 1 : -1, :]
                full_token_logps = pad_input(
                    hidden_states=token_logps_rmpad.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                ).squeeze(-1)
                token_log_probs = full_token_logps[:, -response_length - 1 : -1]

                entropy = None
                if calculate_entropy:
                    log_probs_rmpad = logits_rmpad - logsumexp_rmpad
                    entropy_rmpad = -(torch.exp(log_probs_rmpad) * log_probs_rmpad).sum(dim=-1)
                    if is_mask_all_zero:
                        entropy_rmpad = entropy_rmpad[:0]
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    ).squeeze(-1)
                    entropy = full_entropy[:, -response_length - 1 : -1]

                if return_topk_indices:
                    full_topk_indices = pad_input(
                        hidden_states=topk_indices_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    topk_indices_out = full_topk_indices[:, -response_length - 1 : -1, :]
                else:
                    topk_indices_out = None

                return token_log_probs, entropy, topk_logps, topk_indices_out

            output = module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            logits = output.logits
            logits = logits[:, -response_length - 1 : -1, :]
            logits = self._scale_logits_by_temperature(logits, temperature=temperature)

            if topk_indices is None:
                topk = min(int(distill_topk), int(logits.shape[-1]))
                topk_logits, topk_indices_out = torch.topk(logits, topk, dim=-1)
            else:
                topk_logits = torch.gather(logits, dim=-1, index=topk_indices)
                topk_indices_out = None

            logsumexp = self._row_chunked_logsumexp(logits)
            token_logits = torch.gather(logits, dim=-1, index=responses.unsqueeze(-1))
            token_log_probs = (token_logits - logsumexp).squeeze(-1)
            topk_logps = topk_logits - logsumexp
            entropy = None
            if calculate_entropy:
                log_probs = logits - logsumexp
                entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1)
            return token_log_probs, entropy, topk_logps, topk_indices_out

    def _compute_sdpo_loss(
        self,
        *,
        model_inputs: dict[str, Any],
        temperature: float,
        loss_agg_mode: str,
        distill_params: SDPODistillParams,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute SDPO loss term and related metrics for a micro-batch."""
        if distill_params.loss_variant == "full_logit" and distill_params.is_clip:
            raise ValueError(
                "distill_params.is_clip is not supported when "
                "distill_params.loss_variant='full_logit'."
            )
        if distill_params.loss_variant == "full_logit" and distill_params.use_stale_coefficient:
            raise ValueError(
                "distill_params.use_stale_coefficient is not supported when "
                "distill_params.loss_variant='full_logit'."
            )
        distill_mask = model_inputs["distill_mask"].float()
        if distill_mask.ndim != 2:
            raise ValueError(f"distill_mask must have shape [B, T], got shape {tuple(distill_mask.shape)}")
        student_inputs = self._build_distill_model_inputs(
            model_inputs,
            prefix="distill_student",
        )

        teacher_module = self.actor_module
        if distill_params.teacher_regularization != "none":
            if self.teacher_module is None:
                raise ValueError(
                    "distillation teacher mode requires teacher_module, but none was attached to actor wrapper"
                )
            teacher_module = self.teacher_module

        teacher_inputs = self._build_distill_model_inputs(
            model_inputs,
            prefix="distill_teacher",
        )

        distill_len = min(
            int(distill_mask.shape[1]),
            int(student_inputs["responses"].shape[1]),
            int(teacher_inputs["responses"].shape[1]),
        )
        student_log_probs: torch.Tensor | None = None
        teacher_log_probs: torch.Tensor | None = None
        teacher_token_log_probs: torch.Tensor | None = None

        student_all_log_probs_used = None
        teacher_all_log_probs_used = None
        student_topk_log_probs_used = None
        teacher_topk_log_probs_used = None
        # When use_stale_coefficient is enabled (for negate_sdpo_loss mode),
        # the detached coefficient (s - t) uses pre-computed old log-probs
        # instead of fresh forward-pass log-probs.  This prevents the
        # positive-feedback loop where the coefficient grows as the policy
        # moves away from the teacher within PPO epochs.
        stale_student_log_probs: torch.Tensor | None = None
        stale_teacher_log_probs: torch.Tensor | None = None

        if distill_params.loss_variant == "non_full":
            student_log_probs = self._forward_with_module_log_probs(
                student_inputs,
                temperature=temperature,
                module=self.actor_module,
            )
            if distill_params.use_stale_coefficient:
                stale_s = model_inputs.get("distill_student_old_log_probs")
                stale_t = model_inputs.get("distill_teacher_old_log_probs")
                if stale_s is None or stale_t is None:
                    raise ValueError(
                        "distill_student_old_log_probs and distill_teacher_old_log_probs "
                        "are required when use_stale_coefficient is enabled."
                    )
                stale_student_log_probs = stale_s
                stale_teacher_log_probs = stale_t
                # Teacher forward pass is not needed for the coefficient;
                # skip it to save compute.
                teacher_log_probs = None
            else:
                with torch.no_grad():
                    teacher_log_prob = self._forward_with_module_log_probs(
                        teacher_inputs,
                        temperature=temperature,
                        module=teacher_module,
                    )
                teacher_log_probs = teacher_log_prob
            distill_len = min(
                distill_len,
                int(student_log_probs.shape[1]),
            )
            if teacher_log_probs is not None:
                distill_len = min(distill_len, int(teacher_log_probs.shape[1]))
        else:
            if self.use_fused_kernels:
                raise ValueError("full_logit distillation requires disabling fused kernels.")
            if self.use_ulysses_sp:
                raise ValueError("full_logit distillation is not yet supported with ulysses sequence parallelism.")
            if distill_params.full_logit_topk > 0:
                (
                    student_log_probs,
                    _,
                    student_topk_log_probs_used,
                    student_topk_indices_used,
                ) = self._forward_topk_with_token_log_probs(
                    student_inputs,
                    temperature=temperature,
                    module=self.actor_module,
                    distill_topk=distill_params.full_logit_topk,
                    topk_indices=None,
                    calculate_entropy=False,
                )
                distill_len = min(
                    distill_len,
                    int(student_log_probs.shape[1]),
                    int(student_topk_log_probs_used.shape[1]),
                )
                teacher_distill_inputs = self._truncate_inputs_for_distill(teacher_inputs, distill_len)
                with torch.no_grad():
                    teacher_token_log_probs, _, teacher_topk_log_probs_used, _ = self._forward_topk_with_token_log_probs(
                        teacher_distill_inputs,
                        temperature=temperature,
                        module=teacher_module,
                        distill_topk=distill_params.full_logit_topk,
                        topk_indices=student_topk_indices_used,
                    )
                distill_len = min(distill_len, int(teacher_topk_log_probs_used.shape[1]))
            else:
                student_log_probs, student_all_log_probs_used = self._forward_with_all_log_probs(
                    model_inputs=student_inputs,
                    temperature=temperature,
                    module=self.actor_module,
                )
                with torch.no_grad():
                    teacher_token_log_probs, teacher_all_log_probs_used = self._forward_with_all_log_probs(
                        model_inputs=teacher_inputs,
                        temperature=temperature,
                        module=teacher_module,
                    )
                distill_len = min(
                    distill_len,
                    int(student_log_probs.shape[1]),
                    int(student_all_log_probs_used.shape[1]),
                    int(teacher_all_log_probs_used.shape[1]),
                )

        if student_log_probs is None:
            raise ValueError("student_log_probs must be computed for SDPO loss.")
        if distill_len <= 0:
            zero = student_log_probs.sum() * 0.0
            return zero, _zero_sdpo_metrics()

        student_log_probs = student_log_probs[:, :distill_len]
        distill_mask = distill_mask[:, :distill_len].to(
            device=student_log_probs.device,
            dtype=student_log_probs.dtype,
        )
        if teacher_log_probs is not None:
            teacher_log_probs = teacher_log_probs[:, :distill_len].to(
                device=student_log_probs.device,
                dtype=student_log_probs.dtype,
            )
        if student_all_log_probs_used is not None:
            student_all_log_probs_used = student_all_log_probs_used[:, :distill_len]
        if teacher_all_log_probs_used is not None:
            teacher_all_log_probs_used = teacher_all_log_probs_used[:, :distill_len]
        if student_topk_log_probs_used is not None:
            student_topk_log_probs_used = student_topk_log_probs_used[:, :distill_len]
        if teacher_topk_log_probs_used is not None:
            teacher_topk_log_probs_used = teacher_topk_log_probs_used[:, :distill_len]

        token_count = float(distill_mask.sum().item())
        active_seq_count = float(_count_active_distill_sequences(distill_mask))
        sdpo_adv_abs_mean = 0.0
        sdpo_adv_abs_max = 0.0
        advantage: torch.Tensor | None = None

        if stale_student_log_probs is not None and stale_teacher_log_probs is not None:
            # Stale-coefficient path: coefficient from pre-computed old log-probs,
            # gradient only through fresh student_log_probs.
            stale_s = stale_student_log_probs[:, :distill_len].to(
                device=student_log_probs.device, dtype=student_log_probs.dtype,
            )
            stale_t = stale_teacher_log_probs[:, :distill_len].to(
                device=student_log_probs.device, dtype=student_log_probs.dtype,
            )
            delta = (stale_s - stale_t).detach()
            advantage = _segmented_reverse_cumsum(delta, distill_mask)
            per_token_loss = advantage * student_log_probs
        elif distill_params.loss_variant == "non_full":
            # Cumulative future log-ratio advantage (unbiased token-level).
            delta = (student_log_probs - teacher_log_probs).detach()
            advantage = _segmented_reverse_cumsum(delta, distill_mask)
            per_token_loss = advantage * student_log_probs
        else:
            # full_logit: KL/JSD over distributions — not REINFORCE-style.
            per_token_loss = _compute_sdpo_per_token_loss(
                student_log_probs=student_log_probs,
                teacher_log_probs=teacher_log_probs,
                loss_variant=distill_params.loss_variant,
                alpha=distill_params.alpha,
                student_all_log_probs=student_all_log_probs_used,
                teacher_all_log_probs=teacher_all_log_probs_used,
                student_topk_log_probs=student_topk_log_probs_used,
                teacher_topk_log_probs=teacher_topk_log_probs_used,
                full_logit_topk=distill_params.full_logit_topk,
                full_logit_add_tail=distill_params.full_logit_add_tail,
            )
        if advantage is not None and token_count > 0:
            masked_advantage = advantage[distill_mask > 0]
            sdpo_adv_abs_mean = float(masked_advantage.abs().mean().item())
            sdpo_adv_abs_max = float(masked_advantage.abs().max().item())

        # --- Diagnostic metrics (no gradient impact) ---
        diag_mean_student_logp = 0.0
        diag_mean_teacher_logp = 0.0
        diag_student_entropy = 0.0
        diag_teacher_entropy = 0.0
        diag_kl_per_token_mean = 0.0

        if token_count > 0:
            diag_mean_student_logp = float((student_log_probs.detach() * distill_mask).sum().item() / token_count)
            if teacher_token_log_probs is not None:
                t_logp = teacher_token_log_probs[:, :distill_len].to(
                    device=student_log_probs.device, dtype=student_log_probs.dtype,
                )
                diag_mean_teacher_logp = float((t_logp * distill_mask).sum().item() / token_count)
            elif teacher_log_probs is not None:
                diag_mean_teacher_logp = float((teacher_log_probs.detach() * distill_mask).sum().item() / token_count)

            if student_topk_log_probs_used is not None and teacher_topk_log_probs_used is not None:
                with torch.no_grad():
                    s_ent = _topk_entropy(student_topk_log_probs_used, distill_params.full_logit_add_tail)
                    t_ent = _topk_entropy(teacher_topk_log_probs_used, distill_params.full_logit_add_tail)
                diag_student_entropy = float((s_ent * distill_mask).sum().item() / token_count)
                diag_teacher_entropy = float((t_ent * distill_mask).sum().item() / token_count)
            elif (
                student_all_log_probs_used is not None
                and teacher_all_log_probs_used is not None
                and student_all_log_probs_used.shape[-1] <= 10000
            ):
                # Guard: skip full-vocab entropy for large vocabularies to avoid OOM.
                with torch.no_grad():
                    s_ent = -(torch.exp(student_all_log_probs_used) * student_all_log_probs_used).sum(dim=-1)
                    t_ent = -(torch.exp(teacher_all_log_probs_used) * teacher_all_log_probs_used).sum(dim=-1)
                diag_student_entropy = float((s_ent * distill_mask).sum().item() / token_count)
                diag_teacher_entropy = float((t_ent * distill_mask).sum().item() / token_count)

            if distill_params.loss_variant == "full_logit":
                diag_kl_per_token_mean = float((per_token_loss.detach() * distill_mask).sum().item() / token_count)

        if distill_params.negate_sdpo_loss:
            per_token_loss = -per_token_loss

        sdpo_clipfrac = 0.0
        sdpo_ratio_gt_clip_frac = 0.0
        if distill_params.is_clip:
            student_old_log_prob = model_inputs.get("distill_student_old_log_probs")
            if student_old_log_prob is None:
                raise ValueError(
                    "distill_student_old_log_probs is required when distill_is_clip is enabled."
                )
            student_old_log_prob = student_old_log_prob[:, :distill_len].to(
                device=student_log_probs.device,
                dtype=student_log_probs.dtype,
            )
            negative_approx_kl = (student_log_probs - student_old_log_prob).detach()
            negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
            ratio = torch.exp(negative_approx_kl)
            clip_low = _resolve_clip_ratio_bound(getattr(self.config, "clip_ratio_low", None), self.config.clip_ratio)
            clip_high = _resolve_clip_ratio_bound(getattr(self.config, "clip_ratio_high", None), self.config.clip_ratio)
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
            sdpo_unclipped = per_token_loss * ratio
            sdpo_clipped = per_token_loss * clipped_ratio
            per_token_loss = torch.maximum(sdpo_unclipped, sdpo_clipped)
            if token_count > 0:
                ratio_outside_clip = ((ratio < (1.0 - clip_low)) | (ratio > (1.0 + clip_high))).to(distill_mask.dtype)
                sdpo_ratio_gt_clip_frac = float((ratio_outside_clip * distill_mask).sum().item() / token_count)
                # This metric tracks how often the pessimistic clipped branch is selected.
                # It is intentionally distinct from ratio-out-of-bounds frequency.
                sdpo_clipfrac = float(((sdpo_clipped > sdpo_unclipped) * distill_mask).sum() / token_count)

        if token_count <= 0:
            # Keep graph/collective structure aligned across ranks even when this
            # rank has no distill-valid tokens in the micro-batch.
            zero_sdpo_loss = per_token_loss.sum() * 0.0
            return zero_sdpo_loss, _zero_sdpo_metrics()

        sdpo_loss = agg_loss(
            loss_mat=per_token_loss,
            loss_mask=distill_mask,
            loss_agg_mode="seq-mean-token-mean",
        )
        return sdpo_loss, {
            # This is the reduced, unscaled SDPO monitoring loss for the
            # current micro-batch before lambda or active-sequence scaling.
            "distill/sdpo_loss": float(sdpo_loss.detach().item()),
            "distill/token_count": token_count,
            "distill/sdpo_clipfrac": sdpo_clipfrac,
            "distill/sdpo_ratio_gt_clip_frac": sdpo_ratio_gt_clip_frac,
            "distill/sdpo_adv_abs_mean": sdpo_adv_abs_mean,
            "distill/sdpo_adv_abs_max": sdpo_adv_abs_max,
            "distill/active_seq_count": active_seq_count,
            "distill/diag_mean_student_logp": diag_mean_student_logp,
            "distill/diag_mean_teacher_logp": diag_mean_teacher_logp,
            "distill/diag_student_entropy": diag_student_entropy,
            "distill/diag_teacher_entropy": diag_teacher_entropy,
            "distill/diag_kl_per_token_mean": diag_kl_per_token_mean,
        }

    def update_policy(self, data: DataProto) -> dict[str, list[float]]:
        """Update actor policy with configurable GRPO + SDPO objective."""
        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        distill_params = self._parse_distill_params(data.meta_info)

        distill_required_keys = {
            "distill_student_input_ids",
            "distill_student_attention_mask",
            "distill_student_position_ids",
            "distill_student_responses",
            "distill_teacher_input_ids",
            "distill_teacher_attention_mask",
            "distill_teacher_position_ids",
            "distill_teacher_responses",
            "distill_mask",
        }
        use_distill = distill_params.enabled and distill_required_keys.issubset(set(data.batch.keys()))
        if use_distill and distill_params.is_clip and "distill_student_old_log_probs" not in data.batch:
            raise ValueError(
                "distill_student_old_log_probs is required when distill_is_clip is enabled."
            )
        if use_distill and distill_params.use_stale_coefficient:
            for _key in ("distill_student_old_log_probs", "distill_teacher_old_log_probs"):
                if _key not in data.batch:
                    raise ValueError(f"{_key} is required when use_stale_coefficient is enabled.")

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if use_distill:
            select_keys.extend(sorted(distill_required_keys))
            if "distill_student_response_mask" in data.batch:
                select_keys.append("distill_student_response_mask")
            if "distill_teacher_response_mask" in data.batch:
                select_keys.append("distill_teacher_response_mask")
            if distill_params.is_clip or distill_params.use_stale_coefficient:
                select_keys.append("distill_student_old_log_probs")
            if distill_params.use_stale_coefficient:
                select_keys.append("distill_teacher_old_log_probs")
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        if use_distill and has_multi_modal_inputs:
            raise ValueError("Multi-modal inputs are not supported with distillation.")
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        mini_batches = data.split(self.config.ppo_mini_batch_size)
        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics: dict[str, list[float]] = {}
        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                mini_batch_distill_seq_count = 0
                if use_distill:
                    mini_batch_distill_seq_count = _count_active_distill_sequences(mini_batch.batch["distill_mask"])

                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode
                    calculate_entropy = entropy_coeff != 0

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation
                    sdpo_loss_scale_factor = 0.0
                    if use_distill and mini_batch_distill_seq_count > 0:
                        sdpo_loss_scale_factor = (
                            _count_active_distill_sequences(model_inputs["distill_mask"]) / mini_batch_distill_seq_count
                        )

                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                    )

                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        old_log_prob = log_prob.detach() if on_policy else model_inputs["old_log_probs"]

                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)
                    pg_loss, pg_metrics = self._compute_pg_loss(
                        use_grpo_loss=distill_params.use_grpo_loss,
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        rollout_is_weights=rollout_is_weights,
                    )

                    micro_batch_metrics: dict[str, float] = dict(pg_metrics)
                    micro_batch_metrics["distill/use_grpo_loss"] = 1.0 if distill_params.use_grpo_loss else 0.0
                    policy_loss = pg_loss
                    loss = policy_loss * loss_scale_factor
                    active_seq_count = 0.0
                    active_seq_frac = 0.0
                    sdpo_loss_scaled = 0.0

                    if use_distill:
                        sdpo_loss, sdpo_metrics = self._compute_sdpo_loss(
                            model_inputs=model_inputs,
                            temperature=temperature,
                            loss_agg_mode=loss_agg_mode,
                            distill_params=distill_params,
                        )
                        active_seq_count = sdpo_metrics["distill/active_seq_count"]
                        active_seq_frac = active_seq_count / max(float(response_mask.shape[0]), 1.0)
                        sdpo_loss_scaled = (
                            distill_params.lambda_coef * sdpo_metrics["distill/sdpo_loss"] * sdpo_loss_scale_factor
                        )
                        loss = loss + distill_params.lambda_coef * sdpo_loss * sdpo_loss_scale_factor
                        # distill/sdpo_loss remains a reduced monitoring scalar.
                        # distill/sdpo_loss_scaled mirrors the exact optimizer term.
                        micro_batch_metrics["distill/sdpo_loss"] = sdpo_metrics["distill/sdpo_loss"] * sdpo_loss_scale_factor
                        micro_batch_metrics["distill/token_count"] = sdpo_metrics["distill/token_count"]
                        micro_batch_metrics["distill/sdpo_clipfrac"] = sdpo_metrics["distill/sdpo_clipfrac"]
                        micro_batch_metrics["distill/sdpo_ratio_gt_clip_frac"] = (
                            sdpo_metrics["distill/sdpo_ratio_gt_clip_frac"]
                        )
                        micro_batch_metrics["distill/sdpo_adv_abs_mean"] = sdpo_metrics["distill/sdpo_adv_abs_mean"]
                        micro_batch_metrics["distill/sdpo_adv_abs_max"] = sdpo_metrics["distill/sdpo_adv_abs_max"]
                        micro_batch_metrics["distill/diag_mean_student_logp"] = sdpo_metrics["distill/diag_mean_student_logp"]
                        micro_batch_metrics["distill/diag_mean_teacher_logp"] = sdpo_metrics["distill/diag_mean_teacher_logp"]
                        micro_batch_metrics["distill/diag_student_entropy"] = sdpo_metrics["distill/diag_student_entropy"]
                        micro_batch_metrics["distill/diag_teacher_entropy"] = sdpo_metrics["distill/diag_teacher_entropy"]
                        micro_batch_metrics["distill/diag_kl_per_token_mean"] = sdpo_metrics["distill/diag_kl_per_token_mean"]
                    else:
                        micro_batch_metrics["distill/sdpo_loss"] = 0.0
                        micro_batch_metrics["distill/token_count"] = 0.0
                        micro_batch_metrics["distill/sdpo_clipfrac"] = 0.0
                        micro_batch_metrics["distill/sdpo_ratio_gt_clip_frac"] = 0.0
                        micro_batch_metrics["distill/sdpo_adv_abs_mean"] = 0.0
                        micro_batch_metrics["distill/sdpo_adv_abs_max"] = 0.0
                        micro_batch_metrics["distill/diag_mean_student_logp"] = 0.0
                        micro_batch_metrics["distill/diag_mean_teacher_logp"] = 0.0
                        micro_batch_metrics["distill/diag_student_entropy"] = 0.0
                        micro_batch_metrics["distill/diag_teacher_entropy"] = 0.0
                        micro_batch_metrics["distill/diag_kl_per_token_mean"] = 0.0

                    micro_batch_metrics["distill/active_seq_count"] = active_seq_count
                    micro_batch_metrics["distill/active_seq_frac"] = active_seq_frac
                    micro_batch_metrics["distill/sdpo_loss_scaled"] = sdpo_loss_scaled

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        loss = loss - entropy_loss * entropy_coeff * loss_scale_factor

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        loss = loss + kl_loss * self.config.kl_loss_coef * loss_scale_factor
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = float(self.config.kl_loss_coef)
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": float(grad_norm.detach().item())})

        self.actor_optimizer.zero_grad()
        return metrics


__all__ = [
    "SDPODataParallelPPOActor",
    "SDPODistillParams",
    "_compute_sdpo_per_token_loss",
    "_count_active_distill_sequences",
    "_segmented_reverse_cumsum",
]
