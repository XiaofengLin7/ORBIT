"""Joint SDPO self-distillation trainer for multi-episode PPO.

This module implements shared-weight self-distillation where each training step
performs:
1. Standard teacher RL update on full multi-episode rollouts.
2. SDPO distillation loss, optionally combined with PPO, in the same actor update.

Distillation uses a strict context-overflow guard on denominator scoring length:
``L_den = len(denominator_prompt_tokens) + len(first_attempt_sequence_tokens)``,
where ``denominator_prompt_tokens = concat(c_N, prompts)``.
If ``L_den > context_limit``, the sample is excluded from distillation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto  # type: ignore
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics

from trainers.multi_episode_trainer import MultiEpisodeAgentPPOTrainer


@dataclass
class DistillSettings:
    """Configuration for SDPO self-distillation."""

    enable: bool = False
    lambda_coef: float = 1.0
    mode: str = "sdpo_self"
    use_grpo_loss: bool = True
    denominator_mode: str = "teacher_adapted_feedback"
    context_limit: int | None = None
    context_overflow_policy: str = "skip_loss"
    min_distill_tokens: int = 1
    loss_variant: str = "non_full"
    alpha: float = 1.0
    is_clip: float | None = None
    full_logit_topk: int = 64
    full_logit_add_tail: bool = True
    teacher_context_attempts: int | None = None
    teacher_regularization: str = "none"
    teacher_update_rate: float = 0.05
    teacher_update_interval: int = 10


@dataclass
class DistillPayload:
    """Prepared distillation payload for one optimization step."""

    denominator_batch: DataProto
    distill_mask: torch.Tensor
    kept_indices: torch.Tensor
    skipped_context_overflow: int
    total_samples: int
    kept_samples: int


def build_hindsight_prompt_tokens_first_n_complete_attempts(
    step_records: Sequence[dict[str, Any]],
    teacher_context_attempts: int | None,
) -> torch.Tensor | None:
    """Construct hindsight context from first-N complete attempts.

    Args:
        step_records: Step-level records from rollout engine.
        teacher_context_attempts: Number of complete attempts to include. If
            ``None``, include all transition-confirmed complete attempts.

    Returns:
        Tensor prompt tokens for denominator context, or ``None`` when records
        are malformed, non-cumulative, missing/invalid boundary metadata at
        transitions, or insufficient complete attempts exist.

    Notes:
        A complete attempt is usually confirmed by a transition to a later
        ``episode_index`` with valid boundary metadata on the previous step.
        For trajectories that end immediately after an episode completes
        (e.g., total step cap reached, no next reset), we also treat the final
        completed attempt as available context using the accumulated
        prompt+completion prefix from the last step.
    """
    if not step_records:
        return None

    first_prompt_ids = step_records[0].get("prompt_ids", [])
    first_completion_ids = step_records[0].get("completion_ids", [])
    if not isinstance(first_prompt_ids, list) or not isinstance(first_completion_ids, list):
        return None

    first_episode_index = step_records[0].get("episode_index", 0)
    try:
        prev_episode_index = int(first_episode_index)
    except (TypeError, ValueError):
        return None

    accumulated = list(first_prompt_ids) + list(first_completion_ids)
    prev_step = step_records[0]
    complete_attempt_contexts: list[list[int]] = []
    for step in step_records[1:]:
        current_prompt_ids = step.get("prompt_ids", [])
        current_completion_ids = step.get("completion_ids", [])
        current_episode_index_raw = step.get("episode_index", prev_episode_index)
        if not isinstance(current_prompt_ids, list) or not isinstance(current_completion_ids, list):
            return None
        try:
            current_episode_index = int(current_episode_index_raw)
        except (TypeError, ValueError):
            return None
        if current_episode_index < prev_episode_index:
            return None
        if len(current_prompt_ids) < len(accumulated):
            return None
        if current_prompt_ids[: len(accumulated)] != accumulated:
            return None

        if current_episode_index > prev_episode_index:
            delta_prompt_tokens = current_prompt_ids[len(accumulated) :]
            boundary_transition = bool(prev_step.get("boundary_transition", False))
            terminal_len_raw = prev_step.get("boundary_terminal_env_token_len", None)
            next_initial_len_raw = prev_step.get("boundary_next_initial_env_token_len", None)
            try:
                terminal_len = int(terminal_len_raw)
                next_initial_len = int(next_initial_len_raw)
            except (TypeError, ValueError):
                return None
            if not boundary_transition:
                return None
            if terminal_len < 0 or next_initial_len < 0:
                return None
            if terminal_len + next_initial_len != len(delta_prompt_tokens):
                return None
            complete_attempt_contexts.append(list(accumulated) + list(delta_prompt_tokens[:terminal_len]))

        accumulated = current_prompt_ids + list(current_completion_ids)
        prev_episode_index = current_episode_index
        prev_step = step

    # If the trajectory ends right after an episode completion, there is no
    # subsequent step to expose a transition. Count this terminal completed
    # attempt so strict teacher_context_attempts does not require a "next"
    # episode that never exists at step-cap end.
    last_step = step_records[-1]
    if bool(last_step.get("episode_done", False)):
        complete_attempt_contexts.append(list(accumulated))

    if not complete_attempt_contexts:
        return None

    if teacher_context_attempts is None:
        selected_context = complete_attempt_contexts[-1]
    else:
        if len(complete_attempt_contexts) < teacher_context_attempts:
            return None
        selected_context = complete_attempt_contexts[teacher_context_attempts - 1]

    return torch.as_tensor(selected_context, dtype=torch.long)


def should_skip_denominator_overflow(
    denominator_prompt_len: int,
    first_attempt_sequence_len: int,
    context_limit: int,
) -> bool:
    """Return whether SDPO distillation should be skipped for overflow.

    The denominator-scored sequence length is:
    ``L_den = len(denominator_prompt_tokens) + len(first_attempt_sequence_tokens)``.
    """
    return (denominator_prompt_len + first_attempt_sequence_len) > context_limit


def extract_first_attempt_prefix(
    response_tokens: torch.Tensor,
    response_mask: torch.Tensor,
    first_attempt_response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Extract first-attempt prefix sequence and aligned masks.

    The prefix ends at the last token position where
    ``first_attempt_response_mask == 1`` (and valid by ``response_mask``). This
    keeps interleaved env/model tokens that appear before that endpoint.

    Args:
        response_tokens: Response token row for one sample.
        response_mask: Valid response-token mask for one sample.
        first_attempt_response_mask: First-attempt model-token mask for one sample.

    Returns:
        Tuple ``(prefix_tokens, prefix_response_mask, prefix_distill_mask)`` or
        ``None`` when no valid first-attempt token exists.
    """
    valid_first_attempt = (first_attempt_response_mask > 0) & (response_mask > 0)
    valid_positions = torch.nonzero(valid_first_attempt, as_tuple=False).flatten()
    if valid_positions.numel() == 0:
        return None

    prefix_len = int(valid_positions[-1].item()) + 1
    prefix_tokens = response_tokens[:prefix_len]
    prefix_response_mask = response_mask[:prefix_len].long()
    prefix_distill_mask = (
        response_mask[:prefix_len].float() * first_attempt_response_mask[:prefix_len].float()
    )
    return prefix_tokens, prefix_response_mask, prefix_distill_mask


class JointSDPOSelfDistillTrainer(MultiEpisodeAgentPPOTrainer):
    """Multi-episode PPO trainer with SDPO self-distillation auxiliary update."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize trainer and parse distillation settings."""
        super().__init__(**kwargs)
        self.distill_settings = self._load_distill_settings()
        self._latest_token_trajectories: list[dict[str, Any]] = []

    def _load_distill_settings(self) -> DistillSettings:
        """Load distillation settings from config with robust defaults."""
        default_limit = int(self.config.data.max_prompt_length) + int(self.config.data.max_response_length)
        cfg = self.config.rllm.get("distill", {})
        if cfg is None:
            cfg = {}
        mode = str(cfg.get("mode", "sdpo_self")).lower()
        valid_modes = {"sdpo_self", "sdpo_pure"}
        if mode not in valid_modes:
            raise ValueError(
                f"rllm.distill.mode must be one of {sorted(valid_modes)}, got {mode}"
            )
        use_grpo_loss = mode == "sdpo_self"
        teacher_regularization = str(cfg.get("teacher_regularization", "none")).lower()
        valid_teacher_regularization = {"none", "ema", "every_n_steps"}
        if teacher_regularization not in valid_teacher_regularization:
            raise ValueError(
                "rllm.distill.teacher_regularization must be one of "
                f"{sorted(valid_teacher_regularization)}, got {teacher_regularization}"
            )
        teacher_update_rate = float(cfg.get("teacher_update_rate", 0.05))
        teacher_update_interval = int(cfg.get("teacher_update_interval", 10))
        if teacher_regularization == "ema" and not (0.0 <= teacher_update_rate <= 1.0):
            raise ValueError(
                f"rllm.distill.teacher_update_rate must be in [0, 1] for ema mode, got {teacher_update_rate}"
            )
        if teacher_regularization == "every_n_steps" and teacher_update_interval < 1:
            raise ValueError(
                "rllm.distill.teacher_update_interval must be >= 1 for every_n_steps mode, "
                f"got {teacher_update_interval}"
            )
        use_kl_in_reward = bool(getattr(self.config.algorithm, "use_kl_in_reward", False))
        actor_cfg = getattr(self.config.actor_rollout_ref, "actor", None)
        use_kl_loss = bool(getattr(actor_cfg, "use_kl_loss", False)) if actor_cfg is not None else False
        if teacher_regularization != "none" and (use_kl_in_reward or use_kl_loss):
            raise ValueError(
                "Teacher-regularized distillation (rllm.distill.teacher_regularization != 'none') "
                "is incompatible with KL reference-policy path. Disable algorithm.use_kl_in_reward "
                "and actor_rollout_ref.actor.use_kl_loss."
            )
        teacher_context_attempts_raw = cfg.get("teacher_context_attempts", None)
        teacher_context_attempts = (
            int(teacher_context_attempts_raw) if teacher_context_attempts_raw is not None else None
        )
        if teacher_context_attempts is not None and teacher_context_attempts < 1:
            raise ValueError(
                f"rllm.distill.teacher_context_attempts must be >= 1 when provided, got {teacher_context_attempts}"
            )
        loss_variant = str(cfg.get("loss_variant", "non_full")).lower()
        if loss_variant not in {"non_full", "full_logit"}:
            raise ValueError(
                f"rllm.distill.loss_variant must be one of ['non_full', 'full_logit'], got {loss_variant}"
            )
        alpha = float(cfg.get("alpha", 1.0))
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"rllm.distill.alpha must be in [0, 1], got {alpha}")
        if loss_variant == "non_full" and alpha != 1.0:
            raise ValueError("rllm.distill.alpha must be 1.0 when rllm.distill.loss_variant='non_full'.")
        is_clip_raw = cfg.get("is_clip", None)
        is_clip = None if is_clip_raw is None else float(is_clip_raw)
        if is_clip is not None and is_clip <= 0:
            raise ValueError(f"rllm.distill.is_clip must be positive when provided, got {is_clip}")
        full_logit_topk = int(cfg.get("full_logit_topk", 64))
        if full_logit_topk <= 0:
            raise ValueError(f"rllm.distill.full_logit_topk must be a positive integer, got {full_logit_topk}")
        return DistillSettings(
            enable=bool(cfg.get("enable", False)),
            lambda_coef=float(cfg.get("lambda", 1.0)),
            mode=mode,
            use_grpo_loss=use_grpo_loss,
            denominator_mode=str(cfg.get("denominator_mode", "teacher_adapted_feedback")),
            context_limit=int(cfg.get("context_limit", default_limit)) if cfg.get("context_limit", None) is not None else default_limit,
            context_overflow_policy=str(cfg.get("context_overflow_policy", "skip_loss")),
            min_distill_tokens=int(cfg.get("min_distill_tokens", 1)),
            loss_variant=loss_variant,
            alpha=alpha,
            is_clip=is_clip,
            full_logit_topk=full_logit_topk,
            full_logit_add_tail=bool(cfg.get("full_logit_add_tail", True)),
            teacher_context_attempts=teacher_context_attempts,
            teacher_regularization=teacher_regularization,
            teacher_update_rate=teacher_update_rate,
            teacher_update_interval=teacher_update_interval,
        )

    def _transform_agent_trajectories(
        self,
        trajectories: list[dict[str, Any]],
    ) -> tuple[DataProto, dict[str, Any]]:
        """Transform trajectories and retain distillation metadata."""
        final_gen_batch_output, metrics = super()._transform_agent_trajectories(trajectories)
        self._latest_token_trajectories = trajectories

        if not trajectories:
            return final_gen_batch_output, metrics

        response_len = final_gen_batch_output.batch["responses"].shape[1]
        first_attempt_masks = torch.zeros(
            (len(trajectories), response_len),
            dtype=torch.float32,
        )
        for i, traj in enumerate(trajectories):
            mask = traj.get("first_attempt_response_mask")
            if mask is None:
                continue
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask, dtype=torch.float32)
            usable = min(response_len, int(mask.numel()))
            if usable > 0:
                first_attempt_masks[i, :usable] = mask[:usable].float()
        final_gen_batch_output.batch["first_attempt_response_mask"] = first_attempt_masks
        final_gen_batch_output.batch["distill_traj_idx"] = torch.arange(
            len(trajectories),
            dtype=torch.long,
        )
        return final_gen_batch_output, metrics

    def _prepare_distill_payload(self, batch: DataProto) -> tuple[DistillPayload | None, dict[str, float]]:
        """Prepare denominator batch and token masks for merged distillation."""
        if not self.distill_settings.enable:
            return None, {}
        if not self._latest_token_trajectories:
            return None, {
                "distill/skipped_context_overflow": 0.0,
                "distill/skipped_hindsight_unavailable": 0.0,
                "distill/kept_ratio": 0.0,
            }

        response_mask = batch.batch["response_mask"].float()
        first_attempt_mask = batch.batch.get("first_attempt_response_mask")
        if first_attempt_mask is None:
            return None, {
                "distill/skipped_context_overflow": 0.0,
                "distill/skipped_hindsight_unavailable": 0.0,
                "distill/kept_ratio": 0.0,
            }
        first_attempt_mask = first_attempt_mask.float()
        distill_mask = response_mask * first_attempt_mask

        traj_indices = batch.batch.get("distill_traj_idx")
        if traj_indices is None:
            traj_indices = torch.arange(
                batch.batch["responses"].shape[0],
                dtype=torch.long,
            )
        elif not isinstance(traj_indices, torch.Tensor):
            traj_indices = torch.as_tensor(traj_indices, dtype=torch.long)
        else:
            traj_indices = traj_indices.long()

        total_samples = int(batch.batch["responses"].shape[0])
        original_response_len = int(batch.batch["responses"].shape[1])
        kept_indices: list[int] = []
        denominator_prompt_tokens_per_sample: list[torch.Tensor] = []
        first_attempt_prefix_response_tokens_per_sample: list[torch.Tensor] = []
        first_attempt_prefix_response_masks_per_sample: list[torch.Tensor] = []
        first_attempt_prefix_distill_masks_per_sample: list[torch.Tensor] = []
        skipped_context_overflow = 0
        skipped_hindsight_unavailable = 0

        limit = int(
            self.distill_settings.context_limit
            if self.distill_settings.context_limit is not None
            else (int(self.config.data.max_prompt_length) + int(self.config.data.max_response_length))
        )
        if self.distill_settings.context_overflow_policy != "skip_loss":
            raise ValueError(
                f"Unsupported context_overflow_policy={self.distill_settings.context_overflow_policy}. "
                "Only 'skip_loss' is currently supported."
            )

        for i in range(total_samples):
            if float(distill_mask[i].sum().item()) <= 0:
                continue

            traj_idx = int(traj_indices[i].item())
            if traj_idx < 0 or traj_idx >= len(self._latest_token_trajectories):
                skipped_hindsight_unavailable += 1
                continue
            trajectory = self._latest_token_trajectories[traj_idx]
            if not isinstance(trajectory, dict):
                skipped_hindsight_unavailable += 1
                continue

            step_records = trajectory.get("step_records", [])
            if not isinstance(step_records, list):
                skipped_hindsight_unavailable += 1
                continue

            first_attempt_prefix = extract_first_attempt_prefix(
                response_tokens=batch.batch["responses"][i],
                response_mask=response_mask[i],
                first_attempt_response_mask=first_attempt_mask[i],
            )
            if first_attempt_prefix is None:
                continue
            (
                prefix_response_tokens,
                prefix_response_mask,
                prefix_distill_mask,
            ) = first_attempt_prefix

            teacher_prompt_tokens = build_hindsight_prompt_tokens_first_n_complete_attempts(
                step_records=step_records,
                teacher_context_attempts=self.distill_settings.teacher_context_attempts,
            )
            if teacher_prompt_tokens is None:
                skipped_hindsight_unavailable += 1
                continue

            prompt_attention_row = batch.batch["attention_mask"][i, :-original_response_len]
            student_prompt_tokens = batch.batch["prompts"][i][prompt_attention_row > 0].long()
            denominator_prompt_tokens = torch.cat(
                [teacher_prompt_tokens.long(), student_prompt_tokens],
                dim=0,
            )

            denominator_prompt_len = int(denominator_prompt_tokens.numel())
            first_attempt_prefix_len = int(prefix_response_tokens.numel())

            if should_skip_denominator_overflow(
                denominator_prompt_len=denominator_prompt_len,
                first_attempt_sequence_len=first_attempt_prefix_len,
                context_limit=limit,
            ):
                skipped_context_overflow += 1
                continue

            kept_indices.append(i)
            denominator_prompt_tokens_per_sample.append(denominator_prompt_tokens)
            first_attempt_prefix_response_tokens_per_sample.append(prefix_response_tokens)
            first_attempt_prefix_response_masks_per_sample.append(prefix_response_mask)
            first_attempt_prefix_distill_masks_per_sample.append(prefix_distill_mask)

        if not kept_indices:
            return None, {
                "distill/skipped_context_overflow": float(skipped_context_overflow),
                "distill/skipped_hindsight_unavailable": float(skipped_hindsight_unavailable),
                "distill/kept_ratio": 0.0,
            }

        selected_den = batch.select_idxs(np.array(kept_indices))

        pad_token_id = int(self.tokenizer.pad_token_id)
        prompt_rows: list[torch.Tensor] = denominator_prompt_tokens_per_sample
        max_prompt_len = max(int(x.numel()) for x in prompt_rows)
        max_response_len = max(
            int(x.numel()) for x in first_attempt_prefix_response_tokens_per_sample
        )

        denom_prompts = torch.full(
            (len(prompt_rows), max_prompt_len),
            fill_value=pad_token_id,
            dtype=torch.long,
        )
        for i, row in enumerate(prompt_rows):
            denom_prompts[i, -row.numel() :] = row

        responses = torch.full(
            (len(prompt_rows), max_response_len),
            fill_value=pad_token_id,
            dtype=torch.long,
        )
        prefix_response_mask = torch.zeros(
            (len(prompt_rows), max_response_len),
            dtype=torch.long,
        )
        selected_distill_mask = torch.zeros(
            (len(prompt_rows), max_response_len),
            dtype=torch.float32,
        )
        for i, row in enumerate(first_attempt_prefix_response_tokens_per_sample):
            row_len = int(row.numel())
            responses[i, :row_len] = row
            prefix_response_mask[i, :row_len] = first_attempt_prefix_response_masks_per_sample[i]
            selected_distill_mask[i, :row_len] = first_attempt_prefix_distill_masks_per_sample[i]

        # Denominator batch uses teacher hindsight prompt with same first-attempt prefix responses.
        prompt_lens = torch.as_tensor([int(x.numel()) for x in prompt_rows], dtype=torch.long)
        prompt_pos = torch.arange(max_prompt_len).unsqueeze(0)
        denominator_prompt_attention = (prompt_pos >= (max_prompt_len - prompt_lens.unsqueeze(1))).long()
        denominator_attention_mask = torch.cat([denominator_prompt_attention, prefix_response_mask], dim=1)
        denominator_position_ids = (torch.cumsum(denominator_attention_mask, dim=1) - 1) * denominator_attention_mask
        denominator_input_ids = torch.cat([denom_prompts, responses], dim=1)

        selected_den.batch["prompts"] = denom_prompts
        selected_den.batch["responses"] = responses
        selected_den.batch["response_mask"] = prefix_response_mask
        selected_den.batch["input_ids"] = denominator_input_ids
        selected_den.batch["attention_mask"] = denominator_attention_mask
        selected_den.batch["position_ids"] = denominator_position_ids

        kept_samples = int(selected_den.batch["responses"].shape[0])
        payload = DistillPayload(
            denominator_batch=selected_den,
            distill_mask=selected_distill_mask,
            kept_indices=torch.as_tensor(kept_indices, dtype=torch.long),
            skipped_context_overflow=skipped_context_overflow,
            total_samples=total_samples,
            kept_samples=kept_samples,
        )
        metrics = {
            "distill/skipped_context_overflow": float(skipped_context_overflow),
            "distill/skipped_hindsight_unavailable": float(skipped_hindsight_unavailable),
            "distill/kept_ratio": float(kept_samples / max(1, total_samples)),
        }
        return payload, metrics

    def _set_distill_meta(self, batch: DataProto, *, enabled_for_step: bool) -> None:
        """Attach per-step distillation settings in batch meta info for actor update."""
        batch.meta_info["distill_enabled"] = bool(enabled_for_step)
        batch.meta_info["distill_lambda"] = float(self.distill_settings.lambda_coef)
        batch.meta_info["distill_use_grpo_loss"] = bool(self.distill_settings.use_grpo_loss)
        batch.meta_info["distill_loss_variant"] = str(self.distill_settings.loss_variant)
        batch.meta_info["distill_alpha"] = float(self.distill_settings.alpha)
        batch.meta_info["distill_is_clip"] = (
            None if self.distill_settings.is_clip is None else float(self.distill_settings.is_clip)
        )
        batch.meta_info["distill_full_logit_topk"] = int(self.distill_settings.full_logit_topk)
        batch.meta_info["distill_full_logit_add_tail"] = bool(self.distill_settings.full_logit_add_tail)
        batch.meta_info["distill_teacher_regularization"] = str(self.distill_settings.teacher_regularization)

    def _attach_distill_payload_to_batch(
        self,
        batch: DataProto,
        payload: DistillPayload | None,
    ) -> dict[str, float]:
        """Attach distillation tensors for direct actor-side SDPO loss computation."""
        metrics = {
            "distill/sdpo_loss": 0.0,
            "distill/token_count": 0.0,
        }
        self._set_distill_meta(batch=batch, enabled_for_step=False)

        if not self.distill_settings.enable or payload is None:
            return metrics

        token_count = float(payload.distill_mask.sum().item())
        metrics["distill/token_count"] = token_count
        if token_count < float(self.distill_settings.min_distill_tokens):
            return metrics

        batch_size = int(batch.batch["responses"].shape[0])
        kept_indices = payload.kept_indices.long()
        if int(kept_indices.numel()) <= 0:
            return metrics

        den_batch = payload.denominator_batch.batch
        max_prompt_len = int(den_batch["prompts"].shape[1])
        max_distill_len = int(den_batch["responses"].shape[1])
        if max_prompt_len <= 0 or max_distill_len <= 0:
            return metrics

        pad_token_id = int(self.tokenizer.pad_token_id)
        teacher_prompts = torch.full((batch_size, max_prompt_len), fill_value=pad_token_id, dtype=torch.long)
        teacher_prompt_attention = torch.zeros((batch_size, max_prompt_len), dtype=torch.long)
        teacher_responses = torch.full((batch_size, max_distill_len), fill_value=pad_token_id, dtype=torch.long)
        teacher_response_mask = torch.zeros((batch_size, max_distill_len), dtype=torch.long)
        distill_mask = torch.zeros((batch_size, max_distill_len), dtype=torch.float32)

        denominator_prompt_attention = den_batch["attention_mask"][:, :max_prompt_len].long()
        for row_idx, kept_idx in enumerate(kept_indices.tolist()):
            if kept_idx < 0 or kept_idx >= batch_size:
                continue
            teacher_prompts[kept_idx] = den_batch["prompts"][row_idx].long()
            teacher_prompt_attention[kept_idx] = denominator_prompt_attention[row_idx]
            teacher_responses[kept_idx] = den_batch["responses"][row_idx].long()
            teacher_response_mask[kept_idx] = den_batch["response_mask"][row_idx].long()
            distill_mask[kept_idx] = payload.distill_mask[row_idx].float()

        teacher_attention_mask = torch.cat([teacher_prompt_attention, teacher_response_mask], dim=1)
        teacher_position_ids = (torch.cumsum(teacher_attention_mask, dim=1) - 1) * teacher_attention_mask
        teacher_input_ids = torch.cat([teacher_prompts, teacher_responses], dim=1)

        batch.batch["distill_teacher_input_ids"] = teacher_input_ids
        batch.batch["distill_teacher_attention_mask"] = teacher_attention_mask
        batch.batch["distill_teacher_position_ids"] = teacher_position_ids
        batch.batch["distill_teacher_responses"] = teacher_responses
        batch.batch["distill_mask"] = distill_mask
        self._set_distill_meta(batch=batch, enabled_for_step=True)
        return metrics

    def _initialize_teacher_snapshot_if_needed(self, timing_raw: dict[str, float]) -> float:
        """Initialize teacher snapshot from actor after checkpoint load."""
        if not self.distill_settings.enable:
            return 0.0
        if self.distill_settings.teacher_regularization == "none":
            return 0.0
        with marked_timer("distill_teacher_init_sync", timing_raw):
            self.actor_rollout_wg.sync_teacher_from_actor(mode="hard", update_rate=0.0)
        return 1.0

    def _maybe_sync_teacher_after_actor_update(self, timing_raw: dict[str, float]) -> float:
        """Apply configured teacher sync schedule after actor update."""
        if not self.distill_settings.enable:
            return 0.0
        mode = self.distill_settings.teacher_regularization
        if mode == "none":
            return 0.0
        if mode == "ema":
            with marked_timer("distill_teacher_sync", timing_raw):
                self.actor_rollout_wg.sync_teacher_from_actor(
                    mode="ema",
                    update_rate=self.distill_settings.teacher_update_rate,
                )
            return 1.0
        if mode == "every_n_steps":
            if self.global_steps % self.distill_settings.teacher_update_interval != 0:
                return 0.0
            with marked_timer("distill_teacher_sync", timing_raw):
                self.actor_rollout_wg.sync_teacher_from_actor(mode="hard", update_rate=0.0)
            return 1.0
        raise ValueError(f"Unsupported teacher_regularization mode: {mode}")

    def fit_agent(self):
        """Run PPO training with additional SDPO self-distillation actor updates."""
        from pprint import pprint
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        self._initialize_teacher_snapshot_if_needed(timing_raw={})

        import time

        start_time = time.time()
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate_agent()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        print(f"Time taken to validate agent: {time.time() - start_time}")
        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            pprint(f"epoch {epoch}, step {self.global_steps} started")
            for batch_dict in self.train_dataloader:
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                    dtype=object,
                )
                batch = batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )

                metrics: dict[str, float] = {}
                timing_raw: dict[str, float] = {}
                if self.distill_settings.enable:
                    metrics["distill/teacher_sync_applied"] = 0.0
                    metrics["distill/sdpo_loss"] = 0.0
                    metrics["distill/token_count"] = 0.0

                batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])

                with marked_timer("step", timing_raw):
                    self.init_envs_and_agents(batch)
                    distill_payload: DistillPayload | None = None
                    if self.config.rllm.stepwise_advantage.enable:
                        final_gen_batch_output = self.generate_agent_steps(
                            timing_raw=timing_raw,
                            meta_info=batch.meta_info,
                            uids=batch.non_tensor_batch["uid"],
                        )
                        repeat_counts = final_gen_batch_output.meta_info["repeat_counts"]
                        batch = batch.sample_level_repeat(repeat_counts)
                        final_gen_batch_output.meta_info.pop("repeat_counts", None)
                        batch = batch.union(final_gen_batch_output)
                        batch = self._pad_dataproto_to_world_size(batch=batch)
                    else:
                        final_gen_batch_output, generate_metrics = self.generate_agent_trajectory(
                            timing_raw=timing_raw,
                            meta_info=batch.meta_info,
                        )
                        batch = batch.union(final_gen_batch_output)
                        metrics.update(generate_metrics)

                    if self.use_critic:
                        with marked_timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if "token_level_scores" not in batch.batch:
                            reward_tensor = self.reward_fn(batch)
                            batch.batch["token_level_scores"] = reward_tensor
                        else:
                            reward_tensor = batch.batch["token_level_scores"]

                        uids = batch.non_tensor_batch["uid"]
                        unique_uids = np.unique(uids)
                        valid_mask = torch.ones(len(uids), dtype=torch.bool)
                        solve_none = 0
                        solve_all = 0
                        for uid in unique_uids:
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(-1)
                            if (uid_rewards <= 0).all():
                                valid_mask[uid_mask] = False
                                solve_none += 1
                            elif (uid_rewards >= 1).all():
                                valid_mask[uid_mask] = False
                                solve_all += 1
                        metrics["batch/solve_none"] = float(solve_none)
                        metrics["batch/solve_all"] = float(solve_all)
                        metrics["batch/solve_partial"] = float(len(unique_uids) - solve_none - solve_all)

                        if self.config.rllm.rejection_sample.enable:
                            if not valid_mask.any():
                                continue
                            batch = batch[valid_mask]
                            num_trainer_replicas = self.actor_rollout_wg.world_size
                            max_batch_size = (
                                batch.batch["input_ids"].shape[0] // num_trainer_replicas
                            ) * num_trainer_replicas
                            if not max_batch_size:
                                continue
                            size_mask = torch.zeros(batch.batch["input_ids"].shape[0], dtype=torch.bool)
                            size_mask[:max_batch_size] = True
                            batch = batch[size_mask]

                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=loss_agg_mode,
                            )
                            metrics["actor/entropy"] = float(entropy_agg.detach().item())
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)

                        if self.use_reference_policy:
                            with marked_timer("ref", timing_raw):
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=self.config.algorithm.norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    if self.config.rllm.mask_truncated_samples:
                        mask = batch.batch["attention_mask"][:, -1] == 1
                        batch = batch[~mask]

                    if self.distill_settings.enable:
                        self._set_distill_meta(batch=batch, enabled_for_step=False)

                    if self.distill_settings.enable and not self.config.rllm.stepwise_advantage.enable:
                        distill_payload, distill_prep_metrics = self._prepare_distill_payload(batch)
                        metrics.update(distill_prep_metrics)
                        distill_metrics = self._attach_distill_payload_to_batch(
                            batch=batch,
                            payload=distill_payload,
                        )
                        metrics.update(distill_metrics)

                    batch = self._pad_dataproto_to_world_size(batch=batch)
                    self._balance_batch(batch, metrics=metrics)
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update({k: float(v) for k, v in critic_output_metrics.items()})

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update({k: float(v) for k, v in actor_output_metrics.items()})
                        teacher_sync_applied = self._maybe_sync_teacher_after_actor_update(timing_raw=timing_raw)
                        if self.distill_settings.enable:
                            metrics["distill/teacher_sync_applied"] = float(teacher_sync_applied)

                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and self.global_steps % self.config.trainer.test_freq == 0
                    ):
                        with marked_timer("testing", timing_raw):
                            val_metrics: dict[str, float] = self._validate_agent()
                        metrics.update(val_metrics)

                    if (
                        self.config.trainer.save_freq > 0
                        and self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                logger.log(data=metrics, step=self.global_steps)
                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate_agent()
                        pprint(f"Final validation metrics: {val_metrics}")
                        logger.log(data=val_metrics, step=self.global_steps)
                    return
