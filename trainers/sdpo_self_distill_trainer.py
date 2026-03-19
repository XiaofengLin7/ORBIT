"""Joint SDPO self-distillation trainer for multi-episode PPO.

This module implements shared-weight self-distillation where each training step
performs:
1. Standard teacher RL update on full multi-episode rollouts.
2. SDPO distillation loss, optionally combined with PPO, in the same actor update.

Distillation uses a strict context-overflow guard on denominator scoring length:
``L_den = len(denominator_prompt_tokens) + len(first_attempt_sequence_tokens)``,
where ``denominator_prompt_tokens = concat(c_N, prompts_wo_system)`` when
system tokens can be reliably stripped from the student prompt.
If ``L_den > context_limit``, the sample is excluded from distillation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from rllm.agents.utils import convert_messages_to_tokens_and_masks
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
    is_clip: bool = False
    full_logit_topk: int = 64
    full_logit_add_tail: bool = True
    negate_sdpo_loss: bool = False
    use_stale_coefficient: bool = False
    strip_system_from_teacher_prompt: bool = True
    teacher_context_attempts: int | None = None
    trajectory_selection: str = "first_attempt_hindsight"
    teacher_regularization: str = "none"
    teacher_update_rate: float = 0.05
    teacher_update_interval: int = 10
    merge_to_advantages: bool = False


@dataclass
class DistillPayload:
    """Prepared distillation payload for one optimization step."""

    student_batch: DataProto
    teacher_batch: DataProto
    distill_mask: torch.Tensor
    kept_indices: torch.Tensor
    skipped_context_overflow: int
    skipped_hindsight_unavailable: int
    skipped_selective_gate: int
    skipped_attempt_incomplete: int
    total_samples: int
    kept_samples: int
    student_old_log_probs: torch.Tensor | None = None
    teacher_old_log_probs: torch.Tensor | None = None


@dataclass(frozen=True)
class AttemptSummary:
    """Summary of one completed attempt."""

    episode_index: int
    context_tokens: torch.Tensor
    success: bool


@dataclass(frozen=True)
class AttemptSequence:
    """Response-sequence view for a selected attempt."""

    response_tokens: torch.Tensor
    response_mask: torch.Tensor
    distill_mask: torch.Tensor


def _resolve_completed_attempt_success(
    step_records: Sequence[dict[str, Any]],
    *,
    start_idx: int,
    end_idx: int,
    episode_index: int,
    fallback_step: dict[str, Any],
) -> bool:
    """Resolve a completed attempt's success from its terminal step.

    Reflection-enabled rollouts can insert extra post-terminal steps after an
    episode has already finished. Those steps may not carry the original
    terminal success flag, so we search within the completed episode slice for
    the first ``episode_done=True`` record and use its success flag. If no such
    terminal step is available, fall back to the boundary step's success field.
    """
    for step_idx in range(start_idx, end_idx + 1):
        step = step_records[step_idx]
        raw_episode_index = step.get("episode_index", episode_index)
        try:
            step_episode_index = int(raw_episode_index)
        except (TypeError, ValueError):
            continue
        if step_episode_index != int(episode_index):
            continue
        if bool(step.get("episode_done", False)):
            return bool(step.get("episode_success", step.get("success", False)))
    return bool(fallback_step.get("episode_success", fallback_step.get("success", False)))


def collect_complete_attempt_summaries(
    step_records: Sequence[dict[str, Any]],
) -> list[AttemptSummary] | None:
    """Collect complete attempts and their serialized context tokens.

    Args:
        step_records: Step-level token metadata from the rollout engine.

    Returns:
        Completed-attempt summaries in episode order, or ``None`` when the
        trajectory metadata is malformed or non-cumulative.
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
    complete_attempts: list[AttemptSummary] = []
    attempt_start_idx = 0

    for step_idx, step in enumerate(step_records[1:], start=1):
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
            attempt_success = _resolve_completed_attempt_success(
                step_records,
                start_idx=attempt_start_idx,
                end_idx=step_idx - 1,
                episode_index=prev_episode_index,
                fallback_step=prev_step,
            )
            complete_attempts.append(
                AttemptSummary(
                    episode_index=prev_episode_index,
                    context_tokens=torch.as_tensor(
                        list(accumulated) + list(delta_prompt_tokens[:terminal_len]),
                        dtype=torch.long,
                    ),
                    success=attempt_success,
                )
            )
            attempt_start_idx = step_idx

        accumulated = current_prompt_ids + list(current_completion_ids)
        prev_episode_index = current_episode_index
        prev_step = step

    if bool(step_records[-1].get("episode_done", False)):
        attempt_success = _resolve_completed_attempt_success(
            step_records,
            start_idx=attempt_start_idx,
            end_idx=len(step_records) - 1,
            episode_index=prev_episode_index,
            fallback_step=step_records[-1],
        )
        complete_attempts.append(
            AttemptSummary(
                episode_index=prev_episode_index,
                context_tokens=torch.as_tensor(list(accumulated), dtype=torch.long),
                success=attempt_success,
            )
        )

    return complete_attempts or None


def get_attempt_summary(
    attempts: Sequence[AttemptSummary],
    *,
    episode_index: int,
) -> AttemptSummary | None:
    """Return the summary for one completed episode index."""
    for attempt in attempts:
        if int(attempt.episode_index) == int(episode_index):
            return attempt
    return None


def extract_attempt_response_sequence(
    step_records: Sequence[dict[str, Any]],
    *,
    target_episode_index: int,
) -> AttemptSequence | None:
    """Extract one attempt's response sequence from cumulative step records.

    Args:
        step_records: Step-level token metadata from the rollout engine.
        target_episode_index: Episode index to extract.

    Returns:
        Attempt-local response tokens and masks, or ``None`` when the records
        are malformed or the selected attempt has no valid model tokens.
    """
    if not step_records:
        return None

    first_prompt_ids = step_records[0].get("prompt_ids", [])
    first_completion_ids = step_records[0].get("completion_ids", [])
    if not isinstance(first_prompt_ids, list) or not isinstance(first_completion_ids, list):
        return None

    first_episode_index_raw = step_records[0].get("episode_index", 0)
    try:
        prev_episode_index = int(first_episode_index_raw)
    except (TypeError, ValueError):
        return None

    accumulated = list(first_prompt_ids)
    response_tokens: list[int] = []
    response_mask: list[int] = []

    if prev_episode_index == target_episode_index:
        response_tokens.extend(list(first_completion_ids))
        response_mask.extend([1] * len(first_completion_ids))
    accumulated.extend(list(first_completion_ids))

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

        delta_prompt_tokens = current_prompt_ids[len(accumulated) :]
        if current_episode_index == target_episode_index:
            if prev_episode_index == target_episode_index:
                response_tokens.extend(list(delta_prompt_tokens))
                response_mask.extend([0] * len(delta_prompt_tokens))
            response_tokens.extend(list(current_completion_ids))
            response_mask.extend([1] * len(current_completion_ids))

        accumulated = current_prompt_ids + list(current_completion_ids)
        prev_episode_index = current_episode_index

    response_mask_tensor = torch.as_tensor(response_mask, dtype=torch.long)
    if int(response_mask_tensor.sum().item()) <= 0:
        return None

    return AttemptSequence(
        response_tokens=torch.as_tensor(response_tokens, dtype=torch.long),
        response_mask=response_mask_tensor,
        distill_mask=response_mask_tensor.float(),
    )


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
    complete_attempts = collect_complete_attempt_summaries(step_records)
    if not complete_attempts:
        return None

    if teacher_context_attempts is None:
        selected_context = complete_attempts[-1].context_tokens
    else:
        if len(complete_attempts) < teacher_context_attempts:
            return None
        selected_context = complete_attempts[teacher_context_attempts - 1].context_tokens

    return selected_context.long()


def build_hindsight_prompt_tokens_latest_successful_complete_attempt(
    step_records: Sequence[dict[str, Any]],
) -> torch.Tensor | None:
    """Construct hindsight context from only the latest successful attempt.

    Args:
        step_records: Step-level records from rollout engine.

    Returns:
        Tensor prompt tokens for the most recent successful completed attempt,
        excluding tokens from earlier attempts. Returns ``None`` when records
        are malformed or no completed successful attempt exists.
    """
    complete_attempts = collect_complete_attempt_summaries(step_records)
    if not complete_attempts:
        return None

    selected_idx: int | None = None
    for idx in range(len(complete_attempts) - 1, -1, -1):
        if bool(complete_attempts[idx].success):
            selected_idx = idx
            break
    if selected_idx is None:
        return None

    selected_context = complete_attempts[selected_idx].context_tokens.long()
    if selected_idx == 0:
        return selected_context

    previous_context = complete_attempts[selected_idx - 1].context_tokens.long()
    previous_len = int(previous_context.numel())
    if previous_len <= 0:
        return selected_context
    if int(selected_context.numel()) <= previous_len:
        return None
    if not torch.equal(selected_context[:previous_len], previous_context):
        return None

    isolated_context = selected_context[previous_len:]
    if int(isolated_context.numel()) <= 0:
        return None
    return isolated_context


def extract_initial_messages_before_first_assistant(
    chat_completions: Any,
) -> list[dict[str, str]] | None:
    """Extract initial chat messages up to the first assistant turn.

    Args:
        chat_completions: Full chat history from token trajectory metadata.

    Returns:
        Ordered messages before the first assistant turn, or ``None`` if the
        input structure is invalid.
    """
    if not isinstance(chat_completions, list):
        return None

    initial_messages: list[dict[str, str]] = []
    for message in chat_completions:
        if not isinstance(message, dict):
            return None
        role_raw = message.get("role")
        if not isinstance(role_raw, str):
            return None
        if role_raw == "assistant":
            break
        content_raw = message.get("content", "")
        content = content_raw if isinstance(content_raw, str) else str(content_raw)
        initial_messages.append({"role": role_raw, "content": content})
    return initial_messages


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

    def _get_distill_chat_parser(self) -> Any | None:
        """Get and cache a chat parser used for system-role stripping.

        Returns:
            Parser object compatible with ``parse(messages, add_generation_prompt=...)``,
            or ``None`` when unavailable.
        """
        cached_parser = getattr(self, "_distill_chat_parser", None)
        if cached_parser is not None:
            return cached_parser
        if bool(getattr(self, "_distill_chat_parser_unavailable", False)):
            return None

        engine = getattr(self, "agent_execution_engine", None)
        engine_parser = getattr(engine, "chat_parser", None)
        if engine_parser is not None:
            self._distill_chat_parser = engine_parser
            return engine_parser

        try:
            from rllm.parser.chat_template_parser import ChatTemplateParser

            rllm_cfg = getattr(self.config, "rllm", None)
            disable_thinking = bool(getattr(rllm_cfg, "disable_thinking", False))
            parser = ChatTemplateParser.get_parser(
                self.tokenizer,
                disable_thinking=disable_thinking,
            )
        except Exception:
            self._distill_chat_parser_unavailable = True
            return None

        self._distill_chat_parser = parser
        return parser

    def _split_student_prompt_system_prefix(
        self,
        *,
        trajectory: dict[str, Any],
        student_prompt_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Split student prompt into exact system-prefix and remaining suffix.

        Args:
            trajectory: Token trajectory payload for one sample.
            student_prompt_tokens: Original prompt tokens used by student.

        Returns:
            Tuple ``(system_prefix_tokens, stripped_prompt_tokens)`` when the
            leading system-token prefix can be reproduced reliably from rollout
            metadata. Returns ``None`` when reconstruction is unavailable or
            unreliable; callers should skip distillation for that sample.
        """
        initial_messages = extract_initial_messages_before_first_assistant(
            trajectory.get("chat_completions")
        )
        if not initial_messages or initial_messages[0].get("role") != "system":
            return None

        parser = self._get_distill_chat_parser()
        if parser is None:
            return None

        leading_system_messages: list[dict[str, str]] = []
        for message in initial_messages:
            if message.get("role") != "system":
                break
            leading_system_messages.append(message)
        if not leading_system_messages:
            return None

        try:
            system_prefix_ids, _ = convert_messages_to_tokens_and_masks(
                leading_system_messages,
                tokenizer=self.tokenizer,
                parser=parser,
                contains_first_msg=True,
                contains_generation_msg=False,
            )
        except Exception:
            return None

        if not isinstance(system_prefix_ids, list):
            return None

        system_prefix_tokens = torch.as_tensor(system_prefix_ids, dtype=torch.long)
        system_prefix_len = int(system_prefix_tokens.numel())
        student_prompt_len = int(student_prompt_tokens.numel())
        if system_prefix_len <= 0 or system_prefix_len >= student_prompt_len:
            return None
        if not torch.equal(student_prompt_tokens[:system_prefix_len], system_prefix_tokens):
            return None

        stripped_prompt_tokens = student_prompt_tokens[system_prefix_len:]
        if int(stripped_prompt_tokens.numel()) <= 0:
            return None
        return system_prefix_tokens, stripped_prompt_tokens

    def _build_student_prompt_without_system(
        self,
        *,
        trajectory: dict[str, Any],
        student_prompt_tokens: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return student prompt tokens with exact leading system-prefix removed.

        Args:
            trajectory: Token trajectory payload for one sample.
            student_prompt_tokens: Original prompt tokens used by student.

        Returns:
            Prompt tokens with the exact leading system-token prefix removed
            when that prefix can be reproduced reliably from rollout metadata.
            Returns ``None`` when stripping is unavailable or unreliable;
            callers should skip distillation for that sample.
        """
        prompt_parts = self._split_student_prompt_system_prefix(
            trajectory=trajectory,
            student_prompt_tokens=student_prompt_tokens,
        )
        if prompt_parts is None:
            return None
        _, stripped_prompt_tokens = prompt_parts
        return stripped_prompt_tokens

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
        trajectory_selection = str(
            cfg.get("trajectory_selection", "first_attempt_hindsight")
        ).lower()
        valid_trajectory_selection = {
            "first_attempt_hindsight",
            "first_attempt_latest_success_hindsight",
            "first_attempt_latest_success_hindsight_first_failure_only",
            "selective_retry_success_n2",
        }
        if trajectory_selection not in valid_trajectory_selection:
            raise ValueError(
                "rllm.distill.trajectory_selection must be one of "
                f"{sorted(valid_trajectory_selection)}, got {trajectory_selection}"
            )
        if (
            trajectory_selection == "selective_retry_success_n2"
            and teacher_context_attempts != 1
        ):
            raise ValueError(
                "rllm.distill.teacher_context_attempts must be 1 when "
                "rllm.distill.trajectory_selection='selective_retry_success_n2'."
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
        is_clip_raw = cfg.get("is_clip", False)
        if is_clip_raw is None:
            is_clip = False
        elif isinstance(is_clip_raw, bool):
            is_clip = is_clip_raw
        elif isinstance(is_clip_raw, str):
            lowered = is_clip_raw.strip().lower()
            if lowered in {"", "0", "false", "none", "null"}:
                is_clip = False
            elif lowered in {"1", "true"}:
                is_clip = True
            else:
                raise ValueError(
                    "rllm.distill.is_clip must be a boolean or boolean-like string, "
                    f"got {is_clip_raw!r}"
                )
        else:
            raise ValueError(
                "rllm.distill.is_clip must be a boolean or boolean-like string, "
                f"got value {is_clip_raw!r} of type {type(is_clip_raw).__name__}"
            )
        full_logit_topk = int(cfg.get("full_logit_topk", 64))
        if full_logit_topk < 0:
            raise ValueError(
                f"rllm.distill.full_logit_topk must be a non-negative integer, got {full_logit_topk}"
            )
        merge_to_advantages = bool(cfg.get("merge_to_advantages", False))
        use_stale_coefficient = bool(cfg.get("use_stale_coefficient", False))
        if merge_to_advantages and loss_variant != "non_full":
            raise ValueError(
                "rllm.distill.merge_to_advantages requires "
                "rllm.distill.loss_variant='non_full'."
            )
        if use_stale_coefficient and loss_variant != "non_full":
            raise ValueError(
                "rllm.distill.use_stale_coefficient requires "
                "rllm.distill.loss_variant='non_full'."
            )
        if is_clip and loss_variant == "full_logit":
            raise ValueError(
                "rllm.distill.is_clip is not supported when "
                "rllm.distill.loss_variant='full_logit'."
            )
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
            negate_sdpo_loss=bool(cfg.get("negate_sdpo_loss", False)),
            use_stale_coefficient=use_stale_coefficient,
            strip_system_from_teacher_prompt=bool(cfg.get("strip_system_from_teacher_prompt", True)),
            teacher_context_attempts=teacher_context_attempts,
            trajectory_selection=trajectory_selection,
            teacher_regularization=teacher_regularization,
            teacher_update_rate=teacher_update_rate,
            teacher_update_interval=teacher_update_interval,
            merge_to_advantages=merge_to_advantages,
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

    def _empty_distill_prep_metrics(self) -> dict[str, float]:
        """Return zeroed distillation-preparation metrics."""
        return {
            "distill/skipped_context_overflow": 0.0,
            "distill/skipped_hindsight_unavailable": 0.0,
            "distill/skipped_hindsight_invalid_traj_idx": 0.0,
            "distill/skipped_hindsight_trajectory_not_dict": 0.0,
            "distill/skipped_hindsight_step_records_not_list": 0.0,
            "distill/skipped_hindsight_complete_attempts_unavailable": 0.0,
            "distill/skipped_hindsight_missing_first_attempt_mask": 0.0,
            "distill/skipped_hindsight_selected_attempt_missing": 0.0,
            "distill/skipped_hindsight_teacher_context_unavailable": 0.0,
            "distill/skipped_hindsight_system_strip_unavailable": 0.0,
            "distill/skipped_selective_gate": 0.0,
            "distill/skipped_attempt_incomplete": 0.0,
            "distill/kept_ratio": 0.0,
        }

    def _build_distill_io_batch(
        self,
        *,
        base_batch: DataProto,
        prompt_rows: list[torch.Tensor],
        response_rows: list[torch.Tensor],
        response_mask_rows: list[torch.Tensor],
    ) -> DataProto:
        """Build a compact prompt/response batch for distillation scoring.

        Args:
            base_batch: Selected batch container to overwrite with compact
                distillation tensors.
            prompt_rows: Prompt-token rows for each kept sample.
            response_rows: Response-token rows for each kept sample.
            response_mask_rows: Response masks aligned to ``response_rows``.

        Returns:
            ``base_batch`` with prompt/response tensors replaced by padded
            distillation inputs.
        """
        pad_token_id = int(self.tokenizer.pad_token_id)
        max_prompt_len = max(int(row.numel()) for row in prompt_rows)
        max_response_len = max(int(row.numel()) for row in response_rows)

        prompts = torch.full(
            (len(prompt_rows), max_prompt_len),
            fill_value=pad_token_id,
            dtype=torch.long,
        )
        responses = torch.full(
            (len(prompt_rows), max_response_len),
            fill_value=pad_token_id,
            dtype=torch.long,
        )
        response_mask = torch.zeros(
            (len(prompt_rows), max_response_len),
            dtype=torch.long,
        )

        for row_idx, prompt_row in enumerate(prompt_rows):
            prompt_len = int(prompt_row.numel())
            prompts[row_idx, -prompt_len:] = prompt_row.long()

        for row_idx, response_row in enumerate(response_rows):
            response_len = int(response_row.numel())
            responses[row_idx, :response_len] = response_row.long()
            response_mask[row_idx, :response_len] = response_mask_rows[row_idx].long()

        prompt_lens = torch.as_tensor([int(row.numel()) for row in prompt_rows], dtype=torch.long)
        prompt_pos = torch.arange(max_prompt_len).unsqueeze(0)
        prompt_attention = (prompt_pos >= (max_prompt_len - prompt_lens.unsqueeze(1))).long()
        attention_mask = torch.cat([prompt_attention, response_mask], dim=1)
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask
        input_ids = torch.cat([prompts, responses], dim=1)

        base_batch.batch["prompts"] = prompts
        base_batch.batch["responses"] = responses
        base_batch.batch["response_mask"] = response_mask
        base_batch.batch["input_ids"] = input_ids
        base_batch.batch["attention_mask"] = attention_mask
        base_batch.batch["position_ids"] = position_ids
        return base_batch

    def _prepare_distill_payload(self, batch: DataProto) -> tuple[DistillPayload | None, dict[str, float]]:
        """Prepare denominator batch and token masks for merged distillation."""
        if not self.distill_settings.enable:
            return None, {}
        if not self._latest_token_trajectories:
            return None, self._empty_distill_prep_metrics()

        response_mask = batch.batch["response_mask"].float()
        trajectory_selection = self.distill_settings.trajectory_selection
        latest_success_hindsight_modes = {
            "first_attempt_latest_success_hindsight",
            "first_attempt_latest_success_hindsight_first_failure_only",
        }
        uses_first_attempt_target = trajectory_selection in {
            "first_attempt_hindsight",
            "first_attempt_latest_success_hindsight",
            "first_attempt_latest_success_hindsight_first_failure_only",
        }
        first_attempt_mask: torch.Tensor | None = None
        first_attempt_distill_mask: torch.Tensor | None = None
        if uses_first_attempt_target:
            first_attempt_mask = batch.batch.get("first_attempt_response_mask")
            if first_attempt_mask is None:
                return None, self._empty_distill_prep_metrics()
            first_attempt_mask = first_attempt_mask.float()
            first_attempt_distill_mask = response_mask * first_attempt_mask

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
        student_prompt_tokens_per_sample: list[torch.Tensor] = []
        teacher_prompt_tokens_per_sample: list[torch.Tensor] = []
        student_response_tokens_per_sample: list[torch.Tensor] = []
        student_response_masks_per_sample: list[torch.Tensor] = []
        student_distill_masks_per_sample: list[torch.Tensor] = []
        skipped_context_overflow = 0
        skipped_hindsight_unavailable = 0
        skipped_hindsight_invalid_traj_idx = 0
        skipped_hindsight_trajectory_not_dict = 0
        skipped_hindsight_step_records_not_list = 0
        skipped_hindsight_complete_attempts_unavailable = 0
        skipped_hindsight_missing_first_attempt_mask = 0
        skipped_hindsight_selected_attempt_missing = 0
        skipped_hindsight_teacher_context_unavailable = 0
        skipped_hindsight_system_strip_unavailable = 0
        skipped_selective_gate = 0
        skipped_attempt_incomplete = 0

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
            if (
                uses_first_attempt_target
                and first_attempt_distill_mask is not None
                and float(first_attempt_distill_mask[i].sum().item()) <= 0
            ):
                continue

            traj_idx = int(traj_indices[i].item())
            if traj_idx < 0 or traj_idx >= len(self._latest_token_trajectories):
                skipped_hindsight_unavailable += 1
                skipped_hindsight_invalid_traj_idx += 1
                continue
            trajectory = self._latest_token_trajectories[traj_idx]
            if not isinstance(trajectory, dict):
                skipped_hindsight_unavailable += 1
                skipped_hindsight_trajectory_not_dict += 1
                continue

            step_records = trajectory.get("step_records", [])
            if not isinstance(step_records, list):
                skipped_hindsight_unavailable += 1
                skipped_hindsight_step_records_not_list += 1
                continue

            selected_attempt: AttemptSequence | None = None
            teacher_context_tokens: torch.Tensor | None = None
            if trajectory_selection == "selective_retry_success_n2":
                complete_attempts = collect_complete_attempt_summaries(step_records)
                if complete_attempts is None:
                    skipped_hindsight_unavailable += 1
                    skipped_hindsight_complete_attempts_unavailable += 1
                    continue
                first_attempt_summary = get_attempt_summary(
                    complete_attempts,
                    episode_index=0,
                )
                second_attempt_summary = get_attempt_summary(
                    complete_attempts,
                    episode_index=1,
                )
                if first_attempt_summary is None or second_attempt_summary is None:
                    skipped_attempt_incomplete += 1
                    continue
                if first_attempt_summary.success or not second_attempt_summary.success:
                    skipped_selective_gate += 1
                    continue
                selected_attempt = extract_attempt_response_sequence(
                    step_records,
                    target_episode_index=1,
                )
                teacher_context_tokens = first_attempt_summary.context_tokens.long()
            else:
                if first_attempt_mask is None:
                    skipped_hindsight_unavailable += 1
                    skipped_hindsight_missing_first_attempt_mask += 1
                    continue
                if trajectory_selection == "first_attempt_latest_success_hindsight_first_failure_only":
                    complete_attempts = collect_complete_attempt_summaries(step_records)
                    if complete_attempts is None:
                        skipped_hindsight_unavailable += 1
                        skipped_hindsight_complete_attempts_unavailable += 1
                        continue
                    first_attempt_summary = get_attempt_summary(
                        complete_attempts,
                        episode_index=0,
                    )
                    if first_attempt_summary is None:
                        skipped_attempt_incomplete += 1
                        continue
                    if first_attempt_summary.success:
                        skipped_selective_gate += 1
                        continue
                    has_later_success = any(
                        int(attempt.episode_index) > 0 and bool(attempt.success)
                        for attempt in complete_attempts
                    )
                    if not has_later_success:
                        skipped_selective_gate += 1
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
                selected_attempt = AttemptSequence(
                    response_tokens=prefix_response_tokens,
                    response_mask=prefix_response_mask,
                    distill_mask=prefix_distill_mask,
                )
                if trajectory_selection in latest_success_hindsight_modes:
                    teacher_context_tokens = (
                        build_hindsight_prompt_tokens_latest_successful_complete_attempt(
                            step_records=step_records,
                        )
                    )
                else:
                    teacher_context_tokens = build_hindsight_prompt_tokens_first_n_complete_attempts(
                        step_records=step_records,
                        teacher_context_attempts=self.distill_settings.teacher_context_attempts,
                    )
            if selected_attempt is None:
                skipped_hindsight_unavailable += 1
                skipped_hindsight_selected_attempt_missing += 1
                continue
            if teacher_context_tokens is None:
                skipped_hindsight_unavailable += 1
                skipped_hindsight_teacher_context_unavailable += 1
                continue

            prompt_attention_row = batch.batch["attention_mask"][i, :-original_response_len]
            student_prompt_tokens = batch.batch["prompts"][i][prompt_attention_row > 0].long()
            teacher_system_prefix: torch.Tensor | None = None
            if (
                trajectory_selection in latest_success_hindsight_modes
                or self.distill_settings.strip_system_from_teacher_prompt
            ):
                prompt_parts = self._split_student_prompt_system_prefix(
                    trajectory=trajectory,
                    student_prompt_tokens=student_prompt_tokens,
                )
                if prompt_parts is None:
                    skipped_hindsight_unavailable += 1
                    skipped_hindsight_system_strip_unavailable += 1
                    continue
                teacher_system_prefix, teacher_suffix = prompt_parts
            else:
                teacher_suffix = student_prompt_tokens
            if trajectory_selection in latest_success_hindsight_modes:
                assert teacher_system_prefix is not None
                teacher_prompt_tokens = torch.cat(
                    [
                        teacher_system_prefix.long(),
                        teacher_context_tokens.long(),
                        teacher_suffix.long(),
                    ],
                    dim=0,
                )
            else:
                teacher_prompt_tokens = torch.cat(
                    [teacher_context_tokens.long(), teacher_suffix.long()],
                    dim=0,
                )

            denominator_prompt_len = int(teacher_prompt_tokens.numel())
            distill_response_len = int(selected_attempt.response_tokens.numel())

            if should_skip_denominator_overflow(
                denominator_prompt_len=denominator_prompt_len,
                first_attempt_sequence_len=distill_response_len,
                context_limit=limit,
            ):
                skipped_context_overflow += 1
                continue

            kept_indices.append(i)
            student_prompt_tokens_per_sample.append(student_prompt_tokens)
            teacher_prompt_tokens_per_sample.append(teacher_prompt_tokens)
            student_response_tokens_per_sample.append(selected_attempt.response_tokens)
            student_response_masks_per_sample.append(selected_attempt.response_mask)
            student_distill_masks_per_sample.append(selected_attempt.distill_mask)

        hindsight_skip_metrics = {
            "distill/skipped_hindsight_invalid_traj_idx": float(skipped_hindsight_invalid_traj_idx),
            "distill/skipped_hindsight_trajectory_not_dict": float(skipped_hindsight_trajectory_not_dict),
            "distill/skipped_hindsight_step_records_not_list": float(skipped_hindsight_step_records_not_list),
            "distill/skipped_hindsight_complete_attempts_unavailable": float(
                skipped_hindsight_complete_attempts_unavailable
            ),
            "distill/skipped_hindsight_missing_first_attempt_mask": float(
                skipped_hindsight_missing_first_attempt_mask
            ),
            "distill/skipped_hindsight_selected_attempt_missing": float(
                skipped_hindsight_selected_attempt_missing
            ),
            "distill/skipped_hindsight_teacher_context_unavailable": float(
                skipped_hindsight_teacher_context_unavailable
            ),
            "distill/skipped_hindsight_system_strip_unavailable": float(
                skipped_hindsight_system_strip_unavailable
            ),
        }
        if not kept_indices:
            return None, {
                "distill/skipped_context_overflow": float(skipped_context_overflow),
                "distill/skipped_hindsight_unavailable": float(skipped_hindsight_unavailable),
                "distill/skipped_selective_gate": float(skipped_selective_gate),
                "distill/skipped_attempt_incomplete": float(skipped_attempt_incomplete),
                "distill/kept_ratio": 0.0,
                **hindsight_skip_metrics,
            }

        selected_student = batch.select_idxs(np.array(kept_indices))
        selected_teacher = batch.select_idxs(np.array(kept_indices))
        selected_student = self._build_distill_io_batch(
            base_batch=selected_student,
            prompt_rows=student_prompt_tokens_per_sample,
            response_rows=student_response_tokens_per_sample,
            response_mask_rows=student_response_masks_per_sample,
        )
        selected_teacher = self._build_distill_io_batch(
            base_batch=selected_teacher,
            prompt_rows=teacher_prompt_tokens_per_sample,
            response_rows=student_response_tokens_per_sample,
            response_mask_rows=student_response_masks_per_sample,
        )

        max_response_len = max(int(row.numel()) for row in student_response_tokens_per_sample)
        selected_distill_mask = torch.zeros(
            (len(student_distill_masks_per_sample), max_response_len),
            dtype=torch.float32,
        )
        for row_idx, row in enumerate(student_distill_masks_per_sample):
            row_len = int(row.numel())
            selected_distill_mask[row_idx, :row_len] = row.float()

        kept_samples = int(selected_student.batch["responses"].shape[0])
        need_student_old = (
            bool(self.distill_settings.is_clip)
            or self.distill_settings.use_stale_coefficient
        )
        need_teacher_old = self.distill_settings.use_stale_coefficient
        student_old_log_probs: torch.Tensor | None = None
        teacher_old_log_probs: torch.Tensor | None = None
        if need_student_old or need_teacher_old:
            if not hasattr(self, "actor_rollout_wg"):
                raise ValueError(
                    "Distillation with is_clip or use_stale_coefficient requires "
                    "actor_rollout_wg to compute old log-probs."
                )
        if need_student_old:
            student_old_log_probs = self._compute_distill_old_log_probs(
                selected_student,
                kept_samples=kept_samples,
                use_teacher=False,
            )
        if need_teacher_old:
            teacher_old_log_probs = self._compute_distill_old_log_probs(
                selected_teacher,
                kept_samples=kept_samples,
                use_teacher=(self.distill_settings.teacher_regularization != "none"),
            )

        payload = DistillPayload(
            student_batch=selected_student,
            teacher_batch=selected_teacher,
            distill_mask=selected_distill_mask,
            kept_indices=torch.as_tensor(kept_indices, dtype=torch.long),
            skipped_context_overflow=skipped_context_overflow,
            skipped_hindsight_unavailable=skipped_hindsight_unavailable,
            skipped_selective_gate=skipped_selective_gate,
            skipped_attempt_incomplete=skipped_attempt_incomplete,
            total_samples=total_samples,
            kept_samples=kept_samples,
            student_old_log_probs=student_old_log_probs,
            teacher_old_log_probs=teacher_old_log_probs,
        )
        metrics = {
            "distill/skipped_context_overflow": float(skipped_context_overflow),
            "distill/skipped_hindsight_unavailable": float(skipped_hindsight_unavailable),
            "distill/skipped_selective_gate": float(skipped_selective_gate),
            "distill/skipped_attempt_incomplete": float(skipped_attempt_incomplete),
            "distill/kept_ratio": float(kept_samples / max(1, total_samples)),
            **hindsight_skip_metrics,
        }
        return payload, metrics

    def _compute_distill_old_log_probs(
        self,
        batch: DataProto,
        *,
        kept_samples: int,
        use_teacher: bool = False,
    ) -> torch.Tensor:
        """Compute old log-probs on a world-size-safe batch.

        Args:
            batch: Student or teacher distillation batch.
            kept_samples: Number of real (non-pad) samples.
            use_teacher: If True, score with the teacher (ref) module
                instead of the actor.
        """
        world_size = int(getattr(self.actor_rollout_wg, "world_size", 1))
        compute_batch = batch
        if world_size > 1 and (kept_samples % world_size) != 0:
            compute_batch = self._pad_dataproto_to_world_size(batch=batch)
        if use_teacher:
            out = self.actor_rollout_wg.compute_teacher_log_prob(compute_batch)
        else:
            out = self.actor_rollout_wg.compute_log_prob(compute_batch)
        key = "ref_log_prob" if "ref_log_prob" in out.batch else "old_log_probs"
        return out.batch[key][:kept_samples].float()

    def _compute_distill_bonus(
        self,
        batch: DataProto,
        payload: DistillPayload | None,
        timing_raw: dict[str, float],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute SDPO-style distillation bonus to merge into actor advantages.

        This reproduces the March 2nd architecture: the bonus is added to
        PPO advantages so that ``agg_loss(combined_advantage, response_mask)``
        naturally applies per-sequence F/T dilution.
        """
        bonus = torch.zeros_like(batch.batch["advantages"])
        zero_stats: dict[str, float] = {
            "distill/sdpo_loss": 0.0,
            "distill/log_ratio_mean": 0.0,
            "distill/log_ratio_std": 0.0,
            "distill/token_count": 0.0,
            "distill/skipped_batches": 1.0,
        }
        if not self.distill_settings.enable or payload is None:
            return bonus, zero_stats

        token_count = int(payload.distill_mask.sum().item())
        if token_count < self.distill_settings.min_distill_tokens:
            zero_stats["distill/token_count"] = float(token_count)
            return bonus, zero_stats

        # Denominator (teacher) log-probs — one forward pass.
        use_teacher = self.distill_settings.teacher_regularization != "none"
        with marked_timer("distill_den_log_prob", timing_raw):
            den_lp = self._compute_distill_old_log_probs(
                payload.teacher_batch,
                kept_samples=payload.kept_samples,
                use_teacher=use_teacher,
            )

        kept_count = payload.kept_samples
        distill_mask = payload.distill_mask[:kept_count]
        max_response_len = int(distill_mask.shape[1])

        # Numerator: rollout log-probs from batch (no extra forward pass).
        kept_indices = payload.kept_indices.to(
            dtype=torch.long,
            device=batch.batch["old_log_probs"].device,
        )
        num_lp = batch.batch["old_log_probs"].index_select(0, kept_indices)[
            :, :max_response_len
        ]
        den_lp = den_lp[:kept_count, :max_response_len].to(
            device=num_lp.device, dtype=num_lp.dtype,
        )
        distill_mask = distill_mask.to(device=num_lp.device, dtype=num_lp.dtype)

        # Compute bonus: lambda * stopgrad(s_old - t_old) * mask
        log_ratio = (num_lp - den_lp).detach()
        masked_ratio = log_ratio * distill_mask
        distill_adv = self.distill_settings.lambda_coef * masked_ratio

        # Scatter into full-batch bonus tensor.
        scatter_indices = kept_indices.to(device=bonus.device)
        bonus[scatter_indices, :max_response_len] = distill_adv.to(
            device=bonus.device, dtype=bonus.dtype,
        )

        # Stats for logging.
        valid = masked_ratio[distill_mask > 0]
        stats: dict[str, float] = {
            "distill/sdpo_loss": float(-(distill_adv[distill_mask > 0].mean().item()))
            if valid.numel() > 0
            else 0.0,
            "distill/log_ratio_mean": float(valid.mean().item()) if valid.numel() > 0 else 0.0,
            "distill/log_ratio_std": float(valid.std(unbiased=False).item())
            if valid.numel() > 1
            else 0.0,
            "distill/token_count": float(token_count),
            "distill/skipped_batches": 0.0,
        }
        return bonus, stats

    def _set_distill_meta(self, batch: DataProto, *, enabled_for_step: bool) -> None:
        """Attach per-step distillation settings in batch meta info for actor update."""
        batch.meta_info["distill_enabled"] = bool(enabled_for_step)
        batch.meta_info["distill_lambda"] = float(self.distill_settings.lambda_coef)
        batch.meta_info["distill_use_grpo_loss"] = bool(self.distill_settings.use_grpo_loss)
        batch.meta_info["distill_loss_variant"] = str(self.distill_settings.loss_variant)
        batch.meta_info["distill_alpha"] = float(self.distill_settings.alpha)
        batch.meta_info["distill_is_clip"] = bool(self.distill_settings.is_clip)
        batch.meta_info["distill_full_logit_topk"] = int(self.distill_settings.full_logit_topk)
        batch.meta_info["distill_full_logit_add_tail"] = bool(self.distill_settings.full_logit_add_tail)
        batch.meta_info["distill_negate_sdpo_loss"] = bool(self.distill_settings.negate_sdpo_loss)
        batch.meta_info["distill_use_stale_coefficient"] = bool(self.distill_settings.use_stale_coefficient)
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

        student_batch = payload.student_batch.batch
        teacher_batch = payload.teacher_batch.batch
        max_student_prompt_len = int(student_batch["prompts"].shape[1])
        max_teacher_prompt_len = int(teacher_batch["prompts"].shape[1])
        max_distill_len = int(payload.distill_mask.shape[1])
        if max_student_prompt_len <= 0 or max_teacher_prompt_len <= 0 or max_distill_len <= 0:
            return metrics

        pad_token_id = int(self.tokenizer.pad_token_id)
        student_prompts = torch.full(
            (batch_size, max_student_prompt_len),
            fill_value=pad_token_id,
            dtype=torch.long,
        )
        student_prompt_attention = torch.zeros(
            (batch_size, max_student_prompt_len),
            dtype=torch.long,
        )
        student_responses = torch.full(
            (batch_size, max_distill_len),
            fill_value=pad_token_id,
            dtype=torch.long,
        )
        student_response_mask = torch.zeros(
            (batch_size, max_distill_len),
            dtype=torch.long,
        )
        teacher_prompts = torch.full(
            (batch_size, max_teacher_prompt_len),
            fill_value=pad_token_id,
            dtype=torch.long,
        )
        teacher_prompt_attention = torch.zeros(
            (batch_size, max_teacher_prompt_len),
            dtype=torch.long,
        )
        teacher_responses = torch.full((batch_size, max_distill_len), fill_value=pad_token_id, dtype=torch.long)
        teacher_response_mask = torch.zeros((batch_size, max_distill_len), dtype=torch.long)
        distill_mask = torch.zeros((batch_size, max_distill_len), dtype=torch.float32)
        student_old_log_probs = None
        if payload.student_old_log_probs is not None:
            student_old_log_probs = torch.zeros((batch_size, max_distill_len), dtype=payload.student_old_log_probs.dtype)
        teacher_old_log_probs = None
        if payload.teacher_old_log_probs is not None:
            teacher_old_log_probs = torch.zeros((batch_size, max_distill_len), dtype=payload.teacher_old_log_probs.dtype)

        student_prompt_attention_compact = student_batch["attention_mask"][:, :max_student_prompt_len].long()
        teacher_prompt_attention_compact = teacher_batch["attention_mask"][:, :max_teacher_prompt_len].long()
        for row_idx, kept_idx in enumerate(kept_indices.tolist()):
            if kept_idx < 0 or kept_idx >= batch_size:
                continue
            student_prompts[kept_idx] = student_batch["prompts"][row_idx].long()
            student_prompt_attention[kept_idx] = student_prompt_attention_compact[row_idx]
            student_responses[kept_idx] = student_batch["responses"][row_idx].long()
            student_response_mask[kept_idx] = student_batch["response_mask"][row_idx].long()
            teacher_prompts[kept_idx] = teacher_batch["prompts"][row_idx].long()
            teacher_prompt_attention[kept_idx] = teacher_prompt_attention_compact[row_idx]
            teacher_responses[kept_idx] = teacher_batch["responses"][row_idx].long()
            teacher_response_mask[kept_idx] = teacher_batch["response_mask"][row_idx].long()
            distill_mask[kept_idx] = payload.distill_mask[row_idx].float()
            if student_old_log_probs is not None and payload.student_old_log_probs is not None:
                student_old_log_probs[kept_idx] = payload.student_old_log_probs[row_idx]
            if teacher_old_log_probs is not None and payload.teacher_old_log_probs is not None:
                teacher_old_log_probs[kept_idx] = payload.teacher_old_log_probs[row_idx]

        student_attention_mask = torch.cat([student_prompt_attention, student_response_mask], dim=1)
        student_position_ids = (torch.cumsum(student_attention_mask, dim=1) - 1) * student_attention_mask
        student_input_ids = torch.cat([student_prompts, student_responses], dim=1)

        teacher_attention_mask = torch.cat([teacher_prompt_attention, teacher_response_mask], dim=1)
        teacher_position_ids = (torch.cumsum(teacher_attention_mask, dim=1) - 1) * teacher_attention_mask
        teacher_input_ids = torch.cat([teacher_prompts, teacher_responses], dim=1)

        batch.batch["distill_student_input_ids"] = student_input_ids
        batch.batch["distill_student_attention_mask"] = student_attention_mask
        batch.batch["distill_student_position_ids"] = student_position_ids
        batch.batch["distill_student_responses"] = student_responses
        batch.batch["distill_student_response_mask"] = student_response_mask
        if student_old_log_probs is not None:
            batch.batch["distill_student_old_log_probs"] = student_old_log_probs
        if teacher_old_log_probs is not None:
            batch.batch["distill_teacher_old_log_probs"] = teacher_old_log_probs
        batch.batch["distill_teacher_input_ids"] = teacher_input_ids
        batch.batch["distill_teacher_attention_mask"] = teacher_attention_mask
        batch.batch["distill_teacher_position_ids"] = teacher_position_ids
        batch.batch["distill_teacher_responses"] = teacher_responses
        batch.batch["distill_teacher_response_mask"] = teacher_response_mask
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
                    metrics["distill/skipped_selective_gate"] = 0.0
                    metrics["distill/skipped_attempt_incomplete"] = 0.0

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
                        if self.distill_settings.merge_to_advantages:
                            distill_bonus, distill_metrics = self._compute_distill_bonus(
                                batch=batch,
                                payload=distill_payload,
                                timing_raw=timing_raw,
                            )
                            batch.batch["advantages"] = batch.batch["advantages"] + distill_bonus
                            metrics.update(distill_metrics)
                            self._set_distill_meta(batch=batch, enabled_for_step=False)
                        else:
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
