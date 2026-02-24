"""Joint SDPO self-distillation trainer for multi-episode PPO.

This module implements shared-weight self-distillation where each training step
performs:
1. Standard teacher RL update on full multi-episode rollouts.
2. SDPO-style auxiliary actor update over first-attempt tokens only.

Distillation uses a strict context-overflow guard: if
``student_context_len + teacher_context_len > context_limit``,
the sample is excluded from distillation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto  # type: ignore
from verl.protocol import pad_dataproto_to_divisor
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
    lambda_coef: float = 0.1
    mode: str = "sdpo_self"
    denominator_mode: str = "teacher_adapted_feedback"
    context_limit: int | None = None
    context_overflow_policy: str = "skip_loss"
    min_distill_tokens: int = 1
    teacher_context_attempts: int | None = None


@dataclass
class DistillPayload:
    """Prepared distillation payload for one optimization step."""

    numerator_batch: DataProto
    denominator_batch: DataProto
    distill_mask: torch.Tensor
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
        are malformed, non-cumulative, or insufficient complete attempts exist.
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
            complete_attempt_contexts.append(list(current_prompt_ids))

        accumulated = current_prompt_ids + list(current_completion_ids)
        prev_episode_index = current_episode_index

    if not complete_attempt_contexts:
        return None

    if teacher_context_attempts is None:
        selected_context = complete_attempt_contexts[-1]
    else:
        if len(complete_attempt_contexts) < teacher_context_attempts:
            return None
        selected_context = complete_attempt_contexts[teacher_context_attempts - 1]

    return torch.as_tensor(selected_context, dtype=torch.long)


def estimate_student_context_len(step_records: Sequence[dict[str, Any]]) -> int:
    """Estimate student context length for first-attempt rollout tokens.

    We approximate ``len(tokenize(x, y<t_max))`` using the largest prompt length
    observed among first-attempt steps.

    Args:
        step_records: Step-level records from rollout engine.

    Returns:
        Estimated context length in tokens.
    """
    prompt_lens: list[int] = []
    for step in step_records:
        episode_index = int(step.get("episode_index", 0))
        if episode_index != 0:
            continue
        prompt_ids = step.get("prompt_ids", [])
        if isinstance(prompt_ids, list):
            prompt_lens.append(len(prompt_ids))
    return max(prompt_lens, default=0)


def should_skip_context_overflow(
    student_context_len: int,
    teacher_context_len: int,
    context_limit: int,
) -> bool:
    """Return whether SDPO distillation should be skipped for overflow."""
    return (student_context_len + teacher_context_len) > context_limit


def compute_sdpo_advantages(
    numerator_log_probs: torch.Tensor,
    denominator_log_probs: torch.Tensor,
    distill_mask: torch.Tensor,
    lambda_coef: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute SDPO-style detached coefficients and token advantages.

    Args:
        numerator_log_probs: Log probs under numerator context.
        denominator_log_probs: Log probs under denominator context.
        distill_mask: Binary mask over response tokens to include in distillation.
        lambda_coef: Distillation weight.

    Returns:
        Tuple ``(advantages, stats)`` where:
            advantages: ``lambda * stopgrad(logp_num - logp_den) * mask``.
            stats: Scalar diagnostics for logging.
    """
    log_ratio = (numerator_log_probs - denominator_log_probs).detach()
    masked_ratio = log_ratio * distill_mask
    advantages = lambda_coef * masked_ratio

    token_count = float(distill_mask.sum().item())
    if token_count <= 0:
        return advantages, {
            "distill/sdpo_loss": 0.0,
            "distill/log_ratio_mean": 0.0,
            "distill/log_ratio_std": 0.0,
            "distill/token_count": 0.0,
        }

    valid = masked_ratio[distill_mask > 0]
    stats = {
        "distill/sdpo_loss": float(-(advantages[distill_mask > 0].mean().item())),
        "distill/log_ratio_mean": float(valid.mean().item()),
        "distill/log_ratio_std": float(valid.std(unbiased=False).item() if valid.numel() > 1 else 0.0),
        "distill/token_count": token_count,
    }
    return advantages, stats


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
        teacher_context_attempts_raw = cfg.get("teacher_context_attempts", None)
        teacher_context_attempts = (
            int(teacher_context_attempts_raw) if teacher_context_attempts_raw is not None else None
        )
        if teacher_context_attempts is not None and teacher_context_attempts < 1:
            raise ValueError(
                f"rllm.distill.teacher_context_attempts must be >= 1 when provided, got {teacher_context_attempts}"
            )
        return DistillSettings(
            enable=bool(cfg.get("enable", False)),
            lambda_coef=float(cfg.get("lambda", 0.1)),
            mode=str(cfg.get("mode", "sdpo_self")),
            denominator_mode=str(cfg.get("denominator_mode", "teacher_adapted_feedback")),
            context_limit=int(cfg.get("context_limit", default_limit)) if cfg.get("context_limit", None) is not None else default_limit,
            context_overflow_policy=str(cfg.get("context_overflow_policy", "skip_loss")),
            min_distill_tokens=int(cfg.get("min_distill_tokens", 1)),
            teacher_context_attempts=teacher_context_attempts,
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
        return final_gen_batch_output, metrics

    def _prepare_distill_payload(self, batch: DataProto) -> tuple[DistillPayload | None, dict[str, float]]:
        """Prepare numerator/denominator batches and token masks for distillation."""
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

        total_samples = min(len(self._latest_token_trajectories), batch.batch["responses"].shape[0])
        kept_indices: list[int] = []
        teacher_prompt_tokens_per_sample: list[torch.Tensor] = []
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

            step_records = self._latest_token_trajectories[i].get("step_records", [])
            if not isinstance(step_records, list):
                skipped_hindsight_unavailable += 1
                continue

            student_context_len = estimate_student_context_len(step_records)
            teacher_prompt_tokens = build_hindsight_prompt_tokens_first_n_complete_attempts(
                step_records=step_records,
                teacher_context_attempts=self.distill_settings.teacher_context_attempts,
            )
            if teacher_prompt_tokens is None:
                skipped_hindsight_unavailable += 1
                continue
            teacher_context_len = int(teacher_prompt_tokens.numel())

            if should_skip_context_overflow(
                student_context_len=student_context_len,
                teacher_context_len=teacher_context_len,
                context_limit=limit,
            ):
                skipped_context_overflow += 1
                continue

            kept_indices.append(i)
            teacher_prompt_tokens_per_sample.append(teacher_prompt_tokens)

        if not kept_indices:
            return None, {
                "distill/skipped_context_overflow": float(skipped_context_overflow),
                "distill/skipped_hindsight_unavailable": float(skipped_hindsight_unavailable),
                "distill/kept_ratio": 0.0,
            }

        selected = batch.select_idxs(np.array(kept_indices))
        selected_den = batch.select_idxs(np.array(kept_indices))

        pad_token_id = int(self.tokenizer.pad_token_id)
        prompt_rows: list[torch.Tensor] = teacher_prompt_tokens_per_sample
        selected_distill_mask = distill_mask[np.array(kept_indices)]
        max_prompt_len = max(int(x.numel()) for x in prompt_rows)

        denom_prompts = torch.full(
            (len(prompt_rows), max_prompt_len),
            fill_value=pad_token_id,
            dtype=torch.long,
        )
        for i, row in enumerate(prompt_rows):
            denom_prompts[i, -row.numel() :] = row

        responses = selected_den.batch["responses"]
        response_attention = selected_den.batch["attention_mask"][:, -responses.shape[1] :]
        prompt_lens = torch.as_tensor([int(x.numel()) for x in prompt_rows], dtype=torch.long)
        prompt_pos = torch.arange(max_prompt_len).unsqueeze(0)
        prompt_attention = (prompt_pos >= (max_prompt_len - prompt_lens.unsqueeze(1))).long()
        attention_mask = torch.cat([prompt_attention, response_attention], dim=1)
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask
        input_ids = torch.cat([denom_prompts, responses], dim=1)

        selected_den.batch["prompts"] = denom_prompts
        selected_den.batch["input_ids"] = input_ids
        selected_den.batch["attention_mask"] = attention_mask
        selected_den.batch["position_ids"] = position_ids

        kept_samples = int(selected.batch["responses"].shape[0])
        payload = DistillPayload(
            numerator_batch=selected,
            denominator_batch=selected_den,
            distill_mask=selected_distill_mask,
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

    def _pad_to_world_size(self, batch: DataProto, world_size: int) -> DataProto:
        """Pad a DataProto batch to world size divisor and adjust masks."""
        original_size = int(batch.batch["responses"].shape[0])
        batch, pad_size = pad_dataproto_to_divisor(batch, world_size)
        if pad_size <= 0:
            return batch
        if "response_mask" in batch.batch:
            batch.batch["response_mask"][original_size:] = 0
        if "advantages" in batch.batch:
            batch.batch["advantages"][original_size:] = 0
        if "returns" in batch.batch:
            batch.batch["returns"][original_size:] = 0
        return batch

    def _run_distill_update(
        self,
        payload: DistillPayload | None,
        timing_raw: dict[str, float],
    ) -> dict[str, float]:
        """Run one SDPO-style auxiliary actor update."""
        if not self.distill_settings.enable or payload is None:
            return {
                "distill/sdpo_loss": 0.0,
                "distill/log_ratio_mean": 0.0,
                "distill/log_ratio_std": 0.0,
                "distill/token_count": 0.0,
                "distill/skipped_batches": 1.0,
            }

        token_count = int(payload.distill_mask.sum().item())
        if token_count < self.distill_settings.min_distill_tokens:
            return {
                "distill/sdpo_loss": 0.0,
                "distill/log_ratio_mean": 0.0,
                "distill/log_ratio_std": 0.0,
                "distill/token_count": float(token_count),
                "distill/skipped_batches": 1.0,
            }

        with marked_timer("distill_num_log_prob", timing_raw):
            numerator_log_prob = self.actor_rollout_wg.compute_log_prob(payload.numerator_batch)
        with marked_timer("distill_den_log_prob", timing_raw):
            denominator_log_prob = self.actor_rollout_wg.compute_log_prob(payload.denominator_batch)

        num_lp = numerator_log_prob.batch["old_log_probs"]
        den_lp = denominator_log_prob.batch["old_log_probs"]
        distill_adv, stats = compute_sdpo_advantages(
            numerator_log_probs=num_lp,
            denominator_log_probs=den_lp,
            distill_mask=payload.distill_mask,
            lambda_coef=self.distill_settings.lambda_coef,
        )

        distill_batch = payload.numerator_batch.union(numerator_log_prob)
        distill_batch.batch["advantages"] = distill_adv
        distill_batch.batch["returns"] = distill_adv

        if self.use_reference_policy and self.config.actor_rollout_ref.actor.use_kl_loss:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(distill_batch)
            distill_batch = distill_batch.union(ref_log_prob)

        distill_batch = self._pad_to_world_size(distill_batch, world_size=self.actor_rollout_wg.world_size)
        with marked_timer("update_actor_distill", timing_raw):
            actor_output = self.actor_rollout_wg.update_actor(distill_batch)
        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
        prefixed_actor_metrics = {f"distill_actor/{k}": float(v) for k, v in actor_output_metrics.items()}

        stats.update(prefixed_actor_metrics)
        stats["distill/skipped_batches"] = 0.0
        return stats

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
                        distill_payload, distill_prep_metrics = self._prepare_distill_payload(batch)
                        metrics.update(distill_prep_metrics)

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

                        distill_metrics = self._run_distill_update(
                            payload=distill_payload,
                            timing_raw=timing_raw,
                        )
                        metrics.update(distill_metrics)

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
