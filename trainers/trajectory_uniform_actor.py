"""Trajectory-uniform PPO actor — overrides verl's update_policy.

Implements the loss aggregation:

    L = (1 / N_G) * sum_{i=1..N_G} L_i,
    L_i = (1 / N_t^i) * sum_{token in traj_i} pg_token

where ``N_G`` is the number of trajectories in the batch and ``N_t^i`` is the
total mask=1 token count for trajectory ``i`` (summed across all of i's
expanded segment rows). Each token's per-token weight is precomputed by the
trainer as

    w_token = 1 / (N_G * N_t^i)   for valid tokens of trajectory i
            = 0                    for mask=0 positions

and stored in ``batch.batch["traj_uniform_weight"]`` (shape ``(B, T)``).

The actor's per-rank, per-micro-batch contribution to the loss is

    numer_local += sum_{token in micro-batch} pg_token * w_token * mask_token

After all micro-batches in a mini-batch are processed on this rank, we
reduce ``numer_local`` across all data-parallel ranks (SUM) — this assembles
the global ``L`` even when a trajectory's tokens are split across GPUs by
verl's ``DataProto.chunk(...)`` dispatch. The summed scalar is ``backward``-d
once and the optimizer step follows.

The class is a drop-in replacement for ``verl.workers.actor.DataParallelPPOActor``;
``install_trajectory_uniform_actor()`` (defined at the bottom) monkey-patches
the verl module attributes so ``ActorRolloutRefWorker.init_model`` instantiates
this subclass instead.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, kl_penalty
from verl.utils.device import get_device_id
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.workers.actor.dp_actor import DataParallelPPOActor

from trainers._pg_surrogate import compute_pg_per_token

__all__ = [
    "TrajectoryUniformPPOActor",
    "compute_pg_per_token",
    "install_trajectory_uniform_actor",
]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Tensor field name we expect the trainer to attach. Matches the field
# populated by MultiEpisodeAgentPPOTrainer._transform_agent_trajectories.
TRAJ_UNIFORM_WEIGHT_KEY = "traj_uniform_weight"


def _data_parallel_world_size() -> int:
    """Number of ranks across which the actor's data is split."""
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def _all_reduce_sum(value: torch.Tensor) -> torch.Tensor:
    """In-place sum across all DP ranks; no-op when not distributed."""
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


class TrajectoryUniformPPOActor(DataParallelPPOActor):
    """``DataParallelPPOActor`` with trajectory-uniform PPO loss aggregation.

    See module docstring for the loss formulation. The override leaves the
    forward pass, KL/entropy auxiliaries, and optimizer step intact — only
    the per-token PPO loss aggregation is replaced.
    """

    @GPUMemoryLogger(role="dp actor (traj-uniform)", logger=logger)
    def update_policy(self, data: DataProto):  # noqa: D401 — match parent signature
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]

        # Same select_keys as the parent, plus our trajectory-uniform weight.
        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            TRAJ_UNIFORM_WEIGHT_KEY,
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
        # is_positive is emitted by build_expanded_dataproto in chunk-
        # discounted mode; absent under GRPO. Carry it through micro-batch
        # splits so the TOPR branch below can consume it per-token.
        if "is_positive" in data.batch.keys():
            select_keys.append("is_positive")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = (
            ["multi_modal_inputs"] if has_multi_modal_inputs else []
        )

        # Hard requirement: the trainer must have attached the weight tensor.
        if TRAJ_UNIFORM_WEIGHT_KEY not in data.batch.keys():
            raise RuntimeError(
                f"TrajectoryUniformPPOActor expects '{TRAJ_UNIFORM_WEIGHT_KEY}' "
                f"in data.batch but it was not present. The trainer must build "
                f"this tensor during _transform_agent_trajectories before union."
            )

        # See plan loophole L2: ``_data_parallel_world_size()`` returns
        # ``dist.get_world_size()`` (the full process group). When sequence
        # parallelism is enabled (ulysses_sequence_parallel_size > 1), the
        # world group includes SP ranks and the dp_world_size scaling below
        # over-counts → the per-rank loss × dp_world_size produces a gradient
        # that's too large by a factor of SP. Until we audit and refactor
        # the scaling for SP > 1, fail loudly here so we can never silently
        # train with a corrupted gradient.
        if self.ulysses_sequence_parallel_size != 1:
            raise NotImplementedError(
                "TrajectoryUniformPPOActor's dp_world_size scaling assumes "
                "ulysses_sequence_parallel_size=1; got "
                f"{self.ulysses_sequence_parallel_size}. Refactor "
                "`_data_parallel_world_size()` to return DP-only size before "
                "enabling sequence parallelism."
            )

        data = data.select(
            batch_keys=select_keys,
            non_tensor_batch_keys=non_tensor_select_keys,
        )

        mini_batches = data.split(self.config.ppo_mini_batch_size)
        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        # FSDP/DDP averages gradients across DP ranks. Our target loss is the
        # global sum (over all rows on all ranks) of pg * w * mask. To make
        # the post-FSDP-mean gradient equal that target, scale each rank's
        # backward'd loss by N_DP. (See module docstring.)
        dp_world_size = _data_parallel_world_size()

        metrics: dict = {}
        for _epoch in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = (
                        self.config.ppo_max_token_len_per_gpu
                        * self.ulysses_sequence_parallel_size
                    )
                    micro_batches, _ = prepare_dynamic_batch(
                        mini_batch, max_token_len=max_token_len
                    )
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size
                        // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(
                        self.config.ppo_micro_batch_size_per_gpu
                    )

                self.actor_optimizer.zero_grad()

                # Track the running per-rank numerator so we can report the
                # global loss after the all-reduce.
                running_numer_local = torch.tensor(0.0, device=get_device_id())

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics: dict = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    advantages = model_inputs["advantages"]
                    traj_uniform_weight = model_inputs[TRAJ_UNIFORM_WEIGHT_KEY]

                    entropy_coeff = self.config.entropy_coeff
                    calculate_entropy = entropy_coeff != 0

                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                    )

                    if (
                        hasattr(self.config, "use_rollout_log_probs")
                        and self.config.use_rollout_log_probs
                    ):
                        old_log_prob = model_inputs["old_log_probs"]
                    elif on_policy:
                        old_log_prob = log_prob.detach()
                    else:
                        old_log_prob = model_inputs["old_log_probs"]

                    # ---- Per-token PPO surrogate + optional TOPR split ----
                    # Standard verl PPO clipping plus the TOPR branch from
                    # arXiv:2512.24873 Eq. 5. TOPR fires iff the batch
                    # carries an ``is_positive`` per-token tensor (which
                    # the trainer attaches in chunk-discounted-topr mode
                    # when its ``topr_split.enable`` is true). Positive
                    # chunks use a plain SL-style update ``-advantages``
                    # (no IS ratio, no clip); negatives fall through to
                    # the existing 3-way-clipped surrogate. Inference/
                    # training mismatch handling is unaffected.
                    #
                    # We gate purely on tensor presence — no read of
                    # ``self.config.topr_split.*`` here, because verl's
                    # FSDPActorConfig is a strict dataclass that rejects
                    # unknown keys (the TOPR enable flag lives instead
                    # under ``rllm.advantage_method.chunk_discounted_topr``,
                    # which the trainer reads when it builds the batch).
                    cliprange = self.config.clip_ratio
                    cliprange_low = (
                        self.config.clip_ratio_low
                        if self.config.clip_ratio_low is not None
                        else cliprange
                    )
                    cliprange_high = (
                        self.config.clip_ratio_high
                        if self.config.clip_ratio_high is not None
                        else cliprange
                    )
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)

                    is_pos_field = model_inputs.get("is_positive")
                    surrogate = compute_pg_per_token(
                        advantages=advantages,
                        log_prob=log_prob,
                        old_log_prob=old_log_prob,
                        cliprange_low=cliprange_low,
                        cliprange_high=cliprange_high,
                        clip_ratio_c=clip_ratio_c,
                        is_positive=is_pos_field,
                        topr_enabled=is_pos_field is not None,
                    )
                    pg_per_token = surrogate["pg_per_token"]
                    pg_losses1 = surrogate["pg_losses1"]
                    pg_losses2 = surrogate["pg_losses2"]
                    clip_pg_losses1 = surrogate["clip_pg_losses1"]
                    pg_losses3 = surrogate["pg_losses3"]
                    negative_approx_kl = surrogate["negative_approx_kl"]

                    # ---- Trajectory-uniform aggregation ----
                    # numer_local += sum_token (pg * w * mask). Weights already
                    # bake in 1 / (N_G * N_t^i), so summing across all rows on
                    # all ranks reproduces L = (1/N_G) Σ_i L_i.
                    numer_micro = (pg_per_token * traj_uniform_weight * response_mask).sum()

                    # Standard PPO diagnostics (unchanged from verl):
                    # masked_mean uses a small epsilon internally; safe even
                    # when response_mask is all zero on a micro-batch.
                    from verl.utils import torch_functional as verl_F

                    pg_clipfrac = verl_F.masked_mean(
                        torch.gt(pg_losses2, pg_losses1).float(), response_mask
                    )
                    pg_clipfrac_lower = verl_F.masked_mean(
                        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(),
                        response_mask,
                    )
                    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

                    micro_batch_metrics["actor/pg_clipfrac"] = pg_clipfrac.detach().item()
                    micro_batch_metrics["actor/ppo_kl"] = ppo_kl.detach().item()
                    micro_batch_metrics["actor/pg_clipfrac_lower"] = (
                        pg_clipfrac_lower.detach().item()
                    )

                    # ---- Optional auxiliary losses ----
                    # Entropy bonus uses verl's standard token-mean — this is
                    # an exploration regularizer, NOT trajectory-aggregated.
                    if calculate_entropy:
                        entropy_loss = agg_loss(
                            loss_mat=entropy,
                            loss_mask=response_mask,
                            loss_agg_mode="token-mean",
                        )
                        # Scale entropy term by dp_world_size to keep it on
                        # the same gradient scale as the trajectory-uniform
                        # numer below (FSDP averages gradients across ranks).
                        entropy_term = -entropy_coeff * entropy_loss * dp_world_size
                    else:
                        entropy_term = torch.tensor(0.0, device=numer_micro.device)

                    # KL loss vs reference, also trajectory-uniform-weighted.
                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kld = kl_penalty(
                            logprob=log_prob,
                            ref_logprob=ref_log_prob,
                            kl_penalty=self.config.kl_loss_type,
                        )
                        kl_micro = (kld * traj_uniform_weight * response_mask).sum()
                        kl_term = self.config.kl_loss_coef * kl_micro
                        micro_batch_metrics["actor/kl_loss"] = (
                            kl_micro.detach().item() * dp_world_size
                        )
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef
                    else:
                        kl_term = torch.tensor(0.0, device=numer_micro.device)

                    # Total per-rank loss for this micro-batch. Multiply by
                    # dp_world_size so post-FSDP-mean-of-gradients equals
                    # d(L_target) / d(theta).
                    loss_micro = (numer_micro + kl_term) * dp_world_size + entropy_term

                    if self.scaler is not None:
                        self.scaler.scale(loss_micro).backward()
                    else:
                        loss_micro.backward()

                    running_numer_local = (
                        running_numer_local + numer_micro.detach()
                    )
                    micro_batch_metrics["actor/pg_loss_micro"] = (
                        numer_micro.detach().item() * dp_world_size
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                # Cross-DP-rank scalar loss for logging.
                global_numer = _all_reduce_sum(running_numer_local.clone())
                pg_loss_global = global_numer.detach().item()
                append_to_dict(metrics, {"actor/pg_loss": pg_loss_global})

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics


def install_trajectory_uniform_actor() -> None:
    """Replace verl's ``DataParallelPPOActor`` with the trajectory-uniform variant.

    Must be called BEFORE the actor worker's ``init_model`` runs (which is
    triggered by ``init_workers`` on the trainer). Idempotent — re-importing
    or re-calling is safe.

    The replacement is at the module-attribute level: ``ActorRolloutRefWorker``
    does ``from verl.workers.actor import DataParallelPPOActor`` inside its
    init, so updating the module attribute is sufficient.
    """
    import verl.workers.actor as _verl_actor_pkg
    import verl.workers.actor.dp_actor as _verl_dp_actor

    # Idempotency check
    if getattr(_verl_actor_pkg.DataParallelPPOActor, "_traj_uniform_installed", False):
        return

    TrajectoryUniformPPOActor._traj_uniform_installed = True  # type: ignore[attr-defined]

    _verl_actor_pkg.DataParallelPPOActor = TrajectoryUniformPPOActor
    _verl_dp_actor.DataParallelPPOActor = TrajectoryUniformPPOActor
