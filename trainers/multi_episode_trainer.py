"""Custom trainer that extracts multi-episode metrics.

This trainer extends AgentPPOTrainer to extract additional metrics from
environments that have a get_metrics() method, such as MultiEpisodeEnv.

It also enables PPO training on every segment of summarized trajectories:
when ``summarization_config["enable"]`` is True, the trainer wraps the
actor's ``update_actor`` so that just before the actual gradient step,
the per-trajectory batch (advantage already computed by GRPO) is expanded
into per-segment rows with broadcast advantages and trajectory-uniform
per-token weights. The actor itself is replaced by
:class:`TrajectoryUniformPPOActor`, which uses those weights to produce a
loss equal to ``L = (1/N_G) Σ_i (1/N_t^i) Σ_{token in i} ℓ_token``.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto  # type: ignore

from rllm.engine.agent_execution_engine import AsyncAgentExecutionEngine
from rllm.trainer.verl.agent_ppo_trainer import AgentPPOTrainer
from rllm.utils import colorful_print

from trainers.chunk_advantage import compute_chunk_returns_for_batch
from trainers.segment_expansion import (
    _mask1_runs,
    build_expanded_dataproto,
    extract_advantage_per_trajectory,
)
# Note: the trajectory-uniform actor patch is NOT applied here in the driver.
# Its eager import of verl.workers.actor.dp_actor would trigger
# torch.cuda.is_available() (called at module top of verl.utils.device) which
# initializes CUDA in the driver process and corrupts Ray's per-worker GPU
# assignment. The patch is applied in EVERY Python process (including Ray
# workers) via a `.pth` file in the conda env's site-packages — see the
# top-level `orbit_segtrain_patch.py` module and the `scripts/train_*.sh`
# setup that writes the .pth file.


# Termination reasons that mark a trajectory as "truncated". Mirrors the
# tuple at ``summarizing_engine.py:627`` used to zero step rewards in the
# engine; we keep a separate copy to avoid importing engine internals
# into the trainer module.
DEFAULT_TRUNCATION_REASONS: tuple[str, ...] = (
    "TRUNCATION",
    "PROMPT_TRUNCATION",
    "SUMMARIZATION_BUDGET_EXCEEDED",
    "SUMMARIZATION_FAILED",
)


def _is_truncated_trajectory(traj: dict, reasons: set[str]) -> bool:
    """True iff ``traj`` was produced by a truncation-class termination."""
    return traj.get("termination_reason") in reasons


def compute_kept_trajectory_indices(
    trajectories: list[dict],
    *,
    enable: bool,
    is_validation: bool,
    reasons: set[str] | None = None,
) -> tuple[list[int], bool]:
    """Return ``(kept_indices_into_input, fallback_applied)``.

    ``kept_indices_into_input`` is a sorted-ascending list of positions
    in ``trajectories`` that survive the filter, suitable for indexing
    both ``raw_trajs`` and the parallel ``DataProto`` (via
    ``select_idxs``) so they stay aligned row-for-row.

    ``fallback_applied`` is True when *all* trajectories would be
    dropped; in that case we return the full index list so the caller
    keeps the unfiltered batch, avoiding an empty-batch crash in
    ``build_expanded_dataproto`` / Ray padding.

    No-ops (returns the full index list, ``fallback=False``) when
    ``enable`` is False, ``is_validation`` is True, or ``trajectories``
    is empty.
    """
    n = len(trajectories)
    full = list(range(n))
    if not enable or is_validation or n == 0:
        return full, False
    reasons_set = (
        set(reasons) if reasons is not None
        else set(DEFAULT_TRUNCATION_REASONS)
    )
    kept = [
        i for i, t in enumerate(trajectories)
        if not _is_truncated_trajectory(t, reasons_set)
    ]
    if not kept:
        return full, True
    return kept, False


def filter_truncated_trajectories(
    trajectories: list[dict],
    *,
    enable: bool,
    is_validation: bool,
    reasons: set[str] | None = None,
) -> tuple[list[dict], bool]:
    """Thin wrapper over :func:`compute_kept_trajectory_indices` that
    returns the actual filtered ``trajectories`` list instead of indices.

    Kept for backwards-compatibility and convenience in callers that
    don't need to slice a parallel DataProto.
    """
    kept_idxs, fallback = compute_kept_trajectory_indices(
        trajectories,
        enable=enable,
        is_validation=is_validation,
        reasons=reasons,
    )
    return [trajectories[i] for i in kept_idxs], fallback


class MultiEpisodeAsyncAgentExecutionEngine(AsyncAgentExecutionEngine):
    """Execution engine that extracts metrics from environments with get_metrics()."""

    async def run_agent_trajectory_async(self, idx, application_id, seed=0, mode="Text", **kwargs):
        """Run trajectory and include environment metrics if available."""
        # Store reference to env before it might be modified
        env = self.envs[idx]

        # Call parent method
        result = await super().run_agent_trajectory_async(
            idx, application_id, seed=seed, mode=mode, **kwargs
        )

        # If Token mode, extract additional metrics from environment
        if mode == "Token" and isinstance(result, dict) and "metrics" in result:
            if hasattr(env, "get_metrics") and callable(env.get_metrics):
                try:
                    env_metrics = env.get_metrics()
                    if isinstance(env_metrics, dict):
                        # Flatten nested keys and add to metrics
                        for key, value in env_metrics.items():
                            flat_key = key.replace("/", "_")
                            result["metrics"][flat_key] = value
                except Exception:
                    pass  # Silently ignore if metrics extraction fails

        return result


class MultiEpisodeAgentPPOTrainer(AgentPPOTrainer):
    """Trainer that uses MultiEpisodeAsyncAgentExecutionEngine for metrics extraction.

    Supports separate environment classes for training and validation, enabling
    scenarios like training with SingleEpisodeEnv but validating with MultiEpisodeEnv.
    """

    def __init__(
        self,
        val_env_class: type | None = None,
        val_env_args: dict | None = None,
        summarization_config: dict | None = None,
        **kwargs,
    ):
        """Initialize the trainer with optional validation environment class.

        Args:
            val_env_class: Optional validation environment class. If provided,
                uses this class instead of env_class during validation.
            val_env_args: Optional validation environment arguments. If provided,
                these override env_args during validation.
            summarization_config: Optional summarization configuration dict.
                When provided, uses SummarizingAgentExecutionEngine.
            **kwargs: Arguments passed to parent AgentPPOTrainer.
        """
        super().__init__(**kwargs)
        self.val_env_class = val_env_class
        self.val_env_args = val_env_args
        self.summarization_config = summarization_config
        self._is_validation_mode = False
        # Accumulates per-trajectory engine metrics (e.g. summarization_count,
        # segment_count) across all val batches. Populated by
        # _transform_agent_trajectories when in validation mode and consumed
        # by _validate_agent at the end to emit val/{data_source}/<k>/<stat>
        # and val/all/<k>/<stat> metrics.
        self._val_traj_metrics_buffer: list[tuple[str, dict]] = []
        # Cache for the most recent rollout's raw trajectory dicts (with
        # their .segments list). Stashed in _transform_agent_trajectories
        # and consumed by the wrapped update_actor (when
        # summarization is enabled) to expand the per-trajectory batch into
        # per-segment rows for the trajectory-uniform PPO loss.
        self._cached_raw_trajectories: list[dict] | None = None

        # Resolve the rollout-correction (TIS) configuration. See
        # _resolve_rollout_correction for the gating rules.
        self._rollout_corr_config, self._effective_rollout_is = (
            self._resolve_rollout_correction()
        )

    def _resolve_rollout_correction(self) -> tuple[Any, str | None]:
        """Validate ``algorithm.rollout_correction`` and resolve effective TIS mode.

        Returns ``(rollout_corr_config, effective_rollout_is)``:

        * ``rollout_corr_config``: the OmegaConf node under
          ``algorithm.rollout_correction`` (or ``None`` if unset).
        * ``effective_rollout_is``: the value of ``rollout_is`` that the
          trainer actually applies — either ``"token"`` (TIS on) or
          ``None`` (off). When TOPR is enabled, this is forced to ``None``
          with a single INFO log line.

        Hard-raises on unsupported verl knobs (sequence-level IS, bypass
        mode, rejection sampling, batch-normalize). The plan defers those
        to a future iteration; failing loudly is preferable to silently
        following an unvalidated code path.
        """
        rc = OmegaConf.select(self.config, "algorithm.rollout_correction", default=None)
        if rc is None:
            return None, None

        rollout_is = rc.get("rollout_is", None)
        if rollout_is not in (None, "token"):
            raise ValueError(
                "algorithm.rollout_correction.rollout_is must be null or 'token'; "
                f"got {rollout_is!r}. Sequence-level IS is out of scope for this trainer."
            )
        if rc.get("bypass_mode", False):
            raise ValueError(
                "algorithm.rollout_correction.bypass_mode=true is not supported. "
                "Pin bypass_mode=false (decoupled mode); only the FSDP-old PPO ratio is validated."
            )
        if rc.get("rollout_rs", None) is not None:
            raise ValueError(
                "algorithm.rollout_correction.rollout_rs is not supported "
                f"(got {rc.get('rollout_rs')!r}). Pin rollout_rs=null; rejection sampling is deferred."
            )
        if rc.get("rollout_is_batch_normalize", False):
            raise ValueError(
                "algorithm.rollout_correction.rollout_is_batch_normalize=true is not supported. "
                "Pin rollout_is_batch_normalize=false."
            )

        # Soft guard: TOPR + TIS is intentionally not validated. Auto-disable.
        topr_enabled = bool(
            OmegaConf.select(
                self.config,
                "rllm.advantage_method.chunk_discounted_topr.enable",
                default=False,
            )
        )
        method_name = OmegaConf.select(
            self.config, "rllm.advantage_method.name", default="grpo"
        )
        topr_active = topr_enabled and method_name == "chunk_discounted_topr"
        if rollout_is == "token" and topr_active:
            print(
                "[trainer] TOPR is enabled; disabling default TIS correction "
                "(algorithm.rollout_correction.rollout_is) for this run."
            )
            return rc, None

        return rc, rollout_is

    def _get_engine_class(self):
        """Return the execution engine class to use."""
        if self.summarization_config and self.summarization_config.get("enable"):
            from trainers.summarizing_engine import SummarizingAgentExecutionEngine

            return SummarizingAgentExecutionEngine
        return MultiEpisodeAsyncAgentExecutionEngine

    def _balance_batch(self, batch, metrics, **kwargs):
        """No-op when segment training is enabled.

        The skinny per-trajectory batch is a fictional carrier whose row
        order is bound to ``_cached_raw_trajectories`` (set in
        :meth:`_transform_agent_trajectories`). The parent's
        :meth:`_balance_batch` reorders rows for DP load balancing using
        ``attention_mask`` over ``segments[0]`` only — that's neither the
        actual training workload (the actor trains on the segment-expanded
        rows produced inside :meth:`_expanded_update_actor`) nor invariant
        under our positional cache lookup. Letting it run silently swaps
        advantages between trajectories.

        We disable the skinny reorder here and apply
        :meth:`RayPPOTrainer._balance_batch` directly to the post-expansion
        batch inside :meth:`_expanded_update_actor`, where the workload
        signal is real and there's nothing to de-sync.
        """
        if self._segment_training_enabled:
            return
        return super()._balance_batch(batch, metrics, **kwargs)

    @property
    def _segment_training_enabled(self) -> bool:
        """True iff we should train PPO on every summarization segment.

        Gated on summarization being enabled in the agent config — when
        disabled, every trajectory has exactly one segment and the
        expansion is a no-op, so there's no reason to incur its overhead.
        """
        return bool(
            self.summarization_config and self.summarization_config.get("enable")
        )

    def init_workers(self):
        """Initialize workers with custom execution engine.

        We do NOT install the trajectory-uniform actor patch here in the
        driver process. The patch must run inside each Ray worker process
        anyway (workers are separate Python processes); that's handled by
        the worker_process_setup_hook registered in
        :func:`trainers.train_multi_episode.run_ppo_agent`. Calling
        ``install_trajectory_uniform_actor`` in the driver eager-imports
        ``verl.workers.actor.dp_actor``, which transitively imports
        ``verl.utils.device`` → ``torch.cuda.is_available()`` → initializes
        CUDA in the driver process. With CUDA initialized in the driver,
        Ray's per-worker GPU assignment downstream goes wrong and the
        actor workers fail to bind their devices with
        "CUDA-capable device(s) is/are busy or unavailable".
        """
        # Call grandparent's init_workers (skip AgentPPOTrainer's)
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        RayPPOTrainer.init_workers(self)

        engine_args = OmegaConf.to_container(self.config.rllm.agent.get("engine_args", {})) or {}
        n_parallel_agents = (
            engine_args.pop("n_parallel_agents", None)
            or self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
        )
        print(f"n_parallel_agents: {n_parallel_agents}")

        engine_cls = self._get_engine_class()
        # Use our custom execution engine instead
        self.agent_execution_engine = engine_cls(
            rollout_engine=self.async_rollout_manager,
            config=self.config,
            engine_name="verl",
            tokenizer=self.tokenizer,
            model_path=self.config.actor_rollout_ref.model.path,
            max_steps=self.config.rllm.agent.max_steps,
            max_response_length=self.config.data.max_response_length,
            max_prompt_length=self.config.data.max_prompt_length,
            agent_class=self.agent_class,
            agent_args=self.agent_args,
            env_class=self.env_class,
            env_args=self.env_args,
            enforce_max_prompt_length=self.config.rllm.stepwise_advantage.enable,
            trajectory_timeout=self.config.rllm.agent.trajectory_timeout,
            overlong_filter=self.config.rllm.agent.get("overlong_filter", False),
            disable_thinking=self.config.rllm.disable_thinking,
            n_parallel_agents=n_parallel_agents,
            **engine_args,
        )

        # When training PPO on every summarization segment, wrap
        # actor_rollout_wg.update_actor so it expands the per-trajectory
        # batch into per-segment rows just before the gradient step. The
        # wrap is a method-attribute swap on the Python wrapper instance
        # (the underlying Ray dispatch is unaffected — we still call the
        # original method from inside the wrapper).
        if self._segment_training_enabled:
            self._install_segment_aware_update_actor()

    def _install_segment_aware_update_actor(self) -> None:
        """Wrap ``actor_rollout_wg.update_actor`` to expand segments first.

        The wrapped function:
          1. Takes the per-trajectory batch (as built by the parent fit_agent
             flow up through ``compute_advantage``).
          2. Reads the cached raw trajectory dicts from
             ``self._cached_raw_trajectories``.
          3. Extracts the per-trajectory scalar advantage A_i from
             ``batch.batch["advantages"]``.
          4. Builds the expanded per-segment DataProto with broadcast
             advantages, ``traj_uniform_weight``, and replicated source
             non-tensor fields.
          5. Re-runs ``compute_log_prob`` on the expanded batch (each segment
             has different prompt+response tokens than the per-trajectory
             segment[0]).
          6. Calls the original ``update_actor`` on the expanded batch (the
             actor is the trajectory-uniform variant, which uses
             ``traj_uniform_weight`` to compute the desired aggregation).
        """
        import torch as _torch

        original_update_actor = self.actor_rollout_wg.update_actor

        # Imported once here for use inside the closure: we need to call the
        # parent class's `_balance_batch` directly to bypass our trainer's
        # no-op override (which protects the skinny per-trajectory batch).
        # Inside `_expanded_update_actor` the batch we balance is either the
        # post-expansion batch (slow path) or the weight-attached skinny
        # batch (fast path), both of which represent real `update_actor`
        # workload and are safe to reorder.
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        def _expanded_update_actor(batch: DataProto):
            raw_trajs = self._cached_raw_trajectories

            # ---- Optional truncation filter ----
            # Filter here (rather than in _transform_agent_trajectories)
            # because the upstream prompt batch has already been unioned
            # with our transformed batch; filtering earlier would break
            # batch.union's size-match assertion. select_idxs keeps the
            # prompt-side rows aligned with the filtered trajectories —
            # both `batch` and `raw_trajs` are sliced by the same sorted
            # ascending `kept_idxs`, so batch[j] still corresponds to
            # raw_trajs[j] downstream (which is what
            # `compute_chunk_returns_for_batch` and the per-row advantage
            # fill loop assume).
            if raw_trajs is not None:
                filter_enable = bool(
                    OmegaConf.select(
                        self.config,
                        "rllm.filter_truncated_trajectories.enable",
                        default=False,
                    )
                )
                reasons_cfg = OmegaConf.select(
                    self.config,
                    "rllm.filter_truncated_trajectories.reasons",
                    default=None,
                )
                kept_idxs, fallback_applied = compute_kept_trajectory_indices(
                    raw_trajs,
                    enable=filter_enable,
                    is_validation=getattr(self, "_is_validation_mode", False),
                    reasons=set(reasons_cfg) if reasons_cfg is not None else None,
                )
                n_before = len(raw_trajs)
                if fallback_applied:
                    colorful_print(
                        f"[filter_truncated] all {n_before} trajectories "
                        "would be dropped this step; falling back to "
                        "unfiltered batch.",
                        "red",
                    )
                elif len(kept_idxs) < n_before:
                    # Slice in lockstep — preserves credit assignment:
                    # raw_trajs[j] still pairs with batch row j after this.
                    batch = batch.select_idxs(kept_idxs)
                    raw_trajs = [raw_trajs[i] for i in kept_idxs]

            if raw_trajs is None or not any(t.get("segments") for t in raw_trajs):
                # No cached segments (or every trajectory was 1-segment) —
                # the expansion is a no-op. Run with the per-trajectory
                # batch directly. We still need traj_uniform_weight = 1/N_t^i
                # per row so the trajectory-uniform actor's loss math works.
                attached = self._attach_traj_uniform_weight_inplace(batch)
                # Skinny batch IS what update_actor trains on in the fast
                # path, so balance it here on the actual workload (the
                # parent method's reorder is bypassed via the explicit
                # RayPPOTrainer call to skip our no-op override).
                RayPPOTrainer._balance_batch(
                    self, attached, metrics={},
                    logging_prefix="expanded_seqlen",
                )
                return original_update_actor(attached)

            n_g = batch.batch["advantages"].shape[0]
            assert n_g == len(raw_trajs), (
                f"advantages batch size {n_g} != cached trajectory count {len(raw_trajs)}"
            )

            # Replicate source non-tensor fields per segment row.
            source_non_tensor = {
                k: v for k, v in batch.non_tensor_batch.items()
                if hasattr(v, "shape") and v.shape and v.shape[0] == n_g
            }

            # Method dispatch:
            #   "grpo" (default): GRPO advantage already in batch; broadcast
            #       the per-trajectory scalar to every valid token of every
            #       segment row (existing behavior — bit-for-bit unchanged).
            #   "chunk_discounted_topr": IPA-style sibling method. Bypass
            #       compute_advantage's GRPO normalization; instead compute
            #       per-chunk discounted returns G_k = γ^Δ · R from the
            #       per-trajectory step_metadata + episode_rewards stamped
            #       on raw_trajs by SummarizingAgentExecutionEngine, and
            #       fill advantages per-chunk plus a per-token is_positive
            #       tensor for the actor's TOPR branch.
            method_name = OmegaConf.select(
                self.config, "rllm.advantage_method.name", default="grpo"
            )
            if method_name == "grpo":
                advantages_per_traj = extract_advantage_per_trajectory(
                    batch.batch["advantages"], batch.batch["response_mask"]
                )
                expansion = build_expanded_dataproto(
                    raw_trajs,
                    advantages_per_trajectory=advantages_per_traj,
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_prompt_length=self.config.data.max_prompt_length,
                    max_response_length=self.config.data.max_response_length,
                    source_non_tensor_batch=source_non_tensor,
                )
            elif method_name == "chunk_discounted_topr":
                cd_cfg = self.config.rllm.advantage_method.chunk_discounted_topr
                scope = OmegaConf.select(cd_cfg, "reward_scope", default="per_episode")
                gamma = float(OmegaConf.select(cd_cfg, "gamma", default=0.95))
                # The TOPR positive/negative split lives under the method
                # config — NOT under actor_rollout_ref.actor, because verl's
                # FSDPActorConfig is a strict dataclass and rejects unknown
                # keys (Hydra would crash at instantiate time).
                topr_enable = bool(
                    OmegaConf.select(cd_cfg, "topr_split.enable", default=True)
                )
                per_chunk_returns = compute_chunk_returns_for_batch(
                    raw_trajs, scope=scope, gamma=gamma,
                )
                # Overwrite the per-trajectory advantage tensor with the
                # chunk-discounted G_k values for segment[0]. The actor
                # trains on ``expanded_batch`` (built below), but verl's
                # ``compute_data_metrics`` reads ``batch.batch["advantages"]``
                # for ``critic/advantages/{mean,max,min}``. Without this
                # overwrite, those metrics would still report GRPO
                # advantages even when the trainer is using the
                # chunk-discounted method, which is misleading.
                _adv = batch.batch["advantages"]
                _adv.zero_()
                _seg0_mask = batch.batch["response_mask"]
                for traj_i in range(n_g):
                    runs = _mask1_runs(_seg0_mask[traj_i])
                    traj_returns = per_chunk_returns[traj_i]
                    for run_idx, (start, end) in enumerate(runs):
                        if run_idx >= len(traj_returns):
                            break
                        G_k, _ = traj_returns[run_idx]
                        _adv[traj_i, start:end] = float(G_k)
                # Diagnostics: surface mismatches between trajectory
                # rewards (what wandb's `traj/<ds>/score_*` reports) and
                # what our chunk-fill ends up writing into `advantages`.
                # The first run after a config change is when these are
                # most useful, so we always print compactly.
                _all_g = [g for traj in per_chunk_returns for g, _ in traj]
                _ep_rewards_all = [
                    r for t in raw_trajs
                    for r in (t.get("episode_rewards") or [])
                ]
                _n_pos_eps = sum(1 for r in _ep_rewards_all if r > 0)
                _n_eps_total = len(_ep_rewards_all)
                _n_chunks_with_nonzero_g = sum(1 for g in _all_g if g != 0.0)
                _n_trajs_with_step_metadata = sum(
                    1 for t in raw_trajs if t.get("step_metadata")
                )
                print(
                    f"[chunk-discounted-topr] step diagnostics: "
                    f"trajs={len(raw_trajs)} "
                    f"trajs_with_step_metadata={_n_trajs_with_step_metadata} "
                    f"episodes_total={_n_eps_total} "
                    f"positive_episodes={_n_pos_eps} "
                    f"chunks_total={len(_all_g)} "
                    f"chunks_with_nonzero_G={_n_chunks_with_nonzero_g}"
                )
                # Spot-check: print episode_rewards for the first couple of
                # trajectories. If wandb shows score>0 but these are all
                # zeros / empty, the engine→cache propagation broke and the
                # method config is reading a stale field.
                for _i, _t in enumerate(raw_trajs[:3]):
                    print(
                        f"[chunk-discounted-topr] traj[{_i}]: "
                        f"episode_rewards={_t.get('episode_rewards')} "
                        f"step_metadata_len={len(_t.get('step_metadata') or [])} "
                        f"summarization_boundaries={_t.get('summarization_boundaries')}"
                    )
                expansion = build_expanded_dataproto(
                    raw_trajs,
                    per_chunk_returns=per_chunk_returns,
                    emit_is_positive=topr_enable,
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_prompt_length=self.config.data.max_prompt_length,
                    max_response_length=self.config.data.max_response_length,
                    source_non_tensor_batch=source_non_tensor,
                )
                # Post-build sanity: confirm the tensor that the actor
                # actually sees has the expected non-zero entries.
                _adv = expansion.data.batch.get("advantages")
                _is_pos = expansion.data.batch.get("is_positive")
                if _adv is not None:
                    print(
                        f"[chunk-discounted-topr] tensor stats: "
                        f"advantages.abs().sum()={float(_adv.abs().sum().item()):.4f} "
                        f"advantages>0 fraction="
                        f"{float((_adv > 0).float().mean().item()):.4f} "
                        f"is_positive>0 fraction="
                        f"{(float((_is_pos > 0).float().mean().item()) if _is_pos is not None else 'n/a')}"
                    )
            else:
                raise ValueError(
                    f"unknown rllm.advantage_method.name {method_name!r}; "
                    "expected 'grpo' or 'chunk_discounted_topr'."
                )
            expanded_batch = expansion.data

            # Carry meta_info forward (temperature, etc.) — required by
            # actor's update_policy.
            expanded_batch.meta_info = dict(batch.meta_info)

            # Pad the expanded batch to a multiple of the actor worker group's
            # world size. Verl's worker dispatch (DataProto.chunk) requires
            # equal-size chunks per rank — Σ K_i isn't generally divisible by
            # world_size. pad_dataproto_to_divisor pads by repeating rows
            # from the front; we then zero traj_uniform_weight on the padded
            # rows so they contribute nothing to the trajectory-uniform loss.
            from verl.protocol import pad_dataproto_to_divisor

            world_size = self.actor_rollout_wg.world_size
            expanded_batch, pad_size = pad_dataproto_to_divisor(
                expanded_batch, world_size
            )
            if pad_size > 0:
                # Zero out the loss contribution of padded rows by killing
                # their per-token weight. The actor's micro-batch loss is
                # sum(pg * traj_uniform_weight * response_mask); with w=0,
                # padded rows add 0 to the loss and 0 to the gradient.
                expanded_batch.batch["traj_uniform_weight"][-pad_size:] = 0.0

            # Balance the EXPANDED batch on actual seq lengths so each DP
            # rank gets a comparable workload during compute_log_prob and
            # update_actor. Bypasses our no-op override (which only protects
            # the skinny pre-expansion batch). Per-row tensors
            # (`advantages`, `traj_uniform_weight`) are already correctly
            # bound to their rows, so reordering preserves correctness —
            # each row's per-token weight still attaches to that row's
            # mask=1 positions.
            RayPPOTrainer._balance_batch(
                self, expanded_batch, metrics={},
                logging_prefix="expanded_seqlen",
            )

            # Recompute old_log_probs on the expanded batch (each segment's
            # tokens are different from the per-trajectory segment[0]).
            old_log_prob_output = self.actor_rollout_wg.compute_log_prob(
                expanded_batch
            )
            # Drop entropies field if present — we don't use it here.
            if "entropys" in old_log_prob_output.batch.keys():
                old_log_prob_output.batch.pop("entropys")
            expanded_batch = expanded_batch.union(old_log_prob_output)

            # Optional rollout/training importance-sampling correction (TIS).
            # Gated on algorithm.rollout_correction.rollout_is=="token" and
            # the segments carrying per-token rollout_log_probs. Verl's
            # helper computes w_t = clamp(exp(old - rollout), max=C),
            # attaches `rollout_is_weights` to the batch, and updates the
            # response_mask if rejection sampling is configured (we pin RS
            # off via _resolve_rollout_correction, so the mask is unchanged
            # in practice).
            tis_metrics: dict | None = None
            if (
                self._effective_rollout_is == "token"
                and "rollout_log_probs" in expanded_batch.batch.keys()
            ):
                # Optional pre-IS debug dump: when ORBIT_TIS_DEBUG=1, print
                # the first few mask=1 token logp pairs (vLLM rollout vs
                # FSDP recompute) and the trailing prompt context. Used to
                # diagnose vLLM↔FSDP logp drift (suspect weights, kv-cache,
                # or context mismatch). Off by default.
                if os.environ.get("ORBIT_TIS_DEBUG") == "1":
                    self._dump_tis_debug(expanded_batch)
                from verl.trainer.ppo.rollout_corr_helper import (
                    compute_rollout_correction_and_add_to_batch,
                )
                expanded_batch, tis_metrics = compute_rollout_correction_and_add_to_batch(
                    expanded_batch, self._rollout_corr_config
                )

            # Hand off to the trajectory-uniform actor.
            actor_output = original_update_actor(expanded_batch)

            # Forward the helper's rollout_corr/* metrics (clipfrac, KL,
            # off-policy gap, etc.) into the actor's metric stream so they
            # land on wandb alongside actor/pg_loss.
            if tis_metrics:
                meta = actor_output.meta_info or {}
                existing = dict(meta.get("metrics", {}) or {})
                existing.update(tis_metrics)
                meta["metrics"] = existing
                actor_output.meta_info = meta

            # Clear the cache so a future stale call (if anything) raises.
            self._cached_raw_trajectories = None
            return actor_output

        self.actor_rollout_wg.update_actor = _expanded_update_actor

    def _attach_traj_uniform_weight_inplace(self, batch: DataProto) -> DataProto:
        """Attach a `traj_uniform_weight` tensor for an unexpanded batch.

        When the segment-training path is enabled but the rollout produced no
        multi-segment trajectories (every traj is 1 segment), we still need
        the actor's ``traj_uniform_weight`` field to be present, with values
        such that the loss math comes out as
        ``L = (1/N_G) Σ_i (1/N_t^i) Σ_token ℓ_token``. For 1-segment
        trajectories, ``N_t^i`` = the row's mask=1 count, so
        ``w[i, t] = 1 / (N_G · sum(response_mask[i]))`` for valid tokens.
        """
        import torch as _torch

        response_mask = batch.batch["response_mask"]
        n_g = response_mask.shape[0]
        n_t_per_row = response_mask.sum(dim=1).clamp(min=1).to(_torch.float32)
        weight_per_row = 1.0 / (n_g * n_t_per_row)
        # Broadcast to (B, T): weight_per_row[i] at mask=1 positions.
        traj_uniform_weight = response_mask.to(_torch.float32) * weight_per_row.unsqueeze(1)
        batch.batch["traj_uniform_weight"] = traj_uniform_weight
        return batch

    def _dump_tis_debug(self, expanded_batch: DataProto) -> None:
        """Print vLLM↔FSDP logp diagnostics: per-row aggregates + worst rows.

        Aggregates mean/max |diff| across all valid (mask=1) positions in
        every segment row, then prints the 3 worst rows in full plus row 0
        for context. A healthy run shows mean |diff| < 0.02 nats and worst
        rows clustered near the mean; a sustained large gap on specific
        rows points to per-trajectory issues (summary vs reflection vs
        non-summary segments behaving differently).
        """
        import torch as _torch
        rlp = expanded_batch.batch.get("rollout_log_probs")
        olp = expanded_batch.batch.get("old_log_probs")
        if rlp is None or olp is None:
            return
        mask = expanded_batch.batch["response_mask"].to(rlp.device)
        resp = expanded_batch.batch["responses"]
        prompt = expanded_batch.batch["prompts"]
        pad_id = self.tokenizer.pad_token_id

        diff = (rlp - olp) * mask  # mask=0 positions zero-out, harmless
        valid_counts = mask.sum(dim=-1).clamp(min=1)
        row_mean = diff.sum(dim=-1) / valid_counts  # signed mean per row
        row_abs_mean = diff.abs().sum(dim=-1) / valid_counts  # |diff| mean
        row_max = diff.abs().masked_fill(mask == 0, 0).max(dim=-1).values

        batch_mean_diff = (diff.sum() / mask.sum().clamp(min=1)).item()
        batch_abs_mean = (diff.abs().sum() / mask.sum().clamp(min=1)).item()
        n_rows = rlp.shape[0]
        n_valid = int(mask.sum().item())

        print("\n[ORBIT_TIS_DEBUG] batch summary")
        print(f"  rows={n_rows}  total_mask1_tokens={n_valid}")
        print(f"  signed mean(rollout - old) over all mask=1 = "
              f"{batch_mean_diff:+.5f}")
        print(f"  mean |diff|                                 = "
              f"{batch_abs_mean:.5f}")
        print(f"  per-row signed mean: min={row_mean.min().item():+.4f}  "
              f"max={row_mean.max().item():+.4f}  median="
              f"{row_mean.median().item():+.4f}")

        def _print_row(label: str, row: int, n_head: int = 5, n_tail: int = 5):
            valid_pos = mask[row].nonzero(as_tuple=True)[0]
            if valid_pos.numel() == 0:
                print(f"  [{label}] row {row}: no mask=1 positions")
                return
            n_valid_row = valid_pos.numel()
            print(f"\n  [{label}] row {row}: n_valid={n_valid_row}  "
                  f"signed_mean={row_mean[row].item():+.4f}  "
                  f"|diff|_mean={row_abs_mean[row].item():.4f}  "
                  f"|diff|_max={row_max[row].item():.4f}")
            head = valid_pos[:n_head].tolist()
            tail = valid_pos[-n_tail:].tolist() if n_valid_row > n_head else []
            ranges = [(head, "head")]
            if tail and tail[0] not in head:
                ranges.append((tail, "tail"))
            print(f"  {'pos':>5} | {'tok_id':>6} | {'rollout_lp':>11} | "
                  f"{'old_lp':>11} | {'diff':>10}")
            for positions, _ in ranges:
                for p in positions:
                    tid = int(resp[row, p].item())
                    rl = float(rlp[row, p].item())
                    ol = float(olp[row, p].item())
                    print(f"  {p:>5d} | {tid:>6d} | {rl:>+11.5f} | "
                          f"{ol:>+11.5f} | {rl - ol:>+10.5f}")
            # Decoded prompt tail (chat template boundary inspection).
            prompt_tokens = prompt[row].tolist()
            first_real = next(
                (i for i, t in enumerate(prompt_tokens) if t != pad_id),
                len(prompt_tokens),
            )
            tail_n = min(24, len(prompt_tokens) - first_real)
            ptail = prompt_tokens[len(prompt_tokens) - tail_n:]
            try:
                decoded = self.tokenizer.decode(ptail)
                print(f"  prompt tail: {decoded!r}")
            except Exception as e:
                print(f"  prompt tail decode failed: {e}")

        _print_row("ROW_0", 0)
        # Worst 3 rows by |diff| mean.
        worst = row_abs_mean.topk(min(3, n_rows)).indices.tolist()
        for k, r in enumerate(worst):
            _print_row(f"WORST_{k+1}", r)

    def _validate_agent(self):
        """Override validation to include environment metrics from MultiEpisodeEnv.

        If val_env_class is set, uses that environment class for validation
        instead of the training env_class.
        """
        # Enable validation mode for init_envs_and_agents
        self._is_validation_mode = True
        # Reset the per-trajectory engine-metric buffer; it is populated
        # during _transform_agent_trajectories calls made below.
        self._val_traj_metrics_buffer = []

        rewards_lst = []
        data_source_lst = []
        uid_lst = []
        env_metrics_lst = []  # Collect environment metrics
        env_data_sources_lst = []  # Collect data_sources for each environment
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)
            n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            test_batch.pop(["input_ids", "attention_mask", "position_ids"])  # these are not needed for environment based interaction
            test_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": False,
                "validate": True,
            }
            # Get data_source after repeat (so it matches env order after init_envs_and_agents)
            batch_data_sources = test_batch.non_tensor_batch.get("data_source", ["unknown"] * len(test_batch.batch))
            
            self.init_envs_and_agents(test_batch)

            if self.config.rllm.stepwise_advantage.enable:
                test_output_gen_batch = self.generate_agent_steps(meta_info=test_batch.meta_info, uids=test_batch.non_tensor_batch["uid"])
                # for validation, we only need the last step
                is_last_step = test_output_gen_batch.non_tensor_batch["is_last_step"]
                last_step_indices = np.where(is_last_step == True)[0]
                test_output_gen_batch = test_output_gen_batch.select_idxs(last_step_indices)  # This batch only has last steps
            else:
                test_output_gen_batch, _ = self.generate_agent_trajectory(meta_info=test_batch.meta_info)

            test_batch = test_batch.union(test_output_gen_batch)

            reward_tensor = test_batch.batch["token_level_scores"]

            rewards_lst.append(reward_tensor.sum(-1).cpu())
            data_source_lst.append(batch_data_sources)
            uid_lst.append(test_batch.non_tensor_batch["uid"])

            # Collect environment metrics if available, grouped by data_source
            if hasattr(self.agent_execution_engine, "envs") and self.agent_execution_engine.envs:
                batch_env_metrics = []
                batch_env_data_sources = []
                for idx, env in enumerate(self.agent_execution_engine.envs):
                    if hasattr(env, "get_metrics") and callable(env.get_metrics):
                        try:
                            env_metrics = env.get_metrics()
                            if isinstance(env_metrics, dict):
                                # Attach task_type from env property (not in metrics dict)
                                tt = getattr(env, "task_type", None)
                                if tt is not None:
                                    env_metrics["_task_type"] = tt
                                batch_env_metrics.append(env_metrics)
                                # Get data_source for this environment (should match batch order)
                                if idx < len(batch_data_sources):
                                    batch_env_data_sources.append(batch_data_sources[idx])
                                else:
                                    batch_env_data_sources.append("unknown")
                        except Exception:
                            pass  # Silently ignore if metrics extraction fails
                env_metrics_lst.append(batch_env_metrics)
                env_data_sources_lst.append(batch_env_data_sources)

        reward_tensor = torch.cat(rewards_lst, dim=0)  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}

        # to group for pass@k
        uid_tensor = np.concatenate(uid_lst, axis=0)
        data_source_uid_pass_rates = {}  # data source to {uid: pass or not}

        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]

            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

            # pass@k
            if data_source not in data_source_uid_pass_rates:
                data_source_uid_pass_rates[data_source] = {}

            uid = uid_tensor[i]
            if uid not in data_source_uid_pass_rates[data_source]:
                data_source_uid_pass_rates[data_source][uid] = 0  # default to not pass
            # take highest score
            data_source_uid_pass_rates[data_source][uid] = max(data_source_uid_pass_rates[data_source][uid], reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            # clip rewards to be between 0 and 1
            rewards_array = np.array(rewards)
            rewards_array = np.clip(rewards_array, 0, 1)
            metric_dict[f"val/test_score/{data_source}"] = np.mean(rewards_array)

        for data_source, pass_rates in data_source_uid_pass_rates.items():
            pass_k_lst = []
            for uid, pass_score in pass_rates.items():
                pass_k_lst.append(pass_score >= 1)  # assuming 1 means passed
            metric_dict[f"val/test_score/pass@k/{data_source}"] = np.mean(pass_k_lst)

        # Aggregate environment metrics if available, grouped by data_source
        if env_metrics_lst:
            # Flatten list of lists and pair with data_sources
            all_env_metrics_with_sources = []
            for batch_idx, batch_metrics in enumerate(env_metrics_lst):
                batch_data_sources = env_data_sources_lst[batch_idx] if batch_idx < len(env_data_sources_lst) else ["unknown"] * len(batch_metrics)
                for env_idx, metrics in enumerate(batch_metrics):
                    data_source = batch_data_sources[env_idx] if env_idx < len(batch_data_sources) else "unknown"
                    all_env_metrics_with_sources.append((data_source, metrics))
            
            if all_env_metrics_with_sources:
                # Group metrics by data_source
                data_source_metrics = {}
                for data_source, metrics in all_env_metrics_with_sources:
                    if data_source not in data_source_metrics:
                        data_source_metrics[data_source] = []
                    data_source_metrics[data_source].append(metrics)

                def _aggregate_numeric_metrics(metrics_list, prefix):
                    """Aggregate numeric metrics, filtering -1 placeholders and string values."""
                    all_keys = set()
                    for metrics in metrics_list:
                        all_keys.update(metrics.keys())
                    for key in all_keys:
                        if key.startswith("_"):
                            continue  # Skip internal keys (_task_type, etc.)
                        values = []
                        for metrics in metrics_list:
                            if key in metrics:
                                value = metrics[key]
                                if isinstance(value, (int, float)) and value >= 0:
                                    values.append(float(value))
                        if values:
                            metric_dict[f"{prefix}/{key}"] = np.mean(values)

                # Aggregate metrics per data_source (overall + per task_type)
                for data_source, metrics_list in data_source_metrics.items():
                    # Overall aggregation for this data_source
                    _aggregate_numeric_metrics(metrics_list, f"val/{data_source}")

                    # Sub-group by task_type if present
                    task_type_groups = {}
                    for metrics in metrics_list:
                        tt = metrics.get("_task_type")
                        if tt is not None:
                            task_type_groups.setdefault(tt, []).append(metrics)
                    for task_type, tt_metrics_list in task_type_groups.items():
                        _aggregate_numeric_metrics(tt_metrics_list, f"val/{data_source}/{task_type}")

        # Per-trajectory engine metrics (summarization_count, segment_count, …)
        # buffered during _transform_agent_trajectories when _is_validation_mode
        # was True. We surface them here so the val dashboard shows, per task
        # variant AND overall, how many summarizations each rollout fired.
        if self._val_traj_metrics_buffer:
            TRAJ_METRIC_KEYS = ("summarization_count", "segment_count")

            by_source: dict[str, list[dict]] = {}
            for ds, m in self._val_traj_metrics_buffer:
                by_source.setdefault(ds, []).append(m)

            overall_by_key: dict[str, list[float]] = {}
            for ds, metrics_list in by_source.items():
                for key in TRAJ_METRIC_KEYS:
                    values = [
                        float(m[key])
                        for m in metrics_list
                        if key in m and isinstance(m[key], (int, float))
                    ]
                    if not values:
                        continue
                    arr = np.array(values)
                    metric_dict[f"val/{ds}/{key}/mean"] = float(arr.mean())
                    metric_dict[f"val/{ds}/{key}/sum"] = float(arr.sum())
                    metric_dict[f"val/{ds}/{key}/min"] = float(arr.min())
                    metric_dict[f"val/{ds}/{key}/max"] = float(arr.max())
                    overall_by_key.setdefault(key, []).extend(values)

            for key, values in overall_by_key.items():
                arr = np.array(values)
                metric_dict[f"val/all/{key}/mean"] = float(arr.mean())
                metric_dict[f"val/all/{key}/sum"] = float(arr.sum())
                metric_dict[f"val/all/{key}/min"] = float(arr.min())
                metric_dict[f"val/all/{key}/max"] = float(arr.max())
                metric_dict[f"val/all/{key}/count_trajectories"] = len(values)

        # Disable validation mode
        self._is_validation_mode = False

        return metric_dict

    def _transform_agent_trajectories(self, trajectories: list[dict]):
        """Override to group traj metrics by data_source.

        We keep ``traj["segments"]`` intact (the parent transform doesn't
        consult it; it uses the top-level ``prompt_tokens`` /
        ``response_tokens`` / ``response_masks``, which equal ``segments[0]``
        for any segmented trajectory) and stash the trajectory list on
        ``self._cached_raw_trajectories`` so the wrapped ``update_actor``
        (when summarization is enabled) can expand into per-segment rows
        with broadcast advantages and trajectory-uniform per-token weights.

        Note: truncated-trajectory filtering happens inside
        ``_expanded_update_actor`` (not here), because the upstream prompt
        batch and the transformed batch must keep matching row counts so
        ``fit_agent``'s ``batch.union(final_gen_batch_output)`` can fire.
        """
        # Cache the segmented payload for the wrapped update_actor.
        # We use a shallow copy so future mutations to `trajectories` don't
        # leak into the cache.
        self._cached_raw_trajectories = list(trajectories)

        # Call parent method to get the base transformation
        final_gen_batch_output, metrics = super()._transform_agent_trajectories(trajectories)

        # Group traj metrics by data_source
        # Trajectories have an 'idx' field that corresponds to the environment index
        # We can map this to data_source using the stored batch data_sources
        if hasattr(self, "_current_batch_data_sources") and trajectories:
            batch_data_sources = self._current_batch_data_sources
            
            # Group trajectories by data_source based on their idx
            traj_metrics_by_source = {}
            for traj in trajectories:
                traj_idx = traj.get("idx")
                if traj_idx is not None and traj_idx < len(batch_data_sources):
                    data_source = batch_data_sources[traj_idx]
                else:
                    data_source = "unknown"

                if data_source not in traj_metrics_by_source:
                    traj_metrics_by_source[data_source] = []

                traj_metrics = traj.get("metrics", {})
                if traj_metrics:
                    traj_metrics_by_source[data_source].append(traj_metrics)
                    # During validation, accumulate raw per-trajectory metrics
                    # so _validate_agent can aggregate them across all val
                    # batches under val/{data_source}/<k>/<stat>.
                    if self._is_validation_mode:
                        self._val_traj_metrics_buffer.append(
                            (str(data_source), dict(traj_metrics))
                        )
            
            # Aggregate metrics per data_source
            for data_source, metrics_list in traj_metrics_by_source.items():
                if not metrics_list:
                    continue
                
                # Collect all metric keys for this data_source
                all_keys = set()
                for m in metrics_list:
                    all_keys.update(m.keys())
                
                # Aggregate each metric (mean, min, max) per data_source
                for k in all_keys:
                    v_list = [m.get(k) for m in metrics_list if k in m]
                    v_list = [v for v in v_list if v is not None and v >= 0]
                    if not v_list:
                        continue
                    v_list = np.array(v_list)
                    metrics.update(
                        {
                            f"traj/{data_source}/{k}_mean": v_list.mean(),
                            f"traj/{data_source}/{k}_min": v_list.min(),
                            f"traj/{data_source}/{k}_max": v_list.max(),
                        }
                    )
        
        return final_gen_batch_output, metrics

    def init_envs_and_agents(self, batch):
        """Override to track data_source and support validation env class.

        When _is_validation_mode is True and val_env_class is set, temporarily
        swaps the env_class and env_args to use the validation-specific ones.
        """
        # Store data_source before calling parent (in case batch gets modified)
        if hasattr(batch, "non_tensor_batch"):
            batch_data_sources = batch.non_tensor_batch.get("data_source")
            if batch_data_sources is not None:
                self._current_batch_data_sources = batch_data_sources

        # If in validation mode and val_env_class is set, swap env class temporarily
        if self._is_validation_mode and self.val_env_class is not None:
            original_env_class = self.env_class
            original_env_args = self.env_args
            self.env_class = self.val_env_class
            if self.val_env_args is not None:
                self.env_args = self.val_env_args
            try:
                return super().init_envs_and_agents(batch)
            finally:
                # Restore original env class and args
                self.env_class = original_env_class
                self.env_args = original_env_args
        else:
            return super().init_envs_and_agents(batch)

