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

        # Alternating actor/summarizer training.
        # When enabled, training alternates every T steps between:
        #   "summarizer" — actor frozen (ref-routing), only summary/reflection trained
        #   "actor"      — summarizer frozen (mask=0 on summary tokens), only actions trained
        alt_cfg = OmegaConf.select(
            self.config, "rllm.agent.summarization.alternating", default=None
        )
        self._alt_enabled: bool = bool(
            alt_cfg is not None and alt_cfg.get("enable", False)
        )
        self._phase_T: int | None = (
            int(alt_cfg.get("T", 100)) if self._alt_enabled else None
        )
        self._training_phase: str = (
            str(alt_cfg.get("initial_phase", "summarizer"))
            if self._alt_enabled
            else "summarizer"
        )
        self._phase_steps_done: int = 0
        # Option B: checkpoint swap + ref vLLM restart at each phase boundary.
        self._alt_actor_ckpt: str | None = (
            str(alt_cfg.get("actor_ckpt_dir", "checkpoints/alternating/actor"))
            if self._alt_enabled else None
        )
        self._alt_summ_ckpt: str | None = (
            str(alt_cfg.get("summarizer_ckpt_dir", "checkpoints/alternating/summarizer"))
            if self._alt_enabled else None
        )
        self._alt_ref_vllm_cmd: str | None = (
            str(alt_cfg.get(
                "ref_vllm_cmd",
                "vllm serve {ckpt} --served-model-name ref-model --port 8765 "
                "--gpu-memory-utilization 0.5 --max-model-len 32768 "
                "--dtype bfloat16 --enforce-eager"
            )) if self._alt_enabled else None
        )
        self._alt_ref_startup_timeout: int = (
            int(alt_cfg.get("ref_startup_timeout", 300)) if self._alt_enabled else 300
        )
        # Extract ref URL from the vllm_cmd (default to port 8765).
        self._alt_ref_url: str = "http://localhost:8765"
        if self._alt_enabled and self._alt_ref_vllm_cmd:
            import re
            m = re.search(r"--port\s+(\d+)", self._alt_ref_vllm_cmd)
            if m:
                self._alt_ref_url = f"http://localhost:{m.group(1)}"
        self._alt_ref_proc = None  # subprocess.Popen handle for the ref vLLM
        if self._alt_enabled:
            print(
                f"[alternating] enabled: T={self._phase_T}, "
                f"initial_phase={self._training_phase}, "
                f"actor_ckpt={self._alt_actor_ckpt}, "
                f"summ_ckpt={self._alt_summ_ckpt}"
            )

    # ------------------------------------------------------------------
    # Two-model alternating training helpers (Option B — checkpoint swap)
    # ------------------------------------------------------------------

    def _phase_ckpt_dir(self, phase: str) -> str:
        """Return the checkpoint directory for the given phase."""
        return self._alt_actor_ckpt if phase == "actor" else self._alt_summ_ckpt

    def _save_phase_checkpoint(self, phase: str) -> None:
        """Save the current FSDP model to the checkpoint directory for ``phase``."""
        import ray
        ckpt_dir = self._phase_ckpt_dir(phase)
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"[alternating] saving {phase} checkpoint → {ckpt_dir}")
        t0 = __import__("time").time()
        ray.get(
            self.actor_rollout_wg.save_checkpoint.remote(
                ckpt_dir, None, getattr(self, "global_steps", 0)
            )
        )
        print(f"[alternating] checkpoint saved in {__import__('time').time()-t0:.1f}s")

    def _load_phase_checkpoint(self, phase: str) -> None:
        """Load the checkpoint for ``phase`` into the FSDP slot (no-op if absent)."""
        import ray
        ckpt_dir = self._phase_ckpt_dir(phase)
        if not os.path.isdir(ckpt_dir):
            print(
                f"[alternating] no {phase} checkpoint yet at {ckpt_dir}, "
                "continuing with current FSDP weights"
            )
            return
        print(f"[alternating] loading {phase} checkpoint ← {ckpt_dir}")
        t0 = __import__("time").time()
        ray.get(
            self.actor_rollout_wg.load_checkpoint.remote(ckpt_dir)
        )
        print(f"[alternating] checkpoint loaded in {__import__('time').time()-t0:.1f}s")

    def _restart_ref_vllm(self, ckpt_path: str) -> None:
        """Kill the current ref vLLM process and restart it with ``ckpt_path`` weights.

        Substitutes ``{ckpt}`` in ``self._alt_ref_vllm_cmd`` with ``ckpt_path``,
        launches via subprocess, then polls ``GET /v1/models`` until the server
        responds or ``ref_startup_timeout`` seconds elapse.
        """
        import subprocess, time, urllib.request

        # Kill any previously tracked process.
        if self._alt_ref_proc is not None:
            try:
                self._alt_ref_proc.terminate()
                self._alt_ref_proc.wait(timeout=30)
            except Exception as e:
                print(f"[alternating] warning: ref vLLM termination: {e}")
            self._alt_ref_proc = None

        cmd = self._alt_ref_vllm_cmd.format(ckpt=ckpt_path)
        print(f"[alternating] starting ref vLLM: {cmd}")
        self._alt_ref_proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        # Wait until health endpoint responds.
        health_url = f"{self._alt_ref_url}/v1/models"
        deadline = time.time() + self._alt_ref_startup_timeout
        last_exc = None
        while time.time() < deadline:
            try:
                urllib.request.urlopen(health_url, timeout=3)
                print("[alternating] ref vLLM ready")
                return
            except Exception as exc:
                last_exc = exc
                time.sleep(5)
        raise RuntimeError(
            f"[alternating] ref vLLM at {health_url} did not become ready within "
            f"{self._alt_ref_startup_timeout}s. Last error: {last_exc}"
        )

    def _do_phase_switch(self) -> None:
        """Execute a full phase switch: save → restart ref vLLM → load → update engine."""
        outgoing = self._training_phase
        incoming = "actor" if outgoing == "summarizer" else "summarizer"
        print(f"[alternating] phase switch: {outgoing} → {incoming}")
        t_start = __import__("time").time()

        # 1. Save the outgoing model so it can serve as the frozen ref.
        self._save_phase_checkpoint(outgoing)

        # 2. Restart the ref vLLM with the outgoing model's weights.
        self._restart_ref_vllm(self._phase_ckpt_dir(outgoing))

        # 3. Invalidate the ref engine cache so the engine reconnects.
        if hasattr(self, "agent_execution_engine") and hasattr(
            self.agent_execution_engine, "_ref_engine_cache"
        ):
            del self.agent_execution_engine._ref_engine_cache

        # 4. Load the incoming model's latest checkpoint (weights + optimizer).
        #    The training vLLM will pick up the new weights via verl's normal
        #    wake + update_weights cycle at the start of the next rollout batch.
        self._load_phase_checkpoint(incoming)

        self._training_phase = incoming
        self._phase_steps_done = 0
        print(
            f"[alternating] switch complete in "
            f"{__import__('time').time() - t_start:.1f}s, now in {incoming} phase"
        )

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

            # Hand off to the trajectory-uniform actor.
            actor_output = original_update_actor(expanded_batch)

            # Alternating training: count completed steps and emit phase metrics.
            if self._alt_enabled:
                self._phase_steps_done += 1
                meta = actor_output.meta_info or {}
                existing = dict(meta.get("metrics", {}) or {})
                existing["training/phase"] = (
                    0.0 if self._training_phase == "summarizer" else 1.0
                )
                existing["training/phase_steps_done"] = float(self._phase_steps_done)
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
        """Override to track data_source, support validation env class, and
        advance the alternating-training phase before each new batch.

        When _is_validation_mode is True and val_env_class is set, temporarily
        swaps the env_class and env_args to use the validation-specific ones.
        """
        # Alternating phase management: check for a phase switch BEFORE the
        # rollout so the engine uses the correct phase during trajectory
        # collection. Phase switches only happen during training, not validation.
        if self._alt_enabled and not self._is_validation_mode:
            if self._phase_steps_done >= self._phase_T:
                self._do_phase_switch()  # save → restart ref vLLM → load → flip phase
            # Push current phase to the engine so trajectory collection uses it.
            if hasattr(self, "agent_execution_engine"):
                self.agent_execution_engine.training_phase = self._training_phase

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

