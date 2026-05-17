#!/bin/bash
set -x

# =============================================================================
# Train an LLM on the maze env (multi-episode meta-RL with summarization).
#   - oracle (rule-based) episodic summarizer enabled by default
#   - selectable advantage method (default: chunk-discounted-TOPR; flip to
#     GRPO with ADVANTAGE_METHOD=grpo)
#   - reward_scope="per_episode" by default (TOPR only — ignored under GRPO)
#
# Layered on top of scripts/train_multi_task_summarizing.sh, which itself
# invokes scripts/train_multi_task_multi_episode.sh (the script that writes
# the trajectory-uniform actor .pth file into the active conda env's
# site-packages).
#
# Activate the icx conda env first.
#
# Usage:
#   conda activate icx
#   bash scripts/train_maze.sh
#
# Env-var overrides:
#   TASKS_CONFIG             yaml config (default: configs/maze.yaml)
#   MODEL_PATH               model (default: Qwen/Qwen3-1.7B)
#   ADVANTAGE_METHOD         "chunk_discounted_topr" (default) or "grpo".
#                            Under "grpo" the GAMMA / topr_split / reward_scope
#                            overrides are skipped — GRPO normalizes per-group
#                            advantages without per-chunk discounting.
#   GAMMA                    chunk discount factor (default: 0.95)
#                            [chunk_discounted_topr only]
#   ORACLE                   "true" to enable the rule-based maze summarizer
#                            (default: true). Set to "false" when benchmarking
#                            $CARRYOVER modes — oracle wins regardless of
#                            episodic_carryover and would collapse all three
#                            modes to the same behavior.
#   CARRYOVER                episode-end carryover form (forwarded to
#                            train_multi_task_summarizing.sh). One of:
#                            freeform | obs_action | obs_action_reflection
#                            (default: obs_action_reflection).
#   EXP_NAME                 wandb experiment_name. Default:
#                              "<model>_${CARRYOVER}_${MODE}" when ORACLE=false,
#                              "<model>_oracle_${MODE}"      when ORACLE=true.
#                              <model> is basename($MODEL_PATH).
#   N_GPUS                   number of GPUs (default: 2). When set, takes
#                            precedence over CUDA_VISIBLE_DEVICES — the script
#                            picks 0..N_GPUS-1.
#   CUDA_VISIBLE_DEVICES     specific GPU ids (e.g. "2,3,5,7"). Used only when
#                            N_GPUS is unset; N_GPUS is then derived from the
#                            count of comma-separated ids.
#
#   Speed knobs (sensible defaults; bump when you have headroom):
#     TRAIN_BATCH            data.train_batch_size            (default: 32)
#     ROLLOUT_N              actor_rollout_ref.rollout.n      (default: 8;
#                            this is the GRPO group size — total rollouts per
#                            optim step = TRAIN_BATCH × ROLLOUT_N)
#     MINI_BATCH             ppo_mini_batch_size              (default: 64)
#     MAX_TOKEN_LEN_PER_GPU  ppo_max_token_len_per_gpu        (default: 32768;
#                            used when use_dynamic_bsz=True — bigger fits more
#                            tokens in a micro-batch up to GPU memory)
#     MICRO_BATCH            ppo_micro_batch_size_per_gpu     (default: unset;
#                            only honored when USE_DYNAMIC_BSZ=False)
#     USE_DYNAMIC_BSZ        actor.use_dynamic_bsz            (default: True)
#     GPU_MEM_UTIL           rollout.gpu_memory_utilization   (default: 0.8)
#
#   Truncation filter (drop truncated rollouts from the actor update; OFF
#   by default — preserves existing behavior). When ON, trajectories whose
#   engine termination_reason is in
#   {TRUNCATION, PROMPT_TRUNCATION, SUMMARIZATION_BUDGET_EXCEEDED,
#    SUMMARIZATION_FAILED} are removed before advantage computation, so
#   GRPO normalization and the trajectory-uniform N_G see only kept
#   rollouts. MAX_STEPS-terminated trajectories (step-budget exhausted
#   without max-token truncation) are NOT filtered. If a step would drop
#   every trajectory, the filter falls back to the unfiltered batch
#   (logged) to avoid an empty-batch crash.
#     FILTER_TRUNCATED        "true" to enable (default: false)
#
#   Length-penalty knobs (overlong reward shaping; OFF by default):
#     LENGTH_PENALTY         "true" to enable per-episode length penalty
#                            (default: false)
#     RESPONSE_LENGTH        reference for data.max_response_length used to
#                            derive L_cache from a fraction (default: 31744 —
#                            matches the parent script). Override if you also
#                            override data.max_response_length.
#     LP_MAX_TOKENS          episode_max_tokens (L_max). Default: unset →
#                            engine uses data.max_response_length.
#     LP_CACHE_TOKENS        episode_cache_tokens (L_cache), absolute.
#                            Default: unset → see LP_CACHE_FRACTION.
#     LP_CACHE_FRACTION      L_cache as a fraction of LP_MAX_TOKENS (if set,
#                            relative to it) else of RESPONSE_LENGTH
#                            (default: 0.25). Set 0.5 → cache = half the
#                            response length. Ignored when LP_CACHE_TOKENS
#                            is set explicitly.
# =============================================================================
# export RAY_object_store_memory=$((50 * 1024 * 1024 * 1024))
export TASKS_CONFIG=${TASKS_CONFIG:-configs/maze.yaml}
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8b}

# ---- Advantage method --------------------------------------------------
# Export the env vars rather than building Hydra overrides locally — the
# summarizing layer (train_multi_task_summarizing.sh) reads these and
# constructs the +rllm.advantage_method.* overrides once. This keeps a
# single source of truth and avoids duplicate `+key=val` Hydra errors
# when env vars propagate through layered shell calls.
#
# Defaults here are TOPR-first (this script's intended identity), but
# users can flip to GRPO via ADVANTAGE_METHOD=grpo.
export ADVANTAGE_METHOD=${ADVANTAGE_METHOD:-chunk_discounted_topr}
export GAMMA=${GAMMA:-0.95}
export REWARD_SCOPE=${REWARD_SCOPE:-per_episode}
export TOPR_SPLIT=${TOPR_SPLIT:-true}
case "$ADVANTAGE_METHOD" in
    chunk_discounted_topr|grpo) ;;
    *)
        echo "Error: ADVANTAGE_METHOD must be chunk_discounted_topr or grpo (got '$ADVANTAGE_METHOD')" >&2
        exit 1
        ;;
esac
echo "[wrapper] ADVANTAGE_METHOD=$ADVANTAGE_METHOD"

# ---- GPU assignment ------------------------------------------------------
# Precedence: explicit N_GPUS > derive from CUDA_VISIBLE_DEVICES > default 2.
# If N_GPUS is set but CUDA_VISIBLE_DEVICES is not, pin to 0..N_GPUS-1 so
# Ray + vLLM see a contiguous device range.
if [ -n "${N_GPUS:-}" ]; then
    if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
        _ids=$(seq -s, 0 $((N_GPUS - 1)))
        export CUDA_VISIBLE_DEVICES=$_ids
    fi
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    N_GPUS=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
else
    N_GPUS=2
    export CUDA_VISIBLE_DEVICES=0,1
fi
export N_GPUS
echo "[wrapper] N_GPUS=$N_GPUS CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"


# ---- Speed knobs ---------------------------------------------------------
TRAIN_BATCH=${TRAIN_BATCH:-32}
ROLLOUT_N=${ROLLOUT_N:-8}
MINI_BATCH=${MINI_BATCH:-128}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-32768}
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-True}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.8}

# MICRO_BATCH is only honored when dynamic batching is OFF; pass it through
# only if both conditions are met to avoid Hydra warnings about an unused key.
SPEED_OVERRIDES=(
    data.train_batch_size=$TRAIN_BATCH
    actor_rollout_ref.rollout.n=$ROLLOUT_N
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL
)
if [ -n "${MICRO_BATCH:-}" ] && [ "$USE_DYNAMIC_BSZ" = "False" ]; then
    SPEED_OVERRIDES+=(actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH)
fi

# ---- Truncation filter ---------------------------------------------------
FILTER_TRUNCATED_OVERRIDES=()
case "${FILTER_TRUNCATED:-false}" in
    true|True)
        FILTER_TRUNCATED_OVERRIDES+=(+rllm.filter_truncated_trajectories.enable=true)
        echo "[wrapper] truncation filter ON: trajectories with TRUNCATION/PROMPT_TRUNCATION/SUMMARIZATION_* are dropped before the actor update"
        ;;
    false|False) ;;
    *)
        echo "Error: FILTER_TRUNCATED must be true or false (got '$FILTER_TRUNCATED')" >&2
        exit 1
        ;;
esac

# ---- Length penalty (overlong reward shaping; off by default) ------------
RESPONSE_LENGTH=${RESPONSE_LENGTH:-31744}
LP_CACHE_FRACTION=${LP_CACHE_FRACTION:-0.25}
LP_OVERRIDES=()
if [ "${LENGTH_PENALTY:-false}" = "true" ] || [ "${LENGTH_PENALTY:-false}" = "True" ]; then
    LP_OVERRIDES+=(+rllm.length_penalty.enable=true)
    if [ -n "${LP_MAX_TOKENS:-}" ]; then
        LP_OVERRIDES+=(+rllm.length_penalty.episode_max_tokens=$LP_MAX_TOKENS)
        _LP_BASE=$LP_MAX_TOKENS
    else
        _LP_BASE=$RESPONSE_LENGTH
    fi
    if [ -z "${LP_CACHE_TOKENS:-}" ]; then
        LP_CACHE_TOKENS=$(awk "BEGIN{printf \"%d\", $_LP_BASE * $LP_CACHE_FRACTION}")
    fi
    LP_OVERRIDES+=(+rllm.length_penalty.episode_cache_tokens=$LP_CACHE_TOKENS)
    echo "[wrapper] length penalty ON: L_max=${LP_MAX_TOKENS:-<engine default = data.max_response_length>} L_cache=$LP_CACHE_TOKENS"
fi

# Episodic summary fits the user's "summarize at episode end" intent;
# threshold is set high so token-trigger never fires (oracle is the only
# summary source we want here).
export MODE=${MODE:-both}
export SUMMARIZATION_THRESHOLD=${SUMMARIZATION_THRESHOLD:-16384}
export SUMMARY_MAX_TOKENS=${SUMMARY_MAX_TOKENS:-4096}

ORACLE=${ORACLE:-true}
ORACLE_OVERRIDES=()
case "$ORACLE" in
    true|True)
        ORACLE_OVERRIDES+=(
            +rllm.agent.summarization.oracle.enable=true
            +rllm.agent.summarization.oracle.scope=maze
        )
        ;;
    false|False) ;;
    *)
        echo "Error: ORACLE must be true or false (got '$ORACLE')" >&2
        exit 1
        ;;
esac

# Build the wandb experiment_name from model + carryover + summarization
# mode so benchmarking runs are visually grouped. When oracle is on, the
# carryover mode is overridden by the oracle path, so the name reflects
# that instead. User can still override with EXP_NAME=...
export CARRYOVER=${CARRYOVER:-obs_action_reflection}
# Use the trailing path component of MODEL_PATH (e.g. "Qwen/Qwen3-8B" → "Qwen3-8B",
# "/path/to/ckpt/global_step_100" → "global_step_100").
_MODEL_TAG=$(basename "$MODEL_PATH")
# Append the advantage-method tag so TOPR / GRPO runs are distinguishable
# in wandb. "topr" is shorter than "chunk_discounted_topr".
case "$ADVANTAGE_METHOD" in
    chunk_discounted_topr) _ADV_TAG="topr" ;;
    grpo)                  _ADV_TAG="grpo" ;;
    *)                     _ADV_TAG="$ADVANTAGE_METHOD" ;;
esac
case "$ORACLE" in
    true|True)  _EXP_LABEL="${_MODEL_TAG}_oracle_${MODE}_${_ADV_TAG}" ;;
    *)          _EXP_LABEL="${_MODEL_TAG}_${CARRYOVER}_${MODE}_${_ADV_TAG}" ;;
esac
export EXP_NAME=${EXP_NAME:-$_EXP_LABEL}

bash scripts/train_multi_task_summarizing.sh \
    "${ORACLE_OVERRIDES[@]}" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.project_name='COMET' \
    "${SPEED_OVERRIDES[@]}" \
    "${LP_OVERRIDES[@]}" \
    "${FILTER_TRUNCATED_OVERRIDES[@]}" \
    "$@"
