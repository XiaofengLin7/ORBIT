#!/bin/bash
set -x

# =============================================================================
# Train Qwen3-1.7B on the maze env with:
#   - oracle (rule-based) episodic summarizer
#   - chunk-discounted-TOPR advantage method (sibling to GRPO)
#   - reward_scope="terminal" (variant A: R_total = sum of episode rewards;
#                              all trainable chunks share R discounted by γ^Δ
#                              from the trajectory's last trainable chunk;
#                              the discount does NOT reset across episodes)
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
#   bash scripts/train_maze_chunked_topr_terminal.sh
#
# Env-var overrides:
#   TASKS_CONFIG             yaml config (default: configs/eval_maze_oracle_10ep.yaml)
#   MODEL_PATH               model (default: Qwen/Qwen3-1.7B)
#   GAMMA                    chunk discount factor (default: 0.95)
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
#   Length-penalty knobs (overlong reward shaping; OFF by default):
#     LENGTH_PENALTY         "true" to enable per-episode length penalty
#                            (default: false)
#     RESPONSE_LENGTH        reference value for data.max_response_length
#                            used to derive L_cache from a fraction
#                            (default: 31744 — matches the parent script's
#                            data.max_response_length). Override if you
#                            also override data.max_response_length.
#     LP_MAX_TOKENS          episode_max_tokens (L_max). Default: unset →
#                            engine uses data.max_response_length.
#     LP_CACHE_TOKENS        episode_cache_tokens (L_cache), absolute.
#                            Default: unset → see LP_CACHE_FRACTION.
#     LP_CACHE_FRACTION      L_cache as a fraction of RESPONSE_LENGTH
#                            (default: 0.25). Set 0.5 to make the cache
#                            half the response length. Ignored when
#                            LP_CACHE_TOKENS is set explicitly.
# =============================================================================

export TASKS_CONFIG=${TASKS_CONFIG:-configs/eval_maze_oracle_10ep.yaml}
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-1.7B}
GAMMA=${GAMMA:-0.95}


# ---- Speed knobs ---------------------------------------------------------
TRAIN_BATCH=${TRAIN_BATCH:-32}
ROLLOUT_N=${ROLLOUT_N:-8}
MINI_BATCH=${MINI_BATCH:-128}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-65536}
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

# ---- Length penalty (overlong reward shaping; off by default) ------------
RESPONSE_LENGTH=${RESPONSE_LENGTH:-31744}
LP_CACHE_FRACTION=${LP_CACHE_FRACTION:-0.25}
LP_OVERRIDES=()
if [ "${LENGTH_PENALTY:-false}" = "true" ] || [ "${LENGTH_PENALTY:-false}" = "True" ]; then
    LP_OVERRIDES+=(+rllm.length_penalty.enable=true)
    # L_max: explicit override only; otherwise the engine defaults it to
    # data.max_response_length.
    if [ -n "${LP_MAX_TOKENS:-}" ]; then
        LP_OVERRIDES+=(+rllm.length_penalty.episode_max_tokens=$LP_MAX_TOKENS)
        _LP_BASE=$LP_MAX_TOKENS
    else
        _LP_BASE=$RESPONSE_LENGTH
    fi
    # L_cache: explicit absolute value wins; otherwise derive from the
    # fraction of the base (either the explicit L_max or RESPONSE_LENGTH).
    if [ -z "${LP_CACHE_TOKENS:-}" ]; then
        LP_CACHE_TOKENS=$(awk "BEGIN{printf \"%d\", $_LP_BASE * $LP_CACHE_FRACTION}")
    fi
    LP_OVERRIDES+=(+rllm.length_penalty.episode_cache_tokens=$LP_CACHE_TOKENS)
    echo "[wrapper] length penalty ON: L_max=${LP_MAX_TOKENS:-<engine default = data.max_response_length>} L_cache=$LP_CACHE_TOKENS"
fi

# Episodic summary fits the user's "summarize at episode end" intent;
# threshold is set high so token-trigger never fires (oracle is the only
# summary source we want here).
export MODE=${MODE:-episodic}
export SUMMARIZATION_THRESHOLD=${SUMMARIZATION_THRESHOLD:-131072}
export SUMMARY_MAX_TOKENS=${SUMMARY_MAX_TOKENS:-4096}

bash scripts/train_multi_task_summarizing.sh \
    +rllm.agent.summarization.oracle.enable=true \
    +rllm.agent.summarization.oracle.scope=maze \
    +rllm.advantage_method.name=chunk_discounted_topr \
    +rllm.advantage_method.chunk_discounted_topr.reward_scope=per_episode \
    +rllm.advantage_method.chunk_discounted_topr.gamma=$GAMMA \
    +rllm.advantage_method.chunk_discounted_topr.topr_split.enable=true \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.project_name='segment-training-validation' \
    "${SPEED_OVERRIDES[@]}" \
    "${LP_OVERRIDES[@]}" \
    "$@"
