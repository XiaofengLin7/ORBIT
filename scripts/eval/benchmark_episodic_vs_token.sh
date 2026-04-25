#!/bin/bash
set -x

# =============================================================================
# Benchmark: base model vs ORBIT model, with token vs episodic summarization,
# on a combined maze+mastermind eval set that covers 3- and 5-episode budgets.
#
# Grid:
#   2 models × 2 summ modes = 4 eval runs.
#
# Each run evaluates all (env × num_episodes) variants listed in $EVAL_CONFIG
# in a single pass. Per-task horizon is announced to the model via a system
# prompt that contains the literal "{num_episodes}" placeholder; GEMTextAgent
# substitutes it on first update_from_env using info["num_episodes"] exposed
# by MultiEpisodeEnv. data_source is suffixed with "-ep{N}" so metrics break
# out per-horizon.
#
# Runs the training infra in eval-only mode (trainer.total_epochs=0,
# trainer.val_before_train=True) so each combination runs val once and exits.
#
# Usage:
#   bash scripts/eval/benchmark_episodic_vs_token.sh
#
# Overridable env vars:
#   BASE_MODEL_PATH           — HF id or local path (default: Qwen/Qwen3-8B)
#   ORBIT_MODEL_PATH          — local checkpoint (no default; set to your trained ORBIT model)
#   EVAL_CONFIG               — yaml with val_tasks (required; e.g. configs/multi_task_summarization_config.yaml)
#   EXPERIMENT_TAG            — suffix for W&B experiment names
#   MAX_CTX                   — max context length (default: 32768)
#   SUMMARY_MAX_TOKENS        — summary completion budget (default: 8192)
#   SUMMARIZATION_THRESHOLD   — token threshold for mode=token (default: 16384)
#   SKIP_BASE                 — set to 1 to skip the base-model runs
#   SKIP_ORBIT                — set to 1 to skip the ORBIT-model runs
#   ONLY_MODES                — comma list from {token,episodic} (default: "token,episodic")
# =============================================================================

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export MKL_THREADING_LAYER=GNU

# Run from the repo root with the appropriate conda environment already
# activated.

# ---- Models ----------------------------------------------------------------
BASE_MODEL_PATH=${BASE_MODEL_PATH:-Qwen/Qwen3-8B}
ORBIT_MODEL_PATH=${ORBIT_MODEL_PATH:?ORBIT_MODEL_PATH must point to a local ORBIT checkpoint}

SKIP_BASE=${SKIP_BASE:-0}
SKIP_ORBIT=${SKIP_ORBIT:-0}

# ---- Sweep dimensions -------------------------------------------------------
ONLY_MODES=${ONLY_MODES:-"token episodic"}
ONLY_MODES=${ONLY_MODES//,/ }

# ---- Shared params ----------------------------------------------------------
EVAL_CONFIG=${EVAL_CONFIG:?EVAL_CONFIG must point to a yaml file with val_tasks (e.g. configs/multi_task_summarization_config.yaml)}
MAX_CTX=${MAX_CTX:-32768}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-episodic-vs-token}

SUMMARIZATION_THRESHOLD=${SUMMARIZATION_THRESHOLD:-8192}
SUMMARY_MAX_TOKENS=${SUMMARY_MAX_TOKENS:-4096}

if [ ! -f "$EVAL_CONFIG" ]; then
    echo "Error: eval config not found: $EVAL_CONFIG" >&2
    exit 1
fi

# Common overrides (match benchmark_summarization.sh's shape)
COMMON=(
    trainer.total_epochs=0
    trainer.n_gpus_per_node=2
    actor_rollout_ref.rollout.val_kwargs.n=1
    data.max_response_length=$((MAX_CTX - 1024))
    actor_rollout_ref.rollout.max_model_len=$MAX_CTX
    data.train_batch_size=4
    data.val_batch_size=128
    +rllm.agent.trajectory_timeout=1200
    +rllm.agent.retry_limit=1
)

# Build the per-mode summarization override array. `token` uses the token
# threshold; `episodic` sets mode=episodic so the trainer auto-enables
# reflection_via_summarization on the env.
summ_overrides() {
    local mode=$1
    local -a out=(
        rllm.agent.name=gem_text_agent_summarizing
        +rllm.agent.summarization.enable=true
        +rllm.agent.summarization.mode="$mode"
        +rllm.agent.summarization.threshold_tokens=$SUMMARIZATION_THRESHOLD
        +rllm.agent.summarization.summary_max_tokens=$SUMMARY_MAX_TOKENS
    )
    printf '%s\n' "${out[@]}"
}

run_eval() {
    local model_tag=$1        # "base" or "orbit"
    local model_path=$2
    local summ_mode=$3        # token or episodic

    local model_name
    model_name=$(basename "$model_path" | tr '[:upper:]' '[:lower:]')

    local exp_name
    exp_name="${model_tag}-maze-mastermind-${summ_mode}-${EXPERIMENT_TAG}"

    local -a SUMM
    mapfile -t SUMM < <(summ_overrides "$summ_mode")

    echo "=========================================="
    echo "RUN: model=${model_tag} (${model_path})"
    echo "     summarization=${summ_mode}"
    echo "     config=${EVAL_CONFIG}  experiment=${exp_name}"
    echo "=========================================="

    # System prompt is baked into scripts/train_multi_episode.py's default
    # (includes a `{num_episodes}` placeholder that GEMTextAgent substitutes
    # per-task on the first env update), so no override needed here.
    TASKS_CONFIG="$EVAL_CONFIG" MODEL_PATH="$model_path" \
    bash scripts/train_multi_task_multi_episode.sh \
        "${COMMON[@]}" \
        "${SUMM[@]}" \
        trainer.project_name=meta-rl-summarization-eval \
        trainer.experiment_name="$exp_name"
    local rc=$?
    echo "[$exp_name] completed with exit code: $rc"
    return $rc
}

STEP=0
TOTAL=0
for model_tag in base orbit; do
    if [ "$model_tag" = "base" ] && [ "$SKIP_BASE" = "1" ]; then continue; fi
    if [ "$model_tag" = "orbit" ] && [ "$SKIP_ORBIT" = "1" ]; then continue; fi
    for summ_mode in $ONLY_MODES; do
        TOTAL=$((TOTAL + 1))
    done
done

echo "Planned runs: $TOTAL"

for model_tag in base orbit; do
    if [ "$model_tag" = "base" ]; then
        if [ "$SKIP_BASE" = "1" ]; then continue; fi
        model_path="$BASE_MODEL_PATH"
    else
        if [ "$SKIP_ORBIT" = "1" ]; then continue; fi
        model_path="$ORBIT_MODEL_PATH"
    fi

    for summ_mode in $ONLY_MODES; do
        STEP=$((STEP + 1))
        echo ">>> [${STEP}/${TOTAL}] ${model_tag} | ${summ_mode}"
        run_eval "$model_tag" "$model_path" "$summ_mode"
    done
done

echo "=========================================="
echo "All ${TOTAL} eval runs complete."
echo "=========================================="
