#!/bin/bash
set -x

# ============================================================
# Full Benchmark: evaluate a checkpoint on all val tasks
# Each benchmark runs twice: reflection OFF (ReAct) and ON (Reflexion).
#
#   1a/1b. GEM val tasks (maze, mastermind, grid)
#   2a/2b. ALFWorld (eval_out_of_distribution)
#   3a/3b. WebShop (test split, 100 tasks)
#
# Usage:
#   MODEL_PATH=/path/to/checkpoint bash scripts/eval/benchmark.sh
#
# Optional env vars:
#   MAX_CTX          — max context length (default: 32768)
#   EXPERIMENT_TAG   — suffix for experiment names (default: "benchmark")
#   SKIP_GEM         — set to 1 to skip GEM val tasks
#   SKIP_ALFWORLD    — set to 1 to skip ALFWorld
#   SKIP_WEBSHOP     — set to 1 to skip WebShop
# ============================================================

source /share/pkg.7/miniconda/23.1.0/install/etc/profile.d/conda.sh
conda activate icx

export ALFWORLD_DATA=/projectnb/replearn/xfl/alfworld_data
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export MKL_THREADING_LAYER=GNU

cd /projectnb/replearn/xfl/explorer

# ---- Required: model checkpoint ----
if [ -z "$MODEL_PATH" ]; then
    echo "Error: MODEL_PATH is not set."
    echo "Usage: MODEL_PATH=/path/to/checkpoint bash scripts/eval/benchmark.sh"
    exit 1
fi

MAX_CTX=${MAX_CTX:-32768}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-benchmark}
MODEL_NAME=$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')

SKIP_GEM=${SKIP_GEM:-0}
SKIP_ALFWORLD=${SKIP_ALFWORLD:-0}
SKIP_WEBSHOP=${SKIP_WEBSHOP:-0}

# Common overrides shared across all benchmarks
COMMON=(
    trainer.total_epochs=0
    trainer.n_gpus_per_node=2
    actor_rollout_ref.rollout.val_kwargs.n=4
    data.max_response_length=$((MAX_CTX - 1024))
    actor_rollout_ref.rollout.max_model_len=$MAX_CTX
    data.train_batch_size=4
    +rllm.agent.trajectory_timeout=600
    +rllm.agent.retry_limit=1
)

STEP=0
TOTAL=6

# ======================== 1. GEM Val Tasks ========================
if [ "$SKIP_GEM" != "1" ]; then
    STEP=$((STEP + 1))
    echo "=========================================="
    echo "[$STEP/$TOTAL] GEM val tasks — no reflection"
    echo "=========================================="
    TASKS_CONFIG=configs/multi_task_multi_episode_config.yaml MODEL_PATH="$MODEL_PATH" \
    bash scripts/train_multi_task_multi_episode.sh \
        "${COMMON[@]}" \
        data.val_batch_size=128 \
        trainer.project_name=gem-benchmark \
        +rllm.env.env_args.enable_reflection=False \
        trainer.experiment_name="gem-${MODEL_NAME}-${EXPERIMENT_TAG}-no-reflection"
    echo "GEM (no reflection) completed with exit code: $?"

    STEP=$((STEP + 1))
    echo "=========================================="
    echo "[$STEP/$TOTAL] GEM val tasks — with reflection"
    echo "=========================================="
    TASKS_CONFIG=configs/multi_task_multi_episode_config.yaml MODEL_PATH="$MODEL_PATH" \
    bash scripts/train_multi_task_multi_episode.sh \
        "${COMMON[@]}" \
        data.val_batch_size=128 \
        trainer.project_name=gem-benchmark \
        +rllm.env.env_args.enable_reflection=True \
        trainer.experiment_name="gem-${MODEL_NAME}-${EXPERIMENT_TAG}-reflection"
    echo "GEM (reflection) completed with exit code: $?"
else
    STEP=$((STEP + 2))
    echo "[GEM] — SKIPPED"
fi

# ======================== 2. ALFWorld ========================
if [ "$SKIP_ALFWORLD" != "1" ]; then
    STEP=$((STEP + 1))
    echo "=========================================="
    echo "[$STEP/$TOTAL] ALFWorld — no reflection"
    echo "=========================================="
    TASKS_CONFIG=configs/eval_alfworld_multi.yaml MODEL_PATH="$MODEL_PATH" \
    bash scripts/train_multi_task_multi_episode.sh \
        "${COMMON[@]}" \
        data.val_batch_size=16 \
        trainer.project_name=alfworld-benchmark \
        +rllm.env.env_args.enable_reflection=False \
        trainer.experiment_name="alfworld-${MODEL_NAME}-${EXPERIMENT_TAG}-no-reflection"
    echo "ALFWorld (no reflection) completed with exit code: $?"

    STEP=$((STEP + 1))
    echo "=========================================="
    echo "[$STEP/$TOTAL] ALFWorld — with reflection"
    echo "=========================================="
    TASKS_CONFIG=configs/eval_alfworld_multi.yaml MODEL_PATH="$MODEL_PATH" \
    bash scripts/train_multi_task_multi_episode.sh \
        "${COMMON[@]}" \
        data.val_batch_size=16 \
        trainer.project_name=alfworld-benchmark \
        +rllm.env.env_args.enable_reflection=True \
        trainer.experiment_name="alfworld-${MODEL_NAME}-${EXPERIMENT_TAG}-reflection"
    echo "ALFWorld (reflection) completed with exit code: $?"
else
    STEP=$((STEP + 2))
    echo "[ALFWorld] — SKIPPED"
fi

# ======================== 3. WebShop ========================
if [ "$SKIP_WEBSHOP" != "1" ]; then
    if [ ! -f ".cache/webshop/webshop.db" ]; then
        echo "Error: .cache/webshop/webshop.db not found."
        exit 1
    fi
    if [ ! -d ".cache/webshop/indexes" ]; then
        echo "Error: .cache/webshop/indexes not found."
        exit 1
    fi

    STEP=$((STEP + 1))
    echo "=========================================="
    echo "[$STEP/$TOTAL] WebShop — no reflection"
    echo "=========================================="
    TASKS_CONFIG=configs/eval_webshop_multi.yaml MODEL_PATH="$MODEL_PATH" \
    bash scripts/train_multi_task_multi_episode.sh \
        "${COMMON[@]}" \
        data.val_batch_size=16 \
        trainer.project_name=webshop-benchmark \
        +rllm.env.env_args.enable_reflection=False \
        trainer.experiment_name="webshop-${MODEL_NAME}-${EXPERIMENT_TAG}-no-reflection"
    echo "WebShop (no reflection) completed with exit code: $?"

    STEP=$((STEP + 1))
    echo "=========================================="
    echo "[$STEP/$TOTAL] WebShop — with reflection"
    echo "=========================================="
    TASKS_CONFIG=configs/eval_webshop_multi.yaml MODEL_PATH="$MODEL_PATH" \
    bash scripts/train_multi_task_multi_episode.sh \
        "${COMMON[@]}" \
        data.val_batch_size=16 \
        trainer.project_name=webshop-benchmark \
        +rllm.env.env_args.enable_reflection=True \
        trainer.experiment_name="webshop-${MODEL_NAME}-${EXPERIMENT_TAG}-reflection"
    echo "WebShop (reflection) completed with exit code: $?"
else
    STEP=$((STEP + 2))
    echo "[WebShop] — SKIPPED"
fi

echo "=========================================="
echo "All benchmarks complete."
echo "=========================================="
