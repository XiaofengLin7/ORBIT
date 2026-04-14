#!/bin/bash
set -x

# ============================================================
# SciWorld Benchmark: evaluate a checkpoint on 6 SciWorld tasks
# Tasks: lifespan-{shortest,longest,both}, find-{non-living,plant}, power-component
# Each benchmark runs twice: reflection OFF (ReAct) and ON (Reflexion).
#
# Usage:
#   MODEL_PATH=/projectnb/ds310/ORBIT_Qwen3_8b bash scripts/eval/sciworld_benchmark.sh
#
# Optional env vars:
#   MAX_CTX          — max context length (default: 32768)
#   EXPERIMENT_TAG   — suffix for experiment names (default: "benchmark")
#   CONDA_ENV        — conda environment name (default: "icx")
# ============================================================

source /share/pkg.7/miniconda/23.1.0/install/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-icx}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export MKL_THREADING_LAYER=GNU

cd /projectnb/replearn/xfl/explorer

# ---- Required: model checkpoint ----
if [ -z "$MODEL_PATH" ]; then
    echo "Error: MODEL_PATH is not set."
    echo "Usage: MODEL_PATH=/projectnb/ds310/ORBIT_Qwen3_8b bash scripts/eval/sciworld_benchmark.sh"
    exit 1
fi

MAX_CTX=${MAX_CTX:-32768}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-benchmark}
MODEL_NAME=$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')

COMMON=(
    trainer.total_epochs=0
    trainer.n_gpus_per_node=2
    actor_rollout_ref.rollout.val_kwargs.n=1
    data.max_response_length=$((MAX_CTX - 1024))
    actor_rollout_ref.rollout.max_model_len=$MAX_CTX
    data.train_batch_size=4
    +rllm.agent.trajectory_timeout=600
    +rllm.agent.retry_limit=1
)

# Append any extra overrides passed as positional args
COMMON+=("$@")

# ======================== SciWorld — no reflection ========================
echo "=========================================="
echo "[1/2] SciWorld — no reflection"
echo "=========================================="
TASKS_CONFIG=configs/eval_sciworld_multi.yaml MODEL_PATH="$MODEL_PATH" \
bash scripts/train_multi_task_multi_episode.sh \
    "${COMMON[@]}" \
    data.val_batch_size=128 \
    trainer.project_name=sciworld-benchmark \
    +rllm.env.env_args.enable_reflection=False \
    trainer.experiment_name="sciworld-${MODEL_NAME}-${EXPERIMENT_TAG}-no-reflection"
echo "SciWorld (no reflection) completed with exit code: $?"

# ======================== SciWorld — with reflection ========================
echo "=========================================="
echo "[2/2] SciWorld — with reflection"
echo "=========================================="
TASKS_CONFIG=configs/eval_sciworld_multi.yaml MODEL_PATH="$MODEL_PATH" \
bash scripts/train_multi_task_multi_episode.sh \
    "${COMMON[@]}" \
    data.val_batch_size=128 \
    trainer.project_name=sciworld-benchmark \
    +rllm.env.env_args.enable_reflection=True \
    trainer.experiment_name="sciworld-${MODEL_NAME}-${EXPERIMENT_TAG}-reflection"
echo "SciWorld (reflection) completed with exit code: $?"

echo "=========================================="
echo "SciWorld benchmark complete."
echo "=========================================="
