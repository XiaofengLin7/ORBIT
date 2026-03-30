#!/bin/bash
set -x

# ============================================================
# ALFWorld Benchmark: ReAct vs Reflexion vs ORBIT
# All 3 methods: multi-episode (3 ep x 10 turns), val_kwargs.n=3
#
# Prerequisites:
#   pip install alfworld    # provides alfworld + textworld
#   export ALFWORLD_DATA=/path/to/alfworld_data  # game files
#
# If using a custom alfworld vendor path instead of pip:
#   export ALFWORLD_VENDOR_PATH=/path/to/alfworld/vendor/dir
# ============================================================

# Activate conda environment
source /share/pkg.7/miniconda/23.1.0/install/etc/profile.d/conda.sh
conda activate icx

export ALFWORLD_DATA=/projectnb/replearn/xfl/alfworld_data
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

BASE_MODEL="Qwen/Qwen3-8B"
ORBIT_MODEL="/projectnb/ds310/actor_hf"
CONFIG=configs/eval_alfworld_multi.yaml

# Shared overrides (only what differs from inner script defaults)
SHARED=(
    trainer.total_epochs=0
    trainer.n_gpus_per_node=2
    actor_rollout_ref.rollout.val_kwargs.n=4
    trainer.project_name=alfworld-benchmark
)

# # --- 1. ReAct: base model, no reflection ---
# echo "=========================================="
# echo "[1/3] ReAct: Qwen3-8B, no reflection"
# echo "=========================================="
# TASKS_CONFIG="$CONFIG" MODEL_PATH="$BASE_MODEL" \
# bash scripts/train_multi_task_multi_episode.sh \
#     "${SHARED[@]}" \
#     +rllm.env.env_args.enable_reflection=False \
#     trainer.experiment_name="alfworld-react-qwen3-8b"
# echo "ReAct completed with exit code: $?"

# # --- 2. Reflexion: base model, with reflection ---
# echo "=========================================="
# echo "[2/3] Reflexion: Qwen3-8B, with reflection"
# echo "=========================================="
# TASKS_CONFIG="$CONFIG" MODEL_PATH="$BASE_MODEL" \
# bash scripts/train_multi_task_multi_episode.sh \
#     "${SHARED[@]}" \
#     +rllm.env.env_args.enable_reflection=True \
#     trainer.experiment_name="alfworld-reflexion-qwen3-8b"
# echo "Reflexion completed with exit code: $?"

# --- 3. ORBIT: fine-tuned model, no reflection ---
echo "=========================================="
echo "[3/3] ORBIT: actor_hf, with reflection"
echo "=========================================="
TASKS_CONFIG="$CONFIG" MODEL_PATH="$ORBIT_MODEL" \
bash scripts/train_multi_task_multi_episode.sh \
    "${SHARED[@]}" \
    +rllm.env.env_args.enable_reflection=True \
    trainer.experiment_name="alfworld-orbit-actor-hf-reflection"
echo "ORBIT completed with exit code: $?"

# --- 4. GPU keep-alive ---
echo "=========================================="
echo "All evaluations complete. Starting GPU keep-alive..."
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python /projectnb/replearn/xfl/REIL/data/dummy.py --gpus 0
