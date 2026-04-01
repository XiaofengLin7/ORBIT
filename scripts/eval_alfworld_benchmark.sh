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

# YaRN rope scaling — matches ORBIT's config for fair comparison.
# Toggle: ENABLE_YARN=0 to disable (base model uses native 40960 context).
ENABLE_YARN=${ENABLE_YARN:-1}
MAX_CTX=${MAX_CTX:-65536}

if [ "$ENABLE_YARN" = "1" ]; then
    YARN_MODEL_DIR=$(mktemp -d /tmp/qwen3-8b-yarn.XXXXXX)
    trap 'rm -rf "$YARN_MODEL_DIR"' EXIT
    HF_LOCAL=$(conda run -n icx python -c "
from huggingface_hub import try_to_load_from_cache; import os
p = try_to_load_from_cache('${BASE_MODEL}', 'config.json')
print(os.path.dirname(p))
")
    for f in "$HF_LOCAL"/*; do
        ln -s "$f" "$YARN_MODEL_DIR/$(basename "$f")"
    done
    rm "$YARN_MODEL_DIR/config.json"
    conda run -n icx python -c "
import json
with open('${HF_LOCAL}/config.json') as f:
    cfg = json.load(f)
cfg['rope_scaling'] = {'rope_type': 'yarn', 'factor': 4.0, 'original_max_position_embeddings': 32768}
cfg['max_position_embeddings'] = 131072
with open('${YARN_MODEL_DIR}/config.json', 'w') as f:
    json.dump(cfg, f, indent=2)
"
    BASE_MODEL="$YARN_MODEL_DIR"
    echo "YaRN enabled: base model patched at $YARN_MODEL_DIR"
else
    MAX_CTX=$(( MAX_CTX < 40960 ? MAX_CTX : 40960 ))
    echo "YaRN disabled: using native context (max ${MAX_CTX})"
fi

# Action-only mode: keep only parsed actions in history (discard thinking).
# Toggle: ACTION_ONLY=1 to enable non-cumulative context with stepwise advantage.
ACTION_ONLY=${ACTION_ONLY:-0}

# Shared overrides (only what differs from inner script defaults)
SHARED=(
    trainer.total_epochs=0
    trainer.n_gpus_per_node=2
    actor_rollout_ref.rollout.val_kwargs.n=4
    trainer.project_name=alfworld-benchmark
    data.max_response_length=$((MAX_CTX - 1024))
    actor_rollout_ref.rollout.max_model_len=$MAX_CTX
)

if [ "$ACTION_ONLY" = "1" ]; then
    echo "Action-only mode: stepwise advantage + non-cumulative context"
    EXP_SUFFIX="-action-only"
    SHARED+=(
        rllm.stepwise_advantage.enable=True
        rllm.stepwise_advantage.mode=broadcast
        data.max_prompt_length=${ACTION_ONLY_PROMPT_LEN:-16384}
        data.max_response_length=${ACTION_ONLY_RESPONSE_LEN:-4096}
        rllm.agent.name=gem_text_agent_noncumulative
    )
else
    EXP_SUFFIX=""
fi

# --- 1. ReAct: base model, no reflection ---
echo "=========================================="
echo "[1/3] ReAct: Qwen3-8B, no reflection"
echo "=========================================="
TASKS_CONFIG="$CONFIG" MODEL_PATH="$BASE_MODEL" \
bash scripts/train_multi_task_multi_episode.sh \
    "${SHARED[@]}" \
    +rllm.env.env_args.enable_reflection=False \
    trainer.experiment_name="alfworld-react-qwen3-8b${EXP_SUFFIX}"
echo "ReAct completed with exit code: $?"

# --- 2. Reflexion: base model, with reflection ---
echo "=========================================="
echo "[2/3] Reflexion: Qwen3-8B, with reflection"
echo "=========================================="
TASKS_CONFIG="$CONFIG" MODEL_PATH="$BASE_MODEL" \
bash scripts/train_multi_task_multi_episode.sh \
    "${SHARED[@]}" \
    +rllm.env.env_args.enable_reflection=True \
    trainer.experiment_name="alfworld-reflexion-qwen3-8b${EXP_SUFFIX}"
echo "Reflexion completed with exit code: $?"

# --- 3. ORBIT: fine-tuned model, no reflection ---
# echo "=========================================="
# echo "[3/3] ORBIT: actor_hf, with reflection"
# echo "=========================================="
# TASKS_CONFIG="$CONFIG" MODEL_PATH="$ORBIT_MODEL" \
# bash scripts/train_multi_task_multi_episode.sh \
#     "${SHARED[@]}" \
#     +rllm.env.env_args.enable_reflection=True \
#     trainer.experiment_name="alfworld-orbit-actor-hf-reflection${EXP_SUFFIX}"
# echo "ORBIT completed with exit code: $?"

# --- 4. GPU keep-alive ---
echo "=========================================="
echo "All evaluations complete. Starting GPU keep-alive..."
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python /projectnb/replearn/xfl/REIL/data/dummy.py --gpus 0
