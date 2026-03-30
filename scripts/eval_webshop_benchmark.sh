#!/bin/bash
set -x

# ============================================================
# WebShop Benchmark: ReAct vs Reflexion vs ORBIT
# All 3 methods: multi-episode (3 ep x 15 turns)
#
# Prerequisites:
#   1. Build WebShop data (run once):
#        python -m gem.envs.webshop.preprocess --mode all
#      This creates .cache/webshop/webshop.db and resources/documents.jsonl
#
#   2. Build Lucene search index (run once):
#        python -m pyserini.index.lucene \
#          --collection JsonCollection \
#          --input .cache/webshop/resources \
#          --index .cache/webshop/indexes \
#          --generator DefaultLuceneDocumentGenerator \
#          --storeRaw \
#          --threads 4
#
#   3. Download spaCy model (run once):
#        python -m spacy download en_core_web_sm
# ============================================================

# Activate conda environment
source /share/pkg.7/miniconda/23.1.0/install/etc/profile.d/conda.sh
conda activate icx

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export MKL_THREADING_LAYER=GNU  # Avoid MKL/libgomp conflict in vLLM workers

# Verify prerequisites
if [ ! -f ".cache/webshop/webshop.db" ]; then
    echo "Error: .cache/webshop/webshop.db not found."
    echo "Run: python -m gem.envs.webshop.preprocess --mode all"
    exit 1
fi
if [ ! -d ".cache/webshop/indexes" ]; then
    echo "Error: .cache/webshop/indexes not found."
    echo "Run: python -m pyserini.index.lucene ..."
    exit 1
fi

BASE_MODEL="Qwen/Qwen3-8B"
ORBIT_MODEL="/projectnb/ds310/actor_hf"
CONFIG=configs/eval_webshop_multi.yaml

# Shared overrides (only what differs from inner script defaults)
SHARED=(
    trainer.total_epochs=0
    trainer.n_gpus_per_node=2
    actor_rollout_ref.rollout.val_kwargs.n=1
    trainer.project_name=webshop-benchmark
    data.val_batch_size=100
)

# --- 1. ReAct: base model, no reflection ---
echo "=========================================="
echo "[1/3] ReAct: Qwen3-8B, no reflection"
echo "=========================================="
TASKS_CONFIG="$CONFIG" MODEL_PATH="$BASE_MODEL" \
bash scripts/train_multi_task_multi_episode.sh \
    "${SHARED[@]}" \
    +rllm.env.env_args.enable_reflection=False \
    trainer.experiment_name="webshop-react-qwen3-8b"
echo "ReAct completed with exit code: $?"

# --- 2. Reflexion: base model, with reflection ---
echo "=========================================="
echo "[2/3] Reflexion: Qwen3-8B, with reflection"
echo "=========================================="
TASKS_CONFIG="$CONFIG" MODEL_PATH="$BASE_MODEL" \
bash scripts/train_multi_task_multi_episode.sh \
    "${SHARED[@]}" \
    +rllm.env.env_args.enable_reflection=True \
    trainer.experiment_name="webshop-reflexion-qwen3-8b"
echo "Reflexion completed with exit code: $?"

# --- 3. ORBIT: fine-tuned model, with reflection ---
echo "=========================================="
echo "[3/3] ORBIT: actor_hf, with reflection"
echo "=========================================="
TASKS_CONFIG="$CONFIG" MODEL_PATH="$ORBIT_MODEL" \
bash scripts/train_multi_task_multi_episode.sh \
    "${SHARED[@]}" \
    +rllm.env.env_args.enable_reflection=True \
    trainer.experiment_name="webshop-orbit-actor-hf-reflection"
echo "ORBIT completed with exit code: $?"

# --- 4. GPU keep-alive ---
echo "=========================================="
echo "All WebShop evaluations complete. Starting GPU keep-alive..."
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python /projectnb/replearn/xfl/REIL/data/dummy.py --gpus 0
