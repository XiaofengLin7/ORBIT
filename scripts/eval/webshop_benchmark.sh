#!/bin/bash
set -x

# ============================================================
# WebShop Sequential Eval: ORBIT / MTSE / BASE Qwen3-8B
#
# Evaluates three Qwen3-8B variants on WebShop test split using the
# newly added "sequential seeds" setting in configs/eval_webshop_multi.yaml
# (seeds = [0..99] → first 100 goals of the test split).
#
#   - ORBIT (reflection OFF only)
#   - MTSE  (reflection OFF only)
#   - BASE  (both reflection OFF == ReAct, and reflection ON == Reflexion)
#
# Uses 2 GPUs. After the eval loop finishes (success or failure), keeps
# the GPUs busy so the allocation isn't reclaimed while you inspect results.
#
# Usage:
#   bash scripts/eval/webshop_benchmark.sh
#
# Optional env vars:
#   CUDA_VISIBLE_DEVICES  (default: 0,1)
#   MAX_CTX               (default: 32768)
#   ENABLE_YARN           (default: 1) — patch BASE model with YaRN to match ORBIT/MTSE
#   SKIP_ORBIT / SKIP_MTSE / SKIP_BASE  (default: 0) — skip individual runs
#   MTSE_MODEL_PATH       (default: /projectnb/ds310/MTSE_3x_rollout_Qwen3-8b)
# ============================================================

source /share/pkg.7/miniconda/23.1.0/install/etc/profile.d/conda.sh
conda activate icx

# Raise FD limit: each concurrent WebShop env opens Lucene/SQLite/HTML FDs.
# With val_kwargs.n × val_batch_size concurrent trajectories, the default 1024 is blown easily.
TARGET_NOFILE=1048576
HARD_NOFILE=$(ulimit -Hn 2>/dev/null || echo unlimited)
if [ "$HARD_NOFILE" = "unlimited" ]; then
    ulimit -n "$TARGET_NOFILE" 2>/dev/null || true
elif [ "$HARD_NOFILE" -lt "$TARGET_NOFILE" ]; then
    ulimit -n "$HARD_NOFILE"
else
    ulimit -n "$TARGET_NOFILE"
fi
echo "ulimit -n: $(ulimit -n)"

# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export MKL_THREADING_LAYER=GNU

cd /projectnb/replearn/xfl/explorer

# ---- Prerequisite checks ----
if [ ! -f ".cache/webshop/webshop.db" ]; then
    echo "Error: .cache/webshop/webshop.db not found."
    echo "Run: python -m gem.envs.webshop.preprocess --mode all"
    exit 1
fi
if [ ! -d ".cache/webshop/indexes" ]; then
    echo "Error: .cache/webshop/indexes not found. Build the Lucene index first."
    exit 1
fi

CONFIG=configs/eval_webshop_multi.yaml
ORBIT_MODEL=/projectnb/ds310/ORBIT_Qwen3_8b
MTSE_MODEL=${MTSE_MODEL_PATH:-/projectnb/ds310/MTSE_3x_rollout_Qwen3-8b}
BASE_MODEL="Qwen/Qwen3-8B"

MAX_CTX=${MAX_CTX:-65536}
ENABLE_YARN=${ENABLE_YARN:-1}

SKIP_ORBIT=${SKIP_ORBIT:-0}
SKIP_MTSE=${SKIP_MTSE:-0}
SKIP_BASE=${SKIP_BASE:-0}

# ---- Patch BASE model with YaRN rope scaling (mirrors ORBIT/MTSE config) ----
if [ "$ENABLE_YARN" = "1" ] && [ "$SKIP_BASE" != "1" ]; then
    YARN_MODEL_DIR=$(mktemp -d /tmp/qwen3-8b-yarn.XXXXXX)
    trap 'rm -rf "$YARN_MODEL_DIR"' EXIT
    HF_LOCAL=$(conda run -n icx python -c "
from huggingface_hub import try_to_load_from_cache; import os
p = try_to_load_from_cache('${BASE_MODEL}', 'config.json')
print(os.path.dirname(p))
")
    if [ -z "$HF_LOCAL" ] || [ ! -d "$HF_LOCAL" ]; then
        echo "Error: Could not locate HF cache for ${BASE_MODEL}. Pre-download it first."
        exit 1
    fi
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
    BASE_MODEL_PATH="$YARN_MODEL_DIR"
    echo "YaRN enabled: BASE model patched at $YARN_MODEL_DIR"
else
    BASE_MODEL_PATH="$BASE_MODEL"
    MAX_CTX=$(( MAX_CTX < 40960 ? MAX_CTX : 40960 ))
    echo "YaRN disabled (or BASE skipped): using native context (max ${MAX_CTX})"
fi

# ---- Shared overrides for 2-GPU eval-only ----
SHARED=(
    trainer.total_epochs=0
    trainer.n_gpus_per_node=2
    actor_rollout_ref.rollout.val_kwargs.n=4
    trainer.project_name=webshop-seq-eval
    data.val_batch_size=100
    data.max_response_length=$((MAX_CTX - 1024))
    actor_rollout_ref.rollout.max_model_len=$MAX_CTX
)

run_eval() {
    # run_eval <tag> <model_path> <reflection_bool>
    local tag="$1"
    local model_path="$2"
    local reflection="$3"
    echo "=========================================="
    echo "[${tag}] model=${model_path}  reflection=${reflection}"
    echo "=========================================="
    TASKS_CONFIG="$CONFIG" MODEL_PATH="$model_path" \
    bash scripts/train_multi_task_multi_episode.sh \
        "${SHARED[@]}" \
        +rllm.env.env_args.enable_reflection="$reflection" \
        trainer.experiment_name="${tag}"
    echo "[${tag}] completed with exit code: $?"
}

# ---- 1. ORBIT (reflection OFF) ----
if [ "$SKIP_ORBIT" != "1" ]; then
    run_eval "webshop-seq-orbit-qwen3-8b-no-reflection" "$ORBIT_MODEL" "False"
else
    echo "[ORBIT] — SKIPPED"
fi

# ---- 2. MTSE (reflection OFF) ----
if [ "$SKIP_MTSE" != "1" ]; then
    run_eval "webshop-seq-mtse-qwen3-8b-no-reflection" "$MTSE_MODEL" "False"
else
    echo "[MTSE] — SKIPPED"
fi

# ---- 3. BASE (reflection OFF == ReAct) ----
if [ "$SKIP_BASE" != "1" ]; then
    run_eval "webshop-seq-base-qwen3-8b-react" "$BASE_MODEL_PATH" "False"

    # ---- 4. BASE (reflection ON == Reflexion) ----
    run_eval "webshop-seq-base-qwen3-8b-reflexion" "$BASE_MODEL_PATH" "True"
else
    echo "[BASE] — SKIPPED"
fi

echo "=========================================="
echo "All WebShop sequential evaluations complete."
echo "=========================================="

# ---- GPU keep-alive (runs regardless of eval success/failure) ----
echo "Starting GPU keep-alive on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (Ctrl+C to stop)..."
python -c "
import torch, time, os, signal, sys
stop = False
def handler(s, f):
    global stop
    stop = True
    print('Stopping keep-alive.', flush=True)
signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)
n = torch.cuda.device_count()
if n == 0:
    print('No GPUs visible.'); sys.exit(0)
ts = [torch.randn(16384, 16384, device=f'cuda:{i}') for i in range(n)]
print(f'Keep-alive active on {n} GPU(s), CUDA_VISIBLE_DEVICES={os.environ.get(\"CUDA_VISIBLE_DEVICES\", \"<unset>\")}', flush=True)
step = 0
while not stop:
    for t in ts:
        for _ in range(10): _ = t @ t.T
    step += 1
    if step % 60 == 0:
        print(f'Keep-alive step {step} at {time.strftime(\"%H:%M:%S\")}', flush=True)
    time.sleep(1)
"
