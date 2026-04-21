#!/bin/bash
set -x

# ============================================================
# Benchmark WITH Context Summarization
#
# Mirrors scripts/eval/benchmark.sh but adds summarization overrides.
# Each benchmark runs with summarization enabled (no reflection variant,
# since summarization replaces the need for explicit reflection prompts).
#
#   1. GEM val tasks (maze, mastermind)
#   2. ALFWorld (eval_out_of_distribution)
#   3. WebShop (test split)
#
# Usage:
#   MODEL_PATH=/path/to/checkpoint bash scripts/eval/benchmark_summarization.sh
#
# Optional env vars:
#   MAX_CTX                    — max context length (default: 32768)
#   EXPERIMENT_TAG             — suffix for experiment names (default: "benchmark-summ")
#   SUMMARIZATION_THRESHOLD    — token threshold to trigger summarization (default: 16384)
#   SUMMARY_MAX_TOKENS         — max tokens for the generated summary (default: 8192)
#   MAX_SUMMARIZATIONS         — cap on summarizations per trajectory (default: 5)
#   SKIP_GEM                   — set to 1 to skip GEM val tasks
#   SKIP_ALFWORLD              — set to 1 to skip ALFWorld
#   SKIP_WEBSHOP               — set to 1 to skip WebShop
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
    echo "Usage: MODEL_PATH=/path/to/checkpoint bash scripts/eval/benchmark_summarization.sh"
    exit 1
fi

MAX_CTX=${MAX_CTX:-32768}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-benchmark-summ}
MODEL_NAME=$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')

# Summarization parameters
SUMMARIZATION_THRESHOLD=${SUMMARIZATION_THRESHOLD:-16384}
SUMMARY_MAX_TOKENS=${SUMMARY_MAX_TOKENS:-8192}
MAX_SUMMARIZATIONS=${MAX_SUMMARIZATIONS:-5}

SKIP_GEM=${SKIP_GEM:-0}
SKIP_ALFWORLD=${SKIP_ALFWORLD:-0}
SKIP_WEBSHOP=${SKIP_WEBSHOP:-0}

# Common overrides shared across all benchmarks
COMMON=(
    trainer.total_epochs=0
    trainer.n_gpus_per_node=4
    actor_rollout_ref.rollout.val_kwargs.n=1
    data.max_response_length=$((MAX_CTX - 1024))
    actor_rollout_ref.rollout.max_model_len=$MAX_CTX
    data.train_batch_size=4
    +rllm.agent.trajectory_timeout=1200
    +rllm.agent.retry_limit=1
)

# Summarization overrides
SUMM=(
    rllm.agent.name=gem_text_agent_summarizing
    +rllm.agent.summarization.enable=true
    +rllm.agent.summarization.threshold_tokens=$SUMMARIZATION_THRESHOLD
    +rllm.agent.summarization.summary_max_tokens=$SUMMARY_MAX_TOKENS
    +rllm.agent.summarization.max_summarizations=$MAX_SUMMARIZATIONS
)

STEP=0
TOTAL=3

# ======================== 1. GEM Val Tasks ========================
if [ "$SKIP_GEM" != "1" ]; then
    STEP=$((STEP + 1))
    echo "=========================================="
    echo "[$STEP/$TOTAL] GEM val tasks — summarization (threshold=${SUMMARIZATION_THRESHOLD})"
    echo "=========================================="
    TASKS_CONFIG=configs/eval_summarization_config.yaml MODEL_PATH="$MODEL_PATH" \
    bash scripts/train_multi_task_multi_episode.sh \
        "${COMMON[@]}" \
        "${SUMM[@]}" \
        data.val_batch_size=128 \
        trainer.project_name=gem-benchmark \
        trainer.experiment_name="gem-${MODEL_NAME}-${EXPERIMENT_TAG}"
    echo "GEM (summarization) completed with exit code: $?"
else
    STEP=$((STEP + 1))
    echo "[GEM] — SKIPPED"
fi

# ======================== 2. ALFWorld ========================
if [ "$SKIP_ALFWORLD" != "1" ]; then
    STEP=$((STEP + 1))
    echo "=========================================="
    echo "[$STEP/$TOTAL] ALFWorld — summarization (threshold=${SUMMARIZATION_THRESHOLD})"
    echo "=========================================="
    TASKS_CONFIG=configs/eval_alfworld_multi.yaml MODEL_PATH="$MODEL_PATH" \
    bash scripts/train_multi_task_multi_episode.sh \
        "${COMMON[@]}" \
        "${SUMM[@]}" \
        data.val_batch_size=16 \
        trainer.project_name=alfworld-benchmark \
        trainer.experiment_name="alfworld-${MODEL_NAME}-${EXPERIMENT_TAG}"
    echo "ALFWorld (summarization) completed with exit code: $?"
else
    STEP=$((STEP + 1))
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
    echo "[$STEP/$TOTAL] WebShop — summarization (threshold=${SUMMARIZATION_THRESHOLD})"
    echo "=========================================="
    TASKS_CONFIG=configs/eval_webshop_multi.yaml MODEL_PATH="$MODEL_PATH" \
    bash scripts/train_multi_task_multi_episode.sh \
        "${COMMON[@]}" \
        "${SUMM[@]}" \
        data.val_batch_size=16 \
        trainer.project_name=webshop-benchmark \
        trainer.experiment_name="webshop-${MODEL_NAME}-${EXPERIMENT_TAG}"
    echo "WebShop (summarization) completed with exit code: $?"
else
    STEP=$((STEP + 1))
    echo "[WebShop] — SKIPPED"
fi

echo "=========================================="
echo "All summarization benchmarks complete."
echo "=========================================="

# Keep GPUs alive after evaluation finishes
echo "Starting GPU keep-alive (Ctrl+C to stop)..."
python -c "
import torch, time, os, signal, sys
stop = False
def handler(s, f): global stop; stop = True; print('Stopping keep-alive.', flush=True)
signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)
n = torch.cuda.device_count()
if n == 0: print('No GPUs visible.'); sys.exit(0)
ts = [torch.randn(16384, 16384, device=f'cuda:{i}') for i in range(n)]
print(f'Keep-alive active on {n} GPU(s), CUDA_VISIBLE_DEVICES={os.environ.get(\"CUDA_VISIBLE_DEVICES\", \"<unset>\")}', flush=True)
step = 0
while not stop:
    for t in ts:
        for _ in range(10): _ = t @ t.T
    step += 1
    if step % 60 == 0: print(f'Keep-alive step {step} at {time.strftime(\"%H:%M:%S\")}', flush=True)
    time.sleep(1)
"
