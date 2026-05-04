#!/bin/bash
set -x

# OpenAI Model Evaluation Script
# Usage: bash scripts/eval/openai.sh
# Override parameters with environment variables or append args: bash scripts/eval/openai.sh --temperature 0.5

# Model configuration
MODEL=${MODEL:-gpt-5.2}
BASE_URL=${BASE_URL:-https://api.openai.com/v1}
# API_KEY defaults to OPENAI_API_KEY environment variable

# Task configuration
CONFIG=${CONFIG:-configs/eval_config.yaml}
SEED=${SEED:-42}
N_ROLLOUTS=${N_ROLLOUTS:-1}
ENV_MODE=${ENV_MODE:-multi}

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

# Execution configuration
N_PARALLEL=${N_PARALLEL:-256}
TRAJECTORY_TIMEOUT=${TRAJECTORY_TIMEOUT:-1200}

# Sampling parameters
TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-0.95}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}  # gpt-4o caps at 16384; gpt-5.2 accepts more — override upward if needed
# Reasoning effort: only set for reasoning models (gpt-5*, o*); leave empty for gpt-4o etc.
REASONING_EFFORT=${REASONING_EFFORT:-}

# Output configuration
OUTPUT_DIR=${OUTPUT_DIR:-results}
mkdir -p "$OUTPUT_DIR"

# Generate output filename from model and config
MODEL_SAFE=$(echo "$MODEL" | tr '/:' '_')
CONFIG_NAME=$(basename "$CONFIG" .yaml | tr '[:upper:]' '[:lower:]')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT=${OUTPUT:-${OUTPUT_DIR}/eval_${MODEL_SAFE}_${CONFIG_NAME}_${TIMESTAMP}.json}

# Run evaluation
EXTRA_ARGS=()
if [ -n "$REASONING_EFFORT" ]; then
    EXTRA_ARGS+=(--reasoning-effort "$REASONING_EFFORT")
fi

python scripts/eval/openai.py \
    --config "$CONFIG" \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --n-parallel "$N_PARALLEL" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --max-response-length "$MAX_RESPONSE_LENGTH" \
    --trajectory-timeout "$TRAJECTORY_TIMEOUT" \
    --seed "$SEED" \
    --n-rollouts "$N_ROLLOUTS" \
    --env-mode "$ENV_MODE" \
    --output "$OUTPUT" \
    --log-chat-completions \
    "${EXTRA_ARGS[@]}" \
    "$@"