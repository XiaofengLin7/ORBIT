#!/bin/bash
set -x

# =============================================================================
# Train a summarizing agent (default: episodic summarization).
#
# Wraps scripts/train_multi_task_multi_episode.sh with summarization-specific
# overrides:
#   - rllm.agent.name = gem_text_agent_summarizing
#   - rllm.agent.summarization.enable = true
#   - rllm.agent.summarization.mode = $MODE  (default: episodic)
# When mode is "episodic" or "both", scripts/train_multi_episode.py also
# auto-sets +rllm.env.env_args.reflection_via_summarization=true so the env
# splits its episode boundary for the engine's start_new_episode() advance.
#
# Usage:
#   bash scripts/train_multi_task_summarizing.sh
#
# Env-var overrides:
#   TASKS_CONFIG               yaml with train_tasks/val_tasks
#                              (default: configs/multi_task_multi_episode_config.yaml)
#   MODEL_PATH                 HF id or local path (default: Qwen/Qwen3-4B,
#                              i.e. inherits the underlying script's default)
#   MODE                       summarization mode: token | episodic | both
#                              (default: episodic)
#   SUMMARIZATION_THRESHOLD    token threshold for mode=token | both
#                              (default: 16384)
#   SUMMARY_MAX_TOKENS         summary completion budget
#                              (default: 4096)
#   MAX_SUMMARIZATIONS         per-trajectory cap (default: 5)
#
# Any additional Hydra overrides given as positional args are forwarded.
# =============================================================================

# Run from the repo root with the appropriate conda environment already
# activated.

TASKS_CONFIG=${TASKS_CONFIG:-configs/multi_task_multi_episode_config.yaml}

if [ ! -f "$TASKS_CONFIG" ]; then
    echo "Error: tasks config not found: $TASKS_CONFIG" >&2
    exit 1
fi

# Summarization knobs
MODE=${MODE:-episodic}
SUMMARIZATION_THRESHOLD=${SUMMARIZATION_THRESHOLD:-16384}
SUMMARY_MAX_TOKENS=${SUMMARY_MAX_TOKENS:-4096}
MAX_SUMMARIZATIONS=${MAX_SUMMARIZATIONS:-5}

case "$MODE" in
    token|episodic|both) ;;
    *)
        echo "Error: MODE must be one of {token, episodic, both} (got '$MODE')" >&2
        exit 1
        ;;
esac

SUMM=(
    rllm.agent.name=gem_text_agent_summarizing
    +rllm.agent.summarization.enable=true
    +rllm.agent.summarization.mode="$MODE"
    +rllm.agent.summarization.threshold_tokens=$SUMMARIZATION_THRESHOLD
    +rllm.agent.summarization.summary_max_tokens=$SUMMARY_MAX_TOKENS
    +rllm.agent.summarization.max_summarizations=$MAX_SUMMARIZATIONS
)

echo "=========================================="
echo "Training summarizing agent"
echo "  TASKS_CONFIG = $TASKS_CONFIG"
echo "  MODE         = $MODE"
echo "  threshold    = $SUMMARIZATION_THRESHOLD"
echo "  summary_max  = $SUMMARY_MAX_TOKENS"
echo "  max_summ     = $MAX_SUMMARIZATIONS"
echo "=========================================="

TASKS_CONFIG="$TASKS_CONFIG" \
bash scripts/train_multi_task_multi_episode.sh \
    "${SUMM[@]}" \
    "$@"
