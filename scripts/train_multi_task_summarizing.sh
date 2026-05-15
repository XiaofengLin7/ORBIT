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
#                              (default: configs/multi_task_summarization_config.yaml)
#   MODEL_PATH                 HF id or local path (default: Qwen/Qwen3-4B,
#                              i.e. inherits the underlying script's default)
#   MODE                       summarization mode: token | episodic | both
#                              (default: episodic)
#   SUMMARIZATION_THRESHOLD    token threshold for mode=token | both
#                              (default: 16384)
#   SUMMARY_MAX_TOKENS         summary completion budget
#                              (default: 4096)
#   CARRYOVER                  episode-end carryover form
#                              freeform | obs_action | obs_action_reflection
#                              (default: obs_action_reflection). Used as the
#                              default experiment_name unless EXP_NAME is set.
#   EXP_NAME                   override the wandb experiment_name (default: $CARRYOVER)
#
# Any additional Hydra overrides given as positional args are forwarded.
# =============================================================================

# Run from the repo root with the appropriate conda environment already
# activated.

TASKS_CONFIG=${TASKS_CONFIG:-configs/multi_task_summarization_config.yaml}

if [ ! -f "$TASKS_CONFIG" ]; then
    echo "Error: tasks config not found: $TASKS_CONFIG" >&2
    exit 1
fi

# Summarization knobs
MODE=${MODE:-episodic}
SUMMARIZATION_THRESHOLD=${SUMMARIZATION_THRESHOLD:-16384}
SUMMARY_MAX_TOKENS=${SUMMARY_MAX_TOKENS:-4096}

case "$MODE" in
    token|episodic|both) ;;
    *)
        echo "Error: MODE must be one of {token, episodic, both} (got '$MODE')" >&2
        exit 1
        ;;
esac

CARRYOVER=${CARRYOVER:-obs_action_reflection}
case "$CARRYOVER" in
    freeform|obs_action|obs_action_reflection) ;;
    *)
        echo "Error: CARRYOVER must be one of {freeform, obs_action, obs_action_reflection} (got '$CARRYOVER')" >&2
        exit 1
        ;;
esac

EXP_NAME=${EXP_NAME:-$CARRYOVER}

SUMM=(
    rllm.agent.name=gem_text_agent_summarizing
    +rllm.agent.summarization.enable=true
    +rllm.agent.summarization.mode="$MODE"
    +rllm.agent.summarization.threshold_tokens=$SUMMARIZATION_THRESHOLD
    +rllm.agent.summarization.summary_max_tokens=$SUMMARY_MAX_TOKENS
    +rllm.agent.summarization.episodic_carryover="$CARRYOVER"
    trainer.experiment_name="$EXP_NAME"
)

echo "=========================================="
echo "Training summarizing agent"
echo "  TASKS_CONFIG = $TASKS_CONFIG"
echo "  MODE         = $MODE"
echo "  CARRYOVER    = $CARRYOVER"
echo "  EXP_NAME     = $EXP_NAME"
echo "  threshold    = $SUMMARIZATION_THRESHOLD"
echo "  summary_max  = $SUMMARY_MAX_TOKENS"
echo "=========================================="

TASKS_CONFIG="$TASKS_CONFIG" \
bash scripts/train_multi_task_multi_episode.sh \
    "${SUMM[@]}" \
    "$@"
