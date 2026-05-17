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
#   ADVANTAGE_METHOD           "grpo" (default) or "chunk_discounted_topr".
#                              Under TOPR the additional knobs below apply:
#     GAMMA                    chunk discount factor (default: 0.95)
#     REWARD_SCOPE             "per_episode" (default) or "terminal"
#     TOPR_SPLIT               "true" (default) or "false" — TOPR pos/neg
#                              split. False = chunk-discounted PPO without
#                              the TOPR REINFORCE branch.
#
# Any additional Hydra overrides given as positional args are forwarded
# AFTER the toggle-derived overrides, so they win on conflict (Hydra's
# last-override-wins semantics for +key=value).
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

ADVANTAGE_METHOD=${ADVANTAGE_METHOD:-grpo}
GAMMA=${GAMMA:-0.95}
REWARD_SCOPE=${REWARD_SCOPE:-per_episode}
TOPR_SPLIT=${TOPR_SPLIT:-true}

case "$ADVANTAGE_METHOD" in
    chunk_discounted_topr)
        ADV_OVERRIDES=(
            +rllm.advantage_method.name=chunk_discounted_topr
            +rllm.advantage_method.chunk_discounted_topr.reward_scope=$REWARD_SCOPE
            +rllm.advantage_method.chunk_discounted_topr.gamma=$GAMMA
            +rllm.advantage_method.chunk_discounted_topr.topr_split.enable=$TOPR_SPLIT
        )
        _ADV_TAG="topr"
        ;;
    grpo)
        # GRPO is the parent trainer's default. We pass the name
        # explicitly for log clarity; no other overrides since GRPO has
        # no per-chunk discount / TOPR split.
        ADV_OVERRIDES=(+rllm.advantage_method.name=grpo)
        _ADV_TAG="grpo"
        ;;
    *)
        echo "Error: ADVANTAGE_METHOD must be chunk_discounted_topr or grpo (got '$ADVANTAGE_METHOD')" >&2
        exit 1
        ;;
esac

# Tag the experiment name with the advantage method so TOPR / GRPO runs
# are distinguishable in wandb. Only applies when EXP_NAME wasn't already
# set by the caller (e.g. the maze script's own EXP_NAME default wins).
EXP_NAME=${EXP_NAME:-${CARRYOVER}_${_ADV_TAG}}

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
echo "  TASKS_CONFIG     = $TASKS_CONFIG"
echo "  MODE             = $MODE"
echo "  CARRYOVER        = $CARRYOVER"
echo "  EXP_NAME         = $EXP_NAME"
echo "  threshold        = $SUMMARIZATION_THRESHOLD"
echo "  summary_max      = $SUMMARY_MAX_TOKENS"
echo "  ADVANTAGE_METHOD = $ADVANTAGE_METHOD"
echo "=========================================="

TASKS_CONFIG="$TASKS_CONFIG" \
bash scripts/train_multi_task_multi_episode.sh \
    "${SUMM[@]}" \
    "${ADV_OVERRIDES[@]}" \
    "$@"
