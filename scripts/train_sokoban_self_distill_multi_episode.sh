#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:False"}
export VLLM_USE_V1=1

# Multi-task Sokoban config.
TASKS_CONFIG=${TASKS_CONFIG:-configs/multi_task_sokoban_self_distill_multi_episode.yaml}
if [ ! -f "$TASKS_CONFIG" ]; then
    echo "Error: Tasks config file not found: $TASKS_CONFIG"
    echo "Set TASKS_CONFIG to a valid YAML path."
    exit 1
fi

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
ACTOR_LR=${ACTOR_LR:-1e-5}
DISTILL_LAMBDA=${DISTILL_LAMBDA:-1.0}
# Distillation mode options: sdpo_self, sdpo_pure.
DISTILL_MODE=${DISTILL_MODE:-sdpo_self}
DISTILL_TRAJECTORY_SELECTION=${DISTILL_TRAJECTORY_SELECTION:-selective_retry_success_n2}
TEACHER_CONTEXT_ATTEMPTS=${TEACHER_CONTEXT_ATTEMPTS:-1}
# Teacher sync options: none, ema, every_n_steps.
TEACHER_REGULARIZATION=${TEACHER_REGULARIZATION:-ema}
TEACHER_UPDATE_RATE=${TEACHER_UPDATE_RATE:-0.05}
TEACHER_UPDATE_INTERVAL=${TEACHER_UPDATE_INTERVAL:-25}
MIN_DISTILL_TOKENS=${MIN_DISTILL_TOKENS:-1}
# Distillation loss options: non_full, full_logit.
DISTILL_LOSS_VARIANT=${DISTILL_LOSS_VARIANT:-full_logit}
DISTILL_IS_CLIP=${DISTILL_IS_CLIP:-2.0}
FULL_LOGIT_TOPK=${FULL_LOGIT_TOPK:-64}
FULL_LOGIT_ADD_TAIL=${FULL_LOGIT_ADD_TAIL:-True}
NEGATE_SDPO_LOSS=${NEGATE_SDPO_LOSS:-False}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}

if [ "$DISTILL_LOSS_VARIANT" = "full_logit" ]; then
    DISTILL_ALPHA=${DISTILL_ALPHA:-0.5}
    ACTOR_USE_REMOVE_PADDING=${ACTOR_USE_REMOVE_PADDING:-True}
else
    DISTILL_ALPHA=${DISTILL_ALPHA:-1.0}
    ACTOR_USE_REMOVE_PADDING=${ACTOR_USE_REMOVE_PADDING:-True}
fi

if [ "$DISTILL_LOSS_VARIANT" != "non_full" ] && [ "$DISTILL_LOSS_VARIANT" != "full_logit" ]; then
    echo "Error: DISTILL_LOSS_VARIANT must be one of: non_full, full_logit"
    exit 1
fi

if [ "$DISTILL_MODE" != "sdpo_self" ] && [ "$DISTILL_MODE" != "sdpo_pure" ]; then
    echo "Error: DISTILL_MODE must be one of: sdpo_self, sdpo_pure"
    exit 1
fi

if [ "$DISTILL_TRAJECTORY_SELECTION" != "first_attempt_hindsight" ] && [ "$DISTILL_TRAJECTORY_SELECTION" != "selective_retry_success_n2" ]; then
    echo "Error: DISTILL_TRAJECTORY_SELECTION must be one of: first_attempt_hindsight, selective_retry_success_n2"
    exit 1
fi

if [ "$DISTILL_TRAJECTORY_SELECTION" = "selective_retry_success_n2" ] && [ "$TEACHER_CONTEXT_ATTEMPTS" != "1" ]; then
    echo "Error: TEACHER_CONTEXT_ATTEMPTS must be 1 when DISTILL_TRAJECTORY_SELECTION=selective_retry_success_n2"
    exit 1
fi

if [ "$DISTILL_LOSS_VARIANT" = "non_full" ]; then
    case "$DISTILL_ALPHA" in
        1|1.0|1.00|1.000|1.0000) ;;
        *)
            echo "Error: DISTILL_ALPHA must be 1.0 when DISTILL_LOSS_VARIANT=non_full (got $DISTILL_ALPHA)"
            exit 1
            ;;
    esac
fi

CONFIG_NAME=$(basename "$TASKS_CONFIG" .yaml | tr '[:upper:]' '[:lower:]' | tr '_' '-')
ENABLE_SELF_DISTILL=${ENABLE_SELF_DISTILL:-True}
ENABLE_REFLECTION=${ENABLE_REFLECTION:-True}
REFLECTION_PROMPT=${REFLECTION_PROMPT:-}
ENV_NAME="sokoban"

append_experiment_tag() {
    local tag="$1"
    if [ -z "$tag" ]; then
        return
    fi
    if [ -n "$TRAINING_CONFIG_NAME" ]; then
        TRAINING_CONFIG_NAME="${TRAINING_CONFIG_NAME}-${tag}"
    else
        TRAINING_CONFIG_NAME="$tag"
    fi
}

if [ "$TEACHER_REGULARIZATION" = "ema" ]; then
    TEACHER_REG_TAG="teacher-ema"
elif [ "$TEACHER_REGULARIZATION" = "every_n_steps" ]; then
    TEACHER_REG_TAG="teacher-every-n-steps"
else
    TEACHER_REG_TAG="teacher-none"
fi

if [ "$ENABLE_REFLECTION" = True ] || [ "$ENABLE_REFLECTION" = "true" ]; then
    REFLECTION_TAG="reflection-on"
else
    REFLECTION_TAG="reflection-off"
fi

TASK_CONFIG_NAME=${CONFIG_NAME#multi-task-}
TASK_CONFIG_NAME=${TASK_CONFIG_NAME#${ENV_NAME}-}
TASK_CONFIG_NAME=${TASK_CONFIG_NAME%-self-distill-multi-episode}
TASK_CONFIG_NAME=${TASK_CONFIG_NAME%-multi-episode}
TASK_CONFIG_NAME=${TASK_CONFIG_NAME%-self-distill}

TRAINING_CONFIG_NAME=""
append_experiment_tag "$TASK_CONFIG_NAME"
append_experiment_tag "$REFLECTION_TAG"
if [ "$ENABLE_SELF_DISTILL" = True ]; then
    append_experiment_tag "$DISTILL_MODE"
    append_experiment_tag "$DISTILL_TRAJECTORY_SELECTION"
    append_experiment_tag "$DISTILL_LOSS_VARIANT"
    append_experiment_tag "a${DISTILL_ALPHA}"
    append_experiment_tag "l${DISTILL_LAMBDA}"
    append_experiment_tag "$TEACHER_REG_TAG"
    if [ "$NEGATE_SDPO_LOSS" = True ] || [ "$NEGATE_SDPO_LOSS" = "true" ]; then
        append_experiment_tag "negate"
    fi
else
    append_experiment_tag "no-distill"
fi

EXPERIMENT_NAME=${EXPERIMENT_NAME:-"${ENV_NAME}-${TRAINING_CONFIG_NAME}"}

EXTRA_ENV_ARGS=()
if [ -n "$REFLECTION_PROMPT" ]; then
    EXTRA_ENV_ARGS+=(+rllm.env.env_args.reflection_prompt="$REFLECTION_PROMPT")
fi

# Enum options for overrides used in the command below:
# - rllm.distill.denominator_mode: teacher_adapted_feedback (currently the only supported mode in this trainer).
# - rllm.distill.context_overflow_policy: skip_loss (currently the only supported policy).
# - actor_rollout_ref.actor.loss_agg_mode: token-mean, seq-mean-token-sum, seq-mean-token-mean, seq-mean-token-sum-norm.
python scripts/train_multi_episode.py \
    data.train_batch_size=32 \
    data.val_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=15360 \
    +data.tasks_config_path="$TASKS_CONFIG" \
    +rllm.env.env_args.success_reward=1.0 \
    +rllm.env.env_args.episode_header="New episode begins." \
    +rllm.env.env_args.enable_reflection=$ENABLE_REFLECTION \
    "${EXTRA_ENV_ARGS[@]}" \
    rllm.distill.enable=$ENABLE_SELF_DISTILL \
    +rllm.distill.lambda=$DISTILL_LAMBDA \
    +rllm.distill.mode=$DISTILL_MODE \
    +rllm.distill.trajectory_selection=$DISTILL_TRAJECTORY_SELECTION \
    +rllm.distill.context_limit=32768 \
    +rllm.distill.denominator_mode=teacher_adapted_feedback \
    +rllm.distill.context_overflow_policy=skip_loss \
    +rllm.distill.min_distill_tokens=$MIN_DISTILL_TOKENS \
    +rllm.distill.teacher_context_attempts=$TEACHER_CONTEXT_ATTEMPTS \
    +rllm.distill.loss_variant=$DISTILL_LOSS_VARIANT \
    +rllm.distill.alpha=$DISTILL_ALPHA \
    +rllm.distill.is_clip=$DISTILL_IS_CLIP \
    +rllm.distill.full_logit_topk=$FULL_LOGIT_TOPK \
    +rllm.distill.full_logit_add_tail=$FULL_LOGIT_ADD_TAIL \
    +rllm.distill.negate_sdpo_loss=$NEGATE_SDPO_LOSS \
    ++rllm.distill.teacher_regularization=$TEACHER_REGULARIZATION \
    ++rllm.distill.teacher_update_rate=$TEACHER_UPDATE_RATE \
    ++rllm.distill.teacher_update_interval=$TEACHER_UPDATE_INTERVAL \
    rllm.disable_thinking=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.model.use_remove_padding=$ACTOR_USE_REMOVE_PADDING \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    rllm.compact_filtering.enable=False \
    rllm.compact_filtering.mask_max_prompt_length_exceeded=True \
    rllm.compact_filtering.mask_max_response_length_exceeded=True \
    rllm.compact_filtering.mask_max_turns_exceeded=False \
    rllm.compact_filtering.mask_timeout=True \
    rllm.rejection_sample.enable=False \
    rllm.rejection_sample.multiplier=1.0 \
    rllm.stepwise_advantage.enable=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rllm-agent' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=10 \
    "$@"
