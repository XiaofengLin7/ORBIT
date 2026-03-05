#!/bin/bash
set -x

# GPU assignment: use 4 GPUs (adjust if vLLM is using one; e.g., skip GPU0 if needed).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1

# Available GEM Tower of Hanoi env_ids:
# - game:TowerofHanoi-v0-easy   (3 disks, 10 turns)
# - game:TowerofHanoi-v0-hard   (5 disks, 35 turns)
# - game:TowerofHanoi-v0-random (randomized)
ENV_ID=${ENV_ID:-game:TowerofHanoi-v0-easy}
TOTAL_STEP_CAP=${TOTAL_STEP_CAP:-30}
MAX_TURNS_PER_EPISODE=${MAX_TURNS_PER_EPISODE:-10}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-1.7B}
DISTILL_LAMBDA=${DISTILL_LAMBDA:-1.0}
DISTILL_MODE=${DISTILL_MODE:-sdpo_self}
TEACHER_CONTEXT_ATTEMPTS=${TEACHER_CONTEXT_ATTEMPTS:-2}
TEACHER_REGULARIZATION=${TEACHER_REGULARIZATION:-ema}
TEACHER_UPDATE_RATE=${TEACHER_UPDATE_RATE:-0.05}
TEACHER_UPDATE_INTERVAL=${TEACHER_UPDATE_INTERVAL:-10}
MIN_DISTILL_TOKENS=${MIN_DISTILL_TOKENS:-1}

# Extract model name (last part after /)
MODEL_NAME=$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')
# Extract env name (part after :, convert to lowercase with hyphens)
ENV_NAME=$(echo "$ENV_ID" | cut -d: -f2 | tr '[:upper:]' '[:lower:]' | tr '_' '-')

if [ "$TEACHER_REGULARIZATION" = "ema" ]; then
    TEACHER_REG_SUFFIX="teacher-ema-a-${TEACHER_UPDATE_RATE}"
elif [ "$TEACHER_REGULARIZATION" = "every_n_steps" ]; then
    TEACHER_REG_SUFFIX="teacher-every-${TEACHER_UPDATE_INTERVAL}"
else
    TEACHER_REG_SUFFIX="teacher-none"
fi

if [ "$DISTILL_MODE" != "sdpo_self" ] && [ "$DISTILL_MODE" != "sdpo_pure" ]; then
    echo "Error: DISTILL_MODE must be one of: sdpo_self, sdpo_pure"
    exit 1
fi

# Construct experiment name
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"gem-${ENV_NAME}-self-distill-multi-episode-${MODEL_NAME}-${TEACHER_REG_SUFFIX}"}

# Multi-episode + SDPO self-distillation.
python scripts/train_multi_episode.py \
    data.train_batch_size=32 \
    data.val_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=16384 \
    +rllm.env.env_args.inner_env_class=envs.gem_env_adapter.GEMEnvAdapter \
    +rllm.env.env_args.inner_env_kwargs.env_id=$ENV_ID \
    +rllm.env.env_args.inner_env_kwargs.env_kwargs.max_turns=$MAX_TURNS_PER_EPISODE \
    +rllm.env.env_args.total_step_cap=$TOTAL_STEP_CAP \
    +rllm.env.env_args.success_reward=1.0 \
    rllm.agent.max_steps=$TOTAL_STEP_CAP \
    +rllm.env.env_args.episode_header="New episode begins." \
    +rllm.env.env_args.enable_reflection=False \
    rllm.distill.enable=True \
    +rllm.distill.lambda=$DISTILL_LAMBDA \
    +rllm.distill.mode=$DISTILL_MODE \
    +rllm.distill.context_limit=32768 \
    +rllm.distill.denominator_mode=teacher_adapted_feedback \
    +rllm.distill.context_overflow_policy=skip_loss \
    +rllm.distill.min_distill_tokens=$MIN_DISTILL_TOKENS \
    +rllm.distill.teacher_context_attempts=$TEACHER_CONTEXT_ATTEMPTS \
    ++rllm.distill.teacher_regularization=$TEACHER_REGULARIZATION \
    ++rllm.distill.teacher_update_rate=$TEACHER_UPDATE_RATE \
    ++rllm.distill.teacher_update_interval=$TEACHER_UPDATE_INTERVAL \
    rllm.disable_thinking=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
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
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=10 \
    "$@"
