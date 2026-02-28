#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1

# Multi-task FrozenLake config.
# This config defines distinct train/val task distributions and leaves `desc`
# unset so each trajectory varies by seed-generated random map.
TASKS_CONFIG=${TASKS_CONFIG:-configs/multi_task_frozenlake_self_distill_multi_episode.yaml}
if [ ! -f "$TASKS_CONFIG" ]; then
    echo "Error: Tasks config file not found: $TASKS_CONFIG"
    echo "Set TASKS_CONFIG to a valid YAML path."
    exit 1
fi

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
DISTILL_LAMBDA=${DISTILL_LAMBDA:-0.1}
TEACHER_CONTEXT_ATTEMPTS=${TEACHER_CONTEXT_ATTEMPTS:-2}
MIN_DISTILL_TOKENS=${MIN_DISTILL_TOKENS:-1}

MODEL_NAME=$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')
CONFIG_NAME=$(basename "$TASKS_CONFIG" .yaml | tr '[:upper:]' '[:lower:]' | tr '_' '-')
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"frozenlake-multi-episode-${CONFIG_NAME}-${MODEL_NAME}"}

python scripts/train_multi_episode.py \
    data.train_batch_size=32 \
    data.val_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=16384 \
    +data.tasks_config_path="$TASKS_CONFIG" \
    +rllm.env.env_args.success_reward=1.0 \
    +rllm.env.env_args.episode_header="New episode begins." \
    +rllm.env.env_args.enable_reflection=False \
    rllm.distill.enable=False \
    +rllm.distill.lambda=$DISTILL_LAMBDA \
    +rllm.distill.mode=sdpo_self \
    +rllm.distill.context_limit=32768 \
    +rllm.distill.denominator_mode=teacher_adapted_feedback \
    +rllm.distill.context_overflow_policy=skip_loss \
    +rllm.distill.min_distill_tokens=$MIN_DISTILL_TOKENS \
    +rllm.distill.teacher_context_attempts=$TEACHER_CONTEXT_ATTEMPTS \
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
