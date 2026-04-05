#!/bin/bash
set -x

# Default GPU setup (override as needed, e.g., export CUDA_VISIBLE_DEVICES=0,1)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1

# Environment and experiment defaults
TOTAL_STEP_CAP=${TOTAL_STEP_CAP:-21}
MAX_TURNS_PER_EPISODE=${MAX_TURNS_PER_EPISODE:-7}
GRID_N=${GRID_N:-4}
GRID_M=${GRID_M:-4}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}

# Names for logging
MODEL_NAME=$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')
EXPERIMENT_NAME="multi-episode-train-grid-${MODEL_NAME}"

# Eval-only run for Grid via the multi-episode environment wrapper.
# You can override any of these hydra args on the command line after "$@".
python scripts/train_multi_episode.py \
    data.train_batch_size=32 \
    data.val_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=32768 \
    +rllm.env.env_args.inner_env_class=envs.grid_env_adapter.GridEnvAdapter \
    +rllm.env.env_args.inner_env_kwargs.env_id=game:Grid-v0 \
    +rllm.env.env_args.inner_env_kwargs.n=$GRID_N \
    +rllm.env.env_args.inner_env_kwargs.m=$GRID_M \
    +rllm.env.env_args.inner_env_kwargs.max_turns=$MAX_TURNS_PER_EPISODE \
    +rllm.env.env_args.total_step_cap=$TOTAL_STEP_CAP \
    +rllm.env.env_args.success_reward=1.0 \
    rllm.agent.max_steps=$TOTAL_STEP_CAP \
    rllm.disable_thinking=False \
    +rllm.env.env_args.episode_header="New episode begins." \
    +rllm.env.env_args.enable_reflection=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=33792 \
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
    trainer.project_name='grid-env' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100000 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=0 \
    "$@"
