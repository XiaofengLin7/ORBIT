#!/bin/bash
set -x

# Suppress core dumps. Ray's auxiliary `dashboard-ReportHead` workers crash
# periodically (Ray-version-dependent telemetry bug) and otherwise leave
# 700 MB+ ELF core files in the cwd on every crash, polluting `git status`
# and chewing through disk for no diagnostic value (training itself is fine).
# Scoped to this shell — does not affect other sessions.
ulimit -c 0

# GPU assignment: use 4 GPUs (adjust if vLLM is using one; e.g., skip GPU0 if needed).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1

# Don't forward the driver's CUDA_VISIBLE_DEVICES into Ray's runtime_env.
# rllm's `_get_forwarded_env_vars` (third_party/rllm/rllm/trainer/verl/ray_runtime_env.py)
# auto-forwards anything starting with `CUDA_`, which clobbers Ray's
# per-worker GPU assignment when `n_gpus_per_node > 1`. With the
# cluster-wide value forced into every worker, the FSDP `init_device_mesh`
# binds workers to overlapping devices and fails with
# "CUDA-capable device(s) is/are busy or unavailable" inside
# `verl/workers/fsdp_workers.py:create_device_mesh`. Excluding it lets
# Ray set per-actor `CUDA_VISIBLE_DEVICES` correctly.
export RLLM_EXCLUDE=${RLLM_EXCLUDE:-CUDA_VISIBLE_DEVICES}

# Install the trajectory-uniform actor patch into the active conda env's
# site-packages via a .pth file. Python auto-loads .pth files at startup
# (via the `site` module) in EVERY Python process — including Ray's FSDP
# worker processes. This is how we get our `TrajectoryUniformPPOActor`
# to replace verl's stock `DataParallelPPOActor` in workers, without
# going through `runtime_env.worker_process_setup_hook` (which on this
# SCC cluster breaks Ray's per-worker GPU isolation — confirmed via
# empty-hook bisection).
#
# Idempotent: rewrites the .pth file every launch with the current
# repo root and the current env's site-packages. Also makes this
# zero-setup for someone cloning the repo on a different machine —
# they just `conda activate <env>` and run the launch script; the .pth
# is written into their env automatically.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
if [ -n "$SITE_PACKAGES" ] && [ -d "$SITE_PACKAGES" ]; then
    cat > "$SITE_PACKAGES/orbit_segtrain.pth" <<EOF
$REPO_ROOT
import orbit_segtrain_patch
EOF
    echo "[orbit] Installed trajectory-uniform actor patch at $SITE_PACKAGES/orbit_segtrain.pth"
else
    echo "[orbit] WARNING: could not locate active site-packages; trajectory-uniform actor patch NOT installed. Workers will fall back to seq-mean-token-mean."
fi

# Multi-task configuration: path to YAML config file (required)
TASKS_CONFIG=${TASKS_CONFIG:-configs/multi_task_multi_episode_config.yaml}

if [ ! -f "$TASKS_CONFIG" ]; then
    echo "Error: Tasks config file not found: $TASKS_CONFIG"
    echo "Please set TASKS_CONFIG environment variable to point to a valid YAML config file."
    exit 1
fi

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}

# Extract model name (last part after /)
MODEL_NAME=$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')

# Construct experiment name from config file name
CONFIG_NAME=$(basename "$TASKS_CONFIG" .yaml | tr '[:upper:]' '[:lower:]' | tr '_' '-')
EXPERIMENT_NAME="gem-multi-task-${CONFIG_NAME}-${MODEL_NAME}"

# Multi-episode via environment wrapper (uses AgentExecutionEngine instead of workflow)
python scripts/train_multi_episode.py \
    data.train_batch_size=32 \
    data.val_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=31744 \
    +data.tasks_config_path="$TASKS_CONFIG" \
    +rllm.env.env_args.success_reward=1.0 \
    +rllm.env.env_args.episode_header="New episode begins." \
    rllm.agent.max_steps=50 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
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

