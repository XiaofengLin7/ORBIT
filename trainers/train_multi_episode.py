"""Custom training entry point for multi-episode training with metrics logging.

This module provides a TaskRunner that uses MultiEpisodeAgentPPOTrainer
instead of the default AgentPPOTrainer, enabling extraction of multi-episode
metrics without modifying vendor code.
"""

from __future__ import annotations

import os
import socket
from typing import Any

import ray
from omegaconf import OmegaConf
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.device import is_cuda_available

from rllm.trainer.env_agent_mappings import AGENT_CLASS_MAPPING, ENV_CLASS_MAPPING
from rllm.trainer.verl.agent_workflow_trainer import AgentWorkflowPPOTrainer
from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env

# Import our custom trainer
from trainers.multi_episode_trainer import MultiEpisodeAgentPPOTrainer
from trainers.sdpo_self_distill_trainer import JointSDPOSelfDistillTrainer


def run_ppo_agent(
    config: Any,
    env_class: type | None = None,
    agent_class: type | None = None,
    env_args: dict | None = None,
    agent_args: dict | None = None,
    val_env_class: type | None = None,
    val_env_args: dict | None = None,
) -> None:
    """Run PPO agent training with custom multi-episode trainer.

    Args:
        config: Training configuration object.
        env_class: Optional environment class (uses config mapping if not provided).
        agent_class: Optional agent class (uses config mapping if not provided).
        env_args: Optional environment arguments.
        agent_args: Optional agent arguments.
        val_env_class: Optional validation environment class. If provided, uses
            this class instead of env_class during validation.
        val_env_args: Optional validation environment arguments. If provided,
            these override env_args during validation.
    """
    if not ray.is_initialized():
        if config is not None and hasattr(config, "ray_init"):
            ray_init_settings = {k: v for k, v in config.ray_init.items() if v is not None}
        else:
            ray_init_settings = {}
        ray.init(runtime_env=get_ppo_ray_runtime_env(), **ray_init_settings)

    if (
        is_cuda_available
        and config.trainer.get("profile_steps") is not None
        and len(config.trainer.get("profile_steps", [])) > 0
    ):
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = MultiEpisodeTaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = MultiEpisodeTaskRunner.remote()

    ray.get(
        runner.run.remote(
            config,
            env_class=env_class,
            agent_class=agent_class,
            env_args=env_args,
            agent_args=agent_args,
            val_env_class=val_env_class,
            val_env_args=val_env_args,
        )
    )

    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)
class MultiEpisodeTaskRunner:
    """Ray remote class for executing multi-episode PPO training.

    Uses MultiEpisodeAgentPPOTrainer for proper metrics extraction.
    """

    def run(
        self,
        config: Any,
        workflow_class: type | None = None,
        workflow_args: dict | None = None,
        agent_class: type | None = None,
        env_class: type | None = None,
        agent_args: dict | None = None,
        env_args: dict | None = None,
        agent_run_func: Any | None = None,
        val_env_class: type | None = None,
        val_env_args: dict | None = None,
    ) -> None:
        """Execute the main PPO training workflow with multi-episode support."""
        from pprint import pprint

        from verl.single_controller.ray import RayWorkerGroup
        from verl.utils.fs import copy_to_local
        from verl.utils import hf_processor, hf_tokenizer
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        print(f"MultiEpisodeTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        OmegaConf.register_new_resolver("mul", lambda x, y: int(x) * int(y))
        OmegaConf.resolve(config)
        pprint(OmegaConf.to_container(config))

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        ray_worker_group_cls = RayWorkerGroup

        # Set up workers based on strategy (matching vendor code pattern)
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import CriticWorker
                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.megatron_workers import (
                ActorRolloutRefWorker,
                AsyncActorRolloutRefWorker,
                CriticWorker,
            )

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
        else:
            raise NotImplementedError(f"Unsupported strategy: {config.actor_rollout_ref.actor.strategy}")

        # Map roles to their corresponding remote worker classes
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        # Define the resource pool specification
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # Add a reference policy worker if KL loss or KL reward is used
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # Load the reward manager
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # Determine which trainer to use
        use_workflow = (
            workflow_class is not None
            or (
                hasattr(config, "rllm")
                and hasattr(config.rllm, "workflow")
                and config.rllm.workflow.get("use_workflow", False)
            )
        )

        if use_workflow:
            # Use workflow trainer (not modified)
            workflow_args = workflow_args or {}
            if config.rllm.workflow.get("workflow_args") is not None:
                for key, value in config.rllm.workflow.get("workflow_args").items():
                    if value is not None:
                        if key in workflow_args and isinstance(workflow_args[key], dict):
                            workflow_args[key].update(value)
                        else:
                            workflow_args[key] = value

            trainer = AgentWorkflowPPOTrainer(
                config=config,
                tokenizer=tokenizer,
                processor=processor,
                role_worker_mapping=role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
                workflow_class=workflow_class,
                workflow_args=workflow_args,
            )
        else:
            # Use our custom multi-episode trainer
            if env_class is None:
                env_class = ENV_CLASS_MAPPING[config.rllm.env.name]
            if agent_class is None:
                agent_class = AGENT_CLASS_MAPPING[config.rllm.agent.name]

            env_args = env_args or {}
            agent_args = agent_args or {}
            if config.rllm.env.get("env_args") is not None:
                env_args.update(config.rllm.env.get("env_args"))
            if config.rllm.agent.get("agent_args") is not None:
                agent_args.update(config.rllm.agent.get("agent_args"))

            use_distill = bool(
                hasattr(config, "rllm")
                and config.rllm.get("distill") is not None
                and config.rllm.distill.get("enable", False)
            )
            trainer_cls = JointSDPOSelfDistillTrainer if use_distill else MultiEpisodeAgentPPOTrainer

            # Use custom trainer instead of AgentPPOTrainer
            trainer = trainer_cls(
                config=config,
                tokenizer=tokenizer,
                role_worker_mapping=role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
                env_class=env_class,
                agent_class=agent_class,
                env_args=env_args,
                agent_args=agent_args,
                val_env_class=val_env_class,
                val_env_args=val_env_args,
            )

        trainer.init_workers()
        try:
            trainer.fit_agent()
        finally:
            trainer.shutdown()
