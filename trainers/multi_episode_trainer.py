"""Custom trainer that extracts multi-episode metrics.

This trainer extends AgentPPOTrainer to extract additional metrics from
environments that have a get_metrics() method, such as MultiEpisodeEnv.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto  # type: ignore

from rllm.engine.agent_execution_engine import AsyncAgentExecutionEngine
from rllm.trainer.verl.agent_ppo_trainer import AgentPPOTrainer


class MultiEpisodeAsyncAgentExecutionEngine(AsyncAgentExecutionEngine):
    """Execution engine that extracts metrics from environments with get_metrics()."""

    async def run_agent_trajectory_async(self, idx, application_id, seed=0, mode="Text", **kwargs):
        """Run trajectory and include environment metrics if available."""
        # Store reference to env before it might be modified
        env = self.envs[idx]

        # Call parent method
        result = await super().run_agent_trajectory_async(
            idx, application_id, seed=seed, mode=mode, **kwargs
        )

        # If Token mode, extract additional metrics from environment
        if mode == "Token" and isinstance(result, dict) and "metrics" in result:
            if hasattr(env, "get_metrics") and callable(env.get_metrics):
                try:
                    env_metrics = env.get_metrics()
                    if isinstance(env_metrics, dict):
                        # Flatten nested keys and add to metrics
                        for key, value in env_metrics.items():
                            flat_key = key.replace("/", "_")
                            result["metrics"][flat_key] = value
                except Exception:
                    pass  # Silently ignore if metrics extraction fails

        return result


class MultiEpisodeAgentPPOTrainer(AgentPPOTrainer):
    """Trainer that uses MultiEpisodeAsyncAgentExecutionEngine for metrics extraction.

    Supports separate environment classes for training and validation, enabling
    scenarios like training with SingleEpisodeEnv but validating with MultiEpisodeEnv.
    """

    def __init__(
        self,
        val_env_class: type | None = None,
        val_env_args: dict | None = None,
        **kwargs,
    ):
        """Initialize the trainer with optional validation environment class.

        Args:
            val_env_class: Optional validation environment class. If provided,
                uses this class instead of env_class during validation.
            val_env_args: Optional validation environment arguments. If provided,
                these override env_args during validation.
            **kwargs: Arguments passed to parent AgentPPOTrainer.
        """
        super().__init__(**kwargs)
        self.val_env_class = val_env_class
        self.val_env_args = val_env_args
        self._is_validation_mode = False

    def init_workers(self):
        """Initialize workers with custom execution engine."""
        # Call grandparent's init_workers (skip AgentPPOTrainer's)
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        RayPPOTrainer.init_workers(self)

        engine_args = OmegaConf.to_container(self.config.rllm.agent.get("engine_args", {})) or {}
        n_parallel_agents = (
            engine_args.pop("n_parallel_agents", None)
            or self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
        )
        print(f"n_parallel_agents: {n_parallel_agents}")

        # Use our custom execution engine instead
        self.agent_execution_engine = MultiEpisodeAsyncAgentExecutionEngine(
            rollout_engine=self.async_rollout_manager,
            config=self.config,
            engine_name="verl",
            tokenizer=self.tokenizer,
            model_path=self.config.actor_rollout_ref.model.path,
            max_steps=self.config.rllm.agent.max_steps,
            max_response_length=self.config.data.max_response_length,
            max_prompt_length=self.config.data.max_prompt_length,
            agent_class=self.agent_class,
            agent_args=self.agent_args,
            env_class=self.env_class,
            env_args=self.env_args,
            enforce_max_prompt_length=self.config.rllm.stepwise_advantage.enable,
            trajectory_timeout=self.config.rllm.agent.trajectory_timeout,
            overlong_filter=self.config.rllm.agent.get("overlong_filter", False),
            disable_thinking=self.config.rllm.disable_thinking,
            n_parallel_agents=n_parallel_agents,
            **engine_args,
        )

    def _validate_agent(self):
        """Override validation to include environment metrics from MultiEpisodeEnv.

        If val_env_class is set, uses that environment class for validation
        instead of the training env_class.
        """
        # Enable validation mode for init_envs_and_agents
        self._is_validation_mode = True

        rewards_lst = []
        data_source_lst = []
        uid_lst = []
        env_metrics_lst = []  # Collect environment metrics
        env_data_sources_lst = []  # Collect data_sources for each environment
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)
            n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            test_batch.pop(["input_ids", "attention_mask", "position_ids"])  # these are not needed for environment based interaction
            test_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": False,
                "validate": True,
            }
            # Get data_source after repeat (so it matches env order after init_envs_and_agents)
            batch_data_sources = test_batch.non_tensor_batch.get("data_source", ["unknown"] * len(test_batch.batch))
            
            self.init_envs_and_agents(test_batch)

            if self.config.rllm.stepwise_advantage.enable:
                test_output_gen_batch = self.generate_agent_steps(meta_info=test_batch.meta_info, uids=test_batch.non_tensor_batch["uid"])
                # for validation, we only need the last step
                is_last_step = test_output_gen_batch.non_tensor_batch["is_last_step"]
                last_step_indices = np.where(is_last_step == True)[0]
                test_output_gen_batch = test_output_gen_batch.select_idxs(last_step_indices)  # This batch only has last steps
            else:
                test_output_gen_batch, _ = self.generate_agent_trajectory(meta_info=test_batch.meta_info)

            test_batch = test_batch.union(test_output_gen_batch)

            reward_tensor = test_batch.batch["token_level_scores"]

            rewards_lst.append(reward_tensor.sum(-1).cpu())
            data_source_lst.append(batch_data_sources)
            uid_lst.append(test_batch.non_tensor_batch["uid"])

            # Collect environment metrics if available, grouped by data_source
            if hasattr(self.agent_execution_engine, "envs") and self.agent_execution_engine.envs:
                batch_env_metrics = []
                batch_env_data_sources = []
                for idx, env in enumerate(self.agent_execution_engine.envs):
                    if hasattr(env, "get_metrics") and callable(env.get_metrics):
                        try:
                            env_metrics = env.get_metrics()
                            if isinstance(env_metrics, dict):
                                batch_env_metrics.append(env_metrics)
                                # Get data_source for this environment (should match batch order)
                                if idx < len(batch_data_sources):
                                    batch_env_data_sources.append(batch_data_sources[idx])
                                else:
                                    batch_env_data_sources.append("unknown")
                        except Exception:
                            pass  # Silently ignore if metrics extraction fails
                env_metrics_lst.append(batch_env_metrics)
                env_data_sources_lst.append(batch_env_data_sources)

        reward_tensor = torch.cat(rewards_lst, dim=0)  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}

        # to group for pass@k
        uid_tensor = np.concatenate(uid_lst, axis=0)
        data_source_uid_pass_rates = {}  # data source to {uid: pass or not}

        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]

            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

            # pass@k
            if data_source not in data_source_uid_pass_rates:
                data_source_uid_pass_rates[data_source] = {}

            uid = uid_tensor[i]
            if uid not in data_source_uid_pass_rates[data_source]:
                data_source_uid_pass_rates[data_source][uid] = 0  # default to not pass
            # take highest score
            data_source_uid_pass_rates[data_source][uid] = max(data_source_uid_pass_rates[data_source][uid], reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            # clip rewards to be between 0 and 1
            rewards_array = np.array(rewards)
            rewards_array = np.clip(rewards_array, 0, 1)
            metric_dict[f"val/test_score/{data_source}"] = np.mean(rewards_array)

        for data_source, pass_rates in data_source_uid_pass_rates.items():
            pass_k_lst = []
            for uid, pass_score in pass_rates.items():
                pass_k_lst.append(pass_score >= 1)  # assuming 1 means passed
            metric_dict[f"val/test_score/pass@k/{data_source}"] = np.mean(pass_k_lst)

        # Aggregate environment metrics if available, grouped by data_source
        if env_metrics_lst:
            # Flatten list of lists and pair with data_sources
            all_env_metrics_with_sources = []
            for batch_idx, batch_metrics in enumerate(env_metrics_lst):
                batch_data_sources = env_data_sources_lst[batch_idx] if batch_idx < len(env_data_sources_lst) else ["unknown"] * len(batch_metrics)
                for env_idx, metrics in enumerate(batch_metrics):
                    data_source = batch_data_sources[env_idx] if env_idx < len(batch_data_sources) else "unknown"
                    all_env_metrics_with_sources.append((data_source, metrics))
            
            if all_env_metrics_with_sources:
                # Group metrics by data_source
                data_source_metrics = {}
                for data_source, metrics in all_env_metrics_with_sources:
                    if data_source not in data_source_metrics:
                        data_source_metrics[data_source] = []
                    data_source_metrics[data_source].append(metrics)
                
                # Aggregate metrics per data_source
                for data_source, metrics_list in data_source_metrics.items():
                    # Collect all metric keys for this data_source
                    all_keys = set()
                    for metrics in metrics_list:
                        all_keys.update(metrics.keys())
                    
                    # Aggregate each metric (filter out -1 values as they indicate missing episodes)
                    for key in all_keys:
                        values = []
                        for metrics in metrics_list:
                            if key in metrics:
                                value = metrics[key]
                                # Filter out -1 values (missing episodes)
                                if isinstance(value, (int, float)) and value >= 0:
                                    values.append(float(value))
                        if values:
                            # Log with data_source prefix
                            metric_dict[f"val/{data_source}/{key}"] = np.mean(values)

                    # Compute episode_2 - episode_1 success rate gap
                    gaps = []
                    for metrics in metrics_list:
                        ep1 = metrics.get("episode_1/success_rate")
                        ep2 = metrics.get("episode_2/success_rate")
                        if ep1 is not None and ep2 is not None and ep1 >= 0 and ep2 >= 0:
                            gaps.append(ep2 - ep1)
                    if gaps:
                        metric_dict[f"val/{data_source}/episode_improvement"] = np.mean(gaps)

        # Disable validation mode
        self._is_validation_mode = False

        return metric_dict

    def _transform_agent_trajectories(self, trajectories: list[dict]):
        """Override to group traj metrics by data_source."""
        # Call parent method to get the base transformation
        final_gen_batch_output, metrics = super()._transform_agent_trajectories(trajectories)

        # Group traj metrics by data_source
        # Trajectories have an 'idx' field that corresponds to the environment index
        # We can map this to data_source using the stored batch data_sources
        if hasattr(self, "_current_batch_data_sources") and trajectories:
            batch_data_sources = self._current_batch_data_sources
            
            # Group trajectories by data_source based on their idx
            traj_metrics_by_source = {}
            for traj in trajectories:
                traj_idx = traj.get("idx")
                if traj_idx is not None and traj_idx < len(batch_data_sources):
                    data_source = batch_data_sources[traj_idx]
                else:
                    data_source = "unknown"
                
                if data_source not in traj_metrics_by_source:
                    traj_metrics_by_source[data_source] = []
                
                traj_metrics = traj.get("metrics", {})
                if traj_metrics:
                    traj_metrics_by_source[data_source].append(traj_metrics)
            
            # Aggregate metrics per data_source
            for data_source, metrics_list in traj_metrics_by_source.items():
                if not metrics_list:
                    continue
                
                # Collect all metric keys for this data_source
                all_keys = set()
                for m in metrics_list:
                    all_keys.update(m.keys())
                
                # Aggregate each metric (mean, min, max) per data_source
                for k in all_keys:
                    v_list = [m.get(k) for m in metrics_list if k in m]
                    v_list = [v for v in v_list if v is not None and v >= 0]
                    if not v_list:
                        continue
                    v_list = np.array(v_list)
                    metrics.update(
                        {
                            f"traj/{data_source}/{k}_mean": v_list.mean(),
                            f"traj/{data_source}/{k}_min": v_list.min(),
                            f"traj/{data_source}/{k}_max": v_list.max(),
                        }
                    )

                # Compute episode_2 - episode_1 success rate gap per trajectory
                gaps = []
                for m in metrics_list:
                    ep1 = m.get("episode_1_success_rate")
                    ep2 = m.get("episode_2_success_rate")
                    if ep1 is not None and ep2 is not None and ep1 >= 0 and ep2 >= 0:
                        gaps.append(ep2 - ep1)
                if gaps:
                    metrics[f"traj/{data_source}/episode_improvement_mean"] = np.mean(gaps)

        return final_gen_batch_output, metrics

    def init_envs_and_agents(self, batch):
        """Override to track data_source and support validation env class.

        When _is_validation_mode is True and val_env_class is set, temporarily
        swaps the env_class and env_args to use the validation-specific ones.
        """
        # Store data_source before calling parent (in case batch gets modified)
        if hasattr(batch, "non_tensor_batch"):
            batch_data_sources = batch.non_tensor_batch.get("data_source")
            if batch_data_sources is not None:
                self._current_batch_data_sources = batch_data_sources

        # If in validation mode and val_env_class is set, swap env class temporarily
        if self._is_validation_mode and self.val_env_class is not None:
            original_env_class = self.env_class
            original_env_args = self.env_args
            self.env_class = self.val_env_class
            if self.val_env_args is not None:
                self.env_args = self.val_env_args
            try:
                return super().init_envs_and_agents(batch)
            finally:
                # Restore original env class and args
                self.env_class = original_env_class
                self.env_args = original_env_args
        else:
            return super().init_envs_and_agents(batch)

