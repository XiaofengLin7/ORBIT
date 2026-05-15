"""Train GEMTextAgent on single-episode GEM environment with custom metrics.

This script uses SingleEpisodeEnv wrapper with a custom execution engine that
extracts episode metrics. The single-episode logic is simpler than multi-episode:
the trajectory ends when the inner environment returns done=True.

Supports both single-task and multi-task training:
- Single-task: Uses prepare_gem_data() with env_id from config
- Multi-task: Uses prepare_multi_task_gem_data() with tasks config file
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.gem_text_agent import GEMTextAgent  # noqa: E402
from data.prepare_gem_data import (  # noqa: E402
    prepare_gem_data,
    prepare_multi_task_gem_data,
)
from envs.multi_episode_env import MultiEpisodeEnv  # noqa: E402
from envs.single_episode_env import SingleEpisodeEnv  # noqa: E402
from rllm.data import DatasetRegistry  # type: ignore  # noqa: E402
from trainers.train_multi_episode import run_ppo_agent  # noqa: E402


def _default_single_episode_prompt() -> str:
    """System prompt for single-episode tasks.

    Delegates to :func:`prompts.system_prompts.build_multi_episode_system_prompt`
    so single-episode training and eval share one prompt source. No
    summarization config is passed: token-mode summarization is not used in
    single-episode training, so the "Memory protocol" section is omitted.
    """
    from prompts.system_prompts import build_multi_episode_system_prompt

    return build_multi_episode_system_prompt(single_episode=True)


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer",
    version_base=None,
)
def main(cfg) -> None:  # type: ignore
    """Entry point for single-episode GEM training using environment wrapper.

    This uses the same MultiEpisodeAgentPPOTrainer infrastructure but with
    SingleEpisodeEnv which provides compatible metrics via get_metrics().

    Supports both single-task and multi-task training:
    - If tasks_config_path is provided in cfg.data, uses multi-task mode
    - Otherwise, falls back to single-task mode using env_id from config
    """
    # Extract environment configuration
    env_args = OmegaConf.to_container(cfg.rllm.env.env_args, resolve=True)  # type: ignore

    # Check for multi-task configuration
    tasks_config_path: Optional[str] = cfg.data.get("tasks_config_path", None)

    if tasks_config_path:
        # Multi-task mode: load tasks from config file
        tasks_config_path = str(Path(tasks_config_path).expanduser().resolve())
        train_dataset, val_dataset = prepare_multi_task_gem_data(
            tasks_config_path=tasks_config_path
        )

        # Calculate max steps needed for rllm.agent.max_steps
        # For training (single episode): use max_turns_per_episode
        # For validation (multi episode): use total_step_cap if available
        import yaml
        with open(tasks_config_path, "r") as f:
            config = yaml.safe_load(f)
        all_tasks = config.get("train_tasks", []) + config.get("val_tasks", [])

        # Get max of both max_turns_per_episode (for training) and total_step_cap (for validation)
        max_turns = max(
            (task.get("max_turns_per_episode", 30) for task in all_tasks),
            default=30
        )
        max_total_step_cap = max(
            (task.get("total_step_cap", task.get("max_turns_per_episode", 30)) for task in all_tasks),
            default=30
        )
        # Use the larger of the two to ensure both training and validation work
        required_max_steps = max(max_turns, max_total_step_cap)

        # Update max_steps if not explicitly set or if it's too small
        if not hasattr(cfg.rllm.agent, "max_steps") or cfg.rllm.agent.max_steps < required_max_steps:
            cfg.rllm.agent.max_steps = required_max_steps
    else:
        # Single-task mode: backward compatibility
        inner_env_kwargs = env_args.get("inner_env_kwargs", {})
        env_id = inner_env_kwargs.get("env_id", "game:GuessTheNumber-v0-hard")
        train_dataset, val_dataset = prepare_gem_data(env_id=env_id)

    # Set dataset paths in config
    if train_dataset is not None and hasattr(cfg, "data"):
        cfg.data.train_files = train_dataset.get_verl_data_path()
    if val_dataset is not None and hasattr(cfg, "data"):
        cfg.data.val_files = val_dataset.get_verl_data_path()

    # Configure agent with single-episode-aware system prompt
    agent_args = OmegaConf.to_container(
        cfg.rllm.agent.get("agent_args", {}), resolve=True
    )  # type: ignore
    agent_args = dict(agent_args or {})
    agent_args.setdefault("system_prompt", _default_single_episode_prompt())

    # Use SingleEpisodeEnv for training, MultiEpisodeEnv for validation
    # This allows training on single episodes but validating with multiple attempts
    run_ppo_agent(
        cfg,
        env_class=SingleEpisodeEnv,
        agent_class=GEMTextAgent,
        agent_args=agent_args,
        val_env_class=MultiEpisodeEnv,
    )


if __name__ == "__main__":
    main()
