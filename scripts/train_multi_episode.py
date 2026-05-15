"""Train GEMTextAgent on multi-episode GEM environment with custom metrics.

This script uses MultiEpisodeEnv wrapper with a custom execution engine that
extracts multi-episode metrics. The multi-episode logic is encapsulated in the
environment, allowing simpler integration with the standard training pipeline.

Supports both single-task and multi-task training:
- Single-task: Uses prepare_gem_data() with env_id from config
- Multi-task: Uses prepare_multi_task_gem_data() with tasks config file
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# Disable Ray's worker-stdout dedup so each trajectory's
# `Trajectory N completed …` / truncation print reaches the driver console.
# Ray's default (RAY_DEDUP_LOGS=1) numeric-masks log lines and collapses
# bursts of similar prints into a single line, hiding per-trajectory output.
os.environ.setdefault("RAY_DEDUP_LOGS", "0")

import hydra
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.gem_text_agent import GEMTextAgent  # noqa: E402
from agents.gem_text_agent_noncumulative import GEMTextAgentNonCumulative  # noqa: E402
from agents.context_summarizer import (  # noqa: E402
    GEMTextAgentWithSummarization,
    GEMTextAgentNonCumulativeWithSummarization,
)
from data.prepare_gem_data import (  # noqa: E402
    prepare_gem_data,
    prepare_multi_task_gem_data,
)
from envs.multi_episode_env import MultiEpisodeEnv  # noqa: E402
from prompts.system_prompts import build_multi_episode_system_prompt  # noqa: E402
from rllm.data import DatasetRegistry  # type: ignore  # noqa: E402
from trainers.train_multi_episode import run_ppo_agent  # noqa: E402

AGENT_CLASSES = {
    "math_agent": GEMTextAgent,
    "gem_text_agent": GEMTextAgent,
    "gem_text_agent_noncumulative": GEMTextAgentNonCumulative,
    "gem_text_agent_summarizing": GEMTextAgentWithSummarization,
    "gem_text_agent_noncumulative_summarizing": GEMTextAgentNonCumulativeWithSummarization,
}


"""System prompt comes from :func:`prompts.system_prompts.build_multi_episode_system_prompt`.

The builder takes the resolved summarization config so the prompt documents
the active carryover protocol (freeform / obs_action / obs_action_reflection
/ oracle / token). The literal ``{num_episodes}`` placeholder is substituted
by :class:`agents.gem_text_agent.GEMTextAgent` on the first ``update_from_env``
using ``info["num_episodes"]``. Callers who want a placeholder-free or fully
custom prompt should override ``rllm.agent.agent_args.system_prompt``.
"""


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer",
    version_base=None,
)
def main(cfg) -> None:  # type: ignore
    """Entry point for multi-episode GEM training using environment wrapper.

    This uses a custom MultiEpisodeAgentPPOTrainer that extracts multi-episode
    metrics from environments with get_metrics() method.

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
        
        # Calculate max total_step_cap across all tasks for rllm.agent.max_steps
        # This ensures the agent can handle the longest task
        import yaml
        with open(tasks_config_path, "r") as f:
            config = yaml.safe_load(f)
        all_tasks = config.get("train_tasks", []) + config.get("val_tasks", [])
        max_total_step_cap = max(
            (task.get("total_step_cap", 30) for task in all_tasks),
            default=30
        )
        # Update max_steps if not explicitly set or if it's too small
        if not hasattr(cfg.rllm.agent, "max_steps") or cfg.rllm.agent.max_steps < max_total_step_cap:
            cfg.rllm.agent.max_steps = max_total_step_cap
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

    # Configure agent with multi-episode-aware system prompt
    agent_args = OmegaConf.to_container(
        cfg.rllm.agent.get("agent_args", {}), resolve=True
    )  # type: ignore
    agent_args = dict(agent_args or {})

    # Resolve agent class from config (supports action-only non-cumulative agent)
    agent_name = cfg.rllm.agent.get("name", "math_agent")
    agent_class = AGENT_CLASSES.get(agent_name, GEMTextAgent)

    # Extract summarization config (if present) and merge into agent_args.
    # Guard against `.get("summarization", {})` returning a plain dict when the
    # key is absent — OmegaConf.to_container requires a DictConfig/ListConfig.
    summarization_cfg = cfg.rllm.agent.get("summarization")
    if summarization_cfg is not None:
        summarization_config = OmegaConf.to_container(summarization_cfg, resolve=True) or {}
    else:
        summarization_config = {}

    # Build the default system prompt AFTER reading the summarization config
    # so the prompt's "Memory protocol" section reflects the active carryover
    # mode. User overrides via rllm.agent.agent_args.system_prompt still win.
    agent_args.setdefault(
        "system_prompt",
        build_multi_episode_system_prompt(summarization=summarization_config),
    )
    if summarization_config.get("enable"):
        for key in (
            "summarization_threshold_tokens",
            "summary_max_tokens",
        ):
            # Map config keys → agent kwargs (config uses shorter names).
            cfg_key = key.replace("summarization_", "")
            if cfg_key in summarization_config:
                agent_args[key] = summarization_config[cfg_key]
            elif key in summarization_config:
                agent_args[key] = summarization_config[key]

        # Forward the summarization mode (token | episodic | both) to the agent.
        if "mode" in summarization_config:
            agent_args["mode"] = summarization_config["mode"]

        # Episodic context-carryover form: freeform | obs_action |
        # obs_action_reflection. Only affects the episode-end trigger.
        if "episodic_carryover" in summarization_config:
            agent_args["episodic_carryover"] = summarization_config[
                "episodic_carryover"
            ]

        # When episodic summarization is enabled, tell the env to defer
        # new-episode advancement to the engine's summarization hook (the
        # engine calls env.start_new_episode() after summarizing). Don't
        # override an explicit user setting.
        mode = summarization_config.get("mode", "episodic")
        if mode in ("episodic", "both"):
            env_args_node = cfg.rllm.env.get("env_args")
            current_env_args = (
                OmegaConf.to_container(env_args_node, resolve=True) or {}
                if env_args_node is not None
                else {}
            )
            if "reflection_via_summarization" not in current_env_args:
                OmegaConf.update(
                    cfg,
                    "rllm.env.env_args.reflection_via_summarization",
                    True,
                    force_add=True,
                )

    # Use our custom training function that uses MultiEpisodeAgentPPOTrainer
    run_ppo_agent(
        cfg,
        env_class=MultiEpisodeEnv,
        agent_class=agent_class,
        agent_args=agent_args,
        summarization_config=summarization_config if summarization_config.get("enable") else None,
    )


if __name__ == "__main__":
    main()

